"""
Owner BOQ Parser - Parse owner-provided Bill of Quantities.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from .parser import ParsedDocument, DocType


@dataclass
class OwnerBOQItem:
    """Single item from owner BOQ."""
    item_no: str
    description: str
    unit: str
    quantity: float
    rate: Optional[float] = None
    amount: Optional[float] = None
    package: str = ""
    sub_package: str = ""
    notes: str = ""
    source_file: str = ""
    source_page: int = 0
    is_provisional: bool = False
    is_alternate: bool = False

    def to_dict(self) -> dict:
        return {
            "item_no": self.item_no,
            "description": self.description,
            "unit": self.unit,
            "quantity": self.quantity,
            "rate": self.rate,
            "amount": self.amount,
            "package": self.package,
            "sub_package": self.sub_package,
            "notes": self.notes,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "is_provisional": self.is_provisional,
            "is_alternate": self.is_alternate,
        }


class OwnerBOQParser:
    """Parse owner BOQ documents."""

    # Common units (India-specific)
    UNITS = [
        'sqm', 'sq.m', 'sq m', 'sqft', 'sq.ft', 'sq ft',
        'cum', 'cu.m', 'cu m', 'cft', 'cu.ft',
        'rmt', 'rm', 'r.m', 'running metre', 'running meter',
        'nos', 'no', 'no.', 'numbers', 'each', 'set', 'pair',
        'kg', 'kgs', 'mt', 'quintal', 'tonne',
        'ls', 'l.s', 'lump sum', 'lumpsum', 'job',
        'litre', 'liter', 'lit', 'kl',
        'bag', 'bags',
        'sqm/coat', 'per coat',
        'point', 'points',
    ]

    # Package detection patterns
    PACKAGE_PATTERNS = {
        "civil_structural": [
            r'excavation', r'earthwork', r'foundation', r'footing',
            r'rcc', r'reinforced', r'concrete', r'shuttering', r'formwork',
            r'steel', r'reinforcement', r'plinth', r'beam', r'column', r'slab',
        ],
        "masonry": [
            r'brick', r'block', r'aac', r'masonry', r'wall',
        ],
        "plaster_finishes": [
            r'plaster', r'rendering', r'putty', r'paint', r'finish',
        ],
        "flooring": [
            r'floor', r'tile', r'vitrified', r'ceramic', r'marble', r'granite',
            r'stone', r'kota', r'shahabad',
        ],
        "waterproofing": [
            r'waterproof', r'damp proof', r'membrane', r'bitumen',
        ],
        "doors_windows": [
            r'door', r'window', r'shutter', r'frame', r'glazing',
        ],
        "plumbing": [
            r'plumb', r'sanitary', r'water supply', r'drainage', r'swr',
            r'cpvc', r'upvc', r'pipe', r'fitting', r'fixture',
        ],
        "electrical": [
            r'electric', r'wiring', r'switch', r'socket', r'light',
            r'db', r'panel', r'mcb', r'earthing',
        ],
        "hvac": [
            r'hvac', r'air condition', r'ac', r'ventilation', r'duct',
        ],
        "fire": [
            r'fire', r'sprinkler', r'hydrant', r'alarm',
        ],
        "external_works": [
            r'external', r'paving', r'compound', r'gate', r'landscape',
            r'drain', r'road', r'parking',
        ],
        "prelims": [
            r'prelim', r'general', r'site', r'mobilization', r'scaffolding',
            r'safety', r'insurance', r'testing',
        ],
    }

    def __init__(self):
        self.unit_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(u) for u in self.UNITS) + r')\b',
            re.IGNORECASE
        )

    def parse(self, documents: List[ParsedDocument]) -> List[OwnerBOQItem]:
        """Parse owner BOQ from documents."""
        items = []

        # Find BOQ documents
        boq_docs = [d for d in documents if d.doc_type == DocType.OWNER_BOQ]

        # Also check bid forms which may contain BOQ
        boq_docs.extend([d for d in documents if d.doc_type == DocType.BID_FORM])

        for doc in boq_docs:
            doc_items = self._parse_document(doc)
            items.extend(doc_items)

        # Deduplicate by item_no
        seen = set()
        unique_items = []
        for item in items:
            key = f"{item.item_no}_{item.description[:50]}"
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def _parse_document(self, doc: ParsedDocument) -> List[OwnerBOQItem]:
        """Parse BOQ items from a single document."""
        items = []

        # Try table-based parsing first
        if doc.tables:
            for table in doc.tables:
                if table.get("type") == "boq":
                    items.extend(self._parse_table_rows(table["rows"], doc))

        # Also try line-by-line parsing
        items.extend(self._parse_text_lines(doc))

        return items

    def _parse_table_rows(
        self, rows: List[str], doc: ParsedDocument
    ) -> List[OwnerBOQItem]:
        """Parse BOQ table rows."""
        items = []
        current_package = ""

        for i, row in enumerate(rows):
            if i == 0:  # Skip header
                continue

            # Check if this is a section header
            if self._is_section_header(row):
                current_package = self._extract_package(row)
                continue

            # Try to parse as BOQ item
            item = self._parse_row(row, doc, current_package)
            if item:
                items.append(item)

        return items

    def _parse_text_lines(self, doc: ParsedDocument) -> List[OwnerBOQItem]:
        """Parse BOQ items from text lines."""
        items = []
        current_package = ""
        current_description = ""
        current_item_no = ""

        lines = doc.text_content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for section header
            if self._is_section_header(line):
                current_package = self._extract_package(line)
                continue

            # Check for item number at start
            item_match = re.match(r'^(\d+(?:\.\d+)?(?:\.[a-z])?)\s+(.+)', line, re.IGNORECASE)
            if item_match:
                current_item_no = item_match.group(1)
                rest = item_match.group(2)

                # Look for quantity and unit
                qty_match = self._find_quantity_unit(rest)
                if qty_match:
                    description = rest[:qty_match.start()].strip()
                    quantity = self._parse_quantity(qty_match.group(1))
                    unit = qty_match.group(2)

                    # Look for rate in remaining text or next line
                    rate = self._find_rate(rest[qty_match.end():])
                    if not rate and i + 1 < len(lines):
                        rate = self._find_rate(lines[i + 1])

                    items.append(OwnerBOQItem(
                        item_no=current_item_no,
                        description=description,
                        unit=self._normalize_unit(unit),
                        quantity=quantity,
                        rate=rate,
                        amount=quantity * rate if rate else None,
                        package=current_package,
                        source_file=doc.file_name,
                        is_provisional="provisional" in line.lower(),
                        is_alternate="alternate" in line.lower() or "option" in line.lower(),
                    ))

        return items

    def _parse_row(
        self, row: str, doc: ParsedDocument, package: str
    ) -> Optional[OwnerBOQItem]:
        """Parse a single BOQ row."""
        # Try to split by tabs or multiple spaces
        parts = re.split(r'\t+|\s{2,}', row)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 3:
            return None

        # Expected format: Item No | Description | Unit | Qty | Rate | Amount
        item_no = ""
        description = ""
        unit = ""
        quantity = 0.0
        rate = None
        amount = None

        # Find item number (first numeric)
        for i, part in enumerate(parts):
            if re.match(r'^\d+(?:\.\d+)?', part):
                item_no = part
                parts = parts[i+1:]
                break

        if not item_no:
            return None

        # Find unit
        for i, part in enumerate(parts):
            if self.unit_pattern.search(part):
                unit = part
                # Everything before is description
                description = " ".join(parts[:i])
                parts = parts[i+1:]
                break

        if not unit:
            # Unit might be embedded in description
            for i, part in enumerate(parts):
                match = self.unit_pattern.search(part)
                if match:
                    unit = match.group(1)
                    description = " ".join(parts[:i]) + " " + part[:match.start()]
                    parts = parts[i+1:]
                    break

        if not unit:
            return None

        # Remaining parts should be quantity, rate, amount
        for part in parts:
            num = self._parse_quantity(part)
            if num > 0:
                if quantity == 0:
                    quantity = num
                elif rate is None:
                    rate = num
                elif amount is None:
                    amount = num

        if quantity == 0:
            return None

        return OwnerBOQItem(
            item_no=item_no,
            description=description.strip(),
            unit=self._normalize_unit(unit),
            quantity=quantity,
            rate=rate,
            amount=amount,
            package=package,
            source_file=doc.file_name,
            is_provisional="provisional" in row.lower(),
            is_alternate="alternate" in row.lower(),
        )

    def _is_section_header(self, line: str) -> bool:
        """Check if line is a section header."""
        line_lower = line.lower()

        # All caps short line
        if line.isupper() and len(line) < 50 and not re.search(r'\d{3,}', line):
            return True

        # Contains package keywords
        for package, patterns in self.PACKAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    # But not if it has quantity indicators
                    if not self.unit_pattern.search(line):
                        return True

        return False

    def _extract_package(self, line: str) -> str:
        """Extract package name from section header."""
        line_lower = line.lower()

        for package, patterns in self.PACKAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    return package

        # Clean up the line as package name
        line = re.sub(r'^[\d.)\s]+', '', line)
        line = re.sub(r'[:\-–].*$', '', line)
        return line.strip()[:50]

    def _find_quantity_unit(self, text: str):
        """Find quantity and unit in text."""
        pattern = r'([\d,]+(?:\.\d+)?)\s*(' + '|'.join(re.escape(u) for u in self.UNITS) + r')'
        return re.search(pattern, text, re.IGNORECASE)

    def _find_rate(self, text: str) -> Optional[float]:
        """Find rate value in text."""
        # Look for rate-like patterns
        patterns = [
            r'rate\s*[:=]?\s*([\d,]+(?:\.\d+)?)',
            r'@\s*([\d,]+(?:\.\d+)?)',
            r'(?:rs\.?|₹)\s*([\d,]+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._parse_quantity(match.group(1))

        return None

    def _parse_quantity(self, text: str) -> float:
        """Parse quantity from text."""
        if not text:
            return 0.0

        # Remove commas and extract number
        text = text.replace(",", "")
        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return 0.0

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit to standard form."""
        unit_lower = unit.lower().strip()

        normalizations = {
            "sq.m": "sqm", "sq m": "sqm", "square metre": "sqm", "square meter": "sqm",
            "sq.ft": "sqft", "sq ft": "sqft", "square feet": "sqft",
            "cu.m": "cum", "cu m": "cum", "cubic metre": "cum", "cubic meter": "cum",
            "cu.ft": "cft", "cu ft": "cft", "cubic feet": "cft",
            "r.m": "rmt", "rm": "rmt", "running metre": "rmt", "running meter": "rmt",
            "no": "nos", "no.": "nos", "numbers": "nos", "each": "nos",
            "kgs": "kg", "kilogram": "kg",
            "l.s": "LS", "lump sum": "LS", "lumpsum": "LS",
            "litre": "liter", "lit": "liter",
        }

        return normalizations.get(unit_lower, unit_lower)
