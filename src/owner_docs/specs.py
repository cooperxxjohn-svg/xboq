"""
Specifications Parser - Extract technical specifications from owner docs.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .parser import ParsedDocument, DocType


@dataclass
class SpecRequirement:
    """Single specification requirement."""
    item: str
    requirement: str
    category: str = ""
    is_code: Optional[str] = None
    brand_make: Optional[str] = None
    grade: Optional[str] = None
    source_file: str = ""
    source_page: int = 0

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "requirement": self.requirement,
            "category": self.category,
            "is_code": self.is_code,
            "brand_make": self.brand_make,
            "grade": self.grade,
            "source_file": self.source_file,
        }


class SpecsParser:
    """Parse technical specifications."""

    # Categories and their patterns
    SPEC_CATEGORIES = {
        "concrete": [
            r'concrete.*grade.*m\s*(\d+)',
            r'm\s*(\d+).*concrete',
            r'cement.*(?:opc|ppc|psc)',
            r'aggregate.*(\d+)\s*mm',
            r'admixture',
        ],
        "steel": [
            r'reinforcement.*fe\s*(\d+)',
            r'tmt.*bar',
            r'steel.*grade',
            r'binding\s+wire',
        ],
        "brick_block": [
            r'brick.*class',
            r'aac\s+block',
            r'fly\s*ash\s+brick',
            r'cement\s+block',
        ],
        "plaster": [
            r'plaster.*(\d+)\s*mm',
            r'cement.*sand.*mortar',
            r'cm\s+(\d+:\d+)',
        ],
        "flooring": [
            r'vitrified.*tile',
            r'ceramic.*tile',
            r'granite.*(\d+)\s*mm',
            r'marble',
            r'kota\s+stone',
        ],
        "waterproofing": [
            r'waterproof',
            r'app\s+membrane',
            r'integral\s+compound',
            r'cementitious\s+coating',
        ],
        "paint": [
            r'acrylic\s+emulsion',
            r'exterior\s+paint',
            r'primer',
            r'putty',
            r'enamel',
        ],
        "doors": [
            r'door.*frame',
            r'flush\s+door',
            r'teak\s+wood',
            r'sal\s+wood',
            r'wpc\s+door',
        ],
        "windows": [
            r'alumin.*window',
            r'upvc.*window',
            r'glass.*(\d+)\s*mm',
            r'powder\s+coat',
        ],
        "plumbing": [
            r'cpvc.*pipe',
            r'upvc.*swr',
            r'gi\s+pipe',
            r'hdpe\s+pipe',
            r'cp\s+fitting',
        ],
        "sanitary": [
            r'ewc',
            r'wash\s+basin',
            r'urinal',
            r'shower',
            r'faucet',
        ],
        "electrical": [
            r'wire.*sq\.?\s*mm',
            r'mcb',
            r'switch.*modular',
            r'conduit',
            r'earthing',
        ],
    }

    # Brand/make patterns
    BRAND_PATTERNS = [
        r'(?:make|brand|manufacture)r?\s*[:=]?\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)',
        r'(?:or\s+)?equivalent\s+(?:to|of)\s+([A-Za-z]+)',
        r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:make|brand)',
    ]

    # IS code patterns
    IS_CODE_PATTERN = r'is\s*[-:]?\s*(\d{3,5})'

    def __init__(self):
        self.is_code_regex = re.compile(self.IS_CODE_PATTERN, re.IGNORECASE)
        self.brand_regex = [re.compile(p, re.IGNORECASE) for p in self.BRAND_PATTERNS]

    def parse(self, documents: List[ParsedDocument]) -> List[SpecRequirement]:
        """Parse specifications from documents."""
        specs = []

        # Find specification documents
        spec_docs = [d for d in documents if d.doc_type == DocType.SPECIFICATIONS]

        # Also check other docs for embedded specs
        spec_docs.extend([d for d in documents if d.doc_type in [
            DocType.CONTRACT_CONDITIONS, DocType.BID_FORM
        ]])

        for doc in spec_docs:
            doc_specs = self._parse_document(doc)
            specs.extend(doc_specs)

        # Deduplicate
        seen = set()
        unique_specs = []
        for spec in specs:
            key = f"{spec.item}_{spec.requirement[:30]}"
            if key not in seen:
                seen.add(key)
                unique_specs.append(spec)

        return unique_specs

    def _parse_document(self, doc: ParsedDocument) -> List[SpecRequirement]:
        """Parse specs from a single document."""
        specs = []
        current_category = ""

        lines = doc.text_content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for category header
            new_cat = self._detect_category(line)
            if new_cat:
                current_category = new_cat

            # Look for specification patterns
            spec = self._extract_spec(line, current_category, doc)
            if spec:
                specs.append(spec)

            # Also check for grade/make specifications
            make_spec = self._extract_make_spec(line, doc)
            if make_spec:
                specs.append(make_spec)

        return specs

    def _detect_category(self, line: str) -> Optional[str]:
        """Detect specification category from line."""
        line_lower = line.lower()

        # Check if it's a header (all caps, short, etc.)
        if len(line) < 60 and (line.isupper() or line.endswith(':')):
            for category, patterns in self.SPEC_CATEGORIES.items():
                for pattern in patterns:
                    if re.search(pattern, line_lower):
                        return category

        return None

    def _extract_spec(
        self, line: str, category: str, doc: ParsedDocument
    ) -> Optional[SpecRequirement]:
        """Extract specification from line."""
        line_lower = line.lower()

        # Must contain some specification indicator
        spec_indicators = [
            'shall be', 'should be', 'must be', 'to be',
            'as per', 'conforming to', 'complying with',
            'grade', 'class', 'type', 'size',
            'thickness', 'strength', 'capacity',
        ]

        has_indicator = any(ind in line_lower for ind in spec_indicators)
        if not has_indicator and not category:
            return None

        # Extract IS code if present
        is_code = None
        code_match = self.is_code_regex.search(line)
        if code_match:
            is_code = f"IS {code_match.group(1)}"

        # Extract brand/make if present
        brand = None
        for pattern in self.brand_regex:
            match = pattern.search(line)
            if match:
                brand = match.group(1)
                break

        # Clean up description
        item = self._extract_item_name(line)
        requirement = line

        if not item:
            return None

        return SpecRequirement(
            item=item,
            requirement=requirement,
            category=category,
            is_code=is_code,
            brand_make=brand,
            source_file=doc.file_name,
        )

    def _extract_make_spec(
        self, line: str, doc: ParsedDocument
    ) -> Optional[SpecRequirement]:
        """Extract make/brand specification."""
        line_lower = line.lower()

        # Look for make specifications
        make_patterns = [
            (r'([a-z\s]+?)(?:\s+make|\s+brand)\s*[:=]?\s*(.+)', 'make'),
            (r'approved\s+make[s]?\s*[:=]?\s*(.+)', 'approved_makes'),
            (r'([a-z\s]+?)\s+shall\s+be\s+(?:of\s+)?([a-z]+)\s+(?:make|brand)', 'make'),
        ]

        for pattern, ptype in make_patterns:
            match = re.search(pattern, line_lower)
            if match:
                if ptype == 'approved_makes':
                    item = "Approved Makes"
                    makes = match.group(1)
                else:
                    item = match.group(1).strip().title()
                    makes = match.group(2) if len(match.groups()) > 1 else match.group(1)

                return SpecRequirement(
                    item=item,
                    requirement=line,
                    category="makes",
                    brand_make=makes.strip(),
                    source_file=doc.file_name,
                )

        return None

    def _extract_item_name(self, line: str) -> str:
        """Extract item name from specification line."""
        # Take first part before "shall", "should", "to be", etc.
        for splitter in ['shall', 'should', 'must', 'to be', 'as per']:
            if splitter in line.lower():
                idx = line.lower().index(splitter)
                return line[:idx].strip()[:50]

        # Take first 50 chars
        return line[:50].strip()

    def get_specs_by_category(
        self, specs: List[SpecRequirement]
    ) -> Dict[str, List[SpecRequirement]]:
        """Group specifications by category."""
        grouped = {}
        for spec in specs:
            cat = spec.category or "general"
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(spec)
        return grouped

    def get_required_makes(
        self, specs: List[SpecRequirement]
    ) -> Dict[str, List[str]]:
        """Extract required makes from specifications."""
        makes = {}
        for spec in specs:
            if spec.brand_make:
                item = spec.item.lower()
                if item not in makes:
                    makes[item] = []
                makes[item].append(spec.brand_make)
        return makes
