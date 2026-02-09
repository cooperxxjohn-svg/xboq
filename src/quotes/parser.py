"""
Quote Parser - Parse subcontractor quotes from various formats.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import re


@dataclass
class QuoteLineItem:
    """Single line item in a quote."""
    item_no: str
    description: str
    unit: str
    quantity: float
    rate: float
    amount: float

    def to_dict(self) -> dict:
        return {
            "item_no": self.item_no,
            "description": self.description,
            "unit": self.unit,
            "quantity": self.quantity,
            "rate": self.rate,
            "amount": self.amount,
        }


@dataclass
class SubcontractorQuote:
    """Complete subcontractor quote."""
    subcontractor_name: str
    package: str
    quote_date: str = ""
    validity_days: int = 30
    line_items: List[QuoteLineItem] = field(default_factory=list)
    inclusions: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    total_amount: float = 0.0
    gst_percent: float = 18.0
    gst_included: bool = False
    payment_terms: str = ""
    completion_days: int = 0
    warranty_months: int = 12
    mobilization_advance: float = 0.0
    retention_percent: float = 5.0
    notes: List[str] = field(default_factory=list)
    leveled_total: float = 0.0  # Set after leveling

    def to_dict(self) -> dict:
        return {
            "subcontractor_name": self.subcontractor_name,
            "package": self.package,
            "quote_date": self.quote_date,
            "validity_days": self.validity_days,
            "line_items": [item.to_dict() for item in self.line_items],
            "inclusions": self.inclusions,
            "exclusions": self.exclusions,
            "total_amount": self.total_amount,
            "gst_percent": self.gst_percent,
            "gst_included": self.gst_included,
            "payment_terms": self.payment_terms,
            "completion_days": self.completion_days,
            "warranty_months": self.warranty_months,
            "mobilization_advance": self.mobilization_advance,
            "retention_percent": self.retention_percent,
            "notes": self.notes,
            "leveled_total": self.leveled_total,
        }


class QuoteParser:
    """Parse subcontractor quotes from various formats."""

    # Common exclusion patterns in Indian construction quotes
    EXCLUSION_PATTERNS = [
        r"scaffold", r"scaffolding",
        r"water.*supply", r"water.*storage",
        r"electricity", r"power.*supply",
        r"transport", r"transportation", r"freight",
        r"unloading", r"stacking",
        r"security", r"watchman",
        r"insurance", r"car\s*policy",
        r"safety.*equipment", r"ppe",
        r"testing", r"cube.*test",
        r"cleaning", r"debris.*removal",
        r"tools.*tackle", r"equipment",
        r"approval.*charges", r"noc",
        r"any.*civil.*work", r"civil.*support",
    ]

    # Common inclusion patterns
    INCLUSION_PATTERNS = [
        r"material.*supply", r"all.*material",
        r"labor", r"labour", r"workmanship",
        r"installation", r"fixing",
        r"consumables", r"fasteners",
        r"warranty", r"guarantee",
        r"supervision", r"site.*engineer",
    ]

    def __init__(self):
        pass

    def parse_dict(self, data: dict) -> Optional[SubcontractorQuote]:
        """Parse quote from dictionary input."""
        try:
            # Extract line items
            line_items = []
            for item_data in data.get("line_items", data.get("items", [])):
                line_items.append(QuoteLineItem(
                    item_no=str(item_data.get("item_no", item_data.get("sl_no", ""))),
                    description=item_data.get("description", item_data.get("desc", "")),
                    unit=item_data.get("unit", ""),
                    quantity=float(item_data.get("quantity", item_data.get("qty", 0)) or 0),
                    rate=float(item_data.get("rate", 0) or 0),
                    amount=float(item_data.get("amount", 0) or 0),
                ))

            # Calculate total if not provided
            total = float(data.get("total_amount", data.get("total", 0)) or 0)
            if total == 0 and line_items:
                total = sum(item.amount for item in line_items)

            # Parse inclusions/exclusions
            inclusions = data.get("inclusions", [])
            exclusions = data.get("exclusions", [])

            # Try to extract from notes if not explicitly provided
            notes = data.get("notes", data.get("remarks", []))
            if isinstance(notes, str):
                notes = [notes]

            if not inclusions or not exclusions:
                for note in notes:
                    inc, exc = self._extract_inclusions_exclusions(note)
                    inclusions.extend(inc)
                    exclusions.extend(exc)

            return SubcontractorQuote(
                subcontractor_name=data.get("subcontractor_name", data.get("vendor", data.get("name", "Unknown"))),
                package=data.get("package", data.get("work_package", "")),
                quote_date=data.get("quote_date", data.get("date", "")),
                validity_days=int(data.get("validity_days", data.get("validity", 30)) or 30),
                line_items=line_items,
                inclusions=inclusions,
                exclusions=exclusions,
                total_amount=total,
                gst_percent=float(data.get("gst_percent", data.get("gst", 18)) or 18),
                gst_included=data.get("gst_included", False),
                payment_terms=data.get("payment_terms", ""),
                completion_days=int(data.get("completion_days", data.get("duration", 0)) or 0),
                warranty_months=int(data.get("warranty_months", data.get("warranty", 12)) or 12),
                mobilization_advance=float(data.get("mobilization_advance", data.get("advance", 0)) or 0),
                retention_percent=float(data.get("retention_percent", data.get("retention", 5)) or 5),
                notes=notes,
            )

        except Exception as e:
            print(f"Error parsing quote dict: {e}")
            return None

    def parse_file(self, file_path: Path) -> Optional[SubcontractorQuote]:
        """Parse quote from file (Excel or PDF)."""
        suffix = file_path.suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            return self._parse_excel(file_path)
        elif suffix == ".pdf":
            return self._parse_pdf(file_path)
        elif suffix == ".csv":
            return self._parse_csv(file_path)
        else:
            print(f"Unsupported file format: {suffix}")
            return None

    def _parse_excel(self, file_path: Path) -> Optional[SubcontractorQuote]:
        """Parse quote from Excel file."""
        try:
            import pandas as pd

            df = pd.read_excel(file_path)

            # Try to identify columns
            col_mapping = self._identify_columns(df.columns.tolist())

            if not col_mapping:
                return None

            # Extract line items
            line_items = []
            for _, row in df.iterrows():
                # Skip empty rows
                desc = str(row.get(col_mapping.get("description", ""), "")).strip()
                if not desc or desc.lower() in ["", "nan", "total", "grand total", "sub total"]:
                    continue

                line_items.append(QuoteLineItem(
                    item_no=str(row.get(col_mapping.get("item_no", ""), "")),
                    description=desc,
                    unit=str(row.get(col_mapping.get("unit", ""), "")),
                    quantity=float(row.get(col_mapping.get("quantity", ""), 0) or 0),
                    rate=float(row.get(col_mapping.get("rate", ""), 0) or 0),
                    amount=float(row.get(col_mapping.get("amount", ""), 0) or 0),
                ))

            # Calculate total
            total = sum(item.amount for item in line_items)

            # Extract subcontractor name from filename
            subcontractor = file_path.stem.replace("_", " ").title()

            return SubcontractorQuote(
                subcontractor_name=subcontractor,
                package="",
                line_items=line_items,
                total_amount=total,
            )

        except Exception as e:
            print(f"Error parsing Excel file: {e}")
            return None

    def _parse_pdf(self, file_path: Path) -> Optional[SubcontractorQuote]:
        """Parse quote from PDF file."""
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                all_text = ""
                tables = []

                for page in pdf.pages:
                    all_text += page.extract_text() or ""
                    page_tables = page.extract_tables()
                    tables.extend(page_tables)

                # Try to extract from tables first
                if tables:
                    return self._parse_pdf_tables(tables, file_path.stem)

                # Fallback to text parsing
                return self._parse_pdf_text(all_text, file_path.stem)

        except Exception as e:
            print(f"Error parsing PDF file: {e}")
            return None

    def _parse_pdf_tables(self, tables: list, subcontractor: str) -> Optional[SubcontractorQuote]:
        """Parse quote from PDF tables."""
        line_items = []

        for table in tables:
            if not table or len(table) < 2:
                continue

            # First row is likely header
            header = [str(h).lower() if h else "" for h in table[0]]

            # Try to identify columns
            col_indices = {
                "item_no": self._find_column_index(header, ["sl", "sr", "item", "no", "#"]),
                "description": self._find_column_index(header, ["description", "desc", "particular", "work"]),
                "unit": self._find_column_index(header, ["unit", "uom"]),
                "quantity": self._find_column_index(header, ["qty", "quantity", "qnty"]),
                "rate": self._find_column_index(header, ["rate", "price", "unit rate"]),
                "amount": self._find_column_index(header, ["amount", "total", "value"]),
            }

            for row in table[1:]:
                if not row or len(row) < 3:
                    continue

                desc_idx = col_indices.get("description", 1)
                if desc_idx >= len(row):
                    continue

                desc = str(row[desc_idx] or "").strip()
                if not desc or desc.lower() in ["", "total", "grand total"]:
                    continue

                line_items.append(QuoteLineItem(
                    item_no=str(row[col_indices.get("item_no", 0)] or ""),
                    description=desc,
                    unit=str(row[col_indices.get("unit", 2)] or "") if col_indices.get("unit", 2) < len(row) else "",
                    quantity=self._safe_float(row[col_indices.get("quantity", 3)]) if col_indices.get("quantity", 3) < len(row) else 0,
                    rate=self._safe_float(row[col_indices.get("rate", 4)]) if col_indices.get("rate", 4) < len(row) else 0,
                    amount=self._safe_float(row[col_indices.get("amount", 5)]) if col_indices.get("amount", 5) < len(row) else 0,
                ))

        if line_items:
            total = sum(item.amount for item in line_items)
            return SubcontractorQuote(
                subcontractor_name=subcontractor.replace("_", " ").title(),
                package="",
                line_items=line_items,
                total_amount=total,
            )

        return None

    def _parse_pdf_text(self, text: str, subcontractor: str) -> Optional[SubcontractorQuote]:
        """Parse quote from PDF text content."""
        # Basic text parsing for quotes without structured tables
        # This is a fallback - tables are preferred

        inclusions = []
        exclusions = []

        # Extract inclusions/exclusions from text
        inc, exc = self._extract_inclusions_exclusions(text)
        inclusions.extend(inc)
        exclusions.extend(exc)

        # Try to find total amount
        total_patterns = [
            r"total.*?(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d{2})?)",
            r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d{2})?)\s*(?:total|grand)",
            r"grand\s*total.*?([\d,]+(?:\.\d{2})?)",
        ]

        total = 0
        for pattern in total_patterns:
            match = re.search(pattern, text.lower())
            if match:
                total = self._safe_float(match.group(1).replace(",", ""))
                break

        return SubcontractorQuote(
            subcontractor_name=subcontractor.replace("_", " ").title(),
            package="",
            inclusions=inclusions,
            exclusions=exclusions,
            total_amount=total,
            notes=[text[:500] + "..."] if len(text) > 500 else [text],
        )

    def _parse_csv(self, file_path: Path) -> Optional[SubcontractorQuote]:
        """Parse quote from CSV file."""
        try:
            import csv

            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return None

            # Try to identify columns
            col_mapping = self._identify_columns(list(rows[0].keys()))

            line_items = []
            for row in rows:
                desc = row.get(col_mapping.get("description", ""), "").strip()
                if not desc:
                    continue

                line_items.append(QuoteLineItem(
                    item_no=row.get(col_mapping.get("item_no", ""), ""),
                    description=desc,
                    unit=row.get(col_mapping.get("unit", ""), ""),
                    quantity=self._safe_float(row.get(col_mapping.get("quantity", ""), 0)),
                    rate=self._safe_float(row.get(col_mapping.get("rate", ""), 0)),
                    amount=self._safe_float(row.get(col_mapping.get("amount", ""), 0)),
                ))

            total = sum(item.amount for item in line_items)

            return SubcontractorQuote(
                subcontractor_name=file_path.stem.replace("_", " ").title(),
                package="",
                line_items=line_items,
                total_amount=total,
            )

        except Exception as e:
            print(f"Error parsing CSV file: {e}")
            return None

    def _identify_columns(self, columns: list) -> dict:
        """Identify column mappings from column names."""
        mapping = {}
        columns_lower = [str(c).lower() for c in columns]

        patterns = {
            "item_no": ["sl", "sr", "item no", "item_no", "sno", "s.no", "#"],
            "description": ["description", "desc", "particular", "particulars", "item", "work"],
            "unit": ["unit", "uom", "units"],
            "quantity": ["qty", "quantity", "qnty", "quan"],
            "rate": ["rate", "price", "unit rate", "unit_rate"],
            "amount": ["amount", "total", "value", "amt"],
        }

        for field, keywords in patterns.items():
            for i, col in enumerate(columns_lower):
                if any(kw in col for kw in keywords):
                    mapping[field] = columns[i]
                    break

        return mapping

    def _find_column_index(self, header: list, keywords: list) -> int:
        """Find column index by keywords."""
        for i, col in enumerate(header):
            if any(kw in col for kw in keywords):
                return i
        return -1

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""
        if value is None:
            return 0.0
        try:
            # Remove commas and currency symbols
            if isinstance(value, str):
                value = re.sub(r"[₹,\s]", "", value)
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _extract_inclusions_exclusions(self, text: str) -> tuple:
        """Extract inclusions and exclusions from text."""
        text_lower = text.lower()
        inclusions = []
        exclusions = []

        for pattern in self.EXCLUSION_PATTERNS:
            if re.search(pattern, text_lower):
                match = re.search(rf"({pattern}[^,.\n]*)", text_lower)
                if match:
                    exclusions.append(match.group(1).strip().title())

        for pattern in self.INCLUSION_PATTERNS:
            if re.search(pattern, text_lower):
                match = re.search(rf"({pattern}[^,.\n]*)", text_lower)
                if match:
                    inclusions.append(match.group(1).strip().title())

        return inclusions, exclusions
