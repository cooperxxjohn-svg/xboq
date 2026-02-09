"""
Revision Table Detector - Extract revision information from drawing sheets.

Detects and parses:
- IS 11669 standard revision tables
- Common consultant formats (tabular and inline)
- Revision clouds and markers
- ASI/RFI notations

India-specific patterns:
- Date formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YY
- Revision notations: R0, R1, Rev A, Rev.1, P1, etc.
- Description patterns: ASI, SI, Client comment, Structural revision
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RevisionType(Enum):
    """Type of revision."""
    DESIGN = "design"
    CLIENT = "client"
    ASI = "asi"  # Architect's Supplementary Instruction
    STRUCTURAL = "structural"
    MEP = "mep"
    COORDINATION = "coordination"
    APPROVAL = "approval"
    UNKNOWN = "unknown"


@dataclass
class RevisionEntry:
    """Single revision entry."""
    revision: str  # R1, Rev A, etc.
    date: Optional[str] = None
    description: str = ""
    revision_type: RevisionType = RevisionType.UNKNOWN
    prepared_by: Optional[str] = None
    approved_by: Optional[str] = None
    cloud_ref: Optional[str] = None  # Reference to revision cloud
    source_page: str = ""
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "revision": self.revision,
            "date": self.date,
            "description": self.description,
            "revision_type": self.revision_type.value,
            "prepared_by": self.prepared_by,
            "approved_by": self.approved_by,
            "cloud_ref": self.cloud_ref,
            "source_page": self.source_page,
            "confidence": self.confidence,
        }


@dataclass
class RevisionTable:
    """Revision table extracted from a sheet."""
    sheet_id: str
    entries: List[RevisionEntry] = field(default_factory=list)
    table_format: str = "unknown"  # is11669, tabular, inline
    latest_revision: Optional[str] = None
    latest_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sheet_id": self.sheet_id,
            "entries": [e.to_dict() for e in self.entries],
            "table_format": self.table_format,
            "latest_revision": self.latest_revision,
            "latest_date": self.latest_date,
        }


class RevisionDetector:
    """Detect revision tables and entries from extraction results."""

    # Revision number patterns
    REV_PATTERNS = [
        r'R(\d+)',           # R0, R1, R2
        r'Rev\.?\s*(\d+)',   # Rev 1, Rev.2
        r'Rev\.?\s*([A-Z])', # Rev A, Rev.B
        r'P(\d+)',           # P1, P2 (preliminary)
        r'(\d+)(?=\s*\|)',   # Just number before pipe
    ]

    # Date patterns (India-specific)
    DATE_PATTERNS = [
        r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})',  # DD/MM/YYYY
        r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{2,4})',
    ]

    # Revision type keywords
    TYPE_KEYWORDS = {
        RevisionType.ASI: ["asi", "a.s.i", "architect supplementary", "supplementary instruction"],
        RevisionType.CLIENT: ["client", "owner", "developer", "builder comment"],
        RevisionType.STRUCTURAL: ["structural", "rcc", "column", "beam", "foundation", "slab"],
        RevisionType.MEP: ["mep", "plumbing", "electrical", "hvac", "fire", "services"],
        RevisionType.COORDINATION: ["coordination", "clash", "interface"],
        RevisionType.APPROVAL: ["approval", "approved", "for construction", "ifc", "gfc"],
        RevisionType.DESIGN: ["design", "layout", "planning", "architectural"],
    }

    # Table header patterns
    TABLE_HEADERS = [
        r'rev(?:ision)?',
        r'date',
        r'description',
        r'prepared\s*by',
        r'approved\s*by',
        r'remarks?',
        r'no\.?',
    ]

    def __init__(self):
        self.rev_regex = [re.compile(p, re.IGNORECASE) for p in self.REV_PATTERNS]
        self.date_regex = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self.header_regex = re.compile(
            '|'.join(self.TABLE_HEADERS), re.IGNORECASE
        )

    def detect_all(self, extraction_results: List[Dict]) -> List[RevisionTable]:
        """Detect revision tables from all extraction results."""
        tables = []

        for result in extraction_results:
            table = self.detect_from_result(result)
            if table and table.entries:
                tables.append(table)

        return tables

    def detect_from_result(self, result: Dict) -> Optional[RevisionTable]:
        """Detect revision table from single extraction result."""
        from pathlib import Path

        file_name = Path(result.get("file_path", "")).stem
        page_num = result.get("page_number", 0) + 1
        sheet_id = f"{file_name}_p{page_num}"

        entries = []

        # 1. Check title block for revision info
        title_block = result.get("title_block", {})
        if title_block:
            entry = self._extract_from_title_block(title_block, sheet_id)
            if entry:
                entries.append(entry)

        # 2. Check revision table
        rev_table = result.get("revision_table", [])
        if rev_table:
            entries.extend(self._parse_revision_table(rev_table, sheet_id))

        # 3. Check text items for revision info
        text_items = result.get("text_items", [])
        entries.extend(self._extract_from_text(text_items, sheet_id))

        # 4. Check notes for revision references
        notes = result.get("notes", [])
        entries.extend(self._extract_from_notes(notes, sheet_id))

        # Deduplicate entries
        entries = self._deduplicate_entries(entries)

        if not entries:
            return None

        # Sort by revision number
        entries = sorted(entries, key=lambda e: self._revision_sort_key(e.revision))

        return RevisionTable(
            sheet_id=sheet_id,
            entries=entries,
            table_format=self._detect_format(result),
            latest_revision=entries[-1].revision if entries else None,
            latest_date=entries[-1].date if entries else None,
        )

    def _extract_from_title_block(
        self, title_block: Dict, sheet_id: str
    ) -> Optional[RevisionEntry]:
        """Extract revision from title block."""
        # Common title block fields
        revision = title_block.get("revision") or title_block.get("rev")
        date = title_block.get("date") or title_block.get("revision_date")

        if not revision:
            # Try to find in other fields
            for key, value in title_block.items():
                if isinstance(value, str):
                    match = self._find_revision(value)
                    if match:
                        revision = match
                        break

        if revision:
            return RevisionEntry(
                revision=self._normalize_revision(revision),
                date=self._normalize_date(date) if date else None,
                description="From title block",
                source_page=sheet_id,
                confidence=0.9,
            )

        return None

    def _parse_revision_table(
        self, rev_table: List[Dict], sheet_id: str
    ) -> List[RevisionEntry]:
        """Parse structured revision table."""
        entries = []

        for row in rev_table:
            revision = row.get("revision") or row.get("rev") or row.get("no")
            if not revision:
                continue

            date = row.get("date")
            description = row.get("description") or row.get("remarks") or ""
            prepared_by = row.get("prepared_by") or row.get("by")
            approved_by = row.get("approved_by") or row.get("approved")

            # Detect revision type
            rev_type = self._detect_revision_type(description)

            entries.append(RevisionEntry(
                revision=self._normalize_revision(str(revision)),
                date=self._normalize_date(date) if date else None,
                description=description,
                revision_type=rev_type,
                prepared_by=prepared_by,
                approved_by=approved_by,
                source_page=sheet_id,
                confidence=0.95,
            ))

        return entries

    def _extract_from_text(
        self, text_items: List[Dict], sheet_id: str
    ) -> List[RevisionEntry]:
        """Extract revision info from text annotations."""
        entries = []

        for item in text_items:
            text = item.get("text", "")
            if not text or len(text) > 200:  # Skip long text
                continue

            # Check for revision table header
            if self.header_regex.search(text):
                # This might be part of a table, try to parse
                pass

            # Check for inline revision notation
            # Pattern: "R1 - 15/01/2024 - ASI comment incorporated"
            inline_match = re.match(
                r'(R\d+|Rev\.?\s*\d+|Rev\.?\s*[A-Z])\s*[-–|]\s*([^-–|]+)(?:[-–|]\s*(.+))?',
                text,
                re.IGNORECASE
            )

            if inline_match:
                revision = inline_match.group(1)
                date_or_desc = inline_match.group(2).strip()
                extra = inline_match.group(3).strip() if inline_match.group(3) else ""

                # Determine if second part is date or description
                date_match = None
                for pattern in self.date_regex:
                    date_match = pattern.search(date_or_desc)
                    if date_match:
                        break

                if date_match:
                    date = date_or_desc
                    description = extra
                else:
                    date = None
                    description = f"{date_or_desc} {extra}".strip()

                entries.append(RevisionEntry(
                    revision=self._normalize_revision(revision),
                    date=self._normalize_date(date) if date else None,
                    description=description,
                    revision_type=self._detect_revision_type(description),
                    source_page=sheet_id,
                    confidence=0.75,
                ))

        return entries

    def _extract_from_notes(
        self, notes: List[Dict], sheet_id: str
    ) -> List[RevisionEntry]:
        """Extract revision references from notes."""
        entries = []

        for note in notes:
            text = note.get("text", "") if isinstance(note, dict) else str(note)

            # Look for revision references
            for pattern in self.rev_regex:
                matches = pattern.finditer(text)
                for match in matches:
                    # Check context
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]

                    # Skip if just a cross-reference
                    if re.search(r'see\s+rev|refer\s+rev', context, re.IGNORECASE):
                        continue

                    entries.append(RevisionEntry(
                        revision=self._normalize_revision(match.group(0)),
                        description=f"Referenced in note: {context[:50]}...",
                        source_page=sheet_id,
                        confidence=0.6,
                    ))
                    break  # Only first match per note

        return entries

    def _find_revision(self, text: str) -> Optional[str]:
        """Find revision number in text."""
        for pattern in self.rev_regex:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _normalize_revision(self, revision: str) -> str:
        """Normalize revision notation."""
        revision = revision.strip().upper()

        # Handle various formats
        if re.match(r'^\d+$', revision):
            return f"R{revision}"
        elif re.match(r'^R\d+$', revision, re.IGNORECASE):
            return revision.upper()
        elif re.match(r'^REV\.?\s*(\d+)$', revision, re.IGNORECASE):
            num = re.search(r'\d+', revision).group()
            return f"R{num}"
        elif re.match(r'^REV\.?\s*([A-Z])$', revision, re.IGNORECASE):
            letter = re.search(r'[A-Z]', revision, re.IGNORECASE).group().upper()
            return f"REV.{letter}"
        elif re.match(r'^P\d+$', revision, re.IGNORECASE):
            return revision.upper()

        return revision

    def _normalize_date(self, date: str) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format."""
        if not date:
            return None

        date = str(date).strip()

        # Try DD/MM/YYYY pattern
        match = re.match(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})', date)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = f"20{year}" if int(year) < 50 else f"19{year}"
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Try text month patterns
        months = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }

        for month_name, month_num in months.items():
            if month_name in date.lower():
                # Extract day and year
                nums = re.findall(r'\d+', date)
                if len(nums) >= 2:
                    day = nums[0] if int(nums[0]) <= 31 else nums[1]
                    year = nums[-1]
                    if len(year) == 2:
                        year = f"20{year}" if int(year) < 50 else f"19{year}"
                    return f"{year}-{month_num}-{str(day).zfill(2)}"

        return date  # Return as-is if can't parse

    def _detect_revision_type(self, description: str) -> RevisionType:
        """Detect revision type from description."""
        desc_lower = description.lower()

        for rev_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return rev_type

        return RevisionType.UNKNOWN

    def _detect_format(self, result: Dict) -> str:
        """Detect revision table format."""
        if result.get("revision_table"):
            return "tabular"

        # Check title block for IS 11669 format
        title_block = result.get("title_block", {})
        if title_block.get("revision") and title_block.get("revision_date"):
            return "is11669"

        return "inline"

    def _revision_sort_key(self, revision: str) -> Tuple[int, str]:
        """Generate sort key for revision."""
        # Extract number or letter
        num_match = re.search(r'\d+', revision)
        letter_match = re.search(r'[A-Z]$', revision, re.IGNORECASE)

        if num_match:
            return (0, num_match.group().zfill(10))
        elif letter_match:
            return (1, letter_match.group().upper())
        else:
            return (2, revision)

    def _deduplicate_entries(
        self, entries: List[RevisionEntry]
    ) -> List[RevisionEntry]:
        """Remove duplicate revision entries."""
        seen = {}
        unique = []

        for entry in entries:
            key = entry.revision
            if key not in seen or entry.confidence > seen[key].confidence:
                seen[key] = entry

        return list(seen.values())
