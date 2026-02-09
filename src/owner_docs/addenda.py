"""
Addenda Parser - Parse tender addenda/corrigenda.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from .parser import ParsedDocument, DocType


@dataclass
class Addendum:
    """Single addendum/corrigendum."""
    addendum_no: str
    date: Optional[str] = None
    title: str = ""
    changes: List[Dict] = field(default_factory=list)
    clarifications: List[str] = field(default_factory=list)
    boq_changes: List[Dict] = field(default_factory=list)
    date_changes: List[Dict] = field(default_factory=list)
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "addendum_no": self.addendum_no,
            "date": self.date,
            "title": self.title,
            "changes": self.changes,
            "clarifications": self.clarifications,
            "boq_changes": self.boq_changes,
            "date_changes": self.date_changes,
            "source_file": self.source_file,
        }


class AddendaParser:
    """Parse addenda and corrigenda."""

    # Addendum number patterns
    ADDENDUM_NO_PATTERNS = [
        r'addendum\s*(?:no\.?|number)?\s*[:=]?\s*(\d+)',
        r'corrigendum\s*(?:no\.?|number)?\s*[:=]?\s*(\d+)',
        r'amendment\s*(?:no\.?|number)?\s*[:=]?\s*(\d+)',
        r'clarification\s*(?:no\.?|number)?\s*[:=]?\s*(\d+)',
    ]

    # Change type patterns
    CHANGE_PATTERNS = {
        "read_as": r'(?:read|revised|changed)\s+(?:as|to)\s*[:=]?\s*(.+)',
        "instead_of": r'instead\s+of\s*[:=]?\s*(.+)',
        "delete": r'(?:delete|remove|omit)\s*[:=]?\s*(.+)',
        "add": r'(?:add|insert|include)\s*[:=]?\s*(.+)',
        "substitute": r'(?:substitute|replace)\s*[:=]?\s*(.+)',
    }

    # Date change patterns
    DATE_CHANGE_PATTERNS = [
        r'(?:submission|due)\s+date.*(?:extended|changed|revised)\s+(?:to|from)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'new\s+(?:submission|due)\s+date\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:pre-?bid|opening)\s+(?:date|meeting).*(?:on|to)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]

    def __init__(self):
        self.addendum_regex = [
            re.compile(p, re.IGNORECASE) for p in self.ADDENDUM_NO_PATTERNS
        ]

    def parse(self, documents: List[ParsedDocument]) -> List[Addendum]:
        """Parse addenda from documents."""
        addenda = []

        # Find addendum documents
        addenda_docs = [d for d in documents if d.doc_type == DocType.ADDENDUM]

        for doc in addenda_docs:
            addendum = self._parse_document(doc)
            if addendum:
                addenda.append(addendum)

        # Sort by addendum number
        addenda.sort(key=lambda a: int(a.addendum_no) if a.addendum_no.isdigit() else 0)

        return addenda

    def _parse_document(self, doc: ParsedDocument) -> Optional[Addendum]:
        """Parse a single addendum document."""
        text = doc.text_content

        # Extract addendum number
        addendum_no = "1"  # Default
        for pattern in self.addendum_regex:
            match = pattern.search(text)
            if match:
                addendum_no = match.group(1)
                break

        # Extract date
        date = self._extract_date(text)

        # Extract title
        title = self._extract_title(text, doc.file_name)

        # Extract changes
        changes = self._extract_changes(text)

        # Extract clarifications
        clarifications = self._extract_clarifications(text)

        # Extract BOQ changes
        boq_changes = self._extract_boq_changes(text)

        # Extract date changes
        date_changes = self._extract_date_changes(text)

        return Addendum(
            addendum_no=addendum_no,
            date=date,
            title=title,
            changes=changes,
            clarifications=clarifications,
            boq_changes=boq_changes,
            date_changes=date_changes,
            source_file=doc.file_name,
        )

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract addendum date."""
        patterns = [
            r'date\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'dated\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'issued\s+on\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_title(self, text: str, filename: str) -> str:
        """Extract addendum title."""
        # Try to find subject line
        patterns = [
            r'subject\s*[:=]?\s*(.+?)(?:\n|$)',
            r're\s*[:=]?\s*(.+?)(?:\n|$)',
            r'regarding\s*[:=]?\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]

        # Use filename
        return filename.replace('.pdf', '').replace('_', ' ').title()

    def _extract_changes(self, text: str) -> List[Dict]:
        """Extract change items."""
        changes = []

        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            for change_type, pattern in self.CHANGE_PATTERNS.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Look for "instead of" in previous lines
                    original = ""
                    if i > 0 and "instead" not in line.lower():
                        prev_line = lines[i-1].strip()
                        instead_match = re.search(r'instead\s+of\s*[:=]?\s*(.+)', prev_line, re.IGNORECASE)
                        if instead_match:
                            original = instead_match.group(1)

                    changes.append({
                        "type": change_type,
                        "original": original,
                        "revised": match.group(1).strip(),
                        "context": line,
                    })
                    break

        return changes

    def _extract_clarifications(self, text: str) -> List[str]:
        """Extract clarification items."""
        clarifications = []

        # Look for Q&A patterns
        qa_pattern = r'(?:q|query|question)\s*[\d.]+\s*[:=]?\s*(.+?)(?:a|answer|reply)\s*[:=]?\s*(.+?)(?=(?:q|query|question)\s*\d|$)'
        matches = re.findall(qa_pattern, text, re.IGNORECASE | re.DOTALL)

        for q, a in matches:
            clarifications.append(f"Q: {q.strip()[:200]}\nA: {a.strip()[:200]}")

        # Also look for numbered clarifications
        numbered = re.findall(r'^\s*\d+[.)]\s*(.{20,200})', text, re.MULTILINE)
        for item in numbered[:20]:  # Limit
            if item not in str(clarifications):
                clarifications.append(item.strip())

        return clarifications[:30]  # Limit

    def _extract_boq_changes(self, text: str) -> List[Dict]:
        """Extract BOQ-related changes."""
        boq_changes = []

        # Pattern for item number changes
        patterns = [
            r'item\s*(?:no\.?)?\s*(\d+(?:\.\d+)?)\s*[:=]?\s*(.+?)(?:quantity|rate|amount)\s*(?:revised|changed|corrected)\s*(?:to|from)\s*(\d+(?:,\d+)?(?:\.\d+)?)',
            r'(?:quantity|rate)\s+(?:for|of)\s+item\s*(\d+(?:\.\d+)?)\s*(?:is|changed|revised)\s*(?:to)?\s*(\d+(?:,\d+)?(?:\.\d+)?)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                boq_changes.append({
                    "item_no": match[0],
                    "description": match[1] if len(match) > 2 else "",
                    "new_value": match[-1],
                })

        return boq_changes

    def _extract_date_changes(self, text: str) -> List[Dict]:
        """Extract date changes."""
        date_changes = []

        for pattern in self.DATE_CHANGE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find what date was changed
                context = text[max(0, text.lower().find(match.lower())-50):text.lower().find(match.lower())+len(match)+50]

                date_type = "submission"
                if "pre-bid" in context.lower() or "pre bid" in context.lower():
                    date_type = "pre_bid"
                elif "opening" in context.lower():
                    date_type = "opening"

                date_changes.append({
                    "date_type": date_type,
                    "new_date": match,
                    "context": context.strip()[:100],
                })

        return date_changes

    def get_latest_dates(self, addenda: List[Addendum]) -> Dict[str, str]:
        """Get latest dates from all addenda."""
        dates = {
            "submission": None,
            "pre_bid": None,
            "opening": None,
        }

        # Process addenda in order (they're already sorted)
        for addendum in addenda:
            for change in addendum.date_changes:
                dates[change["date_type"]] = change["new_date"]

        return dates

    def get_all_boq_changes(self, addenda: List[Addendum]) -> List[Dict]:
        """Get all BOQ changes from all addenda."""
        all_changes = []
        for addendum in addenda:
            for change in addendum.boq_changes:
                change["addendum_no"] = addendum.addendum_no
                all_changes.append(change)
        return all_changes
