"""
Revision Tracker - Build and maintain revision history across sheets.

Provides:
- Per-sheet revision tracking
- Project-wide revision timeline
- Revision grouping by date/description
- Latest revision detection
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from .detector import RevisionEntry, RevisionTable


@dataclass
class SheetRevision:
    """Revision history for a single sheet."""
    sheet_id: str
    sheet_name: str = ""
    revisions: List[RevisionEntry] = field(default_factory=list)
    current_revision: Optional[str] = None
    current_date: Optional[str] = None
    revision_count: int = 0

    def to_dict(self) -> dict:
        return {
            "sheet_id": self.sheet_id,
            "sheet_name": self.sheet_name,
            "revisions": [r.to_dict() for r in self.revisions],
            "current_revision": self.current_revision,
            "current_date": self.current_date,
            "revision_count": self.revision_count,
        }


@dataclass
class RevisionTimelineEntry:
    """Entry in project-wide revision timeline."""
    revision: str
    date: Optional[str]
    sheet_count: int
    sheets: List[str]
    description: str
    revision_types: List[str]

    def to_dict(self) -> dict:
        return {
            "revision": self.revision,
            "date": self.date,
            "sheet_count": self.sheet_count,
            "sheets": self.sheets,
            "description": self.description,
            "revision_types": self.revision_types,
        }


@dataclass
class RevisionHistory:
    """Complete revision history for a project."""
    project_id: str = ""
    sheets: Dict[str, SheetRevision] = field(default_factory=dict)
    timeline: List[RevisionTimelineEntry] = field(default_factory=list)
    latest_revision: Optional[str] = None
    latest_date: Optional[str] = None
    total_revisions: int = 0
    sheets_without_revisions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "sheets": {k: v.to_dict() for k, v in self.sheets.items()},
            "timeline": [t.to_dict() for t in self.timeline],
            "latest_revision": self.latest_revision,
            "latest_date": self.latest_date,
            "total_revisions": self.total_revisions,
            "sheets_without_revisions": self.sheets_without_revisions,
        }


class RevisionTracker:
    """Build and track revision history."""

    def __init__(self):
        pass

    def build_history(
        self,
        pages: List[Dict],
        revision_tables: List[RevisionTable],
    ) -> RevisionHistory:
        """
        Build complete revision history.

        Args:
            pages: Indexed pages
            revision_tables: Detected revision tables

        Returns:
            RevisionHistory with all data
        """
        history = RevisionHistory()

        # Index revision tables by sheet
        table_by_sheet = {t.sheet_id: t for t in revision_tables}

        # Process each page
        all_revisions = []
        for page in pages:
            sheet_id = self._get_sheet_id(page)
            sheet_name = page.get("sheet_name", "") or page.get("drawing_title", "")

            if sheet_id in table_by_sheet:
                table = table_by_sheet[sheet_id]
                sheet_rev = SheetRevision(
                    sheet_id=sheet_id,
                    sheet_name=sheet_name,
                    revisions=table.entries,
                    current_revision=table.latest_revision,
                    current_date=table.latest_date,
                    revision_count=len(table.entries),
                )
                history.sheets[sheet_id] = sheet_rev
                all_revisions.extend(table.entries)
            else:
                # Sheet without detected revisions
                history.sheets_without_revisions.append(sheet_id)

        # Build timeline
        history.timeline = self._build_timeline(all_revisions)

        # Find latest revision
        if all_revisions:
            sorted_revs = sorted(
                all_revisions,
                key=lambda r: (self._date_sort_key(r.date), self._rev_sort_key(r.revision)),
                reverse=True
            )
            history.latest_revision = sorted_revs[0].revision
            history.latest_date = sorted_revs[0].date

        history.total_revisions = len(all_revisions)

        return history

    def _get_sheet_id(self, page: Dict) -> str:
        """Get sheet ID from page dict."""
        if "page_id" in page:
            return page["page_id"]

        file_path = page.get("file_path", "unknown")
        page_num = page.get("page_number", 0)
        if isinstance(page_num, int):
            page_num += 1

        from pathlib import Path
        file_name = Path(file_path).stem
        return f"{file_name}_p{page_num}"

    def _build_timeline(
        self, revisions: List[RevisionEntry]
    ) -> List[RevisionTimelineEntry]:
        """Build project-wide revision timeline."""
        # Group by revision number
        by_revision: Dict[str, List[RevisionEntry]] = {}
        for rev in revisions:
            key = rev.revision
            if key not in by_revision:
                by_revision[key] = []
            by_revision[key].append(rev)

        # Create timeline entries
        timeline = []
        for rev_num, entries in by_revision.items():
            # Get date (most common or latest)
            dates = [e.date for e in entries if e.date]
            date = max(dates) if dates else None

            # Get sheets
            sheets = list(set(e.source_page for e in entries))

            # Get descriptions (combine unique)
            descriptions = list(set(e.description for e in entries if e.description))
            combined_desc = "; ".join(descriptions[:3])

            # Get revision types
            types = list(set(e.revision_type.value for e in entries))

            timeline.append(RevisionTimelineEntry(
                revision=rev_num,
                date=date,
                sheet_count=len(sheets),
                sheets=sheets,
                description=combined_desc,
                revision_types=types,
            ))

        # Sort by revision
        timeline.sort(key=lambda t: self._rev_sort_key(t.revision))

        return timeline

    def _date_sort_key(self, date: Optional[str]) -> str:
        """Generate sort key for date."""
        if not date:
            return "0000-00-00"

        # Assume YYYY-MM-DD format from normalization
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return date

    def _rev_sort_key(self, revision: str) -> tuple:
        """Generate sort key for revision."""
        import re

        # Extract number or letter
        num_match = re.search(r'\d+', revision)
        letter_match = re.search(r'[A-Z]$', revision, re.IGNORECASE)

        if num_match:
            return (0, int(num_match.group()))
        elif letter_match:
            return (1, ord(letter_match.group().upper()))
        else:
            return (2, 0)

    def get_sheets_at_revision(
        self, history: RevisionHistory, revision: str
    ) -> List[str]:
        """Get list of sheets that have a specific revision."""
        sheets = []
        for sheet_id, sheet_rev in history.sheets.items():
            for rev in sheet_rev.revisions:
                if rev.revision == revision:
                    sheets.append(sheet_id)
                    break
        return sheets

    def get_sheets_after_date(
        self, history: RevisionHistory, date: str
    ) -> List[str]:
        """Get sheets revised after a specific date."""
        sheets = []
        for sheet_id, sheet_rev in history.sheets.items():
            for rev in sheet_rev.revisions:
                if rev.date and rev.date >= date:
                    sheets.append(sheet_id)
                    break
        return sheets

    def find_revision_gaps(self, history: RevisionHistory) -> List[str]:
        """Find sheets that may be missing revisions."""
        gaps = []

        if not history.latest_revision:
            return gaps

        latest = history.latest_revision

        for sheet_id, sheet_rev in history.sheets.items():
            if not sheet_rev.current_revision:
                gaps.append(sheet_id)
            elif sheet_rev.current_revision != latest:
                # Sheet is behind current revision
                gaps.append(sheet_id)

        return gaps
