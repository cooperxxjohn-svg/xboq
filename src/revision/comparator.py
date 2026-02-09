"""
Sheet Comparator - Compare drawing sets to identify changes.

Provides:
- Hash-based sheet comparison
- Content diff detection
- Geometry change detection
- Text change detection
- Changed sheet identification for delta processing
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class ChangeType(Enum):
    """Type of change detected."""
    ADDED = "added"          # New sheet
    REMOVED = "removed"      # Sheet deleted
    MODIFIED = "modified"    # Content changed
    GEOMETRY = "geometry"    # Geometry/dimensions changed
    TEXT = "text"            # Text/annotations changed
    REVISION = "revision"    # Only revision table updated
    UNCHANGED = "unchanged"  # No changes


@dataclass
class SheetDiff:
    """Difference detected for a sheet."""
    sheet_id: str
    change_type: ChangeType
    summary: str
    geometry_changes: List[str] = field(default_factory=list)
    text_changes: List[str] = field(default_factory=list)
    dimension_changes: List[Dict] = field(default_factory=list)
    added_elements: List[str] = field(default_factory=list)
    removed_elements: List[str] = field(default_factory=list)
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "sheet_id": self.sheet_id,
            "change_type": self.change_type.value,
            "summary": self.summary,
            "geometry_changes": self.geometry_changes,
            "text_changes": self.text_changes,
            "dimension_changes": self.dimension_changes,
            "added_elements": self.added_elements,
            "removed_elements": self.removed_elements,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "confidence": self.confidence,
        }


@dataclass
class ChangeReport:
    """Report of all changes between two drawing sets."""
    diffs: List[SheetDiff] = field(default_factory=list)
    added_sheet_ids: List[str] = field(default_factory=list)
    removed_sheet_ids: List[str] = field(default_factory=list)
    changed_sheet_ids: List[str] = field(default_factory=list)
    unchanged_sheet_ids: List[str] = field(default_factory=list)
    total_changes: int = 0
    geometry_changes_count: int = 0
    text_changes_count: int = 0
    revision_only_count: int = 0

    def to_dict(self) -> dict:
        return {
            "diffs": [d.to_dict() for d in self.diffs],
            "added_sheet_ids": self.added_sheet_ids,
            "removed_sheet_ids": self.removed_sheet_ids,
            "changed_sheet_ids": self.changed_sheet_ids,
            "unchanged_sheet_ids": self.unchanged_sheet_ids,
            "total_changes": self.total_changes,
            "geometry_changes_count": self.geometry_changes_count,
            "text_changes_count": self.text_changes_count,
            "revision_only_count": self.revision_only_count,
        }


class SheetComparator:
    """Compare drawing sets to identify changes."""

    def __init__(self):
        pass

    def compare_sets(
        self,
        current_pages: List[Dict],
        previous_pages: List[Dict],
        current_results: List[Dict],
        previous_results: List[Dict],
    ) -> ChangeReport:
        """
        Compare two drawing sets.

        Args:
            current_pages: Current version indexed pages
            previous_pages: Previous version indexed pages
            current_results: Current extraction results
            previous_results: Previous extraction results

        Returns:
            ChangeReport with all detected differences
        """
        report = ChangeReport()

        # Index results by sheet ID
        current_by_id = self._index_by_sheet(current_results)
        previous_by_id = self._index_by_sheet(previous_results)

        current_ids = set(current_by_id.keys())
        previous_ids = set(previous_by_id.keys())

        # Find added sheets
        added = current_ids - previous_ids
        for sheet_id in added:
            report.added_sheet_ids.append(sheet_id)
            report.diffs.append(SheetDiff(
                sheet_id=sheet_id,
                change_type=ChangeType.ADDED,
                summary="New sheet added",
                new_hash=self._compute_hash(current_by_id[sheet_id]),
            ))

        # Find removed sheets
        removed = previous_ids - current_ids
        for sheet_id in removed:
            report.removed_sheet_ids.append(sheet_id)
            report.diffs.append(SheetDiff(
                sheet_id=sheet_id,
                change_type=ChangeType.REMOVED,
                summary="Sheet removed from set",
                old_hash=self._compute_hash(previous_by_id[sheet_id]),
            ))

        # Compare common sheets
        common = current_ids & previous_ids
        for sheet_id in common:
            diff = self._compare_sheets(
                sheet_id,
                current_by_id[sheet_id],
                previous_by_id[sheet_id],
            )

            if diff.change_type == ChangeType.UNCHANGED:
                report.unchanged_sheet_ids.append(sheet_id)
            else:
                report.changed_sheet_ids.append(sheet_id)
                report.diffs.append(diff)

                if diff.change_type == ChangeType.GEOMETRY:
                    report.geometry_changes_count += 1
                elif diff.change_type == ChangeType.TEXT:
                    report.text_changes_count += 1
                elif diff.change_type == ChangeType.REVISION:
                    report.revision_only_count += 1

        report.total_changes = len(report.diffs)

        return report

    def _index_by_sheet(self, results: List[Dict]) -> Dict[str, Dict]:
        """Index extraction results by sheet ID."""
        from pathlib import Path

        indexed = {}
        for result in results:
            file_name = Path(result.get("file_path", "")).stem
            page_num = result.get("page_number", 0) + 1
            sheet_id = f"{file_name}_p{page_num}"
            indexed[sheet_id] = result

        return indexed

    def _compute_hash(self, result: Dict) -> str:
        """Compute content hash for a sheet."""
        # Hash key elements
        hasher = hashlib.sha256()

        # Hash rooms
        rooms = result.get("rooms", [])
        for room in sorted(rooms, key=lambda r: r.get("room_id", "")):
            room_str = f"{room.get('room_id', '')}:{room.get('label', '')}:{room.get('area_sqm', 0):.2f}"
            hasher.update(room_str.encode())

        # Hash dimensions
        dimensions = result.get("dimensions", [])
        for dim in sorted(dimensions, key=lambda d: str(d)):
            hasher.update(str(dim).encode())

        # Hash text items (excluding revision table)
        text_items = result.get("text_items", [])
        for item in sorted(text_items, key=lambda t: t.get("text", "")):
            text = item.get("text", "")
            # Skip revision-related text
            if not self._is_revision_text(text):
                hasher.update(text.encode())

        # Hash doors/windows
        openings = result.get("openings", [])
        for opening in sorted(openings, key=lambda o: o.get("tag", "")):
            hasher.update(str(opening).encode())

        return hasher.hexdigest()[:16]

    def _is_revision_text(self, text: str) -> bool:
        """Check if text is revision-related."""
        import re

        patterns = [
            r'^R\d+$',
            r'^Rev\.?\s*\d+$',
            r'^Rev\.?\s*[A-Z]$',
            r'revision',
            r'date',
            r'prepared by',
            r'approved by',
        ]

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _compare_sheets(
        self,
        sheet_id: str,
        current: Dict,
        previous: Dict,
    ) -> SheetDiff:
        """Compare two versions of a sheet."""
        # Compute hashes
        current_hash = self._compute_hash(current)
        previous_hash = self._compute_hash(previous)

        # Quick check - if hashes match, no changes
        if current_hash == previous_hash:
            return SheetDiff(
                sheet_id=sheet_id,
                change_type=ChangeType.UNCHANGED,
                summary="No changes detected",
                old_hash=previous_hash,
                new_hash=current_hash,
            )

        # Detailed comparison
        geometry_changes = []
        text_changes = []
        dimension_changes = []
        added_elements = []
        removed_elements = []

        # Compare rooms
        current_rooms = {r.get("room_id"): r for r in current.get("rooms", [])}
        previous_rooms = {r.get("room_id"): r for r in previous.get("rooms", [])}

        for room_id in set(current_rooms.keys()) | set(previous_rooms.keys()):
            if room_id in current_rooms and room_id not in previous_rooms:
                added_elements.append(f"Room: {room_id}")
            elif room_id not in current_rooms and room_id in previous_rooms:
                removed_elements.append(f"Room: {room_id}")
            elif room_id in current_rooms and room_id in previous_rooms:
                curr = current_rooms[room_id]
                prev = previous_rooms[room_id]

                # Check area change
                curr_area = curr.get("area_sqm", 0)
                prev_area = prev.get("area_sqm", 0)
                if abs(curr_area - prev_area) > 0.1:
                    geometry_changes.append(
                        f"Room {room_id}: area {prev_area:.2f} → {curr_area:.2f} sqm"
                    )

                # Check label change
                if curr.get("label") != prev.get("label"):
                    text_changes.append(
                        f"Room {room_id}: label '{prev.get('label')}' → '{curr.get('label')}'"
                    )

        # Compare dimensions
        current_dims = set(str(d) for d in current.get("dimensions", []))
        previous_dims = set(str(d) for d in previous.get("dimensions", []))

        for dim in current_dims - previous_dims:
            dimension_changes.append({"type": "added", "value": dim})
        for dim in previous_dims - current_dims:
            dimension_changes.append({"type": "removed", "value": dim})

        # Compare text items (simplified)
        current_texts = set(t.get("text", "") for t in current.get("text_items", []))
        previous_texts = set(t.get("text", "") for t in previous.get("text_items", []))

        for text in current_texts - previous_texts:
            if not self._is_revision_text(text) and len(text) < 100:
                text_changes.append(f"Added: {text[:50]}")
        for text in previous_texts - current_texts:
            if not self._is_revision_text(text) and len(text) < 100:
                text_changes.append(f"Removed: {text[:50]}")

        # Determine change type
        if geometry_changes or dimension_changes or added_elements or removed_elements:
            change_type = ChangeType.GEOMETRY
        elif text_changes:
            change_type = ChangeType.TEXT
        else:
            # Only revision changed
            change_type = ChangeType.REVISION

        # Generate summary
        summary_parts = []
        if geometry_changes:
            summary_parts.append(f"{len(geometry_changes)} geometry changes")
        if text_changes:
            summary_parts.append(f"{len(text_changes)} text changes")
        if dimension_changes:
            summary_parts.append(f"{len(dimension_changes)} dimension changes")
        if added_elements:
            summary_parts.append(f"{len(added_elements)} elements added")
        if removed_elements:
            summary_parts.append(f"{len(removed_elements)} elements removed")

        summary = "; ".join(summary_parts) if summary_parts else "Revision update only"

        return SheetDiff(
            sheet_id=sheet_id,
            change_type=change_type,
            summary=summary,
            geometry_changes=geometry_changes,
            text_changes=text_changes,
            dimension_changes=dimension_changes,
            added_elements=added_elements,
            removed_elements=removed_elements,
            old_hash=previous_hash,
            new_hash=current_hash,
        )

    def get_sheets_requiring_reprocessing(
        self, report: ChangeReport
    ) -> List[str]:
        """Get list of sheets that need re-extraction."""
        sheets = []

        # Added sheets need full processing
        sheets.extend(report.added_sheet_ids)

        # Changed sheets need reprocessing
        for diff in report.diffs:
            if diff.change_type in [ChangeType.MODIFIED, ChangeType.GEOMETRY]:
                sheets.append(diff.sheet_id)

        return list(set(sheets))
