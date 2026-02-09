"""
Conflict Detection Engine for XBOQ RFI System.

Detects conflicts between:
- Multiple notes specifying different values (wall thickness, heights)
- Scale note vs dimension-derived scale
- Schedule sizes vs measured opening widths
- Finish legend vs room template assignments
- Multiple schedules disagreeing on same tag

India-specific construction estimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts detected."""
    WALL_THICKNESS = "wall_thickness"
    CEILING_HEIGHT = "ceiling_height"
    SCALE_MISMATCH = "scale_mismatch"
    DOOR_SIZE = "door_size"
    WINDOW_SIZE = "window_size"
    FINISH_ASSIGNMENT = "finish_assignment"
    SCHEDULE_CONFLICT = "schedule_conflict"
    DIMENSION_CONFLICT = "dimension_conflict"
    SPEC_CONFLICT = "spec_conflict"


@dataclass
class ConflictSource:
    """Source of a conflicting value."""
    source_type: str  # note, schedule, legend, dimension, default
    value: Any
    page_id: str
    snippet: str  # Text snippet as evidence
    confidence: float


@dataclass
class Conflict:
    """Detected conflict between sources."""
    conflict_id: str
    conflict_type: ConflictType
    description: str
    sources: List[ConflictSource] = field(default_factory=list)
    severity: str = "medium"  # high, medium, low
    affected_items: List[str] = field(default_factory=list)
    resolution_suggestion: str = ""


@dataclass
class ConflictReport:
    """Complete conflict detection report."""
    project_id: str
    conflicts: List[Conflict] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class ConflictDetector:
    """
    Detects conflicts between various data sources in the drawing set.
    """

    def __init__(self):
        self.conflicts: List[Conflict] = []
        self._conflict_counter = 0

    def detect_all_conflicts(
        self,
        project_id: str,
        extraction_results: List[Dict],
        schedules: List[Dict],
        notes: List[Dict],
        legends: List[Dict],
        override_report: Dict,
        triangulation_report: Dict,
    ) -> ConflictReport:
        """
        Detect all conflicts in the project data.

        Args:
            project_id: Project identifier
            extraction_results: Page extraction results
            schedules: Extracted schedule tables
            notes: Extracted notes
            legends: Extracted legends
            override_report: Override analysis results
            triangulation_report: Triangulation results

        Returns:
            Complete conflict report
        """
        self.conflicts = []
        self._conflict_counter = 0

        # Detect wall thickness conflicts
        self._detect_wall_thickness_conflicts(notes, extraction_results)

        # Detect ceiling height conflicts
        self._detect_ceiling_height_conflicts(notes, extraction_results)

        # Detect scale conflicts
        self._detect_scale_conflicts(extraction_results, triangulation_report)

        # Detect door/window size conflicts
        self._detect_opening_size_conflicts(schedules, extraction_results)

        # Detect finish assignment conflicts
        self._detect_finish_conflicts(schedules, legends, extraction_results)

        # Detect schedule conflicts (multiple schedules for same tag)
        self._detect_schedule_conflicts(schedules)

        # Detect specification conflicts from notes
        self._detect_spec_conflicts(notes)

        # Build report
        report = self._build_report(project_id)

        return report

    def _generate_conflict_id(self) -> str:
        """Generate unique conflict ID."""
        self._conflict_counter += 1
        return f"CONF-{self._conflict_counter:04d}"

    def _detect_wall_thickness_conflicts(
        self,
        notes: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Detect conflicting wall thickness specifications."""

        wall_thickness_values = []

        # Pattern to extract wall thickness from notes
        thickness_pattern = re.compile(
            r"(\d+)\s*(?:mm)?\s*(?:thick|thk|wall)",
            re.IGNORECASE
        )

        # Extract from notes
        for note in notes:
            text = note.get("text", "")
            page_id = note.get("page_id", "unknown")

            for match in thickness_pattern.finditer(text):
                thickness = int(match.group(1))
                # Validate reasonable range (75-450mm)
                if 75 <= thickness <= 450:
                    wall_thickness_values.append(ConflictSource(
                        source_type="note",
                        value=thickness,
                        page_id=page_id,
                        snippet=match.group(0),
                        confidence=0.85,
                    ))

        # Extract from extraction results (detected walls)
        for result in extraction_results:
            if result.get("page_type") == "floor_plan":
                page_id = f"{result.get('file_path', '').split('/')[-1].split('.')[0]}_p{result.get('page_number', 0) + 1}"

                for wall in result.get("walls", []):
                    thickness = wall.get("thickness_mm", 0)
                    if 75 <= thickness <= 450:
                        wall_thickness_values.append(ConflictSource(
                            source_type="dimension",
                            value=thickness,
                            page_id=page_id,
                            snippet=f"Measured wall thickness: {thickness}mm",
                            confidence=0.7,
                        ))

        # Find conflicts (different values)
        if len(wall_thickness_values) >= 2:
            unique_values = set(s.value for s in wall_thickness_values)
            if len(unique_values) > 1:
                # Check if difference is significant (not just 230 vs 225)
                values_list = sorted(unique_values)
                if values_list[-1] - values_list[0] > 20:  # >20mm difference
                    self.conflicts.append(Conflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.WALL_THICKNESS,
                        description=f"Multiple wall thickness values found: {sorted(unique_values)}mm",
                        sources=wall_thickness_values,
                        severity="medium",
                        affected_items=["all_walls"],
                        resolution_suggestion="Confirm wall thickness for external (typically 230mm) and internal (typically 115mm) walls separately.",
                    ))

    def _detect_ceiling_height_conflicts(
        self,
        notes: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Detect conflicting ceiling height specifications."""

        height_values = []

        # Pattern for ceiling height
        height_pattern = re.compile(
            r"(?:ceiling|clg|floor\s*to\s*ceiling|clear)\s*(?:ht|height)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:m|mm|ft)?",
            re.IGNORECASE
        )

        for note in notes:
            text = note.get("text", "")
            page_id = note.get("page_id", "unknown")

            for match in height_pattern.finditer(text):
                height = float(match.group(1))
                # Normalize to meters
                if height > 100:  # Likely mm
                    height = height / 1000
                elif height > 10:  # Likely ft
                    height = height * 0.3048

                if 2.0 <= height <= 6.0:
                    height_values.append(ConflictSource(
                        source_type="note",
                        value=round(height, 2),
                        page_id=page_id,
                        snippet=match.group(0),
                        confidence=0.85,
                    ))

        # Find conflicts
        if len(height_values) >= 2:
            unique_values = set(s.value for s in height_values)
            if len(unique_values) > 1:
                values_list = sorted(unique_values)
                if values_list[-1] - values_list[0] > 0.15:  # >150mm difference
                    self.conflicts.append(Conflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.CEILING_HEIGHT,
                        description=f"Multiple ceiling heights found: {sorted(unique_values)}m",
                        sources=height_values,
                        severity="medium",
                        affected_items=["all_rooms"],
                        resolution_suggestion="Confirm floor-to-floor height and clear ceiling height for each floor level.",
                    ))

    def _detect_scale_conflicts(
        self,
        extraction_results: List[Dict],
        triangulation_report: Dict,
    ) -> None:
        """Detect scale note vs dimension-derived scale conflicts."""

        for result in extraction_results:
            if result.get("page_type") != "floor_plan":
                continue

            page_id = f"{result.get('file_path', '').split('/')[-1].split('.')[0]}_p{result.get('page_number', 0) + 1}"

            scale_note = result.get("scale", {}).get("scale_note", "")
            scale_derived = result.get("scale", {}).get("dimension_derived", "")
            scale_value = result.get("scale", {}).get("value", 0)
            derived_value = result.get("scale", {}).get("derived_value", 0)

            if scale_value > 0 and derived_value > 0:
                diff_pct = abs(scale_value - derived_value) / max(scale_value, derived_value) * 100
                if diff_pct > 10:  # >10% difference
                    sources = [
                        ConflictSource(
                            source_type="scale_note",
                            value=f"1:{scale_value}",
                            page_id=page_id,
                            snippet=scale_note or f"Scale 1:{scale_value}",
                            confidence=0.9,
                        ),
                        ConflictSource(
                            source_type="dimension_derived",
                            value=f"1:{derived_value}",
                            page_id=page_id,
                            snippet=f"Derived from dimensions: 1:{derived_value}",
                            confidence=0.75,
                        ),
                    ]

                    self.conflicts.append(Conflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.SCALE_MISMATCH,
                        description=f"Scale note (1:{scale_value}) conflicts with dimension-derived scale (1:{derived_value}), {diff_pct:.1f}% difference",
                        sources=sources,
                        severity="high",
                        affected_items=[page_id],
                        resolution_suggestion="Verify correct scale. All area and length calculations depend on accurate scale.",
                    ))

    def _detect_opening_size_conflicts(
        self,
        schedules: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Detect door/window schedule sizes vs measured sizes."""

        # Build schedule lookup
        schedule_sizes = {}  # tag -> {width, height, source}
        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            page_id = schedule.get("page_id", "unknown")

            if "door" in schedule_type or "window" in schedule_type:
                for entry in schedule.get("entries", []):
                    tag = entry.get("mark", entry.get("tag", entry.get("type", ""))).upper()
                    size = entry.get("size", entry.get("dimensions", ""))

                    if tag and size:
                        # Parse size (e.g., "900x2100", "3'-0"x7'-0"")
                        width, height = self._parse_size(size)
                        if width and height:
                            schedule_sizes[tag] = {
                                "width": width,
                                "height": height,
                                "size_str": size,
                                "page_id": page_id,
                            }

        # Compare with measured openings
        for result in extraction_results:
            if result.get("page_type") != "floor_plan":
                continue

            page_id = f"{result.get('file_path', '').split('/')[-1].split('.')[0]}_p{result.get('page_number', 0) + 1}"

            # Check doors
            for door in result.get("doors", []):
                tag = door.get("tag", door.get("mark", "")).upper()
                measured_width = door.get("width_mm", 0)

                if tag in schedule_sizes and measured_width > 0:
                    schedule_width = schedule_sizes[tag]["width"]
                    diff = abs(measured_width - schedule_width)

                    if diff > 100:  # >100mm difference
                        sources = [
                            ConflictSource(
                                source_type="schedule",
                                value=schedule_width,
                                page_id=schedule_sizes[tag]["page_id"],
                                snippet=f"Door {tag}: {schedule_sizes[tag]['size_str']}",
                                confidence=0.95,
                            ),
                            ConflictSource(
                                source_type="measured",
                                value=measured_width,
                                page_id=page_id,
                                snippet=f"Measured opening: {measured_width}mm",
                                confidence=0.7,
                            ),
                        ]

                        self.conflicts.append(Conflict(
                            conflict_id=self._generate_conflict_id(),
                            conflict_type=ConflictType.DOOR_SIZE,
                            description=f"Door {tag} schedule size ({schedule_width}mm) differs from measured ({measured_width}mm) by {diff}mm",
                            sources=sources,
                            severity="medium",
                            affected_items=[f"door_{tag}"],
                            resolution_suggestion=f"Confirm correct width for door {tag}. Schedule sizes are typically more reliable.",
                        ))

            # Check windows
            for window in result.get("windows", []):
                tag = window.get("tag", window.get("mark", "")).upper()
                measured_width = window.get("width_mm", 0)

                if tag in schedule_sizes and measured_width > 0:
                    schedule_width = schedule_sizes[tag]["width"]
                    diff = abs(measured_width - schedule_width)

                    if diff > 100:
                        sources = [
                            ConflictSource(
                                source_type="schedule",
                                value=schedule_width,
                                page_id=schedule_sizes[tag]["page_id"],
                                snippet=f"Window {tag}: {schedule_sizes[tag]['size_str']}",
                                confidence=0.95,
                            ),
                            ConflictSource(
                                source_type="measured",
                                value=measured_width,
                                page_id=page_id,
                                snippet=f"Measured opening: {measured_width}mm",
                                confidence=0.7,
                            ),
                        ]

                        self.conflicts.append(Conflict(
                            conflict_id=self._generate_conflict_id(),
                            conflict_type=ConflictType.WINDOW_SIZE,
                            description=f"Window {tag} schedule size ({schedule_width}mm) differs from measured ({measured_width}mm) by {diff}mm",
                            sources=sources,
                            severity="medium",
                            affected_items=[f"window_{tag}"],
                            resolution_suggestion=f"Confirm correct width for window {tag}.",
                        ))

    def _detect_finish_conflicts(
        self,
        schedules: List[Dict],
        legends: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Detect finish specification conflicts."""

        # Build finish lookup from schedules
        schedule_finishes = {}  # room_type -> {flooring, wall, ceiling}
        for schedule in schedules:
            if "finish" in schedule.get("type", "").lower():
                page_id = schedule.get("page_id", "unknown")

                for entry in schedule.get("entries", []):
                    room_type = entry.get("room_type", entry.get("space", "")).lower()
                    if room_type:
                        schedule_finishes[room_type] = {
                            "flooring": entry.get("floor", entry.get("flooring", "")),
                            "wall": entry.get("wall", ""),
                            "ceiling": entry.get("ceiling", ""),
                            "page_id": page_id,
                        }

        # Build finish lookup from legends
        legend_finishes = {}
        for legend in legends:
            if "finish" in legend.get("type", "").lower():
                page_id = legend.get("page_id", "unknown")

                for item in legend.get("items", []):
                    room_type = item.get("room_type", "").lower()
                    if room_type:
                        legend_finishes[room_type] = {
                            "flooring": item.get("flooring", ""),
                            "wall": item.get("wall", ""),
                            "page_id": page_id,
                        }

        # Compare schedule vs legend
        for room_type in set(schedule_finishes.keys()) & set(legend_finishes.keys()):
            sched = schedule_finishes[room_type]
            legend = legend_finishes[room_type]

            # Compare flooring
            if sched["flooring"] and legend["flooring"]:
                if sched["flooring"].lower() != legend["flooring"].lower():
                    sources = [
                        ConflictSource(
                            source_type="schedule",
                            value=sched["flooring"],
                            page_id=sched["page_id"],
                            snippet=f"{room_type}: {sched['flooring']}",
                            confidence=0.9,
                        ),
                        ConflictSource(
                            source_type="legend",
                            value=legend["flooring"],
                            page_id=legend["page_id"],
                            snippet=f"{room_type}: {legend['flooring']}",
                            confidence=0.85,
                        ),
                    ]

                    self.conflicts.append(Conflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.FINISH_ASSIGNMENT,
                        description=f"Flooring finish conflict for {room_type}: schedule says '{sched['flooring']}', legend says '{legend['flooring']}'",
                        sources=sources,
                        severity="medium",
                        affected_items=[room_type],
                        resolution_suggestion=f"Confirm correct flooring finish for {room_type}.",
                    ))

    def _detect_schedule_conflicts(
        self,
        schedules: List[Dict],
    ) -> None:
        """Detect conflicts between multiple schedules for same tag."""

        # Group schedule entries by tag
        tag_entries = {}  # tag -> list of {value, page_id}

        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            page_id = schedule.get("page_id", "unknown")

            for entry in schedule.get("entries", []):
                tag = entry.get("mark", entry.get("tag", entry.get("type", ""))).upper()
                if not tag:
                    continue

                # Create a hashable representation of the entry
                entry_repr = {
                    "size": entry.get("size", entry.get("dimensions", "")),
                    "material": entry.get("material", ""),
                    "type": entry.get("type", entry.get("door_type", entry.get("window_type", ""))),
                    "page_id": page_id,
                    "schedule_type": schedule_type,
                }

                if tag not in tag_entries:
                    tag_entries[tag] = []
                tag_entries[tag].append(entry_repr)

        # Find conflicts
        for tag, entries in tag_entries.items():
            if len(entries) < 2:
                continue

            # Compare entries
            conflicts_found = False
            conflict_details = []

            for i, e1 in enumerate(entries):
                for e2 in entries[i+1:]:
                    # Compare sizes
                    if e1["size"] and e2["size"] and e1["size"] != e2["size"]:
                        conflicts_found = True
                        conflict_details.append(f"Size: {e1['size']} vs {e2['size']}")

                    # Compare materials
                    if e1["material"] and e2["material"] and e1["material"].lower() != e2["material"].lower():
                        conflicts_found = True
                        conflict_details.append(f"Material: {e1['material']} vs {e2['material']}")

            if conflicts_found:
                sources = [
                    ConflictSource(
                        source_type="schedule",
                        value=f"{e['size']} / {e['material']}",
                        page_id=e["page_id"],
                        snippet=f"From {e['schedule_type']}",
                        confidence=0.9,
                    )
                    for e in entries
                ]

                self.conflicts.append(Conflict(
                    conflict_id=self._generate_conflict_id(),
                    conflict_type=ConflictType.SCHEDULE_CONFLICT,
                    description=f"Multiple schedule entries for {tag} have conflicting values: {'; '.join(conflict_details)}",
                    sources=sources,
                    severity="high",
                    affected_items=[tag],
                    resolution_suggestion=f"Verify which schedule is authoritative for {tag}.",
                ))

    def _detect_spec_conflicts(
        self,
        notes: List[Dict],
    ) -> None:
        """Detect conflicting specifications in notes."""

        # Patterns for specifications
        spec_patterns = {
            "concrete_grade": (
                re.compile(r"(M\d{2})", re.IGNORECASE),
                "concrete grade",
            ),
            "steel_grade": (
                re.compile(r"(Fe\d{3}D?)", re.IGNORECASE),
                "steel grade",
            ),
            "paint_type": (
                re.compile(r"(acrylic emulsion|oil bound distemper|OBD|enamel paint)", re.IGNORECASE),
                "paint type",
            ),
            "tile_size": (
                re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})\s*(?:mm)?", re.IGNORECASE),
                "tile size",
            ),
        }

        spec_values = {}  # spec_type -> list of {value, page_id, snippet}

        for note in notes:
            text = note.get("text", "")
            page_id = note.get("page_id", "unknown")

            for spec_type, (pattern, description) in spec_patterns.items():
                for match in pattern.finditer(text):
                    value = match.group(0)
                    if spec_type not in spec_values:
                        spec_values[spec_type] = []
                    spec_values[spec_type].append({
                        "value": value.upper(),
                        "page_id": page_id,
                        "snippet": text[:100],
                    })

        # Find conflicts in each spec type
        for spec_type, values in spec_values.items():
            unique_values = set(v["value"] for v in values)
            if len(unique_values) > 1:
                sources = [
                    ConflictSource(
                        source_type="note",
                        value=v["value"],
                        page_id=v["page_id"],
                        snippet=v["snippet"],
                        confidence=0.8,
                    )
                    for v in values
                ]

                description = spec_patterns[spec_type][1]
                self.conflicts.append(Conflict(
                    conflict_id=self._generate_conflict_id(),
                    conflict_type=ConflictType.SPEC_CONFLICT,
                    description=f"Multiple {description} values found: {sorted(unique_values)}",
                    sources=sources,
                    severity="medium",
                    affected_items=[spec_type],
                    resolution_suggestion=f"Confirm correct {description} to use.",
                ))

    def _parse_size(self, size_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse size string to width and height in mm."""
        if not size_str:
            return (None, None)

        size_str = size_str.strip()

        # Pattern: 900x2100 or 900 x 2100
        match = re.search(r"(\d+)\s*[x×]\s*(\d+)", size_str)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            if w > 100 and h > 100:
                return (w, h)
            elif w < 10 and h < 10:
                # Likely feet
                return (int(w * 304.8), int(h * 304.8))

        # Pattern: 3'-0" x 7'-0"
        match = re.search(r"(\d+)['\-](\d*)[\"']?\s*[x×]\s*(\d+)['\-](\d*)[\"']?", size_str)
        if match:
            w_ft = int(match.group(1))
            w_in = int(match.group(2)) if match.group(2) else 0
            h_ft = int(match.group(3))
            h_in = int(match.group(4)) if match.group(4) else 0
            w_mm = int((w_ft * 12 + w_in) * 25.4)
            h_mm = int((h_ft * 12 + h_in) * 25.4)
            return (w_mm, h_mm)

        return (None, None)

    def _build_report(self, project_id: str) -> ConflictReport:
        """Build conflict detection report."""

        report = ConflictReport(project_id=project_id)
        report.conflicts = self.conflicts

        # Summary
        by_type = {}
        by_severity = {}

        for conflict in self.conflicts:
            t = conflict.conflict_type.value
            s = conflict.severity

            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1

        report.summary = {
            "total_conflicts": len(self.conflicts),
            "by_type": by_type,
            "by_severity": by_severity,
            "high_priority": by_severity.get("high", 0),
        }

        return report


def detect_conflicts(
    project_id: str,
    extraction_results: List[Dict],
    schedules: List[Dict],
    notes: List[Dict],
    legends: List[Dict],
    override_report: Dict,
    triangulation_report: Dict,
) -> ConflictReport:
    """
    Convenience function to detect all conflicts.
    """
    detector = ConflictDetector()
    return detector.detect_all_conflicts(
        project_id=project_id,
        extraction_results=extraction_results,
        schedules=schedules,
        notes=notes,
        legends=legends,
        override_report=override_report,
        triangulation_report=triangulation_report,
    )
