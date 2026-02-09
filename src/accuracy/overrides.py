"""
Cross-Sheet Override Engine for XBOQ.

Uses schedule tables, legends, and notes to override model outputs
with authoritative data from drawings.

India-specific construction estimation accuracy layer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class OverrideSource(Enum):
    """Source of override data."""
    DOOR_SCHEDULE = "door_schedule"
    WINDOW_SCHEDULE = "window_schedule"
    FINISH_LEGEND = "finish_legend"
    FINISH_SCHEDULE = "finish_schedule"
    GENERAL_NOTES = "general_notes"
    STRUCTURAL_NOTES = "structural_notes"
    WATERPROOFING_NOTES = "waterproofing_notes"
    MEP_LEGEND = "mep_legend"
    TITLE_BLOCK = "title_block"
    SPECIFICATION = "specification"


class OverrideType(Enum):
    """Type of override applied."""
    SIZE_CORRECTION = "size_correction"        # Door/window sizes from schedule
    FINISH_ASSIGNMENT = "finish_assignment"    # Room finish from legend
    DIMENSION_OVERRIDE = "dimension_override"  # Wall thickness, ceiling height
    SCOPE_DETECTION = "scope_detection"        # Waterproofing, MEP scope
    QUANTITY_CORRECTION = "quantity_correction"  # Count corrections
    MATERIAL_SPECIFICATION = "material_specification"  # Material grade/type


@dataclass
class Override:
    """Single override applied to model output."""
    override_id: str
    override_type: OverrideType
    source: OverrideSource
    target: str  # What is being overridden (e.g., "door_D01", "room_R003")
    original_value: Any
    override_value: Any
    confidence: float  # 0-100
    source_reference: str  # Page/cell reference
    notes: str = ""


@dataclass
class OverrideReport:
    """Complete override report for a project."""
    project_id: str
    overrides: List[Override] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    scope_detections: List[str] = field(default_factory=list)  # Scope items detected from notes


class CrossSheetOverrideEngine:
    """
    Cross-sheet intelligence engine.

    Uses authoritative schedule/legend data to override model outputs.
    """

    # India-specific wall thickness patterns (in mm)
    WALL_THICKNESS_PATTERNS = {
        r"(\d+)\s*mm\s*(thick|thk|wall)": "wall_thickness",
        r"wall\s*[:=]?\s*(\d+)\s*mm": "wall_thickness",
        r"(\d+)\s*mm\s*aac": "aac_wall",
        r"(\d+)\s*mm\s*brick": "brick_wall",
        r"(\d+)\s*mm\s*block": "block_wall",
    }

    # Ceiling height patterns
    CEILING_HEIGHT_PATTERNS = {
        r"ceiling\s*(?:ht|height)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:m|mm|ft)": "ceiling_height",
        r"floor\s*to\s*(?:ceiling|soffit)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:m|mm|ft)": "ceiling_height",
        r"clear\s*(?:ht|height)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:m|mm|ft)": "clear_height",
        r"(\d+(?:\.\d+)?)\s*m\s*(?:clr|clear|flr to clg)": "ceiling_height",
    }

    # Waterproofing scope detection patterns
    WATERPROOFING_PATTERNS = {
        r"waterproof(?:ing)?\s*(?:to|in|at)\s*(\w+)": "location",
        r"(\w+)\s*waterproof(?:ing)?": "location",
        r"wp\s*(?:treatment|system|membrane)": "system",
        r"app\s*membrane": "app_membrane",
        r"sbr\s*(?:coat|treatment)": "sbr_treatment",
        r"cementitious\s*(?:coat|waterproof)": "cementitious",
        r"bituminous\s*(?:coat|felt)": "bituminous",
        r"puddle\s*concrete": "puddle_concrete",
        r"water\s*tank\s*(?:lining|treatment)": "tank_lining",
    }

    # Finish legend patterns
    FINISH_PATTERNS = {
        r"vitrified\s*(?:tiles?|flooring)": ("flooring", "vitrified_tiles"),
        r"ceramic\s*(?:tiles?|flooring)": ("flooring", "ceramic_tiles"),
        r"marble\s*(?:flooring)?": ("flooring", "marble"),
        r"granite\s*(?:flooring)?": ("flooring", "granite"),
        r"kota\s*(?:stone)?": ("flooring", "kota_stone"),
        r"ips\s*(?:flooring)?": ("flooring", "ips"),
        r"anti[\-\s]?skid": ("flooring", "anti_skid_tiles"),
        r"rustic\s*(?:tiles?)?": ("flooring", "rustic_tiles"),
        r"wood(?:en)?\s*(?:flooring|laminate)": ("flooring", "wooden"),
        r"false\s*ceiling": ("ceiling", "false_ceiling"),
        r"pop\s*(?:ceiling|punning)": ("ceiling", "pop"),
        r"gypsum\s*(?:board|ceiling)": ("ceiling", "gypsum"),
        r"grid\s*ceiling": ("ceiling", "grid_ceiling"),
        r"acrylic\s*(?:emulsion|paint)": ("wall_finish", "acrylic_emulsion"),
        r"oil\s*bound\s*distemper": ("wall_finish", "obd"),
        r"texture\s*(?:paint|finish)": ("wall_finish", "texture"),
        r"wall\s*(?:tiles?|cladding)": ("wall_finish", "wall_tiles"),
    }

    def __init__(self):
        self.overrides: List[Override] = []
        self.scope_detections: List[str] = []
        self._override_counter = 0

    def process_cross_sheet_data(
        self,
        project_id: str,
        extraction_results: List[Dict],
        schedules: List[Dict],
        notes: List[Dict],
        legends: List[Dict],
        boq_entries: List[Dict],
        scope_register: Dict,
    ) -> OverrideReport:
        """
        Process all cross-sheet data and generate overrides.

        Args:
            project_id: Project identifier
            extraction_results: Page extraction results
            schedules: Extracted schedule tables
            notes: Extracted notes from drawings
            legends: Extracted legends
            boq_entries: Current BOQ entries
            scope_register: Current scope register

        Returns:
            Complete override report
        """
        self.overrides = []
        self.scope_detections = []
        self._override_counter = 0

        # Process door schedules
        self._process_door_schedules(schedules, extraction_results)

        # Process window schedules
        self._process_window_schedules(schedules, extraction_results)

        # Process finish legends/schedules
        self._process_finish_data(schedules, legends, extraction_results)

        # Process notes for dimension overrides
        self._process_notes_for_dimensions(notes, extraction_results)

        # Process notes for scope detection
        self._process_notes_for_scope(notes, scope_register)

        # Build report
        report = self._build_report(project_id)

        return report

    def _generate_override_id(self) -> str:
        """Generate unique override ID."""
        self._override_counter += 1
        return f"OVR-{self._override_counter:04d}"

    def _process_door_schedules(
        self,
        schedules: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Process door schedules to override detected door sizes."""

        door_schedule_data = {}

        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            if "door" not in schedule_type:
                continue

            page_ref = schedule.get("page_id", "unknown")

            for entry in schedule.get("entries", []):
                door_mark = entry.get("mark", entry.get("type", entry.get("id", "")))
                if not door_mark:
                    continue

                # Extract size (common formats: 900x2100, 3'-0"x7'-0", etc.)
                size_str = entry.get("size", entry.get("dimensions", ""))
                width, height = self._parse_door_size(size_str)

                # Extract material/type
                material = entry.get("material", entry.get("type", ""))
                finish = entry.get("finish", "")
                qty = entry.get("quantity", entry.get("qty", 1))

                door_schedule_data[door_mark.upper()] = {
                    "width_mm": width,
                    "height_mm": height,
                    "material": material,
                    "finish": finish,
                    "quantity": qty,
                    "page_ref": page_ref,
                }

        # Apply overrides to detected doors
        for page in extraction_results:
            if page.get("page_type") != "floor_plan":
                continue

            for door in page.get("doors", []):
                door_tag = door.get("tag", door.get("mark", "")).upper()
                if door_tag in door_schedule_data:
                    sched_data = door_schedule_data[door_tag]

                    # Size override
                    detected_width = door.get("width_mm", 0)
                    schedule_width = sched_data["width_mm"]

                    if schedule_width and detected_width:
                        if abs(detected_width - schedule_width) > 50:  # >50mm difference
                            self.overrides.append(Override(
                                override_id=self._generate_override_id(),
                                override_type=OverrideType.SIZE_CORRECTION,
                                source=OverrideSource.DOOR_SCHEDULE,
                                target=f"door_{door_tag}",
                                original_value={"width_mm": detected_width},
                                override_value={"width_mm": schedule_width},
                                confidence=95,
                                source_reference=f"Door Schedule ({sched_data['page_ref']})",
                                notes=f"Door {door_tag}: pixel-detected {detected_width}mm → schedule {schedule_width}mm"
                            ))

                    # Material specification
                    if sched_data["material"]:
                        self.overrides.append(Override(
                            override_id=self._generate_override_id(),
                            override_type=OverrideType.MATERIAL_SPECIFICATION,
                            source=OverrideSource.DOOR_SCHEDULE,
                            target=f"door_{door_tag}",
                            original_value=door.get("material", "unknown"),
                            override_value=sched_data["material"],
                            confidence=95,
                            source_reference=f"Door Schedule ({sched_data['page_ref']})",
                            notes=f"Door {door_tag}: material from schedule"
                        ))

    def _process_window_schedules(
        self,
        schedules: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Process window schedules to override detected window sizes."""

        window_schedule_data = {}

        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            if "window" not in schedule_type:
                continue

            page_ref = schedule.get("page_id", "unknown")

            for entry in schedule.get("entries", []):
                window_mark = entry.get("mark", entry.get("type", entry.get("id", "")))
                if not window_mark:
                    continue

                size_str = entry.get("size", entry.get("dimensions", ""))
                width, height = self._parse_window_size(size_str)

                material = entry.get("material", entry.get("type", ""))
                sill_height = entry.get("sill_height", entry.get("sill", ""))
                qty = entry.get("quantity", entry.get("qty", 1))

                window_schedule_data[window_mark.upper()] = {
                    "width_mm": width,
                    "height_mm": height,
                    "material": material,
                    "sill_height": sill_height,
                    "quantity": qty,
                    "page_ref": page_ref,
                }

        # Apply overrides
        for page in extraction_results:
            if page.get("page_type") != "floor_plan":
                continue

            for window in page.get("windows", []):
                window_tag = window.get("tag", window.get("mark", "")).upper()
                if window_tag in window_schedule_data:
                    sched_data = window_schedule_data[window_tag]

                    detected_width = window.get("width_mm", 0)
                    schedule_width = sched_data["width_mm"]

                    if schedule_width and detected_width:
                        if abs(detected_width - schedule_width) > 50:
                            self.overrides.append(Override(
                                override_id=self._generate_override_id(),
                                override_type=OverrideType.SIZE_CORRECTION,
                                source=OverrideSource.WINDOW_SCHEDULE,
                                target=f"window_{window_tag}",
                                original_value={"width_mm": detected_width},
                                override_value={"width_mm": schedule_width},
                                confidence=95,
                                source_reference=f"Window Schedule ({sched_data['page_ref']})",
                                notes=f"Window {window_tag}: pixel-detected {detected_width}mm → schedule {schedule_width}mm"
                            ))

                    if sched_data["material"]:
                        self.overrides.append(Override(
                            override_id=self._generate_override_id(),
                            override_type=OverrideType.MATERIAL_SPECIFICATION,
                            source=OverrideSource.WINDOW_SCHEDULE,
                            target=f"window_{window_tag}",
                            original_value=window.get("material", "unknown"),
                            override_value=sched_data["material"],
                            confidence=95,
                            source_reference=f"Window Schedule ({sched_data['page_ref']})",
                            notes=f"Window {window_tag}: material from schedule"
                        ))

    def _process_finish_data(
        self,
        schedules: List[Dict],
        legends: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Process finish schedules/legends to override room finishes."""

        room_finishes = {}  # room_type -> {flooring, wall, ceiling}

        # Extract from finish schedules
        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            if "finish" not in schedule_type:
                continue

            page_ref = schedule.get("page_id", "unknown")

            for entry in schedule.get("entries", []):
                room_type = entry.get("room_type", entry.get("area", entry.get("space", ""))).lower()
                if not room_type:
                    continue

                floor_finish = entry.get("floor", entry.get("flooring", ""))
                wall_finish = entry.get("wall", entry.get("wall_finish", ""))
                ceiling_finish = entry.get("ceiling", "")
                skirting = entry.get("skirting", "")
                dado = entry.get("dado", "")

                room_finishes[room_type] = {
                    "flooring": floor_finish,
                    "wall": wall_finish,
                    "ceiling": ceiling_finish,
                    "skirting": skirting,
                    "dado": dado,
                    "page_ref": page_ref,
                }

        # Extract from legends
        for legend in legends:
            legend_type = legend.get("type", "").lower()
            if "finish" not in legend_type:
                continue

            page_ref = legend.get("page_id", "unknown")

            for item in legend.get("items", []):
                # Parse finish pattern
                text = item.get("text", item.get("description", "")).lower()
                for pattern, (category, finish_type) in self.FINISH_PATTERNS.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        # Check if associated with a room type
                        room_match = re.search(r"(bedroom|living|kitchen|toilet|bathroom|balcony|passage|dining)", text)
                        if room_match:
                            room_type = room_match.group(1)
                            if room_type not in room_finishes:
                                room_finishes[room_type] = {"page_ref": page_ref}
                            room_finishes[room_type][category] = finish_type

        # Apply overrides to rooms
        for page in extraction_results:
            if page.get("page_type") != "floor_plan":
                continue

            for room in page.get("rooms", []):
                room_label = room.get("label", "").lower()

                # Find matching finish specification
                for room_type, finishes in room_finishes.items():
                    if room_type in room_label or room_label in room_type:
                        # Apply finish overrides
                        if finishes.get("flooring"):
                            current_floor = room.get("floor_finish", "assumed_vitrified")
                            self.overrides.append(Override(
                                override_id=self._generate_override_id(),
                                override_type=OverrideType.FINISH_ASSIGNMENT,
                                source=OverrideSource.FINISH_SCHEDULE,
                                target=f"room_{room.get('id', room_label)}",
                                original_value={"flooring": current_floor},
                                override_value={"flooring": finishes["flooring"]},
                                confidence=90,
                                source_reference=f"Finish Schedule ({finishes['page_ref']})",
                                notes=f"Room '{room_label}': floor finish from schedule"
                            ))

                        if finishes.get("wall"):
                            current_wall = room.get("wall_finish", "assumed_paint")
                            self.overrides.append(Override(
                                override_id=self._generate_override_id(),
                                override_type=OverrideType.FINISH_ASSIGNMENT,
                                source=OverrideSource.FINISH_SCHEDULE,
                                target=f"room_{room.get('id', room_label)}",
                                original_value={"wall_finish": current_wall},
                                override_value={"wall_finish": finishes["wall"]},
                                confidence=90,
                                source_reference=f"Finish Schedule ({finishes['page_ref']})",
                                notes=f"Room '{room_label}': wall finish from schedule"
                            ))
                        break

    def _process_notes_for_dimensions(
        self,
        notes: List[Dict],
        extraction_results: List[Dict],
    ) -> None:
        """Process notes for wall thickness and ceiling height overrides."""

        wall_thickness_mm = None
        ceiling_height_m = None
        note_refs = []

        for note in notes:
            note_text = note.get("text", "").lower()
            page_ref = note.get("page_id", "unknown")

            # Check wall thickness patterns
            for pattern, _ in self.WALL_THICKNESS_PATTERNS.items():
                match = re.search(pattern, note_text, re.IGNORECASE)
                if match:
                    thickness = int(match.group(1))
                    # Validate reasonable thickness (75mm - 450mm)
                    if 75 <= thickness <= 450:
                        wall_thickness_mm = thickness
                        note_refs.append(page_ref)
                        break

            # Check ceiling height patterns
            for pattern, _ in self.CEILING_HEIGHT_PATTERNS.items():
                match = re.search(pattern, note_text, re.IGNORECASE)
                if match:
                    height = float(match.group(1))
                    # Convert if needed
                    if height > 10:  # Likely in mm
                        height = height / 1000
                    elif height > 3 and "ft" in note_text:  # Likely in feet
                        height = height * 0.3048
                    # Validate (2m - 6m)
                    if 2.0 <= height <= 6.0:
                        ceiling_height_m = height
                        note_refs.append(page_ref)
                        break

        # Apply wall thickness override
        if wall_thickness_mm:
            self.overrides.append(Override(
                override_id=self._generate_override_id(),
                override_type=OverrideType.DIMENSION_OVERRIDE,
                source=OverrideSource.GENERAL_NOTES,
                target="all_walls",
                original_value={"wall_thickness_mm": 230},  # Default assumption
                override_value={"wall_thickness_mm": wall_thickness_mm},
                confidence=85,
                source_reference=f"General Notes ({', '.join(set(note_refs))})",
                notes=f"Wall thickness specified as {wall_thickness_mm}mm in notes"
            ))

        # Apply ceiling height override
        if ceiling_height_m:
            self.overrides.append(Override(
                override_id=self._generate_override_id(),
                override_type=OverrideType.DIMENSION_OVERRIDE,
                source=OverrideSource.GENERAL_NOTES,
                target="all_rooms",
                original_value={"ceiling_height_m": 3.0},  # Default assumption
                override_value={"ceiling_height_m": ceiling_height_m},
                confidence=85,
                source_reference=f"General Notes ({', '.join(set(note_refs))})",
                notes=f"Ceiling height specified as {ceiling_height_m}m in notes"
            ))

    def _process_notes_for_scope(
        self,
        notes: List[Dict],
        scope_register: Dict,
    ) -> None:
        """Process notes to detect scope items (waterproofing, MEP, etc.)."""

        detected_scope = set()

        for note in notes:
            note_text = note.get("text", "").lower()
            page_ref = note.get("page_id", "unknown")

            # Check waterproofing patterns
            for pattern, wp_type in self.WATERPROOFING_PATTERNS.items():
                if re.search(pattern, note_text, re.IGNORECASE):
                    # Determine location
                    location = "general"
                    if "toilet" in note_text or "bathroom" in note_text or "wc" in note_text:
                        location = "toilet"
                    elif "terrace" in note_text or "roof" in note_text:
                        location = "terrace"
                    elif "basement" in note_text or "below" in note_text:
                        location = "basement"
                    elif "tank" in note_text or "sump" in note_text:
                        location = "tank"
                    elif "sunken" in note_text:
                        location = "sunken"

                    scope_key = f"waterproofing_{location}"
                    if scope_key not in detected_scope:
                        detected_scope.add(scope_key)
                        self.scope_detections.append(scope_key)

                        self.overrides.append(Override(
                            override_id=self._generate_override_id(),
                            override_type=OverrideType.SCOPE_DETECTION,
                            source=OverrideSource.WATERPROOFING_NOTES,
                            target=f"scope_{scope_key}",
                            original_value="UNKNOWN",
                            override_value="DETECTED",
                            confidence=90,
                            source_reference=f"Notes ({page_ref})",
                            notes=f"Waterproofing scope detected: {location} ({wp_type})"
                        ))

            # Check for other MEP scope items
            mep_patterns = {
                r"fire\s*(?:alarm|detection|sprinkler|hydrant)": "fire_safety",
                r"hvac|air\s*conditioning|ac\s*(?:system|unit)": "hvac",
                r"lift|elevator": "lift",
                r"dg\s*set|generator": "dg_set",
                r"solar\s*(?:panel|system|water)": "solar",
                r"stp|sewage\s*treatment": "stp",
                r"rainwater\s*harvesting|rwh": "rwh",
            }

            for pattern, scope_type in mep_patterns.items():
                if re.search(pattern, note_text, re.IGNORECASE):
                    scope_key = f"mep_{scope_type}"
                    if scope_key not in detected_scope:
                        detected_scope.add(scope_key)
                        self.scope_detections.append(scope_key)

                        self.overrides.append(Override(
                            override_id=self._generate_override_id(),
                            override_type=OverrideType.SCOPE_DETECTION,
                            source=OverrideSource.MEP_LEGEND,
                            target=f"scope_{scope_key}",
                            original_value="UNKNOWN",
                            override_value="DETECTED",
                            confidence=85,
                            source_reference=f"Notes ({page_ref})",
                            notes=f"MEP scope detected: {scope_type}"
                        ))

    def _parse_door_size(self, size_str: str) -> Tuple[int, int]:
        """Parse door size string into width and height in mm."""
        if not size_str:
            return (0, 0)

        size_str = size_str.lower().strip()

        # Pattern: 900x2100 or 900 x 2100
        match = re.search(r"(\d+)\s*[x×]\s*(\d+)", size_str)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            # If values seem like mm, use directly
            if w > 100 and h > 100:
                return (w, h)
            # If values are small, might be in feet or meters
            if w < 10 and h < 10:
                # Assume feet
                return (int(w * 304.8), int(h * 304.8))

        # Pattern: 3'-0" x 7'-0" (imperial)
        match = re.search(r"(\d+)['\-](\d*)[\"']?\s*[x×]\s*(\d+)['\-](\d*)[\"']?", size_str)
        if match:
            w_ft = int(match.group(1))
            w_in = int(match.group(2)) if match.group(2) else 0
            h_ft = int(match.group(3))
            h_in = int(match.group(4)) if match.group(4) else 0
            w_mm = int((w_ft * 12 + w_in) * 25.4)
            h_mm = int((h_ft * 12 + h_in) * 25.4)
            return (w_mm, h_mm)

        return (0, 0)

    def _parse_window_size(self, size_str: str) -> Tuple[int, int]:
        """Parse window size string into width and height in mm."""
        # Same logic as door
        return self._parse_door_size(size_str)

    def _build_report(self, project_id: str) -> OverrideReport:
        """Build final override report."""

        report = OverrideReport(project_id=project_id)
        report.overrides = self.overrides
        report.scope_detections = self.scope_detections

        # Summary by type
        type_counts = {}
        source_counts = {}
        for override in self.overrides:
            t = override.override_type.value
            s = override.source.value
            type_counts[t] = type_counts.get(t, 0) + 1
            source_counts[s] = source_counts.get(s, 0) + 1

        report.summary = {
            "total_overrides": len(self.overrides),
            "by_type": type_counts,
            "by_source": source_counts,
            "scope_items_detected": len(self.scope_detections),
            "high_confidence_overrides": len([o for o in self.overrides if o.confidence >= 90]),
        }

        return report


def run_cross_sheet_overrides(
    project_id: str,
    extraction_results: List[Dict],
    schedules: List[Dict],
    notes: List[Dict],
    legends: List[Dict],
    boq_entries: List[Dict],
    scope_register: Dict,
) -> OverrideReport:
    """
    Run cross-sheet override engine.

    Args:
        project_id: Project identifier
        extraction_results: Page extraction results
        schedules: Extracted schedule tables
        notes: Extracted notes
        legends: Extracted legends
        boq_entries: Current BOQ entries
        scope_register: Current scope register

    Returns:
        Complete override report
    """
    engine = CrossSheetOverrideEngine()
    return engine.process_cross_sheet_data(
        project_id=project_id,
        extraction_results=extraction_results,
        schedules=schedules,
        notes=notes,
        legends=legends,
        boq_entries=boq_entries,
        scope_register=scope_register,
    )
