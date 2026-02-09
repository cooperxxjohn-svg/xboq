"""
Finish BOQ Calculator
Room-wise finish quantity calculations based on templates.

Features:
- Room type to finish mapping
- Floor, skirting, wall, ceiling calculations
- Opening deductions
- Waterproofing for wet areas
- India-specific finish types
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinishBOQItem:
    """BOQ line item for finish work."""
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    room_id: Optional[str] = None
    room_label: Optional[str] = None
    finish_type: Optional[str] = None
    element: Optional[str] = None  # floor, wall, ceiling, skirting


@dataclass
class RoomFinishResult:
    """Finish quantities for a single room."""
    room_id: str
    room_label: str
    room_type: str
    floor_area_sqm: float
    wall_area_sqm: float
    ceiling_area_sqm: float
    skirting_length_m: float
    boq_items: List[FinishBOQItem] = field(default_factory=list)
    deductions: Dict[str, float] = field(default_factory=dict)
    is_wet_area: bool = False
    is_external: bool = False


@dataclass
class FinishBOQResult:
    """Complete finish BOQ result."""
    room_results: List[RoomFinishResult]
    boq_items: List[FinishBOQItem]
    totals: Dict[str, float]
    assumptions_used: List[str]


class FinishBOQCalculator:
    """
    Calculate finish BOQ from room data and templates.

    Uses finish_templates.yaml for room-to-finish mapping.
    """

    # Excluded room types (no finishes)
    EXCLUDED_ROOMS = {"shaft", "duct", "ots", "open_to_sky", "void", "lift", "elevator"}

    # External room types
    EXTERNAL_ROOMS = {"balcony", "terrace", "deck", "verandah", "porch", "sit_out", "sitout"}

    # Wet areas
    WET_AREAS = {"toilet", "bathroom", "wc", "bath", "washroom", "utility", "laundry"}

    def __init__(
        self,
        templates_path: Optional[Path] = None,
        ceiling_height_mm: int = 3000,
    ):
        self.ceiling_height_mm = ceiling_height_mm
        self.ceiling_height_m = ceiling_height_mm / 1000
        self.templates: Dict = {}
        self.defaults: Dict = {}
        self.assumptions_used: List[str] = []

        # Load templates
        if templates_path is None:
            templates_path = Path(__file__).parent.parent.parent / "rules" / "finish_templates.yaml"

        self._load_templates(templates_path)

    def _load_templates(self, path: Path) -> None:
        """Load finish templates from YAML."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                self.templates = data.get("templates", {})
                self.defaults = data.get("defaults", {})

                # Load default template as fallback
                self.default_template = data.get("default", {})

        except Exception as e:
            logger.warning(f"Could not load templates: {e}")
            self._set_default_templates()

    def _set_default_templates(self) -> None:
        """Set hardcoded default templates if YAML fails."""
        self.templates = {
            "living": {
                "floor": {"type": "vitrified_tiles", "item_code": "FLR-VIT-01"},
                "walls": {"type": "plastic_emulsion", "item_code": "PNT-INT-01"},
                "ceiling": {"type": "plastic_emulsion", "item_code": "PNT-CLG-01"},
                "skirting": {"type": "tile_skirting", "item_code": "SKT-VIT-01"},
            },
            "bedroom": {
                "floor": {"type": "vitrified_tiles", "item_code": "FLR-VIT-01"},
                "walls": {"type": "plastic_emulsion", "item_code": "PNT-INT-01"},
                "ceiling": {"type": "plastic_emulsion", "item_code": "PNT-CLG-01"},
                "skirting": {"type": "tile_skirting", "item_code": "SKT-VIT-01"},
            },
            "toilet": {
                "floor": {"type": "anti_skid_tiles", "item_code": "FLR-AST-01"},
                "walls": {"type": "ceramic_tiles", "height_mm": 2100, "item_code": "WTL-CER-01"},
                "ceiling": {"type": "plastic_emulsion", "item_code": "PNT-CLG-01"},
            },
        }
        self.default_template = {
            "floor": {"type": "cement_flooring", "item_code": "FLR-CEM-01"},
            "walls": {"type": "plastic_emulsion", "item_code": "PNT-INT-01"},
            "ceiling": {"type": "plastic_emulsion", "item_code": "PNT-CLG-01"},
        }
        self.assumptions_used.append("Using default finish templates")

    def _match_room_type(self, room_label: str) -> str:
        """Match room label to template type."""
        label_lower = room_label.lower().strip()

        # Check each template for alias match
        for template_name, template_data in self.templates.items():
            aliases = template_data.get("aliases", [template_name])
            for alias in aliases:
                if alias.lower() in label_lower or label_lower in alias.lower():
                    return template_name

        # Check for partial matches
        for template_name in self.templates.keys():
            if template_name.lower() in label_lower:
                return template_name

        return "default"

    def _is_excluded(self, room_label: str) -> bool:
        """Check if room should be excluded from finishes."""
        label_lower = room_label.lower()
        for excluded in self.EXCLUDED_ROOMS:
            if excluded in label_lower:
                return True
        return False

    def _is_external(self, room_label: str) -> bool:
        """Check if room is external."""
        label_lower = room_label.lower()
        for external in self.EXTERNAL_ROOMS:
            if external in label_lower:
                return True
        return False

    def _is_wet_area(self, room_label: str) -> bool:
        """Check if room is wet area."""
        label_lower = room_label.lower()
        for wet in self.WET_AREAS:
            if wet in label_lower:
                return True
        return False

    def _calculate_perimeter_from_polygon(self, polygon: List[List[float]], scale_px_per_mm: float = 0.1) -> float:
        """Calculate perimeter from polygon points."""
        if not polygon or len(polygon) < 3:
            return 0.0

        perimeter_px = 0.0
        n = len(polygon)

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            perimeter_px += dist

        # Convert to meters using scale
        if scale_px_per_mm and scale_px_per_mm > 0:
            perimeter_m = perimeter_px / scale_px_per_mm / 1000
        else:
            # Assume typical scale if not provided
            perimeter_m = perimeter_px / 0.1 / 1000  # 0.1 px/mm default
            assumption = "Perimeter scale assumed (0.1 px/mm)"
            if assumption not in self.assumptions_used:
                self.assumptions_used.append(assumption)

        return perimeter_m

    def _estimate_perimeter_from_area(self, area_sqm: float) -> float:
        """Estimate perimeter assuming roughly rectangular room with 3:4 aspect ratio."""
        if area_sqm <= 0:
            return 0.0

        # Assume 3:4 aspect ratio (typical room shape)
        # area = 3k * 4k = 12k²  → k = sqrt(area/12)
        # perimeter = 2(3k + 4k) = 14k
        import math
        k = math.sqrt(area_sqm / 12)
        perimeter = 14 * k
        # Only add assumption once
        assumption = "Perimeter estimated from area (3:4 ratio assumed)"
        if assumption not in self.assumptions_used:
            self.assumptions_used.append(assumption)
        return perimeter

    def calculate_room_finishes(
        self,
        room: Dict,
        openings: Optional[List[Dict]] = None,
        ceiling_height_m: Optional[float] = None,
    ) -> RoomFinishResult:
        """
        Calculate finish quantities for a single room.

        Args:
            room: Room dict with area_sqm, perimeter_m, label
            openings: List of openings in this room
            ceiling_height_m: Override ceiling height

        Returns:
            RoomFinishResult with quantities and BOQ items
        """
        room_id = room.get("room_id", room.get("id", ""))
        room_label = room.get("label", "Unknown")
        floor_area = room.get("area_sqm", 0)

        # Get perimeter - try multiple sources
        perimeter_m = room.get("perimeter_m", 0)

        if not perimeter_m or perimeter_m <= 0:
            # Try to calculate from polygon
            polygon = room.get("polygon", [])
            scale = room.get("scale_px_per_mm", 0.1)
            if polygon:
                perimeter_m = self._calculate_perimeter_from_polygon(polygon, scale)
            else:
                # Estimate from area
                perimeter_m = self._estimate_perimeter_from_area(floor_area)

        if ceiling_height_m:
            height_m = ceiling_height_m
        else:
            height_m = self.ceiling_height_m

        boq_items = []
        deductions = {}

        # Check if excluded
        if self._is_excluded(room_label):
            return RoomFinishResult(
                room_id=room_id,
                room_label=room_label,
                room_type="excluded",
                floor_area_sqm=0,
                wall_area_sqm=0,
                ceiling_area_sqm=0,
                skirting_length_m=0,
                boq_items=[],
                deductions={},
                is_wet_area=False,
                is_external=False,
            )

        # Match room type
        room_type = self._match_room_type(room_label)
        template = self.templates.get(room_type, self.default_template)

        is_external = self._is_external(room_label)
        is_wet_area = self._is_wet_area(room_label)

        # Calculate opening deductions
        door_width_total = 0
        opening_area = 0

        if openings:
            for opening in openings:
                w = opening.get("width_m", 0.9)
                h = opening.get("height_m", 2.1)
                if w:
                    door_width_total += w
                if w and h:
                    opening_area += w * h

        deductions["door_widths_m"] = door_width_total
        deductions["opening_area_sqm"] = opening_area

        # Calculate areas
        gross_wall_area = perimeter_m * height_m
        net_wall_area = max(0, gross_wall_area - opening_area)
        skirting_length = max(0, perimeter_m - door_width_total)
        ceiling_area = floor_area

        # --- Floor finish ---
        floor_spec = template.get("floor", self.default_template.get("floor", {}))
        if floor_spec and floor_spec.get("type") != "none":
            wastage = floor_spec.get("wastage", 0.05)
            floor_qty = floor_area * (1 + wastage)

            boq_items.append(FinishBOQItem(
                item_code=floor_spec.get("item_code", "FLR-001"),
                description=floor_spec.get("description", f"Floor {floor_spec.get('type', 'flooring')}"),
                qty=round(floor_qty, 2),
                unit="sqm",
                derived_from="room_finish_mapping",
                confidence=0.85,
                room_id=room_id,
                room_label=room_label,
                finish_type=floor_spec.get("type"),
                element="floor",
            ))

        # --- Skirting ---
        skirting_spec = template.get("skirting", {})
        if skirting_spec and skirting_spec.get("type") != "none" and skirting_length > 0:
            boq_items.append(FinishBOQItem(
                item_code=skirting_spec.get("item_code", "SKT-001"),
                description=skirting_spec.get("description", "Skirting"),
                qty=round(skirting_length, 2),
                unit="rm",
                derived_from="room_finish_mapping",
                confidence=0.82,
                room_id=room_id,
                room_label=room_label,
                finish_type=skirting_spec.get("type"),
                element="skirting",
            ))

        # --- Wall finish ---
        wall_spec = template.get("walls", self.default_template.get("walls", {}))
        if wall_spec:
            wall_type = wall_spec.get("type")

            if wall_type == "mixed":
                # Dado + paint above
                dado_spec = wall_spec.get("dado", {})
                dado_height = dado_spec.get("height_mm", 900) / 1000
                dado_area = perimeter_m * dado_height

                boq_items.append(FinishBOQItem(
                    item_code=dado_spec.get("item_code", "DAD-001"),
                    description=dado_spec.get("description", "Dado tiles"),
                    qty=round(dado_area, 2),
                    unit="sqm",
                    derived_from="room_finish_mapping",
                    confidence=0.80,
                    room_id=room_id,
                    room_label=room_label,
                    finish_type="dado",
                    element="wall",
                ))

                # Paint above dado
                paint_area = max(0, net_wall_area - dado_area)
                above_dado_spec = wall_spec.get("above_dado", {})
                boq_items.append(FinishBOQItem(
                    item_code=above_dado_spec.get("item_code", "PNT-INT-01"),
                    description=above_dado_spec.get("description", "Wall paint above dado"),
                    qty=round(paint_area, 2),
                    unit="sqm",
                    derived_from="room_finish_mapping",
                    confidence=0.78,
                    room_id=room_id,
                    room_label=room_label,
                    finish_type="paint",
                    element="wall",
                ))

            elif wall_type == "ceramic_tiles":
                # Full wall tiles (toilet)
                tile_height = wall_spec.get("height_mm", 2100) / 1000
                tile_area = perimeter_m * tile_height
                wastage = wall_spec.get("wastage", 0.07)

                boq_items.append(FinishBOQItem(
                    item_code=wall_spec.get("item_code", "WTL-001"),
                    description=wall_spec.get("description", f"Wall tiles up to {tile_height}m"),
                    qty=round(tile_area * (1 + wastage), 2),
                    unit="sqm",
                    derived_from="room_finish_mapping",
                    confidence=0.83,
                    room_id=room_id,
                    room_label=room_label,
                    finish_type="wall_tiles",
                    element="wall",
                ))

                # Paint above tiles
                above_tiles = wall_spec.get("above_tiles", {})
                if above_tiles:
                    paint_height = height_m - tile_height
                    if paint_height > 0:
                        paint_area = perimeter_m * paint_height - opening_area * (paint_height / height_m)
                        boq_items.append(FinishBOQItem(
                            item_code=above_tiles.get("item_code", "PNT-INT-01"),
                            description=above_tiles.get("description", "Wall paint above tiles"),
                            qty=round(max(0, paint_area), 2),
                            unit="sqm",
                            derived_from="room_finish_mapping",
                            confidence=0.75,
                            room_id=room_id,
                            room_label=room_label,
                            finish_type="paint",
                            element="wall",
                        ))

            else:
                # Simple paint
                boq_items.append(FinishBOQItem(
                    item_code=wall_spec.get("item_code", "PNT-INT-01"),
                    description=wall_spec.get("description", "Wall paint"),
                    qty=round(net_wall_area, 2),
                    unit="sqm",
                    derived_from="room_finish_mapping",
                    confidence=0.82,
                    room_id=room_id,
                    room_label=room_label,
                    finish_type=wall_type,
                    element="wall",
                ))

        # --- Ceiling finish ---
        ceiling_spec = template.get("ceiling", self.default_template.get("ceiling", {}))
        if ceiling_spec:
            boq_items.append(FinishBOQItem(
                item_code=ceiling_spec.get("item_code", "PNT-CLG-01"),
                description=ceiling_spec.get("description", "Ceiling paint"),
                qty=round(ceiling_area, 2),
                unit="sqm",
                derived_from="room_finish_mapping",
                confidence=0.80,
                room_id=room_id,
                room_label=room_label,
                finish_type=ceiling_spec.get("type"),
                element="ceiling",
            ))

        # --- Waterproofing (wet areas) ---
        if is_wet_area:
            wp_spec = template.get("waterproofing", {})
            if wp_spec:
                # Floor + walls up to 150mm
                wp_area = floor_area + (perimeter_m * 0.15)
                boq_items.append(FinishBOQItem(
                    item_code=wp_spec.get("item_code", "WPR-001"),
                    description=wp_spec.get("description", "Waterproofing treatment"),
                    qty=round(wp_area, 2),
                    unit="sqm",
                    derived_from="room_finish_mapping",
                    confidence=0.75,
                    room_id=room_id,
                    room_label=room_label,
                    finish_type="waterproofing",
                    element="waterproofing",
                ))

        return RoomFinishResult(
            room_id=room_id,
            room_label=room_label,
            room_type=room_type,
            floor_area_sqm=round(floor_area, 2),
            wall_area_sqm=round(net_wall_area, 2),
            ceiling_area_sqm=round(ceiling_area, 2),
            skirting_length_m=round(skirting_length, 2),
            boq_items=boq_items,
            deductions=deductions,
            is_wet_area=is_wet_area,
            is_external=is_external,
        )

    def calculate_all_rooms(
        self,
        rooms: List[Dict],
        room_openings_map: Optional[Dict[str, List[Dict]]] = None,
    ) -> FinishBOQResult:
        """
        Calculate finish BOQ for all rooms.

        Args:
            rooms: List of room dicts
            room_openings_map: Mapping of room_id to openings

        Returns:
            FinishBOQResult with all quantities
        """
        room_results = []
        all_boq_items = []

        for room in rooms:
            room_id = room.get("room_id", room.get("id", ""))

            # Get openings for this room
            openings = None
            if room_openings_map and room_id in room_openings_map:
                openings = room_openings_map[room_id]

            result = self.calculate_room_finishes(room, openings)
            room_results.append(result)
            all_boq_items.extend(result.boq_items)

        # Calculate totals
        totals = {
            "total_floor_sqm": sum(r.floor_area_sqm for r in room_results),
            "total_wall_sqm": sum(r.wall_area_sqm for r in room_results),
            "total_ceiling_sqm": sum(r.ceiling_area_sqm for r in room_results),
            "total_skirting_m": sum(r.skirting_length_m for r in room_results),
            "wet_area_sqm": sum(r.floor_area_sqm for r in room_results if r.is_wet_area),
            "external_area_sqm": sum(r.floor_area_sqm for r in room_results if r.is_external),
        }

        # Consolidate BOQ items by item_code
        consolidated = self._consolidate_boq_items(all_boq_items)

        return FinishBOQResult(
            room_results=room_results,
            boq_items=consolidated,
            totals=totals,
            assumptions_used=self.assumptions_used,
        )

    def _consolidate_boq_items(
        self,
        items: List[FinishBOQItem],
    ) -> List[FinishBOQItem]:
        """Consolidate BOQ items by item_code."""
        consolidated: Dict[str, FinishBOQItem] = {}

        for item in items:
            key = item.item_code

            if key in consolidated:
                # Add quantity
                consolidated[key].qty = round(consolidated[key].qty + item.qty, 2)
                # Average confidence
                consolidated[key].confidence = (consolidated[key].confidence + item.confidence) / 2
            else:
                # Create new consolidated item (without room-specific info)
                consolidated[key] = FinishBOQItem(
                    item_code=item.item_code,
                    description=item.description,
                    qty=item.qty,
                    unit=item.unit,
                    derived_from=item.derived_from,
                    confidence=item.confidence,
                    finish_type=item.finish_type,
                    element=item.element,
                )

        return list(consolidated.values())
