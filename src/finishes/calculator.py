"""
Finish Calculator Module
Calculates finish quantities for Indian residential projects.

Calculations:
- Floor finish area = room area - shafts/ducts
- Skirting length = room perimeter - door widths
- Wall paint area = perimeter × height - openings area
- Ceiling paint area = room area

India defaults:
- Ceiling height: 3.0m (editable)
- Skirting height: 100mm
- Door height: 2.1m
- Bathroom wall tile height: 2.1m (if Toilet/Bath/WC)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import yaml
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class RoomFinishes:
    """Calculated finish quantities for a single room."""
    room_id: str
    room_label: str
    room_category: str  # internal, external, wet_area, shaft

    # Areas and lengths
    floor_area_sqm: float
    carpet_area_sqm: float  # Minus shafts/ducts
    perimeter_m: float
    skirting_length_m: float  # Minus door widths
    wall_area_sqm: float  # Perimeter × height - openings
    ceiling_area_sqm: float

    # Heights used
    ceiling_height_m: float
    tile_height_m: Optional[float] = None  # For wet areas

    # Deductions
    deductions: Dict[str, float] = field(default_factory=dict)

    # Additional info
    openings_count: int = 0
    is_external: bool = False
    is_wet_area: bool = False
    needs_dado: bool = False

    # Assumptions used
    assumptions_used: List[str] = field(default_factory=list)


@dataclass
class OpeningDeduction:
    """Deduction for a single opening."""
    opening_id: str
    opening_type: str
    width_m: float
    height_m: float
    area_sqm: float


class FinishCalculator:
    """
    Calculate finish quantities for rooms.

    Uses room polygons, detected openings, and configurable defaults
    to compute floor, wall, ceiling, and skirting quantities.
    """

    # Room categories
    EXTERNAL_ROOMS = {
        "balcony", "terrace", "deck", "verandah", "porch",
        "sit_out", "sitout", "open_terrace", "utility_balcony",
    }

    WET_AREAS = {
        "toilet", "bathroom", "wc", "bath", "washroom",
        "shower", "utility", "wash_area", "laundry",
    }

    SHAFT_ROOMS = {
        "shaft", "duct", "ots", "open_to_sky", "lift_well",
        "lift", "elevator", "staircase_void", "void",
    }

    # Default heights (mm)
    DEFAULT_CEILING_HEIGHT_MM = 3000
    DEFAULT_SKIRTING_HEIGHT_MM = 100
    DEFAULT_DADO_HEIGHT_MM = 900
    DEFAULT_BATHROOM_TILE_HEIGHT_MM = 2100
    DEFAULT_DOOR_HEIGHT_MM = 2100
    DEFAULT_WINDOW_HEIGHT_MM = 1200

    def __init__(
        self,
        config_path: Optional[Path] = None,
        ceiling_height_mm: int = 3000,
        skirting_height_mm: int = 100,
    ):
        """
        Initialize calculator.

        Args:
            config_path: Path to assumptions.yaml
            ceiling_height_mm: Default ceiling height
            skirting_height_mm: Skirting height
        """
        self.ceiling_height_mm = ceiling_height_mm
        self.skirting_height_mm = skirting_height_mm

        # Room-specific ceiling heights
        self.room_ceiling_heights: Dict[str, int] = {}

        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)

        # Track assumptions
        self.assumptions_log: List[Dict] = []

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            if "finishes" in data:
                finishes = data["finishes"]
                self.ceiling_height_mm = finishes.get("ceiling_height_mm", self.ceiling_height_mm)
                self.skirting_height_mm = finishes.get("skirting_height_mm", self.skirting_height_mm)

                if "ceiling_heights" in finishes:
                    for room_type, height in finishes["ceiling_heights"].items():
                        self.room_ceiling_heights[room_type.lower()] = height

        except Exception as e:
            logger.warning(f"Could not load config: {e}")

    def _log_assumption(
        self,
        room_id: str,
        parameter: str,
        value: Any,
        reason: str,
    ) -> None:
        """Log an assumption made during calculation."""
        self.assumptions_log.append({
            "room_id": room_id,
            "parameter": parameter,
            "value": value,
            "reason": reason,
        })

    def categorize_room(self, room_label: str) -> str:
        """
        Categorize room based on label.

        Categories:
        - external: Balconies, terraces
        - wet_area: Toilets, bathrooms
        - shaft: Ducts, shafts, voids
        - internal: Everything else
        """
        label_lower = room_label.lower().replace(" ", "_")

        for keyword in self.EXTERNAL_ROOMS:
            if keyword in label_lower:
                return "external"

        for keyword in self.WET_AREAS:
            if keyword in label_lower:
                return "wet_area"

        for keyword in self.SHAFT_ROOMS:
            if keyword in label_lower:
                return "shaft"

        return "internal"

    def get_ceiling_height(
        self,
        room_label: str,
        room_category: str,
    ) -> int:
        """Get ceiling height for room type."""
        label_lower = room_label.lower().replace(" ", "_")

        # Check room-specific heights
        for room_type, height in self.room_ceiling_heights.items():
            if room_type in label_lower:
                return height

        # Category defaults
        if room_category == "wet_area":
            return 2700
        elif room_category == "external":
            return 2700

        return self.ceiling_height_mm

    def calculate_room_finishes(
        self,
        room: Dict,
        openings: Optional[List[Dict]] = None,
    ) -> RoomFinishes:
        """
        Calculate finish quantities for a single room.

        Args:
            room: Room dict with area_sqm, perimeter_m, label, id
            openings: List of openings in this room

        Returns:
            RoomFinishes object
        """
        room_id = room.get("room_id", room.get("id", ""))
        room_label = room.get("label", "Unknown")
        room_area = room.get("area_sqm", 0)
        room_perimeter = room.get("perimeter_m", 0)

        # Categorize room
        room_category = self.categorize_room(room_label)

        # Get ceiling height
        ceiling_height_mm = self.get_ceiling_height(room_label, room_category)
        ceiling_height_m = ceiling_height_mm / 1000

        assumptions_used = []

        # For shafts, no finishes
        if room_category == "shaft":
            return RoomFinishes(
                room_id=room_id,
                room_label=room_label,
                room_category=room_category,
                floor_area_sqm=0,
                carpet_area_sqm=0,
                perimeter_m=0,
                skirting_length_m=0,
                wall_area_sqm=0,
                ceiling_area_sqm=0,
                ceiling_height_m=0,
                is_external=False,
                is_wet_area=False,
                assumptions_used=["Shaft excluded from finishes"],
            )

        # Calculate floor area (minus deductions if any)
        floor_area = room_area
        carpet_area = room_area

        # Calculate wall area deductions from openings
        opening_deductions = []
        total_door_width = 0
        total_opening_area = 0

        if openings:
            for opening in openings:
                width_m = opening.get("width_m", 0.9)
                height_m = opening.get("height_m", 2.1)

                if "door" in opening.get("type", "").lower():
                    # Use actual or default door height
                    if not height_m:
                        height_m = self.DEFAULT_DOOR_HEIGHT_MM / 1000
                        assumptions_used.append(f"Door height assumed: {self.DEFAULT_DOOR_HEIGHT_MM}mm")

                    total_door_width += width_m
                    opening_area = width_m * height_m
                else:
                    # Window/ventilator
                    if not height_m:
                        height_m = self.DEFAULT_WINDOW_HEIGHT_MM / 1000
                        assumptions_used.append(f"Window height assumed: {self.DEFAULT_WINDOW_HEIGHT_MM}mm")

                    opening_area = width_m * height_m

                total_opening_area += opening_area
                opening_deductions.append({
                    "id": opening.get("id", ""),
                    "type": opening.get("type", ""),
                    "area_sqm": opening_area,
                })

        # Skirting length = perimeter - door widths
        skirting_length = max(0, room_perimeter - total_door_width)

        # Wall paint area = perimeter × height - openings
        gross_wall_area = room_perimeter * ceiling_height_m
        wall_area = max(0, gross_wall_area - total_opening_area)

        # Ceiling area = floor area
        ceiling_area = floor_area

        # Special handling for wet areas
        is_wet_area = room_category == "wet_area"
        tile_height_m = None

        if is_wet_area:
            tile_height_m = self.DEFAULT_BATHROOM_TILE_HEIGHT_MM / 1000
            assumptions_used.append(f"Bathroom tile height: {self.DEFAULT_BATHROOM_TILE_HEIGHT_MM}mm")

        # External rooms
        is_external = room_category == "external"

        # Build deductions dict
        deductions = {
            "door_widths_m": total_door_width,
            "opening_area_sqm": total_opening_area,
        }

        if ceiling_height_mm != self.DEFAULT_CEILING_HEIGHT_MM:
            assumptions_used.append(f"Ceiling height: {ceiling_height_mm}mm")

        return RoomFinishes(
            room_id=room_id,
            room_label=room_label,
            room_category=room_category,
            floor_area_sqm=round(floor_area, 2),
            carpet_area_sqm=round(carpet_area, 2),
            perimeter_m=round(room_perimeter, 2),
            skirting_length_m=round(skirting_length, 2),
            wall_area_sqm=round(wall_area, 2),
            ceiling_area_sqm=round(ceiling_area, 2),
            ceiling_height_m=ceiling_height_m,
            tile_height_m=tile_height_m,
            deductions=deductions,
            openings_count=len(openings) if openings else 0,
            is_external=is_external,
            is_wet_area=is_wet_area,
            needs_dado=room_label.lower() in ["kitchen", "utility"],
            assumptions_used=assumptions_used,
        )

    def calculate_all_rooms(
        self,
        rooms: List[Dict],
        openings: Optional[List[Dict]] = None,
        room_openings_map: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[RoomFinishes]:
        """
        Calculate finishes for all rooms.

        Args:
            rooms: List of room dicts
            openings: List of all openings
            room_openings_map: Pre-computed mapping of room_id to openings

        Returns:
            List of RoomFinishes objects
        """
        results = []

        # Build room-openings map if not provided
        if room_openings_map is None and openings:
            room_openings_map = self._build_room_openings_map(openings)

        for room in rooms:
            room_id = room.get("room_id", room.get("id", ""))

            # Get openings for this room
            room_openings = []
            if room_openings_map and room_id in room_openings_map:
                room_openings = room_openings_map[room_id]

            finishes = self.calculate_room_finishes(room, room_openings)
            results.append(finishes)

        return results

    def _build_room_openings_map(
        self,
        openings: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Build mapping of room_id to openings."""
        room_map: Dict[str, List[Dict]] = {}

        for opening in openings:
            # Add to left room
            left_id = opening.get("room_left_id")
            if left_id:
                if left_id not in room_map:
                    room_map[left_id] = []
                room_map[left_id].append(opening)

            # Add to right room (for doors)
            right_id = opening.get("room_right_id")
            if right_id and "door" in opening.get("type", "").lower():
                if right_id not in room_map:
                    room_map[right_id] = []
                room_map[right_id].append(opening)

        return room_map

    def calculate_wall_lengths(
        self,
        rooms: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate wall centerline lengths by room.

        Args:
            rooms: List of room dicts with perimeter_m

        Returns:
            Dict mapping room_id to wall length in meters
        """
        wall_lengths = {}

        for room in rooms:
            room_id = room.get("room_id", room.get("id", ""))
            perimeter = room.get("perimeter_m", 0)

            # Wall centerline ≈ perimeter / 2 (for internal walls shared)
            # For simplicity, using full perimeter here
            wall_lengths[room_id] = round(perimeter, 2)

        return wall_lengths

    def get_totals(
        self,
        room_finishes: List[RoomFinishes],
    ) -> Dict[str, float]:
        """
        Calculate total quantities.

        Args:
            room_finishes: List of RoomFinishes objects

        Returns:
            Dict with total quantities
        """
        # Filter out shafts
        valid = [r for r in room_finishes if r.room_category != "shaft"]

        # Separate internal and external
        internal = [r for r in valid if not r.is_external]
        external = [r for r in valid if r.is_external]

        return {
            "total_floor_area_sqm": sum(r.floor_area_sqm for r in valid),
            "total_internal_floor_sqm": sum(r.floor_area_sqm for r in internal),
            "total_external_floor_sqm": sum(r.floor_area_sqm for r in external),
            "total_skirting_length_m": sum(r.skirting_length_m for r in internal),
            "total_wall_area_sqm": sum(r.wall_area_sqm for r in valid),
            "total_ceiling_area_sqm": sum(r.ceiling_area_sqm for r in valid),
            "total_wet_area_sqm": sum(r.floor_area_sqm for r in valid if r.is_wet_area),
            "room_count": len(valid),
        }

    def export_assumptions(self) -> Dict:
        """Export all assumptions used during calculations."""
        return {
            "defaults": {
                "ceiling_height_mm": self.ceiling_height_mm,
                "skirting_height_mm": self.skirting_height_mm,
                "bathroom_tile_height_mm": self.DEFAULT_BATHROOM_TILE_HEIGHT_MM,
                "door_height_mm": self.DEFAULT_DOOR_HEIGHT_MM,
                "window_height_mm": self.DEFAULT_WINDOW_HEIGHT_MM,
            },
            "room_ceiling_heights": self.room_ceiling_heights,
            "calculation_log": self.assumptions_log,
        }
