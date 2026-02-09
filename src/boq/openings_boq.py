"""
Openings BOQ Calculator
Door and window schedule for BOQ.

Features:
- Counts by type and size
- Frame and shutter quantities
- Hardware allowances
- Indian standard sizes and materials
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpeningBOQItem:
    """BOQ line item for door/window."""
    item_code: str
    description: str
    qty: float
    unit: str  # nos, sqm, rm
    derived_from: str
    confidence: float
    opening_type: Optional[str] = None  # door, window, ventilator
    size_mm: Optional[Tuple[int, int]] = None  # width x height
    material: Optional[str] = None


@dataclass
class OpeningScheduleEntry:
    """Schedule entry for a door/window type."""
    tag: str
    opening_type: str
    width_mm: int
    height_mm: int
    quantity: int
    material: str
    frame_material: str
    remarks: Optional[str] = None


@dataclass
class OpeningsBOQResult:
    """Complete openings BOQ result."""
    door_schedule: List[OpeningScheduleEntry]
    window_schedule: List[OpeningScheduleEntry]
    boq_items: List[OpeningBOQItem]
    totals: Dict[str, Any]
    assumptions_used: List[str]


class OpeningsBOQCalculator:
    """
    Calculate openings BOQ from detected doors and windows.

    Indian conventions:
    - Main door: 1000x2100mm, wooden frame + flush shutter
    - Internal doors: 900x2100mm
    - Toilet doors: 750x2100mm
    - Windows: 1200x1200mm typical
    - Ventilators: 600x450mm
    """

    # Default materials by opening type
    DOOR_MATERIALS = {
        "main_door": {
            "frame": "teak_wood",
            "shutter": "flush_door_bwp",
            "hardware": "mortise_lock_brass",
        },
        "internal_door": {
            "frame": "sal_wood",
            "shutter": "flush_door_commercial",
            "hardware": "mortise_lock_ss",
        },
        "toilet_door": {
            "frame": "sal_wood",
            "shutter": "pvc_door",
            "hardware": "tower_bolt",
        },
        "sliding_door": {
            "frame": "aluminium",
            "shutter": "aluminium_glass",
            "hardware": "sliding_track",
        },
    }

    WINDOW_MATERIALS = {
        "standard_window": {
            "frame": "aluminium",
            "shutter": "aluminium_glass",
            "mesh": "ss_mosquito_mesh",
        },
        "large_window": {
            "frame": "aluminium",
            "shutter": "aluminium_glass",
            "mesh": "ss_mosquito_mesh",
        },
        "toilet_window": {
            "frame": "aluminium",
            "shutter": "aluminium_glass_frosted",
            "mesh": "none",
        },
        "ventilator": {
            "frame": "aluminium",
            "shutter": "aluminium_louver",
            "mesh": "ss_mosquito_mesh",
        },
    }

    # Frame sizes (width x depth in mm)
    FRAME_SIZES = {
        "teak_wood": (100, 75),
        "sal_wood": (75, 60),
        "aluminium": (50, 40),
        "upvc": (70, 50),
    }

    def __init__(self):
        self.assumptions_used: List[str] = []

    def calculate_from_openings(
        self,
        doors: List[Dict],
        windows: List[Dict],
    ) -> OpeningsBOQResult:
        """
        Calculate openings BOQ from detected openings.

        Args:
            doors: List of door dicts with id, type, tag, width_m, height_m
            windows: List of window dicts

        Returns:
            OpeningsBOQResult with schedules and BOQ items
        """
        door_schedule = self._build_door_schedule(doors)
        window_schedule = self._build_window_schedule(windows)

        boq_items = []

        # Door BOQ items
        boq_items.extend(self._calculate_door_boq(door_schedule))

        # Window BOQ items
        boq_items.extend(self._calculate_window_boq(window_schedule))

        # Hardware items
        boq_items.extend(self._calculate_hardware_boq(door_schedule, window_schedule))

        # Totals
        totals = {
            "total_doors": sum(e.quantity for e in door_schedule),
            "total_windows": sum(e.quantity for e in window_schedule if e.opening_type != "ventilator"),
            "total_ventilators": sum(e.quantity for e in window_schedule if e.opening_type == "ventilator"),
            "total_door_area_sqm": sum(
                e.quantity * (e.width_mm / 1000) * (e.height_mm / 1000)
                for e in door_schedule
            ),
            "total_window_area_sqm": sum(
                e.quantity * (e.width_mm / 1000) * (e.height_mm / 1000)
                for e in window_schedule
            ),
        }

        return OpeningsBOQResult(
            door_schedule=door_schedule,
            window_schedule=window_schedule,
            boq_items=boq_items,
            totals=totals,
            assumptions_used=self.assumptions_used,
        )

    def _build_door_schedule(
        self,
        doors: List[Dict],
    ) -> List[OpeningScheduleEntry]:
        """Build door schedule grouped by type and size."""
        schedule_map: Dict[str, OpeningScheduleEntry] = {}

        for door in doors:
            tag = door.get("tag") or door.get("type", "D")
            door_type = door.get("type", "internal_door")

            # Get size
            width_mm = int((door.get("width_m", 0.9) or 0.9) * 1000)
            height_mm = int((door.get("height_m", 2.1) or 2.1) * 1000)

            # Normalize to standard sizes
            width_mm = self._normalize_size(width_mm, [750, 900, 1000, 1200, 1500, 1800])
            height_mm = self._normalize_size(height_mm, [2100, 2400])

            key = f"{tag}_{width_mm}x{height_mm}"

            # Get materials
            materials = self.DOOR_MATERIALS.get(door_type, self.DOOR_MATERIALS["internal_door"])

            if key in schedule_map:
                schedule_map[key].quantity += 1
            else:
                schedule_map[key] = OpeningScheduleEntry(
                    tag=tag,
                    opening_type=door_type,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    quantity=1,
                    material=materials["shutter"],
                    frame_material=materials["frame"],
                )

        return list(schedule_map.values())

    def _build_window_schedule(
        self,
        windows: List[Dict],
    ) -> List[OpeningScheduleEntry]:
        """Build window schedule grouped by type and size."""
        schedule_map: Dict[str, OpeningScheduleEntry] = {}

        for window in windows:
            tag = window.get("tag") or window.get("type", "W")
            window_type = window.get("type", "standard_window")

            # Get size
            width_mm = int((window.get("width_m", 1.2) or 1.2) * 1000)
            height_mm = int((window.get("height_m", 1.2) or 1.2) * 1000)

            # Normalize
            width_mm = self._normalize_size(width_mm, [600, 900, 1200, 1500, 1800])
            height_mm = self._normalize_size(height_mm, [450, 600, 900, 1200, 1500])

            key = f"{tag}_{width_mm}x{height_mm}"

            # Get materials
            materials = self.WINDOW_MATERIALS.get(window_type, self.WINDOW_MATERIALS["standard_window"])

            if key in schedule_map:
                schedule_map[key].quantity += 1
            else:
                schedule_map[key] = OpeningScheduleEntry(
                    tag=tag,
                    opening_type=window_type,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    quantity=1,
                    material=materials["shutter"],
                    frame_material=materials["frame"],
                )

        return list(schedule_map.values())

    def _normalize_size(self, value: int, standards: List[int]) -> int:
        """Normalize size to nearest standard."""
        min_diff = float('inf')
        nearest = standards[0]

        for std in standards:
            diff = abs(value - std)
            if diff < min_diff:
                min_diff = diff
                nearest = std

        if min_diff > 100:  # More than 100mm difference
            self.assumptions_used.append(f"Size {value}mm normalized to {nearest}mm")

        return nearest

    def _calculate_door_boq(
        self,
        schedule: List[OpeningScheduleEntry],
    ) -> List[OpeningBOQItem]:
        """Calculate BOQ items for doors."""
        items = []

        for entry in schedule:
            # Door frame
            frame_perimeter = 2 * (entry.width_mm + entry.height_mm) / 1000  # meters
            total_frame = frame_perimeter * entry.quantity

            items.append(OpeningBOQItem(
                item_code=f"DRF-{entry.tag}",
                description=f"Door frame {entry.frame_material} ({entry.width_mm}x{entry.height_mm}mm)",
                qty=round(total_frame, 2),
                unit="rm",
                derived_from="opening_detection",
                confidence=0.85,
                opening_type="door_frame",
                size_mm=(entry.width_mm, entry.height_mm),
                material=entry.frame_material,
            ))

            # Door shutter
            shutter_area = (entry.width_mm / 1000) * (entry.height_mm / 1000)
            total_shutter = shutter_area * entry.quantity

            items.append(OpeningBOQItem(
                item_code=f"DRS-{entry.tag}",
                description=f"Door shutter {entry.material} ({entry.width_mm}x{entry.height_mm}mm)",
                qty=entry.quantity,
                unit="nos",
                derived_from="opening_detection",
                confidence=0.85,
                opening_type="door_shutter",
                size_mm=(entry.width_mm, entry.height_mm),
                material=entry.material,
            ))

        return items

    def _calculate_window_boq(
        self,
        schedule: List[OpeningScheduleEntry],
    ) -> List[OpeningBOQItem]:
        """Calculate BOQ items for windows."""
        items = []

        for entry in schedule:
            # Window frame + shutter (usually combined for aluminium)
            window_area = (entry.width_mm / 1000) * (entry.height_mm / 1000)
            total_area = window_area * entry.quantity

            items.append(OpeningBOQItem(
                item_code=f"WND-{entry.tag}",
                description=f"Window {entry.material} ({entry.width_mm}x{entry.height_mm}mm)",
                qty=round(total_area, 2),
                unit="sqm",
                derived_from="opening_detection",
                confidence=0.85,
                opening_type="window",
                size_mm=(entry.width_mm, entry.height_mm),
                material=entry.material,
            ))

            # Mosquito mesh (if applicable)
            if entry.opening_type not in ["ventilator", "toilet_window"]:
                items.append(OpeningBOQItem(
                    item_code=f"MSH-{entry.tag}",
                    description=f"SS mosquito mesh for window ({entry.width_mm}x{entry.height_mm}mm)",
                    qty=round(total_area, 2),
                    unit="sqm",
                    derived_from="opening_detection",
                    confidence=0.80,
                    opening_type="mesh",
                    material="ss_mesh",
                ))

        return items

    def _calculate_hardware_boq(
        self,
        door_schedule: List[OpeningScheduleEntry],
        window_schedule: List[OpeningScheduleEntry],
    ) -> List[OpeningBOQItem]:
        """Calculate hardware BOQ items."""
        items = []

        total_doors = sum(e.quantity for e in door_schedule)
        total_windows = sum(e.quantity for e in window_schedule)

        # Door hardware
        if total_doors > 0:
            # Hinges (3 per door)
            items.append(OpeningBOQItem(
                item_code="HDW-HNG-01",
                description="SS butt hinges (100mm) for doors",
                qty=total_doors * 3,
                unit="nos",
                derived_from="calculation",
                confidence=0.90,
                opening_type="hardware",
                material="stainless_steel",
            ))

            # Door stoppers
            items.append(OpeningBOQItem(
                item_code="HDW-STP-01",
                description="Door stoppers (floor mounted)",
                qty=total_doors,
                unit="nos",
                derived_from="calculation",
                confidence=0.85,
                opening_type="hardware",
            ))

            # Locks (1 per door, type based on door type)
            main_doors = sum(e.quantity for e in door_schedule if "main" in e.opening_type.lower())
            internal_doors = sum(e.quantity for e in door_schedule if "internal" in e.opening_type.lower())
            toilet_doors = sum(e.quantity for e in door_schedule if "toilet" in e.opening_type.lower())

            if main_doors > 0:
                items.append(OpeningBOQItem(
                    item_code="HDW-LCK-01",
                    description="Mortise lock (brass) for main doors",
                    qty=main_doors,
                    unit="nos",
                    derived_from="calculation",
                    confidence=0.85,
                    opening_type="hardware",
                ))

            if internal_doors > 0:
                items.append(OpeningBOQItem(
                    item_code="HDW-LCK-02",
                    description="Mortise lock (SS) for internal doors",
                    qty=internal_doors,
                    unit="nos",
                    derived_from="calculation",
                    confidence=0.85,
                    opening_type="hardware",
                ))

            if toilet_doors > 0:
                items.append(OpeningBOQItem(
                    item_code="HDW-LCK-03",
                    description="Tower bolt + door closer for toilet doors",
                    qty=toilet_doors,
                    unit="set",
                    derived_from="calculation",
                    confidence=0.85,
                    opening_type="hardware",
                ))

        # Window hardware
        if total_windows > 0:
            # Window stays
            items.append(OpeningBOQItem(
                item_code="HDW-WST-01",
                description="Window stays (SS) for windows",
                qty=total_windows * 2,
                unit="nos",
                derived_from="calculation",
                confidence=0.80,
                opening_type="hardware",
            ))

            # Tower bolts
            items.append(OpeningBOQItem(
                item_code="HDW-TBT-01",
                description="Tower bolts for windows",
                qty=total_windows,
                unit="nos",
                derived_from="calculation",
                confidence=0.80,
                opening_type="hardware",
            ))

        return items


def create_openings_schedule_csv(
    result: OpeningsBOQResult,
) -> str:
    """Generate CSV content for openings schedule."""
    lines = ["Type,Tag,Width (mm),Height (mm),Qty,Material,Frame,Remarks"]

    for entry in result.door_schedule:
        lines.append(
            f"Door,{entry.tag},{entry.width_mm},{entry.height_mm},"
            f"{entry.quantity},{entry.material},{entry.frame_material},{entry.remarks or ''}"
        )

    for entry in result.window_schedule:
        lines.append(
            f"Window,{entry.tag},{entry.width_mm},{entry.height_mm},"
            f"{entry.quantity},{entry.material},{entry.frame_material},{entry.remarks or ''}"
        )

    return "\n".join(lines)
