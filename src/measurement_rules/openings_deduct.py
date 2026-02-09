"""
Openings Deduction Engine

Uses openings_schedule.csv to deduct areas from:
- Plaster (internal and external)
- Paint
- Masonry

Follows IS 1200 method of measurement rules.
"""

import csv
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Opening:
    """Represents a door or window opening."""
    tag: str
    opening_type: str  # door, window, ventilator
    width_mm: float
    height_mm: float
    sill_height_mm: float = 0  # For windows
    location: str = ""
    room: str = ""
    count: int = 1
    frame_type: str = ""  # UPVC, wooden, steel, aluminium

    @property
    def area_sqm(self) -> float:
        """Calculate opening area in sqm."""
        return (self.width_mm / 1000) * (self.height_mm / 1000)

    @property
    def total_area_sqm(self) -> float:
        """Total area including count."""
        return self.area_sqm * self.count


class OpeningsDeductor:
    """
    Deducts opening areas from plaster, paint, and masonry BOQ items.

    Loads openings from:
    1. openings_schedule.csv (if exists)
    2. Passed opening list
    3. Detected openings from image processing
    """

    def __init__(self):
        self.openings: List[Opening] = []
        self.deductions_applied: List[Dict] = []

    def load_from_csv(self, csv_path: Path) -> int:
        """
        Load openings from schedule CSV.

        Expected columns:
        - tag/id: Opening identifier (D1, W1, etc.)
        - type: door/window/ventilator
        - width_mm or width: Width in mm
        - height_mm or height: Height in mm
        - sill_height_mm (optional): Sill height for windows
        - location/room: Location in building
        - count/qty: Number of openings of this type
        - frame_type: Frame material
        """
        if not csv_path.exists():
            logger.warning(f"Openings schedule not found: {csv_path}")
            return 0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Normalize column names
                tag = row.get("tag") or row.get("id") or row.get("Tag") or row.get("ID", "")
                opening_type = row.get("type") or row.get("Type", "door")

                width = float(row.get("width_mm") or row.get("width") or row.get("Width", 0))
                height = float(row.get("height_mm") or row.get("height") or row.get("Height", 0))
                sill = float(row.get("sill_height_mm") or row.get("sill_height") or row.get("Sill", 0))

                location = row.get("location") or row.get("room") or row.get("Location", "")
                count = int(row.get("count") or row.get("qty") or row.get("Count", 1))
                frame = row.get("frame_type") or row.get("frame") or row.get("Frame", "")

                if width > 0 and height > 0:
                    self.openings.append(Opening(
                        tag=tag,
                        opening_type=opening_type.lower(),
                        width_mm=width,
                        height_mm=height,
                        sill_height_mm=sill,
                        location=location,
                        count=count,
                        frame_type=frame,
                    ))

        logger.info(f"Loaded {len(self.openings)} openings from {csv_path}")
        return len(self.openings)

    def load_from_list(self, openings: List[Dict]) -> int:
        """Load openings from a list of dictionaries."""
        for o in openings:
            self.openings.append(Opening(
                tag=o.get("tag") or o.get("id", f"O{len(self.openings)+1}"),
                opening_type=o.get("type", "door").lower(),
                width_mm=float(o.get("width_mm") or o.get("width", 0)),
                height_mm=float(o.get("height_mm") or o.get("height", 0)),
                sill_height_mm=float(o.get("sill_height_mm") or o.get("sill_height", 0)),
                location=o.get("location") or o.get("room", ""),
                count=int(o.get("count") or o.get("qty", 1)),
                frame_type=o.get("frame_type") or o.get("frame", ""),
            ))

        logger.info(f"Loaded {len(openings)} openings from list")
        return len(openings)

    def calculate_total_deductions(
        self,
        item_type: str,
        location_filter: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate total deductions for an item type.

        Args:
            item_type: plaster_internal, plaster_external, paint, masonry
            location_filter: Filter by location (internal, external, etc.)

        Returns:
            Dict with total_area, door_area, window_area
        """
        # IS 1200 thresholds
        thresholds = {
            "plaster_internal": 0.5,
            "plaster_external": 0.5,
            "paint": 0.5,
            "masonry": 0.1,
        }

        threshold = thresholds.get(item_type, 0.5)

        door_area = 0.0
        window_area = 0.0
        ventilator_area = 0.0

        for opening in self.openings:
            # Apply location filter
            if location_filter:
                if location_filter == "internal" and "external" in opening.location.lower():
                    continue
                if location_filter == "external" and "external" not in opening.location.lower():
                    continue

            # Check threshold
            if opening.area_sqm <= threshold:
                continue

            if "door" in opening.opening_type:
                door_area += opening.total_area_sqm
            elif "window" in opening.opening_type:
                window_area += opening.total_area_sqm
            elif "ventilator" in opening.opening_type:
                ventilator_area += opening.total_area_sqm

        return {
            "total_area": door_area + window_area + ventilator_area,
            "door_area": door_area,
            "window_area": window_area,
            "ventilator_area": ventilator_area,
            "threshold_applied": threshold,
            "item_type": item_type,
        }

    def apply_to_boq(
        self,
        boq_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Apply opening deductions to BOQ items.

        Returns updated BOQ with deductions applied.
        """
        self.deductions_applied = []
        adjusted_boq = []

        for item in boq_items:
            item_type = self._classify_item(item)

            if item_type in ["plaster_internal", "plaster_external", "paint", "masonry"]:
                adjusted_item = self._apply_deduction(item, item_type)
                adjusted_boq.append(adjusted_item)
            else:
                adjusted_boq.append(item.copy())

        return adjusted_boq

    def _classify_item(self, item: Dict[str, Any]) -> str:
        """Classify BOQ item type."""
        desc = item.get("description", "").lower()

        if "plaster" in desc:
            if "external" in desc or "outer" in desc:
                return "plaster_external"
            return "plaster_internal"
        elif "paint" in desc or "distemper" in desc or "emulsion" in desc:
            return "paint"
        elif "masonry" in desc or "brick" in desc or "block" in desc:
            return "masonry"

        return "other"

    def _apply_deduction(
        self,
        item: Dict[str, Any],
        item_type: str,
    ) -> Dict[str, Any]:
        """Apply opening deduction to a single item."""
        adjusted = item.copy()
        gross_qty = item.get("quantity", 0)

        # Get location filter from item
        location_filter = None
        if item_type == "plaster_external":
            location_filter = "external"
        elif item_type == "plaster_internal":
            location_filter = "internal"

        # Calculate deductions
        deductions = self.calculate_total_deductions(item_type, location_filter)
        total_deduction = deductions["total_area"]

        # For masonry, need to convert to volume if unit is cum
        unit = item.get("unit", "sqm").lower()
        if item_type == "masonry" and unit in ["cum", "m3"]:
            # Get wall thickness from description
            thickness_m = self._extract_wall_thickness(item) / 1000
            total_deduction = total_deduction * thickness_m

        # Apply deduction
        adjusted["gross_quantity"] = gross_qty
        adjusted["opening_deduction"] = round(total_deduction, 2)
        adjusted["quantity"] = round(max(0, gross_qty - total_deduction), 2)
        adjusted["amount"] = adjusted["quantity"] * item.get("rate", 0)
        adjusted["deduction_details"] = {
            "door_area": deductions["door_area"],
            "window_area": deductions["window_area"],
            "threshold": deductions["threshold_applied"],
            "rule": f"IS 1200: Deduct openings > {deductions['threshold_applied']} sqm",
        }

        # Log deduction
        if total_deduction > 0:
            self.deductions_applied.append({
                "item_id": item.get("item_id", ""),
                "item_description": item.get("description", ""),
                "gross_qty": gross_qty,
                "deduction": total_deduction,
                "net_qty": adjusted["quantity"],
                "unit": unit,
                "door_deduct": deductions["door_area"],
                "window_deduct": deductions["window_area"],
                "rule": deductions["threshold_applied"],
            })

        return adjusted

    def _extract_wall_thickness(self, item: Dict[str, Any]) -> float:
        """Extract wall thickness from description."""
        desc = item.get("description", "").lower()

        if "230" in desc or "9 inch" in desc:
            return 230
        elif "200" in desc:
            return 200
        elif "150" in desc:
            return 150
        elif "115" in desc:
            return 115
        elif "100" in desc:
            return 100

        return 200  # Default

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of openings and deductions."""
        doors = [o for o in self.openings if "door" in o.opening_type]
        windows = [o for o in self.openings if "window" in o.opening_type]

        return {
            "total_openings": len(self.openings),
            "doors": len(doors),
            "windows": len(windows),
            "total_door_area": sum(o.total_area_sqm for o in doors),
            "total_window_area": sum(o.total_area_sqm for o in windows),
            "deductions_applied": len(self.deductions_applied),
            "total_deducted_area": sum(d["deduction"] for d in self.deductions_applied),
        }

    def export_openings_csv(self, output_path: Path) -> None:
        """Export openings list to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tag", "type", "width_mm", "height_mm", "sill_height_mm",
                "area_sqm", "count", "total_area_sqm", "location", "frame_type"
            ])

            for o in self.openings:
                writer.writerow([
                    o.tag, o.opening_type, o.width_mm, o.height_mm,
                    o.sill_height_mm, round(o.area_sqm, 2), o.count,
                    round(o.total_area_sqm, 2), o.location, o.frame_type
                ])

        logger.info(f"Exported {len(self.openings)} openings to {output_path}")

    def export_deductions_csv(self, output_path: Path) -> None:
        """Export deductions log to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "item_id", "item_description", "gross_qty", "deduction",
                "net_qty", "unit", "door_deduct", "window_deduct", "rule"
            ])

            for d in self.deductions_applied:
                writer.writerow([
                    d["item_id"], d["item_description"],
                    f"{d['gross_qty']:.2f}", f"{d['deduction']:.2f}",
                    f"{d['net_qty']:.2f}", d["unit"],
                    f"{d['door_deduct']:.2f}", f"{d['window_deduct']:.2f}",
                    d["rule"]
                ])

        logger.info(f"Exported {len(self.deductions_applied)} deductions to {output_path}")
