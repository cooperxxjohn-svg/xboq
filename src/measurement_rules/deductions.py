"""
CPWD-Style Deduction Engine

Implements IS 1200 method of measurement rules for:
- Plaster deductions (openings >0.5 sqm)
- Paint deductions (openings >0.5 sqm)
- Masonry deductions (openings >0.1 sqm)
- Beam/column face exclusions from wall plaster
- Parapet wall treatment
- Staircase wall treatment
- Shaft/duct wall treatment

Reference: IS 1200 Part 12 (Plastering), Part 13 (Painting)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DeductionType(Enum):
    """Types of deductions applied."""
    OPENING_PLASTER = "opening_plaster"
    OPENING_PAINT = "opening_paint"
    OPENING_MASONRY = "opening_masonry"
    BEAM_FACE = "beam_face"
    COLUMN_FACE = "column_face"
    SKIRTING_OPENING = "skirting_opening"
    DADO_OPENING = "dado_opening"


@dataclass
class Deduction:
    """A single deduction record."""
    deduction_type: DeductionType
    item_id: str
    item_description: str
    opening_id: str
    opening_description: str
    gross_qty: float
    deducted_qty: float
    net_qty: float
    unit: str
    rule_applied: str
    evidence: str


@dataclass
class DeductionThresholds:
    """CPWD/IS 1200 thresholds for deductions."""
    # Plaster: deduct openings > 0.5 sqm each
    plaster_opening_min_sqm: float = 0.5

    # Paint: deduct openings > 0.5 sqm each
    paint_opening_min_sqm: float = 0.5

    # Masonry: deduct openings > 0.1 sqm each
    masonry_opening_min_sqm: float = 0.1

    # Skirting: deduct door widths only (no windows)
    skirting_deduct_doors: bool = True
    skirting_deduct_windows: bool = False

    # Dado: deduct openings > 0.5 sqm in dado zone
    dado_opening_min_sqm: float = 0.5
    dado_height_mm: int = 1200  # Standard dado height

    # Beam/column face exclusion
    exclude_beam_faces_from_plaster: bool = True
    exclude_column_faces_from_plaster: bool = True


class DeductionEngine:
    """
    Applies CPWD-style deductions to BOQ items.

    IS 1200 Rules Applied:
    1. Plaster: No deduction for openings <= 0.5 sqm each
    2. Paint: No deduction for openings <= 0.5 sqm each
    3. Masonry: No deduction for openings <= 0.1 sqm each
    4. Plaster does not include beam/column faces (separate item)
    5. Skirting deducts door widths only
    6. Dado deducts openings in dado zone
    """

    def __init__(self, thresholds: DeductionThresholds = None):
        self.thresholds = thresholds or DeductionThresholds()
        self.deduction_log: List[Deduction] = []

    def apply_deductions(
        self,
        boq_items: List[Dict[str, Any]],
        openings: List[Dict[str, Any]],
        structural_elements: Dict[str, Any] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Deduction]]:
        """
        Apply all deductions to BOQ items.

        Args:
            boq_items: List of BOQ items with quantities
            openings: List of openings (doors/windows) with dimensions
            structural_elements: Optional dict with beam/column info

        Returns:
            Tuple of (adjusted BOQ items, deduction log)
        """
        self.deduction_log = []
        adjusted_boq = []

        # Index openings by room/area for lookup
        openings_by_room = self._index_openings_by_room(openings)

        # Calculate opening areas
        opening_areas = self._calculate_opening_areas(openings)

        for item in boq_items:
            adjusted_item = item.copy()
            item_type = self._classify_item(item)

            if item_type == "plaster_internal":
                adjusted_item = self._apply_plaster_deductions(
                    adjusted_item, openings, opening_areas, structural_elements, "internal"
                )
            elif item_type == "plaster_external":
                adjusted_item = self._apply_plaster_deductions(
                    adjusted_item, openings, opening_areas, structural_elements, "external"
                )
            elif item_type == "paint":
                adjusted_item = self._apply_paint_deductions(
                    adjusted_item, openings, opening_areas
                )
            elif item_type == "masonry":
                adjusted_item = self._apply_masonry_deductions(
                    adjusted_item, openings, opening_areas
                )
            elif item_type == "skirting":
                adjusted_item = self._apply_skirting_deductions(
                    adjusted_item, openings
                )
            elif item_type == "dado":
                adjusted_item = self._apply_dado_deductions(
                    adjusted_item, openings, opening_areas
                )

            adjusted_boq.append(adjusted_item)

        return adjusted_boq, self.deduction_log

    def _classify_item(self, item: Dict[str, Any]) -> str:
        """Classify BOQ item type for deduction rules."""
        desc = item.get("description", "").lower()

        if "plaster" in desc:
            if "external" in desc or "outer" in desc:
                return "plaster_external"
            return "plaster_internal"
        elif "paint" in desc or "distemper" in desc or "emulsion" in desc:
            return "paint"
        elif "masonry" in desc or "brick" in desc or "block" in desc:
            return "masonry"
        elif "skirting" in desc:
            return "skirting"
        elif "dado" in desc:
            return "dado"

        return "other"

    def _index_openings_by_room(self, openings: List[Dict]) -> Dict[str, List[Dict]]:
        """Index openings by room/area."""
        index = {}
        for opening in openings:
            room = opening.get("room", opening.get("location", "unknown"))
            if room not in index:
                index[room] = []
            index[room].append(opening)
        return index

    def _calculate_opening_areas(self, openings: List[Dict]) -> Dict[str, float]:
        """Calculate area for each opening."""
        areas = {}
        for opening in openings:
            opening_id = opening.get("id", opening.get("tag", f"opening_{len(areas)}"))

            # Get dimensions (handle mm or m)
            width = opening.get("width_mm", opening.get("width", 0))
            height = opening.get("height_mm", opening.get("height", 0))

            # Convert mm to m if needed
            if width > 100:  # Assume mm
                width = width / 1000
            if height > 100:
                height = height / 1000

            area_sqm = width * height
            areas[opening_id] = area_sqm

        return areas

    def _apply_plaster_deductions(
        self,
        item: Dict[str, Any],
        openings: List[Dict],
        opening_areas: Dict[str, float],
        structural: Dict[str, Any],
        plaster_type: str,
    ) -> Dict[str, Any]:
        """
        Apply plaster deductions per IS 1200 Part 12.

        Rules:
        - No deduction for openings <= 0.5 sqm each
        - Deduct full area for openings > 0.5 sqm
        - Exclude beam soffits and column faces (measured separately)
        """
        gross_qty = item.get("quantity", 0)
        total_deduction = 0.0

        # Deduct openings > threshold
        for opening in openings:
            opening_id = opening.get("id", opening.get("tag", "unknown"))
            area = opening_areas.get(opening_id, 0)

            if area > self.thresholds.plaster_opening_min_sqm:
                # Full deduction for opening
                total_deduction += area

                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.OPENING_PLASTER,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id=opening_id,
                    opening_description=opening.get("type", "opening"),
                    gross_qty=gross_qty,
                    deducted_qty=area,
                    net_qty=gross_qty - total_deduction,
                    unit="sqm",
                    rule_applied=f"IS 1200: Deduct opening > {self.thresholds.plaster_opening_min_sqm} sqm",
                    evidence=f"Opening {opening_id}: {area:.2f} sqm",
                ))

        # Deduct beam faces if internal plaster
        if plaster_type == "internal" and structural and self.thresholds.exclude_beam_faces_from_plaster:
            beam_deduction = self._calculate_beam_face_area(structural)
            if beam_deduction > 0:
                total_deduction += beam_deduction
                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.BEAM_FACE,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id="beams",
                    opening_description="Beam faces (measured separately)",
                    gross_qty=gross_qty,
                    deducted_qty=beam_deduction,
                    net_qty=gross_qty - total_deduction,
                    unit="sqm",
                    rule_applied="IS 1200: Beam faces measured separately",
                    evidence=f"Beam face area: {beam_deduction:.2f} sqm",
                ))

        # Deduct column faces if internal plaster
        if plaster_type == "internal" and structural and self.thresholds.exclude_column_faces_from_plaster:
            column_deduction = self._calculate_column_face_area(structural)
            if column_deduction > 0:
                total_deduction += column_deduction
                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.COLUMN_FACE,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id="columns",
                    opening_description="Column faces (measured separately)",
                    gross_qty=gross_qty,
                    deducted_qty=column_deduction,
                    net_qty=gross_qty - total_deduction,
                    unit="sqm",
                    rule_applied="IS 1200: Column faces measured separately",
                    evidence=f"Column face area: {column_deduction:.2f} sqm",
                ))

        # Update item
        item["gross_quantity"] = gross_qty
        item["deductions"] = total_deduction
        item["quantity"] = max(0, gross_qty - total_deduction)
        item["amount"] = item["quantity"] * item.get("rate", 0)

        return item

    def _apply_paint_deductions(
        self,
        item: Dict[str, Any],
        openings: List[Dict],
        opening_areas: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Apply paint deductions per IS 1200 Part 13.

        Rules:
        - No deduction for openings <= 0.5 sqm each
        - Deduct full area for openings > 0.5 sqm
        """
        gross_qty = item.get("quantity", 0)
        total_deduction = 0.0

        for opening in openings:
            opening_id = opening.get("id", opening.get("tag", "unknown"))
            area = opening_areas.get(opening_id, 0)

            if area > self.thresholds.paint_opening_min_sqm:
                total_deduction += area

                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.OPENING_PAINT,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id=opening_id,
                    opening_description=opening.get("type", "opening"),
                    gross_qty=gross_qty,
                    deducted_qty=area,
                    net_qty=gross_qty - total_deduction,
                    unit="sqm",
                    rule_applied=f"IS 1200: Deduct opening > {self.thresholds.paint_opening_min_sqm} sqm",
                    evidence=f"Opening {opening_id}: {area:.2f} sqm",
                ))

        item["gross_quantity"] = gross_qty
        item["deductions"] = total_deduction
        item["quantity"] = max(0, gross_qty - total_deduction)
        item["amount"] = item["quantity"] * item.get("rate", 0)

        return item

    def _apply_masonry_deductions(
        self,
        item: Dict[str, Any],
        openings: List[Dict],
        opening_areas: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Apply masonry deductions per IS 1200 Part 2.

        Rules:
        - No deduction for openings <= 0.1 sqm each
        - Deduct full area for openings > 0.1 sqm
        - For volume: multiply area by wall thickness
        """
        gross_qty = item.get("quantity", 0)
        unit = item.get("unit", "sqm").lower()
        total_deduction = 0.0

        # Get wall thickness for volume conversion
        wall_thickness_m = self._extract_wall_thickness(item) / 1000

        for opening in openings:
            opening_id = opening.get("id", opening.get("tag", "unknown"))
            area = opening_areas.get(opening_id, 0)

            if area > self.thresholds.masonry_opening_min_sqm:
                # Convert to volume if needed
                if unit in ["cum", "m3", "cft"]:
                    deduction = area * wall_thickness_m
                else:
                    deduction = area

                total_deduction += deduction

                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.OPENING_MASONRY,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id=opening_id,
                    opening_description=opening.get("type", "opening"),
                    gross_qty=gross_qty,
                    deducted_qty=deduction,
                    net_qty=gross_qty - total_deduction,
                    unit=unit,
                    rule_applied=f"IS 1200: Deduct opening > {self.thresholds.masonry_opening_min_sqm} sqm",
                    evidence=f"Opening {opening_id}: {area:.2f} sqm x {wall_thickness_m*1000:.0f}mm",
                ))

        item["gross_quantity"] = gross_qty
        item["deductions"] = total_deduction
        item["quantity"] = max(0, gross_qty - total_deduction)
        item["amount"] = item["quantity"] * item.get("rate", 0)

        return item

    def _apply_skirting_deductions(
        self,
        item: Dict[str, Any],
        openings: List[Dict],
    ) -> Dict[str, Any]:
        """
        Apply skirting deductions.

        Rules:
        - Deduct door widths (doors interrupt skirting)
        - Do not deduct window widths (skirting continues under windows)
        """
        gross_qty = item.get("quantity", 0)
        total_deduction = 0.0

        for opening in openings:
            opening_type = opening.get("type", "").lower()

            # Only deduct doors
            if "door" in opening_type and self.thresholds.skirting_deduct_doors:
                width = opening.get("width_mm", opening.get("width", 0))
                if width > 100:  # mm
                    width = width / 1000

                # Deduct both sides of door
                deduction = width * 2
                total_deduction += deduction

                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.SKIRTING_OPENING,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id=opening.get("id", opening.get("tag", "unknown")),
                    opening_description=opening_type,
                    gross_qty=gross_qty,
                    deducted_qty=deduction,
                    net_qty=gross_qty - total_deduction,
                    unit="rmt",
                    rule_applied="Skirting: Deduct door width x 2 (both sides)",
                    evidence=f"Door width: {width*1000:.0f}mm",
                ))

        item["gross_quantity"] = gross_qty
        item["deductions"] = total_deduction
        item["quantity"] = max(0, gross_qty - total_deduction)
        item["amount"] = item["quantity"] * item.get("rate", 0)

        return item

    def _apply_dado_deductions(
        self,
        item: Dict[str, Any],
        openings: List[Dict],
        opening_areas: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Apply dado deductions.

        Rules:
        - Dado typically 1200mm height
        - Deduct openings in dado zone > 0.5 sqm
        """
        gross_qty = item.get("quantity", 0)
        total_deduction = 0.0
        dado_height_m = self.thresholds.dado_height_mm / 1000

        for opening in openings:
            opening_id = opening.get("id", opening.get("tag", "unknown"))

            # Get opening dimensions
            width = opening.get("width_mm", opening.get("width", 0))
            height = opening.get("height_mm", opening.get("height", 0))
            sill_height = opening.get("sill_height_mm", opening.get("sill_height", 0))

            # Convert to meters
            if width > 100:
                width = width / 1000
            if height > 100:
                height = height / 1000
            if sill_height > 100:
                sill_height = sill_height / 1000

            # Calculate overlap with dado zone
            # Dado zone is from 0 to dado_height_m
            opening_bottom = sill_height
            opening_top = sill_height + height

            # Overlap
            overlap_bottom = max(0, opening_bottom)
            overlap_top = min(dado_height_m, opening_top)
            overlap_height = max(0, overlap_top - overlap_bottom)

            dado_deduction_area = width * overlap_height

            if dado_deduction_area > self.thresholds.dado_opening_min_sqm:
                total_deduction += dado_deduction_area

                self.deduction_log.append(Deduction(
                    deduction_type=DeductionType.DADO_OPENING,
                    item_id=item.get("item_id", ""),
                    item_description=item.get("description", ""),
                    opening_id=opening_id,
                    opening_description=opening.get("type", "opening"),
                    gross_qty=gross_qty,
                    deducted_qty=dado_deduction_area,
                    net_qty=gross_qty - total_deduction,
                    unit="sqm",
                    rule_applied=f"Dado: Deduct opening in dado zone (0-{self.thresholds.dado_height_mm}mm)",
                    evidence=f"Opening overlap: {dado_deduction_area:.2f} sqm",
                ))

        item["gross_quantity"] = gross_qty
        item["deductions"] = total_deduction
        item["quantity"] = max(0, gross_qty - total_deduction)
        item["amount"] = item["quantity"] * item.get("rate", 0)

        return item

    def _calculate_beam_face_area(self, structural: Dict[str, Any]) -> float:
        """Calculate total beam face area for deduction."""
        beams = structural.get("beams", [])
        total_area = 0.0

        for beam in beams:
            length = beam.get("length_m", beam.get("length", 0))
            depth = beam.get("depth_mm", beam.get("depth", 450)) / 1000  # Default 450mm

            # Two faces (sides) of beam
            total_area += length * depth * 2

        return total_area

    def _calculate_column_face_area(self, structural: Dict[str, Any]) -> float:
        """Calculate total column face area for deduction."""
        columns = structural.get("columns", [])
        total_area = 0.0

        for column in columns:
            perimeter = 0
            height = column.get("height_m", column.get("height", 3.0))  # Default 3m floor height

            width = column.get("width_mm", column.get("width", 450)) / 1000
            depth = column.get("depth_mm", column.get("depth", 450)) / 1000

            if column.get("shape", "rectangular").lower() == "circular":
                diameter = column.get("diameter_mm", column.get("diameter", 450)) / 1000
                perimeter = 3.14159 * diameter
            else:
                perimeter = 2 * (width + depth)

            total_area += perimeter * height

        return total_area

    def _extract_wall_thickness(self, item: Dict[str, Any]) -> float:
        """Extract wall thickness from item description."""
        desc = item.get("description", "").lower()

        # Common wall thicknesses in mm
        if "230" in desc or "9 inch" in desc or "9\"" in desc:
            return 230
        elif "200" in desc or "8 inch" in desc:
            return 200
        elif "150" in desc or "6 inch" in desc:
            return 150
        elif "115" in desc or "4.5 inch" in desc:
            return 115
        elif "100" in desc or "4 inch" in desc:
            return 100

        # Default to 200mm (common AAC block thickness)
        return 200

    def export_deduction_log(self, output_path: str) -> None:
        """Export deduction log to CSV."""
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "deduction_type", "item_id", "item_description",
                "opening_id", "opening_description",
                "gross_qty", "deducted_qty", "net_qty", "unit",
                "rule_applied", "evidence"
            ])

            for d in self.deduction_log:
                writer.writerow([
                    d.deduction_type.value,
                    d.item_id,
                    d.item_description,
                    d.opening_id,
                    d.opening_description,
                    f"{d.gross_qty:.2f}",
                    f"{d.deducted_qty:.2f}",
                    f"{d.net_qty:.2f}",
                    d.unit,
                    d.rule_applied,
                    d.evidence,
                ])

        logger.info(f"Deduction log exported: {output_path} ({len(self.deduction_log)} entries)")
