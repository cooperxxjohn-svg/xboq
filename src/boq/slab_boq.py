"""
Slab BOQ Calculator
RCC slab quantity calculations.

Features:
- Slab area from room union or boundary detection
- Concrete volume calculation
- Formwork (centering/shuttering) area
- Thickness from notes or defaults
- Deductions for shafts, ducts, OTS
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlabBOQItem:
    """BOQ line item for slab work."""
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    slab_type: Optional[str] = None
    thickness_mm: Optional[int] = None
    deductions: Dict[str, float] = field(default_factory=dict)


@dataclass
class SlabBOQResult:
    """Complete slab BOQ result."""
    gross_area_sqm: float
    net_area_sqm: float
    deducted_area_sqm: float
    thickness_mm: int
    concrete_volume_cum: float
    formwork_area_sqm: float
    boq_items: List[SlabBOQItem]
    deduction_breakdown: Dict[str, float]
    assumptions_used: List[str]


class SlabBOQCalculator:
    """
    Calculate slab BOQ from room data or slab boundary.

    Indian RCC slab standards:
    - Residential: 100-150mm thick
    - Spans up to 4m: 125mm
    - Spans 4-5m: 150mm
    - Sunken slab (toilet): 200mm depth
    """

    # Standard slab thicknesses (mm)
    STANDARD_THICKNESSES = {
        100: "Light residential slab",
        125: "Standard residential slab (spans < 4m)",
        150: "Residential slab (spans 4-5m)",
        175: "Commercial/heavy loads",
        200: "Sunken slab / larger spans",
    }

    # Concrete grades
    CONCRETE_GRADES = {
        "M20": "Standard residential",
        "M25": "Recommended for slabs",
        "M30": "Heavy duty / special",
    }

    # Deduction room types
    DEDUCTION_ROOMS = {
        "shaft", "duct", "ots", "open_to_sky", "void",
        "lift", "elevator", "lift_well", "staircase_void",
        "court", "courtyard", "atrium",
    }

    def __init__(
        self,
        default_thickness_mm: int = 125,
        concrete_grade: str = "M25",
    ):
        self.default_thickness_mm = default_thickness_mm
        self.concrete_grade = concrete_grade
        self.assumptions_used: List[str] = []

    def calculate_from_rooms(
        self,
        rooms: List[Dict],
        thickness_mm: Optional[int] = None,
        include_balconies: bool = True,
    ) -> SlabBOQResult:
        """
        Calculate slab BOQ from room data.

        Sums room areas, deducts shafts/ducts/OTS, calculates concrete volume.

        Args:
            rooms: List of room dicts with area_sqm, label
            thickness_mm: Override slab thickness
            include_balconies: Include balcony/terrace in slab

        Returns:
            SlabBOQResult with quantities
        """
        if thickness_mm:
            slab_thickness = thickness_mm
        else:
            slab_thickness = self.default_thickness_mm
            self.assumptions_used.append(f"Slab thickness assumed: {slab_thickness}mm")

        gross_area = 0.0
        deducted_area = 0.0
        deduction_breakdown: Dict[str, float] = {}

        external_rooms = {"balcony", "terrace", "deck", "verandah", "porch", "sit_out"}

        for room in rooms:
            room_label = room.get("label", "").lower()
            room_area = room.get("area_sqm", 0)

            # Check if should be deducted
            should_deduct = False
            for deduct_type in self.DEDUCTION_ROOMS:
                if deduct_type in room_label:
                    should_deduct = True
                    deduction_breakdown[deduct_type] = deduction_breakdown.get(deduct_type, 0) + room_area
                    deducted_area += room_area
                    break

            # Check if external
            is_external = any(ext in room_label for ext in external_rooms)

            if not should_deduct:
                if is_external and not include_balconies:
                    # Don't include external areas
                    deduction_breakdown["external"] = deduction_breakdown.get("external", 0) + room_area
                    deducted_area += room_area
                else:
                    gross_area += room_area

        net_area = gross_area
        thickness_m = slab_thickness / 1000

        # Concrete volume
        concrete_volume = net_area * thickness_m

        # Formwork area (bottom + edge)
        # Edge formwork = perimeter * thickness (approximated)
        # Assume perimeter = 4 * sqrt(area) for rough estimate
        import math
        approx_perimeter = 4 * math.sqrt(net_area)
        edge_formwork = approx_perimeter * thickness_m
        formwork_area = net_area + edge_formwork

        # Build BOQ items
        boq_items = []

        # Concrete
        boq_items.append(SlabBOQItem(
            item_code=f"RCC-SLB-{self.concrete_grade}",
            description=f"Providing and laying RCC {self.concrete_grade} in slab ({slab_thickness}mm thick)",
            qty=round(concrete_volume, 2),
            unit="cum",
            derived_from="slab_area",
            confidence=0.75,
            slab_type="floor_slab",
            thickness_mm=slab_thickness,
            deductions=deduction_breakdown,
        ))

        # Formwork
        boq_items.append(SlabBOQItem(
            item_code="FWK-SLB-01",
            description="Providing centering and shuttering for RCC slab",
            qty=round(formwork_area, 2),
            unit="sqm",
            derived_from="slab_area",
            confidence=0.72,
            slab_type="formwork",
        ))

        # Curing
        boq_items.append(SlabBOQItem(
            item_code="CUR-SLB-01",
            description="Curing of RCC slab for 7 days",
            qty=round(net_area, 2),
            unit="sqm",
            derived_from="slab_area",
            confidence=0.90,
            slab_type="curing",
        ))

        return SlabBOQResult(
            gross_area_sqm=round(gross_area + deducted_area, 2),
            net_area_sqm=round(net_area, 2),
            deducted_area_sqm=round(deducted_area, 2),
            thickness_mm=slab_thickness,
            concrete_volume_cum=round(concrete_volume, 2),
            formwork_area_sqm=round(formwork_area, 2),
            boq_items=boq_items,
            deduction_breakdown=deduction_breakdown,
            assumptions_used=self.assumptions_used,
        )

    def calculate_from_boundary(
        self,
        boundary_mask: np.ndarray,
        room_masks: Optional[Dict[str, np.ndarray]] = None,
        scale_px_per_mm: Optional[float] = None,
        thickness_mm: Optional[int] = None,
    ) -> SlabBOQResult:
        """
        Calculate slab BOQ from boundary mask.

        Args:
            boundary_mask: Binary mask of slab boundary
            room_masks: Dict of room_type -> mask for deductions
            scale_px_per_mm: Scale factor
            thickness_mm: Slab thickness

        Returns:
            SlabBOQResult with quantities
        """
        if thickness_mm:
            slab_thickness = thickness_mm
        else:
            slab_thickness = self.default_thickness_mm
            self.assumptions_used.append(f"Slab thickness assumed: {slab_thickness}mm")

        # Calculate area in pixels
        if scale_px_per_mm and scale_px_per_mm > 0:
            area_px = cv2.countNonZero(boundary_mask)
            area_sqm = (area_px / (scale_px_per_mm ** 2)) / 1_000_000
        else:
            self.assumptions_used.append("Scale assumed for area calculation")
            area_px = cv2.countNonZero(boundary_mask)
            area_sqm = area_px * 0.0001  # Rough estimate

        gross_area = area_sqm
        deducted_area = 0.0
        deduction_breakdown: Dict[str, float] = {}

        # Deduct from room masks
        if room_masks:
            for room_type, mask in room_masks.items():
                if any(d in room_type.lower() for d in self.DEDUCTION_ROOMS):
                    deduct_px = cv2.countNonZero(mask)
                    if scale_px_per_mm and scale_px_per_mm > 0:
                        deduct_sqm = (deduct_px / (scale_px_per_mm ** 2)) / 1_000_000
                    else:
                        deduct_sqm = deduct_px * 0.0001

                    deduction_breakdown[room_type] = deduct_sqm
                    deducted_area += deduct_sqm

        net_area = max(0, gross_area - deducted_area)
        thickness_m = slab_thickness / 1000
        concrete_volume = net_area * thickness_m

        # Formwork
        import math
        approx_perimeter = 4 * math.sqrt(net_area)
        edge_formwork = approx_perimeter * thickness_m
        formwork_area = net_area + edge_formwork

        # BOQ items
        boq_items = [
            SlabBOQItem(
                item_code=f"RCC-SLB-{self.concrete_grade}",
                description=f"Providing and laying RCC {self.concrete_grade} in slab ({slab_thickness}mm thick)",
                qty=round(concrete_volume, 2),
                unit="cum",
                derived_from="boundary_detection",
                confidence=0.70,
                slab_type="floor_slab",
                thickness_mm=slab_thickness,
            ),
            SlabBOQItem(
                item_code="FWK-SLB-01",
                description="Providing centering and shuttering for RCC slab",
                qty=round(formwork_area, 2),
                unit="sqm",
                derived_from="boundary_detection",
                confidence=0.68,
            ),
            SlabBOQItem(
                item_code="CUR-SLB-01",
                description="Curing of RCC slab for 7 days",
                qty=round(net_area, 2),
                unit="sqm",
                derived_from="boundary_detection",
                confidence=0.85,
            ),
        ]

        return SlabBOQResult(
            gross_area_sqm=round(gross_area, 2),
            net_area_sqm=round(net_area, 2),
            deducted_area_sqm=round(deducted_area, 2),
            thickness_mm=slab_thickness,
            concrete_volume_cum=round(concrete_volume, 2),
            formwork_area_sqm=round(formwork_area, 2),
            boq_items=boq_items,
            deduction_breakdown=deduction_breakdown,
            assumptions_used=self.assumptions_used,
        )

    def calculate_sunken_slab(
        self,
        wet_area_sqm: float,
        sunken_depth_mm: int = 200,
    ) -> List[SlabBOQItem]:
        """
        Calculate additional quantities for sunken slab in wet areas.

        Args:
            wet_area_sqm: Total wet area (toilets, bathrooms)
            sunken_depth_mm: Sunken depth (typically 200mm)

        Returns:
            List of BOQ items for sunken slab work
        """
        items = []

        # Additional concrete for sunken portion
        depth_m = sunken_depth_mm / 1000
        additional_concrete = wet_area_sqm * depth_m * 0.5  # Approximate fill

        items.append(SlabBOQItem(
            item_code="RCC-SNK-M25",
            description=f"Additional RCC M25 for sunken slab ({sunken_depth_mm}mm depth)",
            qty=round(additional_concrete, 2),
            unit="cum",
            derived_from="sunken_calculation",
            confidence=0.70,
            slab_type="sunken_slab",
            thickness_mm=sunken_depth_mm,
        ))

        # Brick bat coba filling
        brick_coba_volume = wet_area_sqm * depth_m * 0.5

        items.append(SlabBOQItem(
            item_code="BBC-SNK-01",
            description="Brick bat coba filling in sunken portion",
            qty=round(brick_coba_volume, 2),
            unit="cum",
            derived_from="sunken_calculation",
            confidence=0.65,
            slab_type="filling",
        ))

        return items

    def detect_slab_boundary(
        self,
        wall_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Detect slab boundary from wall mask.

        Fills the area enclosed by walls to get slab extent.
        """
        # Invert wall mask
        inverted = cv2.bitwise_not(wall_mask)

        # Flood fill from edges to mark exterior
        h, w = inverted.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        # Fill from corners
        cv2.floodFill(inverted, flood_mask, (0, 0), 128)
        cv2.floodFill(inverted, flood_mask, (w - 1, 0), 128)
        cv2.floodFill(inverted, flood_mask, (0, h - 1), 128)
        cv2.floodFill(inverted, flood_mask, (w - 1, h - 1), 128)

        # Interior is what's not filled (still 255)
        interior = (inverted == 255).astype(np.uint8) * 255

        return interior
