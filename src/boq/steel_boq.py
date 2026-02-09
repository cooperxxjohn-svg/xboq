"""
Steel BOQ Calculator
Reinforcement steel estimation for RCC work.

Methods:
1. BBS/Schedule parsing (if available) - exact quantities
2. Rule of thumb (kg/m³) - estimation when BBS not available

Indian standards:
- TMT Fe500/Fe500D bars
- Standard diameters: 8, 10, 12, 16, 20, 25, 32mm
- Typical steel content: 80-200 kg/m³ depending on element
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class SteelBOQItem:
    """BOQ line item for reinforcement steel."""
    item_code: str
    description: str
    qty: float
    unit: str  # kg or MT
    derived_from: str  # bbs_schedule, rule_of_thumb
    confidence: float
    element_type: Optional[str] = None  # slab, beam, column, footing
    bar_diameter: Optional[int] = None  # mm
    steel_grade: str = "Fe500"
    assumptions: List[str] = field(default_factory=list)


@dataclass
class SteelBOQResult:
    """Complete steel BOQ result."""
    total_steel_kg: float
    total_steel_mt: float
    by_element: Dict[str, float]  # element_type -> kg
    by_diameter: Dict[int, float]  # diameter_mm -> kg
    boq_items: List[SteelBOQItem]
    estimation_method: str
    assumptions_used: List[str]


class SteelBOQCalculator:
    """
    Calculate reinforcement steel quantities.

    Uses BBS schedule when available, otherwise applies
    rule-of-thumb factors based on Indian construction practice.
    """

    # Steel content factors (kg per cubic meter of concrete)
    # Based on typical Indian residential construction
    STEEL_FACTORS = {
        "slab": {
            "min": 80,
            "max": 120,
            "typical": 90,
            "description": "Residential slab (100-150mm thick)",
        },
        "beam": {
            "min": 120,
            "max": 180,
            "typical": 150,
            "description": "RCC beam (typical spans)",
        },
        "column": {
            "min": 150,
            "max": 250,
            "typical": 200,
            "description": "RCC column (2-4% reinforcement)",
        },
        "footing": {
            "min": 70,
            "max": 120,
            "typical": 90,
            "description": "Isolated/combined footing",
        },
        "plinth_beam": {
            "min": 100,
            "max": 150,
            "typical": 120,
            "description": "Plinth beam / tie beam",
        },
        "staircase": {
            "min": 100,
            "max": 150,
            "typical": 120,
            "description": "Waist slab + steps",
        },
        "lintel": {
            "min": 80,
            "max": 120,
            "typical": 100,
            "description": "Lintel beam above openings",
        },
        "chajja": {
            "min": 90,
            "max": 130,
            "typical": 110,
            "description": "Sunshade / chajja",
        },
    }

    # Bar unit weights (kg per meter)
    BAR_WEIGHTS = {
        6: 0.222,
        8: 0.395,
        10: 0.617,
        12: 0.889,
        16: 1.580,
        20: 2.469,
        25: 3.858,
        32: 6.321,
    }

    # Typical bar distribution by element
    BAR_DISTRIBUTION = {
        "slab": {8: 0.30, 10: 0.50, 12: 0.20},
        "beam": {8: 0.15, 12: 0.25, 16: 0.40, 20: 0.20},
        "column": {8: 0.20, 16: 0.30, 20: 0.30, 25: 0.20},
        "footing": {10: 0.30, 12: 0.50, 16: 0.20},
    }

    def __init__(
        self,
        steel_grade: str = "Fe500",
        wastage_factor: float = 0.05,  # 5% cutting/lapping wastage
    ):
        self.steel_grade = steel_grade
        self.wastage_factor = wastage_factor
        self.assumptions_used: List[str] = []

    def calculate_from_bbs(
        self,
        bbs_data: List[Dict],
    ) -> SteelBOQResult:
        """
        Calculate steel quantities from Bar Bending Schedule.

        Args:
            bbs_data: List of BBS entries with:
                - bar_mark: Bar identification
                - diameter_mm: Bar diameter
                - length_m: Cut length
                - quantity: Number of bars
                - element: Element type (optional)

        Returns:
            SteelBOQResult with exact quantities
        """
        by_element: Dict[str, float] = {}
        by_diameter: Dict[int, float] = {}
        boq_items = []

        total_weight = 0.0

        for entry in bbs_data:
            diameter = entry.get("diameter_mm", 12)
            length = entry.get("length_m", 0)
            quantity = entry.get("quantity", 0)
            element = entry.get("element", "general")

            unit_weight = self.BAR_WEIGHTS.get(diameter, 0.889)  # Default 12mm
            weight = quantity * length * unit_weight

            total_weight += weight
            by_element[element] = by_element.get(element, 0) + weight
            by_diameter[diameter] = by_diameter.get(diameter, 0) + weight

        # Apply wastage
        total_with_wastage = total_weight * (1 + self.wastage_factor)

        # Create BOQ items by diameter
        for diameter, weight in by_diameter.items():
            weight_with_wastage = weight * (1 + self.wastage_factor)
            boq_items.append(SteelBOQItem(
                item_code=f"STL-{diameter:02d}",
                description=f"TMT {self.steel_grade} bars - {diameter}mm diameter",
                qty=round(weight_with_wastage, 1),
                unit="kg",
                derived_from="bbs_schedule",
                confidence=0.95,
                bar_diameter=diameter,
                steel_grade=self.steel_grade,
            ))

        # Summary item
        boq_items.append(SteelBOQItem(
            item_code="STL-TOT",
            description=f"Total TMT {self.steel_grade} reinforcement steel",
            qty=round(total_with_wastage / 1000, 2),
            unit="MT",
            derived_from="bbs_schedule",
            confidence=0.95,
            steel_grade=self.steel_grade,
        ))

        return SteelBOQResult(
            total_steel_kg=round(total_with_wastage, 1),
            total_steel_mt=round(total_with_wastage / 1000, 2),
            by_element=by_element,
            by_diameter=by_diameter,
            boq_items=boq_items,
            estimation_method="bbs_schedule",
            assumptions_used=[f"Wastage factor: {self.wastage_factor * 100}%"],
        )

    def calculate_from_concrete_volumes(
        self,
        concrete_volumes: Dict[str, float],
        factor_type: str = "typical",
    ) -> SteelBOQResult:
        """
        Calculate steel using rule-of-thumb factors.

        Args:
            concrete_volumes: Dict of element_type -> volume_cum
            factor_type: "min", "max", or "typical"

        Returns:
            SteelBOQResult with estimated quantities
        """
        self.assumptions_used.append(f"Steel estimated using {factor_type} kg/m³ factors")

        by_element: Dict[str, float] = {}
        by_diameter: Dict[int, float] = {}
        boq_items = []

        total_weight = 0.0

        for element_type, volume in concrete_volumes.items():
            # Get steel factor
            factors = self.STEEL_FACTORS.get(element_type, self.STEEL_FACTORS["slab"])
            factor = factors.get(factor_type, factors["typical"])

            steel_weight = volume * factor
            total_weight += steel_weight
            by_element[element_type] = steel_weight

            self.assumptions_used.append(
                f"{element_type}: {factor} kg/m³ × {volume:.2f} m³ = {steel_weight:.1f} kg"
            )

            # Distribute by diameter based on element type
            distribution = self.BAR_DISTRIBUTION.get(element_type, {12: 1.0})
            for diameter, proportion in distribution.items():
                dia_weight = steel_weight * proportion
                by_diameter[diameter] = by_diameter.get(diameter, 0) + dia_weight

            # Add element-level BOQ item
            boq_items.append(SteelBOQItem(
                item_code=f"STL-{element_type[:3].upper()}",
                description=f"TMT {self.steel_grade} reinforcement for {element_type}",
                qty=round(steel_weight, 1),
                unit="kg",
                derived_from="rule_of_thumb",
                confidence=0.60,
                element_type=element_type,
                steel_grade=self.steel_grade,
                assumptions=[f"Steel factor: {factor} kg/m³"],
            ))

        # Apply wastage
        total_with_wastage = total_weight * (1 + self.wastage_factor)

        # Summary items by diameter
        for diameter, weight in by_diameter.items():
            weight_with_wastage = weight * (1 + self.wastage_factor)
            boq_items.append(SteelBOQItem(
                item_code=f"STL-{diameter:02d}",
                description=f"TMT {self.steel_grade} bars - {diameter}mm diameter (estimated)",
                qty=round(weight_with_wastage, 1),
                unit="kg",
                derived_from="rule_of_thumb",
                confidence=0.55,
                bar_diameter=diameter,
                steel_grade=self.steel_grade,
            ))

        # Total summary
        boq_items.append(SteelBOQItem(
            item_code="STL-TOT",
            description=f"Total TMT {self.steel_grade} reinforcement steel (estimated)",
            qty=round(total_with_wastage / 1000, 2),
            unit="MT",
            derived_from="rule_of_thumb",
            confidence=0.60,
            steel_grade=self.steel_grade,
        ))

        return SteelBOQResult(
            total_steel_kg=round(total_with_wastage, 1),
            total_steel_mt=round(total_with_wastage / 1000, 2),
            by_element=by_element,
            by_diameter=by_diameter,
            boq_items=boq_items,
            estimation_method="rule_of_thumb",
            assumptions_used=self.assumptions_used,
        )

    def calculate_slab_steel(
        self,
        slab_area_sqm: float,
        slab_thickness_mm: int = 125,
        factor_type: str = "typical",
    ) -> SteelBOQResult:
        """
        Quick calculation for slab steel only.

        Args:
            slab_area_sqm: Slab area in square meters
            slab_thickness_mm: Slab thickness
            factor_type: "min", "max", or "typical"

        Returns:
            SteelBOQResult for slab steel
        """
        volume_cum = slab_area_sqm * (slab_thickness_mm / 1000)

        return self.calculate_from_concrete_volumes(
            {"slab": volume_cum},
            factor_type=factor_type,
        )

    def calculate_lintel_steel(
        self,
        openings: List[Dict],
        lintel_depth_mm: int = 150,
        lintel_width_mm: int = 230,
    ) -> SteelBOQResult:
        """
        Calculate steel for lintels above openings.

        Args:
            openings: List of openings with width_m
            lintel_depth_mm: Lintel depth
            lintel_width_mm: Lintel width (wall thickness)

        Returns:
            SteelBOQResult for lintel steel
        """
        total_length = 0.0

        for opening in openings:
            width = opening.get("width_m", 0.9)
            # Lintel length = opening width + 150mm bearing each side
            lintel_length = width + 0.3
            total_length += lintel_length

        # Lintel volume
        volume_cum = (total_length *
                      (lintel_depth_mm / 1000) *
                      (lintel_width_mm / 1000))

        self.assumptions_used.append(f"Lintel size: {lintel_width_mm}x{lintel_depth_mm}mm")
        self.assumptions_used.append(f"Total lintel length: {total_length:.2f}m")

        return self.calculate_from_concrete_volumes(
            {"lintel": volume_cum},
            factor_type="typical",
        )

    def parse_bbs_from_text(
        self,
        text: str,
    ) -> List[Dict]:
        """
        Parse BBS data from text (OCR or extracted).

        Looks for patterns like:
        - Bar mark, diameter, length, quantity
        - Typical BBS table formats

        Returns:
            List of BBS entries
        """
        entries = []

        # Pattern for BBS entries: mark, dia, length, qty
        patterns = [
            r"(\w+)\s+(\d+)\s*[mM][mM]\s+(\d+\.?\d*)\s*[mM]?\s+(\d+)",  # A 12mm 2.5m 10
            r"(\d+)\s*[#φ]\s*(\d+\.?\d*)\s+(\d+)",  # 10#12 2.5 20
        ]

        lines = text.split('\n')
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    try:
                        entry = {
                            "bar_mark": groups[0],
                            "diameter_mm": int(groups[1]),
                            "length_m": float(groups[2]),
                            "quantity": int(groups[3]),
                        }
                        entries.append(entry)
                    except (ValueError, IndexError):
                        continue

        return entries


def estimate_steel_for_building(
    built_up_area_sqm: float,
    num_floors: int = 1,
    building_type: str = "residential",
) -> Dict[str, float]:
    """
    Quick steel estimation for entire building.

    Uses typical steel consumption rates per sqm of built-up area.

    Returns:
        Dict with steel quantity estimates
    """
    # Typical steel consumption: 3-5 kg/sqft for residential
    # Converting to sqm: 32-54 kg/sqm

    if building_type == "residential":
        if num_floors <= 2:
            rate_kg_per_sqm = 35
        elif num_floors <= 4:
            rate_kg_per_sqm = 40
        else:
            rate_kg_per_sqm = 45
    else:  # commercial
        rate_kg_per_sqm = 50

    total_builtup = built_up_area_sqm * num_floors
    total_steel_kg = total_builtup * rate_kg_per_sqm

    return {
        "total_builtup_sqm": total_builtup,
        "steel_rate_kg_per_sqm": rate_kg_per_sqm,
        "estimated_steel_kg": round(total_steel_kg, 0),
        "estimated_steel_mt": round(total_steel_kg / 1000, 2),
        "note": "Rough estimate - actual may vary ±20%",
    }
