"""
Measurement Reasonableness Validator — Sanity checks for derived quantities.

Validates that measurement adjustments (deductions, formwork derivation,
steel estimation, etc.) produce reasonable results. Catches:
- Deductions exceeding gross area
- Formwork ratios outside IS 1200 norms
- Steel consumption outside CPWD norms
- Negative or zero quantities
- Unit mismatches
- Outlier quantities by trade

Based on IS 1200, IS 456, CPWD DSR 2024, and industry norms.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION SEVERITY
# =============================================================================

class Severity(Enum):
    """Validation result severity."""
    ERROR = "error"           # Likely wrong, needs correction
    WARNING = "warning"       # Possibly wrong, needs review
    INFO = "info"             # Informational, unusual but could be correct


# =============================================================================
# NORM RANGES (IS 1200 / CPWD / Industry)
# =============================================================================

# Formwork sqm per cum of RCC (typical ranges)
FORMWORK_NORMS: Dict[str, Dict[str, float]] = {
    "slab": {"min": 6.0, "typical": 8.0, "max": 12.0},       # 100-200mm thick slabs
    "beam": {"min": 8.0, "typical": 12.0, "max": 18.0},       # 200-600mm beams
    "column": {"min": 10.0, "typical": 16.0, "max": 24.0},    # 300-600mm columns
    "footing": {"min": 3.0, "typical": 5.0, "max": 8.0},      # Shallow footings
    "staircase": {"min": 8.0, "typical": 12.0, "max": 16.0},
    "lintel": {"min": 6.0, "typical": 10.0, "max": 15.0},
    "wall": {"min": 4.0, "typical": 8.0, "max": 12.0},        # RCC walls
    "general": {"min": 5.0, "typical": 10.0, "max": 20.0},
}

# Steel kg per cum of RCC (IS 456 typical ranges)
STEEL_NORMS: Dict[str, Dict[str, float]] = {
    "slab": {"min": 40.0, "typical": 65.0, "max": 120.0},
    "beam": {"min": 60.0, "typical": 100.0, "max": 180.0},
    "column": {"min": 60.0, "typical": 80.0, "max": 200.0},
    "footing": {"min": 30.0, "typical": 50.0, "max": 100.0},
    "staircase": {"min": 50.0, "typical": 80.0, "max": 130.0},
    "lintel": {"min": 40.0, "typical": 60.0, "max": 100.0},
    "wall": {"min": 40.0, "typical": 60.0, "max": 100.0},
    "raft": {"min": 50.0, "typical": 80.0, "max": 150.0},
    "general": {"min": 40.0, "typical": 80.0, "max": 200.0},
}

# Maximum reasonable deduction percentages by trade
MAX_DEDUCTION_PCT: Dict[str, float] = {
    "plaster_internal": 15.0,    # Openings rarely > 15% of wall area
    "plaster_external": 12.0,
    "painting_internal": 15.0,
    "painting_external": 12.0,
    "masonry": 20.0,             # More openings in masonry walls
    "tiling": 10.0,
    "skirting": 25.0,            # Door widths deducted from perimeter
    "dado": 20.0,
    "general": 15.0,
}

# Typical room area ranges (sqm)
ROOM_AREA_NORMS: Dict[str, Dict[str, float]] = {
    "bedroom": {"min": 9.0, "typical": 14.0, "max": 30.0},
    "living": {"min": 12.0, "typical": 18.0, "max": 50.0},
    "kitchen": {"min": 5.0, "typical": 9.0, "max": 20.0},
    "toilet": {"min": 2.0, "typical": 3.5, "max": 8.0},
    "bathroom": {"min": 2.5, "typical": 4.0, "max": 10.0},
    "store": {"min": 2.0, "typical": 4.0, "max": 10.0},
    "balcony": {"min": 1.5, "typical": 4.0, "max": 12.0},
    "corridor": {"min": 2.0, "typical": 6.0, "max": 20.0},
    "general": {"min": 2.0, "typical": 12.0, "max": 50.0},
}

# Unit expectations by trade
EXPECTED_UNITS: Dict[str, List[str]] = {
    "rcc": ["cum", "m3"],
    "pcc": ["cum", "m3"],
    "formwork": ["sqm", "m2"],
    "masonry": ["cum", "m3"],
    "plaster": ["sqm", "m2"],
    "painting": ["sqm", "m2"],
    "flooring": ["sqm", "m2"],
    "steel": ["kg", "mt", "tonne"],
    "waterproofing": ["sqm", "m2"],
    "earthwork": ["cum", "m3"],
    "skirting": ["rmt", "m"],
    "dado": ["sqm", "m2"],
    "door": ["nos", "no", "set"],
    "window": ["nos", "no", "set"],
    "plumbing_point": ["nos", "no", "point"],
    "electrical_point": ["nos", "no", "point"],
}


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationIssue:
    """Single validation issue found."""
    severity: Severity
    category: str         # e.g., "formwork_ratio", "deduction_pct", "unit_mismatch"
    item_id: str
    description: str
    actual_value: float
    expected_range: str   # e.g., "5.0 - 20.0"
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "item_id": self.item_id,
            "description": self.description,
            "actual_value": self.actual_value,
            "expected_range": self.expected_range,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    issues: List[ValidationIssue] = field(default_factory=list)
    items_checked: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0

    @property
    def pass_rate(self) -> float:
        if self.items_checked == 0:
            return 100.0
        return round((1 - self.errors / max(self.items_checked, 1)) * 100, 1)

    @property
    def is_clean(self) -> bool:
        return self.errors == 0 and self.warnings == 0

    def add(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity == Severity.ERROR:
            self.errors += 1
        elif issue.severity == Severity.WARNING:
            self.warnings += 1
        else:
            self.infos += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items_checked": self.items_checked,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "pass_rate": self.pass_rate,
            "is_clean": self.is_clean,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        lines = [f"Validation: {self.items_checked} items checked"]
        lines.append(f"  ✗ {self.errors} errors, ⚠ {self.warnings} warnings, ℹ {self.infos} info")
        lines.append(f"  Pass rate: {self.pass_rate}%")
        return "\n".join(lines)


# =============================================================================
# VALIDATORS
# =============================================================================

class MeasurementValidator:
    """Validates measurement-adjusted BOQ for reasonableness."""

    def __init__(self):
        self.report = ValidationReport()

    def validate_all(
        self,
        boq_items: List[Dict[str, Any]],
        structural_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """
        Run all validation checks on BOQ items.

        Args:
            boq_items: BOQ items with adjusted quantities
            structural_data: Optional structural element data for cross-checks

        Returns:
            ValidationReport
        """
        self.report = ValidationReport()
        self.report.items_checked = len(boq_items)

        for item in boq_items:
            self._validate_quantity(item)
            self._validate_unit(item)
            self._validate_deduction(item)

        # Cross-item checks
        self._validate_formwork_ratios(boq_items)
        self._validate_steel_ratios(boq_items)
        self._validate_quantity_relationships(boq_items)

        # IS 1200 norm checks
        self._validate_plaster_thickness(boq_items)
        self._validate_excavation_depth(boq_items)
        self._validate_dado_height(boq_items)

        logger.info(self.report.summary())
        return self.report

    def _validate_quantity(self, item: Dict[str, Any]):
        """Check quantity is positive and reasonable."""
        qty = float(item.get("quantity", item.get("qty", item.get("final_qty", 0))))
        item_id = item.get("item_id", item.get("unified_item_no", ""))
        desc = item.get("description", "")

        # Negative quantity
        if qty < 0:
            self.report.add(ValidationIssue(
                severity=Severity.ERROR,
                category="negative_quantity",
                item_id=item_id,
                description=f"Negative quantity: {desc}",
                actual_value=qty,
                expected_range="> 0",
                suggestion="Check deductions — deducted area may exceed gross area",
            ))

        # Zero quantity with assigned rate (should have been flagged earlier)
        if qty == 0 and float(item.get("rate", 0)) > 0:
            self.report.add(ValidationIssue(
                severity=Severity.WARNING,
                category="zero_quantity",
                item_id=item_id,
                description=f"Zero quantity with rate assigned: {desc}",
                actual_value=0,
                expected_range="> 0",
                suggestion="Verify if item should be removed or quantity needs recalculation",
            ))

    def _validate_unit(self, item: Dict[str, Any]):
        """Check unit matches expected unit for trade."""
        unit = item.get("unit", "").lower().strip()
        desc = item.get("description", "").lower()
        item_id = item.get("item_id", item.get("unified_item_no", ""))

        if not unit:
            return

        for trade, expected_units in EXPECTED_UNITS.items():
            if trade in desc:
                if unit not in [u.lower() for u in expected_units]:
                    self.report.add(ValidationIssue(
                        severity=Severity.WARNING,
                        category="unit_mismatch",
                        item_id=item_id,
                        description=f"Unit '{unit}' unusual for {trade}: {desc[:50]}",
                        actual_value=0,
                        expected_range=f"Expected: {', '.join(expected_units)}",
                        suggestion=f"Verify unit — {trade} items typically use {expected_units[0]}",
                    ))
                break  # Only check first matching trade

    def _validate_deduction(self, item: Dict[str, Any]):
        """Check deduction percentage is reasonable."""
        gross = float(item.get("gross_quantity", item.get("original_qty", 0)))
        net = float(item.get("quantity", item.get("qty", item.get("final_qty", 0))))
        item_id = item.get("item_id", item.get("unified_item_no", ""))
        desc = item.get("description", "")

        if gross <= 0 or net >= gross:
            return  # No deduction applied

        deduction_pct = (gross - net) / gross * 100

        # Determine max deduction for this trade
        max_pct = MAX_DEDUCTION_PCT.get("general", 15.0)
        for trade, threshold in MAX_DEDUCTION_PCT.items():
            if trade.replace("_", " ") in desc.lower():
                max_pct = threshold
                break

        if deduction_pct > max_pct:
            severity = Severity.ERROR if deduction_pct > max_pct * 1.5 else Severity.WARNING
            self.report.add(ValidationIssue(
                severity=severity,
                category="excessive_deduction",
                item_id=item_id,
                description=f"Deduction {deduction_pct:.1f}% exceeds norm for: {desc[:50]}",
                actual_value=round(deduction_pct, 1),
                expected_range=f"< {max_pct}%",
                suggestion="Review opening schedule — deduction area seems too high",
            ))

    def _validate_formwork_ratios(self, boq_items: List[Dict[str, Any]]):
        """Check formwork-to-RCC ratios are within IS norms."""
        rcc_items = {}
        formwork_items = {}

        for item in boq_items:
            desc = item.get("description", "").lower()
            qty = float(item.get("quantity", item.get("qty", item.get("final_qty", 0))))
            item_id = item.get("item_id", item.get("unified_item_no", ""))

            if qty <= 0:
                continue

            # Detect element type
            for element in ["slab", "beam", "column", "footing", "staircase", "lintel"]:
                if element in desc:
                    if "formwork" in desc or "shuttering" in desc:
                        formwork_items[element] = formwork_items.get(element, 0) + qty
                    elif "rcc" in desc or "concrete" in desc:
                        rcc_items[element] = rcc_items.get(element, 0) + qty
                    break

        # Check ratios
        for element in set(rcc_items.keys()) & set(formwork_items.keys()):
            rcc_vol = rcc_items[element]
            fw_area = formwork_items[element]
            if rcc_vol <= 0:
                continue

            ratio = fw_area / rcc_vol
            norms = FORMWORK_NORMS.get(element, FORMWORK_NORMS["general"])

            if ratio < norms["min"]:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="low_formwork_ratio",
                    item_id=f"formwork_{element}",
                    description=f"Formwork ratio {ratio:.1f} sqm/cum for {element} is below minimum",
                    actual_value=round(ratio, 1),
                    expected_range=f"{norms['min']} - {norms['max']} sqm/cum",
                    suggestion=f"Typical {element} formwork is ~{norms['typical']} sqm/cum. Check dimensions.",
                ))
            elif ratio > norms["max"]:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="high_formwork_ratio",
                    item_id=f"formwork_{element}",
                    description=f"Formwork ratio {ratio:.1f} sqm/cum for {element} exceeds maximum",
                    actual_value=round(ratio, 1),
                    expected_range=f"{norms['min']} - {norms['max']} sqm/cum",
                    suggestion=f"Verify formwork quantity — seems too high for {element}.",
                ))

    def _validate_steel_ratios(self, boq_items: List[Dict[str, Any]]):
        """Check steel-to-RCC ratios are within IS 456 norms."""
        rcc_items = {}
        steel_items = {}

        for item in boq_items:
            desc = item.get("description", "").lower()
            qty = float(item.get("quantity", item.get("qty", item.get("final_qty", 0))))

            if qty <= 0:
                continue

            for element in ["slab", "beam", "column", "footing", "staircase", "raft"]:
                if element in desc:
                    if any(kw in desc for kw in ["steel", "reinforcement", "rebar", "tmt"]):
                        steel_items[element] = steel_items.get(element, 0) + qty
                    elif "rcc" in desc or "concrete" in desc:
                        rcc_items[element] = rcc_items.get(element, 0) + qty
                    break

        for element in set(rcc_items.keys()) & set(steel_items.keys()):
            rcc_vol = rcc_items[element]
            steel_kg = steel_items[element]
            if rcc_vol <= 0:
                continue

            ratio = steel_kg / rcc_vol
            norms = STEEL_NORMS.get(element, STEEL_NORMS["general"])

            if ratio < norms["min"]:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="low_steel_ratio",
                    item_id=f"steel_{element}",
                    description=f"Steel ratio {ratio:.0f} kg/cum for {element} is below IS 456 minimum",
                    actual_value=round(ratio, 0),
                    expected_range=f"{norms['min']} - {norms['max']} kg/cum",
                    suggestion=f"Check steel schedule — minimum reinforcement for {element} is ~{norms['min']} kg/cum.",
                ))
            elif ratio > norms["max"]:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="high_steel_ratio",
                    item_id=f"steel_{element}",
                    description=f"Steel ratio {ratio:.0f} kg/cum for {element} exceeds typical maximum",
                    actual_value=round(ratio, 0),
                    expected_range=f"{norms['min']} - {norms['max']} kg/cum",
                    suggestion=f"Verify bar schedule — steel seems too high for {element}.",
                ))

    def _validate_quantity_relationships(self, boq_items: List[Dict[str, Any]]):
        """Check cross-item quantity consistency."""
        total_rcc = 0.0
        total_formwork = 0.0
        total_steel = 0.0

        # Count RCC items — also match grade descriptions like M20, M25
        rcc_pattern = re.compile(r'\brcc\b|m\s*-?\s*\d{2}|\bconcrete\b', re.IGNORECASE)

        for item in boq_items:
            desc = item.get("description", "").lower()
            qty = float(item.get("quantity", item.get("qty", item.get("final_qty", 0))))
            if qty <= 0:
                continue

            if (rcc_pattern.search(item.get("description", "")) or "reinforced" in desc) \
                    and "formwork" not in desc and "steel" not in desc:
                total_rcc += qty
            elif "formwork" in desc or "shuttering" in desc:
                total_formwork += qty
            elif any(kw in desc for kw in ["steel", "reinforcement", "rebar", "tmt"]) and "rcc" not in desc:
                total_steel += qty

        # Overall formwork ratio check
        if total_rcc > 0 and total_formwork > 0:
            overall_ratio = total_formwork / total_rcc
            if overall_ratio < 4.0:
                self.report.add(ValidationIssue(
                    severity=Severity.INFO,
                    category="low_overall_formwork",
                    item_id="overall",
                    description=f"Overall formwork ratio {overall_ratio:.1f} sqm/cum is low",
                    actual_value=round(overall_ratio, 1),
                    expected_range="6.0 - 15.0 sqm/cum overall",
                    suggestion="Some formwork items may be missing from BOQ.",
                ))
            elif overall_ratio > 20.0:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="high_overall_formwork",
                    item_id="overall",
                    description=f"Overall formwork ratio {overall_ratio:.1f} sqm/cum is very high",
                    actual_value=round(overall_ratio, 1),
                    expected_range="6.0 - 15.0 sqm/cum overall",
                    suggestion="Check for duplicate formwork items.",
                ))

        # Overall steel ratio check
        if total_rcc > 0 and total_steel > 0:
            steel_ratio = total_steel / total_rcc
            if steel_ratio < 30.0:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="low_overall_steel",
                    item_id="overall",
                    description=f"Overall steel ratio {steel_ratio:.0f} kg/cum is below minimum",
                    actual_value=round(steel_ratio, 0),
                    expected_range="40 - 120 kg/cum overall",
                    suggestion="Steel quantity seems too low — verify bar bending schedule.",
                ))
            elif steel_ratio > 200.0:
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="high_overall_steel",
                    item_id="overall",
                    description=f"Overall steel ratio {steel_ratio:.0f} kg/cum exceeds norms",
                    actual_value=round(steel_ratio, 0),
                    expected_range="40 - 120 kg/cum overall",
                    suggestion="Steel quantity seems too high — verify for duplicates.",
                ))

        # ---------------------------------------------------------------
        # IS 1200 cross-item presence checks
        # ---------------------------------------------------------------
        all_descs = [item.get("description", "").lower() for item in boq_items]
        all_descs_joined = " | ".join(all_descs)

        # -- Curing check (IS 456) --
        concrete_grade_pattern = re.compile(r'\brcc\b|\bconcrete\b|m\s*-?\s*\d{2}', re.IGNORECASE)
        has_concrete = any(concrete_grade_pattern.search(d) for d in all_descs)
        has_curing = any("curing" in d for d in all_descs)
        if has_concrete and not has_curing:
            self.report.add(ValidationIssue(
                severity=Severity.WARNING,
                category="curing",
                item_id="curing_check",
                description="RCC/concrete items found but no curing item in BOQ",
                actual_value=0,
                expected_range="IS 456 requires curing for all concrete work",
                suggestion="Add curing of concrete item to BOQ",
            ))

        # -- Anti-termite check (IS 6313) --
        has_foundation = any(
            kw in d for d in all_descs for kw in ["footing", "foundation", "excavation"]
        )
        has_anti_termite = any("anti-termite" in d or "anti termite" in d or "antitermite" in d for d in all_descs)
        if has_foundation and not has_anti_termite:
            self.report.add(ValidationIssue(
                severity=Severity.WARNING,
                category="anti_termite",
                item_id="anti_termite_check",
                description="Foundation items found but no anti-termite treatment in BOQ",
                actual_value=0,
                expected_range="IS 6313 requires anti-termite for ground contact structures",
                suggestion="Add anti-termite treatment item to BOQ",
            ))

        # -- Backfill needs compaction check --
        has_backfill = any("backfill" in d or "earth filling" in d for d in all_descs)
        has_compaction = any("compaction" in d or "watering" in d for d in all_descs)
        if has_backfill and not has_compaction:
            self.report.add(ValidationIssue(
                severity=Severity.INFO,
                category="compaction",
                item_id="compaction_check",
                description="Backfill items found but no compaction/watering item in BOQ",
                actual_value=0,
                expected_range="IS 1200 Part 1: compaction required for backfill",
                suggestion="Add compaction or watering item for backfill work",
            ))

        # -- Waterproofing for wet areas check --
        has_wet_area = any(
            kw in d for d in all_descs for kw in ["toilet", "bathroom", "wc"]
        )
        has_waterproofing = any("waterproofing" in d or "water proofing" in d for d in all_descs)
        if has_wet_area and not has_waterproofing:
            self.report.add(ValidationIssue(
                severity=Severity.WARNING,
                category="waterproofing_wet_area",
                item_id="waterproofing_wet_area_check",
                description="Wet area floor items found but no waterproofing in BOQ",
                actual_value=0,
                expected_range="Waterproofing required for toilets, bathrooms, WC areas",
                suggestion="Add waterproofing treatment item for wet areas",
            ))

        # -- Painting needs plaster check --
        has_painting = any("painting" in d or "emulsion" in d for d in all_descs)
        has_plaster = any("plaster" in d for d in all_descs)
        if has_painting and not has_plaster:
            self.report.add(ValidationIssue(
                severity=Severity.INFO,
                category="plaster_before_paint",
                item_id="plaster_before_paint_check",
                description="Painting items found but no plaster items in BOQ",
                actual_value=0,
                expected_range="Plaster is typically required before painting",
                suggestion="Verify if plaster is included or walls are fair-faced",
            ))

    def _validate_plaster_thickness(self, boq_items: List[Dict[str, Any]]):
        """Validate plaster thickness against IS 1200 Part 5 norms."""
        PLASTER_NORMS = {
            "internal": (12, 20),
            "external": (15, 25),
            "ceiling": (6, 12),
        }
        for item in boq_items:
            desc = item.get("description", "").lower()
            if "plaster" not in desc:
                continue
            # Extract thickness from description (e.g., "12mm", "15 mm", "20 mm thick")
            thickness_match = re.search(r'(\d+)\s*mm', desc)
            if not thickness_match:
                continue
            thickness = int(thickness_match.group(1))

            plaster_type = "internal"  # default
            if "external" in desc or "outer" in desc:
                plaster_type = "external"
            elif "ceiling" in desc:
                plaster_type = "ceiling"

            min_t, max_t = PLASTER_NORMS[plaster_type]
            if thickness < min_t or thickness > max_t:
                item_id = item.get("item_id", item.get("unified_item_no", ""))
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="plaster_thickness",
                    item_id=item_id or "plaster_thickness_check",
                    description=f"Plaster thickness {thickness}mm outside IS 1200 range for {plaster_type}",
                    actual_value=float(thickness),
                    expected_range=f"{min_t}-{max_t}mm for {plaster_type} plaster",
                    suggestion=f"IS 1200 Part 5: {plaster_type} plaster should be {min_t}-{max_t}mm",
                ))

    def _validate_excavation_depth(self, boq_items: List[Dict[str, Any]]):
        """Validate excavation items against IS 1200 Part 1."""
        has_shoring = any(
            "shor" in item.get("description", "").lower()
            or "timber" in item.get("description", "").lower()
            or "breast" in item.get("description", "").lower()
            for item in boq_items
        )
        for item in boq_items:
            desc = item.get("description", "").lower()
            if "excavat" not in desc:
                continue
            # Check for depth indication
            depth_match = re.search(r'(\d+\.?\d*)\s*m\s*(deep|depth)', desc)
            if depth_match:
                depth = float(depth_match.group(1))
                if depth > 1.5 and not has_shoring:
                    item_id = item.get("item_id", item.get("unified_item_no", ""))
                    self.report.add(ValidationIssue(
                        severity=Severity.WARNING,
                        category="excavation_depth",
                        item_id=item_id or "excavation_depth_check",
                        description=f"Excavation depth {depth}m > 1.5m but no shoring/timbering found",
                        actual_value=depth,
                        expected_range="IS 1200 Part 1: shoring required for depth > 1.5m",
                        suggestion="Add shoring/timbering item for excavation deeper than 1.5m",
                    ))

    def _validate_dado_height(self, boq_items: List[Dict[str, Any]]):
        """Validate dado height consistency."""
        for item in boq_items:
            desc = item.get("description", "").lower()
            if "dado" not in desc:
                continue
            height_match = re.search(r'(\d+)\s*mm', desc)
            if not height_match:
                continue
            height = int(height_match.group(1))
            # Typical dado: 1200mm general, up to 2100mm for toilets
            if height < 600 or height > 2400:
                item_id = item.get("item_id", item.get("unified_item_no", ""))
                self.report.add(ValidationIssue(
                    severity=Severity.WARNING,
                    category="dado_height",
                    item_id=item_id or "dado_height_check",
                    description=f"Dado height {height}mm outside typical range (600-2400mm)",
                    actual_value=float(height),
                    expected_range="600-2400mm (1200mm standard, up to 2100mm for toilets)",
                    suggestion=f"Verify dado height — {height}mm is unusual",
                ))


def validate_boq(
    boq_items: List[Dict[str, Any]],
    structural_data: Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    """
    Convenience function to validate a BOQ.

    Args:
        boq_items: BOQ items
        structural_data: Optional structural data

    Returns:
        ValidationReport
    """
    validator = MeasurementValidator()
    return validator.validate_all(boq_items, structural_data)
