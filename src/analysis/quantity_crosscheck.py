"""
Quantity Cross-Check Engine — Per-Material Variance Detection.

Cross-validates BOQ quantities against multiple sources with material-specific
tolerance thresholds. Catches the most common estimation errors:
1. Formwork quantity derivable from RCC volumes
2. Painting area derivable from plaster area
3. Steel weight derivable from RCC volume + element type
4. Waterproofing from wet area floor + terrace areas
5. Skirting from room perimeters
6. Door/window count consistency

Indian construction standard basis: IS 1200, IS 456, CPWD norms.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


class VarianceSeverity(Enum):
    CRITICAL = "critical"    # > 30% variance on high-value item
    HIGH = "high"            # > 20% variance
    MEDIUM = "medium"        # > 10% variance
    LOW = "low"              # > 5% variance (informational)
    OK = "ok"                # Within tolerance


@dataclass
class CrossCheckResult:
    """Result of one cross-validation check."""
    check_type: str              # e.g., "formwork_from_rcc", "paint_from_plaster"
    item_description: str
    boq_qty: float
    derived_qty: float
    variance_pct: float          # (boq - derived) / derived * 100
    severity: VarianceSeverity
    unit: str
    explanation: str
    is_standard: str = ""        # IS standard reference

    def to_dict(self):
        return {
            "check_type": self.check_type,
            "item_description": self.item_description,
            "boq_qty": self.boq_qty,
            "derived_qty": round(self.derived_qty, 2),
            "variance_pct": round(self.variance_pct, 1),
            "severity": self.severity.value,
            "unit": self.unit,
            "explanation": self.explanation,
            "is_standard": self.is_standard,
        }


# =============================================================================
# PER-MATERIAL VARIANCE THRESHOLDS (% deviation allowed)
# =============================================================================

MATERIAL_THRESHOLDS = {
    # High-value items: tight thresholds
    "steel": {"low": 3, "medium": 8, "high": 15, "critical": 25},
    "rcc_concrete": {"low": 5, "medium": 10, "high": 20, "critical": 30},
    "structural_steel": {"low": 3, "medium": 8, "high": 15, "critical": 25},

    # Medium-value items
    "formwork": {"low": 8, "medium": 15, "high": 25, "critical": 40},
    "masonry": {"low": 10, "medium": 20, "high": 30, "critical": 50},
    "plaster": {"low": 10, "medium": 20, "high": 30, "critical": 50},
    "flooring": {"low": 5, "medium": 15, "high": 25, "critical": 40},
    "painting": {"low": 10, "medium": 20, "high": 35, "critical": 50},
    "waterproofing": {"low": 8, "medium": 15, "high": 25, "critical": 40},

    # Lower-value items: wider thresholds
    "skirting": {"low": 15, "medium": 25, "high": 40, "critical": 60},
    "doors_windows": {"low": 0, "medium": 10, "high": 20, "critical": 50},
    "plumbing": {"low": 10, "medium": 20, "high": 35, "critical": 50},
    "electrical": {"low": 10, "medium": 20, "high": 35, "critical": 50},

    # Default
    "default": {"low": 10, "medium": 20, "high": 30, "critical": 50},
}


# =============================================================================
# DERIVED QUANTITY RATIOS (IS 1200 / IS 456 / CPWD norms)
# =============================================================================

DERIVED_RATIOS = {
    # Formwork sqm per cum of RCC by element type
    "formwork_per_rcc": {
        "footing": 4.0,      # 4 sqm formwork per cum RCC footing
        "column": 14.0,      # 14 sqm per cum (tall thin elements)
        "beam": 12.0,        # 12 sqm per cum
        "slab": 8.0,         # 8 sqm per cum (mostly soffit)
        "staircase": 10.0,
        "lintel": 12.0,
        "wall": 10.0,
        "default": 8.0,
    },
    # Steel kg per cum of RCC by element type (IS 456 typical)
    "steel_per_rcc": {
        "footing": 80,       # kg/cum
        "column": 180,
        "beam": 150,
        "slab": 90,
        "staircase": 110,
        "lintel": 120,
        "raft": 100,
        "wall": 80,
        "default": 120,
    },
    # Paint area = plaster area x 1.0 (same surface)
    "paint_per_plaster": 1.0,
    # Waterproofing area check ratios
    "wp_toilet_sqm_typical": 4.0,   # avg toilet floor area
    "wp_terrace_pct_of_slab": 0.15, # terrace typically 15% of total slab area
}


# =============================================================================
# CROSS-CHECK REPORT
# =============================================================================


@dataclass
class CrossCheckReport:
    """Complete cross-check report aggregating all individual check results."""
    checks: List[CrossCheckResult] = field(default_factory=list)

    @property
    def critical_issues(self) -> List[CrossCheckResult]:
        """Return all checks flagged as critical severity."""
        return [c for c in self.checks if c.severity == VarianceSeverity.CRITICAL]

    @property
    def high_issues(self) -> List[CrossCheckResult]:
        """Return all checks flagged as high severity."""
        return [c for c in self.checks if c.severity == VarianceSeverity.HIGH]

    @property
    def issues_count(self) -> int:
        """Count checks with severity above LOW (i.e., MEDIUM, HIGH, CRITICAL)."""
        return sum(
            1 for c in self.checks
            if c.severity not in (VarianceSeverity.OK, VarianceSeverity.LOW)
        )

    @property
    def overall_confidence(self) -> float:
        """
        Compute 0-100 confidence score. High score = few/no variances.

        Each check contributes a penalty based on its severity:
            CRITICAL = 15, HIGH = 8, MEDIUM = 3, LOW = 1, OK = 0.
        Score = (1 - total_penalty / max_possible_penalty) * 100, floored at 0.
        """
        if not self.checks:
            return 100.0
        penalties = {
            VarianceSeverity.CRITICAL: 15,
            VarianceSeverity.HIGH: 8,
            VarianceSeverity.MEDIUM: 3,
            VarianceSeverity.LOW: 1,
            VarianceSeverity.OK: 0,
        }
        total_penalty = sum(penalties[c.severity] for c in self.checks)
        max_penalty = len(self.checks) * 15
        return max(0.0, round((1 - total_penalty / max(max_penalty, 1)) * 100, 1))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report to a dict suitable for JSON output."""
        return {
            "total_checks": len(self.checks),
            "issues_found": self.issues_count,
            "critical": len(self.critical_issues),
            "high": len(self.high_issues),
            "overall_confidence": self.overall_confidence,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        """One-line human-readable summary of findings."""
        return (
            f"Cross-check: {len(self.checks)} checks, "
            f"{self.issues_count} issues ({len(self.critical_issues)} critical, "
            f"{len(self.high_issues)} high), "
            f"confidence={self.overall_confidence}%"
        )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _classify_severity(variance_pct: float, material: str) -> VarianceSeverity:
    """
    Classify variance severity using per-material thresholds.

    Looks up the threshold band for the given material in MATERIAL_THRESHOLDS.
    Falls back to "default" if material is not recognized. Uses the absolute
    value of variance_pct to classify into CRITICAL / HIGH / MEDIUM / LOW / OK.

    Args:
        variance_pct: Signed percentage deviation (positive = BOQ is higher).
        material: Material key matching a MATERIAL_THRESHOLDS entry.

    Returns:
        VarianceSeverity enum member.
    """
    thresholds = MATERIAL_THRESHOLDS.get(material, MATERIAL_THRESHOLDS["default"])
    abs_var = abs(variance_pct)
    if abs_var >= thresholds["critical"]:
        return VarianceSeverity.CRITICAL
    elif abs_var >= thresholds["high"]:
        return VarianceSeverity.HIGH
    elif abs_var >= thresholds["medium"]:
        return VarianceSeverity.MEDIUM
    elif abs_var >= thresholds["low"]:
        return VarianceSeverity.LOW
    return VarianceSeverity.OK


def _detect_element_type(description: str) -> str:
    """
    Detect structural element type from a BOQ item description string.

    Scans for known element keywords (footing, column, beam, slab, etc.).
    Returns the first match found, or "default" if no match.

    Args:
        description: BOQ item description text.

    Returns:
        Element type string, one of: footing, column, beam, slab, staircase,
        lintel, raft, wall, or "default".
    """
    desc = description.lower()
    for element in [
        "footing", "column", "beam", "slab", "staircase",
        "lintel", "raft", "wall",
    ]:
        if element in desc:
            return element
    return "default"


def _extract_qty(item: Dict) -> float:
    """
    Extract numeric quantity from a BOQ item dict.

    Looks for keys in order: 'quantity', 'qty', 'final_qty'. Casts to float.
    Returns 0.0 on missing/invalid values.

    Args:
        item: BOQ item dict.

    Returns:
        Quantity as a float, or 0.0.
    """
    qty = item.get("quantity", item.get("qty", item.get("final_qty", 0)))
    try:
        return float(qty) if qty else 0.0
    except (ValueError, TypeError):
        return 0.0


def _get_description(item: Dict) -> str:
    """
    Extract description string from a BOQ item dict.

    Checks 'description', 'item_description', and 'desc' keys.

    Args:
        item: BOQ item dict.

    Returns:
        Lowercased description string, or empty string.
    """
    desc = item.get("description", item.get("item_description", item.get("desc", "")))
    return (desc or "").lower()


def _get_unit(item: Dict) -> str:
    """
    Extract unit string from a BOQ item dict.

    Checks 'unit' and 'uom' keys.

    Args:
        item: BOQ item dict.

    Returns:
        Lowercased unit string, or empty string.
    """
    unit = item.get("unit", item.get("uom", ""))
    return (unit or "").lower()


def _variance_pct(boq_qty: float, derived_qty: float) -> float:
    """
    Compute signed variance percentage: (boq - derived) / derived * 100.

    Returns 0.0 if derived_qty is zero to avoid division by zero.

    Args:
        boq_qty: Quantity stated in the BOQ.
        derived_qty: Quantity derived from cross-check logic.

    Returns:
        Signed variance percentage.
    """
    if derived_qty == 0:
        return 0.0
    return (boq_qty - derived_qty) / derived_qty * 100


# =============================================================================
# REGEX PATTERNS FOR ITEM CLASSIFICATION
# =============================================================================

_RCC_RE = re.compile(
    r'\b(rcc|r\.c\.c|reinforced\s+cement\s+concrete|reinforced\s+concrete)\b',
    re.IGNORECASE,
)
_PCC_RE = re.compile(
    r'\b(pcc|p\.c\.c|plain\s+cement\s+concrete|plain\s+concrete|lean\s+concrete)\b',
    re.IGNORECASE,
)
_FORMWORK_RE = re.compile(
    r'\b(formwork|shuttering|centering|form\s*work)\b',
    re.IGNORECASE,
)
_STEEL_RE = re.compile(
    r'\b(reinforcement|rebar|steel|tor\s*steel|tmt|bar\s*bending)\b',
    re.IGNORECASE,
)
_PLASTER_RE = re.compile(
    r'\b(plaster|plastering|rendering|cement\s*mortar\s*plaster)\b',
    re.IGNORECASE,
)
_PAINT_RE = re.compile(
    r'\b(paint|painting|distemper|emulsion|primer|enamel|putty)\b',
    re.IGNORECASE,
)
_FLOORING_RE = re.compile(
    r'\b(flooring|floor\s*tile|vitrified|ceramic\s*tile|marble\s*floor|'
    r'granite\s*floor|kota\s*stone|shahabad)\b',
    re.IGNORECASE,
)
_SKIRTING_RE = re.compile(
    r'\b(skirting|dado)\b',
    re.IGNORECASE,
)
_DOOR_FRAME_RE = re.compile(
    r'\b(door\s*frame|chowkhath?|chaukhat|wooden\s*frame.*door)\b',
    re.IGNORECASE,
)
_DOOR_SHUTTER_RE = re.compile(
    r'\b(door\s*shutter|flush\s*door|panel\s*door|pvc\s*door|'
    r'frp\s*door|wooden\s*shutter)\b',
    re.IGNORECASE,
)
_DOOR_HARDWARE_RE = re.compile(
    r'\b(door\s*hardware|aldrop|tower\s*bolt|handle|hinges?\s*for\s*door|'
    r'door\s*closer|door\s*lock|door\s*stopper)\b',
    re.IGNORECASE,
)
_WINDOW_FRAME_RE = re.compile(
    r'\b(window\s*frame|aluminium\s*window|upvc\s*window|steel\s*window)\b',
    re.IGNORECASE,
)
_WINDOW_SHUTTER_RE = re.compile(
    r'\b(window\s*shutter|window\s*glass|glazing.*window)\b',
    re.IGNORECASE,
)
_WATERPROOFING_RE = re.compile(
    r'\b(waterproof|water\s*proof|wp\s*treatment|damp\s*proof|'
    r'integral\s*waterproof|membrane|bituminous)\b',
    re.IGNORECASE,
)
_WET_AREA_RE = re.compile(
    r'\b(toilet|bathroom|wash\s*room|lavatory|wc|wet\s*area|'
    r'kitchen|utility|balcony|terrace)\b',
    re.IGNORECASE,
)
_EARTHWORK_RE = re.compile(
    r'\b(excavat|earth\s*work|earthwork|trenching|digging|'
    r'earth\s*cutting|cutting\s*in\s*earth)\b',
    re.IGNORECASE,
)
_FOUNDATION_RE = re.compile(
    r'\b(foundation|footing|raft|pile\s*cap|plinth\s*beam|'
    r'grade\s*beam|sub\s*structure)\b',
    re.IGNORECASE,
)


def _is_volume_unit(unit: str) -> bool:
    """Check if unit represents volume (cum, m3, cubic metre, etc.)."""
    u = unit.lower().strip()
    return bool(re.search(r'\b(cum|cu\.?\s*m|m3|cubic\s*met|cmt)\b', u))


def _is_area_unit(unit: str) -> bool:
    """Check if unit represents area (sqm, m2, square metre, etc.)."""
    u = unit.lower().strip()
    return bool(re.search(r'\b(sqm|sq\.?\s*m|m2|square\s*met|smt)\b', u))


def _is_weight_unit(unit: str) -> bool:
    """Check if unit represents weight (kg, mt, tonne, quintal, etc.)."""
    u = unit.lower().strip()
    return bool(re.search(r'\b(kg|kgs|mt|tonne|quintal|q)\b', u))


def _is_running_unit(unit: str) -> bool:
    """Check if unit represents running length (rmt, rm, m, metre, etc.)."""
    u = unit.lower().strip()
    return bool(re.search(r'\b(rmt|rm|r\.m|running\s*met|met|m)\b', u))


def _is_number_unit(unit: str) -> bool:
    """Check if unit represents count (nos, no, each, set, etc.)."""
    u = unit.lower().strip()
    return bool(re.search(r'\b(nos?|each|set|pair|unit|number)\b', u))


# =============================================================================
# CROSS-CHECK FUNCTIONS
# =============================================================================


def _check_formwork_vs_rcc(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate formwork quantities against RCC volumes.

    For each structural element type found in the BOQ (footing, column, beam,
    slab, etc.), the expected formwork area is computed as:

        expected_formwork_sqm = RCC_volume_cum * formwork_per_rcc_ratio

    The actual formwork quantity for that element type is then compared. If there
    is a significant variance, a CrossCheckResult is produced.

    Ratios are based on IS 1200 (Part 2) norms for typical formwork requirements
    per cubic metre of concrete by element type.

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult for each element type checked.
    """
    results: List[CrossCheckResult] = []

    # Collect RCC volumes by element type
    rcc_by_element: Dict[str, float] = {}
    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)
        if _RCC_RE.search(desc) and _is_volume_unit(unit) and qty > 0:
            etype = _detect_element_type(desc)
            rcc_by_element[etype] = rcc_by_element.get(etype, 0) + qty

    if not rcc_by_element:
        return results

    # Collect formwork areas by element type
    formwork_by_element: Dict[str, float] = {}
    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)
        if _FORMWORK_RE.search(desc) and _is_area_unit(unit) and qty > 0:
            etype = _detect_element_type(desc)
            formwork_by_element[etype] = formwork_by_element.get(etype, 0) + qty

    # Also collect total formwork if element-level matching yields nothing
    total_formwork = sum(formwork_by_element.values())

    # Cross-check per element type
    ratios = DERIVED_RATIOS["formwork_per_rcc"]
    checked_elements = set()

    for etype, rcc_vol in rcc_by_element.items():
        ratio = ratios.get(etype, ratios["default"])
        expected_formwork = rcc_vol * ratio
        actual_formwork = formwork_by_element.get(etype, 0)

        if actual_formwork > 0:
            checked_elements.add(etype)
            var = _variance_pct(actual_formwork, expected_formwork)
            severity = _classify_severity(var, "formwork")
            results.append(CrossCheckResult(
                check_type="formwork_from_rcc",
                item_description=f"Formwork for {etype} (RCC {rcc_vol:.1f} cum)",
                boq_qty=actual_formwork,
                derived_qty=expected_formwork,
                variance_pct=var,
                severity=severity,
                unit="sqm",
                explanation=(
                    f"Expected {expected_formwork:.1f} sqm formwork for "
                    f"{rcc_vol:.1f} cum RCC {etype} "
                    f"(ratio {ratio} sqm/cum). "
                    f"BOQ has {actual_formwork:.1f} sqm. "
                    f"Variance: {var:+.1f}%."
                ),
                is_standard="IS 1200 Part 2",
            ))

    # Aggregate check: if no per-element formwork items matched but there is
    # total formwork and total RCC, do a combined check.
    if not checked_elements and total_formwork == 0:
        total_rcc = sum(rcc_by_element.values())
        if total_rcc > 0:
            # Formwork is expected but entirely missing from BOQ
            default_ratio = ratios["default"]
            expected = total_rcc * default_ratio
            results.append(CrossCheckResult(
                check_type="formwork_from_rcc",
                item_description="Formwork (missing — RCC items found)",
                boq_qty=0,
                derived_qty=expected,
                variance_pct=-100.0,
                severity=VarianceSeverity.CRITICAL,
                unit="sqm",
                explanation=(
                    f"BOQ contains {total_rcc:.1f} cum of RCC but no formwork "
                    f"items found. Expected ~{expected:.0f} sqm formwork. "
                    f"Formwork may be missing or described differently."
                ),
                is_standard="IS 1200 Part 2",
            ))
    elif not checked_elements and total_formwork > 0:
        # Formwork exists but couldn't be matched to element types; do aggregate
        total_rcc = sum(rcc_by_element.values())
        # Weighted average ratio based on element mix
        total_expected = sum(
            vol * ratios.get(et, ratios["default"])
            for et, vol in rcc_by_element.items()
        )
        var = _variance_pct(total_formwork, total_expected)
        severity = _classify_severity(var, "formwork")
        results.append(CrossCheckResult(
            check_type="formwork_from_rcc",
            item_description=f"Formwork total (vs {total_rcc:.1f} cum RCC total)",
            boq_qty=total_formwork,
            derived_qty=total_expected,
            variance_pct=var,
            severity=severity,
            unit="sqm",
            explanation=(
                f"Aggregate check: {total_formwork:.1f} sqm formwork "
                f"vs {total_expected:.1f} sqm expected from "
                f"{total_rcc:.1f} cum RCC. Variance: {var:+.1f}%."
            ),
            is_standard="IS 1200 Part 2",
        ))

    return results


def _check_steel_vs_rcc(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate reinforcement steel quantities against RCC volumes.

    For each structural element type, the expected steel weight is:

        expected_steel_kg = RCC_volume_cum * steel_per_rcc_ratio

    Ratios are based on IS 456 typical reinforcement densities and CPWD norms
    for various element types. Column and beam elements have higher steel
    densities than slabs and footings.

    Handles both kg and MT (metric tonne) units by normalizing to kg.

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult for each element type checked.
    """
    results: List[CrossCheckResult] = []

    # Collect RCC volumes by element type
    rcc_by_element: Dict[str, float] = {}
    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)
        if _RCC_RE.search(desc) and _is_volume_unit(unit) and qty > 0:
            etype = _detect_element_type(desc)
            rcc_by_element[etype] = rcc_by_element.get(etype, 0) + qty

    if not rcc_by_element:
        return results

    # Collect steel quantities (normalize to kg)
    steel_by_element: Dict[str, float] = {}
    total_steel_kg = 0.0
    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)
        if _STEEL_RE.search(desc) and _is_weight_unit(unit) and qty > 0:
            # Normalize to kg
            u = unit.lower()
            if "mt" in u or "tonne" in u:
                qty_kg = qty * 1000
            elif "quintal" in u or u.strip() == "q":
                qty_kg = qty * 100
            else:
                qty_kg = qty  # assume kg

            etype = _detect_element_type(desc)
            steel_by_element[etype] = steel_by_element.get(etype, 0) + qty_kg
            total_steel_kg += qty_kg

    # Cross-check per element type
    ratios = DERIVED_RATIOS["steel_per_rcc"]
    checked_elements = set()

    for etype, rcc_vol in rcc_by_element.items():
        ratio = ratios.get(etype, ratios["default"])
        expected_steel = rcc_vol * ratio
        actual_steel = steel_by_element.get(etype, 0)

        if actual_steel > 0:
            checked_elements.add(etype)
            var = _variance_pct(actual_steel, expected_steel)
            severity = _classify_severity(var, "steel")
            results.append(CrossCheckResult(
                check_type="steel_from_rcc",
                item_description=f"Steel for {etype} (RCC {rcc_vol:.1f} cum)",
                boq_qty=actual_steel,
                derived_qty=expected_steel,
                variance_pct=var,
                severity=severity,
                unit="kg",
                explanation=(
                    f"Expected {expected_steel:.0f} kg steel for "
                    f"{rcc_vol:.1f} cum RCC {etype} "
                    f"(ratio {ratio} kg/cum). "
                    f"BOQ has {actual_steel:.0f} kg. "
                    f"Variance: {var:+.1f}%."
                ),
                is_standard="IS 456:2000",
            ))

    # Aggregate check if no element-level matching possible
    if not checked_elements and total_steel_kg > 0:
        total_rcc = sum(rcc_by_element.values())
        total_expected = sum(
            vol * ratios.get(et, ratios["default"])
            for et, vol in rcc_by_element.items()
        )
        var = _variance_pct(total_steel_kg, total_expected)
        severity = _classify_severity(var, "steel")
        results.append(CrossCheckResult(
            check_type="steel_from_rcc",
            item_description=f"Steel total (vs {total_rcc:.1f} cum RCC total)",
            boq_qty=total_steel_kg,
            derived_qty=total_expected,
            variance_pct=var,
            severity=severity,
            unit="kg",
            explanation=(
                f"Aggregate check: {total_steel_kg:.0f} kg steel "
                f"vs {total_expected:.0f} kg expected from "
                f"{total_rcc:.1f} cum RCC. Variance: {var:+.1f}%."
            ),
            is_standard="IS 456:2000",
        ))
    elif not checked_elements and total_steel_kg == 0 and rcc_by_element:
        # Steel missing entirely
        total_rcc = sum(rcc_by_element.values())
        total_expected = sum(
            vol * ratios.get(et, ratios["default"])
            for et, vol in rcc_by_element.items()
        )
        results.append(CrossCheckResult(
            check_type="steel_from_rcc",
            item_description="Steel reinforcement (missing — RCC items found)",
            boq_qty=0,
            derived_qty=total_expected,
            variance_pct=-100.0,
            severity=VarianceSeverity.CRITICAL,
            unit="kg",
            explanation=(
                f"BOQ contains {total_rcc:.1f} cum of RCC but no steel "
                f"reinforcement items found. Expected ~{total_expected:.0f} kg. "
                f"Steel may be in a separate schedule or missing."
            ),
            is_standard="IS 456:2000",
        ))

    return results


def _check_paint_vs_plaster(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate painting area against plaster area.

    Painting and plastering cover approximately the same wall/ceiling surfaces.
    The expected relationship is:

        painting_area ~= plaster_area * paint_per_plaster_ratio (1.0)

    A significant deviation suggests either painting or plastering items are
    under/over-counted, or surfaces are specified differently.

    Note: Internal and external surfaces may use different paint types but the
    total area should still roughly equal plastered area. Some surfaces may be
    left exposed (fair-face concrete) or have wallpaper instead of paint, which
    can cause legitimate deviations.

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult (at most one check).
    """
    results: List[CrossCheckResult] = []

    plaster_area = 0.0
    paint_area = 0.0

    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)

        if _is_area_unit(unit) and qty > 0:
            if _PLASTER_RE.search(desc):
                plaster_area += qty
            if _PAINT_RE.search(desc):
                paint_area += qty

    if plaster_area <= 0 or paint_area <= 0:
        # Can't cross-check without both — skip
        return results

    ratio = DERIVED_RATIOS["paint_per_plaster"]
    expected_paint = plaster_area * ratio
    var = _variance_pct(paint_area, expected_paint)
    severity = _classify_severity(var, "painting")

    results.append(CrossCheckResult(
        check_type="paint_from_plaster",
        item_description="Painting area vs plaster area",
        boq_qty=paint_area,
        derived_qty=expected_paint,
        variance_pct=var,
        severity=severity,
        unit="sqm",
        explanation=(
            f"Total plaster area: {plaster_area:.1f} sqm. "
            f"Total painting area: {paint_area:.1f} sqm. "
            f"Expected ratio ~{ratio}. "
            f"Variance: {var:+.1f}%. "
            f"Large positive = excess paint area (extra coats or surfaces?). "
            f"Large negative = surfaces not painted."
        ),
        is_standard="IS 1200 Part 13/14",
    ))

    return results


def _check_skirting_vs_flooring(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate skirting quantity against flooring area.

    Skirting runs along the perimeter of floored rooms. For a room of area A,
    the approximate perimeter is:

        perimeter ~= 4 * sqrt(A)   (assuming roughly square rooms)

    So total skirting (rmt) should approximate 4 * sqrt(total_flooring_sqm)
    summed across rooms. Since BOQ often has a single total flooring area,
    we approximate by assuming N rooms of equal size:

        If flooring is one lump sum area, treat it as multiple typical rooms
        (say, 15 sqm each) for perimeter estimation.

    This is inherently approximate, so skirting thresholds are wider.

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult (at most one check).
    """
    results: List[CrossCheckResult] = []

    flooring_sqm = 0.0
    skirting_rmt = 0.0

    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)

        if qty > 0:
            if _FLOORING_RE.search(desc) and _is_area_unit(unit):
                flooring_sqm += qty
            elif _SKIRTING_RE.search(desc) and (_is_running_unit(unit) or _is_area_unit(unit)):
                skirting_rmt += qty

    if flooring_sqm <= 0 or skirting_rmt <= 0:
        return results

    # Estimate number of rooms assuming ~15 sqm average room size (Indian std)
    typical_room_area = 15.0
    num_rooms = max(1, flooring_sqm / typical_room_area)
    # Perimeter of each room ~ 4 * sqrt(typical_room_area)
    perimeter_per_room = 4 * math.sqrt(typical_room_area)
    # Subtract ~2m per room for door openings (no skirting at doors)
    effective_perimeter = perimeter_per_room - 2.0
    expected_skirting = num_rooms * effective_perimeter

    var = _variance_pct(skirting_rmt, expected_skirting)
    severity = _classify_severity(var, "skirting")

    results.append(CrossCheckResult(
        check_type="skirting_from_flooring",
        item_description="Skirting rmt vs flooring area",
        boq_qty=skirting_rmt,
        derived_qty=expected_skirting,
        variance_pct=var,
        severity=severity,
        unit="rmt",
        explanation=(
            f"Total flooring: {flooring_sqm:.1f} sqm. "
            f"Estimated ~{num_rooms:.0f} rooms of ~{typical_room_area} sqm. "
            f"Expected skirting: {expected_skirting:.0f} rmt "
            f"(perimeter minus door openings). "
            f"BOQ has {skirting_rmt:.1f} rmt. "
            f"Variance: {var:+.1f}%."
        ),
        is_standard="IS 1200 Part 12",
    ))

    return results


def _check_door_window_consistency(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate door and window component counts for consistency.

    Each door in a building should have matching components:
        - 1 frame (chowkhath)
        - 1 shutter (flush/panel/FRP door leaf)
        - 1 set of hardware (aldrop, tower bolt, handle, hinges, lock)

    Similarly for windows:
        - 1 frame
        - 1 shutter/glass

    Mismatches indicate missing items (common: hardware forgotten, or
    shutters specified but not frames).

    Counts items in "nos" or "each" units. Handles cases where a single
    item covers both frame + shutter (e.g., "door frame and shutter complete").

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult for door and window consistency.
    """
    results: List[CrossCheckResult] = []

    door_frames = 0.0
    door_shutters = 0.0
    door_hardware = 0.0
    window_frames = 0.0
    window_shutters = 0.0

    # Combined pattern: "door frame and shutter" or "door complete"
    _door_combined_re = re.compile(
        r'\b(door\s*(frame\s*(&|and)\s*shutter|complete|with\s*frame))\b',
        re.IGNORECASE,
    )

    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)

        if qty <= 0:
            continue

        # Only count items in number units
        if not _is_number_unit(unit):
            continue

        # Check for combined door items first
        if _door_combined_re.search(desc):
            door_frames += qty
            door_shutters += qty
            continue

        # Individual door components
        if _DOOR_FRAME_RE.search(desc):
            door_frames += qty
        if _DOOR_SHUTTER_RE.search(desc):
            door_shutters += qty
        if _DOOR_HARDWARE_RE.search(desc):
            door_hardware += qty

        # Window components
        if _WINDOW_FRAME_RE.search(desc):
            window_frames += qty
        if _WINDOW_SHUTTER_RE.search(desc):
            window_shutters += qty

    # Door frame vs shutter check
    if door_frames > 0 or door_shutters > 0:
        max_doors = max(door_frames, door_shutters)
        if max_doors > 0:
            # Frame vs shutter
            if door_frames > 0 and door_shutters > 0:
                var = _variance_pct(door_shutters, door_frames)
                severity = _classify_severity(var, "doors_windows")
                results.append(CrossCheckResult(
                    check_type="door_frame_shutter",
                    item_description="Door frames vs door shutters",
                    boq_qty=door_shutters,
                    derived_qty=door_frames,
                    variance_pct=var,
                    severity=severity,
                    unit="nos",
                    explanation=(
                        f"Door frames: {door_frames:.0f} nos. "
                        f"Door shutters: {door_shutters:.0f} nos. "
                        f"Each door needs 1 frame + 1 shutter. "
                        f"Variance: {var:+.1f}%."
                    ),
                    is_standard="IS 1200 Part 10",
                ))
            elif door_frames > 0 and door_shutters == 0:
                results.append(CrossCheckResult(
                    check_type="door_frame_shutter",
                    item_description="Door shutters missing",
                    boq_qty=0,
                    derived_qty=door_frames,
                    variance_pct=-100.0,
                    severity=VarianceSeverity.HIGH,
                    unit="nos",
                    explanation=(
                        f"Found {door_frames:.0f} door frames but no door "
                        f"shutter items. Shutters may be in a separate "
                        f"schedule or missing from BOQ."
                    ),
                    is_standard="IS 1200 Part 10",
                ))
            elif door_shutters > 0 and door_frames == 0:
                results.append(CrossCheckResult(
                    check_type="door_frame_shutter",
                    item_description="Door frames missing",
                    boq_qty=door_shutters,
                    derived_qty=0,
                    variance_pct=100.0,
                    severity=VarianceSeverity.HIGH,
                    unit="nos",
                    explanation=(
                        f"Found {door_shutters:.0f} door shutters but no "
                        f"door frame items. Frames may be included in "
                        f"another line item or missing."
                    ),
                    is_standard="IS 1200 Part 10",
                ))

            # Hardware check (only if doors are found)
            expected_hardware = max_doors
            if door_hardware > 0:
                var = _variance_pct(door_hardware, expected_hardware)
                severity = _classify_severity(var, "doors_windows")
                results.append(CrossCheckResult(
                    check_type="door_hardware",
                    item_description="Door hardware sets vs door count",
                    boq_qty=door_hardware,
                    derived_qty=expected_hardware,
                    variance_pct=var,
                    severity=severity,
                    unit="nos/sets",
                    explanation=(
                        f"Doors identified: {expected_hardware:.0f}. "
                        f"Hardware items: {door_hardware:.0f}. "
                        f"Each door typically needs 1 set of hardware. "
                        f"Variance: {var:+.1f}%."
                    ),
                    is_standard="IS 1200 Part 10",
                ))
            elif max_doors >= 1:
                # Hardware completely missing
                results.append(CrossCheckResult(
                    check_type="door_hardware",
                    item_description="Door hardware missing",
                    boq_qty=0,
                    derived_qty=expected_hardware,
                    variance_pct=-100.0,
                    severity=VarianceSeverity.MEDIUM,
                    unit="nos/sets",
                    explanation=(
                        f"Found {max_doors:.0f} doors but no hardware items "
                        f"(aldrop, tower bolt, lock, handle). Hardware may "
                        f"be included as a lump sum or missing."
                    ),
                    is_standard="IS 1200 Part 10",
                ))

    # Window frame vs shutter check
    if window_frames > 0 and window_shutters > 0:
        var = _variance_pct(window_shutters, window_frames)
        severity = _classify_severity(var, "doors_windows")
        results.append(CrossCheckResult(
            check_type="window_frame_shutter",
            item_description="Window frames vs window shutters/glass",
            boq_qty=window_shutters,
            derived_qty=window_frames,
            variance_pct=var,
            severity=severity,
            unit="nos",
            explanation=(
                f"Window frames: {window_frames:.0f} nos. "
                f"Window shutters/glass: {window_shutters:.0f} nos. "
                f"Variance: {var:+.1f}%."
            ),
            is_standard="IS 1200 Part 10",
        ))

    return results


def _check_waterproofing_coverage(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate waterproofing quantities against wet area indicators.

    In Indian construction, waterproofing is required for:
        - Toilet/bathroom floors and walls (up to 1.5m height)
        - Kitchen wet areas
        - Terraces and balconies
        - Basements

    This check verifies:
    1. If wet area items (toilets, bathrooms) exist in BOQ, waterproofing
       items should also exist.
    2. If slab area is known and waterproofing exists, the WP area should be
       at least 15% of slab area (typical terrace proportion).

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult.
    """
    results: List[CrossCheckResult] = []

    wp_area = 0.0
    wet_area_count = 0
    slab_rcc_vol = 0.0
    has_toilet_items = False
    has_terrace_items = False

    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)

        if qty > 0:
            if _WATERPROOFING_RE.search(desc) and _is_area_unit(unit):
                wp_area += qty

            if _WET_AREA_RE.search(desc):
                wet_area_count += 1
                if re.search(r'\b(toilet|bathroom|wash\s*room|lavatory|wc)\b', desc):
                    has_toilet_items = True
                if re.search(r'\b(terrace|balcony)\b', desc):
                    has_terrace_items = True

            # Track slab RCC volume for terrace WP estimation
            if _RCC_RE.search(desc) and "slab" in desc and _is_volume_unit(unit):
                slab_rcc_vol += qty

    # Check 1: Wet areas exist but no waterproofing
    if wet_area_count > 0 and wp_area == 0:
        results.append(CrossCheckResult(
            check_type="waterproofing_coverage",
            item_description="Waterproofing missing for wet areas",
            boq_qty=0,
            derived_qty=wet_area_count,
            variance_pct=-100.0,
            severity=VarianceSeverity.HIGH,
            unit="sqm",
            explanation=(
                f"Found {wet_area_count} BOQ items referencing wet areas "
                f"(toilet, bathroom, terrace, kitchen) but no waterproofing "
                f"items. WP treatment is typically mandatory for these areas."
            ),
            is_standard="IS 3067 (Waterproofing)",
        ))

    # Check 2: If we know slab volume and WP area exists, sanity-check coverage
    if slab_rcc_vol > 0 and wp_area > 0:
        # Estimate slab area from volume (assume 150mm avg slab thickness)
        estimated_slab_area = slab_rcc_vol / 0.15
        # Terrace + toilet WP should be at least 15% of slab area
        min_expected_wp = estimated_slab_area * DERIVED_RATIOS["wp_terrace_pct_of_slab"]

        var = _variance_pct(wp_area, min_expected_wp)
        severity = _classify_severity(var, "waterproofing")

        results.append(CrossCheckResult(
            check_type="waterproofing_coverage",
            item_description="Waterproofing area vs estimated requirement",
            boq_qty=wp_area,
            derived_qty=min_expected_wp,
            variance_pct=var,
            severity=severity,
            unit="sqm",
            explanation=(
                f"Estimated total slab area: {estimated_slab_area:.0f} sqm "
                f"(from {slab_rcc_vol:.1f} cum RCC slab at 150mm thick). "
                f"Minimum WP area (15% of slab for terrace + wet areas): "
                f"{min_expected_wp:.0f} sqm. "
                f"BOQ has {wp_area:.0f} sqm WP. "
                f"Variance: {var:+.1f}%."
            ),
            is_standard="IS 3067 (Waterproofing)",
        ))

    return results


def _check_earthwork_vs_foundation(boq_items: List[Dict]) -> List[CrossCheckResult]:
    """
    Cross-validate excavation volume against foundation concrete volumes.

    Excavation volume should be greater than or equal to foundation concrete
    (RCC + PCC) volume, because:
        1. You must excavate at least as much as you fill with concrete.
        2. In practice, excavation is larger due to working space, side slopes,
           and the void that remains around the foundation.

    Typical relationship:
        excavation_volume >= (foundation_RCC + PCC) * 1.5 to 2.5

    If excavation < foundation concrete, it is a strong indicator of an error.

    Args:
        boq_items: List of BOQ item dicts.

    Returns:
        List of CrossCheckResult.
    """
    results: List[CrossCheckResult] = []

    earthwork_vol = 0.0
    foundation_rcc_vol = 0.0
    pcc_vol = 0.0

    for item in boq_items:
        desc = _get_description(item)
        unit = _get_unit(item)
        qty = _extract_qty(item)

        if qty > 0 and _is_volume_unit(unit):
            if _EARTHWORK_RE.search(desc):
                earthwork_vol += qty
            if _RCC_RE.search(desc) and _FOUNDATION_RE.search(desc):
                foundation_rcc_vol += qty
            if _PCC_RE.search(desc):
                pcc_vol += qty

    total_foundation_concrete = foundation_rcc_vol + pcc_vol

    if earthwork_vol <= 0 or total_foundation_concrete <= 0:
        return results

    # Excavation should be at least 1.5x the concrete volume
    # (working space + side clearance + leveling course + backfill void)
    min_expected_earthwork = total_foundation_concrete * 1.5

    if earthwork_vol < total_foundation_concrete:
        # Earthwork is less than concrete itself — clear error
        var = _variance_pct(earthwork_vol, total_foundation_concrete)
        results.append(CrossCheckResult(
            check_type="earthwork_vs_foundation",
            item_description="Excavation less than foundation concrete",
            boq_qty=earthwork_vol,
            derived_qty=total_foundation_concrete,
            variance_pct=var,
            severity=VarianceSeverity.CRITICAL,
            unit="cum",
            explanation=(
                f"Earthwork excavation: {earthwork_vol:.1f} cum. "
                f"Foundation concrete (RCC + PCC): "
                f"{total_foundation_concrete:.1f} cum. "
                f"Excavation MUST be >= concrete volume (you cannot pour "
                f"concrete in a hole smaller than the pour). "
                f"Likely error in earthwork quantities."
            ),
            is_standard="IS 1200 Part 1",
        ))
    else:
        # Check against the 1.5x minimum
        var = _variance_pct(earthwork_vol, min_expected_earthwork)
        severity = _classify_severity(var, "rcc_concrete")

        # If earthwork is dramatically higher than expected, also flag it
        # (could indicate scope creep or over-estimation)
        max_reasonable = total_foundation_concrete * 3.0
        if earthwork_vol > max_reasonable:
            var_high = _variance_pct(earthwork_vol, max_reasonable)
            severity = _classify_severity(var_high, "rcc_concrete")
            results.append(CrossCheckResult(
                check_type="earthwork_vs_foundation",
                item_description="Excavation significantly exceeds foundation",
                boq_qty=earthwork_vol,
                derived_qty=max_reasonable,
                variance_pct=var_high,
                severity=severity,
                unit="cum",
                explanation=(
                    f"Earthwork: {earthwork_vol:.1f} cum vs foundation "
                    f"concrete {total_foundation_concrete:.1f} cum. "
                    f"Ratio: {earthwork_vol / total_foundation_concrete:.1f}x. "
                    f"Typical range: 1.5x-3.0x. Excess may indicate "
                    f"basement excavation, site leveling, or over-estimation."
                ),
                is_standard="IS 1200 Part 1",
            ))
        else:
            results.append(CrossCheckResult(
                check_type="earthwork_vs_foundation",
                item_description="Excavation vs foundation concrete",
                boq_qty=earthwork_vol,
                derived_qty=min_expected_earthwork,
                variance_pct=var,
                severity=severity,
                unit="cum",
                explanation=(
                    f"Earthwork: {earthwork_vol:.1f} cum. Foundation concrete "
                    f"(RCC + PCC): {total_foundation_concrete:.1f} cum. "
                    f"Ratio: {earthwork_vol / total_foundation_concrete:.1f}x "
                    f"(typical: 1.5x-3.0x). "
                    f"Variance from 1.5x baseline: {var:+.1f}%."
                ),
                is_standard="IS 1200 Part 1",
            ))

    return results


# =============================================================================
# PUBLIC API
# =============================================================================


def cross_check_boq(boq_items: List[Dict]) -> CrossCheckReport:
    """
    Run all cross-validation checks on BOQ items.

    Executes seven independent cross-checks against the BOQ item list:
        1. Formwork area vs RCC volume (by element type)
        2. Steel reinforcement vs RCC volume (by element type)
        3. Painting area vs plaster area
        4. Skirting running metres vs flooring area
        5. Door/window component consistency (frame/shutter/hardware)
        6. Waterproofing coverage vs wet area indicators
        7. Earthwork excavation vs foundation concrete volume

    Each check produces zero or more CrossCheckResult entries with severity
    classification based on per-material thresholds.

    Args:
        boq_items: List of BOQ item dicts. Each dict should have at minimum:
            - description (str): Item description text.
            - quantity / qty / final_qty (float): Item quantity.
            - unit / uom (str): Unit of measurement.

    Returns:
        CrossCheckReport with all findings, confidence score, and summary.
    """
    if not boq_items:
        logger.warning("cross_check_boq called with empty boq_items list")
        return CrossCheckReport()

    report = CrossCheckReport()

    report.checks.extend(_check_formwork_vs_rcc(boq_items))
    report.checks.extend(_check_steel_vs_rcc(boq_items))
    report.checks.extend(_check_paint_vs_plaster(boq_items))
    report.checks.extend(_check_skirting_vs_flooring(boq_items))
    report.checks.extend(_check_door_window_consistency(boq_items))
    report.checks.extend(_check_waterproofing_coverage(boq_items))
    report.checks.extend(_check_earthwork_vs_foundation(boq_items))

    logger.info(report.summary())
    return report
