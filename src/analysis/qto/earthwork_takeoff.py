"""
Earthwork Takeoff — Area-based QTO for ground-up construction projects.

Computes earthwork and substructure preparation quantities from building
footprint data for:
  - Excavation in ordinary soil for foundations
  - Filling in foundation trenches with earth
  - Disposal of surplus excavated earth
  - Anti-termite treatment to soil (pre-construction)
  - PCC 1:4:8 bed in foundation
  - Basement excavation (if applicable)

All quantities follow CPWD Schedule of Rates norms and Indian construction
practice. The module gracefully degrades on zero or None inputs — returning
empty items rather than raising errors.

Design constraints:
- NO cv2 / OpenCV — pure arithmetic.
- Graceful None handling throughout.
- Generates items in the same dict format as other QTO modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Spread / over-excavation factor for foundation trenches
# (accounts for working space around footing edges)
_FOUNDATION_SPREAD_FACTOR: float = 1.3

# Fraction of excavation volume used for backfilling trenches
_BACKFILL_FRACTION: float = 0.70

# Fraction of excavation volume disposed off-site
_DISPOSAL_FRACTION: float = 0.30

# Bulkage factor — converts in-situ volume to loose (bank) measure
_BULKAGE_FACTOR: float = 1.15

# PCC bed thickness (metres) — 75 mm lean concrete bed
_PCC_THICKNESS_M: float = 0.075

# Basement over-excavation factor (less working space needed vs strip footings)
_BASEMENT_SPREAD_FACTOR: float = 1.1

# Soil type descriptions for item description
_SOIL_TYPE_LABELS: dict = {
    "ordinary":   "ordinary soil",
    "hard":       "hard soil / murrum",
    "soft rock":  "soft rock",
    "hard rock":  "hard rock",
    "waterlogged": "waterlogged / marshy soil",
}


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class EarthworkResult:
    line_items:           List[dict] = field(default_factory=list)
    mode:                 str = "area_estimate"
    warnings:             List[str] = field(default_factory=list)
    total_excavation_cum: float = 0.0   # total in-situ excavation volume (m³)
    total_filling_cum:    float = 0.0   # total backfilling volume (m³)


# =============================================================================
# ITEM BUILDER
# =============================================================================

def _make_item(
    description: str,
    qty: float,
    unit: str,
) -> dict:
    return {
        "description": description,
        "trade":       "earthwork",
        "unit":        unit,
        "qty":         round(qty, 2),
        "source":      "earthwork_takeoff",
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_earthwork_takeoff(
    floor_area_sqm: float,
    floors: int = 1,
    has_basement: bool = False,
    basement_depth_m: float = 3.0,
    foundation_depth_m: float = 1.5,
    soil_type: str = "ordinary",
    _footprint_is_total_bua: bool = False,
) -> EarthworkResult:
    """
    Compute earthwork and substructure preparation quantities for a ground-up
    construction project using area-based rule-of-thumb norms.

    Args:
        floor_area_sqm:     Gross floor area of a single floor (sqm), OR total
                            built-up area when _footprint_is_total_bua=True.
                            This is used as the building footprint for
                            foundation excavation extents.
        floors:             Number of storeys above ground.  Not used directly
                            in excavation volumes but retained for context.
        has_basement:       Whether the building includes a basement level.
        basement_depth_m:   Depth of basement excavation below GL (metres).
                            Only used when has_basement is True.
        foundation_depth_m: Depth of strip/isolated foundation excavation
                            below GL (metres).  Default 1.5 m (typical
                            Indian residential/institutional).
        soil_type:          Soil classification string.  Affects item
                            description only.  Accepted values: "ordinary",
                            "hard", "soft rock", "hard rock", "waterlogged".
        _footprint_is_total_bua: When True, floor_area_sqm is treated as total
                            BUA and divided by floors to obtain the footprint.
                            Prevents the double-division bug when the pipeline
                            pre-divides and then this function divides again.

    Returns:
        EarthworkResult with line_items, mode, warnings, and summary totals.
    """
    # --- Input sanitisation ---------------------------------------------------
    floor_area_sqm   = max(float(floor_area_sqm or 0.0), 0.0)
    floors           = max(int(floors or 1), 1)
    basement_depth_m = max(float(basement_depth_m or 3.0), 0.0)
    foundation_depth_m = max(float(foundation_depth_m or 1.5), 0.0)
    soil_type        = (soil_type or "ordinary").strip().lower()

    # Convert total BUA to footprint if caller passes total BUA
    if _footprint_is_total_bua and floors > 1:
        floor_area_sqm = round(floor_area_sqm / floors, 2)

    result = EarthworkResult(mode="area_estimate")
    result.warnings.append(
        "Earthwork quantities are area-based estimates using rule-of-thumb "
        "factors (spread factor 1.3, bulkage 1.15). Verify against actual "
        "foundation drawing and soil investigation report before submission."
    )

    # Early exit on degenerate inputs
    if floor_area_sqm <= 0:
        logger.warning(
            "earthwork_takeoff: floor_area_sqm=%.2f — returning empty result.",
            floor_area_sqm,
        )
        result.warnings.append(
            "floor_area_sqm is zero or not provided; no quantities computed."
        )
        return result

    # Resolve soil type label for descriptions
    soil_label = _SOIL_TYPE_LABELS.get(soil_type, f"{soil_type} soil")

    # =========================================================================
    # FOUNDATION EXCAVATION
    # =========================================================================

    # In-situ excavation volume (m³)
    excavation_insitu = (
        floor_area_sqm * foundation_depth_m * _FOUNDATION_SPREAD_FACTOR
    )
    # Loose (bank) measure for billing
    excavation_loose = excavation_insitu * _BULKAGE_FACTOR

    backfill_vol  = excavation_insitu * _BACKFILL_FRACTION
    disposal_vol  = excavation_insitu * _DISPOSAL_FRACTION

    result.total_excavation_cum = round(excavation_insitu, 2)
    result.total_filling_cum    = round(backfill_vol, 2)

    items: List[dict] = []

    # Item 1 — Foundation excavation
    items.append(_make_item(
        description=(
            f"Excavation in {soil_label} for foundations, "
            f"depth upto {foundation_depth_m:.1f} m, "
            "all leads and lifts, dressing sides and bottom"
        ),
        qty=excavation_loose,
        unit="cum",
    ))

    # Item 2 — Backfilling
    items.append(_make_item(
        description=(
            "Filling in foundation trenches and plinth with good earth, "
            "watering, ramming and consolidating in 150 mm layers"
        ),
        qty=backfill_vol,
        unit="cum",
    ))

    # Item 3 — Surplus disposal
    items.append(_make_item(
        description=(
            "Disposal of surplus excavated earth by mechanical means "
            "(lead upto 50 m, lift upto 1.5 m)"
        ),
        qty=disposal_vol,
        unit="cum",
    ))

    # Item 4 — Anti-termite treatment
    items.append(_make_item(
        description=(
            "Anti-termite treatment to soil — pre-construction stage, "
            "chemical emulsion treatment to bottom and sides of excavation "
            "and to filled earth around foundation (IS 6313 Part 2)"
        ),
        qty=floor_area_sqm,
        unit="sqm",
    ))

    # Item 5 — PCC lean concrete bed
    pcc_vol = floor_area_sqm * _PCC_THICKNESS_M
    items.append(_make_item(
        description=(
            f"PCC (Plain cement concrete) M10 1:3:6 in foundation "
            f"bed, {int(_PCC_THICKNESS_M * 1000)} mm thick, including shuttering"
        ),
        qty=pcc_vol,
        unit="cum",
    ))

    # =========================================================================
    # BASEMENT EXCAVATION (optional)
    # =========================================================================

    if has_basement:
        if basement_depth_m <= 0:
            result.warnings.append(
                "has_basement=True but basement_depth_m is zero; "
                "basement excavation item skipped."
            )
        else:
            basement_excavation = (
                floor_area_sqm * basement_depth_m * _BASEMENT_SPREAD_FACTOR
            )
            basement_loose = basement_excavation * _BULKAGE_FACTOR
            result.total_excavation_cum = round(
                result.total_excavation_cum + basement_excavation, 2
            )
            items.append(_make_item(
                description=(
                    f"Excavation in {soil_label} for basement, "
                    f"depth upto {basement_depth_m:.1f} m below GL, "
                    "all leads and lifts, shoring / strutting as required, "
                    "dressing sides and bottom"
                ),
                qty=basement_loose,
                unit="cum",
            ))

    # Filter out any zero-quantity items (defensive)
    result.line_items = [i for i in items if i["qty"] > 0]

    logger.info(
        "earthwork_takeoff complete | mode=%s | items=%d | "
        "excavation=%.1f cum | filling=%.1f cum",
        result.mode,
        len(result.line_items),
        result.total_excavation_cum,
        result.total_filling_cum,
    )

    return result
