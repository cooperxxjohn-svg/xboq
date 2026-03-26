"""
Sitework Takeoff — Area-based civil and site work quantity estimation.

Algorithm:
1. Accept plot area + building footprint + total floor area
2. Derive missing dimensions from standard Indian coverage ratios
3. Generate BOQ line items for all civil/site work trades:
   earthwork, PCC, anti-termite, paving, compound wall, drainage, etc.

Design constraints:
- Always runs in area_estimate mode — no schedule text to parse for site work
- Quantities derived from IS codes and Indian construction norms
- Compatible with rate_engine.apply_rates() for costing
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Typical Indian plot coverage ratio: built area / plot area ≈ 0.40
_DEFAULT_COVERAGE_RATIO = 0.40

# Inverse: if only footprint known, plot ≈ footprint / 0.40 = footprint × 2.5
_PLOT_FROM_FOOTPRINT_FACTOR = 2.5

# Bulkage factor for excavated earth (loose / bank measure ratio)
_BULKAGE_FACTOR = 1.15

# Basement depth assumption (metres)
_BASEMENT_DEPTH_M = 3.0

# Foundation bulk excavation depth assumption (metres)
_FOUNDATION_EXCAVATION_DEPTH_M = 1.50

# Topsoil strip depth (metres)
_TOPSOIL_DEPTH_M = 0.30

# Hardcore filling under foundation (metres)
_HARDCORE_DEPTH_M = 0.30

# PCC 1:4:8 thickness under foundation (metres)
_PCC_THICKNESS_M = 0.075

# External paving fraction of open area
_PAVING_FRACTION = 0.40

# Landscaping fraction of open area (residential only)
_LANDSCAPE_FRACTION = 0.30

# Compound wall height factor: 0.75 × perimeter (one side open for access)
_COMPOUND_WALL_FACTOR = 0.75

# RCC sump tank size options by building type (number of tanks)
_SUMP_COUNT = {"residential": 1, "commercial": 2, "industrial": 2}

# Overhead tank: 1 per 4 floors (HDPE 5000L)
_OHT_FLOORS_PER_TANK = 4

TRADE = "Civil / Site Work"


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class SiteworkResult:
    """Output of sitework area-based QTO."""
    line_items: List[dict]
    mode: str               # always "area_estimate"
    warnings: List[str]
    plot_area_sqm: float
    built_area_sqm: float


# =============================================================================
# HELPER — BOQ ITEM BUILDER
# =============================================================================

def _item(
    description: str,
    unit: str,
    qty: float,
    spec: str = "",
    section: str = "SITE WORK",
    confidence: float = 0.70,
) -> dict:
    """Build a BOQ line item dict compatible with the wider xBOQ item schema."""
    return {
        "item_no":          None,
        "description":      description + (f" (Ref: {spec})" if spec else ""),
        "unit":             unit,
        "unit_inferred":    False,
        "qty":              round(qty, 3),
        "rate":             None,
        "trade":            TRADE,
        "section":          section,
        "source_page":      0,
        "source":           "qto_sitework",
        "confidence":       confidence,
        "is_priceable":     True,
        "priceable_reason": "priceable",
        "qto_method":       "area_estimate",
    }


# =============================================================================
# CORE ESTIMATION LOGIC
# =============================================================================

def _estimate_plot_area(built_area_sqm: float) -> float:
    """Estimate plot area from footprint using standard Indian coverage ratio."""
    return round(built_area_sqm * _PLOT_FROM_FOOTPRINT_FACTOR, 2)


def _estimate_built_area(total_floor_area_sqm: float, floors: int) -> float:
    """Estimate ground-floor footprint from total BUA and number of floors."""
    floors = max(1, floors)
    return round(total_floor_area_sqm / floors, 2)


def _plot_perimeter(plot_area_sqm: float) -> float:
    """Perimeter of an assumed-square plot (metres)."""
    return round(4.0 * math.sqrt(plot_area_sqm), 2)


# =============================================================================
# FULL PIPELINE ENTRY POINT
# =============================================================================

def run_sitework_takeoff(
    plot_area_sqm: float,
    built_area_sqm: float,
    total_floor_area_sqm: float,
    floors: int = 1,
    has_basement: bool = False,
    building_type: str = "residential",
) -> SiteworkResult:
    """
    Generate site work and civil BOQ quantities from building area parameters.

    Args:
        plot_area_sqm:          Total site / plot area (sqm).
        built_area_sqm:         Ground-floor footprint area (sqm).
        total_floor_area_sqm:   All floors combined built-up area (sqm).
        floors:                 Number of above-ground floors (default 1).
        has_basement:           Include basement excavation if True.
        building_type:          "residential" | "commercial" | "industrial".

    Returns:
        SiteworkResult with line_items, warnings and resolved area values.
    """
    warnings: List[str] = []
    items: List[dict] = []

    # ── 1. Resolve missing area inputs ───────────────────────────────────────
    plot_area_sqm = max(0.0, plot_area_sqm)
    built_area_sqm = max(0.0, built_area_sqm)
    total_floor_area_sqm = max(0.0, total_floor_area_sqm)
    floors = max(1, int(floors))

    if built_area_sqm == 0.0 and total_floor_area_sqm > 0.0:
        built_area_sqm = _estimate_built_area(total_floor_area_sqm, floors)
        warnings.append(
            f"built_area_sqm not provided — estimated as "
            f"{total_floor_area_sqm:.1f} / {floors} floors = {built_area_sqm:.1f} sqm"
        )

    if plot_area_sqm == 0.0 and built_area_sqm > 0.0:
        plot_area_sqm = _estimate_plot_area(built_area_sqm)
        warnings.append(
            f"plot_area_sqm not provided — estimated as footprint × "
            f"{_PLOT_FROM_FOOTPRINT_FACTOR} = {plot_area_sqm:.1f} sqm "
            f"(assumes ~{int(_DEFAULT_COVERAGE_RATIO * 100)}% plot coverage)"
        )

    # ── 2. Guard: insufficient data ───────────────────────────────────────────
    if plot_area_sqm < 10.0 and built_area_sqm < 10.0:
        warnings.append(
            "Both plot_area_sqm and built_area_sqm are < 10 sqm — "
            "insufficient data for sitework takeoff. No items generated."
        )
        return SiteworkResult(
            line_items=[],
            mode="area_estimate",
            warnings=warnings,
            plot_area_sqm=plot_area_sqm,
            built_area_sqm=built_area_sqm,
        )

    open_area_sqm = max(0.0, plot_area_sqm - built_area_sqm)
    perimeter_lm = _plot_perimeter(plot_area_sqm)

    # ── 3. Generate BOQ items ─────────────────────────────────────────────────

    # 3.1  Site clearance and grubbing
    items.append(_item(
        description=(
            "Site clearance, grubbing up roots, removal of vegetation "
            "and surface debris, disposal off site"
        ),
        unit="sqm",
        qty=plot_area_sqm,
        spec="IS 3764",
        section="SITE PREPARATION",
        confidence=0.80,
    ))

    # 3.2  Topsoil excavation 300mm
    topsoil_cum = round(built_area_sqm * _TOPSOIL_DEPTH_M, 3)
    items.append(_item(
        description=(
            f"Excavation and removal of topsoil to a depth of "
            f"{int(_TOPSOIL_DEPTH_M * 1000)}mm, stockpile or dispose as directed"
        ),
        unit="cum",
        qty=topsoil_cum,
        section="EARTHWORK",
        confidence=0.75,
    ))

    # 3.3  Bulk excavation for foundations
    bulk_exc_cum = round(built_area_sqm * _FOUNDATION_EXCAVATION_DEPTH_M, 3)
    items.append(_item(
        description=(
            f"Bulk excavation for foundations to an average depth of "
            f"{_FOUNDATION_EXCAVATION_DEPTH_M:.1f}m in all types of soil, "
            f"including levelling of pit bottom"
        ),
        unit="cum",
        qty=bulk_exc_cum,
        section="EARTHWORK",
        confidence=0.70,
    ))

    # 3.4  Basement excavation (optional)
    basement_exc_cum = 0.0
    if has_basement:
        basement_exc_cum = round(built_area_sqm * _BASEMENT_DEPTH_M, 3)
        items.append(_item(
            description=(
                f"Additional bulk excavation for basement to "
                f"{_BASEMENT_DEPTH_M:.1f}m depth including dewatering "
                f"and shoring as required"
            ),
            unit="cum",
            qty=basement_exc_cum,
            section="EARTHWORK",
            confidence=0.65,
        ))
        warnings.append(
            f"Basement excavation added: {basement_exc_cum:.1f} cum "
            f"({_BASEMENT_DEPTH_M}m depth × {built_area_sqm:.1f} sqm footprint)"
        )

    # 3.5  Disposal of excavated earth (including bulkage)
    total_exc_cum = bulk_exc_cum + topsoil_cum + basement_exc_cum
    disposal_cum = round(total_exc_cum * _BULKAGE_FACTOR, 3)
    items.append(_item(
        description=(
            f"Loading, transporting and disposing of excavated earth off site "
            f"(bulkage factor {_BULKAGE_FACTOR}×)"
        ),
        unit="cum",
        qty=disposal_cum,
        section="EARTHWORK",
        confidence=0.70,
    ))

    # 3.6  Hardcore / broken stone filling under foundation
    hardcore_cum = round(built_area_sqm * _HARDCORE_DEPTH_M, 3)
    items.append(_item(
        description=(
            f"Providing and laying hardcore / broken stone filling "
            f"{int(_HARDCORE_DEPTH_M * 1000)}mm thick under foundation beds, "
            f"well compacted in layers"
        ),
        unit="cum",
        qty=hardcore_cum,
        section="FOUNDATION BEDS",
        confidence=0.75,
    ))

    # 3.7  PCC 1:4:8 under foundation
    pcc_cum = round(built_area_sqm * _PCC_THICKNESS_M, 3)
    items.append(_item(
        description=(
            f"Plain cement concrete (PCC) mix 1:4:8 (cement:sand:coarse "
            f"aggregate) {int(_PCC_THICKNESS_M * 1000)}mm thick blinding layer "
            f"under footings and foundation beds"
        ),
        unit="cum",
        qty=pcc_cum,
        section="FOUNDATION BEDS",
        confidence=0.80,
    ))

    # 3.8  Anti-termite treatment
    items.append(_item(
        description=(
            "Pre-construction anti-termite chemical soil treatment to "
            "foundation trenches, pit bottoms and plinth fill — "
            "approved chlorpyrifos / imidacloprid emulsion"
        ),
        unit="sqm",
        qty=built_area_sqm,
        spec="IS 6313",
        section="SITE PROTECTION",
        confidence=0.85,
    ))

    # 3.9  External paving / pathway
    paving_sqm = round(open_area_sqm * _PAVING_FRACTION, 2)
    if paving_sqm > 0:
        items.append(_item(
            description=(
                "Providing and laying interlocking concrete block paving "
                "60mm thick over 50mm sand bed for external pathways, "
                "parking apron and approach roads"
            ),
            unit="sqm",
            qty=paving_sqm,
            section="EXTERNAL PAVING",
            confidence=0.60,
        ))

    # 3.10  Compound wall
    compound_wall_lm = round(perimeter_lm * _COMPOUND_WALL_FACTOR, 2)
    items.append(_item(
        description=(
            "Providing and constructing compound / boundary wall "
            "2.1m high, 230mm thick brick/concrete block masonry with "
            "plastering and coping on top"
        ),
        unit="lm",
        qty=compound_wall_lm,
        section="BOUNDARY WORKS",
        confidence=0.65,
    ))

    # 3.11  Main gate (MS fabricated)
    items.append(_item(
        description=(
            "Providing and fixing main entrance gate — MS fabricated, "
            "double-leaf, 4.5m × 2.1m, powder-coated finish, with "
            "pedestrian wicket gate"
        ),
        unit="No",
        qty=1.0,
        section="BOUNDARY WORKS",
        confidence=0.90,
    ))

    # 3.12  Underground sump tank (RCC)
    sump_count = float(_SUMP_COUNT.get(building_type, 1))
    items.append(_item(
        description=(
            "Providing and constructing underground RCC sump tank "
            "10,000 litre capacity with inlet, outlet, overflow, "
            "vent and access manholes"
        ),
        unit="No",
        qty=sump_count,
        section="WATER SUPPLY",
        confidence=0.80,
    ))

    # 3.13  Overhead water tank (HDPE 5000L)
    oht_count = float(math.ceil(floors / _OHT_FLOORS_PER_TANK))
    items.append(_item(
        description=(
            "Providing and installing overhead HDPE water storage "
            "tank 5,000 litre capacity on RCC/MS frame support "
            "with ball valve, outlet and overflow connections"
        ),
        unit="No",
        qty=oht_count,
        section="WATER SUPPLY",
        confidence=0.80,
    ))

    # 3.14  Septic tank / STP
    if building_type == "residential":
        sewage_desc = (
            "Providing and constructing RCC septic tank with soak pit "
            "designed for building occupancy — two-chamber design with "
            "inspection chambers, per IS 2470"
        )
        sewage_spec = "IS 2470"
    else:
        sewage_desc = (
            "Providing, supplying and commissioning package sewage "
            "treatment plant (STP) for commercial building — "
            "MBR/MBBR technology, inlet screening, treated water "
            "reuse provision"
        )
        sewage_spec = "CPCB norms"

    items.append(_item(
        description=sewage_desc,
        unit="No",
        qty=1.0,
        spec=sewage_spec,
        section="DRAINAGE & SEWAGE",
        confidence=0.75,
    ))

    # 3.15  External stormwater drainage
    items.append(_item(
        description=(
            "Providing and laying external stormwater drainage — "
            "NP2 RCC hume pipes / UPVC pipes in trenches, with "
            "catch pits, inspection chambers and outfall connection"
        ),
        unit="lm",
        qty=perimeter_lm,
        section="DRAINAGE & SEWAGE",
        confidence=0.60,
    ))

    # 3.16  Garden / landscaping (residential only)
    if building_type == "residential" and open_area_sqm > 0:
        landscape_sqm = round(open_area_sqm * _LANDSCAPE_FRACTION, 2)
        if landscape_sqm > 0:
            items.append(_item(
                description=(
                    "Landscape development including topsoil preparation "
                    "300mm deep, turfing / planting, edge kerbs, "
                    "irrigation points and garden path finishing"
                ),
                unit="sqm",
                qty=landscape_sqm,
                section="LANDSCAPING",
                confidence=0.55,
            ))

    # ── 4. Assign sequential item numbers ─────────────────────────────────────
    for idx, it in enumerate(items, start=1):
        it["item_no"] = idx

    logger.info(
        "Sitework takeoff complete: %d items, plot=%.1f sqm, footprint=%.1f sqm",
        len(items), plot_area_sqm, built_area_sqm,
    )

    return SiteworkResult(
        line_items=items,
        mode="area_estimate",
        warnings=warnings,
        plot_area_sqm=plot_area_sqm,
        built_area_sqm=built_area_sqm,
    )
