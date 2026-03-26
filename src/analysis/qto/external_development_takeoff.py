"""
External Development Takeoff — Area-based QTO for campus external works.

Covers all works OUTSIDE the building footprint that appear in every Indian
campus tender (WAPCOS, EPI, CPWD, IIT, NIT) but are typically zero in raw
BOQ extraction:

  1. Compound wall (brick masonry, 3m high including foundation)
  2. Main gate + wicket gate (fixed campus fixtures)
  3. Internal roads (WBM + bituminous, 4.5m wide carriageway)
  4. Storm water drains (brick masonry U-drain, 450×450mm)
  5. Water supply (GI pipe network, OHT, sump)
  6. Sewerage (PVC pipe network, STP)
  7. External electrification (HT/LT, transformer, cable trench)
  8. Landscaping / horticulture (topsoil, plantation)

All norms follow CPWD DSR 2023 and Indian campus planning practice.
Quantities are rule-of-thumb; verify against site survey before submission.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Site area auto-derivation: buildings typically cover ~29% of campus site.
# footprint = total_area / floors  →  site = footprint × 3.5
_SITE_FROM_FOOTPRINT_FACTOR: float = 3.5

# Compound wall: 230mm thick × 3.0m high (wall body only; foundation extra)
_COMPOUND_WALL_THICKNESS_M: float = 0.230
_COMPOUND_WALL_HEIGHT_M: float = 3.0   # height including coping

# Road parameters
_ROAD_WIDTH_M: float = 4.5             # carriageway width
_ROAD_LENGTH_FACTOR: float = 0.8       # road length = sqrt(site_area) × 0.8

# Storm water drain: 450×450mm U-drain alongside roads + periphery
_DRAIN_SECTION_M: float = 0.450        # internal width = internal depth = 0.45m
_DRAIN_WALL_THICKNESS_M: float = 0.115 # half-brick side walls
_DRAIN_BASE_THICKNESS_M: float = 0.075 # PCC base
_DRAIN_LENGTH_FACTOR: float = 1.2      # drain length = road_length × 1.2

# Water supply
_WS_PIPE_LENGTH_FACTOR: float = 1.5    # GI pipe length = sqrt(site_area) × 1.5
_LPCD_WATER: int = 135                 # litres/person/day (IS 1172)
_OHT_ROUND_KL: int = 5                 # round OHT capacity to nearest 5KL

# Sewerage
_SWR_PIPE_LENGTH_FACTOR: float = 1.5   # PVC sewer pipe = sqrt(site_area) × 1.5
_LPCD_SEWAGE: float = 85.0             # litres/person/day sewage generation
_STP_ROUND_KLD: int = 5                # round STP capacity to nearest 5KLD

# Electrification
_KVA_PER_SQM: float = 0.015            # transformer loading (KVA/sqm)
_KVA_ROUND: int = 100                  # round KVA to nearest 100
_LT_TRENCH_FACTOR: float = 2.0         # LT cable trench = sqrt(site_area) × 2.0

# Landscaping: minimum 20% of site must be green
_LANDSCAPING_MIN_FRACTION: float = 0.20


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ExternalDevResult:
    """Output of external development takeoff."""
    line_items: List[dict] = field(default_factory=list)
    site_area_sqm: float = 0.0
    compound_wall_rm: float = 0.0
    road_area_sqm: float = 0.0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# HELPERS
# =============================================================================

def _make_item(
    description: str,
    trade: str,
    unit: str,
    quantity: float,
) -> dict:
    """Build a BOQ line item dict in the standard xBOQ format."""
    qty_rounded = round(quantity, 3)
    return {
        "description": description,
        "trade": trade,
        "unit": unit,
        "qty":      qty_rounded,    # canonical key for rate_engine.apply_rates()
        "quantity": qty_rounded,    # backward compat for UI tabs
        "source": "norm_based",
        "building": "Site Development",
    }


def _round_to_nearest(value: float, multiple: int) -> int:
    """Round *value* up to the nearest *multiple* (always >= 1 × multiple)."""
    if value <= 0:
        return multiple
    return max(multiple, int(math.ceil(value / multiple) * multiple))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_external_dev_takeoff(
    total_area_sqm: float,
    floors: int,
    occupancy: int = 0,
    site_area_sqm: float = 0.0,
    building_count: int = 1,
    building_type: str = "hostel",
) -> ExternalDevResult:
    """
    Estimate external development works for an Indian campus tender.

    Parameters
    ----------
    total_area_sqm : float
        Total built-up area across all floors (sqm).
    floors : int
        Average number of floors (used to derive building footprint).
    occupancy : int
        Total campus occupancy (students + staff). Used for utility sizing.
        If 0, a norm-based estimate is derived from total_area_sqm.
    site_area_sqm : float
        If known from tender document, pass it here.
        If 0 (default), auto-derived as: footprint × 3.5
        where footprint = total_area_sqm / floors.
    building_count : int
        Number of buildings on campus. Scales compound wall perimeter via a
        1.2 density factor per additional building.
    building_type : str
        Dominant campus building type. Accepted: "hostel", "academic",
        "residential", "office". Affects landscaping description only.

    Returns
    -------
    ExternalDevResult
    """
    result = ExternalDevResult()

    # ── Input sanitisation ──────────────────────────────────────────────────
    total_area_sqm = max(float(total_area_sqm or 0.0), 0.0)
    floors = max(int(floors or 1), 1)
    occupancy = max(int(occupancy or 0), 0)
    site_area_sqm = max(float(site_area_sqm or 0.0), 0.0)
    building_count = max(int(building_count or 1), 1)
    building_type = (building_type or "hostel").strip().lower()

    if total_area_sqm <= 0:
        result.warnings.append(
            "total_area_sqm is zero — no external development items computed."
        )
        return result

    # ── Site area ────────────────────────────────────────────────────────────
    footprint_sqm = total_area_sqm / floors
    if site_area_sqm <= 0:
        site_area_sqm = footprint_sqm * _SITE_FROM_FOOTPRINT_FACTOR
        result.warnings.append(
            f"site_area_sqm not provided — auto-derived as "
            f"{total_area_sqm:.1f} / {floors} floors × {_SITE_FROM_FOOTPRINT_FACTOR} "
            f"= {site_area_sqm:.1f} sqm"
        )
    result.site_area_sqm = round(site_area_sqm, 2)

    # ── Occupancy fallback ────────────────────────────────────────────────────
    if occupancy == 0:
        # Norm: 10 sqm of BUA per person (conservative institutional estimate)
        occupancy = max(1, int(total_area_sqm / 10))
        result.warnings.append(
            f"occupancy=0 — estimated as {total_area_sqm:.0f} / 10 = {occupancy} persons"
        )

    sqrt_site = math.sqrt(site_area_sqm)

    # =========================================================================
    # 1. COMPOUND WALL
    # =========================================================================
    # Perimeter of site (assumed square) scaled by building_count density
    base_perimeter = 4.0 * sqrt_site
    # Each additional building adds 1.2× density factor
    compound_wall_rm = base_perimeter * (1.0 + (building_count - 1) * 0.2)
    compound_wall_rm = round(compound_wall_rm, 2)
    result.compound_wall_rm = compound_wall_rm

    # Volume of wall body: length × thickness × height
    wall_vol_cum = compound_wall_rm * _COMPOUND_WALL_THICKNESS_M * _COMPOUND_WALL_HEIGHT_M

    result.line_items.append(_make_item(
        description=(
            f"Compound / boundary wall — brick masonry in CM 1:6, "
            f"{int(_COMPOUND_WALL_THICKNESS_M * 1000)}mm thick × {_COMPOUND_WALL_HEIGHT_M:.1f}m high "
            f"(including 1.2m RCC strip foundation), plastered both faces with coping"
        ),
        trade="civil",
        unit="rm",
        quantity=compound_wall_rm,
    ))

    result.line_items.append(_make_item(
        description=(
            f"Brick masonry in CM 1:6 for compound wall body — "
            f"{int(_COMPOUND_WALL_THICKNESS_M * 1000)}mm thick × {_COMPOUND_WALL_HEIGHT_M:.1f}m high"
        ),
        trade="civil",
        unit="cum",
        quantity=wall_vol_cum,
    ))

    # =========================================================================
    # 2. MAIN GATE + WICKET GATE
    # =========================================================================
    result.line_items.append(_make_item(
        description=(
            "Providing and fixing main entrance gate — MS fabricated, "
            "double-leaf, 5.0m wide × 2.4m high, hot-dip galvanised and "
            "powder-coated finish, with security cabin provision"
        ),
        trade="external",
        unit="No",
        quantity=1.0,
    ))

    result.line_items.append(_make_item(
        description=(
            "Providing and fixing wicket / pedestrian gate adjacent to main gate — "
            "MS fabricated, single-leaf, 1.2m wide × 2.1m high, "
            "powder-coated finish with locking arrangement"
        ),
        trade="external",
        unit="No",
        quantity=1.0,
    ))

    # =========================================================================
    # 3. INTERNAL ROADS
    # =========================================================================
    road_length_m = sqrt_site * _ROAD_LENGTH_FACTOR
    road_area_sqm = road_length_m * _ROAD_WIDTH_M
    result.road_area_sqm = round(road_area_sqm, 2)

    # 3a. Road bed earthwork (200mm formation cutting/filling)
    road_bed_cum = road_area_sqm * 0.20   # 200mm formation
    result.line_items.append(_make_item(
        description=(
            f"Earthwork in cutting / filling for road formation, "
            f"{_ROAD_WIDTH_M:.1f}m wide carriageway, dressed to camber, "
            f"200mm compacted sub-grade, all leads"
        ),
        trade="civil",
        unit="cum",
        quantity=road_bed_cum,
    ))

    # 3b. WBM base (75mm compacted)
    wbm_area_sqm = road_area_sqm
    result.line_items.append(_make_item(
        description=(
            f"Water Bound Macadam (WBM) base course — 75mm compacted thickness, "
            f"including spreading, rolling and blinding with screenings, "
            f"{_ROAD_WIDTH_M:.1f}m wide, per MORTH/IRC SP:20"
        ),
        trade="civil",
        unit="sqm",
        quantity=wbm_area_sqm,
    ))

    # 3c. Bituminous carpet (25mm DBM + 20mm BC)
    result.line_items.append(_make_item(
        description=(
            f"Bituminous macadam (DBM) 25mm + bituminous concrete (BC) 20mm "
            f"surfacing on WBM base, {_ROAD_WIDTH_M:.1f}m wide carriageway, "
            f"including tack coat and prime coat, per MORTH spec"
        ),
        trade="civil",
        unit="sqm",
        quantity=road_area_sqm,
    ))

    # =========================================================================
    # 4. STORM WATER DRAINS
    # =========================================================================
    drain_length_m = road_length_m * _DRAIN_LENGTH_FACTOR

    # Brick masonry U-drain volume:
    # Each rm of drain = 2 side walls + base slab
    # Side wall (per side): thickness × height = 0.115m × 0.45m per rm
    # Base slab: (0.45 + 2×0.115) × 0.075 per rm = 0.68 × 0.075
    side_vol_per_rm = 2 * _DRAIN_WALL_THICKNESS_M * _DRAIN_SECTION_M
    base_vol_per_rm = (_DRAIN_SECTION_M + 2 * _DRAIN_WALL_THICKNESS_M) * _DRAIN_BASE_THICKNESS_M
    drain_masonry_cum = drain_length_m * (side_vol_per_rm + base_vol_per_rm)

    result.line_items.append(_make_item(
        description=(
            f"Brick masonry U-drain — {int(_DRAIN_SECTION_M * 1000)}mm × "
            f"{int(_DRAIN_SECTION_M * 1000)}mm internal section, "
            f"{int(_DRAIN_WALL_THICKNESS_M * 1000)}mm side walls in CM 1:4, "
            f"75mm PCC base, including excavation and RCC cover slabs"
        ),
        trade="civil",
        unit="rm",
        quantity=drain_length_m,
    ))

    result.line_items.append(_make_item(
        description=(
            f"Brick masonry in CM 1:4 for storm water U-drain — "
            f"side walls + base, {int(_DRAIN_SECTION_M * 1000)}mm section"
        ),
        trade="civil",
        unit="cum",
        quantity=drain_masonry_cum,
    ))

    # =========================================================================
    # 5. WATER SUPPLY
    # =========================================================================
    # OHT capacity: occupancy × 135 lpcd, round to nearest 5KL
    oht_litres = occupancy * _LPCD_WATER
    oht_kl = _round_to_nearest(oht_litres / 1000.0, _OHT_ROUND_KL)

    # Sump: 50% of OHT
    sump_kl = _round_to_nearest(oht_kl * 0.50, _OHT_ROUND_KL)

    # GI pipe network
    ws_pipe_length_m = sqrt_site * _WS_PIPE_LENGTH_FACTOR

    result.line_items.append(_make_item(
        description=(
            f"Overhead water tank (OHT) — RCC elevated/ground-level tank, "
            f"{oht_kl} KL capacity, including inlet, outlet, overflow, "
            f"vent, level indicator and chlorination arrangement"
        ),
        trade="mep",
        unit="KL",
        quantity=float(oht_kl),
    ))

    result.line_items.append(_make_item(
        description=(
            f"Underground sump / cistern — RCC, {sump_kl} KL capacity "
            f"(50% of OHT), including inlet, outlet, overflow and "
            f"access manholes with covers"
        ),
        trade="mep",
        unit="KL",
        quantity=float(sump_kl),
    ))

    result.line_items.append(_make_item(
        description=(
            "GI class-C pipe network — external campus water supply distribution, "
            "50-100mm dia as required, in trenches with sand bedding, "
            "including isolating valves, air valves and pressure testing"
        ),
        trade="mep",
        unit="rm",
        quantity=ws_pipe_length_m,
    ))

    # =========================================================================
    # 6. SEWERAGE
    # =========================================================================
    # STP capacity: occupancy × 85 lpcd, round to nearest 5KLD
    stp_litres = occupancy * _LPCD_SEWAGE
    stp_kld = _round_to_nearest(stp_litres / 1000.0, _STP_ROUND_KLD)

    swr_pipe_length_m = sqrt_site * _SWR_PIPE_LENGTH_FACTOR

    result.line_items.append(_make_item(
        description=(
            f"Sewage treatment plant (STP) — package type, MBR/SBR technology, "
            f"{stp_kld} KLD capacity, inlet screening, aeration, secondary "
            f"clarifier, disinfection and treated water reuse provision, "
            f"CPCB Class A standards"
        ),
        trade="mep",
        unit="KLD",
        quantity=float(stp_kld),
    ))

    result.line_items.append(_make_item(
        description=(
            "PVC SWR pipe network — external campus sewerage collection system, "
            "110-200mm dia as required, in trenches with granular bedding, "
            "including manholes at 30m centres, inspection chambers"
        ),
        trade="mep",
        unit="rm",
        quantity=swr_pipe_length_m,
    ))

    # =========================================================================
    # 7. EXTERNAL ELECTRIFICATION
    # =========================================================================
    # Transformer KVA: total_area × 0.015 KVA/sqm, round to nearest 100KVA
    transformer_kva_raw = total_area_sqm * _KVA_PER_SQM
    transformer_kva = _round_to_nearest(transformer_kva_raw, _KVA_ROUND)

    lt_trench_length_m = sqrt_site * _LT_TRENCH_FACTOR

    result.line_items.append(_make_item(
        description=(
            "HT power supply arrangement — HT (11KV) cable laying from "
            "nearest source, metering cubicle, surge arrestors and "
            "earthing as per CEA regulations"
        ),
        trade="electrical",
        unit="ls",
        quantity=1.0,
    ))

    result.line_items.append(_make_item(
        description=(
            f"Supply, installation, testing and commissioning of "
            f"oil-cooled distribution transformer — {transformer_kva} KVA, "
            f"11KV/415V, ONAN cooled, with HT/LT switchgear, metering "
            f"panel and transformer yard civil works"
        ),
        trade="electrical",
        unit="No",
        quantity=1.0,
    ))

    result.line_items.append(_make_item(
        description=(
            f"LT cable trench — 600mm wide × 900mm deep, brick-lined, "
            f"sand bedded, with RCC cover slabs, for LT distribution "
            f"network throughout campus"
        ),
        trade="electrical",
        unit="rm",
        quantity=lt_trench_length_m,
    ))

    result.line_items.append(_make_item(
        description=(
            "External campus lighting — street light poles (9m GI octagonal), "
            "LED luminaires 80W, underground feeder cable, control panel "
            "with photocell and timer, earthing"
        ),
        trade="electrical",
        unit="ls",
        quantity=1.0,
    ))

    # =========================================================================
    # 8. LANDSCAPING / HORTICULTURE
    # =========================================================================
    landscape_sqm = site_area_sqm - footprint_sqm - road_area_sqm
    min_landscape_sqm = site_area_sqm * _LANDSCAPING_MIN_FRACTION
    landscape_sqm = max(landscape_sqm, min_landscape_sqm)
    landscape_sqm = round(landscape_sqm, 2)

    result.line_items.append(_make_item(
        description=(
            f"Topsoil filling and preparation for landscaping — 300mm deep "
            f"fertile topsoil, tilling, levelling and consolidation "
            f"before plantation"
        ),
        trade="landscaping",
        unit="sqm",
        quantity=landscape_sqm,
    ))

    result.line_items.append(_make_item(
        description=(
            "Horticulture and plantation — lawn turfing, avenue trees "
            "(native species 3m ht), shrubs, ground cover plants, "
            "garden path edging with brick and drip irrigation network"
        ),
        trade="landscaping",
        unit="sqm",
        quantity=landscape_sqm,
    ))

    logger.info(
        "external_dev_takeoff complete | items=%d | site=%.0f sqm | "
        "wall=%.1f rm | road=%.1f sqm | OHT=%d KL | STP=%d KLD | xfmr=%d KVA",
        len(result.line_items),
        result.site_area_sqm,
        result.compound_wall_rm,
        result.road_area_sqm,
        oht_kl,
        stp_kld,
        transformer_kva,
    )

    return result
