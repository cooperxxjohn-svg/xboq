"""
Foundation Takeoff — rule-based BOQ generation for all Indian foundation types.

Handles strip, isolated pad, raft, and bored cast-in-situ pile foundations.
Foundation type can be specified explicitly or auto-selected from building type,
floor count, basement presence, and soil condition.

All quantities follow CPWD DSR 2023 norms and standard Indian QS practice.
Units: cum (cubic metres), sqm (square metres), rm (running metres), kg (kilograms).

Design constraints:
- NO cv2 / OpenCV — pure arithmetic.
- Graceful None / zero handling throughout.
- line_item dict format consistent with all other QTO modules.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Grid spacing assumption for column count (sqm per column)
_COLUMN_GRID_SQM: float = 25.0

# ── Strip foundation ──────────────────────────────────────────────────────────
_STRIP_PCC_THICKNESS_M:  float = 0.075   # PCC M10 bed thickness
_STRIP_RCC_THICKNESS_M:  float = 0.120   # RCC M20 strip footing depth
_STRIP_DPC_FACTOR:       float = 1.10    # DPC area = footprint * factor

# ── Isolated pad footing ─────────────────────────────────────────────────────
_PAD_PIT_WIDTH_M:        float = 2.0     # excavation pit width
_PAD_PIT_DEPTH_M:        float = 1.5     # excavation pit depth
_PAD_PCC_WIDTH_M:        float = 2.2     # PCC bed slightly larger than pad
_PAD_PCC_THICKNESS_M:    float = 0.075   # PCC M10 bed thickness
_PAD_RCC_WIDTH_M:        float = 2.0     # RCC pad plan dimension
_PAD_RCC_DEPTH_M:        float = 0.45    # RCC pad depth
_PAD_STEEL_KG_PER_CUM:  float = 110.0   # reinforcement intensity
_PAD_PEDESTAL_W_M:       float = 0.45    # pedestal width
_PAD_PEDESTAL_H_M:       float = 0.60    # pedestal height

# ── Raft foundation ───────────────────────────────────────────────────────────
_RAFT_DEPTH_DEFAULT_M:   float = 1.2     # default raft excavation depth
_RAFT_BULKAGE_FACTOR:    float = 1.30    # excavation bulkage
_RAFT_PCC_THICKNESS_M:   float = 0.10    # PCC M10 under raft
_RAFT_SLAB_THICKNESS_M:  float = 0.60    # default raft slab thickness
_RAFT_STEEL_KG_PER_CUM:  float = 130.0  # reinforcement intensity
_RAFT_WP_FACTOR:         float = 1.05    # waterproofing area = footprint * factor

# ── Bored cast-in-situ piles ─────────────────────────────────────────────────
_PILE_GROUP_FACTOR:      float = 1.30    # pile count = column_count * factor
_PILE_DIA_SMALL_MM:      int   = 450     # diameter for < 8 floors
_PILE_DIA_LARGE_MM:      int   = 600     # diameter for >= 8 floors
_PILE_DEPTH_DEFAULT_M:   float = 12.0   # default pile depth

# Pile depth auto-selection table: (floor_min, floor_max, soil_keywords, depth_m)
_PILE_DEPTH_TABLE: list = [
    (1,  5,  ("normal", "hard", "rock", "dense", "gravel"),  9.0),
    (1,  5,  ("fill", "clay", "soft", "loose", "waterlogged"), 15.0),
    (6,  9,  ("normal", "hard", "rock", "dense", "gravel"), 12.0),
    (6,  9,  ("fill", "clay", "soft", "loose", "waterlogged"), 18.0),
    (10, 15, ("normal", "hard", "rock", "dense", "gravel"), 15.0),
    (10, 15, ("fill", "clay", "soft", "loose", "waterlogged"), 22.0),
    (16, 99, ("normal", "hard", "rock", "dense", "gravel"), 20.0),
    (16, 99, ("fill", "clay", "soft", "loose", "waterlogged"), 28.0),
]
_PILE_STEEL_KG_PER_CUM:  float = 150.0  # reinforcement intensity
_PILE_CAP_W_M:           float = 1.5    # pile cap plan dimension
_PILE_CAP_H_M:           float = 0.75   # pile cap thickness
_PILE_CAP_GROUP_DIV:     float = 1.30   # shared cap factor (caps shared between columns)
_PILE_CAP_PCC_FACTOR:    float = 0.15   # PCC under pile cap = cap_cum * factor
_PILE_CUTOFF_M:          float = 0.60   # average cutoff length per pile


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class FoundationResult:
    line_items:      List[dict] = field(default_factory=list)
    foundation_type: str = ""          # selected foundation type
    pile_count:      int = 0           # 0 for non-pile foundations
    column_count:    int = 0
    warnings:        List[str] = field(default_factory=list)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _make_item(description: str, qty: float, unit: str) -> dict:
    """Build a line-item dict in the standard QTO format."""
    qty_rounded = round(qty, 2)
    return {
        "description": description,
        "trade":       "structural",
        "unit":        unit,
        "qty":         qty_rounded,    # canonical key for rate_engine.apply_rates()
        "quantity":    qty_rounded,    # backward compat for UI tabs
        "source":      "norm_based",
        "building":    "Foundation",
    }


def _auto_select_foundation(
    floors: int,
    building_type: str,
    has_basement: bool,
    soil_type: str,
) -> str:
    """Return the most appropriate foundation type from building parameters."""
    bt = building_type.lower().strip()
    st = soil_type.lower().strip()

    if bt in ("hospital", "aiims", "highrise") or floors >= 7:
        return "pile_bored"
    if has_basement or st in ("fill", "clay", "soft"):
        return "raft"
    if floors >= 4:
        return "isolated_pad"
    return "strip"


# =============================================================================
# FOUNDATION TYPE BUILDERS
# =============================================================================

def _build_strip(footprint: float, result: FoundationResult) -> None:
    """Add concrete/masonry items for strip foundation."""
    pcc_vol  = footprint * _STRIP_PCC_THICKNESS_M
    rcc_vol  = footprint * _STRIP_RCC_THICKNESS_M
    dpc_area = footprint * _STRIP_DPC_FACTOR

    result.line_items.append(_make_item(
        "PCC M10 (1:3:6) lean concrete bed in strip foundation, 75mm thick",
        pcc_vol, "cum",
    ))
    result.line_items.append(_make_item(
        "RCC M20 strip footing including shuttering and reinforcement",
        rcc_vol, "cum",
    ))
    result.line_items.append(_make_item(
        "DPC 75mm cement concrete 1:2:4 at plinth level over strip footing",
        dpc_area, "sqm",
    ))


def _build_isolated_pad(
    footprint: float,
    floor_area_sqm: float,
    result: FoundationResult,
) -> None:
    """Add all items for isolated pad (column) footings."""
    col_count = math.ceil(floor_area_sqm / _COLUMN_GRID_SQM)
    result.column_count = col_count

    # Excavation (loose measure, no bulkage factor — match earthwork norms)
    pit_vol_each = _PAD_PIT_WIDTH_M * _PAD_PIT_WIDTH_M * _PAD_PIT_DEPTH_M
    excavation   = col_count * pit_vol_each
    result.line_items.append(_make_item(
        f"Excavation for isolated pad footings {_PAD_PIT_WIDTH_M:.1f}m × "
        f"{_PAD_PIT_WIDTH_M:.1f}m × {_PAD_PIT_DEPTH_M:.1f}m deep, "
        "ordinary soil, all leads and lifts",
        excavation, "cum",
    ))

    # PCC M10 bed
    pcc_vol = col_count * (_PAD_PCC_WIDTH_M ** 2) * _PAD_PCC_THICKNESS_M
    result.line_items.append(_make_item(
        "PCC M10 (1:3:6) bed under isolated pad footing, 75mm thick",
        pcc_vol, "cum",
    ))

    # RCC M20 pad
    rcc_vol = col_count * (_PAD_RCC_WIDTH_M ** 2) * _PAD_RCC_DEPTH_M
    result.line_items.append(_make_item(
        "RCC M20 isolated pad footing including shuttering",
        rcc_vol, "cum",
    ))

    # Reinforcement steel
    steel_kg = rcc_vol * _PAD_STEEL_KG_PER_CUM
    result.line_items.append(_make_item(
        "Reinforcement steel Fe-500 in isolated pad footing",
        steel_kg, "kg",
    ))

    # Pedestal / column stub
    pedestal_vol = col_count * (_PAD_PEDESTAL_W_M ** 2) * _PAD_PEDESTAL_H_M
    result.line_items.append(_make_item(
        "RCC M25 column pedestal (stub) above pad footing up to plinth level",
        pedestal_vol, "cum",
    ))


def _build_raft(footprint: float, result: FoundationResult) -> None:
    """Add all items for raft foundation."""
    # Excavation
    excavation = footprint * _RAFT_DEPTH_DEFAULT_M * _RAFT_BULKAGE_FACTOR
    result.line_items.append(_make_item(
        f"Bulk excavation for raft foundation, depth {_RAFT_DEPTH_DEFAULT_M:.1f}m, "
        "including disposal of surplus earth",
        excavation, "cum",
    ))

    # PCC M10 blinding
    pcc_vol = footprint * _RAFT_PCC_THICKNESS_M
    result.line_items.append(_make_item(
        "PCC M10 (1:3:6) blinding / levelling course under raft slab, 100mm thick",
        pcc_vol, "cum",
    ))

    # RCC M25 raft slab
    rcc_vol = footprint * _RAFT_SLAB_THICKNESS_M
    result.line_items.append(_make_item(
        "RCC M25 raft slab including shuttering and two-way reinforcement",
        rcc_vol, "cum",
    ))

    # Reinforcement steel
    steel_kg = rcc_vol * _RAFT_STEEL_KG_PER_CUM
    result.line_items.append(_make_item(
        "Reinforcement steel Fe-500 in raft slab",
        steel_kg, "kg",
    ))

    # Waterproofing below raft
    wp_area = footprint * _RAFT_WP_FACTOR
    result.line_items.append(_make_item(
        "Waterproofing membrane below raft slab (SBS/APP torch-applied, "
        "min 3mm thick, lapped 150mm at joints)",
        wp_area, "sqm",
    ))


def _build_pile_bored(
    footprint: float,
    floor_area_sqm: float,
    floors: int,
    pile_depth_m: float,
    result: FoundationResult,
) -> None:
    """Add all items for bored cast-in-situ RCC pile foundation."""
    col_count  = math.ceil(floor_area_sqm / _COLUMN_GRID_SQM)
    pile_count = math.ceil(col_count * _PILE_GROUP_FACTOR)
    result.column_count = col_count
    result.pile_count   = pile_count

    # Pile diameter selection
    dia_mm = _PILE_DIA_LARGE_MM if floors >= 8 else _PILE_DIA_SMALL_MM
    dia_m  = dia_mm / 1000.0
    radius = dia_m / 2.0

    # Boring (running metres)
    boring_rm = pile_count * pile_depth_m
    result.line_items.append(_make_item(
        f"Boring for cast-in-situ RCC piles, diameter {dia_mm}mm, "
        f"depth upto {pile_depth_m:.0f}m, in ordinary/clay soil",
        boring_rm, "rm",
    ))

    # Concrete M25 in piles
    pile_conc_cum = pile_count * math.pi * (radius ** 2) * pile_depth_m
    result.line_items.append(_make_item(
        f"Concrete M25 in bored cast-in-situ piles {dia_mm}mm dia, "
        "including tremie/pump placement",
        pile_conc_cum, "cum",
    ))

    # Reinforcement in piles
    pile_steel_kg = pile_conc_cum * _PILE_STEEL_KG_PER_CUM
    result.line_items.append(_make_item(
        "Reinforcement steel Fe-500 in bored piles (cage assembly and lowering)",
        pile_steel_kg, "kg",
    ))

    # Pile cap RCC M25
    pile_cap_cum = (
        pile_count
        * (_PILE_CAP_W_M ** 2)
        * _PILE_CAP_H_M
        / _PILE_CAP_GROUP_DIV
    )
    result.line_items.append(_make_item(
        "RCC M25 pile cap including shuttering and reinforcement",
        pile_cap_cum, "cum",
    ))

    # PCC under pile cap
    pcc_under_cap = pile_cap_cum * _PILE_CAP_PCC_FACTOR
    result.line_items.append(_make_item(
        "PCC M10 (1:3:6) blinding course under pile caps, 75mm thick",
        pcc_under_cap, "cum",
    ))

    # Cutoff of pile heads
    cutoff_rm = pile_count * _PILE_CUTOFF_M
    result.line_items.append(_make_item(
        f"Cutting off pile heads to design cut-off level, {dia_mm}mm dia piles",
        cutoff_rm, "rm",
    ))


# =============================================================================
# PILE DEPTH AUTO-SELECTION
# =============================================================================

def _auto_pile_depth(floors: int, soil_type: str) -> float:
    """Select pile depth from standard table based on floor count and soil type."""
    soil_lower = (soil_type or "normal").lower().strip()
    for fmin, fmax, soil_kws, depth in _PILE_DEPTH_TABLE:
        if fmin <= floors <= fmax:
            if any(kw in soil_lower for kw in soil_kws):
                return depth
    return _PILE_DEPTH_DEFAULT_M


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_foundation_takeoff(
    floor_area_sqm: float,
    floors: int,
    building_type: str = "hostel",
    foundation_type: str = "auto",
    has_basement: bool = False,
    soil_type: str = "normal",
    pile_depth_m: float = 12.0,
) -> FoundationResult:
    """
    Auto-generate foundation BOQ items for a ground-up construction project.

    Parameters
    ----------
    floor_area_sqm : float
        Total gross floor area (all floors combined, sqm).
    floors : int
        Number of storeys above ground.
    building_type : str
        Affects auto-selection logic. Key values: "hospital", "aiims",
        "highrise", "hostel", "academic", "residential", etc.
    foundation_type : str
        "auto" triggers selection logic, or one of: "strip", "isolated_pad",
        "raft", "pile_bored".
    has_basement : bool
        If True, biases auto-selection toward raft.
    soil_type : str
        "normal", "fill", "clay", "soft", "hard". Affects auto-selection.
    pile_depth_m : float
        Design pile depth in metres (used only for pile_bored).

    Returns
    -------
    FoundationResult
        line_items, foundation_type, pile_count, column_count, warnings.
    """
    # ── Sanitise inputs ──────────────────────────────────────────────────────
    floor_area_sqm  = max(float(floor_area_sqm or 0.0), 0.0)
    floors          = max(int(floors or 1), 1)
    pile_depth_m    = float(pile_depth_m or 0.0)   # 0 triggers auto-selection below
    foundation_type = (foundation_type or "auto").strip().lower()
    building_type   = (building_type or "hostel").strip().lower()
    soil_type       = (soil_type or "normal").strip().lower()

    result = FoundationResult()
    result.warnings.append(
        "Foundation quantities are area-based estimates using rule-of-thumb norms "
        "(CPWD DSR 2023). Verify against actual foundation drawings and soil "
        "investigation report (IS 1892) before submission."
    )

    if floor_area_sqm <= 0:
        logger.warning("foundation_takeoff: floor_area_sqm=%.2f — returning empty result.", floor_area_sqm)
        result.warnings.append("floor_area_sqm is zero or not provided; no quantities computed.")
        return result

    # ── Foundation type selection ────────────────────────────────────────────
    if foundation_type == "auto":
        foundation_type = _auto_select_foundation(floors, building_type, has_basement, soil_type)
        logger.debug(
            "foundation_takeoff auto-selected: %s (floors=%d, type=%s, basement=%s, soil=%s)",
            foundation_type, floors, building_type, has_basement, soil_type,
        )

    result.foundation_type = foundation_type

    # ── Auto-select pile depth when pile_depth_m <= 0 ───────────────────────
    if foundation_type in ("pile_bored", "pile_driven") and pile_depth_m <= 0:
        pile_depth_m = _auto_pile_depth(floors, soil_type)
        result.warnings.append(
            f"pile_depth_m auto-selected: {pile_depth_m:.1f}m "
            f"(floors={floors}, soil='{soil_type}')"
        )
    elif pile_depth_m < 1.0:
        pile_depth_m = _PILE_DEPTH_DEFAULT_M

    # Footprint = floor area divided by number of floors
    footprint = floor_area_sqm / max(floors, 1)

    # ── Dispatch to builder ──────────────────────────────────────────────────
    if foundation_type == "strip":
        result.column_count = 0
        _build_strip(footprint, result)

    elif foundation_type == "isolated_pad":
        _build_isolated_pad(footprint, floor_area_sqm, result)

    elif foundation_type == "raft":
        result.column_count = math.ceil(floor_area_sqm / _COLUMN_GRID_SQM)
        _build_raft(footprint, result)

    elif foundation_type == "pile_bored":
        _build_pile_bored(footprint, floor_area_sqm, floors, pile_depth_m, result)

    elif foundation_type == "pile_driven":
        # Pile driven — treat as pile_bored with a different description note
        result.warnings.append(
            "pile_driven selected — quantities computed same as pile_bored; "
            "adjust rates and descriptions for driven precast piles."
        )
        _build_pile_bored(footprint, floor_area_sqm, floors, pile_depth_m, result)
        result.foundation_type = "pile_driven"

    else:
        result.warnings.append(
            f"Unknown foundation_type '{foundation_type}'; defaulting to strip."
        )
        result.foundation_type = "strip"
        _build_strip(footprint, result)

    # Filter out zero-quantity items (defensive)
    result.line_items = [i for i in result.line_items if (i.get("qty") or i.get("quantity") or 0) > 0]

    logger.info(
        "foundation_takeoff complete | type=%s | items=%d | columns=%d | piles=%d",
        result.foundation_type,
        len(result.line_items),
        result.column_count,
        result.pile_count,
    )
    return result
