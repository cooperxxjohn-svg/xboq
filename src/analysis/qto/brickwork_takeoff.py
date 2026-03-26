"""
Brickwork Takeoff — masonry quantity estimation for Indian construction.

Derives brick masonry volumes from wall areas (wall_area_calculator) or
from floor area norms when no wall data is available.

Covers:
  - 230mm (1-brick) external walls in CM 1:6
  - 115mm (half-brick) internal partitions in CM 1:4
  - 75mm block partitions (lightweight internal walls, optional)
  - Deductions for lintels / bond beams (estimated at 5% of volume)

Units and specs per CPWD DSR 2023.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .wall_area_calculator import WallAreaResult, compute_wall_areas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXTERNAL_WALL_THICKNESS_M  = 0.230   # 1-brick wall
_PARTITION_THICKNESS_M       = 0.115   # half-brick
_LINTEL_DEDUCTION_FACTOR     = 0.05    # 5% deduction for lintels / sills

# ── AAC block masonry variant ──────────────────────────────────────────────────
_MASONRY_SYSTEM_BRICK = "brick"
_MASONRY_SYSTEM_AAC   = "aac"
_MASONRY_SYSTEM_MIXED = "mixed"   # AAC internal + brick external

_AAC_EXT_THICKNESS_M  = 0.200   # 200mm AAC external wall
_AAC_INT_THICKNESS_M  = 0.100   # 100mm AAC internal partition
_AAC_LINTEL_DEDUCTION = 0.03    # 3% (pre-lintel slots in AAC)

_AAC_KEYWORDS: frozenset = frozenset({
    "aac", "autoclaved aerated concrete", "aac block", "aerated block",
    "siporex", "ytong", "hebel", "lightweight concrete block",
    "lightweight block",
})


@dataclass
class BrickworkResult:
    line_items: List[dict] = field(default_factory=list)
    mode: str = "area_estimate"
    warnings: List[str] = field(default_factory=list)
    total_brickwork_cum: float = 0.0
    masonry_system: str = "brick"   # resolved masonry system


def _detect_masonry_system(spec_text: str) -> str:
    """
    Scan spec/NIT text for masonry material indicators.
    Returns "aac", "mixed", or "brick" (default).
    """
    if not spec_text:
        return _MASONRY_SYSTEM_BRICK
    lower = spec_text.lower()
    has_aac = any(kw in lower for kw in _AAC_KEYWORDS)
    has_brick = any(kw in lower for kw in ("brick masonry", "brickwork", "230mm brick"))
    if has_aac and has_brick:
        return _MASONRY_SYSTEM_MIXED
    if has_aac:
        return _MASONRY_SYSTEM_AAC
    return _MASONRY_SYSTEM_BRICK


def run_brickwork_takeoff(
    floor_area_sqm: float,
    floors: int = 1,
    building_type: str = "academic",
    wall_areas: Optional[WallAreaResult] = None,
    masonry_system: str = "auto",   # NEW: "auto"|"brick"|"aac"|"mixed"
    spec_text: str = "",            # NEW: used when masonry_system="auto"
) -> BrickworkResult:
    """
    Estimate brickwork quantities.

    Parameters
    ----------
    floor_area_sqm : float
        Total built-up area (all floors).
    floors : int
    building_type : str
    wall_areas : WallAreaResult, optional
        Pre-computed wall areas. If None, computed internally.
    masonry_system : str
        "auto" (detect from spec_text), "brick", "aac", or "mixed".
    spec_text : str
        Specification / NIT text used when masonry_system="auto".
    """
    result = BrickworkResult()

    if not floor_area_sqm or floor_area_sqm <= 0:
        result.warnings.append("floor_area_sqm zero — no brickwork computed.")
        return result

    if wall_areas is None:
        wall_areas = compute_wall_areas(floor_area_sqm, floors, building_type)

    # ── Resolve masonry system ─────────────────────────────────────────────────
    if masonry_system == "auto":
        masonry_system = _detect_masonry_system(spec_text)

    # Select thicknesses and descriptions based on masonry system
    if masonry_system == _MASONRY_SYSTEM_AAC:
        ext_thick = _AAC_EXT_THICKNESS_M
        int_thick = _AAC_INT_THICKNESS_M
        lintel_ded = _AAC_LINTEL_DEDUCTION
        ext_desc = (
            "AAC block masonry in CM 1:6 for external walls, "
            "200mm thick (IS 2185 Part 3), including jointing mortar"
        )
        int_desc = (
            "AAC block masonry in CM 1:4 for internal partitions, "
            "100mm thick (IS 2185 Part 3)"
        )
    elif masonry_system == _MASONRY_SYSTEM_MIXED:
        ext_thick = _EXTERNAL_WALL_THICKNESS_M   # brick external (existing constant)
        int_thick = _AAC_INT_THICKNESS_M          # AAC internal
        lintel_ded = _LINTEL_DEDUCTION_FACTOR      # existing constant
        ext_desc = (
            "Brick masonry in CM 1:6 in superstructure (230mm external walls)"
        )
        int_desc = (
            "AAC block masonry in CM 1:4 for internal partitions, "
            "100mm thick (IS 2185 Part 3)"
        )
    else:   # brick (default)
        ext_thick = _EXTERNAL_WALL_THICKNESS_M
        int_thick = _PARTITION_THICKNESS_M
        lintel_ded = _LINTEL_DEDUCTION_FACTOR
        ext_desc = (
            "Brick masonry in CM 1:6 in superstructure (230mm external walls)"
        )
        int_desc = (
            "Brick masonry in CM 1:4 in partitions (115mm internal walls)"
        )

    result.masonry_system = masonry_system

    ext_net  = wall_areas.external_wall_net_sqm
    int_net  = wall_areas.internal_partition_sqm

    # ── External walls ───────────────────────────────────────────────────
    ext_vol = ext_net * ext_thick * (1.0 - lintel_ded)
    if ext_vol > 0:
        qty_rounded = round(ext_vol, 2)
        result.line_items.append({
            "description": ext_desc,
            "trade":    "masonry",
            "unit":     "cum",
            "qty":      qty_rounded,
            "quantity": qty_rounded,
            "source":   "brickwork_takeoff",
        })

    # ── Internal partitions ──────────────────────────────────────────────
    int_vol = int_net * int_thick * (1.0 - lintel_ded)
    if int_vol > 0:
        qty_rounded = round(int_vol, 2)
        result.line_items.append({
            "description": int_desc,
            "trade":    "masonry",
            "unit":     "cum",
            "qty":      qty_rounded,
            "quantity": qty_rounded,
            "source":   "brickwork_takeoff",
        })

    # ── DPC — 50mm cement concrete at plinth ───────────────────────────
    # DPC runs on all external walls: perimeter × 0.23m width × 0.05m thick
    # Approximate perimeter from external gross area / storey height
    if wall_areas.external_wall_gross_sqm > 0 and wall_areas.storey_height_m > 0:
        perimeter = wall_areas.external_wall_gross_sqm / wall_areas.storey_height_m / floors
        dpc_area = perimeter * _EXTERNAL_WALL_THICKNESS_M
        qty_rounded = round(dpc_area, 2)
        result.line_items.append({
            "description": "Damp-proof course (DPC) — 50mm cement concrete 1:2:4 at plinth level",
            "trade":    "masonry",
            "unit":     "sqm",
            "qty":      qty_rounded,
            "quantity": qty_rounded,
            "source":   "brickwork_takeoff",
        })

    # ── Lintel / sunshade concrete (estimated) ──────────────────────────
    # Lintels over openings — assume 0.15m depth × 0.23m width × opening width
    # Opening area ≈ 30% of external gross → opening perimeter ≈ area / avg height 1.2m
    if ext_net > 0:
        opening_area = wall_areas.external_wall_gross_sqm * 0.30
        lintel_vol = opening_area / 1.2 * 0.15 * 0.23  # lm × depth × width
        qty_rounded = round(lintel_vol, 2)
        result.line_items.append({
            "description": "RCC lintels / sunshades M20 over openings",
            "trade":    "structural",
            "unit":     "cum",
            "qty":      qty_rounded,
            "quantity": qty_rounded,
            "source":   "brickwork_takeoff",
        })

    result.total_brickwork_cum = round(ext_vol + int_vol, 2)

    if not result.line_items:
        result.warnings.append("No brickwork items generated — check wall area inputs.")

    logger.debug(
        "brickwork: ext=%.1f cum, int=%.1f cum, total=%.1f cum",
        ext_vol, int_vol, result.total_brickwork_cum,
    )
    return result
