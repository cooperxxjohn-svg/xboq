"""
Plaster Takeoff — cement plaster / rendering quantity estimation.

Covers all standard plaster items per CPWD DSR 2023:
  - Internal plaster 12mm CM 1:6 (walls both faces)
  - Ceiling plaster 6mm CM 1:4
  - External rendering 20mm CM 1:4 (outer face of external walls)
  - Skirting plaster (base of internal walls, below flooring)
  - Soffit plaster (underside of staircase / beam soffits, estimated)

Inputs are WallAreaResult from wall_area_calculator + floor area for ceilings.
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

_INTERNAL_PLASTER_BOTH_FACES_FACTOR = 2.0   # both faces of each wall
_DEDUCTION_FOR_OPENINGS = 0.15              # was 0.10 — CPWD institutional norm (wide doorways)
_SKIRTING_HEIGHT_M = 0.15                   # 150mm skirting
_SOFFIT_AREA_FACTOR = 0.05                  # soffit ≈ 5% of total floor area
_EXT_RENDERING_OPENING_DEDUCTION = 0.25     # 25% for institutional (vs 30% residential)


@dataclass
class PlasterResult:
    line_items: List[dict] = field(default_factory=list)
    mode: str = "area_estimate"
    warnings: List[str] = field(default_factory=list)
    total_plaster_sqm: float = 0.0


def run_plaster_takeoff(
    floor_area_sqm: float,
    floors: int = 1,
    building_type: str = "academic",
    wall_areas: Optional[WallAreaResult] = None,
) -> PlasterResult:
    """
    Estimate plaster/rendering quantities.

    Parameters
    ----------
    floor_area_sqm : float
        Total built-up area (all floors).
    floors : int
    building_type : str
    wall_areas : WallAreaResult, optional
        Pre-computed wall areas; computed internally if not provided.
    """
    result = PlasterResult()

    if not floor_area_sqm or floor_area_sqm <= 0:
        result.warnings.append("floor_area_sqm zero — no plaster computed.")
        return result

    if wall_areas is None:
        wall_areas = compute_wall_areas(floor_area_sqm, floors, building_type)

    ext_net = wall_areas.external_wall_net_sqm
    int_net = wall_areas.internal_partition_sqm
    total_floor = floor_area_sqm   # includes all floors already

    # ── 1. Internal plaster 12mm CM 1:6 (both faces of all walls) ───────
    internal_plaster = (ext_net + int_net) * _INTERNAL_PLASTER_BOTH_FACES_FACTOR
    internal_plaster *= (1.0 - _DEDUCTION_FOR_OPENINGS)
    if internal_plaster > 0:
        result.line_items.append({
            "description": "Cement plaster 12mm CM 1:6 to internal walls (IS 1661)",
            "trade":  "finishing",
            "unit":   "sqm",
            "qty":    round(internal_plaster, 2),
            "source": "plaster_takeoff",
        })

    # ── 2. Ceiling plaster 6mm CM 1:4 ───────────────────────────────────
    ceiling_area = total_floor * 1.02  # slight uplift for beam soffits
    if ceiling_area > 0:
        result.line_items.append({
            "description": "Cement plaster 6mm CM 1:4 to RCC soffits / ceiling",
            "trade":  "finishing",
            "unit":   "sqm",
            "qty":    round(ceiling_area, 2),
            "source": "plaster_takeoff",
        })

    # ── 3. External rendering 20mm CM 1:4 ───────────────────────────────
    # Only external face of outer walls
    ext_rendering = wall_areas.external_wall_gross_sqm * (1.0 - _EXT_RENDERING_OPENING_DEDUCTION)  # deduct openings
    if ext_rendering > 0:
        result.line_items.append({
            "description": "Cement plaster 20mm CM 1:4 to external walls / rendering",
            "trade":  "finishing",
            "unit":   "sqm",
            "qty":    round(ext_rendering, 2),
            "source": "plaster_takeoff",
        })

    # ── 4. Skirting plaster ──────────────────────────────────────────────
    # Perimeter of all walls at ground × skirting height
    if wall_areas.external_wall_gross_sqm > 0 and wall_areas.storey_height_m > 0:
        all_wall_perimeter = (
            (wall_areas.external_wall_gross_sqm + wall_areas.internal_partition_sqm)
            / wall_areas.storey_height_m
        )
        skirting = all_wall_perimeter * _SKIRTING_HEIGHT_M
        if skirting > 0:
            result.line_items.append({
                "description": "Skirting — cement plaster 15mm CM 1:3 (150mm ht)",
                "trade":  "finishing",
                "unit":   "sqm",
                "qty":    round(skirting, 2),
                "source": "plaster_takeoff",
            })

    # ── 5. Soffit plaster (staircase / beam underside) ───────────────────
    soffit = total_floor * _SOFFIT_AREA_FACTOR
    if soffit > 0:
        result.line_items.append({
            "description": "Cement plaster 12mm to staircase soffits and beam sides",
            "trade":  "finishing",
            "unit":   "sqm",
            "qty":    round(soffit, 2),
            "source": "plaster_takeoff",
        })

    result.total_plaster_sqm = round(
        internal_plaster + ceiling_area + ext_rendering, 2
    )

    if not result.line_items:
        result.warnings.append("No plaster items generated — check wall area inputs.")

    logger.debug(
        "plaster: internal=%.0f ext=%.0f ceiling=%.0f total=%.0f sqm",
        internal_plaster, ext_rendering, ceiling_area, result.total_plaster_sqm,
    )
    return result
