"""
Wall Area Calculator — derives external and internal wall areas from floor area.

Used by brickwork_takeoff and plaster_takeoff as their primary input when no
drawing data is available (spec-estimate mode).

Norms sourced from CPWD DSR 2023, IS 875, and standard Indian QS practice.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

# Reference floor plate area (sqm) for which ext_pa_ratio norms are calibrated.
# Perimeter ∝ sqrt(floor_plate), so we correct for non-reference floor plates.
_REF_FLOOR_PLATE = 500.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perimeter-to-floor-area ratios (sqm wall per sqm floor) by building type
# Based on typical Indian institutional building shapes
# ---------------------------------------------------------------------------

# (external_p_a_ratio, internal_partition_factor, storey_height_m)
_BUILDING_WALL_NORMS = {
    "hostel":        (0.18, 1.8, 3.20),  # cellular rooms → many partitions
    "school":        (0.20, 1.2, 3.60),  # larger classrooms, fewer partitions
    "academic":      (0.17, 1.0, 3.60),  # lecture halls / labs
    "office":        (0.15, 0.8, 3.30),  # open plan → fewer partitions
    "hospital":      (0.20, 2.2, 3.50),  # wards + clinical rooms
    "residential":   (0.22, 1.5, 3.00),  # individual units
    "staff_quarters":(0.24, 1.6, 3.00),  # units with many rooms
    "dining":        (0.16, 0.4, 4.00),  # large open hall
    "laboratory":    (0.18, 1.2, 3.60),
    "auditorium":    (0.12, 0.2, 6.00),
    "library":       (0.16, 0.6, 4.00),
}
_DEFAULT_NORMS = (0.17, 1.2, 3.20)

# Deduction for openings (windows + doors) in external wall area
_OPENING_DEDUCTION = 0.30   # 30% of external wall gross area

# Internal partition openings (doorways)
_PARTITION_OPENING_DEDUCTION = 0.10  # 10%

# Building plan shape factors — perimeter correction vs square footprint.
# shape_factor > 1.0 means more perimeter than a square of equal area.
# Derivation: for L:W = r:1, shape_factor = (r+1) / (2*sqrt(r))
# r=1.0 → 1.00 | r=2.0 → 1.06 | r=3.0 → 1.15 | r=4.0 → 1.25
_SHAPE_FACTORS: dict = {
    "hostel":         1.15,   # corridor block, typically 3:1 plan
    "school":         1.10,
    "academic":       1.10,
    "office":         1.06,
    "hospital":       1.12,
    "residential":    1.06,
    "staff_quarters": 1.12,
    "dining":         1.06,
    "laboratory":     1.10,
    "research":       1.10,
    "auditorium":     1.00,   # typically square/circular
    "library":        1.06,
    "sports":         1.00,
    "workshop":       1.06,
    "health_centre":  1.06,
}
_DEFAULT_SHAPE_FACTOR: float = 1.08   # conservative default


@dataclass
class WallAreaResult:
    """Computed wall areas for use by brickwork and plaster modules."""
    external_wall_gross_sqm: float = 0.0
    external_wall_net_sqm: float = 0.0     # after deducting openings
    internal_partition_sqm: float = 0.0
    total_wall_sqm: float = 0.0            # net external + internal
    storey_height_m: float = 3.20
    building_type: str = "academic"
    shape_factor: float = 1.0   # plan shape correction applied
    warnings: list = field(default_factory=list)


def compute_wall_areas(
    floor_area_sqm: float,
    floors: int = 1,
    building_type: str = "academic",
    storey_height_m: Optional[float] = None,
    plan_shape_factor: Optional[float] = None,   # NEW — overrides table lookup
) -> WallAreaResult:
    """
    Derive wall areas from floor plate dimensions.

    Parameters
    ----------
    floor_area_sqm : float
        Total built-up area (all floors combined). If per-floor area is known,
        pass floor_area_sqm / floors and set floors=floors.
    floors : int
        Number of storeys (including ground).
    building_type : str
        One of the keys in _BUILDING_WALL_NORMS.
    storey_height_m : float, optional
        Override default storey height for this building type.

    Returns
    -------
    WallAreaResult
    """
    result = WallAreaResult(building_type=building_type)

    if not floor_area_sqm or floor_area_sqm <= 0:
        result.warnings.append("floor_area_sqm is zero — no wall area computed.")
        return result

    norms = _BUILDING_WALL_NORMS.get(building_type, _DEFAULT_NORMS)
    ext_pa_ratio, int_factor, default_height = norms

    h = storey_height_m if storey_height_m and storey_height_m > 0 else default_height
    result.storey_height_m = h

    # Per-floor plate area
    floor_plate = floor_area_sqm / max(floors, 1)

    # External wall: derived from perimeter × height × floors.
    # Perimeter scales as sqrt(floor_plate) (square-footprint approximation),
    # so total external wall = ratio * sqrt(floor_plate/ref) * ref * h * floors.
    # This correctly captures: same total area + more floors → more external envelope.
    # Resolve shape factor
    if plan_shape_factor is not None:
        shape_factor = plan_shape_factor
    else:
        shape_factor = _SHAPE_FACTORS.get(building_type, _DEFAULT_SHAPE_FACTOR)

    area_correction = math.sqrt(floor_plate / _REF_FLOOR_PLATE) * shape_factor
    result.shape_factor = shape_factor
    ext_gross_per_floor = ext_pa_ratio * _REF_FLOOR_PLATE * area_correction * h
    ext_gross_total = ext_gross_per_floor * floors
    ext_net_total = ext_gross_total * (1.0 - _OPENING_DEDUCTION)

    # Internal partitions: factor × external gross (accounts for cellular layout)
    int_gross = ext_gross_total * int_factor
    int_net = int_gross * (1.0 - _PARTITION_OPENING_DEDUCTION)

    result.external_wall_gross_sqm = round(ext_gross_total, 2)
    result.external_wall_net_sqm   = round(ext_net_total,   2)
    result.internal_partition_sqm  = round(int_net,         2)
    result.total_wall_sqm          = round(ext_net_total + int_net, 2)

    logger.debug(
        "wall_area: type=%s floors=%d area=%.0f → ext_gross=%.0f ext_net=%.0f int=%.0f",
        building_type, floors, floor_area_sqm,
        ext_gross_total, ext_net_total, int_net,
    )
    return result
