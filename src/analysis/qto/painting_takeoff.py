"""
Painting Takeoff — Room-based QTO for Indian construction projects.

Computes painting quantities from RoomData objects (finish_takeoff.py) for:
  - Interior walls (putty + primer + OBD)
  - Ceilings (primer + OBD)
  - Exterior walls (weather shield / textured paint)
  - Wood/steel surfaces (enamel paint — estimated from door/window count)
  - MS grilles/railings (anti-corrosive + enamel — estimated from floor area)
  - Damp-proof coating (wet rooms below tile line)

Room height assumptions follow standard Indian QS practice; wet rooms
(toilet/bathroom) have their walls excluded from paint because finish_takeoff
already accounts for ceramic dado there.

Design constraints:
- NO cv2 / OpenCV — pure arithmetic + room data.
- Graceful None handling throughout (rooms may have None area/dims).
- Generates items in the same dict format as other QTO modules.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Floor-to-ceiling heights (m) by room name keyword — Indian residential/commercial
_ROOM_HEIGHTS: Dict[str, float] = {
    "LIVING":    2.85,
    "DRAWING":   2.85,
    "HALL":      2.85,
    "DINING":    2.75,
    "BEDROOM":   2.75,
    "MASTER":    2.75,
    "KITCHEN":   2.60,
    "TOILET":    2.50,
    "BATHROOM":  2.50,
    "BALCONY":   2.50,
    "TERRACE":   2.50,
    "STORE":     2.40,
    "PASSAGE":   2.50,
}
_DEFAULT_HEIGHT: float = 2.70

# Percentage of gross wall area deducted for doors + windows (Indian QS standard)
_OPENING_DEDUCTION: float = 0.15

# Kitchen: lower 2.1 m of walls are ceramic tiled — only paint above this height
_KITCHEN_TILE_HEIGHT: float = 2.10

# Exterior average storey height used when computing external wall area
_EXT_STOREY_HEIGHT: float = 3.00

# Enamel paint to wood/steel: fraction of interior wall area
_ENAMEL_FRACTION: float = 0.10

# MS grille/railing: sqm per 10 sqm floor area (balcony + staircase railings)
_RAILING_FACTOR: float = 0.10   # 1 sqm per 10 sqm → divide floor area by 10

# Damp-proof coating: height of treatment below wall tiles in wet rooms (m)
_DPC_HEIGHT: float = 2.10       # matches tile dato height

# Area-based fallback multipliers (when no room data available)
_FALLBACK_INTERIOR_WALL_FACTOR: float = 3.2
_FALLBACK_CEILING_FACTOR:       float = 1.0
_FALLBACK_EXTERIOR_FACTOR:      float = 0.8


# =============================================================================
# ROOM CLASSIFICATION HELPERS
# =============================================================================

# Rooms whose walls are FULLY tiled — exclude entirely from interior paint
_FULL_TILE_WALL_KEYWORDS = ("TOILET", "BATHROOM", "WET")

# Rooms whose walls are PARTIALLY tiled (lower portion) — kitchen
_PARTIAL_TILE_WALL_KEYWORDS = ("KITCHEN",)

# Rooms classified as wet for damp-proof coating
_WET_ROOM_KEYWORDS = ("TOILET", "BATHROOM", "WET")


def _room_height(room_name: str) -> float:
    """Return the assumed ceiling height (m) for a given room name."""
    name_upper = (room_name or "").upper()
    for keyword, height in _ROOM_HEIGHTS.items():
        if keyword in name_upper:
            return height
    return _DEFAULT_HEIGHT


def _is_full_tile_wall(room_name: str) -> bool:
    """True if the room walls are fully tiled (no interior paint)."""
    name_upper = (room_name or "").upper()
    return any(kw in name_upper for kw in _FULL_TILE_WALL_KEYWORDS)


def _is_partial_tile_wall(room_name: str) -> bool:
    """True if only the lower portion of the room walls are tiled (kitchen)."""
    name_upper = (room_name or "").upper()
    return any(kw in name_upper for kw in _PARTIAL_TILE_WALL_KEYWORDS)


def _is_wet_room(room_name: str) -> bool:
    """True if the room requires damp-proof coating under tiles."""
    name_upper = (room_name or "").upper()
    return any(kw in name_upper for kw in _WET_ROOM_KEYWORDS)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _room_perimeter(room: Any) -> Optional[float]:
    """
    Calculate room perimeter (m).

    Priority:
    1. dim_l + dim_w from RoomData (convert mm/cm if needed)
    2. Estimate from area: 4 × sqrt(area)
    """
    dim_l = getattr(room, "dim_l", None)
    dim_w = getattr(room, "dim_w", None)

    if dim_l is not None and dim_w is not None:
        # Normalise to metres (same logic as finish_takeoff._dims_to_sqm)
        if dim_l > 100 and dim_w > 100:          # mm
            l_m = dim_l / 1000.0
            w_m = dim_w / 1000.0
        elif dim_l > 10 and dim_w > 10:          # cm
            l_m = dim_l / 100.0
            w_m = dim_w / 100.0
        else:                                     # already metres
            l_m = dim_l
            w_m = dim_w
        return 2.0 * (l_m + w_m)

    area = getattr(room, "area_sqm", None)
    if area and area > 0:
        return 4.0 * math.sqrt(area)

    return None


def _safe_area(room: Any) -> float:
    """Return room.area_sqm, defaulting to 0.0 if None or negative."""
    area = getattr(room, "area_sqm", None)
    if area is None or area < 0:
        return 0.0
    return float(area)


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class PaintingResult:
    line_items: List[dict] = field(default_factory=list)
    mode: str = "from_rooms"          # "from_rooms" | "area_estimate"
    warnings: List[str] = field(default_factory=list)
    total_interior_wall_sqm: float = 0.0
    total_ceiling_sqm: float = 0.0
    total_exterior_wall_sqm: float = 0.0


# =============================================================================
# ITEM BUILDERS
# =============================================================================

def _make_item(
    description: str,
    qty: float,
    unit: str,
    spec: str,
    source: str = "painting_takeoff",
) -> dict:
    return {
        "description": description,
        "qty":         round(qty, 2),
        "unit":        unit,
        "trade":       "Painting",
        "spec":        spec,
        "source":      source,
    }


# =============================================================================
# ROOM-BASED CALCULATION
# =============================================================================

def _compute_from_rooms(
    rooms: List[Any],
    floor_area_sqm: float,
    floors: int,
    exterior_perimeter_m: float,
) -> PaintingResult:
    """
    Derive all painting quantities from RoomData objects.
    """
    result = PaintingResult(mode="from_rooms")

    net_interior_wall_sqm = 0.0
    total_ceiling_sqm     = 0.0
    wet_room_perimeter_sum = 0.0
    skipped_rooms          = 0

    for room in rooms:
        name    = getattr(room, "name", "") or ""
        area    = _safe_area(room)
        perim   = _room_perimeter(room)
        height  = _room_height(name)

        if perim is None:
            skipped_rooms += 1
            continue

        gross_wall = perim * height
        net_wall   = gross_wall * (1.0 - _OPENING_DEDUCTION)

        # Ceiling — all rooms contribute
        if area > 0:
            total_ceiling_sqm += area

        # Wall paint logic
        if _is_full_tile_wall(name):
            # No paint on walls; but record perimeter for damp-proof coating
            wet_room_perimeter_sum += perim
            # No contribution to interior wall paint total
        elif _is_partial_tile_wall(name):
            # Only paint the zone ABOVE the tile height
            paintable_height = max(0.0, height - _KITCHEN_TILE_HEIGHT)
            net_paintable = (perim * paintable_height) * (1.0 - _OPENING_DEDUCTION)
            net_interior_wall_sqm += net_paintable
        else:
            net_interior_wall_sqm += net_wall

    if skipped_rooms:
        result.warnings.append(
            f"{skipped_rooms} room(s) skipped — no area or dimension data available."
        )

    # External wall area
    if exterior_perimeter_m > 0:
        avg_height = _EXT_STOREY_HEIGHT
        ext_wall_sqm = exterior_perimeter_m * avg_height * floors
    else:
        # Estimate perimeter from single-floor footprint
        single_floor_area = floor_area_sqm / max(floors, 1)
        est_perimeter     = 4.0 * math.sqrt(single_floor_area)
        ext_wall_sqm      = est_perimeter * _EXT_STOREY_HEIGHT * floors
        result.warnings.append(
            "Exterior perimeter not provided; estimated from floor area "
            f"({est_perimeter:.1f} m × {_EXT_STOREY_HEIGHT} m × {floors} floors)."
        )

    result.total_interior_wall_sqm = round(net_interior_wall_sqm, 2)
    result.total_ceiling_sqm       = round(total_ceiling_sqm, 2)
    result.total_exterior_wall_sqm = round(ext_wall_sqm, 2)

    # Derived quantities
    enamel_sqm   = net_interior_wall_sqm * _ENAMEL_FRACTION
    railing_sqm  = floor_area_sqm * _RAILING_FACTOR
    dpc_sqm      = wet_room_perimeter_sum * _DPC_HEIGHT

    # Build line items
    items: List[dict] = []

    if net_interior_wall_sqm > 0:
        items.append(_make_item(
            description="Interior wall putty + primer + 2 coats OBD to internal walls",
            qty=net_interior_wall_sqm,
            unit="sqm",
            spec="IS 428 (putty); IS 106 (primer); IS 428 (OBD)",
        ))

    if total_ceiling_sqm > 0:
        items.append(_make_item(
            description="Interior ceiling primer + 2 coats oil bound distemper (OBD)",
            qty=total_ceiling_sqm,
            unit="sqm",
            spec="IS 428",
        ))

    if ext_wall_sqm > 0:
        items.append(_make_item(
            description="Exterior weather shield / textured paint to external walls (2 coats)",
            qty=ext_wall_sqm,
            unit="sqm",
            spec="IS 15489",
        ))

    if enamel_sqm > 0:
        items.append(_make_item(
            description="Enamel paint to wood/steel surfaces (door frames, window frames, beadings)",
            qty=enamel_sqm,
            unit="sqm",
            spec="IS 133",
        ))

    if railing_sqm > 0:
        items.append(_make_item(
            description="Anti-corrosive primer + synthetic enamel to MS grilles, railings and gates",
            qty=railing_sqm,
            unit="sqm",
            spec="IS 2074",
        ))

    if dpc_sqm > 0:
        items.append(_make_item(
            description="Damp-proof coating to wet room walls (under tile zone, 2.1 m height)",
            qty=dpc_sqm,
            unit="sqm",
            spec="IS 2645",
        ))

    result.line_items = items
    return result


# =============================================================================
# AREA-BASED FALLBACK
# =============================================================================

def _compute_from_area(
    floor_area_sqm: float,
    floors: int,
    exterior_perimeter_m: float,
) -> PaintingResult:
    """
    Estimate painting quantities using floor area multipliers when no room
    data is available.
    """
    result = PaintingResult(mode="area_estimate")
    result.warnings.append(
        "No room data provided; painting quantities estimated from floor area "
        "using rule-of-thumb multipliers (interior walls ×3.2, ceiling ×1.0, "
        "exterior ×0.8). Verify against actual plans."
    )

    interior_wall_sqm = floor_area_sqm * _FALLBACK_INTERIOR_WALL_FACTOR
    ceiling_sqm       = floor_area_sqm * _FALLBACK_CEILING_FACTOR

    if exterior_perimeter_m > 0:
        ext_wall_sqm = exterior_perimeter_m * _EXT_STOREY_HEIGHT * floors
    else:
        ext_wall_sqm = floor_area_sqm * _FALLBACK_EXTERIOR_FACTOR

    enamel_sqm  = interior_wall_sqm * _ENAMEL_FRACTION
    railing_sqm = floor_area_sqm * _RAILING_FACTOR
    dpc_sqm     = floor_area_sqm * 0.05   # ~5% of floor area for wet zones

    result.total_interior_wall_sqm = round(interior_wall_sqm, 2)
    result.total_ceiling_sqm       = round(ceiling_sqm, 2)
    result.total_exterior_wall_sqm = round(ext_wall_sqm, 2)

    items: List[dict] = [
        _make_item(
            description="Interior wall putty + primer + 2 coats OBD to internal walls",
            qty=interior_wall_sqm,
            unit="sqm",
            spec="IS 428 (putty); IS 106 (primer); IS 428 (OBD)",
        ),
        _make_item(
            description="Interior ceiling primer + 2 coats oil bound distemper (OBD)",
            qty=ceiling_sqm,
            unit="sqm",
            spec="IS 428",
        ),
        _make_item(
            description="Exterior weather shield / textured paint to external walls (2 coats)",
            qty=ext_wall_sqm,
            unit="sqm",
            spec="IS 15489",
        ),
        _make_item(
            description="Enamel paint to wood/steel surfaces (door frames, window frames, beadings)",
            qty=enamel_sqm,
            unit="sqm",
            spec="IS 133",
        ),
        _make_item(
            description="Anti-corrosive primer + synthetic enamel to MS grilles, railings and gates",
            qty=railing_sqm,
            unit="sqm",
            spec="IS 2074",
        ),
        _make_item(
            description="Damp-proof coating to wet room walls (under tile zone, 2.1 m height)",
            qty=dpc_sqm,
            unit="sqm",
            spec="IS 2645",
        ),
    ]
    result.line_items = [i for i in items if i["qty"] > 0]
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_painting_takeoff(
    rooms: List[Any],
    floor_area_sqm: float,
    floors: int = 1,
    exterior_perimeter_m: float = 0.0,
    door_count: int = 0,
    window_count: int = 0,
) -> PaintingResult:
    """
    Compute painting quantities for an Indian construction project.

    Args:
        rooms:                 List of finish_takeoff.RoomData objects (may be empty).
        floor_area_sqm:        Total gross floor area (sqm). Used as fallback when
                               no rooms are provided, and for railing/exterior estimates.
        floors:                Number of floors (storeys above ground). Default 1.
        exterior_perimeter_m:  Known building perimeter (m). If 0, estimated from area.
        door_count:            Total door count from DW takeoff (informational; not
                               currently used in calculation — opening deduction is
                               applied as a flat 15% per IS QS practice).
        window_count:          Total window count (informational).

    Returns:
        PaintingResult with line_items, mode, warnings, and summary totals.
    """
    floors          = max(floors, 1)
    floor_area_sqm  = max(floor_area_sqm or 0.0, 0.0)

    # Validate that we have usable room data
    usable_rooms = [
        r for r in (rooms or [])
        if (_safe_area(r) > 0 or _room_perimeter(r) is not None)
    ]

    if usable_rooms:
        result = _compute_from_rooms(
            rooms=usable_rooms,
            floor_area_sqm=floor_area_sqm,
            floors=floors,
            exterior_perimeter_m=exterior_perimeter_m,
        )
        if len(usable_rooms) < len(rooms or []):
            dropped = len(rooms) - len(usable_rooms)
            result.warnings.append(
                f"{dropped} room(s) had no area or dimension data and were excluded."
            )
    else:
        if not floor_area_sqm:
            # Nothing to work with
            logger.warning("painting_takeoff: no rooms and no floor_area_sqm — returning empty result.")
            result = PaintingResult(mode="area_estimate")
            result.warnings.append(
                "No room data and no floor area provided. Cannot compute painting quantities."
            )
            return result

        result = _compute_from_area(
            floor_area_sqm=floor_area_sqm,
            floors=floors,
            exterior_perimeter_m=exterior_perimeter_m,
        )

    if door_count or window_count:
        result.warnings.append(
            f"door_count={door_count}, window_count={window_count} noted; "
            "opening deduction applied as flat 15% of gross wall area per IS QS practice."
        )

    logger.info(
        "painting_takeoff complete | mode=%s | items=%d | "
        "int_wall=%.1f sqm | ceiling=%.1f sqm | ext_wall=%.1f sqm",
        result.mode,
        len(result.line_items),
        result.total_interior_wall_sqm,
        result.total_ceiling_sqm,
        result.total_exterior_wall_sqm,
    )

    return result
