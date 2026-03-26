"""
Waterproofing Takeoff — Room-based QTO for Indian construction projects.

Computes waterproofing quantities from RoomData objects (finish_takeoff.py) for:
  - Crystalline WP to toilet/bathroom floors (IS 2645)
  - Polymer-modified cement mortar WP to wet room walls (300 mm upturn)
  - APP membrane WP to terrace/roof (IS 15396)
  - Crystalline WP to kitchen floors
  - Bituminous tanking WP to basement (if applicable)
  - WP to balcony/verandah floors
  - Sealant to expansion joints (estimated)
  - Grouting to wall-floor junctions in wet areas

Room classification follows Indian construction practice; the module gracefully
degrades to area-based estimates when no room data is provided.

Design constraints:
- NO cv2 / OpenCV — pure arithmetic + room data.
- Graceful None handling throughout (rooms may have None area/dims).
- Generates items in the same dict format as other QTO modules.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Area uplift factors (account for skirting / upturns / overlaps)
_WET_FLOOR_UPLIFT:     float = 1.05   # 5% extra for skirting at floor perimeter
_ROOF_UPLIFT:          float = 1.10   # 10% extra for upturn at parapet / kerb
_KITCHEN_FLOOR_UPLIFT: float = 1.05
_BALCONY_FLOOR_UPLIFT: float = 1.05

# Polymer-modified mortar upturn height on wet room walls above floor level (m)
_WALL_WP_UPTURN_HEIGHT: float = 0.30   # 300 mm per IS 2645

# Basement area factor: total WP area = floor + walls approximation
_BASEMENT_AREA_FACTOR: float = 2.5

# Expansion joint sealant: linear metres per 10 sqm of floor area
_EXPANSION_JOINT_FACTOR: float = 0.05   # 0.5 lm per 10 sqm = 0.05 lm / sqm

# Fallback wet-area fraction of total floor area (typical residential)
_FALLBACK_WET_FRACTION:  float = 0.15

# =============================================================================
# ROOM CLASSIFICATION KEYWORDS
# =============================================================================

# Fully wet rooms: crystalline WP floor + polymer wall upturn
_WET_ROOM_KEYWORDS = ("TOILET", "BATHROOM", "WET AREA", "WET", "PANTRY", "LAUNDRY")

# Kitchen-type rooms: crystalline WP floor only (wall WP = tiles in finish_takeoff)
_KITCHEN_KEYWORDS = ("KITCHEN", "COOK", "SCULLERY")

# Outdoor/terrace areas: APP membrane WP
_TERRACE_KEYWORDS = ("TERRACE", "ROOF", "BALCONY", "VERANDAH")

# Basement/underground areas: bituminous tanking
_BASEMENT_KEYWORDS = ("BASEMENT", "UNDERGROUND", "SUMP", "TANK")


# =============================================================================
# ROOM CLASSIFICATION HELPERS
# =============================================================================

def _classify_room(room_name: str) -> str:
    """
    Return one of: "wet", "kitchen", "terrace", "basement", "dry".
    Classification is mutually exclusive; first keyword match wins in priority order.
    """
    name_upper = (room_name or "").upper()

    if any(kw in name_upper for kw in _BASEMENT_KEYWORDS):
        return "basement"
    if any(kw in name_upper for kw in _WET_ROOM_KEYWORDS):
        return "wet"
    if any(kw in name_upper for kw in _KITCHEN_KEYWORDS):
        return "kitchen"
    if any(kw in name_upper for kw in _TERRACE_KEYWORDS):
        return "terrace"
    return "dry"


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _safe_area(room: Any) -> float:
    """Return room.area_sqm as a non-negative float; 0.0 on None / negative."""
    area = getattr(room, "area_sqm", None)
    if area is None or area < 0:
        return 0.0
    return float(area)


def _room_perimeter(room: Any) -> float:
    """
    Return room perimeter (m).

    Priority:
    1. dim_l + dim_w (converts mm/cm to m automatically)
    2. Estimate from area: 4 × sqrt(area)
    3. 0.0 if neither is available
    """
    dim_l = getattr(room, "dim_l", None)
    dim_w = getattr(room, "dim_w", None)

    if dim_l is not None and dim_w is not None:
        if dim_l > 100 and dim_w > 100:      # mm
            l_m, w_m = dim_l / 1000.0, dim_w / 1000.0
        elif dim_l > 10 and dim_w > 10:      # cm
            l_m, w_m = dim_l / 100.0, dim_w / 100.0
        else:                                 # m
            l_m, w_m = dim_l, dim_w
        return 2.0 * (l_m + w_m)

    area = _safe_area(room)
    if area > 0:
        return 4.0 * math.sqrt(area)

    return 0.0


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class WaterproofingResult:
    line_items: List[dict] = field(default_factory=list)
    mode: str = "from_rooms"          # "from_rooms" | "area_estimate"
    warnings: List[str] = field(default_factory=list)
    wet_area_sqm: float = 0.0         # total wet room floor area
    roof_area_sqm: float = 0.0        # total terrace/roof area waterproofed


# =============================================================================
# ITEM BUILDER
# =============================================================================

def _make_item(
    description: str,
    qty: float,
    unit: str,
    spec: str,
    source: str = "waterproofing_takeoff",
) -> dict:
    return {
        "description": description,
        "qty":         round(qty, 2),
        "unit":        unit,
        "trade":       "Waterproofing",
        "spec":        spec,
        "source":      source,
    }


# =============================================================================
# ROOM-BASED CALCULATION
# =============================================================================

def _compute_from_rooms(
    rooms: List[Any],
    roof_area_sqm: float,
    floor_area_sqm: float,
    floors: int,
    has_basement: bool,
    basement_area_sqm: float,
) -> WaterproofingResult:
    """
    Derive all waterproofing quantities from RoomData objects.
    """
    result = WaterproofingResult(mode="from_rooms")

    # Accumulators by classification
    wet_floor_area       = 0.0
    wet_perimeter_sum    = 0.0   # for wall upturn WP and grouting
    kitchen_floor_area   = 0.0
    terrace_floor_area   = 0.0
    skipped_rooms        = 0

    for room in rooms:
        name       = getattr(room, "name", "") or ""
        area       = _safe_area(room)
        perimeter  = _room_perimeter(room)
        category   = _classify_room(name)

        if area <= 0 and perimeter <= 0:
            skipped_rooms += 1
            continue

        if category == "wet":
            wet_floor_area    += area
            wet_perimeter_sum += perimeter

        elif category == "kitchen":
            kitchen_floor_area += area

        elif category == "terrace":
            terrace_floor_area += area

        elif category == "basement":
            # Basement rooms contribute to basement_area_sqm if has_basement=True;
            # individual room basement areas supplement the parameter value.
            if has_basement:
                basement_area_sqm = max(basement_area_sqm, area)

        # "dry" rooms: no waterproofing

    if skipped_rooms:
        result.warnings.append(
            f"{skipped_rooms} room(s) skipped — no area or perimeter data available."
        )

    # Resolve roof area
    effective_roof_sqm = roof_area_sqm
    if effective_roof_sqm <= 0 and terrace_floor_area <= 0:
        # Fall back to footprint-based estimate
        single_floor = floor_area_sqm / max(floors, 1)
        effective_roof_sqm = single_floor
        result.warnings.append(
            "No roof area or terrace rooms detected; roof WP area estimated as "
            f"single-floor footprint ({effective_roof_sqm:.1f} sqm)."
        )
    elif effective_roof_sqm <= 0:
        effective_roof_sqm = terrace_floor_area

    # Combine terrace rooms with explicit roof area (take the larger)
    total_roof_for_wp = max(effective_roof_sqm, terrace_floor_area)

    result.wet_area_sqm  = round(wet_floor_area, 2)
    result.roof_area_sqm = round(total_roof_for_wp, 2)

    # Expansion joint sealant over total floor area
    sealant_lm    = floor_area_sqm * _EXPANSION_JOINT_FACTOR
    # Grouting at wall-floor junctions: total wet room perimeter
    grouting_lm   = wet_perimeter_sum

    # Basement WP
    basement_wp_sqm = 0.0
    if has_basement and basement_area_sqm > 0:
        basement_wp_sqm = basement_area_sqm * _BASEMENT_AREA_FACTOR

    # Build line items (suppress zero-quantity items)
    items: List[dict] = []

    if wet_floor_area > 0:
        items.append(_make_item(
            description=(
                "Crystalline waterproofing compound to toilet/bathroom floors "
                "(incl. 50 mm skirting upturn)"
            ),
            qty=wet_floor_area * _WET_FLOOR_UPLIFT,
            unit="sqm",
            spec="IS 2645 / manufacturer's specification",
        ))

    if wet_perimeter_sum > 0:
        items.append(_make_item(
            description=(
                "Polymer-modified cement mortar waterproofing to toilet/bathroom "
                "walls (300 mm upturn above floor level)"
            ),
            qty=wet_perimeter_sum * _WALL_WP_UPTURN_HEIGHT,
            unit="sqm",
            spec="IS 2645",
        ))

    if total_roof_for_wp > 0:
        items.append(_make_item(
            description=(
                "APP modified bitumen membrane waterproofing to terrace/roof "
                "(incl. 300 mm upturns at parapet/kerb)"
            ),
            qty=total_roof_for_wp * _ROOF_UPLIFT,
            unit="sqm",
            spec="IS 15396",
        ))

    if kitchen_floor_area > 0:
        items.append(_make_item(
            description=(
                "Crystalline waterproofing compound to kitchen floor "
                "(incl. 50 mm skirting upturn)"
            ),
            qty=kitchen_floor_area * _KITCHEN_FLOOR_UPLIFT,
            unit="sqm",
            spec="IS 2645 / manufacturer's specification",
        ))

    if basement_wp_sqm > 0:
        items.append(_make_item(
            description=(
                "Bituminous tanking waterproofing to basement walls and floor "
                "(multi-coat system, floor + walls)"
            ),
            qty=basement_wp_sqm,
            unit="sqm",
            spec="IS 1580",
        ))

    if terrace_floor_area > 0:
        items.append(_make_item(
            description=(
                "Waterproofing to balcony/verandah floor (crystalline slurry coat, "
                "incl. 50 mm skirting upturn)"
            ),
            qty=terrace_floor_area * _BALCONY_FLOOR_UPLIFT,
            unit="sqm",
            spec="IS 2645",
        ))

    if sealant_lm > 0:
        items.append(_make_item(
            description=(
                "Polyurethane sealant to expansion joints and construction joints "
                "(est. 0.5 lm per 10 sqm floor area)"
            ),
            qty=sealant_lm,
            unit="lm",
            spec="IS 11433",
        ))

    if grouting_lm > 0:
        items.append(_make_item(
            description=(
                "Polymer cement grouting / sealing at wall-floor junction in "
                "wet areas (cove formation)"
            ),
            qty=grouting_lm,
            unit="lm",
            spec="IS 2645",
        ))

    result.line_items = items
    return result


# =============================================================================
# AREA-BASED FALLBACK
# =============================================================================

def _compute_from_area(
    roof_area_sqm: float,
    floor_area_sqm: float,
    floors: int,
    has_basement: bool,
    basement_area_sqm: float,
) -> WaterproofingResult:
    """
    Estimate waterproofing quantities using area-based rules when no room
    data is available.
    """
    result = WaterproofingResult(mode="area_estimate")
    result.warnings.append(
        "No room data provided; waterproofing quantities estimated from floor "
        "area using rule-of-thumb factors "
        "(wet area = 15% of floor area, roof = floor area / floors). "
        "Verify against actual room schedule."
    )

    wet_floor_area   = floor_area_sqm * _FALLBACK_WET_FRACTION
    effective_roof   = roof_area_sqm if roof_area_sqm > 0 else (floor_area_sqm / max(floors, 1))
    kitchen_area     = floor_area_sqm * 0.05   # typical kitchen ~5% of floor plate
    sealant_lm       = floor_area_sqm * _EXPANSION_JOINT_FACTOR

    # Estimated wet room perimeter from area
    wet_perimeter_est = 4.0 * math.sqrt(wet_floor_area) if wet_floor_area > 0 else 0.0

    result.wet_area_sqm  = round(wet_floor_area, 2)
    result.roof_area_sqm = round(effective_roof, 2)

    items: List[dict] = [
        _make_item(
            description=(
                "Crystalline waterproofing compound to toilet/bathroom floors "
                "(incl. 50 mm skirting upturn)"
            ),
            qty=wet_floor_area * _WET_FLOOR_UPLIFT,
            unit="sqm",
            spec="IS 2645 / manufacturer's specification",
        ),
        _make_item(
            description=(
                "Polymer-modified cement mortar waterproofing to toilet/bathroom "
                "walls (300 mm upturn above floor level)"
            ),
            qty=wet_perimeter_est * _WALL_WP_UPTURN_HEIGHT,
            unit="sqm",
            spec="IS 2645",
        ),
        _make_item(
            description=(
                "APP modified bitumen membrane waterproofing to terrace/roof "
                "(incl. 300 mm upturns at parapet/kerb)"
            ),
            qty=effective_roof * _ROOF_UPLIFT,
            unit="sqm",
            spec="IS 15396",
        ),
        _make_item(
            description=(
                "Crystalline waterproofing compound to kitchen floor "
                "(incl. 50 mm skirting upturn)"
            ),
            qty=kitchen_area * _KITCHEN_FLOOR_UPLIFT,
            unit="sqm",
            spec="IS 2645 / manufacturer's specification",
        ),
        _make_item(
            description=(
                "Polyurethane sealant to expansion joints and construction joints "
                "(est. 0.5 lm per 10 sqm floor area)"
            ),
            qty=sealant_lm,
            unit="lm",
            spec="IS 11433",
        ),
        _make_item(
            description=(
                "Polymer cement grouting / sealing at wall-floor junction in "
                "wet areas (cove formation)"
            ),
            qty=wet_perimeter_est,
            unit="lm",
            spec="IS 2645",
        ),
    ]

    if has_basement and basement_area_sqm > 0:
        items.append(_make_item(
            description=(
                "Bituminous tanking waterproofing to basement walls and floor "
                "(multi-coat system, floor + walls)"
            ),
            qty=basement_area_sqm * _BASEMENT_AREA_FACTOR,
            unit="sqm",
            spec="IS 1580",
        ))

    result.line_items = [i for i in items if i["qty"] > 0]
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_waterproofing_takeoff(
    rooms: List[Any],
    roof_area_sqm: float = 0.0,
    floor_area_sqm: float = 0.0,
    floors: int = 1,
    has_basement: bool = False,
    basement_area_sqm: float = 0.0,
) -> WaterproofingResult:
    """
    Compute waterproofing quantities for an Indian construction project.

    Args:
        rooms:              List of finish_takeoff.RoomData objects (may be empty).
        roof_area_sqm:      Known terrace/roof area (sqm). If 0, derived from
                            terrace rooms or estimated from floor_area_sqm / floors.
        floor_area_sqm:     Total gross floor area (sqm). Used for fallback estimates
                            and expansion joint sealant quantities.
        floors:             Number of floors (storeys). Default 1.
        has_basement:       Whether the building has a basement requiring tanking WP.
        basement_area_sqm:  Basement footprint area (sqm). Required when has_basement
                            is True for accurate tanking quantities.

    Returns:
        WaterproofingResult with line_items, mode, warnings, and summary totals.
    """
    floors             = max(floors, 1)
    floor_area_sqm     = max(floor_area_sqm or 0.0, 0.0)
    roof_area_sqm      = max(roof_area_sqm or 0.0, 0.0)
    basement_area_sqm  = max(basement_area_sqm or 0.0, 0.0)

    if has_basement and basement_area_sqm <= 0:
        logger.warning(
            "waterproofing_takeoff: has_basement=True but basement_area_sqm=0; "
            "basement tanking WP will be skipped unless basement rooms are found."
        )

    # Determine whether we have usable room data
    usable_rooms = [
        r for r in (rooms or [])
        if (_safe_area(r) > 0 or _room_perimeter(r) > 0)
    ]

    if usable_rooms:
        result = _compute_from_rooms(
            rooms=usable_rooms,
            roof_area_sqm=roof_area_sqm,
            floor_area_sqm=floor_area_sqm,
            floors=floors,
            has_basement=has_basement,
            basement_area_sqm=basement_area_sqm,
        )
        dropped = len(rooms or []) - len(usable_rooms)
        if dropped > 0:
            result.warnings.append(
                f"{dropped} room(s) had no area or dimension data and were excluded."
            )
    else:
        if not floor_area_sqm:
            logger.warning(
                "waterproofing_takeoff: no rooms and no floor_area_sqm — "
                "returning empty result."
            )
            result = WaterproofingResult(mode="area_estimate")
            result.warnings.append(
                "No room data and no floor area provided. "
                "Cannot compute waterproofing quantities."
            )
            return result

        result = _compute_from_area(
            roof_area_sqm=roof_area_sqm,
            floor_area_sqm=floor_area_sqm,
            floors=floors,
            has_basement=has_basement,
            basement_area_sqm=basement_area_sqm,
        )

    logger.info(
        "waterproofing_takeoff complete | mode=%s | items=%d | "
        "wet_area=%.1f sqm | roof=%.1f sqm",
        result.mode,
        len(result.line_items),
        result.wet_area_sqm,
        result.roof_area_sqm,
    )

    return result
