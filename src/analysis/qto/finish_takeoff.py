"""
Finish Takeoff — Room-name based quantity inference.

Algorithm:
1. Extract room names + dimensions from plan page OCR text
2. Match room names to finish schedule entries
3. Generate quantity line items (sqm floor, sqm walls, sqm ceiling) per room

This is the schedule-first QTO approach: no CV, no geometry — pure text
extraction from plan drawing annotations and schedule data already extracted
by the pipeline.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ROOM EXTRACTION
# =============================================================================

# Room name patterns
_ROOM_NAME_RE = re.compile(
    r'\b(BEDROOM|BED ROOM|MASTER BEDROOM|LIVING|DRAWING|DINING|KITCHEN|'
    r'TOILET|BATHROOM|BATH ROOM|WC|LOBBY|CORRIDOR|PASSAGE|STAIRCASE|STAIR|'
    r'VERANDAH|BALCONY|TERRACE|STORE|UTILITY|STUDY|OFFICE|HALL|FOYER|'
    r'RECEPTION|CONFERENCE|CABIN|PANTRY|LIFT|SHAFT|RAMP|PARKING|GARAGE|'
    r'PLANT ROOM|PUMP ROOM|GENERATOR|SUBSTATION|TOILET BLOCK|COMMON TOILET|'
    r'GENTS|LADIES|ACCESSIBLE)\b(?:\s+\d{1,2}(?!\d))?',
    re.IGNORECASE,
)

# Dimension patterns: "3600 x 4200", "3.6 x 4.2", "3600X4200"
_DIM_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)'
)

# Explicit area patterns: "18.00 sqm", "194 sft", "AREA = 18.00"
_AREA_RE = re.compile(
    r'(?:area\s*[=:]?\s*)?(\d+(?:\.\d+)?)\s*(sqm|sq\.m|sq\.ft|sft|m2)',
    re.IGNORECASE,
)


@dataclass
class RoomData:
    name: str                        # normalised room name e.g. "BEDROOM 1"
    raw_name: str                    # as found in text
    area_sqm: Optional[float]        # computed or extracted area in sqm
    dim_l: Optional[float]           # length in mm or m (ambiguous)
    dim_w: Optional[float]           # width in mm or m (ambiguous)
    source_page: int
    confidence: float                # 0.4 if dims-only, 0.8 if explicit sqm


def _dims_to_sqm(l: float, w: float) -> float:
    """Convert dimension pair to sqm — handle mm vs m heuristically."""
    if l > 100 and w > 100:         # likely mm
        return round((l / 1000) * (w / 1000), 2)
    elif l > 10 and w > 10:         # likely cm
        return round((l / 100) * (w / 100), 2)
    else:                            # likely m
        return round(l * w, 2)


def extract_rooms_from_plan(text: str, source_page: int) -> List[RoomData]:
    """Extract room names and areas from a floor plan page's OCR text."""
    rooms = []
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if line contains a room name
        m_room = _ROOM_NAME_RE.search(line)
        if not m_room:
            i += 1
            continue

        room_name = m_room.group(0).strip().upper()
        raw_name = m_room.group(0).strip()

        # Look for area/dimensions in same line or next 3 lines
        search_text = ' '.join(lines[i:min(i + 4, len(lines))])

        area_sqm = None
        dim_l = dim_w = None
        confidence = 0.4

        # Try explicit area first
        m_area = _AREA_RE.search(search_text)
        if m_area:
            val = float(m_area.group(1))
            unit = m_area.group(2).lower()
            if 'ft' in unit or 'sft' in unit:
                area_sqm = round(val * 0.0929, 2)  # sqft -> sqm
            else:
                area_sqm = val
            confidence = 0.8
        else:
            # Try dimension pair
            m_dim = _DIM_RE.search(search_text)
            if m_dim:
                dim_l = float(m_dim.group(1))
                dim_w = float(m_dim.group(2))
                area_sqm = _dims_to_sqm(dim_l, dim_w)
                confidence = 0.55

        if room_name and (area_sqm or dim_l):
            rooms.append(RoomData(
                name=room_name,
                raw_name=raw_name,
                area_sqm=area_sqm,
                dim_l=dim_l,
                dim_w=dim_w,
                source_page=source_page,
                confidence=confidence,
            ))

        i += 1

    return rooms


# =============================================================================
# FINISH SCHEDULE LOOKUP HELPERS
# =============================================================================

# Standard finish types and their BOQ descriptions
_FINISH_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "floor": {
        "vitrified": "Providing & fixing {size}mm vitrified tiles in flooring with CM 1:4",
        "ceramic":   "Providing & fixing {size}mm ceramic tiles in flooring with CM 1:4",
        "marble":    "Providing & fixing marble flooring {thickness}mm thick with CM 1:4",
        "granite":   "Providing & fixing granite flooring {thickness}mm thick with CM 1:4",
        "kota":      "Providing & fixing Kota stone flooring with CM 1:4",
        "ips":       "IPS flooring {thickness}mm thick with neat cement finish",
        "cc":        "CC flooring M15 grade {thickness}mm thick",
        "default":   "Flooring as per finish schedule",
    },
    "wall": {
        "ceramic":   "Providing & fixing {size}mm ceramic tile dado with CM 1:3",
        "paint":     "Painting of walls with 2 coats acrylic emulsion over primer",
        "obd":       "Applying oil bound distemper 2 coats over primer coat",
        "texture":   "Applying texture paint finish to walls",
        "default":   "Wall finish as per finish schedule",
    },
    "ceiling": {
        "pop":       "Providing false ceiling with POP on MS framework",
        "gypsum":    "Providing gypsum board false ceiling on MS framework",
        "paint":     "Painting of ceiling with 2 coats acrylic emulsion",
        "default":   "Ceiling finish as per finish schedule",
    },
}

# Typical floor-to-ceiling height by room type (metres) — for wall area calc
_ROOM_HEIGHT: Dict[str, float] = {
    "BEDROOM": 3.0, "MASTER BEDROOM": 3.0, "LIVING": 3.2, "DRAWING": 3.2,
    "DINING": 3.0, "KITCHEN": 2.7, "TOILET": 2.7, "BATHROOM": 2.7, "WC": 2.7,
    "LOBBY": 3.5, "CORRIDOR": 2.8, "PASSAGE": 2.8, "STAIRCASE": 3.0,
    "OFFICE": 3.0, "CONFERENCE": 3.2, "RECEPTION": 3.5,
}
_DEFAULT_HEIGHT = 3.0


def _get_room_height(room_name: str) -> float:
    """Estimate ceiling height for a room type."""
    name_upper = room_name.upper()
    for key, height in _ROOM_HEIGHT.items():
        if key in name_upper:
            return height
    return _DEFAULT_HEIGHT


def _wall_area_from_floor(floor_sqm: float, room_name: str) -> float:
    """Rough wall area: perimeter x height. Assume square-ish room."""
    side = math.sqrt(floor_sqm)
    perimeter = 4 * side
    height = _get_room_height(room_name)
    # Deduct ~15% for openings (door/window)
    return round(perimeter * height * 0.85, 2)


# =============================================================================
# ITEM GENERATION
# =============================================================================

def generate_finish_items(
    rooms: List[RoomData],
    finish_schedules: List[dict],
    source: str = "qto_finish",
) -> List[dict]:
    """
    Generate priceable line items from room data + finish schedules.

    For each room with known area:
    - Floor item (sqm)
    - Wall item (sqm, estimated from floor area)
    - Ceiling item (sqm = floor area)

    Returns list of item dicts compatible with build_line_items() input.
    """
    items = []

    # Build finish lookup by room name from schedules
    # Finish schedule items typically have: room_name/location, finish_type, material
    finish_by_room: Dict[str, dict] = {}
    for fs in (finish_schedules or []):
        location = (
            fs.get("location") or fs.get("room") or fs.get("area") or ""
        ).upper().strip()
        if location:
            finish_by_room[location] = fs

    for room in rooms:
        if not room.area_sqm or room.area_sqm <= 0:
            continue

        # Try to find matching finish schedule entry
        finish_entry = finish_by_room.get(room.name) or {}

        # Floor item
        floor_desc = _resolve_finish_desc("floor", finish_entry)
        items.append({
            "item_no":          None,
            "description":      f"{floor_desc} — {room.name}",
            "unit":             "sqm",
            "unit_inferred":    False,
            "qty":              room.area_sqm,
            "rate":             None,
            "trade":            _room_to_trade(room.name, "floor"),
            "section":          f"FLOOR FINISHES — {room.name}",
            "source_page":      room.source_page,
            "source":           source,
            "confidence":       room.confidence * 0.9,
            "is_priceable":     True,
            "priceable_reason": "priceable",
            "qto_method":       "room_area",
        })

        # Wall item (estimated)
        wall_sqm = _wall_area_from_floor(room.area_sqm, room.name)
        wall_desc = _resolve_finish_desc("wall", finish_entry)
        items.append({
            "item_no":          None,
            "description":      f"{wall_desc} — {room.name}",
            "unit":             "sqm",
            "unit_inferred":    False,
            "qty":              wall_sqm,
            "rate":             None,
            "trade":            "finishes",
            "section":          f"WALL FINISHES — {room.name}",
            "source_page":      room.source_page,
            "source":           source,
            "confidence":       room.confidence * 0.7,   # lower — wall area is estimated
            "is_priceable":     True,
            "priceable_reason": "priceable",
            "qto_method":       "room_perimeter_estimate",
        })

        # Ceiling item
        ceiling_desc = _resolve_finish_desc("ceiling", finish_entry)
        items.append({
            "item_no":          None,
            "description":      f"{ceiling_desc} — {room.name}",
            "unit":             "sqm",
            "unit_inferred":    False,
            "qty":              room.area_sqm,
            "rate":             None,
            "trade":            "finishes",
            "section":          f"CEILING FINISHES — {room.name}",
            "source_page":      room.source_page,
            "source":           source,
            "confidence":       room.confidence * 0.85,
            "is_priceable":     True,
            "priceable_reason": "priceable",
            "qto_method":       "room_area",
        })

    return items


def _resolve_finish_desc(element: str, finish_entry: dict) -> str:
    """Build a description string for floor/wall/ceiling from a finish entry."""
    lookup = _FINISH_DESCRIPTIONS.get(element, {})
    if not finish_entry:
        return lookup.get("default", f"{element.title()} finish as per schedule")

    material = (
        finish_entry.get("material") or finish_entry.get("finish_type") or ""
    ).lower()
    size = finish_entry.get("size") or finish_entry.get("tile_size") or "600x600"
    thickness = finish_entry.get("thickness") or "12"

    for key, template in lookup.items():
        if key in material:
            return template.format(size=size, thickness=thickness)

    return finish_entry.get("description") or lookup.get(
        "default", f"{element.title()} finish"
    )


def _room_to_trade(room_name: str, element: str) -> str:
    name_upper = room_name.upper()
    if any(k in name_upper for k in ("TOILET", "BATHROOM", "WC")):
        return "plumbing"
    return "finishes"


# =============================================================================
# FULL PIPELINE ENTRY POINT
# =============================================================================

@dataclass
class FinishResult:
    """Standard result object for finish takeoff (mirrors other QTO Result types)."""
    rooms: List[RoomData] = field(default_factory=list)
    line_items: List[dict] = field(default_factory=list)
    mode: str = "room_name"
    warnings: List[str] = field(default_factory=list)

    # Convenience accessors used by pipeline destructuring shim
    def as_tuple(self) -> Tuple[List[RoomData], List[dict]]:
        return self.rooms, self.line_items


def run_finish_takeoff(
    page_texts: List[Tuple[int, str, str]],   # (page_idx, text, doc_type)
    finish_schedules: List[dict],
) -> Tuple[List[RoomData], List[dict]]:
    """
    Full pipeline: extract rooms from all plan pages, generate finish items.

    Args:
        page_texts: list of (page_idx, ocr_text, doc_type) for all processed pages
        finish_schedules: finish schedule items from extraction_result.schedules
                          filtered to schedule_type == "finish"

    Returns:
        (rooms_found, finish_line_items)

    Note: call run_finish_takeoff_result() to get a FinishResult object instead.
    """
    all_rooms: List[RoomData] = []

    for page_idx, text, doc_type in page_texts:
        if doc_type not in ("plan", "layout", "floor_plan", "drawing"):
            continue
        rooms = extract_rooms_from_plan(text, source_page=page_idx)
        all_rooms.extend(rooms)

    finish_items = generate_finish_items(all_rooms, finish_schedules)
    return all_rooms, finish_items


def run_finish_takeoff_result(
    page_texts: List[Tuple[int, str, str]],
    finish_schedules: List[dict],
) -> "FinishResult":
    """Same as run_finish_takeoff() but returns a FinishResult (standard QTO pattern)."""
    rooms, items = run_finish_takeoff(page_texts, finish_schedules)
    return FinishResult(rooms=rooms, line_items=items)


# =============================================================================
# FLOORING TAKEOFF — AREA-BASED QTO
# =============================================================================

# Building types that require IPS flooring in service/store areas
_IPS_BUILDING_TYPES = ("hostel", "school", "academic")

# ── Flooring coverage by building type ────────────────────────────────────────
_VITRIFIED_COVERAGE: dict = {
    "hostel":         0.80,   # rooms vitrified, toilets ceramic, corridors Kota
    "school":         0.70,   # classrooms vitrified, labs anti-skid, corridors Kota
    "academic":       0.65,   # lecture halls + labs; labs get anti-skid or IPS
    "office":         0.85,
    "hospital":       0.75,   # clinical: vinyl/anti-bacterial; wards: vitrified
    "residential":    0.90,
    "dining":         0.80,
    "staff_quarters": 0.85,
    "library":        0.85,
    "auditorium":     0.80,
    "workshop":       0.60,
    "research":       0.65,
    "default":        0.82,
}

_WET_AREA_FRACTION: dict = {
    "hostel":         0.12,   # attached toilets per room + common toilets
    "school":         0.08,
    "academic":       0.07,
    "office":         0.06,
    "hospital":       0.18,   # ward toilets, utility, sluice rooms
    "residential":    0.10,
    "dining":         0.06,
    "staff_quarters": 0.10,
    "default":        0.09,
}

# Skirting factor (running metres per sqm of floor area)
# Derived from avg room size: hostel rooms ~12 sqm (high perimeter/area),
# academic lecture halls ~80 sqm (low perimeter/area)
_SKIRTING_RM_PER_SQM: dict = {
    "hostel":         0.50,
    "school":         0.30,
    "academic":       0.25,
    "office":         0.20,
    "hospital":       0.35,
    "residential":    0.40,
    "staff_quarters": 0.40,
    "dining":         0.18,
    "library":        0.18,
    "auditorium":     0.15,
    "default":        0.30,
}


@dataclass
class FlooringResult:
    line_items:      List[dict] = field(default_factory=list)
    mode:            str = "area_estimate"
    warnings:        List[str] = field(default_factory=list)
    total_floor_sqm: float = 0.0   # total vitrified tile area (primary finish)


def _make_flooring_item(description: str, qty: float, unit: str) -> dict:
    return {
        "description": description,
        "trade":       "finishing",
        "unit":        unit,
        "qty":         round(qty, 2),
        "source":      "flooring_takeoff",
    }


def run_flooring_takeoff(
    floor_area_sqm: float,
    floors: int = 1,
    has_basement: bool = False,
    building_type: str = "residential",
    rooms: list = None,
) -> FlooringResult:
    """
    Compute floor-finishing quantities for an Indian construction project
    using area-based rule-of-thumb norms.

    Args:
        floor_area_sqm: Gross floor area of a single floor (sqm).
        floors:         Number of storeys above ground.  Default 1.
        has_basement:   Retained for future use; does not affect current
                        quantities (basement floors typically IPS/PCC).
        building_type:  Building use classification.  Drives inclusion of
                        IPS service-area flooring item for hostel/school/
                        academic buildings.
        rooms:          Optional list of RoomData objects.  Not used in the
                        current area-estimate implementation but accepted for
                        API compatibility with other QTO modules.

    Returns:
        FlooringResult with line_items, mode, warnings, and total_floor_sqm.
    """
    # --- Input sanitisation --------------------------------------------------
    floor_area_sqm = max(float(floor_area_sqm or 0.0), 0.0)
    floors         = max(int(floors or 1), 1)
    building_type  = (building_type or "residential").strip().lower()

    result = FlooringResult(mode="area_estimate")
    result.warnings.append(
        "Flooring quantities are area-based estimates using rule-of-thumb "
        "coverage factors.  Verify against actual room schedule and finish "
        "drawings before submission."
    )

    if floor_area_sqm <= 0:
        logger.warning(
            "flooring_takeoff: floor_area_sqm=%.2f — returning empty result.",
            floor_area_sqm,
        )
        result.warnings.append(
            "floor_area_sqm is zero or not provided; no quantities computed."
        )
        return result

    items: List[dict] = []

    # -------------------------------------------------------------------------
    # Item 1 — Vitrified tile flooring (primary finish, coverage by building type)
    # -------------------------------------------------------------------------
    _cov = _VITRIFIED_COVERAGE.get(building_type, _VITRIFIED_COVERAGE["default"])
    vitrified_sqm = floor_area_sqm * floors * _cov
    result.total_floor_sqm = round(vitrified_sqm, 2)
    items.append(_make_flooring_item(
        description=(
            "Providing & fixing vitrified tiles 600×600 mm in flooring, "
            "with CM 1:4 (IS 13712), including base preparation and grouting"
        ),
        qty=vitrified_sqm,
        unit="sqm",
    ))

    # -------------------------------------------------------------------------
    # Item 2 — Ceramic tile flooring in wet areas (fraction by building type)
    # -------------------------------------------------------------------------
    _wet_frac = _WET_AREA_FRACTION.get(building_type, _WET_AREA_FRACTION["default"])
    wet_area_sqm = floor_area_sqm * floors * _wet_frac
    items.append(_make_flooring_item(
        description=(
            "Providing & fixing ceramic tiles 300×300 mm in flooring of "
            "toilets/bathrooms, with CM 1:4, including base preparation "
            "and grouting"
        ),
        qty=wet_area_sqm,
        unit="sqm",
    ))

    # -------------------------------------------------------------------------
    # Item 3 — Ceramic dado to toilet walls (perimeter at 1.8 m height)
    # -------------------------------------------------------------------------
    dado_sqm = wet_area_sqm * 4.0
    items.append(_make_flooring_item(
        description=(
            "Providing & fixing ceramic tiles 300×450 mm as dado to toilet/"
            "bathroom walls, height 1.8 m, with CM 1:3, including base "
            "preparation and grouting"
        ),
        qty=dado_sqm,
        unit="sqm",
    ))

    # -------------------------------------------------------------------------
    # Item 4 — Kota/granite landing and staircase treads (15 sqm per floor)
    # -------------------------------------------------------------------------
    stair_sqm = floors * 15.0
    items.append(_make_flooring_item(
        description=(
            "Providing & fixing Kota stone / granite flooring to staircase "
            "treads, risers and landings, with CM 1:4, machine polished "
            "finish including nosing"
        ),
        qty=stair_sqm,
        unit="sqm",
    ))

    # -------------------------------------------------------------------------
    # Item 5 — Vitrified tile skirting 100 mm ht (perimeter estimate by building type)
    # -------------------------------------------------------------------------
    _skirting_fac = _SKIRTING_RM_PER_SQM.get(building_type, _SKIRTING_RM_PER_SQM["default"])
    skirting_rm = floor_area_sqm * floors * _skirting_fac
    items.append(_make_flooring_item(
        description=(
            "Providing & fixing vitrified tile skirting 100 mm height, "
            "matching flooring tile, with CM 1:4, including grouting "
            "and edge polishing"
        ),
        qty=skirting_rm,
        unit="rm",
    ))

    # -------------------------------------------------------------------------
    # Item 6 — IPS flooring to service/store areas (hostel/school/academic only)
    # -------------------------------------------------------------------------
    if building_type in _IPS_BUILDING_TYPES:
        ips_sqm = floor_area_sqm * floors * 0.05
        if ips_sqm > 0:
            items.append(_make_flooring_item(
                description=(
                    "IPS flooring 40 mm thick M15 grade cement concrete to "
                    "service areas, store rooms and utility spaces, with neat "
                    "cement finish"
                ),
                qty=ips_sqm,
                unit="sqm",
            ))

    # Filter out zero-quantity items (defensive)
    result.line_items = [i for i in items if i["qty"] > 0]

    logger.info(
        "flooring_takeoff complete | mode=%s | building_type=%s | "
        "items=%d | primary_floor=%.1f sqm",
        result.mode,
        building_type,
        len(result.line_items),
        result.total_floor_sqm,
    )

    return result
