"""
Scope Disaggregator — splits multi-building EPC tender scopes into per-building
parameter sets for accurate QTO.

Problem: EMRS, NESTS, campus tenders describe 5–7 distinct buildings in one
scope paragraph. If we feed the total area to every QTO module, proportions
come out wrong (school has low wall-to-floor ratio, hostel has high ratio).

Solution: parse the scope, detect distinct building components, assign area
norms per component, return a list of BuildingScope objects. Each is run
through QTO independently and aggregated.

Area norms per CPWD DSR 2023 / WAPCOS standard drawings:
  hostel:         15 sqm/student
  school:         8  sqm/student
  dining:         1.5 sqm/student (combined dining + kitchen)
  staff_quarters: 65 sqm/unit (Type-III Govt quarters)
  office:         12 sqm/person
  laboratory:     20 sqm/person
  library:        3  sqm/reader
  auditorium:     1.5 sqm/seat
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Area norms (sqm per occupant/unit, default floors)
# ---------------------------------------------------------------------------

_NORMS: Dict[str, Tuple[float, int]] = {
    "hostel":        (15.0, 4),
    "school":        (8.0,  2),
    "dining":        (1.5,  1),
    "staff_quarters":(65.0, 3),
    "residential":   (60.0, 3),
    "office":        (12.0, 4),
    "laboratory":    (20.0, 3),
    "library":       (3.0,  3),
    "auditorium":    (1.5,  2),
    "academic":      (10.0, 4),
    "hospital":      (50.0, 3),
    "utility":       (0.0,  1),   # substation, STP, gate — fixed area
}

# Fixed area estimates for utility structures when no occupancy stated (sqm)
_UTILITY_AREAS: Dict[str, float] = {
    "substation":   120.0,
    "stp":          200.0,
    "etp":          200.0,
    "pump house":   80.0,
    "gate":         30.0,
    "security":     30.0,
    "guard room":   20.0,
    "parking":      500.0,
}

_MIN_AREA = 100.0  # sqm — below this a component is too small to QTO separately


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BuildingScope:
    """Parameters for a single building / component in a multi-building scope."""
    building_type: str
    label: str                    # human-readable name e.g. "Boys Hostel"
    occupancy: int = 0            # students / beds / seats
    units: int = 0                # for quarters / blocks
    total_area_sqm: float = 0.0
    floors: int = 1
    source: str = "norm_based"    # "stated" if area explicitly in text


@dataclass
class DisaggregatedScope:
    buildings: List[BuildingScope] = field(default_factory=list)
    total_area_sqm: float = 0.0
    warnings: List[str] = field(default_factory=list)
    single_building: bool = False  # True if scope is a single building


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_BUILDING_PATTERNS = [
    # (regex, building_type, label_template)
    (r"(\d{2,4})[- ](?:capacity|seated|seater|student)s?\s+boys?[- ]hostel", "hostel", "Boys Hostel ({n} students)"),
    (r"(\d{2,4})[- ](?:capacity|seated|seater|student)s?\s+girls?[- ]hostel", "hostel", "Girls Hostel ({n} students)"),
    (r"boys?[- ]hostel[^,\n]{0,30}?(\d{2,4})\s*student", "hostel", "Boys Hostel ({n} students)"),
    (r"girls?[- ]hostel[^,\n]{0,30}?(\d{2,4})\s*student", "hostel", "Girls Hostel ({n} students)"),
    (r"(\d{2,4})[- ](?:capacity|seated|seater|student)s?\s+hostel", "hostel", "Hostel ({n} students)"),
    (r"hostel[^,\n]{0,30}?(\d{2,4})\s*student", "hostel", "Hostel ({n} students)"),
    (r"(\d{2,4})\s+seated\s+\w+\s+hostel", "hostel", "Hostel ({n} students)"),
    (r"school\s+building", "school", "School Building"),
    (r"academic\s+block", "academic", "Academic Block"),
    (r"kitchen\s*[&and]+\s*din(?:ing)?", "dining", "Kitchen & Dining Block"),
    (r"dining\s*(?:hall|block|complex)", "dining", "Dining Hall"),
    (r"(\d+)\s+blocks?\s+(?:of\s+)?type[- ]*(?:iii|iv|v|i|ii)\s+quarters?", "staff_quarters", "{n} Blocks Type-III Quarters"),
    (r"type[- ]*(?:iii|iv|v|i|ii)\s+quarters?[^,\n]{0,20}?(\d+)\s+(?:nos?|blocks?|units?)", "staff_quarters", "Type-III Quarters ({n} units)"),
    (r"residential\s+quarters?", "residential", "Residential Quarters"),
    (r"staff\s+quarters?", "staff_quarters", "Staff Quarters"),
    (r"faculty\s+(?:residenc|quarters?|housing)", "staff_quarters", "Faculty Residences"),
    (r"laboratory\s+block", "laboratory", "Laboratory Block"),
    (r"library\s+block", "library", "Library Block"),
    (r"auditorium", "auditorium", "Auditorium"),
    (r"admin(?:istrative)?\s+(?:block|building|complex)", "office", "Administrative Block"),
    (r"office\s+(?:building|block|complex)", "office", "Office Building"),
    (r"hospital\b", "hospital", "Hospital"),
    (r"substation", "utility", "Electrical Substation"),
    (r"\bstp\b|sewage\s+treatment", "utility", "STP"),
    (r"\betp\b|effluent\s+treatment", "utility", "ETP"),
    (r"pump\s*house", "utility", "Pump House"),
    (r"security\s*(?:cabin|post|room)|guard\s*room", "utility", "Security Post"),
]


def _extract_occupancy_from_text(text: str) -> int:
    m = re.search(r"(\d{2,4})\s*(?:capacity|seated|seater|students?|beds?|seats?)", text, re.I)
    if m:
        return int(m.group(1))
    return 0


def _extract_floors_from_text(text: str) -> int:
    m = re.search(r"[gG]\s*\+\s*(\d+)|(\d+)\s*storeyed?|ground\s*\+\s*(\d+)", text, re.I)
    if m:
        val = int(next(g for g in m.groups() if g is not None))
        return val + 1
    return 0


def _area_for_component(btype: str, occupancy: int, units: int, label: str = "") -> Tuple[float, str]:
    """Return (area_sqm, source) for a building component."""
    norm_sqm, _ = _NORMS.get(btype, (12.0, 3))
    if btype == "utility":
        # Match the specific utility type from the label
        label_lower = label.lower()
        for key, fixed_area in _UTILITY_AREAS.items():
            if key in label_lower:
                return fixed_area, "fixed_norm"
        return _UTILITY_AREAS.get("substation", 120.0), "fixed_norm"
    if occupancy > 0:
        return max(occupancy * norm_sqm, _MIN_AREA), "norm_based"
    if units > 0:
        return max(units * norm_sqm, _MIN_AREA), "norm_based"
    # fallback: use a sensible default
    return max(norm_sqm * 20, _MIN_AREA), "norm_based"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def disaggregate_scope(
    page_texts: List[Tuple[int, str, str]],
    max_pages: int = 6,
    fallback_area_sqm: float = 0.0,
    fallback_floors: int = 1,
) -> DisaggregatedScope:
    """
    Parse scope text and return per-building parameters.

    Parameters
    ----------
    page_texts : list of (page_idx, text, doc_type)
    max_pages  : pages to scan (scope is always near the front)
    fallback_area_sqm : total area to use if disaggregation produces nothing
    fallback_floors   : floor count fallback

    Returns
    -------
    DisaggregatedScope
    """
    result = DisaggregatedScope()

    combined = "\n".join(t for _, t, _ in page_texts[:max_pages]).lower()
    if not combined.strip():
        result.warnings.append("No text in first pages — cannot disaggregate scope.")
        result.single_building = True
        return result

    buildings: List[BuildingScope] = []

    for pattern, btype, label_tmpl in _BUILDING_PATTERNS:
        for m in re.finditer(pattern, combined, re.I):
            # Extract occupancy from capture group or surrounding text
            raw_n = 0
            if m.lastindex and m.lastindex >= 1:
                try:
                    raw_n = int(m.group(1))
                except (ValueError, IndexError):
                    pass

            # Extract from wider context window (±100 chars)
            ctx_start = max(0, m.start() - 60)
            ctx_end   = min(len(combined), m.end() + 60)
            ctx = combined[ctx_start:ctx_end]

            occupancy = raw_n if btype in ("hostel", "school", "dining") else 0
            units     = raw_n if btype in ("staff_quarters", "residential") else 0

            ctx_floors = _extract_floors_from_text(ctx)
            _, default_floors = _NORMS.get(btype, (12.0, 3))
            floors = ctx_floors if ctx_floors > 0 else default_floors

            label = label_tmpl.replace("{n}", str(raw_n)) if raw_n else label_tmpl.replace(" ({n} students)", "").replace(" ({n} units)", "")

            area, src = _area_for_component(btype, occupancy, units, label)

            # Avoid duplicate components (same type + similar occupancy)
            duplicate = any(
                b.building_type == btype
                and abs(b.occupancy - occupancy) < 20
                and abs(b.total_area_sqm - area) < area * 0.2
                for b in buildings
            )
            if not duplicate:
                buildings.append(BuildingScope(
                    building_type=btype,
                    label=label,
                    occupancy=occupancy,
                    units=units,
                    total_area_sqm=area,
                    floors=floors,
                    source=src,
                ))
                logger.debug("scope_disaggregator: detected %s area=%.0f floors=%d occ=%d",
                             label, area, floors, occupancy)

    if not buildings:
        result.single_building = True
        result.warnings.append(
            "No multi-building patterns found — treating as single building."
        )
        if fallback_area_sqm > 0:
            result.total_area_sqm = fallback_area_sqm
        return result

    result.buildings = buildings
    result.total_area_sqm = round(sum(b.total_area_sqm for b in buildings), 2)
    result.single_building = len(buildings) == 1

    logger.info(
        "scope_disaggregator: %d buildings detected, total area %.0f sqm",
        len(buildings), result.total_area_sqm,
    )
    return result


def run_qto_for_scope(
    scope: DisaggregatedScope,
    fallback_area_sqm: float = 0.0,
    fallback_floors: int = 1,
) -> List[dict]:
    """
    Run all QTO modules for each building in the scope and aggregate items.

    Returns a flat list of BOQ items with building label in description prefix.
    Items from the same trade/description across buildings are NOT merged —
    keeps per-building traceability.
    """
    from .qto.wall_area_calculator import compute_wall_areas
    from .qto.brickwork_takeoff    import run_brickwork_takeoff
    from .qto.plaster_takeoff      import run_plaster_takeoff
    from .qto.earthwork_takeoff    import run_earthwork_takeoff
    from .qto.painting_takeoff     import run_painting_takeoff
    from .qto.waterproofing_takeoff import run_waterproofing_takeoff

    all_items: List[dict] = []

    buildings = scope.buildings
    if not buildings:
        # Single-building fallback
        if fallback_area_sqm > 0:
            buildings = [BuildingScope(
                building_type="academic",
                label="Main Building",
                total_area_sqm=fallback_area_sqm,
                floors=fallback_floors,
            )]
        else:
            return []

    # Earthwork is for the whole site, run once
    total_footprint = sum(b.total_area_sqm / max(b.floors, 1) for b in buildings)
    if total_footprint > 0:
        ew = run_earthwork_takeoff(floor_area_sqm=total_footprint, floors=1)
        for item in ew.line_items:
            tagged = dict(item)
            tagged["building"] = "Site Works"
            all_items.append(tagged)

    for bldg in buildings:
        area  = bldg.total_area_sqm
        floors = bldg.floors
        btype  = bldg.building_type
        prefix = bldg.label

        if area <= 0:
            continue

        wall_areas = compute_wall_areas(area, floors, btype)

        for fn, kwargs in [
            (run_brickwork_takeoff,    dict(floor_area_sqm=area, floors=floors, building_type=btype, wall_areas=wall_areas)),
            (run_plaster_takeoff,      dict(floor_area_sqm=area, floors=floors, building_type=btype, wall_areas=wall_areas)),
            (run_painting_takeoff,     dict(rooms=[], floor_area_sqm=area, floors=floors)),
            (run_waterproofing_takeoff,dict(rooms=[], floor_area_sqm=area, floors=floors)),
        ]:
            try:
                res = fn(**kwargs)
                for item in res.line_items:
                    tagged = dict(item)
                    tagged["building"] = prefix
                    all_items.append(tagged)
            except Exception as e:
                logger.warning("scope QTO failed for %s / %s: %s", prefix, fn.__name__, e)

    logger.info("scope_disaggregator: generated %d items across %d buildings",
                len(all_items), len(buildings))
    return all_items
