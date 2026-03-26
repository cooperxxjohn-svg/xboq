"""
Prelims Takeoff — auto-generate preliminary / general conditions BOQ items.

Derives item quantities from project area, floor count, occupancy and
duration.  All computation is norm-based (CPWD / Indian construction practice).

Prelims typically represent 8–12% of total construction cost in Indian
contracts and cover:
  - Site establishment (site office, labour camp, fencing, watchman shed)
  - Plant & machinery (mixers, vibrators, bar bending, scaffolding, hoist)
  - Temporary works (water, electrical, temporary roads)
  - Project management (site engineers, foremen, safety officer)
  - Quality control (cube testing, soil testing, water testing)
  - Insurances & bonds (performance bond, CAR, workmen compensation)

Units and headings follow CPWD Schedule of Rates / DSR conventions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOREY_HEIGHT_M = 3.3          # typical RCC-frame storey height
_EXT_WALL_RATIO  = 0.40         # external wall area ≈ 40% of total floor area / floor
_CONCRETE_NORM   = 0.35         # cum of structural concrete per sqm of built-up area (legacy default)
_DURATION_AREA   = 800          # sqm per month — legacy constant (kept for backward compat)

# CPWD project duration norms (area_sqm_threshold, duration_months)
_DURATION_TABLE: list = [
    (1000,         12),
    (3000,         18),
    (10000,        24),
    (20000,        30),
    (float("inf"), 36),
]

_CONCRETE_NORM_BY_TYPE: dict = {
    "hostel":      0.36,
    "academic":    0.38,
    "school":      0.34,
    "hospital":    0.43,
    "office":      0.38,
    "residential": 0.35,
    "dining":      0.34,
    "sports":      0.28,
    "default":     0.36,
}


def _auto_duration(total_area_sqm: float) -> int:
    """Derive project duration using CPWD schedule norms."""
    for area_max, months in _DURATION_TABLE:
        if total_area_sqm <= area_max:
            return months
    return 36


def _auto_concrete(total_area_sqm: float, building_type: str = "default") -> float:
    """Estimate structural concrete volume (cum) from BUA."""
    norm = _CONCRETE_NORM_BY_TYPE.get(building_type, _CONCRETE_NORM_BY_TYPE["default"])
    return round(total_area_sqm * norm, 1)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PrelimsResult:
    """Outcome of run_prelims_takeoff()."""
    line_items: List[dict] = field(default_factory=list)
    project_duration_months: int = 0
    concrete_volume_cum: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _item(description: str, unit: str, quantity: float) -> dict:
    qty_rounded = round(quantity, 2)
    return {
        "description": description,
        "trade": "prelims",
        "unit": unit,
        "qty":      qty_rounded,    # canonical key for rate_engine.apply_rates()
        "quantity": qty_rounded,    # backward compat for UI tabs
        "source": "norm_based",
        "building": "General",
    }


def _site_establishment(
    total_area_sqm: float,
    occupancy: int,
    floors: int,
) -> List[dict]:
    """Site office, labour camp, boundary fencing, watchman shed."""
    items: List[dict] = []

    # Site office
    site_office_sqm = 100.0 if total_area_sqm >= 5000 else 50.0
    items.append(_item(
        "Providing and fixing prefabricated site office including furniture, "
        "electrical fittings and communication facilities",
        "sqm",
        site_office_sqm,
    ))

    # Labour camp — 4 sqm per occupant, rounded up to nearest 50 sqm, minimum 50
    if occupancy > 0:
        raw_camp = occupancy * 4.0
    else:
        # estimate from floor area: 1 worker per 20 sqm
        raw_camp = (total_area_sqm / 20.0) * 4.0
    camp_sqm = max(50.0, math.ceil(raw_camp / 50.0) * 50.0)
    items.append(_item(
        "Providing and fixing temporary labour camp / accommodation including "
        "sanitation, water supply and lighting",
        "sqm",
        camp_sqm,
    ))

    # Boundary fencing — derive perimeter from floor plate area (ground floor)
    floor_plate_sqm = total_area_sqm / max(floors, 1)
    # approximate perimeter of a square footprint
    perimeter_m = 4.0 * math.sqrt(floor_plate_sqm) * 1.5  # 1.5× for site margin
    items.append(_item(
        "Providing and erecting GI barbed-wire / chain-link boundary fencing "
        "with MS angle posts at site perimeter",
        "rm",
        round(perimeter_m, 1),
    ))

    # Watchman shed — fixed 15 sqm
    items.append(_item(
        "Providing and erecting watchman / security post shed with toilet",
        "sqm",
        15.0,
    ))

    return items


def _plant_machinery(
    total_area_sqm: float,
    floors: int,
    duration_months: int,
    external_wall_area_sqm: float,
) -> List[dict]:
    """Concrete mixers, vibrators, bar bending, scaffolding, tower crane/hoist."""
    items: List[dict] = []

    # Concrete mixer 0.2 cum: 1 per 3000 sqm
    mixer_count = max(1, math.ceil(total_area_sqm / 3000.0))
    items.append(_item(
        "Hire / mobilisation of concrete mixer 0.2 cum capacity",
        "months",
        float(mixer_count * duration_months),
    ))

    # Needle vibrator: same count as mixers
    items.append(_item(
        "Hire / mobilisation of needle vibrator with flexible shaft",
        "months",
        float(mixer_count * duration_months),
    ))

    # Bar bending machine: 1 per 5000 sqm, min 1
    bbs_count = max(1, math.ceil(total_area_sqm / 5000.0))
    items.append(_item(
        "Hire / mobilisation of bar bending and cutting machine",
        "months",
        float(bbs_count * duration_months),
    ))

    # Scaffolding: external_wall_area × 1.2 (multiple use cycles)
    scaffolding_sqm = external_wall_area_sqm * 1.2
    items.append(_item(
        "Providing, erecting and dismantling tubular steel scaffolding for "
        "external works (multiple lifts assumed)",
        "sqm",
        round(scaffolding_sqm, 1),
    ))

    # Tower crane / material hoist: only for ≥5 floors
    if floors >= 5:
        items.append(_item(
            "Mobilisation and operation of tower crane / material hoist "
            "including foundation and dismantling",
            "months",
            float(duration_months),
        ))

    return items


def _temporary_works(
    total_area_sqm: float,
    floors: int,
    duration_months: int,
) -> List[dict]:
    """Temporary water, electrical supply and internal roads."""
    items: List[dict] = []

    items.append(_item(
        "Providing temporary water supply to site including pump, storage tank, "
        "distribution piping and maintenance",
        "months",
        float(duration_months),
    ))

    items.append(_item(
        "Providing temporary electrical supply / power connection to site "
        "including DG set standby, distribution boards and wiring",
        "months",
        float(duration_months),
    ))

    # Temporary roads: ground floor plate area × 0.1
    floor_plate_sqm = total_area_sqm / max(floors, 1)
    temp_road_sqm = floor_plate_sqm * 0.1
    items.append(_item(
        "Providing and maintaining temporary access roads within site "
        "(granular sub-base, compacted)",
        "sqm",
        round(temp_road_sqm, 1),
    ))

    return items


def _project_management(
    total_area_sqm: float,
    duration_months: int,
) -> List[dict]:
    """Site engineer, foreman, safety officer person-months."""
    items: List[dict] = []

    # Site engineer: 1 per 2000 sqm, min 1
    eng_count = max(1, math.ceil(total_area_sqm / 2000.0))
    items.append(_item(
        "Providing qualified site engineer(s) for full-time supervision",
        "person-months",
        float(eng_count * duration_months),
    ))

    # Foreman: 1 per 1500 sqm, min 1
    fm_count = max(1, math.ceil(total_area_sqm / 1500.0))
    items.append(_item(
        "Providing experienced foreman(s) for site supervision and coordination",
        "person-months",
        float(fm_count * duration_months),
    ))

    # Safety officer: 1 if total area > 3000 sqm
    if total_area_sqm > 3000.0:
        items.append(_item(
            "Providing dedicated safety officer for HSE compliance, toolbox talks "
            "and statutory reporting",
            "person-months",
            float(duration_months),
        ))

    return items


def _quality_control(concrete_volume_cum: float) -> List[dict]:
    """Cube testing, soil testing, water testing."""
    items: List[dict] = []

    # Cube testing: 1 set per 30 cum, minimum 10 sets
    cube_sets = max(10, math.ceil(concrete_volume_cum / 30.0))
    items.append(_item(
        "Concrete cube testing (6 cubes per set, tested at 7 and 28 days) "
        "including curing, transport and lab charges",
        "sets",
        float(cube_sets),
    ))

    # Soil testing: fixed 5
    items.append(_item(
        "Subsoil investigation and geotechnical testing "
        "(bore holes, SPT, sieve analysis, compaction tests)",
        "tests",
        5.0,
    ))

    # Water testing: fixed 2
    items.append(_item(
        "Water quality testing for construction and potable use "
        "(chemical and bacteriological analysis)",
        "tests",
        2.0,
    ))

    return items


def _insurances_bonds() -> List[dict]:
    """Performance bond, CAR insurance, workmen compensation — all lump sum."""
    items: List[dict] = []

    items.append(_item(
        "Performance bank guarantee / security deposit charges "
        "(bank commission for BG as per contract conditions)",
        "LS",
        1.0,
    ))

    items.append(_item(
        "Contractor All Risk (CAR) insurance policy for the full contract period "
        "covering material damage and third-party liability",
        "LS",
        1.0,
    ))

    items.append(_item(
        "Workmen compensation insurance as per Workmen's Compensation Act / ESIC "
        "for all labour deployed on site",
        "LS",
        1.0,
    ))

    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_prelims_takeoff(
    total_area_sqm: float,
    floors: int,
    building_type: str = "hostel",
    occupancy: int = 0,
    project_duration_months: int = 0,   # 0 = auto-derive
    concrete_volume_cum: float = 0.0,   # 0 = auto-derive
) -> PrelimsResult:
    """
    Auto-generate preliminary / general conditions BOQ items.

    Parameters
    ----------
    total_area_sqm : float
        Total built-up area across all floors.
    floors : int
        Number of storeys (excluding basement).
    building_type : str
        Descriptive building type (hostel, academic, office …).
    occupancy : int
        Design occupancy (students, beds, etc.).  0 = auto-estimate from area.
    project_duration_months : int
        Contract period in months.  0 = auto-derived from area / CPWD norm.
    concrete_volume_cum : float
        Estimated structural concrete volume.  0 = auto-derived from area norm.

    Returns
    -------
    PrelimsResult
    """
    result = PrelimsResult()
    warnings: List[str] = []

    # ── Guard ──────────────────────────────────────────────────────────────
    if not total_area_sqm or total_area_sqm <= 0:
        warnings.append("total_area_sqm zero or missing — no prelims computed.")
        result.warnings = warnings
        return result

    floors = max(1, floors)

    # ── Derived parameters ─────────────────────────────────────────────────
    if project_duration_months <= 0:
        project_duration_months = _auto_duration(total_area_sqm)
        warnings.append(
            f"project_duration_months auto-derived: {project_duration_months} months "
            f"(CPWD schedule norms, area={total_area_sqm:.0f} sqm)"
        )

    if concrete_volume_cum <= 0:
        concrete_volume_cum = _auto_concrete(total_area_sqm, building_type)
        norm_used = _CONCRETE_NORM_BY_TYPE.get(building_type, _CONCRETE_NORM_BY_TYPE["default"])
        warnings.append(
            f"concrete_volume_cum auto-derived: {concrete_volume_cum} cum "
            f"(norm: {norm_used} cum/sqm for building_type='{building_type}')"
        )

    result.project_duration_months = project_duration_months
    result.concrete_volume_cum     = concrete_volume_cum

    # External wall area (used for scaffolding)
    ext_wall_area = total_area_sqm * _EXT_WALL_RATIO

    # ── Build line items ────────────────────────────────────────────────────
    line_items: List[dict] = []

    line_items.extend(_site_establishment(total_area_sqm, occupancy, floors))
    line_items.extend(_plant_machinery(
        total_area_sqm, floors, project_duration_months, ext_wall_area
    ))
    line_items.extend(_temporary_works(total_area_sqm, floors, project_duration_months))
    line_items.extend(_project_management(total_area_sqm, project_duration_months))
    line_items.extend(_quality_control(concrete_volume_cum))
    line_items.extend(_insurances_bonds())

    result.line_items = line_items
    result.warnings   = warnings

    logger.debug(
        "prelims: area=%.0f sqm, floors=%d, duration=%d mo, items=%d",
        total_area_sqm, floors, project_duration_months, len(line_items),
    )
    return result
