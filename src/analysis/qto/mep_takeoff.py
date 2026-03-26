"""
MEP Takeoff — Text-Only QTO from OCR Schedule Data.

Parses Electrical, Plumbing, and HVAC schedules from drawing/spec OCR text and
generates priceable BOQ line items compatible with the pipeline's line_items format.

Supports Indian construction notation:
  Electrical : Lighting Fixture Schedule, DB/Panel Schedule, Socket Schedule
  Plumbing   : Sanitary/Plumbing Fixture Schedule, drainage schedule
  HVAC       : AHU / FCU / VRV / Split-AC schedules, exhaust fan schedules

Algorithm:
1. Scan pages with relevant doc_types for MEP schedule tables in OCR text.
2. State-machine parser: detect section headers → extract rows.
3. Generate BOQ items: each fixture/equipment → Nos item; ancillaries estimated.
4. Fallback assumption mode when no schedule is found.

Design constraints:
- NO cv2 / OpenCV — pure text regex + arithmetic.
- Generates items in the same dict format as structural_takeoff.py / finish_takeoff.py.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS — ANCILLARY ALLOWANCES
# =============================================================================

# Metres of wiring per light fixture (circuit + switch leg)
_WIRING_PER_FIXTURE_M = 12.0

# Metres of wiring per socket/outlet
_WIRING_PER_SOCKET_M = 8.0

# Metres of conduit per fixture (average run)
_CONDUIT_PER_FIXTURE_M = 6.0

# Metres of internal water supply pipe per plumbing fixture
_PIPE_PER_FIXTURE_M = 4.0

# Metres of drainage pipe per sanitary fixture
_DRAIN_PER_FIXTURE_M = 3.0

# Metres of refrigerant piping per AC/FCU unit
_REFRIGERANT_PER_UNIT_M = 7.0

# MCB+wiring allowed per DB
_MCB_PER_DB = 12


# Building-type MEP fixture density defaults (NBC 2016 + IS standards)
# Tuple: (sqm_per_light, sqm_per_socket, sqm_per_db, sqm_per_person,
#         tr_per_sqm, hvac_threshold_sqm, exhaust_sqm, wc_area_sqm)
_MEP_DEFAULTS_BY_TYPE: dict = {
    "hostel":      (8.0,  6.0, 250, 6,  40, 500,  70, 30),
    "residential": (8.0,  6.0, 300, 8,  35, 1500, 80, 50),
    "hospital":    (6.0,  4.0, 150, 4,  20, 200,  40, 20),
    "office":      (10.0, 5.0, 300, 5,  30, 800,  80, 25),
    "academic":    (9.0,  7.0, 280, 8,  35, 600,  75, 35),
    "industrial":  (12.0, 8.0, 400, 10, 40, 2000, 90, 40),
    "default":     (10.0, 6.0, 300, 7,  30, 1500, 80, 35),
}


def _resolve_mep_defaults(building_type: str) -> tuple:
    """Return (sqm_per_light, sqm_per_socket, sqm_per_db, sqm_per_person,
               tr_per_sqm, hvac_threshold_sqm, exhaust_sqm, wc_area_sqm)."""
    bt = (building_type or "").lower()
    for key in _MEP_DEFAULTS_BY_TYPE:
        if key != "default" and key in bt:
            return _MEP_DEFAULTS_BY_TYPE[key]
    return _MEP_DEFAULTS_BY_TYPE["default"]


# =============================================================================
# SECTION HEADER PATTERNS
# =============================================================================

# ── Electrical ────────────────────────────────────────────────────────────────
_ELEC_FIXTURE_HEADER = re.compile(
    r'(?:LIGHTING|LUMINAIRE|ELECTRICAL\s+FIXTURE|LIGHT\s+FIXTURE|FIXTURE)\s*'
    r'(?:SCHEDULE|LIST|LEGEND|SCHEDULE\s+OF)',
    re.IGNORECASE,
)
_ELEC_SOCKET_HEADER = re.compile(
    r'(?:POWER|SOCKET|OUTLET|PLUG)\s*(?:SCHEDULE|LIST|PLAN)',
    re.IGNORECASE,
)
_ELEC_PANEL_HEADER = re.compile(
    r'(?:PANEL|DB|DISTRIBUTION\s*BOARD|LT\s*PANEL|LOAD)\s*'
    r'(?:SCHEDULE|LIST)',
    re.IGNORECASE,
)

# ── Plumbing ──────────────────────────────────────────────────────────────────
_PLMB_HEADER = re.compile(
    r'(?:SANITARY|PLUMBING|DRAINAGE|WATER\s*SUPPLY)\s*'
    r'(?:FIXTURE|SCHEDULE|LIST|LEGEND)',
    re.IGNORECASE,
)

# ── HVAC ──────────────────────────────────────────────────────────────────────
_HVAC_HEADER = re.compile(
    r'(?:HVAC|AIR\s*(?:CONDITIONING|HANDLING)|AHU|FCU|VRV|SPLIT|EXHAUST|'
    r'MECHANICAL|EQUIPMENT)\s*(?:SCHEDULE|LIST|LEGEND)',
    re.IGNORECASE,
)

# Generic schedule (catches tables labelled just "SCHEDULE")
_GENERIC_SCHED = re.compile(r'\bSCHEDULE\b', re.IGNORECASE)


# =============================================================================
# ROW-LEVEL EXTRACTION PATTERNS
# =============================================================================

# Tag/label like "L1", "WC-1", "AHU-03", "DB-B"
_TAG_PAT = re.compile(
    r'(?<!\w)([A-Z]{1,4}-?\d{1,3}[A-Z]?)(?!\w)',
    re.IGNORECASE,
)

# Quantity: number explicitly followed by a unit keyword ("45 Nos", "2 Sets")
_QTY_WITH_UNIT_PAT = re.compile(
    r'(?<![A-Za-z\d\-])(\d{1,4}(?:\.\d{1,2})?)\s*'
    r'(?:Nos?\.?|Sets?\.?|Ea\.?|Each|Units?\.?|Pcs?\.?)\b',
    re.IGNORECASE,
)

# Wattage / capacity patterns to strip before looking for qty
_STRIP_WATT_CAP = re.compile(
    r'\d+(?:\.\d+)?\s*(?:kW|W|TR|Ton(?:ne)?|CFM|BTU|sqm|mm|cm|m²)\b',
    re.IGNORECASE,
)

# Dimension patterns to strip: "600x600", "300X150", "1500 x 600"
# Allow up to 2 spaces around × to avoid matching "2x    20" (qty mismatch)
_STRIP_DIMENSION = re.compile(r'\d+\s{0,2}[xX]\s{0,2}\d+')

# Pure integer token (used after stripping above noise)
_PURE_INT_TOKEN = re.compile(r'^\d{1,4}$')

# Wattage: "36W", "2x18W", "250 W", "1.5 kW"
_WATT_PAT = re.compile(
    r'(\d{1,4}(?:\.\d)?)\s*(?:kW|W)\b',
    re.IGNORECASE,
)

# Capacity: "1.5 TR", "10000 CFM", "24 TR", "5 kW", "18000 BTU"
_CAP_PAT = re.compile(
    r'(\d{1,5}(?:\.\d{1,2})?)\s*(TR|Ton(?:ne)?|CFM|BTU|kW)\b',
    re.IGNORECASE,
)

# Description: 4+ consecutive chars that aren't just digits/tag
_DESC_PAT = re.compile(r'[A-Za-z][A-Za-z\s/\-]{3,}')

# End-of-section guard: blank line or start of next major heading
_SECTION_BREAK = re.compile(
    r'(?:^|\n)\s*(?:SECTION|DRAWING|SHEET\s+NO|NOTE|GENERAL\s+NOTES|'
    r'CIVIL\s+WORKS|STRUCTURAL\s+WORKS|ARCHITECTURAL)\b',
    re.IGNORECASE,
)

# Known non-fixture words to exclude from description capture
_EXCLUDE_WORDS = frozenset([
    'schedule', 'fixture', 'description', 'type', 'reference', 'remarks',
    'quantity', 'total', 'no.', 'nos', 'sets', 'unit', 'make', 'model',
    'wattage', 'capacity', 'phase', 'voltage', 'notes', 'legend',
    'symbol', 'mark', 'tag', 'sr.', 'serial', 'item', 'category',
])


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MEPElement:
    """A single line from an MEP schedule."""
    trade: str          # "electrical" | "plumbing" | "hvac"
    subtype: str        # "fixture" | "socket" | "panel" | "sanitary" | "equipment"
    label: str          # e.g. "L1", "WC-1", "AHU-03"
    description: str    # e.g. "LED Panel 600×600 36W"
    qty: float          # number of units
    unit: str           # "Nos" | "Set"
    wattage_w: float    # 0.0 if not electrical
    capacity: str       # "1.5 TR", "10 kW", "" if N/A
    source_page: int


@dataclass
class MEPQTO:
    """Result of the MEP takeoff pass."""
    elements: List[MEPElement] = field(default_factory=list)
    line_items: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mode: str = "none"          # "schedule" | "assumption" | "none"
    electrical_total_nos: int = 0
    plumbing_total_nos: int = 0
    hvac_total_nos: int = 0


# =============================================================================
# SCHEDULE PARSER
# =============================================================================

def _detect_section(line: str) -> Optional[str]:
    """
    Return section type if this line is a schedule header, else None.
    Returns: "elec_fixture" | "elec_socket" | "elec_panel" | "plumbing" | "hvac"

    Order matters — more specific patterns are checked before general ones so that
    "SANITARY FIXTURE SCHEDULE" routes to plumbing (not elec_fixture) even though
    _ELEC_FIXTURE_HEADER has a bare "FIXTURE" alternative.
    """
    # HVAC and plumbing first — their keywords don't appear in electrical headers
    if _HVAC_HEADER.search(line):
        return "hvac"
    if _PLMB_HEADER.search(line):
        return "plumbing"
    if _ELEC_PANEL_HEADER.search(line):
        return "elec_panel"
    if _ELEC_SOCKET_HEADER.search(line):
        return "elec_socket"
    if _ELEC_FIXTURE_HEADER.search(line):
        return "elec_fixture"
    return None


def _parse_qty_from_row(row: str) -> float:
    """
    Extract quantity from a schedule row.

    Strategy:
    1. Look for a number explicitly followed by a unit keyword (45 Nos, 2 Sets).
    2. Otherwise strip wattage/capacity/dimension tokens, then scan remaining
       whitespace-separated tokens for a standalone pure integer.
       Take the LAST such integer (qty tends to appear after the description).
    Returns 0.0 if nothing found.
    """
    # 1. Explicit unit keyword — most reliable
    m = _QTY_WITH_UNIT_PAT.search(row)
    if m:
        v = float(m.group(1))
        if 1 <= v <= 2000:
            return v

    # 2. Strip noise: dimensions first (before watt strips part of them),
    #    then wattage/capacity
    clean = _STRIP_DIMENSION.sub(' ', row)
    clean = _STRIP_WATT_CAP.sub(' ', clean)

    # 3. Tokenise and collect standalone pure integers
    candidates: list[float] = []
    for tok in clean.split():
        stripped = tok.strip('.,;:')
        if _PURE_INT_TOKEN.match(stripped):
            v = float(stripped)
            if 1 <= v <= 2000:
                candidates.append(v)

    if not candidates:
        return 0.0

    # Take the last candidate (qty column appears after tag + description)
    return candidates[-1]


def _parse_wattage_from_row(row: str) -> float:
    """Extract wattage (W) from a row. kW is converted. Returns 0 if none."""
    total = 0.0
    for m in _WATT_PAT.finditer(row):
        val = float(m.group(1))
        unit = m.group(0).lower()
        if 'kw' in unit:
            total += val * 1000
        else:
            total += val
    return total


def _parse_capacity_from_row(row: str) -> str:
    """Extract capacity string (e.g. '1.5 TR') from a row."""
    m = _CAP_PAT.search(row)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    return ""


def _parse_description_from_row(row: str, tag: str) -> str:
    """Extract a human-readable description from a schedule row."""
    # Strip the tag itself
    cleaned = re.sub(re.escape(tag), '', row, count=1, flags=re.IGNORECASE)
    # Strip purely numeric tokens and units
    cleaned = re.sub(r'\b\d{1,5}(?:\.\d+)?\s*(?:W|kW|TR|CFM|Nos?|Sets?)\b', '', cleaned, flags=re.IGNORECASE)
    # Collect word groups
    words = []
    for m in _DESC_PAT.finditer(cleaned):
        w = m.group(0).strip()
        if w.lower() not in _EXCLUDE_WORDS and len(w) > 2:
            words.append(w)
    desc = ' '.join(words[:4]).strip()
    return desc if desc else row.strip()[:60]


def parse_mep_schedules_from_text(
    text: str,
    source_page: int = 0,
) -> List[MEPElement]:
    """
    State-machine parser over OCR text from one page.
    Returns a list of MEPElement (one per schedule row).
    """
    elements: List[MEPElement] = []
    lines = text.split('\n')

    current_section: Optional[str] = None
    blank_count = 0
    header_skip = 0  # skip the column-header row immediately after section header

    for raw_line in lines:
        line = raw_line.strip()

        # Detect new section
        new_section = _detect_section(line)
        if new_section:
            current_section = new_section
            blank_count = 0
            header_skip = 1  # skip next line (column headers row)
            continue

        # End section on 3+ consecutive blanks or a section-break keyword
        if current_section:
            if not line:
                blank_count += 1
                if blank_count >= 3:
                    current_section = None
                continue
            else:
                blank_count = 0

            if _SECTION_BREAK.search(line):
                current_section = None
                continue

            if header_skip > 0:
                # Skip only if this looks like a column-header row (no digits).
                # If the row already contains digits it's data — don't skip it.
                if not re.search(r'\d', line):
                    header_skip -= 1
                    continue
                else:
                    header_skip = 0  # row has data — fall through to parsing

            # --- Parse this row ---
            # Must have at least a quantity or a tag to be a real row
            tag_m = _TAG_PAT.search(line)
            qty = _parse_qty_from_row(line)

            if qty <= 0 and not tag_m:
                continue   # skip non-data lines

            tag = tag_m.group(0).upper() if tag_m else ""
            if not tag:
                # synthesize a tag from section type
                prefix = {
                    "elec_fixture": "LF",
                    "elec_socket": "SK",
                    "elec_panel": "DB",
                    "plumbing": "PF",
                    "hvac": "HV",
                }.get(current_section, "ME")
                tag = f"{prefix}{len(elements)+1:02d}"

            if qty <= 0:
                qty = 1.0  # at least 1

            description = _parse_description_from_row(line, tag)
            wattage = _parse_wattage_from_row(line)
            capacity = _parse_capacity_from_row(line)

            # Map section to trade/subtype
            trade_map = {
                "elec_fixture": ("electrical", "fixture"),
                "elec_socket":  ("electrical", "socket"),
                "elec_panel":   ("electrical", "panel"),
                "plumbing":     ("plumbing",   "sanitary"),
                "hvac":         ("hvac",       "equipment"),
            }
            trade, subtype = trade_map.get(current_section, ("electrical", "fixture"))

            elements.append(MEPElement(
                trade=trade,
                subtype=subtype,
                label=tag,
                description=description,
                qty=qty,
                unit="Nos",
                wattage_w=wattage,
                capacity=capacity,
                source_page=source_page,
            ))

    return elements


# =============================================================================
# DEDUPLICATION
# =============================================================================

def _dedup_elements(elements: List[MEPElement]) -> List[MEPElement]:
    """
    Merge duplicate (trade, subtype, label) rows across pages.
    Takes the row with the higher qty (usually the schedule row, not a summary).
    """
    seen: Dict[Tuple, MEPElement] = {}
    for el in elements:
        key = (el.trade, el.subtype, el.label.upper())
        if key not in seen or el.qty > seen[key].qty:
            seen[key] = el
    return list(seen.values())


# =============================================================================
# BOQ ITEM GENERATOR
# =============================================================================

def _item(description: str, qty: float, unit: str, trade: str,
          spec: str = "", source: str = "mep_schedule") -> dict:
    return {
        "description": description,
        "qty": round(qty, 2),
        "unit": unit,
        "trade": trade,
        "spec": spec,
        "source": source,
    }


def generate_mep_items(
    elements: List[MEPElement],
    floors: int = 1,
    total_area_sqm: float = 0.0,
) -> Tuple[List[dict], List[str]]:
    """
    Convert parsed MEP elements into BOQ line items.

    Returns:
        (line_items, warnings)
    """
    items: List[dict] = []
    warnings: List[str] = []

    elec_fixtures = [e for e in elements if e.trade == "electrical" and e.subtype == "fixture"]
    elec_sockets  = [e for e in elements if e.trade == "electrical" and e.subtype == "socket"]
    elec_panels   = [e for e in elements if e.trade == "electrical" and e.subtype == "panel"]
    plumbing      = [e for e in elements if e.trade == "plumbing"]
    hvac          = [e for e in elements if e.trade == "hvac"]

    # ── ELECTRICAL: Fixtures ─────────────────────────────────────────────────
    if elec_fixtures:
        total_fixture_nos = 0
        total_wattage = 0.0
        for el in elec_fixtures:
            desc = el.description or f"Light Fixture {el.label}"
            items.append(_item(
                f"Supply, install and connect {desc} ({el.label})",
                el.qty,
                "Nos",
                "Electrical",
                spec=f"IS 3646 — {el.wattage_w:.0f}W" if el.wattage_w else "IS 3646",
            ))
            total_fixture_nos += int(el.qty)
            total_wattage += el.wattage_w * el.qty

        # Wiring + conduit allowance
        wiring_m = total_fixture_nos * _WIRING_PER_FIXTURE_M
        conduit_m = total_fixture_nos * _CONDUIT_PER_FIXTURE_M
        items.append(_item(
            "PVC insulated copper wiring 2.5 sqmm (lighting circuits)",
            wiring_m,
            "m",
            "Electrical",
            spec="IS 694",
        ))
        items.append(_item(
            "25mm dia PVC conduit (concealed) for lighting",
            conduit_m,
            "m",
            "Electrical",
            spec="IS 9537",
        ))
        if total_wattage > 0:
            items.append(_item(
                f"Total connected load — lighting ({total_wattage/1000:.1f} kW connected)",
                1,
                "LS",
                "Electrical",
                spec="reference only",
            ))

    # ── ELECTRICAL: Sockets / Power Outlets ─────────────────────────────────
    if elec_sockets:
        total_socket_nos = sum(int(e.qty) for e in elec_sockets)
        for el in elec_sockets:
            items.append(_item(
                f"Supply, install and connect {el.description or 'Power socket/outlet'} ({el.label})",
                el.qty,
                "Nos",
                "Electrical",
                spec="IS 1293",
            ))
        wiring_m = total_socket_nos * _WIRING_PER_SOCKET_M
        items.append(_item(
            "PVC insulated copper wiring 4 sqmm (power circuits)",
            wiring_m,
            "m",
            "Electrical",
            spec="IS 694",
        ))

    # ── ELECTRICAL: DB / Panels ──────────────────────────────────────────────
    if elec_panels:
        for el in elec_panels:
            desc = el.description or f"Distribution Board {el.label}"
            items.append(_item(
                f"Supply & install {desc} complete with MCBs/MCCBs ({el.label})",
                el.qty,
                "Set",
                "Electrical",
                spec="IS 8623",
            ))
        # Earthing
        items.append(_item(
            "Earthing system — pipe earth electrode with GI conductor (complete)",
            len(elec_panels) * 2,
            "Set",
            "Electrical",
            spec="IS 3043",
        ))

    # If any electrical items were generated, add earthing if not already there
    if (elec_fixtures or elec_sockets) and not elec_panels:
        warnings.append("No DB/Panel schedule found — earthing allowance added as LS")
        items.append(_item(
            "Earthing & bonding complete installation (allowance)",
            1,
            "LS",
            "Electrical",
        ))

    # ── PLUMBING: Sanitary Fixtures ──────────────────────────────────────────
    if plumbing:
        total_plmb_nos = 0
        for el in plumbing:
            desc = el.description or f"Sanitary fixture {el.label}"
            items.append(_item(
                f"Supply & fix {desc} complete with CP fittings ({el.label})",
                el.qty,
                "Nos",
                "Plumbing",
                spec="IS 2556 / IS 7231",
            ))
            total_plmb_nos += int(el.qty)

        # Internal water supply piping allowance
        pipe_m = total_plmb_nos * _PIPE_PER_FIXTURE_M
        items.append(_item(
            "CPVC pipe class 5 (internal water supply) including fittings",
            pipe_m,
            "m",
            "Plumbing",
            spec="IS 15778",
        ))
        # Internal drainage
        drain_m = total_plmb_nos * _DRAIN_PER_FIXTURE_M
        items.append(_item(
            "uPVC drainage pipe 110mm dia (internal sewerage) including fittings",
            drain_m,
            "m",
            "Plumbing",
            spec="IS 4985",
        ))
        # Sump / overhead tank (LS per building)
        items.append(_item(
            "Underground sump pump set with controls (allowance)",
            1,
            "Set",
            "Plumbing",
        ))

    # ── HVAC: Equipment ──────────────────────────────────────────────────────
    if hvac:
        total_hvac_nos = 0
        total_tr = 0.0
        for el in hvac:
            desc = el.description or f"HVAC Equipment {el.label}"
            cap_str = f" — {el.capacity}" if el.capacity else ""
            items.append(_item(
                f"Supply, install and commission {desc}{cap_str} ({el.label})",
                el.qty,
                "Nos",
                "HVAC",
                spec="ASHRAE / NBC 2016 Ch.8",
            ))
            total_hvac_nos += int(el.qty)
            # Try to extract TR from capacity
            m = re.search(r'(\d+(?:\.\d+)?)\s*TR', el.capacity, re.IGNORECASE)
            if m:
                total_tr += float(m.group(1)) * el.qty

        # Refrigerant piping allowance
        refrig_m = total_hvac_nos * _REFRIGERANT_PER_UNIT_M
        items.append(_item(
            "Copper refrigerant piping insulated (suction & liquid lines)",
            refrig_m,
            "m",
            "HVAC",
            spec="ASTM B280",
        ))
        # Electrical connection for AC units
        items.append(_item(
            "Electrical connection and cabling for HVAC units (allowance)",
            total_hvac_nos,
            "Set",
            "HVAC",
        ))
        # Commissioning
        items.append(_item(
            "Testing, balancing and commissioning — HVAC system",
            1,
            "LS",
            "HVAC",
        ))

    return items, warnings


# =============================================================================
# ASSUMPTION MODE (no schedule found)
# =============================================================================

def _assumption_mep_items(
    floors: int,
    total_area_sqm: float,
    building_type: str = "residential",
) -> Tuple[List[dict], List[str]]:
    """
    Generate MEP BOQ based on floor area assumptions when no schedule is found.
    Indian IS + NBC 2016 thumb rules.
    """
    items: List[dict] = []
    warnings: List[str] = []

    if total_area_sqm <= 0:
        warnings.append("MEP assumption mode: no area given — using 1000 sqm default")
        total_area_sqm = 1000.0

    (sqm_per_light, sqm_per_socket, sqm_per_db, sqm_per_person,
     tr_per_sqm, hvac_threshold_sqm, exhaust_sqm, wc_area_sqm) = _resolve_mep_defaults(building_type)

    is_residential = "resident" in building_type.lower() or "housing" in building_type.lower()
    is_commercial  = "commerc" in building_type.lower() or "office" in building_type.lower()

    # ── Electrical assumption ────────────────────────────────────────────────
    light_nos = max(1, round(total_area_sqm / sqm_per_light))
    socket_nos = max(1, round(total_area_sqm / sqm_per_socket))
    db_nos = max(1, math.ceil(total_area_sqm / sqm_per_db))

    items += [
        _item(f"LED light fittings complete with wiring (estimated {light_nos} Nos)",
              light_nos, "Nos", "Electrical", source="mep_assumption"),
        _item("PVC insulated copper wiring 2.5 sqmm (lighting circuits)",
              light_nos * _WIRING_PER_FIXTURE_M, "m", "Electrical", source="mep_assumption"),
        _item(f"Power sockets 5/15A complete with wiring (estimated {socket_nos} Nos)",
              socket_nos, "Nos", "Electrical", source="mep_assumption"),
        _item("PVC insulated copper wiring 4 sqmm (power circuits)",
              socket_nos * _WIRING_PER_SOCKET_M, "m", "Electrical", source="mep_assumption"),
        _item("PVC conduit 25mm dia concealed",
              (light_nos + socket_nos) * 5, "m", "Electrical", source="mep_assumption"),
        _item("Distribution board complete with MCBs/MCCBs",
              db_nos, "Set", "Electrical", source="mep_assumption"),
        _item("Main LT panel with MCCB and metering",
              1, "Set", "Electrical", source="mep_assumption"),
        _item("Earthing complete (GI pipe electrodes)",
              db_nos * 2, "Set", "Electrical", source="mep_assumption"),
    ]

    # ── Plumbing assumption ──────────────────────────────────────────────────
    if is_residential:
        # 1 bathroom + 0.5 toilet per floor typical
        toilets_per_floor = max(2, round(total_area_sqm / floors / wc_area_sqm))
        wc_nos  = toilets_per_floor * floors
        wb_nos  = toilets_per_floor * floors
        shower_nos = max(1, math.ceil(toilets_per_floor * floors * 0.5))
    else:
        # Commercial: NBC 2016 Table 2 — 1 WC per 25 females + 1 per 50 males
        occupants = round(total_area_sqm / sqm_per_person)
        wc_nos  = max(2, round(occupants / 35))
        wb_nos  = wc_nos
        shower_nos = 0

    total_plmb = wc_nos + wb_nos + shower_nos
    items += [
        _item("Water closet EWC white vitreous china c/w cistern",
              wc_nos, "Nos", "Plumbing", spec="IS 2556", source="mep_assumption"),
        _item("Wash basin white vitreous china with CP fittings",
              wb_nos, "Nos", "Plumbing", spec="IS 2556", source="mep_assumption"),
    ]
    if shower_nos:
        items.append(_item("Shower unit complete",
                           shower_nos, "Nos", "Plumbing", source="mep_assumption"))
    items += [
        _item("CPVC pipe cl.5 internal water supply inc. fittings",
              total_plmb * _PIPE_PER_FIXTURE_M, "m", "Plumbing",
              spec="IS 15778", source="mep_assumption"),
        _item("uPVC drainage pipe 110mm dia internal sewerage inc. fittings",
              total_plmb * _DRAIN_PER_FIXTURE_M, "m", "Plumbing",
              spec="IS 4985", source="mep_assumption"),
        _item("Underground sump with pump set (allowance)",
              1, "Set", "Plumbing", source="mep_assumption"),
        _item("Overhead water storage tank (allowance)",
              1, "Set", "Plumbing", source="mep_assumption"),
    ]

    # ── HVAC assumption — only if commercial or large area ───────────────────
    if is_commercial or total_area_sqm > hvac_threshold_sqm:
        # Rule of thumb: 1 TR per tr_per_sqm sqm
        total_tr = round(total_area_sqm / tr_per_sqm)
        split_units = max(2, round(total_tr / 1.5))
        items += [
            _item(f"Split / cassette AC unit 1.5 TR (estimated {split_units} Nos)",
                  split_units, "Nos", "HVAC",
                  spec="IS 1391 / NBC 2016", source="mep_assumption"),
            _item("Copper refrigerant piping insulated",
                  split_units * _REFRIGERANT_PER_UNIT_M, "m", "HVAC", source="mep_assumption"),
            _item("Testing balancing and commissioning — HVAC",
                  1, "LS", "HVAC", source="mep_assumption"),
        ]
    items += [
        _item("Exhaust fans (toilets, kitchen, common areas)",
              max(2, round(total_area_sqm / exhaust_sqm)), "Nos", "HVAC", source="mep_assumption"),
    ]

    warnings.append(
        f"MEP ASSUMPTION MODE: no schedule found. Items estimated from {total_area_sqm:.0f} sqm "
        f"({building_type}). Verify against drawings — quantities may be ±40%."
    )
    return items, warnings


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

_MEP_DOC_TYPES = frozenset((
    "electrical", "mechanical", "plumbing", "hvac", "mep",
    "services", "drawing", "plan", "spec", "specification",
    "service_drawing", "services_drawing",
))


def run_mep_takeoff(
    page_texts: List[Tuple[int, str, str]],
    floors: int = 1,
    total_area_sqm: float = 0.0,
    building_type: str = "residential",
) -> MEPQTO:
    """
    Main MEP takeoff runner.

    Args:
        page_texts:       [(page_idx, ocr_text, doc_type), ...]
        floors:           number of storeys (for assumption fallback)
        total_area_sqm:   gross floor area (for assumption fallback)
        building_type:    "residential" | "commercial" | "industrial"

    Returns:
        MEPQTO with elements, line_items, warnings, mode
    """
    result = MEPQTO()

    # Filter to relevant pages
    relevant_pages = [
        (idx, text, dt)
        for idx, text, dt in page_texts
        if dt.lower() in _MEP_DOC_TYPES or any(
            kw in text[:500].upper()
            for kw in ("FIXTURE SCHEDULE", "PANEL SCHEDULE", "LIGHTING SCHEDULE",
                       "SANITARY", "PLUMBING FIXTURE", "HVAC", "AHU", "FCU",
                       "DISTRIBUTION BOARD", "DB SCHEDULE",
                       "SOCKET SCHEDULE", "ELECTRICAL SCHEDULE")
        )
    ]

    if not relevant_pages:
        # Broaden: scan all pages
        relevant_pages = [(idx, text, dt) for idx, text, dt in page_texts]

    # Parse schedules from all relevant pages
    all_elements: List[MEPElement] = []
    for page_idx, text, dt in relevant_pages:
        try:
            page_elements = parse_mep_schedules_from_text(text, source_page=page_idx)
            all_elements.extend(page_elements)
        except Exception as exc:
            logger.debug("MEP parse error page %d: %s", page_idx, exc)

    # Deduplicate
    all_elements = _dedup_elements(all_elements)

    if all_elements:
        result.mode = "schedule"
        result.elements = all_elements

        items, warnings = generate_mep_items(
            all_elements, floors=floors, total_area_sqm=total_area_sqm
        )
        result.line_items = items
        result.warnings = warnings
        result.electrical_total_nos = sum(
            int(e.qty) for e in all_elements if e.trade == "electrical"
        )
        result.plumbing_total_nos = sum(
            int(e.qty) for e in all_elements if e.trade == "plumbing"
        )
        result.hvac_total_nos = sum(
            int(e.qty) for e in all_elements if e.trade == "hvac"
        )
        logger.info(
            "MEP schedule mode: %d elements → %d BOQ items "
            "(E:%d P:%d H:%d)",
            len(all_elements), len(items),
            result.electrical_total_nos,
            result.plumbing_total_nos,
            result.hvac_total_nos,
        )
    else:
        result.mode = "assumption"
        items, warnings = _assumption_mep_items(
            floors=floors,
            total_area_sqm=total_area_sqm,
            building_type=building_type,
        )
        result.line_items = items
        result.warnings = warnings
        logger.info(
            "MEP assumption mode: %d items for %.0f sqm",
            len(items), total_area_sqm,
        )

    return result
