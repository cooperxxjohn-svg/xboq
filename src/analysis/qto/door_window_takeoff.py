"""
Door & Window Takeoff — Schedule-based QTO from OCR text.

Parses door, window, and ventilator schedules from PDF drawing text and
generates priceable BOQ line items compatible with the pipeline's line_items
format.

Supports Indian construction schedule notation:
  Door schedules      : DOOR SCHEDULE / DOOR LIST
  Window schedules    : WINDOW SCHEDULE / WINDOW LIST
  Ventilator schedules: VENTILATOR SCHEDULE

Algorithm:
1. Scan pages with relevant doc_types for D/W/V schedule tables in OCR text.
2. State-machine parser: detect section headers → extract rows.
3. Classify each element: material, is_main, is_toilet, area.
4. Generate BOQ items: frame, shutter/glazing, ironmongery, grille, sill, etc.
5. Fallback assumption mode when no schedule is found.

Design constraints:
- NO cv2 / OpenCV — pure text regex + arithmetic.
- Generates items in the same dict format as structural_takeoff.py / mep_takeoff.py.
- IS code specs referenced: IS 4021 (timber doors), IS 4962 (steel doors),
  IS 1038 (steel windows), IS 7452 (aluminium windows), IS 2835 (glass).
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION HEADER PATTERNS
# =============================================================================

_DOOR_HEADER = re.compile(
    r'DOOR\s+SCHEDULE|DOOR\s+LIST',
    re.IGNORECASE,
)
_WINDOW_HEADER = re.compile(
    r'WINDOW\s+SCHEDULE|WINDOW\s+LIST',
    re.IGNORECASE,
)
_VENTILATOR_HEADER = re.compile(
    r'VENTILATOR\s+SCHEDULE',
    re.IGNORECASE,
)

# End-of-section guard
_SECTION_BREAK = re.compile(
    r'(?:^|\n)\s*(?:SECTION|DRAWING|SHEET\s+NO|GENERAL\s+NOTES?|'
    r'CIVIL\s+WORKS|STRUCTURAL|ARCHITECTURAL|MEP|PLUMBING|'
    r'ELECTRICAL|HVAC)\b',
    re.IGNORECASE,
)

# Column header row detector (no digits → likely a header row)
_HAS_DIGIT = re.compile(r'\d')


# =============================================================================
# ROW-LEVEL EXTRACTION PATTERNS
# =============================================================================

# D1, W2, V1, DS1, WF3 — door/window tag
_DW_TAG_PAT = re.compile(
    r'(?<!\w)([DVW][A-Z]?\d{1,3}[A-Z]?)(?!\w)',
    re.IGNORECASE,
)

# Size patterns: "900×2100", "900 x 2100", "900X2100", "0.9×2.1"
# Unicode multiplication sign U+00D7 (×) and ASCII x/X
_SIZE_PAT = re.compile(
    r'(\d+(?:\.\d+)?)\s*[xX\u00d7]\s*(\d+(?:\.\d+)?)',
)

# Qty with explicit unit: "4 Nos", "2 No.", "6 Sets"
_QTY_WITH_UNIT = re.compile(
    r'(?<![A-Za-z\d\-])(\d{1,4}(?:\.\d{1,2})?)\s*(?:Nos?\.?|Sets?\.?|No\.?)\b',
    re.IGNORECASE,
)

# Strip dimension tokens before scanning for standalone qty integer
_STRIP_SIZE = re.compile(r'\d+(?:\.\d+)?\s*[xX\u00d7]\s*\d+(?:\.\d+)?')

# Strip mm/cm/m unit suffixes attached to numbers
_STRIP_UNITS = re.compile(
    r'\d+(?:\.\d+)?\s*(?:mm|cm|sqm|sqft|m²|m2|lm|kg|kN)\b',
    re.IGNORECASE,
)

# Pure integer token (1–4 digits)
_PURE_INT = re.compile(r'^\d{1,4}$')

# Material hints
_MAT_ROLLING = re.compile(r'rolling\s*shutter|shutter', re.IGNORECASE)
_MAT_HW = re.compile(r'\b(?:hw|hardwood|hard\s*wood|flush|panel)\b', re.IGNORECASE)
_MAT_STEEL = re.compile(r'\b(?:ms|steel|gi|galvanised|galvanized)\b', re.IGNORECASE)
_MAT_UPVC = re.compile(r'\bupvc\b', re.IGNORECASE)
_MAT_ALUM = re.compile(r'\b(?:alum(?:inium)?|aluminum)\b', re.IGNORECASE)

# Remark hints for door classification
_HINT_MAIN = re.compile(
    r'\b(?:main|entrance|entry|front|principal)\b', re.IGNORECASE
)
_HINT_TOILET = re.compile(
    r'\b(?:toilet|bath(?:room)?|wc|sanitary|lavatory)\b', re.IGNORECASE
)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class DWElement:
    """A single parsed entry from a door/window/ventilator schedule."""
    element_type: str    # "door" | "window" | "ventilator"
    tag: str             # "D1", "W2", "V1"
    count: int
    width_m: float
    height_m: float
    material: str        # "HW Flush", "UPVC", "Aluminium", "Rolling Shutter", ...
    area_sqm: float      # width_m × height_m
    is_main: bool        # True if tag is D1 or remark mentions main/entrance
    is_toilet: bool      # True if remark mentions toilet/bath/WC
    source_page: int


@dataclass
class DWResult:
    """Complete result of the door/window takeoff pass."""
    elements: List[DWElement] = field(default_factory=list)
    line_items: List[dict] = field(default_factory=list)
    mode: str = "none"           # "schedule" | "assumption" | "none"
    warnings: List[str] = field(default_factory=list)
    door_count: int = 0
    window_count: int = 0


# =============================================================================
# CORE PARSING HELPERS
# =============================================================================

def _parse_size(text: str) -> Tuple[float, float]:
    """
    Extract (width_m, height_m) from a text fragment.

    Handles:
      - "900×2100"   → (0.9, 2.1)   [mm assumed when > 100]
      - "900 x 2100" → (0.9, 2.1)
      - "0.9×2.1"    → (0.9, 2.1)   [already metres]
      - "300×300"    → (0.3, 0.3)

    Returns (0.0, 0.0) if no size found.
    """
    m = _SIZE_PAT.search(text)
    if not m:
        return (0.0, 0.0)
    w = float(m.group(1))
    h = float(m.group(2))
    # If both values > 100, treat as mm
    if w > 100:
        w = w / 1000.0
    if h > 100:
        h = h / 1000.0
    return (round(w, 4), round(h, 4))


def _parse_qty(row: str) -> int:
    """
    Extract quantity count from a schedule row.

    Strategy:
    1. Look for an explicit Nos/No. unit keyword ("4 Nos", "2 No.").
    2. Strip size tokens and unit suffixes, then scan for the last
       standalone pure integer in the plausible range [1, 500].

    Returns 0 if nothing found.
    """
    # 1. Explicit unit keyword — most reliable
    m = _QTY_WITH_UNIT.search(row)
    if m:
        v = int(float(m.group(1)))
        if 1 <= v <= 500:
            return v

    # 2. Strip noise then find last standalone integer
    clean = _STRIP_SIZE.sub(' ', row)
    clean = _STRIP_UNITS.sub(' ', clean)

    candidates: List[int] = []
    for tok in clean.split():
        stripped = tok.strip('.,;:()')
        if _PURE_INT.match(stripped):
            v = int(stripped)
            if 1 <= v <= 500:
                candidates.append(v)

    if not candidates:
        return 0

    return candidates[-1]


def _detect_material_door(row: str) -> str:
    """Return normalised door material string from a row."""
    if _MAT_ROLLING.search(row):
        return "Rolling Shutter"
    if _MAT_HW.search(row):
        return "HW Flush"
    if _MAT_STEEL.search(row):
        return "MS Steel"
    return "HW Flush"   # sensible default for Indian residential


def _detect_material_window(row: str) -> str:
    """Return normalised window material string from a row."""
    if _MAT_ALUM.search(row):
        return "Aluminium"
    if _MAT_UPVC.search(row):
        return "UPVC"
    if _MAT_STEEL.search(row):
        return "MS Steel"
    return "UPVC"        # sensible default for Indian residential


def _is_main_door(tag: str, row: str) -> bool:
    """True if this is a main/entrance door."""
    # Tag D1 is conventionally the main door in Indian drawings
    if re.match(r'^D1$', tag.strip(), re.IGNORECASE):
        return True
    return bool(_HINT_MAIN.search(row))


def _is_toilet_door(tag: str, row: str) -> bool:
    """True if this is a toilet/bathroom door."""
    return bool(_HINT_TOILET.search(row))


# =============================================================================
# SCHEDULE SECTION PARSER (state machine)
# =============================================================================

def _detect_section(line: str) -> Optional[str]:
    """
    Return section type if the line is a schedule header, else None.

    Check order: DOOR before WINDOW for specificity (avoids false WINDOW
    match on lines that happen to contain "DOOR AND WINDOW SCHEDULE").
    """
    # If line mentions both, prefer DOOR section (the table with D-tags comes first)
    if _DOOR_HEADER.search(line):
        return "door"
    if _WINDOW_HEADER.search(line):
        return "window"
    if _VENTILATOR_HEADER.search(line):
        return "ventilator"
    return None


def parse_dw_schedules_from_text(
    text: str,
    source_page: int = 0,
) -> List[DWElement]:
    """
    State-machine parser over OCR text from one page.

    Returns a list of DWElement (one per schedule data row).
    """
    elements: List[DWElement] = []
    lines = text.split('\n')

    current_section: Optional[str] = None
    blank_count = 0
    header_skip = 0   # skip the column-header row after section heading

    for raw_line in lines:
        line = raw_line.strip()

        # Check for new section header
        new_section = _detect_section(line)
        if new_section:
            current_section = new_section
            blank_count = 0
            header_skip = 1
            continue

        if current_section is None:
            continue

        # End section on 3+ blank lines or explicit section break keyword
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

        # Skip column-header row (contains no digits)
        if header_skip > 0:
            if not _HAS_DIGIT.search(line):
                header_skip -= 1
                continue
            else:
                header_skip = 0   # data row came early — fall through

        # ── Parse data row ─────────────────────────────────────────────────
        tag_m = _DW_TAG_PAT.search(line)
        size_w, size_h = _parse_size(line)
        qty = _parse_qty(line)

        # Require at least a tag or a size to be a real data row
        if not tag_m and (size_w == 0.0):
            continue

        tag = tag_m.group(0).upper() if tag_m else ""
        if not tag:
            # Synthesise a tag from section type
            prefix = {"door": "D", "window": "W", "ventilator": "V"}.get(
                current_section, "X"
            )
            tag = f"{prefix}{len(elements) + 1}"

        if qty <= 0:
            qty = 1   # assume at least 1 if not parseable

        if size_w <= 0.0:
            # Use typical defaults when size is not found in row
            if current_section == "door":
                size_w, size_h = 0.9, 2.1
            elif current_section == "window":
                size_w, size_h = 1.2, 1.2
            else:
                size_w, size_h = 0.3, 0.3

        area = round(size_w * size_h, 4)

        if current_section == "door":
            material = _detect_material_door(line)
            is_main = _is_main_door(tag, line)
            is_toilet = _is_toilet_door(tag, line)
        elif current_section == "window":
            material = _detect_material_window(line)
            is_main = False
            is_toilet = False
        else:
            # ventilator
            material = "Brick / Block"
            is_main = False
            is_toilet = False

        elements.append(DWElement(
            element_type=current_section,
            tag=tag,
            count=qty,
            width_m=size_w,
            height_m=size_h,
            material=material,
            area_sqm=area,
            is_main=is_main,
            is_toilet=is_toilet,
            source_page=source_page,
        ))

    return elements


# =============================================================================
# DEDUPLICATION (same tag across multiple pages)
# =============================================================================

def _dedup_elements(elements: List[DWElement]) -> List[DWElement]:
    """
    Merge duplicate (element_type, tag) rows across pages.
    Keep the row with the higher count (schedule page beats summary block).
    """
    seen: Dict[Tuple[str, str], DWElement] = {}
    for el in elements:
        key = (el.element_type, el.tag.upper())
        if key not in seen or el.count > seen[key].count:
            seen[key] = el
    return list(seen.values())


# =============================================================================
# BOQ ITEM GENERATOR
# =============================================================================

def _item(
    description: str,
    qty: float,
    unit: str,
    spec: str = "",
    source: str = "dw_schedule",
) -> dict:
    """Build a single BOQ line item dict."""
    return {
        "description": description,
        "qty": round(qty, 3),
        "unit": unit,
        "trade": "Doors & Windows",
        "spec": spec,
        "source": source,
    }


def _generate_door_items(el: DWElement) -> List[dict]:
    """
    Generate BOQ line items for a single door schedule entry.

    Items produced:
    - Door frame (hardwood/MS): qty = count (No)
    - Door shutter (flush): qty = count × area (sqm)
    - Door ironmongery set: qty = count (No)
    - Extra ironmongery for main doors (tower bolts both sides + peephole)
    - Extra ironmongery for toilet doors (sliding tower bolt)
    - Rolling shutter: area sqm + gearbox (No) + guide channels (lm)
    """
    items: List[dict] = []
    n = el.count
    area_each = el.area_sqm
    total_area = round(n * area_each, 3)
    w, h = el.width_m, el.height_m

    is_rolling = el.material == "Rolling Shutter"
    frame_spec = "IS 4021" if "HW" in el.material or "Flush" in el.material else "IS 4962"

    if is_rolling:
        # MS Rolling shutter
        items.append(_item(
            f"MS rolling shutter {int(w * 1000)}×{int(h * 1000)}mm "
            f"(Tag {el.tag}) — supply, install and commission",
            total_area,
            "sqm",
            spec="IS 6248",
        ))
        items.append(_item(
            f"Rolling shutter gearbox / motorised operator (Tag {el.tag})",
            float(n),
            "No",
            spec="IS 6248",
        ))
        # Guide channels: 2 sides × height per shutter
        channel_lm = round(n * h * 2, 3)
        items.append(_item(
            f"MS guide channel for rolling shutter, height {h:.2f}m (Tag {el.tag})",
            channel_lm,
            "lm",
            spec="IS 6248",
        ))
        return items

    # ── Standard door ──────────────────────────────────────────────────────
    mat_label = el.material  # "HW Flush", "MS Steel", ...

    # 1. Door frame
    items.append(_item(
        f"{mat_label} door frame {int(w * 1000)}×{int(h * 1000)}mm "
        f"(Tag {el.tag}) — supply and fix",
        float(n),
        "No",
        spec=frame_spec,
    ))

    # 2. Door shutter (area-based)
    items.append(_item(
        f"{mat_label} flush panel door shutter {int(w * 1000)}×{int(h * 1000)}mm "
        f"(Tag {el.tag}) — supply and fix",
        total_area,
        "sqm",
        spec=frame_spec,
    ))

    # 3. Standard ironmongery set
    #    Hinges 3 Nos, mortise lock, tower bolt ×1, door stopper
    items.append(_item(
        f"Door ironmongery set — 3 No. SS hinges, mortise lock, "
        f"tower bolt, door stopper (Tag {el.tag})",
        float(n),
        "No",
        spec="IS 1341 / IS 208",
    ))

    # 4. Main door extras
    if el.is_main:
        items.append(_item(
            f"Additional ironmongery — tower bolts both sides + peephole "
            f"for main/entrance door (Tag {el.tag})",
            float(n),
            "No",
            spec="IS 1341",
        ))

    # 5. Toilet door extras
    if el.is_toilet:
        items.append(_item(
            f"Sliding tower bolt / latch for toilet/bathroom door (Tag {el.tag})",
            float(n),
            "No",
            spec="IS 1341",
        ))

    return items


def _generate_window_items(
    el: DWElement,
    is_ground_floor_residential: bool = True,
) -> List[dict]:
    """
    Generate BOQ line items for a single window schedule entry.

    Items produced:
    - Window frame (UPVC / aluminium / MS): qty = count (No)
    - Window glazing 5mm plain/tinted: qty = count × area (sqm)
    - MS grille (ground floor residential): qty = count × area (sqm)
    - Window sill (granite/kota): qty = count × width (lm)
    """
    items: List[dict] = []
    n = el.count
    area_each = el.area_sqm
    total_area = round(n * area_each, 3)
    w = el.width_m
    h = el.height_m   # noqa: F841  (retained for future use)

    # Material → spec map
    if el.material == "Aluminium":
        frame_spec = "IS 7452"
        mat_label = "Aluminium"
    elif el.material == "MS Steel":
        frame_spec = "IS 1038"
        mat_label = "MS Steel"
    else:
        frame_spec = "IS 1081 / UPVC standard"
        mat_label = "UPVC"

    # 1. Window frame
    items.append(_item(
        f"{mat_label} window frame {int(el.width_m * 1000)}×{int(el.height_m * 1000)}mm "
        f"(Tag {el.tag}) — supply and fix complete",
        float(n),
        "No",
        spec=frame_spec,
    ))

    # 2. Glazing (5mm float glass)
    items.append(_item(
        f"5mm plain float glass glazing for window (Tag {el.tag}) — "
        f"supply and fix with putty/beading",
        total_area,
        "sqm",
        spec="IS 2835",
    ))

    # 3. MS grille — ground floor residential only
    if is_ground_floor_residential:
        items.append(_item(
            f"MS window grille fabricated and fixed (Tag {el.tag})",
            total_area,
            "sqm",
            spec="IS 2062",
        ))

    # 4. Window sill — count × width lm
    sill_lm = round(n * w, 3)
    items.append(_item(
        f"Granite / Kota stone window sill 200mm wide (Tag {el.tag})",
        sill_lm,
        "lm",
        spec="IS 1130",
    ))

    return items


def _generate_ventilator_items(el: DWElement) -> List[dict]:
    """Generate BOQ line items for a single ventilator schedule entry."""
    items: List[dict] = []
    n = el.count
    total_area = round(n * el.area_sqm, 3)

    items.append(_item(
        f"Ventilator frame and louver/fixed glass "
        f"{int(el.width_m * 1000)}×{int(el.height_m * 1000)}mm "
        f"(Tag {el.tag}) — supply and fix",
        float(n),
        "No",
        spec="IS 1038",
    ))
    items.append(_item(
        f"4mm plain glass / wire mesh for ventilator (Tag {el.tag})",
        total_area,
        "sqm",
        spec="IS 2835",
    ))
    return items


def generate_dw_items(
    elements: List[DWElement],
    floors: int = 1,
) -> Tuple[List[dict], List[str]]:
    """
    Convert parsed DWElement list into BOQ line items.

    Ground-floor MS grille is added for residential windows by default.
    Returns (line_items, warnings).
    """
    items: List[dict] = []
    warnings: List[str] = []

    doors = [e for e in elements if e.element_type == "door"]
    windows = [e for e in elements if e.element_type == "window"]
    ventilators = [e for e in elements if e.element_type == "ventilator"]

    if not elements:
        warnings.append("No D/W/V elements found — nothing to generate.")
        return items, warnings

    # Doors
    for el in doors:
        try:
            items.extend(_generate_door_items(el))
        except Exception as exc:
            warnings.append(f"Door item generation failed for {el.tag}: {exc}")
            logger.debug("Door item error tag=%s: %s", el.tag, exc)

    # Windows — ground floor grilles for first floor only
    for el in windows:
        try:
            is_gf_res = floors <= 4   # for buildings ≤ G+3 assume residential grilles
            items.extend(_generate_window_items(el, is_ground_floor_residential=is_gf_res))
        except Exception as exc:
            warnings.append(f"Window item generation failed for {el.tag}: {exc}")
            logger.debug("Window item error tag=%s: %s", el.tag, exc)

    # Ventilators
    for el in ventilators:
        try:
            items.extend(_generate_ventilator_items(el))
        except Exception as exc:
            warnings.append(f"Ventilator item generation failed for {el.tag}: {exc}")
            logger.debug("Ventilator item error tag=%s: %s", el.tag, exc)

    return items, warnings


# =============================================================================
# ASSUMPTION FALLBACK (no schedule found)
# =============================================================================

def _assumption_dw_items(
    floors: int,
    total_area_sqm: float,
) -> Tuple[List[DWElement], List[dict], List[str]]:
    """
    Generate door/window elements and BOQ items based on floor area thumb rules
    when no schedule is present in the drawings.

    Indian residential rules:
      Doors  : 1 door per 25 sqm
               60% HW Flush 900×2100 (internal)
               30% HW Flush 750×2100 (toilet)
               10% HW Flush 1200×2100 (main/entrance)
      Windows: 1.5 windows per 25 sqm
               70% UPVC 1200×1200
               30% UPVC 600×1200
    """
    items: List[dict] = []
    warnings: List[str] = []
    elements: List[DWElement] = []

    if total_area_sqm <= 0:
        # No floor area detected — return empty tuple rather than guessing from
        # an arbitrary 150 sqm default which produces wildly wrong cost totals
        # for large government tenders. Caller will show "area not detected" warning.
        warnings.append(
            "DW takeoff skipped: no floor area detected in drawings. "
            "Upload floor plan drawings with scale notation to enable D&W quantity takeoff."
        )
        return [], [], warnings

    total_doors = max(1, round(total_area_sqm / 25.0))
    total_windows = max(1, round(total_area_sqm / 25.0 * 1.5))

    # Door distribution
    internal_n = max(1, round(total_doors * 0.60))
    toilet_n = max(1, round(total_doors * 0.30))
    main_n = max(1, total_doors - internal_n - toilet_n)

    door_types = [
        ("DA1", internal_n, 0.9,  2.1, "HW Flush", False, False),
        ("DA2", toilet_n,   0.75, 2.1, "HW Flush", False, True),
        ("DA3", main_n,     1.2,  2.1, "HW Flush", True,  False),
    ]
    for tag, count, w, h, mat, is_main, is_toilet in door_types:
        if count <= 0:
            continue
        el = DWElement(
            element_type="door",
            tag=tag,
            count=count,
            width_m=w,
            height_m=h,
            material=mat,
            area_sqm=round(w * h, 4),
            is_main=is_main,
            is_toilet=is_toilet,
            source_page=-1,
        )
        elements.append(el)
        items.extend(_generate_door_items(el))

    # Window distribution
    standard_n = max(1, round(total_windows * 0.70))
    small_n = max(0, total_windows - standard_n)

    window_types = [
        ("WA1", standard_n, 1.2, 1.2, "UPVC"),
        ("WA2", small_n,    0.6, 1.2, "UPVC"),
    ]
    for tag, count, w, h, mat in window_types:
        if count <= 0:
            continue
        el = DWElement(
            element_type="window",
            tag=tag,
            count=count,
            width_m=w,
            height_m=h,
            material=mat,
            area_sqm=round(w * h, 4),
            is_main=False,
            is_toilet=False,
            source_page=-1,
        )
        elements.append(el)
        items.extend(_generate_window_items(el, is_ground_floor_residential=True))

    warnings.append(
        f"DW ASSUMPTION MODE: no door/window schedule found. "
        f"Items estimated from {total_area_sqm:.0f} sqm — "
        f"{total_doors} doors + {total_windows} windows (thumb rule: 1 door/25 sqm, "
        f"1.5 windows/25 sqm). Verify against drawings — quantities may be ±40%."
    )

    # Tag all items as assumption source
    for it in items:
        it["source"] = "dw_assumption"

    return elements, items, warnings


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

_DW_DOC_TYPES = frozenset((
    "drawing", "plan", "floor_plan", "layout", "elevation", "section",
    "detail", "schedule", "spec", "specification", "architectural",
    "door_schedule", "window_schedule",
))

_DW_KEYWORDS = (
    "DOOR SCHEDULE", "DOOR LIST",
    "WINDOW SCHEDULE", "WINDOW LIST",
    "VENTILATOR SCHEDULE",
)


def run_dw_takeoff(
    page_texts: List[Tuple[int, str, str]],  # (page_idx, text, doc_type)
    floors: int = 1,
    total_area_sqm: float = 0.0,
) -> DWResult:
    """
    Main door/window takeoff runner.

    Args:
        page_texts:     [(page_idx, ocr_text, doc_type), ...]
        floors:         number of storeys (used for grille decision + assumption scale)
        total_area_sqm: gross floor area (for assumption fallback)

    Returns:
        DWResult with elements, line_items, warnings, mode, door_count, window_count
    """
    result = DWResult()

    if not page_texts:
        result.warnings.append("DW takeoff: no pages provided — running assumption mode.")
        elements, items, warnings = _assumption_dw_items(
            floors=floors,
            total_area_sqm=total_area_sqm,
        )
        result.mode = "assumption"
        result.elements = elements
        result.line_items = items
        result.warnings.extend(warnings)
        result.door_count = sum(e.count for e in elements if e.element_type == "door")
        result.window_count = sum(e.count for e in elements if e.element_type == "window")
        return result

    # Filter to relevant pages: doc_type match or keyword present in first 500 chars
    relevant_pages = [
        (idx, text, dt)
        for idx, text, dt in page_texts
        if dt.lower() in _DW_DOC_TYPES
        or any(kw in text[:500].upper() for kw in _DW_KEYWORDS)
    ]

    if not relevant_pages:
        # Broaden: scan all pages (schedules sometimes appear on cover sheets)
        relevant_pages = list(page_texts)

    # Parse schedules from all relevant pages
    all_elements: List[DWElement] = []
    for page_idx, text, dt in relevant_pages:
        try:
            page_elements = parse_dw_schedules_from_text(text, source_page=page_idx)
            all_elements.extend(page_elements)
        except Exception as exc:
            logger.debug("DW parse error page %d: %s", page_idx, exc)
            result.warnings.append(f"Parse error on page {page_idx}: {exc}")

    # Deduplicate across pages
    all_elements = _dedup_elements(all_elements)

    if all_elements:
        result.mode = "schedule"
        result.elements = all_elements

        items, warnings = generate_dw_items(all_elements, floors=floors)
        result.line_items = items
        result.warnings.extend(warnings)

        result.door_count = sum(
            e.count for e in all_elements if e.element_type == "door"
        )
        result.window_count = sum(
            e.count for e in all_elements if e.element_type == "window"
        )

        logger.info(
            "DW schedule mode: %d elements → %d BOQ items "
            "(doors=%d windows=%d)",
            len(all_elements), len(items),
            result.door_count, result.window_count,
        )

    else:
        result.mode = "assumption"

        elements, items, warnings = _assumption_dw_items(
            floors=floors,
            total_area_sqm=total_area_sqm,
        )
        result.elements = elements
        result.line_items = items
        result.warnings.extend(warnings)

        result.door_count = sum(
            e.count for e in elements if e.element_type == "door"
        )
        result.window_count = sum(
            e.count for e in elements if e.element_type == "window"
        )

        logger.info(
            "DW assumption mode: %d items for %.0f sqm "
            "(doors=%d windows=%d)",
            len(items), total_area_sqm,
            result.door_count, result.window_count,
        )

    return result
