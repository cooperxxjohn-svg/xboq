"""
Structural Takeoff — Text-Only QTO from OCR Schedule Data.

Algorithm:
1. Scan all pages with doc_type in (structural, plan, drawing) for column/beam/
   footing/slab schedule tables in OCR text.
2. Parse element sizes, concrete grades, counts.
3. Compute concrete volumes + steel weights using IS-based kg/m³ factors.
4. Generate priceable BOQ line items (cum, kg, sqm) compatible with
   build_line_items() — same format as finish_takeoff.py output.

Design constraints:
- NO cv2 / OpenCV dependency — pure text-based.
- Works alongside the image-based pipeline_structural.py (which needs cv2).
- Intended for the common case where structural sheets have tabular schedules
  readable by PDF text extraction (vector PDF) or good OCR quality.
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Steel intensity factors (kg per m³ of concrete) — IS 456 rule-of-thumb
_STEEL_KG_PER_M3: Dict[str, float] = {
    "column": 200.0,   # 160–250 typical; 200 conservative
    "beam":   150.0,   # 120–180 typical
    "slab":    90.0,   # 70–110 typical
    "footing": 90.0,   # 60–120 typical
    "wall":   100.0,   # RCC retaining/shear walls
    "staircase": 120.0,
}

# Default concrete grade by element type (used when not specified in schedule)
_DEFAULT_GRADE: Dict[str, str] = {
    "column":    "M25",
    "beam":      "M25",
    "slab":      "M25",
    "footing":   "M20",
    "wall":      "M20",
    "staircase": "M20",
}

# Default steel grade (Fe500 is standard for HYSD bars, IS 1786)
_DEFAULT_STEEL_GRADE = "Fe500"

# Formwork factor — sqm of shuttering per m³ of concrete
_FORMWORK_FACTOR: Dict[str, float] = {
    "column":    8.0,   # 4 faces; perimeter/volume ratio depends on size
    "beam":      5.0,   # 3 exposed faces
    "slab":      1.0,   # bottom soffit only (top is open)
    "footing":   2.0,   # 4 side faces
    "wall":      2.0,   # both faces
}

# Default element sizes (mm) when not found in schedule
_DEFAULT_SIZES: Dict[str, Tuple] = {
    "column":    (230, 450, None),   # width, depth, length=height
    "beam":      (230, 450, 3000),   # width, depth, clear_span
    "slab":      (None, None, 125),  # None, None, thickness
    "footing":   (1500, 1500, 450),  # l, w, d
    "wall":      (200, None, 3000),  # thickness, None, height
}

# Typical storey heights (mm)
_STOREY_HEIGHT_DEFAULT = 3000


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Schedule section headers
_SECTION_HEADERS = {
    "column":    re.compile(
        r'\b(?:column|col\.?)\s+(?:schedule|sizing|details?|reinforcement)\b',
        re.IGNORECASE,
    ),
    "beam":      re.compile(
        r'\b(?:beam|bm\.?)\s+(?:schedule|sizing|details?|reinforcement)\b',
        re.IGNORECASE,
    ),
    "footing":   re.compile(
        r'\b(?:footing|ftg\.?|foundation)\s+(?:schedule|sizing|details?)\b',
        re.IGNORECASE,
    ),
    "slab":      re.compile(
        r'\b(?:slab|flat\s*slab)\s+(?:schedule|thicknesses?|details?)\b',
        re.IGNORECASE,
    ),
    "staircase": re.compile(
        r'\b(?:staircase|stair)\s+(?:schedule|details?|reinforcement)\b',
        re.IGNORECASE,
    ),
}

# Element label patterns per type
_LABEL_PATTERN: Dict[str, re.Pattern] = {
    "column":    re.compile(r'\b(C-?\d{1,3}[A-Z]?)\b'),
    "beam":      re.compile(r'\b(B-?\d{1,3}[A-Z]?|BM-?\d{1,3})\b'),
    "footing":   re.compile(r'\b([ICF]?F-?\d{1,3}[A-Z]?|IF-?\d{1,3})\b'),
    "slab":      re.compile(r'\b(S-?\d{1,3}[A-Z]?|SL-?\d{1,3})\b'),
    "staircase": re.compile(r'\b(ST-?\d{1,3}|STR-?\d{1,3})\b'),
}

# Size patterns: 230x450, 230 X 450, 230×450, 230mm x 450mm
_SIZE_2D = re.compile(r'(\d{2,4})\s*[xX×]\s*(\d{2,4})')
_SIZE_3D = re.compile(r'(\d{3,4})\s*[xX×]\s*(\d{3,4})\s*[xX×]\s*(\d{2,4})')

# Concrete grade: M25, M-25, M 25
_GRADE_RE = re.compile(r'\bM[-\s]?(\d{2,3})\b')

# Count: "8 Nos", "Nos. 4", "12 no.", "Qty: 4"
_COUNT_RE = re.compile(
    r'(?:nos?\.?\s*:?\s*(\d{1,3})|(\d{1,3})\s*nos?\.?|qty\s*:?\s*(\d{1,3}))',
    re.IGNORECASE,
)

# Bar schedule: "4Y16", "6T20", "8-Y12", "4Fe20", "4#16"
_BAR_RE = re.compile(
    r'(\d{1,2})\s*[-]?\s*[YTy#Fe]+\s*(\d{2,3})',
    re.IGNORECASE,
)

# Slab thickness: "125mm", "175 thk", "200mm thk"
_SLAB_THK = re.compile(r'(\d{2,3})\s*(?:mm\s*(?:thk|thick)?|thk|thick)', re.IGNORECASE)

# Floor count line: "G+4", "G + 4 Floors", "4 Storeyed", "B+G+4"
_FLOORS_RE = re.compile(
    r'(?:G\+|B\+G\+|basement\+)?(\d+)\s*(?:floors?|storeys?|storeyed)',
    re.IGNORECASE,
)

# Total area: "Built-up area: 450 sqm", "Total area 4500 sft"
_AREA_RE = re.compile(
    r'(?:built.?up|total|floor|plinth)?\s*area\s*[=:]\s*(\d+(?:\.\d+)?)\s*(sqm|sq\.m|sft|sqft)',
    re.IGNORECASE,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScheduledElement:
    """One row from a structural schedule (text-parsed)."""
    element_type: str           # column / beam / footing / slab / staircase
    label: str                  # e.g. C1, B2, F3
    width_mm: Optional[int]     # for columns/beams: width; for footings: L
    depth_mm: Optional[int]     # for columns/beams: depth; for footings: W
    length_mm: Optional[int]    # height for columns, span for beams, depth for footings
    concrete_grade: str         # M25 etc.
    steel_grade: str            # Fe500
    count: int                  # number of that element label
    source_page: int
    confidence: float           # 0.4–0.9


@dataclass
class StructuralQTO:
    """Output of structural text-based QTO."""
    elements: List[ScheduledElement]
    line_items: List[dict]      # BOQ-compatible item dicts
    mode: str                   # "schedule" | "assumption"
    floors: int
    total_area_sqm: float
    warnings: List[str]
    assumptions: List[str]
    scale_ratio: int = 0        # known drawing scale ratio (e.g. 100 for 1:100)
    px_per_mm: float = 0.0      # pixels per real mm from scale_detector
    area_source: str = ""       # "measured" | "metadata" | "assumption"


# =============================================================================
# SCHEDULE TEXT PARSER
# =============================================================================

def _extract_count(text: str) -> Optional[int]:
    m = _COUNT_RE.search(text)
    if m:
        val = m.group(1) or m.group(2) or m.group(3)
        if val:
            return int(val)
    return None


def _extract_size_2d(text: str) -> Optional[Tuple[int, int]]:
    m = _SIZE_2D.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _extract_size_3d(text: str) -> Optional[Tuple[int, int, int]]:
    m = _SIZE_3D.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _extract_grade(text: str) -> Optional[str]:
    m = _GRADE_RE.search(text)
    if m:
        return f"M{m.group(1)}"
    return None


def _extract_slab_thickness(text: str) -> Optional[int]:
    m = _SLAB_THK.search(text)
    if m:
        return int(m.group(1))
    return None


def _parse_schedule_section(
    lines: List[str],
    element_type: str,
    source_page: int,
) -> List[ScheduledElement]:
    """
    Parse a block of lines that belong to one schedule section type.
    Returns extracted ScheduledElement objects.
    """
    elements: List[ScheduledElement] = []
    label_pat = _LABEL_PATTERN[element_type]

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        m_label = label_pat.search(line)
        if not m_label:
            continue

        label = m_label.group(1).upper().replace("-", "")
        grade = _extract_grade(line) or _DEFAULT_GRADE.get(element_type, "M20")
        count = _extract_count(line) or 1
        confidence = 0.65

        w = d = l = None

        if element_type == "footing":
            s3 = _extract_size_3d(line)
            if s3:
                w, d, l = s3
                confidence = 0.80
            else:
                s2 = _extract_size_2d(line)
                if s2:
                    w, d = s2
                    l = 450  # assume depth
                    confidence = 0.65
        elif element_type == "slab":
            thk = _extract_slab_thickness(line)
            if thk:
                l = thk
                confidence = 0.75
        else:
            # column / beam / staircase
            s2 = _extract_size_2d(line)
            if s2:
                w, d = s2
                confidence = 0.75

        # Only keep if we got at least something meaningful
        if element_type == "slab" and l is None:
            continue
        if element_type != "slab" and w is None and d is None:
            continue

        elements.append(ScheduledElement(
            element_type=element_type,
            label=label,
            width_mm=w,
            depth_mm=d,
            length_mm=l,
            concrete_grade=grade,
            steel_grade=_DEFAULT_STEEL_GRADE,
            count=count,
            source_page=source_page,
            confidence=confidence,
        ))

    return elements


def parse_structural_schedules_from_text(
    text: str,
    source_page: int,
) -> List[ScheduledElement]:
    """
    Scan a page's OCR text for structural schedules and parse all elements found.
    """
    lines = text.split("\n")
    elements: List[ScheduledElement] = []

    # State machine: track which schedule section we're in
    current_section: Optional[str] = None
    section_lines: List[str] = []

    def flush_section():
        nonlocal section_lines, current_section
        if current_section and section_lines:
            parsed = _parse_schedule_section(section_lines, current_section, source_page)
            elements.extend(parsed)
        section_lines = []
        current_section = None

    for line in lines:
        # Check for a new schedule header
        for etype, pat in _SECTION_HEADERS.items():
            if pat.search(line):
                flush_section()
                current_section = etype
                break

        if current_section:
            section_lines.append(line)

    flush_section()

    # Fallback: even without a section header, scan every line for labelled entries
    if not elements:
        for etype, label_pat in _LABEL_PATTERN.items():
            for line in lines:
                m = label_pat.search(line)
                if m and (_extract_size_2d(line) or _extract_size_3d(line) or
                          (etype == "slab" and _extract_slab_thickness(line))):
                    label = m.group(1).upper().replace("-", "")
                    grade = _extract_grade(line) or _DEFAULT_GRADE.get(etype, "M20")
                    count = _extract_count(line) or 1
                    w = d = l = None
                    if etype == "footing":
                        s3 = _extract_size_3d(line)
                        if s3:
                            w, d, l = s3
                        else:
                            s2 = _extract_size_2d(line)
                            if s2:
                                w, d = s2; l = 450
                    elif etype == "slab":
                        l = _extract_slab_thickness(line)
                    else:
                        s2 = _extract_size_2d(line)
                        if s2:
                            w, d = s2
                    if w is not None or l is not None:
                        elements.append(ScheduledElement(
                            element_type=etype,
                            label=label,
                            width_mm=w, depth_mm=d, length_mm=l,
                            concrete_grade=grade,
                            steel_grade=_DEFAULT_STEEL_GRADE,
                            count=count,
                            source_page=source_page,
                            confidence=0.50,
                        ))

    return elements


# =============================================================================
# QUANTITY COMPUTATION
# =============================================================================

def _volume_m3(elem: ScheduledElement, storey_height_mm: int) -> float:
    """Compute concrete volume for one instance of an element (m³)."""
    etype = elem.element_type

    w = elem.width_mm or 0
    d = elem.depth_mm or 0

    if etype == "column":
        h = storey_height_mm
        return (w / 1000) * (d / 1000) * (h / 1000)

    elif etype == "beam":
        span = elem.length_mm or 3000
        return (w / 1000) * (d / 1000) * (span / 1000)

    elif etype == "footing":
        L = elem.width_mm or 1500
        W = elem.depth_mm or 1500
        D = elem.length_mm or 450
        return (L / 1000) * (W / 1000) * (D / 1000)

    elif etype in ("slab", "staircase"):
        # Slab: volume per m² × area. Here we compute per element if area known;
        # otherwise qty=thickness only and area comes from total_area at caller.
        thk = elem.length_mm or 125
        return thk / 1000   # m³ per sqm — caller multiplies by area

    elif etype == "wall":
        thk = elem.width_mm or 200
        h = elem.length_mm or storey_height_mm
        # Wall length unknown at element level; return per-m-run volume
        return (thk / 1000) * (h / 1000)  # m³ per metre run

    return 0.0


def _formwork_m2(elem: ScheduledElement, vol_m3: float, storey_height_mm: int) -> float:
    """Compute formwork area (sqm) for one element instance."""
    etype = elem.element_type
    fac = _FORMWORK_FACTOR.get(etype, 4.0)
    return round(vol_m3 * fac, 3)


# =============================================================================
# BOQ LINE ITEM GENERATION
# =============================================================================

def _concrete_description(etype: str, elem: ScheduledElement) -> str:
    """Build a clear concrete item description."""
    grade = elem.concrete_grade
    w, d = elem.width_mm, elem.depth_mm

    if etype == "column":
        size_str = f"{w}×{d}mm" if w and d else ""
        return (
            f"Providing, mixing, placing and compacting RCC {grade} for "
            f"columns {size_str} including all formwork, vibration and curing"
        )
    elif etype == "beam":
        size_str = f"{w}×{d}mm" if w and d else ""
        return (
            f"Providing, mixing, placing and compacting RCC {grade} for "
            f"beams {size_str} including formwork, vibration and curing"
        )
    elif etype == "footing":
        L, W, D = elem.width_mm, elem.depth_mm, elem.length_mm
        size_str = f"{L}×{W}×{D}mm" if L and W and D else ""
        return (
            f"Providing, mixing, placing and compacting RCC {grade} for "
            f"isolated footings {size_str} including formwork and curing"
        )
    elif etype == "slab":
        thk = elem.length_mm or 125
        return (
            f"Providing, mixing, placing and compacting RCC {grade} for "
            f"solid flat slab {thk}mm thick including bottom formwork, "
            f"vibration and curing"
        )
    elif etype == "staircase":
        return (
            f"Providing, mixing, placing and compacting RCC {grade} for "
            f"staircase including waist slab, steps, formwork and curing"
        )
    elif etype == "wall":
        thk = elem.width_mm or 200
        return (
            f"Providing, mixing, placing and compacting RCC {grade} for "
            f"{thk}mm thick RCC walls including formwork, vibration and curing"
        )
    return f"Providing RCC {grade} for {etype}"


def _steel_description(etype: str, elem: ScheduledElement) -> str:
    grade = elem.steel_grade
    return (
        f"Providing, cutting, bending, placing and binding {grade} "
        f"deformed bars for {etype}s including lapping and binding wire"
    )


def _formwork_description(etype: str) -> str:
    return (
        f"Centering, shuttering, propping and de-shuttering for "
        f"{etype}s using steel/timber formwork"
    )


def generate_structural_items(
    elements: List[ScheduledElement],
    floors: int,
    total_area_sqm: float,
    storey_height_mm: int = _STOREY_HEIGHT_DEFAULT,
    source: str = "qto_structural",
    include_formwork: bool = True,
) -> Tuple[List[dict], List[str]]:
    """
    Convert parsed ScheduledElement list + building parameters into BOQ line items.

    Returns:
        (items, warnings)
    """
    items: List[dict] = []
    warnings: List[str] = []

    # Group elements by type for grouping into single BOQ items where sensible
    by_type: Dict[str, List[ScheduledElement]] = {}
    for elem in elements:
        by_type.setdefault(elem.element_type, []).append(elem)

    for etype, elems in by_type.items():
        section_label = etype.upper() + "S"

        # ── Determine dominant grade for this element type ────────────────
        from collections import Counter
        grade_counter: Counter = Counter(e.concrete_grade for e in elems)
        dominant_grade = grade_counter.most_common(1)[0][0]
        steel_grade = elems[0].steel_grade

        # ── Concrete volume ──────────────────────────────────────────────
        total_vol_m3 = 0.0
        vol_source = "schedule"

        for elem in elems:
            one_vol = _volume_m3(elem, storey_height_mm)

            if etype in ("slab", "staircase") and total_area_sqm > 0:
                # one_vol is m³/m² — multiply by area per floor
                total_vol_m3 += one_vol * total_area_sqm * floors * elem.count
            elif etype == "footing":
                # footings don't repeat per floor
                total_vol_m3 += one_vol * elem.count
            elif etype == "wall":
                warnings.append(f"Wall '{elem.label}' — length unknown; qty left as m³/m-run")
                total_vol_m3 += one_vol * floors * elem.count
            else:
                # columns, beams: multiply by number of floors
                total_vol_m3 += one_vol * floors * elem.count

        total_vol_m3 = round(total_vol_m3, 3)
        if total_vol_m3 <= 0:
            warnings.append(f"Zero concrete volume for {etype} — skipped")
            continue

        # Confidence: average of elements, adjusted for assumption usage
        avg_conf = sum(e.confidence for e in elems) / len(elems)
        if etype in ("slab",) and total_area_sqm == 0:
            avg_conf *= 0.6  # area unknown
            warnings.append("Slab area not detected — slab qty may be inaccurate")

        # Representative size for description
        rep = elems[0]
        # Override grade with dominant
        rep_desc = ScheduledElement(**{**rep.__dict__, "concrete_grade": dominant_grade})

        # Concrete item
        items.append({
            "item_no":          None,
            "description":      _concrete_description(etype, rep_desc),
            "unit":             "cum",
            "unit_inferred":    False,
            "qty":              total_vol_m3,
            "rate":             None,
            "trade":            "structural" if etype in ("column", "beam", "slab", "staircase") else "civil",
            "section":         f"STRUCTURAL — {section_label}",
            "source_page":      elems[0].source_page,
            "source":           source,
            "confidence":       round(avg_conf, 2),
            "is_priceable":     True,
            "priceable_reason": "priceable",
            "qto_method":       "schedule_text",
            "concrete_grade":   dominant_grade,
        })

        # Steel (rebar) item — kg
        steel_factor = _STEEL_KG_PER_M3.get(etype, 150.0)
        steel_kg = round(total_vol_m3 * steel_factor, 1)

        items.append({
            "item_no":          None,
            "description":      _steel_description(etype, rep_desc),
            "unit":             "kg",
            "unit_inferred":    False,
            "qty":              steel_kg,
            "rate":             None,
            "trade":            "structural",
            "section":         f"STRUCTURAL — {section_label}",
            "source_page":      elems[0].source_page,
            "source":           source,
            "confidence":       round(avg_conf * 0.85, 2),  # slightly lower — est. factor
            "is_priceable":     True,
            "priceable_reason": "priceable",
            "qto_method":       "schedule_text",
            "steel_grade":      steel_grade,
        })

        # Formwork item — sqm (optional)
        if include_formwork and etype not in ("slab",):
            fw_factor = _FORMWORK_FACTOR.get(etype, 4.0)
            fw_sqm = round(total_vol_m3 * fw_factor, 1)
            items.append({
                "item_no":          None,
                "description":      _formwork_description(etype),
                "unit":             "sqm",
                "unit_inferred":    False,
                "qty":              fw_sqm,
                "rate":             None,
                "trade":            "structural",
                "section":         f"STRUCTURAL — {section_label}",
                "source_page":      elems[0].source_page,
                "source":           source,
                "confidence":       round(avg_conf * 0.8, 2),
                "is_priceable":     True,
                "priceable_reason": "priceable",
                "qto_method":       "schedule_text",
            })

    return items, warnings


# =============================================================================
# ASSUMPTION MODE — when no schedule found
# =============================================================================

def _assumption_structural_items(
    total_area_sqm: float,
    floors: int,
    building_type: str = "residential",
    column_grid_m: float = 4.5,
    storey_height_mm: int = _STOREY_HEIGHT_DEFAULT,
    source_page: int = 0,
    source: str = "qto_structural_assumption",
) -> Tuple[List[ScheduledElement], List[str]]:
    """
    Generate assumed structural elements when no schedule is available.
    Uses standard rule-of-thumb approach (column grid method).
    """
    if total_area_sqm <= 0:
        return [], ["No area data — assumption mode skipped"]

    warnings = ["Structural quantities computed using ASSUMPTION MODE (no schedule found). "
                "Accuracy ±30%. Verify with structural drawings."]
    elements: List[ScheduledElement] = []

    # Column count
    side_m = math.sqrt(total_area_sqm)
    cols_per_side = max(2, round(side_m / column_grid_m) + 1)
    n_columns = cols_per_side * cols_per_side

    col_w, col_d = (230, 450) if building_type == "residential" else (300, 600)
    elements.append(ScheduledElement(
        element_type="column",
        label="C_TYP",
        width_mm=col_w, depth_mm=col_d, length_mm=storey_height_mm,
        concrete_grade="M25", steel_grade="Fe500",
        count=n_columns,
        source_page=source_page,
        confidence=0.40,
    ))

    # Beam count: ~2 beams per column at grid spacing
    n_beams = n_columns * 2
    bm_span = int(column_grid_m * 900)   # ~90% of grid = clear span
    elements.append(ScheduledElement(
        element_type="beam",
        label="B_TYP",
        width_mm=230, depth_mm=450, length_mm=bm_span,
        concrete_grade="M25", steel_grade="Fe500",
        count=n_beams,
        source_page=source_page,
        confidence=0.35,
    ))

    # Slab
    elements.append(ScheduledElement(
        element_type="slab",
        label="S_TYP",
        width_mm=None, depth_mm=None, length_mm=125,
        concrete_grade="M25", steel_grade="Fe500",
        count=1,
        source_page=source_page,
        confidence=0.45,
    ))

    # Footing: one per column
    elements.append(ScheduledElement(
        element_type="footing",
        label="F_TYP",
        width_mm=1500, depth_mm=1500, length_mm=450,
        concrete_grade="M20", steel_grade="Fe500",
        count=n_columns,
        source_page=source_page,
        confidence=0.35,
    ))

    return elements, warnings


# =============================================================================
# FULL PIPELINE ENTRY POINT
# =============================================================================

def run_structural_takeoff(
    page_texts: List[Tuple[int, str, str]],   # (page_idx, ocr_text, doc_type)
    floors: int = 1,
    total_area_sqm: float = 0.0,
    building_type: str = "residential",
    storey_height_mm: int = _STOREY_HEIGHT_DEFAULT,
    include_formwork: bool = True,
    px_per_mm: float = 0.0,
    known_scale_ratio: int = 0,
    pdf_path: str = "",
    llm_client: Any = None,
) -> StructuralQTO:
    """
    Full pipeline: scan all structural/plan pages, extract schedules, compute
    quantities, generate BOQ line items.

    Args:
        page_texts:         list of (page_idx, ocr_text, doc_type)
        floors:             number of floors (including ground, excluding basement)
        total_area_sqm:     total built-up area per floor (sqm) — needed for slab qty
        building_type:      "residential" | "commercial" | "industrial"
        storey_height_mm:   typical floor-to-floor height in mm
        include_formwork:   whether to add formwork line items
        px_per_mm:          pixels per real mm from scale_detector (stored in result,
                            used to validate assumption-mode area estimates)
        known_scale_ratio:  drawing scale ratio from scale_detector (e.g. 100 = 1:100)
        pdf_path:           optional path to the primary PDF (used for visual fallback)
        llm_client:         optional LLM client passed to visual_element_detector

    Returns:
        StructuralQTO with elements + line_items
        mode is one of "schedule" | "assumption" | "visual"
    """
    all_elements: List[ScheduledElement] = []
    warnings: List[str] = []
    assumptions: List[str] = []

    # Eligible doc types
    _STRUCTURAL_TYPES = frozenset(
        ("structural", "plan", "drawing", "floor_plan", "foundation", "column", "rcc")
    )

    for page_idx, text, doc_type in page_texts:
        if not text or not text.strip():
            continue
        if doc_type.lower() not in _STRUCTURAL_TYPES:
            continue

        page_elements = parse_structural_schedules_from_text(text, source_page=page_idx)
        if page_elements:
            logger.debug(
                "Page %d (%s): found %d structural elements",
                page_idx, doc_type, len(page_elements),
            )
        all_elements.extend(page_elements)

    # Deduplicate by (type, label) — keep highest-confidence version
    seen: Dict[Tuple[str, str], ScheduledElement] = {}
    for elem in all_elements:
        key = (elem.element_type, elem.label)
        if key not in seen or elem.confidence > seen[key].confidence:
            seen[key] = elem
    all_elements = list(seen.values())

    mode = "schedule"
    area_source = "measured" if total_area_sqm > 0 else "assumption"

    # Visual fallback: if no schedule elements found from text and pdf_path provided
    if not all_elements and pdf_path:
        try:
            from src.analysis.qto.visual_element_detector import run_visual_detection
            _vis = run_visual_detection(
                pdf_path=pdf_path,
                page_texts=page_texts,
                llm_client=llm_client,
            )
            _vis_elements = getattr(_vis, "elements", []) or []
            for vel in _vis_elements:
                el_type = getattr(vel, "element_type", "")
                if el_type not in ("column", "beam", "footing", "slab", "wall"):
                    continue
                count = int(getattr(vel, "count", 1) or 1)
                # Default dimensions by element type
                _dims = {
                    "column":  (300, 450),
                    "beam":    (230, 450),
                    "slab":    (150, 150),
                    "footing": (1200, 500),
                    "wall":    (230, 3000),
                }.get(el_type, (230, 300))
                all_elements.append(ScheduledElement(
                    element_type=el_type,
                    label=f"{el_type[:3].upper()}_VIS",
                    width_mm=_dims[0],
                    depth_mm=_dims[1],
                    length_mm=None,
                    concrete_grade="M25",
                    steel_grade="Fe500",
                    count=count,
                    source_page=int(getattr(vel, "source_page", 0) or 0),
                    confidence=round(float(getattr(vel, "confidence", 0.7) or 0.7) * 0.85, 3),
                ))
            if all_elements:
                mode = "visual"
        except Exception:
            pass  # visual detection failed — proceed to assumption mode

    if not all_elements:
        # Fall back to assumption mode
        mode = "assumption"
        area_source = "assumption"
        all_elements, assumption_warnings = _assumption_structural_items(
            total_area_sqm=total_area_sqm,
            floors=floors,
            building_type=building_type,
            storey_height_mm=storey_height_mm,
        )
        warnings.extend(assumption_warnings)
        scale_note = (
            f", scale 1:{known_scale_ratio}" if known_scale_ratio > 0 else ""
        )
        assumptions.append(
            f"No structural schedules detected — quantities estimated from "
            f"{total_area_sqm:.0f} sqm area, {floors} floors, "
            f"{building_type} type{scale_note}"
        )
        if known_scale_ratio > 0 and total_area_sqm <= 0:
            warnings.append(
                f"Scale 1:{known_scale_ratio} detected but no floor area available; "
                "structural quantities use rule-of-thumb defaults only"
            )
    else:
        area_source = "measured" if total_area_sqm > 0 else "schedule"
        logger.info(
            "Structural takeoff: %d unique element types/labels from schedules "
            "(scale 1:%d, px_per_mm=%.4f)",
            len(all_elements), known_scale_ratio, px_per_mm,
        )

    line_items, item_warnings = generate_structural_items(
        elements=all_elements,
        floors=floors,
        total_area_sqm=total_area_sqm,
        storey_height_mm=storey_height_mm,
        include_formwork=include_formwork,
    )
    warnings.extend(item_warnings)

    return StructuralQTO(
        elements=all_elements,
        line_items=line_items,
        mode=mode,
        floors=floors,
        total_area_sqm=total_area_sqm,
        warnings=warnings,
        assumptions=assumptions,
        scale_ratio=known_scale_ratio,
        px_per_mm=px_per_mm,
        area_source=area_source,
    )
