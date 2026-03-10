"""
Drawing Callout Extractor — Minimal extraction from plan/detail/section/elevation pages.

Handles doc_types: plan, detail, section, elevation, structural, drawing, floor_plan

Extracts:
- Dimensions (1200mm, 3'-6", etc.)
- Material callouts (RCC M25, 200mm THK BRICK, etc.)
- Tag references (D1, W2, etc.)
- Detail/section references (DETAIL 3/A-5.01, SECTION A-A)

Confidence is doc_type-aware:
  - Tags (D-01, W-01) are very precise format → 0.90 everywhere
  - Construction-code materials (Fe415, M25, IS:456) → 0.85
  - Generic materials (granite, brick) → 0.75
  - Scale on drawing page with standard ratio → 0.92; non-drawing → 0.85
  - Room names on plan pages → 0.80; other drawing types → 0.65
  - Dimensions on detail/section pages → 0.80; plan → 0.72; default → 0.70
  - Section refs on drawing pages → 0.72
"""

import re
from typing import List, Optional


# =============================================================================
# DRAWING PAGE TYPE SETS
# =============================================================================

_DRAWING_DOC_TYPES = frozenset(
    ("drawing", "plan", "floor_plan", "structural", "elevation", "section", "detail")
)
_PLAN_DOC_TYPES = frozenset(("plan", "floor_plan", "drawing"))
_DETAIL_DOC_TYPES = frozenset(("detail", "section", "elevation"))

# Standard architectural scale ratios (mirrors scale_detector._VALID_RATIOS)
_VALID_SCALE_RATIOS = frozenset({5, 10, 20, 25, 50, 100, 200, 500, 1000, 1250, 2500})


# =============================================================================
# PATTERNS
# =============================================================================

# Dimension patterns
DIMENSION_PATTERNS = [
    re.compile(r'\b(\d{2,5})\s*mm\b', re.IGNORECASE),                    # 1200mm
    re.compile(r'\b(\d{1,2})\s*m\b(?!\s*m)'),                             # 3m (but not 3mm)
    re.compile(r"(\d{1,2})['\u2032]\s*-?\s*(\d{1,2})[\"'\u2033]"),       # 3'-6"
    re.compile(r'\b(\d{1,5})\s*[xX\u00d7]\s*(\d{1,5})\b'),               # 300x600
    re.compile(r'\b(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\b', re.IGNORECASE),  # 300mm x 600mm
]

# HIGH-PRECISION material patterns: construction codes, very specific vocabulary
_MATERIAL_PATTERNS_HIGH = [
    re.compile(r'\bRCC\s+M-?\d{2,3}\b', re.IGNORECASE),           # RCC M25, RCC M-30
    re.compile(r'\bM-?\d{2,3}\s+(?:grade|concrete)\b', re.IGNORECASE),
    re.compile(r'\bPCC\s+(?:M-?\d{2,3}|1\s*:\s*\d)', re.IGNORECASE),  # PCC M15, PCC 1:4:8
    re.compile(r'\bFe\s*\d{3}\b'),                                  # Fe415, Fe500
    re.compile(r'\bIS\s*:\s*\d{3,5}\b'),                            # IS:456, IS:2062
    re.compile(r'\b(?:TMT|HYSD)\s+Fe\s*\d{3}\b', re.IGNORECASE),  # TMT Fe500
    re.compile(r'\b(?:cement|lime)\s+mortar\s+1\s*:\s*\d', re.IGNORECASE),
]

# STANDARD-PRECISION material patterns: common words with occasional false positives
_MATERIAL_PATTERNS_STD = [
    re.compile(r'\b\d{2,3}\s*mm\s+THK\b', re.IGNORECASE),          # 200mm THK
    re.compile(r'\b\d{2,3}\s*mm\s+(?:thick|thk)\b', re.IGNORECASE),
    re.compile(r'\b(?:AAC|CLC|FLY\s*ASH)\s+block', re.IGNORECASE),
    re.compile(r'\b(?:red|clay)\s+brick', re.IGNORECASE),
    re.compile(r'\bgranite\b|\bmarble\b|\bkota\s+stone\b', re.IGNORECASE),
    re.compile(r'\bvitrified\s+tile\b|\bceramic\s+tile\b', re.IGNORECASE),
]

# Backwards-compatible flat alias used by callers that iterate MATERIAL_PATTERNS
MATERIAL_PATTERNS = _MATERIAL_PATTERNS_HIGH + _MATERIAL_PATTERNS_STD

# Door/window tag patterns
TAG_PATTERNS = [
    re.compile(r'\b(D-?\d{1,3}[A-Z]?)\b'),           # D1, D-01, D1A
    re.compile(r'\b(DR-?\d{1,3})\b'),                 # DR1, DR-01
    re.compile(r'\b(DOOR\s*\d{1,3})\b'),              # DOOR 1
    re.compile(r'\b(W-?\d{1,3}[A-Z]?)\b'),            # W1, W-01
    re.compile(r'\b(WN-?\d{1,3})\b'),                  # WN1
    re.compile(r'\b(WINDOW\s*\d{1,3})\b'),             # WINDOW 1
]

# Room name patterns
ROOM_PATTERNS = [
    re.compile(r'\b(BEDROOM|BED\s*ROOM|BR)\s*[-#]?\s*\d*\b', re.IGNORECASE),
    re.compile(r'\b(LIVING\s*ROOM?|LIVING|DRAWING\s*ROOM)\b', re.IGNORECASE),
    re.compile(r'\b(KITCHEN|KIT)\b', re.IGNORECASE),
    re.compile(r'\b(BATHROOM|BATH\s*ROOM|TOILET|WC|W\.C\.)\b', re.IGNORECASE),
    re.compile(r'\b(DINING|DINING\s*ROOM)\b', re.IGNORECASE),
    re.compile(r'\b(BALCONY|BLCNY)\b', re.IGNORECASE),
    re.compile(r'\b(LOBBY|FOYER|ENTRANCE)\b', re.IGNORECASE),
    re.compile(r'\b(STAIR|STAIRCASE)\b', re.IGNORECASE),
    re.compile(r'\b(OFFICE|RECEPTION|CONFERENCE)\b', re.IGNORECASE),
]

# Section / detail reference patterns
SECTION_REF_PATTERNS = [
    re.compile(r'\bSECTION\s+([A-Z](?:-[A-Z])?)\b', re.IGNORECASE),
    re.compile(r'\bSEC\.\s*([A-Z](?:-[A-Z])?)\b', re.IGNORECASE),
]

DETAIL_REF_PATTERNS = [
    re.compile(r'\bDETAIL\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\b(\d+)/([A-Z]-?\d+)\b'),  # 1/A-5.01
]

# Scale patterns (kept for backwards compatibility)
SCALE_PATTERNS = [
    re.compile(r'\b1\s*:\s*(\d+)\b'),
    re.compile(r'\bSCALE\s*[=:]?\s*1\s*:\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'\bNTS\b', re.IGNORECASE),
    re.compile(r'\bN\.T\.S\.', re.IGNORECASE),
]

# Internal helpers for scale parsing
_SCALE_KEYWORD_RE = re.compile(r'\bSCALE\s*[=:]?\s*1\s*:\s*(\d{1,4})\b', re.IGNORECASE)
_SCALE_BARE_RE = re.compile(r'\b1\s*:\s*(\d{1,4})\b')
_NTS_RE = re.compile(r'\bN\.?\s*T\.?\s*S\.?\b', re.IGNORECASE)


# =============================================================================
# CONFIDENCE HELPERS
# =============================================================================

def _dim_conf(doc_type: str) -> float:
    """detail/section → 0.80 | plan → 0.72 | other → 0.70"""
    dt = doc_type.lower()
    if dt in _DETAIL_DOC_TYPES:
        return 0.80
    if dt in _PLAN_DOC_TYPES:
        return 0.72
    return 0.70


def _room_conf(doc_type: str) -> float:
    """plan → 0.80 | other drawing → 0.65 | non-drawing → 0.55"""
    dt = doc_type.lower()
    if dt in _PLAN_DOC_TYPES:
        return 0.80
    if dt in _DRAWING_DOC_TYPES:
        return 0.65
    return 0.55


def _scale_conf(doc_type: str, ratio: Optional[int]) -> float:
    """
    standard ratio on drawing page → 0.92
    standard ratio on other page  → 0.85
    non-standard ratio             → 0.62
    NTS (ratio=None)               → 0.80
    """
    if ratio is None:
        return 0.80
    if ratio not in _VALID_SCALE_RATIOS:
        return 0.62
    dt = doc_type.lower()
    return 0.92 if any(k in dt for k in _DRAWING_DOC_TYPES) else 0.85


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_drawing_callouts(
    text: str,
    source_page: int,
    sheet_id: Optional[str],
    doc_type: str = "drawing",
) -> List[dict]:
    """
    Extract callouts from a drawing page (plan/detail/section/elevation).

    Args:
        text:        OCR or native text from the page.
        source_page: Page index (0-based).
        sheet_id:    Drawing sheet identifier (e.g. "A-101"), if known.
        doc_type:    Page classification from page_index.py.  Drives per-pattern
                     confidence levels.  Defaults to "drawing".

    Returns:
        List of dicts: [{text, callout_type, source_page, sheet_id, confidence}]
    """
    callouts: List[dict] = []
    seen: set = set()

    def _add(callout_text: str, callout_type: str, confidence: float) -> None:
        callout_text = " ".join(callout_text.split())
        key = f"{callout_type}:{callout_text[:50]}"
        if key in seen:
            return
        seen.add(key)
        callouts.append({
            "text": callout_text[:200],
            "callout_type": callout_type,
            "source_page": source_page,
            "sheet_id": sheet_id,
            "confidence": confidence,
        })

    text_upper = text.upper()

    # 1. Dimensions — confidence depends on page type
    for pattern in DIMENSION_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(0), "dimension", _dim_conf(doc_type))

    # 2. Material callouts — split by precision tier
    for pattern in _MATERIAL_PATTERNS_HIGH:
        for match in pattern.finditer(text):
            start = text.rfind('\n', 0, match.start()) + 1
            end = text.find('\n', match.end())
            if end == -1:
                end = min(match.end() + 80, len(text))
            _add(text[start:end].strip(), "material", 0.85)

    for pattern in _MATERIAL_PATTERNS_STD:
        for match in pattern.finditer(text):
            start = text.rfind('\n', 0, match.start()) + 1
            end = text.find('\n', match.end())
            if end == -1:
                end = min(match.end() + 80, len(text))
            _add(text[start:end].strip(), "material", 0.75)

    # 3. Door/window tags — very precise format → 0.90 everywhere
    for pattern in TAG_PATTERNS:
        for match in pattern.finditer(text_upper):
            _add(match.group(1), "tag", 0.90)

    # 4. Room names — most reliable on plan pages
    for pattern in ROOM_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(0), "room", _room_conf(doc_type))

    # 5. Section references — slightly raised now we're drawing-page-only
    for pattern in SECTION_REF_PATTERNS:
        for match in pattern.finditer(text_upper):
            _add(match.group(0), "section_ref", 0.72)

    # 6. Detail references
    for pattern in DETAIL_REF_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(0), "detail_ref", 0.65)

    # 7. Scale — doc_type-aware confidence; prefer "SCALE 1:100" over bare "1:100"
    m = _SCALE_KEYWORD_RE.search(text)
    if m:
        try:
            ratio = int(m.group(1))
            if 0 < ratio <= 5000:
                _add(m.group(0), "scale", _scale_conf(doc_type, ratio))
        except (ValueError, IndexError):
            pass
    else:
        m = _SCALE_BARE_RE.search(text)
        if m:
            try:
                ratio = int(m.group(1))
                if 0 < ratio <= 5000:
                    _add(m.group(0), "scale", _scale_conf(doc_type, ratio))
            except (ValueError, IndexError):
                pass
        elif _NTS_RE.search(text):
            _add("NTS", "scale", _scale_conf(doc_type, None))

    return callouts
