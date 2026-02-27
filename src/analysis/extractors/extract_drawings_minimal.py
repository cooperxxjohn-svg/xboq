"""
Drawing Callout Extractor — Minimal extraction from plan/detail/section/elevation pages.

Handles doc_types: plan, detail, section, elevation

Extracts:
- Dimensions (1200mm, 3'-6", etc.)
- Material callouts (RCC M25, 200mm THK BRICK, etc.)
- Tag references (D1, W2, etc.)
- Detail/section references (DETAIL 3/A-5.01, SECTION A-A)
"""

import re
from typing import List, Optional


# =============================================================================
# PATTERNS
# =============================================================================

# Dimension patterns
DIMENSION_PATTERNS = [
    re.compile(r'\b(\d{2,5})\s*mm\b', re.IGNORECASE),                    # 1200mm
    re.compile(r'\b(\d{1,2})\s*m\b(?!\s*m)'),                             # 3m (but not 3mm)
    re.compile(r"(\d{1,2})['\u2032]\s*-?\s*(\d{1,2})[\"'\u2033]"),       # 3'-6"
    re.compile(r'\b(\d{1,5})\s*[xX×]\s*(\d{1,5})\b'),                    # 300x600
    re.compile(r'\b(\d+)\s*mm\s*[xX×]\s*(\d+)\s*mm\b', re.IGNORECASE),  # 300mm x 600mm
]

# Material callout patterns
MATERIAL_PATTERNS = [
    re.compile(r'\bRCC\s+M-?\d{2,3}\b', re.IGNORECASE),         # RCC M25, RCC M-30
    re.compile(r'\bM-?\d{2,3}\s+(?:grade|concrete)\b', re.IGNORECASE),
    re.compile(r'\bPCC\s+(?:M-?\d{2,3}|1\s*:\s*\d)', re.IGNORECASE),  # PCC M15, PCC 1:4:8
    re.compile(r'\b\d{2,3}\s*mm\s+THK\b', re.IGNORECASE),       # 200mm THK
    re.compile(r'\b\d{2,3}\s*mm\s+(?:thick|thk)\b', re.IGNORECASE),
    re.compile(r'\bFe\s*\d{3}\b'),                                # Fe415, Fe500
    re.compile(r'\b(?:cement|lime)\s+mortar\s+1\s*:\s*\d', re.IGNORECASE),
    re.compile(r'\b(?:AAC|CLC|FLY\s*ASH)\s+block', re.IGNORECASE),
    re.compile(r'\b(?:red|clay)\s+brick', re.IGNORECASE),
    re.compile(r'\bgranite\b|\bmarble\b|\bkota\s+stone\b', re.IGNORECASE),
    re.compile(r'\bvitrified\s+tile\b|\bceramic\s+tile\b', re.IGNORECASE),
]

# Door/window tag patterns (from plan_graph.py)
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

# Reference patterns (section callout, detail marker)
SECTION_REF_PATTERNS = [
    re.compile(r'\bSECTION\s+([A-Z](?:-[A-Z])?)\b', re.IGNORECASE),
    re.compile(r'\bSEC\.\s*([A-Z](?:-[A-Z])?)\b', re.IGNORECASE),
]

DETAIL_REF_PATTERNS = [
    re.compile(r'\bDETAIL\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\b(\d+)/([A-Z]-?\d+)\b'),  # 1/A-5.01
]

# Scale patterns
SCALE_PATTERNS = [
    re.compile(r'\b1\s*:\s*(\d+)\b'),
    re.compile(r'\bSCALE\s*[=:]?\s*1\s*:\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'\bNTS\b', re.IGNORECASE),
    re.compile(r'\bN\.T\.S\.', re.IGNORECASE),
]


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_drawing_callouts(
    text: str,
    source_page: int,
    sheet_id: Optional[str],
) -> List[dict]:
    """
    Extract callouts from a drawing page (plan/detail/section/elevation).

    Returns list of dicts:
        [{text, callout_type, source_page, sheet_id, confidence}]
    """
    callouts = []
    seen = set()

    def _add(callout_text: str, callout_type: str, confidence: float = 0.6):
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

    # 1. Dimensions
    for pattern in DIMENSION_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(0), "dimension", 0.7)

    # 2. Material callouts
    for pattern in MATERIAL_PATTERNS:
        for match in pattern.finditer(text):
            # Get line context
            start = text.rfind('\n', 0, match.start()) + 1
            end = text.find('\n', match.end())
            if end == -1:
                end = min(match.end() + 80, len(text))
            context = text[start:end].strip()
            _add(context, "material", 0.75)

    # 3. Door/window tags
    for pattern in TAG_PATTERNS:
        for match in pattern.finditer(text_upper):
            _add(match.group(1), "tag", 0.8)

    # 4. Room names
    for pattern in ROOM_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(0), "room", 0.7)

    # 5. Section references
    for pattern in SECTION_REF_PATTERNS:
        for match in pattern.finditer(text_upper):
            _add(match.group(0), "section_ref", 0.65)

    # 6. Detail references
    for pattern in DETAIL_REF_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(0), "detail_ref", 0.65)

    # 7. Scale
    for pattern in SCALE_PATTERNS:
        match = pattern.search(text)
        if match:
            _add(match.group(0), "scale", 0.8)
            break  # only need one scale per page

    return callouts
