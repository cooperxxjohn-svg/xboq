"""
Floor Count Extractor — reads floor count from drawing / specification OCR text.

Handles all common Indian architectural notation styles:
  G+4          → 5 floors (ground + 4 upper)
  B+G+4        → 5 floors above ground (basement not counted for structural)
  4 Storeyed   → 4 floors
  Ground Floor + 4 Upper Floors → 5
  Total Floors: 5
  5 Storey Building
  Building Height: 15m  → estimate 15/3 = 5 floors

Returns a FloorData with count, whether basement exists, and confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FloorData:
    """Extracted floor count information."""
    count: int              # structural floors above foundation (≥1)
    has_basement: bool      # True if "B+" or "basement" notation found
    source_text: str        # the matched text fragment
    source_page: int
    confidence: float       # 0.5 – 0.95


# =============================================================================
# REGEX PATTERNS  (compiled once at module load)
# =============================================================================

# G+4, G + 4, G+4+T (T = terrace, ignored for count)
_GP_PATTERN = re.compile(
    r'\bG\s*\+\s*(\d{1,2})(?:\s*\+\s*T(?:ERRACE)?)?\b',
    re.IGNORECASE,
)

# B+G+4, B + G + 4
_BGP_PATTERN = re.compile(
    r'\bB\s*\+\s*G\s*\+\s*(\d{1,2})\b',
    re.IGNORECASE,
)

# B+G+4+T, basement + ground + 4
_BGPT_PATTERN = re.compile(
    r'\bBASEMENT\s*\+\s*GROUND\s*\+\s*(\d{1,2})\b',
    re.IGNORECASE,
)

# "4 Storeyed", "4-storeyed", "4 Storey building"
_STOREYED_PATTERN = re.compile(
    r'\b(\d{1,2})\s*[-]?\s*(?:storeyed?|storey\s+building|storied?)\b',
    re.IGNORECASE,
)

# "4 Floors", "Total Floors: 4", "No. of Floors = 4"
_FLOORS_PATTERN = re.compile(
    r'(?:total\s+floors?|no\.?\s+of\s+floors?|number\s+of\s+floors?|floors?)\s*[=:]\s*(\d{1,2})',
    re.IGNORECASE,
)

# "Ground Floor + 4 Upper Floors", "Ground + 4 Upper"
_GROUND_UPPER_PATTERN = re.compile(
    r'ground\s+(?:floor\s*\+\s*)?(\d{1,2})\s+upper\s+floors?',
    re.IGNORECASE,
)

# "6 Floors", "5-floor building", "a 7-storey tower"
_PLAIN_FLOORS_PATTERN = re.compile(
    r'\b(\d{1,2})\s*[-]?\s*(?:floors?|storey(?:s|ed)?)\b',
    re.IGNORECASE,
)

# Building height: "15m height", "Height = 18.5m" → floors = height / 3.3
_HEIGHT_PATTERN = re.compile(
    r'(?:building\s+)?height\s*[=:]\s*(\d+(?:\.\d+)?)\s*m(?:etres?|eters?)?\b',
    re.IGNORECASE,
)

# "Basement" presence check (doesn't give count, just flag)
_BASEMENT_PATTERN = re.compile(r'\bbasement\b', re.IGNORECASE)


# =============================================================================
# CORE PARSER
# =============================================================================

def _clamp(n: int, lo: int = 1, hi: int = 60) -> int:
    """Clamp floor count to sane range."""
    return max(lo, min(hi, n))


def extract_floor_count_from_text(
    text: str,
    source_page: int = 0,
) -> Optional[FloorData]:
    """
    Attempt to extract floor count from a single page's OCR text.

    Returns the highest-confidence match, or None if no pattern fires.
    """
    if not text or not text.strip():
        return None

    candidates: List[FloorData] = []

    # ── G+4 notation (most reliable in Indian drawings) ──────────────────
    for m in _GP_PATTERN.finditer(text):
        upper = int(m.group(1))
        total = upper + 1   # +1 for ground
        candidates.append(FloorData(
            count=_clamp(total),
            has_basement=bool(_BASEMENT_PATTERN.search(text)),
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.90,
        ))

    # ── B+G+4 notation ────────────────────────────────────────────────────
    for m in _BGP_PATTERN.finditer(text):
        upper = int(m.group(1))
        total = upper + 1   # +1 for ground (basement is below grade)
        candidates.append(FloorData(
            count=_clamp(total),
            has_basement=True,
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.88,
        ))

    # ── Basement + Ground + N ─────────────────────────────────────────────
    for m in _BGPT_PATTERN.finditer(text):
        upper = int(m.group(1))
        total = upper + 1
        candidates.append(FloorData(
            count=_clamp(total),
            has_basement=True,
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.85,
        ))

    # ── "4 Storeyed" ─────────────────────────────────────────────────────
    for m in _STOREYED_PATTERN.finditer(text):
        n = int(m.group(1))
        candidates.append(FloorData(
            count=_clamp(n),
            has_basement=bool(_BASEMENT_PATTERN.search(text)),
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.85,
        ))

    # ── Total Floors: N ──────────────────────────────────────────────────
    for m in _FLOORS_PATTERN.finditer(text):
        n = int(m.group(1))
        candidates.append(FloorData(
            count=_clamp(n),
            has_basement=bool(_BASEMENT_PATTERN.search(text)),
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.82,
        ))

    # ── Ground + N Upper Floors ───────────────────────────────────────────
    for m in _GROUND_UPPER_PATTERN.finditer(text):
        upper = int(m.group(1))
        total = upper + 1
        candidates.append(FloorData(
            count=_clamp(total),
            has_basement=bool(_BASEMENT_PATTERN.search(text)),
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.82,
        ))

    # ── "N Floors" plain ─────────────────────────────────────────────────
    for m in _PLAIN_FLOORS_PATTERN.finditer(text):
        n = int(m.group(1))
        if 2 <= n <= 50:    # ignore "1 floor" (ambiguous) and huge values
            candidates.append(FloorData(
                count=_clamp(n),
                has_basement=bool(_BASEMENT_PATTERN.search(text)),
                source_text=m.group(0),
                source_page=source_page,
                confidence=0.65,
            ))

    # ── Height → floors ──────────────────────────────────────────────────
    for m in _HEIGHT_PATTERN.finditer(text):
        h_m = float(m.group(1))
        estimated = max(1, round(h_m / 3.3))   # 3.3m typical floor-to-floor
        candidates.append(FloorData(
            count=_clamp(estimated),
            has_basement=bool(_BASEMENT_PATTERN.search(text)),
            source_text=m.group(0),
            source_page=source_page,
            confidence=0.55,
        ))

    if not candidates:
        return None

    # Return highest-confidence candidate
    return max(candidates, key=lambda c: c.confidence)


def extract_floor_count(
    page_texts: List[Tuple[int, str, str]],  # (page_idx, ocr_text, doc_type)
) -> Optional[FloorData]:
    """
    Scan all pages for floor count notation. Returns the most confident result.

    Prioritises cover sheets, title blocks, and specification pages.
    """
    _HIGH_PRIORITY = frozenset(("cover", "title", "specification", "spec", "notes", "general"))
    _LOW_PRIORITY  = frozenset(("boq", "bill"))

    all_candidates: List[FloorData] = []

    for page_idx, text, doc_type in page_texts:
        if not text:
            continue
        result = extract_floor_count_from_text(text, source_page=page_idx)
        if result is None:
            continue
        # Boost confidence on high-priority pages
        dt = doc_type.lower()
        if any(k in dt for k in _HIGH_PRIORITY):
            result = FloorData(
                count=result.count,
                has_basement=result.has_basement,
                source_text=result.source_text,
                source_page=result.source_page,
                confidence=min(0.95, result.confidence + 0.10),
            )
        elif any(k in dt for k in _LOW_PRIORITY):
            result = FloorData(
                count=result.count,
                has_basement=result.has_basement,
                source_text=result.source_text,
                source_page=result.source_page,
                confidence=max(0.30, result.confidence - 0.15),
            )
        all_candidates.append(result)

    if not all_candidates:
        return None

    return max(all_candidates, key=lambda c: c.confidence)
