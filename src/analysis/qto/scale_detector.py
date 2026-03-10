"""
Drawing Scale Detector — extracts scale from drawing page OCR text and
computes the px-per-mm conversion factor needed for real-world measurement.

Common Indian architectural drawing scales:
  1:100  — site plans, floor plans (A1/A0)
  1:50   — detailed plans, sections (A1)
  1:20   — stair/toilet details
  1:10   — joinery, door/window details
  1:5    — ironmongery, fixings
  1:200  — large site plans
  1:500  — campus/master plan

Usage:
    scale = detect_scale_from_text(page_ocr_text)
    # scale.ratio = 100  (meaning 1mm on paper = 100mm real)
    # real_mm = pixel_distance / scale.px_per_mm

    # Optional: cross-check against dimension callouts
    scale = confirm_scale_from_dimensions(scale, ["1200mm", "3000mm", "450mm"])
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ScaleInfo:
    ratio: int              # denominator — 100 means 1:100
    is_nts: bool            # "Not to Scale" — measurement invalid
    source_text: str
    source_page: int
    confidence: float
    px_per_mm: float = 0.0  # set by compute_px_per_mm(); 0 if unknown


# Standard drawing DPI used by fitz (PyMuPDF) when zoom=1.0 is 72 DPI.
# At zoom=2.0 (default in render_pdf_page_preview), it's 144 DPI.
_BASE_DPI = 72.0
_POINTS_PER_MM = 72.0 / 25.4   # PDF points per mm


# Regex: "1:100", "1 : 100", "SCALE 1:50", "Scale = 1:200"
_RATIO_PATTERN = re.compile(
    r'(?:scale\s*[=:]?\s*)?1\s*:\s*(\d{1,4})',
    re.IGNORECASE,
)

_NTS_PATTERN = re.compile(r'\bN\.?\s*T\.?\s*S\.?\b', re.IGNORECASE)

# "Scale Bar" presence (doesn't give ratio but indicates drawing is scaled)
_SCALE_BAR_PATTERN = re.compile(r'\bscale\s+bar\b', re.IGNORECASE)

# Valid ratios for architectural drawings (sanity check)
_VALID_RATIOS = {5, 10, 20, 25, 50, 100, 200, 500, 1000, 1250, 2500}

# ── Dimension parsing for confirm_scale_from_dimensions() ────────────────────
_DIM_MM_RE = re.compile(r'\b(\d{2,5})\s*mm\b', re.IGNORECASE)       # 1200mm
_DIM_M_RE  = re.compile(r'\b(\d{1,2}(?:\.\d{1,3})?)\s*m\b(?!m)')    # 3m, 2.5m
_DIM_FT_RE = re.compile(r"(\d{1,2})['\u2032]\s*-?\s*(\d{1,2})[\"'\u2033]")  # 3'-6"

# Plausible range for a single building element (real world, in mm)
_DIM_REAL_MIN_MM = 50.0       # 5 cm — smallest meaningful construction dimension
_DIM_REAL_MAX_MM = 50_000.0   # 50 m  — longest practical span

# Plausible paper-distance range for an A0/A1/A2 sheet annotation (mm)
_PAPER_MIN_MM = 1.5
_PAPER_MAX_MM = 2_000.0


def detect_scale_from_text(
    text: str,
    source_page: int = 0,
    zoom: float = 2.0,
) -> Optional[ScaleInfo]:
    """
    Extract drawing scale from OCR text.

    Args:
        text:        OCR text from the page
        source_page: page index for reference
        zoom:        fitz render zoom factor (used to compute px_per_mm)

    Returns:
        ScaleInfo or None if no scale found
    """
    if not text:
        return None

    # NTS check first
    if _NTS_PATTERN.search(text):
        return ScaleInfo(
            ratio=0,
            is_nts=True,
            source_text="NTS",
            source_page=source_page,
            confidence=0.90,
            px_per_mm=0.0,
        )

    candidates = []
    for m in _RATIO_PATTERN.finditer(text):
        try:
            ratio = int(m.group(1))
        except ValueError:
            continue
        if ratio <= 0 or ratio > 5000:
            continue

        # Standard ratios raised from 0.85 → 0.92: these patterns fire almost
        # exclusively on true scale annotations, not in random spec text.
        conf = 0.92 if ratio in _VALID_RATIOS else 0.60

        px_per_mm = compute_px_per_mm(ratio, zoom=zoom)
        candidates.append(ScaleInfo(
            ratio=ratio,
            is_nts=False,
            source_text=m.group(0).strip(),
            source_page=source_page,
            confidence=conf,
            px_per_mm=px_per_mm,
        ))

    if not candidates:
        return None

    # Most confident — prefer standard ratios; tie-break by first occurrence
    return max(candidates, key=lambda s: s.confidence)


def detect_scale(
    page_texts: List[Tuple[int, str, str]],
    zoom: float = 2.0,
) -> Optional[ScaleInfo]:
    """
    Scan all pages for scale notation.  Structural/drawing pages take priority.
    """
    _DRAWING_TYPES = frozenset(
        ("structural", "plan", "drawing", "section", "elevation", "detail", "floor_plan")
    )

    candidates = []
    for page_idx, text, doc_type in page_texts:
        result = detect_scale_from_text(text, source_page=page_idx, zoom=zoom)
        if result is None:
            continue
        dt = doc_type.lower()
        if any(k in dt for k in _DRAWING_TYPES):
            # Drawing-page boost: +0.05 on top of the already-raised base.
            # Cap at 0.97 to leave room for confirm_scale_from_dimensions() to
            # push confirmed scales to the absolute ceiling.
            boosted = min(0.97, result.confidence + 0.05)
            result = ScaleInfo(
                ratio=result.ratio,
                is_nts=result.is_nts,
                source_text=result.source_text,
                source_page=result.source_page,
                confidence=boosted,
                px_per_mm=result.px_per_mm,
            )
        candidates.append(result)

    if not candidates:
        return None
    return max(candidates, key=lambda s: s.confidence)


# =============================================================================
# Scale confirmation via dimension cross-check
# =============================================================================

def _parse_dims_to_mm(dimension_texts: List[str]) -> List[float]:
    """
    Convert dimension callout strings (e.g. "1200mm", "3m", "3'-6\"") to mm values.
    Returns only values in the plausible building range.
    """
    values: List[float] = []
    for raw in dimension_texts:
        # Try mm  (most common in Indian drawings)
        for m in _DIM_MM_RE.finditer(raw):
            try:
                v = float(m.group(1))
                if _DIM_REAL_MIN_MM <= v <= _DIM_REAL_MAX_MM:
                    values.append(v)
            except ValueError:
                pass
        # Try bare metres
        for m in _DIM_M_RE.finditer(raw):
            try:
                v = float(m.group(1)) * 1000.0
                if _DIM_REAL_MIN_MM <= v <= _DIM_REAL_MAX_MM:
                    values.append(v)
            except ValueError:
                pass
        # Try feet-inches (convert to mm)
        for m in _DIM_FT_RE.finditer(raw):
            try:
                feet = int(m.group(1))
                inches = int(m.group(2))
                v = (feet * 12 + inches) * 25.4
                if _DIM_REAL_MIN_MM <= v <= _DIM_REAL_MAX_MM:
                    values.append(v)
            except ValueError:
                pass
    return values


def confirm_scale_from_dimensions(
    scale_info: ScaleInfo,
    dimension_texts: List[str],
) -> ScaleInfo:
    """
    Cross-check a detected scale against dimension callouts for geometric
    plausibility, then adjust confidence accordingly.

    Logic
    -----
    For each dimension value D (real mm) and scale ratio R, the paper
    distance would be D/R mm.  A dimension is *plausible* if:
      • D is in [50 mm, 50 000 mm]  (5 cm – 50 m, normal building elements)
      • D/R is in [1.5 mm, 2 000 mm] (fits on an A0/A1/A2 sheet)

    Outcome rules (require ≥ 3 dimensions evaluated):
      • ≥ 60% plausible  →  confidence += 0.05, cap 0.97
      • < 30% plausible  →  confidence -= 0.10, floor 0.30
      • Otherwise        →  no change

    Args:
        scale_info:       ScaleInfo from detect_scale_from_text / detect_scale.
        dimension_texts:  List of raw dimension strings from callout extractor,
                          e.g. ["1200mm", "3m", "450 x 300"].

    Returns:
        New ScaleInfo with adjusted confidence (original is not mutated).
    """
    if scale_info.is_nts or scale_info.ratio <= 0:
        return scale_info  # NTS — nothing to cross-check

    dim_values = _parse_dims_to_mm(dimension_texts)
    if len(dim_values) < 3:
        return scale_info  # Not enough data to make a call

    ratio = scale_info.ratio
    plausible = 0
    for d_mm in dim_values:
        paper_mm = d_mm / ratio
        if _PAPER_MIN_MM <= paper_mm <= _PAPER_MAX_MM:
            plausible += 1

    total = len(dim_values)
    fraction = plausible / total

    current_conf = scale_info.confidence
    if fraction >= 0.60:
        new_conf = min(0.97, current_conf + 0.05)
    elif fraction < 0.30:
        new_conf = max(0.30, current_conf - 0.10)
    else:
        new_conf = current_conf

    if new_conf == current_conf:
        return scale_info  # unchanged — return same object

    return ScaleInfo(
        ratio=scale_info.ratio,
        is_nts=scale_info.is_nts,
        source_text=scale_info.source_text,
        source_page=scale_info.source_page,
        confidence=new_conf,
        px_per_mm=scale_info.px_per_mm,
    )


# =============================================================================
# Unit-conversion helpers
# =============================================================================

def compute_px_per_mm(scale_ratio: int, dpi: float = _BASE_DPI, zoom: float = 2.0) -> float:
    """
    Given a 1:N scale, return how many rendered pixels correspond to 1 real mm.

    Formula:
        1 mm on paper → scale_ratio mm real
        1 mm on paper at 72dpi = 72/25.4 ≈ 2.835 points = 2.835 pixels at zoom=1
        At zoom=Z, 1mm paper = 2.835 × Z pixels
        So: px_per_real_mm = (2.835 × zoom) / scale_ratio

    Example: scale=1:100, zoom=2
        px_per_real_mm = (2.835 × 2) / 100 = 0.0567 px/mm
        → 1 metre real = 56.7 pixels on screen
    """
    if scale_ratio <= 0:
        return 0.0
    px_per_paper_mm = _POINTS_PER_MM * zoom
    return px_per_paper_mm / scale_ratio


def pixels_to_mm(px_distance: float, scale_ratio: int, zoom: float = 2.0) -> float:
    """Convert a pixel distance on screen to real-world mm."""
    ppm = compute_px_per_mm(scale_ratio, zoom=zoom)
    if ppm <= 0:
        return 0.0
    return px_distance / ppm


def pixels_to_m(px_distance: float, scale_ratio: int, zoom: float = 2.0) -> float:
    """Convert a pixel distance on screen to real-world metres."""
    return pixels_to_mm(px_distance, scale_ratio, zoom) / 1000.0


def polygon_area_px(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula — area of polygon in pixels² from list of (x, y) points."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def pixels_area_to_sqm(area_px2: float, scale_ratio: int, zoom: float = 2.0) -> float:
    """Convert polygon area in pixels² to real-world square metres."""
    ppm = compute_px_per_mm(scale_ratio, zoom=zoom)
    if ppm <= 0:
        return 0.0
    # 1 sqm = 1e6 mm²
    real_mm2 = area_px2 / (ppm ** 2)
    return real_mm2 / 1_000_000.0


def common_scales() -> List[int]:
    """Return list of standard architectural scales sorted ascending."""
    return sorted(_VALID_RATIOS)
