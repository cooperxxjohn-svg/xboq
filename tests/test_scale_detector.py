"""
Tests for src/analysis/qto/scale_detector.py

Covers:
  - detect_scale_from_text(): standard ratio patterns, NTS, invalid ratios
  - compute_px_per_mm(): formula correctness
  - pixels_to_mm() / pixels_to_m(): unit conversion
  - polygon_area_px(): Shoelace formula
  - pixels_area_to_sqm(): area conversion
  - detect_scale(): multi-page, doc_type boost
  - common_scales()
"""

from __future__ import annotations

import math
import pytest
from src.analysis.qto.scale_detector import (
    ScaleInfo,
    detect_scale_from_text,
    detect_scale,
    confirm_scale_from_dimensions,
    compute_px_per_mm,
    pixels_to_mm,
    pixels_to_m,
    polygon_area_px,
    pixels_area_to_sqm,
    common_scales,
    _BASE_DPI,
    _POINTS_PER_MM,
)


# =============================================================================
# detect_scale_from_text
# =============================================================================

class TestDetectScaleFromText:
    # ── ratio detection ────────────────────────────────────────────────────────
    def test_simple_1_colon_100(self):
        result = detect_scale_from_text("Scale 1:100")
        assert result is not None
        assert result.ratio == 100
        assert result.is_nts is False

    def test_1_colon_50(self):
        result = detect_scale_from_text("1:50")
        assert result is not None
        assert result.ratio == 50

    def test_scale_equals_1_100(self):
        result = detect_scale_from_text("SCALE = 1:100")
        assert result is not None
        assert result.ratio == 100

    def test_scale_with_spaces(self):
        result = detect_scale_from_text("1 : 200")
        assert result is not None
        assert result.ratio == 200

    def test_bare_ratio_no_keyword(self):
        result = detect_scale_from_text("Drawing 1:50 elevation")
        assert result is not None
        assert result.ratio == 50

    def test_standard_ratio_gets_high_confidence(self):
        result = detect_scale_from_text("1:100")
        assert result is not None
        assert result.confidence == pytest.approx(0.92)

    def test_non_standard_ratio_gets_lower_confidence(self):
        # 75 is not in _VALID_RATIOS
        result = detect_scale_from_text("1:75")
        assert result is not None
        assert result.confidence == pytest.approx(0.60)

    def test_ratio_too_large_rejected(self):
        result = detect_scale_from_text("1:9999")
        # 9999 > 5000 → rejected
        assert result is None

    def test_ratio_zero_rejected(self):
        # 1:0 can't be a valid scale
        result = detect_scale_from_text("1:0")
        assert result is None

    def test_empty_text_returns_none(self):
        assert detect_scale_from_text("") is None

    def test_none_text_returns_none(self):
        assert detect_scale_from_text(None) is None

    def test_no_scale_returns_none(self):
        result = detect_scale_from_text("Floor plan of residential building")
        assert result is None

    # ── NTS ───────────────────────────────────────────────────────────────────
    def test_nts_detected(self):
        result = detect_scale_from_text("NOT TO SCALE (NTS)")
        assert result is not None
        assert result.is_nts is True
        assert result.ratio == 0

    def test_nts_with_dots(self):
        result = detect_scale_from_text("N.T.S.")
        assert result is not None
        assert result.is_nts is True

    def test_nts_confidence_090(self):
        result = detect_scale_from_text("N.T.S.")
        assert result is not None
        assert result.confidence == pytest.approx(0.90)

    def test_nts_px_per_mm_zero(self):
        result = detect_scale_from_text("NTS")
        assert result is not None
        assert result.px_per_mm == 0.0

    # ── px_per_mm populated ───────────────────────────────────────────────────
    def test_px_per_mm_set_for_1_100_zoom2(self):
        result = detect_scale_from_text("1:100", zoom=2.0)
        expected = compute_px_per_mm(100, zoom=2.0)
        assert result is not None
        assert result.px_per_mm == pytest.approx(expected, rel=1e-6)

    def test_source_page_stored(self):
        result = detect_scale_from_text("1:50", source_page=3)
        assert result is not None
        assert result.source_page == 3

    def test_source_text_contains_match(self):
        result = detect_scale_from_text("SCALE 1:100")
        assert result is not None
        assert "1:100" in result.source_text or "1" in result.source_text

    # ── multiple ratios: picks highest confidence ─────────────────────────────
    def test_prefers_standard_ratio_over_nonstandard(self):
        # "1:75" (conf=0.60) and "1:100" (conf=0.92) → 1:100 wins
        result = detect_scale_from_text("1:75 or 1:100")
        assert result is not None
        assert result.ratio == 100


# =============================================================================
# compute_px_per_mm
# =============================================================================

class TestComputePxPerMm:
    def test_formula_1_100_zoom2(self):
        # (72/25.4 × 2) / 100 ≈ 0.05669
        expected = (_POINTS_PER_MM * 2.0) / 100
        assert compute_px_per_mm(100, zoom=2.0) == pytest.approx(expected, rel=1e-6)

    def test_formula_1_50_zoom1(self):
        expected = (_POINTS_PER_MM * 1.0) / 50
        assert compute_px_per_mm(50, zoom=1.0) == pytest.approx(expected, rel=1e-6)

    def test_zero_scale_returns_zero(self):
        assert compute_px_per_mm(0) == 0.0

    def test_negative_scale_returns_zero(self):
        assert compute_px_per_mm(-1) == 0.0

    def test_larger_zoom_gives_more_pixels_per_mm(self):
        ppm1 = compute_px_per_mm(100, zoom=1.0)
        ppm2 = compute_px_per_mm(100, zoom=2.0)
        assert ppm2 > ppm1

    def test_smaller_scale_ratio_gives_more_pixels(self):
        # 1:50 drawing is bigger on screen than 1:100
        ppm50 = compute_px_per_mm(50)
        ppm100 = compute_px_per_mm(100)
        assert ppm50 > ppm100


# =============================================================================
# pixels_to_mm / pixels_to_m
# =============================================================================

class TestPixelConversions:
    """
    At scale 1:100, zoom=2:
      px_per_mm ≈ (72/25.4 × 2) / 100 ≈ 0.05669
      56.69 px → 1000 mm = 1 m
    """
    def _ppm(self, scale=100, zoom=2.0):
        return compute_px_per_mm(scale, zoom=zoom)

    def test_pixels_to_mm_round_trip(self):
        scale, zoom = 100, 2.0
        ppm = self._ppm(scale, zoom)
        # 100 mm real → ppm * 100 pixels
        px = ppm * 100
        result = pixels_to_mm(px, scale, zoom)
        assert result == pytest.approx(100.0, rel=1e-5)

    def test_pixels_to_m_1000mm(self):
        scale, zoom = 100, 2.0
        ppm = self._ppm(scale, zoom)
        px = ppm * 1000  # 1 m real
        result = pixels_to_m(px, scale, zoom)
        assert result == pytest.approx(1.0, rel=1e-5)

    def test_zero_pixels_gives_zero_mm(self):
        assert pixels_to_mm(0, 100) == pytest.approx(0.0)

    def test_zero_scale_gives_zero(self):
        assert pixels_to_mm(100, 0) == 0.0

    def test_pixels_to_m_is_pixels_to_mm_divided_by_1000(self):
        px = 57.0
        mm_val = pixels_to_mm(px, 100, zoom=2.0)
        m_val = pixels_to_m(px, 100, zoom=2.0)
        assert m_val == pytest.approx(mm_val / 1000.0, rel=1e-9)


# =============================================================================
# polygon_area_px  (Shoelace)
# =============================================================================

class TestPolygonAreaPx:
    def test_unit_square(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert polygon_area_px(points) == pytest.approx(1.0)

    def test_2x3_rectangle(self):
        points = [(0, 0), (2, 0), (2, 3), (0, 3)]
        assert polygon_area_px(points) == pytest.approx(6.0)

    def test_right_triangle(self):
        # base=3, height=4 → area = 6
        points = [(0, 0), (3, 0), (0, 4)]
        assert polygon_area_px(points) == pytest.approx(6.0)

    def test_clockwise_and_ccw_same_area(self):
        # Shoelace returns abs value
        ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        cw  = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert polygon_area_px(ccw) == pytest.approx(polygon_area_px(cw))

    def test_fewer_than_3_points_returns_zero(self):
        assert polygon_area_px([]) == 0.0
        assert polygon_area_px([(0, 0)]) == 0.0
        assert polygon_area_px([(0, 0), (1, 0)]) == 0.0

    def test_large_polygon(self):
        # 100px × 100px square
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert polygon_area_px(points) == pytest.approx(10000.0)


# =============================================================================
# pixels_area_to_sqm
# =============================================================================

class TestPixelsAreaToSqm:
    """
    At scale 1:100, zoom=2:
      px_per_mm ≈ 0.05669
      A 10m × 10m room = 10000 sqm … wait no, 10m × 10m = 100 sqm
      10m = 10000mm real → on screen at px_per_mm = 566.9 px per 10m
      area_px2 = 566.9² = 321,375 px²
      → should give 100 sqm
    """
    def test_known_area_sqm(self):
        scale, zoom = 100, 2.0
        ppm = compute_px_per_mm(scale, zoom)
        # 10m × 10m = 100 sqm; 10m = 10000mm
        side_px = ppm * 10_000   # pixels for 10m
        area_px2 = side_px ** 2
        result = pixels_area_to_sqm(area_px2, scale, zoom)
        assert result == pytest.approx(100.0, rel=1e-5)

    def test_zero_area(self):
        assert pixels_area_to_sqm(0, 100) == 0.0

    def test_zero_scale_returns_zero(self):
        assert pixels_area_to_sqm(1000, 0) == 0.0


# =============================================================================
# detect_scale  (multi-page)
# =============================================================================

class TestDetectScale:
    def test_picks_best_scale_across_pages(self):
        pages = [
            (0, "1:100", "drawing"),
            (1, "1:50", "spec"),
        ]
        result = detect_scale(pages)
        assert result is not None
        # drawing page: 0.92 + 0.05 = 0.97; spec page: 0.92 → drawing wins
        assert result.ratio == 100

    def test_drawing_page_type_boosts_confidence(self):
        pages = [(0, "1:50", "structural")]
        result = detect_scale(pages)
        assert result is not None
        # 0.92 + 0.05 = 0.97
        assert result.confidence == pytest.approx(0.97)

    def test_non_drawing_page_no_boost(self):
        pages = [(0, "1:50", "spec")]
        result = detect_scale(pages)
        assert result is not None
        assert result.confidence == pytest.approx(0.92)

    def test_nts_on_drawing_page_still_returns_nts(self):
        pages = [(0, "NTS", "drawing")]
        result = detect_scale(pages)
        assert result is not None
        assert result.is_nts is True

    def test_empty_pages_returns_none(self):
        assert detect_scale([]) is None

    def test_all_empty_text_returns_none(self):
        pages = [(0, "", "drawing"), (1, "   ", "plan")]
        assert detect_scale(pages) is None

    def test_confidence_cap_at_097(self):
        pages = [(0, "1:100", "structural")]  # 0.92 + 0.05 = 0.97
        result = detect_scale(pages)
        assert result is not None
        assert result.confidence <= 0.97

    def test_floor_plan_type_boosts(self):
        pages = [(0, "1:200", "floor_plan")]
        result = detect_scale(pages)
        assert result is not None
        assert result.confidence > 0.92  # boosted above base

    def test_multiple_pages_selects_most_confident(self):
        pages = [
            (0, "1:75", "unknown"),    # 0.60
            (1, "1:100", "drawing"),   # 0.92 + 0.05 = 0.97
        ]
        result = detect_scale(pages)
        assert result.ratio == 100


# =============================================================================
# common_scales
# =============================================================================

class TestCommonScales:
    def test_returns_sorted_list(self):
        scales = common_scales()
        assert scales == sorted(scales)

    def test_contains_standard_values(self):
        scales = common_scales()
        for expected in [50, 100, 200, 500]:
            assert expected in scales

    def test_all_positive(self):
        for s in common_scales():
            assert s > 0

    def test_returns_list(self):
        assert isinstance(common_scales(), list)


# =============================================================================
# confirm_scale_from_dimensions
# =============================================================================

def _make_scale(ratio: int = 100, conf: float = 0.92) -> ScaleInfo:
    """Helper: build a ScaleInfo with given ratio and confidence."""
    return ScaleInfo(
        ratio=ratio,
        is_nts=False,
        source_text=f"1:{ratio}",
        source_page=0,
        confidence=conf,
        px_per_mm=compute_px_per_mm(ratio),
    )


class TestConfirmScaleFromDimensions:
    # ── NTS / zero ratio passthrough ────────────────────────────────────────
    def test_nts_returns_unchanged(self):
        nts = ScaleInfo(ratio=0, is_nts=True, source_text="NTS",
                        source_page=0, confidence=0.90, px_per_mm=0.0)
        result = confirm_scale_from_dimensions(nts, ["1200mm", "3000mm", "450mm"])
        assert result is nts

    def test_zero_ratio_returns_unchanged(self):
        info = ScaleInfo(ratio=0, is_nts=False, source_text="",
                         source_page=0, confidence=0.50, px_per_mm=0.0)
        result = confirm_scale_from_dimensions(info, ["1200mm", "3000mm", "450mm"])
        assert result is info

    # ── insufficient data passthrough ──────────────────────────────────────
    def test_fewer_than_3_dims_returns_unchanged(self):
        info = _make_scale(100, 0.92)
        result = confirm_scale_from_dimensions(info, ["1200mm", "3000mm"])
        assert result is info  # only 2 values → no change

    def test_empty_dims_returns_unchanged(self):
        info = _make_scale(100, 0.92)
        result = confirm_scale_from_dimensions(info, [])
        assert result is info

    # ── plausible dimensions boost confidence ──────────────────────────────
    def test_all_plausible_dims_boost_confidence(self):
        # At 1:100, these paper distances are all well within A0 sheet
        # 1200mm / 100 = 12mm paper; 3000mm / 100 = 30mm; 600mm / 100 = 6mm
        # all in [1.5mm, 2000mm] → 100% plausible → +0.05
        info = _make_scale(100, 0.92)
        result = confirm_scale_from_dimensions(info, ["1200mm", "3000mm", "600mm"])
        assert result.confidence == pytest.approx(0.92 + 0.05)

    def test_boost_capped_at_097(self):
        info = _make_scale(100, 0.95)  # already high
        result = confirm_scale_from_dimensions(info, ["1200mm", "3000mm", "600mm"])
        assert result.confidence <= 0.97

    # ── implausible dimensions reduce confidence ───────────────────────────
    def test_implausible_dims_reduce_confidence(self):
        # At 1:100, 1mm / 100 = 0.01mm paper — below 1.5mm minimum → implausible
        # Use tiny dims (< 50mm real, also filtered by _DIM_REAL_MIN_MM)
        # Actually tiny paper: 50_000mm / 100 = 500mm — fits.
        # Use wrong scale scenario: 1:5, but dims are 50m spans (50000mm)
        # 50000 / 5 = 10000mm paper — way over 2000mm limit → implausible
        info = _make_scale(5, 0.92)
        # Real 20m–50m spans at 1:5 won't fit on any sheet
        result = confirm_scale_from_dimensions(
            info, ["25000mm", "40000mm", "30000mm", "35000mm"]
        )
        assert result.confidence < 0.92  # penalised

    def test_confidence_floor_at_030(self):
        # Very wrong scale with many implausible dims
        info = _make_scale(5, 0.35)  # start at floor edge
        result = confirm_scale_from_dimensions(
            info, ["25000mm", "40000mm", "30000mm", "35000mm"]
        )
        assert result.confidence >= 0.30

    # ── middle zone: no change ────────────────────────────────────────────
    def test_mixed_plausibility_no_change(self):
        # 50% plausible is between 30% and 60% — no change
        # At 1:100: 1200mm→12mm (ok), 3000mm→30mm (ok), 25000mm→250mm (ok),
        # 40000mm→400mm (ok) — all actually fine at 1:100
        # Force ~50%: mix of very tiny and normal
        # 30mm/100=0.3mm (below 1.5mm, implausible), 2000mm/100=20mm (ok)
        # But 30mm real is < _DIM_REAL_MIN_MM (50mm) so it's filtered out
        # Use metres: "0.03m" is 30mm — again filtered.
        # Use a scale where exactly half are implausible:
        # 1:1000: 1200mm/1000=1.2mm (< 1.5mm → implausible), 3000mm/1000=3mm (ok)
        # 600mm/1000=0.6mm (implausible), 5000mm/1000=5mm (ok)
        # → 2 plausible / 4 total = 50% → in range [30%, 60%) → no change
        info = _make_scale(1000, 0.92)
        result = confirm_scale_from_dimensions(
            info, ["1200mm", "3000mm", "600mm", "5000mm"]
        )
        assert result.confidence == pytest.approx(0.92)
        assert result is info  # same object returned when unchanged

    # ── dimension parsing: mm, m, ft-in ──────────────────────────────────
    def test_parses_metres(self):
        # "3m" = 3000mm, "2.5m" = 2500mm, "1.5m" = 1500mm
        # At 1:100: 3000/100=30mm, 2500/100=25mm, 1500/100=15mm → all plausible
        info = _make_scale(100, 0.92)
        result = confirm_scale_from_dimensions(info, ["3m", "2.5m", "1.5m"])
        assert result.confidence > 0.92

    def test_parses_feet_inches(self):
        # 3'-6" = (3*12+6)*25.4 = 1066mm; 8'-0" = 2438mm; 4'-2" = 1270mm
        # At 1:100: 10.66mm, 24.38mm, 12.7mm → all plausible
        info = _make_scale(100, 0.92)
        result = confirm_scale_from_dimensions(info, ["3'-6\"", "8'-0\"", "4'-2\""])
        assert result.confidence > 0.92

    # ── immutability: original ScaleInfo not mutated ──────────────────────
    def test_original_not_mutated(self):
        info = _make_scale(100, 0.92)
        original_conf = info.confidence
        _ = confirm_scale_from_dimensions(info, ["1200mm", "3000mm", "600mm"])
        assert info.confidence == pytest.approx(original_conf)
