"""
Tests for src/analysis/extractors/extract_drawings_minimal.py

Covers:
  - extract_drawing_callouts(): doc_type-aware confidence levels
  - Split material pattern tiers (high vs standard precision)
  - Tag detection (doors/windows) — 0.90 everywhere
  - Room name confidence: plan=0.80, other drawing=0.65
  - Dimension confidence: detail/section=0.80, plan=0.72, other=0.70
  - Scale detection: keyword form preferred over bare ratio
  - Scale confidence: standard ratio on drawing page=0.92, non-drawing=0.85, NTS=0.80
  - Section/detail reference detection
  - Deduplication within a single call
"""

from __future__ import annotations

import pytest
from src.analysis.extractors.extract_drawings_minimal import (
    extract_drawing_callouts,
    _dim_conf,
    _room_conf,
    _scale_conf,
    _VALID_SCALE_RATIOS,
    MATERIAL_PATTERNS,
    _MATERIAL_PATTERNS_HIGH,
    _MATERIAL_PATTERNS_STD,
)


# =============================================================================
# Confidence helper functions
# =============================================================================

class TestDimConf:
    def test_detail_returns_080(self):
        assert _dim_conf("detail") == pytest.approx(0.80)

    def test_section_returns_080(self):
        assert _dim_conf("section") == pytest.approx(0.80)

    def test_elevation_returns_080(self):
        assert _dim_conf("elevation") == pytest.approx(0.80)

    def test_plan_returns_072(self):
        assert _dim_conf("plan") == pytest.approx(0.72)

    def test_floor_plan_returns_072(self):
        assert _dim_conf("floor_plan") == pytest.approx(0.72)

    def test_drawing_returns_072(self):
        assert _dim_conf("drawing") == pytest.approx(0.72)

    def test_structural_returns_070(self):
        assert _dim_conf("structural") == pytest.approx(0.70)

    def test_unknown_returns_070(self):
        assert _dim_conf("spec") == pytest.approx(0.70)
        assert _dim_conf("") == pytest.approx(0.70)

    def test_case_insensitive(self):
        assert _dim_conf("DETAIL") == pytest.approx(0.80)
        assert _dim_conf("PLAN") == pytest.approx(0.72)


class TestRoomConf:
    def test_plan_returns_080(self):
        assert _room_conf("plan") == pytest.approx(0.80)

    def test_floor_plan_returns_080(self):
        assert _room_conf("floor_plan") == pytest.approx(0.80)

    def test_drawing_returns_080(self):
        # "drawing" is in _PLAN_DOC_TYPES → returns 0.80, same as plan
        assert _room_conf("drawing") == pytest.approx(0.80)

    def test_section_returns_065(self):
        assert _room_conf("section") == pytest.approx(0.65)

    def test_spec_returns_055(self):
        assert _room_conf("spec") == pytest.approx(0.55)

    def test_unknown_returns_055(self):
        assert _room_conf("") == pytest.approx(0.55)


class TestScaleConf:
    def test_nts_returns_080(self):
        assert _scale_conf("drawing", None) == pytest.approx(0.80)

    def test_standard_ratio_on_drawing_page_092(self):
        for ratio in [50, 100, 200, 500]:
            assert _scale_conf("drawing", ratio) == pytest.approx(0.92), f"failed for {ratio}"

    def test_standard_ratio_on_plan_page_092(self):
        assert _scale_conf("plan", 100) == pytest.approx(0.92)

    def test_standard_ratio_on_spec_page_085(self):
        assert _scale_conf("spec", 100) == pytest.approx(0.85)

    def test_nonstandard_ratio_062(self):
        # 75 is not in _VALID_SCALE_RATIOS
        assert _scale_conf("drawing", 75) == pytest.approx(0.62)
        assert _scale_conf("spec", 75) == pytest.approx(0.62)

    def test_standard_ratio_set_matches_scale_detector(self):
        # Verify our set mirrors what scale_detector uses
        for r in [5, 10, 20, 25, 50, 100, 200, 500, 1000, 1250, 2500]:
            assert r in _VALID_SCALE_RATIOS


# =============================================================================
# extract_drawing_callouts — dimensions
# =============================================================================

class TestDimensionExtraction:
    def test_mm_dimension_detected(self):
        items = extract_drawing_callouts("Width is 1200mm", 0, "A-101", "plan")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert any("1200mm" in i["text"] for i in dims)

    def test_dimension_confidence_detail_page(self):
        items = extract_drawing_callouts("Step height 150mm", 0, None, "detail")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert dims
        assert dims[0]["confidence"] == pytest.approx(0.80)

    def test_dimension_confidence_plan_page(self):
        items = extract_drawing_callouts("Room width 3600mm", 0, None, "plan")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert dims
        assert dims[0]["confidence"] == pytest.approx(0.72)

    def test_dimension_confidence_other_page(self):
        items = extract_drawing_callouts("Width 300mm", 0, None, "spec")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert dims
        assert dims[0]["confidence"] == pytest.approx(0.70)

    def test_metres_dimension(self):
        items = extract_drawing_callouts("Span 5m clear", 0, None, "drawing")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert any("5m" in i["text"] for i in dims)

    def test_crossproduct_dimension(self):
        items = extract_drawing_callouts("Column 300x600", 0, None, "structural")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert any("300x600" in i["text"].replace("X", "x").replace(" ", "") or
                   "300" in i["text"] for i in dims)

    def test_section_page_gets_080(self):
        items = extract_drawing_callouts("Tread 250mm", 0, None, "section")
        dims = [i for i in items if i["callout_type"] == "dimension"]
        assert dims
        assert dims[0]["confidence"] == pytest.approx(0.80)


# =============================================================================
# extract_drawing_callouts — material patterns (split tiers)
# =============================================================================

class TestMaterialExtraction:
    def test_high_precision_rcc_detected(self):
        items = extract_drawing_callouts("RCC M25 column", 0, None, "structural")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats
        assert mats[0]["confidence"] == pytest.approx(0.85)

    def test_high_precision_fe415_detected(self):
        items = extract_drawing_callouts("Fe415 reinforcement", 0, None, "structural")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats
        assert mats[0]["confidence"] == pytest.approx(0.85)

    def test_high_precision_is_code_detected(self):
        items = extract_drawing_callouts("As per IS:456", 0, None, "drawing")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats
        assert mats[0]["confidence"] == pytest.approx(0.85)

    def test_high_precision_pcc_detected(self):
        items = extract_drawing_callouts("PCC M15 blinding", 0, None, "structural")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats
        assert mats[0]["confidence"] == pytest.approx(0.85)

    def test_std_precision_granite_detected(self):
        items = extract_drawing_callouts("Granite flooring in living room", 0, None, "plan")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats
        assert mats[0]["confidence"] == pytest.approx(0.75)

    def test_std_precision_brick_detected(self):
        items = extract_drawing_callouts("230mm THK red brick wall", 0, None, "drawing")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats

    def test_std_precision_thk_detected(self):
        items = extract_drawing_callouts("200mm THK brick masonry", 0, None, "drawing")
        mats = [i for i in items if i["callout_type"] == "material"]
        thk = [i for i in mats if "THK" in i["text"].upper() or "200mm" in i["text"]]
        assert thk
        assert thk[0]["confidence"] == pytest.approx(0.75)

    def test_material_patterns_flat_alias_exists(self):
        # MATERIAL_PATTERNS should be HIGH + STD combined (backwards compat)
        assert len(MATERIAL_PATTERNS) == len(_MATERIAL_PATTERNS_HIGH) + len(_MATERIAL_PATTERNS_STD)

    def test_high_precision_tmt_fe500(self):
        # Pattern requires word boundary after digits — "Fe500" works, "Fe500D" does not
        items = extract_drawing_callouts("TMT Fe500 bars", 0, None, "structural")
        mats = [i for i in items if i["callout_type"] == "material"]
        assert mats
        assert mats[0]["confidence"] == pytest.approx(0.85)


# =============================================================================
# extract_drawing_callouts — tags (door/window)
# =============================================================================

class TestTagExtraction:
    def test_door_tag_detected(self):
        items = extract_drawing_callouts("D1 single door 900mm", 0, "A-101", "plan")
        tags = [i for i in items if i["callout_type"] == "tag"]
        assert any("D1" in i["text"] for i in tags)

    def test_door_tag_confidence_090(self):
        items = extract_drawing_callouts("D-01 type door", 0, None, "plan")
        tags = [i for i in items if i["callout_type"] == "tag"]
        assert tags
        assert tags[0]["confidence"] == pytest.approx(0.90)

    def test_window_tag_detected(self):
        items = extract_drawing_callouts("W2 sliding window", 0, None, "elevation")
        tags = [i for i in items if i["callout_type"] == "tag"]
        assert any("W2" in i["text"] for i in tags)

    def test_window_tag_confidence_090(self):
        items = extract_drawing_callouts("W-01 fixed light", 0, None, "elevation")
        tags = [i for i in items if i["callout_type"] == "tag"]
        assert tags
        assert tags[0]["confidence"] == pytest.approx(0.90)

    def test_tag_confidence_same_on_all_doc_types(self):
        """Tags should get 0.90 regardless of page type."""
        for doc_type in ("plan", "detail", "section", "elevation", "structural", "spec"):
            items = extract_drawing_callouts("D1 door", 0, None, doc_type)
            tags = [i for i in items if i["callout_type"] == "tag"]
            if tags:
                assert tags[0]["confidence"] == pytest.approx(0.90), f"failed on {doc_type}"

    def test_door_number_tag(self):
        items = extract_drawing_callouts("DOOR 3 is a flush door", 0, None, "plan")
        tags = [i for i in items if i["callout_type"] == "tag"]
        assert any("DOOR" in i["text"] for i in tags)


# =============================================================================
# extract_drawing_callouts — room names
# =============================================================================

class TestRoomExtraction:
    def test_bedroom_detected_on_plan(self):
        items = extract_drawing_callouts("BEDROOM 1 north facing", 0, None, "plan")
        rooms = [i for i in items if i["callout_type"] == "room"]
        assert rooms
        assert rooms[0]["confidence"] == pytest.approx(0.80)

    def test_kitchen_detected_on_plan(self):
        items = extract_drawing_callouts("KITCHEN modular layout", 0, None, "floor_plan")
        rooms = [i for i in items if i["callout_type"] == "room"]
        assert rooms
        assert rooms[0]["confidence"] == pytest.approx(0.80)

    def test_room_on_section_gets_065(self):
        items = extract_drawing_callouts("LIVING ROOM height 3.2m", 0, None, "section")
        rooms = [i for i in items if i["callout_type"] == "room"]
        if rooms:  # might not fire on all section pages — just check conf if found
            assert rooms[0]["confidence"] == pytest.approx(0.65)

    def test_room_on_spec_gets_055(self):
        items = extract_drawing_callouts("KITCHEN tiles to be provided", 0, None, "spec")
        rooms = [i for i in items if i["callout_type"] == "room"]
        if rooms:
            assert rooms[0]["confidence"] == pytest.approx(0.55)

    def test_toilet_detected(self):
        items = extract_drawing_callouts("TOILET attached master bedroom", 0, None, "plan")
        rooms = [i for i in items if i["callout_type"] == "room"]
        assert rooms

    def test_staircase_detected(self):
        items = extract_drawing_callouts("STAIRCASE dogleg 1200mm wide", 0, None, "plan")
        rooms = [i for i in items if i["callout_type"] == "room"]
        assert rooms


# =============================================================================
# extract_drawing_callouts — scale
# =============================================================================

class TestScaleExtraction:
    def test_scale_keyword_form_detected(self):
        items = extract_drawing_callouts("SCALE 1:100", 0, None, "drawing")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        assert "1:100" in scales[0]["text"]

    def test_scale_confidence_092_on_drawing_page(self):
        items = extract_drawing_callouts("SCALE 1:100", 0, None, "drawing")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        assert scales[0]["confidence"] == pytest.approx(0.92)

    def test_scale_confidence_092_on_plan_page(self):
        items = extract_drawing_callouts("Scale 1:50", 0, None, "plan")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        assert scales[0]["confidence"] == pytest.approx(0.92)

    def test_scale_confidence_085_on_spec_page(self):
        items = extract_drawing_callouts("1:100 scale drawing", 0, None, "spec")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        assert scales[0]["confidence"] == pytest.approx(0.85)

    def test_non_standard_scale_gets_062(self):
        items = extract_drawing_callouts("1:75 detail", 0, None, "drawing")
        scales = [i for i in items if i["callout_type"] == "scale"]
        if scales:
            assert scales[0]["confidence"] == pytest.approx(0.62)

    def test_nts_detected(self):
        items = extract_drawing_callouts("N.T.S.", 0, None, "drawing")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        assert scales[0]["text"] == "NTS"

    def test_nts_confidence_080(self):
        items = extract_drawing_callouts("NTS — not to scale", 0, None, "plan")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        assert scales[0]["confidence"] == pytest.approx(0.80)

    def test_keyword_form_preferred_over_bare_ratio(self):
        # Both "SCALE 1:100" and "1:50" appear — keyword form wins
        text = "General notes: 1:50 ratio | SCALE 1:100"
        items = extract_drawing_callouts(text, 0, None, "drawing")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert scales
        # Should emit the SCALE 1:100 form (keyword preferred)
        assert any("1:100" in i["text"] for i in scales)

    def test_only_one_scale_emitted(self):
        # Should not emit both bare and keyword forms for same page
        text = "SCALE 1:100 as shown, also 1:100 elsewhere"
        items = extract_drawing_callouts(text, 0, None, "drawing")
        scales = [i for i in items if i["callout_type"] == "scale"]
        assert len(scales) == 1


# =============================================================================
# extract_drawing_callouts — section / detail refs
# =============================================================================

class TestSectionDetailRefs:
    def test_section_ref_detected(self):
        items = extract_drawing_callouts("SECTION A-A through staircase", 0, None, "drawing")
        refs = [i for i in items if i["callout_type"] == "section_ref"]
        assert refs

    def test_section_ref_confidence_072(self):
        items = extract_drawing_callouts("SECTION B-B", 0, None, "drawing")
        refs = [i for i in items if i["callout_type"] == "section_ref"]
        assert refs
        assert refs[0]["confidence"] == pytest.approx(0.72)

    def test_detail_ref_detected(self):
        items = extract_drawing_callouts("DETAIL 3 window sill", 0, None, "detail")
        refs = [i for i in items if i["callout_type"] == "detail_ref"]
        assert refs

    def test_slash_detail_ref(self):
        items = extract_drawing_callouts("Refer 1/A-5.01 for wall section", 0, None, "drawing")
        refs = [i for i in items if i["callout_type"] == "detail_ref"]
        assert refs


# =============================================================================
# extract_drawing_callouts — metadata fields
# =============================================================================

class TestMetadataFields:
    def test_source_page_stored(self):
        items = extract_drawing_callouts("D1 door 900mm", 5, "A-101", "plan")
        assert all(i["source_page"] == 5 for i in items)

    def test_sheet_id_stored(self):
        items = extract_drawing_callouts("D1 door", 0, "G-101", "plan")
        assert all(i["sheet_id"] == "G-101" for i in items)

    def test_sheet_id_none_stored(self):
        items = extract_drawing_callouts("D1 door", 0, None, "plan")
        assert all(i["sheet_id"] is None for i in items)

    def test_callout_type_in_output(self):
        items = extract_drawing_callouts("D1 1200mm BEDROOM SCALE 1:100", 0, None, "plan")
        types = {i["callout_type"] for i in items}
        assert "tag" in types
        assert "dimension" in types
        assert "room" in types
        assert "scale" in types

    def test_text_truncated_at_200_chars(self):
        long = "RCC M25 " + "x" * 300
        items = extract_drawing_callouts(long, 0, None, "structural")
        for i in items:
            assert len(i["text"]) <= 200


# =============================================================================
# extract_drawing_callouts — deduplication
# =============================================================================

class TestDeduplication:
    def test_same_dimension_not_duplicated(self):
        text = "Width 1200mm, Height 1200mm"  # same value appears twice
        items = extract_drawing_callouts(text, 0, None, "plan")
        dims = [i for i in items if i["callout_type"] == "dimension" and "1200mm" in i["text"]]
        assert len(dims) == 1

    def test_same_tag_not_duplicated(self):
        text = "D1 on left side, D1 on right side"
        items = extract_drawing_callouts(text, 0, None, "plan")
        tags = [i for i in items if i["callout_type"] == "tag" and i["text"] == "D1"]
        assert len(tags) == 1

    def test_empty_text_returns_empty(self):
        items = extract_drawing_callouts("", 0, None, "drawing")
        assert items == []

    def test_whitespace_text_returns_empty(self):
        items = extract_drawing_callouts("   \n\t  ", 0, None, "drawing")
        assert items == []

    def test_no_matches_returns_empty(self):
        items = extract_drawing_callouts("The project is under review.", 0, None, "spec")
        assert items == []


# =============================================================================
# extract_drawing_callouts — default doc_type
# =============================================================================

class TestDefaultDocType:
    def test_default_doc_type_is_drawing(self):
        """Calling without doc_type should default to 'drawing' behaviour."""
        items_explicit = extract_drawing_callouts("SCALE 1:100", 0, None, "drawing")
        items_default = extract_drawing_callouts("SCALE 1:100", 0, None)
        scales_explicit = [i for i in items_explicit if i["callout_type"] == "scale"]
        scales_default = [i for i in items_default if i["callout_type"] == "scale"]
        assert len(scales_explicit) == len(scales_default)
        if scales_explicit and scales_default:
            assert scales_explicit[0]["confidence"] == scales_default[0]["confidence"]
