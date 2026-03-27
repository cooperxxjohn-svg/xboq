"""Tests for T5-3 scope gap detector."""
import pytest
from src.analysis.scope_gap import detect_scope_gaps, _collect_drawing_text, ScopeGap

PAYLOAD_WITH_DRAWINGS = {
    "diagnostics": {
        "page_index": {
            "pages": [
                {"doc_type": "drawing", "discipline": "structural", "text_snippet": "footing details column reinforcement slab"},
                {"doc_type": "drawing", "discipline": "mep", "text_snippet": "water supply plumbing electrical wiring panel"},
                {"doc_type": "drawing", "discipline": "architectural", "text_snippet": "brick masonry window door schedule"},
            ]
        }
    },
    "boq_items": [
        {"trade": "civil", "description": "Excavation in hard rock", "quantity": 500, "unit": "cum"},
        {"trade": "civil", "description": "RCC Footing M25", "quantity": 100, "unit": "cum"},
        {"trade": "civil", "description": "RCC Column M25", "quantity": 80, "unit": "cum"},
        {"trade": "architectural", "description": "Brick masonry 230mm", "quantity": 800, "unit": "sqm"},
        {"trade": "architectural", "description": "Door frames and shutters", "quantity": 40, "unit": "nos"},
        # Missing: plumbing, electrical, windows
    ],
    "rfis": [],
}

MINIMAL_PAYLOAD = {
    "boq_items": [{"trade": "civil", "description": "Excavation", "quantity": 100, "unit": "cum"}],
    "rfis": [],
}


def test_returns_result():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    assert result is not None
    assert hasattr(result, "gaps")
    assert hasattr(result, "total_gaps")
    assert hasattr(result, "summary")

def test_detects_missing_plumbing():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    gap_ids = [g.gap_id for g in result.gaps]
    assert "MEP-01" in gap_ids  # plumbing signal in drawings but not in BOQ

def test_detects_missing_electrical():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    gap_ids = [g.gap_id for g in result.gaps]
    assert "MEP-03" in gap_ids  # electrical wiring in drawings but not in BOQ

def test_no_false_positive_footing():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    gap_ids = [g.gap_id for g in result.gaps]
    assert "CIV-02" not in gap_ids  # footing IS in BOQ

def test_no_false_positive_column():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    gap_ids = [g.gap_id for g in result.gaps]
    assert "CIV-03" not in gap_ids  # column IS in BOQ

def test_severity_present():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    for gap in result.gaps:
        assert gap.severity in ("high", "medium", "low")

def test_empty_payload():
    result = detect_scope_gaps({})
    assert result.total_gaps == 0

def test_minimal_payload_no_crash():
    result = detect_scope_gaps(MINIMAL_PAYLOAD)
    assert result is not None

def test_coverage_pct_range():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    assert 0.0 <= result.coverage_pct <= 100.0

def test_summary_present():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    assert len(result.summary) > 0

def test_high_severity_count():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    assert result.high_severity == sum(1 for g in result.gaps if g.severity == "high")

def test_recommendation_text():
    result = detect_scope_gaps(PAYLOAD_WITH_DRAWINGS)
    for gap in result.gaps:
        assert len(gap.recommendation) > 0
        assert gap.description in gap.recommendation

def test_collect_drawing_text():
    text = _collect_drawing_text(PAYLOAD_WITH_DRAWINGS)
    assert "footing" in text
    assert "plumbing" in text
    assert "electrical" in text
