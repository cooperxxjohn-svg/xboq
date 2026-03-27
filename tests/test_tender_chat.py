"""Tests for T5-1 tender chat."""
import pytest
from src.analysis.tender_chat import answer_query, _detect_intent, _detect_trade, _detect_threshold

SAMPLE_PAYLOAD = {
    "project_name": "Test Hospital Project",
    "processing_stats": {"total_pages": 120, "deep_processed_pages": 80, "ocr_pages": 20},
    "boq_items": [
        {"trade": "civil", "description": "Excavation in hard rock", "quantity": 500, "unit": "cum", "rate_inr": 1200, "total_inr": 600000},
        {"trade": "architectural", "description": "Brick masonry 230mm", "quantity": 800, "unit": "sqm", "rate_inr": 950, "total_inr": 760000},
        {"trade": "mep", "description": "GI pipe 25mm", "quantity": 200, "unit": "rmt", "rate_inr": 450, "total_inr": 90000},
        {"trade": "civil", "description": "RCC M25 columns", "quantity": 120, "unit": "cum", "rate_inr": 6500, "total_inr": 780000},
    ],
    "rfis": [
        {"trade": "civil", "question": "Confirm bearing capacity of soil", "priority": "high"},
        {"trade": "mep", "question": "Confirm pipe material spec", "priority": "medium"},
        {"trade": "civil", "question": "Column reinforcement details", "priority": "high"},
    ],
    "blockers": [
        {"trade": "civil", "description": "Soil test report missing"},
    ],
    "qa_score": {"overall_score": 78},
}


def test_detect_intent_rfis():
    assert _detect_intent("show me all RFIs") == "rfis"
    assert _detect_intent("list clarifications") == "rfis"

def test_detect_intent_boq():
    assert _detect_intent("show BOQ items") == "boq"
    assert _detect_intent("line items over 10L") == "boq"

def test_detect_intent_cost():
    assert _detect_intent("total cost") == "cost"
    assert _detect_intent("what is the total \u20b9 value") == "cost"

def test_detect_trade():
    assert _detect_trade("civil RFIs") == "civil"
    assert _detect_trade("MEP costs") == "mep"
    assert _detect_trade("brick masonry items") == "architectural"
    assert _detect_trade("random text") is None

def test_detect_threshold():
    assert _detect_threshold("over \u20b910L") == 1_000_000
    assert _detect_threshold("above 500k") == 500_000
    assert _detect_threshold("no threshold") is None

def test_answer_rfis_all():
    resp = answer_query("show all RFIs", SAMPLE_PAYLOAD)
    assert resp["count"] == 3
    assert "3" in resp["answer"]

def test_answer_rfis_trade_filter():
    resp = answer_query("show civil RFIs", SAMPLE_PAYLOAD)
    assert resp["count"] == 2

def test_answer_rfis_high_priority():
    resp = answer_query("high priority RFIs", SAMPLE_PAYLOAD)
    assert resp["count"] == 2

def test_answer_boq_trade():
    resp = answer_query("show civil BOQ items", SAMPLE_PAYLOAD)
    assert resp["count"] == 2

def test_answer_boq_threshold():
    resp = answer_query("BOQ items over \u20b95L", SAMPLE_PAYLOAD)
    assert resp["count"] == 3  # 600k, 760k, 780k all > 500000

def test_answer_cost_total():
    resp = answer_query("what is the total cost", SAMPLE_PAYLOAD)
    assert "2,230,000" in resp["answer"].replace(",", ",")
    assert resp["intent"] == "cost"

def test_answer_blockers():
    resp = answer_query("what are the blockers", SAMPLE_PAYLOAD)
    assert resp["count"] == 1

def test_answer_pages():
    resp = answer_query("how many pages", SAMPLE_PAYLOAD)
    assert "120" in resp["answer"]

def test_answer_quality():
    resp = answer_query("what is the quality score", SAMPLE_PAYLOAD)
    assert "78" in resp["answer"]

def test_answer_summary():
    resp = answer_query("give me a summary", SAMPLE_PAYLOAD)
    assert "Hospital" in resp["answer"]
    assert resp["intent"] == "summary"

def test_empty_payload():
    resp = answer_query("show RFIs", {})
    assert "No tender loaded" in resp["answer"]

def test_no_crash_on_any_query():
    queries = ["?", "!", "\u20b9\u20b9\u20b9", "a" * 500, "show everything"]
    for q in queries:
        resp = answer_query(q, SAMPLE_PAYLOAD)
        assert "answer" in resp
