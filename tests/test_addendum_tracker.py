"""Tests for T5-2 addendum tracker."""
import pytest
from src.analysis.addendum_tracker import compare_payloads, _item_key, _safe_float

BASE_ITEMS = [
    {"trade": "civil", "description": "Excavation in hard rock", "quantity": 500, "unit": "cum", "rate_inr": 1200, "total_inr": 600000},
    {"trade": "architectural", "description": "Brick masonry 230mm", "quantity": 800, "unit": "sqm", "rate_inr": 950, "total_inr": 760000},
    {"trade": "mep", "description": "GI pipe 25mm", "quantity": 200, "unit": "rmt", "rate_inr": 450, "total_inr": 90000},
]
BASE_RFIS = [{"question": "Confirm bearing capacity", "priority": "high", "trade": "civil"}]

BASE_PAYLOAD = {"boq_items": BASE_ITEMS, "rfis": BASE_RFIS}


def _make_revised(items, rfis=None):
    return {"boq_items": items, "rfis": rfis or BASE_RFIS}


def test_no_changes():
    result = compare_payloads(BASE_PAYLOAD, BASE_PAYLOAD)
    assert result.total_changes == 0
    assert not result.has_changes

def test_added_item():
    new_items = BASE_ITEMS + [{"trade": "civil", "description": "RCC Slab 150mm", "quantity": 200, "unit": "sqm", "rate_inr": 3500, "total_inr": 700000}]
    result = compare_payloads(BASE_PAYLOAD, _make_revised(new_items))
    assert len(result.added_items) == 1
    assert result.added_items[0].description == "RCC Slab 150mm"
    assert result.cost_delta_inr == pytest.approx(700000)

def test_deleted_item():
    new_items = BASE_ITEMS[:2]  # remove mep pipe
    result = compare_payloads(BASE_PAYLOAD, _make_revised(new_items))
    assert len(result.deleted_items) == 1
    assert result.cost_delta_inr == pytest.approx(-90000)

def test_qty_changed():
    new_items = [
        {"trade": "civil", "description": "Excavation in hard rock", "quantity": 600, "unit": "cum", "rate_inr": 1200, "total_inr": 720000},
        BASE_ITEMS[1], BASE_ITEMS[2],
    ]
    result = compare_payloads(BASE_PAYLOAD, _make_revised(new_items))
    assert len(result.changed_items) == 1
    assert result.changed_items[0].pct_qty_change == pytest.approx(20.0)

def test_rate_changed():
    new_items = [
        BASE_ITEMS[0],
        {"trade": "architectural", "description": "Brick masonry 230mm", "quantity": 800, "unit": "sqm", "rate_inr": 1050, "total_inr": 840000},
        BASE_ITEMS[2],
    ]
    result = compare_payloads(BASE_PAYLOAD, _make_revised(new_items))
    assert len(result.changed_items) == 1
    assert result.changed_items[0].change_type == "rate_changed"

def test_new_rfi():
    new_rfis = BASE_RFIS + [{"question": "Column reinforcement details", "priority": "high", "trade": "civil"}]
    result = compare_payloads(BASE_PAYLOAD, _make_revised(BASE_ITEMS, new_rfis))
    assert len(result.new_rfis) == 1

def test_empty_payloads():
    result = compare_payloads({}, {})
    assert result.total_changes == 0

def test_summary_text():
    new_items = BASE_ITEMS + [{"trade": "civil", "description": "Extra item", "quantity": 10, "unit": "nos", "rate_inr": 100, "total_inr": 1000}]
    result = compare_payloads(BASE_PAYLOAD, _make_revised(new_items))
    assert "added" in result.summary

def test_safe_float():
    assert _safe_float(None) == 0.0
    assert _safe_float("") == 0.0
    assert _safe_float(42) == 42.0
    assert _safe_float("3.14") == pytest.approx(3.14)
