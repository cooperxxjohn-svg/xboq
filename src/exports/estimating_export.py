"""
Estimating Software Exports — T4-4.

One-click CSV export formatted for:
  - Sage 100 Contractor
  - Buildertrend
  - Procore
  - Generic (universal fallback)

All functions accept a list of BOQ item dicts and return a CSV string.
"""

from __future__ import annotations

import csv
import io
from typing import List, Dict

SUPPORTED_FORMATS = ["sage100", "buildertrend", "procore", "generic"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(val) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_csv(rows: List[Dict], fieldnames: List[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore",
                            lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Format: Sage 100 Contractor
# ---------------------------------------------------------------------------

_SAGE100_FIELDS = ["Cost Code", "Description", "Trade", "Quantity", "Unit",
                   "Unit Cost", "Total"]


def export_sage100_csv(boq_items: List[dict]) -> str:
    """
    Sage 100 Contractor CSV format.

    Columns: Cost Code, Description, Trade, Quantity, Unit, Unit Cost, Total
    """
    rows = []
    for item in boq_items:
        qty   = _to_float(item.get("quantity") or item.get("qty"))
        rate  = _to_float(item.get("rate_inr") or item.get("rate"))
        total = _to_float(item.get("total_inr") or item.get("total"))
        if total == 0 and qty and rate:
            total = qty * rate
        rows.append({
            "Cost Code":   item.get("item_id") or item.get("code") or "",
            "Description": item.get("description", ""),
            "Trade":       item.get("trade", ""),
            "Quantity":    round(qty, 3),
            "Unit":        item.get("unit", ""),
            "Unit Cost":   round(rate, 2),
            "Total":       round(total, 2),
        })
    return _write_csv(rows, _SAGE100_FIELDS)


# ---------------------------------------------------------------------------
# Format: Buildertrend
# ---------------------------------------------------------------------------

_BUILDERTREND_FIELDS = ["Category", "Item", "Qty", "Unit", "Cost/Unit",
                        "Total Cost", "Notes"]


def export_buildertrend_csv(boq_items: List[dict]) -> str:
    """
    Buildertrend CSV format.

    Columns: Category, Item, Qty, Unit, Cost/Unit, Total Cost, Notes
    """
    rows = []
    for item in boq_items:
        qty   = _to_float(item.get("quantity") or item.get("qty"))
        rate  = _to_float(item.get("rate_inr") or item.get("rate"))
        total = _to_float(item.get("total_inr") or item.get("total"))
        if total == 0 and qty and rate:
            total = qty * rate
        rows.append({
            "Category":  item.get("trade", "General"),
            "Item":      item.get("description", ""),
            "Qty":       round(qty, 3),
            "Unit":      item.get("unit", ""),
            "Cost/Unit": round(rate, 2),
            "Total Cost": round(total, 2),
            "Notes":     item.get("notes") or item.get("spec") or "",
        })
    return _write_csv(rows, _BUILDERTREND_FIELDS)


# ---------------------------------------------------------------------------
# Format: Procore
# ---------------------------------------------------------------------------

_PROCORE_FIELDS = ["Cost Code", "Description", "Trade", "UOM",
                   "Estimated Qty", "Unit Cost", "Budgeted Amount"]


def export_procore_csv(boq_items: List[dict]) -> str:
    """
    Procore CSV format.

    Columns: Cost Code, Description, Trade, UOM, Estimated Qty, Unit Cost,
             Budgeted Amount
    """
    rows = []
    for item in boq_items:
        qty   = _to_float(item.get("quantity") or item.get("qty"))
        rate  = _to_float(item.get("rate_inr") or item.get("rate"))
        total = _to_float(item.get("total_inr") or item.get("total"))
        if total == 0 and qty and rate:
            total = qty * rate
        rows.append({
            "Cost Code":        item.get("item_id") or item.get("code") or "",
            "Description":      item.get("description", ""),
            "Trade":            item.get("trade", ""),
            "UOM":              item.get("unit", ""),
            "Estimated Qty":    round(qty, 3),
            "Unit Cost":        round(rate, 2),
            "Budgeted Amount":  round(total, 2),
        })
    return _write_csv(rows, _PROCORE_FIELDS)


# ---------------------------------------------------------------------------
# Format: Generic (universal fallback)
# ---------------------------------------------------------------------------

_GENERIC_FIELDS = ["trade", "description", "quantity", "unit", "rate_inr",
                   "total_inr", "item_id", "source_page"]


def export_generic_csv(boq_items: List[dict]) -> str:
    """
    Generic CSV with all standard xBOQ fields.

    Columns: trade, description, quantity, unit, rate_inr, total_inr,
             item_id, source_page
    """
    rows = []
    for item in boq_items:
        qty   = _to_float(item.get("quantity") or item.get("qty"))
        rate  = _to_float(item.get("rate_inr") or item.get("rate"))
        total = _to_float(item.get("total_inr") or item.get("total"))
        if total == 0 and qty and rate:
            total = qty * rate
        rows.append({
            "trade":       item.get("trade", ""),
            "description": item.get("description", ""),
            "quantity":    round(qty, 3),
            "unit":        item.get("unit", ""),
            "rate_inr":    round(rate, 2),
            "total_inr":   round(total, 2),
            "item_id":     item.get("item_id") or item.get("code") or "",
            "source_page": item.get("source_page") or item.get("page") or "",
        })
    return _write_csv(rows, _GENERIC_FIELDS)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def export_estimating_csv(boq_items: List[dict], fmt: str = "generic") -> str:
    """Route to the appropriate format exporter."""
    fmt = fmt.lower().replace("-", "").replace("_", "")
    dispatch = {
        "sage100":      export_sage100_csv,
        "sage":         export_sage100_csv,
        "buildertrend": export_buildertrend_csv,
        "procore":      export_procore_csv,
        "generic":      export_generic_csv,
    }
    fn = dispatch.get(fmt, export_generic_csv)
    return fn(boq_items)
