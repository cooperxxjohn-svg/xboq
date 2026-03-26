"""
BOQ Auto-fill from Drawings.

When a tender package provides only drawings (no BOQ PDF/Excel), this module
generates a priced Bill of Quantities from the QTO extraction results already
stored in the pipeline payload.

Usage:
    from src.analysis.boq_from_drawings import (
        can_autofill,
        autofill_boq,
        AutofillResult,
    )

    if can_autofill(payload):
        result = autofill_boq(payload, region="tier2")
        # result.items  — list of dicts (description, trade, unit, qty, rate_inr, amount_inr)
        # result.trade_summary  — {trade: {item_count, total_amount}}
        # result.total_inr
        # result.source_modules — which QTO modules contributed items
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class AutofillResult:
    """Output of BOQ auto-fill from drawings."""
    items: List[dict] = field(default_factory=list)
    trade_summary: Dict[str, dict] = field(default_factory=dict)
    total_inr: float = 0.0
    source_modules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    region: str = "tier1"

    def to_dict(self) -> dict:
        return {
            "items":          self.items,
            "trade_summary":  self.trade_summary,
            "total_inr":      self.total_inr,
            "source_modules": self.source_modules,
            "warnings":       self.warnings,
            "region":         self.region,
            "item_count":     len(self.items),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def can_autofill(payload: Dict[str, Any]) -> bool:
    """
    Return True when the payload has QTO data but an empty / absent BOQ.

    Conditions:
    - payload["boq_items"] is empty or missing, AND
    - at least one QTO module produced line items
      (structural_takeoff, qto_data, dw_takeoff, painting_result,
       waterproofing_result)
    """
    boq_items = payload.get("boq_items") or []
    if boq_items:
        return False  # BOQ already present — no need to auto-fill

    # Check for any QTO source
    qto_keys = [
        "structural_takeoff",
        "qto_data",
        "dw_takeoff",
        "painting_result",
        "waterproofing_result",
        "mep_qto",
        "sitework_result",
        "brickwork_result",
        "plaster_result",
        "earthwork_result",
        "flooring_result",
        "foundation_result",
        "extdev_result",
        "prelims_result",
        "elv_result",
    ]
    return any(payload.get(k) for k in qto_keys)


def _items_from_structural(payload: Dict[str, Any]) -> List[dict]:
    """Extract line items from structural_takeoff payload section.

    Handles two schemas:
    - Sprint 38 QTO schema: {line_items: [...]}
    - Sprint 20A StructuralPipeline schema: {quantities: [{element_type, concrete_m3, steel_kg, ...}]}
    """
    st = payload.get("structural_takeoff") or {}
    if not isinstance(st, dict):
        return []

    # Sprint 38 QTO schema
    if st.get("line_items"):
        items = st["line_items"]
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append({
                "description": item.get("description") or item.get("item") or "",
                "trade":       item.get("trade") or "structural",
                "unit":        item.get("unit") or "",
                "qty":         float(item.get("qty") or item.get("quantity") or 0),
                "source":      "structural_takeoff",
            })
        return out

    # Sprint 20A StructuralPipeline schema: aggregate by element type, not instance
    quantities = st.get("quantities") or []

    # Group by element_type and sum concrete + steel
    _agg: dict = {}  # {element_type: {concrete_m3: float, steel_kg: float}}
    for q in quantities:
        if not isinstance(q, dict):
            continue
        etype = (q.get("element_type") or q.get("type") or "structural element").lower()
        concrete = float(q.get("concrete_m3") or 0)
        steel_raw = q.get("steel_kg") or 0
        steel = float(steel_raw.get("total") if isinstance(steel_raw, dict) else steel_raw)
        if etype not in _agg:
            _agg[etype] = {"concrete_m3": 0.0, "steel_kg": 0.0}
        _agg[etype]["concrete_m3"] += concrete
        _agg[etype]["steel_kg"]    += steel

    out = []
    for etype, totals in _agg.items():
        concrete = round(totals["concrete_m3"], 2)
        steel    = round(totals["steel_kg"],    2)
        if concrete > 0:
            out.append({
                "description": f"RCC {etype} M25 — concrete",
                "trade":       "structural",
                "unit":        "cum",
                "qty":         concrete,
                "source":      "structural_takeoff",
            })
        if steel > 0:
            out.append({
                "description": f"Reinforcement steel TMT Fe500 for {etype}",
                "trade":       "structural",
                "unit":        "kg",
                "qty":         steel,
                "source":      "structural_takeoff",
            })

    # Roll up to single slab concrete entry if multiple slab entries exist
    # (common when structural pipeline processes per-floor)
    _slab_concrete = sum(i["qty"] for i in out if "slab" in i["description"] and i["unit"] == "cum")
    _slab_steel    = sum(i["qty"] for i in out if "slab" in i["description"] and i["unit"] == "kg")
    out = [i for i in out if "slab" not in i["description"]]
    if _slab_concrete > 0:
        out.append({"description": "RCC slabs M25 — concrete", "trade": "structural",
                    "unit": "cum", "qty": round(_slab_concrete, 2), "source": "structural_takeoff"})
    if _slab_steel > 0:
        out.append({"description": "Reinforcement steel TMT Fe500 for slabs", "trade": "structural",
                    "unit": "kg", "qty": round(_slab_steel, 2), "source": "structural_takeoff"})

    return out


def _items_from_qto_data(payload: Dict[str, Any]) -> List[dict]:
    """Extract line items from generic qto_data payload section."""
    qto = payload.get("qto_data") or {}
    if not isinstance(qto, dict):
        return []
    items = qto.get("line_items") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "description": item.get("description") or "",
            "trade":       item.get("trade") or "civil",
            "unit":        item.get("unit") or "",
            "qty":         float(item.get("qty") or item.get("quantity") or 0),
            "source":      "qto_data",
        })
    return out


def _items_from_dw(payload: Dict[str, Any]) -> List[dict]:
    """Extract line items from door/window takeoff."""
    dw = payload.get("dw_takeoff") or {}
    if not isinstance(dw, dict):
        return []
    items = dw.get("line_items") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "description": item.get("description") or "",
            "trade":       item.get("trade") or "architectural",
            "unit":        item.get("unit") or "no",
            "qty":         float(item.get("qty") or 0),
            "source":      "dw_takeoff",
        })
    return out


def _items_from_painting(payload: Dict[str, Any]) -> List[dict]:
    pr = payload.get("painting_result") or {}
    if not isinstance(pr, dict):
        return []
    items = pr.get("line_items") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "description": item.get("description") or "",
            "trade":       "finishing",
            "unit":        item.get("unit") or "sqm",
            "qty":         float(item.get("qty") or 0),
            "source":      "painting_result",
        })
    return out


def _items_from_waterproofing(payload: Dict[str, Any]) -> List[dict]:
    wr = payload.get("waterproofing_result") or {}
    if not isinstance(wr, dict):
        return []
    items = wr.get("line_items") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "description": item.get("description") or "",
            "trade":       "waterproofing",
            "unit":        item.get("unit") or "sqm",
            "qty":         float(item.get("qty") or 0),
            "source":      "waterproofing_result",
        })
    return out


def _items_from_mep(payload: Dict[str, Any]) -> List[dict]:
    mep = payload.get("mep_qto") or {}
    if not isinstance(mep, dict):
        return []
    items = mep.get("line_items") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "description": item.get("description") or "",
            "trade":       item.get("trade") or "mep",
            "unit":        item.get("unit") or "ls",
            "qty":         float(item.get("qty") or 0),
            "source":      "mep_qto",
        })
    return out


def _items_from_generic(payload: Dict[str, Any], key: str, default_trade: str) -> List[dict]:
    """Generic extractor for any payload key with a line_items list."""
    section = payload.get(key) or {}
    if not isinstance(section, dict):
        return []
    items = section.get("line_items") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "description": item.get("description") or "",
            "trade":       item.get("trade") or default_trade,
            "unit":        item.get("unit") or "",
            "qty":         float(item.get("qty") or 0),
            "source":      key,
        })
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def autofill_boq(
    payload: Dict[str, Any],
    region: str = "tier1",
) -> AutofillResult:
    """
    Generate a priced BOQ from QTO data already in the payload.

    Parameters
    ----------
    payload : dict
        Full pipeline payload (must pass can_autofill() check first).
    region : str
        Rate region — "tier1" (metros), "tier2" (mid-tier), "tier3" (rural).

    Returns
    -------
    AutofillResult
        items: priced BOQ items with rate_inr / amount_inr fields.
        trade_summary: subtotals per trade.
        total_inr: grand total.
    """
    result = AutofillResult(region=region)

    # ── Collect raw items from each QTO module ─────────────────────────────
    collectors = [
        ("structural_takeoff",   _items_from_structural),
        ("qto_data",             _items_from_qto_data),
        ("dw_takeoff",           _items_from_dw),
        ("brickwork_result",     lambda p: _items_from_generic(p, "brickwork_result",  "masonry")),
        ("plaster_result",       lambda p: _items_from_generic(p, "plaster_result",    "finishing")),
        ("earthwork_result",     lambda p: _items_from_generic(p, "earthwork_result",  "earthwork")),
        ("flooring_result",      lambda p: _items_from_generic(p, "flooring_result",   "finishing")),
        ("painting_result",      _items_from_painting),
        ("waterproofing_result", _items_from_waterproofing),
        ("mep_qto",              _items_from_mep),
        ("sitework_result",      lambda p: _items_from_generic(p, "sitework_result",   "external")),
        ("foundation_result",    lambda p: _items_from_generic(p, "foundation_result", "structural")),
        ("extdev_result",        lambda p: _items_from_generic(p, "extdev_result",     "external")),
        ("prelims_result",       lambda p: _items_from_generic(p, "prelims_result",    "prelims")),
        ("elv_result",           lambda p: _items_from_generic(p, "elv_result",        "elv")),
    ]

    all_items: List[dict] = []
    for key, fn in collectors:
        if payload.get(key):
            items = fn(payload)
            if items:
                all_items.extend(items)
                result.source_modules.append(key)

    if not all_items:
        result.warnings.append(
            "No QTO items found — ensure at least one takeoff module ran."
        )
        return result

    # ── Apply rates via rate_engine ────────────────────────────────────────
    try:
        from src.analysis.qto.rate_engine import apply_rates, compute_trade_summary
        rated = apply_rates(all_items, region=region)
        result.items = rated
        result.trade_summary = compute_trade_summary(rated)
        result.total_inr = sum(
            info.get("total_amount", 0) for info in result.trade_summary.values()
        )
    except ImportError:
        logger.warning("rate_engine not available — BOQ items will have no rates")
        result.items = all_items
        result.warnings.append("rate_engine not available; rates not applied")
    except Exception as exc:
        logger.warning("apply_rates failed: %s", exc)
        result.items = all_items
        result.warnings.append(f"Rate application failed: {exc}")

    # ── Validate ───────────────────────────────────────────────────────────
    unrated = sum(1 for i in result.items if not i.get("rate_inr"))
    if unrated:
        pct = round(100 * unrated / len(result.items))
        result.warnings.append(
            f"{unrated} of {len(result.items)} items ({pct}%) have no matched rate — "
            "manually review these rows."
        )

    logger.info(
        "boq_from_drawings: generated %d items from %s, total INR %.0f",
        len(result.items), result.source_modules, result.total_inr,
    )
    return result
