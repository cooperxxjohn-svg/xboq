"""
src/analysis/payload_validator.py

Validates the analysis payload before UI rendering.
Catches malformed data (negative quantities, missing required keys, wrong types)
before they can crash a tab render.

Usage:
    from src.analysis.payload_validator import validate_payload, PayloadWarning
    warnings = validate_payload(payload)
    # warnings is a list of PayloadWarning dataclasses
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PayloadWarning:
    field: str
    severity: str   # "error" | "warning" | "info"
    message: str
    suggested_fix: str = ""


def validate_payload(payload: dict) -> List[PayloadWarning]:
    """
    Validate the analysis payload. Returns a list of PayloadWarning objects.
    Empty list = payload is clean.
    Never raises — always returns a list (possibly empty).
    """
    warnings: List[PayloadWarning] = []
    if not isinstance(payload, dict):
        warnings.append(PayloadWarning(
            field="payload",
            severity="error",
            message="Payload is not a dict",
            suggested_fix="Re-run analysis",
        ))
        return warnings

    # ── BOQ items ──────────────────────────────────────────────────────────
    boq_items = payload.get("boq_items") or []
    if not isinstance(boq_items, list):
        warnings.append(PayloadWarning(
            field="boq_items",
            severity="error",
            message=f"boq_items is {type(boq_items).__name__}, expected list",
            suggested_fix="Re-run analysis pipeline",
        ))
    else:
        for i, item in enumerate(boq_items):
            if not isinstance(item, dict):
                warnings.append(PayloadWarning(
                    field=f"boq_items[{i}]",
                    severity="warning",
                    message=f"Item {i} is not a dict: {type(item).__name__}",
                ))
                continue
            qty = item.get("qty") or item.get("quantity") or 0
            try:
                qty_f = float(qty)
                if qty_f < 0:
                    warnings.append(PayloadWarning(
                        field=f"boq_items[{i}].qty",
                        severity="warning",
                        message=f"Negative quantity ({qty_f}) for item: {item.get('description','?')[:60]}",
                        suggested_fix="Manual review required",
                    ))
            except (TypeError, ValueError):
                warnings.append(PayloadWarning(
                    field=f"boq_items[{i}].qty",
                    severity="info",
                    message=f"Non-numeric quantity '{qty}' for: {item.get('description','?')[:60]}",
                ))
            rate = item.get("rate_inr") or item.get("rate") or 0
            try:
                rate_f = float(rate)
                if rate_f < 0:
                    warnings.append(PayloadWarning(
                        field=f"boq_items[{i}].rate_inr",
                        severity="warning",
                        message=f"Negative rate ({rate_f}) for: {item.get('description','?')[:60]}",
                    ))
            except (TypeError, ValueError):
                pass  # non-numeric rate is OK (may be missing)

    # ── RFIs ───────────────────────────────────────────────────────────────
    rfis = payload.get("rfis") or []
    if not isinstance(rfis, list):
        warnings.append(PayloadWarning(
            field="rfis",
            severity="warning",
            message=f"rfis is {type(rfis).__name__}, expected list",
        ))

    # ── QTO summary ────────────────────────────────────────────────────────
    qto = payload.get("qto_summary") or {}
    if not isinstance(qto, dict):
        warnings.append(PayloadWarning(
            field="qto_summary",
            severity="warning",
            message=f"qto_summary is {type(qto).__name__}, expected dict",
        ))

    # ── Processing stats ───────────────────────────────────────────────────
    stats = payload.get("processing_stats")
    if stats is None:
        stats = {}
    if isinstance(stats, dict) and stats:  # only check when stats key is actually present
        total = stats.get("total_pages", 0)
        if isinstance(total, (int, float)) and total == 0:
            warnings.append(PayloadWarning(
                field="processing_stats.total_pages",
                severity="warning",
                message="0 pages processed — document may be empty or unreadable",
                suggested_fix="Check that the uploaded PDF is not password-protected or corrupted",
            ))

    # ── Blockers ───────────────────────────────────────────────────────────
    blockers = payload.get("blockers") or []
    if not isinstance(blockers, list):
        warnings.append(PayloadWarning(
            field="blockers",
            severity="info",
            message=f"blockers is {type(blockers).__name__}, expected list",
        ))

    # ── Trade confidence ───────────────────────────────────────────────────
    tc = payload.get("trade_confidence") or {}
    if isinstance(tc, dict):
        for trade, conf in tc.items():
            try:
                c = float(conf)
                if not 0 <= c <= 1:
                    warnings.append(PayloadWarning(
                        field=f"trade_confidence.{trade}",
                        severity="info",
                        message=f"Confidence {c} out of [0,1] range for trade '{trade}'",
                    ))
            except (TypeError, ValueError):
                pass

    if warnings:
        errors = [w for w in warnings if w.severity == "error"]
        warns  = [w for w in warnings if w.severity == "warning"]
        logger.info(
            "Payload validation: %d error(s), %d warning(s)",
            len(errors), len(warns),
        )
    return warnings


def payload_is_valid(payload: dict, allow_warnings: bool = True) -> bool:
    """
    Quick boolean check. Returns False if any errors exist.
    If allow_warnings=False, also returns False for warnings.
    """
    warnings = validate_payload(payload)
    if allow_warnings:
        return not any(w.severity == "error" for w in warnings)
    return len(warnings) == 0
