"""
Ground Truth Diff — compare xBOQ outputs against human ground truth.

Compares quantities, schedules (doors), and BOQ items.
All functions are pure (no Streamlit, no I/O).

Sprint 20: Pilot Conversion + Paired Dataset Capture.
"""

import re
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convert to float or return None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _normalize_desc(desc: str) -> str:
    """Lowercase, strip, collapse whitespace for fuzzy matching."""
    return re.sub(r'\s+', ' ', desc.lower().strip())


def _fuzzy_match_score(a: str, b: str) -> float:
    """Simple token-overlap Jaccard similarity."""
    ta = set(_normalize_desc(a).split())
    tb = set(_normalize_desc(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Diff functions ───────────────────────────────────────────────────────

def diff_quantities(
    our_quantities: List[dict],
    gt_quantities: List[dict],
    match_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Compare our unified quantities against ground truth quantities.

    Matching by item name (fuzzy Jaccard). For matches, compares qty values.

    Args:
        our_quantities: From payload["quantities"], each has {item, unit, qty, ...}.
        gt_quantities: From gt_data, each has {item, unit, qty, ...}.
        match_threshold: Jaccard similarity threshold to consider a match.

    Returns:
        Dict with:
        - match_rate: float (0-1), fraction of GT items matched
        - matched_count: int
        - gt_count: int
        - our_count: int
        - top_mismatches: list of {gt_item, our_item, gt_qty, our_qty, delta, delta_pct}
        - missing_in_ours: list of GT item names with no match
        - extra_in_ours: list of our item names with no GT match
    """
    matched = []
    used_our: set = set()
    missing_in_ours = []

    for gt_row in gt_quantities:
        gt_item = gt_row.get("item", "")
        best_score = 0.0
        best_idx = -1
        for i, our_row in enumerate(our_quantities):
            if i in used_our:
                continue
            score = _fuzzy_match_score(gt_item, our_row.get("item", ""))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= match_threshold and best_idx >= 0:
            used_our.add(best_idx)
            matched.append((gt_row, our_quantities[best_idx], best_score))
        else:
            missing_in_ours.append(gt_row)

    extra_in_ours = [
        our_quantities[i] for i in range(len(our_quantities)) if i not in used_our
    ]

    # Build mismatch list
    top_mismatches = []
    for gt_row, our_row, score in matched:
        gt_qty = _safe_float(gt_row.get("qty"))
        our_qty = _safe_float(our_row.get("qty"))
        if gt_qty is not None and our_qty is not None and gt_qty != our_qty:
            delta = our_qty - gt_qty
            delta_pct = (delta / gt_qty * 100) if gt_qty != 0 else None
            top_mismatches.append({
                "gt_item": gt_row.get("item", ""),
                "our_item": our_row.get("item", ""),
                "gt_qty": gt_qty,
                "our_qty": our_qty,
                "delta": delta,
                "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
            })

    # Sort mismatches by absolute delta descending
    top_mismatches.sort(key=lambda m: abs(m.get("delta", 0)), reverse=True)

    gt_count = len(gt_quantities)
    match_rate = len(matched) / gt_count if gt_count > 0 else 1.0

    return {
        "match_rate": round(match_rate, 4),
        "matched_count": len(matched),
        "gt_count": gt_count,
        "our_count": len(our_quantities),
        "top_mismatches": top_mismatches[:20],
        "missing_in_ours": [r.get("item", "") for r in missing_in_ours],
        "extra_in_ours": [r.get("item", "") for r in extra_in_ours],
    }


def diff_boq(
    our_boq: List[dict],
    gt_boq: List[dict],
    match_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Compare our BOQ items against ground truth BOQ.

    Same structure as diff_quantities but keyed on description instead of item.
    """
    our_norm = [{"item": r.get("description", ""), "qty": r.get("qty", "")} for r in our_boq]
    gt_norm = [{"item": r.get("description", ""), "qty": r.get("qty", "")} for r in gt_boq]
    return diff_quantities(our_norm, gt_norm, match_threshold)


def diff_schedules(
    our_schedules: List[dict],
    gt_schedules: List[dict],
) -> Dict[str, Any]:
    """
    Compare our schedule rows against GT schedules by mark.

    Exact match on mark field, then compare qty.
    """
    our_by_mark = {s.get("mark", ""): s for s in our_schedules if s.get("mark")}
    gt_by_mark = {s.get("mark", ""): s for s in gt_schedules if s.get("mark")}

    all_marks = set(our_by_mark.keys()) | set(gt_by_mark.keys())
    matched = 0
    mismatches = []
    missing_in_ours: List[str] = []
    extra_in_ours: List[str] = []

    for mark in sorted(all_marks):
        in_ours = mark in our_by_mark
        in_gt = mark in gt_by_mark
        if in_ours and in_gt:
            matched += 1
            our_qty = _safe_float(our_by_mark[mark].get("qty"))
            gt_qty = _safe_float(gt_by_mark[mark].get("qty"))
            if our_qty is not None and gt_qty is not None and our_qty != gt_qty:
                mismatches.append({
                    "mark": mark,
                    "gt_qty": gt_qty,
                    "our_qty": our_qty,
                    "delta": our_qty - gt_qty,
                })
        elif in_gt and not in_ours:
            missing_in_ours.append(mark)
        elif in_ours and not in_gt:
            extra_in_ours.append(mark)

    gt_count = len(gt_by_mark)
    match_rate = matched / gt_count if gt_count > 0 else 1.0

    return {
        "match_rate": round(match_rate, 4),
        "matched_count": matched,
        "gt_count": gt_count,
        "our_count": len(our_by_mark),
        "top_mismatches": mismatches[:20],
        "missing_in_ours": missing_in_ours,
        "extra_in_ours": extra_in_ours,
    }


# ── Master diff ──────────────────────────────────────────────────────────

def compute_gt_diff(
    payload: Dict[str, Any],
    gt_boq: List[dict],
    gt_schedules: List[dict],
    gt_quantities: List[dict],
) -> Dict[str, Any]:
    """
    Master diff function. Compares all available GT data against payload.

    Args:
        payload: Full analysis payload.
        gt_boq: Ground truth BOQ rows (may be empty).
        gt_schedules: Ground truth schedule rows (may be empty).
        gt_quantities: Ground truth quantity rows (may be empty).

    Returns:
        Dict with per-category diff results + overall match_rate.
    """
    result: Dict[str, Any] = {"categories": {}, "overall_match_rate": None}

    # Extract our data from payload
    extraction = payload.get("extraction_summary") or {}
    our_boq = extraction.get("boq_items", [])
    our_schedules = extraction.get("schedules", [])
    our_quantities = payload.get("quantities", [])

    rates = []

    if gt_boq:
        boq_diff = diff_boq(our_boq, gt_boq)
        result["categories"]["boq"] = boq_diff
        rates.append(boq_diff["match_rate"])

    if gt_schedules:
        sched_diff = diff_schedules(our_schedules, gt_schedules)
        result["categories"]["schedules"] = sched_diff
        rates.append(sched_diff["match_rate"])

    if gt_quantities:
        qty_diff = diff_quantities(our_quantities, gt_quantities)
        result["categories"]["quantities"] = qty_diff
        rates.append(qty_diff["match_rate"])

    if rates:
        result["overall_match_rate"] = round(sum(rates) / len(rates), 4)

    return result
