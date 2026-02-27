"""
Quantity Reconciliation — compare counts across schedule, BOQ, and drawing sources.

Builds a per-category reconciliation table (doors, windows, finishes) showing
where sources agree/disagree. Provides action application for mismatch resolution.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import csv
import io
import re
from typing import Any, Dict, List, Optional

from .reconciler import (
    _get_schedule_types,
    _get_door_schedule_count,
    _get_boq_door_qty,
    _get_window_schedule_count,
    _get_boq_window_qty,
)


# =============================================================================
# FINISH KEYWORD DETECTION
# =============================================================================

_FINISH_RE = re.compile(
    r'\b(tile|tiling|flooring|painting|plaster|putty|pop|false ceiling|'
    r'dado|skirting|vitrified|ceramic|marble|granite|wallpaper|laminate)\b',
    re.IGNORECASE,
)

_FINISH_SCHEDULE_TYPES = {"finish", "finish_schedule", "finishes", "room_finish"}


def _get_finish_schedule_count(schedule_types: Dict[str, List[dict]]) -> Optional[int]:
    """Count unique room marks in finish schedules."""
    marks = set()
    for stype, rows in schedule_types.items():
        if stype in _FINISH_SCHEDULE_TYPES or "finish" in stype:
            for r in rows:
                m = r.get("mark")
                if m:
                    marks.add(m)
    return len(marks) if marks else None


def _get_boq_finish_qty(boq_items: List[dict]) -> Optional[int]:
    """Count BOQ items mentioning finish keywords."""
    count = 0
    for item in boq_items:
        desc = (item.get("description", "") or "").lower()
        if _FINISH_RE.search(desc):
            count += 1
    return count if count > 0 else None


def _get_callout_count(quantities: List[dict], category: str) -> Optional[int]:
    """Get callout-sourced count for a category from the unified quantities list."""
    for q in quantities:
        if q.get("source_type") != "callout":
            continue
        item_lower = q.get("item", "").lower()
        if category == "doors" and "door" in item_lower:
            return int(q.get("qty", 0))
        if category == "windows" and "window" in item_lower:
            return int(q.get("qty", 0))
        if category == "rooms" and "room" in item_lower:
            return int(q.get("qty", 0))
    return None


# =============================================================================
# RECONCILIATION
# =============================================================================

def reconcile_quantities(
    quantities: List[dict],
    schedules: List[dict],
    boq_items: List[dict],
    callouts: List[dict],
) -> List[dict]:
    """
    Build quantity reconciliation table comparing counts across sources.

    Compares schedule vs BOQ vs drawing counts for doors, windows, and finishes.

    Args:
        quantities: Unified quantity list from build_all_quantities().
        schedules: Raw schedule rows.
        boq_items: Raw BOQ items.
        callouts: Raw callout list (used indirectly via quantities).

    Returns:
        List of reconciliation row dicts.
    """
    schedule_types = _get_schedule_types(schedules or [])
    recon_rows = []

    # --- Doors ---
    sched_doors = _get_door_schedule_count(schedule_types)
    boq_doors = _get_boq_door_qty(boq_items or [])
    draw_doors = _get_callout_count(quantities, "doors")

    # Only include if at least one source has data
    sources_doors = [v for v in [sched_doors, boq_doors, draw_doors] if v is not None and v > 0]
    if sources_doors:
        sched_v = sched_doors if sched_doors and sched_doors > 0 else None
        boq_v = boq_doors if boq_doors and boq_doors > 0 else None
        draw_v = draw_doors if draw_doors and draw_doors > 0 else None
        non_none = [v for v in [sched_v, boq_v, draw_v] if v is not None]
        mismatch = len(set(non_none)) > 1 if len(non_none) >= 2 else False
        max_delta = max(non_none) - min(non_none) if len(non_none) >= 2 else 0
        recon_rows.append({
            "category": "doors",
            "schedule_count": sched_v,
            "boq_count": boq_v,
            "drawing_count": draw_v,
            "mismatch": mismatch,
            "max_delta": max_delta,
            "preferred_source": None,
            "preferred_qty": None,
            "action": None,
            "evidence_refs": [],
            "notes": "",
        })

    # --- Windows ---
    sched_windows = _get_window_schedule_count(schedule_types)
    boq_windows = _get_boq_window_qty(boq_items or [])
    draw_windows = _get_callout_count(quantities, "windows")

    sources_windows = [v for v in [sched_windows, boq_windows, draw_windows] if v is not None and v > 0]
    if sources_windows:
        sched_v = sched_windows if sched_windows and sched_windows > 0 else None
        boq_v = boq_windows if boq_windows and boq_windows > 0 else None
        draw_v = draw_windows if draw_windows and draw_windows > 0 else None
        non_none = [v for v in [sched_v, boq_v, draw_v] if v is not None]
        mismatch = len(set(non_none)) > 1 if len(non_none) >= 2 else False
        max_delta = max(non_none) - min(non_none) if len(non_none) >= 2 else 0
        recon_rows.append({
            "category": "windows",
            "schedule_count": sched_v,
            "boq_count": boq_v,
            "drawing_count": draw_v,
            "mismatch": mismatch,
            "max_delta": max_delta,
            "preferred_source": None,
            "preferred_qty": None,
            "action": None,
            "evidence_refs": [],
            "notes": "",
        })

    # --- Finishes ---
    sched_finishes = _get_finish_schedule_count(schedule_types)
    boq_finishes = _get_boq_finish_qty(boq_items or [])

    sources_finishes = [v for v in [sched_finishes, boq_finishes] if v is not None and v > 0]
    if sources_finishes:
        sched_v = sched_finishes if sched_finishes and sched_finishes > 0 else None
        boq_v = boq_finishes if boq_finishes and boq_finishes > 0 else None
        non_none = [v for v in [sched_v, boq_v] if v is not None]
        mismatch = len(set(non_none)) > 1 if len(non_none) >= 2 else False
        max_delta = max(non_none) - min(non_none) if len(non_none) >= 2 else 0
        recon_rows.append({
            "category": "finishes",
            "schedule_count": sched_v,
            "boq_count": boq_v,
            "drawing_count": None,
            "mismatch": mismatch,
            "max_delta": max_delta,
            "preferred_source": None,
            "preferred_qty": None,
            "action": None,
            "evidence_refs": [],
            "notes": "",
        })

    return recon_rows


# =============================================================================
# ACTION APPLICATION (pure function)
# =============================================================================

def apply_reconciliation_action(
    recon_row: dict,
    action: str,
    note: str = "",
) -> dict:
    """
    Return a copy of recon_row with action applied.

    Actions:
        "prefer_schedule" → preferred_source="schedule", preferred_qty=schedule_count
        "prefer_boq"      → preferred_source="boq", preferred_qty=boq_count
        "create_rfi"      → action="create_rfi"
        "add_assumption"   → action="add_assumption"

    Returns:
        New dict (no mutation).
    """
    updated = dict(recon_row)

    if action == "prefer_schedule":
        updated["preferred_source"] = "schedule"
        updated["preferred_qty"] = recon_row.get("schedule_count")
        updated["action"] = "prefer_schedule"
    elif action == "prefer_boq":
        updated["preferred_source"] = "boq"
        updated["preferred_qty"] = recon_row.get("boq_count")
        updated["action"] = "prefer_boq"
    elif action == "create_rfi":
        updated["action"] = "create_rfi"
    elif action == "add_assumption":
        updated["action"] = "add_assumption"

    if note:
        updated["notes"] = note

    return updated


# =============================================================================
# EXPORT
# =============================================================================

def export_reconciliation_csv(recon_rows: List[dict]) -> str:
    """Export reconciliation table as CSV string."""
    buf = io.StringIO()
    fieldnames = [
        "category", "schedule_count", "boq_count", "drawing_count",
        "mismatch", "max_delta", "preferred_source", "preferred_qty",
        "action", "notes",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in recon_rows:
        writer.writerow({
            "category": row.get("category", ""),
            "schedule_count": row.get("schedule_count") if row.get("schedule_count") is not None else "",
            "boq_count": row.get("boq_count") if row.get("boq_count") is not None else "",
            "drawing_count": row.get("drawing_count") if row.get("drawing_count") is not None else "",
            "mismatch": row.get("mismatch", False),
            "max_delta": row.get("max_delta", 0),
            "preferred_source": row.get("preferred_source") or "",
            "preferred_qty": row.get("preferred_qty") if row.get("preferred_qty") is not None else "",
            "action": row.get("action") or "",
            "notes": row.get("notes", ""),
        })
    return buf.getvalue()
