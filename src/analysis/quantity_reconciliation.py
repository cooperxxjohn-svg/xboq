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
# A5: QUANTITY RECONCILIATION EXTENSION CONSTANTS
# =============================================================================

# Expected prelims as % of civil cost (band: min, max)
_PRELIMS_PCT_BY_TYPE: dict = {
    "hostel":      (0.08, 0.10),
    "hospital":    (0.10, 0.12),
    "office":      (0.07, 0.09),
    "residential": (0.07, 0.09),
    "academic":    (0.08, 0.10),
    "industrial":  (0.06, 0.08),
    "default":     (0.05, 0.12),
}
_PAINT_RATIO_MIN = 2.5
_PAINT_RATIO_MAX = 4.0
_PAINT_RATIO_MID = 3.2
_WP_WET_FRACTION_MIN = 0.08
_WP_WET_FRACTION_MAX = 0.25


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
    qto_context: Optional[dict] = None,
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

    # Sprint 22 Phase 4: BOQ item-level reconciliation against structural takeoff
    # Compare key BOQ items (concrete, steel) against drawing-derived quantities
    recon_rows.extend(
        _reconcile_boq_vs_structural(boq_items, quantities)
    )

    # A5: Extend with prelims, waterproofing, and painting area checks
    if qto_context:
        recon_rows = list(recon_rows) if not isinstance(recon_rows, list) else recon_rows
        recon_rows.extend(_reconcile_prelims(
            boq_items,
            qto_context.get("total_area_sqm", 0.0),
            qto_context.get("building_type", "default"),
        ))
        wp_wet   = qto_context.get("wp_wet_area_sqm", 0.0)
        wp_roof  = qto_context.get("wp_roof_area_sqm", 0.0)
        total_a  = qto_context.get("total_area_sqm", 0.0)
        paint_w  = qto_context.get("paint_int_wall_sqm", 0.0)
        if wp_wet > 0 or wp_roof > 0:
            recon_rows.extend(_reconcile_waterproofing_area(wp_wet, wp_roof, total_a))
        if paint_w > 0:
            recon_rows.extend(_reconcile_painting_area(paint_w, total_a))

    return recon_rows


# =============================================================================
# A5: PRELIMS, WATERPROOFING, AND PAINTING RECONCILIATION HELPERS
# =============================================================================

def _reconcile_prelims(boq_items: list, total_area_sqm: float, building_type: str = "default") -> list:
    """Check prelims BOQ total falls within expected % of civil cost."""
    prelims_keywords = {"prelim", "mobilisation", "site establishment", "overhead", "temporary work"}
    civil_cost = sum(
        float(i.get("qty") or 0) * float(i.get("rate_inr") or i.get("rate") or 0)
        for i in boq_items
        if not any(k in (i.get("description") or "").lower() for k in prelims_keywords)
    )
    prelims_cost = sum(
        float(i.get("qty") or 0) * float(i.get("rate_inr") or i.get("rate") or 0)
        for i in boq_items
        if any(k in (i.get("description") or "").lower() for k in prelims_keywords)
    )
    if civil_cost <= 0:
        return []
    actual_pct = prelims_cost / civil_cost
    bt = (building_type or "default").lower()
    band = next(
        (v for k, v in _PRELIMS_PCT_BY_TYPE.items() if k != "default" and k in bt),
        _PRELIMS_PCT_BY_TYPE["default"],
    )
    lo, hi = band
    mismatch = not (lo <= actual_pct <= hi)
    return [{
        "category": "prelims_pct",
        "schedule_count": None,
        "boq_count": round(actual_pct * 100, 2),
        "drawing_count": round(((lo + hi) / 2) * 100, 2),
        "mismatch": mismatch,
        "max_delta": round(abs(actual_pct - (lo + hi) / 2) * 100, 2),
        "preferred_source": None,
        "preferred_qty": None,
        "action": "review_prelims_cost" if mismatch else None,
        "evidence_refs": [],
        "notes": f"Prelims {actual_pct*100:.1f}% of civil cost; expected {lo*100:.0f}–{hi*100:.0f}%",
    }]


def _reconcile_waterproofing_area(wp_wet_area_sqm: float, wp_roof_area_sqm: float, total_area_sqm: float) -> list:
    """Waterproofing area sanity check."""
    rows = []
    if total_area_sqm <= 0:
        return rows
    if wp_wet_area_sqm > 0:
        ratio = wp_wet_area_sqm / total_area_sqm
        expected = total_area_sqm * 0.15
        mismatch = not (_WP_WET_FRACTION_MIN <= ratio <= _WP_WET_FRACTION_MAX)
        rows.append({
            "category": "wp_wet_area",
            "schedule_count": None,
            "boq_count": round(wp_wet_area_sqm, 1),
            "drawing_count": round(expected, 1),
            "mismatch": mismatch,
            "max_delta": round(abs(wp_wet_area_sqm - expected), 1),
            "preferred_source": None,
            "preferred_qty": None,
            "action": "verify_wet_area" if mismatch else None,
            "evidence_refs": [],
            "notes": f"WP wet area {wp_wet_area_sqm:.0f} sqm = {ratio*100:.1f}% of BUA; expected 8–25%",
        })
    if wp_roof_area_sqm > 0:
        expected_roof = total_area_sqm * 0.20
        mismatch = wp_roof_area_sqm < expected_roof * 0.4
        rows.append({
            "category": "wp_roof_area",
            "schedule_count": None,
            "boq_count": round(wp_roof_area_sqm, 1),
            "drawing_count": round(expected_roof, 1),
            "mismatch": mismatch,
            "max_delta": round(abs(wp_roof_area_sqm - expected_roof), 1),
            "preferred_source": None,
            "preferred_qty": None,
            "action": "verify_roof_waterproofing" if mismatch else None,
            "evidence_refs": [],
            "notes": f"WP roof area {wp_roof_area_sqm:.0f} sqm vs expected ≥{expected_roof:.0f} sqm",
        })
    return rows


def _reconcile_painting_area(paint_int_wall_sqm: float, total_area_sqm: float) -> list:
    """Painting area sanity check: interior wall paint expected 2.5–4.0× floor area."""
    if total_area_sqm <= 0 or paint_int_wall_sqm <= 0:
        return []
    ratio = paint_int_wall_sqm / total_area_sqm
    expected = total_area_sqm * _PAINT_RATIO_MID
    mismatch = not (_PAINT_RATIO_MIN <= ratio <= _PAINT_RATIO_MAX)
    return [{
        "category": "painting_int_wall",
        "schedule_count": None,
        "boq_count": round(paint_int_wall_sqm, 1),
        "drawing_count": round(expected, 1),
        "mismatch": mismatch,
        "max_delta": round(abs(paint_int_wall_sqm - expected), 1),
        "preferred_source": None,
        "preferred_qty": None,
        "action": "verify_painting_area" if mismatch else None,
        "evidence_refs": [],
        "notes": f"Int. paint area {paint_int_wall_sqm:.0f} sqm = {ratio:.1f}× BUA; expected 2.5–4.0×",
    }]


# =============================================================================
# Sprint 22: BOQ vs STRUCTURAL TAKEOFF RECONCILIATION
# =============================================================================

_BOQ_CONCRETE_RE = re.compile(
    r'\b(rcc|pcc|concrete|m\s*\d{2})\b', re.IGNORECASE
)
_BOQ_STEEL_RE = re.compile(
    r'\b(reinforcement|rebar|steel|tmt|fe\s*\d{3}|bar\s*bending)\b', re.IGNORECASE
)
_BOQ_MASONRY_RE = re.compile(
    r'\b(brick|block|masonry|aac)\b', re.IGNORECASE
)
_BOQ_PLASTER_RE = re.compile(
    r'\b(plaster|rendering|cement\s*mortar)\b', re.IGNORECASE
)
_BOQ_PAINT_RE = re.compile(
    r'\b(paint|primer|putty|distemper|emulsion)\b', re.IGNORECASE
)


def _reconcile_boq_vs_structural(
    boq_items: List[dict],
    quantities: List[dict],
) -> List[dict]:
    """Reconcile BOQ items against structural/drawing-derived quantities.

    Groups BOQ items by material type (concrete, steel, masonry, etc.)
    and compares against any drawing-derived quantities of the same type.

    Returns additional reconciliation rows for significant discrepancies.
    """
    if not boq_items:
        return []

    recon_rows = []

    # Group BOQ items by material type
    material_groups: Dict[str, List[dict]] = {
        "concrete": [],
        "steel": [],
        "masonry": [],
        "plaster": [],
        "paint": [],
    }
    material_re = {
        "concrete": _BOQ_CONCRETE_RE,
        "steel": _BOQ_STEEL_RE,
        "masonry": _BOQ_MASONRY_RE,
        "plaster": _BOQ_PLASTER_RE,
        "paint": _BOQ_PAINT_RE,
    }

    for item in boq_items:
        desc = (item.get("description") or "").lower()
        for mat_type, pattern in material_re.items():
            if pattern.search(desc):
                material_groups[mat_type].append(item)
                break  # assign to first matching group

    # Get drawing-derived quantities for comparison
    drawing_qtys: Dict[str, float] = {}
    for q in (quantities or []):
        if q.get("source_type") in ("callout", "schedule"):
            item_lower = q.get("item", "").lower()
            for mat_type, pattern in material_re.items():
                if pattern.search(item_lower):
                    drawing_qtys[mat_type] = drawing_qtys.get(mat_type, 0) + (q.get("qty") or 0)

    # Build reconciliation rows for each material group
    for mat_type, items in material_groups.items():
        if not items:
            continue

        # Sum BOQ quantities for this material type
        boq_total = 0.0
        boq_unit = None
        for item in items:
            qty = item.get("qty")
            if qty is not None:
                try:
                    boq_total += float(qty)
                except (TypeError, ValueError):
                    pass
            if not boq_unit and item.get("unit"):
                boq_unit = item.get("unit")

        if boq_total <= 0:
            continue

        drawing_total = drawing_qtys.get(mat_type)

        # Calculate delta percentage
        delta_pct = None
        mismatch = False
        if drawing_total is not None and drawing_total > 0:
            delta_pct = ((boq_total - drawing_total) / drawing_total) * 100
            mismatch = abs(delta_pct) > 15

        recon_rows.append({
            "category": mat_type,
            "schedule_count": None,
            "boq_count": round(boq_total, 1),
            "drawing_count": round(drawing_total, 1) if drawing_total is not None else None,
            "boq_unit": boq_unit,
            "boq_item_count": len(items),
            "mismatch": mismatch,
            "max_delta": round(abs(delta_pct), 1) if delta_pct is not None else 0,
            "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
            "preferred_source": None,
            "preferred_qty": None,
            "action": "verify_qty" if mismatch else None,
            "evidence_refs": [],
            "notes": (
                f"BOQ {mat_type}: {boq_total:.1f} {boq_unit or ''} from {len(items)} items"
                + (f", drawing: {drawing_total:.1f}" if drawing_total else "")
                + (f", delta: {delta_pct:+.1f}%" if delta_pct is not None else "")
            ),
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
