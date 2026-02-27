"""
Quantities — build unified quantity list from schedules, BOQ items, and drawing callouts.

Every quantity row includes: item, unit, qty, confidence, source_type, evidence_refs.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional


def _make_bundle_id(source_type: str, item: str, evidence_refs: list) -> str:
    """Deterministic evidence bundle ID from source + item + pages."""
    pages_str = ",".join(
        str(e.get("page", ""))
        for e in sorted(evidence_refs, key=lambda e: e.get("page", 0))
    )
    raw = f"{source_type}:{item}:{pages_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# =============================================================================
# TRADE INFERENCE — map BOQ item_no prefixes to trades
# =============================================================================

_TRADE_PREFIX_MAP = {
    "1": "civil",
    "2": "concrete",
    "3": "masonry",
    "4": "plastering",
    "5": "flooring",
    "6": "doors_windows",
    "7": "painting",
    "8": "plumbing",
    "9": "electrical",
    "10": "waterproofing",
}

_TRADE_KEYWORDS = {
    "excavat": "civil",
    "earth": "civil",
    "foundation": "civil",
    "concrete": "concrete",
    "rcc": "concrete",
    "pcc": "concrete",
    "reinforcement": "concrete",
    "steel": "concrete",
    "brick": "masonry",
    "block": "masonry",
    "masonry": "masonry",
    "wall": "masonry",
    "plaster": "plastering",
    "rendering": "plastering",
    "tile": "flooring",
    "flooring": "flooring",
    "marble": "flooring",
    "granite": "flooring",
    "door": "doors_windows",
    "window": "doors_windows",
    "paint": "painting",
    "whitewash": "painting",
    "primer": "painting",
    "plumb": "plumbing",
    "pipe": "plumbing",
    "sanitary": "plumbing",
    "electric": "electrical",
    "wiring": "electrical",
    "switch": "electrical",
    "waterproof": "waterproofing",
}


def _infer_trade(item_no: str, description: str) -> str:
    """Infer trade from BOQ item number prefix or description keywords."""
    # Try item_no prefix
    if item_no:
        prefix = item_no.split(".")[0].strip()
        if prefix in _TRADE_PREFIX_MAP:
            return _TRADE_PREFIX_MAP[prefix]

    # Try description keywords
    desc_lower = (description or "").lower()
    for keyword, trade in _TRADE_KEYWORDS.items():
        if keyword in desc_lower:
            return trade

    return "general"


# =============================================================================
# SCHEDULE → QUANTITIES
# =============================================================================

def build_quantities_from_schedules(schedules: List[dict]) -> List[dict]:
    """
    Build quantity rows from schedule data (door/window/finish schedules).

    Args:
        schedules: List of schedule row dicts from extract_schedule_rows().
            Expected keys: mark, fields, schedule_type, source_page, has_qty, qty.

    Returns:
        List of quantity row dicts with unified schema.
    """
    if not schedules:
        return []

    # Group by schedule_type → mark
    by_type_mark: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in schedules:
        if not isinstance(row, dict):
            continue
        stype = row.get("schedule_type", "unknown")
        mark = row.get("mark", "")
        if mark:
            by_type_mark[stype][mark].append(row)

    quantities = []
    for stype, marks in by_type_mark.items():
        unit = "nos" if stype in ("door", "window", "fixture") else "sqm"
        for mark, rows in marks.items():
            # Use explicit qty if available, else count occurrences
            explicit_qty = None
            for r in rows:
                if r.get("has_qty") and r.get("qty") is not None:
                    try:
                        explicit_qty = float(r["qty"])
                    except (ValueError, TypeError):
                        pass
                    break

            qty = explicit_qty if explicit_qty is not None else float(len(rows))
            confidence = 0.8 if explicit_qty is not None else 0.5

            # Build description from fields
            fields = rows[0].get("fields", {}) if rows else {}
            desc_parts = [f"{stype.title()} {mark}"]
            for key in ("type", "material", "description", "size"):
                val = fields.get(key)
                if val and str(val).strip():
                    desc_parts.append(str(val).strip()[:40])
            item_desc = " — ".join(desc_parts[:3])

            # Collect evidence pages
            evidence_refs = []
            seen_pages = set()
            for r in rows:
                pg = r.get("source_page")
                if pg is not None and pg not in seen_pages:
                    seen_pages.add(pg)
                    evidence_refs.append({"doc_id": None, "page": pg})

            quantities.append({
                "item": item_desc,
                "unit": unit,
                "qty": qty,
                "confidence": confidence,
                "source_type": "schedule",
                "evidence_refs": evidence_refs,
                "evidence_bundle_id": _make_bundle_id("schedule", item_desc, evidence_refs),
                "derivation": f"schedule:{stype}:{mark}:{'explicit_qty' if explicit_qty is not None else 'occurrence_count'}",
            })

    return quantities


# =============================================================================
# BOQ → QUANTITIES
# =============================================================================

def build_quantities_from_boq(boq_items: List[dict]) -> List[dict]:
    """
    Build quantity rows from BOQ items.

    Args:
        boq_items: List of BOQ item dicts from extract_boq_items().
            Expected keys: item_no, description, unit, qty, rate, source_page, confidence.

    Returns:
        List of quantity row dicts with unified schema + trade/rate/total extras.
    """
    if not boq_items:
        return []

    quantities = []
    for item in boq_items:
        if not isinstance(item, dict):
            continue

        item_no = item.get("item_no", "")
        description = item.get("description", "")
        unit = item.get("unit") or "nos"
        qty = item.get("qty")
        rate = item.get("rate")
        confidence = item.get("confidence", 0.5)
        source_page = item.get("source_page")

        if qty is None:
            continue

        try:
            qty_f = float(qty)
        except (ValueError, TypeError):
            continue

        trade = _infer_trade(item_no, description)
        total = round(qty_f * float(rate), 2) if rate is not None else None

        item_desc = f"{item_no}: {description[:80]}" if item_no else description[:80]

        evidence_refs = []
        if source_page is not None:
            evidence_refs.append({"doc_id": None, "page": source_page})

        quantities.append({
            "item": item_desc,
            "unit": unit,
            "qty": qty_f,
            "confidence": confidence,
            "source_type": "boq",
            "evidence_refs": evidence_refs,
            "trade": trade,
            "rate": rate,
            "total": total,
            "evidence_bundle_id": _make_bundle_id("boq", item_desc, evidence_refs),
            "derivation": f"boq:{item_no}:line_item:{'has_rate' if rate is not None else 'no_rate'}",
        })

    return quantities


# =============================================================================
# CALLOUT → QUANTITIES
# =============================================================================

def build_quantities_from_callouts(
    callouts: List[dict],
    drawing_overview: Optional[dict] = None,
) -> List[dict]:
    """
    Build quantity rows from drawing callouts (tag counts, room counts).

    Args:
        callouts: List of callout dicts from extract_drawing_callouts().
            Expected keys: text, callout_type, source_page, confidence.
        drawing_overview: Optional overview dict with door_tags_found, etc.

    Returns:
        List of quantity row dicts for tag/room counts.
    """
    quantities = []

    # Count from callouts
    door_tags: Dict[str, list] = defaultdict(list)
    window_tags: Dict[str, list] = defaultdict(list)
    room_names: Dict[str, list] = defaultdict(list)

    for co in (callouts or []):
        if not isinstance(co, dict):
            continue
        ctype = co.get("callout_type", "")
        text = co.get("text", "").strip()
        page = co.get("source_page")

        if ctype == "tag" and text:
            text_upper = text.upper()
            if text_upper.startswith(("D", "DR", "DOOR")):
                door_tags[text_upper].append(page)
            elif text_upper.startswith(("W", "WN", "WINDOW")):
                window_tags[text_upper].append(page)
        elif ctype == "room" and text:
            room_names[text.title()].append(page)

    # Fallback to drawing_overview counts if no callout data
    if drawing_overview and isinstance(drawing_overview, dict):
        ov_doors = drawing_overview.get("door_tags_found", 0)
        ov_windows = drawing_overview.get("window_tags_found", 0)
        ov_rooms = drawing_overview.get("room_names_found", 0)

        if not door_tags and ov_doors > 0:
            _item = "Doors (from drawing overview)"
            quantities.append({
                "item": _item,
                "unit": "nos",
                "qty": float(ov_doors),
                "confidence": 0.5,
                "source_type": "callout",
                "evidence_refs": [],
                "evidence_bundle_id": _make_bundle_id("callout", _item, []),
                "derivation": "callout:overview_fallback:doors",
            })

        if not window_tags and ov_windows > 0:
            _item = "Windows (from drawing overview)"
            quantities.append({
                "item": _item,
                "unit": "nos",
                "qty": float(ov_windows),
                "confidence": 0.5,
                "source_type": "callout",
                "evidence_refs": [],
                "evidence_bundle_id": _make_bundle_id("callout", _item, []),
                "derivation": "callout:overview_fallback:windows",
            })

        if not room_names and ov_rooms > 0:
            _item = "Rooms (from drawing overview)"
            quantities.append({
                "item": _item,
                "unit": "nos",
                "qty": float(ov_rooms),
                "confidence": 0.4,
                "source_type": "callout",
                "evidence_refs": [],
                "evidence_bundle_id": _make_bundle_id("callout", _item, []),
                "derivation": "callout:overview_fallback:rooms",
            })

    # Emit per-tag quantities
    if door_tags:
        all_door_pages = set()
        for pages in door_tags.values():
            all_door_pages.update(p for p in pages if p is not None)
        _item = f"Doors ({len(door_tags)} distinct tags: {', '.join(sorted(door_tags.keys())[:5])})"
        _erefs = [{"doc_id": None, "page": p} for p in sorted(all_door_pages)[:10]]
        quantities.append({
            "item": _item,
            "unit": "nos",
            "qty": float(len(door_tags)),
            "confidence": 0.7,
            "source_type": "callout",
            "evidence_refs": _erefs,
            "evidence_bundle_id": _make_bundle_id("callout", _item, _erefs),
            "derivation": "callout:tag_count:doors",
        })

    if window_tags:
        all_win_pages = set()
        for pages in window_tags.values():
            all_win_pages.update(p for p in pages if p is not None)
        _item = f"Windows ({len(window_tags)} distinct tags: {', '.join(sorted(window_tags.keys())[:5])})"
        _erefs = [{"doc_id": None, "page": p} for p in sorted(all_win_pages)[:10]]
        quantities.append({
            "item": _item,
            "unit": "nos",
            "qty": float(len(window_tags)),
            "confidence": 0.7,
            "source_type": "callout",
            "evidence_refs": _erefs,
            "evidence_bundle_id": _make_bundle_id("callout", _item, _erefs),
            "derivation": "callout:tag_count:windows",
        })

    if room_names:
        all_room_pages = set()
        for pages in room_names.values():
            all_room_pages.update(p for p in pages if p is not None)
        _item = f"Rooms ({len(room_names)} detected: {', '.join(sorted(room_names.keys())[:5])})"
        _erefs = [{"doc_id": None, "page": p} for p in sorted(all_room_pages)[:10]]
        quantities.append({
            "item": _item,
            "unit": "nos",
            "qty": float(len(room_names)),
            "confidence": 0.6,
            "source_type": "callout",
            "evidence_refs": _erefs,
            "evidence_bundle_id": _make_bundle_id("callout", _item, _erefs),
            "derivation": "callout:tag_count:rooms",
        })

    return quantities


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def build_all_quantities(
    schedules: List[dict],
    boq_items: List[dict],
    callouts: List[dict],
    drawing_overview: Optional[dict] = None,
) -> List[dict]:
    """
    Build unified quantity list from all three sources.

    Merges schedule, BOQ, and callout quantities, deduplicates overlapping
    door/window counts between schedules and callouts.

    Returns:
        Sorted list of quantity row dicts.
    """
    sched_qtys = build_quantities_from_schedules(schedules or [])
    boq_qtys = build_quantities_from_boq(boq_items or [])
    callout_qtys = build_quantities_from_callouts(callouts or [], drawing_overview)

    # Deduplication: if schedule already has door/window items, skip callout door/window
    has_schedule_doors = any(
        "door" in q.get("item", "").lower() and q.get("source_type") == "schedule"
        for q in sched_qtys
    )
    has_schedule_windows = any(
        "window" in q.get("item", "").lower() and q.get("source_type") == "schedule"
        for q in sched_qtys
    )

    filtered_callout = []
    for q in callout_qtys:
        item_lower = q.get("item", "").lower()
        if has_schedule_doors and "door" in item_lower:
            continue  # skip — schedule is more authoritative
        if has_schedule_windows and "window" in item_lower:
            continue
        filtered_callout.append(q)

    all_qtys = sched_qtys + boq_qtys + filtered_callout

    # Sort: BOQ first (most authoritative), then schedule, then callout, stable tiebreaker
    from src.analysis.determinism import stable_sort_key
    source_order = {"boq": 0, "schedule": 1, "callout": 2}
    all_qtys.sort(key=lambda q: (
        source_order.get(q.get("source_type", ""), 9),
        q.get("item", ""),
        stable_sort_key(q, ("evidence_bundle_id",)),
    ))

    return all_qtys
