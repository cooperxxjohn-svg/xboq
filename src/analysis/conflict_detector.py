"""
Conflict Detector — Detect changes between original BOQ items and addendum items.

When an addendum modifies the original tender BOQ (changing quantities, rates,
descriptions, adding or deleting items), this module surfaces those changes as
structured Conflict objects for review.
"""

import re
from dataclasses import dataclass, asdict
from typing import Any, List, Optional


@dataclass
class BOQConflict:
    conflict_type: str    # "qty_change" | "rate_change" | "description_change" | "unit_change" | "new_item" | "deleted_item" | "multiple_changes"
    item_no: str
    description: str       # original description (or addendum if new)
    original_value: Any    # None for new_item
    addendum_value: Any    # None for deleted_item
    field_changed: str     # "qty" | "rate" | "description" | "unit" | "item"
    severity: str          # "critical" | "high" | "medium" | "low"
    change_pct: Optional[float]  # % change for numeric fields (None for text)
    source_page: int       # addendum source page

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def _qty_severity(change_pct: Optional[float]) -> str:
    if change_pct is None:
        return "medium"
    if change_pct > 20:
        return "high"
    if change_pct >= 5:
        return "medium"
    return "low"


def _rate_severity(change_pct: Optional[float]) -> str:
    if change_pct is None:
        return "medium"
    if change_pct > 15:
        return "critical"
    if change_pct >= 5:
        return "high"
    return "medium"


_SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _highest_severity(severities: List[str]) -> str:
    if not severities:
        return "low"
    return max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if not possible or effectively zero/missing."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if f != 0.0 else None
    except (TypeError, ValueError):
        return None


def _change_pct(old: Any, new: Any) -> Optional[float]:
    old_f = _to_float(old)
    new_f = _to_float(new)
    if old_f is None or new_f is None:
        return None
    return abs(new_f - old_f) / old_f * 100


# ---------------------------------------------------------------------------
# Text normalisation for fuzzy matching
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two normalised strings."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Field-level change detection
# ---------------------------------------------------------------------------

def _check_fields(orig: dict, addn: dict, item_no: str, source_page: int) -> List[BOQConflict]:
    """Return a list of BOQConflict objects for each field that changed."""
    conflicts: List[BOQConflict] = []
    description = orig.get("description") or addn.get("description") or ""

    # --- qty ---
    orig_qty = orig.get("qty")
    addn_qty = addn.get("qty")
    orig_qty_f = _to_float(orig_qty)
    addn_qty_f = _to_float(addn_qty)
    if orig_qty_f is not None and addn_qty_f is not None and orig_qty_f != addn_qty_f:
        pct = _change_pct(orig_qty_f, addn_qty_f)
        conflicts.append(BOQConflict(
            conflict_type="qty_change",
            item_no=item_no,
            description=description,
            original_value=orig_qty,
            addendum_value=addn_qty,
            field_changed="qty",
            severity=_qty_severity(pct),
            change_pct=pct,
            source_page=source_page,
        ))
    elif orig_qty_f is None and addn_qty_f is not None:
        # None → value
        conflicts.append(BOQConflict(
            conflict_type="qty_change",
            item_no=item_no,
            description=description,
            original_value=orig_qty,
            addendum_value=addn_qty,
            field_changed="qty",
            severity="medium",
            change_pct=None,
            source_page=source_page,
        ))

    # --- rate ---
    orig_rate = orig.get("rate")
    addn_rate = addn.get("rate")
    orig_rate_f = _to_float(orig_rate)
    addn_rate_f = _to_float(addn_rate)
    if orig_rate_f is not None and addn_rate_f is not None and orig_rate_f != addn_rate_f:
        pct = _change_pct(orig_rate_f, addn_rate_f)
        conflicts.append(BOQConflict(
            conflict_type="rate_change",
            item_no=item_no,
            description=description,
            original_value=orig_rate,
            addendum_value=addn_rate,
            field_changed="rate",
            severity=_rate_severity(pct),
            change_pct=pct,
            source_page=source_page,
        ))

    # --- unit ---
    orig_unit = (orig.get("unit") or "").strip().lower()
    addn_unit = (addn.get("unit") or "").strip().lower()
    if orig_unit and addn_unit and orig_unit != addn_unit:
        conflicts.append(BOQConflict(
            conflict_type="unit_change",
            item_no=item_no,
            description=description,
            original_value=orig.get("unit"),
            addendum_value=addn.get("unit"),
            field_changed="unit",
            severity="high",
            change_pct=None,
            source_page=source_page,
        ))

    # --- description ---
    orig_desc = _normalise(orig.get("description") or "")
    addn_desc = _normalise(addn.get("description") or "")
    if orig_desc and addn_desc and orig_desc != addn_desc:
        conflicts.append(BOQConflict(
            conflict_type="description_change",
            item_no=item_no,
            description=orig.get("description") or "",
            original_value=orig.get("description"),
            addendum_value=addn.get("description"),
            field_changed="description",
            severity="medium",
            change_pct=None,
            source_page=source_page,
        ))

    return conflicts


def _merge_field_conflicts(field_conflicts: List[BOQConflict], item_no: str, source_page: int) -> List[BOQConflict]:
    """If multiple fields changed on same item, emit a single multiple_changes conflict."""
    if not field_conflicts:
        return []
    if len(field_conflicts) == 1:
        return field_conflicts
    # Multiple changes — emit one "multiple_changes" conflict plus keep the individual ones
    severities = [c.severity for c in field_conflicts]
    best = _highest_severity(severities)
    multi = BOQConflict(
        conflict_type="multiple_changes",
        item_no=item_no,
        description=field_conflicts[0].description,
        original_value=None,
        addendum_value=None,
        field_changed="item",
        severity=best,
        change_pct=None,
        source_page=source_page,
    )
    return [multi] + field_conflicts


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def detect_conflicts(
    original_items: List[dict],
    addendum_items: List[dict],
    match_threshold: float = 0.75,
) -> List[BOQConflict]:
    """
    Compare original BOQ items to addendum BOQ items and return a list of
    BOQConflict objects describing each detected change.

    Matching strategy:
    1. Exact match on item_no (stripped, uppercase).
    2. Fuzzy description match using token-Jaccard overlap for unmatched items.
    3. Unmatched addendum items → new_item conflicts.
    4. Unmatched original items → deleted_item conflicts.
    """
    conflicts: List[BOQConflict] = []

    def _norm_item_no(item: dict) -> str:
        raw = item.get("item_no") or item.get("item_number") or ""
        return str(raw).strip().upper()

    # Build lookup by item_no for originals
    orig_by_no: dict[str, dict] = {}
    for item in original_items:
        key = _norm_item_no(item)
        if key:
            orig_by_no[key] = item

    addn_by_no: dict[str, dict] = {}
    for item in addendum_items:
        key = _norm_item_no(item)
        if key:
            addn_by_no[key] = item

    matched_orig_keys: set[str] = set()
    matched_addn_keys: set[str] = set()

    # --- Pass 1: exact item_no match ---
    for addn_key, addn_item in addn_by_no.items():
        if addn_key in orig_by_no:
            orig_item = orig_by_no[addn_key]
            matched_orig_keys.add(addn_key)
            matched_addn_keys.add(addn_key)
            source_page = addn_item.get("source_page") or addn_item.get("page") or 0
            field_conflicts = _check_fields(orig_item, addn_item, addn_key, source_page)
            conflicts.extend(_merge_field_conflicts(field_conflicts, addn_key, source_page))

    # Items with no item_no in either list — collect for fuzzy matching
    orig_unkeyed = [item for item in original_items if not _norm_item_no(item)]
    addn_unkeyed = [item for item in addendum_items if not _norm_item_no(item)]

    # Also collect items that had a key but were not matched
    orig_unmatched_keyed = [
        item for item in original_items
        if _norm_item_no(item) and _norm_item_no(item) not in matched_orig_keys
    ]
    addn_unmatched_keyed = [
        item for item in addendum_items
        if _norm_item_no(item) and _norm_item_no(item) not in matched_addn_keys
    ]

    orig_for_fuzzy = orig_unkeyed + orig_unmatched_keyed
    addn_for_fuzzy = addn_unkeyed + addn_unmatched_keyed

    # --- Pass 2: fuzzy description match ---
    fuzzy_matched_orig_indices: set[int] = set()
    fuzzy_matched_addn_indices: set[int] = set()

    for ai, addn_item in enumerate(addn_for_fuzzy):
        addn_desc_norm = _normalise(addn_item.get("description") or "")
        best_score = 0.0
        best_oi = -1
        for oi, orig_item in enumerate(orig_for_fuzzy):
            if oi in fuzzy_matched_orig_indices:
                continue
            orig_desc_norm = _normalise(orig_item.get("description") or "")
            score = _token_overlap(orig_desc_norm, addn_desc_norm)
            if score > best_score:
                best_score = score
                best_oi = oi

        if best_score >= match_threshold and best_oi >= 0:
            fuzzy_matched_orig_indices.add(best_oi)
            fuzzy_matched_addn_indices.add(ai)
            orig_item = orig_for_fuzzy[best_oi]
            item_no = _norm_item_no(addn_item) or _norm_item_no(orig_item) or f"FUZZY_{ai}"
            source_page = addn_item.get("source_page") or addn_item.get("page") or 0
            field_conflicts = _check_fields(orig_item, addn_item, item_no, source_page)
            conflicts.extend(_merge_field_conflicts(field_conflicts, item_no, source_page))

    # --- New items (addendum items with no original match) ---
    for ai, addn_item in enumerate(addn_for_fuzzy):
        if ai in fuzzy_matched_addn_indices:
            continue
        item_no = _norm_item_no(addn_item) or f"NEW_{ai}"
        description = addn_item.get("description") or ""
        source_page = addn_item.get("source_page") or addn_item.get("page") or 0
        conflicts.append(BOQConflict(
            conflict_type="new_item",
            item_no=item_no,
            description=description,
            original_value=None,
            addendum_value=addn_item,
            field_changed="item",
            severity="high",
            change_pct=None,
            source_page=source_page,
        ))

    # Also treat unmatched addendum keyed items that were not matched in pass 1 — already in addn_for_fuzzy
    # (handled above)

    # --- Deleted items (original items with no addendum match) ---
    for oi, orig_item in enumerate(orig_for_fuzzy):
        if oi in fuzzy_matched_orig_indices:
            continue
        item_no = _norm_item_no(orig_item) or f"DEL_{oi}"
        description = orig_item.get("description") or ""
        source_page = 0  # deleted items have no addendum page
        conflicts.append(BOQConflict(
            conflict_type="deleted_item",
            item_no=item_no,
            description=description,
            original_value=orig_item,
            addendum_value=None,
            field_changed="item",
            severity="critical",
            change_pct=None,
            source_page=source_page,
        ))

    return conflicts


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_conflicts(conflicts: List[BOQConflict]) -> dict:
    """Return summary stats: total, by_type, by_severity, critical_count."""
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for c in conflicts:
        by_type[c.conflict_type] = by_type.get(c.conflict_type, 0) + 1
        by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
    return {
        "total": len(conflicts),
        "by_type": by_type,
        "by_severity": by_severity,
        "critical_count": by_severity.get("critical", 0),
    }


# ---------------------------------------------------------------------------
# Pipeline payload helper
# ---------------------------------------------------------------------------

def conflicts_from_payload(payload: dict) -> List[BOQConflict]:
    """
    Extract conflicts from a pipeline payload.
    Uses payload["extraction_summary"]["boq_items"] as original
    and payload["conflicts"] (pre-populated from addendum extraction) as addendum items.
    Falls back gracefully if keys missing.
    """
    original_items: List[dict] = []
    addendum_items: List[dict] = []

    try:
        extraction_summary = payload.get("extraction_summary") or {}
        original_items = extraction_summary.get("boq_items") or []
        if not isinstance(original_items, list):
            original_items = []
    except Exception:
        original_items = []

    try:
        addendum_items = payload.get("conflicts") or []
        if not isinstance(addendum_items, list):
            addendum_items = []
    except Exception:
        addendum_items = []

    return detect_conflicts(original_items, addendum_items)
