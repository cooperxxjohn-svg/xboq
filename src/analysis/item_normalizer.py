"""
Item Normaliser — Universal Line Item Builder (Sprint 21).

Combines BOQ items, spec items, and schedule stubs into a single
sorted list of UnifiedLineItem objects, each with:
  - Taxonomy match (via match_boq_text)
  - Unit family resolution
  - Quality flags (unit_inferred, qty_missing, rate_missing)

Usage:
    from .item_normalizer import build_line_items
    line_items = build_line_items(
        boq_items      = extraction_result.boq_items,
        spec_items     = extraction_result.spec_items,
        schedule_stubs = recon.stub_items,
    )
    payload["line_items"] = [li.to_dict() for li in line_items]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup (for standalone use / tests)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.analysis_models import UnifiedLineItem

# Module-level store for dedup stats from the most recent build_line_items() call.
# Thread-unsafe but acceptable for single-threaded pipeline use.
_last_dedup_stats: dict = {}


def get_last_dedup_stats() -> dict:
    """Return dedup stats from the most recent build_line_items(dedup=True) call."""
    return dict(_last_dedup_stats)

# ---------------------------------------------------------------------------
# Unit-family lookup (mirrors matcher.py _UNIT_FAMILIES)
# ---------------------------------------------------------------------------

_FAMILY_MAP = {
    "sqm": "AREA",   "sqft": "AREA",   "m2": "AREA",  "sft": "AREA",
    "sq.m":"AREA",   "sq.ft":"AREA",   "sqmt":"AREA",  "sqmtr":"AREA",
    "cum": "VOLUME", "cft":  "VOLUME", "m3":  "VOLUME","cuft": "VOLUME",
    "rmt": "LINEAR", "m":    "LINEAR", "rm":  "LINEAR", "lm":  "LINEAR",
    "mtr": "LINEAR", "rlm":  "LINEAR", "rft": "LINEAR",
    "nos": "COUNT",  "each": "COUNT",  "no":  "COUNT", "nr":   "COUNT",
    "set": "COUNT",  "unit": "COUNT",  "pcs": "COUNT", "pc":   "COUNT",
    "kg":  "WEIGHT", "mt":   "WEIGHT", "kgs": "WEIGHT","tonne":"WEIGHT",
    "ls":  "LUMP",   "lump": "LUMP",   "job": "LUMP",  "lot":  "LUMP",
}


def _resolve_unit_family(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    return _FAMILY_MAP.get(unit.lower().strip().rstrip("."))


# ---------------------------------------------------------------------------
# Source-specific field extractors
# ---------------------------------------------------------------------------

def _item_from_boq(raw: dict, idx: int) -> UnifiedLineItem:
    desc    = raw.get("description") or raw.get("desc") or ""
    unit    = raw.get("unit") or None
    qty_raw = raw.get("qty")
    rate_raw = raw.get("rate")
    try:
        qty = float(qty_raw) if qty_raw not in (None, "", "N/A") else None
    except (ValueError, TypeError):
        qty = None
    try:
        rate = float(rate_raw) if rate_raw not in (None, "", "N/A") else None
    except (ValueError, TypeError):
        rate = None

    return UnifiedLineItem(
        id           = f"LI-{idx:04d}",
        source       = "boq",
        item_no      = raw.get("item_no") or raw.get("sr_no") or None,
        description  = desc,
        unit         = unit,
        unit_family  = _resolve_unit_family(unit),
        qty          = qty,
        rate         = rate,
        trade        = raw.get("trade") or "general",
        section      = raw.get("section") or None,
        source_page  = int(raw.get("source_page") or 0),
        unit_inferred= False,
        qty_missing  = (qty is None),
        rate_missing = (rate is None),
    )


def _item_from_spec(raw: dict, idx: int) -> UnifiedLineItem:
    desc  = raw.get("description") or ""
    unit  = raw.get("unit") or None
    qty_raw = raw.get("qty")
    try:
        qty = float(qty_raw) if qty_raw not in (None, "", "N/A") else None
    except (ValueError, TypeError):
        qty = None

    return UnifiedLineItem(
        id           = f"LI-{idx:04d}",
        source       = "spec_item",
        item_no      = raw.get("item_no") or None,
        description  = desc,
        unit         = unit,
        unit_family  = _resolve_unit_family(unit),
        qty          = qty,
        rate         = None,
        trade        = raw.get("trade") or "general",
        section      = raw.get("section") or None,
        source_page  = int(raw.get("source_page") or 0),
        unit_inferred= bool(raw.get("unit_inferred", False)),
        qty_missing  = (qty is None),
        rate_missing = True,  # spec items never have rates
    )


def _item_from_stub(raw: dict, idx: int) -> UnifiedLineItem:
    desc = raw.get("description") or ""
    unit = raw.get("unit") or "nos"
    qty_raw = raw.get("qty")
    try:
        qty = float(qty_raw) if qty_raw not in (None, "", "N/A") else None
    except (ValueError, TypeError):
        qty = None

    return UnifiedLineItem(
        id           = f"LI-{idx:04d}",
        source       = "schedule_stub",
        item_no      = raw.get("item_no") or None,
        description  = desc,
        unit         = unit,
        unit_family  = _resolve_unit_family(unit),
        qty          = qty,
        rate         = None,
        trade        = raw.get("trade") or "architectural",
        section      = raw.get("section") or None,
        source_page  = int(raw.get("source_page") or 0),
        unit_inferred= False,
        qty_missing  = (qty is None),
        rate_missing = True,
    )


# ---------------------------------------------------------------------------
# Taxonomy matching
# ---------------------------------------------------------------------------

def _apply_taxonomy(li: UnifiedLineItem) -> UnifiedLineItem:
    """Run match_boq_text and stamp taxonomy fields onto the item in-place."""
    try:
        from src.knowledge_base.matcher import match_boq_text
        result = match_boq_text(
            li.description,
            min_confidence=0.30,
            unit=li.unit or "",
            section=li.section or "",
        )
        if result and result.taxonomy_id:
            li.taxonomy_id         = result.taxonomy_id
            li.taxonomy_discipline = result.discipline
            li.taxonomy_unit       = result.unit
            li.match_confidence    = result.confidence
            li.match_method        = result.match_method
            li.taxonomy_matched    = True
    except Exception:
        pass  # silently skip — taxonomy is advisory
    return li


# ---------------------------------------------------------------------------
# Sort key
# ---------------------------------------------------------------------------

def _sort_key(li: UnifiedLineItem):
    """Sort by (trade, section, item_no) — all falling back to ""."""
    return (
        li.trade or "",
        li.section or "",
        li.item_no or "zz",
    )


# ---------------------------------------------------------------------------
# Source priority (lower = higher priority; BOQ wins over vision)
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY: Dict[str, int] = {
    "boq":           0,
    "spec_item":     1,
    "schedule_stub": 2,
    "visual_detect": 3,
    "vision_count":  3,
    "mep_detect":    4,
    "dw_takeoff":    4,
    "paint_takeoff": 5,
    "wp_takeoff":    5,
    "sitework":      5,
}


def _sig_words(text: str) -> set:
    """Return significant words (≥4 chars) from text, lowercased."""
    return set(re.findall(r"\w{4,}", text.lower()))


def _dedup_unified_items(
    items: List[UnifiedLineItem],
    overlap_threshold: float = 0.50,
) -> Tuple[List[UnifiedLineItem], dict]:
    """
    Remove near-duplicate line items, keeping the higher-priority source.

    Two items are considered duplicates when:
    - They share the same trade, AND
    - Their descriptions share ≥ overlap_threshold fraction of significant words.

    Priority order (lower number wins):
        boq > spec_item > schedule_stub > visual_detect = vision_count >
        mep_detect = dw_takeoff > paint_takeoff = wp_takeoff = sitework

    Returns:
        (deduplicated list, stats dict with "removed" count and "merged_pairs" list)
    """
    final: List[UnifiedLineItem] = []
    removed: List[UnifiedLineItem] = []
    merged_pairs: List[Tuple[str, str]] = []

    for item in items:
        item_words = _sig_words(item.description)
        is_dup = False

        for idx, kept in enumerate(final):
            if kept.trade != item.trade:
                continue
            kept_words = _sig_words(kept.description)
            if not kept_words or not item_words:
                continue

            union = kept_words | item_words
            if not union:
                continue
            overlap = len(item_words & kept_words) / len(union)

            if overlap >= overlap_threshold:
                # Determine which has higher source priority
                item_prio = _SOURCE_PRIORITY.get(item.source, 9)
                kept_prio = _SOURCE_PRIORITY.get(kept.source, 9)

                if item_prio < kept_prio:
                    # Incoming item wins — replace kept
                    merged_pairs.append((item.id, kept.id))
                    removed.append(kept)
                    final[idx] = item
                else:
                    # Kept wins
                    merged_pairs.append((kept.id, item.id))
                    removed.append(item)

                is_dup = True
                break

        if not is_dup:
            final.append(item)

    stats = {
        "removed": len(removed),
        "merged_pairs": [(a, b) for a, b in merged_pairs],
    }
    return final, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _collect_raw(
    boq_items: List[dict],
    spec_items: List[dict],
    schedule_stubs: List[dict],
    min_confidence: float,
) -> tuple:
    """
    Partition raw dicts into (priceable, contractual) lists.

    BOQ items and schedule stubs are always priceable.
    Spec items are split using the ``is_priceable`` flag stamped by
    extract_spec_items._classify_priceable().
    """
    priceable: List[tuple] = []    # (source_tag, raw_dict)
    contractual: List[dict] = []   # raw spec_item dicts that failed priceable check

    for r in (boq_items or []):
        conf = r.get("confidence", 1.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        if conf >= min_confidence:
            priceable.append(("boq", r))

    for r in (spec_items or []):
        conf = r.get("confidence", 0.45)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.45
        if conf < min_confidence:
            continue
        # Route on the is_priceable flag; default True for backward-compat
        if r.get("is_priceable", True):
            priceable.append(("spec_item", r))
        else:
            contractual.append(r)

    for r in (schedule_stubs or []):
        conf = r.get("confidence", 0.40)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.40
        if conf >= min_confidence:
            priceable.append(("schedule_stub", r))

    return priceable, contractual


def build_line_items(
    boq_items: List[dict],
    spec_items: List[dict],
    schedule_stubs: List[dict],
    min_confidence: float = 0.35,
    dedup: bool = True,
) -> List[UnifiedLineItem]:
    """
    Build the unified priceable line-items list from all extraction sources.

    Non-priceable contractual/administrative spec items (security deposits,
    liquidated damages clauses, etc.) are excluded.  Retrieve them separately
    with build_contractual_items().

    Args:
        boq_items:       Raw BOQ item dicts from extraction_result.boq_items.
        spec_items:      Numbered spec clause dicts from extraction_result.spec_items.
        schedule_stubs:  Stub items from ReconciliationResult.stub_items.
        min_confidence:  Minimum item confidence to include (filters noise).
        dedup:           Whether to run near-duplicate removal (default True).

    Returns:
        Sorted list of UnifiedLineItem (priceable only), IDs LI-0001 … LI-NNNN.

    Note:
        The returned list has a ``_dedup_stats`` attribute (a dict) attached
        when dedup=True, accessible as ``getattr(result_list, "_dedup_stats", {})``.
        This is stored on a plain Python list via monkey-patch to avoid breaking
        callers that expect a plain list.
    """
    priceable_raw, _ = _collect_raw(boq_items, spec_items, schedule_stubs, min_confidence)

    _builders = {
        "boq":           _item_from_boq,
        "spec_item":     _item_from_spec,
        "schedule_stub": _item_from_stub,
    }

    unsorted: List[UnifiedLineItem] = []
    for i, (source, raw) in enumerate(priceable_raw):
        li = _builders[source](raw, i + 1)
        li = _apply_taxonomy(li)
        unsorted.append(li)

    unsorted.sort(key=_sort_key)

    global _last_dedup_stats
    dedup_stats: dict = {}
    if dedup:
        unsorted, dedup_stats = _dedup_unified_items(unsorted)
        unsorted.sort(key=_sort_key)
        _last_dedup_stats = dedup_stats

    for seq, li in enumerate(unsorted, start=1):
        li.id = f"LI-{seq:04d}"

    return unsorted


def build_contractual_items(
    spec_items: List[dict],
    min_confidence: float = 0.35,
) -> List[dict]:
    """
    Return spec items classified as non-priceable contractual / administrative
    clauses (security deposits, liquidated damages, refund clauses, etc.).

    These are NOT included in build_line_items() output and should be stored
    separately in payload["contractual_items"].

    Each returned dict is the original spec_item dict enriched with:
        - ``priceable_reason``: the trigger that caused exclusion
    """
    result: List[dict] = []
    for r in (spec_items or []):
        conf = r.get("confidence", 0.45)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.45
        if conf < min_confidence:
            continue
        if not r.get("is_priceable", True):
            result.append(r)
    return result
