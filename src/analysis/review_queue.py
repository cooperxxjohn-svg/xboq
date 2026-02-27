"""
Review Queue — unified list of review items from all analysis sources.

Consolidates:
  1. quantity_reconciliation mismatches
  2. conflicts with delta_confidence < threshold
  3. pages_skipped (unknown/not processed)
  4. toxic pages
  5. risk checklist hits (HIGH impact)

Pure module, no Streamlit dependency. Can be tested independently.
"""

from typing import Any, Dict, List, Optional


# ─── Constants ────────────────────────────────────────────────────────────

REVIEW_TYPE_RECON_MISMATCH = "recon_mismatch"
REVIEW_TYPE_CONFLICT = "conflict"
REVIEW_TYPE_SKIPPED_PAGE = "skipped_page"
REVIEW_TYPE_TOXIC_PAGE = "toxic_page"
REVIEW_TYPE_RISK_HIT = "risk_hit"

# Severity sort order (lower = higher priority)
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Type priority for tie-breaking within same severity
TYPE_ORDER = {
    REVIEW_TYPE_RISK_HIT: 0,
    REVIEW_TYPE_RECON_MISMATCH: 1,
    REVIEW_TYPE_CONFLICT: 2,
    REVIEW_TYPE_TOXIC_PAGE: 3,
    REVIEW_TYPE_SKIPPED_PAGE: 4,
}

# Doc types that indicate meaningful skipped content
_IMPORTANT_DOC_TYPES = {"boq", "schedule", "conditions", "spec"}


# ─── Source Builders ──────────────────────────────────────────────────────

def _severity_for_recon_mismatch(recon_row: dict) -> str:
    """Map reconciliation mismatch to severity based on max_delta."""
    delta = recon_row.get("max_delta", 0)
    if delta >= 5:
        return "high"
    elif delta >= 2:
        return "medium"
    return "low"


def _build_recon_items(recon_rows: List[dict]) -> List[dict]:
    """Convert quantity_reconciliation mismatches to review items."""
    items = []
    for row in recon_rows:
        if not row.get("mismatch"):
            continue
        severity = _severity_for_recon_mismatch(row)
        cat = row.get("category", "unknown")
        items.append({
            "type": REVIEW_TYPE_RECON_MISMATCH,
            "severity": severity,
            "title": f"{cat.title()} quantity mismatch (delta: {row.get('max_delta', 0)})",
            "doc_refs": [],
            "page_refs": [],
            "evidence_bundle": {
                "category": cat,
                "schedule_count": row.get("schedule_count"),
                "boq_count": row.get("boq_count"),
                "drawing_count": row.get("drawing_count"),
                "max_delta": row.get("max_delta", 0),
            },
            "recommended_action": f"prefer_schedule_for_{cat}",
            "source_key": f"recon:{cat}",
        })
    return items


def _build_conflict_items(
    conflicts: List[dict],
    confidence_threshold: float = 0.85,
) -> List[dict]:
    """Convert conflicts with delta_confidence < threshold to review items."""
    items = []
    for c in conflicts:
        dc = c.get("delta_confidence", 1.0)
        if dc >= confidence_threshold:
            continue
        # Skip already-resolved intentional revisions
        if c.get("resolution") == "intentional_revision":
            continue
        ctype = c.get("type", "unknown")
        severity = "high" if dc < 0.7 else "medium"
        # Build title from available identifiers
        title_parts = []
        if c.get("item_no"):
            title_parts.append(f"Item {c['item_no']}")
        if c.get("mark"):
            title_parts.append(f"Mark {c['mark']}")
        title_parts.append(f"{ctype.replace('_', ' ')} (confidence: {dc:.0%})")
        page_refs = [p for p in [c.get("base_page"), c.get("addendum_page")] if p is not None]
        items.append({
            "type": REVIEW_TYPE_CONFLICT,
            "severity": severity,
            "title": " ".join(title_parts),
            "doc_refs": [],
            "page_refs": page_refs,
            "evidence_bundle": {
                k: v for k, v in c.items()
                if k in ("type", "item_no", "mark", "delta_confidence",
                         "base_page", "addendum_page", "changes", "description",
                         "base_text", "addendum_text", "similarity", "text")
            },
            "recommended_action": "review_conflict",
            "source_key": f"conflict:{ctype}:{c.get('item_no', c.get('mark', ''))}",
        })
    return items


def _build_skipped_page_items(pages_skipped: List[dict]) -> List[dict]:
    """Convert skipped pages to review items, grouped by doc_type."""
    if not pages_skipped:
        return []
    by_type: Dict[str, List[dict]] = {}
    for ps in pages_skipped:
        dt = ps.get("doc_type", "unknown")
        by_type.setdefault(dt, []).append(ps)

    items = []
    for dt in sorted(by_type.keys()):  # sorted for determinism
        page_list = by_type[dt]
        page_idxs = sorted(
            p.get("page_idx") for p in page_list if p.get("page_idx") is not None
        )
        severity = "medium" if dt in _IMPORTANT_DOC_TYPES else "low"
        items.append({
            "type": REVIEW_TYPE_SKIPPED_PAGE,
            "severity": severity,
            "title": f"{len(page_list)} {dt} page(s) skipped (not processed)",
            "doc_refs": [],
            "page_refs": page_idxs[:10],
            "evidence_bundle": {
                "doc_type": dt,
                "count": len(page_list),
                "pages": page_idxs,
            },
            "recommended_action": "acknowledge_skipped",
            "source_key": f"skipped:{dt}",
        })
    return items


def _build_toxic_page_items(toxic_summary: Optional[dict]) -> List[dict]:
    """Convert toxic (unrecoverable) pages to review items."""
    if not toxic_summary:
        return []
    toxic_pages = [p for p in toxic_summary.get("pages", []) if p.get("toxic")]
    if not toxic_pages:
        return []
    page_idxs = sorted(p.get("page_idx") for p in toxic_pages if p.get("page_idx") is not None)
    return [{
        "type": REVIEW_TYPE_TOXIC_PAGE,
        "severity": "medium",
        "title": f"{len(toxic_pages)} page(s) failed OCR (toxic)",
        "doc_refs": [],
        "page_refs": page_idxs,
        "evidence_bundle": {
            "toxic_count": len(toxic_pages),
            "pages": [
                {"page_idx": p.get("page_idx"), "reason": p.get("reason", "")}
                for p in toxic_pages
            ],
        },
        "recommended_action": "acknowledge_toxic",
        "source_key": "toxic_pages",
    }]


def _build_risk_hit_items(risk_results: List[dict]) -> List[dict]:
    """Convert HIGH-impact risk checklist hits to review items."""
    items = []
    for r in risk_results:
        if r.get("impact") != "high" or not r.get("found"):
            continue
        hits = r.get("hits", [])
        page_refs = sorted(
            h.get("page_idx") for h in hits if h.get("page_idx") is not None
        )
        items.append({
            "type": REVIEW_TYPE_RISK_HIT,
            "severity": "high",
            "title": f"Risk: {r.get('label', r.get('template_id', 'Unknown'))}",
            "doc_refs": [],
            "page_refs": page_refs[:10],
            "evidence_bundle": {
                "template_id": r.get("template_id"),
                "label": r.get("label"),
                "hit_count": len(hits),
                "hits": hits[:5],  # cap for payload size
            },
            "recommended_action": "review_risk_clause",
            "source_key": f"risk:{r.get('template_id', '')}",
        })
    return items


# ─── Main Builder ─────────────────────────────────────────────────────────

def build_review_queue(
    quantity_reconciliation: List[dict] = None,
    conflicts: List[dict] = None,
    pages_skipped: List[dict] = None,
    toxic_summary: Optional[dict] = None,
    risk_results: List[dict] = None,
    confidence_threshold: float = 0.85,
) -> List[dict]:
    """
    Build a unified, deterministically ordered review queue.

    Sort order: severity (high > medium > low), then type priority
    (risk > recon > conflict > toxic > skipped), then title alphabetically.

    Args:
        quantity_reconciliation: Reconciliation rows from reconcile_quantities().
        conflicts: Conflict dicts from delta_detector.
        pages_skipped: Skipped page dicts from run_coverage.
        toxic_summary: Toxic pages summary from toxic_pages module.
        risk_results: Risk checklist results from search_risk_items().
        confidence_threshold: Conflicts with delta_confidence below this are included.

    Returns:
        List of review item dicts, deterministically sorted.
    """
    items: List[dict] = []
    items.extend(_build_recon_items(quantity_reconciliation or []))
    items.extend(_build_conflict_items(conflicts or [], confidence_threshold))
    items.extend(_build_skipped_page_items(pages_skipped or []))
    items.extend(_build_toxic_page_items(toxic_summary))
    items.extend(_build_risk_hit_items(risk_results or []))

    # Deterministic sort: severity, type priority, title, stable tiebreaker
    from src.analysis.determinism import stable_sort_key
    items.sort(key=lambda x: (
        SEVERITY_ORDER.get(x.get("severity", "low"), 9),
        TYPE_ORDER.get(x.get("type", ""), 9),
        x.get("title", ""),
        stable_sort_key(x, ("source_key",)),
    ))

    return items
