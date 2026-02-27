"""
Delta Detection — compare base vs addendum-sourced extracted items.

Detects changes in BOQ items, schedule rows, and requirements
between base documents and addendum documents.

All conflict records include a `delta_confidence` field (0.0-1.0).

All functions are pure (no Streamlit, no I/O).
"""

from typing import List, Dict, Any, Optional, Union
from difflib import SequenceMatcher


def detect_boq_deltas(
    base_items: List[dict],
    addendum_items: List[dict],
) -> List[dict]:
    """
    Compare BOQ items by item_no. Detect qty/unit/rate/description changes.

    Args:
        base_items: BOQ items from non-addendum pages.
        addendum_items: BOQ items from addendum pages.

    Returns:
        List of conflict dicts:
        - {type: "boq_change", item_no, changes: [{field, base_value, addendum_value}],
           base_page, addendum_page, delta_confidence}
        - {type: "boq_new_item", item_no, description, addendum_page, delta_confidence}
    """
    conflicts: List[dict] = []
    base_by_no: Dict[str, dict] = {}
    for item in base_items:
        item_no = item.get("item_no")
        if item_no:
            base_by_no[item_no] = item

    for add_item in addendum_items:
        item_no = add_item.get("item_no")
        if not item_no:
            continue

        if item_no in base_by_no:
            base = base_by_no[item_no]
            changes = []
            for field_name in ("qty", "unit", "rate", "description"):
                bv = base.get(field_name)
                av = add_item.get(field_name)
                if bv is not None and av is not None and bv != av:
                    changes.append({
                        "field": field_name,
                        "base_value": bv,
                        "addendum_value": av,
                    })
            if changes:
                conflicts.append({
                    "type": "boq_change",
                    "item_no": item_no,
                    "changes": changes,
                    "base_page": base.get("source_page"),
                    "addendum_page": add_item.get("source_page"),
                    "delta_confidence": 0.9,
                })
        else:
            # New item in addendum not found in base
            conflicts.append({
                "type": "boq_new_item",
                "item_no": item_no,
                "description": add_item.get("description", ""),
                "addendum_page": add_item.get("source_page"),
                "delta_confidence": 0.8,
            })

    return conflicts


def detect_schedule_deltas(
    base_rows: List[dict],
    addendum_rows: List[dict],
) -> List[dict]:
    """
    Compare schedule rows by mark. Detect size/qty/field changes.

    Args:
        base_rows: Schedule rows from non-addendum pages.
        addendum_rows: Schedule rows from addendum pages.

    Returns:
        List of conflict dicts:
        - {type: "schedule_change", mark, changes: [...], base_page, addendum_page,
           delta_confidence}
    """
    conflicts: List[dict] = []
    base_by_mark: Dict[str, dict] = {}
    for row in base_rows:
        mark = row.get("mark")
        if mark:
            base_by_mark[mark] = row

    for add_row in addendum_rows:
        mark = add_row.get("mark")
        if not mark:
            continue

        if mark in base_by_mark:
            base = base_by_mark[mark]
            changes = []

            # Compare top-level fields
            for field_name in ("size", "qty"):
                bv = base.get(field_name)
                av = add_row.get(field_name)
                if bv is not None and av is not None and bv != av:
                    changes.append({
                        "field": field_name,
                        "base_value": bv,
                        "addendum_value": av,
                    })

            # Compare nested fields dicts (schedule rows may store data there)
            bf = base.get("fields", {}) or {}
            af = add_row.get("fields", {}) or {}
            all_keys = set(list(bf.keys()) + list(af.keys()))
            for k in sorted(all_keys):
                bv = bf.get(k)
                av = af.get(k)
                if bv is not None and av is not None and bv != av:
                    changes.append({
                        "field": k,
                        "base_value": bv,
                        "addendum_value": av,
                    })

            if changes:
                conflicts.append({
                    "type": "schedule_change",
                    "mark": mark,
                    "changes": changes,
                    "base_page": base.get("source_page"),
                    "addendum_page": add_row.get("source_page"),
                    "delta_confidence": 0.9,
                })

    return conflicts


def detect_requirement_deltas(
    base_reqs: List[dict],
    addendum_reqs: List[dict],
    similarity_threshold: float = 0.7,
    ocr_coverage_pct: Optional[float] = None,
) -> List[dict]:
    """
    Detect new/modified requirements by text similarity.

    Uses difflib.SequenceMatcher for fuzzy matching on pre-normalized text.
    - ratio >= 0.95: considered identical (no conflict)
    - 0.7 <= ratio < 0.95: modified requirement
    - ratio < 0.7: new requirement

    Confidence scoring:
    - requirement_modified: similarity * coverage_factor
    - requirement_new: 0.7 (always somewhat uncertain)
    - coverage_factor: min(1.0, ocr_coverage_pct / 80.0) if available, else 0.85

    Args:
        base_reqs: Requirements from non-addendum pages.
        addendum_reqs: Requirements from addendum pages.
        similarity_threshold: Minimum ratio to consider a match.
        ocr_coverage_pct: Optional OCR page coverage percentage (0-100)
            for confidence downgrading.

    Returns:
        List of conflict dicts:
        - {type: "requirement_modified", base_text, addendum_text, similarity,
           base_page, addendum_page, delta_confidence}
        - {type: "requirement_new", text, addendum_page, delta_confidence}
    """
    from .normalize import normalize_requirement_text

    conflicts: List[dict] = []

    # Compute coverage factor for confidence scoring
    if ocr_coverage_pct is not None and ocr_coverage_pct > 0:
        coverage_factor = min(1.0, ocr_coverage_pct / 80.0)
    else:
        coverage_factor = 0.85  # assume moderate coverage

    # Pre-normalize base texts for comparison
    base_normalized = [normalize_requirement_text(r.get("text", "")) for r in base_reqs]

    for add_req in addendum_reqs:
        add_norm = normalize_requirement_text(add_req.get("text", ""))
        if not add_norm:
            continue

        # Find best matching base requirement
        best_ratio = 0.0
        best_idx = -1
        for i, bt in enumerate(base_normalized):
            if not bt:
                continue
            ratio = SequenceMatcher(None, bt, add_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_ratio >= similarity_threshold and best_idx >= 0:
            if best_ratio < 0.95:
                # Modified requirement (similar but not identical)
                conf = round(best_ratio * coverage_factor, 2)
                conflicts.append({
                    "type": "requirement_modified",
                    "base_text": base_reqs[best_idx].get("text", ""),
                    "addendum_text": add_req.get("text", ""),
                    "similarity": round(best_ratio, 2),
                    "base_page": base_reqs[best_idx].get("source_page"),
                    "addendum_page": add_req.get("source_page"),
                    "delta_confidence": conf,
                })
            # else: ratio >= 0.95 means identical — no conflict
        else:
            # New requirement not found in base
            conflicts.append({
                "type": "requirement_new",
                "text": add_req.get("text", ""),
                "addendum_page": add_req.get("source_page"),
                "delta_confidence": 0.7,
            })

    return conflicts


# =============================================================================
# SUPERSEDE RESOLUTION TAGGING (Sprint 9)
# =============================================================================

def tag_conflicts_with_resolution(
    conflicts: List[dict],
    ocr_text_by_page: Dict[Union[int, str], str],
) -> List[dict]:
    """
    Post-process conflicts to add 'resolution' field from supersede detection.

    Delegates to supersedes_detector.tag_conflicts_with_supersedes().

    Args:
        conflicts: List of conflict dicts with 'addendum_page' key.
        ocr_text_by_page: Full OCR text cache (page_idx -> text).

    Returns:
        New list with 'resolution' field on each conflict.
    """
    from .supersedes_detector import tag_conflicts_with_supersedes

    # Build addendum page texts subset
    addendum_pages = set()
    for c in conflicts:
        ap = c.get("addendum_page")
        if ap is not None:
            addendum_pages.add(ap)

    addendum_page_texts = {}
    for pg in addendum_pages:
        text = ocr_text_by_page.get(pg, "")
        if not text:
            text = ocr_text_by_page.get(str(pg), "")
        if text:
            addendum_page_texts[pg] = text

    return tag_conflicts_with_supersedes(conflicts, addendum_page_texts)
