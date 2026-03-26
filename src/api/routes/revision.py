"""
POST /api/revision/diff  — Compare two completed analysis jobs.

Uses the existing SheetComparator (src/revision/comparator.py) to detect
added/removed/modified pages between two pipeline runs of the same tender.

Request body (JSON):
    {"job_id_a": "...", "job_id_b": "..."}
    job_id_a = original run   (baseline)
    job_id_b = revised run    (new version)

Response: ChangeReport dict with:
    added_sheet_ids, removed_sheet_ids, changed_sheet_ids,
    unchanged_sheet_ids, total_changes, geometry_changes_count,
    text_changes_count, revision_only_count, diffs[]

Each diff has:
    sheet_id, change_type, summary, geometry_changes[],
    text_changes[], dimension_changes[], added_elements[],
    removed_elements[], old_hash, new_hash, confidence

GET /api/revision/boq-diff/{job_id_a}/{job_id_b}
    Compare BOQ items between two runs. Returns:
    {added: [...], removed: [...], changed: [...], unchanged_count: N}
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["revision"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class DiffRequest(BaseModel):
    job_id_a: str   # baseline (original)
    job_id_b: str   # revised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_complete_payload(job_id: str) -> dict:
    """Return payload for a completed job or raise HTTPException."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=422,
            detail=f"Job '{job_id}' is not complete (status: {job.status}). "
                   "Both jobs must be complete to compute a diff.",
        )
    if not job.payload:
        raise HTTPException(status_code=422, detail=f"Job '{job_id}' has no payload")
    return job.payload


def _payload_to_pages(payload: dict) -> list[dict]:
    """Extract page-level records from a payload for SheetComparator."""
    # page_index is stored as a list of IndexedPage dicts
    pages = payload.get("page_index", [])
    if not pages:
        # Fall back to processing_stats-derived stub list
        total = payload.get("processing_stats", {}).get("total_pages", 0)
        pages = [{"page_number": i, "file_path": "unknown"} for i in range(total)]
    return pages if isinstance(pages, list) else []


def _payload_to_results(payload: dict) -> list[dict]:
    """Extract extraction results compatible with SheetComparator._index_by_sheet()."""
    results = []

    # plan_graph.rooms gives room-level data
    plan_graph = payload.get("plan_graph", {})
    rooms = plan_graph.get("rooms", []) if isinstance(plan_graph, dict) else []

    # Build page-level records from available data
    page_index = payload.get("page_index", [])
    for page in page_index:
        if not isinstance(page, dict):
            continue
        page_num = page.get("page_number", 0)
        file_path = page.get("file_path", "")
        results.append({
            "file_path":   file_path,
            "page_number": page_num,
            "rooms":       [r for r in rooms if r.get("source_page") == page_num],
            "dimensions":  page.get("dimensions", []),
            "text_items":  [{"text": t} for t in page.get("text_snippets", [])],
            "openings":    page.get("openings", []),
        })

    return results


def _diff_boq_items(items_a: list[dict], items_b: list[dict]) -> dict:
    """Identify added / removed / changed BOQ items between two runs."""

    def _key(item: dict) -> str:
        return f"{item.get('trade', '')}::{item.get('description', '')[:80]}"

    by_key_a = {_key(i): i for i in items_a}
    by_key_b = {_key(i): i for i in items_b}

    keys_a = set(by_key_a)
    keys_b = set(by_key_b)

    added   = [by_key_b[k] for k in (keys_b - keys_a)]
    removed = [by_key_a[k] for k in (keys_a - keys_b)]

    changed = []
    for k in keys_a & keys_b:
        ia, ib = by_key_a[k], by_key_b[k]
        qty_a = ia.get("quantity") or ia.get("qty")
        qty_b = ib.get("quantity") or ib.get("qty")
        rate_a = ia.get("rate") or ia.get("rate_inr")
        rate_b = ib.get("rate") or ib.get("rate_inr")
        if qty_a != qty_b or rate_a != rate_b:
            changed.append({
                "description": ib.get("description", k),
                "trade":       ib.get("trade", ""),
                "qty_before":  qty_a,
                "qty_after":   qty_b,
                "rate_before": rate_a,
                "rate_after":  rate_b,
            })

    return {
        "added":           added,
        "removed":         removed,
        "changed":         changed,
        "unchanged_count": len(keys_a & keys_b) - len(changed),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/api/revision/diff")
def revision_diff(req: DiffRequest) -> JSONResponse:
    """
    Compare drawing sets between two completed analysis jobs.

    Returns a ChangeReport with sheet-level diffs (added/removed/modified pages,
    geometry changes, text changes).
    """
    payload_a = _get_complete_payload(req.job_id_a)
    payload_b = _get_complete_payload(req.job_id_b)

    try:
        from src.revision.comparator import SheetComparator
        comparator = SheetComparator()
        report = comparator.compare_sets(
            current_pages=_payload_to_pages(payload_b),
            previous_pages=_payload_to_pages(payload_a),
            current_results=_payload_to_results(payload_b),
            previous_results=_payload_to_results(payload_a),
        )
        return JSONResponse(content={
            "job_id_a": req.job_id_a,
            "job_id_b": req.job_id_b,
            "diff":     report.to_dict(),
        })
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Revision comparator not available (src.revision.comparator missing)",
        )
    except Exception as exc:
        logger.exception("revision_diff failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Diff computation failed: {exc}")


@router.get("/api/revision/boq-diff/{job_id_a}/{job_id_b}")
def boq_diff(job_id_a: str, job_id_b: str) -> JSONResponse:
    """
    Compare BOQ line items between two completed analysis jobs.

    Returns added/removed/changed BOQ items so the estimator can see scope deltas
    at the item level.
    """
    payload_a = _get_complete_payload(job_id_a)
    payload_b = _get_complete_payload(job_id_b)

    items_a = payload_a.get("boq_items", []) or payload_a.get("line_items", [])
    items_b = payload_b.get("boq_items", []) or payload_b.get("line_items", [])

    result = _diff_boq_items(items_a, items_b)
    result.update({
        "job_id_a": job_id_a,
        "job_id_b": job_id_b,
        "total_a":  len(items_a),
        "total_b":  len(items_b),
    })
    return JSONResponse(content=result)
