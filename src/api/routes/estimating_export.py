"""
Estimating Software Export REST endpoint — T4-4.

GET /api/jobs/{job_id}/export/estimating?format=sage100|buildertrend|procore|generic
    Returns text/csv with Content-Disposition: attachment
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter(tags=["exports"])


@router.get("/api/jobs/{job_id}/export/estimating")
def export_estimating(
    job_id: str,
    format: str = Query(default="generic",
                        description="Export format: sage100, buildertrend, procore, generic"),
) -> Response:
    """Export BOQ items as CSV formatted for estimating software."""
    from src.api.job_store import get_job
    from src.exports.estimating_export import export_estimating_csv, SUPPORTED_FORMATS

    fmt = format.lower().replace("-", "")
    _norm = {"sage100": "sage100", "sage": "sage100",
             "buildertrend": "buildertrend",
             "procore": "procore",
             "generic": "generic"}
    if fmt not in _norm and fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{format}'. Choose from: {SUPPORTED_FORMATS}",
        )

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    payload = job.get("payload") or job.get("result") or {}
    boq_items = payload.get("boq_items") or []

    csv_content = export_estimating_csv(boq_items, format)
    filename = f"{job_id}_{format}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
