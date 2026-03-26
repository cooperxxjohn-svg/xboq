"""
GET /api/jobs/{job_id}                   — job status + payload
GET /api/jobs/{job_id}/export/{format}   — download export file
GET /api/jobs/{job_id}/report            — shareable read-only summary (no auth required)
POST /api/jobs/{job_id}/rfi-feedback     — thumbs up/down on an RFI (JSONL store)
GET /api/jobs/history                    — last N jobs for the requesting org
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.api.job_store import job_store
from src.api.analytics import log_event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])


def _sanitize_payload(obj: Any) -> Any:
    """
    Recursively replace NaN/Inf floats with None so the payload is JSON-safe.

    The pipeline can produce NaN from numpy operations, Pydantic coercion, or
    missing rate lookups.  This is the last-mile safety net — callers should
    fix root causes, but this prevents 500s reaching the client.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    return obj

_CONTENT_TYPES = {
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pdf":   "application/pdf",
    "word":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_FILE_EXTENSIONS = {
    "excel": ".xlsx",
    "pdf":   ".pdf",
    "word":  ".docx",
}


# ---------------------------------------------------------------------------
# Job history — MUST be registered before /{job_id} to avoid route shadowing
# ---------------------------------------------------------------------------

@router.get("/api/jobs/history")
async def job_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    org_id: Optional[str] = Query(default=None),
) -> JSONResponse:
    """
    Return the most recent analysis jobs for the requesting organisation.

    org_id resolution order:
      1. ?org_id= query param (explicit override, admin use)
      2. Authenticated tenant context (Bearer JWT / X-Org-Id header)
      3. "local" (unauthenticated / dev)
    """
    resolved_org = org_id or "local"
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        ctx = await get_tenant_context(request)
        if ctx.authenticated and not org_id:
            resolved_org = ctx.org_id
    except Exception:
        pass

    jobs = job_store.list_jobs_by_org(resolved_org, limit=limit, offset=offset)

    return JSONResponse(content={
        "org_id": resolved_org,
        "limit":  limit,
        "offset": offset,
        "count":  len(jobs),
        "jobs": [
            {
                "job_id":           j.job_id,
                "status":           j.status,
                "project_name":     j.project_name,
                "run_mode":         j.run_mode,
                "progress":         j.progress,
                "progress_message": j.progress_message,
                "queue_position":   j.queue_position,
                "created_at":       j.created_at.isoformat(),
                "completed_at":     j.completed_at.isoformat() if j.completed_at else None,
                "has_payload":      bool(j.payload_path),
                "errors":           j.errors,
            }
            for j in jobs
        ],
    })


# ---------------------------------------------------------------------------
# Status + payload
# ---------------------------------------------------------------------------

@router.get("/api/jobs/{job_id}", summary="Get job status and result",
            response_description="Job metadata; includes payload when status=complete")
def get_job(job_id: str) -> JSONResponse:
    """Return current status and (when complete) full result payload for a job."""
    # Load metadata + payload from disk for complete jobs
    job = job_store.get_job_with_payload(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    body: dict = {
        "job_id":           job.job_id,
        "status":           job.status,
        "progress":         job.progress,
        "progress_message": job.progress_message,
        "queue_position":   job.queue_position,
        "errors":           job.errors,
        "org_id":           job.org_id,
        "project_name":     job.project_name,
        "run_mode":         job.run_mode,
        "created_at":       job.created_at.isoformat(),
        "completed_at":     job.completed_at.isoformat() if job.completed_at else None,
    }

    if job.status == "complete" and job.payload:
        log_event("report_viewed", job_id=job_id, org_id=job.org_id, run_mode=job.run_mode)
        p = job.payload
        body["payload"]            = p
        body["line_items"]         = p.get("line_items", [])
        body["contractual_items"]  = p.get("contractual_items", [])
        body["line_items_summary"] = p.get("line_items_summary", {})
        body["qto_summary"]        = p.get("qto_summary", {})
        body["cache_stats"]        = p.get("cache_stats", {})

    # Sanitize any NaN/Inf floats that may come from numpy/rate engine before
    # serialising to JSON — a missing rate lookup returns NaN which would 500
    return JSONResponse(content=_sanitize_payload(body))


# ---------------------------------------------------------------------------
# Export download
# ---------------------------------------------------------------------------

@router.get("/api/jobs/{job_id}/export/{format}",
            summary="Download export file",
            response_description="Binary file (Excel / PDF / Word)")
def get_export(job_id: str, format: str) -> FileResponse:
    """Download a generated export file (excel / pdf / word) for a completed job."""
    if format not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown format '{format}'. Supported: {', '.join(_CONTENT_TYPES)}",
        )

    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status != "complete":
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' is not complete (status: {job.status})",
        )

    file_path_str = job.output_files.get(format)
    if not file_path_str:
        raise HTTPException(
            status_code=404,
            detail=f"Export format '{format}' was not generated for job '{job_id}'",
        )

    file_path = Path(file_path_str)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Export file for '{format}' no longer exists on disk",
        )

    log_event("export_downloaded", job_id=job_id, org_id=job.org_id, run_mode=job.run_mode,
              extra={"format": format})

    return FileResponse(
        path=str(file_path),
        media_type=_CONTENT_TYPES[format],
        filename=f"{job_id}{_FILE_EXTENSIONS[format]}",
    )


# ---------------------------------------------------------------------------
# Shareable report — read-only summary, no auth required
# ---------------------------------------------------------------------------

@router.get(
    "/api/jobs/{job_id}/report",
    summary="Shareable read-only report",
    response_description="Curated summary of a completed job suitable for sharing",
)
def get_report(job_id: str) -> JSONResponse:
    """
    Return a minimal read-only JSON summary of a completed job.

    No authentication required — suitable for sharing with someone who doesn't
    have an account.  Sensitive rate data is excluded.

    Returns:
      - 404 if job not found
      - 202 + {"status": "pending"} if job not yet complete
      - 200 + curated summary for complete jobs
    """
    log_event("report_shared", job_id=job_id)

    job = job_store.get_job_with_payload(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status != "complete":
        return JSONResponse(
            status_code=202,
            content={"status": "pending", "job_id": job_id},
        )

    p = job.payload or {}

    # Extract curated fields — safe to expose, no rate data
    rfis: list = p.get("rfis", [])
    rfi_summary = [
        {"id": r.get("id", ""), "trade": r.get("trade", ""), "question": r.get("question", "")}
        for r in rfis[:10]
    ]

    blockers: list = p.get("blockers", [])
    qa_score_raw = p.get("qa_score", {})
    if isinstance(qa_score_raw, dict):
        qa_score = qa_score_raw.get("overall", qa_score_raw.get("score", 0))
    else:
        qa_score = qa_score_raw or 0

    cost_summary_raw = p.get("cost_summary", {})
    cost_summary = {
        "tender_value_inr": cost_summary_raw.get("tender_value_inr"),
        "qto_total_inr":    cost_summary_raw.get("qto_total_inr"),
    }

    summary = {
        "job_id":        job.job_id,
        "project_name":  job.project_name,
        "run_mode":      job.run_mode,
        "completed_at":  job.completed_at.isoformat() if job.completed_at else None,
        "qa_score":      qa_score,
        "total_rfis":    len(rfis),
        "blockers":      len(blockers),
        "rfi_summary":   rfi_summary,
        "cost_summary":  cost_summary,
        "share_url":     f"/api/jobs/{job_id}/report",
    }

    return JSONResponse(content=summary)


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------

@router.delete(
    "/api/jobs/{job_id}",
    summary="Cancel a job",
    response_description="Confirmation that the job was cancelled",
)
async def cancel_job(job_id: str, request: Request) -> JSONResponse:
    """
    Cancel a queued or processing job.

    The job is marked as 'cancelled' in the database. Running pipeline
    threads check for this status and stop gracefully on their next
    progress update. Already-terminal jobs (complete / error / cancelled)
    return 409.

    Requires editor+ role.
    """
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        from src.auth.rbac import require_role
        ctx = await get_tenant_context(request)
        require_role(ctx, "editor")
    except Exception as exc:
        from fastapi import HTTPException as _HTTPException
        if isinstance(exc, _HTTPException):
            raise

    cancelled = job_store.cancel_job(job_id)
    if cancelled is None or job_store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if not cancelled:
        job = job_store.get_job(job_id)
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is in terminal state '{job.status}' and cannot be cancelled",
        )
    return JSONResponse(content={"cancelled": True, "job_id": job_id})


# ---------------------------------------------------------------------------
# RFI feedback — lightweight thumbs up/down, JSONL store
# ---------------------------------------------------------------------------

class RFIFeedbackRequest(BaseModel):
    rfi_id: str
    useful:  bool
    job_id:  str = ""


def _rfi_feedback_path() -> Path:
    """Resolve the RFI feedback JSONL file path."""
    override = os.environ.get("XBOQ_RFI_FEEDBACK_FILE", "")
    if override:
        return Path(override)
    return Path.home() / ".xboq" / "rfi_feedback.jsonl"


@router.post(
    "/api/jobs/{job_id}/rfi-feedback",
    status_code=201,
    summary="Submit thumbs up/down on an RFI",
    response_description="201 Created when feedback is stored",
)
def post_rfi_feedback(job_id: str, body: RFIFeedbackRequest) -> JSONResponse:
    """
    Record a thumbs up/down vote on a specific RFI.

    Appended to ~/.xboq/rfi_feedback.jsonl — no database required.
    """
    record = {
        "ts":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_id": job_id,
        "rfi_id": body.rfi_id,
        "useful": body.useful,
    }

    try:
        path = _rfi_feedback_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write rfi_feedback: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to store RFI feedback: {exc}")

    log_event("rfi_feedback", job_id=job_id, rfi_id=body.rfi_id,
              extra={"useful": body.useful})

    return JSONResponse(
        status_code=201,
        content={"message": "RFI feedback recorded.", "rfi_id": body.rfi_id, "useful": body.useful},
    )
