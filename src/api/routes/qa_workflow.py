"""
QA Workflow REST endpoints — T4-1.

POST /api/qa/{job_id}/create     — initialise QA job from payload
GET  /api/qa/{job_id}/status     — current review state
POST /api/qa/{job_id}/review     — body: {item_id, status, corrected_qty, note, reviewer}
POST /api/qa/{job_id}/verify     — mark entire job as verified
GET  /api/qa/{job_id}/report     — verified items only
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["qa"])


class ReviewRequest(BaseModel):
    item_id: str
    status: str                          # "approved"|"rejected"|"corrected"
    corrected_qty: Optional[float] = None
    note: str = ""
    reviewer: str = ""


class VerifyRequest(BaseModel):
    verified_by: str = ""


@router.post("/api/qa/{job_id}/create")
def create_qa_job(job_id: str) -> JSONResponse:
    """Initialise a QA job from the stored pipeline payload."""
    from src.api.job_store import get_job
    from src.analysis.qa_workflow import create_qa_job as _create

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    payload = job.get("payload") or job.get("result") or {}
    qa_job = _create(job_id, payload)
    return JSONResponse(content={
        "job_id": job_id,
        "item_count": len(qa_job.items),
        "verified": qa_job.verified,
        "created_at": qa_job.created_at,
    })


@router.get("/api/qa/{job_id}/status")
def qa_status(job_id: str) -> JSONResponse:
    from src.analysis.qa_workflow import get_qa_status

    qa_job = get_qa_status(job_id)
    if qa_job is None:
        raise HTTPException(status_code=404, detail=f"QA job '{job_id}' not found — call /create first")

    total = len(qa_job.items)
    pending = sum(1 for i in qa_job.items if i.status == "pending")
    return JSONResponse(content={
        **qa_job.to_dict(),
        "total_items": total,
        "pending_items": pending,
        "reviewed_items": total - pending,
    })


@router.post("/api/qa/{job_id}/review")
def submit_review(job_id: str, req: ReviewRequest) -> JSONResponse:
    from src.analysis.qa_workflow import submit_review as _submit

    try:
        qa_job = _submit(
            job_id,
            item_id=req.item_id,
            status=req.status,
            corrected_quantity=req.corrected_qty,
            note=req.note,
            reviewer=req.reviewer,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    pending = sum(1 for i in qa_job.items if i.status == "pending")
    return JSONResponse(content={
        "job_id": job_id,
        "item_id": req.item_id,
        "new_status": req.status,
        "pending_remaining": pending,
    })


@router.post("/api/qa/{job_id}/verify")
def verify_job(job_id: str, req: VerifyRequest) -> JSONResponse:
    from src.analysis.qa_workflow import mark_verified

    try:
        qa_job = mark_verified(job_id, verified_by=req.verified_by)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return JSONResponse(content={
        "job_id": job_id,
        "verified": qa_job.verified,
        "verified_at": qa_job.verified_at,
        "verified_by": qa_job.verified_by,
    })


@router.get("/api/qa/{job_id}/report")
def qa_report(job_id: str) -> JSONResponse:
    from src.analysis.qa_workflow import get_qa_status

    qa_job = get_qa_status(job_id)
    if qa_job is None:
        raise HTTPException(status_code=404, detail=f"QA job '{job_id}' not found")

    reviewed = [i.to_dict() for i in qa_job.items if i.status != "pending"]
    return JSONResponse(content={
        "job_id": job_id,
        "verified": qa_job.verified,
        "verified_by": qa_job.verified_by,
        "verified_at": qa_job.verified_at,
        "items": reviewed,
        "item_count": len(reviewed),
    })
