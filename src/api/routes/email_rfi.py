"""
POST /api/jobs/{job_id}/email-rfis  — Send RFI batch email from a completed job.

Request body (JSON):
    {
        "to": ["engineer@contractor.com"],
        "project_name": "Sonipat Hospital",   // optional
        "from_addr": "bids@mycompany.com",    // optional
        "include_drafts": false               // optional
    }

Response:
    {
        "sent": true,
        "transport_used": "smtp",
        "rfi_count": 7,
        "to": ["engineer@contractor.com"],
        "error": ""
    }

Environment variables needed for real delivery:
    SENDGRID_API_KEY  — SendGrid API key (preferred)
    SMTP_HOST         — SMTP server hostname
    SMTP_PORT         — SMTP port (default 587)
    SMTP_USER         — SMTP username
    SMTP_PASSWORD     — SMTP password
    SMTP_TLS          — "1" (default) for STARTTLS, "0" for SSL
    XBOQ_EMAIL_FROM   — Sender address (default: noreply@xboq.ai)
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rfis"])


class EmailRFIRequest(BaseModel):
    to: List[str]
    project_name: Optional[str] = ""
    from_addr: Optional[str] = ""
    include_drafts: Optional[bool] = False


@router.post("/api/jobs/{job_id}/email-rfis")
async def email_rfis(job_id: str, req: EmailRFIRequest, request: Request) -> JSONResponse:
    """
    Send the approved RFIs from a completed job as an email batch.
    """
    from src.api.middleware.tenant_auth import get_tenant_context, TenantContext
    ctx: TenantContext = TenantContext(org_id="local", role="viewer", plan="free", authenticated=False)
    try:
        ctx = await get_tenant_context(request)
    except Exception:
        pass
    try:
        from src.auth.rbac import require_role
        require_role(ctx, "editor")
    except HTTPException:
        raise
    except Exception:
        pass  # non-fatal in dev mode

    if not req.to:
        raise HTTPException(status_code=422, detail="'to' list must not be empty")

    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=422,
            detail=f"Job '{job_id}' is not complete (status: {job.status})",
        )

    payload = job.payload or {}
    rfis = payload.get("rfis") or []

    try:
        from src.notifications.email_sender import send_rfi_batch
        result = send_rfi_batch(
            rfis=rfis,
            to=req.to,
            project_name=req.project_name or job_id,
            from_addr=req.from_addr or "",
            include_drafts=req.include_drafts or False,
        )
    except Exception as exc:
        logger.exception("send_rfi_batch failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Email send failed: {exc}")

    return JSONResponse(content={
        "sent":            result.success,
        "transport_used":  result.transport_used,
        "rfi_count":       len(rfis),
        "to":              req.to,
        "error":           result.error,
        "message_id":      result.message_id,
    })
