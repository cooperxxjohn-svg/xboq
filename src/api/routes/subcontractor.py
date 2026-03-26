"""
Subcontractor Portal endpoints.

Allows sharing a read-only scope package with a subcontractor via a
time-limited token. The sub fills in their rates; data feeds back into
the job's BOQ.

Endpoints:
    POST /api/subcontractor/share/{job_id}      — generate a share token
    GET  /api/subcontractor/portal/{token}       — get read-only scope package
    POST /api/subcontractor/portal/{token}/rates — submit rates
    GET  /api/subcontractor/tokens/{job_id}      — list active tokens for a job

Token storage: ~/.xboq/subcon_tokens.json  (in-memory + file backed)

Token payload:
    {
      "token":       "uuid4",
      "job_id":      "...",
      "trade":       "civil",          # optional — limit to one trade
      "org_name":    "Sub Co Pvt Ltd",
      "created_at":  "ISO8601",
      "expires_at":  "ISO8601",        # 7-day default
      "submitted":   false,
      "submitted_at": null,
      "rate_count":   0
    }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["subcontractor"])

# ---------------------------------------------------------------------------
# Token store (in-memory + file backed)
# ---------------------------------------------------------------------------

_TOKEN_FILE = Path.home() / ".xboq" / "subcon_tokens.json"
_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

# token_str → token_dict
_tokens: Dict[str, dict] = {}


def _load_tokens() -> None:
    global _tokens
    if _TOKEN_FILE.exists():
        try:
            _tokens = json.loads(_TOKEN_FILE.read_text("utf-8"))
        except Exception:
            _tokens = {}


def _save_tokens() -> None:
    _TOKEN_FILE.write_text(json.dumps(_tokens, indent=2), encoding="utf-8")


_load_tokens()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expire_dt(days: int = 7) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _token_valid(tok: dict) -> bool:
    try:
        exp = datetime.fromisoformat(tok["expires_at"])
        return datetime.now(timezone.utc) < exp
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ShareRequest(BaseModel):
    trade: Optional[str] = None          # filter to one trade; None = all trades
    org_name: Optional[str] = ""
    expire_days: Optional[int] = 7


class RateSubmissionItem(BaseModel):
    item_id: str
    rate_inr: float
    notes: Optional[str] = ""


class RateSubmission(BaseModel):
    org_name: Optional[str] = ""
    contact_email: Optional[str] = ""
    items: List[RateSubmissionItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_job(job_id: str):
    """Load job or raise 404."""
    try:
        from src.api.job_store import job_store
        job = job_store.get_job(job_id)
    except Exception:
        job = None
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


def _boq_items_for_trade(payload: dict, trade: Optional[str]) -> List[dict]:
    """Return BOQ items optionally filtered by trade, stripping sensitive rate data."""
    items = payload.get("boq_items", []) or []
    if trade:
        items = [i for i in items if (i.get("trade") or "").lower() == trade.lower()]
    # Return read-only view: no rates
    public_fields = ("item_id", "description", "unit", "quantity", "trade",
                     "spec_ref", "drawing_ref", "notes")
    return [
        {k: i.get(k, "") for k in public_fields}
        for i in items
    ]


def _scope_summary(payload: dict, trade: Optional[str]) -> dict:
    """Build a lightweight scope summary for the sub portal."""
    project_name = payload.get("project_name") or payload.get("project_id") or "Tender"

    # Gather RFIs relevant to the trade
    rfis = payload.get("rfis", []) or []
    if trade:
        rfis = [r for r in rfis if (r.get("trade") or "").lower() == trade.lower()]

    # Gather drawings references
    drawings = payload.get("page_index", []) or []
    dwg_refs = []
    for p in drawings:
        if p.get("doc_type") in ("drawing", "floorplan"):
            if not trade or (p.get("discipline") or "").lower() == trade.lower():
                dwg_refs.append({
                    "page": p.get("page_number"),
                    "description": p.get("heading") or p.get("doc_type"),
                })

    return {
        "project_name": project_name,
        "trade": trade or "all",
        "drawing_refs": dwg_refs[:20],
        "rfi_count": len(rfis),
        "rfis": [
            {
                "id": r.get("rfi_id") or r.get("id", ""),
                "question": r.get("question") or r.get("title", ""),
                "trade": r.get("trade", ""),
            }
            for r in rfis[:10]
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/subcontractor/share/{job_id}", status_code=201)
async def share_scope(job_id: str, req: ShareRequest, request: Request) -> JSONResponse:
    """
    Generate a share token for a job. The token URL can be sent to a
    subcontractor so they can view the scope and submit rates.
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

    job = _get_job(job_id)
    if job.status != "complete" or not job.payload:
        raise HTTPException(status_code=400, detail="Job must be complete before sharing")

    token = str(uuid.uuid4())
    tok_data = {
        "token":        token,
        "job_id":       job_id,
        "trade":        (req.trade or "").lower() or None,
        "org_name":     req.org_name or "",
        "created_at":   _utcnow(),
        "expires_at":   _expire_dt(req.expire_days or 7),
        "submitted":    False,
        "submitted_at": None,
        "rate_count":   0,
        "submitted_rates": [],
    }
    _tokens[token] = tok_data
    _save_tokens()

    portal_url = f"/api/subcontractor/portal/{token}"
    logger.info("Subcontractor share token created: job=%s token=%s", job_id, token[:8])

    return JSONResponse(
        content={
            "token":       token,
            "portal_url":  portal_url,
            "job_id":      job_id,
            "trade":       tok_data["trade"],
            "expires_at":  tok_data["expires_at"],
        },
        status_code=201,
    )


@router.get("/api/subcontractor/portal/{token}")
def get_portal(token: str) -> JSONResponse:
    """
    Return the read-only scope package for a subcontractor.
    No rate/pricing data is included.
    """
    tok = _tokens.get(token)
    if not tok:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    if not _token_valid(tok):
        raise HTTPException(status_code=410, detail="Token has expired")

    job = _get_job(tok["job_id"])
    payload = job.payload or {}
    trade = tok.get("trade")

    boq_items = _boq_items_for_trade(payload, trade)
    scope_summary = _scope_summary(payload, trade)

    return JSONResponse(content={
        "token":         token,
        "job_id":        tok["job_id"],
        "org_name":      tok.get("org_name", ""),
        "trade":         trade or "all",
        "expires_at":    tok["expires_at"],
        "submitted":     tok.get("submitted", False),
        "scope_summary": scope_summary,
        "boq_items":     boq_items,
        "item_count":    len(boq_items),
    })


@router.post("/api/subcontractor/portal/{token}/rates")
def submit_rates(token: str, submission: RateSubmission) -> JSONResponse:
    """
    Subcontractor submits rates for BOQ items. Rates are written back
    into the job payload's boq_items as `sub_rate_inr`.
    """
    tok = _tokens.get(token)
    if not tok:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    if not _token_valid(tok):
        raise HTTPException(status_code=410, detail="Token has expired")

    job = _get_job(tok["job_id"])
    payload = job.payload or {}

    # Map item_id → submitted rate
    rate_map = {item.item_id: item for item in submission.items}

    # Apply rates to boq_items
    updated = 0
    boq_items = payload.get("boq_items") or []
    for item in boq_items:
        iid = item.get("item_id") or item.get("id") or ""
        if iid in rate_map:
            r = rate_map[iid]
            item["sub_rate_inr"]  = r.rate_inr
            item["sub_notes"]     = r.notes or ""
            item["sub_org"]       = submission.org_name or tok.get("org_name", "")
            updated += 1

    # Persist token state
    tok["submitted"]      = True
    tok["submitted_at"]   = _utcnow()
    tok["rate_count"]     = len(submission.items)
    tok["submitted_rates"] = [
        {"item_id": r.item_id, "rate_inr": r.rate_inr, "notes": r.notes}
        for r in submission.items
    ]
    tok["sub_contact"] = submission.contact_email or ""
    _save_tokens()

    # Persist updated payload back to job store
    try:
        from src.api.job_store import job_store
        job.payload = payload
        job_store._jobs[tok["job_id"]] = job   # direct update; save if store supports it
        if hasattr(job_store, "save"):
            job_store.save()
    except Exception as exc:
        logger.warning("Could not persist subcon rates to job store: %s", exc)

    logger.info(
        "Subcon rates submitted: job=%s token=%s items=%d updated=%d",
        tok["job_id"], token[:8], len(submission.items), updated,
    )

    return JSONResponse(content={
        "ok":          True,
        "job_id":      tok["job_id"],
        "items_received": len(submission.items),
        "items_updated":  updated,
        "org_name":    submission.org_name or tok.get("org_name", ""),
    })


@router.get("/api/subcontractor/tokens/{job_id}")
def list_tokens(job_id: str) -> JSONResponse:
    """List all share tokens for a job (active and expired)."""
    result = []
    for tok in _tokens.values():
        if tok.get("job_id") == job_id:
            result.append({
                "token":       tok["token"][:8] + "…",  # partial for security
                "trade":       tok.get("trade"),
                "org_name":    tok.get("org_name", ""),
                "created_at":  tok.get("created_at"),
                "expires_at":  tok.get("expires_at"),
                "valid":       _token_valid(tok),
                "submitted":   tok.get("submitted", False),
                "rate_count":  tok.get("rate_count", 0),
            })
    result.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return JSONResponse(content={"job_id": job_id, "tokens": result, "count": len(result)})
