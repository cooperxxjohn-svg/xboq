"""
Audit log read API.

GET /api/audit          — list audit events for the caller's org (admin only)
GET /api/audit/stats    — count events by type for the caller's org
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["audit"])


@router.get("/api/audit")
async def list_audit_events(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None),
) -> JSONResponse:
    """
    Return audit log entries for the caller's org, newest first.

    Requires admin role — audit log contains sensitive operational data.
    """
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import require_role
    from src.api.audit_log import get_audit_log

    ctx = await get_tenant_context(request)
    require_role(ctx, "admin")

    events = get_audit_log(
        org_id=ctx.org_id,
        limit=limit,
        offset=offset,
        event_type=event_type,
    )
    return JSONResponse(content={
        "org_id": ctx.org_id,
        "count": len(events),
        "events": events,
    })


@router.get("/api/audit/stats")
async def audit_stats(request: Request) -> JSONResponse:
    """
    Return event counts grouped by type for the caller's org.
    Requires admin role.
    """
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import require_role
    from src.api.models import AuditLogModel
    from src.api.db import SessionLocal
    from sqlalchemy import select, func

    ctx = await get_tenant_context(request)
    require_role(ctx, "admin")

    with SessionLocal() as db:
        rows = db.execute(
            select(AuditLogModel.event_type, func.count(AuditLogModel.id).label("n"))
            .where(AuditLogModel.org_id == ctx.org_id)
            .group_by(AuditLogModel.event_type)
            .order_by(func.count(AuditLogModel.id).desc())
        ).all()

    stats = {row.event_type: row.n for row in rows}
    return JSONResponse(content={"org_id": ctx.org_id, "stats": stats})
