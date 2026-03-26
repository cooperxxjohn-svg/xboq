"""
Tenant management REST endpoints.

POST /api/tenants/register          — create org, default plan=free
GET  /api/tenants/me/usage          — current quota + usage stats
GET  /api/tenants/me/projects       — list org's projects (via project_store)
POST /api/tenants/me/upgrade        — change plan (requires XBOQ_ADMIN_KEY header)
GET  /api/tenants                   — list all tenants (admin only)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.middleware.tenant_auth import TenantContext, get_tenant_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tenants"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    org_id: str
    plan: Optional[str] = "free"


class UpgradeRequest(BaseModel):
    new_plan: str


# ---------------------------------------------------------------------------
# Admin key check
# ---------------------------------------------------------------------------

def _require_admin(x_admin_key: str = Header(default="")) -> None:
    admin_key = os.environ.get("XBOQ_ADMIN_KEY", "")
    if admin_key and x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Admin key required")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/tenants/register", status_code=201)
def register_tenant(req: RegisterRequest) -> JSONResponse:
    """Register a new organisation. Idempotent."""
    from src.auth.tenant_manager import register_tenant as _reg
    if not req.org_id:
        raise HTTPException(status_code=400, detail="org_id is required")
    quota = _reg(req.org_id, plan=req.plan or "free")
    return JSONResponse(content=quota.to_dict(), status_code=201)


@router.get("/api/tenants/me/usage")
def get_usage(tenant: TenantContext = Depends(get_tenant_context)) -> JSONResponse:
    """Return quota usage for the calling organisation."""
    from src.auth.tenant_manager import get_quota
    quota = get_quota(tenant.org_id)
    return JSONResponse(content=quota.to_dict())


@router.get("/api/tenants/me/projects")
def list_projects(
    limit: int = 20,
    tenant: TenantContext = Depends(get_tenant_context),
) -> JSONResponse:
    """List projects belonging to this organisation."""
    try:
        from src.auth.project_store import list_projects as _lp
        projects = _lp(org_id=tenant.org_id, limit=limit)
        return JSONResponse(content={"org_id": tenant.org_id,
                                     "projects": projects,
                                     "count": len(projects)})
    except Exception as exc:
        logger.warning("list_projects failed: %s", exc)
        return JSONResponse(content={"org_id": tenant.org_id, "projects": [], "count": 0})


@router.post("/api/tenants/me/upgrade")
def upgrade_plan(
    req: UpgradeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    _admin: None = Depends(_require_admin),
) -> JSONResponse:
    """Change the plan for the calling organisation (admin key required)."""
    from src.auth.tenant_manager import upgrade_plan as _up, VALID_PLANS
    if req.new_plan not in VALID_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plan '{req.new_plan}'. Valid: {sorted(VALID_PLANS)}"
        )
    quota = _up(tenant.org_id, req.new_plan)
    return JSONResponse(content=quota.to_dict())


@router.get("/api/tenants")
def list_all_tenants(
    _admin: None = Depends(_require_admin),
) -> JSONResponse:
    """List all registered organisations (admin only)."""
    from src.auth.tenant_manager import list_tenants as _lt
    tenants = _lt()
    return JSONResponse(content={"tenants": tenants, "count": len(tenants)})
