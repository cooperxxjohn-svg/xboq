"""
Project membership management API.

GET  /api/projects/{project_id}/members          — list members
POST /api/projects/{project_id}/members          — add / update member
DELETE /api/projects/{project_id}/members/{uid}  — remove member
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.audit_log import audit, AuditEvent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["projects"])


class MemberRequest(BaseModel):
    user_id: str
    role: str   # "viewer" | "editor" | "admin"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/projects/{project_id}/members")
async def list_members(project_id: str, request: Request) -> JSONResponse:
    """List all members of a project. Requires viewer+."""
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import list_members as _list, require_role
    ctx = await get_tenant_context(request)
    require_role(ctx, "viewer", project_id=project_id)
    return JSONResponse(content={"project_id": project_id, "members": _list(project_id)})


@router.post("/api/projects/{project_id}/members", status_code=201)
async def add_member(
    project_id: str,
    body: MemberRequest,
    request: Request,
) -> JSONResponse:
    """
    Add or update a project member. Requires admin.

    The first call on a project with no members bootstraps the caller as admin.
    """
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import add_member as _add, list_members as _list, require_role

    ctx = await get_tenant_context(request)

    # Bootstrap: if no members exist, the authenticated caller becomes admin
    existing = _list(project_id)
    if not existing:
        _add(project_id, ctx.org_id, "admin", added_by=ctx.org_id)
    else:
        require_role(ctx, "admin", project_id=project_id)

    result = _add(project_id, body.user_id, body.role, added_by=ctx.org_id)
    audit(AuditEvent.MEMBER_ADDED, ctx,
          resource_type="member", resource_id=body.user_id,
          detail={"project_id": project_id, "role": body.role}, request=request)
    return JSONResponse(status_code=201, content=result)


@router.delete("/api/projects/{project_id}/members/{user_id}")
async def remove_member(
    project_id: str,
    user_id: str,
    request: Request,
) -> JSONResponse:
    """Remove a member from a project. Requires admin."""
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import remove_member as _remove, require_role

    ctx = await get_tenant_context(request)
    require_role(ctx, "admin", project_id=project_id)

    removed = _remove(project_id, user_id)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' is not a member of project '{project_id}'",
        )
    audit(AuditEvent.MEMBER_REMOVED, ctx,
          resource_type="member", resource_id=user_id,
          detail={"project_id": project_id}, request=request)
    return JSONResponse(content={"removed": True, "user_id": user_id, "project_id": project_id})
