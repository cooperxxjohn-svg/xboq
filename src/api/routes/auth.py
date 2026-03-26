"""
Authentication endpoints.

POST /api/auth/login    — validate credentials, issue JWT
POST /api/auth/refresh  — extend an existing valid token
POST /api/auth/logout   — client-side token invalidation (stateless — advisory)
GET  /api/auth/me       — return identity from current token
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.api.audit_log import audit, AuditEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    tenant_id: str
    password: str
    role: Optional[str] = None   # override role if org allows it (future)


class RefreshRequest(BaseModel):
    token: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_response(org_id: str, user_id: str, role: str, ttl_hours: int = 8) -> dict:
    from src.auth.session_tokens import issue_token
    token = issue_token(org_id=org_id, user_id=user_id, role=role, ttl_hours=ttl_hours)
    return {
        "token":      token,
        "org_id":     org_id,
        "user_id":    user_id,
        "role":       role,
        "expires_in": ttl_hours * 3600,
        "token_type": "Bearer",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/api/auth/login")
async def login(body: LoginRequest, request: Request) -> JSONResponse:
    """
    Authenticate with tenant_id + password.

    Returns a JWT on success (200) or 401 on invalid credentials.
    """
    from src.auth.simple_auth import SimpleAuth
    from src.auth.tenant_manager import get_quota

    auth = SimpleAuth()
    result = auth.authenticate(body.tenant_id, body.password)
    if result is None:
        audit(AuditEvent.AUTH_LOGIN_FAILED, None,
              resource_type="auth", resource_id=body.tenant_id,
              detail={"reason": "invalid_credentials"}, request=request)
        raise HTTPException(
            status_code=401,
            detail="Invalid tenant ID or password",
        )

    # Determine role: default to "editor" for authenticated tenants
    # Admins are marked in tenants.json (future) — for now use form param or default
    role = body.role or "editor"
    if role not in ("admin", "editor", "viewer"):
        role = "editor"

    # Ensure quota record exists for this tenant
    try:
        get_quota(body.tenant_id)
    except Exception:
        try:
            from src.auth.tenant_manager import register_tenant
            register_tenant(body.tenant_id)
        except Exception:
            pass

    resp = _token_response(
        org_id=body.tenant_id,
        user_id=result.get("tenant_id", body.tenant_id),
        role=role,
    )
    audit(AuditEvent.AUTH_LOGIN, None,
          resource_type="auth", resource_id=body.tenant_id,
          detail={"role": role}, request=request)
    return JSONResponse(content=resp)


@router.post("/api/auth/refresh")
async def refresh(body: RefreshRequest) -> JSONResponse:
    """
    Issue a new token from a still-valid token (extend session).
    Returns 401 if the token is expired or invalid.
    """
    from src.auth.session_tokens import verify_token, TokenError
    try:
        payload = verify_token(body.token)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    return JSONResponse(content=_token_response(
        org_id=payload.get("org_id", "local"),
        user_id=payload.get("user_id", ""),
        role=payload.get("role", "viewer"),
    ))


@router.post("/api/auth/logout")
async def logout() -> JSONResponse:
    """
    Advisory logout endpoint — tokens are stateless so this is client-side only.
    The client should discard the token after calling this.
    """
    return JSONResponse(content={"detail": "Logged out. Discard your token client-side."})


@router.get("/api/auth/me")
async def me(request: Request) -> JSONResponse:
    """Return the identity extracted from the current Bearer token."""
    from src.api.middleware.tenant_auth import get_tenant_context
    ctx = await get_tenant_context(request)
    if not ctx.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Provide Authorization: Bearer <token>",
        )
    return JSONResponse(content={
        "org_id":        ctx.org_id,
        "user_id":       ctx.user_id,
        "role":          ctx.role,
        "plan":          ctx.plan,
        "authenticated": ctx.authenticated,
    })
