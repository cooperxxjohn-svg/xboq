"""
GET /api/config — read-only view of current configuration (secrets redacted).

Useful for ops to confirm which env vars are active without logging into the server.
Admin key required in production.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.config import settings

router = APIRouter(tags=["admin"])


def _require_admin(request: Request) -> None:
    """
    Require X-Admin-Key header in production.
    In dev mode (XBOQ_DEV_MODE=true) with no admin key configured: open.
    In production with no admin key configured: still blocked (403) to prevent
    accidental config exposure.
    """
    import os
    dev_mode = os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes")
    if not settings.admin_key:
        if dev_mode:
            return  # dev-only shortcut: no key needed
        raise HTTPException(
            status_code=403,
            detail="Admin key required. Set XBOQ_ADMIN_KEY to enable this endpoint.",
        )
    provided = request.headers.get("X-Admin-Key", "")
    if provided != settings.admin_key:
        raise HTTPException(status_code=403, detail="Admin key required")


@router.get(
    "/api/config",
    summary="Show active configuration (admin)",
    response_description="Current settings with secrets redacted",
    dependencies=[Depends(_require_admin)],
)
def get_config() -> JSONResponse:
    """
    Return active configuration as JSON.  All secret values (API keys, tokens)
    are replaced with '***'.

    Requires X-Admin-Key header when XBOQ_ADMIN_KEY env var is set.
    """
    cfg = settings.as_dict(redact_secrets=True)
    # Add computed properties
    cfg["has_llm"]   = settings.has_llm
    cfg["has_redis"] = settings.has_redis
    cfg["effective_jobs_dir"] = str(settings.effective_jobs_dir)
    return JSONResponse(content=cfg)
