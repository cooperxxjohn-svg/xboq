"""
xBOQ.ai — Immutable compliance audit log.

Every significant platform action is appended here so that L&T's internal
compliance team, government auditors, or security reviewers can answer:

  - Who ran which analysis, on which tender, at what time?
  - Who changed rates and what were the before/after values?
  - Who was added to / removed from a project?
  - Were there failed login attempts?

Rules
-----
  1. Rows are NEVER updated or deleted — only INSERT.
  2. detail_json stores arbitrary structured context (JSON).
  3. ip_address is optional but populated from FastAPI Request when available.
  4. All callers are non-blocking: failures are logged as warnings, never raised.

Usage
-----
  from src.api.audit_log import audit

  # From a route handler
  audit("job.created", ctx, resource_type="job", resource_id=job_id,
        detail={"run_mode": run_mode, "file_count": n}, request=request)

  # From non-request code (no IP available)
  audit("rates.override_set", ctx, resource_type="rate",
        resource_id=material_key, detail={"value": rate_inr})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type constants  (dot-namespaced: <resource>.<action>)
# ---------------------------------------------------------------------------

class AuditEvent:
    # Jobs
    JOB_CREATED   = "job.created"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED    = "job.failed"

    # Auth
    AUTH_LOGIN         = "auth.login"
    AUTH_LOGIN_FAILED  = "auth.login_failed"
    AUTH_LOGOUT        = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"

    # Rates
    RATE_OVERRIDE_SET     = "rates.override_set"
    RATE_OVERRIDE_DELETED = "rates.override_deleted"
    RATE_BULK_IMPORT      = "rates.bulk_import"

    # Project members
    MEMBER_ADDED   = "member.added"
    MEMBER_REMOVED = "member.removed"


# ---------------------------------------------------------------------------
# Core log function
# ---------------------------------------------------------------------------

def audit(
    event_type: str,
    ctx: Any,                           # TenantContext — optional, graceful if None
    *,
    resource_type: str = "",
    resource_id: str = "",
    detail: Optional[dict] = None,
    request: Any = None,                # FastAPI Request — for IP extraction
) -> None:
    """
    Append one row to audit_log.

    Never raises — all errors are swallowed and logged as warnings.
    """
    try:
        _write_audit(event_type, ctx, resource_type, resource_id, detail or {}, request)
    except Exception as exc:
        logger.warning("audit_log write failed [%s]: %s", event_type, exc)


def _write_audit(
    event_type: str,
    ctx: Any,
    resource_type: str,
    resource_id: str,
    detail: dict,
    request: Any,
) -> None:
    from src.api.models import AuditLogModel
    from src.api.db import SessionLocal

    org_id  = getattr(ctx, "org_id",  "") or "" if ctx else ""
    user_id = getattr(ctx, "user_id", "") or "" if ctx else ""

    ip_address = ""
    if request is not None:
        try:
            # FastAPI: check X-Forwarded-For (reverse proxy) then client host
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                ip_address = forwarded.split(",")[0].strip()
            elif hasattr(request, "client") and request.client:
                ip_address = request.client.host or ""
        except Exception:
            pass

    row = AuditLogModel(
        event_type=event_type,
        org_id=org_id,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        detail_json=json.dumps(detail, default=str),
        ip_address=ip_address,
        created_at=datetime.now(timezone.utc),
    )

    with SessionLocal() as db:
        db.add(row)
        db.commit()

    logger.info(
        "AUDIT %s  org=%s user=%s res=%s/%s",
        event_type, org_id, user_id, resource_type, resource_id,
    )


# ---------------------------------------------------------------------------
# Query helpers (for /api/audit endpoint)
# ---------------------------------------------------------------------------

def get_audit_log(
    org_id: str,
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[str] = None,
) -> list[dict]:
    """Return audit log entries for an org, newest first."""
    from src.api.models import AuditLogModel
    from src.api.db import SessionLocal
    from sqlalchemy import select, desc

    with SessionLocal() as db:
        q = select(AuditLogModel).where(AuditLogModel.org_id == org_id)
        if event_type:
            q = q.where(AuditLogModel.event_type == event_type)
        q = q.order_by(desc(AuditLogModel.created_at)).limit(limit).offset(offset)
        rows = db.execute(q).scalars().all()
        return [r.to_dict() for r in rows]
