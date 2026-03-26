"""
Role-Based Access Control for xBOQ.ai.

Roles (additive — higher roles inherit lower-role permissions):
    viewer  — read payload, download reports
    editor  — viewer + run analysis, edit rates, raise RFIs, submit QA reviews
    admin   — editor + invite/remove members, change roles, delete runs, manage project

Project membership is stored in:
    ~/.xboq/project_members/{project_id}.json   (local)

Schema:
    {
        "members": [
            {"user_id": "acme_corp", "role": "admin",  "added_at": "2025-01-01T..."},
            {"user_id": "bob",       "role": "editor", "added_at": "2025-01-02T..."}
        ]
    }

Usage:
    from src.auth.rbac import require_role, can, add_member, get_role

    # In a FastAPI dependency chain:
    ctx = await get_tenant_context(request)
    require_role(ctx, "editor", project_id=job.project_id)

    # Inline check:
    if can(ctx, "editor", project_id=job.project_id):
        ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLES = ("viewer", "editor", "admin")
_ROLE_RANK = {r: i for i, r in enumerate(ROLES)}   # viewer=0, editor=1, admin=2

_MEMBERS_DIR = Path.home() / ".xboq" / "project_members"
_MEMBERS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Membership storage — DB-backed with file fallback
#
# DB (ProjectMemberModel) is used when DATABASE_URL points to PostgreSQL or
# an accessible SQLite file.  Falls back to flat-file JSON when the DB is not
# reachable (e.g. unit tests that don't bootstrap the DB).
# ---------------------------------------------------------------------------

def _members_path(project_id: str) -> Path:
    safe = "".join(c for c in project_id if c.isalnum() or c in "-_")
    return _MEMBERS_DIR / f"{safe}.json"


def _load_members(project_id: str) -> list[dict]:
    """Load project membership — DB first, file fallback."""
    try:
        from src.api.db import SessionLocal
        from src.api.models import ProjectMemberModel
        from sqlalchemy import select
        with SessionLocal() as db:
            rows = db.execute(
                select(ProjectMemberModel).where(ProjectMemberModel.project_id == project_id)
            ).scalars().all()
        if rows:
            return [r.to_dict() for r in rows]
        # DB returned empty — may be a new project or pre-migration data; try file
    except Exception:
        pass  # DB unavailable — fall through to file

    # File fallback
    path = _members_path(project_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())["members"]
    except Exception:
        return []


def _save_members(project_id: str, members: list[dict]) -> None:
    """Persist membership — DB primary, file mirror for local dev."""
    _db_saved = False
    try:
        from src.api.db import SessionLocal
        from src.api.models import ProjectMemberModel
        from sqlalchemy import select, delete as sa_delete
        with SessionLocal() as db:
            # Upsert: delete existing rows for project, re-insert
            db.execute(sa_delete(ProjectMemberModel).where(
                ProjectMemberModel.project_id == project_id
            ))
            for m in members:
                db.add(ProjectMemberModel(
                    project_id=project_id,
                    user_id=m["user_id"],
                    role=m.get("role", "viewer"),
                    added_by=m.get("added_by", ""),
                    added_at=datetime.now(timezone.utc),
                ))
            db.commit()
        _db_saved = True
    except Exception as exc:
        logger.warning("DB member save failed, falling back to file: %s", exc)

    # Always mirror to file (enables local dev without DB)
    path = _members_path(project_id)
    try:
        path.write_text(json.dumps({"members": members}, indent=2, default=str))
    except Exception as exc:
        if not _db_saved:
            logger.error("Both DB and file member save failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_role(user_id: str, project_id: str) -> Optional[str]:
    """
    Return the role of user_id on project_id, or None if not a member.

    If no membership records exist for the project, the first user to touch it
    is treated as an implicit admin (bootstrap behaviour for new projects).
    """
    members = _load_members(project_id)
    if not members:
        return None   # project has no explicit members yet
    for m in members:
        if m.get("user_id") == user_id:
            return m.get("role", "viewer")
    return None


def add_member(
    project_id: str,
    user_id: str,
    role: str,
    added_by: str = "",
) -> dict:
    """
    Add or update a member's role on a project.

    Returns the membership dict.
    Raises ValueError if role is invalid.
    """
    if role not in ROLES:
        raise ValueError(f"Invalid role '{role}'. Choose from: {', '.join(ROLES)}")
    members = _load_members(project_id)
    now = datetime.now(timezone.utc).isoformat()
    for m in members:
        if m["user_id"] == user_id:
            m["role"] = role
            m["updated_at"] = now
            break
    else:
        members.append({
            "user_id":  user_id,
            "role":     role,
            "added_by": added_by,
            "added_at": now,
        })
    _save_members(project_id, members)
    return {"user_id": user_id, "role": role, "project_id": project_id}


def remove_member(project_id: str, user_id: str) -> bool:
    """Remove a member from a project. Returns True if removed, False if not found."""
    members = _load_members(project_id)
    new = [m for m in members if m.get("user_id") != user_id]
    if len(new) == len(members):
        return False
    _save_members(project_id, new)
    return True


def list_members(project_id: str) -> list[dict]:
    """Return all members of a project."""
    return _load_members(project_id)


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------

def effective_role(user_id: str, project_id: str, org_default: str = "viewer") -> str:
    """
    Return the effective role for user_id on project_id.

    Priority: explicit project membership > org_default.
    When a project has NO members, any authenticated user gets org_default role.
    """
    role = get_role(user_id, project_id)
    return role if role is not None else org_default


def can(ctx, min_role: str, project_id: str = "") -> bool:
    """
    Return True if TenantContext has at least min_role on the project.

    If project_id is empty, only the role embedded in the JWT is checked.
    """
    if min_role not in _ROLE_RANK:
        return False
    if not ctx.authenticated:
        # Unauthenticated requests never pass role checks in production
        import os
        if not os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes"):
            return False
        # Dev mode: trust the role already assigned to the context
        # (get_tenant_context sets role="editor" in dev mode)
        return _ROLE_RANK.get(ctx.role, -1) >= _ROLE_RANK[min_role]

    # Resolve role: project membership overrides JWT role
    user_role = ctx.role
    if project_id:
        user_role = effective_role(ctx.org_id, project_id, org_default=ctx.role)

    return _ROLE_RANK.get(user_role, -1) >= _ROLE_RANK[min_role]


def require_role(ctx, min_role: str, project_id: str = "") -> None:
    """
    Raise HTTP 403 if the caller does not have at least min_role.
    Raise HTTP 401 if not authenticated at all (non-dev mode).
    """
    import os
    dev_mode = os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes")

    if not ctx.authenticated and not dev_mode:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide Authorization: Bearer <token>",
        )

    if not can(ctx, min_role, project_id=project_id):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Role '{ctx.role}' is insufficient for this action. "
                f"Requires '{min_role}' or higher."
            ),
        )
