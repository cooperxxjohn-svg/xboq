"""
Scope Change Alert Engine — REST endpoints.

Wraps src/analysis/scope_watcher.py in a FastAPI router.

Endpoints:
    POST /api/scope-watch/{project_id}/set-url      — configure URL monitoring
    POST /api/scope-watch/{project_id}/set-dir      — configure dir monitoring
    POST /api/scope-watch/{project_id}/check        — run a check now
    GET  /api/scope-watch/{project_id}/status       — current watcher state
    GET  /api/scope-watch                           — list all watchers
    POST /api/scope-watch/{project_id}/recipients   — update notification list
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["scope"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SetURLRequest(BaseModel):
    url: str
    notify: Optional[List[str]] = None


class SetDirRequest(BaseModel):
    path: str
    notify: Optional[List[str]] = None


class RecipientsRequest(BaseModel):
    emails: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_watcher(project_id: str):
    from src.analysis.scope_watcher import ScopeWatcher
    return ScopeWatcher(project_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/scope-watch/{project_id}/set-url")
def set_url(project_id: str, req: SetURLRequest) -> JSONResponse:
    """Configure (or reconfigure) a watcher to monitor a URL."""
    if not req.url:
        raise HTTPException(status_code=400, detail="url is required")

    w = _get_watcher(project_id)
    w.set_url(req.url, notify=req.notify)
    return JSONResponse(content={
        "project_id": project_id,
        "source_type": "url",
        "source": req.url,
        "notify": req.notify or w.status()["notify"],
    })


@router.post("/api/scope-watch/{project_id}/set-dir")
def set_dir(project_id: str, req: SetDirRequest) -> JSONResponse:
    """Configure (or reconfigure) a watcher to monitor a local directory."""
    if not req.path:
        raise HTTPException(status_code=400, detail="path is required")

    w = _get_watcher(project_id)
    w.set_dir(req.path, notify=req.notify)
    return JSONResponse(content={
        "project_id": project_id,
        "source_type": "dir",
        "source": req.path,
        "notify": req.notify or w.status()["notify"],
    })


@router.post("/api/scope-watch/{project_id}/check")
def run_check(project_id: str) -> JSONResponse:
    """
    Run a scope check right now.

    Returns the check result and sends an alert email if the content changed.
    """
    w = _get_watcher(project_id)
    result = w.check()

    alert_sent = False
    if result.changed:
        alert_sent = w.send_alert(result, project_name=project_id)
        logger.info(
            "Scope change detected for %s: %s→%s (alert_sent=%s)",
            project_id, result.old_hash[:8], result.new_hash[:8], alert_sent,
        )

    return JSONResponse(content={
        **result.to_dict(),
        "alert_sent": alert_sent,
    })


@router.get("/api/scope-watch/{project_id}/status")
def watcher_status(project_id: str) -> JSONResponse:
    """Return the current state of a watcher."""
    w = _get_watcher(project_id)
    return JSONResponse(content=w.status())


@router.get("/api/scope-watch")
def list_watchers() -> JSONResponse:
    """List all configured watchers and their states."""
    from src.analysis.scope_watcher import list_watchers as _lw
    watchers = _lw()
    return JSONResponse(content={"watchers": watchers, "count": len(watchers)})


@router.post("/api/scope-watch/{project_id}/recipients")
def set_recipients(project_id: str, req: RecipientsRequest) -> JSONResponse:
    """Update the notification recipient list for a watcher."""
    w = _get_watcher(project_id)
    w.set_recipients(req.emails)
    return JSONResponse(content={
        "project_id": project_id,
        "notify": req.emails,
    })
