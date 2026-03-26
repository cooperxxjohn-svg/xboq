"""
Collaboration REST endpoints.

GET  /api/collab/{job_id}                         — list all entries
POST /api/collab/{job_id}/comment                 — add a comment
POST /api/collab/{job_id}/assign                  — assign entity to user
POST /api/collab/{job_id}/sign-off                — sign off on entity
GET  /api/collab/{job_id}/thread/{type}/{id}      — get entity thread
GET  /api/collab/{job_id}/pending/{user}          — user's pending assignments
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.job_store import job_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["collaboration"])


def _get_store(job_id: str):
    """Return a CollabStore for the given job's output directory."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    from src.analysis.collab_store import CollabStore
    # Use the job's output dir if available, else default
    out_dir = None
    if job.output_files:
        # Any output file gives us the directory
        first_path = next(iter(job.output_files.values()), None)
        if first_path:
            out_dir = Path(first_path).parent
    return CollabStore(project_id=job_id, project_dir=out_dir)


class CommentRequest(BaseModel):
    entity_type: str
    entity_id: str
    text: str
    author: Optional[str] = "anonymous"


class AssignRequest(BaseModel):
    entity_type: str
    entity_id: str
    assigned_to: str
    author: Optional[str] = "anonymous"


class SignOffRequest(BaseModel):
    entity_type: str
    entity_id: str
    role: Optional[str] = "estimator"
    author: Optional[str] = "anonymous"


class StatusChangeRequest(BaseModel):
    entity_type: str
    entity_id: str
    new_status: str
    old_status: Optional[str] = ""
    author: Optional[str] = "anonymous"


@router.get("/api/collab/{job_id}")
def list_entries(job_id: str) -> JSONResponse:
    """Return all collaboration entries for a job."""
    store = _get_store(job_id)
    return JSONResponse(content={
        "job_id":       job_id,
        "entries":      store.all(),
        "collaborators": store.collaborators(),
        "total":        len(store.all()),
    })


@router.post("/api/collab/{job_id}/comment")
def add_comment(job_id: str, req: CommentRequest) -> JSONResponse:
    """Add a text comment to an entity (RFI, BOQ item, blocker, etc.)."""
    store = _get_store(job_id)
    entry = store.add_comment(req.entity_type, req.entity_id,
                              req.text, author=req.author or "")
    return JSONResponse(content={"entry": entry}, status_code=201)


@router.post("/api/collab/{job_id}/assign")
def assign_entity(job_id: str, req: AssignRequest) -> JSONResponse:
    """Assign an entity to a team member."""
    store = _get_store(job_id)
    entry = store.assign(req.entity_type, req.entity_id,
                         req.assigned_to, author=req.author or "")
    return JSONResponse(content={"entry": entry}, status_code=201)


@router.post("/api/collab/{job_id}/sign-off")
def sign_off(job_id: str, req: SignOffRequest) -> JSONResponse:
    """Record a sign-off on an entity."""
    store = _get_store(job_id)
    entry = store.add_sign_off(req.entity_type, req.entity_id,
                               role=req.role or "estimator",
                               author=req.author or "")
    return JSONResponse(content={"entry": entry}, status_code=201)


@router.post("/api/collab/{job_id}/status")
def change_status(job_id: str, req: StatusChangeRequest) -> JSONResponse:
    """Record a status change on an entity."""
    store = _get_store(job_id)
    entry = store.change_status(req.entity_type, req.entity_id,
                                req.new_status, old_status=req.old_status or "",
                                author=req.author or "")
    return JSONResponse(content={"entry": entry}, status_code=201)


@router.get("/api/collab/{job_id}/thread/{entity_type}/{entity_id}")
def get_thread(job_id: str, entity_type: str, entity_id: str) -> JSONResponse:
    """Return full collaboration thread for a specific entity."""
    store = _get_store(job_id)
    thread = store.get_thread(entity_type, entity_id)
    return JSONResponse(content=thread)


@router.get("/api/collab/{job_id}/pending/{username}")
def pending_assignments(job_id: str, username: str) -> JSONResponse:
    """Return all items currently assigned to a specific user."""
    store = _get_store(job_id)
    items = store.pending_assignments(username)
    return JSONResponse(content={"job_id": job_id, "assigned_to": username,
                                  "items": items, "total": len(items)})
