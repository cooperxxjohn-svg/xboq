"""
Project history store — saves pipeline results per org/user in Supabase.

Table schema (run migrate.sql to create):
    projects (
        id          uuid primary key default gen_random_uuid(),
        org_id      text not null,
        user_id     uuid references auth.users,
        filename    text,
        created_at  timestamptz default now(),
        summary     jsonb,   -- line_items_summary, qto_summary, etc.
        payload     jsonb    -- full pipeline payload (large)
    )
"""
from __future__ import annotations
import json
from typing import Optional, List, Dict
from .supabase_client import get_client

_TABLE = "projects"


def save_project(
    org_id: str,
    user_id: str,
    filename: str,
    summary: dict,
    payload: dict,
) -> Optional[str]:
    """
    Save a pipeline result. Returns project id or None on failure.
    Strips large binary fields before saving.
    """
    sb = get_client()
    if not sb:
        return None
    # Trim payload to avoid Supabase 1MB row limit
    try:
        from src.analysis.payload_pruner import prune_for_supabase
        slim_payload = prune_for_supabase(payload)
    except ImportError:
        slim_payload = {
            k: v for k, v in payload.items()
            if k not in ("raw_text", "all_page_texts", "page_images")
        }
    try:
        resp = sb.table(_TABLE).insert({
            "org_id":   org_id,
            "user_id":  user_id,
            "filename": filename,
            "summary":  summary,
            "payload":  slim_payload,
        }).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception:
        return None


def list_projects(org_id: str, limit: int = 20) -> List[Dict]:
    """List recent projects for an org."""
    sb = get_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table(_TABLE)
            .select("id, filename, created_at, summary")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def get_project(project_id: str, org_id: str) -> Optional[Dict]:
    """Fetch a single project (with full payload)."""
    sb = get_client()
    if not sb:
        return None
    try:
        resp = (
            sb.table(_TABLE)
            .select("*")
            .eq("id", project_id)
            .eq("org_id", org_id)
            .single()
            .execute()
        )
        return resp.data
    except Exception:
        return None
