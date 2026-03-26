"""
Collaboration Store — multi-user annotation layer on top of the existing
append-only JSONL collaboration system.

Extends src/analysis/collaboration.py with:
    - User identity tracking (author, org_id)
    - Supabase realtime broadcasting (optional — gracefully absent)
    - Thread-safe in-memory cache for the current session
    - REST-friendly serialisation

Usage:
    from src.analysis.collab_store import CollabStore

    store = CollabStore(project_id="proj_abc", project_dir=Path("out/proj_abc"))

    # Add a comment on an RFI
    store.add_comment("rfi", "RFI-003", "Need 200mm slab confirmed", author="alice")

    # Assign RFI to Bob
    store.assign("rfi", "RFI-003", assigned_to="bob@contractor.com", author="alice")

    # Get thread for one RFI
    thread = store.get_thread("rfi", "RFI-003")

    # Get all annotations as list-of-dicts (for REST API)
    all_entries = store.all()
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CollabStore:
    """
    Thread-safe collaboration store for a single project.

    Persists entries to JSONL via the existing collaboration.py helpers
    and optionally broadcasts to Supabase realtime.
    """

    def __init__(
        self,
        project_id: str,
        project_dir: Optional[Path] = None,
        org_id: str = "",
    ) -> None:
        self.project_id = project_id
        self.org_id = org_id
        self._dir = project_dir or (Path.home() / ".xboq" / "projects" / project_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # RLock allows re-entry from same thread
        self._cache: List[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal load / save
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from src.analysis.collaboration import load_collaboration
            entries = load_collaboration(self._dir)
            with self._lock:
                self._cache = entries
                self._loaded = True
        except Exception as exc:
            logger.debug("CollabStore._ensure_loaded failed: %s", exc)
            with self._lock:
                self._cache = []
                self._loaded = True

    def _persist(self, entry: dict) -> None:
        try:
            from src.analysis.collaboration import append_collaboration
            append_collaboration(entry, self._dir)
        except Exception as exc:
            logger.warning("CollabStore._persist failed: %s", exc)

    def _broadcast(self, entry: dict) -> None:
        """Optionally push to Supabase realtime (non-blocking, best-effort)."""
        try:
            from src.auth.supabase_client import get_client, is_configured
            if not is_configured():
                return
            client = get_client()
            if client is None:
                return
            table = client.table("collaboration_entries")
            table.insert({
                "project_id": self.project_id,
                "org_id":     self.org_id,
                **entry,
            }).execute()
        except Exception:
            pass  # Supabase not configured — silent

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def _add_entry(
        self,
        entity_type: str,
        entity_id: str,
        action_type: str,
        data: dict,
        author: str = "",
    ) -> dict:
        from src.analysis.collaboration import make_collaboration_entry
        entry = make_collaboration_entry(
            entity_type=entity_type,
            entity_id=entity_id,
            action_type=action_type,
            data=data,
            author=author or "anonymous",
        )
        # Override timestamp to UTC
        entry["timestamp"] = _utcnow()
        with self._lock:
            self._ensure_loaded()
            self._cache.append(entry)
        self._persist(entry)
        self._broadcast(entry)
        return entry

    def add_comment(
        self,
        entity_type: str,
        entity_id: str,
        text: str,
        author: str = "",
    ) -> dict:
        """Add a text comment to an RFI, BOQ item, blocker, or quantity."""
        return self._add_entry(entity_type, entity_id, "comment",
                               {"text": text}, author=author)

    def assign(
        self,
        entity_type: str,
        entity_id: str,
        assigned_to: str,
        author: str = "",
    ) -> dict:
        """Assign an item to a team member."""
        return self._add_entry(entity_type, entity_id, "assign",
                               {"assigned_to": assigned_to}, author=author)

    def set_due_date(
        self,
        entity_type: str,
        entity_id: str,
        due_date: str,          # ISO date string
        author: str = "",
    ) -> dict:
        """Set a due date on an item."""
        return self._add_entry(entity_type, entity_id, "due_date",
                               {"due_date": due_date}, author=author)

    def change_status(
        self,
        entity_type: str,
        entity_id: str,
        new_status: str,
        old_status: str = "",
        author: str = "",
    ) -> dict:
        """Record a status transition (e.g., draft → approved)."""
        return self._add_entry(entity_type, entity_id, "status_change",
                               {"old_status": old_status, "new_status": new_status},
                               author=author)

    def add_sign_off(
        self,
        entity_type: str,
        entity_id: str,
        role: str = "estimator",
        author: str = "",
    ) -> dict:
        """Record a sign-off (e.g., BOQ item approved by estimator)."""
        return self._add_entry(entity_type, entity_id, "sign_off",
                               {"role": role, "signed_at": _utcnow()}, author=author)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def all(self) -> List[dict]:
        """Return all collaboration entries for this project."""
        self._ensure_loaded()
        with self._lock:
            return list(self._cache)

    def get_thread(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """
        Return aggregated thread for a single entity.

        Returns:
            {
                "entity_type": str,
                "entity_id": str,
                "comments": [{"author", "text", "timestamp"}],
                "assigned_to": str,
                "due_date": str,
                "status_history": [{"author", "old", "new", "timestamp"}],
                "sign_offs": [{"author", "role", "timestamp"}],
            }
        """
        try:
            from src.analysis.collaboration import get_entity_collaboration
            entries = self.all()
            base = get_entity_collaboration(entries, entity_type, entity_id)
        except Exception:
            base = {"comments": [], "assigned_to": "", "due_date": ""}

        # Augment with sign-offs and status history
        all_for_entity = [
            e for e in self.all()
            if e.get("entity_type") == entity_type and e.get("entity_id") == entity_id
        ]

        status_history = []
        sign_offs = []
        for e in all_for_entity:
            if e.get("action_type") == "status_change":
                d = e.get("data", {})
                status_history.append({
                    "author":    e.get("author", ""),
                    "old":       d.get("old_status", ""),
                    "new":       d.get("new_status", ""),
                    "timestamp": e.get("timestamp", ""),
                })
            elif e.get("action_type") == "sign_off":
                d = e.get("data", {})
                sign_offs.append({
                    "author":     e.get("author", ""),
                    "role":       d.get("role", ""),
                    "signed_at":  d.get("signed_at", e.get("timestamp", "")),
                })

        base["entity_type"]    = entity_type
        base["entity_id"]      = entity_id
        base["status_history"] = status_history
        base["sign_offs"]      = sign_offs
        return base

    def collaborators(self) -> List[str]:
        """Return deduplicated list of authors who have contributed."""
        entries = self.all()
        seen: set = set()
        result = []
        for e in entries:
            a = e.get("author", "")
            if a and a not in seen:
                seen.add(a)
                result.append(a)
        return result

    def pending_assignments(self, assigned_to: str) -> List[dict]:
        """Return all items currently assigned to a given user."""
        try:
            from src.analysis.collaboration import get_all_assignments
            return [
                a for a in get_all_assignments(self.all())
                if a.get("assigned_to") == assigned_to
            ]
        except Exception:
            return []
