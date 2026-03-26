"""
Scope Change Alert Engine.

Monitors a tender URL or a local directory for addenda / revised PDFs.
When new content is detected, it can:
    1. Log a delta entry to the project's collaboration JSONL
    2. Send an email notification (via src/notifications/email_sender.py)
    3. Optionally trigger a re-run callback (caller provides it)

Storage: `~/.xboq/watchers/<watch_id>.json`
    Tracks: url/path, last_seen_hash, last_checked, notification recipients

Usage:
    from src.analysis.scope_watcher import ScopeWatcher

    watcher = ScopeWatcher("hosp_tender")
    watcher.set_url("https://cpwd.gov.in/tenders/hosp_boq_v2.pdf",
                    notify=["bids@myco.com"])
    result = watcher.check()  # → ScopeCheckResult(changed=True, ...)
    if result.changed:
        watcher.send_alert(result, project_name="Hospital Block")

Scheduled polling:
    # Call watcher.check() from a cron job / background thread
    # API: POST /api/scope-watch/{project_id}/check
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_WATCHER_DIR = Path.home() / ".xboq" / "watchers"
_WATCHER_DIR.mkdir(parents=True, exist_ok=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScopeCheckResult:
    """Result of a single scope check."""
    project_id: str
    changed: bool
    source: str              # URL or file path
    old_hash: str
    new_hash: str
    checked_at: str = field(default_factory=_utcnow)
    error: str = ""
    bytes_downloaded: int = 0

    def to_dict(self) -> dict:
        return {
            "project_id":       self.project_id,
            "changed":          self.changed,
            "source":           self.source,
            "old_hash":         self.old_hash,
            "new_hash":         self.new_hash,
            "checked_at":       self.checked_at,
            "error":            self.error,
            "bytes_downloaded": self.bytes_downloaded,
        }


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _fetch_url_bytes(url: str, timeout: int = 30) -> bytes:
    """Download URL content; raise on error."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "xBOQ-ScopeWatcher/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        raise IOError(f"Failed to fetch {url}: {exc}") from exc


def _hash_dir(path: Path) -> str:
    """Stable hash of all file mtimes + names in a directory."""
    if not path.is_dir():
        return ""
    items = sorted(
        (f.name, f.stat().st_mtime) for f in path.rglob("*") if f.is_file()
    )
    return _hash_bytes(json.dumps(items).encode())


# ---------------------------------------------------------------------------
# ScopeWatcher
# ---------------------------------------------------------------------------

class ScopeWatcher:
    """
    Watch a URL or local directory for scope changes.

    State is persisted to `~/.xboq/watchers/<project_id>.json`.
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._state_path = _WATCHER_DIR / f"{project_id}.json"
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text("utf-8"))
            except Exception:
                pass
        return {
            "project_id":    self.project_id,
            "source":        "",
            "source_type":   "",   # "url" | "dir"
            "last_hash":     "",
            "last_checked":  "",
            "notify":        [],
            "check_count":   0,
            "change_count":  0,
        }

    def _save_state(self) -> None:
        self._state_path.write_text(
            json.dumps(self._state, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_url(self, url: str, notify: Optional[List[str]] = None) -> None:
        """Configure watcher to monitor a URL."""
        self._state["source"] = url
        self._state["source_type"] = "url"
        if notify:
            self._state["notify"] = notify
        self._save_state()

    def set_dir(self, path: str, notify: Optional[List[str]] = None) -> None:
        """Configure watcher to monitor a local directory."""
        self._state["source"] = path
        self._state["source_type"] = "dir"
        if notify:
            self._state["notify"] = notify
        self._save_state()

    def set_recipients(self, emails: List[str]) -> None:
        self._state["notify"] = emails
        self._save_state()

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(self) -> ScopeCheckResult:
        """
        Check the configured source for changes.

        Returns ScopeCheckResult.changed=True if content changed since
        the last check. Updates internal state.
        """
        source = self._state.get("source", "")
        if not source:
            return ScopeCheckResult(
                self.project_id, changed=False, source="",
                old_hash="", new_hash="", error="No source configured"
            )

        old_hash = self._state.get("last_hash", "")
        new_hash = ""
        error = ""
        bytes_dl = 0

        source_type = self._state.get("source_type", "url")
        try:
            if source_type == "dir":
                new_hash = _hash_dir(Path(source))
            else:
                data = _fetch_url_bytes(source)
                new_hash = _hash_bytes(data)
                bytes_dl = len(data)
        except Exception as exc:
            error = str(exc)
            logger.warning("ScopeWatcher.check failed for %s: %s", source, exc)

        changed = bool(new_hash and new_hash != old_hash and old_hash)
        # First check — record hash but don't flag as changed
        first_check = not old_hash

        self._state["last_hash"]    = new_hash or old_hash
        self._state["last_checked"] = _utcnow()
        self._state["check_count"]  = self._state.get("check_count", 0) + 1
        if changed:
            self._state["change_count"] = self._state.get("change_count", 0) + 1
        self._save_state()

        return ScopeCheckResult(
            project_id=self.project_id,
            changed=changed and not first_check,
            source=source,
            old_hash=old_hash,
            new_hash=new_hash,
            error=error,
            bytes_downloaded=bytes_dl,
        )

    # ------------------------------------------------------------------
    # Alert
    # ------------------------------------------------------------------

    def send_alert(
        self,
        result: ScopeCheckResult,
        project_name: str = "",
        rerun_callback: Optional[Callable] = None,
    ) -> bool:
        """
        Send scope-change notification email and/or call the re-run callback.

        Returns True if email was sent (or log-only succeeded).
        """
        recipients = self._state.get("notify", [])
        sent = False

        if recipients:
            subject = f"Scope Change Detected — {project_name or self.project_id}"
            body = (
                f"A change was detected in the tender documents for "
                f"{project_name or self.project_id}.\n\n"
                f"Source: {result.source}\n"
                f"Previous hash: {result.old_hash}\n"
                f"New hash:      {result.new_hash}\n"
                f"Detected at:   {result.checked_at}\n\n"
                "Please re-review the tender documents and rerun the xBOQ analysis "
                "to capture the updated scope.\n\n"
                "— xBOQ Scope Watcher"
            )
            try:
                from src.notifications.email_sender import EmailMessage, send_email
                msg = EmailMessage(to=recipients, subject=subject, body=body)
                r = send_email(msg)
                sent = r.success
            except Exception as exc:
                logger.warning("scope_watcher alert email failed: %s", exc)

        if rerun_callback and callable(rerun_callback):
            try:
                rerun_callback(self.project_id, result)
            except Exception as exc:
                logger.warning("scope_watcher rerun_callback failed: %s", exc)

        return sent

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "project_id":   self.project_id,
            "source":       self._state.get("source", ""),
            "source_type":  self._state.get("source_type", ""),
            "last_hash":    self._state.get("last_hash", ""),
            "last_checked": self._state.get("last_checked", ""),
            "notify":       self._state.get("notify", []),
            "check_count":  self._state.get("check_count", 0),
            "change_count": self._state.get("change_count", 0),
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (for API endpoints)
# ---------------------------------------------------------------------------

def list_watchers() -> List[dict]:
    """Return status for all configured watchers."""
    results = []
    for p in sorted(_WATCHER_DIR.glob("*.json")):
        try:
            state = json.loads(p.read_text("utf-8"))
            results.append(state)
        except Exception:
            pass
    return results
