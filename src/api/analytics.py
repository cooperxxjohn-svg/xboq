"""
Lightweight event analytics — logs to ~/.xboq/analytics.jsonl (JSONL format).

Usage:
    from src.api.analytics import log_event
    log_event("analysis_started", job_id="abc123", org_id="acme", tenant_id="t1", extra={"run_mode": "full_audit"})

Events we instrument (10 key events):
    analysis_started        — job submitted
    analysis_complete       — job finished successfully
    analysis_failed         — job errored
    report_viewed           — GET /api/jobs/{id} called for a complete job
    report_shared           — GET /api/jobs/{id}/report called (shareable link)
    feedback_submitted      — estimator submitted a correction
    rfi_feedback            — thumbs up/down on an RFI
    rate_override_set       — org rate override saved
    export_downloaded       — export file downloaded (excel/pdf/word)
    onboarding_completed    — first upload after zero-job state
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _analytics_path() -> Path:
    """Resolve the analytics JSONL file path.

    Priority:
      1. XBOQ_ANALYTICS_FILE env var (override)
      2. ~/.xboq/analytics.jsonl (default)
    """
    override = os.environ.get("XBOQ_ANALYTICS_FILE", "")
    if override:
        return Path(override)
    return Path.home() / ".xboq" / "analytics.jsonl"


def log_event(event_name: str, **kwargs) -> None:
    """Append a JSONL analytics event.

    Args:
        event_name: Name of the event (e.g. "analysis_started").
        **kwargs:   Arbitrary key-value pairs (job_id, org_id, tenant_id, extra, etc.).

    Non-blocking — all exceptions are caught silently so callers never crash.
    """
    try:
        path = _analytics_path()

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        record: dict = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event_name,
        }
        record.update(kwargs)

        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    except Exception:
        # Never crash the caller — analytics are best-effort
        pass
