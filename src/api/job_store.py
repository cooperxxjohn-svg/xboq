"""
Job store for xBOQ.ai API.

The singleton `job_store` is now backed by SQLite via JobDB (persistent across
restarts).  The module keeps its original public surface for full backward
compatibility:

  - job_store.create_job(job_id)      → PersistentJob
  - job_store.get_job(job_id)         → Optional[PersistentJob]  (no payload)
  - job_store.update_job(job_id, ...) → None
  - job_store.list_jobs()             → List[PersistentJob]
  - get_job(job_id)                   → Optional[dict]  (free fn, checks _store)

New additions:
  - job_store.get_job_with_payload(job_id) → Optional[PersistentJob]
  - job_store.list_jobs_by_org(org_id, limit, offset)
  - Job                          re-exported for callers that import it directly
  - OUTPUTS_DIR / job_db         re-exported for callers that need the path
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.job_db import JobDB, PersistentJob, job_db, OUTPUTS_DIR

# Backward-compatibility alias — old code imports "from src.api.job_store import JobStore"
JobStore = JobDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export Job as an alias so existing imports (from src.api.job_store import Job)
# continue to work unchanged.
# ---------------------------------------------------------------------------
Job = PersistentJob


# ---------------------------------------------------------------------------
# Singleton — backed by SQLite
# ---------------------------------------------------------------------------

job_store: JobDB = job_db


# ---------------------------------------------------------------------------
# Flat dict store (used by line-items routes and tests that inject payloads
# directly without going through the full pipeline).
#
# On a cache miss here, get_job() falls back to loading from disk so that
# jobs persisted in previous server sessions are still accessible.
# ---------------------------------------------------------------------------

_store: Dict[str, dict] = {}
_store_lock: threading.Lock = threading.Lock()  # guards all accesses to _store


def get_job(job_id: str) -> Optional[dict]:
    """
    Return a plain-dict job record compatible with line-items routes.

    Lookup order:
      1. In-memory flat dict (populated when pipeline completes in this session)
      2. SQLite + disk payload (jobs from previous server sessions)

    Thread-safe: all _store reads/writes are guarded by _store_lock.
    """
    # 1. Fast path: in-memory store populated by the current session's pipeline
    with _store_lock:
        if job_id in _store:
            return _store[job_id]

    # 2. Persistent store — load metadata + payload from disk (outside lock to
    #    avoid holding the lock during potentially slow disk I/O)
    job = job_store.get_job_with_payload(job_id)
    if job is None:
        return None

    record = {
        "job_id": job.job_id,
        "status": job.status,
        "result": job.payload or {},
    }

    # Cache in memory so subsequent calls in this session are fast
    if job.status == "complete" and job.payload:
        with _store_lock:
            _store[job_id] = record

    return record


def set_job(job_id: str, record: dict) -> None:
    """Thread-safe write to the in-memory flat store."""
    with _store_lock:
        _store[job_id] = record
