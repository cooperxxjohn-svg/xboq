"""
SQLAlchemy-backed persistent job store for xBOQ.ai.

Replaces the previous raw-sqlite3 implementation. Supports both SQLite
(local dev) and PostgreSQL (production) transparently via DATABASE_URL.

Same public interface as before — all existing callers work unchanged.

Storage layout
--------------
  DATABASE_URL                               <- job metadata (SQLAlchemy)
  $XBOQ_JOBS_DIR/{job_id}/                   <- job files (PDFs, output/)
  $XBOQ_JOBS_DIR/{job_id}/payload.json       <- analysis result (on completion)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

_XBOQ_HOME = Path(os.environ.get("XBOQ_HOME", str(Path.home() / ".xboq")))
OUTPUTS_DIR = Path(os.environ.get("XBOQ_JOBS_DIR", str(_XBOQ_HOME / "job_outputs")))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_JOB_TTL_DAYS = int(os.environ.get("XBOQ_JOB_TTL_DAYS", "90"))

# ---------------------------------------------------------------------------
# PersistentJob dataclass (public API — unchanged)
# ---------------------------------------------------------------------------

@dataclass
class PersistentJob:
    """Job record returned from the store. All fields have safe defaults."""
    job_id:           str
    status:           str           = "queued"
    org_id:           str           = "local"
    project_name:     str           = ""
    run_mode:         str           = "demo_fast"
    created_at:       datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at:     Optional[datetime] = None
    progress:         float         = 0.0
    progress_message: str           = ""
    errors:           List[str]     = field(default_factory=list)
    output_files:     Dict[str, str] = field(default_factory=dict)
    payload_path:     str           = ""
    queue_position:   int           = 0
    payload:          Optional[dict] = None   # not persisted; loaded on demand


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_to_job(row, payload: Optional[dict] = None) -> PersistentJob:
    return PersistentJob(
        job_id=row.job_id,
        status=row.status,
        org_id=row.org_id,
        project_name=row.project_name or "",
        run_mode=row.run_mode or "demo_fast",
        created_at=row.created_at or datetime.now(timezone.utc),
        completed_at=row.completed_at,
        progress=float(row.progress or 0.0),
        progress_message=row.progress_message or "",
        errors=row.errors,
        output_files=row.output_files,
        payload_path=row.payload_path or "",
        queue_position=int(row.queue_position or 0),
        payload=payload,
    )


def _payload_path_for(job_id: str) -> Path:
    return OUTPUTS_DIR / job_id / "payload.json"


_SENTINEL = object()


# ---------------------------------------------------------------------------
# JobDB
# ---------------------------------------------------------------------------

class JobDB:
    """
    SQLAlchemy-backed job store.

    Supports SQLite (dev) and PostgreSQL (prod) via DATABASE_URL env var.
    Thread-safe: each call opens a short-lived session from the pool.

    Parameters
    ----------
    db_path : Path or str, optional
        If provided, creates a private SQLite engine for this instance.
        Used by tests for isolation.  Production code leaves this unset
        and uses the shared engine from src.api.db.
    """

    OUTPUTS_DIR = OUTPUTS_DIR

    def __init__(self, db_path=None) -> None:
        self._payload_cache: Dict[str, dict] = {}
        self._private_session_factory = None

        if db_path is not None:
            # Per-instance engine for test isolation
            from sqlalchemy import create_engine, event
            from sqlalchemy.orm import sessionmaker
            _url = f"sqlite:///{db_path}"
            _eng = create_engine(_url, connect_args={"check_same_thread": False})

            @event.listens_for(_eng, "connect")
            def _set_pragmas(dbapi_conn, _cr):
                c = dbapi_conn.cursor()
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA foreign_keys=ON")
                c.close()

            from src.api.models import Base
            Base.metadata.create_all(bind=_eng)
            self._private_session_factory = sessionmaker(
                autocommit=False, autoflush=False, bind=_eng
            )
        else:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        try:
            from src.api.db import init_db
            init_db()
        except Exception as exc:
            logger.error("Failed to initialise database: %s", exc)

    def _session(self):
        if self._private_session_factory is not None:
            return self._private_session_factory()
        from src.api.db import SessionLocal
        return SessionLocal()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        org_id: str = "local",
        project_name: str = "",
        run_mode: str = "demo_fast",
    ) -> PersistentJob:
        from src.api.models import JobModel
        now = datetime.now(timezone.utc)
        with self._session() as db:
            if db.get(JobModel, job_id) is None:
                db.add(JobModel(
                    job_id=job_id, status="queued", org_id=org_id,
                    project_name=project_name, run_mode=run_mode,
                    created_at=now, errors_json="[]", output_files_json="{}",
                ))
                db.commit()
        (OUTPUTS_DIR / job_id).mkdir(parents=True, exist_ok=True)
        return PersistentJob(job_id=job_id, org_id=org_id,
                             project_name=project_name, run_mode=run_mode,
                             created_at=now)

    def get_job(self, job_id: str) -> Optional[PersistentJob]:
        from src.api.models import JobModel
        with self._session() as db:
            row = db.get(JobModel, job_id)
            if row is None:
                return None
            return _model_to_job(row, payload=self._payload_cache.get(job_id))

    def get_job_with_payload(self, job_id: str) -> Optional[PersistentJob]:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.payload is None and job.payload_path:
            path = Path(job.payload_path)
            if path.exists():
                try:
                    job.payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Failed to load payload for %s: %s", job_id, exc)
        return job

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        from src.api.models import JobModel

        payload = kwargs.pop("payload", _SENTINEL)
        if payload is not _SENTINEL and payload is not None:
            pp = _payload_path_for(job_id)
            pp.parent.mkdir(parents=True, exist_ok=True)
            try:
                pp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
                kwargs["payload_path"] = str(pp)
            except Exception as exc:
                logger.error("Failed to write payload for job %s: %s", job_id, exc)
            self._payload_cache[job_id] = payload

        if not kwargs:
            return

        with self._session() as db:
            row = db.get(JobModel, job_id)
            if row is None:
                logger.warning("update_job: job %s not found", job_id)
                return
            for key, val in kwargs.items():
                if key == "errors":
                    row.errors = val
                elif key == "output_files":
                    row.output_files = val
                elif hasattr(row, key):
                    setattr(row, key, val)
                else:
                    logger.debug("update_job: unknown field %s", key)
            db.commit()

    def list_jobs(self) -> List[PersistentJob]:
        from src.api.models import JobModel
        from sqlalchemy import select, desc
        with self._session() as db:
            rows = db.execute(select(JobModel).order_by(desc(JobModel.created_at))).scalars().all()
            return [_model_to_job(r) for r in rows]

    def list_jobs_by_org(self, org_id: str, limit: int = 20, offset: int = 0) -> List[PersistentJob]:
        from src.api.models import JobModel
        from sqlalchemy import select, desc
        with self._session() as db:
            rows = db.execute(
                select(JobModel)
                .where(JobModel.org_id == org_id)
                .order_by(desc(JobModel.created_at))
                .limit(limit).offset(offset)
            ).scalars().all()
            return [_model_to_job(r) for r in rows]

    def expire_old_jobs(self, ttl_days: int = 0) -> int:
        """
        Delete jobs older than ttl_days (default: XBOQ_JOB_TTL_DAYS env var, default 90).

        Also removes the corresponding output directory from disk so files don't
        accumulate indefinitely. Returns the number of jobs deleted.
        """
        import shutil
        from datetime import timedelta
        from src.api.models import JobModel
        from sqlalchemy import select, delete as sa_delete
        days = ttl_days or _JOB_TTL_DAYS
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with self._session() as db:
            # Collect job_ids before deletion so we can clean up disk
            old_ids = db.execute(
                select(JobModel.job_id).where(JobModel.created_at < cutoff)
            ).scalars().all()

            if not old_ids:
                return 0

            result = db.execute(sa_delete(JobModel).where(JobModel.created_at < cutoff))
            db.commit()
            deleted = result.rowcount

        # Remove output directories for deleted jobs
        removed_dirs = 0
        for job_id in old_ids:
            job_dir = OUTPUTS_DIR / job_id
            if job_dir.exists():
                try:
                    shutil.rmtree(job_dir)
                    removed_dirs += 1
                except Exception as exc:
                    logger.warning("Failed to remove job dir %s: %s", job_dir, exc)

        if deleted:
            logger.info("Job TTL cleanup: removed %d jobs (%d output dirs) older than %d days",
                        deleted, removed_dirs, days)
        return deleted

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued or processing job.

        Sets status to "cancelled" and records the timestamp.
        Returns True if the job existed and was cancellable, False otherwise.
        Running pipeline threads are not forcibly killed — they will detect the
        "cancelled" status on next progress update and stop gracefully.
        """
        from src.api.models import JobModel
        from sqlalchemy import select
        with self._session() as db:
            row = db.execute(
                select(JobModel).where(JobModel.job_id == job_id)
            ).scalar_one_or_none()
            if row is None:
                return False
            if row.status in ("complete", "error", "cancelled"):
                return False  # already terminal
            row.status = "cancelled"
            row.completed_at = datetime.now(timezone.utc)
            row.progress_message = "Cancelled by user"
            db.commit()
            logger.info("Job %s cancelled", job_id)
        return True

    def job_output_dir(self, job_id: str) -> Path:
        d = OUTPUTS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

job_db: JobDB = JobDB()
