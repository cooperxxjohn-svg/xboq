"""
xBOQ.ai — Celery tasks.

Each task wraps the existing pipeline thread function so the logic
lives in one place. Celery handles retry, ACK, and crash recovery.

Tasks
-----
run_pipeline_task   POST /api/analyze — analyse a tender document set
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import Celery app — if celery isn't installed this module is never used
# ---------------------------------------------------------------------------

from src.api.worker import celery_app  # noqa: E402  (celery may not be installed)


# ---------------------------------------------------------------------------
# Shared task implementation
# ---------------------------------------------------------------------------

def _run(self, job_id: str, input_paths: List[str], excel_paths: List[str],
         run_mode: str, project_name: str, job_dir: str, org_id: str = "local") -> dict:
    """Shared pipeline execution used by both fast and full task variants."""
    try:
        from src.api.routes.analyze import _run_pipeline_in_thread
        _run_pipeline_in_thread(
            job_id=job_id,
            input_paths=[Path(p) for p in input_paths],
            excel_paths=[Path(p) for p in excel_paths],
            run_mode=run_mode,
            project_name=project_name,
            job_dir=Path(job_dir),
            org_id=org_id,
        )
        return {"job_id": job_id, "status": "complete"}
    except Exception as exc:
        logger.exception("Pipeline task failed for job %s: %s", job_id, exc)
        try:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            return {"job_id": job_id, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Fast queue — demo_fast and quick modes (soft 14m / hard 15m)
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="xboq.run_pipeline_fast",
    max_retries=1,
    default_retry_delay=15,
    time_limit=900,       # 15 min hard kill
    soft_time_limit=840,  # 14 min soft warning
    queue="xboq_fast",
)
def run_pipeline_fast_task(
    self,
    job_id: str,
    input_paths: List[str],
    excel_paths: List[str],
    run_mode: str,
    project_name: str,
    job_dir: str,
    org_id: str = "local",
) -> dict:
    """Fast queue task for demo_fast / interactive runs."""
    return _run(self, job_id, input_paths, excel_paths, run_mode, project_name, job_dir, org_id)


# ---------------------------------------------------------------------------
# Full queue — standard_review and full_audit modes (soft 55m / hard 60m)
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="xboq.run_pipeline_full",
    max_retries=2,
    default_retry_delay=30,
    queue="xboq_full",
)
def run_pipeline_full_task(
    self,
    job_id: str,
    input_paths: List[str],
    excel_paths: List[str],
    run_mode: str,
    project_name: str,
    job_dir: str,
    org_id: str = "local",
) -> dict:
    """Full queue task for standard_review / full_audit runs."""
    return _run(self, job_id, input_paths, excel_paths, run_mode, project_name, job_dir, org_id)


# ---------------------------------------------------------------------------
# Legacy task — preserved for backward compatibility with in-flight jobs
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="xboq.run_pipeline",
    max_retries=2,
    default_retry_delay=30,  # seconds between retries
)
def run_pipeline_task(
    self,
    job_id: str,
    input_paths: List[str],    # Path objects serialised to strings for JSON transport
    excel_paths: List[str],
    run_mode: str,
    project_name: str,
    job_dir: str,
    org_id: str = "local",
) -> dict:
    """
    Legacy pipeline task — preserved for backward compatibility with in-flight jobs.
    New submissions use run_pipeline_fast_task or run_pipeline_full_task.
    """
    return _run(self, job_id, input_paths, excel_paths, run_mode, project_name, job_dir, org_id)
