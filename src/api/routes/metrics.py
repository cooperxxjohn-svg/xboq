"""
GET /api/metrics — Prometheus-format metrics endpoint.

Exposes operational counters that L&T's (or any enterprise) ops team can
scrape with Prometheus and display in Grafana.

Metrics exposed
---------------
xboq_jobs_total{status}         — cumulative job count by status
xboq_jobs_active                — jobs currently in processing state
xboq_queue_depth                — jobs waiting in executor/Celery queue
xboq_pipeline_duration_seconds  — last 100 job durations (histogram)
xboq_errors_total               — cumulative pipeline error count
xboq_api_requests_total         — cumulative HTTP request count (from middleware)
xboq_uptime_seconds             — seconds since API startup
xboq_worker_mode                — "celery" or "threadpool"

Usage with Prometheus:
  # prometheus.yml
  scrape_configs:
    - job_name: xboq
      static_configs:
        - targets: ['api:8000']
      metrics_path: /api/metrics
"""

from __future__ import annotations

import time
import logging
from typing import Dict

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])

# ---------------------------------------------------------------------------
# Startup time (set once on import)
# ---------------------------------------------------------------------------

_STARTUP_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory counters (lightweight — no prometheus_client dep required)
# Incremented by the analyze route and job store.
# ---------------------------------------------------------------------------

_counters: Dict[str, int] = {
    "requests_total": 0,
    "jobs_queued": 0,
    "jobs_complete": 0,
    "jobs_error": 0,
    "jobs_processing": 0,
}

# Ring buffer for job durations (last 100 jobs)
_durations: list[float] = []
_MAX_DURATION_SAMPLES = 100


def record_request() -> None:
    """Call from request middleware to increment request counter."""
    _counters["requests_total"] += 1


def record_job_queued() -> None:
    _counters["jobs_queued"] += 1


def record_job_complete(duration_secs: float = 0.0) -> None:
    _counters["jobs_complete"] += 1
    if duration_secs > 0:
        _durations.append(duration_secs)
        if len(_durations) > _MAX_DURATION_SAMPLES:
            _durations.pop(0)


def record_job_error() -> None:
    _counters["jobs_error"] += 1


# ---------------------------------------------------------------------------
# Queue depth helpers
# ---------------------------------------------------------------------------

def _celery_queue_depth() -> int:
    """Return approximate Celery queue length from Redis."""
    try:
        from src.api.worker import celery_app, CELERY_AVAILABLE
        if not CELERY_AVAILABLE or celery_app is None:
            return 0
        with celery_app.connection_for_read() as conn:
            return conn.default_channel.client.llen("xboq") or 0
    except Exception:
        return 0


def _threadpool_queue_depth() -> int:
    """Return ThreadPoolExecutor work queue depth (fallback)."""
    try:
        from src.api.routes.analyze import _pipeline_executor
        return _pipeline_executor._work_queue.qsize()  # type: ignore[attr-defined]
    except Exception:
        return 0


def _active_jobs() -> int:
    """Count jobs with status='processing' in the DB."""
    try:
        from src.api.db import SessionLocal
        from src.api.models import JobModel
        from sqlalchemy import select, func
        with SessionLocal() as db:
            result = db.execute(
                select(func.count()).where(JobModel.status == "processing")
            ).scalar()
            return int(result or 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _histogram_lines(name: str, durations: list[float]) -> list[str]:
    """Emit a simple Prometheus histogram (5 buckets + sum + count)."""
    if not durations:
        return [
            f"{name}_bucket{{le=\"60\"}} 0",
            f"{name}_bucket{{le=\"300\"}} 0",
            f"{name}_bucket{{le=\"600\"}} 0",
            f"{name}_bucket{{le=\"1800\"}} 0",
            f"{name}_bucket{{le=\"+Inf\"}} 0",
            f"{name}_sum 0",
            f"{name}_count 0",
        ]
    buckets = [60, 300, 600, 1800]
    lines = []
    for b in buckets:
        count = sum(1 for d in durations if d <= b)
        lines.append(f"{name}_bucket{{le=\"{b}\"}} {count}")
    lines.append(f"{name}_bucket{{le=\"+Inf\"}} {len(durations)}")
    lines.append(f"{name}_sum {sum(durations):.3f}")
    lines.append(f"{name}_count {len(durations)}")
    return lines


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------

def _worker_mode() -> str:
    try:
        from src.api.worker import CELERY_AVAILABLE
        return "celery" if CELERY_AVAILABLE else "threadpool"
    except Exception:
        return "threadpool"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/api/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics(request: Request) -> PlainTextResponse:
    """
    Prometheus-format metrics.

    Not in the OpenAPI schema (internal ops endpoint).
    Protect with network-level access control — do not expose to the internet.
    """
    uptime = time.time() - _STARTUP_TIME
    queue_depth = _celery_queue_depth() or _threadpool_queue_depth()
    active = _active_jobs()
    mode = _worker_mode()

    lines = [
        "# HELP xboq_uptime_seconds Seconds since API startup",
        "# TYPE xboq_uptime_seconds gauge",
        f"xboq_uptime_seconds {uptime:.1f}",
        "",
        "# HELP xboq_queue_depth Number of jobs waiting in the pipeline queue",
        "# TYPE xboq_queue_depth gauge",
        f"xboq_queue_depth {queue_depth}",
        "",
        "# HELP xboq_jobs_active Jobs currently being processed",
        "# TYPE xboq_jobs_active gauge",
        f"xboq_jobs_active {active}",
        "",
        "# HELP xboq_jobs_total Cumulative job count by status",
        "# TYPE xboq_jobs_total counter",
        f"xboq_jobs_total{{status=\"queued\"}} {_counters['jobs_queued']}",
        f"xboq_jobs_total{{status=\"complete\"}} {_counters['jobs_complete']}",
        f"xboq_jobs_total{{status=\"error\"}} {_counters['jobs_error']}",
        "",
        "# HELP xboq_api_requests_total Cumulative HTTP requests received",
        "# TYPE xboq_api_requests_total counter",
        f"xboq_api_requests_total {_counters['requests_total']}",
        "",
        "# HELP xboq_pipeline_duration_seconds Pipeline execution time distribution",
        "# TYPE xboq_pipeline_duration_seconds histogram",
        *_histogram_lines("xboq_pipeline_duration_seconds", list(_durations)),
        "",
        "# HELP xboq_worker_mode 1 if celery, 0 if threadpool",
        "# TYPE xboq_worker_mode gauge",
        f"xboq_worker_mode{{mode=\"{mode}\"}} 1",
        "",
    ]

    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")
