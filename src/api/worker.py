"""
xBOQ.ai — Celery application.

Broker: Redis (CELERY_BROKER_URL or REDIS_URL env var).
Falls back gracefully: if Redis/Celery is not reachable the analyze route
uses the ThreadPoolExecutor fallback — no hard crash.

Worker startup (docker-compose handles this automatically):
    celery -A src.api.worker.celery_app worker --loglevel=info --concurrency=3 -Q xboq

Flower monitoring UI:
    celery -A src.api.worker.celery_app flower --port=5555

Environment variables
---------------------
CELERY_BROKER_URL     Redis URL (default: redis://localhost:6379/0)
REDIS_URL             Alternative to CELERY_BROKER_URL (Heroku/Render compat)
CELERY_RESULT_BACKEND Backend URL (default: same as broker)
CELERY_ALWAYS_EAGER   "true" → run tasks synchronously (used in tests)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Broker / backend URLs
# ---------------------------------------------------------------------------

BROKER_URL: str = (
    os.environ.get("CELERY_BROKER_URL")
    or os.environ.get("REDIS_URL")
    or "redis://localhost:6379/0"
)
RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND") or BROKER_URL

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

try:
    from celery import Celery

    # In eager/test mode use in-memory backend so no Redis connection is made
    _always_eager = os.environ.get("CELERY_ALWAYS_EAGER", "false").lower() in ("1", "true", "yes")
    _effective_backend = "cache+memory://" if _always_eager else RESULT_BACKEND

    celery_app = Celery(
        "xboq",
        broker=BROKER_URL,
        backend=_effective_backend,
        include=["src.api.tasks"],
    )

    celery_app.conf.update(
        # Serialisation
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,

        # Reliability — never lose a job
        task_acks_late=True,              # ACK only after task completes
        task_reject_on_worker_lost=True,  # Requeue if worker crashes mid-task
        worker_prefetch_multiplier=1,     # Fair queue — one task at a time per worker slot

        # Visibility
        task_track_started=True,          # Tasks report "started" state to backend

        # Timeouts — pipeline should finish well within 1 hour
        task_time_limit=3600,             # Hard kill after 1 hour
        task_soft_time_limit=3300,        # Soft warning at 55 minutes

        # Retry defaults
        task_max_retries=2,

        # Test mode: CELERY_ALWAYS_EAGER=true runs tasks inline (synchronous)
        task_always_eager=os.environ.get("CELERY_ALWAYS_EAGER", "false").lower() in ("1", "true", "yes"),
        # In eager mode, use in-memory backend so tests don't need a real Redis
        task_always_eager_propagates=True,

        # Priority queues
        # xboq_fast — demo_fast / quick runs, max 15 min, for interactive use
        # xboq_full — full analysis runs, up to 1 hour, for batch/overnight
        # xboq      — legacy default, treated as alias for xboq_full
        task_default_queue="xboq",
        task_queues={
            "xboq_fast": {"exchange": "xboq_fast", "routing_key": "xboq_fast"},
            "xboq_full": {"exchange": "xboq_full", "routing_key": "xboq_full"},
            "xboq":      {"exchange": "xboq",      "routing_key": "xboq"},
        },
        task_routes={
            "xboq.run_pipeline_fast": {"queue": "xboq_fast"},
            "xboq.run_pipeline_full": {"queue": "xboq_full"},
            "xboq.run_pipeline":      {"queue": "xboq"},  # legacy compat
        },
        # Fast queue soft/hard timeouts — 15 min
        # Full queue uses task_time_limit/task_soft_time_limit above (1 hour)
    )

    CELERY_AVAILABLE = True
    logger.info("Celery configured — broker: %s", BROKER_URL.split("@")[-1])  # hide creds

except ImportError:
    celery_app = None  # type: ignore[assignment]
    CELERY_AVAILABLE = False
    logger.warning("celery not installed — pipeline will use ThreadPoolExecutor fallback")
