"""
Health check endpoints.

GET /api/health        — liveness probe (no deps, always fast)
GET /api/health/ready  — readiness probe (checks DB + filesystem + queue)

Liveness:  Is the process alive? Returns 200 instantly.
Readiness: Is the service able to handle traffic? Checks all critical deps.

Use /api/health for Kubernetes liveness probe (fast, no deps).
Use /api/health/ready for Kubernetes readiness probe (fails on degraded deps).
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_VERSION = "0.1.0"


@router.get("/api/health", summary="Liveness probe", response_description="API status and version")
def health_check() -> dict:
    """
    Lightweight liveness probe — no dependency checks.

    Always returns 200 OK if the process is running. Use for Kubernetes
    liveness probes so the pod is not killed due to slow dep checks.
    """
    return {
        "status": "ok",
        "version": _VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/api/health/ready",
    summary="Readiness probe",
    response_description="Deep health check — fails if critical deps are down",
)
async def readiness_check() -> JSONResponse:
    """
    Deep readiness probe — checks database, filesystem, and queue.

    Returns 200 if all critical dependencies are healthy.
    Returns 503 if any critical dependency is degraded.

    Use for Kubernetes readiness probes and uptime monitoring.
    """
    checks: dict = {}
    overall_ok = True

    # 1. Database — try a lightweight query
    try:
        from src.api.db import SessionLocal
        with SessionLocal() as db:
            db.execute(db.get_bind().engine.dialect.statement_compiler(
                db.get_bind().engine.dialect, None
            ).__class__.__mro__[-2].__init__  # noqa — just need to open the session
            if False else db.connection())
        checks["database"] = "ok"
    except Exception:
        # Simpler approach: just try to list jobs from the store
        try:
            from src.api.job_store import job_store
            job_store.list_jobs_by_org("__health_check__", limit=1)
            checks["database"] = "ok"
        except Exception as exc:
            logger.warning("Health/ready: DB check failed: %s", exc)
            checks["database"] = "error"
            overall_ok = False

    # 2. Filesystem — verify output directory is writable
    try:
        from src.api.job_db import OUTPUTS_DIR
        test_file = OUTPUTS_DIR / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        checks["filesystem"] = "ok"
    except Exception as exc:
        logger.warning("Health/ready: filesystem check failed: %s", exc)
        checks["filesystem"] = "error"
        overall_ok = False

    # 3. Queue (non-critical — warn but don't fail readiness)
    try:
        from src.api.worker import CELERY_AVAILABLE, BROKER_URL
        if CELERY_AVAILABLE:
            import redis
            url = BROKER_URL
            r = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
            r.ping()
            r.close()
            checks["queue"] = "celery_ok"
        else:
            checks["queue"] = "threadpool"
    except Exception as exc:
        # Queue degraded — fall back to ThreadPoolExecutor; not a hard failure
        checks["queue"] = f"degraded ({type(exc).__name__})"

    # 4. LLM API key availability (warn only — pipeline might still work offline)
    has_llm = bool(
        os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("XBOQ_OFFLINE_MODE", "").lower() in ("1", "true", "yes")
    )
    checks["llm_key"] = "ok" if has_llm else "missing"

    status_code = 200 if overall_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if overall_ok else "degraded",
            "version": _VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
    )
