"""
xBOQ.ai FastAPI application.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Ensure project root is on sys.path so all src.* imports resolve
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Auto-load .env ────────────────────────────────────────────────────────────
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    try:
        for _line in _ENV_FILE.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ[_k.strip()] = _v.strip()
    except Exception:
        pass

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.auth import router as auth_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.feedback import router as feedback_router
from src.api.routes.accuracy import router as accuracy_router
from src.api.routes.audit import router as audit_router
from src.api.routes.data_flow import router as data_flow_router
from src.api.routes.health import router as health_router
from src.api.routes.project_members import router as project_members_router
from src.api.routes.project_rates_api import router as project_rates_router
from src.api.routes.analyze import router as analyze_router
from src.api.routes.jobs import router as jobs_router
from src.api.routes.line_items import router as line_items_router
from src.api.routes.stream import router as stream_router
from src.api.routes.revision import router as revision_router
from src.api.routes.email_rfi import router as email_rfi_router
from src.api.routes.collaboration import router as collab_router
from src.api.routes.rate_history import router as rate_history_router
from src.api.routes.webhook import router as webhook_router
from src.api.routes.scope_watch import router as scope_watch_router
from src.api.routes.subcontractor import router as subcontractor_router
# Tier 3 — platform plays
from src.api.routes.tenants import router as tenants_router
from src.api.routes.benchmarks import router as benchmarks_router
from src.api.routes.meeting import router as meeting_router
from src.api.routes.award_predict import router as award_predict_router
# Tier 4 — attentive.ai parity
from src.api.routes.qa_workflow import router as qa_workflow_router
from src.api.routes.estimating_export import router as estimating_export_router
from src.api.routes.aerial_measurement import router as aerial_measurement_router
from src.api.routes.qto_modules import router as qto_modules_router
from src.api.routes.config_info import router as config_info_router
from src.api.routes.admin import router as admin_router

# ---------------------------------------------------------------------------
# Logging configuration
#
# In production (XBOQ_DEV_MODE != true) we emit JSON-structured logs so that
# log aggregators (Datadog, CloudWatch, ELK, etc.) can parse fields natively.
# In dev mode we keep the human-readable format.
# ---------------------------------------------------------------------------

import os as _os

_DEV_MODE = _os.environ.get("XBOQ_DEV_MODE", "false").lower() in ("1", "true", "yes")

if _DEV_MODE:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
else:
    # JSON-structured: one log event per line, machine-parseable
    # Fields: timestamp, level, logger, message
    # Extra fields (job_id, org_id, etc.) appear inline in the message — callers
    # that need truly structured fields should use structlog; this is a lightweight
    # in-process solution that needs zero extra dependencies.
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","lvl":"%(levelname)s","log":"%(name)s","msg":%(message)r}',
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production startup safety guard
#
# Refuse to start (or warn loudly) when required secrets are missing.
# This prevents accidental insecure deployments — if someone forgets to set
# XBOQ_JWT_SECRET the app emits a CRITICAL log and exits in production mode.
# ---------------------------------------------------------------------------

def _check_startup_config() -> None:
    """Validate critical environment variables on startup."""
    import secrets as _secrets
    dev_mode = os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes")

    # 1. XBOQ_JWT_SECRET must be set and not the placeholder value
    jwt_secret = os.environ.get("XBOQ_JWT_SECRET", "")
    _placeholder_values = {"", "CHANGE_ME", "CHANGE_ME_generate_with_secrets_token_hex_32"}
    if jwt_secret in _placeholder_values:
        if dev_mode:
            # Dev mode: generate ephemeral secret and continue
            _ephemeral = _secrets.token_hex(32)
            os.environ["XBOQ_JWT_SECRET"] = _ephemeral
            logger.warning(
                "XBOQ_JWT_SECRET not set — using ephemeral secret for this session. "
                "Set XBOQ_JWT_SECRET in .env for persistent sessions."
            )
        else:
            logger.critical(
                "XBOQ_JWT_SECRET is not set or uses the placeholder value. "
                "Generate one with: python3 -c \"import secrets; print(secrets.token_hex(32))\" "
                "and set it in your .env file. Refusing to start in production mode."
            )
            raise SystemExit(1)

    # 2. XBOQ_DEV_MODE must not be enabled in a likely-production environment
    if dev_mode and jwt_secret not in _placeholder_values and len(jwt_secret) >= 32:
        # Has a real secret but DEV_MODE is on — warn but allow (might be staging)
        logger.warning(
            "XBOQ_DEV_MODE=true is set but a real JWT secret is configured. "
            "Ensure this is intentional — DEV_MODE disables authentication checks."
        )

    # 3. ALLOWED_ORIGINS should be set in production
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "")
    if not allowed_origins and not dev_mode:
        logger.warning(
            "ALLOWED_ORIGINS is not set. All cross-origin requests will be blocked. "
            "Set ALLOWED_ORIGINS=https://your-domain.com in .env."
        )

    # 4. Report queue backend
    try:
        from src.api.worker import CELERY_AVAILABLE, BROKER_URL
        if CELERY_AVAILABLE:
            logger.info("Job queue: Celery (broker: %s)", BROKER_URL.split("@")[-1])
        else:
            logger.warning("Job queue: ThreadPoolExecutor (Celery/Redis not available — limited to 3 concurrent jobs)")
    except Exception:
        pass

    logger.info("Startup config check passed (dev_mode=%s)", dev_mode)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


# Initialise DB tables on startup (idempotent)
try:
    from src.api.db import init_db as _init_db
    _init_db()
except Exception as _dbe:
    logging.getLogger(__name__).error("DB init failed: %s", _dbe)

_OPENAPI_TAGS = [
    {"name": "health",            "description": "Liveness and readiness probes."},
    {"name": "auth",              "description": "JWT login, token refresh, and current-user info."},
    {"name": "analysis",          "description": "Upload files and start the analysis pipeline. Returns a job_id for polling."},
    {"name": "jobs",              "description": "Poll job status, retrieve results, and download exports."},
    {"name": "line-items",        "description": "Structured BOQ line-items, rate benchmarks, and contractual clauses."},
    {"name": "exports",           "description": "One-click CSV exports formatted for Sage 100, Buildertrend, Procore, or generic."},
    {"name": "rfis",              "description": "Email RFI packages to consultants with one click."},
    {"name": "revision",          "description": "Revision diff — compare current run to previous cached result."},
    {"name": "qa",                "description": "Human QA workflow: approve / reject / correct extracted line items."},
    {"name": "collaboration",     "description": "Comments, assignments, sign-offs, and threaded discussions on jobs."},
    {"name": "tenants",           "description": "Multi-tenant workspace management, quota, and plan upgrades."},
    {"name": "projects",          "description": "Project member roles (RBAC) and per-project rate overrides."},
    {"name": "rates",             "description": "Rate history tracking and trade-level benchmark comparisons."},
    {"name": "benchmarks",        "description": "Anonymised cost/sqm benchmarks across all processed tenders."},
    {"name": "meeting",           "description": "Pre-bid meeting assistant: transcript → Q&A → RFIs."},
    {"name": "award-prediction",  "description": "Win-probability prediction using logistic regression on historical bids."},
    {"name": "scope",             "description": "Scope watcher: monitor document sources for addendum changes."},
    {"name": "subcontractor",     "description": "Subcontractor portal: share trade packages and collect rate submissions."},
    {"name": "aerial-measurement","description": "Satellite imagery → site footprint / road / laydown area measurement."},
    {"name": "data-flow",         "description": "Payload field registry — lists all keys produced by the pipeline."},
    {"name": "feedback",          "description": "Estimator corrections for AI values — ground truth for model calibration."},
    {"name": "accuracy",          "description": "Accuracy benchmark dashboard: qty/rate delta, recall/precision by tender type."},
    {"name": "audit",             "description": "Immutable audit log of all job and auth events."},
    {"name": "webhooks",          "description": "Telegram and WhatsApp bot webhooks for field team updates."},
    {"name": "metrics",           "description": "Prometheus-format operational metrics (queue depth, job counts, durations)."},
]

app = FastAPI(
    title="xBOQ.ai API",
    description=(
        "REST API for xBOQ.ai — AI bid engineer copilot for RCC construction tenders.\n\n"
        "**Pipeline summary:** Upload PDFs/DXFs → page classification → OCR extraction → "
        "QTO takeoff → rate engine → RFI generation → structured payload.\n\n"
        "**Authentication:** Pass `X-API-Key` header or `Authorization: Bearer <JWT>`. "
        "Health endpoint is unauthenticated."
    ),
    version="0.1.0",
    openapi_tags=_OPENAPI_TAGS,
    contact={"name": "xBOQ.ai Engineering", "email": "eng@xboq.ai"},
    license_info={"name": "Proprietary"},
)


@app.on_event("startup")
async def _on_startup() -> None:
    """Run startup config validation and background maintenance tasks."""
    _check_startup_config()
    # Background: prune jobs older than TTL (non-blocking, best-effort)
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _cleanup_old_jobs)
    # Schedule periodic cleanup every 24 hours
    loop.create_task(_schedule_periodic_cleanup())


async def _schedule_periodic_cleanup() -> None:
    """
    Periodic job cleanup — runs every 24 hours for the lifetime of the process.

    This ensures old jobs are pruned even when the server runs continuously
    for weeks without a restart.  The cleanup is a lightweight SQLite query
    and disk scan, so it's safe to run in the background asyncio task.
    """
    import asyncio
    _CLEANUP_INTERVAL_S = int(os.environ.get("XBOQ_CLEANUP_INTERVAL_S", str(24 * 60 * 60)))
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_S)
        asyncio.get_event_loop().run_in_executor(None, _cleanup_old_jobs)


def _cleanup_old_jobs() -> None:
    """Prune expired jobs + output dirs. Runs in a thread at startup and periodically."""
    try:
        from src.api.job_store import job_store
        deleted = job_store.expire_old_jobs()
        if deleted:
            logger.info("Cleanup: expired %d old job(s)", deleted)
    except Exception as exc:
        logger.warning("Job cleanup failed (non-fatal): %s", exc)

# ---------------------------------------------------------------------------
# CORS  (P0-4: locked to explicit origin list)
# ---------------------------------------------------------------------------
# Set ALLOWED_ORIGINS env var (comma-separated) in production, e.g.:
#   ALLOWED_ORIGINS=https://app.xboq.ai,https://xboq.netlify.app
# In XBOQ_DEV_MODE=true localhost origins are automatically included.
# ---------------------------------------------------------------------------

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
_allowed_origins: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Always allow localhost in dev mode
if os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes"):
    _allowed_origins += [
        "http://localhost:8501",
        "http://localhost:8502",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8502",
    ]

# Warn loudly if running in production with no origins configured
if not _allowed_origins:
    logger.warning(
        "CORS: ALLOWED_ORIGINS is not set and XBOQ_DEV_MODE is off — "
        "all cross-origin requests will be blocked. "
        "Set ALLOWED_ORIGINS=https://your-domain.com or XBOQ_DEV_MODE=true."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Org-Id"],
)

# ---------------------------------------------------------------------------
# API v1 path compatibility middleware
#
# Rewrites /api/v1/... → /api/... so clients can use either prefix.
# This is a transparent internal rewrite — no redirect, no response change.
# All existing routes are automatically available under both prefixes.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def v1_api_compat(request: Request, call_next):
    if request.url.path.startswith("/api/v1/"):
        new_path = "/api/" + request.url.path[len("/api/v1/"):]
        request.scope["path"] = new_path
        request.scope["raw_path"] = new_path.encode()
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request-ID middleware
#
# Attaches a unique X-Request-ID to every request/response for distributed
# tracing. Accepts client-supplied IDs (for end-to-end correlation) or
# generates a new one. All log lines in downstream handlers can reference
# request.state.request_id for correlation.
# ---------------------------------------------------------------------------

import uuid as _uuid

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Accept client-supplied ID or generate a new one
    req_id = (
        request.headers.get("X-Request-ID")
        or request.headers.get("X-Correlation-ID")
        or _uuid.uuid4().hex[:16]
    )
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Request / Response Logging Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    logger.info("-> %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("<- %s %s %d  %.1f ms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response

# ---------------------------------------------------------------------------
# Global exception handler — prevents stack traces leaking to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# ---------------------------------------------------------------------------
# API Key Authentication Middleware
# ---------------------------------------------------------------------------

_API_KEY_HEADER = "X-API-Key"
_SKIP_AUTH_PATHS = {"/api/health", "/api/auth/login", "/api/auth/refresh", "/api/auth/logout"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    required_key = os.environ.get("XBOQ_API_KEY", "")
    if required_key and request.url.path not in _SKIP_AUTH_PATHS:
        provided_key = request.headers.get(_API_KEY_HEADER, "")
        if provided_key != required_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing API key. Provide X-API-Key header."},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Security headers middleware
# Adds browser security headers to every response.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


# ---------------------------------------------------------------------------
# /api/metrics — protect with admin key
# ---------------------------------------------------------------------------

@app.middleware("http")
async def metrics_auth_middleware(request: Request, call_next):
    if request.url.path == "/api/metrics":
        admin_key = os.environ.get("XBOQ_ADMIN_KEY", "")
        if admin_key:
            provided = request.headers.get("X-Admin-Key", "")
            if provided != admin_key:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "X-Admin-Key required to access metrics"},
                )
    return await call_next(request)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(audit_router)
app.include_router(metrics_router)
app.include_router(feedback_router)
app.include_router(accuracy_router)
app.include_router(data_flow_router)
app.include_router(health_router)
app.include_router(project_members_router)
app.include_router(project_rates_router)
app.include_router(analyze_router)
app.include_router(jobs_router)
app.include_router(line_items_router)
app.include_router(stream_router)
app.include_router(revision_router)
# Tier 2 — new capabilities
app.include_router(email_rfi_router)
app.include_router(collab_router)
app.include_router(rate_history_router)
app.include_router(webhook_router)
app.include_router(scope_watch_router)
app.include_router(subcontractor_router)
# Tier 3 — platform plays
app.include_router(tenants_router)
app.include_router(benchmarks_router)
app.include_router(meeting_router)
app.include_router(award_predict_router)
# Tier 4 — attentive.ai parity
app.include_router(qa_workflow_router)
app.include_router(estimating_export_router)
app.include_router(aerial_measurement_router)
app.include_router(qto_modules_router)
app.include_router(config_info_router)
app.include_router(admin_router)


# ---------------------------------------------------------------------------
# Mobile summary endpoint (returns HTML)
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse


@app.get("/api/mobile-summary/{job_id}", response_class=HTMLResponse)
async def mobile_summary(job_id: str) -> HTMLResponse:
    """Return a mobile-optimised single-page HTML summary for a job."""
    from app.mobile_summary import build_mobile_summary_response
    html = build_mobile_summary_response(job_id)
    return HTMLResponse(content=html)
