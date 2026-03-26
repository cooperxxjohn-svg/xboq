"""
Data flow transparency endpoint.

GET /api/data-flow

Returns a machine-readable manifest of all external services xBOQ.ai
can call, what data is sent, under what conditions, and whether the current
deployment has those features enabled or disabled.

This endpoint is intentionally public (no auth required) so that L&T's
security team can review it without needing a user account.

Reviewed by: security team / DPA review
"""

from __future__ import annotations

import os
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["data-flow"])


def _is_offline() -> bool:
    return os.environ.get("XBOQ_OFFLINE_MODE", "").lower() in ("1", "true", "yes")


def _key_present(env_var: str) -> bool:
    return bool(os.environ.get(env_var, "").strip())


@router.get("/api/data-flow")
async def data_flow() -> JSONResponse:
    """
    Return a manifest of all external data flows.

    For each service:
      - name:        service name
      - purpose:     what it is used for
      - data_sent:   description of what user data leaves the system
      - trigger:     when is this called
      - enabled:     whether it is currently configured / active
      - can_disable: env var to disable it
    """
    offline = _is_offline()

    services = [
        {
            "name": "Anthropic Claude API",
            "url": "https://api.anthropic.com",
            "purpose": "LLM-based specification extraction from tender documents",
            "data_sent": "Extracted text from specification pages of uploaded PDF tender documents",
            "trigger": "When ANTHROPIC_API_KEY is set and XBOQ_OFFLINE_MODE is not true",
            "enabled": _key_present("ANTHROPIC_API_KEY") and not offline,
            "can_disable": "Set XBOQ_OFFLINE_MODE=true to prevent all LLM calls",
            "data_residency": "Data processed on Anthropic US servers. Not stored per Anthropic API policy.",
        },
        {
            "name": "OpenAI API",
            "url": "https://api.openai.com",
            "purpose": "LLM enrichment fallback when Anthropic key is not configured",
            "data_sent": "Extracted text from uploaded tender documents (fallback only)",
            "trigger": "When OPENAI_API_KEY is set, ANTHROPIC_API_KEY is not set, and XBOQ_OFFLINE_MODE is not true",
            "enabled": _key_present("OPENAI_API_KEY") and not _key_present("ANTHROPIC_API_KEY") and not offline,
            "can_disable": "Set XBOQ_OFFLINE_MODE=true or configure ANTHROPIC_API_KEY (which takes priority)",
            "data_residency": "Data processed on OpenAI US servers. Not stored per OpenAI API policy.",
        },
        {
            "name": "OpenAI Embeddings API",
            "url": "https://api.openai.com/v1/embeddings",
            "purpose": "Document embeddings for semantic search (optional — local sentence-transformers used by default)",
            "data_sent": "Text chunks from tender documents for embedding generation",
            "trigger": "Only when embedder.py backend is explicitly set to 'openai'. Default backend is local sentence-transformers.",
            "enabled": False,  # default is local — openai embedding requires explicit opt-in
            "can_disable": "Default is already local. Set XBOQ_OFFLINE_MODE=true for hard block.",
            "data_residency": "N/A — disabled by default",
        },
        {
            "name": "Sentry Error Tracking",
            "url": "https://sentry.io",
            "purpose": "Application error reporting and performance monitoring",
            "data_sent": "Stack traces, error messages, performance metrics. No tender document content.",
            "trigger": "When SENTRY_DSN is configured",
            "enabled": _key_present("SENTRY_DSN"),
            "can_disable": "Remove or unset SENTRY_DSN environment variable",
            "data_residency": "Error data on Sentry cloud (EU region available). No tender content transmitted.",
        },
        {
            "name": "Webhook endpoints (user-configured)",
            "url": "User-defined",
            "purpose": "Notify external systems on job completion",
            "data_sent": "Job status, summary statistics. Configurable — content depends on user webhook config.",
            "trigger": "When webhook URL is configured per-tenant and job completes",
            "enabled": False,  # only active when tenant configures a webhook URL
            "can_disable": "Remove webhook URL from tenant configuration",
            "data_residency": "Depends on webhook destination configured by the user",
        },
    ]

    offline_note = (
        "XBOQ_OFFLINE_MODE=true is active. All external LLM API calls are blocked. "
        "Only Sentry (if configured) and user-configured webhooks may send data externally."
        if offline else
        "XBOQ_OFFLINE_MODE is not active. LLM APIs are called if keys are configured."
    )

    return JSONResponse(content={
        "offline_mode": offline,
        "offline_mode_note": offline_note,
        "data_sovereignty": {
            "on_premise_deployment": "Available — use docker-compose.yml for fully on-premise deployment",
            "tender_pdf_storage": "PDFs stored locally in XBOQ_JOBS_DIR (default ~/.xboq/job_outputs). Never uploaded to cloud storage.",
            "database": "Job metadata stored in DATABASE_URL (SQLite or PostgreSQL). Configured by operator — can be on-premise.",
            "audit_log": "All actions logged to the same DATABASE_URL. Append-only, never transmitted externally.",
        },
        "services": services,
        "contact": "security@xboq.ai for DPA requests, data processing agreements, or security reviews",
    })
