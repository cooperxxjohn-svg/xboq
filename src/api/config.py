"""
xBOQ API Configuration — single source of truth for all env vars.

Import:
    from src.api.config import settings

All env vars are documented here with their defaults and purposes.
Override any value by setting the corresponding environment variable.

Sections:
  API / Auth      — XBOQ_JWT_SECRET, XBOQ_ADMIN_KEY, XBOQ_JWKS_*
  Database        — DATABASE_URL
  Queue / Workers — REDIS_URL, CELERY_BROKER_URL, CELERY_RESULT_BACKEND, CELERY_ALWAYS_EAGER
  Storage         — XBOQ_HOME, XBOQ_JOBS_DIR, XBOQ_JOB_TTL_DAYS
  LLM             — ANTHROPIC_API_KEY, OPENAI_API_KEY, XBOQ_ANTHROPIC_MODEL, XBOQ_OPENAI_MODEL
  CORS            — ALLOWED_ORIGINS
  Integrations    — TELEGRAM_BOT_TOKEN, TWILIO_*, WEBHOOK_SECRET, GOOGLE_MAPS_API_KEY
  Feature flags   — XBOQ_OFFLINE_MODE, XBOQ_DEV_MODE, XBOQ_DISABLE_QTO,
                    XBOQ_DISABLE_CACHE, XBOQ_DISABLE_IMPLIED_ITEMS
  Pipeline tuning — XBOQ_OCR_THREAD_POOL, XBOQ_CACHE_DIR,
                    XBOQ_VIS_*, XBOQ_VMEAS_*
  Dev helpers     — DEBUG_PIPELINE, XBOQ_DEFAULT_JOB_ID, XBOQ_DEFAULT_ORG_ID
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return default


def _env_list(key: str, default: str = "") -> List[str]:
    raw = os.environ.get(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass
class Settings:
    """
    Runtime configuration object.  Created once at module level as ``settings``.
    All attributes are derived from environment variables at import time.

    In tests, set env vars *before* importing this module, or patch attributes
    directly on the ``settings`` singleton.
    """

    # -----------------------------------------------------------------
    # API / Auth
    # -----------------------------------------------------------------
    jwt_secret: str = field(default_factory=lambda: _env("XBOQ_JWT_SECRET", "change-me-in-production"))
    admin_key: str  = field(default_factory=lambda: _env("XBOQ_ADMIN_KEY", ""))
    api_key: str    = field(default_factory=lambda: _env("XBOQ_API_KEY", ""))
    jwks_url: str   = field(default_factory=lambda: _env("XBOQ_JWKS_URL", ""))
    jwks_audience: str = field(default_factory=lambda: _env("XBOQ_JWKS_AUDIENCE", ""))

    # -----------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL",
            "sqlite:///" + str(Path.home() / ".xboq" / "xboq.db"),
        )
    )

    # -----------------------------------------------------------------
    # Queue / Workers
    # -----------------------------------------------------------------
    redis_url: str           = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    celery_broker_url: str   = field(default_factory=lambda: _env("CELERY_BROKER_URL", ""))
    celery_result_backend: str = field(default_factory=lambda: _env("CELERY_RESULT_BACKEND", ""))
    celery_always_eager: bool  = field(default_factory=lambda: _env_bool("CELERY_ALWAYS_EAGER", False))

    # -----------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------
    xboq_home: Path    = field(default_factory=lambda: Path(_env("XBOQ_HOME", str(Path.home() / ".xboq"))))
    jobs_dir: Optional[Path] = field(
        default_factory=lambda: Path(_env("XBOQ_JOBS_DIR", "")) if _env("XBOQ_JOBS_DIR") else None
    )
    job_ttl_days: int  = field(default_factory=lambda: _env_int("XBOQ_JOB_TTL_DAYS", 30))

    # -----------------------------------------------------------------
    # LLM
    # -----------------------------------------------------------------
    anthropic_api_key: str  = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    openai_api_key: str     = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    anthropic_model: str    = field(
        default_factory=lambda: _env("XBOQ_ANTHROPIC_MODEL", "claude-opus-4-6")
    )
    openai_model: str       = field(
        default_factory=lambda: _env("XBOQ_OPENAI_MODEL", "gpt-4o")
    )

    # -----------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------
    allowed_origins: List[str] = field(
        default_factory=lambda: _env_list(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8501,http://localhost:8000",
        )
    )

    # -----------------------------------------------------------------
    # Integrations
    # -----------------------------------------------------------------
    telegram_bot_token: str  = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    twilio_account_sid: str  = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID", ""))
    twilio_auth_token: str   = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN", ""))
    twilio_whatsapp_from: str= field(default_factory=lambda: _env("TWILIO_WHATSAPP_FROM", ""))
    webhook_secret: str      = field(default_factory=lambda: _env("WEBHOOK_SECRET", ""))
    google_maps_api_key: str = field(default_factory=lambda: _env("GOOGLE_MAPS_API_KEY", ""))

    # -----------------------------------------------------------------
    # Feature flags
    # -----------------------------------------------------------------
    offline_mode: bool        = field(default_factory=lambda: _env_bool("XBOQ_OFFLINE_MODE", False))
    dev_mode: bool            = field(default_factory=lambda: _env_bool("XBOQ_DEV_MODE", False))
    disable_cache: bool       = field(default_factory=lambda: _env_bool("XBOQ_DISABLE_CACHE", False))
    disable_implied_items: bool = field(
        default_factory=lambda: _env_bool("XBOQ_DISABLE_IMPLIED_ITEMS", False)
    )
    # Comma-separated list of QTO module names to disable at runtime
    disable_qto_modules: List[str] = field(
        default_factory=lambda: _env_list("XBOQ_DISABLE_QTO", "")
    )

    # -----------------------------------------------------------------
    # Pipeline tuning
    # -----------------------------------------------------------------
    ocr_thread_pool: int     = field(default_factory=lambda: _env_int("XBOQ_OCR_THREAD_POOL", 4))
    cache_dir: Optional[Path] = field(
        default_factory=lambda: Path(_env("XBOQ_CACHE_DIR", "")) if _env("XBOQ_CACHE_DIR") else None
    )
    vis_confidence_min: float = field(
        default_factory=lambda: float(_env("XBOQ_VIS_CONFIDENCE_MIN", "0.5"))
    )
    vis_jpeg_quality: int     = field(default_factory=lambda: _env_int("XBOQ_VIS_JPEG_QUALITY", 85))
    vis_max_pages: int        = field(default_factory=lambda: _env_int("XBOQ_VIS_MAX_PAGES", 30))
    vmeas_jpeg_quality: int   = field(default_factory=lambda: _env_int("XBOQ_VMEAS_JPEG_QUALITY", 85))
    vmeas_max_pages: int      = field(default_factory=lambda: _env_int("XBOQ_VMEAS_MAX_PAGES", 20))
    vmeas_max_px: int         = field(default_factory=lambda: _env_int("XBOQ_VMEAS_MAX_PX", 2048))

    # -----------------------------------------------------------------
    # Dev helpers (non-production)
    # -----------------------------------------------------------------
    debug_pipeline: bool      = field(default_factory=lambda: _env_bool("DEBUG_PIPELINE", False))
    default_job_id: str       = field(default_factory=lambda: _env("XBOQ_DEFAULT_JOB_ID", ""))
    default_org_id: str       = field(default_factory=lambda: _env("XBOQ_DEFAULT_ORG_ID", "local"))

    # -----------------------------------------------------------------
    # Computed helpers
    # -----------------------------------------------------------------

    @property
    def has_llm(self) -> bool:
        """True if at least one LLM API key is configured."""
        return bool(self.anthropic_api_key or self.openai_api_key)

    @property
    def has_redis(self) -> bool:
        """True if a Redis URL is configured (Celery available)."""
        return bool(self.celery_broker_url or self.redis_url != "redis://localhost:6379/0")

    @property
    def effective_jobs_dir(self) -> Path:
        """Resolved path for job output storage."""
        if self.jobs_dir:
            return self.jobs_dir
        return self.xboq_home / "job_outputs"

    def as_dict(self, redact_secrets: bool = True) -> dict:
        """
        Return a JSON-serialisable dict of settings.
        Secrets are replaced with '***' when redact_secrets=True.
        """
        SECRET_KEYS = {
            "jwt_secret", "admin_key", "api_key", "anthropic_api_key",
            "openai_api_key", "twilio_auth_token", "webhook_secret",
            "google_maps_api_key", "telegram_bot_token",
        }
        result = {}
        for k, v in self.__dict__.items():
            if redact_secrets and k in SECRET_KEYS and v:
                result[k] = "***"
            elif isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = v
        return result


# Module-level singleton — imported everywhere
settings = Settings()
