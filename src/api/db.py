"""
xBOQ.ai — SQLAlchemy database engine.

Supports both SQLite (local dev) and PostgreSQL (production) via DATABASE_URL:
  - SQLite (default):    sqlite:///~/.xboq/jobs.db
  - PostgreSQL:          postgresql://user:pass@host:5432/xboq

Environment variables
---------------------
  DATABASE_URL    Full SQLAlchemy URL. Defaults to SQLite at ~/.xboq/jobs.db.
                  Heroku-style "postgres://" URLs are auto-rewritten to
                  "postgresql://" for SQLAlchemy 2.x compatibility.
  XBOQ_HOME       Override the xBOQ home directory (default ~/.xboq).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_XBOQ_HOME = Path(os.environ.get("XBOQ_HOME", str(Path.home() / ".xboq")))
_XBOQ_HOME.mkdir(parents=True, exist_ok=True)

_DEFAULT_SQLITE_URL = f"sqlite:///{_XBOQ_HOME / 'jobs.db'}"

# ---------------------------------------------------------------------------
# DATABASE_URL resolution
# ---------------------------------------------------------------------------

def _resolve_database_url() -> str:
    """Return the SQLAlchemy database URL from env or default SQLite."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return _DEFAULT_SQLITE_URL
    # Heroku / Render use "postgres://" which SQLAlchemy 2.x rejects
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


DATABASE_URL: str = _resolve_database_url()

_is_sqlite = DATABASE_URL.startswith("sqlite")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine_kwargs: dict = {
    "pool_pre_ping": True,  # verify connection is alive before using from pool
}

if _is_sqlite:
    # SQLite requires check_same_thread=False for use across FastAPI threads
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL — set reasonable pool sizes for a 3-worker pipeline setup
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10
    _engine_kwargs["pool_timeout"] = 30

engine = create_engine(DATABASE_URL, **_engine_kwargs)

# Enable WAL mode for SQLite to allow concurrent reads alongside writes
if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    """FastAPI dependency: yield a SQLAlchemy session, close on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    from src.api import models as _models  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised: %s", DATABASE_URL.split("@")[-1])  # hide creds


def db_backend() -> str:
    """Return 'sqlite' or 'postgresql' for the current engine."""
    return "sqlite" if _is_sqlite else "postgresql"
