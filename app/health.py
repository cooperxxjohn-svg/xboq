"""
Minimal health-check endpoint for uptime monitoring.
Can be imported by the FastAPI app or called standalone.
"""

from __future__ import annotations


def get_health() -> dict:
    """Return a health-check payload."""
    return {"status": "ok", "version": "1.0.0"}


# Allow running as a tiny ASGI app directly: uvicorn app.health:app
try:
    from fastapi import FastAPI

    app = FastAPI(title="xBOQ health")

    @app.get("/health")
    def health() -> dict:
        return get_health()

except ImportError:
    app = None  # type: ignore[assignment]
