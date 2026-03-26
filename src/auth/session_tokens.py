"""
Session token management for xBOQ.ai.

Issues and verifies signed JWTs for API authentication.

Token payload:
    {
        "org_id":  "acme_corp",
        "user_id": "usr_abc123",
        "role":    "editor",
        "exp":     <unix timestamp>,
        "iat":     <unix timestamp>,
        "iss":     "xboq"
    }

Config:
    XBOQ_JWT_SECRET   — HS256 signing secret (required in production)
    XBOQ_TOKEN_TTL_H  — token lifetime in hours (default: 8)
    XBOQ_DEV_MODE     — set "true" to allow missing JWT secret (dev only)

Usage:
    from src.auth.session_tokens import issue_token, verify_token

    token = issue_token(org_id="acme", user_id="usr_1", role="editor")
    payload = verify_token(token)   # returns dict or raises TokenError
"""

from __future__ import annotations

import logging
import os
import secrets
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEV_MODE = os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes")
_TOKEN_TTL_H = int(os.environ.get("XBOQ_TOKEN_TTL_H", "8"))
_ISSUER = "xboq"


def _get_secret() -> str:
    """
    Return the JWT signing secret.

    Production: must set XBOQ_JWT_SECRET env var.
    Dev mode:   auto-generates a random ephemeral secret (tokens expire on restart).
    """
    secret = os.environ.get("XBOQ_JWT_SECRET", "")
    if secret:
        return secret

    if _DEV_MODE:
        # Lazy-generate a per-process ephemeral secret (dev only)
        global _EPHEMERAL_SECRET  # noqa: PLW0603
        if not _EPHEMERAL_SECRET:
            _EPHEMERAL_SECRET = secrets.token_hex(32)
            warnings.warn(
                "XBOQ_JWT_SECRET is not set — using an ephemeral random secret. "
                "All tokens will be invalidated on server restart. "
                "Set XBOQ_JWT_SECRET in production.",
                stacklevel=3,
            )
        return _EPHEMERAL_SECRET

    raise RuntimeError(
        "XBOQ_JWT_SECRET environment variable is required in production. "
        "Set it to a random 32+ character string, or set XBOQ_DEV_MODE=true for local dev."
    )


_EPHEMERAL_SECRET: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TokenError(Exception):
    """Raised when a JWT cannot be verified."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def issue_token(
    org_id: str,
    user_id: str = "",
    role: str = "viewer",
    ttl_hours: Optional[int] = None,
) -> str:
    """
    Issue a signed JWT for the given identity.

    Returns the token string.
    Raises RuntimeError if XBOQ_JWT_SECRET is not configured in production.
    """
    try:
        import jwt as _jwt
    except ImportError:
        raise RuntimeError("PyJWT is required: pip install PyJWT>=2.8.0")

    now = datetime.now(timezone.utc)
    ttl = ttl_hours if ttl_hours is not None else _TOKEN_TTL_H
    payload = {
        "iss":    _ISSUER,
        "iat":    int(now.timestamp()),
        "exp":    int((now + timedelta(hours=ttl)).timestamp()),
        "org_id": org_id,
        "user_id": user_id,
        "role":   role,
    }
    return _jwt.encode(payload, _get_secret(), algorithm="HS256")


def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT.

    Returns the decoded payload dict on success.
    Raises TokenError on any failure (expired, invalid signature, malformed).
    """
    try:
        import jwt as _jwt
        import jwt.exceptions as _jwtex
    except ImportError:
        raise TokenError("PyJWT is not installed")

    try:
        payload = _jwt.decode(
            token,
            _get_secret(),
            algorithms=["HS256"],
            issuer=_ISSUER,
            options={"require": ["exp", "iat", "org_id"]},
        )
        return payload
    except _jwtex.ExpiredSignatureError:
        raise TokenError("Token has expired — please log in again")
    except _jwtex.InvalidIssuerError:
        raise TokenError("Token issuer is invalid")
    except _jwtex.InvalidSignatureError:
        raise TokenError("Token signature is invalid")
    except Exception as exc:
        raise TokenError(f"Token verification failed: {exc}") from exc


def decode_token_unverified(token: str) -> dict:
    """
    Decode JWT without signature verification (for logging / debugging only).
    NEVER use this result for access control.
    """
    try:
        import jwt as _jwt
        return _jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return {}
