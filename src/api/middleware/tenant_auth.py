"""
Tenant Authentication Middleware.

Extracts TenantContext from incoming requests.

Token resolution order:
  1. API key (X-API-Key header)      → backward compat, local-org editor
  2. xBOQ signed JWT                 → org_id / user_id / role from verified payload
  3. Enterprise IdP JWT (JWKS)       → validated against XBOQ_JWKS_URL endpoint
  4. Supabase JWT                    → org_id from user.user_metadata["org_id"]
  5. X-Org-Id header                 → unsigned, accepted only in XBOQ_DEV_MODE=true
  6. Dev fallback                    → XBOQ_DEV_MODE=true only, plan="free"

Enterprise SSO (step 3)
  Set XBOQ_JWKS_URL to your identity provider's JWKS endpoint:
    Azure AD:  https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys
    Okta:      https://your-domain.okta.com/oauth2/default/v1/keys
    Google WS: https://www.googleapis.com/oauth2/v3/certs

  The JWT must include:
    sub   (or email / preferred_username) → used as user_id
    aud   → must include XBOQ_JWKS_AUDIENCE if set (recommended)
    role  → optional; maps to xBOQ role (admin/editor/viewer); defaults to "editor"
    org   or  org_id  → optional; used as org_id if present

Security notes:
  - In production (XBOQ_DEV_MODE unset), unauthenticated requests get
    plan="free" / authenticated=False.
  - The X-Org-Id header bypass only works in XBOQ_DEV_MODE=true.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from fastapi import Request

logger = logging.getLogger(__name__)

def _is_dev_mode() -> bool:
    """Read XBOQ_DEV_MODE at call time (not import time) so tests can toggle it."""
    return os.environ.get("XBOQ_DEV_MODE", "").lower() in ("1", "true", "yes")


@dataclass
class TenantContext:
    org_id: str
    user_id: str  = ""
    role:    str  = "viewer"    # "admin" | "editor" | "viewer"
    plan:    str  = "free"
    authenticated: bool = False


async def get_tenant_context(request: Request) -> TenantContext:
    """
    FastAPI dependency — extracts TenantContext from the incoming request.

    Never raises — falls back gracefully to an unauthenticated free-tier context.
    """
    # ── 0. API-key auth — treat as local-org editor (highest priority shortcut) ─
    # This preserves backward compat: existing tests/clients using X-API-Key
    # are not forced to also provide a JWT.
    _api_key_header = request.headers.get("X-API-Key", "")
    _required_key   = os.environ.get("XBOQ_API_KEY", "")
    if _required_key and _api_key_header == _required_key:
        default_org = os.environ.get("XBOQ_DEFAULT_ORG_ID", "local")
        return TenantContext(
            org_id=default_org,
            role="editor",
            plan=_plan_for_org(default_org),
            authenticated=True,
        )

    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

        # ── 1. Try xBOQ signed JWT first ─────────────────────────────────
        ctx = _context_from_xboq_jwt(token)
        if ctx is not None:
            return ctx

        # ── 2. Try enterprise IdP JWT via JWKS ───────────────────────────
        ctx = _context_from_jwks_jwt(token)
        if ctx is not None:
            return ctx

        # ── 3. Try Supabase JWT ───────────────────────────────────────────
        ctx = _context_from_supabase_jwt(token)
        if ctx is not None:
            return ctx

    # ── 3. X-Org-Id header (dev / service-to-service only) ───────────────
    # Accepted without signature in XBOQ_is_dev_mode() only.  In production this
    # is intentionally ignored so a client cannot spoof another tenant's org.
    if _is_dev_mode():
        org_id_header = request.headers.get("X-Org-Id", "").strip()
        if org_id_header:
            plan = _plan_for_org(org_id_header)
            return TenantContext(
                org_id=org_id_header,
                role="editor",
                plan=plan,
                authenticated=True,
            )

    # ── 4. Dev / single-tenant fallback ──────────────────────────────────
    # Returns authenticated=False and plan="free" (NOT enterprise).
    # Production deployments should configure a JWT secret and require auth.
    default_org = os.environ.get("XBOQ_DEFAULT_ORG_ID", "local")
    plan = "enterprise" if _is_dev_mode() else "free"
    return TenantContext(
        org_id=default_org,
        role="editor" if _is_dev_mode() else "viewer",
        plan=plan,
        authenticated=False,
    )


# ---------------------------------------------------------------------------
# Token verification helpers
# ---------------------------------------------------------------------------

def _context_from_xboq_jwt(token: str) -> TenantContext | None:
    """Verify an xBOQ-issued JWT and return TenantContext."""
    try:
        from src.auth.session_tokens import verify_token, TokenError
        payload = verify_token(token)
        org_id  = payload.get("org_id", "")
        if not org_id:
            return None
        return TenantContext(
            org_id=org_id,
            user_id=payload.get("user_id", ""),
            role=payload.get("role", "viewer"),
            plan=_plan_for_org(org_id),
            authenticated=True,
        )
    except Exception as exc:
        logger.debug("xBOQ JWT validation failed: %s", exc)
        return None


def _context_from_supabase_jwt(token: str) -> TenantContext | None:
    """Validate JWT with Supabase and extract org_id."""
    try:
        from src.auth.supabase_client import get_client, is_configured
        if not is_configured():
            return None
        client = get_client()
        if client is None:
            return None
        resp = client.auth.get_user(token)
        user = resp.user if hasattr(resp, "user") else resp
        if user is None:
            return None
        meta   = getattr(user, "user_metadata", {}) or {}
        org_id = meta.get("org_id") or str(getattr(user, "id", "")) or ""
        if not org_id:
            return None
        user_id = str(getattr(user, "id", ""))
        return TenantContext(
            org_id=org_id,
            user_id=user_id,
            role=meta.get("role", "viewer"),
            plan=_plan_for_org(org_id),
            authenticated=True,
        )
    except Exception as exc:
        logger.debug("Supabase JWT validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# JWKS cache (module-level, refreshed every 24h)
# ---------------------------------------------------------------------------

_JWKS_CACHE: dict = {}          # {jwks_url: {"keys": [...], "fetched_at": float}}
_JWKS_TTL_SECONDS = 86400       # 24 hours


def _fetch_jwks(jwks_url: str) -> list:
    """Return the list of JWK key dicts from the JWKS URL, using a module-level cache."""
    import time, json, urllib.request
    cached = _JWKS_CACHE.get(jwks_url, {})
    if cached and (time.time() - cached.get("fetched_at", 0)) < _JWKS_TTL_SECONDS:
        return cached["keys"]
    try:
        with urllib.request.urlopen(jwks_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        keys = data.get("keys", [])
        _JWKS_CACHE[jwks_url] = {"keys": keys, "fetched_at": time.time()}
        logger.info("JWKS refreshed from %s (%d key(s))", jwks_url, len(keys))
        return keys
    except Exception as exc:
        logger.warning("JWKS fetch failed for %s: %s", jwks_url, exc)
        return cached.get("keys", [])


def _context_from_jwks_jwt(token: str) -> TenantContext | None:
    """
    Validate a JWT against the enterprise IdP JWKS endpoint.

    Reads XBOQ_JWKS_URL and (optionally) XBOQ_JWKS_AUDIENCE from environment.
    Returns None if XBOQ_JWKS_URL is not configured or validation fails.
    """
    jwks_url  = os.environ.get("XBOQ_JWKS_URL", "").strip()
    if not jwks_url:
        return None

    audience  = os.environ.get("XBOQ_JWKS_AUDIENCE", "").strip() or None
    algorithms = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

    try:
        import jwt as _jwt           # PyJWT >= 2.8

        keys = _fetch_jwks(jwks_url)
        if not keys:
            return None

        # Try each key until one succeeds
        last_exc: Exception = Exception("no keys available")
        for jwk in keys:
            try:
                pub_key = _jwt.algorithms.RSAAlgorithm.from_jwk(jwk)
                options: dict = {"verify_exp": True}
                decode_kwargs: dict = {
                    "algorithms": algorithms,
                    "options":    options,
                    "key":        pub_key,
                }
                if audience:
                    decode_kwargs["audience"] = audience

                payload = _jwt.decode(token, **decode_kwargs)

                # Extract identity fields from standard JWT claims
                sub      = str(payload.get("sub", "") or "")
                email    = str(payload.get("email", payload.get("preferred_username", "")) or "")
                user_id  = email or sub

                # org_id: look for "org", "org_id", "tenant", "tid" (Azure AD tenant id)
                org_id = (
                    payload.get("org")
                    or payload.get("org_id")
                    or payload.get("tenant")
                    or payload.get("tid")
                    or os.environ.get("XBOQ_DEFAULT_ORG_ID", "enterprise")
                )

                # role: map from JWT claim (allow "admin"/"editor"/"viewer" passthrough)
                raw_role = str(payload.get("role", payload.get("xboq_role", "")) or "")
                role = raw_role if raw_role in ("admin", "editor", "viewer") else "editor"

                logger.info("JWKS auth OK: user=%s org=%s role=%s", user_id, org_id, role)
                return TenantContext(
                    org_id=str(org_id),
                    user_id=user_id,
                    role=role,
                    plan=_plan_for_org(str(org_id)),
                    authenticated=True,
                )
            except Exception as exc:
                last_exc = exc
                continue

        logger.debug("JWKS JWT validation failed (tried %d key(s)): %s", len(keys), last_exc)
        return None

    except ImportError:
        logger.warning("PyJWT not installed — JWKS SSO unavailable")
        return None
    except Exception as exc:
        logger.debug("JWKS auth error: %s", exc)
        return None


def _plan_for_org(org_id: str) -> str:
    """Look up the org's current plan from local tenant storage."""
    try:
        from src.auth.tenant_manager import get_quota
        return get_quota(org_id).plan
    except Exception:
        return "free"
