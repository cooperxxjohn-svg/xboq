"""
Supabase client singleton for xboq.ai.

Usage:
    from src.auth.supabase_client import get_client, is_configured

    if is_configured():
        sb = get_client()
        user = sb.auth.get_user(token)
"""
from __future__ import annotations
import os
from typing import Optional

_client = None

def is_configured() -> bool:
    """Return True if Supabase env vars are set."""
    return bool(
        os.environ.get("SUPABASE_URL") and
        os.environ.get("SUPABASE_ANON_KEY")
    )

def get_client():
    """Return singleton Supabase client, or None if not configured."""
    global _client
    if _client is not None:
        return _client
    if not is_configured():
        return None
    try:
        from supabase import create_client
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_ANON_KEY"],
        )
        return _client
    except ImportError:
        return None
    except Exception:
        return None

def sign_in(email: str, password: str) -> dict:
    """Sign in with email/password. Returns {user, session, error}."""
    sb = get_client()
    if not sb:
        return {"user": None, "session": None, "error": "Supabase not configured"}
    try:
        resp = sb.auth.sign_in_with_password({"email": email, "password": password})
        return {"user": resp.user, "session": resp.session, "error": None}
    except Exception as e:
        return {"user": None, "session": None, "error": str(e)}

def sign_out(token: str) -> None:
    """Sign out current user."""
    sb = get_client()
    if sb:
        try:
            sb.auth.sign_out()
        except Exception:
            pass

def get_user(token: str) -> Optional[dict]:
    """Validate token and return user dict, or None."""
    sb = get_client()
    if not sb or not token:
        return None
    try:
        resp = sb.auth.get_user(token)
        if resp and resp.user:
            return {
                "id": resp.user.id,
                "email": resp.user.email,
                "org_id": (resp.user.user_metadata or {}).get("org_id"),
            }
    except Exception:
        pass
    return None
