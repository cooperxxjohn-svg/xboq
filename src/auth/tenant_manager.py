"""
Tenant Manager — multi-company SaaS quota and plan management.

Each organisation (org_id) gets an isolated workspace with a usage quota
based on their plan tier.

Storage:
    Local:    ~/.xboq/tenants/{org_id}.json
    Cloud:    Supabase `tenants` table (when SUPABASE_URL is set)

Plans:
    free        →  5 tenders / month
    starter     → 25 tenders / month
    pro         → 100 tenders / month
    enterprise  → unlimited

Usage:
    from src.auth.tenant_manager import register_tenant, check_quota, increment_usage

    # On new company signup
    quota = register_tenant("acme_corp", plan="starter")

    # Before running pipeline
    check_quota("acme_corp")   # raises QuotaExceededError if over limit

    # After successful pipeline run
    increment_usage("acme_corp")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TENANTS_DIR = Path.home() / ".xboq" / "tenants"
_TENANTS_DIR.mkdir(parents=True, exist_ok=True)

PLAN_LIMITS: dict[str, int] = {
    "free":       5,
    "starter":    25,
    "pro":        100,
    "enterprise": -1,   # unlimited
}

VALID_PLANS = set(PLAN_LIMITS.keys())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QuotaExceededError(Exception):
    """Raised when a tenant has exhausted their monthly tender quota."""
    def __init__(self, org_id: str, used: int, limit: int) -> None:
        self.org_id = org_id
        self.used = used
        self.limit = limit
        super().__init__(
            f"Quota exceeded for '{org_id}': {used}/{limit} tenders used this month. "
            f"Upgrade your plan to continue."
        )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_of_next_month() -> str:
    now = datetime.now(timezone.utc)
    if now.month == 12:
        nxt = now.replace(year=now.year + 1, month=1, day=1,
                          hour=0, minute=0, second=0, microsecond=0)
    else:
        nxt = now.replace(month=now.month + 1, day=1,
                          hour=0, minute=0, second=0, microsecond=0)
    return nxt.isoformat()


@dataclass
class TenantQuota:
    org_id: str
    plan: str = "free"
    tender_limit: int = 5
    tenders_used: int = 0
    reset_at: str = ""       # ISO8601 — when usage resets
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.reset_at:
            self.reset_at = _first_of_next_month()
        if not self.created_at:
            self.created_at = _utcnow()
        if not self.updated_at:
            self.updated_at = _utcnow()
        # Normalise plan → limit
        self.tender_limit = PLAN_LIMITS.get(self.plan, 5)

    @property
    def is_unlimited(self) -> bool:
        return self.tender_limit == -1

    @property
    def remaining(self) -> int:
        if self.is_unlimited:
            return 999_999
        return max(0, self.tender_limit - self.tenders_used)

    @property
    def is_over_limit(self) -> bool:
        if self.is_unlimited:
            return False
        return self.tenders_used >= self.tender_limit

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "remaining": self.remaining,
            "is_unlimited": self.is_unlimited,
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _quota_path(org_id: str) -> Path:
    # Sanitise org_id for filesystem safety
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in org_id)
    return _TENANTS_DIR / f"{safe}.json"


def _load_quota(org_id: str) -> Optional[TenantQuota]:
    path = _quota_path(org_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
        # Remove computed fields before reconstruction
        data.pop("remaining", None)
        data.pop("is_unlimited", None)
        return TenantQuota(**{k: v for k, v in data.items()
                              if k in TenantQuota.__dataclass_fields__})
    except Exception as exc:
        logger.warning("Failed to load tenant quota for %s: %s", org_id, exc)
        return None


def _save_quota(quota: TenantQuota) -> None:
    quota.updated_at = _utcnow()
    _quota_path(quota.org_id).write_text(
        json.dumps(quota.to_dict(), indent=2), encoding="utf-8"
    )
    _sync_to_supabase(quota)


def _sync_to_supabase(quota: TenantQuota) -> None:
    """Best-effort sync to Supabase `tenants` table."""
    try:
        from src.auth.supabase_client import get_client, is_configured
        if not is_configured():
            return
        client = get_client()
        if client is None:
            return
        client.table("tenants").upsert({
            "org_id":        quota.org_id,
            "plan":          quota.plan,
            "tender_limit":  quota.tender_limit,
            "tenders_used":  quota.tenders_used,
            "reset_at":      quota.reset_at,
            "updated_at":    quota.updated_at,
        }, on_conflict="org_id").execute()
    except Exception:
        pass  # Supabase not configured — silent


def _check_and_reset(quota: TenantQuota) -> TenantQuota:
    """Reset monthly usage if reset_at has passed."""
    try:
        reset_dt = datetime.fromisoformat(quota.reset_at)
        if datetime.now(timezone.utc) >= reset_dt:
            quota.tenders_used = 0
            quota.reset_at = _first_of_next_month()
            _save_quota(quota)
    except Exception:
        pass
    return quota


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_tenant(org_id: str, plan: str = "free") -> TenantQuota:
    """
    Register a new organisation. Idempotent — returns existing quota if already registered.

    Args:
        org_id: Unique organisation identifier (e.g., "acme_corp", UUID, Supabase user.id)
        plan:   "free" | "starter" | "pro" | "enterprise"

    Returns:
        TenantQuota for the organisation.
    """
    existing = _load_quota(org_id)
    if existing is not None:
        return _check_and_reset(existing)

    if plan not in VALID_PLANS:
        plan = "free"

    quota = TenantQuota(org_id=org_id, plan=plan)
    _save_quota(quota)
    logger.info("Registered tenant: org_id=%s plan=%s limit=%d", org_id, plan, quota.tender_limit)
    return quota


def get_quota(org_id: str) -> TenantQuota:
    """
    Return quota for an org. Auto-registers on first access (free plan).

    Returns:
        TenantQuota — never None.
    """
    quota = _load_quota(org_id)
    if quota is None:
        return register_tenant(org_id, plan="free")
    return _check_and_reset(quota)


def check_quota(org_id: str) -> None:
    """
    Verify the org has capacity for one more tender analysis.

    Raises:
        QuotaExceededError if over limit.
    """
    quota = get_quota(org_id)
    if quota.is_over_limit:
        raise QuotaExceededError(org_id, quota.tenders_used, quota.tender_limit)


def increment_usage(org_id: str) -> TenantQuota:
    """
    Record one tender analysis consumed by the org.

    Raises:
        QuotaExceededError if already at limit.
    """
    quota = get_quota(org_id)
    if quota.is_over_limit:
        raise QuotaExceededError(org_id, quota.tenders_used, quota.tender_limit)
    quota.tenders_used += 1
    _save_quota(quota)
    return quota


def upgrade_plan(org_id: str, new_plan: str) -> TenantQuota:
    """
    Change an organisation's plan tier.

    Args:
        org_id:   Organisation identifier.
        new_plan: Target plan ("free"|"starter"|"pro"|"enterprise").

    Returns:
        Updated TenantQuota.

    Raises:
        ValueError if new_plan is not a valid plan name.
    """
    if new_plan not in VALID_PLANS:
        raise ValueError(f"Invalid plan '{new_plan}'. Choose from: {sorted(VALID_PLANS)}")
    quota = get_quota(org_id)
    quota.plan = new_plan
    quota.tender_limit = PLAN_LIMITS[new_plan]
    _save_quota(quota)
    logger.info("Upgraded tenant: org_id=%s → plan=%s limit=%d", org_id, new_plan, quota.tender_limit)
    return quota


def list_tenants() -> list[dict]:
    """Return all registered tenant quotas (for admin use)."""
    results = []
    for p in sorted(_TENANTS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text("utf-8"))
            results.append(data)
        except Exception:
            pass
    return results
