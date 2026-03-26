"""
Rate intelligence pipeline — DSR / SOR update tracking.

Keeps xBOQ's rate engine current by:
  1. Checking official DSR / State SOR sources for updates
  2. Recording rate history so estimators can see "steel was ₹82K in Jan, ₹91K now"
  3. Flagging rate staleness on BOQ line items (rate_last_updated > 90 days)

Sources tracked
---------------
  DSR 2023          — Central PWD, updated annually (baseline)
  CPWD Circular     — Quarterly steel/cement corrections
  State SOR         — Maharashtra, Delhi, UP (quarterly)
  Market index      — Indian Steel Association spot price (weekly)

Storage
-------
  rate_history table in PostgreSQL (RateHistoryModel)
  Also mirrors to ~/.xboq/rate_history.json for offline access

Usage
-----
  from src.analysis.rate_intelligence import check_for_updates, get_rate_trend

  # Check if any official rates have been updated (call weekly via cron)
  updates = check_for_updates()

  # Get 6-month trend for steel
  trend = get_rate_trend("fe500", days=180)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_XBOQ_HOME = Path.home() / ".xboq"
_RATE_HISTORY_FILE = _XBOQ_HOME / "rate_history.json"

# ---------------------------------------------------------------------------
# Rate sources — publicly accessible, no login required
# ---------------------------------------------------------------------------

RATE_SOURCES: dict[str, dict] = {
    "dsr_2023": {
        "name": "CPWD DSR 2023",
        "type": "annual",
        "url": "https://cpwd.gov.in/Publication/DSR2023.pdf",
        "notes": "Central Public Works Dept Schedule of Rates 2023",
        "checked_interval_days": 30,
    },
    "cpwd_circular": {
        "name": "CPWD Rate Circular",
        "type": "quarterly",
        "url": "https://cpwd.gov.in/circulars/",
        "notes": "Quarterly corrections for steel, cement, diesel",
        "checked_interval_days": 7,
    },
    "state_sor_maharashtra": {
        "name": "Maharashtra PWD SOR",
        "type": "quarterly",
        "url": "https://mahapwd.com/sor/",
        "notes": "Maharashtra state schedule of rates",
        "checked_interval_days": 30,
    },
    "state_sor_delhi": {
        "name": "Delhi PWD SOR",
        "type": "quarterly",
        "url": "https://pwddel.gov.in/tenders/SOR/",
        "notes": "Delhi PWD schedule of rates",
        "checked_interval_days": 30,
    },
    "state_sor_up": {
        "name": "UP PWD SOR",
        "type": "quarterly",
        "url": "https://uppwd.gov.in/scheduleofrates/",
        "notes": "Uttar Pradesh schedule of rates",
        "checked_interval_days": 30,
    },
}

# ---------------------------------------------------------------------------
# Rate history — DB-backed with file fallback
# ---------------------------------------------------------------------------

def record_rate_snapshot(
    material_key: str,
    rate_inr: float,
    source: str,
    region: str = "national",
    org_id: str = "system",
    notes: str = "",
) -> None:
    """
    Record a rate data point in the history table.

    Called when:
    - A rate override is saved (captures actual procurement prices)
    - The rate intelligence checker finds an official update
    - A user manually records a market rate

    Parameters
    ----------
    material_key : canonical key e.g. "fe500"
    rate_inr     : rate in INR
    source       : "dsr_2023" | "cpwd_circular" | "project_override" | "market" etc
    region       : "national" | "maharashtra" | "delhi" | "up" etc
    org_id       : who recorded this (system for scraped rates)
    """
    try:
        from src.api.db import SessionLocal
        from src.api.models import RateHistoryModel  # added below
        with SessionLocal() as db:
            db.add(RateHistoryModel(
                material_key=material_key,
                rate_inr=rate_inr,
                source=source,
                region=region,
                org_id=org_id,
                notes=notes,
                recorded_at=datetime.now(timezone.utc),
            ))
            db.commit()
    except Exception as exc:
        logger.debug("DB rate history record failed: %s", exc)

    # Mirror to file
    try:
        history = _load_history_file()
        if material_key not in history:
            history[material_key] = []
        history[material_key].append({
            "rate_inr": rate_inr,
            "source": source,
            "region": region,
            "org_id": org_id,
            "notes": notes,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 365 data points per material
        history[material_key] = history[material_key][-365:]
        _save_history_file(history)
    except Exception as exc:
        logger.debug("File rate history mirror failed: %s", exc)


def get_rate_trend(
    material_key: str,
    days: int = 180,
    region: str = "",
    org_id: str = "",
) -> list[dict]:
    """
    Return rate history for a material over the last N days.

    Returns list of {recorded_at, rate_inr, source, region} dicts,
    sorted oldest-first (for charting).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Try DB first
    try:
        from src.api.db import SessionLocal
        from src.api.models import RateHistoryModel
        from sqlalchemy import select

        with SessionLocal() as db:
            q = select(RateHistoryModel).where(
                RateHistoryModel.material_key == material_key,
                RateHistoryModel.recorded_at >= cutoff,
            )
            if region:
                q = q.where(RateHistoryModel.region == region)
            if org_id:
                q = q.where(RateHistoryModel.org_id == org_id)
            q = q.order_by(RateHistoryModel.recorded_at)
            rows = db.execute(q).scalars().all()
            if rows:
                return [
                    {
                        "recorded_at": r.recorded_at.isoformat(),
                        "rate_inr": r.rate_inr,
                        "source": r.source,
                        "region": r.region,
                    }
                    for r in rows
                ]
    except Exception:
        pass

    # File fallback
    history = _load_history_file()
    items = history.get(material_key, [])
    result = []
    for item in items:
        try:
            ts = datetime.fromisoformat(item["recorded_at"])
            if ts >= cutoff:
                if region and item.get("region") != region:
                    continue
                result.append(item)
        except Exception:
            continue
    return result


def get_rate_staleness(material_key: str, max_age_days: int = 90) -> dict:
    """
    Check if a material's rate is stale (not updated in max_age_days).

    Returns:
        is_stale: bool
        days_since_update: int
        last_source: str
        recommendation: str
    """
    trend = get_rate_trend(material_key, days=max_age_days * 2)
    if not trend:
        return {
            "is_stale": True,
            "days_since_update": max_age_days + 1,
            "last_source": "unknown",
            "recommendation": f"No rate history for '{material_key}'. Using DSR 2023 default.",
        }

    latest = trend[-1]
    try:
        last_ts = datetime.fromisoformat(latest["recorded_at"].replace("Z", "+00:00"))
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        days_ago = (datetime.now(timezone.utc) - last_ts).days
    except Exception:
        days_ago = max_age_days + 1

    is_stale = days_ago > max_age_days
    return {
        "is_stale": is_stale,
        "days_since_update": days_ago,
        "last_rate_inr": latest.get("rate_inr"),
        "last_source": latest.get("source", "unknown"),
        "recommendation": (
            f"Rate for '{material_key}' is {days_ago} days old. "
            "Consider updating with current procurement rates."
            if is_stale else
            f"Rate for '{material_key}' is current ({days_ago} days old)."
        ),
    }


def check_for_updates() -> list[dict]:
    """
    Check official sources for rate updates.

    Returns list of update notices. In offline mode, returns empty list.
    This is a lightweight check — it doesn't download the full PDFs,
    it checks HTTP Last-Modified or ETag headers to detect changes.
    """
    import os
    if os.environ.get("XBOQ_OFFLINE_MODE", "").lower() in ("1", "true", "yes"):
        logger.debug("XBOQ_OFFLINE_MODE=true — skipping rate update check")
        return []

    updates = []
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        return []

    for source_id, source in RATE_SOURCES.items():
        try:
            req = urllib.request.Request(source["url"], method="HEAD")
            req.add_header("User-Agent", "xBOQ-RateChecker/1.0")
            with urllib.request.urlopen(req, timeout=5) as resp:
                last_modified = resp.headers.get("Last-Modified", "")
                etag = resp.headers.get("ETag", "")
                updates.append({
                    "source_id": source_id,
                    "name": source["name"],
                    "url": source["url"],
                    "last_modified": last_modified,
                    "etag": etag,
                    "status": "reachable",
                })
        except Exception as exc:
            updates.append({
                "source_id": source_id,
                "name": source["name"],
                "url": source["url"],
                "status": "unreachable",
                "error": str(exc)[:100],
            })

    return updates


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _load_history_file() -> dict:
    if not _RATE_HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(_RATE_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_history_file(data: dict) -> None:
    _XBOQ_HOME.mkdir(parents=True, exist_ok=True)
    _RATE_HISTORY_FILE.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )
