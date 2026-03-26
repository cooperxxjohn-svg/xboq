"""
Competitor / Historical Rate Database.

Tracks your company's own bid rates per trade across past projects so you
can compare current BOQ rates to historical wins/losses.

Storage: `~/.xboq/rate_history/` — one JSON file per trade, append-only.

Schema (each entry):
    {
        "project_id":    str,
        "project_name":  str,
        "trade":         str,        # canonical lowercase trade name
        "description":   str,        # BOQ item description
        "unit":          str,        # e.g. "cum", "sqm", "no"
        "rate_inr":      float,      # your submitted rate (INR per unit)
        "market_rate":   float,      # competitor rate if known (0 = unknown)
        "won":           bool|None,  # True=won, False=lost, None=pending
        "region":        str,        # "tier1" | "tier2" | "tier3"
        "recorded_at":   str,        # ISO datetime UTC
    }

Usage:
    from src.analysis.rate_history import RateHistory

    rh = RateHistory()

    # Record a submitted rate
    rh.record(project_id="proj_abc", project_name="Hospital Block",
               trade="structural", description="RCC M25 column",
               unit="cum", rate_inr=14500, won=True)

    # Get benchmark for a trade
    bench = rh.trade_benchmarks("structural")
    # → {"count": 12, "mean_rate": 13800, "p25": 12000, "p75": 15500,
    #    "win_rate_pct": 67, "recent_rates": [...]}

    # Compare BOQ items to own history
    flags = rh.compare_to_history(boq_items, trade="structural")
    # → each item gets "hist_pct_diff" and "hist_flag" ("above"|"below"|"ok"|"no_data")
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HISTORY_DIR = Path.home() / ".xboq" / "rate_history"
_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trade_file(trade: str) -> Path:
    """Return the JSONL file path for a given trade."""
    safe = re.sub(r"[^\w]", "_", trade.lower().strip()) or "unknown"
    return _HISTORY_DIR / f"{safe}.jsonl"


# ---------------------------------------------------------------------------
# Core write
# ---------------------------------------------------------------------------

def record(
    project_id: str,
    trade: str,
    description: str,
    unit: str,
    rate_inr: float,
    project_name: str = "",
    market_rate: float = 0.0,
    won: Optional[bool] = None,
    region: str = "tier1",
) -> dict:
    """
    Record a submitted bid rate for one BOQ item.

    Parameters
    ----------
    project_id : str
    trade : str         — e.g. "structural", "mep", "finishing"
    description : str   — BOQ item description
    unit : str          — unit of measurement
    rate_inr : float    — your bid rate (INR)
    project_name : str  — human-readable project name
    market_rate : float — competitor / market rate if known (0 = unknown)
    won : bool or None  — bid outcome
    region : str        — tier1 / tier2 / tier3

    Returns
    -------
    dict — the recorded entry
    """
    entry = {
        "project_id":   project_id,
        "project_name": project_name,
        "trade":        trade.lower().strip(),
        "description":  description,
        "unit":         unit,
        "rate_inr":     float(rate_inr),
        "market_rate":  float(market_rate),
        "won":          won,
        "region":       region,
        "recorded_at":  _utcnow(),
    }
    path = _trade_file(trade)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    logger.debug("rate_history: recorded %s @ %.0f INR/%s for %s",
                 description[:60], rate_inr, unit, project_id)
    return entry


def update_outcome(project_id: str, trade: str, won: bool) -> int:
    """
    Update the 'won' field on all entries for a project+trade combination.

    Returns the number of records updated.
    """
    path = _trade_file(trade)
    if not path.exists():
        return 0

    updated = 0
    new_lines = []
    with _LOCK:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("project_id") == project_id:
                        entry["won"] = won
                        updated += 1
                    new_lines.append(json.dumps(entry))
                except json.JSONDecodeError:
                    new_lines.append(line)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


# ---------------------------------------------------------------------------
# Read / analytics
# ---------------------------------------------------------------------------

def load_trade(trade: str) -> List[dict]:
    """Load all recorded entries for a given trade."""
    path = _trade_file(trade)
    if not path.exists():
        return []
    entries = []
    with _LOCK:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def list_trades() -> List[str]:
    """Return list of trades that have historical data."""
    return [p.stem for p in sorted(_HISTORY_DIR.glob("*.jsonl"))]


def trade_benchmarks(trade: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute statistical benchmarks for a trade from historical data.

    Parameters
    ----------
    trade : str
    region : str or None — filter to this region if specified

    Returns
    -------
    dict with:
        count           — number of data points
        mean_rate       — average INR rate
        median_rate     — median INR rate
        p25, p75        — 25th / 75th percentile rates
        win_rate_pct    — % of recorded bids that were won (None if no outcomes)
        recent_rates    — last 10 entries [{description, rate_inr, won, recorded_at}]
    """
    entries = load_trade(trade)
    if region:
        entries = [e for e in entries if e.get("region") == region]

    if not entries:
        return {
            "trade": trade, "count": 0, "mean_rate": 0, "median_rate": 0,
            "p25": 0, "p75": 0, "win_rate_pct": None, "recent_rates": [],
        }

    rates = [e["rate_inr"] for e in entries if e.get("rate_inr", 0) > 0]
    sorted_rates = sorted(rates)
    n = len(sorted_rates)

    def _pct(p: float) -> float:
        if not sorted_rates:
            return 0.0
        idx = int(p * (n - 1))
        return round(sorted_rates[idx], 2)

    outcomes = [e["won"] for e in entries if e.get("won") is not None]
    win_rate = round(100 * sum(1 for w in outcomes if w) / len(outcomes), 1) if outcomes else None

    recent = sorted(entries, key=lambda e: e.get("recorded_at", ""), reverse=True)[:10]

    return {
        "trade":        trade,
        "count":        len(rates),
        "mean_rate":    round(statistics.mean(rates), 2) if rates else 0,
        "median_rate":  round(statistics.median(rates), 2) if rates else 0,
        "p25":          _pct(0.25),
        "p75":          _pct(0.75),
        "win_rate_pct": win_rate,
        "recent_rates": [
            {
                "description": r.get("description", ""),
                "rate_inr":    r.get("rate_inr", 0),
                "won":         r.get("won"),
                "project":     r.get("project_name", r.get("project_id", "")),
                "recorded_at": r.get("recorded_at", ""),
            }
            for r in recent
        ],
    }


def compare_to_history(
    items: List[dict],
    trade: str,
    threshold_pct: float = 15.0,
) -> List[dict]:
    """
    Compare BOQ items against historical rate data for the same trade.

    Adds to each item dict:
        hist_pct_diff   float  — % above (+) or below (-) historical mean
        hist_flag       str    — "above" | "below" | "ok" | "no_data"

    Parameters
    ----------
    items : list of BOQ item dicts (must have "rate_inr" or "rate")
    trade : str
    threshold_pct : float — deviation % to flag as above/below

    Returns
    -------
    Same list with hist_* fields added in-place.
    """
    bench = trade_benchmarks(trade)
    mean = bench.get("mean_rate", 0)

    for item in items:
        rate = float(item.get("rate_inr") or item.get("rate") or 0)
        if mean <= 0 or bench["count"] < 3:
            item["hist_pct_diff"] = None
            item["hist_flag"] = "no_data"
        else:
            diff_pct = round(100 * (rate - mean) / mean, 1)
            item["hist_pct_diff"] = diff_pct
            if diff_pct > threshold_pct:
                item["hist_flag"] = "above"
            elif diff_pct < -threshold_pct:
                item["hist_flag"] = "below"
            else:
                item["hist_flag"] = "ok"

    return items


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

class RateHistory:
    """Object-oriented facade over the module-level functions."""

    def record(self, **kwargs) -> dict:
        return record(**kwargs)

    def update_outcome(self, project_id: str, trade: str, won: bool) -> int:
        return update_outcome(project_id, trade, won)

    def trade_benchmarks(self, trade: str, region: Optional[str] = None) -> dict:
        return trade_benchmarks(trade, region)

    def compare_to_history(self, items: List[dict], trade: str,
                           threshold_pct: float = 15.0) -> List[dict]:
        return compare_to_history(items, trade, threshold_pct)

    def list_trades(self) -> List[str]:
        return list_trades()
