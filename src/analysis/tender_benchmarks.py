"""
Tender Benchmark Database — T3-2.

Anonymised cost/sqm and trade breakdown across all uploaded tenders.
Enables comparisons like:
  "This ₹4,200/sqm bid is 18% above market for G+5 residential in Delhi NCR."

Storage: ~/.xboq/tender_benchmarks.jsonl  (JSONL, one record per tender)

Privacy:
  - org_id is SHA-256-hashed before storage / query (k-anonymity guard).
  - benchmark_cost_per_sqm() returns "no_data" status if < 3 records match.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BENCHMARK_FILE = Path.home() / ".xboq" / "tender_benchmarks.jsonl"
_MIN_K = 3          # k-anonymity minimum record count


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TenderRecord:
    project_id: str
    org_id_hash: str            # SHA-256 of org_id — never stores raw org_id
    building_type: str          # "residential"|"commercial"|"institutional"|"industrial"|"unknown"
    region: str                 # "delhi_ncr"|"mumbai"|"bangalore"|"tier1"|"tier2"|"unknown"
    floors: int
    total_area_sqm: float
    total_cost_inr: float
    cost_per_sqm_inr: float
    trade_breakdown: Dict[str, float] = field(default_factory=dict)   # {trade: cost_inr}
    recorded_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_org(org_id: str) -> str:
    return hashlib.sha256(org_id.encode()).hexdigest()[:16]


def _load_records() -> List[dict]:
    if not _BENCHMARK_FILE.exists():
        return []
    records: List[dict] = []
    with _BENCHMARK_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _append_record(record: TenderRecord) -> None:
    _BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _BENCHMARK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), default=str) + "\n")


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (pct / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


# ---------------------------------------------------------------------------
# Building-type detection
# ---------------------------------------------------------------------------

_RESIDENTIAL_KEYWORDS = {"residential", "apartment", "housing", "flat", "villa", "duplex", "bungalow"}
_COMMERCIAL_KEYWORDS = {"commercial", "office", "mall", "retail", "hotel", "plaza", "showroom"}
_INSTITUTIONAL_KEYWORDS = {"school", "college", "hospital", "clinic", "university", "academic", "institutional"}
_INDUSTRIAL_KEYWORDS = {"factory", "warehouse", "industrial", "plant", "shed"}


def detect_building_type(payload: dict) -> str:
    """Infer building type from page_index discipline tags and project name."""
    text_sources = [
        str(payload.get("project_name", "")),
        str(payload.get("tender_title", "")),
    ]
    # Pull discipline tags from page_index if present
    for pg in payload.get("page_index", []):
        text_sources.append(str(pg.get("discipline", "")))
        text_sources.append(str(pg.get("notes", "")))

    combined = " ".join(text_sources).lower()

    if any(kw in combined for kw in _RESIDENTIAL_KEYWORDS):
        return "residential"
    if any(kw in combined for kw in _COMMERCIAL_KEYWORDS):
        return "commercial"
    if any(kw in combined for kw in _INSTITUTIONAL_KEYWORDS):
        return "institutional"
    if any(kw in combined for kw in _INDUSTRIAL_KEYWORDS):
        return "industrial"
    return "unknown"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def record_tender(
    project_id: str,
    org_id: str,
    payload: dict,
    building_type: str = "",
    region: str = "tier1",
) -> TenderRecord:
    """Persist anonymised tender statistics to the benchmark store."""
    building_type = building_type or detect_building_type(payload)

    total_area = float(payload.get("total_area_sqm") or 0.0)
    floors = int(payload.get("floors") or payload.get("num_floors") or 1)

    # Compute total cost from boq_items
    total_cost = 0.0
    trade_breakdown: Dict[str, float] = {}
    for item in payload.get("boq_items", []):
        trade = str(item.get("trade", "general")).lower()
        amount = float(item.get("total_inr") or item.get("amount_inr") or 0.0)
        total_cost += amount
        trade_breakdown[trade] = trade_breakdown.get(trade, 0.0) + amount

    # Fallback: use summary field if available
    if total_cost == 0.0:
        total_cost = float(payload.get("total_cost_inr") or 0.0)

    cost_per_sqm = (total_cost / total_area) if total_area > 0 else 0.0

    record = TenderRecord(
        project_id=project_id,
        org_id_hash=_hash_org(org_id),
        building_type=building_type,
        region=region,
        floors=floors,
        total_area_sqm=total_area,
        total_cost_inr=total_cost,
        cost_per_sqm_inr=cost_per_sqm,
        trade_breakdown=trade_breakdown,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )
    _append_record(record)
    return record


def benchmark_cost_per_sqm(building_type: str, region: str) -> dict:
    """
    Return descriptive statistics for cost/sqm for matching tenders.

    k-anonymity: returns {"status": "no_data"} when < 3 matching records.
    """
    records = _load_records()
    values = [
        r["cost_per_sqm_inr"]
        for r in records
        if r.get("building_type") == building_type
        and r.get("region") == region
        and r.get("cost_per_sqm_inr", 0) > 0
    ]

    if len(values) < _MIN_K:
        return {"status": "no_data", "count": len(values), "building_type": building_type, "region": region}

    return {
        "status": "ok",
        "building_type": building_type,
        "region": region,
        "count": len(values),
        "mean": round(statistics.mean(values), 0),
        "median": round(statistics.median(values), 0),
        "p25": round(_percentile(values, 25), 0),
        "p75": round(_percentile(values, 75), 0),
        "min": round(min(values), 0),
        "max": round(max(values), 0),
    }


def benchmark_trade_rates(trade: str, region: str) -> dict:
    """
    Return descriptive statistics for a specific trade cost across all tenders.

    k-anonymity: returns {"status": "no_data"} when < 3 matching records.
    """
    records = _load_records()
    values = []
    for r in records:
        if r.get("region") != region:
            continue
        breakdown = r.get("trade_breakdown", {})
        if trade.lower() in breakdown:
            cost_inr = breakdown[trade.lower()]
            area = r.get("total_area_sqm", 0)
            if area > 0 and cost_inr > 0:
                values.append(cost_inr / area)

    if len(values) < _MIN_K:
        return {"status": "no_data", "count": len(values), "trade": trade, "region": region}

    return {
        "status": "ok",
        "trade": trade,
        "region": region,
        "count": len(values),
        "mean_per_sqm": round(statistics.mean(values), 0),
        "median_per_sqm": round(statistics.median(values), 0),
        "p25": round(_percentile(values, 25), 0),
        "p75": round(_percentile(values, 75), 0),
    }


def compare_tender_to_market(payload: dict, building_type: str, region: str) -> dict:
    """
    Compare this tender's cost/sqm to market benchmarks.

    Returns:
        {
            cost_per_sqm, market_mean, pct_diff, flag,
            insight_text,    # human-readable summary
            status           # "ok" | "no_benchmark" | "no_cost_data"
        }
    """
    total_area = float(payload.get("total_area_sqm") or 0.0)
    total_cost = float(payload.get("total_cost_inr") or 0.0)
    if total_cost == 0.0:
        total_cost = sum(
            float(item.get("total_inr") or item.get("amount_inr") or 0)
            for item in payload.get("boq_items", [])
        )

    if total_area <= 0 or total_cost <= 0:
        return {"status": "no_cost_data", "insight_text": "Insufficient data to compare."}

    cost_per_sqm = total_cost / total_area
    bm = benchmark_cost_per_sqm(building_type, region)

    if bm.get("status") != "ok":
        return {
            "status": "no_benchmark",
            "cost_per_sqm": round(cost_per_sqm, 0),
            "insight_text": f"No market benchmark available for {building_type} in {region} yet.",
        }

    market_mean = bm["mean"]
    if market_mean > 0:
        pct_diff = ((cost_per_sqm - market_mean) / market_mean) * 100
    else:
        pct_diff = 0.0

    if pct_diff > 10:
        flag = "above_market"
        direction = f"{abs(pct_diff):.0f}% above"
    elif pct_diff < -10:
        flag = "below_market"
        direction = f"{abs(pct_diff):.0f}% below"
    else:
        flag = "at_market"
        direction = "within 10% of"

    insight_text = (
        f"This ₹{cost_per_sqm:,.0f}/sqm bid is {direction} market "
        f"(₹{market_mean:,.0f}/sqm mean) for {building_type} in {region}."
    )

    return {
        "status": "ok",
        "cost_per_sqm": round(cost_per_sqm, 0),
        "market_mean": market_mean,
        "market_median": bm["median"],
        "market_p25": bm["p25"],
        "market_p75": bm["p75"],
        "pct_diff": round(pct_diff, 1),
        "flag": flag,
        "insight_text": insight_text,
        "sample_count": bm["count"],
    }
