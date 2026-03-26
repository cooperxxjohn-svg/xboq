"""
Trade-wise Confidence Scores.

Computes per-trade extraction confidence from the pipeline payload and
returns a structured summary for display in the UI.

Confidence is derived from two sources:
    1. page_index entries — each page has a `confidence` float (OCR + classification)
    2. trade_coverage list — each entry has coverage_pct (scope completeness)

The combined score blends:
    - Mean page confidence for pages classified to that trade          (60%)
    - Normalised coverage_pct from trade_coverage (scope completeness) (40%)

Usage:
    from src.analysis.trade_confidence import compute_trade_confidence, TradeScore

    scores = compute_trade_confidence(payload)
    for s in scores:
        print(f"{s.trade}: {s.confidence_pct}% ({s.label})")

The result is also written to payload["trade_confidence"] by pipeline.py
so it can be serialised and displayed without recomputing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Minimum page confidence threshold below which pages are considered noisy
_MIN_PAGE_CONFIDENCE = 0.3

# Display colours keyed by label
LABEL_COLOR: Dict[str, str] = {
    "High":   "green",
    "Medium": "orange",
    "Low":    "red",
}


@dataclass
class TradeScore:
    """Confidence score for a single trade."""
    trade: str
    confidence_pct: int       # 0–100
    label: str                 # "High" | "Medium" | "Low"
    page_count: int = 0        # number of pages attributed to this trade
    coverage_pct: float = 0.0  # from trade_coverage entry
    mean_page_conf: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trade":           self.trade,
            "confidence_pct":  self.confidence_pct,
            "label":           self.label,
            "page_count":      self.page_count,
            "coverage_pct":    self.coverage_pct,
            "mean_page_conf":  round(self.mean_page_conf, 3),
        }


def _label_from_pct(pct: int) -> str:
    if pct >= 75:
        return "High"
    if pct >= 50:
        return "Medium"
    return "Low"


def compute_trade_confidence(payload: Dict[str, Any]) -> List[TradeScore]:
    """
    Compute per-trade confidence scores from a pipeline payload.

    Returns a list of TradeScore objects sorted descending by confidence.
    Returns an empty list if the payload has insufficient data.
    """
    if not isinstance(payload, dict):
        return []

    # ── Build per-trade page confidence from page_index ───────────────────
    page_index = payload.get("page_index", [])
    trade_pages: Dict[str, List[float]] = {}

    for page in page_index:
        if not isinstance(page, dict):
            continue
        trade = (
            page.get("discipline")
            or page.get("trade")
            or page.get("doc_type")
            or ""
        ).lower().strip()
        conf = float(page.get("confidence") or 0.0)
        if not trade or conf < _MIN_PAGE_CONFIDENCE:
            continue
        trade_pages.setdefault(trade, []).append(conf)

    # ── Load trade_coverage for scope completeness ─────────────────────────
    trade_coverage_list = payload.get("trade_coverage", [])
    coverage_by_trade: Dict[str, float] = {}
    for tc in trade_coverage_list:
        if not isinstance(tc, dict):
            continue
        t = (tc.get("trade") or "").lower().strip()
        cov = float(tc.get("coverage_pct") or tc.get("coverage") or 0.0)
        if t:
            coverage_by_trade[t] = cov

    # coverage_pct is already on a 0-100 scale — normalise against 100 (not
    # the relative max) so a single trade with coverage_pct=10 maps to 0.10,
    # not 1.0 (which would happen if we divided by itself).
    max_cov = 100.0

    # ── Collect all trade names ────────────────────────────────────────────
    all_trades = set(trade_pages.keys()) | set(coverage_by_trade.keys())

    # Canonical trade labels (for display tidiness)
    _CANONICAL = {
        "structural": "Structural",
        "architectural": "Architectural",
        "mep": "MEP",
        "civil": "Civil",
        "finishing": "Finishing",
        "waterproofing": "Waterproofing",
        "electrical": "Electrical",
        "plumbing": "Plumbing",
        "hvac": "HVAC",
        "firefighting": "Firefighting",
        "external": "External Works",
        "sitework": "Sitework",
    }

    if not all_trades:
        return []

    scores: List[TradeScore] = []
    for raw_trade in all_trades:
        pages = trade_pages.get(raw_trade, [])
        mean_conf = sum(pages) / len(pages) if pages else 0.0
        coverage  = coverage_by_trade.get(raw_trade, 0.0)

        # Blend: 60% page confidence + 40% coverage completeness
        coverage_norm = (coverage / max_cov) if max_cov else 0.0
        blended = 0.6 * mean_conf + 0.4 * coverage_norm

        pct = min(100, max(0, round(blended * 100)))
        label = _label_from_pct(pct)
        display_trade = _CANONICAL.get(raw_trade, raw_trade.title())

        scores.append(TradeScore(
            trade=display_trade,
            confidence_pct=pct,
            label=label,
            page_count=len(pages),
            coverage_pct=coverage,
            mean_page_conf=mean_conf,
        ))

    # Sort by confidence descending, then trade name
    scores.sort(key=lambda s: (-s.confidence_pct, s.trade))
    return scores


def format_confidence_badge(scores: List[TradeScore]) -> str:
    """
    Return a compact single-line badge string for display.

    Example:  "Structural 94% · MEP 61% · Finishes 40%"
    """
    if not scores:
        return "No confidence data"
    parts = [f"{s.trade} {s.confidence_pct}%" for s in scores]
    return " · ".join(parts)
