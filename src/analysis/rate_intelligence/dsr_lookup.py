"""
DSR Rate Lookup — match line items to DSR 2023 schedule rates.

Usage:
    from src.analysis.rate_intelligence.dsr_lookup import RateLookup
    lookup = RateLookup()
    result = lookup.benchmark(description="RCC M25 for columns", unit="cum", rate=9500)
    # result.dsr_rate = 8850
    # result.deviation_pct = +7.3
    # result.status = "ABOVE_SCHEDULE"

State-aware usage:
    lookup = RateLookup(state_code="MH")
    result = lookup.benchmark(description="RCC M25 for columns", unit="cum", rate=9500)
    # Prefers Maharashtra PWD SOR 2023 rates, falls back to national DSR
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

_RATES_FILE = Path(__file__).parent.parent.parent.parent / "rates" / "dsr_2023_rates.json"

_STATE_RATE_FILES = {
    "MH": Path(__file__).parent.parent.parent.parent / "rates" / "mh_pwd_2023_rates.json",
    "DL": Path(__file__).parent.parent.parent.parent / "rates" / "dl_cpwd_2023_rates.json",
}

# Deviation thresholds
_ABOVE_HIGH   = 20.0   # >20% above DSR → flag red
_ABOVE_MEDIUM = 10.0   # 10-20% above → flag amber
_BELOW_LOW    = -10.0  # >10% below → flag (suspiciously cheap)


@dataclass
class RateBenchmark:
    dsr_id: str
    dsr_description: str
    dsr_rate: float
    dsr_unit: str
    item_rate: Optional[float]          # the rate on the line item (may be None)
    deviation_pct: Optional[float]      # (item_rate - dsr_rate) / dsr_rate * 100
    status: str                          # "ABOVE_SCHEDULE" | "AT_SCHEDULE" | "BELOW_SCHEDULE" | "UNRATED" | "NO_MATCH"
    match_confidence: float              # 0-1
    match_method: str                    # "keyword" | "unit_filter" | "taxonomy"

    def to_dict(self) -> dict:
        return {
            "dsr_id":           self.dsr_id,
            "dsr_description":  self.dsr_description,
            "dsr_rate":         self.dsr_rate,
            "dsr_unit":         self.dsr_unit,
            "item_rate":        self.item_rate,
            "deviation_pct":    round(self.deviation_pct, 1) if self.deviation_pct is not None else None,
            "status":           self.status,
            "match_confidence": round(self.match_confidence, 2),
            "match_method":     self.match_method,
        }


class RateLookup:
    """
    Loads DSR 2023 rates from JSON and matches line items by keyword similarity.
    Supports optional state_code to prefer state-specific schedule rates (e.g. MH
    for Maharashtra PWD SOR 2023) before falling back to national DSR.
    Thread-safe after __init__ (read-only after load).
    """

    def __init__(
        self,
        rates_file: Path = _RATES_FILE,
        state_code: Optional[str] = None,
    ):
        self._items: List[dict] = []
        self._by_trade: Dict[str, List[dict]] = {}
        self._state_items: List[dict] = []
        self._state_items_by_trade: Dict[str, List[dict]] = {}
        self._state_code: Optional[str] = state_code.upper() if state_code else None

        self._load(rates_file)

        if self._state_code and self._state_code in _STATE_RATE_FILES:
            self._load_state(_STATE_RATE_FILES[self._state_code])

    def _load(self, path: Path):
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._items = data.get("items", [])
        for item in self._items:
            trade = item.get("trade", "general")
            self._by_trade.setdefault(trade, []).append(item)

    def _load_state(self, path: Path):
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._state_items = data.get("items", [])
        for item in self._state_items:
            trade = item.get("trade", "general")
            self._state_items_by_trade.setdefault(trade, []).append(item)

    def _score(self, description: str, unit: Optional[str], dsr_item: dict) -> float:
        """Score a DSR item against a line item description. Returns 0.0-1.0."""
        desc_lower = description.lower()
        keywords = dsr_item.get("keywords", [])

        if not keywords:
            return 0.0

        # Keyword overlap score
        hits = sum(1 for kw in keywords if kw.lower() in desc_lower)
        kw_score = hits / len(keywords)

        # Unit match bonus
        unit_bonus = 0.0
        if unit and dsr_item.get("unit"):
            if unit.lower().strip() == dsr_item["unit"].lower().strip():
                unit_bonus = 0.2

        return min(1.0, kw_score + unit_bonus)

    def find_best_match(
        self,
        description: str,
        unit: Optional[str] = None,
        trade: Optional[str] = None,
        min_score: float = 0.25,
    ) -> Optional[tuple]:  # (dsr_item, score)
        """
        Find the best matching DSR item for a line item description.
        When state items are loaded, tries state-specific candidates first;
        falls back to national DSR if no state match meets min_score.
        """
        if not self._items or not description:
            return None

        # --- State-specific pass (priority) ---
        if self._state_items:
            state_candidates = self._state_items_by_trade.get(trade, []) if trade else []
            if not state_candidates:
                state_candidates = self._state_items

            best_state_item = None
            best_state_score = 0.0
            for dsr_item in state_candidates:
                score = self._score(description, unit, dsr_item)
                if score > best_state_score:
                    best_state_score = score
                    best_state_item = dsr_item

            if best_state_score >= min_score and best_state_item:
                return best_state_item, best_state_score

        # --- National DSR fallback ---
        candidates = self._by_trade.get(trade, []) if trade else []
        if not candidates:
            candidates = self._items

        best_item = None
        best_score = 0.0
        for dsr_item in candidates:
            score = self._score(description, unit, dsr_item)
            if score > best_score:
                best_score = score
                best_item = dsr_item

        if best_score >= min_score and best_item:
            return best_item, best_score
        return None

    def benchmark(
        self,
        description: str,
        unit: Optional[str] = None,
        rate: Optional[float] = None,
        trade: Optional[str] = None,
    ) -> RateBenchmark:
        """
        Benchmark a line item against schedule rates.

        When state_code is set, prefers state SOR rates over national DSR.
        Returns RateBenchmark with dsr_rate, deviation_pct, and status.
        """
        match = self.find_best_match(description, unit, trade)

        if not match:
            return RateBenchmark(
                dsr_id="", dsr_description="", dsr_rate=0.0, dsr_unit="",
                item_rate=rate, deviation_pct=None,
                status="NO_MATCH", match_confidence=0.0, match_method="",
            )

        dsr_item, confidence = match
        dsr_rate = float(dsr_item.get("rate", 0))

        # No rate on the line item
        if rate is None:
            return RateBenchmark(
                dsr_id=dsr_item.get("dsr_id", ""),
                dsr_description=dsr_item.get("description", ""),
                dsr_rate=dsr_rate,
                dsr_unit=dsr_item.get("unit", ""),
                item_rate=None, deviation_pct=None,
                status="UNRATED",
                match_confidence=confidence,
                match_method="keyword",
            )

        # Compute deviation
        if dsr_rate > 0:
            deviation_pct = (rate - dsr_rate) / dsr_rate * 100
        else:
            deviation_pct = None

        # Determine status
        if deviation_pct is None:
            status = "NO_MATCH"
        elif deviation_pct > _ABOVE_MEDIUM:
            status = "ABOVE_SCHEDULE"
        elif deviation_pct < _BELOW_LOW:
            status = "BELOW_SCHEDULE"
        else:
            status = "AT_SCHEDULE"

        return RateBenchmark(
            dsr_id=dsr_item.get("dsr_id", ""),
            dsr_description=dsr_item.get("description", ""),
            dsr_rate=dsr_rate,
            dsr_unit=dsr_item.get("unit", ""),
            item_rate=rate,
            deviation_pct=deviation_pct,
            status=status,
            match_confidence=confidence,
            match_method="keyword",
        )

    def benchmark_items(self, line_items: list) -> list:
        """
        Benchmark a list of line item dicts or UnifiedLineItem objects.
        Returns list of dicts with original item + 'rate_benchmark' key added.
        """
        result = []
        for item in line_items:
            if hasattr(item, 'to_dict'):
                d = item.to_dict()
            else:
                d = dict(item)

            desc  = d.get("description", "")
            unit  = d.get("unit")
            rate  = d.get("rate")
            trade = d.get("trade")

            bm = self.benchmark(desc, unit, rate, trade)
            d["rate_benchmark"] = bm.to_dict()
            result.append(d)
        return result
