"""
src/analysis/rate_source.py

Pluggable RateSource interface for xBOQ.ai.

Instead of hardcoding DSR 2023, any rate schedule can be plugged in:
  - DSRRateSource (DSR 2023 national — default)
  - MHPWDRateSource (Maharashtra PWD SOR)
  - DLCPWDRateSource (Delhi CPWD SOR)
  - KARNATAKARateSource (Karnataka SOR)
  - CustomRateSource (user-supplied JSON/CSV)

Usage:
    from src.analysis.rate_source import get_rate_source, RateSourceRegistry
    rs = get_rate_source("dsr_national")
    rate = rs.lookup("brick masonry 230mm", unit="sqm", region="tier1")

    # Register a custom source
    RateSourceRegistry.register("my_sor", MyCustomRateSource())
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RATES_DIR = _PROJECT_ROOT / "rates"


# ── Base interface ────────────────────────────────────────────────────────────

class RateSource(ABC):
    """
    Abstract base for a rate schedule source.

    Implementations must provide lookup() and list_items().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g., 'DSR 2023 National')."""

    @property
    @abstractmethod
    def source_key(self) -> str:
        """Short machine key (e.g., 'dsr_national', 'mh_pwd', 'dl_cpwd')."""

    @property
    def region(self) -> str:
        """Region this source applies to (e.g., 'national', 'mh', 'dl')."""
        return "national"

    @abstractmethod
    def lookup(
        self,
        description: str,
        unit: str = "",
        trade: str = "",
        region: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Lookup a rate by description.

        Parameters
        ----------
        description : str   Item description to match
        unit : str          Expected unit (for disambiguation)
        trade : str         Trade category (structural, civil, mep, etc.)
        region : str        Region hint (tier1, tier2, mh, dl, etc.)

        Returns
        -------
        dict | None
            {"rate": float, "unit": str, "description": str, "source": str, "confidence": float}
            or None if not found.
        """

    @abstractmethod
    def list_items(self, trade: str = "") -> List[Dict[str, Any]]:
        """Return all items (optionally filtered by trade)."""

    def get_region_multiplier(self, region: str) -> float:
        """Return a cost multiplier for the given region (default 1.0)."""
        _REGION_MULTIPLIERS = {
            "tier1": 1.0,   # Delhi, Mumbai, Bangalore, Chennai, Hyderabad
            "tier2": 0.88,  # Pune, Ahmedabad, Surat, Lucknow, Jaipur, etc.
            "tier3": 0.78,  # Smaller cities, district towns
            "rural": 0.70,
            "mh":    1.05,  # Mumbai region premium
            "dl":    1.02,  # Delhi premium
            "ka":    0.96,  # Bangalore
            "tn":    0.93,  # Chennai
            "ap":    0.89,  # Andhra Pradesh
            "up":    0.85,  # Uttar Pradesh
            "rj":    0.83,  # Rajasthan
            "hr":    0.90,  # Haryana
            "gj":    0.88,  # Gujarat
            "wb":    0.87,  # West Bengal
        }
        return _REGION_MULTIPLIERS.get(str(region).lower(), 1.0)


# ── JSON-backed base ──────────────────────────────────────────────────────────

class JSONRateSource(RateSource):
    """
    Rate source backed by a JSON file in rates/.

    JSON format (either flat list or dict with 'items'/'rates' key):
    [
      {"description": "...", "unit": "...", "rate": 8500, "trade": "structural", "code": "DSR-4.1.1"},
      ...
    ]
    """

    def __init__(self, json_path: Path, source_key: str, name: str, region: str = "national"):
        self._json_path = json_path
        self._source_key = source_key
        self._name = name
        self._region = region
        self._items: Optional[List[dict]] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_key(self) -> str:
        return self._source_key

    @property
    def region(self) -> str:
        return self._region

    def _load(self) -> List[dict]:
        if self._items is not None:
            return self._items
        if not self._json_path.exists():
            logger.warning("RateSource %s: file not found: %s", self._source_key, self._json_path)
            self._items = []
            return self._items
        try:
            with open(self._json_path) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("items", data.get("rates", []))
            self._items = [i for i in items if isinstance(i, dict)]
            logger.debug("RateSource %s: loaded %d items", self._source_key, len(self._items))
        except Exception as e:
            logger.warning("RateSource %s: load failed — %s", self._source_key, e)
            self._items = []
        return self._items

    def list_items(self, trade: str = "") -> List[Dict[str, Any]]:
        items = self._load()
        if trade:
            trade_l = trade.lower()
            items = [i for i in items if trade_l in str(i.get("trade", "")).lower()]
        return items

    def lookup(
        self,
        description: str,
        unit: str = "",
        trade: str = "",
        region: str = "",
    ) -> Optional[Dict[str, Any]]:
        items = self._load()
        if not items:
            return None

        desc_lower = description.lower()
        trade_lower = trade.lower() if trade else ""

        # Keyword tokenize
        desc_tokens = set(w for w in desc_lower.split() if len(w) > 3)

        best_match = None
        best_score = 0.0

        for item in items:
            item_desc = str(item.get("description") or item.get("item_description") or "").lower()
            item_trade = str(item.get("trade") or item.get("category") or "").lower()
            item_unit = str(item.get("unit") or "").lower()

            # Skip trade mismatch (if trade filter given)
            if trade_lower and item_trade and trade_lower not in item_trade and item_trade not in trade_lower:
                continue

            # Token overlap score
            item_tokens = set(w for w in item_desc.split() if len(w) > 3)
            if not item_tokens:
                continue
            overlap = len(desc_tokens & item_tokens) / max(len(desc_tokens | item_tokens), 1)

            # Unit match bonus
            unit_bonus = 0.1 if (unit and unit.lower() == item_unit) else 0.0

            score = overlap + unit_bonus
            if score > best_score:
                best_score = score
                best_match = item

        if best_match is None or best_score < 0.15:
            return None

        # Apply region multiplier
        multiplier = self.get_region_multiplier(region) if region else 1.0
        raw_rate = float(best_match.get("rate") or best_match.get("amount") or 0)
        adjusted_rate = round(raw_rate * multiplier, 2)

        return {
            "rate": adjusted_rate,
            "raw_rate": raw_rate,
            "unit": str(best_match.get("unit") or ""),
            "description": str(best_match.get("description") or best_match.get("item_description") or ""),
            "source": self._source_key,
            "code": str(best_match.get("code") or best_match.get("item_code") or ""),
            "confidence": round(min(best_score, 1.0), 3),
            "region_multiplier": multiplier,
        }


# ── Built-in sources ──────────────────────────────────────────────────────────

class DSRNationalRateSource(JSONRateSource):
    def __init__(self):
        super().__init__(
            json_path=_RATES_DIR / "dsr_2023_rates.json",
            source_key="dsr_national",
            name="DSR 2023 National",
            region="national",
        )


class MHPWDRateSource(JSONRateSource):
    def __init__(self):
        super().__init__(
            json_path=_RATES_DIR / "mh_pwd_2023_rates.json",
            source_key="mh_pwd",
            name="Maharashtra PWD SOR 2023",
            region="mh",
        )


class DLCPWDRateSource(JSONRateSource):
    def __init__(self):
        super().__init__(
            json_path=_RATES_DIR / "dl_cpwd_2023_rates.json",
            source_key="dl_cpwd",
            name="Delhi CPWD SOR 2023",
            region="dl",
        )


# ── Registry ──────────────────────────────────────────────────────────────────

class _RateSourceRegistryMeta:
    """Singleton registry for all rate sources."""

    def __init__(self):
        self._sources: Dict[str, RateSource] = {}
        self._initialized = False

    def _init_defaults(self) -> None:
        if self._initialized:
            return
        for cls in [DSRNationalRateSource, MHPWDRateSource, DLCPWDRateSource]:
            try:
                src = cls()
                self._sources[src.source_key] = src
            except Exception as e:
                logger.debug("Could not initialize rate source %s: %s", cls.__name__, e)
        self._initialized = True

    def register(self, key: str, source: RateSource) -> None:
        """Register a custom rate source."""
        self._sources[key] = source
        logger.info("RateSourceRegistry: registered '%s' (%s)", key, source.name)

    def get(self, key: str) -> Optional[RateSource]:
        self._init_defaults()
        return self._sources.get(key)

    def list_sources(self) -> List[str]:
        self._init_defaults()
        return list(self._sources.keys())

    def lookup_across_sources(
        self,
        description: str,
        unit: str = "",
        trade: str = "",
        region: str = "",
        preferred_source: str = "dsr_national",
    ) -> Optional[Dict[str, Any]]:
        """
        Try lookup in preferred source first, then fall back to others.
        Appends 'fallback_sources' list to result if preferred source missed.
        """
        self._init_defaults()
        # Try preferred first
        preferred = self._sources.get(preferred_source)
        if preferred:
            result = preferred.lookup(description, unit=unit, trade=trade, region=region)
            if result and result.get("confidence", 0) >= 0.20:
                return result
        # Try region-matched source
        if region:
            for key, src in self._sources.items():
                if key == preferred_source:
                    continue
                if src.region.lower() == region.lower():
                    result = src.lookup(description, unit=unit, trade=trade, region=region)
                    if result and result.get("confidence", 0) >= 0.15:
                        result["fallback_from"] = preferred_source
                        return result
        # Try all remaining
        for key, src in self._sources.items():
            if key == preferred_source:
                continue
            result = src.lookup(description, unit=unit, trade=trade, region=region)
            if result and result.get("confidence", 0) >= 0.15:
                result["fallback_from"] = preferred_source
                return result
        return None


RateSourceRegistry = _RateSourceRegistryMeta()


def get_rate_source(key: str = "dsr_national") -> Optional[RateSource]:
    """Get a rate source by key. Returns None if not found."""
    return RateSourceRegistry.get(key)
