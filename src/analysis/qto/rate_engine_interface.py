"""
RateEngine interface for xBOQ.ai.

Provides a stable, injectable facade over the rate application logic so that
callers (QTO modules, pipeline stages, BOQ auto-fill) do not need to know
whether rates come from the in-memory keyword DB, DSR CSV, project overrides,
or a future external API.

Usage (preferred — replaces ad-hoc ``from rate_engine import apply_rates``):

    from src.analysis.qto.rate_engine_interface import get_rate_engine

    re = get_rate_engine()
    rated_items = re.apply_rates(items, region="tier2")
    summary     = re.compute_trade_summary(rated_items)
    single      = re.lookup("brick masonry 230mm", unit="sqm", region="tier1")

The singleton is created on the first ``get_rate_engine()`` call and reused
for the lifetime of the process.

In tests, replace the singleton via ``reset_rate_engine(stub_instance)`` and
restore it with ``reset_rate_engine()`` in teardown.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — the stable contract
# ---------------------------------------------------------------------------

@runtime_checkable
class RateEngineProtocol(Protocol):
    """
    Stable contract for rate engine implementations.

    All methods must return gracefully (never raise) — return [] / {} / 0.0
    when the underlying source is unavailable.
    """

    def apply_rates(
        self,
        items: List[dict],
        region: str = "tier1",
        use_dsr_fallback: bool = True,
        expand_composites: bool = True,
        project_rates: Optional[dict] = None,
    ) -> List[dict]:
        """Apply unit rates to a list of BOQ/QTO items.

        Adds ``rate_inr``, ``amount_inr``, ``rate_source``, ``rate_confidence``,
        ``rate_region`` keys to each item.  Returns the extended list.
        Composite items are replaced by their BOM components when
        ``expand_composites=True``.
        """
        ...

    def compute_trade_summary(self, rated_items: List[dict]) -> Dict[str, dict]:
        """Aggregate rated items into per-trade totals.

        Returns a dict keyed by trade name with keys:
        ``total_inr``, ``item_count``, ``avg_confidence``.
        """
        ...

    def lookup(
        self,
        description: str,
        unit: str = "",
        region: str = "tier1",
        trade: str = "",
    ) -> Dict[str, Any]:
        """Single-item rate lookup.

        Returns dict with at minimum ``rate_inr``, ``unit``, ``confidence``,
        ``source`` keys.  Returns empty dict if no match.
        """
        ...

    def rate_summary_text(self, summary: Dict[str, dict]) -> str:
        """Format a trade summary dict into a readable string."""
        ...


# ---------------------------------------------------------------------------
# Default implementation — thin wrapper over existing rate_engine functions
# ---------------------------------------------------------------------------

class _DefaultRateEngine:
    """
    Default RateEngine backed by ``src.analysis.qto.rate_engine`` functions
    and the ``RateSourceRegistry`` for multi-source DSR lookups.
    """

    def apply_rates(
        self,
        items: List[dict],
        region: str = "tier1",
        use_dsr_fallback: bool = True,
        expand_composites: bool = True,
        project_rates: Optional[dict] = None,
    ) -> List[dict]:
        try:
            from src.analysis.qto.rate_engine import apply_rates as _apply_rates
            return _apply_rates(
                items,
                region=region,
                use_dsr_fallback=use_dsr_fallback,
                expand_composites=expand_composites,
                project_rates=project_rates,
            )
        except Exception as exc:
            logger.error("RateEngine.apply_rates failed: %s", exc)
            return items

    def compute_trade_summary(self, rated_items: List[dict]) -> Dict[str, dict]:
        try:
            from src.analysis.qto.rate_engine import compute_trade_summary as _cts
            return _cts(rated_items)
        except Exception as exc:
            logger.error("RateEngine.compute_trade_summary failed: %s", exc)
            return {}

    def lookup(
        self,
        description: str,
        unit: str = "",
        region: str = "tier1",
        trade: str = "",
    ) -> Dict[str, Any]:
        """Single-item rate lookup via RateSourceRegistry."""
        try:
            from src.analysis.rate_source import RateSourceRegistry as _RSR
            result = _RSR.lookup_across_sources(
                description, unit=unit, trade=trade, region=region
            )
            return result or {}
        except Exception as exc:
            logger.debug("RateEngine.lookup failed: %s", exc)
            return {}

    def rate_summary_text(self, summary: Dict[str, dict]) -> str:
        try:
            from src.analysis.qto.rate_engine import rate_summary_text as _rst
            return _rst(summary)
        except Exception as exc:
            logger.debug("RateEngine.rate_summary_text failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_engine_singleton: Optional[Any] = None


def get_rate_engine() -> "RateEngineProtocol":  # type: ignore[return]
    """Return the process-wide RateEngine singleton.

    Thread-safe at the Python GIL level.  The first call creates the default
    ``_DefaultRateEngine`` instance; subsequent calls return the cached object.
    """
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = _DefaultRateEngine()
    return _engine_singleton  # type: ignore[return-value]


def reset_rate_engine(replacement=None) -> None:
    """Replace (or clear) the singleton.

    In tests::

        from src.analysis.qto.rate_engine_interface import reset_rate_engine

        class _StubEngine:
            def apply_rates(self, items, region="tier1", **kw): return items
            def compute_trade_summary(self, rated_items): return {}
            def lookup(self, description, unit="", region="tier1", trade=""): return {}
            def rate_summary_text(self, summary): return ""

        reset_rate_engine(_StubEngine())
        # ... run tests ...
        reset_rate_engine()   # restore default

    Args:
        replacement: A ``RateEngineProtocol``-compatible object, or ``None``
                     to clear the singleton so it is recreated on next call.
    """
    global _engine_singleton
    _engine_singleton = replacement
