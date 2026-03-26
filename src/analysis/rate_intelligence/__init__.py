# Rate intelligence package
# Re-exports from the intelligence module for backward compat + new functions
from src.analysis.rate_intelligence.intelligence import (
    record_rate_snapshot,
    get_rate_trend,
    get_rate_staleness,
    check_for_updates,
    RATE_SOURCES,
)

__all__ = [
    "record_rate_snapshot",
    "get_rate_trend",
    "get_rate_staleness",
    "check_for_updates",
    "RATE_SOURCES",
]
