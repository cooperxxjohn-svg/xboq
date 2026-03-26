"""
Rate Range Validator

Validates BOQ item rates against knowledge base taxonomy rate ranges.
Flags outliers (too low = possible under-quoting, too high = possible over-pricing).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class RateAnomaly:
    """A single rate anomaly detected."""
    item_description: str
    item_rate: float
    item_unit: str
    taxonomy_id: str
    taxonomy_name: str
    expected_min: float
    expected_max: float
    deviation_pct: float        # negative = below min, positive = above max
    severity: str               # "critical" | "high" | "medium" | "low"
    explanation: str
    match_confidence: float     # How confident the taxonomy match was


@dataclass
class RateValidationResult:
    """Result of rate validation for a set of BOQ items."""
    total_items: int = 0
    items_with_rates: int = 0
    items_matched: int = 0
    items_validated: int = 0     # matched AND has rate range in taxonomy
    anomalies: List[RateAnomaly] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    @property
    def anomaly_rate(self) -> float:
        return len(self.anomalies) / max(self.items_validated, 1)

    @property
    def health_score(self) -> float:
        """0-100 score where 100 = no anomalies."""
        if self.items_validated == 0:
            return 50.0  # Unknown
        penalty = (self.critical_count * 15 + self.high_count * 8 +
                   self.medium_count * 3 + self.low_count * 1)
        return max(0, 100 - penalty)


class RateValidator:
    """Validates BOQ rates against taxonomy rate ranges."""

    def __init__(self):
        self._rate_index = {}   # taxonomy_id -> (min, max, unit)
        self._loaded = False

    def load(self):
        """Build rate range index from taxonomy."""
        if self._loaded:
            return

        from src.knowledge_base import get_taxonomy_items

        items = get_taxonomy_items()
        for item in items:
            rate_range = getattr(item, 'typical_rate_range', None) or {}
            if isinstance(rate_range, dict) and rate_range.get("min") and rate_range.get("max"):
                min_rate = float(rate_range["min"])
                max_rate = float(rate_range["max"])
                if min_rate > 0 and max_rate > 0:
                    item_id = getattr(item, 'id', '')
                    unit = getattr(item, 'unit', '')
                    self._rate_index[item_id] = (min_rate, max_rate, unit)

        self._loaded = True
        logger.info("RateValidator loaded: %d items with rate ranges", len(self._rate_index))

    def validate_items(self, boq_items: List[Dict[str, Any]], min_match_confidence: float = 0.4) -> RateValidationResult:
        """
        Validate rates for a list of BOQ items.

        Args:
            boq_items: List of dicts with at least 'description' (or 'item_name') and 'rate' keys
            min_match_confidence: Minimum taxonomy match confidence to validate

        Returns:
            RateValidationResult with anomalies
        """
        self.load()

        from src.knowledge_base.matcher import match_boq_text

        result = RateValidationResult()
        result.total_items = len(boq_items)

        for item in boq_items:
            desc = item.get("description") or item.get("item_name", "")
            rate = _extract_rate(item)

            if rate and rate > 0:
                result.items_with_rates += 1
            else:
                continue

            # Match to taxonomy — pass unit and section for disambiguation
            boq_unit    = item.get("unit", "")
            boq_section = item.get("section", "")
            match = match_boq_text(desc, min_confidence=min_match_confidence,
                                   unit=boq_unit, section=boq_section)
            if not match.matched:
                continue
            result.items_matched += 1

            # Check if taxonomy item has rate range
            tax_id = match.taxonomy_id
            if tax_id not in self._rate_index:
                continue
            result.items_validated += 1

            min_rate, max_rate, expected_unit = self._rate_index[tax_id]

            # Calculate deviation
            if rate < min_rate:
                deviation_pct = ((rate - min_rate) / min_rate) * 100
                if deviation_pct < -50:
                    severity = "critical"
                    result.critical_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is {abs(deviation_pct):.0f}% below expected minimum Rs {min_rate:,.0f}. Possible under-quoting or missing scope."
                elif deviation_pct < -25:
                    severity = "high"
                    result.high_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is {abs(deviation_pct):.0f}% below expected minimum Rs {min_rate:,.0f}. Verify scope inclusions."
                else:
                    severity = "medium"
                    result.medium_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is slightly below expected range Rs {min_rate:,.0f}-{max_rate:,.0f}."

                result.anomalies.append(RateAnomaly(
                    item_description=desc,
                    item_rate=rate,
                    item_unit=item.get("unit", expected_unit),
                    taxonomy_id=tax_id,
                    taxonomy_name=match.canonical_name,
                    expected_min=min_rate,
                    expected_max=max_rate,
                    deviation_pct=deviation_pct,
                    severity=severity,
                    explanation=explanation,
                    match_confidence=match.confidence,
                ))

            elif rate > max_rate:
                deviation_pct = ((rate - max_rate) / max_rate) * 100
                if deviation_pct > 100:
                    severity = "critical"
                    result.critical_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is {deviation_pct:.0f}% above expected maximum Rs {max_rate:,.0f}. Possible over-pricing or premium spec."
                elif deviation_pct > 50:
                    severity = "high"
                    result.high_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is {deviation_pct:.0f}% above expected maximum Rs {max_rate:,.0f}. Verify specification."
                elif deviation_pct > 20:
                    severity = "medium"
                    result.medium_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is above expected range Rs {min_rate:,.0f}-{max_rate:,.0f}."
                else:
                    severity = "low"
                    result.low_count += 1
                    explanation = f"Rate Rs {rate:,.0f}/{expected_unit} is slightly above expected range."

                result.anomalies.append(RateAnomaly(
                    item_description=desc,
                    item_rate=rate,
                    item_unit=item.get("unit", expected_unit),
                    taxonomy_id=tax_id,
                    taxonomy_name=match.canonical_name,
                    expected_min=min_rate,
                    expected_max=max_rate,
                    deviation_pct=deviation_pct,
                    severity=severity,
                    explanation=explanation,
                    match_confidence=match.confidence,
                ))

        # Sort anomalies by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        result.anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))

        return result


def _extract_rate(item: Dict[str, Any]) -> Optional[float]:
    """Extract numeric unit rate from a BOQ item dict.

    Tries direct rate fields first (highest accuracy), then falls back to
    deriving the unit rate from total_cost / qty (common in scanned BOQs where
    only the line total is OCR-extracted, not the unit rate column).
    """
    # Direct unit-rate keys — ordered by reliability
    _direct_keys = [
        "rate", "unit_rate", "rate_per_unit",
        "basic_rate", "quoted_rate", "unit_cost",
        "amount",   # sometimes stores unit rate
        "value",    # alternate naming in some BOQ formats
        "cost",     # alternate naming
    ]
    for key in _direct_keys:
        val = item.get(key)
        if val is not None:
            try:
                rate = float(val)
                if rate > 0:
                    return rate
            except (ValueError, TypeError):
                pass

    # Fallback: derive unit rate from line total ÷ quantity
    # (common in scanned BOQs where unit rate column is missing/merged)
    for total_key in ("total_amount", "total_cost", "total", "line_total"):
        total_val = item.get(total_key)
        if total_val is None:
            continue
        try:
            total = float(total_val)
        except (ValueError, TypeError):
            continue
        if total <= 0:
            continue
        # Look for quantity
        for qty_key in ("qty", "quantity", "quantity_value", "q", "nos"):
            qty_val = item.get(qty_key)
            if qty_val is None:
                continue
            try:
                qty = float(qty_val)
            except (ValueError, TypeError):
                continue
            if qty > 0:
                return total / qty

    return None


# -- Module-level convenience --

_validator_singleton = None

def get_validator() -> RateValidator:
    global _validator_singleton
    if _validator_singleton is None:
        _validator_singleton = RateValidator()
    return _validator_singleton

def validate_boq_rates(boq_items: List[Dict[str, Any]], min_match_confidence: float = 0.4) -> RateValidationResult:
    """Convenience: validate BOQ item rates against KB."""
    return get_validator().validate_items(boq_items, min_match_confidence)
