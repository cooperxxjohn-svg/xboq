"""
Rate Escalation Engine — Time-based price projection for Indian construction.

Applies material-specific escalation indices to project rates based on
duration, material category, and base year. Covers steel, cement, aggregates,
labor, fuel, and general construction inflation.

Index Sources (modeled from):
- Steel: NCCI / JSW-TATA index, historically ±8-15% annual
- Cement: ACC-Ambuja index, historically ±5-10% annual
- Aggregates: State mineral dept., historically ±3-6% annual
- Labor: CPI-IW / minimum wage notifications, historically ±6-10% annual
- Fuel/Transport: WPI fuel index, historically ±5-12% annual

All rates are base = Delhi 2024-Q1.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .location_factors import get_city_factor, get_material_city_factor, adjust_rate_for_location

logger = logging.getLogger(__name__)


# =============================================================================
# ESCALATION INDEX DATA
# =============================================================================

class MaterialCategory(Enum):
    """Material categories for escalation indices."""
    STEEL = "steel"
    CEMENT = "cement"
    AGGREGATES = "aggregates"
    BRICKS = "bricks"
    TIMBER = "timber"
    LABOR = "labor"
    FUEL_TRANSPORT = "fuel_transport"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    PAINT = "paint"
    TILES = "tiles"
    WATERPROOFING = "waterproofing"
    GENERAL = "general"


# Annual escalation rates (% per year)
# Based on 5-year moving averages from Indian construction indices 2019-2024
ANNUAL_ESCALATION_PCT: Dict[str, Dict[str, float]] = {
    # scenario: low / normal / high
    "steel": {"low": 3.0, "normal": 8.0, "high": 15.0},
    "cement": {"low": 2.0, "normal": 5.0, "high": 10.0},
    "aggregates": {"low": 1.5, "normal": 4.0, "high": 7.0},
    "bricks": {"low": 2.0, "normal": 4.5, "high": 8.0},
    "timber": {"low": 3.0, "normal": 6.0, "high": 10.0},
    "labor": {"low": 4.0, "normal": 7.0, "high": 12.0},
    "fuel_transport": {"low": 2.0, "normal": 6.0, "high": 14.0},
    "electrical": {"low": 2.0, "normal": 5.0, "high": 9.0},
    "plumbing": {"low": 2.0, "normal": 5.0, "high": 9.0},
    "paint": {"low": 1.5, "normal": 4.0, "high": 8.0},
    "tiles": {"low": 1.0, "normal": 3.5, "high": 7.0},
    "waterproofing": {"low": 2.0, "normal": 5.0, "high": 9.0},
    "general": {"low": 2.5, "normal": 5.5, "high": 10.0},
}

# Typical cost breakdown by trade (what % of an item is steel vs cement vs labor)
# Used to compute blended escalation for composite items like "RCC M25"
TRADE_COST_BREAKDOWNS: Dict[str, Dict[str, float]] = {
    "rcc": {"steel": 0.40, "cement": 0.20, "aggregates": 0.15, "labor": 0.20, "general": 0.05},
    "pcc": {"cement": 0.35, "aggregates": 0.30, "labor": 0.25, "general": 0.10},
    "masonry": {"bricks": 0.40, "cement": 0.20, "labor": 0.30, "general": 0.10},
    "plaster": {"cement": 0.30, "labor": 0.50, "general": 0.20},
    "flooring": {"tiles": 0.45, "cement": 0.15, "labor": 0.30, "general": 0.10},
    "painting": {"paint": 0.40, "labor": 0.45, "general": 0.15},
    "waterproofing": {"waterproofing": 0.45, "cement": 0.10, "labor": 0.35, "general": 0.10},
    "plumbing": {"plumbing": 0.50, "labor": 0.35, "general": 0.15},
    "electrical": {"electrical": 0.55, "labor": 0.30, "general": 0.15},
    "formwork": {"timber": 0.35, "labor": 0.50, "general": 0.15},
    "earthwork": {"fuel_transport": 0.30, "labor": 0.55, "general": 0.15},
    "steel_work": {"steel": 0.65, "labor": 0.25, "general": 0.10},
    "doors_windows": {"timber": 0.40, "labor": 0.35, "general": 0.25},
    "external": {"aggregates": 0.25, "cement": 0.15, "fuel_transport": 0.20, "labor": 0.30, "general": 0.10},
}

# Base date for all rates in the system
BASE_DATE = date(2024, 1, 1)  # Delhi 2024-Q1


# =============================================================================
# WORK TYPE TO TRADE MAPPING
# =============================================================================

# Map BOQ description keywords to trade type for blended escalation
_TRADE_KEYWORDS: Dict[str, List[str]] = {
    "rcc": ["rcc", "reinforced cement concrete", "reinforced concrete", "r.c.c"],
    "pcc": ["pcc", "plain cement concrete", "lean concrete", "blinding", "leveling"],
    "masonry": ["brick", "masonry", "block work", "aac block"],
    "plaster": ["plaster", "cement plaster", "rendering", "neeru", "punning"],
    "flooring": ["flooring", "tiling", "vitrified", "ceramic", "marble", "granite", "kota"],
    "painting": ["paint", "emulsion", "primer", "putty", "distemper", "texture"],
    "waterproofing": ["waterproof", "dpc", "membrane", "bituminous"],
    "plumbing": ["plumbing", "pipe", "cpvc", "upvc", "sanitary", "cistern"],
    "electrical": ["electrical", "wiring", "conduit", "mcb", "switch", "socket", "db box"],
    "formwork": ["formwork", "shuttering", "centering"],
    "earthwork": ["excavation", "earthwork", "backfill", "filling", "compaction"],
    "steel_work": ["reinforcement", "steel", "rebar", "tmt", "bar bending"],
    "doors_windows": ["door", "window", "frame", "shutter", "hardware", "grill"],
    "external": ["compound wall", "gate", "parking", "road", "drain", "septic"],
}


# =============================================================================
# ESCALATION FUNCTIONS
# =============================================================================

def detect_trade(description: str) -> str:
    """
    Detect the construction trade from a BOQ item description.

    Args:
        description: BOQ item description text

    Returns:
        Trade name (e.g., 'rcc', 'masonry', 'flooring') or 'general'
    """
    desc_lower = description.lower()

    # Priority-ordered matching (more specific first)
    priority_order = [
        "waterproofing", "formwork", "steel_work", "rcc", "pcc",
        "flooring", "painting", "plaster", "masonry",
        "plumbing", "electrical", "doors_windows", "earthwork", "external",
    ]

    for trade in priority_order:
        keywords = _TRADE_KEYWORDS.get(trade, [])
        for kw in keywords:
            if kw in desc_lower:
                return trade

    return "general"


def get_annual_escalation(
    material_category: str,
    scenario: str = "normal",
) -> float:
    """
    Get annual escalation percentage for a material category.

    Args:
        material_category: One of MaterialCategory values
        scenario: 'low', 'normal', or 'high'

    Returns:
        Annual escalation percentage (e.g., 8.0 for 8%)
    """
    cat_data = ANNUAL_ESCALATION_PCT.get(material_category)
    if not cat_data:
        cat_data = ANNUAL_ESCALATION_PCT["general"]

    return cat_data.get(scenario, cat_data.get("normal", 5.5))


def get_blended_escalation(
    trade: str,
    scenario: str = "normal",
) -> float:
    """
    Get blended annual escalation for a composite trade item.

    Uses cost breakdown weights to compute weighted-average escalation.
    E.g., RCC = 40% steel (8%) + 20% cement (5%) + 15% agg (4%) + 20% labor (7%) + 5% general (5.5%)
         = 3.2 + 1.0 + 0.6 + 1.4 + 0.275 = 6.475%

    Args:
        trade: Trade name (e.g., 'rcc', 'masonry')
        scenario: Escalation scenario

    Returns:
        Blended annual escalation percentage
    """
    breakdown = TRADE_COST_BREAKDOWNS.get(trade)
    if not breakdown:
        return get_annual_escalation("general", scenario)

    blended = 0.0
    for material, weight in breakdown.items():
        esc_pct = get_annual_escalation(material, scenario)
        blended += weight * esc_pct

    return round(blended, 2)


def escalate_rate(
    base_rate: float,
    description: str = "",
    trade: str = "",
    months_from_base: int = 0,
    scenario: str = "normal",
    base_date: Optional[date] = None,
    target_date: Optional[date] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Escalate a rate from base date to target date.

    Args:
        base_rate: Base rate in INR
        description: BOQ item description (for auto trade detection)
        trade: Explicit trade override (if empty, auto-detected from description)
        months_from_base: Months from base date (alternative to target_date)
        scenario: 'low', 'normal', or 'high'
        base_date: Override base date (default: 2024-Q1)
        target_date: Target date for escalation
        location: Optional city name.  When provided, the escalated rate is
            further adjusted by the city-level cost factor from
            ``location_factors.py`` (base = Delhi = 1.0).

    Returns:
        Dict with escalated_rate, escalation_pct, trade, breakdown
    """
    if base_rate <= 0:
        result = {
            "base_rate": base_rate,
            "escalated_rate": base_rate,
            "escalation_pct": 0.0,
            "escalation_amount": 0.0,
            "trade": trade or "general",
            "scenario": scenario,
            "months": 0,
        }
        if location:
            result["location"] = location
            result["location_factor"] = get_city_factor(location)
        return result

    # Determine trade
    if not trade and description:
        trade = detect_trade(description)
    elif not trade:
        trade = "general"

    # Determine months
    if target_date:
        bd = base_date or BASE_DATE
        delta_days = (target_date - bd).days
        months_from_base = max(0, delta_days / 30.44)  # Average days per month

    months = max(0, months_from_base)
    years = months / 12.0

    # Get blended annual escalation
    annual_pct = get_blended_escalation(trade, scenario)

    # Compound escalation: rate * (1 + pct/100) ^ years
    escalation_factor = (1 + annual_pct / 100.0) ** years
    escalated_rate = round(base_rate * escalation_factor, 2)

    # Apply location factor if a city is specified
    location_factor = 1.0
    if location:
        location_factor = get_city_factor(location)
        escalated_rate = round(escalated_rate * location_factor, 2)

    escalation_pct = round((escalated_rate / base_rate - 1) * 100, 2) if base_rate else 0.0
    escalation_amount = round(escalated_rate - base_rate, 2)

    result = {
        "base_rate": base_rate,
        "escalated_rate": escalated_rate,
        "escalation_pct": escalation_pct,
        "escalation_amount": escalation_amount,
        "escalation_factor": round(escalation_factor, 4),
        "annual_pct": annual_pct,
        "trade": trade,
        "scenario": scenario,
        "months": round(months, 1),
        "years": round(years, 2),
    }

    if location:
        result["location"] = location
        result["location_factor"] = location_factor

    return result


def escalate_boq(
    boq_items: List[Dict[str, Any]],
    months_from_base: int = 0,
    scenario: str = "normal",
    target_date: Optional[date] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Escalate an entire BOQ to a target date.

    Args:
        boq_items: List of BOQ items with 'description', 'rate', 'quantity'
        months_from_base: Months from base date
        scenario: Escalation scenario
        target_date: Target date
        location: Optional city name.  When provided, each escalated rate
            is further adjusted by the city-level cost factor.

    Returns:
        Dict with escalated_boq, summary, total_escalation
    """
    escalated_items = []
    total_base = 0.0
    total_escalated = 0.0
    trade_summary: Dict[str, Dict[str, float]] = {}

    for item in boq_items:
        description = item.get("description", "")
        base_rate = float(item.get("rate", 0))
        qty = float(item.get("quantity", item.get("qty", 0)))

        result = escalate_rate(
            base_rate=base_rate,
            description=description,
            months_from_base=months_from_base,
            scenario=scenario,
            target_date=target_date,
            location=location,
        )

        base_amount = base_rate * qty
        escalated_amount = result["escalated_rate"] * qty
        total_base += base_amount
        total_escalated += escalated_amount

        escalated_item = item.copy()
        escalated_item["base_rate"] = base_rate
        escalated_item["escalated_rate"] = result["escalated_rate"]
        escalated_item["escalation_pct"] = result["escalation_pct"]
        escalated_item["escalated_amount"] = round(escalated_amount, 2)
        escalated_item["trade_detected"] = result["trade"]
        escalated_items.append(escalated_item)

        # Trade-level summary
        trade = result["trade"]
        if trade not in trade_summary:
            trade_summary[trade] = {"base_amount": 0.0, "escalated_amount": 0.0, "items": 0}
        trade_summary[trade]["base_amount"] += base_amount
        trade_summary[trade]["escalated_amount"] += escalated_amount
        trade_summary[trade]["items"] += 1

    # Compute trade-level escalation percentages
    for trade, data in trade_summary.items():
        if data["base_amount"] > 0:
            data["escalation_pct"] = round(
                (data["escalated_amount"] - data["base_amount"]) / data["base_amount"] * 100, 2
            )
        else:
            data["escalation_pct"] = 0.0
        data["base_amount"] = round(data["base_amount"], 2)
        data["escalated_amount"] = round(data["escalated_amount"], 2)

    overall_pct = round((total_escalated - total_base) / total_base * 100, 2) if total_base > 0 else 0.0

    result = {
        "escalated_boq": escalated_items,
        "total_base": round(total_base, 2),
        "total_escalated": round(total_escalated, 2),
        "total_escalation": round(total_escalated - total_base, 2),
        "overall_escalation_pct": overall_pct,
        "scenario": scenario,
        "months": months_from_base,
        "trade_summary": trade_summary,
        "items_count": len(escalated_items),
    }

    if location:
        result["location"] = location
        result["location_factor"] = get_city_factor(location)

    return result


@dataclass
class EscalationRiskProfile:
    """Risk profile for escalation impact."""
    total_base_cost: float = 0.0
    low_scenario_cost: float = 0.0
    normal_scenario_cost: float = 0.0
    high_scenario_cost: float = 0.0
    low_escalation_pct: float = 0.0
    normal_escalation_pct: float = 0.0
    high_escalation_pct: float = 0.0
    recommended_contingency_pct: float = 0.0
    top_risk_trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_base_cost": self.total_base_cost,
            "scenarios": {
                "low": {"cost": self.low_scenario_cost, "escalation_pct": self.low_escalation_pct},
                "normal": {"cost": self.normal_scenario_cost, "escalation_pct": self.normal_escalation_pct},
                "high": {"cost": self.high_scenario_cost, "escalation_pct": self.high_escalation_pct},
            },
            "recommended_contingency_pct": self.recommended_contingency_pct,
            "top_risk_trades": self.top_risk_trades,
        }


def compute_escalation_risk(
    boq_items: List[Dict[str, Any]],
    project_duration_months: int = 18,
) -> EscalationRiskProfile:
    """
    Compute escalation risk profile across low/normal/high scenarios.

    Useful for bid contingency planning — how much buffer to add for
    material/labor cost increases over project lifetime.

    Args:
        boq_items: BOQ items
        project_duration_months: Total project duration

    Returns:
        EscalationRiskProfile with 3-scenario analysis
    """
    # Mid-point escalation (average over project duration)
    mid_months = project_duration_months // 2

    low = escalate_boq(boq_items, mid_months, "low")
    normal = escalate_boq(boq_items, mid_months, "normal")
    high = escalate_boq(boq_items, mid_months, "high")

    profile = EscalationRiskProfile(
        total_base_cost=low["total_base"],
        low_scenario_cost=low["total_escalated"],
        normal_scenario_cost=normal["total_escalated"],
        high_scenario_cost=high["total_escalated"],
        low_escalation_pct=low["overall_escalation_pct"],
        normal_escalation_pct=normal["overall_escalation_pct"],
        high_escalation_pct=high["overall_escalation_pct"],
    )

    # Recommended contingency: midpoint between normal and high
    profile.recommended_contingency_pct = round(
        (profile.normal_escalation_pct + profile.high_escalation_pct) / 2, 1
    )

    # Find top risk trades (highest escalation exposure)
    if normal["trade_summary"]:
        sorted_trades = sorted(
            normal["trade_summary"].items(),
            key=lambda x: x[1].get("escalated_amount", 0) - x[1].get("base_amount", 0),
            reverse=True,
        )
        profile.top_risk_trades = [
            {
                "trade": trade,
                "base_amount": data["base_amount"],
                "escalation_amount": round(data["escalated_amount"] - data["base_amount"], 2),
                "escalation_pct": data["escalation_pct"],
            }
            for trade, data in sorted_trades[:5]
        ]

    return profile
