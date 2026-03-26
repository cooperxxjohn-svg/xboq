"""
Bid Scenario Comparator — 3-tier bid variant generator.

Sprint 30: Generates Aggressive / Balanced / Conservative bid scenarios
from the analysis payload.  Uses grade factors from rate_builder.py and
contingency levels from risk/pricing.py — zero new dependencies.

Each scenario applies different grade factors, margins, and contingency
levels to the extracted BOQ to produce a total bid amount.

Usage:
    from src.bid_scenarios import generate_scenarios

    comparison = generate_scenarios(payload)
    for s in comparison.scenarios:
        print(f"{s.label}: INR {s.total_bid:,.0f}")
    print(f"Recommended: {comparison.recommendation}")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SCENARIO PROFILES
# =============================================================================

SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "aggressive": {
        "label": "Aggressive",
        "grade": "basic",
        "grade_factor": 0.80,
        "margin_pct": 5.0,
        "contingency_pct": 2.0,
        "oh_pct": 8.0,
        "description": "Minimum margins, basic-grade finishes, lowest contingency. "
                       "Best for must-win or repeat-client situations.",
    },
    "balanced": {
        "label": "Balanced",
        "grade": "standard",
        "grade_factor": 1.00,
        "margin_pct": 8.0,
        "contingency_pct": 5.0,
        "oh_pct": 12.0,
        "description": "Standard grades, moderate margins, typical contingency. "
                       "Suitable for most competitive tenders.",
    },
    "conservative": {
        "label": "Conservative",
        "grade": "premium",
        "grade_factor": 1.30,
        "margin_pct": 12.0,
        "contingency_pct": 10.0,
        "oh_pct": 15.0,
        "description": "Premium grades, max margins, highest contingency. "
                       "Use when scope is unclear or risk is high.",
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PackageBreakdown:
    """Cost breakdown for a single trade/package in a scenario."""
    trade: str
    item_count: int
    base_cost: float
    adjusted_cost: float
    contingency: float
    total: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BidScenario:
    """Single bid scenario result."""
    name: str                     # "aggressive", "balanced", "conservative"
    label: str                    # "Aggressive", "Balanced", "Conservative"
    description: str
    grade: str                    # "basic", "standard", "premium"
    grade_factor: float
    base_cost: float              # Sum of BOQ amounts before adjustments
    adjusted_cost: float          # After grade factor
    oh_pct: float
    oh_amount: float
    margin_pct: float
    margin_amount: float
    contingency_pct: float
    contingency_amount: float
    total_bid: float              # adjusted_cost + OH + margin + contingency
    risk_score: float             # 0–100 weighted from blockers/RFIs
    packages: List[PackageBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "grade": self.grade,
            "grade_factor": self.grade_factor,
            "base_cost": round(self.base_cost, 2),
            "adjusted_cost": round(self.adjusted_cost, 2),
            "oh_pct": self.oh_pct,
            "oh_amount": round(self.oh_amount, 2),
            "margin_pct": self.margin_pct,
            "margin_amount": round(self.margin_amount, 2),
            "contingency_pct": self.contingency_pct,
            "contingency_amount": round(self.contingency_amount, 2),
            "total_bid": round(self.total_bid, 2),
            "risk_score": round(self.risk_score, 1),
            "packages": [p.to_dict() for p in self.packages],
        }


@dataclass
class ScenarioComparison:
    """Three-scenario comparison result."""
    scenarios: List[BidScenario]
    base_cost: float              # Unadjusted total (balanced = base)
    spread_amount: float          # Conservative - Aggressive
    spread_pct: float             # Spread as % of balanced total
    recommendation: str           # "aggressive" | "balanced" | "conservative"
    recommendation_reason: str

    def to_dict(self) -> dict:
        return {
            "base_cost": round(self.base_cost, 2),
            "spread_amount": round(self.spread_amount, 2),
            "spread_pct": round(self.spread_pct, 1),
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# =============================================================================
# COST COMPUTATION
# =============================================================================

def compute_base_cost(payload: dict) -> Tuple[float, List[Dict[str, Any]]]:
    """Extract BOQ items and compute base cost from analysis payload.

    Looks in multiple payload locations for BOQ data:
    1. extraction_summary.boq_items
    2. boq_items (top-level)
    3. quantities (fallback)

    Returns:
        (total_base_cost, list_of_items_with_trade)
    """
    items = []

    # Try extraction_summary first
    ext_summary = payload.get("extraction_summary", {})
    boq_items = ext_summary.get("boq_items", [])

    # Fallback to top-level
    if not boq_items:
        boq_items = payload.get("boq_items", [])

    # Fallback to quantities
    if not boq_items:
        boq_items = payload.get("quantities", [])

    total = 0.0
    for item in boq_items:
        if not isinstance(item, dict):
            continue
        qty = _to_float(item.get("qty", item.get("quantity", 0)))
        rate = _to_float(item.get("rate", item.get("unit_rate", 0)))
        amount = _to_float(item.get("amount", 0))

        # Use explicit amount if available, else compute
        if amount > 0:
            cost = amount
        elif qty > 0 and rate > 0:
            cost = qty * rate
        else:
            continue

        trade = item.get("trade", item.get("package", item.get("category", "general")))
        items.append({
            "trade": str(trade).lower() if trade else "general",
            "description": item.get("description", ""),
            "qty": qty,
            "rate": rate,
            "amount": cost,
        })
        total += cost

    return total, items


def compute_package_breakdown(
    items: List[Dict[str, Any]],
    grade_factor: float,
    contingency_pct: float,
) -> List[PackageBreakdown]:
    """Group items by trade and compute per-package costs."""
    by_trade: Dict[str, List[Dict]] = {}
    for item in items:
        trade = item.get("trade", "general")
        by_trade.setdefault(trade, []).append(item)

    packages = []
    for trade, trade_items in sorted(by_trade.items()):
        base = sum(i["amount"] for i in trade_items)
        adjusted = base * grade_factor
        contingency = adjusted * contingency_pct / 100
        packages.append(PackageBreakdown(
            trade=trade,
            item_count=len(trade_items),
            base_cost=base,
            adjusted_cost=adjusted,
            contingency=contingency,
            total=adjusted + contingency,
        ))
    return packages


def compute_risk_score(payload: dict) -> float:
    """Compute aggregate risk score (0–100) from payload signals.

    Higher = more risky.
    """
    score = 0.0

    # Blockers contribute heavily
    blockers = payload.get("blockers", [])
    n_critical = sum(1 for b in blockers if isinstance(b, dict) and
                     b.get("severity", "").upper() in ("CRITICAL", "HIGH"))
    n_medium = sum(1 for b in blockers if isinstance(b, dict) and
                   b.get("severity", "").upper() == "MEDIUM")
    score += n_critical * 10 + n_medium * 5

    # RFI count
    rfis = payload.get("rfis", [])
    score += min(len(rfis) * 2, 30)  # Cap at 30

    # Low readiness score adds risk
    readiness = payload.get("readiness_score", 50)
    if isinstance(readiness, (int, float)):
        if readiness < 50:
            score += 20
        elif readiness < 70:
            score += 10

    # Low trade coverage adds risk
    trade_cov = payload.get("trade_coverage", [])
    low_cov = sum(1 for tc in trade_cov if isinstance(tc, dict) and
                  tc.get("coverage_pct", 100) < 50)
    score += low_cov * 5

    return min(100.0, score)


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

def generate_scenarios(
    payload: dict,
    custom_profiles: Optional[Dict[str, Dict]] = None,
) -> ScenarioComparison:
    """Generate 3 bid scenarios from analysis payload.

    Args:
        payload: Analysis payload dict.
        custom_profiles: Override default SCENARIO_PROFILES.

    Returns:
        ScenarioComparison with 3 BidScenario objects.
    """
    profiles = custom_profiles or SCENARIO_PROFILES
    base_cost, items = compute_base_cost(payload)
    risk = compute_risk_score(payload)

    scenarios = []
    for name in ("aggressive", "balanced", "conservative"):
        profile = profiles.get(name, SCENARIO_PROFILES[name])

        grade_factor = profile["grade_factor"]
        margin_pct = profile["margin_pct"]
        contingency_pct = profile["contingency_pct"]
        oh_pct = profile.get("oh_pct", 12.0)

        adjusted = base_cost * grade_factor
        oh_amount = adjusted * oh_pct / 100
        margin_amount = adjusted * margin_pct / 100
        contingency_amount = adjusted * contingency_pct / 100
        total_bid = adjusted + oh_amount + margin_amount + contingency_amount

        packages = compute_package_breakdown(items, grade_factor, contingency_pct)

        scenarios.append(BidScenario(
            name=name,
            label=profile["label"],
            description=profile.get("description", ""),
            grade=profile["grade"],
            grade_factor=grade_factor,
            base_cost=base_cost,
            adjusted_cost=adjusted,
            oh_pct=oh_pct,
            oh_amount=oh_amount,
            margin_pct=margin_pct,
            margin_amount=margin_amount,
            contingency_pct=contingency_pct,
            contingency_amount=contingency_amount,
            total_bid=total_bid,
            risk_score=risk,
            packages=packages,
        ))

    # Compute spread
    if len(scenarios) >= 3:
        spread = scenarios[2].total_bid - scenarios[0].total_bid
        balanced_total = scenarios[1].total_bid
        spread_pct = (spread / balanced_total * 100) if balanced_total > 0 else 0
    else:
        spread = 0
        spread_pct = 0

    # Recommendation
    rec, reason = recommend_scenario(payload, risk)

    return ScenarioComparison(
        scenarios=scenarios,
        base_cost=base_cost,
        spread_amount=spread,
        spread_pct=spread_pct,
        recommendation=rec,
        recommendation_reason=reason,
    )


def recommend_scenario(payload: dict, risk_score: float) -> Tuple[str, str]:
    """Pick recommended scenario based on risk and readiness.

    Returns:
        (scenario_name, reason_text)
    """
    readiness = payload.get("readiness_score", 50)
    if not isinstance(readiness, (int, float)):
        readiness = 50

    blockers = payload.get("blockers", [])
    n_critical = sum(1 for b in blockers if isinstance(b, dict) and
                     b.get("severity", "").upper() in ("CRITICAL", "HIGH"))

    decision = payload.get("decision", "CONDITIONAL")

    # Conservative if high risk
    if risk_score >= 60 or n_critical >= 3:
        return ("conservative",
                f"High risk score ({risk_score:.0f}/100) with {n_critical} critical blockers. "
                "Conservative pricing protects margins.")

    # Aggressive if very confident
    if readiness >= 80 and n_critical == 0 and risk_score < 25:
        return ("aggressive",
                f"Strong readiness ({readiness}/100), no critical blockers, low risk ({risk_score:.0f}). "
                "Aggressive pricing improves competitiveness.")

    # Balanced is the default
    return ("balanced",
            f"Moderate readiness ({readiness}/100) and risk ({risk_score:.0f}/100). "
            "Balanced pricing offers a good tradeoff between competitiveness and protection.")


# =============================================================================
# UTILITIES
# =============================================================================

def _to_float(val: Any) -> float:
    """Safely convert a value to float."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Handle Indian number format: "1,00,000" → 100000
        cleaned = val.replace(",", "").replace(" ", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0
