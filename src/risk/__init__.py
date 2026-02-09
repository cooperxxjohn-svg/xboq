"""
Bid Strategy Layer - Risk Analysis and Pricing Guidance

Components:
1. Risk Pricing Engine - Package-wise risk and contingency recommendations
2. Rate Sensitivity Engine - Impact analysis for material/labour rate changes
3. Quote Planning Engine - RFQ prioritization based on risk
4. Exclusions & Assumptions - Auto-generated bid qualifications
5. Bid Strategy Sheet - Summary for commercial decision-making

India-specific construction bidding terminology.
"""

from .pricing import RiskPricingEngine, run_risk_pricing
from .sensitivity import RateSensitivityEngine, run_sensitivity_analysis
from .quote_plan import QuotePlanningEngine, run_quote_planning
from .bid_strategy import BidStrategyGenerator, run_bid_strategy

__all__ = [
    "RiskPricingEngine",
    "run_risk_pricing",
    "RateSensitivityEngine",
    "run_sensitivity_analysis",
    "QuotePlanningEngine",
    "run_quote_planning",
    "BidStrategyGenerator",
    "run_bid_strategy",
]
