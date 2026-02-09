"""
Bid Strategy Sheet Generator

Summarizes all risk analysis into actionable bid strategy:
- Safe packages to price aggressively
- Risky packages to keep margin
- Packages needing quotes
- Top 10 risk drivers
- Recommended bid approach

Output: bid_strategy.md

India-specific commercial decision support.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BidStrategyConfig:
    """Configuration for bid strategy generation."""
    # Margin recommendations by risk level
    margin_aggressive: float = 5.0  # Low risk
    margin_standard: float = 8.0  # Medium risk
    margin_conservative: float = 12.0  # High risk
    margin_defensive: float = 15.0  # Very high risk

    # Value thresholds (INR)
    high_value_threshold: float = 1000000  # 10 lakh
    very_high_value_threshold: float = 5000000  # 50 lakh

    # Confidence thresholds
    safe_confidence: float = 85.0
    cautious_confidence: float = 70.0


class BidStrategyGenerator:
    """
    Generates comprehensive bid strategy document.

    Sections:
    1. Bid Summary - Key numbers
    2. Safe Packages - Can price aggressively
    3. Risk Packages - Need margin protection
    4. Quote Requirements - SC quotes needed
    5. Top Risk Drivers - What could go wrong
    6. Pricing Recommendations - Margin guidance
    7. Go/No-Go Assessment - Should we bid?
    """

    def __init__(self, config: BidStrategyConfig = None):
        self.config = config or BidStrategyConfig()
        self.strategy_data = {}

    def generate(
        self,
        risk_profiles: List[Dict[str, Any]],
        sensitivity_results: Dict[str, Any] = None,
        quote_plan: List[Dict[str, Any]] = None,
        gate_result: Dict[str, Any] = None,
        project_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate bid strategy.

        Args:
            risk_profiles: Package risk analysis
            sensitivity_results: Rate sensitivity analysis
            quote_plan: Quote planning results
            gate_result: Bid gate assessment
            project_params: Project parameters

        Returns:
            Strategy data
        """
        sensitivity_results = sensitivity_results or {}
        quote_plan = quote_plan or []
        gate_result = gate_result or {}
        project_params = project_params or {}

        # Categorize packages
        safe_packages = [p for p in risk_profiles if p.get("risk_level") == "low"]
        medium_packages = [p for p in risk_profiles if p.get("risk_level") == "medium"]
        risky_packages = [p for p in risk_profiles if p.get("risk_level") in ["high", "very_high"]]

        # Calculate totals
        total_value = sum(p.get("package_value", 0) for p in risk_profiles)
        safe_value = sum(p.get("package_value", 0) for p in safe_packages)
        risky_value = sum(p.get("package_value", 0) for p in risky_packages)

        # Collect all risk drivers
        all_drivers = []
        for p in risk_profiles:
            for driver in p.get("risk_drivers", []):
                all_drivers.append({
                    "package": p.get("package_name", p.get("package")),
                    "driver": driver,
                    "risk_score": p.get("risk_score", 0),
                })

        # Sort and take top 10
        all_drivers.sort(key=lambda x: -x["risk_score"])
        top_drivers = all_drivers[:10]

        # Quote requirements
        urgent_quotes = [q for q in quote_plan if q.get("priority") == 1]
        recommended_quotes = [q for q in quote_plan if q.get("priority") == 2]

        # Calculate weighted contingency
        if total_value > 0:
            weighted_contingency = sum(
                p.get("suggested_contingency_pct", 5) * p.get("package_value", 0) / total_value
                for p in risk_profiles
            )
        else:
            weighted_contingency = 5.0

        # Go/No-Go assessment
        go_nogo = self._assess_go_nogo(
            gate_result, risk_profiles, quote_plan, project_params
        )

        self.strategy_data = {
            "project_name": project_params.get("project_name", "Project"),
            "total_value": total_value,
            "safe_packages": safe_packages,
            "medium_packages": medium_packages,
            "risky_packages": risky_packages,
            "safe_value_pct": (safe_value / total_value * 100) if total_value > 0 else 0,
            "risky_value_pct": (risky_value / total_value * 100) if total_value > 0 else 0,
            "top_drivers": top_drivers,
            "urgent_quotes": urgent_quotes,
            "recommended_quotes": recommended_quotes,
            "weighted_contingency": round(weighted_contingency, 2),
            "sensitivity": sensitivity_results,
            "gate_status": gate_result.get("status", "UNKNOWN"),
            "gate_score": gate_result.get("score", 0),
            "go_nogo": go_nogo,
        }

        return self.strategy_data

    def _assess_go_nogo(
        self,
        gate_result: Dict,
        risk_profiles: List[Dict],
        quote_plan: List[Dict],
        project_params: Dict,
    ) -> Dict[str, Any]:
        """Assess go/no-go recommendation."""
        positives = []
        negatives = []
        recommendation = "GO"

        # Gate status
        gate_status = gate_result.get("status", "UNKNOWN")
        if gate_status == "PASS":
            positives.append("Bid gate PASSED - all checks clear")
        elif gate_status == "PASS_WITH_RESERVATIONS":
            positives.append("Bid gate PASSED with reservations")
            negatives.append(f"{gate_result.get('reservations_count', 0)} reservations noted")
        else:
            negatives.append("Bid gate FAILED - critical issues")
            recommendation = "NO-GO"

        # Risk distribution
        very_high_risk = [p for p in risk_profiles if p.get("risk_level") == "very_high"]
        if len(very_high_risk) > 2:
            negatives.append(f"{len(very_high_risk)} packages at VERY HIGH risk")
            if recommendation != "NO-GO":
                recommendation = "CAUTION"

        low_risk = [p for p in risk_profiles if p.get("risk_level") == "low"]
        if len(low_risk) >= len(risk_profiles) // 2:
            positives.append(f"{len(low_risk)}/{len(risk_profiles)} packages at LOW risk")

        # Quote coverage
        urgent_no_quote = [q for q in quote_plan if q.get("priority") == 1 and not q.get("has_quote", False)]
        if urgent_no_quote:
            negatives.append(f"{len(urgent_no_quote)} urgent quotes still pending")
            if recommendation == "GO":
                recommendation = "CAUTION"

        # Duration/complexity
        duration = project_params.get("duration_months", 12)
        if duration > 24:
            negatives.append(f"Long duration project ({duration} months)")
        elif duration <= 12:
            positives.append(f"Short duration project ({duration} months)")

        return {
            "recommendation": recommendation,
            "positives": positives,
            "negatives": negatives,
            "confidence": "HIGH" if recommendation == "GO" else ("MEDIUM" if recommendation == "CAUTION" else "LOW"),
        }

    def export_markdown(self, output_path: Path) -> None:
        """Export bid strategy as markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        d = self.strategy_data
        c = self.config

        with open(output_path, "w") as f:
            # Header
            f.write("# BID STRATEGY SHEET\n\n")
            f.write(f"**Project**: {d.get('project_name', 'N/A')}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%d-%b-%Y %H:%M')}\n\n")

            # Go/No-Go banner
            go_nogo = d.get("go_nogo", {})
            rec = go_nogo.get("recommendation", "UNKNOWN")
            emoji = {"GO": "✅", "CAUTION": "⚠️", "NO-GO": "❌"}.get(rec, "❓")

            f.write("---\n")
            f.write(f"## {emoji} RECOMMENDATION: {rec}\n\n")

            if go_nogo.get("positives"):
                f.write("**Positives:**\n")
                for p in go_nogo.get("positives", []):
                    f.write(f"- ✓ {p}\n")

            if go_nogo.get("negatives"):
                f.write("\n**Concerns:**\n")
                for n in go_nogo.get("negatives", []):
                    f.write(f"- ✗ {n}\n")

            f.write("\n---\n\n")

            # Key Numbers
            f.write("## 1. KEY NUMBERS\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total BOQ Value | ₹{d.get('total_value', 0):,.0f} |\n")
            f.write(f"| Safe Package Value | {d.get('safe_value_pct', 0):.1f}% |\n")
            f.write(f"| Risky Package Value | {d.get('risky_value_pct', 0):.1f}% |\n")
            f.write(f"| Weighted Contingency | {d.get('weighted_contingency', 5):.1f}% |\n")
            f.write(f"| Gate Status | {d.get('gate_status', 'N/A')} ({d.get('gate_score', 0):.0f}/100) |\n\n")

            # Safe Packages
            f.write("## 2. SAFE PACKAGES (Price Aggressively)\n\n")
            safe = d.get("safe_packages", [])
            if safe:
                f.write("These packages have high confidence and low risk. Consider competitive pricing:\n\n")
                f.write("| Package | Value | Confidence | Suggested Margin |\n")
                f.write("|---------|-------|------------|------------------|\n")
                for p in safe:
                    value = p.get("package_value", 0)
                    value_str = f"₹{value/100000:.1f}L" if value >= 100000 else f"₹{value:,.0f}"
                    f.write(f"| {p.get('package_name', p.get('package'))} | {value_str} | {p.get('quantity_certainty', 0):.0f}% | {c.margin_aggressive:.0f}% |\n")
            else:
                f.write("*No packages classified as LOW risk*\n")
            f.write("\n")

            # Risky Packages
            f.write("## 3. RISKY PACKAGES (Protect Margin)\n\n")
            risky = d.get("risky_packages", [])
            if risky:
                f.write("These packages have significant risk. Maintain margin buffer:\n\n")
                f.write("| Package | Value | Risk Level | Risk Drivers | Margin |\n")
                f.write("|---------|-------|------------|--------------|--------|\n")
                for p in risky:
                    value = p.get("package_value", 0)
                    value_str = f"₹{value/100000:.1f}L" if value >= 100000 else f"₹{value:,.0f}"
                    drivers = "; ".join(p.get("risk_drivers", [])[:2])
                    margin = c.margin_defensive if p.get("risk_level") == "very_high" else c.margin_conservative
                    f.write(f"| {p.get('package_name', p.get('package'))} | {value_str} | {p.get('risk_level', '')} | {drivers[:40]} | {margin:.0f}% |\n")
            else:
                f.write("*No packages classified as HIGH/VERY HIGH risk*\n")
            f.write("\n")

            # Quote Requirements
            f.write("## 4. QUOTE REQUIREMENTS\n\n")
            urgent = d.get("urgent_quotes", [])
            recommended = d.get("recommended_quotes", [])

            if urgent:
                f.write("### Urgent (Before Submission)\n\n")
                for q in urgent:
                    f.write(f"- **{q.get('package_name', q.get('package'))}**: {q.get('reason', '')}\n")
                f.write("\n")

            if recommended:
                f.write("### Recommended\n\n")
                for q in recommended:
                    f.write(f"- {q.get('package_name', q.get('package'))}\n")
                f.write("\n")

            if not urgent and not recommended:
                f.write("*No subcontractor quotes urgently required*\n\n")

            # Top Risk Drivers
            f.write("## 5. TOP 10 RISK DRIVERS\n\n")
            drivers = d.get("top_drivers", [])
            if drivers:
                f.write("| # | Package | Risk Driver |\n")
                f.write("|---|---------|-------------|\n")
                for i, drv in enumerate(drivers[:10], 1):
                    f.write(f"| {i} | {drv.get('package', '')} | {drv.get('driver', '')} |\n")
            else:
                f.write("*No significant risk drivers identified*\n")
            f.write("\n")

            # Sensitivity
            f.write("## 6. RATE SENSITIVITY\n\n")
            sensitivity = d.get("sensitivity", {})
            if sensitivity.get("summary"):
                summary = sensitivity["summary"]
                f.write(f"- **Max Upside Risk**: {summary.get('max_upside_risk', {}).get('scenario', 'N/A')} ({summary.get('max_upside_risk', {}).get('impact_pct', 0):+.2f}%)\n")
                f.write(f"- **Max Downside Opportunity**: {summary.get('max_downside_opportunity', {}).get('scenario', 'N/A')} ({summary.get('max_downside_opportunity', {}).get('impact_pct', 0):.2f}%)\n")
            else:
                f.write("*Sensitivity analysis not performed*\n")
            f.write("\n")

            # Pricing Recommendations
            f.write("## 7. PRICING RECOMMENDATIONS\n\n")
            f.write("| Risk Level | Packages | Suggested Margin | Approach |\n")
            f.write("|------------|----------|------------------|----------|\n")
            f.write(f"| LOW | {len(d.get('safe_packages', []))} | {c.margin_aggressive:.0f}% | Price aggressively to win |\n")
            f.write(f"| MEDIUM | {len(d.get('medium_packages', []))} | {c.margin_standard:.0f}% | Standard competitive pricing |\n")
            f.write(f"| HIGH/VERY HIGH | {len(d.get('risky_packages', []))} | {c.margin_conservative:.0f}-{c.margin_defensive:.0f}% | Protect margin |\n\n")

            f.write(f"**Recommended Overall Margin**: {d.get('weighted_contingency', 5) + 5:.1f}% (contingency + profit)\n\n")

            # Action Items
            f.write("## 8. ACTION ITEMS\n\n")
            action_num = 1

            if urgent:
                f.write(f"{action_num}. **Urgent**: Obtain quotes for {len(urgent)} packages before submission\n")
                action_num += 1

            if d.get("gate_status") == "PASS_WITH_RESERVATIONS":
                f.write(f"{action_num}. Review and accept reservations in clarifications letter\n")
                action_num += 1

            high_drivers = [dr for dr in drivers if dr.get("risk_score", 0) > 60]
            if high_drivers:
                f.write(f"{action_num}. Address top risk drivers through RFIs or allowances\n")
                action_num += 1

            f.write(f"{action_num}. Final management review before submission\n")

            f.write("\n---\n\n")
            f.write("*This strategy sheet is for internal commercial decision-making. Do not share with client.*\n")

        logger.info(f"Bid strategy exported: {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary."""
        d = self.strategy_data
        return {
            "recommendation": d.get("go_nogo", {}).get("recommendation", "UNKNOWN"),
            "safe_packages": len(d.get("safe_packages", [])),
            "risky_packages": len(d.get("risky_packages", [])),
            "weighted_contingency": d.get("weighted_contingency", 0),
            "urgent_quotes": len(d.get("urgent_quotes", [])),
            "top_risk_driver": d.get("top_drivers", [{}])[0].get("driver", "None") if d.get("top_drivers") else "None",
        }


def run_bid_strategy(
    risk_profiles: List[Dict[str, Any]],
    sensitivity_results: Dict[str, Any] = None,
    quote_plan: List[Dict[str, Any]] = None,
    gate_result: Dict[str, Any] = None,
    project_params: Dict[str, Any] = None,
    output_path: Path = None,
) -> Dict[str, Any]:
    """
    Generate bid strategy.

    Args:
        risk_profiles: Package risk analysis
        sensitivity_results: Rate sensitivity
        quote_plan: Quote planning
        gate_result: Gate assessment
        project_params: Project info
        output_path: Output path for markdown

    Returns:
        Strategy results
    """
    generator = BidStrategyGenerator()
    strategy = generator.generate(
        risk_profiles, sensitivity_results, quote_plan, gate_result, project_params
    )

    if output_path:
        generator.export_markdown(output_path)

    return {
        "strategy": strategy,
        "summary": generator.get_summary(),
    }
