"""
Rate Sensitivity Engine

Simulates impact of rate changes on total bid:
- Steel rate +/- 10%
- Cement rate +/- 10%
- Labour rate +/- 10%
- Prelim duration +/- 20%

Output: sensitivity_report.md with impact analysis

India-specific material and labour rate analysis.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SensitivityScenario:
    """A single sensitivity scenario."""
    name: str
    description: str
    variable: str
    change_pct: float
    base_value: float
    new_value: float
    impact_amount: float
    impact_pct: float


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    steel_variation_pct: float = 10.0
    cement_variation_pct: float = 10.0
    labour_variation_pct: float = 10.0
    prelim_duration_variation_pct: float = 20.0

    # Material contribution to RCC cost (typical Indian breakdown)
    steel_pct_of_rcc: float = 45.0  # Steel is ~45% of RCC cost
    cement_pct_of_rcc: float = 20.0  # Cement ~20% of RCC cost

    # Labour contribution to various items
    labour_pct_of_masonry: float = 40.0
    labour_pct_of_rcc: float = 25.0
    labour_pct_of_finishes: float = 50.0
    labour_pct_of_plumbing: float = 35.0
    labour_pct_of_electrical: float = 40.0


class RateSensitivityEngine:
    """
    Analyzes sensitivity of bid to rate changes.

    Typical Indian construction cost breakdown:
    - RCC: Steel (45%), Cement (20%), Aggregates (15%), Labour (20%)
    - Masonry: Materials (60%), Labour (40%)
    - Finishes: Materials (50%), Labour (50%)
    """

    def __init__(self, config: SensitivityConfig = None):
        self.config = config or SensitivityConfig()
        self.scenarios: List[SensitivityScenario] = []
        self.base_values = {}

    def analyze_sensitivity(
        self,
        boq_items: List[Dict[str, Any]],
        prelims_items: List[Dict[str, Any]] = None,
        project_duration_months: int = 12,
    ) -> List[SensitivityScenario]:
        """
        Analyze sensitivity to rate changes.

        Args:
            boq_items: BOQ items with quantities and rates
            prelims_items: Preliminary items
            project_duration_months: Project duration

        Returns:
            List of sensitivity scenarios
        """
        self.scenarios = []
        prelims_items = prelims_items or []

        # Calculate base values
        self.base_values = self._calculate_base_values(boq_items, prelims_items)

        # Steel sensitivity
        self._analyze_steel_sensitivity(boq_items)

        # Cement sensitivity
        self._analyze_cement_sensitivity(boq_items)

        # Labour sensitivity
        self._analyze_labour_sensitivity(boq_items)

        # Prelim duration sensitivity
        self._analyze_duration_sensitivity(prelims_items, project_duration_months)

        return self.scenarios

    def _calculate_base_values(
        self,
        boq_items: List[Dict],
        prelims_items: List[Dict],
    ) -> Dict[str, float]:
        """Calculate base values for each category."""
        # RCC value
        rcc_value = sum(
            i.get("amount", 0) for i in boq_items
            if self._is_rcc_item(i)
        )

        # Masonry value
        masonry_value = sum(
            i.get("amount", 0) for i in boq_items
            if i.get("package") == "masonry"
        )

        # Finishes value
        finishes_value = sum(
            i.get("amount", 0) for i in boq_items
            if i.get("package") in ["finishes", "flooring"]
        )

        # MEP value
        mep_value = sum(
            i.get("amount", 0) for i in boq_items
            if i.get("package") in ["plumbing", "electrical", "hvac"]
        )

        # Total BOQ
        total_boq = sum(i.get("amount", 0) for i in boq_items)

        # Time-based prelims
        time_based_prelims = sum(
            i.get("amount", 0) for i in prelims_items
            if i.get("unit") in ["months", "man-months"]
        )

        # Total prelims
        total_prelims = sum(i.get("amount", 0) for i in prelims_items)

        # Grand total
        grand_total = total_boq + total_prelims

        return {
            "rcc_value": rcc_value,
            "masonry_value": masonry_value,
            "finishes_value": finishes_value,
            "mep_value": mep_value,
            "total_boq": total_boq,
            "time_based_prelims": time_based_prelims,
            "total_prelims": total_prelims,
            "grand_total": grand_total,
        }

    def _is_rcc_item(self, item: Dict) -> bool:
        """Check if item is RCC."""
        desc = item.get("description", "").lower()
        pkg = item.get("package", "")
        return "rcc" in desc or "concrete" in desc or pkg == "rcc"

    def _analyze_steel_sensitivity(self, boq_items: List[Dict]) -> None:
        """Analyze sensitivity to steel price changes."""
        c = self.config
        rcc_value = self.base_values["rcc_value"]
        grand_total = self.base_values["grand_total"]

        # Steel component of RCC
        steel_base = rcc_value * c.steel_pct_of_rcc / 100

        # Scenarios: +10%, -10%
        for change in [c.steel_variation_pct, -c.steel_variation_pct]:
            impact = steel_base * change / 100
            impact_pct = (impact / grand_total) * 100 if grand_total > 0 else 0

            direction = "increase" if change > 0 else "decrease"
            self.scenarios.append(SensitivityScenario(
                name=f"Steel +{change:.0f}%" if change > 0 else f"Steel {change:.0f}%",
                description=f"Steel reinforcement rate {direction} by {abs(change):.0f}%",
                variable="steel_rate",
                change_pct=change,
                base_value=steel_base,
                new_value=steel_base * (1 + change / 100),
                impact_amount=impact,
                impact_pct=round(impact_pct, 2),
            ))

    def _analyze_cement_sensitivity(self, boq_items: List[Dict]) -> None:
        """Analyze sensitivity to cement price changes."""
        c = self.config
        rcc_value = self.base_values["rcc_value"]
        masonry_value = self.base_values["masonry_value"]
        grand_total = self.base_values["grand_total"]

        # Cement component (in RCC + masonry plaster)
        cement_in_rcc = rcc_value * c.cement_pct_of_rcc / 100
        cement_in_masonry = masonry_value * 0.15  # ~15% of masonry is cement
        cement_base = cement_in_rcc + cement_in_masonry

        for change in [c.cement_variation_pct, -c.cement_variation_pct]:
            impact = cement_base * change / 100
            impact_pct = (impact / grand_total) * 100 if grand_total > 0 else 0

            direction = "increase" if change > 0 else "decrease"
            self.scenarios.append(SensitivityScenario(
                name=f"Cement +{change:.0f}%" if change > 0 else f"Cement {change:.0f}%",
                description=f"Cement rate {direction} by {abs(change):.0f}%",
                variable="cement_rate",
                change_pct=change,
                base_value=cement_base,
                new_value=cement_base * (1 + change / 100),
                impact_amount=impact,
                impact_pct=round(impact_pct, 2),
            ))

    def _analyze_labour_sensitivity(self, boq_items: List[Dict]) -> None:
        """Analyze sensitivity to labour rate changes."""
        c = self.config
        grand_total = self.base_values["grand_total"]

        # Calculate labour component
        labour_in_rcc = self.base_values["rcc_value"] * c.labour_pct_of_rcc / 100
        labour_in_masonry = self.base_values["masonry_value"] * c.labour_pct_of_masonry / 100
        labour_in_finishes = self.base_values["finishes_value"] * c.labour_pct_of_finishes / 100
        labour_in_mep = self.base_values["mep_value"] * 0.35  # Avg 35%

        labour_base = labour_in_rcc + labour_in_masonry + labour_in_finishes + labour_in_mep

        for change in [c.labour_variation_pct, -c.labour_variation_pct]:
            impact = labour_base * change / 100
            impact_pct = (impact / grand_total) * 100 if grand_total > 0 else 0

            direction = "increase" if change > 0 else "decrease"
            self.scenarios.append(SensitivityScenario(
                name=f"Labour +{change:.0f}%" if change > 0 else f"Labour {change:.0f}%",
                description=f"Labour rates {direction} by {abs(change):.0f}%",
                variable="labour_rate",
                change_pct=change,
                base_value=labour_base,
                new_value=labour_base * (1 + change / 100),
                impact_amount=impact,
                impact_pct=round(impact_pct, 2),
            ))

    def _analyze_duration_sensitivity(
        self,
        prelims_items: List[Dict],
        duration_months: int,
    ) -> None:
        """Analyze sensitivity to project duration changes."""
        c = self.config
        time_based = self.base_values["time_based_prelims"]
        grand_total = self.base_values["grand_total"]

        for change in [c.prelim_duration_variation_pct, -c.prelim_duration_variation_pct]:
            impact = time_based * change / 100
            impact_pct = (impact / grand_total) * 100 if grand_total > 0 else 0

            new_duration = int(duration_months * (1 + change / 100))
            direction = "increase" if change > 0 else "decrease"

            self.scenarios.append(SensitivityScenario(
                name=f"Duration +{change:.0f}%" if change > 0 else f"Duration {change:.0f}%",
                description=f"Project duration {direction} to {new_duration} months",
                variable="project_duration",
                change_pct=change,
                base_value=time_based,
                new_value=time_based * (1 + change / 100),
                impact_amount=impact,
                impact_pct=round(impact_pct, 2),
            ))

    def export_report(self, output_path: Path) -> None:
        """Export sensitivity report as markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        grand_total = self.base_values.get("grand_total", 0)

        with open(output_path, "w") as f:
            f.write("# Rate Sensitivity Analysis Report\n\n")

            f.write("## Base Values\n\n")
            f.write("| Component | Value (INR) | % of Total |\n")
            f.write("|-----------|-------------|------------|\n")

            components = [
                ("RCC Structural", self.base_values.get("rcc_value", 0)),
                ("Masonry & Plaster", self.base_values.get("masonry_value", 0)),
                ("Finishes & Flooring", self.base_values.get("finishes_value", 0)),
                ("MEP (Plumbing/Electrical)", self.base_values.get("mep_value", 0)),
                ("Preliminaries", self.base_values.get("total_prelims", 0)),
            ]

            for name, value in components:
                pct = (value / grand_total * 100) if grand_total > 0 else 0
                f.write(f"| {name} | {value:,.2f} | {pct:.1f}% |\n")

            f.write(f"| **Grand Total** | **{grand_total:,.2f}** | **100%** |\n\n")

            f.write("## Sensitivity Scenarios\n\n")
            f.write("| Scenario | Description | Base (INR) | Impact (INR) | Impact % |\n")
            f.write("|----------|-------------|------------|--------------|----------|\n")

            for s in self.scenarios:
                impact_sign = "+" if s.impact_amount >= 0 else ""
                f.write(f"| {s.name} | {s.description} | {s.base_value:,.0f} | {impact_sign}{s.impact_amount:,.0f} | {impact_sign}{s.impact_pct:.2f}% |\n")

            f.write("\n## Summary\n\n")

            # Group by variable
            by_variable = {}
            for s in self.scenarios:
                if s.variable not in by_variable:
                    by_variable[s.variable] = []
                by_variable[s.variable].append(s)

            f.write("### Impact by Variable\n\n")

            for var, scenarios in by_variable.items():
                var_name = var.replace("_", " ").title()
                max_impact = max(s.impact_pct for s in scenarios)
                min_impact = min(s.impact_pct for s in scenarios)

                f.write(f"**{var_name}**: ")
                f.write(f"Range of impact: {min_impact:+.2f}% to {max_impact:+.2f}%\n\n")

            f.write("### Risk Assessment\n\n")

            # Find highest impact scenarios
            sorted_scenarios = sorted(self.scenarios, key=lambda x: abs(x.impact_pct), reverse=True)

            f.write("**Top 3 Risk Factors:**\n\n")
            for i, s in enumerate(sorted_scenarios[:3], 1):
                f.write(f"{i}. **{s.name}**: ±{abs(s.impact_pct):.2f}% impact on bid\n")

            f.write("\n### Recommendations\n\n")

            # Steel
            steel_impact = max(abs(s.impact_pct) for s in self.scenarios if s.variable == "steel_rate")
            if steel_impact > 2:
                f.write(f"- **Steel**: High sensitivity ({steel_impact:.1f}%). Consider rate lock with supplier or escalation clause.\n")

            # Labour
            labour_impact = max(abs(s.impact_pct) for s in self.scenarios if s.variable == "labour_rate")
            if labour_impact > 2:
                f.write(f"- **Labour**: Moderate sensitivity ({labour_impact:.1f}%). Monitor market rates during execution.\n")

            # Duration
            duration_impact = max(abs(s.impact_pct) for s in self.scenarios if s.variable == "project_duration")
            if duration_impact > 1:
                f.write(f"- **Duration**: Programme-linked prelims ({duration_impact:.1f}%). Ensure realistic schedule.\n")

        logger.info(f"Sensitivity report exported: {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get sensitivity analysis summary."""
        if not self.scenarios:
            return {"error": "No analysis performed"}

        positive_scenarios = [s for s in self.scenarios if s.impact_pct > 0]
        negative_scenarios = [s for s in self.scenarios if s.impact_pct < 0]
        max_positive = max(positive_scenarios, key=lambda x: x.impact_pct)
        max_negative = min(negative_scenarios, key=lambda x: x.impact_pct)

        total_upside_risk = sum(s.impact_pct for s in self.scenarios if s.impact_pct > 0)
        total_downside_risk = sum(s.impact_pct for s in self.scenarios if s.impact_pct < 0)

        return {
            "scenarios_analyzed": len(self.scenarios),
            "grand_total": self.base_values.get("grand_total", 0),
            "max_upside_risk": {
                "scenario": max_positive.name,
                "impact_pct": max_positive.impact_pct,
                "impact_amount": max_positive.impact_amount,
            },
            "max_downside_opportunity": {
                "scenario": max_negative.name,
                "impact_pct": max_negative.impact_pct,
                "impact_amount": max_negative.impact_amount,
            },
            "total_upside_exposure": round(total_upside_risk, 2),
            "total_downside_opportunity": round(abs(total_downside_risk), 2),
        }


def run_sensitivity_analysis(
    boq_items: List[Dict[str, Any]],
    prelims_items: List[Dict[str, Any]] = None,
    project_duration_months: int = 12,
    output_path: Path = None,
) -> Dict[str, Any]:
    """
    Run rate sensitivity analysis.

    Args:
        boq_items: BOQ items
        prelims_items: Preliminary items
        project_duration_months: Project duration
        output_path: Path to export report

    Returns:
        Analysis results
    """
    engine = RateSensitivityEngine()
    scenarios = engine.analyze_sensitivity(
        boq_items, prelims_items, project_duration_months
    )

    if output_path:
        engine.export_report(output_path)

    return {
        "scenarios": [
            {
                "name": s.name,
                "variable": s.variable,
                "change_pct": s.change_pct,
                "impact_amount": s.impact_amount,
                "impact_pct": s.impact_pct,
            }
            for s in scenarios
        ],
        "base_values": engine.base_values,
        "summary": engine.get_summary(),
    }
