"""
Quote Planning Engine

Generates RFQ prioritization plan based on:
- Risk level per package
- Scope gaps
- Provisional items
- Missing specifications

Output: quote_plan.csv with:
- package
- needs_quote (yes/no/optional)
- reason
- rfq_sheet_path
- risk_level
- priority (1=urgent, 2=recommended, 3=optional)

India-specific subcontractor quoting practices.
"""

import csv
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class QuotePriority(Enum):
    """Quote priority levels."""
    URGENT = 1  # Must get quotes before bid submission
    RECOMMENDED = 2  # Strongly recommended
    OPTIONAL = 3  # Nice to have
    NOT_REQUIRED = 4  # Can price in-house


@dataclass
class QuotePlanEntry:
    """Quote plan entry for a package."""
    package: str
    package_name: str
    needs_quote: str  # yes, no, optional
    reason: str
    rfq_sheet_path: str
    risk_level: str
    priority: int
    package_value: float
    provisional_value: float
    confidence: float


class QuotePlanningEngine:
    """
    Generates subcontractor quote requirements based on risk and gaps.

    Factors:
    1. Risk level - High/Very High packages need quotes
    2. Provisional items - Items on allowance need quotes
    3. Specialization - MEP, waterproofing typically subcontracted
    4. Value threshold - High-value packages warrant quotes
    5. Scope gaps - Unclear scope needs subcontractor input
    """

    # Packages typically subcontracted in India
    TYPICALLY_SUBCONTRACTED = {
        "plumbing": "Specialised MEP trade",
        "electrical": "Specialised MEP trade",
        "hvac": "Specialised MEP trade",
        "firefighting": "Specialised MEP trade with licensing",
        "lift": "OEM/specialised vendor",
        "waterproof": "Specialised application",
        "doors_windows": "Fabrication and installation",
        "flooring": "Specialised laying (marble/vitrified)",
        "aluminium": "Fabrication trade",
        "structural_steel": "Fabrication and erection",
    }

    # Package display names
    PACKAGE_NAMES = {
        "rcc": "RCC Structural",
        "masonry": "Masonry & Plastering",
        "waterproof": "Waterproofing",
        "flooring": "Flooring & Tiling",
        "doors_windows": "Doors & Windows",
        "plumbing": "Plumbing Works",
        "electrical": "Electrical Works",
        "external": "External Development",
        "finishes": "Finishes & Painting",
        "prelims": "Preliminaries",
        "hvac": "HVAC",
        "firefighting": "Fire Fighting",
        "lift": "Lifts & Elevators",
    }

    def __init__(self, packages_dir: Path = None):
        self.packages_dir = packages_dir
        self.quote_plan: List[QuotePlanEntry] = []

    def generate_plan(
        self,
        boq_items: List[Dict[str, Any]],
        risk_profiles: List[Dict[str, Any]] = None,
        scope_gaps: List[Dict[str, Any]] = None,
        value_threshold: float = 500000,  # 5 lakh minimum for quote
    ) -> List[QuotePlanEntry]:
        """
        Generate quote plan for all packages.

        Args:
            boq_items: BOQ items
            risk_profiles: Risk analysis results per package
            scope_gaps: List of scope gaps
            value_threshold: Minimum package value to warrant quote

        Returns:
            List of QuotePlanEntry
        """
        self.quote_plan = []
        risk_profiles = risk_profiles or []
        scope_gaps = scope_gaps or []

        # Group items by package
        packages = self._group_by_package(boq_items)

        # Index risk profiles
        risk_by_pkg = {r["package"]: r for r in risk_profiles}

        # Index scope gaps
        gaps_by_pkg = self._index_gaps_by_package(scope_gaps)

        for pkg_key, items in packages.items():
            entry = self._evaluate_package(
                pkg_key,
                items,
                risk_by_pkg.get(pkg_key, {}),
                gaps_by_pkg.get(pkg_key, []),
                value_threshold,
            )
            self.quote_plan.append(entry)

        # Sort by priority then value
        self.quote_plan.sort(key=lambda x: (x.priority, -x.package_value))

        return self.quote_plan

    def _group_by_package(self, boq_items: List[Dict]) -> Dict[str, List[Dict]]:
        """Group BOQ items by package."""
        packages = {}
        for item in boq_items:
            pkg = item.get("package", "other")
            if pkg not in packages:
                packages[pkg] = []
            packages[pkg].append(item)
        return packages

    def _index_gaps_by_package(self, gaps: List[Dict]) -> Dict[str, List[Dict]]:
        """Index scope gaps by package."""
        index = {}
        for gap in gaps:
            pkg = gap.get("package", "general")
            if pkg not in index:
                index[pkg] = []
            index[pkg].append(gap)
        return index

    def _evaluate_package(
        self,
        pkg_key: str,
        items: List[Dict],
        risk_profile: Dict,
        scope_gaps: List[Dict],
        value_threshold: float,
    ) -> QuotePlanEntry:
        """Evaluate quote requirements for a package."""
        pkg_name = self.PACKAGE_NAMES.get(pkg_key, pkg_key.replace("_", " ").title())

        # Calculate metrics
        pkg_value = sum(i.get("amount", 0) for i in items)
        provisional_value = sum(
            i.get("amount", 0) for i in items
            if i.get("is_provisional", False)
        )
        avg_confidence = (
            sum(i.get("confidence", 0.7) for i in items) / len(items)
            if items else 0.5
        )

        risk_level = risk_profile.get("risk_level", "medium")
        risk_score = risk_profile.get("risk_score", 50)

        # Determine if quote needed and why
        needs_quote, reason, priority = self._determine_quote_need(
            pkg_key, pkg_value, provisional_value, avg_confidence,
            risk_level, risk_score, scope_gaps, value_threshold
        )

        # RFQ sheet path
        rfq_path = ""
        if needs_quote in ["yes", "optional"] and self.packages_dir:
            rfq_file = self.packages_dir / pkg_key / f"{pkg_key}_rfq_sheet.csv"
            if rfq_file.exists():
                rfq_path = str(rfq_file)
            else:
                rfq_path = f"packages/{pkg_key}/{pkg_key}_rfq_sheet.csv"

        return QuotePlanEntry(
            package=pkg_key,
            package_name=pkg_name,
            needs_quote=needs_quote,
            reason=reason,
            rfq_sheet_path=rfq_path,
            risk_level=risk_level,
            priority=priority,
            package_value=round(pkg_value, 2),
            provisional_value=round(provisional_value, 2),
            confidence=round(avg_confidence * 100, 1),
        )

    def _determine_quote_need(
        self,
        pkg_key: str,
        pkg_value: float,
        prov_value: float,
        confidence: float,
        risk_level: str,
        risk_score: float,
        scope_gaps: List[Dict],
        threshold: float,
    ) -> tuple:
        """Determine if package needs quote, why, and priority."""

        reasons = []
        priority = QuotePriority.NOT_REQUIRED.value

        # Check if typically subcontracted
        if pkg_key in self.TYPICALLY_SUBCONTRACTED:
            reasons.append(self.TYPICALLY_SUBCONTRACTED[pkg_key])
            priority = min(priority, QuotePriority.RECOMMENDED.value)

        # Check risk level
        if risk_level in ["high", "very_high"]:
            reasons.append(f"High risk package ({risk_level})")
            priority = min(priority, QuotePriority.URGENT.value if risk_level == "very_high" else QuotePriority.RECOMMENDED.value)

        # Check provisional percentage
        if pkg_value > 0 and prov_value / pkg_value > 0.3:
            prov_pct = prov_value / pkg_value * 100
            reasons.append(f"High provisional ({prov_pct:.0f}%)")
            priority = min(priority, QuotePriority.URGENT.value)
        elif pkg_value > 0 and prov_value / pkg_value > 0.15:
            prov_pct = prov_value / pkg_value * 100
            reasons.append(f"Provisional items ({prov_pct:.0f}%)")
            priority = min(priority, QuotePriority.RECOMMENDED.value)

        # Check confidence
        if confidence < 0.6:
            reasons.append(f"Low quantity confidence ({confidence*100:.0f}%)")
            priority = min(priority, QuotePriority.RECOMMENDED.value)

        # Check scope gaps
        if len(scope_gaps) > 2:
            reasons.append(f"{len(scope_gaps)} scope gaps")
            priority = min(priority, QuotePriority.RECOMMENDED.value)

        # Check value threshold
        if pkg_value > threshold * 3 and priority == QuotePriority.NOT_REQUIRED.value:
            reasons.append(f"High value package (>{threshold*3/100000:.0f}L)")
            priority = QuotePriority.OPTIONAL.value

        # Determine needs_quote
        if priority == QuotePriority.URGENT.value:
            needs_quote = "yes"
        elif priority == QuotePriority.RECOMMENDED.value:
            needs_quote = "yes"
        elif priority == QuotePriority.OPTIONAL.value:
            needs_quote = "optional"
        else:
            needs_quote = "no"
            reasons = ["In-house capability / low risk"]

        return needs_quote, "; ".join(reasons), priority

    def export_csv(self, output_path: Path) -> None:
        """Export quote plan to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "package", "package_name", "needs_quote", "reason",
                "rfq_sheet_path", "risk_level", "priority",
                "package_value", "provisional_value", "confidence"
            ])

            for e in self.quote_plan:
                writer.writerow([
                    e.package, e.package_name, e.needs_quote, e.reason,
                    e.rfq_sheet_path, e.risk_level, e.priority,
                    f"{e.package_value:.2f}", f"{e.provisional_value:.2f}",
                    f"{e.confidence:.1f}"
                ])

        logger.info(f"Quote plan exported: {output_path}")

    def export_markdown(self, output_path: Path) -> None:
        """Export quote plan as markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        urgent = [e for e in self.quote_plan if e.priority == QuotePriority.URGENT.value]
        recommended = [e for e in self.quote_plan if e.priority == QuotePriority.RECOMMENDED.value]
        optional = [e for e in self.quote_plan if e.priority == QuotePriority.OPTIONAL.value]

        with open(output_path, "w") as f:
            f.write("# Subcontractor Quote Plan\n\n")

            total_value = sum(e.package_value for e in self.quote_plan)
            quote_value = sum(e.package_value for e in self.quote_plan if e.needs_quote == "yes")

            f.write("## Summary\n\n")
            f.write(f"- **Total Packages**: {len(self.quote_plan)}\n")
            f.write(f"- **Urgent Quotes**: {len(urgent)}\n")
            f.write(f"- **Recommended Quotes**: {len(recommended)}\n")
            f.write(f"- **Optional Quotes**: {len(optional)}\n")
            f.write(f"- **Quote Coverage**: {quote_value/total_value*100:.1f}% of BOQ value\n\n")

            if urgent:
                f.write("## Urgent (Must Have Before Submission)\n\n")
                f.write("| Package | Value | Reason | RFQ Sheet |\n")
                f.write("|---------|-------|--------|----------|\n")
                for e in urgent:
                    value_str = f"₹{e.package_value/100000:.1f}L" if e.package_value >= 100000 else f"₹{e.package_value:,.0f}"
                    rfq = f"[RFQ]({e.rfq_sheet_path})" if e.rfq_sheet_path else "-"
                    f.write(f"| {e.package_name} | {value_str} | {e.reason[:50]} | {rfq} |\n")
                f.write("\n")

            if recommended:
                f.write("## Recommended\n\n")
                f.write("| Package | Value | Reason | Risk |\n")
                f.write("|---------|-------|--------|------|\n")
                for e in recommended:
                    value_str = f"₹{e.package_value/100000:.1f}L" if e.package_value >= 100000 else f"₹{e.package_value:,.0f}"
                    f.write(f"| {e.package_name} | {value_str} | {e.reason[:40]} | {e.risk_level} |\n")
                f.write("\n")

            if optional:
                f.write("## Optional\n\n")
                f.write("| Package | Value | Reason |\n")
                f.write("|---------|-------|--------|\n")
                for e in optional:
                    value_str = f"₹{e.package_value/100000:.1f}L" if e.package_value >= 100000 else f"₹{e.package_value:,.0f}"
                    f.write(f"| {e.package_name} | {value_str} | {e.reason[:50]} |\n")
                f.write("\n")

            # In-house
            inhouse = [e for e in self.quote_plan if e.needs_quote == "no"]
            if inhouse:
                f.write("## In-House Pricing\n\n")
                f.write("| Package | Value | Confidence |\n")
                f.write("|---------|-------|------------|\n")
                for e in inhouse:
                    value_str = f"₹{e.package_value/100000:.1f}L" if e.package_value >= 100000 else f"₹{e.package_value:,.0f}"
                    f.write(f"| {e.package_name} | {value_str} | {e.confidence:.0f}% |\n")

        logger.info(f"Quote plan markdown exported: {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get quote plan summary."""
        if not self.quote_plan:
            return {"error": "No plan generated"}

        urgent = [e for e in self.quote_plan if e.priority == QuotePriority.URGENT.value]
        needs_quote = [e for e in self.quote_plan if e.needs_quote == "yes"]

        return {
            "total_packages": len(self.quote_plan),
            "urgent_quotes": len(urgent),
            "quotes_needed": len(needs_quote),
            "total_value": sum(e.package_value for e in self.quote_plan),
            "quote_value": sum(e.package_value for e in needs_quote),
            "urgent_packages": [e.package_name for e in urgent],
        }


def run_quote_planning(
    boq_items: List[Dict[str, Any]],
    risk_profiles: List[Dict[str, Any]] = None,
    scope_gaps: List[Dict[str, Any]] = None,
    packages_dir: Path = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run quote planning.

    Args:
        boq_items: BOQ items
        risk_profiles: Risk analysis per package
        scope_gaps: Scope gaps
        packages_dir: Directory with RFQ sheets
        output_dir: Output directory

    Returns:
        Quote plan results
    """
    engine = QuotePlanningEngine(packages_dir=packages_dir)
    plan = engine.generate_plan(boq_items, risk_profiles, scope_gaps)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        engine.export_csv(output_dir / "quote_plan.csv")
        engine.export_markdown(output_dir / "quote_plan.md")

    return {
        "quote_plan": [
            {
                "package": e.package,
                "package_name": e.package_name,
                "needs_quote": e.needs_quote,
                "reason": e.reason,
                "priority": e.priority,
                "package_value": e.package_value,
            }
            for e in plan
        ],
        "summary": engine.get_summary(),
    }
