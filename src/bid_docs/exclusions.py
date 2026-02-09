"""
Exclusions & Assumptions Generator

Generates bid qualifications based on:
- Scope gaps and missing items
- Drawing/document gaps
- Owner input gaps
- Provisional items
- Project-specific conditions

Output:
- exclusions.md - Items NOT included in bid
- assumptions.md - Assumptions made in pricing

India-specific construction bid terminology.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExclusionItem:
    """A single exclusion item."""
    category: str
    item: str
    reason: str
    impact: str = ""


@dataclass
class AssumptionItem:
    """A single assumption item."""
    category: str
    assumption: str
    basis: str
    risk_if_wrong: str = ""


class ExclusionsAssumptionsGenerator:
    """
    Generates exclusions and assumptions lists for bid submission.

    Categories:
    - Scope exclusions (work not included)
    - Specification assumptions
    - Commercial assumptions
    - Site condition assumptions
    - Programme assumptions
    """

    # Standard Indian construction exclusions
    STANDARD_EXCLUSIONS = [
        ExclusionItem("Scope", "Furniture, fixtures, and furnishings (FF&F)", "Not part of civil construction contract", "Owner to procure separately"),
        ExclusionItem("Scope", "Modular kitchen including appliances", "Interior fit-out scope", "Owner to engage specialist"),
        ExclusionItem("Scope", "Air conditioning and HVAC system", "MEP specialist scope unless specified", "Separate tender/contract"),
        ExclusionItem("Scope", "Lift/elevator installation", "Specialised OEM supply and install", "Separate contract with OEM"),
        ExclusionItem("Scope", "Solar water heating / Solar PV panels", "Green building scope unless specified", "Separate quotation"),
        ExclusionItem("Scope", "Swimming pool and associated equipment", "Specialised construction", "Separate contract"),
        ExclusionItem("Scope", "Landscaping, gardening and soft areas", "Post-construction scope", "Horticulture specialist"),
        ExclusionItem("Scope", "External development beyond site boundary", "Not in contractor's scope", "Municipal/owner responsibility"),
        ExclusionItem("Commercial", "GST on works contract (currently 18%)", "Statutory levy on actuals", "To be paid as per invoice"),
        ExclusionItem("Commercial", "Government fees, approvals and liaison charges", "Owner's responsibility", "Including plan approval, completion certificate"),
        ExclusionItem("Site", "Rock excavation (if encountered)", "Unknown subsurface condition", "Actual measurement basis"),
        ExclusionItem("Site", "Dewatering beyond normal conditions", "Subject to site conditions", "If water table encountered"),
        ExclusionItem("Site", "Shoring and underpinning of adjacent structures", "If required", "Specialist assessment needed"),
        ExclusionItem("Scope", "Demolition of existing structures (if any)", "Unless specifically included", "Separate quotation"),
        ExclusionItem("Scope", "Asbestos/hazardous material removal", "Specialist disposal required", "If encountered"),
    ]

    # Standard assumptions
    STANDARD_ASSUMPTIONS = [
        AssumptionItem("Specifications", "Concrete grade M25 for all RCC work unless specified otherwise", "Standard residential grade", "Higher grade = cost increase"),
        AssumptionItem("Specifications", "Steel reinforcement Fe500D grade", "IS 1786 compliant", "Material cost stable"),
        AssumptionItem("Specifications", "Ceiling height 3.0m (clear) floor to ceiling", "Standard residential", "Affects plaster/paint quantities"),
        AssumptionItem("Specifications", "Floor to floor height 3.3m", "Including slab thickness", "Affects masonry quantities"),
        AssumptionItem("Specifications", "Slab thickness 150mm for typical floors", "Unless noted in drawings", "Structural adequacy assumed"),
        AssumptionItem("Site", "Normal soil conditions for foundation (no rock)", "As per standard practice", "Rock = extra cost"),
        AssumptionItem("Site", "Water table below foundation level", "Normal conditions", "Dewatering extra"),
        AssumptionItem("Site", "Clear access for material delivery and crane operation", "8-hour working window", "Restrictions = delay/cost"),
        AssumptionItem("Site", "Electricity and water available at site boundary", "For construction use", "To be provided by owner"),
        AssumptionItem("Programme", "Normal working hours (8 AM to 6 PM)", "6 days per week", "Night work = premium"),
        AssumptionItem("Programme", "Monsoon period work possible with precautions", "June-September", "Productivity factors applied"),
        AssumptionItem("Commercial", "Payment terms: Monthly RA bills within 15 days", "Standard practice", "Delayed payment = escalation"),
        AssumptionItem("Commercial", "Material price validity: 60 days from bid date", "Subject to market conditions", "Beyond = re-quote"),
        AssumptionItem("Commercial", "No price escalation for project duration < 18 months", "Fixed price contract", "Longer projects = escalation clause"),
    ]

    def __init__(self):
        self.exclusions: List[ExclusionItem] = []
        self.assumptions: List[AssumptionItem] = []

    def generate(
        self,
        scope_gaps: List[Dict[str, Any]] = None,
        missing_drawings: List[str] = None,
        missing_specs: List[str] = None,
        owner_input_gaps: List[str] = None,
        provisional_items: List[Dict[str, Any]] = None,
        project_params: Dict[str, Any] = None,
    ) -> None:
        """
        Generate exclusions and assumptions.

        Args:
            scope_gaps: Missing scope items
            missing_drawings: Missing drawing sheets
            missing_specs: Missing specifications
            owner_input_gaps: Missing owner inputs
            provisional_items: Items priced on provisional basis
            project_params: Project parameters
        """
        scope_gaps = scope_gaps or []
        missing_drawings = missing_drawings or []
        missing_specs = missing_specs or []
        owner_input_gaps = owner_input_gaps or []
        provisional_items = provisional_items or []
        project_params = project_params or {}

        # Start with standard exclusions
        self.exclusions = self.STANDARD_EXCLUSIONS.copy()
        self.assumptions = self.STANDARD_ASSUMPTIONS.copy()

        # Add scope gap exclusions
        for gap in scope_gaps:
            if isinstance(gap, dict):
                item = gap.get("item", gap.get("description", ""))
                reason = gap.get("reason", "Not detailed in drawings/specs")
            else:
                item = str(gap)
                reason = "Not detailed in drawings/specs"

            self.exclusions.append(ExclusionItem(
                category="Scope Gap",
                item=item,
                reason=reason,
                impact="To be confirmed with owner"
            ))

        # Add drawing-related assumptions
        if missing_drawings:
            for dwg in missing_drawings[:5]:  # Limit to top 5
                self.assumptions.append(AssumptionItem(
                    category="Drawings",
                    assumption=f"Drawing {dwg} to be provided before construction",
                    basis="Referenced but not received",
                    risk_if_wrong="Quantity/scope variation"
                ))

        # Add spec assumptions from gaps
        for spec in missing_specs[:5]:
            self.assumptions.append(AssumptionItem(
                category="Specifications",
                assumption=f"{spec} - standard specification assumed",
                basis="Not specified in tender",
                risk_if_wrong="Brand/quality variation"
            ))

        # Add provisional exclusions
        provisional_categories = set()
        for item in provisional_items:
            pkg = item.get("package", "other")
            if pkg not in provisional_categories:
                provisional_categories.add(pkg)
                self.assumptions.append(AssumptionItem(
                    category="Provisional",
                    assumption=f"{pkg.replace('_', ' ').title()} work on provisional allowance",
                    basis="Detailed drawings/specs not available",
                    risk_if_wrong="Actual vs provisional variation"
                ))

        # Add project-specific assumptions
        self._add_project_assumptions(project_params)

    def _add_project_assumptions(self, params: Dict[str, Any]) -> None:
        """Add project-specific assumptions."""
        # MEP assumptions
        if not params.get("mep_drawings_provided", True):
            self.assumptions.append(AssumptionItem(
                category="MEP",
                assumption="Plumbing and electrical on lumpsum allowance basis",
                basis="MEP drawings not provided",
                risk_if_wrong="Actual measurement may vary ±25%"
            ))

        # Structural assumptions
        if params.get("drawings_type") == "approval":
            self.assumptions.append(AssumptionItem(
                category="Drawings",
                assumption="Quantities based on approval drawings (not GFC)",
                basis="GFC drawings not issued",
                risk_if_wrong="Structural changes during GFC"
            ))

        # Duration assumptions
        duration = params.get("duration_months")
        if duration:
            self.assumptions.append(AssumptionItem(
                category="Programme",
                assumption=f"Project completion in {duration} months from site handover",
                basis="Contractual requirement",
                risk_if_wrong="Extension claims if delayed by owner"
            ))

        # Location assumptions
        location = params.get("location", "")
        if location:
            self.assumptions.append(AssumptionItem(
                category="Commercial",
                assumption=f"Material rates based on {location} market prices",
                basis="Q1 2024 market survey",
                risk_if_wrong="Regional price variation"
            ))

    def export_exclusions_md(self, output_path: Path) -> None:
        """Export exclusions as markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Group by category
        by_category: Dict[str, List[ExclusionItem]] = {}
        for exc in self.exclusions:
            if exc.category not in by_category:
                by_category[exc.category] = []
            by_category[exc.category].append(exc)

        with open(output_path, "w") as f:
            f.write("# BID EXCLUSIONS\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%d-%b-%Y')}*\n\n")
            f.write("The following items are **NOT INCLUDED** in our bid and shall be the responsibility of the Owner unless specifically agreed otherwise:\n\n")

            for category, items in by_category.items():
                f.write(f"## {category}\n\n")

                for i, exc in enumerate(items, 1):
                    f.write(f"{i}. **{exc.item}**\n")
                    f.write(f"   - Reason: {exc.reason}\n")
                    if exc.impact:
                        f.write(f"   - Note: {exc.impact}\n")
                    f.write("\n")

            f.write("---\n\n")
            f.write("*This exclusions list forms an integral part of our bid submission.*\n")

        logger.info(f"Exclusions exported: {output_path}")

    def export_assumptions_md(self, output_path: Path) -> None:
        """Export assumptions as markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Group by category
        by_category: Dict[str, List[AssumptionItem]] = {}
        for asm in self.assumptions:
            if asm.category not in by_category:
                by_category[asm.category] = []
            by_category[asm.category].append(asm)

        with open(output_path, "w") as f:
            f.write("# BID ASSUMPTIONS\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%d-%b-%Y')}*\n\n")
            f.write("The following assumptions have been made in preparing this bid. Any deviation from these assumptions may result in variation to the quoted price:\n\n")

            for category, items in by_category.items():
                f.write(f"## {category}\n\n")

                for i, asm in enumerate(items, 1):
                    f.write(f"{i}. **{asm.assumption}**\n")
                    f.write(f"   - Basis: {asm.basis}\n")
                    if asm.risk_if_wrong:
                        f.write(f"   - Risk: {asm.risk_if_wrong}\n")
                    f.write("\n")

            f.write("---\n\n")
            f.write("*These assumptions form an integral part of our bid submission. Please confirm acceptance or advise modifications.*\n")

        logger.info(f"Assumptions exported: {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of exclusions and assumptions."""
        exc_by_cat = {}
        for exc in self.exclusions:
            exc_by_cat[exc.category] = exc_by_cat.get(exc.category, 0) + 1

        asm_by_cat = {}
        for asm in self.assumptions:
            asm_by_cat[asm.category] = asm_by_cat.get(asm.category, 0) + 1

        return {
            "total_exclusions": len(self.exclusions),
            "total_assumptions": len(self.assumptions),
            "exclusions_by_category": exc_by_cat,
            "assumptions_by_category": asm_by_cat,
        }


def run_exclusions_assumptions(
    scope_gaps: List[Dict[str, Any]] = None,
    missing_drawings: List[str] = None,
    missing_specs: List[str] = None,
    owner_input_gaps: List[str] = None,
    provisional_items: List[Dict[str, Any]] = None,
    project_params: Dict[str, Any] = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Generate exclusions and assumptions.

    Args:
        Various gap inputs
        output_dir: Output directory

    Returns:
        Generation results
    """
    generator = ExclusionsAssumptionsGenerator()
    generator.generate(
        scope_gaps, missing_drawings, missing_specs,
        owner_input_gaps, provisional_items, project_params
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generator.export_exclusions_md(output_dir / "exclusions.md")
        generator.export_assumptions_md(output_dir / "assumptions.md")

    return {
        "exclusions": [
            {"category": e.category, "item": e.item, "reason": e.reason}
            for e in generator.exclusions
        ],
        "assumptions": [
            {"category": a.category, "assumption": a.assumption, "basis": a.basis}
            for a in generator.assumptions
        ],
        "summary": generator.get_summary(),
    }
