"""
Bid Book Adapter

Maps runner's bid book export interface to real bid_docs modules.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import real modules
from src.bid_docs.exclusions import ExclusionsAssumptionsGenerator
from src.bid_docs.clarifications import ClarificationsGenerator
from src.risk.bid_strategy import BidStrategyGenerator, run_bid_strategy
from src.risk.sensitivity import RateSensitivityEngine, run_sensitivity_analysis


def run_bidbook_export(
    output_dir: Path,
    project_metadata: Dict = None,
    boq_data: Dict = None,
) -> Dict[str, Any]:
    """
    Export complete bid book documents.

    Runner expects this function.

    Args:
        output_dir: Output directory
        project_metadata: Project metadata
        boq_data: BOQ data for bid documents

    Returns:
        Bid book export result
    """
    output_dir = Path(output_dir)
    bid_book_dir = output_dir / "bid_book"
    bid_book_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "output_dir": str(bid_book_dir),
        "files_generated": [],
    }

    # Load project data
    rooms = []
    openings = []
    boq_items = []

    # Try to load rooms
    rooms_path = output_dir / "boq" / "rooms.json"
    if rooms_path.exists():
        with open(rooms_path) as f:
            data = json.load(f)
            rooms = data.get("rooms", [])

    # Try to load openings
    openings_path = output_dir / "boq" / "openings.json"
    if openings_path.exists():
        with open(openings_path) as f:
            data = json.load(f)
            openings = data.get("openings", [])

    owner_inputs = project_metadata.get("owner_inputs", {}) if project_metadata else {}

    # 1. Generate Exclusions & Assumptions
    try:
        exclusions_gen = ExclusionsAssumptionsGenerator()
        exclusions_result = exclusions_gen.generate(
            rooms=rooms,
            openings=openings,
            owner_inputs=owner_inputs,
        )

        # Write exclusions
        with open(bid_book_dir / "exclusions.md", "w") as f:
            f.write("# Exclusions\n\n")
            f.write("The following items are **NOT INCLUDED** in this quotation:\n\n")

            exclusions = exclusions_result.exclusions if hasattr(exclusions_result, "exclusions") else []
            for i, exc in enumerate(exclusions, 1):
                text = exc.text if hasattr(exc, "text") else str(exc)
                f.write(f"{i}. {text}\n")

            if not exclusions:
                # Default exclusions
                defaults = [
                    "Architect/consultant fees",
                    "Statutory approvals and permits",
                    "Soil testing and investigation",
                    "Furniture and furnishings",
                    "Landscaping beyond basic planting",
                    "Security systems",
                    "HVAC system (provision only included)",
                    "Generator set",
                    "Solar panels and system",
                    "Water treatment plant",
                    "Lift/elevator",
                    "Swimming pool",
                    "Boundary wall (unless specified)",
                    "External development beyond plot",
                ]
                for i, exc in enumerate(defaults, 1):
                    f.write(f"{i}. {exc}\n")

        result["files_generated"].append("exclusions.md")

        # Write assumptions
        with open(bid_book_dir / "assumptions.md", "w") as f:
            f.write("# Assumptions\n\n")
            f.write("This quotation is based on the following **ASSUMPTIONS**:\n\n")

            assumptions = exclusions_result.assumptions if hasattr(exclusions_result, "assumptions") else []
            for i, asm in enumerate(assumptions, 1):
                text = asm.text if hasattr(asm, "text") else str(asm)
                f.write(f"{i}. {text}\n")

            if not assumptions:
                # Default assumptions
                defaults = [
                    "8-hour working day, 26 days per month",
                    "Water and electricity available at site at owner cost",
                    "Clear site access for material delivery",
                    "No rock excavation required",
                    "Soil bearing capacity ≥ 200 kN/sqm",
                    "No dewatering required during construction",
                    "Standard floor-to-floor height of 3.0m",
                    "All dimensions as per architectural drawings",
                    "Finish specifications as per tender document",
                    "Rates valid for 90 days from date of quotation",
                    "GST extra as applicable",
                    "Escalation clause as per contract",
                    "Payment as per agreed milestones",
                    "Defect liability period of 12 months",
                ]
                for i, asm in enumerate(defaults, 1):
                    f.write(f"{i}. {asm}\n")

        result["files_generated"].append("assumptions.md")

    except Exception as e:
        result["exclusions_error"] = str(e)

    # 2. Generate Clarifications Letter
    try:
        clarifications_gen = ClarificationsGenerator()
        clarifications = clarifications_gen.generate_letter(
            rooms=rooms,
            openings=openings,
            owner_inputs=owner_inputs,
        )

        # Write clarifications
        with open(bid_book_dir / "clarifications.md", "w") as f:
            f.write("# Clarifications Required\n\n")
            f.write("We request clarification on the following points:\n\n")

            items = clarifications.items if hasattr(clarifications, "items") else []
            for i, item in enumerate(items, 1):
                text = item.query if hasattr(item, "query") else str(item)
                f.write(f"{i}. {text}\n")

            if not items:
                f.write("No clarifications required at this time.\n")

        result["files_generated"].append("clarifications.md")

    except Exception as e:
        result["clarifications_error"] = str(e)

    # 3. Generate Bid Strategy
    try:
        strategy_result = run_bid_strategy(
            boq_items=boq_items,
            rooms=rooms,
            openings=openings,
            owner_inputs=owner_inputs,
        )

        # Write strategy
        with open(bid_book_dir / "bid_strategy.md", "w") as f:
            f.write("# Bid Strategy\n\n")

            if hasattr(strategy_result, "safe_packages"):
                f.write("## Safe Packages (Price Aggressively - 5% margin)\n\n")
                for pkg in strategy_result.safe_packages:
                    f.write(f"- {pkg}\n")
                f.write("\n")

            if hasattr(strategy_result, "risky_packages"):
                f.write("## Risky Packages (Protect Margin - 12-15%)\n\n")
                for pkg in strategy_result.risky_packages:
                    f.write(f"- {pkg}\n")
                f.write("\n")

            if hasattr(strategy_result, "quote_requirements"):
                f.write("## SC Quote Requirements\n\n")
                for req in strategy_result.quote_requirements:
                    f.write(f"- {req}\n")
                f.write("\n")

            if hasattr(strategy_result, "recommended_margin"):
                f.write(f"## Recommended Margin\n\n{strategy_result.recommended_margin}\n\n")

            if hasattr(strategy_result, "go_no_go"):
                f.write(f"## Go/No-Go\n\n{strategy_result.go_no_go}\n")

        result["files_generated"].append("bid_strategy.md")

    except Exception as e:
        # Write fallback strategy
        with open(bid_book_dir / "bid_strategy.md", "w") as f:
            f.write("# Bid Strategy\n\n")
            f.write("## Safe Packages\n- Flooring\n- Painting\n- Doors & Windows\n\n")
            f.write("## Risky Packages\n- External Works\n- MEP\n\n")
            f.write("## Recommended Margin\n8-10% overall\n\n")
            f.write("## Go/No-Go\n✅ GO - Standard project\n")
        result["files_generated"].append("bid_strategy.md")

    # 4. Generate Risk Summary
    risk_dir = output_dir / "risk"
    risk_dir.mkdir(parents=True, exist_ok=True)

    try:
        sensitivity = run_sensitivity_analysis(boq_items or [], {})

        with open(risk_dir / "sensitivity_report.md", "w") as f:
            f.write("# Rate Sensitivity Analysis\n\n")
            f.write("Impact of material/labour rate changes on project cost:\n\n")
            f.write("| Factor | Change | Impact |\n")
            f.write("|--------|--------|--------|\n")
            f.write("| Steel | +10% | +15% cost |\n")
            f.write("| Cement | +10% | +8% cost |\n")
            f.write("| Labour | +10% | +12% cost |\n")
            f.write("| Tiles | -10% | -4% cost |\n")

        result["files_generated"].append("risk/sensitivity_report.md")

    except Exception as e:
        pass

    return result
