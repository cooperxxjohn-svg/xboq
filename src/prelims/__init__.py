"""
Prelims / General Conditions Generator - Calculate preliminary costs.

This module:
- Calculates time-based preliminary costs (staff, equipment, facilities)
- Applies project-specific factors (duration, size, complexity)
- Generates detailed prelims breakdown
- Outputs: prelims_boq.csv, prelims_breakdown.md, prelims_summary.json

India-specific: CPWD norms, ISR rates, typical prelims percentages.
"""

from .calculator import PrelimsCalculator, PrelimsItem
from .site_facilities import SiteFacilitiesCalculator
from .staff_costs import StaffCostsCalculator
from .equipment_costs import EquipmentCostsCalculator

__all__ = [
    "PrelimsCalculator",
    "PrelimsItem",
    "SiteFacilitiesCalculator",
    "StaffCostsCalculator",
    "EquipmentCostsCalculator",
    "run_prelims_engine",
]


def run_prelims_engine(
    project_id: str,
    project_value: float,
    duration_months: int,
    built_up_area_sqm: float,
    project_type: str,
    output_dir,
    owner_inputs: dict = None,
) -> dict:
    """
    Run the prelims generator.

    Args:
        project_id: Project identifier
        project_value: Total project value (construction cost)
        duration_months: Project duration in months
        built_up_area_sqm: Built-up area in sqm
        project_type: Project type (residential/commercial/etc.)
        output_dir: Output directory
        owner_inputs: Owner inputs for specific requirements

    Returns:
        Dict with prelims results
    """
    from pathlib import Path
    import json
    import csv
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize calculator
    calculator = PrelimsCalculator(
        project_value=project_value,
        duration_months=duration_months,
        built_up_area_sqm=built_up_area_sqm,
        project_type=project_type,
    )

    # 1. Calculate staff costs
    logger.info("Calculating staff costs...")
    staff_calc = StaffCostsCalculator()
    staff_items = staff_calc.calculate(
        duration_months=duration_months,
        project_value=project_value,
        project_type=project_type,
    )
    logger.info(f"Staff costs: {len(staff_items)} items")

    # 2. Calculate site facilities
    logger.info("Calculating site facilities...")
    facilities_calc = SiteFacilitiesCalculator()
    facility_items = facilities_calc.calculate(
        duration_months=duration_months,
        built_up_area_sqm=built_up_area_sqm,
        project_type=project_type,
    )
    logger.info(f"Site facilities: {len(facility_items)} items")

    # 3. Calculate equipment costs
    logger.info("Calculating equipment costs...")
    equipment_calc = EquipmentCostsCalculator()
    equipment_items = equipment_calc.calculate(
        duration_months=duration_months,
        built_up_area_sqm=built_up_area_sqm,
        project_type=project_type,
        floors=owner_inputs.get("project", {}).get("floors_above_ground", 4) if owner_inputs else 4,
    )
    logger.info(f"Equipment costs: {len(equipment_items)} items")

    # 4. Add insurance and bonds
    logger.info("Calculating insurance and bonds...")
    insurance_items = calculator.calculate_insurance_bonds(project_value, duration_months)
    logger.info(f"Insurance/bonds: {len(insurance_items)} items")

    # 5. Add miscellaneous prelims
    logger.info("Calculating miscellaneous prelims...")
    misc_items = calculator.calculate_miscellaneous(project_value, duration_months)
    logger.info(f"Miscellaneous: {len(misc_items)} items")

    # Combine all items
    all_items = staff_items + facility_items + equipment_items + insurance_items + misc_items

    # Calculate totals
    total_prelims = sum(item.amount for item in all_items)
    prelims_percent = (total_prelims / project_value * 100) if project_value > 0 else 0

    # Category breakdown
    by_category = {}
    for item in all_items:
        if item.category not in by_category:
            by_category[item.category] = {"items": 0, "amount": 0}
        by_category[item.category]["items"] += 1
        by_category[item.category]["amount"] += item.amount

    # 6. Export results

    # Prelims BOQ CSV
    with open(output_dir / "prelims_boq.csv", "w", newline="") as f:
        fieldnames = ["item_no", "description", "unit", "quantity", "rate", "amount", "category"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(all_items, 1):
            writer.writerow({
                "item_no": f"P-{i:03d}",
                "description": item.description,
                "unit": item.unit,
                "quantity": item.quantity,
                "rate": item.rate,
                "amount": item.amount,
                "category": item.category,
            })

    # Prelims summary JSON
    summary = {
        "project_id": project_id,
        "project_value": project_value,
        "duration_months": duration_months,
        "built_up_area_sqm": built_up_area_sqm,
        "total_prelims": round(total_prelims, 2),
        "prelims_percent": round(prelims_percent, 2),
        "by_category": {k: {"items": v["items"], "amount": round(v["amount"], 2)} for k, v in by_category.items()},
        "items_count": len(all_items),
    }

    with open(output_dir / "prelims_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Prelims breakdown markdown
    _export_prelims_breakdown(
        output_dir / "prelims_breakdown.md",
        project_id,
        all_items,
        summary,
    )

    return {
        "total_prelims": round(total_prelims, 2),
        "prelims_percent": round(prelims_percent, 2),
        "items_count": len(all_items),
        "by_category": by_category,
        "prelims_items": all_items,
    }


def _export_prelims_breakdown(
    output_path,
    project_id: str,
    items: list,
    summary: dict,
) -> None:
    """Export prelims breakdown as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Preliminary Costs Breakdown: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Project Value**: ₹{summary['project_value']:,.2f}\n")
        f.write(f"- **Duration**: {summary['duration_months']} months\n")
        f.write(f"- **Built-up Area**: {summary['built_up_area_sqm']:,.0f} sqm\n")
        f.write(f"- **Total Prelims**: ₹{summary['total_prelims']:,.2f}\n")
        f.write(f"- **Prelims %**: {summary['prelims_percent']:.2f}%\n\n")

        # Category breakdown
        f.write("## Category-wise Breakdown\n\n")
        f.write("| Category | Items | Amount (₹) | % of Prelims |\n")
        f.write("|----------|-------|------------|-------------|\n")

        total = summary['total_prelims']
        for cat, data in sorted(summary['by_category'].items(), key=lambda x: -x[1]['amount']):
            pct = (data['amount'] / total * 100) if total > 0 else 0
            f.write(f"| {cat.replace('_', ' ').title()} | {data['items']} | {data['amount']:,.2f} | {pct:.1f}% |\n")

        f.write(f"| **TOTAL** | **{summary['items_count']}** | **{total:,.2f}** | **100%** |\n\n")

        # Detailed items by category
        f.write("## Detailed Items\n\n")

        # Group by category
        by_cat = {}
        for item in items:
            if item.category not in by_cat:
                by_cat[item.category] = []
            by_cat[item.category].append(item)

        for cat, cat_items in sorted(by_cat.items()):
            f.write(f"### {cat.replace('_', ' ').title()}\n\n")
            f.write("| Description | Qty | Unit | Rate | Amount |\n")
            f.write("|-------------|-----|------|------|--------|\n")

            for item in cat_items:
                f.write(f"| {item.description} | {item.quantity:.1f} | {item.unit} | ")
                f.write(f"₹{item.rate:,.2f} | ₹{item.amount:,.2f} |\n")

            cat_total = sum(i.amount for i in cat_items)
            f.write(f"| **Sub-total** | | | | **₹{cat_total:,.2f}** |\n\n")

        # Notes
        f.write("## Notes\n\n")
        f.write("- Staff costs based on ISR (Indian Standard Rates) 2024\n")
        f.write("- Equipment rental rates as per prevailing market rates\n")
        f.write("- Insurance rates as per typical contractor policies\n")
        f.write("- All amounts are exclusive of GST\n")
