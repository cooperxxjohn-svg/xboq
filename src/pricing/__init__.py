"""
Rate Build-up / Pricing Engine - Calculate rates with full cost breakdown.

This module:
- Builds rates from material + labor + equipment + overhead
- Applies location multipliers for different cities
- Tracks rate assumptions and evidence
- Outputs: rate_analysis.json, priced_boq.csv, rate_buildups.md

India-specific: CPWD/DSR rate basis, ISR labor rates, regional material prices.
"""

from .rate_builder import RateBuilder, RateBuildUp
from .material_prices import MaterialPriceBook
from .labor_rates import LaborRateBook
from .location_factors import LocationMultiplier

__all__ = [
    "RateBuilder",
    "RateBuildUp",
    "MaterialPriceBook",
    "LaborRateBook",
    "LocationMultiplier",
    "run_pricing_engine",
]


def run_pricing_engine(
    project_id: str,
    unified_boq: list,
    owner_inputs: dict,
    output_dir,
    base_city: str = "Delhi",
) -> dict:
    """
    Run the pricing engine.

    Args:
        project_id: Project identifier
        unified_boq: Unified BOQ from alignment engine
        owner_inputs: Owner inputs with specifications
        output_dir: Output directory
        base_city: Base city for rate calculation (default Delhi/CPWD)

    Returns:
        Dict with pricing results
    """
    from pathlib import Path
    import json
    import csv
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get project location for multiplier
    project_city = owner_inputs.get("project", {}).get("location", {}).get("city", base_city)
    finish_grade = owner_inputs.get("finishes", {}).get("grade", "standard")

    # 1. Initialize pricing components
    logger.info("Loading material price book...")
    material_book = MaterialPriceBook()

    logger.info("Loading labor rate book...")
    labor_book = LaborRateBook()

    logger.info("Loading location multipliers...")
    location_mult = LocationMultiplier()
    city_factor = location_mult.get_multiplier(project_city)
    logger.info(f"Location factor for {project_city}: {city_factor}")

    # 2. Build rates for each BOQ item
    logger.info("Building rates for BOQ items...")
    rate_builder = RateBuilder(
        material_book=material_book,
        labor_book=labor_book,
        location_factor=city_factor,
        finish_grade=finish_grade,
    )

    priced_boq = []
    rate_buildups = []
    missing_rates = []

    for item in unified_boq:
        # Build rate for this item
        buildup = rate_builder.build_rate(item)

        if buildup.rate > 0:
            rate_buildups.append(buildup)

            # Calculate amount
            qty = float(item.get("quantity", 0))
            amount = qty * buildup.rate

            priced_item = item.copy()
            priced_item["rate"] = round(buildup.rate, 2)
            priced_item["amount"] = round(amount, 2)
            priced_item["rate_source"] = buildup.source
            priced_item["rate_confidence"] = buildup.confidence
            priced_boq.append(priced_item)
        else:
            # No rate found
            missing_rates.append({
                "item_id": item.get("unified_item_no", ""),
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
                "reason": "No matching rate found in database",
            })

            priced_item = item.copy()
            priced_item["rate"] = 0
            priced_item["amount"] = 0
            priced_item["rate_source"] = "NOT_FOUND"
            priced_item["rate_confidence"] = 0
            priced_boq.append(priced_item)

    logger.info(f"Priced {len(rate_buildups)} items, {len(missing_rates)} missing rates")

    # 3. Calculate totals by package
    totals_by_package = {}
    for item in priced_boq:
        pkg = item.get("package", "miscellaneous")
        if pkg not in totals_by_package:
            totals_by_package[pkg] = {"items": 0, "amount": 0}
        totals_by_package[pkg]["items"] += 1
        totals_by_package[pkg]["amount"] += item.get("amount", 0)

    grand_total = sum(p["amount"] for p in totals_by_package.values())

    # 4. Export results

    # Rate analysis JSON
    rate_analysis = {
        "project_id": project_id,
        "base_city": base_city,
        "project_city": project_city,
        "location_factor": city_factor,
        "finish_grade": finish_grade,
        "total_items": len(priced_boq),
        "priced_items": len(rate_buildups),
        "missing_rates": len(missing_rates),
        "grand_total": round(grand_total, 2),
        "totals_by_package": {k: {"items": v["items"], "amount": round(v["amount"], 2)} for k, v in totals_by_package.items()},
        "missing_rate_items": missing_rates,
    }

    with open(output_dir / "rate_analysis.json", "w") as f:
        json.dump(rate_analysis, f, indent=2)

    # Rate buildups JSON
    with open(output_dir / "rate_buildups.json", "w") as f:
        json.dump([b.to_dict() for b in rate_buildups], f, indent=2)

    # Priced BOQ CSV
    if priced_boq:
        with open(output_dir / "priced_boq.csv", "w", newline="") as f:
            fieldnames = ["unified_item_no", "description", "unit", "quantity", "rate", "amount", "package", "rate_source", "rate_confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(priced_boq)

    # Rate buildups markdown
    _export_rate_buildups(
        output_dir / "rate_buildups.md",
        project_id,
        rate_buildups,
        rate_analysis,
    )

    return {
        "total_items": len(priced_boq),
        "priced_items": len(rate_buildups),
        "missing_rates": len(missing_rates),
        "grand_total": round(grand_total, 2),
        "totals_by_package": totals_by_package,
        "location_factor": city_factor,
        "priced_boq": priced_boq,
    }


def _export_rate_buildups(
    output_path,
    project_id: str,
    rate_buildups: list,
    rate_analysis: dict,
) -> None:
    """Export rate buildups as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Rate Build-up Analysis: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Base City**: {rate_analysis['base_city']}\n")
        f.write(f"- **Project City**: {rate_analysis['project_city']}\n")
        f.write(f"- **Location Factor**: {rate_analysis['location_factor']:.2f}\n")
        f.write(f"- **Finish Grade**: {rate_analysis['finish_grade']}\n")
        f.write(f"- **Total Items**: {rate_analysis['total_items']}\n")
        f.write(f"- **Priced Items**: {rate_analysis['priced_items']}\n")
        f.write(f"- **Missing Rates**: {rate_analysis['missing_rates']}\n")
        f.write(f"- **Grand Total**: ₹{rate_analysis['grand_total']:,.2f}\n\n")

        # Package-wise totals
        f.write("## Package-wise Summary\n\n")
        f.write("| Package | Items | Amount (₹) | % of Total |\n")
        f.write("|---------|-------|------------|------------|\n")

        grand_total = rate_analysis["grand_total"]
        for pkg, data in sorted(rate_analysis["totals_by_package"].items(), key=lambda x: -x[1]["amount"]):
            pct = (data["amount"] / grand_total * 100) if grand_total > 0 else 0
            f.write(f"| {pkg.replace('_', ' ').title()} | {data['items']} | {data['amount']:,.2f} | {pct:.1f}% |\n")

        f.write(f"| **TOTAL** | **{rate_analysis['total_items']}** | **{grand_total:,.2f}** | **100%** |\n\n")

        # Sample rate buildups (first 20)
        f.write("## Sample Rate Build-ups\n\n")

        for buildup in rate_buildups[:20]:
            f.write(f"### {buildup.item_id}: {buildup.description[:50]}...\n\n")
            f.write(f"**Unit**: {buildup.unit}\n\n")
            f.write(f"**Rate**: ₹{buildup.rate:.2f} per {buildup.unit}\n\n")

            f.write("| Component | Description | Qty | Unit | Rate | Amount |\n")
            f.write("|-----------|-------------|-----|------|------|--------|\n")

            for comp in buildup.components:
                f.write(f"| {comp['type'].title()} | {comp['description'][:30]} | ")
                f.write(f"{comp['quantity']:.3f} | {comp['unit']} | ")
                f.write(f"{comp['rate']:.2f} | {comp['amount']:.2f} |\n")

            f.write(f"| **Sub-total** | | | | | **{buildup.subtotal:.2f}** |\n")
            if buildup.overheads > 0:
                f.write(f"| Overheads ({buildup.overhead_percent:.1f}%) | | | | | {buildup.overheads:.2f} |\n")
            f.write(f"| **TOTAL** | | | | | **{buildup.rate:.2f}** |\n\n")

            f.write(f"**Source**: {buildup.source}\n\n")
            f.write("---\n\n")

        # Missing rates
        if rate_analysis["missing_rate_items"]:
            f.write("## Missing Rates\n\n")
            f.write("The following items need rate input:\n\n")
            f.write("| Item | Description | Unit |\n")
            f.write("|------|-------------|------|\n")

            for item in rate_analysis["missing_rate_items"]:
                f.write(f"| {item['item_id']} | {item['description'][:50]} | {item['unit']} |\n")

            f.write("\n**Action**: Provide rates or quotes for these items.\n")
