"""
Package Outputs / RFQ Splitter - Phase 25

Splits final BOQ into work packages and generates RFQ-ready sheets.

Packages:
- RCC/Structure
- Masonry/Plaster
- Waterproofing
- Flooring/Finishes
- Doors/Windows
- Plumbing
- Electrical
- External Works
- Prelims

For each package generates:
- pkg_boq.csv
- pkg_scope.md
- pkg_risks.md
- rfq_sheet.csv
"""

from .splitter import PackageSplitter, Package

__all__ = [
    "PackageSplitter",
    "Package",
    "run_package_splitter",
]


def run_package_splitter(
    project_id: str,
    priced_boq: list,
    prelims_items: list,
    bid_data: dict,
    output_dir,
) -> dict:
    """
    Run package splitter.

    Args:
        project_id: Project identifier
        priced_boq: Priced BOQ items
        prelims_items: Prelims items
        bid_data: Complete bid data
        output_dir: Output directory

    Returns:
        Splitter results
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize splitter
    splitter = PackageSplitter()

    # Split BOQ into packages
    packages = splitter.split(priced_boq, prelims_items, bid_data)

    # Export each package
    for package in packages:
        pkg_dir = output_dir / package.code
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Export package BOQ
        splitter.export_package_boq(package, pkg_dir / "pkg_boq.csv")

        # Export package scope
        splitter.export_package_scope(package, pkg_dir / "pkg_scope.md", bid_data)

        # Export package risks
        splitter.export_package_risks(package, pkg_dir / "pkg_risks.md")

        # Export RFQ sheet
        splitter.export_rfq_sheet(package, pkg_dir / "rfq_sheet.csv", bid_data)

        logger.info(f"Package {package.code}: {package.items_count} items, ₹{package.total_value:,.0f}")

    # Export package summary
    summary = splitter.generate_summary(packages)
    with open(output_dir / "packages_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Export package index
    index_md = splitter.generate_index(packages, project_id)
    with open(output_dir / "packages_index.md", "w") as f:
        f.write(index_md)

    return {
        "packages_created": len(packages),
        "total_items": sum(p.items_count for p in packages),
        "total_value": sum(p.total_value for p in packages),
        "packages": {p.code: {"items": p.items_count, "value": p.total_value} for p in packages},
    }
