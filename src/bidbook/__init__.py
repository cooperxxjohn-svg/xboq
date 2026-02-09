"""
Bid Book Export - Generate complete bid submission package.

This module:
- Compiles all bid components into submission-ready formats
- Generates Executive Summary
- Creates priced BOQ in required format
- Generates assumptions and clarifications document
- Outputs: bid_summary.md, priced_boq.xlsx, assumptions.md, bid_analysis.json

India-specific: Tender format compliance, GST calculations, technical submissions.
"""

from .exporter import BidBookExporter
from .summary_generator import SummaryGenerator
from .excel_generator import ExcelBOQGenerator

__all__ = [
    "BidBookExporter",
    "SummaryGenerator",
    "ExcelBOQGenerator",
    "run_bidbook_export",
]


def run_bidbook_export(
    project_id: str,
    bid_data: dict,
    output_dir,
) -> dict:
    """
    Run the bid book export.

    Args:
        project_id: Project identifier
        bid_data: Compiled bid data from all previous phases
        output_dir: Output directory

    Returns:
        Dict with export results
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize exporter
    exporter = BidBookExporter(project_id, output_dir)

    # 1. Generate Executive Summary
    logger.info("Generating executive summary...")
    summary_gen = SummaryGenerator()
    summary = summary_gen.generate(bid_data)

    with open(output_dir / "bid_summary.md", "w") as f:
        f.write(summary)
    logger.info("Executive summary generated")

    # 2. Generate Priced BOQ Excel
    logger.info("Generating priced BOQ Excel...")
    excel_gen = ExcelBOQGenerator()
    excel_path = excel_gen.generate(
        output_dir / "priced_boq.xlsx",
        bid_data.get("priced_boq", []),
        bid_data.get("prelims_items", []),
        bid_data.get("project_info", {}),
    )
    logger.info(f"Priced BOQ Excel generated: {excel_path}")

    # 3. Generate Assumptions Document
    logger.info("Generating assumptions document...")
    assumptions = exporter.generate_assumptions(bid_data)
    with open(output_dir / "assumptions.md", "w") as f:
        f.write(assumptions)
    logger.info("Assumptions document generated")

    # 4. Generate Clarifications/Deviations
    logger.info("Generating clarifications document...")
    clarifications = exporter.generate_clarifications(bid_data)
    with open(output_dir / "clarifications.md", "w") as f:
        f.write(clarifications)
    logger.info("Clarifications document generated")

    # 5. Generate Bid Analysis JSON
    logger.info("Generating bid analysis...")
    analysis = exporter.generate_analysis(bid_data)
    with open(output_dir / "bid_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Bid analysis generated")

    # 6. Generate RFI List
    logger.info("Generating RFI list...")
    rfis = exporter.compile_rfis(bid_data)
    with open(output_dir / "rfis_to_raise.md", "w") as f:
        f.write(rfis)
    logger.info(f"RFI list generated with {bid_data.get('total_rfis', 0)} items")

    # 7. Generate checklist
    checklist = exporter.generate_submission_checklist(bid_data)
    with open(output_dir / "submission_checklist.md", "w") as f:
        f.write(checklist)

    return {
        "outputs": [
            "bid_summary.md",
            "priced_boq.xlsx",
            "assumptions.md",
            "clarifications.md",
            "bid_analysis.json",
            "rfis_to_raise.md",
            "submission_checklist.md",
        ],
        "total_bid_value": bid_data.get("grand_total", 0),
        "rfis_count": bid_data.get("total_rfis", 0),
        "assumptions_count": len(bid_data.get("assumptions", [])),
    }
