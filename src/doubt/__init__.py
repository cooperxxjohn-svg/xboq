"""
Estimator Doubt Engine - Flag missing critical sheets and generate high-priority RFIs.

This module detects:
- Missing section/elevation sheets
- Missing site plan
- Missing plumbing layouts
- Missing electrical layouts
- Missing structural sheets
- Missing schedule sheets
- Missing detail sheets

India-specific expectations for complete drawing sets.
"""

from .detector import DoubtDetector, MissingSheet, DoubtReport
from .rfi_generator import DoubtRFIGenerator

__all__ = [
    "DoubtDetector",
    "MissingSheet",
    "DoubtReport",
    "DoubtRFIGenerator",
]


def run_doubt_engine(
    project_id: str,
    page_index: list,
    routing_manifest: dict,
    extraction_results: list,
    scope_register: dict,
    output_dir,
) -> dict:
    """
    Run the estimator doubt engine.

    Args:
        project_id: Project identifier
        page_index: Indexed pages
        routing_manifest: Page routing manifest
        extraction_results: Extraction results
        scope_register: Scope register
        output_dir: Output directory

    Returns:
        Dict with doubt engine results
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)

    # 1. Detect missing sheets
    logger.info("Detecting missing critical sheets...")
    detector = DoubtDetector()
    doubt_report = detector.analyze(
        page_index,
        routing_manifest,
        extraction_results,
        scope_register,
    )
    logger.info(f"Found {len(doubt_report.missing_sheets)} missing sheet types")

    # 2. Generate high-priority RFIs
    logger.info("Generating doubt-based RFIs...")
    rfi_generator = DoubtRFIGenerator()
    rfis = rfi_generator.generate(doubt_report)
    logger.info(f"Generated {len(rfis)} doubt-based RFIs")

    # 3. Export results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Doubt report JSON
    with open(output_dir / "doubt_report.json", "w") as f:
        json.dump(doubt_report.to_dict(), f, indent=2)

    # Doubt report MD
    _export_doubt_report(
        output_dir / "doubt_report.md",
        project_id,
        doubt_report,
        rfis,
    )

    # Doubt RFIs JSON
    with open(output_dir / "doubt_rfis.json", "w") as f:
        json.dump([r.to_dict() for r in rfis], f, indent=2)

    return {
        "missing_sheets": len(doubt_report.missing_sheets),
        "critical_missing": doubt_report.critical_count,
        "important_missing": doubt_report.important_count,
        "optional_missing": doubt_report.optional_count,
        "completeness_score": doubt_report.completeness_score,
        "completeness_grade": doubt_report.completeness_grade,
        "doubt_rfis": len(rfis),
        "high_priority_rfis": len([r for r in rfis if r.priority == "high"]),
    }


def _export_doubt_report(
    output_path,
    project_id: str,
    doubt_report,
    rfis: list,
) -> None:
    """Export doubt report as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Estimator Doubt Report: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Drawing Set Completeness**: {doubt_report.completeness_score:.0f}/100 ")
        f.write(f"(Grade: {doubt_report.completeness_grade})\n")
        f.write(f"- **Critical Missing**: {doubt_report.critical_count}\n")
        f.write(f"- **Important Missing**: {doubt_report.important_count}\n")
        f.write(f"- **Optional Missing**: {doubt_report.optional_count}\n")
        f.write(f"- **Doubt RFIs Generated**: {len(rfis)}\n\n")

        # Present Sheets
        f.write("## Present Sheets\n\n")
        if doubt_report.present_types:
            f.write("| Sheet Type | Count | Status |\n")
            f.write("|------------|-------|--------|\n")
            for sheet_type, count in sorted(doubt_report.present_types.items()):
                f.write(f"| {sheet_type.replace('_', ' ').title()} | {count} | ✅ Found |\n")
        else:
            f.write("No sheets found.\n")
        f.write("\n")

        # Missing Sheets
        f.write("## Missing Sheets\n\n")
        if doubt_report.missing_sheets:
            # Group by severity
            critical = [m for m in doubt_report.missing_sheets if m.severity == "critical"]
            important = [m for m in doubt_report.missing_sheets if m.severity == "important"]
            optional = [m for m in doubt_report.missing_sheets if m.severity == "optional"]

            if critical:
                f.write("### 🔴 Critical (Must Have)\n\n")
                f.write("| Sheet Type | Why Needed | Impact |\n")
                f.write("|------------|------------|--------|\n")
                for missing in critical:
                    f.write(f"| {missing.sheet_type.replace('_', ' ').title()} | ")
                    f.write(f"{missing.why_needed} | {missing.impact} |\n")
                f.write("\n")

            if important:
                f.write("### 🟠 Important (Should Have)\n\n")
                f.write("| Sheet Type | Why Needed | Impact |\n")
                f.write("|------------|------------|--------|\n")
                for missing in important:
                    f.write(f"| {missing.sheet_type.replace('_', ' ').title()} | ")
                    f.write(f"{missing.why_needed} | {missing.impact} |\n")
                f.write("\n")

            if optional:
                f.write("### 🟡 Optional (Nice to Have)\n\n")
                f.write("| Sheet Type | Why Needed |\n")
                f.write("|------------|------------|\n")
                for missing in optional:
                    f.write(f"| {missing.sheet_type.replace('_', ' ').title()} | ")
                    f.write(f"{missing.why_needed} |\n")
                f.write("\n")
        else:
            f.write("All critical sheets are present. ✅\n\n")

        # Generated RFIs
        if rfis:
            f.write("## Doubt-Based RFIs\n\n")
            f.write("The following RFIs are generated due to missing drawing information:\n\n")

            for rfi in rfis:
                priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.priority, "⚪")
                f.write(f"### {priority_emoji} {rfi.rfi_id}: {rfi.sheet_type}\n\n")
                f.write(f"**Question**: {rfi.question}\n\n")
                f.write(f"**Why Needed**: {rfi.why_needed}\n\n")
                f.write(f"**Impacted Packages**: {', '.join(rfi.impacted_packages)}\n\n")
                f.write("---\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        if doubt_report.recommendations:
            for i, rec in enumerate(doubt_report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("Drawing set appears complete for estimation.\n")
