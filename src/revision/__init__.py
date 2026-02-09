"""
Revision Intelligence Engine - Track drawing revisions and their impacts.

This module provides:
- Revision table detection on sheets
- Revision history tracking per sheet
- Change detection between sheet versions
- Impacted scope/BOQ identification
- Delta takeoff for changed sheets only

Key India-specific patterns:
- IS 11669 revision table formats
- Common consultant revision notation (R0, Rev A, etc.)
- Description patterns (ASI, ASI reply, Client comment, etc.)
"""

from .detector import RevisionDetector, RevisionEntry, RevisionTable
from .tracker import RevisionTracker, SheetRevision, RevisionHistory
from .comparator import SheetComparator, ChangeType, SheetDiff, ChangeReport
from .impact import ImpactAnalyzer, ImpactReport, ImpactedItem

__all__ = [
    # Detection
    "RevisionDetector",
    "RevisionEntry",
    "RevisionTable",
    # Tracking
    "RevisionTracker",
    "SheetRevision",
    "RevisionHistory",
    # Comparison
    "SheetComparator",
    "ChangeType",
    "SheetDiff",
    "ChangeReport",
    # Impact
    "ImpactAnalyzer",
    "ImpactReport",
    "ImpactedItem",
]


def run_revision_engine(
    project_id: str,
    current_pages: list,
    previous_pages: list,
    extraction_results: list,
    previous_extraction_results: list,
    scope_register: dict,
    boq_entries: list,
    output_dir,
) -> dict:
    """
    Run the complete revision intelligence pipeline.

    Args:
        project_id: Project identifier
        current_pages: Current indexed pages
        previous_pages: Previous version indexed pages (empty if first run)
        extraction_results: Current extraction results
        previous_extraction_results: Previous extraction results
        scope_register: Current scope register
        boq_entries: Current BOQ entries
        output_dir: Output directory

    Returns:
        Dict with revision analysis results
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)

    # 1. Detect revision tables on current sheets
    logger.info("Detecting revision tables...")
    detector = RevisionDetector()
    revision_tables = detector.detect_all(extraction_results)

    # 2. Build revision history
    logger.info("Building revision history...")
    tracker = RevisionTracker()
    revision_history = tracker.build_history(
        current_pages,
        revision_tables,
    )

    # 3. Compare with previous version (if available)
    change_report = None
    sheets_to_reprocess = []
    if previous_pages:
        logger.info("Comparing with previous version...")
        comparator = SheetComparator()
        change_report = comparator.compare_sets(
            current_pages,
            previous_pages,
            extraction_results,
            previous_extraction_results,
        )
        sheets_to_reprocess = change_report.changed_sheet_ids
        logger.info(f"Changed sheets: {len(sheets_to_reprocess)}")

    # 4. Analyze impact on scope and BOQ
    logger.info("Analyzing revision impact...")
    impact_analyzer = ImpactAnalyzer()
    impact_report = impact_analyzer.analyze(
        revision_history,
        change_report,
        scope_register,
        boq_entries,
    )

    # 5. Export results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Revision history JSON
    history_data = revision_history.to_dict()
    with open(output_dir / "revision_history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    # Revision report MD
    _export_revision_report(
        output_dir / "revision_report.md",
        project_id,
        revision_history,
        change_report,
        impact_report,
    )

    # Impact summary JSON
    impact_data = impact_report.to_dict()
    with open(output_dir / "revision_impact.json", "w") as f:
        json.dump(impact_data, f, indent=2)

    return {
        "sheets_with_revisions": len(revision_tables),
        "total_revisions": sum(len(t.entries) for t in revision_tables),
        "latest_revision": revision_history.latest_revision,
        "sheets_changed": len(sheets_to_reprocess) if change_report else 0,
        "sheets_to_reprocess": sheets_to_reprocess,
        "impacted_scope_items": len(impact_report.impacted_scope_items),
        "impacted_boq_items": len(impact_report.impacted_boq_items),
        "high_impact_changes": impact_report.high_impact_count,
    }


def _export_revision_report(
    output_path,
    project_id: str,
    revision_history,
    change_report,
    impact_report,
) -> None:
    """Export revision report as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Revision Report: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Sheets with Revisions**: {len(revision_history.sheets)}\n")
        f.write(f"- **Latest Revision**: {revision_history.latest_revision or 'R0'}\n")
        f.write(f"- **Latest Date**: {revision_history.latest_date or 'N/A'}\n")

        if change_report:
            f.write(f"- **Sheets Changed**: {len(change_report.changed_sheet_ids)}\n")
            f.write(f"- **Sheets Added**: {len(change_report.added_sheet_ids)}\n")
            f.write(f"- **Sheets Removed**: {len(change_report.removed_sheet_ids)}\n")
        f.write("\n")

        # Revision Timeline
        f.write("## Revision Timeline\n\n")
        if revision_history.timeline:
            f.write("| Revision | Date | Sheets Affected | Description |\n")
            f.write("|----------|------|-----------------|-------------|\n")
            for rev in revision_history.timeline:
                f.write(f"| {rev.revision} | {rev.date or 'N/A'} | {rev.sheet_count} | {rev.description[:50]}... |\n")
        else:
            f.write("No revision history detected.\n")
        f.write("\n")

        # Sheet Revisions
        f.write("## Sheet Revisions\n\n")
        for sheet_id, sheet_rev in sorted(revision_history.sheets.items()):
            f.write(f"### {sheet_id}\n\n")
            if sheet_rev.revisions:
                f.write("| Rev | Date | Description |\n")
                f.write("|-----|------|-------------|\n")
                for rev in sheet_rev.revisions[-5:]:  # Last 5 revisions
                    f.write(f"| {rev.revision} | {rev.date or 'N/A'} | {rev.description} |\n")
            else:
                f.write("No revision table detected.\n")
            f.write("\n")

        # Change Report (if comparing versions)
        if change_report and change_report.diffs:
            f.write("## Changes from Previous Version\n\n")

            # Added sheets
            if change_report.added_sheet_ids:
                f.write("### New Sheets\n\n")
                for sheet_id in change_report.added_sheet_ids:
                    f.write(f"- {sheet_id}\n")
                f.write("\n")

            # Removed sheets
            if change_report.removed_sheet_ids:
                f.write("### Removed Sheets\n\n")
                for sheet_id in change_report.removed_sheet_ids:
                    f.write(f"- ⚠️ {sheet_id}\n")
                f.write("\n")

            # Changed sheets
            if change_report.diffs:
                f.write("### Modified Sheets\n\n")
                f.write("| Sheet | Change Type | Summary |\n")
                f.write("|-------|-------------|----------|\n")
                for diff in change_report.diffs:
                    f.write(f"| {diff.sheet_id} | {diff.change_type.value} | {diff.summary} |\n")
                f.write("\n")

        # Impact Analysis
        f.write("## Impact Analysis\n\n")

        if impact_report.impacted_scope_items:
            f.write("### Impacted Scope Items\n\n")
            f.write("| Package | Item | Impact | Reason |\n")
            f.write("|---------|------|--------|--------|\n")
            for item in impact_report.impacted_scope_items[:20]:
                impact_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(item.impact_level, "⚪")
                f.write(f"| {item.package} | {item.item_id} | {impact_emoji} {item.impact_level} | {item.reason} |\n")
            if len(impact_report.impacted_scope_items) > 20:
                f.write(f"\n... and {len(impact_report.impacted_scope_items) - 20} more impacted items\n")
            f.write("\n")

        if impact_report.impacted_boq_items:
            f.write("### Impacted BOQ Items\n\n")
            f.write("| Item | Old Qty | New Qty | Delta | Impact |\n")
            f.write("|------|---------|---------|-------|--------|\n")
            for item in impact_report.impacted_boq_items[:20]:
                delta = item.new_quantity - item.old_quantity if item.old_quantity else item.new_quantity
                delta_str = f"+{delta}" if delta > 0 else str(delta)
                f.write(f"| {item.item_id} | {item.old_quantity or 'N/A'} | {item.new_quantity} | {delta_str} | {item.impact_level} |\n")
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        if impact_report.recommendations:
            for i, rec in enumerate(impact_report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("No specific recommendations at this time.\n")
