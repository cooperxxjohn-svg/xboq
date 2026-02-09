"""
RFI Engine Module for XBOQ.

Generates estimator-grade RFIs for Indian construction projects
based on drawing set analysis.

Components:
- SignalCollector: Collects all input signals
- ConflictDetector: Detects data conflicts
- ReferenceDetector: Finds missing detail references
- RFIGenerator: Generates RFIs from all sources
"""

from dataclasses import asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json
import csv
from datetime import datetime

from .signals import (
    SignalCollector,
    SignalCollection,
    collect_signals,
)
from .conflicts import (
    ConflictDetector,
    ConflictReport,
    Conflict,
    ConflictType,
    detect_conflicts,
)
from .references import (
    ReferenceDetector,
    ReferenceReport,
    MissingReference,
    analyze_references,
)
from .generator import (
    RFIGenerator,
    RFIReport,
    RFI,
    IssueType,
    Priority,
    generate_rfis,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Signals
    "SignalCollector",
    "SignalCollection",
    "collect_signals",
    # Conflicts
    "ConflictDetector",
    "ConflictReport",
    "Conflict",
    "ConflictType",
    "detect_conflicts",
    # References
    "ReferenceDetector",
    "ReferenceReport",
    "MissingReference",
    "analyze_references",
    # Generator
    "RFIGenerator",
    "RFIReport",
    "RFI",
    "IssueType",
    "Priority",
    "generate_rfis",
    # Main
    "run_rfi_engine",
]


def run_rfi_engine(
    project_id: str,
    extraction_results: List[Dict],
    project_graph: Dict,
    page_index: List[Dict],
    scope_register: Dict,
    completeness_report: Dict,
    coverage_report: Dict,
    missing_inputs: List[Dict],
    triangulation_report: Dict,
    override_report: Dict,
    evidence_data: List[Dict],
    schedules: List[Dict],
    notes: List[Dict],
    legends: List[Dict],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run complete RFI engine and generate all outputs.

    Args:
        project_id: Project identifier
        extraction_results: Page extraction results
        project_graph: Joined project graph
        page_index: Page index data
        scope_register: Scope register data
        completeness_report: Completeness scoring report
        coverage_report: Coverage analysis report
        missing_inputs: Missing inputs list
        triangulation_report: Triangulation results
        override_report: Override results
        evidence_data: Extracted evidence
        schedules: Extracted schedules
        notes: Extracted notes
        legends: Extracted legends
        output_dir: Output directory

    Returns:
        Dict with RFI summary and file paths
    """
    rfi_dir = output_dir / "rfi"
    rfi_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all signals
    logger.info("Collecting signals from all sources...")
    signals = collect_signals(
        project_id=project_id,
        scope_register=scope_register,
        completeness_report=completeness_report,
        coverage_report=coverage_report,
        missing_inputs=missing_inputs,
        triangulation_report=triangulation_report,
        override_report=override_report,
        evidence_data=evidence_data,
        extraction_results=extraction_results,
        project_graph=project_graph,
        page_index=page_index,
    )
    logger.info(f"Collected signals: {len(signals.scope_signals)} scope, {len(signals.coverage_signals)} coverage, {len(signals.triangulation_signals)} triangulation")

    # 2. Detect conflicts
    logger.info("Detecting conflicts...")
    conflict_report = detect_conflicts(
        project_id=project_id,
        extraction_results=extraction_results,
        schedules=schedules,
        notes=notes,
        legends=legends,
        override_report=override_report,
        triangulation_report=triangulation_report,
    )
    logger.info(f"Detected {len(conflict_report.conflicts)} conflicts")

    # 3. Analyze references
    logger.info("Analyzing cross-references...")
    reference_report = analyze_references(
        project_id=project_id,
        extraction_results=extraction_results,
        page_index=page_index,
    )
    logger.info(f"Found {len(reference_report.missing_references)} missing references")

    # 4. Generate RFIs
    logger.info("Generating RFIs...")
    rfi_report = generate_rfis(
        project_id=project_id,
        signals=signals,
        conflict_report=conflict_report,
        reference_report=reference_report,
    )
    logger.info(f"Generated {len(rfi_report.rfis)} RFIs")

    # 5. Save outputs
    _save_rfi_csv(rfi_report, rfi_dir / "rfi_log.csv")
    _save_rfi_markdown(rfi_report, rfi_dir / "rfi_log.md")
    _save_rfi_summary(rfi_report, conflict_report, reference_report, rfi_dir / "rfi_summary.json")

    logger.info(f"Saved RFI log to: {rfi_dir / 'rfi_log.csv'}")
    logger.info(f"Saved RFI markdown to: {rfi_dir / 'rfi_log.md'}")
    logger.info(f"Saved RFI summary to: {rfi_dir / 'rfi_summary.json'}")

    # Return summary
    return {
        "total_rfis": len(rfi_report.rfis),
        "by_priority": rfi_report.summary.get("by_priority", {}),
        "by_package": rfi_report.summary.get("by_package", {}),
        "high_priority": rfi_report.summary.get("high_priority_count", 0),
        "conflicts": len(conflict_report.conflicts),
        "missing_references": len(reference_report.missing_references),
        "top_rfis": [
            {
                "id": rfi.rfi_id,
                "priority": rfi.priority.value,
                "package": rfi.package,
                "question": rfi.question[:100],
            }
            for rfi in rfi_report.rfis[:10]
        ],
    }


def _save_rfi_csv(rfi_report: RFIReport, filepath: Path) -> None:
    """Save RFI log as CSV."""
    fieldnames = [
        "rfi_id",
        "priority",
        "package",
        "issue_type",
        "question",
        "why_needed",
        "impacted_boq_items",
        "evidence_pages",
        "evidence_snippets",
        "missing_info",
        "suggested_resolution",
        "confidence",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rfi in rfi_report.rfis:
            writer.writerow({
                "rfi_id": rfi.rfi_id,
                "priority": rfi.priority.value,
                "package": rfi.package,
                "issue_type": rfi.issue_type.value,
                "question": rfi.question,
                "why_needed": rfi.why_needed,
                "impacted_boq_items": "; ".join(rfi.impacted_boq_items),
                "evidence_pages": "; ".join(rfi.evidence_pages),
                "evidence_snippets": "; ".join(rfi.evidence_snippets[:3]),  # Limit snippets
                "missing_info": rfi.missing_info,
                "suggested_resolution": rfi.suggested_resolution,
                "confidence": rfi.confidence,
            })


def _save_rfi_markdown(rfi_report: RFIReport, filepath: Path) -> None:
    """Save RFI log as markdown."""
    lines = [
        f"# RFI Log: {rfi_report.project_id}",
        "",
        f"**Generated:** {rfi_report.generated}",
        f"**Total RFIs:** {len(rfi_report.rfis)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Priority | Count |",
        "|----------|-------|",
    ]

    for priority in ["high", "medium", "low"]:
        count = rfi_report.summary.get("by_priority", {}).get(priority, 0)
        emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(priority, "⚪")
        lines.append(f"| {emoji} {priority.title()} | {count} |")

    lines.extend([
        "",
        "---",
        "",
    ])

    # Group by package
    by_package = {}
    for rfi in rfi_report.rfis:
        pkg = rfi.package
        if pkg not in by_package:
            by_package[pkg] = []
        by_package[pkg].append(rfi)

    # Sort packages by count (descending)
    sorted_packages = sorted(by_package.items(), key=lambda x: -len(x[1]))

    for package, rfis in sorted_packages:
        lines.append(f"## {package.replace('_', ' ').title()} ({len(rfis)} RFIs)")
        lines.append("")

        # Sort RFIs within package by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_rfis = sorted(rfis, key=lambda r: priority_order.get(r.priority.value, 2))

        for rfi in sorted_rfis:
            priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.priority.value, "⚪")

            lines.append(f"### {priority_emoji} [{rfi.rfi_id}] {rfi.issue_type.value.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"**Question:** {rfi.question}")
            lines.append("")
            lines.append(f"**Why Needed:** {rfi.why_needed}")
            lines.append("")

            if rfi.evidence_pages and rfi.evidence_pages != ["N/A - no evidence"]:
                lines.append(f"**Evidence Pages:** {', '.join(rfi.evidence_pages)}")
                lines.append("")

            if rfi.evidence_snippets:
                lines.append("**Evidence:**")
                for snippet in rfi.evidence_snippets[:3]:
                    lines.append(f"- {snippet}")
                lines.append("")

            lines.append(f"**Impacted Items:** {', '.join(rfi.impacted_boq_items)}")
            lines.append("")

            if rfi.suggested_resolution:
                lines.append(f"**Suggested Resolution:** {rfi.suggested_resolution}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Top 10 RFIs section at the end
    lines.extend([
        "## Top 10 Priority RFIs",
        "",
        "| # | ID | Priority | Package | Question |",
        "|---|-----|----------|---------|----------|",
    ])

    for i, rfi in enumerate(rfi_report.rfis[:10], 1):
        priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.priority.value, "⚪")
        question_short = rfi.question[:60] + "..." if len(rfi.question) > 60 else rfi.question
        lines.append(f"| {i} | {rfi.rfi_id} | {priority_emoji} {rfi.priority.value} | {rfi.package} | {question_short} |")

    lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_rfi_summary(
    rfi_report: RFIReport,
    conflict_report: ConflictReport,
    reference_report: ReferenceReport,
    filepath: Path,
) -> None:
    """Save RFI summary as JSON."""
    summary = {
        "project_id": rfi_report.project_id,
        "generated": rfi_report.generated,
        "total_rfis": len(rfi_report.rfis),
        "summary": rfi_report.summary,
        "conflicts_detected": len(conflict_report.conflicts),
        "conflict_summary": conflict_report.summary,
        "missing_references": len(reference_report.missing_references),
        "reference_summary": reference_report.summary,
        "rfis": [
            {
                "rfi_id": rfi.rfi_id,
                "priority": rfi.priority.value,
                "package": rfi.package,
                "issue_type": rfi.issue_type.value,
                "question": rfi.question,
                "why_needed": rfi.why_needed,
                "impacted_boq_items": rfi.impacted_boq_items,
                "evidence_pages": rfi.evidence_pages,
                "evidence_snippets": rfi.evidence_snippets,
                "missing_info": rfi.missing_info,
                "suggested_resolution": rfi.suggested_resolution,
                "confidence": rfi.confidence,
            }
            for rfi in rfi_report.rfis
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
