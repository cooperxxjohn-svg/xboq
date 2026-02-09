"""
Accuracy Enhancement Module for XBOQ.

Provides estimator-grade confidence through:
- Triangulation: Multi-method quantity verification
- Cross-Sheet Overrides: Schedule/legend-based corrections
- Paranoia Rules: Experienced estimator inference
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json
from datetime import datetime

from .triangulation import (
    TriangulationEngine,
    TriangulationReport,
    TriangulationResult,
    AgreementLevel,
    run_triangulation,
)
from .overrides import (
    CrossSheetOverrideEngine,
    OverrideReport,
    Override,
    OverrideSource,
    OverrideType,
    run_cross_sheet_overrides,
)
from .paranoia import (
    EstimatorParanoiaEngine,
    ParanoiaReport,
    InferredItem,
    InferencePriority,
    run_paranoia_engine,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Triangulation
    "TriangulationEngine",
    "TriangulationReport",
    "TriangulationResult",
    "AgreementLevel",
    "run_triangulation",
    # Overrides
    "CrossSheetOverrideEngine",
    "OverrideReport",
    "Override",
    "OverrideSource",
    "OverrideType",
    "run_cross_sheet_overrides",
    # Paranoia
    "EstimatorParanoiaEngine",
    "ParanoiaReport",
    "InferredItem",
    "InferencePriority",
    "run_paranoia_engine",
    # Combined
    "AccuracyReport",
    "run_accuracy_engine",
]


@dataclass
class AccuracyReport:
    """Combined accuracy enhancement report."""
    project_id: str
    generated: str = ""
    triangulation: Optional[TriangulationReport] = None
    overrides: Optional[OverrideReport] = None
    paranoia: Optional[ParanoiaReport] = None
    combined_summary: Dict[str, Any] = field(default_factory=dict)
    discrepancies_for_review: List[Dict] = field(default_factory=list)
    confidence_grade: str = "F"
    confidence_score: float = 0.0


def run_accuracy_engine(
    project_id: str,
    extraction_results: List[Dict],
    project_graph: Dict,
    boq_entries: List[Dict],
    schedules: List[Dict],
    notes: List[Dict],
    legends: List[Dict],
    scope_register: Dict,
    output_dir: Path,
) -> AccuracyReport:
    """
    Run complete accuracy enhancement engine.

    Args:
        project_id: Project identifier
        extraction_results: Page extraction results
        project_graph: Joined project graph
        boq_entries: Current BOQ entries
        schedules: Extracted schedule tables
        notes: Extracted notes
        legends: Extracted legends
        scope_register: Current scope register
        output_dir: Output directory for reports

    Returns:
        Combined accuracy report
    """
    report = AccuracyReport(
        project_id=project_id,
        generated=datetime.now().isoformat(),
    )

    # 1. Run Triangulation
    logger.info("Running triangulation engine...")
    try:
        report.triangulation = run_triangulation(
            project_id=project_id,
            extraction_results=extraction_results,
            project_graph=project_graph,
            boq_entries=boq_entries,
            schedules=schedules,
        )
        logger.info(f"Triangulation: {len(report.triangulation.results)} quantities verified")
    except Exception as e:
        logger.error(f"Triangulation failed: {e}")
        report.triangulation = TriangulationReport(project_id=project_id)

    # 2. Run Cross-Sheet Overrides
    logger.info("Running cross-sheet override engine...")
    try:
        report.overrides = run_cross_sheet_overrides(
            project_id=project_id,
            extraction_results=extraction_results,
            schedules=schedules,
            notes=notes,
            legends=legends,
            boq_entries=boq_entries,
            scope_register=scope_register,
        )
        logger.info(f"Overrides: {len(report.overrides.overrides)} applied")
    except Exception as e:
        logger.error(f"Override engine failed: {e}")
        report.overrides = OverrideReport(project_id=project_id)

    # 3. Run Paranoia Rules
    logger.info("Running estimator paranoia engine...")
    try:
        report.paranoia = run_paranoia_engine(
            project_id=project_id,
            extraction_results=extraction_results,
            project_graph=project_graph,
            scope_register=scope_register,
            existing_boq=boq_entries,
        )
        logger.info(f"Paranoia: {len(report.paranoia.inferred_items)} items inferred")
    except Exception as e:
        logger.error(f"Paranoia engine failed: {e}")
        report.paranoia = ParanoiaReport(project_id=project_id)

    # 4. Build combined summary
    report.combined_summary = _build_combined_summary(report)
    report.discrepancies_for_review = _collect_discrepancies(report)
    report.confidence_score, report.confidence_grade = _calculate_confidence(report)

    # 5. Save reports
    _save_reports(report, output_dir)

    return report


def _build_combined_summary(report: AccuracyReport) -> Dict[str, Any]:
    """Build combined summary from all engines."""
    summary = {
        "triangulation": {},
        "overrides": {},
        "paranoia": {},
        "overall": {},
    }

    # Triangulation summary
    if report.triangulation:
        summary["triangulation"] = {
            "quantities_verified": len(report.triangulation.results),
            "agreement_score": report.triangulation.overall_agreement,
            "critical_discrepancies": len(report.triangulation.critical_discrepancies),
            "recommendation": report.triangulation.summary.get("recommendation", ""),
        }

    # Override summary
    if report.overrides:
        summary["overrides"] = {
            "total_overrides": len(report.overrides.overrides),
            "by_source": report.overrides.summary.get("by_source", {}),
            "scope_detected": len(report.overrides.scope_detections),
            "high_confidence": report.overrides.summary.get("high_confidence_overrides", 0),
        }

    # Paranoia summary
    if report.paranoia:
        summary["paranoia"] = {
            "items_inferred": len(report.paranoia.inferred_items),
            "by_trigger": report.paranoia.summary.get("by_trigger", {}),
            "provisional_boq_items": len(report.paranoia.provisional_boq),
            "estimated_value": report.paranoia.summary.get("estimated_value_range", {}),
        }

    # Overall
    summary["overall"] = {
        "data_quality_checks": len(report.triangulation.results) if report.triangulation else 0,
        "corrections_applied": len(report.overrides.overrides) if report.overrides else 0,
        "scope_items_inferred": len(report.paranoia.inferred_items) if report.paranoia else 0,
        "review_items": len(report.discrepancies_for_review),
    }

    return summary


def _collect_discrepancies(report: AccuracyReport) -> List[Dict]:
    """Collect all items requiring review."""
    discrepancies = []

    # From triangulation - POOR or DISCREPANCY agreement
    if report.triangulation:
        for result in report.triangulation.results:
            if result.agreement_level in [AgreementLevel.POOR, AgreementLevel.DISCREPANCY]:
                discrepancies.append({
                    "source": "triangulation",
                    "type": "quantity_discrepancy",
                    "item": result.quantity_name,
                    "severity": "HIGH" if result.agreement_level == AgreementLevel.DISCREPANCY else "MEDIUM",
                    "variance": f"{result.variance_pct}%",
                    "methods": [
                        {"name": m.method_name, "value": m.value, "confidence": m.confidence}
                        for m in result.methods
                    ],
                    "action": result.recommended_action,
                })

    # From overrides - scope detections that need confirmation
    if report.overrides:
        for detection in report.overrides.scope_detections:
            discrepancies.append({
                "source": "override",
                "type": "scope_detection",
                "item": detection,
                "severity": "INFO",
                "action": f"Verify scope item '{detection}' detected from notes",
            })

    # From paranoia - critical inferred items
    if report.paranoia:
        critical_inferred = [
            i for i in report.paranoia.inferred_items
            if i.priority == InferencePriority.CRITICAL
        ]
        for item in critical_inferred[:5]:  # Top 5
            discrepancies.append({
                "source": "paranoia",
                "type": "inferred_scope",
                "item": item.description,
                "severity": "HIGH",
                "trigger": f"{item.trigger_type}:{item.trigger_name}",
                "action": f"Confirm implied item: {item.description}",
            })

    return discrepancies


def _calculate_confidence(report: AccuracyReport) -> tuple:
    """Calculate overall confidence score and grade."""
    scores = []
    weights = []

    # Triangulation contributes 40%
    if report.triangulation and report.triangulation.results:
        tri_score = report.triangulation.overall_agreement
        scores.append(tri_score)
        weights.append(40)

    # Override coverage contributes 30%
    if report.overrides:
        override_count = len(report.overrides.overrides)
        # More overrides generally means better data quality
        # Score: 50 base + 5 per override up to 100
        override_score = min(100, 50 + override_count * 5)
        scores.append(override_score)
        weights.append(30)

    # Paranoia completeness contributes 30%
    if report.paranoia:
        # Having inferred items means the system is working
        # But too many might indicate missing data
        inferred_count = len(report.paranoia.inferred_items)
        # Sweet spot: 5-20 inferred items
        if 5 <= inferred_count <= 20:
            paranoia_score = 85
        elif inferred_count < 5:
            paranoia_score = 70  # Might be missing inferences
        else:
            paranoia_score = max(50, 90 - (inferred_count - 20) * 2)
        scores.append(paranoia_score)
        weights.append(30)

    # Calculate weighted average
    if weights:
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        weighted_score = 50  # Default if no data

    # Assign grade
    if weighted_score >= 90:
        grade = "A"
    elif weighted_score >= 80:
        grade = "B"
    elif weighted_score >= 70:
        grade = "C"
    elif weighted_score >= 60:
        grade = "D"
    else:
        grade = "F"

    return round(weighted_score, 1), grade


def _save_reports(report: AccuracyReport, output_dir: Path) -> None:
    """Save accuracy reports to output directory."""
    accuracy_dir = output_dir / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    # Save triangulation report
    if report.triangulation:
        tri_data = {
            "project_id": report.triangulation.project_id,
            "overall_agreement": report.triangulation.overall_agreement,
            "critical_discrepancies": report.triangulation.critical_discrepancies,
            "summary": report.triangulation.summary,
            "results": [
                {
                    "quantity": r.quantity_name,
                    "type": r.quantity_type,
                    "final_value": r.final_value,
                    "unit": r.unit,
                    "agreement_level": r.agreement_level.value,
                    "agreement_score": r.agreement_score,
                    "variance_pct": r.variance_pct,
                    "methods": [
                        {
                            "name": m.method_name,
                            "value": m.value,
                            "confidence": m.confidence,
                            "source": m.source,
                            "notes": m.notes,
                        }
                        for m in r.methods
                    ],
                    "discrepancy_notes": r.discrepancy_notes,
                    "recommended_action": r.recommended_action,
                }
                for r in report.triangulation.results
            ],
        }
        with open(accuracy_dir / "triangulation.json", "w") as f:
            json.dump(tri_data, f, indent=2)
        logger.info(f"Saved triangulation report to: {accuracy_dir / 'triangulation.json'}")

    # Save override report
    if report.overrides:
        override_data = {
            "project_id": report.overrides.project_id,
            "summary": report.overrides.summary,
            "scope_detections": report.overrides.scope_detections,
            "overrides": [
                {
                    "id": o.override_id,
                    "type": o.override_type.value,
                    "source": o.source.value,
                    "target": o.target,
                    "original": o.original_value,
                    "override": o.override_value,
                    "confidence": o.confidence,
                    "reference": o.source_reference,
                    "notes": o.notes,
                }
                for o in report.overrides.overrides
            ],
        }
        with open(accuracy_dir / "overrides.json", "w") as f:
            json.dump(override_data, f, indent=2)
        logger.info(f"Saved override report to: {accuracy_dir / 'overrides.json'}")

    # Save paranoia report
    if report.paranoia:
        paranoia_data = {
            "project_id": report.paranoia.project_id,
            "summary": report.paranoia.summary,
            "scope_additions": report.paranoia.scope_additions,
            "inferred_items": [
                {
                    "id": i.inference_id,
                    "trigger_type": i.trigger_type,
                    "trigger_name": i.trigger_name,
                    "trigger_source": i.trigger_source,
                    "item_code": i.item_code,
                    "description": i.description,
                    "boq_code": i.boq_code,
                    "unit": i.unit,
                    "quantity": i.quantity,
                    "rate_range": list(i.rate_range),
                    "priority": i.priority.value,
                    "confidence": i.confidence,
                    "notes": i.notes,
                }
                for i in report.paranoia.inferred_items
            ],
            "provisional_boq": report.paranoia.provisional_boq,
        }
        with open(accuracy_dir / "paranoia.json", "w") as f:
            json.dump(paranoia_data, f, indent=2)
        logger.info(f"Saved paranoia report to: {accuracy_dir / 'paranoia.json'}")

    # Save combined summary
    combined_data = {
        "project_id": report.project_id,
        "generated": report.generated,
        "confidence_score": report.confidence_score,
        "confidence_grade": report.confidence_grade,
        "summary": report.combined_summary,
        "discrepancies_for_review": report.discrepancies_for_review,
    }
    with open(accuracy_dir / "accuracy_summary.json", "w") as f:
        json.dump(combined_data, f, indent=2)
    logger.info(f"Saved accuracy summary to: {accuracy_dir / 'accuracy_summary.json'}")

    # Save human-readable discrepancies report
    _save_discrepancies_md(report, accuracy_dir / "discrepancies.md")


def _save_discrepancies_md(report: AccuracyReport, filepath: Path) -> None:
    """Save human-readable discrepancies report."""
    lines = [
        f"# Accuracy Discrepancies Report: {report.project_id}",
        "",
        f"**Generated:** {report.generated}",
        f"**Confidence Score:** {report.confidence_score}/100 (Grade: {report.confidence_grade})",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    summary = report.combined_summary
    lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
        f"| Quantities Verified | {summary.get('triangulation', {}).get('quantities_verified', 0)} |",
        f"| Overrides Applied | {summary.get('overrides', {}).get('total_overrides', 0)} |",
        f"| Items Inferred | {summary.get('paranoia', {}).get('items_inferred', 0)} |",
        f"| Review Items | {len(report.discrepancies_for_review)} |",
        "",
    ])

    # Discrepancies requiring review
    if report.discrepancies_for_review:
        lines.extend([
            "---",
            "",
            "## Items Requiring Review",
            "",
        ])

        # Group by severity
        high_items = [d for d in report.discrepancies_for_review if d.get("severity") == "HIGH"]
        medium_items = [d for d in report.discrepancies_for_review if d.get("severity") == "MEDIUM"]
        info_items = [d for d in report.discrepancies_for_review if d.get("severity") == "INFO"]

        if high_items:
            lines.append("### 🔴 High Priority")
            lines.append("")
            for i, item in enumerate(high_items, 1):
                lines.append(f"**{i}. {item.get('item', 'Unknown')}**")
                lines.append(f"- Source: {item.get('source', '')}")
                lines.append(f"- Type: {item.get('type', '')}")
                if item.get("variance"):
                    lines.append(f"- Variance: {item.get('variance')}")
                lines.append(f"- Action: {item.get('action', '')}")
                lines.append("")

        if medium_items:
            lines.append("### 🟠 Medium Priority")
            lines.append("")
            for i, item in enumerate(medium_items, 1):
                lines.append(f"**{i}. {item.get('item', 'Unknown')}**")
                lines.append(f"- Action: {item.get('action', '')}")
                lines.append("")

        if info_items:
            lines.append("### 🟡 Information")
            lines.append("")
            for item in info_items:
                lines.append(f"- {item.get('item', '')}: {item.get('action', '')}")
            lines.append("")

    # Triangulation agreement table
    if report.triangulation and report.triangulation.results:
        lines.extend([
            "---",
            "",
            "## Triangulation Agreement Table",
            "",
            "| Quantity | Final Value | Agreement | Variance | Methods |",
            "|----------|-------------|-----------|----------|---------|",
        ])

        for result in report.triangulation.results:
            methods_str = ", ".join([f"{m.method_name}={m.value}" for m in result.methods])
            agreement_emoji = {
                "EXCELLENT": "✅",
                "GOOD": "🟢",
                "FAIR": "🟡",
                "POOR": "🟠",
                "DISCREPANCY": "🔴",
            }.get(result.agreement_level.value, "⚪")

            lines.append(
                f"| {result.quantity_name} | {result.final_value} {result.unit} | "
                f"{agreement_emoji} {result.agreement_level.value} | {result.variance_pct}% | {methods_str} |"
            )
        lines.append("")

    # Overrides applied
    if report.overrides and report.overrides.overrides:
        lines.extend([
            "---",
            "",
            "## Overrides Applied",
            "",
        ])

        # Group by type
        by_type = {}
        for o in report.overrides.overrides:
            t = o.override_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(o)

        for override_type, overrides in by_type.items():
            lines.append(f"### {override_type.replace('_', ' ').title()}")
            lines.append("")
            for o in overrides[:5]:  # Top 5 per type
                lines.append(f"- **{o.target}**: {o.original_value} → {o.override_value}")
                lines.append(f"  - Source: {o.source_reference}")
            if len(overrides) > 5:
                lines.append(f"- ... and {len(overrides) - 5} more")
            lines.append("")

    # Inferred items
    if report.paranoia and report.paranoia.inferred_items:
        lines.extend([
            "---",
            "",
            "## Inferred Scope Items (Paranoia Rules)",
            "",
        ])

        # Group by trigger
        by_trigger = {}
        for item in report.paranoia.inferred_items:
            trigger = item.trigger_name
            if trigger not in by_trigger:
                by_trigger[trigger] = []
            by_trigger[trigger].append(item)

        for trigger, items in by_trigger.items():
            lines.append(f"### {trigger.replace('_', ' ').title()}")
            lines.append("")
            for item in items[:5]:
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "⚪",
                }.get(item.priority.value, "⚪")
                lines.append(f"- {priority_emoji} {item.description}")
                lines.append(f"  - Code: `{item.boq_code}` | Qty: {item.quantity} {item.unit}")
            if len(items) > 5:
                lines.append(f"- ... and {len(items) - 5} more items")
            lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
