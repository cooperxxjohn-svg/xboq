"""
Estimator Bid Gate - GO/REVIEW/NO-GO Assessment

Generates bid_gate.md with:
1. Measurement coverage analysis
2. Missing scope assessment
3. Risk evaluation
4. Clear recommendation (GO/REVIEW/NO-GO)
5. Action items for each status

This is the estimator's decision support output.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BidRecommendation(Enum):
    """Bid recommendation status."""
    GO = "GO"
    REVIEW = "REVIEW"
    NO_GO = "NO-GO"


@dataclass
class BidGateMetrics:
    """Metrics for bid gate assessment."""
    # Quantity metrics
    total_items: int = 0
    measured_items: int = 0
    inferred_items: int = 0
    needs_review_count: int = 0

    # Percentages
    measured_pct: float = 0.0
    inferred_pct: float = 0.0
    review_pct: float = 0.0

    # Missing scope
    missing_scope_count: int = 0
    high_priority_missing: int = 0
    medium_priority_missing: int = 0

    # Confidence
    avg_confidence: float = 0.0
    low_confidence_count: int = 0

    # RFIs
    rfi_count: int = 0
    critical_rfi_count: int = 0


@dataclass
class BidGateResult:
    """Result of bid gate assessment."""
    recommendation: BidRecommendation
    score: int  # 0-100
    metrics: BidGateMetrics
    reasons: List[str]
    action_items: List[str]
    risks: List[str]
    go_criteria_met: Dict[str, bool] = field(default_factory=dict)


# =============================================================================
# BID GATE CRITERIA
# =============================================================================

# GO criteria (all must be met for GO recommendation)
GO_CRITERIA = {
    "measured_coverage": ("Measured coverage ≥ 60%", lambda m: m.measured_pct >= 60),
    "no_high_priority_missing": ("No high-priority missing scope", lambda m: m.high_priority_missing == 0),
    "low_review_items": ("Items needing review ≤ 20%", lambda m: m.review_pct <= 20),
    "avg_confidence": ("Average confidence ≥ 0.7", lambda m: m.avg_confidence >= 0.7),
    "critical_rfis": ("Critical RFIs ≤ 3", lambda m: m.critical_rfi_count <= 3),
}

# REVIEW criteria (if not GO, check these)
REVIEW_CRITERIA = {
    "measured_coverage": ("Measured coverage ≥ 30%", lambda m: m.measured_pct >= 30),
    "high_priority_missing": ("High-priority missing ≤ 5", lambda m: m.high_priority_missing <= 5),
    "avg_confidence": ("Average confidence ≥ 0.5", lambda m: m.avg_confidence >= 0.5),
}


def assess_bid_gate(
    estimator_view: List[Dict],
    missing_scope: List[Dict],
    rfis: List[Dict] = None,
) -> BidGateResult:
    """
    Assess bid gate and generate recommendation.

    Args:
        estimator_view: Merged BOQ items
        missing_scope: Missing scope items
        rfis: RFIs generated

    Returns:
        BidGateResult with recommendation and metrics
    """
    rfis = rfis or []

    # Calculate metrics
    metrics = _calculate_metrics(estimator_view, missing_scope, rfis)

    # Check GO criteria
    go_criteria_met = {}
    go_reasons = []
    for name, (desc, check) in GO_CRITERIA.items():
        passed = check(metrics)
        go_criteria_met[name] = passed
        if not passed:
            go_reasons.append(f"❌ {desc}")
        else:
            go_reasons.append(f"✅ {desc}")

    # Determine recommendation
    all_go = all(go_criteria_met.values())

    if all_go:
        recommendation = BidRecommendation.GO
        score = 80 + int(metrics.measured_pct * 0.2)  # 80-100
    else:
        # Check REVIEW criteria
        review_passed = all(
            check(metrics) for _, (_, check) in REVIEW_CRITERIA.items()
        )

        if review_passed:
            recommendation = BidRecommendation.REVIEW
            score = 40 + int(metrics.measured_pct * 0.4)  # 40-80
        else:
            recommendation = BidRecommendation.NO_GO
            score = int(metrics.measured_pct * 0.4)  # 0-40

    # Generate action items
    action_items = _generate_action_items(recommendation, metrics, go_criteria_met)

    # Identify risks
    risks = _identify_risks(metrics)

    return BidGateResult(
        recommendation=recommendation,
        score=score,
        metrics=metrics,
        reasons=go_reasons,
        action_items=action_items,
        risks=risks,
        go_criteria_met=go_criteria_met,
    )


def _calculate_metrics(
    estimator_view: List[Dict],
    missing_scope: List[Dict],
    rfis: List[Dict],
) -> BidGateMetrics:
    """Calculate bid gate metrics."""
    total = len(estimator_view)
    measured = sum(1 for x in estimator_view if x.get("source") == "measured")
    inferred = sum(1 for x in estimator_view if x.get("source") == "inferred")
    needs_review = sum(1 for x in estimator_view if x.get("needs_review") == "YES")

    # Confidence
    confidences = []
    for x in estimator_view:
        try:
            conf = float(x.get("confidence", 0.5))
            confidences.append(conf)
        except (ValueError, TypeError):
            confidences.append(0.5)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
    low_conf = sum(1 for c in confidences if c < 0.6)

    # Missing scope
    high_missing = sum(1 for x in missing_scope if x.get("priority") == "HIGH")
    med_missing = sum(1 for x in missing_scope if x.get("priority") == "MEDIUM")

    # RFIs
    critical_rfis = sum(
        1 for r in rfis
        if r.get("priority") == "HIGH" or r.get("severity") == "HIGH"
    )

    return BidGateMetrics(
        total_items=total,
        measured_items=measured,
        inferred_items=inferred,
        needs_review_count=needs_review,
        measured_pct=(measured / total * 100) if total > 0 else 0,
        inferred_pct=(inferred / total * 100) if total > 0 else 0,
        review_pct=(needs_review / total * 100) if total > 0 else 0,
        missing_scope_count=len(missing_scope),
        high_priority_missing=high_missing,
        medium_priority_missing=med_missing,
        avg_confidence=avg_conf,
        low_confidence_count=low_conf,
        rfi_count=len(rfis),
        critical_rfi_count=critical_rfis,
    )


def _generate_action_items(
    recommendation: BidRecommendation,
    metrics: BidGateMetrics,
    go_criteria: Dict[str, bool],
) -> List[str]:
    """Generate action items based on recommendation."""
    actions = []

    if recommendation == BidRecommendation.GO:
        actions.append("✅ Proceed with bid submission")
        actions.append("Review final quantities with project team")
        actions.append("Complete rate entry in Excel workbook")
        if metrics.needs_review_count > 0:
            actions.append(f"Optional: Review {metrics.needs_review_count} flagged items for accuracy")

    elif recommendation == BidRecommendation.REVIEW:
        actions.append("⚠️ Manual review required before bid submission")

        if not go_criteria.get("measured_coverage", True):
            actions.append(f"→ Improve measurement coverage (currently {metrics.measured_pct:.0f}%, need 60%)")
            actions.append("  Consider: Re-process with manual scale, request clearer drawings")

        if not go_criteria.get("no_high_priority_missing", True):
            actions.append(f"→ Address {metrics.high_priority_missing} high-priority missing scope items")
            actions.append("  Consider: Send RFIs, make conservative assumptions")

        if not go_criteria.get("low_review_items", True):
            actions.append(f"→ Review {metrics.needs_review_count} items flagged for review")

        if not go_criteria.get("avg_confidence", True):
            actions.append(f"→ Improve confidence (currently {metrics.avg_confidence:.2f}, need 0.7)")

        actions.append("Obtain senior estimator approval before submission")

    else:  # NO_GO
        actions.append("❌ Do not submit bid without significant improvements")
        actions.append("Request additional/clearer drawings from client")

        if metrics.measured_pct < 30:
            actions.append(f"→ Measurement coverage critically low ({metrics.measured_pct:.0f}%)")

        if metrics.high_priority_missing > 5:
            actions.append(f"→ {metrics.high_priority_missing} high-priority scope gaps")

        actions.append("Consider: Decline to bid, request time extension, qualify bid heavily")

    return actions


def _identify_risks(metrics: BidGateMetrics) -> List[str]:
    """Identify risks based on metrics."""
    risks = []

    if metrics.inferred_pct > 50:
        risks.append(f"HIGH: {metrics.inferred_pct:.0f}% of quantities are inferred, not measured")

    if metrics.high_priority_missing > 0:
        risks.append(f"HIGH: {metrics.high_priority_missing} critical scope items may be missing")

    if metrics.avg_confidence < 0.6:
        risks.append(f"MEDIUM: Low average confidence ({metrics.avg_confidence:.2f})")

    if metrics.low_confidence_count > 10:
        risks.append(f"MEDIUM: {metrics.low_confidence_count} items have low confidence")

    if metrics.critical_rfi_count > 0:
        risks.append(f"MEDIUM: {metrics.critical_rfi_count} critical RFIs pending")

    if metrics.missing_scope_count > 100:
        risks.append(f"LOW: Large number of missing scope items ({metrics.missing_scope_count})")

    return risks


def generate_bid_gate_report(
    output_dir: Path,
    result: BidGateResult,
    project_id: str = "",
) -> Path:
    """
    Generate bid_gate.md report.

    Args:
        output_dir: Output directory
        result: BidGateResult from assessment
        project_id: Project identifier

    Returns:
        Path to generated report
    """
    estimator_dir = output_dir / "estimator"
    estimator_dir.mkdir(parents=True, exist_ok=True)
    report_path = estimator_dir / "bid_gate.md"

    m = result.metrics

    # Recommendation emoji
    rec_emoji = {
        BidRecommendation.GO: "✅",
        BidRecommendation.REVIEW: "⚠️",
        BidRecommendation.NO_GO: "❌",
    }

    content = f"""# Bid Gate Assessment

**Project**: {project_id or 'Unknown'}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## {rec_emoji[result.recommendation]} RECOMMENDATION: {result.recommendation.value}

**Score**: {result.score}/100

---

## Measurement Analysis

| Metric | Value | Status |
|--------|-------|--------|
| Total BOQ Items | {m.total_items} | - |
| Measured (from geometry) | {m.measured_items} | {m.measured_pct:.1f}% |
| Inferred (assumptions) | {m.inferred_items} | {m.inferred_pct:.1f}% |
| Needs Review | {m.needs_review_count} | {m.review_pct:.1f}% |
| Average Confidence | {m.avg_confidence:.2f} | {'✅' if m.avg_confidence >= 0.7 else '⚠️'} |

## Missing Scope Analysis

| Metric | Value |
|--------|-------|
| Total Missing Scope Items | {m.missing_scope_count} |
| High Priority | {m.high_priority_missing} |
| Medium Priority | {m.medium_priority_missing} |

## RFI Status

| Metric | Value |
|--------|-------|
| Total RFIs | {m.rfi_count} |
| Critical RFIs | {m.critical_rfi_count} |

---

## GO Criteria Assessment

"""

    for reason in result.reasons:
        content += f"- {reason}\n"

    content += """
---

## Action Items

"""

    for i, action in enumerate(result.action_items, 1):
        if action.startswith("→") or action.startswith("  "):
            content += f"   {action}\n"
        else:
            content += f"{i}. {action}\n"

    if result.risks:
        content += """
---

## Risk Assessment

"""
        for risk in result.risks:
            content += f"- **{risk}**\n"

    content += f"""
---

## Estimator Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Prepared By | _____________ | ________ | _________ |
| Reviewed By | _____________ | ________ | _________ |
| Approved By | _____________ | ________ | _________ |

---

*Generated by XBOQ Estimator Assistant*
"""

    with open(report_path, "w") as f:
        f.write(content)

    logger.info(f"Generated bid gate report at {report_path}")

    return report_path


def quick_bid_gate(
    output_dir: Path,
    project_id: str = "",
) -> BidGateResult:
    """
    Quick bid gate assessment from existing files.

    Args:
        output_dir: Output directory with reconciliation results
        project_id: Project identifier

    Returns:
        BidGateResult
    """
    import json
    import csv

    estimator_dir = output_dir / "estimator"

    # Load estimator view
    estimator_view = []
    view_path = estimator_dir / "boq_estimator_view.csv"
    if view_path.exists():
        with open(view_path, newline='') as f:
            reader = csv.DictReader(f)
            estimator_view = list(reader)

    # Load missing scope
    missing_scope = []
    missing_path = estimator_dir / "missing_scope.csv"
    if missing_path.exists():
        with open(missing_path, newline='') as f:
            reader = csv.DictReader(f)
            missing_scope = list(reader)

    # Load RFIs
    rfis = []
    rfi_path = output_dir / "rfi" / "rfi_log.json"
    if rfi_path.exists():
        with open(rfi_path) as f:
            rfis = json.load(f).get("rfis", [])

    # Run assessment
    result = assess_bid_gate(estimator_view, missing_scope, rfis)

    # Generate report
    generate_bid_gate_report(output_dir, result, project_id)

    return result
