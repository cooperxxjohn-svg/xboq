"""
Bid Gate - Safety gate for bid submission.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
from datetime import datetime
import yaml


class GateStatus(Enum):
    """Bid gate status."""
    PASS = "PASS"
    PASS_WITH_RESERVATIONS = "PASS_WITH_RESERVATIONS"
    FAIL = "FAIL"


@dataclass
class Reservation:
    """A reservation or concern about the bid."""
    code: str
    category: str
    description: str
    impact: str
    recommendation: str
    evidence: str = ""
    severity: str = "medium"  # low / medium / high

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "category": self.category,
            "description": self.description,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "severity": self.severity,
        }


@dataclass
class CheckResult:
    """Result of a single gate check."""
    check_name: str
    status: str  # pass / reservation / fail
    value: Any
    threshold_pass: Any
    threshold_reservation: Any
    weight: float
    score: float  # 0-100 contribution
    message: str

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "status": self.status,
            "value": self.value,
            "threshold_pass": self.threshold_pass,
            "threshold_reservation": self.threshold_reservation,
            "weight": self.weight,
            "score": self.score,
            "message": self.message,
        }


@dataclass
class GateResult:
    """Complete bid gate result."""
    project_id: str
    status: GateStatus
    score: float
    checks: List[CheckResult] = field(default_factory=list)
    reservations: List[Reservation] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stamp: str = ""
    evaluated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "status": self.status.value,
            "score": self.score,
            "checks": [c.to_dict() for c in self.checks],
            "reservations": [r.to_dict() for r in self.reservations],
            "critical_failures": self.critical_failures,
            "recommendations": self.recommendations,
            "stamp": self.stamp,
            "evaluated_at": self.evaluated_at,
        }


class BidGate:
    """Bid submission safety gate."""

    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        self.thresholds = self.config.get("thresholds", {})

    def _load_config(self, config_path: Path = None) -> dict:
        """Load gate configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "rules" / "bid_gate.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        else:
            # Return defaults
            return self._default_config()

    def _default_config(self) -> dict:
        """Default configuration if file not found."""
        return {
            "thresholds": {
                "scale_confidence": {"weight": 15, "pass": 0.85, "reservation": 0.60},
                "schedule_mapping": {"weight": 20, "pass": 0.90, "reservation": 0.70},
                "missing_sheets": {"weight": 15, "pass": 0, "reservation": 3},
                "owner_inputs_completeness": {"weight": 20, "pass": 0.80, "reservation": 0.50},
                "high_priority_rfis": {"weight": 15, "pass": 0, "reservation": 5},
                "external_works": {"weight": 8, "pass": "known", "reservation": "partial"},
                "mep_coverage": {"weight": 7, "pass": "covered", "reservation": "partial"},
            },
            "scoring": {
                "pass_threshold": 80,
                "reservation_threshold": 60,
            },
            "output": {
                "failed_stamp_text": "⚠️ NOT SUBMITTABLE - REQUIRES CLARIFICATIONS ⚠️",
            },
        }

    def evaluate(self, project_id: str, bid_data: dict) -> GateResult:
        """Evaluate bid against gate criteria."""
        checks = []
        reservations = []
        critical_failures = []

        # 1. Check for critical failures first
        critical_failures = self._check_critical_failures(bid_data)

        if critical_failures:
            return GateResult(
                project_id=project_id,
                status=GateStatus.FAIL,
                score=0,
                checks=[],
                reservations=[],
                critical_failures=critical_failures,
                recommendations=["Resolve critical failures before proceeding"],
                stamp=self.config.get("output", {}).get("failed_stamp_text", "NOT SUBMITTABLE"),
                evaluated_at=datetime.now().isoformat(),
            )

        # 2. Run threshold checks
        checks.append(self._check_scale_confidence(bid_data))
        checks.append(self._check_schedule_mapping(bid_data))
        checks.append(self._check_missing_sheets(bid_data))
        checks.append(self._check_owner_inputs(bid_data))
        checks.append(self._check_high_priority_rfis(bid_data))
        checks.append(self._check_external_works(bid_data))
        checks.append(self._check_mep_coverage(bid_data))

        # 3. Calculate weighted score
        total_weight = sum(c.weight for c in checks)
        total_score = sum(c.score * c.weight for c in checks) / total_weight if total_weight > 0 else 0

        # 4. Collect reservations from checks
        for check in checks:
            if check.status == "reservation":
                reservations.append(Reservation(
                    code=f"RES-{check.check_name.upper()[:8]}",
                    category=check.check_name,
                    description=check.message,
                    impact="May affect bid accuracy",
                    recommendation=f"Review {check.check_name.replace('_', ' ')} before submission",
                    evidence=f"Value: {check.value}, Threshold: {check.threshold_pass}",
                    severity="medium",
                ))

        # 5. Check additional reservation triggers
        additional_reservations = self._check_reservation_triggers(bid_data)
        reservations.extend(additional_reservations)

        # 6. Determine final status
        scoring = self.config.get("scoring", {})
        pass_threshold = scoring.get("pass_threshold", 80)
        reservation_threshold = scoring.get("reservation_threshold", 60)

        if total_score >= pass_threshold and not reservations:
            status = GateStatus.PASS
        elif total_score >= reservation_threshold:
            status = GateStatus.PASS_WITH_RESERVATIONS
        else:
            status = GateStatus.FAIL

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(checks, reservations)

        # 8. Set stamp for failed bids
        stamp = ""
        if status == GateStatus.FAIL:
            stamp = self.config.get("output", {}).get("failed_stamp_text", "NOT SUBMITTABLE")

        return GateResult(
            project_id=project_id,
            status=status,
            score=round(total_score, 1),
            checks=checks,
            reservations=reservations,
            critical_failures=[],
            recommendations=recommendations,
            stamp=stamp,
            evaluated_at=datetime.now().isoformat(),
        )

    def _check_critical_failures(self, bid_data: dict) -> List[str]:
        """Check for critical failures that immediately fail the gate."""
        failures = []

        # No drawings processed
        if bid_data.get("drawings_processed", 0) == 0:
            failures.append("No drawing sheets were processed")

        # Zero BOQ items
        if len(bid_data.get("priced_boq", [])) == 0:
            failures.append("BOQ has zero priced items")

        # Project value zero
        if bid_data.get("grand_total", 0) <= 0:
            failures.append("Total project value is zero or negative")

        # No scale detected
        if bid_data.get("scale_confidence", 0) == 0 and bid_data.get("drawings_processed", 0) > 0:
            failures.append("Could not detect scale on any drawing")

        # Tender deadline passed
        deadline = bid_data.get("tender_deadline")
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline)
                if datetime.now() > deadline_dt:
                    failures.append("Tender submission deadline has passed")
            except (ValueError, TypeError):
                pass

        return failures

    def _check_scale_confidence(self, bid_data: dict) -> CheckResult:
        """Check scale detection confidence."""
        thresh = self.thresholds.get("scale_confidence", {})
        value = bid_data.get("scale_confidence", 0.5)

        pass_thresh = thresh.get("pass", 0.85)
        res_thresh = thresh.get("reservation", 0.60)
        weight = thresh.get("weight", 15)

        if value >= pass_thresh:
            status = "pass"
            score = 100
            message = f"Scale confidence {value:.0%} meets threshold"
        elif value >= res_thresh:
            status = "reservation"
            score = 70
            message = f"Scale confidence {value:.0%} below ideal - quantities may have variance"
        else:
            status = "fail"
            score = 30
            message = f"Scale confidence {value:.0%} too low - quantities unreliable"

        return CheckResult(
            check_name="scale_confidence",
            status=status,
            value=value,
            threshold_pass=pass_thresh,
            threshold_reservation=res_thresh,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_schedule_mapping(self, bid_data: dict) -> CheckResult:
        """Check schedule mapping coverage."""
        thresh = self.thresholds.get("schedule_mapping", {})

        # Calculate from bid data
        total_schedule_items = bid_data.get("total_schedule_items", 0)
        mapped_items = bid_data.get("mapped_schedule_items", 0)
        value = mapped_items / total_schedule_items if total_schedule_items > 0 else 1.0

        pass_thresh = thresh.get("pass", 0.90)
        res_thresh = thresh.get("reservation", 0.70)
        weight = thresh.get("weight", 20)

        if value >= pass_thresh:
            status = "pass"
            score = 100
            message = f"Schedule mapping {value:.0%} - good coverage"
        elif value >= res_thresh:
            status = "reservation"
            score = 70
            message = f"Schedule mapping {value:.0%} - some items not mapped to drawings"
        else:
            status = "fail"
            score = 30
            message = f"Schedule mapping {value:.0%} - significant gaps in schedule coverage"

        return CheckResult(
            check_name="schedule_mapping",
            status=status,
            value=value,
            threshold_pass=pass_thresh,
            threshold_reservation=res_thresh,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_missing_sheets(self, bid_data: dict) -> CheckResult:
        """Check for missing referenced sheets."""
        thresh = self.thresholds.get("missing_sheets", {})
        value = bid_data.get("missing_sheets_count", 0)

        pass_thresh = thresh.get("pass", 0)
        res_thresh = thresh.get("reservation", 3)
        weight = thresh.get("weight", 15)

        if value <= pass_thresh:
            status = "pass"
            score = 100
            message = "All referenced sheets available"
        elif value <= res_thresh:
            status = "reservation"
            score = 70
            message = f"{value} referenced sheets missing - scope may be incomplete"
        else:
            status = "fail"
            score = 30
            message = f"{value} sheets missing - cannot accurately estimate scope"

        return CheckResult(
            check_name="missing_sheets",
            status=status,
            value=value,
            threshold_pass=pass_thresh,
            threshold_reservation=res_thresh,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_owner_inputs(self, bid_data: dict) -> CheckResult:
        """Check owner inputs completeness."""
        thresh = self.thresholds.get("owner_inputs_completeness", {})
        value = bid_data.get("owner_inputs_completeness", 0) / 100  # Convert to decimal

        pass_thresh = thresh.get("pass", 0.80)
        res_thresh = thresh.get("reservation", 0.50)
        weight = thresh.get("weight", 20)

        if value >= pass_thresh:
            status = "pass"
            score = 100
            message = f"Owner inputs {value:.0%} complete"
        elif value >= res_thresh:
            status = "reservation"
            score = 70
            message = f"Owner inputs {value:.0%} complete - defaults applied for missing"
        else:
            status = "fail"
            score = 30
            message = f"Owner inputs {value:.0%} complete - too many assumptions required"

        return CheckResult(
            check_name="owner_inputs_completeness",
            status=status,
            value=value,
            threshold_pass=pass_thresh,
            threshold_reservation=res_thresh,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_high_priority_rfis(self, bid_data: dict) -> CheckResult:
        """Check count of high-priority unresolved RFIs."""
        thresh = self.thresholds.get("high_priority_rfis", {})
        value = bid_data.get("high_priority_rfis_count", 0)

        pass_thresh = thresh.get("pass", 0)
        res_thresh = thresh.get("reservation", 5)
        weight = thresh.get("weight", 15)

        if value <= pass_thresh:
            status = "pass"
            score = 100
            message = "No high-priority RFIs pending"
        elif value <= res_thresh:
            status = "reservation"
            score = 70
            message = f"{value} high-priority RFIs pending - bid includes assumptions"
        else:
            status = "fail"
            score = 30
            message = f"{value} high-priority RFIs pending - too much uncertainty"

        return CheckResult(
            check_name="high_priority_rfis",
            status=status,
            value=value,
            threshold_pass=pass_thresh,
            threshold_reservation=res_thresh,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_external_works(self, bid_data: dict) -> CheckResult:
        """Check external works scope clarity."""
        thresh = self.thresholds.get("external_works", {})
        value = bid_data.get("external_works_status", "unknown")

        pass_val = thresh.get("pass", "known")
        res_val = thresh.get("reservation", "partial")
        weight = thresh.get("weight", 8)

        if value == pass_val or value == "known":
            status = "pass"
            score = 100
            message = "External works scope is defined"
        elif value == res_val or value == "partial":
            status = "reservation"
            score = 70
            message = "External works scope partially defined - provisional included"
        else:
            status = "reservation"  # External works unknown is a reservation, not fail
            score = 50
            message = "External works scope unknown - excluded from bid"

        return CheckResult(
            check_name="external_works",
            status=status,
            value=value,
            threshold_pass=pass_val,
            threshold_reservation=res_val,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_mep_coverage(self, bid_data: dict) -> CheckResult:
        """Check MEP scope coverage."""
        thresh = self.thresholds.get("mep_coverage", {})
        value = bid_data.get("mep_coverage_status", "unknown")

        pass_val = thresh.get("pass", "covered")
        res_val = thresh.get("reservation", "partial")
        weight = thresh.get("weight", 7)

        if value == pass_val or value == "covered":
            status = "pass"
            score = 100
            message = "MEP scope is covered"
        elif value == res_val or value == "partial":
            status = "reservation"
            score = 70
            message = "MEP scope partially covered - provisionals included"
        else:
            status = "reservation"  # MEP unknown is a reservation, not fail
            score = 50
            message = "MEP scope unknown - provisionals/exclusions noted"

        return CheckResult(
            check_name="mep_coverage",
            status=status,
            value=value,
            threshold_pass=pass_val,
            threshold_reservation=res_val,
            weight=weight,
            score=score,
            message=message,
        )

    def _check_reservation_triggers(self, bid_data: dict) -> List[Reservation]:
        """Check for additional reservation triggers."""
        reservations = []

        # Provisional items exceed 10%
        total_value = bid_data.get("grand_total", 1)
        provisional_value = bid_data.get("provisional_value", 0)
        if total_value > 0 and (provisional_value / total_value) > 0.10:
            reservations.append(Reservation(
                code="RES-PROV10",
                category="provisional_items",
                description=f"Provisional items exceed 10% of bid value ({provisional_value/total_value:.1%})",
                impact="Final cost may vary significantly",
                recommendation="Review provisional items with owner before finalization",
                evidence=f"Provisional: ₹{provisional_value:,.0f} / Total: ₹{total_value:,.0f}",
                severity="high",
            ))

        # Subcontractor quotes missing for major packages
        packages_without_quotes = bid_data.get("packages_without_quotes", [])
        major_packages = ["plumbing", "electrical", "flooring", "doors_windows"]
        missing_major = [p for p in packages_without_quotes if p in major_packages]
        if missing_major:
            reservations.append(Reservation(
                code="RES-SUBQ",
                category="subcontractor_quotes",
                description=f"No subcontractor quotes for: {', '.join(missing_major)}",
                impact="Package rates based on estimates only",
                recommendation="Obtain competitive quotes before submission",
                evidence=f"Missing quotes for {len(missing_major)} major packages",
                severity="medium",
            ))

        # Rate analysis incomplete for high-value items
        items_without_buildup = bid_data.get("items_without_rate_buildup", 0)
        if items_without_buildup > 5:
            reservations.append(Reservation(
                code="RES-RATES",
                category="rate_analysis",
                description=f"{items_without_buildup} items without detailed rate build-up",
                impact="Rate accuracy not verified",
                recommendation="Complete rate analysis for high-value items",
                evidence=f"{items_without_buildup} items using database rates without build-up",
                severity="medium",
            ))

        # Structural drawings missing
        if not bid_data.get("structural_drawings_provided", True):
            reservations.append(Reservation(
                code="RES-STRUCT",
                category="missing_drawings",
                description="Structural drawings not provided",
                impact="RCC quantities based on assumptions",
                recommendation="Request structural drawings or note quantities as indicative",
                evidence="Structural quantities derived from architectural drawings",
                severity="high",
            ))

        # MEP drawings missing
        if not bid_data.get("mep_drawings_provided", True):
            reservations.append(Reservation(
                code="RES-MEP",
                category="missing_drawings",
                description="MEP drawings not provided",
                impact="Plumbing/Electrical quantities based on norms",
                recommendation="Include MEP provisional or request drawings",
                evidence="MEP quantities based on standard norms per sqm",
                severity="medium",
            ))

        # Approval drawings only
        if bid_data.get("drawings_type") == "approval":
            reservations.append(Reservation(
                code="RES-APPRV",
                category="drawing_status",
                description="Only approval drawings provided, no GFC",
                impact="Quantities may change in GFC stage",
                recommendation="Note quantities as indicative, subject to GFC",
                evidence="Drawings marked as approval/submission stage",
                severity="medium",
            ))

        return reservations

    def _generate_recommendations(
        self,
        checks: List[CheckResult],
        reservations: List[Reservation],
    ) -> List[str]:
        """Generate recommendations based on checks and reservations."""
        recommendations = []

        # From failed/reservation checks
        for check in checks:
            if check.status == "fail":
                recommendations.append(f"CRITICAL: {check.message} - resolve before submission")
            elif check.status == "reservation":
                recommendations.append(f"REVIEW: {check.message}")

        # From reservations
        for res in reservations:
            if res.severity == "high":
                recommendations.append(f"HIGH PRIORITY: {res.recommendation}")

        # General recommendations
        if any(c.status == "reservation" for c in checks):
            recommendations.append("Document all reservations in clarifications letter")

        if len(reservations) > 3:
            recommendations.append("Consider requesting tender deadline extension to resolve uncertainties")

        return recommendations

    def generate_report(self, result: GateResult) -> str:
        """Generate markdown report for gate result."""
        lines = []

        # Stamp for failed bids
        if result.status == GateStatus.FAIL:
            lines.append(f"\n{result.stamp}\n")
            lines.append("=" * 60 + "\n")

        lines.append(f"# Bid Gate Report: {result.project_id}\n\n")
        lines.append(f"**Evaluated**: {result.evaluated_at}\n\n")

        # Status box
        status_emoji = {
            GateStatus.PASS: "✅",
            GateStatus.PASS_WITH_RESERVATIONS: "🟡",
            GateStatus.FAIL: "❌",
        }

        lines.append("## Gate Status\n\n")
        lines.append(f"| Status | Score | Reservations |\n")
        lines.append(f"|--------|-------|-------------|\n")
        lines.append(f"| {status_emoji[result.status]} **{result.status.value}** | {result.score:.1f}/100 | {len(result.reservations)} |\n\n")

        # Critical failures
        if result.critical_failures:
            lines.append("## ❌ Critical Failures\n\n")
            for failure in result.critical_failures:
                lines.append(f"- **{failure}**\n")
            lines.append("\n")

        # Check results
        lines.append("## Check Results\n\n")
        lines.append("| Check | Status | Value | Threshold | Score |\n")
        lines.append("|-------|--------|-------|-----------|-------|\n")

        for check in result.checks:
            status_icon = {"pass": "✅", "reservation": "🟡", "fail": "❌"}.get(check.status, "⚪")
            value_str = f"{check.value:.0%}" if isinstance(check.value, float) and check.value <= 1 else str(check.value)
            lines.append(f"| {check.check_name.replace('_', ' ').title()} | {status_icon} | {value_str} | {check.threshold_pass} | {check.score:.0f} |\n")

        lines.append("\n")

        # Reservations
        if result.reservations:
            lines.append("## Reservations\n\n")
            lines.append("The following reservations apply to this bid:\n\n")

            for i, res in enumerate(result.reservations, 1):
                severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(res.severity, "⚪")
                lines.append(f"### {severity_icon} {res.code}: {res.category.replace('_', ' ').title()}\n\n")
                lines.append(f"**Issue**: {res.description}\n\n")
                lines.append(f"**Impact**: {res.impact}\n\n")
                lines.append(f"**Recommendation**: {res.recommendation}\n\n")
                if res.evidence:
                    lines.append(f"*Evidence*: {res.evidence}\n\n")
                lines.append("---\n\n")

        # Recommendations
        if result.recommendations:
            lines.append("## Recommendations\n\n")
            for rec in result.recommendations:
                lines.append(f"- {rec}\n")
            lines.append("\n")

        # Summary
        lines.append("## Summary\n\n")

        if result.status == GateStatus.PASS:
            lines.append("✅ **Bid is ready for submission.**\n\n")
            lines.append("All safety checks passed. Proceed with confidence.\n")
        elif result.status == GateStatus.PASS_WITH_RESERVATIONS:
            lines.append("🟡 **Bid is submittable with documented reservations.**\n\n")
            lines.append("Ensure all reservations are documented in the clarifications letter.\n")
            lines.append("Owner should be made aware of uncertainties before contract award.\n")
        else:
            lines.append("❌ **Bid is NOT ready for submission.**\n\n")
            lines.append("Critical issues must be resolved before this bid can be submitted.\n")
            lines.append("If submission deadline is imminent, document all issues as qualifications.\n")

        return "".join(lines)
