"""
Completeness Scoring - Score each package and overall project.

Scores based on:
- Evidence coverage
- Schedule mapping coverage
- Takeoff presence
- Cross-reference completeness

Outputs:
- out/<project_id>/scope/completeness_report.json
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from .register import ScopeRegister, ScopeItem, ScopeStatus

logger = logging.getLogger(__name__)


@dataclass
class PackageScore:
    """Score for a single package."""
    package: str
    package_name: str
    score: float  # 0-100
    evidence_score: float
    schedule_score: float
    takeoff_score: float
    subpackage_scores: Dict[str, float] = field(default_factory=dict)
    detected_count: int = 0
    implied_count: int = 0
    unknown_count: int = 0
    missing_count: int = 0
    total_subpackages: int = 0
    highest_risk_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package": self.package,
            "package_name": self.package_name,
            "score": round(self.score, 1),
            "breakdown": {
                "evidence": round(self.evidence_score, 1),
                "schedule": round(self.schedule_score, 1),
                "takeoff": round(self.takeoff_score, 1),
            },
            "subpackage_scores": {k: round(v, 1) for k, v in self.subpackage_scores.items()},
            "status_counts": {
                "DETECTED": self.detected_count,
                "IMPLIED": self.implied_count,
                "UNKNOWN": self.unknown_count,
                "MISSING_INPUT": self.missing_count,
            },
            "total_subpackages": self.total_subpackages,
            "highest_risk_items": self.highest_risk_items,
        }


@dataclass
class RiskItem:
    """A high-risk scope item."""
    rank: int
    package: str
    subpackage: str
    description: str
    status: str
    score: float
    impact: str
    action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "package": self.package,
            "subpackage": self.subpackage,
            "description": self.description,
            "status": self.status,
            "score": round(self.score, 1),
            "impact": self.impact,
            "action": self.action,
        }


@dataclass
class CompletenessReport:
    """Complete project completeness report."""
    project_id: str
    overall_score: float = 0.0  # 0-100
    grade: str = "F"  # A, B, C, D, F
    package_scores: List[PackageScore] = field(default_factory=list)
    top_risks: List[RiskItem] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "generated": datetime.now().isoformat(),
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "summary": self.summary,
            "package_scores": [p.to_dict() for p in self.package_scores],
            "top_risks": [r.to_dict() for r in self.top_risks],
        }

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved completeness report to: {path}")


class CompletenessScorer:
    """
    Scores project scope completeness.
    """

    # Package weights for overall score
    PACKAGE_WEIGHTS = {
        "rcc_concrete": 15,
        "reinforcement_steel": 10,
        "masonry_partitions": 10,
        "waterproofing": 10,
        "doors_windows": 10,
        "finishes_floor": 8,
        "finishes_wall": 8,
        "finishes_ceiling": 5,
        "plumbing_water_supply": 8,
        "plumbing_sanitary_swr": 5,
        "electrical_power_lighting": 8,
        "fire_hvac": 3,
        "external_works_paving": 5,
        "external_works_drainage": 5,
        "external_works_compound": 3,
        # Lower weight for prelims and others
        "prelims_general_conditions": 2,
        "demolition": 1,
        "earthwork": 3,
        "formwork_shuttering": 3,
        "testing_commissioning": 1,
        "plumbing_storm_rainwater": 2,
    }

    # Critical packages that heavily impact score
    CRITICAL_PACKAGES = [
        "waterproofing",
        "fire_hvac",
        "plumbing_sanitary_swr",
        "electrical_power_lighting",
    ]

    def __init__(self):
        pass

    def score(
        self,
        register: ScopeRegister,
        evidence_store: Dict[str, Any],
    ) -> CompletenessReport:
        """
        Score project completeness.

        Args:
            register: Scope register
            evidence_store: Evidence store dict

        Returns:
            CompletenessReport
        """
        report = CompletenessReport(project_id=register.project_id)

        # Group items by package
        by_package: Dict[str, List[ScopeItem]] = {}
        for item in register.items:
            if item.package not in by_package:
                by_package[item.package] = []
            by_package[item.package].append(item)

        # Score each package
        all_risks: List[Tuple[float, ScopeItem]] = []

        for pkg_key, items in by_package.items():
            pkg_score = self._score_package(pkg_key, items, evidence_store)
            report.package_scores.append(pkg_score)

            # Collect risk items
            for item in items:
                if item.status in [ScopeStatus.UNKNOWN, ScopeStatus.MISSING_INPUT]:
                    item_score = self._calculate_item_score(item)
                    all_risks.append((item_score, item))
                elif item.status == ScopeStatus.IMPLIED and item.confidence < 0.5:
                    item_score = self._calculate_item_score(item)
                    all_risks.append((item_score, item))

        # Sort packages by score
        report.package_scores.sort(key=lambda x: x.score)

        # Calculate overall score
        report.overall_score = self._calculate_overall_score(report.package_scores)
        report.grade = self._assign_grade(report.overall_score)

        # Get top 10 risks
        all_risks.sort(key=lambda x: x[0])
        report.top_risks = self._create_risk_items(all_risks[:10])

        # Summary statistics
        report.summary = self._create_summary(register, report)

        return report

    def _score_package(
        self,
        pkg_key: str,
        items: List[ScopeItem],
        evidence_store: Dict[str, Any],
    ) -> PackageScore:
        """Score a single package."""
        pkg_name = items[0].package_name if items else pkg_key

        score = PackageScore(
            package=pkg_key,
            package_name=pkg_name,
            score=0,
            evidence_score=0,
            schedule_score=0,
            takeoff_score=0,
            total_subpackages=len(items),
        )

        if not items:
            return score

        # Count by status
        for item in items:
            if item.status == ScopeStatus.DETECTED:
                score.detected_count += 1
            elif item.status == ScopeStatus.IMPLIED:
                score.implied_count += 1
            elif item.status == ScopeStatus.UNKNOWN:
                score.unknown_count += 1
            elif item.status == ScopeStatus.MISSING_INPUT:
                score.missing_count += 1

        # Score each subpackage
        subpkg_scores = []
        for item in items:
            subpkg_score = self._calculate_item_score(item)
            score.subpackage_scores[item.subpackage] = subpkg_score
            subpkg_scores.append(subpkg_score)

            # Track highest risk items
            if subpkg_score < 50:
                score.highest_risk_items.append(item.subpackage_name)

        # Evidence score (based on matched keywords and snippets)
        total_evidence = sum(len(item.evidence_pages) for item in items)
        total_keywords = sum(len(item.keywords_found) for item in items)
        score.evidence_score = min(
            (total_evidence * 10 + total_keywords * 5) / len(items),
            100
        )

        # Schedule score (based on has_schedule flag)
        schedule_items = [item for item in items if item.has_schedule]
        score.schedule_score = (len(schedule_items) / len(items)) * 100 if items else 0

        # Takeoff score (based on has_takeoff flag)
        takeoff_items = [item for item in items if item.has_takeoff]
        score.takeoff_score = (len(takeoff_items) / len(items)) * 100 if items else 0

        # Overall package score (weighted average)
        avg_subpkg_score = sum(subpkg_scores) / len(subpkg_scores) if subpkg_scores else 0
        score.score = (
            avg_subpkg_score * 0.4 +
            score.evidence_score * 0.3 +
            score.schedule_score * 0.15 +
            score.takeoff_score * 0.15
        )

        # Penalty for critical packages with low scores
        if pkg_key in self.CRITICAL_PACKAGES and score.unknown_count > 0:
            penalty = score.unknown_count * 10
            score.score = max(0, score.score - penalty)

        return score

    def _calculate_item_score(self, item: ScopeItem) -> float:
        """Calculate score for a single item."""
        base_score = {
            ScopeStatus.DETECTED: 100,
            ScopeStatus.IMPLIED: 60,
            ScopeStatus.UNKNOWN: 10,
            ScopeStatus.MISSING_INPUT: 20,
            ScopeStatus.NOT_APPLICABLE: 100,
        }.get(item.status, 0)

        # Adjust based on confidence
        score = base_score * item.confidence if item.confidence > 0 else base_score * 0.5

        # Bonus for evidence
        if item.evidence_pages:
            score += min(len(item.evidence_pages) * 5, 20)

        # Bonus for schedule
        if item.has_schedule:
            score += 10

        # Bonus for takeoff
        if item.has_takeoff:
            score += 10

        return min(score, 100)

    def _calculate_overall_score(
        self,
        package_scores: List[PackageScore]
    ) -> float:
        """Calculate weighted overall score."""
        total_weight = 0
        weighted_sum = 0

        for pkg_score in package_scores:
            weight = self.PACKAGE_WEIGHTS.get(pkg_score.package, 1)
            weighted_sum += pkg_score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        return weighted_sum / total_weight

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _create_risk_items(
        self,
        risks: List[Tuple[float, ScopeItem]]
    ) -> List[RiskItem]:
        """Create ranked risk items."""
        result = []
        for rank, (score, item) in enumerate(risks, 1):
            # Determine impact
            if item.package in self.CRITICAL_PACKAGES:
                impact = "Critical - Major cost/quality impact"
            elif item.status == ScopeStatus.UNKNOWN:
                impact = "High - Scope completely undefined"
            else:
                impact = "Medium - Partial information available"

            result.append(RiskItem(
                rank=rank,
                package=item.package_name,
                subpackage=item.subpackage_name,
                description=f"Status: {item.status.value}",
                status=item.status.value,
                score=score,
                impact=impact,
                action=item.recommended_action or "Confirm scope with client/architect",
            ))

        return result

    def _create_summary(
        self,
        register: ScopeRegister,
        report: CompletenessReport
    ) -> Dict[str, Any]:
        """Create summary statistics."""
        status_summary = register.get_summary()

        # Calculate percentages
        total = len(register.items)
        detected_pct = (status_summary.get("DETECTED", 0) / total * 100) if total else 0
        implied_pct = (status_summary.get("IMPLIED", 0) / total * 100) if total else 0
        unknown_pct = (status_summary.get("UNKNOWN", 0) / total * 100) if total else 0
        missing_pct = (status_summary.get("MISSING_INPUT", 0) / total * 100) if total else 0

        # Find weakest packages
        weak_packages = [
            p.package_name
            for p in sorted(report.package_scores, key=lambda x: x.score)[:5]
            if p.score < 60
        ]

        # Find strongest packages
        strong_packages = [
            p.package_name
            for p in sorted(report.package_scores, key=lambda x: -x.score)[:5]
            if p.score >= 80
        ]

        return {
            "total_scope_items": total,
            "status_breakdown": {
                "DETECTED": f"{detected_pct:.0f}%",
                "IMPLIED": f"{implied_pct:.0f}%",
                "UNKNOWN": f"{unknown_pct:.0f}%",
                "MISSING_INPUT": f"{missing_pct:.0f}%",
            },
            "weakest_packages": weak_packages,
            "strongest_packages": strong_packages,
            "critical_gaps": len([r for r in report.top_risks if "Critical" in r.impact]),
            "high_gaps": len([r for r in report.top_risks if "High" in r.impact]),
            "recommendation": self._get_recommendation(report.overall_score, report.top_risks),
        }

    def _get_recommendation(
        self,
        score: float,
        risks: List[RiskItem]
    ) -> str:
        """Get overall recommendation based on score."""
        critical_count = len([r for r in risks if "Critical" in r.impact])

        if score >= 80 and critical_count == 0:
            return "Scope is well-documented. Proceed with detailed estimation."
        elif score >= 60:
            return "Scope is partially documented. Clarify items in checklist before bid submission."
        elif score >= 40:
            return "Significant scope gaps exist. Extensive clarification required before pricing."
        else:
            return "CAUTION: Major scope uncertainty. Consider requesting complete drawing set."


def score_completeness(
    register: ScopeRegister,
    evidence_store: Dict[str, Any],
    output_dir: Path,
) -> CompletenessReport:
    """
    Convenience function to score and save completeness report.

    Args:
        register: Scope register
        evidence_store: Evidence store dict
        output_dir: Output directory

    Returns:
        CompletenessReport
    """
    scorer = CompletenessScorer()
    report = scorer.score(register, evidence_store)

    # Save report
    scope_dir = output_dir / "scope"
    scope_dir.mkdir(parents=True, exist_ok=True)
    report.save(scope_dir / "completeness_report.json")

    return report
