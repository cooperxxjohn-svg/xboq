"""
Bid Readiness Report Composer

Builds comprehensive, operationally-usable NO-GO reports with:
1. Executive Summary (decision, scores, top reasons, next fixes)
2. Missing Dependencies (detailed with evidence)
3. Flagged Areas / Risks (specific, actionable)
4. RFIs Generated (grouped, deduped, exportable)
5. Trade Coverage & Priceability
6. Assumptions/Exclusions (auto-suggested, editable)
7. Audit Trail / Drawing Set Overview

All sections tied together by IDs for traceability.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import io
import zipfile
import csv

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.analysis_models import (
    PlanSetGraph, Blocker, RFIItem, TradeCoverage, BOQSkeletonItem,
    DeepAnalysisResult, ReadinessScore, EvidenceRef,
    Severity, BidImpact, RiskLevel, Trade, SheetType,
    BOQItemStatus, BOQ_SKELETON_TEMPLATE,
)


# =============================================================================
# SCORE DELTA CONFIG
# =============================================================================

# Estimated score improvement when each dependency is fixed
SCORE_DELTA_CONFIG = {
    "door_schedule": 10,
    "window_schedule": 10,
    "finish_schedule": 10,
    "section_drawings": 8,
    "elevation_drawings": 5,
    "mep_drawings": 25,
    "structural_drawings": 20,
    "scale_notation": 15,
    "legend": 3,
    "specifications": 12,
}

# Assumption templates by dependency type
ASSUMPTION_TEMPLATES = {
    "door_schedule": {
        "title": "Door Specification Assumption",
        "text": "In the absence of a door schedule, doors are assumed to be flush doors of standard commercial grade with hollow core construction. Hardware allowance of Rs. 2,500 per door included.",
        "impact": "Actual door specifications may require different materials, resulting in 15-30% cost variance.",
    },
    "window_schedule": {
        "title": "Window Specification Assumption",
        "text": "In the absence of a window schedule, windows are assumed to be 2-track aluminium sliding type with 5mm clear glass. Standard sizes as per room types.",
        "impact": "Actual window specifications may differ significantly in material and glazing, resulting in 20-40% cost variance.",
    },
    "finish_schedule": {
        "title": "Finishes Assumption",
        "text": "In the absence of a finish schedule, finishes are assumed as: Flooring - Vitrified tiles (Rs. 60/sq.ft), Walls - Plastic emulsion paint, Ceiling - POP false ceiling in living areas.",
        "impact": "Premium finishes or different materials can result in 30-50% cost variance.",
    },
    "mep_drawings": {
        "title": "MEP Scope Assumption",
        "text": "MEP scope is assumed as per standard residential/commercial norms. Electrical: 6 points per room avg. Plumbing: CP fittings of standard grade. No HVAC included unless separately specified.",
        "impact": "Actual MEP scope may be significantly different, potentially 40-60% cost variance on MEP package.",
    },
    "structural_drawings": {
        "title": "Structural Scope Assumption",
        "text": "Structural scope is provisional based on architectural drawings only. RCC framed structure assumed with M25 grade concrete and Fe500D steel.",
        "impact": "Actual structural design may vary significantly. Steel quantity provisional only.",
    },
    "scale_notation": {
        "title": "Scale Assumption",
        "text": "Drawings without explicit scale are assumed to be at 1:100 scale. All measurements derived accordingly.",
        "impact": "Incorrect scale assumption can result in systematic measurement errors affecting all quantities.",
    },
}


# =============================================================================
# REPORT DATA STRUCTURES
# =============================================================================

@dataclass
class DependencyItem:
    """A missing or incomplete dependency."""
    id: str
    dependency_type: str
    status: str  # "missing", "partial", "inconsistent"
    why_needed: str
    evidence: Dict[str, Any]
    impact_trade: str
    impact_bid: str  # "blocks_pricing", "forces_allowance", "clarification"
    cost_risk: str
    schedule_risk: str
    fix_options: List[str]
    score_delta: int
    related_rfi_ids: List[str] = field(default_factory=list)
    related_blocker_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dependency_type": self.dependency_type,
            "status": self.status,
            "why_needed": self.why_needed,
            "evidence": self.evidence,
            "impact_trade": self.impact_trade,
            "impact_bid": self.impact_bid,
            "cost_risk": self.cost_risk,
            "schedule_risk": self.schedule_risk,
            "fix_options": self.fix_options,
            "score_delta": self.score_delta,
            "related_rfi_ids": self.related_rfi_ids,
            "related_blocker_ids": self.related_blocker_ids,
        }


@dataclass
class FlaggedArea:
    """A flagged risk or conflict area."""
    id: str
    title: str
    trade: str
    severity: str
    what_flagged: str
    why_flagged: str
    evidence_pages: List[int]
    recommended_action: str
    related_rfi_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "trade": self.trade,
            "severity": self.severity,
            "what_flagged": self.what_flagged,
            "why_flagged": self.why_flagged,
            "evidence_pages": self.evidence_pages,
            "recommended_action": self.recommended_action,
            "related_rfi_ids": self.related_rfi_ids,
        }


@dataclass
class RFISummary:
    """Summary stats for RFI pack."""
    total_rfis: int
    critical_rfis: int
    high_rfis: int
    trades_affected: List[str]
    top_5_by_impact: List[str]


@dataclass
class AssumptionItem:
    """Auto-suggested assumption."""
    id: str
    title: str
    text: str
    linked_blocker_ids: List[str]
    impact_if_wrong: str
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "linked_blocker_ids": self.linked_blocker_ids,
            "impact_if_wrong": self.impact_if_wrong,
            "accepted": self.accepted,
        }


@dataclass
class DrawingSetOverview:
    """Audit trail for drawing set."""
    total_pages: int
    disciplines_detected: List[str]
    sheet_types: Dict[str, int]
    schedules_detected: List[Dict[str, Any]]
    pages_with_scale: int
    pages_without_scale: int
    rooms_detected: int
    room_names: List[str]
    door_tags_count: int
    window_tags_count: int
    door_tags_sample: List[str]
    window_tags_sample: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "disciplines_detected": self.disciplines_detected,
            "sheet_types": self.sheet_types,
            "schedules_detected": self.schedules_detected,
            "pages_with_scale": self.pages_with_scale,
            "pages_without_scale": self.pages_without_scale,
            "rooms_detected": self.rooms_detected,
            "room_names": self.room_names,
            "door_tags_count": self.door_tags_count,
            "window_tags_count": self.window_tags_count,
            "door_tags_sample": self.door_tags_sample,
            "window_tags_sample": self.window_tags_sample,
        }


@dataclass
class ExecutiveSummary:
    """Top-level summary for report."""
    decision: str  # "GO", "CONDITIONAL", "NO-GO"
    readiness_score: int
    sub_scores: Dict[str, int]
    top_3_reasons: List[str]
    top_3_fixes: List[Dict[str, Any]]  # {action, score_delta, description}
    artifacts_generated: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "readiness_score": self.readiness_score,
            "sub_scores": self.sub_scores,
            "top_3_reasons": self.top_3_reasons,
            "top_3_fixes": self.top_3_fixes,
            "artifacts_generated": self.artifacts_generated,
        }


@dataclass
class TradeCoverageRow:
    """Single trade coverage row."""
    trade: str
    coverage_pct: float
    priceable_count: int
    blocked_count: int
    top_missing: List[str]
    confidence: str
    cost_risk: str
    schedule_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade": self.trade,
            "coverage_pct": self.coverage_pct,
            "priceable_count": self.priceable_count,
            "blocked_count": self.blocked_count,
            "top_missing": self.top_missing,
            "confidence": self.confidence,
            "cost_risk": self.cost_risk,
            "schedule_risk": self.schedule_risk,
        }


@dataclass
class ReportData:
    """Complete report data structure."""
    project_id: str
    created_at: datetime

    # Section 1
    executive_summary: ExecutiveSummary

    # Section 2
    missing_dependencies: List[DependencyItem]

    # Section 3
    flagged_areas: List[FlaggedArea]
    flag_summary: Dict[str, Any]

    # Section 4
    rfis: List[Dict[str, Any]]
    rfi_summary: RFISummary
    rfis_by_trade: Dict[str, List[Dict[str, Any]]]

    # Section 5
    trade_coverage: List[TradeCoverageRow]
    priceable_categories: List[str]
    blocked_categories: List[str]

    # Section 6
    assumptions: List[AssumptionItem]

    # Section 7
    drawing_set_overview: DrawingSetOverview

    # Raw data for exports
    plan_graph: Optional[Dict[str, Any]] = None
    deep_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "executive_summary": self.executive_summary.to_dict(),
            "missing_dependencies": [d.to_dict() for d in self.missing_dependencies],
            "flagged_areas": [f.to_dict() for f in self.flagged_areas],
            "flag_summary": self.flag_summary,
            "rfis": self.rfis,
            "rfi_summary": {
                "total_rfis": self.rfi_summary.total_rfis,
                "critical_rfis": self.rfi_summary.critical_rfis,
                "high_rfis": self.rfi_summary.high_rfis,
                "trades_affected": self.rfi_summary.trades_affected,
                "top_5_by_impact": self.rfi_summary.top_5_by_impact,
            },
            "rfis_by_trade": self.rfis_by_trade,
            "trade_coverage": [t.to_dict() for t in self.trade_coverage],
            "priceable_categories": self.priceable_categories,
            "blocked_categories": self.blocked_categories,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "drawing_set_overview": self.drawing_set_overview.to_dict(),
        }


# =============================================================================
# REPORT BUILDER
# =============================================================================

class BidReadinessReportBuilder:
    """Builds comprehensive bid readiness reports from analysis results."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.blockers: List[Blocker] = []
        self.rfis: List[RFIItem] = []
        self.trade_coverage: List[TradeCoverage] = []
        self.boq_skeleton: List[BOQSkeletonItem] = []
        self.plan_graph: Optional[PlanSetGraph] = None
        self.readiness_score: Optional[ReadinessScore] = None

    def load_from_deep_analysis(self, result: DeepAnalysisResult):
        """Load data from DeepAnalysisResult."""
        self.blockers = result.blockers
        self.rfis = result.rfis
        self.trade_coverage = result.trade_coverage
        self.boq_skeleton = result.boq_skeleton
        self.plan_graph = result.plan_graph
        self.readiness_score = result.readiness_score

    def load_from_files(self, output_dir: Path):
        """Load data from output files."""
        # Load deep analysis
        deep_path = output_dir / "deep_analysis.json"
        if deep_path.exists():
            with open(deep_path) as f:
                data = json.load(f)
                self._parse_deep_analysis(data)

        # Load plan graph
        graph_path = output_dir / "plan_graph.json"
        if graph_path.exists():
            with open(graph_path) as f:
                data = json.load(f)
                self._parse_plan_graph(data)

    def _parse_deep_analysis(self, data: Dict[str, Any]):
        """Parse deep analysis JSON into models."""
        # Parse blockers
        for b_data in data.get("blockers", []):
            evidence = EvidenceRef(
                pages=b_data.get("evidence", {}).get("pages", []),
                sheets=b_data.get("evidence", {}).get("sheets", []),
                detected_entities=b_data.get("evidence", {}).get("detected_entities", {}),
                search_attempts=b_data.get("evidence", {}).get("search_attempts", {}),
                confidence=b_data.get("evidence", {}).get("confidence", 0.5),
                confidence_reason=b_data.get("evidence", {}).get("confidence_reason", ""),
            )
            blocker = Blocker(
                id=b_data.get("id", ""),
                title=b_data.get("title", ""),
                trade=Trade(b_data.get("trade", "general")),
                severity=Severity(b_data.get("severity", "medium")),
                description=b_data.get("description", ""),
                missing_dependency=b_data.get("missing_dependency", []),
                impact_cost=RiskLevel(b_data.get("impact_cost", "medium")),
                impact_schedule=RiskLevel(b_data.get("impact_schedule", "low")),
                bid_impact=BidImpact(b_data.get("bid_impact", "clarification")),
                evidence=evidence,
                fix_actions=b_data.get("fix_actions", []),
                score_delta_estimate=b_data.get("score_delta_estimate", 0),
                unlocks_boq_categories=b_data.get("unlocks_boq_categories", []),
                issue_type=b_data.get("issue_type", ""),
            )
            self.blockers.append(blocker)

        # Parse RFIs
        for r_data in data.get("rfis", []):
            evidence = EvidenceRef(
                pages=r_data.get("evidence", {}).get("pages", []),
                detected_entities=r_data.get("evidence", {}).get("detected_entities", {}),
                search_attempts=r_data.get("evidence", {}).get("search_attempts", {}),
                confidence=r_data.get("evidence", {}).get("confidence", 0.5),
            )
            rfi = RFIItem(
                id=r_data.get("id", ""),
                trade=Trade(r_data.get("trade", "general")),
                priority=Severity(r_data.get("priority", "medium")),
                question=r_data.get("question", ""),
                why_it_matters=r_data.get("why_it_matters", ""),
                evidence=evidence,
                suggested_resolution=r_data.get("suggested_resolution", ""),
                related_blocker_id=r_data.get("related_blocker_id"),
                issue_type=r_data.get("issue_type", ""),
            )
            self.rfis.append(rfi)

        # Parse trade coverage
        for t_data in data.get("trade_coverage", []):
            tc = TradeCoverage(
                trade=Trade(t_data.get("trade", "general")),
                coverage_pct=t_data.get("coverage_pct", 0),
                total_categories=t_data.get("total_categories", 0),
                priceable_count=t_data.get("priceable_count", 0),
                blocked_count=t_data.get("blocked_count", 0),
                missing_dependencies=t_data.get("missing_dependencies", []),
                cost_risk=RiskLevel(t_data.get("cost_risk", "medium")),
                schedule_risk=RiskLevel(t_data.get("schedule_risk", "low")),
                next_action=t_data.get("next_action", ""),
            )
            self.trade_coverage.append(tc)

        # Parse readiness score
        score_data = data.get("readiness_score")
        if score_data:
            self.readiness_score = ReadinessScore(
                total_score=score_data.get("total_score", 0),
                status=score_data.get("status", "NO-GO"),
                completeness_score=score_data.get("completeness_score", 0),
                measurement_score=score_data.get("measurement_score", 0),
                coverage_score=score_data.get("coverage_score", 0),
                blocker_score=score_data.get("blocker_score", 0),
                score_breakdown=score_data.get("score_breakdown", {}),
            )

    def _parse_plan_graph(self, data: Dict[str, Any]):
        """Parse plan graph JSON."""
        # Store as dict for now, we'll use it in drawing set overview
        self._plan_graph_data = data

    def build(self) -> ReportData:
        """Build complete report data."""
        # Section 1: Executive Summary
        exec_summary = self._build_executive_summary()

        # Section 2: Missing Dependencies
        dependencies = self._build_missing_dependencies()

        # Section 3: Flagged Areas
        flagged, flag_summary = self._build_flagged_areas()

        # Section 4: RFIs
        rfis_list, rfi_summary, rfis_by_trade = self._build_rfis()

        # Section 5: Trade Coverage
        coverage_rows, priceable, blocked = self._build_trade_coverage()

        # Section 6: Assumptions
        assumptions = self._build_assumptions(dependencies)

        # Section 7: Drawing Set Overview
        overview = self._build_drawing_set_overview()

        return ReportData(
            project_id=self.project_id,
            created_at=datetime.now(),
            executive_summary=exec_summary,
            missing_dependencies=dependencies,
            flagged_areas=flagged,
            flag_summary=flag_summary,
            rfis=rfis_list,
            rfi_summary=rfi_summary,
            rfis_by_trade=rfis_by_trade,
            trade_coverage=coverage_rows,
            priceable_categories=priceable,
            blocked_categories=blocked,
            assumptions=assumptions,
            drawing_set_overview=overview,
            plan_graph=getattr(self, '_plan_graph_data', None),
            deep_analysis=None,  # Can be added if needed
        )

    def _build_executive_summary(self) -> ExecutiveSummary:
        """Build Section 1: Executive Summary."""
        # Decision and scores
        if self.readiness_score:
            score = self.readiness_score.total_score
            status = self.readiness_score.status
            sub_scores = {
                "coverage": self.readiness_score.coverage_score,
                "measurement": self.readiness_score.measurement_score,
                "completeness": self.readiness_score.completeness_score,
                "blocker": self.readiness_score.blocker_score,
            }
        else:
            score = 0
            status = "NO-GO"
            sub_scores = {"coverage": 0, "measurement": 0, "completeness": 0, "blocker": 0}

        # Top 3 reasons for NO-GO
        reasons = []
        critical_blockers = [b for b in self.blockers if b.severity in [Severity.CRITICAL, Severity.HIGH]]
        for blocker in critical_blockers[:3]:
            reasons.append(blocker.title)

        if len(reasons) < 3:
            # Add generic reasons based on scores
            if sub_scores.get("measurement", 0) < 50:
                reasons.append("Scale not detected on multiple pages - measurements unreliable")
            if sub_scores.get("completeness", 0) < 50:
                reasons.append("Key schedules or specifications missing")
            if sub_scores.get("coverage", 0) < 50:
                reasons.append("Major disciplines (MEP/Structural) not found in drawing set")

        reasons = reasons[:3]

        # Top 3 fixes with score deltas
        fixes = []
        for blocker in sorted(self.blockers, key=lambda b: b.score_delta_estimate, reverse=True)[:3]:
            fixes.append({
                "action": blocker.fix_actions[0] if blocker.fix_actions else "Resolve blocker",
                "score_delta": blocker.score_delta_estimate or SCORE_DELTA_CONFIG.get(
                    blocker.missing_dependency[0] if blocker.missing_dependency else "", 5
                ),
                "description": blocker.title,
            })

        # Artifacts
        artifacts = [
            "Bid Readiness Packet (HTML/ZIP)",
            "RFI Pack (CSV + Email Drafts)",
            "Pricing Readiness Sheet (XLSX)",
        ]

        return ExecutiveSummary(
            decision=status,
            readiness_score=score,
            sub_scores=sub_scores,
            top_3_reasons=reasons,
            top_3_fixes=fixes,
            artifacts_generated=artifacts,
        )

    def _build_missing_dependencies(self) -> List[DependencyItem]:
        """Build Section 2: Missing Dependencies."""
        dependencies = []
        dep_id = 0

        for blocker in self.blockers:
            for missing in blocker.missing_dependency:
                dep_id += 1

                # Build evidence dict
                evidence = {
                    "pages": blocker.evidence.pages[:10] if blocker.evidence.pages else [],
                    "sheets": blocker.evidence.sheets[:10] if blocker.evidence.sheets else [],
                    "detected_entities": blocker.evidence.detected_entities,
                    "search_attempts": blocker.evidence.search_attempts,
                    "confidence": blocker.evidence.confidence,
                }

                # Determine why needed
                why_needed = blocker.description
                if blocker.unlocks_boq_categories:
                    why_needed += f" Blocks pricing of: {', '.join(blocker.unlocks_boq_categories[:3])}"

                # Fix options
                fix_options = list(blocker.fix_actions)
                if not fix_options:
                    fix_options = [
                        f"Upload {missing.replace('_', ' ')}",
                        f"Accept allowance with assumptions",
                    ]

                # Find related RFIs
                related_rfis = [
                    rfi.id for rfi in self.rfis
                    if rfi.related_blocker_id == blocker.id
                ]

                dep = DependencyItem(
                    id=f"DEP-{dep_id:04d}",
                    dependency_type=missing.replace('_', ' ').title(),
                    status="missing",
                    why_needed=why_needed,
                    evidence=evidence,
                    impact_trade=blocker.trade.value,
                    impact_bid=blocker.bid_impact.value,
                    cost_risk=blocker.impact_cost.value,
                    schedule_risk=blocker.impact_schedule.value,
                    fix_options=fix_options,
                    score_delta=blocker.score_delta_estimate or SCORE_DELTA_CONFIG.get(missing, 5),
                    related_rfi_ids=related_rfis,
                    related_blocker_ids=[blocker.id],
                )
                dependencies.append(dep)

        return dependencies

    def _build_flagged_areas(self) -> Tuple[List[FlaggedArea], Dict[str, Any]]:
        """Build Section 3: Flagged Areas."""
        flagged = []
        flag_id = 0

        for blocker in self.blockers:
            flag_id += 1

            # Find related RFIs
            related_rfis = [
                rfi.id for rfi in self.rfis
                if rfi.related_blocker_id == blocker.id
            ]

            flag = FlaggedArea(
                id=f"FLAG-{flag_id:04d}",
                title=blocker.title,
                trade=blocker.trade.value,
                severity=blocker.severity.value,
                what_flagged=blocker.description,
                why_flagged=f"Missing: {', '.join(blocker.missing_dependency)}. " if blocker.missing_dependency else "" +
                           f"Cost risk: {blocker.impact_cost.value}. " +
                           f"Bid impact: {blocker.bid_impact.value}.",
                evidence_pages=blocker.evidence.pages[:10] if blocker.evidence.pages else [],
                recommended_action=blocker.fix_actions[0] if blocker.fix_actions else "Generate RFI",
                related_rfi_ids=related_rfis,
            )
            flagged.append(flag)

        # Summary table
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_trade = defaultdict(set)

        for flag in flagged:
            by_severity[flag.severity] += 1
            by_trade[flag.trade].add(flag.id)

        summary = {
            "total_flags": len(flagged),
            "by_severity": dict(by_severity),
            "by_trade": {t: len(ids) for t, ids in by_trade.items()},
            "highest_severity": "critical" if by_severity.get("critical", 0) > 0 else (
                "high" if by_severity.get("high", 0) > 0 else "medium"
            ),
        }

        return flagged, summary

    def _build_rfis(self) -> Tuple[List[Dict[str, Any]], RFISummary, Dict[str, List[Dict[str, Any]]]]:
        """Build Section 4: RFIs."""
        # Convert RFIs to dicts
        rfis_list = []
        for rfi in self.rfis:
            rfi_dict = {
                "id": rfi.id,
                "trade": rfi.trade.value,
                "priority": rfi.priority.value,
                "question": rfi.question,
                "why_it_matters": rfi.why_it_matters,
                "evidence": {
                    "pages": rfi.evidence.pages[:10],
                    "detected_entities": rfi.evidence.detected_entities,
                    "search_attempts": rfi.evidence.search_attempts,
                    "confidence": rfi.evidence.confidence,
                },
                "suggested_resolution": rfi.suggested_resolution,
                "related_blocker_id": rfi.related_blocker_id,
                "issue_type": rfi.issue_type,
            }
            rfis_list.append(rfi_dict)

        # Group by trade
        by_trade = defaultdict(list)
        for rfi in rfis_list:
            by_trade[rfi["trade"]].append(rfi)

        # Summary
        critical_count = sum(1 for r in self.rfis if r.priority == Severity.CRITICAL)
        high_count = sum(1 for r in self.rfis if r.priority == Severity.HIGH)
        trades = list(set(r.trade.value for r in self.rfis))

        # Top 5 by impact (high/critical first, then by trade importance)
        sorted_rfis = sorted(
            self.rfis,
            key=lambda r: (
                0 if r.priority == Severity.CRITICAL else (1 if r.priority == Severity.HIGH else 2),
                r.trade.value
            )
        )
        top_5 = [r.question[:80] for r in sorted_rfis[:5]]

        summary = RFISummary(
            total_rfis=len(self.rfis),
            critical_rfis=critical_count,
            high_rfis=high_count,
            trades_affected=trades,
            top_5_by_impact=top_5,
        )

        return rfis_list, summary, dict(by_trade)

    def _build_trade_coverage(self) -> Tuple[List[TradeCoverageRow], List[str], List[str]]:
        """Build Section 5: Trade Coverage."""
        rows = []
        priceable = []
        blocked = []

        for tc in self.trade_coverage:
            # Confidence band
            if tc.coverage_pct >= 70:
                confidence = "high"
            elif tc.coverage_pct >= 40:
                confidence = "medium"
            else:
                confidence = "low"

            row = TradeCoverageRow(
                trade=tc.trade.value,
                coverage_pct=tc.coverage_pct,
                priceable_count=tc.priceable_count,
                blocked_count=tc.blocked_count,
                top_missing=tc.missing_dependencies[:3],
                confidence=confidence,
                cost_risk=tc.cost_risk.value,
                schedule_risk=tc.schedule_risk.value,
            )
            rows.append(row)

        # Build priceable/blocked lists from BOQ skeleton
        for item in self.boq_skeleton:
            if item.status == BOQItemStatus.PRICEABLE:
                priceable.append(f"{item.trade.value}: {item.item_name}")
            else:
                reason = f"blocked by {', '.join(item.blocked_by[:2])}" if item.blocked_by else "missing info"
                blocked.append(f"{item.trade.value}: {item.item_name} ({reason})")

        return rows, priceable[:8], blocked[:8]

    def _build_assumptions(self, dependencies: List[DependencyItem]) -> List[AssumptionItem]:
        """Build Section 6: Assumptions."""
        assumptions = []
        seen_types = set()
        assumption_id = 0

        for dep in dependencies:
            # Normalize dependency type to key
            dep_key = dep.dependency_type.lower().replace(' ', '_')

            if dep_key in seen_types:
                continue
            seen_types.add(dep_key)

            template = ASSUMPTION_TEMPLATES.get(dep_key)
            if template:
                assumption_id += 1
                assumption = AssumptionItem(
                    id=f"ASMP-{assumption_id:04d}",
                    title=template["title"],
                    text=template["text"],
                    linked_blocker_ids=dep.related_blocker_ids,
                    impact_if_wrong=template["impact"],
                    accepted=False,
                )
                assumptions.append(assumption)

        return assumptions[:10]  # Max 10 assumptions

    def _build_drawing_set_overview(self) -> DrawingSetOverview:
        """Build Section 7: Drawing Set Overview."""
        graph_data = getattr(self, '_plan_graph_data', {}) or {}

        # Schedules detected
        schedules = []
        if graph_data.get("has_door_schedule"):
            schedules.append({"type": "door_schedule", "status": "found"})
        if graph_data.get("has_window_schedule"):
            schedules.append({"type": "window_schedule", "status": "found"})
        if graph_data.get("has_finish_schedule"):
            schedules.append({"type": "finish_schedule", "status": "found"})

        return DrawingSetOverview(
            total_pages=graph_data.get("total_pages", 0),
            disciplines_detected=graph_data.get("disciplines_found", []),
            sheet_types=graph_data.get("sheet_types_found", {}),
            schedules_detected=schedules,
            pages_with_scale=graph_data.get("pages_with_scale", 0),
            pages_without_scale=graph_data.get("pages_without_scale", 0),
            rooms_detected=len(graph_data.get("all_room_names", [])),
            room_names=graph_data.get("all_room_names", [])[:20],
            door_tags_count=len(graph_data.get("all_door_tags", [])),
            window_tags_count=len(graph_data.get("all_window_tags", [])),
            door_tags_sample=graph_data.get("all_door_tags", [])[:10],
            window_tags_sample=graph_data.get("all_window_tags", [])[:10],
        )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def generate_report_html(report: ReportData) -> str:
    """Generate HTML report from report data."""
    summary = report.executive_summary

    # Build HTML sections
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bid Readiness Report - {report.project_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', -apple-system, sans-serif; line-height: 1.6; color: #1a1a1a; max-width: 900px; margin: 0 auto; padding: 2rem; }}
        h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.4rem; margin: 2rem 0 1rem; border-bottom: 2px solid #e5e5e5; padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.1rem; margin: 1rem 0 0.5rem; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 4px; font-weight: 600; }}
        .badge-nogo {{ background: #fee2e2; color: #dc2626; }}
        .badge-review {{ background: #fef3c7; color: #d97706; }}
        .badge-go {{ background: #dcfce7; color: #16a34a; }}
        .score {{ font-size: 2rem; font-weight: 700; }}
        .sub-scores {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
        .sub-score {{ background: #f5f5f5; padding: 0.5rem 1rem; border-radius: 4px; }}
        .sub-score span {{ font-weight: 600; }}
        .reasons {{ background: #fef2f2; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        .reasons ul {{ margin-left: 1.5rem; }}
        .fixes {{ background: #f0fdf4; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .card {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }}
        .severity-critical, .severity-high {{ color: #dc2626; }}
        .severity-medium {{ color: #d97706; }}
        .severity-low {{ color: #16a34a; }}
        .evidence {{ background: #f9fafb; padding: 0.75rem; border-radius: 4px; font-size: 0.875rem; margin: 0.5rem 0; }}
        .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e5e5; font-size: 0.875rem; color: #666; }}
    </style>
</head>
<body>
    <h1>Bid Readiness Report</h1>
    <p>Project: <strong>{report.project_id}</strong> | Generated: {report.created_at.strftime('%Y-%m-%d %H:%M')}</p>

    <!-- Section 1: Executive Summary -->
    <h2>1. Executive Summary</h2>
    <p class="badge badge-{'nogo' if summary.decision == 'NO-GO' else 'review' if summary.decision == 'REVIEW' else 'go'}">{summary.decision}</p>
    <p class="score">{summary.readiness_score}/100</p>

    <div class="sub-scores">
        <div class="sub-score">Coverage: <span>{summary.sub_scores.get('coverage', 0)}</span></div>
        <div class="sub-score">Measurement: <span>{summary.sub_scores.get('measurement', 0)}</span></div>
        <div class="sub-score">Completeness: <span>{summary.sub_scores.get('completeness', 0)}</span></div>
        <div class="sub-score">Blockers: <span>{summary.sub_scores.get('blocker', 0)}</span></div>
    </div>

    <div class="reasons">
        <h3>Top Reasons for {summary.decision}</h3>
        <ul>
            {''.join(f'<li>{r}</li>' for r in summary.top_3_reasons)}
        </ul>
    </div>

    <div class="fixes">
        <h3>What You Can Do Next</h3>
        <ul>
            {''.join(f'<li><strong>{f["action"]}</strong> (+{f["score_delta"]} pts) - {f["description"]}</li>' for f in summary.top_3_fixes)}
        </ul>
    </div>

    <!-- Section 2: Missing Dependencies -->
    <h2>2. Missing Dependencies ({len(report.missing_dependencies)})</h2>
    {''.join(_render_dependency_html(d) for d in report.missing_dependencies)}

    <!-- Section 3: Flagged Areas -->
    <h2>3. Flagged Areas ({len(report.flagged_areas)})</h2>
    <table>
        <tr>
            <th>Severity</th>
            <th>Count</th>
        </tr>
        <tr><td>Critical/High</td><td>{report.flag_summary.get('by_severity', {}).get('critical', 0) + report.flag_summary.get('by_severity', {}).get('high', 0)}</td></tr>
        <tr><td>Medium</td><td>{report.flag_summary.get('by_severity', {}).get('medium', 0)}</td></tr>
        <tr><td>Low</td><td>{report.flag_summary.get('by_severity', {}).get('low', 0)}</td></tr>
    </table>
    {''.join(_render_flagged_html(f) for f in report.flagged_areas[:10])}

    <!-- Section 4: RFIs -->
    <h2>4. RFIs Generated ({report.rfi_summary.total_rfis})</h2>
    <p>Critical: {report.rfi_summary.critical_rfis} | High: {report.rfi_summary.high_rfis} | Trades: {', '.join(report.rfi_summary.trades_affected)}</p>
    <h3>Top 5 by Impact</h3>
    <ol>
        {''.join(f'<li>{q}</li>' for q in report.rfi_summary.top_5_by_impact)}
    </ol>
    {''.join(_render_rfi_html(r) for r in report.rfis[:10])}

    <!-- Section 5: Trade Coverage -->
    <h2>5. Trade Coverage & Priceability</h2>
    <table>
        <tr>
            <th>Trade</th>
            <th>Coverage</th>
            <th>Priceable</th>
            <th>Blocked</th>
            <th>Cost Risk</th>
            <th>Confidence</th>
        </tr>
        {''.join(_render_coverage_row_html(t) for t in report.trade_coverage)}
    </table>

    <h3>What Can Be Priced Now</h3>
    <ul>{''.join(f'<li>{p}</li>' for p in report.priceable_categories)}</ul>

    <h3>What Is Blocked</h3>
    <ul>{''.join(f'<li>{b}</li>' for b in report.blocked_categories)}</ul>

    <!-- Section 6: Assumptions -->
    <h2>6. Suggested Assumptions ({len(report.assumptions)})</h2>
    {''.join(_render_assumption_html(a) for a in report.assumptions)}

    <!-- Section 7: Audit Trail -->
    <h2>7. Drawing Set Overview</h2>
    <table>
        <tr><td>Total Pages</td><td>{report.drawing_set_overview.total_pages}</td></tr>
        <tr><td>Disciplines Found</td><td>{', '.join(report.drawing_set_overview.disciplines_detected) or 'None'}</td></tr>
        <tr><td>Pages with Scale</td><td>{report.drawing_set_overview.pages_with_scale}</td></tr>
        <tr><td>Pages without Scale</td><td>{report.drawing_set_overview.pages_without_scale}</td></tr>
        <tr><td>Rooms Detected</td><td>{report.drawing_set_overview.rooms_detected}</td></tr>
        <tr><td>Door Tags</td><td>{report.drawing_set_overview.door_tags_count} ({', '.join(report.drawing_set_overview.door_tags_sample[:5])}...)</td></tr>
        <tr><td>Window Tags</td><td>{report.drawing_set_overview.window_tags_count} ({', '.join(report.drawing_set_overview.window_tags_sample[:5])}...)</td></tr>
    </table>

    <h3>Schedules Detected</h3>
    <ul>
        {''.join(f"<li>{s['type'].replace('_', ' ').title()}: {s['status']}</li>" for s in report.drawing_set_overview.schedules_detected) or '<li>No schedules detected</li>'}
    </ul>

    <div class="footer">
        <p>Generated by XBOQ - Pre-Bid Scope & Risk Check</p>
        <p>Report ID: {report.project_id} | {report.created_at.isoformat()}</p>
    </div>
</body>
</html>"""

    return html


def _render_dependency_html(dep: DependencyItem) -> str:
    """Render single dependency as HTML."""
    evidence_text = ""
    if dep.evidence.get("pages"):
        evidence_text += f"Pages: {', '.join(map(str, dep.evidence['pages'][:5]))}"
    if dep.evidence.get("detected_entities"):
        for k, v in list(dep.evidence["detected_entities"].items())[:2]:
            if isinstance(v, list):
                evidence_text += f" | {k}: {len(v)} items"
            elif isinstance(v, (int, float)):
                evidence_text += f" | {k}: {v}"

    return f"""
    <div class="card">
        <h3>{dep.id}: {dep.dependency_type}</h3>
        <p><strong>Status:</strong> {dep.status.upper()} | <strong>Trade:</strong> {dep.impact_trade} | <strong>Bid Impact:</strong> {dep.impact_bid.replace('_', ' ')}</p>
        <p>{dep.why_needed}</p>
        <div class="evidence">{evidence_text or 'No evidence details'}</div>
        <p><strong>Cost Risk:</strong> {dep.cost_risk} | <strong>Schedule Risk:</strong> {dep.schedule_risk} | <strong>Score Delta:</strong> +{dep.score_delta} pts</p>
        <p><strong>Fix Options:</strong> {' OR '.join(dep.fix_options[:2])}</p>
    </div>"""


def _render_flagged_html(flag: FlaggedArea) -> str:
    """Render single flagged area as HTML."""
    return f"""
    <div class="card">
        <h3 class="severity-{flag.severity}">{flag.id}: {flag.title}</h3>
        <p><strong>Trade:</strong> {flag.trade} | <strong>Severity:</strong> {flag.severity.upper()}</p>
        <p>{flag.what_flagged}</p>
        <p><em>{flag.why_flagged}</em></p>
        <p><strong>Action:</strong> {flag.recommended_action}</p>
    </div>"""


def _render_rfi_html(rfi: Dict[str, Any]) -> str:
    """Render single RFI as HTML."""
    return f"""
    <div class="card">
        <h3>{rfi['id']}: {rfi['question'][:80]}...</h3>
        <p><strong>Trade:</strong> {rfi['trade']} | <strong>Priority:</strong> {rfi['priority'].upper()}</p>
        <p>{rfi['why_it_matters']}</p>
        <p><strong>Resolution:</strong> {rfi['suggested_resolution']}</p>
    </div>"""


def _render_coverage_row_html(tc: TradeCoverageRow) -> str:
    """Render trade coverage row as HTML."""
    return f"""
    <tr>
        <td>{tc.trade.title()}</td>
        <td>{tc.coverage_pct:.0f}%</td>
        <td>{tc.priceable_count}</td>
        <td>{tc.blocked_count}</td>
        <td>{tc.cost_risk}</td>
        <td>{tc.confidence}</td>
    </tr>"""


def _render_assumption_html(a: AssumptionItem) -> str:
    """Render assumption as HTML."""
    return f"""
    <div class="card">
        <h3>{a.id}: {a.title}</h3>
        <p>{a.text}</p>
        <p><em>If wrong: {a.impact_if_wrong}</em></p>
    </div>"""


def generate_rfi_csv(report: ReportData) -> str:
    """Generate RFI CSV from report data."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "RFI ID", "Trade", "Priority", "Question", "Why It Matters",
        "Evidence Pages", "Suggested Resolution", "Related Blocker"
    ])
    writer.writeheader()

    for rfi in report.rfis:
        writer.writerow({
            "RFI ID": rfi["id"],
            "Trade": rfi["trade"].title(),
            "Priority": rfi["priority"].upper(),
            "Question": rfi["question"],
            "Why It Matters": rfi["why_it_matters"],
            "Evidence Pages": ", ".join(map(str, rfi["evidence"].get("pages", []))),
            "Suggested Resolution": rfi["suggested_resolution"],
            "Related Blocker": rfi.get("related_blocker_id", ""),
        })

    return output.getvalue()


def generate_rfi_emails(report: ReportData) -> str:
    """Generate RFI email drafts from report data."""
    emails = []

    for rfi in report.rfis:
        email = f"""
================================================================================
RFI: {rfi['question']}
================================================================================
ID: {rfi['id']}
Trade: {rfi['trade'].title()}
Priority: {rfi['priority'].upper()}

Issue:
{rfi['why_it_matters']}

Evidence:
- Pages: {', '.join(map(str, rfi['evidence'].get('pages', []))) or 'N/A'}

Requested Action:
{rfi['suggested_resolution']}

"""
        emails.append(email)

    return "\n".join(emails)


def generate_pricing_readiness_csv(report: ReportData) -> str:
    """Generate pricing readiness CSV from report data."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "Trade", "Coverage %", "Priceable Items", "Blocked Items",
        "Top Missing", "Confidence", "Cost Risk", "Schedule Risk"
    ])
    writer.writeheader()

    for tc in report.trade_coverage:
        writer.writerow({
            "Trade": tc.trade.title(),
            "Coverage %": f"{tc.coverage_pct:.0f}%",
            "Priceable Items": tc.priceable_count,
            "Blocked Items": tc.blocked_count,
            "Top Missing": ", ".join(tc.top_missing),
            "Confidence": tc.confidence.title(),
            "Cost Risk": tc.cost_risk.title(),
            "Schedule Risk": tc.schedule_risk.title(),
        })

    return output.getvalue()


def export_report_bundle(report: ReportData, output_dir: Path) -> Dict[str, Path]:
    """Export all report files to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Generate all report files with error handling
    report_files = [
        ("html_report", f"bid_readiness_report_{report.project_id}.html",
         lambda: generate_report_html(report)),
        ("rfi_csv", f"rfi_tracker_{report.project_id}.csv",
         lambda: generate_rfi_csv(report)),
        ("rfi_emails", f"rfi_emails_{report.project_id}.txt",
         lambda: generate_rfi_emails(report)),
        ("pricing_csv", f"pricing_readiness_{report.project_id}.csv",
         lambda: generate_pricing_readiness_csv(report)),
    ]

    for key, filename, generator in report_files:
        file_path = output_dir / filename
        try:
            with open(file_path, 'w') as f:
                f.write(generator())
            outputs[key] = file_path
        except (OSError, IOError) as e:
            logger.error(f"Failed to write {filename}: {e}")

    # Report JSON
    json_path = output_dir / f"report_data_{report.project_id}.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        outputs["report_json"] = json_path
    except (OSError, IOError, TypeError) as e:
        logger.error(f"Failed to write report JSON: {e}")

    # ZIP bundle
    zip_path = output_dir / f"bid_readiness_bundle_{report.project_id}.zip"
    try:
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for name, path in outputs.items():
                zf.write(path, path.name)
        outputs["zip_bundle"] = zip_path
    except (OSError, IOError) as e:
        logger.error(f"Failed to create ZIP bundle: {e}")

    return outputs


# =============================================================================
# ENTRY POINTS
# =============================================================================

def build_report_data(project_id: str, output_dir: Optional[Path] = None) -> ReportData:
    """
    Build complete report data from project outputs.

    Args:
        project_id: Project identifier
        output_dir: Path to output directory (default: out/{project_id})

    Returns:
        ReportData with all sections populated
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "out" / project_id

    builder = BidReadinessReportBuilder(project_id)
    builder.load_from_files(output_dir)
    return builder.build()


def build_report_from_analysis(
    project_id: str,
    result: DeepAnalysisResult
) -> ReportData:
    """
    Build report data directly from DeepAnalysisResult.

    Args:
        project_id: Project identifier
        result: DeepAnalysisResult from analysis pipeline

    Returns:
        ReportData with all sections populated
    """
    builder = BidReadinessReportBuilder(project_id)
    builder.load_from_deep_analysis(result)
    return builder.build()
