"""
Dependency Reasoner

Applies deterministic rules to detect missing dependencies in a plan set.
Takes PlanSetGraph as input and outputs Blockers with evidence.

Rules are based on construction industry logic:
- Detected doors without door schedule → Blocker
- Detected windows without window schedule → Blocker
- Floor plans without sections → Blocker
- Pages without scale → Blocker
- Missing MEP disciplines → Blocker
etc.

================================================================================
DECISION LOGIC & BLOCKER MAP
================================================================================

BLOCKER DEFINITIONS (Lines 53-515):
  - DoorsNeedScheduleRule      → BLK-0010 (Trade: ARCHITECTURAL)
  - WindowsNeedScheduleRule    → BLK-0011 (Trade: ARCHITECTURAL)
  - FinishScheduleRule         → BLK-0012 (Trade: FINISHES)
  - ScaleMissingRule           → BLK-0013 (Trade: GENERAL, affects: all measured items)
  - NoSectionsRule             → BLK-0002 (Trade: ARCHITECTURAL)
  - NoElevationsRule           → BLK-0001 (Trade: ARCHITECTURAL)
  - NoMEPDrawingsRule          → BLK-0003 (Trade: MEP)
  - NoStructuralDrawingsRule   → BLK-0007 (Trade: STRUCTURAL)

BLOCKER EVALUATION (Lines 541-571):
  - DependencyReasoner.reason() applies all rules to graph
  - Each rule returns Optional[Blocker] based on graph state
  - Blockers sorted by severity (CRITICAL > HIGH > MEDIUM > LOW)

TRADE COVERAGE (Lines 588-658):
  - DependencyReasoner.compute_trade_coverage() computes per-trade coverage
  - Coverage = (priceable_categories / total_categories) * 100
  - Categories blocked if they appear in blocker.unlocks_boq_categories
  - Risk levels derived from HIGH/CRITICAL blocker counts per trade

DECISION (See llm_enrichment.py Lines 366-446):
  - calculate_readiness_score() produces final score and decision
  - Components: completeness (30%), measurement (25%), coverage (25%), blocker (20%)
  - Decision: PASS (>=70, 0 critical), CONDITIONAL (>=50 or some trades OK), NO-GO (<50)

BOQ SKELETON (Lines 660-708):
  - build_boq_skeleton() maps blockers to BOQ line items
  - Items marked PRICEABLE or BLOCKED based on blocker dependencies
================================================================================
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.analysis_models import (
    PlanSetGraph, PlanSheet,
    Blocker, RFIItem, EvidenceRef, TradeCoverage, BOQSkeletonItem,
    Severity, BidImpact, RiskLevel, Trade, SheetType, Discipline,
    BOQItemStatus, BOQ_SKELETON_TEMPLATE,
    create_blocker_id, create_rfi_id,
    RunCoverage, CoverageStatus, SelectionMode,
)


# =============================================================================
# DEPENDENCY RULES
# =============================================================================

class DependencyRule:
    """Base class for dependency rules."""

    def __init__(self, rule_id: str, description: str):
        self.rule_id = rule_id
        self.description = description

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        """
        Check the rule against the graph.

        Args:
            graph: PlanSetGraph to analyze
            run_coverage: Optional RunCoverage for coverage-gated assertions

        Returns:
            Blocker if rule is violated, None otherwise
        """
        raise NotImplementedError


class DoorsNeedScheduleRule(DependencyRule):
    """Detected doors without door schedule."""

    def __init__(self):
        super().__init__(
            "DOORS_NEED_SCHEDULE",
            "Door tags detected but no door schedule found"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        door_tags = graph.all_door_tags
        if not door_tags:
            return None  # No doors detected

        if graph.has_door_schedule:
            return None  # Schedule exists

        # Find pages where doors were detected
        evidence_pages = []
        for sheet in graph.sheets:
            if sheet.detected.get('door_tags'):
                evidence_pages.append(sheet.page_index)

        # --- Coverage gate ---
        # Schedule pages may not have been deep-processed in FAST_BUDGET mode.
        # If schedule doc_type was NOT fully covered, we can't confirm absence.
        sched_status = CoverageStatus.NOT_FOUND_AFTER_SEARCH  # default: full confidence
        if run_coverage and run_coverage.selection_mode == SelectionMode.FAST_BUDGET:
            sched_status = run_coverage.is_doc_type_covered("schedule")

        if sched_status == CoverageStatus.UNKNOWN_NOT_PROCESSED:
            # Downgrade: can't confirm absence — emit coverage-gap blocker
            return Blocker(
                id=create_blocker_id(10),
                title=f"Door schedule required - {len(set(door_tags))} door types detected",
                trade=Trade.ARCHITECTURAL,
                severity=Severity.MEDIUM,  # downgraded from HIGH
                description=(
                    f"Detected {len(door_tags)} door instances with {len(set(door_tags))} unique tags. "
                    f"Door schedule needed to price doors, frames, and hardware. "
                    f"NOTE: Schedule pages were not fully processed due to OCR budget — "
                    f"schedule may exist on unprocessed pages."
                ),
                missing_dependency=["door_schedule"],
                impact_cost=RiskLevel.MEDIUM,
                impact_schedule=RiskLevel.LOW,
                bid_impact=BidImpact.CLARIFICATION_NEEDED,  # downgraded from BLOCKS_PRICING
                evidence=EvidenceRef(
                    pages=evidence_pages[:10],
                    detected_entities={
                        "door_tags": door_tags[:20],
                        "door_count": len(door_tags),
                        "unique_types": len(set(door_tags)),
                    },
                    search_attempts={
                        "schedule_keywords": ["door schedule", "door sch", "door listing"],
                        "schedule_sheet_type": "checked for SheetType.SCHEDULE with 'door'",
                        "coverage_note": "schedule doc_type NOT fully processed",
                    },
                    confidence=0.4,  # reduced — can't confirm absence
                    confidence_reason="Door tags detected but schedule pages not fully processed (coverage gap)"
                ),
                fix_actions=[
                    "Provide door schedule with columns: Mark, Size (W x H), Type, Frame Material, Shutter Material, Hardware",
                    "Or clarify if doors are out of scope for this tender",
                ],
                score_delta_estimate=4,  # reduced impact
                unlocks_boq_categories=["doors"],
                issue_type="missing_schedule",
                coverage_status=CoverageStatus.UNKNOWN_NOT_PROCESSED.value,
            )

        # Full confidence: schedule pages were searched and schedule not found
        return Blocker(
            id=create_blocker_id(10),  # Reserve 10-19 for schedule blockers
            title=f"Door schedule required - {len(set(door_tags))} door types detected",
            trade=Trade.ARCHITECTURAL,
            severity=Severity.HIGH,
            description=f"Detected {len(door_tags)} door instances with {len(set(door_tags))} unique tags. "
                       f"Door schedule needed to price doors, frames, and hardware.",
            missing_dependency=["door_schedule"],
            impact_cost=RiskLevel.HIGH,
            impact_schedule=RiskLevel.MEDIUM,
            bid_impact=BidImpact.BLOCKS_PRICING,
            evidence=EvidenceRef(
                pages=evidence_pages[:10],  # First 10 pages
                detected_entities={
                    "door_tags": door_tags[:20],  # First 20 unique tags
                    "door_count": len(door_tags),
                    "unique_types": len(set(door_tags)),
                },
                search_attempts={
                    "schedule_keywords": ["door schedule", "door sch", "door listing"],
                    "schedule_sheet_type": "checked for SheetType.SCHEDULE with 'door'",
                },
                confidence=0.9,
                confidence_reason="Door tags clearly detected in floor plans"
            ),
            fix_actions=[
                "Provide door schedule with columns: Mark, Size (W x H), Type, Frame Material, Shutter Material, Hardware",
                "Or clarify if doors are out of scope for this tender",
            ],
            score_delta_estimate=8,
            unlocks_boq_categories=["doors"],
            issue_type="missing_schedule",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class WindowsNeedScheduleRule(DependencyRule):
    """Detected windows without window schedule."""

    def __init__(self):
        super().__init__(
            "WINDOWS_NEED_SCHEDULE",
            "Window tags detected but no window schedule found"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        window_tags = graph.all_window_tags
        if not window_tags:
            return None

        if graph.has_window_schedule:
            return None

        evidence_pages = []
        for sheet in graph.sheets:
            if sheet.detected.get('window_tags'):
                evidence_pages.append(sheet.page_index)

        # --- Coverage gate ---
        sched_status = CoverageStatus.NOT_FOUND_AFTER_SEARCH
        if run_coverage and run_coverage.selection_mode == SelectionMode.FAST_BUDGET:
            sched_status = run_coverage.is_doc_type_covered("schedule")

        if sched_status == CoverageStatus.UNKNOWN_NOT_PROCESSED:
            return Blocker(
                id=create_blocker_id(11),
                title=f"Window schedule required - {len(set(window_tags))} window types detected",
                trade=Trade.ARCHITECTURAL,
                severity=Severity.MEDIUM,
                description=(
                    f"Detected {len(window_tags)} window instances with {len(set(window_tags))} unique tags. "
                    f"Window schedule needed to price windows, frames, and glazing. "
                    f"NOTE: Schedule pages were not fully processed due to OCR budget — "
                    f"schedule may exist on unprocessed pages."
                ),
                missing_dependency=["window_schedule"],
                impact_cost=RiskLevel.MEDIUM,
                impact_schedule=RiskLevel.LOW,
                bid_impact=BidImpact.CLARIFICATION_NEEDED,
                evidence=EvidenceRef(
                    pages=evidence_pages[:10],
                    detected_entities={
                        "window_tags": window_tags[:20],
                        "window_count": len(window_tags),
                        "unique_types": len(set(window_tags)),
                    },
                    search_attempts={
                        "schedule_keywords": ["window schedule", "window sch", "glazing schedule"],
                        "schedule_sheet_type": "checked for SheetType.SCHEDULE with 'window'",
                        "coverage_note": "schedule doc_type NOT fully processed",
                    },
                    confidence=0.4,
                    confidence_reason="Window tags detected but schedule pages not fully processed (coverage gap)"
                ),
                fix_actions=[
                    "Provide window schedule with columns: Mark, Size (W x H), Type, Frame, Glass spec, Grille",
                    "Or clarify if windows are out of scope for this tender",
                ],
                score_delta_estimate=4,
                unlocks_boq_categories=["windows"],
                issue_type="missing_schedule",
                coverage_status=CoverageStatus.UNKNOWN_NOT_PROCESSED.value,
            )

        return Blocker(
            id=create_blocker_id(11),
            title=f"Window schedule required - {len(set(window_tags))} window types detected",
            trade=Trade.ARCHITECTURAL,
            severity=Severity.HIGH,
            description=f"Detected {len(window_tags)} window instances with {len(set(window_tags))} unique tags. "
                       f"Window schedule needed to price windows, frames, and glazing.",
            missing_dependency=["window_schedule"],
            impact_cost=RiskLevel.HIGH,
            impact_schedule=RiskLevel.MEDIUM,
            bid_impact=BidImpact.BLOCKS_PRICING,
            evidence=EvidenceRef(
                pages=evidence_pages[:10],
                detected_entities={
                    "window_tags": window_tags[:20],
                    "window_count": len(window_tags),
                    "unique_types": len(set(window_tags)),
                },
                search_attempts={
                    "schedule_keywords": ["window schedule", "window sch", "glazing schedule"],
                    "schedule_sheet_type": "checked for SheetType.SCHEDULE with 'window'",
                },
                confidence=0.9,
                confidence_reason="Window tags clearly detected in floor plans"
            ),
            fix_actions=[
                "Provide window schedule with columns: Mark, Size (W x H), Type, Frame, Glass spec, Grille",
                "Or clarify if windows are out of scope for this tender",
            ],
            score_delta_estimate=8,
            unlocks_boq_categories=["windows"],
            issue_type="missing_schedule",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class FinishScheduleRule(DependencyRule):
    """Rooms detected but no finish schedule found."""

    def __init__(self):
        super().__init__(
            "ROOMS_NEED_FINISH_SCHEDULE",
            "Rooms detected but no finish schedule found"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        room_names = graph.all_room_names
        if not room_names:
            return None

        if graph.has_finish_schedule:
            return None

        evidence_pages = []
        for sheet in graph.sheets:
            if sheet.detected.get('room_names'):
                evidence_pages.append(sheet.page_index)

        # --- Coverage gate ---
        sched_status = CoverageStatus.NOT_FOUND_AFTER_SEARCH
        if run_coverage and run_coverage.selection_mode == SelectionMode.FAST_BUDGET:
            sched_status = run_coverage.is_doc_type_covered("schedule")

        if sched_status == CoverageStatus.UNKNOWN_NOT_PROCESSED:
            return Blocker(
                id=create_blocker_id(12),
                title=f"Finish schedule required - {len(room_names)} room types detected",
                trade=Trade.FINISHES,
                severity=Severity.MEDIUM,
                description=(
                    f"Detected {len(room_names)} room types. Finish schedule needed to "
                    f"specify flooring, wall finish, ceiling, and other finishes per room. "
                    f"NOTE: Schedule pages were not fully processed due to OCR budget — "
                    f"schedule may exist on unprocessed pages."
                ),
                missing_dependency=["finish_schedule"],
                impact_cost=RiskLevel.MEDIUM,
                impact_schedule=RiskLevel.LOW,
                bid_impact=BidImpact.CLARIFICATION_NEEDED,
                evidence=EvidenceRef(
                    pages=evidence_pages[:10],
                    detected_entities={
                        "room_names": room_names[:20],
                        "room_count": len(room_names),
                    },
                    search_attempts={
                        "schedule_keywords": ["finish schedule", "room finish", "interior finish"],
                        "schedule_sheet_type": "checked for SheetType.SCHEDULE with 'finish'",
                        "coverage_note": "schedule doc_type NOT fully processed",
                    },
                    confidence=0.4,
                    confidence_reason="Room names detected but schedule pages not fully processed (coverage gap)"
                ),
                fix_actions=[
                    "Provide finish schedule with columns: Room Name, Floor Finish, Wall Finish, Ceiling Finish, Skirting",
                    "Or provide specifications document with finish details",
                ],
                score_delta_estimate=4,
                unlocks_boq_categories=["wall_painting", "floor_tiling", "wall_tiling", "false_ceiling"],
                issue_type="missing_schedule",
                coverage_status=CoverageStatus.UNKNOWN_NOT_PROCESSED.value,
            )

        return Blocker(
            id=create_blocker_id(12),
            title=f"Finish schedule required - {len(room_names)} room types detected",
            trade=Trade.FINISHES,
            severity=Severity.HIGH,
            description=f"Detected {len(room_names)} room types. Finish schedule needed to "
                       f"specify flooring, wall finish, ceiling, and other finishes per room.",
            missing_dependency=["finish_schedule"],
            impact_cost=RiskLevel.HIGH,
            impact_schedule=RiskLevel.MEDIUM,
            bid_impact=BidImpact.BLOCKS_PRICING,
            evidence=EvidenceRef(
                pages=evidence_pages[:10],
                detected_entities={
                    "room_names": room_names[:20],
                    "room_count": len(room_names),
                },
                search_attempts={
                    "schedule_keywords": ["finish schedule", "room finish", "interior finish"],
                    "schedule_sheet_type": "checked for SheetType.SCHEDULE with 'finish'",
                },
                confidence=0.85,
                confidence_reason="Room names detected in floor plans"
            ),
            fix_actions=[
                "Provide finish schedule with columns: Room Name, Floor Finish, Wall Finish, Ceiling Finish, Skirting",
                "Or provide specifications document with finish details",
            ],
            score_delta_estimate=8,
            unlocks_boq_categories=["wall_painting", "floor_tiling", "wall_tiling", "false_ceiling"],
            issue_type="missing_schedule",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class ScaleMissingRule(DependencyRule):
    """Pages without detected scale."""

    def __init__(self):
        super().__init__(
            "SCALE_MISSING",
            "Pages without detectable scale notation"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        pages_without = graph.pages_without_scale
        total_pages = graph.total_pages

        if pages_without == 0:
            return None

        # Only flag if significant portion missing scale
        if pages_without < 3 and pages_without / max(total_pages, 1) < 0.2:
            return None

        # Get pages without scale
        evidence_pages = []
        for sheet in graph.sheets:
            if not sheet.detected.get('has_scale'):
                evidence_pages.append(sheet.page_index)

        # Calculate severity and bid_impact based on proportion
        scale_ratio = pages_without / max(total_pages, 1)

        # More nuanced severity levels
        # If scale found on some pages, demote severity (can infer scale from other pages)
        if graph.pages_with_scale > 0:
            severity = Severity.MEDIUM
            bid_impact = BidImpact.FORCES_ALLOWANCE
        elif scale_ratio > 0.8:
            severity = Severity.HIGH
            bid_impact = BidImpact.FORCES_ALLOWANCE
        else:
            severity = Severity.MEDIUM
            bid_impact = BidImpact.FORCES_ALLOWANCE

        # Scale check uses graph (all pages) — no coverage gate needed
        return Blocker(
            id=create_blocker_id(13),
            title=f"Scale not detected on {pages_without} drawing pages",
            trade=Trade.GENERAL,
            severity=severity,
            affected_trades=[Trade.CIVIL, Trade.STRUCTURAL, Trade.ARCHITECTURAL, Trade.FINISHES],
            description=f"{pages_without} of {total_pages} pages have no detectable scale. "
                       f"Area measurements may be unreliable without scale notation. "
                       f"Dimension strings on drawings can still be used for pricing.",
            missing_dependency=["scale_notation"],
            impact_cost=RiskLevel.MEDIUM,
            impact_schedule=RiskLevel.LOW,
            bid_impact=bid_impact,
            evidence=EvidenceRef(
                pages=evidence_pages[:15],
                detected_entities={
                    "pages_without_scale": pages_without,
                    "pages_with_scale": graph.pages_with_scale,
                    "total_pages": total_pages,
                },
                search_attempts={
                    "scale_patterns": ["1:100", "1:50", "1:200", "SCALE ="],
                    "search_type": "regex pattern matching",
                },
                confidence=0.8,
                confidence_reason="Scale notation pattern search"
            ),
            fix_actions=[
                "Confirm scale with architect/engineer",
                "Use dimension strings from drawings for area calculations",
                "Or provide scale in tender documents",
            ],
            score_delta_estimate=10,
            unlocks_boq_categories=["all_measured_items"],
            issue_type="scale_issue",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class NoSectionsRule(DependencyRule):
    """Floor plans without sections."""

    def __init__(self):
        super().__init__(
            "NO_SECTIONS",
            "Floor plans found but no section drawings"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        has_floor_plans = SheetType.FLOOR_PLAN.value in graph.sheet_types_found
        has_sections = SheetType.SECTION.value in graph.sheet_types_found

        if not has_floor_plans:
            return None

        if has_sections:
            return None

        floor_plan_pages = []
        for sheet in graph.sheets:
            if sheet.sheet_type == SheetType.FLOOR_PLAN:
                floor_plan_pages.append(sheet.page_index)

        # Sections detected at page_index time (all pages) — no coverage gate needed
        return Blocker(
            id=create_blocker_id(2),
            title="No sections found in drawing set",
            trade=Trade.ARCHITECTURAL,
            severity=Severity.HIGH,
            description=f"Drawing set has {graph.total_pages} pages with floor plans but no sections detected. "
                       f"Sections are needed to verify heights, levels, and vertical scope.",
            missing_dependency=["section_drawings"],
            impact_cost=RiskLevel.MEDIUM,
            impact_schedule=RiskLevel.LOW,
            bid_impact=BidImpact.CLARIFICATION_NEEDED,
            evidence=EvidenceRef(
                pages=floor_plan_pages[:5],
                detected_entities={
                    "floor_plan_count": graph.sheet_types_found.get(SheetType.FLOOR_PLAN.value, 0),
                    "section_count": 0,
                },
                search_attempts={
                    "sheet_type": "SheetType.SECTION",
                    "keywords": ["section", "sectional", "cross section"],
                },
                confidence=0.85,
                confidence_reason="No sheets classified as sections"
            ),
            fix_actions=[
                "Provide section drawings",
                "Or confirm if not applicable to project scope",
            ],
            score_delta_estimate=5,
            unlocks_boq_categories=["rcc_columns", "brick_masonry", "internal_plaster"],
            issue_type="missing_drawing",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class NoElevationsRule(DependencyRule):
    """Floor plans without elevations."""

    def __init__(self):
        super().__init__(
            "NO_ELEVATIONS",
            "Floor plans found but no elevation drawings"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        has_floor_plans = SheetType.FLOOR_PLAN.value in graph.sheet_types_found
        has_elevations = SheetType.ELEVATION.value in graph.sheet_types_found

        if not has_floor_plans:
            return None

        if has_elevations:
            return None

        # Elevations detected at page_index time (all pages) — no coverage gate needed
        return Blocker(
            id=create_blocker_id(1),
            title="No elevations found in drawing set",
            trade=Trade.ARCHITECTURAL,
            severity=Severity.MEDIUM,
            description=f"Drawing set has {graph.total_pages} pages but no elevations detected. "
                       f"Elevations show external finishes and facade elements.",
            missing_dependency=["elevation_drawings"],
            impact_cost=RiskLevel.MEDIUM,
            impact_schedule=RiskLevel.LOW,
            bid_impact=BidImpact.CLARIFICATION_NEEDED,
            evidence=EvidenceRef(
                detected_entities={
                    "floor_plan_count": graph.sheet_types_found.get(SheetType.FLOOR_PLAN.value, 0),
                    "elevation_count": 0,
                },
                search_attempts={
                    "sheet_type": "SheetType.ELEVATION",
                    "keywords": ["elevation", "facade", "front elevation"],
                },
                confidence=0.8,
                confidence_reason="No sheets classified as elevations"
            ),
            fix_actions=[
                "Provide elevation drawings",
                "Or confirm if not applicable to project scope",
            ],
            score_delta_estimate=3,
            unlocks_boq_categories=["external_plaster", "texture_coating"],
            issue_type="missing_drawing",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class NoMEPDrawingsRule(DependencyRule):
    """No MEP discipline drawings."""

    def __init__(self):
        super().__init__(
            "NO_MEP_DRAWINGS",
            "No MEP drawings found in drawing set"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        mep_disciplines = {
            Discipline.MECHANICAL.value,
            Discipline.ELECTRICAL.value,
            Discipline.PLUMBING.value,
        }

        found_mep = any(d in graph.disciplines_found for d in mep_disciplines)

        if found_mep:
            return None

        # Also check sheet types
        mep_types = {
            SheetType.MEP.value,
            SheetType.ELECTRICAL.value,
            SheetType.PLUMBING.value,
        }
        found_mep_type = any(t in graph.sheet_types_found for t in mep_types)

        if found_mep_type:
            return None

        # Disciplines from page_index (all pages) — no coverage gate needed
        return Blocker(
            id=create_blocker_id(3),
            title="No MEP drawings found in drawing set",
            trade=Trade.MEP,
            severity=Severity.MEDIUM,
            description=f"Drawing set has {graph.total_pages} pages but no MEP (Mechanical, "
                       f"Electrical, Plumbing) drawings detected.",
            missing_dependency=["mep_drawings"],
            impact_cost=RiskLevel.HIGH,
            impact_schedule=RiskLevel.MEDIUM,
            bid_impact=BidImpact.CLARIFICATION_NEEDED,
            evidence=EvidenceRef(
                detected_entities={
                    "disciplines_found": graph.disciplines_found,
                    "total_pages": graph.total_pages,
                },
                search_attempts={
                    "disciplines": ["M", "E", "P"],
                    "sheet_types": ["MEP", "ELECTRICAL", "PLUMBING"],
                },
                confidence=0.85,
                confidence_reason="No MEP discipline sheets found"
            ),
            fix_actions=[
                "Provide MEP drawings",
                "Or confirm if MEP is out of scope for this tender",
            ],
            score_delta_estimate=12,
            unlocks_boq_categories=["electrical_wiring", "electrical_fixtures",
                                    "plumbing_supply", "plumbing_drainage", "sanitary_fixtures"],
            issue_type="missing_drawing",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


class NoStructuralDrawingsRule(DependencyRule):
    """No structural drawings."""

    def __init__(self):
        super().__init__(
            "NO_STRUCTURAL_DRAWINGS",
            "No structural drawings found"
        )

    def check(self, graph: PlanSetGraph, run_coverage: Optional['RunCoverage'] = None) -> Optional[Blocker]:
        has_structural = Discipline.STRUCTURAL.value in graph.disciplines_found
        has_structural_type = SheetType.STRUCTURAL.value in graph.sheet_types_found
        has_foundation = SheetType.FOUNDATION.value in graph.sheet_types_found

        if has_structural or has_structural_type or has_foundation:
            return None

        # Only flag if we have floor plans (i.e., it's a building project)
        if SheetType.FLOOR_PLAN.value not in graph.sheet_types_found:
            return None

        # Disciplines from page_index (all pages) — no coverage gate needed
        return Blocker(
            id=create_blocker_id(7),
            title="No structural drawings found in drawing set",
            trade=Trade.STRUCTURAL,
            severity=Severity.MEDIUM,
            description=f"Drawing set has {graph.total_pages} pages with floor plans but "
                       f"no structural drawings detected.",
            missing_dependency=["structural_drawings"],
            impact_cost=RiskLevel.HIGH,
            impact_schedule=RiskLevel.HIGH,
            bid_impact=BidImpact.CLARIFICATION_NEEDED,
            evidence=EvidenceRef(
                detected_entities={
                    "disciplines_found": graph.disciplines_found,
                },
                search_attempts={
                    "discipline": "S",
                    "sheet_types": ["STRUCTURAL", "FOUNDATION"],
                },
                confidence=0.8,
                confidence_reason="No structural discipline sheets found"
            ),
            fix_actions=[
                "Provide structural drawings",
                "Or confirm if structural is out of scope",
            ],
            score_delta_estimate=10,
            unlocks_boq_categories=["rcc_footings", "rcc_columns", "rcc_beams",
                                    "rcc_slabs", "reinforcement_steel"],
            issue_type="missing_drawing",
            coverage_status=CoverageStatus.NOT_FOUND_AFTER_SEARCH.value,
        )


# =============================================================================
# DEPENDENCY REASONER
# =============================================================================

class DependencyReasoner:
    """
    Applies all dependency rules to a PlanSetGraph.

    Returns list of Blockers and RFIs with evidence.
    """

    def __init__(self):
        self.rules: List[DependencyRule] = [
            DoorsNeedScheduleRule(),
            WindowsNeedScheduleRule(),
            FinishScheduleRule(),
            ScaleMissingRule(),
            NoSectionsRule(),
            NoElevationsRule(),
            NoMEPDrawingsRule(),
            NoStructuralDrawingsRule(),
        ]

    def reason(self, graph: PlanSetGraph, run_coverage: Optional[RunCoverage] = None) -> Tuple[List[Blocker], List[RFIItem]]:
        """
        Apply all rules and return blockers and RFIs.

        Args:
            graph: PlanSetGraph to analyze
            run_coverage: Optional RunCoverage for coverage-gated assertions

        Returns:
            Tuple of (blockers, rfis)
        """
        blockers = []
        rfis = []

        for rule in self.rules:
            blocker = rule.check(graph, run_coverage=run_coverage)
            if blocker:
                blockers.append(blocker)

                # Create corresponding RFI
                rfi = self._blocker_to_rfi(blocker)
                rfis.append(rfi)

        # Sort by severity
        blockers.sort(key=lambda b: {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }.get(b.severity, 4))

        return blockers, rfis

    def _blocker_to_rfi(self, blocker: Blocker) -> RFIItem:
        """Convert a blocker to an RFI."""
        return RFIItem(
            id=f"RFI-{blocker.id.split('-')[1]}",
            trade=blocker.trade,
            priority=blocker.severity,
            question=blocker.title,
            why_it_matters=blocker.description,
            evidence=blocker.evidence,
            suggested_resolution=blocker.fix_actions[0] if blocker.fix_actions else "",
            acceptable_alternatives=blocker.fix_actions[1:] if len(blocker.fix_actions) > 1 else [],
            related_blocker_id=blocker.id,
            issue_type=blocker.issue_type,
            coverage_status=blocker.coverage_status,
        )

    def compute_trade_coverage(self, graph: PlanSetGraph, blockers: List[Blocker]) -> List[TradeCoverage]:
        """
        Compute trade-level coverage based on REAL extracted data and blockers.

        Coverage is now computed from:
        1. What disciplines/sheet types were actually found
        2. What entities were detected (doors, windows, rooms)
        3. Which blockers affect each trade (primary trade + affected_trades)

        Args:
            graph: PlanSetGraph
            blockers: List of blockers from reasoning

        Returns:
            List of TradeCoverage
        """
        coverage = []

        # Map disciplines to trades
        discipline_to_trade = {
            'A': Trade.ARCHITECTURAL,
            'S': Trade.STRUCTURAL,
            'M': Trade.MEP,
            'E': Trade.MEP,
            'P': Trade.MEP,
            'C': Trade.CIVIL,
        }

        # Determine which trades have actual evidence in the drawing set
        trades_with_evidence = set()
        for disc in graph.disciplines_found:
            if disc in discipline_to_trade:
                trades_with_evidence.add(discipline_to_trade[disc])

        # Check sheet types for evidence
        sheet_type_to_trade = {
            'floor_plan': Trade.ARCHITECTURAL,
            'elevation': Trade.ARCHITECTURAL,
            'section': Trade.ARCHITECTURAL,
            'structural': Trade.STRUCTURAL,
            'foundation': Trade.STRUCTURAL,
            'mep': Trade.MEP,
            'electrical': Trade.MEP,
            'plumbing': Trade.MEP,
            'site_plan': Trade.CIVIL,
        }
        for sheet_type in graph.sheet_types_found.keys():
            if sheet_type in sheet_type_to_trade:
                trades_with_evidence.add(sheet_type_to_trade[sheet_type])

        # Check detected entities for evidence
        if graph.all_door_tags or graph.all_window_tags:
            trades_with_evidence.add(Trade.ARCHITECTURAL)
        if graph.all_room_names:
            trades_with_evidence.add(Trade.FINISHES)

        # Group blockers by ALL affected trades (primary + affected_trades)
        blockers_by_trade: Dict[Trade, List[Blocker]] = {}
        for blocker in blockers:
            # Get all trades affected by this blocker
            affected = blocker.all_affected_trades if hasattr(blocker, 'all_affected_trades') else [blocker.trade]
            for trade in affected:
                if trade not in blockers_by_trade:
                    blockers_by_trade[trade] = []
                blockers_by_trade[trade].append(blocker)

        for trade in Trade:
            template_categories = BOQ_SKELETON_TEMPLATE.get(trade.value, [])
            if not template_categories:
                continue

            trade_blockers = blockers_by_trade.get(trade, [])
            has_evidence = trade in trades_with_evidence

            # Compute blocked categories
            blocked_categories = set()
            for blocker in trade_blockers:
                # Only count as blocked if it's a pricing blocker (not just clarification)
                if blocker.bid_impact == BidImpact.BLOCKS_PRICING:
                    blocked_categories.update(blocker.unlocks_boq_categories)

            total = len(template_categories)

            # NEW: Coverage based on REAL evidence, not just absence of blockers
            if not has_evidence:
                # No drawings for this trade - coverage is 0 but NOT necessarily blocked
                coverage_pct = 0.0
                priceable = 0
                blocked = 0
                assumed = total  # Would need to assume everything
                cost_risk = RiskLevel.UNKNOWN
                schedule_risk = RiskLevel.UNKNOWN
                next_action = "No drawings found for this trade"
            else:
                # Have evidence - calculate based on blockers
                blocked = len(blocked_categories.intersection(set(template_categories)))
                # Handle 'all_measured_items' special case
                if 'all_measured_items' in blocked_categories:
                    # This affects ALL categories but doesn't make them completely blocked
                    # Just reduces confidence/adds risk
                    blocked = min(blocked, total // 3)  # At most 1/3 blocked for scale issues
                    assumed = total // 3  # Another 1/3 assumed
                else:
                    assumed = 0

                priceable = total - blocked - assumed
                coverage_pct = (priceable / total * 100) if total > 0 else 0

                # Determine risk levels based on blockers
                high_blockers = sum(1 for b in trade_blockers
                                   if b.severity in [Severity.CRITICAL, Severity.HIGH])
                cost_risk = RiskLevel.HIGH if high_blockers >= 2 else (
                    RiskLevel.MEDIUM if high_blockers >= 1 else RiskLevel.LOW
                )
                schedule_risk = RiskLevel.HIGH if any(
                    b.impact_schedule == RiskLevel.HIGH for b in trade_blockers
                ) else RiskLevel.MEDIUM if trade_blockers else RiskLevel.LOW

                # Next action
                if high_blockers > 0:
                    next_action = f"Resolve {high_blockers} RFI(s) before pricing"
                elif trade_blockers:
                    next_action = f"Address {len(trade_blockers)} clarification(s)"
                else:
                    next_action = "Ready for pricing"

            coverage.append(TradeCoverage(
                trade=trade,
                coverage_pct=coverage_pct,
                total_categories=total,
                priceable_count=priceable,
                blocked_count=blocked,
                assumed_count=assumed,
                missing_dependencies=[d for b in trade_blockers for d in b.missing_dependency],
                cost_risk=cost_risk,
                schedule_risk=schedule_risk,
                next_action=next_action,
            ))

        return coverage

    def build_boq_skeleton(self, graph: PlanSetGraph, blockers: List[Blocker]) -> List[BOQSkeletonItem]:
        """
        Build BOQ skeleton with status per item.

        Args:
            graph: PlanSetGraph
            blockers: List of blockers

        Returns:
            List of BOQSkeletonItem
        """
        skeleton = []

        # Create mapping from category to blockers
        category_blockers: Dict[str, List[str]] = {}
        for blocker in blockers:
            for cat in blocker.unlocks_boq_categories:
                if cat not in category_blockers:
                    category_blockers[cat] = []
                category_blockers[cat].append(blocker.id)

        for trade_value, categories in BOQ_SKELETON_TEMPLATE.items():
            trade = Trade(trade_value)

            for category in categories:
                blocker_ids = category_blockers.get(category, [])

                if blocker_ids:
                    status = BOQItemStatus.BLOCKED
                else:
                    status = BOQItemStatus.PRICEABLE

                # Create evidence based on what we know
                evidence = EvidenceRef(
                    confidence=0.7 if status == BOQItemStatus.PRICEABLE else 0.5,
                    confidence_reason="Inferred from plan set analysis"
                )

                skeleton.append(BOQSkeletonItem(
                    trade=trade,
                    category=category,
                    item_name=category.replace('_', ' ').title(),
                    status=status,
                    blocked_by=blocker_ids,
                    evidence=evidence,
                    confidence=evidence.confidence,
                ))

        return skeleton


# =============================================================================
# ENTRY POINT
# =============================================================================

def reason_dependencies(graph: PlanSetGraph, run_coverage: Optional[RunCoverage] = None) -> Tuple[List[Blocker], List[RFIItem], List[TradeCoverage], List[BOQSkeletonItem]]:
    """
    Run dependency reasoning on a plan set graph.

    Args:
        graph: PlanSetGraph to analyze
        run_coverage: Optional RunCoverage for coverage-gated assertions

    Returns:
        Tuple of (blockers, rfis, trade_coverage, boq_skeleton)
    """
    reasoner = DependencyReasoner()
    blockers, rfis = reasoner.reason(graph, run_coverage=run_coverage)
    trade_coverage = reasoner.compute_trade_coverage(graph, blockers)
    boq_skeleton = reasoner.build_boq_skeleton(graph, blockers)

    return blockers, rfis, trade_coverage, boq_skeleton


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from src.analysis.plan_graph import build_plan_graph

    # Test with sample texts that should trigger blockers
    sample_texts = [
        """
        SHEET A-101
        GROUND FLOOR PLAN
        SCALE 1:100

        LIVING ROOM
        KITCHEN
        BEDROOM 1

        D1 D2 D3
        W1 W2
        """,
        """
        SHEET A-102
        FIRST FLOOR PLAN
        No scale shown

        BEDROOM 2
        BEDROOM 3
        BATHROOM

        D4 D5
        W3 W4
        """,
    ]

    # Build graph
    graph = build_plan_graph("test_project", sample_texts)

    # Run reasoning
    blockers, rfis, coverage, skeleton = reason_dependencies(graph)

    print("=== BLOCKERS ===")
    for b in blockers:
        print(f"{b.id}: {b.title} [{b.severity.value}]")
        print(f"  Evidence: {b.evidence.summary()}")
        print()

    print("\n=== TRADE COVERAGE ===")
    for c in coverage:
        print(f"{c.trade.value}: {c.coverage_pct:.0f}% - {c.next_action}")
