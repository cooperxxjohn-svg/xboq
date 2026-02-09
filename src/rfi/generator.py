"""
RFI Generator for XBOQ.

Generates estimator-grade RFIs based on:
- Collected signals from all analysis phases
- Detected conflicts
- Missing references
- India-specific RFI rules

All RFIs are grounded in evidence from the drawing set.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from enum import Enum
import logging
import yaml
from datetime import datetime

from .signals import SignalCollection, ScopeSignal, CoverageSignal
from .conflicts import ConflictReport, Conflict, ConflictType
from .references import ReferenceReport, MissingReference

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """RFI issue types."""
    MISSING_INPUT = "missing_input"
    CONFLICT = "conflict"
    UNCLEAR_SPEC = "unclear_spec"
    MISSING_DIMENSION = "missing_dimension"
    MISSING_SCHEDULE = "missing_schedule"
    MISSING_DETAIL_REFERENCE = "missing_detail_reference"


class Priority(Enum):
    """RFI priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RFI:
    """Single RFI item."""
    rfi_id: str
    priority: Priority
    package: str
    issue_type: IssueType
    question: str
    why_needed: str
    impacted_boq_items: List[str] = field(default_factory=list)
    evidence_pages: List[str] = field(default_factory=list)
    evidence_snippets: List[str] = field(default_factory=list)
    missing_info: str = ""
    suggested_resolution: str = ""
    confidence: float = 0.8


@dataclass
class RFIReport:
    """Complete RFI report for a project."""
    project_id: str
    generated: str = ""
    rfis: List[RFI] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class RFIGenerator:
    """
    Generates RFIs based on all collected signals and analysis results.

    RFIs are grounded in evidence and use India-specific terminology.
    """

    # Package priority weights for scoring
    PACKAGE_WEIGHTS = {
        "rcc_structural": 10,
        "waterproofing": 9,
        "external_works": 8,
        "plumbing": 8,
        "electrical": 8,
        "levels_dimensions": 10,
        "doors_windows": 6,
        "tiles_finishes": 5,
        "painting": 4,
        "masonry": 7,
    }

    def __init__(self, rules_path: Optional[Path] = None):
        """Initialize with rules file."""
        if rules_path is None:
            rules_path = Path(__file__).parent.parent.parent / "rules" / "rfi_rules.yaml"

        self.rules = self._load_rules(rules_path)
        self.rfis: List[RFI] = []
        self._rfi_counter = 0

    def _load_rules(self, rules_path: Path) -> Dict:
        """Load RFI rules from YAML."""
        try:
            if rules_path.exists():
                with open(rules_path) as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load RFI rules: {e}")

        return {"packages": {}, "conflict_rules": {}}

    def generate_rfis(
        self,
        project_id: str,
        signals: SignalCollection,
        conflict_report: ConflictReport,
        reference_report: ReferenceReport,
    ) -> RFIReport:
        """
        Generate all RFIs for the project.

        Args:
            project_id: Project identifier
            signals: Collected signals from all sources
            conflict_report: Conflict detection results
            reference_report: Reference analysis results

        Returns:
            Complete RFI report
        """
        self.rfis = []
        self._rfi_counter = 0

        # Generate RFIs from different sources
        self._generate_from_scope_gaps(signals)
        self._generate_from_coverage_gaps(signals)
        self._generate_from_triangulation(signals)
        self._generate_from_conflicts(conflict_report)
        self._generate_from_missing_references(reference_report)
        self._generate_from_missing_schedules(signals)
        self._generate_from_package_rules(signals)

        # Deduplicate similar RFIs
        self._deduplicate_rfis()

        # Sort by priority
        self._sort_rfis()

        # Build report
        report = self._build_report(project_id)

        return report

    def _generate_rfi_id(self) -> str:
        """Generate unique RFI ID."""
        self._rfi_counter += 1
        return f"RFI-{self._rfi_counter:04d}"

    def _generate_from_scope_gaps(self, signals: SignalCollection) -> None:
        """Generate RFIs from scope register gaps."""

        # Group unknown/missing packages
        critical_packages = {
            "waterproofing": "Waterproofing specification critical for wet areas",
            "plumbing_sanitary_swr": "Plumbing scope affects MEP coordination",
            "electrical_power_lighting": "Electrical scope affects MEP coordination",
            "fire_hvac": "Fire/HVAC systems require early coordination",
            "external_works_drainage": "External drainage affects site development",
        }

        for signal in signals.scope_signals:
            if signal.status not in ["UNKNOWN", "MISSING_INPUT"]:
                continue

            package = signal.package
            subpackage = signal.subpackage

            # Determine priority based on package
            if package in critical_packages or signal.package in self.PACKAGE_WEIGHTS:
                priority = Priority.HIGH if self.PACKAGE_WEIGHTS.get(package, 5) >= 8 else Priority.MEDIUM
            else:
                priority = Priority.MEDIUM

            # Build question using rules if available
            package_rules = self.rules.get("packages", {}).get(package, {})
            ambiguities = package_rules.get("common_ambiguities", [])

            question = f"Is {subpackage.replace('_', ' ')} included in the project scope?"
            why_needed = f"No evidence found for {subpackage.replace('_', ' ')} in the drawing set. This item must be confirmed to avoid scope gaps."

            # Check for specific ambiguity templates
            for ambig in ambiguities:
                if ambig.get("id", "").lower() in subpackage.lower():
                    template = ambig.get("template", "")
                    if template:
                        question = template.split("\n")[0].strip()
                        break

            self.rfis.append(RFI(
                rfi_id=self._generate_rfi_id(),
                priority=priority,
                package=package,
                issue_type=IssueType.MISSING_INPUT,
                question=question,
                why_needed=why_needed,
                impacted_boq_items=[subpackage],
                evidence_pages=signal.source_pages if signal.source_pages else ["N/A - no evidence"],
                evidence_snippets=["No specification found in drawing set"],
                missing_info=f"{subpackage} specification/scope confirmation",
                suggested_resolution=f"Provide specification or confirm if {subpackage.replace('_', ' ')} is excluded from scope",
                confidence=0.85,
            ))

    def _generate_from_coverage_gaps(self, signals: SignalCollection) -> None:
        """Generate RFIs from coverage analysis gaps."""

        for signal in signals.coverage_signals:
            if signal.item_type == "missing_schedule":
                # Already handled separately
                continue

            priority = Priority.HIGH if signal.severity == "high" else Priority.MEDIUM

            self.rfis.append(RFI(
                rfi_id=self._generate_rfi_id(),
                priority=priority,
                package="general",
                issue_type=IssueType.MISSING_INPUT,
                question=f"Please provide {signal.item_name.replace('_', ' ')}",
                why_needed=signal.suggestion,
                impacted_boq_items=[signal.item_name],
                evidence_pages=[signal.source],
                evidence_snippets=[f"Coverage analysis identified: {signal.item_name}"],
                missing_info=signal.item_name,
                suggested_resolution=signal.suggestion,
                confidence=0.8,
            ))

    def _generate_from_triangulation(self, signals: SignalCollection) -> None:
        """Generate RFIs from triangulation discrepancies."""

        for signal in signals.triangulation_signals:
            if signal.agreement_level not in ["POOR", "DISCREPANCY"]:
                continue

            priority = Priority.HIGH if signal.agreement_level == "DISCREPANCY" else Priority.MEDIUM

            # Build method comparison
            method_details = []
            for method in signal.methods:
                method_details.append(f"{method.get('name', '')}: {method.get('value', '')} (conf: {method.get('confidence', 0)}%)")

            question = f"Please confirm correct value for {signal.quantity_name}. Multiple methods show variance of {signal.variance_pct}%."
            why_needed = f"Triangulation shows {signal.variance_pct}% variance between methods. Accurate {signal.quantity_name} is essential for BOQ."

            self.rfis.append(RFI(
                rfi_id=self._generate_rfi_id(),
                priority=priority,
                package="quantities",
                issue_type=IssueType.CONFLICT,
                question=question,
                why_needed=why_needed,
                impacted_boq_items=[signal.quantity_name.lower().replace(" ", "_")],
                evidence_pages=["triangulation_analysis"],
                evidence_snippets=method_details,
                missing_info=f"Confirmed {signal.quantity_name}",
                suggested_resolution="; ".join(signal.discrepancy_notes) if signal.discrepancy_notes else "Verify with authoritative source",
                confidence=0.75,
            ))

    def _generate_from_conflicts(self, conflict_report: ConflictReport) -> None:
        """Generate RFIs from detected conflicts."""

        for conflict in conflict_report.conflicts:
            priority = Priority.HIGH if conflict.severity == "high" else Priority.MEDIUM

            # Map conflict type to issue type
            issue_type_map = {
                ConflictType.SCALE_MISMATCH: IssueType.CONFLICT,
                ConflictType.WALL_THICKNESS: IssueType.CONFLICT,
                ConflictType.CEILING_HEIGHT: IssueType.CONFLICT,
                ConflictType.DOOR_SIZE: IssueType.CONFLICT,
                ConflictType.WINDOW_SIZE: IssueType.CONFLICT,
                ConflictType.FINISH_ASSIGNMENT: IssueType.CONFLICT,
                ConflictType.SCHEDULE_CONFLICT: IssueType.CONFLICT,
                ConflictType.SPEC_CONFLICT: IssueType.UNCLEAR_SPEC,
            }
            issue_type = issue_type_map.get(conflict.conflict_type, IssueType.CONFLICT)

            # Build evidence
            evidence_pages = list(set(s.page_id for s in conflict.sources))
            evidence_snippets = [f"{s.source_type}: {s.value} ({s.snippet})" for s in conflict.sources]

            # Map conflict type to package
            package_map = {
                ConflictType.SCALE_MISMATCH: "general",
                ConflictType.WALL_THICKNESS: "masonry",
                ConflictType.CEILING_HEIGHT: "levels_dimensions",
                ConflictType.DOOR_SIZE: "doors_windows",
                ConflictType.WINDOW_SIZE: "doors_windows",
                ConflictType.FINISH_ASSIGNMENT: "tiles_finishes",
                ConflictType.SCHEDULE_CONFLICT: "general",
                ConflictType.SPEC_CONFLICT: "general",
            }
            package = package_map.get(conflict.conflict_type, "general")

            self.rfis.append(RFI(
                rfi_id=self._generate_rfi_id(),
                priority=priority,
                package=package,
                issue_type=issue_type,
                question=conflict.description,
                why_needed="Conflicting information found in drawings. Clarification needed to avoid estimation errors.",
                impacted_boq_items=conflict.affected_items,
                evidence_pages=evidence_pages,
                evidence_snippets=evidence_snippets,
                missing_info="Confirmed value",
                suggested_resolution=conflict.resolution_suggestion,
                confidence=0.9,
            ))

    def _generate_from_missing_references(self, reference_report: ReferenceReport) -> None:
        """Generate RFIs from missing detail/section references."""

        for missing in reference_report.missing_references:
            ref = missing.reference

            self.rfis.append(RFI(
                rfi_id=self._generate_rfi_id(),
                priority=Priority.HIGH if missing.severity == "high" else Priority.MEDIUM,
                package="general",
                issue_type=IssueType.MISSING_DETAIL_REFERENCE,
                question=f"Referenced {ref.reference_type} not found: {ref.reference_text}",
                why_needed=missing.impact,
                impacted_boq_items=[ref.target_item],
                evidence_pages=[ref.source_page],
                evidence_snippets=[ref.source_context],
                missing_info=f"Sheet {ref.target_sheet} with {ref.target_item}",
                suggested_resolution=missing.suggestion,
                confidence=ref.confidence,
            ))

    def _generate_from_missing_schedules(self, signals: SignalCollection) -> None:
        """Generate RFIs for missing schedules with detected tags."""

        # Door schedule
        if "door" in signals.missing_schedules:
            door_tags = signals.detected_tags.get("door", [])
            if door_tags:
                self.rfis.append(RFI(
                    rfi_id=self._generate_rfi_id(),
                    priority=Priority.HIGH,
                    package="doors_windows",
                    issue_type=IssueType.MISSING_SCHEDULE,
                    question=f"Door schedule not provided but door tags detected: {', '.join(door_tags[:10])}{'...' if len(door_tags) > 10 else ''}",
                    why_needed="Door schedule is essential for accurate door quantities and specifications. Tags detected but sizes/types unknown.",
                    impacted_boq_items=["doors", "door_frames", "door_hardware"],
                    evidence_pages=["floor_plans"],
                    evidence_snippets=[f"Detected {len(door_tags)} door tags: {', '.join(door_tags[:5])}"],
                    missing_info="Door schedule with sizes, types, materials, and hardware",
                    suggested_resolution="Provide door schedule (typically A-### or DOOR SCHEDULE sheet) with columns for: Mark, Size (W x H), Type, Frame Material, Shutter Material, Hardware",
                    confidence=0.95,
                ))

        # Window schedule
        if "window" in signals.missing_schedules:
            window_tags = signals.detected_tags.get("window", [])
            if window_tags:
                self.rfis.append(RFI(
                    rfi_id=self._generate_rfi_id(),
                    priority=Priority.HIGH,
                    package="doors_windows",
                    issue_type=IssueType.MISSING_SCHEDULE,
                    question=f"Window schedule not provided but window tags detected: {', '.join(window_tags[:10])}{'...' if len(window_tags) > 10 else ''}",
                    why_needed="Window schedule is essential for accurate window quantities and specifications. Tags detected but sizes/types unknown.",
                    impacted_boq_items=["windows", "window_frames", "glass", "mosquito_mesh"],
                    evidence_pages=["floor_plans"],
                    evidence_snippets=[f"Detected {len(window_tags)} window tags: {', '.join(window_tags[:5])}"],
                    missing_info="Window schedule with sizes, types, materials, and glass specification",
                    suggested_resolution="Provide window schedule (typically A-### or WINDOW SCHEDULE sheet) with columns for: Mark, Size (W x H), Type (sliding/casement/fixed), Frame Material, Glass Type & Thickness",
                    confidence=0.95,
                ))

        # Finish schedule
        if "finish" in signals.missing_schedules:
            room_types = signals.detected_tags.get("room", [])
            self.rfis.append(RFI(
                rfi_id=self._generate_rfi_id(),
                priority=Priority.MEDIUM,
                package="tiles_finishes",
                issue_type=IssueType.MISSING_SCHEDULE,
                question="Finish schedule not provided for room finishes",
                why_needed="Finish schedule needed to determine flooring, wall finish, and ceiling treatment for each room type.",
                impacted_boq_items=["flooring", "wall_tiles", "painting", "false_ceiling"],
                evidence_pages=["floor_plans"],
                evidence_snippets=[f"Room types detected: {', '.join(room_types[:8])}"] if room_types else ["Multiple rooms detected"],
                missing_info="Finish schedule with room-wise floor, wall, ceiling, skirting, dado specifications",
                suggested_resolution="Provide finish schedule with columns for: Room Type, Floor Finish, Wall Finish, Ceiling, Skirting, Dado",
                confidence=0.85,
            ))

    def _generate_from_package_rules(self, signals: SignalCollection) -> None:
        """Generate RFIs based on package-specific rules."""

        package_rules = self.rules.get("packages", {})

        # Check each package with rules
        for package_name, rules in package_rules.items():
            # Check if package is in scope but has missing evidence
            package_in_scope = any(
                s.package == package_name for s in signals.scope_signals
            )

            if not package_in_scope:
                continue

            # Check required evidence
            required_evidence = rules.get("required_evidence", [])
            common_ambiguities = rules.get("common_ambiguities", [])

            # For each ambiguity, check if we need to raise RFI
            for ambig in common_ambiguities:
                trigger_patterns = ambig.get("trigger_patterns", [])
                required_if_room = ambig.get("required_if_room_type", [])

                # Check if any trigger patterns match evidence
                trigger_found = False
                for signal in signals.evidence_signals:
                    content = signal.content.lower()
                    if any(p.lower() in content for p in trigger_patterns):
                        trigger_found = True
                        break

                # Check room-based triggers
                for signal in signals.takeoff_signals:
                    if signal.item_type == "room":
                        for room_type in signal.tags_found:
                            if any(rt in room_type.lower() for rt in required_if_room):
                                trigger_found = True
                                break

                if trigger_found:
                    # Check if we have the required specification
                    missing_keywords = ambig.get("missing_keywords", [])
                    has_spec = False

                    if missing_keywords:
                        for signal in signals.evidence_signals:
                            content = signal.content.lower()
                            if any(kw.lower() in content for kw in missing_keywords):
                                has_spec = True
                                break

                    if not has_spec:
                        template = ambig.get("template", "")
                        question = template.split("\n")[0].strip() if template else ambig.get("description", "")

                        priority_weight = rules.get("priority_weight", 5)
                        priority = Priority.HIGH if priority_weight >= 8 else Priority.MEDIUM

                        self.rfis.append(RFI(
                            rfi_id=self._generate_rfi_id(),
                            priority=priority,
                            package=package_name,
                            issue_type=IssueType.UNCLEAR_SPEC,
                            question=question,
                            why_needed=ambig.get("description", "Specification needed for accurate estimation"),
                            impacted_boq_items=[ambig.get("id", package_name)],
                            evidence_pages=["general_notes", "specifications"],
                            evidence_snippets=[f"Trigger found but specification missing: {', '.join(trigger_patterns[:3])}"],
                            missing_info=ambig.get("description", "Detailed specification"),
                            suggested_resolution=template if template else "Please provide detailed specification",
                            confidence=0.8,
                        ))

    def _deduplicate_rfis(self) -> None:
        """Remove duplicate or very similar RFIs."""
        seen_keys = set()
        unique_rfis = []

        for rfi in self.rfis:
            # Create a key based on package + issue type + first impacted item
            key = f"{rfi.package}_{rfi.issue_type.value}_{rfi.impacted_boq_items[0] if rfi.impacted_boq_items else ''}"

            # Also check question similarity
            question_key = rfi.question[:50].lower()

            combined_key = f"{key}_{question_key}"

            if combined_key not in seen_keys:
                seen_keys.add(combined_key)
                unique_rfis.append(rfi)

        self.rfis = unique_rfis

    def _sort_rfis(self) -> None:
        """Sort RFIs by priority and package weight."""

        def sort_key(rfi: RFI) -> tuple:
            priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
            package_weight = self.PACKAGE_WEIGHTS.get(rfi.package, 5)
            return (priority_order[rfi.priority], -package_weight, rfi.rfi_id)

        self.rfis.sort(key=sort_key)

    def _build_report(self, project_id: str) -> RFIReport:
        """Build final RFI report."""

        report = RFIReport(
            project_id=project_id,
            generated=datetime.now().isoformat(),
        )
        report.rfis = self.rfis

        # Summary
        by_priority = {}
        by_package = {}
        by_issue_type = {}

        for rfi in self.rfis:
            p = rfi.priority.value
            pkg = rfi.package
            t = rfi.issue_type.value

            by_priority[p] = by_priority.get(p, 0) + 1
            by_package[pkg] = by_package.get(pkg, 0) + 1
            by_issue_type[t] = by_issue_type.get(t, 0) + 1

        report.summary = {
            "total_rfis": len(self.rfis),
            "by_priority": by_priority,
            "by_package": by_package,
            "by_issue_type": by_issue_type,
            "high_priority_count": by_priority.get("high", 0),
            "requires_immediate_attention": by_priority.get("high", 0) > 5,
        }

        return report


def generate_rfis(
    project_id: str,
    signals: SignalCollection,
    conflict_report: ConflictReport,
    reference_report: ReferenceReport,
    rules_path: Optional[Path] = None,
) -> RFIReport:
    """
    Convenience function to generate RFIs.
    """
    generator = RFIGenerator(rules_path)
    return generator.generate_rfis(
        project_id=project_id,
        signals=signals,
        conflict_report=conflict_report,
        reference_report=reference_report,
    )
