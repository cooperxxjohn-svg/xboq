"""
RFI Signal Collector for XBOQ.

Collects all input signals from various sources to drive RFI generation:
- Scope register and completeness report
- Coverage report and missing inputs
- Triangulation outputs (agreement scores, discrepancies)
- Overrides (conflicts found)
- Evidence from notes/specs/legends
- Cross-references and takeoff outputs

India-specific construction estimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ScopeSignal:
    """Signal from scope completeness analysis."""
    package: str
    subpackage: str
    status: str  # DETECTED, IMPLIED, UNKNOWN, MISSING_INPUT
    evidence_count: int
    confidence: float
    source_pages: List[str] = field(default_factory=list)


@dataclass
class CoverageSignal:
    """Signal from coverage analysis."""
    item_type: str  # missing_schedule, missing_drawing, low_coverage
    item_name: str
    severity: str  # high, medium, low
    suggestion: str
    source: str


@dataclass
class TriangulationSignal:
    """Signal from triangulation analysis."""
    quantity_name: str
    agreement_level: str  # EXCELLENT, GOOD, FAIR, POOR, DISCREPANCY
    variance_pct: float
    methods: List[Dict]
    discrepancy_notes: List[str]


@dataclass
class OverrideSignal:
    """Signal from cross-sheet overrides."""
    override_type: str
    target: str
    original_value: Any
    override_value: Any
    source: str
    is_conflict: bool


@dataclass
class EvidenceSignal:
    """Signal from evidence extraction."""
    evidence_type: str  # note, spec, legend, detail_ref, code_ref
    content: str
    page_id: str
    confidence: float
    related_package: Optional[str] = None


@dataclass
class ReferenceSignal:
    """Signal from cross-reference detection."""
    reference_text: str
    reference_type: str  # detail, section, schedule
    target_sheet: str
    exists: bool
    source_page: str


@dataclass
class TakeoffSignal:
    """Signal from takeoff outputs."""
    item_type: str  # opening, finish, wall, etc.
    count: int
    has_schedule: bool
    tags_found: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalCollection:
    """Complete collection of all signals for RFI generation."""
    project_id: str
    scope_signals: List[ScopeSignal] = field(default_factory=list)
    coverage_signals: List[CoverageSignal] = field(default_factory=list)
    triangulation_signals: List[TriangulationSignal] = field(default_factory=list)
    override_signals: List[OverrideSignal] = field(default_factory=list)
    evidence_signals: List[EvidenceSignal] = field(default_factory=list)
    reference_signals: List[ReferenceSignal] = field(default_factory=list)
    takeoff_signals: List[TakeoffSignal] = field(default_factory=list)

    # Quick lookup indexes
    missing_schedules: Set[str] = field(default_factory=set)
    unknown_scope_packages: Set[str] = field(default_factory=set)
    detected_tags: Dict[str, List[str]] = field(default_factory=dict)  # tag_type -> list of tags
    page_index: Dict[str, str] = field(default_factory=dict)  # page_id -> file_path


class SignalCollector:
    """
    Collects all input signals for RFI generation.

    Aggregates data from:
    - Scope completeness engine
    - Coverage analysis
    - Triangulation engine
    - Override engine
    - Evidence extraction
    - Cross-reference detection
    - Takeoff outputs
    """

    def __init__(self):
        self.signals = None

    def collect_all_signals(
        self,
        project_id: str,
        scope_register: Dict,
        completeness_report: Dict,
        coverage_report: Dict,
        missing_inputs: List[Dict],
        triangulation_report: Dict,
        override_report: Dict,
        evidence_data: List[Dict],
        extraction_results: List[Dict],
        project_graph: Dict,
        page_index: List[Dict],
    ) -> SignalCollection:
        """
        Collect all signals from various sources.

        Args:
            project_id: Project identifier
            scope_register: Scope register data
            completeness_report: Completeness scoring report
            coverage_report: Coverage analysis report
            missing_inputs: Missing inputs list
            triangulation_report: Triangulation results
            override_report: Override results
            evidence_data: Extracted evidence
            extraction_results: Page extraction results
            project_graph: Joined project graph
            page_index: Page index data

        Returns:
            Complete signal collection
        """
        self.signals = SignalCollection(project_id=project_id)

        # Build page index for reference
        self._build_page_index(page_index)

        # Collect from each source
        self._collect_scope_signals(scope_register, completeness_report)
        self._collect_coverage_signals(coverage_report, missing_inputs)
        self._collect_triangulation_signals(triangulation_report)
        self._collect_override_signals(override_report)
        self._collect_evidence_signals(evidence_data)
        self._collect_reference_signals(extraction_results)
        self._collect_takeoff_signals(extraction_results, project_graph)

        # Build quick lookup indexes
        self._build_indexes()

        return self.signals

    def _build_page_index(self, page_index: List[Dict]) -> None:
        """Build page ID to file path mapping."""
        for page in page_index:
            page_id = page.get("page_id", "")
            file_path = page.get("file_path", "")
            if page_id and file_path:
                self.signals.page_index[page_id] = file_path

    def _collect_scope_signals(
        self,
        scope_register: Dict,
        completeness_report: Dict,
    ) -> None:
        """Collect signals from scope analysis."""

        # From scope register items
        for item in scope_register.get("items", []):
            signal = ScopeSignal(
                package=item.get("package", ""),
                subpackage=item.get("subpackage", ""),
                status=item.get("status", "UNKNOWN"),
                evidence_count=item.get("evidence_count", 0),
                confidence=item.get("confidence", 0),
                source_pages=item.get("source_pages", []),
            )
            self.signals.scope_signals.append(signal)

            # Track unknown packages
            if signal.status in ["UNKNOWN", "MISSING_INPUT"]:
                self.signals.unknown_scope_packages.add(signal.package)

        # From completeness report - package scores
        for pkg_score in completeness_report.get("package_scores", []):
            # Low-scoring packages need RFIs
            if pkg_score.get("score", 0) < 30:
                package = pkg_score.get("package", "")
                if package:
                    self.signals.unknown_scope_packages.add(package)

    def _collect_coverage_signals(
        self,
        coverage_report: Dict,
        missing_inputs: List[Dict],
    ) -> None:
        """Collect signals from coverage analysis."""

        # From coverage report
        schedules = coverage_report.get("schedules", {})
        if not schedules.get("door_schedule"):
            self.signals.coverage_signals.append(CoverageSignal(
                item_type="missing_schedule",
                item_name="door_schedule",
                severity="high",
                suggestion="Provide door schedule",
                source="coverage_report",
            ))
            self.signals.missing_schedules.add("door")

        if not schedules.get("window_schedule"):
            self.signals.coverage_signals.append(CoverageSignal(
                item_type="missing_schedule",
                item_name="window_schedule",
                severity="high",
                suggestion="Provide window schedule",
                source="coverage_report",
            ))
            self.signals.missing_schedules.add("window")

        if not schedules.get("finish_schedule"):
            self.signals.coverage_signals.append(CoverageSignal(
                item_type="missing_schedule",
                item_name="finish_schedule",
                severity="medium",
                suggestion="Provide finish schedule",
                source="coverage_report",
            ))
            self.signals.missing_schedules.add("finish")

        # Room coverage
        rooms = coverage_report.get("rooms", {})
        if rooms.get("label_coverage_pct", 100) < 50:
            self.signals.coverage_signals.append(CoverageSignal(
                item_type="low_coverage",
                item_name="room_labels",
                severity="medium",
                suggestion="Many rooms lack labels",
                source="coverage_report",
            ))

        # From missing inputs list
        for item in missing_inputs:
            self.signals.coverage_signals.append(CoverageSignal(
                item_type=item.get("type", "missing_input"),
                item_name=item.get("item", ""),
                severity=item.get("severity", "medium"),
                suggestion=item.get("suggestion", ""),
                source="missing_inputs",
            ))

    def _collect_triangulation_signals(
        self,
        triangulation_report: Dict,
    ) -> None:
        """Collect signals from triangulation analysis."""

        for result in triangulation_report.get("results", []):
            signal = TriangulationSignal(
                quantity_name=result.get("quantity", ""),
                agreement_level=result.get("agreement_level", "FAIR"),
                variance_pct=result.get("variance_pct", 0),
                methods=result.get("methods", []),
                discrepancy_notes=result.get("discrepancy_notes", []),
            )
            self.signals.triangulation_signals.append(signal)

    def _collect_override_signals(
        self,
        override_report: Dict,
    ) -> None:
        """Collect signals from override analysis."""

        for override in override_report.get("overrides", []):
            # Detect if this is a conflict (significant difference)
            is_conflict = self._is_significant_conflict(
                override.get("original"),
                override.get("override"),
            )

            signal = OverrideSignal(
                override_type=override.get("type", ""),
                target=override.get("target", ""),
                original_value=override.get("original", ""),
                override_value=override.get("override", ""),
                source=override.get("reference", ""),
                is_conflict=is_conflict,
            )
            self.signals.override_signals.append(signal)

    def _collect_evidence_signals(
        self,
        evidence_data: List[Dict],
    ) -> None:
        """Collect signals from evidence extraction."""

        # Handle both list of evidence items and dict with "items" key
        items = evidence_data
        if isinstance(evidence_data, dict):
            items = evidence_data.get("items", [])

        for evidence in items:
            if not isinstance(evidence, dict):
                continue

            signal = EvidenceSignal(
                evidence_type=evidence.get("evidence_type", evidence.get("type", "note")),
                content=evidence.get("snippet", evidence.get("content", evidence.get("text", ""))),
                page_id=evidence.get("page_id", str(evidence.get("source_file", ""))),
                confidence=evidence.get("confidence", 0.5),
                related_package=evidence.get("package", None),
            )
            self.signals.evidence_signals.append(signal)

    def _collect_reference_signals(
        self,
        extraction_results: List[Dict],
    ) -> None:
        """Collect cross-reference signals from extraction results."""
        import re

        # Reference patterns to detect
        detail_pattern = re.compile(
            r"(?:Detail|DET|det)[.\s-]*(\d+)[/\\](\w+-?\d+)",
            re.IGNORECASE
        )
        section_pattern = re.compile(
            r"(?:Section|SEC|sec)[.\s-]*([A-Z])[/\\-]?([A-Z])?[/\\]?(\w+-?\d+)?",
            re.IGNORECASE
        )
        schedule_ref_pattern = re.compile(
            r"(?:See|Refer)\s+(?:to\s+)?(\w+)\s+schedule",
            re.IGNORECASE
        )

        # Collect all sheet IDs present
        present_sheets = set()
        for result in extraction_results:
            file_stem = Path(result.get("file_path", "")).stem
            present_sheets.add(file_stem.upper())
            # Also add common variations
            present_sheets.add(file_stem.lower())

            # Check for title block sheet number
            title_block = result.get("title_block", {})
            sheet_no = title_block.get("sheet_number", "")
            if sheet_no:
                present_sheets.add(sheet_no.upper())

        # Scan all text items for references
        for result in extraction_results:
            page_id = f"{Path(result.get('file_path', '')).stem}_p{result.get('page_number', 0) + 1}"

            # Check text items
            for text_item in result.get("text_items", []):
                text = text_item.get("text", "")

                # Detail references
                for match in detail_pattern.finditer(text):
                    detail_id = match.group(1)
                    sheet_id = match.group(2)
                    exists = sheet_id.upper() in present_sheets

                    self.signals.reference_signals.append(ReferenceSignal(
                        reference_text=match.group(0),
                        reference_type="detail",
                        target_sheet=sheet_id,
                        exists=exists,
                        source_page=page_id,
                    ))

                # Section references
                for match in section_pattern.finditer(text):
                    section_id = match.group(1)
                    sheet_id = match.group(3) if match.group(3) else ""
                    exists = not sheet_id or sheet_id.upper() in present_sheets

                    self.signals.reference_signals.append(ReferenceSignal(
                        reference_text=match.group(0),
                        reference_type="section",
                        target_sheet=sheet_id,
                        exists=exists,
                        source_page=page_id,
                    ))

                # Schedule references
                for match in schedule_ref_pattern.finditer(text):
                    schedule_type = match.group(1).lower()
                    # Check if we have that schedule
                    exists = schedule_type not in self.signals.missing_schedules

                    self.signals.reference_signals.append(ReferenceSignal(
                        reference_text=match.group(0),
                        reference_type="schedule",
                        target_sheet=f"{schedule_type}_schedule",
                        exists=exists,
                        source_page=page_id,
                    ))

    def _collect_takeoff_signals(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
    ) -> None:
        """Collect takeoff-related signals."""

        # Collect door tags
        door_tags = []
        for result in extraction_results:
            if result.get("page_type") == "floor_plan":
                for door in result.get("doors", []):
                    tag = door.get("tag", door.get("mark", ""))
                    if tag:
                        door_tags.append(tag)

        if door_tags:
            self.signals.takeoff_signals.append(TakeoffSignal(
                item_type="door",
                count=len(door_tags),
                has_schedule="door" not in self.signals.missing_schedules,
                tags_found=list(set(door_tags)),
            ))
            self.signals.detected_tags["door"] = list(set(door_tags))

        # Collect window tags
        window_tags = []
        for result in extraction_results:
            if result.get("page_type") == "floor_plan":
                for window in result.get("windows", []):
                    tag = window.get("tag", window.get("mark", ""))
                    if tag:
                        window_tags.append(tag)

        if window_tags:
            self.signals.takeoff_signals.append(TakeoffSignal(
                item_type="window",
                count=len(window_tags),
                has_schedule="window" not in self.signals.missing_schedules,
                tags_found=list(set(window_tags)),
            ))
            self.signals.detected_tags["window"] = list(set(window_tags))

        # Room types for finish signals
        room_types = {}
        for result in extraction_results:
            if result.get("page_type") == "floor_plan":
                for room in result.get("rooms", []):
                    label = room.get("label", "unlabeled").lower()
                    if label not in room_types:
                        room_types[label] = 0
                    room_types[label] += 1

        self.signals.takeoff_signals.append(TakeoffSignal(
            item_type="room",
            count=sum(room_types.values()),
            has_schedule="finish" not in self.signals.missing_schedules,
            tags_found=list(room_types.keys()),
            details={"by_type": room_types},
        ))

    def _build_indexes(self) -> None:
        """Build quick lookup indexes."""
        # Already built during collection
        pass

    def _is_significant_conflict(self, original: Any, override: Any) -> bool:
        """Determine if override represents a significant conflict."""
        if original is None or override is None:
            return False

        # If both are numbers, check percentage difference
        try:
            if isinstance(original, dict) and isinstance(override, dict):
                # Compare dict values
                for key in original:
                    if key in override:
                        orig_val = original[key]
                        ovr_val = override[key]
                        if isinstance(orig_val, (int, float)) and isinstance(ovr_val, (int, float)):
                            if orig_val > 0:
                                diff_pct = abs(ovr_val - orig_val) / orig_val * 100
                                if diff_pct > 10:
                                    return True
            elif isinstance(original, (int, float)) and isinstance(override, (int, float)):
                if original > 0:
                    diff_pct = abs(override - original) / original * 100
                    return diff_pct > 10
        except Exception:
            pass

        # If different types or strings that differ
        return str(original) != str(override)


def collect_signals(
    project_id: str,
    scope_register: Dict,
    completeness_report: Dict,
    coverage_report: Dict,
    missing_inputs: List[Dict],
    triangulation_report: Dict,
    override_report: Dict,
    evidence_data: List[Dict],
    extraction_results: List[Dict],
    project_graph: Dict,
    page_index: List[Dict],
) -> SignalCollection:
    """
    Convenience function to collect all signals.
    """
    collector = SignalCollector()
    return collector.collect_all_signals(
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
