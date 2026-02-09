"""
Coverage Report - Compute coverage metrics for estimator-grade outputs.

Computes:
- Scale coverage: % floor plan pages with high-confidence scale
- Room coverage: avg rooms/page, % rooms labeled
- Opening coverage: count detected, % tagged, % matched to schedule
- Wall thickness: clusters, stability
- Schedule coverage: which schedules found
- CPWD mapping: % BOQ lines mapped

Outputs:
- coverage_report.json
- missing_inputs.md
- unmapped_tags.csv
"""

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ScaleCoverage:
    """Scale detection coverage metrics."""
    total_floor_plans: int = 0
    with_scale: int = 0
    high_confidence: int = 0  # confidence > 0.7
    methods_used: Dict[str, int] = field(default_factory=dict)
    coverage_pct: float = 0.0
    high_conf_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_floor_plans": self.total_floor_plans,
            "with_scale": self.with_scale,
            "high_confidence": self.high_confidence,
            "methods_used": self.methods_used,
            "coverage_pct": round(self.coverage_pct, 1),
            "high_conf_pct": round(self.high_conf_pct, 1),
        }


@dataclass
class RoomCoverage:
    """Room detection coverage metrics."""
    total_rooms: int = 0
    labeled_rooms: int = 0
    unlabeled_rooms: int = 0
    rooms_per_page: float = 0.0
    label_coverage_pct: float = 0.0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    pages_with_rooms: int = 0
    pages_without_rooms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rooms": self.total_rooms,
            "labeled_rooms": self.labeled_rooms,
            "unlabeled_rooms": self.unlabeled_rooms,
            "rooms_per_page": round(self.rooms_per_page, 2),
            "label_coverage_pct": round(self.label_coverage_pct, 1),
            "label_distribution": dict(sorted(
                self.label_distribution.items(),
                key=lambda x: -x[1]
            )[:20]),  # Top 20 labels
            "pages_with_rooms": self.pages_with_rooms,
            "pages_without_rooms": self.pages_without_rooms,
        }


@dataclass
class OpeningCoverage:
    """Opening (door/window) detection coverage."""
    doors_detected: int = 0
    windows_detected: int = 0
    doors_tagged: int = 0
    windows_tagged: int = 0
    doors_matched_to_schedule: int = 0
    windows_matched_to_schedule: int = 0
    doors_with_defaults: int = 0
    windows_with_defaults: int = 0
    door_tag_coverage_pct: float = 0.0
    window_tag_coverage_pct: float = 0.0
    door_schedule_match_pct: float = 0.0
    window_schedule_match_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doors_detected": self.doors_detected,
            "windows_detected": self.windows_detected,
            "doors_tagged": self.doors_tagged,
            "windows_tagged": self.windows_tagged,
            "doors_matched_to_schedule": self.doors_matched_to_schedule,
            "windows_matched_to_schedule": self.windows_matched_to_schedule,
            "doors_with_defaults": self.doors_with_defaults,
            "windows_with_defaults": self.windows_with_defaults,
            "door_tag_coverage_pct": round(self.door_tag_coverage_pct, 1),
            "window_tag_coverage_pct": round(self.window_tag_coverage_pct, 1),
            "door_schedule_match_pct": round(self.door_schedule_match_pct, 1),
            "window_schedule_match_pct": round(self.window_schedule_match_pct, 1),
        }


@dataclass
class WallCoverage:
    """Wall thickness coverage metrics."""
    thickness_clusters: List[float] = field(default_factory=list)
    cluster_counts: Dict[str, int] = field(default_factory=dict)
    pages_analyzed: int = 0
    consistent_across_pages: bool = False
    expected_thicknesses_found: List[float] = field(default_factory=list)
    unexpected_thicknesses: List[float] = field(default_factory=list)

    # Standard Indian wall thicknesses in mm
    STANDARD_THICKNESSES = [115, 150, 200, 230, 300, 450]
    TOLERANCE = 20  # mm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thickness_clusters": [round(t, 1) for t in self.thickness_clusters],
            "cluster_counts": self.cluster_counts,
            "pages_analyzed": self.pages_analyzed,
            "consistent_across_pages": self.consistent_across_pages,
            "expected_thicknesses_found": [round(t, 1) for t in self.expected_thicknesses_found],
            "unexpected_thicknesses": [round(t, 1) for t in self.unexpected_thicknesses],
        }


@dataclass
class ScheduleCoverage:
    """Schedule extraction coverage."""
    schedules_found: List[str] = field(default_factory=list)
    schedules_missing: List[str] = field(default_factory=list)
    schedule_entry_counts: Dict[str, int] = field(default_factory=dict)
    door_schedule: bool = False
    window_schedule: bool = False
    finish_schedule: bool = False
    structural_schedule: bool = False
    bbs_schedule: bool = False
    area_schedule: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedules_found": self.schedules_found,
            "schedules_missing": self.schedules_missing,
            "schedule_entry_counts": self.schedule_entry_counts,
            "door_schedule": self.door_schedule,
            "window_schedule": self.window_schedule,
            "finish_schedule": self.finish_schedule,
            "structural_schedule": self.structural_schedule,
            "bbs_schedule": self.bbs_schedule,
            "area_schedule": self.area_schedule,
        }


@dataclass
class CPWDMapping:
    """CPWD mapping coverage."""
    total_boq_lines: int = 0
    mapped_lines: int = 0
    unmapped_lines: int = 0
    mapping_pct: float = 0.0
    unmapped_items: List[Dict[str, str]] = field(default_factory=list)
    mapping_by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_boq_lines": self.total_boq_lines,
            "mapped_lines": self.mapped_lines,
            "unmapped_lines": self.unmapped_lines,
            "mapping_pct": round(self.mapping_pct, 1),
            "unmapped_items_count": len(self.unmapped_items),
            "mapping_by_category": self.mapping_by_category,
        }


@dataclass
class CoverageReport:
    """Complete coverage report."""
    project_id: str
    scale: ScaleCoverage = field(default_factory=ScaleCoverage)
    rooms: RoomCoverage = field(default_factory=RoomCoverage)
    openings: OpeningCoverage = field(default_factory=OpeningCoverage)
    walls: WallCoverage = field(default_factory=WallCoverage)
    schedules: ScheduleCoverage = field(default_factory=ScheduleCoverage)
    cpwd: CPWDMapping = field(default_factory=CPWDMapping)
    overall_score: float = 0.0
    grade: str = "F"  # A/B/C/D/F

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "scale": self.scale.to_dict(),
            "rooms": self.rooms.to_dict(),
            "openings": self.openings.to_dict(),
            "walls": self.walls.to_dict(),
            "schedules": self.schedules.to_dict(),
            "cpwd": self.cpwd.to_dict(),
        }

    def compute_overall_score(self) -> None:
        """Compute overall coverage score (0-100)."""
        scores = []
        weights = []

        # Scale coverage (weight: 20)
        if self.scale.total_floor_plans > 0:
            scores.append(self.scale.high_conf_pct)
            weights.append(20)

        # Room coverage (weight: 25)
        if self.rooms.total_rooms > 0:
            scores.append(self.rooms.label_coverage_pct)
            weights.append(25)

        # Opening coverage (weight: 15)
        door_score = (self.openings.door_tag_coverage_pct +
                     self.openings.door_schedule_match_pct) / 2
        window_score = (self.openings.window_tag_coverage_pct +
                       self.openings.window_schedule_match_pct) / 2
        if self.openings.doors_detected > 0 or self.openings.windows_detected > 0:
            opening_score = (door_score + window_score) / 2
            scores.append(opening_score)
            weights.append(15)

        # Schedule coverage (weight: 20)
        expected_schedules = ["door", "window"]
        found = sum(1 for s in expected_schedules if s in self.schedules.schedules_found)
        schedule_score = (found / len(expected_schedules)) * 100 if expected_schedules else 0
        scores.append(schedule_score)
        weights.append(20)

        # CPWD mapping (weight: 20)
        scores.append(self.cpwd.mapping_pct)
        weights.append(20)

        # Compute weighted average
        if weights:
            self.overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            self.overall_score = 0

        # Assign grade
        if self.overall_score >= 90:
            self.grade = "A"
        elif self.overall_score >= 75:
            self.grade = "B"
        elif self.overall_score >= 60:
            self.grade = "C"
        elif self.overall_score >= 40:
            self.grade = "D"
        else:
            self.grade = "F"


@dataclass
class MissingInput:
    """A missing input item."""
    category: str
    item: str
    reason: str
    suggestion: str
    priority: str  # high, medium, low

    def to_markdown(self) -> str:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(self.priority, "⚪")
        return f"- {icon} **{self.item}** ({self.category})\n  - Reason: {self.reason}\n  - Action: {self.suggestion}"


class CoverageAnalyzer:
    """
    Analyzes extraction results to compute coverage metrics.
    """

    # CPWD mapping file (if available)
    CPWD_MAPPING_PATH = Path("data/cpwd_mapping.csv")

    def __init__(self):
        self.cpwd_mapping: Dict[str, str] = {}
        self._load_cpwd_mapping()

    def _load_cpwd_mapping(self) -> None:
        """Load CPWD mapping if available."""
        if self.CPWD_MAPPING_PATH.exists():
            try:
                with open(self.CPWD_MAPPING_PATH) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row.get("item_code", "").lower()
                        cpwd = row.get("cpwd_code", "")
                        if key and cpwd:
                            self.cpwd_mapping[key] = cpwd
                logger.info(f"Loaded {len(self.cpwd_mapping)} CPWD mappings")
            except Exception as e:
                logger.warning(f"Could not load CPWD mapping: {e}")

    def analyze(
        self,
        project_id: str,
        extraction_results: List[Dict[str, Any]],
        project_graph: Dict[str, Any],
        boq_entries: List[Dict[str, Any]],
    ) -> CoverageReport:
        """
        Analyze extraction results and compute coverage.

        Args:
            project_id: Project identifier
            extraction_results: List of extraction result dicts
            project_graph: Project graph dict
            boq_entries: List of BOQ entry dicts

        Returns:
            CoverageReport with all metrics
        """
        report = CoverageReport(project_id=project_id)

        # Analyze each coverage area
        self._analyze_scale(report, extraction_results)
        self._analyze_rooms(report, extraction_results)
        self._analyze_openings(report, extraction_results, project_graph)
        self._analyze_walls(report, extraction_results)
        self._analyze_schedules(report, project_graph)
        self._analyze_cpwd(report, boq_entries)

        # Compute overall score
        report.compute_overall_score()

        return report

    def _analyze_scale(
        self,
        report: CoverageReport,
        results: List[Dict[str, Any]]
    ) -> None:
        """Analyze scale detection coverage."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]
        report.scale.total_floor_plans = len(floor_plans)

        methods = Counter()
        for fp in floor_plans:
            scale_info = fp.get("scale_info", {})
            if scale_info:
                report.scale.with_scale += 1
                method = scale_info.get("method", "unknown")
                methods[method] += 1

                conf = scale_info.get("confidence", 0)
                if conf > 0.7:
                    report.scale.high_confidence += 1

        report.scale.methods_used = dict(methods)

        if report.scale.total_floor_plans > 0:
            report.scale.coverage_pct = (
                report.scale.with_scale / report.scale.total_floor_plans * 100
            )
            report.scale.high_conf_pct = (
                report.scale.high_confidence / report.scale.total_floor_plans * 100
            )

    def _analyze_rooms(
        self,
        report: CoverageReport,
        results: List[Dict[str, Any]]
    ) -> None:
        """Analyze room detection coverage."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        label_counts = Counter()
        for fp in floor_plans:
            rooms = fp.get("rooms", [])
            if rooms:
                report.rooms.pages_with_rooms += 1
            else:
                report.rooms.pages_without_rooms += 1

            for room in rooms:
                report.rooms.total_rooms += 1
                label = room.get("label", "")

                # Check if labeled (not generic "Room N" or empty)
                if label and not label.lower().startswith("room ") and label.lower() != "room":
                    report.rooms.labeled_rooms += 1
                    label_counts[label] += 1
                else:
                    report.rooms.unlabeled_rooms += 1

        report.rooms.label_distribution = dict(label_counts)

        if floor_plans:
            report.rooms.rooms_per_page = report.rooms.total_rooms / len(floor_plans)

        if report.rooms.total_rooms > 0:
            report.rooms.label_coverage_pct = (
                report.rooms.labeled_rooms / report.rooms.total_rooms * 100
            )

    def _analyze_openings(
        self,
        report: CoverageReport,
        results: List[Dict[str, Any]],
        project_graph: Dict[str, Any]
    ) -> None:
        """Analyze opening (door/window) coverage."""
        # Get detected elements from project graph
        detected = project_graph.get("detected_elements", {})

        for page_key, elements in detected.items():
            for elem in elements:
                elem_type = elem.get("type", "")
                tag = elem.get("tag", "")

                if elem_type == "door":
                    report.openings.doors_detected += 1
                    if tag:
                        report.openings.doors_tagged += 1
                elif elem_type == "window":
                    report.openings.windows_detected += 1
                    if tag:
                        report.openings.windows_tagged += 1

        # Check schedule matches
        tag_matches = project_graph.get("tag_matches", [])
        for match in tag_matches:
            tag_type = match.get("type", "")
            if tag_type == "door":
                report.openings.doors_matched_to_schedule += 1
            elif tag_type == "window":
                report.openings.windows_matched_to_schedule += 1

        # Calculate percentages
        if report.openings.doors_detected > 0:
            report.openings.door_tag_coverage_pct = (
                report.openings.doors_tagged / report.openings.doors_detected * 100
            )
            report.openings.door_schedule_match_pct = (
                report.openings.doors_matched_to_schedule / report.openings.doors_detected * 100
            )

        if report.openings.windows_detected > 0:
            report.openings.window_tag_coverage_pct = (
                report.openings.windows_tagged / report.openings.windows_detected * 100
            )
            report.openings.window_schedule_match_pct = (
                report.openings.windows_matched_to_schedule / report.openings.windows_detected * 100
            )

    def _analyze_walls(
        self,
        report: CoverageReport,
        results: List[Dict[str, Any]]
    ) -> None:
        """Analyze wall thickness coverage."""
        all_thicknesses = []

        for result in results:
            if result.get("page_type") != "floor_plan":
                continue

            wall_info = result.get("wall_info", {})
            thicknesses = wall_info.get("thicknesses", [])
            if thicknesses:
                all_thicknesses.extend(thicknesses)
                report.walls.pages_analyzed += 1

        if not all_thicknesses:
            return

        # Cluster thicknesses
        thickness_counter = Counter()
        for t in all_thicknesses:
            # Round to nearest 5mm for clustering
            rounded = round(t / 5) * 5
            thickness_counter[rounded] += 1

        # Get main clusters (more than 5% occurrence)
        total = sum(thickness_counter.values())
        main_clusters = [
            t for t, count in thickness_counter.items()
            if count / total > 0.05
        ]

        report.walls.thickness_clusters = sorted(main_clusters)
        report.walls.cluster_counts = {
            f"{int(t)}mm": thickness_counter[t]
            for t in main_clusters
        }

        # Check against standard thicknesses
        for cluster in main_clusters:
            is_standard = any(
                abs(cluster - std) <= WallCoverage.TOLERANCE
                for std in WallCoverage.STANDARD_THICKNESSES
            )
            if is_standard:
                report.walls.expected_thicknesses_found.append(cluster)
            else:
                report.walls.unexpected_thicknesses.append(cluster)

        # Check consistency across pages
        # (simplified: if main clusters are consistent)
        report.walls.consistent_across_pages = len(main_clusters) <= 3

    def _analyze_schedules(
        self,
        report: CoverageReport,
        project_graph: Dict[str, Any]
    ) -> None:
        """Analyze schedule extraction coverage."""
        schedules = project_graph.get("schedules", {})

        expected = ["door", "window", "finish", "structural", "bbs", "area"]

        for stype in expected:
            entries = schedules.get(stype, [])
            if entries:
                report.schedules.schedules_found.append(stype)
                report.schedules.schedule_entry_counts[stype] = len(entries)

                # Set flags
                if stype == "door":
                    report.schedules.door_schedule = True
                elif stype == "window":
                    report.schedules.window_schedule = True
                elif stype == "finish":
                    report.schedules.finish_schedule = True
                elif stype == "structural":
                    report.schedules.structural_schedule = True
                elif stype == "bbs":
                    report.schedules.bbs_schedule = True
                elif stype == "area":
                    report.schedules.area_schedule = True
            else:
                report.schedules.schedules_missing.append(stype)

    def _analyze_cpwd(
        self,
        report: CoverageReport,
        boq_entries: List[Dict[str, Any]]
    ) -> None:
        """Analyze CPWD mapping coverage."""
        report.cpwd.total_boq_lines = len(boq_entries)

        category_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "mapped": 0}
        )

        for entry in boq_entries:
            category = entry.get("category", "Other")
            item_code = entry.get("item_code", "").lower()

            category_stats[category]["total"] += 1

            # Check if mapped to CPWD
            cpwd_code = entry.get("cpwd_code", "")
            if cpwd_code or item_code in self.cpwd_mapping:
                report.cpwd.mapped_lines += 1
                category_stats[category]["mapped"] += 1
            else:
                report.cpwd.unmapped_lines += 1
                report.cpwd.unmapped_items.append({
                    "category": category,
                    "item_code": entry.get("item_code", ""),
                    "description": entry.get("description", ""),
                })

        report.cpwd.mapping_by_category = {
            cat: {"total": stats["total"], "mapped": stats["mapped"]}
            for cat, stats in category_stats.items()
        }

        if report.cpwd.total_boq_lines > 0:
            report.cpwd.mapping_pct = (
                report.cpwd.mapped_lines / report.cpwd.total_boq_lines * 100
            )

    def generate_missing_inputs(
        self,
        report: CoverageReport
    ) -> List[MissingInput]:
        """
        Generate list of missing inputs based on coverage gaps.

        Returns:
            List of MissingInput items
        """
        missing = []

        # Scale issues
        if report.scale.total_floor_plans > 0:
            if report.scale.high_conf_pct < 50:
                missing.append(MissingInput(
                    category="Scale",
                    item="Scale reference",
                    reason=f"Only {report.scale.high_conf_pct:.0f}% of floor plans have high-confidence scale",
                    suggestion="Provide at least one known dimension (e.g., room width, door width) or scale bar",
                    priority="high"
                ))

        # Room labeling issues
        if report.rooms.total_rooms > 0 and report.rooms.label_coverage_pct < 70:
            missing.append(MissingInput(
                category="Room Labels",
                item="Room label annotations",
                reason=f"Only {report.rooms.label_coverage_pct:.0f}% of rooms are labeled",
                suggestion="Ensure floor plans include room labels (e.g., 'BEDROOM', 'KITCHEN')",
                priority="medium"
            ))

        # Schedule issues
        if not report.schedules.door_schedule:
            missing.append(MissingInput(
                category="Schedules",
                item="Door schedule",
                reason="No door schedule detected",
                suggestion="Provide door schedule sheet (typically A-### or DOOR SCHEDULE)",
                priority="high"
            ))

        if not report.schedules.window_schedule:
            missing.append(MissingInput(
                category="Schedules",
                item="Window schedule",
                reason="No window schedule detected",
                suggestion="Provide window schedule sheet (typically A-### or WINDOW SCHEDULE)",
                priority="high"
            ))

        if not report.schedules.finish_schedule:
            missing.append(MissingInput(
                category="Schedules",
                item="Finish schedule",
                reason="No finish schedule detected",
                suggestion="Provide finish schedule for accurate material estimation",
                priority="medium"
            ))

        # Opening matching issues
        if report.openings.doors_detected > 0 and report.openings.door_schedule_match_pct < 50:
            missing.append(MissingInput(
                category="Door Tags",
                item="Door tag annotations",
                reason=f"Only {report.openings.door_schedule_match_pct:.0f}% of doors matched to schedule",
                suggestion="Ensure door tags (D1, D2, etc.) are visible near door symbols",
                priority="medium"
            ))

        if report.openings.windows_detected > 0 and report.openings.window_schedule_match_pct < 50:
            missing.append(MissingInput(
                category="Window Tags",
                item="Window tag annotations",
                reason=f"Only {report.openings.window_schedule_match_pct:.0f}% of windows matched to schedule",
                suggestion="Ensure window tags (W1, W2, etc.) are visible near window symbols",
                priority="medium"
            ))

        # Wall thickness issues
        if report.walls.unexpected_thicknesses:
            unexpected = ", ".join(f"{int(t)}mm" for t in report.walls.unexpected_thicknesses)
            missing.append(MissingInput(
                category="Wall Thickness",
                item="Non-standard wall thicknesses",
                reason=f"Detected non-standard thicknesses: {unexpected}",
                suggestion="Verify wall specifications or provide wall thickness notes",
                priority="low"
            ))

        # CPWD mapping issues
        if report.cpwd.unmapped_lines > 0 and report.cpwd.mapping_pct < 80:
            missing.append(MissingInput(
                category="CPWD Mapping",
                item="Item code mapping",
                reason=f"{report.cpwd.unmapped_lines} BOQ items not mapped to CPWD codes",
                suggestion="Review unmapped_tags.csv and update cpwd_mapping.csv",
                priority="low"
            ))

        # Floor plan issues
        if report.rooms.pages_without_rooms > 0:
            missing.append(MissingInput(
                category="Floor Plans",
                item="Floor plan quality",
                reason=f"{report.rooms.pages_without_rooms} floor plan pages yielded no rooms",
                suggestion="Check image quality or provide cleaner floor plan drawings",
                priority="medium"
            ))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        missing.sort(key=lambda x: priority_order.get(x.priority, 3))

        return missing

    def export_coverage_report(
        self,
        report: CoverageReport,
        output_dir: Path
    ) -> Path:
        """Export coverage report to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "coverage_report.json"
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved coverage report to: {path}")
        return path

    def export_missing_inputs(
        self,
        report: CoverageReport,
        missing: List[MissingInput],
        output_dir: Path
    ) -> Path:
        """Export missing inputs checklist to Markdown."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "missing_inputs.md"

        with open(path, "w") as f:
            f.write(f"# Missing Inputs Checklist: {report.project_id}\n\n")
            f.write(f"**Coverage Score:** {report.overall_score:.0f}/100 (Grade: {report.grade})\n\n")
            f.write("---\n\n")

            if not missing:
                f.write("✅ **All required inputs appear to be present.**\n\n")
                f.write("No critical missing items detected.\n")
            else:
                f.write("## Required Actions\n\n")
                f.write("The following items would improve extraction quality:\n\n")

                # Group by priority
                high = [m for m in missing if m.priority == "high"]
                medium = [m for m in missing if m.priority == "medium"]
                low = [m for m in missing if m.priority == "low"]

                if high:
                    f.write("### 🔴 High Priority\n\n")
                    for item in high:
                        f.write(item.to_markdown() + "\n\n")

                if medium:
                    f.write("### 🟡 Medium Priority\n\n")
                    for item in medium:
                        f.write(item.to_markdown() + "\n\n")

                if low:
                    f.write("### 🟢 Low Priority\n\n")
                    for item in low:
                        f.write(item.to_markdown() + "\n\n")

            # Summary statistics
            f.write("---\n\n")
            f.write("## Coverage Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Scale Coverage | {report.scale.coverage_pct:.0f}% "
                   f"({report.scale.high_confidence}/{report.scale.total_floor_plans} high-conf) |\n")
            f.write(f"| Room Labeling | {report.rooms.label_coverage_pct:.0f}% "
                   f"({report.rooms.labeled_rooms}/{report.rooms.total_rooms}) |\n")
            f.write(f"| Door Schedule Match | {report.openings.door_schedule_match_pct:.0f}% |\n")
            f.write(f"| Window Schedule Match | {report.openings.window_schedule_match_pct:.0f}% |\n")
            f.write(f"| CPWD Mapping | {report.cpwd.mapping_pct:.0f}% "
                   f"({report.cpwd.mapped_lines}/{report.cpwd.total_boq_lines}) |\n")
            f.write(f"| Schedules Found | {', '.join(report.schedules.schedules_found) or 'None'} |\n")

        logger.info(f"Saved missing inputs to: {path}")
        return path

    def export_unmapped_tags(
        self,
        report: CoverageReport,
        output_dir: Path
    ) -> Path:
        """Export unmapped tags to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "unmapped_tags.csv"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Item Code", "Description", "Suggested CPWD"])

            for item in report.cpwd.unmapped_items:
                writer.writerow([
                    item.get("category", ""),
                    item.get("item_code", ""),
                    item.get("description", ""),
                    "",  # Empty for user to fill
                ])

        logger.info(f"Saved unmapped tags to: {path}")
        return path


def analyze_coverage(
    project_id: str,
    extraction_results: List[Dict[str, Any]],
    project_graph: Dict[str, Any],
    boq_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[CoverageReport, List[MissingInput]]:
    """
    Convenience function to analyze coverage and export all reports.

    Args:
        project_id: Project identifier
        extraction_results: Extraction results list
        project_graph: Project graph dict
        boq_entries: BOQ entries list
        output_dir: Output directory

    Returns:
        Tuple of (CoverageReport, List[MissingInput])
    """
    analyzer = CoverageAnalyzer()

    # Analyze coverage
    report = analyzer.analyze(
        project_id, extraction_results, project_graph, boq_entries
    )

    # Generate missing inputs
    missing = analyzer.generate_missing_inputs(report)

    # Export all files
    analyzer.export_coverage_report(report, output_dir)
    analyzer.export_missing_inputs(report, missing, output_dir)
    analyzer.export_unmapped_tags(report, output_dir)

    return report, missing
