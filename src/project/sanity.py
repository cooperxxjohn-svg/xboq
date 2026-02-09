"""
Sanity Checks - Cross-validation and reasonableness checks.

Implements:
- Total room area vs building boundary area
- Door/window detection vs schedule count discrepancy
- Toilet presence checks (doors/ventilators)
- Wall cluster reasonableness
- Multi-floor unit signature comparison

Outputs:
- sanity_checks.md with PASS/WARN/FAIL and recommended actions
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a sanity check."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"  # Not enough data to check


@dataclass
class SanityCheck:
    """A single sanity check result."""
    name: str
    category: str
    status: CheckStatus
    message: str
    details: str = ""
    recommendation: str = ""
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "recommendation": self.recommendation,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
        }

    def to_markdown(self) -> str:
        icon = {
            CheckStatus.PASS: "✅",
            CheckStatus.WARN: "⚠️",
            CheckStatus.FAIL: "❌",
            CheckStatus.SKIP: "⏭️"
        }.get(self.status, "❓")

        md = f"### {icon} {self.name}\n\n"
        md += f"**Status:** {self.status.value}\n\n"
        md += f"{self.message}\n\n"

        if self.details:
            md += f"**Details:** {self.details}\n\n"

        if self.metric_value is not None and self.threshold is not None:
            md += f"**Metric:** {self.metric_value:.1f} (threshold: {self.threshold:.1f})\n\n"

        if self.recommendation and self.status in [CheckStatus.WARN, CheckStatus.FAIL]:
            md += f"**Recommendation:** {self.recommendation}\n\n"

        return md


@dataclass
class SanityReport:
    """Complete sanity check report."""
    project_id: str
    checks: List[SanityCheck] = field(default_factory=list)
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    overall_status: CheckStatus = CheckStatus.PASS

    def add_check(self, check: SanityCheck) -> None:
        """Add a check and update counts."""
        self.checks.append(check)
        if check.status == CheckStatus.PASS:
            self.pass_count += 1
        elif check.status == CheckStatus.WARN:
            self.warn_count += 1
        elif check.status == CheckStatus.FAIL:
            self.fail_count += 1
        else:
            self.skip_count += 1

        # Update overall status
        if check.status == CheckStatus.FAIL:
            self.overall_status = CheckStatus.FAIL
        elif check.status == CheckStatus.WARN and self.overall_status != CheckStatus.FAIL:
            self.overall_status = CheckStatus.WARN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "overall_status": self.overall_status.value,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "checks": [c.to_dict() for c in self.checks],
        }


class SanityChecker:
    """
    Performs sanity checks on extraction results.
    """

    # Standard Indian wall thicknesses (mm)
    STANDARD_WALL_THICKNESSES = [115, 150, 200, 230, 300, 450]
    WALL_THICKNESS_TOLERANCE = 25  # mm

    # Toilet-related room labels
    TOILET_LABELS = [
        "toilet", "wc", "bathroom", "bath", "washroom", "lavatory",
        "w.c.", "w/c", "restroom", "powder room", "half bath"
    ]

    # Area reasonableness thresholds
    MIN_ROOM_AREA_SQM = 2.0
    MAX_ROOM_AREA_SQM = 200.0
    MIN_BUILDING_AREA_SQM = 20.0
    MAX_BUILDING_AREA_SQM = 10000.0

    def __init__(self):
        pass

    def run_all_checks(
        self,
        project_id: str,
        extraction_results: List[Dict[str, Any]],
        project_graph: Dict[str, Any],
        coverage_report: Optional[Dict[str, Any]] = None,
    ) -> SanityReport:
        """
        Run all sanity checks.

        Args:
            project_id: Project identifier
            extraction_results: List of extraction result dicts
            project_graph: Project graph dict
            coverage_report: Optional coverage report dict

        Returns:
            SanityReport with all check results
        """
        report = SanityReport(project_id=project_id)

        # Area checks
        report.add_check(self._check_room_area_sum(extraction_results))
        report.add_check(self._check_room_area_ranges(extraction_results))

        # Opening checks
        report.add_check(self._check_door_schedule_discrepancy(
            extraction_results, project_graph
        ))
        report.add_check(self._check_window_schedule_discrepancy(
            extraction_results, project_graph
        ))

        # Toilet checks
        report.add_check(self._check_toilet_openings(extraction_results))

        # Wall checks
        report.add_check(self._check_wall_thicknesses(extraction_results))

        # Multi-floor checks
        report.add_check(self._check_floor_consistency(extraction_results))

        # Scale checks
        report.add_check(self._check_scale_consistency(extraction_results))

        # Label checks
        report.add_check(self._check_room_label_duplicates(extraction_results))

        return report

    def _check_room_area_sum(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check if total room area is reasonable."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        if not floor_plans:
            return SanityCheck(
                name="Total Room Area",
                category="Area",
                status=CheckStatus.SKIP,
                message="No floor plans to analyze",
            )

        total_area = 0.0
        room_count = 0
        for fp in floor_plans:
            for room in fp.get("rooms", []):
                area = room.get("area_sqm", 0)
                if area > 0:
                    total_area += area
                    room_count += 1

        if total_area == 0:
            return SanityCheck(
                name="Total Room Area",
                category="Area",
                status=CheckStatus.WARN,
                message="No room areas computed",
                recommendation="Check scale detection or room detection quality",
            )

        # Check reasonableness
        avg_area = total_area / room_count if room_count > 0 else 0

        if total_area < self.MIN_BUILDING_AREA_SQM:
            return SanityCheck(
                name="Total Room Area",
                category="Area",
                status=CheckStatus.WARN,
                message=f"Total area ({total_area:.1f} sqm) seems too small for a building",
                metric_value=total_area,
                threshold=self.MIN_BUILDING_AREA_SQM,
                recommendation="Verify scale detection is correct",
            )

        if total_area > self.MAX_BUILDING_AREA_SQM:
            return SanityCheck(
                name="Total Room Area",
                category="Area",
                status=CheckStatus.WARN,
                message=f"Total area ({total_area:.1f} sqm) seems very large",
                metric_value=total_area,
                threshold=self.MAX_BUILDING_AREA_SQM,
                recommendation="Verify this is expected for the project scope",
            )

        return SanityCheck(
            name="Total Room Area",
            category="Area",
            status=CheckStatus.PASS,
            message=f"Total area: {total_area:.1f} sqm across {room_count} rooms",
            details=f"Average room size: {avg_area:.1f} sqm",
            metric_value=total_area,
        )

    def _check_room_area_ranges(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check for unreasonably small or large rooms."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        too_small = []
        too_large = []

        for fp in floor_plans:
            page_label = f"{Path(fp.get('file_path', '')).stem}_p{fp.get('page_number', 0) + 1}"
            for room in fp.get("rooms", []):
                area = room.get("area_sqm", 0)
                label = room.get("label", "Unknown")
                room_id = room.get("room_id", "")

                if 0 < area < self.MIN_ROOM_AREA_SQM:
                    too_small.append(f"{room_id} ({label}): {area:.1f} sqm on {page_label}")
                elif area > self.MAX_ROOM_AREA_SQM:
                    too_large.append(f"{room_id} ({label}): {area:.1f} sqm on {page_label}")

        issues = []
        if too_small:
            issues.append(f"{len(too_small)} rooms < {self.MIN_ROOM_AREA_SQM} sqm")
        if too_large:
            issues.append(f"{len(too_large)} rooms > {self.MAX_ROOM_AREA_SQM} sqm")

        if issues:
            details = []
            if too_small:
                details.append("Too small: " + "; ".join(too_small[:3]))
            if too_large:
                details.append("Too large: " + "; ".join(too_large[:3]))

            return SanityCheck(
                name="Room Area Ranges",
                category="Area",
                status=CheckStatus.WARN,
                message=f"Found rooms with unusual areas: {', '.join(issues)}",
                details="\n".join(details),
                recommendation="Review scale detection or check if these are sub-spaces (closets, niches)",
            )

        return SanityCheck(
            name="Room Area Ranges",
            category="Area",
            status=CheckStatus.PASS,
            message="All room areas within reasonable ranges",
        )

    def _check_door_schedule_discrepancy(
        self,
        results: List[Dict[str, Any]],
        project_graph: Dict[str, Any]
    ) -> SanityCheck:
        """Check door detection vs schedule count."""
        # Count detected doors
        detected_count = 0
        detected_elements = project_graph.get("detected_elements", {})
        for page_key, elements in detected_elements.items():
            for elem in elements:
                if elem.get("type") == "door":
                    detected_count += 1

        # Count schedule entries
        schedules = project_graph.get("schedules", {})
        door_schedule = schedules.get("door", [])
        schedule_count = len(door_schedule)

        if detected_count == 0 and schedule_count == 0:
            return SanityCheck(
                name="Door Count Verification",
                category="Openings",
                status=CheckStatus.SKIP,
                message="No doors detected and no door schedule found",
            )

        if schedule_count == 0:
            return SanityCheck(
                name="Door Count Verification",
                category="Openings",
                status=CheckStatus.WARN,
                message=f"Detected {detected_count} doors but no door schedule found",
                recommendation="Provide door schedule for verification",
            )

        if detected_count == 0:
            return SanityCheck(
                name="Door Count Verification",
                category="Openings",
                status=CheckStatus.WARN,
                message=f"Door schedule has {schedule_count} entries but no doors detected",
                recommendation="Check door detection quality on floor plans",
            )

        # Calculate discrepancy
        discrepancy_pct = abs(detected_count - schedule_count) / max(detected_count, schedule_count) * 100

        if discrepancy_pct > 50:
            status = CheckStatus.FAIL
        elif discrepancy_pct > 25:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS

        return SanityCheck(
            name="Door Count Verification",
            category="Openings",
            status=status,
            message=f"Detected {detected_count} doors vs {schedule_count} in schedule ({discrepancy_pct:.0f}% discrepancy)",
            metric_value=discrepancy_pct,
            threshold=25.0,
            recommendation="Review floor plans for missing door detections or check schedule accuracy" if status != CheckStatus.PASS else "",
        )

    def _check_window_schedule_discrepancy(
        self,
        results: List[Dict[str, Any]],
        project_graph: Dict[str, Any]
    ) -> SanityCheck:
        """Check window detection vs schedule count."""
        # Count detected windows
        detected_count = 0
        detected_elements = project_graph.get("detected_elements", {})
        for page_key, elements in detected_elements.items():
            for elem in elements:
                if elem.get("type") == "window":
                    detected_count += 1

        # Count schedule entries
        schedules = project_graph.get("schedules", {})
        window_schedule = schedules.get("window", [])
        schedule_count = len(window_schedule)

        if detected_count == 0 and schedule_count == 0:
            return SanityCheck(
                name="Window Count Verification",
                category="Openings",
                status=CheckStatus.SKIP,
                message="No windows detected and no window schedule found",
            )

        if schedule_count == 0:
            return SanityCheck(
                name="Window Count Verification",
                category="Openings",
                status=CheckStatus.WARN,
                message=f"Detected {detected_count} windows but no window schedule found",
                recommendation="Provide window schedule for verification",
            )

        if detected_count == 0:
            return SanityCheck(
                name="Window Count Verification",
                category="Openings",
                status=CheckStatus.WARN,
                message=f"Window schedule has {schedule_count} entries but no windows detected",
                recommendation="Check window detection quality on floor plans",
            )

        # Calculate discrepancy
        discrepancy_pct = abs(detected_count - schedule_count) / max(detected_count, schedule_count) * 100

        if discrepancy_pct > 50:
            status = CheckStatus.FAIL
        elif discrepancy_pct > 25:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS

        return SanityCheck(
            name="Window Count Verification",
            category="Openings",
            status=status,
            message=f"Detected {detected_count} windows vs {schedule_count} in schedule ({discrepancy_pct:.0f}% discrepancy)",
            metric_value=discrepancy_pct,
            threshold=25.0,
            recommendation="Review floor plans for missing window detections" if status != CheckStatus.PASS else "",
        )

    def _check_toilet_openings(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check that toilets have appropriate openings."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        toilets_found = []
        toilets_with_doors = []
        toilets_with_ventilation = []

        for fp in floor_plans:
            page_label = f"{Path(fp.get('file_path', '')).stem}_p{fp.get('page_number', 0) + 1}"
            rooms = fp.get("rooms", [])
            openings = fp.get("openings", [])

            for room in rooms:
                label = room.get("label", "").lower()
                room_id = room.get("room_id", "")

                # Check if toilet
                is_toilet = any(t in label for t in self.TOILET_LABELS)
                if not is_toilet:
                    continue

                toilets_found.append(f"{room_id} on {page_label}")

                # Check for door (simplified - would need geometry matching in real impl)
                has_door = any(
                    o.get("type") == "door" and o.get("adjacent_room") == room_id
                    for o in openings
                )
                if has_door:
                    toilets_with_doors.append(room_id)

                # Check for ventilation (window or exhaust)
                has_vent = any(
                    (o.get("type") == "window" and o.get("adjacent_room") == room_id) or
                    o.get("is_ventilator")
                    for o in openings
                )
                if has_vent:
                    toilets_with_ventilation.append(room_id)

        if not toilets_found:
            return SanityCheck(
                name="Toilet Openings",
                category="Openings",
                status=CheckStatus.SKIP,
                message="No toilet rooms detected",
            )

        issues = []
        missing_doors = len(toilets_found) - len(toilets_with_doors)
        missing_vents = len(toilets_found) - len(toilets_with_ventilation)

        if missing_doors > 0:
            issues.append(f"{missing_doors} toilets without detected doors")
        if missing_vents > 0:
            issues.append(f"{missing_vents} toilets without ventilation")

        if issues:
            return SanityCheck(
                name="Toilet Openings",
                category="Openings",
                status=CheckStatus.WARN,
                message=f"Found {len(toilets_found)} toilets: {', '.join(issues)}",
                details=f"Toilets: {', '.join(toilets_found[:5])}",
                recommendation="Verify toilet doors and ventilators are detected",
            )

        return SanityCheck(
            name="Toilet Openings",
            category="Openings",
            status=CheckStatus.PASS,
            message=f"All {len(toilets_found)} toilets have expected openings",
        )

    def _check_wall_thicknesses(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check wall thickness clusters are reasonable."""
        all_thicknesses = []

        for result in results:
            if result.get("page_type") != "floor_plan":
                continue
            wall_info = result.get("wall_info", {})
            thicknesses = wall_info.get("thicknesses", [])
            all_thicknesses.extend(thicknesses)

        if not all_thicknesses:
            return SanityCheck(
                name="Wall Thickness Reasonableness",
                category="Walls",
                status=CheckStatus.SKIP,
                message="No wall thickness data available",
            )

        # Cluster by rounding
        thickness_counter = Counter(round(t / 10) * 10 for t in all_thicknesses)
        main_clusters = [t for t, c in thickness_counter.most_common(5)]

        # Check against standard thicknesses
        unexpected = []
        expected = []
        for thickness in main_clusters:
            is_standard = any(
                abs(thickness - std) <= self.WALL_THICKNESS_TOLERANCE
                for std in self.STANDARD_WALL_THICKNESSES
            )
            if is_standard:
                expected.append(thickness)
            else:
                unexpected.append(thickness)

        if unexpected and len(unexpected) > len(expected):
            return SanityCheck(
                name="Wall Thickness Reasonableness",
                category="Walls",
                status=CheckStatus.WARN,
                message=f"Found non-standard wall thicknesses: {unexpected}",
                details=f"Standard thicknesses detected: {expected}",
                recommendation="Verify wall specifications or check scale accuracy",
            )

        if len(main_clusters) > 4:
            return SanityCheck(
                name="Wall Thickness Reasonableness",
                category="Walls",
                status=CheckStatus.WARN,
                message=f"Many wall thickness clusters ({len(main_clusters)}) - may indicate noise",
                details=f"Clusters: {main_clusters}",
                recommendation="Check wall detection quality",
            )

        return SanityCheck(
            name="Wall Thickness Reasonableness",
            category="Walls",
            status=CheckStatus.PASS,
            message=f"Wall thicknesses are reasonable: {main_clusters}",
        )

    def _check_floor_consistency(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check consistency across multiple floor plans (for multi-story buildings)."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        if len(floor_plans) < 2:
            return SanityCheck(
                name="Floor Plan Consistency",
                category="Multi-Floor",
                status=CheckStatus.SKIP,
                message="Not enough floor plans to compare",
            )

        # Build label histograms for each floor
        floor_histograms = []
        for fp in floor_plans:
            labels = Counter()
            for room in fp.get("rooms", []):
                label = room.get("label", "Room").lower()
                # Normalize label
                label = label.replace("1", "").replace("2", "").replace("3", "").strip()
                labels[label] += 1

            if labels:
                floor_histograms.append({
                    "page": f"{Path(fp.get('file_path', '')).stem}_p{fp.get('page_number', 0) + 1}",
                    "labels": labels,
                    "total": sum(labels.values())
                })

        if len(floor_histograms) < 2:
            return SanityCheck(
                name="Floor Plan Consistency",
                category="Multi-Floor",
                status=CheckStatus.SKIP,
                message="Not enough labeled rooms to compare",
            )

        # Compare histograms (simplified Jaccard similarity)
        similarities = []
        for i in range(len(floor_histograms)):
            for j in range(i + 1, len(floor_histograms)):
                h1 = set(floor_histograms[i]["labels"].keys())
                h2 = set(floor_histograms[j]["labels"].keys())
                if h1 or h2:
                    jaccard = len(h1 & h2) / len(h1 | h2)
                    similarities.append(jaccard)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        if avg_similarity < 0.3:
            return SanityCheck(
                name="Floor Plan Consistency",
                category="Multi-Floor",
                status=CheckStatus.WARN,
                message=f"Floor plans have low similarity ({avg_similarity:.0%})",
                metric_value=avg_similarity * 100,
                threshold=30.0,
                recommendation="Verify these are floors of the same building or check room labeling",
            )

        return SanityCheck(
            name="Floor Plan Consistency",
            category="Multi-Floor",
            status=CheckStatus.PASS,
            message=f"Floor plans show consistent room patterns ({avg_similarity:.0%} similarity)",
            metric_value=avg_similarity * 100,
        )

    def _check_scale_consistency(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check if scale is consistent across floor plans."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        scales = []
        for fp in floor_plans:
            scale_info = fp.get("scale_info", {})
            ppm = scale_info.get("pixels_per_mm", 0)
            if ppm > 0:
                scales.append({
                    "page": f"{Path(fp.get('file_path', '')).stem}_p{fp.get('page_number', 0) + 1}",
                    "ppm": ppm,
                    "method": scale_info.get("method", "unknown"),
                    "confidence": scale_info.get("confidence", 0),
                })

        if len(scales) < 2:
            return SanityCheck(
                name="Scale Consistency",
                category="Scale",
                status=CheckStatus.SKIP,
                message="Not enough scale data to compare",
            )

        # Check variance
        ppms = [s["ppm"] for s in scales]
        avg_ppm = sum(ppms) / len(ppms)
        max_deviation = max(abs(p - avg_ppm) / avg_ppm * 100 for p in ppms) if avg_ppm > 0 else 0

        if max_deviation > 20:
            inconsistent = [
                f"{s['page']}: {s['ppm']:.3f}"
                for s in scales
                if abs(s['ppm'] - avg_ppm) / avg_ppm * 100 > 10
            ]
            return SanityCheck(
                name="Scale Consistency",
                category="Scale",
                status=CheckStatus.WARN,
                message=f"Scale varies significantly across pages ({max_deviation:.0f}% max deviation)",
                details=f"Inconsistent pages: {', '.join(inconsistent[:3])}",
                metric_value=max_deviation,
                threshold=20.0,
                recommendation="Verify all pages use the same scale or re-run with explicit scale",
            )

        return SanityCheck(
            name="Scale Consistency",
            category="Scale",
            status=CheckStatus.PASS,
            message=f"Scale is consistent across {len(scales)} pages ({max_deviation:.1f}% max deviation)",
            metric_value=max_deviation,
        )

    def _check_room_label_duplicates(
        self,
        results: List[Dict[str, Any]]
    ) -> SanityCheck:
        """Check for unexpected duplicate room labels on same floor."""
        floor_plans = [r for r in results if r.get("page_type") == "floor_plan"]

        pages_with_duplicates = []

        for fp in floor_plans:
            page_label = f"{Path(fp.get('file_path', '')).stem}_p{fp.get('page_number', 0) + 1}"
            labels = [r.get("label", "") for r in fp.get("rooms", []) if r.get("label")]

            # Count labels
            label_counts = Counter(labels)
            duplicates = [label for label, count in label_counts.items()
                         if count > 1 and label.lower() not in ["room", "passage", "corridor"]]

            if duplicates:
                pages_with_duplicates.append(f"{page_label}: {duplicates}")

        if pages_with_duplicates:
            return SanityCheck(
                name="Room Label Uniqueness",
                category="Labels",
                status=CheckStatus.WARN,
                message=f"Found duplicate room labels on {len(pages_with_duplicates)} pages",
                details="; ".join(pages_with_duplicates[:3]),
                recommendation="Verify room labels are correct or add distinguishing numbers",
            )

        return SanityCheck(
            name="Room Label Uniqueness",
            category="Labels",
            status=CheckStatus.PASS,
            message="Room labels are appropriately unique per page",
        )

    def export_report(
        self,
        report: SanityReport,
        output_dir: Path
    ) -> Path:
        """Export sanity report to Markdown."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "sanity_checks.md"

        with open(path, "w") as f:
            f.write(f"# Sanity Checks: {report.project_id}\n\n")

            # Overall status
            icon = {
                CheckStatus.PASS: "✅",
                CheckStatus.WARN: "⚠️",
                CheckStatus.FAIL: "❌",
            }.get(report.overall_status, "❓")

            f.write(f"**Overall Status:** {icon} {report.overall_status.value}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write("| Status | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| ✅ PASS | {report.pass_count} |\n")
            f.write(f"| ⚠️ WARN | {report.warn_count} |\n")
            f.write(f"| ❌ FAIL | {report.fail_count} |\n")
            f.write(f"| ⏭️ SKIP | {report.skip_count} |\n")
            f.write("\n---\n\n")

            # Group by category
            by_category: Dict[str, List[SanityCheck]] = defaultdict(list)
            for check in report.checks:
                by_category[check.category].append(check)

            for category, checks in sorted(by_category.items()):
                f.write(f"## {category}\n\n")
                for check in checks:
                    f.write(check.to_markdown())
                    f.write("---\n\n")

        logger.info(f"Saved sanity checks to: {path}")
        return path


def run_sanity_checks(
    project_id: str,
    extraction_results: List[Dict[str, Any]],
    project_graph: Dict[str, Any],
    output_dir: Path,
    coverage_report: Optional[Dict[str, Any]] = None,
) -> SanityReport:
    """
    Convenience function to run all sanity checks and export report.

    Args:
        project_id: Project identifier
        extraction_results: Extraction results list
        project_graph: Project graph dict
        output_dir: Output directory
        coverage_report: Optional coverage report dict

    Returns:
        SanityReport
    """
    checker = SanityChecker()
    report = checker.run_all_checks(
        project_id, extraction_results, project_graph, coverage_report
    )
    checker.export_report(report, output_dir)
    return report
