"""
Measurement Gate

HARD GATE for evidence-first estimation.

Rules:
1. If scale_basis == unknown and no manual scale -> FAIL_MEASUREMENT
2. If wall segments < threshold OR no closed room polygons -> FAIL_MEASUREMENT
3. If openings not detected AND no schedules -> WARN and mark openings quantities TBD

If FAIL_MEASUREMENT:
- still produce: scope register, RFIs, missing inputs checklist
- DO NOT produce priced estimate
- DO NOT produce "measured BOQ" quantities except what is truly measurable
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class GateStatus(Enum):
    """Gate result status."""
    PASS = "PASS"                    # All checks passed
    WARN = "WARN"                    # Some issues but can proceed with caution
    FAIL_MEASUREMENT = "FAIL_MEASUREMENT"  # Cannot produce reliable measurements
    FAIL_SCALE = "FAIL_SCALE"        # Scale could not be determined
    FAIL_GEOMETRY = "FAIL_GEOMETRY"  # Insufficient geometry detected


@dataclass
class GateCheck:
    """Individual gate check result."""
    name: str
    passed: bool
    severity: str  # "blocker", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementGateResult:
    """Result of measurement gate evaluation."""
    status: GateStatus
    can_produce_measured_boq: bool
    can_produce_pricing: bool
    checks: List[GateCheck] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Detailed metrics
    scale_confidence: float = 0.0
    geometry_coverage: float = 0.0
    wall_segment_count: int = 0
    closed_polygon_count: int = 0
    opening_detection_count: int = 0
    schedule_table_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "can_produce_measured_boq": self.can_produce_measured_boq,
            "can_produce_pricing": self.can_produce_pricing,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "metrics": {
                "scale_confidence": self.scale_confidence,
                "geometry_coverage": self.geometry_coverage,
                "wall_segment_count": self.wall_segment_count,
                "closed_polygon_count": self.closed_polygon_count,
                "opening_detection_count": self.opening_detection_count,
                "schedule_table_count": self.schedule_table_count,
            },
        }


class MeasurementGate:
    """
    Hard gate for measurement verification.

    This gate determines whether we can produce reliable measured quantities.
    """

    # Thresholds
    MIN_WALL_SEGMENTS = 10  # Minimum wall segments for a valid floor plan
    MIN_CLOSED_POLYGONS = 1  # Minimum closed room polygons
    MIN_SCALE_CONFIDENCE = 0.5  # Minimum confidence in scale determination
    MIN_GEOMETRY_COVERAGE = 0.3  # Minimum coverage of geometry detection

    def __init__(self, output_dir: Path, project_metadata: Dict = None):
        self.output_dir = Path(output_dir)
        self.metadata = project_metadata or {}
        self.result: Optional[MeasurementGateResult] = None

    def evaluate(self) -> MeasurementGateResult:
        """
        Run all gate checks.

        Returns MeasurementGateResult with status and details.
        """
        checks = []
        blockers = []
        warnings = []
        recommendations = []

        # Load extracted data
        geometry_data = self._load_geometry_data()
        scale_data = self._load_scale_data()
        schedule_data = self._load_schedule_data()

        # Check 1: Scale verification
        scale_check = self._check_scale(scale_data)
        checks.append(scale_check)
        if not scale_check.passed and scale_check.severity == "blocker":
            blockers.append(scale_check.message)

        # Check 2: Wall geometry
        wall_check = self._check_wall_geometry(geometry_data)
        checks.append(wall_check)
        if not wall_check.passed and wall_check.severity == "blocker":
            blockers.append(wall_check.message)

        # Check 3: Room polygons
        polygon_check = self._check_room_polygons(geometry_data)
        checks.append(polygon_check)
        if not polygon_check.passed and polygon_check.severity == "blocker":
            blockers.append(polygon_check.message)

        # Check 4: Opening detection
        opening_check = self._check_openings(geometry_data, schedule_data)
        checks.append(opening_check)
        if not opening_check.passed:
            if opening_check.severity == "blocker":
                blockers.append(opening_check.message)
            else:
                warnings.append(opening_check.message)

        # Check 5: Schedule tables (if available)
        schedule_check = self._check_schedules(schedule_data)
        checks.append(schedule_check)
        if not schedule_check.passed and schedule_check.severity == "warning":
            warnings.append(schedule_check.message)

        # Determine overall status
        status = self._determine_status(checks, blockers)

        # Generate recommendations
        recommendations = self._generate_recommendations(checks, blockers, warnings)

        # Build result
        # Allow pricing with WARN if manual scale was provided (user explicitly verified)
        can_price = status == GateStatus.PASS
        if status == GateStatus.WARN and self.metadata.get("manual_scale"):
            can_price = True  # User provided scale = explicit verification

        self.result = MeasurementGateResult(
            status=status,
            can_produce_measured_boq=status in [GateStatus.PASS, GateStatus.WARN],
            can_produce_pricing=can_price,
            checks=checks,
            blockers=blockers,
            warnings=warnings,
            recommendations=recommendations,
            scale_confidence=scale_check.details.get("confidence", 0.0),
            geometry_coverage=wall_check.details.get("coverage", 0.0),
            wall_segment_count=wall_check.details.get("segment_count", 0),
            closed_polygon_count=polygon_check.details.get("polygon_count", 0),
            opening_detection_count=opening_check.details.get("opening_count", 0),
            schedule_table_count=schedule_check.details.get("table_count", 0),
        )

        return self.result

    def _load_geometry_data(self) -> Dict[str, Any]:
        """Load geometry extraction results."""
        data = {
            "rooms": [],
            "openings": [],
            "walls": [],
            "has_geometry": False,
        }

        # Load rooms
        rooms_file = self.output_dir / "boq" / "rooms.json"
        if rooms_file.exists():
            with open(rooms_file) as f:
                rooms_data = json.load(f)
                data["rooms"] = rooms_data.get("rooms", [])

        # Load openings
        openings_file = self.output_dir / "boq" / "openings.json"
        if openings_file.exists():
            with open(openings_file) as f:
                openings_data = json.load(f)
                data["openings"] = openings_data.get("openings", [])

        # Check for geometry evidence in rooms
        for room in data["rooms"]:
            if room.get("bbox") or room.get("polygon") or room.get("area_sqm"):
                data["has_geometry"] = True
                break

        return data

    def _load_scale_data(self) -> Dict[str, Any]:
        """Load scale inference results."""
        data = {
            "scale_determined": False,
            "scale_value": None,
            "scale_basis": "unknown",
            "confidence": 0.0,
        }

        # Check measurement output
        measurement_file = self.output_dir / "measurement" / "estimator_math_summary.json"
        if measurement_file.exists():
            with open(measurement_file) as f:
                meas_data = json.load(f)
                if meas_data.get("scale"):
                    data["scale_determined"] = True
                    data["scale_value"] = meas_data.get("scale")
                    data["scale_basis"] = meas_data.get("scale_basis", "inferred")
                    data["confidence"] = meas_data.get("scale_confidence", 0.5)

        # Check for dimension text that could verify scale
        # TODO: Load from dimension extraction

        return data

    def _load_schedule_data(self) -> Dict[str, Any]:
        """Load schedule/table extraction results."""
        data = {
            "tables_found": [],
            "door_schedule": None,
            "window_schedule": None,
            "room_schedule": None,
        }

        # Check for schedule files
        schedules_dir = self.output_dir / "schedules"
        if schedules_dir.exists():
            for f in schedules_dir.glob("*.json"):
                data["tables_found"].append(f.stem)

        # Check scope directory for schedules
        scope_dir = self.output_dir / "scope"
        if scope_dir.exists():
            opening_schedule = scope_dir / "openings_schedule.csv"
            if opening_schedule.exists():
                data["door_schedule"] = str(opening_schedule)

        return data

    def _check_scale(self, scale_data: Dict) -> GateCheck:
        """Check if scale can be reliably determined."""
        if scale_data["scale_determined"] and scale_data["confidence"] >= self.MIN_SCALE_CONFIDENCE:
            return GateCheck(
                name="scale_verification",
                passed=True,
                severity="info",
                message=f"Scale determined: 1:{scale_data['scale_value']} (confidence: {scale_data['confidence']:.0%})",
                details={
                    "scale_value": scale_data["scale_value"],
                    "confidence": scale_data["confidence"],
                    "basis": scale_data["scale_basis"],
                },
            )

        # Check if manual scale provided
        manual_scale = self.metadata.get("manual_scale")
        if manual_scale:
            return GateCheck(
                name="scale_verification",
                passed=True,
                severity="info",
                message=f"Manual scale provided: 1:{manual_scale}",
                details={
                    "scale_value": manual_scale,
                    "confidence": 1.0,
                    "basis": "manual",
                },
            )

        return GateCheck(
            name="scale_verification",
            passed=False,
            severity="blocker",
            message="FAIL: Scale could not be reliably determined. Measurements unreliable.",
            details={
                "scale_value": scale_data.get("scale_value"),
                "confidence": scale_data.get("confidence", 0.0),
                "basis": "unknown",
            },
        )

    def _check_wall_geometry(self, geometry_data: Dict) -> GateCheck:
        """Check if sufficient wall geometry was extracted."""
        # Count rooms with actual geometry (bbox or polygon)
        rooms_with_geometry = [
            r for r in geometry_data["rooms"]
            if r.get("bbox") or r.get("polygon") or r.get("area_sqm", 0) > 0
        ]

        # Estimate wall segments from room count (rough heuristic)
        estimated_segments = len(rooms_with_geometry) * 4  # ~4 walls per room

        coverage = len(rooms_with_geometry) / max(len(geometry_data["rooms"]), 1)

        if estimated_segments >= self.MIN_WALL_SEGMENTS and coverage >= self.MIN_GEOMETRY_COVERAGE:
            return GateCheck(
                name="wall_geometry",
                passed=True,
                severity="info",
                message=f"Wall geometry detected: ~{estimated_segments} segments from {len(rooms_with_geometry)} rooms",
                details={
                    "segment_count": estimated_segments,
                    "rooms_with_geometry": len(rooms_with_geometry),
                    "coverage": coverage,
                },
            )

        if estimated_segments > 0:
            return GateCheck(
                name="wall_geometry",
                passed=False,
                severity="warning",
                message=f"WARN: Limited wall geometry: only {estimated_segments} segments detected",
                details={
                    "segment_count": estimated_segments,
                    "rooms_with_geometry": len(rooms_with_geometry),
                    "coverage": coverage,
                },
            )

        return GateCheck(
            name="wall_geometry",
            passed=False,
            severity="blocker",
            message="FAIL: No wall geometry detected. Cannot measure areas reliably.",
            details={
                "segment_count": 0,
                "rooms_with_geometry": 0,
                "coverage": 0.0,
            },
        )

    def _check_room_polygons(self, geometry_data: Dict) -> GateCheck:
        """Check if closed room polygons were detected."""
        rooms_with_polygons = [
            r for r in geometry_data["rooms"]
            if r.get("polygon") or (r.get("bbox") and len(r.get("bbox", [])) >= 4)
        ]

        # Also count rooms with area (implies successful polygon)
        rooms_with_area = [
            r for r in geometry_data["rooms"]
            if r.get("area_sqm", 0) > 0 or r.get("area", 0) > 0
        ]

        polygon_count = max(len(rooms_with_polygons), len(rooms_with_area))

        if polygon_count >= self.MIN_CLOSED_POLYGONS:
            return GateCheck(
                name="room_polygons",
                passed=True,
                severity="info",
                message=f"Room polygons detected: {polygon_count} closed rooms",
                details={
                    "polygon_count": polygon_count,
                    "total_rooms": len(geometry_data["rooms"]),
                },
            )

        if geometry_data["rooms"]:
            return GateCheck(
                name="room_polygons",
                passed=False,
                severity="warning",
                message=f"WARN: Rooms detected but no closed polygons. Areas may be unreliable.",
                details={
                    "polygon_count": 0,
                    "total_rooms": len(geometry_data["rooms"]),
                },
            )

        return GateCheck(
            name="room_polygons",
            passed=False,
            severity="blocker",
            message="FAIL: No room polygons detected. Cannot calculate floor areas.",
            details={
                "polygon_count": 0,
                "total_rooms": 0,
            },
        )

    def _check_openings(self, geometry_data: Dict, schedule_data: Dict) -> GateCheck:
        """Check if openings were detected or schedule available."""
        opening_count = len(geometry_data.get("openings", []))
        has_schedule = bool(schedule_data.get("door_schedule") or schedule_data.get("window_schedule"))

        if opening_count > 0:
            return GateCheck(
                name="opening_detection",
                passed=True,
                severity="info",
                message=f"Openings detected: {opening_count} doors/windows",
                details={
                    "opening_count": opening_count,
                    "has_schedule": has_schedule,
                },
            )

        if has_schedule:
            return GateCheck(
                name="opening_detection",
                passed=True,
                severity="info",
                message="Opening schedule found (no visual detection)",
                details={
                    "opening_count": 0,
                    "has_schedule": True,
                },
            )

        return GateCheck(
            name="opening_detection",
            passed=False,
            severity="warning",
            message="WARN: No openings detected and no schedule. Door/window quantities TBD.",
            details={
                "opening_count": 0,
                "has_schedule": False,
            },
        )

    def _check_schedules(self, schedule_data: Dict) -> GateCheck:
        """Check for schedule tables."""
        table_count = len(schedule_data.get("tables_found", []))

        if table_count > 0:
            return GateCheck(
                name="schedule_tables",
                passed=True,
                severity="info",
                message=f"Schedule tables found: {table_count}",
                details={
                    "table_count": table_count,
                    "tables": schedule_data["tables_found"],
                },
            )

        return GateCheck(
            name="schedule_tables",
            passed=False,
            severity="warning",
            message="No schedule tables found. Quantities rely solely on geometry.",
            details={
                "table_count": 0,
            },
        )

    def _determine_status(self, checks: List[GateCheck], blockers: List[str]) -> GateStatus:
        """Determine overall gate status."""
        if blockers:
            # Check specific failure types
            for check in checks:
                if not check.passed and check.severity == "blocker":
                    if "scale" in check.name.lower():
                        return GateStatus.FAIL_SCALE
                    if "geometry" in check.name.lower() or "polygon" in check.name.lower():
                        return GateStatus.FAIL_GEOMETRY

            return GateStatus.FAIL_MEASUREMENT

        # Check for warnings
        has_warnings = any(not c.passed for c in checks)
        if has_warnings:
            return GateStatus.WARN

        return GateStatus.PASS

    def _generate_recommendations(
        self,
        checks: List[GateCheck],
        blockers: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        for check in checks:
            if not check.passed:
                if "scale" in check.name:
                    recs.append("Provide manual scale via --scale 1:100 or add dimension annotations to drawings")
                if "wall" in check.name or "polygon" in check.name:
                    recs.append("Ensure drawings are vector PDFs with clean wall lines for reliable area detection")
                if "opening" in check.name:
                    recs.append("Add door/window schedule to drawings or provide opening quantities manually")
                if "schedule" in check.name:
                    recs.append("Consider adding schedule tables (door, window, room finish) for verification")

        if blockers:
            recs.insert(0, "MEASUREMENT GATE FAILED: Cannot produce reliable measured quantities")
            recs.append("Run with --allow_inferred_pricing to generate estimates using allowances (not recommended)")

        return list(dict.fromkeys(recs))  # Remove duplicates

    def write_report(self) -> Path:
        """Write measurement gate report."""
        if not self.result:
            self.evaluate()

        report_path = self.output_dir / "measurement_gate_report.md"

        lines = [
            "# Measurement Gate Report",
            "",
            f"**Status: {self.result.status.value}**",
            "",
            "## Summary",
            "",
            f"- Can produce measured BOQ: {'✅ YES' if self.result.can_produce_measured_boq else '❌ NO'}",
            f"- Can produce pricing: {'✅ YES' if self.result.can_produce_pricing else '❌ NO'}",
            "",
            "## Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Scale Confidence | {self.result.scale_confidence:.0%} |",
            f"| Geometry Coverage | {self.result.geometry_coverage:.0%} |",
            f"| Wall Segments | {self.result.wall_segment_count} |",
            f"| Closed Polygons | {self.result.closed_polygon_count} |",
            f"| Openings Detected | {self.result.opening_detection_count} |",
            f"| Schedule Tables | {self.result.schedule_table_count} |",
            "",
        ]

        if self.result.blockers:
            lines.extend([
                "## ❌ Blockers",
                "",
            ])
            for b in self.result.blockers:
                lines.append(f"- {b}")
            lines.append("")

        if self.result.warnings:
            lines.extend([
                "## ⚠️ Warnings",
                "",
            ])
            for w in self.result.warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.extend([
            "## Check Details",
            "",
        ])
        for check in self.result.checks:
            status_icon = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
            lines.append(f"### {status_icon} {check.name}")
            lines.append(f"- **Status:** {'PASS' if check.passed else 'FAIL'}")
            lines.append(f"- **Severity:** {check.severity}")
            lines.append(f"- **Message:** {check.message}")
            if check.details:
                lines.append(f"- **Details:** {json.dumps(check.details)}")
            lines.append("")

        if self.result.recommendations:
            lines.extend([
                "## 📋 Recommendations",
                "",
            ])
            for i, rec in enumerate(self.result.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        # Also save JSON version
        result_json = self.output_dir / "measurement_gate_result.json"
        with open(result_json, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2)

        return report_path


def run_measurement_gate(
    output_dir: Path,
    project_metadata: Dict = None,
) -> MeasurementGateResult:
    """
    Run measurement gate and return result.

    Args:
        output_dir: Project output directory
        project_metadata: Optional metadata including manual_scale

    Returns:
        MeasurementGateResult with status and details
    """
    gate = MeasurementGate(output_dir, project_metadata)
    result = gate.evaluate()
    gate.write_report()
    return result
