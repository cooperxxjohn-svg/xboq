"""
Triangulation Engine for XBOQ.

Computes key quantities using 2-3 independent methods and compares
results to identify discrepancies and compute agreement scores.

India-specific construction estimation accuracy layer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class AgreementLevel(Enum):
    """Agreement level between triangulation methods."""
    EXCELLENT = "EXCELLENT"  # <2% variance
    GOOD = "GOOD"            # 2-5% variance
    FAIR = "FAIR"            # 5-10% variance
    POOR = "POOR"            # 10-20% variance
    DISCREPANCY = "DISCREPANCY"  # >20% variance


@dataclass
class TriangulationMethod:
    """Single triangulation method result."""
    method_name: str
    value: float
    unit: str
    confidence: float  # 0-100
    source: str  # e.g., "room_polygons", "schedule_table", "wall_sum"
    notes: str = ""


@dataclass
class TriangulationResult:
    """Result of triangulating a single quantity."""
    quantity_name: str
    quantity_type: str  # "area", "length", "count"
    methods: List[TriangulationMethod] = field(default_factory=list)
    final_value: float = 0.0
    unit: str = ""
    agreement_level: AgreementLevel = AgreementLevel.DISCREPANCY
    agreement_score: float = 0.0  # 0-100
    variance_pct: float = 0.0
    discrepancy_notes: List[str] = field(default_factory=list)
    recommended_action: str = ""


@dataclass
class TriangulationReport:
    """Complete triangulation report for a project."""
    project_id: str
    results: List[TriangulationResult] = field(default_factory=list)
    overall_agreement: float = 0.0  # 0-100
    critical_discrepancies: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class TriangulationEngine:
    """
    Multi-method quantity verification engine.

    Computes quantities using independent methods and flags discrepancies.
    """

    # Variance thresholds for agreement levels
    THRESHOLDS = {
        AgreementLevel.EXCELLENT: 2.0,
        AgreementLevel.GOOD: 5.0,
        AgreementLevel.FAIR: 10.0,
        AgreementLevel.POOR: 20.0,
    }

    def __init__(self):
        self.results: List[TriangulationResult] = []

    def triangulate_project(
        self,
        project_id: str,
        extraction_results: List[Dict],
        project_graph: Dict,
        boq_entries: List[Dict],
        schedules: List[Dict],
    ) -> TriangulationReport:
        """
        Run full triangulation on a project.

        Args:
            project_id: Project identifier
            extraction_results: Page extraction results
            project_graph: Joined project graph
            boq_entries: BOQ quantity entries
            schedules: Extracted schedule tables

        Returns:
            Complete triangulation report
        """
        self.results = []

        # Triangulate floor area
        area_result = self._triangulate_floor_area(
            extraction_results, project_graph, boq_entries
        )
        if area_result:
            self.results.append(area_result)

        # Triangulate wall length
        wall_result = self._triangulate_wall_length(
            extraction_results, project_graph, boq_entries
        )
        if wall_result:
            self.results.append(wall_result)

        # Triangulate door count
        door_result = self._triangulate_door_count(
            extraction_results, project_graph, schedules, boq_entries
        )
        if door_result:
            self.results.append(door_result)

        # Triangulate window count
        window_result = self._triangulate_window_count(
            extraction_results, project_graph, schedules, boq_entries
        )
        if window_result:
            self.results.append(window_result)

        # Triangulate room count
        room_result = self._triangulate_room_count(
            extraction_results, project_graph
        )
        if room_result:
            self.results.append(room_result)

        # Triangulate toilet count (critical for MEP)
        toilet_result = self._triangulate_toilet_count(
            extraction_results, project_graph, schedules
        )
        if toilet_result:
            self.results.append(toilet_result)

        # Build report
        report = self._build_report(project_id)

        return report

    def _triangulate_floor_area(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
        boq_entries: List[Dict],
    ) -> Optional[TriangulationResult]:
        """Triangulate total floor area using multiple methods."""

        result = TriangulationResult(
            quantity_name="Total Floor Area",
            quantity_type="area",
            unit="sqm",
        )

        # Method 1: Sum of room polygon areas
        room_area_total = 0.0
        room_count = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                for room in page.get("rooms", []):
                    area = room.get("area_sqm", 0)
                    if area > 0:
                        room_area_total += area
                        room_count += 1

        if room_area_total > 0:
            result.methods.append(TriangulationMethod(
                method_name="Room Polygon Sum",
                value=round(room_area_total, 2),
                unit="sqm",
                confidence=85 if room_count > 3 else 70,
                source="room_polygons",
                notes=f"Sum of {room_count} detected rooms"
            ))

        # Method 2: BOQ flooring entries sum
        boq_floor_area = 0.0
        flooring_items = 0
        for entry in boq_entries:
            item_type = entry.get("item_type", "").lower()
            if any(kw in item_type for kw in ["flooring", "tile", "floor"]):
                qty = entry.get("quantity", 0)
                unit = entry.get("unit", "").lower()
                if unit in ["sqm", "sq.m", "sq m", "m2"]:
                    boq_floor_area += qty
                    flooring_items += 1

        if boq_floor_area > 0:
            result.methods.append(TriangulationMethod(
                method_name="BOQ Flooring Sum",
                value=round(boq_floor_area, 2),
                unit="sqm",
                confidence=90 if flooring_items > 2 else 75,
                source="boq_entries",
                notes=f"Sum of {flooring_items} flooring line items"
            ))

        # Method 3: Carpet area from notes/title block
        # (would be extracted from text analysis - placeholder)
        graph_area = project_graph.get("summary", {}).get("total_area_sqm", 0)
        if graph_area > 0:
            result.methods.append(TriangulationMethod(
                method_name="Project Graph Total",
                value=round(graph_area, 2),
                unit="sqm",
                confidence=80,
                source="project_graph",
                notes="Aggregated from project graph"
            ))

        # Compute agreement
        if len(result.methods) >= 2:
            self._compute_agreement(result)
            return result
        elif len(result.methods) == 1:
            result.final_value = result.methods[0].value
            result.agreement_level = AgreementLevel.FAIR
            result.agreement_score = 60
            result.discrepancy_notes.append("Only one method available - verify manually")
            return result

        return None

    def _triangulate_wall_length(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
        boq_entries: List[Dict],
    ) -> Optional[TriangulationResult]:
        """Triangulate total wall length using multiple methods."""

        result = TriangulationResult(
            quantity_name="Total Wall Length",
            quantity_type="length",
            unit="rmt",
        )

        # Method 1: Sum of room perimeters (walls counted once)
        # Approximate: sum perimeters, divide by 2 for shared walls
        total_perimeter = 0.0
        room_count = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                for room in page.get("rooms", []):
                    # Estimate perimeter from area (assume square-ish)
                    area = room.get("area_sqm", 0)
                    if area > 0:
                        # Perimeter = 4 * sqrt(area) for square
                        # Adjust for typical room aspect ratio (1.5:1)
                        perimeter = 2 * (math.sqrt(area * 1.5) + math.sqrt(area / 1.5))
                        total_perimeter += perimeter
                        room_count += 1

        # Shared walls counted twice, so divide by ~1.5
        estimated_wall_length = total_perimeter / 1.5 if total_perimeter > 0 else 0

        if estimated_wall_length > 0:
            result.methods.append(TriangulationMethod(
                method_name="Room Perimeter Estimate",
                value=round(estimated_wall_length, 2),
                unit="rmt",
                confidence=60,  # Lower confidence - estimate only
                source="room_perimeters",
                notes=f"Estimated from {room_count} room areas"
            ))

        # Method 2: BOQ masonry entries
        boq_wall_length = 0.0
        masonry_items = 0
        for entry in boq_entries:
            item_type = entry.get("item_type", "").lower()
            if any(kw in item_type for kw in ["wall", "masonry", "brick", "block"]):
                qty = entry.get("quantity", 0)
                unit = entry.get("unit", "").lower()
                if unit in ["rmt", "rm", "m", "lm", "running"]:
                    boq_wall_length += qty
                    masonry_items += 1
                elif unit in ["sqm", "sq.m"] and "height" not in entry:
                    # Wall area - assume 3m height to get length
                    boq_wall_length += qty / 3.0
                    masonry_items += 1

        if boq_wall_length > 0:
            result.methods.append(TriangulationMethod(
                method_name="BOQ Masonry Sum",
                value=round(boq_wall_length, 2),
                unit="rmt",
                confidence=85,
                source="boq_entries",
                notes=f"From {masonry_items} masonry line items"
            ))

        # Method 3: Graph wall elements
        wall_elements = project_graph.get("elements", {}).get("walls", [])
        graph_wall_length = sum(w.get("length_m", 0) for w in wall_elements)

        if graph_wall_length > 0:
            result.methods.append(TriangulationMethod(
                method_name="Wall Element Sum",
                value=round(graph_wall_length, 2),
                unit="rmt",
                confidence=75,
                source="wall_elements",
                notes=f"From {len(wall_elements)} wall elements"
            ))

        if len(result.methods) >= 2:
            self._compute_agreement(result)
            return result
        elif len(result.methods) == 1:
            result.final_value = result.methods[0].value
            result.agreement_level = AgreementLevel.FAIR
            result.agreement_score = 60
            result.discrepancy_notes.append("Only one method available - verify manually")
            return result

        return None

    def _triangulate_door_count(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
        schedules: List[Dict],
        boq_entries: List[Dict],
    ) -> Optional[TriangulationResult]:
        """Triangulate door count using multiple methods."""

        result = TriangulationResult(
            quantity_name="Total Door Count",
            quantity_type="count",
            unit="nos",
        )

        # Method 1: Detected doors in floor plans
        detected_doors = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                doors = page.get("doors", [])
                detected_doors += len(doors)
                # Also check elements
                for elem in page.get("elements", []):
                    if elem.get("type", "").lower() == "door":
                        detected_doors += 1

        if detected_doors > 0:
            result.methods.append(TriangulationMethod(
                method_name="Detected Doors",
                value=detected_doors,
                unit="nos",
                confidence=70,
                source="floor_plan_detection",
                notes="Doors detected in floor plan analysis"
            ))

        # Method 2: Door schedule count
        schedule_doors = 0
        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            if "door" in schedule_type:
                entries = schedule.get("entries", [])
                for entry in entries:
                    qty = entry.get("quantity", entry.get("qty", 1))
                    schedule_doors += qty

        if schedule_doors > 0:
            result.methods.append(TriangulationMethod(
                method_name="Door Schedule",
                value=schedule_doors,
                unit="nos",
                confidence=95,  # High confidence - authoritative source
                source="door_schedule",
                notes="From door schedule table"
            ))

        # Method 3: Room-based estimate (heuristic)
        # Typical: each room has 1-2 doors
        room_count = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                room_count += len(page.get("rooms", []))

        if room_count > 0:
            # Heuristic: 1.3 doors per room on average
            estimated_doors = int(room_count * 1.3)
            result.methods.append(TriangulationMethod(
                method_name="Room Heuristic",
                value=estimated_doors,
                unit="nos",
                confidence=40,  # Low confidence - estimate only
                source="room_heuristic",
                notes=f"Estimated ~1.3 doors per {room_count} rooms"
            ))

        # Method 4: BOQ door entries
        boq_doors = 0
        for entry in boq_entries:
            item_type = entry.get("item_type", "").lower()
            if "door" in item_type and "hardware" not in item_type:
                qty = entry.get("quantity", 0)
                unit = entry.get("unit", "").lower()
                if unit in ["nos", "no", "nos.", "numbers", "ea", "each"]:
                    boq_doors += int(qty)

        if boq_doors > 0:
            result.methods.append(TriangulationMethod(
                method_name="BOQ Door Count",
                value=boq_doors,
                unit="nos",
                confidence=90,
                source="boq_entries",
                notes="From BOQ door line items"
            ))

        if len(result.methods) >= 2:
            self._compute_agreement(result)
            return result
        elif len(result.methods) == 1:
            result.final_value = result.methods[0].value
            result.agreement_level = AgreementLevel.FAIR
            result.agreement_score = 60
            result.discrepancy_notes.append("Only one method available - cross-check with door schedule")
            return result

        return None

    def _triangulate_window_count(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
        schedules: List[Dict],
        boq_entries: List[Dict],
    ) -> Optional[TriangulationResult]:
        """Triangulate window count using multiple methods."""

        result = TriangulationResult(
            quantity_name="Total Window Count",
            quantity_type="count",
            unit="nos",
        )

        # Method 1: Detected windows in floor plans
        detected_windows = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                windows = page.get("windows", [])
                detected_windows += len(windows)
                for elem in page.get("elements", []):
                    if elem.get("type", "").lower() == "window":
                        detected_windows += 1

        if detected_windows > 0:
            result.methods.append(TriangulationMethod(
                method_name="Detected Windows",
                value=detected_windows,
                unit="nos",
                confidence=70,
                source="floor_plan_detection",
                notes="Windows detected in floor plan analysis"
            ))

        # Method 2: Window schedule count
        schedule_windows = 0
        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            if "window" in schedule_type:
                entries = schedule.get("entries", [])
                for entry in entries:
                    qty = entry.get("quantity", entry.get("qty", 1))
                    schedule_windows += qty

        if schedule_windows > 0:
            result.methods.append(TriangulationMethod(
                method_name="Window Schedule",
                value=schedule_windows,
                unit="nos",
                confidence=95,
                source="window_schedule",
                notes="From window schedule table"
            ))

        # Method 3: Room-based estimate
        room_count = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                room_count += len(page.get("rooms", []))

        if room_count > 0:
            # Heuristic: 0.8 windows per room (not all rooms have windows)
            estimated_windows = int(room_count * 0.8)
            result.methods.append(TriangulationMethod(
                method_name="Room Heuristic",
                value=estimated_windows,
                unit="nos",
                confidence=35,
                source="room_heuristic",
                notes=f"Estimated ~0.8 windows per {room_count} rooms"
            ))

        # Method 4: BOQ window entries
        boq_windows = 0
        for entry in boq_entries:
            item_type = entry.get("item_type", "").lower()
            if "window" in item_type and "hardware" not in item_type:
                qty = entry.get("quantity", 0)
                unit = entry.get("unit", "").lower()
                if unit in ["nos", "no", "nos.", "numbers", "ea", "each"]:
                    boq_windows += int(qty)

        if boq_windows > 0:
            result.methods.append(TriangulationMethod(
                method_name="BOQ Window Count",
                value=boq_windows,
                unit="nos",
                confidence=90,
                source="boq_entries",
                notes="From BOQ window line items"
            ))

        if len(result.methods) >= 2:
            self._compute_agreement(result)
            return result
        elif len(result.methods) == 1:
            result.final_value = result.methods[0].value
            result.agreement_level = AgreementLevel.FAIR
            result.agreement_score = 60
            result.discrepancy_notes.append("Only one method available - cross-check with window schedule")
            return result

        return None

    def _triangulate_room_count(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
    ) -> Optional[TriangulationResult]:
        """Triangulate room count."""

        result = TriangulationResult(
            quantity_name="Total Room Count",
            quantity_type="count",
            unit="nos",
        )

        # Method 1: Detected rooms in floor plans
        detected_rooms = 0
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                detected_rooms += len(page.get("rooms", []))

        if detected_rooms > 0:
            result.methods.append(TriangulationMethod(
                method_name="Detected Rooms",
                value=detected_rooms,
                unit="nos",
                confidence=85,
                source="floor_plan_detection",
                notes="Rooms detected in floor plan analysis"
            ))

        # Method 2: Project graph room count
        graph_rooms = len(project_graph.get("rooms", []))
        if graph_rooms > 0:
            result.methods.append(TriangulationMethod(
                method_name="Project Graph",
                value=graph_rooms,
                unit="nos",
                confidence=80,
                source="project_graph",
                notes="From joined project graph"
            ))

        if len(result.methods) >= 2:
            self._compute_agreement(result)
            return result
        elif len(result.methods) == 1:
            result.final_value = result.methods[0].value
            result.agreement_level = AgreementLevel.GOOD
            result.agreement_score = 75
            return result

        return None

    def _triangulate_toilet_count(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
        schedules: List[Dict],
    ) -> Optional[TriangulationResult]:
        """Triangulate toilet/WC count (critical for MEP)."""

        result = TriangulationResult(
            quantity_name="Toilet/WC Count",
            quantity_type="count",
            unit="nos",
        )

        # Method 1: Rooms labeled as toilet/WC/bathroom
        toilet_labels = ["toilet", "wc", "bathroom", "bath", "w/c", "w.c", "lavatory", "washroom"]
        detected_toilets = 0

        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                for room in page.get("rooms", []):
                    label = room.get("label", "").lower()
                    if any(t in label for t in toilet_labels):
                        detected_toilets += 1

        if detected_toilets > 0:
            result.methods.append(TriangulationMethod(
                method_name="Labeled Toilets",
                value=detected_toilets,
                unit="nos",
                confidence=90,
                source="room_labels",
                notes="Rooms labeled as toilet/WC/bathroom"
            ))

        # Method 2: Sanitary fixture schedule
        schedule_toilets = 0
        for schedule in schedules:
            schedule_type = schedule.get("type", "").lower()
            if "sanitary" in schedule_type or "fixture" in schedule_type:
                entries = schedule.get("entries", [])
                for entry in entries:
                    item = entry.get("item", entry.get("description", "")).lower()
                    if any(t in item for t in ["wc", "closet", "toilet", "commode"]):
                        qty = entry.get("quantity", entry.get("qty", 1))
                        schedule_toilets += qty

        if schedule_toilets > 0:
            result.methods.append(TriangulationMethod(
                method_name="Sanitary Schedule",
                value=schedule_toilets,
                unit="nos",
                confidence=95,
                source="sanitary_schedule",
                notes="WC count from sanitary fixture schedule"
            ))

        # Method 3: Project graph
        graph_toilets = 0
        for room in project_graph.get("rooms", []):
            label = room.get("label", "").lower()
            if any(t in label for t in toilet_labels):
                graph_toilets += 1

        if graph_toilets > 0:
            result.methods.append(TriangulationMethod(
                method_name="Project Graph",
                value=graph_toilets,
                unit="nos",
                confidence=80,
                source="project_graph",
                notes="From project graph room labels"
            ))

        if len(result.methods) >= 2:
            self._compute_agreement(result)
            return result
        elif len(result.methods) == 1:
            result.final_value = result.methods[0].value
            result.agreement_level = AgreementLevel.GOOD
            result.agreement_score = 75
            return result

        return None

    def _compute_agreement(self, result: TriangulationResult) -> None:
        """Compute agreement level and final value from methods."""

        if not result.methods:
            return

        # Weight by confidence
        total_weight = sum(m.confidence for m in result.methods)
        if total_weight == 0:
            total_weight = 1

        # Weighted average
        weighted_sum = sum(m.value * m.confidence for m in result.methods)
        result.final_value = round(weighted_sum / total_weight, 2)

        # Compute variance
        values = [m.value for m in result.methods]
        mean_value = sum(values) / len(values) if values else 0

        if mean_value > 0:
            max_deviation = max(abs(v - mean_value) for v in values)
            result.variance_pct = round((max_deviation / mean_value) * 100, 1)
        else:
            result.variance_pct = 0

        # Determine agreement level
        if result.variance_pct <= self.THRESHOLDS[AgreementLevel.EXCELLENT]:
            result.agreement_level = AgreementLevel.EXCELLENT
            result.agreement_score = 95
        elif result.variance_pct <= self.THRESHOLDS[AgreementLevel.GOOD]:
            result.agreement_level = AgreementLevel.GOOD
            result.agreement_score = 85
        elif result.variance_pct <= self.THRESHOLDS[AgreementLevel.FAIR]:
            result.agreement_level = AgreementLevel.FAIR
            result.agreement_score = 70
        elif result.variance_pct <= self.THRESHOLDS[AgreementLevel.POOR]:
            result.agreement_level = AgreementLevel.POOR
            result.agreement_score = 50
            result.discrepancy_notes.append(f"Variance {result.variance_pct}% - manual review recommended")
        else:
            result.agreement_level = AgreementLevel.DISCREPANCY
            result.agreement_score = 30
            result.discrepancy_notes.append(f"CRITICAL: Variance {result.variance_pct}% exceeds threshold")

            # Find the most discrepant pair
            for i, m1 in enumerate(result.methods):
                for m2 in result.methods[i+1:]:
                    diff = abs(m1.value - m2.value)
                    if mean_value > 0:
                        diff_pct = (diff / mean_value) * 100
                        if diff_pct > 15:
                            result.discrepancy_notes.append(
                                f"{m1.method_name} ({m1.value}) vs {m2.method_name} ({m2.value}): {diff_pct:.1f}% difference"
                            )

        # Set recommended action
        if result.agreement_level in [AgreementLevel.POOR, AgreementLevel.DISCREPANCY]:
            # Prefer schedule data if available
            schedule_methods = [m for m in result.methods if "schedule" in m.source.lower()]
            if schedule_methods:
                result.recommended_action = f"Use {schedule_methods[0].method_name} value and verify others"
            else:
                result.recommended_action = "Manual measurement verification required"
        elif result.agreement_level == AgreementLevel.FAIR:
            result.recommended_action = "Spot-check recommended before final submission"
        else:
            result.recommended_action = "Values consistent - proceed with confidence"

    def _build_report(self, project_id: str) -> TriangulationReport:
        """Build final triangulation report."""

        report = TriangulationReport(project_id=project_id)
        report.results = self.results

        # Overall agreement
        if self.results:
            total_score = sum(r.agreement_score for r in self.results)
            report.overall_agreement = round(total_score / len(self.results), 1)

        # Critical discrepancies
        for result in self.results:
            if result.agreement_level == AgreementLevel.DISCREPANCY:
                report.critical_discrepancies.append(
                    f"{result.quantity_name}: {result.variance_pct}% variance"
                )

        # Summary stats
        level_counts = {}
        for result in self.results:
            level = result.agreement_level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        report.summary = {
            "total_quantities": len(self.results),
            "agreement_breakdown": level_counts,
            "critical_discrepancies": len(report.critical_discrepancies),
            "overall_score": report.overall_agreement,
            "recommendation": self._get_overall_recommendation(report),
        }

        return report

    def _get_overall_recommendation(self, report: TriangulationReport) -> str:
        """Get overall recommendation based on triangulation results."""

        if report.overall_agreement >= 85:
            return "Quantities consistent - high confidence for bid submission"
        elif report.overall_agreement >= 70:
            return "Minor discrepancies - review flagged items before submission"
        elif report.overall_agreement >= 50:
            return "Significant discrepancies - detailed review required"
        else:
            return "CAUTION: Major quantity discrepancies - do not proceed without resolution"


def run_triangulation(
    project_id: str,
    extraction_results: List[Dict],
    project_graph: Dict,
    boq_entries: List[Dict],
    schedules: List[Dict],
) -> TriangulationReport:
    """
    Run triangulation engine on project data.

    Args:
        project_id: Project identifier
        extraction_results: Page extraction results
        project_graph: Joined project graph
        boq_entries: BOQ quantity entries
        schedules: Extracted schedule tables

    Returns:
        Complete triangulation report
    """
    engine = TriangulationEngine()
    return engine.triangulate_project(
        project_id=project_id,
        extraction_results=extraction_results,
        project_graph=project_graph,
        boq_entries=boq_entries,
        schedules=schedules,
    )
