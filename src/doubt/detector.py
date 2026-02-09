"""
Doubt Detector - Detect missing critical sheets.

Analyzes drawing sets against expected sheet types for Indian construction projects.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum


class SheetSeverity(Enum):
    """Severity of missing sheet."""
    CRITICAL = "critical"
    IMPORTANT = "important"
    OPTIONAL = "optional"


@dataclass
class MissingSheet:
    """Missing sheet type."""
    sheet_type: str
    severity: str
    why_needed: str
    impact: str
    expected_content: List[str] = field(default_factory=list)
    impacted_packages: List[str] = field(default_factory=list)
    evidence_triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sheet_type": self.sheet_type,
            "severity": self.severity,
            "why_needed": self.why_needed,
            "impact": self.impact,
            "expected_content": self.expected_content,
            "impacted_packages": self.impacted_packages,
            "evidence_triggers": self.evidence_triggers,
        }


@dataclass
class DoubtReport:
    """Report of missing sheets and drawing set completeness."""
    missing_sheets: List[MissingSheet] = field(default_factory=list)
    present_types: Dict[str, int] = field(default_factory=dict)
    completeness_score: float = 0.0
    completeness_grade: str = "F"
    critical_count: int = 0
    important_count: int = 0
    optional_count: int = 0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "missing_sheets": [m.to_dict() for m in self.missing_sheets],
            "present_types": self.present_types,
            "completeness_score": self.completeness_score,
            "completeness_grade": self.completeness_grade,
            "critical_count": self.critical_count,
            "important_count": self.important_count,
            "optional_count": self.optional_count,
            "recommendations": self.recommendations,
        }


class DoubtDetector:
    """Detect missing critical sheets in drawing sets."""

    # Expected sheet types for complete Indian construction drawing set
    EXPECTED_SHEETS = {
        # Architectural - Critical
        "floor_plan": {
            "severity": "critical",
            "why_needed": "Floor plans are essential for room areas, wall lengths, and spatial layout",
            "impact": "Cannot estimate floor finishes, wall finishes, doors, windows",
            "keywords": ["floor plan", "layout plan", "plan", "ground floor", "first floor"],
            "impacted_packages": ["finishes_floor", "finishes_wall", "doors_windows", "masonry_partitions"],
        },
        "site_plan": {
            "severity": "critical",
            "why_needed": "Site plan needed for external works, paving, compound wall, drainage",
            "impact": "Cannot estimate external works, parking, landscaping",
            "keywords": ["site plan", "layout", "key plan", "location plan"],
            "impacted_packages": ["external_works", "earthwork", "drainage"],
        },
        "section": {
            "severity": "critical",
            "why_needed": "Sections show floor heights, slab thicknesses, staircase dimensions",
            "impact": "Cannot verify heights, estimate plaster, staircase quantities",
            "keywords": ["section", "cross section", "longitudinal section", "building section"],
            "impacted_packages": ["rcc_concrete", "finishes_wall", "staircase"],
        },
        "elevation": {
            "severity": "critical",
            "why_needed": "Elevations show external finishes, window/door heights, facade details",
            "impact": "Cannot estimate external paint, facade, window/door sizes",
            "keywords": ["elevation", "front elevation", "rear elevation", "side elevation"],
            "impacted_packages": ["finishes_wall", "doors_windows", "external_works"],
        },

        # Structural - Critical
        "foundation_plan": {
            "severity": "critical",
            "why_needed": "Foundation plan needed for footing sizes, plinth beam layout",
            "impact": "Cannot estimate foundation concrete, steel, excavation",
            "keywords": ["foundation", "footing", "plinth", "substructure"],
            "impacted_packages": ["earthwork", "rcc_concrete", "reinforcement_steel"],
        },
        "structural_plan": {
            "severity": "critical",
            "why_needed": "Structural plans show column, beam grid and slab layout",
            "impact": "Cannot estimate RCC, formwork, steel quantities",
            "keywords": ["structural", "framing plan", "column layout", "beam layout", "slab layout"],
            "impacted_packages": ["rcc_concrete", "reinforcement_steel", "formwork_shuttering"],
        },

        # MEP - Important
        "plumbing_layout": {
            "severity": "important",
            "why_needed": "Plumbing layout shows pipe runs, fixture locations, shaft sizes",
            "impact": "Cannot estimate plumbing pipes, fittings, fixtures",
            "keywords": ["plumbing", "water supply", "drainage", "sanitary", "swr"],
            "impacted_packages": ["plumbing_water_supply", "plumbing_sanitary_swr"],
        },
        "electrical_layout": {
            "severity": "important",
            "why_needed": "Electrical layout shows wiring points, panel locations, conduit runs",
            "impact": "Cannot estimate electrical wiring, points, panels",
            "keywords": ["electrical", "power", "lighting", "wiring", "db"],
            "impacted_packages": ["electrical_power_lighting"],
        },
        "fire_layout": {
            "severity": "important",
            "why_needed": "Fire layout shows hydrant, sprinkler, alarm locations",
            "impact": "Cannot estimate fire fighting system",
            "keywords": ["fire", "sprinkler", "hydrant", "fire alarm"],
            "impacted_packages": ["fire_hvac"],
        },

        # Schedules - Important
        "door_schedule": {
            "severity": "important",
            "why_needed": "Door schedule shows door types, sizes, materials, hardware",
            "impact": "Cannot verify door quantities and specifications",
            "keywords": ["door schedule", "door table"],
            "impacted_packages": ["doors_windows"],
        },
        "window_schedule": {
            "severity": "important",
            "why_needed": "Window schedule shows window types, sizes, materials",
            "impact": "Cannot verify window quantities and specifications",
            "keywords": ["window schedule", "window table"],
            "impacted_packages": ["doors_windows"],
        },
        "finish_schedule": {
            "severity": "important",
            "why_needed": "Finish schedule shows room-wise floor, wall, ceiling finishes",
            "impact": "Cannot determine finish specifications per room",
            "keywords": ["finish schedule", "room schedule", "room data"],
            "impacted_packages": ["finishes_floor", "finishes_wall", "finishes_ceiling"],
        },
        "column_schedule": {
            "severity": "important",
            "why_needed": "Column schedule shows column sizes, reinforcement details",
            "impact": "Cannot estimate column concrete and steel",
            "keywords": ["column schedule", "column detail"],
            "impacted_packages": ["rcc_concrete", "reinforcement_steel"],
        },

        # Details - Important
        "toilet_detail": {
            "severity": "important",
            "why_needed": "Toilet details show waterproofing, tiling, CP fixtures layout",
            "impact": "Cannot verify waterproofing scope and specifications",
            "keywords": ["toilet detail", "bathroom detail", "wc detail"],
            "impacted_packages": ["waterproofing", "tiles_finishes", "plumbing_sanitary_swr"],
        },
        "staircase_detail": {
            "severity": "important",
            "why_needed": "Staircase details show tread/riser dimensions, railing specs",
            "impact": "Cannot estimate staircase finishes and railing",
            "keywords": ["stair detail", "staircase detail", "step detail"],
            "impacted_packages": ["rcc_concrete", "finishes_floor", "miscellaneous"],
        },
        "terrace_detail": {
            "severity": "important",
            "why_needed": "Terrace details show waterproofing treatment, slope, weathering course",
            "impact": "Cannot verify terrace waterproofing scope",
            "keywords": ["terrace detail", "roof detail", "waterproofing detail"],
            "impacted_packages": ["waterproofing", "external_works"],
        },

        # Optional
        "reflected_ceiling_plan": {
            "severity": "optional",
            "why_needed": "RCP shows false ceiling layout, lighting positions",
            "impact": "May miss ceiling details",
            "keywords": ["rcp", "reflected ceiling", "ceiling plan"],
            "impacted_packages": ["finishes_ceiling", "electrical_power_lighting"],
        },
        "landscaping_plan": {
            "severity": "optional",
            "why_needed": "Landscaping plan shows planting, hardscape, water features",
            "impact": "May miss landscaping scope",
            "keywords": ["landscape", "planting", "garden"],
            "impacted_packages": ["external_works"],
        },
    }

    # Sheet type detection patterns
    SHEET_TYPE_PATTERNS = {
        "floor_plan": [
            r'floor\s+plan',
            r'layout\s+plan',
            r'ground\s+floor',
            r'first\s+floor',
            r'(?:gf|ff|sf|tf)\s+plan',
            r'typical\s+floor',
            r'basement\s+plan',
        ],
        "site_plan": [
            r'site\s+plan',
            r'key\s+plan',
            r'location\s+plan',
            r'site\s+layout',
            r'index\s+plan',
        ],
        "section": [
            r'section',
            r'cross\s+section',
            r'longitudinal\s+section',
            r'building\s+section',
            r'sectional\s+elevation',
        ],
        "elevation": [
            r'elevation',
            r'front\s+elevation',
            r'rear\s+elevation',
            r'side\s+elevation',
            r'external\s+elevation',
        ],
        "foundation_plan": [
            r'foundation',
            r'footing',
            r'plinth\s+plan',
            r'substructure',
        ],
        "structural_plan": [
            r'structural',
            r'framing',
            r'column\s+layout',
            r'beam\s+layout',
            r'slab\s+layout',
            r'rcc\s+plan',
        ],
        "plumbing_layout": [
            r'plumbing',
            r'water\s+supply',
            r'drainage',
            r'sanitary',
            r'swr\s+layout',
            r'plumbing\s+layout',
        ],
        "electrical_layout": [
            r'electrical',
            r'power\s+layout',
            r'lighting\s+layout',
            r'wiring\s+layout',
            r'single\s+line',
        ],
        "fire_layout": [
            r'fire\s+fighting',
            r'sprinkler',
            r'fire\s+alarm',
            r'fire\s+layout',
        ],
        "door_schedule": [
            r'door\s+schedule',
            r'door\s+table',
            r'door\s+details',
        ],
        "window_schedule": [
            r'window\s+schedule',
            r'window\s+table',
            r'window\s+details',
        ],
        "finish_schedule": [
            r'finish\s+schedule',
            r'room\s+schedule',
            r'room\s+data',
            r'finishes\s+schedule',
        ],
        "column_schedule": [
            r'column\s+schedule',
            r'column\s+details',
            r'col\.\s+schedule',
        ],
        "toilet_detail": [
            r'toilet\s+detail',
            r'bathroom\s+detail',
            r'wc\s+detail',
            r'toilet\s+enlarged',
        ],
        "staircase_detail": [
            r'stair\s+detail',
            r'staircase\s+detail',
            r'step\s+detail',
            r'tread\s+riser',
        ],
        "terrace_detail": [
            r'terrace\s+detail',
            r'roof\s+detail',
            r'waterproofing\s+detail',
            r'weathering\s+course',
        ],
        "reflected_ceiling_plan": [
            r'rcp',
            r'reflected\s+ceiling',
            r'ceiling\s+plan',
            r'false\s+ceiling\s+layout',
        ],
        "landscaping_plan": [
            r'landscape',
            r'planting\s+plan',
            r'garden\s+layout',
            r'softscape',
        ],
    }

    def __init__(self):
        self.compiled_patterns = {}
        for sheet_type, patterns in self.SHEET_TYPE_PATTERNS.items():
            self.compiled_patterns[sheet_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def analyze(
        self,
        page_index: List[Dict],
        routing_manifest: Dict,
        extraction_results: List[Dict],
        scope_register: Dict,
    ) -> DoubtReport:
        """Analyze drawing set for missing sheets."""
        report = DoubtReport()

        # Detect present sheet types
        present_types = self._detect_present_types(page_index, extraction_results)
        report.present_types = present_types

        # Check for missing sheets
        for sheet_type, sheet_info in self.EXPECTED_SHEETS.items():
            if sheet_type not in present_types:
                # Check if sheet is actually needed based on scope
                if self._is_sheet_needed(sheet_type, scope_register, extraction_results):
                    report.missing_sheets.append(MissingSheet(
                        sheet_type=sheet_type,
                        severity=sheet_info["severity"],
                        why_needed=sheet_info["why_needed"],
                        impact=sheet_info["impact"],
                        impacted_packages=sheet_info.get("impacted_packages", []),
                    ))

        # Count by severity
        for missing in report.missing_sheets:
            if missing.severity == "critical":
                report.critical_count += 1
            elif missing.severity == "important":
                report.important_count += 1
            else:
                report.optional_count += 1

        # Calculate completeness score
        total_critical = len([s for s, i in self.EXPECTED_SHEETS.items()
                              if i["severity"] == "critical"])
        total_important = len([s for s, i in self.EXPECTED_SHEETS.items()
                               if i["severity"] == "important"])

        present_critical = total_critical - report.critical_count
        present_important = total_important - report.important_count

        # Weighted score (critical = 60%, important = 40%)
        critical_score = (present_critical / total_critical) * 60 if total_critical > 0 else 60
        important_score = (present_important / total_important) * 40 if total_important > 0 else 40

        report.completeness_score = round(critical_score + important_score, 0)

        # Assign grade
        if report.completeness_score >= 90:
            report.completeness_grade = "A"
        elif report.completeness_score >= 75:
            report.completeness_grade = "B"
        elif report.completeness_score >= 60:
            report.completeness_grade = "C"
        elif report.completeness_score >= 40:
            report.completeness_grade = "D"
        else:
            report.completeness_grade = "F"

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _detect_present_types(
        self,
        page_index: List[Dict],
        extraction_results: List[Dict],
    ) -> Dict[str, int]:
        """Detect which sheet types are present."""
        present = {}

        # Check page index
        for page in page_index:
            sheet_type = self._classify_page(page)
            if sheet_type:
                present[sheet_type] = present.get(sheet_type, 0) + 1

        # Also check extraction results
        for result in extraction_results:
            page_type = result.get("page_type", "")
            if page_type:
                # Map page_type to our sheet types
                mapped_type = self._map_page_type(page_type)
                if mapped_type:
                    if mapped_type not in present:
                        present[mapped_type] = 0
                    present[mapped_type] += 1

            # Check schedules
            if result.get("schedules"):
                for schedule in result["schedules"]:
                    schedule_type = schedule.get("type", "")
                    if "door" in schedule_type.lower():
                        present["door_schedule"] = present.get("door_schedule", 0) + 1
                    elif "window" in schedule_type.lower():
                        present["window_schedule"] = present.get("window_schedule", 0) + 1
                    elif "column" in schedule_type.lower():
                        present["column_schedule"] = present.get("column_schedule", 0) + 1
                    elif "finish" in schedule_type.lower():
                        present["finish_schedule"] = present.get("finish_schedule", 0) + 1

        return present

    def _classify_page(self, page: Dict) -> Optional[str]:
        """Classify a page into sheet type."""
        # Collect text from page
        texts = []
        texts.append(page.get("sheet_name", ""))
        texts.append(page.get("drawing_title", ""))
        texts.append(page.get("title", ""))

        title_block = page.get("title_block", {})
        if title_block:
            texts.append(title_block.get("drawing_title", ""))
            texts.append(title_block.get("sheet_name", ""))

        combined = " ".join(filter(None, texts)).lower()

        # Match against patterns
        for sheet_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(combined):
                    return sheet_type

        return None

    def _map_page_type(self, page_type: str) -> Optional[str]:
        """Map extraction page_type to our sheet types."""
        mapping = {
            "floor_plan": "floor_plan",
            "site_plan": "site_plan",
            "section": "section",
            "elevation": "elevation",
            "structural_plan": "structural_plan",
            "foundation_plan": "foundation_plan",
            "plumbing": "plumbing_layout",
            "electrical": "electrical_layout",
            "schedule_table": None,  # Handle separately
            "detail": None,  # Handle in detail module
        }
        return mapping.get(page_type.lower())

    def _is_sheet_needed(
        self,
        sheet_type: str,
        scope_register: Dict,
        extraction_results: List[Dict],
    ) -> bool:
        """Check if sheet is actually needed based on project scope."""
        # All critical sheets are always needed
        if self.EXPECTED_SHEETS.get(sheet_type, {}).get("severity") == "critical":
            return True

        # Check scope register for related packages
        impacted_packages = self.EXPECTED_SHEETS.get(sheet_type, {}).get("impacted_packages", [])

        scope_items = scope_register.get("items", [])
        if isinstance(scope_items, dict):
            scope_items = list(scope_items.values())

        for item in scope_items:
            item_package = item.get("package", "").lower()
            for pkg in impacted_packages:
                if pkg in item_package:
                    return True

        return True  # Default to needed

    def _generate_recommendations(self, report: DoubtReport) -> List[str]:
        """Generate recommendations based on missing sheets."""
        recommendations = []

        if report.critical_count > 0:
            critical_types = [m.sheet_type for m in report.missing_sheets
                             if m.severity == "critical"]
            recommendations.append(
                f"⚠️ CRITICAL: {report.critical_count} essential sheet types are missing: "
                f"{', '.join(critical_types)}. Request these immediately before proceeding."
            )

        if report.important_count > 0:
            recommendations.append(
                f"Request {report.important_count} important sheet types to improve estimate accuracy."
            )

        # Specific recommendations
        for missing in report.missing_sheets:
            if missing.sheet_type == "site_plan":
                recommendations.append(
                    "Without site plan: Assume scope for external works, compound wall, paving, "
                    "and drainage based on typical requirements."
                )
            elif missing.sheet_type == "section":
                recommendations.append(
                    "Without sections: Assume standard floor heights (3.0m clear, 150mm slab). "
                    "Verify with architect before finalizing."
                )
            elif missing.sheet_type == "plumbing_layout":
                recommendations.append(
                    "Without plumbing layout: Estimate pipe lengths using typical factors "
                    "(1.2x fixture point count). Add contingency."
                )
            elif missing.sheet_type == "electrical_layout":
                recommendations.append(
                    "Without electrical layout: Use standard point density "
                    "(1 point per 4 sqm for residential). Verify with MEP consultant."
                )

        if report.completeness_score < 60:
            recommendations.append(
                "Drawing set is significantly incomplete. Consider flagging this as a "
                "preliminary estimate only, subject to revision when complete drawings are available."
            )

        return recommendations
