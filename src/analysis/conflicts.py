"""
XBOQ Conflict Detection Module
Detects and documents conflicts with specific evidence and resolution steps.

Conflicts are precise, cite evidence, and provide actionable next steps.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models.estimate_schema import (
    Conflict,
    ConflictType,
    Evidence,
    EvidenceSource,
    Severity,
    create_conflict,
    create_evidence,
)


@dataclass
class ConflictContext:
    """Context for conflict detection."""
    pdf_text: str = ""
    notes_text: str = ""
    scale_detected: Optional[str] = None
    concrete_grades: Set[str] = field(default_factory=set)
    steel_grades: Set[str] = field(default_factory=set)
    schedule_column_marks: Set[str] = field(default_factory=set)
    plan_column_marks: Set[str] = field(default_factory=set)
    schedule_footing_marks: Set[str] = field(default_factory=set)
    plan_footing_marks: Set[str] = field(default_factory=set)
    rebar_patterns: List[Tuple[str, int]] = field(default_factory=list)  # (pattern, page)
    has_schedule_sheet: bool = False
    has_plan_sheet: bool = False
    page_count: int = 1


class ConflictDetector:
    """
    Detects conflicts with specific evidence and actionable resolutions.
    """

    def __init__(self, context: ConflictContext):
        self.context = context
        self.conflicts: List[Conflict] = []

    def detect_all(self) -> List[Conflict]:
        """Run all conflict detection checks."""
        self._check_missing_scale()
        self._check_grade_mismatch()
        self._check_schedule_vs_plan_marks()
        self._check_ambiguous_rebar_patterns()
        self._check_missing_schedules()
        self._check_missing_labels()
        self._check_dimension_issues()

        return self.conflicts

    def _check_missing_scale(self):
        """Check for missing or ambiguous scale."""
        if not self.context.scale_detected:
            # Look for scale mentions in text
            scale_patterns = [
                r'scale[:\s]*1\s*[:]\s*\d+',
                r'1\s*[:]\s*\d+\s*scale',
                r'NTS',  # Not to scale
            ]

            found_mentions = []
            for pattern in scale_patterns:
                matches = re.findall(pattern, self.context.pdf_text.lower())
                found_mentions.extend(matches)

            if found_mentions:
                # Found mentions but couldn't parse
                snippet = f"Found scale references: {', '.join(found_mentions[:3])}"
                description = f"Scale note found but could not be parsed: '{found_mentions[0]}'. Cannot compute accurate quantities without confirmed scale."
            else:
                snippet = "No scale note found in drawing"
                description = "Scale note not found anywhere in the drawing. Cannot compute quantities from dimensions without known scale."

            self.conflicts.append(create_conflict(
                conflict_type="missing_scale",
                description=description,
                severity="high",
                suggested_resolution="Add scale note to drawing OR manually input scale in settings. Check title block area for scale information.",
                evidence=[create_evidence(
                    page=0,
                    source="pdf_text",
                    snippet=snippet,
                )],
            ))

    def _check_grade_mismatch(self):
        """Check for conflicting concrete/steel grades."""
        # Concrete grades
        if len(self.context.concrete_grades) > 1:
            grades_list = sorted(self.context.concrete_grades)
            snippet = f"Multiple concrete grades found: {', '.join(grades_list)}"

            self.conflicts.append(create_conflict(
                conflict_type="grade_mismatch",
                description=f"Multiple concrete grades detected: {', '.join(grades_list)}. Different elements may use different grades, but this needs verification.",
                severity="med",
                suggested_resolution=f"Verify which grade applies to which elements. Common: {grades_list[0]} for footings, higher grade for columns. Check general notes section.",
                evidence=[create_evidence(
                    page=0,
                    source="pdf_text",
                    snippet=snippet,
                )],
            ))

        # Steel grades
        if len(self.context.steel_grades) > 1:
            grades_list = sorted(self.context.steel_grades)
            snippet = f"Multiple steel grades found: {', '.join(grades_list)}"

            self.conflicts.append(create_conflict(
                conflict_type="grade_mismatch",
                description=f"Multiple steel grades detected: {', '.join(grades_list)}. Verify grade for main bars vs ties.",
                severity="low",
                suggested_resolution="Typically Fe500/Fe500D for main bars, Fe500 for ties. Check bar schedule header.",
                evidence=[create_evidence(
                    page=0,
                    source="pdf_text",
                    snippet=snippet,
                )],
            ))

    def _check_schedule_vs_plan_marks(self):
        """Check for column/footing marks mismatch between schedule and plan."""
        # Column marks
        if self.context.schedule_column_marks and self.context.plan_column_marks:
            schedule_only = self.context.schedule_column_marks - self.context.plan_column_marks
            plan_only = self.context.plan_column_marks - self.context.schedule_column_marks

            if schedule_only:
                marks_sample = sorted(list(schedule_only))[:5]
                snippet = f"Schedule contains: {', '.join(marks_sample)}... but not found in plan extraction"

                self.conflicts.append(create_conflict(
                    conflict_type="schedule_vs_plan",
                    description=f"Column marks in schedule not found in plan: {', '.join(marks_sample)}{'...' if len(schedule_only) > 5 else ''}. Either plan sheet is different or extraction missed them.",
                    severity="med",
                    suggested_resolution="Verify correct sheet uploaded. Upload column layout plan sheet if separate. Check if marks are in a different format on plan.",
                    evidence=[create_evidence(
                        page=0,
                        source="camelot",
                        snippet=snippet,
                    )],
                ))

            if plan_only and len(plan_only) > 3:
                marks_sample = sorted(list(plan_only))[:5]
                snippet = f"Plan contains: {', '.join(marks_sample)}... but not in schedule"

                self.conflicts.append(create_conflict(
                    conflict_type="schedule_vs_plan",
                    description=f"Column marks in plan not found in schedule: {', '.join(marks_sample)}{'...' if len(plan_only) > 5 else ''}. Schedule may be incomplete.",
                    severity="med",
                    suggested_resolution="Check if additional schedule sheets exist. Some columns may share reinforcement details with others.",
                    evidence=[create_evidence(
                        page=0,
                        source="heuristic",
                        snippet=snippet,
                    )],
                ))

        # If schedule has marks but no plan marks detected
        elif self.context.schedule_column_marks and not self.context.plan_column_marks:
            marks_sample = sorted(list(self.context.schedule_column_marks))[:5]
            self.conflicts.append(create_conflict(
                conflict_type="schedule_vs_plan",
                description=f"Schedule contains column marks ({', '.join(marks_sample)}...) but no column layout plan detected. Cannot determine column counts.",
                severity="high",
                suggested_resolution="Upload column layout plan sheet showing column positions on grid. This is needed for quantity takeoff.",
                evidence=[create_evidence(
                    page=0,
                    source="camelot",
                    snippet=f"Schedule marks: {', '.join(marks_sample)}",
                )],
            ))

        # Footing marks (similar logic)
        if self.context.schedule_footing_marks and not self.context.plan_footing_marks:
            marks_sample = sorted(list(self.context.schedule_footing_marks))[:5]
            self.conflicts.append(create_conflict(
                conflict_type="schedule_vs_plan",
                description=f"Footing schedule marks ({', '.join(marks_sample)}) found but no footing layout plan detected.",
                severity="med",
                suggested_resolution="Upload footing layout plan sheet for accurate footing counts.",
                evidence=[create_evidence(
                    page=0,
                    source="camelot",
                    snippet=f"Footing marks: {', '.join(marks_sample)}",
                )],
            ))

    def _check_ambiguous_rebar_patterns(self):
        """Check for ambiguous rebar patterns like '18-25'."""
        ambiguous_patterns = []

        for pattern, page in self.context.rebar_patterns:
            # Patterns like "X-Y" where both could be qty or dia
            match = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', pattern.strip())
            if match:
                val1, val2 = int(match.group(1)), int(match.group(2))
                # Both could reasonably be either
                if val1 > 10 and val2 > 10 and val1 < 40 and val2 < 40:
                    ambiguous_patterns.append((pattern, page, val1, val2))

        if ambiguous_patterns:
            pattern, page, v1, v2 = ambiguous_patterns[0]
            self.conflicts.append(create_conflict(
                conflict_type="ambiguous_mark",
                description=f"Rebar pattern '{pattern}' is ambiguous: could be {v1} bars of {v2}mm OR {v1}mm dia at spacing. Check schedule header for format.",
                severity="med",
                suggested_resolution=f"Look at schedule column headers. Common format is 'qty-dia' (e.g., 18 bars of 25mm). Verify by checking if {v2}mm is a standard diameter (8,10,12,16,20,25,28,32).",
                evidence=[create_evidence(
                    page=page,
                    source="camelot",
                    snippet=f"Found pattern: {pattern}",
                )],
            ))

    def _check_missing_schedules(self):
        """Check for missing schedule sheets."""
        if not self.context.has_schedule_sheet:
            self.conflicts.append(create_conflict(
                conflict_type="missing_labels",
                description="No reinforcement schedule table detected. Cannot extract bar details without schedule.",
                severity="high",
                suggested_resolution="Upload sheet containing 'COLUMN REINFORCEMENT SCHEDULE', 'BEAM SCHEDULE', or 'BAR BENDING SCHEDULE'. Schedule sheets typically have tabular data with column marks, sections, and rebar details.",
                evidence=[create_evidence(
                    page=0,
                    source="pdf_text",
                    snippet="No schedule keywords found in document",
                )],
            ))

    def _check_missing_labels(self):
        """Check for missing critical labels."""
        # Check for grid labels
        grid_pattern = r'[A-Z]\s*[-–]\s*[A-Z]|[1-9]\s*[-–]\s*\d+'
        has_grid = bool(re.search(grid_pattern, self.context.pdf_text))

        if not has_grid and self.context.has_plan_sheet:
            self.conflicts.append(create_conflict(
                conflict_type="missing_labels",
                description="Grid lines (A-B, 1-2, etc.) not clearly detected in plan. Column positions may be uncertain.",
                severity="low",
                suggested_resolution="Ensure grid labels are clearly visible in the plan. Check if plan has grid references.",
                evidence=[create_evidence(
                    page=0,
                    source="pdf_text",
                    snippet="No grid pattern like 'A-B' or '1-2' found",
                )],
            ))

    def _check_dimension_issues(self):
        """Check for dimension-related issues."""
        # Look for dimension patterns
        dim_pattern = r'\d{3,4}\s*[xX×]\s*\d{3,4}'
        dimensions = re.findall(dim_pattern, self.context.pdf_text)

        if not dimensions and self.context.has_plan_sheet:
            self.conflicts.append(create_conflict(
                conflict_type="dimension_mismatch",
                description="No clear dimension patterns (e.g., 300x450) detected. Element sizes may not be extractable.",
                severity="low",
                suggested_resolution="Check if dimensions are in schedule tables rather than on plan. Dimension annotations may use different format.",
                evidence=[create_evidence(
                    page=0,
                    source="pdf_text",
                    snippet="No dimension pattern like '300x450' found in text",
                )],
            ))


def detect_conflicts(
    pdf_text: str = "",
    notes_text: str = "",
    scale: Optional[str] = None,
    column_schedules: Optional[List[Dict]] = None,
    columns_detected: Optional[List[Dict]] = None,
    footing_schedules: Optional[List[Dict]] = None,
    footings_detected: Optional[List[Dict]] = None,
    has_schedule_tables: bool = False,
    has_plan_view: bool = False,
) -> List[Conflict]:
    """
    Detect conflicts in extracted data.

    Args:
        pdf_text: Full text from PDF
        notes_text: Notes section text
        scale: Detected scale or None
        column_schedules: Column schedule data from tables
        columns_detected: Columns detected from plan
        footing_schedules: Footing schedule data
        footings_detected: Footings detected from plan
        has_schedule_tables: Whether schedule tables were found
        has_plan_view: Whether plan view was detected

    Returns:
        List of Conflict objects
    """
    # Build context
    context = ConflictContext(
        pdf_text=pdf_text,
        notes_text=notes_text,
        scale_detected=scale,
        has_schedule_sheet=has_schedule_tables,
        has_plan_sheet=has_plan_view,
    )

    # Extract concrete grades
    grade_matches = re.findall(r'M\s*(\d{2,3})', pdf_text, re.IGNORECASE)
    for g in grade_matches:
        context.concrete_grades.add(f"M{g}")

    # Extract steel grades
    if "fe500d" in pdf_text.lower():
        context.steel_grades.add("Fe500D")
    if "fe500" in pdf_text.lower() and "fe500d" not in pdf_text.lower():
        context.steel_grades.add("Fe500")
    if "fe415" in pdf_text.lower():
        context.steel_grades.add("Fe415")

    # Extract column marks from schedules
    if column_schedules:
        for sched in column_schedules:
            marks = sched.get("column_marks", [])
            context.schedule_column_marks.update(marks)

    # Extract column marks from plan
    if columns_detected:
        for col in columns_detected:
            mark = col.get("mark", "")
            if mark:
                context.plan_column_marks.add(mark)

    # Extract footing marks
    if footing_schedules:
        for sched in footing_schedules:
            marks = sched.get("footing_marks", [])
            context.schedule_footing_marks.update(marks)

    if footings_detected:
        for ftg in footings_detected:
            mark = ftg.get("mark", "")
            if mark:
                context.plan_footing_marks.add(mark)

    # Look for rebar patterns
    rebar_patterns = re.findall(r'\b(\d{1,2}\s*[-–]\s*\d{2})\b', pdf_text)
    for p in rebar_patterns:
        context.rebar_patterns.append((p, 0))

    # Run detector
    detector = ConflictDetector(context)
    return detector.detect_all()
