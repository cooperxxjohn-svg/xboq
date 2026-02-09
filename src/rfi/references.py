"""
Missing Reference Detection for XBOQ RFI System.

Parses references like "Detail 3/S-402" and verifies that
referenced sheets exist. Creates high priority RFIs for
missing references.

India-specific construction estimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """Detected cross-reference in drawings."""
    reference_id: str
    reference_type: str  # detail, section, schedule, drawing
    reference_text: str  # Original text (e.g., "Detail 3/S-402")
    target_sheet: str  # Sheet ID being referenced
    target_item: str  # Item on sheet (e.g., "Detail 3")
    source_page: str  # Page where reference was found
    source_context: str  # Surrounding text for context
    exists: bool  # Whether target sheet exists
    confidence: float


@dataclass
class MissingReference:
    """A reference to a non-existent sheet/detail."""
    reference: Reference
    severity: str  # high, medium, low
    impact: str  # Description of estimation impact
    suggestion: str  # Suggested resolution


@dataclass
class ReferenceReport:
    """Complete reference analysis report."""
    project_id: str
    all_references: List[Reference] = field(default_factory=list)
    missing_references: List[MissingReference] = field(default_factory=list)
    present_sheets: Set[str] = field(default_factory=set)
    summary: Dict[str, Any] = field(default_factory=dict)


class ReferenceDetector:
    """
    Detects and validates cross-references in drawing sets.

    Parses references to details, sections, and schedules,
    and verifies that referenced sheets exist.
    """

    # Reference patterns commonly used in Indian drawings
    DETAIL_PATTERNS = [
        # "Detail 3/S-402" or "DET.3/S-402"
        re.compile(r"(?:Detail|DET|det)[.\s-]*(\d+)[/\\](\w+-?\d+)", re.IGNORECASE),
        # "See detail 3 on S-402"
        re.compile(r"See\s+(?:detail|det)[.\s]*(\d+)\s+(?:on|in)\s+(\w+-?\d+)", re.IGNORECASE),
        # "Refer detail A on sheet S-402"
        re.compile(r"Refer\s+(?:detail|det)[.\s]*([A-Za-z0-9]+)\s+(?:on|in)\s+(?:sheet)?\s*(\w+-?\d+)", re.IGNORECASE),
        # "D3/S-402" short form
        re.compile(r"D(\d+)[/\\](\w+-?\d+)", re.IGNORECASE),
    ]

    SECTION_PATTERNS = [
        # "Section A-A/S-301"
        re.compile(r"(?:Section|SEC|sec)[.\s-]*([A-Z])-?([A-Z])?[/\\](\w+-?\d+)", re.IGNORECASE),
        # "See section A-A on S-301"
        re.compile(r"See\s+(?:section|sec)[.\s]*([A-Z])-?([A-Z])?\s+(?:on|in)\s+(\w+-?\d+)", re.IGNORECASE),
        # "S/A-S-301" short form
        re.compile(r"S/([A-Z])[/\\-](\w+-?\d+)", re.IGNORECASE),
    ]

    SCHEDULE_PATTERNS = [
        # "See door schedule" or "Refer window schedule"
        re.compile(r"(?:See|Refer)\s+(?:to\s+)?(\w+)\s+schedule", re.IGNORECASE),
        # "As per door schedule on A-201"
        re.compile(r"(?:As\s+per|Per)\s+(\w+)\s+schedule\s+(?:on)?\s*(\w+-?\d+)?", re.IGNORECASE),
    ]

    DRAWING_PATTERNS = [
        # "See A-301" or "Refer S-402"
        re.compile(r"(?:See|Refer)\s+(?:to\s+)?(?:sheet)?\s*([A-Z]-?\d{3})", re.IGNORECASE),
        # "On sheet A-301"
        re.compile(r"(?:On|In)\s+(?:sheet|dwg|drawing)\s+([A-Z]-?\d{3})", re.IGNORECASE),
    ]

    # Impact descriptions by reference type
    IMPACT_DESCRIPTIONS = {
        "detail": "Structural/finish details affect material quantities and specifications",
        "section": "Building sections provide critical height and layer information",
        "schedule": "Schedule data is essential for accurate quantity takeoff",
        "drawing": "Referenced drawing may contain critical scope information",
    }

    def __init__(self):
        self.references: List[Reference] = []
        self.missing_refs: List[MissingReference] = []
        self.present_sheets: Set[str] = set()
        self._ref_counter = 0

    def analyze_references(
        self,
        project_id: str,
        extraction_results: List[Dict],
        page_index: List[Dict],
    ) -> ReferenceReport:
        """
        Analyze all cross-references in the drawing set.

        Args:
            project_id: Project identifier
            extraction_results: Page extraction results
            page_index: Page index data

        Returns:
            Complete reference analysis report
        """
        self.references = []
        self.missing_refs = []
        self.present_sheets = set()
        self._ref_counter = 0

        # Build set of present sheets
        self._build_present_sheets(extraction_results, page_index)

        # Scan all pages for references
        self._scan_for_references(extraction_results)

        # Identify missing references
        self._identify_missing_references()

        # Build report
        report = self._build_report(project_id)

        return report

    def _generate_ref_id(self) -> str:
        """Generate unique reference ID."""
        self._ref_counter += 1
        return f"REF-{self._ref_counter:04d}"

    def _build_present_sheets(
        self,
        extraction_results: List[Dict],
        page_index: List[Dict],
    ) -> None:
        """Build set of sheet IDs present in the drawing set."""

        # From page index
        for page in page_index:
            file_path = page.get("file_path", "")
            file_stem = Path(file_path).stem.upper()
            self.present_sheets.add(file_stem)

            # Also add without extension variations
            self.present_sheets.add(file_stem.replace("_", "-"))
            self.present_sheets.add(file_stem.replace("-", "_"))

        # From extraction results - check title blocks
        for result in extraction_results:
            file_stem = Path(result.get("file_path", "")).stem.upper()
            self.present_sheets.add(file_stem)

            title_block = result.get("title_block", {})
            sheet_no = title_block.get("sheet_number", "")
            if sheet_no:
                self.present_sheets.add(sheet_no.upper())
                self.present_sheets.add(sheet_no.upper().replace("-", ""))
                self.present_sheets.add(sheet_no.upper().replace("_", ""))

            # Drawing number
            drawing_no = title_block.get("drawing_number", "")
            if drawing_no:
                self.present_sheets.add(drawing_no.upper())

        # Common variations
        sheets_copy = set(self.present_sheets)
        for sheet in sheets_copy:
            # Add S-401 -> S401 variations
            self.present_sheets.add(sheet.replace("-", ""))
            self.present_sheets.add(sheet.replace("_", ""))
            # Add lowercase
            self.present_sheets.add(sheet.lower())

    def _scan_for_references(
        self,
        extraction_results: List[Dict],
    ) -> None:
        """Scan all text items for cross-references."""

        for result in extraction_results:
            page_id = f"{Path(result.get('file_path', '')).stem}_p{result.get('page_number', 0) + 1}"

            # Collect all text to scan
            texts_to_scan = []

            # Text items
            for text_item in result.get("text_items", []):
                text = text_item.get("text", "")
                if text:
                    texts_to_scan.append(text)

            # Notes
            for note in result.get("notes", []):
                text = note.get("text", "")
                if text:
                    texts_to_scan.append(text)

            # Title block
            title_block = result.get("title_block", {})
            for key in ["notes", "description", "remarks"]:
                if title_block.get(key):
                    texts_to_scan.append(title_block[key])

            # Scan all collected text
            for text in texts_to_scan:
                self._extract_references(text, page_id)

    def _extract_references(self, text: str, page_id: str) -> None:
        """Extract all references from a text string."""

        # Detail references
        for pattern in self.DETAIL_PATTERNS:
            for match in pattern.finditer(text):
                detail_id = match.group(1)
                sheet_id = match.group(2).upper() if match.group(2) else ""

                if sheet_id:
                    exists = self._sheet_exists(sheet_id)

                    self.references.append(Reference(
                        reference_id=self._generate_ref_id(),
                        reference_type="detail",
                        reference_text=match.group(0),
                        target_sheet=sheet_id,
                        target_item=f"Detail {detail_id}",
                        source_page=page_id,
                        source_context=self._get_context(text, match.start(), match.end()),
                        exists=exists,
                        confidence=0.9,
                    ))

        # Section references
        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                section_id = groups[0]
                if len(groups) > 1 and groups[1]:
                    section_id += f"-{groups[1]}"
                sheet_id = groups[-1].upper() if groups[-1] else ""

                if sheet_id:
                    exists = self._sheet_exists(sheet_id)

                    self.references.append(Reference(
                        reference_id=self._generate_ref_id(),
                        reference_type="section",
                        reference_text=match.group(0),
                        target_sheet=sheet_id,
                        target_item=f"Section {section_id}",
                        source_page=page_id,
                        source_context=self._get_context(text, match.start(), match.end()),
                        exists=exists,
                        confidence=0.85,
                    ))

        # Schedule references
        for pattern in self.SCHEDULE_PATTERNS:
            for match in pattern.finditer(text):
                schedule_type = match.group(1).lower()
                sheet_id = match.group(2).upper() if len(match.groups()) > 1 and match.group(2) else ""

                # For schedules, we check if we have that schedule type
                # This is handled differently - mark as exists=False if schedule type is common
                common_schedules = ["door", "window", "finish", "column", "beam"]
                exists = schedule_type.lower() not in common_schedules or sheet_id != ""

                self.references.append(Reference(
                    reference_id=self._generate_ref_id(),
                    reference_type="schedule",
                    reference_text=match.group(0),
                    target_sheet=sheet_id or f"{schedule_type}_schedule",
                    target_item=f"{schedule_type.title()} Schedule",
                    source_page=page_id,
                    source_context=self._get_context(text, match.start(), match.end()),
                    exists=exists,
                    confidence=0.8,
                ))

        # Drawing references
        for pattern in self.DRAWING_PATTERNS:
            for match in pattern.finditer(text):
                sheet_id = match.group(1).upper()
                exists = self._sheet_exists(sheet_id)

                self.references.append(Reference(
                    reference_id=self._generate_ref_id(),
                    reference_type="drawing",
                    reference_text=match.group(0),
                    target_sheet=sheet_id,
                    target_item=f"Drawing {sheet_id}",
                    source_page=page_id,
                    source_context=self._get_context(text, match.start(), match.end()),
                    exists=exists,
                    confidence=0.85,
                ))

    def _sheet_exists(self, sheet_id: str) -> bool:
        """Check if a sheet ID exists in the drawing set."""
        sheet_upper = sheet_id.upper()

        # Direct match
        if sheet_upper in self.present_sheets:
            return True

        # Try without hyphens/underscores
        normalized = sheet_upper.replace("-", "").replace("_", "")
        if normalized in self.present_sheets:
            return True

        # Try partial match (e.g., "S-402" matches "ABC_S-402")
        for present in self.present_sheets:
            if sheet_upper in present or normalized in present.replace("-", "").replace("_", ""):
                return True

        return False

    def _get_context(self, text: str, start: int, end: int, context_chars: int = 50) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(text), end + context_chars)

        context = text[ctx_start:ctx_end]
        if ctx_start > 0:
            context = "..." + context
        if ctx_end < len(text):
            context = context + "..."

        return context.replace("\n", " ").strip()

    def _identify_missing_references(self) -> None:
        """Identify references to non-existent sheets."""

        for ref in self.references:
            if not ref.exists:
                # Determine severity based on reference type
                if ref.reference_type == "detail":
                    severity = "high"
                    impact = "Detail drawing missing - cannot verify construction method, dimensions, or material specifications"
                elif ref.reference_type == "section":
                    severity = "high"
                    impact = "Section drawing missing - cannot verify heights, layers, and vertical dimensions"
                elif ref.reference_type == "schedule":
                    severity = "high"
                    impact = "Schedule missing - cannot verify quantities, sizes, and specifications"
                else:
                    severity = "medium"
                    impact = self.IMPACT_DESCRIPTIONS.get(ref.reference_type, "Referenced information unavailable")

                suggestion = f"Please provide {ref.target_sheet} containing {ref.target_item}"

                self.missing_refs.append(MissingReference(
                    reference=ref,
                    severity=severity,
                    impact=impact,
                    suggestion=suggestion,
                ))

    def _build_report(self, project_id: str) -> ReferenceReport:
        """Build reference analysis report."""

        report = ReferenceReport(project_id=project_id)
        report.all_references = self.references
        report.missing_references = self.missing_refs
        report.present_sheets = self.present_sheets

        # Summary
        ref_by_type = {}
        missing_by_type = {}

        for ref in self.references:
            t = ref.reference_type
            ref_by_type[t] = ref_by_type.get(t, 0) + 1

        for missing in self.missing_refs:
            t = missing.reference.reference_type
            missing_by_type[t] = missing_by_type.get(t, 0) + 1

        report.summary = {
            "total_references": len(self.references),
            "missing_references": len(self.missing_refs),
            "present_sheets": len(self.present_sheets),
            "references_by_type": ref_by_type,
            "missing_by_type": missing_by_type,
            "high_severity_missing": len([m for m in self.missing_refs if m.severity == "high"]),
        }

        return report


def analyze_references(
    project_id: str,
    extraction_results: List[Dict],
    page_index: List[Dict],
) -> ReferenceReport:
    """
    Convenience function to analyze all references.
    """
    detector = ReferenceDetector()
    return detector.analyze_references(
        project_id=project_id,
        extraction_results=extraction_results,
        page_index=page_index,
    )
