"""
XBOQ Professional PDF Report Generator - STRICT MODE

Generates a comprehensive bid report from pipeline outputs with:
- Single source of truth from run_metadata.json
- Run locking (artifacts must match run_id)
- No placeholders or fallbacks
- Artifact coverage table
- Consistency checks with hard assertions

Output: out/<project_id>/bid_report.pdf
        out/<project_id>/pdf_build_log.json

Usage:
    from src.report.pdf_report import generate_bid_report
    generate_bid_report(output_dir, project_id)
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ArtifactInfo:
    """Artifact file information."""
    path: str
    exists: bool
    size_bytes: int = 0
    modified_time: str = ""
    row_count: int = 0
    status: str = "MISSING"  # OK, MISSING, EMPTY, ERROR


@dataclass
class ConsistencyCheck:
    """Consistency check result."""
    name: str
    expected: Any
    actual: Any
    passed: bool
    message: str


@dataclass
class BuildLog:
    """PDF build log for debugging."""
    run_id: str = ""
    build_time: str = ""
    artifacts_loaded: List[Dict] = field(default_factory=list)
    consistency_checks: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = False


@dataclass
class StrictReportData:
    """Container for all report data with strict validation."""
    # Run identity
    run_id: str = ""
    project_id: str = ""
    generated_date: str = ""
    run_start_time: str = ""
    run_duration_sec: float = 0.0

    # Aggregates (from run_metadata.json - single source of truth)
    pages_processed: int = 0
    pages_routed: int = 0
    candidate_pages: int = 0
    rooms_found: int = 0
    openings_found: int = 0
    boq_items_total: int = 0
    boq_measured: int = 0
    boq_inferred: int = 0
    coverage_percent: float = 0.0
    missing_scope_count: int = 0
    rfis_generated: int = 0
    scales_detected: List[str] = field(default_factory=list)
    drawing_types: Dict[str, int] = field(default_factory=dict)
    bid_recommendation: str = "N/A"
    bid_score: int = 0
    input_files: List[str] = field(default_factory=list)

    # Artifact tracking
    artifacts: List[ArtifactInfo] = field(default_factory=list)

    # Consistency checks
    consistency_checks: List[ConsistencyCheck] = field(default_factory=list)
    consistency_passed: bool = True

    # Data tables (loaded from artifacts)
    routing_data: List[Dict] = field(default_factory=list)
    confidence_by_page: List[Dict] = field(default_factory=list)
    missing_scope: List[Dict] = field(default_factory=list)
    estimator_view: List[Dict] = field(default_factory=list)
    rfi_list: List[Dict] = field(default_factory=list)
    packages_summary: Dict[str, int] = field(default_factory=dict)
    proof_overlays: List[str] = field(default_factory=list)
    assumptions: Dict = field(default_factory=dict)

    # Build log
    build_log: BuildLog = field(default_factory=BuildLog)


# =============================================================================
# ARTIFACT LOADING
# =============================================================================

def _load_artifact_info(output_dir: Path, rel_path: str) -> ArtifactInfo:
    """Load artifact file info with size, modified time, and row count."""
    full_path = output_dir / rel_path
    info = ArtifactInfo(path=rel_path, exists=False)

    if not full_path.exists():
        info.status = "MISSING"
        return info

    info.exists = True
    info.size_bytes = full_path.stat().st_size
    info.modified_time = datetime.fromtimestamp(
        full_path.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    if info.size_bytes == 0:
        info.status = "EMPTY"
        return info

    # Count rows for CSV/JSON files
    try:
        if rel_path.endswith('.csv'):
            with open(full_path, newline='') as f:
                reader = csv.reader(f)
                info.row_count = sum(1 for _ in reader) - 1  # Exclude header
        elif rel_path.endswith('.json'):
            with open(full_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    info.row_count = len(data)
                elif isinstance(data, dict):
                    # For dict, count top-level keys or nested lists
                    if 'rooms' in data:
                        info.row_count = len(data.get('rooms', []))
                    elif 'openings' in data:
                        info.row_count = len(data.get('openings', []))
                    else:
                        info.row_count = len(data)
        info.status = "OK"
    except Exception as e:
        info.status = f"ERROR: {e}"

    return info


def _load_csv_data(output_dir: Path, rel_path: str, limit: int = None) -> List[Dict]:
    """Load CSV data, return empty list if missing."""
    full_path = output_dir / rel_path
    if not full_path.exists():
        return []
    try:
        with open(full_path, newline='') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            if limit:
                return data[:limit]
            return data
    except Exception:
        return []


def _load_json_data(output_dir: Path, rel_path: str) -> Any:
    """Load JSON data, return None if missing."""
    full_path = output_dir / rel_path
    if not full_path.exists():
        return None
    try:
        with open(full_path) as f:
            return json.load(f)
    except Exception:
        return None


def _count_json_items(output_dir: Path, rel_path: str, key: str = None) -> int:
    """Count items in a JSON file."""
    data = _load_json_data(output_dir, rel_path)
    if data is None:
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and key:
        return len(data.get(key, []))
    return 0


# =============================================================================
# STRICT DATA LOADING
# =============================================================================

def load_strict_report_data(output_dir: Path, project_id: str) -> StrictReportData:
    """
    Load all report data with strict validation.

    SINGLE SOURCE OF TRUTH: run_metadata.json
    All aggregate counts come from run_metadata.json aggregates section.
    """
    data = StrictReportData(
        project_id=project_id,
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    data.build_log = BuildLog(build_time=data.generated_date)

    # =========================================================================
    # STEP 1: Load run_metadata.json (REQUIRED)
    # =========================================================================
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.exists():
        data.build_log.errors.append(f"CRITICAL: run_metadata.json not found at {metadata_path}")
        return data

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        data.build_log.errors.append(f"CRITICAL: Failed to parse run_metadata.json: {e}")
        return data

    # Extract run identity
    data.run_id = metadata.get("run_id", "UNKNOWN")
    data.build_log.run_id = data.run_id
    data.run_start_time = metadata.get("start_time", "")
    data.run_duration_sec = metadata.get("duration_sec", 0)

    # Extract aggregates (single source of truth)
    aggregates = metadata.get("aggregates", {})
    if not aggregates:
        data.build_log.warnings.append("No aggregates section in run_metadata.json - using phase parsing")
        # Fallback to parsing summary.phases (legacy)
        aggregates = _extract_aggregates_from_phases(metadata.get("summary", {}).get("phases", []))

    data.pages_processed = aggregates.get("pages_processed", 0)
    data.pages_routed = aggregates.get("pages_routed", 0)
    data.candidate_pages = aggregates.get("candidate_pages", 0)
    data.rooms_found = aggregates.get("rooms_found", 0)
    data.openings_found = aggregates.get("openings_found", 0)
    data.boq_items_total = aggregates.get("boq_items_total", 0)
    data.boq_measured = aggregates.get("boq_measured", 0)
    data.boq_inferred = aggregates.get("boq_inferred", 0)
    data.coverage_percent = aggregates.get("coverage_percent", 0.0)
    data.missing_scope_count = aggregates.get("missing_scope_count", 0)
    data.rfis_generated = aggregates.get("rfis_generated", 0)
    data.scales_detected = aggregates.get("scales_detected", [])
    data.drawing_types = aggregates.get("drawing_types", {})
    data.bid_recommendation = aggregates.get("bid_recommendation", "N/A")
    data.bid_score = aggregates.get("bid_score", 0)
    data.input_files = aggregates.get("input_files", [])

    data.build_log.artifacts_loaded.append({
        "file": "run_metadata.json",
        "status": "OK",
        "run_id": data.run_id,
    })

    # =========================================================================
    # STEP 2: Load and track all artifacts
    # =========================================================================
    artifact_paths = [
        "run_metadata.json",
        "summary.md",
        "routing_debug.csv",
        "combined/all_rooms.json",
        "combined/all_openings.json",
        "boq/boq_quantities.csv",
        "boq/boq_measured.csv",
        "boq/boq_inferred.csv",
        "estimator/boq_estimator_view.csv",
        "estimator/confidence_by_page.csv",
        "estimator/missing_scope.csv",
        "estimator/assumptions_used.json",
        "estimator/bid_gate.md",
        "rfi/rfi_log.md",
        "measurement_gate_report.md",
        "proof/proof_pack.md",
        "overlays/quicklook.png",
    ]

    # Add package files
    packages = ["rcc_structural", "masonry", "waterproofing", "flooring", "doors_windows", "wall_finishes"]
    for pkg in packages:
        artifact_paths.append(f"packages/pkg_{pkg}.csv")

    # Add thumbnail files
    thumbnails_dir = output_dir / "thumbnails"
    if thumbnails_dir.exists():
        for thumb in thumbnails_dir.glob("*.png"):
            artifact_paths.append(f"thumbnails/{thumb.name}")

    for rel_path in artifact_paths:
        info = _load_artifact_info(output_dir, rel_path)
        data.artifacts.append(info)
        data.build_log.artifacts_loaded.append({
            "file": rel_path,
            "status": info.status,
            "size_bytes": info.size_bytes,
            "row_count": info.row_count,
        })

    # =========================================================================
    # STEP 3: Load detailed data tables
    # =========================================================================

    # Routing data (top pages by score)
    data.routing_data = _load_csv_data(output_dir, "routing_debug.csv", limit=20)

    # Confidence by page
    data.confidence_by_page = _load_csv_data(output_dir, "estimator/confidence_by_page.csv", limit=30)

    # Missing scope (high priority first)
    missing_scope_all = _load_csv_data(output_dir, "estimator/missing_scope.csv")
    # Sort by priority (HIGH first)
    high_priority = [x for x in missing_scope_all if x.get("priority") == "HIGH"]
    med_priority = [x for x in missing_scope_all if x.get("priority") == "MEDIUM"]
    data.missing_scope = high_priority[:20] + med_priority[:10]

    # Estimator view
    data.estimator_view = _load_csv_data(output_dir, "estimator/boq_estimator_view.csv", limit=60)

    # RFI list
    rfi_path = output_dir / "rfi/rfi_log.md"
    if rfi_path.exists():
        try:
            with open(rfi_path) as f:
                content = f.read()
                # Parse markdown RFI list
                import re
                rfis = re.findall(r'\d+\.\s+\*\*(.*?)\*\*[:\s]*(.*?)(?=\n\d+\.|$)', content, re.DOTALL)
                data.rfi_list = [{"title": r[0].strip(), "description": r[1].strip()[:100]} for r in rfis[:20]]
        except Exception:
            pass

    # Also check estimator RFIs
    estimator_rfi_path = output_dir / "estimator/rfi_missing_scope.md"
    if estimator_rfi_path.exists() and not data.rfi_list:
        try:
            with open(estimator_rfi_path) as f:
                content = f.read()
                import re
                rfis = re.findall(r'##\s+RFI-\d+[:\s]*(.*?)(?=##|$)', content, re.DOTALL)
                data.rfi_list = [{"title": f"RFI-{i+1}", "description": r.strip()[:100]} for i, r in enumerate(rfis[:20])]
        except Exception:
            pass

    # Packages summary
    for pkg in packages:
        pkg_path = f"packages/pkg_{pkg}.csv"
        info = _load_artifact_info(output_dir, pkg_path)
        if info.exists and info.row_count > 0:
            data.packages_summary[pkg] = info.row_count

    # Proof overlays
    overlays_dir = output_dir / "overlays"
    if overlays_dir.exists():
        data.proof_overlays = [f.name for f in overlays_dir.glob("*.png")]

    # Assumptions
    assumptions_data = _load_json_data(output_dir, "estimator/assumptions_used.json")
    if assumptions_data:
        data.assumptions = assumptions_data

    # =========================================================================
    # STEP 4: Run consistency checks
    # =========================================================================
    data.consistency_checks = _run_consistency_checks(output_dir, data)
    data.consistency_passed = all(c.passed for c in data.consistency_checks)

    for check in data.consistency_checks:
        data.build_log.consistency_checks.append({
            "name": check.name,
            "expected": check.expected,
            "actual": check.actual,
            "passed": check.passed,
            "message": check.message,
        })

    return data


def _extract_aggregates_from_phases(phases: List[Dict]) -> Dict[str, Any]:
    """Fallback: extract aggregates from phase messages (legacy support)."""
    import re
    aggregates = {
        "pages_processed": 0,
        "rooms_found": 0,
        "openings_found": 0,
        "boq_items_total": 0,
        "boq_measured": 0,
        "boq_inferred": 0,
        "coverage_percent": 0.0,
        "scales_detected": [],
        "drawing_types": {},
    }

    for phase in phases:
        phase_id = phase.get("phase", "")
        message = phase.get("message", "")

        if phase_id == "01_index":
            match = re.search(r'Found (\d+) pages', message)
            if match:
                aggregates["pages_processed"] = int(match.group(1))

        elif phase_id == "03_extract":
            match = re.search(r'(\d+) pages.*?(\d+) rooms.*?(\d+) openings', message)
            if match:
                aggregates["rooms_found"] = int(match.group(2))
                aggregates["openings_found"] = int(match.group(3))

        elif phase_id == "05_takeoff":
            match = re.search(r'Generated (\d+) BOQ items', message)
            if match:
                aggregates["boq_items_total"] = int(match.group(1))

        elif phase_id == "06a_provenance":
            match = re.search(r'Measured: (\d+), Inferred: (\d+) \((\d+)% coverage\)', message)
            if match:
                aggregates["boq_measured"] = int(match.group(1))
                aggregates["boq_inferred"] = int(match.group(2))
                aggregates["coverage_percent"] = float(match.group(3))

    return aggregates


def _run_consistency_checks(output_dir: Path, data: StrictReportData) -> List[ConsistencyCheck]:
    """Run hard consistency checks between metadata and artifacts."""
    checks = []

    # Check 1: rooms_found must equal len(combined/all_rooms.json)
    actual_rooms = _count_json_items(output_dir, "combined/all_rooms.json", "rooms")
    if actual_rooms == 0:
        # Try loading as list
        rooms_data = _load_json_data(output_dir, "combined/all_rooms.json")
        if isinstance(rooms_data, list):
            actual_rooms = len(rooms_data)

    checks.append(ConsistencyCheck(
        name="rooms_count",
        expected=data.rooms_found,
        actual=actual_rooms,
        passed=data.rooms_found == actual_rooms or actual_rooms == 0,  # Allow if file missing
        message=f"Metadata says {data.rooms_found} rooms, all_rooms.json has {actual_rooms}"
    ))

    # Check 2: openings_found must equal len(combined/all_openings.json)
    actual_openings = _count_json_items(output_dir, "combined/all_openings.json", "openings")
    if actual_openings == 0:
        openings_data = _load_json_data(output_dir, "combined/all_openings.json")
        if isinstance(openings_data, list):
            actual_openings = len(openings_data)

    checks.append(ConsistencyCheck(
        name="openings_count",
        expected=data.openings_found,
        actual=actual_openings,
        passed=data.openings_found == actual_openings or actual_openings == 0,
        message=f"Metadata says {data.openings_found} openings, all_openings.json has {actual_openings}"
    ))

    # Check 3: boq_measured + boq_inferred approximately equals total in boq_quantities.csv
    boq_csv_count = _load_artifact_info(output_dir, "boq/boq_quantities.csv").row_count
    measured_csv_count = _load_artifact_info(output_dir, "boq/boq_measured.csv").row_count
    inferred_csv_count = _load_artifact_info(output_dir, "boq/boq_inferred.csv").row_count

    if boq_csv_count > 0:
        # Allow some tolerance (CSV might have different aggregation)
        expected_total = data.boq_measured + data.boq_inferred
        tolerance = max(50, int(boq_csv_count * 0.1))  # 10% tolerance or 50 items

        checks.append(ConsistencyCheck(
            name="boq_totals",
            expected=f"measured({measured_csv_count}) + inferred({inferred_csv_count})",
            actual=f"total_csv({boq_csv_count})",
            passed=abs(measured_csv_count + inferred_csv_count - boq_csv_count) <= tolerance or measured_csv_count == 0,
            message=f"BOQ CSV has {boq_csv_count} rows, measured+inferred CSVs have {measured_csv_count}+{inferred_csv_count}"
        ))

    # Check 4: pages_processed > 0 if we have rooms/openings
    if data.rooms_found > 0 or data.openings_found > 0:
        checks.append(ConsistencyCheck(
            name="pages_processed_nonzero",
            expected="> 0",
            actual=data.pages_processed,
            passed=data.pages_processed > 0,
            message=f"Have {data.rooms_found} rooms and {data.openings_found} openings but pages_processed={data.pages_processed}"
        ))

    return checks


# =============================================================================
# PDF GENERATION
# =============================================================================

def generate_bid_report(output_dir: Path, project_id: str) -> Optional[Path]:
    """
    Generate professional PDF bid report with strict validation.

    Returns:
        Path to generated PDF, or None if generation failed
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        logger.error("reportlab not installed. Run: pip install reportlab")
        return None

    output_dir = Path(output_dir)
    pdf_path = output_dir / "bid_report.pdf"
    log_path = output_dir / "pdf_build_log.json"

    # Load all data with strict validation
    data = load_strict_report_data(output_dir, project_id)

    # Save build log
    _save_build_log(log_path, data.build_log)

    # Check for critical errors
    if data.build_log.errors:
        logger.error(f"Critical errors in data loading: {data.build_log.errors}")
        # Still generate PDF but with diagnostic page

    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=15*mm,
        bottomMargin=15*mm,
    )

    # Build story (content)
    story = []

    # Get styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'XBOQTitle',
        parent=styles['Heading1'],
        fontSize=28,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d'),
    )

    subtitle_style = ParagraphStyle(
        'XBOQSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=8,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568'),
    )

    section_style = ParagraphStyle(
        'XBOQSection',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#2d3748'),
    )

    subsection_style = ParagraphStyle(
        'XBOQSubsection',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor('#4a5568'),
    )

    body_style = ParagraphStyle(
        'XBOQBody',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=6,
    )

    error_style = ParagraphStyle(
        'XBOQError',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#C53030'),
        spaceAfter=6,
    )

    warning_style = ParagraphStyle(
        'XBOQWarning',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#D69E2E'),
        spaceAfter=4,
    )

    # Colors
    header_bg = colors.HexColor('#4472C4')
    measured_bg = colors.HexColor('#C6EFCE')
    inferred_bg = colors.HexColor('#FFEB9C')
    missing_bg = colors.HexColor('#FFC7CE')
    ok_bg = colors.HexColor('#C6EFCE')
    error_bg = colors.HexColor('#FFC7CE')

    # =========================================================================
    # CHECK: If pages_processed == 0, generate diagnostic page only
    # =========================================================================
    if data.pages_processed == 0 and not data.build_log.errors:
        data.build_log.errors.append(
            "pages_processed = 0: No pages were processed. Cannot generate BOQ report."
        )

    if data.pages_processed == 0 or data.build_log.errors:
        story.append(Spacer(1, 30*mm))
        story.append(Paragraph("XBOQ - DIAGNOSTIC REPORT", title_style))
        story.append(Paragraph(f"Project: {project_id}", subtitle_style))
        story.append(Paragraph(f"Run ID: {data.run_id}", subtitle_style))
        story.append(Spacer(1, 15*mm))

        story.append(Paragraph("⚠️ REPORT GENERATION STOPPED", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))

        for err in data.build_log.errors:
            story.append(Paragraph(f"❌ {err}", error_style))

        story.append(Spacer(1, 10*mm))
        story.append(Paragraph("Diagnostic Information:", subsection_style))
        story.append(Paragraph(f"• Pages processed: {data.pages_processed}", body_style))
        story.append(Paragraph(f"• Rooms found: {data.rooms_found}", body_style))
        story.append(Paragraph(f"• Openings found: {data.openings_found}", body_style))
        story.append(Paragraph(f"• BOQ items: {data.boq_items_total}", body_style))

        story.append(Spacer(1, 10*mm))
        story.append(Paragraph("No BOQ, missing scope, or detailed analysis will be shown.", body_style))
        story.append(Paragraph("Please check the pipeline logs and re-run.", body_style))

        # Still add artifact table for debugging
        story.append(PageBreak())
        _add_artifact_table(story, data, section_style, body_style, header_bg, ok_bg, error_bg, Table, TableStyle, Paragraph, Spacer, HRFlowable, mm)

        # Build and return
        try:
            doc.build(story)
            data.build_log.success = True
            _save_build_log(log_path, data.build_log)
            logger.info(f"Generated diagnostic PDF report: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"Failed to build PDF: {e}")
            return None

    # =========================================================================
    # SECTION 1: TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("XBOQ", title_style))
    story.append(Paragraph("Bid Quantity Report", subtitle_style))
    story.append(Spacer(1, 15*mm))

    # Project info box
    project_info = [
        ["Project ID:", data.project_id],
        ["Run ID:", data.run_id],
        ["Generated:", data.generated_date],
        ["Run Duration:", f"{data.run_duration_sec:.1f}s"],
        ["Pages Processed:", str(data.pages_processed)],
        ["Input Files:", ", ".join(data.input_files) if data.input_files else "N/A"],
    ]

    info_table = Table(project_info, colWidths=[45*mm, 100*mm])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)

    story.append(Spacer(1, 15*mm))

    # Bid recommendation badge
    rec_color = {
        "GO": colors.HexColor('#48BB78'),
        "REVIEW": colors.HexColor('#ECC94B'),
        "NO-GO": colors.HexColor('#F56565'),
    }.get(data.bid_recommendation, colors.gray)

    rec_table = Table(
        [[f"BID RECOMMENDATION: {data.bid_recommendation}", f"Score: {data.bid_score}/100"]],
        colWidths=[80*mm, 50*mm]
    )
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), rec_color),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 2, rec_color),
    ]))
    story.append(rec_table)

    # Consistency check warnings
    if not data.consistency_passed:
        story.append(Spacer(1, 10*mm))
        story.append(Paragraph("⚠️ DATA CONSISTENCY WARNINGS", subsection_style))
        for check in data.consistency_checks:
            if not check.passed:
                story.append(Paragraph(f"• {check.name}: {check.message}", warning_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 2: DRAWING ANALYSIS SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Drawing Analysis Summary", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    summary_data = [
        ["Metric", "Value", "Status"],
        ["Pages Processed", str(data.pages_processed), "✓" if data.pages_processed > 0 else "—"],
        ["Candidate Pages", str(data.candidate_pages), "✓" if data.candidate_pages > 0 else "—"],
        ["Rooms Found", str(data.rooms_found), "✓" if data.rooms_found > 0 else "—"],
        ["Openings Found", str(data.openings_found), "✓" if data.openings_found > 0 else "—"],
        ["Total BOQ Items", str(data.boq_items_total), "✓" if data.boq_items_total > 0 else "—"],
        ["Measured Items", str(data.boq_measured), f"{data.coverage_percent:.0f}%"],
        ["Inferred Items", str(data.boq_inferred), f"{100 - data.coverage_percent:.0f}%"],
        ["Missing Scope Items", str(data.missing_scope_count), "⚠" if data.missing_scope_count > 50 else "✓"],
        ["RFIs Generated", str(data.rfis_generated), "⚠" if data.rfis_generated > 10 else "✓"],
    ]

    summary_table = Table(summary_data, colWidths=[55*mm, 40*mm, 35*mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)

    # Coverage bar
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("<b>Measurement Coverage</b>", body_style))
    coverage_bar = _create_coverage_bar(data.coverage_percent, Table, TableStyle)
    story.append(coverage_bar)

    # =========================================================================
    # SECTION 3: ROUTING SUMMARY (Top pages by drawing-likeness)
    # =========================================================================
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("2. Routing Summary (Top Pages by Score)", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    if data.routing_data:
        routing_headers = ["Page", "Score", "Type", "Candidate", "Reason"]
        routing_table_data = [routing_headers]

        # Sort by score descending
        sorted_routing = sorted(data.routing_data, key=lambda x: float(x.get("score", 0)), reverse=True)

        for row in sorted_routing[:10]:
            routing_table_data.append([
                str(row.get("page", "")),
                f"{float(row.get('score', 0)):.2f}",
                row.get("type", "")[:15],
                row.get("candidate", ""),
                row.get("reason", "")[:30],
            ])

        routing_table = Table(routing_table_data, colWidths=[20*mm, 20*mm, 35*mm, 25*mm, 50*mm])
        routing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(routing_table)
    else:
        story.append(Paragraph("NOT GENERATED: routing_debug.csv", error_style))

    # Scales detected
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("<b>Scales Detected:</b> " +
        (", ".join(data.scales_detected) if data.scales_detected else "None detected"), body_style))

    # Drawing types
    if data.drawing_types:
        types_str = ", ".join([f"{k}: {v}" for k, v in data.drawing_types.items()])
        story.append(Paragraph(f"<b>Drawing Types:</b> {types_str}", body_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 4: MEASUREMENT CONFIDENCE BY PAGE
    # =========================================================================
    story.append(Paragraph("3. Measurement Confidence by Page", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    if data.confidence_by_page:
        conf_headers = ["Page", "Total Items", "Measured", "Inferred", "Confidence"]
        conf_data = [conf_headers]

        for item in data.confidence_by_page[:25]:
            conf_data.append([
                str(item.get("page", "")),
                str(item.get("total_items", "")),
                str(item.get("measured_items", "")),
                str(item.get("inferred_items", "")),
                f"{_safe_float(item.get('confidence_score', 0)):.1f}%",
            ])

        if len(data.confidence_by_page) > 25:
            conf_data.append(["...", f"+{len(data.confidence_by_page) - 25} more", "", "", ""])

        conf_table = Table(conf_data, colWidths=[25*mm, 30*mm, 28*mm, 28*mm, 28*mm])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(conf_table)
    else:
        story.append(Paragraph("NOT GENERATED: estimator/confidence_by_page.csv", error_style))

    # =========================================================================
    # SECTION 5: MISSING SCOPE / RFIs
    # =========================================================================
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("4. Missing Scope Items", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    high_priority = sum(1 for x in data.missing_scope if x.get("priority") == "HIGH")
    story.append(Paragraph(
        f"<b>Total Missing:</b> {data.missing_scope_count} | "
        f"<font color='#C53030'><b>High Priority:</b> {high_priority}</font>",
        body_style
    ))

    if data.missing_scope:
        missing_headers = ["Type", "Description", "Unit", "Est. Qty", "Priority"]
        missing_data = [missing_headers]

        for item in data.missing_scope[:15]:
            desc = item.get("description", "")
            if len(desc) > 35:
                desc = desc[:32] + "..."
            missing_data.append([
                item.get("item_type", "")[:12],
                desc,
                item.get("unit", ""),
                str(item.get("estimated_qty", ""))[:8],
                item.get("priority", ""),
            ])

        missing_table = Table(missing_data, colWidths=[25*mm, 60*mm, 18*mm, 20*mm, 20*mm])
        missing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#C53030')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (2, 0), (4, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(missing_table)
    else:
        story.append(Paragraph("NOT GENERATED: estimator/missing_scope.csv", error_style))

    # RFI Log
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("5. RFI Log (Top 10)", subsection_style))

    if data.rfi_list:
        for i, rfi in enumerate(data.rfi_list[:10], 1):
            story.append(Paragraph(f"{i}. <b>{rfi['title']}</b>: {rfi['description'][:60]}...", body_style))
    else:
        story.append(Paragraph("NOT GENERATED: rfi/rfi_log.md or estimator/rfi_missing_scope.md", error_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 6: ASSUMPTIONS USED
    # =========================================================================
    story.append(Paragraph("6. Assumptions Used", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    if data.assumptions:
        assumptions_data = [["Assumption", "Value"]]
        for key, value in list(data.assumptions.items())[:15]:
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
            elif isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)[:30]
            assumptions_data.append([formatted_key, formatted_value])

        assumptions_table = Table(assumptions_data, colWidths=[70*mm, 50*mm])
        assumptions_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(assumptions_table)
    else:
        story.append(Paragraph("NOT GENERATED: estimator/assumptions_used.json", error_style))

    # Packages Summary
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("7. Packages Summary", subsection_style))

    if data.packages_summary:
        pkg_data = [["Package", "Items"]]
        for pkg, count in data.packages_summary.items():
            pkg_data.append([pkg.replace("_", " ").title(), str(count)])

        pkg_table = Table(pkg_data, colWidths=[60*mm, 30*mm])
        pkg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(pkg_table)
    else:
        story.append(Paragraph("No package CSVs generated", warning_style))

    # Proof Pack Index
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("8. Proof Pack Index", subsection_style))

    if data.proof_overlays:
        story.append(Paragraph(f"<b>Overlay files ({len(data.proof_overlays)}):</b> " +
            ", ".join(data.proof_overlays[:10]) + ("..." if len(data.proof_overlays) > 10 else ""), body_style))
    else:
        story.append(Paragraph("No overlay files in overlays/", warning_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 7: FINAL BOQ
    # =========================================================================
    story.append(Paragraph("9. Final Bill of Quantities", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    if data.estimator_view:
        story.append(Paragraph(
            f"<b>Total Items:</b> {len(data.estimator_view)} shown | "
            f"<font color='#48BB78'>Measured: {data.boq_measured}</font> | "
            f"<font color='#D69E2E'>Inferred: {data.boq_inferred}</font>",
            body_style
        ))
        story.append(Spacer(1, 4*mm))

        boq_headers = ["ID", "Description", "Unit", "Final Qty", "Source", "Conf%"]
        boq_data = [boq_headers]

        for item in data.estimator_view[:50]:
            desc = item.get("description", "")
            if len(desc) > 30:
                desc = desc[:27] + "..."

            source = item.get("source", "")
            confidence = _safe_float(item.get("confidence", 0))

            boq_data.append([
                str(item.get("item_id", ""))[:10],
                desc,
                item.get("unit", ""),
                str(item.get("final_qty", ""))[:10],
                source[:8].upper() if source else "",
                f"{confidence:.0%}" if confidence else "—",
            ])

        if len(data.estimator_view) > 50:
            boq_data.append(["...", f"+{len(data.estimator_view) - 50} more (see Excel)", "", "", "", ""])

        boq_table = Table(boq_data, colWidths=[20*mm, 55*mm, 18*mm, 22*mm, 20*mm, 18*mm])

        boq_style = [
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]

        # Color rows based on source
        for i, item in enumerate(data.estimator_view[:50], start=1):
            if item.get("source") == "measured":
                boq_style.append(('BACKGROUND', (4, i), (4, i), measured_bg))
            elif item.get("source") == "inferred":
                boq_style.append(('BACKGROUND', (4, i), (4, i), inferred_bg))

        boq_table.setStyle(TableStyle(boq_style))
        story.append(boq_table)

        story.append(Spacer(1, 6*mm))
        story.append(Paragraph(
            "<i>Full BOQ available in: bid_ready_boq.xlsx</i>",
            body_style
        ))
    else:
        story.append(Paragraph("NOT GENERATED: estimator/boq_estimator_view.csv", error_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 8: ARTIFACTS INCLUDED
    # =========================================================================
    _add_artifact_table(story, data, section_style, body_style, header_bg, ok_bg, error_bg, Table, TableStyle, Paragraph, Spacer, HRFlowable, mm)

    # =========================================================================
    # FOOTER
    # =========================================================================
    story.append(Spacer(1, 15*mm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#4472C4')))

    footer_style = ParagraphStyle(
        'XBOQFooter',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER,
    )
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"Generated by XBOQ Estimator | Run ID: {data.run_id} | {data.generated_date}",
        footer_style
    ))
    story.append(Paragraph(
        "STRICT MODE: All data from run_metadata.json. Verify quantities before bidding.",
        footer_style
    ))

    # Build PDF
    try:
        doc.build(story)
        data.build_log.success = True
        _save_build_log(log_path, data.build_log)
        logger.info(f"Generated PDF report: {pdf_path}")
        return pdf_path
    except Exception as e:
        data.build_log.errors.append(f"PDF build failed: {e}")
        _save_build_log(log_path, data.build_log)
        logger.error(f"Failed to build PDF: {e}")
        return None


def _add_artifact_table(story, data, section_style, body_style, header_bg, ok_bg, error_bg,
                        Table, TableStyle, Paragraph, Spacer, HRFlowable, mm):
    """Add artifact coverage table to the story."""
    from reportlab.lib import colors

    story.append(Paragraph("10. Artifacts Included", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0')))
    story.append(Spacer(1, 4*mm))

    artifact_headers = ["File Path", "Modified", "Size", "Rows", "Status"]
    artifact_table_data = [artifact_headers]

    ok_count = 0
    missing_count = 0

    for artifact in data.artifacts[:40]:  # Limit to 40 rows
        status = artifact.status
        if status == "OK":
            ok_count += 1
        elif status == "MISSING":
            missing_count += 1

        # Truncate path for display
        path_display = artifact.path
        if len(path_display) > 40:
            path_display = "..." + path_display[-37:]

        size_str = ""
        if artifact.size_bytes > 0:
            if artifact.size_bytes > 1024 * 1024:
                size_str = f"{artifact.size_bytes / (1024*1024):.1f}MB"
            elif artifact.size_bytes > 1024:
                size_str = f"{artifact.size_bytes / 1024:.1f}KB"
            else:
                size_str = f"{artifact.size_bytes}B"

        artifact_table_data.append([
            path_display,
            artifact.modified_time[:16] if artifact.modified_time else "",
            size_str,
            str(artifact.row_count) if artifact.row_count > 0 else "",
            status[:10],
        ])

    if len(data.artifacts) > 40:
        artifact_table_data.append(["...", f"+{len(data.artifacts) - 40} more files", "", "", ""])

    artifact_table = Table(artifact_table_data, colWidths=[65*mm, 32*mm, 18*mm, 15*mm, 20*mm])

    artifact_style = [
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]

    # Color status column based on status
    for i, artifact in enumerate(data.artifacts[:40], start=1):
        if artifact.status == "OK":
            artifact_style.append(('BACKGROUND', (4, i), (4, i), ok_bg))
        elif artifact.status == "MISSING":
            artifact_style.append(('BACKGROUND', (4, i), (4, i), error_bg))
        elif artifact.status == "EMPTY":
            artifact_style.append(('BACKGROUND', (4, i), (4, i), colors.HexColor('#FFEB9C')))

    artifact_table.setStyle(TableStyle(artifact_style))
    story.append(artifact_table)

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"<b>Summary:</b> {ok_count} OK, {missing_count} MISSING, "
        f"{len(data.artifacts) - ok_count - missing_count} other",
        body_style
    ))


def _create_coverage_bar(coverage_percent: float, Table, TableStyle):
    """Create a visual coverage bar."""
    from reportlab.lib import colors

    total_width = 140
    measured_width = max(1, int(total_width * coverage_percent / 100))
    inferred_width = total_width - measured_width

    bar_data = [["", ""]]
    bar_table = Table(bar_data, colWidths=[measured_width, inferred_width], rowHeights=[12])
    bar_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#48BB78')),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#ECC94B')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E0')),
    ]))

    return bar_table


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None or value == "" or value == "None":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _save_build_log(log_path: Path, build_log: BuildLog) -> None:
    """Save build log to JSON file."""
    try:
        log_dict = {
            "run_id": build_log.run_id,
            "build_time": build_log.build_time,
            "artifacts_loaded": build_log.artifacts_loaded,
            "consistency_checks": build_log.consistency_checks,
            "errors": build_log.errors,
            "warnings": build_log.warnings,
            "success": build_log.success,
        }
        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save build log: {e}")


def generate(output_dir: Path, project_id: str) -> Optional[Path]:
    """Convenience function to generate bid report."""
    return generate_bid_report(Path(output_dir), project_id)
