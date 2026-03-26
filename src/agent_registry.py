"""
Agent Registry — Central catalog of all xBOQ agents.

Each agent is a standalone module with clear inputs/outputs that can be
invoked independently via CLI or the Streamlit dashboard.

Sprint 23: Agent Registry + CLI Runner + Dashboard.

Usage:
    from src.agent_registry import list_agents, get_agent, resolve_fn
    agents = list_agents(category="extractor")
    spec = get_agent("extract_boq")
    fn = resolve_fn(spec)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


# =============================================================================
# DATA MODEL
# =============================================================================

class AgentCategory(str, Enum):
    PIPELINE = "pipeline"
    EXTRACTOR = "extractor"
    ANALYSIS = "analysis"
    STRUCTURAL = "structural"
    OUTPUT = "output"
    PROJECT = "project"


@dataclass
class AgentParam:
    """One input or output parameter of an agent."""
    name: str
    type: str           # "Path", "str", "dict", "List[dict]", "PageIndex", etc.
    required: bool = True
    description: str = ""


@dataclass
class AgentSpec:
    """Metadata for a single agent in the registry."""
    name: str                          # slug: "extract_boq"
    label: str                         # display: "BOQ Extractor"
    category: AgentCategory
    description: str                   # 1-2 sentences
    module_path: str                   # importable: "src.analysis.extractors.extract_boq"
    entry_fn: str                      # function name: "extract_boq_items"
    inputs: List[AgentParam] = field(default_factory=list)
    outputs: List[AgentParam] = field(default_factory=list)
    can_run_standalone: bool = True
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category.value,
            "description": self.description,
            "module_path": self.module_path,
            "entry_fn": self.entry_fn,
            "inputs": [
                {"name": i.name, "type": i.type, "required": i.required, "description": i.description}
                for i in self.inputs
            ],
            "outputs": [
                {"name": o.name, "type": o.type, "description": o.description}
                for o in self.outputs
            ],
            "can_run_standalone": self.can_run_standalone,
            "tags": self.tags,
        }


# =============================================================================
# AGENT DEFINITIONS — 40 agents across 6 categories
# =============================================================================

AGENTS: List[AgentSpec] = [

    # ─── Pipeline Stages (8) ─────────────────────────────────────────────

    AgentSpec(
        name="load",
        label="PDF Loader",
        category=AgentCategory.PIPELINE,
        description="Load PDF pages and extract raw native text layer (no OCR).",
        module_path="src.analysis.pipeline",
        entry_fn="load_pdf_pages",
        inputs=[
            AgentParam("pdf_path", "Path", True, "Path to PDF file"),
        ],
        outputs=[
            AgentParam("page_texts", "List[str]", description="Raw text per page"),
            AgentParam("page_count", "int", description="Total number of pages"),
        ],
        tags=["pdf", "ingest"],
    ),

    AgentSpec(
        name="index",
        label="Page Indexer",
        category=AgentCategory.PIPELINE,
        description="Classify every page by doc_type (boq, schedule, plan, etc.) and discipline (structural, architectural, etc.) using text + header-strip OCR.",
        module_path="src.analysis.page_index",
        entry_fn="build_page_index",
        inputs=[
            AgentParam("pdf_path", "Path", True, "Path to PDF file"),
            AgentParam("existing_texts", "List[str]", True, "Raw text per page from load stage"),
        ],
        outputs=[
            AgentParam("page_index", "PageIndex", description="Per-page classification with doc_type, discipline, confidence"),
        ],
        tags=["classification", "ocr"],
    ),

    AgentSpec(
        name="select",
        label="Page Selector",
        category=AgentCategory.PIPELINE,
        description="Select which pages to OCR within a budget. Tier 1 (always): BOQ, schedule, conditions. Tier 2 (fill): plans, details. Tier 3 (sample): unknown, spec.",
        module_path="src.analysis.page_selection",
        entry_fn="select_pages",
        inputs=[
            AgentParam("page_index", "PageIndex", True, "Page classifications from index stage"),
            AgentParam("budget_pages", "int", False, "Max pages to process (default: 80)"),
        ],
        outputs=[
            AgentParam("selected_pages", "SelectedPages", description="Selected page indices with audit trail"),
        ],
        tags=["selection", "budget"],
    ),

    AgentSpec(
        name="extract",
        label="Extraction Orchestrator",
        category=AgentCategory.PIPELINE,
        description="Route selected pages to specialized extractors by doc_type. Aggregates BOQ items, schedules, requirements, commercial terms, and callouts.",
        module_path="src.analysis.extractors",
        entry_fn="run_extractors",
        inputs=[
            AgentParam("ocr_texts", "Dict[int,str]", True, "OCR text keyed by page index"),
            AgentParam("page_index", "PageIndex", True, "Page classifications"),
            AgentParam("pdf_path", "Path", False, "PDF path for table extraction"),
        ],
        outputs=[
            AgentParam("extraction_result", "ExtractionResult", description="Combined BOQ, schedules, requirements, terms, callouts"),
        ],
        can_run_standalone=False,
        tags=["extraction", "orchestrator"],
    ),

    AgentSpec(
        name="graph",
        label="Plan Graph Builder",
        category=AgentCategory.PIPELINE,
        description="Build a structured graph of the plan set with sheet classifications, detected entities, cross-references between sheets, and scale warnings.",
        module_path="src.analysis.plan_graph",
        entry_fn="build_plan_graph",
        inputs=[
            AgentParam("project_id", "str", True, "Project identifier"),
            AgentParam("page_texts", "List[str]", True, "Raw text per page"),
        ],
        outputs=[
            AgentParam("plan_graph", "PlanSetGraph", description="Plan set graph with all sheet metadata"),
        ],
        tags=["graph", "sheets"],
    ),

    AgentSpec(
        name="reason",
        label="Dependency Reasoner",
        category=AgentCategory.PIPELINE,
        description="Apply rule-based analysis to detect missing dependencies (no schedule, no structural drawings, no scale, etc.) and compute trade coverage.",
        module_path="src.analysis.dependency_reasoner",
        entry_fn="DependencyReasoner",
        inputs=[
            AgentParam("plan_graph", "PlanSetGraph", True, "Plan set graph"),
            AgentParam("run_coverage", "RunCoverage", False, "Coverage from page selection"),
        ],
        outputs=[
            AgentParam("blockers", "List[Blocker]", description="Items blocking accurate pricing"),
            AgentParam("trade_coverage", "TradeCoverage", description="Per-trade coverage % and risk"),
        ],
        can_run_standalone=False,
        tags=["rules", "blockers", "coverage"],
    ),

    AgentSpec(
        name="rfi",
        label="RFI Generator",
        category=AgentCategory.PIPELINE,
        description="Checklist-driven RFI generation (~30-50 items per discipline). Evaluates missing fields, conflicts, and generates evidence-backed questions.",
        module_path="src.analysis.rfi_engine",
        entry_fn="generate_rfis",
        inputs=[
            AgentParam("extracted", "ExtractionResult", True, "Extraction results"),
            AgentParam("page_index", "PageIndex", True, "Page classifications"),
            AgentParam("selected", "SelectedPages", True, "Selected pages"),
            AgentParam("plan_graph", "PlanSetGraph", False, "Plan graph"),
            AgentParam("run_coverage", "RunCoverage", False, "Coverage metadata"),
        ],
        outputs=[
            AgentParam("rfis", "List[RFIItem]", description="Generated RFIs with evidence"),
        ],
        can_run_standalone=False,
        tags=["rfi", "checklist"],
    ),

    AgentSpec(
        name="export",
        label="Payload Assembler",
        category=AgentCategory.PIPELINE,
        description="Consolidate all analysis outputs into the final JSON payload with ~70 keys. Computes summary statistics and metrics.",
        module_path="src.analysis.pipeline",
        entry_fn="run_analysis_pipeline",
        inputs=[
            AgentParam("pdf_paths", "List[Path]", True, "Input PDF file paths"),
        ],
        outputs=[
            AgentParam("payload", "dict", description="Full analysis payload (~70 keys)"),
        ],
        can_run_standalone=False,
        tags=["export", "payload"],
    ),

    # ─── Extractors (6) ──────────────────────────────────────────────────

    AgentSpec(
        name="extract_boq",
        label="BOQ Extractor",
        category=AgentCategory.EXTRACTOR,
        description="Parse Bill of Quantities line items from OCR text using regex patterns. Handles Indian number formats, unit normalization (30+ aliases), and tabular OCR.",
        module_path="src.analysis.extractors.extract_boq",
        entry_fn="extract_boq_items",
        inputs=[
            AgentParam("text", "str", True, "OCR text from a BOQ page"),
            AgentParam("source_page", "int", True, "Page number"),
            AgentParam("pdf_path", "str", False, "PDF path for table extraction"),
        ],
        outputs=[
            AgentParam("boq_items", "List[dict]", description="BOQ items with item_no, description, unit, qty, rate"),
        ],
        tags=["boq", "regex", "ocr"],
    ),

    AgentSpec(
        name="extract_schedule",
        label="Schedule Extractor",
        category=AgentCategory.EXTRACTOR,
        description="Extract door, window, finish, and fixture schedules. Detects schedule type, marks (D1, W2), sizes, quantities, and materials.",
        module_path="src.analysis.extractors.extract_schedule_tables",
        entry_fn="extract_schedule_rows",
        inputs=[
            AgentParam("text", "str", True, "OCR text from a schedule page"),
            AgentParam("source_page", "int", True, "Page number"),
            AgentParam("sheet_id", "str", False, "Sheet identifier"),
            AgentParam("pdf_path", "str", False, "PDF path for table extraction"),
        ],
        outputs=[
            AgentParam("schedule_rows", "List[dict]", description="Schedule rows with mark, type, size, qty, material"),
        ],
        tags=["schedule", "doors", "windows", "finishes"],
    ),

    AgentSpec(
        name="extract_commercial",
        label="Commercial Terms Extractor",
        category=AgentCategory.EXTRACTOR,
        description="Parse contract terms from conditions/spec pages: LD, retention, DLP, EMD, PBG, escalation, payment terms, penalties. 17 pattern types.",
        module_path="src.analysis.extractors.extract_commercial_terms",
        entry_fn="extract_commercial_terms",
        inputs=[
            AgentParam("text", "str", True, "OCR text from conditions page"),
            AgentParam("source_page", "int", True, "Page number"),
            AgentParam("sheet_id", "str", False, "Sheet identifier"),
            AgentParam("doc_type", "str", False, "Document type (default: conditions)"),
        ],
        outputs=[
            AgentParam("terms", "List[dict]", description="Contract terms with term_type, value, unit, snippet"),
        ],
        tags=["commercial", "gcc", "scc", "contract"],
    ),

    AgentSpec(
        name="extract_requirements",
        label="Requirements Extractor",
        category=AgentCategory.EXTRACTOR,
        description="Extract specification requirements from text-heavy pages. Finds IS/BS/ASTM codes, material grades, workmanship clauses, and India-specific patterns.",
        module_path="src.analysis.extractors.extract_notes",
        entry_fn="extract_requirements",
        inputs=[
            AgentParam("text", "str", True, "OCR text from spec/notes page"),
            AgentParam("source_page", "int", True, "Page number"),
            AgentParam("sheet_id", "str", False, "Sheet identifier"),
            AgentParam("doc_type", "str", True, "Document type"),
        ],
        outputs=[
            AgentParam("requirements", "List[dict]", description="Requirements with text, category, trade, codes"),
        ],
        tags=["requirements", "specifications", "standards"],
    ),

    AgentSpec(
        name="extract_callouts",
        label="Drawing Callouts Extractor",
        category=AgentCategory.EXTRACTOR,
        description="Extract dimension callouts, material references, tag references, and detail markers from architectural/structural drawing text.",
        module_path="src.analysis.extractors.extract_drawings_minimal",
        entry_fn="extract_drawing_callouts",
        inputs=[
            AgentParam("text", "str", True, "OCR text from drawing page"),
            AgentParam("source_page", "int", True, "Page number"),
        ],
        outputs=[
            AgentParam("callouts", "List[dict]", description="Drawing callouts with dimension, material, tag"),
        ],
        tags=["drawings", "callouts", "dimensions"],
    ),

    AgentSpec(
        name="excel_boq",
        label="Excel BOQ Parser",
        category=AgentCategory.EXTRACTOR,
        description="Parse BOQ line items from Excel files (.xlsx/.xls). Fuzzy column mapping, Indian number format (lakh), sheet detection, totals skipping.",
        module_path="src.analysis.excel_boq",
        entry_fn="parse_boq_excels",
        inputs=[
            AgentParam("excel_paths", "List[Path]", True, "Paths to Excel BOQ files"),
        ],
        outputs=[
            AgentParam("boq_items", "List[dict]", description="BOQ items from Excel"),
        ],
        tags=["boq", "excel", "xlsx"],
    ),

    # ─── Analysis Modules (10) ───────────────────────────────────────────

    AgentSpec(
        name="table_router",
        label="Table Extraction Router",
        category=AgentCategory.ANALYSIS,
        description="Multi-method table extraction with deterministic fallback: pdfplumber → camelot lattice → camelot stream → OCR row reconstruction.",
        module_path="src.analysis.table_router",
        entry_fn="extract_table_rows_from_page",
        inputs=[
            AgentParam("page_input", "str", False, "PDF path or page image"),
            AgentParam("page_meta", "dict", True, "Page metadata (page_number, doc_type)"),
            AgentParam("ocr_text", "str", True, "OCR text for the page"),
            AgentParam("config", "dict", False, "Extraction config"),
        ],
        outputs=[
            AgentParam("table_result", "TableExtractionResult", description="Extracted rows, headers, method used, confidence"),
        ],
        tags=["tables", "pdfplumber", "camelot"],
    ),

    AgentSpec(
        name="reconciliation",
        label="Quantity Reconciler",
        category=AgentCategory.ANALYSIS,
        description="Cross-check quantities from schedules, BOQ, and drawings. Flags mismatches > ±15% with delta % calculation for concrete, steel, masonry.",
        module_path="src.analysis.quantity_reconciliation",
        entry_fn="reconcile_quantities",
        inputs=[
            AgentParam("quantities", "List[dict]", True, "Unified quantity list from drawings"),
            AgentParam("schedules", "List[dict]", True, "Schedule rows"),
            AgentParam("boq_items", "List[dict]", True, "BOQ items"),
            AgentParam("callouts", "List[dict]", True, "Drawing callouts"),
        ],
        outputs=[
            AgentParam("reconciliation", "List[dict]", description="Reconciliation rows with mismatch flags and max_delta"),
        ],
        tags=["reconciliation", "quantities", "mismatch"],
    ),

    AgentSpec(
        name="conflicts",
        label="Conflict Detector",
        category=AgentCategory.ANALYSIS,
        description="Detect conflicts across BOQ, schedules, requirements, and callouts. Tags each conflict with resolution suggestions and delta confidence.",
        module_path="src.analysis.conflicts",
        entry_fn="detect_conflicts",
        inputs=[
            AgentParam("extraction_result", "dict", True, "Extraction result as dict"),
            AgentParam("page_index", "dict", False, "Page index as dict"),
        ],
        outputs=[
            AgentParam("conflicts", "List[dict]", description="Conflicts with type, severity, resolution"),
        ],
        tags=["conflicts", "detection"],
    ),

    AgentSpec(
        name="delta_detector",
        label="Addendum Delta Detector",
        category=AgentCategory.ANALYSIS,
        description="Detect changes between base and addendum BOQ items, schedules, and requirements. Computes delta confidence scores.",
        module_path="src.analysis.delta_detector",
        entry_fn="detect_boq_deltas",
        inputs=[
            AgentParam("base_items", "List[dict]", True, "Base document items"),
            AgentParam("addendum_items", "List[dict]", True, "Addendum document items"),
        ],
        outputs=[
            AgentParam("deltas", "List[dict]", description="Detected changes with delta_confidence"),
        ],
        tags=["addendum", "changes", "delta"],
    ),

    AgentSpec(
        name="qa_score",
        label="Quality Scorer",
        category=AgentCategory.ANALYSIS,
        description="Compute overall quality score (0-100) from extraction completeness, schedule presence, scale detection, conflict density, and coverage gaps.",
        module_path="src.analysis.qa_score",
        entry_fn="compute_qa_score",
        inputs=[
            AgentParam("payload", "dict", True, "Full analysis payload"),
        ],
        outputs=[
            AgentParam("qa_result", "dict", description="Score 0-100 with 5 sub-components and improvement actions"),
        ],
        tags=["quality", "scoring"],
    ),

    AgentSpec(
        name="pricing_guidance",
        label="Pricing Guidance",
        category=AgentCategory.ANALYSIS,
        description="Compute bid contingency range, recommended exclusions, clarifications, and VE suggestions based on analysis quality and conflicts.",
        module_path="src.analysis.pricing_guidance",
        entry_fn="compute_pricing_guidance",
        inputs=[
            AgentParam("qa_score", "dict", False, "QA score result"),
            AgentParam("addendum_index", "List[dict]", False, "Addenda list"),
            AgentParam("conflicts", "List[dict]", False, "Detected conflicts"),
            AgentParam("owner_profile", "dict", False, "Owner/client profile"),
            AgentParam("run_coverage", "dict", False, "Coverage metadata"),
        ],
        outputs=[
            AgentParam("guidance", "dict", description="Contingency range, exclusions, clarifications, VE suggestions"),
        ],
        tags=["pricing", "contingency", "exclusions"],
    ),

    AgentSpec(
        name="review_queue",
        label="Review Queue Builder",
        category=AgentCategory.ANALYSIS,
        description="Consolidate review items from 5 sources (reconciliation mismatches, conflicts, skipped pages, toxic pages, risk hits) into a unified priority queue.",
        module_path="src.analysis.review_queue",
        entry_fn="build_review_queue",
        inputs=[
            AgentParam("quantity_reconciliation", "List[dict]", True, "Reconciliation rows"),
            AgentParam("conflicts", "List[dict]", True, "Detected conflicts"),
            AgentParam("pages_skipped", "List[int]", True, "Skipped page indices"),
            AgentParam("toxic_summary", "dict", False, "Toxic clause summary"),
            AgentParam("risk_results", "List[dict]", True, "Risk checklist hits"),
        ],
        outputs=[
            AgentParam("review_items", "List[dict]", description="Unified review queue sorted by severity"),
        ],
        tags=["review", "curation"],
    ),

    AgentSpec(
        name="bulk_actions",
        label="Bulk Actions",
        category=AgentCategory.ANALYSIS,
        description="Batch operations: prefer schedule for mismatches, generate RFIs for high-impact items, mark intentional revisions reviewed.",
        module_path="src.analysis.bulk_actions",
        entry_fn="generate_rfis_for_high_mismatches",
        inputs=[
            AgentParam("recon_rows", "List[dict]", True, "Reconciliation rows"),
            AgentParam("existing_rfis", "List[dict]", True, "Current RFI list"),
        ],
        outputs=[
            AgentParam("new_rfis", "List[dict]", description="Auto-generated RFIs for high-impact mismatches"),
            AgentParam("updated_rows", "List[dict]", description="Updated reconciliation rows"),
        ],
        can_run_standalone=False,
        tags=["batch", "bulk"],
    ),

    AgentSpec(
        name="rfi_clustering",
        label="RFI Clusterer",
        category=AgentCategory.ANALYSIS,
        description="Group similar RFIs by trade, page overlap, and text similarity. Reduces noise and merges duplicate questions.",
        module_path="src.analysis.rfi_clustering",
        entry_fn="cluster_rfis",
        inputs=[
            AgentParam("rfis", "List[dict]", True, "List of generated RFIs"),
        ],
        outputs=[
            AgentParam("clusters", "List[dict]", description="Clustered RFIs with merged questions"),
        ],
        tags=["rfi", "dedup", "clustering"],
    ),

    AgentSpec(
        name="meeting_agenda",
        label="Meeting Agenda Builder",
        category=AgentCategory.ANALYSIS,
        description="Auto-generate pre-bid meeting agenda from review queue items and team assignments.",
        module_path="src.analysis.meeting_agenda",
        entry_fn="build_meeting_agenda",
        inputs=[
            AgentParam("review_items", "List[dict]", True, "Review queue items"),
            AgentParam("assignments", "List[dict]", False, "Team assignments"),
        ],
        outputs=[
            AgentParam("agenda", "dict", description="Meeting agenda with sections for priorities, RFIs, assumptions"),
        ],
        tags=["meeting", "agenda", "collaboration"],
    ),

    # ─── Structural Pipeline (6) ─────────────────────────────────────────

    AgentSpec(
        name="structural_pipeline",
        label="Structural Analysis Pipeline",
        category=AgentCategory.STRUCTURAL,
        description="Full structural quantity estimation: detect columns, beams, footings from floor plans, compute concrete volumes and steel weights.",
        module_path="src.structural.pipeline_structural",
        entry_fn="StructuralPipeline",
        inputs=[
            AgentParam("pdf_path", "Path", True, "Path to structural PDF"),
            AgentParam("mode", "str", False, "Processing mode: auto, assumption, structural"),
            AgentParam("floors", "int", False, "Number of floors (default: 1)"),
        ],
        outputs=[
            AgentParam("result", "StructuralPipelineResult", description="Detected elements, quantities, QC report"),
        ],
        tags=["structural", "rcc", "quantities"],
    ),

    AgentSpec(
        name="detect_columns",
        label="Column Detector",
        category=AgentCategory.STRUCTURAL,
        description="Image-based column detection using morphological operations and keyword matching on floor plan images.",
        module_path="src.structural.detect_columns",
        entry_fn="ColumnDetector",
        inputs=[
            AgentParam("image", "ndarray", True, "Floor plan image (OpenCV)"),
        ],
        outputs=[
            AgentParam("result", "ColumnDetectionResult", description="Detected columns with positions and sizes"),
        ],
        can_run_standalone=False,
        tags=["structural", "columns", "cv2"],
    ),

    AgentSpec(
        name="detect_beams",
        label="Beam Detector",
        category=AgentCategory.STRUCTURAL,
        description="Detect beams as thick lines connecting columns. Builds column-beam connectivity graph.",
        module_path="src.structural.detect_beams",
        entry_fn="BeamDetector",
        inputs=[
            AgentParam("image", "ndarray", True, "Floor plan image (OpenCV)"),
            AgentParam("columns", "List[DetectedColumn]", True, "Detected columns"),
        ],
        outputs=[
            AgentParam("result", "BeamDetectionResult", description="Detected beams with connectivity"),
        ],
        can_run_standalone=False,
        tags=["structural", "beams", "cv2"],
    ),

    AgentSpec(
        name="detect_footings",
        label="Footing Detector",
        category=AgentCategory.STRUCTURAL,
        description="Foundation type detection and size extraction from structural drawings.",
        module_path="src.structural.detect_footings",
        entry_fn="FootingDetector",
        inputs=[
            AgentParam("image", "ndarray", True, "Floor plan image (OpenCV)"),
        ],
        outputs=[
            AgentParam("result", "FootingDetectionResult", description="Detected footings with types and sizes"),
        ],
        can_run_standalone=False,
        tags=["structural", "footings", "foundation"],
    ),

    AgentSpec(
        name="quantity_engine",
        label="Structural Quantity Engine",
        category=AgentCategory.STRUCTURAL,
        description="Compute concrete volumes and steel quantities from detected structural elements (columns, beams, footings).",
        module_path="src.structural.quantity_engine",
        entry_fn="QuantityEngine",
        inputs=[
            AgentParam("elements", "List[dict]", True, "Detected structural elements"),
            AgentParam("floors", "int", False, "Number of floors"),
        ],
        outputs=[
            AgentParam("quantities", "QuantitySummary", description="Concrete m³, steel kg, per-element breakdown"),
        ],
        can_run_standalone=False,
        tags=["structural", "concrete", "steel"],
    ),

    AgentSpec(
        name="steel_estimator",
        label="Steel Estimator",
        category=AgentCategory.STRUCTURAL,
        description="Steel weight estimation from BBS data or kg-per-m³ assumptions for different structural elements.",
        module_path="src.structural.steel_estimator",
        entry_fn="SteelEstimator",
        inputs=[
            AgentParam("elements", "List[dict]", True, "Structural elements with dimensions"),
        ],
        outputs=[
            AgentParam("steel_estimate", "dict", description="Steel weight estimates by element type"),
        ],
        can_run_standalone=False,
        tags=["structural", "steel", "bbs"],
    ),

    # ─── Output Generators (5) ───────────────────────────────────────────

    AgentSpec(
        name="bid_summary",
        label="Bid Summary Generator",
        category=AgentCategory.OUTPUT,
        description="Generate 1-2 page markdown bid summary from analysis payload. Includes at-a-glance, cost drivers, BOQ completeness, reconciliation table.",
        module_path="app.bid_summary",
        entry_fn="generate_bid_summary_markdown",
        inputs=[
            AgentParam("payload", "dict", True, "Full analysis payload"),
            AgentParam("bid_strategy", "dict", False, "Bid strategy parameters"),
            AgentParam("assumptions", "List[dict]", False, "Estimator assumptions"),
        ],
        outputs=[
            AgentParam("markdown", "str", description="Formatted markdown bid summary"),
        ],
        tags=["summary", "markdown", "report"],
    ),

    AgentSpec(
        name="bid_summary_pdf",
        label="Bid Summary PDF",
        category=AgentCategory.OUTPUT,
        description="Render bid summary markdown as a formatted PDF document.",
        module_path="app.bid_summary_pdf",
        entry_fn="generate_bid_summary_pdf",
        inputs=[
            AgentParam("payload", "dict", True, "Full analysis payload"),
        ],
        outputs=[
            AgentParam("pdf_bytes", "bytes", description="PDF document as bytes"),
        ],
        can_run_standalone=False,
        tags=["pdf", "report"],
    ),

    AgentSpec(
        name="submission_pack",
        label="Submission Pack Builder",
        category=AgentCategory.OUTPUT,
        description="Build ZIP submission pack with 6 folders: Bid Summary, RFIs, Exclusions, Quantities, Evidence Appendix.",
        module_path="app.submission_pack",
        entry_fn="generate_submission_pack",
        inputs=[
            AgentParam("csv_buffers", "dict", True, "Dict of filename → content"),
            AgentParam("project_id", "str", False, "Project identifier"),
            AgentParam("project_name", "str", False, "Project display name"),
        ],
        outputs=[
            AgentParam("zip_bytes", "bytes", description="ZIP file as bytes"),
        ],
        can_run_standalone=False,
        tags=["zip", "export", "submission"],
    ),

    AgentSpec(
        name="docx_exports",
        label="DOCX Exporter",
        category=AgentCategory.OUTPUT,
        description="Generate DOCX exports for RFI log, exclusions/clarifications, and bid summary.",
        module_path="app.docx_exports",
        entry_fn="generate_rfis_docx",
        inputs=[
            AgentParam("rfis", "List[dict]", True, "RFI list"),
        ],
        outputs=[
            AgentParam("docx_bytes", "bytes", description="DOCX document as bytes"),
        ],
        can_run_standalone=False,
        tags=["docx", "export"],
    ),

    AgentSpec(
        name="evidence_appendix",
        label="Evidence Appendix PDF",
        category=AgentCategory.OUTPUT,
        description="Generate evidence appendix PDF with page references and supporting images for RFIs and blockers.",
        module_path="app.evidence_appendix_pdf",
        entry_fn="generate_evidence_appendix_pdf",
        inputs=[
            AgentParam("payload", "dict", True, "Full analysis payload"),
            AgentParam("pdf_path", "Path", False, "Source PDF for page images"),
        ],
        outputs=[
            AgentParam("pdf_bytes", "bytes", description="Evidence appendix PDF as bytes"),
        ],
        can_run_standalone=False,
        tags=["evidence", "pdf", "appendix"],
    ),

    # ─── Project Management (5) ──────────────────────────────────────────

    AgentSpec(
        name="projects",
        label="Project Manager",
        category=AgentCategory.PROJECT,
        description="Project CRUD operations. Create, list, load, and update projects stored in ~/.xboq/projects/.",
        module_path="src.analysis.projects",
        entry_fn="list_projects",
        inputs=[],
        outputs=[
            AgentParam("projects", "List[dict]", description="List of project metadata"),
        ],
        can_run_standalone=False,
        tags=["project", "crud"],
    ),

    AgentSpec(
        name="collaboration",
        label="Collaboration Manager",
        category=AgentCategory.PROJECT,
        description="Comments, assignments, and due dates on any entity (RFI, conflict, assumption, quantity). Append-only JSONL storage.",
        module_path="src.analysis.collaboration",
        entry_fn="make_collaboration_entry",
        inputs=[
            AgentParam("entity_type", "str", True, "Entity type (rfi, conflict, assumption, quantity)"),
            AgentParam("entity_id", "str", True, "Entity identifier"),
            AgentParam("action", "str", True, "Action type (comment, assign, due_date)"),
        ],
        outputs=[
            AgentParam("entry", "dict", description="Collaboration entry dict"),
        ],
        can_run_standalone=False,
        tags=["collaboration", "comments", "assignments"],
    ),

    AgentSpec(
        name="playbooks",
        label="Company Playbooks",
        category=AgentCategory.PROJECT,
        description="Company estimating strategy storage: exclusions, assumptions, VE standards, contingency adjustments. Stored in ~/.xboq/playbooks/.",
        module_path="src.analysis.company_playbooks",
        entry_fn="load_playbook",
        inputs=[
            AgentParam("company_name", "str", True, "Company name"),
        ],
        outputs=[
            AgentParam("playbook", "dict", description="Playbook with exclusions, assumptions, VE items"),
        ],
        can_run_standalone=False,
        tags=["playbook", "strategy", "company"],
    ),

    AgentSpec(
        name="owner_profiles",
        label="Owner Profiles",
        category=AgentCategory.PROJECT,
        description="Owner/client relationship tracking: relationship type, past work history, dispute record, preferences.",
        module_path="src.analysis.owner_profiles",
        entry_fn="load_profile",
        inputs=[
            AgentParam("owner_name", "str", True, "Owner/client name"),
        ],
        outputs=[
            AgentParam("profile", "dict", description="Owner profile with history and preferences"),
        ],
        can_run_standalone=False,
        tags=["owner", "client", "profile"],
    ),

    AgentSpec(
        name="approval_states",
        label="Approval State Tracker",
        category=AgentCategory.PROJECT,
        description="Status tracking for RFIs (draft/approved/sent), quantities (draft/confirmed), and conflicts (unreviewed/reviewed).",
        module_path="src.analysis.approval_states",
        entry_fn="set_rfi_status",
        inputs=[
            AgentParam("rfi", "dict", True, "RFI dict"),
            AgentParam("status", "str", True, "New status (draft, approved, sent, rejected)"),
        ],
        outputs=[
            AgentParam("updated_rfi", "dict", description="RFI with updated status"),
        ],
        can_run_standalone=False,
        tags=["approval", "status", "workflow"],
    ),
]


# =============================================================================
# LOOKUP HELPERS
# =============================================================================

_BY_NAME: Dict[str, AgentSpec] = {a.name: a for a in AGENTS}
_BY_CATEGORY: Dict[AgentCategory, List[AgentSpec]] = {}
for _a in AGENTS:
    _BY_CATEGORY.setdefault(_a.category, []).append(_a)


def get_agent(name: str) -> Optional[AgentSpec]:
    """Get agent spec by slug name. Returns None if not found."""
    return _BY_NAME.get(name)


def list_agents(category: Optional[str] = None) -> List[AgentSpec]:
    """List agents, optionally filtered by category slug."""
    if category:
        cat = AgentCategory(category)
        return list(_BY_CATEGORY.get(cat, []))
    return list(AGENTS)


def list_categories() -> List[str]:
    """Return sorted list of category slugs."""
    return sorted(c.value for c in AgentCategory)


def resolve_fn(spec: AgentSpec) -> Callable:
    """
    Import and return the entry function/class for an agent.

    Uses lazy importing so the registry can be loaded without importing
    heavy dependencies (cv2, numpy, etc.).
    """
    import importlib
    mod = importlib.import_module(spec.module_path)
    return getattr(mod, spec.entry_fn)


def agent_count() -> int:
    """Total number of registered agents."""
    return len(AGENTS)


def standalone_agents() -> List[AgentSpec]:
    """Return only agents that can run standalone."""
    return [a for a in AGENTS if a.can_run_standalone]
