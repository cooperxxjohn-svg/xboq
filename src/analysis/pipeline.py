"""
Analysis Pipeline

Provides a streamlit-compatible interface to run the analysis pipeline
with progress callbacks for real-time UI updates.
"""

import json
import sys
import traceback
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class AnalysisStage:
    """Represents a stage in the analysis pipeline."""
    id: str
    name: str
    description: str
    progress: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    message: str = ""
    error: str = ""


@dataclass
class AnalysisResult:
    """Result of the analysis pipeline."""
    success: bool
    project_id: str
    output_dir: Path
    stages: List[AnalysisStage] = field(default_factory=list)
    error_message: str = ""
    stack_trace: str = ""
    files_generated: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    # Analysis payload for immediate UI display
    payload: Dict[str, Any] = field(default_factory=dict)


def load_pdf_pages(pdf_path: Path) -> Tuple[List[str], int]:
    """
    Load PDF and extract raw text from each page (without OCR).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of:
            - List of text strings, one per page
            - Page count
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page_texts = []
        for page in doc:
            text = page.get_text()
            page_texts.append(text)
        page_count = len(page_texts)
        doc.close()
        return page_texts, page_count
    except Exception as e:
        return [], 0


def run_ocr_extraction(
    pdf_path: Path,
    page_texts: List[str],
    preflight: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable] = None,
    selected_pages: Optional[List[int]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Run OCR extraction on pages that need it, with adaptive strategy.

    Args:
        pdf_path: Path to PDF file
        page_texts: List of already-extracted text per page
        preflight: Optional preflight result (from pdf_preflight)
        progress_callback: Optional callable(page_idx, total, message)
        selected_pages: Optional list of page indices to OCR (from page_selection)

    Returns:
        Tuple of:
            - Updated list of text strings (with OCR results merged)
            - OCR metadata dict
    """
    metadata = {
        'ocr_used': False,
        'ocr_pages': [],
        'scales_from_ocr': [],
        'disciplines_from_ocr': [],
        'page_profiles': [],
        'preflight': None,
    }

    try:
        from .ocr_fallback import (
            is_ocr_available,
            extract_ocr_for_all_pages,
            page_needs_ocr,
            pdf_preflight as _pdf_preflight,
        )

        if not is_ocr_available():
            return page_texts, metadata

        # Run preflight if not provided
        if preflight is None:
            preflight = _pdf_preflight(pdf_path, page_texts)
        metadata['preflight'] = preflight

        strategy = preflight.get('ocr_strategy', 'full')

        # Adaptive strategy: skip OCR if most pages have text
        if strategy == 'skip':
            return page_texts, metadata

        # Determine max pages to OCR
        max_ocr_pages = 0  # unlimited
        if strategy == 'adaptive':
            max_ocr_pages = 80  # hard cap for very large scanned PDFs

        # If selected_pages provided, use them directly (overrides max_ocr_pages)
        if selected_pages is not None:
            max_ocr_pages = 0  # no cap needed, selection already limited

        page_texts, ocr_metadata = extract_ocr_for_all_pages(
            pdf_path, page_texts, dpi=150,
            progress_callback=progress_callback,
            max_ocr_pages=max_ocr_pages,
            selected_pages=selected_pages,
        )
        metadata['ocr_used'] = len(ocr_metadata.get('pages_with_ocr', [])) > 0
        metadata['ocr_pages'] = ocr_metadata.get('pages_with_ocr', [])
        metadata['scales_from_ocr'] = ocr_metadata.get('scales_detected', [])
        metadata['disciplines_from_ocr'] = ocr_metadata.get('disciplines_detected', [])
        metadata['page_profiles'] = ocr_metadata.get('page_profiles', [])

    except ImportError:
        # OCR module not available, continue without
        pass

    return page_texts, metadata


def extract_text_from_pdf(pdf_path: Path, use_ocr_fallback: bool = True) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract text from each page of a PDF, with optional OCR fallback.
    (Legacy wrapper for backward compatibility)

    Args:
        pdf_path: Path to PDF file
        use_ocr_fallback: Whether to use OCR for pages with no text

    Returns:
        Tuple of:
            - List of text strings, one per page
            - Dict with extraction metadata (OCR info, etc.)
    """
    page_texts, _ = load_pdf_pages(pdf_path)

    if use_ocr_fallback:
        page_texts, metadata = run_ocr_extraction(pdf_path, page_texts)
    else:
        metadata = {
            'ocr_used': False,
            'ocr_pages': [],
            'scales_from_ocr': [],
            'disciplines_from_ocr': [],
        }

    return page_texts, metadata


def _compute_run_compare(prev: Optional[dict], current: dict) -> Optional[dict]:
    """
    Compute diffs between the previous cached run and the current run.

    Returns None if no previous run available.
    """
    if not prev or not isinstance(prev, dict):
        return None

    result = {}

    # QA score delta
    prev_qa = (prev.get("qa_score") or {}).get("score")
    curr_qa = (current.get("qa_score") or {}).get("score")
    if prev_qa is not None and curr_qa is not None:
        result["qa_score_delta"] = curr_qa - prev_qa
    else:
        result["qa_score_delta"] = None

    # New RFIs (questions in current not in previous)
    prev_questions = {r.get("question", "") for r in (prev.get("rfis") or [])}
    curr_rfis = current.get("rfis") or []
    new_rfis = [
        r.get("question", "")[:100]
        for r in curr_rfis
        if r.get("question", "") and r.get("question", "") not in prev_questions
    ]
    result["new_rfis"] = new_rfis[:20]

    # New conflicts
    prev_descs = {c.get("description", "") for c in (prev.get("conflicts") or [])}
    curr_conflicts = current.get("conflicts") or []
    new_conflicts = [
        c.get("description", "")[:100]
        for c in curr_conflicts
        if c.get("description", "") and c.get("description", "") not in prev_descs
    ]
    result["new_conflicts"] = new_conflicts[:20]

    # BOQ / quantities delta
    prev_qty_count = prev.get("quantities_count", 0)
    curr_qty_count = len(current.get("quantities") or [])
    result["boq_delta_summary"] = {
        "prev_count": prev_qty_count,
        "curr_count": curr_qty_count,
        "delta": curr_qty_count - prev_qty_count,
    }

    # Sprint 13: Approval deltas
    prev_rfi_statuses = prev.get("rfi_status_snapshot", {})
    curr_rfi_statuses = current.get("rfi_status_snapshot", {})
    newly_approved = [
        rfi_id for rfi_id, status in curr_rfi_statuses.items()
        if status in ("approved", "sent") and prev_rfi_statuses.get(rfi_id) == "draft"
    ]
    result["newly_approved_rfis"] = newly_approved

    prev_qty_accepted = prev.get("quantities_accepted_count", 0)
    curr_qty_accepted = current.get("quantities_accepted_count", 0)
    result["qty_accepted_delta"] = curr_qty_accepted - prev_qty_accepted

    prev_conflicts_reviewed = prev.get("conflicts_reviewed_count", 0)
    curr_conflicts_reviewed = current.get("conflicts_reviewed_count", 0)
    result["conflicts_reviewed_delta"] = curr_conflicts_reviewed - prev_conflicts_reviewed

    return result


def _check_guardrails(page_index, extraction_result, rfis) -> List[dict]:
    """
    Post-analysis guardrails: warn if extraction seems incomplete.

    Returns list of warning dicts.
    """
    warnings = []

    if not page_index or not extraction_result:
        return warnings

    # Schedule pages indexed but nothing extracted
    schedule_count = page_index.counts_by_type.get("schedule", 0)
    if schedule_count > 0 and len(extraction_result.schedules) == 0:
        warnings.append({
            "type": "schedule_extraction_gap",
            "message": (
                f"{schedule_count} schedule pages indexed but 0 schedule rows extracted. "
                "OCR quality may be poor on these pages."
            ),
        })

    # BOQ pages indexed but very few items extracted
    boq_count = page_index.counts_by_type.get("boq", 0)
    if boq_count > 2 and len(extraction_result.boq_items) < 3:
        warnings.append({
            "type": "boq_extraction_gap",
            "message": (
                f"{boq_count} BOQ pages indexed but only {len(extraction_result.boq_items)} "
                "items extracted. BOQ pages may be tabular images that OCR cannot parse."
            ),
        })

    # Low RFI count on large tenders
    if page_index.total_pages > 100 and len(rfis) < 10:
        warnings.append({
            "type": "low_rfi_count",
            "message": (
                f"Only {len(rfis)} RFIs generated for a {page_index.total_pages}-page tender. "
                "Expected >15. Extraction may be incomplete."
            ),
        })

    # No requirements extracted from notes/spec pages
    notes_count = (
        page_index.counts_by_type.get("notes", 0) +
        page_index.counts_by_type.get("spec", 0) +
        page_index.counts_by_type.get("conditions", 0)
    )
    if notes_count > 3 and len(extraction_result.requirements) < 3:
        warnings.append({
            "type": "notes_extraction_gap",
            "message": (
                f"{notes_count} notes/spec/conditions pages indexed but only "
                f"{len(extraction_result.requirements)} requirements extracted."
            ),
        })

    return warnings


# ── Sprint 19: BOQ Stats + Requirements by Trade ─────────────────────────

def _compute_boq_stats(boq_items: list) -> dict:
    """Compute summary statistics for BOQ items.

    Returns: {total_items, by_trade, flagged_items, flagged_count}
    """
    if not boq_items:
        return {"total_items": 0, "by_trade": {}, "flagged_items": [], "flagged_count": 0}

    by_trade: dict = {}
    flagged = []
    for item in boq_items:
        trade = item.get("trade", "general")
        by_trade[trade] = by_trade.get(trade, 0) + 1
        flags = item.get("flags", [])
        if flags:
            flagged.append({
                "item_no": item.get("item_no", ""),
                "description": (item.get("description") or "")[:80],
                "flags": flags,
            })

    return {
        "total_items": len(boq_items),
        "by_trade": by_trade,
        "flagged_items": flagged,
        "flagged_count": len(flagged),
    }


def _build_requirements_by_trade(requirements: list) -> dict:
    """Group requirements by their 'trade' field.

    Returns: {trade: [req_dicts]}
    """
    if not requirements:
        return {}
    by_trade: dict = {}
    for req in requirements:
        trade = req.get("trade", "general")
        by_trade.setdefault(trade, []).append(req)
    return by_trade


# ── Sprint 18: Processing Stats Builder ───────────────────────────────────

def _build_processing_stats(
    page_index_result=None,
    selected_result=None,
    ocr_metadata=None,
    run_coverage=None,
    toxic_summary=None,
    file_info=None,
    extraction_result=None,
) -> dict:
    """Compute reliable page-processing counters from pipeline data.

    Returns a stable dict with:
        total_pages, deep_processed_pages, ocr_pages, text_layer_pages,
        skipped_pages, per_doc (if multi-doc).
        Sprint 20F: Also table_attempt_pages, table_success_pages,
        selection_mode, selected_pages_count.
    All fields default to 0 if data is unavailable.
    """
    stats = {
        "total_pages": 0,
        "deep_processed_pages": 0,
        "ocr_pages": 0,
        "text_layer_pages": 0,
        "skipped_pages": 0,
        "per_doc": {},
        # Sprint 20F additions
        "table_attempt_pages": 0,
        "table_success_pages": 0,
        "selection_mode": None,
        "selected_pages_count": 0,
    }

    try:
        # Total pages from page_index (most reliable source)
        if page_index_result and hasattr(page_index_result, 'total_pages'):
            stats["total_pages"] = page_index_result.total_pages
        elif run_coverage and hasattr(run_coverage, 'pages_total'):
            stats["total_pages"] = run_coverage.pages_total

        # Deep processed = pages that went through full extraction (selected)
        if selected_result and hasattr(selected_result, 'selected'):
            stats["deep_processed_pages"] = len(selected_result.selected)
        elif run_coverage and hasattr(run_coverage, 'pages_deep_processed'):
            stats["deep_processed_pages"] = run_coverage.pages_deep_processed

        # OCR pages vs text-layer pages from stage2 profiles
        ocr_metadata = ocr_metadata or {}
        profiles = ocr_metadata.get("page_profiles", [])
        if profiles:
            stats["ocr_pages"] = sum(1 for p in profiles if p.get("ocr_used"))
            stats["text_layer_pages"] = sum(
                1 for p in profiles
                if p.get("has_text_layer") and not p.get("ocr_used")
            )
        else:
            # Fallback: if all selected pages were OCR'd (common for scanned docs)
            ocr_page_list = ocr_metadata.get("ocr_pages", [])
            stats["ocr_pages"] = len(ocr_page_list)
            # Text-layer = deep_processed minus OCR
            stats["text_layer_pages"] = max(
                0, stats["deep_processed_pages"] - stats["ocr_pages"]
            )

        # Skipped pages = total minus deep-processed
        stats["skipped_pages"] = max(
            0, stats["total_pages"] - stats["deep_processed_pages"]
        )

        # Toxic/failed pages (subset of skipped)
        if toxic_summary and isinstance(toxic_summary, dict):
            stats["toxic_pages"] = toxic_summary.get("total_toxic", 0)
        elif toxic_summary and isinstance(toxic_summary, list):
            stats["toxic_pages"] = len(toxic_summary)
        else:
            stats["toxic_pages"] = 0

        # Per-doc breakdown (from file_info if available)
        if file_info and isinstance(file_info, list):
            for fi in file_info:
                if not isinstance(fi, dict):
                    continue
                doc_id = fi.get("filename", fi.get("doc_id", "unknown"))
                doc_pages = fi.get("pages", 0)
                doc_ocr = fi.get("ocr_pages", 0)
                stats["per_doc"][doc_id] = {
                    "total_pages": doc_pages,
                    "ocr_pages": doc_ocr,
                }

        # Sprint 20F: Selection mode + selected count
        if selected_result and hasattr(selected_result, 'selected'):
            stats["selected_pages_count"] = len(selected_result.selected)
        if run_coverage:
            sel_mode = None
            if hasattr(run_coverage, 'selection_mode'):
                sel_mode = run_coverage.selection_mode
            elif isinstance(run_coverage, dict):
                sel_mode = run_coverage.get("selection_mode")
            stats["selection_mode"] = sel_mode

        # Sprint 20F: Table attempt/success from extraction_diagnostics
        if extraction_result:
            diag = None
            if hasattr(extraction_result, 'extraction_diagnostics'):
                diag = extraction_result.extraction_diagnostics
            elif isinstance(extraction_result, dict):
                diag = extraction_result.get("extraction_diagnostics")
            if diag and isinstance(diag, dict):
                boq_diag = diag.get("boq", {})
                sched_diag = diag.get("schedules", {})
                stats["table_attempt_pages"] = (
                    boq_diag.get("pages_attempted", 0) +
                    sched_diag.get("pages_attempted", 0)
                )
                stats["table_success_pages"] = (
                    boq_diag.get("pages_parsed", 0) +
                    sched_diag.get("pages_parsed", 0)
                )

    except Exception:
        pass  # Return whatever we computed so far

    return stats


def run_analysis_pipeline(
    input_files: List[Path],
    project_id: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
    run_mode: Optional[str] = None,
    boq_excel_paths: Optional[List[Path]] = None,
) -> AnalysisResult:
    """
    Run the analysis pipeline on uploaded files.

    Args:
        input_files: List of PDF file paths
        project_id: Project identifier
        output_dir: Directory to write output files
        progress_callback: Callback(stage_id, message, progress_pct) for UI updates
        run_mode: Sprint 20G — one of "demo_fast", "standard_review", "full_audit".
                  Defaults to "demo_fast" if not provided (backward compatible).

    Returns:
        AnalysisResult with success status and generated files
    """
    start_time = datetime.now()

    # Sprint 20G: Resolve run_mode to deep_cap / force_full_read
    from src.models.analysis_models import RunMode
    _run_mode_enum = RunMode.DEMO_FAST  # default: backward compatible
    if run_mode:
        try:
            _run_mode_enum = RunMode(run_mode)
        except ValueError:
            _run_mode_enum = RunMode.DEMO_FAST
    _deep_cap = _run_mode_enum.deep_cap          # 80 | 220 | None
    _force_full = _run_mode_enum == RunMode.FULL_AUDIT

    # Define stages (8-stage pipeline)
    stages = [
        AnalysisStage("load", "Load PDFs", "Loading and indexing PDF files"),
        AnalysisStage("index", "Index Pages", "Classifying every page by type and discipline"),
        AnalysisStage("select", "Select Pages", "Prioritizing pages within OCR budget"),
        AnalysisStage("extract", "Extract Text", "OCR + specialized extraction"),
        AnalysisStage("graph", "Build Graph", "Building plan set graph structure"),
        AnalysisStage("reason", "Analyze Dependencies", "Detecting blockers and gaps"),
        AnalysisStage("rfi", "Generate RFIs", "Creating evidence-backed RFI recommendations"),
        AnalysisStage("export", "Export Results", "Saving analysis outputs"),
    ]

    result = AnalysisResult(
        success=False,
        project_id=project_id,
        output_dir=output_dir,
        stages=stages,
    )

    def update_progress(stage_id: str, message: str, progress: float):
        """Update stage progress and call callback."""
        for stage in result.stages:
            if stage.id == stage_id:
                stage.message = message
                stage.progress = progress
                if progress >= 1.0:
                    stage.status = "completed"
                elif progress > 0:
                    stage.status = "running"
                break
        if progress_callback:
            progress_callback(stage_id, message, progress)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialise run_coverage early — referenced in addendum delta block
        run_coverage = None

        # Stage timing
        stage_times = {}

        # =================================================================
        # STAGE 1: Load PDFs (just open and index files)
        # =================================================================
        import time
        stage_start = time.perf_counter()
        update_progress("load", "Scanning input files...", 0.1)

        all_raw_texts = []  # Raw text before OCR
        file_info = []
        pdf_paths_for_ocr = []  # Track which files need OCR processing

        for i, pdf_path in enumerate(input_files):
            update_progress("load", f"Loading {pdf_path.name}...", 0.2 + (0.6 * i / max(len(input_files), 1)))

            try:
                page_texts, page_count = load_pdf_pages(pdf_path)
                all_raw_texts.append((pdf_path, page_texts))
                file_info.append({
                    "name": pdf_path.name,
                    "pages": page_count,
                    "path": str(pdf_path),
                    "ocr_used": False,  # Will be updated in extract stage
                })

            except Exception as e:
                file_info.append({
                    "name": pdf_path.name,
                    "pages": 0,
                    "error": str(e),
                })

        total_pages = sum(len(texts) for _, texts in all_raw_texts)
        update_progress("load", f"Loaded {total_pages} pages from {len(input_files)} files", 1.0)
        stage_times["load"] = time.perf_counter() - stage_start

        if total_pages == 0:
            raise ValueError("No pages could be loaded from the uploaded PDFs. Please ensure they are readable.")

        # Sprint 9: Build multi-doc index
        from .multi_doc import build_multi_doc_index
        multi_doc_index = build_multi_doc_index(file_info)

        # Sprint 10: Setup persistent cache
        from .pipeline_cache import (
            compute_cache_key, get_cache_dir,
            load_cached_stage, save_cached_stage,
            build_cache_stats, _measure_cache_bytes,
        )
        _pdf_paths_for_cache = [pdf_path for pdf_path, _ in all_raw_texts]
        _cache_config = {"dpi": 150, "budget_pages": _deep_cap if _deep_cap is not None else 80, "run_mode": _run_mode_enum.value}
        cache_key = compute_cache_key(_pdf_paths_for_cache, _cache_config)
        cache_dir = get_cache_dir(output_dir, cache_key)
        _cache_hits, _cache_misses = [], []
        _cache_time_saved = 0.0

        # =================================================================
        # STAGE 2: Index Pages (header-strip OCR classification)
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("index", "Classifying pages by type and discipline...", 0.1)

        from .page_index import build_page_index, PageIndex, IndexedPage
        from .page_selection import select_pages, SelectedPages
        from .extractors import run_extractors, ExtractionResult
        from .rfi_engine import generate_rfis
        from collections import Counter

        page_index_result = None
        selected_result = None
        extraction_result = None

        # Sprint 10: Try loading page_index from cache
        _cache_pi_loaded = False
        _cached_pi = load_cached_stage(cache_dir, "page_index")
        if _cached_pi and all_raw_texts:
            try:
                primary_pdf_path, primary_raw_texts = all_raw_texts[0]
                page_index_result = PageIndex(
                    pdf_name=_cached_pi.get("pdf_name", ""),
                    total_pages=_cached_pi.get("total_pages", 0),
                    pages=[
                        IndexedPage(**{k: v for k, v in p.items()
                                       if k in IndexedPage.__dataclass_fields__})
                        for p in _cached_pi.get("pages", [])
                    ],
                    counts_by_type=_cached_pi.get("counts_by_type", {}),
                    counts_by_discipline=_cached_pi.get("counts_by_discipline", {}),
                    indexing_time_s=_cached_pi.get("indexing_time_s", 0),
                )
                _cache_hits.append("page_index")
                _cache_time_saved += _cached_pi.get("indexing_time_s", 0)
                update_progress("index", f"Page index loaded from cache ({page_index_result.total_pages} pages)", 1.0)
                stage_times["index"] = 0.01
                _cache_pi_loaded = True
            except Exception:
                pass  # Cache corrupted, fall through to fresh index

        # Sprint 9: Index ALL PDFs (not just the first)
        if all_raw_texts and not _cache_pi_loaded:
            primary_pdf_path, primary_raw_texts = all_raw_texts[0]
            all_page_indices = []

            for pdf_i, (pdf_path_i, raw_texts_i) in enumerate(all_raw_texts):
                def index_progress(page_idx, total, msg, _i=pdf_i):
                    pct = 0.1 + 0.85 * (page_idx + 1) / max(total, 1)
                    update_progress("index", msg, min(pct, 0.95))

                pi = build_page_index(
                    pdf_path_i, raw_texts_i,
                    progress_cb=index_progress,
                )
                all_page_indices.append(pi)

            page_index_result = all_page_indices[0]

            # Stitch additional PDFs into combined PageIndex with global offsets
            if len(all_page_indices) > 1:
                combined_pages = list(page_index_result.pages)
                offset = page_index_result.total_pages
                for pi_extra in all_page_indices[1:]:
                    for p in pi_extra.pages:
                        adjusted = IndexedPage(
                            page_idx=p.page_idx + offset,
                            doc_type=p.doc_type,
                            discipline=p.discipline,
                            sheet_id=p.sheet_id,
                            title=p.title,
                            confidence=p.confidence,
                            keywords_hit=p.keywords_hit,
                            has_text_layer=p.has_text_layer,
                            strip_ocr_time_s=p.strip_ocr_time_s,
                        )
                        combined_pages.append(adjusted)
                    offset += pi_extra.total_pages
                type_counter = Counter(p.doc_type for p in combined_pages)
                disc_counter = Counter(p.discipline for p in combined_pages)
                page_index_result = PageIndex(
                    pdf_name=page_index_result.pdf_name,
                    total_pages=offset,
                    pages=combined_pages,
                    counts_by_type=dict(type_counter),
                    counts_by_discipline=dict(disc_counter),
                    indexing_time_s=sum(pi.indexing_time_s for pi in all_page_indices),
                )

            update_progress(
                "index",
                page_index_result.summary_line(),
                1.0,
            )
        if "index" not in stage_times:
            stage_times["index"] = time.perf_counter() - stage_start

        # Sprint 10: Save page_index to cache if freshly computed
        if page_index_result and not _cache_pi_loaded:
            try:
                save_cached_stage(cache_dir, "page_index", page_index_result.to_dict())
                _cache_misses.append("page_index")
            except Exception:
                pass

        # =================================================================
        # STAGE 3: Select Pages (intelligent prioritization)
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("select", "Prioritizing pages for OCR...", 0.2)

        if page_index_result:
            # Sprint 20G: Pass deep_cap and force_full_read from run_mode
            _sel_budget = _deep_cap if _deep_cap is not None else 80
            selected_result = select_pages(
                page_index_result,
                budget_pages=_sel_budget,
                force_full_read=_force_full,
                deep_cap=_deep_cap,
            )
            update_progress("select", selected_result.summary_line(), 1.0)
        stage_times["select"] = time.perf_counter() - stage_start

        # =================================================================
        # STAGE 4: Extract Text (OCR on selected pages + specialized extraction)
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("extract", "Checking for OCR requirements...", 0.1)

        all_page_texts = []
        ocr_metadata_all = {
            'ocr_used': False,
            'ocr_pages': [],
            'scales_from_ocr': [],
            'disciplines_from_ocr': [],
            'page_profiles': [],
            'preflight': None,
        }

        num_files = max(len(all_raw_texts), 1)

        for i, (pdf_path, raw_texts) in enumerate(all_raw_texts):
            # Run preflight to determine OCR strategy
            try:
                from .ocr_fallback import pdf_preflight
                preflight = pdf_preflight(pdf_path, raw_texts)
            except ImportError:
                preflight = None

            if preflight:
                ocr_metadata_all['preflight'] = preflight
                update_progress(
                    "extract",
                    f"{pdf_path.name}: {preflight['explain']}",
                    0.12 + (0.03 * i / num_files),
                )

            # Per-page progress sub-callback
            ocr_start_time = time.perf_counter()

            def page_progress(page_idx, total, msg, _i=i):
                elapsed = time.perf_counter() - ocr_start_time
                done = page_idx + 1
                if done > 0 and elapsed > 0:
                    avg_sec = elapsed / done
                    remaining = avg_sec * max(total - done, 0)
                    msg = f"{msg} | {avg_sec:.1f}s/page · est. {remaining:.0f}s remaining"
                file_pct = 0.20 + (0.60 * done / max(total, 1))
                overall_pct = (_i + file_pct) / num_files
                update_progress("extract", msg, min(overall_pct, 0.75))

            # Use selected_pages if available (only for the primary PDF)
            sel_pages = None
            if selected_result and i == 0:
                sel_pages = selected_result.selected

            # Run OCR if needed (with preflight + per-page progress + selected pages)
            processed_texts, ocr_meta = run_ocr_extraction(
                pdf_path, raw_texts,
                preflight=preflight,
                progress_callback=page_progress,
                selected_pages=sel_pages,
            )
            all_page_texts.extend(processed_texts)

            # Update file info with OCR status
            for fi in file_info:
                if fi["name"] == pdf_path.name:
                    fi["ocr_used"] = ocr_meta.get('ocr_used', False)
                    break

            # Aggregate OCR metadata
            if ocr_meta.get('ocr_used'):
                ocr_metadata_all['ocr_used'] = True
                ocr_metadata_all['ocr_pages'].extend(ocr_meta.get('ocr_pages', []))
                ocr_metadata_all['scales_from_ocr'].extend(ocr_meta.get('scales_from_ocr', []))
                ocr_metadata_all['disciplines_from_ocr'].extend(ocr_meta.get('disciplines_from_ocr', []))
            ocr_metadata_all['page_profiles'].extend(ocr_meta.get('page_profiles', []))

        # Run specialized extractors on OCR'd text
        update_progress("extract", "Running specialized extractors...", 0.78)

        if page_index_result and selected_result:
            # Build text map for selected pages
            ocr_text_by_page = {}
            for idx in selected_result.selected:
                if idx < len(all_page_texts) and all_page_texts[idx].strip():
                    ocr_text_by_page[idx] = all_page_texts[idx]

            extraction_result = run_extractors(
                ocr_text_by_page, page_index_result,
                pdf_path=str(primary_pdf_path) if primary_pdf_path else None,
            )

            # ── Normalize extracted data (Sprint 6 + Sprint 19) ────
            from .normalize import (
                normalize_boq_items, normalize_schedule_rows,
                normalize_requirements, normalize_commercial_terms,
            )
            extraction_result.boq_items = normalize_boq_items(extraction_result.boq_items)
            extraction_result.schedules = normalize_schedule_rows(extraction_result.schedules)
            extraction_result.requirements = normalize_requirements(extraction_result.requirements)
            extraction_result.commercial_terms = normalize_commercial_terms(
                extraction_result.commercial_terms
            )

            update_progress(
                "extract",
                f"Extracted {len(extraction_result.requirements)} requirements, "
                f"{len(extraction_result.schedules)} schedule rows, "
                f"{len(extraction_result.boq_items)} BOQ items, "
                f"{len(extraction_result.callouts)} callouts",
                0.92,
            )
        else:
            extraction_result = ExtractionResult()

        # ── Sprint 21C: Excel BOQ override ─────────────────────────
        _boq_source = "pdf"
        if boq_excel_paths:
            try:
                from .excel_boq import parse_boq_excels
                from .normalize import normalize_boq_items as _norm_boq
                excel_items, excel_stats = parse_boq_excels(boq_excel_paths)
                if excel_items:
                    excel_items = _norm_boq(excel_items)
                    extraction_result.boq_items = excel_items
                    _boq_source = "excel"
                    update_progress(
                        "extract",
                        f"Excel BOQ: {len(excel_items)} items from "
                        f"{excel_stats.get('sheets_parsed', 0)} sheet(s)",
                        0.93,
                    )
            except Exception as _excel_err:
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"Excel BOQ parsing failed: {_excel_err}"
                )

        # ── Addendum detection + delta (Sprint 6) ──────────────────
        from .addendum_adapter import extract_addenda
        from .delta_detector import (
            detect_boq_deltas, detect_schedule_deltas, detect_requirement_deltas,
        )

        addenda_list = []
        all_conflicts = []
        if page_index_result and extraction_result and extraction_result.pages_processed > 0:
            addenda_list = extract_addenda(ocr_text_by_page, page_index_result)
            if addenda_list:
                addendum_page_idxs = {
                    p.page_idx for p in page_index_result.pages
                    if p.doc_type == "addendum"
                }
                base_boq = [i for i in extraction_result.boq_items
                            if i.get("source_page") not in addendum_page_idxs]
                add_boq = [i for i in extraction_result.boq_items
                           if i.get("source_page") in addendum_page_idxs]
                base_sched = [r for r in extraction_result.schedules
                              if r.get("source_page") not in addendum_page_idxs]
                add_sched = [r for r in extraction_result.schedules
                             if r.get("source_page") in addendum_page_idxs]
                base_reqs = [r for r in extraction_result.requirements
                             if r.get("doc_type") != "addendum"]
                add_reqs = [r for r in extraction_result.requirements
                            if r.get("doc_type") == "addendum"]
                all_conflicts.extend(detect_boq_deltas(base_boq, add_boq))
                all_conflicts.extend(detect_schedule_deltas(base_sched, add_sched))
                # Sprint 7: pass OCR coverage pct for confidence scoring
                _ocr_cov_pct = None
                if run_coverage and run_coverage.pages_total > 0:
                    _ocr_cov_pct = (run_coverage.pages_deep_processed / run_coverage.pages_total) * 100
                all_conflicts.extend(detect_requirement_deltas(
                    base_reqs, add_reqs, ocr_coverage_pct=_ocr_cov_pct,
                ))

        # ── Sprint 9: Tag conflicts with supersede resolution ──────────
        if all_conflicts and ocr_text_by_page:
            from .delta_detector import tag_conflicts_with_resolution
            all_conflicts = tag_conflicts_with_resolution(all_conflicts, ocr_text_by_page)

        # ── Scope Reconciliation (Sprint 7) ───────────────────────────
        from .reconciler import reconcile_scope
        reconciliation_findings = []
        if extraction_result and extraction_result.pages_processed > 0:
            reconciliation_findings = reconcile_scope(
                extraction_result.requirements,
                extraction_result.schedules,
                extraction_result.boq_items,
            )

        # ── Sprint 10: Toxic page retry ───────────────────────────────
        toxic_results = []
        try:
            from .toxic_pages import identify_failed_pages, process_toxic_pages, build_toxic_pages_summary
            _failed_pages = identify_failed_pages(ocr_metadata_all)
            if _failed_pages and all_raw_texts:
                toxic_results = process_toxic_pages(primary_pdf_path, _failed_pages)
                # Merge recovered text back into all_page_texts
                for tr in toxic_results:
                    if not tr.get("toxic") and tr.get("text"):
                        pidx = tr["page_idx"]
                        if pidx < len(all_page_texts):
                            all_page_texts[pidx] = tr["text"]
        except Exception:
            pass  # Toxic retry is non-critical

        # ── Build OCR text cache for NOT_FOUND proof ────────────────
        ocr_text_cache = {}
        for idx, text in enumerate(all_page_texts):
            if text and len(text.strip()) > 10:
                ocr_text_cache[idx] = text[:10000]  # cap at 10KB per page

        # Sprint 10: Cache OCR text
        try:
            save_cached_stage(cache_dir, "ocr_text", ocr_text_cache)
        except Exception:
            pass

        # ── Surya OCR for bounding boxes ─────────────────────────────
        page_bboxes = {}  # Dict[int, List[List[float]]]
        ocr_bbox_meta = {"engine": "none", "avg_confidence": 0.0, "pages_with_bbox": 0, "pages_total": total_pages}
        try:
            from .surya_ocr import HAS_SURYA, extract_with_surya, build_ocr_bbox_meta
            if HAS_SURYA and all_raw_texts:
                _surya_pages = list(range(min(total_pages, 50)))  # cap at 50 pages for Surya
                surya_results = extract_with_surya(primary_pdf_path, _surya_pages)
                for pg_idx, sr in surya_results.items():
                    if sr.bboxes:
                        page_bboxes[pg_idx] = sr.bboxes
                ocr_bbox_meta = build_ocr_bbox_meta(surya_results, total_pages)
        except Exception:
            pass  # Surya not available or failed — continue without

        # Clean and normalize text
        update_progress("extract", "Cleaning extracted text...", 0.95)
        cleaned_texts = []
        for text in all_page_texts:
            cleaned = text.strip()
            if len(cleaned) > 10:  # Skip near-empty pages
                cleaned_texts.append(cleaned)

        ocr_msg = " (OCR used)" if ocr_metadata_all['ocr_used'] else ""
        update_progress("extract", f"Extracted {len(cleaned_texts)} pages with content{ocr_msg}", 1.0)
        stage_times["extract"] = time.perf_counter() - stage_start

        if not all_page_texts:
            raise ValueError("No text could be extracted from the uploaded PDFs. Please ensure they are readable.")

        # =================================================================
        # BUILD RunCoverage (after extract, before reason/rfi)
        # =================================================================
        from src.models.analysis_models import RunCoverage, CoverageStatus, SelectionMode

        run_coverage = None
        if page_index_result and selected_result:
            # Determine which doc types are fully / partially / not covered
            detected_types = dict(page_index_result.counts_by_type)   # {doc_type: total}
            selected_types_map = dict(selected_result.coverage_summary)  # {doc_type: selected}

            fully_covered = []
            partially_covered = []
            not_covered = []
            for dtype, total in detected_types.items():
                sel = selected_types_map.get(dtype, 0)
                if sel >= total:
                    fully_covered.append(dtype)
                elif sel > 0:
                    partially_covered.append(dtype)
                else:
                    not_covered.append(dtype)

            # Build skipped pages list
            selected_set = set(selected_result.selected)
            pages_skipped = []
            for p in page_index_result.pages:
                if p.page_idx not in selected_set:
                    pages_skipped.append({
                        "page_idx": p.page_idx,
                        "doc_type": p.doc_type,
                        "discipline": p.discipline,
                        "reason": "budget_exceeded",
                    })

            sel_mode = (
                SelectionMode.FULL_READ
                if selected_result.selection_mode == "full_read"
                else SelectionMode.FAST_BUDGET
            )

            run_coverage = RunCoverage(
                pages_total=page_index_result.total_pages,
                pages_indexed=page_index_result.total_pages,
                pages_deep_processed=len(selected_result.selected),
                pages_skipped=pages_skipped,
                doc_types_detected=detected_types,
                disciplines_detected=dict(page_index_result.counts_by_discipline),
                doc_types_fully_covered=fully_covered,
                doc_types_partially_covered=partially_covered,
                doc_types_not_covered=not_covered,
                selection_mode=sel_mode,
                ocr_budget_pages=selected_result.budget_total,
            )

        # =================================================================
        # STAGE 3: Build Plan Graph
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("graph", "Building plan set structure...", 0.2)

        from .plan_graph import build_plan_graph, save_plan_graph

        # Use all page texts for graph building (even if some are sparse)
        plan_graph = build_plan_graph(project_id, all_page_texts)
        update_progress("graph", f"Identified {plan_graph.total_pages} sheets", 0.7)

        # Save plan graph
        graph_path = save_plan_graph(plan_graph, output_dir)
        result.files_generated.append(str(graph_path))
        update_progress("graph", "Plan graph saved", 1.0)
        stage_times["graph"] = time.perf_counter() - stage_start

        # =================================================================
        # STAGE 4: Dependency Reasoning
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("reason", "Analyzing scope completeness...", 0.2)

        from .dependency_reasoner import reason_dependencies

        blockers, rfis, coverage, boq_skeleton = reason_dependencies(plan_graph, run_coverage=run_coverage)
        update_progress("reason", f"Found {len(blockers)} blockers, {len(rfis)} RFIs", 0.7)

        # Calculate readiness score (pass trade_coverage for better decision making)
        from .llm_enrichment import calculate_readiness_score
        score_result = calculate_readiness_score(plan_graph, blockers, trade_coverage=coverage)
        update_progress("reason", f"Readiness: {score_result.status} ({score_result.total_score}/100)", 1.0)
        stage_times["reason"] = time.perf_counter() - stage_start

        # =================================================================
        # STAGE 7: Generate Evidence-Backed RFIs
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("rfi", "Running discipline checklist...", 0.2)

        # Generate new checklist-driven RFIs
        if extraction_result and page_index_result and selected_result:
            new_rfis = generate_rfis(
                extraction_result, page_index_result, selected_result, plan_graph,
                run_coverage=run_coverage,
            )
            update_progress("rfi", f"Checklist produced {len(new_rfis)} RFIs, merging with dependency RFIs...", 0.6)

            # Merge with blocker-derived RFIs (dedup by issue_type + trade)
            existing_keys = {(r.issue_type, r.trade.value) for r in new_rfis}
            for old_rfi in rfis:
                key = (old_rfi.issue_type, old_rfi.trade.value)
                if key not in existing_keys:
                    new_rfis.append(old_rfi)
                    existing_keys.add(key)

            rfis = new_rfis

        rfi_list = [rfi.to_dict() if hasattr(rfi, 'to_dict') else rfi for rfi in rfis]
        update_progress("rfi", f"Generated {len(rfi_list)} RFIs", 1.0)
        stage_times["rfi"] = time.perf_counter() - stage_start

        # =================================================================
        # STAGE 6: Export Results
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("export", "Saving analysis results...", 0.2)

        # Build deep analysis result
        from src.models.analysis_models import DeepAnalysisResult  # Keep absolute for models

        deep_result = DeepAnalysisResult(
            project_id=project_id,
            plan_graph=plan_graph,
            blockers=blockers,
            rfis=rfis,
            trade_coverage=coverage,
            boq_skeleton=boq_skeleton,
            readiness_score=score_result,
        )

        # Save all outputs
        from .evaluation import save_all_outputs
        output_paths = save_all_outputs(deep_result, output_dir)

        for name, path in output_paths.items():
            result.files_generated.append(str(path))

        update_progress("export", f"Saved {len(output_paths)} output files", 0.8)

        # Also save a simple analysis.json for compatibility
        analysis_summary = {
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "readiness_score": score_result.total_score,
            "decision": score_result.status,
            "blockers_count": len(blockers),
            "rfis_count": len(rfis),
            "total_pages": plan_graph.total_pages,
            "files_analyzed": [f["name"] for f in file_info],
        }

        analysis_path = output_dir / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        result.files_generated.append(str(analysis_path))

        # Save eval.json
        eval_data = {
            "project_id": project_id,
            "score": score_result.total_score,
            "status": score_result.status,
            "sub_scores": {
                "completeness": getattr(score_result, 'completeness_score', 50),
                "coverage": getattr(score_result, 'coverage_score', 50),
                "measurement": getattr(score_result, 'measurement_score', 50),
            },
        }
        eval_path = output_dir / "eval.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        result.files_generated.append(str(eval_path))

        update_progress("export", "All outputs saved successfully", 1.0)
        stage_times["export"] = time.perf_counter() - stage_start

        # Build the full analysis payload for immediate UI display
        # Convert blockers to serializable format
        blockers_payload = []
        for b in blockers:
            blocker_dict = b.to_dict() if hasattr(b, 'to_dict') else (
                b if isinstance(b, dict) else {"title": str(b)}
            )
            blockers_payload.append(blocker_dict)

        # Convert RFIs to serializable format
        rfis_payload = []
        for r in rfis:
            rfi_dict = r.to_dict() if hasattr(r, 'to_dict') else (
                r if isinstance(r, dict) else {"question": str(r)}
            )
            rfis_payload.append(rfi_dict)

        # Assign bbox_ids to all blockers and RFIs
        from src.models.analysis_models import assign_all_bbox_ids
        assign_all_bbox_ids(blockers_payload, rfis_payload)

        # Convert trade coverage to serializable format
        coverage_payload = []
        for c in coverage:
            cov_dict = c.to_dict() if hasattr(c, 'to_dict') else (
                c if isinstance(c, dict) else {"trade": str(c)}
            )
            coverage_payload.append(cov_dict)

        # Build drawing overview
        drawing_overview = {
            "files": [f["name"] for f in file_info],
            "pages_total": plan_graph.total_pages,
            "disciplines_detected": getattr(plan_graph, 'disciplines_found', []),
            "sheet_types_count": len(getattr(plan_graph, 'sheet_types_found', {})),
            "scale_found_pages": getattr(plan_graph, 'pages_with_scale', 0),
            "schedules_detected_count": sum([
                1 if getattr(plan_graph, 'has_door_schedule', False) else 0,
                1 if getattr(plan_graph, 'has_window_schedule', False) else 0,
                1 if getattr(plan_graph, 'has_finish_schedule', False) else 0,
            ]),
            "door_tags_found": len(getattr(plan_graph, 'all_door_tags', [])),
            "window_tags_found": len(getattr(plan_graph, 'all_window_tags', [])),
            "room_names_found": len(getattr(plan_graph, 'all_room_names', [])),
            # OCR info
            "ocr_used": ocr_metadata_all.get('ocr_used', False),
            "ocr_pages_count": len(ocr_metadata_all.get('ocr_pages', [])),
            "scales_from_ocr": ocr_metadata_all.get('scales_from_ocr', []),
        }

        # =============================================================
        # DRAWING PRESENCE CHECK
        # If no drawing-like signals found, this PDF is probably not
        # a construction drawing set (e.g., homework, report, etc.).
        # =============================================================
        is_drawing_set = True
        no_drawing_reason = ""

        disciplines = drawing_overview.get("disciplines_detected", [])
        schedules = drawing_overview.get("schedules_detected_count", 0)
        doors = drawing_overview.get("door_tags_found", 0)
        windows = drawing_overview.get("window_tags_found", 0)
        rooms = drawing_overview.get("room_names_found", 0)
        scale_pages = drawing_overview.get("scale_found_pages", 0)
        total_pg = drawing_overview.get("pages_total", 0)

        drawing_signals = len(disciplines) + schedules + doors + windows + rooms + scale_pages
        if drawing_signals == 0:
            is_drawing_set = False
            no_drawing_reason = (
                f"No construction drawing indicators found in {total_pg} page(s): "
                "no disciplines, schedules, door/window tags, room names, or scale notations detected."
            )

        drawing_overview["is_drawing_set"] = is_drawing_set

        if not is_drawing_set:
            # Override decision and scores for non-drawing PDFs
            final_decision = "NO_DRAWINGS"
            final_score = 0
            final_sub_scores = {
                "completeness": 0,
                "coverage": 0,
                "measurement": 0,
                "blocker": 0,
            }
            # Replace blockers with a single clear blocker
            blockers_payload = [
                {
                    "id": "BLK-NO-DRAWINGS",
                    "title": "No construction drawings detected in uploaded PDF",
                    "trade": "general",
                    "severity": "critical",
                    "affected_trades": [],
                    "description": no_drawing_reason,
                    "missing_dependency": ["construction_drawings"],
                    "impact_cost": "critical",
                    "impact_schedule": "critical",
                    "bid_impact": "cannot_bid",
                    "evidence": {
                        "pages": list(range(total_pg)),
                        "confidence": 0.95,
                        "confidence_reason": "No drawing-like content detected by any extractor",
                    },
                    "fix_actions": [
                        "Upload construction drawing PDFs (floor plans, structural, MEP, etc.)",
                        "Ensure drawings are not password-protected or image-only without OCR support",
                    ],
                    "score_delta_estimate": 0,
                    "issue_type": "no_drawings",
                    "created_at": datetime.now().isoformat(),
                }
            ]
            rfis_payload = []
            coverage_payload = []
        else:
            final_decision = score_result.status
            final_score = score_result.total_score
            final_sub_scores = {
                "completeness": getattr(score_result, 'completeness_score', 50),
                "coverage": getattr(score_result, 'coverage_score', 50),
                "measurement": getattr(score_result, 'measurement_score', 50),
                "blocker": getattr(score_result, 'blocker_score', 50),
            }

        # Guardrails
        guardrail_warnings = _check_guardrails(
            page_index_result, extraction_result, rfis
        )

        # Sprint 10: Compute QA score + RFI clustering
        _toxic_summary = None
        try:
            from .toxic_pages import build_toxic_pages_summary
            _toxic_summary = build_toxic_pages_summary(toxic_results) if toxic_results else None
        except Exception:
            pass

        _qa_result = None
        try:
            from .qa_score import compute_qa_score
            _qa_partial = {
                "run_coverage": run_coverage.to_dict() if run_coverage else None,
                "conflicts": all_conflicts,
                "addendum_index": [a.to_dict() for a in addenda_list],
                "extraction_summary": extraction_result.to_dict() if extraction_result else {},
                "toxic_pages": _toxic_summary,
            }
            _qa_result = compute_qa_score(_qa_partial)
        except Exception:
            pass

        _rfi_clusters = []
        try:
            from .rfi_clustering import cluster_rfis
            _rfi_clusters = cluster_rfis(
                rfis_payload,
                multi_doc_index=multi_doc_index.to_dict() if multi_doc_index else None,
            )
        except Exception:
            pass

        # Sprint 11: Build unified quantities
        _quantities = []
        try:
            from .quantities import build_all_quantities
            _quantities = build_all_quantities(
                schedules=extraction_result.schedules if extraction_result else [],
                boq_items=extraction_result.boq_items if extraction_result else [],
                callouts=extraction_result.callouts if extraction_result else [],
                drawing_overview=drawing_overview,
            )
        except Exception:
            pass

        # Sprint 11: Compute pricing guidance
        _pricing_guidance = None
        try:
            from .pricing_guidance import compute_pricing_guidance
            _run_cov_dict = run_coverage.to_dict() if run_coverage else None
            _pricing_guidance = compute_pricing_guidance(
                qa_score=_qa_result,
                addendum_index=[a.to_dict() for a in addenda_list],
                conflicts=all_conflicts,
                owner_profile=None,
                run_coverage=_run_cov_dict,
            )
        except Exception:
            pass

        # Sprint 12: Quantity reconciliation
        _qty_reconciliation = []
        try:
            from .quantity_reconciliation import reconcile_quantities
            _qty_reconciliation = reconcile_quantities(
                quantities=_quantities,
                schedules=extraction_result.schedules if extraction_result else [],
                boq_items=extraction_result.boq_items if extraction_result else [],
                callouts=extraction_result.callouts if extraction_result else [],
            )
        except Exception:
            pass

        # Sprint 12: Finish takeoff
        _finish_takeoff = None
        try:
            from .finish_takeoff import build_finish_takeoff
            _finish_takeoff = build_finish_takeoff(
                schedules=extraction_result.schedules if extraction_result else [],
            )
        except Exception:
            pass

        # Sprint 13: Review queue (risk_results computed in UI, not pipeline)
        _review_queue = []
        try:
            from .review_queue import build_review_queue
            _rq_pages_skipped = []
            if run_coverage:
                _rq_pages_skipped = run_coverage.to_dict().get("pages_skipped", [])
            _review_queue = build_review_queue(
                quantity_reconciliation=_qty_reconciliation,
                conflicts=all_conflicts,
                pages_skipped=_rq_pages_skipped,
                toxic_summary=_toxic_summary,
                risk_results=[],
            )
        except Exception:
            pass

        # Sprint 13: Add default status fields
        for _rfi in rfis_payload:
            _rfi.setdefault("status", "draft")
        for _qty in _quantities:
            _qty.setdefault("status", "draft")
        for _c in all_conflicts:
            _c.setdefault("review_status", "unreviewed")

        # Sprint 10: Build cache stats
        _cache_bytes = _measure_cache_bytes(cache_dir)
        _cache_stats = build_cache_stats(_cache_hits, _cache_misses, _cache_time_saved, _cache_bytes)

        # Sprint 18: Compute processing_stats from existing pipeline data
        _processing_stats = _build_processing_stats(
            page_index_result=page_index_result,
            selected_result=selected_result,
            ocr_metadata=ocr_metadata_all,
            run_coverage=run_coverage,
            toxic_summary=_toxic_summary,
            file_info=file_info,
            extraction_result=extraction_result,
        )

        # Sprint 20G: Attach run_mode to processing_stats
        _processing_stats["run_mode"] = _run_mode_enum.value

        # Sprint 19: Compute BOQ stats + requirements by trade
        _boq_stats = _compute_boq_stats(
            extraction_result.boq_items if extraction_result else []
        )
        _req_by_trade = _build_requirements_by_trade(
            extraction_result.requirements if extraction_result else []
        )

        # Sprint 20A: Optional structural takeoff step
        _structural_takeoff = None
        try:
            _has_structural = False
            if page_index_result:
                _disc_map = page_index_result.counts_by_discipline or {}
                _has_structural = _disc_map.get("structural", 0) > 0
            if _has_structural and primary_pdf_path:
                from src.structural.pipeline_structural import (
                    StructuralPipeline,
                    StructuralPipelineConfig,
                )
                _st_config = StructuralPipelineConfig(
                    mode="auto",
                    floors=1,
                    output_dir=output_dir / "structural" if output_dir else Path("./out/structural"),
                    generate_overlays=True,
                )
                _st_pipeline = StructuralPipeline(_st_config)
                _st_result = _st_pipeline.process(Path(primary_pdf_path))
                if _st_result.success and _st_result.quantity_result:
                    _st_summary = _st_result.quantity_result.summary.to_dict()
                    _st_qc = _st_result.qc_report.to_dict() if _st_result.qc_report else {}
                    _st_quantities = [
                        e.to_dict() for e in _st_result.quantity_result.elements
                    ]
                    _st_assumptions = _st_result.quantity_result.assumptions_used or []
                    _st_export_paths = {
                        k: str(v) for k, v in (_st_result.output_paths or {}).items()
                    }
                    # Multi-doc awareness: tag source file
                    _st_source_file = Path(primary_pdf_path).name if primary_pdf_path else ""
                    _structural_takeoff = {
                        "mode": _st_result.mode,
                        "summary": {
                            "concrete_m3": _st_summary.get("concrete_m3", {}).get("total", 0),
                            "steel_kg": _st_summary.get("steel_kg", {}).get("total", 0),
                            "steel_tons": _st_summary.get("steel_tonnes", 0),
                            "element_counts": _st_summary.get("counts", {}),
                            "detail": _st_summary,
                        },
                        "quantities": _st_quantities,
                        "qc": {
                            "confidence": _st_qc.get("confidence", {}).get("overall", 0),
                            "issues": _st_qc.get("issues", {}),
                            "assumptions": _st_qc.get("assumptions", {}),
                        },
                        "exports": _st_export_paths,
                        "source_file": _st_source_file,
                        "warnings": _st_result.warnings or [],
                    }
                elif _st_result.errors:
                    _structural_takeoff = {
                        "mode": "error",
                        "summary": {},
                        "quantities": [],
                        "qc": {},
                        "exports": {},
                        "warnings": _st_result.errors,
                    }
        except Exception as _st_exc:
            # Structural step failed — add warning but don't break pipeline
            _structural_takeoff = {
                "mode": "error",
                "summary": {},
                "quantities": [],
                "qc": {},
                "exports": {},
                "warnings": [f"Structural takeoff failed: {type(_st_exc).__name__}: {_st_exc}"],
            }

        # Assemble full payload
        result.payload = {
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "drawing_overview": drawing_overview,
            "readiness_score": final_score,
            "decision": final_decision,
            "sub_scores": final_sub_scores,
            "blockers": blockers_payload,
            "rfis": rfis_payload,
            "trade_coverage": coverage_payload,
            "extraction_summary": extraction_result.to_dict() if extraction_result else {},
            "addendum_index": [a.to_dict() for a in addenda_list],
            "conflicts": all_conflicts,
            "reconciliation_findings": reconciliation_findings,
            "run_coverage": run_coverage.to_dict() if run_coverage else None,
            "guardrail_warnings": guardrail_warnings,
            "timings": {
                "load_s": stage_times.get("load", 0),
                "index_s": stage_times.get("index", 0),
                "select_s": stage_times.get("select", 0),
                "extract_s": stage_times.get("extract", 0),
                "graph_s": stage_times.get("graph", 0),
                "reason_s": stage_times.get("reason", 0),
                "rfi_s": stage_times.get("rfi", 0),
                "export_s": stage_times.get("export", 0),
                "total_s": (datetime.now() - start_time).total_seconds(),
            },
            "diagnostics": {
                "stage2_profile": ocr_metadata_all.get("page_profiles", []),
                "preflight": ocr_metadata_all.get("preflight"),
                "page_index": page_index_result.to_dict() if page_index_result else None,
                "selected_pages": selected_result.to_dict() if selected_result else None,
            },
            # PDF path for evidence viewer (page image rendering)
            "primary_pdf_path": str(primary_pdf_path) if all_raw_texts else None,
            "file_info": file_info,
            # Sprint 9: Multi-doc index
            "multi_doc_index": multi_doc_index.to_dict() if multi_doc_index else None,
            # Sprint 4a: OCR text cache + bbox metadata
            "ocr_text_cache": ocr_text_cache,
            "ocr_bbox_meta": ocr_bbox_meta,
            # Sprint 10: Cache stats, toxic pages, QA score, RFI clusters
            "cache_stats": _cache_stats,
            "toxic_pages": _toxic_summary,
            "qa_score": _qa_result,
            "rfi_clusters": _rfi_clusters,
            # Sprint 11: Quantities, pricing guidance
            "quantities": _quantities,
            "pricing_guidance": _pricing_guidance,
            # Sprint 12: Quantity reconciliation, finish takeoff
            "quantity_reconciliation": _qty_reconciliation,
            "finish_takeoff": _finish_takeoff,
            # Sprint 13: Review queue
            "review_queue": _review_queue,
            # Sprint 18: Processing stats (reliable page counters)
            "processing_stats": _processing_stats,
            # Sprint 19: Commercial terms, BOQ stats, requirements by trade
            "commercial_terms": extraction_result.commercial_terms if extraction_result else [],
            "boq_stats": _boq_stats,
            "requirements_by_trade": _req_by_trade,
            # Sprint 20A: Structural takeoff (optional)
            "structural_takeoff": _structural_takeoff,
            # Sprint 20C: Estimating playbook (attached from session if available)
            "estimating_playbook": None,
            # Sprint 21C: BOQ source tracking (pdf or excel)
            "boq_source": _boq_source,
            # Sprint 20F: Extraction diagnostics (hybrid table extraction)
            "extraction_diagnostics": extraction_result.extraction_diagnostics if extraction_result else {},
        }

        # Sprint 13: Add approval snapshots to payload for run compare
        result.payload["rfi_status_snapshot"] = {
            r.get("id", ""): r.get("status", "draft")
            for r in rfis_payload if r.get("id")
        }
        result.payload["quantities_accepted_count"] = sum(
            1 for q in _quantities if q.get("status") == "accepted"
        )
        result.payload["conflicts_reviewed_count"] = sum(
            1 for c in all_conflicts if c.get("review_status") == "reviewed"
        )

        # Sprint 11: Run Compare — diff against previous cached payload
        try:
            _prev_payload = load_cached_stage(cache_dir, "last_payload")
            _run_compare = _compute_run_compare(_prev_payload, result.payload)
            if _run_compare:
                result.payload["_run_compare"] = _run_compare
        except Exception:
            pass

        # Also save the full payload as analysis.json (replace the minimal one)
        full_analysis_path = output_dir / "analysis.json"
        with open(full_analysis_path, 'w') as f:
            json.dump(result.payload, f, indent=2, default=str)

        # Sprint 11: Save current payload snapshot for next run compare
        try:
            _compare_snapshot = {
                "qa_score": result.payload.get("qa_score"),
                "rfis": [{"question": r.get("question", "")} for r in rfis_payload[:50]],
                "conflicts": [{"description": c.get("description", "")} for c in all_conflicts[:50]],
                "quantities_count": len(_quantities),
                "blockers_count": len(blockers_payload),
                "reconciliation_mismatches": sum(1 for r in _qty_reconciliation if r.get("mismatch")),
                "finish_takeoff_has_areas": (_finish_takeoff or {}).get("has_areas", False),
                # Sprint 13: Approval snapshot
                "rfi_status_snapshot": {
                    r.get("id", ""): r.get("status", "draft")
                    for r in rfis_payload if r.get("id")
                },
                "quantities_accepted_count": sum(
                    1 for q in _quantities if q.get("status") == "accepted"
                ),
                "conflicts_reviewed_count": sum(
                    1 for c in all_conflicts if c.get("review_status") == "reviewed"
                ),
            }
            save_cached_stage(cache_dir, "last_payload", _compare_snapshot)
        except Exception:
            pass

        # Sprint 14: Auto-save run record to project workspace (if project exists)
        try:
            from .projects import save_run as _save_project_run, load_project as _load_proj
            if _load_proj(project_id):
                _run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                _save_project_run(
                    project_id,
                    _run_id,
                    str(full_analysis_path),
                    export_paths=[str(p) for p in result.files_generated],
                    run_metadata={
                        "readiness_score": final_score,
                        "decision": final_decision,
                        "rfis_count": len(rfis_payload),
                        "blockers_count": len(blockers_payload),
                    },
                )
        except Exception:
            pass

        # Debug: Print paths for verification
        print(f"\n{'='*60}")
        print(f"[DEBUG] Analysis complete for project: {project_id}")
        print(f"[DEBUG] Output directory: {output_dir}")
        print(f"[DEBUG] Files generated:")
        for f in result.files_generated:
            fpath = Path(f)
            exists = fpath.exists()
            size = fpath.stat().st_size if exists else 0
            print(f"  - {fpath.name}: {'OK' if exists else 'MISSING'} ({size} bytes)")
        print(f"\n[DEBUG] Stage timings:")
        for stage, duration in stage_times.items():
            print(f"  - {stage}: {duration:.2f}s")
        print(f"\n[DEBUG] Payload summary:")
        print(f"  - Blockers: {len(blockers_payload)}")
        print(f"  - RFIs: {len(rfis_payload)}")
        print(f"  - Trade coverage entries: {len(coverage_payload)}")
        print(f"  - Readiness score: {score_result.total_score}/100")

        # Success
        result.success = True
        result.duration_sec = (datetime.now() - start_time).total_seconds()
        print(f"\n[DEBUG] Total duration: {result.duration_sec:.2f} seconds")
        print(f"{'='*60}\n")

    except Exception as e:
        result.error_message = str(e)
        result.stack_trace = traceback.format_exc()
        result.duration_sec = (datetime.now() - start_time).total_seconds()

        # Mark current stage as failed
        for stage in result.stages:
            if stage.status == "running":
                stage.status = "failed"
                stage.error = str(e)
                break

    return result


def save_uploaded_files(
    uploaded_files: List[Any],  # Streamlit UploadedFile objects
    project_id: str,
    uploads_dir: Path,
) -> List[Path]:
    """
    Save uploaded files to disk.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        project_id: Project identifier
        uploads_dir: Base uploads directory

    Returns:
        List of saved file paths
    """
    project_uploads = uploads_dir / project_id
    project_uploads.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = project_uploads / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def generate_project_id() -> str:
    """Generate a unique project ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"project_{timestamp}_{short_uuid}"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample PDF
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analysis_runner.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    project_id = generate_project_id()
    output_dir = PROJECT_ROOT / "out" / project_id

    def progress_cb(stage_id: str, message: str, progress: float):
        print(f"[{stage_id}] {message} ({progress*100:.0f}%)")

    result = run_analysis_pipeline(
        input_files=[pdf_path],
        project_id=project_id,
        output_dir=output_dir,
        progress_callback=progress_cb,
    )

    if result.success:
        print(f"\n✅ Analysis complete!")
        print(f"   Project: {result.project_id}")
        print(f"   Output: {result.output_dir}")
        print(f"   Files: {result.files_generated}")
    else:
        print(f"\n❌ Analysis failed: {result.error_message}")
        print(result.stack_trace)
