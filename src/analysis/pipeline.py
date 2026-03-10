"""
Analysis Pipeline

Provides a streamlit-compatible interface to run the analysis pipeline
with progress callbacks for real-time UI updates.
"""

import json
import logging
import sys
import traceback
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

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
            - List of text strings, one per page (empty string for image-only pages)
            - Page count

    Raises:
        ImportError: if PyMuPDF (fitz) is not installed
        Exception: propagated so the caller can surface the real error in the UI
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. Add 'PyMuPDF>=1.23.0' to requirements.txt."
        )

    try:
        doc = fitz.open(str(pdf_path))
        page_texts = []
        for page in doc:
            text = page.get_text()
            page_texts.append(text)
        page_count = len(page_texts)
        doc.close()
        return page_texts, page_count
    except Exception as e:
        logging.getLogger(__name__).error("Failed to load %s: %s", pdf_path.name, e)
        raise


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

def _detect_epc_mode(boq_items: list) -> bool:
    """Detect EPC/Turnkey contracts from BOQ structure.

    Signal: all priceable BOQ items are lump-sum (unit=LS/LUMP) AND item count is ≤5.
    This is characteristic of EPC tenders where the entire scope is priced as a single
    or very small number of lump-sum packages; the real scope lives in the DPR/tech-spec.

    Args:
        boq_items: Extracted BOQ items list.

    Returns:
        True if this looks like an EPC/turnkey lump-sum BOQ.
    """
    if not boq_items:
        return False
    # Only consider rows that have at least a description (exclude pure section headers)
    data_items = [
        it for it in boq_items
        if (it.get("description") or "").strip()
        and len((it.get("description") or "").strip()) > 5
    ]
    if not data_items:
        return False
    ls_units = {"LS", "LUMP", "L.S", "L.S.", "JOB", "LOT", "LUMP SUM", "LUMPSUM"}
    ls_items = [
        it for it in data_items
        if (it.get("unit") or "").strip().upper() in ls_units
        or "lump sum" in (it.get("description") or "").lower()
        or "complete works" in (it.get("description") or "").lower()
    ]
    return len(ls_items) >= max(1, len(data_items)) and len(data_items) <= 5


def _compute_boq_stats(boq_items: list) -> dict:
    """Compute summary statistics for BOQ items.

    Returns: {total_items, by_trade, flagged_items, flagged_count, epc_mode}
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

    epc_mode = _detect_epc_mode(boq_items)
    return {
        "total_items": len(boq_items),
        "by_trade": by_trade,
        "flagged_items": flagged,
        "flagged_count": len(flagged),
        "epc_mode": epc_mode,
        "epc_mode_note": (
            "EPC/Turnkey contract detected — entire scope priced as lump sum(s). "
            "Detailed scope of work is in the DPR / technical specification documents. "
            "Line items above are extracted from spec/conditions pages only."
        ) if epc_mode else None,
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

    except Exception as e:
        logger.debug("Processing stats computation partial: %s", e)

    return stats


def run_analysis_pipeline(
    input_files: List[Path],
    project_id: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
    run_mode: Optional[str] = None,
    boq_excel_paths: Optional[List[Path]] = None,
    llm_client=None,
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
        llm_client: Optional LLM client (openai.OpenAI or anthropic.Anthropic) for
                    enrichment. If None, enrichment uses template fallbacks.

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
            # Surface the real error from each file so it shows in the UI
            errors = [f"{f['name']}: {f.get('error', 'unknown error')}"
                      for f in file_info if f.get("error")]
            detail = ("  •  " + "\n  •  ".join(errors)) if errors else "no error details available"
            raise ValueError(
                f"No pages could be loaded from the uploaded PDFs.\n{detail}"
            )

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
            except Exception as e:
                logger.debug("Page index cache load failed, will re-index: %s", e)

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
            except Exception as e:
                logger.debug("Page index cache save failed: %s", e)

        # -----------------------------------------------------------------
        # Stage 2b: Package Classification (runs after page_index is ready)
        # -----------------------------------------------------------------
        _pkg_classification = None
        if page_index_result:
            try:
                from .package_classifier import classify_package
                _pkg_classification = classify_package(page_index_result)
                logger.info(
                    "Package classified as %s (confidence=%.2f): %s",
                    _pkg_classification.package_type.value,
                    _pkg_classification.confidence,
                    _pkg_classification.reasoning,
                )
            except Exception as _cls_exc:
                logger.warning("Package classification failed (non-fatal): %s", _cls_exc)

        # =================================================================
        # STAGE 3: Select Pages (intelligent prioritization)
        # =================================================================
        stage_start = time.perf_counter()
        update_progress("select", "Prioritizing pages for OCR...", 0.2)

        if page_index_result:
            # Sprint 20G: Pass deep_cap and force_full_read from run_mode
            _sel_budget = _deep_cap if _deep_cap is not None else 80
            # For TENDER/MIXED packages every text page is priceable content —
            # lift the sub-caps so no spec/conditions/notes page is dropped.
            _force_text = (
                _pkg_classification is not None
                and _pkg_classification.package_type.value in ("tender", "mixed")
            )
            selected_result = select_pages(
                page_index_result,
                budget_pages=_sel_budget,
                force_full_read=_force_full,
                deep_cap=_deep_cap,
                force_text_pages=_force_text,
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

        # ── Sprint 21C: Schedule ↔ BOQ reconciliation ──────────────
        # Links schedule marks (D-01, W-05) to BOQ descriptions;
        # creates stub BOQ items for unmatched marks.
        _schedule_boq_links: dict = {}
        try:
            from .reconciler import link_schedules_to_boq
            _recon = link_schedules_to_boq(
                extraction_result.schedules,
                extraction_result.boq_items,   # mutated in-place
            )
            extraction_result.boq_items.extend(_recon.stub_items)
            _schedule_boq_links = {
                "linked_pairs":             _recon.linked_pairs,
                "unmatched_schedule_marks": _recon.unmatched_schedule_marks,
                "stub_items_added":         len(_recon.stub_items),
            }
        except Exception as _recon_err:
            import logging as _rlog
            _rlog.getLogger(__name__).warning(
                f"Schedule-BOQ reconciliation failed: {_recon_err}"
            )
            _recon = None

        # ── Sprint 21D: Universal item normaliser ───────────────────
        # Build unified line_items list from BOQ + spec_items + stubs.
        _line_items_payload: list = []
        _line_items_summary: dict = {}
        _dedup_stats: dict = {}
        _stub_items_list: list = []
        try:
            from .item_normalizer import build_line_items, build_contractual_items
            _spec_items_list  = getattr(extraction_result, "spec_items", [])
            _stub_items_list  = _recon.stub_items if _recon else []
            _unified_items    = build_line_items(
                boq_items      = extraction_result.boq_items,
                spec_items     = _spec_items_list,
                schedule_stubs = _stub_items_list,
            )
            _line_items_payload = [li.to_dict() for li in _unified_items]

            # ── Rate intelligence: stamp DSR 2023 benchmark on each item ──
            try:
                from .rate_intelligence.dsr_lookup import RateLookup as _RateLookup
                _rate_lookup = _RateLookup()
                _line_items_payload = _rate_lookup.benchmark_items(_line_items_payload)
            except Exception as _ri_err:
                import logging as _rilog
                _rilog.getLogger(__name__).debug(
                    f"Rate intelligence benchmarking skipped: {_ri_err}"
                )

            _by_source: dict = {}
            _by_trade: dict  = {}
            _rate_benchmarked: int = 0
            for _li_dict in _line_items_payload:
                _by_source[_li_dict.get("source", "unknown")] = (
                    _by_source.get(_li_dict.get("source", "unknown"), 0) + 1
                )
                _by_trade[_li_dict.get("trade", "general")] = (
                    _by_trade.get(_li_dict.get("trade", "general"), 0) + 1
                )
                _bm = _li_dict.get("rate_benchmark", {})
                if _bm and _bm.get("status") not in ("NO_MATCH", "UNRATED", None):
                    _rate_benchmarked += 1
            _line_items_summary = {
                "total":            len(_line_items_payload),
                "by_source":        _by_source,
                "by_trade":         _by_trade,
                "taxonomy_matched": sum(1 for li in _unified_items if li.taxonomy_matched),
                "qty_present":      sum(1 for li in _unified_items if not li.qty_missing),
                "unit_present":     sum(1 for li in _unified_items if li.unit),
                "rate_benchmarked": _rate_benchmarked,
            }
            # Contractual / administrative clauses (not priceable work items)
            _contractual_raw = build_contractual_items(_spec_items_list)
        except Exception as _norm_err:
            import logging as _nlog
            _nlog.getLogger(__name__).warning(
                f"Item normalisation failed: {_norm_err}"
            )
            _contractual_raw = []

        # ── QTO: Room-finish takeoff from plan drawings ──────────────────
        _qto_finish_items: list = []
        _qto_rooms: list = []
        try:
            from .qto.finish_takeoff import run_finish_takeoff
            # Gather (page_idx, text, doc_type) for all processed pages using
            # all_page_texts (flat list indexed by page_idx) + page_index_result
            _plan_page_texts = []
            if page_index_result and all_page_texts:
                for pg in page_index_result.pages:
                    pidx = pg.page_idx
                    if pidx < len(all_page_texts):
                        _plan_page_texts.append((
                            pidx,
                            all_page_texts[pidx],
                            pg.doc_type,
                        ))
            # Filter finish schedules
            _finish_scheds = [
                s for s in getattr(extraction_result, 'schedules', [])
                if str(s.get('schedule_type', '')).lower() in ('finish', 'finishes', 'room_finish')
            ]
            if _plan_page_texts:
                _qto_rooms, _qto_finish_items = run_finish_takeoff(
                    _plan_page_texts, _finish_scheds
                )
            if _qto_finish_items:
                # Add QTO items into spec_items for normaliser to pick up
                _spec_items_list = list(_spec_items_list) + _qto_finish_items
        except Exception as _qto_err:
            import logging as _qlog
            _qlog.getLogger(__name__).warning(f"QTO finish takeoff failed: {_qto_err}")

        # ── Scale Detection (Sprint 43 — early, feeds structural + visual) ──
        # Runs against all page texts before any geometry-dependent QTO module.
        # Result: ScaleInfo with px_per_mm, ratio, confidence.
        _detected_scale = None
        _all_page_texts_for_scale: list = []
        try:
            from .qto.scale_detector import detect_scale as _detect_scale_fn
            if page_index_result and all_page_texts:
                for pg in page_index_result.pages:
                    pidx = pg.page_idx
                    if pidx < len(all_page_texts):
                        _all_page_texts_for_scale.append((
                            pidx,
                            all_page_texts[pidx] or "",
                            getattr(pg, "doc_type", "unknown"),
                        ))
            elif all_page_texts:
                _all_page_texts_for_scale = [
                    (i, t or "", "unknown") for i, t in enumerate(all_page_texts)
                ]
            _detected_scale = _detect_scale_fn(_all_page_texts_for_scale)
        except Exception as _scale_early_err:
            import logging as _scaleearly_log
            _scaleearly_log.getLogger(__name__).debug(
                "Early scale detection failed (non-critical): %s", _scale_early_err
            )

        # ── QTO: Structural takeoff from schedule text ──────────────────
        _qto_structural_items: list = []
        _qto_structural_elements: list = []
        _qto_structural_mode = "none"
        _qto_structural_warnings: list = []
        try:
            from .qto.structural_takeoff import run_structural_takeoff

            # Collect (page_idx, text, doc_type) for structural/plan pages
            _structural_page_texts = []
            if page_index_result and all_page_texts:
                for pg in page_index_result.pages:
                    pidx = pg.page_idx
                    if pidx < len(all_page_texts):
                        _structural_page_texts.append((
                            pidx,
                            all_page_texts[pidx],
                            pg.doc_type,
                        ))

            # ── Floor count: text extractor first, metadata fallback ────────
            _st_floors = 1
            _st_area_sqm = 0.0
            _st_floor_source = "default"
            try:
                from .qto.floor_count_extractor import extract_floor_count
                _fc = extract_floor_count(_structural_page_texts)
                if _fc and _fc.count >= 1:
                    _st_floors = _fc.count
                    _st_floor_source = (
                        f"text:{_fc.source_text!r} (p{_fc.source_page+1}, "
                        f"conf={_fc.confidence:.0%})"
                    )
            except Exception as _fc_err:
                import logging as _fclog
                _fclog.getLogger(__name__).debug(f"Floor count extractor: {_fc_err}")

            # Override / supplement from metadata if extractor found nothing
            if hasattr(extraction_result, 'metadata') and extraction_result.metadata:
                _meta = extraction_result.metadata or {}
                if _st_floors == 1 and _st_floor_source == "default":
                    _meta_floors = int(_meta.get('floors', _meta.get('storeys', 1)) or 1)
                    if _meta_floors > 1:
                        _st_floors = _meta_floors
                        _st_floor_source = "metadata"

            # Built-up area from rooms or metadata
            if _qto_rooms:
                _st_area_sqm = sum(
                    r.area_sqm for r in _qto_rooms if r.area_sqm and r.area_sqm > 0
                )
            if not _st_area_sqm:
                _meta2 = getattr(extraction_result, 'metadata', None) or {}
                _st_area_sqm = float(_meta2.get('total_area_sqm', 0) or 0)

            if _structural_page_texts:
                _st_result = run_structural_takeoff(
                    page_texts=_structural_page_texts,
                    floors=max(1, _st_floors),
                    total_area_sqm=_st_area_sqm,
                    px_per_mm=_detected_scale.px_per_mm if _detected_scale else 0.0,
                    known_scale_ratio=_detected_scale.ratio if _detected_scale else 0,
                )
                _qto_structural_items = _st_result.line_items
                _qto_structural_elements = _st_result.elements
                _qto_structural_mode = _st_result.mode
                _qto_structural_warnings = _st_result.warnings

                if _qto_structural_items:
                    _spec_items_list = list(_spec_items_list) + _qto_structural_items
        except Exception as _st_err:
            import logging as _stlog
            _stlog.getLogger(__name__).warning(
                f"QTO structural takeoff failed: {_st_err}"
            )

        # ── QTO: Implied items rule engine ──────────────────────────────
        _qto_implied_items: list = []
        _qto_implied_rules_triggered: list = []
        try:
            from .qto.implied_items import run_implied_rules, build_rule_context

            # Collect any drawing callouts from extraction result
            _drawing_callouts: list = []
            if extraction_result and hasattr(extraction_result, 'drawing_callouts'):
                _drawing_callouts = list(extraction_result.drawing_callouts or [])

            _implied_ctx = build_rule_context(
                structural_elements=_qto_structural_elements,
                structural_items=_qto_structural_items,
                rooms=_qto_rooms,
                total_area_sqm=_st_area_sqm,
                floors=max(1, _st_floors),
                building_type="residential",
                storey_height_mm=3000,
                drawing_callouts=_drawing_callouts,
            )
            _qto_implied_items, _qto_implied_rules_triggered = run_implied_rules(_implied_ctx)
            if _qto_implied_items:
                _spec_items_list = list(_spec_items_list) + _qto_implied_items
        except Exception as _imp_err:
            import logging as _implog
            _implog.getLogger(__name__).warning(
                f"QTO implied items failed: {_imp_err}"
            )

        # ── QTO: MEP Takeoff (Sprint 37) ────────────────────────────
        _qto_mep_elements: list = []
        _qto_mep_items: list = []
        _qto_mep_mode = "none"
        _qto_mep_warnings: list = []
        _mep_page_texts: list = []
        try:
            from .qto.mep_takeoff import run_mep_takeoff

            # Reuse page texts already collected for scale detection, or rebuild
            _mep_page_texts = _all_page_texts_for_scale if _all_page_texts_for_scale else []
            if not _mep_page_texts:
                if page_index_result and all_page_texts:
                    for pg in page_index_result.pages:
                        pidx = pg.page_idx
                        if pidx < len(all_page_texts):
                            _mep_page_texts.append((
                                pidx,
                                all_page_texts[pidx] or "",
                                getattr(pg, "doc_type", "unknown"),
                            ))
                elif all_page_texts:
                    _mep_page_texts = [
                        (i, t or "", "unknown") for i, t in enumerate(all_page_texts)
                    ]

            _mep_result = run_mep_takeoff(
                page_texts=_mep_page_texts,
                floors=max(1, _st_floors),
                total_area_sqm=_st_area_sqm,
                building_type="residential",
            )
            _qto_mep_elements = _mep_result.elements
            _qto_mep_items    = _mep_result.line_items
            _qto_mep_mode     = _mep_result.mode
            _qto_mep_warnings = _mep_result.warnings

            if _qto_mep_items:
                _spec_items_list = list(_spec_items_list) + _qto_mep_items

        except Exception as _mep_err:
            import logging as _meplog
            _meplog.getLogger(__name__).warning(
                f"MEP takeoff failed: {_mep_err}"
            )

        # ── QTO: Visual Element Detection (Sprint 37) ────────────────
        _qto_visual_elements: list = []
        _qto_visual_items: list = []
        _qto_visual_mode = "none"
        _qto_visual_warnings: list = []
        _qto_visual_scale = ""
        _qto_visual_area = 0.0
        _primary_pdf = payload.get("primary_pdf_path") if payload else None
        if not _primary_pdf:
            _primary_pdf = getattr(result, 'pdf_path', None)
        try:
            if llm_client is not None and _primary_pdf and _mep_page_texts:
                from .qto.visual_element_detector import run_visual_detection
                _vis_result = run_visual_detection(
                    pdf_path=_primary_pdf,
                    page_texts=_mep_page_texts,
                    llm_client=llm_client,
                )
                _qto_visual_elements = _vis_result.elements
                _qto_visual_items    = _vis_result.line_items
                _qto_visual_mode     = _vis_result.mode
                _qto_visual_warnings = _vis_result.warnings
                _qto_visual_scale    = _vis_result.detected_scale
                _qto_visual_area     = _vis_result.detected_area_sqm

                if _qto_visual_items:
                    _spec_items_list = list(_spec_items_list) + _qto_visual_items
        except Exception as _vis_err:
            import logging as _vislog
            _vislog.getLogger(__name__).warning(
                f"Visual element detection failed: {_vis_err}"
            )

        # ── QTO: Visual Measurement (Sprint 37) ──────────────────────
        _qto_vmeas_rooms: list = []
        _qto_vmeas_items: list = []
        _qto_vmeas_mode = "none"
        _qto_vmeas_warnings: list = []
        _qto_vmeas_scale = ""
        _qto_vmeas_area = 0.0
        _qto_vmeas_room_schedule: list = []
        try:
            if llm_client is not None and _primary_pdf and _mep_page_texts:
                from .qto.visual_measurement import run_visual_measurement
                _vmeas_result = run_visual_measurement(
                    pdf_path=_primary_pdf,
                    page_texts=_mep_page_texts,
                    llm_client=llm_client,
                    known_scale_ratio=_detected_scale.ratio if _detected_scale else 0,
                    known_px_per_mm=_detected_scale.px_per_mm if _detected_scale else 0.0,
                )
                _qto_vmeas_rooms         = _vmeas_result.all_rooms
                _qto_vmeas_items         = _vmeas_result.line_items
                _qto_vmeas_mode          = _vmeas_result.mode
                _qto_vmeas_warnings      = _vmeas_result.warnings
                _qto_vmeas_scale         = _vmeas_result.detected_scale
                _qto_vmeas_area          = _vmeas_result.total_area_sqm
                _qto_vmeas_room_schedule = _vmeas_result.room_schedule

                # Feed measured rooms into _qto_rooms (used by finish_takeoff)
                if _vmeas_result.room_schedule and not _qto_rooms:
                    _qto_rooms = list(_vmeas_result.room_schedule)

                # Add measurement-derived finish items to spec items
                if _qto_vmeas_items:
                    _spec_items_list = list(_spec_items_list) + _qto_vmeas_items
        except Exception as _vmeas_err:
            import logging as _vmeaslog
            _vmeaslog.getLogger(__name__).warning(
                f"Visual measurement failed: {_vmeas_err}"
            )

        # ── QTO: Door & Window Takeoff (Sprint 38) ───────────────────
        _qto_dw_elements: list = []
        _qto_dw_items: list = []
        _qto_dw_mode = "none"
        _qto_dw_warnings: list = []
        _qto_dw_door_count = 0
        _qto_dw_window_count = 0
        try:
            from .qto.door_window_takeoff import run_dw_takeoff
            _dw_result = run_dw_takeoff(
                page_texts=_mep_page_texts,
                floors=max(1, _st_floors),
                total_area_sqm=_st_area_sqm,
            )
            _qto_dw_elements    = _dw_result.elements
            _qto_dw_items       = _dw_result.line_items
            _qto_dw_mode        = _dw_result.mode
            _qto_dw_warnings    = _dw_result.warnings
            _qto_dw_door_count  = _dw_result.door_count
            _qto_dw_window_count= _dw_result.window_count
            if _qto_dw_items:
                _spec_items_list = list(_spec_items_list) + _qto_dw_items
        except Exception as _dw_err:
            import logging as _dwlog
            _dwlog.getLogger(__name__).warning(f"Door/window takeoff failed: {_dw_err}")

        # ── QTO: Painting Takeoff (Sprint 38) ────────────────────────
        _qto_paint_items: list = []
        _qto_paint_mode = "none"
        _qto_paint_warnings: list = []
        _qto_paint_int_wall = 0.0
        _qto_paint_ceiling = 0.0
        _qto_paint_ext_wall = 0.0
        try:
            from .qto.painting_takeoff import run_painting_takeoff
            _paint_result = run_painting_takeoff(
                rooms=_qto_rooms,
                floor_area_sqm=_st_area_sqm,
                floors=max(1, _st_floors),
                door_count=_qto_dw_door_count,
                window_count=_qto_dw_window_count,
            )
            _qto_paint_items    = _paint_result.line_items
            _qto_paint_mode     = _paint_result.mode
            _qto_paint_warnings = _paint_result.warnings
            _qto_paint_int_wall = _paint_result.total_interior_wall_sqm
            _qto_paint_ceiling  = _paint_result.total_ceiling_sqm
            _qto_paint_ext_wall = _paint_result.total_exterior_wall_sqm
            if _qto_paint_items:
                _spec_items_list = list(_spec_items_list) + _qto_paint_items
        except Exception as _paint_err:
            import logging as _paintlog
            _paintlog.getLogger(__name__).warning(f"Painting takeoff failed: {_paint_err}")

        # ── QTO: Waterproofing Takeoff (Sprint 38) ───────────────────
        _qto_wp_items: list = []
        _qto_wp_mode = "none"
        _qto_wp_warnings: list = []
        _qto_wp_wet_area = 0.0
        _qto_wp_roof_area = 0.0
        try:
            from .qto.waterproofing_takeoff import run_waterproofing_takeoff
            _wp_result = run_waterproofing_takeoff(
                rooms=_qto_rooms,
                floor_area_sqm=_st_area_sqm,
                floors=max(1, _st_floors),
            )
            _qto_wp_items    = _wp_result.line_items
            _qto_wp_mode     = _wp_result.mode
            _qto_wp_warnings = _wp_result.warnings
            _qto_wp_wet_area = _wp_result.wet_area_sqm
            _qto_wp_roof_area= _wp_result.roof_area_sqm
            if _qto_wp_items:
                _spec_items_list = list(_spec_items_list) + _qto_wp_items
        except Exception as _wp_err:
            import logging as _wplog
            _wplog.getLogger(__name__).warning(f"Waterproofing takeoff failed: {_wp_err}")

        # ── QTO: Site Work Takeoff (Sprint 38) ───────────────────────
        _qto_sw_items: list = []
        _qto_sw_mode = "none"
        _qto_sw_warnings: list = []
        try:
            from .qto.sitework_takeoff import run_sitework_takeoff
            _sw_result = run_sitework_takeoff(
                plot_area_sqm=0.0,         # not yet extracted; fallback estimates it
                built_area_sqm=_st_area_sqm / max(1, _st_floors),
                total_floor_area_sqm=_st_area_sqm,
                floors=max(1, _st_floors),
            )
            _qto_sw_items    = _sw_result.line_items
            _qto_sw_mode     = _sw_result.mode
            _qto_sw_warnings = _sw_result.warnings
            if _qto_sw_items:
                _spec_items_list = list(_spec_items_list) + _qto_sw_items
        except Exception as _sw_err:
            import logging as _swlog
            _swlog.getLogger(__name__).warning(f"Sitework takeoff failed: {_sw_err}")

        # ── Rate Engine: Apply rates to ALL spec items (Sprint 38) ───
        _qto_rated_items: list = []
        _qto_trade_summary: dict = {}
        _qto_grand_total_inr: float = 0.0
        try:
            from .qto.rate_engine import apply_rates, compute_trade_summary
            _qto_rated_items = apply_rates(list(_spec_items_list), region="tier1")
            _qto_trade_summary = compute_trade_summary(_qto_rated_items)
            _qto_grand_total_inr = sum(
                t.get("total_amount", 0) for t in _qto_trade_summary.values()
            )
            # Store rated items back as spec_items_list so export sees rates
            _spec_items_list = _qto_rated_items
        except Exception as _rate_err:
            import logging as _ratelog
            _ratelog.getLogger(__name__).warning(f"Rate engine failed: {_rate_err}")

        # ── Sprint 41: Rebuild unified line_items after all QTO modules ──
        # The initial build_line_items() call (Sprint 21D above) ran with only
        # extraction_result.spec_items.  All QTO modules (MEP, visual, D&W,
        # painting, waterproofing, sitework, rate engine) have now appended to
        # _spec_items_list.  Rebuild so payload["line_items"] is complete.
        _dedup_stats: dict = {}
        try:
            from .item_normalizer import build_line_items as _build_line_items_full
            _full_unified = _build_line_items_full(
                boq_items      = extraction_result.boq_items,
                spec_items     = list(_spec_items_list),  # full list with QTO items + rates
                schedule_stubs = _stub_items_list if _recon else [],
                dedup          = True,
            )
            from .item_normalizer import get_last_dedup_stats as _get_dedup_stats
            _dedup_stats = _get_dedup_stats()
            _line_items_payload = [li.to_dict() for li in _full_unified]
        except Exception as _rebuild_err:
            import logging as _rebuildlog
            _rebuildlog.getLogger(__name__).warning(
                "Sprint 41: line_items rebuild failed (non-critical): %s", _rebuild_err
            )

        # ── Excel Export (Sprint 38) ─────────────────────────────────
        _qto_excel_bytes: bytes = b""
        try:
            from .export.excel_exporter import export_to_excel
            _excel_meta = {
                "total_area_sqm": _st_area_sqm,
                "floors": _st_floors,
            }
            _proj_name = (
                payload.get("project_name") if payload else None
            ) or "Project"
            _excel_result = export_to_excel(
                list(_spec_items_list),
                project_name=_proj_name,
                project_meta=_excel_meta,
            )
            _qto_excel_bytes = _excel_result or b""
        except Exception as _excel_err:
            import logging as _excellog
            _excellog.getLogger(__name__).warning(f"Excel export failed: {_excel_err}")

        payload = getattr(result, 'payload', {})
        payload["qto_summary"] = {
            "rooms_detected": len(_qto_rooms),
            "finish_items_generated": len(_qto_finish_items),
            "structural_elements_detected": len(_qto_structural_elements),
            "structural_items_generated": len(_qto_structural_items),
            "structural_mode": _qto_structural_mode,
            "structural_warnings": _qto_structural_warnings,
            "implied_items_generated": len(_qto_implied_items),
            "implied_rules_triggered": _qto_implied_rules_triggered,
            "mep_elements_detected": len(_qto_mep_elements),
            "mep_items_generated": len(_qto_mep_items),
            "mep_mode": _qto_mep_mode,
            "mep_warnings": _qto_mep_warnings,
            "visual_elements_detected": len(_qto_visual_elements),
            "visual_items_generated": len(_qto_visual_items),
            "visual_mode": _qto_visual_mode,
            "visual_warnings": _qto_visual_warnings,
            "visual_scale": _qto_visual_scale,
            "visual_area_sqm": _qto_visual_area,
            # Visual measurement
            "vmeas_rooms_measured": len(_qto_vmeas_rooms),
            "vmeas_items_generated": len(_qto_vmeas_items),
            "vmeas_mode": _qto_vmeas_mode,
            "vmeas_warnings": _qto_vmeas_warnings,
            "vmeas_scale": _qto_vmeas_scale,
            "vmeas_area_sqm": _qto_vmeas_area,
            "vmeas_room_schedule": [
                {
                    "name": getattr(r, "name", ""),
                    "raw_name": getattr(r, "raw_name", ""),
                    "area_sqm": getattr(r, "area_sqm", 0),
                    "dim_l": getattr(r, "dim_l", None),
                    "dim_w": getattr(r, "dim_w", None),
                    "source_page": getattr(r, "source_page", 0),
                    "confidence": getattr(r, "confidence", 0),
                }
                for r in _qto_vmeas_room_schedule
            ],
            # Sprint 38 — new trades
            "dw_elements_detected": len(_qto_dw_elements),
            "dw_items_generated": len(_qto_dw_items),
            "dw_mode": _qto_dw_mode,
            "dw_warnings": _qto_dw_warnings,
            "dw_door_count": _qto_dw_door_count,
            "dw_window_count": _qto_dw_window_count,
            "paint_items_generated": len(_qto_paint_items),
            "paint_mode": _qto_paint_mode,
            "paint_warnings": _qto_paint_warnings,
            "paint_int_wall_sqm": _qto_paint_int_wall,
            "paint_ceiling_sqm": _qto_paint_ceiling,
            "paint_ext_wall_sqm": _qto_paint_ext_wall,
            "wp_items_generated": len(_qto_wp_items),
            "wp_mode": _qto_wp_mode,
            "wp_warnings": _qto_wp_warnings,
            "wp_wet_area_sqm": _qto_wp_wet_area,
            "wp_roof_area_sqm": _qto_wp_roof_area,
            "sw_items_generated": len(_qto_sw_items),
            "sw_mode": _qto_sw_mode,
            "sw_warnings": _qto_sw_warnings,
            # Sprint 42 — detected drawing scale
            "detected_scale": (
                {
                    "ratio": _detected_scale.ratio,
                    "is_nts": _detected_scale.is_nts,
                    "px_per_mm": round(_detected_scale.px_per_mm, 6),
                    "confidence": round(_detected_scale.confidence, 3),
                    "source_page": _detected_scale.source_page,
                    "source_text": _detected_scale.source_text,
                }
                if _detected_scale is not None else None
            ),
            # Cost estimate
            "grand_total_inr": _qto_grand_total_inr,
            "trade_summary": {
                trade: {
                    "item_count": info.get("item_count", 0),
                    "total_amount": info.get("total_amount", 0),
                }
                for trade, info in _qto_trade_summary.items()
            },
            "total_spec_items": len(_spec_items_list),
            # Excel bytes stored separately to avoid bloating JSON
            "_excel_available": len(_qto_excel_bytes) > 0,
        }
        # Store Excel bytes separately on result for download
        if _qto_excel_bytes:
            payload["_excel_bytes"] = _qto_excel_bytes
        # Store rated items list for Excel download
        if _qto_rated_items:
            payload["spec_items"] = _qto_rated_items

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

        # Track optional-stage failures so contractors can see what was skipped.
        # These populate payload["pipeline_warnings"] shown as UI banners.
        _pipeline_warnings: List[str] = []

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
        except Exception as e:
            logger.warning("Toxic page retry failed (non-critical): %s", e)

        # ── Build OCR text cache for NOT_FOUND proof ────────────────
        ocr_text_cache = {}
        for idx, text in enumerate(all_page_texts):
            if text and len(text.strip()) > 10:
                ocr_text_cache[idx] = text[:10000]  # cap at 10KB per page

        # Sprint 10: Cache OCR text
        try:
            save_cached_stage(cache_dir, "ocr_text", ocr_text_cache)
        except Exception as e:
            logger.debug("OCR text cache save failed: %s", e)

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
        except Exception as e:
            logger.debug("Surya OCR not available or failed (non-critical): %s", e)

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

        # Stamp package classification onto graph
        if _pkg_classification is not None:
            plan_graph.package_type = _pkg_classification.package_type.value
            plan_graph.package_type_confidence = _pkg_classification.confidence

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
        from .llm_enrichment import calculate_readiness_score, enrich_analysis
        score_result = calculate_readiness_score(plan_graph, blockers, trade_coverage=coverage)
        update_progress("reason", f"Readiness: {score_result.status} ({score_result.total_score}/100)", 1.0)

        # LLM enrichment — runs only when llm_client is provided (optional)
        if llm_client is not None:
            try:
                from src.models.analysis_models import DeepAnalysisResult
                _deep = DeepAnalysisResult(
                    blockers=blockers,
                    trade_coverage=coverage or [],
                    plan_graph=plan_graph,
                    readiness_score=score_result,
                )
                _deep = enrich_analysis(_deep, llm_client=llm_client)
                # Propagate enriched score back
                score_result = _deep.readiness_score
                blockers = _deep.blockers
            except Exception as _llm_exc:
                logger.warning("LLM enrichment skipped: %s", _llm_exc, exc_info=True)
                _pipeline_warnings.append(
                    "LLM enrichment skipped — blocker risk assessments use template defaults. "
                    f"Set OPENAI_API_KEY or ANTHROPIC_API_KEY for enriched analysis. ({type(_llm_exc).__name__})"
                )

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
                package_classification=_pkg_classification,
                blockers=blockers,
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

        # --- Knowledge Base RFI extension (additive) ---
        try:
            from .rfi_engine import generate_knowledge_base_rfis
            _scope_gaps = []
            if plan_graph and hasattr(plan_graph, 'scope_analysis') and plan_graph.scope_analysis:
                _sa = plan_graph.scope_analysis
                _scope_gaps = [
                    {"missing_item": g.get("item", g.get("missing_item", "")),
                     "trade": g.get("trade", "")}
                    for g in (getattr(_sa, 'gaps', None) or
                              getattr(_sa, 'missing_items', None) or [])
                    if isinstance(g, dict)
                ]
            _project_params = {
                "building_type": getattr(plan_graph, 'building_type', "all") if plan_graph else "all",
            }
            # Pass actual BOQ items so the KB can validate rates + units
            _actual_boq = extraction_result.boq_items if extraction_result else []
            _kb_rfis = generate_knowledge_base_rfis(
                _scope_gaps, rfi_list, _project_params,
                actual_boq_items=_actual_boq,
            )
            if _kb_rfis:
                _existing_questions = {r.get("question", "")[:80] for r in rfi_list}
                _added = 0
                for _kr in _kb_rfis:
                    if _kr.get("question", "")[:80] not in _existing_questions:
                        rfi_list.append(_kr)
                        _existing_questions.add(_kr.get("question", "")[:80])
                        _added += 1
                if _added:
                    import logging as _log
                    _log.getLogger(__name__).info(
                        "Knowledge base: +%d RFIs (total: %d)", _added, len(rfi_list)
                    )
        except Exception as _kb_err:
            import logging as _log
            _log.getLogger(__name__).debug("KB RFI extension skipped: %s", _kb_err)

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
        except Exception as e:
            logger.warning("Toxic pages summary failed (non-critical): %s", e)

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
        except Exception as e:
            logger.warning("QA score computation failed (non-critical): %s", e)

        _rfi_clusters = []
        try:
            from .rfi_clustering import cluster_rfis
            _rfi_clusters = cluster_rfis(
                rfis_payload,
                multi_doc_index=multi_doc_index.to_dict() if multi_doc_index else None,
            )
        except Exception as e:
            logger.warning("RFI clustering failed (non-critical): %s", e, exc_info=True)
            _pipeline_warnings.append(f"RFI grouping skipped — RFIs shown ungrouped. ({type(e).__name__}: {e})")

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
        except Exception as e:
            logger.warning("Quantities build failed (non-critical): %s", e, exc_info=True)
            _pipeline_warnings.append(f"Quantity schedule not built — quantity tab will be empty. ({type(e).__name__}: {e})")

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
        except Exception as e:
            logger.warning("Pricing guidance failed (non-critical): %s", e, exc_info=True)
            _pipeline_warnings.append(f"Pricing guidance skipped — cost risk summary not available. ({type(e).__name__}: {e})")

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
        except Exception as e:
            logger.warning("Quantity reconciliation failed (non-critical): %s", e, exc_info=True)
            _pipeline_warnings.append(f"Quantity reconciliation skipped — BOQ vs drawing quantity check not run. ({type(e).__name__}: {e})")

        # Sprint 22 Phase 4C: Auto-generate RFIs for significant qty discrepancies
        try:
            if _qty_reconciliation:
                from .bulk_actions import generate_rfis_for_high_mismatches
                _recon_rfis, _qty_reconciliation = generate_rfis_for_high_mismatches(
                    _qty_reconciliation, rfis_payload,
                )
                if _recon_rfis:
                    for _rr in _recon_rfis:
                        _rr["source"] = "qty_reconciliation"
                    rfis_payload.extend(_recon_rfis)
        except Exception as e:
            logger.warning("Auto RFI generation from reconciliation failed (non-critical): %s", e)

        # Sprint 12: Finish takeoff
        _finish_takeoff = None
        try:
            from .finish_takeoff import build_finish_takeoff
            _finish_takeoff = build_finish_takeoff(
                schedules=extraction_result.schedules if extraction_result else [],
            )
        except Exception as e:
            logger.warning("Finish takeoff failed (non-critical): %s", e, exc_info=True)
            _pipeline_warnings.append(f"Finish takeoff skipped — finish schedule quantities not extracted. ({type(e).__name__}: {e})")

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
        except Exception as e:
            logger.warning("Review queue build failed (non-critical): %s", e)

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

        # Page-level cache stats (from extractors module-level instance)
        _page_cache_stats: dict = {}
        try:
            from .extractors import _get_page_cache as _get_pc
            _pc = _get_pc()
            if _pc is not None:
                _page_cache_stats = _pc.stats()
        except Exception as _pc_exc:
            logger.debug("Page cache stats unavailable: %s", _pc_exc)

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
            "line_items": _line_items_payload,
            "line_items_summary": _line_items_summary,
            "dedup_stats": _dedup_stats,
            "contractual_items": _contractual_raw,
            "schedule_boq_links": _schedule_boq_links,
            "epc_mode": _boq_stats.get("epc_mode", False),
            "epc_mode_note": _boq_stats.get("epc_mode_note"),
            "addendum_index": [a.to_dict() for a in addenda_list],
            "conflicts": all_conflicts,
            "reconciliation_findings": reconciliation_findings,
            "run_coverage": run_coverage.to_dict() if run_coverage else None,
            "guardrail_warnings": guardrail_warnings,
            # Visible warnings for skipped optional stages (shown in UI as yellow banners)
            "pipeline_warnings": _pipeline_warnings,
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
                # Sprint 25: plan_graph summary (door/window/room aggregates)
                "plan_graph": {
                    "all_door_tags": getattr(plan_graph, 'all_door_tags', []),
                    "all_window_tags": getattr(plan_graph, 'all_window_tags', []),
                    "all_room_names": getattr(plan_graph, 'all_room_names', []),
                    "has_door_schedule": getattr(plan_graph, 'has_door_schedule', False),
                    "has_window_schedule": getattr(plan_graph, 'has_window_schedule', False),
                    "has_finish_schedule": getattr(plan_graph, 'has_finish_schedule', False),
                    "total_pages": getattr(plan_graph, 'total_pages', 0),
                    "pages_with_scale": getattr(plan_graph, 'pages_with_scale', 0),
                    "pages_without_scale": getattr(plan_graph, 'pages_without_scale', 0),
                    "disciplines_found": getattr(plan_graph, 'disciplines_found', []),
                } if plan_graph else None,
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
            # Page-level extraction cache stats
            "page_cache_stats": _page_cache_stats,
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
            # QTO: Room-finish takeoff summary
            "qto_summary": payload.get("qto_summary", {}),
        }

        # ── Accuracy report ─────────────────────────────────────────
        try:
            from src.reporting.accuracy_report import generate_report as _gen_report
            _accuracy_report = _gen_report({
                "line_items": _line_items_payload,
                "contractual_items": _contractual_raw,
                "line_items_summary": _line_items_summary,
                "qto_summary": payload.get("qto_summary", {}),
                "cache_stats": payload.get("cache_stats", {}),
                "page_index": page_index_result.__dict__ if page_index_result else {},
            })
        except Exception:
            _accuracy_report = {}
        result.payload["accuracy_report"] = _accuracy_report

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
        except Exception as e:
            logger.warning("Run comparison failed (non-critical): %s", e)

        # Also save the full payload as analysis.json (replace the minimal one)
        full_analysis_path = output_dir / "analysis.json"
        with open(full_analysis_path, 'w') as f:
            json.dump(result.payload, f, indent=2, default=str)

        # Sprint 26: Generate printable exports (summary.md, rfis.csv, etc.)
        try:
            from src.summary_report import save_exports as _save_report_exports
            _export_paths = _save_report_exports(result.payload, str(output_dir))
            result.files_generated.extend(_export_paths)
        except Exception as _exp_exc:
            logger.warning("Sprint 26 export generation failed: %s", _exp_exc)

        # Sprint 28: Pre-generate SVGs for drawing pages (non-blocking)
        try:
            if page_index_result and input_files:
                _svg_doc_types = {"plan", "detail", "section", "elevation"}
                _svg_dir = output_dir / "svgs"
                _svg_pages = [
                    p for p in page_index_result.pages
                    if getattr(p, "doc_type", "") in _svg_doc_types
                ]
                if _svg_pages:
                    import fitz as _svg_fitz
                    _svg_dir.mkdir(parents=True, exist_ok=True)
                    _svg_generated = []
                    for _sp in _svg_pages:
                        try:
                            _svg_doc = _svg_fitz.open(str(input_files[0]))
                            if _sp.page_idx < len(_svg_doc):
                                _svg_str = _svg_doc[_sp.page_idx].get_svg_image(text_as_path=True)
                                _svg_path = _svg_dir / f"page_{_sp.page_idx:04d}.svg"
                                _svg_path.write_text(_svg_str, encoding="utf-8")
                                _svg_generated.append(str(_svg_path))
                            _svg_doc.close()
                        except Exception as _sp_exc:
                            logger.debug("SVG generation failed for page %d: %s", _sp.page_idx, _sp_exc)
                            continue
                    result.files_generated.extend(_svg_generated)
                    if _svg_generated:
                        logger.info("Sprint 28: Generated %d SVGs for drawing pages", len(_svg_generated))
        except Exception as _svg_exc:
            logger.warning("Sprint 28 SVG generation failed (non-critical): %s", _svg_exc)

        # Sprint 29: Build and save vector index for RAG semantic search
        try:
            from src.vectorstore import build_index_from_payload
            _vec_index = build_index_from_payload(result.payload)
            if _vec_index.size > 0:
                _vec_path = output_dir / "vector_index.json"
                _vec_index.save(str(_vec_path))
                result.files_generated.append(str(_vec_path))
                logger.info("Sprint 29: Vector index saved — %d chunks, %d vocab terms",
                            _vec_index.size, _vec_index.vocab_size)
        except Exception as _vec_exc:
            logger.warning("Sprint 29 vector index build failed (non-critical): %s", _vec_exc)

        # ── Semantic Intelligence Layer (Sprint 40) ───────────────────────────
        # ChromaDB dense embeddings + gap analysis + bid synthesis
        # Gracefully skipped if chromadb / sentence-transformers not installed
        try:
            from src.embeddings.embedder import Embedder as _Embedder
            from src.embeddings.chroma_store import BidChromaStore as _BidChromaStore
            from src.reasoning.gap_analyzer import analyze_gaps as _analyze_gaps
            from src.reasoning.cost_impact import estimate_cost_impact as _estimate_cost_impact
            from src.reasoning.bid_synthesizer import synthesize_bid as _synthesize_bid
            from dataclasses import asdict as _asdict

            _intel_embedder = _Embedder(backend="auto")
            _intel_store = _BidChromaStore(
                project_id=project_id,
                persist_dir=str(output_dir / ".chroma"),
            )
            _intel_n_chunks = _intel_store.index_payload(result.payload, _intel_embedder)

            _intel_gaps = _analyze_gaps(
                result.payload, _intel_store, _intel_embedder, llm_client,
            )
            _intel_impacts = [
                _estimate_cost_impact(g, result.payload) for g in _intel_gaps
            ]
            _intel_synthesis = _synthesize_bid(
                result.payload, _intel_gaps, _intel_impacts, llm_client,
            )

            result.payload["bid_synthesis"] = _intel_synthesis.to_dict()
            result.payload["gaps"] = [
                {
                    "id": g.id,
                    "trade": g.trade,
                    "severity": g.severity,
                    "description": g.description,
                    "evidence": g.evidence,
                    "action_required": g.action_required,
                    "cost_impact": g.cost_impact,
                    "source": g.source,
                }
                for g in _intel_gaps
            ]
            result.payload["chroma_indexed_chunks"] = _intel_n_chunks

            logger.info(
                "Sprint 40: Intelligence layer — %d chunks, %d gaps, readiness=%d",
                _intel_n_chunks, len(_intel_gaps), _intel_synthesis.bid_readiness_score,
            )
        except ImportError:
            logger.info(
                "Sprint 40: chromadb/sentence-transformers not installed; "
                "install them for full bid intelligence: pip install chromadb sentence-transformers"
            )
        except Exception as _intel_exc:
            logger.warning("Sprint 40 intelligence layer failed (non-critical): %s", _intel_exc)

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
        except Exception as e:
            logger.debug("Compare snapshot save failed: %s", e)

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
        except Exception as e:
            logger.warning("Project run save failed (non-critical): %s", e)

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
