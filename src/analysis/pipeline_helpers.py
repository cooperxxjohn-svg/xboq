"""
pipeline_helpers.py — Standalone helpers extracted from pipeline.py.

Extracted so they can be imported, tested, and read independently without
pulling in the 3,000-line pipeline orchestrator.

All public names are re-exported by pipeline.py for backward compatibility.
"""

import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Data-classes
# =============================================================================

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


# =============================================================================
# PDF loading helpers
# =============================================================================

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
    metadata: Dict[str, Any] = {
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


def extract_text_from_pdf(
    pdf_path: Path, use_ocr_fallback: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
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


# =============================================================================
# Run diff / guardrails helpers
# =============================================================================

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

    # Approval deltas
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
    warnings: List[dict] = []

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


# =============================================================================
# BOQ stats / EPC detection helpers
# =============================================================================

def _detect_epc_mode(boq_items: list) -> bool:
    """Detect EPC/Turnkey contracts from BOQ structure.

    Signal: all priceable BOQ items are lump-sum (unit=LS/LUMP) AND item count is ≤5.
    """
    if not boq_items:
        return False
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


# =============================================================================
# Processing stats builder
# =============================================================================

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
        Also table_attempt_pages, table_success_pages, selection_mode,
        selected_pages_count.
    All fields default to 0 if data is unavailable.
    """
    stats: dict = {
        "total_pages": 0,
        "deep_processed_pages": 0,
        "ocr_pages": 0,
        "text_layer_pages": 0,
        "skipped_pages": 0,
        "per_doc": {},
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
            ocr_page_list = ocr_metadata.get("ocr_pages", [])
            stats["ocr_pages"] = len(ocr_page_list)
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

        # Selection mode + selected count
        if selected_result and hasattr(selected_result, 'selected'):
            stats["selected_pages_count"] = len(selected_result.selected)
        if run_coverage:
            sel_mode = None
            if hasattr(run_coverage, 'selection_mode'):
                sel_mode = run_coverage.selection_mode
            elif isinstance(run_coverage, dict):
                sel_mode = run_coverage.get("selection_mode")
            stats["selection_mode"] = sel_mode

        # Table attempt/success from extraction_diagnostics
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


# =============================================================================
# File helpers
# =============================================================================

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
