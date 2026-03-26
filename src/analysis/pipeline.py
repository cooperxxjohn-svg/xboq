"""
Analysis Pipeline

Provides a streamlit-compatible interface to run the analysis pipeline
with progress callbacks for real-time UI updates.

Standalone helpers (dataclasses, PDF loaders, stats builders) live in
pipeline_helpers.py and are re-exported here for backward compatibility.
"""

import json
import logging
import sys
import traceback
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export helpers so all existing imports from pipeline.py keep working
# ---------------------------------------------------------------------------
from .pipeline_helpers import (  # noqa: F401 (re-exports)
    PROJECT_ROOT,
    AnalysisStage,
    AnalysisResult,
    load_pdf_pages,
    run_ocr_extraction,
    extract_text_from_pdf,
    _compute_run_compare,
    _check_guardrails,
    _detect_epc_mode,
    _compute_boq_stats,
    _build_requirements_by_trade,
    _build_processing_stats,
    save_uploaded_files,
    generate_project_id,
)

import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT))
_sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run_analysis_pipeline(
    input_files: List[Path],
    project_id: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
    run_mode: Optional[str] = None,
    boq_excel_paths: Optional[List[Path]] = None,
    llm_client=None,
    sub_callback: Optional[Callable[[str, str, str, int], None]] = None,
    tenant_id: Optional[str] = None,
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
        sub_callback: Optional Callback(agent_id, status, message, items) for
                      fine-grained agent-level UI updates (Agent Office panel).
                      status is one of: "working", "done", "error", "skipped".
        llm_client: Optional LLM client (openai.OpenAI or anthropic.Anthropic) for
                    enrichment. If None, enrichment uses template fallbacks.
        tenant_id: Optional Supabase org_id / tenant identifier.  When set,
                   the completed payload is persisted to Supabase via
                   src.auth.project_store.save_project().

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

    def fire_sub(agent_id: str, status: str, message: str = "", items: int = 0) -> None:
        """Fire a sub-agent status update to the Agent Office callback (no-op if not wired)."""
        if sub_callback:
            try:
                sub_callback(agent_id, status, message, items)
            except Exception as _sub_exc:
                logger.debug("Agent Office sub_callback error (non-critical): %s", _sub_exc)

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

        # Low-confidence flags accumulated during pipeline run
        _low_confidence_flags: list = []

        # Table coverage tracking
        _table_tracker = None
        try:
            from src.analysis.table_coverage import TableCoverageTracker
            _table_tracker = TableCoverageTracker()
        except Exception as _tct_err:
            logger.debug("TableCoverageTracker unavailable: %s", _tct_err)

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

        # T4-2: Merge DXF pages into page_index (if any .dxf files were passed)
        try:
            _dxf_files = [str(f) for f in input_files if str(f).lower().endswith(".dxf")]
            if _dxf_files and page_index_result is not None:
                from src.analysis.cad_page_index import build_cad_page_index, merge_into_page_index
                _cad_idx = build_cad_page_index(_dxf_files,
                                                page_offset=page_index_result.total_pages)
                _merged = merge_into_page_index(page_index_result.to_dict(), _cad_idx)
                _merged_pages = []
                for _p in _merged["pages"]:
                    if isinstance(_p, dict):
                        _fields = {k: v for k, v in _p.items()
                                   if k in IndexedPage.__dataclass_fields__}
                        _merged_pages.append(IndexedPage(**_fields))
                    else:
                        _merged_pages.append(_p)
                page_index_result = PageIndex(
                    pdf_name=_merged.get("pdf_name", page_index_result.pdf_name),
                    total_pages=_merged["total_pages"],
                    pages=_merged_pages,
                    counts_by_type=_merged.get("counts_by_type", {}),
                    counts_by_discipline=_merged.get("counts_by_discipline", {}),
                )
                logger.debug("Merged %d DXF page(s) into page index", len(_cad_idx["pages"]))
        except Exception as _dxf_exc:
            logger.debug("DXF merge skipped: %s", _dxf_exc)

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

            # ── Feed TableCoverageTracker from extraction_diagnostics ──
            if _table_tracker and hasattr(extraction_result, 'extraction_diagnostics'):
                _diag = extraction_result.extraction_diagnostics or {}
                _boq_diag = _diag.get("boq", {})
                _sched_diag = _diag.get("schedules", {})
                _methods_used = _diag.get("table_methods_used", {})
                # Record BOQ page attempts as aggregate entries
                _boq_attempted = _boq_diag.get("pages_attempted", 0)
                _boq_parsed = _boq_diag.get("pages_parsed", 0)
                for _pg_num in range(_boq_attempted):
                    _method = "pdfplumber"
                    if _methods_used:
                        # Use most-used method as representative
                        _method = max(_methods_used, key=lambda k: _methods_used[k]) if _methods_used else "pdfplumber"
                    _success = _pg_num < _boq_parsed
                    _table_tracker.record_attempt(
                        page=_pg_num,
                        method=_method,
                        success=_success,
                        n_rows=_boq_diag.get("items_extracted", 0) // max(_boq_parsed, 1) if _success else 0,
                        doc_type="boq",
                    )
                # Record schedule page attempts
                _sched_attempted = _sched_diag.get("pages_attempted", 0)
                _sched_parsed = _sched_diag.get("pages_parsed", 0)
                for _pg_num in range(_sched_attempted):
                    _success = _pg_num < _sched_parsed
                    _table_tracker.record_attempt(
                        page=_pg_num,
                        method=_method if _methods_used else "pdfplumber",
                        success=_success,
                        n_rows=_sched_diag.get("rows_extracted", 0) // max(_sched_parsed, 1) if _success else 0,
                        doc_type="schedule",
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
                logger.warning("Excel BOQ parsing failed: %s", _excel_err)

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
            logger.warning("Schedule-BOQ reconciliation failed: %s", _recon_err)
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
                logger.debug("Rate intelligence benchmarking skipped: %s", _ri_err)

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
            logger.warning("Item normalisation failed: %s", _norm_err)
            _contractual_raw = []

        # ── QTO module cascade (21 modules) ─────────────────────────────────
        # Extracted to src/analysis/pipeline_stages/qto_runner.py to reduce
        # pipeline.py size by ~890 lines.  All logic is preserved unchanged.
        from .pipeline_stages import run_qto_modules, QTOInputs
        _qto = run_qto_modules(QTOInputs(
            page_index_result   = page_index_result,
            all_page_texts      = all_page_texts,
            extraction_result   = extraction_result,
            spec_items_list     = _spec_items_list,
            stub_items_list     = _stub_items_list,
            recon               = _recon,
            fire_sub            = fire_sub,
            low_confidence_flags= _low_confidence_flags,
            tenant_id           = tenant_id,
            project_id          = project_id,
            input_files         = input_files,
            llm_client          = llm_client,
            primary_pdf_path    = locals().get("primary_pdf_path"),
        ))

        # ── Unpack QTO outputs ────────────────────────────────────────────
        _qto_rated_items      = _qto.qto_rated_items
        _qto_grand_total_inr  = _qto.qto_grand_total_inr
        _line_items_payload   = _qto.line_items_payload
        _dedup_stats          = _qto.dedup_stats
        _spec_needs_qty       = _qto.spec_needs_qty
        _spec_params_payload  = _qto.spec_params_payload
        _st_area_sqm          = _qto.st_area_sqm
        _st_floors            = _qto.st_floors
        _qto_paint_items      = _qto.qto_paint_items
        _qto_wp_items         = _qto.qto_wp_items
        _qto_dw_items         = _qto.qto_dw_items
        _qto_mep_items        = _qto.qto_mep_items
        _qto_sw_items         = _qto.qto_sw_items
        _qto_brickwork_items  = _qto.qto_brickwork_items
        _qto_plaster_items    = _qto.qto_plaster_items
        _qto_earthwork_items  = _qto.qto_earthwork_items
        _qto_flooring_items   = _qto.qto_flooring_items
        _qto_foundation_items = _qto.qto_foundation_items
        _qto_extdev_items     = _qto.qto_extdev_items
        _qto_prelims_items    = _qto.qto_prelims_items
        _qto_elv_items        = _qto.qto_elv_items
        _qto_wp_wet_area      = _qto.qto_wp_wet_area
        _qto_wp_roof_area     = _qto.qto_wp_roof_area
        _qto_paint_int_wall   = _qto.qto_paint_int_wall

        # ── Payload workspace: assign qto_summary and Excel artefacts ────
        payload = getattr(result, 'payload', {})
        payload["qto_summary"] = _qto.qto_summary_dict
        if _qto.qto_excel_bytes:
            payload["_excel_bytes"] = _qto.qto_excel_bytes
        if _qto.qto_rated_items:
            payload["spec_items"] = _qto.qto_rated_items

        # ── Bid Margin Analysis: tender value vs contractor cost ────
        try:
            from src.analysis.bid_margin import (
                compute_bid_margin as _compute_bid_margin,
                extract_nit_value as _extract_nit_value,
            )
            _boq_items_for_margin = (
                extraction_result.boq_items
                if extraction_result is not None
                else []
            )
            # Try to extract the NIT stated estimated cost (used when BOQ has no rates)
            _nit_value_inr = _extract_nit_value(payload.get("commercial_terms") or [])
            _bid_margin = _compute_bid_margin(
                _qto_rated_items,
                boq_items=_boq_items_for_margin,
                nit_value_inr=_nit_value_inr,
            )
            # If contractor cost is zero (rated items had no amount_inr), fall back to
            # qto_summary grand_total which is always computed from market rates × qty
            _qto_grand = payload.get("qto_summary", {}).get("grand_total_inr", 0) or 0
            if _bid_margin.contractor_cost_inr == 0 and _qto_grand > 0:
                import dataclasses as _dc
                _bid_margin = _dc.replace(
                    _bid_margin,
                    contractor_cost_inr=_qto_grand,
                    margin_inr=_bid_margin.tender_value_inr - _qto_grand,
                    margin_pct=(
                        (_bid_margin.tender_value_inr - _qto_grand)
                        / _bid_margin.tender_value_inr * 100.0
                        if _bid_margin.tender_value_inr > 0 else 0.0
                    ),
                )
            payload["bid_margin"] = {
                "tender_value_inr": _bid_margin.tender_value_inr,
                "contractor_cost_inr": _bid_margin.contractor_cost_inr,
                "margin_inr": _bid_margin.margin_inr,
                "margin_pct": round(_bid_margin.margin_pct, 1),
                "coverage_pct": round(_bid_margin.coverage_pct, 1),
                "reliable": _bid_margin.reliable,
                "note": _bid_margin.note,
                "nit_value_inr": _nit_value_inr or 0,
                "scope_coverage_pct": round(_bid_margin.scope_coverage_pct, 1),
                "by_trade": [
                    {
                        "trade": t.trade,
                        "tender_value_inr": t.tender_value_inr,
                        "contractor_cost_inr": t.contractor_cost_inr,
                        "margin_inr": t.margin_inr,
                        "margin_pct": round(t.margin_pct, 1),
                        "coverage_pct": round(t.coverage_pct, 1),
                    }
                    for t in _bid_margin.by_trade
                ],
            }
        except Exception as _bm_exc:
            import logging as _logging
            _logging.getLogger(__name__).debug("bid_margin failed: %s", _bm_exc)

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

        # BUG-12 FIX: plan_graph._extract_sheet_no() and _detect_scale() run on ALL
        # pages regardless of doc_type.  On NIT/BOQ documents:
        #   - Spec codes (M-25, E-350, IS-875, CO-2) match the sheet_no pattern on
        #     BOQ/conditions pages and inflate pages_with_sheet_no.
        #   - Standard citations like "ISO 8501-1:1988" produce "1:1988" which matched
        #     the old scale pattern (fixed in plan_graph.py BUG-12b) — but other
        #     contextual false positives may still exist on text pages.
        # Correction: recount both metrics using only non-text-doc-type pages so that
        # the is_drawing_set gate is not triggered by material codes in BOQ body text.
        #
        # BUG-13 FIX: Large composite NITs (473+ pages, e.g. IIM Rohtak, IITK Visitor
        # Hostel) contain hundreds of pages classified as "unknown" — pages whose text
        # didn't match any specific doc_type pattern (generic conditions, foreword, etc).
        # These "unknown" pages still carry spec codes (M-25, IS-875, ratio 1:4) that
        # match sheet_no/scale patterns, inflating counts even after BUG-12 fix.
        # Resolution: treat "unknown" the same as text pages — exclude from drawing
        # detection counts.  Genuine drawing pages are always classified as plan/section/
        # elevation/detail/rcp/site_plan; if OCR returned zero text the page is
        # unclassifiable and cannot be evidence of a drawing set.
        _TEXT_PLAN_DOC_TYPES = frozenset(
            {"boq", "conditions", "addendum", "spec", "notes", "legend", "unknown"}
        )
        if page_index_result is not None:
            _pi_type_by_idx = {
                ip.page_idx: ip.doc_type
                for ip in getattr(page_index_result, "pages", [])
            }
            _sheet_count = 0
            _scale_count = 0
            for _sheet in getattr(plan_graph, "sheets", []):
                _pg_type = _pi_type_by_idx.get(_sheet.page_index, "unknown")
                if _pg_type in _TEXT_PLAN_DOC_TYPES:
                    continue  # text page — don't count its false sheet_no/scale hits
                if _sheet.sheet_no is not None:
                    _sheet_count += 1
                if _sheet.detected.get("has_scale"):
                    _scale_count += 1
            plan_graph.pages_with_sheet_no = _sheet_count
            plan_graph.pages_with_scale = _scale_count

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
        fire_sub("rfi_generator", "working", "generating RFIs")

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
                    logger.info("Knowledge base: +%d RFIs (total: %d)", _added, len(rfi_list))
        except Exception as _kb_err:
            logger.debug("KB RFI extension skipped: %s", _kb_err)

        update_progress("rfi", f"Generated {len(rfi_list)} RFIs", 1.0)
        fire_sub("rfi_generator", "done", f"{len(rfi_list)} RFIs", len(rfi_list))
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
            "pages_with_sheet_no": getattr(plan_graph, 'pages_with_sheet_no', 0),
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
        pages_with_sheet_no = drawing_overview.get("pages_with_sheet_no", 0)
        total_pg = drawing_overview.get("pages_total", 0)

        # BUG-4 FIX (v2): use two complementary hard drawing signals.
        #
        # scale_pages  — pages with a drawing scale notation (1:100, 1"=10', NTS …).
        #                NOTE: the generic "1:N" scale pattern was narrowed to denom ≥ 10
        #                to avoid matching cement mix ratios (1:3, 1:6) in spec text.
        #
        # sheet_no_ratio — fraction of pages that carry an engineering sheet number
        #                (A-101, S-2.01 …).  Drawing sets have ≥ 40 % coverage;
        #                spec documents referencing drawings have < 10 % coverage.
        #
        # Soft signals (door/window tags, room names, schedule mentions) appear in
        # specification documents too and are not used as the gate criterion.
        sheet_no_ratio = pages_with_sheet_no / max(total_pg, 1)
        is_drawing_set_flag = (scale_pages >= 1) or (sheet_no_ratio >= 0.25)

        # Legacy composite kept for diagnostics / downstream code that reads it
        hard_signals = scale_pages + len(disciplines)
        soft_signals = schedules + doors + windows + rooms
        drawing_signals = hard_signals + soft_signals

        if not is_drawing_set_flag:
            is_drawing_set = False
            no_drawing_reason = (
                f"No confirmed construction drawing indicators in {total_pg} page(s): "
                f"scale notation found on {scale_pages} page(s) "
                f"(need ≥ 1) and only {pages_with_sheet_no}/{total_pg} pages "
                f"({sheet_no_ratio:.0%}) carry a sheet number (need ≥ 25 %).  "
                "Document appears to be a specification, report, or schedule only."
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
            # Keep any RFIs already assembled (e.g. spec/BOQ-only tenders still
            # produce valid checklist RFIs); only reset if genuinely none exist.
            if not rfis_payload:
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
                qto_context={
                    "total_area_sqm": _st_area_sqm,
                    "building_type": (
                        (_spec_params_payload.get("building_types") or ["default"])[0]
                        if _spec_params_payload else "default"
                    ),
                    "wp_wet_area_sqm": _qto_wp_wet_area,
                    "wp_roof_area_sqm": _qto_wp_roof_area,
                    "paint_int_wall_sqm": _qto_paint_int_wall,
                },
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
            # Sprint 38 + Q1/Q2: QTO module results (used by boq_from_drawings autofill)
            "painting_result":      {"line_items": _qto_paint_items}      if _qto_paint_items      else None,
            "waterproofing_result": {"line_items": _qto_wp_items}         if _qto_wp_items         else None,
            "dw_takeoff":           {"line_items": _qto_dw_items}         if _qto_dw_items         else None,
            "mep_qto":              {"line_items": _qto_mep_items}        if _qto_mep_items        else None,
            "sitework_result":      {"line_items": _qto_sw_items}         if _qto_sw_items         else None,
            "brickwork_result":     {"line_items": _qto_brickwork_items}  if _qto_brickwork_items  else None,
            "plaster_result":       {"line_items": _qto_plaster_items}    if _qto_plaster_items    else None,
            "earthwork_result":     {"line_items": _qto_earthwork_items}  if _qto_earthwork_items  else None,
            "flooring_result":      {"line_items": _qto_flooring_items}   if _qto_flooring_items   else None,
            "foundation_result":    {"line_items": _qto_foundation_items} if _qto_foundation_items else None,
            "extdev_result":        {"line_items": _qto_extdev_items}     if _qto_extdev_items     else None,
            "prelims_result":       {"line_items": _qto_prelims_items}    if _qto_prelims_items    else None,
            "elv_result":           {"line_items": _qto_elv_items}        if _qto_elv_items        else None,
            "items_needs_qty":      _spec_needs_qty                       if _spec_needs_qty       else None,
            # Spec-driven BOQ generation params (populated for EPC tenders with no BOQ tables)
            "spec_params": _spec_params_payload or None,
            # Sprint 20C: Estimating playbook (attached from session if available)
            "estimating_playbook": None,
            # Sprint 21C: BOQ source tracking (pdf or excel)
            "boq_source": _boq_source,
            # Sprint 20F: Extraction diagnostics (hybrid table extraction)
            "extraction_diagnostics": extraction_result.extraction_diagnostics if extraction_result else {},
            # QTO: Room-finish takeoff summary
            "qto_summary": payload.get("qto_summary", {}),
        }

        # ── Table extraction coverage summary ───────────────────────
        if _table_tracker and _table_tracker.attempt_count > 0:
            result.payload["table_extraction_coverage"] = _table_tracker.summary().to_dict()

        # ── Low-confidence flags accumulated during run ───────────
        if _low_confidence_flags:
            result.payload["_low_confidence_flags"] = _low_confidence_flags

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

        # Tier-1: Compute per-trade confidence scores
        try:
            from src.analysis.trade_confidence import compute_trade_confidence as _ctc
            _trade_conf = [s.to_dict() for s in _ctc(result.payload)]
            result.payload["trade_confidence"] = _trade_conf
        except Exception as _tc_exc:
            logger.warning("trade_confidence computation skipped: %s", _tc_exc)
            result.payload["trade_confidence"] = []

        # Tier-1: BOQ auto-fill from drawings (when no BOQ in tender)
        try:
            from src.analysis.boq_from_drawings import can_autofill as _can_af, autofill_boq as _af
            if _can_af(result.payload):
                _af_result = _af(result.payload)
                result.payload["boq_autofill"] = _af_result.to_dict()
                logger.info(
                    "boq_from_drawings: auto-filled %d items", len(_af_result.items)
                )
        except Exception as _af_exc:
            logger.warning("boq_from_drawings skipped: %s", _af_exc)

        # IS code compliance check
        try:
            from src.analysis.is_code_compliance import check_is_compliance, compliance_summary as _cs
            _is_violations = check_is_compliance(result.payload)
            result.payload["is_compliance"] = _cs(_is_violations)
            if _is_violations:
                logger.info("IS compliance: %d violations found", len(_is_violations))
        except Exception as _isc_err:
            logger.debug("IS compliance check skipped: %s", _isc_err)

        # Scope conflict detection
        try:
            from src.analysis.scope_conflict_detector import detect_scope_conflicts, conflict_summary as _cfs
            _scope_conflicts = detect_scope_conflicts(result.payload)
            result.payload["scope_conflicts"] = _cfs(_scope_conflicts)
            if _scope_conflicts:
                logger.info("Scope conflicts: %d found (%d high severity)",
                            len(_scope_conflicts),
                            sum(1 for c in _scope_conflicts if c.severity == "high"))
        except Exception as _scd_err:
            logger.debug("Scope conflict detection skipped: %s", _scd_err)

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
        fire_sub("vector_indexer", "working", "building vector index")
        try:
            from src.vectorstore import build_index_from_payload
            _vec_index = build_index_from_payload(result.payload)
            if _vec_index.size > 0:
                _vec_path = output_dir / "vector_index.json"
                _vec_index.save(str(_vec_path))
                result.files_generated.append(str(_vec_path))
                logger.info("Sprint 29: Vector index saved — %d chunks, %d vocab terms",
                            _vec_index.size, _vec_index.vocab_size)
            fire_sub("vector_indexer", "done",
                     f"{getattr(_vec_index, 'size', 0)} chunks")
        except Exception as _vec_exc:
            logger.warning("Sprint 29 vector index build failed (non-critical): %s", _vec_exc)
            fire_sub("vector_indexer", "error", str(_vec_exc)[:80])

        # ── Semantic Intelligence Layer (Sprint 40) ───────────────────────────
        # ChromaDB dense embeddings + gap analysis + bid synthesis
        # Gracefully skipped if chromadb / sentence-transformers not installed
        fire_sub("gap_analyzer",   "working", "analyzing gaps")
        fire_sub("cost_impact",    "working", "estimating cost impact")
        fire_sub("bid_synthesizer","working", "synthesizing bid intelligence")
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

            # Ensure domain KB is up to date (no-op if already built)
            try:
                from src.embeddings.kb_interface import get_kb as _get_kb
                _domain_kb = _get_kb()
                _domain_kb.build_if_stale(embedder=_intel_embedder)
            except Exception as _dkb_err:
                logger.debug("DomainKB build_if_stale skipped: %s", _dkb_err)
                _domain_kb = None

            # Build BM25 index alongside ChromaDB
            _bm25_index = None
            try:
                from src.embeddings.bm25_index import BM25Index
                _bm25_index = BM25Index()
                # Build from OCR text chunks that were indexed into ChromaDB
                _bm25_texts = []
                _ocr_cache = result.payload.get("ocr_text_cache") or {}
                for _page_key, _page_text in _ocr_cache.items():
                    if isinstance(_page_text, str) and _page_text.strip():
                        _bm25_texts.append(_page_text)
                # Also add BOQ items
                for _item in (result.payload.get("line_items") or []):
                    _desc = _item.get("description", "")
                    if _desc:
                        _bm25_texts.append(_desc)
                if _bm25_texts:
                    _bm25_index.build(_bm25_texts)
                    logger.debug("BM25 index built: %d documents", _bm25_index.doc_count)
            except Exception as _bm25_err:
                logger.debug("BM25 index build failed (non-critical): %s", _bm25_err)
                _bm25_index = None

            _intel_gaps = _analyze_gaps(
                result.payload, _intel_store, _intel_embedder, llm_client,
            )
            fire_sub("gap_analyzer", "done",
                     f"{len(_intel_gaps)} gaps", len(_intel_gaps))

            _intel_impacts = [
                _estimate_cost_impact(g, result.payload) for g in _intel_gaps
            ]
            fire_sub("cost_impact", "done",
                     f"{len(_intel_impacts)} estimates", len(_intel_impacts))

            _intel_synthesis = _synthesize_bid(
                result.payload, _intel_gaps, _intel_impacts, llm_client,
            )
            fire_sub("bid_synthesizer", "done",
                     f"readiness={_intel_synthesis.bid_readiness_score}")

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
            fire_sub("gap_analyzer",    "skipped", "chromadb not installed")
            fire_sub("cost_impact",     "skipped", "chromadb not installed")
            fire_sub("bid_synthesizer", "skipped", "chromadb not installed")
        except Exception as _intel_exc:
            logger.warning("Sprint 40 intelligence layer failed (non-critical): %s", _intel_exc)
            fire_sub("gap_analyzer",    "error", str(_intel_exc)[:80])
            fire_sub("cost_impact",     "error", str(_intel_exc)[:80])
            fire_sub("bid_synthesizer", "error", str(_intel_exc)[:80])

        # ── Tier 3: Tender benchmarks + award probability ─────────────────
        try:
            from src.analysis.tender_benchmarks import (
                record_tender as _record_tender,
                compare_tender_to_market as _compare_to_market,
                detect_building_type as _detect_bt,
            )
            _bt = _detect_bt(result.payload)
            _region = getattr(kwargs if "kwargs" in dir() else object(), "region", None) or "tier1"
            _bm = _compare_to_market(result.payload, _bt, _region)
            result.payload["benchmark_comparison"] = _bm
            if project_id and tenant_id:
                _record_tender(project_id, tenant_id or "local", result.payload, _bt, _region)
        except Exception as _bm_exc:
            logger.debug("tender_benchmarks skipped: %s", _bm_exc)

        try:
            from src.reasoning.award_predictor import AwardPredictor as _AwardPredictor
            _ap = _AwardPredictor()
            _ap_features = _ap.extract_features(result.payload)
            _ap_pred = _ap.predict(_ap_features)
            result.payload["award_probability"] = _ap_pred.probability_pct
            result.payload["award_prediction"] = _ap_pred.to_dict()
            # Also set on bid_synthesis if present
            if result.payload.get("bid_synthesis"):
                result.payload["bid_synthesis"]["award_probability"] = _ap_pred.probability_pct
        except Exception as _ap_exc:
            logger.debug("award_predictor skipped: %s", _ap_exc)

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

        # Validate payload before returning
        try:
            from src.analysis.payload_validator import validate_payload as _validate_payload
            _pv_warnings = _validate_payload(result.payload)
            _pv_errors = [w for w in _pv_warnings if w.severity == "error"]
            if _pv_errors:
                logger.warning("Payload validation: %d error(s) — %s",
                               len(_pv_errors),
                               "; ".join(w.message for w in _pv_errors[:3]))
            result.payload["_validation_warnings"] = [
                {"field": w.field, "severity": w.severity, "message": w.message}
                for w in _pv_warnings
            ]
        except Exception as _val_err:
            logger.debug("Payload validation skipped: %s", _val_err)

        # Success
        result.success = True
        result.duration_sec = (datetime.now() - start_time).total_seconds()
        logger.info("Analysis complete: %s (%.2fs)", project_id, result.duration_sec)

        # Persist to Supabase when tenant_id is provided
        if tenant_id and result.payload:
            try:
                from src.auth.project_store import save_project
                from src.auth.supabase_client import is_configured
                if is_configured():
                    payload = result.payload
                    summary = {
                        "line_items_count": len(payload.get("line_items", [])),
                        "rfi_count": len(payload.get("rfis", [])),
                        "boq_items_count": len(payload.get("boq_items", [])),
                        "qa_score": (payload.get("qa_score") or {}).get("total_score"),
                        "duration_sec": result.duration_sec,
                    }
                    save_project(
                        org_id=tenant_id,
                        user_id=tenant_id,
                        filename=str(input_files[0].name) if input_files else project_id,
                        summary=summary,
                        payload=payload,
                    )
                    logger.info("Pipeline result persisted to Supabase for tenant %s", tenant_id)
            except Exception as _sb_exc:
                logger.warning("Supabase persistence skipped: %s", _sb_exc)

        # BOQ versioning — save snapshot for revision tracking
        try:
            from src.analysis.boq_versioning import BOQVersionStore as _BOQVersionStore
            _boq_vs = _BOQVersionStore()
            _snapshot_project_id = tenant_id or project_id or "local"
            _snap_run_id = _boq_vs.save_snapshot(_snapshot_project_id, result.payload)
            result.payload["boq_snapshot_run_id"] = _snap_run_id
            logger.info("BOQ snapshot saved: run_id=%s", _snap_run_id)
        except Exception as _vs_err:
            logger.debug("BOQ versioning skipped: %s", _vs_err)

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


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Ensure project root on sys.path so relative imports resolve when run as script
    import sys
    _proj_root = str(Path(__file__).resolve().parent.parent.parent)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    # Re-run as module so package context exists (fixes relative import errors)
    if not __package__:
        import runpy
        runpy.run_module("src.analysis.pipeline", run_name="__main__", alter_sys=True)
        raise SystemExit(0)

    # Test with sample PDF
    if len(sys.argv) < 2:
        print("Usage: python src/analysis/pipeline.py <pdf_path>")
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
