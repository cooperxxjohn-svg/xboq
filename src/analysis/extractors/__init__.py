"""
Extractors Package — Specialized text extraction per page type.

Routes OCR'd pages to the correct extractor based on doc_type from PageIndex.
Sprint 20F: Passes pdf_path and page_meta to extractors for table_router.
            Collects extraction_diagnostics for pipeline reporting.
"""

import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..page_index import PageIndex, IndexedPage

from .extract_notes import extract_requirements
from .extract_schedule_tables import extract_schedule_rows
from .extract_boq import extract_boq_items
from .extract_drawings_minimal import extract_drawing_callouts
from .extract_commercial_terms import extract_commercial_terms
from .extract_spec_items import extract_spec_items

# Page-level result cache (lazy-initialised on first use)
_page_cache = None

def _get_page_cache():
    """Return the shared PageCache instance, or None if cache is disabled."""
    if os.environ.get("XBOQ_DISABLE_CACHE", "").strip() == "1":
        return None
    global _page_cache
    if _page_cache is None:
        try:
            from ..page_cache import PageCache
            _page_cache = PageCache()
        except Exception:
            pass
    return _page_cache


# =============================================================================
# RESULT MODEL
# =============================================================================

@dataclass
class ExtractionResult:
    """Combined output of all extractors."""
    requirements: List[dict] = field(default_factory=list)
    schedules: List[dict] = field(default_factory=list)
    boq_items: List[dict] = field(default_factory=list)
    callouts: List[dict] = field(default_factory=list)
    commercial_terms: List[dict] = field(default_factory=list)
    spec_items: List[dict] = field(default_factory=list)   # numbered spec/notes clauses
    extraction_times: Dict[str, float] = field(default_factory=dict)
    pages_processed: int = 0
    pages_by_extractor: Dict[str, int] = field(default_factory=dict)
    # Sprint 20F: Per-extractor diagnostics
    extraction_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "requirements": self.requirements,
            "schedules": self.schedules,
            "boq_items": self.boq_items,
            "callouts": self.callouts,
            "commercial_terms": self.commercial_terms,
            "spec_items": self.spec_items,
            "extraction_times": self.extraction_times,
            "pages_processed": self.pages_processed,
            "pages_by_extractor": self.pages_by_extractor,
            "counts": {
                "requirements": len(self.requirements),
                "schedules": len(self.schedules),
                "boq_items": len(self.boq_items),
                "callouts": len(self.callouts),
                "commercial_terms": len(self.commercial_terms),
                "spec_items": len(self.spec_items),
            },
            "extraction_diagnostics": self.extraction_diagnostics,
        }


# =============================================================================
# ROUTING TABLE
# =============================================================================

# doc_type -> extractor function name
EXTRACTOR_ROUTING = {
    "notes":      "notes",
    "legend":     "notes",
    "spec":       "notes",
    "conditions": "notes",
    "addendum":   "notes",
    "schedule":   "schedule",
    "boq":        "boq",
    "plan":       "drawings",
    "detail":     "drawings",
    "section":    "drawings",
    "elevation":  "drawings",
}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_extractors(
    ocr_text_by_page: Dict[int, str],
    page_index: PageIndex,
    pdf_path: Optional[str] = None,
    enable_debug: bool = False,
) -> ExtractionResult:
    """
    Route each OCR'd page to the correct extractor based on its doc_type.

    Sprint 20F: Now passes pdf_path and page_meta to BOQ/schedule extractors
    for table_router integration.  Collects extraction diagnostics.

    Args:
        ocr_text_by_page: Dict of page_idx -> OCR text (only for selected pages).
        page_index: PageIndex with classification for every page.
        pdf_path: Optional path to PDF for table_router native extraction.
        enable_debug: If True, extractors attach per-page debug payloads.

    Returns:
        ExtractionResult with all extracted data merged.
    """
    result = ExtractionResult()
    extractor_counts: Dict[str, int] = defaultdict(int)
    extractor_times: Dict[str, float] = defaultdict(float)

    # Sprint 20F: Diagnostics accumulators
    boq_pages_attempted = 0
    boq_pages_parsed = 0
    schedule_pages_attempted = 0
    schedule_pages_parsed = 0
    spec_pages_attempted = 0
    spec_items_extracted = 0
    table_methods_used: Dict[str, int] = defaultdict(int)

    # Build lookup: page_idx -> IndexedPage
    idx_map: Dict[int, IndexedPage] = {p.page_idx: p for p in page_index.pages}

    _cache = _get_page_cache()

    for page_idx, text in ocr_text_by_page.items():
        if not text or not text.strip():
            continue

        indexed = idx_map.get(page_idx)
        doc_type = indexed.doc_type if indexed else "unknown"
        sheet_id = indexed.sheet_id if indexed else None
        extractor_name = EXTRACTOR_ROUTING.get(doc_type)

        if not extractor_name:
            continue

        t0 = time.perf_counter()

        # Build page_meta for table_router-enabled extractors
        page_meta = {
            "page_number": page_idx,
            "doc_type": doc_type,
            "discipline": indexed.discipline if indexed else None,
            "has_text_layer": indexed.has_text_layer if indexed else True,
        }

        # ── Page-level cache lookup ──────────────────────────────────────
        # Cache key incorporates doc_type so the same text classified
        # differently on re-run still gets a fresh extraction.
        _cached_page = None
        if _cache is not None:
            try:
                _cached_page = _cache.get(text, doc_type)
            except Exception:
                _cached_page = None

        if _cached_page is not None:
            # Cache hit: unpack pre-computed results and skip LLM calls.
            result.requirements.extend(_cached_page.get("requirements", []))
            result.commercial_terms.extend(_cached_page.get("commercial_terms", []))
            result.spec_items.extend(_cached_page.get("spec_items", []))
            result.schedules.extend(_cached_page.get("schedules", []))
            result.boq_items.extend(_cached_page.get("boq_items", []))
            result.callouts.extend(_cached_page.get("callouts", []))

            # Update diagnostic counters from cached metadata
            _cd = _cached_page.get("_diagnostics", {})
            if extractor_name == "notes":
                if _cd.get("spec_attempted"):
                    spec_pages_attempted += 1
                spec_items_extracted += len(_cached_page.get("spec_items", []))
            elif extractor_name == "schedule":
                schedule_pages_attempted += 1
                if _cached_page.get("schedules"):
                    schedule_pages_parsed += 1
            elif extractor_name == "boq":
                boq_pages_attempted += 1
                if _cached_page.get("boq_items"):
                    boq_pages_parsed += 1

            elapsed = time.perf_counter() - t0
            extractor_counts[extractor_name] += 1
            extractor_times[extractor_name] += elapsed
            result.pages_processed += 1
            continue

        # ── Cache miss: run live extraction ─────────────────────────────
        _page_requirements: List[dict] = []
        _page_commercial: List[dict] = []
        _page_spec: List[dict] = []
        _page_schedules: List[dict] = []
        _page_boq: List[dict] = []
        _page_callouts: List[dict] = []
        _diag: dict = {}

        if extractor_name == "notes":
            items = extract_requirements(text, page_idx, sheet_id, doc_type)
            _page_requirements = items
            result.requirements.extend(items)
            # Also extract commercial terms from same pages
            t_com = time.perf_counter()
            com_items = extract_commercial_terms(text, page_idx, sheet_id, doc_type)
            _page_commercial = com_items
            result.commercial_terms.extend(com_items)
            extractor_times["commercial"] += time.perf_counter() - t_com
            # Extract numbered priceable spec items (for TENDER/MIXED line-item assembly)
            if doc_type in ("spec", "notes", "conditions", "addendum", "legend"):
                spec_pages_attempted += 1
                _diag["spec_attempted"] = True
                t_spec = time.perf_counter()
                s_items = extract_spec_items(text, page_idx, doc_type)
                _page_spec = s_items
                result.spec_items.extend(s_items)
                spec_items_extracted += len(s_items)
                extractor_times["spec_items"] = (
                    extractor_times.get("spec_items", 0.0)
                    + time.perf_counter() - t_spec
                )

        elif extractor_name == "schedule":
            schedule_pages_attempted += 1
            items = extract_schedule_rows(
                text, page_idx, sheet_id,
                pdf_path=pdf_path,
                page_meta=page_meta,
                enable_debug=enable_debug,
            )
            # Pop debug keys before caching so they don't pollute stored results
            _clean_items = []
            for it in items:
                debug = it.pop("_schedule_page_debug", None)
                if debug:
                    m = debug.get("method_used", "regex")
                    table_methods_used[m] += 1
                _clean_items.append(it)
            _page_schedules = _clean_items
            result.schedules.extend(_clean_items)
            if _clean_items:
                schedule_pages_parsed += 1

        elif extractor_name == "boq":
            boq_pages_attempted += 1
            items = extract_boq_items(
                text, page_idx,
                pdf_path=pdf_path,
                page_meta=page_meta,
                enable_debug=enable_debug,
            )
            _clean_items = []
            for it in items:
                debug = it.pop("_boq_page_debug", None)
                if debug:
                    m = debug.get("method_used", "regex")
                    table_methods_used[m] += 1
                _clean_items.append(it)
            _page_boq = _clean_items
            result.boq_items.extend(_clean_items)
            if _clean_items:
                boq_pages_parsed += 1

        elif extractor_name == "drawings":
            items = extract_drawing_callouts(text, page_idx, sheet_id, doc_type=doc_type)
            _page_callouts = items
            result.callouts.extend(items)

        elapsed = time.perf_counter() - t0
        extractor_counts[extractor_name] += 1
        extractor_times[extractor_name] += elapsed
        result.pages_processed += 1

        # ── Cache store (fire-and-forget, never raises) ──────────────────
        if _cache is not None:
            try:
                _cache.put(text, doc_type, {
                    "requirements":    _page_requirements,
                    "commercial_terms": _page_commercial,
                    "spec_items":      _page_spec,
                    "schedules":       _page_schedules,
                    "boq_items":       _page_boq,
                    "callouts":        _page_callouts,
                    "_diagnostics":    _diag,
                })
            except Exception:
                pass

    result.pages_by_extractor = dict(extractor_counts)
    result.extraction_times = {k: round(v, 4) for k, v in extractor_times.items()}

    # ─── Sprint 24 Phase 2: Secondary schedule scan ──────────────────────
    # When zero pages are classified as "schedule", schedule data may be
    # embedded in spec or drawing pages. Scan those pages for schedule-like
    # content (mark patterns, header rows) and run the schedule extractor
    # on any pages that look like they contain schedule tables.
    schedule_secondary_attempted = 0
    schedule_secondary_parsed = 0
    if schedule_pages_attempted == 0:
        _schedule_kw_re = re.compile(
            r'door\s+schedule|window\s+schedule|finish\s+schedule|'
            r'schedule\s+of\s+doors|schedule\s+of\s+windows|'
            r'room\s+finish|fixture\s+schedule|hardware\s+schedule',
            re.IGNORECASE,
        )
        _mark_re = re.compile(r'\b[DW]-?\d{1,3}[A-Z]?\b')

        for page_idx, text in ocr_text_by_page.items():
            if not text or not text.strip():
                continue
            indexed = idx_map.get(page_idx)
            if not indexed:
                continue
            # Only scan spec, plan, detail, notes pages
            if indexed.doc_type not in ("spec", "plan", "detail", "notes", "conditions"):
                continue
            # Quick content check: schedule keywords OR 2+ mark patterns
            has_schedule_kw = bool(_schedule_kw_re.search(text))
            mark_hits = len(_mark_re.findall(text))
            if not has_schedule_kw and mark_hits < 2:
                continue

            schedule_secondary_attempted += 1
            t_sched = time.perf_counter()
            page_meta = {
                "page_number": page_idx,
                "doc_type": indexed.doc_type,
                "discipline": indexed.discipline,
                "has_text_layer": indexed.has_text_layer,
            }

            # Cache lookup for secondary schedule scan
            # Use a distinct doc_type key so it doesn't collide with the primary pass.
            _sec_cache_key = f"{indexed.doc_type}__schedule_secondary"
            _sec_cached = None
            if _cache is not None:
                try:
                    _sec_cached = _cache.get(text, _sec_cache_key)
                except Exception:
                    _sec_cached = None

            if _sec_cached is not None:
                items = _sec_cached.get("schedules", [])
            else:
                items = extract_schedule_rows(
                    text, page_idx, indexed.sheet_id,
                    pdf_path=pdf_path,
                    page_meta=page_meta,
                    enable_debug=enable_debug,
                )
                # Pop debug keys before caching
                for it in items:
                    it.pop("_schedule_page_debug", None)
                # Store in cache (fire-and-forget)
                if _cache is not None:
                    try:
                        _cache.put(text, _sec_cache_key, {"schedules": items})
                    except Exception:
                        pass

            if items:
                # Deduplicate against any marks already found
                existing_marks = {r["mark"] for r in result.schedules}
                new_items = [it for it in items if it["mark"] not in existing_marks]
                if new_items:
                    for it in new_items:
                        it["_secondary_scan"] = True
                    result.schedules.extend(new_items)
                    schedule_secondary_parsed += 1
            extractor_times["schedule"] = extractor_times.get("schedule", 0) + (time.perf_counter() - t_sched)

    # Sprint 20F: Populate extraction diagnostics
    result.extraction_diagnostics = {
        "boq": {
            "pages_attempted": boq_pages_attempted,
            "pages_parsed": boq_pages_parsed,
            "items_extracted": len(result.boq_items),
        },
        "schedules": {
            "pages_attempted": schedule_pages_attempted + schedule_secondary_attempted,
            "pages_parsed": schedule_pages_parsed + schedule_secondary_parsed,
            "rows_extracted": len(result.schedules),
            "secondary_scan_pages": schedule_secondary_attempted,
            "secondary_scan_rows": sum(1 for s in result.schedules if s.get("_secondary_scan")),
        },
        "spec_items": {
            "pages_attempted": spec_pages_attempted,
            "items_extracted": spec_items_extracted,
        },
        "table_methods_used": dict(table_methods_used),
    }

    return result
