"""
Extractors Package — Specialized text extraction per page type.

Routes OCR'd pages to the correct extractor based on doc_type from PageIndex.
Sprint 20F: Passes pdf_path and page_meta to extractors for table_router.
            Collects extraction_diagnostics for pipeline reporting.
"""

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
            "extraction_times": self.extraction_times,
            "pages_processed": self.pages_processed,
            "pages_by_extractor": self.pages_by_extractor,
            "counts": {
                "requirements": len(self.requirements),
                "schedules": len(self.schedules),
                "boq_items": len(self.boq_items),
                "callouts": len(self.callouts),
                "commercial_terms": len(self.commercial_terms),
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
    table_methods_used: Dict[str, int] = defaultdict(int)

    # Build lookup: page_idx -> IndexedPage
    idx_map: Dict[int, IndexedPage] = {p.page_idx: p for p in page_index.pages}

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

        if extractor_name == "notes":
            items = extract_requirements(text, page_idx, sheet_id, doc_type)
            result.requirements.extend(items)
            # Also extract commercial terms from same pages
            t_com = time.perf_counter()
            com_items = extract_commercial_terms(text, page_idx, sheet_id, doc_type)
            result.commercial_terms.extend(com_items)
            extractor_times["commercial"] += time.perf_counter() - t_com

        elif extractor_name == "schedule":
            schedule_pages_attempted += 1
            items = extract_schedule_rows(
                text, page_idx, sheet_id,
                pdf_path=pdf_path,
                page_meta=page_meta,
                enable_debug=enable_debug,
            )
            result.schedules.extend(items)
            if items:
                schedule_pages_parsed += 1
                # Track method used from debug info
                for it in items:
                    debug = it.pop("_schedule_page_debug", None)
                    if debug:
                        m = debug.get("method_used", "regex")
                        table_methods_used[m] += 1
                        break  # One per page

        elif extractor_name == "boq":
            boq_pages_attempted += 1
            items = extract_boq_items(
                text, page_idx,
                pdf_path=pdf_path,
                page_meta=page_meta,
                enable_debug=enable_debug,
            )
            result.boq_items.extend(items)
            if items:
                boq_pages_parsed += 1
                for it in items:
                    debug = it.pop("_boq_page_debug", None)
                    if debug:
                        m = debug.get("method_used", "regex")
                        table_methods_used[m] += 1
                        break

        elif extractor_name == "drawings":
            items = extract_drawing_callouts(text, page_idx, sheet_id)
            result.callouts.extend(items)

        elapsed = time.perf_counter() - t0
        extractor_counts[extractor_name] += 1
        extractor_times[extractor_name] += elapsed
        result.pages_processed += 1

    result.pages_by_extractor = dict(extractor_counts)
    result.extraction_times = {k: round(v, 4) for k, v in extractor_times.items()}

    # Sprint 20F: Populate extraction diagnostics
    result.extraction_diagnostics = {
        "boq": {
            "pages_attempted": boq_pages_attempted,
            "pages_parsed": boq_pages_parsed,
            "items_extracted": len(result.boq_items),
        },
        "schedules": {
            "pages_attempted": schedule_pages_attempted,
            "pages_parsed": schedule_pages_parsed,
            "rows_extracted": len(result.schedules),
        },
        "table_methods_used": dict(table_methods_used),
    }

    return result
