"""
CAD Page Index — T4-2.

Builds a PageIndex-compatible dict from one or more DXF files.
Assigns page numbers sequentially after a given offset so DXF pages
can be merged into the main pipeline page_index.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


def build_cad_page_index(dxf_paths: List[str], page_offset: int = 0) -> dict:
    """
    Parse each DXF file and return a PageIndex-compatible dict.

    Args:
        dxf_paths:   List of .dxf file paths.
        page_offset: Start numbering from this index (appended after PDF pages).

    Returns:
        dict with keys: pdf_name, total_pages, pages, counts_by_type,
                        counts_by_discipline, indexing_time_s
    """
    from src.adapters.cad_adapter import parse_dxf
    import time

    t0 = time.perf_counter()
    all_pages: List[dict] = []
    current_idx = page_offset

    for path in dxf_paths:
        pages = parse_dxf(path)
        for p in pages:
            p = dict(p)
            p["page_idx"] = current_idx
            current_idx += 1
            all_pages.append(p)

    counts_by_type: dict = {}
    counts_by_discipline: dict = {}
    for p in all_pages:
        dt = p.get("doc_type", "unknown")
        ds = p.get("discipline", "unknown")
        counts_by_type[dt] = counts_by_type.get(dt, 0) + 1
        counts_by_discipline[ds] = counts_by_discipline.get(ds, 0) + 1

    elapsed = time.perf_counter() - t0

    return {
        "pdf_name": "cad_files",
        "total_pages": len(all_pages),
        "pages": all_pages,
        "counts_by_type": counts_by_type,
        "counts_by_discipline": counts_by_discipline,
        "indexing_time_s": round(elapsed, 3),
        "source": "dxf",
    }


def merge_into_page_index(base_index: dict, cad_index: dict) -> dict:
    """
    Append cad_index pages into base_index (mutates and returns base_index).
    Used in pipeline.py to merge DXF pages after PDF indexing.
    """
    base_pages = base_index.get("pages") or []
    cad_pages = cad_index.get("pages") or []

    # Renumber cad pages after PDF pages
    offset = base_index.get("total_pages", len(base_pages))
    renumbered = []
    for i, p in enumerate(cad_pages):
        p2 = dict(p)
        p2["page_idx"] = offset + i
        renumbered.append(p2)

    base_index["pages"] = base_pages + renumbered
    base_index["total_pages"] = len(base_index["pages"])

    # Merge counts
    for key in ("counts_by_type", "counts_by_discipline"):
        base_counts = base_index.get(key) or {}
        for k, v in (cad_index.get(key) or {}).items():
            base_counts[k] = base_counts.get(k, 0) + v
        base_index[key] = base_counts

    return base_index
