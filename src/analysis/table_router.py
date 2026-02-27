"""
Table Router — Hybrid table extraction with deterministic fallback chain.

Sprint 20F: Unified entry point for extracting table-like content from
BOQ / schedule pages using multiple backends with graceful degradation.

Fallback order (deterministic):
  1. pdfplumber  (text PDFs, robust table detection)
  2. camelot lattice  (text PDFs, grid-lined tables)
  3. camelot stream  (text PDFs, whitespace-aligned tables)
  4. OCR row reconstruction  (scanned pages, heuristic column splitting)
  5. "none" — all methods failed

All optional dependencies are imported lazily.  If a library is unavailable
the method is skipped and recorded in diagnostics, never crashes.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT MODEL
# =============================================================================

@dataclass
class TableExtractionResult:
    """Result of table extraction from a single page."""
    method_used: str = "none"        # e.g. "pdfplumber", "camelot_lattice", "ocr_row_reconstruct"
    rows: List[Any] = field(default_factory=list)          # list[dict] or list[list[str]]
    headers: Optional[List[str]] = None
    confidence: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method_used": self.method_used,
            "row_count": len(self.rows),
            "headers": self.headers,
            "confidence": round(self.confidence, 3),
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        }


# =============================================================================
# SAFE OPTIONAL IMPORTS
# =============================================================================

def _try_import_pdfplumber():
    """Lazily import pdfplumber, return module or None."""
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        return None


def _try_import_camelot():
    """Lazily import camelot, return module or None."""
    try:
        import camelot
        return camelot
    except ImportError:
        return None
    except Exception:
        # camelot may fail if ghostscript is missing
        return None


# =============================================================================
# METHOD 1: PDFPLUMBER
# =============================================================================

def _extract_via_pdfplumber(
    pdf_path: str,
    page_number: int,  # 0-indexed
) -> Optional[Dict[str, Any]]:
    """Extract tables via pdfplumber.  Returns dict with rows/headers or None."""
    pdfplumber = _try_import_pdfplumber()
    if pdfplumber is None:
        return None

    t0 = time.perf_counter()
    try:
        pdf = pdfplumber.open(pdf_path)
        if page_number >= len(pdf.pages):
            pdf.close()
            return None
        page = pdf.pages[page_number]
        tables = page.extract_tables()
        pdf.close()

        if not tables:
            return {
                "rows": [],
                "headers": None,
                "time_s": round(time.perf_counter() - t0, 4),
                "table_count": 0,
            }

        # Use largest table (most rows)
        best = max(tables, key=len)

        # First row is likely header
        headers = None
        data_rows = best
        if len(best) > 1:
            first_row = best[0]
            # Header heuristic: mostly non-numeric short strings
            if first_row and all(
                isinstance(c, str) and len(c) < 60
                for c in first_row if c
            ):
                headers = [str(c).strip() if c else "" for c in first_row]
                data_rows = best[1:]

        rows = []
        for row in data_rows:
            cells = [str(c).strip() if c else "" for c in row]
            if any(cells):
                rows.append(cells)

        return {
            "rows": rows,
            "headers": headers,
            "time_s": round(time.perf_counter() - t0, 4),
            "table_count": len(tables),
        }
    except Exception as e:
        return {"error": str(e), "time_s": round(time.perf_counter() - t0, 4)}


# =============================================================================
# METHOD 2/3: CAMELOT (lattice + stream)
# =============================================================================

def _extract_via_camelot(
    pdf_path: str,
    page_number: int,  # 0-indexed
    flavor: str = "lattice",
) -> Optional[Dict[str, Any]]:
    """Extract tables via camelot lattice or stream.  Returns dict or None."""
    camelot = _try_import_camelot()
    if camelot is None:
        return None

    t0 = time.perf_counter()
    try:
        # camelot uses 1-indexed pages
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_number + 1),
            flavor=flavor,
        )
        if not tables or len(tables) == 0:
            return {
                "rows": [],
                "headers": None,
                "time_s": round(time.perf_counter() - t0, 4),
                "table_count": 0,
            }

        # Pick table with best accuracy (camelot scoring)
        best_table = max(tables, key=lambda t: t.accuracy if hasattr(t, 'accuracy') else 0)
        df = best_table.df

        headers = [str(c).strip() for c in df.iloc[0]] if len(df) > 1 else None
        data_start = 1 if headers else 0
        rows = []
        for _, row in df.iloc[data_start:].iterrows():
            cells = [str(c).strip() for c in row]
            if any(cells):
                rows.append(cells)

        accuracy = best_table.accuracy if hasattr(best_table, 'accuracy') else 0

        return {
            "rows": rows,
            "headers": headers,
            "time_s": round(time.perf_counter() - t0, 4),
            "table_count": len(tables),
            "accuracy": accuracy,
        }
    except Exception as e:
        return {"error": str(e), "time_s": round(time.perf_counter() - t0, 4)}


# =============================================================================
# METHOD 4: OCR ROW RECONSTRUCTION
# =============================================================================

# Common column-split pattern: 2+ spaces or tab
_COL_SPLIT = re.compile(r'\s{2,}|\t')

# Heuristic: lines that look like table rows (have ≥3 cell-like segments)
_NUM_PATTERN = re.compile(r'\d+(?:\.\d+)?')


def _reconstruct_rows_from_ocr(
    ocr_text: str,
) -> Dict[str, Any]:
    """Reconstruct table rows from OCR text using whitespace splitting.

    Works best for BOQ/schedule pages where columns are separated by
    multiple spaces.
    """
    t0 = time.perf_counter()
    if not ocr_text or not ocr_text.strip():
        return {
            "rows": [],
            "headers": None,
            "time_s": round(time.perf_counter() - t0, 4),
        }

    lines = ocr_text.split('\n')
    candidate_rows = []
    header_candidates = []

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue

        parts = _COL_SPLIT.split(stripped)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 2:
            continue

        # Classify: header vs data
        # Headers: mostly alpha, short cells, ≥3 columns
        alpha_ratio = sum(1 for p in parts if p.replace(' ', '').isalpha()) / max(len(parts), 1)

        if alpha_ratio > 0.6 and len(parts) >= 3 and not header_candidates:
            header_candidates.append(parts)
        else:
            candidate_rows.append(parts)

    headers = header_candidates[0] if header_candidates else None

    return {
        "rows": candidate_rows,
        "headers": headers,
        "time_s": round(time.perf_counter() - t0, 4),
    }


# =============================================================================
# MAIN ROUTER
# =============================================================================

def extract_table_rows_from_page(
    page_input: Optional[str] = None,
    page_meta: Optional[Dict[str, Any]] = None,
    ocr_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> TableExtractionResult:
    """
    Unified table extraction with deterministic fallback chain.

    Args:
        page_input: Path to PDF file (str or Path-like).  None for OCR-only path.
        page_meta: Dict with at least 'page_number' (0-indexed), optionally
                   'doc_type', 'discipline', 'has_text_layer'.
        ocr_text: Pre-extracted OCR text for the page (used for OCR path).
        config: Optional config dict.  Keys:
                - 'enable_debug': bool (default False)
                - 'skip_methods': list of method names to skip

    Returns:
        TableExtractionResult with best extraction and full diagnostics.
    """
    config = config or {}
    page_meta = page_meta or {}
    skip_methods = set(config.get("skip_methods", []))

    page_number = page_meta.get("page_number", 0)
    has_text_layer = page_meta.get("has_text_layer", True)

    result = TableExtractionResult()
    result.diagnostics = {
        "page_number": page_number,
        "doc_type": page_meta.get("doc_type"),
        "discipline": page_meta.get("discipline"),
        "has_text_layer": has_text_layer,
        "methods_attempted": [],
        "methods_skipped": [],
        "candidate_row_counts": {},
        "parse_times": {},
        "failure_reasons": {},
        "selection_reason": "",
    }

    best_rows = []
    best_headers = None
    best_method = "none"
    best_confidence = 0.0

    # ── TEXT PDF PATH (methods 1-3) ──────────────────────────────────────
    if page_input and has_text_layer:
        pdf_path = str(page_input)

        # Method 1: pdfplumber
        if "pdfplumber" not in skip_methods:
            result.diagnostics["methods_attempted"].append("pdfplumber")
            out = _extract_via_pdfplumber(pdf_path, page_number)
            if out is None:
                result.diagnostics["methods_skipped"].append("pdfplumber")
                result.diagnostics["failure_reasons"]["pdfplumber"] = "import_unavailable"
                result.warnings.append("pdfplumber not available")
            elif "error" in out:
                result.diagnostics["failure_reasons"]["pdfplumber"] = out["error"]
                result.diagnostics["parse_times"]["pdfplumber"] = out.get("time_s", 0)
            else:
                rows = out.get("rows", [])
                result.diagnostics["candidate_row_counts"]["pdfplumber"] = len(rows)
                result.diagnostics["parse_times"]["pdfplumber"] = out.get("time_s", 0)
                if rows and len(rows) > len(best_rows):
                    best_rows = rows
                    best_headers = out.get("headers")
                    best_method = "pdfplumber"
                    best_confidence = min(0.85, 0.5 + 0.02 * len(rows))
        else:
            result.diagnostics["methods_skipped"].append("pdfplumber")

        # Method 2: camelot lattice
        if "camelot_lattice" not in skip_methods:
            result.diagnostics["methods_attempted"].append("camelot_lattice")
            out = _extract_via_camelot(pdf_path, page_number, flavor="lattice")
            if out is None:
                result.diagnostics["methods_skipped"].append("camelot_lattice")
                result.diagnostics["failure_reasons"]["camelot_lattice"] = "import_unavailable"
                if "camelot not available" not in [w for w in result.warnings]:
                    result.warnings.append("camelot not available (check ghostscript)")
            elif "error" in out:
                result.diagnostics["failure_reasons"]["camelot_lattice"] = out["error"]
                result.diagnostics["parse_times"]["camelot_lattice"] = out.get("time_s", 0)
            else:
                rows = out.get("rows", [])
                result.diagnostics["candidate_row_counts"]["camelot_lattice"] = len(rows)
                result.diagnostics["parse_times"]["camelot_lattice"] = out.get("time_s", 0)
                accuracy = out.get("accuracy", 0)
                conf = min(0.9, 0.5 + accuracy / 200)
                if rows and (len(rows) > len(best_rows) or conf > best_confidence):
                    best_rows = rows
                    best_headers = out.get("headers")
                    best_method = "camelot_lattice"
                    best_confidence = conf
        else:
            result.diagnostics["methods_skipped"].append("camelot_lattice")

        # Method 3: camelot stream
        if "camelot_stream" not in skip_methods:
            result.diagnostics["methods_attempted"].append("camelot_stream")
            out = _extract_via_camelot(pdf_path, page_number, flavor="stream")
            if out is None:
                result.diagnostics["methods_skipped"].append("camelot_stream")
                result.diagnostics["failure_reasons"]["camelot_stream"] = "import_unavailable"
            elif "error" in out:
                result.diagnostics["failure_reasons"]["camelot_stream"] = out["error"]
                result.diagnostics["parse_times"]["camelot_stream"] = out.get("time_s", 0)
            else:
                rows = out.get("rows", [])
                result.diagnostics["candidate_row_counts"]["camelot_stream"] = len(rows)
                result.diagnostics["parse_times"]["camelot_stream"] = out.get("time_s", 0)
                accuracy = out.get("accuracy", 0)
                conf = min(0.75, 0.4 + accuracy / 200)
                if rows and len(rows) > len(best_rows) and conf > best_confidence:
                    best_rows = rows
                    best_headers = out.get("headers")
                    best_method = "camelot_stream"
                    best_confidence = conf
        else:
            result.diagnostics["methods_skipped"].append("camelot_stream")

    # ── OCR PATH (method 4) ──────────────────────────────────────────────
    if ocr_text and "ocr_row_reconstruct" not in skip_methods:
        result.diagnostics["methods_attempted"].append("ocr_row_reconstruct")
        out = _reconstruct_rows_from_ocr(ocr_text)
        rows = out.get("rows", [])
        result.diagnostics["candidate_row_counts"]["ocr_row_reconstruct"] = len(rows)
        result.diagnostics["parse_times"]["ocr_row_reconstruct"] = out.get("time_s", 0)

        # OCR reconstruction has lower base confidence
        conf = min(0.6, 0.2 + 0.015 * len(rows))
        if rows and (not best_rows or (len(rows) > len(best_rows) * 1.5 and conf >= best_confidence * 0.7)):
            best_rows = rows
            best_headers = out.get("headers")
            best_method = "ocr_row_reconstruct"
            best_confidence = conf

    # ── FINAL RESULT ─────────────────────────────────────────────────────
    result.method_used = best_method
    result.rows = best_rows
    result.headers = best_headers
    result.confidence = best_confidence

    if best_method == "none":
        result.diagnostics["selection_reason"] = "all methods failed or returned 0 rows"
        if not page_input and not ocr_text:
            result.diagnostics["selection_reason"] = "no page_input or ocr_text provided"
    else:
        result.diagnostics["selection_reason"] = (
            f"selected {best_method} with {len(best_rows)} rows "
            f"(confidence {best_confidence:.2f})"
        )

    return result
