"""
OCR Fallback for Image-Only PDFs

Provides OCR text extraction for PDF pages that have no embedded text layer.
Used to detect scale labels, sheet titles, and discipline markers in scanned drawings.

Uses pytesseract (Tesseract OCR) with optimized settings for construction drawings.
"""

import re
import os
import time
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import io

logger = logging.getLogger(__name__)
DEBUG_PIPELINE = os.environ.get("DEBUG_PIPELINE", "0") == "1"

# Optional imports - graceful fallback if not available
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    pytesseract = None
    Image = None

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    fitz = None


# =============================================================================
# OCR CONFIGURATION
# =============================================================================

# Tesseract configuration for construction drawings
# PSM 6 = Assume uniform block of text (good for title blocks)
# PSM 11 = Sparse text (good for drawings with scattered labels)
OCR_CONFIG_TITLE_BLOCK = '--psm 6 --oem 3'
OCR_CONFIG_SPARSE = '--psm 11 --oem 3'

# Minimum text length to consider a page as having "real" text
MIN_TEXT_LENGTH = 50

# Scale patterns to search for in OCR text
SCALE_PATTERNS = [
    r'\b1\s*:\s*(\d+)\b',                    # 1:100, 1 : 50
    r'\bSCALE\s*[=:]?\s*1\s*:\s*(\d+)\b',    # SCALE = 1:100
    r'\bSCALE\s*[=:]?\s*1/(\d+)\b',          # SCALE = 1/100
    r'\b(\d+)\s*["\']?\s*=\s*1\s*[\'"\-]',   # 1/4" = 1'-0"
    r'\bNTS\b',                               # Not To Scale
    r'\bN\.T\.S\.',                           # N.T.S.
]

# Sheet title/discipline patterns
SHEET_TITLE_PATTERNS = [
    r'(?:SHEET|DWG|DRAWING)\s*(?:NO\.?|#)?\s*([A-Z]?\d+)',
    r'\b([A-Z]{1,2})-?(\d{1,3}(?:\.\d{1,2})?)\b',  # A-101, S-2.01
]

DISCIPLINE_KEYWORDS = {
    'A': ['ARCHITECTURAL', 'FLOOR PLAN', 'ELEVATION', 'SECTION'],
    'S': ['STRUCTURAL', 'FOUNDATION', 'BEAM', 'COLUMN', 'FOOTING'],
    'M': ['MECHANICAL', 'HVAC', 'AIR CONDITIONING', 'DUCT'],
    'E': ['ELECTRICAL', 'LIGHTING', 'POWER', 'PANEL'],
    'P': ['PLUMBING', 'DRAINAGE', 'SANITARY', 'WATER SUPPLY'],
}

SHEET_TYPE_KEYWORDS = {
    'floor_plan': ['FLOOR PLAN', 'GROUND FLOOR', 'FIRST FLOOR', 'TYPICAL FLOOR', 'LEVEL'],
    'elevation': ['ELEVATION', 'FRONT ELEVATION', 'SIDE ELEVATION', 'REAR ELEVATION'],
    'section': ['SECTION', 'SECTIONAL', 'CROSS SECTION'],
    'detail': ['DETAIL', 'ENLARGED DETAIL', 'TYPICAL DETAIL'],
    'site_plan': ['SITE PLAN', 'SITE LAYOUT', 'LOCATION PLAN'],
}


# =============================================================================
# OCR EXTRACTION FUNCTIONS
# =============================================================================

def is_ocr_available() -> bool:
    """Check if OCR dependencies are available."""
    return HAS_OCR and HAS_FITZ


def page_needs_ocr(text: str) -> bool:
    """
    Determine if a page needs OCR based on its extracted text.

    Args:
        text: Text extracted from PDF page (may be empty)

    Returns:
        True if OCR should be attempted
    """
    if not text:
        return True

    # Clean text
    cleaned = text.strip()

    # Too short - probably just artifacts
    if len(cleaned) < MIN_TEXT_LENGTH:
        return True

    # Check if text is mostly garbage (common in scanned PDFs with bad OCR)
    alphanumeric = sum(c.isalnum() for c in cleaned)
    if alphanumeric / max(len(cleaned), 1) < 0.3:
        return True

    return False


# =============================================================================
# FAST PREFLIGHT
# =============================================================================

def pdf_preflight(pdf_path: Path, page_texts: List[str]) -> Dict[str, Any]:
    """
    Fast scan of PDF to estimate OCR workload. No rendering, no OCR.
    Opens PDF once, counts embedded images. Uses already-extracted text
    to classify pages.

    Args:
        pdf_path: Path to PDF
        page_texts: Already-extracted text per page (from load_pdf_pages)

    Returns:
        Dict with file_size_mb, page_count, pages_with_text, pages_needing_ocr,
        pct_text_layer, total_images, is_scanned_only, estimated_ocr_seconds,
        ocr_strategy ("skip"|"full"|"adaptive"), explain (human-readable).
    """
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    page_count = len(page_texts)

    # Classify pages using existing page_needs_ocr()
    pages_needing_ocr = sum(1 for t in page_texts if page_needs_ocr(t))
    pages_with_text = page_count - pages_needing_ocr
    pct_text_layer = (pages_with_text / max(page_count, 1)) * 100

    # Count embedded images (fast -- just xref lookup, no rendering)
    total_images = 0
    if HAS_FITZ:
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                total_images += len(page.get_images(full=False))
            doc.close()
        except Exception:
            pass

    is_scanned_only = pct_text_layer < 10 and total_images > 0

    # Estimate: ~4 seconds per page at 150 DPI on average hardware
    SEC_PER_PAGE_OCR = 4.0
    estimated_ocr_seconds = pages_needing_ocr * SEC_PER_PAGE_OCR

    # Determine strategy
    if pages_needing_ocr == 0:
        ocr_strategy = "skip"
        explain = "All pages have text layers. No OCR needed."
    elif pct_text_layer > 70:
        ocr_strategy = "skip"
        explain = (f"{pages_with_text}/{page_count} pages have text layers "
                   f"({pct_text_layer:.0f}%). OCR skipped for speed.")
    elif pages_needing_ocr > 80:
        ocr_strategy = "adaptive"
        est_min = estimated_ocr_seconds / 60
        est_max = estimated_ocr_seconds * 1.5 / 60
        explain = (f"PDF looks {'scanned-only' if is_scanned_only else 'mostly scanned'} "
                   f"(text layer on {pct_text_layer:.0f}% of pages). "
                   f"{pages_needing_ocr} pages need OCR; capping at 80. "
                   f"Est. runtime {est_min:.0f}-{est_max:.0f} min.")
    else:
        est_min = estimated_ocr_seconds / 60
        ocr_strategy = "full"
        explain = (f"{pages_needing_ocr}/{page_count} pages need OCR. "
                   f"Est. runtime ~{est_min:.1f} min.")

    result = {
        'file_size_mb': round(file_size_mb, 1),
        'page_count': page_count,
        'pages_with_text': pages_with_text,
        'pages_needing_ocr': pages_needing_ocr,
        'pct_text_layer': round(pct_text_layer, 1),
        'total_images': total_images,
        'is_scanned_only': is_scanned_only,
        'estimated_ocr_seconds': round(estimated_ocr_seconds, 1),
        'ocr_strategy': ocr_strategy,
        'explain': explain,
    }

    if DEBUG_PIPELINE:
        logger.info(f"[Preflight] {pdf_path.name}: {result}")

    return result


# =============================================================================
# OCR POST-PROCESSING (shared between main process and workers)
# =============================================================================

def _analyze_ocr_text(ocr_text: str) -> Dict[str, Any]:
    """
    Analyze OCR text to detect scale, discipline, sheet type, sheet number.
    Extracted as a standalone function so it can be called from workers.
    """
    result = {
        'scale_detected': False,
        'scale_value': None,
        'discipline': None,
        'sheet_type': None,
        'sheet_number': None,
        'confidence': 0.0,
    }

    text_upper = ocr_text.upper()

    # Detect scale
    for pattern in SCALE_PATTERNS:
        match = re.search(pattern, text_upper)
        if match:
            result['scale_detected'] = True
            if 'NTS' in pattern or 'N.T.S' in pattern:
                result['scale_value'] = 'NTS'
            else:
                groups = match.groups()
                if groups:
                    result['scale_value'] = f"1:{groups[0]}"
            break

    # Detect discipline
    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        if any(kw in text_upper for kw in keywords):
            result['discipline'] = disc
            break

    # Detect sheet type
    for sheet_type, keywords in SHEET_TYPE_KEYWORDS.items():
        if any(kw in text_upper for kw in keywords):
            result['sheet_type'] = sheet_type
            break

    # Detect sheet number
    for pattern in SHEET_TITLE_PATTERNS:
        match = re.search(pattern, text_upper)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                result['sheet_number'] = f"{groups[0]}-{groups[1]}"
            elif groups:
                result['sheet_number'] = groups[0]
            break

    # Calculate confidence
    confidence = 0.3  # Base for successful OCR
    if result['scale_detected']:
        confidence += 0.25
    if result['discipline']:
        confidence += 0.2
    if result['sheet_type']:
        confidence += 0.15
    if result['sheet_number']:
        confidence += 0.1

    result['confidence'] = min(confidence, 1.0)
    return result


# =============================================================================
# PARALLEL OCR WORKER (module-level for pickling)
# =============================================================================

def _ocr_worker(png_bytes: bytes, page_index: int, config: str) -> Dict[str, Any]:
    """
    Worker function for parallel OCR. Runs in a separate process.
    Receives pre-rendered PNG bytes, returns OCR result dict.
    """
    import io as _io
    import time as _time
    from PIL import Image as _Image
    import pytesseract as _pytesseract

    t0 = _time.perf_counter()

    img = _Image.open(_io.BytesIO(png_bytes))
    t_pil = _time.perf_counter()

    try:
        ocr_text = _pytesseract.image_to_string(img, config=config)
    except Exception:
        try:
            ocr_text = _pytesseract.image_to_string(img)
        except Exception:
            ocr_text = ''
    t_ocr = _time.perf_counter()

    # Post-processing
    analysis = _analyze_ocr_text(ocr_text)
    t_post = _time.perf_counter()

    del img

    return {
        'page_index': page_index,
        'text': ocr_text,
        'ocr_attempted': True,
        **analysis,
        'timings': {
            'time_pil_convert_s': round(t_pil - t0, 4),
            'time_ocr_s': round(t_ocr - t_pil, 4),
            'time_postprocess_s': round(t_post - t_ocr, 4),
            'total_worker_s': round(t_post - t0, 4),
        },
    }


# =============================================================================
# SINGLE-PAGE OCR (legacy, still used for standalone calls)
# =============================================================================

def extract_ocr_text_for_page(
    pdf_path: Path,
    page_index: int,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Extract text from a PDF page using OCR.

    Args:
        pdf_path: Path to PDF file
        page_index: 0-indexed page number
        dpi: Resolution for rendering (higher = better OCR but slower)

    Returns:
        Dict with text, scale_detected, scale_value, discipline, sheet_type,
        sheet_number, confidence, timings.
    """
    result = {
        'text': '',
        'scale_detected': False,
        'scale_value': None,
        'discipline': None,
        'sheet_type': None,
        'sheet_number': None,
        'confidence': 0.0,
        'ocr_attempted': False,
        'timings': {},
    }

    if not is_ocr_available():
        return result

    try:
        t0 = time.perf_counter()

        # Open PDF and render page to image
        doc = fitz.open(str(pdf_path))
        if page_index >= len(doc):
            doc.close()
            return result

        page = doc[page_index]
        t_open = time.perf_counter()

        # Render page to pixmap at specified DPI
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        t_render = time.perf_counter()

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        t_pil = time.perf_counter()

        # Close PDF
        doc.close()

        result['ocr_attempted'] = True

        # Run OCR with sparse text config (better for drawings)
        try:
            ocr_text = pytesseract.image_to_string(img, config=OCR_CONFIG_SPARSE)
        except Exception:
            # Fallback to default config
            ocr_text = pytesseract.image_to_string(img)
        t_ocr = time.perf_counter()

        result['text'] = ocr_text

        # Analyze OCR text
        analysis = _analyze_ocr_text(ocr_text)
        result.update(analysis)
        t_post = time.perf_counter()

        result['timings'] = {
            'time_open_pdf_s': round(t_open - t0, 4),
            'time_render_page_s': round(t_render - t_open, 4),
            'time_pil_convert_s': round(t_pil - t_render, 4),
            'time_ocr_s': round(t_ocr - t_pil, 4),
            'time_postprocess_s': round(t_post - t_ocr, 4),
            'total_page_s': round(t_post - t0, 4),
        }

        del img, img_data, pix

    except Exception as e:
        result['error'] = str(e)

    return result


# =============================================================================
# BATCH OCR: single PDF open + parallel workers
# =============================================================================

def render_page_png(
    pdf_path: Path,
    page_idx: int,
    dpi: int = 150,
    grayscale: bool = False,
) -> bytes:
    """
    Render a single PDF page to PNG bytes.

    Extracted as a reusable helper for toxic page retry at different DPI.

    Args:
        pdf_path: Path to PDF file.
        page_idx: 0-indexed page number.
        dpi: Rendering resolution.
        grayscale: If True, convert to grayscale before returning.

    Returns:
        PNG bytes of the rendered page.
    """
    if not HAS_FITZ:
        return b""

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        if grayscale:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        else:
            pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        del pix
        return png_bytes
    finally:
        doc.close()


def extract_ocr_for_all_pages(
    pdf_path: Path,
    existing_texts: List[str],
    dpi: int = 150,
    progress_callback: Optional[Callable] = None,
    max_ocr_pages: int = 0,
    max_workers: int = 0,
    selected_pages: Optional[List[int]] = None,
    wall_clock_budget_s: float = 600.0,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract OCR text for all pages that need it, merging with existing text.

    Opens the PDF once, renders all OCR-needed pages to PNG, then runs
    Tesseract in parallel using a process pool.

    Args:
        pdf_path: Path to PDF file
        existing_texts: List of text per page (from PDF text layer)
        dpi: Resolution for OCR rendering
        progress_callback: Optional callable(page_idx, total, message)
        max_ocr_pages: Cap on pages to OCR (0 = unlimited)
        max_workers: Number of parallel OCR workers (0 = auto)
        selected_pages: Optional list of page indices to OCR (overrides auto-detection)
        wall_clock_budget_s: Max total seconds for OCR (0 = unlimited, default 600)

    Returns:
        Tuple of:
            - List of texts (merged: original + OCR where needed)
            - Dict with OCR metadata per page
    """
    merged_texts = list(existing_texts)
    ocr_metadata = {
        'pages_with_ocr': [],
        'scales_detected': [],
        'disciplines_detected': [],
        'sheet_types_detected': [],
        'total_ocr_confidence': 0.0,
        'page_profiles': [],
    }

    if not is_ocr_available():
        return merged_texts, ocr_metadata

    # Identify which pages need OCR
    if selected_pages is not None:
        pages_to_ocr = [i for i in selected_pages if i < len(existing_texts) and page_needs_ocr(existing_texts[i])]
    else:
        pages_to_ocr = [i for i, text in enumerate(existing_texts) if page_needs_ocr(text)]

    if not pages_to_ocr:
        return merged_texts, ocr_metadata

    # Apply hard cap
    if max_ocr_pages > 0 and len(pages_to_ocr) > max_ocr_pages:
        logger.warning(
            f"Capping OCR to {max_ocr_pages} pages (had {len(pages_to_ocr)})"
        )
        pages_to_ocr = pages_to_ocr[:max_ocr_pages]

    total_ocr = len(pages_to_ocr)

    # ── Phase 1: Render all pages to PNG (sequential, single PDF open) ──
    rendered_pages = []  # List of (page_index, png_bytes)
    render_timings = {}

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    try:
        for step, page_index in enumerate(pages_to_ocr):
            t_r0 = time.perf_counter()

            pix = doc[page_index].get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            rendered_pages.append((page_index, png_bytes))
            del pix  # Free native memory immediately

            render_timings[page_index] = round(time.perf_counter() - t_r0, 4)

            if progress_callback:
                progress_callback(
                    step, total_ocr,
                    f"Rendering page {step + 1}/{total_ocr} for OCR..."
                )
    finally:
        doc.close()

    # ── Phase 2: Parallel OCR ──
    if max_workers == 0:
        max_workers = min(4, max(1, mp.cpu_count() // 2))

    # For small batches, stay single-process to avoid pool overhead
    if len(rendered_pages) <= 2:
        max_workers = 1

    results_map = {}
    ocr_start = time.perf_counter()

    if max_workers == 1:
        # Sequential fallback
        for step, (page_index, png_bytes) in enumerate(rendered_pages):
            # Wall-clock budget check
            if wall_clock_budget_s > 0 and (time.perf_counter() - ocr_start) > wall_clock_budget_s:
                logger.warning(
                    f"Wall-clock budget ({wall_clock_budget_s}s) exceeded after {step} pages, "
                    f"skipping remaining {len(rendered_pages) - step} pages"
                )
                # Record skipped pages
                for _, (skip_idx, _) in enumerate(rendered_pages[step:]):
                    results_map[skip_idx] = {
                        'text': '',
                        'ocr_attempted': False,
                        'error': 'wall_clock_exceeded',
                        'timings': {},
                    }
                break

            result = _ocr_worker(png_bytes, page_index, OCR_CONFIG_SPARSE)
            results_map[page_index] = result

            if progress_callback:
                elapsed = time.perf_counter() - ocr_start
                done = step + 1
                if done > 0:
                    remaining = (elapsed / done) * (total_ocr - done)
                    progress_callback(
                        step, total_ocr,
                        f"OCR complete: {done}/{total_ocr} pages "
                        f"(est. {remaining:.0f}s remaining)"
                    )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for page_index, png_bytes in rendered_pages:
                future = executor.submit(
                    _ocr_worker, png_bytes, page_index, OCR_CONFIG_SPARSE
                )
                futures[future] = page_index

            wall_clock_exceeded = False
            for step, future in enumerate(as_completed(futures)):
                page_index = futures[future]

                # Wall-clock budget check
                if wall_clock_budget_s > 0 and (time.perf_counter() - ocr_start) > wall_clock_budget_s:
                    if not wall_clock_exceeded:
                        logger.warning(
                            f"Wall-clock budget ({wall_clock_budget_s}s) exceeded, "
                            f"cancelling remaining OCR futures"
                        )
                        wall_clock_exceeded = True
                    # Still collect completed results, but don't wait for pending
                    if future.done():
                        try:
                            result = future.result(timeout=1)
                            results_map[page_index] = result
                        except Exception:
                            results_map[page_index] = {
                                'text': '',
                                'ocr_attempted': True,
                                'error': 'wall_clock_exceeded',
                                'timings': {},
                            }
                    else:
                        future.cancel()
                        results_map[page_index] = {
                            'text': '',
                            'ocr_attempted': False,
                            'error': 'wall_clock_exceeded',
                            'timings': {},
                        }
                    continue

                try:
                    result = future.result(timeout=60)
                    results_map[page_index] = result
                except Exception as e:
                    logger.error(f"OCR worker failed for page {page_index}: {e}")
                    results_map[page_index] = {
                        'text': '',
                        'ocr_attempted': True,
                        'error': str(e),
                        'timings': {},
                    }

                if progress_callback:
                    elapsed = time.perf_counter() - ocr_start
                    done = step + 1
                    if done > 0:
                        remaining = (elapsed / done) * (total_ocr - done)
                        progress_callback(
                            step, total_ocr,
                            f"OCR complete: {done}/{total_ocr} pages "
                            f"(est. {remaining:.0f}s remaining)"
                        )

    # Free rendered PNG bytes
    del rendered_pages

    # ── Phase 3: Merge results and collect metadata ──
    ocr_count = 0
    confidence_sum = 0.0

    for page_index in sorted(results_map.keys()):
        ocr_result = results_map[page_index]
        existing_text = existing_texts[page_index]

        # Build page profile for diagnostics
        profile = {
            'page_index': page_index,
            'page_total': len(existing_texts),
            'has_text_layer': not page_needs_ocr(existing_text),
            'ocr_used': ocr_result.get('ocr_attempted', False),
            'raster_dpi': dpi,
            'time_render_s': render_timings.get(page_index, 0),
        }
        if ocr_result.get('timings'):
            profile.update(ocr_result['timings'])
        profile['total_page_s'] = round(
            profile.get('time_render_s', 0) +
            profile.get('total_worker_s', 0), 4
        )

        ocr_metadata['page_profiles'].append(profile)

        if DEBUG_PIPELINE:
            logger.info(
                f"[Stage2] page {page_index + 1}/{len(existing_texts)} | "
                f"ocr={'yes' if ocr_result.get('ocr_attempted') else 'no'} | "
                f"render={profile.get('time_render_s', 0):.2f}s | "
                f"ocr_time={profile.get('time_ocr_s', 0):.2f}s | "
                f"total={profile.get('total_page_s', 0):.2f}s"
            )

        if ocr_result.get('ocr_attempted') and ocr_result.get('text'):
            # Merge OCR text with any existing text
            merged_texts[page_index] = (
                existing_text + "\n\n[OCR TEXT]\n" + ocr_result['text']
            )
            ocr_count += 1
            confidence_sum += ocr_result.get('confidence', 0)

            ocr_metadata['pages_with_ocr'].append(page_index)

            if ocr_result.get('scale_detected'):
                ocr_metadata['scales_detected'].append({
                    'page': page_index,
                    'scale': ocr_result.get('scale_value'),
                })

            if ocr_result.get('discipline'):
                ocr_metadata['disciplines_detected'].append({
                    'page': page_index,
                    'discipline': ocr_result.get('discipline'),
                })

            if ocr_result.get('sheet_type'):
                ocr_metadata['sheet_types_detected'].append({
                    'page': page_index,
                    'sheet_type': ocr_result.get('sheet_type'),
                })

    if ocr_count > 0:
        ocr_metadata['total_ocr_confidence'] = confidence_sum / ocr_count

    return merged_texts, ocr_metadata


# =============================================================================
# SPRINT 20F: OCR QUALITY ROUTING FOR TABLE-HEAVY SCANNED PAGES
# =============================================================================

# Doc types that benefit from enhanced OCR preprocessing
TABLE_HEAVY_DOC_TYPES = {"boq", "schedule", "spec", "conditions"}


def ocr_quality_route(
    pdf_path: Path,
    page_idx: int,
    doc_type: str,
    existing_text: str,
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Enhanced OCR routing for difficult pages (table-heavy scans).

    If a page is classified as BOQ/schedule/spec/conditions and the existing
    OCR text is poor (low density / low confidence), runs one additional
    preprocessing variant (grayscale + higher contrast) and picks the best.

    Reuses the toxic-page retry philosophy: deterministic, fast, no heavy deps.

    Args:
        pdf_path: Path to PDF file.
        page_idx: 0-indexed page number.
        doc_type: Classified doc_type from page_index.
        existing_text: Already-extracted text for this page.
        dpi: Base DPI for rendering.

    Returns:
        Dict with keys:
            text: str — best OCR text (may be same as existing)
            variant_used: str — "original" | "grayscale_enhanced"
            improved: bool
            diagnostics: dict
    """
    result = {
        "text": existing_text,
        "variant_used": "original",
        "improved": False,
        "diagnostics": {
            "page_idx": page_idx,
            "doc_type": doc_type,
            "attempted": False,
        },
    }

    if not is_ocr_available():
        return result

    # Only route table-heavy doc types
    if doc_type not in TABLE_HEAVY_DOC_TYPES:
        return result

    # Check if existing text is poor enough to warrant retry
    existing_clean = (existing_text or "").strip()
    existing_len = len(existing_clean)

    # Heuristic: text is "poor" if short or low alphanumeric ratio
    if existing_len > 200:
        alpha_ratio = sum(c.isalnum() for c in existing_clean) / max(existing_len, 1)
        if alpha_ratio > 0.4:
            # Text quality seems acceptable, skip enhancement
            result["diagnostics"]["skip_reason"] = "text_quality_acceptable"
            return result

    result["diagnostics"]["attempted"] = True
    result["diagnostics"]["original_text_len"] = existing_len

    try:
        # Render page as grayscale at slightly higher DPI
        enhanced_dpi = min(dpi + 50, 300)  # Cap at 300 to stay fast
        png_bytes = render_page_png(pdf_path, page_idx, dpi=enhanced_dpi, grayscale=True)

        if not png_bytes:
            result["diagnostics"]["failure"] = "render_failed"
            return result

        # OCR the enhanced image
        t0 = time.perf_counter()
        enhanced_result = _ocr_worker(png_bytes, page_idx, OCR_CONFIG_SPARSE)
        t_ocr = time.perf_counter() - t0

        enhanced_text = enhanced_result.get("text", "")
        enhanced_len = len(enhanced_text.strip())

        result["diagnostics"]["enhanced_text_len"] = enhanced_len
        result["diagnostics"]["enhanced_dpi"] = enhanced_dpi
        result["diagnostics"]["enhanced_ocr_time_s"] = round(t_ocr, 4)

        # Use enhanced text if it produced more content
        if enhanced_len > existing_len * 1.2 and enhanced_len > 100:
            result["text"] = enhanced_text
            result["variant_used"] = "grayscale_enhanced"
            result["improved"] = True
            result["diagnostics"]["improvement_ratio"] = round(enhanced_len / max(existing_len, 1), 2)

    except Exception as e:
        result["diagnostics"]["error"] = str(e)

    return result


# =============================================================================
# DETECTION HELPERS (Using OCR Results)
# =============================================================================

def detect_scale_from_ocr(ocr_text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect scale notation from OCR text.

    Returns:
        Tuple of (found: bool, scale_value: str or None)
    """
    text_upper = ocr_text.upper()

    for pattern in SCALE_PATTERNS:
        match = re.search(pattern, text_upper)
        if match:
            if 'NTS' in pattern or 'N.T.S' in pattern:
                return True, 'NTS'
            groups = match.groups()
            if groups:
                return True, f"1:{groups[0]}"
            return True, 'detected'

    return False, None


def detect_discipline_from_ocr(ocr_text: str) -> Optional[str]:
    """
    Detect discipline from OCR text.

    Returns:
        Discipline code (A, S, M, E, P) or None
    """
    text_upper = ocr_text.upper()

    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        if any(kw in text_upper for kw in keywords):
            return disc

    return None


def detect_sheet_type_from_ocr(ocr_text: str) -> Optional[str]:
    """
    Detect sheet type from OCR text.

    Returns:
        Sheet type (floor_plan, elevation, etc.) or None
    """
    text_upper = ocr_text.upper()

    for sheet_type, keywords in SHEET_TYPE_KEYWORDS.items():
        if any(kw in text_upper for kw in keywords):
            return sheet_type

    return None


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG if DEBUG_PIPELINE else logging.INFO)

    print(f"OCR Available: {is_ocr_available()}")
    print(f"DEBUG_PIPELINE: {DEBUG_PIPELINE}")

    if len(sys.argv) < 2:
        print("Usage: python ocr_fallback.py <pdf_path> [page_index]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    page_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\nExtracting OCR from page {page_idx} of {pdf_path.name}...")
    result = extract_ocr_text_for_page(pdf_path, page_idx, dpi=200)

    print(f"\n{'='*60}")
    print(f"OCR Result:")
    print(f"  Text length: {len(result.get('text', ''))}")
    print(f"  Scale detected: {result.get('scale_detected')} ({result.get('scale_value')})")
    print(f"  Discipline: {result.get('discipline')}")
    print(f"  Sheet type: {result.get('sheet_type')}")
    print(f"  Sheet number: {result.get('sheet_number')}")
    print(f"  Confidence: {result.get('confidence', 0):.2f}")
    if result.get('timings'):
        print(f"  Timings: {result['timings']}")
    print(f"{'='*60}")

    if result.get('text'):
        print(f"\nFirst 500 chars of OCR text:")
        print(result['text'][:500])
