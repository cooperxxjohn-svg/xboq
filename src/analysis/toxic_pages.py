"""
Toxic Page Isolation — per-page retry policy with DPI fallback.

Retry policy:
  1. Normal DPI failed during batch OCR
  2. Retry at LOW_DPI (72) grayscale
  3. If still fails → mark as toxic and skip

Pure module, no Streamlit dependency. Can be tested independently.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

# Retry constants
NORMAL_DPI = 150
LOW_DPI = 72
RETRY_TIMEOUT_S = 30


# =============================================================================
# IDENTIFY FAILED PAGES
# =============================================================================

def identify_failed_pages(ocr_metadata: dict) -> List[int]:
    """
    Scan OCR metadata for pages that failed or produced empty text.

    Looks at page_profiles for error fields or empty text after OCR attempt.

    Args:
        ocr_metadata: Dict from extract_ocr_for_all_pages() second return value.
            Expected keys: page_profiles (list of per-page dicts).

    Returns:
        List of 0-indexed page indices that failed OCR.
    """
    failed = []
    profiles = ocr_metadata.get("page_profiles", [])
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        page_idx = profile.get("page_index")
        if page_idx is None:
            continue
        # Check for explicit error
        if profile.get("error"):
            failed.append(page_idx)
            continue
        # Check for OCR attempted but empty result
        if profile.get("ocr_used") and not profile.get("text_length", 1):
            failed.append(page_idx)

    return sorted(set(failed))


# =============================================================================
# RETRY SINGLE PAGE
# =============================================================================

def retry_toxic_page(
    pdf_path: Path,
    page_idx: int,
    normal_dpi: int = NORMAL_DPI,
) -> dict:
    """
    Retry a single failed page at low DPI grayscale.

    Step 1: Re-render at LOW_DPI (72) grayscale → OCR
    Step 2: If still fails → mark toxic

    Args:
        pdf_path: Path to PDF file.
        page_idx: 0-indexed page number.
        normal_dpi: Original DPI that failed (for logging).

    Returns:
        Dict with:
            page_idx: int
            text: str (recovered text, or empty if toxic)
            toxic: bool (True if page is unrecoverable)
            reason: str (failure reason or 'recovered')
            retry_dpi: int (DPI used for retry)
            retry_time_s: float (time spent on retry)
    """
    t0 = time.perf_counter()

    try:
        from .ocr_fallback import render_page_png, _ocr_worker, OCR_CONFIG_SPARSE

        # Retry at low DPI grayscale
        png_bytes = render_page_png(pdf_path, page_idx, dpi=LOW_DPI, grayscale=True)
        if not png_bytes:
            return {
                "page_idx": page_idx,
                "text": "",
                "toxic": True,
                "reason": "render_failed_low_dpi",
                "retry_dpi": LOW_DPI,
                "retry_time_s": round(time.perf_counter() - t0, 3),
            }

        # Run OCR on low-DPI grayscale render
        result = _ocr_worker(png_bytes, page_idx, OCR_CONFIG_SPARSE)
        text = result.get("text", "").strip()

        if text and len(text) > 10:
            # Recovery success
            return {
                "page_idx": page_idx,
                "text": text,
                "toxic": False,
                "reason": "recovered_low_dpi_grayscale",
                "retry_dpi": LOW_DPI,
                "retry_time_s": round(time.perf_counter() - t0, 3),
            }
        else:
            # Still no text — mark as toxic
            return {
                "page_idx": page_idx,
                "text": "",
                "toxic": True,
                "reason": "empty_after_low_dpi_retry",
                "retry_dpi": LOW_DPI,
                "retry_time_s": round(time.perf_counter() - t0, 3),
            }

    except Exception as e:
        return {
            "page_idx": page_idx,
            "text": "",
            "toxic": True,
            "reason": f"retry_exception: {type(e).__name__}: {e}",
            "retry_dpi": LOW_DPI,
            "retry_time_s": round(time.perf_counter() - t0, 3),
        }


# =============================================================================
# BATCH PROCESS TOXIC PAGES
# =============================================================================

def process_toxic_pages(
    pdf_path: Path,
    failed_page_indices: List[int],
    normal_dpi: int = NORMAL_DPI,
) -> List[dict]:
    """
    Batch retry all failed pages with the toxic page retry policy.

    Args:
        pdf_path: Path to PDF file.
        failed_page_indices: List of 0-indexed page numbers that failed.
        normal_dpi: Original DPI that failed.

    Returns:
        List of retry result dicts (one per page), including both
        recovered and toxic pages.
    """
    if not failed_page_indices:
        return []

    results = []
    for page_idx in failed_page_indices:
        result = retry_toxic_page(pdf_path, page_idx, normal_dpi)
        results.append(result)

    return results


# =============================================================================
# SUMMARY
# =============================================================================

def build_toxic_pages_summary(toxic_results: List[dict]) -> dict:
    """
    Build a summary of toxic page retry results for the payload.

    Args:
        toxic_results: List of retry result dicts from process_toxic_pages().

    Returns:
        Dict with: toxic_count, recovered_count, pages, total_retry_time_s.
    """
    if not toxic_results:
        return {
            "toxic_count": 0,
            "recovered_count": 0,
            "pages": [],
            "total_retry_time_s": 0.0,
        }

    toxic_count = sum(1 for r in toxic_results if r.get("toxic"))
    recovered_count = sum(1 for r in toxic_results if not r.get("toxic"))
    total_time = sum(r.get("retry_time_s", 0) for r in toxic_results)

    return {
        "toxic_count": toxic_count,
        "recovered_count": recovered_count,
        "pages": toxic_results,
        "total_retry_time_s": round(total_time, 2),
    }
