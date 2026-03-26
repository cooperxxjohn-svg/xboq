"""
Page Thumbnail Generator — Sprint 27

Generates and caches lightweight JPEG thumbnails for PDF pages.
Uses PyMuPDF (fitz), same library as demo_page.py's page renderer.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Optional


@st.cache_data(show_spinner=False, max_entries=500)
def generate_thumbnail(pdf_path: str, page_idx: int, width: int = 200) -> Optional[bytes]:
    """Render a single PDF page as a small JPEG thumbnail.

    Args:
        pdf_path: Absolute path to the PDF file.
        page_idx: 0-indexed page number.
        width: Target thumbnail width in pixels (height scales proportionally).

    Returns:
        JPEG bytes if successful, None on error or missing file.
    """
    if not pdf_path:
        return None
    try:
        import fitz
        p = Path(pdf_path)
        if not p.exists():
            return None
        doc = fitz.open(str(p))
        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None
        page = doc[page_idx]
        # Scale so width matches target; height scales proportionally
        page_width = page.rect.width
        if page_width <= 0:
            doc.close()
            return None
        zoom = width / page_width
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        jpeg_bytes = pix.tobytes("jpeg")
        doc.close()
        return jpeg_bytes
    except Exception:
        return None


def generate_all_thumbnails(
    pdf_path: str,
    total_pages: int,
    width: int = 200,
) -> Dict[int, bytes]:
    """Generate thumbnails for all pages in a PDF.

    Args:
        pdf_path: Absolute path to the PDF file.
        total_pages: Number of pages to generate (0..total_pages-1).
        width: Target thumbnail width in pixels.

    Returns:
        Dict mapping page_idx → JPEG bytes. Pages that fail render are omitted.
    """
    thumbs: Dict[int, bytes] = {}
    for idx in range(total_pages):
        data = generate_thumbnail(pdf_path, idx, width)
        if data:
            thumbs[idx] = data
    return thumbs


def get_or_create_thumbnails(
    pdf_path: str,
    total_pages: int,
    width: int = 200,
) -> Dict[int, bytes]:
    """Get thumbnails from session state, generating if needed.

    Stores thumbnails in ``st.session_state["_page_thumbnails"]`` so they
    survive Streamlit reruns without re-rendering.

    Args:
        pdf_path: Absolute path to the PDF file.
        total_pages: Number of pages.
        width: Target thumbnail width.

    Returns:
        Dict mapping page_idx → JPEG bytes.
    """
    cache_key = "_page_thumbnails"
    cached = st.session_state.get(cache_key)
    if cached and isinstance(cached, dict) and cached.get("_pdf_path") == pdf_path:
        return {k: v for k, v in cached.items() if isinstance(k, int)}
    # Generate fresh
    thumbs = generate_all_thumbnails(pdf_path, total_pages, width)
    # Store with path marker for cache invalidation
    store = dict(thumbs)
    store["_pdf_path"] = pdf_path  # type: ignore[assignment]
    st.session_state[cache_key] = store
    return thumbs
