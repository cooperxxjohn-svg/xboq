"""
SVG Vector Renderer — Sprint 28

Renders PDF pages as SVG for crisp, zoomable, lightweight web display.
Uses PyMuPDF (fitz) page.get_svg_image() — same library as thumbnails.py.

Drawing-type pages (plan, detail, section, elevation) get vector SVG.
Text-heavy pages (spec, conditions, boq) stay raster PNG.
"""

import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Doc types that benefit from vector SVG rendering (geometric content)
SVG_DOC_TYPES = {"plan", "detail", "section", "elevation"}


def is_svg_doc_type(doc_type: str) -> bool:
    """Check if a doc_type benefits from SVG vector rendering.

    Args:
        doc_type: Page classification string from page_index.

    Returns:
        True for geometric drawing types (plan, detail, section, elevation).
    """
    return doc_type in SVG_DOC_TYPES


@st.cache_data(show_spinner=False, max_entries=50)
def _cached_page_svg(pdf_path: str, page_idx: int, text_mode: str = "path") -> Optional[str]:
    """Render a PDF page to SVG string using PyMuPDF.

    Args:
        pdf_path: Absolute path to the PDF file.
        page_idx: 0-indexed page number.
        text_mode: "path" (text as vector curves, crisp but not selectable)
                   or "text" (text as <text> elements, selectable/searchable).

    Returns:
        SVG string if successful, None on error or missing file.
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
        text_as_path = (text_mode != "text")
        svg_str = page.get_svg_image(text_as_path=text_as_path)
        doc.close()
        return svg_str
    except Exception:
        return None


def get_svg_page_dimensions(pdf_path: str, page_idx: int) -> Optional[Tuple[float, float]]:
    """Get page dimensions in PDF points (for SVG viewBox mapping).

    Args:
        pdf_path: Absolute path to the PDF file.
        page_idx: 0-indexed page number.

    Returns:
        (width, height) in points, or None on error.
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
        w, h = page.rect.width, page.rect.height
        doc.close()
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def inject_bbox_overlay_svg(
    svg_str: Optional[str],
    bboxes: Optional[List[List[Any]]],
    page_width: float,
    page_height: float,
    label_boxes: bool = False,
) -> Optional[str]:
    """Inject <rect> elements into SVG for evidence bounding-box overlay.

    Uses the same confidence-based color scheme as the PNG overlay in
    render_pdf_page_with_overlay() (demo_page.py lines 1386-1409).

    Args:
        svg_str: Base SVG string from get_svg_image(). None returns None.
        bboxes: List of [x0_rel, y0_rel, x1_rel, y1_rel, confidence?, bbox_id?]
                in page-relative coords (0.0-1.0). None = no overlay.
        page_width: Page width in PDF points (from page.rect.width).
        page_height: Page height in PDF points (from page.rect.height).
        label_boxes: If True, add numbered circle labels on each box.

    Returns:
        SVG string with overlay <rect> elements injected before </svg>,
        or original svg_str if no bboxes, or None if svg_str is None.
    """
    if svg_str is None:
        return None
    if not bboxes:
        return svg_str

    overlay_elements = []
    box_num = 0

    for box in bboxes:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x0_rel, y0_rel, x1_rel, y1_rel = box[:4]
        conf = box[4] if len(box) > 4 else 0.5

        # Suppress low-confidence boxes (same threshold as PNG overlay)
        if conf < 0.4:
            continue
        box_num += 1

        # Clamp to valid range
        x0_rel = max(0.0, min(1.0, float(x0_rel)))
        y0_rel = max(0.0, min(1.0, float(y0_rel)))
        x1_rel = max(0.0, min(1.0, float(x1_rel)))
        y1_rel = max(0.0, min(1.0, float(y1_rel)))

        # Convert relative coords to SVG viewBox coords (PDF points)
        x0 = x0_rel * page_width
        y0 = y0_rel * page_height
        x1 = x1_rel * page_width
        y1 = y1_rel * page_height
        w = x1 - x0
        h = y1 - y0

        # Color by confidence (exact same scheme as PNG overlay)
        if conf >= 0.8:
            stroke = "rgba(34,197,94,0.86)"    # green
            fill = "rgba(34,197,94,0.12)"
        elif conf >= 0.6:
            stroke = "rgba(245,158,11,0.86)"   # amber
            fill = "rgba(245,158,11,0.12)"
        else:
            stroke = "rgba(239,68,68,0.86)"    # red
            fill = "rgba(239,68,68,0.12)"

        overlay_elements.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" />'
        )

        if label_boxes:
            cx = x0 + 12
            cy = y0 + 12
            overlay_elements.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" '
                f'fill="rgba(245,158,11,0.78)" stroke="white" stroke-width="1" />'
                f'<text x="{cx:.1f}" y="{cy + 4:.1f}" '
                f'text-anchor="middle" font-size="11" font-weight="bold" '
                f'fill="black">{box_num}</text>'
            )

    if not overlay_elements:
        return svg_str

    # Inject before </svg> closing tag
    overlay_group = '<g class="xboq-bbox-overlay">' + ''.join(overlay_elements) + '</g>'
    return svg_str.replace('</svg>', overlay_group + '</svg>')


def render_svg_html(svg_str: str, height: int = 600) -> str:
    """Wrap SVG in a responsive HTML container for Streamlit display.

    The container provides scrollable pan for large drawings.

    Args:
        svg_str: SVG markup string.
        height: Container height in pixels.

    Returns:
        HTML string suitable for st.components.v1.html().
    """
    return f'''<div style="width:100%;height:{height}px;overflow:auto;border:1px solid #e0e0e0;
                border-radius:4px;background:#fafafa;">
        {svg_str}
    </div>'''
