"""
Interactive Drawing Measurement Tool
=====================================
Renders PDF drawing pages and lets the estimator click to measure:
  - LINEAR: click two points → distance in mm / m
  - AREA:   click 3+ points → area in sqm

Measurements become BOQ line items added to session state.

Dependencies (all already installed):
  - streamlit_image_coordinates  → click coords from st.image()
  - PyMuPDF (fitz)               → render PDF pages to PIL images
  - Pillow                        → draw measurement overlays

Usage (from demo_page.py or standalone):
    from app.measurement_tool import render_measurement_tool
    render_measurement_tool(pdf_path, page_texts, session_key="measure")
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── local imports (graceful if missing) ─────────────────────────────────────
try:
    from streamlit_image_coordinates import streamlit_image_coordinates as img_coords
    _HAS_COORDS = True
except ImportError:
    _HAS_COORDS = False

try:
    import fitz as _fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

try:
    from src.analysis.qto.scale_detector import (
        detect_scale_from_text,
        detect_scale,
        pixels_to_mm,
        pixels_to_m,
        polygon_area_px,
        pixels_area_to_sqm,
        common_scales,
        ScaleInfo,
    )
    _HAS_SCALE = True
except ImportError:
    _HAS_SCALE = False


# =============================================================================
# PDF RENDERING
# =============================================================================

_ZOOM = 2.0   # render at 2× for higher resolution measurement accuracy


def _render_page(pdf_path: str, page_idx: int, zoom: float = _ZOOM) -> Optional[Image.Image]:
    """Render a PDF page to a PIL Image."""
    if not _HAS_FITZ:
        return None
    try:
        doc = _fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None
        page = doc[page_idx]
        mat = _fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception:
        return None


def _page_count(pdf_path: str) -> int:
    if not _HAS_FITZ:
        return 0
    try:
        doc = _fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


# =============================================================================
# OVERLAY DRAWING (PIL)
# =============================================================================

_POINT_RADIUS = 8
_LINE_WIDTH   = 3
_FILL_ALPHA   = 60    # polygon fill transparency (0–255)

_COLORS = {
    "point":   (255, 80,  0),    # orange
    "line":    (255, 80,  0),
    "area":    (0,  160, 255),   # blue
    "dim":     (255, 255, 0),    # yellow text bg
    "text":    (30,  30,  30),
}


def _overlay_points(
    base_img: Image.Image,
    points: List[Tuple[int, int]],
    mode: str,
    dim_text: str = "",
) -> Image.Image:
    """Draw measurement overlay on top of the base drawing image."""
    img = base_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, (x, y) in enumerate(points):
        r = _POINT_RADIUS
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=(*_COLORS["point"], 220),
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.text((x + r + 3, y - r), str(i + 1), fill=(255, 255, 255, 230))

    if mode == "linear" and len(points) >= 2:
        for i in range(len(points) - 1):
            draw.line(
                [points[i], points[i + 1]],
                fill=(*_COLORS["line"], 220),
                width=_LINE_WIDTH,
            )

    elif mode == "area" and len(points) >= 2:
        for i in range(len(points) - 1):
            draw.line(
                [points[i], points[i + 1]],
                fill=(*_COLORS["area"], 200),
                width=_LINE_WIDTH,
            )
        if len(points) >= 3:
            # Filled polygon
            fill_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            fill_draw = ImageDraw.Draw(fill_overlay)
            fill_draw.polygon(
                points,
                fill=(*_COLORS["area"], _FILL_ALPHA),
                outline=(*_COLORS["area"], 200),
            )
            overlay = Image.alpha_composite(overlay, fill_overlay)
            # Close line
            draw.line(
                [points[-1], points[0]],
                fill=(*_COLORS["area"], 180),
                width=1,
            )

    # Dimension label near centroid
    if dim_text and len(points) >= 2:
        cx = sum(p[0] for p in points) // len(points)
        cy = sum(p[1] for p in points) // len(points)
        # Yellow badge
        font_size = 22
        tw, th = 200, 30
        badge_x, badge_y = cx - tw // 2, cy - th // 2
        draw.rectangle(
            [badge_x - 4, badge_y - 4, badge_x + tw + 4, badge_y + th + 4],
            fill=(255, 240, 0, 200),
        )
        draw.text((badge_x, badge_y), dim_text, fill=_COLORS["text"])

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# =============================================================================
# MEASUREMENT STATE HELPERS
# =============================================================================

def _state(key: str, session_key: str) -> Any:
    return st.session_state.get(f"_meas_{session_key}_{key}")


def _set_state(key: str, val: Any, session_key: str):
    st.session_state[f"_meas_{session_key}_{key}"] = val


def _init_state(session_key: str):
    for k, default in [
        ("points", []),
        ("mode", "linear"),
        ("scale_ratio", 100),
        ("zoom", _ZOOM),
        ("manual_items", []),
        ("dim_result", ""),
    ]:
        sk = f"_meas_{session_key}_{k}"
        if sk not in st.session_state:
            st.session_state[sk] = default


def _pixel_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_measurement_tool(
    pdf_path: Optional[str],
    page_texts: Optional[List[Tuple[int, str, str]]] = None,
    session_key: str = "default",
):
    """
    Render the full interactive measurement panel.

    Args:
        pdf_path:    path to the uploaded PDF (or None if not available)
        page_texts:  list of (page_idx, ocr_text, doc_type) from pipeline
        session_key: unique key to namespace session state (for multi-tool use)
    """
    _init_state(session_key)

    st.markdown("## 📏 Drawing Measurement Tool")
    st.caption(
        "Click on the drawing to place measurement points. "
        "Linear: 2 points = distance. Area: 3+ points = sqm."
    )

    if not _HAS_FITZ:
        st.error("PyMuPDF (fitz) not installed — cannot render PDF pages.")
        return

    if not pdf_path or not Path(pdf_path).exists():
        st.info("Upload a drawing PDF to enable measurement. "
                "Use the main uploader above, then return here.")
        _render_manual_entry_only(session_key)
        return

    n_pages = _page_count(pdf_path)
    if n_pages == 0:
        st.warning("Could not open PDF.")
        return

    # ── Controls row ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])

    with ctrl1:
        page_idx = st.number_input(
            "Page",
            min_value=1, max_value=n_pages, value=1,
            key=f"_meas_{session_key}_page_input",
        ) - 1

    # Auto-detect scale from this page's OCR text
    auto_scale = None
    if page_texts and _HAS_SCALE:
        pg_text = next(
            (t for i, t, _ in page_texts if i == page_idx),
            "",
        )
        auto_scale = detect_scale_from_text(pg_text, source_page=page_idx, zoom=_ZOOM)

    with ctrl2:
        scale_options = common_scales() if _HAS_SCALE else [50, 100, 200]
        default_ratio = auto_scale.ratio if (auto_scale and not auto_scale.is_nts) else 100
        if default_ratio not in scale_options:
            scale_options = sorted(set(scale_options + [default_ratio]))
        scale_ratio = st.selectbox(
            "Scale  (1 : N)",
            options=scale_options,
            index=scale_options.index(default_ratio),
            format_func=lambda x: f"1 : {x}",
            key=f"_meas_{session_key}_scale_select",
        )
        if auto_scale and not auto_scale.is_nts:
            st.caption(f"🔍 Auto-detected: 1:{auto_scale.ratio}")
        elif auto_scale and auto_scale.is_nts:
            st.caption("⚠️ Drawing marked NTS — manual scale required")

    with ctrl3:
        mode = st.radio(
            "Mode",
            options=["linear", "area"],
            format_func=lambda x: "📏 Linear" if x == "linear" else "⬛ Area",
            horizontal=True,
            key=f"_meas_{session_key}_mode_radio",
        )

    with ctrl4:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Points", key=f"_meas_{session_key}_clear"):
            _set_state("points", [], session_key)
            _set_state("dim_result", "", session_key)
            st.rerun()

    # ── Render the drawing page ───────────────────────────────────────────────
    base_img = _render_page(pdf_path, page_idx, zoom=_ZOOM)
    if base_img is None:
        st.error(f"Could not render page {page_idx + 1}.")
        return

    points: List[Tuple[int, int]] = _state("points", session_key) or []
    dim_result: str = _state("dim_result", session_key) or ""

    # Build overlay image
    overlay_img = _overlay_points(base_img, points, mode, dim_result)

    # ── Click-to-measure ─────────────────────────────────────────────────────
    if _HAS_COORDS:
        st.markdown("**Click on the drawing to place points:**")
        click = img_coords(
            overlay_img,
            key=f"_meas_{session_key}_img_p{page_idx}",
            use_column_width=True,
        )
        if click is not None:
            cx, cy = int(click["x"]), int(click["y"])
            # Scale the click coord: streamlit_image_coordinates returns coords
            # relative to displayed width. We need to scale to original pixel space.
            # The image is displayed at full column width; get natural dimensions.
            disp_w = click.get("width") or base_img.width
            scale_factor = base_img.width / disp_w
            cx_nat = int(cx * scale_factor)
            cy_nat = int(cy * scale_factor)

            new_points = points + [(cx_nat, cy_nat)]
            _set_state("points", new_points, session_key)

            # Compute dimension
            dim_text = _compute_dimension(new_points, mode, scale_ratio)
            _set_state("dim_result", dim_text, session_key)
            st.rerun()
    else:
        # Fallback: show image without click interaction
        st.image(overlay_img, use_container_width=True)
        st.warning(
            "Install `streamlit-image-coordinates` for click-to-measure. "
            "Using manual coordinate entry below."
        )

    # ── Dimension display ─────────────────────────────────────────────────────
    if points:
        _render_dimension_display(points, mode, scale_ratio, dim_result, session_key)

    # ── Add to BOQ panel ──────────────────────────────────────────────────────
    _render_add_to_boq(points, mode, scale_ratio, dim_result, session_key)

    # ── Manual items list ─────────────────────────────────────────────────────
    _render_manual_items_table(session_key)


# =============================================================================
# DIMENSION COMPUTATION
# =============================================================================

def _compute_dimension(
    points: List[Tuple[int, int]],
    mode: str,
    scale_ratio: int,
) -> str:
    """Return human-readable dimension string."""
    if not points or not _HAS_SCALE:
        return ""

    if mode == "linear":
        if len(points) < 2:
            return f"Point 1 placed — click second point"
        # Sum all segment lengths
        total_px = sum(
            _pixel_distance(points[i], points[i + 1])
            for i in range(len(points) - 1)
        )
        mm = pixels_to_mm(total_px, scale_ratio, zoom=_ZOOM)
        m = mm / 1000.0
        if m >= 1:
            return f"{m:.3f} m  ({mm:.0f} mm)"
        return f"{mm:.1f} mm"

    elif mode == "area":
        if len(points) < 3:
            n = len(points)
            return f"{n} point{'s' if n != 1 else ''} placed — need 3+ for area"
        area_px2 = polygon_area_px(points)
        sqm = pixels_area_to_sqm(area_px2, scale_ratio, zoom=_ZOOM)
        return f"{sqm:.3f} sqm"

    return ""


def _compute_qty(
    points: List[Tuple[int, int]],
    mode: str,
    scale_ratio: int,
) -> Tuple[float, str]:
    """Return (qty, unit) for the current measurement."""
    if not points or not _HAS_SCALE:
        return 0.0, ""

    if mode == "linear" and len(points) >= 2:
        total_px = sum(
            _pixel_distance(points[i], points[i + 1])
            for i in range(len(points) - 1)
        )
        m = pixels_to_m(total_px, scale_ratio, zoom=_ZOOM)
        return round(m, 3), "m"

    if mode == "area" and len(points) >= 3:
        area_px2 = polygon_area_px(points)
        sqm = pixels_area_to_sqm(area_px2, scale_ratio, zoom=_ZOOM)
        return round(sqm, 3), "sqm"

    return 0.0, ""


# =============================================================================
# UI SUB-PANELS
# =============================================================================

def _render_dimension_display(
    points: List[Tuple[int, int]],
    mode: str,
    scale_ratio: int,
    dim_result: str,
    session_key: str,
):
    """Show current measurement prominently."""
    st.markdown("---")
    d1, d2, d3 = st.columns([3, 2, 2])

    with d1:
        st.markdown(f"### 📐 {dim_result or '—'}")
        st.caption(f"{len(points)} point(s) | Scale 1:{scale_ratio} | Mode: {mode}")

    with d2:
        if len(points) >= 2 and mode == "linear":
            if st.button("↩️ Remove Last Point", key=f"_meas_{session_key}_undo"):
                _set_state("points", points[:-1], session_key)
                _set_state("dim_result", "", session_key)
                st.rerun()

    with d3:
        if mode == "area" and len(points) >= 3:
            if st.button("⬛ Close Polygon", key=f"_meas_{session_key}_close"):
                # Points already closed visually; just keep as-is
                qty, unit = _compute_qty(points, mode, scale_ratio)
                _set_state("dim_result", f"{qty} {unit} (closed)", session_key)
                st.rerun()


def _render_add_to_boq(
    points: List[Tuple[int, int]],
    mode: str,
    scale_ratio: int,
    dim_result: str,
    session_key: str,
):
    """Form to add the current measurement as a BOQ item."""
    qty, unit = _compute_qty(points, mode, scale_ratio)
    if qty <= 0:
        return

    with st.expander("➕ Add Measurement to BOQ", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            desc = st.text_input(
                "Item description",
                value="",
                placeholder="e.g. RCC column 300×600mm, Brick wall 230mm thk...",
                key=f"_meas_{session_key}_desc",
            )
        with c2:
            trade_opts = ["structural", "civil", "finishes", "plumbing",
                          "electrical", "waterproofing", "external"]
            trade = st.selectbox(
                "Trade",
                options=trade_opts,
                key=f"_meas_{session_key}_trade",
            )

        c3, c4, c5 = st.columns([2, 2, 2])
        with c3:
            qty_input = st.number_input(
                f"Quantity ({unit})",
                value=float(qty),
                min_value=0.0,
                step=0.1,
                key=f"_meas_{session_key}_qty",
            )
        with c4:
            unit_opts = ["m", "sqm", "cum", "kg", "nos", "rm", "lump"]
            default_unit_idx = unit_opts.index(unit) if unit in unit_opts else 0
            unit_sel = st.selectbox(
                "Unit",
                options=unit_opts,
                index=default_unit_idx,
                key=f"_meas_{session_key}_unit",
            )
        with c5:
            rate = st.number_input(
                "Rate (₹)",
                value=0.0,
                min_value=0.0,
                step=100.0,
                key=f"_meas_{session_key}_rate",
            )

        if st.button(
            "✅ Add to BOQ",
            key=f"_meas_{session_key}_add_btn",
            disabled=not desc,
            type="primary",
        ):
            new_item = {
                "description":  desc,
                "unit":         unit_sel,
                "qty":          qty_input,
                "rate":         rate if rate > 0 else None,
                "amount":       round(qty_input * rate, 2) if rate > 0 else None,
                "trade":        trade,
                "source":       "manual_measurement",
                "source_page":  0,
                "confidence":   0.95,
                "is_priceable": True,
                "priceable_reason": "priceable",
                "qto_method":   f"manual:{mode}",
                "dim_text":     dim_result,
                "scale":        f"1:{scale_ratio}",
            }
            existing = _state("manual_items", session_key) or []
            _set_state("manual_items", existing + [new_item], session_key)
            _set_state("points", [], session_key)
            _set_state("dim_result", "", session_key)
            st.success(f"Added: {desc} — {qty_input} {unit_sel}")
            st.rerun()


def _render_manual_entry_only(session_key: str):
    """When no PDF is available, show a manual entry form."""
    st.markdown("### ✏️ Manual Measurement Entry")
    st.caption("Enter measurements directly without a drawing.")

    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
    with c1:
        desc = st.text_input("Description", key=f"_meas_{session_key}_m_desc")
    with c2:
        qty = st.number_input("Qty", min_value=0.0, key=f"_meas_{session_key}_m_qty")
    with c3:
        unit = st.selectbox("Unit", ["sqm", "cum", "m", "kg", "nos"],
                             key=f"_meas_{session_key}_m_unit")
    with c4:
        trade = st.selectbox("Trade", ["civil", "structural", "finishes", "plumbing",
                                        "electrical", "waterproofing"],
                              key=f"_meas_{session_key}_m_trade")

    if st.button("Add", key=f"_meas_{session_key}_m_add", disabled=not desc or qty <= 0):
        new_item = {
            "description": desc, "unit": unit, "qty": qty,
            "rate": None, "amount": None, "trade": trade,
            "source": "manual_entry", "confidence": 0.95,
            "is_priceable": True, "priceable_reason": "priceable",
            "qto_method": "manual:entry",
        }
        existing = _state("manual_items", session_key) or []
        _set_state("manual_items", existing + [new_item], session_key)
        st.rerun()

    _render_manual_items_table(session_key)


def _render_manual_items_table(session_key: str):
    """Display the accumulated manual measurements as a table."""
    import pandas as pd

    items = _state("manual_items", session_key) or []
    if not items:
        return

    st.markdown("---")
    st.markdown(f"### 📋 Measured Items ({len(items)})")

    rows = []
    for i, item in enumerate(items):
        rate = item.get("rate") or 0
        amt  = item.get("amount") or (item["qty"] * rate if rate else 0)
        rows.append({
            "#":           i + 1,
            "Description": item["description"],
            "Unit":        item["unit"],
            "Qty":         item["qty"],
            "Rate (₹)":   f"₹{rate:,.0f}" if rate else "—",
            "Amount (₹)": f"₹{amt:,.0f}" if amt else "—",
            "Trade":       item["trade"],
            "Method":      item.get("qto_method", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export button
    col_dl, col_clr = st.columns([2, 1])
    with col_dl:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="manual_measurements.csv",
            mime="text/csv",
            key=f"_meas_{session_key}_download",
        )
    with col_clr:
        if st.button("🗑️ Clear All Items", key=f"_meas_{session_key}_clear_items"):
            _set_state("manual_items", [], session_key)
            st.rerun()

    # Summary
    total_by_trade: Dict[str, float] = {}
    for item in items:
        if item.get("amount"):
            total_by_trade[item["trade"]] = (
                total_by_trade.get(item["trade"], 0) + item["amount"]
            )
    if total_by_trade:
        st.markdown("**Trade summary (measured items):**")
        for trade, total in sorted(total_by_trade.items()):
            st.write(f"  {trade}: ₹{total:,.0f}")


# =============================================================================
# ACCESSOR — get manual items for pipeline merge
# =============================================================================

def get_manual_items(session_key: str = "default") -> List[dict]:
    """
    Return accumulated manual measurement items from session state.
    Call from pipeline or post-analysis to merge into main BOQ.
    """
    return _state("manual_items", session_key) or []


def clear_manual_items(session_key: str = "default"):
    """Clear all manual measurements from session state."""
    _set_state("manual_items", [], session_key)
