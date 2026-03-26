"""
Sheet Takeoff Tab — T4-3.

Per-sheet view: drawing page metadata alongside extracted quantities.
One expandable row per indexed page, grouped by discipline.

Usage:
    from app.sheet_takeoff import render_sheet_takeoff_tab
    render_sheet_takeoff_tab(payload)
"""

from __future__ import annotations

import hashlib
from typing import Dict, List


# Trade → discipline mapping (mirrors rfi_engine classification order)
_TRADE_DISCIPLINE: Dict[str, str] = {
    "structural": "structural",
    "civil":      "civil",
    "concrete":   "structural",
    "rebar":      "structural",
    "steel":      "structural",
    "formwork":   "structural",
    "masonry":    "architectural",
    "plaster":    "architectural",
    "painting":   "architectural",
    "finishing":  "architectural",
    "tiles":      "architectural",
    "flooring":   "architectural",
    "doors":      "architectural",
    "windows":    "architectural",
    "electrical": "mep",
    "plumbing":   "mep",
    "hvac":       "mep",
    "mep":        "mep",
    "fire":       "fire",
    "sprinkler":  "fire",
    "sitework":   "civil",
    "waterproofing": "civil",
}


def _make_widget_key(*parts) -> str:
    """Deterministic widget key (avoids circular import with demo_page.py)."""
    raw = "_".join(str(p) for p in parts)
    return "wk_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def _trade_to_discipline(trade: str) -> str:
    return _TRADE_DISCIPLINE.get(trade.lower(), "other")


def render_sheet_takeoff_tab(payload: dict) -> None:
    """
    Render the per-sheet takeoff view inside a Streamlit tab.

    Reads:
      - payload["diagnostics"]["page_index"]["pages"] — indexed pages
      - payload["boq_items"]                           — extracted line items

    Groups items by discipline to correlate with pages.
    """
    import streamlit as st

    st.markdown("#### 📐 Sheet Takeoff")
    st.caption("Per-sheet breakdown of indexed pages and extracted quantities")

    # ── Pull data ────────────────────────────────────────────────────────────
    diagnostics = payload.get("diagnostics") or {}
    page_index  = diagnostics.get("page_index") or payload.get("page_index") or {}
    pages: List[dict] = page_index.get("pages") or []
    boq_items: List[dict] = payload.get("boq_items") or []

    if not pages:
        st.info("No page index data available for this tender.")
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    all_disciplines = sorted({p.get("discipline", "unknown") for p in pages})
    all_doc_types   = sorted({p.get("doc_type", "unknown") for p in pages})

    col_d, col_t = st.columns(2)
    with col_d:
        sel_disc = st.multiselect(
            "Filter by discipline",
            options=all_disciplines,
            default=all_disciplines,
            key=_make_widget_key("sheet_disc_filter"),
        )
    with col_t:
        sel_type = st.multiselect(
            "Filter by page type",
            options=all_doc_types,
            default=all_doc_types,
            key=_make_widget_key("sheet_type_filter"),
        )

    filtered_pages = [
        p for p in pages
        if p.get("discipline", "unknown") in sel_disc
        and p.get("doc_type", "unknown") in sel_type
    ]

    st.caption(f"Showing **{len(filtered_pages)}** of **{len(pages)}** pages")
    st.markdown("---")

    # ── Build discipline → items lookup ──────────────────────────────────────
    disc_items: Dict[str, List[dict]] = {}
    for item in boq_items:
        trade = item.get("trade", "")
        disc  = _trade_to_discipline(trade)
        disc_items.setdefault(disc, []).append(item)

    # ── Render per-page expanders ────────────────────────────────────────────
    if not filtered_pages:
        st.warning("No pages match the selected filters.")
        return

    for page in filtered_pages:
        page_num  = page.get("page_idx", "?")
        doc_type  = page.get("doc_type", "unknown")
        discipline = page.get("discipline", "unknown")
        title     = page.get("title") or page.get("sheet_id") or f"Page {page_num}"
        conf      = page.get("confidence", 0.0)
        source    = page.get("source", "pdf")

        tag = "🔷" if source == "dxf" else "📄"
        label = f"{tag} **Page {page_num}** · {doc_type} · {discipline} · {title}"

        with st.expander(label, expanded=False):
            meta_col, items_col = st.columns([1, 2])

            with meta_col:
                st.markdown("**Page metadata**")
                st.write(f"Doc type: `{doc_type}`")
                st.write(f"Discipline: `{discipline}`")
                st.write(f"Confidence: `{conf:.0%}`")
                if page.get("sheet_id"):
                    st.write(f"Sheet ID: `{page['sheet_id']}`")
                if page.get("keywords_hit"):
                    kw = page["keywords_hit"][:8]
                    st.write(f"Keywords: {', '.join(str(k) for k in kw)}")
                if source == "dxf" and page.get("entities"):
                    st.write(f"Entities: `{page['entities']}`")

            with items_col:
                matched = disc_items.get(discipline, [])
                if matched:
                    import pandas as pd
                    rows = []
                    for it in matched[:50]:
                        rows.append({
                            "Trade":       it.get("trade", ""),
                            "Description": it.get("description", "")[:60],
                            "Qty":         it.get("quantity") or it.get("qty") or "",
                            "Unit":        it.get("unit", ""),
                            "Rate (₹)":    it.get("rate_inr") or it.get("rate") or "",
                            "Total (₹)":   it.get("total_inr") or it.get("total") or "",
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                        key=_make_widget_key("sheet_items", page_num),
                    )
                    if len(matched) > 50:
                        st.caption(f"… and {len(matched) - 50} more items")
                else:
                    st.info("No items extracted for this discipline.")
