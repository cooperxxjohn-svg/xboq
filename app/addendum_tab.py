"""
T5-2: Addendum / Corrigendum Tracker tab.
"""
import streamlit as st
import json
from pathlib import Path


def render_addendum_tab(payload: dict, widget_key_fn) -> None:
    """Render the Addendum Tracker tab."""
    try:
        from src.analysis.addendum_tracker import compare_payloads
    except ImportError as e:
        st.error(f"addendum_tracker module not available: {e}")
        return

    st.markdown("### \U0001f4cb Addendum / Corrigendum Tracker")
    st.caption("Upload a revised tender payload to compare changes against the current run.")

    if not payload:
        st.info("Load a tender first, then upload the revised version to compare.")
        return

    uploaded = st.file_uploader(
        "Upload revised tender payload (JSON)",
        type=["json"],
        key=widget_key_fn("addendum_upload"),
        help="Export the new run's payload as JSON and upload here.",
    )

    if uploaded is None:
        st.markdown("---")
        st.markdown("**How to use:**")
        st.markdown(
            "1. Run the original tender \u2192 this is your **base**\n"
            "2. When an addendum is issued, re-run the analysis with the updated documents\n"
            "3. Export the new payload as JSON and upload above\n"
            "4. Changes are highlighted automatically"
        )
        return

    try:
        new_payload = json.loads(uploaded.read())
    except Exception as e:
        st.error(f"Could not parse uploaded file: {e}")
        return

    with st.spinner("Comparing payloads\u2026"):
        result = compare_payloads(
            base_payload=payload,
            new_payload=new_payload,
            base_run_id="base",
            new_run_id="revised",
        )

    # Summary banner
    delta_sign = "+" if result.cost_delta_inr >= 0 else ""
    color = "#ef4444" if result.cost_delta_inr > 0 else "#22c55e" if result.cost_delta_inr < 0 else "#6b7280"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
        <div style="font-size:0.9rem;color:#a1a1aa;margin-bottom:0.5rem;">Addendum Summary</div>
        <div style="font-size:1.1rem;font-weight:600;color:#e4e4e7;">{result.summary}</div>
        <div style="margin-top:0.5rem;font-size:0.9rem;color:{color};">
            Cost delta: {delta_sign}\u20b9{result.cost_delta_inr:,.0f} ({delta_sign}{result.cost_delta_pct}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not result.has_changes:
        st.success("No changes detected between the two runs.")
        return

    # Tabs for change types
    tabs = st.tabs([
        f"\u2795 Added ({len(result.added_items)})",
        f"\u2796 Deleted ({len(result.deleted_items)})",
        f"\u270f\ufe0f Changed ({len(result.changed_items)})",
        f"\U0001f514 New RFIs ({len(result.new_rfis)})",
    ])

    with tabs[0]:
        if result.added_items:
            for item in result.added_items:
                st.markdown(f"**{item.description}** _(trade: {item.trade})_")
                st.caption(f"Qty: {item.new_quantity} | Rate: \u20b9{item.new_rate:,.0f} | Total: \u20b9{item.new_total:,.0f}")
        else:
            st.info("No new items added.")

    with tabs[1]:
        if result.deleted_items:
            for item in result.deleted_items:
                st.markdown(f"~~{item.description}~~ _(trade: {item.trade})_")
                st.caption(f"Was: Qty {item.old_quantity} | Rate \u20b9{item.old_rate:,.0f} | Total \u20b9{item.old_total:,.0f}")
        else:
            st.info("No items deleted.")

    with tabs[2]:
        if result.changed_items:
            for item in result.changed_items:
                st.markdown(f"**{item.description}** _(trade: {item.trade})_")
                col1, col2, col3 = st.columns(3)
                if item.pct_qty_change is not None:
                    sign = "+" if item.pct_qty_change > 0 else ""
                    col1.metric("Quantity", f"{item.new_quantity}", f"{sign}{item.pct_qty_change}%")
                if item.pct_rate_change is not None:
                    sign = "+" if item.pct_rate_change > 0 else ""
                    col2.metric("Rate (\u20b9)", f"{item.new_rate:,.0f}", f"{sign}{item.pct_rate_change}%")
                old_tot = item.old_total or 0
                new_tot = item.new_total or 0
                col3.metric("Total (\u20b9)", f"{new_tot:,.0f}", f"{new_tot - old_tot:+,.0f}")
                st.markdown("---")
        else:
            st.info("No items changed.")

    with tabs[3]:
        if result.new_rfis:
            for rfi in result.new_rfis:
                st.markdown(f"- **[{rfi.get('priority','?').upper()}]** {rfi.get('question', rfi.get('rfi_text', ''))[:200]}")
        else:
            st.info("No new RFIs in revised tender.")
