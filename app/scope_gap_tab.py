"""
T5-3: Scope Gap Detector tab.
"""
import streamlit as st


_SEVERITY_COLORS = {
    "high": "#ef4444",
    "medium": "#f97316",
    "low": "#6b7280",
}
_SEVERITY_ICONS = {
    "high": "\U0001f534",
    "medium": "\U0001f7e0",
    "low": "\u26aa",
}


def render_scope_gap_tab(payload: dict, widget_key_fn) -> None:
    """Render the Scope Gap Detector tab."""
    try:
        from src.analysis.scope_gap import detect_scope_gaps
    except ImportError as e:
        st.error(f"scope_gap module not available: {e}")
        return

    if not payload:
        st.info("Load a tender to detect scope gaps.")
        return

    st.markdown("### \U0001f50d Scope Gap Detector")
    st.caption("Cross-references drawing signals against BOQ items to flag potentially missing scope.")

    run_key = widget_key_fn("scope_gap_run")
    result_key = widget_key_fn("scope_gap_result")

    if st.button("Run Scope Gap Analysis", key=run_key, type="primary"):
        with st.spinner("Analysing scope coverage\u2026"):
            result = detect_scope_gaps(payload)
        st.session_state[result_key] = result

    result = st.session_state.get(result_key)
    if result is None:
        st.markdown("---")
        st.markdown(
            "Scope gap detection cross-checks **19 common scope categories** across civil, "
            "architectural, MEP, sitework, and structural trades against your BOQ."
        )
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Gaps", result.total_gaps)
    col2.metric("High Severity", result.high_severity, delta=None)
    col3.metric("Medium Severity", result.medium_severity)
    col4.metric("BOQ Coverage", f"{result.coverage_pct}%")

    st.markdown(f"_{result.summary}_")
    st.markdown("---")

    if result.total_gaps == 0:
        st.success("No scope gaps detected.")
        return

    # Filter
    severity_filter = st.selectbox(
        "Filter by severity",
        ["All", "High", "Medium"],
        key=widget_key_fn("scope_gap_filter"),
    )
    trade_options = ["All"] + sorted(set(g.trade for g in result.gaps))
    trade_filter = st.selectbox(
        "Filter by trade",
        trade_options,
        key=widget_key_fn("scope_gap_trade"),
    )

    gaps = result.gaps
    if severity_filter != "All":
        gaps = [g for g in gaps if g.severity.lower() == severity_filter.lower()]
    if trade_filter != "All":
        gaps = [g for g in gaps if g.trade.lower() == trade_filter.lower()]

    st.markdown(f"Showing **{len(gaps)}** gap{'s' if len(gaps) != 1 else ''}")

    for gap in gaps:
        icon = _SEVERITY_ICONS.get(gap.severity, "\u26aa")
        color = _SEVERITY_COLORS.get(gap.severity, "#6b7280")
        with st.expander(f"{icon} [{gap.gap_id}] {gap.description} \u2014 _{gap.trade}_"):
            st.markdown(f"**Severity:** <span style='color:{color}'>{gap.severity.upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"**Drawing signals found:** {', '.join(gap.drawing_signals_found)}")
            st.markdown(f"**BOQ coverage:** {gap.boq_coverage}")
            st.markdown(f"**Recommendation:** {gap.recommendation}")
