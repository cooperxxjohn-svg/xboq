"""
XBOQ Demo - Pre-Bid Scope & Risk Check

User uploads drawings → Gets scope analysis + RFIs + GO/NO-GO.
Includes export buttons for operational deliverables.
Now with evidence display and "Why?" expanders.
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exports.pricing_readiness import get_pricing_readiness_buffer
from exports.rfi_pack import get_rfi_pack_buffer
from exports.bid_packet import get_bid_packet_buffer

# Try to import deep analysis models (optional)
try:
    from models.analysis_models import EvidenceRef, Blocker, RFIItem
    HAS_DEEP_ANALYSIS = True
except ImportError:
    HAS_DEEP_ANALYSIS = False

# Page config
st.set_page_config(
    page_title="XBOQ - Pre-Bid Scope Check",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, rgb(0,0,0) 0%, rgb(10,10,12) 100%);
    }

    .main .block-container {
        max-width: 1000px;
        padding: 2rem 1rem;
    }

    h1, h2, h3, p, span, div {
        font-family: 'Inter', sans-serif !important;
    }

    /* Export buttons styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        color: white !important;
        border: none !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
    }
</style>
""", unsafe_allow_html=True)


def load_demo_results(project_id: str) -> dict:
    """Load results from output directory."""
    base_path = Path(__file__).parent.parent / "out" / project_id

    results = {
        "loaded": False,
        "rfis": [],
        "trade_summary": {},
        "bid_gate": {},
        "metrics": {},
        "critical_blockers": [],
    }

    # Load RFIs
    rfi_path = base_path / "rfi" / "rfis.json"
    if rfi_path.exists():
        with open(rfi_path) as f:
            rfi_data = json.load(f)
            results["rfis"] = rfi_data.get("rfis", [])
            results["trade_summary"] = rfi_data.get("by_trade", {})
            results["critical_blockers"] = rfi_data.get("critical_blockers", [])

    # Load bid gate
    bid_path = base_path / "bid_gate_result.json"
    if bid_path.exists():
        with open(bid_path) as f:
            results["bid_gate"] = json.load(f)

    # Load run metadata
    meta_path = base_path / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            agg = meta.get("aggregates", {})
            results["metrics"] = {
                "pages": agg.get("pages_processed", 0),
                "rooms": agg.get("rooms_found", 0),
                "openings": agg.get("openings_found", 0),
            }

    # Load deep analysis if available
    deep_path = base_path / "deep_analysis.json"
    if deep_path.exists():
        with open(deep_path) as f:
            results["deep_analysis"] = json.load(f)
    else:
        results["deep_analysis"] = None

    # Load plan graph if available
    graph_path = base_path / "plan_graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            results["plan_graph"] = json.load(f)
    else:
        results["plan_graph"] = None

    results["loaded"] = len(results["rfis"]) > 0 or results["bid_gate"]
    return results


def render_evidence_expander(evidence: dict, title: str = "Why?"):
    """Render an expandable evidence section."""
    if not evidence:
        return

    with st.expander(f"📋 {title}"):
        # Pages
        pages = evidence.get("pages", [])
        if pages:
            page_nums = [str(p + 1) for p in pages[:10]]  # 1-indexed for display
            st.markdown(f"**Found on pages:** {', '.join(page_nums)}")

        # Detected entities
        entities = evidence.get("detected_entities", {})
        if entities:
            for key, val in entities.items():
                if isinstance(val, list) and val:
                    display_items = val[:8]
                    more = len(val) - 8 if len(val) > 8 else 0
                    items_str = ", ".join(str(v) for v in display_items)
                    if more > 0:
                        items_str += f" (+{more} more)"
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {items_str}")
                elif isinstance(val, (int, float)):
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {val}")

        # Search attempts
        search = evidence.get("search_attempts", {})
        if search:
            searched_items = []
            for key, val in search.items():
                if isinstance(val, list):
                    searched_items.extend(val[:3])
                elif isinstance(val, str):
                    searched_items.append(val[:50])
            if searched_items:
                st.markdown(f"**Searched for:** {', '.join(searched_items[:5])}")

        # Confidence
        confidence = evidence.get("confidence", 0)
        if confidence:
            confidence_pct = int(confidence * 100)
            confidence_text = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
            st.caption(f"Confidence: {confidence_text} ({confidence_pct}%)")

        reason = evidence.get("confidence_reason", "")
        if reason:
            st.caption(f"*{reason}*")


def render_export_section(results: dict, project_id: str):
    """Render export buttons section."""
    st.markdown("### 📦 Export Deliverables")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Bid Readiness Packet**")
        st.caption("Full report + assumptions")

        # ZIP download
        zip_buffer = get_bid_packet_buffer(
            project_id=project_id,
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            bid_gate=results["bid_gate"],
            metrics=results["metrics"],
            format="zip"
        )
        st.download_button(
            "⬇️ Download ZIP",
            data=zip_buffer,
            file_name=f"bid_packet_{project_id}.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # HTML download
        html_buffer = get_bid_packet_buffer(
            project_id=project_id,
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            bid_gate=results["bid_gate"],
            metrics=results["metrics"],
            format="html"
        )
        st.download_button(
            "📄 Download HTML",
            data=html_buffer,
            file_name=f"bid_packet_{project_id}.html",
            mime="text/html",
            use_container_width=True,
        )

    with col2:
        st.markdown("**RFI Pack**")
        st.caption("Tracker + email drafts")

        # CSV download
        csv_buffer = get_rfi_pack_buffer(
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            project_id=project_id,
            format="csv"
        )
        st.download_button(
            "📊 Download CSV",
            data=csv_buffer,
            file_name=f"rfi_tracker_{project_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Email draft download
        txt_buffer = get_rfi_pack_buffer(
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            project_id=project_id,
            format="txt"
        )
        st.download_button(
            "✉️ Email Drafts",
            data=txt_buffer,
            file_name=f"rfi_emails_{project_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # HTML view download
        rfi_html_buffer = get_rfi_pack_buffer(
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            project_id=project_id,
            format="html"
        )
        st.download_button(
            "📄 Print View",
            data=rfi_html_buffer,
            file_name=f"rfi_pack_{project_id}.html",
            mime="text/html",
            use_container_width=True,
        )

    with col3:
        st.markdown("**Pricing Readiness**")
        st.caption("Trade-wise pricing status")

        # Excel download
        xlsx_buffer = get_pricing_readiness_buffer(
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            bid_gate=results["bid_gate"],
            format="xlsx"
        )
        st.download_button(
            "📈 Download Excel",
            data=xlsx_buffer,
            file_name=f"pricing_readiness_{project_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # CSV download
        pricing_csv_buffer = get_pricing_readiness_buffer(
            rfis=results["rfis"],
            trade_summary=results["trade_summary"],
            bid_gate=results["bid_gate"],
            format="csv"
        )
        st.download_button(
            "📊 Download CSV",
            data=pricing_csv_buffer,
            file_name=f"pricing_readiness_{project_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_fix_loop_panel(results: dict):
    """Render the fix loop panel showing how to improve score."""
    bid_gate = results["bid_gate"]
    score = bid_gate.get("score", 0)
    blockers = results.get("critical_blockers", [])

    if score >= 80:
        return  # Already good, no need for fix panel

    st.markdown("### 🔧 Improve Your Score")

    # Calculate potential score
    potential_fixes = []
    potential_score = score

    # Check for scale issues
    for blocker in bid_gate.get("blockers", []):
        if "scale" in blocker.lower():
            potential_fixes.append({
                "action": "Calibrate scale",
                "delta": 10,
                "description": "Set scale for unscaled pages",
            })
            potential_score += 10
            break

    # Check for missing schedules
    for blocker in blockers:
        if "schedule" in blocker.lower():
            potential_fixes.append({
                "action": "Upload schedules",
                "delta": 8,
                "description": "Add door/window/finish schedules",
            })
            potential_score += 8
            break

    # Check for missing MEP
    for blocker in blockers:
        if "mep" in blocker.lower() or "electrical" in blocker.lower():
            potential_fixes.append({
                "action": "Upload MEP drawings",
                "delta": 12,
                "description": "Add electrical/plumbing drawings",
            })
            potential_score += 12
            break

    if not potential_fixes:
        potential_fixes.append({
            "action": "Resolve RFIs",
            "delta": 5,
            "description": "Get clarifications for open items",
        })
        potential_score += 5

    potential_score = min(potential_score, 95)

    st.info(f"You can reach **~{potential_score}/100** by fixing {len(potential_fixes)} item(s)")

    for fix in potential_fixes[:3]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{fix['action']}** (+{fix['delta']} pts)")
            st.caption(fix['description'])
        with col2:
            if fix['action'] == "Calibrate scale":
                if st.button("Set Scale", key="fix_scale"):
                    st.session_state.show_scale_calibration = True
            elif "Upload" in fix['action']:
                st.file_uploader(
                    "Upload",
                    type=["pdf"],
                    key=f"upload_{fix['action']}",
                    label_visibility="collapsed"
                )

    # Scale calibration modal
    if st.session_state.get("show_scale_calibration"):
        st.markdown("---")
        st.markdown("**Scale Calibration**")
        scale_col1, scale_col2 = st.columns(2)
        with scale_col1:
            scale = st.selectbox("Select scale", ["1:100", "1:50", "1:200", "1:500", "Custom"])
        with scale_col2:
            if scale == "Custom":
                st.text_input("Enter scale (e.g., 1:75)")
        if st.button("Apply Scale"):
            st.success("Scale set to " + scale + " - Re-run analysis to update score")
            st.session_state.show_scale_calibration = False


def main():
    # Header
    st.markdown("# XBOQ")
    st.caption("Pre-Bid Scope & Risk Check")

    # Check for demo project
    project_id = st.query_params.get("project_id", "")

    if not project_id:
        # =====================
        # UPLOAD INTERFACE
        # =====================
        st.markdown("---")
        st.markdown("### Upload your tender drawings")
        st.markdown("Drop your drawing PDFs to detect scope gaps before you bid.")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            analyze_btn = st.button("Analyze Drawings", type="primary", use_container_width=True)

        if uploaded_files and analyze_btn:
            with st.spinner("Processing drawings..."):
                st.info(f"Uploaded {len(uploaded_files)} file(s). Full processing coming soon...")

        st.markdown("---")
        st.markdown("#### See it in action")
        if st.button("View Demo Analysis →", type="secondary"):
            st.query_params["project_id"] = "bitmesra_clean"
            st.rerun()

    else:
        # =====================
        # RESULTS VIEW
        # =====================
        results = load_demo_results(project_id)

        if not results["loaded"]:
            st.error(f"No results found for project: {project_id}")
            if st.button("← Back to upload"):
                st.query_params.clear()
                st.rerun()
            return

        # Back button
        if st.button("← Upload new drawings"):
            st.query_params.clear()
            st.rerun()

        st.caption(f"Analysis: **{project_id}**")

        # -----------------------
        # DECISION BANNER
        # -----------------------
        bid_gate = results["bid_gate"]
        status = bid_gate.get("status", "NO-GO")
        score = bid_gate.get("score", 0)

        if status == "GO":
            st.success(f"### ✓ GO\nBid Readiness Score: {score}/100")
        elif status in ["REVIEW", "CONDITIONAL"]:
            st.warning(f"### ⚠ REVIEW NEEDED\nBid Readiness Score: {score}/100")
        else:
            st.error(f"### ✗ NO-GO\nBid Readiness Score: {score}/100")

        # -----------------------
        # EXPORT BUTTONS (prominent)
        # -----------------------
        render_export_section(results, project_id)

        st.markdown("---")

        # -----------------------
        # METRICS
        # -----------------------
        metrics = results["metrics"]
        rfis = results["rfis"]
        high_priority = sum(1 for r in rfis if r.get("priority") == "high")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pages Analyzed", metrics.get("pages", 0))
        col2.metric("RFIs Generated", len(rfis))
        col3.metric("Critical Issues", high_priority)
        col4.metric("Rooms Detected", metrics.get("rooms", 0))

        st.markdown("---")

        # -----------------------
        # FIX LOOP PANEL
        # -----------------------
        render_fix_loop_panel(results)

        st.markdown("---")

        # -----------------------
        # CRITICAL BLOCKERS (with evidence)
        # -----------------------
        deep = results.get("deep_analysis", {})
        deep_blockers = deep.get("blockers", []) if deep else []

        if deep_blockers:
            # Use deep analysis blockers with structured evidence
            critical_blockers = [b for b in deep_blockers if b.get("severity") in ["critical", "high"]]
            if critical_blockers:
                st.markdown("### 🚫 Critical Blockers")
                st.caption("These must be resolved before pricing:")

                for blocker in critical_blockers[:6]:
                    with st.container():
                        severity = blocker.get("severity", "high").upper()
                        title = blocker.get("title", "Unknown issue")
                        st.markdown(f"**{blocker.get('id', 'BLK')}:** {title}")

                        # Show impact
                        impact_cost = blocker.get("impact_cost", "")
                        impact_schedule = blocker.get("impact_schedule", "")
                        if impact_cost or impact_schedule:
                            impact_text = []
                            if impact_cost:
                                impact_text.append(f"Cost: {impact_cost}")
                            if impact_schedule:
                                impact_text.append(f"Schedule: {impact_schedule}")
                            st.caption(f"Impact: {' | '.join(impact_text)}")

                        # Evidence expander
                        evidence = blocker.get("evidence", {})
                        if evidence:
                            render_evidence_expander(evidence, "Why is this a blocker?")

                        # Fix actions
                        fix_actions = blocker.get("fix_actions", [])
                        if fix_actions:
                            with st.expander("🔧 How to fix"):
                                for i, action in enumerate(fix_actions[:3], 1):
                                    st.markdown(f"{i}. {action}")

                                # Show what it unlocks
                                unlocks = blocker.get("unlocks_boq_categories", [])
                                if unlocks:
                                    st.caption(f"Unlocks: {', '.join(unlocks[:5])}")

                        st.markdown("")
        else:
            # Fallback to legacy blockers
            blockers = results.get("critical_blockers", [])
            if blockers:
                st.markdown("### 🚫 Critical Blockers")
                st.caption("These must be resolved before pricing:")
                for blocker in blockers[:6]:
                    st.markdown(f"- {blocker}")
                st.markdown("")

        # -----------------------
        # TRADE SUMMARY
        # -----------------------
        trade_summary = results.get("trade_summary", {})
        if trade_summary:
            st.markdown("### Trade-Wise Gap Summary")

            table_data = []
            for trade, data in trade_summary.items():
                if data.get("rfi_count", 0) == 0:
                    continue
                gaps = ", ".join(data.get("gaps", [])[:2]) or "—"
                high = data.get("high_priority", 0)
                priority = f"🔴 {high} critical" if high > 0 else "—"
                table_data.append({
                    "Trade": trade.title(),
                    "RFIs": data.get("rfi_count", 0),
                    "Priority": priority,
                    "Key Gaps": gaps
                })

            if table_data:
                st.dataframe(table_data, use_container_width=True, hide_index=True)

        st.markdown("---")

        # -----------------------
        # RFIs with Evidence
        # -----------------------
        st.markdown("### RFIs to Resolve Before Pricing")

        # Check for deep analysis RFIs first
        deep = results.get("deep_analysis", {})
        deep_rfis = deep.get("rfis", []) if deep else []

        if deep_rfis:
            # Use deep analysis RFIs with structured evidence
            high_rfis = [r for r in deep_rfis if r.get("priority") in ["high", "critical"]]
            medium_rfis = [r for r in deep_rfis if r.get("priority") == "medium"]

            for rfi in high_rfis[:6]:
                with st.container():
                    st.markdown(f"**🔴 HIGH** — {rfi.get('question', rfi.get('title', 'Untitled'))}")
                    st.caption(rfi.get('why_it_matters', rfi.get('description', ''))[:200])

                    # Evidence expander with full details
                    evidence = rfi.get("evidence", {})
                    if evidence and (evidence.get("pages") or evidence.get("detected_entities")):
                        render_evidence_expander(evidence, "Why is this blocking?")

                    # Resolution suggestion
                    resolution = rfi.get("suggested_resolution", "")
                    if resolution:
                        st.markdown(f"✅ **Fix:** {resolution}")

                    st.markdown("")
        else:
            # Fallback to legacy RFIs
            high_rfis = [r for r in rfis if r.get("priority") == "high"]
            medium_rfis = [r for r in rfis if r.get("priority") == "medium"]

            for rfi in high_rfis[:6]:
                with st.container():
                    st.markdown(f"**🔴 HIGH** — {rfi.get('title', 'Untitled')}")
                    st.caption(rfi.get('description', '')[:200])

                    # Legacy evidence section
                    evidence_pages = rfi.get("evidence_pages", [])
                    detected_tags = rfi.get("detected_tags", [])
                    if evidence_pages or detected_tags:
                        evidence_text = []
                        if evidence_pages:
                            evidence_text.append(f"Pages: {', '.join(map(str, evidence_pages))}")
                        if detected_tags:
                            evidence_text.append(f"Tags: {', '.join(detected_tags[:5])}")
                        st.markdown(f"*Evidence: {' | '.join(evidence_text)}*")

                    impact = rfi.get("impact", "")
                    if impact:
                        st.markdown(f"*→ {impact}*")
                    st.markdown("")

        if medium_rfis:
            with st.expander(f"View {len(medium_rfis)} more RFIs (Medium Priority)"):
                for rfi in medium_rfis[:10]:
                    question = rfi.get('question', rfi.get('title', 'Untitled'))
                    desc = rfi.get('why_it_matters', rfi.get('description', ''))[:150]
                    st.markdown(f"**🟡 MEDIUM** — {question}")
                    st.caption(desc)
                    st.markdown("")

        # -----------------------
        # DEBUG: Plan Graph Download
        # -----------------------
        plan_graph = results.get("plan_graph")
        if plan_graph:
            st.markdown("---")
            with st.expander("🔍 Debug: Plan Set Graph"):
                st.caption("Download the plan graph for debugging analysis accuracy.")

                # Show summary
                st.markdown(f"**Total Pages:** {plan_graph.get('total_pages', 0)}")
                st.markdown(f"**Disciplines:** {', '.join(plan_graph.get('disciplines_found', [])) or 'None detected'}")
                st.markdown(f"**Sheet Types:** {json.dumps(plan_graph.get('sheet_types_found', {}))}")
                st.markdown(f"**Door Tags:** {len(plan_graph.get('all_door_tags', []))} unique")
                st.markdown(f"**Window Tags:** {len(plan_graph.get('all_window_tags', []))} unique")
                st.markdown(f"**Scale:** {plan_graph.get('pages_with_scale', 0)} with / {plan_graph.get('pages_without_scale', 0)} without")

                # Download button
                st.download_button(
                    "📥 Download plan_graph.json",
                    data=json.dumps(plan_graph, indent=2),
                    file_name=f"plan_graph_{project_id}.json",
                    mime="application/json",
                )

        # Footer
        st.markdown("---")
        st.caption("Generated by XBOQ • Pre-Bid Scope & Risk Check")


if __name__ == "__main__":
    main()
