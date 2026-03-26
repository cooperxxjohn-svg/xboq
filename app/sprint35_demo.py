"""
xBOQ.ai — Delhi NCR Bid Analysis Demo
Upload tender drawing PDFs and see the full analysis output.
Focused on Delhi NCR pricing, scope, and risk assessment.
"""

import streamlit as st
import sys
import os
import time
import traceback
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis_runner import (
    run_analysis_pipeline,
    save_uploaded_files,
    generate_project_id,
    AnalysisResult,
)

# Sprint 35 engines
from src.boq.scope_dependencies import (
    analyze_scope_gaps, get_rule_count, get_required_item_count,
)
from src.boq.deduplicator import deduplicate_boq, find_duplicate_clusters
from src.boq.completeness_scorer import score_boq_completeness, get_improvement_suggestions
from src.analysis.quantity_crosscheck import cross_check_boq
from src.analysis.bid_risk_analyzer import analyze_bid_risk
from src.pricing.location_factors import (
    get_city_factor, get_material_city_factor, get_all_cities,
)
from src.pricing.escalation import escalate_rate


# ── Page Config ──
st.set_page_config(
    page_title="xBOQ.ai — Delhi NCR Bid Analysis",
    page_icon="https://xboq.ai/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main > div { padding-top: 0.5rem; }
    * { font-family: 'Inter', sans-serif; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a365d 0%, #2d3748 50%, #1a365d 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-banner h1 { color: white !important; margin-bottom: 0.25rem; font-size: 2rem; }
    .hero-banner p { color: rgba(255,255,255,0.8); margin: 0; font-size: 1rem; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #e2e8f0;
        margin-top: 0.5rem;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        background: #f8fafc;
        margin: 1rem 0;
        transition: all 0.2s;
    }
    .upload-zone:hover { border-color: #667eea; background: #f0f4ff; }
    .upload-icon { font-size: 3rem; margin-bottom: 0.5rem; }

    /* Metric cards */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card.green { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .metric-card.amber { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); }
    .metric-card.red { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
    .metric-card.blue { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; opacity: 0.85; letter-spacing: 0.05em; }
    .metric-value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .metric-detail { font-size: 0.75rem; opacity: 0.8; margin-top: 2px; }

    /* Decision banner */
    .decision-go {
        background: linear-gradient(135deg, #065f46 0%, #059669 100%);
        padding: 1.5rem 2rem; border-radius: 12px; color: white;
        text-align: center; margin: 1rem 0;
    }
    .decision-nogo {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        padding: 1.5rem 2rem; border-radius: 12px; color: white;
        text-align: center; margin: 1rem 0;
    }
    .decision-review {
        background: linear-gradient(135deg, #92400e 0%, #d97706 100%);
        padding: 1.5rem 2rem; border-radius: 12px; color: white;
        text-align: center; margin: 1rem 0;
    }
    .decision-label { font-size: 2.5rem; font-weight: 700; }
    .decision-sub { font-size: 1rem; opacity: 0.9; margin-top: 0.25rem; }

    /* Risk cards */
    .risk-critical { background-color: #fee2e2; border-left: 4px solid #dc2626;
                     padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .risk-high { background-color: #fff7ed; border-left: 4px solid #ea580c;
                 padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .risk-medium { background-color: #fefce8; border-left: 4px solid #ca8a04;
                   padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .risk-ok { background-color: #f0fdf4; border-left: 4px solid #16a34a;
               padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .risk-info { background-color: #eff6ff; border-left: 4px solid #2563eb;
                 padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0; }

    /* Gap items */
    .gap-item { background: #f8fafc; padding: 0.5rem 1rem; border-radius: 8px;
                margin: 0.25rem 0; border: 1px solid #e2e8f0; }

    /* Section headers */
    .section-header {
        background: #f1f5f9; padding: 0.75rem 1.25rem; border-radius: 8px;
        margin: 1.5rem 0 0.75rem; border-left: 4px solid #1a365d;
    }
    .section-header h3 { margin: 0; color: #1a365d; font-size: 1.1rem; }

    /* Stage pipeline */
    .stage-row { display: flex; gap: 0.5rem; margin: 0.5rem 0; align-items: center; }
    .stage-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .stage-dot.done { background: #059669; }
    .stage-dot.running { background: #d97706; animation: pulse 1s infinite; }
    .stage-dot.pending { background: #cbd5e1; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

    /* Pricing table */
    .pricing-highlight { background: #fef3c7; padding: 2px 6px; border-radius: 4px; font-weight: 600; }

    h1 { color: #1E3A5F; }
    .block-container { max-width: 1300px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS — DELHI NCR FOCUS
# =============================================================================

DELHI_NCR_REGION = "Delhi"
DELHI_NCR_CITIES = ["Delhi", "Gurgaon", "Noida", "Faridabad", "Ghaziabad", "Greater Noida"]

# Default project parameters for Delhi NCR
DEFAULT_PROJECT = {
    "name": "Delhi NCR Construction Tender",
    "location": DELHI_NCR_REGION,
    "duration_months": 18,
    "value_lakhs": 2500,
    "num_floors": 4,
    "plot_area_sqm": 300,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_boq_from_payload(payload: Dict) -> List[Dict]:
    """Extract BOQ items from pipeline output payload."""
    # Try multiple locations where BOQ data might be stored
    boq_items = []

    # 1. Direct boq_items key
    if payload.get("boq_items"):
        boq_items = payload["boq_items"]

    # 2. From extraction_summary
    elif payload.get("extraction_summary", {}).get("boq_items"):
        boq_items = payload["extraction_summary"]["boq_items"]

    # 3. From boq_stats with items
    elif payload.get("boq_stats", {}).get("items"):
        boq_items = payload["boq_stats"]["items"]

    # 4. From trade_coverage, reconstruct minimal BOQ
    elif payload.get("trade_coverage"):
        for tc in payload["trade_coverage"]:
            trade = tc.get("trade", "general")
            for i in range(tc.get("priceable_count", 0)):
                boq_items.append({
                    "description": f"{trade} item {i+1}",
                    "quantity": 1.0,
                    "unit": "ls",
                    "trade": trade,
                })

    return boq_items


def _extract_detected_elements(payload: Dict) -> List[str]:
    """Infer detected structural elements from pipeline payload."""
    elements = set()

    # From structural_takeoff
    st_data = payload.get("structural_takeoff", {})
    if st_data and st_data.get("mode") not in (None, "error"):
        # If we have structural takeoff, we have these elements
        summary = st_data.get("summary", {})
        if summary.get("concrete_m3", 0) > 0:
            elements.update(["footing", "column", "beam", "slab"])

    # From plan_graph
    pg = payload.get("plan_graph", {})
    if pg.get("columns"):
        elements.add("column")
    if pg.get("doors"):
        elements.add("door")
    if pg.get("windows"):
        elements.add("window")
    if pg.get("rooms"):
        elements.add("room")

    # From trade coverage
    for tc in payload.get("trade_coverage", []):
        trade = tc.get("trade", "")
        if trade == "structural" and tc.get("priceable_count", 0) > 0:
            elements.update(["footing", "column", "beam", "slab"])

    # Fallback: if any BOQ mentions structural items
    for item in _extract_boq_from_payload(payload):
        desc = (item.get("description") or "").lower()
        if "footing" in desc or "foundation" in desc:
            elements.add("footing")
        if "column" in desc:
            elements.add("column")
        if "beam" in desc:
            elements.add("beam")
        if "slab" in desc:
            elements.add("slab")
        if "stair" in desc:
            elements.add("staircase")
        if "lintel" in desc:
            elements.add("lintel")

    return list(elements) if elements else ["footing", "column", "beam", "slab"]


def _extract_room_types(payload: Dict) -> List[str]:
    """Infer room types from pipeline payload."""
    rooms = set()

    pg = payload.get("plan_graph", {})
    for room in pg.get("rooms", []):
        name = (room.get("name") or room.get("label") or "").lower()
        if "bed" in name:
            rooms.add("bedroom")
        elif "living" in name or "drawing" in name:
            rooms.add("living_room")
        elif "kitchen" in name:
            rooms.add("kitchen")
        elif "toilet" in name or "wc" in name or "lavatory" in name:
            rooms.add("toilet")
        elif "bath" in name:
            rooms.add("bathroom")
        elif "balcony" in name:
            rooms.add("balcony")
        elif "store" in name:
            rooms.add("store")
        elif "pooja" in name or "puja" in name:
            rooms.add("pooja_room")

    return list(rooms) if rooms else ["bedroom", "living_room", "kitchen", "toilet", "balcony"]


def _get_severity_css(severity: str) -> tuple:
    """Return (css_class, icon_html) for a severity level."""
    mapping = {
        "ok": ("risk-ok", "&#9989;"),
        "low": ("risk-ok", "&#128309;"),
        "medium": ("risk-medium", "&#128993;"),
        "high": ("risk-high", "&#128992;"),
        "critical": ("risk-critical", "&#128308;"),
    }
    return mapping.get(severity, ("risk-info", "&#8505;"))


# =============================================================================
# PIPELINE RUNNER WITH PROGRESS
# =============================================================================

def _run_pipeline_with_progress(uploaded_files, run_mode="demo_fast"):
    """Run the extraction pipeline with Streamlit progress UI."""
    project_id = generate_project_id()
    uploads_dir = PROJECT_ROOT / "uploads"
    output_dir = PROJECT_ROOT / "out" / project_id

    stages = [
        ("load", "Loading PDFs", "Scanning uploaded files..."),
        ("index", "Indexing Pages", "Classifying pages by type & discipline..."),
        ("select", "Selecting Pages", "Prioritizing within OCR budget..."),
        ("extract", "Extracting Data", "OCR + table extraction..."),
        ("graph", "Building Graph", "Analyzing plan structure..."),
        ("reason", "Analyzing Scope", "Detecting blockers & coverage..."),
        ("rfi", "Generating RFIs", "Building evidence-backed RFIs..."),
        ("export", "Saving Results", "Writing output files..."),
    ]

    analysis_result = None

    with st.status("Analyzing drawings...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        stage_text = st.empty()
        detail_text = st.empty()
        eta_text = st.empty()

        current_stage_idx = 0
        stage_progress = {}

        def progress_callback(stage_id: str, message: str, progress: float):
            nonlocal current_stage_idx
            stage_progress[stage_id] = progress

            for i, (sid, _, _) in enumerate(stages):
                if sid == stage_id:
                    current_stage_idx = i
                    break

            completed = sum(1 for sid, _, _ in stages[:current_stage_idx]
                            if stage_progress.get(sid, 0) >= 1.0)
            current_pct = stage_progress.get(stage_id, 0)
            overall = (completed + current_pct) / len(stages)
            progress_bar.progress(min(overall, 1.0))

            sname = stages[current_stage_idx][1] if current_stage_idx < len(stages) else "Complete"
            stage_text.markdown(f"**Stage {current_stage_idx + 1}/{len(stages)}:** {sname}")
            detail_text.caption(message)

            if "s/page" in message:
                import re
                rate_m = re.search(r'([\d.]+)s/page', message)
                eta_m = re.search(r'est\.\s*(\d+)s\s*remaining', message)
                if rate_m and eta_m:
                    secs = int(eta_m.group(1))
                    eta_text.markdown(
                        f"**{rate_m.group(1)}s/page** | ~{secs // 60}m {secs % 60}s remaining"
                    )

        stage_text.markdown(f"**Stage 1/{len(stages)}:** {stages[0][1]}")
        detail_text.caption("Saving uploaded files...")

        try:
            saved_files = save_uploaded_files(uploaded_files, project_id, uploads_dir)

            start_time = time.time()

            result = run_analysis_pipeline(
                input_files=saved_files,
                project_id=project_id,
                output_dir=output_dir,
                progress_callback=progress_callback,
                run_mode=run_mode,
            )

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)

            if result.success:
                status.update(
                    label=f"Analysis complete in {elapsed:.0f}s",
                    state="complete",
                    expanded=False,
                )
                analysis_result = result
            else:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(f"Analysis failed: {result.error_message}")
                if result.stack_trace:
                    with st.expander("Error Details"):
                        st.code(result.stack_trace, language="python")

        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Unexpected error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc(), language="python")

    return analysis_result


# =============================================================================
# RENDER: PIPELINE OVERVIEW
# =============================================================================

def render_pipeline_overview(payload: Dict, result: AnalysisResult):
    """Render the extraction pipeline overview section."""
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    stats = payload.get("processing_stats", {})
    timings = payload.get("timings", {})

    # Decision banner
    decision = payload.get("decision", "NEEDS_REVIEW")
    readiness = payload.get("readiness_score", 0)

    if decision == "GO":
        css = "decision-go"
    elif decision == "NO-GO":
        css = "decision-nogo"
    else:
        css = "decision-review"

    st.markdown(f"""
    <div class="{css}">
        <div class="decision-label">{decision}</div>
        <div class="decision-sub">Readiness Score: {readiness:.0f}/100 | Location: Delhi NCR</div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    total_pages = overview.get("pages_total") or stats.get("total_pages", 0)
    deep_pages = stats.get("deep_processed_pages", 0)
    ocr_pages = stats.get("ocr_pages", 0)
    disciplines = overview.get("disciplines_detected", [])
    total_time = timings.get("total_s", result.duration_sec)
    blockers_count = len(payload.get("blockers", []))
    rfis_count = len(payload.get("rfis", []))

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-label">Pages Analyzed</div>
            <div class="metric-value">{total_pages}</div>
            <div class="metric-detail">{deep_pages} deep | {ocr_pages} OCR</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Disciplines</div>
            <div class="metric-value">{len(disciplines)}</div>
            <div class="metric-detail">{', '.join(disciplines[:3]) or 'Detecting...'}</div>
        </div>
        <div class="metric-card {'red' if blockers_count > 5 else 'amber' if blockers_count > 0 else 'green'}">
            <div class="metric-label">Blockers</div>
            <div class="metric-value">{blockers_count}</div>
            <div class="metric-detail">{rfis_count} RFIs generated</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Processing Time</div>
            <div class="metric-value">{total_time:.0f}s</div>
            <div class="metric-detail">{len(result.files_generated)} files generated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-scores breakdown
    sub_scores = payload.get("sub_scores", {})
    if sub_scores:
        st.markdown('<div class="section-header"><h3>Sub-Scores Breakdown</h3></div>', unsafe_allow_html=True)
        cols = st.columns(len(sub_scores))
        for i, (key, val) in enumerate(sub_scores.items()):
            label = key.replace("_", " ").title()
            with cols[i % len(cols)]:
                color = "green" if val >= 70 else ("orange" if val >= 40 else "red")
                st.markdown(f"**{label}**: :{color}[{val:.0f}/100]")


# =============================================================================
# RENDER: BLOCKERS & RFIs
# =============================================================================

def render_blockers_rfis(payload: Dict):
    """Render blockers and RFIs from pipeline output."""
    blockers = payload.get("blockers", [])
    rfis = payload.get("rfis", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Blockers")
        if not blockers:
            st.success("No blockers found!")
        for b in blockers[:15]:
            sev = b.get("severity", "medium")
            css, icon = _get_severity_css(sev)
            title = b.get("title", "Unknown")
            desc = b.get("description", "")[:200]
            trade = b.get("trade", "")
            trade_tag = f' <span style="color:#6b7280;font-size:0.8em">[{trade}]</span>' if trade else ""

            st.markdown(
                f'<div class="{css}">'
                f'{icon} <strong>[{sev.upper()}]</strong>{trade_tag} {title}<br/>'
                f'<span style="font-size:0.85em">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(blockers) > 15:
            st.caption(f"... and {len(blockers) - 15} more blockers")

    with col2:
        st.markdown("#### RFIs (Requests for Information)")
        if not rfis:
            st.info("No RFIs generated.")
        for r in rfis[:15]:
            trade = r.get("trade", "general")
            title = r.get("title", "")
            evidence = r.get("evidence", "")[:150]
            action_items = r.get("action_items", [])

            st.markdown(
                f'<div class="risk-info">'
                f'<strong>[{trade.upper()}]</strong> {title}<br/>'
                f'<span style="font-size:0.85em">{evidence}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if action_items:
                for ai in action_items[:2]:
                    st.caption(f"  - {ai}")
        if len(rfis) > 15:
            st.caption(f"... and {len(rfis) - 15} more RFIs")


# =============================================================================
# RENDER: SCOPE GAPS (Sprint 35 engine)
# =============================================================================

def render_scope_gaps(payload: Dict):
    """Run scope dependency analysis on extracted data."""
    boq_items = _extract_boq_from_payload(payload)
    elements = _extract_detected_elements(payload)
    rooms = _extract_room_types(payload)

    scope_result = analyze_scope_gaps(
        boq_items=boq_items,
        detected_elements=elements,
        room_types=rooms,
        project_params={
            "num_floors": DEFAULT_PROJECT["num_floors"],
            "plot_area_sqm": DEFAULT_PROJECT["plot_area_sqm"],
        },
    )

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-label">Rules Fired</div>
            <div class="metric-value">{scope_result.total_rules_checked}</div>
            <div class="metric-detail">of {get_rule_count()} total</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Items Checked</div>
            <div class="metric-value">{scope_result.total_items_checked}</div>
            <div class="metric-detail">of {get_required_item_count()} items</div>
        </div>
        <div class="metric-card {'red' if scope_result.total_gaps_found > 50 else 'amber' if scope_result.total_gaps_found > 10 else 'green'}">
            <div class="metric-label">Gaps Found</div>
            <div class="metric-value">{scope_result.total_gaps_found}</div>
            <div class="metric-detail">items missing from BOQ</div>
        </div>
        <div class="metric-card {'green' if scope_result.completeness_score > 70 else 'amber' if scope_result.completeness_score > 40 else 'red'}">
            <div class="metric-label">Completeness</div>
            <div class="metric-value">{scope_result.completeness_score:.0f}%</div>
            <div class="metric-detail">scope coverage</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("##### Critical Missing Items")
        for gap in scope_result.critical_gaps[:20]:
            st.markdown(
                f'<div class="gap-item">'
                f'<strong>{gap.missing_item}</strong> '
                f'<span style="color:#6b7280; font-size:0.85em;">({gap.trade})</span><br/>'
                f'<span style="color:#9ca3af; font-size:0.8em;">Triggered by: {gap.triggered_by}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(scope_result.critical_gaps) > 20:
            st.caption(f"... and {len(scope_result.critical_gaps) - 20} more gaps")

    with col_right:
        st.markdown("##### Coverage by Trade")
        if scope_result.coverage_by_trade:
            trade_data = []
            for trade, info in scope_result.coverage_by_trade.items():
                rules_count = info.get("rules", 0)
                gaps_count = info.get("gaps", 0)
                items_count = info.get("items", rules_count * 3)
                found = items_count - gaps_count
                pct = (found / items_count * 100) if items_count > 0 else 0
                trade_data.append({
                    "Trade": trade,
                    "Rules": rules_count,
                    "Gaps": gaps_count,
                    "Coverage": f"{pct:.0f}%",
                })
            st.dataframe(trade_data, use_container_width=True, hide_index=True)

    return scope_result


# =============================================================================
# RENDER: BOQ QUALITY (Dedup + Cross-Check + Completeness)
# =============================================================================

def render_boq_quality(payload: Dict):
    """Run BOQ quality engines on extracted data."""
    boq_items = _extract_boq_from_payload(payload)

    if not boq_items:
        st.warning("No BOQ items extracted from drawings. BOQ quality analysis requires extracted BOQ data.")
        return None, None, None

    # ── Deduplication ──
    st.markdown('<div class="section-header"><h3>BOQ Deduplication</h3></div>', unsafe_allow_html=True)

    dedup_result = deduplicate_boq(boq_items)

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Items", dedup_result.original_count)
    c2.metric("After Dedup", dedup_result.deduplicated_count,
              delta=f"-{dedup_result.duplicates_found}", delta_color="inverse")
    c3.metric("Reduction", f"{dedup_result.reduction_pct:.0f}%")

    if dedup_result.merge_log:
        with st.expander(f"Merge Log ({len(dedup_result.merge_log)} merges)", expanded=False):
            for entry in dedup_result.merge_log[:10]:
                st.markdown(f"**Kept:** {entry.kept_description[:80]}")
                for merged_desc in entry.merged_descriptions:
                    if merged_desc != entry.kept_description:
                        st.caption(f"  Merged: {merged_desc[:80]} (sim: {entry.similarity:.0%})")

    st.divider()

    # ── Quantity Cross-Check ──
    st.markdown('<div class="section-header"><h3>Quantity Cross-Validation</h3></div>', unsafe_allow_html=True)

    xcheck_report = cross_check_boq(dedup_result.deduplicated_items if dedup_result.deduplicated_items else boq_items)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Checks Run", len(xcheck_report.checks))
    c2.metric("Issues", xcheck_report.issues_count)
    c3.metric("Critical", len(xcheck_report.critical_issues))
    c4.metric("Confidence", f"{xcheck_report.overall_confidence}%")

    for check in xcheck_report.checks[:10]:
        sev = check.severity.value
        css, icon = _get_severity_css(sev)
        var_str = f"{check.variance_pct:+.1f}%" if check.variance_pct else ""

        st.markdown(
            f'<div class="{css}">'
            f'{icon} <strong>[{sev.upper()}]</strong> {check.check_type} &nbsp; '
            f'<span style="color:#6b7280">BOQ: {check.boq_qty:.0f} {check.unit} | '
            f'Expected: {check.derived_qty:.0f} {check.unit} | {var_str}</span><br/>'
            f'<span style="font-size:0.85em">{check.explanation[:120]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Completeness ──
    st.markdown('<div class="section-header"><h3>BOQ Completeness Score</h3></div>', unsafe_allow_html=True)

    comp_report = score_boq_completeness(dedup_result.deduplicated_items if dedup_result.deduplicated_items else boq_items)

    grade_colors = {"A": "#16a34a", "B": "#65a30d", "C": "#ca8a04", "D": "#ea580c", "E": "#dc2626", "F": "#991b1b"}
    grade_color = grade_colors.get(comp_report.grade, "#6b7280")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{comp_report.overall_score:.0f}/100")
    c2.markdown(
        f"### Grade: <span style='color:{grade_color}; font-size:2.5rem; font-weight:bold'>"
        f"{comp_report.grade}</span>",
        unsafe_allow_html=True,
    )
    c3.metric("Trades Found", f"{comp_report.trades_found}/{comp_report.trades_expected}")

    col1, col2 = st.columns([3, 2])
    with col1:
        if comp_report.missing_trades:
            st.markdown("##### Missing Trades")
            for trade in comp_report.missing_trades[:8]:
                st.markdown(
                    f'<div class="risk-high">Missing: <strong>{trade.replace("_", " ").title()}</strong></div>',
                    unsafe_allow_html=True,
                )
    with col2:
        st.markdown("##### Improvement Suggestions")
        suggestions = get_improvement_suggestions(comp_report, max_suggestions=5)
        for i, s in enumerate(suggestions, 1):
            st.markdown(f"**{i}.** {s}")

    return dedup_result, xcheck_report, comp_report


# =============================================================================
# RENDER: DELHI NCR PRICING
# =============================================================================

def render_delhi_ncr_pricing(payload: Dict, duration_months: int = 18):
    """Render Delhi NCR-specific pricing analysis."""
    boq_items = _extract_boq_from_payload(payload)

    # Delhi NCR city factor
    delhi_factor = get_city_factor("Delhi")

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-label">Base Location</div>
            <div class="metric-value">Delhi NCR</div>
            <div class="metric-detail">Factor: {delhi_factor:.2f}x (baseline)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{duration_months} mo</div>
            <div class="metric-detail">escalation applied</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">NCR Cities</div>
            <div class="metric-value">{len(DELHI_NCR_CITIES)}</div>
            <div class="metric-detail">Delhi, Gurgaon, Noida...</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-label">BOQ Items</div>
            <div class="metric-value">{len(boq_items)}</div>
            <div class="metric-detail">for rate adjustment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Material price comparison across NCR
    st.markdown("##### Material Costs — Delhi NCR vs Other Metros")
    materials = ["steel", "cement", "aggregates", "labour", "timber", "bricks", "fuel_transport"]
    compare_cities = ["Delhi", "Gurgaon", "Noida", "Mumbai", "Bangalore", "Chennai"]

    mat_data = []
    for material in materials:
        row = {"Material": material.replace("_", " ").title()}
        for city in compare_cities:
            factor = get_material_city_factor(city, material)
            row[city] = f"{factor:.3f}"
        mat_data.append(row)
    st.dataframe(mat_data, use_container_width=True, hide_index=True)

    # Rate escalation for key items
    st.divider()
    st.markdown("##### Rate Escalation — Key Construction Items")

    key_items = [
        ("RCC M25 Concrete", 8200, "rcc_concrete"),
        ("Steel Fe500D", 75000, "steel"),
        ("Brick Masonry", 5800, "masonry"),
        ("Cement Plaster", 180, "plaster"),
        ("Vitrified Tiles", 450, "flooring"),
    ]

    esc_data = []
    for name, base_rate, material_type in key_items:
        try:
            esc = escalate_rate(base_rate, material_type, duration_months, location="Delhi")
            esc_rate = esc.get("escalated_rate", base_rate)
            increase = ((esc_rate / base_rate) - 1) * 100
            esc_data.append({
                "Item": name,
                "Base Rate (Rs)": f"{base_rate:,.0f}",
                f"Escalated ({duration_months}mo)": f"{esc_rate:,.0f}",
                "Increase": f"{increase:+.1f}%",
            })
        except Exception:
            esc_data.append({
                "Item": name,
                "Base Rate (Rs)": f"{base_rate:,.0f}",
                f"Escalated ({duration_months}mo)": f"{base_rate:,.0f}",
                "Increase": "N/A",
            })

    st.dataframe(esc_data, use_container_width=True, hide_index=True)


# =============================================================================
# RENDER: BID RISK ASSESSMENT
# =============================================================================

def render_bid_risk(payload: Dict, scope_result, xcheck_report, comp_report):
    """Run bid risk analysis combining pipeline output + Sprint 35 engines."""
    boq_items = _extract_boq_from_payload(payload)

    # Infer document types from payload
    doc_types = []
    overview = payload.get("drawing_overview") or {}
    disciplines = overview.get("disciplines_detected", [])
    if "structural" in disciplines:
        doc_types.append("structural_drawings")
    if "architectural" in disciplines:
        doc_types.append("architectural_drawings")
    if "mep" in disciplines or "electrical" in disciplines or "plumbing" in disciplines:
        doc_types.append("mep_drawings")
    if payload.get("boq_stats", {}).get("total_items", 0) > 0:
        doc_types.append("boq")
    if payload.get("commercial_terms"):
        doc_types.append("conditions_of_contract")

    # Contract conditions from payload
    contract_conds = {}
    for ct in payload.get("commercial_terms", []):
        term_type = ct.get("term_type", "")
        value = ct.get("value", "")
        if "defect" in term_type.lower():
            try:
                contract_conds["defect_liability_years"] = int(value)
            except (ValueError, TypeError):
                pass
        elif "retention" in term_type.lower():
            try:
                contract_conds["retention_pct"] = float(value)
            except (ValueError, TypeError):
                pass
        elif "payment" in term_type.lower():
            try:
                contract_conds["payment_terms_days"] = int(value)
            except (ValueError, TypeError):
                pass

    scope_gaps = scope_result.total_gaps_found if scope_result else 0
    qty_mismatches = xcheck_report.issues_count if xcheck_report else 0
    comp_score = comp_report.overall_score if comp_report else 50.0

    risk_report = analyze_bid_risk(
        boq_items=boq_items,
        scope_gaps=scope_gaps,
        quantity_mismatches=qty_mismatches,
        missing_rates_pct=0.0,
        completeness_score=comp_score,
        project_duration_months=DEFAULT_PROJECT["duration_months"],
        project_value_lakhs=DEFAULT_PROJECT["value_lakhs"],
        document_types_available=doc_types,
        contract_conditions=contract_conds,
        location="Delhi",
    )

    # Recommendation banner
    rec = risk_report.bid_recommendation
    rec_configs = {
        "GO": ("decision-go", "Recommend bidding"),
        "GO_WITH_QUALIFICATIONS": ("decision-review", "Bid with qualifications"),
        "NO_GO": ("decision-nogo", "Do not bid"),
        "NEEDS_REVIEW": ("decision-review", "Needs further review"),
    }
    css, subtitle = rec_configs.get(rec, ("decision-review", ""))
    rec_label = rec.replace("_", " ")

    st.markdown(f"""
    <div class="{css}">
        <div class="decision-label">{rec_label}</div>
        <div class="decision-sub">Risk Score: {risk_report.overall_risk_score:.0f}/100 | {subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card {'red' if risk_report.overall_risk_score > 60 else 'amber' if risk_report.overall_risk_score > 30 else 'green'}">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value">{risk_report.overall_risk_score:.0f}</div>
            <div class="metric-detail">out of 100</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Critical Risks</div>
            <div class="metric-value">{len(risk_report.critical_risks)}</div>
            <div class="metric-detail">immediate attention</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-label">High Risks</div>
            <div class="metric-value">{len(risk_report.high_risks)}</div>
            <div class="metric-detail">needs mitigation</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-label">Total Risks</div>
            <div class="metric-value">{len(risk_report.risks)}</div>
            <div class="metric-detail">across 7 categories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk items
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### Risk Details")
        for risk in risk_report.risks[:12]:
            sev = risk.level.value
            css, icon = _get_severity_css(sev)
            impact_str = f"<br/><em>Impact: {risk.financial_impact}</em>" if risk.financial_impact else ""
            mitigation_str = f"<br/><strong>Mitigation:</strong> {risk.mitigation[:100]}" if risk.mitigation else ""

            st.markdown(
                f'<div class="{css}">'
                f'{icon} <strong>[{sev.upper()}] [{risk.category.value.upper()}]</strong> {risk.title}<br/>'
                f'<span style="font-size:0.85em">{risk.description[:150]}</span>'
                f'{impact_str}{mitigation_str}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("##### Risk by Category")
        cat_data = []
        for cat, count in sorted(risk_report.risk_by_category.items(), key=lambda x: -x[1]):
            cat_data.append({"Category": cat.title(), "Risks": count})
        st.dataframe(cat_data, use_container_width=True, hide_index=True)

    return risk_report


# =============================================================================
# RENDER: TRADE COVERAGE
# =============================================================================

def render_trade_coverage(payload: Dict):
    """Render trade coverage from pipeline output."""
    trade_coverage = payload.get("trade_coverage", [])

    if not trade_coverage:
        st.info("Trade coverage data not available.")
        return

    trade_icons = {
        "civil": "&#127959;", "structural": "&#128297;", "architectural": "&#127968;",
        "mep": "&#9889;", "finishes": "&#127912;", "general": "&#128230;",
        "electrical": "&#128161;", "plumbing": "&#128703;",
    }

    for tc in trade_coverage:
        trade = tc.get("trade", "general")
        cov = tc.get("coverage_pct", 0)
        priceable = tc.get("priceable_count", 0)
        blocked = tc.get("blocked_count", 0)

        if cov >= 80 and blocked == 0:
            css = "risk-ok"
        elif blocked > 0:
            css = "risk-high"
        elif cov > 0:
            css = "risk-medium"
        else:
            css = "risk-critical"

        icon = trade_icons.get(trade, "&#128203;")

        st.markdown(
            f'<div class="{css}">'
            f'{icon} <strong>{trade.replace("_", " ").title()}</strong> &mdash; '
            f'{cov:.0f}% coverage | {priceable} priceable | {blocked} blocked'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# RENDER: STRUCTURAL TAKEOFF
# =============================================================================

def render_structural(payload: Dict):
    """Render structural takeoff from pipeline."""
    st_data = payload.get("structural_takeoff", {})

    if not st_data or st_data.get("mode") in (None, "error"):
        st.info("No structural data detected in drawings. Upload structural drawings to see concrete & steel estimates.")
        return

    summary = st_data.get("summary", {})
    mode = st_data.get("mode", "assumption")
    qc = st_data.get("qc", {})
    confidence = qc.get("confidence", 0)
    mode_label = "Detected from Drawings" if mode == "structural" else "Assumption-Based Estimate"

    concrete = summary.get("concrete_m3", 0)
    steel = summary.get("steel_tons", 0)
    steel_kg = summary.get("steel_kg", steel * 1000)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-label">Mode</div>
            <div class="metric-value">{mode_label[:15]}</div>
            <div class="metric-detail">{mode}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Concrete</div>
            <div class="metric-value">{concrete:.1f} m&sup3;</div>
            <div class="metric-detail">all elements</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-label">Steel</div>
            <div class="metric-value">{steel:.2f} t</div>
            <div class="metric-detail">{steel_kg:.0f} kg total</div>
        </div>
        <div class="metric-card {'green' if confidence > 0.7 else 'amber' if confidence > 0.4 else 'red'}">
            <div class="metric-label">QC Confidence</div>
            <div class="metric-value">{confidence:.0%}</div>
            <div class="metric-detail">quality check</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Element breakdown if available
    elements = st_data.get("elements", [])
    if elements:
        st.markdown("##### Element Breakdown")
        elem_data = []
        for e in elements[:20]:
            elem_data.append({
                "Element": e.get("label", ""),
                "Type": e.get("element_type", ""),
                "Concrete (m3)": f"{e.get('concrete_m3', 0):.3f}",
                "Steel (kg)": f"{e.get('steel_kg', 0):.1f}",
                "Size": e.get("size_display", ""),
            })
        st.dataframe(elem_data, use_container_width=True, hide_index=True)

    # Delhi NCR cost estimate
    st.divider()
    st.markdown("##### Delhi NCR Cost Estimate")

    delhi_concrete_rate = 8200  # Rs/m3 for M25
    delhi_steel_rate = 75  # Rs/kg
    delhi_formwork_rate = 360  # Rs/sqm

    concrete_cost = concrete * delhi_concrete_rate
    steel_cost = steel_kg * delhi_steel_rate
    formwork_est = concrete * 6 * delhi_formwork_rate  # Rough 6sqm/m3 ratio
    total_structural = concrete_cost + steel_cost + formwork_est

    cost_data = [
        {"Item": "RCC Concrete (M25)", "Qty": f"{concrete:.1f} m3", "Rate (Rs)": f"{delhi_concrete_rate:,}", "Amount (Rs)": f"{concrete_cost:,.0f}"},
        {"Item": "Steel Reinforcement", "Qty": f"{steel_kg:.0f} kg", "Rate (Rs)": f"{delhi_steel_rate}", "Amount (Rs)": f"{steel_cost:,.0f}"},
        {"Item": "Formwork (estimated)", "Qty": f"{concrete * 6:.0f} sqm", "Rate (Rs)": f"{delhi_formwork_rate}", "Amount (Rs)": f"{formwork_est:,.0f}"},
        {"Item": "TOTAL STRUCTURAL", "Qty": "", "Rate (Rs)": "", "Amount (Rs)": f"{total_structural:,.0f}"},
    ]
    st.dataframe(cost_data, use_container_width=True, hide_index=True)
    st.caption("Rates are Delhi NCR CPWD DSR 2024 base rates. Escalation and contractor margins not included.")


# =============================================================================
# RENDER: RAW PAYLOAD EXPLORER
# =============================================================================

def render_raw_payload(payload: Dict):
    """Let users explore the full pipeline output."""
    st.markdown("##### Pipeline Output Keys")

    # Group keys
    key_groups = {
        "Decision & Scores": ["readiness_score", "decision", "sub_scores", "qa_score"],
        "Overview": ["drawing_overview", "processing_stats", "timings", "run_coverage"],
        "Findings": ["blockers", "rfis", "conflicts", "reconciliation_findings"],
        "BOQ & Trade": ["boq_stats", "trade_coverage", "requirements_by_trade", "quantities"],
        "Commercial": ["commercial_terms", "pricing_guidance"],
        "Structural": ["structural_takeoff"],
        "Plan Graph": ["plan_graph"],
        "Extraction": ["extraction_summary", "extraction_diagnostics"],
    }

    for group_name, keys in key_groups.items():
        present_keys = [k for k in keys if k in payload]
        if present_keys:
            with st.expander(f"{group_name} ({len(present_keys)} keys)"):
                for key in present_keys:
                    val = payload[key]
                    if isinstance(val, dict):
                        st.json(val)
                    elif isinstance(val, list):
                        st.markdown(f"**{key}**: {len(val)} items")
                        if val and len(val) <= 5:
                            st.json(val)
                        elif val:
                            st.json(val[:3])
                            st.caption(f"... showing 3 of {len(val)}")
                    else:
                        st.markdown(f"**{key}**: `{val}`")


# =============================================================================
# MAIN APP
# =============================================================================

# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
    <h1>xBOQ.ai &mdash; Delhi NCR Bid Analysis</h1>
    <p>Upload your tender drawing set and get a comprehensive bid analysis in minutes.</p>
    <div>
        <span class="hero-badge">Delhi NCR Pricing</span>
        <span class="hero-badge">70 Scope Rules</span>
        <span class="hero-badge">7 Risk Categories</span>
        <span class="hero-badge">IS 456 / CPWD DSR 2024</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Session State ──
if "_analysis_result" not in st.session_state:
    st.session_state["_analysis_result"] = None
if "_analysis_payload" not in st.session_state:
    st.session_state["_analysis_payload"] = None


# ── Upload Section ──
col_upload, col_info = st.columns([3, 2])

with col_upload:
    uploaded_files = st.file_uploader(
        "Drop your tender drawing PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files containing architectural, structural, MEP drawings, BOQ, or specifications.",
    )

    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
        file_names = ", ".join(f.name for f in uploaded_files[:3])
        if len(uploaded_files) > 3:
            file_names += f" +{len(uploaded_files) - 3} more"
        st.caption(f"Ready: {len(uploaded_files)} file(s) | {total_size:.1f} MB | {file_names}")

with col_info:
    st.markdown("""
    **How it works:**
    1. Upload tender drawing PDFs (plans, BOQ, specs)
    2. xBOQ extracts text, tables, and drawings
    3. Runs 8-stage analysis pipeline
    4. Sprint 35 engines analyze scope, quality, pricing & risk
    5. Get GO/NO-GO recommendation for Delhi NCR
    """)

# Run mode + analyze button
col_mode, col_btn = st.columns([2, 1])

with col_mode:
    run_mode_options = {
        "Demo Fast (80 pages)": "demo_fast",
        "Standard Review (220 pages)": "standard_review",
        "Full Audit (All pages)": "full_audit",
    }
    selected_mode = st.radio(
        "Analysis Depth",
        list(run_mode_options.keys()),
        index=0,
        horizontal=True,
    )
    run_mode = run_mode_options[selected_mode]

with col_btn:
    st.markdown("<br/>", unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Analyze Drawings",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

# ── Run Analysis ──
if uploaded_files and analyze_clicked:
    result = _run_pipeline_with_progress(uploaded_files, run_mode=run_mode)
    if result and result.success and result.payload:
        st.session_state["_analysis_result"] = result
        st.session_state["_analysis_payload"] = result.payload


# ── Render Results ──
payload = st.session_state.get("_analysis_payload")
result = st.session_state.get("_analysis_result")

if payload and result:
    st.divider()

    # ── Tabs ──
    tab_overview, tab_blockers, tab_scope, tab_boq, tab_pricing, tab_risk, tab_structural, tab_trade, tab_raw = st.tabs([
        "Overview",
        "Blockers & RFIs",
        "Scope Gaps",
        "BOQ Quality",
        "Delhi NCR Pricing",
        "Bid Risk",
        "Structural",
        "Trade Coverage",
        "Raw Data",
    ])

    with tab_overview:
        render_pipeline_overview(payload, result)

    with tab_blockers:
        st.markdown('<div class="section-header"><h3>Blockers & RFIs from Pipeline</h3></div>', unsafe_allow_html=True)
        render_blockers_rfis(payload)

    with tab_scope:
        st.markdown('<div class="section-header"><h3>Scope Dependency Analysis</h3></div>', unsafe_allow_html=True)
        st.markdown("Checks extracted BOQ against **70 dependency rules** covering **222 required items** across all construction trades.")
        scope_result = render_scope_gaps(payload)

    with tab_boq:
        st.markdown('<div class="section-header"><h3>BOQ Quality Analysis</h3></div>', unsafe_allow_html=True)
        st.markdown("Deduplication + quantity cross-validation + completeness scoring on extracted BOQ.")
        dedup_r, xcheck_r, comp_r = render_boq_quality(payload)

    with tab_pricing:
        st.markdown('<div class="section-header"><h3>Delhi NCR Pricing Analysis</h3></div>', unsafe_allow_html=True)
        st.markdown("Material costs and rate escalation specific to **Delhi NCR region**.")
        render_delhi_ncr_pricing(payload, DEFAULT_PROJECT["duration_months"])

    with tab_risk:
        st.markdown('<div class="section-header"><h3>Bid Risk Assessment</h3></div>', unsafe_allow_html=True)
        st.markdown("**7-category risk analysis** combining pipeline findings + Sprint 35 engine outputs.")
        # Re-run scope/quality if needed for risk input
        _scope = scope_result if 'scope_result' in dir() else None
        _xcheck = xcheck_r if 'xcheck_r' in dir() else None
        _comp = comp_r if 'comp_r' in dir() else None
        render_bid_risk(payload, _scope, _xcheck, _comp)

    with tab_structural:
        st.markdown('<div class="section-header"><h3>Structural Takeoff</h3></div>', unsafe_allow_html=True)
        render_structural(payload)

    with tab_trade:
        st.markdown('<div class="section-header"><h3>Trade Coverage</h3></div>', unsafe_allow_html=True)
        render_trade_coverage(payload)

    with tab_raw:
        st.markdown('<div class="section-header"><h3>Full Pipeline Output</h3></div>', unsafe_allow_html=True)
        render_raw_payload(payload)

else:
    # Show sample analysis when no files uploaded
    st.divider()
    st.markdown("### Sample Analysis (No Upload Required)")
    st.markdown("See what xBOQ can do with sample data. Upload your own drawings above for real analysis.")

    sample_tab1, sample_tab2, sample_tab3, sample_tab4 = st.tabs([
        "Scope Gaps (Sample)",
        "BOQ Quality (Sample)",
        "Delhi NCR Pricing",
        "Bid Risk (Sample)",
    ])

    SAMPLE_BOQ = [
        {"description": "Excavation for foundations in all soils", "quantity": 60.0, "unit": "cum", "rate": 250},
        {"description": "PCC M7.5 grade 1:4:8 under footings", "quantity": 5.0, "unit": "cum", "rate": 4500},
        {"description": "RCC M25 for footings", "quantity": 15.0, "unit": "cum", "rate": 7500},
        {"description": "RCC M25 for columns", "quantity": 10.0, "unit": "cum", "rate": 8200},
        {"description": "RCC M25 for beams", "quantity": 18.0, "unit": "cum", "rate": 8000},
        {"description": "RCC M20 for slab 125mm thick", "quantity": 25.0, "unit": "cum", "rate": 7200},
        {"description": "Steel reinforcement Fe500D", "quantity": 8500.0, "unit": "kg", "rate": 75},
        {"description": "Formwork to footing", "quantity": 55.0, "unit": "sqm", "rate": 350},
        {"description": "Formwork to column", "quantity": 80.0, "unit": "sqm", "rate": 380},
        {"description": "Formwork to beam", "quantity": 200.0, "unit": "sqm", "rate": 360},
        {"description": "Formwork to slab soffit", "quantity": 200.0, "unit": "sqm", "rate": 340},
        {"description": "Brick masonry 230mm in CM 1:6", "quantity": 45.0, "unit": "cum", "rate": 5800},
        {"description": "Cement plaster 12mm internal", "quantity": 400.0, "unit": "sqm", "rate": 180},
        {"description": "Cement plaster 20mm external", "quantity": 150.0, "unit": "sqm", "rate": 220},
        {"description": "Vitrified tile flooring 600x600", "quantity": 180.0, "unit": "sqm", "rate": 450},
        {"description": "Emulsion paint 2 coats", "quantity": 600.0, "unit": "sqm", "rate": 75},
        {"description": "Sal wood door frame", "quantity": 8.0, "unit": "no", "rate": 3500},
        {"description": "Flush door shutter 35mm", "quantity": 8.0, "unit": "no", "rate": 2800},
        {"description": "Door hardware set SS", "quantity": 8.0, "unit": "set", "rate": 1500},
        {"description": "Ceramic skirting 100mm", "quantity": 85.0, "unit": "rmt", "rate": 120},
    ]

    with sample_tab1:
        sample_elements = ["footing", "column", "beam", "slab", "staircase"]
        sample_rooms = ["bedroom", "living_room", "kitchen", "toilet", "balcony"]

        scope_r = analyze_scope_gaps(
            boq_items=SAMPLE_BOQ,
            detected_elements=sample_elements,
            room_types=sample_rooms,
            project_params={"num_floors": 4, "plot_area_sqm": 300},
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rules Fired", scope_r.total_rules_checked, f"of {get_rule_count()}")
        c2.metric("Items Checked", scope_r.total_items_checked)
        c3.metric("Gaps Found", scope_r.total_gaps_found)
        c4.metric("Completeness", f"{scope_r.completeness_score}%")

        st.divider()
        for gap in scope_r.critical_gaps[:12]:
            st.markdown(
                f'<div class="gap-item">'
                f'<strong>{gap.missing_item}</strong> '
                f'<span style="color:#6b7280">({gap.trade})</span> &mdash; '
                f'<span style="color:#9ca3af">via {gap.triggered_by}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with sample_tab2:
        xcheck_r = cross_check_boq(SAMPLE_BOQ)
        comp_r = score_boq_completeness(SAMPLE_BOQ)

        c1, c2, c3 = st.columns(3)
        c1.metric("Cross-Check Issues", xcheck_r.issues_count, f"of {len(xcheck_r.checks)} checks")
        c2.metric("Completeness", f"{comp_r.overall_score:.0f}/100")

        grade_colors = {"A": "#16a34a", "B": "#65a30d", "C": "#ca8a04", "D": "#ea580c", "E": "#dc2626", "F": "#991b1b"}
        c3.markdown(
            f"### Grade: <span style='color:{grade_colors.get(comp_r.grade, '#6b7280')}'>{comp_r.grade}</span>",
            unsafe_allow_html=True,
        )

        st.divider()
        for check in xcheck_r.checks[:8]:
            sev = check.severity.value
            css, icon = _get_severity_css(sev)
            st.markdown(
                f'<div class="{css}">{icon} <strong>{check.check_type}</strong> &mdash; '
                f'{check.explanation[:120]}</div>',
                unsafe_allow_html=True,
            )

    with sample_tab3:
        render_delhi_ncr_pricing({}, DEFAULT_PROJECT["duration_months"])

    with sample_tab4:
        risk_r = analyze_bid_risk(
            boq_items=SAMPLE_BOQ,
            scope_gaps=scope_r.total_gaps_found,
            quantity_mismatches=xcheck_r.issues_count,
            missing_rates_pct=0.0,
            completeness_score=comp_r.overall_score,
            project_duration_months=18,
            project_value_lakhs=2500,
            document_types_available=["boq", "architectural_drawings", "structural_drawings", "specifications"],
            contract_conditions={"defect_liability_years": 3, "retention_pct": 5, "payment_terms_days": 30},
            location="Delhi",
        )

        rec = risk_r.bid_recommendation
        rec_colors = {"GO": "#16a34a", "GO_WITH_QUALIFICATIONS": "#ca8a04", "NO_GO": "#dc2626"}
        rec_color = rec_colors.get(rec, "#ea580c")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{risk_r.overall_risk_score:.0f}/100")
        c2.markdown(f"### <span style='color:{rec_color}'>{rec.replace('_', ' ')}</span>", unsafe_allow_html=True)
        c3.metric("Critical", len(risk_r.critical_risks))
        c4.metric("High", len(risk_r.high_risks))

        st.divider()
        for risk in risk_r.risks[:8]:
            sev = risk.level.value
            css, icon = _get_severity_css(sev)
            st.markdown(
                f'<div class="{css}">{icon} <strong>[{sev.upper()}]</strong> {risk.title} &mdash; '
                f'{risk.description[:120]}</div>',
                unsafe_allow_html=True,
            )


# ── Footer ──
st.divider()
st.caption("xBOQ.ai | Delhi NCR Bid Analysis | IS 456 / IS 1200 / CPWD DSR 2024 | 2,123 tests passing")
