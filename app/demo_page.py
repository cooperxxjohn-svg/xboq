"""
XBOQ Demo - Pre-Bid Scope & Risk Check

Comprehensive Bid Readiness Report with 7 sections:
1. Executive Summary
2. Missing Dependencies (detailed with evidence)
3. Flagged Areas / Risks
4. RFIs Generated (grouped, exportable)
5. Trade Coverage & Priceability
6. Assumptions/Exclusions
7. Drawing Set Overview / Audit Trail
"""

import streamlit as st
import json
import sys
import io
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exports.pricing_readiness import get_pricing_readiness_buffer
from exports.rfi_pack import get_rfi_pack_buffer
from exports.bid_packet import get_bid_packet_buffer

# Try to import report builder
try:
    from reports.bid_readiness_report import (
        build_report_data, export_report_bundle, ReportData,
        generate_report_html, generate_rfi_csv, generate_rfi_emails,
        generate_pricing_readiness_csv,
    )
    HAS_REPORT_BUILDER = True
except ImportError:
    HAS_REPORT_BUILDER = False

# Page config
st.set_page_config(
    page_title="XBOQ - Pre-Bid Scope Check",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: linear-gradient(180deg, rgb(0,0,0) 0%, rgb(10,10,12) 100%); }
    .main .block-container { max-width: 1100px; padding: 2rem 1rem; }
    h1, h2, h3, p, span, div { font-family: 'Inter', sans-serif !important; }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        color: white !important; border: none !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
    }
    .decision-badge {
        display: inline-block; padding: 0.5rem 1.5rem; border-radius: 8px;
        font-size: 1.5rem; font-weight: 700;
    }
    .badge-nogo { background: #fee2e2; color: #dc2626; }
    .badge-review { background: #fef3c7; color: #d97706; }
    .badge-go { background: #dcfce7; color: #16a34a; }
    .sub-score-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
    .sub-score { background: rgba(255,255,255,0.05); padding: 0.75rem 1rem;
                 border-radius: 8px; text-align: center; }
    .sub-score-label { font-size: 0.8rem; color: #999; }
    .sub-score-value { font-size: 1.5rem; font-weight: 700; }
    .reasons-box { background: rgba(220,38,38,0.1); padding: 1rem;
                   border-radius: 8px; border-left: 4px solid #dc2626; }
    .fixes-box { background: rgba(22,163,74,0.1); padding: 1rem;
                 border-radius: 8px; border-left: 4px solid #16a34a; }
    .dep-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .severity-critical, .severity-high { color: #ef4444; }
    .severity-medium { color: #f59e0b; }
    .severity-low { color: #22c55e; }
</style>
""", unsafe_allow_html=True)


def load_demo_results(project_id: str) -> dict:
    """Load results from output directory."""
    base_path = Path(__file__).parent.parent / "out" / project_id
    results = {
        "loaded": False, "rfis": [], "trade_summary": {}, "bid_gate": {},
        "metrics": {}, "critical_blockers": [], "deep_analysis": None, "plan_graph": None,
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

    # Load plan graph if available
    graph_path = base_path / "plan_graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            results["plan_graph"] = json.load(f)

    results["loaded"] = len(results["rfis"]) > 0 or results["bid_gate"]
    return results


def render_evidence_expander(evidence: dict, title: str = "Why?"):
    """Render an expandable evidence section."""
    if not evidence:
        return
    with st.expander(f"📋 {title}"):
        pages = evidence.get("pages", [])
        if pages:
            page_nums = [str(p + 1) for p in pages[:10]]
            st.markdown(f"**Found on pages:** {', '.join(page_nums)}")
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
        confidence = evidence.get("confidence", 0)
        if confidence:
            confidence_pct = int(confidence * 100)
            confidence_text = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
            st.caption(f"Confidence: {confidence_text} ({confidence_pct}%)")
            if confidence < 0.5:
                st.warning("⚠️ Low confidence - verify manually")


# =============================================================================
# SECTION RENDERERS
# =============================================================================

def render_section_1_summary(report: dict):
    """Render Section 1: Executive Summary."""
    summary = report.get("executive_summary", {})
    decision = summary.get("decision", "NO-GO")
    score = summary.get("readiness_score", 0)
    sub_scores = summary.get("sub_scores", {})

    # Decision Badge
    badge_class = "badge-nogo" if decision == "NO-GO" else ("badge-review" if decision in ["REVIEW", "CONDITIONAL"] else "badge-go")
    st.markdown(f'<div class="decision-badge {badge_class}">{decision}</div>', unsafe_allow_html=True)
    st.markdown(f"## Bid Readiness Score: {score}/100")

    # Sub-scores
    st.markdown('<div class="sub-score-grid">', unsafe_allow_html=True)
    cols = st.columns(4)
    score_labels = [("Coverage", "coverage"), ("Measurement", "measurement"),
                    ("Completeness", "completeness"), ("Blockers", "blocker")]
    for col, (label, key) in zip(cols, score_labels):
        val = sub_scores.get(key, 0)
        col.metric(label, f"{val}/100")
    st.markdown('</div>', unsafe_allow_html=True)

    # Top 3 Reasons
    reasons = summary.get("top_3_reasons", [])
    if reasons:
        st.markdown("### Top Reasons for " + decision)
        st.markdown('<div class="reasons-box">', unsafe_allow_html=True)
        for i, reason in enumerate(reasons, 1):
            st.markdown(f"**{i}.** {reason}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Top 3 Fixes
    fixes = summary.get("top_3_fixes", [])
    if fixes:
        st.markdown("### What You Can Do Next")
        st.markdown('<div class="fixes-box">', unsafe_allow_html=True)
        for fix in fixes:
            delta = fix.get("score_delta", 0)
            action = fix.get("action", "")
            desc = fix.get("description", "")
            st.markdown(f"✅ **{action}** (+{delta} pts) — {desc}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_section_2_dependencies(report: dict):
    """Render Section 2: Missing Dependencies."""
    deps = report.get("missing_dependencies", [])
    st.markdown(f"## Missing Dependencies ({len(deps)})")
    st.caption("These are blocking accurate pricing. Each shows what's detected, what's missing, and how to fix.")

    for dep in deps:
        with st.container():
            st.markdown(f"### {dep.get('id', 'DEP')}: {dep.get('dependency_type', 'Unknown')}")

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Status:** {dep.get('status', 'missing').upper()}")
            col2.markdown(f"**Trade:** {dep.get('impact_trade', 'general').title()}")
            col3.markdown(f"**Bid Impact:** {dep.get('impact_bid', '').replace('_', ' ').title()}")

            st.markdown(dep.get("why_needed", ""))

            # Evidence
            evidence = dep.get("evidence", {})
            if evidence:
                render_evidence_expander(evidence, "Evidence Details")

            # Risk & Score
            cost_risk = dep.get("cost_risk", "medium")
            schedule_risk = dep.get("schedule_risk", "low")
            score_delta = dep.get("score_delta", 0)
            st.markdown(f"**Risk:** Cost: {cost_risk.upper()} | Schedule: {schedule_risk.upper()} | **Score if fixed: +{score_delta} pts**")

            # Fix options
            fixes = dep.get("fix_options", [])
            if fixes:
                with st.expander("🔧 How to Fix"):
                    for i, fix in enumerate(fixes, 1):
                        st.markdown(f"{i}. {fix}")

            st.markdown("---")


def render_section_3_flagged(report: dict):
    """Render Section 3: Flagged Areas / Risks."""
    flags = report.get("flagged_areas", [])
    flag_summary = report.get("flag_summary", {})

    st.markdown(f"## Flagged Areas ({len(flags)})")

    # Summary table
    by_severity = flag_summary.get("by_severity", {})
    if by_severity:
        cols = st.columns(4)
        cols[0].metric("Total Flags", flag_summary.get("total_flags", 0))
        cols[1].metric("Critical/High", by_severity.get("critical", 0) + by_severity.get("high", 0))
        cols[2].metric("Medium", by_severity.get("medium", 0))
        cols[3].metric("Low", by_severity.get("low", 0))

    st.markdown("---")

    for flag in flags[:10]:
        severity = flag.get("severity", "medium")
        severity_class = f"severity-{severity}"
        st.markdown(f'<div class="dep-card">', unsafe_allow_html=True)
        st.markdown(f"### <span class='{severity_class}'>{flag.get('id', 'FLAG')}</span>: {flag.get('title', 'Unknown')}", unsafe_allow_html=True)
        st.markdown(f"**Trade:** {flag.get('trade', 'general').title()} | **Severity:** {severity.upper()}")
        st.markdown(flag.get("what_flagged", ""))
        st.caption(flag.get("why_flagged", ""))

        pages = flag.get("evidence_pages", [])
        if pages:
            st.caption(f"Evidence pages: {', '.join(map(str, pages[:5]))}")

        st.markdown(f"**Recommended Action:** {flag.get('recommended_action', 'Generate RFI')}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_section_4_rfis(report: dict):
    """Render Section 4: RFIs Generated."""
    rfis = report.get("rfis", [])
    rfi_summary = report.get("rfi_summary", {})
    rfis_by_trade = report.get("rfis_by_trade", {})

    st.markdown(f"## RFIs Generated ({rfi_summary.get('total_rfis', len(rfis))})")

    # Summary
    cols = st.columns(4)
    cols[0].metric("Total RFIs", rfi_summary.get("total_rfis", len(rfis)))
    cols[1].metric("Critical", rfi_summary.get("critical_rfis", 0))
    cols[2].metric("High Priority", rfi_summary.get("high_rfis", 0))
    cols[3].metric("Trades Affected", len(rfi_summary.get("trades_affected", [])))

    # Top 5 by impact
    top_5 = rfi_summary.get("top_5_by_impact", [])
    if top_5:
        st.markdown("### Top 5 by Impact")
        for i, q in enumerate(top_5, 1):
            st.markdown(f"**{i}.** {q}")

    st.markdown("---")

    # Grouped by trade
    for trade, trade_rfis in rfis_by_trade.items():
        with st.expander(f"📁 {trade.title()} ({len(trade_rfis)} RFIs)"):
            for rfi in trade_rfis[:10]:
                priority = rfi.get("priority", "medium")
                priority_icon = "🔴" if priority in ["high", "critical"] else "🟡" if priority == "medium" else "🟢"
                st.markdown(f"**{priority_icon} {rfi.get('id', 'RFI')}:** {rfi.get('question', 'Unknown')[:100]}")
                st.caption(rfi.get("why_it_matters", "")[:150])

                evidence = rfi.get("evidence", {})
                if evidence and (evidence.get("pages") or evidence.get("detected_entities")):
                    render_evidence_expander(evidence, "Evidence")

                resolution = rfi.get("suggested_resolution", "")
                if resolution:
                    st.markdown(f"✅ **Suggested Resolution:** {resolution}")
                st.markdown("")

    # Export buttons
    st.markdown("### Export RFI Pack")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_content = _generate_rfi_csv_from_report(rfis)
        st.download_button("📊 Download CSV", csv_content, "rfi_tracker.csv", "text/csv", use_container_width=True)
    with col2:
        email_content = _generate_rfi_emails_from_report(rfis)
        st.download_button("✉️ Email Drafts", email_content, "rfi_emails.txt", "text/plain", use_container_width=True)


def render_section_5_coverage(report: dict):
    """Render Section 5: Trade Coverage & Priceability."""
    coverage = report.get("trade_coverage", [])
    priceable = report.get("priceable_categories", [])
    blocked = report.get("blocked_categories", [])

    st.markdown("## Trade Coverage & Priceability")

    # Table
    if coverage:
        table_data = []
        for tc in coverage:
            table_data.append({
                "Trade": tc.get("trade", "").title(),
                "Coverage": f"{tc.get('coverage_pct', 0):.0f}%",
                "Priceable": tc.get("priceable_count", 0),
                "Blocked": tc.get("blocked_count", 0),
                "Confidence": tc.get("confidence", "low").title(),
                "Cost Risk": tc.get("cost_risk", "medium").title(),
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ What Can Be Priced Now")
        if priceable:
            for item in priceable:
                st.markdown(f"- {item}")
        else:
            st.caption("No categories fully priceable yet")

    with col2:
        st.markdown("### ⛔ What Is Blocked")
        if blocked:
            for item in blocked:
                st.markdown(f"- {item}")
        else:
            st.caption("No blocked categories")


def render_section_6_assumptions(report: dict):
    """Render Section 6: Assumptions/Exclusions."""
    assumptions = report.get("assumptions", [])

    st.markdown(f"## Suggested Assumptions ({len(assumptions)})")
    st.caption("These assumptions can be included in your bid if the missing information is not provided.")

    for asmp in assumptions:
        with st.container():
            st.markdown(f"### {asmp.get('id', 'ASMP')}: {asmp.get('title', 'Assumption')}")
            st.markdown(f"**Draft Text:**")
            st.info(asmp.get("text", ""))

            impact = asmp.get("impact_if_wrong", "")
            if impact:
                st.warning(f"⚠️ **If wrong:** {impact}")

            linked = asmp.get("linked_blocker_ids", [])
            if linked:
                st.caption(f"Linked to: {', '.join(linked)}")

            st.markdown("---")


def render_section_7_audit(report: dict):
    """Render Section 7: Drawing Set Overview / Audit Trail."""
    overview = report.get("drawing_set_overview", {})

    st.markdown("## Drawing Set Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", overview.get("total_pages", 0))
    col2.metric("Pages with Scale", overview.get("pages_with_scale", 0))
    col3.metric("Pages without Scale", overview.get("pages_without_scale", 0))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rooms Detected", overview.get("rooms_detected", 0))
    col1.metric("Door Tags", overview.get("door_tags_count", 0))
    col2.metric("Window Tags", overview.get("window_tags_count", 0))

    # Disciplines
    disciplines = overview.get("disciplines_detected", [])
    if disciplines:
        st.markdown(f"**Disciplines Found:** {', '.join(disciplines)}")
    else:
        st.markdown("**Disciplines Found:** None detected")

    # Sheet Types
    sheet_types = overview.get("sheet_types", {})
    if sheet_types:
        st.markdown("**Sheet Types:**")
        for stype, count in sheet_types.items():
            st.caption(f"- {stype}: {count}")

    # Schedules
    schedules = overview.get("schedules_detected", [])
    if schedules:
        st.markdown("**Schedules Detected:**")
        for s in schedules:
            st.markdown(f"- {s.get('type', 'unknown').replace('_', ' ').title()}: {s.get('status', 'unknown')}")
    else:
        st.warning("No schedules detected in drawing set")

    # Room names sample
    rooms = overview.get("room_names", [])
    if rooms:
        st.markdown(f"**Room Types:** {', '.join(rooms[:10])}")

    # Download buttons
    st.markdown("### Debug Downloads")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Report JSON",
            json.dumps(report, indent=2, default=str),
            "bid_readiness_report.json",
            "application/json",
            use_container_width=True
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_rfi_csv_from_report(rfis: list) -> str:
    """Generate CSV from RFI list."""
    output = io.StringIO()
    import csv
    writer = csv.DictWriter(output, fieldnames=[
        "RFI ID", "Trade", "Priority", "Question", "Why It Matters",
        "Evidence Pages", "Suggested Resolution"
    ])
    writer.writeheader()
    for rfi in rfis:
        writer.writerow({
            "RFI ID": rfi.get("id", ""),
            "Trade": rfi.get("trade", "").title(),
            "Priority": rfi.get("priority", "").upper(),
            "Question": rfi.get("question", ""),
            "Why It Matters": rfi.get("why_it_matters", ""),
            "Evidence Pages": ", ".join(map(str, rfi.get("evidence", {}).get("pages", []))),
            "Suggested Resolution": rfi.get("suggested_resolution", ""),
        })
    return output.getvalue()


def _generate_rfi_emails_from_report(rfis: list) -> str:
    """Generate email drafts from RFI list."""
    emails = []
    for rfi in rfis:
        email = f"""
================================================================================
RFI: {rfi.get('question', 'Unknown')}
================================================================================
ID: {rfi.get('id', '')}
Trade: {rfi.get('trade', '').title()}
Priority: {rfi.get('priority', '').upper()}

Issue:
{rfi.get('why_it_matters', '')}

Evidence:
- Pages: {', '.join(map(str, rfi.get('evidence', {}).get('pages', []))) or 'N/A'}

Requested Action:
{rfi.get('suggested_resolution', '')}

"""
        emails.append(email)
    return "\n".join(emails)


def build_report_from_results(results: dict, project_id: str) -> dict:
    """Build report dict from loaded results."""
    deep = results.get("deep_analysis") or {}
    plan_graph = results.get("plan_graph") or {}
    bid_gate = results.get("bid_gate") or {}

    # Build executive summary
    score_data = deep.get("readiness_score", {})
    blockers = deep.get("blockers", [])

    top_reasons = []
    for b in blockers[:3]:
        if b.get("severity") in ["critical", "high"]:
            top_reasons.append(b.get("title", "Unknown issue"))

    top_fixes = []
    for b in sorted(blockers, key=lambda x: x.get("score_delta_estimate", 0), reverse=True)[:3]:
        top_fixes.append({
            "action": b.get("fix_actions", ["Resolve issue"])[0] if b.get("fix_actions") else "Resolve issue",
            "score_delta": b.get("score_delta_estimate", 5),
            "description": b.get("title", ""),
        })

    executive_summary = {
        "decision": score_data.get("status", bid_gate.get("status", "NO-GO")),
        "readiness_score": score_data.get("total_score", bid_gate.get("score", 0)),
        "sub_scores": {
            "coverage": score_data.get("coverage_score", 0),
            "measurement": score_data.get("measurement_score", 0),
            "completeness": score_data.get("completeness_score", 0),
            "blocker": score_data.get("blocker_score", 0),
        },
        "top_3_reasons": top_reasons or ["Missing schedules", "Scale not detected", "Incomplete scope"],
        "top_3_fixes": top_fixes or [{"action": "Upload schedules", "score_delta": 10, "description": "Door/window schedules"}],
        "artifacts_generated": ["Bid Readiness Packet", "RFI Pack", "Pricing Readiness Sheet"],
    }

    # Build missing dependencies
    missing_deps = []
    dep_id = 0
    for b in blockers:
        for dep in b.get("missing_dependency", []):
            dep_id += 1
            missing_deps.append({
                "id": f"DEP-{dep_id:04d}",
                "dependency_type": dep.replace("_", " ").title(),
                "status": "missing",
                "why_needed": b.get("description", ""),
                "evidence": b.get("evidence", {}),
                "impact_trade": b.get("trade", "general"),
                "impact_bid": b.get("bid_impact", "clarification"),
                "cost_risk": b.get("impact_cost", "medium"),
                "schedule_risk": b.get("impact_schedule", "low"),
                "fix_options": b.get("fix_actions", []),
                "score_delta": b.get("score_delta_estimate", 5),
                "related_blocker_ids": [b.get("id", "")],
            })

    # Build flagged areas
    flagged_areas = []
    for i, b in enumerate(blockers):
        flagged_areas.append({
            "id": f"FLAG-{i+1:04d}",
            "title": b.get("title", "Unknown"),
            "trade": b.get("trade", "general"),
            "severity": b.get("severity", "medium"),
            "what_flagged": b.get("description", ""),
            "why_flagged": f"Missing: {', '.join(b.get('missing_dependency', []))}. Bid impact: {b.get('bid_impact', 'unknown')}",
            "evidence_pages": b.get("evidence", {}).get("pages", []),
            "recommended_action": b.get("fix_actions", ["Generate RFI"])[0] if b.get("fix_actions") else "Generate RFI",
        })

    by_severity = {}
    for f in flagged_areas:
        sev = f.get("severity", "medium")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    flag_summary = {
        "total_flags": len(flagged_areas),
        "by_severity": by_severity,
    }

    # RFIs
    rfis = deep.get("rfis", [])
    rfis_by_trade = {}
    for rfi in rfis:
        trade = rfi.get("trade", "general")
        if trade not in rfis_by_trade:
            rfis_by_trade[trade] = []
        rfis_by_trade[trade].append(rfi)

    critical_count = sum(1 for r in rfis if r.get("priority") == "critical")
    high_count = sum(1 for r in rfis if r.get("priority") == "high")
    trades_affected = list(set(r.get("trade", "general") for r in rfis))

    rfi_summary = {
        "total_rfis": len(rfis),
        "critical_rfis": critical_count,
        "high_rfis": high_count,
        "trades_affected": trades_affected,
        "top_5_by_impact": [r.get("question", "")[:80] for r in rfis[:5]],
    }

    # Trade coverage
    trade_coverage = []
    for tc in deep.get("trade_coverage", []):
        trade_coverage.append({
            "trade": tc.get("trade", "general"),
            "coverage_pct": tc.get("coverage_pct", 0),
            "priceable_count": tc.get("priceable_count", 0),
            "blocked_count": tc.get("blocked_count", 0),
            "top_missing": tc.get("missing_dependencies", [])[:3],
            "confidence": "high" if tc.get("coverage_pct", 0) >= 70 else ("medium" if tc.get("coverage_pct", 0) >= 40 else "low"),
            "cost_risk": tc.get("cost_risk", "medium"),
            "schedule_risk": tc.get("schedule_risk", "low"),
        })

    # Priceable/blocked from BOQ skeleton
    priceable = []
    blocked = []
    for item in deep.get("boq_skeleton", []):
        name = f"{item.get('trade', '')}: {item.get('item_name', '')}"
        if item.get("status") == "priceable":
            priceable.append(name)
        else:
            blocked.append(name)

    # Assumptions
    assumption_templates = {
        "door_schedule": ("Door Specification Assumption", "Doors assumed as flush doors, standard commercial grade.", "15-30% cost variance if actual specs differ"),
        "window_schedule": ("Window Specification Assumption", "Windows assumed as 2-track aluminium sliding with 5mm glass.", "20-40% cost variance possible"),
        "finish_schedule": ("Finishes Assumption", "Finishes assumed as vitrified tile floors, emulsion paint walls.", "30-50% variance for premium finishes"),
        "mep_drawings": ("MEP Scope Assumption", "MEP as per standard norms. 6 points/room avg.", "40-60% variance on MEP package"),
        "scale_notation": ("Scale Assumption", "Drawings assumed at 1:100 scale.", "Measurement errors if scale incorrect"),
    }

    assumptions = []
    seen = set()
    for dep in missing_deps:
        dep_key = dep.get("dependency_type", "").lower().replace(" ", "_")
        if dep_key in assumption_templates and dep_key not in seen:
            seen.add(dep_key)
            title, text, impact = assumption_templates[dep_key]
            assumptions.append({
                "id": f"ASMP-{len(assumptions)+1:04d}",
                "title": title,
                "text": text,
                "impact_if_wrong": impact,
                "linked_blocker_ids": dep.get("related_blocker_ids", []),
            })

    # Drawing set overview
    drawing_set_overview = {
        "total_pages": plan_graph.get("total_pages", 0),
        "disciplines_detected": plan_graph.get("disciplines_found", []),
        "sheet_types": plan_graph.get("sheet_types_found", {}),
        "schedules_detected": [],
        "pages_with_scale": plan_graph.get("pages_with_scale", 0),
        "pages_without_scale": plan_graph.get("pages_without_scale", 0),
        "rooms_detected": len(plan_graph.get("all_room_names", [])),
        "room_names": plan_graph.get("all_room_names", [])[:20],
        "door_tags_count": len(plan_graph.get("all_door_tags", [])),
        "window_tags_count": len(plan_graph.get("all_window_tags", [])),
        "door_tags_sample": plan_graph.get("all_door_tags", [])[:10],
        "window_tags_sample": plan_graph.get("all_window_tags", [])[:10],
    }

    if plan_graph.get("has_door_schedule"):
        drawing_set_overview["schedules_detected"].append({"type": "door_schedule", "status": "found"})
    if plan_graph.get("has_window_schedule"):
        drawing_set_overview["schedules_detected"].append({"type": "window_schedule", "status": "found"})
    if plan_graph.get("has_finish_schedule"):
        drawing_set_overview["schedules_detected"].append({"type": "finish_schedule", "status": "found"})

    return {
        "project_id": project_id,
        "executive_summary": executive_summary,
        "missing_dependencies": missing_deps,
        "flagged_areas": flagged_areas,
        "flag_summary": flag_summary,
        "rfis": rfis,
        "rfi_summary": rfi_summary,
        "rfis_by_trade": rfis_by_trade,
        "trade_coverage": trade_coverage,
        "priceable_categories": priceable[:8],
        "blocked_categories": blocked[:8],
        "assumptions": assumptions[:10],
        "drawing_set_overview": drawing_set_overview,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.markdown("# XBOQ")
    st.caption("Pre-Bid Scope & Risk Check")

    project_id = st.query_params.get("project_id", "")

    if not project_id:
        # Upload interface
        st.markdown("---")
        st.markdown("### Upload your tender drawings")
        st.markdown("Drop your drawing PDFs to detect scope gaps before you bid.")

        uploaded_files = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True,
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
        # Results view
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

        # Build report from results
        report = build_report_from_results(results, project_id)

        # ===================
        # TABBED SECTIONS
        # ===================
        tabs = st.tabs([
            "📊 Summary",
            "📋 Missing Dependencies",
            "⚠️ Flagged Areas",
            "📝 RFIs",
            "💰 Coverage",
            "📄 Assumptions",
            "🔍 Audit"
        ])

        with tabs[0]:
            render_section_1_summary(report)

            # Export buttons
            st.markdown("---")
            st.markdown("### 📦 Export All Deliverables")
            col1, col2, col3 = st.columns(3)
            with col1:
                html_content = f"""<!DOCTYPE html><html><head><title>Bid Readiness Report</title></head>
                <body><h1>Bid Readiness Report: {project_id}</h1>
                <p>Score: {report['executive_summary']['readiness_score']}/100</p>
                <p>Decision: {report['executive_summary']['decision']}</p>
                <h2>Missing Dependencies</h2>
                {''.join(f"<p>{d['dependency_type']}: {d['why_needed'][:100]}</p>" for d in report['missing_dependencies'])}
                </body></html>"""
                st.download_button("📄 Download HTML Report", html_content, f"bid_report_{project_id}.html", "text/html", use_container_width=True)
            with col2:
                csv_content = _generate_rfi_csv_from_report(report["rfis"])
                st.download_button("📊 Download RFI CSV", csv_content, f"rfis_{project_id}.csv", "text/csv", use_container_width=True)
            with col3:
                st.download_button("📥 Download Full JSON", json.dumps(report, indent=2, default=str), f"report_{project_id}.json", "application/json", use_container_width=True)

        with tabs[1]:
            render_section_2_dependencies(report)

        with tabs[2]:
            render_section_3_flagged(report)

        with tabs[3]:
            render_section_4_rfis(report)

        with tabs[4]:
            render_section_5_coverage(report)

        with tabs[5]:
            render_section_6_assumptions(report)

        with tabs[6]:
            render_section_7_audit(report)

        # Footer
        st.markdown("---")
        st.caption("Generated by XBOQ • Pre-Bid Scope & Risk Check")


if __name__ == "__main__":
    main()
