"""
XBOQ Demo - Pre-Bid Scope & Risk Check

Comprehensive Bid Readiness Report showing at-a-glance:
1. What tender was analyzed (project info, files, pages)
2. What xBOQ did (processing steps + timing)
3. Coverage, risk, RFIs, and trade breakdown
4. How decision ties back to risk profile

7-tab detailed view:
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
import os
import sys
import io
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

# Add src and app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# DATA MODELS FOR UI
# =============================================================================

@dataclass
class AnalysisStepTiming:
    """Timing info for a single analysis step."""
    step_id: str
    step_name: str
    duration_s: float
    status: str  # "complete", "in_progress", "pending"
    detail: str = ""


@dataclass
class RiskProfile:
    """Risk assessment summary."""
    overall_risk: str  # "low", "medium", "high"
    cost_risk: str
    schedule_risk: str
    risk_drivers: List[str]
    risk_score: int  # 0-100, higher = more risk


@dataclass
class TradeSummaryCard:
    """Summary card for a single trade."""
    trade: str
    trade_display: str  # Human-readable name
    coverage_pct: float
    status: str  # "ready", "blocked", "partial", "not_found"
    priceable_items: int
    blocked_items: int
    top_blocker: Optional[str]
    rfi_count: int
    icon: str


@dataclass
class DemoAnalysis:
    """
    Complete data model for demo analysis UI.

    Structures all data needed for the at-a-glance estimator view.
    """
    # Project identity
    project_id: str
    timestamp: str
    files_analyzed: List[str]

    # Drawing overview
    pages_total: int
    disciplines_detected: List[str]
    door_tags_found: int
    window_tags_found: int
    room_names_found: int
    scale_found_pages: int
    ocr_used: bool

    # Decision
    decision: str  # "PASS", "CONDITIONAL", "NO-GO"
    readiness_score: int
    sub_scores: Dict[str, int]
    decision_reasons: List[str]

    # Risk profile
    risk_profile: RiskProfile

    # Analysis steps
    analysis_steps: List[AnalysisStepTiming]
    total_time_s: float

    # Blockers
    blockers_count: int
    critical_blockers_count: int
    top_blockers: List[Dict[str, Any]]

    # RFIs
    rfis_count: int
    rfis_by_trade: Dict[str, int]
    top_rfis: List[Dict[str, Any]]

    # Trade breakdown
    trade_cards: List[TradeSummaryCard]

    # Raw payload for detailed tabs
    raw_payload: Dict[str, Any]


def build_demo_analysis(payload: Dict[str, Any], project_id: str) -> DemoAnalysis:
    """
    Build DemoAnalysis from raw analysis payload.

    This transforms the analysis.json output into a structured UI model.
    """
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    timings = payload.get("timings", {})
    blockers = payload.get("blockers", [])
    rfis = payload.get("rfis", [])
    trade_coverage = payload.get("trade_coverage", [])
    sub_scores = payload.get("sub_scores", {})

    # Build analysis steps from timings
    step_names = {
        "load": ("📥", "Load PDFs", "Scanning and indexing files"),
        "index": ("🗂️", "Index Pages", "Classifying all pages by type"),
        "select": ("🎯", "Select Pages", "Prioritizing pages within OCR budget"),
        "extract": ("📝", "Extract Text", "OCR and text extraction"),
        "graph": ("🔗", "Build Graph", "Analyzing drawing structure"),
        "reason": ("🧠", "Analyze Scope", "Detecting blockers"),
        "rfi": ("📋", "Generate RFIs", "Creating recommendations"),
        "export": ("💾", "Export Results", "Saving output files"),
    }

    analysis_steps = []
    for step_id, (icon, name, detail) in step_names.items():
        duration = timings.get(f"{step_id}_s", None)
        # Treat None and 0 as "not recorded", very small times as "completed fast"
        has_timing = duration is not None and duration > 0
        analysis_steps.append(AnalysisStepTiming(
            step_id=step_id,
            step_name=f"{icon} {name}",
            duration_s=duration if duration else 0,
            status="complete" if has_timing else "pending",
            detail=detail
        ))

    # Build risk profile
    critical_count = sum(1 for b in blockers if b.get("severity") in ["critical", "high"])
    risk_drivers = [b.get("title", "Unknown") for b in blockers[:3] if b.get("severity") in ["critical", "high"]]

    overall_risk = "low"
    if critical_count >= 3 or payload.get("readiness_score", 0) < 40:
        overall_risk = "high"
    elif critical_count >= 1 or payload.get("readiness_score", 0) < 70:
        overall_risk = "medium"

    # Calculate cost/schedule risk from blockers
    cost_impacts = [b.get("impact_cost", "medium") for b in blockers]
    schedule_impacts = [b.get("impact_schedule", "low") for b in blockers]

    cost_risk = "high" if cost_impacts.count("high") >= 2 else ("medium" if "high" in cost_impacts or "medium" in cost_impacts else "low")
    schedule_risk = "high" if schedule_impacts.count("high") >= 2 else ("medium" if "high" in schedule_impacts else "low")

    risk_profile = RiskProfile(
        overall_risk=overall_risk,
        cost_risk=cost_risk,
        schedule_risk=schedule_risk,
        risk_drivers=risk_drivers,
        risk_score=100 - payload.get("readiness_score", 0)
    )

    # Build trade cards
    trade_icons = {
        "civil": "🏗️",
        "structural": "🔩",
        "architectural": "🏠",
        "mep": "⚡",
        "finishes": "🎨",
        "general": "📦",
        "electrical": "💡",
        "plumbing": "🚿",
    }

    trade_cards = []
    for tc in trade_coverage:
        trade = tc.get("trade", "general")
        cov_pct = tc.get("coverage_pct", 0)
        blocked = tc.get("blocked_count", 0)
        priceable = tc.get("priceable_count", 0)

        # Determine status
        if cov_pct >= 80 and blocked == 0:
            status = "ready"
        elif cov_pct == 0 and priceable == 0:
            status = "not_found"
        elif blocked > 0:
            status = "blocked"
        else:
            status = "partial"

        # Find top blocker for this trade
        trade_blockers = [b for b in blockers if b.get("trade") == trade]
        top_blocker = trade_blockers[0].get("title") if trade_blockers else None

        # Count RFIs for this trade
        trade_rfis = [r for r in rfis if r.get("trade") == trade]

        trade_cards.append(TradeSummaryCard(
            trade=trade,
            trade_display=trade.replace("_", " ").title(),
            coverage_pct=cov_pct,
            status=status,
            priceable_items=priceable,
            blocked_items=blocked,
            top_blocker=top_blocker,
            rfi_count=len(trade_rfis),
            icon=trade_icons.get(trade, "📋")
        ))

    # Build decision reasons
    decision_reasons = []
    if blockers:
        for b in blockers[:3]:
            reason = b.get("title") or b.get("description", "Unknown issue")
            decision_reasons.append(reason)

    # Count RFIs by trade
    rfis_by_trade = {}
    for r in rfis:
        trade = r.get("trade", "general")
        rfis_by_trade[trade] = rfis_by_trade.get(trade, 0) + 1

    return DemoAnalysis(
        project_id=project_id,
        timestamp=payload.get("timestamp", datetime.now().isoformat()),
        files_analyzed=overview.get("files", []),
        pages_total=overview.get("pages_total", 0),
        disciplines_detected=overview.get("disciplines_detected", []),
        door_tags_found=overview.get("door_tags_found", 0),
        window_tags_found=overview.get("window_tags_found", 0),
        room_names_found=overview.get("room_names_found", 0),
        scale_found_pages=overview.get("scale_found_pages", 0),
        ocr_used=overview.get("ocr_used", False),
        decision=payload.get("decision", "NO-GO"),
        readiness_score=payload.get("readiness_score", 0),
        sub_scores=sub_scores,
        decision_reasons=decision_reasons,
        risk_profile=risk_profile,
        analysis_steps=analysis_steps,
        total_time_s=timings.get("total_s", 0),
        blockers_count=len(blockers),
        critical_blockers_count=critical_count,
        top_blockers=blockers[:5],
        rfis_count=len(rfis),
        rfis_by_trade=rfis_by_trade,
        top_rfis=rfis[:5],
        trade_cards=trade_cards,
        raw_payload=payload
    )

# Import analysis runner
from analysis_runner import (
    run_analysis_pipeline,
    save_uploaded_files,
    generate_project_id,
    AnalysisResult,
)

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: #09090b; }
    .main .block-container { max-width: 1100px; padding: 2rem 1.5rem; }
    h1, h2, h3, h4, p, span, div, li, a, code, label { font-family: 'Inter', -apple-system, sans-serif !important; }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: #7c3aed !important; color: white !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; padding: 0.6rem 1.5rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover { background: #6d28d9 !important; transform: translateY(-1px); }
    .stButton > button[kind="secondary"] {
        background: transparent !important; color: #a78bfa !important;
        border: 1px solid rgba(124,58,237,0.4) !important; border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="secondary"]:hover { border-color: #7c3aed !important; background: rgba(124,58,237,0.08) !important; }
    .stDownloadButton > button {
        background: rgba(124,58,237,0.12) !important; color: #c4b5fd !important;
        border: 1px solid rgba(124,58,237,0.25) !important; border-radius: 8px !important;
    }
    .stDownloadButton > button:hover { background: rgba(124,58,237,0.2) !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem !important; color: #71717a !important;
        font-weight: 500 !important; font-size: 0.85rem !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #7c3aed !important; }

    /* ── Hero landing ── */
    .hero-section {
        text-align: center; padding: 3rem 1rem 2rem;
    }
    .hero-section h1 {
        font-size: 3.5rem; font-weight: 800; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #c4b5fd, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-tagline {
        font-size: 1.15rem; color: #a1a1aa; font-weight: 400;
        max-width: 500px; margin: 0 auto 2.5rem; line-height: 1.6;
    }
    .hero-steps {
        display: flex; justify-content: center; gap: 2.5rem; margin: 2rem 0;
    }
    .hero-step {
        text-align: center; max-width: 160px;
    }
    .hero-step-num {
        width: 32px; height: 32px; border-radius: 50%;
        background: rgba(124,58,237,0.15); color: #a78bfa;
        font-weight: 700; font-size: 0.85rem;
        display: inline-flex; align-items: center; justify-content: center;
        margin-bottom: 0.5rem;
    }
    .hero-step-label { font-size: 0.85rem; color: #71717a; line-height: 1.4; }

    /* ── Decision badges ── */
    .decision-badge {
        display: inline-block; padding: 0.4rem 1.2rem; border-radius: 8px;
        font-size: 1.1rem; font-weight: 700; letter-spacing: 0.02em;
    }
    .badge-nogo { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
    .badge-conditional { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
    .badge-pass { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }

    /* ── Score ring ── */
    .score-ring {
        text-align: center; padding: 1rem 0;
    }
    .score-ring .score-number {
        font-size: 3.5rem; font-weight: 800; line-height: 1;
        background: linear-gradient(135deg, #c4b5fd, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .score-ring .score-label { font-size: 0.8rem; color: #71717a; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }

    /* ── Sub-score pills ── */
    .sub-score-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.75rem 0; }
    .sub-score-pill {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px; padding: 0.5rem 0.75rem; text-align: center; flex: 1; min-width: 100px;
    }
    .sub-score-pill .pill-value { font-size: 1.2rem; font-weight: 700; color: #e4e4e7; }
    .sub-score-pill .pill-label { font-size: 0.7rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ── Quick stats bar ── */
    .quick-stats {
        display: flex; gap: 0; padding: 0;
        background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; margin: 1rem 0; overflow: hidden;
    }
    .quick-stat {
        flex: 1; text-align: center; padding: 0.75rem 0.5rem;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .quick-stat:last-child { border-right: none; }
    .quick-stat-value { font-size: 1.3rem; font-weight: 700; color: #e4e4e7; }
    .quick-stat-label { font-size: 0.65rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ── Project header ── */
    .project-header {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
    }
    .project-header h2 { margin: 0 0 0.25rem 0; color: #e4e4e7; font-size: 1.2rem; font-weight: 600; }
    .project-meta { color: #71717a; font-size: 0.8rem; display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }
    .project-meta .meta-dot { color: #3f3f46; }

    /* ── Pipeline steps ── */
    .pipeline-row { display: flex; gap: 0.25rem; margin: 0.5rem 0; }
    .pipeline-step {
        flex: 1; text-align: center; padding: 0.6rem 0.25rem;
        background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px; transition: border-color 0.2s;
    }
    .pipeline-step:hover { border-color: rgba(124,58,237,0.3); }
    .pipeline-step .step-icon { font-size: 1.1rem; }
    .pipeline-step .step-name { font-size: 0.7rem; font-weight: 500; color: #a1a1aa; margin-top: 0.15rem; }
    .pipeline-step .step-time { font-size: 0.75rem; color: #22c55e; font-weight: 600; margin-top: 0.1rem; }
    .pipeline-step .step-time.pending { color: #3f3f46; }

    /* ── Trade cards ── */
    .trade-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 1rem;
        height: 100%; transition: border-color 0.2s;
    }
    .trade-card:hover { border-color: rgba(124,58,237,0.3); }
    .trade-card.status-ready { border-left: 3px solid #22c55e; }
    .trade-card.status-blocked { border-left: 3px solid #ef4444; }
    .trade-card.status-partial { border-left: 3px solid #f59e0b; }
    .trade-card.status-not_found { border-left: 3px solid #27272a; opacity: 0.5; }
    .trade-card-header { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.5rem; }
    .trade-card-icon { font-size: 1.2rem; }
    .trade-card-name { font-weight: 600; font-size: 0.95rem; color: #e4e4e7; }
    .trade-card-coverage { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .trade-card-coverage.high { color: #4ade80; }
    .trade-card-coverage.medium { color: #fbbf24; }
    .trade-card-coverage.low { color: #f87171; }
    .trade-card-stats { font-size: 0.78rem; color: #71717a; }
    .trade-card-blocker {
        font-size: 0.75rem; color: #f87171; margin-top: 0.5rem;
        padding: 0.3rem 0.5rem; background: rgba(239,68,68,0.08);
        border-radius: 6px; border: 1px solid rgba(239,68,68,0.15);
    }

    /* ── Risk indicators ── */
    .risk-indicator {
        display: inline-flex; align-items: center;
        padding: 0.2rem 0.6rem; border-radius: 6px;
        font-weight: 600; font-size: 0.8rem; letter-spacing: 0.02em;
    }
    .risk-low { background: rgba(34,197,94,0.1); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
    .risk-medium { background: rgba(245,158,11,0.1); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
    .risk-high { background: rgba(239,68,68,0.1); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

    /* ── Section headers ── */
    .section-header {
        font-size: 0.75rem; font-weight: 600; color: #71717a;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 0.75rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Severity colors ── */
    .severity-critical, .severity-high { color: #f87171; }
    .severity-medium { color: #fbbf24; }
    .severity-low { color: #4ade80; }

    /* ── Reason/Fix boxes ── */
    .reasons-box { background: rgba(239,68,68,0.06); padding: 1rem;
                   border-radius: 8px; border-left: 3px solid rgba(239,68,68,0.4); }
    .fixes-box { background: rgba(34,197,94,0.06); padding: 1rem;
                 border-radius: 8px; border-left: 3px solid rgba(34,197,94,0.4); }
    .dep-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
                border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }

    /* ── BOQ category pills ── */
    .boq-pill {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px;
        font-size: 0.72rem; font-weight: 500; margin: 0.15rem 0.2rem;
        background: rgba(124,58,237,0.1); color: #c4b5fd;
        border: 1px solid rgba(124,58,237,0.2);
    }
    /* ── Score delta badge ── */
    .score-delta {
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px;
        font-size: 0.75rem; font-weight: 600;
        background: rgba(34,197,94,0.1); color: #4ade80;
        border: 1px solid rgba(34,197,94,0.2);
    }
    /* ── FYI info box ── */
    .fyi-box {
        background: rgba(59,130,246,0.06); padding: 0.75rem 1rem;
        border-radius: 8px; border-left: 3px solid rgba(59,130,246,0.4);
        margin: 0.5rem 0; font-size: 0.85rem; color: #93c5fd;
    }
    /* ── Quantified outputs table ── */
    .quantified-table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; }
    .quantified-table th {
        text-align: left; padding: 0.5rem 0.75rem; font-size: 0.7rem;
        color: #71717a; text-transform: uppercase; letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .quantified-table td {
        padding: 0.5rem 0.75rem; font-size: 0.85rem; color: #d4d4d8;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .quantified-table td.count-val { font-weight: 700; font-size: 1rem; color: #e4e4e7; }

    /* ── Blocker / RFI result tables ── */
    .result-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem 0; }
    .result-table th {
        text-align: left; padding: 0.6rem 0.75rem; font-size: 0.68rem;
        color: #71717a; text-transform: uppercase; letter-spacing: 0.5px;
        border-bottom: 2px solid rgba(255,255,255,0.08); white-space: nowrap;
    }
    .result-table td {
        padding: 0.6rem 0.75rem; font-size: 0.82rem; color: #d4d4d8;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        vertical-align: top; line-height: 1.4;
    }
    .result-table tr:hover td { background: rgba(255,255,255,0.02); }
    .sev-pill {
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px;
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    }
    .sev-critical, .sev-high { background: rgba(239,68,68,0.12); color: #f87171; }
    .sev-medium { background: rgba(251,191,36,0.12); color: #fbbf24; }
    .sev-low { background: rgba(74,222,128,0.12); color: #4ade80; }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(124,58,237,0.25) !important;
        border-radius: 12px !important; background: rgba(124,58,237,0.03) !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(124,58,237,0.5) !important; }

    /* ── Metric overrides ── */
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }

    /* ── Evidence viewer ── */
    .ev-snippet { background: rgba(124,58,237,0.06); padding: 0.5rem 0.75rem;
                  border-radius: 6px; border-left: 3px solid rgba(124,58,237,0.3);
                  font-size: 0.82rem; margin: 0.25rem 0; }
    .ev-rule-tag { display: inline-block; background: rgba(124,58,237,0.12);
                   padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem;
                   font-family: monospace; color: #c4b5fd; margin-right: 0.5rem; }

    /* ── Coverage dashboard ── */
    .cov-found { color: #4ade80; font-weight: 600; }
    .cov-partial { color: #fbbf24; font-weight: 600; }
    .cov-not-processed { color: #f87171; font-weight: 600; }
    .mini-bar { background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; width: 100%; }
    .mini-bar-fill { border-radius: 4px; height: 8px; }

    /* ── Confidence badges (Sprint 4a) ── */
    .ev-conf-high { color: #4ade80; font-weight: 600; }
    .ev-conf-med  { color: #fbbf24; font-weight: 600; }
    .ev-conf-low  { color: #f87171; font-weight: 600; }

    /* ── Inline citations (Sprint 4a) ── */
    .ev-citation { color: #a78bfa; font-size: 0.8em; font-weight: 500;
                   margin-left: 0.25rem; }

    /* ── Coverage heatmap (Sprint 4a) ── */
    .heatmap-grid { display: flex; flex-wrap: wrap; gap: 3px; margin: 0.5rem 0 1rem 0; }
    .heatmap-cell { width: 28px; height: 28px; border-radius: 4px; display: flex;
                    align-items: center; justify-content: center; font-size: 0.6rem;
                    color: rgba(0,0,0,0.7); cursor: default; font-weight: 500; }

    /* ── Bbox selector (Sprint 5) ── */
    .bbox-row { display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.5rem;
                border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.82rem; }
    .bbox-num { display: inline-flex; align-items: center; justify-content: center;
                width: 22px; height: 22px; border-radius: 50%;
                background: rgba(245,158,11,0.2); font-size: 0.7rem; font-weight: 700;
                color: #fbbf24; }
    .bbox-id { font-family: monospace; color: #a78bfa; font-size: 0.75rem; }
    .bbox-detail { background: rgba(124,58,237,0.06); border: 1px solid rgba(124,58,237,0.15);
                   border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; }

    /* ── Bid strategy dials (Sprint 5) ── */
    .bid-dial { text-align: center; padding: 1rem; background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; }
    .bid-dial-label { font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-bottom: 0.25rem; }
    .bid-dial-score { font-size: 2rem; font-weight: 700; }
    .bid-dial-conf { font-size: 0.7rem; margin-top: 0.15rem; }
    .bid-dial-unknown { color: #6b7280; font-size: 1.5rem; }
    .bid-rec { background: rgba(245,158,11,0.06); padding: 0.5rem 0.75rem;
               border-radius: 6px; border-left: 3px solid rgba(245,158,11,0.3);
               font-size: 0.85rem; margin: 0.25rem 0; }

    /* ── Addenda & Conflicts (Sprint 6) ── */
    .addendum-badge { display: inline-block; background: rgba(168,85,247,0.15);
                      color: #c4b5fd; padding: 0.1rem 0.45rem; border-radius: 4px;
                      font-size: 0.72rem; font-weight: 600; margin-right: 0.35rem; }
    .conflict-badge { display: inline-block; background: rgba(239,68,68,0.15);
                      color: #fca5a5; padding: 0.1rem 0.45rem; border-radius: 4px;
                      font-size: 0.72rem; font-weight: 600; }

    /* ── Global search (Sprint 5) ── */
    .search-result { padding: 0.35rem 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.04);
                     font-size: 0.85rem; }
    .search-page-pill { display: inline-block; background: rgba(59,130,246,0.15);
                        color: #93c5fd; padding: 0.1rem 0.4rem; border-radius: 4px;
                        font-size: 0.72rem; font-weight: 600; margin-right: 0.35rem; }

    /* ── Sprint 20B: Recording polish ── */
    /* Better tab spacing at common recording widths */
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 0.75rem; white-space: nowrap;
                                    font-size: 0.82rem; min-height: 2.2rem; }
    /* Metric cards readable in both themes */
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; text-transform: uppercase;
                                     letter-spacing: 0.03em; opacity: 0.85; }
    /* Prevent button row wrapping */
    .stButton > button { min-width: 0 !important; font-size: 0.82rem !important; }
    /* Table headers cleaner */
    .quantified-table th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
                           color: #a1a1aa; border-bottom: 1px solid rgba(255,255,255,0.08); }
    .quantified-table td { padding: 0.4rem 0.5rem; }
    /* Recorder step strip */
    .recorder-step { text-align: center; padding: 0.5rem; }
    /* Dark mode dataframe readability */
    .stDataFrame { font-size: 0.82rem !important; }
    /* Expander spacing */
    .streamlit-expanderHeader { font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)


def load_demo_results(project_id: str) -> dict:
    """
    Load results for a project.

    Search order:
    1. demo_cache/<project_id>/analysis.json (stable cached demos)
    2. out/<project_id>/analysis.json (fresh analysis output)
    3. out/<project_id>/*.json (legacy format)

    This ensures YC demos use reliable cached data while fresh uploads work too.
    """
    project_root = Path(__file__).parent.parent

    results = {
        "loaded": False, "rfis": [], "trade_summary": {}, "bid_gate": {},
        "metrics": {}, "critical_blockers": [], "deep_analysis": None, "plan_graph": None,
    }

    # ===== PRIORITY 1: Check demo cache (stable for demos) =====
    cache_path = project_root / "demo_cache" / project_id / "analysis.json"
    if cache_path.exists():
        with open(cache_path) as f:
            results["analysis"] = json.load(f)
            results["loaded"] = True
            results["source"] = "demo_cache"
            # Extract metrics from analysis.json
            overview = results["analysis"].get("drawing_overview") or results["analysis"].get("overview") or {}
            results["metrics"] = {
                "pages": overview.get("pages_total", 0),
                "rooms": overview.get("room_names_found", 0),
                "openings": overview.get("door_tags_found", 0) + overview.get("window_tags_found", 0),
            }
            # Extract RFIs from analysis.json
            results["rfis"] = results["analysis"].get("rfis", [])
            results["critical_blockers"] = results["analysis"].get("blockers", [])
            return results

    # ===== PRIORITY 2: Check output directory =====
    base_path = project_root / "out" / project_id

    # Try to load analysis.json directly (new format)
    analysis_path = base_path / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            results["analysis"] = json.load(f)
            results["loaded"] = True
            results["source"] = "output"
            # Extract metrics from analysis.json
            overview = results["analysis"].get("drawing_overview") or results["analysis"].get("overview") or {}
            results["metrics"] = {
                "pages": overview.get("pages_total", 0),
                "rooms": overview.get("room_names_found", 0),
                "openings": overview.get("door_tags_found", 0) + overview.get("window_tags_found", 0),
            }
            # Extract RFIs from analysis.json
            results["rfis"] = results["analysis"].get("rfis", [])
            results["critical_blockers"] = results["analysis"].get("blockers", [])
            return results

    # ===== PRIORITY 3: Legacy format (for old projects) =====
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

    # Consider loaded if we have any meaningful data
    results["loaded"] = (
        len(results["rfis"]) > 0 or
        results["bid_gate"] or
        results["deep_analysis"] is not None or
        results["plan_graph"] is not None
    )
    results["source"] = "legacy" if results["loaded"] else None

    return results


# =============================================================================
# AT-A-GLANCE DASHBOARD RENDERERS
# =============================================================================

def render_project_header(demo: DemoAnalysis):
    """Render the project identity header - what tender was analyzed."""
    files_str = ", ".join(demo.files_analyzed) if demo.files_analyzed else "Uploaded drawings"

    # Parse timestamp for display
    try:
        ts = datetime.fromisoformat(demo.timestamp.replace("Z", "+00:00"))
        time_str = ts.strftime("%b %d, %Y at %H:%M")
    except Exception:
        time_str = demo.timestamp

    # Build meta items
    meta_items = [f"{demo.pages_total} pages"]
    if demo.disciplines_detected:
        meta_items.append(f"{len(demo.disciplines_detected)} disciplines")
    if demo.ocr_used:
        meta_items.append("OCR")
    meta_html = f' <span class="meta-dot">&bull;</span> '.join(meta_items)

    st.markdown(f"""
    <div class="project-header">
        <h2>{files_str}</h2>
        <div class="project-meta">
            <span>{time_str}</span>
            <span class="meta-dot">&bull;</span>
            {meta_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_decision_banner(demo: DemoAnalysis):
    """Render the main decision banner with score and quick metrics."""
    decision = demo.decision.upper()
    if decision in ["NO-GO", "NOGO", "NO_DRAWINGS"]:
        badge_class = "badge-nogo"
    elif decision in ["CONDITIONAL", "REVIEW"]:
        badge_class = "badge-conditional"
    else:
        badge_class = "badge-pass"

    # Clean display label
    if decision == "NO_DRAWINGS":
        decision = "NO DRAWINGS"

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(f"""
        <div class="score-ring">
            <div class="score-number">{demo.readiness_score}</div>
            <div class="score-label">Readiness</div>
            <div style="margin-top: 0.75rem;">
                <span class="decision-badge {badge_class}">{decision}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Quick stats row
        st.markdown("""
        <div class="quick-stats">
            <div class="quick-stat">
                <div class="quick-stat-value">{pages}</div>
                <div class="quick-stat-label">Pages</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{blockers}</div>
                <div class="quick-stat-label">Blockers</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{rfis}</div>
                <div class="quick-stat-label">RFIs</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{disciplines}</div>
                <div class="quick-stat-label">Disciplines</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{time:.1f}s</div>
                <div class="quick-stat-label">Time</div>
            </div>
        </div>
        """.format(
            pages=demo.pages_total,
            blockers=demo.blockers_count,
            rfis=demo.rfis_count,
            disciplines=len(demo.disciplines_detected),
            time=demo.total_time_s
        ), unsafe_allow_html=True)

        # Sub-scores as pills — Sprint 20E: add "X/Y pages processed" sublabel
        # when in FAST_BUDGET mode to avoid misleading 100% coverage impression.
        sub = demo.sub_scores
        _ps_top = demo.raw_payload.get("processing_stats") or {}
        _deep_top = _ps_top.get("deep_processed_pages")
        _total_top = _ps_top.get("total_pages") or demo.pages_total

        # Sublabel for Coverage pill — Sprint 20G: show "Selected Scope Coverage"
        # unless FULL_AUDIT completed all pages.
        _cov_sublabel = ""
        _cov_label = "Coverage"
        _run_mode_top = _ps_top.get("run_mode", "demo_fast")
        if _deep_top is not None and _total_top and _deep_top < _total_top:
            _cov_sublabel = (
                f'<div class="pill-sublabel">Processed {_deep_top}/{_total_top} pages</div>'
            )
            _cov_label = "Selected Scope Coverage"
        elif _run_mode_top == "full_audit":
            _cov_label = "Full Coverage"

        st.markdown(f"""
        <style>.pill-sublabel {{font-size:0.6rem;color:#71717a;margin-top:2px;}}</style>
        <div class="sub-score-row">
            <div class="sub-score-pill">
                <div class="pill-value">{sub.get("coverage", 0)}</div>
                <div class="pill-label">{_cov_label}</div>
                {_cov_sublabel}
            </div>
            <div class="sub-score-pill">
                <div class="pill-value">{sub.get("completeness", 0)}</div>
                <div class="pill-label">Completeness</div>
            </div>
            <div class="sub-score-pill">
                <div class="pill-value">{sub.get("measurement", 0)}</div>
                <div class="pill-label">Measurement</div>
            </div>
            <div class="sub-score-pill">
                <div class="pill-value">{sub.get("blocker", 0)}</div>
                <div class="pill-label">Blockers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    - >= 1s: show as "X.XXs"
    - 0.01 to 1s: show as "X.XXs"
    - < 0.01s: show as "<0.01s"
    - 0 or missing: show as "—"
    """
    if seconds is None or seconds <= 0:
        return "—"
    elif seconds < 0.01:
        return "<0.01s"
    elif seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.1f}s"


def format_page_refs(pages: list, max_show: int = 5) -> str:
    """Format 0-indexed page numbers into 'pp. 1, 3, 7' display string."""
    if not pages:
        return ""
    display_pages = [str(p + 1) for p in pages[:max_show]]
    suffix = f" +{len(pages) - max_show} more" if len(pages) > max_show else ""
    return f"pp. {', '.join(display_pages)}{suffix}"


def is_mep_fyi_blocker(blocker: dict) -> bool:
    """Check if a blocker should be rendered as FYI instead of a red blocker.

    MEP 'missing drawing' blockers are FYIs, not true blockers,
    when MEP drawings were not included in the tender package.
    """
    return (
        blocker.get("issue_type") == "missing_drawing"
        and blocker.get("trade", "").lower() == "mep"
    )


def has_rfi_evidence(rfi: dict) -> bool:
    """Check if an RFI has meaningful evidence (pages or detected entities)."""
    evidence = rfi.get("evidence", {})
    if not evidence:
        return False
    has_pages = bool(evidence.get("pages"))
    has_entities = bool(evidence.get("detected_entities"))
    return has_pages or has_entities


def _make_widget_key(*parts) -> str:
    """Build a unique Streamlit widget key from variable parts.

    Joins non-empty parts with colons.  Truncates each part to 40 chars
    to keep keys short.  Ensures uniqueness when callers supply tab context
    + item id + loop index.
    """
    clean = [str(p).replace(" ", "_")[:40] for p in parts if p is not None and str(p)]
    return ":".join(clean) if clean else f"wk_{id(parts)}"


def _safe_str(val) -> str:
    """Convert any value to a clean display string.

    Prevents '._arr' / raw object repr from leaking into the UI.
    - Lists become comma-joined strings
    - Dicts become key:value pairs
    - None becomes ''
    """
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        return ", ".join(_safe_str(v) for v in val[:10])
    if isinstance(val, dict):
        parts = []
        for k, v in list(val.items())[:6]:
            parts.append(f"{k}: {_safe_str(v)}")
        return "; ".join(parts)
    return str(val)


# Mapping blocker IDs → relevant doc_types (for skipped-page linkage)
_BLOCKER_DOC_TYPE_MAP = {
    "BLK-0010": ["schedule"],   # Doors need schedule
    "BLK-0011": ["schedule"],   # Windows need schedule
    "BLK-0012": ["schedule"],   # Finish schedule
    "BLK-0013": ["plan", "detail", "section", "elevation"],  # Scale on all drawings
    "BLK-0002": ["section"],    # Missing sections
    "BLK-0001": ["elevation"],  # Missing elevations
    "BLK-0003": ["plan"],       # MEP drawings
    "BLK-0007": ["plan"],       # Structural drawings
}
# Same map for RFIs (keyed by related blocker prefix)
_RFI_DOC_TYPE_MAP = {
    "RFI-0010": ["schedule"],
    "RFI-0011": ["schedule"],
    "RFI-0012": ["schedule"],
    "RFI-0013": ["plan", "detail", "section", "elevation"],
    "RFI-0002": ["section"],
    "RFI-0001": ["elevation"],
    "RFI-0003": ["plan"],
    "RFI-0007": ["plan"],
}


def _item_doc_types(item: dict) -> List[str]:
    """Get relevant doc_types for a blocker or RFI (for skipped-page linkage)."""
    item_id = item.get("id", "")
    result = _BLOCKER_DOC_TYPE_MAP.get(item_id) or _RFI_DOC_TYPE_MAP.get(item_id, [])
    return result


def normalize_blocker(b) -> dict:
    """Ensure a blocker is a clean dict with stable string values for display."""
    if not isinstance(b, dict):
        return {"title": _safe_str(b), "severity": "medium", "trade": "general", "coverage_status": ""}
    return {
        "id": _safe_str(b.get("id", "")),
        "title": _safe_str(b.get("title") or b.get("description") or b.get("blocker_type", "Unknown")),
        "trade": _safe_str(b.get("trade", "general")),
        "severity": _safe_str(b.get("severity", "medium")),
        "description": _safe_str(b.get("description", "")),
        "pages": format_page_refs(b.get("evidence", {}).get("pages", [])),
        "sheets": _safe_str(b.get("evidence", {}).get("sheets", [])),
        "impact_cost": _safe_str(b.get("impact_cost", "")),
        "impact_schedule": _safe_str(b.get("impact_schedule", "")),
        "unlocks": ", ".join(c.replace("_", " ").title() for c in b.get("unlocks_boq_categories", [])),
        "fix_actions": b.get("fix_actions") or b.get("fix_options") or [],
        "score_delta": b.get("score_delta_estimate", 0),
        "evidence": b.get("evidence", {}),
        "issue_type": _safe_str(b.get("issue_type", "")),
        "coverage_status": _safe_str(b.get("coverage_status", "")),
    }


def normalize_rfi(r) -> dict:
    """Ensure an RFI is a clean dict with stable string values for display."""
    if not isinstance(r, dict):
        return {"question": _safe_str(r), "trade": "general", "priority": "medium", "coverage_status": ""}
    return {
        "id": _safe_str(r.get("id", "")),
        "trade": _safe_str(r.get("trade", "general")),
        "priority": _safe_str(r.get("priority", "medium")),
        "question": _safe_str(r.get("question") or r.get("title", "Unknown")),
        "why_it_matters": _safe_str(r.get("why_it_matters", "")),
        "pages": format_page_refs(r.get("evidence", {}).get("pages", [])),
        "sheets": _safe_str(r.get("evidence", {}).get("sheets", [])),
        "suggested_resolution": _safe_str(r.get("suggested_resolution", "")),
        "confidence": r.get("evidence", {}).get("confidence", 0),
        "evidence": r.get("evidence", {}),
        "coverage_status": _safe_str(r.get("coverage_status", "")),
    }


def render_what_xboq_did(demo: DemoAnalysis):
    """Render the analysis pipeline steps as a compact row."""
    st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)

    steps_html = ""
    for step in demo.analysis_steps:
        time_str = format_duration(step.duration_s)
        has_time = step.duration_s is not None and step.duration_s > 0
        time_class = "" if has_time else " pending"
        icon = step.step_name.split()[0]
        name = " ".join(step.step_name.split()[1:])
        steps_html += f"""
        <div class="pipeline-step">
            <div class="step-icon">{icon}</div>
            <div class="step-name">{name}</div>
            <div class="step-time{time_class}">{time_str}</div>
        </div>"""

    st.markdown(f'<div class="pipeline-row">{steps_html}</div>', unsafe_allow_html=True)


def render_risk_profile(demo: DemoAnalysis):
    """Render the risk profile summary."""
    risk = demo.risk_profile

    st.markdown('<div class="section-header">Risk Profile</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem;">
        <div>
            <div style="color: #52525b; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem;">Overall</div>
            <span class="risk-indicator risk-{risk.overall_risk}">{risk.overall_risk.upper()}</span>
        </div>
        <div>
            <div style="color: #52525b; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem;">Cost</div>
            <span class="risk-indicator risk-{risk.cost_risk}">{risk.cost_risk.upper()}</span>
        </div>
        <div>
            <div style="color: #52525b; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem;">Schedule</div>
            <span class="risk-indicator risk-{risk.schedule_risk}">{risk.schedule_risk.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if risk.risk_drivers:
        for driver in risk.risk_drivers[:3]:
            st.caption(f"- {driver}")


def render_trade_cards(demo: DemoAnalysis):
    """Render trade breakdown as visual cards."""
    st.markdown('<div class="section-header">Trade Breakdown</div>', unsafe_allow_html=True)

    # Filter out trades with no data
    active_trades = [t for t in demo.trade_cards if t.status != "not_found" or t.coverage_pct > 0]
    inactive_trades = [t for t in demo.trade_cards if t.status == "not_found" and t.coverage_pct == 0]

    if active_trades:
        cols = st.columns(min(len(active_trades), 3))
        for i, trade in enumerate(active_trades):
            with cols[i % 3]:
                # Coverage color
                if trade.coverage_pct >= 80:
                    cov_class = "high"
                elif trade.coverage_pct >= 40:
                    cov_class = "medium"
                else:
                    cov_class = "low"

                blocker_html = ""
                if trade.top_blocker:
                    blocker_html = '<div class="trade-card-blocker">⛔ Blocked</div>'

                st.markdown(f"""
                <div class="trade-card status-{trade.status}">
                    <div class="trade-card-header">
                        <span class="trade-card-icon">{trade.icon}</span>
                        <span class="trade-card-name">{trade.trade_display}</span>
                    </div>
                    <div class="trade-card-coverage {cov_class}">{trade.coverage_pct:.0f}%</div>
                    <div class="trade-card-stats">
                        ✅ {trade.priceable_items} priceable &nbsp;|&nbsp; ⛔ {trade.blocked_items} blocked
                    </div>
                    {blocker_html}
                </div>
                """, unsafe_allow_html=True)

    # Show inactive trades as subtle caption (not as a banner-like expander)
    if inactive_trades:
        inactive_names = ", ".join([t.trade_display for t in inactive_trades])
        st.caption(f"Not in drawings: {inactive_names}")


def render_top_blockers_preview(demo: DemoAnalysis):
    """Render top blockers as a quick preview with inline page references."""
    # Filter out MEP FYI blockers for the count display
    real_blockers = [b for b in demo.top_blockers if not is_mep_fyi_blocker(b)]
    mep_fyi_blockers = [b for b in demo.top_blockers if is_mep_fyi_blocker(b)]

    st.markdown(f'<div class="section-header">Blockers ({len(real_blockers)})</div>', unsafe_allow_html=True)

    if not real_blockers:
        st.markdown('<span style="color: #4ade80; font-size: 0.85rem;">No blockers detected</span>', unsafe_allow_html=True)
    else:
        for b in real_blockers[:3]:
            title = b.get("title") or b.get("description", "Unknown")
            severity = b.get("severity", "medium")
            trade = b.get("trade", "general")

            # Build page ref suffix
            pages = b.get("evidence", {}).get("pages", [])
            page_ref = ""
            if pages:
                page_ref = f' <span style="color:#71717a;font-size:0.75rem;">\u2014 {format_page_refs(pages)}</span>'

            sev_color = "#f87171" if severity in ["critical", "high"] else "#fbbf24" if severity == "medium" else "#4ade80"
            st.markdown(f"""
            <div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                <span style="color: {sev_color}; font-weight: 600; font-size: 0.75rem; text-transform: uppercase;">{trade}</span>
                <div style="color: #d4d4d8; font-size: 0.85rem; margin-top: 0.15rem;">{title}{page_ref}</div>
            </div>
            """, unsafe_allow_html=True)

    # Show MEP FYI blockers as info boxes
    for b in mep_fyi_blockers[:1]:
        title = b.get("title") or b.get("description", "Unknown")
        st.markdown(f'<div class="fyi-box">\u2139\ufe0f <strong>FYI:</strong> {title}</div>', unsafe_allow_html=True)


def render_top_rfis_preview(demo: DemoAnalysis):
    """Render top RFIs as a quick preview with inline page references."""
    st.markdown(f'<div class="section-header">RFIs ({demo.rfis_count})</div>', unsafe_allow_html=True)

    if not demo.top_rfis:
        st.markdown('<span style="color: #4ade80; font-size: 0.85rem;">No RFIs needed</span>', unsafe_allow_html=True)
        return

    for r in demo.top_rfis[:3]:
        question = r.get("question") or r.get("title", "Unknown")
        priority = r.get("priority", "medium")
        trade = r.get("trade", "general")

        # Page ref suffix
        pages = r.get("evidence", {}).get("pages", [])
        page_ref = ""
        if pages:
            page_ref = f' <span style="color:#71717a;font-size:0.75rem;">\u2014 {format_page_refs(pages)}</span>'

        # FYI downgrade label
        has_ev = has_rfi_evidence(r)
        label = "FYI" if not has_ev else trade
        label_color = "#60a5fa" if not has_ev else ("#f87171" if priority in ["high", "critical"] else "#fbbf24")

        st.markdown(f"""
        <div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
            <span style="color: {label_color}; font-weight: 600; font-size: 0.75rem; text-transform: uppercase;">{label}</span>
            <div style="color: #d4d4d8; font-size: 0.85rem; margin-top: 0.15rem;">{question[:90]}{page_ref}</div>
        </div>
        """, unsafe_allow_html=True)


def render_at_a_glance_dashboard(demo: DemoAnalysis):
    """
    Render the complete at-a-glance dashboard.

    This is the main entry point for the new UI showing:
    1. What tender was analyzed
    2. What xBOQ did (steps + timing)
    3. Decision + risk profile
    4. Trade breakdown
    5. Top blockers/RFIs preview
    """
    render_project_header(demo)
    render_decision_banner(demo)

    st.markdown("---")

    # Two-column layout for steps and risk
    col1, col2 = st.columns([2, 1])
    with col1:
        render_what_xboq_did(demo)
    with col2:
        render_risk_profile(demo)

    st.markdown("---")

    # Trade breakdown
    render_trade_cards(demo)

    st.markdown("---")

    # What we CAN quantify -- always shown
    render_quantified_outputs(demo)

    st.markdown("---")

    # Sprint 19: Key Line Items (commercial, BOQ, requirements)
    render_key_line_items(demo.raw_payload)

    # Sprint 20A: Structural summary card (only if structural data exists)
    _st_payload = demo.raw_payload.get("structural_takeoff")
    if _st_payload and _st_payload.get("mode") not in (None, "error"):
        st.markdown("---")
        _st_sum = _st_payload.get("summary", {})
        _st_mode = _st_payload.get("mode", "assumption")
        _st_conf = _st_payload.get("qc", {}).get("confidence", 0)
        _mode_label = "Detected" if _st_mode == "structural" else "Assumption"
        st.markdown("##### \U0001f3d7\ufe0f Structural Takeoff")
        _sc_cols = st.columns(4)
        with _sc_cols[0]:
            st.metric("Mode", _mode_label)
        with _sc_cols[1]:
            st.metric("Concrete", f"{_st_sum.get('concrete_m3', 0):.1f} m\u00b3")
        with _sc_cols[2]:
            st.metric("Steel", f"{_st_sum.get('steel_tons', 0):.2f} t")
        with _sc_cols[3]:
            st.metric("QC Confidence", f"{_st_conf:.0%}" if isinstance(_st_conf, (int, float)) else str(_st_conf))

    st.markdown("---")

    # Blockers and RFIs side by side
    col1, col2 = st.columns(2)
    with col1:
        render_top_blockers_preview(demo)
    with col2:
        render_top_rfis_preview(demo)


def render_pdf_page_preview(pdf_path: Optional[str], page_idx: int, zoom: float = 1.5) -> Optional[bytes]:
    """Render a single PDF page to PNG bytes using fitz (PyMuPDF).

    Args:
        pdf_path: Path to PDF file (or None for demo_cache mode)
        page_idx: 0-indexed page number
        zoom: Rendering zoom factor (1.0 = 72dpi, 1.5 = 108dpi)

    Returns:
        PNG bytes if successful, None if PDF unavailable or error
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
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception:
        return None


# Streamlit-cached wrapper to avoid re-rendering same page on rerun
@st.cache_data(show_spinner=False, max_entries=50)
def _cached_page_png(pdf_path: str, page_idx: int, zoom: float = 1.5) -> Optional[bytes]:
    """Cached wrapper for render_pdf_page_preview."""
    return render_pdf_page_preview(pdf_path, page_idx, zoom)


def render_pdf_page_with_overlay(
    pdf_path: Optional[str], page_idx: int,
    bboxes: Optional[List[List[float]]] = None,
    zoom: float = 1.5,
    label_boxes: bool = False,
) -> Optional[bytes]:
    """Render a PDF page with optional bounding-box highlight overlay.

    Args:
        pdf_path: Path to PDF (or None for demo cache)
        page_idx: 0-indexed page number
        bboxes: List of [x0_rel, y0_rel, x1_rel, y1_rel, confidence?, bbox_id?] in
                page-relative coords (0.0-1.0). None = no overlay.
        zoom: Rendering zoom factor
        label_boxes: If True, draw circled number labels on each box

    Returns:
        PNG bytes with overlay drawn, or plain page if no bboxes, or None if no PDF.
    """
    base_png = render_pdf_page_preview(pdf_path, page_idx, zoom)
    if base_png is None:
        return None
    if not bboxes:
        return base_png
    try:
        import io
        from PIL import Image, ImageDraw, ImageFont
        img = Image.open(io.BytesIO(base_png)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load a small font for labels
        font = None
        if label_boxes:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except Exception:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

        box_num = 0
        for box in bboxes:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x0_rel, y0_rel, x1_rel, y1_rel = box[:4]
            conf = box[4] if len(box) > 4 else 0.5
            if conf < 0.4:
                continue  # suppress low-confidence boxes
            box_num += 1
            # Clamp to valid range
            x0_rel = max(0.0, min(1.0, x0_rel))
            y0_rel = max(0.0, min(1.0, y0_rel))
            x1_rel = max(0.0, min(1.0, x1_rel))
            y1_rel = max(0.0, min(1.0, y1_rel))
            # Convert page-relative -> pixel at rendered size
            x0 = x0_rel * img.width
            y0 = y0_rel * img.height
            x1 = x1_rel * img.width
            y1 = y1_rel * img.height
            # Color by confidence
            if conf >= 0.8:
                outline_color = (34, 197, 94, 220)   # green
                fill_color = (34, 197, 94, 30)
            elif conf >= 0.6:
                outline_color = (245, 158, 11, 220)  # yellow/amber
                fill_color = (245, 158, 11, 30)
            else:
                outline_color = (239, 68, 68, 220)   # red
                fill_color = (239, 68, 68, 30)
            draw.rectangle([x0, y0, x1, y1], fill=fill_color,
                           outline=outline_color, width=3)

            # Draw numbered label in top-left corner
            if label_boxes and font:
                label = str(box_num)
                # Draw circle background
                cx, cy = x0 + 2, y0 + 2
                r = 10
                draw.ellipse([cx, cy, cx + r*2, cy + r*2],
                             fill=(245, 158, 11, 200), outline=(255, 255, 255, 255))
                # Draw number
                draw.text((cx + r - 3, cy + r - 7), label, fill=(0, 0, 0, 255), font=font)

        img = Image.alpha_composite(img, overlay)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return base_png  # fallback to plain page on any PIL error


def render_evidence_expander(evidence: dict, title: str = "Why?",
                             pdf_path: Optional[str] = None, item_id: str = "",
                             multi_doc_index=None, key_prefix: str = ""):
    """Render an expandable evidence section with optional PDF page preview.

    Args:
        evidence: Evidence dict with pages, snippets, detected_entities, etc.
        title: Expander title
        pdf_path: Path to PDF for page rendering (None = use session_state)
        key_prefix: Prefix for widget keys to guarantee uniqueness across tabs
        item_id: Unique ID for session state keys (e.g. 'BLK-0010')
        multi_doc_index: Optional MultiDocIndex for multi-doc page resolution.
    """
    if not evidence:
        return
    # Resolve pdf_path from session state if not passed
    if pdf_path is None:
        pdf_path = st.session_state.get("_xboq_pdf_path")

    with st.expander(f"📋 {title}"):
        pages = evidence.get("pages", [])

        # Session state key for page navigation (shared by PDF viewer + jump buttons)
        _kp = key_prefix  # shorthand
        nav_key = _make_widget_key(_kp, "ev_idx", item_id, title)

        # ── PDF Page Preview with navigation ──────────────────────
        if pages and pdf_path:
            if nav_key not in st.session_state:
                st.session_state[nav_key] = 0
            current_idx = st.session_state[nav_key]
            current_idx = max(0, min(current_idx, len(pages) - 1))

            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 2, 1.2, 1.2, 1])
            with nav_col1:
                if st.button("◀ Prev", key=_make_widget_key(_kp, "ev_prev", item_id, title),
                             disabled=current_idx == 0):
                    st.session_state[nav_key] = current_idx - 1
                    st.rerun()
            with nav_col5:
                if st.button("Next ▶", key=_make_widget_key(_kp, "ev_next", item_id, title),
                             disabled=current_idx >= len(pages) - 1):
                    st.session_state[nav_key] = current_idx + 1
                    st.rerun()
            with nav_col2:
                # Sprint 9: Multi-doc page caption
                _page_label = f"PDF page {pages[current_idx] + 1}"
                if multi_doc_index:
                    _mdi_docs = multi_doc_index.get("docs", [])
                    if len(_mdi_docs) > 1:
                        _gp = pages[current_idx]
                        for _d in reversed(_mdi_docs):
                            if _gp >= _d.get("global_page_start", 0):
                                _lp = _gp - _d.get("global_page_start", 0)
                                _page_label = f"{_d.get('filename', '?')} p.{_lp + 1}"
                                break
                st.caption(
                    f"Evidence page {current_idx + 1} of {len(pages)} "
                    f"| {_page_label}"
                )
            with nav_col3:
                zoom_key = _make_widget_key(_kp, "ev_zoom", item_id, title)
                zoom_opts = {"1\u00d7": 1.0, "1.5\u00d7": 1.5, "2\u00d7": 2.0}
                sel_zoom = st.selectbox(
                    "Zoom", list(zoom_opts.keys()), index=1,
                    key=zoom_key, label_visibility="collapsed",
                )
                zoom = zoom_opts[sel_zoom]
            with nav_col4:
                overlay_key = _make_widget_key(_kp, "ev_overlay", item_id, title)
                show_overlay = st.checkbox("Show highlights", value=True, key=overlay_key)

            # Resolve per-page bboxes from evidence
            all_bboxes = evidence.get("bbox")
            current_page_bboxes = None
            if show_overlay and all_bboxes and isinstance(all_bboxes, list) and current_idx < len(all_bboxes):
                candidate = all_bboxes[current_idx]
                if isinstance(candidate, list):
                    current_page_bboxes = candidate

            # Sprint 9: resolve doc-specific PDF path for multi-doc
            _render_pdf = pdf_path
            _render_page = pages[current_idx]
            if multi_doc_index:
                _mdi_docs = multi_doc_index.get("docs", [])
                if len(_mdi_docs) > 1:
                    _gp = pages[current_idx]
                    for _d in reversed(_mdi_docs):
                        if _gp >= _d.get("global_page_start", 0):
                            _render_pdf = _d.get("path", pdf_path)
                            _render_page = _gp - _d.get("global_page_start", 0)
                            break

            page_png = render_pdf_page_with_overlay(
                _render_pdf, _render_page, current_page_bboxes, zoom,
                label_boxes=True,
            )
            if page_png:
                st.image(page_png, caption=f"Page {pages[current_idx] + 1}",
                         use_container_width=True)
            else:
                st.info(f"Page preview unavailable (page {pages[current_idx] + 1})")
        elif pages and not pdf_path:
            page_nums = [str(p + 1) for p in pages[:10]]
            st.markdown(f"**Found on pages:** {', '.join(page_nums)}")
            st.caption("📄 PDF preview not available in cached demo mode")
        elif pages:
            page_nums = [str(p + 1) for p in pages[:10]]
            st.markdown(f"**Found on pages:** {', '.join(page_nums)}")

        # ── Evidence snippets with inline citations ─────────────────
        snippets = evidence.get("snippets", [])
        sheets = evidence.get("sheets", [])
        if snippets:
            for i, snippet in enumerate(snippets[:3]):
                # Build citation text
                if i < len(pages):
                    pg = pages[i]
                    sheet = sheets[i] if i < len(sheets) else ""
                    cite = f" ({sheet + ', ' if sheet else ''}p.{pg + 1})" if pages else ""
                else:
                    cite = ""
                st.markdown(
                    f'<div class="ev-snippet">{_safe_str(snippet)[:200]}'
                    f'<span class="ev-citation">{cite}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Page-jump buttons (below snippets) ──────────────────
        if pages and pdf_path:
            jump_cols = st.columns(min(len(pages), 6))
            for i, pg in enumerate(pages[:6]):
                sheet = sheets[i] if i < len(sheets) else ""
                label = f"{sheet} p.{pg + 1}" if sheet else f"p.{pg + 1}"
                with jump_cols[i]:
                    if st.button(label, key=_make_widget_key(_kp, "ev_jump", item_id, pg)):
                        st.session_state[nav_key] = i
                        st.rerun()

        # ── Rule ID + Confidence badge ────────────────────────────
        if item_id:
            confidence = evidence.get("confidence", 0)
            conf_pct = int(confidence * 100) if confidence else 0
            conf_reason = evidence.get("confidence_reason", "")
            if confidence >= 0.8:
                conf_band, conf_css = "High", "ev-conf-high"
            elif confidence >= 0.5:
                conf_band, conf_css = "Medium", "ev-conf-med"
            else:
                conf_band, conf_css = "Low", "ev-conf-low"
            st.markdown(
                f'<span class="ev-rule-tag">{item_id}</span> '
                f'<span class="{conf_css}">● {conf_band} ({conf_pct}%)</span>'
                f'{" — " + _safe_str(conf_reason)[:80] if conf_reason else ""}',
                unsafe_allow_html=True,
            )
            if confidence and confidence < 0.5:
                st.warning("⚠️ Low confidence — verify manually")

        # ── Detected entities ─────────────────────────────────────
        entities = evidence.get("detected_entities", {})
        if entities:
            if isinstance(entities, list):
                display_items = entities[:12]
                more = len(entities) - 12 if len(entities) > 12 else 0
                items_str = ", ".join(str(v) for v in display_items)
                if more > 0:
                    items_str += f" (+{more} more)"
                st.markdown(f"**Detected:** {items_str}")
            elif isinstance(entities, dict):
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

        # ── Search attempts ───────────────────────────────────────
        search = evidence.get("search_attempts") or evidence.get("search_performed", "")
        if search:
            if isinstance(search, str):
                st.markdown(f"**Searched for:** {search}")
            elif isinstance(search, dict):
                searched_items = []
                for key, val in search.items():
                    if isinstance(val, list):
                        searched_items.extend(val[:3])
                    elif isinstance(val, str):
                        searched_items.append(val[:50])
                if searched_items:
                    st.markdown(f"**Searched for:** {', '.join(searched_items[:5])}")

        # ── Not found field ───────────────────────────────────────
        not_found = evidence.get("not_found", "")
        if not_found:
            st.markdown(f"**Not found:** {not_found}")

        # ── NOT_FOUND proof panel ────────────────────────────────
        coverage_status = evidence.get("coverage_status", "")
        if coverage_status == "not_found_after_search":
            st.markdown("---")
            st.markdown("**🔍 Search Proof** — We looked and didn't find it")

            # Searched pages count + list + jump buttons
            searched = evidence.get("searched_pages") or evidence.get("pages", [])
            if searched:
                st.markdown(f"**Pages searched:** {len(searched)} pages")
                page_chips = ", ".join(str(p + 1) for p in searched[:20])
                if len(searched) > 20:
                    page_chips += f" (+{len(searched) - 20} more)"
                st.caption(f"Pages: {page_chips}")
                # Jump buttons for searched pages (Sprint 4a)
                if pdf_path and searched:
                    sjump_cols = st.columns(min(len(searched[:8]), 8))
                    for si, spg in enumerate(searched[:8]):
                        with sjump_cols[si]:
                            if st.button(f"p.{spg+1}", key=_make_widget_key(_kp, "ev_sjump", item_id, spg)):
                                if spg in pages:
                                    st.session_state[nav_key] = pages.index(spg)
                                else:
                                    st.session_state[nav_key] = 0
                                st.rerun()

            # Search terms from search_attempts
            search_attempts = evidence.get("search_attempts", {})
            if isinstance(search_attempts, dict) and search_attempts:
                terms = []
                for key, val in search_attempts.items():
                    if key == "coverage_note":
                        continue
                    if isinstance(val, list):
                        terms.extend(str(v) for v in val[:3])
                    elif isinstance(val, str):
                        terms.append(val[:40])
                if terms:
                    st.markdown(f"**Search terms:** `{'`, `'.join(terms[:8])}`")

            # Text coverage %
            text_cov = evidence.get("text_coverage_pct")
            if text_cov is not None:
                st.markdown(f"**Text analyzed:** {text_cov:.0f}% of searched pages' content")

            # ── Show Analyzed Text from OCR cache (Sprint 4a) ────
            ocr_cache = st.session_state.get("_xboq_ocr_cache", {})
            if ocr_cache and searched:
                st.markdown("**📄 Analyzed text from searched pages:**")
                for pg in searched[:5]:  # cap at 5 pages
                    page_text = ocr_cache.get(pg) or ocr_cache.get(str(pg))
                    if not page_text:
                        continue
                    with st.expander(f"Page {pg + 1} — {len(page_text):,} chars"):
                        # Highlight search terms in display text
                        display_text = page_text[:3000]
                        if isinstance(search_attempts, dict):
                            for key, val in search_attempts.items():
                                if key == "coverage_note":
                                    continue
                                hl_terms = val if isinstance(val, list) else [val] if isinstance(val, str) else []
                                for term in hl_terms[:5]:
                                    if isinstance(term, str) and len(term) > 2:
                                        display_text = display_text.replace(
                                            term, f"**`{term}`**"
                                        )
                        st.text(display_text)
                        if len(page_text) > 3000:
                            st.caption(f"Showing first 3,000 of {len(page_text):,} characters")

            # Coverage note
            if isinstance(search_attempts, dict):
                cov_note = search_attempts.get("coverage_note", "")
                if cov_note:
                    st.caption(f"ℹ️ {cov_note}")

        # ── Confidence (fallback if no item_id was passed) ────────
        if not item_id:
            confidence = evidence.get("confidence", 0)
            if confidence:
                confidence_pct = int(confidence * 100)
                confidence_text = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
                st.caption(f"Confidence: {confidence_text} ({confidence_pct}%)")
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence - verify manually")


def render_bbox_selector(evidence: dict, item_id: str, all_blockers: list, all_rfis: list):
    """Render numbered bbox list below the PDF image with filters and detail panel.

    Each bbox row shows: number, bbox_id, confidence badge, Select button.
    Selecting a bbox shows the detail panel.
    """
    all_bboxes = evidence.get("bbox")
    pages = evidence.get("pages", [])
    if not all_bboxes:
        return

    # Flatten all boxes across pages with metadata
    flat_boxes = []
    for page_pos, page_boxes in enumerate(all_bboxes):
        for box_idx, box in enumerate(page_boxes):
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            conf = box[4] if len(box) > 4 else 0.5
            bbox_id = box[5] if len(box) > 5 else f"unknown-P{page_pos}-{box_idx}"
            page_idx = pages[page_pos] if page_pos < len(pages) else -1
            flat_boxes.append({
                "num": len(flat_boxes) + 1,
                "bbox_id": bbox_id,
                "conf": conf,
                "page_idx": page_idx,
                "page_pos": page_pos,
                "box_idx": box_idx,
            })

    if not flat_boxes:
        return

    # ── Filters ──────────────────────────────────────────────
    filter_cols = st.columns([2, 2])
    with filter_cols[0]:
        # Type filter based on detected_entities
        entities = evidence.get("detected_entities", {})
        type_options = ["All"]
        if "door_tags" in entities:
            type_options.append("Tags")
        if any(k for k in entities if "dimension" in k.lower()):
            type_options.append("Dimensions")
        if any(k for k in entities if "room" in k.lower()):
            type_options.append("Rooms")
        if any(k for k in entities if "schedule" in k.lower()):
            type_options.append("Schedules")
        if len(type_options) > 1:
            st.selectbox("Filter type", type_options, key=f"ev_btype_{item_id}")
    with filter_cols[1]:
        conf_threshold = st.slider(
            "Min confidence", 0.0, 1.0, 0.4, 0.1,
            key=f"ev_bconf_{item_id}",
        )

    # Apply confidence filter
    visible_boxes = [b for b in flat_boxes if b["conf"] >= conf_threshold]

    if not visible_boxes:
        st.caption("No boxes match current filters")
        return

    # ── Numbered bbox list ────────────────────────────────────
    st.caption(f"**{len(visible_boxes)} region(s)** detected")
    for bx in visible_boxes:
        conf = bx["conf"]
        if conf >= 0.8:
            conf_css = "ev-conf-high"
        elif conf >= 0.5:
            conf_css = "ev-conf-med"
        else:
            conf_css = "ev-conf-low"

        cols = st.columns([0.5, 2, 1.2, 1])
        with cols[0]:
            st.markdown(f'<span class="bbox-num">{bx["num"]}</span>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<span class="bbox-id">{bx["bbox_id"]}</span> p.{bx["page_idx"]+1}',
                        unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<span class="{conf_css}">{conf:.0%}</span>',
                        unsafe_allow_html=True)
        with cols[3]:
            if st.button("Select", key=_make_widget_key(_kp, "ev_sel", item_id, bx['num'])):
                st.session_state[f"ev_sel_bbox_{item_id}"] = bx["bbox_id"]
                st.rerun()

    # ── Detail panel for selected bbox ────────────────────────
    sel_bbox = st.session_state.get(f"ev_sel_bbox_{item_id}")
    if sel_bbox:
        render_bbox_detail_panel(sel_bbox, all_blockers, all_rfis)


def render_bbox_detail_panel(bbox_id: str, all_blockers: list, all_rfis: list):
    """Render detail panel for a selected bbox_id.

    Parses bbox_id to extract item_id, finds matching blocker/RFI,
    shows evidence snippets + linked RFIs with copy buttons.
    """
    # Parse bbox_id to get item_id (everything before first '-P')
    p_idx = bbox_id.find("-P")
    if p_idx > 0:
        item_id = bbox_id[:p_idx]
    else:
        item_id = bbox_id

    st.markdown(f'<div class="bbox-detail">', unsafe_allow_html=True)
    st.markdown(f"**Selected:** `{bbox_id}`")

    # Find matching blocker
    found_item = None
    item_type = ""
    for b in all_blockers:
        if b.get("id") == item_id:
            found_item = b
            item_type = "Blocker"
            break
    if not found_item:
        for r in all_rfis:
            if r.get("id") == item_id:
                found_item = r
                item_type = "RFI"
                break

    if not found_item:
        st.info(f"Item `{item_id}` not found in blockers or RFIs")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown(f"**{item_type}:** {found_item.get('title', found_item.get('question', ''))}")

    # Show snippets with copy buttons
    ev = found_item.get("evidence", {})
    snippets = ev.get("snippets", [])
    if snippets:
        st.markdown("**Evidence snippets:**")
        for i, snippet in enumerate(snippets[:5]):
            st.code(snippet, language=None)

    # Show linked RFIs (for blockers)
    if item_type == "Blocker":
        linked_rfis = [r for r in all_rfis if r.get("related_blocker_id") == item_id]
        if linked_rfis:
            st.markdown(f"**Linked RFIs ({len(linked_rfis)}):**")
            for rfi in linked_rfis:
                rfi_text = f"[{rfi.get('id', '?')}] {rfi.get('question', '')}"
                st.code(rfi_text, language=None)

    st.markdown('</div>', unsafe_allow_html=True)


def render_global_search(payload: dict):
    """Render global search input + results above the detailed tabs.

    Searches across ocr_text_cache, shows results with highlighted context
    and jump-to-page buttons.
    """
    from search import search_ocr_text

    ocr_cache = payload.get("ocr_text_cache") or st.session_state.get("_xboq_ocr_cache") or {}

    if not ocr_cache:
        return

    query = st.text_input(
        "Search analyzed text...",
        key="_xboq_search_query",
        placeholder="Search across all analyzed pages (e.g. 'door schedule', '1:100', 'D1')",
    )

    if not query:
        return

    if len(query) < 2:
        st.caption("Enter at least 2 characters")
        return

    mdi_data = payload.get("multi_doc_index")
    results = search_ocr_text(ocr_cache, query, multi_doc_index=mdi_data)
    total_count = len(results)
    display_results = results[:20]

    if total_count == 0:
        st.caption("No results found")
        return

    if total_count > 20:
        st.caption(f"**{total_count} results** for \"{query}\" (showing first 20)")
    else:
        st.caption(f"**{total_count} result(s)** for \"{query}\"")

    for i, r in enumerate(display_results):
        cols = st.columns([0.8, 6, 1.2])
        with cols[0]:
            st.markdown(
                f'<span class="search-page-pill">{r["page_display"]}</span>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            # Show snippet (already has **highlight**)
            snippet = r["snippet"][:200]
            st.markdown(f'<div class="search-result">{snippet}</div>', unsafe_allow_html=True)
        with cols[2]:
            if st.button("Jump", key=_make_widget_key(_kp, "search_jump", i, r['page_idx'])):
                st.session_state["_xboq_search_jump"] = r["page_idx"]
                st.rerun()


def render_risk_checklist_tab(payload: dict):
    """Render the Risk Checklist tab with templated keyword searches (Sprint 7/8)."""
    from risk_checklist import search_risk_items

    st.markdown("#### Risk Checklist")
    st.caption(
        "Automated scan of tender documents for common contractual risk items. "
        "Each item shows whether relevant clauses were found."
    )

    ocr_cache = payload.get("ocr_text_cache", {})
    if not ocr_cache:
        st.info("No OCR text cache available — re-run analysis to see risk checklist.")
        return

    # ── Doc-type scoped search (Sprint 8) ─────────────────────
    diagnostics = payload.get("diagnostics", {})
    page_index_data = diagnostics.get("page_index", {})
    pi_pages = page_index_data.get("pages", []) if isinstance(page_index_data, dict) else []
    page_doc_types = {}
    for p in pi_pages:
        if isinstance(p, dict):
            page_doc_types[p.get("page_idx", -1)] = p.get("doc_type", "unknown")

    # Sprint 9: document-level scope when multi-doc
    _risk_mdi = payload.get("multi_doc_index")
    _risk_selected_doc_pages = None
    if _risk_mdi and len(_risk_mdi.get("docs", [])) > 1:
        _doc_names = [d.get("filename", f"Doc {d.get('doc_id', '?')}") for d in _risk_mdi["docs"]]
        _doc_opts = ["All Documents"] + _doc_names
        _sel_doc = st.selectbox("Search within document:", _doc_opts, key="_risk_doc_scope")
        if _sel_doc != "All Documents":
            _sel_idx = _doc_names.index(_sel_doc)
            _sel_info = _risk_mdi["docs"][_sel_idx]
            _d_start = _sel_info.get("global_page_start", 0)
            _d_end = _d_start + _sel_info.get("page_count", 0)
            _risk_selected_doc_pages = set(range(_d_start, _d_end))

    st.markdown("**Search Scope:**")
    toggle_cols = st.columns(4)
    search_spec = toggle_cols[0].checkbox("Specifications", value=True, key="_risk_scope_spec")
    search_conditions = toggle_cols[1].checkbox("Conditions", value=True, key="_risk_scope_conditions")
    search_addendum = toggle_cols[2].checkbox("Addendum", value=True, key="_risk_scope_addendum")
    search_all = toggle_cols[3].checkbox("All Pages", value=False, key="_risk_scope_all")

    allowed_doc_types = None
    if not search_all:
        allowed = set()
        if search_spec:
            allowed.add("spec")
        if search_conditions:
            allowed.add("conditions")
        if search_addendum:
            allowed.add("addendum")
        if allowed:
            allowed_doc_types = allowed

    # Sprint 9: filter OCR cache by selected doc pages
    _risk_ocr_cache = ocr_cache
    if _risk_selected_doc_pages is not None:
        _risk_ocr_cache = {
            k: v for k, v in ocr_cache.items()
            if int(k) in _risk_selected_doc_pages
        }

    results = search_risk_items(
        _risk_ocr_cache,
        page_doc_types=page_doc_types if (allowed_doc_types or page_doc_types) else None,
        allowed_doc_types=allowed_doc_types,
    )

    # Sprint 13: Store risk results for Review Queue tab
    st.session_state["_risk_results"] = results

    for item in results:
        impact = item["impact"].upper()
        impact_badge = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2"}.get(impact, "\u26aa")
        found = item["found"]
        status_pill = "\u2705 Found" if found else "\u26a0\ufe0f Not Found"

        col1, col2, col3 = st.columns([0.5, 2, 1])
        col1.markdown(f"{impact_badge}")
        col2.markdown(f"**{item['label']}** \u2014 {status_pill}")
        if found:
            col3.caption(f"{len(item['hits'])} hit(s)")
        else:
            # Sprint 8: show searched metadata for NOT_FOUND
            searched_count = item.get("searched_pages_count", len(ocr_cache))
            searched_types = item.get("searched_doc_types", [])
            if searched_types:
                col3.caption(f"Searched {searched_count} pg ({', '.join(searched_types)})")
            else:
                col3.caption(f"Searched {searched_count} pages")

        if found and item["hits"]:
            with st.expander(f"Citations for {item['label']}", expanded=False):
                for hit in item["hits"][:5]:
                    pg = hit.get("page_display", f"Page {hit.get('page_idx', 0) + 1}")
                    keyword = hit.get("keyword", "")
                    snippet = hit.get("snippet", "")[:200]
                    st.markdown(f"**{pg}** \u2014 keyword: `{keyword}`")
                    if snippet:
                        st.caption(snippet)


def render_bid_strategy_tab(payload: dict):
    """Render the Bid Strategy tab with form inputs, 4 dials, and recommendations."""
    from bid_strategy_scorer import compute_bid_strategy
    from src.analysis.owner_profiles import save_profile, load_profile, list_profiles, diff_inputs

    st.markdown("#### Bid Strategy")
    st.caption(
        "Combine document analysis with your client/market knowledge. "
        "Scores show UNKNOWN when inputs are missing."
    )

    # ── Owner/Client Profile (Sprint 8) ───────────────────────
    profile_names = list_profiles()
    profile_cols = st.columns([2, 2, 1])
    with profile_cols[0]:
        owner_options = ["-- New Owner --"] + profile_names
        selected_owner = st.selectbox(
            "Owner / Client",
            owner_options,
            key="_bid_owner_select",
        )

    # Auto-load saved profile when owner selected
    _loaded_profile = None
    if selected_owner and selected_owner != "-- New Owner --":
        _loaded_profile = load_profile(selected_owner)
        if _loaded_profile and not st.session_state.get("_bid_profile_loaded") == selected_owner:
            saved_inputs = _loaded_profile.get("inputs", {})
            _key_map = {
                "relationship_level": "_bid_relationship",
                "past_work_count": "_bid_past_count",
                "last_project_date": "_bid_last_project",
                "payment_delays": "_bid_payment_delays",
                "disputes": "_bid_disputes",
                "high_co_rate": "_bid_high_co",
                "competitors": "_bid_competitors",
                "market_pressure": "_bid_market_pressure",
                "target_margin": "_bid_target_margin",
                "win_probability": "_bid_win_prob",
            }
            for input_key, ss_key in _key_map.items():
                if input_key in saved_inputs:
                    val = saved_inputs[input_key]
                    # Competitors stored as list, form expects newline-separated text
                    if input_key == "competitors" and isinstance(val, list):
                        val = "\n".join(val)
                    st.session_state[ss_key] = val
            st.session_state["_bid_profile_loaded"] = selected_owner

    # ── Input form ─────────────────────────────────────────────
    with st.form("bid_strategy_form"):
        st.markdown("**Client & Competition Inputs**")

        form_cols = st.columns(2)
        with form_cols[0]:
            relationship = st.selectbox(
                "Client Relationship",
                ["", "New", "Repeat", "Preferred"],
                key="_bid_relationship",
            )
            past_count = st.number_input("Past Projects", min_value=0, max_value=100,
                                          value=0, key="_bid_past_count")
            last_project = st.text_input("Last Project Date", placeholder="e.g. 2024-06",
                                          key="_bid_last_project")

        with form_cols[1]:
            payment_delays = st.checkbox("Payment Delays", key="_bid_payment_delays")
            disputes = st.checkbox("Disputes", key="_bid_disputes")
            high_co_rate = st.checkbox("High CO Rate", key="_bid_high_co")

        competitors_text = st.text_area(
            "Known Competitors (one per line)",
            key="_bid_competitors",
            height=80,
        )
        competitors = [c.strip() for c in competitors_text.split("\n") if c.strip()] if competitors_text else []

        slider_cols = st.columns(3)
        with slider_cols[0]:
            market_pressure = st.slider("Market Pressure", 0, 10, 5, key="_bid_market_pressure")
        with slider_cols[1]:
            target_margin = st.number_input("Target Margin %", min_value=0.0, max_value=50.0,
                                             value=0.0, step=0.5, key="_bid_target_margin")
        with slider_cols[2]:
            win_prob = st.number_input("Win Probability %", min_value=0.0, max_value=100.0,
                                        value=0.0, step=5.0, key="_bid_win_prob")

        submitted = st.form_submit_button("Calculate Strategy", use_container_width=True)

    # ── Compute scores ─────────────────────────────────────────
    inputs = {
        "relationship_level": relationship if submitted else st.session_state.get("_bid_relationship", ""),
        "past_work_count": past_count if submitted else st.session_state.get("_bid_past_count", 0),
        "last_project_date": last_project if submitted else st.session_state.get("_bid_last_project", ""),
        "payment_delays": payment_delays if submitted else st.session_state.get("_bid_payment_delays", False),
        "disputes": disputes if submitted else st.session_state.get("_bid_disputes", False),
        "high_co_rate": high_co_rate if submitted else st.session_state.get("_bid_high_co", False),
        "competitors": competitors if submitted else [],
        "market_pressure": market_pressure if submitted else st.session_state.get("_bid_market_pressure", 0),
        "target_margin": target_margin if submitted else st.session_state.get("_bid_target_margin", 0),
        "win_probability": win_prob if submitted else st.session_state.get("_bid_win_prob", 0),
    }

    # Store inputs in session state
    if submitted:
        st.session_state["_xboq_bid_inputs"] = inputs

    # ── Save / Changes since last bid (Sprint 8) ──────────────
    with profile_cols[1]:
        save_name = st.text_input(
            "Save as",
            value=selected_owner if selected_owner != "-- New Owner --" else "",
            key="_bid_save_name",
        )
    with profile_cols[2]:
        st.markdown("")  # spacer
        if st.button("Save Profile", key="_bid_save_profile"):
            if save_name and save_name.strip():
                save_profile(save_name.strip(), inputs)
                st.success(f"Saved profile for {save_name.strip()}")

    if _loaded_profile and submitted:
        changes = diff_inputs(_loaded_profile.get("inputs", {}), inputs)
        if changes:
            st.markdown("**Changes since last bid:**")
            for key, vals in changes.items():
                st.caption(f"  {key}: {vals['saved']} \u2192 {vals['current']}")

    # Sprint 20C: Pass estimating playbook to bid strategy scorer
    _ep_playbook = st.session_state.get("_ep_playbook")
    strategy = compute_bid_strategy(inputs, payload, estimating_playbook=_ep_playbook)

    # Sprint 20C: Show playbook badge if active
    if strategy.get("playbook_applied"):
        st.markdown(
            '<div style="display:inline-block;background:#e0e7ff;color:#3730a3;'
            'padding:2px 10px;border-radius:4px;font-size:0.82rem;'
            'margin-bottom:8px;">'
            '\U0001f4d8 Using Estimating Playbook</div>',
            unsafe_allow_html=True,
        )

    # ── Score dials ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Score Overview**")

    dial_cols = st.columns(4)
    dials = [
        strategy["client_fit"],
        strategy["risk_score"],
        strategy["competition_score"],
        strategy["readiness_score"],
    ]

    for col, dial in zip(dial_cols, dials):
        with col:
            score = dial["score"]
            conf = dial["confidence"]
            name = dial["name"]

            if score is not None:
                # Color based on score
                if name == "Risk Score":
                    # Risk: lower is better
                    if score <= 30:
                        color = "#4ade80"
                    elif score <= 60:
                        color = "#fbbf24"
                    else:
                        color = "#f87171"
                else:
                    # Others: higher is better
                    if score >= 70:
                        color = "#4ade80"
                    elif score >= 40:
                        color = "#fbbf24"
                    else:
                        color = "#f87171"

                conf_color = {"HIGH": "#4ade80", "MEDIUM": "#fbbf24", "LOW": "#f87171"}.get(conf, "#6b7280")
                st.markdown(f"""
                <div class="bid-dial">
                    <div class="bid-dial-label">{name}</div>
                    <div class="bid-dial-score" style="color:{color};">{score}</div>
                    <div class="bid-dial-conf" style="color:{conf_color};">{conf}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bid-dial">
                    <div class="bid-dial-label">{name}</div>
                    <div class="bid-dial-unknown">&mdash;</div>
                    <div class="bid-dial-conf" style="color:#6b7280;">UNKNOWN</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Based on breakdown ─────────────────────────────────────
    st.markdown("**Based on:**")
    for dial in dials:
        name = dial["name"]
        based_on = dial.get("based_on", [])
        if based_on:
            items = "; ".join(based_on[:4])
            st.markdown(f"- **{name}:** {items}")

    # Missing inputs
    missing = strategy.get("missing_inputs", [])
    if missing:
        st.caption(f"Missing inputs: {', '.join(missing)}")

    # ── Document-Driven Recommendations ────────────────────────
    st.markdown("---")
    st.markdown("**Document-Driven Recommendations**")
    recs = strategy.get("recommendations", [])
    for rec in recs:
        st.markdown(f'<div class="bid-rec">{rec}</div>', unsafe_allow_html=True)


def render_quantified_outputs(demo: "DemoAnalysis"):
    """Render 'What We CAN Quantify' table from drawing_overview data.

    Always shown -- even when blockers exist, even when scale is missing.
    Shows what the system actually detected and can count.
    Sprint 20B: Added page-level metrics, "\u2014" for unavailable, consistent labels.
    """
    st.markdown('<div class="section-header">What We CAN Quantify</div>', unsafe_allow_html=True)

    # Sprint 20B: Pull processing stats from payload for deep-process / OCR counts
    _ps = demo.raw_payload.get("processing_stats") or {}
    _deep = _ps.get("deep_processed_pages")
    _ocr = _ps.get("ocr_pages")

    def _display(val, suffix=""):
        """Show value or \u2014 if unavailable (None). Never show 0 for None."""
        if val is None:
            return "\u2014"
        return f"{val}{suffix}"

    # Sprint 20E: Prioritize cost-driving outputs; demote doors/windows.
    _payload = demo.raw_payload
    _boq_count = (_payload.get("boq_stats") or {}).get("total_items")
    _comm_terms = _payload.get("commercial_terms")
    _comm_count = len(_comm_terms) if isinstance(_comm_terms, list) and _comm_terms else None
    _req_by_trade = _payload.get("requirements_by_trade") or {}
    _req_count = sum(len(v) for v in _req_by_trade.values()) if _req_by_trade else None
    _finish_rows = (_payload.get("finish_takeoff") or {}).get("finish_rows")
    _finish_count = len(_finish_rows) if isinstance(_finish_rows, list) else None
    # Sprint 20F: Structural quantities count (if present)
    _st_takeoff = _payload.get("structural_takeoff") or {}
    _st_quantities = _st_takeoff.get("quantities")
    _st_count = len(_st_quantities) if isinstance(_st_quantities, list) and _st_quantities else None

    rows = [
        ("Pages", _display(demo.pages_total), "total pages in drawing set"),
        ("Deep Processed", _display(_deep), "pages with full extraction"),
        ("OCR Pages", _display(_ocr), "pages processed with OCR"),
        ("Disciplines", _display(len(demo.disciplines_detected)),
         ", ".join(demo.disciplines_detected) if demo.disciplines_detected else "none detected"),
        ("BOQ Items", _display(_boq_count), "line items extracted from BOQ pages"),
        ("Commercial Terms", _display(_comm_count), "contract terms identified"),
        ("Requirements", _display(_req_count), "specifications and standards found"),
        ("Finish Rows", _display(_finish_count), "finish schedule entries"),
        ("Structural Quantities", _display(_st_count), "structural elements detected"),
        ("Blockers", _display(demo.blockers_count), "issues blocking the bid"),
        ("RFIs", _display(demo.rfis_count), "requests for information generated"),
        ("Pages with scale", _display(demo.scale_found_pages), f"of {demo.pages_total} pages"),
        ("Rooms detected", _display(demo.room_names_found), "room labels found"),
        ("Doors detected", _display(demo.door_tags_found), "door tags found in plans"),
        ("Windows detected", _display(demo.window_tags_found), "window tags found in plans"),
    ]

    rows_html = ""
    for label, count, note in rows:
        rows_html += (
            f'<tr><td>{label}</td>'
            f'<td class="count-val">{count}</td>'
            f'<td style="color:#71717a;font-size:0.78rem;">{note}</td></tr>'
        )

    st.markdown(f"""
    <table class="quantified-table">
        <thead><tr><th>Item</th><th>Count</th><th>Detail</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


def render_key_line_items(payload: dict):
    """Show commercial terms, BOQ stats, requirements count, commercial RFIs.

    Sprint 19: All reads use .get() — graceful fallback for older payloads.
    """
    commercial = payload.get("commercial_terms", [])
    boq_stats = payload.get("boq_stats", {})
    req_by_trade = payload.get("requirements_by_trade", {})
    rfis = payload.get("rfis", [])
    commercial_rfis = [r for r in rfis if r.get("trade") == "commercial"]

    # Only render if at least one data source has content
    if not commercial and not boq_stats.get("total_items") and not req_by_trade:
        return

    st.markdown("### Key Line Items Found")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Commercial Terms", len(commercial))
    _boq_source_label = payload.get("boq_source", "pdf").upper()
    col2.metric("BOQ Items", boq_stats.get("total_items", 0),
                help=f"Source: {_boq_source_label}")
    col3.metric(
        "Requirements",
        sum(len(v) for v in req_by_trade.values()) if req_by_trade else 0,
    )
    col4.metric("Commercial RFIs", len(commercial_rfis))

    # Commercial terms mini-table (if any)
    if commercial:
        with st.expander(f"Commercial Terms ({len(commercial)})", expanded=False):
            for ct in commercial:
                val = _safe_str(ct.get("value", ""))
                unit = _safe_str(ct.get("unit", ""))
                term_label = _safe_str(ct.get("term_type", "unknown")).replace("_", " ").title()
                page = (ct.get("source_page", 0) or 0) + 1
                st.caption(f"**{term_label}**: {val}{unit} (p.{page})")

    # BOQ flags (if any)
    flagged = boq_stats.get("flagged_items", [])
    if flagged:
        with st.expander(f"BOQ Flags ({len(flagged)})", expanded=False):
            for fi in flagged[:10]:
                _flags_raw = fi.get("flags", [])
                flags_str = ", ".join(
                    _safe_str(f) for f in (_flags_raw if isinstance(_flags_raw, list) else [_flags_raw])
                )
                desc = _safe_str(fi.get("description") or "")[:80]
                st.caption(f"\u2022 {fi.get('item_no', '')}: {flags_str} \u2014 {desc}")

    # Requirements by trade summary
    if req_by_trade:
        with st.expander(f"Requirements by Trade ({sum(len(v) for v in req_by_trade.values())})", expanded=False):
            for trade, reqs in sorted(req_by_trade.items(), key=lambda x: -len(x[1])):
                st.caption(f"**{trade.title()}**: {len(reqs)} requirements")


# =============================================================================
# EXTRACTED ARTIFACTS & SELECTION/COVERAGE TABS
# =============================================================================

def render_bid_pack_tab(payload: dict):
    """Render the '📋 Bid Pack' tab with sortable tables for BOQ, schedules, requirements."""
    import pandas as pd
    from collections import defaultdict

    extraction = payload.get("extraction_summary")
    if not extraction:
        st.info("Extraction data not available. Upload a tender PDF and run analysis to see extracted BOQ items, schedules, and requirements.")
        return

    pdf_path = st.session_state.get("_xboq_pdf_path")
    pages_processed = extraction.get("pages_processed", 0)
    st.caption(f"Extracted from {pages_processed} analyzed pages")

    # ── BOQ Items (sortable table) ────────────────────────────────────
    boq_items = extraction.get("boq_items", [])
    _bid_boq_source = payload.get("boq_source", "pdf")
    st.markdown(f"#### BOQ Line Items ({len(boq_items)})")
    if _bid_boq_source == "excel":
        st.caption("📊 BOQ items extracted from Excel file")
    if boq_items:
        # Sprint 21C: Show source columns for Excel BOQ items
        _has_excel_source = any(item.get("source_file") for item in boq_items)
        if _has_excel_source:
            df_boq = pd.DataFrame([
                {
                    "Item No": _safe_str(item.get("item_no", "")),
                    "Description": _safe_str(item.get("description", ""))[:100],
                    "Unit": _safe_str(item.get("unit", "")) or "\u2014",
                    "Qty": item.get("qty", ""),
                    "Rate": item.get("rate", ""),
                    "File": _safe_str(item.get("source_file", "")),
                    "Sheet": _safe_str(item.get("source_sheet", "")),
                    "Row": item.get("source_row", ""),
                }
                for item in boq_items
            ])
        else:
            df_boq = pd.DataFrame([
                {
                    "Item No": _safe_str(item.get("item_no", "")),
                    "Description": _safe_str(item.get("description", ""))[:100],
                    "Unit": _safe_str(item.get("unit", "")) or "\u2014",
                    "Qty": item.get("qty", ""),
                    "Rate": item.get("rate", ""),
                    "Page": (item.get("source_page", 0) or 0) + 1,
                }
                for item in boq_items
            ])
        st.dataframe(df_boq, use_container_width=True, hide_index=True)

        # Page jump viewer
        if pdf_path:
            boq_pages = sorted(set(int(p) for p in df_boq["Page"].dropna().unique() if p))
            if boq_pages:
                jump_col1, jump_col2 = st.columns([1, 3])
                with jump_col1:
                    sel_page = st.selectbox("Jump to page:", boq_pages, key="boq_page_jump")
                with jump_col2:
                    if st.button("Show Page", key="boq_show_page"):
                        page_png = _cached_page_png(pdf_path, sel_page - 1, zoom=1.5)
                        if page_png:
                            st.image(page_png, caption=f"Page {sel_page}", use_container_width=True)
    else:
        st.info("No BOQ items found in this document. This may mean the tender does not include a BOQ, or the pages were not processed. Try increasing the OCR budget.")

    # ── Schedules (sortable, grouped by type) ─────────────────────────
    schedules = extraction.get("schedules", [])
    st.markdown(f"#### Door / Window / Finish Schedules ({len(schedules)})")
    if schedules:
        grouped = defaultdict(list)
        for s in schedules:
            grouped[s.get("schedule_type", "other")].append(s)

        for stype, items in sorted(grouped.items()):
            st.markdown(f"**{stype.replace('_', ' ').title()} Schedule** ({len(items)})")
            rows = []
            for item in items:
                fields = item.get("fields", {})
                row = {"Mark": _safe_str(item.get("mark", ""))}
                if isinstance(fields, dict):
                    for k, v in list(fields.items())[:5]:
                        row[k.replace("_", " ").title()] = _safe_str(v)
                else:
                    row["Fields"] = _safe_str(fields)
                row["Sheet"] = _safe_str(item.get("sheet_id", "")) or "\u2014"
                row["Page"] = (item.get("source_page", 0) or 0) + 1
                rows.append(row)
            df_sched = pd.DataFrame(rows)
            st.dataframe(df_sched, use_container_width=True, hide_index=True)
    else:
        st.info("No door/window/finish schedules found. The drawing set may not contain schedule pages, or they may have been classified differently.")

    # ── Requirements (grouped by category/trade) ──────────────────────
    requirements = extraction.get("requirements", [])
    st.markdown(f"#### Notes & Requirements ({len(requirements)})")
    if requirements:
        by_cat = defaultdict(list)
        for r in requirements:
            by_cat[r.get("category", "general")].append(r)

        for cat, reqs in sorted(by_cat.items()):
            st.markdown(f"**{cat.replace('_', ' ').title()}** ({len(reqs)})")
            df_req = pd.DataFrame([
                {
                    "Requirement": _safe_str(r.get("text", ""))[:120],
                    "Confidence": f"{int(r.get('confidence', 0) * 100)}%",
                    "Page": (r.get("source_page", 0) or 0) + 1,
                }
                for r in sorted(reqs, key=lambda x: x.get("confidence", 0), reverse=True)
            ])
            st.dataframe(df_req, use_container_width=True, hide_index=True)
    else:
        st.info("No specification requirements found. The tender may not contain spec/notes pages, or they were not in the OCR budget.")

    # ── Callout Counts ────────────────────────────────────────────────
    callouts = extraction.get("callouts", [])
    if callouts:
        st.markdown(f"#### Drawing Callouts ({len(callouts)})")
        type_counts = {}
        for c in callouts:
            ctype = c.get("callout_type", "other")
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        cols = st.columns(min(len(type_counts), 5))
        for i, (ctype, count) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
            if i < 5:
                cols[i].metric(ctype.replace("_", " ").title(), count)

    # ── Addenda & Corrigenda (Sprint 6) ──────────────────────────────
    addenda = payload.get("addendum_index", [])
    st.markdown(f"#### Addenda & Corrigenda ({len(addenda)})")
    if addenda:
        for a in addenda:
            a_no = a.get("addendum_no", "?")
            a_title = a.get("title", "Untitled")
            a_date = f" ({a['date']})" if a.get("date") else ""
            with st.expander(f"Addendum {a_no}{a_date} — {a_title}"):
                changes = a.get("changes", [])
                if changes:
                    st.markdown("**Changes:**")
                    for ch in changes[:10]:
                        ch_type = ch.get("type", "").replace("_", " ").title()
                        revised = ch.get("revised", "")[:100]
                        st.markdown(f"- {ch_type}: {revised}")
                boq_ch = a.get("boq_changes", [])
                if boq_ch:
                    st.markdown("**BOQ Changes:**")
                    for bc in boq_ch:
                        st.markdown(f"- Item {bc.get('item_no', '?')}: new value {bc.get('new_value', '?')}")
                date_ch = a.get("date_changes", [])
                if date_ch:
                    st.markdown("**Date Changes:**")
                    for dc in date_ch:
                        dtype = dc.get("date_type", "").replace("_", " ").title()
                        st.markdown(f"- {dtype}: {dc.get('new_date', '')}")
                clarifs = a.get("clarifications", [])
                if clarifs:
                    st.markdown("**Clarifications:**")
                    for cl in clarifs[:5]:
                        st.markdown(f"- {_safe_str(cl)[:150]}")
    else:
        st.caption("No addenda detected in tender documents.")

    # ── Conflicts (Sprint 6 + Sprint 7 confidence + Sprint 9 supersedes) ─
    conflicts = payload.get("conflicts", [])
    # Sprint 9: count intentional revisions
    _revision_count = sum(1 for c in conflicts if c.get("resolution") == "intentional_revision")
    _true_conflict_count = len(conflicts) - _revision_count
    st.markdown(f"#### Conflicts ({len(conflicts)})")
    if _revision_count:
        st.caption(f"{_true_conflict_count} true conflict(s) + {_revision_count} intentional revision(s)")
    if conflicts:
        # Sprint 7: confidence filter toggle
        _filter_cols = st.columns(2)
        with _filter_cols[0]:
            show_low_conf = st.checkbox(
                "Show low-confidence conflicts",
                value=False,
                key="show_low_conf_conflicts",
            )
        # Sprint 9: supersedes toggle
        with _filter_cols[1]:
            hide_revisions = st.checkbox(
                "Hide intentional revisions",
                value=False,
                key="hide_intentional_revisions",
            )

        filtered_conflicts = [
            c for c in conflicts
            if (show_low_conf or c.get("delta_confidence", 1.0) >= 0.7)
            and (not hide_revisions or c.get("resolution") != "intentional_revision")
        ]

        # Sprint 9: sort — true conflicts first, intentional revisions last
        filtered_conflicts.sort(
            key=lambda c: (0 if c.get("resolution") is None else 1)
        )

        conflict_rows = []
        for c in filtered_conflicts:
            ctype = c.get("type", "")
            # Sprint 9: prefix with icon for intentional revisions
            _type_display = ctype.replace("_", " ").title()
            if c.get("resolution") == "intentional_revision":
                _type_display = "\U0001f504 " + _type_display
            row = {
                "Type": _type_display,
                "Item": c.get("item_no") or c.get("mark") or "",
                "Detail": "",
                "Confidence": "",
                "Resolution": c.get("resolution") or "",
                "Base Page": "",
                "Addendum Page": "",
            }
            changes = c.get("changes", [])
            if changes:
                row["Detail"] = "; ".join(
                    f"{ch['field']}: {ch['base_value']} \u2192 {ch['addendum_value']}"
                    for ch in changes[:3]
                )
            elif c.get("text"):
                row["Detail"] = _safe_str(c["text"])[:80]
            elif c.get("addendum_text"):
                row["Detail"] = _safe_str(c["addendum_text"])[:80]
            # Sprint 7: confidence column
            dc = c.get("delta_confidence")
            if dc is not None:
                row["Confidence"] = f"{dc:.0%}"
            if c.get("base_page") is not None:
                row["Base Page"] = str((c.get("base_page") or 0) + 1)
            if c.get("addendum_page") is not None:
                row["Addendum Page"] = str((c.get("addendum_page") or 0) + 1)
            conflict_rows.append(row)

        if conflict_rows:
            import pandas as pd
            df_conflicts = pd.DataFrame(conflict_rows)
            st.dataframe(df_conflicts, use_container_width=True, hide_index=True)
        else:
            st.caption("All conflicts are below the confidence threshold or hidden. Toggle checkboxes above to show.")
    else:
        st.caption("No conflicts detected between base documents and addenda.")

    # ── Reconciliation Findings + Actions (Sprint 7/8) ────────────────
    recon_findings = payload.get("reconciliation_findings", [])
    if recon_findings:
        from src.analysis.recon_actions import (
            generate_proposals, create_recon_rfi, create_recon_assumption,
        )
        proposals = generate_proposals(recon_findings)

        st.markdown(f"#### Scope Reconciliation ({len(recon_findings)} findings)")

        for idx, proposal in enumerate(proposals):
            finding = proposal["finding"]
            p_rfi = proposal["proposed_rfi"]
            p_asmp = proposal["proposed_assumption"]

            impact = finding.get("impact", "").upper()
            impact_badge = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2"}.get(impact, "\u26aa")
            ftype = finding.get("type", "").title()
            desc = _safe_str(finding.get("description", ""))[:120]
            st.markdown(f"- {impact_badge} **[{ftype}]** {desc}")
            action = finding.get("suggested_action", "")
            if action:
                st.caption(f"   \u2192 {action}")

            with st.expander(f"Proposed: {p_rfi['question'][:80]}", expanded=False):
                st.markdown(f"**RFI Question:** {p_rfi['question']}")
                st.markdown(f"**Category:** {p_rfi['category']} | **Confidence:** {p_rfi['confidence']:.0%}")
                pages_display = [str(p + 1) for p in p_rfi["evidence_refs"].get("pages", [])]
                if pages_display:
                    st.caption(f"Evidence pages: {', '.join(pages_display)}")

                st.markdown(f"**Assumption:** {p_asmp['assumption_text'][:150]}")
                st.caption(f"Risk level: {p_asmp['risk_level']} | Scope: {p_asmp['scope']}")

                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("\U0001f4dd Create RFI", key=f"recon_rfi_{idx}"):
                        existing_rfis = payload.get("rfis", []) + st.session_state.get("_recon_rfis", [])
                        new_rfi = create_recon_rfi(p_rfi, existing_rfis, finding)
                        recon_rfis = st.session_state.get("_recon_rfis", [])
                        recon_rfis.append(new_rfi)
                        st.session_state["_recon_rfis"] = recon_rfis
                        st.success(f"Created {new_rfi['id']}")
                with btn_cols[1]:
                    if st.button("\U0001f4cb Add Assumption", key=f"recon_asmp_{idx}"):
                        existing_asmps = st.session_state.get("_assumptions_log", [])
                        new_asmp = create_recon_assumption(p_asmp, existing_asmps, finding)
                        existing_asmps.append(new_asmp)
                        st.session_state["_assumptions_log"] = existing_asmps
                        st.success(f"Added {new_asmp['id']}")

        # Show running counts
        recon_rfi_count = len(st.session_state.get("_recon_rfis", []))
        asmp_count = len(st.session_state.get("_assumptions_log", []))
        if recon_rfi_count or asmp_count:
            st.caption(f"Session: {recon_rfi_count} RFI(s) created, {asmp_count} assumption(s) logged")

    # ── Sprint 9: Assumption Review Panel ─────────────────────────────
    assumptions_log = st.session_state.get("_assumptions_log", [])
    if assumptions_log:
        st.markdown("---")
        st.markdown("#### Assumption Review")
        for a_idx, asmp in enumerate(assumptions_log):
            a_status = asmp.get("status", "draft")
            status_icons = {"draft": "📝", "accepted": "✅", "rejected": "❌"}
            a_icon = status_icons.get(a_status, "📝")

            with st.expander(
                f"{a_icon} {asmp.get('id', '')} — {asmp.get('title', '')}",
                expanded=(a_status == "draft"),
            ):
                st.markdown(f"**Text:** {asmp.get('text', '')[:200]}")
                st.caption(
                    f"Risk: {asmp.get('risk_level', '—')} | "
                    f"Source: {asmp.get('source', '—')} | "
                    f"Status: {a_status}"
                )

                acol1, acol2, acol3 = st.columns([1, 1, 2])
                with acol1:
                    if st.button(
                        "✅ Accept", key=f"accept_{asmp.get('id', a_idx)}",
                        disabled=(a_status == "accepted"),
                    ):
                        from src.analysis.recon_actions import update_assumption_status
                        assumptions_log[a_idx] = update_assumption_status(
                            asmp, "accepted", approved_by="User",
                        )
                        st.session_state["_assumptions_log"] = assumptions_log
                        st.rerun()
                with acol2:
                    if st.button(
                        "❌ Reject", key=f"reject_{asmp.get('id', a_idx)}",
                        disabled=(a_status == "rejected"),
                    ):
                        from src.analysis.recon_actions import update_assumption_status
                        assumptions_log[a_idx] = update_assumption_status(
                            asmp, "rejected", approved_by="User",
                        )
                        st.session_state["_assumptions_log"] = assumptions_log
                        st.rerun()
                with acol3:
                    cost_val = st.number_input(
                        "Cost Impact ($)",
                        value=float(asmp.get("cost_impact") or 0),
                        step=100.0,
                        key=f"cost_{asmp.get('id', a_idx)}",
                    )
                    if cost_val != float(asmp.get("cost_impact") or 0):
                        assumptions_log[a_idx]["cost_impact"] = cost_val if cost_val != 0.0 else None
                        st.session_state["_assumptions_log"] = assumptions_log

                scope_val = st.text_input(
                    "Scope Tag",
                    value=asmp.get("scope_tag", ""),
                    key=f"scope_{asmp.get('id', a_idx)}",
                )
                if scope_val != asmp.get("scope_tag", ""):
                    assumptions_log[a_idx]["scope_tag"] = scope_val
                    st.session_state["_assumptions_log"] = assumptions_log

        # Summary counts
        draft_cnt = sum(1 for a in assumptions_log if a.get("status") == "draft")
        accepted_cnt = sum(1 for a in assumptions_log if a.get("status") == "accepted")
        rejected_cnt = sum(1 for a in assumptions_log if a.get("status") == "rejected")
        st.caption(f"📊 {draft_cnt} draft, {accepted_cnt} accepted, {rejected_cnt} rejected")

    # ── CSV Exports + ZIP Bundle (stable schema) ──────────────────────
    st.markdown("---")
    st.markdown("#### Export Data")

    # Sprint 13: Export approval filter toggle
    _include_drafts = st.checkbox(
        "Include draft/unreviewed items in exports",
        value=True,
        key="export_drafts_toggle",
        help="When unchecked, only approved/accepted/reviewed items appear in exports.",
    )

    import csv
    import zipfile

    # Sprint 18: Standardized filename helper
    def _demo_filename(base: str, pname: str, ext: str) -> str:
        """Build '{ProjectName}_{base}_{date}.{ext}' or '{base}.{ext}'."""
        from datetime import date as _date_mod
        if pname:
            safe = pname.replace(" ", "_")[:30]
            return f"{safe}_{base}_{_date_mod.today().isoformat()}.{ext}"
        return f"{base}.{ext}"

    _export_pname = st.session_state.get("_active_project_name", "")

    csv_buffers = {}

    # BOQ CSV (Sprint 21C: includes Excel source columns when present)
    if boq_items:
        buf = io.StringIO()
        _boq_csv_fields = [
            "item_no", "description", "unit", "qty", "rate",
            "source_page", "source_file", "source_sheet", "source_row",
        ]
        writer = csv.DictWriter(buf, fieldnames=_boq_csv_fields, extrasaction="ignore")
        writer.writeheader()
        for item in boq_items:
            writer.writerow({
                "item_no": item.get("item_no", ""),
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
                "qty": item.get("qty", ""),
                "rate": item.get("rate", ""),
                "source_page": (item.get("source_page", 0) or 0) + 1,
                "source_file": item.get("source_file", ""),
                "source_sheet": item.get("source_sheet", ""),
                "source_row": item.get("source_row", ""),
            })
        csv_buffers["boq.csv"] = buf.getvalue()

    # Schedules CSV
    if schedules:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["schedule_type", "mark", "fields", "source_page"])
        writer.writeheader()
        for s in schedules:
            fields = s.get("fields", {})
            fields_str = "; ".join(f"{k}: {v}" for k, v in fields.items()) if isinstance(fields, dict) else str(fields)
            writer.writerow({
                "schedule_type": s.get("schedule_type", ""),
                "mark": s.get("mark", ""),
                "fields": fields_str,
                "source_page": (s.get("source_page", 0) or 0) + 1,
            })
        csv_buffers["schedules.csv"] = buf.getvalue()

    # Requirements CSV
    if requirements:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["category", "text", "confidence", "source_page"])
        writer.writeheader()
        for r in requirements:
            writer.writerow({
                "category": r.get("category", ""),
                "text": r.get("text", ""),
                "confidence": r.get("confidence", 0),
                "source_page": (r.get("source_page", 0) or 0) + 1,
            })
        csv_buffers["requirements.csv"] = buf.getvalue()

    # RFIs CSV — merge pipeline + recon-generated RFIs (Sprint 8)
    rfi_list = payload.get("rfis", [])
    recon_rfis = st.session_state.get("_recon_rfis", [])
    all_rfis = rfi_list + recon_rfis
    if all_rfis:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "id", "trade", "priority", "question", "coverage_status",
            "suggested_resolution", "source",
        ])
        writer.writeheader()
        for r in all_rfis:
            if isinstance(r, dict):
                writer.writerow({
                    "id": r.get("id", ""),
                    "trade": r.get("trade", ""),
                    "priority": r.get("priority", ""),
                    "question": r.get("question", ""),
                    "coverage_status": r.get("coverage_status", ""),
                    "suggested_resolution": r.get("suggested_resolution", ""),
                    "source": r.get("source", "pipeline"),
                })
        csv_buffers["rfis.csv"] = buf.getvalue()

    # Assumptions CSV (Sprint 8 + Sprint 9 fields)
    assumptions_log = st.session_state.get("_assumptions_log", [])
    if assumptions_log:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "id", "title", "text", "status", "cost_impact", "scope_tag",
            "impact_if_wrong", "risk_level", "basis_pages",
            "approved_by", "approved_at", "source", "created_at",
        ])
        writer.writeheader()
        for a in assumptions_log:
            writer.writerow({
                "id": a.get("id", ""),
                "title": a.get("title", ""),
                "text": a.get("text", ""),
                "status": a.get("status", "draft"),
                "cost_impact": a.get("cost_impact", ""),
                "scope_tag": a.get("scope_tag", ""),
                "impact_if_wrong": a.get("impact_if_wrong", ""),
                "risk_level": a.get("risk_level", ""),
                "basis_pages": ";".join(str(p + 1) for p in a.get("basis_pages", [])),
                "approved_by": a.get("approved_by", ""),
                "approved_at": a.get("approved_at", ""),
                "source": a.get("source", ""),
                "created_at": a.get("created_at", ""),
            })
        csv_buffers["assumptions.csv"] = buf.getvalue()

    # Sprint 9: Exclusions & Clarifications export
    if assumptions_log:
        non_draft = [a for a in assumptions_log if a.get("status") in ("accepted", "rejected")]
        if non_draft:
            from src.analysis.recon_actions import generate_exclusions_clarifications
            excl_txt, excl_csv = generate_exclusions_clarifications(assumptions_log)
            csv_buffers["exclusions_clarifications.txt"] = excl_txt
            csv_buffers["exclusions_clarifications.csv"] = excl_csv

    # Sprint 10: Grouped RFIs / Assumptions CSV exports
    rfi_clusters = payload.get("rfi_clusters", [])
    if rfi_clusters:
        from src.analysis.rfi_clustering import export_grouped_csv
        csv_buffers["grouped_rfis.csv"] = export_grouped_csv(rfi_clusters, item_type="rfi")

    # Bid Summary markdown (Sprint 6 + Sprint 9 assumptions)
    from bid_summary import generate_bid_summary_markdown
    summary_md = generate_bid_summary_markdown(payload, assumptions=assumptions_log)
    csv_buffers["bid_summary.md"] = summary_md

    # Standalone Bid Summary downloads
    _dl_cols = st.columns(2)
    _dl_cols[0].download_button(
        label="\U0001f4c4 Download Bid Summary (Markdown)",
        data=summary_md,
        file_name=_demo_filename("Bid_Summary", _export_pname, "md"),
        mime="text/markdown",
        key="bid_summary_md_dl",
    )
    # Sprint 7: PDF download
    try:
        from bid_summary_pdf import generate_bid_summary_pdf
        # Sprint 18: Pass branding params when in demo mode
        _cover_brand = False
        _pdf_watermark = ""
        try:
            from src.demo.demo_config import is_demo_mode as _eb_dm
            _cover_brand = _eb_dm()
            if _eb_dm() and st.session_state.get("_xboq_demo_watermark", False):
                _pdf_watermark = "DEMO"
        except Exception:
            pass
        pdf_bytes = generate_bid_summary_pdf(
            payload, assumptions=assumptions_log, include_drafts=_include_drafts,
            project_name=_export_pname,
            cover_branding=_cover_brand,
            watermark=_pdf_watermark,
        )
        csv_buffers["bid_summary.pdf"] = pdf_bytes
        _dl_cols[1].download_button(
            label="\U0001f4c4 Download Bid Summary (PDF)",
            data=pdf_bytes,
            file_name=_demo_filename("Bid_Summary", _export_pname, "pdf"),
            mime="application/pdf",
            key="bid_summary_pdf_dl",
        )
    except Exception:
        pass  # ReportLab not available — skip PDF

    # Sprint 11: Quantities CSV
    quantities = payload.get("quantities", [])
    if quantities:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["item", "unit", "qty", "confidence", "source_type", "trade"])
        writer.writeheader()
        for q in quantities:
            writer.writerow({
                "item": q.get("item", ""),
                "unit": q.get("unit", ""),
                "qty": q.get("qty", ""),
                "confidence": q.get("confidence", ""),
                "source_type": q.get("source_type", ""),
                "trade": q.get("trade", ""),
            })
        csv_buffers["quantities.csv"] = buf.getvalue()

    # Sprint 11: Pricing guidance TXT
    pricing_guidance = payload.get("pricing_guidance")
    if pricing_guidance and isinstance(pricing_guidance, dict):
        pg_lines = ["PRICING GUIDANCE", "=" * 40, ""]
        cont = pricing_guidance.get("contingency_range", {})
        if cont:
            pg_lines.append(f"Contingency Range: {cont.get('low_pct', 0)}% - {cont.get('high_pct', 0)}%")
            pg_lines.append(f"Recommended: {cont.get('recommended_pct', 0)}%")
            pg_lines.append(f"Rationale: {cont.get('rationale', '')}")
            pg_lines.append("")
        excl = pricing_guidance.get("recommended_exclusions", [])
        if excl:
            pg_lines.append("RECOMMENDED EXCLUSIONS:")
            for e in excl:
                pg_lines.append(f"  - {e}")
            pg_lines.append("")
        clar = pricing_guidance.get("recommended_clarifications", [])
        if clar:
            pg_lines.append("RECOMMENDED CLARIFICATIONS:")
            for c in clar:
                pg_lines.append(f"  - {c}")
            pg_lines.append("")
        ve = pricing_guidance.get("suggested_alternates_ve", [])
        if ve:
            pg_lines.append("SUGGESTED ALTERNATES / VE:")
            for v in ve:
                pg_lines.append(f"  - {v.get('item', '')} — {v.get('reason', '')}")
        csv_buffers["pricing_guidance.txt"] = "\n".join(pg_lines)

    # Sprint 11: DOCX exports
    try:
        from docx_exports import generate_rfis_docx, generate_exclusions_docx, generate_bid_summary_docx

        rfi_list_docx = payload.get("rfis", [])
        if rfi_list_docx:
            csv_buffers["rfis.docx"] = generate_rfis_docx(
                rfi_list_docx, project_id=project_id, include_drafts=_include_drafts,
            )

        csv_buffers["exclusions_clarifications.docx"] = generate_exclusions_docx(
            assumptions=assumptions_log,
            pricing_guidance=pricing_guidance,
            project_id=project_id,
        )

        # Sprint 14: Load collaboration entries if project active
        _docx_collab_entries = None
        _docx_pid = st.session_state.get("_active_project_id", "")
        if _docx_pid:
            try:
                from src.analysis.collaboration import load_collaboration
                from src.analysis.projects import project_dir as _project_dir
                _docx_collab_entries = load_collaboration(_project_dir(_docx_pid))
            except Exception:
                pass

        csv_buffers["bid_summary.docx"] = generate_bid_summary_docx(
            payload,
            bid_strategy=None,
            assumptions=assumptions_log,
            include_drafts=_include_drafts,
            collaboration_entries=_docx_collab_entries,
        )
    except Exception:
        pass  # python-docx not available — skip DOCX

    # Sprint 12: Quantity reconciliation CSV
    qty_recon_export = payload.get("quantity_reconciliation", [])
    if qty_recon_export:
        from src.analysis.quantity_reconciliation import export_reconciliation_csv
        csv_buffers["quantity_reconciliation.csv"] = export_reconciliation_csv(qty_recon_export)

    # Sprint 12: Finishes takeoff CSV
    finish_takeoff_export = payload.get("finish_takeoff")
    if finish_takeoff_export and isinstance(finish_takeoff_export, dict) and finish_takeoff_export.get("finish_rows"):
        from src.analysis.finish_takeoff import export_finishes_csv
        csv_buffers["finishes_takeoff.csv"] = export_finishes_csv(finish_takeoff_export)

    # Sprint 12: Feedback JSONL download
    feedback_log = st.session_state.get("_feedback_log", [])
    if feedback_log:
        feedback_jsonl = "\n".join(json.dumps(entry, default=str) for entry in feedback_log)
        csv_buffers["feedback.jsonl"] = feedback_jsonl

    # Sprint 15: Evidence Appendix PDF
    try:
        from evidence_appendix_pdf import generate_evidence_appendix_pdf
        csv_buffers["evidence_appendix.pdf"] = generate_evidence_appendix_pdf(
            rfis=payload.get("rfis", []),
            conflicts=payload.get("conflicts", []),
            assumptions=assumptions_log,
            include_drafts=_include_drafts,
        )
    except Exception:
        pass  # ReportLab not available or no data

    # Sprint 15: Email Drafts
    try:
        from src.exports.email_drafts import generate_all_email_drafts
        _email_drafts = generate_all_email_drafts(
            rfis=payload.get("rfis", []),
            assumptions=assumptions_log,
            include_drafts=_include_drafts,
            project_name=project_id,
        )
        for _ed_fname, _ed_content in _email_drafts.items():
            csv_buffers[_ed_fname] = _ed_content
    except Exception:
        pass

    # Sprint 20A: Structural takeoff export files
    _st_export = payload.get("structural_takeoff")
    if _st_export and _st_export.get("mode") not in (None, "error"):
        try:
            # structural_summary.json
            _st_summary_export = _st_export.get("summary", {})
            csv_buffers["structural_summary.json"] = json.dumps(
                _st_summary_export, indent=2, default=str
            )
            # structural_qc.json
            _st_qc_export = _st_export.get("qc", {})
            csv_buffers["structural_qc.json"] = json.dumps(
                _st_qc_export, indent=2, default=str
            )
            # structural_quantities.csv
            _st_qty_export = _st_export.get("quantities", [])
            if _st_qty_export:
                import io as _st_io
                import csv as _st_csv
                _st_csv_buf = _st_io.StringIO()
                _st_fieldnames = [
                    "element_id", "type", "label", "count",
                    "width_mm", "depth_mm", "length_mm",
                    "concrete_m3", "steel_kg_total",
                    "size_source", "height_source", "steel_source",
                ]
                _st_writer = _st_csv.DictWriter(_st_csv_buf, fieldnames=_st_fieldnames)
                _st_writer.writeheader()
                for _eq in _st_qty_export:
                    _dims = _eq.get("dimensions_mm", {})
                    _steel = _eq.get("steel_kg", {})
                    _sources = _eq.get("sources", {})
                    _st_writer.writerow({
                        "element_id": _eq.get("element_id", ""),
                        "type": _eq.get("type", ""),
                        "label": _eq.get("label", ""),
                        "count": _eq.get("count", 0),
                        "width_mm": _dims.get("width", 0),
                        "depth_mm": _dims.get("depth", 0),
                        "length_mm": _dims.get("length", 0),
                        "concrete_m3": _eq.get("concrete_m3", 0),
                        "steel_kg_total": _steel.get("total", 0),
                        "size_source": _sources.get("size", ""),
                        "height_source": _sources.get("height", ""),
                        "steel_source": _sources.get("steel", ""),
                    })
                csv_buffers["structural_quantities.csv"] = _st_csv_buf.getvalue()
        except Exception:
            pass  # structural export optional

    if csv_buffers:
        cols = st.columns(min(len(csv_buffers) + 1, 6))
        for i, (fname, content) in enumerate(csv_buffers.items()):
            cols[i].download_button(
                label=f"Download {fname}",
                data=content,
                file_name=fname,
                mime="text/csv",
                key=f"csv_dl_{fname}",
            )

        # ZIP bundle
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, content in csv_buffers.items():
                zf.writestr(fname, content)
        zip_buf.seek(0)
        cols[-1].download_button(
            label="Download All (ZIP)",
            data=zip_buf.getvalue(),
            file_name="xboq_export.zip",
            mime="application/zip",
            key="zip_dl",
        )

        # ── Sprint 14: Submission Pack (structured 5-folder ZIP) ──────
        st.markdown("---")
        st.markdown("##### 📦 Submission Pack")
        try:
            from submission_pack import generate_submission_pack, get_submission_manifest

            _sp_manifest = get_submission_manifest(csv_buffers)
            with st.expander("Preview folder structure"):
                for _sp_folder, _sp_files in sorted(_sp_manifest.items()):
                    st.markdown(f"**{_sp_folder}/**")
                    for _sp_f in _sp_files:
                        st.caption(f"  · {_sp_f}")

            # Build collaboration appendix if project context available
            _sp_collab_text = ""
            _sp_pid = st.session_state.get("_active_project_id", "")
            _sp_pname = st.session_state.get("_active_project_name", "")
            if _sp_pid:
                try:
                    from src.analysis.collaboration import load_collaboration, build_collaboration_appendix
                    from src.analysis.projects import project_dir as _project_dir
                    _sp_entries = load_collaboration(_project_dir(_sp_pid))
                    _sp_collab_text = build_collaboration_appendix(_sp_entries)
                except Exception:
                    pass

            _sp_bytes = generate_submission_pack(
                csv_buffers,
                project_id=_sp_pid or project_id,
                project_name=_sp_pname,
                collaboration_appendix=_sp_collab_text,
            )
            st.download_button(
                label="📦 Download Submission Pack",
                data=_sp_bytes,
                file_name=_demo_filename("Submission_Pack", _export_pname or project_id, "zip"),
                mime="application/zip",
                key="submission_pack_dl",
            )

            # Sprint 17: Success banner + counts
            _sp_rfi_count = len([r for r in payload.get("rfis", []) if r.get("status") == "approved"])
            _sp_qty_count = len(payload.get("quantities", []))
            _sp_file_count = len(csv_buffers)
            st.success(
                f"Submission pack ready — {_sp_file_count} files across 5 folders. "
                f"{_sp_rfi_count} approved RFIs, {_sp_qty_count} quantities included."
            )

            # Sprint 18: Enhanced narration panel (demo mode only)
            try:
                from src.demo.demo_config import is_demo_mode as _n18_dm
                if _n18_dm():
                    from src.demo.narration import build_narration_script
                    _narr18 = build_narration_script(
                        payload, project_name=_sp_pname or project_id)

                    _show_pv = st.checkbox(
                        "Presenter View", key="_xboq_presenter_view", value=False)

                    if st.button("Copy 60s narration", key="copy_narration_60s"):
                        st.code(_narr18, language=None)
                        st.caption("Copy the text above for your YC demo narration.")

                    if _show_pv:
                        with st.expander("Narration Panel", expanded=True):
                            st.markdown(_narr18)
            except Exception:
                pass

        except Exception:
            pass
    else:
        st.caption("No data to export.")


# =============================================================================
# SPRINT 14: COLLABORATION WIDGET HELPER
# =============================================================================

def _render_collab_widget(entity_type: str, entity_id: str, widget_key: str = ""):
    """
    Render an inline collaboration widget (comments, assign, due date) for an entity.

    Only rendered when a project is active. Loads/saves from collaboration.jsonl.

    Args:
        entity_type: One of "rfi", "conflict", "assumption", "quantity", "review_item"
        entity_id: Unique identifier for the entity
        widget_key: Unique suffix for Streamlit widget keys (to avoid duplicates)
    """
    active_pid = st.session_state.get("_active_project_id", "")
    if not active_pid:
        return

    try:
        from src.analysis.collaboration import (
            load_collaboration, get_entity_collaboration,
            make_collaboration_entry, append_collaboration,
        )
        from src.analysis.projects import project_dir as _project_dir

        proj_dir = _project_dir(active_pid)
        all_entries = load_collaboration(proj_dir)
        collab = get_entity_collaboration(all_entries, entity_type, entity_id)

        # Show last 3 comments
        comments = collab.get("comments", [])
        if comments:
            for c in comments[-3:]:
                _author = c.get("author", "?")
                _ts = c.get("timestamp", "")[:16]
                _text = c.get("text", "")
                st.caption(f"💬 [{_ts}] **{_author}**: {_text}")

        assigned = collab.get("assigned_to", "")
        due = collab.get("due_date", "")
        if assigned or due:
            _info_parts = []
            if assigned:
                _info_parts.append(f"Assigned: {assigned}")
            if due:
                _info_parts.append(f"Due: {due}")
            st.caption(" · ".join(_info_parts))

        # Inline form
        _k = f"collab_{entity_type}_{entity_id}_{widget_key}"
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        with c1:
            _comment = st.text_input("Comment", key=f"{_k}_comment", label_visibility="collapsed", placeholder="Add comment...")
        with c2:
            _assign = st.text_input("Assign", key=f"{_k}_assign", label_visibility="collapsed", placeholder="Assign to...")
        with c3:
            _due = st.text_input("Due", key=f"{_k}_due", label_visibility="collapsed", placeholder="Due date")
        with c4:
            if st.button("💾", key=f"{_k}_save", help="Save collaboration"):
                saved = False
                if _comment.strip():
                    entry = make_collaboration_entry(entity_type, entity_id, "comment",
                                                     {"text": _comment.strip()}, author=_assign or "User")
                    append_collaboration(entry, proj_dir)
                    saved = True
                if _assign.strip():
                    entry = make_collaboration_entry(entity_type, entity_id, "assign",
                                                     {"assigned_to": _assign.strip()})
                    append_collaboration(entry, proj_dir)
                    saved = True
                if _due.strip():
                    entry = make_collaboration_entry(entity_type, entity_id, "due_date",
                                                     {"due_date": _due.strip()})
                    append_collaboration(entry, proj_dir)
                    saved = True
                if saved:
                    st.rerun()
    except Exception:
        pass  # Graceful fallback


# =============================================================================
# SPRINT 10: GROUPED VIEW HELPERS
# =============================================================================

def _render_rfi_grouped_view(rfi_clusters, norm_rfis, project_id):
    """Render RFIs in clustered/grouped view."""
    st.caption(f"{len(rfi_clusters)} cluster(s) from {len(norm_rfis)} RFIs")
    for cluster in rfi_clusters:
        cid = cluster.get("cluster_id", "?")
        label = cluster.get("label", "Cluster")
        count = cluster.get("count", len(cluster.get("rfis", [])))
        merged_q = cluster.get("merged_question", "")
        trade = cluster.get("trade", "").title()
        priority = cluster.get("priority", "medium").upper()
        with st.expander(f"📁 {cid}: {label} ({count} RFIs) — {trade} [{priority}]"):
            st.markdown(f"**Merged Question:** {merged_q}")
            ev_pages = cluster.get("evidence_pages", [])
            if ev_pages:
                st.caption(f"Evidence pages: {', '.join(str(p + 1) for p in ev_pages[:10])}")
            doc_ids = cluster.get("doc_ids", [])
            if doc_ids:
                st.caption(f"Documents: {', '.join(doc_ids)}")
            st.markdown("**Individual RFIs:**")
            for ref in cluster.get("rfis", []):
                rid = ref.get("id", "?")
                rq = ref.get("question", "")[:100]
                st.markdown(f"- {rid}: {rq}")

    # Grouped CSV export
    st.markdown("---")
    from src.analysis.rfi_clustering import export_grouped_csv
    grouped_csv = export_grouped_csv(rfi_clusters, item_type="rfi")
    _gc1, _gc2 = st.columns(2)
    with _gc1:
        st.download_button(
            "📊 Download Grouped RFIs CSV", grouped_csv,
            f"grouped_rfis_{project_id}.csv", "text/csv",
            use_container_width=True, key="dl_grouped_rfis",
        )


def _render_rfi_raw_view(norm_rfis, rfis, blockers, payload, project_id):
    """Render RFIs in the original raw table view."""
    # Build HTML table
    rows_html = ""
    for nr in norm_rfis:
        has_ev = bool(nr["pages"]) or nr.get("confidence", 0) > 0
        pri = nr["priority"]
        pri_cls = "sev-critical" if pri in ("critical", "high") else "sev-medium" if pri == "medium" else "sev-low"
        label = nr["id"] if has_ev else f'{nr["id"]} [FYI]'
        conf_pct = int(nr.get("confidence", 0) * 100) if nr.get("confidence") else ""
        conf_str = f"{conf_pct}%" if conf_pct else "\u2014"
        # Evidence snippet column
        snippets = nr["evidence"].get("snippets", []) if nr["evidence"] else []
        snippet_str = _safe_str(snippets[0])[:50] if snippets else "\u2014"
        # Suggested assumption column
        assumption_str = nr["suggested_resolution"][:50] if nr["suggested_resolution"] else "\u2014"
        # Coverage status column
        cov = nr.get("coverage_status", "")
        if cov == "not_found_after_search":
            cov_label = '<span style="color:#4ade80;">&#10003; Searched</span>'
        elif cov == "unknown_not_processed":
            cov_label = '<span style="color:#fbbf24;">&#9888; Not Checked</span>'
        else:
            cov_label = "\u2014"

        rows_html += f"""<tr>
            <td><span class="sev-pill {pri_cls}">{pri.upper()}</span></td>
            <td style="font-weight:500;">{nr['trade'].title()}</td>
            <td>{label}: {nr['question'][:80]}</td>
            <td style="color:#71717a;font-size:0.78rem;">{nr['pages'] or chr(8212)}</td>
            <td style="font-size:0.82rem;color:#71717a;">{snippet_str}</td>
            <td style="font-size:0.82rem;">{assumption_str}</td>
            <td>{cov_label}</td>
            <td>{conf_str}</td>
        </tr>"""

    st.markdown(f"""
    <table class="result-table">
        <thead><tr>
            <th>Priority</th><th>Trade</th><th>RFI Question</th>
            <th>Pages</th><th>Evidence</th><th>Assumption</th><th>Coverage</th><th>Confidence</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Expandable evidence per RFI (with PDF page preview)
    for idx, nr in enumerate(norm_rfis):
        raw_r = rfis[idx] if idx < len(rfis) else {}
        has_ev = has_rfi_evidence(raw_r)
        if has_ev:
            ev = dict(nr["evidence"]) if nr["evidence"] else {}
            ev["coverage_status"] = nr.get("coverage_status", "")
            render_evidence_expander(
                ev,
                title=f"Evidence: {nr['question'][:60]}",
                item_id=nr["id"],
            )
            # Bbox selector (Sprint 5)
            if ev.get("bbox"):
                render_bbox_selector(ev, nr["id"], blockers, rfis)
            # Skipped-page linkage for UNKNOWN items
            if nr.get("coverage_status") == "unknown_not_processed":
                _skipped_all = payload.get("run_coverage", {}).get("pages_skipped", [])
                _rel_types = _item_doc_types(nr)
                _rel_skipped = [s for s in _skipped_all if s.get("doc_type") in _rel_types]
                if _rel_skipped:
                    _skip_pages = [str(s.get("page_idx", 0) + 1) for s in _rel_skipped[:5]]
                    st.caption(
                        f"⚠️ Related pages were skipped: {', '.join(_skip_pages)} "
                        f"— re-run with higher budget to verify"
                    )
            if nr["suggested_resolution"]:
                st.caption(f"\u2705 Suggested: {nr['suggested_resolution'][:100]}")

    # Export CSV (columns match table)
    st.markdown("---")
    import csv as csv_mod
    csv_io = io.StringIO()
    writer = csv_mod.DictWriter(csv_io, fieldnames=[
        "ID", "Trade", "Priority", "Question", "Pages",
        "Evidence Snippet", "Assumption", "Impact",
        "Suggested Resolution", "Confidence"
    ])
    writer.writeheader()
    for nr in norm_rfis:
        snippets = nr["evidence"].get("snippets", []) if nr["evidence"] else []
        writer.writerow({
            "ID": nr["id"],
            "Trade": nr["trade"].title(),
            "Priority": nr["priority"].upper(),
            "Question": nr["question"],
            "Pages": nr["pages"],
            "Evidence Snippet": _safe_str(snippets[0])[:100] if snippets else "",
            "Assumption": nr["suggested_resolution"],
            "Impact": nr["why_it_matters"],
            "Suggested Resolution": nr["suggested_resolution"],
            "Confidence": f"{int(nr.get('confidence', 0)*100)}%" if nr.get("confidence") else "",
        })
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "\U0001f4ca Download RFI CSV", csv_io.getvalue(),
            f"rfis_{project_id}.csv", "text/csv",
            use_container_width=True
        )


def render_coverage_dashboard(payload: dict):
    """Render the '📊 Coverage Dashboard' with color-coded doc_type/discipline heatmap."""
    from collections import Counter

    diagnostics = payload.get("diagnostics", {})
    page_index = diagnostics.get("page_index")
    selected_pages = diagnostics.get("selected_pages")
    run_coverage = payload.get("run_coverage", {})

    if not page_index and not selected_pages and not run_coverage:
        st.info("Coverage data not available. Run a new analysis to see page classification, discipline breakdown, and processing details.")
        return

    # ── Coverage Mode Banner (Sprint 20D: enhanced with processing_stats) ──
    _cov_ps = payload.get("processing_stats") or {}
    if run_coverage and isinstance(run_coverage, dict):
        sel_mode = run_coverage.get("selection_mode", "")
        total = _cov_ps.get("total_pages") or run_coverage.get("pages_total", 0)
        processed = _cov_ps.get("deep_processed_pages") or run_coverage.get("pages_deep_processed", 0)
        not_covered_list = run_coverage.get("doc_types_not_covered", [])

        # Sprint 20G: Show run_mode label in coverage banner
        _cov_run_mode = _cov_ps.get("run_mode", "")
        _mode_labels = {
            "demo_fast": "DEMO FAST",
            "standard_review": "STANDARD REVIEW",
            "full_audit": "FULL AUDIT",
        }
        if sel_mode == "full_read":
            _mode_tag = _mode_labels.get(_cov_run_mode, "FULL READ")
            st.success(f"**{_mode_tag}** \u2014 All {processed} pages deep-processed. Coverage is complete.")
        elif sel_mode == "fast_budget":
            _mode_tag = _mode_labels.get(_cov_run_mode, "FAST BUDGET")
            skipped = run_coverage.get("pages_skipped", [])
            _cov_ocr = _cov_ps.get("ocr_pages")
            _cov_toxic = 0
            _tp = payload.get("toxic_pages")
            if _tp and isinstance(_tp, dict):
                _cov_toxic = len([p for p in _tp.get("pages", []) if p.get("toxic")])
            _cov_detail = f"**{processed}/{total}** pages deep-processed"
            if _cov_ocr is not None:
                _cov_detail += f" \u00b7 **{_cov_ocr}** OCR\u2019d"
            _cov_detail += f" \u00b7 **{len(skipped)}** skipped"
            if _cov_toxic:
                _cov_detail += f" \u00b7 **{_cov_toxic}** toxic"
            _cov_detail += f" \u00b7 **{len(not_covered_list)}** doc type(s) not covered"
            st.warning(
                f"**{_mode_tag}** \u2014 {_cov_detail}. "
                f"Some pages were indexed but not deep-processed."
            )

        # Skipped pages summary
        skipped_pages = run_coverage.get("pages_skipped", [])
        if skipped_pages:
            skip_reasons = Counter(s.get("reason", "unknown") for s in skipped_pages)
            skip_types = Counter(s.get("doc_type", "unknown") for s in skipped_pages)
            reason_str = ", ".join(f"{r}: {c}" for r, c in skip_reasons.items())
            type_str = ", ".join(f"{t}: {c}" for t, c in skip_types.most_common(5))
            st.caption(f"Skip reasons: {reason_str} | By type: {type_str}")

    # ── Sprint 10: Cache Stats ───────────────────────────────────────────
    cache_stats = payload.get("cache_stats")
    if cache_stats and isinstance(cache_stats, dict):
        hits = cache_stats.get("hits", [])
        misses = cache_stats.get("misses", [])
        time_saved = cache_stats.get("time_saved_s", 0)
        hit_rate = cache_stats.get("hit_rate", 0)
        cache_kb = cache_stats.get("cache_bytes", 0) / 1024
        st.caption(
            f"⚡ Cache: {len(hits)} hit(s), {len(misses)} miss(es) "
            f"({hit_rate:.0%} hit rate) | {time_saved:.1f}s saved | {cache_kb:.0f} KB on disk"
        )

    # ── Sprint 10: Toxic Pages ───────────────────────────────────────────
    toxic_data = payload.get("toxic_pages")
    if toxic_data and isinstance(toxic_data, dict) and toxic_data.get("toxic_count", 0) > 0:
        toxic_count = toxic_data["toxic_count"]
        recovered = toxic_data.get("recovered_count", 0)
        retry_time = toxic_data.get("total_retry_time_s", 0)
        st.warning(
            f"⚠️ **{toxic_count} toxic page(s)** detected "
            f"({recovered} recovered via low-DPI retry, {retry_time:.1f}s total retry time)"
        )
        toxic_pages_list = toxic_data.get("pages", [])
        if toxic_pages_list:
            with st.expander(f"View {len(toxic_pages_list)} toxic page(s)"):
                import pandas as pd
                toxic_df = pd.DataFrame([
                    {
                        "Page": tp.get("page_idx", 0) + 1,
                        "Status": "Recovered" if not tp.get("toxic", True) else "Toxic (skipped)",
                        "Reason": tp.get("reason", "unknown"),
                        "Retry DPI": tp.get("retry_dpi", "—"),
                        "Retry Time (s)": f"{tp.get('retry_time_s', 0):.1f}",
                    }
                    for tp in toxic_pages_list
                ])
                st.dataframe(toxic_df, use_container_width=True, hide_index=True)

    # ── Sprint 11: Run Compare ────────────────────────────────────────
    _compare_data = payload.get("_run_compare")
    if _compare_data and isinstance(_compare_data, dict):
        with st.expander("🔄 Compare to Last Run"):
            _cmp_cols = st.columns(3)
            qa_delta = _compare_data.get("qa_score_delta")
            if qa_delta is not None:
                _cmp_cols[0].metric("QA Score Change", delta=qa_delta)
            else:
                _cmp_cols[0].metric("QA Score Change", value="N/A")

            boq_delta = _compare_data.get("boq_delta_summary", {})
            _cmp_cols[1].metric(
                "Quantity Rows",
                value=boq_delta.get("curr_count", 0),
                delta=boq_delta.get("delta", 0),
            )

            new_rfis = _compare_data.get("new_rfis", [])
            _cmp_cols[2].metric(
                "New RFIs",
                value=len(new_rfis),
            )

            if new_rfis:
                st.markdown("**New RFIs since last run:**")
                for rfi_q in new_rfis[:10]:
                    st.markdown(f"- {rfi_q}")
                if len(new_rfis) > 10:
                    st.caption(f"...and {len(new_rfis) - 10} more")

            new_conflicts = _compare_data.get("new_conflicts", [])
            if new_conflicts:
                st.markdown("**New Conflicts since last run:**")
                for conf_d in new_conflicts[:10]:
                    st.markdown(f"- {conf_d}")
                if len(new_conflicts) > 10:
                    st.caption(f"...and {len(new_conflicts) - 10} more")

            # Sprint 13: Approval deltas
            _appr_cols = st.columns(3)
            _newly_approved = _compare_data.get("newly_approved_rfis", [])
            _appr_cols[0].metric("Newly Approved RFIs", value=len(_newly_approved))
            _qty_acc_delta = _compare_data.get("qty_accepted_delta", 0)
            _appr_cols[1].metric("Qty Accepted Δ", value=_qty_acc_delta)
            _conf_rev_delta = _compare_data.get("conflicts_reviewed_delta", 0)
            _appr_cols[2].metric("Conflicts Reviewed Δ", value=_conf_rev_delta)

    # ── Sprint 9: Per-Document Coverage Breakdown ──────────────────────
    mdi_data = payload.get("multi_doc_index")
    if mdi_data and len(mdi_data.get("docs", [])) > 1:
        st.markdown("#### Per-Document Coverage")
        pi_pages_list = diagnostics.get("page_index", {}).get("pages", []) if isinstance(diagnostics.get("page_index"), dict) else []
        for doc_info in mdi_data["docs"]:
            d_start = doc_info.get("global_page_start", 0)
            d_end = d_start + doc_info.get("page_count", 0)
            doc_pg_list = [
                p for p in pi_pages_list
                if isinstance(p, dict) and d_start <= p.get("page_idx", -1) < d_end
            ]
            doc_types = Counter(p.get("doc_type", "unknown") for p in doc_pg_list)
            type_str = ", ".join(f"{k}: {v}" for k, v in doc_types.most_common(5))
            st.markdown(f"**{doc_info.get('filename', '?')}** ({doc_info.get('page_count', 0)} pages)")
            st.caption(type_str if type_str else "No classified pages")

    # ── Page Coverage Heatmap (Sprint 4a) ──────────────────────────────
    if run_coverage and isinstance(run_coverage, dict):
        total_pages_hm = run_coverage.get("pages_total", 0)
        if 0 < total_pages_hm <= 500:
            st.markdown("#### 📊 Page Coverage Map")

            # Build page status map
            skipped_set = {s.get("page_idx") for s in run_coverage.get("pages_skipped", [])}
            processed_count = run_coverage.get("pages_deep_processed", 0)

            # Get page_index for doc_type info per page
            page_index_data = diagnostics.get("page_index", {})
            pi_pages = page_index_data.get("pages", []) if isinstance(page_index_data, dict) else []
            page_types = {}
            for p in pi_pages:
                if isinstance(p, dict):
                    page_types[p.get("page_idx", -1)] = p.get("doc_type", "unknown")

            cells_html = ""
            for pg in range(total_pages_hm):
                if pg in skipped_set:
                    color = "#f87171"  # red — skipped
                    tooltip = f"Page {pg+1}: SKIPPED"
                elif pg < processed_count or pg not in skipped_set:
                    color = "#4ade80"  # green — processed
                    dt = page_types.get(pg, "")
                    tooltip = f"Page {pg+1}: {dt}" if dt else f"Page {pg+1}: processed"
                else:
                    color = "#6b7280"  # grey — unknown
                    tooltip = f"Page {pg+1}"
                cells_html += (
                    f'<div class="heatmap-cell" style="background:{color};" '
                    f'title="{tooltip}">{pg+1}</div>'
                )

            st.markdown(
                f'<div class="heatmap-grid">{cells_html}</div>',
                unsafe_allow_html=True,
            )

            # Compact legend
            st.markdown(
                '<div style="display:flex;gap:1rem;font-size:0.75rem;margin:0.25rem 0;">'
                '<span><span style="color:#4ade80;">■</span> Processed</span>'
                '<span><span style="color:#f87171;">■</span> Skipped</span>'
                '<span><span style="color:#6b7280;">■</span> Unknown</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Doc Type Coverage Table (color-coded status) ──────────────────
    if run_coverage and isinstance(run_coverage, dict):
        doc_types = run_coverage.get("doc_types_detected", {})
        fully_covered = set(run_coverage.get("doc_types_fully_covered", []))
        partially_covered = set(run_coverage.get("doc_types_partially_covered", []))
        not_covered = set(run_coverage.get("doc_types_not_covered", []))
        coverage_summary = selected_pages.get("coverage_summary", {}) if selected_pages else {}

        if doc_types:
            st.markdown("#### Coverage by Doc Type")
            rows_html = ""
            for dtype in sorted(doc_types.keys(), key=lambda d: -doc_types[d]):
                count = doc_types[d] if (d := dtype) else 0  # noqa
                count = doc_types.get(dtype, 0)
                selected = coverage_summary.get(dtype, 0)
                if dtype in fully_covered:
                    status_label = "FOUND"
                    css_class = "cov-found"
                    bg = "rgba(34,197,94,0.06)"
                elif dtype in partially_covered:
                    status_label = "PARTIAL"
                    css_class = "cov-partial"
                    bg = "rgba(245,158,11,0.06)"
                elif dtype in not_covered:
                    status_label = "NOT PROCESSED"
                    css_class = "cov-not-processed"
                    bg = "rgba(239,68,68,0.06)"
                else:
                    status_label = "\u2014"
                    css_class = ""
                    bg = "transparent"
                always_types = {"schedule", "boq", "notes", "legend", "conditions", "addendum"}
                tier_badge = ' <span class="boq-pill">always</span>' if dtype in always_types else ""
                rows_html += f'''<tr style="background:{bg};">
                    <td style="font-weight:500;">{dtype.replace("_", " ").title()}{tier_badge}</td>
                    <td style="text-align:right;">{count}</td>
                    <td style="text-align:right;">{selected}</td>
                    <td class="{css_class}">{status_label}</td>
                </tr>'''

            st.markdown(f'''
            <table class="result-table">
                <thead><tr>
                    <th>Doc Type</th><th>Pages</th><th>Processed</th><th>Coverage Status</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            ''', unsafe_allow_html=True)

        # ── Actionable Recommendations ────────────────────────────
        partial_list = run_coverage.get("doc_types_partially_covered", [])
        not_covered_list_action = run_coverage.get("doc_types_not_covered", [])
        sel_mode = run_coverage.get("selection_mode", "full_read")

        if partial_list or not_covered_list_action:
            st.markdown("#### 💡 Recommended Actions")

            for dt in not_covered_list_action:
                page_count = doc_types.get(dt, 0)
                st.markdown(
                    f'<div class="ev-snippet">'
                    f'<strong>{dt.replace("_", " ").title()}</strong> — '
                    f'{page_count} pages detected but none processed<br>'
                    f'<em>Action:</em> Re-run with higher OCR budget or force FULL_READ mode'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            for dt in partial_list:
                page_count = doc_types.get(dt, 0)
                st.markdown(
                    f'<div class="ev-snippet">'
                    f'<strong>{dt.replace("_", " ").title()}</strong> — '
                    f'only some of {page_count} pages processed<br>'
                    f'<em>Action:</em> Increase OCR budget to cover remaining {dt} pages'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if sel_mode == "fast_budget":
                budget = run_coverage.get("ocr_budget_pages", 80)
                st.info(
                    "💡 **To increase coverage:**\n"
                    "- Set `OCR_BUDGET_PAGES=150` (or higher) in your environment\n"
                    "- Or pass `--full-read` flag to force processing all pages\n"
                    f"- Current budget: {budget} pages"
                )

            # Recommended pages to process
            skipped_recs = run_coverage.get("pages_skipped", [])
            if skipped_recs:
                target_types = set(partial_list + not_covered_list_action)
                recommended = [s for s in skipped_recs if s.get("doc_type") in target_types]
                if recommended:
                    rec_pages = sorted(set(r.get("page_idx", 0) + 1 for r in recommended))[:20]
                    st.markdown(
                        f"**Recommended pages to process:** "
                        f"{', '.join(str(p) for p in rec_pages)}"
                    )

    # ── Discipline Breakdown (with progress bars) ─────────────────────
    disciplines = run_coverage.get("disciplines_detected", {}) if run_coverage else {}
    if not disciplines and page_index:
        disciplines = page_index.get("counts_by_discipline", {})

    if disciplines:
        st.markdown("#### Discipline Breakdown")
        # Derive processed count per discipline from skipped_pages
        skipped_pages = run_coverage.get("pages_skipped", []) if run_coverage else []
        skipped_by_disc = Counter(s.get("discipline", "unknown") for s in skipped_pages)

        rows_html = ""
        for disc in sorted(disciplines.keys(), key=lambda d: -disciplines[d]):
            total_d = disciplines.get(disc, 0)
            skipped_d = skipped_by_disc.get(disc, 0)
            processed_d = total_d - skipped_d
            pct = int(processed_d / total_d * 100) if total_d > 0 else 0
            if pct == 100:
                bar_color = "#4ade80"
            elif pct >= 50:
                bar_color = "#fbbf24"
            else:
                bar_color = "#f87171"
            bar_html = (
                f'<div class="mini-bar">'
                f'<div class="mini-bar-fill" style="background:{bar_color};width:{pct}%;"></div>'
                f'</div>'
            )
            rows_html += f'''<tr>
                <td style="font-weight:500;">{disc.replace("_", " ").title()}</td>
                <td style="text-align:right;">{processed_d}/{total_d}</td>
                <td style="text-align:right;">{pct}%</td>
                <td style="width:120px;">{bar_html}</td>
            </tr>'''

        st.markdown(f'''
        <table class="result-table">
            <thead><tr>
                <th>Discipline</th><th>Processed / Total</th><th>Coverage</th><th></th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        ''', unsafe_allow_html=True)

    # ── Status Legend ──────────────────────────────────────────────────
    st.markdown("#### Status Legend")
    st.markdown('''
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin:0.5rem 0 1rem 0;font-size:0.85rem;">
        <span><span class="cov-found">&#9679;</span> FOUND — All pages deep-processed</span>
        <span><span class="cov-partial">&#9679;</span> PARTIAL — Some pages processed (budget limit)</span>
        <span><span class="cov-not-processed">&#9679;</span> NOT PROCESSED — Zero pages processed</span>
    </div>
    ''', unsafe_allow_html=True)

    # ── Skipped Pages Report ─────────────────────────────────────────
    skipped_report = run_coverage.get("pages_skipped", []) if run_coverage else []
    if skipped_report:
        import pandas as pd
        st.markdown("#### 📋 Skipped Pages")
        skip_df = pd.DataFrame([
            {
                "Page": s.get("page_idx", 0) + 1,
                "Doc Type": s.get("doc_type", "unknown").replace("_", " ").title(),
                "Discipline": s.get("discipline", "unknown").replace("_", " ").title(),
                "Reason": s.get("reason", "budget_limit").replace("_", " ").title(),
            }
            for s in skipped_report
        ])
        st.dataframe(skip_df, use_container_width=True, hide_index=True)

        csv_data = skip_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download skipped_pages.csv",
            csv_data, "skipped_pages.csv", "text/csv",
            key="dl_skipped_pages",
        )

    # ── OCR Budget Bar ────────────────────────────────────────────────
    if selected_pages:
        budget_total = selected_pages.get("budget_total", 80)
        budget_used = selected_pages.get("budget_used", 0)
        pct = int(budget_used / budget_total * 100) if budget_total > 0 else 0
        st.markdown("#### OCR Budget")
        st.progress(min(pct / 100, 1.0))
        st.caption(f"OCR Budget: **{budget_used}** / {budget_total} pages analyzed ({pct}%)")

        # Selection tier breakdown
        st.markdown("#### Selection Tiers")
        always_count = selected_pages.get("always_include_count", 0)
        sample_count = selected_pages.get("sample_count", 0)
        skipped_types = selected_pages.get("skipped_types", {})
        skipped_count = sum(skipped_types.values()) if isinstance(skipped_types, dict) else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Must-Read", always_count, help="Schedule, BOQ, notes, conditions — always included")
        col2.metric("Sampled", sample_count, help="Plans, details, specs — round-robin across disciplines")
        col3.metric("Skipped", skipped_count, help="Pages beyond OCR budget")

        if skipped_types and isinstance(skipped_types, dict):
            skip_items = ", ".join(f"{k}: {v}" for k, v in skipped_types.items() if v > 0)
            if skip_items:
                st.caption(f"Skipped by type: {skip_items}")

    # ── Guardrail Warnings ────────────────────────────────────────────
    warnings = payload.get("guardrail_warnings", [])
    if warnings:
        st.markdown("#### Guardrail Warnings")
        for w in warnings:
            msg = w.get("message", "Unknown warning") if isinstance(w, dict) else str(w)
            wtype = w.get("type", "warning") if isinstance(w, dict) else "warning"
            st.warning(f"⚠️ **{wtype.replace('_', ' ').title()}**: {msg}")


# =============================================================================
# SPRINT 20F: EXTRACTION DIAGNOSTICS PANEL
# =============================================================================

def render_extraction_diagnostics(payload: dict, expanded: bool = False):
    """Render 'Extraction Diagnostics' panel showing why extraction succeeded/failed.

    Sprint 20F: Compact panel showing BOQ/schedule extraction attempts,
    table methods used, and top rejection reasons.
    Visible in dev mode by default, under expander in normal mode.
    """
    ext_diag = payload.get("extraction_diagnostics") or {}
    ps = payload.get("processing_stats") or {}

    if not ext_diag and not ps:
        return

    boq_diag = ext_diag.get("boq", {})
    sched_diag = ext_diag.get("schedules", {})
    table_methods = ext_diag.get("table_methods_used", {})

    boq_attempted = boq_diag.get("pages_attempted", 0)
    boq_parsed = boq_diag.get("pages_parsed", 0)
    boq_items = boq_diag.get("items_extracted", 0)
    sched_attempted = sched_diag.get("pages_attempted", 0)
    sched_parsed = sched_diag.get("pages_parsed", 0)
    sched_rows = sched_diag.get("rows_extracted", 0)

    table_attempt = ps.get("table_attempt_pages", 0)
    table_success = ps.get("table_success_pages", 0)

    with st.expander("Extraction Diagnostics", expanded=expanded):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**BOQ Extraction**")
            st.caption(
                f"Pages attempted: **{boq_attempted}** | "
                f"Pages parsed: **{boq_parsed}** | "
                f"Items extracted: **{boq_items}**"
            )
            if boq_attempted > 0 and boq_parsed == 0:
                st.warning("No BOQ pages produced items. Check OCR quality or table structure.")
        with c2:
            st.markdown("**Schedule Extraction**")
            st.caption(
                f"Pages attempted: **{sched_attempted}** | "
                f"Pages parsed: **{sched_parsed}** | "
                f"Rows extracted: **{sched_rows}**"
            )
            if sched_attempted > 0 and sched_parsed == 0:
                st.warning("No schedule pages produced rows. Headers may not be detected.")

        if table_methods:
            st.markdown("**Table Methods Used**")
            method_parts = [f"{m}: {c}" for m, c in sorted(table_methods.items(), key=lambda x: -x[1])]
            st.caption(" | ".join(method_parts))

        st.caption(
            f"Table extraction: **{table_success}/{table_attempt}** pages succeeded"
        )

        # Guardrail warnings related to extraction
        guardrails = payload.get("guardrail_warnings", [])
        extraction_guardrails = [
            g for g in guardrails
            if isinstance(g, dict) and g.get("type", "").endswith("_extraction_gap")
        ]
        if extraction_guardrails:
            st.markdown("**Extraction Gaps Detected**")
            for g in extraction_guardrails:
                st.warning(g.get("message", ""))


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
    if decision in ["NO-GO", "NO_DRAWINGS"]:
        badge_class = "badge-nogo"
    elif decision in ["REVIEW", "CONDITIONAL"]:
        badge_class = "badge-review"
    else:
        badge_class = "badge-go"
    display_decision = "NO DRAWINGS" if decision == "NO_DRAWINGS" else decision
    st.markdown(f'<div class="decision-badge {badge_class}">{display_decision}</div>', unsafe_allow_html=True)
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
    deps = report.get("missing_dependencies") or []
    if not deps:
        st.markdown("## Missing Dependencies (0)")
        st.info("No missing dependencies detected.")
        return

    st.markdown(f"## Missing Dependencies ({len(deps)})")
    st.caption("These are blocking accurate pricing. Each shows what's detected, what's missing, and how to fix.")

    for _dep_i, dep in enumerate(deps):
        if not isinstance(dep, dict):
            continue
        # Stable ID: prefer dep["id"], fall back to hash, then index
        _dep_id = dep.get("id") or ""
        if not _dep_id:
            try:
                import hashlib, json as _json
                _dep_id = "DEP-" + hashlib.sha256(
                    _json.dumps(dep, sort_keys=True, default=str).encode()
                ).hexdigest()[:8]
            except Exception:
                _dep_id = f"DEP-{_dep_i}"

        with st.container():
            st.markdown(f"### {_dep_id}: {dep.get('dependency_type', 'Unknown')}")

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Status:** {dep.get('status', 'missing').upper()}")
            col2.markdown(f"**Trade:** {dep.get('impact_trade', 'general').title()}")
            col3.markdown(f"**Bid Impact:** {dep.get('impact_bid', '').replace('_', ' ').title()}")

            st.markdown(dep.get("why_needed", ""))

            # Evidence
            evidence = dep.get("evidence", {})
            if evidence:
                ev = dict(evidence)
                ev["coverage_status"] = dep.get("coverage_status", "")
                render_evidence_expander(
                    ev, "Evidence Details",
                    item_id=_dep_id,
                )

            # Risk & Score
            cost_risk = dep.get("cost_risk", "medium")
            schedule_risk = dep.get("schedule_risk", "low")
            score_delta = dep.get("score_delta", 0)
            st.markdown(f"**Risk:** Cost: {cost_risk.upper()} | Schedule: {schedule_risk.upper()} | **Score if fixed: +{score_delta} pts**")

            # Fix options
            fixes = dep.get("fix_options", [])
            if fixes:
                with st.expander("How to Fix"):
                    for _fix_i, fix in enumerate(fixes, 1):
                        st.markdown(f"{_fix_i}. {fix}")

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
                rfi_text = rfi.get("question") or rfi.get("title", "Unknown")
                st.markdown(f"**{priority_icon} {rfi.get('id', 'RFI')}:** {rfi_text[:100]}")
                why = rfi.get("why_it_matters") or rfi.get("missing_info", "")
                st.caption(why[:150])

                evidence = rfi.get("evidence", {})
                if evidence and (evidence.get("pages") or evidence.get("detected_entities")):
                    ev = dict(evidence)
                    ev["coverage_status"] = rfi.get("coverage_status", "")
                    render_evidence_expander(
                        ev, "Evidence",
                        item_id=rfi.get("id", f"RFI-{i}"),
                    )

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

    # Row 1: Page counts
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", overview.get("total_pages", 0))
    col2.metric("Total Sheets", overview.get("total_sheets") or overview.get("total_pages", 0))
    col3.metric("Pages with Scale", overview.get("pages_with_scale", 0))
    col4.metric("Pages without Scale", overview.get("pages_without_scale", 0))

    # Row 2: Entity counts
    detected_ents = overview.get("detected_entities", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Doors", detected_ents.get("doors") or overview.get("door_tags_count", 0))
    col2.metric("Windows", detected_ents.get("windows") or overview.get("window_tags_count", 0))
    col3.metric("Rooms", detected_ents.get("rooms") or overview.get("rooms_detected", 0))
    col4.metric("Structural", detected_ents.get("structural_elements", 0))

    st.markdown("---")

    # Disciplines found and missing
    disciplines = overview.get("disciplines_detected", [])
    missing_disc = overview.get("missing_disciplines", [])

    col1, col2 = st.columns(2)
    with col1:
        if disciplines:
            st.markdown(f"**✅ Disciplines Found:** {', '.join(disciplines)}")
        else:
            st.markdown("**Disciplines Found:** None detected")
    with col2:
        if missing_disc:
            st.markdown(f"**⚠️ Missing Disciplines:** {', '.join(missing_disc)}")

    # Scale status
    scale_status = overview.get("scale_status", "unknown")
    if scale_status == "partial":
        st.warning(f"⚠️ Scale Status: Partial - {overview.get('pages_without_scale', 0)} pages without scale detected")
    elif scale_status == "none":
        st.error("❌ Scale Status: No scale detected on any page")
    elif scale_status == "all_scaled":
        st.success("✅ Scale Status: All pages have scale")

    # Revision and files
    revision_info = overview.get("revision_info", "")
    files = overview.get("files_analyzed", [])
    if revision_info:
        st.markdown(f"**Revision:** {revision_info}")
    if files:
        st.markdown(f"**Files Analyzed:** {', '.join(files)}")

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
    """Build report dict from loaded results.

    Supports both new analysis.json format and legacy deep_analysis.json format.
    """
    # Use analysis.json (new format) if available, fall back to deep_analysis (legacy)
    deep = results.get("analysis") or results.get("deep_analysis") or {}
    plan_graph = results.get("plan_graph") or {}
    bid_gate = results.get("bid_gate") or {}

    # Build executive summary
    # Handle both int and dict formats for readiness_score
    raw_score = deep.get("readiness_score", {})
    if isinstance(raw_score, (int, float)):
        _deep_sub = deep.get("sub_scores", {})
        score_data = {
            "total_score": raw_score,
            "status": deep.get("decision", "NO-GO"),
            "coverage_score": _deep_sub.get("coverage", 0),
            "measurement_score": _deep_sub.get("measurement", 0),
            "completeness_score": _deep_sub.get("completeness", 0),
            "blocker_score": _deep_sub.get("blocker", 0),
        }
    else:
        score_data = raw_score if raw_score else {}
    blockers = deep.get("blockers", [])

    top_reasons = []
    for b in blockers[:3]:
        if b.get("severity") in ["critical", "high"]:
            # Handle both 'title' and 'description' field names
            reason = b.get("title") or b.get("description", "Unknown issue")
            top_reasons.append(reason)

    # Use decision_reasons from deep analysis if available
    if not top_reasons and deep.get("decision_reasons"):
        top_reasons = deep.get("decision_reasons", [])[:3]

    top_fixes = []
    # Use top_fixes from deep analysis if available
    if deep.get("top_fixes"):
        for fix in deep.get("top_fixes", [])[:3]:
            top_fixes.append({
                "action": fix.get("fix", "Resolve issue"),
                "score_delta": fix.get("score_delta", 5),
                "description": fix.get("fix", ""),
            })
    else:
        for b in sorted(blockers, key=lambda x: x.get("fix_score_delta") or x.get("score_delta_estimate", 0), reverse=True)[:3]:
            fix_opts = b.get("fix_options", [])
            top_fixes.append({
                "action": fix_opts[0] if fix_opts else "Resolve issue",
                "score_delta": b.get("fix_score_delta") or b.get("score_delta_estimate", 5),
                "description": b.get("title") or b.get("description", ""),
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
        "top_5_by_impact": [(r.get("question") or r.get("title", ""))[:80] for r in rfis[:5]],
    }

    # Trade coverage
    trade_coverage = []
    for tc in deep.get("trade_coverage", []):
        cov_pct = tc.get("coverage_pct") or tc.get("scope_coverage_pct", 0)
        trade_coverage.append({
            "trade": tc.get("trade", "general"),
            "coverage_pct": cov_pct,
            "priceable_count": tc.get("priceable_count") or tc.get("priceable_items", 0),
            "blocked_count": tc.get("blocked_count") or tc.get("blocked_items", 0),
            "top_missing": tc.get("missing_dependencies") or tc.get("gaps", [])[:3],
            "confidence": tc.get("confidence") or ("high" if cov_pct >= 70 else ("medium" if cov_pct >= 40 else "low")),
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

    # Drawing set overview - try deep.drawing_set_overview first, fallback to plan_graph
    ds_overview = deep.get("drawing_set_overview", {})
    drawing_set_overview = {
        "total_pages": ds_overview.get("total_pages") or plan_graph.get("total_pages", 0),
        "total_sheets": ds_overview.get("total_sheets", 0),
        "disciplines_detected": ds_overview.get("disciplines") or ds_overview.get("disciplines_detected") or plan_graph.get("disciplines_found", []),
        "missing_disciplines": ds_overview.get("missing_disciplines", []),
        "sheet_types": plan_graph.get("sheet_types_found", {}),
        "schedules_detected": [],
        "scale_status": ds_overview.get("scale_status", "unknown"),
        "pages_with_scale": ds_overview.get("pages_with_scale") or plan_graph.get("pages_with_scale", 0),
        "pages_without_scale": ds_overview.get("pages_without_scale") or plan_graph.get("pages_without_scale", 0),
        "detected_entities": ds_overview.get("detected_entities", {}),
        "rooms_detected": ds_overview.get("detected_entities", {}).get("rooms") or len(plan_graph.get("all_room_names", [])),
        "room_names": plan_graph.get("all_room_names", [])[:20],
        "door_tags_count": ds_overview.get("detected_entities", {}).get("doors") or len(plan_graph.get("all_door_tags", [])),
        "window_tags_count": ds_overview.get("detected_entities", {}).get("windows") or len(plan_graph.get("all_window_tags", [])),
        "door_tags_sample": plan_graph.get("all_door_tags", [])[:10],
        "window_tags_sample": plan_graph.get("all_window_tags", [])[:10],
        "revision_info": ds_overview.get("revision_info", ""),
        "files_analyzed": ds_overview.get("files_analyzed", []),
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
# ANALYSIS RESULTS PREVIEW (rendered immediately after analysis)
# =============================================================================

def _is_yc_demo() -> bool:
    """Check if YC Demo Mode toggle is active (Sprint 20B)."""
    return bool(st.session_state.get("_xboq_yc_demo", False))


# ── Sprint 20B: Narration hints per screen (only in YC Demo Mode) ─────
_NARRATION_HINTS = {
    "summary": [
        "Open with: 'We uploaded a real tender PDF — {pages} pages.'",
        "Point out the readiness score and decision badge.",
        "Highlight how many pages xBOQ processed vs. skipped.",
        "Mention total RFIs and blockers found automatically.",
        "Click into a blocker to show evidence drill-down.",
    ],
    "structural": [
        "Say: 'xBOQ also extracted structural quantities automatically.'",
        "Point out concrete volume and steel tonnage.",
        "Highlight QC confidence — shows what's assumed vs. detected.",
        "Open the assumptions expander to show transparency.",
    ],
    "export": [
        "Say: 'Everything is exportable in one click.'",
        "Show the Submission Pack structure (5 folders).",
        "Mention Demo Snapshot for a clean, compact bundle.",
        "Highlight approved RFI count and quantities included.",
    ],
    "coverage": [
        "Say: 'We automatically classify every page by discipline and type.'",
        "Point out the disciplines breakdown and coverage %.",
        "Show how pages with missing scales or data are flagged.",
    ],
}


def _render_narration_hints(screen_key: str, payload: dict = None):
    """Render narration hint box if YC Demo Mode is active."""
    if not _is_yc_demo():
        return
    hints = _NARRATION_HINTS.get(screen_key, [])
    if not hints:
        return
    with st.expander("\U0001f3ac Narration hints", expanded=False):
        for h in hints:
            if payload and "{pages}" in h:
                _total = (payload.get("drawing_overview") or {}).get("pages_total", "?")
                h = h.replace("{pages}", str(_total))
            st.caption(f"\u2022 {h}")


def _render_analysis_results_preview(result, project_id: str, uploaded_files: List[Any]):
    """
    Render analysis results immediately after completion.
    Uses the new at-a-glance dashboard for better UX.
    """
    payload = result.payload

    # Sprint 20C: Attach estimating playbook from session state (if available)
    if payload and st.session_state.get("_ep_playbook"):
        payload["estimating_playbook"] = st.session_state["_ep_playbook"]

    # If no payload, show minimal fallback
    if not payload:
        st.warning("Analysis completed but no detailed results available.")
        st.info(f"**Project ID:** `{project_id}`")
        if st.button("View Analysis Results →", type="primary", use_container_width=True):
            st.query_params["project_id"] = project_id
            st.rerun()
        return

    # Store PDF path and OCR cache in session state for Evidence Viewer
    st.session_state["_xboq_pdf_path"] = payload.get("primary_pdf_path")
    st.session_state["_xboq_ocr_cache"] = payload.get("ocr_text_cache", {})

    # ── Sprint 18: YC Summary Card (demo mode only) ───────────────────
    try:
        from src.demo.demo_config import is_demo_mode as _s18_dm
        if _s18_dm():
            from src.demo.summary_card import build_summary_card
            _sc_name = st.session_state.get("_active_project_name", "")
            _sc = build_summary_card(payload, project_name=_sc_name)
            with st.container():
                st.markdown("#### YC Demo Summary")
                _sc_c1, _sc_c2, _sc_c3 = st.columns(3)
                with _sc_c1:
                    st.metric("Total Pages", _sc["total_pages"])
                    st.metric("Deep Processed", _sc["deep_pages"])
                    st.metric("OCR Pages", _sc["ocr_pages"])
                    if _sc.get("skipped_pages"):
                        st.caption(f"Skipped: {_sc['skipped_pages']}")
                    st.caption(f"Cache: {_sc['cache_time_saved']}")
                with _sc_c2:
                    st.metric("QA Score", f"{_sc['qa_score']}/100")
                    st.caption(f"Decision: {_sc['decision']}")
                    for _sc_act in _sc["top_actions"][:2]:
                        st.caption(f"  - {_sc_act}")
                with _sc_c3:
                    st.metric("Approved RFIs", _sc["approved_rfis"])
                    st.metric("Quantities", _sc["accepted_quantities"])
                    st.metric("Assumptions", _sc["accepted_assumptions"])
                    _sc_pack = "Ready" if _sc["submission_pack_ready"] else "Not yet"
                    st.caption(f"Pack: {_sc_pack}")
    except Exception:
        pass

    # =========================================================================
    # NEW: Use the at-a-glance dashboard
    # =========================================================================
    st.markdown("---")

    # Build DemoAnalysis model from payload
    demo = build_demo_analysis(payload, project_id)

    # Render the at-a-glance dashboard
    render_at_a_glance_dashboard(demo)

    # Sprint 20G: "Process remaining pages" resume button
    _ps_resume = payload.get("processing_stats") or {}
    _resume_mode = _ps_resume.get("run_mode", "demo_fast")
    _resume_deep = _ps_resume.get("deep_processed_pages", 0)
    _resume_total = _ps_resume.get("total_pages", 0)
    if _resume_mode != "full_audit" and _resume_deep < _resume_total and uploaded_files:
        _next_modes = {
            "demo_fast": ("standard_review", "Standard Review (220 pages)"),
            "standard_review": ("full_audit", "Full Audit (All pages)"),
        }
        _next_mode, _next_label = _next_modes.get(_resume_mode, ("full_audit", "Full Audit (All pages)"))
        st.markdown(f"""
        <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.25);
                    border-radius:8px;padding:0.8rem 1rem;margin:0.5rem 0;">
            <span style="color:#c4b5fd;font-size:0.85rem;">
                Processed {_resume_deep}/{_resume_total} pages.
                Upgrade to <b>{_next_label}</b> to process more.
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Process remaining pages ({_next_label})", type="secondary"):
            _run_analysis_with_progress(uploaded_files, dev_mode=False, run_mode=_next_mode)

    # ── Sprint 10: QA Score metric ──────────────────────────────────────
    qa_score_data = payload.get("qa_score")
    if qa_score_data and isinstance(qa_score_data, dict):
        qa_val = qa_score_data.get("score", 0)
        qa_conf = qa_score_data.get("confidence", "")
        qa_cols = st.columns([1, 3])
        with qa_cols[0]:
            st.metric("Bid Pack QA", f"{qa_val}/100", help=f"Confidence: {qa_conf}")
        with qa_cols[1]:
            breakdown = qa_score_data.get("breakdown", {})
            if breakdown:
                with st.expander("QA Score Breakdown"):
                    for comp_key, comp_val in breakdown.items():
                        label = comp_key.replace("_", " ").title()
                        st.markdown(f"- **{label}**: {comp_val}/20")
                    top_actions = qa_score_data.get("top_actions", [])
                    if top_actions:
                        st.markdown("**Top Actions to Improve:**")
                        for action in top_actions[:5]:
                            st.markdown(f"  - {action}")

    # ── Sprint 17: Highlights Panel ─────────────────────────────────────
    try:
        from src.analysis.highlights import build_highlights
        _highlights = build_highlights(payload)
        if _highlights:
            st.markdown("---")
            st.markdown("### Highlights")
            _hl_cols = st.columns(min(len(_highlights), 4))
            for _hl_idx, _hl in enumerate(_highlights):
                with _hl_cols[_hl_idx % len(_hl_cols)]:
                    _hl_color = {"good": "green", "warn": "orange", "bad": "red"}.get(
                        _hl.get("severity", "warn"), "gray")
                    st.markdown(
                        f"**{_hl['icon']} {_hl['label']}**\n\n"
                        f":{_hl_color}[{_hl['value']}]"
                    )
                    if _hl.get("detail"):
                        st.caption(_hl["detail"])
    except Exception:
        pass

    # Get data for detailed tabs
    blockers = payload.get("blockers", [])
    rfis = payload.get("rfis", [])
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    timings = payload.get("timings", {})

    # =========================================================================
    # GLOBAL SEARCH (Sprint 5) — above tabs, always visible
    # =========================================================================
    st.markdown("---")
    render_global_search(payload)

    # ── Sprint 18: Jump UX buttons (demo mode only) ──────────────────
    try:
        from src.demo.demo_config import is_demo_mode as _j18_dm, JUMP_TARGETS
        if _j18_dm():
            st.markdown("---")
            _jmp_cols = st.columns(len(JUMP_TARGETS))
            for _ji, _jt in enumerate(JUMP_TARGETS):
                with _jmp_cols[_ji]:
                    st.button(
                        f"{_jt['key']}: {_jt['label']}",
                        key=f"jump_{_jt['key']}",
                        use_container_width=True,
                    )
            st.caption("Quick jump reference for demo flow")
    except Exception:
        pass

    # =========================================================================
    # TABBED RESULTS DETAIL (dashboard above has the at-a-glance view)
    # =========================================================================

    # Sprint 20B: Recorder-friendly 3-step strip (above tabs)
    _yc_active = _is_yc_demo()
    if _yc_active:
        _render_narration_hints("summary", payload)
        _blocker_n = len(payload.get("blockers", []))
        _rfi_n = len(payload.get("rfis", []))
        _has_structural = bool(
            payload.get("structural_takeoff", {}).get("mode") not in (None, "error")
            if payload.get("structural_takeoff") else False
        )
        _step_labels = [
            ("\u2460", "What we found",
             f"{(payload.get('drawing_overview') or {}).get('pages_total', 0)} pages analyzed"),
            ("\u2461", "What blocks the bid",
             f"{_blocker_n} blockers \u00b7 {_rfi_n} RFIs"),
            ("\u2462", "What to send",
             "RFIs / assumptions / exports"),
        ]
        if _has_structural:
            _st_conc = payload.get("structural_takeoff", {}).get("summary", {}).get("concrete_m3", 0)
            _step_labels.append(
                ("\u2463", "Structural quantity draft", f"{_st_conc:.1f} m\u00b3 concrete")
            )
        _step_cols = st.columns(len(_step_labels))
        for _si, (_icon, _label, _detail) in enumerate(_step_labels):
            with _step_cols[_si]:
                st.markdown(f"**{_icon} {_label}**")
                st.caption(_detail)
        st.markdown("---")

    st.markdown("### Detailed Results")

    # Sprint 20B: Cleaner tab labels in YC Demo Mode
    if _yc_active:
        _tab_labels = [
            "\u2611\ufe0f Issues to Resolve",
            "\U0001f6ab Blockers", "\U0001f4dd RFIs",
            "\U0001f4cb Bid Pack", "\U0001f4ca Coverage",
            "\U0001f4c8 Bid Strategy",
            "\u26a0\ufe0f Risk Checklist",
            "\U0001f4d0 Quantities",
            "\U0001f5d3\ufe0f Pre-bid Meeting",
            "\U0001f4ca Quality",
            "\U0001f4c1 Raw JSON",
            "\U0001f4e5 Ground Truth",
            "\U0001f3d7\ufe0f Structural Takeoff",
            "\u2699\ufe0f Estimating Playbook",
        ]
    else:
        _tab_labels = [
            "\u2611\ufe0f Review Queue",
            "\U0001f6ab Blockers", "\U0001f4dd RFIs",
            "\U0001f4cb Bid Pack", "\U0001f4ca Coverage Dashboard",
            "\U0001f4c8 Bid Strategy",
            "\u26a0\ufe0f Risk Checklist",
            "\U0001f4d0 Quantities",
            "\U0001f5d3\ufe0f Pre-bid Meeting",
            "\U0001f4ca Quality Dashboard",
            "\U0001f4c1 Raw JSON",
            "\U0001f4e5 Ground Truth",
            "\U0001f3d7\ufe0f Structural Takeoff",
            "\u2699\ufe0f Estimating Playbook",
        ]

    # Sprint 20B: Stability guard helper — prevents one tab crash from blanking all tabs
    def _safe_tab(tab_name: str, render_fn, *args, **kwargs):
        """Call render_fn inside try/except; show friendly error on failure."""
        try:
            render_fn(*args, **kwargs)
        except Exception as _tab_err:
            if _yc_active:
                st.error(f"This section could not be loaded.")
                with st.expander("Technical details", expanded=False):
                    st.code(f"{type(_tab_err).__name__}: {_tab_err}")
            else:
                st.error(f"{tab_name} encountered an error: {type(_tab_err).__name__}")
                with st.expander("Traceback", expanded=False):
                    import traceback
                    st.code(traceback.format_exc())

    def _handle_tab_error(tab_name: str, err: Exception):
        """Sprint 20E: Show error card for inline tab code (not wrapped in _safe_tab)."""
        if _yc_active:
            st.error("This section could not be loaded.")
            with st.expander("Technical details", expanded=False):
                st.code(f"{type(err).__name__}: {err}")
        else:
            st.error(f"{tab_name} encountered an error: {type(err).__name__}")
            with st.expander("Traceback", expanded=False):
                import traceback
                st.code(traceback.format_exc())

    preview_tabs = st.tabs(_tab_labels)

    # ------------------------------------------------------------------
    # TAB 0: Review Queue (Sprint 13 + Sprint 20D bugfix)
    # ------------------------------------------------------------------
    with preview_tabs[0]:
        try:  # Sprint 20E: tab error containment
            st.markdown("#### Issues to Resolve" if _yc_active else "#### Review Queue")
            st.caption("Unified list of items requiring estimator review")

            # Sprint 20D: FAST_BUDGET explanation banner
            _rq_run_cov = payload.get("run_coverage") or {}
            _rq_sel_mode = _rq_run_cov.get("selection_mode", "")
            _rq_ps = payload.get("processing_stats") or {}
            _rq_total = _rq_ps.get("total_pages") or _rq_run_cov.get("pages_total", 0)
            _rq_deep = _rq_ps.get("deep_processed_pages")
            _rq_ocr = _rq_ps.get("ocr_pages")
            _rq_skipped_count = _rq_ps.get("skipped_pages") or len(_rq_run_cov.get("pages_skipped", []))
            _rq_toxic_count = 0
            _rq_toxic_data = payload.get("toxic_pages")
            if _rq_toxic_data and isinstance(_rq_toxic_data, dict):
                _rq_toxic_count = len([p for p in _rq_toxic_data.get("pages", []) if p.get("toxic")])

            if _rq_sel_mode == "fast_budget" and _rq_skipped_count > 0:
                _fb_parts = [
                    f"**{_rq_deep if _rq_deep is not None else '\u2014'}/{_rq_total}** pages deep-processed",
                ]
                if _rq_ocr is not None:
                    _fb_parts.append(f"**{_rq_ocr}** OCR\u2019d")
                _fb_parts.append(f"**{_rq_skipped_count}** skipped")
                if _rq_toxic_count:
                    _fb_parts.append(f"**{_rq_toxic_count}** toxic")
                st.info(
                    "\U0001f4a1 **FAST BUDGET mode** \u2014 This tender was indexed but not all pages "
                    "were deep-processed due to page count / time budget. "
                    + " \u00b7 ".join(_fb_parts)
                )

                # Sprint 20E: Skipped pages detail panel
                _rq_skipped_list = _rq_run_cov.get("pages_skipped", [])
                if _rq_skipped_list:
                    from collections import Counter as _Counter
                    _skip_by_type = _Counter(s.get("doc_type", "unknown") for s in _rq_skipped_list)
                    with st.expander(f"Skipped Pages Detail ({len(_rq_skipped_list)} pages)", expanded=False):
                        _HIGH_SKIP = {"boq", "schedule", "addendum"}
                        _MED_SKIP = {"conditions", "spec"}
                        for _sdt, _sdc in _skip_by_type.most_common():
                            _skip_icon = (
                                "\U0001f534" if _sdt in _HIGH_SKIP
                                else "\U0001f7e1" if _sdt in _MED_SKIP
                                else "\u26aa"
                            )
                            st.caption(f"{_skip_icon} **{_sdt}**: {_sdc} page(s) skipped")
                        _rq_cap = _rq_ps.get("deep_processed_pages") or "\u2014"
                        st.caption(f"Deep process budget: **{_rq_cap}** / {_rq_total} total")

            # Sprint 20D: Use session-state-patched data for review queue rebuild
            # Bulk actions write updated structures to session state; use those
            # as source of truth when available so queue reflects mutations.
            from src.analysis.review_queue import build_review_queue
            _risk_results = st.session_state.get("_risk_results", [])
            _rq_recon = payload.get("quantity_reconciliation", [])
            _rq_conflicts_raw = payload.get("conflicts", [])
            # Sprint 20D: Prefer session-state conflicts if bulk-reviewed
            _rq_conflicts = st.session_state.get("_reviewed_conflicts", _rq_conflicts_raw)
            _rq_skipped = _rq_run_cov.get("pages_skipped", [])
            _rq_toxic = payload.get("toxic_pages")
            review_items = build_review_queue(
                quantity_reconciliation=_rq_recon,
                conflicts=_rq_conflicts,
                pages_skipped=_rq_skipped,
                toxic_summary=_rq_toxic,
                risk_results=_risk_results,
            )

            if not review_items:
                st.success("No items requiring review.")
            else:
                # Summary counts
                high_count = sum(1 for r in review_items if r["severity"] == "high")
                med_count = sum(1 for r in review_items if r["severity"] == "medium")
                low_count = sum(1 for r in review_items if r["severity"] == "low")
                st.markdown(
                    f"**{len(review_items)} items** | "
                    f":red[{high_count} HIGH] | "
                    f":orange[{med_count} MED] | "
                    f":green[{low_count} LOW]"
                )

                # ── Sprint 20D: Compute eligibility counts ────────────
                _ba_recon = payload.get("quantity_reconciliation", [])
                _ba_sched_eligible = sum(
                    1 for r in _ba_recon
                    if r.get("mismatch") and r.get("category") in ("doors", "windows")
                )
                _ba_high_eligible = sum(
                    1 for r in _ba_recon
                    if r.get("mismatch") and r.get("max_delta", 0) >= 5
                )
                _ba_conflicts_raw = payload.get("conflicts", [])
                _ba_rev_eligible = sum(
                    1 for c in _ba_conflicts_raw
                    if c.get("resolution") == "intentional_revision"
                    and c.get("_review_status") != "reviewed"
                )
                # Adjust for already-applied session state
                if st.session_state.get("_reviewed_conflicts"):
                    _ba_rev_eligible = sum(
                        1 for c in st.session_state["_reviewed_conflicts"]
                        if c.get("resolution") == "intentional_revision"
                        and c.get("_review_status") != "reviewed"
                    )

                # ── Bulk action buttons (Sprint 20D: counts + disable + persistence fix) ─
                st.markdown("##### Bulk Actions")
                bcols_bulk = st.columns(3)
                with bcols_bulk[0]:
                    _ba1_label = f"Prefer Schedule ({_ba_sched_eligible})"
                    _ba1_disabled = _ba_sched_eligible == 0
                    if st.button(
                        _ba1_label, key="bulk_prefer_sched",
                        disabled=_ba1_disabled,
                        help="No eligible door/window mismatches" if _ba1_disabled else None,
                    ):
                        from src.analysis.bulk_actions import prefer_schedule_for_mismatches
                        recon = payload.get("quantity_reconciliation", [])
                        updated_recon, n = prefer_schedule_for_mismatches(recon)
                        if n > 0:
                            # Sprint 20D: Persist full updated recon list and per-category actions
                            _recon_state = st.session_state.get("_qty_recon_actions", {})
                            for r in updated_recon:
                                if r.get("action") == "prefer_schedule":
                                    _recon_state[r.get("category", "")] = r
                            st.session_state["_qty_recon_actions"] = _recon_state
                            st.session_state["_bulk_updated_recon"] = updated_recon
                            st.toast(f"Updated {n} mismatches \u2014 schedule preferred for door/window")
                        else:
                            st.toast("No eligible door/window mismatches found")
                        st.rerun()

                with bcols_bulk[1]:
                    _ba2_label = f"Generate RFIs for HIGH ({_ba_high_eligible})"
                    _ba2_disabled = _ba_high_eligible == 0
                    if st.button(
                        _ba2_label, key="bulk_rfi_high",
                        disabled=_ba2_disabled,
                        help="No HIGH mismatches to convert into RFIs" if _ba2_disabled else None,
                    ):
                        from src.analysis.bulk_actions import generate_rfis_for_high_mismatches
                        recon = payload.get("quantity_reconciliation", [])
                        existing = payload.get("rfis", []) + st.session_state.get("_recon_rfis", [])
                        new_rfis, _updated_recon = generate_rfis_for_high_mismatches(recon, existing)
                        if new_rfis:
                            # Sprint 20D: Extend (not replace) existing session RFIs
                            recon_rfis = list(st.session_state.get("_recon_rfis", []))
                            recon_rfis.extend(new_rfis)
                            st.session_state["_recon_rfis"] = recon_rfis
                            st.session_state["_bulk_updated_recon"] = _updated_recon
                            st.toast(f"Created {len(new_rfis)} RFIs for HIGH impact mismatches")
                        else:
                            st.toast("No HIGH mismatches to convert into RFIs")
                        st.rerun()

                with bcols_bulk[2]:
                    _ba3_label = f"Mark Revisions Reviewed ({_ba_rev_eligible})"
                    _ba3_disabled = _ba_rev_eligible == 0
                    if st.button(
                        _ba3_label, key="bulk_mark_reviewed",
                        disabled=_ba3_disabled,
                        help="No intentional revisions pending review" if _ba3_disabled else None,
                    ):
                        from src.analysis.bulk_actions import mark_intentional_revisions_reviewed
                        conflicts = st.session_state.get("_reviewed_conflicts", payload.get("conflicts", []))
                        updated, n = mark_intentional_revisions_reviewed(conflicts)
                        if n > 0:
                            # Sprint 20D: Persist to session state so review queue re-reads it
                            st.session_state["_reviewed_conflicts"] = updated
                            st.toast(f"Marked {n} intentional revisions as reviewed")
                        else:
                            st.toast("No intentional revisions pending review")
                        st.rerun()

                st.markdown("---")

                # ── Sprint 20D: Rebuilt review item rendering (no overlap) ──
                for ri_idx, ri in enumerate(review_items):
                    sev = ri["severity"]
                    sev_icon = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}.get(sev, "\u26aa")
                    ri_type = ri.get("type", "")
                    type_label = ri_type.replace("_", " ").title()

                    with st.expander(
                        f"{sev_icon} [{type_label}] {ri['title']}",
                        expanded=(sev == "high"),
                    ):
                        # Line 1: Page references (if any)
                        _ri_page_refs = ri.get("page_refs", [])
                        if _ri_page_refs:
                            _ri_pages_str = ", ".join(str(p + 1) for p in _ri_page_refs[:10])
                            if len(_ri_page_refs) > 10:
                                _ri_pages_str += f" +{len(_ri_page_refs) - 10} more"
                            st.caption(f"\U0001f4c4 Pages: {_ri_pages_str}")

                        # Line 2: Recommended action (human-readable)
                        _ri_action = ri.get("recommended_action", "review")
                        _ri_action_display = _ri_action.replace("_", " ").replace("acknowledge ", "").title()
                        st.caption(f"\u27a1\ufe0f Recommended: {_ri_action_display}")

                        # Line 3: Type-specific detail (no raw JSON dump)
                        _ri_eb = ri.get("evidence_bundle", {})
                        if ri_type == "recon_mismatch":
                            _ri_cat = _ri_eb.get("category", "")
                            st.markdown(
                                f"**{_ri_cat.title()}**: "
                                f"Schedule={_ri_eb.get('schedule_count', '\u2014')}, "
                                f"BOQ={_ri_eb.get('boq_count', '\u2014')}, "
                                f"Drawing={_ri_eb.get('drawing_count', '\u2014')} "
                                f"(delta: {_ri_eb.get('max_delta', 0)})"
                            )
                        elif ri_type == "conflict":
                            _ri_desc = _ri_eb.get("description", "")
                            _ri_dc = _ri_eb.get("delta_confidence")
                            if _ri_desc:
                                st.markdown(f"{_ri_desc[:200]}")
                            if _ri_dc is not None:
                                st.caption(f"Confidence: {_ri_dc:.0%}")
                        elif ri_type == "skipped_page":
                            _ri_dt = _ri_eb.get("doc_type", "")
                            _ri_cnt = _ri_eb.get("count", 0)
                            st.markdown(
                                f"**{_ri_cnt}** `{_ri_dt}` page(s) were indexed but not "
                                f"deep-processed. Re-run with a higher OCR budget to include them."
                            )
                        elif ri_type == "toxic_page":
                            _ri_tc = _ri_eb.get("toxic_count", 0)
                            st.markdown(
                                f"**{_ri_tc}** page(s) failed OCR and could not be recovered. "
                                f"These may be scanned images with very low resolution."
                            )
                        elif ri_type == "risk_hit":
                            _ri_label = _ri_eb.get("label", "")
                            _ri_hits = _ri_eb.get("hit_count", 0)
                            st.markdown(f"**{_ri_label}** \u2014 {_ri_hits} hit(s) found")
                        else:
                            # Fallback: render evidence as compact key-value, not raw JSON
                            for _ek, _ev in _ri_eb.items():
                                if _ek not in ("pages", "hits") and _ev:
                                    st.caption(f"{_ek.replace('_', ' ').title()}: {_ev}")

                        # Debug: source_key in collapsed caption (not inline)
                        _ri_source_key = ri.get("source_key", f"rq_{ri_idx}")
                        with st.container():
                            st.caption(f"Source: `{_ri_source_key}`")

                        # Sprint 14: Collaboration widget
                        _render_collab_widget("review_item", _ri_source_key, widget_key=f"rq_{ri_idx}")

        except Exception as _tab_err_0:
            _handle_tab_error("Review Queue", _tab_err_0)

    # ------------------------------------------------------------------
    # TAB 1: Blockers (table + expandable evidence)
    # ------------------------------------------------------------------
    with preview_tabs[1]:
        try:  # Sprint 20E: tab error containment
            # Normalize all blockers to safe dicts
            norm_blockers = [normalize_blocker(b) for b in blockers]
            real_blockers = [nb for nb in norm_blockers if not (nb["issue_type"] == "missing_drawing" and nb["trade"].lower() == "mep")]
            mep_fyi = [nb for nb in norm_blockers if nb["issue_type"] == "missing_drawing" and nb["trade"].lower() == "mep"]

            st.markdown(f"#### Blockers ({len(real_blockers)})")

            if real_blockers:
                # Build HTML table
                rows_html = ""
                for nb in real_blockers:
                    sev = nb["severity"]
                    sev_cls = "sev-critical" if sev in ("critical", "high") else "sev-medium" if sev == "medium" else "sev-low"
                    # Confidence column
                    ev_conf = nb["evidence"].get("confidence", 0) if nb["evidence"] else 0
                    conf_pct = int(ev_conf * 100) if ev_conf else 0
                    conf_str = f"{conf_pct}%" if conf_pct else "\u2014"
                    # Recommended fix column (first fix action)
                    fix_list = nb["fix_actions"] or []
                    rec_fix = _safe_str(fix_list[0])[:60] if fix_list else "\u2014"
                    # Coverage status column
                    cov = nb.get("coverage_status", "")
                    if cov == "not_found_after_search":
                        cov_label = '<span style="color:#4ade80;">&#10003; Searched</span>'
                    elif cov == "unknown_not_processed":
                        cov_label = '<span style="color:#fbbf24;">&#9888; Not Checked</span>'
                    else:
                        cov_label = "\u2014"
                    rows_html += f"""<tr>
                        <td><span class="sev-pill {sev_cls}">{sev.upper()}</span></td>
                        <td style="font-weight:500;">{nb['trade'].title()}</td>
                        <td>{nb['title']}</td>
                        <td style="color:#71717a;font-size:0.78rem;">{nb['pages'] or '\u2014'}</td>
                        <td>{conf_str}</td>
                        <td>{cov_label}</td>
                        <td>{nb['unlocks'] or '\u2014'}</td>
                        <td style="font-size:0.82rem;">{rec_fix}</td>
                    </tr>"""

                st.markdown(f"""
                <table class="result-table">
                    <thead><tr>
                        <th>Severity</th><th>Trade</th><th>Issue</th>
                        <th>Pages</th><th>Confidence</th><th>Coverage</th><th>Impacted Outputs</th><th>Recommended Fix</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

                # Expandable evidence per blocker (with PDF page preview)
                for idx, nb in enumerate(real_blockers):
                    raw_b = blockers[idx] if idx < len(blockers) else {}
                    ev = dict(nb["evidence"]) if nb["evidence"] else {}
                    ev["coverage_status"] = nb.get("coverage_status", "")
                    render_evidence_expander(
                        ev,
                        title=f"Evidence: {nb['title'][:60]}",
                        item_id=nb["id"],
                    )
                    # Bbox selector (Sprint 5)
                    if ev.get("bbox"):
                        render_bbox_selector(ev, nb["id"], blockers, rfis)
                    # Skipped-page linkage for UNKNOWN items
                    if nb.get("coverage_status") == "unknown_not_processed":
                        _skipped_all = payload.get("run_coverage", {}).get("pages_skipped", [])
                        _rel_types = _item_doc_types(nb)
                        _rel_skipped = [s for s in _skipped_all if s.get("doc_type") in _rel_types]
                        if _rel_skipped:
                            _skip_pages = [str(s.get("page_idx", 0) + 1) for s in _rel_skipped[:5]]
                            st.caption(
                                f"⚠️ Related pages were skipped: {', '.join(_skip_pages)} "
                                f"— re-run with higher budget to verify"
                            )
                    # What's blocked + fix actions (inline, below the expander)
                    raw_unlocks = raw_b.get("unlocks_boq_categories", [])
                    if raw_unlocks:
                        pills_html = "".join(
                            f'<span class="boq-pill">{cat.replace("_", " ").title()}</span>'
                            for cat in raw_unlocks
                        )
                        st.markdown(f"**What's Blocked:** {pills_html}", unsafe_allow_html=True)
                    if nb["fix_actions"]:
                        with st.expander(f"Fix options: {nb['title'][:40]}"):
                            for fix in nb["fix_actions"][:3]:
                                st.markdown(f"- {_safe_str(fix)}")
                            if nb["score_delta"] > 0:
                                st.markdown(f'<span class="score-delta">+{nb["score_delta"]} pts if fixed</span>', unsafe_allow_html=True)
            else:
                st.info("No blockers detected. Drawing set appears complete.")

            # MEP FYI
            if mep_fyi:
                st.markdown("---")
                st.markdown("#### FYI Notes")
                for nb in mep_fyi:
                    st.markdown(
                        f'<div class="fyi-box">\u2139\ufe0f <strong>{nb["title"]}</strong>'
                        f'{"<br><span style=font-size:0.8rem>" + nb["description"] + "</span>" if nb["description"] else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Sprint 20E: Coverage gap warnings from run_coverage
            _cg_run_cov = payload.get("run_coverage") or {}
            _cg_not_covered = _cg_run_cov.get("doc_types_not_covered", [])
            _cg_partial = _cg_run_cov.get("doc_types_partially_covered", [])
            _cg_sel_mode = _cg_run_cov.get("selection_mode", "")

            _cg_warnings = []
            if "boq" in _cg_not_covered:
                _cg_warnings.append(("\U0001f534", "BOQ pages not processed",
                                     "BOQ pages were indexed but skipped. Quantification may be incomplete."))
            if "addendum" in _cg_not_covered:
                _cg_warnings.append(("\U0001f534", "Addenda not processed",
                                     "Addendum pages were detected but not analyzed. Changes may be missed."))
            if "conditions" in _cg_partial or "conditions" in _cg_not_covered:
                _cond_total = (_cg_run_cov.get("doc_types_detected") or {}).get("conditions", 0)
                _cg_warnings.append(("\U0001f7e1", "Commercial conditions partially skipped",
                                     f"{_cond_total} conditions page(s) detected, not all deep-processed."))
            if "spec" in _cg_not_covered:
                _cg_warnings.append(("\U0001f7e1", "Specification pages skipped",
                                     "Spec pages were indexed but not extracted."))

            # Check for zero BOQ items when BOQ pages existed
            _cg_boq_items = (payload.get("boq_stats") or {}).get("total_items", 0)
            _cg_boq_pages_detected = (_cg_run_cov.get("doc_types_detected") or {}).get("boq", 0)
            if _cg_boq_items == 0 and _cg_boq_pages_detected > 0:
                _cg_warnings.append(("\U0001f534", "No BOQ items extracted",
                                     f"{_cg_boq_pages_detected} BOQ page(s) detected but 0 line items found."))

            if _cg_warnings:
                st.markdown("---")
                st.markdown("##### Coverage Gaps")
                for _cg_icon, _cg_title, _cg_desc in _cg_warnings:
                    st.warning(f"{_cg_icon} **{_cg_title}**: {_cg_desc}")

        except Exception as _tab_err_1:
            _handle_tab_error("Blockers", _tab_err_1)

    # ------------------------------------------------------------------
    # TAB 2: RFIs (table + expandable evidence + CSV export)
    # ------------------------------------------------------------------
    with preview_tabs[2]:
        try:  # Sprint 20E: tab error containment
            norm_rfis = [normalize_rfi(r) for r in rfis]
            rfis_with_evidence = [nr for nr in norm_rfis if nr["pages"] or nr.get("confidence", 0) > 0]

            if not norm_rfis:
                st.info("No RFIs generated. Drawing set appears complete.")
            elif not rfis_with_evidence:
                st.warning("Insufficient data to generate evidence-backed RFIs.")
                st.caption(f"{len(norm_rfis)} items found but none have page-level evidence.")
            else:
                st.markdown(f"#### RFIs ({len(norm_rfis)})")

                # ── Sprint 10: Grouped view toggle ──────────────────────────
                rfi_clusters = payload.get("rfi_clusters", [])
                _show_grouped_rfis = False
                if rfi_clusters:
                    _show_grouped_rfis = st.checkbox(
                        "Grouped view", value=False, key="_rfi_grouped_view",
                        help="Group similar RFIs by trade and text similarity",
                    )

                if _show_grouped_rfis and rfi_clusters:
                    _render_rfi_grouped_view(rfi_clusters, norm_rfis, project_id)

                if not _show_grouped_rfis:
                    _render_rfi_raw_view(norm_rfis, rfis, blockers, payload, project_id)

                # ── Sprint 12: RFI Feedback ──────────────────────────────
                st.markdown("---")
                st.markdown("##### RFI Feedback")
                rfi_list_fb = payload.get("rfis", [])
                if rfi_list_fb:
                    rfi_labels = [
                        f"{r.get('id', f'RFI-{i+1}')}: {r.get('question', '')[:60]}"
                        for i, r in enumerate(rfi_list_fb)
                    ]
                    selected_rfi_label = st.selectbox("Select RFI:", rfi_labels, key="fb_rfi_select")
                    sel_rfi_idx = rfi_labels.index(selected_rfi_label) if selected_rfi_label in rfi_labels else 0
                    sel_rfi = rfi_list_fb[sel_rfi_idx]
                    rfi_id = sel_rfi.get("id", f"RFI-{sel_rfi_idx + 1}")
                    rfi_pages = [e.get("page") for e in sel_rfi.get("evidence_refs", []) if isinstance(e, dict)]
                    fb_rfi_cols = st.columns(3)
                    with fb_rfi_cols[0]:
                        if st.button("\u2705 Correct", key="fb_rfi_ok"):
                            from src.analysis.feedback import make_feedback_entry
                            entry = make_feedback_entry("rfi", rfi_id, "correct", page_refs=rfi_pages)
                            _fb_log = st.session_state.get("_feedback_log", [])
                            _fb_log.append(entry)
                            st.session_state["_feedback_log"] = _fb_log
                            st.toast(f"RFI {rfi_id} marked correct")
                    with fb_rfi_cols[1]:
                        if st.button("\u274c Wrong", key="fb_rfi_wrong"):
                            from src.analysis.feedback import make_feedback_entry
                            entry = make_feedback_entry("rfi", rfi_id, "wrong", page_refs=rfi_pages)
                            _fb_log = st.session_state.get("_feedback_log", [])
                            _fb_log.append(entry)
                            st.session_state["_feedback_log"] = _fb_log
                            st.toast(f"RFI {rfi_id} marked wrong")
                    with fb_rfi_cols[2]:
                        corrected_rfi = st.text_input("Corrected question:", key="fb_rfi_corrected")
                        if st.button("\u270f\ufe0f Save Edit", key="fb_rfi_save") and corrected_rfi:
                            from src.analysis.feedback import make_feedback_entry
                            entry = make_feedback_entry("rfi", rfi_id, "edited",
                                corrected_value=corrected_rfi, original_value=sel_rfi.get("question", ""),
                                page_refs=rfi_pages)
                            _fb_log = st.session_state.get("_feedback_log", [])
                            _fb_log.append(entry)
                            st.session_state["_feedback_log"] = _fb_log
                            st.toast(f"RFI edit saved for {rfi_id}")
                    _fb_count = len(st.session_state.get("_feedback_log", []))
                    if _fb_count > 0:
                        st.caption(f"{_fb_count} feedback entries recorded this session")

            # ── Sprint 13: RFI Approval Status ──────────────────────────
            st.markdown("---")
            st.markdown("##### RFI Approval Status")
            _rfi_statuses = st.session_state.get("_rfi_statuses", {})
            _all_rfis_for_approval = payload.get("rfis", [])
            if _all_rfis_for_approval:
                for _ra_i, _ra_rfi in enumerate(_all_rfis_for_approval):
                    _ra_id = _ra_rfi.get("id", f"rfi_{_ra_i}")
                    _ra_current = _rfi_statuses.get(_ra_id, _ra_rfi.get("status", "draft"))
                    _ra_col_q, _ra_col_s = st.columns([4, 1])
                    with _ra_col_q:
                        st.caption(f"**{_ra_id}** {_ra_rfi.get('question', '')[:80]}")
                    with _ra_col_s:
                        _ra_opts = ["draft", "approved", "sent"]
                        _ra_new = st.selectbox(
                            "Status",
                            options=_ra_opts,
                            index=_ra_opts.index(_ra_current) if _ra_current in _ra_opts else 0,
                            key=f"rfi_status_{_ra_id}",
                            label_visibility="collapsed",
                        )
                        if _ra_new != _ra_current:
                            _rfi_statuses[_ra_id] = _ra_new
                            st.session_state["_rfi_statuses"] = _rfi_statuses
                    # Sprint 14: Collaboration widget for each RFI
                    _render_collab_widget("rfi", _ra_id, widget_key=f"rfi_{_ra_i}")
            else:
                st.caption("No RFIs to review.")

        except Exception as _tab_err_2:
            _handle_tab_error("RFIs", _tab_err_2)

    # ------------------------------------------------------------------
    # TAB 3: Extracted Artifacts
    # ------------------------------------------------------------------
    with preview_tabs[3]:
        _safe_tab("Bid Pack", render_bid_pack_tab, payload)

    # ------------------------------------------------------------------
    # TAB 4: Selection & Coverage
    # ------------------------------------------------------------------
    with preview_tabs[4]:
        _safe_tab("Coverage Dashboard", render_coverage_dashboard, payload)
        # Sprint 20F: Extraction diagnostics panel (under coverage tab)
        _safe_tab("Extraction Diagnostics", render_extraction_diagnostics, payload)

    # ------------------------------------------------------------------
    # TAB 5: Bid Strategy
    # ------------------------------------------------------------------
    with preview_tabs[5]:
        _safe_tab("Bid Strategy", render_bid_strategy_tab, payload)

    # ------------------------------------------------------------------
    # TAB 6: Risk Checklist (Sprint 7)
    # ------------------------------------------------------------------
    with preview_tabs[6]:
        _safe_tab("Risk Checklist", render_risk_checklist_tab, payload)

    # ------------------------------------------------------------------
    # TAB 7: Quantities (Sprint 11)
    # ------------------------------------------------------------------
    with preview_tabs[7]:
        try:  # Sprint 20E: tab error containment
            st.markdown("#### Quantities")
            quantities = payload.get("quantities", [])
            if quantities:
                import pandas as pd

                # Build dataframe
                qty_rows = []
                for q in quantities:
                    pages = q.get("evidence_refs", [])
                    page_str = ", ".join(str((p.get("page", 0) or 0) + 1) for p in pages[:5]) if pages else ""
                    qty_rows.append({
                        "Item": q.get("item", ""),
                        "Unit": q.get("unit", ""),
                        "Qty": q.get("qty", 0),
                        "Confidence": f"{q.get('confidence', 0):.0%}",
                        "Source": q.get("source_type", ""),
                        "Trade": q.get("trade", ""),
                        "Pages": page_str,
                    })
                qty_df = pd.DataFrame(qty_rows)

                # Filter by source type
                source_types = sorted(qty_df["Source"].unique().tolist())
                selected_sources = st.multiselect(
                    "Filter by source",
                    options=source_types,
                    default=source_types,
                    key="qty_source_filter",
                )
                if selected_sources:
                    filtered_df = qty_df[qty_df["Source"].isin(selected_sources)]
                else:
                    filtered_df = qty_df

                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                st.caption(f"{len(filtered_df)} of {len(qty_df)} quantity rows shown")

                # Download button
                csv_data = qty_df.to_csv(index=False)
                st.download_button(
                    "Download quantities.csv",
                    csv_data, "quantities.csv", "text/csv",
                    key="dl_quantities_csv",
                )

                # ── Sprint 12: Quantity Feedback ──────────────────────────────
                st.markdown("---")
                st.markdown("##### Quantity Feedback")
                qty_items = [q.get("item", f"Row {i}") for i, q in enumerate(quantities)]
                selected_qty_label = st.selectbox("Select quantity item:", qty_items, key="fb_qty_select")
                sel_idx = qty_items.index(selected_qty_label) if selected_qty_label in qty_items else 0
                sel_q = quantities[sel_idx]
                bundle_id = sel_q.get("evidence_bundle_id", f"qty_{sel_idx}")
                fb_cols = st.columns(3)
                with fb_cols[0]:
                    if st.button("\u2705 Correct", key="fb_qty_ok"):
                        from src.analysis.feedback import make_feedback_entry
                        entry = make_feedback_entry("quantity", bundle_id, "correct")
                        _fb_log = st.session_state.get("_feedback_log", [])
                        _fb_log.append(entry)
                        st.session_state["_feedback_log"] = _fb_log
                        st.toast("Quantity marked correct")
                with fb_cols[1]:
                    if st.button("\u274c Wrong", key="fb_qty_wrong"):
                        from src.analysis.feedback import make_feedback_entry
                        entry = make_feedback_entry("quantity", bundle_id, "wrong",
                            original_value=str(sel_q.get("qty", "")))
                        _fb_log = st.session_state.get("_feedback_log", [])
                        _fb_log.append(entry)
                        st.session_state["_feedback_log"] = _fb_log
                        st.toast("Quantity marked wrong")
                with fb_cols[2]:
                    corrected_qty = st.text_input("Corrected value:", key="fb_qty_corrected")
                    if st.button("\u270f\ufe0f Save Edit", key="fb_qty_edit") and corrected_qty:
                        from src.analysis.feedback import make_feedback_entry
                        entry = make_feedback_entry("quantity", bundle_id, "edited",
                            corrected_value=corrected_qty, original_value=str(sel_q.get("qty", "")))
                        _fb_log = st.session_state.get("_feedback_log", [])
                        _fb_log.append(entry)
                        st.session_state["_feedback_log"] = _fb_log
                        st.toast("Quantity edit saved")

            # ── Sprint 13: Quantity Acceptance ──────────────────────────
            if quantities:
                st.markdown("---")
                st.markdown("##### Quantity Acceptance")
                _qty_statuses = st.session_state.get("_qty_statuses", {})

                # Bulk accept all
                if st.button("Accept All Quantities", key="bulk_accept_qtys"):
                    for _qa_q in quantities:
                        _qa_bid = _qa_q.get("evidence_bundle_id", f"qty_{quantities.index(_qa_q)}")
                        _qty_statuses[_qa_bid] = "accepted"
                    st.session_state["_qty_statuses"] = _qty_statuses
                    st.toast(f"Accepted {len(quantities)} quantities")
                    st.rerun()

                # Summary
                _accepted_count = sum(
                    1 for q in quantities
                    if _qty_statuses.get(q.get("evidence_bundle_id", ""), q.get("status", "draft")) == "accepted"
                )
                st.caption(f"{_accepted_count} of {len(quantities)} quantities accepted")

                # Per-item checkboxes (compact: 2-column layout)
                _qa_cols = st.columns(2)
                for _qa_i, _qa_q in enumerate(quantities):
                    _qa_bid = _qa_q.get("evidence_bundle_id", f"qty_{_qa_i}")
                    _qa_current = _qty_statuses.get(_qa_bid, _qa_q.get("status", "draft"))
                    with _qa_cols[_qa_i % 2]:
                        _qa_checked = st.checkbox(
                            f"{_qa_q.get('item', '')[:35]} ({_qa_q.get('qty', '?')} {_qa_q.get('unit', '')})",
                            value=(_qa_current == "accepted"),
                            key=f"qty_accept_{_qa_bid}",
                        )
                        _qty_statuses[_qa_bid] = "accepted" if _qa_checked else "draft"
                        # Sprint 14: Collaboration widget per quantity
                        _render_collab_widget("quantity", _qa_bid, widget_key=f"qty_{_qa_i}")
                st.session_state["_qty_statuses"] = _qty_statuses

            # ── Sprint 12: Quantity Reconciliation ──────────────────────
            st.markdown("---")
            st.markdown("##### Quantity Reconciliation")
            qty_recon = payload.get("quantity_reconciliation", [])
            if qty_recon:
                import pandas as pd
                recon_rows_ui = []
                for r in qty_recon:
                    recon_rows_ui.append({
                        "Category": r.get("category", "").title(),
                        "Schedule": r.get("schedule_count") if r.get("schedule_count") is not None else "\u2014",
                        "BOQ": r.get("boq_count") if r.get("boq_count") is not None else "\u2014",
                        "Drawing": r.get("drawing_count") if r.get("drawing_count") is not None else "\u2014",
                        "Mismatch": "\u26a0\ufe0f Yes" if r.get("mismatch") else "\u2705 No",
                        "Delta": r.get("max_delta", 0),
                    })
                st.dataframe(pd.DataFrame(recon_rows_ui), use_container_width=True, hide_index=True)

                # Action buttons per mismatch row
                mismatches = [r for r in qty_recon if r.get("mismatch")]
                if mismatches:
                    st.markdown("**Resolve mismatches:**")
                    for m_idx, mismatch in enumerate(mismatches):
                        cat = mismatch.get("category", "unknown")
                        st.caption(
                            f"**{cat.title()}** \u2014 "
                            f"Schedule: {mismatch.get('schedule_count', '\u2014')}, "
                            f"BOQ: {mismatch.get('boq_count', '\u2014')}, "
                            f"Drawing: {mismatch.get('drawing_count', '\u2014')}"
                        )
                        bcols = st.columns(4)
                        _recon_state = st.session_state.get("_qty_recon_actions", {})
                        with bcols[0]:
                            if st.button("Prefer Schedule", key=f"recon_sched_{m_idx}"):
                                from src.analysis.quantity_reconciliation import apply_reconciliation_action
                                _recon_state[cat] = apply_reconciliation_action(mismatch, "prefer_schedule")
                                st.session_state["_qty_recon_actions"] = _recon_state
                                st.success(f"Preferred schedule for {cat}")
                        with bcols[1]:
                            if st.button("Prefer BOQ", key=f"recon_boq_{m_idx}"):
                                from src.analysis.quantity_reconciliation import apply_reconciliation_action
                                _recon_state[cat] = apply_reconciliation_action(mismatch, "prefer_boq")
                                st.session_state["_qty_recon_actions"] = _recon_state
                                st.success(f"Preferred BOQ for {cat}")
                        with bcols[2]:
                            if st.button("Create RFI", key=f"recon_rfi_{m_idx}"):
                                from src.analysis.quantity_reconciliation import apply_reconciliation_action
                                _recon_state[cat] = apply_reconciliation_action(mismatch, "create_rfi")
                                st.session_state["_qty_recon_actions"] = _recon_state
                                st.info(f"RFI flagged for {cat}")
                        with bcols[3]:
                            if st.button("Add Assumption", key=f"recon_asmp_{m_idx}"):
                                from src.analysis.quantity_reconciliation import apply_reconciliation_action
                                _recon_state[cat] = apply_reconciliation_action(mismatch, "add_assumption")
                                st.session_state["_qty_recon_actions"] = _recon_state
                                st.info(f"Assumption flagged for {cat}")
            else:
                st.caption("No quantity reconciliation data available.")

            # ── Sprint 12: Finishes Takeoff ────────────────────────────
            st.markdown("---")
            st.markdown("##### Finishes")
            finish_takeoff = payload.get("finish_takeoff")
            if finish_takeoff and isinstance(finish_takeoff, dict):
                if finish_takeoff.get("has_areas"):
                    import pandas as pd
                    finish_rows_data = []
                    for fr in finish_takeoff.get("finish_rows", []):
                        finish_rows_data.append({
                            "Finish Type": fr.get("finish_type", "").title(),
                            "Material": fr.get("material", ""),
                            "Total Area (sqm)": fr.get("total_area_sqm") if fr.get("total_area_sqm") is not None else "\u2014",
                            "Room Count": fr.get("room_count", 0),
                            "Rooms": ", ".join(fr.get("rooms", [])[:5]),
                        })
                    if finish_rows_data:
                        st.dataframe(pd.DataFrame(finish_rows_data), use_container_width=True, hide_index=True)
                    st.caption(finish_takeoff.get("summary", ""))
                else:
                    missing = finish_takeoff.get("rooms_missing_area", [])
                    if finish_takeoff.get("finish_rows"):
                        st.warning("Finish schedule found but room areas are missing.")
                        if missing:
                            st.caption(f"Rooms lacking area data: {', '.join(missing[:20])}")
                    else:
                        st.caption(finish_takeoff.get("summary", "No finish schedules detected."))
            else:
                st.caption("No finish schedules detected.")

        except Exception as _tab_err_7:
            _handle_tab_error("Quantities", _tab_err_7)

    # ------------------------------------------------------------------
    # TAB 8: Pre-bid Meeting (Sprint 15)
    # ------------------------------------------------------------------
    with preview_tabs[8]:
        st.markdown("#### Pre-bid Meeting Agenda")
        st.caption("Auto-generated agenda from top review items and open assignments")

        try:
            from src.analysis.meeting_agenda import build_meeting_agenda, generate_agenda_docx, generate_agenda_pdf
            from src.analysis.review_queue import build_review_queue
            from src.analysis.collaboration import load_collaboration, get_all_assignments

            # Rebuild review queue (same sources as TAB 0)
            _mtg_recon = payload.get("quantity_reconciliation", [])
            _mtg_conflicts = payload.get("conflicts", [])
            _mtg_skipped = (payload.get("run_coverage") or {}).get("pages_skipped", [])
            _mtg_toxic = payload.get("toxic_pages")
            _mtg_risk = st.session_state.get("_risk_results", [])
            _mtg_review_items = build_review_queue(
                quantity_reconciliation=_mtg_recon,
                conflicts=_mtg_conflicts,
                pages_skipped=_mtg_skipped,
                toxic_summary=_mtg_toxic,
                risk_results=_mtg_risk,
            )

            # Load assignments if project active
            _mtg_assignments = []
            _mtg_pid = st.session_state.get("_active_project_id", "")
            if _mtg_pid:
                try:
                    from src.analysis.projects import project_dir as _project_dir
                    _mtg_entries = load_collaboration(_project_dir(_mtg_pid))
                    _mtg_assignments = get_all_assignments(_mtg_entries)
                except Exception:
                    pass

            _mtg_agenda = build_meeting_agenda(
                review_items=_mtg_review_items,
                assignments=_mtg_assignments,
                rfis=payload.get("rfis", []),
                project_name=st.session_state.get("_active_project_name", project_id),
            )

            # Summary metrics
            _mtg_summary = _mtg_agenda.get("summary", {})
            _mtg_mc = st.columns(3)
            _mtg_mc[0].metric("Total Items", _mtg_summary.get("total_items", 0))
            _mtg_mc[1].metric("High Priority", _mtg_summary.get("high_count", 0))
            _mtg_mc[2].metric("Assigned", _mtg_summary.get("assigned_count", 0))

            # Render sections
            for _mtg_sec_idx, _mtg_section in enumerate(_mtg_agenda.get("sections", [])):
                st.markdown(f"##### {_mtg_section['title']}")
                _mtg_sec_items = _mtg_section.get("items", [])
                if not _mtg_sec_items:
                    st.caption("No items in this section.")
                    continue

                for _mtg_i, _mtg_item in enumerate(_mtg_sec_items):
                    _mtg_c1, _mtg_c2, _mtg_c3, _mtg_c4 = st.columns([3, 1.5, 1.5, 2])
                    with _mtg_c1:
                        _sev_icon = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}.get(
                            _mtg_item.get("severity", ""), "\u26aa")
                        st.write(f"{_sev_icon} {_mtg_item.get('title', '')[:60]}")
                    with _mtg_c2:
                        st.caption(_mtg_item.get("assigned_to", "") or "Unassigned")
                    with _mtg_c3:
                        st.caption(_mtg_item.get("due_date", "") or "No date")
                    with _mtg_c4:
                        _mtg_bk = f"mtg_{_mtg_sec_idx}_{_mtg_i}"
                        _mtg_bc = st.columns(3)
                        with _mtg_bc[0]:
                            if st.button("\u2705", key=f"{_mtg_bk}_resolve", help="Mark resolved"):
                                st.toast(f"Marked {_mtg_item.get('id', '')} as resolved")
                        with _mtg_bc[1]:
                            _mtg_reassign = st.text_input("", key=f"{_mtg_bk}_reassign",
                                                          placeholder="Assign", label_visibility="collapsed")
                            if _mtg_reassign and _mtg_pid:
                                try:
                                    from src.analysis.collaboration import make_collaboration_entry, append_collaboration
                                    _mtg_entry = make_collaboration_entry(
                                        _mtg_item.get("type", "review_item"),
                                        _mtg_item.get("id", ""),
                                        "assign", {"assigned_to": _mtg_reassign})
                                    append_collaboration(_mtg_entry, _project_dir(_mtg_pid))
                                except Exception:
                                    pass
                        with _mtg_bc[2]:
                            _mtg_due = st.text_input("", key=f"{_mtg_bk}_due",
                                                     placeholder="Due", label_visibility="collapsed")
                            if _mtg_due and _mtg_pid:
                                try:
                                    from src.analysis.collaboration import make_collaboration_entry, append_collaboration
                                    _mtg_entry = make_collaboration_entry(
                                        _mtg_item.get("type", "review_item"),
                                        _mtg_item.get("id", ""),
                                        "due_date", {"due_date": _mtg_due})
                                    append_collaboration(_mtg_entry, _project_dir(_mtg_pid))
                                except Exception:
                                    pass

            # Export buttons
            st.markdown("---")
            _mtg_dl = st.columns(2)
            with _mtg_dl[0]:
                try:
                    _mtg_docx = generate_agenda_docx(_mtg_agenda)
                    st.download_button("\U0001f4c4 Download agenda.docx", _mtg_docx,
                                       "agenda.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       key="mtg_docx_dl")
                except Exception:
                    st.caption("DOCX export unavailable")
            with _mtg_dl[1]:
                try:
                    _mtg_pdf = generate_agenda_pdf(_mtg_agenda)
                    st.download_button("\U0001f4c4 Download agenda.pdf", _mtg_pdf,
                                       "agenda.pdf", "application/pdf", key="mtg_pdf_dl")
                except Exception:
                    st.caption("PDF export unavailable")
        except Exception as _mtg_err:
            _handle_tab_error("Pre-bid Meeting", _mtg_err)

    # ------------------------------------------------------------------
    # TAB 9: Quality Dashboard (Sprint 15)
    # ------------------------------------------------------------------
    with preview_tabs[9]:
        st.markdown("#### Quality Dashboard")
        st.caption("Trends from feedback and run history")

        try:
            from src.analysis.quality_dashboard import compute_quality_metrics

            # Load feedback
            _qd_feedback = st.session_state.get("_feedback_log", [])

            # Load run history if project active
            _qd_runs = []
            _qd_pid = st.session_state.get("_active_project_id", "")
            if _qd_pid:
                try:
                    from src.analysis.projects import list_runs
                    _qd_runs = list_runs(_qd_pid)
                except Exception:
                    pass

            # Cache stats from payload
            _qd_cache = payload.get("cache_stats", {})

            _qd_metrics = compute_quality_metrics(_qd_feedback, _qd_runs, _qd_cache)

            # Metric cards
            _qd_mc = st.columns(4)
            _qd_mc[0].metric("RFI Acceptance Rate", f"{_qd_metrics['rfi_acceptance_rate']:.0%}")
            _qd_mc[1].metric("Correction Rate", f"{_qd_metrics['correction_rate']:.0%}")
            _qd_mc[2].metric("Cache Time Saved", f"{_qd_metrics['cache_time_saved_seconds']:.0f}s")
            _qd_mc[3].metric("Total Feedback", str(_qd_metrics['total_feedback_count']))

            # Top noisy checks
            _qd_noisy = _qd_metrics.get("top_noisy_checks", [])
            if _qd_noisy:
                st.markdown("##### Top Noisy Checks")
                for _nc in _qd_noisy:
                    st.write(f"- **{_nc['check_id']}**: {_nc['wrong_count']} wrong / {_nc['total_count']} total")

            # Readiness Score Trend
            _qd_trend = _qd_metrics.get("trend_data", [])
            if _qd_trend:
                st.markdown("##### Readiness Score Trend")
                try:
                    import pandas as pd
                    _qd_df = pd.DataFrame(_qd_trend)
                    if "readiness_score" in _qd_df.columns and "timestamp" in _qd_df.columns:
                        st.line_chart(_qd_df.set_index("timestamp")["readiness_score"])
                except Exception:
                    st.caption("Chart unavailable")

            # Feedback by day
            _qd_fbd = _qd_metrics.get("feedback_by_day", [])
            if _qd_fbd:
                st.markdown("##### Feedback Volume by Day")
                try:
                    import pandas as pd
                    _qd_df_fb = pd.DataFrame(_qd_fbd)
                    if "date" in _qd_df_fb.columns:
                        st.bar_chart(_qd_df_fb.set_index("date"))
                except Exception:
                    st.caption("Chart unavailable")

            if not _qd_feedback and not _qd_runs:
                st.info("No feedback or run history data available yet. "
                        "Rate RFIs/quantities in the relevant tabs to build quality metrics.")
        except Exception as _qd_err:
            _handle_tab_error("Quality Dashboard", _qd_err)

    # ------------------------------------------------------------------
    # TAB 10: Raw JSON
    # ------------------------------------------------------------------
    with preview_tabs[10]:
        try:  # Sprint 20E: tab error containment
            if _yc_active:
                st.info("Raw JSON payload hidden in YC Demo Mode. Disable Demo Mode to view.")
            else:
                st.markdown("#### Full Analysis Payload")
                st.json(payload)
            st.download_button(
                "\U0001f4e5 Download analysis.json",
                json.dumps(payload, indent=2, default=str),
                f"analysis_{project_id}.json",
                "application/json",
                use_container_width=True
            )

        except Exception as _tab_err_10:
            _handle_tab_error("Raw JSON", _tab_err_10)

    # ------------------------------------------------------------------
    # TAB 11: Ground Truth (Sprint 20)
    # ------------------------------------------------------------------
    with preview_tabs[11]:
        st.markdown("#### Ground Truth & Pilot Tools")
        try:
            from src.analysis.ground_truth import (
                generate_template_csv,
                parse_excel_sheets,
                read_excel_sheet,
                read_csv_file,
                apply_column_mapping,
                save_gt_mapping,
                load_gt_mapping,
                save_gt_data,
                load_gt_data,
                GT_BOQ_COLUMNS,
                GT_SCHEDULES_DOORS_COLUMNS,
                GT_QUANTITIES_COLUMNS,
            )
            from src.analysis.ground_truth_diff import compute_gt_diff
            from src.analysis.training_pack import build_training_pack
            from app.pilot_docs import (
                generate_pilot_agreement_docx,
                generate_data_handling_docx,
                generate_pilot_checklist_docx,
            )
            from src.analysis.projects import load_project

            _gt_pid = st.session_state.get("_active_project_id", "")
            from datetime import datetime as _dt_gt
            _gt_run_id = f"run_{_dt_gt.now().strftime('%Y%m%d_%H%M%S')}"

            # ── 1. Template downloads ──────────────────────────────
            st.markdown("##### GT Templates")
            st.caption("Download blank templates, fill in your estimate data, then upload below.")
            _gt_dl_cols = st.columns(3)
            with _gt_dl_cols[0]:
                st.download_button(
                    "\U0001f4cb GT BOQ Template",
                    generate_template_csv("gt_boq"),
                    "gt_boq_template.csv",
                    "text/csv",
                    key="gt_dl_boq",
                    use_container_width=True,
                )
            with _gt_dl_cols[1]:
                st.download_button(
                    "\U0001f6aa GT Schedules Template",
                    generate_template_csv("gt_schedules_doors"),
                    "gt_schedules_doors_template.csv",
                    "text/csv",
                    key="gt_dl_sched",
                    use_container_width=True,
                )
            with _gt_dl_cols[2]:
                st.download_button(
                    "\U0001f4d0 GT Quantities Template",
                    generate_template_csv("gt_quantities"),
                    "gt_quantities_template.csv",
                    "text/csv",
                    key="gt_dl_qty",
                    use_container_width=True,
                )

            st.markdown("---")

            # ── 2. File upload ─────────────────────────────────────
            st.markdown("##### Upload Ground Truth")
            _gt_type = st.selectbox(
                "Ground truth type",
                ["gt_boq", "gt_schedules_doors", "gt_quantities"],
                format_func=lambda x: {"gt_boq": "BOQ", "gt_schedules_doors": "Schedules (Doors)", "gt_quantities": "Quantities"}.get(x, x),
                key="_gt_type_select",
            )
            _gt_file = st.file_uploader(
                "Upload ground truth (xlsx or csv)",
                type=["xlsx", "csv"],
                key="_gt_file_upload",
            )

            if _gt_file is not None:
                _gt_bytes = _gt_file.getvalue()
                _gt_fname = _gt_file.name

                _canonical_cols_map = {
                    "gt_boq": GT_BOQ_COLUMNS,
                    "gt_schedules_doors": GT_SCHEDULES_DOORS_COLUMNS,
                    "gt_quantities": GT_QUANTITIES_COLUMNS,
                }
                _canonical_cols = _canonical_cols_map.get(_gt_type, [])

                if _gt_fname.lower().endswith(".xlsx"):
                    # Sheet picker
                    _sheets = parse_excel_sheets(_gt_bytes)
                    _selected_sheet = st.selectbox(
                        "Select sheet", _sheets, key="_gt_sheet_select",
                    )
                    if _selected_sheet:
                        _raw_headers, _raw_rows = read_excel_sheet(_gt_bytes, _selected_sheet)
                        st.caption(f"Found {len(_raw_rows)} rows, {len(_raw_headers)} columns")
                else:
                    _raw_headers, _raw_rows = read_csv_file(_gt_bytes)
                    _selected_sheet = None
                    st.caption(f"Found {len(_raw_rows)} rows, {len(_raw_headers)} columns")

                # Column mapper
                if _raw_headers:
                    st.markdown("**Column Mapping**")
                    st.caption("Map your file columns to canonical GT columns.")
                    _col_mapping = {}
                    _mapper_cols = st.columns(min(len(_canonical_cols), 4))
                    for _ci, _ccol in enumerate(_canonical_cols):
                        with _mapper_cols[_ci % min(len(_canonical_cols), 4)]:
                            _auto_match = ""
                            for _rh in _raw_headers:
                                if _rh.lower().strip() == _ccol.lower().strip():
                                    _auto_match = _rh
                                    break
                            _sel = st.selectbox(
                                _ccol,
                                ["(skip)"] + _raw_headers,
                                index=(
                                    _raw_headers.index(_auto_match) + 1
                                    if _auto_match else 0
                                ),
                                key=f"_gt_map_{_ccol}",
                            )
                            if _sel != "(skip)":
                                _col_mapping[_ccol] = _sel

                    # Save mapping + data
                    if st.button("Save Ground Truth", key="_gt_save_btn"):
                        if _gt_pid and _col_mapping:
                            from src.analysis.projects import DEFAULT_PROJECTS_DIR
                            _mapped_rows = apply_column_mapping(
                                _raw_headers, _raw_rows, _col_mapping,
                            )
                            _gt_type_short = _gt_type.replace("gt_", "")
                            save_gt_mapping(_gt_pid, {
                                "gt_type": _gt_type_short,
                                "source_file": _gt_fname,
                                "sheet_name": _selected_sheet if _gt_fname.lower().endswith(".xlsx") else None,
                                "column_map": _col_mapping,
                            }, DEFAULT_PROJECTS_DIR)
                            save_gt_data(
                                _gt_pid, _gt_type_short,
                                _mapped_rows, DEFAULT_PROJECTS_DIR,
                            )
                            st.success(
                                f"Saved {len(_mapped_rows)} {_gt_type_short} rows."
                            )
                        elif not _gt_pid:
                            st.warning("Select or create a project first.")
                        else:
                            st.warning("Map at least one column.")

            st.markdown("---")

            # ── 3. Diff results ────────────────────────────────────
            st.markdown("##### Diff Report (vs Ground Truth)")
            if _gt_pid:
                from src.analysis.projects import DEFAULT_PROJECTS_DIR
                _gt_boq_data = load_gt_data(_gt_pid, "boq", DEFAULT_PROJECTS_DIR)
                _gt_sched_data = load_gt_data(_gt_pid, "schedules_doors", DEFAULT_PROJECTS_DIR)
                _gt_qty_data = load_gt_data(_gt_pid, "quantities", DEFAULT_PROJECTS_DIR)

                if _gt_boq_data or _gt_sched_data or _gt_qty_data:
                    _gt_diff = compute_gt_diff(
                        payload, _gt_boq_data, _gt_sched_data, _gt_qty_data,
                    )

                    # Overall match
                    _overall = _gt_diff.get("overall_match_rate")
                    if _overall is not None:
                        st.metric("Overall Match Rate", f"{_overall:.1%}")

                    # Per-category
                    _cats = _gt_diff.get("categories", {})
                    if _cats:
                        _diff_cols = st.columns(len(_cats))
                        for _di, (_cname, _cdata) in enumerate(_cats.items()):
                            with _diff_cols[_di]:
                                st.metric(
                                    f"{_cname.title()} Match",
                                    f"{_cdata.get('match_rate', 0):.1%}",
                                    help=f"Matched {_cdata.get('matched_count', 0)}/{_cdata.get('gt_count', 0)} GT items",
                                )

                        # Mismatches table
                        for _cname, _cdata in _cats.items():
                            _mismatches = _cdata.get("top_mismatches", [])
                            if _mismatches:
                                st.markdown(f"**{_cname.title()} — Top Mismatches**")
                                import pandas as pd
                                _mdf = pd.DataFrame(_mismatches[:10])
                                st.dataframe(_mdf, use_container_width=True, hide_index=True)

                            _missing = _cdata.get("missing_in_ours", [])
                            if _missing:
                                st.markdown(f"**{_cname.title()} — Missing in Our Output** ({len(_missing)} items)")
                                st.caption(", ".join(_missing[:20]))
                else:
                    st.info("No ground truth uploaded yet. Upload data above to see diff results.")
            else:
                st.info("Select a project to view diff results.")

            st.markdown("---")

            # ── 4. Pilot docs ──────────────────────────────────────
            st.markdown("##### Pilot Documents")
            _pilot_meta = {}
            if _gt_pid:
                try:
                    _pilot_meta = load_project(_gt_pid) or {}
                except Exception:
                    pass

            _pd_cols = st.columns(3)
            with _pd_cols[0]:
                st.download_button(
                    "\U0001f4c4 Pilot Agreement",
                    generate_pilot_agreement_docx(_pilot_meta),
                    "pilot_agreement.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="pd_agree",
                    use_container_width=True,
                )
            with _pd_cols[1]:
                st.download_button(
                    "\U0001f512 Data Handling Options",
                    generate_data_handling_docx(_pilot_meta),
                    "data_handling_options.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="pd_data",
                    use_container_width=True,
                )
            with _pd_cols[2]:
                st.download_button(
                    "\u2611\ufe0f Pilot Checklist",
                    generate_pilot_checklist_docx(_pilot_meta),
                    "pilot_checklist.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="pd_check",
                    use_container_width=True,
                )

            st.markdown("---")

            # ── 5. Export Training Pack ────────────────────────────
            st.markdown("##### Export Training Pack")
            st.caption("Bundle inputs, outputs, ground truth, diff, and feedback into a ZIP.")
            if st.button("Build Training Pack", key="_gt_build_tp"):
                if _gt_pid:
                    from src.analysis.projects import DEFAULT_PROJECTS_DIR
                    from src.analysis.feedback import load_feedback

                    _gt_boq_data_tp = load_gt_data(_gt_pid, "boq", DEFAULT_PROJECTS_DIR)
                    _gt_sched_data_tp = load_gt_data(_gt_pid, "schedules_doors", DEFAULT_PROJECTS_DIR)
                    _gt_qty_data_tp = load_gt_data(_gt_pid, "quantities", DEFAULT_PROJECTS_DIR)

                    _tp_diff = None
                    if _gt_boq_data_tp or _gt_sched_data_tp or _gt_qty_data_tp:
                        _tp_diff = compute_gt_diff(
                            payload, _gt_boq_data_tp, _gt_sched_data_tp, _gt_qty_data_tp,
                        )

                    _tp_feedback = []
                    try:
                        _tp_feedback = load_feedback(_gt_pid, DEFAULT_PROJECTS_DIR)
                    except Exception:
                        pass

                    _tp_meta = {}
                    try:
                        _tp_meta = load_project(_gt_pid) or {}
                    except Exception:
                        pass

                    _tp_bytes = build_training_pack(
                        project_id=_gt_pid,
                        run_id=_gt_run_id,
                        payload=payload,
                        gt_diff=_tp_diff,
                        gt_boq=_gt_boq_data_tp or None,
                        gt_schedules=_gt_sched_data_tp or None,
                        gt_quantities=_gt_qty_data_tp or None,
                        feedback_entries=_tp_feedback or None,
                        project_metadata=_tp_meta,
                    )
                    st.session_state["_training_pack_bytes"] = _tp_bytes
                    st.success("Training pack built!")

            if st.session_state.get("_training_pack_bytes"):
                st.download_button(
                    "\U0001f4e6 Download Training Pack (ZIP)",
                    st.session_state["_training_pack_bytes"],
                    f"training_pack_{project_id}_{_gt_run_id}.zip",
                    "application/zip",
                    key="tp_download",
                    use_container_width=True,
                )

        except Exception as _gt_err:
            _handle_tab_error("Ground Truth", _gt_err)

    # ------------------------------------------------------------------
    # TAB 12: Structural Takeoff (Sprint 20A)
    # ------------------------------------------------------------------
    with preview_tabs[12]:
      try:
        st.markdown("#### Structural Takeoff")
        _render_narration_hints("structural", payload)
        _st_data = payload.get("structural_takeoff")
        if _st_data and _st_data.get("mode") != "error":
            _st_mode = _st_data.get("mode", "assumption")
            _st_summary = _st_data.get("summary", {})
            _st_qc = _st_data.get("qc", {})
            _st_quantities = _st_data.get("quantities", [])
            _st_warnings = _st_data.get("warnings", [])
            _st_source = _st_data.get("source_file", "")

            # Mode badge
            _mode_color = "green" if _st_mode == "structural" else "orange"
            st.markdown(
                f"**Mode:** :{_mode_color}[{_st_mode.upper()}]"
                + (f" &nbsp; **Source:** `{_st_source}`" if _st_source else "")
            )

            # Summary metrics cards
            _el_counts = _st_summary.get("element_counts", {})
            _st_m_cols = st.columns(5)
            with _st_m_cols[0]:
                st.metric("Concrete", f"{_st_summary.get('concrete_m3', 0):.1f} m\u00b3")
            with _st_m_cols[1]:
                st.metric("Steel", f"{_st_summary.get('steel_tons', 0):.2f} t")
            with _st_m_cols[2]:
                st.metric("Columns", _el_counts.get("columns", 0))
            with _st_m_cols[3]:
                st.metric("Beams", _el_counts.get("beams", 0))
            with _st_m_cols[4]:
                st.metric("Footings", _el_counts.get("footings", 0))

            # QC confidence + issues
            st.markdown("---")
            st.markdown("##### QC Report")
            _st_conf = _st_qc.get("confidence", 0)
            _st_issues = _st_qc.get("issues", {})
            _qc_cols = st.columns(4)
            with _qc_cols[0]:
                st.metric("Confidence", f"{_st_conf:.0%}" if isinstance(_st_conf, (int, float)) else str(_st_conf))
            with _qc_cols[1]:
                st.metric("Errors", _st_issues.get("errors", 0))
            with _qc_cols[2]:
                st.metric("Warnings", _st_issues.get("warnings", 0))
            with _qc_cols[3]:
                st.metric("Info", _st_issues.get("info", 0))

            _issue_details = _st_issues.get("details", [])
            if _issue_details:
                with st.expander(f"QC Issues ({len(_issue_details)})", expanded=False):
                    for _iss in _issue_details[:20]:
                        _sev = _iss.get("severity", "info")
                        _sev_icon = {"error": "\u274c", "warning": "\u26a0\ufe0f", "info": "\u2139\ufe0f"}.get(_sev, "\u2022")
                        st.markdown(f"{_sev_icon} **{_iss.get('code', '')}** — {_iss.get('message', '')}")
                        if _iss.get("suggestion"):
                            st.caption(f"Suggestion: {_iss['suggestion']}")

            # Assumptions (collapsible)
            _st_assumptions = _st_qc.get("assumptions", {})
            _assum_details = _st_assumptions.get("details", [])
            if _assum_details:
                with st.expander(f"Assumptions Used ({len(_assum_details)})", expanded=False):
                    for _a in _assum_details[:30]:
                        st.markdown(f"\u2022 {_a.get('description', _a.get('message', str(_a)))}")

            # Warnings
            if _st_warnings:
                with st.expander(f"Warnings ({len(_st_warnings)})", expanded=False):
                    for _w in _st_warnings:
                        st.warning(_w)

            # Quantities dataframe
            st.markdown("---")
            st.markdown("##### Quantities Detail")
            if _st_quantities:
                import pandas as pd
                _st_rows = []
                for _eq in _st_quantities:
                    _dims = _eq.get("dimensions_mm", {})
                    _steel = _eq.get("steel_kg", {})
                    _st_rows.append({
                        "ID": _eq.get("element_id", ""),
                        "Type": _eq.get("type", ""),
                        "Label": _eq.get("label", ""),
                        "Count": _eq.get("count", 0),
                        "W (mm)": _dims.get("width", 0),
                        "D (mm)": _dims.get("depth", 0),
                        "L (mm)": _dims.get("length", 0),
                        "Concrete m\u00b3": _eq.get("concrete_m3", 0),
                        "Steel kg": _steel.get("total", 0),
                    })
                _st_df = pd.DataFrame(_st_rows)
                st.dataframe(_st_df, use_container_width=True, hide_index=True)
            else:
                st.info("No element-level quantities available.")

        elif _st_data and _st_data.get("mode") == "error":
            st.warning("Structural takeoff encountered errors:")
            for _w in (_st_data.get("warnings") or []):
                st.error(_w)
        else:
            st.info(
                "No structural takeoff run for this project "
                "(no structural pages detected or feature disabled)."
            )
      except Exception as _tab12_err:
          _handle_tab_error("Structural Takeoff", _tab12_err)

    # ------------------------------------------------------------------
    # TAB 13: Estimating Playbook (Sprint 20C)
    # ------------------------------------------------------------------
    with preview_tabs[13]:
      try:
        st.markdown("#### Estimating Playbook")
        from src.analysis.estimating_playbook import (
            default_playbook,
            validate_playbook,
            merge_playbook,
            diff_playbook,
            summarize_playbook_for_exports,
            compute_playbook_contingency_adjustments,
            RISK_POSTURES,
            ASSUMPTION_POLICIES,
            COMPETITION_LEVELS,
            MATERIAL_TRENDS,
            LABOR_AVAILABILITY,
            LOGISTICS_DIFFICULTY,
            WEATHER_FACTORS,
        )
        from src.analysis.company_playbooks import (
            save_playbook as _save_pb,
            load_playbook as _load_pb,
            list_playbooks as _list_pb,
        )

        # Load existing playbook from session or start with defaults
        if "_ep_playbook" not in st.session_state:
            st.session_state["_ep_playbook"] = default_playbook()
        _ep = st.session_state["_ep_playbook"]

        # ── Load / Save company profile ────────────────────────
        st.markdown("##### Company Playbook")
        _ep_saved = _list_pb()
        _ep_names = [s["company_name"] for s in _ep_saved] if _ep_saved else []
        _ep_load_cols = st.columns([3, 1, 1])
        with _ep_load_cols[0]:
            _ep_selected = st.selectbox(
                "Load company profile",
                ["(new)"] + _ep_names,
                key="_ep_load_select",
            )
        with _ep_load_cols[1]:
            if st.button("Load", key="_ep_load_btn") and _ep_selected != "(new)":
                _loaded = _load_pb(_ep_selected)
                if _loaded:
                    st.session_state["_ep_playbook"] = _loaded
                    st.success(f"Loaded: {_ep_selected}")
                    st.rerun()
        with _ep_load_cols[2]:
            if st.button("Reset Defaults", key="_ep_reset_btn"):
                st.session_state["_ep_playbook"] = default_playbook()
                st.success("Reset to defaults")
                st.rerun()

        st.markdown("---")

        # ── Company defaults form ──────────────────────────────
        _ep_company = _ep.get("company", {})
        with st.expander("Company Defaults", expanded=True):
            _c_name = st.text_input("Company name", value=_ep_company.get("name", ""),
                                     key="_ep_c_name")
            _c_cols = st.columns(3)
            with _c_cols[0]:
                _c_posture = st.selectbox(
                    "Risk posture", list(RISK_POSTURES),
                    index=list(RISK_POSTURES).index(_ep_company.get("risk_posture", "balanced")),
                    key="_ep_c_posture",
                )
            with _c_cols[1]:
                _c_policy = st.selectbox(
                    "Assumption policy", list(ASSUMPTION_POLICIES),
                    index=list(ASSUMPTION_POLICIES).index(
                        _ep_company.get("assumption_policy", "rfi_first")),
                    key="_ep_c_policy",
                )
            with _c_cols[2]:
                pass  # placeholder column
            _c_pct_cols = st.columns(3)
            with _c_pct_cols[0]:
                _c_cont = st.number_input("Default contingency %",
                                           value=float(_ep_company.get("default_contingency_pct", 5.0)),
                                           min_value=0.0, max_value=50.0, step=0.5,
                                           key="_ep_c_cont")
            with _c_pct_cols[1]:
                _c_oh = st.number_input("Default OH %",
                                         value=float(_ep_company.get("default_oh_pct", 10.0)),
                                         min_value=0.0, max_value=50.0, step=0.5,
                                         key="_ep_c_oh")
            with _c_pct_cols[2]:
                _c_profit = st.number_input("Default profit %",
                                             value=float(_ep_company.get("default_profit_pct", 8.0)),
                                             min_value=0.0, max_value=50.0, step=0.5,
                                             key="_ep_c_profit")

        # ── Project overrides form ─────────────────────────────
        _ep_project = _ep.get("project", {})
        with st.expander("Project Overrides", expanded=False):
            _p_cols = st.columns(2)
            with _p_cols[0]:
                _p_type = st.text_input("Project type", value=_ep_project.get("project_type", ""),
                                         key="_ep_p_type")
                _p_client = st.text_input("Client name", value=_ep_project.get("client_name", ""),
                                           key="_ep_p_client")
                _p_contract = st.text_input("Contract type",
                                             value=_ep_project.get("contract_type", ""),
                                             key="_ep_p_contract")
            with _p_cols[1]:
                _p_bid_date = st.text_input("Bid due date",
                                             value=_ep_project.get("bid_due_date", "") or "",
                                             placeholder="YYYY-MM-DD",
                                             key="_ep_p_bid_date")
                _p_comp = st.selectbox(
                    "Competition intensity", list(COMPETITION_LEVELS),
                    index=list(COMPETITION_LEVELS).index(
                        _ep_project.get("competition_intensity", "med")),
                    key="_ep_p_comp",
                )
                _p_cont_over = st.number_input(
                    "Contingency override % (blank = use company default)",
                    value=float(_ep_project.get("contingency_override_pct") or 0),
                    min_value=0.0, max_value=50.0, step=0.5,
                    key="_ep_p_cont_over",
                )
            _p_flag_cols = st.columns(3)
            with _p_flag_cols[0]:
                _p_must_win = st.checkbox("Must-win bid",
                                           value=_ep_project.get("must_win", False),
                                           key="_ep_p_must_win")
            with _p_flag_cols[1]:
                _p_rel_bid = st.checkbox("Relationship bid",
                                          value=_ep_project.get("relationship_bid", False),
                                          key="_ep_p_rel_bid")
            # Location
            _ep_loc = _ep_project.get("location", {})
            _loc_cols = st.columns(3)
            with _loc_cols[0]:
                _p_country = st.text_input("Country", value=_ep_loc.get("country", ""),
                                            key="_ep_p_country")
            with _loc_cols[1]:
                _p_state = st.text_input("State", value=_ep_loc.get("state", ""),
                                          key="_ep_p_state")
            with _loc_cols[2]:
                _p_city = st.text_input("City", value=_ep_loc.get("city", ""),
                                         key="_ep_p_city")

        # ── Market snapshot form ───────────────────────────────
        _ep_market = _ep.get("market_snapshot", {})
        with st.expander("Market Snapshot", expanded=False):
            _m_cols = st.columns(4)
            with _m_cols[0]:
                _m_mat = st.selectbox("Material trend", list(MATERIAL_TRENDS),
                                       index=list(MATERIAL_TRENDS).index(
                                           _ep_market.get("material_trend", "stable")),
                                       key="_ep_m_mat")
            with _m_cols[1]:
                _m_labor = st.selectbox("Labor availability", list(LABOR_AVAILABILITY),
                                         index=list(LABOR_AVAILABILITY).index(
                                             _ep_market.get("labor_availability", "normal")),
                                         key="_ep_m_labor")
            with _m_cols[2]:
                _m_log = st.selectbox("Logistics difficulty", list(LOGISTICS_DIFFICULTY),
                                       index=list(LOGISTICS_DIFFICULTY).index(
                                           _ep_market.get("logistics_difficulty", "easy")),
                                       key="_ep_m_log")
            with _m_cols[3]:
                _m_weather = st.selectbox("Weather factor", list(WEATHER_FACTORS),
                                           index=list(WEATHER_FACTORS).index(
                                               _ep_market.get("weather_factor", "normal")),
                                           key="_ep_m_weather")
            _m_idx_cols = st.columns(4)
            with _m_idx_cols[0]:
                _m_steel = st.number_input("Steel cost index",
                                            value=float(_ep_market.get("steel_cost_index") or 0),
                                            step=0.1, key="_ep_m_steel")
            with _m_idx_cols[1]:
                _m_cement = st.number_input("Cement cost index",
                                             value=float(_ep_market.get("cement_cost_index") or 0),
                                             step=0.1, key="_ep_m_cement")
            with _m_idx_cols[2]:
                _m_labor_r = st.number_input("Labor rate factor",
                                              value=float(_ep_market.get("labor_rate_factor") or 0),
                                              step=0.1, key="_ep_m_labor_r")
            with _m_idx_cols[3]:
                _m_freight = st.number_input("Freight factor",
                                              value=float(_ep_market.get("freight_factor") or 0),
                                              step=0.1, key="_ep_m_freight")
            _m_notes = st.text_area("Market notes", value=_ep_market.get("notes", ""),
                                     height=68, key="_ep_m_notes")

        # ── Save button ────────────────────────────────────────
        st.markdown("---")
        _save_cols = st.columns([2, 1])
        with _save_cols[0]:
            if st.button("Save Playbook", key="_ep_save_btn", type="primary"):
                _updated_pb = {
                    "company": {
                        "name": _c_name,
                        "risk_posture": _c_posture,
                        "default_contingency_pct": _c_cont,
                        "default_oh_pct": _c_oh,
                        "default_profit_pct": _c_profit,
                        "assumption_policy": _c_policy,
                        "trade_scope_defaults": _ep_company.get("trade_scope_defaults", {}),
                        "measurement_prefs": _ep_company.get("measurement_prefs", {}),
                        "output_prefs": _ep_company.get("output_prefs", {}),
                    },
                    "project": {
                        "project_type": _p_type,
                        "location": {"country": _p_country, "state": _p_state, "city": _p_city},
                        "client_name": _p_client,
                        "contract_type": _p_contract,
                        "bid_due_date": _p_bid_date or None,
                        "must_win": _p_must_win,
                        "relationship_bid": _p_rel_bid,
                        "competition_intensity": _p_comp,
                        "contingency_override_pct": _p_cont_over if _p_cont_over > 0 else None,
                    },
                    "market_snapshot": {
                        "material_trend": _m_mat,
                        "labor_availability": _m_labor,
                        "logistics_difficulty": _m_log,
                        "weather_factor": _m_weather,
                        "steel_cost_index": _m_steel if _m_steel > 0 else None,
                        "cement_cost_index": _m_cement if _m_cement > 0 else None,
                        "labor_rate_factor": _m_labor_r if _m_labor_r > 0 else None,
                        "freight_factor": _m_freight if _m_freight > 0 else None,
                        "notes": _m_notes,
                    },
                }
                _is_valid, _pb_warnings = validate_playbook(_updated_pb)
                if _is_valid:
                    st.session_state["_ep_playbook"] = _updated_pb
                    if _c_name.strip():
                        _save_pb(_c_name, _updated_pb)
                    st.success("Playbook saved.")
                    for _w in _pb_warnings:
                        st.warning(_w)
                else:
                    st.error("Invalid playbook.")
                    for _w in _pb_warnings:
                        st.error(_w)

        # ── Changes from defaults panel ────────────────────────
        _pb_current = st.session_state.get("_ep_playbook", {})
        _pb_base = default_playbook()
        _pb_diffs = diff_playbook(_pb_base, _pb_current)
        if _pb_diffs:
            with st.expander(f"Changes from defaults ({len(_pb_diffs)})", expanded=False):
                for _d in _pb_diffs:
                    st.markdown(
                        f"\u2022 **{_d['section']}.{_d['field']}**: "
                        f"`{_d['from']}` \u2192 `{_d['to']}`"
                    )

        # ── Contingency recommendation preview ─────────────────
        _adj = compute_playbook_contingency_adjustments(_pb_current)
        st.markdown("---")
        st.markdown("##### Contingency Recommendation")
        _adj_cols = st.columns(4)
        with _adj_cols[0]:
            st.metric("Base", f"{_adj['base_pct']}%")
        with _adj_cols[1]:
            st.metric("Posture Adj", f"{_adj['posture_adj_pct']:+.1f}%")
        with _adj_cols[2]:
            st.metric("Market Adj", f"{_adj['market_adj_pct']:+.1f}%")
        with _adj_cols[3]:
            st.metric("Recommended", f"{_adj['recommended_pct']}%")
        if _adj.get("override_pct") is not None:
            st.info(f"Project override active: {_adj['override_pct']}%")
        with st.expander("Basis of recommendation"):
            for _b in _adj.get("basis", []):
                st.caption(f"\u2022 {_b}")

      except Exception as _tab13_err:
          _handle_tab_error("Estimating Playbook", _tab13_err)

    # =========================================================================
    # ACTION BUTTONS
    # =========================================================================
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Full Analysis Results →", type="primary", use_container_width=True):
            st.query_params["project_id"] = project_id
            st.rerun()
    with col2:
        st.download_button(
            "📥 Download Full Report (JSON)",
            json.dumps(payload, indent=2, default=str),
            f"report_{project_id}.json",
            "application/json",
            use_container_width=True
        )

    # Sprint 20B: Demo Snapshot Export (compact investor-friendly ZIP)
    if _yc_active:
        st.markdown("---")
        st.markdown("##### \U0001f3ac Demo Snapshot Export")
        st.caption("Compact bundle with only investor-friendly outputs (no debug files).")
        if st.button("Build Demo Snapshot", key="_demo_snapshot_btn"):
            try:
                import zipfile as _ds_zf
                _ds_buf = io.BytesIO()
                with _ds_zf.ZipFile(_ds_buf, "w", _ds_zf.ZIP_DEFLATED) as _ds_zip:
                    # manifest.json
                    _ds_manifest = {
                        "project_id": project_id,
                        "timestamp": payload.get("timestamp", ""),
                        "decision": payload.get("decision", ""),
                        "readiness_score": payload.get("readiness_score", 0),
                        "pages_total": (payload.get("drawing_overview") or {}).get("pages_total", 0),
                        "blockers_count": len(payload.get("blockers", [])),
                        "rfis_count": len(payload.get("rfis", [])),
                    }
                    _ds_zip.writestr("manifest.json", json.dumps(_ds_manifest, indent=2, default=str))
                    # bid_summary.md (if in session)
                    _ds_bid_md = st.session_state.get("_bid_summary_md", "")
                    if _ds_bid_md:
                        _ds_zip.writestr("bid_summary.md", _ds_bid_md)
                    # approved RFIs CSV
                    _ds_rfis = [r for r in payload.get("rfis", []) if r.get("status") == "approved"]
                    if _ds_rfis:
                        import csv as _ds_csv
                        _ds_rfi_buf = io.StringIO()
                        _ds_rfi_w = _ds_csv.DictWriter(_ds_rfi_buf, fieldnames=["id", "title", "trade", "severity", "status"])
                        _ds_rfi_w.writeheader()
                        for _r in _ds_rfis:
                            _ds_rfi_w.writerow({
                                "id": _r.get("id", ""), "title": _r.get("title", ""),
                                "trade": _r.get("trade", ""), "severity": _r.get("severity", ""),
                                "status": _r.get("status", ""),
                            })
                        _ds_zip.writestr("approved_rfis.csv", _ds_rfi_buf.getvalue())
                    # exclusions/clarifications
                    _ds_excl = payload.get("guardrail_warnings", [])
                    if _ds_excl:
                        _ds_zip.writestr("exclusions_clarifications.txt",
                                         "\n".join(str(w) for w in _ds_excl[:50]))
                    # structural files (if present)
                    _ds_st = payload.get("structural_takeoff")
                    if _ds_st and _ds_st.get("mode") not in (None, "error"):
                        _ds_zip.writestr("structural_summary.json",
                                         json.dumps(_ds_st.get("summary", {}), indent=2, default=str))
                        _ds_zip.writestr("structural_qc.json",
                                         json.dumps(_ds_st.get("qc", {}), indent=2, default=str))
                        if _ds_st.get("quantities"):
                            _ds_qty_rows = []
                            for _eq in _ds_st["quantities"]:
                                _dims = _eq.get("dimensions_mm", {})
                                _steel = _eq.get("steel_kg", {})
                                _ds_qty_rows.append(
                                    f"{_eq.get('element_id','')},{_eq.get('type','')},{_eq.get('label','')},"
                                    f"{_eq.get('count',0)},{_eq.get('concrete_m3',0)},{_steel.get('total',0)}"
                                )
                            _ds_zip.writestr("structural_quantities.csv",
                                             "element_id,type,label,count,concrete_m3,steel_kg_total\n" +
                                             "\n".join(_ds_qty_rows))
                _ds_buf.seek(0)
                st.session_state["_demo_snapshot_bytes"] = _ds_buf.getvalue()
                st.success("Demo snapshot built!")
            except Exception as _ds_err:
                st.error(f"Could not build demo snapshot: {_ds_err}")
        if st.session_state.get("_demo_snapshot_bytes"):
            st.download_button(
                "\U0001f4e6 Download Demo Snapshot (ZIP)",
                st.session_state["_demo_snapshot_bytes"],
                f"demo_snapshot_{project_id}.zip",
                "application/zip",
                key="_demo_snapshot_dl",
                use_container_width=True,
            )

    # Show project info at bottom
    with st.expander("\U0001f4c1 Output Files", expanded=False):
        st.write(f"**Project ID:** `{project_id}`")
        st.write(f"**Output directory:** `{result.output_dir}`")
        for f in result.files_generated:
            fpath = Path(f)
            exists = fpath.exists()
            size = fpath.stat().st_size if exists else 0
            icon = "✅" if exists else "❌"
            st.caption(f"{icon} {fpath.name} ({size} bytes)")


# =============================================================================
# ANALYSIS RUNNER WITH PROGRESS
# =============================================================================

def _run_analysis_with_progress(uploaded_files: List[Any], dev_mode: bool = False, run_mode: str = "demo_fast"):
    """
    Run analysis pipeline with Streamlit progress UI.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        dev_mode: If True, show stack traces on error
        run_mode: Sprint 20G — "demo_fast", "standard_review", or "full_audit"
    """
    # Project setup
    project_id = generate_project_id()
    project_root = Path(__file__).parent.parent
    uploads_dir = project_root / "uploads"
    output_dir = project_root / "out" / project_id

    # Stage definitions for progress UI
    stages = [
        ("load", "Loading PDFs", "Scanning and indexing uploaded files..."),
        ("index", "Indexing Pages", "Classifying every page by type and discipline..."),
        ("select", "Selecting Pages", "Prioritizing pages within OCR budget..."),
        ("extract", "Extracting Text", "OCR + specialized extraction..."),
        ("graph", "Building Graph", "Analyzing plan set structure..."),
        ("reason", "Analyzing Scope", "Detecting blockers and coverage gaps..."),
        ("rfi", "Generating RFIs", "Creating evidence-backed RFI recommendations..."),
        ("export", "Saving Results", "Writing output files..."),
    ]

    # Create progress UI
    st.markdown("---")

    # Track result outside the status block so we can render results after it
    analysis_result = None
    analysis_error = None

    with st.status("Analyzing drawings...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        stage_text = st.empty()
        detail_text = st.empty()
        eta_metric = st.empty()

        current_stage_idx = 0
        stage_progress = {}

        def progress_callback(stage_id: str, message: str, progress: float):
            """Update progress UI from analysis pipeline."""
            nonlocal current_stage_idx
            stage_progress[stage_id] = progress

            for i, (sid, _, _) in enumerate(stages):
                if sid == stage_id:
                    current_stage_idx = i
                    break

            completed_stages = sum(1 for sid, _, _ in stages[:current_stage_idx] if stage_progress.get(sid, 0) >= 1.0)
            current_pct = stage_progress.get(stage_id, 0)
            overall = (completed_stages + current_pct) / len(stages)
            progress_bar.progress(min(overall, 1.0))

            stage_name = stages[current_stage_idx][1] if current_stage_idx < len(stages) else "Complete"
            stage_text.markdown(f"**Stage {current_stage_idx + 1}/{len(stages)}:** {stage_name}")
            detail_text.caption(message)

            # Rolling ETA metric from pipeline's per-page messages
            if "s/page" in message:
                import re as _re
                rate_m = _re.search(r'([\d.]+)s/page', message)
                eta_m = _re.search(r'est\.\s*(\d+)s\s*remaining', message)
                if rate_m and eta_m:
                    secs = int(eta_m.group(1))
                    eta_metric.markdown(
                        f"⏱️ **{rate_m.group(1)}s/page** · "
                        f"~{secs // 60}m {secs % 60}s remaining"
                    )

        stage_text.markdown(f"**Stage 1/{len(stages)}:** {stages[0][1]}")
        detail_text.caption(stages[0][2])

        try:
            detail_text.caption("Saving uploaded files...")
            saved_files = save_uploaded_files(uploaded_files, project_id, uploads_dir)

            # Sprint 16: Try async job queue if available
            _use_async = os.environ.get("XBOQ_ASYNC_JOBS", "").lower() == "true"
            if _use_async:
                try:
                    from src.jobs import LocalThreadQueue
                    if "_xboq_job_queue" not in st.session_state:
                        st.session_state["_xboq_job_queue"] = LocalThreadQueue(max_workers=1)
                    _queue = st.session_state["_xboq_job_queue"]
                    _job_id = _queue.submit(
                        run_analysis_pipeline,
                        input_files=saved_files,
                        project_id=project_id,
                        output_dir=output_dir,
                        run_mode=run_mode,
                    )
                    st.session_state["_xboq_active_job"] = _job_id
                    st.session_state["_xboq_active_job_project"] = project_id
                    status.update(label="Job submitted — processing in background...", state="complete")
                    time.sleep(0.5)
                    st.rerun()
                    return
                except Exception:
                    pass  # Fall through to synchronous path

            result = run_analysis_pipeline(
                input_files=saved_files,
                project_id=project_id,
                output_dir=output_dir,
                progress_callback=progress_callback,
                run_mode=run_mode,
            )

            progress_bar.progress(1.0)

            if result.success:
                status.update(label="Analysis complete!", state="complete", expanded=False)
                analysis_result = result
            else:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(f"Analysis failed: {result.error_message}")

                if dev_mode and result.stack_trace:
                    with st.expander("Stack Trace (Dev Mode)", expanded=True):
                        st.code(result.stack_trace, language="python")
                elif result.stack_trace:
                    with st.expander("Technical Details"):
                        st.code(result.stack_trace, language="python")

                for stage in result.stages:
                    if stage.status == "failed":
                        st.warning(f"Failed at stage: **{stage.name}**")
                        if stage.error:
                            st.caption(stage.error)
                        break

        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            analysis_error = e
            st.error(f"Unexpected error: {str(e)}")

            if dev_mode:
                with st.expander("Stack Trace (Dev Mode)", expanded=True):
                    st.code(traceback.format_exc(), language="python")
            else:
                with st.expander("Technical Details"):
                    st.code(traceback.format_exc(), language="python")

    # ── Render results OUTSIDE the status block so they're always visible ──
    if analysis_result is not None:
        _render_analysis_results_preview(analysis_result, project_id, uploaded_files)
    elif analysis_error is not None:
        if st.button("Try Again", type="secondary"):
            st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def _render_login_page():
    """Sprint 16: Render tenant login form when auth is enabled."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <h1 style="font-size: 2.5rem; color: #1a365d;">xBOQ Bid Engineer</h1>
        <p style="color: #718096;">Sign in to your tenant workspace</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.form("xboq_login"):
            tenant_id = st.text_input("Tenant ID")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if not tenant_id.strip() or not password:
                    st.error("Enter both tenant ID and password.")
                else:
                    try:
                        from src.auth import SimpleAuth
                        auth = SimpleAuth()
                        tenant = auth.authenticate(tenant_id.strip(), password)
                        if tenant:
                            st.session_state["_xboq_tenant_id"] = tenant_id.strip()
                            st.session_state["_xboq_tenant_name"] = tenant["name"]
                            st.rerun()
                        else:
                            st.error("Invalid tenant ID or password.")
                    except Exception as e:
                        st.error(f"Authentication error: {e}")


def _render_job_progress(job):
    """Sprint 16/17: Render async job progress view with stage breakdown and ETA."""
    _STAGE_ICONS = {
        "load": "\U0001f4e5", "index": "\U0001f5c2\ufe0f", "select": "\U0001f3af",
        "extract": "\U0001f4dd", "graph": "\U0001f517", "reason": "\U0001f9e0",
        "rfi": "\U0001f4cb", "export": "\U0001f4be",
    }
    _ALL_STAGES = ["load", "index", "select", "extract", "graph", "reason", "rfi", "export"]

    st.markdown("---")

    # Stage name with icon
    _stage_id = (job.stage or "").split(":")[0].lower().strip()
    _icon = _STAGE_ICONS.get(_stage_id, "\u23f3")
    st.markdown(f"### {_icon} Processing: {job.stage or 'Starting...'}")

    # Progress bar + percentage
    _pct = min(max(job.progress, 0.0), 1.0)
    st.progress(_pct)
    st.caption(f"{_pct:.0%} complete")

    if job.message:
        st.info(job.message)

    # Elapsed + ETA
    if job.started_at:
        try:
            started = datetime.fromisoformat(job.started_at)
            elapsed = (datetime.now() - started).total_seconds()
            mins, secs = divmod(int(elapsed), 60)

            _eta_str = ""
            if _pct > 0.05:
                _remaining = (elapsed / _pct) * (1.0 - _pct)
                _eta_m, _eta_s = divmod(int(_remaining), 60)
                _eta_str = f" | Est. remaining: {_eta_m}m {_eta_s}s"

            st.caption(f"Elapsed: {mins}m {secs}s{_eta_str}")

            # Stall warning
            if elapsed > 60 and _pct < 0.1:
                st.warning("Still working... likely OCR-heavy pages.")
        except Exception:
            pass

    # Visual stage pipeline row
    _stage_cols = st.columns(len(_ALL_STAGES))
    _current_idx = _ALL_STAGES.index(_stage_id) if _stage_id in _ALL_STAGES else -1
    for _si, _sname in enumerate(_ALL_STAGES):
        with _stage_cols[_si]:
            _s_icon = _STAGE_ICONS.get(_sname, "")
            if _si == _current_idx:
                st.markdown(f"**{_s_icon}**")
            elif _si < _current_idx:
                st.markdown(f"~~{_s_icon}~~")
            else:
                st.markdown(f"{_s_icon}")
            st.caption(_sname)

    st.caption("Analysis is running in the background. This page will auto-refresh.")


def main():
    project_id = st.query_params.get("project_id", "")

    # Check for dev mode (show stack traces)
    dev_mode = st.query_params.get("dev", "") == "1"

    # ── Sprint 16: Auth gate (opt-in via XBOQ_AUTH_ENABLED=true) ─────
    _auth_enabled = os.environ.get("XBOQ_AUTH_ENABLED", "").lower() == "true"

    if _auth_enabled:
        if "_xboq_tenant_id" not in st.session_state:
            _render_login_page()
            return

    # ── Sprint 16: Job status check ──────────────────────────────────
    if "_xboq_active_job" in st.session_state:
        try:
            _queue = st.session_state.get("_xboq_job_queue")
            if _queue:
                from src.jobs import JobStatus as _JobStatus
                _job = _queue.get_status(st.session_state["_xboq_active_job"])
                if _job and _job.status == _JobStatus.RUNNING:
                    _render_job_progress(_job)
                    time.sleep(2)
                    st.rerun()
                    return
                elif _job and _job.status == _JobStatus.COMPLETED:
                    _completed_result = _job.result
                    _completed_project_id = st.session_state.pop("_xboq_active_job_project", "")
                    del st.session_state["_xboq_active_job"]
                    if _completed_result and hasattr(_completed_result, "success") and _completed_result.success:
                        st.success("Analysis complete!")
                        _render_analysis_results_preview(
                            _completed_result, _completed_project_id, [])
                        return
                elif _job and _job.status == _JobStatus.FAILED:
                    del st.session_state["_xboq_active_job"]
                    st.error(f"Pipeline failed: {_job.error[:500]}")
                elif _job and _job.status.value == "cancelled":
                    del st.session_state["_xboq_active_job"]
                    st.warning("Job was cancelled.")
                else:
                    # Job not found or unknown state — clean up
                    del st.session_state["_xboq_active_job"]
        except Exception:
            # Graceful fallback — clear stale job state
            st.session_state.pop("_xboq_active_job", None)

    # ── Sprint 14: Project selector sidebar ──────────────────────────
    try:
        from src.analysis.projects import (
            list_projects, create_project, list_runs, project_dir as _project_dir,
        )

        with st.sidebar:
            st.markdown("### 📁 Projects")
            existing_projects = list_projects()
            project_options = ["— None —"] + [
                f"{p.get('name', p.get('project_id', '?'))} ({p.get('project_id', '')[:20]})"
                for p in existing_projects
            ]
            selected_idx = st.selectbox(
                "Active project",
                range(len(project_options)),
                format_func=lambda i: project_options[i],
                key="_project_selector",
            )
            if selected_idx and selected_idx > 0:
                _active_meta = existing_projects[selected_idx - 1]
                st.session_state["_active_project_id"] = _active_meta.get("project_id", "")
                st.session_state["_active_project_name"] = _active_meta.get("name", "")
                st.caption(f"Owner: {_active_meta.get('owner', '—')}")
                st.caption(f"Bid date: {_active_meta.get('bid_date', '—')}")

                # Run history
                _runs = list_runs(st.session_state["_active_project_id"])
                if _runs:
                    with st.expander(f"Run history ({len(_runs)})"):
                        for _run in _runs[:10]:
                            _ts = _run.get("timestamp", "")[:16]
                            _score = _run.get("readiness_score", "?")
                            _dec = _run.get("decision", "?")
                            st.caption(f"{_ts} — {_score}/100 {_dec}")
            else:
                st.session_state["_active_project_id"] = ""
                st.session_state["_active_project_name"] = ""

            # New project expander
            with st.expander("➕ New Project"):
                _np_name = st.text_input("Project name", key="_np_name")
                _np_owner = st.text_input("Owner / Client", key="_np_owner")
                _np_bid = st.text_input("Bid date", key="_np_bid", placeholder="YYYY-MM-DD")
                _np_notes = st.text_area("Notes", key="_np_notes", height=68)
                if st.button("Create Project", key="_create_project_btn"):
                    if _np_name.strip():
                        _new_meta = create_project(
                            name=_np_name,
                            owner=_np_owner,
                            bid_date=_np_bid,
                            notes=_np_notes,
                        )
                        st.session_state["_active_project_id"] = _new_meta["project_id"]
                        st.session_state["_active_project_name"] = _new_meta["name"]
                        st.success(f"Created: {_new_meta['name']}")
                        st.rerun()
                    else:
                        st.warning("Enter a project name.")
    except Exception:
        pass  # Graceful fallback if projects module unavailable

    # ── Sprint 20: Pilot Mode toggle + intake ────────────────────────
    try:
        with st.sidebar:
            st.markdown("---")
            _pilot_mode = st.toggle("Pilot Mode", key="_xboq_pilot_mode",
                                    help="Enable pilot conversion features")
            if _pilot_mode:
                with st.expander("Pilot Intake", expanded=True):
                    with st.form("pilot_intake_form"):
                        _pi_company = st.text_input("Company name", key="_pi_company")
                        _pi_owner = st.text_input("Owner / Client", key="_pi_owner")
                        _pi_bid_date = st.text_input("Bid due date", key="_pi_bid_date",
                                                      placeholder="YYYY-MM-DD")
                        _pi_trades = st.multiselect(
                            "Trades in scope",
                            ["structural", "architectural", "mep", "electrical",
                             "plumbing", "finishes", "civil", "landscaping"],
                            key="_pi_trades",
                        )
                        _pi_prefs = st.multiselect(
                            "Output preferences",
                            ["Full BOQ", "RFI Log", "Scope Check", "Quantities",
                             "Bid Summary PDF", "Training Pack"],
                            default=["Full BOQ", "RFI Log", "Scope Check"],
                            key="_pi_prefs",
                        )
                        _pi_submit = st.form_submit_button("Save Pilot Config")
                        if _pi_submit and _pi_company.strip():
                            from src.analysis.projects import update_project, create_project as _cp
                            _active_pid = st.session_state.get("_active_project_id", "")
                            if _active_pid:
                                update_project(_active_pid, {
                                    "company_name": _pi_company,
                                    "owner": _pi_owner,
                                    "bid_date": _pi_bid_date,
                                    "trades_in_scope": _pi_trades,
                                    "output_preferences": {"selected": _pi_prefs},
                                    "pilot_mode": True,
                                })
                                st.success("Pilot config saved.")
                            else:
                                _new = _cp(
                                    name=_pi_company,
                                    owner=_pi_owner,
                                    company_name=_pi_company,
                                    bid_date=_pi_bid_date,
                                    trades_in_scope=_pi_trades,
                                    output_preferences={"selected": _pi_prefs},
                                    pilot_mode=True,
                                )
                                st.session_state["_active_project_id"] = _new["project_id"]
                                st.session_state["_active_project_name"] = _new["name"]
                                st.success(f"Pilot project created: {_new['name']}")
                                st.rerun()
    except Exception:
        pass

    # ── Sprint 17: Demo Projects sidebar (shown when DEMO_MODE=true) ─
    try:
        from src.demo.demo_config import is_demo_mode, DEMO_PROJECTS
        from src.demo.demo_assets import resolve_demo_cache

        if is_demo_mode():
            with st.sidebar:
                st.markdown("---")
                st.markdown("### Demo Projects")
                st.caption("Pre-loaded demo datasets for recording")
                for _dp in DEMO_PROJECTS:
                    _dp_id = _dp["project_id"]
                    _dp_cache = resolve_demo_cache(_dp_id)
                    _dp_col1, _dp_col2 = st.columns([3, 1])
                    with _dp_col1:
                        st.markdown(f"**{_dp['name']}**")
                        st.caption(_dp["description"])
                    with _dp_col2:
                        if st.button(
                            "View" if _dp_cache else "N/A",
                            key=f"demo_btn_{_dp_id}",
                            disabled=not _dp_cache,
                            use_container_width=True,
                        ):
                            st.query_params["project_id"] = _dp_id
                            st.rerun()
    except Exception:
        pass

    # ── Sprint 18 + 20B: Demo safety rails + YC Demo Mode ───────────
    try:
        from src.demo.demo_config import is_demo_mode as _sr_dm, DEMO_FREEZE_DEFAULTS
        if _sr_dm():
            with st.sidebar:
                st.markdown("---")
                st.markdown("### Demo Controls")

                # Sprint 20B: YC Demo Mode toggle
                st.toggle("YC Demo Mode", key="_xboq_yc_demo",
                          help="Investor-friendly view: hide debug panels, cleaner labels")

                if st.checkbox("Freeze Demo UI", key="_xboq_freeze_ui",
                               help="Lock filters to safe defaults"):
                    for _fk, _fv in DEMO_FREEZE_DEFAULTS.items():
                        st.session_state[f"_xboq_{_fk}"] = _fv

                st.checkbox("Show watermark on PDFs", key="_xboq_demo_watermark")

                if st.button("Reset Demo State", key="_xboq_reset_demo",
                             type="secondary"):
                    _keep = {"_xboq_tenant_id", "_xboq_job_queue"}
                    _keys_to_clear = [k for k in list(st.session_state.keys())
                                      if k.startswith("_xboq") and k not in _keep]
                    for _ck in _keys_to_clear:
                        del st.session_state[_ck]
                    st.query_params.clear()
                    st.rerun()
    except Exception:
        pass
    # ─────────────────────────────────────────────────────────────────

    if not project_id:
        # ── Hero landing page ──
        st.markdown("""
        <div class="hero-section">
            <h1>xBOQ</h1>
            <div class="hero-tagline">
                Upload tender drawings. Get a scope risk report in seconds.<br>
                Know what's missing before you bid.
            </div>
            <div class="hero-steps">
                <div class="hero-step">
                    <div class="hero-step-num">1</div>
                    <div class="hero-step-label">Upload drawing PDFs</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">2</div>
                    <div class="hero-step-label">xBOQ reads every page</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">3</div>
                    <div class="hero-step-label">Get bid risk report + RFIs</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop your tender drawing PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            file_summary = ", ".join([f.name for f in uploaded_files[:3]])
            if len(uploaded_files) > 3:
                file_summary += f" +{len(uploaded_files) - 3} more"
            st.caption(f"Ready: {file_summary}")

        # Sprint 20G: Run Mode selector
        _run_mode_options = {
            "Demo Fast (80 pages)": "demo_fast",
            "Standard Review (220 pages)": "standard_review",
            "Full Audit (All pages)": "full_audit",
        }
        _col_mode1, _col_mode2, _col_mode3 = st.columns([1, 2, 1])
        with _col_mode2:
            _selected_mode_label = st.radio(
                "Run Mode",
                list(_run_mode_options.keys()),
                index=0,
                horizontal=True,
                help="Controls how many pages are deep-processed.",
            )
            _selected_run_mode = _run_mode_options[_selected_mode_label]
            if _selected_run_mode == "full_audit":
                st.caption("May take longer; recommended async.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button(
                "Analyze Drawings",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_files,
            )

        if uploaded_files and analyze_btn:
            _run_analysis_with_progress(uploaded_files, dev_mode, run_mode=_selected_run_mode)

        st.markdown("---")
        col_l, col_r = st.columns([3, 1])
        with col_r:
            if st.button("View Demo →", type="secondary", use_container_width=True):
                st.query_params["project_id"] = "pwd_garage"
                st.rerun()

    else:
        # Results view
        results = load_demo_results(project_id)

        if not results["loaded"]:
            st.error(f"No results found for project: {project_id}")

            # Show what we looked for (debug info)
            base_path = Path(__file__).parent.parent / "out" / project_id
            with st.expander("Debug: Expected output location"):
                st.code(f"Output directory: {base_path}")
                st.write(f"Directory exists: {base_path.exists()}")
                if base_path.exists():
                    st.write("Files found:")
                    for f in base_path.iterdir():
                        st.caption(f"  - {f.name}")
                else:
                    st.warning("Output directory does not exist")

            if st.button("← Back to upload"):
                st.query_params.clear()
                st.rerun()
            return

        # Nav bar
        col_back, col_spacer, col_export = st.columns([1, 2, 1])
        with col_back:
            if st.button("← New analysis", type="secondary", use_container_width=True):
                st.query_params.clear()
                st.rerun()

        # Check if we have direct analysis.json (new format) or need to build from legacy
        analysis_payload = results.get("analysis")

        # ===================
        # AT-A-GLANCE DASHBOARD
        # ===================
        if analysis_payload:
            demo = build_demo_analysis(analysis_payload, project_id)
            render_at_a_glance_dashboard(demo)

            # Store OCR cache in session state for search
            st.session_state["_xboq_pdf_path"] = analysis_payload.get("primary_pdf_path")
            st.session_state["_xboq_ocr_cache"] = analysis_payload.get("ocr_text_cache", {})

            # Sprint 18: YC Summary Card (cached-results, demo mode only)
            try:
                from src.demo.demo_config import is_demo_mode as _s18c_dm
                if _s18c_dm():
                    from src.demo.summary_card import build_summary_card as _bsc2
                    _sc2_name = st.session_state.get("_active_project_name", "")
                    _sc2 = _bsc2(analysis_payload, project_name=_sc2_name, cache_used=True)
                    with st.container():
                        st.markdown("#### YC Demo Summary")
                        _sc2_c1, _sc2_c2, _sc2_c3 = st.columns(3)
                        with _sc2_c1:
                            st.metric("Total Pages", _sc2["total_pages"])
                            st.metric("Deep Processed", _sc2["deep_pages"])
                            st.metric("OCR Pages", _sc2["ocr_pages"])
                            if _sc2.get("skipped_pages"):
                                st.caption(f"Skipped: {_sc2['skipped_pages']}")
                            st.caption(f"Cache: {_sc2['cache_time_saved']}")
                        with _sc2_c2:
                            st.metric("QA Score", f"{_sc2['qa_score']}/100")
                            st.caption(f"Decision: {_sc2['decision']}")
                            for _sc2_act in _sc2["top_actions"][:2]:
                                st.caption(f"  - {_sc2_act}")
                        with _sc2_c3:
                            st.metric("Approved RFIs", _sc2["approved_rfis"])
                            st.metric("Quantities", _sc2["accepted_quantities"])
                            st.metric("Assumptions", _sc2["accepted_assumptions"])
                            _sc2_pack = "Ready" if _sc2["submission_pack_ready"] else "Not yet"
                            st.caption(f"Pack: {_sc2_pack}")
            except Exception:
                pass

            # Sprint 17: Highlights panel (cached-results path)
            try:
                from src.analysis.highlights import build_highlights
                _hl2 = build_highlights(analysis_payload)
                if _hl2:
                    st.markdown("---")
                    st.markdown("### Highlights")
                    _hl2_cols = st.columns(min(len(_hl2), 4))
                    for _hl2_idx, _hl2_item in enumerate(_hl2):
                        with _hl2_cols[_hl2_idx % len(_hl2_cols)]:
                            _hl2_c = {"good": "green", "warn": "orange", "bad": "red"}.get(
                                _hl2_item.get("severity", "warn"), "gray")
                            st.markdown(
                                f"**{_hl2_item['icon']} {_hl2_item['label']}**\n\n"
                                f":{_hl2_c}[{_hl2_item['value']}]"
                            )
                            if _hl2_item.get("detail"):
                                st.caption(_hl2_item["detail"])
            except Exception:
                pass

            # Global search (Sprint 5)
            st.markdown("---")
            render_global_search(analysis_payload)

        # Build report from results (for detailed tabs)
        report = build_report_from_results(results, project_id)

        # ===================
        # DETAILED TABS
        # ===================
        st.markdown("---")

        _tab_labels = [
            "Summary",
            "Missing Deps",
            "Flagged",
            "RFIs",
            "Coverage",
            "Assumptions",
            "Audit",
        ]
        if analysis_payload:
            _tab_labels.append("\U0001f4c8 Bid Strategy")
        tabs = st.tabs(_tab_labels)

        with tabs[0]:
            if analysis_payload:
                st.caption("See dashboard above for at-a-glance view")
            render_section_1_summary(report)

            # Export row
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                html_content = f"""<!DOCTYPE html><html><head><title>Bid Readiness Report</title></head>
                <body><h1>Bid Readiness Report: {project_id}</h1>
                <p>Score: {report['executive_summary']['readiness_score']}/100</p>
                <p>Decision: {report['executive_summary']['decision']}</p>
                <h2>Missing Dependencies</h2>
                {''.join(f"<p>{d['dependency_type']}: {d['why_needed'][:100]}</p>" for d in report['missing_dependencies'])}
                </body></html>"""
                st.download_button("HTML Report", html_content, f"bid_report_{project_id}.html", "text/html", use_container_width=True)
            with col2:
                csv_content = _generate_rfi_csv_from_report(report["rfis"])
                st.download_button("RFI CSV", csv_content, f"rfis_{project_id}.csv", "text/csv", use_container_width=True)
            with col3:
                st.download_button("Full JSON", json.dumps(report, indent=2, default=str), f"report_{project_id}.json", "application/json", use_container_width=True)

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

        if analysis_payload:
            with tabs[7]:
                render_bid_strategy_tab(analysis_payload)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem;">
            <span style="color: #3f3f46; font-size: 0.75rem;">xBOQ &bull; Pre-Bid Scope & Risk Check</span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
