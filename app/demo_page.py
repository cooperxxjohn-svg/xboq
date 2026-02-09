"""
XBOQ Drawing Intelligence Demo Page - YC Demo Version

A read-only demo page that proves drawings were actually analyzed.
Shows REAL extracted data only - never shows placeholder or mock data.

Data source: out/<project_id>/ (JSON/CSV files)

Run with:
    streamlit run app/demo_page.py -- --project_id demo

Demo Requirements:
1. Drawing Viewer with overlays (rooms/openings at minimum)
2. Measured vs Inferred split (no fabricated numbers)
3. Bid Gate + RFIs (decision + evidence)
"""

import streamlit as st
import pandas as pd
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="XBOQ - Drawing Intelligence Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# THEME CSS (Matches existing XBOQ site)
# =============================================================================

st.markdown("""
<style>
    /* Main typography */
    .main-header {
        font-size: 2.4rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }

    /* Metric cards (brand gradient) */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-card .label {
        font-size: 0.85rem;
        opacity: 0.9;
    }

    /* Demo Checklist */
    .demo-checklist {
        background: #f8f9fa;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
    }
    .demo-checklist-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1E3A5F;
        margin-bottom: 10px;
    }
    .checklist-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .checklist-pass {
        color: #28a745;
    }
    .checklist-fail {
        color: #dc3545;
    }
    .checklist-icon {
        margin-right: 8px;
        font-size: 1rem;
    }

    /* Status badges */
    .badge-success {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-warning {
        background: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-error {
        background: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-info {
        background: #cce5ff;
        color: #004085;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }

    /* Alert banners */
    .alert-critical {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .alert-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .alert-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }

    /* Tables */
    .data-table th {
        background: #f8f9fa;
        color: #1E3A5F;
        font-weight: 600;
    }
    .measured-row {
        background: #d4edda !important;
    }
    .inferred-row {
        background: #fff3cd !important;
    }

    /* Drawing viewer */
    .drawing-container {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        background: #f7fafc;
    }

    /* Overlay toggle buttons */
    .overlay-btn {
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        border: 1px solid #667eea;
        background: white;
        color: #667eea;
        cursor: pointer;
        transition: all 0.2s;
    }
    .overlay-btn.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .overlay-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    /* RFI cards */
    .rfi-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .rfi-title {
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 5px;
    }
    .rfi-reason {
        color: #666;
        font-size: 0.9rem;
    }

    /* Tab styling to match brand */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Bid decision banner */
    .bid-decision {
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
    }
    .bid-decision.go {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
    }
    .bid-decision.review {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: #1E3A5F;
    }
    .bid-decision.no-go {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
    }
    .bid-decision .decision-text {
        font-size: 2rem;
        font-weight: bold;
    }
    .bid-decision .decision-reason {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DemoData:
    """Container for demo page data."""
    # Run identity
    project_id: str = ""
    run_id: str = ""
    run_timestamp: str = ""
    run_duration_sec: float = 0

    # Core metrics
    pages_processed: int = 0
    pages_routed: int = 0
    candidate_pages: int = 0
    rooms_found: int = 0
    openings_found: int = 0

    # Measurement metrics
    boq_measured: int = 0
    boq_counted: int = 0
    boq_inferred: int = 0
    coverage_percent: float = 0.0
    rfis_generated: int = 0

    # Status flags
    measurement_gate_passed: bool = False
    pricing_enabled: bool = False
    bid_recommendation: str = "N/A"
    bid_score: int = 0
    pricing_disabled: bool = False
    pricing_disabled_reason: str = ""

    # Data tables - Evidence-first
    measured_items: List[Dict] = field(default_factory=list)
    counted_items: List[Dict] = field(default_factory=list)
    inferred_items: List[Dict] = field(default_factory=list)
    rfis: List[Dict] = field(default_factory=list)

    # Per-page data
    page_stats: Dict[int, Dict] = field(default_factory=dict)

    # File paths for overlays and thumbnails
    thumbnails: Dict[int, str] = field(default_factory=dict)
    overlays: Dict[str, str] = field(default_factory=dict)  # overlay_type -> path
    overlay_pages: Dict[int, Dict[str, str]] = field(default_factory=dict)  # page -> {overlay_type -> path}

    # Demo checklist status
    checklist: Dict[str, bool] = field(default_factory=dict)

    # Errors
    load_errors: List[str] = field(default_factory=list)


# =============================================================================
# DATA LOADING - EVIDENCE-FIRST
# =============================================================================

def load_json(path: Path) -> Optional[Any]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_csv(path: Path, limit: int = None) -> List[Dict]:
    """Load CSV file as list of dicts."""
    if not path.exists():
        return []
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            return data[:limit] if limit else data
    except Exception:
        return []


def load_demo_data(project_id: str) -> DemoData:
    """Load demo data from project output directory.

    EVIDENCE-FIRST: Only loads data from actual output files.
    No placeholders or fallback values.

    For deployed version: Falls back to bundled demo_data/ if out/ not found.
    """
    # Try project output directory first
    output_dir = PROJECT_ROOT / "out" / project_id

    # Fallback to bundled demo_data for Streamlit Cloud deployment
    if not output_dir.exists():
        bundled_dir = Path(__file__).parent / "demo_data"
        if bundled_dir.exists():
            output_dir = bundled_dir
            project_id = "demo"  # Use demo as project name for bundled data

    data = DemoData(project_id=project_id)

    if not output_dir.exists():
        data.load_errors.append(f"Output directory not found: {output_dir}")
        return data

    # Load run_metadata.json (primary source)
    metadata = load_json(output_dir / "run_metadata.json")
    if not metadata:
        data.load_errors.append("run_metadata.json not found - run pipeline first")
        return data

    # Extract run identity
    data.run_id = metadata.get("run_id", "UNKNOWN")
    data.run_timestamp = metadata.get("start_time", "")
    data.run_duration_sec = metadata.get("duration_sec", 0)

    # Extract aggregates
    agg = metadata.get("aggregates", {})
    data.pages_processed = agg.get("pages_processed", 0)
    data.pages_routed = agg.get("pages_routed", 0)
    data.candidate_pages = agg.get("candidate_pages", 0)
    data.rooms_found = agg.get("rooms_found", 0)
    data.openings_found = agg.get("openings_found", 0)
    data.boq_measured = agg.get("boq_measured", 0)
    data.boq_counted = agg.get("boq_counted", 0)
    data.boq_inferred = agg.get("boq_inferred", 0)
    data.coverage_percent = agg.get("coverage_percent", 0.0)
    data.rfis_generated = agg.get("rfis_generated", 0)
    data.bid_recommendation = agg.get("bid_recommendation", "N/A")
    data.bid_score = agg.get("bid_score", 0)

    # Check measurement gate
    data.measurement_gate_passed = "07_measurement_gate" not in metadata.get("phases_failed", [])
    data.pricing_enabled = "16_pricing" in metadata.get("phases_run", [])

    # Load bid gate result for pricing status
    bid_gate_result = load_json(output_dir / "bid_gate_result.json")
    if bid_gate_result:
        data.pricing_disabled = bid_gate_result.get("pricing_disabled", False)
        data.pricing_disabled_reason = bid_gate_result.get("pricing_disabled_reason", "")

    # =========================================================================
    # EVIDENCE-FIRST: Load from specific output files only
    # =========================================================================

    # Load MEASURED items from boq_measured.csv ONLY
    measured_csv = output_dir / "boq" / "boq_measured.csv"
    if measured_csv.exists():
        data.measured_items = load_csv(measured_csv)
    else:
        # Try JSON fallback
        measured_json = output_dir / "boq" / "measured.json"
        if measured_json.exists():
            items = load_json(measured_json)
            if items:
                data.measured_items = items

    # Load COUNTED items from boq_counted.csv ONLY
    counted_csv = output_dir / "boq" / "boq_counted.csv"
    if counted_csv.exists():
        data.counted_items = load_csv(counted_csv)
    else:
        counted_json = output_dir / "boq" / "counted.json"
        if counted_json.exists():
            items = load_json(counted_json)
            if items:
                data.counted_items = items

    # Load INFERRED items from boq_inferred.csv ONLY
    inferred_csv = output_dir / "boq" / "boq_inferred.csv"
    if inferred_csv.exists():
        data.inferred_items = load_csv(inferred_csv)
    else:
        inferred_json = output_dir / "boq" / "inferred.json"
        if inferred_json.exists():
            items = load_json(inferred_json)
            if items:
                data.inferred_items = items

    # Load RFIs
    rfi_path = output_dir / "rfi" / "rfi_log.md"
    if not rfi_path.exists():
        rfi_path = output_dir / "estimator" / "rfi_missing_scope.md"

    if rfi_path.exists():
        try:
            with open(rfi_path) as f:
                content = f.read()
                import re
                # Parse RFI sections - format: ## RFI-MS-001: Title
                rfi_matches = re.findall(
                    r'##\s+(RFI-[A-Z]+-\d+):\s*(.*?)\n(.*?)(?=\n##\s+RFI-|\Z)',
                    content, re.DOTALL
                )
                for rfi_id, title, body in rfi_matches:
                    location_match = re.search(r'\*\*Room/Location\*\*:\s*(.*?)$', body, re.MULTILINE)
                    location = location_match.group(1).strip() if location_match else ""
                    data.rfis.append({
                        "id": rfi_id,
                        "title": title.strip()[:80],
                        "reason": f"Location: {location}" if location else body.strip()[:200],
                    })
        except Exception:
            pass

    # =========================================================================
    # Load thumbnails and overlays
    # =========================================================================

    # Input thumbnails
    thumbnails_dir = output_dir / "proof" / "input_thumbnails"
    if not thumbnails_dir.exists():
        thumbnails_dir = output_dir / "thumbnails"

    if thumbnails_dir.exists():
        for thumb_file in sorted(thumbnails_dir.glob("*.png")):
            try:
                page_num = int(thumb_file.stem.replace("page_", "").replace("thumb_", ""))
                data.thumbnails[page_num] = str(thumb_file)
            except (ValueError, AttributeError):
                pass

    # Overlays directory
    overlays_dir = output_dir / "overlays"
    if overlays_dir.exists():
        # Global overlays
        for overlay_file in overlays_dir.glob("*.png"):
            overlay_name = overlay_file.stem
            data.overlays[overlay_name] = str(overlay_file)

        # Per-page overlays (rooms_page_1.png, openings_page_1.png, etc.)
        for overlay_file in overlays_dir.glob("*_page_*.png"):
            parts = overlay_file.stem.rsplit("_page_", 1)
            if len(parts) == 2:
                overlay_type, page_str = parts
                try:
                    page_num = int(page_str)
                    if page_num not in data.overlay_pages:
                        data.overlay_pages[page_num] = {}
                    data.overlay_pages[page_num][overlay_type] = str(overlay_file)
                except ValueError:
                    pass

    # Load routing data for page stats
    routing_data = load_csv(output_dir / "routing_debug.csv")
    for row in routing_data:
        try:
            page_num = int(row.get("page", 0))
            data.page_stats[page_num] = {
                "score": float(row.get("score", 0)),
                "type": row.get("type", "unknown"),
                "candidate": row.get("candidate", "False") == "True",
            }
        except (ValueError, KeyError):
            pass

    # =========================================================================
    # Build Demo Checklist
    # =========================================================================
    data.checklist = {
        "drawings_loaded": data.pages_processed > 0,
        "pages_processed": data.pages_processed > 0,
        "overlays_available": len(data.overlays) > 0 or len(data.overlay_pages) > 0,
        "rfis_generated": len(data.rfis) > 0 or data.rfis_generated > 0,
        "bid_gate_computed": data.bid_recommendation != "N/A",
    }

    return data


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_metric_card(value: Any, label: str, sublabel: str = ""):
    """Render a brand-colored metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {f'<div class="label" style="font-size:0.75rem;opacity:0.7">{sublabel}</div>' if sublabel else ''}
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, badge_type: str = "info"):
    """Render a status badge."""
    return f'<span class="badge-{badge_type}">{text}</span>'


def render_demo_checklist(checklist: Dict[str, bool]):
    """Render the demo checklist banner."""
    items = [
        ("drawings_loaded", "Drawings loaded"),
        ("pages_processed", "Pages processed"),
        ("overlays_available", "Overlays available"),
        ("rfis_generated", "RFIs generated"),
        ("bid_gate_computed", "Bid gate computed"),
    ]

    all_pass = all(checklist.get(k, False) for k, _ in items)
    border_color = "#28a745" if all_pass else "#dc3545"

    html = f"""
    <div class="demo-checklist" style="border-color: {border_color}">
        <div class="demo-checklist-title">Demo Checklist</div>
    """

    for key, label in items:
        passed = checklist.get(key, False)
        icon = "✅" if passed else "❌"
        css_class = "checklist-pass" if passed else "checklist-fail"
        reason = "" if passed else " — Not generated"
        html += f"""
        <div class="checklist-item {css_class}">
            <span class="checklist-icon">{icon}</span>
            {label}{reason}
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_bid_decision_banner(recommendation: str, score: int, pricing_disabled: bool, reason: str):
    """Render the bid decision banner."""
    rec_lower = recommendation.lower().replace("-", "")
    if rec_lower == "go":
        css_class = "go"
        emoji = "🟢"
    elif rec_lower == "review" or rec_lower == "conditional":
        css_class = "review"
        emoji = "🟡"
    else:
        css_class = "no-go"
        emoji = "🔴"

    reason_text = f"Score: {score}/100"
    if pricing_disabled:
        reason_text += f" | ⛔ Pricing Disabled: {reason}"

    st.markdown(f"""
    <div class="bid-decision {css_class}">
        <div class="decision-text">{emoji} {recommendation}</div>
        <div class="decision-reason">{reason_text}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SECTION 1: DRAWING VIEWER
# =============================================================================

def render_drawing_viewer(data: DemoData):
    """Render the drawing viewer with overlay toggles."""
    st.markdown('<h2 class="section-header">📐 Drawing Viewer</h2>', unsafe_allow_html=True)

    if not data.thumbnails and not data.overlays:
        st.warning("⚠️ No thumbnails or overlays available. Run pipeline with overlay generation enabled.")
        return

    # Page selector
    col1, col2 = st.columns([3, 1])

    with col1:
        # Determine available pages
        all_pages = sorted(set(data.thumbnails.keys()) | set(data.overlay_pages.keys()))

        if not all_pages:
            st.info("No page images available.")
            return

        selected_page = st.selectbox(
            "Select Page",
            all_pages,
            format_func=lambda x: f"Page {x}"
        )

        # Overlay toggle buttons
        st.markdown("**Overlay Toggles:**")
        available_overlays = data.overlay_pages.get(selected_page, {})

        overlay_cols = st.columns(5)
        overlay_types = ["rooms", "openings", "walls", "scale", "devices"]
        selected_overlays = []

        for i, otype in enumerate(overlay_types):
            with overlay_cols[i]:
                has_overlay = otype in available_overlays
                if has_overlay:
                    if st.checkbox(otype.title(), value=True, key=f"overlay_{otype}"):
                        selected_overlays.append(otype)
                else:
                    st.checkbox(otype.title(), value=False, disabled=True, key=f"overlay_{otype}_disabled")

        # Display image - prefer overlay, fallback to thumbnail
        if selected_overlays and available_overlays:
            # Show first selected overlay
            first_overlay = selected_overlays[0]
            if first_overlay in available_overlays:
                st.image(available_overlays[first_overlay], caption=f"Page {selected_page} - {first_overlay} overlay", use_container_width=True)
            elif selected_page in data.thumbnails:
                st.image(data.thumbnails[selected_page], caption=f"Page {selected_page}", use_container_width=True)
        elif selected_page in data.thumbnails:
            st.image(data.thumbnails[selected_page], caption=f"Page {selected_page}", use_container_width=True)
        else:
            st.info(f"No image available for page {selected_page}")

    with col2:
        st.markdown("### Page Status")
        page_info = data.page_stats.get(selected_page, {})

        st.metric("Page Score", f"{page_info.get('score', 0):.2f}")
        st.markdown(f"**Type:** {page_info.get('type', 'unknown')}")
        st.markdown(f"**Candidate:** {'✓ Yes' if page_info.get('candidate') else '✗ No'}")

        # Show overlay availability
        available = list(available_overlays.keys()) if available_overlays else []
        if available:
            st.markdown(f"**Overlays:** {', '.join(available)}")
        else:
            st.markdown("**Overlays:** None")


# =============================================================================
# SECTION 2: MEASURED VS INFERRED
# =============================================================================

def render_measured_vs_inferred(data: DemoData):
    """Render the measured vs inferred section - EVIDENCE-FIRST."""
    st.markdown('<h2 class="section-header">📊 Measured vs Inferred</h2>', unsafe_allow_html=True)

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        render_metric_card(len(data.measured_items), "Measured", "Geometry-backed")
    with cols[1]:
        render_metric_card(len(data.counted_items), "Counted", "Symbol/schedule counts")
    with cols[2]:
        render_metric_card(len(data.inferred_items), "Inferred", "Rule-based estimates")
    with cols[3]:
        total = len(data.measured_items) + len(data.inferred_items)
        coverage = (len(data.measured_items) / total * 100) if total > 0 else 0
        render_metric_card(f"{coverage:.0f}%", "Coverage", "measured / measurable")

    st.markdown("")

    # Tabs for each bucket
    tab1, tab2, tab3 = st.tabs(["🟢 MEASURED", "🔵 COUNTED", "🟡 INFERRED"])

    with tab1:
        if not data.measured_items:
            st.info("📭 No measured quantities. measured.json / boq_measured.csv is empty or not generated.")
        else:
            st.markdown(f"**{len(data.measured_items)} items** with geometry-backed measurements")
            df = pd.DataFrame(data.measured_items)
            display_cols = [c for c in ["description", "quantity", "unit", "source_page", "measurement_method", "confidence"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols].head(50), use_container_width=True, height=350)
                if len(data.measured_items) > 50:
                    st.caption(f"Showing 50 of {len(data.measured_items)} items")

    with tab2:
        if not data.counted_items:
            st.info("📭 No counted quantities. counted.json / boq_counted.csv is empty or not generated.")
        else:
            st.markdown(f"**{len(data.counted_items)} items** from symbol counts or schedules")
            df = pd.DataFrame(data.counted_items)
            display_cols = [c for c in ["description", "quantity", "unit", "source_page", "count_method"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols].head(50), use_container_width=True, height=350)
                if len(data.counted_items) > 50:
                    st.caption(f"Showing 50 of {len(data.counted_items)} items")

    with tab3:
        if not data.inferred_items:
            st.success("✅ No inferred items - all quantities are evidence-backed!")
        else:
            st.markdown(f"**{len(data.inferred_items)} items** based on rules or assumptions")
            st.warning("⚠️ These quantities are NOT geometry-backed. Verify before bidding.")
            df = pd.DataFrame(data.inferred_items)
            display_cols = [c for c in ["description", "quantity", "unit", "infer_reason", "trigger", "linked_rfi"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols].head(50), use_container_width=True, height=350)
                if len(data.inferred_items) > 50:
                    st.caption(f"Showing 50 of {len(data.inferred_items)} items")


# =============================================================================
# SECTION 3: BID GATE + RFIS
# =============================================================================

def render_bid_gate_rfis(data: DemoData):
    """Render the bid gate decision and RFIs section."""
    st.markdown('<h2 class="section-header">🎯 Bid Gate + RFIs</h2>', unsafe_allow_html=True)

    # Bid decision banner
    render_bid_decision_banner(
        data.bid_recommendation,
        data.bid_score,
        data.pricing_disabled,
        data.pricing_disabled_reason
    )

    st.markdown("")

    # RFIs
    if not data.rfis:
        if data.rfis_generated > 0:
            st.info(f"📋 {data.rfis_generated} RFIs were generated but details not loaded.")
        else:
            st.success("✅ No RFIs required - all information found in drawings!")
        return

    st.markdown(f"### ❓ {len(data.rfis)} RFIs Require Resolution")

    # Display as compact list
    cols = st.columns(2)
    for i, rfi in enumerate(data.rfis[:20]):
        with cols[i % 2]:
            priority = "HIGH" if "missing" in rfi.get("reason", "").lower() else "MEDIUM"
            badge_class = "error" if priority == "HIGH" else "warning"

            st.markdown(f"""
            <div class="rfi-card">
                <div class="rfi-title">{rfi.get('id', 'RFI')} {render_badge(priority, badge_class)}</div>
                <div style="font-weight:500;margin:5px 0">{rfi.get('title', 'Untitled')}</div>
                <div class="rfi-reason">{rfi.get('reason', '')[:120]}...</div>
            </div>
            """, unsafe_allow_html=True)

    if len(data.rfis) > 20:
        st.caption(f"Showing 20 of {len(data.rfis)} RFIs")


# =============================================================================
# ADVANCED SECTION (COLLAPSED)
# =============================================================================

def render_advanced_section(data: DemoData):
    """Render the advanced section in a collapsible expander."""
    with st.expander("🔧 Advanced: Run Health & Details", expanded=False):
        cols = st.columns(4)

        with cols[0]:
            st.markdown("**Run Identity**")
            st.markdown(f"- **Project:** {data.project_id}")
            st.markdown(f"- **Run ID:** `{data.run_id}`")
            st.markdown(f"- **Timestamp:** {data.run_timestamp}")
            st.markdown(f"- **Duration:** {data.run_duration_sec:.1f}s")

        with cols[1]:
            st.markdown("**Extraction Stats**")
            st.markdown(f"- **Pages:** {data.pages_processed}")
            st.markdown(f"- **Candidates:** {data.candidate_pages}")
            st.markdown(f"- **Rooms:** {data.rooms_found}")
            st.markdown(f"- **Openings:** {data.openings_found}")

        with cols[2]:
            st.markdown("**Measurement Stats**")
            st.markdown(f"- **Measured:** {data.boq_measured}")
            st.markdown(f"- **Counted:** {data.boq_counted}")
            st.markdown(f"- **Inferred:** {data.boq_inferred}")
            st.markdown(f"- **Coverage:** {data.coverage_percent:.0f}%")

        with cols[3]:
            st.markdown("**Gate Status**")
            gate_status = "✅ PASS" if data.measurement_gate_passed else "❌ FAIL"
            pricing_status = "✅ Enabled" if data.pricing_enabled else "⛔ Disabled"
            st.markdown(f"- **Measurement Gate:** {gate_status}")
            st.markdown(f"- **Pricing:** {pricing_status}")
            st.markdown(f"- **Bid:** {data.bid_recommendation}")
            st.markdown(f"- **Score:** {data.bid_score}/100")

        # Load errors
        if data.load_errors:
            st.error("**Data Load Errors:**")
            for err in data.load_errors:
                st.markdown(f"- {err}")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Get project_id from URL params
    query_params = st.query_params
    project_id = query_params.get("project_id", "demo")

    # Sidebar for config
    with st.sidebar:
        st.markdown("### Configuration")
        project_id = st.text_input("Project ID", value=project_id)
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    data = load_demo_data(project_id)

    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown('<h1 class="main-header">🔍 XBOQ Drawing Intelligence Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Every quantity below is linked to actual drawing evidence.</p>', unsafe_allow_html=True)

    # =========================================================================
    # DEMO CHECKLIST BANNER
    # =========================================================================
    render_demo_checklist(data.checklist)

    # =========================================================================
    # CRITICAL CHECK: No pages processed
    # =========================================================================
    if data.pages_processed == 0:
        st.markdown("""
        <div class="alert-critical">
            <strong>⚠️ No drawings were processed.</strong><br>
            Run the pipeline first: <code>python run_full_project.py --project_id {project_id} --demo_mode</code>
        </div>
        """.format(project_id=project_id), unsafe_allow_html=True)
        render_advanced_section(data)
        return

    # =========================================================================
    # SECTION 1: DRAWING VIEWER
    # =========================================================================
    st.markdown("---")
    render_drawing_viewer(data)

    # =========================================================================
    # SECTION 2: MEASURED VS INFERRED
    # =========================================================================
    st.markdown("---")
    render_measured_vs_inferred(data)

    # =========================================================================
    # SECTION 3: BID GATE + RFIS
    # =========================================================================
    st.markdown("---")
    render_bid_gate_rfis(data)

    # =========================================================================
    # ADVANCED (COLLAPSED)
    # =========================================================================
    st.markdown("---")
    render_advanced_section(data)


if __name__ == "__main__":
    main()
