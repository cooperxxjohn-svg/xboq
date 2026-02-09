"""
XBOQ - India-First Preconstruction Platform
Structural Takeoff & Scope Extraction Tool

Features:
- High Recall Mode for maximum scope coverage
- Multi-source extraction (PDF, OCR, Camelot tables)
- India-first terminology (RCC, PCC, shuttering, saria, etc.)
- Scope checklist with detected/inferred/missing status
- Conflict detection and coverage reporting
- CPWD-style Excel exports

NO PRICING - pure quantity takeoff and scope definition.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import tempfile
import logging
import json
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import UI utilities first (for sanitization)
from src.ui.streamlit_utils import (
    sanitize_df_for_streamlit,
    render_evidence_viewer,
    render_confidence_badge,
    render_source_badge,
    render_status_badge,
    render_severity_badge,
    render_needs_review_banner,
    boq_items_to_df,
    scope_items_to_df,
    conflicts_to_df,
    coverage_to_df,
)

from src.ui.export_utils import (
    export_boq_to_excel,
    export_scope_to_excel,
    export_full_package_to_excel,
    export_boq_to_csv,
    export_takeoff_package_zip,
    export_vendor_column_schedule_to_excel,
    export_vendor_column_schedule_to_csv,
    export_vendor_schedule_summary,
)

# Import models and pipeline
from src.models.estimate_schema import (
    EstimatePackage,
    BOQItem,
    ScopeItem,
    Conflict,
    QtyStatus,
    ScopeStatus,
    Severity,
)

from src.pipeline.takeoff_pipeline import run_takeoff_pipeline

# Legacy imports for backward compatibility
try:
    from src.extractors import (
        extract_column_schedule,
        ColumnScheduleResult,
    )
    TABLE_EXTRACTION_AVAILABLE = True
except ImportError:
    TABLE_EXTRACTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="XBOQ - Structural Takeoff",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .status-detected { color: #28a745; font-weight: bold; }
    .status-inferred { color: #ffc107; font-weight: bold; }
    .status-missing { color: #dc3545; font-weight: bold; }
    .confidence-high { background-color: #d4edda; }
    .confidence-med { background-color: #fff3cd; }
    .confidence-low { background-color: #f8d7da; }
    .source-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .source-explicit { background-color: #d4edda; color: #155724; }
    .source-inferred { background-color: #cce5ff; color: #004085; }
    .source-schedule { background-color: #e2e3e5; color: #383d41; }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if 'estimate_package' not in st.session_state:
    st.session_state.estimate_package = None
if 'column_schedule' not in st.session_state:
    st.session_state.column_schedule = None
if 'high_recall' not in st.session_state:
    st.session_state.high_recall = True
if 'show_evidence' not in st.session_state:
    st.session_state.show_evidence = False


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🏗️ XBOQ Structural Takeoff</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">India-First Preconstruction Platform | Scope & Quantity Extraction | NO PRICING</p>',
        unsafe_allow_html=True
    )

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("📐 Drawing Context")
        st.caption("_Supplements extracted data when not found in drawings_")

        floors = st.number_input(
            "Number of Floors",
            min_value=1, max_value=50, value=4,
            help="Used for scaling if not detected from drawing. Overridden by schedule data when available."
        )

        storey_height = st.number_input(
            "Storey Height (mm)",
            min_value=2400, max_value=5000, value=3000, step=100,
            help="Fallback height when not specified in drawing notes. Extracted values take priority."
        )

        st.divider()

        # Extraction Mode
        st.subheader("🎯 Extraction Mode")
        high_recall = st.toggle(
            "High Recall Mode",
            value=True,
            help="Include inferred scope items typical for RCC buildings"
        )
        st.session_state.high_recall = high_recall

        if high_recall:
            st.caption("✓ Detected + Inferred + Missing items shown")
        else:
            st.caption("✓ Only directly detected items")

        show_evidence = st.toggle(
            "Show Evidence Details",
            value=False,
            help="Display source snippets and page references"
        )
        st.session_state.show_evidence = show_evidence

        st.divider()

        # File Upload
        st.subheader("📤 Upload Drawing")
        uploaded_file = st.file_uploader(
            "Structural/Foundation Plan (PDF)",
            type=['pdf'],
            help="Upload foundation layout or reinforcement schedule"
        )

        if uploaded_file:
            st.success(f"✓ {uploaded_file.name}")

            if st.button("🚀 Run Takeoff", type="primary", use_container_width=True):
                with st.spinner("Extracting scope and quantities..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

                    # Run the new pipeline
                    package = run_takeoff_pipeline(
                        tmp_path,
                        floors=floors,
                        storey_height_mm=storey_height,
                        high_recall=high_recall
                    )

                    st.session_state.estimate_package = package

                    # Also extract column schedule for dedicated tab
                    if TABLE_EXTRACTION_AVAILABLE:
                        try:
                            col_schedule = extract_column_schedule(str(tmp_path), page_number=0)
                            st.session_state.column_schedule = col_schedule
                        except Exception as e:
                            logger.warning(f"Column schedule extraction failed: {e}")

                    st.success("✅ Takeoff complete!")
                    st.rerun()

        st.divider()

        # Quick Stats
        if st.session_state.estimate_package:
            package = st.session_state.estimate_package
            stats = package.stats

            st.subheader("📊 Quick Stats")
            st.metric("BOQ Items", stats['total_boq_items'])
            st.metric("Scope Items", stats['total_scope_items'])
            st.metric("Conflicts", stats['total_conflicts'])

            conf = package.drawing.confidence_overall
            conf_color = "normal" if conf >= 0.7 else ("off" if conf >= 0.5 else "inverse")
            st.metric("Confidence", f"{conf:.0%}", delta_color=conf_color)

    # =========================================================================
    # MAIN CONTENT
    # =========================================================================

    if st.session_state.estimate_package:
        package = st.session_state.estimate_package
        stats = package.stats

        # Needs Review Banner
        needs_review_reasons = []
        if package.drawing.confidence_overall < 0.6:
            needs_review_reasons.append("Low overall confidence")
        if not package.drawing.scale:
            needs_review_reasons.append("Drawing scale not detected")
        if stats.get('high_severity_conflicts', 0) > 0:
            needs_review_reasons.append(f"{stats['high_severity_conflicts']} high severity conflicts")

        unknown_qty = stats.get('boq_by_qty_status', {}).get('unknown', 0)
        if unknown_qty > len(package.boq) * 0.5:
            needs_review_reasons.append(f"{unknown_qty} items with unknown quantities")

        if needs_review_reasons:
            render_needs_review_banner(
                "This takeoff needs human review before use",
                needs_review_reasons
            )

        # Top Metrics Bar
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📄 Drawing", package.drawing.file_name[:20] + "..." if len(package.drawing.file_name) > 20 else package.drawing.file_name)
        with col2:
            st.metric("📏 Scale", package.drawing.scale or "Not detected")
        with col3:
            st.metric("🏗️ Discipline", package.drawing.discipline.value.title())
        with col4:
            detected = stats.get('scope_by_status', {}).get('detected', 0)
            st.metric("✅ Detected Scope", detected)
        with col5:
            computed = stats.get('boq_by_qty_status', {}).get('computed', 0)
            st.metric("📊 Computed Qty", computed)

        st.divider()

        # =====================================================================
        # TABS
        # =====================================================================
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Scope Checklist",
            "📊 BOQ Items",
            "🔩 Column Schedule",
            "📦 Vendor Schedule",
            "📈 Coverage Report",
            "⚠️ Conflicts",
            "📊 Element Schedules",
            "📥 Export"
        ])

        # =================================================================
        # TAB 1: SCOPE CHECKLIST
        # =================================================================
        with tab1:
            st.subheader("📋 Scope Checklist")
            st.caption("What's detected, inferred, or missing from the drawing")

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    ["detected", "inferred", "missing", "needs_review"],
                    default=["detected", "inferred", "missing"]
                )
            with col2:
                categories = sorted(set(s.category.value for s in package.scope))
                category_filter = st.multiselect(
                    "Filter by Category",
                    categories,
                    default=categories
                )
            with col3:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)

            # Build filtered DataFrame
            filtered_scope = [
                s for s in package.scope
                if s.status.value in status_filter
                and s.category.value in category_filter
                and s.confidence >= min_confidence
            ]

            if filtered_scope:
                scope_df = scope_items_to_df(filtered_scope)
                st.dataframe(
                    sanitize_df_for_streamlit(scope_df),
                    use_container_width=True,
                    hide_index=True
                )

                # Summary by status
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    detected = sum(1 for s in package.scope if s.status == ScopeStatus.DETECTED)
                    st.metric("✅ Detected", detected)
                with col2:
                    inferred = sum(1 for s in package.scope if s.status == ScopeStatus.INFERRED)
                    st.metric("🔍 Inferred", inferred)
                with col3:
                    missing = sum(1 for s in package.scope if s.status == ScopeStatus.MISSING)
                    st.metric("❌ Missing", missing)
                with col4:
                    review = sum(1 for s in package.scope if s.status == ScopeStatus.NEEDS_REVIEW)
                    st.metric("⚠️ Needs Review", review)

                # Evidence viewer
                if st.session_state.show_evidence:
                    st.divider()
                    st.subheader("🔍 Evidence Details")
                    for item in filtered_scope[:10]:
                        with st.expander(f"{render_status_badge(item.status.value)} {item.trade[:50]}"):
                            st.write(f"**Category:** {item.category.value.title()}")
                            st.write(f"**Reason:** {item.reason}")
                            st.write(f"**Confidence:** {render_confidence_badge(item.confidence)}")
                            if item.evidence:
                                render_evidence_viewer(item.evidence)
            else:
                st.info("No scope items match the current filters")

        # =================================================================
        # TAB 2: BOQ ITEMS
        # =================================================================
        with tab2:
            st.subheader("📊 Bill of Quantities")
            st.caption("Extracted scope items with quantities (NO PRICING)")

            # Filters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                systems = sorted(set(b.system for b in package.boq))
                system_filter = st.multiselect("System", systems, default=systems)
            with col2:
                subsystems = sorted(set(b.subsystem for b in package.boq))
                subsystem_filter = st.multiselect("Subsystem", subsystems, default=subsystems)
            with col3:
                qty_status_filter = st.multiselect(
                    "Qty Status",
                    ["computed", "partial", "unknown"],
                    default=["computed", "partial", "unknown"]
                )
            with col4:
                source_filter = st.multiselect(
                    "Source",
                    ["explicit", "synonym", "inferred"],
                    default=["explicit", "synonym", "inferred"]
                )

            # Filter BOQ items
            filtered_boq = [
                b for b in package.boq
                if b.system in system_filter
                and b.subsystem in subsystem_filter
                and b.qty_status.value in qty_status_filter
                and b.source in source_filter
            ]

            if filtered_boq:
                boq_df = boq_items_to_df(filtered_boq)

                # Style the dataframe
                def highlight_qty_status(row):
                    status = str(row.get('Qty Status', '')).lower()
                    if status == 'computed':
                        return ['background-color: #d4edda'] * len(row)
                    elif status == 'partial':
                        return ['background-color: #fff3cd'] * len(row)
                    elif status == 'unknown':
                        return ['background-color: #f8d7da'] * len(row)
                    return [''] * len(row)

                styled_df = boq_df.style.apply(highlight_qty_status, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Summary
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    computed = sum(1 for b in package.boq if b.qty_status == QtyStatus.COMPUTED)
                    st.metric("✅ Computed Qty", computed)
                with col2:
                    partial = sum(1 for b in package.boq if b.qty_status == QtyStatus.PARTIAL)
                    st.metric("⚠️ Partial Qty", partial)
                with col3:
                    unknown = sum(1 for b in package.boq if b.qty_status == QtyStatus.UNKNOWN)
                    st.metric("❓ Unknown Qty", unknown)
                with col4:
                    high_conf = sum(1 for b in package.boq if b.confidence >= 0.8)
                    st.metric("🎯 High Confidence", high_conf)

                # Dependencies view
                st.divider()
                st.subheader("📌 Missing Dependencies")
                all_deps = set()
                for b in package.boq:
                    if b.qty_status != QtyStatus.COMPUTED:
                        all_deps.update(b.dependencies)

                if all_deps:
                    dep_counts = {}
                    for b in package.boq:
                        for d in b.dependencies:
                            dep_counts[d] = dep_counts.get(d, 0) + 1

                    for dep, count in sorted(dep_counts.items(), key=lambda x: -x[1])[:10]:
                        st.warning(f"**{dep}** - affects {count} item(s)")
                else:
                    st.success("✓ All dependencies satisfied!")

                # Evidence viewer
                if st.session_state.show_evidence:
                    st.divider()
                    st.subheader("🔍 Item Details")
                    for item in filtered_boq[:8]:
                        with st.expander(f"{render_source_badge(item.source)} {item.item_name[:60]}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Unit:** {item.unit or '-'}")
                                st.write(f"**Qty:** {item.qty if item.qty else '-'}")
                                st.write(f"**Status:** {render_status_badge(item.qty_status.value)}")
                            with col2:
                                st.write(f"**Confidence:** {render_confidence_badge(item.confidence)}")
                                st.write(f"**Source:** {render_source_badge(item.source)}")
                            if item.measurement_rule:
                                st.caption(f"📐 {item.measurement_rule}")
                            if item.dependencies:
                                st.caption(f"📌 Needs: {', '.join(item.dependencies)}")
                            if item.evidence:
                                render_evidence_viewer(item.evidence)
            else:
                st.info("No BOQ items match the current filters")

        # =================================================================
        # TAB 3: COLUMN SCHEDULE
        # =================================================================
        with tab3:
            st.subheader("🔩 Column Reinforcement Schedule")
            st.caption("Extracted from Camelot/pdfplumber table detection")

            column_schedule = st.session_state.column_schedule

            if column_schedule and column_schedule.raw_dataframe is not None and not column_schedule.raw_dataframe.empty:
                st.success(f"✅ Table extracted using: **{column_schedule.extraction_method}**")

                # Raw table
                st.subheader("📊 Raw Extracted Table")
                st.dataframe(
                    sanitize_df_for_streamlit(column_schedule.raw_dataframe),
                    use_container_width=True,
                    hide_index=False
                )

                if column_schedule.headers_detected:
                    st.caption(f"Headers: {', '.join(str(h) for h in column_schedule.headers_detected[:8])}")

                st.divider()

                # Parsed entries
                if column_schedule.entries:
                    st.subheader("📋 Parsed Interpretation")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entries", len(column_schedule.entries))
                    with col2:
                        total_cols = sum(len(e.column_marks) for e in column_schedule.entries)
                        st.metric("Total Columns", total_cols)
                    with col3:
                        st.metric("Method", column_schedule.extraction_method)
                    with col4:
                        st.metric("Confidence", f"{column_schedule.confidence:.0%}")

                    # Summary table
                    summary_data = []
                    for entry in column_schedule.entries:
                        long_str = ""
                        if entry.longitudinal_parsed:
                            long_str = " + ".join(f"{r.quantity}Y{r.diameter_mm}" for r in entry.longitudinal_parsed)
                        else:
                            long_str = entry.longitudinal_raw[:30] if entry.longitudinal_raw else "-"

                        tie_str = ""
                        if entry.ties_parsed:
                            t = entry.ties_parsed
                            tie_str = f"{t.legs}L Y{t.diameter_mm}@{t.spacing_mm}"
                        else:
                            tie_str = entry.ties_raw[:20] if entry.ties_raw else "-"

                        summary_data.append({
                            'Row': entry.row_number,
                            'Column Marks': ', '.join(entry.column_marks[:3]) + ('...' if len(entry.column_marks) > 3 else ''),
                            'Section': entry.section_size or '-',
                            'Longitudinal': long_str,
                            'Ties': tie_str,
                            'Confidence': f"{entry.confidence:.0%}"
                        })

                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(sanitize_df_for_streamlit(summary_df), use_container_width=True, hide_index=True)
                else:
                    st.warning("Table extracted but entries could not be parsed")
            else:
                if TABLE_EXTRACTION_AVAILABLE:
                    st.info("No column schedule detected. Upload a sheet with 'COLUMN REINFORCEMENT SCHEDULE'.")
                else:
                    st.warning("Table extraction not available. Install: `pip install camelot-py[cv]`")

        # =================================================================
        # TAB 4: VENDOR COLUMN SCHEDULE
        # =================================================================
        with tab4:
            st.subheader("📦 Vendor Column Schedule")
            st.caption("Normalized schedule for vendor handoff and BBS preparation")

            column_schedule = st.session_state.column_schedule

            if column_schedule and hasattr(column_schedule, 'entries') and column_schedule.entries:
                # Summary stats
                summary = export_vendor_schedule_summary(column_schedule)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Columns", summary["total_columns"])
                with col2:
                    st.metric("Schedule Entries", summary["total_entries"])
                with col3:
                    st.metric("Section Sizes", len(summary["section_sizes"]))
                with col4:
                    st.metric("Avg Confidence", f"{summary['avg_confidence']:.0%}")

                st.divider()

                # Vendor-ready table
                st.subheader("📋 Normalized Schedule")

                vendor_rows = []
                for entry in column_schedule.entries:
                    # Parse longitudinal bars
                    long_bars = []
                    total_bar_qty = 0
                    if entry.longitudinal_parsed:
                        for rebar in entry.longitudinal_parsed:
                            long_bars.append(f"{rebar.quantity}Y{rebar.diameter_mm}")
                            total_bar_qty += rebar.quantity

                    # Parse ties
                    tie_str = "-"
                    if entry.ties_parsed:
                        t = entry.ties_parsed
                        tie_str = f"{t.legs}L Y{t.diameter_mm}@{t.spacing_mm}"

                    vendor_rows.append({
                        "Column Marks": ", ".join(entry.column_marks[:5]) + ("..." if len(entry.column_marks) > 5 else ""),
                        "Count": len(entry.column_marks),
                        "Section": entry.section_size or "-",
                        "Longitudinal": " + ".join(long_bars) if long_bars else "-",
                        "Total Bars": total_bar_qty if total_bar_qty > 0 else "-",
                        "Ties": tie_str,
                        "Confidence": f"{entry.confidence:.0%}",
                    })

                vendor_df = pd.DataFrame(vendor_rows)
                st.dataframe(sanitize_df_for_streamlit(vendor_df), use_container_width=True, hide_index=True)

                st.divider()

                # Export buttons
                st.subheader("📥 Vendor Exports")

                col1, col2, col3 = st.columns(3)
                with col1:
                    vendor_excel = export_vendor_column_schedule_to_excel(
                        column_schedule,
                        package.drawing.file_name
                    )
                    st.download_button(
                        "📥 column_schedule_vendor.xlsx",
                        data=vendor_excel,
                        file_name=f"column_schedule_vendor_{package.drawing.file_name.replace('.pdf', '')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                with col2:
                    vendor_csv = export_vendor_column_schedule_to_csv(column_schedule)
                    st.download_button(
                        "📥 column_schedule_vendor.csv",
                        data=vendor_csv,
                        file_name=f"column_schedule_vendor_{package.drawing.file_name.replace('.pdf', '')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col3:
                    # JSON summary
                    summary_json = json.dumps(summary, indent=2)
                    st.download_button(
                        "📥 schedule_summary.json",
                        data=summary_json,
                        file_name="schedule_summary.json",
                        mime="application/json",
                        use_container_width=True
                    )

                st.divider()

                # All column marks list
                if summary["all_column_marks"]:
                    with st.expander(f"🏷️ All Column Marks ({len(summary['all_column_marks'])})"):
                        # Show in 6 columns
                        mark_cols = st.columns(6)
                        for i, mark in enumerate(sorted(set(summary["all_column_marks"]))):
                            with mark_cols[i % 6]:
                                st.write(f"`{mark}`")

                # Diameters used
                if summary["diameters_used"]:
                    st.caption(f"**Diameters used:** {', '.join(f'{d}mm' for d in summary['diameters_used'])}")

                # Section sizes
                if summary["section_sizes"]:
                    st.caption(f"**Section sizes:** {', '.join(summary['section_sizes'])}")

            else:
                st.info("No column schedule data available. Upload a drawing with a column reinforcement schedule to extract vendor data.")

                if TABLE_EXTRACTION_AVAILABLE:
                    st.caption("Table extraction is enabled. Upload a sheet with 'COLUMN REINFORCEMENT SCHEDULE' header.")
                else:
                    st.warning("Table extraction not available. Install: `pip install camelot-py[cv]`")

        # =================================================================
        # TAB 5: COVERAGE REPORT
        # =================================================================
        with tab5:
            st.subheader("📈 Coverage Report")
            st.caption("How well each BOQ item is supported by evidence")

            if package.coverage:
                coverage_df = coverage_to_df(package.coverage, package.boq)
                st.dataframe(sanitize_df_for_streamlit(coverage_df), use_container_width=True, hide_index=True)

                # Summary
                st.divider()
                avg_coverage = sum(c.coverage_score for c in package.coverage) / len(package.coverage)
                high_coverage = sum(1 for c in package.coverage if c.coverage_score >= 0.7)
                low_coverage = sum(1 for c in package.coverage if c.coverage_score < 0.3)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Coverage", f"{avg_coverage:.0%}")
                with col2:
                    st.metric("High Coverage Items", high_coverage)
                with col3:
                    st.metric("Low Coverage Items", low_coverage)

                # Sources used
                st.divider()
                st.subheader("📊 Evidence Sources")
                all_sources = set()
                for c in package.coverage:
                    all_sources.update(s.value if hasattr(s, 'value') else str(s) for s in c.sources_used)

                for source in sorted(all_sources):
                    count = sum(1 for c in package.coverage if any(
                        (s.value if hasattr(s, 'value') else str(s)) == source for s in c.sources_used
                    ))
                    st.write(f"{render_source_badge(source)} - used by {count} items")
            else:
                st.info("Coverage data not available")

        # =================================================================
        # TAB 6: CONFLICTS
        # =================================================================
        with tab6:
            st.subheader("⚠️ Conflicts & Missing Info")
            st.caption("Issues that need attention before finalizing takeoff")

            if package.conflicts:
                # Summary by severity
                col1, col2, col3 = st.columns(3)
                with col1:
                    high = sum(1 for c in package.conflicts if c.severity == Severity.HIGH)
                    st.metric("🔴 High", high)
                with col2:
                    med = sum(1 for c in package.conflicts if c.severity == Severity.MED)
                    st.metric("🟡 Medium", med)
                with col3:
                    low = sum(1 for c in package.conflicts if c.severity == Severity.LOW)
                    st.metric("🟢 Low", low)

                st.divider()

                # Conflict list
                for conflict in sorted(package.conflicts, key=lambda c: {'high': 0, 'med': 1, 'low': 2}.get(c.severity.value, 3)):
                    severity_icon = {'high': '🔴', 'med': '🟡', 'low': '🟢'}.get(conflict.severity.value, '⚪')

                    with st.expander(f"{severity_icon} [{conflict.severity.value.upper()}] {conflict.type.value.replace('_', ' ').title()}"):
                        st.write(f"**Description:** {conflict.description}")
                        if conflict.suggested_resolution:
                            st.info(f"💡 **Suggested Resolution:** {conflict.suggested_resolution}")
                        if conflict.evidence:
                            render_evidence_viewer(conflict.evidence, "Related Evidence")
            else:
                st.success("✅ No conflicts detected!")

        # =================================================================
        # TAB 7: ELEMENT SCHEDULES (Legacy)
        # =================================================================
        with tab7:
            st.subheader("📊 Element Schedules")
            st.caption("Detected structural elements from visual analysis")

            # Show any detected elements from the pipeline
            # This is a simplified view based on pipeline data

            st.info("Element schedules are being migrated to the new BOQ format. See the BOQ Items tab for detailed element data.")

            # Quick summary from BOQ
            rcc_items = [b for b in package.boq if b.subsystem == 'rcc']
            if rcc_items:
                st.subheader("RCC Elements Summary")
                for item in rcc_items[:10]:
                    status_icon = {'computed': '✅', 'partial': '⚠️', 'unknown': '❓'}.get(item.qty_status.value, '❓')
                    qty_str = f"{item.qty} {item.unit}" if item.qty else "Qty unknown"
                    st.write(f"{status_icon} **{item.item_name}**: {qty_str}")

        # =================================================================
        # TAB 8: EXPORT
        # =================================================================
        with tab8:
            st.subheader("📥 Export Takeoff Package")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Excel Exports")

                # BOQ Excel
                boq_excel = export_boq_to_excel(package.boq, package.drawing.file_name)
                st.download_button(
                    "📥 BOQ.xlsx (CPWD Format)",
                    data=boq_excel,
                    file_name=f"BOQ_{package.drawing.file_name.replace('.pdf', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

                # Scope Excel
                scope_excel = export_scope_to_excel(package.scope, package.drawing.file_name)
                st.download_button(
                    "📥 ScopeChecklist.xlsx",
                    data=scope_excel,
                    file_name=f"ScopeChecklist_{package.drawing.file_name.replace('.pdf', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

                # Full package Excel
                full_excel = export_full_package_to_excel(package)
                st.download_button(
                    "📥 FullTakeoff.xlsx (All Sheets)",
                    data=full_excel,
                    file_name=f"FullTakeoff_{package.drawing.file_name.replace('.pdf', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col2:
                st.markdown("#### 📦 Package Exports")

                # CSV
                boq_csv = export_boq_to_csv(package.boq)
                st.download_button(
                    "📥 BOQ.csv",
                    data=boq_csv,
                    file_name=f"BOQ_{package.drawing.file_name.replace('.pdf', '')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # ZIP Package
                zip_package = export_takeoff_package_zip(package)
                st.download_button(
                    "📥 TakeoffPackage.zip (Complete)",
                    data=zip_package,
                    file_name=f"TakeoffPackage_{package.drawing.file_name.replace('.pdf', '')}.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )

                st.caption("ZIP includes: BOQ, Scope, Conflicts, Coverage + JSON dump")

                # JSON
                json_data = package.model_dump_json(indent=2)
                st.download_button(
                    "📥 EstimatePackage.json",
                    data=json_data,
                    file_name=f"EstimatePackage_{package.package_id}.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.divider()

            # Package summary
            st.markdown("#### 📋 Package Summary")
            stats = package.stats

            summary_data = {
                "Metric": [
                    "Drawing File",
                    "Package ID",
                    "Drawing Confidence",
                    "Total BOQ Items",
                    "  - Computed Qty",
                    "  - Unknown Qty",
                    "Total Scope Items",
                    "  - Detected",
                    "  - Missing",
                    "Conflicts",
                ],
                "Value": [
                    package.drawing.file_name,
                    package.package_id,
                    f"{package.drawing.confidence_overall:.0%}",
                    str(stats['total_boq_items']),
                    str(stats.get('boq_by_qty_status', {}).get('computed', 0)),
                    str(stats.get('boq_by_qty_status', {}).get('unknown', 0)),
                    str(stats['total_scope_items']),
                    str(stats.get('scope_by_status', {}).get('detected', 0)),
                    str(stats.get('scope_by_status', {}).get('missing', 0)),
                    str(stats['total_conflicts']),
                ]
            }
            st.dataframe(
                sanitize_df_for_streamlit(pd.DataFrame(summary_data)),
                use_container_width=True,
                hide_index=True
            )

    else:
        # No data - show welcome screen
        st.info("""
        ### 🏗️ Welcome to XBOQ Structural Takeoff

        **What this tool does:**
        - Extracts scope and quantities from structural drawings
        - Detects columns, footings, beams from foundation plans
        - Reads reinforcement schedules using table detection
        - Generates BOQ items with India-first terminology
        - Identifies missing information and conflicts

        **How to use:**
        1. Upload a PDF drawing (foundation plan or schedule sheet)
        2. Set the number of floors and storey height
        3. Enable **High Recall Mode** to capture all possible scope
        4. Click **Run Takeoff** to extract

        **Output includes:**
        - ✅ **Detected** items - Found in the drawing
        - 🔍 **Inferred** items - Typical for RCC buildings
        - ❌ **Missing** items - Expected but not found
        - ⚠️ **Conflicts** - Issues needing attention

        **NO PRICING** - This is a scope and quantity tool only.
        """)

        # Sample metrics for demo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BOQ Items", "-")
        with col2:
            st.metric("Scope Items", "-")
        with col3:
            st.metric("Conflicts", "-")
        with col4:
            st.metric("Confidence", "-")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
    <strong>XBOQ v2.0</strong> | India-First Preconstruction Platform<br>
    Structural Takeoff & Scope Extraction | IS 456:2000 Compliant<br>
    <em>NO PRICING - Pure scope and quantity extraction</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
