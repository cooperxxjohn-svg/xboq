"""
XBOQ Pre-Construction Engine - Streamlit QC Application
Complete BOQ generation and QC interface.

Tabs:
1. Rooms - Room detection and areas
2. Walls - Wall thickness and masonry BOQ
3. Openings - Door/window schedule
4. Finishes - Room finish mapping
5. BOQ Preview - Complete BOQ with CPWD mapping
6. Export - CSV, JSON, and reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
import logging
from typing import Optional, Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BOQ modules
from src.boq.engine import BOQEngine, BOQResult
from src.boq.wall_boq import WallBOQCalculator
from src.boq.finish_boq import FinishBOQCalculator
from src.boq.slab_boq import SlabBOQCalculator
from src.boq.steel_boq import SteelBOQCalculator
from src.boq.openings_boq import OpeningsBOQCalculator
from src.boq.confidence import ConfidenceCalculator, create_confidence_summary
from src.boq.export import BOQExporter, create_boq_csv_content
from src.boq.schema import BOQItem, BOQValidator, load_profile

# Import CPWD mapping
from src.rates.mapper import CPWDMapper, map_boq_to_cpwd, get_mapping_coverage

# Import openings detection
from src.openings.pipeline import OpeningsPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="XBOQ Pre-Construction Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
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
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .boq-table th {
        background-color: #f8f9fa;
    }
    .assumption-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if 'boq_result' not in st.session_state:
    st.session_state.boq_result = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'wall_mask' not in st.session_state:
    st.session_state.wall_mask = None
if 'rooms_data' not in st.session_state:
    st.session_state.rooms_data = None
if 'openings_data' not in st.session_state:
    st.session_state.openings_data = None
if 'ceiling_height' not in st.session_state:
    st.session_state.ceiling_height = 3000
if 'slab_thickness' not in st.session_state:
    st.session_state.slab_thickness = 125
if 'profile' not in st.session_state:
    st.session_state.profile = 'typical'
if 'cpwd_mapping_result' not in st.session_state:
    st.session_state.cpwd_mapping_result = None
if 'steel_slab_factor' not in st.session_state:
    st.session_state.steel_slab_factor = 90.0
if 'steel_beam_factor' not in st.session_state:
    st.session_state.steel_beam_factor = 150.0
if 'steel_column_factor' not in st.session_state:
    st.session_state.steel_column_factor = 200.0
if 'project_summary' not in st.session_state:
    st.session_state.project_summary = None
if 'summary_md_content' not in st.session_state:
    st.session_state.summary_md_content = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sqft(sqm: float) -> float:
    """Convert sqm to sqft."""
    return sqm * 10.764


def run_project_pipeline(project_dir: Path, profile: str = "typical") -> Dict[str, Any]:
    """Run the project pipeline and return summary."""
    try:
        # Import the project runner
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_project",
            str(Path(__file__).parent.parent / "scripts" / "run_project.py")
        )
        runner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner_module)

        # Run the pipeline
        runner = runner_module.ProjectRunner(
            project_dir=project_dir,
            output_dir=Path("out") / project_dir.name,
            profile=profile,
        )
        summary = runner.run()

        # Read the generated summary.md
        summary_md_path = runner.output_dir / "summary.md"
        summary_md = ""
        if summary_md_path.exists():
            with open(summary_md_path) as f:
                summary_md = f.read()

        return {
            "success": summary.status == "success",
            "summary": summary.to_dict(),
            "summary_md": summary_md,
            "output_dir": str(runner.output_dir),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "summary": None,
            "summary_md": f"# Error\n\nPipeline failed: {e}",
        }

def detect_walls_simple(gray: np.ndarray) -> np.ndarray:
    """Simple wall detection."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 10
    )
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    wall_mask = cv2.bitwise_or(horizontal, vertical)
    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)
    return wall_mask


def close_wall_gaps(wall_mask: np.ndarray) -> np.ndarray:
    """Close gaps in walls."""
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
    closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_h)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
    return closed


def boq_items_to_df(items: List[Any]) -> pd.DataFrame:
    """Convert BOQ items to DataFrame."""
    rows = []
    for item in items:
        if hasattr(item, 'item_code'):
            rows.append({
                "Item Code": item.item_code,
                "Description": getattr(item, 'description', ''),
                "Qty": getattr(item, 'qty', 0),
                "Unit": getattr(item, 'unit', ''),
                "Source": getattr(item, 'derived_from', ''),
                "Confidence": f"{getattr(item, 'confidence', 0.5):.0%}",
            })
    return pd.DataFrame(rows)


def get_confidence_color(conf: float) -> str:
    """Get confidence color class."""
    if conf >= 0.75:
        return "confidence-high"
    elif conf >= 0.50:
        return "confidence-medium"
    else:
        return "confidence-low"


def _render_project_summary():
    """Render the V1 project summary in Streamlit."""
    summary = st.session_state.project_summary
    summary_md = st.session_state.summary_md_content

    if not summary:
        return

    # Extract V1 summary sections
    openings = summary.get("openings", {})
    finishes = summary.get("finishes", {})

    # Quick stats in columns
    st.subheader("📊 V1 Project Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏠 Rooms", summary.get("rooms_count", 0))

    total_doors = openings.get("total_doors", 0)
    total_windows = openings.get("total_windows", 0)
    col2.metric("🚪 Doors", total_doors)
    col3.metric("🪟 Windows", total_windows)
    col4.metric("📋 BOQ Items", summary.get("boq_lines_count", 0))

    cpwd_pct = summary.get("cpwd_coverage_percent", 0)
    if cpwd_pct >= 80:
        cpwd_icon = "🟢"
    elif cpwd_pct >= 50:
        cpwd_icon = "🟡"
    else:
        cpwd_icon = "🔴"
    col5.metric(f"{cpwd_icon} CPWD", f"{cpwd_pct:.0f}%")

    # Total area metric
    total_area = summary.get("total_area_sqm", 0)
    st.info(f"**Total Area:** {total_area:.1f} sqm ({sqft(total_area):.0f} sqft)")

    # Quantities table
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎨 Finishes Summary**")

        # Floor by type
        floor_by_type = finishes.get("floor_by_type", {})
        if floor_by_type:
            floor_data = [
                {"Type": ftype.replace("_", " ").title(), "Area (sqm)": f"{area:.1f}", "Area (sqft)": f"{sqft(area):.0f}"}
                for ftype, area in floor_by_type.items()
            ]
            st.dataframe(pd.DataFrame(floor_data), use_container_width=True, hide_index=True)

        # Wall finishes
        wall_tile = finishes.get("wall_tile_area_sqm", 0)
        wall_dado = finishes.get("wall_dado_area_sqm", 0)
        wall_paint = finishes.get("wall_paint_area_sqm", 0)
        ceiling = finishes.get("ceiling_paint_area_sqm", 0)
        skirting = finishes.get("skirting_length_m", 0)

        wall_data = [
            {"Item": "Wall Tiles (Toilet)", "Qty": f"{wall_tile:.1f} sqm", "Notes": "Up to 2100mm"},
            {"Item": "Wall Dado (Kitchen)", "Qty": f"{wall_dado:.1f} sqm", "Notes": "600mm height"},
            {"Item": "Wall Paint", "Qty": f"{wall_paint:.1f} sqm", "Notes": "Interior emulsion"},
            {"Item": "Ceiling Paint", "Qty": f"{ceiling:.1f} sqm", "Notes": ""},
            {"Item": "Skirting", "Qty": f"{skirting:.1f} m", "Notes": "100mm height"},
        ]
        st.dataframe(pd.DataFrame(wall_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**🚪 Openings Schedule**")

        # Door schedule
        door_schedule = openings.get("door_schedule", [])
        if door_schedule:
            door_df = pd.DataFrame(door_schedule)
            door_df = door_df.rename(columns={
                "tag": "Tag", "width_mm": "Width", "height_mm": "Height",
                "quantity": "Qty", "material": "Material"
            })
            st.dataframe(door_df, use_container_width=True, hide_index=True)

        # Window schedule
        window_schedule = openings.get("window_schedule", [])
        if window_schedule:
            st.markdown("**Windows:**")
            win_df = pd.DataFrame(window_schedule)
            win_df = win_df.rename(columns={
                "tag": "Tag", "width_mm": "Width", "height_mm": "Height",
                "quantity": "Qty", "material": "Material"
            })
            st.dataframe(win_df, use_container_width=True, hide_index=True)

        # Confidence breakdown
        st.markdown("**📊 Confidence Breakdown**")
        conf_data = [
            {"Level": "🟢 High (≥75%)", "Count": summary.get("high_confidence_count", 0)},
            {"Level": "🟡 Medium (50-75%)", "Count": summary.get("medium_confidence_count", 0)},
            {"Level": "🔴 Low (<50%)", "Count": summary.get("low_confidence_count", 0)},
        ]
        st.dataframe(pd.DataFrame(conf_data), use_container_width=True, hide_index=True)

    # Warnings and Next Actions
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        warnings = summary.get("warnings", [])
        if warnings:
            with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=True):
                for w in warnings:
                    st.write(f"- {w}")
        else:
            st.success("✅ No warnings")

    with col2:
        next_actions = summary.get("next_actions", [])
        if next_actions:
            with st.expander("✅ Next Actions", expanded=True):
                for i, action in enumerate(next_actions, 1):
                    st.write(f"{i}. {action}")

    # Full summary markdown
    if summary_md:
        with st.expander("📄 Full Summary Report (Markdown)"):
            st.markdown(summary_md)

            # Download button for summary.md
            st.download_button(
                "📥 Download summary.md",
                summary_md,
                "summary.md",
                "text/markdown",
            )

    # Clear button
    if st.button("🗑️ Clear Summary"):
        st.session_state.project_summary = None
        st.session_state.summary_md_content = None
        st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🏗️ XBOQ Pre-Construction Engine</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">India-First BOQ Generation | Walls • Finishes • Openings • Structural</p>',
        unsafe_allow_html=True
    )

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Project Settings")

        # Profile selection
        profile_options = ["conservative", "typical", "premium"]
        profile_descriptions = {
            "conservative": "Lower estimates, fewer assumptions",
            "typical": "Standard Indian residential (default)",
            "premium": "Higher-end specifications",
        }
        profile = st.selectbox(
            "Estimation Profile",
            options=profile_options,
            index=profile_options.index(st.session_state.profile),
            help="Affects steel factors, wastage, and finish specs",
            format_func=lambda x: f"{x.title()} - {profile_descriptions[x]}"
        )
        st.session_state.profile = profile

        # Load profile defaults
        profile_config = load_profile(profile)

        st.divider()
        st.subheader("📏 Dimensions")

        # Ceiling height
        ceiling_height = st.number_input(
            "Ceiling Height (mm)",
            min_value=2400, max_value=5000,
            value=st.session_state.ceiling_height, step=100,
            help="Default floor-to-ceiling height"
        )
        st.session_state.ceiling_height = ceiling_height

        # Slab thickness
        slab_thickness = st.number_input(
            "Slab Thickness (mm)",
            min_value=100, max_value=250,
            value=st.session_state.slab_thickness, step=25,
            help="RCC slab thickness"
        )
        st.session_state.slab_thickness = slab_thickness

        st.divider()
        st.subheader("🔩 Structural")

        # Concrete grade
        concrete_grade = st.selectbox(
            "Concrete Grade",
            options=["M20", "M25", "M30"],
            index=1,
            help="RCC concrete grade"
        )

        # Steel grade
        steel_grade = st.selectbox(
            "Steel Grade",
            options=["Fe415", "Fe500", "Fe500D"],
            index=1,
            help="TMT steel grade"
        )

        # Steel factor overrides (expandable)
        with st.expander("🔧 Steel Factors (kg/m³)", expanded=False):
            st.caption("Override default factors from profile")

            # Get default factors from profile
            multiplier = profile_config.get("steel_factor_multiplier", 1.0)
            default_slab = 90.0 * multiplier
            default_beam = 150.0 * multiplier
            default_column = 200.0 * multiplier

            steel_slab_factor = st.number_input(
                "Slab",
                min_value=50.0, max_value=150.0,
                value=default_slab, step=5.0,
                help="Steel kg per m³ of slab concrete"
            )
            st.session_state.steel_slab_factor = steel_slab_factor

            steel_beam_factor = st.number_input(
                "Beam",
                min_value=100.0, max_value=250.0,
                value=default_beam, step=5.0,
                help="Steel kg per m³ of beam concrete"
            )
            st.session_state.steel_beam_factor = steel_beam_factor

            steel_column_factor = st.number_input(
                "Column",
                min_value=150.0, max_value=300.0,
                value=default_column, step=5.0,
                help="Steel kg per m³ of column concrete"
            )
            st.session_state.steel_column_factor = steel_column_factor

            st.caption(f"Profile multiplier: {multiplier:.2f}x")

        st.divider()

        # File uploads
        st.subheader("📤 Upload Files")

        uploaded_image = st.file_uploader(
            "Floor Plan (PNG/JPG/PDF)",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload architectural floor plan"
        )

        rooms_file = st.file_uploader(
            "Rooms Data (JSON) - Optional",
            type=['json'],
            help="Pre-computed room detection results"
        )

        openings_file = st.file_uploader(
            "Openings Data (JSON) - Optional",
            type=['json'],
            help="Pre-computed door/window detection"
        )

        if uploaded_image:
            st.success(f"✓ {uploaded_image.name}")

            if st.button("🚀 Generate BOQ", type="primary", use_container_width=True):
                with st.spinner("Processing plan and generating BOQ..."):
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_image.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_image.getvalue())
                        tmp_path = Path(tmp.name)

                    # Load image
                    image = cv2.imread(str(tmp_path))
                    if image is None:
                        st.error("Could not load image")
                        return

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    wall_mask = detect_walls_simple(gray)
                    wall_mask_closed = close_wall_gaps(wall_mask)

                    # Load rooms
                    rooms_data = []
                    if rooms_file:
                        data = json.load(rooms_file)
                        rooms_data = data.get("rooms", data) if isinstance(data, dict) else data
                    else:
                        # Generate placeholder rooms
                        st.warning("No rooms data - using placeholder")
                        rooms_data = [{
                            "id": "R001",
                            "label": "Living",
                            "area_sqm": 25.0,
                            "perimeter_m": 20.0,
                        }]

                    # Load or detect openings
                    openings_data = []
                    if openings_file:
                        data = json.load(openings_file)
                        openings_data = data.get("openings", [])
                    else:
                        # Run openings detection
                        pipeline = OpeningsPipeline()
                        result = pipeline.run(
                            image=image,
                            wall_mask=wall_mask,
                            wall_mask_closed=wall_mask_closed,
                            rooms=rooms_data,
                            plan_id=uploaded_image.name,
                        )
                        # Convert to dicts
                        for door in result.doors:
                            openings_data.append({
                                "id": door.id,
                                "type": door.type,
                                "tag": door.tag,
                                "width_m": door.width_m,
                                "height_m": door.height_m,
                                "room_left_id": door.room_left_id,
                                "room_right_id": door.room_right_id,
                                "confidence": door.confidence,
                                "bbox": list(door.bbox),
                            })
                        for window in result.windows:
                            openings_data.append({
                                "id": window.id,
                                "type": window.type,
                                "tag": window.tag,
                                "width_m": window.width_m,
                                "height_m": window.height_m,
                                "room_left_id": window.room_left_id,
                                "confidence": window.confidence,
                                "bbox": list(window.bbox),
                            })

                    # Create output directory
                    output_dir = Path(tempfile.mkdtemp())

                    # Run BOQ engine
                    engine = BOQEngine(
                        output_dir=output_dir,
                        ceiling_height_mm=ceiling_height,
                        slab_thickness_mm=slab_thickness,
                        concrete_grade=concrete_grade,
                        steel_grade=steel_grade,
                    )

                    boq_result = engine.run(
                        image=image,
                        wall_mask=wall_mask,
                        rooms=rooms_data,
                        openings=openings_data,
                        project_id=uploaded_image.name,
                    )

                    # Store in session
                    st.session_state.image = image
                    st.session_state.wall_mask = wall_mask
                    st.session_state.rooms_data = rooms_data
                    st.session_state.openings_data = openings_data
                    st.session_state.boq_result = boq_result

                    st.success("✅ BOQ generation complete!")
                    st.rerun()

    # =========================================================================
    # MAIN CONTENT
    # =========================================================================

    if st.session_state.boq_result is None:
        st.info("👆 Upload a floor plan and click 'Generate BOQ' to start, or use 'Run Project' below")

        # =====================================================================
        # RUN PROJECT SECTION
        # =====================================================================
        st.divider()
        st.subheader("🚀 Run Project Pipeline")

        col1, col2 = st.columns([3, 1])

        with col1:
            project_path = st.text_input(
                "Project Directory",
                value="data/projects/sample_project",
                help="Path to project folder containing plan.png and optional rooms.json, openings.json",
            )

        with col2:
            st.write("")  # Spacing
            st.write("")
            run_clicked = st.button("▶️ Run Project", type="primary", use_container_width=True)

        if run_clicked:
            project_dir = Path(project_path)
            if not project_dir.exists():
                st.error(f"Project directory not found: {project_dir}")
            else:
                with st.spinner("🔄 Running full pipeline... This may take a moment."):
                    result = run_project_pipeline(
                        project_dir=project_dir,
                        profile=st.session_state.profile,
                    )

                    st.session_state.project_summary = result.get("summary")
                    st.session_state.summary_md_content = result.get("summary_md")

                    if result.get("success"):
                        st.success(f"✅ Pipeline complete! Output: {result.get('output_dir')}")
                    else:
                        st.warning(f"⚠️ Pipeline completed with issues")

                    st.rerun()

        # Show project summary if available
        if st.session_state.project_summary:
            st.divider()
            _render_project_summary()

        # =====================================================================
        # ABOUT SECTION
        # =====================================================================
        st.divider()
        with st.expander("📖 About XBOQ"):
            st.markdown("""
            **XBOQ Pre-Construction Engine** generates Bill of Quantities (BOQ) from floor plans.

            **Modules:**
            - **Walls**: Brickwork volume, plaster area
            - **Finishes**: Floor tiles, wall paint, ceiling, skirting
            - **Openings**: Door/window schedules with hardware
            - **Structural**: Slab concrete, reinforcement steel

            **Features:**
            - India-first: Uses Indian construction terms (CM 1:6, TMT Fe500, etc.)
            - Confidence scoring: Green/yellow/red for measured vs. assumed
            - Assumption tracking: All defaults logged
            - Export: CSV, JSON, Markdown report

            **Project Structure:**
            ```
            data/projects/<project_id>/
            ├── plan.png (or plan.jpg)
            ├── rooms.json (optional)
            ├── openings.json (optional)
            └── config.json (optional overrides)
            ```
            """)
        return

    result = st.session_state.boq_result

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📋 BOQ Items", result.totals.get("total_items", 0))
    with col2:
        st.metric("🧱 Brick Volume", f"{result.totals.get('brick_volume_cum', 0):.1f} cum")
    with col3:
        st.metric("🏗️ Slab Concrete", f"{result.totals.get('slab_concrete_cum', 0):.1f} cum")
    with col4:
        st.metric("🔩 Steel", f"{result.totals.get('steel_mt', 0):.2f} MT")
    with col5:
        conf = result.totals.get("overall_confidence", 0.5)
        conf_class = get_confidence_color(conf)
        st.metric("📊 Confidence", f"{conf:.0%}")

    # Tabs
    tabs = st.tabs([
        "🏠 Rooms",
        "🧱 Walls",
        "🚪 Openings",
        "🎨 Finishes",
        "📋 BOQ Preview",
        "📥 Export"
    ])

    # --- TAB 1: ROOMS ---
    with tabs[0]:
        st.subheader("Room Detection & Areas")

        rooms = st.session_state.rooms_data or []

        if rooms:
            rooms_df = pd.DataFrame([
                {
                    "Room": r.get("label", "Unknown"),
                    "Area (sqm)": f"{r.get('area_sqm', 0):.1f}",
                    "Perimeter (m)": f"{r.get('perimeter_m', 0):.1f}",
                }
                for r in rooms
            ])
            st.dataframe(rooms_df, use_container_width=True)

            # Totals
            total_area = sum(r.get("area_sqm", 0) for r in rooms)
            st.info(f"**Total Area:** {total_area:.1f} sqm ({total_area * 10.764:.0f} sqft)")
        else:
            st.warning("No room data available")

    # --- TAB 2: WALLS ---
    with tabs[1]:
        st.subheader("Wall BOQ - Masonry & Plaster")

        if result.wall_result:
            wr = result.wall_result

            col1, col2, col3 = st.columns(3)
            col1.metric("Wall Length", f"{wr.total_wall_length_m:.1f} m")
            col2.metric("Brick Volume", f"{wr.total_brick_volume_cum:.2f} cum")
            col3.metric("Plaster Area", f"{wr.total_plaster_area_sqm:.1f} sqm")

            st.divider()

            # Thickness breakdown
            st.write("**Wall Thickness Distribution:**")
            for thickness, length in wr.thickness_clusters.items():
                st.write(f"- {thickness}mm walls: {length:.1f} m")

            st.divider()

            # BOQ items
            st.write("**Wall BOQ Items:**")
            df = boq_items_to_df(wr.boq_items)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No wall BOQ data")

    # --- TAB 3: OPENINGS ---
    with tabs[2]:
        st.subheader("Openings Schedule - Doors & Windows")

        if result.openings_result:
            opr = result.openings_result

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Doors", opr.totals.get("total_doors", 0))
            col2.metric("Total Windows", opr.totals.get("total_windows", 0))
            col3.metric("Total Area", f"{opr.totals.get('total_door_area_sqm', 0) + opr.totals.get('total_window_area_sqm', 0):.1f} sqm")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Door Schedule:**")
                if opr.door_schedule:
                    door_df = pd.DataFrame([
                        {
                            "Tag": e.tag,
                            "Size": f"{e.width_mm}x{e.height_mm}",
                            "Qty": e.quantity,
                            "Material": e.material,
                        }
                        for e in opr.door_schedule
                    ])
                    st.dataframe(door_df, use_container_width=True)

            with col2:
                st.write("**Window Schedule:**")
                if opr.window_schedule:
                    window_df = pd.DataFrame([
                        {
                            "Tag": e.tag,
                            "Size": f"{e.width_mm}x{e.height_mm}",
                            "Qty": e.quantity,
                            "Material": e.material,
                        }
                        for e in opr.window_schedule
                    ])
                    st.dataframe(window_df, use_container_width=True)
        else:
            st.info("No openings data")

    # --- TAB 4: FINISHES ---
    with tabs[3]:
        st.subheader("Room Finishes BOQ")

        if result.finish_result:
            fr = result.finish_result

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Floor Area", f"{fr.totals.get('total_floor_sqm', 0):.1f} sqm")
            col2.metric("Wall Paint", f"{fr.totals.get('total_wall_sqm', 0):.1f} sqm")
            col3.metric("Ceiling", f"{fr.totals.get('total_ceiling_sqm', 0):.1f} sqm")
            col4.metric("Skirting", f"{fr.totals.get('total_skirting_m', 0):.1f} m")

            st.divider()

            # Room-wise breakdown
            st.write("**Room-wise Finish Areas:**")
            room_df = pd.DataFrame([
                {
                    "Room": rr.room_label,
                    "Type": rr.room_type,
                    "Floor (sqm)": f"{rr.floor_area_sqm:.1f}",
                    "Wall (sqm)": f"{rr.wall_area_sqm:.1f}",
                    "Ceiling (sqm)": f"{rr.ceiling_area_sqm:.1f}",
                    "Wet Area": "✓" if rr.is_wet_area else "",
                }
                for rr in fr.room_results
            ])
            st.dataframe(room_df, use_container_width=True)

            st.divider()

            # Consolidated BOQ
            st.write("**Finish BOQ Items (Consolidated):**")
            df = boq_items_to_df(fr.boq_items)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No finish BOQ data")

    # --- TAB 5: BOQ PREVIEW ---
    with tabs[4]:
        st.subheader("Complete BOQ Preview")

        # Confidence summary
        conf_summary = create_confidence_summary(result.all_boq_items)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Items", conf_summary.get("count", 0))
        col2.metric("🟢 High Conf", f"{conf_summary.get('high_count', 0)} ({conf_summary.get('high_percent', 0):.0f}%)")
        col3.metric("🟡 Medium Conf", f"{conf_summary.get('medium_count', 0)} ({conf_summary.get('medium_percent', 0):.0f}%)")
        col4.metric("🔴 Low Conf", f"{conf_summary.get('low_count', 0)} ({conf_summary.get('low_percent', 0):.0f}%)")

        st.divider()

        # CPWD Mapping Coverage Section
        st.subheader("📊 CPWD Mapping Coverage")

        # Convert BOQ items to dicts for CPWD mapping
        boq_dicts = []
        for item in result.all_boq_items:
            boq_dicts.append({
                "item_code": getattr(item, 'item_code', ''),
                "description": getattr(item, 'description', ''),
                "qty": getattr(item, 'qty', 0),
                "unit": getattr(item, 'unit', ''),
                "derived_from": getattr(item, 'derived_from', ''),
                "confidence": getattr(item, 'confidence', 0.5),
                "category": getattr(item, 'category', None),
            })

        # Get CPWD mapping coverage
        cpwd_coverage = get_mapping_coverage(boq_dicts)
        st.session_state.cpwd_mapping_result = map_boq_to_cpwd(boq_dicts)

        col1, col2, col3 = st.columns(3)
        coverage_pct = cpwd_coverage.get("coverage_percent", 0)

        # Color based on coverage
        if coverage_pct >= 80:
            coverage_color = "🟢"
        elif coverage_pct >= 50:
            coverage_color = "🟡"
        else:
            coverage_color = "🔴"

        col1.metric(
            f"{coverage_color} CPWD Mapped",
            f"{cpwd_coverage.get('mapped', 0)} items",
            f"{coverage_pct:.0f}%"
        )
        col2.metric(
            "⚠️ Unmapped",
            f"{cpwd_coverage.get('unmapped', 0)} items",
        )
        col3.metric(
            "Total Items",
            cpwd_coverage.get("total_items", 0),
        )

        # Show unmapped codes
        if cpwd_coverage.get("unmapped_codes"):
            with st.expander(f"🔍 Unmapped Item Codes ({len(cpwd_coverage['unmapped_codes'])})"):
                for code in cpwd_coverage["unmapped_codes"]:
                    st.write(f"- `{code}`")
                st.caption("These items need CPWD mapping in rates/cpwd_mapping.csv")

        st.divider()

        # Complete BOQ table
        st.subheader("📋 BOQ Items")
        all_df = boq_items_to_df(result.all_boq_items)

        if not all_df.empty:
            # Add category column
            categories = []
            for item in result.all_boq_items:
                derived = getattr(item, 'derived_from', '').lower()
                if 'wall' in derived or 'plaster' in derived:
                    categories.append("Masonry")
                elif 'finish' in derived or 'room' in derived:
                    categories.append("Finishes")
                elif 'opening' in derived or 'door' in derived:
                    categories.append("Openings")
                elif 'slab' in derived or 'concrete' in derived:
                    categories.append("Structural")
                elif 'steel' in derived:
                    categories.append("Steel")
                else:
                    categories.append("Other")

            all_df.insert(0, "Category", categories)

            # Filter by category
            categories_list = all_df["Category"].unique().tolist()
            selected_cats = st.multiselect(
                "Filter by Category",
                options=categories_list,
                default=categories_list,
            )

            filtered_df = all_df[all_df["Category"].isin(selected_cats)]
            st.dataframe(filtered_df, use_container_width=True, height=400)

        # Profile and Settings Info
        with st.expander("⚙️ Active Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Profile:** {st.session_state.profile.title()}")
                st.write(f"**Ceiling Height:** {st.session_state.ceiling_height} mm")
                st.write(f"**Slab Thickness:** {st.session_state.slab_thickness} mm")
            with col2:
                st.write(f"**Steel Factor (Slab):** {st.session_state.steel_slab_factor:.0f} kg/m³")
                st.write(f"**Steel Factor (Beam):** {st.session_state.steel_beam_factor:.0f} kg/m³")
                st.write(f"**Steel Factor (Column):** {st.session_state.steel_column_factor:.0f} kg/m³")

        # Assumptions
        if result.assumptions:
            with st.expander(f"📝 Assumptions Made ({len(result.assumptions)})"):
                for i, assumption in enumerate(result.assumptions[:50], 1):
                    st.write(f"{i}. {assumption}")
                if len(result.assumptions) > 50:
                    st.write(f"*... and {len(result.assumptions) - 50} more*")

    # --- TAB 6: EXPORT ---
    with tabs[5]:
        st.subheader("Export BOQ Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📊 BOQ CSV Exports**")

            # Main BOQ
            if result.all_boq_items:
                csv_content = create_boq_csv_content(result.all_boq_items)
                st.download_button(
                    "📄 Complete BOQ (CSV)",
                    csv_content,
                    "boq_quantities.csv",
                    "text/csv",
                )

            # Wall BOQ
            if result.wall_result:
                wall_csv = create_boq_csv_content(result.wall_result.boq_items)
                st.download_button(
                    "📄 Wall BOQ (CSV)",
                    wall_csv,
                    "wall_boq.csv",
                    "text/csv",
                )

            # Finishes BOQ
            if result.finish_result:
                finish_csv = create_boq_csv_content(result.finish_result.boq_items)
                st.download_button(
                    "📄 Finishes BOQ (CSV)",
                    finish_csv,
                    "finishes_boq.csv",
                    "text/csv",
                )

        with col2:
            st.markdown("**🏛️ CPWD Mapped Exports**")

            # CPWD Mapped BOQ
            if st.session_state.cpwd_mapping_result:
                mapping_result = st.session_state.cpwd_mapping_result

                # Create CPWD mapped CSV content
                cpwd_rows = []
                cpwd_rows.append(",".join([
                    "item_code", "description", "qty", "unit", "derived_from",
                    "confidence", "cpwd_item_no", "cpwd_description", "cpwd_unit", "is_mapped"
                ]))
                for item in mapping_result.mapped_items:
                    cpwd_rows.append(",".join([
                        item.item_code,
                        f'"{item.description}"',
                        f"{item.qty:.2f}",
                        item.unit,
                        item.derived_from,
                        f"{item.confidence:.2f}",
                        item.cpwd_item_no or "",
                        f'"{item.cpwd_description or ""}"',
                        item.cpwd_unit or "",
                        "Yes" if item.is_mapped else "No",
                    ]))
                cpwd_csv = "\n".join(cpwd_rows)

                st.download_button(
                    "📄 BOQ with CPWD Map (CSV)",
                    cpwd_csv,
                    "boq_with_cpwd_map.csv",
                    "text/csv",
                )

                # Coverage stats
                st.caption(f"✓ {mapping_result.mapped_count} mapped, ⚠ {mapping_result.unmapped_count} unmapped")
                st.caption(f"Coverage: {mapping_result.coverage_percent:.0f}%")

        with col3:
            st.markdown("**📋 Other Formats**")

            # JSON export
            if result.all_boq_items:
                export_data = {
                    "project_id": result.project_id,
                    "profile": st.session_state.profile,
                    "settings": {
                        "ceiling_height_mm": st.session_state.ceiling_height,
                        "slab_thickness_mm": st.session_state.slab_thickness,
                        "steel_factors": {
                            "slab_kg_per_m3": st.session_state.steel_slab_factor,
                            "beam_kg_per_m3": st.session_state.steel_beam_factor,
                            "column_kg_per_m3": st.session_state.steel_column_factor,
                        },
                    },
                    "totals": result.totals,
                    "items": [
                        {
                            "item_code": getattr(i, 'item_code', ''),
                            "description": getattr(i, 'description', ''),
                            "qty": getattr(i, 'qty', 0),
                            "unit": getattr(i, 'unit', ''),
                            "derived_from": getattr(i, 'derived_from', ''),
                            "confidence": getattr(i, 'confidence', 0.5),
                        }
                        for i in result.all_boq_items
                    ],
                    "assumptions": result.assumptions,
                    "cpwd_coverage": {
                        "mapped": st.session_state.cpwd_mapping_result.mapped_count if st.session_state.cpwd_mapping_result else 0,
                        "unmapped": st.session_state.cpwd_mapping_result.unmapped_count if st.session_state.cpwd_mapping_result else 0,
                        "coverage_percent": st.session_state.cpwd_mapping_result.coverage_percent if st.session_state.cpwd_mapping_result else 0,
                    },
                }

                st.download_button(
                    "📄 Complete Data (JSON)",
                    json.dumps(export_data, indent=2),
                    "boq_complete.json",
                    "application/json",
                )

            # Assumptions
            if result.assumptions:
                assumptions_data = {
                    "count": len(result.assumptions),
                    "assumptions": result.assumptions,
                }
                st.download_button(
                    "📄 Assumptions (JSON)",
                    json.dumps(assumptions_data, indent=2),
                    "assumptions.json",
                    "application/json",
                )

        # Confidence heatmap
        st.divider()
        st.markdown("**🗺️ Confidence Heatmap**")

        if result.confidence_heatmap and result.confidence_heatmap.image is not None:
            heatmap_rgb = cv2.cvtColor(result.confidence_heatmap.image, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, use_container_width=True)

            # Download heatmap
            _, buffer = cv2.imencode('.png', result.confidence_heatmap.image)
            st.download_button(
                "📷 Download Heatmap (PNG)",
                buffer.tobytes(),
                "confidence_heatmap.png",
                "image/png",
            )


if __name__ == "__main__":
    main()
