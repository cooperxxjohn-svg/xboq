"""
XBOQ Openings & Finishes QC Application
Streamlit UI for door/window detection and finish takeoff.

Features:
- Upload floor plan PDF/image
- Toggle layers: rooms / walls / doors / windows
- Click to edit openings (type, tag, size)
- Manual add/delete openings
- Adjustable ceiling height and deduction rules
- Export corrected schedules and BOQ
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

from src.openings.pipeline import OpeningsPipeline, OpeningsResult
from src.openings.detect_doors import DoorDetector, DetectedDoor
from src.openings.detect_windows import WindowDetector, DetectedWindow
from src.openings.export import OpeningsExporter
from src.finishes.calculator import FinishCalculator, RoomFinishes
from src.finishes.export import FinishExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="XBOQ - Openings & Finishes",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        text-align: center;
    }
    .door-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .window-badge {
        background-color: #007bff;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .vent-badge {
        background-color: #6f42c1;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if 'openings_result' not in st.session_state:
    st.session_state.openings_result = None
if 'room_finishes' not in st.session_state:
    st.session_state.room_finishes = None
if 'rooms_data' not in st.session_state:
    st.session_state.rooms_data = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'wall_mask' not in st.session_state:
    st.session_state.wall_mask = None
if 'selected_opening' not in st.session_state:
    st.session_state.selected_opening = None
if 'ceiling_height' not in st.session_state:
    st.session_state.ceiling_height = 3000
if 'show_doors' not in st.session_state:
    st.session_state.show_doors = True
if 'show_windows' not in st.session_state:
    st.session_state.show_windows = True
if 'show_rooms' not in st.session_state:
    st.session_state.show_rooms = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_walls(gray: np.ndarray) -> np.ndarray:
    """Simple wall detection."""
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
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


def draw_overlay(
    image: np.ndarray,
    result: OpeningsResult,
    show_doors: bool = True,
    show_windows: bool = True,
    show_rooms: bool = True,
    rooms_data: Optional[List[Dict]] = None,
) -> np.ndarray:
    """Draw overlay with detected elements."""
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    # Draw rooms
    if show_rooms and rooms_data:
        for room in rooms_data:
            vertices = room.get("polygon_vertices", room.get("vertices", []))
            if vertices and len(vertices) >= 3:
                pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, (200, 200, 200), 1)

                centroid = room.get("centroid")
                if centroid:
                    label = room.get("label", "")
                    cv2.putText(
                        overlay, label,
                        (int(centroid[0]) - 20, int(centroid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1
                    )

    # Draw doors
    if show_doors and result:
        for door in result.doors:
            x1, y1, x2, y2 = door.bbox
            color = (0, 255, 0)  # Green
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = door.tag if door.tag else door.id
            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # Draw windows
    if show_windows and result:
        for window in result.windows:
            x1, y1, x2, y2 = window.bbox
            if window.type == "ventilator":
                color = (255, 0, 255)  # Magenta
            else:
                color = (255, 0, 0)  # Blue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.line(overlay, (x1, (y1+y2)//2), (x2, (y1+y2)//2), color, 1)
            label = window.tag if window.tag else window.id
            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    return overlay


def openings_to_df(result: OpeningsResult) -> pd.DataFrame:
    """Convert openings to DataFrame."""
    rows = []

    for door in result.doors:
        rows.append({
            "ID": door.id,
            "Type": "Door",
            "Tag": door.tag or "",
            "Subtype": door.type,
            "Width (mm)": int(door.width_m * 1000) if door.width_m else "",
            "Height (mm)": int(door.height_m * 1000) if door.height_m else "",
            "Room Left": door.room_left_id or "",
            "Room Right": door.room_right_id or "",
            "Confidence": f"{door.confidence:.2f}",
        })

    for window in result.windows:
        rows.append({
            "ID": window.id,
            "Type": "Window" if window.type != "ventilator" else "Ventilator",
            "Tag": window.tag or "",
            "Subtype": window.type,
            "Width (mm)": int(window.width_m * 1000) if window.width_m else "",
            "Height (mm)": int(window.height_m * 1000) if window.height_m else "",
            "Room Left": window.room_left_id or "",
            "Room Right": "",
            "Confidence": f"{window.confidence:.2f}",
        })

    return pd.DataFrame(rows)


def finishes_to_df(room_finishes: List[RoomFinishes]) -> pd.DataFrame:
    """Convert finishes to DataFrame."""
    rows = []

    for room in room_finishes:
        rows.append({
            "Room": room.room_label,
            "Category": room.room_category,
            "Floor (sqm)": f"{room.floor_area_sqm:.2f}",
            "Skirting (m)": f"{room.skirting_length_m:.2f}",
            "Wall (sqm)": f"{room.wall_area_sqm:.2f}",
            "Ceiling (sqm)": f"{room.ceiling_area_sqm:.2f}",
            "Openings": room.openings_count,
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🚪 XBOQ Openings & Finishes</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Door & Window Detection | Finish Takeoff | QC Editor</p>',
        unsafe_allow_html=True
    )

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("📐 Settings")

        # Ceiling height
        ceiling_height = st.number_input(
            "Ceiling Height (mm)",
            min_value=2400, max_value=5000, value=st.session_state.ceiling_height, step=100,
            help="Default ceiling height for wall area calculations"
        )
        st.session_state.ceiling_height = ceiling_height

        st.divider()

        # Layer toggles
        st.subheader("🎨 Display Layers")
        st.session_state.show_rooms = st.toggle("Show Rooms", value=st.session_state.show_rooms)
        st.session_state.show_doors = st.toggle("Show Doors", value=st.session_state.show_doors)
        st.session_state.show_windows = st.toggle("Show Windows", value=st.session_state.show_windows)

        st.divider()

        # File upload
        st.subheader("📤 Upload Plan")
        uploaded_file = st.file_uploader(
            "Floor Plan (PNG/JPG/PDF)",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload architectural floor plan"
        )

        # Optional rooms JSON
        rooms_file = st.file_uploader(
            "Rooms Data (JSON) - Optional",
            type=['json'],
            help="Pre-computed room data from pipeline"
        )

        if uploaded_file:
            st.success(f"✓ {uploaded_file.name}")

            if st.button("🚀 Detect Openings", type="primary", use_container_width=True):
                with st.spinner("Detecting doors and windows..."):
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

                    # Load image
                    image = cv2.imread(str(tmp_path))
                    if image is None:
                        st.error("Could not load image")
                        return

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Detect walls
                    wall_mask = detect_walls(gray)
                    wall_mask_closed = close_wall_gaps(wall_mask)

                    # Load rooms if provided
                    rooms_data = None
                    if rooms_file:
                        rooms_data = json.load(rooms_file)
                        if "rooms" in rooms_data:
                            rooms_data = rooms_data["rooms"]

                    # Run detection
                    pipeline = OpeningsPipeline()
                    result = pipeline.run(
                        image=image,
                        wall_mask=wall_mask,
                        wall_mask_closed=wall_mask_closed,
                        rooms=rooms_data,
                        plan_id=uploaded_file.name,
                    )

                    # Store in session
                    st.session_state.image = image
                    st.session_state.wall_mask = wall_mask
                    st.session_state.openings_result = result
                    st.session_state.rooms_data = rooms_data

                    # Calculate finishes if rooms available
                    if rooms_data:
                        calc = FinishCalculator(ceiling_height_mm=ceiling_height)

                        # Build openings list
                        openings_list = []
                        for door in result.doors:
                            openings_list.append({
                                "id": door.id,
                                "type": door.type,
                                "width_m": door.width_m,
                                "height_m": door.height_m,
                                "room_left_id": door.room_left_id,
                                "room_right_id": door.room_right_id,
                            })
                        for window in result.windows:
                            openings_list.append({
                                "id": window.id,
                                "type": window.type,
                                "width_m": window.width_m,
                                "height_m": window.height_m,
                                "room_left_id": window.room_left_id,
                            })

                        room_finishes = calc.calculate_all_rooms(
                            rooms_data,
                            openings=openings_list,
                        )
                        st.session_state.room_finishes = room_finishes

                    st.success("✅ Detection complete!")
                    st.rerun()

    # =========================================================================
    # MAIN CONTENT
    # =========================================================================

    if st.session_state.openings_result is None:
        st.info("👆 Upload a floor plan to get started")

        # Show sample instructions
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Upload a floor plan** (PNG, JPG, or PDF)
            2. Optionally upload **rooms JSON** if you have pre-computed room data
            3. Click **Detect Openings** to run the detection
            4. Review detected doors and windows
            5. **Edit** any misdetections using the table
            6. **Export** corrected schedules

            **Indian Conventions Supported:**
            - Door tags: D, D1, D2, MD (main), SD (sliding), TD (toilet), BD (balcony)
            - Window tags: W, W1, W2, LW (large), TW (toilet), V (ventilator)
            """)
        return

    result = st.session_state.openings_result

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🚪 Doors", result.total_doors)
    with col2:
        st.metric("🪟 Windows", result.total_windows)
    with col3:
        st.metric("💨 Ventilators", result.total_ventilators)
    with col4:
        warning_count = len(result.warnings)
        st.metric("⚠️ Warnings", warning_count)

    # Tabs
    tabs = st.tabs([
        "📸 Plan View",
        "📋 Openings Schedule",
        "🎨 Finishes BOQ",
        "⚙️ Edit Openings",
        "📥 Export"
    ])

    # Tab 1: Plan View
    with tabs[0]:
        if st.session_state.image is not None:
            overlay = draw_overlay(
                st.session_state.image,
                result,
                show_doors=st.session_state.show_doors,
                show_windows=st.session_state.show_windows,
                show_rooms=st.session_state.show_rooms,
                rooms_data=st.session_state.rooms_data,
            )

            # Convert BGR to RGB for display
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, use_container_width=True)

            # Legend
            st.markdown("""
            **Legend:**
            <span style='color: green'>■</span> Door
            <span style='color: blue'>■</span> Window
            <span style='color: magenta'>■</span> Ventilator
            """, unsafe_allow_html=True)

    # Tab 2: Openings Schedule
    with tabs[1]:
        df = openings_to_df(result)

        if not df.empty:
            # Summary by type
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Door Schedule")
                door_df = df[df["Type"] == "Door"]
                if not door_df.empty:
                    st.dataframe(door_df, use_container_width=True)
                else:
                    st.info("No doors detected")

            with col2:
                st.subheader("Window Schedule")
                window_df = df[df["Type"].isin(["Window", "Ventilator"])]
                if not window_df.empty:
                    st.dataframe(window_df, use_container_width=True)
                else:
                    st.info("No windows detected")

            # Warnings
            if result.warnings:
                st.subheader("⚠️ Warnings")
                for warning in result.warnings:
                    st.warning(warning)
        else:
            st.info("No openings detected")

    # Tab 3: Finishes BOQ
    with tabs[2]:
        if st.session_state.room_finishes:
            room_finishes = st.session_state.room_finishes

            # Totals
            valid_rooms = [r for r in room_finishes if r.room_category != "shaft"]
            total_floor = sum(r.floor_area_sqm for r in valid_rooms)
            total_skirting = sum(r.skirting_length_m for r in valid_rooms if not r.is_external)
            total_wall = sum(r.wall_area_sqm for r in valid_rooms)
            total_ceiling = sum(r.ceiling_area_sqm for r in valid_rooms)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Floor Area", f"{total_floor:.1f} sqm")
            col2.metric("Skirting", f"{total_skirting:.1f} m")
            col3.metric("Wall Area", f"{total_wall:.1f} sqm")
            col4.metric("Ceiling Area", f"{total_ceiling:.1f} sqm")

            st.divider()

            # Room breakdown
            st.subheader("Room-wise Breakdown")
            df = finishes_to_df(room_finishes)
            st.dataframe(df, use_container_width=True)

            # Wet areas
            wet_areas = [r for r in room_finishes if r.is_wet_area]
            if wet_areas:
                st.subheader("🚿 Wet Areas (Tile Heights)")
                for room in wet_areas:
                    tile_height = room.tile_height_m or 2.1
                    st.write(f"- {room.room_label}: Wall tiles up to {tile_height}m")
        else:
            st.info("Upload rooms JSON to calculate finishes")

    # Tab 4: Edit Openings
    with tabs[3]:
        st.subheader("Edit Detected Openings")

        # Select opening to edit
        all_openings = [(f"Door: {d.id} ({d.tag or 'no tag'})", "door", i)
                       for i, d in enumerate(result.doors)]
        all_openings += [(f"Window: {w.id} ({w.tag or 'no tag'})", "window", i)
                        for i, w in enumerate(result.windows)]

        if all_openings:
            selected = st.selectbox(
                "Select opening to edit",
                options=range(len(all_openings)),
                format_func=lambda x: all_openings[x][0]
            )

            opening_type, idx = all_openings[selected][1], all_openings[selected][2]

            if opening_type == "door":
                opening = result.doors[idx]
            else:
                opening = result.windows[idx]

            col1, col2 = st.columns(2)

            with col1:
                new_tag = st.text_input("Tag", value=opening.tag or "")
                new_type = st.selectbox(
                    "Type",
                    options=["door", "window", "ventilator"] if opening_type == "window"
                           else ["door", "main_door", "sliding_door", "toilet_door", "french_door"],
                    index=0
                )

            with col2:
                new_width = st.number_input(
                    "Width (mm)",
                    value=int(opening.width_m * 1000) if opening.width_m else 900,
                    step=50
                )
                new_height = st.number_input(
                    "Height (mm)",
                    value=int(opening.height_m * 1000) if opening.height_m else 2100,
                    step=50
                )

            if st.button("💾 Save Changes"):
                opening.tag = new_tag if new_tag else None
                opening.type = new_type
                opening.width_m = new_width / 1000
                opening.height_m = new_height / 1000
                st.success("Changes saved!")
                st.rerun()

            if st.button("🗑️ Delete Opening", type="secondary"):
                if opening_type == "door":
                    result.doors.pop(idx)
                else:
                    result.windows.pop(idx)
                st.success("Opening deleted!")
                st.rerun()
        else:
            st.info("No openings to edit")

    # Tab 5: Export
    with tabs[4]:
        st.subheader("📥 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Openings**")

            # Door schedule CSV
            df = openings_to_df(result)
            door_csv = df[df["Type"] == "Door"].to_csv(index=False)
            st.download_button(
                "📄 Door Schedule (CSV)",
                door_csv,
                "door_schedule.csv",
                "text/csv",
            )

            # Window schedule CSV
            window_csv = df[df["Type"].isin(["Window", "Ventilator"])].to_csv(index=False)
            st.download_button(
                "📄 Window Schedule (CSV)",
                window_csv,
                "window_schedule.csv",
                "text/csv",
            )

        with col2:
            st.markdown("**Finishes**")

            if st.session_state.room_finishes:
                # Finishes CSV
                finish_df = finishes_to_df(st.session_state.room_finishes)
                finish_csv = finish_df.to_csv(index=False)
                st.download_button(
                    "📄 Finishes BOQ (CSV)",
                    finish_csv,
                    "finishes_boq.csv",
                    "text/csv",
                )
            else:
                st.info("Upload rooms JSON for finishes")

        # JSON export
        st.divider()
        st.markdown("**Complete Data (JSON)**")

        export_data = {
            "plan_id": result.plan_id,
            "doors": [
                {
                    "id": d.id,
                    "type": d.type,
                    "tag": d.tag,
                    "width_m": d.width_m,
                    "height_m": d.height_m,
                    "room_left_id": d.room_left_id,
                    "room_right_id": d.room_right_id,
                    "confidence": d.confidence,
                }
                for d in result.doors
            ],
            "windows": [
                {
                    "id": w.id,
                    "type": w.type,
                    "tag": w.tag,
                    "width_m": w.width_m,
                    "height_m": w.height_m,
                    "room_left_id": w.room_left_id,
                    "confidence": w.confidence,
                }
                for w in result.windows
            ],
            "warnings": result.warnings,
        }

        st.download_button(
            "📄 Complete Data (JSON)",
            json.dumps(export_data, indent=2),
            "openings_complete.json",
            "application/json",
        )


if __name__ == "__main__":
    main()
