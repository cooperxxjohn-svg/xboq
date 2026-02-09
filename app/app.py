"""
Floor Plan Room & Area Engine - Streamlit QC UI
Interactive tool for reviewing and correcting floor plan analysis.
"""

import streamlit as st
import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FloorPlanPipeline, PipelineConfig, PipelineResult
from src.scale import ScaleResult, ScaleMethod
from src.area import AreaComputer, RoomWithArea
from src.qc import QualityChecker
from src.export import PlanExporter

# Page config
st.set_page_config(
    page_title="Floor Plan Room & Area Engine",
    page_icon="🏠",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 10px;
    }
    .room-card {
        background: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 8px;
    }
    .warning-badge {
        background: #fff3cd;
        color: #856404;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .error-badge {
        background: #f8d7da;
        color: #721c24;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .success-badge {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Room type options for dropdown
ROOM_TYPES = [
    "Bedroom", "Master Bedroom", "Living", "Drawing", "Dining", "Kitchen",
    "Toilet", "Bathroom", "Balcony", "Terrace", "Passage", "Lobby",
    "Store", "Utility", "Study", "Pooja", "Dressing", "Staircase",
    "Lift", "Duct", "Parking", "Wash", "Room", "Other"
]

# Room colors for visualization
ROOM_COLORS = [
    (255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201),
    (186, 225, 255), (218, 186, 255), (255, 186, 255), (186, 255, 255),
    (255, 218, 185), (221, 160, 221), (176, 224, 230), (152, 251, 152),
]


def init_session_state():
    """Initialize session state variables."""
    if 'pipeline_result' not in st.session_state:
        st.session_state.pipeline_result = None
    if 'rooms' not in st.session_state:
        st.session_state.rooms = []
    if 'scale' not in st.session_state:
        st.session_state.scale = None
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'selected_rooms' not in st.session_state:
        st.session_state.selected_rooms = set()
    if 'calibration_points' not in st.session_state:
        st.session_state.calibration_points = []
    if 'calibration_mode' not in st.session_state:
        st.session_state.calibration_mode = False


def create_overlay_image(
    image: np.ndarray,
    rooms: List[RoomWithArea],
    selected_rooms: set = None,
    show_labels: bool = True,
    show_areas: bool = True
) -> np.ndarray:
    """Create annotated overlay image."""
    if image.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    overlay = image.copy()
    result = image.copy()

    # Draw rooms
    for i, room in enumerate(rooms):
        if not room.polygon.points:
            continue

        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pts = np.array(room.polygon.points, dtype=np.int32)

        # Fill with semi-transparent color
        cv2.fillPoly(overlay, [pts], color)

        # Highlight selected rooms
        if selected_rooms and room.room_id in selected_rooms:
            cv2.polylines(result, [pts], True, (0, 255, 0), 3)
        else:
            cv2.polylines(result, [pts], True, (0, 0, 0), 1)

    # Blend
    result = cv2.addWeighted(overlay, 0.4, result, 0.6, 0)

    # Add labels and areas
    if show_labels or show_areas:
        for room in rooms:
            if not room.polygon.points:
                continue

            cx, cy = int(room.polygon.centroid[0]), int(room.polygon.centroid[1])

            texts = []
            if show_labels:
                texts.append(room.label)
            if show_areas:
                texts.append(f"{room.area.area_sqm:.1f} sqm")

            # Draw text background
            for j, text in enumerate(texts):
                y_offset = cy - 10 + j * 20
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    result,
                    (cx - tw//2 - 3, y_offset - th - 3),
                    (cx + tw//2 + 3, y_offset + 3),
                    (255, 255, 255), -1
                )
                cv2.putText(
                    result, text,
                    (cx - tw//2, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )

    return result


def render_sidebar():
    """Render sidebar with navigation and tools."""
    with st.sidebar:
        st.markdown("### 🏠 Floor Plan Engine")
        st.markdown("---")

        # File upload
        st.markdown("#### Upload Plan")
        uploaded_file = st.file_uploader(
            "Select floor plan",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload a floor plan PDF or image"
        )

        if uploaded_file:
            if st.button("Process Plan", type="primary", use_container_width=True):
                process_uploaded_file(uploaded_file)

        st.markdown("---")

        # Scale calibration
        st.markdown("#### Scale Calibration")
        if st.session_state.scale:
            scale = st.session_state.scale
            st.info(f"Method: {scale.method.value}")
            st.info(f"Confidence: {scale.confidence:.0%}")
            if scale.scale_ratio:
                st.info(f"Ratio: 1:{scale.scale_ratio}")

        if st.button("Manual Scale Calibration"):
            st.session_state.calibration_mode = True
            st.session_state.calibration_points = []
            st.info("Click two points on the image, then enter the real distance")

        if st.session_state.calibration_mode:
            real_length = st.number_input(
                "Real length (mm)",
                min_value=100,
                max_value=50000,
                value=3000,
                step=100
            )
            if st.button("Apply Calibration"):
                apply_manual_calibration(real_length)

        st.markdown("---")

        # Display options
        st.markdown("#### Display Options")
        show_labels = st.checkbox("Show Labels", value=True)
        show_areas = st.checkbox("Show Areas", value=True)

        st.markdown("---")

        # Export
        st.markdown("#### Export")
        if st.session_state.rooms:
            if st.button("Export JSON", use_container_width=True):
                export_json()
            if st.button("Export CSV", use_container_width=True):
                export_csv()

        return show_labels, show_areas


def process_uploaded_file(uploaded_file):
    """Process an uploaded floor plan file."""
    with st.spinner("Processing floor plan..."):
        # Save uploaded file temporarily
        temp_path = Path("/tmp") / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run pipeline
        config = PipelineConfig(
            dpi=300,
            output_dir=Path("/tmp/floorplan_out")
        )
        pipeline = FloorPlanPipeline(config)
        result = pipeline.process(temp_path)

        # Store in session state
        st.session_state.pipeline_result = result
        st.session_state.rooms = result.rooms_with_area if result.rooms_with_area else []
        st.session_state.scale = result.scale
        st.session_state.image = result.plan.image if result.plan else None

        if result.success:
            st.success(f"Processed successfully! Found {len(result.rooms_with_area)} rooms.")
        else:
            st.error(f"Processing failed: {', '.join(result.errors)}")


def apply_manual_calibration(real_length_mm: float):
    """Apply manual scale calibration."""
    if len(st.session_state.calibration_points) >= 2:
        p1 = st.session_state.calibration_points[0]
        p2 = st.session_state.calibration_points[1]

        pixel_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        pixels_per_mm = pixel_length / real_length_mm

        new_scale = ScaleResult(
            method=ScaleMethod.MANUAL,
            pixels_per_mm=pixels_per_mm,
            confidence=0.95
        )

        st.session_state.scale = new_scale
        st.session_state.calibration_mode = False
        st.session_state.calibration_points = []

        # Recompute areas
        if st.session_state.rooms:
            recompute_areas()

        st.success("Scale calibration applied!")
    else:
        st.warning("Please click two points on the image first")


def recompute_areas():
    """Recompute areas with current scale."""
    if not st.session_state.scale or not st.session_state.pipeline_result:
        return

    result = st.session_state.pipeline_result
    area_computer = AreaComputer(st.session_state.scale)

    new_rooms = []
    for lr in result.labeled_rooms:
        area_result = area_computer.compute_area(lr.polygon)
        confidence = lr.label.confidence * st.session_state.scale.confidence

        # Find existing room to preserve label changes
        existing = None
        for r in st.session_state.rooms:
            if r.polygon.room_id == lr.polygon.room_id:
                existing = r
                break

        new_rooms.append(RoomWithArea(
            room_id=lr.polygon.room_id,
            label=existing.label if existing else lr.label.canonical,
            polygon=lr.polygon,
            area=area_result,
            confidence=confidence
        ))

    st.session_state.rooms = new_rooms


def export_json():
    """Export rooms to JSON."""
    if not st.session_state.rooms:
        st.warning("No rooms to export")
        return

    data = {
        'plan_id': st.session_state.pipeline_result.plan_id if st.session_state.pipeline_result else 'unknown',
        'units': 'sqm',
        'scale': {
            'method': st.session_state.scale.method.value if st.session_state.scale else 'unknown',
            'confidence': st.session_state.scale.confidence if st.session_state.scale else 0
        },
        'rooms': [
            {
                'room_id': r.room_id,
                'label': r.label,
                'area_sqm': r.area.area_sqm,
                'area_sqft': r.area.area_sqft,
                'confidence': r.confidence,
                'polygon': [[round(p[0], 1), round(p[1], 1)] for p in r.polygon.points]
            }
            for r in st.session_state.rooms
        ]
    }

    json_str = json.dumps(data, indent=2)
    st.download_button(
        "Download JSON",
        json_str,
        file_name="rooms.json",
        mime="application/json"
    )


def export_csv():
    """Export rooms to CSV."""
    if not st.session_state.rooms:
        st.warning("No rooms to export")
        return

    df = pd.DataFrame([
        {
            'Room ID': r.room_id,
            'Label': r.label,
            'Area (sqm)': r.area.area_sqm,
            'Area (sqft)': r.area.area_sqft,
            'Confidence': f"{r.confidence:.0%}"
        }
        for r in st.session_state.rooms
    ])

    csv_str = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv_str,
        file_name="rooms.csv",
        mime="text/csv"
    )


def render_main_content(show_labels: bool, show_areas: bool):
    """Render main content area."""
    st.markdown('<div class="main-header">🏠 Floor Plan Room & Area Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive tool for analyzing Indian residential floor plans</div>', unsafe_allow_html=True)

    if st.session_state.image is None:
        st.info("👈 Upload a floor plan to get started")

        # Show sample workflow
        st.markdown("### How it works:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**1. Upload**")
            st.markdown("Upload PDF or image of floor plan")
        with col2:
            st.markdown("**2. Analyze**")
            st.markdown("Automatic room detection")
        with col3:
            st.markdown("**3. Review**")
            st.markdown("Verify and correct labels")
        with col4:
            st.markdown("**4. Export**")
            st.markdown("Download JSON/CSV results")
        return

    # Results view
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Floor Plan")

        # Create overlay
        overlay = create_overlay_image(
            st.session_state.image,
            st.session_state.rooms,
            st.session_state.selected_rooms,
            show_labels,
            show_areas
        )

        # Convert BGR to RGB for display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, use_container_width=True)

        # Handle calibration clicks
        if st.session_state.calibration_mode:
            st.markdown("**Click to add calibration points:**")
            # Note: Streamlit doesn't natively support click events on images
            # This would need a custom component or workaround
            st.info("For manual calibration, use the coordinate input below")

            cal_col1, cal_col2 = st.columns(2)
            with cal_col1:
                x1 = st.number_input("Point 1 X", min_value=0, value=100)
                y1 = st.number_input("Point 1 Y", min_value=0, value=100)
            with cal_col2:
                x2 = st.number_input("Point 2 X", min_value=0, value=500)
                y2 = st.number_input("Point 2 Y", min_value=0, value=100)

            st.session_state.calibration_points = [(x1, y1), (x2, y2)]

    with col2:
        st.markdown("### Room Details")

        if not st.session_state.rooms:
            st.warning("No rooms detected")
            return

        # Summary metrics
        total_sqm = sum(r.area.area_sqm for r in st.session_state.rooms)
        total_sqft = sum(r.area.area_sqft for r in st.session_state.rooms)

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Rooms", len(st.session_state.rooms))
        with metric_col2:
            st.metric("Total Area", f"{total_sqm:.1f} sqm")

        st.markdown("---")

        # Room list with editing
        st.markdown("#### Rooms")

        for i, room in enumerate(st.session_state.rooms):
            with st.expander(f"**{room.label}** - {room.area.area_sqm:.1f} sqm", expanded=False):
                # Label editing
                new_label = st.selectbox(
                    "Room Type",
                    ROOM_TYPES,
                    index=ROOM_TYPES.index(room.label) if room.label in ROOM_TYPES else len(ROOM_TYPES) - 1,
                    key=f"label_{room.room_id}"
                )

                if new_label == "Other":
                    new_label = st.text_input(
                        "Custom Label",
                        value=room.label,
                        key=f"custom_{room.room_id}"
                    )

                if new_label != room.label:
                    room.label = new_label

                # Display details
                st.write(f"**Area:** {room.area.area_sqm:.2f} sqm ({room.area.area_sqft:.2f} sqft)")
                st.write(f"**Perimeter:** {room.area.perimeter_m:.2f} m")
                st.write(f"**Confidence:** {room.confidence:.0%}")

                if room.area.warnings:
                    for w in room.area.warnings:
                        st.markdown(f'<span class="warning-badge">{w}</span>', unsafe_allow_html=True)

                # Actions
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("Select", key=f"select_{room.room_id}"):
                        if room.room_id in st.session_state.selected_rooms:
                            st.session_state.selected_rooms.remove(room.room_id)
                        else:
                            st.session_state.selected_rooms.add(room.room_id)
                        st.rerun()

                with action_col2:
                    if st.button("Delete", key=f"delete_{room.room_id}"):
                        st.session_state.rooms = [r for r in st.session_state.rooms if r.room_id != room.room_id]
                        st.rerun()

        # Merge selected rooms
        if len(st.session_state.selected_rooms) >= 2:
            st.markdown("---")
            if st.button("Merge Selected Rooms"):
                merge_selected_rooms()

        # QC Report
        if st.session_state.pipeline_result and st.session_state.pipeline_result.qc_report:
            st.markdown("---")
            st.markdown("#### Quality Report")
            qc = st.session_state.pipeline_result.qc_report

            if qc.is_valid:
                st.markdown('<span class="success-badge">PASS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="error-badge">FAIL</span>', unsafe_allow_html=True)

            st.write(f"Confidence: {qc.overall_confidence:.0%}")

            if qc.warnings:
                st.markdown("**Warnings:**")
                for w in qc.warnings[:5]:
                    severity_class = "error-badge" if w.severity == "error" else "warning-badge"
                    st.markdown(f'<span class="{severity_class}">{w.code}</span> {w.message}', unsafe_allow_html=True)


def merge_selected_rooms():
    """Merge selected rooms into one."""
    selected = [r for r in st.session_state.rooms if r.room_id in st.session_state.selected_rooms]

    if len(selected) < 2:
        st.warning("Select at least 2 rooms to merge")
        return

    try:
        from src.polygons import merge_polygons

        polygons = [r.polygon for r in selected]
        merged_polygon = merge_polygons(polygons)

        if merged_polygon:
            # Create new room
            area_computer = AreaComputer(st.session_state.scale)
            area_result = area_computer.compute_area(merged_polygon)

            merged_room = RoomWithArea(
                room_id=f"merged_{selected[0].room_id}",
                label=selected[0].label,
                polygon=merged_polygon,
                area=area_result,
                confidence=min(r.confidence for r in selected)
            )

            # Remove old rooms, add merged
            st.session_state.rooms = [r for r in st.session_state.rooms if r.room_id not in st.session_state.selected_rooms]
            st.session_state.rooms.append(merged_room)
            st.session_state.selected_rooms = set()

            st.success("Rooms merged successfully!")
            st.rerun()
        else:
            st.error("Could not merge rooms")

    except Exception as e:
        st.error(f"Merge failed: {e}")


def main():
    """Main application."""
    init_session_state()
    show_labels, show_areas = render_sidebar()
    render_main_content(show_labels, show_areas)


if __name__ == "__main__":
    main()
