"""
Export Module for Openings
Exports door/window schedules, overlays, and JSON data.

Outputs:
- openings.json: Complete opening data
- door_schedule.csv: Door schedule table
- window_schedule.csv: Window schedule table
- overlay_openings.png: Visualization with marked openings
- debug/: Debug artifacts
"""

import json
import csv
import cv2
import numpy as np
from dataclasses import asdict
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class OpeningsExporter:
    """
    Export openings data to various formats.

    Formats:
    - JSON: Complete structured data
    - CSV: Tabular schedules
    - PNG: Visual overlays
    """

    # Colors for visualization (BGR)
    DOOR_COLOR = (0, 255, 0)        # Green
    WINDOW_COLOR = (255, 0, 0)      # Blue
    VENTILATOR_COLOR = (255, 0, 255) # Magenta
    LABEL_COLOR = (255, 255, 255)   # White
    CONFIDENCE_COLORS = {
        "high": (0, 200, 0),
        "medium": (0, 200, 200),
        "low": (0, 100, 200),
    }

    def __init__(self, output_dir: Path):
        """
        Initialize exporter.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create debug subdirectory
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)

    def export_all(
        self,
        plan_id: str,
        doors: List[Any],
        windows: List[Any],
        assignments: Optional[List[Any]] = None,
        image: Optional[np.ndarray] = None,
        assumptions_used: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """
        Export all opening data.

        Args:
            plan_id: Identifier for the plan
            doors: List of DetectedDoor objects
            windows: List of DetectedWindow objects
            assignments: List of OpeningAssignment objects
            image: Original image for overlay
            assumptions_used: Dict of assumptions made during inference

        Returns:
            Dict mapping output type to file path
        """
        outputs = {}

        # Export JSON
        json_path = self.export_json(plan_id, doors, windows, assignments, assumptions_used)
        outputs["json"] = json_path

        # Export door schedule CSV
        door_csv = self.export_door_schedule_csv(doors)
        outputs["door_schedule"] = door_csv

        # Export window schedule CSV
        window_csv = self.export_window_schedule_csv(windows)
        outputs["window_schedule"] = window_csv

        # Export overlay if image provided
        if image is not None:
            overlay_path = self.export_overlay(plan_id, doors, windows, image)
            outputs["overlay"] = overlay_path

            # Debug overlays
            door_debug = self.export_debug_overlay(doors, image, "doors")
            outputs["door_candidates"] = door_debug

            window_debug = self.export_debug_overlay(windows, image, "windows")
            outputs["window_candidates"] = window_debug

        # Export assumptions
        if assumptions_used:
            assumptions_path = self.output_dir / "assumptions.json"
            with open(assumptions_path, "w") as f:
                json.dump(assumptions_used, f, indent=2, cls=NumpyEncoder)
            outputs["assumptions"] = assumptions_path

        logger.info(f"Exported {len(outputs)} files to {self.output_dir}")

        return outputs

    def export_json(
        self,
        plan_id: str,
        doors: List[Any],
        windows: List[Any],
        assignments: Optional[List[Any]] = None,
        assumptions_used: Optional[Dict] = None,
    ) -> Path:
        """Export complete opening data as JSON."""

        # Build openings list
        openings = []

        for door in doors:
            opening_data = {
                "id": door.id,
                "type": door.type,
                "tag": door.tag,
                "confidence": float(door.confidence) if door.confidence else 0.0,
                "bbox": [int(x) for x in door.bbox],
                "wall_segment_id": door.wall_segment_id,
                "room_left_id": door.room_left_id,
                "room_right_id": door.room_right_id,
                "width_m": float(door.width_m) if door.width_m else None,
                "height_m": float(door.height_m) if door.height_m else None,
                "source": door.source,
            }
            openings.append(opening_data)

        for window in windows:
            opening_data = {
                "id": window.id,
                "type": window.type,
                "tag": window.tag,
                "confidence": float(window.confidence) if window.confidence else 0.0,
                "bbox": [int(x) for x in window.bbox],
                "wall_segment_id": window.wall_segment_id,
                "room_left_id": window.room_left_id,
                "room_right_id": window.room_right_id,
                "width_m": float(window.width_m) if window.width_m else None,
                "height_m": float(window.height_m) if window.height_m else None,
                "source": window.source,
            }
            openings.append(opening_data)

        # Build schedules
        door_schedule = self._build_schedule_summary(doors)
        window_schedule = self._build_schedule_summary(windows)

        # Build warnings
        warnings = self._generate_warnings(doors, windows)

        # Complete JSON structure
        data = {
            "plan_id": plan_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_doors": len(doors),
                "total_windows": len([w for w in windows if w.type != "ventilator"]),
                "total_ventilators": len([w for w in windows if w.type == "ventilator"]),
            },
            "openings": openings,
            "schedules": {
                "doors": door_schedule,
                "windows": window_schedule,
            },
            "warnings": warnings,
        }

        if assumptions_used:
            data["assumptions_used"] = assumptions_used

        # Write JSON
        json_path = self.output_dir / "openings.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

        return json_path

    def _build_schedule_summary(
        self,
        openings: List[Any],
    ) -> List[Dict]:
        """Build schedule summary grouped by tag."""
        by_tag: Dict[str, List] = {}

        for opening in openings:
            tag = opening.tag or opening.id
            base_tag = tag.split("-")[0] if "-" in tag else tag

            if base_tag not in by_tag:
                by_tag[base_tag] = []
            by_tag[base_tag].append(opening)

        schedule = []
        for tag, items in by_tag.items():
            widths = [o.width_m for o in items if o.width_m]
            heights = [o.height_m for o in items if o.height_m]

            entry = {
                "tag": tag,
                "count": len(items),
                "avg_width_m": sum(widths) / len(widths) if widths else None,
                "avg_height_m": sum(heights) / len(heights) if heights else None,
            }
            schedule.append(entry)

        return schedule

    def _generate_warnings(
        self,
        doors: List[Any],
        windows: List[Any],
    ) -> List[str]:
        """Generate QC warnings."""
        warnings = []

        # Check for unassigned openings
        unassigned_doors = [d for d in doors if not d.room_left_id]
        if unassigned_doors:
            warnings.append(f"opening_unassigned: {len(unassigned_doors)} doors not assigned to rooms")

        unassigned_windows = [w for w in windows if not w.room_left_id]
        if unassigned_windows:
            warnings.append(f"opening_unassigned: {len(unassigned_windows)} windows not assigned to rooms")

        # Check for low confidence
        low_conf_doors = [d for d in doors if d.confidence < 0.5]
        if low_conf_doors:
            warnings.append(f"low_confidence: {len(low_conf_doors)} doors with confidence < 0.5")

        low_conf_windows = [w for w in windows if w.confidence < 0.5]
        if low_conf_windows:
            warnings.append(f"low_confidence: {len(low_conf_windows)} windows with confidence < 0.5")

        # Check for missing tags
        untagged = [o for o in doors + windows if not o.tag]
        if untagged:
            warnings.append(f"no_tag: {len(untagged)} openings without OCR tags")

        # Check for unusual counts
        if len(doors) > 30:
            warnings.append(f"too_many_openings: {len(doors)} doors detected (unusually high)")
        if len(windows) > 50:
            warnings.append(f"too_many_openings: {len(windows)} windows detected (unusually high)")

        return warnings

    def export_door_schedule_csv(
        self,
        doors: List[Any],
    ) -> Path:
        """Export door schedule as CSV."""
        csv_path = self.output_dir / "door_schedule.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "ID", "Tag", "Type", "Width (mm)", "Height (mm)",
                "Room Left", "Room Right", "Confidence", "Source"
            ])

            # Data rows
            for door in doors:
                width_mm = int(door.width_m * 1000) if door.width_m else ""
                height_mm = int(door.height_m * 1000) if door.height_m else ""

                writer.writerow([
                    door.id,
                    door.tag or "",
                    door.type,
                    width_mm,
                    height_mm,
                    door.room_left_id or "",
                    door.room_right_id or "",
                    f"{door.confidence:.2f}",
                    door.source.get("symbol", "") if door.source else "",
                ])

        return csv_path

    def export_window_schedule_csv(
        self,
        windows: List[Any],
    ) -> Path:
        """Export window schedule as CSV."""
        csv_path = self.output_dir / "window_schedule.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "ID", "Tag", "Type", "Width (mm)", "Height (mm)",
                "Sill Height (mm)", "Room", "Confidence", "Source"
            ])

            # Data rows
            for window in windows:
                width_mm = int(window.width_m * 1000) if window.width_m else ""
                height_mm = int(window.height_m * 1000) if window.height_m else ""
                sill_mm = int(window.sill_height_m * 1000) if getattr(window, "sill_height_m", None) else ""

                writer.writerow([
                    window.id,
                    window.tag or "",
                    window.type,
                    width_mm,
                    height_mm,
                    sill_mm,
                    window.room_left_id or "",
                    f"{window.confidence:.2f}",
                    window.source.get("symbol", "") if window.source else "",
                ])

        return csv_path

    def export_overlay(
        self,
        plan_id: str,
        doors: List[Any],
        windows: List[Any],
        image: np.ndarray,
    ) -> Path:
        """Export visual overlay with marked openings."""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        # Draw doors
        for door in doors:
            x1, y1, x2, y2 = door.bbox
            color = self._get_confidence_color(door.confidence)

            # Draw filled rectangle with transparency
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.DOOR_COLOR, 2)

            # Draw door swing arc if swing direction known
            if door.swing_direction:
                self._draw_door_swing(overlay, door)

            # Label
            label = door.tag if door.tag else door.id
            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.DOOR_COLOR, 2
            )

        # Draw windows
        for window in windows:
            x1, y1, x2, y2 = window.bbox

            if window.type == "ventilator":
                color = self.VENTILATOR_COLOR
            else:
                color = self.WINDOW_COLOR

            # Draw cross pattern for windows
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.line(overlay, (x1, (y1+y2)//2), (x2, (y1+y2)//2), color, 1)

            # Label
            label = window.tag if window.tag else window.id
            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Add legend
        self._draw_legend(overlay)

        # Save
        overlay_path = self.output_dir / "overlay_openings.png"
        cv2.imwrite(str(overlay_path), overlay)

        return overlay_path

    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence >= 0.7:
            return self.CONFIDENCE_COLORS["high"]
        elif confidence >= 0.5:
            return self.CONFIDENCE_COLORS["medium"]
        else:
            return self.CONFIDENCE_COLORS["low"]

    def _draw_door_swing(
        self,
        overlay: np.ndarray,
        door: Any,
    ) -> None:
        """Draw door swing arc."""
        x1, y1, x2, y2 = door.bbox
        center = door.center
        radius = int(door.width_px * 0.8)

        # Draw quarter circle
        cv2.ellipse(
            overlay,
            center,
            (radius, radius),
            0, 0, 90,
            self.DOOR_COLOR,
            1
        )

    def _draw_legend(self, overlay: np.ndarray) -> None:
        """Draw legend on overlay."""
        h, w = overlay.shape[:2]

        # Legend background
        legend_h = 80
        cv2.rectangle(overlay, (10, h - legend_h - 10), (200, h - 10), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, h - legend_h - 10), (200, h - 10), (255, 255, 255), 1)

        # Legend items
        y = h - legend_h
        cv2.rectangle(overlay, (20, y + 5), (40, y + 15), self.DOOR_COLOR, -1)
        cv2.putText(overlay, "Door", (50, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.LABEL_COLOR, 1)

        cv2.rectangle(overlay, (20, y + 25), (40, y + 35), self.WINDOW_COLOR, -1)
        cv2.putText(overlay, "Window", (50, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.LABEL_COLOR, 1)

        cv2.rectangle(overlay, (20, y + 45), (40, y + 55), self.VENTILATOR_COLOR, -1)
        cv2.putText(overlay, "Ventilator", (50, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.LABEL_COLOR, 1)

    def export_debug_overlay(
        self,
        openings: List[Any],
        image: np.ndarray,
        name: str,
    ) -> Path:
        """Export debug overlay with detailed info."""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        for opening in openings:
            x1, y1, x2, y2 = opening.bbox

            # Color by confidence
            color = self._get_confidence_color(opening.confidence)

            # Draw bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Draw center
            cv2.circle(overlay, opening.center, 5, color, -1)

            # Detailed label
            label = f"{opening.id}"
            if opening.tag:
                label += f" ({opening.tag})"
            label += f" [{opening.confidence:.2f}]"

            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

            # Show signals
            signals = ",".join(opening.signals)
            cv2.putText(
                overlay, signals,
                (x1, y2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1
            )

        debug_path = self.debug_dir / f"{name}_candidates.png"
        cv2.imwrite(str(debug_path), overlay)

        return debug_path


def export_combined_schedule(
    doors: List[Any],
    windows: List[Any],
    output_path: Path,
) -> None:
    """Export combined door + window schedule as single CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Type", "ID", "Tag", "Width (mm)", "Height (mm)",
            "Room", "Confidence"
        ])

        for door in doors:
            width_mm = int(door.width_m * 1000) if door.width_m else ""
            height_mm = int(door.height_m * 1000) if door.height_m else ""

            writer.writerow([
                "Door",
                door.id,
                door.tag or "",
                width_mm,
                height_mm,
                door.room_left_id or "",
                f"{door.confidence:.2f}",
            ])

        for window in windows:
            width_mm = int(window.width_m * 1000) if window.width_m else ""
            height_mm = int(window.height_m * 1000) if window.height_m else ""

            writer.writerow([
                "Window" if window.type != "ventilator" else "Ventilator",
                window.id,
                window.tag or "",
                width_mm,
                height_mm,
                window.room_left_id or "",
                f"{window.confidence:.2f}",
            ])
