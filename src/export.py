"""
Floor Plan Export Module
Exports results to JSON, CSV, and annotated overlays.
"""

import logging
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import csv

from .area import RoomWithArea
from .scale import ScaleResult
from .qc import QCReport
from .ingest import IngestedPlan

logger = logging.getLogger(__name__)


# Color palette for room visualization
ROOM_COLORS = [
    (255, 179, 186),  # Light pink
    (255, 223, 186),  # Light orange
    (255, 255, 186),  # Light yellow
    (186, 255, 201),  # Light green
    (186, 225, 255),  # Light blue
    (218, 186, 255),  # Light purple
    (255, 186, 255),  # Light magenta
    (186, 255, 255),  # Light cyan
    (255, 218, 185),  # Peach
    (221, 160, 221),  # Plum
    (176, 224, 230),  # Powder blue
    (152, 251, 152),  # Pale green
]


class PlanExporter:
    """
    Exports floor plan analysis results.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize exporter.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        plan: IngestedPlan,
        rooms: List[RoomWithArea],
        scale: ScaleResult,
        qc_report: QCReport,
        wall_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Path]:
        """
        Export all outputs for a plan.

        Args:
            plan: Ingested plan
            rooms: Rooms with areas
            scale: Scale result
            qc_report: QC report
            wall_mask: Optional wall mask

        Returns:
            Dictionary of output paths
        """
        plan_dir = self.output_dir / plan.plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)

        debug_dir = plan_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

        outputs = {}

        # Export JSON
        json_path = self._export_json(plan_dir, plan, rooms, scale, qc_report)
        outputs['json'] = json_path

        # Export CSV
        csv_path = self._export_csv(plan_dir, rooms)
        outputs['csv'] = csv_path

        # Export overlay image
        overlay_path = self._export_overlay(plan_dir, plan.image, rooms, scale)
        outputs['overlay'] = overlay_path

        # Export debug artifacts
        debug_paths = self._export_debug(debug_dir, plan, wall_mask)
        outputs['debug'] = debug_paths

        logger.info(f"Exported results to {plan_dir}")
        return outputs

    def _export_json(
        self,
        output_dir: Path,
        plan: IngestedPlan,
        rooms: List[RoomWithArea],
        scale: ScaleResult,
        qc_report: QCReport
    ) -> Path:
        """Export results to JSON."""

        # Build rooms data
        rooms_data = []
        for room in rooms:
            room_dict = {
                'room_id': room.room_id,
                'label': room.label,
                'confidence': round(room.confidence, 2),
                'polygon': [[round(p[0], 1), round(p[1], 1)] for p in room.polygon.points],
                'area_sqm': room.area.area_sqm,
                'area_sqft': room.area.area_sqft,
                'bbox': list(room.polygon.bbox),
                'source': {
                    'boundary': 'walls' if room.polygon.is_valid else 'heuristic'
                }
            }

            if room.area.carpet_area_sqm:
                room_dict['carpet_area_sqm'] = room.area.carpet_area_sqm

            if room.area.warnings:
                room_dict['warnings'] = room.area.warnings

            rooms_data.append(room_dict)

        # Build scale data
        scale_data = {
            'method': scale.method.value,
            'pixels_per_mm': round(scale.pixels_per_mm, 6),
            'confidence': round(scale.confidence, 2)
        }

        if scale.scale_ratio:
            scale_data['ratio'] = f"1:{scale.scale_ratio}"

        if scale.dimension_used:
            scale_data['calibration'] = {
                'text': scale.dimension_used.text,
                'value_mm': scale.dimension_used.value_mm,
                'pixel_length': scale.dimension_used.pixel_length
            }

        # Build full output
        output = {
            'plan_id': plan.plan_id,
            'source_file': str(plan.source_path),
            'plan_type': plan.plan_type.value,
            'units': 'sqm',
            'scale': scale_data,
            'rooms': rooms_data,
            'statistics': qc_report.statistics,
            'warnings': [
                {
                    'code': w.code,
                    'severity': w.severity,
                    'message': w.message,
                    'room_id': w.room_id
                }
                for w in qc_report.warnings
            ],
            'qc': {
                'is_valid': qc_report.is_valid,
                'overall_confidence': qc_report.overall_confidence,
                'suggestions': qc_report.suggestions
            }
        }

        # Write JSON
        json_path = output_dir / "rooms.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return json_path

    def _export_csv(self, output_dir: Path, rooms: List[RoomWithArea]) -> Path:
        """Export room summary to CSV."""

        csv_path = output_dir / "rooms.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Room ID',
                'Label',
                'Area (sqm)',
                'Area (sqft)',
                'Carpet Area (sqm)',
                'Perimeter (m)',
                'Confidence',
                'Warnings'
            ])

            # Data rows
            for room in rooms:
                writer.writerow([
                    room.room_id,
                    room.label,
                    room.area.area_sqm,
                    room.area.area_sqft,
                    room.area.carpet_area_sqm or '',
                    room.area.perimeter_m,
                    f"{room.confidence:.0%}",
                    '; '.join(room.area.warnings) if room.area.warnings else ''
                ])

            # Summary row
            total_sqm = sum(r.area.area_sqm for r in rooms)
            total_sqft = sum(r.area.area_sqft for r in rooms)
            writer.writerow([])
            writer.writerow(['TOTAL', '', total_sqm, total_sqft, '', '', '', ''])

        return csv_path

    def _export_overlay(
        self,
        output_dir: Path,
        image: np.ndarray,
        rooms: List[RoomWithArea],
        scale: ScaleResult
    ) -> Path:
        """Export annotated overlay image."""

        if image.size == 0:
            logger.warning("No image to create overlay")
            return output_dir / "overlay.png"

        # Create copy of image
        overlay = image.copy()

        # Draw rooms with semi-transparent fill
        for i, room in enumerate(rooms):
            color = ROOM_COLORS[i % len(ROOM_COLORS)]

            # Draw filled polygon
            pts = np.array(room.polygon.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)

        # Blend with original
        alpha = 0.4
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw room boundaries and labels
        for i, room in enumerate(rooms):
            pts = np.array(room.polygon.points, dtype=np.int32)

            # Draw boundary
            cv2.polylines(result, [pts], True, (0, 0, 0), 2)

            # Draw label
            cx, cy = room.polygon.centroid
            cx, cy = int(cx), int(cy)

            # Label text
            label_text = room.label
            area_text = f"{room.area.area_sqm:.1f} sqm"

            # Get text sizes
            (label_w, label_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            (area_w, area_h), _ = cv2.getTextSize(
                area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw background rectangle for text
            max_w = max(label_w, area_w) + 10
            total_h = label_h + area_h + 15
            x1 = cx - max_w // 2
            y1 = cy - total_h // 2
            cv2.rectangle(result, (x1, y1), (x1 + max_w, y1 + total_h), (255, 255, 255), -1)
            cv2.rectangle(result, (x1, y1), (x1 + max_w, y1 + total_h), (0, 0, 0), 1)

            # Draw label text
            cv2.putText(
                result, label_text,
                (x1 + 5, y1 + label_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

            # Draw area text
            cv2.putText(
                result, area_text,
                (x1 + 5, y1 + label_h + area_h + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 1
            )

        # Add scale info in corner
        scale_text = f"Scale: {scale.method.value}"
        if scale.scale_ratio:
            scale_text += f" (1:{scale.scale_ratio})"
        scale_text += f" | Confidence: {scale.confidence:.0%}"

        cv2.putText(
            result, scale_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        # Save
        overlay_path = output_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), result)

        return overlay_path

    def _export_debug(
        self,
        debug_dir: Path,
        plan: IngestedPlan,
        wall_mask: Optional[np.ndarray]
    ) -> Dict[str, Path]:
        """Export debug artifacts."""

        outputs = {}

        # Original image
        if plan.image.size > 0:
            orig_path = debug_dir / "original.png"
            cv2.imwrite(str(orig_path), plan.image)
            outputs['original'] = orig_path

        # Wall mask
        if wall_mask is not None:
            wall_path = debug_dir / "wall_mask.png"
            cv2.imwrite(str(wall_path), wall_mask)
            outputs['wall_mask'] = wall_path

        # Vector text locations (if available)
        if plan.vector_texts and plan.image.size > 0:
            text_vis = plan.image.copy()
            for vt in plan.vector_texts:
                x0, y0, x1, y1 = [int(v) for v in vt.bbox]
                cv2.rectangle(text_vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
                cv2.putText(text_vis, vt.text[:20], (x0, y0-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            text_path = debug_dir / "text_detection.png"
            cv2.imwrite(str(text_path), text_vis)
            outputs['text_detection'] = text_path

        return outputs


def export_plan(
    plan: IngestedPlan,
    rooms: List[RoomWithArea],
    scale: ScaleResult,
    qc_report: QCReport,
    output_dir: Path,
    wall_mask: Optional[np.ndarray] = None
) -> Dict[str, Path]:
    """
    Export plan analysis results.

    Args:
        plan: Ingested plan
        rooms: Rooms with areas
        scale: Scale result
        qc_report: QC report
        output_dir: Output directory
        wall_mask: Optional wall mask

    Returns:
        Dictionary of output paths
    """
    exporter = PlanExporter(output_dir)
    return exporter.export_all(plan, rooms, scale, qc_report, wall_mask)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Export module - use with actual data")
