"""
Structural Export Module
Exports structural takeoff results to various formats:
- JSON (detailed with traceability)
- CSV (BOQ-ready format)
- Annotated overlays (visual verification)
"""

import logging
import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import cv2

from .detect_columns import DetectedColumn, ColumnDetectionResult
from .detect_beams import DetectedBeam, BeamDetectionResult
from .detect_footings import DetectedFooting, FootingDetectionResult
from .quantity_engine import QuantityResult, ElementQuantity, QuantitySummary
from .qc_structural import StructuralQCReport

logger = logging.getLogger(__name__)


class StructuralExporter:
    """
    Exports structural takeoff results.
    """

    def __init__(self, output_dir: Path):
        """Initialize exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        plan_id: str,
        column_result: ColumnDetectionResult = None,
        beam_result: BeamDetectionResult = None,
        footing_result: FootingDetectionResult = None,
        quantity_result: QuantityResult = None,
        qc_report: StructuralQCReport = None,
        image: np.ndarray = None
    ) -> Dict[str, Path]:
        """
        Export all results.

        Returns:
            Dictionary of output file paths
        """
        paths = {}

        # Create plan output directory
        plan_dir = self.output_dir / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Export JSON (full detail)
        json_path = self.export_json(
            plan_dir / "structural_takeoff.json",
            plan_id, column_result, beam_result, footing_result,
            quantity_result, qc_report
        )
        paths['json'] = json_path

        # Export CSV files
        if quantity_result:
            csv_paths = self.export_csv(
                plan_dir,
                quantity_result
            )
            paths.update(csv_paths)

        # Export BOQ summary
        if quantity_result:
            boq_path = self.export_boq(
                plan_dir / "boq_summary.csv",
                quantity_result
            )
            paths['boq'] = boq_path

        # Export overlay images
        if image is not None:
            overlay_paths = self.export_overlays(
                plan_dir,
                image,
                column_result,
                beam_result,
                footing_result
            )
            paths.update(overlay_paths)

        # Export QC report
        if qc_report:
            qc_path = self.export_qc_report(
                plan_dir / "qc_report.json",
                qc_report
            )
            paths['qc'] = qc_path

        logger.info(f"Exported {len(paths)} files to {plan_dir}")
        return paths

    def export_json(
        self,
        output_path: Path,
        plan_id: str,
        column_result: ColumnDetectionResult = None,
        beam_result: BeamDetectionResult = None,
        footing_result: FootingDetectionResult = None,
        quantity_result: QuantityResult = None,
        qc_report: StructuralQCReport = None
    ) -> Path:
        """Export detailed JSON with full traceability."""

        data = {
            'plan_id': plan_id,
            'version': '1.0',
            'mode': quantity_result.mode if quantity_result else 'unknown',

            'detection': {
                'columns': self._serialize_columns(column_result),
                'beams': self._serialize_beams(beam_result),
                'footings': self._serialize_footings(footing_result)
            },

            'quantities': {
                'elements': [elem.to_dict() for elem in quantity_result.elements]
                            if quantity_result else [],
                'summary': quantity_result.summary.to_dict() if quantity_result else {},
                'assumptions': quantity_result.assumptions_used if quantity_result else []
            },

            'qc': qc_report.to_dict() if qc_report else {}
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported JSON: {output_path}")
        return output_path

    def _serialize_columns(self, result: ColumnDetectionResult) -> Dict:
        """Serialize column results."""
        if not result:
            return {}

        return {
            'count': len(result.columns),
            'unique_labels': result.unique_labels,
            'size_mappings': result.size_mappings,
            'items': [
                {
                    'id': col.column_id,
                    'label': col.label,
                    'center': col.center,
                    'size_mm': col.size_mm,
                    'grid_location': col.grid_location,
                    'confidence': col.confidence,
                    'source': col.source
                }
                for col in result.columns
            ]
        }

    def _serialize_beams(self, result: BeamDetectionResult) -> Dict:
        """Serialize beam results."""
        if not result:
            return {}

        return {
            'count': len(result.beams),
            'unique_labels': result.unique_labels,
            'items': [
                {
                    'id': beam.beam_id,
                    'label': beam.label,
                    'from_column': beam.from_column,
                    'to_column': beam.to_column,
                    'length_mm': beam.length_mm,
                    'size_mm': beam.size_mm,
                    'confidence': beam.confidence
                }
                for beam in result.beams
            ]
        }

    def _serialize_footings(self, result: FootingDetectionResult) -> Dict:
        """Serialize footing results."""
        if not result:
            return {}

        return {
            'count': len(result.footings),
            'unique_labels': result.unique_labels,
            'column_map': result.column_footing_map,
            'items': [
                {
                    'id': ftg.footing_id,
                    'label': ftg.label,
                    'type': ftg.footing_type,
                    'size_mm': ftg.size_mm,
                    'associated_columns': ftg.associated_columns,
                    'confidence': ftg.confidence
                }
                for ftg in result.footings
            ]
        }

    def export_csv(
        self,
        output_dir: Path,
        quantity_result: QuantityResult
    ) -> Dict[str, Path]:
        """Export detailed CSV files by element type."""

        paths = {}

        # Group elements by type
        by_type = {}
        for elem in quantity_result.elements:
            etype = elem.element_type
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(elem)

        # Export each type
        for etype, elements in by_type.items():
            path = output_dir / f"{etype}s.csv"

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                if etype == 'column':
                    writer.writerow([
                        'Element ID', 'Label', 'Count',
                        'Width (mm)', 'Depth (mm)', 'Height (mm)',
                        'Concrete (m³)', 'Steel (kg)',
                        'Size Source', 'Steel Source', 'Assumptions'
                    ])
                elif etype == 'beam':
                    writer.writerow([
                        'Element ID', 'Label', 'Count',
                        'Width (mm)', 'Depth (mm)', 'Span (mm)',
                        'Concrete (m³)', 'Steel (kg)',
                        'Size Source', 'Assumptions'
                    ])
                elif etype == 'footing':
                    writer.writerow([
                        'Element ID', 'Label', 'Count',
                        'Length (mm)', 'Width (mm)', 'Depth (mm)',
                        'Concrete (m³)', 'Steel (kg)',
                        'Size Source', 'Assumptions'
                    ])
                elif etype == 'slab':
                    writer.writerow([
                        'Element ID', 'Label', 'Count',
                        'Area (sqm)', 'Thickness (mm)',
                        'Concrete (m³)', 'Steel (kg)',
                        'Assumptions'
                    ])

                # Data rows
                for elem in elements:
                    assumptions_str = '; '.join(elem.assumptions)

                    if etype == 'column':
                        writer.writerow([
                            elem.element_id, elem.label, elem.count,
                            elem.width, elem.depth, elem.length,
                            f"{elem.concrete_volume_m3:.4f}",
                            f"{elem.steel_total_kg:.2f}",
                            elem.size_source, elem.steel_source,
                            assumptions_str
                        ])
                    elif etype == 'beam':
                        writer.writerow([
                            elem.element_id, elem.label, elem.count,
                            elem.width, elem.depth, elem.length,
                            f"{elem.concrete_volume_m3:.4f}",
                            f"{elem.steel_total_kg:.2f}",
                            elem.size_source,
                            assumptions_str
                        ])
                    elif etype == 'footing':
                        writer.writerow([
                            elem.element_id, elem.label, elem.count,
                            elem.length, elem.width, elem.depth,
                            f"{elem.concrete_volume_m3:.4f}",
                            f"{elem.steel_total_kg:.2f}",
                            elem.size_source,
                            assumptions_str
                        ])
                    elif etype == 'slab':
                        area = elem.width * elem.depth / 1e6 if elem.width else 0
                        writer.writerow([
                            elem.element_id, elem.label, elem.count,
                            f"{area:.2f}", elem.length,
                            f"{elem.concrete_volume_m3:.4f}",
                            f"{elem.steel_total_kg:.2f}",
                            assumptions_str
                        ])

            paths[f'{etype}s_csv'] = path
            logger.info(f"Exported {etype}s CSV: {path}")

        return paths

    def export_boq(
        self,
        output_path: Path,
        quantity_result: QuantityResult
    ) -> Path:
        """Export BOQ summary in standard format."""

        summary = quantity_result.summary

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['STRUCTURAL BOQ SUMMARY'])
            writer.writerow([])
            writer.writerow(['Mode:', quantity_result.mode.upper()])
            writer.writerow(['Floors:', quantity_result.floors])
            writer.writerow([])

            # Concrete section
            writer.writerow(['A. CONCRETE WORKS'])
            writer.writerow(['S.No', 'Description', 'Unit', 'Quantity'])
            writer.writerow([1, 'RCC Columns', 'm³', f"{summary.column_concrete_m3:.3f}"])
            writer.writerow([2, 'RCC Beams', 'm³', f"{summary.beam_concrete_m3:.3f}"])
            writer.writerow([3, 'RCC Footings', 'm³', f"{summary.footing_concrete_m3:.3f}"])
            writer.writerow([4, 'RCC Slabs', 'm³', f"{summary.slab_concrete_m3:.3f}"])
            writer.writerow(['', 'TOTAL CONCRETE', 'm³', f"{summary.total_concrete_m3:.3f}"])
            writer.writerow([])

            # Steel section
            writer.writerow(['B. REINFORCEMENT STEEL (TMT Fe500)'])
            writer.writerow(['S.No', 'Description', 'Unit', 'Quantity'])
            writer.writerow([1, 'Column Reinforcement', 'kg', f"{summary.column_steel_kg:.2f}"])
            writer.writerow([2, 'Beam Reinforcement', 'kg', f"{summary.beam_steel_kg:.2f}"])
            writer.writerow([3, 'Footing Reinforcement', 'kg', f"{summary.footing_steel_kg:.2f}"])
            writer.writerow([4, 'Slab Reinforcement', 'kg', f"{summary.slab_steel_kg:.2f}"])
            writer.writerow(['', 'TOTAL STEEL', 'kg', f"{summary.total_steel_kg:.2f}"])
            writer.writerow(['', 'TOTAL STEEL', 'tonnes', f"{summary.total_steel_tonnes:.3f}"])
            writer.writerow([])

            # Element counts
            writer.writerow(['C. ELEMENT COUNTS'])
            writer.writerow(['Element Type', 'Count'])
            writer.writerow(['Columns', summary.column_count])
            writer.writerow(['Beams', summary.beam_count])
            writer.writerow(['Footings', summary.footing_count])
            writer.writerow(['Slabs', summary.slab_count])
            writer.writerow([])

            # Warnings
            if quantity_result.warnings:
                writer.writerow(['D. WARNINGS'])
                for warning in quantity_result.warnings:
                    writer.writerow(['⚠️', warning])

        logger.info(f"Exported BOQ: {output_path}")
        return output_path

    def export_overlays(
        self,
        output_dir: Path,
        image: np.ndarray,
        column_result: ColumnDetectionResult = None,
        beam_result: BeamDetectionResult = None,
        footing_result: FootingDetectionResult = None
    ) -> Dict[str, Path]:
        """Export annotated overlay images."""

        paths = {}

        # Colors (BGR)
        COLORS = {
            'column': (0, 0, 255),     # Red
            'beam': (255, 0, 0),       # Blue
            'footing': (0, 255, 0),    # Green
            'text': (0, 0, 0)          # Black
        }

        # Combined overlay
        combined = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw columns
        if column_result:
            col_overlay = combined.copy()
            for col in column_result.columns:
                x, y, w, h = col.bbox
                cv2.rectangle(col_overlay, (x, y), (x+w, y+h), COLORS['column'], 2)

                # Label
                label = col.label or col.column_id
                cv2.putText(
                    col_overlay, label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['column'], 1
                )

            col_path = output_dir / "overlay_columns.png"
            cv2.imwrite(str(col_path), col_overlay)
            paths['overlay_columns'] = col_path

            # Add to combined
            combined = cv2.addWeighted(combined, 0.7, col_overlay, 0.3, 0)

        # Draw beams
        if beam_result:
            beam_overlay = combined.copy()
            for beam in beam_result.beams:
                pt1 = (int(beam.start_point[0]), int(beam.start_point[1]))
                pt2 = (int(beam.end_point[0]), int(beam.end_point[1]))
                cv2.line(beam_overlay, pt1, pt2, COLORS['beam'], 3)

                # Label at midpoint
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)
                cv2.putText(
                    beam_overlay, beam.label,
                    (mid_x, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['beam'], 1
                )

            beam_path = output_dir / "overlay_beams.png"
            cv2.imwrite(str(beam_path), beam_overlay)
            paths['overlay_beams'] = beam_path

        # Draw footings
        if footing_result:
            ftg_overlay = combined.copy()
            for ftg in footing_result.footings:
                x, y, w, h = ftg.bbox
                cv2.rectangle(ftg_overlay, (x, y), (x+w, y+h), COLORS['footing'], 2)

                # Type indicator
                type_abbr = {'isolated': 'IF', 'combined': 'CF', 'strip': 'SF', 'raft': 'RF'}
                label = f"{ftg.label} ({type_abbr.get(ftg.footing_type, 'F')})"
                cv2.putText(
                    ftg_overlay, label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['footing'], 1
                )

            ftg_path = output_dir / "overlay_footings.png"
            cv2.imwrite(str(ftg_path), ftg_overlay)
            paths['overlay_footings'] = ftg_path

        # Save combined overlay
        combined_path = output_dir / "overlay_combined.png"
        cv2.imwrite(str(combined_path), combined)
        paths['overlay_combined'] = combined_path

        logger.info(f"Exported {len(paths)} overlay images")
        return paths

    def export_qc_report(
        self,
        output_path: Path,
        qc_report: StructuralQCReport
    ) -> Path:
        """Export QC report as JSON."""

        with open(output_path, 'w') as f:
            json.dump(qc_report.to_dict(), f, indent=2)

        logger.info(f"Exported QC report: {output_path}")
        return output_path


def export_structural(
    output_dir: Path,
    plan_id: str,
    quantity_result: QuantityResult = None,
    qc_report: StructuralQCReport = None,
    image: np.ndarray = None
) -> Dict[str, Path]:
    """Convenience function to export all results."""
    exporter = StructuralExporter(output_dir)
    return exporter.export_all(
        plan_id=plan_id,
        quantity_result=quantity_result,
        qc_report=qc_report,
        image=image
    )


if __name__ == "__main__":
    from .quantity_engine import QuantityEngine
    from .qc_structural import StructuralQC

    logging.basicConfig(level=logging.INFO)

    # Generate test data
    engine = QuantityEngine()
    qty_result = engine.compute_assumption(
        total_area_sqm=100,
        floors=4
    )

    qc = StructuralQC()
    qc_report = qc.generate_report(quantity_result=qty_result, mode="assumption")

    # Export
    output_dir = Path(__file__).parent.parent.parent / "out" / "test_structural"
    paths = export_structural(
        output_dir,
        "test_structural",
        quantity_result=qty_result,
        qc_report=qc_report
    )

    print("\nExported files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
