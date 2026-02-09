"""
Project Exporter - Generates summary and BOQ outputs.

Phase 4: Export:
- summary.md (project-level report)
- boq_quantities.csv (unified BOQ)
"""

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from .indexer import ProjectIndex
from .router import RoutingResult, PageType
from .runner import RunnerResult, ExtractionResult
from .joiner import ProjectGraph

logger = logging.getLogger(__name__)


@dataclass
class BOQEntry:
    """A single BOQ line item with provenance."""
    category: str
    item_code: str
    description: str
    unit: str
    quantity: float
    source_page: str
    notes: str = ""
    # Provenance columns
    source_pages: str = ""  # Comma-separated page refs
    measured_or_assumed: str = "measured"
    provenance_ids: str = ""  # Linked object IDs
    detection_method: str = ""
    confidence: float = 1.0

    def to_row(self) -> List[str]:
        return [
            self.category,
            self.item_code,
            self.description,
            self.unit,
            f"{self.quantity:.2f}",
            self.source_page,
            self.notes,
            self.source_pages,
            self.measured_or_assumed,
            self.provenance_ids,
        ]


class ProjectExporter:
    """
    Exports project results to summary and BOQ formats.
    """

    BOQ_HEADERS = [
        "Category",
        "Item Code",
        "Description",
        "Unit",
        "Quantity",
        "Source Page",
        "Notes",
        "Source Pages",
        "Measured/Assumed",
        "Provenance IDs",
    ]

    def __init__(self):
        pass

    def export_all(
        self,
        project_index: ProjectIndex,
        routing_result: RoutingResult,
        runner_result: RunnerResult,
        project_graph: ProjectGraph,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Export all outputs.

        Args:
            project_index: Project index
            routing_result: Routing result
            runner_result: Runner result
            project_graph: Project graph
            output_dir: Output directory

        Returns:
            Dict of output paths
        """
        output_dir = Path(output_dir) / project_index.project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        boq_dir = output_dir / "boq"
        boq_dir.mkdir(exist_ok=True)

        outputs = {}

        # Generate summary
        summary_path = output_dir / "summary.md"
        self._export_summary(
            summary_path,
            project_index,
            routing_result,
            runner_result,
            project_graph,
        )
        outputs["summary"] = summary_path

        # Generate BOQ
        boq_path = boq_dir / "boq_quantities.csv"
        self._export_boq(boq_path, runner_result, project_graph)
        outputs["boq"] = boq_path

        # Generate room summary CSV
        rooms_path = boq_dir / "rooms_summary.csv"
        self._export_rooms(rooms_path, runner_result)
        outputs["rooms"] = rooms_path

        logger.info(f"Exported all outputs to: {output_dir}")

        return outputs

    def _export_summary(
        self,
        path: Path,
        project_index: ProjectIndex,
        routing_result: RoutingResult,
        runner_result: RunnerResult,
        project_graph: ProjectGraph,
    ) -> None:
        """Export project summary markdown."""
        with open(path, "w") as f:
            f.write(f"# Project Summary: {project_index.project_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Overview
            f.write("## Overview\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Source | `{project_index.source_path}` |\n")
            f.write(f"| Total Files | {project_index.total_files} |\n")
            f.write(f"| Total Pages | {project_index.total_pages} |\n")
            f.write(f"| Processing Time | {runner_result.total_time_sec:.1f}s |\n")
            f.write("\n")

            # Page Classification
            f.write("## Page Classification\n\n")
            f.write("| Page Type | Count |\n")
            f.write("|-----------|-------|\n")
            for ptype, count in sorted(routing_result.type_counts.items()):
                if count > 0:
                    f.write(f"| {ptype} | {count} |\n")
            f.write("\n")

            # Processing Results
            f.write("## Processing Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Pages Processed | {runner_result.processed_pages} |\n")
            f.write(f"| Pages Skipped | {runner_result.skipped_pages} |\n")
            f.write(f"| Pages Failed | {runner_result.failed_pages} |\n")
            f.write(f"| Success Rate | {runner_result.summary.get('extraction_success_rate', 0):.1%} |\n")
            f.write("\n")

            # Extraction Summary
            f.write("## Extraction Summary\n\n")
            summary = runner_result.summary
            f.write(f"- **Total Rooms Detected:** {summary.get('total_rooms_detected', 0)}\n")
            f.write(f"- **Total Area:** {summary.get('total_area_sqm', 0):.1f} sqm "
                   f"({summary.get('total_area_sqm', 0) * 10.764:.1f} sqft)\n")
            f.write(f"- **Schedules Extracted:** {summary.get('schedules_extracted', 0)}\n")
            f.write(f"- **Floor Plans Processed:** {summary.get('floor_plans_processed', 0)}\n")
            f.write("\n")

            # Schedule Summary
            f.write("## Schedules Found\n\n")
            if project_graph.schedules:
                for stype, entries in sorted(project_graph.schedules.items()):
                    f.write(f"### {stype.title()} Schedule\n")
                    f.write(f"- Entries: {len(entries)}\n")
                    if entries:
                        f.write(f"- Sample tags: {', '.join(e.tag for e in entries[:5])}\n")
                    f.write("\n")
            else:
                f.write("*No schedules extracted*\n\n")

            # Mapping Coverage
            f.write("## Mapping Coverage\n\n")
            coverage = project_graph.coverage
            f.write("| Metric | Coverage |\n")
            f.write("|--------|----------|\n")
            for metric, value in sorted(coverage.items()):
                f.write(f"| {metric} | {value:.1%} |\n")
            f.write("\n")

            # Unresolved Items
            if project_graph.unresolved_drawing_tags or project_graph.unresolved_schedule_tags:
                f.write("## Unresolved Tags\n\n")
                if project_graph.unresolved_drawing_tags:
                    f.write(f"**In drawings but not schedules:** "
                           f"{', '.join(project_graph.unresolved_drawing_tags[:10])}")
                    if len(project_graph.unresolved_drawing_tags) > 10:
                        f.write(f" (+{len(project_graph.unresolved_drawing_tags) - 10} more)")
                    f.write("\n\n")
                if project_graph.unresolved_schedule_tags:
                    f.write(f"**In schedules but not drawings:** "
                           f"{', '.join(project_graph.unresolved_schedule_tags[:10])}")
                    if len(project_graph.unresolved_schedule_tags) > 10:
                        f.write(f" (+{len(project_graph.unresolved_schedule_tags) - 10} more)")
                    f.write("\n\n")

            # Floor Plans Detail
            f.write("## Floor Plans\n\n")
            floor_plan_results = [
                r for r in runner_result.extraction_results
                if r.page_type == "floor_plan" and r.success
            ]

            if floor_plan_results:
                for result in floor_plan_results:
                    page_label = f"{Path(result.file_path).stem} p{result.page_number + 1}"
                    total_area = sum(r.get("area_sqm", 0) for r in result.rooms)
                    f.write(f"### {page_label}\n")
                    f.write(f"- Rooms: {len(result.rooms)}\n")
                    f.write(f"- Total Area: {total_area:.1f} sqm\n")

                    if result.scale_info:
                        f.write(f"- Scale: {result.scale_info.get('method', 'unknown')} "
                               f"(conf: {result.scale_info.get('confidence', 0):.0%})\n")

                    if result.rooms:
                        f.write("\n| Room | Label | Area (sqm) |\n")
                        f.write("|------|-------|------------|\n")
                        for room in result.rooms[:10]:
                            f.write(f"| {room.get('room_id', '')} | "
                                   f"{room.get('label', '')} | "
                                   f"{room.get('area_sqm', 0):.1f} |\n")
                        if len(result.rooms) > 10:
                            f.write(f"| ... | +{len(result.rooms) - 10} more | |\n")
                    f.write("\n")
            else:
                f.write("*No floor plans processed*\n\n")

            # Warnings
            warnings = []
            for result in runner_result.extraction_results:
                for w in result.warnings:
                    warnings.append(f"{Path(result.file_path).stem} p{result.page_number + 1}: {w}")
            if warnings:
                f.write("## Warnings\n\n")
                for w in warnings[:20]:
                    f.write(f"- {w}\n")
                if len(warnings) > 20:
                    f.write(f"\n*+{len(warnings) - 20} more warnings*\n")
                f.write("\n")

            # Outputs
            f.write("## Output Files\n\n")
            f.write("- `summary.md` - This summary\n")
            f.write("- `boq/boq_quantities.csv` - Unified BOQ\n")
            f.write("- `boq/rooms_summary.csv` - Room area summary\n")
            f.write("- `project_graph.json` - Full project graph\n")
            f.write("- `unresolved_tags.csv` - Unresolved tags report\n")
            f.write("- `manifest.json` - Page routing manifest\n")
            f.write("- `index.json` - Page index\n")

        logger.info(f"Saved summary to: {path}")

    def _export_boq(
        self,
        path: Path,
        runner_result: RunnerResult,
        project_graph: ProjectGraph,
    ) -> None:
        """Export BOQ CSV with provenance columns."""
        entries: List[BOQEntry] = []

        # Add room areas with provenance
        for result in runner_result.extraction_results:
            if result.page_type != "floor_plan":
                continue

            page_label = f"{Path(result.file_path).stem}_p{result.page_number + 1}"

            for room in result.rooms:
                label = room.get("label", "Room")
                area = room.get("area_sqm", 0)

                if area > 0:
                    # Get provenance info from room dict
                    is_measured = room.get("is_measured", True)
                    detection_method = room.get("detection_method", "raster")
                    confidence = room.get("confidence", 0.8)
                    scale_method = room.get("scale_method", "unknown")

                    entries.append(BOQEntry(
                        category="Floor Area",
                        item_code=room.get("room_id", ""),
                        description=f"{label}",
                        unit="sqm",
                        quantity=area,
                        source_page=page_label,
                        # Provenance columns
                        source_pages=page_label,
                        measured_or_assumed="measured" if is_measured else "assumed",
                        provenance_ids=room.get("room_id", ""),
                        detection_method=detection_method,
                        confidence=confidence,
                    ))

        # Add schedule entries with provenance
        for stype, schedule_entries in project_graph.schedules.items():
            for entry in schedule_entries:
                props = entry.properties
                page_label = f"{Path(entry.source_file).stem}_p{entry.source_page + 1}"

                if stype == "door":
                    entries.append(BOQEntry(
                        category="Doors",
                        item_code=entry.tag,
                        description=f"Door {props.get('width', 0)}x{props.get('height', 0)}mm",
                        unit="nos",
                        quantity=1,
                        source_page=page_label,
                        source_pages=page_label,
                        measured_or_assumed="measured",
                        provenance_ids=f"sched_{entry.tag}",
                        detection_method="schedule",
                        confidence=0.9,
                    ))
                elif stype == "window":
                    entries.append(BOQEntry(
                        category="Windows",
                        item_code=entry.tag,
                        description=f"Window {props.get('width', 0)}x{props.get('height', 0)}mm",
                        unit="nos",
                        quantity=1,
                        source_page=page_label,
                        source_pages=page_label,
                        measured_or_assumed="measured",
                        provenance_ids=f"sched_{entry.tag}",
                        detection_method="schedule",
                        confidence=0.9,
                    ))
                elif stype == "bbs":
                    entries.append(BOQEntry(
                        category="Reinforcement",
                        item_code=entry.tag,
                        description=f"Rebar dia {props.get('diameter', 0)}mm",
                        unit="nos",
                        quantity=props.get("quantity", 1),
                        source_page=page_label,
                        notes=f"Length: {props.get('length', 0)}mm",
                        source_pages=page_label,
                        measured_or_assumed="measured",
                        provenance_ids=f"sched_{entry.tag}",
                        detection_method="schedule",
                        confidence=0.85,
                    ))

        # Write CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.BOQ_HEADERS)
            for entry in entries:
                writer.writerow(entry.to_row())

        logger.info(f"Saved BOQ with {len(entries)} entries to: {path}")

    def _export_rooms(
        self,
        path: Path,
        runner_result: RunnerResult,
    ) -> None:
        """Export rooms summary CSV."""
        headers = ["Page", "Room ID", "Label", "Area (sqm)", "Area (sqft)"]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for result in runner_result.extraction_results:
                if result.page_type != "floor_plan":
                    continue

                page_label = f"{Path(result.file_path).stem}_p{result.page_number + 1}"

                for room in result.rooms:
                    writer.writerow([
                        page_label,
                        room.get("room_id", ""),
                        room.get("label", ""),
                        f"{room.get('area_sqm', 0):.2f}",
                        f"{room.get('area_sqft', 0):.2f}",
                    ])

        logger.info(f"Saved rooms summary to: {path}")


def export_project(
    project_index: ProjectIndex,
    routing_result: RoutingResult,
    runner_result: RunnerResult,
    project_graph: ProjectGraph,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Convenience function to export project.

    Args:
        project_index: Project index
        routing_result: Routing result
        runner_result: Runner result
        project_graph: Project graph
        output_dir: Output directory

    Returns:
        Dict of output paths
    """
    exporter = ProjectExporter()
    return exporter.export_all(
        project_index, routing_result, runner_result, project_graph, output_dir
    )


if __name__ == "__main__":
    print("Use process.py to run the full pipeline with exports.")
