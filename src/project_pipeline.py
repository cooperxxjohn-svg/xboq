"""
Multi-Page Project Pipeline.

Processes entire architect drawing sets with:
- Sheet-aware multi-page ingestion
- Per-page scale inference from dimensions
- Room detection only on floor_plan pages
- Comprehensive project summary
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import cv2

from .multipage import (
    MultiPageIngester, SheetClassifier, SheetType,
    PageData, SheetClassification, ProjectManifest
)
from .scale_dimensions import (
    DimensionScaleInferrer, DimensionScaleResult
)
from .scale import ScaleInferrer, ScaleResult, ScaleMethod
from .preprocess import Preprocessor, PreprocessConfig
from .walls import WallDetector
from .regions import RegionDetector
from .polygons import PolygonExtractor
from .labeling import RoomLabeler
from .area import AreaComputer, RoomWithArea
from .ingest import IngestedPlan, PlanType, VectorText


# Inline QC stub for project pipeline (avoids import conflict)
@dataclass
class QCReport:
    """Quality control report."""
    plan_id: str
    is_valid: bool = True
    overall_confidence: float = 0.8
    warnings: List[Dict] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class QualityChecker:
    """Simplified QC checker for project pipeline."""

    TYPICAL_ROOM_SIZES = {
        'Bedroom': (9, 50),
        'Living': (12, 100),
        'Kitchen': (5, 30),
        'Toilet': (2, 15),
        'Bathroom': (3, 20),
    }

    def check(
        self,
        plan_id: str,
        rooms: List[RoomWithArea],
        scale: ScaleResult,
        wall_mask: np.ndarray,
        boundary: Any
    ) -> QCReport:
        """Run quality checks."""
        warnings = []
        statistics = {}

        # Basic statistics
        total_area = sum(r.area.area_sqm for r in rooms)
        statistics["total_area_sqm"] = total_area
        statistics["num_rooms"] = len(rooms)
        statistics["scale_confidence"] = scale.confidence

        # Check for unusual room sizes
        for room in rooms:
            label = room.labeled_room.label.canonical if hasattr(room.labeled_room, 'label') else "Room"
            area = room.area.area_sqm

            if label in self.TYPICAL_ROOM_SIZES:
                min_size, max_size = self.TYPICAL_ROOM_SIZES[label]
                if area < min_size * 0.5 or area > max_size * 2:
                    warnings.append({
                        "code": "unusual_room_size",
                        "message": f"{label} has unusual area: {area:.1f} sqm",
                        "room_id": room.labeled_room.polygon.room_id,
                    })

        # Calculate confidence
        confidence = scale.confidence
        if warnings:
            confidence *= 0.9

        return QCReport(
            plan_id=plan_id,
            is_valid=len(warnings) < 5,
            overall_confidence=confidence,
            warnings=warnings,
            statistics=statistics,
            suggestions=[],
        )

logger = logging.getLogger(__name__)


@dataclass
class PageProcessingResult:
    """Result of processing a single page."""
    page_number: int
    classification: SheetClassification
    scale_result: Optional[DimensionScaleResult] = None
    rooms: List[RoomWithArea] = field(default_factory=list)
    qc_report: Optional[QCReport] = None
    success: bool = False
    skip_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0


@dataclass
class ProjectResult:
    """Result of processing entire project."""
    project_id: str
    source_path: str
    total_pages: int
    pages_processed: int
    pages_skipped: int
    page_results: List[PageProcessingResult] = field(default_factory=list)
    total_rooms: int = 0
    total_area_sqm: float = 0.0
    processing_time_sec: float = 0.0
    manifest: Optional[ProjectManifest] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source_path": self.source_path,
            "total_pages": self.total_pages,
            "pages_processed": self.pages_processed,
            "pages_skipped": self.pages_skipped,
            "total_rooms": self.total_rooms,
            "total_area_sqm": round(self.total_area_sqm, 2),
            "processing_time_sec": round(self.processing_time_sec, 2),
            "errors": self.errors,
            "page_results": [
                {
                    "page_number": pr.page_number,
                    "sheet_type": pr.classification.sheet_type.value,
                    "processed": pr.success,
                    "skip_reason": pr.skip_reason,
                    "scale_method": pr.scale_result.method if pr.scale_result else None,
                    "scale_confidence": pr.scale_result.confidence if pr.scale_result else None,
                    "num_rooms": len(pr.rooms),
                    "area_sqm": sum(r.area.area_sqm for r in pr.rooms) if pr.rooms else 0,
                    "errors": pr.errors,
                }
                for pr in self.page_results
            ]
        }


@dataclass
class ProjectConfig:
    """Configuration for project processing."""
    dpi: int = 300
    default_scale: int = 100
    enable_dimension_scale: bool = True  # Use dimension-based scale
    fallback_to_scale_note: bool = True  # Fallback to 1:X note
    save_debug_images: bool = True
    min_room_area_ratio: float = 0.001
    max_room_area_ratio: float = 0.5
    gap_close_size: int = 15


class ProjectPipeline:
    """
    End-to-end pipeline for multi-page architect drawing sets.
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or ProjectConfig()

        # Initialize components
        self.ingester = MultiPageIngester(dpi=self.config.dpi)
        self.dim_scale_inferrer = DimensionScaleInferrer(dpi=self.config.dpi)
        self.scale_inferrer = ScaleInferrer(dpi=self.config.dpi)
        self.preprocessor = Preprocessor(PreprocessConfig())
        self.wall_detector = WallDetector()
        self.wall_detector.gap_close_size = self.config.gap_close_size
        self.region_detector = RegionDetector()
        self.region_detector.min_room_area_ratio = self.config.min_room_area_ratio
        self.region_detector.max_room_area_ratio = self.config.max_room_area_ratio
        self.polygon_extractor = PolygonExtractor()
        self.labeler = RoomLabeler()
        self.qc_checker = QualityChecker()

    def process_project(
        self,
        file_path: Path,
        output_dir: Path
    ) -> ProjectResult:
        """
        Process a multi-page project.

        Args:
            file_path: Path to PDF or image
            output_dir: Output directory

        Returns:
            ProjectResult
        """
        import time
        start_time = time.time()

        file_path = Path(file_path)
        output_dir = Path(output_dir) / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing project: {file_path}")

        # Step 1: Ingest and classify pages
        logger.info("[1/4] Ingesting and classifying pages...")
        pages_with_class, manifest = self.ingester.ingest_project(
            file_path,
            output_dir if self.config.save_debug_images else None
        )

        project_result = ProjectResult(
            project_id=file_path.stem,
            source_path=str(file_path),
            total_pages=len(pages_with_class),
            pages_processed=0,
            pages_skipped=0,
            manifest=manifest
        )

        # Step 2: Process each page
        logger.info("[2/4] Processing individual pages...")
        page_results = []

        for page_data, classification in pages_with_class:
            page_start = time.time()
            page_num = page_data.page_number

            logger.info(f"  Page {page_num + 1}: {classification.sheet_type.value}")

            if not classification.should_process_rooms:
                # Skip non-floor-plan pages
                page_results.append(PageProcessingResult(
                    page_number=page_num,
                    classification=classification,
                    success=False,
                    skip_reason=f"Sheet type: {classification.sheet_type.value}",
                    processing_time_sec=time.time() - page_start
                ))
                project_result.pages_skipped += 1
                continue

            # Process floor plan page
            try:
                page_result = self._process_floor_plan_page(
                    page_data, classification, output_dir
                )
                page_result.processing_time_sec = time.time() - page_start
                page_results.append(page_result)

                if page_result.success:
                    project_result.pages_processed += 1
                    project_result.total_rooms += len(page_result.rooms)
                    project_result.total_area_sqm += sum(
                        r.area.area_sqm for r in page_result.rooms
                    )
                else:
                    project_result.pages_skipped += 1

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                page_results.append(PageProcessingResult(
                    page_number=page_num,
                    classification=classification,
                    success=False,
                    errors=[str(e)],
                    processing_time_sec=time.time() - page_start
                ))
                project_result.pages_skipped += 1

        project_result.page_results = page_results

        # Step 3: Generate project summary
        logger.info("[3/4] Generating project summary...")
        self._generate_summary(project_result, output_dir)

        # Step 4: Save manifest
        logger.info("[4/4] Saving manifest...")
        manifest_path = output_dir / "manifest.json"
        manifest.save(manifest_path)

        # Save full results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(project_result.to_dict(), f, indent=2)

        project_result.processing_time_sec = time.time() - start_time

        logger.info(f"Project complete: {project_result.pages_processed} pages processed, "
                   f"{project_result.total_rooms} rooms, "
                   f"{project_result.total_area_sqm:.1f} sqm")

        return project_result

    def _process_floor_plan_page(
        self,
        page_data: PageData,
        classification: SheetClassification,
        output_dir: Path
    ) -> PageProcessingResult:
        """Process a single floor plan page."""
        result = PageProcessingResult(
            page_number=page_data.page_number,
            classification=classification,
        )

        image = page_data.image
        page_num = page_data.page_number

        # Convert vector texts to expected format
        vector_texts = [
            VectorText(
                text=vt.get('text', ''),
                bbox=vt.get('bbox', (0, 0, 0, 0)),
                font_size=vt.get('size', 10)
            )
            for vt in page_data.vector_texts
        ]

        # Step A: Infer scale from dimensions
        if self.config.enable_dimension_scale:
            debug_path = None
            if self.config.save_debug_images:
                debug_path = output_dir / "debug" / f"page_{page_num + 1:03d}_scale_dims.png"

            dim_scale = self.dim_scale_inferrer.infer_scale(
                image, vector_texts
            )

            if debug_path:
                self.dim_scale_inferrer.save_debug_image(image, dim_scale, debug_path)

            if dim_scale.confidence > 0.5 and dim_scale.pixels_per_mm > 0:
                result.scale_result = dim_scale
                logger.info(f"    Scale from dimensions: {dim_scale.pixels_per_mm:.4f} px/mm "
                           f"(conf: {dim_scale.confidence:.0%})")
            else:
                logger.info(f"    Dimension scale low confidence, trying fallback...")

        # Fallback to scale note if needed
        if result.scale_result is None and self.config.fallback_to_scale_note:
            note_scale = self.scale_inferrer.infer_scale(
                image, vector_texts, self.config.default_scale
            )
            if note_scale.confidence > 0.5:
                result.scale_result = DimensionScaleResult(
                    pixels_per_mm=note_scale.pixels_per_mm,
                    confidence=note_scale.confidence,
                    method=f"scale_note_1:{note_scale.scale_ratio}" if note_scale.scale_ratio else "fallback"
                )
                logger.info(f"    Scale from note: 1:{note_scale.scale_ratio} "
                           f"(conf: {note_scale.confidence:.0%})")
            else:
                # Use default
                result.scale_result = DimensionScaleResult(
                    pixels_per_mm=self.config.dpi / 25.4 / self.config.default_scale,
                    confidence=0.3,
                    method="default",
                    warnings=["Using default scale"]
                )

        # Step B: Preprocess
        preprocess = self.preprocessor.process(image)

        # Step C: Create IngestedPlan for compatibility
        plan = IngestedPlan(
            plan_id=f"page_{page_num + 1}",
            source_path=Path(f"page_{page_num + 1}"),
            plan_type=PlanType.IMAGE,
            image=image,
            image_dpi=self.config.dpi,
            vector_texts=vector_texts,
        )

        # Step D: Detect walls
        walls = self.wall_detector.detect(plan, preprocess.cleaned)

        if len(walls.wall_segments) == 0:
            result.errors.append("No walls detected")
            return result

        # Step E: Detect regions
        regions = self.region_detector.detect(
            walls.wall_mask_closed,
            walls.external_boundary
        )

        valid_regions = [r for r in regions.regions if r.is_valid and not r.is_hole]
        if not valid_regions:
            result.errors.append("No valid room regions detected")
            return result

        # Step F: Extract polygons
        polygons = []
        for region in valid_regions:
            polygon = self.polygon_extractor.extract(
                region.mask,
                f"R{region.region_id:03d}"
            )
            if polygon.is_valid:
                polygons.append(polygon)

        # Step G: Label rooms
        labeled_rooms = self.labeler.label_rooms(
            polygons, vector_texts, image
        )

        # Step H: Compute areas
        scale_result = ScaleResult(
            method=ScaleMethod.DIMENSION if "dimension" in result.scale_result.method else ScaleMethod.DEFAULT,
            pixels_per_mm=result.scale_result.pixels_per_mm,
            confidence=result.scale_result.confidence
        )

        area_computer = AreaComputer(scale_result)
        rooms_with_area = area_computer.compute_all(labeled_rooms)

        result.rooms = rooms_with_area
        result.success = True

        # Step I: QC check
        qc_report = self.qc_checker.check(
            f"page_{page_num + 1}",
            rooms_with_area,
            scale_result,
            walls.wall_mask_closed,
            walls.external_boundary
        )
        result.qc_report = qc_report

        logger.info(f"    Detected {len(rooms_with_area)} rooms, "
                   f"total area: {sum(r.area.area_sqm for r in rooms_with_area):.1f} sqm")

        return result

    def _generate_summary(
        self,
        result: ProjectResult,
        output_dir: Path
    ) -> None:
        """Generate summary.md with project overview."""
        summary_path = output_dir / "summary.md"

        with open(summary_path, "w") as f:
            f.write(f"# Project Summary: {result.project_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Overview
            f.write("## Overview\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Source | `{result.source_path}` |\n")
            f.write(f"| Total Pages | {result.total_pages} |\n")
            f.write(f"| Pages Processed | {result.pages_processed} |\n")
            f.write(f"| Pages Skipped | {result.pages_skipped} |\n")
            f.write(f"| Total Rooms | {result.total_rooms} |\n")
            f.write(f"| Total Area | {result.total_area_sqm:.1f} sqm |\n")
            f.write(f"| Processing Time | {result.processing_time_sec:.1f}s |\n")
            f.write("\n")

            # Page Classification
            f.write("## Page Classification\n\n")
            f.write("| Page | Sheet Type | Confidence | Processed | Reason |\n")
            f.write("|------|------------|------------|-----------|--------|\n")

            for pr in result.page_results:
                processed = "Yes" if pr.success else "No"
                reason = pr.skip_reason or (pr.classification.reason[:40] + "..." if len(pr.classification.reason) > 40 else pr.classification.reason)
                f.write(f"| {pr.page_number + 1} | {pr.classification.sheet_type.value} | "
                       f"{pr.classification.confidence:.0%} | {processed} | {reason} |\n")

            f.write("\n")

            # Scale Information
            f.write("## Scale Detection\n\n")
            f.write("| Page | Method | Pixels/mm | Confidence | Warnings |\n")
            f.write("|------|--------|-----------|------------|----------|\n")

            for pr in result.page_results:
                if pr.scale_result:
                    sr = pr.scale_result
                    warnings = ", ".join(sr.warnings[:2]) if sr.warnings else "-"
                    f.write(f"| {pr.page_number + 1} | {sr.method} | "
                           f"{sr.pixels_per_mm:.4f} | {sr.confidence:.0%} | {warnings} |\n")

            f.write("\n")

            # Room Summary (for processed pages)
            f.write("## Room Summary\n\n")

            for pr in result.page_results:
                if pr.success and pr.rooms:
                    f.write(f"### Page {pr.page_number + 1}\n\n")
                    f.write("| Room | Label | Area (sqm) | Area (sqft) |\n")
                    f.write("|------|-------|------------|-------------|\n")

                    for room in pr.rooms:
                        label = room.labeled_room.label.canonical if hasattr(room.labeled_room, 'label') else "Room"
                        f.write(f"| {room.labeled_room.polygon.room_id} | {label} | "
                               f"{room.area.area_sqm:.1f} | {room.area.area_sqft:.1f} |\n")

                    total = sum(r.area.area_sqm for r in pr.rooms)
                    f.write(f"| **Total** | | **{total:.1f}** | **{total * 10.764:.1f}** |\n")
                    f.write("\n")

            # Warnings and Errors
            if any(pr.errors for pr in result.page_results):
                f.write("## Warnings and Errors\n\n")
                for pr in result.page_results:
                    if pr.errors:
                        f.write(f"**Page {pr.page_number + 1}:**\n")
                        for err in pr.errors:
                            f.write(f"- {err}\n")
                        f.write("\n")

        logger.info(f"Saved project summary to: {summary_path}")


def process_project(
    file_path: Path,
    output_dir: Path,
    config: Optional[ProjectConfig] = None
) -> ProjectResult:
    """
    Process a multi-page architect drawing set.

    Args:
        file_path: Path to PDF or image
        output_dir: Output directory
        config: Optional configuration

    Returns:
        ProjectResult
    """
    pipeline = ProjectPipeline(config)
    return pipeline.process_project(file_path, output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")

        result = process_project(file_path, output_dir)

        print(f"\nProject: {result.project_id}")
        print(f"Pages processed: {result.pages_processed}/{result.total_pages}")
        print(f"Total rooms: {result.total_rooms}")
        print(f"Total area: {result.total_area_sqm:.1f} sqm")
        print(f"Processing time: {result.processing_time_sec:.1f}s")
        print(f"\nOutputs saved to: {output_dir / result.project_id}/")
    else:
        print("Usage: python project_pipeline.py <file.pdf> [output_dir]")
