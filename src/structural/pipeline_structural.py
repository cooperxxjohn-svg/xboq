"""
Structural Takeoff Pipeline
Main orchestration module for structural quantity estimation.

Supports two modes:
1. ASSUMPTION MODE: Uses architectural floor plan + assumptions
2. STRUCTURAL MODE: Uses actual structural drawings with schedules
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import cv2

from ..ingest import ingest_plan, IngestedPlan
from ..scale import infer_scale, ScaleResult
from ..pages import classify_drawing, DrawingType, PageClassificationResult
from .detect_columns import ColumnDetector, ColumnDetectionResult
from .detect_beams import BeamDetector, BeamDetectionResult
from .detect_footings import FootingDetector, FootingDetectionResult
from .schedule_extractor import ScheduleExtractor, ScheduleExtractionResult
from .quantity_engine import QuantityEngine, QuantityResult
from .steel_estimator import SteelEstimator
from .qc_structural import StructuralQC, StructuralQCReport
from .export_structural import StructuralExporter

logger = logging.getLogger(__name__)


@dataclass
class StructuralPipelineConfig:
    """Configuration for structural pipeline."""
    # Input
    dpi: int = 300
    default_scale: int = 100  # 1:100

    # Mode
    mode: str = "auto"  # "assumption", "structural", "auto"

    # Building parameters
    floors: int = 1
    storey_height_mm: int = 3000
    building_type: str = "residential"
    column_grid_m: float = 4.0  # For assumption mode

    # Output
    output_dir: Path = Path("./out")

    # Processing options
    detect_columns: bool = True
    detect_beams: bool = True
    detect_footings: bool = True
    extract_schedules: bool = True

    # QC
    generate_qc_report: bool = True
    generate_overlays: bool = True


@dataclass
class StructuralPipelineResult:
    """Result of structural pipeline."""
    success: bool = False
    plan_id: str = ""
    mode: str = "assumption"

    # Detection results
    column_result: Optional[ColumnDetectionResult] = None
    beam_result: Optional[BeamDetectionResult] = None
    footing_result: Optional[FootingDetectionResult] = None
    schedule_result: Optional[ScheduleExtractionResult] = None

    # Quantities
    quantity_result: Optional[QuantityResult] = None

    # QC
    qc_report: Optional[StructuralQCReport] = None

    # Scale
    scale: Optional[ScaleResult] = None

    # Output paths
    output_paths: Dict[str, Path] = field(default_factory=dict)

    # Page classification
    page_types: Dict[int, DrawingType] = field(default_factory=dict)

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StructuralPipeline:
    """
    Main pipeline for structural takeoff.
    """

    def __init__(self, config: StructuralPipelineConfig = None):
        """Initialize pipeline."""
        self.config = config or StructuralPipelineConfig()

        # Initialize components
        self.column_detector = ColumnDetector()
        self.beam_detector = BeamDetector()
        self.footing_detector = FootingDetector()
        self.schedule_extractor = ScheduleExtractor()
        self.quantity_engine = QuantityEngine()
        self.steel_estimator = SteelEstimator()
        self.qc = StructuralQC()

    def process(
        self,
        input_path: Path,
        architectural_plan: Path = None,
        structural_plans: List[Path] = None
    ) -> StructuralPipelineResult:
        """
        Process structural takeoff.

        Args:
            input_path: Main input file (PDF or image)
            architectural_plan: Optional separate architectural plan
            structural_plans: Optional list of structural drawings

        Returns:
            StructuralPipelineResult
        """
        result = StructuralPipelineResult()
        result.plan_id = input_path.stem

        logger.info(f"Starting structural pipeline for: {input_path}")

        try:
            # Step 1: Ingest input
            ingest_result = ingest_plan(input_path, dpi=self.config.dpi)

            if ingest_result.image.size == 0:
                result.errors.append("Ingest failed: no image extracted")
                return result

            # Step 2: Classify pages (for multi-page PDFs)
            page_types = self._classify_pages(ingest_result)
            result.page_types = page_types

            # Step 3: Determine mode
            mode = self._determine_mode(page_types)
            result.mode = mode
            logger.info(f"Operating in {mode.upper()} mode")

            # Step 4: Infer scale
            scale = infer_scale(
                ingest_result.image,
                ingest_result.vector_texts,
                default_scale=self.config.default_scale
            )
            result.scale = scale
            scale_factor = scale.pixels_per_mm

            # Step 5: Run detection based on mode
            if mode == "structural":
                result = self._process_structural_mode(
                    result, ingest_result, page_types, scale_factor
                )
            else:
                result = self._process_assumption_mode(
                    result, ingest_result, scale_factor
                )

            # Step 6: Generate QC report
            if self.config.generate_qc_report:
                result.qc_report = self.qc.generate_report(
                    column_result=result.column_result,
                    beam_result=result.beam_result,
                    footing_result=result.footing_result,
                    quantity_result=result.quantity_result,
                    mode=result.mode,
                    input_files=[str(input_path)]
                )

            # Step 7: Export results
            result.output_paths = self._export_results(result, ingest_result)

            result.success = True

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.errors.append(str(e))

        return result

    def _classify_pages(
        self,
        ingest_result: IngestedPlan
    ) -> Dict[int, DrawingType]:
        """Classify each page in the document."""
        page_types = {}

        # For now, classify single image
        classification = classify_drawing(ingest_result.image, ingest_result.vector_texts)
        page_types[0] = classification.drawing_type
        logger.info(f"Page classification: {classification.drawing_type.value} "
                   f"(confidence: {classification.confidence:.0%})")

        return page_types

    def _determine_mode(self, page_types: Dict[int, DrawingType]) -> str:
        """Determine processing mode based on page types."""
        if self.config.mode != "auto":
            return self.config.mode

        # Check if we have structural drawings
        structural_types = {
            DrawingType.COLUMN_LAYOUT,
            DrawingType.BEAM_LAYOUT,
            DrawingType.FOUNDATION_PLAN,
            DrawingType.STRUCTURAL_FRAMING
        }

        has_structural = any(pt in structural_types for pt in page_types.values())

        if has_structural:
            return "structural"
        else:
            return "assumption"

    def _process_structural_mode(
        self,
        result: StructuralPipelineResult,
        ingest_result: IngestedPlan,
        page_types: Dict[int, DrawingType],
        scale_factor: float
    ) -> StructuralPipelineResult:
        """Process using structural drawings."""
        logger.info("Processing in STRUCTURAL mode")

        # Find relevant pages
        column_pages = [i for i, t in page_types.items()
                       if t == DrawingType.COLUMN_LAYOUT]
        beam_pages = [i for i, t in page_types.items()
                    if t in (DrawingType.BEAM_LAYOUT, DrawingType.STRUCTURAL_FRAMING)]
        foundation_pages = [i for i, t in page_types.items()
                          if t == DrawingType.FOUNDATION_PLAN]
        schedule_pages = [i for i, t in page_types.items()
                        if t == DrawingType.COLUMN_SCHEDULE or t == DrawingType.BEAM_SCHEDULE]

        # Use single image (future: support multi-page PDFs)
        main_image = ingest_result.image
        vector_texts = ingest_result.vector_texts

        # Extract schedules first (for size mappings)
        if self.config.extract_schedules and schedule_pages:
            sched_result = self.schedule_extractor.extract(main_image, vector_texts)
            result.schedule_result = sched_result

        # Detect columns
        if self.config.detect_columns:
            result.column_result = self.column_detector.detect(
                main_image, vector_texts, scale_factor
            )

            # Apply schedule sizes
            if result.schedule_result:
                for col in result.column_result.columns:
                    if col.label in result.schedule_result.column_sizes:
                        col.size_mm = result.schedule_result.column_sizes[col.label]
                        col.source = "schedule"

        # Detect beams
        if self.config.detect_beams and result.column_result:
            result.beam_result = self.beam_detector.detect(
                main_image,
                result.column_result.columns,
                vector_texts,
                scale_factor
            )

            # Apply schedule sizes
            if result.schedule_result:
                for beam in result.beam_result.beams:
                    if beam.label in result.schedule_result.beam_sizes:
                        beam.size_mm = result.schedule_result.beam_sizes[beam.label]
                        beam.source = "schedule"

        # Detect footings
        if self.config.detect_footings:
            result.footing_result = self.footing_detector.detect(
                main_image,
                result.column_result.columns if result.column_result else None,
                vector_texts,
                scale_factor
            )

            # Apply schedule sizes
            if result.schedule_result:
                for ftg in result.footing_result.footings:
                    if ftg.label in result.schedule_result.footing_sizes:
                        ftg.size_mm = result.schedule_result.footing_sizes[ftg.label]
                        ftg.source = "schedule"

        # Compute quantities
        result.quantity_result = self.quantity_engine.compute_structural(
            columns=result.column_result.columns if result.column_result else None,
            beams=result.beam_result.beams if result.beam_result else None,
            footings=result.footing_result.footings if result.footing_result else None,
            column_sizes=result.schedule_result.column_sizes if result.schedule_result else None,
            beam_sizes=result.schedule_result.beam_sizes if result.schedule_result else None,
            footing_sizes=result.schedule_result.footing_sizes if result.schedule_result else None,
            floors=self.config.floors,
            storey_height_mm=self.config.storey_height_mm
        )

        return result

    def _process_assumption_mode(
        self,
        result: StructuralPipelineResult,
        ingest_result: IngestedPlan,
        scale_factor: float
    ) -> StructuralPipelineResult:
        """Process using assumptions from architectural plan."""
        logger.info("Processing in ASSUMPTION mode")

        main_image = ingest_result.image

        # We need room/area data from the architectural pipeline
        # For now, estimate from image dimensions
        h, w = main_image.shape[:2]
        area_sqmm = (w / scale_factor) * (h / scale_factor)
        area_sqm = area_sqmm / 1e6

        # Limit to reasonable building size
        area_sqm = min(area_sqm, 500)  # Cap at 500 sqm

        logger.info(f"Estimated area: {area_sqm:.1f} sqm")

        # Compute quantities using assumptions
        result.quantity_result = self.quantity_engine.compute_assumption(
            total_area_sqm=area_sqm,
            building_type=self.config.building_type,
            floors=self.config.floors,
            column_grid_m=self.config.column_grid_m
        )

        result.warnings.append(
            "Using ASSUMPTION MODE - quantities are estimates based on typical Indian RCC construction. "
            "For accurate takeoff, provide structural drawings."
        )

        return result

    def _export_results(
        self,
        result: StructuralPipelineResult,
        ingest_result: IngestedPlan
    ) -> Dict[str, Path]:
        """Export all results."""
        exporter = StructuralExporter(self.config.output_dir)

        return exporter.export_all(
            plan_id=result.plan_id,
            column_result=result.column_result,
            beam_result=result.beam_result,
            footing_result=result.footing_result,
            quantity_result=result.quantity_result,
            qc_report=result.qc_report,
            image=ingest_result.image if self.config.generate_overlays else None
        )


def process_structural(
    input_path: Path,
    output_dir: Path = None,
    floors: int = 1,
    mode: str = "auto"
) -> StructuralPipelineResult:
    """
    Convenience function to run structural pipeline.

    Args:
        input_path: Input file path
        output_dir: Output directory
        floors: Number of floors
        mode: "assumption", "structural", or "auto"

    Returns:
        StructuralPipelineResult
    """
    config = StructuralPipelineConfig(
        output_dir=output_dir or Path("./out"),
        floors=floors,
        mode=mode
    )

    pipeline = StructuralPipeline(config)
    return pipeline.process(Path(input_path))


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        floors = int(sys.argv[2]) if len(sys.argv) > 2 else 1

        result = process_structural(input_path, floors=floors)

        print(f"\n{'='*60}")
        print(f"STRUCTURAL TAKEOFF RESULT")
        print(f"{'='*60}")
        print(f"Plan: {result.plan_id}")
        print(f"Mode: {result.mode.upper()}")
        print(f"Success: {result.success}")

        if result.quantity_result:
            summary = result.quantity_result.summary
            print(f"\nQUANTITY SUMMARY:")
            print(f"  Total Concrete: {summary.total_concrete_m3:.2f} m³")
            print(f"  Total Steel: {summary.total_steel_kg:.0f} kg ({summary.total_steel_tonnes:.2f} tonnes)")

        if result.qc_report:
            print(f"\nQC REPORT:")
            print(f"  Confidence: {result.qc_report.overall_confidence:.0%}")
            print(f"  Issues: {len(result.qc_report.issues)} "
                  f"({result.qc_report.error_count} errors, {result.qc_report.warning_count} warnings)")

        if result.output_paths:
            print(f"\nOUTPUT FILES:")
            for name, path in result.output_paths.items():
                print(f"  {name}: {path}")
    else:
        print("Usage: python -m src.structural.pipeline_structural <input_file> [floors]")
