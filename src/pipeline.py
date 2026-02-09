"""
Floor Plan Analysis Pipeline
Main entry point for processing floor plans end-to-end.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .ingest import PlanIngester, IngestedPlan
from .preprocess import Preprocessor, PreprocessResult, PreprocessConfig
from .scale import ScaleInferrer, ScaleResult
from .walls import WallDetector, WallDetectionResult
from .regions import RegionDetector, RegionDetectionResult
from .polygons import PolygonExtractor, RoomPolygon
from .labeling import RoomLabeler, LabeledRoom
from .area import AreaComputer, RoomWithArea
from .qc import QualityChecker, QCReport
from .export import PlanExporter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    # Ingestion
    dpi: int = 300

    # Preprocessing
    enable_deskew: bool = True
    enable_denoise: bool = True
    adaptive_block_size: int = 35

    # Scale
    default_scale: int = 100

    # Walls
    gap_close_size: int = 15

    # Regions
    min_room_area_ratio: float = 0.001
    max_room_area_ratio: float = 0.5

    # Output
    output_dir: Path = Path("./out")


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    plan_id: str
    success: bool
    plan: Optional[IngestedPlan] = None
    preprocess: Optional[PreprocessResult] = None
    scale: Optional[ScaleResult] = None
    walls: Optional[WallDetectionResult] = None
    regions: Optional[RegionDetectionResult] = None
    polygons: List[RoomPolygon] = field(default_factory=list)
    labeled_rooms: List[LabeledRoom] = field(default_factory=list)
    rooms_with_area: List[RoomWithArea] = field(default_factory=list)
    qc_report: Optional[QCReport] = None
    output_paths: Dict[str, Path] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class FloorPlanPipeline:
    """
    End-to-end floor plan analysis pipeline.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.ingester = PlanIngester(dpi=self.config.dpi)
        self.preprocessor = Preprocessor(PreprocessConfig(
            enable_deskew=self.config.enable_deskew,
            enable_denoise=self.config.enable_denoise,
            adaptive_block_size=self.config.adaptive_block_size
        ))
        self.scale_inferrer = ScaleInferrer(dpi=self.config.dpi)
        self.wall_detector = WallDetector()
        self.wall_detector.gap_close_size = self.config.gap_close_size
        self.region_detector = RegionDetector()
        self.region_detector.min_room_area_ratio = self.config.min_room_area_ratio
        self.region_detector.max_room_area_ratio = self.config.max_room_area_ratio
        self.polygon_extractor = PolygonExtractor()
        self.labeler = RoomLabeler()
        self.qc_checker = QualityChecker()
        self.exporter = PlanExporter(self.config.output_dir)

    def process(self, file_path: Path, page_num: int = 0) -> PipelineResult:
        """
        Process a single floor plan file.

        Args:
            file_path: Path to floor plan file
            page_num: Page number for multi-page PDFs

        Returns:
            PipelineResult
        """
        file_path = Path(file_path)
        result = PipelineResult(
            plan_id=file_path.stem,
            success=False
        )

        try:
            # Step 1: Ingest
            logger.info(f"[1/8] Ingesting plan: {file_path}")
            plan = self.ingester.ingest(file_path, page_num)
            result.plan = plan

            if plan.image.size == 0:
                result.errors.append("Failed to load plan image")
                return result

            # Step 2: Preprocess
            logger.info("[2/8] Preprocessing image")
            preprocess = self.preprocessor.process(plan.image)
            result.preprocess = preprocess

            # Step 3: Infer scale
            logger.info("[3/8] Inferring scale")
            scale = self.scale_inferrer.infer_scale(
                plan.image,
                plan.vector_texts,
                self.config.default_scale
            )
            result.scale = scale
            logger.info(f"  Scale: {scale.method.value}, confidence: {scale.confidence:.0%}")

            # Step 4: Detect walls
            logger.info("[4/8] Detecting walls")
            walls = self.wall_detector.detect(plan, preprocess.cleaned)
            result.walls = walls
            logger.info(f"  Found {len(walls.wall_segments)} wall segments")

            # Step 5: Detect regions
            logger.info("[5/8] Detecting regions")
            regions = self.region_detector.detect(
                walls.wall_mask_closed,
                walls.external_boundary
            )
            result.regions = regions
            valid_regions = [r for r in regions.regions if r.is_valid and not r.is_hole]
            logger.info(f"  Found {len(valid_regions)} valid regions")

            if not valid_regions:
                result.errors.append("No valid room regions detected")
                # Continue anyway to generate outputs

            # Step 6: Extract polygons
            logger.info("[6/8] Extracting polygons")
            polygons = []
            for region in valid_regions:
                polygon = self.polygon_extractor.extract(
                    region.mask,
                    f"R{region.region_id:03d}"
                )
                if polygon.is_valid:
                    polygons.append(polygon)

            result.polygons = polygons
            logger.info(f"  Extracted {len(polygons)} valid polygons")

            # Step 7: Label rooms
            logger.info("[7/8] Labeling rooms")
            labeled_rooms = self.labeler.label_rooms(
                polygons,
                plan.vector_texts,
                plan.image
            )
            result.labeled_rooms = labeled_rooms

            labeled_count = len([r for r in labeled_rooms if r.label.canonical not in ('Room', 'Unknown')])
            logger.info(f"  Labeled {labeled_count}/{len(labeled_rooms)} rooms")

            # Step 8: Compute areas and run QC
            logger.info("[8/8] Computing areas and running QC")
            area_computer = AreaComputer(scale)
            rooms_with_area = area_computer.compute_all(labeled_rooms)
            result.rooms_with_area = rooms_with_area

            total_area = sum(r.area.area_sqm for r in rooms_with_area)
            logger.info(f"  Total area: {total_area:.1f} sqm")

            # Run QC
            qc_report = self.qc_checker.check(
                plan.plan_id,
                rooms_with_area,
                scale,
                walls.wall_mask_closed,
                walls.external_boundary
            )
            result.qc_report = qc_report

            logger.info(f"  QC: {'PASS' if qc_report.is_valid else 'FAIL'}, "
                       f"confidence: {qc_report.overall_confidence:.0%}, "
                       f"warnings: {len(qc_report.warnings)}")

            # Export results
            logger.info("Exporting results...")
            output_paths = self.exporter.export_all(
                plan,
                rooms_with_area,
                scale,
                qc_report,
                walls.wall_mask_closed
            )
            result.output_paths = output_paths

            result.success = True
            logger.info(f"Pipeline completed for {plan.plan_id}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.errors.append(str(e))

        return result

    def process_batch(self, file_paths: List[Path]) -> List[PipelineResult]:
        """
        Process multiple floor plans.

        Args:
            file_paths: List of file paths

        Returns:
            List of PipelineResult objects
        """
        results = []

        for i, path in enumerate(file_paths, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing plan {i}/{len(file_paths)}: {path.name}")
            logger.info(f"{'='*60}")

            result = self.process(path)
            results.append(result)

        # Summary
        success_count = len([r for r in results if r.success])
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch complete: {success_count}/{len(results)} successful")
        logger.info(f"{'='*60}")

        return results


def process_plan(
    file_path: Path,
    output_dir: Path = Path("./out"),
    config: Optional[PipelineConfig] = None
) -> PipelineResult:
    """
    Process a single floor plan.

    Args:
        file_path: Path to floor plan
        output_dir: Output directory
        config: Optional configuration

    Returns:
        PipelineResult
    """
    if config is None:
        config = PipelineConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    pipeline = FloorPlanPipeline(config)
    return pipeline.process(file_path)


def process_batch(
    file_paths: List[Path],
    output_dir: Path = Path("./out"),
    config: Optional[PipelineConfig] = None
) -> List[PipelineResult]:
    """
    Process multiple floor plans.

    Args:
        file_paths: List of file paths
        output_dir: Output directory
        config: Optional configuration

    Returns:
        List of PipelineResult
    """
    if config is None:
        config = PipelineConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    pipeline = FloorPlanPipeline(config)
    return pipeline.process_batch(file_paths)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")

        result = process_plan(file_path, output_dir)

        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Rooms detected: {len(result.rooms_with_area)}")
        if result.qc_report:
            print(f"Total area: {result.qc_report.statistics.get('total_area_sqm', 0)} sqm")
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.output_paths:
            print(f"Outputs: {list(result.output_paths.keys())}")
    else:
        print("Usage: python pipeline.py <plan_file> [output_dir]")
