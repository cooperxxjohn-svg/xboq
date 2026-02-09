"""
Project Runner - Targeted extraction with parallelism.

Phase 2: Run appropriate extractors based on page type:
- floor_plan -> rooms/walls/openings + scale
- schedule_table -> table extraction
- structural_plan -> structural detection (optional)
- others -> text summary only
"""

import json
import logging
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import time

from .indexer import ProjectIndex, PageIndex
from .router import RoutingResult, PageRouting, PageType

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for project runner."""
    dpi: int = 300
    default_scale: int = 100
    enable_dimension_scale: bool = True
    enable_structural: bool = False
    only_types: Optional[Set[str]] = None  # None = all, or {"floor_plan", "schedule_table"}
    max_workers: int = 0  # 0 = cpu_count // 2
    resume: bool = True
    save_debug_images: bool = False


@dataclass
class ExtractionResult:
    """Result of extracting a single page."""
    file_path: str
    page_number: int
    page_type: str
    success: bool
    processing_time_sec: float
    rooms: List[Dict] = field(default_factory=list)
    openings: List[Dict] = field(default_factory=list)
    walls: List[Dict] = field(default_factory=list)
    schedule_data: Optional[Dict] = None
    scale_info: Optional[Dict] = None
    wall_info: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Provenance tracking
    provenance: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunnerResult:
    """Result of running all extractions."""
    project_id: str
    total_pages: int
    processed_pages: int
    skipped_pages: int
    failed_pages: int
    extraction_results: List[ExtractionResult]
    total_time_sec: float
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "skipped_pages": self.skipped_pages,
            "failed_pages": self.failed_pages,
            "total_time_sec": round(self.total_time_sec, 2),
            "summary": self.summary,
            "extraction_results": [r.to_dict() for r in self.extraction_results],
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ProjectRunner:
    """
    Runs targeted extraction on routed pages.

    Uses multiprocessing for parallel extraction.
    Supports resume via completed results cache.
    """

    PROCESSABLE_TYPES = {
        PageType.FLOOR_PLAN,
        PageType.SCHEDULE_TABLE,
        PageType.STRUCTURAL_PLAN,
    }

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()
        if self.config.max_workers == 0:
            self.config.max_workers = max(1, mp.cpu_count() // 2)

    def run(
        self,
        project_index: ProjectIndex,
        routing_result: RoutingResult,
        output_dir: Path,
    ) -> RunnerResult:
        """
        Run extraction on all routed pages.

        Args:
            project_index: Project index
            routing_result: Routing result
            output_dir: Output directory

        Returns:
            RunnerResult
        """
        start_time = time.time()
        output_dir = Path(output_dir) / project_index.project_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing results for resume
        results_path = output_dir / "extraction_results.json"
        completed = {}
        if self.config.resume and results_path.exists():
            try:
                with open(results_path) as f:
                    data = json.load(f)
                completed = {
                    (r["file_path"], r["page_number"]): ExtractionResult(**r)
                    for r in data.get("extraction_results", [])
                    if r.get("success")
                }
                logger.info(f"Loaded {len(completed)} completed extractions for resume")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")

        # Build page lookup
        page_lookup = {
            (p.file_path, p.page_number): p
            for p in project_index.pages
        }

        # Filter pages to process
        pages_to_process = []
        skipped_count = 0

        for page_type in PageType:
            if page_type not in self.PROCESSABLE_TYPES:
                skipped_count += len(routing_result.get_pages(page_type))
                continue

            if self.config.only_types and page_type.value not in self.config.only_types:
                skipped_count += len(routing_result.get_pages(page_type))
                continue

            # Skip structural if not enabled
            if page_type == PageType.STRUCTURAL_PLAN and not self.config.enable_structural:
                skipped_count += len(routing_result.get_pages(page_type))
                continue

            for routing in routing_result.get_pages(page_type):
                key = (routing.file_path, routing.page_number)
                if key in completed:
                    continue  # Skip already completed
                if key not in page_lookup:
                    continue  # Skip if not in index
                pages_to_process.append((routing, page_lookup[key]))

        logger.info(f"Processing {len(pages_to_process)} pages, skipping {skipped_count}")

        # Process pages
        results = list(completed.values())

        if pages_to_process:
            if len(pages_to_process) == 1 or self.config.max_workers == 1:
                # Single-threaded for debugging
                for routing, page in pages_to_process:
                    result = self._process_page(routing, page, output_dir)
                    results.append(result)
                    logger.info(f"Processed page {routing.page_number + 1}: {result.page_type} "
                               f"({'OK' if result.success else 'FAILED'})")
            else:
                # Parallel processing
                results.extend(self._process_parallel(pages_to_process, output_dir))

        # Build result
        successful = len([r for r in results if r.success])
        failed = len([r for r in results if not r.success])

        summary = self._build_summary(results, routing_result)

        runner_result = RunnerResult(
            project_id=project_index.project_id,
            total_pages=project_index.total_pages,
            processed_pages=successful,
            skipped_pages=skipped_count,
            failed_pages=failed,
            extraction_results=results,
            total_time_sec=time.time() - start_time,
            summary=summary,
        )

        # Save results
        runner_result.save(results_path)
        logger.info(f"Saved extraction results to: {results_path}")

        return runner_result

    def _process_parallel(
        self,
        pages: List[tuple],
        output_dir: Path
    ) -> List[ExtractionResult]:
        """Process pages in parallel."""
        results = []

        # Prepare tasks
        tasks = [(r, p, str(output_dir)) for r, p in pages]

        logger.info(f"Starting parallel processing with {self.config.max_workers} workers")

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(_process_page_worker, t, self.config): t
                for t in tasks
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                    if i % 5 == 0 or i == len(futures):
                        logger.info(f"Completed {i}/{len(futures)} pages")
                except Exception as e:
                    task = futures[future]
                    logger.error(f"Worker error for page {task[0].page_number}: {e}")
                    results.append(ExtractionResult(
                        file_path=task[0].file_path,
                        page_number=task[0].page_number,
                        page_type=task[0].page_type.value,
                        success=False,
                        processing_time_sec=0,
                        errors=[str(e)],
                    ))

        return results

    def _process_page(
        self,
        routing: PageRouting,
        page: PageIndex,
        output_dir: Path
    ) -> ExtractionResult:
        """Process a single page (main thread)."""
        return _process_page_impl(routing, page, output_dir, self.config)

    def _build_summary(
        self,
        results: List[ExtractionResult],
        routing: RoutingResult
    ) -> Dict[str, Any]:
        """Build extraction summary."""
        total_rooms = 0
        total_area = 0.0
        schedules_extracted = 0

        for result in results:
            if result.success:
                total_rooms += len(result.rooms)
                total_area += sum(r.get("area_sqm", 0) for r in result.rooms)
                if result.schedule_data:
                    schedules_extracted += 1

        return {
            "total_rooms_detected": total_rooms,
            "total_area_sqm": round(total_area, 1),
            "schedules_extracted": schedules_extracted,
            "floor_plans_processed": len([
                r for r in results
                if r.page_type == "floor_plan" and r.success
            ]),
            "extraction_success_rate": (
                len([r for r in results if r.success]) / len(results)
                if results else 0
            ),
        }


def _process_page_worker(task: tuple, config: RunnerConfig) -> ExtractionResult:
    """Worker function for parallel processing."""
    routing, page, output_dir = task
    return _process_page_impl(routing, page, Path(output_dir), config)


def _process_page_impl(
    routing: PageRouting,
    page: PageIndex,
    output_dir: Path,
    config: RunnerConfig
) -> ExtractionResult:
    """
    Process a single page based on its type.

    This is the core extraction logic.
    """
    start_time = time.time()

    result = ExtractionResult(
        file_path=routing.file_path,
        page_number=routing.page_number,
        page_type=routing.page_type.value,
        success=False,
        processing_time_sec=0,
    )

    try:
        if routing.page_type == PageType.FLOOR_PLAN:
            _extract_floor_plan(result, page, output_dir, config)
        elif routing.page_type == PageType.SCHEDULE_TABLE:
            _extract_schedule(result, page, config)
        elif routing.page_type == PageType.STRUCTURAL_PLAN:
            _extract_structural(result, page, config)
        else:
            result.warnings.append(f"No extractor for type: {routing.page_type.value}")

        result.success = len(result.errors) == 0

    except Exception as e:
        result.errors.append(f"Extraction failed: {e}")
        logger.error(f"Error processing {page.file_path} p{page.page_number}: {e}")
        logger.debug(traceback.format_exc())

    result.processing_time_sec = time.time() - start_time
    return result


def _extract_floor_plan(
    result: ExtractionResult,
    page: PageIndex,
    output_dir: Path,
    config: RunnerConfig
) -> None:
    """Extract rooms/walls/openings from floor plan."""
    import cv2

    # Import extraction modules
    from ..ingest import PlanIngester
    from ..preprocess import Preprocessor, PreprocessConfig
    from ..scale_dimensions import DimensionScaleInferrer
    from ..scale import ScaleInferrer, ScaleResult, ScaleMethod
    from ..walls import WallDetector
    from ..regions import RegionDetector
    from ..polygons import PolygonExtractor
    from ..labeling import RoomLabeler
    from ..area import AreaComputer

    # Load full-resolution image
    ingester = PlanIngester(dpi=config.dpi)
    plan = ingester.ingest(Path(page.file_path), page.page_number)

    if plan.image.size == 0:
        result.errors.append("Could not load page image")
        return

    # Preprocess
    preprocessor = Preprocessor(PreprocessConfig())
    preprocess = preprocessor.process(plan.image)

    # Scale inference
    scale = None
    if config.enable_dimension_scale:
        dim_inferrer = DimensionScaleInferrer(dpi=config.dpi)
        dim_result = dim_inferrer.infer_scale(plan.image, plan.vector_texts)
        if dim_result.confidence > 0.5:
            scale = ScaleResult(
                method=ScaleMethod.DIMENSION,
                pixels_per_mm=dim_result.pixels_per_mm,
                confidence=dim_result.confidence,
            )
            result.scale_info = {
                "method": "dimension",
                "pixels_per_mm": dim_result.pixels_per_mm,
                "confidence": dim_result.confidence,
            }

    if scale is None:
        scale_inferrer = ScaleInferrer(dpi=config.dpi)
        scale = scale_inferrer.infer_scale(
            plan.image, plan.vector_texts, config.default_scale
        )
        result.scale_info = {
            "method": scale.method.value,
            "pixels_per_mm": scale.pixels_per_mm,
            "confidence": scale.confidence,
            "scale_ratio": scale.scale_ratio,
        }

    # Detect walls
    wall_detector = WallDetector()
    wall_detector.gap_close_size = 15
    walls = wall_detector.detect(plan, preprocess.cleaned)

    if len(walls.wall_segments) == 0:
        result.warnings.append("No walls detected")
        return

    # Detect regions
    region_detector = RegionDetector()
    region_detector.min_room_area_ratio = 0.001
    region_detector.max_room_area_ratio = 0.5
    regions = region_detector.detect(walls.wall_mask_closed, walls.external_boundary)

    valid_regions = [r for r in regions.regions if r.is_valid and not r.is_hole]
    if not valid_regions:
        result.warnings.append("No valid room regions")
        return

    # Extract polygons
    polygon_extractor = PolygonExtractor()
    polygons = []
    for region in valid_regions:
        polygon = polygon_extractor.extract(region.mask, f"R{region.region_id:03d}")
        if polygon.is_valid:
            polygons.append(polygon)

    # Label rooms
    labeler = RoomLabeler()
    labeled_rooms = labeler.label_rooms(polygons, plan.vector_texts, plan.image)

    # Compute areas
    area_computer = AreaComputer(scale)
    rooms_with_area = area_computer.compute_all(labeled_rooms)

    # Collect wall thickness info
    wall_thicknesses = []
    if hasattr(walls, 'wall_segments'):
        for seg in walls.wall_segments:
            if hasattr(seg, 'thickness') and seg.thickness > 0:
                # Convert from pixels to mm using scale
                thickness_mm = seg.thickness / scale.pixels_per_mm if scale.pixels_per_mm > 0 else 0
                wall_thicknesses.append(thickness_mm)

    result.wall_info = {
        "segment_count": len(walls.wall_segments) if hasattr(walls, 'wall_segments') else 0,
        "thicknesses": wall_thicknesses,
    }

    # Convert to dict for result with provenance
    scale_method = result.scale_info.get("method", "unknown") if result.scale_info else "unknown"
    scale_confidence = result.scale_info.get("confidence", 0) if result.scale_info else 0

    for room in rooms_with_area:
        label = room.label if room.label else "Room"
        bbox = None
        if room.polygon and hasattr(room.polygon, 'bounds'):
            b = room.polygon.bounds
            bbox = [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]

        room_dict = {
            "room_id": room.room_id,
            "label": label,
            "area_sqm": round(room.area.area_sqm, 2),
            "area_sqft": round(room.area.area_sqft, 2),
            "centroid": list(room.polygon.centroid) if room.polygon and hasattr(room.polygon, 'centroid') else [0, 0],
            # Provenance fields
            "detection_method": "raster",
            "confidence": room.confidence if hasattr(room, 'confidence') else 0.8,
            "is_measured": scale_method not in ["assumed", "default"],
            "scale_method": scale_method,
            "scale_confidence": scale_confidence,
        }

        result.rooms.append(room_dict)

        # Add provenance record
        result.provenance.append({
            "object_id": room.room_id,
            "object_type": "room",
            "source_page": page.page_number,
            "source_file": str(page.file_path),
            "detection_method": "raster",
            "confidence": room.confidence if hasattr(room, 'confidence') else 0.8,
            "is_measured": scale_method not in ["assumed", "default"],
            "method_details": {
                "area_sqm": round(room.area.area_sqm, 2),
                "scale_method": scale_method,
                "scale_confidence": scale_confidence,
            },
        })


def _extract_schedule(
    result: ExtractionResult,
    page: PageIndex,
    config: RunnerConfig
) -> None:
    """Extract table data from schedule page."""
    import cv2
    import re

    # Try to parse table structure from text
    text = page.extracted_text
    lines = text.split('\n') if text else []

    # Simple table detection: look for repeated patterns
    entries = []

    # Door/window schedule patterns
    door_pattern = r'([DW]\d+)\s+(\d+)\s*[xX×]\s*(\d+)'
    for line in lines:
        match = re.search(door_pattern, line)
        if match:
            entries.append({
                "tag": match.group(1),
                "width": int(match.group(2)),
                "height": int(match.group(3)),
                "unit": "mm",
            })

    # BBS patterns
    bbs_pattern = r'([A-Z]\d+)\s+(\d+)\s+(\d+)\s+(\d+)'  # tag, dia, nos, length
    for line in lines:
        match = re.search(bbs_pattern, line)
        if match and match.group(2).isdigit():
            dia = int(match.group(2))
            if 6 <= dia <= 32:  # Valid rebar diameters
                entries.append({
                    "tag": match.group(1),
                    "diameter": dia,
                    "quantity": int(match.group(3)),
                    "length": int(match.group(4)),
                })

    result.schedule_data = {
        "entries": entries,
        "raw_lines": len(lines),
        "parsed_entries": len(entries),
    }

    # Add provenance for schedule entries
    for entry in entries:
        result.provenance.append({
            "object_id": f"sched_{entry.get('tag', 'unknown')}",
            "object_type": "schedule_entry",
            "source_page": page.page_number,
            "source_file": str(page.file_path),
            "detection_method": "ocr",
            "confidence": 0.85,  # OCR-based parsing
            "is_measured": True,  # Schedule values are from drawings
            "method_details": entry,
        })

    if not entries:
        result.warnings.append("No schedule entries parsed")


def _extract_structural(
    result: ExtractionResult,
    page: PageIndex,
    config: RunnerConfig
) -> None:
    """Extract structural elements (simplified)."""
    # Placeholder - structural extraction would be more complex
    result.warnings.append("Structural extraction not fully implemented")
    result.schedule_data = {
        "structural_type": "detected",
        "elements": [],
    }


def run_extraction(
    project_index: ProjectIndex,
    routing_result: RoutingResult,
    output_dir: Path,
    config: Optional[RunnerConfig] = None,
) -> RunnerResult:
    """
    Convenience function to run extraction.

    Args:
        project_index: Project index
        routing_result: Routing result
        output_dir: Output directory
        config: Runner configuration

    Returns:
        RunnerResult
    """
    runner = ProjectRunner(config)
    return runner.run(project_index, routing_result, output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 2:
        index_path = Path(sys.argv[1])
        manifest_path = Path(sys.argv[2])
        output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./out")

        # Load data
        project_index = ProjectIndex.load(index_path)

        with open(manifest_path) as f:
            manifest_data = json.load(f)

        # Reconstruct routing result (simplified)
        from .router import route_project
        routing = route_project(project_index)

        # Run extraction
        config = RunnerConfig(max_workers=2)
        result = run_extraction(project_index, routing, output_dir, config)

        print(f"\nExtraction Result: {result.project_id}")
        print(f"Processed: {result.processed_pages}")
        print(f"Skipped: {result.skipped_pages}")
        print(f"Failed: {result.failed_pages}")
        print(f"Time: {result.total_time_sec:.1f}s")
        print(f"Summary: {result.summary}")
    else:
        print("Usage: python runner.py <index.json> <manifest.json> [output_dir]")
