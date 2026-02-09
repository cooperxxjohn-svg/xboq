"""
Openings Detection Pipeline
Orchestrates door/window detection, room assignment, and export.

Usage:
    python -m src.openings.extract --plan <file>
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
import logging

from .detect_doors import DoorDetector, DetectedDoor
from .detect_windows import WindowDetector, DetectedWindow
from .tags import TagExtractor, OpeningTag
from .sizes import SizeInferrer
from .assign import RoomAssigner, OpeningAssignment, create_room_openings_summary
from .export import OpeningsExporter

logger = logging.getLogger(__name__)


@dataclass
class OpeningsResult:
    """Result of openings detection pipeline."""
    plan_id: str
    doors: List[DetectedDoor] = field(default_factory=list)
    windows: List[DetectedWindow] = field(default_factory=list)
    tags: List[OpeningTag] = field(default_factory=list)
    assignments: List[OpeningAssignment] = field(default_factory=list)
    room_summary: Dict[str, Dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    assumptions_used: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_doors(self) -> int:
        return len(self.doors)

    @property
    def total_windows(self) -> int:
        return len([w for w in self.windows if w.type != "ventilator"])

    @property
    def total_ventilators(self) -> int:
        return len([w for w in self.windows if w.type == "ventilator"])


class OpeningsPipeline:
    """
    Main pipeline for detecting doors and windows in floor plans.

    Steps:
    1. Detect doors using multi-signal approach
    2. Detect windows using multi-signal approach
    3. Extract and associate tags
    4. Infer sizes from schedules/defaults
    5. Assign openings to rooms
    6. Generate warnings and QC checks
    7. Export results
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config_path: Path to assumptions.yaml
            output_dir: Output directory for exports
        """
        self.config_path = config_path
        self.output_dir = output_dir

        # Initialize components
        self.door_detector = DoorDetector()
        self.window_detector = WindowDetector()
        self.tag_extractor = TagExtractor(config_path)
        self.size_inferrer = SizeInferrer(config_path)
        self.room_assigner = RoomAssigner()

    def run(
        self,
        image: np.ndarray,
        wall_mask: np.ndarray,
        wall_mask_closed: Optional[np.ndarray] = None,
        texts: Optional[List[Dict]] = None,
        rooms: Optional[List[Dict]] = None,
        external_boundary: Optional[np.ndarray] = None,
        scale_px_per_mm: Optional[float] = None,
        plan_id: str = "plan",
    ) -> OpeningsResult:
        """
        Run full openings detection pipeline.

        Args:
            image: Input image (grayscale or color)
            wall_mask: Binary wall mask
            wall_mask_closed: Wall mask with gaps closed
            texts: OCR text boxes
            rooms: Room data for assignment
            external_boundary: External area mask
            scale_px_per_mm: Scale factor
            plan_id: Plan identifier

        Returns:
            OpeningsResult with all detections
        """
        logger.info(f"Starting openings pipeline for {plan_id}")

        result = OpeningsResult(plan_id=plan_id)
        assumptions_used = {}

        # Step 1: Detect doors
        logger.info("Step 1: Detecting doors...")
        doors = self.door_detector.detect(
            image=image,
            wall_mask=wall_mask,
            wall_mask_closed=wall_mask_closed,
            texts=texts,
            scale_px_per_mm=scale_px_per_mm,
        )
        result.doors = doors
        logger.info(f"Detected {len(doors)} doors")

        # Step 2: Detect windows
        logger.info("Step 2: Detecting windows...")
        windows = self.window_detector.detect(
            image=image,
            wall_mask=wall_mask,
            wall_mask_closed=wall_mask_closed,
            texts=texts,
            scale_px_per_mm=scale_px_per_mm,
        )
        result.windows = windows
        logger.info(f"Detected {len(windows)} windows/ventilators")

        # Step 3: Extract tags
        if texts:
            logger.info("Step 3: Extracting tags...")
            tags = self.tag_extractor.extract_from_texts(texts)
            result.tags = tags
            logger.info(f"Extracted {len(tags)} opening tags")

        # Step 4: Infer sizes
        logger.info("Step 4: Inferring sizes...")
        for door in doors:
            if not door.width_m:
                size = self.size_inferrer.infer_size(
                    tag=door.tag,
                    opening_type="door",
                    category=door.type,
                    width_px=door.width_px,
                    scale_px_per_mm=scale_px_per_mm,
                )
                door.width_m = size.width_mm / 1000
                door.height_m = size.height_mm / 1000

                if size.source == "default":
                    assumptions_used[f"door_{door.id}_size"] = {
                        "width_mm": size.width_mm,
                        "height_mm": size.height_mm,
                        "source": size.source,
                    }

        for window in windows:
            if not window.width_m:
                size = self.size_inferrer.infer_size(
                    tag=window.tag,
                    opening_type=window.type,
                    category=window.type,
                    width_px=window.width_px,
                    scale_px_per_mm=scale_px_per_mm,
                )
                window.width_m = size.width_mm / 1000
                window.height_m = size.height_mm / 1000

                if size.source == "default":
                    assumptions_used[f"window_{window.id}_size"] = {
                        "width_mm": size.width_mm,
                        "height_mm": size.height_mm,
                        "source": size.source,
                    }

        # Step 5: Assign to rooms
        if rooms:
            logger.info("Step 5: Assigning to rooms...")
            self.room_assigner.load_rooms(rooms)
            if external_boundary is not None:
                self.room_assigner.external_boundary = external_boundary

            all_openings = list(doors) + list(windows)
            assignments = self.room_assigner.assign_all(all_openings)
            result.assignments = assignments

            # Create room summary
            result.room_summary = create_room_openings_summary(assignments)
            logger.info(f"Assigned openings to {len(result.room_summary)} rooms")

        # Step 6: Generate warnings
        logger.info("Step 6: Generating warnings...")
        result.warnings = self._generate_warnings(result)

        result.assumptions_used = assumptions_used

        return result

    def _generate_warnings(self, result: OpeningsResult) -> List[str]:
        """Generate QC warnings."""
        warnings = []

        # Check for no scale
        doors_with_default = [d for d in result.doors if not d.width_m or d.width_m == 0.9]
        if len(doors_with_default) > len(result.doors) / 2:
            warnings.append("no_scale: Many doors using default sizes - scale may be missing")

        # Check for unassigned
        unassigned = [a for a in result.assignments if not a.room_left_id]
        if unassigned:
            warnings.append(f"opening_unassigned: {len(unassigned)} openings not assigned to rooms")

        # Check for ambiguous tags
        tags_seen = {}
        for door in result.doors:
            if door.tag:
                tags_seen[door.tag] = tags_seen.get(door.tag, 0) + 1
        for window in result.windows:
            if window.tag:
                tags_seen[window.tag] = tags_seen.get(window.tag, 0) + 1

        ambiguous = [t for t, c in tags_seen.items() if c > 1]
        if ambiguous:
            warnings.append(f"ambiguous_tag: Tags used multiple times: {', '.join(ambiguous)}")

        # Check for unusual counts
        if result.total_doors > 20:
            warnings.append(f"too_many_openings: {result.total_doors} doors detected")
        if result.total_windows > 30:
            warnings.append(f"too_many_openings: {result.total_windows} windows detected")

        # Check for overlapping openings
        overlaps = self._check_overlaps(result.doors, result.windows)
        if overlaps:
            warnings.append(f"overlaps_detected: {len(overlaps)} overlapping openings found")

        return warnings

    def _check_overlaps(
        self,
        doors: List[DetectedDoor],
        windows: List[DetectedWindow],
    ) -> List[Tuple[str, str]]:
        """Check for overlapping bounding boxes."""
        overlaps = []
        all_openings = [(d.id, d.bbox) for d in doors] + [(w.id, w.bbox) for w in windows]

        for i, (id1, bbox1) in enumerate(all_openings):
            for j, (id2, bbox2) in enumerate(all_openings):
                if j <= i:
                    continue

                # Check IoU
                x1_1, y1_1, x2_1, y2_1 = bbox1
                x1_2, y1_2, x2_2, y2_2 = bbox2

                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)

                if x2_i > x1_i and y2_i > y1_i:
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

                    if intersection > 0.3 * min(area1, area2):
                        overlaps.append((id1, id2))

        return overlaps

    def export_results(
        self,
        result: OpeningsResult,
        image: np.ndarray,
    ) -> Dict[str, Path]:
        """Export all results."""
        if not self.output_dir:
            logger.warning("No output directory specified")
            return {}

        exporter = OpeningsExporter(self.output_dir)

        outputs = exporter.export_all(
            plan_id=result.plan_id,
            doors=result.doors,
            windows=result.windows,
            assignments=result.assignments,
            image=image,
            assumptions_used=result.assumptions_used,
        )

        return outputs


def run_openings_pipeline(
    image_path: Path,
    wall_mask_path: Optional[Path] = None,
    rooms_json_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> OpeningsResult:
    """
    Convenience function to run openings pipeline from files.

    Args:
        image_path: Path to plan image
        wall_mask_path: Path to wall mask (optional)
        rooms_json_path: Path to rooms JSON (optional)
        output_dir: Output directory
        config_path: Path to config YAML

    Returns:
        OpeningsResult
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Generate or load wall mask
    if wall_mask_path and wall_mask_path.exists():
        wall_mask = cv2.imread(str(wall_mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        # Simple wall detection fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, wall_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Close gaps in wall mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    wall_mask_closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    wall_mask_closed = cv2.morphologyEx(wall_mask_closed, cv2.MORPH_CLOSE, kernel)

    # Load rooms
    rooms = None
    if rooms_json_path and rooms_json_path.exists():
        with open(rooms_json_path, "r") as f:
            data = json.load(f)
            rooms = data.get("rooms", [])

    # Set up output
    if not output_dir:
        output_dir = image_path.parent / "out" / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    pipeline = OpeningsPipeline(config_path=config_path, output_dir=output_dir)

    result = pipeline.run(
        image=image,
        wall_mask=wall_mask,
        wall_mask_closed=wall_mask_closed,
        texts=None,  # Would come from OCR
        rooms=rooms,
        plan_id=image_path.stem,
    )

    # Export
    pipeline.export_results(result, image)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect doors and windows in floor plans")
    parser.add_argument("--plan", required=True, help="Path to plan image")
    parser.add_argument("--wall-mask", help="Path to wall mask")
    parser.add_argument("--rooms", help="Path to rooms JSON")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--config", help="Path to config YAML")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = run_openings_pipeline(
        image_path=Path(args.plan),
        wall_mask_path=Path(args.wall_mask) if args.wall_mask else None,
        rooms_json_path=Path(args.rooms) if args.rooms else None,
        output_dir=Path(args.output) if args.output else None,
        config_path=Path(args.config) if args.config else None,
    )

    print(f"\nResults:")
    print(f"  Doors: {result.total_doors}")
    print(f"  Windows: {result.total_windows}")
    print(f"  Ventilators: {result.total_ventilators}")
    print(f"  Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"    - {warning}")
