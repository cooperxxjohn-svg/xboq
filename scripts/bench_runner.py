#!/usr/bin/env python3
"""
Standalone benchmark runner script.

This script runs the floor plan pipeline and outputs results as JSON.
Used by the benchmark evaluator to avoid import conflicts.

Usage:
    python scripts/bench_runner.py <image_path> <output_dir> [params_json]
"""

import os
import sys
import json
from pathlib import Path

# Change to repo root and set up paths
repo_root = Path(__file__).parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

# Now import normally - this should work since we're at repo root
from src.ingest import PlanIngester, IngestedPlan
from src.preprocess import Preprocessor, PreprocessConfig, PreprocessResult
from src.scale import ScaleInferrer, ScaleResult
from src.walls import WallDetector, WallDetectionResult
from src.regions import RegionDetector, RegionDetectionResult
from src.polygons import PolygonExtractor, RoomPolygon
from src.labeling import RoomLabeler, LabeledRoom
from src.area import AreaComputer, RoomWithArea
# Note: src.qc is a package, QualityChecker is in src/qc.py file
# We don't need it for basic evaluation


def run_pipeline(image_path: str, output_dir: str, params: dict = None) -> dict:
    """Run pipeline and return results as dict."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    params = params or {}

    file_path = Path(image_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output = {
        "success": False,
        "rooms": [],
        "scale": None,
        "errors": [],
    }

    try:
        # Step 1: Ingest
        ingester = PlanIngester(dpi=300)
        plan = ingester.ingest(file_path, 0)

        if plan.image.size == 0:
            output["errors"].append("Failed to load plan image")
            return output

        # Step 2: Preprocess
        preprocess_config = PreprocessConfig(
            enable_deskew=params.get("enable_deskew", True),
            enable_denoise=params.get("enable_denoise", True),
            adaptive_block_size=params.get("adaptive_block_size", 35),
        )
        preprocessor = Preprocessor(preprocess_config)
        preprocess = preprocessor.process(plan.image)

        # Step 3: Infer scale
        scale_inferrer = ScaleInferrer(dpi=300)
        scale = scale_inferrer.infer_scale(plan.image, plan.vector_texts, 100)

        # Step 4: Detect walls
        wall_detector = WallDetector()
        wall_detector.gap_close_size = params.get("gap_close_size", 15)
        walls = wall_detector.detect(plan, preprocess.cleaned)

        # Step 5: Detect regions
        region_detector = RegionDetector()
        region_detector.min_room_area_ratio = params.get("min_room_area_ratio", 0.001)
        region_detector.max_room_area_ratio = params.get("max_room_area_ratio", 0.5)
        regions = region_detector.detect(walls.wall_mask_closed, walls.external_boundary)

        valid_regions = [r for r in regions.regions if r.is_valid and not r.is_hole]

        if not valid_regions:
            output["errors"].append("No valid room regions detected")
            return output

        # Step 6: Extract polygons
        polygon_extractor = PolygonExtractor()
        polygons = []
        for region in valid_regions:
            polygon = polygon_extractor.extract(region.mask, f"R{region.region_id:03d}")
            if polygon.is_valid:
                polygons.append(polygon)

        # Step 7: Label rooms
        labeler = RoomLabeler()
        labeled_rooms = labeler.label_rooms(polygons, plan.vector_texts, plan.image)

        # Step 8: Compute areas
        area_computer = AreaComputer(scale)
        rooms_with_area = area_computer.compute_all(labeled_rooms)

        # Build output
        for room in rooms_with_area:
            # Handle label - it might be a LabeledRoom or just a string
            label = "Room"
            if hasattr(room, "label"):
                if hasattr(room.label, "canonical"):
                    label = room.label.canonical
                else:
                    label = str(room.label)

            # Handle area
            area = 0
            if hasattr(room, "area"):
                if hasattr(room.area, "area_sqm"):
                    area = room.area.area_sqm
                else:
                    area = float(room.area) if room.area else 0

            room_data = {
                "id": room.polygon.room_id,
                "polygon": room.polygon.points,
                "label": label,
                "area": area,
                "bbox": room.polygon.bbox,
            }
            output["rooms"].append(room_data)

        output["scale"] = {
            "px_per_mm": scale.pixels_per_mm,
            "confidence": scale.confidence,
            "method": scale.method.value if hasattr(scale.method, "value") else str(scale.method),
        }

        output["success"] = True

    except Exception as e:
        output["errors"].append(str(e))

    return output


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/bench_runner.py <image_path> <output_dir> [params_json]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}

    try:
        result = run_pipeline(image_path, output_dir, params)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "rooms": [], "scale": None, "errors": [str(e)]}))
        sys.exit(1)
