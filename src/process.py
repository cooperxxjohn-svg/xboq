"""
Main Processing Script
Process floor plans to extract rooms, openings, and finish quantities.

Usage:
    python -m src.process --input ./data/plans --output ./out
    python -m src.process --input ./data/plans --output ./out --profile typical
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
import argparse
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import existing pipeline components
try:
    from .pipeline import FloorplanPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

from .openings.pipeline import OpeningsPipeline, OpeningsResult
from .openings.detect_doors import DoorDetector
from .openings.detect_windows import WindowDetector
from .openings.assign import RoomAssigner
from .openings.export import OpeningsExporter

from .finishes.calculator import FinishCalculator
from .finishes.export import FinishExporter

from .boq.schema import BOQItem, BOQValidator, load_profile, merge_boq_items
from .rates.mapper import CPWDMapper, map_boq_to_cpwd

logger = logging.getLogger(__name__)


class FullTakeoffPipeline:
    """
    Full takeoff pipeline combining:
    1. Room detection and area computation
    2. Door/window detection
    3. Finish quantity calculations
    4. BOQ generation with CPWD mapping
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        ceiling_height_mm: int = 3000,
        slab_thickness_mm: int = 125,
        profile: str = "typical",
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.ceiling_height_mm = ceiling_height_mm
        self.slab_thickness_mm = slab_thickness_mm
        self.profile_name = profile

        # Load profile
        self.profile = load_profile(profile)
        logger.info(f"Using profile: {profile} - {self.profile.get('description', '')}")

        # Apply profile adjustments
        self._apply_profile()

        # Initialize components
        self.openings_pipeline = OpeningsPipeline(config_path, output_dir)
        self.finish_calculator = FinishCalculator(config_path, ceiling_height_mm)
        self.cpwd_mapper = CPWDMapper()

        # Track assumptions
        self.assumptions_used: List[str] = []

    def _apply_profile(self) -> None:
        """Apply profile settings."""
        profile = self.profile

        # Steel factor multiplier
        self.steel_multiplier = profile.get("steel_factor_multiplier", 1.0)

        # Wastage multiplier
        self.wastage_multiplier = profile.get("wastage_factor_multiplier", 1.0)

        # Confidence threshold
        self.confidence_threshold = profile.get("confidence_threshold", 0.5)

        self.assumptions_used.append(f"Profile: {self.profile_name}")

    def process_plan(
        self,
        image_path: Path,
        rooms_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single floor plan.

        Args:
            image_path: Path to plan image
            rooms_data: Pre-computed room data (optional)

        Returns:
            Complete takeoff results
        """
        plan_id = image_path.stem
        logger.info(f"Processing plan: {plan_id}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Generate wall mask (simple approach if no external wall detector)
        wall_mask = self._detect_walls(gray)

        # Close gaps for door detection
        wall_mask_closed = self._close_wall_gaps(wall_mask)

        # Run openings detection
        openings_result = self.openings_pipeline.run(
            image=image,
            wall_mask=wall_mask,
            wall_mask_closed=wall_mask_closed,
            texts=None,  # Would come from OCR
            rooms=rooms_data,
            plan_id=plan_id,
        )

        # Convert openings to dicts for finish calculator
        openings_list = self._openings_to_dicts(openings_result)

        # Calculate finishes if room data available
        room_finishes = []
        if rooms_data:
            # Build room-openings map
            room_openings_map = self._build_room_openings_map(openings_list)

            room_finishes = self.finish_calculator.calculate_all_rooms(
                rooms_data,
                openings=openings_list,
                room_openings_map=room_openings_map,
            )

        # Calculate wall lengths
        wall_lengths = {}
        if rooms_data:
            wall_lengths = self.finish_calculator.calculate_wall_lengths(rooms_data)

        # Collect all BOQ items and map to CPWD
        all_boq_items = self._collect_boq_items(room_finishes, openings_list, rooms_data)
        cpwd_result = self.cpwd_mapper.map_items(all_boq_items)

        # Export results
        output_paths = self._export_all(
            plan_id=plan_id,
            image=image,
            openings_result=openings_result,
            room_finishes=room_finishes,
            wall_lengths=wall_lengths,
            openings_list=openings_list,
            cpwd_result=cpwd_result,
        )

        # Compile results
        results = {
            "plan_id": plan_id,
            "profile": self.profile_name,
            "generated_at": datetime.now().isoformat(),
            "openings": {
                "doors": len(openings_result.doors),
                "windows": openings_result.total_windows,
                "ventilators": openings_result.total_ventilators,
            },
            "rooms": len(rooms_data) if rooms_data else 0,
            "finishes": {
                "total_floor_sqm": sum(r.floor_area_sqm for r in room_finishes),
                "total_wall_sqm": sum(r.wall_area_sqm for r in room_finishes),
                "total_ceiling_sqm": sum(r.ceiling_area_sqm for r in room_finishes),
                "total_skirting_m": sum(r.skirting_length_m for r in room_finishes),
            },
            "boq": {
                "total_items": len(all_boq_items),
                "cpwd_mapped": cpwd_result.mapped_count,
                "cpwd_unmapped": cpwd_result.unmapped_count,
                "cpwd_coverage_percent": cpwd_result.coverage_percent,
            },
            "assumptions": self.assumptions_used + openings_result.assumptions_used,
            "warnings": openings_result.warnings,
            "output_paths": {str(k): str(v) for k, v in output_paths.items()},
        }

        return results

    def _detect_walls(self, gray: np.ndarray) -> np.ndarray:
        """Simple wall detection using thresholding."""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35, 10
        )

        # Find thick lines (walls)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        wall_mask = cv2.bitwise_or(horizontal, vertical)

        # Dilate to connect
        kernel = np.ones((3, 3), np.uint8)
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)

        return wall_mask

    def _close_wall_gaps(self, wall_mask: np.ndarray) -> np.ndarray:
        """Close gaps in walls (doors, windows) for room detection."""
        # Directional closing
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))

        closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_h)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)

        return closed

    def _openings_to_dicts(self, result: OpeningsResult) -> List[Dict]:
        """Convert openings to dicts."""
        openings = []

        for door in result.doors:
            openings.append({
                "id": door.id,
                "type": door.type,
                "tag": door.tag,
                "width_m": door.width_m,
                "height_m": door.height_m,
                "room_left_id": door.room_left_id,
                "room_right_id": door.room_right_id,
                "confidence": door.confidence,
            })

        for window in result.windows:
            openings.append({
                "id": window.id,
                "type": window.type,
                "tag": window.tag,
                "width_m": window.width_m,
                "height_m": window.height_m,
                "room_left_id": window.room_left_id,
                "room_right_id": window.room_right_id,
                "confidence": window.confidence,
            })

        return openings

    def _build_room_openings_map(
        self,
        openings: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Build mapping of room_id to openings."""
        room_map = {}

        for opening in openings:
            left_id = opening.get("room_left_id")
            if left_id:
                if left_id not in room_map:
                    room_map[left_id] = []
                room_map[left_id].append(opening)

            right_id = opening.get("room_right_id")
            if right_id and "door" in opening.get("type", "").lower():
                if right_id not in room_map:
                    room_map[right_id] = []
                room_map[right_id].append(opening)

        return room_map

    def _collect_boq_items(
        self,
        room_finishes: List,
        openings_list: List[Dict],
        rooms_data: Optional[List[Dict]],
    ) -> List[Dict]:
        """Collect all BOQ items from various sources."""
        items = []

        # Collect from room finishes
        for room_finish in room_finishes:
            for item in getattr(room_finish, "boq_items", []):
                items.append({
                    "item_code": getattr(item, "item_code", ""),
                    "description": getattr(item, "description", ""),
                    "qty": getattr(item, "qty", 0),
                    "unit": getattr(item, "unit", "nos"),
                    "derived_from": getattr(item, "derived_from", "room_finish_mapping"),
                    "confidence": getattr(item, "confidence", 0.5),
                    "category": "Finishes",
                })

        # Collect from openings (placeholder - real items from openings_boq)
        # This would integrate with the full BOQ engine

        return items

    def _export_all(
        self,
        plan_id: str,
        image: np.ndarray,
        openings_result: OpeningsResult,
        room_finishes: List,
        wall_lengths: Dict,
        openings_list: List[Dict],
        cpwd_result: Any,
    ) -> Dict[str, Path]:
        """Export all results."""
        if not self.output_dir:
            return {}

        output_paths = {}

        # Create plan-specific output directory
        plan_output = self.output_dir / plan_id
        plan_output.mkdir(parents=True, exist_ok=True)
        boq_output = plan_output / "boq"
        boq_output.mkdir(exist_ok=True)

        # Export openings
        openings_exporter = OpeningsExporter(plan_output)
        openings_paths = openings_exporter.export_all(
            plan_id=plan_id,
            doors=openings_result.doors,
            windows=openings_result.windows,
            assignments=openings_result.assignments,
            image=image,
            assumptions_used=openings_result.assumptions_used,
        )
        output_paths.update(openings_paths)

        # Export finishes
        if room_finishes:
            finish_exporter = FinishExporter(plan_output)
            finish_paths = finish_exporter.export_all(
                room_finishes=room_finishes,
                openings=openings_list,
                wall_lengths=wall_lengths,
                assumptions=self.finish_calculator.export_assumptions(),
                plan_id=plan_id,
            )
            output_paths.update(finish_paths)

        # Export CPWD-mapped BOQ
        if cpwd_result and cpwd_result.mapped_items:
            cpwd_path = boq_output / "boq_with_cpwd_map.csv"
            self.cpwd_mapper.export_csv(cpwd_result.mapped_items, cpwd_path)
            output_paths["cpwd_boq"] = cpwd_path

        # Export processing config used
        config_path = plan_output / "processing_config.json"
        config_data = {
            "profile": self.profile_name,
            "ceiling_height_mm": self.ceiling_height_mm,
            "slab_thickness_mm": self.slab_thickness_mm,
            "steel_multiplier": self.steel_multiplier,
            "wastage_multiplier": self.wastage_multiplier,
            "confidence_threshold": self.confidence_threshold,
            "generated_at": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        output_paths["config"] = config_path

        return output_paths


def process_directory(
    input_dir: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    profile: str = "typical",
    ceiling_height_mm: int = 3000,
    slab_thickness_mm: int = 125,
) -> List[Dict]:
    """
    Process all plans in a directory.

    Args:
        input_dir: Directory containing plan images
        output_dir: Output directory
        config_path: Path to config YAML
        profile: Profile name (conservative, typical, premium)
        ceiling_height_mm: Ceiling height
        slab_thickness_mm: Slab thickness

    Returns:
        List of results for each plan
    """
    # Find plan files
    extensions = [".png", ".jpg", ".jpeg", ".pdf"]
    plan_files = []
    for ext in extensions:
        plan_files.extend(input_dir.glob(f"*{ext}"))
        plan_files.extend(input_dir.glob(f"*{ext.upper()}"))

    logger.info(f"Found {len(plan_files)} plan files")

    # Initialize pipeline
    pipeline = FullTakeoffPipeline(
        config_path=config_path,
        output_dir=output_dir,
        ceiling_height_mm=ceiling_height_mm,
        slab_thickness_mm=slab_thickness_mm,
        profile=profile,
    )

    results = []

    for plan_file in plan_files:
        try:
            # Check for corresponding rooms JSON
            rooms_json = plan_file.with_suffix(".json")
            rooms_data = None

            if rooms_json.exists():
                with open(rooms_json, "r") as f:
                    data = json.load(f)
                    rooms_data = data.get("rooms", [])

            # Also check in out directory
            if not rooms_data:
                out_json = output_dir / plan_file.stem / "rooms.json"
                if out_json.exists():
                    with open(out_json, "r") as f:
                        data = json.load(f)
                        rooms_data = data.get("rooms", [])

            result = pipeline.process_plan(plan_file, rooms_data)
            results.append(result)

            logger.info(f"Processed {plan_file.name}: {result['openings']['doors']} doors, "
                       f"{result['openings']['windows']} windows")

        except Exception as e:
            logger.error(f"Error processing {plan_file}: {e}")
            results.append({
                "plan_id": plan_file.stem,
                "error": str(e),
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process floor plans for openings and finish takeoff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  conservative - Lower estimates, fewer assumptions
  typical      - Standard Indian construction practices (default)
  premium      - Higher quality finishes, more generous estimates

Examples:
  python -m src.process --input ./data/plans --output ./out
  python -m src.process --input ./data/plans --output ./out --profile premium
  python -m src.process --input ./data/plans --output ./out --ceiling-height 3300
"""
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing plans",
    )
    parser.add_argument(
        "--output", "-o",
        default="./out",
        help="Output directory",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--profile", "-p",
        choices=["conservative", "typical", "premium"],
        default="typical",
        help="Estimation profile (default: typical)",
    )
    parser.add_argument(
        "--ceiling-height",
        type=int,
        default=3000,
        help="Ceiling height in mm (default: 3000)",
    )
    parser.add_argument(
        "--slab-thickness",
        type=int,
        default=125,
        help="Slab thickness in mm (default: 125)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config) if args.config else None

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    results = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        profile=args.profile,
        ceiling_height_mm=args.ceiling_height,
        slab_thickness_mm=args.slab_thickness,
    )

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Profile: {args.profile}")
    print(f"Ceiling height: {args.ceiling_height}mm")
    print(f"Slab thickness: {args.slab_thickness}mm")
    print()

    total_doors = sum(r.get("openings", {}).get("doors", 0) for r in results if "error" not in r)
    total_windows = sum(r.get("openings", {}).get("windows", 0) for r in results if "error" not in r)
    total_items = sum(r.get("boq", {}).get("total_items", 0) for r in results if "error" not in r)
    avg_coverage = sum(r.get("boq", {}).get("cpwd_coverage_percent", 0) for r in results if "error" not in r)
    if results:
        avg_coverage /= len([r for r in results if "error" not in r])

    errors = sum(1 for r in results if "error" in r)

    print(f"Plans processed: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Total doors detected: {total_doors}")
    print(f"Total windows detected: {total_windows}")
    print(f"Total BOQ items: {total_items}")
    print(f"CPWD mapping coverage: {avg_coverage:.1f}%")
    print(f"Output directory: {output_dir}")

    # Save summary
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "profile": args.profile,
            "plans_processed": len(results),
            "errors": errors,
            "total_doors": total_doors,
            "total_windows": total_windows,
            "total_boq_items": total_items,
            "cpwd_coverage_percent": avg_coverage,
            "results": results,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
