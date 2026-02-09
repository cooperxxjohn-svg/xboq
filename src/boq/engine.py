"""
BOQ Engine - Main Orchestrator
Integrates all BOQ modules for complete quantity takeoff.

Pipeline:
1. Wall detection → Wall BOQ (brick, plaster)
2. Room detection → Finish BOQ (floor, wall, ceiling)
3. Opening detection → Openings BOQ (doors, windows)
4. Slab calculation → Slab BOQ (concrete, formwork)
5. Steel estimation → Steel BOQ (reinforcement)
6. Confidence scoring → Heatmap
7. Export → CSV, JSON, Report
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import logging

from .wall_boq import WallBOQCalculator, WallBOQResult
from .finish_boq import FinishBOQCalculator, FinishBOQResult
from .slab_boq import SlabBOQCalculator, SlabBOQResult
from .steel_boq import SteelBOQCalculator, SteelBOQResult
from .openings_boq import OpeningsBOQCalculator, OpeningsBOQResult
from .confidence import ConfidenceCalculator, ConfidenceHeatmap
from .export import BOQExporter

logger = logging.getLogger(__name__)


@dataclass
class BOQResult:
    """Complete BOQ result from engine."""
    project_id: str
    wall_result: Optional[WallBOQResult] = None
    finish_result: Optional[FinishBOQResult] = None
    slab_result: Optional[SlabBOQResult] = None
    steel_result: Optional[SteelBOQResult] = None
    openings_result: Optional[OpeningsBOQResult] = None
    confidence_heatmap: Optional[ConfidenceHeatmap] = None
    all_boq_items: List[Any] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    output_paths: Dict[str, Path] = field(default_factory=dict)


class BOQEngine:
    """
    Main BOQ calculation engine.

    Orchestrates all BOQ modules and generates complete takeoff.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
        ceiling_height_mm: int = 3000,
        slab_thickness_mm: int = 125,
        concrete_grade: str = "M25",
        steel_grade: str = "Fe500",
    ):
        self.output_dir = output_dir
        self.config_path = config_path
        self.ceiling_height_mm = ceiling_height_mm
        self.slab_thickness_mm = slab_thickness_mm
        self.concrete_grade = concrete_grade
        self.steel_grade = steel_grade

        # Initialize calculators
        self.wall_calc = WallBOQCalculator(ceiling_height_mm=ceiling_height_mm)
        self.finish_calc = FinishBOQCalculator(
            templates_path=config_path.parent / "finish_templates.yaml" if config_path else None,
            ceiling_height_mm=ceiling_height_mm,
        )
        self.slab_calc = SlabBOQCalculator(
            default_thickness_mm=slab_thickness_mm,
            concrete_grade=concrete_grade,
        )
        self.steel_calc = SteelBOQCalculator(steel_grade=steel_grade)
        self.openings_calc = OpeningsBOQCalculator()
        self.confidence_calc = ConfidenceCalculator()

        if output_dir:
            self.exporter = BOQExporter(output_dir)
        else:
            self.exporter = None

    def run(
        self,
        image: np.ndarray,
        wall_mask: np.ndarray,
        rooms: List[Dict],
        openings: Optional[List[Dict]] = None,
        scale_px_per_mm: Optional[float] = None,
        project_id: str = "project",
        structural_mode: bool = False,
        structural_data: Optional[Dict] = None,
    ) -> BOQResult:
        """
        Run complete BOQ calculation.

        Args:
            image: Original plan image
            wall_mask: Binary wall detection mask
            rooms: List of room dicts with area, perimeter, label
            openings: List of detected openings
            scale_px_per_mm: Scale factor
            project_id: Project identifier
            structural_mode: If True, use structural drawings
            structural_data: Pre-computed structural quantities

        Returns:
            BOQResult with all quantities
        """
        logger.info(f"Starting BOQ engine for {project_id}")

        result = BOQResult(project_id=project_id)
        all_assumptions = []

        # --- MODULE 1: Wall BOQ ---
        logger.info("Module 1: Wall BOQ calculation...")
        wall_result = self.wall_calc.calculate_from_wall_mask(
            wall_mask=wall_mask,
            openings=openings,
            scale_px_per_mm=scale_px_per_mm,
        )
        result.wall_result = wall_result
        all_assumptions.extend(wall_result.assumptions_used)
        logger.info(f"Wall BOQ: {len(wall_result.boq_items)} items")

        # --- MODULE 2: Room Finish BOQ ---
        logger.info("Module 2: Finish BOQ calculation...")

        # Build room-openings map
        room_openings_map = self._build_room_openings_map(openings) if openings else None

        finish_result = self.finish_calc.calculate_all_rooms(
            rooms=rooms,
            room_openings_map=room_openings_map,
        )
        result.finish_result = finish_result
        all_assumptions.extend(finish_result.assumptions_used)
        logger.info(f"Finish BOQ: {len(finish_result.boq_items)} items")

        # --- MODULE 3: Openings BOQ ---
        if openings:
            logger.info("Module 3: Openings BOQ calculation...")
            doors = [o for o in openings if "door" in o.get("type", "").lower()]
            windows = [o for o in openings if "door" not in o.get("type", "").lower()]

            openings_result = self.openings_calc.calculate_from_openings(doors, windows)
            result.openings_result = openings_result
            all_assumptions.extend(openings_result.assumptions_used)
            logger.info(f"Openings BOQ: {len(openings_result.boq_items)} items")
        else:
            logger.info("Module 3: Skipped (no openings)")

        # --- MODULE 4: Slab BOQ ---
        logger.info("Module 4: Slab BOQ calculation...")

        # Check for wet areas (sunken slab)
        wet_area_sqm = sum(
            r.get("area_sqm", 0)
            for r in rooms
            if any(wet in r.get("label", "").lower() for wet in ["toilet", "bathroom", "wc", "utility"])
        )

        slab_result = self.slab_calc.calculate_from_rooms(
            rooms=rooms,
            thickness_mm=self.slab_thickness_mm,
        )

        # Add sunken slab items if wet areas present
        if wet_area_sqm > 0:
            sunken_items = self.slab_calc.calculate_sunken_slab(wet_area_sqm)
            slab_result.boq_items.extend(sunken_items)

        result.slab_result = slab_result
        all_assumptions.extend(slab_result.assumptions_used)
        logger.info(f"Slab BOQ: {len(slab_result.boq_items)} items")

        # --- MODULE 5: Steel BOQ ---
        logger.info("Module 5: Steel BOQ calculation...")

        if structural_mode and structural_data:
            # Use exact structural data
            concrete_volumes = structural_data.get("concrete_volumes", {})
            steel_result = self.steel_calc.calculate_from_concrete_volumes(concrete_volumes)
            all_assumptions.append("Steel from structural drawings")
        else:
            # Estimate from slab
            steel_result = self.steel_calc.calculate_slab_steel(
                slab_area_sqm=slab_result.net_area_sqm,
                slab_thickness_mm=self.slab_thickness_mm,
            )
            all_assumptions.append("Steel estimated using rule-of-thumb factors")

            # Add lintel steel if openings present
            if openings:
                lintel_result = self.steel_calc.calculate_lintel_steel(openings)
                steel_result.boq_items.extend(lintel_result.boq_items)
                steel_result.total_steel_kg += lintel_result.total_steel_kg

        result.steel_result = steel_result
        all_assumptions.extend(steel_result.assumptions_used)
        logger.info(f"Steel BOQ: {len(steel_result.boq_items)} items, {steel_result.total_steel_mt:.2f} MT")

        # --- MODULE 6: Confidence Heatmap ---
        logger.info("Module 6: Generating confidence heatmap...")

        openings_with_bbox = []
        if openings:
            for o in openings:
                if "bbox" in o:
                    openings_with_bbox.append(o)

        heatmap = self.confidence_calc.generate_heatmap(
            base_image=image,
            rooms=rooms,
            openings=openings_with_bbox,
            walls_mask=wall_mask,
        )
        result.confidence_heatmap = heatmap

        # --- Collect all items ---
        all_items = []
        if wall_result:
            all_items.extend(wall_result.boq_items)
        if finish_result:
            all_items.extend(finish_result.boq_items)
        if result.openings_result:
            all_items.extend(result.openings_result.boq_items)
        if slab_result:
            all_items.extend(slab_result.boq_items)
        if steel_result:
            all_items.extend(steel_result.boq_items)

        result.all_boq_items = all_items
        result.assumptions = all_assumptions

        # --- Calculate totals ---
        result.totals = self._calculate_totals(result)

        # --- Export ---
        if self.exporter:
            logger.info("Exporting BOQ...")
            output_paths = self.exporter.export_complete_boq(
                wall_items=wall_result.boq_items if wall_result else [],
                finish_items=finish_result.boq_items if finish_result else [],
                slab_items=slab_result.boq_items if slab_result else [],
                steel_items=steel_result.boq_items if steel_result else [],
                openings_items=result.openings_result.boq_items if result.openings_result else [],
                assumptions=all_assumptions,
                project_id=project_id,
            )

            # Save heatmap
            if heatmap:
                heatmap_path = self.exporter.save_overlay(heatmap.image, "confidence_heatmap")
                output_paths["confidence_heatmap"] = heatmap_path

            result.output_paths = output_paths

        logger.info(f"BOQ engine complete: {len(all_items)} total items")

        return result

    def _build_room_openings_map(
        self,
        openings: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Build mapping of room_id to openings."""
        room_map: Dict[str, List[Dict]] = {}

        for opening in openings:
            left_id = opening.get("room_left_id")
            right_id = opening.get("room_right_id")

            if left_id:
                if left_id not in room_map:
                    room_map[left_id] = []
                room_map[left_id].append(opening)

            if right_id and "door" in opening.get("type", "").lower():
                if right_id not in room_map:
                    room_map[right_id] = []
                room_map[right_id].append(opening)

        return room_map

    def _calculate_totals(self, result: BOQResult) -> Dict[str, Any]:
        """Calculate summary totals."""
        totals = {
            "total_items": len(result.all_boq_items),
            "total_assumptions": len(result.assumptions),
        }

        if result.wall_result:
            totals["wall_length_m"] = result.wall_result.total_wall_length_m
            totals["brick_volume_cum"] = result.wall_result.total_brick_volume_cum
            totals["plaster_area_sqm"] = result.wall_result.total_plaster_area_sqm

        if result.finish_result:
            totals["floor_area_sqm"] = result.finish_result.totals.get("total_floor_sqm", 0)
            totals["wall_finish_sqm"] = result.finish_result.totals.get("total_wall_sqm", 0)
            totals["ceiling_area_sqm"] = result.finish_result.totals.get("total_ceiling_sqm", 0)

        if result.slab_result:
            totals["slab_area_sqm"] = result.slab_result.net_area_sqm
            totals["slab_concrete_cum"] = result.slab_result.concrete_volume_cum

        if result.steel_result:
            totals["steel_mt"] = result.steel_result.total_steel_mt

        if result.openings_result:
            totals["total_doors"] = result.openings_result.totals.get("total_doors", 0)
            totals["total_windows"] = result.openings_result.totals.get("total_windows", 0)

        if result.confidence_heatmap:
            totals["overall_confidence"] = result.confidence_heatmap.overall_confidence

        return totals


def run_boq_pipeline(
    image_path: Path,
    wall_mask_path: Optional[Path] = None,
    rooms_json_path: Optional[Path] = None,
    openings_json_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    ceiling_height_mm: int = 3000,
    slab_thickness_mm: int = 125,
) -> BOQResult:
    """
    Convenience function to run BOQ pipeline from files.

    Args:
        image_path: Path to plan image
        wall_mask_path: Path to wall mask (optional, will detect)
        rooms_json_path: Path to rooms JSON
        openings_json_path: Path to openings JSON
        output_dir: Output directory
        config_path: Config YAML path
        ceiling_height_mm: Ceiling height
        slab_thickness_mm: Slab thickness

    Returns:
        BOQResult
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load or detect wall mask
    if wall_mask_path and wall_mask_path.exists():
        wall_mask = cv2.imread(str(wall_mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        # Simple wall detection
        _, wall_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)

    # Load rooms
    rooms = []
    if rooms_json_path and rooms_json_path.exists():
        with open(rooms_json_path, "r") as f:
            data = json.load(f)
            rooms = data.get("rooms", data) if isinstance(data, dict) else data

    # Load openings
    openings = []
    if openings_json_path and openings_json_path.exists():
        with open(openings_json_path, "r") as f:
            data = json.load(f)
            openings = data.get("openings", [])

    # Set up output
    if not output_dir:
        output_dir = image_path.parent / "out" / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run engine
    engine = BOQEngine(
        output_dir=output_dir,
        config_path=config_path,
        ceiling_height_mm=ceiling_height_mm,
        slab_thickness_mm=slab_thickness_mm,
    )

    result = engine.run(
        image=image,
        wall_mask=wall_mask,
        rooms=rooms,
        openings=openings,
        project_id=image_path.stem,
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BOQ engine on floor plan")
    parser.add_argument("--image", required=True, help="Path to plan image")
    parser.add_argument("--rooms", help="Path to rooms JSON")
    parser.add_argument("--openings", help="Path to openings JSON")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--ceiling-height", type=int, default=3000, help="Ceiling height in mm")
    parser.add_argument("--slab-thickness", type=int, default=125, help="Slab thickness in mm")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = run_boq_pipeline(
        image_path=Path(args.image),
        rooms_json_path=Path(args.rooms) if args.rooms else None,
        openings_json_path=Path(args.openings) if args.openings else None,
        output_dir=Path(args.output) if args.output else None,
        ceiling_height_mm=args.ceiling_height,
        slab_thickness_mm=args.slab_thickness,
    )

    print(f"\n{'='*60}")
    print("BOQ RESULTS")
    print(f"{'='*60}")
    print(f"Total BOQ Items: {result.totals.get('total_items', 0)}")
    print(f"Assumptions Made: {result.totals.get('total_assumptions', 0)}")
    print(f"\nWall Length: {result.totals.get('wall_length_m', 0):.1f} m")
    print(f"Brick Volume: {result.totals.get('brick_volume_cum', 0):.2f} cum")
    print(f"Slab Concrete: {result.totals.get('slab_concrete_cum', 0):.2f} cum")
    print(f"Steel: {result.totals.get('steel_mt', 0):.2f} MT")
    print(f"Overall Confidence: {result.totals.get('overall_confidence', 0):.1%}")
    print(f"\nOutputs: {result.output_paths}")
