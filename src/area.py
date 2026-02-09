"""
Floor Plan Area Computation Module
Computes room areas with unit conversion.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .scale import ScaleResult
from .polygons import RoomPolygon
from .labeling import LabeledRoom

logger = logging.getLogger(__name__)


@dataclass
class AreaResult:
    """Computed area for a room."""
    area_pixels: float
    area_sqm: float
    area_sqft: float
    perimeter_m: float
    is_estimated: bool = False  # True if scale was uncertain
    carpet_area_sqm: Optional[float] = None  # Carpet area (excluding walls)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RoomWithArea:
    """Room with computed area."""
    room_id: str
    label: str
    polygon: RoomPolygon
    area: AreaResult
    confidence: float


class AreaComputer:
    """
    Computes room areas from polygons using scale factor.
    """

    # Standard wall thickness for carpet area calculation (mm)
    DEFAULT_WALL_THICKNESS_MM = 150

    def __init__(self, scale: ScaleResult):
        """
        Initialize area computer.

        Args:
            scale: Scale result for unit conversion
        """
        self.scale = scale

    def compute_area(
        self,
        polygon: RoomPolygon,
        compute_carpet: bool = True
    ) -> AreaResult:
        """
        Compute area for a single polygon.

        Args:
            polygon: Room polygon
            compute_carpet: Whether to compute carpet area

        Returns:
            AreaResult
        """
        warnings = []

        # Get area in pixels
        area_pixels = polygon.area_pixels

        # Handle invalid polygons
        if area_pixels <= 0 or not polygon.is_valid:
            return AreaResult(
                area_pixels=0,
                area_sqm=0,
                area_sqft=0,
                perimeter_m=0,
                is_estimated=True,
                warnings=["invalid_polygon"]
            )

        # Convert to real units using scale
        area_sqm = self.scale.sqpixel_to_sqm(area_pixels)
        area_sqft = self.scale.sqpixel_to_sqft(area_pixels)

        # Perimeter
        perimeter_pixels = polygon.perimeter_pixels
        perimeter_mm = self.scale.pixel_to_mm(perimeter_pixels)
        perimeter_m = perimeter_mm / 1000

        # Check if scale is uncertain
        is_estimated = self.scale.confidence < 0.7
        if is_estimated:
            warnings.append("scale_uncertain")

        # Sanity checks
        if area_sqm < 0.5:
            warnings.append("tiny_area")
        elif area_sqm > 500:
            warnings.append("unusually_large_area")

        # Carpet area (subtracting wall thickness)
        carpet_area_sqm = None
        if compute_carpet and area_sqm > 1:
            carpet_area_sqm = self._compute_carpet_area(
                polygon, area_sqm, perimeter_m
            )

        return AreaResult(
            area_pixels=area_pixels,
            area_sqm=round(area_sqm, 2),
            area_sqft=round(area_sqft, 2),
            perimeter_m=round(perimeter_m, 2),
            is_estimated=is_estimated,
            carpet_area_sqm=round(carpet_area_sqm, 2) if carpet_area_sqm else None,
            warnings=warnings
        )

    def _compute_carpet_area(
        self,
        polygon: RoomPolygon,
        gross_area_sqm: float,
        perimeter_m: float
    ) -> float:
        """
        Compute carpet area (gross area minus wall area).
        Assumes half of wall thickness belongs to this room.
        """
        # Half wall thickness in meters
        half_wall_m = (self.DEFAULT_WALL_THICKNESS_MM / 2) / 1000

        # Approximate wall area contribution
        # This is perimeter * half_wall_thickness
        wall_area_sqm = perimeter_m * half_wall_m

        # Carpet area
        carpet_area = gross_area_sqm - wall_area_sqm

        return max(carpet_area, gross_area_sqm * 0.9)  # At least 90% of gross

    def compute_all(
        self,
        labeled_rooms: List[LabeledRoom]
    ) -> List[RoomWithArea]:
        """
        Compute areas for all rooms.

        Args:
            labeled_rooms: List of labeled rooms

        Returns:
            List of RoomWithArea objects
        """
        results = []

        for i, lr in enumerate(labeled_rooms):
            area_result = self.compute_area(lr.polygon)

            # Overall confidence combines label and scale confidence
            confidence = lr.label.confidence * self.scale.confidence

            results.append(RoomWithArea(
                room_id=f"R{i+1:03d}",
                label=lr.label.canonical,
                polygon=lr.polygon,
                area=area_result,
                confidence=confidence
            ))

        return results

    def compute_totals(
        self,
        rooms: List[RoomWithArea]
    ) -> dict:
        """
        Compute total areas.

        Args:
            rooms: List of rooms with areas

        Returns:
            Dictionary with total metrics
        """
        total_sqm = sum(r.area.area_sqm for r in rooms)
        total_sqft = sum(r.area.area_sqft for r in rooms)

        # Group by room type
        by_type = {}
        for room in rooms:
            # Get base type (remove numbers)
            base_type = room.label.rsplit(' ', 1)[0] if room.label[-1].isdigit() else room.label

            if base_type not in by_type:
                by_type[base_type] = {'count': 0, 'area_sqm': 0, 'area_sqft': 0}

            by_type[base_type]['count'] += 1
            by_type[base_type]['area_sqm'] += room.area.area_sqm
            by_type[base_type]['area_sqft'] += room.area.area_sqft

        # Round totals
        for bt in by_type.values():
            bt['area_sqm'] = round(bt['area_sqm'], 2)
            bt['area_sqft'] = round(bt['area_sqft'], 2)

        return {
            'total_rooms': len(rooms),
            'total_sqm': round(total_sqm, 2),
            'total_sqft': round(total_sqft, 2),
            'by_type': by_type
        }


def compute_areas(
    labeled_rooms: List[LabeledRoom],
    scale: ScaleResult
) -> List[RoomWithArea]:
    """
    Convenience function to compute areas.

    Args:
        labeled_rooms: Labeled room list
        scale: Scale result

    Returns:
        List of RoomWithArea
    """
    computer = AreaComputer(scale)
    return computer.compute_all(labeled_rooms)


def compute_polygon_area(
    polygon: RoomPolygon,
    scale: ScaleResult
) -> AreaResult:
    """
    Compute area for a single polygon.

    Args:
        polygon: Room polygon
        scale: Scale result

    Returns:
        AreaResult
    """
    computer = AreaComputer(scale)
    return computer.compute_area(polygon)


if __name__ == "__main__":
    import sys
    from .scale import ScaleResult, ScaleMethod

    logging.basicConfig(level=logging.INFO)

    # Test with mock data
    scale = ScaleResult(
        method=ScaleMethod.DEFAULT,
        pixels_per_mm=300 / 25.4 / 100,  # 300 DPI, 1:100 scale
        confidence=0.8,
        scale_ratio=100
    )

    # Create a test polygon (10m x 10m room = 100 sqm)
    # At 1:100 scale, this is 100mm x 100mm on drawing
    # At 300 DPI, 100mm = 100 * 300/25.4 = 1181 pixels
    side_pixels = 1181

    test_polygon = RoomPolygon(
        room_id="test",
        points=[
            (0, 0), (side_pixels, 0),
            (side_pixels, side_pixels), (0, side_pixels)
        ],
        area_pixels=side_pixels * side_pixels,
        perimeter_pixels=side_pixels * 4,
        bbox=(0, 0, side_pixels, side_pixels),
        centroid=(side_pixels/2, side_pixels/2),
        is_valid=True
    )

    computer = AreaComputer(scale)
    result = computer.compute_area(test_polygon)

    print("Area computation test (10m x 10m room):")
    print(f"  Pixels: {result.area_pixels:.0f}")
    print(f"  Area (sqm): {result.area_sqm:.2f} (expected ~100)")
    print(f"  Area (sqft): {result.area_sqft:.2f} (expected ~1076)")
    print(f"  Perimeter (m): {result.perimeter_m:.2f} (expected ~40)")
    print(f"  Carpet area: {result.carpet_area_sqm}")
    print(f"  Warnings: {result.warnings}")
