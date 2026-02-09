"""
Floor Plan Region Detection Module
Identifies enclosed regions (rooms) from wall masks.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import numpy as np
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class RegionCandidate:
    """A candidate room region."""
    region_id: int
    mask: np.ndarray  # Binary mask of this region
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_pixels: int
    centroid: Tuple[float, float]
    is_valid: bool = True
    touches_boundary: bool = False
    is_hole: bool = False  # Internal hole (column, shaft)
    merge_with: Optional[int] = None  # ID of region to merge with


@dataclass
class RegionDetectionResult:
    """Result of region detection."""
    regions: List[RegionCandidate]
    room_mask: np.ndarray  # Labeled mask (each room has unique ID)
    background_mask: np.ndarray  # External area mask
    discarded_regions: List[RegionCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RegionDetector:
    """
    Detects enclosed regions (rooms) from wall mask.
    """

    def __init__(self):
        # Size thresholds (in pixels, will be scaled by image size)
        self.min_room_area_ratio = 0.001  # Minimum room as ratio of image
        self.max_room_area_ratio = 0.5  # Maximum room as ratio of image
        self.min_aspect_ratio = 0.1  # Minimum width/height ratio
        self.max_aspect_ratio = 10.0  # Maximum width/height ratio
        self.min_solidity = 0.3  # Minimum area/convex_hull ratio
        self.hole_max_area_ratio = 0.005  # Maximum area for internal holes

    def detect(
        self,
        wall_mask: np.ndarray,
        external_boundary: Optional[np.ndarray] = None
    ) -> RegionDetectionResult:
        """
        Detect room regions from wall mask.

        Args:
            wall_mask: Binary wall mask (white = walls)
            external_boundary: Optional external boundary contour

        Returns:
            RegionDetectionResult
        """
        logger.info("Detecting regions from wall mask")

        h, w = wall_mask.shape
        image_area = h * w

        # Compute size thresholds based on image size
        min_room_area = int(image_area * self.min_room_area_ratio)
        max_room_area = int(image_area * self.max_room_area_ratio)
        max_hole_area = int(image_area * self.hole_max_area_ratio)

        # Invert wall mask - rooms become white, walls become black
        room_space = cv2.bitwise_not(wall_mask)

        # Create background mask from flood fill at edges
        background_mask = self._create_background_mask(room_space, external_boundary)

        # Remove background from room space
        room_space_internal = cv2.bitwise_and(
            room_space,
            cv2.bitwise_not(background_mask)
        )

        # Label connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            room_space_internal, connectivity=4
        )

        logger.info(f"Found {num_labels - 1} connected components")

        regions = []
        discarded = []
        warnings = []

        for i in range(1, num_labels):  # Skip background (0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            # Create region mask
            region_mask = (labels == i).astype(np.uint8) * 255

            # Check if touches boundary
            touches_boundary = self._touches_boundary(region_mask, background_mask)

            candidate = RegionCandidate(
                region_id=i,
                mask=region_mask,
                bbox=(x, y, width, height),
                area_pixels=area,
                centroid=(cx, cy),
                touches_boundary=touches_boundary
            )

            # Validate region
            is_valid, reason = self._validate_region(
                candidate, min_room_area, max_room_area, max_hole_area
            )

            if is_valid:
                regions.append(candidate)
            else:
                candidate.is_valid = False
                discarded.append(candidate)
                if reason:
                    warnings.append(f"Region {i}: {reason}")

        # Post-process: identify holes and potential merges
        regions = self._identify_holes(regions, max_hole_area)
        regions = self._identify_merges(regions, wall_mask)

        # Create labeled room mask
        room_mask = np.zeros((h, w), dtype=np.int32)
        for i, region in enumerate(regions, 1):
            if region.is_valid and not region.is_hole:
                room_mask[region.mask > 0] = i

        logger.info(f"Detected {len([r for r in regions if r.is_valid and not r.is_hole])} valid rooms")

        return RegionDetectionResult(
            regions=regions,
            room_mask=room_mask,
            background_mask=background_mask,
            discarded_regions=discarded,
            warnings=warnings
        )

    def _create_background_mask(
        self,
        room_space: np.ndarray,
        external_boundary: Optional[np.ndarray]
    ) -> np.ndarray:
        """Create mask of external (outside building) area."""
        h, w = room_space.shape
        background = np.zeros((h, w), dtype=np.uint8)

        # If we have external boundary, use it
        if external_boundary is not None:
            # Fill outside the boundary
            cv2.drawContours(background, [external_boundary], -1, 255, -1)
            background = cv2.bitwise_not(background)
            background = cv2.bitwise_and(background, room_space)
            return background

        # Otherwise, flood fill from edges
        # Work on a copy with padding
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = room_space

        # Flood fill from corners and edges
        fill_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)

        # Fill from edges
        edge_points = []
        for x in range(0, w, 20):
            edge_points.extend([(x, 0), (x, h-1)])
        for y in range(0, h, 20):
            edge_points.extend([(0, y), (w-1, y)])

        for px, py in edge_points:
            if padded[py + 1, px + 1] > 0:  # Only fill if it's open space
                cv2.floodFill(
                    padded, fill_mask,
                    (px + 1, py + 1), 128,
                    flags=4 | (128 << 8)
                )

        # Extract filled region
        background = (padded[1:-1, 1:-1] == 128).astype(np.uint8) * 255

        return background

    def _touches_boundary(
        self,
        region_mask: np.ndarray,
        background_mask: np.ndarray
    ) -> bool:
        """Check if region touches external boundary."""
        # Dilate region slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(region_mask, kernel)

        # Check overlap with background
        overlap = cv2.bitwise_and(dilated, background_mask)
        return np.sum(overlap > 0) > 0

    def _validate_region(
        self,
        region: RegionCandidate,
        min_area: int,
        max_area: int,
        max_hole_area: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate if region is a valid room."""

        # Size check
        if region.area_pixels < min_area:
            return False, "tiny_region_discarded"

        if region.area_pixels > max_area:
            return False, "region_too_large"

        # Aspect ratio check
        x, y, w, h = region.bbox
        aspect = w / h if h > 0 else 0

        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return False, "invalid_aspect_ratio"

        # Solidity check (detect irregular shapes)
        contours, _ = cv2.findContours(
            region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = region.area_pixels / hull_area
                if solidity < self.min_solidity:
                    return False, "irregular_shape"

        return True, None

    def _identify_holes(
        self,
        regions: List[RegionCandidate],
        max_hole_area: int
    ) -> List[RegionCandidate]:
        """Identify small regions that are likely internal holes (columns, shafts)."""

        for region in regions:
            if region.area_pixels < max_hole_area:
                # Check if completely surrounded by other regions
                # Small regions inside larger ones are holes
                surrounding_count = 0
                for other in regions:
                    if other.region_id != region.region_id:
                        # Check if other region surrounds this one
                        ox, oy, ow, oh = other.bbox
                        rx, ry, rw, rh = region.bbox

                        if (ox <= rx and oy <= ry and
                            ox + ow >= rx + rw and oy + oh >= ry + rh):
                            surrounding_count += 1

                if surrounding_count > 0:
                    region.is_hole = True

        return regions

    def _identify_merges(
        self,
        regions: List[RegionCandidate],
        wall_mask: np.ndarray
    ) -> List[RegionCandidate]:
        """
        Identify regions that should be merged.
        (e.g., L-shaped rooms that got split)
        """
        # For now, skip merge detection - it's complex and often causes issues
        # Can be implemented later with more sophisticated logic
        return regions


def detect_regions(
    wall_mask: np.ndarray,
    external_boundary: Optional[np.ndarray] = None
) -> RegionDetectionResult:
    """
    Convenience function to detect regions.

    Args:
        wall_mask: Binary wall mask
        external_boundary: Optional external boundary

    Returns:
        RegionDetectionResult
    """
    detector = RegionDetector()
    return detector.detect(wall_mask, external_boundary)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        wall_mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if wall_mask is not None:
            result = detect_regions(wall_mask)

            print(f"Total regions: {len(result.regions)}")
            print(f"Valid rooms: {len([r for r in result.regions if r.is_valid and not r.is_hole])}")
            print(f"Holes: {len([r for r in result.regions if r.is_hole])}")
            print(f"Discarded: {len(result.discarded_regions)}")
            print(f"Warnings: {result.warnings}")

            # Visualize
            vis = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 0), (0, 128, 0), (0, 0, 128),
            ]

            for i, region in enumerate(result.regions):
                if region.is_valid and not region.is_hole:
                    color = colors[i % len(colors)]
                    vis[region.mask > 0] = color

            cv2.imwrite("regions_vis.png", vis)
            print("Saved regions visualization")
    else:
        print("Usage: python regions.py <wall_mask_image>")
