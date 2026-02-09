"""
Floor Plan Wall Detection Module
Hybrid approach: vector extraction + morphological operations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage

from .ingest import IngestedPlan, VectorLine, PlanType

logger = logging.getLogger(__name__)


@dataclass
class WallSegment:
    """Represents a wall segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    thickness: float = 1.0
    is_external: bool = False

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx*dx + dy*dy)

    @property
    def is_horizontal(self) -> bool:
        return abs(self.end[1] - self.start[1]) < 3.0

    @property
    def is_vertical(self) -> bool:
        return abs(self.end[0] - self.start[0]) < 3.0


@dataclass
class WallDetectionResult:
    """Result of wall detection."""
    wall_mask: np.ndarray  # Binary mask of walls
    wall_mask_closed: np.ndarray  # Mask with gaps closed
    wall_segments: List[WallSegment] = field(default_factory=list)
    external_boundary: Optional[np.ndarray] = None  # Contour of outer wall
    door_openings: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    method: str = "hybrid"


class WallDetector:
    """
    Detects walls in floor plans using hybrid approach.
    """

    def __init__(self):
        self.min_wall_length = 30  # Minimum wall segment length
        self.wall_thickness_range = (2, 30)  # Expected wall thickness in pixels
        self.gap_close_size = 15  # Maximum gap to close (door openings)

    def detect(self, plan: IngestedPlan, binary_image: np.ndarray = None) -> WallDetectionResult:
        """
        Detect walls in floor plan.

        Args:
            plan: Ingested plan with vector/raster data
            binary_image: Pre-binarized image (optional)

        Returns:
            WallDetectionResult
        """
        logger.info(f"Detecting walls for plan: {plan.plan_id}")

        if plan.plan_type == PlanType.VECTOR_PDF and plan.has_vector_content:
            # Use vector-based detection
            result = self._detect_from_vectors(plan)
        else:
            # Use morphological detection on raster
            result = self._detect_from_raster(plan.image, binary_image)

        # Post-process: close gaps and detect boundary
        result.wall_mask_closed = self._close_gaps(result.wall_mask)
        result.external_boundary = self._find_external_boundary(result.wall_mask_closed)

        return result

    def _detect_from_vectors(self, plan: IngestedPlan) -> WallDetectionResult:
        """Detect walls from vector lines."""
        h, w = plan.image_shape

        # Create empty mask
        wall_mask = np.zeros((h, w), dtype=np.uint8)

        wall_segments = []

        # Filter lines that are likely walls
        for line in plan.vector_lines:
            # Skip very short lines
            if line.length < self.min_wall_length:
                continue

            # Check line width - walls tend to be thicker
            if line.width < self.wall_thickness_range[0]:
                continue

            # Draw line on mask
            pt1 = (int(line.start[0]), int(line.start[1]))
            pt2 = (int(line.end[0]), int(line.end[1]))
            thickness = max(2, int(line.width))

            cv2.line(wall_mask, pt1, pt2, 255, thickness)

            wall_segments.append(WallSegment(
                start=line.start,
                end=line.end,
                thickness=line.width
            ))

        # Also process closed paths (rectangles often represent rooms)
        for path in plan.vector_paths:
            if len(path.points) >= 3:
                points = np.array(path.points, dtype=np.int32)
                thickness = max(2, int(path.stroke_width))

                # Draw path
                if path.is_closed:
                    cv2.polylines(wall_mask, [points], True, 255, thickness)
                else:
                    cv2.polylines(wall_mask, [points], False, 255, thickness)

        # Fill in wall thickness if mask is too thin
        if np.sum(wall_mask > 0) < h * w * 0.01:  # Less than 1%
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)

        return WallDetectionResult(
            wall_mask=wall_mask,
            wall_mask_closed=wall_mask.copy(),
            wall_segments=wall_segments,
            method="vector"
        )

    def _detect_from_raster(
        self,
        image: np.ndarray,
        binary: np.ndarray = None
    ) -> WallDetectionResult:
        """Detect walls from raster image using morphology."""

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # Binarize if not provided
        if binary is None:
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                35, 10
            )
        else:
            # Resize binary to match gray if needed
            bh, bw = binary.shape[:2]
            if (bh, bw) != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # Multi-scale wall detection
        wall_mask = np.zeros((h, w), dtype=np.uint8)

        # Detect horizontal lines at different scales
        for length in [50, 100, 200]:
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
            wall_mask = cv2.bitwise_or(wall_mask, h_lines)

        # Detect vertical lines at different scales
        for length in [50, 100, 200]:
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
            wall_mask = cv2.bitwise_or(wall_mask, v_lines)

        # Detect thick elements (walls are usually thicker than other lines)
        thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thick_elements = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thick_kernel)

        # Combine with directional detections
        wall_mask = cv2.bitwise_or(wall_mask, thick_elements)

        # Thicken walls slightly to ensure connectivity
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        wall_mask = cv2.dilate(wall_mask, dilate_kernel, iterations=1)

        # Extract wall segments using Hough transform
        wall_segments = self._extract_wall_segments(wall_mask)

        return WallDetectionResult(
            wall_mask=wall_mask,
            wall_mask_closed=wall_mask.copy(),
            wall_segments=wall_segments,
            method="morphological"
        )

    def _close_gaps(self, wall_mask: np.ndarray) -> np.ndarray:
        """
        Close small gaps in walls (door openings, windows).
        Controlled closing to avoid merging rooms.
        """
        closed = wall_mask.copy()

        # Direction-specific closing
        # Horizontal gaps (doors in vertical walls)
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.gap_close_size, 3)
        )
        h_closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, h_kernel)

        # Vertical gaps (doors in horizontal walls)
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, self.gap_close_size)
        )
        v_closed = cv2.morphologyEx(h_closed, cv2.MORPH_CLOSE, v_kernel)

        # Small diagonal gaps
        sq_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )
        closed = cv2.morphologyEx(v_closed, cv2.MORPH_CLOSE, sq_kernel)

        return closed

    def _find_external_boundary(self, wall_mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the external boundary of the plan."""
        # Invert mask - rooms become white, walls become black
        inverted = cv2.bitwise_not(wall_mask)

        # Flood fill from corners to mark external area
        h, w = wall_mask.shape
        external = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood fill from all edges
        for x in range(0, w, 50):
            cv2.floodFill(inverted, external, (x, 0), 128)
            cv2.floodFill(inverted, external, (x, h-1), 128)
        for y in range(0, h, 50):
            cv2.floodFill(inverted, external, (0, y), 128)
            cv2.floodFill(inverted, external, (w-1, y), 128)

        # External area is now marked with 128
        external_mask = (inverted == 128).astype(np.uint8) * 255

        # Find contours of external area
        contours, _ = cv2.findContours(
            cv2.bitwise_not(external_mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Return largest contour (main building boundary)
            largest = max(contours, key=cv2.contourArea)
            return largest

        return None

    def _extract_wall_segments(self, wall_mask: np.ndarray) -> List[WallSegment]:
        """Extract wall segments using Hough transform."""
        segments = []

        # Detect lines
        lines = cv2.HoughLinesP(
            wall_mask,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_wall_length,
            maxLineGap=10
        )

        if lines is None:
            return segments

        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(WallSegment(
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
                thickness=2.0
            ))

        return segments

    def detect_door_openings(
        self,
        original_mask: np.ndarray,
        closed_mask: np.ndarray
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect door openings by comparing original and closed masks.
        """
        # Difference shows where gaps were closed
        diff = cv2.subtract(closed_mask, original_mask)

        # Find connected components in difference
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            diff, connectivity=8
        )

        openings = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by size (door openings are typically 600-1200mm)
            if 20 < area < 2000:
                openings.append(((x, y), (x + w, y + h)))

        return openings


def detect_walls(plan: IngestedPlan, binary_image: np.ndarray = None) -> WallDetectionResult:
    """
    Convenience function to detect walls.

    Args:
        plan: Ingested floor plan
        binary_image: Optional pre-binarized image

    Returns:
        WallDetectionResult
    """
    detector = WallDetector()
    return detector.detect(plan, binary_image)


if __name__ == "__main__":
    import sys
    from .ingest import ingest_plan

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        plan = ingest_plan(sys.argv[1])
        result = detect_walls(plan)

        print(f"Detection method: {result.method}")
        print(f"Wall segments: {len(result.wall_segments)}")
        print(f"Has external boundary: {result.external_boundary is not None}")

        # Save masks
        cv2.imwrite("wall_mask.png", result.wall_mask)
        cv2.imwrite("wall_mask_closed.png", result.wall_mask_closed)
        print("Saved wall masks")
    else:
        print("Usage: python walls.py <plan_file>")
