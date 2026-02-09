"""
Floor Plan Polygon Extraction Module
Converts region masks to simplified polygons.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try to import shapely for advanced polygon operations
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not available - some polygon operations disabled")


@dataclass
class RoomPolygon:
    """Represents a room as a polygon."""
    room_id: str
    points: List[Tuple[float, float]]  # Vertices in image coordinates
    area_pixels: float
    perimeter_pixels: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    is_valid: bool = True
    is_convex: bool = False
    has_holes: List[List[Tuple[float, float]]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_vertices(self) -> int:
        return len(self.points)

    def to_numpy(self) -> np.ndarray:
        """Convert points to numpy array."""
        return np.array(self.points, dtype=np.float32)

    def to_cv_contour(self) -> np.ndarray:
        """Convert to OpenCV contour format."""
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))


class PolygonExtractor:
    """
    Extracts and simplifies polygons from region masks.
    """

    def __init__(self):
        self.simplification_epsilon = 2.0  # RDP simplification tolerance
        self.min_vertices = 4  # Minimum vertices after simplification
        self.max_vertices = 100  # Maximum vertices
        self.spike_angle_threshold = 15  # Degrees - spikes below this are removed

    def extract(
        self,
        region_mask: np.ndarray,
        region_id: str,
        simplify: bool = True
    ) -> RoomPolygon:
        """
        Extract polygon from region mask.

        Args:
            region_mask: Binary mask of region
            region_id: ID for the room
            simplify: Whether to simplify polygon

        Returns:
            RoomPolygon
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            region_mask,
            cv2.RETR_CCOMP,  # Get both outer and inner contours
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning(f"No contours found for region {region_id}")
            return RoomPolygon(
                room_id=region_id,
                points=[],
                area_pixels=0,
                perimeter_pixels=0,
                bbox=(0, 0, 0, 0),
                centroid=(0, 0),
                is_valid=False,
                warnings=["no_contours"]
            )

        # Find the largest contour (main room boundary)
        main_contour = max(contours, key=cv2.contourArea)

        # Simplify contour
        if simplify:
            epsilon = self.simplification_epsilon
            simplified = cv2.approxPolyDP(main_contour, epsilon, True)

            # If too few points, reduce epsilon
            while len(simplified) < self.min_vertices and epsilon > 0.5:
                epsilon *= 0.5
                simplified = cv2.approxPolyDP(main_contour, epsilon, True)

            # If too many points, increase epsilon
            while len(simplified) > self.max_vertices and epsilon < 20:
                epsilon *= 1.5
                simplified = cv2.approxPolyDP(main_contour, epsilon, True)

            main_contour = simplified

        # Convert to points list
        points = [(float(p[0][0]), float(p[0][1])) for p in main_contour]

        # Remove spikes (very acute angles)
        points = self._remove_spikes(points)

        # Ensure consistent orientation (counter-clockwise)
        if self._is_clockwise(points):
            points = points[::-1]

        # Calculate properties
        contour_np = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        area = cv2.contourArea(contour_np)
        perimeter = cv2.arcLength(contour_np, True)
        bbox = cv2.boundingRect(contour_np.astype(np.int32))

        # Calculate centroid
        moments = cv2.moments(contour_np)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = points[0] if points else (0, 0)

        # Check convexity
        is_convex = cv2.isContourConvex(contour_np.astype(np.int32))

        # Find holes (inner contours)
        holes = []
        if hierarchy is not None:
            for i, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
                # Inner contour has parent (hier[3] != -1) and no children
                if hier[3] != -1 and cv2.contourArea(contour) > 50:
                    hole_points = [(float(p[0][0]), float(p[0][1])) for p in contour]
                    holes.append(hole_points)

        # Validate polygon
        warnings = []
        is_valid = True

        if len(points) < 3:
            is_valid = False
            warnings.append("too_few_vertices")

        if area < 100:
            is_valid = False
            warnings.append("tiny_area")

        # Check for self-intersection using shapely if available
        if SHAPELY_AVAILABLE and len(points) >= 3:
            try:
                poly = ShapelyPolygon(points)
                if not poly.is_valid:
                    # Try to fix
                    fixed = make_valid(poly)
                    if fixed.geom_type == 'Polygon':
                        points = list(fixed.exterior.coords)[:-1]  # Remove closing point
                        warnings.append("self_intersection_fixed")
                    else:
                        warnings.append("self_intersection_unfixable")
            except Exception as e:
                logger.debug(f"Shapely validation failed: {e}")

        return RoomPolygon(
            room_id=region_id,
            points=points,
            area_pixels=area,
            perimeter_pixels=perimeter,
            bbox=bbox,
            centroid=(cx, cy),
            is_valid=is_valid,
            is_convex=is_convex,
            has_holes=holes,
            warnings=warnings
        )

    def _remove_spikes(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove spike vertices (very acute angles)."""
        if len(points) < 4:
            return points

        threshold = np.radians(self.spike_angle_threshold)
        filtered = []

        n = len(points)
        for i in range(n):
            p_prev = points[(i - 1) % n]
            p_curr = points[i]
            p_next = points[(i + 1) % n]

            # Calculate vectors
            v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            # Calculate angle
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if len1 > 0 and len2 > 0:
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                # Keep point if angle is not too acute
                if angle > threshold:
                    filtered.append(p_curr)
            else:
                filtered.append(p_curr)

        return filtered if len(filtered) >= 3 else points

    def _is_clockwise(self, points: List[Tuple[float, float]]) -> bool:
        """Check if polygon is clockwise oriented."""
        if len(points) < 3:
            return False

        # Shoelace formula for signed area
        total = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            total += (x2 - x1) * (y2 + y1)

        return total > 0


def merge_polygons(polygons: List[RoomPolygon]) -> Optional[RoomPolygon]:
    """
    Merge multiple polygons into one.
    Useful for L-shaped rooms that were split.
    """
    if not SHAPELY_AVAILABLE:
        logger.warning("Shapely required for polygon merging")
        return None

    if not polygons:
        return None

    if len(polygons) == 1:
        return polygons[0]

    try:
        shapely_polys = []
        for p in polygons:
            if len(p.points) >= 3:
                shapely_polys.append(ShapelyPolygon(p.points))

        if not shapely_polys:
            return None

        merged = unary_union(shapely_polys)

        if merged.geom_type != 'Polygon':
            # Union resulted in multiple polygons - take largest
            if merged.geom_type == 'MultiPolygon':
                merged = max(merged.geoms, key=lambda g: g.area)
            else:
                return None

        # Convert back to RoomPolygon
        points = list(merged.exterior.coords)[:-1]
        bbox = (
            int(merged.bounds[0]),
            int(merged.bounds[1]),
            int(merged.bounds[2] - merged.bounds[0]),
            int(merged.bounds[3] - merged.bounds[1])
        )

        return RoomPolygon(
            room_id=f"merged_{polygons[0].room_id}",
            points=points,
            area_pixels=merged.area,
            perimeter_pixels=merged.length,
            bbox=bbox,
            centroid=(merged.centroid.x, merged.centroid.y),
            is_valid=True,
            is_convex=merged.convex_hull.area == merged.area,
            warnings=["merged_polygon"]
        )

    except Exception as e:
        logger.warning(f"Polygon merge failed: {e}")
        return None


def extract_polygon(
    region_mask: np.ndarray,
    region_id: str,
    simplify: bool = True
) -> RoomPolygon:
    """
    Convenience function to extract polygon from mask.
    """
    extractor = PolygonExtractor()
    return extractor.extract(region_mask, region_id, simplify)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            polygon = extract_polygon(mask, "test_room")

            print(f"Room ID: {polygon.room_id}")
            print(f"Vertices: {polygon.num_vertices}")
            print(f"Area (px): {polygon.area_pixels:.0f}")
            print(f"Perimeter (px): {polygon.perimeter_pixels:.0f}")
            print(f"BBox: {polygon.bbox}")
            print(f"Centroid: {polygon.centroid}")
            print(f"Convex: {polygon.is_convex}")
            print(f"Holes: {len(polygon.has_holes)}")
            print(f"Valid: {polygon.is_valid}")
            print(f"Warnings: {polygon.warnings}")

            # Visualize
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            pts = polygon.to_cv_contour()
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            # Draw vertices
            for pt in polygon.points:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

            # Draw centroid
            cv2.circle(vis, (int(polygon.centroid[0]), int(polygon.centroid[1])), 5, (255, 0, 0), -1)

            cv2.imwrite("polygon_vis.png", vis)
            print("Saved polygon visualization")
    else:
        print("Usage: python polygons.py <mask_image>")
