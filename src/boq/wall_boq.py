"""
Wall BOQ Calculator
- Wall thickness clustering (230mm, 115mm, etc.)
- Brick/block volume calculation
- Plaster area (both faces)
- Opening deductions

Indian construction practices:
- 230mm (9") external walls - brick/AAC block
- 115mm (4.5") internal walls - brick
- 75mm partition walls - AAC block
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from collections import defaultdict
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class WallSegment:
    """Individual wall segment with measurements."""
    segment_id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length_m: float
    thickness_mm: int
    wall_type: str  # external, internal, partition
    orientation: str  # horizontal, vertical, diagonal
    confidence: float

    @property
    def area_sqm(self) -> float:
        """Wall face area assuming standard height."""
        return self.length_m * 3.0  # Default 3m height

    def area_with_height(self, height_m: float) -> float:
        """Wall face area with specified height."""
        return self.length_m * height_m


@dataclass
class WallBOQItem:
    """BOQ line item for wall-related work."""
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    wall_type: Optional[str] = None
    thickness_mm: Optional[int] = None
    deductions: Dict[str, float] = field(default_factory=dict)


@dataclass
class WallBOQResult:
    """Complete wall BOQ result."""
    segments: List[WallSegment]
    thickness_clusters: Dict[int, float]  # thickness_mm -> total_length_m
    boq_items: List[WallBOQItem]
    total_wall_length_m: float
    total_brick_volume_cum: float
    total_plaster_area_sqm: float
    assumptions_used: List[str]


class WallBOQCalculator:
    """
    Calculate wall BOQ from wall detection results.

    Features:
    - Thickness clustering using K-means
    - External/internal wall classification
    - Brick/block volume calculation
    - Plaster area with opening deductions
    """

    # Standard wall thicknesses in mm (Indian construction)
    STANDARD_THICKNESSES = {
        230: {"name": "9-inch brick", "type": "external", "mortar": "CM 1:6"},
        200: {"name": "AAC block 200mm", "type": "external", "mortar": "CM 1:6"},
        150: {"name": "AAC block 150mm", "type": "internal", "mortar": "CM 1:4"},
        115: {"name": "4.5-inch brick", "type": "internal", "mortar": "CM 1:4"},
        100: {"name": "AAC block 100mm", "type": "partition", "mortar": "CM 1:4"},
        75: {"name": "3-inch partition", "type": "partition", "mortar": "CM 1:4"},
    }

    # Mortar consumption per cum of brickwork
    MORTAR_CONSUMPTION = {
        230: 0.30,  # cum mortar per cum brickwork
        200: 0.02,  # AAC block - thin bed mortar
        150: 0.02,
        115: 0.28,
        100: 0.02,
        75: 0.25,
    }

    def __init__(
        self,
        ceiling_height_mm: int = 3000,
        plaster_thickness_mm: int = 12,
        scale_px_per_mm: Optional[float] = None,
    ):
        self.ceiling_height_mm = ceiling_height_mm
        self.ceiling_height_m = ceiling_height_mm / 1000
        self.plaster_thickness_mm = plaster_thickness_mm
        self.scale_px_per_mm = scale_px_per_mm
        self.assumptions_used: List[str] = []

    def cluster_wall_thicknesses(
        self,
        wall_mask: np.ndarray,
        num_clusters: int = 3,
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Cluster wall segments by thickness.

        Uses distance transform and contour analysis to measure
        wall thicknesses and cluster them into standard sizes.
        """
        if wall_mask is None or wall_mask.size == 0:
            return {}

        # Ensure binary
        _, binary = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)

        # Distance transform to find wall centerline distances
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Find skeleton/centerline
        skeleton = self._skeletonize(binary)

        # Sample thickness at centerline points
        thickness_samples = []
        ys, xs = np.where(skeleton > 0)

        for y, x in zip(ys[::10], xs[::10]):  # Sample every 10th point
            thickness_px = dist_transform[y, x] * 2  # Full wall thickness
            if thickness_px > 2:
                thickness_samples.append((x, y, thickness_px))

        if not thickness_samples:
            return {}

        # Extract just thickness values for clustering
        thicknesses = np.array([t[2] for t in thickness_samples]).reshape(-1, 1)

        # Cluster
        n_clusters = min(num_clusters, len(thickness_samples))
        if n_clusters < 1:
            return {}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(thicknesses)

        # Group by cluster
        clusters = defaultdict(list)
        for i, (x, y, t) in enumerate(thickness_samples):
            cluster_center = kmeans.cluster_centers_[labels[i]][0]
            # Map to standard thickness
            std_thickness = self._map_to_standard_thickness(cluster_center)
            clusters[std_thickness].append((x, y, int(t), int(t)))

        return dict(clusters)

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Create skeleton of binary image using thinning."""
        skeleton = np.zeros_like(binary)
        temp = binary.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(temp, kernel)
            temp_opened = cv2.dilate(eroded, kernel)
            temp_diff = cv2.subtract(temp, temp_opened)
            skeleton = cv2.bitwise_or(skeleton, temp_diff)
            temp = eroded.copy()

            if cv2.countNonZero(temp) == 0:
                break

        return skeleton

    def _map_to_standard_thickness(self, thickness_px: float) -> int:
        """Map measured thickness to nearest standard thickness."""
        if self.scale_px_per_mm and self.scale_px_per_mm > 0:
            thickness_mm = thickness_px / self.scale_px_per_mm
        else:
            # Assume typical scale if not provided
            thickness_mm = thickness_px * 2.5  # Rough estimate
            # Only add assumption once
            if "Wall thickness scale assumed" not in self.assumptions_used:
                self.assumptions_used.append("Wall thickness scale assumed")

        # Find nearest standard
        min_diff = float('inf')
        nearest = 230

        for std_mm in self.STANDARD_THICKNESSES.keys():
            diff = abs(thickness_mm - std_mm)
            if diff < min_diff:
                min_diff = diff
                nearest = std_mm

        return nearest

    def analyze_wall_segments(
        self,
        wall_mask: np.ndarray,
        thickness_clusters: Dict[int, List],
    ) -> List[WallSegment]:
        """
        Analyze wall segments from mask and thickness clusters.

        Returns list of WallSegment objects with measurements.
        """
        segments = []
        segment_id = 0

        # Find contours
        contours, _ = cv2.findContours(
            wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Fit minimum area rectangle
            rect = cv2.minAreaRect(contour)
            center, (w, h), angle = rect

            # Determine orientation
            if w > h:
                length_px = w
                thickness_px = h
                orientation = "horizontal" if abs(angle) < 45 else "vertical"
            else:
                length_px = h
                thickness_px = w
                orientation = "vertical" if abs(angle) < 45 else "horizontal"

            # Convert to real units
            if self.scale_px_per_mm and self.scale_px_per_mm > 0:
                length_m = (length_px / self.scale_px_per_mm) / 1000
            else:
                length_m = length_px * 0.003  # Rough estimate

            # Map thickness to standard
            thickness_mm = self._map_to_standard_thickness(thickness_px)

            # Determine wall type
            wall_info = self.STANDARD_THICKNESSES.get(thickness_mm, {})
            wall_type = wall_info.get("type", "internal")

            # Calculate endpoints
            box = cv2.boxPoints(rect)
            if w > h:
                start = tuple(box[0])
                end = tuple(box[2])
            else:
                start = tuple(box[1])
                end = tuple(box[3])

            segment = WallSegment(
                segment_id=f"W-{segment_id:03d}",
                start_point=start,
                end_point=end,
                length_m=length_m,
                thickness_mm=thickness_mm,
                wall_type=wall_type,
                orientation=orientation,
                confidence=0.7,
            )
            segments.append(segment)
            segment_id += 1

        return segments

    def calculate_wall_boq(
        self,
        segments: List[WallSegment],
        openings: Optional[List[Dict]] = None,
        ceiling_height_m: Optional[float] = None,
    ) -> WallBOQResult:
        """
        Calculate complete wall BOQ from segments.

        Includes:
        - Brickwork/blockwork volumes
        - Plaster areas (both faces)
        - Opening deductions
        """
        if ceiling_height_m:
            self.ceiling_height_m = ceiling_height_m

        boq_items = []

        # Group by thickness
        by_thickness: Dict[int, List[WallSegment]] = defaultdict(list)
        for seg in segments:
            by_thickness[seg.thickness_mm].append(seg)

        # Calculate opening deductions
        opening_area = 0.0
        opening_perimeter = 0.0
        if openings:
            for opening in openings:
                w = opening.get("width_m", 0.9)
                h = opening.get("height_m", 2.1)
                if w and h:
                    opening_area += w * h
                    opening_perimeter += 2 * (w + h)

        total_wall_length = sum(seg.length_m for seg in segments)
        total_brick_volume = 0.0
        total_plaster_area = 0.0

        thickness_totals: Dict[int, float] = {}

        for thickness_mm, segs in by_thickness.items():
            total_length = sum(s.length_m for s in segs)
            thickness_totals[thickness_mm] = total_length

            wall_info = self.STANDARD_THICKNESSES.get(thickness_mm, {})
            wall_name = wall_info.get("name", f"{thickness_mm}mm wall")
            wall_type = wall_info.get("type", "internal")
            mortar = wall_info.get("mortar", "CM 1:4")

            # Gross wall area
            gross_area = total_length * self.ceiling_height_m

            # Deduct openings (proportionally distribute)
            # Cap opening deduction at 30% of gross area to avoid over-deduction
            proportion = total_length / total_wall_length if total_wall_length > 0 else 0
            opening_deduction = opening_area * proportion
            max_deduction = gross_area * 0.30  # Max 30% for openings
            opening_deduction = min(opening_deduction, max_deduction)
            net_area = max(0, gross_area - opening_deduction)

            # Brick/block volume
            volume_cum = net_area * (thickness_mm / 1000)
            total_brick_volume += volume_cum

            # Add brickwork item
            boq_items.append(WallBOQItem(
                item_code=f"MSN-{thickness_mm:03d}",
                description=f"Providing and laying {wall_name} in {mortar} mortar",
                qty=round(volume_cum, 2),
                unit="cum",
                derived_from="wall_detection",
                confidence=0.75,
                wall_type=wall_type,
                thickness_mm=thickness_mm,
                deductions={"openings_sqm": round(opening_deduction, 2)},
            ))

            # Plaster area (both faces)
            plaster_area = net_area * 2
            total_plaster_area += plaster_area

            # Add plaster item
            plaster_type = "internal" if wall_type != "external" else "external"
            boq_items.append(WallBOQItem(
                item_code=f"PLT-{plaster_type[:3].upper()}-01",
                description=f"Providing {self.plaster_thickness_mm}mm cement plaster (1:4) on {wall_type} walls",
                qty=round(plaster_area, 2),
                unit="sqm",
                derived_from="wall_detection",
                confidence=0.72,
                wall_type=wall_type,
                thickness_mm=thickness_mm,
            ))

        # Add mortar items
        for thickness_mm, segs in by_thickness.items():
            total_length = sum(s.length_m for s in segs)
            gross_area = total_length * self.ceiling_height_m
            volume_cum = gross_area * (thickness_mm / 1000)

            mortar_rate = self.MORTAR_CONSUMPTION.get(thickness_mm, 0.25)
            mortar_volume = volume_cum * mortar_rate

            boq_items.append(WallBOQItem(
                item_code="MTR-CM-01",
                description=f"Cement mortar for {thickness_mm}mm brickwork",
                qty=round(mortar_volume, 2),
                unit="cum",
                derived_from="calculation",
                confidence=0.65,
                thickness_mm=thickness_mm,
            ))

        return WallBOQResult(
            segments=segments,
            thickness_clusters=thickness_totals,
            boq_items=boq_items,
            total_wall_length_m=round(total_wall_length, 2),
            total_brick_volume_cum=round(total_brick_volume, 2),
            total_plaster_area_sqm=round(total_plaster_area, 2),
            assumptions_used=self.assumptions_used,
        )

    def calculate_from_wall_mask(
        self,
        wall_mask: np.ndarray,
        openings: Optional[List[Dict]] = None,
        scale_px_per_mm: Optional[float] = None,
    ) -> WallBOQResult:
        """
        Complete wall BOQ calculation from wall mask.

        Combines thickness clustering, segment analysis, and BOQ calculation.
        """
        if scale_px_per_mm:
            self.scale_px_per_mm = scale_px_per_mm

        # Cluster thicknesses
        thickness_clusters = self.cluster_wall_thicknesses(wall_mask)

        # Analyze segments
        segments = self.analyze_wall_segments(wall_mask, thickness_clusters)

        # Calculate BOQ
        result = self.calculate_wall_boq(segments, openings)

        return result


def calculate_wall_running_meters(
    wall_mask: np.ndarray,
    scale_px_per_mm: float,
) -> Dict[str, float]:
    """
    Calculate total running meters of walls by type.

    Quick utility function for wall length estimation.
    """
    # Skeletonize to get centerline
    skeleton = np.zeros_like(wall_mask)
    temp = wall_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(temp, kernel)
        opened = cv2.dilate(eroded, kernel)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # Count skeleton pixels and convert to meters
    skeleton_length_px = cv2.countNonZero(skeleton)
    length_mm = skeleton_length_px / scale_px_per_mm
    length_m = length_mm / 1000

    return {
        "total_running_m": round(length_m, 2),
        "skeleton_pixels": skeleton_length_px,
    }
