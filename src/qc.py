"""
Floor Plan Quality Control Module
Performs validation and generates warnings.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

from .area import RoomWithArea
from .scale import ScaleResult
from .polygons import RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class QCWarning:
    """A quality control warning."""
    code: str
    severity: str  # "error", "warning", "info"
    message: str
    room_id: Optional[str] = None
    details: Dict = field(default_factory=dict)


@dataclass
class QCReport:
    """Quality control report for a plan."""
    plan_id: str
    is_valid: bool
    overall_confidence: float
    warnings: List[QCWarning]
    statistics: Dict
    suggestions: List[str]


class QualityChecker:
    """
    Performs quality checks on extracted floor plan data.
    """

    # Typical room size ranges (sqm)
    ROOM_SIZE_RANGES = {
        'Bedroom': (9, 50),
        'Living': (12, 100),
        'Dining': (8, 40),
        'Kitchen': (5, 30),
        'Toilet': (2, 15),
        'Balcony': (2, 30),
        'Terrace': (5, 200),
        'Passage': (2, 30),
        'Store': (2, 20),
        'Study': (6, 30),
        'Pooja': (2, 15),
        'Dressing': (3, 20),
        'Staircase': (3, 30),
        'Lift': (2, 20),
        'Duct': (0.5, 10),
        'Utility': (3, 20),
        'Wash': (2, 10),
        'Parking': (10, 100),
    }

    # Minimum expected rooms for a residential unit
    MIN_EXPECTED_ROOMS = {
        'Toilet': 1,
        'Kitchen': 1,
    }

    def __init__(self):
        pass

    def check(
        self,
        plan_id: str,
        rooms: List[RoomWithArea],
        scale: ScaleResult,
        wall_mask: Optional[np.ndarray] = None,
        external_boundary: Optional[np.ndarray] = None
    ) -> QCReport:
        """
        Run quality checks on extracted data.

        Args:
            plan_id: Plan identifier
            rooms: List of rooms with areas
            scale: Scale result
            wall_mask: Wall mask for boundary checks
            external_boundary: External boundary contour

        Returns:
            QCReport
        """
        warnings = []
        statistics = {}
        suggestions = []

        # Scale checks
        scale_warnings = self._check_scale(scale)
        warnings.extend(scale_warnings)

        # Room count checks
        count_warnings, count_suggestions = self._check_room_counts(rooms)
        warnings.extend(count_warnings)
        suggestions.extend(count_suggestions)

        # Room size checks
        size_warnings = self._check_room_sizes(rooms)
        warnings.extend(size_warnings)

        # Overlap checks
        overlap_warnings = self._check_overlaps(rooms)
        warnings.extend(overlap_warnings)

        # Boundary checks
        if wall_mask is not None:
            boundary_warnings = self._check_boundaries(rooms, wall_mask, external_boundary)
            warnings.extend(boundary_warnings)

        # Label checks
        label_warnings = self._check_labels(rooms)
        warnings.extend(label_warnings)

        # Calculate statistics
        statistics = self._compute_statistics(rooms, scale)

        # Calculate overall confidence
        error_count = len([w for w in warnings if w.severity == 'error'])
        warning_count = len([w for w in warnings if w.severity == 'warning'])

        base_confidence = scale.confidence
        penalty = error_count * 0.15 + warning_count * 0.05
        overall_confidence = max(0.1, base_confidence - penalty)

        # Determine validity
        is_valid = error_count == 0 and len(rooms) > 0

        return QCReport(
            plan_id=plan_id,
            is_valid=is_valid,
            overall_confidence=round(overall_confidence, 2),
            warnings=warnings,
            statistics=statistics,
            suggestions=suggestions
        )

    def _check_scale(self, scale: ScaleResult) -> List[QCWarning]:
        """Check scale quality."""
        warnings = []

        if scale.confidence < 0.5:
            warnings.append(QCWarning(
                code="scale_uncertain",
                severity="error",
                message=f"Scale confidence is very low ({scale.confidence:.0%})",
                details={'confidence': scale.confidence, 'method': scale.method.value}
            ))
        elif scale.confidence < 0.7:
            warnings.append(QCWarning(
                code="scale_uncertain",
                severity="warning",
                message=f"Scale confidence is low ({scale.confidence:.0%})",
                details={'confidence': scale.confidence, 'method': scale.method.value}
            ))

        if scale.method.value == "default":
            warnings.append(QCWarning(
                code="scale_not_detected",
                severity="warning",
                message="Scale was not auto-detected, using default 1:100"
            ))

        return warnings

    def _check_room_counts(
        self,
        rooms: List[RoomWithArea]
    ) -> Tuple[List[QCWarning], List[str]]:
        """Check room counts."""
        warnings = []
        suggestions = []

        if not rooms:
            warnings.append(QCWarning(
                code="no_rooms",
                severity="error",
                message="No rooms detected in floor plan"
            ))
            return warnings, suggestions

        # Count rooms by type
        room_counts = {}
        for room in rooms:
            base_type = room.label.rsplit(' ', 1)[0] if room.label[-1].isdigit() else room.label
            room_counts[base_type] = room_counts.get(base_type, 0) + 1

        # Check minimum expected
        for room_type, min_count in self.MIN_EXPECTED_ROOMS.items():
            if room_counts.get(room_type, 0) < min_count:
                warnings.append(QCWarning(
                    code="missing_room_type",
                    severity="warning",
                    message=f"Expected at least {min_count} {room_type}(s), found {room_counts.get(room_type, 0)}",
                    details={'room_type': room_type, 'found': room_counts.get(room_type, 0)}
                ))
                suggestions.append(f"Check if {room_type} was labeled correctly or merged with another room")

        # Check for unlabeled rooms
        unlabeled = [r for r in rooms if r.label in ('Room', 'Unknown')]
        if unlabeled:
            warnings.append(QCWarning(
                code="unlabeled_rooms",
                severity="warning",
                message=f"{len(unlabeled)} room(s) could not be labeled",
                details={'count': len(unlabeled), 'room_ids': [r.room_id for r in unlabeled]}
            ))

        return warnings, suggestions

    def _check_room_sizes(self, rooms: List[RoomWithArea]) -> List[QCWarning]:
        """Check room sizes against typical ranges."""
        warnings = []

        for room in rooms:
            base_type = room.label.rsplit(' ', 1)[0] if room.label[-1].isdigit() else room.label

            if base_type in self.ROOM_SIZE_RANGES:
                min_size, max_size = self.ROOM_SIZE_RANGES[base_type]

                if room.area.area_sqm < min_size * 0.5:
                    warnings.append(QCWarning(
                        code="room_too_small",
                        severity="warning",
                        message=f"{room.label} ({room.area.area_sqm:.1f} sqm) is unusually small for a {base_type}",
                        room_id=room.room_id,
                        details={'area_sqm': room.area.area_sqm, 'expected_min': min_size}
                    ))
                elif room.area.area_sqm > max_size * 1.5:
                    warnings.append(QCWarning(
                        code="room_too_large",
                        severity="info",
                        message=f"{room.label} ({room.area.area_sqm:.1f} sqm) is unusually large for a {base_type}",
                        room_id=room.room_id,
                        details={'area_sqm': room.area.area_sqm, 'expected_max': max_size}
                    ))

            # General sanity checks
            if room.area.area_sqm < 0.5:
                warnings.append(QCWarning(
                    code="tiny_region_discarded",
                    severity="warning",
                    message=f"{room.label} is extremely small ({room.area.area_sqm:.2f} sqm)",
                    room_id=room.room_id
                ))

        return warnings

    def _check_overlaps(self, rooms: List[RoomWithArea]) -> List[QCWarning]:
        """Check for overlapping rooms."""
        warnings = []

        # Check bbox overlaps (fast pre-filter)
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms):
                if j <= i:
                    continue

                # Bbox overlap check
                b1 = room1.polygon.bbox
                b2 = room2.polygon.bbox

                # Check if bboxes overlap
                if not (b1[0] + b1[2] < b2[0] or b2[0] + b2[2] < b1[0] or
                        b1[1] + b1[3] < b2[1] or b2[1] + b2[3] < b1[1]):

                    # Detailed polygon overlap check
                    overlap_area = self._compute_overlap_area(
                        room1.polygon, room2.polygon
                    )

                    # Significant overlap if > 10% of smaller room
                    min_area = min(room1.area.area_pixels, room2.area.area_pixels)
                    if overlap_area > min_area * 0.1:
                        warnings.append(QCWarning(
                            code="overlapping_rooms",
                            severity="warning",
                            message=f"{room1.label} and {room2.label} appear to overlap",
                            room_id=room1.room_id,
                            details={'other_room': room2.room_id, 'overlap_ratio': overlap_area / min_area}
                        ))

        return warnings

    def _compute_overlap_area(
        self,
        poly1: RoomPolygon,
        poly2: RoomPolygon
    ) -> float:
        """Compute overlap area between two polygons."""
        try:
            from shapely.geometry import Polygon

            if len(poly1.points) < 3 or len(poly2.points) < 3:
                return 0

            p1 = Polygon(poly1.points)
            p2 = Polygon(poly2.points)

            if not p1.is_valid or not p2.is_valid:
                return 0

            intersection = p1.intersection(p2)
            return intersection.area

        except ImportError:
            # Fallback: use mask intersection
            if poly1.bbox and poly2.bbox:
                # Create small masks around the intersection region
                x1 = min(poly1.bbox[0], poly2.bbox[0])
                y1 = min(poly1.bbox[1], poly2.bbox[1])
                x2 = max(poly1.bbox[0] + poly1.bbox[2], poly2.bbox[0] + poly2.bbox[2])
                y2 = max(poly1.bbox[1] + poly1.bbox[3], poly2.bbox[1] + poly2.bbox[3])

                w, h = int(x2 - x1), int(y2 - y1)
                if w > 0 and h > 0 and w < 5000 and h < 5000:
                    mask1 = np.zeros((h, w), dtype=np.uint8)
                    mask2 = np.zeros((h, w), dtype=np.uint8)

                    pts1 = np.array([(p[0] - x1, p[1] - y1) for p in poly1.points], dtype=np.int32)
                    pts2 = np.array([(p[0] - x1, p[1] - y1) for p in poly2.points], dtype=np.int32)

                    cv2.fillPoly(mask1, [pts1], 255)
                    cv2.fillPoly(mask2, [pts2], 255)

                    intersection = cv2.bitwise_and(mask1, mask2)
                    return np.sum(intersection > 0)

            return 0

    def _check_boundaries(
        self,
        rooms: List[RoomWithArea],
        wall_mask: np.ndarray,
        external_boundary: Optional[np.ndarray]
    ) -> List[QCWarning]:
        """Check if rooms have proper boundaries."""
        warnings = []

        # Create mask of all room areas
        h, w = wall_mask.shape
        rooms_mask = np.zeros((h, w), dtype=np.uint8)

        for room in rooms:
            pts = np.array(room.polygon.points, dtype=np.int32)
            cv2.fillPoly(rooms_mask, [pts], 255)

        # Check for rooms touching external area
        if external_boundary is not None:
            external_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(external_mask, [external_boundary], -1, 255, -1)
            external_mask = cv2.bitwise_not(external_mask)  # Invert - outside is white

            # Dilate rooms slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            rooms_dilated = cv2.dilate(rooms_mask, kernel)

            # Check overlap with external
            overlap = cv2.bitwise_and(rooms_dilated, external_mask)
            overlap_pixels = np.sum(overlap > 0)

            if overlap_pixels > 100:
                warnings.append(QCWarning(
                    code="open_boundaries",
                    severity="warning",
                    message="Some rooms may have incomplete boundaries (touching outside)",
                    details={'overlap_pixels': int(overlap_pixels)}
                ))

        return warnings

    def _check_labels(self, rooms: List[RoomWithArea]) -> List[QCWarning]:
        """Check label quality."""
        warnings = []

        # Check for duplicate labels
        labels = [r.label for r in rooms]
        seen = set()
        duplicates = set()

        for label in labels:
            if label in seen:
                duplicates.add(label)
            seen.add(label)

        if duplicates:
            warnings.append(QCWarning(
                code="duplicate_labels",
                severity="info",
                message=f"Some room labels are duplicated: {', '.join(duplicates)}",
                details={'duplicates': list(duplicates)}
            ))

        # Check for low confidence labels
        low_conf = [r for r in rooms if r.confidence < 0.5]
        if low_conf:
            warnings.append(QCWarning(
                code="label_ambiguous",
                severity="warning",
                message=f"{len(low_conf)} room(s) have uncertain labels",
                details={'room_ids': [r.room_id for r in low_conf]}
            ))

        return warnings

    def _compute_statistics(
        self,
        rooms: List[RoomWithArea],
        scale: ScaleResult
    ) -> Dict:
        """Compute summary statistics."""
        if not rooms:
            return {'total_rooms': 0}

        total_sqm = sum(r.area.area_sqm for r in rooms)
        total_sqft = sum(r.area.area_sqft for r in rooms)

        # Count by type
        by_type = {}
        for room in rooms:
            base_type = room.label.rsplit(' ', 1)[0] if room.label[-1].isdigit() else room.label
            if base_type not in by_type:
                by_type[base_type] = {'count': 0, 'area_sqm': 0}
            by_type[base_type]['count'] += 1
            by_type[base_type]['area_sqm'] += room.area.area_sqm

        return {
            'total_rooms': len(rooms),
            'total_area_sqm': round(total_sqm, 2),
            'total_area_sqft': round(total_sqft, 2),
            'rooms_by_type': by_type,
            'avg_room_area_sqm': round(total_sqm / len(rooms), 2),
            'scale_method': scale.method.value,
            'scale_confidence': scale.confidence,
            'labeled_count': len([r for r in rooms if r.label not in ('Room', 'Unknown')]),
            'unlabeled_count': len([r for r in rooms if r.label in ('Room', 'Unknown')]),
        }


def run_quality_check(
    plan_id: str,
    rooms: List[RoomWithArea],
    scale: ScaleResult,
    wall_mask: Optional[np.ndarray] = None,
    external_boundary: Optional[np.ndarray] = None
) -> QCReport:
    """
    Run quality checks.

    Args:
        plan_id: Plan identifier
        rooms: Rooms with areas
        scale: Scale result
        wall_mask: Optional wall mask
        external_boundary: Optional boundary

    Returns:
        QCReport
    """
    checker = QualityChecker()
    return checker.check(plan_id, rooms, scale, wall_mask, external_boundary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Quality checker module - run with actual data")
