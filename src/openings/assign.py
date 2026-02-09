"""
Room Assignment Module
Assigns detected openings to rooms based on spatial relationships.

Features:
- Doors connect two rooms (room_left, room_right)
- Windows connect room to exterior
- Uses room polygons for containment checks
- Handles wall boundary cases
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import math

try:
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import nearest_points
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Shapely not available - using fallback geometry")

logger = logging.getLogger(__name__)


@dataclass
class OpeningAssignment:
    """Assignment of an opening to room(s)."""
    opening_id: str
    opening_type: str
    room_left_id: Optional[str] = None
    room_left_label: Optional[str] = None
    room_right_id: Optional[str] = None
    room_right_label: Optional[str] = None
    wall_segment_id: Optional[str] = None
    is_external: bool = False  # Opens to exterior
    confidence: float = 0.0


@dataclass
class RoomPolygon:
    """Room polygon for spatial queries."""
    room_id: str
    label: str
    vertices: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    area: float
    _shapely_poly: Any = field(default=None, repr=False)

    def __post_init__(self):
        if SHAPELY_AVAILABLE and self.vertices and len(self.vertices) >= 3:
            try:
                self._shapely_poly = Polygon(self.vertices)
            except Exception:
                pass

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside polygon."""
        if self._shapely_poly:
            return self._shapely_poly.contains(Point(point))
        return self._point_in_polygon_fallback(point)

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate distance from polygon boundary to point."""
        if self._shapely_poly:
            return self._shapely_poly.exterior.distance(Point(point))
        return self._distance_fallback(point)

    def _point_in_polygon_fallback(self, point: Tuple[float, float]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(self.vertices)
        inside = False

        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _distance_fallback(self, point: Tuple[float, float]) -> float:
        """Simple distance to nearest vertex."""
        min_dist = float("inf")
        for vx, vy in self.vertices:
            dist = math.sqrt((vx - point[0]) ** 2 + (vy - point[1]) ** 2)
            min_dist = min(min_dist, dist)
        return min_dist


class RoomAssigner:
    """
    Assigns openings to rooms based on spatial relationships.

    Logic:
    - Doors: Find two rooms sharing the wall where door is located
    - Windows: Find room on interior side, exterior on other side
    - Ventilators: Same as windows
    """

    def __init__(
        self,
        proximity_threshold_px: float = 50.0,
        external_boundary: Optional[np.ndarray] = None,
    ):
        """
        Initialize room assigner.

        Args:
            proximity_threshold_px: Max distance to associate opening with room
            external_boundary: Mask of external/outside area (255 = external)
        """
        self.proximity_threshold = proximity_threshold_px
        self.external_boundary = external_boundary
        self.rooms: List[RoomPolygon] = []

    def load_rooms(
        self,
        rooms: List[Dict],
    ) -> None:
        """
        Load room data for assignment.

        Args:
            rooms: List of room dicts with id, label, vertices, centroid, area
        """
        self.rooms = []

        for room_data in rooms:
            room = RoomPolygon(
                room_id=room_data.get("id", room_data.get("room_id", "")),
                label=room_data.get("label", "Unknown"),
                vertices=room_data.get("vertices", room_data.get("polygon_vertices", [])),
                centroid=room_data.get("centroid", (0, 0)),
                area=room_data.get("area", room_data.get("area_sqm", 0)),
            )
            self.rooms.append(room)

        logger.info(f"Loaded {len(self.rooms)} rooms for assignment")

    def assign_opening(
        self,
        opening: Any,
    ) -> OpeningAssignment:
        """
        Assign a single opening to room(s).

        Args:
            opening: DetectedDoor or DetectedWindow object

        Returns:
            OpeningAssignment with room associations
        """
        opening_id = opening.id
        opening_type = getattr(opening, "type", "unknown")
        center = opening.center
        bbox = opening.bbox

        # Find nearby rooms
        nearby = self._find_nearby_rooms(center, bbox)

        if not nearby:
            return OpeningAssignment(
                opening_id=opening_id,
                opening_type=opening_type,
                is_external=True,
                confidence=0.3,
            )

        # Check if on external boundary
        is_external = self._is_on_external_boundary(center)

        # Doors typically connect two rooms
        if "door" in opening_type.lower():
            return self._assign_door(opening_id, opening_type, center, bbox, nearby, is_external)
        else:
            # Windows/ventilators connect room to exterior
            return self._assign_window(opening_id, opening_type, center, nearby, is_external)

    def _find_nearby_rooms(
        self,
        center: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> List[Tuple[RoomPolygon, float]]:
        """Find rooms near the opening."""
        nearby = []

        for room in self.rooms:
            # Check if opening center or bbox overlaps/near room
            dist = room.distance_to_point(center)

            if dist <= self.proximity_threshold:
                nearby.append((room, dist))
            elif room.contains_point(center):
                nearby.append((room, 0.0))

        # Sort by distance
        nearby.sort(key=lambda x: x[1])

        return nearby

    def _is_on_external_boundary(
        self,
        center: Tuple[int, int],
    ) -> bool:
        """Check if opening is on external building boundary."""
        if self.external_boundary is None:
            return False

        x, y = int(center[0]), int(center[1])

        # Check in small neighborhood
        h, w = self.external_boundary.shape[:2]
        search_radius = 20

        for dy in range(-search_radius, search_radius + 1, 5):
            for dx in range(-search_radius, search_radius + 1, 5):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if self.external_boundary[ny, nx] > 0:
                        return True

        return False

    def _assign_door(
        self,
        opening_id: str,
        opening_type: str,
        center: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        nearby: List[Tuple[RoomPolygon, float]],
        is_external: bool,
    ) -> OpeningAssignment:
        """Assign door to two rooms (or one room + exterior)."""

        if len(nearby) >= 2:
            # Door connects two rooms
            room1, dist1 = nearby[0]
            room2, dist2 = nearby[1]

            # Determine left/right based on position
            # (Convention: left is the room with lower x centroid)
            if room1.centroid[0] <= room2.centroid[0]:
                left_room, right_room = room1, room2
            else:
                left_room, right_room = room2, room1

            return OpeningAssignment(
                opening_id=opening_id,
                opening_type=opening_type,
                room_left_id=left_room.room_id,
                room_left_label=left_room.label,
                room_right_id=right_room.room_id,
                room_right_label=right_room.label,
                is_external=False,
                confidence=0.8,
            )

        elif len(nearby) == 1:
            # Door connects one room to exterior (or unknown)
            room, dist = nearby[0]

            return OpeningAssignment(
                opening_id=opening_id,
                opening_type=opening_type,
                room_left_id=room.room_id,
                room_left_label=room.label,
                room_right_id=None,
                room_right_label="Exterior" if is_external else "Unknown",
                is_external=is_external,
                confidence=0.6,
            )

        else:
            return OpeningAssignment(
                opening_id=opening_id,
                opening_type=opening_type,
                is_external=True,
                confidence=0.3,
            )

    def _assign_window(
        self,
        opening_id: str,
        opening_type: str,
        center: Tuple[int, int],
        nearby: List[Tuple[RoomPolygon, float]],
        is_external: bool,
    ) -> OpeningAssignment:
        """Assign window to one room (interior side)."""

        if nearby:
            room, dist = nearby[0]

            return OpeningAssignment(
                opening_id=opening_id,
                opening_type=opening_type,
                room_left_id=room.room_id,
                room_left_label=room.label,
                room_right_id=None,
                room_right_label="Exterior",
                is_external=True,  # Windows always open to exterior
                confidence=0.7,
            )

        return OpeningAssignment(
            opening_id=opening_id,
            opening_type=opening_type,
            is_external=True,
            confidence=0.3,
        )

    def assign_all(
        self,
        openings: List[Any],
    ) -> List[OpeningAssignment]:
        """
        Assign all openings to rooms.

        Args:
            openings: List of DetectedDoor/DetectedWindow objects

        Returns:
            List of OpeningAssignment objects
        """
        assignments = []

        for opening in openings:
            assignment = self.assign_opening(opening)
            assignments.append(assignment)

            # Also update the opening object if it has these fields
            if hasattr(opening, "room_left_id"):
                opening.room_left_id = assignment.room_left_id
            if hasattr(opening, "room_right_id"):
                opening.room_right_id = assignment.room_right_id
            if hasattr(opening, "wall_segment_id"):
                opening.wall_segment_id = assignment.wall_segment_id

        return assignments


def create_room_openings_summary(
    assignments: List[OpeningAssignment],
) -> Dict[str, Dict]:
    """
    Create per-room summary of openings.

    Args:
        assignments: List of OpeningAssignment objects

    Returns:
        Dict mapping room_id to opening counts
    """
    summary: Dict[str, Dict] = {}

    for assignment in assignments:
        # Add to left room
        if assignment.room_left_id:
            if assignment.room_left_id not in summary:
                summary[assignment.room_left_id] = {
                    "label": assignment.room_left_label,
                    "doors": [],
                    "windows": [],
                    "ventilators": [],
                }

            if "door" in assignment.opening_type.lower():
                summary[assignment.room_left_id]["doors"].append(assignment.opening_id)
            elif "vent" in assignment.opening_type.lower():
                summary[assignment.room_left_id]["ventilators"].append(assignment.opening_id)
            else:
                summary[assignment.room_left_id]["windows"].append(assignment.opening_id)

        # Add to right room (for doors)
        if assignment.room_right_id and "door" in assignment.opening_type.lower():
            if assignment.room_right_id not in summary:
                summary[assignment.room_right_id] = {
                    "label": assignment.room_right_label,
                    "doors": [],
                    "windows": [],
                    "ventilators": [],
                }

            summary[assignment.room_right_id]["doors"].append(assignment.opening_id)

    return summary
