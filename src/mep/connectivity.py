"""
Connectivity Inference

Infers connections between MEP devices and panels/risers:
- Electrical: Device to DB/panel connections with cable length estimation
- Plumbing: Fixture to riser connections with pipe length estimation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import logging

from .device_detector import DetectedDevice
from .device_types import DeviceTypeRegistry, load_device_types

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """A connection between devices."""
    id: str
    source_device_id: str
    target_device_id: str

    # Connection type
    connection_type: str  # electrical, plumbing, hvac
    medium: str  # cable, pipe, duct

    # Specs
    medium_spec: str  # "2.5 sq mm FR", "20mm CPVC"
    conduit_spec: str = ""  # For electrical
    length_m: float = 0.0

    # Path
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    routing_method: str = "wall_graph"  # wall_graph, direct, manhattan

    # Provenance
    is_inferred: bool = True
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_device_id": self.source_device_id,
            "target_device_id": self.target_device_id,
            "connection_type": self.connection_type,
            "medium": self.medium,
            "medium_spec": self.medium_spec,
            "conduit_spec": self.conduit_spec,
            "length_m": self.length_m,
            "path_points": self.path_points,
            "routing_method": self.routing_method,
            "is_inferred": self.is_inferred,
            "confidence": self.confidence,
        }


class WallGraph:
    """
    Graph representation of walls for cable/pipe routing.
    """

    def __init__(self, rooms: List[Dict[str, Any]]):
        self.rooms = rooms
        self.nodes: Dict[str, Tuple[float, float]] = {}  # node_id -> (x, y)
        self.edges: Dict[str, List[str]] = {}  # node_id -> connected node_ids
        self._build_graph()

    def _build_graph(self):
        """Build wall graph from room boundaries."""
        # Extract wall corners as nodes
        for i, room in enumerate(self.rooms):
            bbox = room.get("bbox", [])
            if len(bbox) < 4:
                continue

            # Add corner nodes
            corners = [
                (bbox[0], bbox[1]),  # Top-left
                (bbox[2], bbox[1]),  # Top-right
                (bbox[2], bbox[3]),  # Bottom-right
                (bbox[0], bbox[3]),  # Bottom-left
            ]

            for j, corner in enumerate(corners):
                node_id = f"r{i}_c{j}"
                self.nodes[node_id] = corner

                # Connect to adjacent corners (walls)
                next_j = (j + 1) % 4
                next_node_id = f"r{i}_c{next_j}"

                if node_id not in self.edges:
                    self.edges[node_id] = []
                self.edges[node_id].append(next_node_id)

        # Add doorway connections between rooms
        self._add_doorway_connections()

    def _add_doorway_connections(self):
        """Add connections through doorways between adjacent rooms."""
        # Simplified: connect rooms that share a wall segment
        for i, room1 in enumerate(self.rooms):
            bbox1 = room1.get("bbox", [])
            if len(bbox1) < 4:
                continue

            for j, room2 in enumerate(self.rooms):
                if i >= j:
                    continue

                bbox2 = room2.get("bbox", [])
                if len(bbox2) < 4:
                    continue

                # Check for shared wall
                if self._shares_wall(bbox1, bbox2):
                    # Connect midpoints of shared wall
                    shared_start, shared_end = self._get_shared_segment(bbox1, bbox2)
                    if shared_start and shared_end:
                        mid_x = (shared_start[0] + shared_end[0]) / 2
                        mid_y = (shared_start[1] + shared_end[1]) / 2

                        node_id = f"door_{i}_{j}"
                        self.nodes[node_id] = (mid_x, mid_y)

                        # Connect to nearest nodes in both rooms
                        self._connect_to_nearest(node_id, f"r{i}")
                        self._connect_to_nearest(node_id, f"r{j}")

    def _shares_wall(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two rooms share a wall."""
        tolerance = 5  # pixels

        # Check vertical wall (left/right)
        if abs(bbox1[2] - bbox2[0]) < tolerance or abs(bbox1[0] - bbox2[2]) < tolerance:
            # Check vertical overlap
            if bbox1[1] < bbox2[3] and bbox2[1] < bbox1[3]:
                return True

        # Check horizontal wall (top/bottom)
        if abs(bbox1[3] - bbox2[1]) < tolerance or abs(bbox1[1] - bbox2[3]) < tolerance:
            # Check horizontal overlap
            if bbox1[0] < bbox2[2] and bbox2[0] < bbox1[2]:
                return True

        return False

    def _get_shared_segment(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Get the shared wall segment between two rooms."""
        tolerance = 5

        # Check each wall of bbox1 against bbox2
        if abs(bbox1[2] - bbox2[0]) < tolerance:
            # Right wall of bbox1, left wall of bbox2
            y_start = max(bbox1[1], bbox2[1])
            y_end = min(bbox1[3], bbox2[3])
            return (bbox1[2], y_start), (bbox1[2], y_end)

        if abs(bbox1[0] - bbox2[2]) < tolerance:
            # Left wall of bbox1, right wall of bbox2
            y_start = max(bbox1[1], bbox2[1])
            y_end = min(bbox1[3], bbox2[3])
            return (bbox1[0], y_start), (bbox1[0], y_end)

        if abs(bbox1[3] - bbox2[1]) < tolerance:
            # Bottom wall of bbox1, top wall of bbox2
            x_start = max(bbox1[0], bbox2[0])
            x_end = min(bbox1[2], bbox2[2])
            return (x_start, bbox1[3]), (x_end, bbox1[3])

        if abs(bbox1[1] - bbox2[3]) < tolerance:
            # Top wall of bbox1, bottom wall of bbox2
            x_start = max(bbox1[0], bbox2[0])
            x_end = min(bbox1[2], bbox2[2])
            return (x_start, bbox1[1]), (x_end, bbox1[1])

        return None, None

    def _connect_to_nearest(self, node_id: str, room_prefix: str):
        """Connect a node to the nearest corner of a room."""
        if node_id not in self.nodes:
            return

        pos = self.nodes[node_id]
        min_dist = float("inf")
        nearest = None

        for nid, npos in self.nodes.items():
            if nid.startswith(room_prefix) and nid != node_id:
                dist = math.sqrt((pos[0] - npos[0])**2 + (pos[1] - npos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = nid

        if nearest:
            if node_id not in self.edges:
                self.edges[node_id] = []
            self.edges[node_id].append(nearest)
            if nearest not in self.edges:
                self.edges[nearest] = []
            self.edges[nearest].append(node_id)

    def find_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Find path along walls from start to end.

        Returns:
            (path_points, total_length)
        """
        # Find nearest nodes to start and end
        start_node = self._find_nearest_node(start)
        end_node = self._find_nearest_node(end)

        if not start_node or not end_node:
            # Fall back to direct path
            length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            return [start, end], length

        # Simple BFS for path finding
        path = self._bfs_path(start_node, end_node)

        if not path:
            # Fall back to direct
            length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            return [start, end], length

        # Convert node path to points
        points = [start]
        for node_id in path:
            points.append(self.nodes[node_id])
        points.append(end)

        # Calculate total length
        total_length = 0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            total_length += math.sqrt(dx*dx + dy*dy)

        return points, total_length

    def _find_nearest_node(self, point: Tuple[float, float]) -> Optional[str]:
        """Find nearest graph node to a point."""
        min_dist = float("inf")
        nearest = None

        for node_id, node_pos in self.nodes.items():
            dist = math.sqrt((point[0] - node_pos[0])**2 + (point[1] - node_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = node_id

        return nearest

    def _bfs_path(self, start: str, end: str) -> List[str]:
        """BFS to find path between nodes."""
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            node, path = queue.pop(0)

            for neighbor in self.edges.get(node, []):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []


class ConnectivityInference:
    """
    Infers connections between MEP devices.
    """

    def __init__(
        self,
        devices: List[DetectedDevice],
        rooms: List[Dict[str, Any]] = None,
        scale: float = 100,
        registry: DeviceTypeRegistry = None
    ):
        self.devices = devices
        self.rooms = rooms or []
        self.scale = scale  # Drawing scale (1:scale)
        self.registry = registry or load_device_types()
        self.connections: List[Connection] = []

        # Build wall graph if rooms available
        self.wall_graph = WallGraph(self.rooms) if self.rooms else None

        # Index devices by type
        self.panels: List[DetectedDevice] = []
        self.devices_by_system: Dict[str, List[DetectedDevice]] = {}

        self._index_devices()

    def _index_devices(self):
        """Index devices by type and system."""
        for device in self.devices:
            # Find panels
            device_type = self.registry.get(device.device_type)
            if device_type and device_type.is_panel:
                self.panels.append(device)

            # Index by system
            system = device.system
            if system:
                if system not in self.devices_by_system:
                    self.devices_by_system[system] = []
                self.devices_by_system[system].append(device)

    def infer_all_connections(self) -> List[Connection]:
        """Infer all connections."""
        connections = []

        # Infer electrical connections
        connections.extend(self._infer_electrical_connections())

        # Infer plumbing connections
        connections.extend(self._infer_plumbing_connections())

        self.connections = connections
        return connections

    def _infer_electrical_connections(self) -> List[Connection]:
        """Infer electrical device to panel connections."""
        connections = []

        # Get connectivity rules
        rules = self.registry.get_connectivity_rules("electrical")
        default_cables = rules.get("default_cable", {})
        conduit_rules = rules.get("conduit", {})
        add_percentage = rules.get("routing", {}).get("add_percentage", 15)

        # Find electrical panels
        electrical_panels = [p for p in self.panels if p.category == "electrical"]

        if not electrical_panels:
            logger.warning("No electrical panels found for connectivity inference")
            return connections

        # Connect each electrical device to nearest panel
        for device in self.devices:
            if device.category != "electrical":
                continue

            # Skip panels themselves
            device_type = self.registry.get(device.device_type)
            if device_type and device_type.is_panel:
                continue

            # Find nearest panel
            nearest_panel = self._find_nearest_device(device, electrical_panels)
            if not nearest_panel:
                continue

            # Determine cable spec based on subsystem
            subsystem = device.subsystem
            cable_spec = default_cables.get(subsystem, default_cables.get("power", "2.5 sq mm FR"))
            conduit_spec = conduit_rules.get("default", "20mm PVC")

            # Calculate path and length
            if self.wall_graph:
                path, length_px = self.wall_graph.find_path(
                    device.centroid,
                    nearest_panel.centroid
                )
            else:
                # Manhattan distance
                dx = abs(device.centroid[0] - nearest_panel.centroid[0])
                dy = abs(device.centroid[1] - nearest_panel.centroid[1])
                length_px = dx + dy
                path = [device.centroid, nearest_panel.centroid]

            # Convert to meters using scale
            length_m = (length_px / self.scale) * (1 + add_percentage / 100)

            connection = Connection(
                id=f"conn_{device.id}_{nearest_panel.id}",
                source_device_id=device.id,
                target_device_id=nearest_panel.id,
                connection_type="electrical",
                medium="cable",
                medium_spec=cable_spec,
                conduit_spec=conduit_spec,
                length_m=round(length_m, 1),
                path_points=path,
                routing_method="wall_graph" if self.wall_graph else "manhattan",
                is_inferred=True,
                confidence=0.6,
            )

            connections.append(connection)

            # Update device with connection info
            device.connected_to = nearest_panel.id
            device.connection_length = round(length_m, 1)

        return connections

    def _infer_plumbing_connections(self) -> List[Connection]:
        """Infer plumbing fixture connections (simplified)."""
        connections = []

        rules = self.registry.get_connectivity_rules("plumbing")
        pipe_sizes = rules.get("pipe_sizes", {})
        add_percentage = rules.get("routing", {}).get("add_percentage", 10)

        # Group plumbing fixtures by room
        plumbing_devices = [d for d in self.devices if d.category == "plumbing"]

        for device in plumbing_devices:
            # Get pipe size based on fixture type
            pipe_size = pipe_sizes.get(device.device_type, "40mm")

            # Estimate pipe run to nearest wall (simplified)
            # In reality, would trace to stack/riser
            estimated_length = 2.0  # Default 2m run

            if device.room_name:
                # Adjust based on room size if known
                estimated_length = 3.0

            connection = Connection(
                id=f"pipe_{device.id}",
                source_device_id=device.id,
                target_device_id="stack",  # Assumed stack
                connection_type="plumbing",
                medium="pipe",
                medium_spec=f"{pipe_size} uPVC",
                length_m=round(estimated_length * (1 + add_percentage / 100), 1),
                routing_method="estimated",
                is_inferred=True,
                confidence=0.4,
            )

            connections.append(connection)

        return connections

    def _find_nearest_device(
        self,
        device: DetectedDevice,
        candidates: List[DetectedDevice]
    ) -> Optional[DetectedDevice]:
        """Find nearest device from candidates."""
        if not candidates:
            return None

        min_dist = float("inf")
        nearest = None

        for candidate in candidates:
            dx = device.centroid[0] - candidate.centroid[0]
            dy = device.centroid[1] - candidate.centroid[1]
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < min_dist:
                min_dist = dist
                nearest = candidate

        return nearest

    def get_cable_summary(self) -> Dict[str, Any]:
        """Get summary of cable/conduit requirements."""
        by_spec = {}

        for conn in self.connections:
            if conn.connection_type != "electrical":
                continue

            spec = conn.medium_spec
            if spec not in by_spec:
                by_spec[spec] = {"count": 0, "total_length_m": 0}

            by_spec[spec]["count"] += 1
            by_spec[spec]["total_length_m"] += conn.length_m

        # Round totals
        for spec in by_spec:
            by_spec[spec]["total_length_m"] = round(by_spec[spec]["total_length_m"], 1)

        return {
            "cables": by_spec,
            "total_connections": len([c for c in self.connections if c.connection_type == "electrical"]),
        }

    def get_pipe_summary(self) -> Dict[str, Any]:
        """Get summary of pipe requirements."""
        by_size = {}

        for conn in self.connections:
            if conn.connection_type != "plumbing":
                continue

            spec = conn.medium_spec
            if spec not in by_size:
                by_size[spec] = {"count": 0, "total_length_m": 0}

            by_size[spec]["count"] += 1
            by_size[spec]["total_length_m"] += conn.length_m

        for size in by_size:
            by_size[size]["total_length_m"] = round(by_size[size]["total_length_m"], 1)

        return {
            "pipes": by_size,
            "total_connections": len([c for c in self.connections if c.connection_type == "plumbing"]),
        }


def infer_connections(
    devices: List[DetectedDevice],
    rooms: List[Dict[str, Any]] = None,
    scale: float = 100,
) -> Tuple[List[Connection], Dict[str, Any]]:
    """
    Convenience function to infer all connections.

    Returns:
        (list of connections, summary dict)
    """
    inference = ConnectivityInference(devices, rooms, scale)
    connections = inference.infer_all_connections()

    summary = {
        "total_connections": len(connections),
        "cables": inference.get_cable_summary(),
        "pipes": inference.get_pipe_summary(),
    }

    return connections, summary
