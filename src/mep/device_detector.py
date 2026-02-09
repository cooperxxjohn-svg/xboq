"""
Device Detector

Detects MEP devices in floor plans using:
- Vector symbol detection
- OCR text detection
- Pattern matching against device types registry
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging

from .device_types import DeviceType, DeviceTypeRegistry, load_device_types

logger = logging.getLogger(__name__)


@dataclass
class DetectedDevice:
    """A detected MEP device."""
    id: str
    device_type: str  # DeviceType.id
    category: str  # electrical, plumbing, hvac, fire_safety

    # Location
    page: int
    bbox: List[float]  # [x1, y1, x2, y2]
    centroid: Tuple[float, float]

    # Room assignment
    room_id: Optional[str] = None
    room_name: Optional[str] = None

    # Detection info
    detection_method: str = "unknown"  # symbol, text, ocr, schedule
    tag: str = ""  # Extracted tag like "L1", "SP-01"
    confidence: float = 0.0

    # Spec (from drawing or default)
    spec: Dict[str, Any] = field(default_factory=dict)
    spec_source: str = "default"  # default, drawing, schedule

    # Provenance
    source_text: str = ""
    source_symbol: str = ""

    # RFIs needed
    rfi_needed: List[str] = field(default_factory=list)

    # System assignment
    system: str = ""
    subsystem: str = ""

    # Connectivity
    connected_to: Optional[str] = None  # Panel/riser ID
    connection_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "device_type": self.device_type,
            "category": self.category,
            "page": self.page,
            "bbox": self.bbox,
            "centroid": list(self.centroid),
            "room_id": self.room_id,
            "room_name": self.room_name,
            "detection_method": self.detection_method,
            "tag": self.tag,
            "confidence": self.confidence,
            "spec": self.spec,
            "spec_source": self.spec_source,
            "source_text": self.source_text,
            "source_symbol": self.source_symbol,
            "rfi_needed": self.rfi_needed,
            "system": self.system,
            "subsystem": self.subsystem,
            "connected_to": self.connected_to,
            "connection_length": self.connection_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectedDevice":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            device_type=data["device_type"],
            category=data["category"],
            page=data.get("page", 0),
            bbox=data.get("bbox", [0, 0, 0, 0]),
            centroid=tuple(data.get("centroid", (0, 0))),
            room_id=data.get("room_id"),
            room_name=data.get("room_name"),
            detection_method=data.get("detection_method", "unknown"),
            tag=data.get("tag", ""),
            confidence=data.get("confidence", 0.0),
            spec=data.get("spec", {}),
            spec_source=data.get("spec_source", "default"),
            source_text=data.get("source_text", ""),
            source_symbol=data.get("source_symbol", ""),
            rfi_needed=data.get("rfi_needed", []),
            system=data.get("system", ""),
            subsystem=data.get("subsystem", ""),
            connected_to=data.get("connected_to"),
            connection_length=data.get("connection_length", 0.0),
        )


class DeviceDetector:
    """
    Detects MEP devices in floor plans.
    """

    def __init__(self, registry: DeviceTypeRegistry = None):
        self.registry = registry or load_device_types()
        self.detected_devices: List[DetectedDevice] = []

        # Detection stats
        self.stats = {
            "text_matches": 0,
            "symbol_matches": 0,
            "schedule_matches": 0,
            "total_detected": 0,
        }

    def detect_from_plan(
        self,
        vector_texts: List[Dict[str, Any]],
        vector_paths: List[Dict[str, Any]] = None,
        page: int = 0,
        rooms: List[Dict[str, Any]] = None,
    ) -> List[DetectedDevice]:
        """
        Detect devices from plan data.

        Args:
            vector_texts: Text elements from PDF/drawing
            vector_paths: Path/shape elements
            page: Page number
            rooms: Room data for assignment

        Returns:
            List of detected devices
        """
        devices = []

        # Detect from text labels
        text_devices = self._detect_from_texts(vector_texts, page)
        devices.extend(text_devices)

        # Detect from symbols/paths
        if vector_paths:
            symbol_devices = self._detect_from_symbols(vector_paths, page)
            devices.extend(symbol_devices)

        # Assign rooms
        if rooms:
            self._assign_rooms(devices, rooms)

        # Deduplicate overlapping detections
        devices = self._deduplicate(devices)

        self.detected_devices.extend(devices)
        self.stats["total_detected"] = len(self.detected_devices)

        return devices

    def _detect_from_texts(
        self,
        texts: List[Dict[str, Any]],
        page: int
    ) -> List[DetectedDevice]:
        """Detect devices from text annotations."""
        devices = []

        for text_item in texts:
            text = text_item.get("text", "")
            if not text or len(text) < 1:
                continue

            # Skip common non-device text
            if self._is_noise_text(text):
                continue

            # Try to match against device types
            matches = self.registry.match_text(text)

            if matches:
                device_type, confidence = matches[0]  # Best match

                # Extract tag if present
                tag = self._extract_tag(text, device_type)

                # Get bbox and centroid
                bbox = text_item.get("bbox", [0, 0, 0, 0])
                if len(bbox) < 4:
                    bbox = [0, 0, 0, 0]
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                device = DetectedDevice(
                    id=f"dev_{page}_{len(devices)}_{uuid.uuid4().hex[:4]}",
                    device_type=device_type.id,
                    category=device_type.category,
                    page=page,
                    bbox=bbox,
                    centroid=centroid,
                    detection_method="text",
                    tag=tag,
                    confidence=confidence,
                    spec=dict(device_type.default_spec),
                    spec_source="default",
                    source_text=text,
                    system=device_type.system,
                    subsystem=device_type.subsystem,
                    rfi_needed=list(device_type.rfi_if_missing),
                )

                devices.append(device)
                self.stats["text_matches"] += 1

        return devices

    def _detect_from_symbols(
        self,
        paths: List[Dict[str, Any]],
        page: int
    ) -> List[DetectedDevice]:
        """Detect devices from symbol shapes."""
        devices = []

        # Group paths by proximity to find symbols
        symbols = self._identify_symbols(paths)

        for symbol in symbols:
            symbol_char = symbol.get("symbol", "")
            bbox = symbol.get("bbox", [0, 0, 0, 0])

            matches = self.registry.match_symbol(symbol_char)

            if matches:
                device_type, confidence = matches[0]
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                device = DetectedDevice(
                    id=f"sym_{page}_{len(devices)}_{uuid.uuid4().hex[:4]}",
                    device_type=device_type.id,
                    category=device_type.category,
                    page=page,
                    bbox=bbox,
                    centroid=centroid,
                    detection_method="symbol",
                    confidence=confidence,
                    spec=dict(device_type.default_spec),
                    spec_source="default",
                    source_symbol=symbol_char,
                    system=device_type.system,
                    subsystem=device_type.subsystem,
                    rfi_needed=list(device_type.rfi_if_missing),
                )

                devices.append(device)
                self.stats["symbol_matches"] += 1

        return devices

    def _identify_symbols(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify symbol shapes from paths."""
        symbols = []

        # Look for small enclosed shapes that could be device symbols
        for path in paths:
            bbox = path.get("bbox", [])
            if len(bbox) < 4:
                continue

            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])

            # Small square-ish shapes are likely symbols
            if 5 < width < 50 and 5 < height < 50:
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if aspect_ratio < 2:  # Roughly square or circular
                    # Try to classify the shape
                    symbol_type = self._classify_shape(path)
                    if symbol_type:
                        symbols.append({
                            "symbol": symbol_type,
                            "bbox": bbox,
                            "path": path,
                        })

        return symbols

    def _classify_shape(self, path: Dict[str, Any]) -> Optional[str]:
        """Classify a path as a symbol type."""
        # Simplified shape classification
        # In production, this would use more sophisticated shape matching

        path_type = path.get("type", "")

        if path_type == "circle":
            return "○"
        elif path_type == "rect":
            return "□"
        elif "fill" in str(path.get("fill", "")):
            return "●"

        return None

    def _is_noise_text(self, text: str) -> bool:
        """Check if text is likely noise (not a device label)."""
        text = text.strip().upper()

        # Skip pure numbers
        if text.isdigit():
            return True

        # Skip very short text
        if len(text) < 2:
            return True

        # Skip common non-device text
        noise_patterns = [
            r'^\d+[\'"″\u2033\u2032]$',  # Dimension text (feet/inches)
            r"^\d+\.\d+$",  # Decimal numbers
            r"^[+-]?\d+$",  # Signed numbers
            r"^(UP|DN|DOWN)$",  # Direction labels
            r"^(SCALE|NOTE|REV|DATE)$",  # Title block
        ]

        for pattern in noise_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False

    def _extract_tag(self, text: str, device_type: DeviceType) -> str:
        """Extract device tag from text."""
        # Look for patterns like "L1", "SP-01", "WC1"
        tag_patterns = [
            r"([A-Z]{1,3}[-_]?\d{1,3})",  # L1, SP-01, DB-1
            r"(\d{1,2}[A-Z]{1,2})",  # 1W, 2W
        ]

        for pattern in tag_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Use the matched text pattern as tag
        return text.strip()[:10]

    def _assign_rooms(
        self,
        devices: List[DetectedDevice],
        rooms: List[Dict[str, Any]]
    ) -> None:
        """Assign devices to rooms based on location."""
        for device in devices:
            best_room = None
            best_overlap = 0

            for room in rooms:
                room_bbox = room.get("bbox", [])
                if len(room_bbox) < 4:
                    continue

                # Check if device is inside room
                if self._point_in_bbox(device.centroid, room_bbox):
                    device.room_id = room.get("id", "")
                    device.room_name = room.get("label", room.get("name", ""))
                    break

                # Check overlap
                overlap = self._bbox_overlap(device.bbox, room_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_room = room

            # Use best overlapping room if no exact match
            if not device.room_id and best_room and best_overlap > 0.5:
                device.room_id = best_room.get("id", "")
                device.room_name = best_room.get("label", best_room.get("name", ""))

    def _point_in_bbox(self, point: Tuple[float, float], bbox: List[float]) -> bool:
        """Check if point is inside bbox."""
        x, y = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    def _bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bboxes."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        if area1 <= 0:
            return 0.0

        return intersection / area1

    def _deduplicate(self, devices: List[DetectedDevice]) -> List[DetectedDevice]:
        """Remove duplicate detections of the same device."""
        unique = []
        seen_locations = set()

        for device in devices:
            # Create location key (rounded centroid)
            loc_key = (
                device.device_type,
                round(device.centroid[0] / 10),
                round(device.centroid[1] / 10),
            )

            if loc_key not in seen_locations:
                seen_locations.add(loc_key)
                unique.append(device)
            else:
                # Keep higher confidence detection
                for i, existing in enumerate(unique):
                    existing_key = (
                        existing.device_type,
                        round(existing.centroid[0] / 10),
                        round(existing.centroid[1] / 10),
                    )
                    if existing_key == loc_key and device.confidence > existing.confidence:
                        unique[i] = device
                        break

        return unique

    def detect_from_schedule(
        self,
        schedule_data: Dict[str, Any],
        page: int = 0
    ) -> List[DetectedDevice]:
        """
        Detect devices from a schedule table.

        Args:
            schedule_data: Parsed schedule table data
            page: Source page number

        Returns:
            List of detected devices
        """
        devices = []

        rows = schedule_data.get("rows", [])
        columns = schedule_data.get("columns", [])
        schedule_type = schedule_data.get("type", "")

        for row in rows:
            # Try to identify device type from row data
            device_text = row.get("description", "") or row.get("type", "")
            matches = self.registry.match_text(device_text)

            if not matches:
                continue

            device_type, confidence = matches[0]

            # Extract quantity
            qty = row.get("qty", 1)
            if isinstance(qty, str):
                try:
                    qty = int(qty)
                except ValueError:
                    qty = 1

            # Extract spec from row
            spec = dict(device_type.default_spec)
            for col in ["size", "wattage", "capacity", "type", "make", "model"]:
                if col in row and row[col]:
                    spec[col] = row[col]

            # Create device(s) based on quantity
            for i in range(qty):
                device = DetectedDevice(
                    id=f"sch_{page}_{len(devices)}_{uuid.uuid4().hex[:4]}",
                    device_type=device_type.id,
                    category=device_type.category,
                    page=page,
                    bbox=[0, 0, 0, 0],  # No location from schedule
                    centroid=(0, 0),
                    detection_method="schedule",
                    tag=row.get("tag", "") or f"{device_type.id[:2].upper()}{i+1}",
                    confidence=0.9,  # High confidence from schedule
                    spec=spec,
                    spec_source="schedule",
                    source_text=device_text,
                    system=device_type.system,
                    subsystem=device_type.subsystem,
                    room_name=row.get("location", "") or row.get("room", ""),
                    rfi_needed=[
                        field for field in device_type.rfi_if_missing
                        if field not in spec or spec[field] == "TBD"
                    ],
                )

                devices.append(device)

            self.stats["schedule_matches"] += 1

        self.detected_devices.extend(devices)
        return devices

    def get_summary(self) -> Dict[str, Any]:
        """Get detection summary."""
        by_category = {}
        by_system = {}
        by_room = {}

        for device in self.detected_devices:
            # By category
            cat = device.category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "types": {}}
            by_category[cat]["count"] += 1
            dt = device.device_type
            by_category[cat]["types"][dt] = by_category[cat]["types"].get(dt, 0) + 1

            # By system
            sys = device.system
            if sys:
                if sys not in by_system:
                    by_system[sys] = 0
                by_system[sys] += 1

            # By room
            room = device.room_name or "Unassigned"
            if room not in by_room:
                by_room[room] = 0
            by_room[room] += 1

        return {
            "total_devices": len(self.detected_devices),
            "by_category": by_category,
            "by_system": by_system,
            "by_room": by_room,
            "detection_stats": self.stats,
            "devices_needing_rfi": sum(1 for d in self.detected_devices if d.rfi_needed),
        }


def detect_devices_in_plan(
    plan_data: Dict[str, Any],
    rooms: List[Dict[str, Any]] = None,
    registry: DeviceTypeRegistry = None,
) -> Tuple[List[DetectedDevice], Dict[str, Any]]:
    """
    Convenience function to detect devices in a plan.

    Args:
        plan_data: Plan data with vector_texts, vector_paths
        rooms: Room data for assignment
        registry: Device type registry

    Returns:
        (list of devices, summary dict)
    """
    detector = DeviceDetector(registry)

    devices = detector.detect_from_plan(
        vector_texts=plan_data.get("vector_texts", []),
        vector_paths=plan_data.get("vector_paths", []),
        page=plan_data.get("page", 0),
        rooms=rooms,
    )

    return devices, detector.get_summary()
