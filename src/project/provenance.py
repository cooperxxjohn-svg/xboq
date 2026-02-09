"""
Provenance / Audit Trail - Track origin of all extracted objects.

Every extracted object includes:
- source_pages (page_no, sheet_id)
- detection_method (vector/raster/ocr/rule)
- confidence
- ids linking back to geometry (room_id, opening_id, wall_id)

Outputs:
- provenance.json (complete audit trail)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Method used to detect/extract an object."""
    VECTOR = "vector"      # From vector PDF text/paths
    RASTER = "raster"      # From image analysis
    OCR = "ocr"            # From OCR text recognition
    RULE = "rule"          # From rule-based inference
    SCHEDULE = "schedule"  # From schedule parsing
    HYBRID = "hybrid"      # Multiple methods combined
    ASSUMED = "assumed"    # Default/assumed value
    MANUAL = "manual"      # Manually provided


@dataclass
class SourcePage:
    """Source page reference."""
    file_path: str
    page_number: int
    sheet_id: Optional[str] = None
    region_bbox: Optional[List[int]] = None  # [x, y, w, h] if localized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "page_number": self.page_number,
            "sheet_id": self.sheet_id,
            "region_bbox": self.region_bbox,
        }

    def to_ref_string(self) -> str:
        """Get compact reference string."""
        name = Path(self.file_path).stem
        sheet = f":{self.sheet_id}" if self.sheet_id else ""
        return f"{name}_p{self.page_number + 1}{sheet}"


@dataclass
class ProvenanceRecord:
    """Provenance record for a single extracted object."""
    object_id: str
    object_type: str  # room, door, window, wall, boq_line, etc.
    source_pages: List[SourcePage]
    detection_method: DetectionMethod
    confidence: float
    timestamp: str = ""

    # Linked IDs for cross-referencing
    linked_ids: Dict[str, str] = field(default_factory=dict)

    # Method-specific details
    method_details: Dict[str, Any] = field(default_factory=dict)

    # Was this measured or assumed?
    is_measured: bool = True

    # Human-readable description
    description: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "source_pages": [sp.to_dict() for sp in self.source_pages],
            "detection_method": self.detection_method.value,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "linked_ids": self.linked_ids,
            "method_details": self.method_details,
            "is_measured": self.is_measured,
            "description": self.description,
        }

    def get_source_refs(self) -> str:
        """Get comma-separated source page references."""
        return ", ".join(sp.to_ref_string() for sp in self.source_pages)

    def get_provenance_summary(self) -> str:
        """Get brief provenance summary for BOQ export."""
        method = self.detection_method.value
        conf = f"{self.confidence:.0%}"
        return f"{method}:{conf}"


@dataclass
class ProvenanceStore:
    """Container for all provenance records."""
    project_id: str
    created_at: str = ""
    records: List[ProvenanceRecord] = field(default_factory=list)

    # Indexes for fast lookup
    _by_id: Dict[str, ProvenanceRecord] = field(default_factory=dict, repr=False)
    _by_type: Dict[str, List[ProvenanceRecord]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def add(self, record: ProvenanceRecord) -> None:
        """Add a provenance record."""
        self.records.append(record)
        self._by_id[record.object_id] = record

        if record.object_type not in self._by_type:
            self._by_type[record.object_type] = []
        self._by_type[record.object_type].append(record)

    def get(self, object_id: str) -> Optional[ProvenanceRecord]:
        """Get record by object ID."""
        return self._by_id.get(object_id)

    def get_by_type(self, object_type: str) -> List[ProvenanceRecord]:
        """Get all records of a given type."""
        return self._by_type.get(object_type, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at,
            "total_records": len(self.records),
            "by_type": {t: len(recs) for t, recs in self._by_type.items()},
            "records": [r.to_dict() for r in self.records],
        }

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved {len(self.records)} provenance records to: {path}")

    @classmethod
    def load(cls, path: Path) -> "ProvenanceStore":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        store = cls(
            project_id=data["project_id"],
            created_at=data.get("created_at", ""),
        )

        for rec_data in data.get("records", []):
            source_pages = [
                SourcePage(**sp) for sp in rec_data.get("source_pages", [])
            ]
            record = ProvenanceRecord(
                object_id=rec_data["object_id"],
                object_type=rec_data["object_type"],
                source_pages=source_pages,
                detection_method=DetectionMethod(rec_data["detection_method"]),
                confidence=rec_data["confidence"],
                timestamp=rec_data.get("timestamp", ""),
                linked_ids=rec_data.get("linked_ids", {}),
                method_details=rec_data.get("method_details", {}),
                is_measured=rec_data.get("is_measured", True),
                description=rec_data.get("description", ""),
            )
            store.add(record)

        return store


class ProvenanceTracker:
    """
    Tracks provenance for all extracted objects.
    """

    def __init__(self, project_id: str):
        self.store = ProvenanceStore(project_id=project_id)
        self._id_counter = 0

    def _generate_id(self, prefix: str = "obj") -> str:
        """Generate unique object ID."""
        self._id_counter += 1
        return f"{prefix}_{self._id_counter:06d}"

    def track_room(
        self,
        room_id: str,
        file_path: str,
        page_number: int,
        sheet_id: Optional[str] = None,
        detection_method: DetectionMethod = DetectionMethod.RASTER,
        confidence: float = 1.0,
        label: str = "",
        area_sqm: float = 0.0,
        scale_method: Optional[str] = None,
        polygon_id: Optional[str] = None,
        bbox: Optional[List[int]] = None,
    ) -> ProvenanceRecord:
        """
        Track provenance for a detected room.

        Args:
            room_id: Room identifier
            file_path: Source file path
            page_number: Page number (0-indexed)
            sheet_id: Optional sheet ID
            detection_method: How the room was detected
            confidence: Detection confidence (0-1)
            label: Room label
            area_sqm: Computed area in sqm
            scale_method: Scale detection method used
            polygon_id: Linked polygon ID
            bbox: Bounding box [x, y, w, h]

        Returns:
            ProvenanceRecord
        """
        source = SourcePage(
            file_path=str(file_path),
            page_number=page_number,
            sheet_id=sheet_id,
            region_bbox=bbox,
        )

        linked_ids = {}
        if polygon_id:
            linked_ids["polygon_id"] = polygon_id

        method_details = {}
        if scale_method:
            method_details["scale_method"] = scale_method
        if area_sqm > 0:
            method_details["area_sqm"] = round(area_sqm, 2)

        record = ProvenanceRecord(
            object_id=room_id,
            object_type="room",
            source_pages=[source],
            detection_method=detection_method,
            confidence=confidence,
            linked_ids=linked_ids,
            method_details=method_details,
            is_measured=area_sqm > 0 and scale_method not in ["assumed", "default"],
            description=f"Room: {label}" if label else f"Room {room_id}",
        )

        self.store.add(record)
        return record

    def track_opening(
        self,
        opening_type: str,  # "door" or "window"
        file_path: str,
        page_number: int,
        tag: Optional[str] = None,
        sheet_id: Optional[str] = None,
        detection_method: DetectionMethod = DetectionMethod.RASTER,
        confidence: float = 1.0,
        dimensions: Optional[Dict[str, float]] = None,
        matched_schedule: bool = False,
        schedule_entry_id: Optional[str] = None,
        adjacent_rooms: Optional[List[str]] = None,
        bbox: Optional[List[int]] = None,
    ) -> ProvenanceRecord:
        """
        Track provenance for a detected opening (door/window).
        """
        opening_id = self._generate_id(opening_type[0].upper())

        source = SourcePage(
            file_path=str(file_path),
            page_number=page_number,
            sheet_id=sheet_id,
            region_bbox=bbox,
        )

        linked_ids = {}
        if schedule_entry_id:
            linked_ids["schedule_entry"] = schedule_entry_id
        if adjacent_rooms:
            for i, room_id in enumerate(adjacent_rooms):
                linked_ids[f"adjacent_room_{i}"] = room_id

        method_details = {}
        if dimensions:
            method_details["dimensions"] = dimensions
        if tag:
            method_details["tag"] = tag
        method_details["matched_to_schedule"] = matched_schedule

        record = ProvenanceRecord(
            object_id=opening_id,
            object_type=opening_type,
            source_pages=[source],
            detection_method=detection_method,
            confidence=confidence,
            linked_ids=linked_ids,
            method_details=method_details,
            is_measured=dimensions is not None and matched_schedule,
            description=f"{opening_type.title()}: {tag or opening_id}",
        )

        self.store.add(record)
        return record

    def track_wall(
        self,
        wall_id: str,
        file_path: str,
        page_number: int,
        sheet_id: Optional[str] = None,
        detection_method: DetectionMethod = DetectionMethod.RASTER,
        confidence: float = 1.0,
        thickness_mm: Optional[float] = None,
        length_mm: Optional[float] = None,
        is_external: bool = False,
        bbox: Optional[List[int]] = None,
    ) -> ProvenanceRecord:
        """Track provenance for a detected wall segment."""
        source = SourcePage(
            file_path=str(file_path),
            page_number=page_number,
            sheet_id=sheet_id,
            region_bbox=bbox,
        )

        method_details = {}
        if thickness_mm:
            method_details["thickness_mm"] = round(thickness_mm, 1)
        if length_mm:
            method_details["length_mm"] = round(length_mm, 1)
        method_details["is_external"] = is_external

        record = ProvenanceRecord(
            object_id=wall_id,
            object_type="wall",
            source_pages=[source],
            detection_method=detection_method,
            confidence=confidence,
            method_details=method_details,
            is_measured=thickness_mm is not None,
            description=f"Wall: {thickness_mm:.0f}mm" if thickness_mm else f"Wall {wall_id}",
        )

        self.store.add(record)
        return record

    def track_schedule_entry(
        self,
        entry_type: str,  # "door_schedule", "window_schedule", etc.
        tag: str,
        file_path: str,
        page_number: int,
        sheet_id: Optional[str] = None,
        detection_method: DetectionMethod = DetectionMethod.SCHEDULE,
        confidence: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """Track provenance for a schedule entry."""
        entry_id = self._generate_id(f"sched_{entry_type[0]}")

        source = SourcePage(
            file_path=str(file_path),
            page_number=page_number,
            sheet_id=sheet_id,
        )

        method_details = {"tag": tag}
        if properties:
            method_details["properties"] = properties

        record = ProvenanceRecord(
            object_id=entry_id,
            object_type=entry_type,
            source_pages=[source],
            detection_method=detection_method,
            confidence=confidence,
            method_details=method_details,
            is_measured=True,  # Schedule values are typically measured
            description=f"{entry_type}: {tag}",
        )

        self.store.add(record)
        return record

    def track_boq_line(
        self,
        boq_id: str,
        category: str,
        item_code: str,
        description: str,
        quantity: float,
        unit: str,
        source_records: List[ProvenanceRecord],
        is_measured: bool = True,
        computation_method: Optional[str] = None,
    ) -> ProvenanceRecord:
        """
        Track provenance for a BOQ line item.

        Args:
            boq_id: BOQ line identifier
            category: BOQ category
            item_code: Item code
            description: Item description
            quantity: Computed quantity
            unit: Unit of measure
            source_records: Source provenance records (rooms, openings, etc.)
            is_measured: Whether quantity is measured or assumed
            computation_method: How quantity was computed

        Returns:
            ProvenanceRecord
        """
        # Aggregate source pages from all source records
        source_pages = []
        seen = set()
        for rec in source_records:
            for sp in rec.source_pages:
                key = (sp.file_path, sp.page_number)
                if key not in seen:
                    source_pages.append(sp)
                    seen.add(key)

        # Aggregate linked IDs
        linked_ids = {}
        for rec in source_records:
            linked_ids[rec.object_id] = rec.object_type

        # Determine overall confidence
        if source_records:
            avg_conf = sum(r.confidence for r in source_records) / len(source_records)
        else:
            avg_conf = 0.5

        # Determine detection method
        if source_records:
            methods = set(r.detection_method for r in source_records)
            if len(methods) == 1:
                detection_method = list(methods)[0]
            else:
                detection_method = DetectionMethod.HYBRID
        else:
            detection_method = DetectionMethod.RULE

        method_details = {
            "category": category,
            "item_code": item_code,
            "quantity": round(quantity, 2),
            "unit": unit,
        }
        if computation_method:
            method_details["computation_method"] = computation_method

        record = ProvenanceRecord(
            object_id=boq_id,
            object_type="boq_line",
            source_pages=source_pages,
            detection_method=detection_method,
            confidence=avg_conf,
            linked_ids=linked_ids,
            method_details=method_details,
            is_measured=is_measured,
            description=f"{category}: {description}",
        )

        self.store.add(record)
        return record

    def track_area_computation(
        self,
        room_id: str,
        area_sqm: float,
        scale_pixels_per_mm: float,
        scale_method: str,
        scale_confidence: float,
        polygon_area_px: float,
    ) -> None:
        """
        Add area computation details to existing room record.
        """
        record = self.store.get(room_id)
        if record:
            record.method_details["area_sqm"] = round(area_sqm, 2)
            record.method_details["scale_pixels_per_mm"] = round(scale_pixels_per_mm, 4)
            record.method_details["scale_method"] = scale_method
            record.method_details["scale_confidence"] = round(scale_confidence, 3)
            record.method_details["polygon_area_px"] = round(polygon_area_px, 1)
            record.is_measured = scale_method not in ["assumed", "default"]

    def get_boq_provenance_columns(
        self,
        boq_id: str
    ) -> Dict[str, str]:
        """
        Get provenance columns for BOQ CSV export.

        Returns:
            Dict with source_pages, measured_or_assumed, provenance_ids
        """
        record = self.store.get(boq_id)
        if not record:
            return {
                "source_pages": "",
                "measured_or_assumed": "assumed",
                "provenance_ids": "",
            }

        return {
            "source_pages": record.get_source_refs(),
            "measured_or_assumed": "measured" if record.is_measured else "assumed",
            "provenance_ids": ",".join(record.linked_ids.keys()),
        }

    def export(self, output_dir: Path) -> Path:
        """Export provenance store to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "provenance.json"
        self.store.save(path)
        return path

    def get_summary(self) -> Dict[str, Any]:
        """Get provenance summary statistics."""
        by_type = {}
        measured_count = 0
        assumed_count = 0

        for record in self.store.records:
            by_type[record.object_type] = by_type.get(record.object_type, 0) + 1
            if record.is_measured:
                measured_count += 1
            else:
                assumed_count += 1

        return {
            "total_records": len(self.store.records),
            "by_type": by_type,
            "measured_count": measured_count,
            "assumed_count": assumed_count,
            "measured_pct": measured_count / len(self.store.records) * 100 if self.store.records else 0,
        }


def create_tracker(project_id: str) -> ProvenanceTracker:
    """Create a new provenance tracker."""
    return ProvenanceTracker(project_id)
