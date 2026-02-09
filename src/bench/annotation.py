"""
Annotation Format and Helpers for Benchmark Data.

Lightweight annotation format for floor plan ground truth:
- rooms: polygon points + label
- openings: bbox + type + tag
- scale: two points + real length

JSON schema per image stored in data/benchmark/annotations/<id>.json
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoomAnnotation:
    """Ground truth annotation for a single room."""
    id: str
    label: str  # Room type (Bedroom, Kitchen, Toilet, etc.)
    polygon: List[Tuple[float, float]]  # Points in image coordinates
    area_sqm: Optional[float] = None  # Ground truth area if known
    aliases: List[str] = field(default_factory=list)  # Alternative labels (WC -> Toilet)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "polygon": self.polygon,
            "area_sqm": self.area_sqm,
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoomAnnotation":
        return cls(
            id=data["id"],
            label=data["label"],
            polygon=[tuple(p) for p in data["polygon"]],
            area_sqm=data.get("area_sqm"),
            aliases=data.get("aliases", []),
        )

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, w, h)."""
        if not self.polygon:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        return (x, y, w, h)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Get centroid of polygon."""
        if not self.polygon:
            return (0, 0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class OpeningAnnotation:
    """Ground truth annotation for a door/window."""
    id: str
    type: str  # door, window, ventilator
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    tag: Optional[str] = None  # D1, W1, etc.
    connects_rooms: List[str] = field(default_factory=list)  # Room IDs it connects
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "bbox": list(self.bbox),
            "tag": self.tag,
            "connects_rooms": self.connects_rooms,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpeningAnnotation":
        return cls(
            id=data["id"],
            type=data["type"],
            bbox=tuple(data["bbox"]),
            tag=data.get("tag"),
            connects_rooms=data.get("connects_rooms", []),
            width_mm=data.get("width_mm"),
            height_mm=data.get("height_mm"),
        )


@dataclass
class ScaleAnnotation:
    """Ground truth annotation for scale."""
    point1: Tuple[float, float]  # First point in image coords
    point2: Tuple[float, float]  # Second point in image coords
    length_mm: float  # Real-world length between points

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point1": list(self.point1),
            "point2": list(self.point2),
            "length_mm": self.length_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaleAnnotation":
        return cls(
            point1=tuple(data["point1"]),
            point2=tuple(data["point2"]),
            length_mm=data["length_mm"],
        )

    @property
    def px_per_mm(self) -> float:
        """Calculate pixels per mm."""
        dx = self.point2[0] - self.point1[0]
        dy = self.point2[1] - self.point1[1]
        px_distance = np.sqrt(dx**2 + dy**2)
        return px_distance / self.length_mm if self.length_mm > 0 else 0


@dataclass
class AnnotationSchema:
    """Complete annotation for a floor plan image."""
    image_id: str
    image_path: str
    image_width: int
    image_height: int
    rooms: List[RoomAnnotation] = field(default_factory=list)
    openings: List[OpeningAnnotation] = field(default_factory=list)
    scale: Optional[ScaleAnnotation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "rooms": [r.to_dict() for r in self.rooms],
            "openings": [o.to_dict() for o in self.openings],
            "scale": self.scale.to_dict() if self.scale else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationSchema":
        return cls(
            image_id=data["image_id"],
            image_path=data["image_path"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            rooms=[RoomAnnotation.from_dict(r) for r in data.get("rooms", [])],
            openings=[OpeningAnnotation.from_dict(o) for o in data.get("openings", [])],
            scale=ScaleAnnotation.from_dict(data["scale"]) if data.get("scale") else None,
            metadata=data.get("metadata", {}),
        )

    @property
    def room_count(self) -> int:
        return len(self.rooms)

    @property
    def opening_count(self) -> int:
        return len(self.openings)

    @property
    def has_scale(self) -> bool:
        return self.scale is not None


def load_annotations(path: Path) -> AnnotationSchema:
    """Load annotations from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AnnotationSchema.from_dict(data)


def save_annotations(annotations: AnnotationSchema, path: Path) -> None:
    """Save annotations to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(annotations.to_dict(), f, indent=2)
    logger.info(f"Saved annotations to {path}")


def create_empty_annotation(
    image_id: str,
    image_path: str,
    image_width: int,
    image_height: int
) -> AnnotationSchema:
    """Create empty annotation template."""
    return AnnotationSchema(
        image_id=image_id,
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
    )


# India-specific room label aliases
INDIA_ROOM_ALIASES = {
    "Toilet": ["WC", "W.C.", "Bath", "Bathroom", "T&B", "OTS", "Attached Bath", "Common Bath"],
    "Bedroom": ["Bed", "Bed Room", "BR", "Master Bed", "MBR", "Guest Room"],
    "Living": ["Living Room", "Drawing", "Drawing Room", "Hall", "Lounge"],
    "Kitchen": ["Kit", "Pantry", "Modular Kitchen", "Open Kitchen"],
    "Dining": ["Dining Room", "Dining Hall"],
    "Balcony": ["Bal", "Verandah", "Sit Out", "Sitout", "Terrace"],
    "Pooja": ["Puja", "Prayer", "Prayer Room", "Mandir"],
    "Utility": ["Utility Room", "Servant", "Maid", "Service"],
    "Store": ["Storage", "Store Room", "Lumber"],
    "Passage": ["Lobby", "Corridor", "Foyer", "Entrance"],
    "Dressing": ["Dress", "Wardrobe", "Walk-in Closet"],
    "Study": ["Office", "Home Office", "Library", "Work Room"],
}


def normalize_room_label(label: str) -> str:
    """Normalize room label to canonical form using India aliases."""
    label_lower = label.lower().strip()

    for canonical, aliases in INDIA_ROOM_ALIASES.items():
        if label_lower == canonical.lower():
            return canonical
        for alias in aliases:
            if label_lower == alias.lower():
                return canonical

    # Return title case of original if no match
    return label.title()


def labels_match(pred_label: str, gt_label: str) -> bool:
    """Check if predicted and ground truth labels match (including aliases)."""
    norm_pred = normalize_room_label(pred_label)
    norm_gt = normalize_room_label(gt_label)
    return norm_pred == norm_gt


if __name__ == "__main__":
    # Example usage
    import sys
    import cv2

    if len(sys.argv) > 1:
        # Create empty annotation for an image
        img_path = Path(sys.argv[1])
        img = cv2.imread(str(img_path))

        if img is not None:
            h, w = img.shape[:2]
            ann = create_empty_annotation(
                image_id=img_path.stem,
                image_path=str(img_path),
                image_width=w,
                image_height=h,
            )

            out_path = Path(f"data/benchmark/annotations/{img_path.stem}.json")
            save_annotations(ann, out_path)
            print(f"Created empty annotation: {out_path}")
        else:
            print(f"Could not load image: {img_path}")
    else:
        # Test label matching
        test_labels = [
            ("Bedroom", "Bed Room"),
            ("Toilet", "WC"),
            ("Toilet", "Bath"),
            ("Kitchen", "Pantry"),
            ("Pooja", "Prayer Room"),
            ("Living", "Drawing"),
        ]

        print("Label matching tests:")
        for pred, gt in test_labels:
            match = labels_match(pred, gt)
            print(f"  {pred} vs {gt}: {'MATCH' if match else 'NO MATCH'}")
