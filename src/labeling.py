"""
Floor Plan Room Labeling Module
Assigns labels to detected rooms using text detection and inference.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import cv2
import yaml

from .polygons import RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    """A detected text box."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float = 0.8

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class RoomLabel:
    """Assigned label for a room."""
    canonical: str  # Standardized name (e.g., "Bedroom")
    original_text: str  # Original detected text
    confidence: float
    source: str  # "vector_text", "ocr", "inference", "manual"
    text_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class LabeledRoom:
    """A room with its assigned label."""
    polygon: RoomPolygon
    label: RoomLabel
    alternative_labels: List[RoomLabel] = field(default_factory=list)


class RoomLabeler:
    """
    Assigns labels to rooms using multiple strategies.
    """

    def __init__(self, rules_path: Optional[Path] = None):
        """
        Initialize labeler with room alias rules.

        Args:
            rules_path: Path to room_aliases.yaml
        """
        self.aliases = {}
        self.patterns = {}
        self.unknown_markers = []

        # Load rules
        if rules_path and rules_path.exists():
            self._load_rules(rules_path)
        else:
            # Default rules path
            default_path = Path(__file__).parent.parent / "rules" / "room_aliases.yaml"
            if default_path.exists():
                self._load_rules(default_path)
            else:
                self._init_default_rules()

    def _load_rules(self, path: Path):
        """Load room alias rules from YAML."""
        try:
            with open(path) as f:
                rules = yaml.safe_load(f)

            categories = rules.get('room_categories', {})
            for cat_name, cat_data in categories.items():
                canonical = cat_data.get('canonical', cat_name.title())
                aliases = cat_data.get('aliases', [])
                patterns = cat_data.get('patterns', [])

                for alias in aliases:
                    self.aliases[alias.lower()] = canonical

                for pattern in patterns:
                    self.patterns[pattern] = canonical

            self.unknown_markers = rules.get('unknown_markers', [])

            logger.info(f"Loaded {len(self.aliases)} aliases, {len(self.patterns)} patterns")

        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self._init_default_rules()

    def _init_default_rules(self):
        """Initialize with minimal default rules."""
        defaults = {
            'bedroom': 'Bedroom', 'bed': 'Bedroom', 'bed room': 'Bedroom',
            'master bed': 'Bedroom', 'mbr': 'Bedroom',
            'living': 'Living', 'living room': 'Living', 'drawing': 'Living',
            'hall': 'Living', 'lounge': 'Living',
            'dining': 'Dining', 'dining room': 'Dining',
            'kitchen': 'Kitchen', 'kit': 'Kitchen', 'pantry': 'Kitchen',
            'toilet': 'Toilet', 'bath': 'Toilet', 'bathroom': 'Toilet',
            'wc': 'Toilet', 'w.c.': 'Toilet', 't&b': 'Toilet', 'ots': 'Toilet',
            'balcony': 'Balcony', 'bal': 'Balcony', 'verandah': 'Balcony',
            'sitout': 'Balcony', 'sit out': 'Balcony',
            'terrace': 'Terrace', 'roof': 'Terrace',
            'passage': 'Passage', 'lobby': 'Passage', 'corridor': 'Passage',
            'foyer': 'Passage', 'entrance': 'Passage',
            'store': 'Store', 'storage': 'Store', 'store room': 'Store',
            'utility': 'Utility', 'servant': 'Utility', 'maid': 'Utility',
            'pooja': 'Pooja', 'puja': 'Pooja', 'prayer': 'Pooja',
            'study': 'Study', 'office': 'Study', 'library': 'Study',
            'dressing': 'Dressing', 'dress': 'Dressing', 'wardrobe': 'Dressing',
            'stair': 'Staircase', 'staircase': 'Staircase', 'stairs': 'Staircase',
            'lift': 'Lift', 'elevator': 'Lift',
            'duct': 'Duct', 'shaft': 'Duct',
            'parking': 'Parking', 'garage': 'Parking',
            'wash': 'Wash', 'laundry': 'Wash',
        }
        self.aliases = defaults
        self.patterns = {
            r'bed\s*\d+': 'Bedroom',
            r'br\s*\d+': 'Bedroom',
            r'toilet\s*\d+': 'Toilet',
            r'bath\s*\d+': 'Toilet',
        }
        self.unknown_markers = ['room', 'space', 'area']

    def label_rooms(
        self,
        polygons: List[RoomPolygon],
        vector_texts: List[Any] = None,
        image: np.ndarray = None
    ) -> List[LabeledRoom]:
        """
        Assign labels to all room polygons.

        Args:
            polygons: List of room polygons
            vector_texts: Text extracted from vector PDF
            image: Image for OCR fallback

        Returns:
            List of LabeledRoom objects
        """
        logger.info(f"Labeling {len(polygons)} rooms")

        # Collect all text boxes
        text_boxes = []

        # From vector texts
        if vector_texts:
            for vt in vector_texts:
                text_boxes.append(TextBox(
                    text=vt.text,
                    bbox=vt.bbox,
                    confidence=0.95
                ))

        # From OCR
        if image is not None:
            ocr_boxes = self._extract_text_ocr(image)
            text_boxes.extend(ocr_boxes)

        logger.info(f"Found {len(text_boxes)} text boxes")

        # Assign labels to each room
        labeled_rooms = []

        for polygon in polygons:
            if not polygon.is_valid:
                continue

            label = self._find_best_label(polygon, text_boxes)
            labeled_rooms.append(LabeledRoom(
                polygon=polygon,
                label=label
            ))

        # Handle duplicate labels (add numbers)
        labeled_rooms = self._deduplicate_labels(labeled_rooms)

        return labeled_rooms

    def _extract_text_ocr(self, image: np.ndarray) -> List[TextBox]:
        """Extract text using OCR."""
        text_boxes = []

        try:
            import pytesseract

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Get OCR data with bounding boxes
            data = pytesseract.image_to_data(
                gray,
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )

            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if text and conf > 30:  # Minimum confidence
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]

                    text_boxes.append(TextBox(
                        text=text,
                        bbox=(x, y, x + w, y + h),
                        confidence=conf / 100.0
                    ))

        except ImportError:
            logger.debug("pytesseract not available")
        except Exception as e:
            logger.warning(f"OCR failed: {e}")

        return text_boxes

    def _find_best_label(
        self,
        polygon: RoomPolygon,
        text_boxes: List[TextBox]
    ) -> RoomLabel:
        """Find the best label for a room polygon."""

        # Find text boxes inside or near the polygon
        candidates = []

        for tb in text_boxes:
            # Check if text center is inside polygon
            if self._point_in_polygon(tb.center, polygon):
                distance = 0  # Inside
            else:
                # Calculate distance to polygon centroid
                distance = self._distance(tb.center, polygon.centroid)

                # Skip if too far (more than half the polygon size)
                max_dist = max(polygon.bbox[2], polygon.bbox[3])
                if distance > max_dist:
                    continue

            # Try to match text to known room type
            canonical = self._match_text(tb.text)

            if canonical:
                score = tb.confidence * (1.0 / (1.0 + distance / 100))
                candidates.append((score, canonical, tb))

        # Sort by score
        candidates.sort(key=lambda x: x[0], reverse=True)

        if candidates:
            score, canonical, tb = candidates[0]
            return RoomLabel(
                canonical=canonical,
                original_text=tb.text,
                confidence=min(score, 0.99),
                source="vector_text" if tb.confidence > 0.9 else "ocr",
                text_bbox=tb.bbox
            )

        # No label found - return unknown
        return RoomLabel(
            canonical="Room",
            original_text="",
            confidence=0.1,
            source="inference"
        )

    def _match_text(self, text: str) -> Optional[str]:
        """Match text to canonical room name."""
        text_lower = text.lower().strip()

        # Skip very short texts
        if len(text_lower) < 2:
            return None

        # Skip unknown markers
        if text_lower in self.unknown_markers:
            return None

        # Direct alias match
        if text_lower in self.aliases:
            return self.aliases[text_lower]

        # Try without punctuation
        text_clean = re.sub(r'[^\w\s]', '', text_lower)
        if text_clean in self.aliases:
            return self.aliases[text_clean]

        # Pattern match
        for pattern, canonical in self.patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return canonical

        # Partial match (text contains alias)
        for alias, canonical in self.aliases.items():
            if len(alias) >= 3 and alias in text_lower:
                return canonical

        # Check if text is contained in any alias
        for alias, canonical in self.aliases.items():
            if len(text_lower) >= 3 and text_lower in alias:
                return canonical

        return None

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: RoomPolygon
    ) -> bool:
        """Check if point is inside polygon."""
        if not polygon.points:
            return False

        pts = np.array(polygon.points, dtype=np.float32)
        result = cv2.pointPolygonTest(pts, point, False)
        return result >= 0

    def _distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _deduplicate_labels(
        self,
        rooms: List[LabeledRoom]
    ) -> List[LabeledRoom]:
        """Add numbers to duplicate labels."""
        label_counts: Dict[str, int] = {}
        label_totals: Dict[str, int] = {}

        # Count occurrences
        for room in rooms:
            canonical = room.label.canonical
            label_totals[canonical] = label_totals.get(canonical, 0) + 1

        # Assign numbers to duplicates
        for room in rooms:
            canonical = room.label.canonical

            if label_totals.get(canonical, 0) > 1:
                count = label_counts.get(canonical, 0) + 1
                label_counts[canonical] = count

                # Update label with number
                room.label.canonical = f"{canonical} {count}"

        return rooms


def label_rooms(
    polygons: List[RoomPolygon],
    vector_texts: List[Any] = None,
    image: np.ndarray = None,
    rules_path: Optional[Path] = None
) -> List[LabeledRoom]:
    """
    Convenience function to label rooms.

    Args:
        polygons: Room polygons
        vector_texts: Vector text objects
        image: Image for OCR
        rules_path: Path to rules file

    Returns:
        List of LabeledRoom objects
    """
    labeler = RoomLabeler(rules_path)
    return labeler.label_rooms(polygons, vector_texts, image)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test with sample texts
    labeler = RoomLabeler()

    test_texts = [
        "Bedroom", "BED ROOM", "Bed 1", "BR2", "Master Bed",
        "Kitchen", "KIT", "Pantry",
        "Toilet", "W.C.", "Bath", "T&B", "OTS",
        "Living", "Drawing Room", "Hall",
        "Balcony", "Sit Out", "Verandah",
        "Pooja Room", "Puja",
        "Utility", "Servant Room", "Maid",
    ]

    print("Room label matching test:")
    print("-" * 40)

    for text in test_texts:
        canonical = labeler._match_text(text)
        print(f"  '{text}' -> {canonical}")
