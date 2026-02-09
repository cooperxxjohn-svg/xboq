"""
Tag Extraction Module - Indian Opening Conventions
Extracts and normalizes opening tags from OCR text.

Indian conventions:
- Doors: D, D1, D2, MD (main door), SD (sliding), TD (toilet), etc.
- Windows: W, W1, W2, LW (large), TW (toilet), KW (kitchen), etc.
- Ventilators: V, V1, VD, VENT
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpeningTag:
    """Extracted and normalized opening tag."""
    raw_text: str
    normalized_tag: str
    opening_type: str  # door, window, ventilator
    category: Optional[str] = None  # main, internal, toilet, sliding, etc.
    size_callout: Optional[str] = None  # e.g., "900x2100"
    width_mm: Optional[int] = None
    height_mm: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0


class TagExtractor:
    """
    Extract and normalize opening tags from OCR text.

    Handles Indian conventions for door/window labeling including:
    - Simple tags: D, W, V
    - Numbered tags: D1, W2, V1
    - Descriptive tags: MD, SD, TD, LW, TW
    - Size callouts: D1 900x2100, W2 1200x1200
    """

    # Tag patterns for Indian conventions
    DOOR_PATTERNS = {
        # Main door variants
        r"^M\.?D\.?$": ("main_door", "MD"),
        r"^MAIN\s*D(OOR)?$": ("main_door", "MD"),
        r"^ENTRANCE\s*D(OOR)?$": ("main_door", "MD"),

        # Sliding door
        r"^S\.?D\.?$": ("sliding_door", "SD"),
        r"^SLID(ING)?\s*D(OOR)?$": ("sliding_door", "SD"),

        # French door
        r"^F\.?D\.?$": ("french_door", "FD"),
        r"^FRENCH\s*D(OOR)?$": ("french_door", "FD"),
        r"^DOUBLE\s*D(OOR)?$": ("french_door", "FD"),

        # Toilet door
        r"^T\.?D\.?$": ("toilet_door", "TD"),
        r"^TOILET\s*D(OOR)?$": ("toilet_door", "TD"),
        r"^WC\s*D(OOR)?$": ("toilet_door", "TD"),
        r"^BATH(ROOM)?\s*D(OOR)?$": ("toilet_door", "TD"),

        # Balcony door
        r"^B\.?D\.?$": ("balcony_door", "BD"),
        r"^BALCONY\s*D(OOR)?$": ("balcony_door", "BD"),

        # Wooden door
        r"^W\.?D\.?$": ("wooden_door", "WD"),

        # Generic numbered doors
        r"^D(\d+)?$": ("internal_door", None),
        r"^DOOR\s*(\d+)?$": ("internal_door", None),
    }

    WINDOW_PATTERNS = {
        # Large window
        r"^L\.?W\.?$": ("large_window", "LW"),
        r"^LARGE\s*W(INDOW)?$": ("large_window", "LW"),
        r"^PICTURE\s*W(INDOW)?$": ("large_window", "LW"),

        # Toilet window
        r"^T\.?W\.?$": ("toilet_window", "TW"),
        r"^TOILET\s*W(INDOW)?$": ("toilet_window", "TW"),
        r"^WC\s*W(INDOW)?$": ("toilet_window", "TW"),
        r"^BATH(ROOM)?\s*W(INDOW)?$": ("toilet_window", "TW"),

        # Kitchen window
        r"^K\.?W\.?$": ("kitchen_window", "KW"),
        r"^KITCHEN\s*W(INDOW)?$": ("kitchen_window", "KW"),

        # Generic numbered windows
        r"^W(\d+)?$": ("standard_window", None),
        r"^WINDOW\s*(\d+)?$": ("standard_window", None),
    }

    VENTILATOR_PATTERNS = {
        r"^V\.?D\.?$": ("ventilator", "VD"),
        r"^V(\d+)?$": ("ventilator", None),
        r"^VENT(ILATOR)?$": ("ventilator", "V"),
    }

    # Size callout pattern: 900x2100, 1200 X 1500, etc.
    SIZE_PATTERN = re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})")

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize tag extractor.

        Args:
            config_path: Optional path to custom aliases YAML file
        """
        self.custom_aliases: Dict[str, Dict] = {}

        if config_path and config_path.exists():
            self._load_custom_aliases(config_path)

    def _load_custom_aliases(self, config_path: Path) -> None:
        """Load custom tag aliases from YAML config."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                self.custom_aliases = data.get("opening_aliases", {})
        except Exception as e:
            logger.warning(f"Could not load custom aliases: {e}")

    def extract_from_texts(
        self,
        texts: List[Dict],
    ) -> List[OpeningTag]:
        """
        Extract opening tags from OCR text boxes.

        Args:
            texts: List of text boxes with {"text", "bbox", "confidence"}

        Returns:
            List of extracted OpeningTag objects
        """
        tags = []

        for text_box in texts:
            raw_text = text_box.get("text", "").strip()
            bbox = text_box.get("bbox")
            text_confidence = text_box.get("confidence", 0.5)

            # Try to extract tag
            tag = self._extract_single_tag(raw_text, bbox, text_confidence)
            if tag:
                tags.append(tag)

        return tags

    def _extract_single_tag(
        self,
        raw_text: str,
        bbox: Optional[Tuple[int, int, int, int]],
        text_confidence: float,
    ) -> Optional[OpeningTag]:
        """Extract tag from a single text string."""
        if not raw_text or len(raw_text) > 30:
            return None

        # Normalize text
        text_upper = raw_text.upper().strip()

        # Check for size callout
        size_match = self.SIZE_PATTERN.search(text_upper)
        size_callout = None
        width_mm = None
        height_mm = None

        if size_match:
            width_mm = int(size_match.group(1))
            height_mm = int(size_match.group(2))
            size_callout = f"{width_mm}x{height_mm}"
            # Remove size from text for tag matching
            text_upper = self.SIZE_PATTERN.sub("", text_upper).strip()

        # Try door patterns
        for pattern, (category, normalized) in self.DOOR_PATTERNS.items():
            match = re.match(pattern, text_upper, re.IGNORECASE)
            if match:
                # Get number if present
                groups = match.groups()
                num = groups[0] if groups and groups[0] else ""

                final_tag = normalized if normalized else f"D{num}"

                return OpeningTag(
                    raw_text=raw_text,
                    normalized_tag=final_tag,
                    opening_type="door",
                    category=category,
                    size_callout=size_callout,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    bbox=bbox,
                    confidence=text_confidence,
                )

        # Try window patterns
        for pattern, (category, normalized) in self.WINDOW_PATTERNS.items():
            match = re.match(pattern, text_upper, re.IGNORECASE)
            if match:
                groups = match.groups()
                num = groups[0] if groups and groups[0] else ""

                final_tag = normalized if normalized else f"W{num}"

                return OpeningTag(
                    raw_text=raw_text,
                    normalized_tag=final_tag,
                    opening_type="window",
                    category=category,
                    size_callout=size_callout,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    bbox=bbox,
                    confidence=text_confidence,
                )

        # Try ventilator patterns
        for pattern, (category, normalized) in self.VENTILATOR_PATTERNS.items():
            match = re.match(pattern, text_upper, re.IGNORECASE)
            if match:
                groups = match.groups()
                num = groups[0] if groups and groups[0] else ""

                final_tag = normalized if normalized else f"V{num}"

                return OpeningTag(
                    raw_text=raw_text,
                    normalized_tag=final_tag,
                    opening_type="ventilator",
                    category=category,
                    size_callout=size_callout,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    bbox=bbox,
                    confidence=text_confidence * 0.9,  # Slight penalty for ventilators
                )

        return None

    def normalize_tag(self, raw_tag: str) -> Optional[str]:
        """
        Normalize a raw tag string.

        Args:
            raw_tag: Raw tag text

        Returns:
            Normalized tag or None if not recognized
        """
        result = self._extract_single_tag(raw_tag, None, 1.0)
        return result.normalized_tag if result else None

    def get_tag_type(self, tag: str) -> Optional[str]:
        """
        Get the opening type for a tag.

        Args:
            tag: Tag string (D1, W2, V, etc.)

        Returns:
            Opening type: door, window, or ventilator
        """
        result = self._extract_single_tag(tag, None, 1.0)
        return result.opening_type if result else None

    def get_tag_category(self, tag: str) -> Optional[str]:
        """
        Get the category for a tag.

        Args:
            tag: Tag string

        Returns:
            Category like main_door, toilet_window, etc.
        """
        result = self._extract_single_tag(tag, None, 1.0)
        return result.category if result else None


def extract_size_from_text(text: str) -> Optional[Tuple[int, int]]:
    """
    Extract size dimensions from text.

    Args:
        text: Text that may contain size (e.g., "900x2100", "D1 1000 X 2100")

    Returns:
        Tuple of (width_mm, height_mm) or None
    """
    pattern = re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})")
    match = pattern.search(text)

    if match:
        return (int(match.group(1)), int(match.group(2)))

    return None


def build_tag_index(tags: List[OpeningTag]) -> Dict[str, List[OpeningTag]]:
    """
    Build an index of tags by normalized name.

    Args:
        tags: List of OpeningTag objects

    Returns:
        Dict mapping normalized tag to list of instances
    """
    index: Dict[str, List[OpeningTag]] = {}

    for tag in tags:
        key = tag.normalized_tag
        if key not in index:
            index[key] = []
        index[key].append(tag)

    return index


def summarize_tags(tags: List[OpeningTag]) -> Dict[str, Dict]:
    """
    Create summary statistics for tags.

    Args:
        tags: List of OpeningTag objects

    Returns:
        Dict with counts and sizes by tag type
    """
    summary = {
        "doors": {},
        "windows": {},
        "ventilators": {},
    }

    for tag in tags:
        if tag.opening_type == "door":
            bucket = summary["doors"]
        elif tag.opening_type == "window":
            bucket = summary["windows"]
        else:
            bucket = summary["ventilators"]

        key = tag.normalized_tag
        if key not in bucket:
            bucket[key] = {
                "count": 0,
                "sizes": [],
                "category": tag.category,
            }

        bucket[key]["count"] += 1
        if tag.width_mm and tag.height_mm:
            bucket[key]["sizes"].append((tag.width_mm, tag.height_mm))

    return summary
