"""
Footing Detection Module
Detects footings from foundation plans and associates with columns.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from pathlib import Path
import yaml

from .detect_columns import DetectedColumn

logger = logging.getLogger(__name__)


@dataclass
class DetectedFooting:
    """A detected footing."""
    footing_id: str
    label: str  # e.g., "F1", "F2", "IF1" (isolated), "CF1" (combined)
    footing_type: str = "isolated"  # "isolated", "combined", "strip", "raft"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    center: Tuple[float, float] = (0, 0)
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    size_mm: Optional[Tuple[int, int, int]] = None  # length x width x depth
    pedestal_size_mm: Optional[Tuple[int, int, int]] = None  # if has pedestal
    associated_columns: List[str] = field(default_factory=list)  # column_ids
    confidence: float = 0.5
    source: str = "detection"  # "detection", "schedule", "callout"
    concrete_grade: Optional[str] = None
    rebar_details: Optional[str] = None  # e.g., "Y12@150 B/W"


@dataclass
class FootingDetectionResult:
    """Result of footing detection."""
    footings: List[DetectedFooting]
    column_footing_map: Dict[str, str] = field(default_factory=dict)  # column_id -> footing_id
    unique_labels: List[str] = field(default_factory=list)
    size_mappings: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    detection_method: str = "morphological"


class FootingDetector:
    """
    Detects footings in foundation plans.
    Footings appear as larger rectangles, often with hatching or dashed outlines.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize detector with configuration."""
        self.config = self._load_config(config_path)
        self.keywords = self._load_keywords()

        # Detection parameters - footings are larger than columns
        self.min_size = self.config.get('min_size_mm', 600)  # Minimum footing dimension
        self.max_size = self.config.get('max_size_mm', 3000)  # Maximum footing dimension
        self.min_aspect = self.config.get('min_aspect_ratio', 0.3)
        self.max_aspect = self.config.get('max_aspect_ratio', 3.0)

    def _load_config(self, path: Optional[Path]) -> Dict:
        """Load configuration from assumptions.yaml."""
        default_path = Path(__file__).parent.parent.parent / "rules" / "assumptions.yaml"
        try:
            with open(path or default_path) as f:
                data = yaml.safe_load(f)
            return data.get('detection', {}).get('footing', {})
        except Exception:
            return {
                'min_size_mm': 600,
                'max_size_mm': 3000,
                'min_aspect_ratio': 0.3,
                'max_aspect_ratio': 3.0
            }

    def _load_keywords(self) -> Dict:
        """Load keywords from structural_keywords.yaml."""
        keywords_path = Path(__file__).parent.parent.parent / "rules" / "structural_keywords.yaml"
        try:
            with open(keywords_path) as f:
                data = yaml.safe_load(f)
            return data.get('element_labels', {}).get('footing', {})
        except Exception:
            return {
                'patterns': [r'F\d+', r'IF\d+', r'CF\d+'],
                'keywords': ['footing', 'foundation', 'ftg']
            }

    def detect(
        self,
        image: np.ndarray,
        columns: List[DetectedColumn] = None,
        vector_texts: List[Any] = None,
        scale_px_per_mm: float = 0.118,
        drawing_type: str = "unknown"
    ) -> FootingDetectionResult:
        """
        Detect footings in image.

        Args:
            image: Drawing image (typically foundation plan)
            columns: Previously detected columns for association
            vector_texts: Text blocks from PDF
            scale_px_per_mm: Scale factor
            drawing_type: Type of drawing for tuning

        Returns:
            FootingDetectionResult
        """
        logger.info("Detecting footings...")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Binarize
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 10
        )

        # Detect footing candidates
        candidates = self._detect_footing_shapes(binary, scale_px_per_mm)

        # Associate with columns
        if columns:
            candidates = self._associate_columns(candidates, columns)

        # Classify footing types
        candidates = self._classify_footing_types(candidates)

        # Extract labels from nearby text
        labeled_footings = self._assign_labels(candidates, vector_texts, image)

        # Build column-footing map
        col_ftg_map = {}
        for ftg in labeled_footings:
            for col_id in ftg.associated_columns:
                col_ftg_map[col_id] = ftg.footing_id

        # Create result
        result = FootingDetectionResult(
            footings=labeled_footings,
            column_footing_map=col_ftg_map,
            unique_labels=list(set(f.label for f in labeled_footings if f.label)),
            detection_method="morphological"
        )

        logger.info(f"Detected {len(result.footings)} footings, "
                   f"{len(result.unique_labels)} unique labels")

        return result

    def _detect_footing_shapes(
        self,
        binary: np.ndarray,
        scale: float
    ) -> List[DetectedFooting]:
        """Detect rectangular shapes that could be footings."""
        footings = []

        # Calculate size thresholds in pixels
        min_size_px = int(self.min_size * scale)
        max_size_px = int(self.max_size * scale)

        # Morphological operations to find footing outlines
        # Footings often have dashed/hatched interiors, so we look for outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter - footings are larger than columns
            if w < min_size_px or h < min_size_px:
                continue
            if w > max_size_px or h > max_size_px:
                continue

            # Aspect ratio filter
            aspect = w / h if h > 0 else 0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Footings can be hollow or hatched
            roi = closed[y:y+h, x:x+w]
            fill_ratio = np.sum(roi > 0) / (w * h)

            # Accept both filled and hollow shapes
            # Hatched footings typically have 20-80% fill
            if fill_ratio < 0.1:  # Too empty
                continue

            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Footings are typically rectangular (4-8 sides for rounded corners)
            if len(approx) < 4 or len(approx) > 12:
                continue

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w/2, y + h/2

            # Convert size to mm
            w_mm = int(w / scale)
            h_mm = int(h / scale)

            # Create footing
            polygon = [(float(p[0][0]), float(p[0][1])) for p in approx]

            # Estimate depth based on typical Indian footing proportions
            # Typically depth = max(300, min(w,h)/3)
            depth_mm = max(300, min(w_mm, h_mm) // 3)

            footings.append(DetectedFooting(
                footing_id=f"F{i+1:03d}",
                label="",
                bbox=(x, y, w, h),
                center=(cx, cy),
                polygon=polygon,
                size_mm=(w_mm, h_mm, depth_mm),
                confidence=0.5
            ))

        return footings

    def _associate_columns(
        self,
        footings: List[DetectedFooting],
        columns: List[DetectedColumn]
    ) -> List[DetectedFooting]:
        """Associate footings with columns based on position overlap."""

        for footing in footings:
            fx, fy, fw, fh = footing.bbox
            footing_rect = (fx, fy, fx + fw, fy + fh)

            for col in columns:
                cx, cy, cw, ch = col.bbox
                col_rect = (cx, cy, cx + cw, cy + ch)

                # Check if column center is within footing
                col_center = col.center
                if (footing_rect[0] <= col_center[0] <= footing_rect[2] and
                    footing_rect[1] <= col_center[1] <= footing_rect[3]):
                    footing.associated_columns.append(col.column_id)
                    continue

                # Check if footing center is near column
                dist = np.sqrt(
                    (footing.center[0] - col.center[0])**2 +
                    (footing.center[1] - col.center[1])**2
                )
                max_dist = max(fw, fh) * 0.6
                if dist <= max_dist:
                    footing.associated_columns.append(col.column_id)

        return footings

    def _classify_footing_types(
        self,
        footings: List[DetectedFooting]
    ) -> List[DetectedFooting]:
        """Classify footings as isolated, combined, strip, etc."""

        for footing in footings:
            num_columns = len(footing.associated_columns)

            if num_columns == 0:
                # Might be a strip footing or unassociated
                w, h = footing.size_mm[0], footing.size_mm[1]
                if max(w, h) / min(w, h) > 3:
                    footing.footing_type = "strip"
                else:
                    footing.footing_type = "isolated"
                footing.confidence = 0.4

            elif num_columns == 1:
                footing.footing_type = "isolated"
                footing.confidence = min(0.85, footing.confidence + 0.2)

            elif num_columns == 2:
                footing.footing_type = "combined"
                footing.confidence = min(0.85, footing.confidence + 0.2)

            else:
                # 3+ columns - could be raft portion
                w, h = footing.size_mm[0], footing.size_mm[1]
                if max(w, h) > 3000:
                    footing.footing_type = "raft"
                else:
                    footing.footing_type = "combined"
                footing.confidence = min(0.8, footing.confidence + 0.15)

        return footings

    def _assign_labels(
        self,
        footings: List[DetectedFooting],
        vector_texts: List[Any],
        image: np.ndarray
    ) -> List[DetectedFooting]:
        """Assign labels to footings from nearby text."""

        label_texts = []

        if vector_texts:
            for vt in vector_texts:
                text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
                bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))
                label_texts.append({
                    'text': text,
                    'bbox': bbox,
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                })

        # Match labels to footings
        patterns = self.keywords.get('patterns', [r'F\d+', r'IF\d+', r'CF\d+'])

        for ftg in footings:
            best_match = None
            best_dist = float('inf')

            for lt in label_texts:
                text = lt['text'].strip().upper()
                is_footing_label = False

                for pattern in patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        is_footing_label = True
                        break

                if not is_footing_label:
                    continue

                # Distance to footing center
                dist = np.sqrt(
                    (lt['center'][0] - ftg.center[0])**2 +
                    (lt['center'][1] - ftg.center[1])**2
                )

                # Label should be close to footing
                max_dist = max(ftg.bbox[2], ftg.bbox[3]) * 2

                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_match = text

            if best_match:
                ftg.label = best_match
                ftg.confidence = min(0.9, ftg.confidence + 0.15)

                # Update type based on label prefix
                if best_match.startswith('IF'):
                    ftg.footing_type = "isolated"
                elif best_match.startswith('CF'):
                    ftg.footing_type = "combined"
                elif best_match.startswith('SF'):
                    ftg.footing_type = "strip"

        # Auto-label unlabeled footings
        auto_num = 1
        for ftg in footings:
            if not ftg.label:
                prefix = ftg.footing_type[0].upper()  # I, C, S, R
                ftg.label = f"{prefix}F{auto_num}"
                ftg.footing_id = f"F{auto_num:03d}"
                auto_num += 1

        return footings

    def detect_from_schedule(
        self,
        schedule_text: str,
        footings: List[DetectedFooting]
    ) -> List[DetectedFooting]:
        """
        Update footings with sizes from schedule.

        Args:
            schedule_text: Text content of footing schedule
            footings: Detected footings to update

        Returns:
            Updated footings
        """
        # Pattern: F1 - 1500x1500x450, F1: 1.5m x 1.5m x 450mm
        patterns = [
            # millimeters
            r'([ICF]?F\d+)\s*[-:]?\s*(\d{3,4})\s*[xX×]\s*(\d{3,4})\s*[xX×]\s*(\d{3,4})',
            # with meters
            r'([ICF]?F\d+)\s*[-:]?\s*(\d+\.?\d*)\s*[mM]\s*[xX×]\s*(\d+\.?\d*)\s*[mM]\s*[xX×]\s*(\d+)'
        ]

        mappings = {}

        for pattern in patterns:
            for match in re.finditer(pattern, schedule_text, re.IGNORECASE):
                label = match.group(1).upper()
                v1, v2, v3 = match.group(2), match.group(3), match.group(4)

                # Check if meters or mm
                if '.' in v1 or float(v1) < 10:
                    # Meters - convert to mm
                    length = int(float(v1) * 1000)
                    width = int(float(v2) * 1000)
                    depth = int(float(v3))
                else:
                    length = int(v1)
                    width = int(v2)
                    depth = int(v3)

                mappings[label] = (length, width, depth)

        # Update footings
        for ftg in footings:
            label_upper = ftg.label.upper()
            if label_upper in mappings:
                ftg.size_mm = mappings[label_upper]
                ftg.source = "schedule"
                ftg.confidence = min(0.95, ftg.confidence + 0.2)

        logger.info(f"Applied {len(mappings)} size mappings from footing schedule")
        return footings


def detect_footings(
    image: np.ndarray,
    columns: List[DetectedColumn] = None,
    vector_texts: List[Any] = None,
    scale_px_per_mm: float = 0.118
) -> FootingDetectionResult:
    """
    Convenience function to detect footings.

    Args:
        image: Drawing image
        columns: Detected columns
        vector_texts: Text blocks
        scale_px_per_mm: Scale factor

    Returns:
        FootingDetectionResult
    """
    detector = FootingDetector()
    return detector.detect(image, columns, vector_texts, scale_px_per_mm)


if __name__ == "__main__":
    import sys
    from .detect_columns import detect_columns

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            # First detect columns
            col_result = detect_columns(img)
            print(f"Detected {len(col_result.columns)} columns")

            # Then detect footings
            ftg_result = detect_footings(img, col_result.columns)
            print(f"Detected {len(ftg_result.footings)} footings")

            for ftg in ftg_result.footings[:10]:
                print(f"  {ftg.label} ({ftg.footing_type}): {ftg.size_mm}, "
                      f"columns={ftg.associated_columns}")
    else:
        print("Usage: python -m src.structural.detect_footings <image>")
