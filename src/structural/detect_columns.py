"""
Column Detection Module
Detects columns from structural drawings or architectural plans.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DetectedColumn:
    """A detected column."""
    column_id: str
    label: str  # e.g., "C1", "C2"
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels
    center: Tuple[float, float]
    polygon: List[Tuple[float, float]]
    size_mm: Optional[Tuple[int, int]] = None  # width x depth
    grid_location: Optional[str] = None  # e.g., "A-1"
    confidence: float = 0.5
    source: str = "detection"  # "detection", "schedule", "callout"
    floor: str = "typical"
    concrete_grade: Optional[str] = None


@dataclass
class ColumnDetectionResult:
    """Result of column detection."""
    columns: List[DetectedColumn]
    column_mask: Optional[np.ndarray] = None
    unique_labels: List[str] = field(default_factory=list)
    size_mappings: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    detection_method: str = "morphological"


class ColumnDetector:
    """
    Detects columns in floor plans and structural drawings.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize detector with configuration."""
        self.config = self._load_config(config_path)
        self.keywords = self._load_keywords()

        # Detection parameters
        self.min_size = self.config.get('min_size_mm', 150)
        self.max_size = self.config.get('max_size_mm', 800)
        self.min_aspect = self.config.get('min_aspect_ratio', 0.5)
        self.max_aspect = self.config.get('max_aspect_ratio', 3.0)

    def _load_config(self, path: Optional[Path]) -> Dict:
        """Load configuration from assumptions.yaml."""
        default_path = Path(__file__).parent.parent.parent / "rules" / "assumptions.yaml"
        try:
            with open(path or default_path) as f:
                data = yaml.safe_load(f)
            return data.get('detection', {}).get('column', {})
        except Exception:
            return {
                'min_size_mm': 150,
                'max_size_mm': 800,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 3.0
            }

    def _load_keywords(self) -> Dict:
        """Load keywords from structural_keywords.yaml."""
        keywords_path = Path(__file__).parent.parent.parent / "rules" / "structural_keywords.yaml"
        try:
            with open(keywords_path) as f:
                data = yaml.safe_load(f)
            return data.get('element_labels', {}).get('column', {})
        except Exception:
            return {
                'patterns': [r'C\d+', r'COL\s*\d+'],
                'keywords': ['column', 'col']
            }

    def detect(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None,
        scale_px_per_mm: float = 0.118,
        drawing_type: str = "unknown"
    ) -> ColumnDetectionResult:
        """
        Detect columns in image.

        Args:
            image: Drawing image
            vector_texts: Text blocks from PDF
            scale_px_per_mm: Scale factor
            drawing_type: Type of drawing for tuning

        Returns:
            ColumnDetectionResult
        """
        logger.info("Detecting columns...")

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

        # Detect column candidates
        candidates = self._detect_column_shapes(binary, scale_px_per_mm)

        # Extract labels from nearby text
        labeled_columns = self._assign_labels(candidates, vector_texts, image)

        # Cluster by size
        clustered = self._cluster_by_size(labeled_columns)

        # Create result
        result = ColumnDetectionResult(
            columns=clustered,
            unique_labels=list(set(c.label for c in clustered if c.label)),
            detection_method="morphological"
        )

        logger.info(f"Detected {len(result.columns)} columns, "
                   f"{len(result.unique_labels)} unique labels")

        return result

    def _detect_column_shapes(
        self,
        binary: np.ndarray,
        scale: float
    ) -> List[DetectedColumn]:
        """Detect rectangular shapes that could be columns."""
        columns = []

        # Calculate size thresholds in pixels
        min_size_px = int(self.min_size * scale)
        max_size_px = int(self.max_size * scale)

        # Morphological operations to find solid rectangles
        # Close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if w < min_size_px or h < min_size_px:
                continue
            if w > max_size_px or h > max_size_px:
                continue

            # Aspect ratio filter
            aspect = w / h if h > 0 else 0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Check if it's filled (not just outline)
            roi = closed[y:y+h, x:x+w]
            fill_ratio = np.sum(roi > 0) / (w * h)
            if fill_ratio < 0.3:  # Too hollow
                continue

            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Columns are typically 4-sided
            if len(approx) < 4 or len(approx) > 8:
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

            # Create column
            polygon = [(float(p[0][0]), float(p[0][1])) for p in approx]

            columns.append(DetectedColumn(
                column_id=f"C{i+1:03d}",
                label="",
                bbox=(x, y, w, h),
                center=(cx, cy),
                polygon=polygon,
                size_mm=(w_mm, h_mm),
                confidence=0.6
            ))

        return columns

    def _assign_labels(
        self,
        columns: List[DetectedColumn],
        vector_texts: List[Any],
        image: np.ndarray
    ) -> List[DetectedColumn]:
        """Assign labels to columns from nearby text."""

        # Build list of column-related text
        label_texts = []

        # From vector texts
        if vector_texts:
            for vt in vector_texts:
                text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
                bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))
                label_texts.append({
                    'text': text,
                    'bbox': bbox,
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                })

        # Try OCR if no vector texts
        if not label_texts:
            label_texts = self._ocr_column_labels(image)

        # Match labels to columns
        patterns = self.keywords.get('patterns', [r'C\d+'])

        for col in columns:
            best_match = None
            best_dist = float('inf')

            for lt in label_texts:
                # Check if text matches column pattern
                text = lt['text'].strip().upper()
                is_column_label = False

                for pattern in patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        is_column_label = True
                        break

                if not is_column_label:
                    continue

                # Calculate distance to column
                dist = np.sqrt(
                    (lt['center'][0] - col.center[0])**2 +
                    (lt['center'][1] - col.center[1])**2
                )

                # Label should be close to column (within 2x column size)
                max_dist = max(col.bbox[2], col.bbox[3]) * 3

                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_match = text

            if best_match:
                col.label = best_match
                col.confidence = min(0.9, col.confidence + 0.2)

        # Assign auto-labels to unlabeled columns
        auto_num = 1
        for col in columns:
            if not col.label:
                col.label = f"C{auto_num}"
                col.column_id = f"C{auto_num:03d}"
                auto_num += 1

        return columns

    def _ocr_column_labels(self, image: np.ndarray) -> List[Dict]:
        """Extract column labels using OCR."""
        labels = []

        try:
            import pytesseract

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # OCR with bounding boxes
            data = pytesseract.image_to_data(
                gray, config='--psm 6',
                output_type=pytesseract.Output.DICT
            )

            for i, text in enumerate(data['text']):
                text = text.strip()
                if not text:
                    continue

                # Check if it looks like a column label
                if re.match(r'^C\d+$', text, re.IGNORECASE):
                    x, y, w, h = (data['left'][i], data['top'][i],
                                 data['width'][i], data['height'][i])
                    labels.append({
                        'text': text,
                        'bbox': (x, y, x + w, y + h),
                        'center': (x + w/2, y + h/2)
                    })

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"OCR failed: {e}")

        return labels

    def _cluster_by_size(self, columns: List[DetectedColumn]) -> List[DetectedColumn]:
        """Group columns by similar sizes to identify column types."""
        if not columns:
            return columns

        # Group by size (within 50mm tolerance)
        size_groups = {}
        tolerance = 50  # mm

        for col in columns:
            if col.size_mm:
                w, h = col.size_mm
                size_key = None

                for key in size_groups:
                    kw, kh = key
                    if abs(w - kw) <= tolerance and abs(h - kh) <= tolerance:
                        size_key = key
                        break

                if size_key is None:
                    size_key = (w, h)

                if size_key not in size_groups:
                    size_groups[size_key] = []
                size_groups[size_key].append(col)

        # Log groupings
        for size, cols in size_groups.items():
            logger.debug(f"Column size {size[0]}x{size[1]}mm: {len(cols)} columns")

        return columns

    def detect_from_schedule(
        self,
        schedule_text: str,
        columns: List[DetectedColumn]
    ) -> List[DetectedColumn]:
        """
        Update columns with sizes from schedule.

        Args:
            schedule_text: Text content of column schedule
            columns: Detected columns to update

        Returns:
            Updated columns
        """
        # Parse schedule for size mappings
        # Pattern: C1 - 230x450, C1: 230 x 450, etc.
        pattern = r'(C\d+)\s*[-:]\s*(\d{3})\s*[xX×]\s*(\d{3})'

        mappings = {}
        for match in re.finditer(pattern, schedule_text, re.IGNORECASE):
            label = match.group(1).upper()
            width = int(match.group(2))
            depth = int(match.group(3))
            mappings[label] = (width, depth)

        # Update columns
        for col in columns:
            if col.label.upper() in mappings:
                col.size_mm = mappings[col.label.upper()]
                col.source = "schedule"
                col.confidence = min(0.95, col.confidence + 0.2)

        logger.info(f"Applied {len(mappings)} size mappings from schedule")
        return columns


def detect_columns(
    image: np.ndarray,
    vector_texts: List[Any] = None,
    scale_px_per_mm: float = 0.118
) -> ColumnDetectionResult:
    """
    Convenience function to detect columns.

    Args:
        image: Drawing image
        vector_texts: Text blocks
        scale_px_per_mm: Scale factor

    Returns:
        ColumnDetectionResult
    """
    detector = ColumnDetector()
    return detector.detect(image, vector_texts, scale_px_per_mm)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = detect_columns(img)
            print(f"Detected {len(result.columns)} columns")
            for col in result.columns[:10]:
                print(f"  {col.label}: {col.size_mm}mm at {col.center}")
    else:
        print("Usage: python detect_columns.py <image>")
