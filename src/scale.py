"""
Floor Plan Scale Inference Module
Detects and computes scale from:
1) Explicit scale notes (1:50, 1:100)
2) Dimension strings with extension lines
3) Manual calibration
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import cv2
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ScaleMethod(Enum):
    """Method used to determine scale."""
    SCALE_NOTE = "scale_note"
    DIMENSION = "dimension"
    MANUAL = "manual"
    DEFAULT = "default"
    UNKNOWN = "unknown"


@dataclass
class DimensionMatch:
    """A detected dimension string."""
    text: str
    value_mm: float  # Value in millimeters
    unit: str  # Original unit
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float = 0.8
    pixel_length: Optional[float] = None  # Measured pixel distance


@dataclass
class ScaleResult:
    """Result of scale inference."""
    method: ScaleMethod
    pixels_per_mm: float  # Core scale factor
    confidence: float
    scale_ratio: Optional[int] = None  # e.g., 100 for 1:100
    dimension_used: Optional[DimensionMatch] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def pixels_per_m(self) -> float:
        return self.pixels_per_mm * 1000

    @property
    def mm_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_mm if self.pixels_per_mm > 0 else 0

    def pixel_to_mm(self, pixels: float) -> float:
        """Convert pixels to millimeters."""
        return pixels * self.mm_per_pixel

    def pixel_to_m(self, pixels: float) -> float:
        """Convert pixels to meters."""
        return self.pixel_to_mm(pixels) / 1000

    def sqpixel_to_sqm(self, sq_pixels: float) -> float:
        """Convert square pixels to square meters."""
        mm_per_px = self.mm_per_pixel
        return sq_pixels * (mm_per_px ** 2) / 1_000_000

    def sqpixel_to_sqft(self, sq_pixels: float) -> float:
        """Convert square pixels to square feet."""
        sqm = self.sqpixel_to_sqm(sq_pixels)
        return sqm * 10.7639


class ScaleInferrer:
    """
    Infers scale from floor plan.
    Uses scale notes, dimension strings, or manual calibration.
    """

    # Common scale ratios (drawing : real)
    COMMON_SCALES = [50, 100, 200, 500, 1000]

    # Scale note patterns
    SCALE_NOTE_PATTERNS = [
        r'scale\s*[=:]?\s*1\s*[:/]\s*(\d+)',
        r'1\s*[:/]\s*(\d+)\s*scale',
        r'drawing\s*scale\s*[=:]?\s*1\s*[:/]\s*(\d+)',
        r'at\s+1\s*[:/]\s*(\d+)',
        r'\b1\s*[:/]\s*(\d+)\b',
    ]

    # Dimension patterns (value + unit)
    DIMENSION_PATTERNS = {
        'mm': [
            (r'(\d{3,5})\s*mm\b', 1.0),  # 3000 mm
            (r'(\d{3,5})\s*MM\b', 1.0),
            (r'\b(\d{4,5})\b(?!\s*(?:sqm|sqft|sq\.?m|sq\.?ft|/|:|x|\d))', 1.0),  # 4-5 digit alone = mm
        ],
        'm': [
            (r'(\d+\.?\d*)\s*m\b(?!m)', 1000.0),  # 3.0 m (not mm)
            (r'(\d+\.?\d*)\s*M\b(?!M)', 1000.0),
            (r'(\d+\.\d{2})\b(?!\s*(?:sqm|sqft|sq|/|:))', 1000.0),  # X.XX format = meters
        ],
        'cm': [
            (r'(\d+)\s*cm\b', 10.0),
            (r'(\d+)\s*CM\b', 10.0),
        ],
        'ft_in': [
            (r"(\d+)'\s*-?\s*(\d+)\"", None),  # 9'-0"
            (r"(\d+)\s*ft\s*(\d+)\s*in", None),
            (r"(\d+)'\s*(\d+)", None),  # 9'6
        ]
    }

    def __init__(self, dpi: int = 300, rules_path: Optional[Path] = None):
        """
        Initialize scale inferrer.

        Args:
            dpi: DPI of rendered image
            rules_path: Path to rules YAML (for additional patterns)
        """
        self.dpi = dpi
        self.rules = {}
        if rules_path and rules_path.exists():
            with open(rules_path) as f:
                self.rules = yaml.safe_load(f)

    def infer_scale(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None,
        default_scale: int = 100
    ) -> ScaleResult:
        """
        Infer scale from floor plan.

        Args:
            image: Plan image
            vector_texts: List of VectorText objects from PDF
            default_scale: Default scale ratio if detection fails

        Returns:
            ScaleResult
        """
        # Try scale note detection first
        scale_result = self._detect_scale_note(image, vector_texts)
        if scale_result and scale_result.confidence > 0.7:
            logger.info(f"Scale from note: 1:{scale_result.scale_ratio}")
            return scale_result

        # Try dimension-based detection
        dim_result = self._detect_from_dimensions(image, vector_texts)
        if dim_result and dim_result.confidence > 0.6:
            logger.info(f"Scale from dimension: {dim_result.pixels_per_mm:.4f} px/mm")
            return dim_result

        # Fall back to default
        logger.warning(f"Using default scale 1:{default_scale}")
        return self._create_default_scale(default_scale)

    def _detect_scale_note(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None
    ) -> Optional[ScaleResult]:
        """Detect explicit scale note (1:50, 1:100, etc.)"""

        # Check vector texts first (more reliable)
        if vector_texts:
            for vt in vector_texts:
                text = vt.text.lower().strip()
                for pattern in self.SCALE_NOTE_PATTERNS:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        scale_ratio = int(match.group(1))
                        if scale_ratio in self.COMMON_SCALES:
                            return self._scale_from_ratio(scale_ratio, confidence=0.95)

        # Try OCR if available
        try:
            import pytesseract

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Focus on margins where scale notes usually appear
            h, w = gray.shape
            regions = [
                gray[0:h//6, :],  # Top margin
                gray[h*5//6:, :],  # Bottom margin
                gray[:, 0:w//6],  # Left margin
                gray[:, w*5//6:],  # Right margin
            ]

            for region in regions:
                if region.size == 0:
                    continue

                text = pytesseract.image_to_string(region, config='--psm 6')
                text = text.lower()

                for pattern in self.SCALE_NOTE_PATTERNS:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        scale_ratio = int(match.group(1))
                        if scale_ratio in self.COMMON_SCALES:
                            return self._scale_from_ratio(scale_ratio, confidence=0.85)

        except ImportError:
            logger.debug("pytesseract not available for OCR")
        except Exception as e:
            logger.debug(f"OCR failed: {e}")

        return None

    def _detect_from_dimensions(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None
    ) -> Optional[ScaleResult]:
        """Detect scale from dimension strings and their pixel spans."""

        dimensions = []

        # Extract dimensions from vector texts
        if vector_texts:
            for vt in vector_texts:
                dim = self._parse_dimension(vt.text, vt.bbox)
                if dim:
                    dimensions.append(dim)

        # Try OCR extraction
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

            for i, text in enumerate(data['text']):
                if text.strip():
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    dim = self._parse_dimension(text, bbox)
                    if dim:
                        dimensions.append(dim)

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"OCR dimension extraction failed: {e}")

        if not dimensions:
            return None

        # Find dimension lines and measure pixel lengths
        dimensions = self._measure_dimension_lengths(image, dimensions)

        # Filter dimensions with measured lengths
        measured = [d for d in dimensions if d.pixel_length and d.pixel_length > 20]

        if not measured:
            return None

        # Calculate pixels per mm for each dimension
        scales = []
        for dim in measured:
            px_per_mm = dim.pixel_length / dim.value_mm
            scales.append((px_per_mm, dim))

        # Use median scale
        scales.sort(key=lambda x: x[0])
        median_idx = len(scales) // 2
        best_scale, best_dim = scales[median_idx]

        # Calculate confidence based on consistency
        scale_values = [s[0] for s in scales]
        std_dev = np.std(scale_values) if len(scale_values) > 1 else 0
        mean_scale = np.mean(scale_values)
        cv = std_dev / mean_scale if mean_scale > 0 else 1.0
        confidence = max(0.3, min(0.9, 1.0 - cv))

        return ScaleResult(
            method=ScaleMethod.DIMENSION,
            pixels_per_mm=best_scale,
            confidence=confidence,
            dimension_used=best_dim,
            warnings=[] if confidence > 0.7 else ["scale_uncertain"]
        )

    def _parse_dimension(self, text: str, bbox: Tuple) -> Optional[DimensionMatch]:
        """Parse a dimension string to extract value in mm."""
        text = text.strip()

        # Try each pattern
        for unit, patterns in self.DIMENSION_PATTERNS.items():
            for pattern, multiplier in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    if unit == 'ft_in':
                        # Feet and inches
                        feet = float(match.group(1))
                        inches = float(match.group(2)) if match.group(2) else 0
                        value_mm = (feet * 304.8) + (inches * 25.4)
                    else:
                        value_mm = float(match.group(1)) * multiplier

                    # Sanity check - dimensions typically 500mm to 50000mm
                    if 500 <= value_mm <= 50000:
                        return DimensionMatch(
                            text=text,
                            value_mm=value_mm,
                            unit=unit,
                            bbox=bbox
                        )

        return None

    def _measure_dimension_lengths(
        self,
        image: np.ndarray,
        dimensions: List[DimensionMatch]
    ) -> List[DimensionMatch]:
        """
        Try to measure pixel length of dimension lines.
        Looks for extension lines near dimension text.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=5
        )

        if lines is None:
            return dimensions

        # Group lines by angle (horizontal/vertical)
        h_lines = []  # Nearly horizontal
        v_lines = []  # Nearly vertical

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if angle < 10 or angle > 170:  # Horizontal
                h_lines.append((min(x1, x2), max(x1, x2), (y1+y2)/2, length))
            elif 80 < angle < 100:  # Vertical
                v_lines.append((min(y1, y2), max(y1, y2), (x1+x2)/2, length))

        # For each dimension, find nearby dimension line
        for dim in dimensions:
            x0, y0, x1, y1 = dim.bbox
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            text_width = x1 - x0
            text_height = y1 - y0

            # Determine if dimension is likely horizontal or vertical based on text
            is_horizontal = text_width > text_height * 1.5

            if is_horizontal:
                # Look for horizontal line near text
                candidates = []
                for lx0, lx1, ly, length in h_lines:
                    # Line should be close vertically and span beyond text
                    if abs(ly - cy) < 50 and lx0 < x0 and lx1 > x1:
                        candidates.append((abs(ly - cy), length, lx1 - lx0))

                if candidates:
                    candidates.sort()
                    dim.pixel_length = candidates[0][2]  # Use span of closest line

            else:
                # Look for vertical line near text
                candidates = []
                for ly0, ly1, lx, length in v_lines:
                    if abs(lx - cx) < 50 and ly0 < y0 and ly1 > y1:
                        candidates.append((abs(lx - cx), length, ly1 - ly0))

                if candidates:
                    candidates.sort()
                    dim.pixel_length = candidates[0][2]

        return dimensions

    def _scale_from_ratio(self, ratio: int, confidence: float = 0.9) -> ScaleResult:
        """Create scale result from drawing ratio (1:X)."""
        # At 1:100, 1 drawing unit = 100 real units
        # If DPI is 300, 1 inch = 300 pixels
        # At 1:100 with metric, 1mm on drawing = 100mm real
        # Standard: assume 1 drawing unit = 1mm at scale

        # Convert DPI to pixels per mm (1 inch = 25.4mm)
        drawing_px_per_mm = self.dpi / 25.4

        # Real pixels per mm = drawing_px_per_mm / ratio
        # Because at 1:100, a 1mm real = 0.01mm on drawing
        pixels_per_mm = drawing_px_per_mm / ratio

        return ScaleResult(
            method=ScaleMethod.SCALE_NOTE,
            pixels_per_mm=pixels_per_mm,
            confidence=confidence,
            scale_ratio=ratio
        )

    def _create_default_scale(self, ratio: int = 100) -> ScaleResult:
        """Create default scale result."""
        result = self._scale_from_ratio(ratio, confidence=0.5)
        result.method = ScaleMethod.DEFAULT
        result.warnings = ["scale_uncertain"]
        return result

    def create_manual_scale(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        real_length_mm: float
    ) -> ScaleResult:
        """
        Create scale from manual calibration.

        Args:
            point1: First point (x, y) in pixels
            point2: Second point (x, y) in pixels
            real_length_mm: Real-world length in mm

        Returns:
            ScaleResult
        """
        pixel_length = np.sqrt(
            (point2[0] - point1[0])**2 +
            (point2[1] - point1[1])**2
        )

        pixels_per_mm = pixel_length / real_length_mm

        return ScaleResult(
            method=ScaleMethod.MANUAL,
            pixels_per_mm=pixels_per_mm,
            confidence=0.95,
            dimension_used=DimensionMatch(
                text=f"{real_length_mm}mm (manual)",
                value_mm=real_length_mm,
                unit="mm",
                bbox=(point1[0], point1[1], point2[0], point2[1]),
                pixel_length=pixel_length
            )
        )


def infer_scale(
    image: np.ndarray,
    vector_texts: List[Any] = None,
    dpi: int = 300,
    default_scale: int = 100
) -> ScaleResult:
    """
    Convenience function to infer scale.

    Args:
        image: Floor plan image
        vector_texts: Vector text objects from PDF
        dpi: Image DPI
        default_scale: Default scale ratio

    Returns:
        ScaleResult
    """
    inferrer = ScaleInferrer(dpi=dpi)
    return inferrer.infer_scale(image, vector_texts, default_scale)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = infer_scale(img)
            print(f"Method: {result.method.value}")
            print(f"Pixels per mm: {result.pixels_per_mm:.4f}")
            print(f"Scale ratio: 1:{result.scale_ratio}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Warnings: {result.warnings}")
    else:
        print("Usage: python scale.py <image_file>")
