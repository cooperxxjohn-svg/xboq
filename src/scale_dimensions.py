"""
Scale Inference from Dimensions (India-optimized).

Parses dimension strings and extension line geometry to infer scale.
Uses RANSAC/MAD for robust estimation from multiple dimensions.

India metric-first: prioritizes mm, m, cm over ft/in.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class DimensionUnit(Enum):
    """Recognized dimension units."""
    MM = "mm"
    M = "m"
    CM = "cm"
    FT_IN = "ft_in"
    UNKNOWN = "unknown"


@dataclass
class DimensionCandidate:
    """A candidate dimension parsed from text."""
    raw_text: str
    value_mm: float  # Normalized to mm
    unit: DimensionUnit
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    center: Tuple[float, float] = (0, 0)
    confidence: float = 0.8

    def __post_init__(self):
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


@dataclass
class ExtensionLine:
    """A detected dimension extension line pair."""
    start: Tuple[float, float]  # First extension line base
    end: Tuple[float, float]  # Second extension line base
    pixel_length: float  # Distance between bases
    orientation: str  # "horizontal" or "vertical"
    confidence: float = 0.7


@dataclass
class DimensionMatch:
    """A matched dimension with pixel measurement."""
    candidate: DimensionCandidate
    extension: Optional[ExtensionLine]
    pixel_length: float
    pixels_per_mm: float
    confidence: float


@dataclass
class DimensionScaleResult:
    """Result of dimension-based scale inference."""
    pixels_per_mm: float
    confidence: float
    method: str = "dimension_inferred"
    num_dimensions_used: int = 0
    dimensions: List[DimensionMatch] = field(default_factory=list)
    inliers: List[DimensionMatch] = field(default_factory=list)
    outliers: List[DimensionMatch] = field(default_factory=list)
    mad: float = 0.0  # Median Absolute Deviation
    warnings: List[str] = field(default_factory=list)

    @property
    def pixels_per_m(self) -> float:
        return self.pixels_per_mm * 1000

    @property
    def mm_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_mm if self.pixels_per_mm > 0 else 0


class DimensionParser:
    """
    Parses dimension strings from text.
    India metric-first: mm, m, cm prioritized over feet/inches.
    """

    # Dimension patterns ordered by priority (metric first)
    PATTERNS = [
        # Explicit mm
        (r'^(\d{3,5})\s*mm$', DimensionUnit.MM, lambda m: float(m.group(1))),
        (r'^(\d{3,5})\s*MM$', DimensionUnit.MM, lambda m: float(m.group(1))),

        # Explicit m (with decimal)
        (r'^(\d+\.?\d*)\s*m$', DimensionUnit.M, lambda m: float(m.group(1)) * 1000),
        (r'^(\d+\.?\d*)\s*M$', DimensionUnit.M, lambda m: float(m.group(1)) * 1000),

        # Explicit cm
        (r'^(\d+)\s*cm$', DimensionUnit.CM, lambda m: float(m.group(1)) * 10),
        (r'^(\d+)\s*CM$', DimensionUnit.CM, lambda m: float(m.group(1)) * 10),

        # Bare numbers (context-dependent)
        # 4-5 digits standalone = likely mm (3000, 4500, etc.)
        (r'^(\d{4,5})$', DimensionUnit.MM, lambda m: float(m.group(1))),

        # 3 digits alone could be mm (900, 600) - common door/window widths
        (r'^(\d{3})$', DimensionUnit.MM, lambda m: float(m.group(1))),

        # X.XX format = likely meters (3.00, 4.50)
        (r'^(\d+\.\d{2})$', DimensionUnit.M, lambda m: float(m.group(1)) * 1000),

        # Feet-inches (lower priority for India)
        (r"^(\d+)'\s*-?\s*(\d+)\"?$", DimensionUnit.FT_IN,
         lambda m: float(m.group(1)) * 304.8 + float(m.group(2)) * 25.4),
        (r"^(\d+)\s*ft\s*(\d+)\s*in$", DimensionUnit.FT_IN,
         lambda m: float(m.group(1)) * 304.8 + float(m.group(2)) * 25.4),
        (r"^(\d+)'(\d+)$", DimensionUnit.FT_IN,
         lambda m: float(m.group(1)) * 304.8 + float(m.group(2)) * 25.4),
    ]

    # Valid range for architectural dimensions (mm)
    MIN_DIMENSION_MM = 100  # 10cm minimum
    MAX_DIMENSION_MM = 50000  # 50m maximum

    def parse(self, text: str, bbox: Tuple[float, float, float, float]) -> Optional[DimensionCandidate]:
        """
        Parse a text string as a dimension.

        Args:
            text: Text to parse
            bbox: Bounding box (x0, y0, x1, y1)

        Returns:
            DimensionCandidate or None
        """
        text = text.strip()

        # Skip if too long or contains spaces in middle (likely not a dimension)
        if len(text) > 15:
            return None

        # Skip common non-dimension patterns
        skip_patterns = [
            r'^\d{1,2}[:/]\d{2,4}$',  # Scale notation 1:100
            r'^\d+x\d+$',  # Size notation like 300x300
            r'^[A-Z]\d+$',  # Grid references like A1, B2
            r'^\d+\.\d{3,}$',  # Too many decimals
            r'^[\+\-]?\d+\.\d+e',  # Scientific notation
        ]

        for pattern in skip_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return None

        # Try each pattern
        for pattern, unit, converter in self.PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value_mm = converter(match)

                    # Validate range
                    if self.MIN_DIMENSION_MM <= value_mm <= self.MAX_DIMENSION_MM:
                        # Confidence based on how explicit the unit is
                        confidence = 0.9 if unit in (DimensionUnit.MM, DimensionUnit.M) else 0.7

                        return DimensionCandidate(
                            raw_text=text,
                            value_mm=value_mm,
                            unit=unit,
                            bbox=bbox,
                            confidence=confidence
                        )
                except (ValueError, TypeError):
                    pass

        return None

    def parse_all(
        self,
        vector_texts: List[Any],
        ocr_data: Optional[Dict] = None
    ) -> List[DimensionCandidate]:
        """
        Parse dimensions from all text sources.

        Args:
            vector_texts: Vector text objects from PDF
            ocr_data: Optional OCR data dict with boxes

        Returns:
            List of dimension candidates
        """
        candidates = []
        seen_texts = set()

        # From vector texts
        for vt in vector_texts:
            if hasattr(vt, 'text'):
                text = vt.text.strip()
                bbox = vt.bbox
            elif isinstance(vt, dict):
                text = vt.get('text', '').strip()
                bbox = vt.get('bbox', (0, 0, 0, 0))
            else:
                continue

            if text and text not in seen_texts:
                seen_texts.add(text)
                candidate = self.parse(text, bbox)
                if candidate:
                    candidates.append(candidate)

        # From OCR data
        if ocr_data and 'text' in ocr_data:
            for i, text in enumerate(ocr_data['text']):
                text = text.strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    bbox = (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    )
                    candidate = self.parse(text, bbox)
                    if candidate:
                        candidates.append(candidate)

        logger.info(f"Parsed {len(candidates)} dimension candidates")
        return candidates


class ExtensionLineDetector:
    """
    Detects dimension extension line geometry.
    Looks for the characteristic pattern:
    - Two parallel short lines (extension lines)
    - Connected by a dimension line with arrows/ticks
    """

    def __init__(self):
        self.min_line_length = 20
        self.max_extension_length = 100
        self.angle_tolerance = 10  # degrees

    def detect(
        self,
        image: np.ndarray,
        candidates: List[DimensionCandidate]
    ) -> List[Tuple[DimensionCandidate, ExtensionLine]]:
        """
        Detect extension lines near dimension text.

        Args:
            image: Grayscale or BGR image
            candidates: Dimension candidates with positions

        Returns:
            List of (candidate, extension_line) pairs
        """
        if image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Detect all lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=30,
            minLineLength=self.min_line_length,
            maxLineGap=5
        )

        if lines is None:
            return []

        # Classify lines by orientation
        h_lines = []  # Horizontal
        v_lines = []  # Vertical

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

            if abs(angle) < self.angle_tolerance or abs(angle) > 180 - self.angle_tolerance:
                h_lines.append({
                    'x1': min(x1, x2), 'x2': max(x1, x2),
                    'y': (y1 + y2) / 2,
                    'length': abs(x2 - x1)
                })
            elif abs(abs(angle) - 90) < self.angle_tolerance:
                v_lines.append({
                    'y1': min(y1, y2), 'y2': max(y1, y2),
                    'x': (x1 + x2) / 2,
                    'length': abs(y2 - y1)
                })

        # Match candidates to extension lines
        results = []

        for candidate in candidates:
            cx, cy = candidate.center
            bbox = candidate.bbox
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Determine likely orientation from text shape
            is_horizontal_text = text_width > text_height * 1.2

            if is_horizontal_text:
                # Look for horizontal dimension line
                extension = self._find_horizontal_extension(
                    cx, cy, bbox, h_lines, text_width
                )
            else:
                # Look for vertical dimension line
                extension = self._find_vertical_extension(
                    cx, cy, bbox, v_lines, text_height
                )

            if extension:
                results.append((candidate, extension))

        logger.info(f"Matched {len(results)} dimensions to extension lines")
        return results

    def _find_horizontal_extension(
        self,
        cx: float, cy: float,
        bbox: Tuple[float, float, float, float],
        h_lines: List[Dict],
        text_width: float
    ) -> Optional[ExtensionLine]:
        """Find horizontal dimension line near text."""
        # Look for lines that:
        # 1. Are at similar Y as text
        # 2. Span beyond text width
        # 3. Could be dimension line

        candidates = []
        search_margin = 50

        for line in h_lines:
            y_dist = abs(line['y'] - cy)
            if y_dist > search_margin:
                continue

            # Line should span beyond text
            if line['x1'] < bbox[0] and line['x2'] > bbox[2]:
                span = line['x2'] - line['x1']
                # Prefer lines that extend reasonably beyond text
                if span > text_width * 1.2:
                    candidates.append({
                        'line': line,
                        'y_dist': y_dist,
                        'span': span
                    })

        if not candidates:
            return None

        # Pick best candidate (closest Y, reasonable span)
        best = min(candidates, key=lambda c: c['y_dist'] + abs(c['span'] - text_width * 1.5) * 0.1)

        line = best['line']
        return ExtensionLine(
            start=(line['x1'], line['y']),
            end=(line['x2'], line['y']),
            pixel_length=line['x2'] - line['x1'],
            orientation="horizontal",
            confidence=0.8 - best['y_dist'] / search_margin * 0.3
        )

    def _find_vertical_extension(
        self,
        cx: float, cy: float,
        bbox: Tuple[float, float, float, float],
        v_lines: List[Dict],
        text_height: float
    ) -> Optional[ExtensionLine]:
        """Find vertical dimension line near text."""
        candidates = []
        search_margin = 50

        for line in v_lines:
            x_dist = abs(line['x'] - cx)
            if x_dist > search_margin:
                continue

            if line['y1'] < bbox[1] and line['y2'] > bbox[3]:
                span = line['y2'] - line['y1']
                if span > text_height * 1.2:
                    candidates.append({
                        'line': line,
                        'x_dist': x_dist,
                        'span': span
                    })

        if not candidates:
            return None

        best = min(candidates, key=lambda c: c['x_dist'])

        line = best['line']
        return ExtensionLine(
            start=(line['x'], line['y1']),
            end=(line['x'], line['y2']),
            pixel_length=line['y2'] - line['y1'],
            orientation="vertical",
            confidence=0.8 - best['x_dist'] / search_margin * 0.3
        )


class RANSACScaleEstimator:
    """
    Robust scale estimation using RANSAC/MAD.
    Handles outliers from misdetected dimensions.
    """

    def __init__(
        self,
        inlier_threshold: float = 0.15,  # 15% deviation threshold
        min_inliers: int = 2,
        max_iterations: int = 100
    ):
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations

    def estimate(
        self,
        matches: List[DimensionMatch]
    ) -> DimensionScaleResult:
        """
        Estimate scale from dimension matches using RANSAC.

        Args:
            matches: List of dimension matches with pixel lengths

        Returns:
            DimensionScaleResult
        """
        if not matches:
            return DimensionScaleResult(
                pixels_per_mm=0,
                confidence=0,
                warnings=["No dimension matches"]
            )

        if len(matches) == 1:
            # Single match - use directly
            m = matches[0]
            return DimensionScaleResult(
                pixels_per_mm=m.pixels_per_mm,
                confidence=m.confidence * 0.7,  # Lower confidence for single
                num_dimensions_used=1,
                dimensions=matches,
                inliers=matches,
                method="single_dimension",
                warnings=["Only one dimension available"]
            )

        # Extract scale values
        scales = np.array([m.pixels_per_mm for m in matches])
        confidences = np.array([m.confidence for m in matches])

        # RANSAC with weighted voting
        best_inliers = []
        best_scale = 0
        best_score = 0

        for i in range(min(self.max_iterations, len(matches))):
            # Sample a scale
            if i < len(matches):
                sample_scale = scales[i]
            else:
                sample_scale = np.random.choice(scales)

            # Find inliers
            deviations = np.abs(scales - sample_scale) / sample_scale
            inlier_mask = deviations < self.inlier_threshold

            if np.sum(inlier_mask) >= self.min_inliers:
                # Score = weighted sum of inliers
                score = np.sum(confidences[inlier_mask])

                if score > best_score:
                    best_score = score
                    best_inliers = inlier_mask
                    # Weighted median of inliers
                    inlier_scales = scales[inlier_mask]
                    inlier_weights = confidences[inlier_mask]
                    best_scale = self._weighted_median(inlier_scales, inlier_weights)

        if best_scale == 0:
            # Fallback to weighted median of all
            best_scale = self._weighted_median(scales, confidences)
            best_inliers = np.ones(len(matches), dtype=bool)

        # Calculate MAD for confidence
        mad = np.median(np.abs(scales - best_scale))
        relative_mad = mad / best_scale if best_scale > 0 else 1.0

        # Confidence based on consistency
        num_inliers = np.sum(best_inliers)
        consistency = num_inliers / len(matches)
        confidence = min(0.95, 0.5 + consistency * 0.3 - relative_mad * 0.5)
        confidence = max(0.3, confidence)

        # Separate inliers and outliers
        inliers = [m for i, m in enumerate(matches) if best_inliers[i]]
        outliers = [m for i, m in enumerate(matches) if not best_inliers[i]]

        warnings = []
        if len(outliers) > len(inliers):
            warnings.append("Many outlier dimensions - check scale")
        if relative_mad > 0.2:
            warnings.append(f"High variance in dimensions (MAD={relative_mad:.1%})")

        return DimensionScaleResult(
            pixels_per_mm=best_scale,
            confidence=round(confidence, 2),
            method="ransac_dimension",
            num_dimensions_used=num_inliers,
            dimensions=matches,
            inliers=inliers,
            outliers=outliers,
            mad=mad,
            warnings=warnings
        )

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted median."""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] / 2

        return float(sorted_values[cumsum >= cutoff][0])


class DimensionScaleInferrer:
    """
    Main class for dimension-based scale inference.
    Combines parsing, geometry detection, and RANSAC estimation.
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.parser = DimensionParser()
        self.line_detector = ExtensionLineDetector()
        self.estimator = RANSACScaleEstimator()

    def infer_scale(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None,
        fallback_method: str = "median"
    ) -> DimensionScaleResult:
        """
        Infer scale from dimensions in image.

        Args:
            image: Floor plan image
            vector_texts: Text objects from PDF
            fallback_method: Method for estimation if geometry fails

        Returns:
            DimensionScaleResult
        """
        vector_texts = vector_texts or []

        # Step 1: Parse dimension candidates
        candidates = self.parser.parse_all(vector_texts)

        if not candidates:
            # Try OCR
            ocr_data = self._run_ocr(image)
            if ocr_data:
                candidates = self.parser.parse_all([], ocr_data)

        if not candidates:
            return DimensionScaleResult(
                pixels_per_mm=0,
                confidence=0,
                method="failed",
                warnings=["No dimensions found in drawing"]
            )

        # Step 2: Detect extension lines
        matched_pairs = self.line_detector.detect(image, candidates)

        # Step 3: Calculate pixel lengths
        matches = []

        for candidate, extension in matched_pairs:
            if extension and extension.pixel_length > 0:
                px_per_mm = extension.pixel_length / candidate.value_mm
                conf = candidate.confidence * extension.confidence

                matches.append(DimensionMatch(
                    candidate=candidate,
                    extension=extension,
                    pixel_length=extension.pixel_length,
                    pixels_per_mm=px_per_mm,
                    confidence=conf
                ))

        # If no geometry matches, try heuristic based on text box size
        if not matches:
            logger.info("No extension line matches, using text heuristics")
            matches = self._estimate_from_text_boxes(candidates, image.shape)

        if not matches:
            return DimensionScaleResult(
                pixels_per_mm=0,
                confidence=0,
                method="failed",
                warnings=["Could not measure any dimensions"]
            )

        # Step 4: RANSAC estimation
        result = self.estimator.estimate(matches)

        return result

    def _run_ocr(self, image: np.ndarray) -> Optional[Dict]:
        """Run OCR on image."""
        try:
            import pytesseract

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            data = pytesseract.image_to_data(
                gray,
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )
            return data
        except ImportError:
            logger.debug("pytesseract not available")
            return None
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return None

    def _estimate_from_text_boxes(
        self,
        candidates: List[DimensionCandidate],
        image_shape: Tuple
    ) -> List[DimensionMatch]:
        """
        Estimate pixel length from text box position heuristics.
        Assumes dimension text is positioned along the dimension.
        """
        matches = []
        h, w = image_shape[:2]

        # Group candidates that might be related
        for candidate in candidates:
            # Rough estimate: dimension text width/height correlates with measured distance
            # This is a fallback heuristic
            bbox = candidate.bbox
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Assume the dimension spans roughly the image width/height
            # Scale text size to estimated span
            is_horizontal = text_width > text_height

            if is_horizontal:
                # Estimate span as ~80% of image width for large dimensions
                estimated_span = min(w * 0.3, text_width * 5)
            else:
                estimated_span = min(h * 0.3, text_height * 5)

            if estimated_span > 50:
                px_per_mm = estimated_span / candidate.value_mm

                matches.append(DimensionMatch(
                    candidate=candidate,
                    extension=None,
                    pixel_length=estimated_span,
                    pixels_per_mm=px_per_mm,
                    confidence=candidate.confidence * 0.4  # Lower confidence
                ))

        return matches

    def save_debug_image(
        self,
        image: np.ndarray,
        result: DimensionScaleResult,
        output_path: Path
    ) -> None:
        """Save debug image showing detected dimensions."""
        debug_img = image.copy()

        # Draw dimension matches
        for match in result.inliers:
            c = match.candidate
            ext = match.extension

            # Draw text bounding box (green for inliers)
            x0, y0, x1, y1 = [int(v) for v in c.bbox]
            cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # Label
            label = f"{c.raw_text} ({match.pixels_per_mm:.3f}px/mm)"
            cv2.putText(debug_img, label, (x0, y0 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw extension line if available
            if ext:
                sx, sy = [int(v) for v in ext.start]
                ex, ey = [int(v) for v in ext.end]
                cv2.line(debug_img, (sx, sy), (ex, ey), (255, 0, 0), 1)

        # Draw outliers in red
        for match in result.outliers:
            c = match.candidate
            x0, y0, x1, y1 = [int(v) for v in c.bbox]
            cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Add summary text
        summary = f"Scale: {result.pixels_per_mm:.4f} px/mm | Conf: {result.confidence:.0%} | Used: {result.num_dimensions_used}"
        cv2.rectangle(debug_img, (0, 0), (500, 30), (255, 255, 255), -1)
        cv2.putText(debug_img, summary, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), debug_img)
        logger.info(f"Saved dimension debug image: {output_path}")


def infer_scale_from_dimensions(
    image: np.ndarray,
    vector_texts: List[Any] = None,
    dpi: int = 300,
    debug_path: Optional[Path] = None
) -> DimensionScaleResult:
    """
    Convenience function to infer scale from dimensions.

    Args:
        image: Floor plan image
        vector_texts: Text objects from PDF
        dpi: Image DPI
        debug_path: Optional path to save debug image

    Returns:
        DimensionScaleResult
    """
    inferrer = DimensionScaleInferrer(dpi=dpi)
    result = inferrer.infer_scale(image, vector_texts)

    if debug_path:
        inferrer.save_debug_image(image, result, debug_path)

    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        img = cv2.imread(str(img_path))

        if img is not None:
            debug_path = Path("./out/debug/scale_dim_candidates.png")
            result = infer_scale_from_dimensions(img, debug_path=debug_path)

            print(f"\nScale Inference Result:")
            print(f"  Method: {result.method}")
            print(f"  Pixels per mm: {result.pixels_per_mm:.4f}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Dimensions used: {result.num_dimensions_used}")
            print(f"  MAD: {result.mad:.4f}")

            if result.warnings:
                print(f"  Warnings: {result.warnings}")

            if result.inliers:
                print(f"\n  Inlier dimensions:")
                for m in result.inliers[:5]:
                    print(f"    {m.candidate.raw_text}: {m.pixels_per_mm:.4f} px/mm")
        else:
            print(f"Could not read image: {img_path}")
    else:
        print("Usage: python scale_dimensions.py <image_file>")
