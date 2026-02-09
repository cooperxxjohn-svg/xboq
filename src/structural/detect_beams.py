"""
Beam Detection Module
Detects beams from structural drawings and builds column-beam graph.
"""

import logging
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
import cv2
from pathlib import Path
import yaml

from .detect_columns import DetectedColumn, ColumnDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class DetectedBeam:
    """A detected beam."""
    beam_id: str
    label: str  # e.g., "B1", "BM1"
    from_column: Optional[str] = None  # column_id
    to_column: Optional[str] = None  # column_id
    start_point: Tuple[float, float] = (0, 0)
    end_point: Tuple[float, float] = (0, 0)
    length_px: float = 0.0
    length_mm: Optional[float] = None
    size_mm: Optional[Tuple[int, int]] = None  # width x depth
    confidence: float = 0.5
    source: str = "detection"  # "detection", "schedule", "callout"
    floor: str = "typical"
    concrete_grade: Optional[str] = None


@dataclass
class ColumnBeamGraph:
    """Graph structure representing column-beam connectivity."""
    nodes: Dict[str, DetectedColumn]  # column_id -> column
    edges: List[Tuple[str, str, DetectedBeam]]  # (from_col, to_col, beam)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)  # column_id -> [connected columns]

    def get_connected_columns(self, column_id: str) -> List[str]:
        """Get all columns connected to a given column via beams."""
        return self.adjacency.get(column_id, [])

    def get_beams_for_column(self, column_id: str) -> List[DetectedBeam]:
        """Get all beams connected to a column."""
        beams = []
        for from_col, to_col, beam in self.edges:
            if from_col == column_id or to_col == column_id:
                beams.append(beam)
        return beams


@dataclass
class BeamDetectionResult:
    """Result of beam detection."""
    beams: List[DetectedBeam]
    graph: Optional[ColumnBeamGraph] = None
    unique_labels: List[str] = field(default_factory=list)
    size_mappings: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    detection_method: str = "line_detection"


class BeamDetector:
    """
    Detects beams in structural drawings.
    Beams appear as thick lines connecting columns.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize detector with configuration."""
        self.config = self._load_config(config_path)
        self.keywords = self._load_keywords()

        # Detection parameters
        self.min_length_mm = self.config.get('min_length_mm', 500)  # Minimum beam length
        self.max_length_mm = self.config.get('max_length_mm', 10000)  # Maximum beam length
        self.min_thickness_px = self.config.get('min_thickness_px', 3)
        self.max_thickness_px = self.config.get('max_thickness_px', 30)
        self.column_snap_distance = self.config.get('column_snap_distance_mm', 100)

    def _load_config(self, path: Optional[Path]) -> Dict:
        """Load configuration from assumptions.yaml."""
        default_path = Path(__file__).parent.parent.parent / "rules" / "assumptions.yaml"
        try:
            with open(path or default_path) as f:
                data = yaml.safe_load(f)
            return data.get('detection', {}).get('beam', {})
        except Exception:
            return {
                'min_length_mm': 500,
                'max_length_mm': 10000,
                'min_thickness_px': 3,
                'max_thickness_px': 30
            }

    def _load_keywords(self) -> Dict:
        """Load keywords from structural_keywords.yaml."""
        keywords_path = Path(__file__).parent.parent.parent / "rules" / "structural_keywords.yaml"
        try:
            with open(keywords_path) as f:
                data = yaml.safe_load(f)
            return data.get('element_labels', {}).get('beam', {})
        except Exception:
            return {
                'patterns': [r'B\d+', r'BM\s*\d+', r'BEAM\s*\d+'],
                'keywords': ['beam', 'bm']
            }

    def detect(
        self,
        image: np.ndarray,
        columns: List[DetectedColumn],
        vector_texts: List[Any] = None,
        scale_px_per_mm: float = 0.118,
        drawing_type: str = "unknown"
    ) -> BeamDetectionResult:
        """
        Detect beams in image.

        Args:
            image: Drawing image
            columns: Previously detected columns
            vector_texts: Text blocks from PDF
            scale_px_per_mm: Scale factor
            drawing_type: Type of drawing for tuning

        Returns:
            BeamDetectionResult with graph
        """
        logger.info("Detecting beams...")

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

        # Detect beam candidates using line detection
        candidates = self._detect_beam_lines(binary, scale_px_per_mm, columns)

        # Snap beams to columns and build graph
        snapped_beams, graph = self._build_column_beam_graph(
            candidates, columns, scale_px_per_mm
        )

        # Assign labels from nearby text
        labeled_beams = self._assign_labels(snapped_beams, vector_texts, image)

        # Create result
        result = BeamDetectionResult(
            beams=labeled_beams,
            graph=graph,
            unique_labels=list(set(b.label for b in labeled_beams if b.label)),
            detection_method="line_detection"
        )

        logger.info(f"Detected {len(result.beams)} beams, "
                   f"{len(result.unique_labels)} unique labels")

        return result

    def _detect_beam_lines(
        self,
        binary: np.ndarray,
        scale: float,
        columns: List[DetectedColumn]
    ) -> List[DetectedBeam]:
        """Detect thick lines that could be beams."""
        beams = []

        # Calculate thresholds
        min_length_px = int(self.min_length_mm * scale)
        max_length_px = int(self.max_length_mm * scale)

        # Create mask excluding column areas
        column_mask = np.zeros_like(binary)
        for col in columns:
            x, y, w, h = col.bbox
            # Expand slightly to avoid edge detection issues
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(binary.shape[1], x + w + pad), min(binary.shape[0], y + h + pad)
            column_mask[y1:y2, x1:x2] = 255

        # Remove columns from binary
        beam_binary = cv2.bitwise_and(binary, cv2.bitwise_not(column_mask))

        # Use morphological operations to isolate beams
        # Beams are typically horizontal or vertical thick lines

        # Horizontal beam detection
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(beam_binary, cv2.MORPH_OPEN, h_kernel)

        # Vertical beam detection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(beam_binary, cv2.MORPH_OPEN, v_kernel)

        # Combine
        combined = cv2.bitwise_or(h_lines, v_lines)

        # Use probabilistic Hough transform
        lines = cv2.HoughLinesP(
            combined,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_length_px,
            maxLineGap=20
        )

        if lines is None:
            return beams

        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            # Calculate length
            length_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length_px < min_length_px or length_px > max_length_px:
                continue

            # Check if line is roughly horizontal or vertical
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            is_horizontal = angle < 0.2 or angle > np.pi - 0.2
            is_vertical = abs(angle - np.pi/2) < 0.2

            if not (is_horizontal or is_vertical):
                # Skip diagonal lines (uncommon for beams in plan)
                continue

            # Convert to mm
            length_mm = length_px / scale

            beams.append(DetectedBeam(
                beam_id=f"B{i+1:03d}",
                label="",
                start_point=(float(x1), float(y1)),
                end_point=(float(x2), float(y2)),
                length_px=length_px,
                length_mm=length_mm,
                confidence=0.5
            ))

        return beams

    def _build_column_beam_graph(
        self,
        beams: List[DetectedBeam],
        columns: List[DetectedColumn],
        scale: float
    ) -> Tuple[List[DetectedBeam], ColumnBeamGraph]:
        """
        Snap beam endpoints to columns and build connectivity graph.
        """
        snap_distance_px = self.column_snap_distance * scale

        # Build column lookup by position
        column_dict = {col.column_id: col for col in columns}

        # Create graph
        graph = ColumnBeamGraph(
            nodes=column_dict,
            edges=[],
            adjacency={col.column_id: [] for col in columns}
        )

        snapped_beams = []

        for beam in beams:
            # Find nearest column to start point
            start_col = self._find_nearest_column(
                beam.start_point, columns, snap_distance_px
            )

            # Find nearest column to end point
            end_col = self._find_nearest_column(
                beam.end_point, columns, snap_distance_px
            )

            if start_col and end_col and start_col != end_col:
                # Valid beam connecting two different columns
                beam.from_column = start_col.column_id
                beam.to_column = end_col.column_id

                # Snap endpoints to column centers
                beam.start_point = start_col.center
                beam.end_point = end_col.center

                # Recalculate length
                beam.length_px = np.sqrt(
                    (beam.end_point[0] - beam.start_point[0])**2 +
                    (beam.end_point[1] - beam.start_point[1])**2
                )
                beam.length_mm = beam.length_px / scale

                beam.confidence = min(0.85, beam.confidence + 0.2)

                # Add to graph
                graph.edges.append((start_col.column_id, end_col.column_id, beam))
                graph.adjacency[start_col.column_id].append(end_col.column_id)
                graph.adjacency[end_col.column_id].append(start_col.column_id)

                snapped_beams.append(beam)

            elif start_col or end_col:
                # Partially connected beam - still useful
                if start_col:
                    beam.from_column = start_col.column_id
                    beam.start_point = start_col.center
                if end_col:
                    beam.to_column = end_col.column_id
                    beam.end_point = end_col.center

                beam.confidence = 0.5
                snapped_beams.append(beam)

        return snapped_beams, graph

    def _find_nearest_column(
        self,
        point: Tuple[float, float],
        columns: List[DetectedColumn],
        max_distance: float
    ) -> Optional[DetectedColumn]:
        """Find the nearest column to a point within max_distance."""
        nearest = None
        min_dist = float('inf')

        for col in columns:
            dist = np.sqrt(
                (point[0] - col.center[0])**2 +
                (point[1] - col.center[1])**2
            )
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = col

        return nearest

    def _assign_labels(
        self,
        beams: List[DetectedBeam],
        vector_texts: List[Any],
        image: np.ndarray
    ) -> List[DetectedBeam]:
        """Assign labels to beams from nearby text."""

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

        # Match labels to beams
        patterns = self.keywords.get('patterns', [r'B\d+', r'BM\d+'])

        for beam in beams:
            # Calculate beam midpoint
            mid_x = (beam.start_point[0] + beam.end_point[0]) / 2
            mid_y = (beam.start_point[1] + beam.end_point[1]) / 2

            best_match = None
            best_dist = float('inf')

            for lt in label_texts:
                text = lt['text'].strip().upper()
                is_beam_label = False

                for pattern in patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        is_beam_label = True
                        break

                if not is_beam_label:
                    continue

                # Distance to beam midpoint
                dist = np.sqrt(
                    (lt['center'][0] - mid_x)**2 +
                    (lt['center'][1] - mid_y)**2
                )

                # Label should be close to beam
                max_dist = beam.length_px * 0.3  # Within 30% of beam length

                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_match = text

            if best_match:
                beam.label = best_match
                beam.confidence = min(0.9, beam.confidence + 0.15)

        # Auto-label unlabeled beams
        auto_num = 1
        for beam in beams:
            if not beam.label:
                beam.label = f"B{auto_num}"
                beam.beam_id = f"B{auto_num:03d}"
                auto_num += 1

        return beams

    def detect_from_schedule(
        self,
        schedule_text: str,
        beams: List[DetectedBeam]
    ) -> List[DetectedBeam]:
        """
        Update beams with sizes from schedule.

        Args:
            schedule_text: Text content of beam schedule
            beams: Detected beams to update

        Returns:
            Updated beams
        """
        # Pattern: B1 - 230x450, B1: 230 x 450, etc.
        pattern = r'(B\d+|BM\d+)\s*[-:]?\s*(\d{3})\s*[xX×]\s*(\d{3})'

        mappings = {}
        for match in re.finditer(pattern, schedule_text, re.IGNORECASE):
            label = match.group(1).upper()
            width = int(match.group(2))
            depth = int(match.group(3))
            mappings[label] = (width, depth)

        # Update beams
        for beam in beams:
            label_upper = beam.label.upper()
            if label_upper in mappings:
                beam.size_mm = mappings[label_upper]
                beam.source = "schedule"
                beam.confidence = min(0.95, beam.confidence + 0.2)

        logger.info(f"Applied {len(mappings)} size mappings from beam schedule")
        return beams


def detect_beams(
    image: np.ndarray,
    columns: List[DetectedColumn],
    vector_texts: List[Any] = None,
    scale_px_per_mm: float = 0.118
) -> BeamDetectionResult:
    """
    Convenience function to detect beams.

    Args:
        image: Drawing image
        columns: Detected columns
        vector_texts: Text blocks
        scale_px_per_mm: Scale factor

    Returns:
        BeamDetectionResult
    """
    detector = BeamDetector()
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

            # Then detect beams
            beam_result = detect_beams(img, col_result.columns)
            print(f"Detected {len(beam_result.beams)} beams")

            for beam in beam_result.beams[:10]:
                print(f"  {beam.label}: {beam.from_column} -> {beam.to_column}, "
                      f"length={beam.length_mm:.0f}mm")
    else:
        print("Usage: python -m src.structural.detect_beams <image>")
