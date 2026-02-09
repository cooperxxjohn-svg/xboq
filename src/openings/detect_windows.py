"""
Window Detection Module - Multi-Signal Approach
Detects windows using:
1. Double parallel lines across wall thickness
2. Wall gaps with smaller size than doors
3. Window symbol patterns (rectangles within walls)
4. OCR tag association

Indian conventions: W, W1, W2, V, VD, LW, TW, KW
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class WindowCandidate:
    """Intermediate window candidate before final classification."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    width_px: float
    height_px: float
    signals: Set[str] = field(default_factory=set)  # "parallel_lines", "gap", "symbol", "tag"
    parallel_info: Optional[Dict] = None
    gap_info: Optional[Dict] = None
    symbol_info: Optional[Dict] = None
    tag: Optional[str] = None
    opening_type: str = "window"  # window, ventilator
    confidence: float = 0.0


@dataclass
class DetectedWindow:
    """Final detected window with all metadata."""
    id: str  # W-01, W-02, V-01
    type: str = "window"  # window, ventilator, picture_window
    tag: Optional[str] = None  # W, W1, W2, V, etc.
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[int, int] = (0, 0)
    width_px: float = 0.0
    height_px: float = 0.0
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    sill_height_m: Optional[float] = None
    wall_segment_id: Optional[str] = None
    room_left_id: Optional[str] = None
    room_right_id: Optional[str] = None
    confidence: float = 0.0
    signals: Set[str] = field(default_factory=set)
    source: Dict[str, str] = field(default_factory=dict)


class WindowDetector:
    """
    Multi-signal window detector for architectural floor plans.

    Detection signals:
    1. Parallel lines - Double lines across wall indicating glazing
    2. Wall gaps - Smaller openings in walls (narrower than doors)
    3. Window symbols - Characteristic rectangles or patterns
    4. Tags - OCR-detected labels like W, W1, V, etc.
    """

    def __init__(
        self,
        min_width_px: int = 15,
        max_width_px: int = 200,
        min_height_px: int = 5,
        max_height_px: int = 50,
        parallel_line_gap_px: int = 5,
        ventilator_max_width_px: int = 40,
    ):
        self.min_width_px = min_width_px
        self.max_width_px = max_width_px
        self.min_height_px = min_height_px
        self.max_height_px = max_height_px
        self.parallel_line_gap_px = parallel_line_gap_px
        self.ventilator_max_width_px = ventilator_max_width_px

        # Window tag patterns (Indian conventions)
        self.window_tags = {
            "W", "W1", "W2", "W3", "W4", "W5",
            "LW", "TW", "KW", "PW",
            "WINDOW", "LARGE", "TOILET", "KITCHEN",
        }
        self.ventilator_tags = {
            "V", "V1", "V2", "VD", "VENT", "VENTILATOR",
        }

    def detect(
        self,
        image: np.ndarray,
        wall_mask: np.ndarray,
        wall_mask_closed: Optional[np.ndarray] = None,
        texts: Optional[List[Dict]] = None,
        scale_px_per_mm: Optional[float] = None,
    ) -> List[DetectedWindow]:
        """
        Detect windows using multiple signals.

        Args:
            image: Original grayscale or color image
            wall_mask: Binary wall mask (walls = 255)
            wall_mask_closed: Wall mask with gaps closed
            texts: List of text boxes with {"text", "bbox", "confidence"}
            scale_px_per_mm: Scale for real-world size computation

        Returns:
            List of DetectedWindow objects
        """
        logger.info("Starting multi-signal window detection")

        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        candidates: List[WindowCandidate] = []

        # Signal 1: Detect parallel lines (window glazing pattern)
        parallel_candidates = self._detect_parallel_lines(gray, wall_mask)
        logger.info(f"Found {len(parallel_candidates)} parallel line candidates")

        # Signal 2: Detect wall gaps that look like windows
        if wall_mask_closed is not None:
            gap_candidates = self._detect_window_gaps(wall_mask, wall_mask_closed)
            logger.info(f"Found {len(gap_candidates)} window gap candidates")
        else:
            gap_candidates = []

        # Signal 3: Detect window symbol patterns
        symbol_candidates = self._detect_window_symbols(gray, wall_mask)
        logger.info(f"Found {len(symbol_candidates)} symbol candidates")

        # Merge candidates from all signals
        candidates = self._merge_candidates(parallel_candidates, gap_candidates, symbol_candidates)
        logger.info(f"Merged into {len(candidates)} unique candidates")

        # Signal 4: Associate tags
        if texts:
            self._associate_tags(candidates, texts)

        # Score and filter candidates
        candidates = self._score_candidates(candidates)
        candidates = [c for c in candidates if c.confidence >= 0.3]

        # Convert to final DetectedWindow objects
        windows = self._finalize_windows(candidates, scale_px_per_mm)
        logger.info(f"Detected {len(windows)} windows/ventilators")

        return windows

    def _detect_parallel_lines(
        self,
        gray: np.ndarray,
        wall_mask: np.ndarray,
    ) -> List[WindowCandidate]:
        """
        Detect window patterns using parallel line detection.

        Windows in floor plans often show as double parallel lines
        (representing glass panes) crossing the wall thickness.
        """
        candidates = []

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=self.min_width_px,
            maxLineGap=5,
        )

        if lines is None:
            return candidates

        # Group parallel lines
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = math.atan2(y2 - y1, x2 - x1)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            line_list.append({
                "points": (x1, y1, x2, y2),
                "length": length,
                "angle": angle,
                "center": center,
            })

        # Find pairs of parallel lines
        used = set()
        for i, line1 in enumerate(line_list):
            if i in used:
                continue

            for j, line2 in enumerate(line_list):
                if j <= i or j in used:
                    continue

                # Check if parallel (similar angle)
                angle_diff = abs(line1["angle"] - line2["angle"])
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff

                if angle_diff < 0.1:  # ~5.7 degrees tolerance
                    # Check if close together but not overlapping
                    dist = self._point_to_line_distance(
                        line2["center"],
                        line1["points"][:2],
                        line1["points"][2:4],
                    )

                    # Window pattern: parallel lines close together
                    if self.min_height_px <= dist <= self.max_height_px:
                        # Check if similar length
                        length_ratio = min(line1["length"], line2["length"]) / max(line1["length"], line2["length"])

                        if length_ratio > 0.7:
                            # Check if on/near a wall
                            cx = int((line1["center"][0] + line2["center"][0]) / 2)
                            cy = int((line1["center"][1] + line2["center"][1]) / 2)

                            # Look for wall nearby
                            search_radius = int(max(dist, line1["length"]) / 2) + 10
                            y1_s = max(0, cy - search_radius)
                            y2_s = min(wall_mask.shape[0], cy + search_radius)
                            x1_s = max(0, cx - search_radius)
                            x2_s = min(wall_mask.shape[1], cx + search_radius)

                            wall_nearby = np.sum(wall_mask[y1_s:y2_s, x1_s:x2_s]) > 0

                            if wall_nearby:
                                # Compute bounding box
                                all_x = [line1["points"][0], line1["points"][2],
                                        line2["points"][0], line2["points"][2]]
                                all_y = [line1["points"][1], line1["points"][3],
                                        line2["points"][1], line2["points"][3]]

                                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
                                width = max(all_x) - min(all_x)
                                height = max(all_y) - min(all_y)

                                candidate = WindowCandidate(
                                    bbox=bbox,
                                    center=(cx, cy),
                                    width_px=max(width, height),
                                    height_px=min(width, height),
                                    signals={"parallel_lines"},
                                    parallel_info={
                                        "line1": line1,
                                        "line2": line2,
                                        "gap": dist,
                                    },
                                )
                                candidates.append(candidate)
                                used.add(i)
                                used.add(j)

        return candidates

    def _point_to_line_distance(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
    ) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return numerator / denominator if denominator > 0 else 0

    def _detect_window_gaps(
        self,
        wall_mask: np.ndarray,
        wall_mask_closed: np.ndarray,
    ) -> List[WindowCandidate]:
        """
        Detect windows by finding smaller gaps in walls.

        Windows typically have narrower gaps than doors.
        """
        candidates = []

        # Find difference (gaps)
        gaps = cv2.subtract(wall_mask_closed, wall_mask)

        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        gaps = cv2.morphologyEx(gaps, cv2.MORPH_OPEN, kernel)

        # Find gap regions
        contours, _ = cv2.findContours(gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            gap_width = max(w, h)
            gap_height = min(w, h)

            # Windows are typically smaller gaps than doors
            # Also check aspect ratio (windows tend to be wider than tall in plan)
            if self.min_width_px <= gap_width <= self.max_width_px:
                if gap_height <= self.max_height_px or gap_width / max(gap_height, 1) > 2:
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Determine if ventilator (smaller)
                    opening_type = "window"
                    if gap_width <= self.ventilator_max_width_px:
                        opening_type = "ventilator"

                    candidate = WindowCandidate(
                        bbox=(x, y, x + w, y + h),
                        center=(center_x, center_y),
                        width_px=gap_width,
                        height_px=gap_height,
                        signals={"gap"},
                        gap_info={
                            "width": gap_width,
                            "height": gap_height,
                            "area": cv2.contourArea(contour),
                            "orientation": "horizontal" if w > h else "vertical",
                        },
                        opening_type=opening_type,
                    )
                    candidates.append(candidate)

        return candidates

    def _detect_window_symbols(
        self,
        gray: np.ndarray,
        wall_mask: np.ndarray,
    ) -> List[WindowCandidate]:
        """
        Detect window symbols (rectangles within walls).

        Windows often appear as small rectangles or patterns
        embedded in wall lines.
        """
        candidates = []

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find regions that are ON the wall (intersection with wall)
        on_wall = cv2.bitwise_and(binary, wall_mask)

        # Dilate wall mask to find "within wall" regions
        wall_dilated = cv2.dilate(wall_mask, np.ones((5, 5), np.uint8), iterations=1)
        wall_eroded = cv2.erode(wall_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Window symbols are often in the wall band
        wall_band = cv2.subtract(wall_dilated, wall_eroded)
        symbols_in_wall = cv2.bitwise_and(binary, wall_band)

        # Find rectangular contours
        contours, _ = cv2.findContours(symbols_in_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check if roughly rectangular
            rect_area = w * h
            if area / rect_area > 0.6:  # Reasonably rectangular
                # Check dimensions
                width = max(w, h)
                height = min(w, h)

                if self.min_width_px <= width <= self.max_width_px:
                    candidate = WindowCandidate(
                        bbox=(x, y, x + w, y + h),
                        center=(x + w // 2, y + h // 2),
                        width_px=width,
                        height_px=height,
                        signals={"symbol"},
                        symbol_info={
                            "area": area,
                            "rectangularity": area / rect_area,
                        },
                    )
                    candidates.append(candidate)

        return candidates

    def _merge_candidates(
        self,
        parallel_candidates: List[WindowCandidate],
        gap_candidates: List[WindowCandidate],
        symbol_candidates: List[WindowCandidate],
    ) -> List[WindowCandidate]:
        """Merge candidates from different signals if they overlap."""
        all_candidates = parallel_candidates + gap_candidates + symbol_candidates

        if not all_candidates:
            return []

        merged = []
        used = set()

        all_candidates.sort(key=lambda c: len(c.signals), reverse=True)

        for i, c1 in enumerate(all_candidates):
            if i in used:
                continue

            overlapping = [c1]

            for j, c2 in enumerate(all_candidates):
                if j <= i or j in used:
                    continue

                iou = self._compute_iou(c1.bbox, c2.bbox)
                if iou > 0.3:
                    overlapping.append(c2)
                    used.add(j)

            merged_candidate = self._merge_overlapping(overlapping)
            merged.append(merged_candidate)
            used.add(i)

        return merged

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_overlapping(
        self,
        candidates: List[WindowCandidate],
    ) -> WindowCandidate:
        """Merge overlapping candidates."""
        if len(candidates) == 1:
            return candidates[0]

        all_signals = set()
        for c in candidates:
            all_signals.update(c.signals)

        centers = [c.center for c in candidates]
        avg_center = (
            int(sum(c[0] for c in centers) / len(centers)),
            int(sum(c[1] for c in centers) / len(centers)),
        )

        avg_width = sum(c.width_px for c in candidates) / len(candidates)
        avg_height = sum(c.height_px for c in candidates) / len(candidates)

        x1 = min(c.bbox[0] for c in candidates)
        y1 = min(c.bbox[1] for c in candidates)
        x2 = max(c.bbox[2] for c in candidates)
        y2 = max(c.bbox[3] for c in candidates)

        parallel_info = next((c.parallel_info for c in candidates if c.parallel_info), None)
        gap_info = next((c.gap_info for c in candidates if c.gap_info), None)
        symbol_info = next((c.symbol_info for c in candidates if c.symbol_info), None)
        tag = next((c.tag for c in candidates if c.tag), None)
        opening_type = next((c.opening_type for c in candidates if c.opening_type != "window"), "window")

        return WindowCandidate(
            bbox=(x1, y1, x2, y2),
            center=avg_center,
            width_px=avg_width,
            height_px=avg_height,
            signals=all_signals,
            parallel_info=parallel_info,
            gap_info=gap_info,
            symbol_info=symbol_info,
            tag=tag,
            opening_type=opening_type,
        )

    def _associate_tags(
        self,
        candidates: List[WindowCandidate],
        texts: List[Dict],
    ) -> None:
        """Associate OCR-detected tags with window candidates."""
        for text_box in texts:
            text = text_box.get("text", "").strip().upper()

            # Check if window or ventilator tag
            is_window = text in self.window_tags or (len(text) <= 3 and text.startswith("W"))
            is_ventilator = text in self.ventilator_tags or text == "V"

            if is_window or is_ventilator:
                bbox = text_box.get("bbox", [0, 0, 0, 0])
                text_center = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                )

                min_dist = float("inf")
                nearest = None

                for candidate in candidates:
                    dist = math.sqrt(
                        (text_center[0] - candidate.center[0]) ** 2 +
                        (text_center[1] - candidate.center[1]) ** 2
                    )

                    if dist < min_dist and dist < max(candidate.width_px, candidate.height_px) * 3:
                        min_dist = dist
                        nearest = candidate

                if nearest:
                    nearest.tag = text
                    nearest.signals.add("tag")
                    if is_ventilator:
                        nearest.opening_type = "ventilator"

    def _score_candidates(
        self,
        candidates: List[WindowCandidate],
    ) -> List[WindowCandidate]:
        """Score candidates based on signal strength."""
        for candidate in candidates:
            score = 0.0

            signal_weights = {
                "parallel_lines": 0.35,
                "gap": 0.25,
                "symbol": 0.20,
                "tag": 0.30,
            }

            for signal in candidate.signals:
                score += signal_weights.get(signal, 0.1)

            if len(candidate.signals) >= 2:
                score += 0.15
            if len(candidate.signals) >= 3:
                score += 0.10

            candidate.confidence = min(1.0, score)

        return candidates

    def _finalize_windows(
        self,
        candidates: List[WindowCandidate],
        scale_px_per_mm: Optional[float],
    ) -> List[DetectedWindow]:
        """Convert candidates to final DetectedWindow objects."""
        windows = []
        window_count = 0
        ventilator_count = 0

        for candidate in candidates:
            # Generate ID
            if candidate.opening_type == "ventilator":
                ventilator_count += 1
                window_id = f"V-{ventilator_count:02d}"
            else:
                window_count += 1
                window_id = f"W-{window_count:02d}"

            # Determine window type
            window_type = candidate.opening_type
            if candidate.tag:
                tag_upper = candidate.tag.upper()
                if "LW" in tag_upper or "LARGE" in tag_upper:
                    window_type = "large_window"
                elif "TW" in tag_upper or "TOILET" in tag_upper:
                    window_type = "toilet_window"
                elif "KW" in tag_upper or "KITCHEN" in tag_upper:
                    window_type = "kitchen_window"

            # Compute real-world size
            width_m = None
            height_m = None
            sill_height_m = None
            if scale_px_per_mm and scale_px_per_mm > 0:
                width_m = (candidate.width_px / scale_px_per_mm) / 1000

                # Default heights based on type
                if window_type == "ventilator":
                    height_m = 0.45
                    sill_height_m = 2.1
                elif window_type == "toilet_window":
                    height_m = 0.45
                    sill_height_m = 1.8
                else:
                    height_m = 1.2
                    sill_height_m = 0.9

            source = {
                "symbol": "parallel_lines" if "parallel_lines" in candidate.signals else (
                    "gap" if "gap" in candidate.signals else "rectangle"
                ),
                "text": "ocr" if "tag" in candidate.signals else "none",
            }

            window = DetectedWindow(
                id=window_id,
                type=window_type,
                tag=candidate.tag,
                bbox=candidate.bbox,
                center=candidate.center,
                width_px=candidate.width_px,
                height_px=candidate.height_px,
                width_m=width_m,
                height_m=height_m,
                sill_height_m=sill_height_m,
                confidence=candidate.confidence,
                signals=candidate.signals,
                source=source,
            )
            windows.append(window)

        return windows

    def draw_debug_overlay(
        self,
        image: np.ndarray,
        windows: List[DetectedWindow],
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Draw detected windows on image for debugging."""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        for window in windows:
            x1, y1, x2, y2 = window.bbox

            # Color by type
            if window.type == "ventilator":
                color = (255, 0, 255)  # Magenta for ventilators
            elif window.confidence >= 0.7:
                color = (255, 0, 0)  # Blue - high confidence window
            elif window.confidence >= 0.5:
                color = (255, 255, 0)  # Cyan - medium
            else:
                color = (255, 165, 0)  # Orange - low

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.circle(overlay, window.center, 4, color, -1)

            label = window.tag if window.tag else window.id
            label += f" ({window.confidence:.2f})"
            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            signals_str = ",".join(window.signals)
            cv2.putText(
                overlay, signals_str,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1
            )

        if output_path:
            cv2.imwrite(str(output_path), overlay)

        return overlay
