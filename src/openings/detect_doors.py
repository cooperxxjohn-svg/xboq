"""
Door Detection Module - Multi-Signal Approach
Detects doors using:
1. Swing arcs (common in architectural plans)
2. Wall gaps/openings
3. Door leaf rectangles
4. OCR tag association

Indian conventions: D, D1, D2, MD, SD, TD, BD, FD, WD
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict, Any
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class DoorCandidate:
    """Intermediate door candidate before final classification."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    width_px: float
    signals: Set[str] = field(default_factory=set)  # "arc", "gap", "leaf", "tag"
    arc_info: Optional[Dict] = None
    gap_info: Optional[Dict] = None
    leaf_info: Optional[Dict] = None
    tag: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DetectedDoor:
    """Final detected door with all metadata."""
    id: str  # D-01, D-02, etc.
    type: str = "door"  # door, sliding_door, french_door
    tag: Optional[str] = None  # D, D1, D2, MD, SD, etc.
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    center: Tuple[int, int] = (0, 0)
    width_px: float = 0.0
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    wall_segment_id: Optional[str] = None
    room_left_id: Optional[str] = None
    room_right_id: Optional[str] = None
    swing_direction: Optional[str] = None  # "left", "right", "both"
    confidence: float = 0.0
    signals: Set[str] = field(default_factory=set)
    source: Dict[str, str] = field(default_factory=dict)


class DoorDetector:
    """
    Multi-signal door detector for architectural floor plans.

    Detection signals:
    1. Swing arcs - Quarter circles indicating door swing
    2. Wall gaps - Breaks in wall mask of typical door width
    3. Door leaves - Thin rectangles attached to wall openings
    4. Tags - OCR-detected labels like D, D1, MD, etc.
    """

    def __init__(
        self,
        min_width_px: int = 20,
        max_width_px: int = 150,
        arc_angle_tolerance: float = 15.0,
        arc_completeness_min: float = 0.6,
        gap_tolerance_px: int = 5,
    ):
        self.min_width_px = min_width_px
        self.max_width_px = max_width_px
        self.arc_angle_tolerance = arc_angle_tolerance
        self.arc_completeness_min = arc_completeness_min
        self.gap_tolerance_px = gap_tolerance_px

        # Door tag patterns (Indian conventions)
        self.door_tags = {
            "D", "D1", "D2", "D3", "D4", "D5",
            "MD", "SD", "TD", "BD", "FD", "WD",
            "DOOR", "MAIN", "SLIDING", "TOILET",
        }

    def detect(
        self,
        image: np.ndarray,
        wall_mask: np.ndarray,
        wall_mask_closed: Optional[np.ndarray] = None,
        texts: Optional[List[Dict]] = None,
        scale_px_per_mm: Optional[float] = None,
    ) -> List[DetectedDoor]:
        """
        Detect doors using multiple signals.

        Args:
            image: Original grayscale or color image
            wall_mask: Binary wall mask (walls = 255)
            wall_mask_closed: Wall mask with gaps closed (for gap detection)
            texts: List of text boxes with {"text", "bbox", "confidence"}
            scale_px_per_mm: Scale for real-world size computation

        Returns:
            List of DetectedDoor objects
        """
        logger.info("Starting multi-signal door detection")

        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        candidates: List[DoorCandidate] = []

        # Signal 1: Detect swing arcs
        arc_candidates = self._detect_swing_arcs(gray, wall_mask)
        logger.info(f"Found {len(arc_candidates)} swing arc candidates")

        # Signal 2: Detect wall gaps
        if wall_mask_closed is not None:
            gap_candidates = self._detect_wall_gaps(wall_mask, wall_mask_closed)
            logger.info(f"Found {len(gap_candidates)} wall gap candidates")
        else:
            gap_candidates = []

        # Signal 3: Detect door leaf rectangles
        leaf_candidates = self._detect_door_leaves(gray, wall_mask)
        logger.info(f"Found {len(leaf_candidates)} door leaf candidates")

        # Merge candidates from all signals
        candidates = self._merge_candidates(arc_candidates, gap_candidates, leaf_candidates)
        logger.info(f"Merged into {len(candidates)} unique candidates")

        # Signal 4: Associate tags
        if texts:
            self._associate_tags(candidates, texts)

        # Score and filter candidates
        candidates = self._score_candidates(candidates)
        candidates = [c for c in candidates if c.confidence >= 0.3]

        # Convert to final DetectedDoor objects
        doors = self._finalize_doors(candidates, scale_px_per_mm)
        logger.info(f"Detected {len(doors)} doors")

        return doors

    def _detect_swing_arcs(
        self,
        gray: np.ndarray,
        wall_mask: np.ndarray,
    ) -> List[DoorCandidate]:
        """
        Detect door swing arcs using Hough circle detection and contour analysis.

        Door swing arcs are typically quarter circles (~90 degrees) near wall openings.
        """
        candidates = []

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate wall mask slightly to find regions near walls
        wall_dilated = cv2.dilate(wall_mask, np.ones((5, 5), np.uint8), iterations=2)

        # Look for arcs only near walls
        edges_near_walls = cv2.bitwise_and(edges, wall_dilated)

        # Find contours that could be arcs
        contours, _ = cv2.findContours(edges_near_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 10:
                continue

            # Fit ellipse if enough points
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse

                    # Check if it's roughly circular (axes similar)
                    major, minor = max(axes), min(axes)
                    if minor > 0 and major / minor < 1.5:
                        # Check if radius is in door width range
                        radius = (major + minor) / 4
                        if self.min_width_px <= radius <= self.max_width_px:
                            # Check arc completeness (should be ~quarter circle)
                            arc_length = cv2.arcLength(contour, False)
                            expected_quarter = math.pi * radius / 2
                            completeness = arc_length / expected_quarter if expected_quarter > 0 else 0

                            if 0.4 <= completeness <= 1.2:  # Allow some tolerance
                                # This looks like a door arc
                                x, y = int(center[0]), int(center[1])
                                width = int(radius * 2)

                                candidate = DoorCandidate(
                                    bbox=(x - width//2, y - width//2, x + width//2, y + width//2),
                                    center=(x, y),
                                    width_px=radius,
                                    signals={"arc"},
                                    arc_info={
                                        "center": center,
                                        "radius": radius,
                                        "completeness": completeness,
                                        "angle": angle,
                                    },
                                )
                                candidates.append(candidate)
                except cv2.error:
                    continue

        # Also try Hough circle detection for cleaner arcs
        circles = cv2.HoughCircles(
            edges_near_walls,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_width_px,
            param1=50,
            param2=30,
            minRadius=self.min_width_px // 2,
            maxRadius=self.max_width_px,
        )

        if circles is not None:
            for circle in circles[0]:
                x, y, radius = int(circle[0]), int(circle[1]), circle[2]

                # Check if this circle is near a wall
                if 0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1]:
                    # Look in neighborhood for wall pixels
                    neighborhood = wall_mask[
                        max(0, y-int(radius)-10):min(wall_mask.shape[0], y+int(radius)+10),
                        max(0, x-int(radius)-10):min(wall_mask.shape[1], x+int(radius)+10)
                    ]
                    if np.sum(neighborhood) > 0:
                        width = int(radius * 2)
                        candidate = DoorCandidate(
                            bbox=(x - width//2, y - width//2, x + width//2, y + width//2),
                            center=(x, y),
                            width_px=radius,
                            signals={"arc"},
                            arc_info={
                                "center": (x, y),
                                "radius": radius,
                                "completeness": 0.5,  # Unknown for Hough
                            },
                        )
                        candidates.append(candidate)

        return candidates

    def _detect_wall_gaps(
        self,
        wall_mask: np.ndarray,
        wall_mask_closed: np.ndarray,
    ) -> List[DoorCandidate]:
        """
        Detect door openings by finding gaps in walls.

        Gaps are identified by comparing original wall mask with a morphologically
        closed version (where small gaps are filled).
        """
        candidates = []

        # Find difference between closed and original (this shows the gaps)
        gaps = cv2.subtract(wall_mask_closed, wall_mask)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        gaps = cv2.morphologyEx(gaps, cv2.MORPH_OPEN, kernel)

        # Find gap regions
        contours, _ = cv2.findContours(gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if size is in door range
            gap_width = max(w, h)
            gap_height = min(w, h)

            if self.min_width_px <= gap_width <= self.max_width_px:
                # This looks like a door-sized gap
                center_x = x + w // 2
                center_y = y + h // 2

                candidate = DoorCandidate(
                    bbox=(x, y, x + w, y + h),
                    center=(center_x, center_y),
                    width_px=gap_width,
                    signals={"gap"},
                    gap_info={
                        "width": gap_width,
                        "height": gap_height,
                        "area": cv2.contourArea(contour),
                        "orientation": "horizontal" if w > h else "vertical",
                    },
                )
                candidates.append(candidate)

        return candidates

    def _detect_door_leaves(
        self,
        gray: np.ndarray,
        wall_mask: np.ndarray,
    ) -> List[DoorCandidate]:
        """
        Detect door leaf rectangles (thin rectangles near wall openings).

        Door leaves are often drawn as thin rectangles representing the door panel.
        """
        candidates = []

        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove walls from consideration
        non_wall = cv2.bitwise_and(binary, cv2.bitwise_not(wall_mask))

        # Find contours
        contours, _ = cv2.findContours(non_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue

            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            center, (w, h), angle = rect

            # Ensure w is the longer side
            if h > w:
                w, h = h, w

            # Check aspect ratio (door leaves are typically thin rectangles)
            if h > 0:
                aspect_ratio = w / h

                # Door leaves: aspect ratio 3:1 to 20:1, length in door range
                if 3 <= aspect_ratio <= 25 and self.min_width_px <= w <= self.max_width_px * 1.5:
                    # Check if near a wall
                    cx, cy = int(center[0]), int(center[1])

                    # Look for wall pixels nearby
                    search_radius = int(max(w, h) / 2) + 10
                    y1 = max(0, cy - search_radius)
                    y2 = min(wall_mask.shape[0], cy + search_radius)
                    x1 = max(0, cx - search_radius)
                    x2 = min(wall_mask.shape[1], cx + search_radius)

                    wall_nearby = np.sum(wall_mask[y1:y2, x1:x2]) > 0

                    if wall_nearby:
                        candidate = DoorCandidate(
                            bbox=(int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)),
                            center=(cx, cy),
                            width_px=w,
                            signals={"leaf"},
                            leaf_info={
                                "width": w,
                                "height": h,
                                "angle": angle,
                                "aspect_ratio": aspect_ratio,
                            },
                        )
                        candidates.append(candidate)

        return candidates

    def _merge_candidates(
        self,
        arc_candidates: List[DoorCandidate],
        gap_candidates: List[DoorCandidate],
        leaf_candidates: List[DoorCandidate],
    ) -> List[DoorCandidate]:
        """
        Merge candidates from different signals if they overlap.

        Higher confidence when multiple signals agree on same location.
        """
        all_candidates = arc_candidates + gap_candidates + leaf_candidates

        if not all_candidates:
            return []

        merged = []
        used = set()

        # Sort by signal count (prefer multi-signal)
        all_candidates.sort(key=lambda c: len(c.signals), reverse=True)

        for i, c1 in enumerate(all_candidates):
            if i in used:
                continue

            # Find overlapping candidates
            overlapping = [c1]

            for j, c2 in enumerate(all_candidates):
                if j <= i or j in used:
                    continue

                # Check overlap using IoU
                iou = self._compute_iou(c1.bbox, c2.bbox)
                if iou > 0.3:  # Significant overlap
                    overlapping.append(c2)
                    used.add(j)

            # Merge overlapping candidates
            merged_candidate = self._merge_overlapping(overlapping)
            merged.append(merged_candidate)
            used.add(i)

        return merged

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_overlapping(
        self,
        candidates: List[DoorCandidate],
    ) -> DoorCandidate:
        """Merge multiple overlapping candidates into one."""
        if len(candidates) == 1:
            return candidates[0]

        # Combine signals
        all_signals = set()
        for c in candidates:
            all_signals.update(c.signals)

        # Average position
        centers = [c.center for c in candidates]
        avg_center = (
            int(sum(c[0] for c in centers) / len(centers)),
            int(sum(c[1] for c in centers) / len(centers)),
        )

        # Average width
        avg_width = sum(c.width_px for c in candidates) / len(candidates)

        # Combine bboxes (union)
        x1 = min(c.bbox[0] for c in candidates)
        y1 = min(c.bbox[1] for c in candidates)
        x2 = max(c.bbox[2] for c in candidates)
        y2 = max(c.bbox[3] for c in candidates)

        # Collect info from all signals
        arc_info = next((c.arc_info for c in candidates if c.arc_info), None)
        gap_info = next((c.gap_info for c in candidates if c.gap_info), None)
        leaf_info = next((c.leaf_info for c in candidates if c.leaf_info), None)
        tag = next((c.tag for c in candidates if c.tag), None)

        return DoorCandidate(
            bbox=(x1, y1, x2, y2),
            center=avg_center,
            width_px=avg_width,
            signals=all_signals,
            arc_info=arc_info,
            gap_info=gap_info,
            leaf_info=leaf_info,
            tag=tag,
        )

    def _associate_tags(
        self,
        candidates: List[DoorCandidate],
        texts: List[Dict],
    ) -> None:
        """Associate OCR-detected tags with door candidates."""
        for text_box in texts:
            text = text_box.get("text", "").strip().upper()

            # Check if this is a door tag
            if text in self.door_tags or (len(text) <= 3 and text.startswith("D")):
                bbox = text_box.get("bbox", [0, 0, 0, 0])
                text_center = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                )

                # Find nearest candidate
                min_dist = float("inf")
                nearest = None

                for candidate in candidates:
                    dist = math.sqrt(
                        (text_center[0] - candidate.center[0]) ** 2 +
                        (text_center[1] - candidate.center[1]) ** 2
                    )

                    # Only associate if reasonably close (within 2x door width)
                    if dist < min_dist and dist < candidate.width_px * 3:
                        min_dist = dist
                        nearest = candidate

                if nearest:
                    nearest.tag = text
                    nearest.signals.add("tag")

    def _score_candidates(
        self,
        candidates: List[DoorCandidate],
    ) -> List[DoorCandidate]:
        """Score candidates based on signal strength."""
        for candidate in candidates:
            score = 0.0

            # Base score per signal
            signal_weights = {
                "arc": 0.35,   # Strong signal
                "gap": 0.30,   # Good signal
                "leaf": 0.20,  # Supporting signal
                "tag": 0.25,   # Strong confirmation
            }

            for signal in candidate.signals:
                score += signal_weights.get(signal, 0.1)

            # Bonus for multiple signals
            if len(candidate.signals) >= 2:
                score += 0.15
            if len(candidate.signals) >= 3:
                score += 0.10

            # Clamp to [0, 1]
            candidate.confidence = min(1.0, score)

        return candidates

    def _finalize_doors(
        self,
        candidates: List[DoorCandidate],
        scale_px_per_mm: Optional[float],
    ) -> List[DetectedDoor]:
        """Convert candidates to final DetectedDoor objects."""
        doors = []

        for i, candidate in enumerate(candidates):
            door_id = f"D-{i+1:02d}"

            # Determine door type from tag or signals
            door_type = "door"
            if candidate.tag:
                tag_upper = candidate.tag.upper()
                if "SD" in tag_upper or "SLID" in tag_upper:
                    door_type = "sliding_door"
                elif "FD" in tag_upper or "FRENCH" in tag_upper:
                    door_type = "french_door"
                elif "MD" in tag_upper or "MAIN" in tag_upper:
                    door_type = "main_door"

            # Compute real-world size if scale available
            width_m = None
            height_m = None
            if scale_px_per_mm and scale_px_per_mm > 0:
                width_m = (candidate.width_px / scale_px_per_mm) / 1000
                height_m = 2.1  # Default door height (Indian standard)

            # Determine swing direction from arc
            swing_direction = None
            if candidate.arc_info:
                # Could analyze arc position relative to gap to determine swing
                swing_direction = "unknown"

            # Build source info
            source = {
                "symbol": "swing_arc" if "arc" in candidate.signals else (
                    "gap" if "gap" in candidate.signals else "rectangle"
                ),
                "text": "ocr" if "tag" in candidate.signals else "none",
            }

            door = DetectedDoor(
                id=door_id,
                type=door_type,
                tag=candidate.tag,
                bbox=candidate.bbox,
                center=candidate.center,
                width_px=candidate.width_px,
                width_m=width_m,
                height_m=height_m,
                swing_direction=swing_direction,
                confidence=candidate.confidence,
                signals=candidate.signals,
                source=source,
            )
            doors.append(door)

        return doors

    def draw_debug_overlay(
        self,
        image: np.ndarray,
        doors: List[DetectedDoor],
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Draw detected doors on image for debugging."""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        for door in doors:
            x1, y1, x2, y2 = door.bbox

            # Color by confidence
            if door.confidence >= 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif door.confidence >= 0.5:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 165, 255)  # Orange - low

            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cv2.circle(overlay, door.center, 5, color, -1)

            # Label
            label = door.tag if door.tag else door.id
            label += f" ({door.confidence:.2f})"
            cv2.putText(
                overlay, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            # Show signals
            signals_str = ",".join(door.signals)
            cv2.putText(
                overlay, signals_str,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1
            )

        if output_path:
            cv2.imwrite(str(output_path), overlay)

        return overlay
