"""
Debug Overlays Export System

Guarantees debug overlays for every page processed:
1. Room boundaries overlay
2. Scale detection overlay
3. Text/OCR detection overlay
4. Openings detection overlay
5. Wall mask overlay

Used for QA validation of engine outputs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# Standard colors for overlays (BGR format)
COLORS = {
    "room_fill": [
        (255, 179, 186),  # Light pink
        (255, 223, 186),  # Light orange
        (255, 255, 186),  # Light yellow
        (186, 255, 201),  # Light green
        (186, 225, 255),  # Light blue
        (218, 186, 255),  # Light purple
    ],
    "room_boundary": (0, 0, 0),  # Black
    "door": (0, 128, 255),  # Orange
    "window": (255, 128, 0),  # Blue
    "text_box": (0, 255, 0),  # Green
    "scale_line": (255, 0, 255),  # Magenta
    "wall": (0, 0, 255),  # Red
    "dimension": (255, 255, 0),  # Cyan
    "rfi_highlight": (0, 0, 255),  # Red
}


@dataclass
class OverlayConfig:
    """Configuration for overlay generation."""
    alpha: float = 0.4  # Blend alpha for fills
    line_thickness: int = 2
    font_scale: float = 0.5
    show_confidence: bool = True
    show_measurements: bool = True
    export_format: str = "png"  # png or jpg
    max_dimension: int = 4000  # Resize if larger


class DebugOverlayGenerator:
    """
    Generates debug overlays for floor plan analysis.

    Guarantees an overlay for every page processed.
    """

    def __init__(self, config: OverlayConfig = None):
        self.config = config or OverlayConfig()

    def generate_all_overlays(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        rooms: List[Dict] = None,
        openings: List[Dict] = None,
        texts: List[Dict] = None,
        scale_info: Dict = None,
        wall_mask: np.ndarray = None,
        dimensions: List[Dict] = None,
        rfis: List[Dict] = None,
    ) -> Dict[str, Path]:
        """
        Generate all debug overlays for a page.

        Args:
            image: Source image (BGR)
            output_dir: Output directory
            page_id: Page identifier
            rooms: Room data with polygons
            openings: Door/window detections
            texts: Text detections with bboxes
            scale_info: Scale detection info
            wall_mask: Binary wall mask
            dimensions: Dimension annotations
            rfis: RFIs to highlight

        Returns:
            Dictionary of overlay paths
        """
        output_dir = Path(output_dir)
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Resize if needed
        image = self._maybe_resize(image)

        # 1. ROOMS overlay (always generate)
        rooms_path = self._generate_rooms_overlay(
            image, debug_dir, page_id, rooms or []
        )
        outputs["rooms"] = rooms_path

        # 2. OPENINGS overlay
        openings_path = self._generate_openings_overlay(
            image, debug_dir, page_id, openings or []
        )
        outputs["openings"] = openings_path

        # 3. TEXT/OCR overlay
        text_path = self._generate_text_overlay(
            image, debug_dir, page_id, texts or []
        )
        outputs["text"] = text_path

        # 4. SCALE overlay
        scale_path = self._generate_scale_overlay(
            image, debug_dir, page_id, scale_info, dimensions or []
        )
        outputs["scale"] = scale_path

        # 5. WALL MASK overlay
        if wall_mask is not None:
            wall_path = self._generate_wall_overlay(
                image, debug_dir, page_id, wall_mask
            )
            outputs["walls"] = wall_path

        # 6. RFI HIGHLIGHT overlay (if any RFIs)
        if rfis:
            rfi_path = self._generate_rfi_overlay(
                image, debug_dir, page_id, rfis
            )
            outputs["rfis"] = rfi_path

        # 7. COMBINED master overlay
        combined_path = self._generate_combined_overlay(
            image, debug_dir, page_id,
            rooms or [], openings or [], texts or [],
            scale_info, dimensions or []
        )
        outputs["combined"] = combined_path

        logger.info(f"Generated {len(outputs)} debug overlays for {page_id}")
        return outputs

    def _maybe_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image if too large."""
        h, w = image.shape[:2]
        max_dim = self.config.max_dimension

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _generate_rooms_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        rooms: List[Dict],
    ) -> Path:
        """Generate rooms overlay with filled polygons and labels."""
        overlay = image.copy()
        fill_layer = image.copy()

        for i, room in enumerate(rooms):
            color = COLORS["room_fill"][i % len(COLORS["room_fill"])]

            # Get polygon points
            polygon = room.get("polygon", room.get("points", []))
            if not polygon:
                continue

            pts = np.array(polygon, dtype=np.int32)

            # Fill polygon
            cv2.fillPoly(fill_layer, [pts], color)

            # Draw boundary
            cv2.polylines(overlay, [pts], True, COLORS["room_boundary"], 2)

        # Blend fill
        result = cv2.addWeighted(fill_layer, self.config.alpha, overlay, 1 - self.config.alpha, 0)

        # Add labels
        for i, room in enumerate(rooms):
            polygon = room.get("polygon", room.get("points", []))
            if not polygon:
                continue

            # Calculate centroid
            pts = np.array(polygon, dtype=np.int32)
            M = cv2.moments(pts)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = pts[0][0], pts[0][1]

            # Draw label
            label = room.get("label", room.get("room_type", f"Room {i+1}"))
            area = room.get("area_sqm", room.get("area", 0))

            self._draw_label_box(result, cx, cy, label, f"{area:.1f} sqm")

        output_path = output_dir / f"{page_id}_rooms.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _generate_openings_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        openings: List[Dict],
    ) -> Path:
        """Generate openings (doors/windows) overlay."""
        result = image.copy()

        door_count = 0
        window_count = 0

        for opening in openings:
            otype = opening.get("type", "door").lower()
            bbox = opening.get("bbox", opening.get("box", []))

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

            if "door" in otype:
                color = COLORS["door"]
                door_count += 1
                label = f"D{door_count}"
            else:
                color = COLORS["window"]
                window_count += 1
                label = f"W{window_count}"

            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                result, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Show confidence if available
            conf = opening.get("confidence", None)
            if conf and self.config.show_confidence:
                conf_text = f"{conf:.0%}"
                cv2.putText(
                    result, conf_text,
                    (x1 + 2, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        # Add summary
        summary = f"Doors: {door_count} | Windows: {window_count}"
        cv2.putText(result, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        output_path = output_dir / f"{page_id}_openings.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _generate_text_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        texts: List[Dict],
    ) -> Path:
        """Generate text detection overlay."""
        result = image.copy()

        for text_item in texts:
            bbox = text_item.get("bbox", text_item.get("box", []))
            text = text_item.get("text", "")

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), COLORS["text_box"], 1)

            # Draw text (truncated)
            display_text = text[:15] + "..." if len(text) > 15 else text
            cv2.putText(
                result, display_text,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
            )

        # Add count
        cv2.putText(
            result, f"Text items: {len(texts)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        output_path = output_dir / f"{page_id}_text.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _generate_scale_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        scale_info: Dict,
        dimensions: List[Dict],
    ) -> Path:
        """Generate scale detection overlay."""
        result = image.copy()

        # Draw dimension annotations
        for dim in dimensions:
            bbox = dim.get("bbox", dim.get("box", []))
            value = dim.get("value", dim.get("text", ""))

            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                cv2.rectangle(result, (x1, y1), (x2, y2), COLORS["dimension"], 2)
                cv2.putText(
                    result, str(value),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dimension"], 2
                )

        # Draw scale bar if detected
        if scale_info:
            scale_bar = scale_info.get("scale_bar", {})
            if scale_bar:
                x1 = scale_bar.get("x1", 0)
                y1 = scale_bar.get("y1", 0)
                x2 = scale_bar.get("x2", 0)
                y2 = scale_bar.get("y2", 0)

                cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), COLORS["scale_line"], 3)

            # Scale info text
            method = scale_info.get("method", "unknown")
            ratio = scale_info.get("ratio", "N/A")
            conf = scale_info.get("confidence", 0)

            info_text = f"Scale: 1:{ratio} | Method: {method} | Conf: {conf:.0%}"
            cv2.putText(result, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        output_path = output_dir / f"{page_id}_scale.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _generate_wall_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        wall_mask: np.ndarray,
    ) -> Path:
        """Generate wall mask overlay."""
        result = image.copy()

        # Create colored wall overlay
        if len(wall_mask.shape) == 2:
            wall_colored = np.zeros_like(result)
            wall_colored[wall_mask > 0] = COLORS["wall"]

            # Blend
            result = cv2.addWeighted(result, 0.7, wall_colored, 0.3, 0)

        output_path = output_dir / f"{page_id}_walls.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _generate_rfi_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        rfis: List[Dict],
    ) -> Path:
        """Generate RFI highlight overlay."""
        result = image.copy()

        for rfi in rfis:
            bbox = rfi.get("bbox", rfi.get("location", {}).get("bbox", []))
            rfi_id = rfi.get("id", rfi.get("rfi_id", "?"))
            severity = rfi.get("severity", "medium")

            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

                # Color by severity
                if severity == "high":
                    color = (0, 0, 255)  # Red
                elif severity == "medium":
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow

                cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    result, f"RFI-{rfi_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        output_path = output_dir / f"{page_id}_rfis.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _generate_combined_overlay(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_id: str,
        rooms: List[Dict],
        openings: List[Dict],
        texts: List[Dict],
        scale_info: Dict,
        dimensions: List[Dict],
    ) -> Path:
        """Generate combined master overlay with all annotations."""
        overlay = image.copy()
        fill_layer = image.copy()

        # 1. Room fills
        for i, room in enumerate(rooms):
            color = COLORS["room_fill"][i % len(COLORS["room_fill"])]
            polygon = room.get("polygon", room.get("points", []))
            if polygon:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(fill_layer, [pts], color)

        result = cv2.addWeighted(fill_layer, 0.3, overlay, 0.7, 0)

        # 2. Room boundaries
        for room in rooms:
            polygon = room.get("polygon", room.get("points", []))
            if polygon:
                pts = np.array(polygon, dtype=np.int32)
                cv2.polylines(result, [pts], True, COLORS["room_boundary"], 2)

        # 3. Openings
        for opening in openings:
            bbox = opening.get("bbox", opening.get("box", []))
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                otype = opening.get("type", "door").lower()
                color = COLORS["door"] if "door" in otype else COLORS["window"]
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 4. Room labels
        for i, room in enumerate(rooms):
            polygon = room.get("polygon", room.get("points", []))
            if polygon:
                pts = np.array(polygon, dtype=np.int32)
                M = cv2.moments(pts)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = room.get("label", f"Room {i+1}")
                    area = room.get("area_sqm", 0)
                    self._draw_label_box(result, cx, cy, label, f"{area:.1f} sqm")

        # 5. Scale info
        if scale_info:
            ratio = scale_info.get("ratio", "N/A")
            conf = scale_info.get("confidence", 0)
            cv2.putText(
                result, f"Scale 1:{ratio} ({conf:.0%})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        output_path = output_dir / f"{page_id}_combined.{self.config.export_format}"
        cv2.imwrite(str(output_path), result)
        return output_path

    def _draw_label_box(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        label: str,
        sublabel: str = "",
    ) -> None:
        """Draw a label box with background at centroid."""
        # Get text sizes
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (sw, sh), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        # Calculate box
        max_w = max(lw, sw) + 10
        total_h = lh + sh + 12 if sublabel else lh + 8
        x1 = cx - max_w // 2
        y1 = cy - total_h // 2

        # Draw background
        cv2.rectangle(image, (x1, y1), (x1 + max_w, y1 + total_h), (255, 255, 255), -1)
        cv2.rectangle(image, (x1, y1), (x1 + max_w, y1 + total_h), (0, 0, 0), 1)

        # Draw label
        cv2.putText(
            image, label,
            (x1 + 5, y1 + lh + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

        # Draw sublabel
        if sublabel:
            cv2.putText(
                image, sublabel,
                (x1 + 5, y1 + lh + sh + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 128), 1
            )


def generate_debug_overlays(
    image: np.ndarray,
    output_dir: Path,
    page_id: str,
    **kwargs,
) -> Dict[str, Path]:
    """
    Generate debug overlays for a page.

    Convenience wrapper around DebugOverlayGenerator.

    Args:
        image: Source image
        output_dir: Output directory
        page_id: Page identifier
        **kwargs: Additional data (rooms, openings, texts, etc.)

    Returns:
        Dictionary of output paths
    """
    generator = DebugOverlayGenerator()
    return generator.generate_all_overlays(image, output_dir, page_id, **kwargs)


def ensure_overlays_for_project(
    project_dir: Path,
    output_dir: Path,
) -> Dict[str, Dict[str, Path]]:
    """
    Ensure debug overlays exist for all pages in a project.

    Scans project directory for processed pages and generates
    any missing overlays.

    Args:
        project_dir: Project data directory
        output_dir: Output directory

    Returns:
        Dictionary of page_id -> overlay paths
    """
    all_overlays = {}

    # Find all processed images
    image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    drawings_dir = project_dir / "drawings"

    if not drawings_dir.exists():
        logger.warning(f"No drawings directory: {drawings_dir}")
        return all_overlays

    for pattern in image_patterns:
        for img_path in drawings_dir.glob(pattern):
            page_id = img_path.stem

            # Check if overlays exist
            debug_dir = output_dir / "debug"
            combined_path = debug_dir / f"{page_id}_combined.png"

            if combined_path.exists():
                logger.debug(f"Overlays exist for {page_id}")
                continue

            # Generate overlays
            logger.info(f"Generating overlays for {page_id}")
            image = cv2.imread(str(img_path))

            if image is None:
                logger.warning(f"Could not read image: {img_path}")
                continue

            # Load any available data
            rooms = _load_json_if_exists(output_dir / f"{page_id}_rooms.json")
            openings = _load_json_if_exists(output_dir / f"{page_id}_openings.json")

            overlays = generate_debug_overlays(
                image=image,
                output_dir=output_dir,
                page_id=page_id,
                rooms=rooms,
                openings=openings,
            )

            all_overlays[page_id] = overlays

    return all_overlays


def _load_json_if_exists(path: Path) -> List[Dict]:
    """Load JSON file if it exists."""
    import json
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "items" in data:
                return data["items"]
    return []
