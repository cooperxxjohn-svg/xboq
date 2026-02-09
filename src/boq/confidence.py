"""
Confidence Calculator and Heatmap Generator
Visual confidence representation for QC.

Features:
- Per-element confidence scoring
- Green/yellow/red zone overlays
- Aggregate confidence for BOQ items
- Assumption tracking
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceZone:
    """Confidence zone for a region."""
    zone_id: str
    region_type: str  # room, wall, opening
    bbox: Tuple[int, int, int, int]
    confidence: float
    source: str
    notes: Optional[str] = None


@dataclass
class ConfidenceHeatmap:
    """Heatmap result."""
    image: np.ndarray
    zones: List[ConfidenceZone]
    overall_confidence: float
    by_category: Dict[str, float]


class ConfidenceCalculator:
    """
    Calculate and visualize detection confidence.

    Confidence levels:
    - High (green): 0.8-1.0 - Measured/detected with high certainty
    - Medium (yellow): 0.5-0.8 - Inferred or partially detected
    - Low (red): 0.0-0.5 - Assumed or estimated
    """

    # Confidence colors (BGR)
    COLORS = {
        "high": (0, 200, 0),      # Green
        "medium": (0, 200, 200),  # Yellow
        "low": (0, 0, 200),       # Red
    }

    # Confidence thresholds
    HIGH_THRESHOLD = 0.75
    MEDIUM_THRESHOLD = 0.50

    # Source confidence weights
    SOURCE_CONFIDENCE = {
        "schedule_detection": 0.90,
        "vector_extraction": 0.85,
        "room_detection": 0.80,
        "wall_detection": 0.75,
        "opening_detection": 0.70,
        "ocr_text": 0.65,
        "room_finish_mapping": 0.60,
        "rule_of_thumb": 0.55,
        "calculation": 0.50,
        "assumption": 0.40,
        "default": 0.30,
    }

    def __init__(self):
        self.zones: List[ConfidenceZone] = []

    def calculate_item_confidence(
        self,
        derived_from: str,
        base_confidence: Optional[float] = None,
        factors: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate confidence for a BOQ item.

        Args:
            derived_from: Source of the quantity
            base_confidence: Override base confidence
            factors: Adjustment factors

        Returns:
            Confidence score 0.0-1.0
        """
        if base_confidence is not None:
            conf = base_confidence
        else:
            conf = self.SOURCE_CONFIDENCE.get(derived_from, 0.50)

        # Apply adjustment factors
        if factors:
            for factor_name, factor_value in factors.items():
                conf *= factor_value

        return min(1.0, max(0.0, conf))

    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level name."""
        if confidence >= self.HIGH_THRESHOLD:
            return "high"
        elif confidence >= self.MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "low"

    def add_zone(
        self,
        zone_id: str,
        region_type: str,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        source: str,
        notes: Optional[str] = None,
    ) -> None:
        """Add a confidence zone."""
        self.zones.append(ConfidenceZone(
            zone_id=zone_id,
            region_type=region_type,
            bbox=bbox,
            confidence=confidence,
            source=source,
            notes=notes,
        ))

    def calculate_overall_confidence(
        self,
        boq_items: List[Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall and per-category confidence.

        Args:
            boq_items: List of BOQ items with confidence attribute

        Returns:
            Tuple of (overall_confidence, by_category)
        """
        if not boq_items:
            return 0.0, {}

        by_category: Dict[str, List[float]] = {}
        all_confidences = []

        for item in boq_items:
            conf = getattr(item, "confidence", 0.5)
            all_confidences.append(conf)

            # Categorize
            derived_from = getattr(item, "derived_from", "unknown")
            category = self._categorize_source(derived_from)

            if category not in by_category:
                by_category[category] = []
            by_category[category].append(conf)

        overall = sum(all_confidences) / len(all_confidences)

        # Average by category
        category_averages = {
            cat: sum(confs) / len(confs)
            for cat, confs in by_category.items()
        }

        return overall, category_averages

    def _categorize_source(self, source: str) -> str:
        """Categorize source into main categories."""
        source_lower = source.lower()

        if "room" in source_lower or "finish" in source_lower:
            return "finishes"
        elif "wall" in source_lower or "plaster" in source_lower:
            return "masonry"
        elif "opening" in source_lower or "door" in source_lower or "window" in source_lower:
            return "openings"
        elif "slab" in source_lower or "concrete" in source_lower or "rcc" in source_lower:
            return "structural"
        elif "steel" in source_lower or "reinforcement" in source_lower:
            return "steel"
        else:
            return "other"

    def generate_heatmap(
        self,
        base_image: np.ndarray,
        rooms: Optional[List[Dict]] = None,
        openings: Optional[List[Dict]] = None,
        walls_mask: Optional[np.ndarray] = None,
        alpha: float = 0.4,
    ) -> ConfidenceHeatmap:
        """
        Generate confidence heatmap overlay.

        Args:
            base_image: Original plan image
            rooms: List of room dicts with confidence
            openings: List of openings with confidence
            walls_mask: Wall detection mask
            alpha: Overlay transparency

        Returns:
            ConfidenceHeatmap with overlay image
        """
        # Ensure color image
        if len(base_image.shape) == 2:
            overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_image.copy()

        # Create confidence overlay
        conf_overlay = np.zeros_like(overlay)

        # Add room confidence zones
        if rooms:
            for room in rooms:
                conf = room.get("confidence", 0.5)
                vertices = room.get("polygon_vertices", room.get("vertices", []))

                if vertices and len(vertices) >= 3:
                    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                    color = self._get_confidence_color(conf)

                    # Fill polygon
                    cv2.fillPoly(conf_overlay, [pts], color)

                    # Add to zones
                    x_coords = [v[0] for v in vertices]
                    y_coords = [v[1] for v in vertices]
                    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    self.add_zone(
                        zone_id=room.get("room_id", room.get("id", "")),
                        region_type="room",
                        bbox=bbox,
                        confidence=conf,
                        source="room_detection",
                    )

        # Add opening confidence zones
        if openings:
            for opening in openings:
                conf = opening.get("confidence", 0.5)
                bbox = opening.get("bbox", [0, 0, 0, 0])

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = self._get_confidence_color(conf)

                    # Draw rectangle
                    cv2.rectangle(conf_overlay, (x1, y1), (x2, y2), color, -1)

                    self.add_zone(
                        zone_id=opening.get("id", ""),
                        region_type="opening",
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        source="opening_detection",
                    )

        # Add wall confidence (if mask provided)
        if walls_mask is not None:
            wall_conf = 0.75  # Default wall detection confidence
            color = self._get_confidence_color(wall_conf)

            # Create colored wall overlay
            wall_colored = np.zeros_like(overlay)
            wall_colored[walls_mask > 0] = color

            # Blend with confidence overlay
            conf_overlay = cv2.addWeighted(conf_overlay, 1.0, wall_colored, 0.3, 0)

        # Blend with original
        result = cv2.addWeighted(overlay, 1 - alpha, conf_overlay, alpha, 0)

        # Add legend
        self._draw_legend(result)

        # Calculate overall confidence
        all_confs = [z.confidence for z in self.zones]
        overall = sum(all_confs) / len(all_confs) if all_confs else 0.5

        by_category = {}
        for z in self.zones:
            cat = z.region_type
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(z.confidence)

        by_category_avg = {
            cat: sum(confs) / len(confs)
            for cat, confs in by_category.items()
        }

        return ConfidenceHeatmap(
            image=result,
            zones=self.zones,
            overall_confidence=overall,
            by_category=by_category_avg,
        )

    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color for confidence level."""
        level = self.get_confidence_level(confidence)
        return self.COLORS.get(level, self.COLORS["medium"])

    def _draw_legend(self, image: np.ndarray) -> None:
        """Draw confidence legend on image."""
        h, w = image.shape[:2]

        # Legend background
        legend_h = 100
        legend_w = 180
        cv2.rectangle(
            image,
            (w - legend_w - 10, 10),
            (w - 10, legend_h),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            image,
            (w - legend_w - 10, 10),
            (w - 10, legend_h),
            (0, 0, 0),
            1
        )

        # Title
        cv2.putText(
            image, "Confidence",
            (w - legend_w, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

        # Legend items
        y = 50
        for level, color in [("High (≥0.75)", self.COLORS["high"]),
                            ("Medium (0.5-0.75)", self.COLORS["medium"]),
                            ("Low (<0.5)", self.COLORS["low"])]:
            cv2.rectangle(image, (w - legend_w, y), (w - legend_w + 15, y + 12), color, -1)
            cv2.putText(
                image, level,
                (w - legend_w + 20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1
            )
            y += 18

    def export_confidence_report(
        self,
        output_path: Path,
        boq_items: List[Any],
    ) -> None:
        """Export confidence report as markdown."""
        overall, by_category = self.calculate_overall_confidence(boq_items)

        with open(output_path, "w") as f:
            f.write("# Confidence Report\n\n")

            f.write(f"## Overall Confidence: {overall:.1%}\n\n")

            level = self.get_confidence_level(overall)
            if level == "high":
                f.write("✅ High confidence - quantities are primarily measured/detected\n\n")
            elif level == "medium":
                f.write("⚠️ Medium confidence - some quantities are inferred or estimated\n\n")
            else:
                f.write("❌ Low confidence - many quantities are assumed or estimated\n\n")

            f.write("## By Category\n\n")
            f.write("| Category | Confidence | Level |\n")
            f.write("|----------|------------|-------|\n")

            for cat, conf in sorted(by_category.items(), key=lambda x: -x[1]):
                level = self.get_confidence_level(conf)
                icon = "🟢" if level == "high" else ("🟡" if level == "medium" else "🔴")
                f.write(f"| {cat.title()} | {conf:.1%} | {icon} {level.title()} |\n")

            f.write("\n## Zones\n\n")
            f.write("| ID | Type | Confidence | Source |\n")
            f.write("|----|------|------------|--------|\n")

            for zone in self.zones[:20]:  # Limit to 20
                f.write(f"| {zone.zone_id} | {zone.region_type} | {zone.confidence:.2f} | {zone.source} |\n")

            if len(self.zones) > 20:
                f.write(f"\n*... and {len(self.zones) - 20} more zones*\n")


def create_confidence_summary(
    boq_items: List[Any],
) -> Dict[str, Any]:
    """
    Create confidence summary for BOQ items.

    Returns:
        Summary dict with statistics
    """
    if not boq_items:
        return {"overall": 0.0, "count": 0}

    confidences = [getattr(item, "confidence", 0.5) for item in boq_items]

    high_count = sum(1 for c in confidences if c >= 0.75)
    medium_count = sum(1 for c in confidences if 0.5 <= c < 0.75)
    low_count = sum(1 for c in confidences if c < 0.5)

    return {
        "overall": sum(confidences) / len(confidences),
        "count": len(confidences),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "high_percent": high_count / len(confidences) * 100,
        "medium_percent": medium_count / len(confidences) * 100,
        "low_percent": low_count / len(confidences) * 100,
        "min": min(confidences),
        "max": max(confidences),
    }
