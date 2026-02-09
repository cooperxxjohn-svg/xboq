"""
Synthetic Floor Plan Generator.

Creates test floor plans with known ground truth for validation.
Supports various layouts common in Indian residential construction.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np


@dataclass
class SyntheticRoom:
    """A synthetic room with ground truth."""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h in mm
    area_sqm: float
    aliases: List[str] = field(default_factory=list)


@dataclass
class SyntheticPlan:
    """A synthetic floor plan with metadata."""
    name: str
    width_mm: int
    height_mm: int
    rooms: List[SyntheticRoom]
    scale: int  # 1:scale
    dpi: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "scale": self.scale,
            "dpi": self.dpi,
            "rooms": [
                {
                    "name": r.name,
                    "bbox_mm": r.bbox,
                    "area_sqm": r.area_sqm,
                    "aliases": r.aliases,
                }
                for r in self.rooms
            ],
        }


def mm_to_pixels(mm: float, scale: int, dpi: int) -> int:
    """
    Convert mm to pixels at given scale and DPI.

    At scale 1:100, 1mm real = 0.01mm on drawing.
    Drawing mm = real mm / scale
    Pixels = drawing_mm * (dpi / 25.4)
    """
    drawing_mm = mm / scale
    pixels = drawing_mm * dpi / 25.4
    return int(pixels)


class SyntheticPlanGenerator:
    """
    Generates synthetic floor plans with various configurations.
    """

    def __init__(self, scale: int = 100, dpi: int = 300):
        self.scale = scale
        self.dpi = dpi
        self.wall_thickness_mm = 200  # 200mm walls
        self.offset = 50  # pixel offset for border

    def _draw_room(
        self,
        img: np.ndarray,
        room: SyntheticRoom,
        wall_thickness: int,
    ) -> None:
        """Draw a room on the image."""
        x = mm_to_pixels(room.bbox[0], self.scale, self.dpi) + self.offset
        y = mm_to_pixels(room.bbox[1], self.scale, self.dpi) + self.offset
        w = mm_to_pixels(room.bbox[2], self.scale, self.dpi)
        h = mm_to_pixels(room.bbox[3], self.scale, self.dpi)

        # Draw room rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), wall_thickness)

        # Add room label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = room.name if len(room.name) <= 12 else room.name[:12]
        label_size = cv2.getTextSize(label, font, 0.4, 1)[0]
        label_x = x + w // 2 - label_size[0] // 2
        label_y = y + h // 2

        cv2.putText(img, label, (label_x, label_y), font, 0.4, (0, 0, 0), 1)

        # Add area annotation
        area_text = f"{room.area_sqm:.1f} sqm"
        cv2.putText(
            img, area_text,
            (label_x, label_y + 18),
            font, 0.35, (100, 100, 100), 1
        )

    def _add_scale_bar(
        self,
        img: np.ndarray,
        length_mm: int,
        position: Tuple[int, int],
    ) -> None:
        """Add a scale bar to the image."""
        start_x, y = position
        end_x = start_x + mm_to_pixels(length_mm, self.scale, self.dpi)

        # Draw scale line
        cv2.line(img, (start_x, y), (end_x, y), (0, 0, 0), 1)
        cv2.line(img, (start_x, y - 5), (start_x, y + 5), (0, 0, 0), 1)
        cv2.line(img, (end_x, y - 5), (end_x, y + 5), (0, 0, 0), 1)

        # Add text
        cv2.putText(
            img, f"{length_mm}",
            ((start_x + end_x) // 2 - 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )

    def _add_scale_note(self, img: np.ndarray, position: Tuple[int, int]) -> None:
        """Add scale note (e.g., 'Scale 1:100')."""
        cv2.putText(
            img, f"Scale 1:{self.scale}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    def generate(self, rooms: List[SyntheticRoom], name: str) -> Tuple[np.ndarray, SyntheticPlan]:
        """
        Generate a floor plan image from room definitions.

        Args:
            rooms: List of room definitions
            name: Plan identifier

        Returns:
            Tuple of (image array, plan metadata)
        """
        # Calculate plan dimensions
        max_x = max(r.bbox[0] + r.bbox[2] for r in rooms)
        max_y = max(r.bbox[1] + r.bbox[3] for r in rooms)

        width_mm = max_x
        height_mm = max_y

        # Convert to pixels
        w_px = mm_to_pixels(width_mm, self.scale, self.dpi)
        h_px = mm_to_pixels(height_mm, self.scale, self.dpi)

        # Create white image with border margin
        img = np.ones((h_px + 100, w_px + 100, 3), dtype=np.uint8) * 255

        wall_thickness = max(2, mm_to_pixels(self.wall_thickness_mm, self.scale, self.dpi))

        # Draw rooms
        for room in rooms:
            self._draw_room(img, room, wall_thickness)

        # Add scale bar
        self._add_scale_bar(img, 4000, (self.offset, h_px + 50))

        # Add scale note
        self._add_scale_note(img, (10, h_px + 80))

        plan = SyntheticPlan(
            name=name,
            width_mm=width_mm,
            height_mm=height_mm,
            rooms=rooms,
            scale=self.scale,
            dpi=self.dpi,
        )

        return img, plan


# ============================================================================
# Pre-defined plan generators
# ============================================================================

def generate_simple_rectangle(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """Generate simplest test case - single rectangle room."""
    generator = SyntheticPlanGenerator(scale, dpi)

    rooms = [
        SyntheticRoom("Room", (0, 0, 10000, 10000), 100.0),
    ]

    return generator.generate(rooms, "simple_rectangle")


def generate_1bhk(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """
    Generate a 1BHK apartment floor plan.

    Layout (approximate):
    +------------------+----------+
    |                  | Toilet   |
    |     Bedroom      | 3x2m     |
    |     4x4m         +----------+
    |                  | Kitchen  |
    +--------+---------+  3x3m    |
    | Balcony| Living  |          |
    | 2x1.5m | 4x4m    +----------+
    +--------+         |
    +------------------+
    """
    generator = SyntheticPlanGenerator(scale, dpi)

    rooms = [
        SyntheticRoom("Bedroom", (0, 0, 4000, 4000), 16.0, ["BR", "Master"]),
        SyntheticRoom("Living", (0, 4000, 4000, 4000), 16.0, ["Hall", "Drawing"]),
        SyntheticRoom("Kitchen", (4000, 4000, 3000, 3000), 9.0, ["Kitch"]),
        SyntheticRoom("Toilet", (4000, 0, 3000, 2000), 6.0, ["WC", "Bath", "T&B"]),
        SyntheticRoom("Bath", (4000, 2000, 3000, 2000), 6.0, ["Bathroom", "Attached"]),
        SyntheticRoom("Balcony", (0, 8000, 2000, 1500), 3.0, ["Blcny", "Deck"]),
    ]

    return generator.generate(rooms, "1bhk")


def generate_2bhk(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """
    Generate a 2BHK apartment floor plan.

    More complex layout with passage and utility.
    """
    generator = SyntheticPlanGenerator(scale, dpi)

    rooms = [
        SyntheticRoom("Master Bedroom", (0, 0, 4500, 4000), 18.0, ["MBR"]),
        SyntheticRoom("Bedroom 2", (4500, 0, 3500, 3500), 12.25, ["BR2", "Kids Room"]),
        SyntheticRoom("Living", (0, 4000, 5000, 4500), 22.5, ["Hall", "Drawing"]),
        SyntheticRoom("Dining", (5000, 4000, 3000, 3000), 9.0, ["Dining Room"]),
        SyntheticRoom("Kitchen", (5000, 7000, 3000, 3000), 9.0, ["Kitch"]),
        SyntheticRoom("Toilet 1", (4500, 0, 2000, 2000), 4.0, ["WC1", "Common Toilet"]),
        SyntheticRoom("Toilet 2", (6500, 0, 1500, 2000), 3.0, ["WC2", "Attached"]),
        SyntheticRoom("Utility", (0, 8500, 2000, 1500), 3.0, ["Store", "Service"]),
        SyntheticRoom("Balcony 1", (0, 10000, 2500, 1500), 3.75),
        SyntheticRoom("Balcony 2", (5000, 10000, 3000, 1500), 4.5),
        SyntheticRoom("Passage", (4500, 2000, 1000, 2000), 2.0, ["Corridor"]),
    ]

    return generator.generate(rooms, "2bhk")


def generate_3bhk(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """Generate a 3BHK apartment floor plan."""
    generator = SyntheticPlanGenerator(scale, dpi)

    rooms = [
        SyntheticRoom("Master Bedroom", (0, 0, 5000, 4500), 22.5, ["MBR"]),
        SyntheticRoom("Bedroom 2", (5000, 0, 4000, 4000), 16.0, ["BR2"]),
        SyntheticRoom("Bedroom 3", (9000, 0, 3500, 4000), 14.0, ["BR3", "Guest"]),
        SyntheticRoom("Living", (0, 4500, 6000, 5000), 30.0, ["Hall", "Drawing"]),
        SyntheticRoom("Dining", (6000, 4500, 3500, 3500), 12.25, ["Dining Room"]),
        SyntheticRoom("Kitchen", (9500, 4500, 3000, 3500), 10.5, ["Kitch"]),
        SyntheticRoom("Toilet 1", (5000, 0, 2000, 2000), 4.0, ["MBR Toilet"]),
        SyntheticRoom("Toilet 2", (7000, 0, 2000, 2000), 4.0, ["Common Toilet"]),
        SyntheticRoom("Toilet 3", (9500, 8000, 2000, 2000), 4.0, ["Guest Toilet"]),
        SyntheticRoom("Utility", (0, 9500, 2500, 2000), 5.0, ["Store", "Service"]),
        SyntheticRoom("Balcony 1", (0, 11500, 3000, 2000), 6.0),
        SyntheticRoom("Balcony 2", (6000, 11500, 3500, 2000), 7.0),
        SyntheticRoom("Passage", (6000, 2000, 1500, 2500), 3.75, ["Corridor"]),
    ]

    return generator.generate(rooms, "3bhk")


def generate_indian_apartment(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """
    Generate a typical Indian apartment with India-specific room names.

    Includes: Pooja room, OTS toilet, Servant room, etc.
    """
    generator = SyntheticPlanGenerator(scale, dpi)

    rooms = [
        SyntheticRoom("Master Bedroom", (0, 0, 4500, 4000), 18.0, ["MBR"]),
        SyntheticRoom("Bedroom", (4500, 0, 3500, 3500), 12.25),
        SyntheticRoom("Drawing Room", (0, 4000, 5000, 4000), 20.0, ["Living", "Hall"]),
        SyntheticRoom("Dining", (5000, 4000, 3000, 3000), 9.0),
        SyntheticRoom("Kitchen", (5000, 7000, 3000, 3000), 9.0, ["Kitch", "Rasoi"]),
        SyntheticRoom("Pooja", (0, 8000, 2000, 1500), 3.0, ["Pooja Room", "Mandir"]),
        SyntheticRoom("T&B", (4500, 0, 2000, 2000), 4.0, ["Toilet", "WC", "Bathroom"]),
        SyntheticRoom("OTS", (6500, 0, 1500, 2000), 3.0, ["Open to Sky", "Utility"]),
        SyntheticRoom("Servant", (0, 9500, 2000, 2000), 4.0, ["Servant Room", "Service"]),
        SyntheticRoom("Balcony", (0, 11500, 3000, 1500), 4.5, ["Blcny"]),
        SyntheticRoom("Dry Balcony", (5000, 10000, 3000, 1500), 4.5),
        SyntheticRoom("Foyer", (3000, 4000, 2000, 1500), 3.0, ["Entrance"]),
    ]

    return generator.generate(rooms, "indian_apartment")


def generate_all_plans(
    output_dir: Path,
    scale: int = 100,
    dpi: int = 300,
) -> List[SyntheticPlan]:
    """
    Generate all predefined synthetic plans and save them.

    Args:
        output_dir: Directory to save images and ground truth
        scale: Drawing scale (default 1:100)
        dpi: Output DPI (default 300)

    Returns:
        List of generated plan metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("simple_rectangle", generate_simple_rectangle),
        ("1bhk", generate_1bhk),
        ("2bhk", generate_2bhk),
        ("3bhk", generate_3bhk),
        ("indian_apartment", generate_indian_apartment),
    ]

    plans = []
    ground_truth = {}

    for name, gen_func in generators:
        img, plan = gen_func(scale, dpi)

        # Save image
        img_path = output_dir / f"synth_{name}.png"
        cv2.imwrite(str(img_path), img)

        plans.append(plan)
        ground_truth[name] = plan.to_dict()

        print(f"Generated: {img_path.name}")

    # Save ground truth
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nSaved ground truth to: {gt_path}")

    return plans


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import sys

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/benchmark/raw")
    plans = generate_all_plans(output_dir)
    print(f"\nGenerated {len(plans)} synthetic plans")
