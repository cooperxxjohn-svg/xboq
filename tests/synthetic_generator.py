"""
Synthetic Floor Plan Generator
Creates test floor plans with known ground truth for validation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


@dataclass
class SyntheticRoom:
    """A synthetic room with ground truth."""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h in mm
    area_sqm: float


@dataclass
class SyntheticPlan:
    """A synthetic floor plan."""
    name: str
    width_mm: int
    height_mm: int
    rooms: List[SyntheticRoom]
    scale: int  # 1:scale
    dpi: int


def mm_to_pixels(mm: float, scale: int, dpi: int) -> int:
    """Convert mm to pixels at given scale and DPI."""
    # At scale 1:100, 1mm real = 0.01mm on drawing
    # Drawing mm = real mm / scale
    drawing_mm = mm / scale
    # Pixels = drawing_mm * (dpi / 25.4)
    pixels = drawing_mm * dpi / 25.4
    return int(pixels)


def generate_simple_1bhk(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """
    Generate a simple 1BHK apartment floor plan.

    Layout:
    +------------------+----------+
    |                  |  Bath    |
    |     Bedroom      |  3x2m    |
    |     4x4m         +----------+
    |                  | Kitchen  |
    +--------+---------+  3x3m    |
    | Balcony|  Living |          |
    | 2x1.5m |  4x4m   +----------+
    +--------+         |
    |        +---------+
    +------------------+
    """

    # Room definitions in mm (real dimensions)
    rooms = [
        SyntheticRoom("Bedroom", (0, 0, 4000, 4000), 16.0),
        SyntheticRoom("Living", (4000, 2000, 4000, 4000), 16.0),
        SyntheticRoom("Kitchen", (4000, 0, 3000, 3000), 9.0),
        SyntheticRoom("Toilet", (4000, 0, 3000, 2000), 6.0),  # Adjusted
        SyntheticRoom("Balcony", (0, 4000, 2000, 1500), 3.0),
    ]

    # Recalculate with proper layout
    rooms = [
        SyntheticRoom("Bedroom", (0, 0, 4000, 4000), 16.0),
        SyntheticRoom("Living", (0, 4000, 4000, 4000), 16.0),
        SyntheticRoom("Kitchen", (4000, 4000, 3000, 3000), 9.0),
        SyntheticRoom("Toilet", (4000, 0, 3000, 2000), 6.0),
        SyntheticRoom("Bath", (4000, 2000, 3000, 2000), 6.0),
        SyntheticRoom("Balcony", (0, 8000, 2000, 1500), 3.0),
    ]

    # Plan dimensions
    plan_width_mm = 7000
    plan_height_mm = 9500

    # Convert to pixels
    w_px = mm_to_pixels(plan_width_mm, scale, dpi)
    h_px = mm_to_pixels(plan_height_mm, scale, dpi)

    # Create image (white background)
    img = np.ones((h_px + 100, w_px + 100, 3), dtype=np.uint8) * 255

    # Offset for border
    offset = 50

    # Draw walls (thick black lines)
    wall_thickness = max(2, mm_to_pixels(200, scale, dpi))

    for room in rooms:
        x = mm_to_pixels(room.bbox[0], scale, dpi) + offset
        y = mm_to_pixels(room.bbox[1], scale, dpi) + offset
        w = mm_to_pixels(room.bbox[2], scale, dpi)
        h = mm_to_pixels(room.bbox[3], scale, dpi)

        # Draw room rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), wall_thickness)

        # Add room label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_size = cv2.getTextSize(room.name, font, 0.5, 1)[0]
        label_x = x + w // 2 - label_size[0] // 2
        label_y = y + h // 2

        cv2.putText(img, room.name, (label_x, label_y), font, 0.5, (0, 0, 0), 1)

        # Add area annotation
        area_text = f"{room.area_sqm:.0f} sqm"
        area_size = cv2.getTextSize(area_text, font, 0.4, 1)[0]
        cv2.putText(img, area_text, (label_x, label_y + 20), font, 0.4, (100, 100, 100), 1)

    # Add scale note
    scale_text = f"Scale 1:{scale}"
    cv2.putText(img, scale_text, (10, h_px + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add dimension line example
    dim_start = mm_to_pixels(0, scale, dpi) + offset
    dim_end = mm_to_pixels(4000, scale, dpi) + offset
    dim_y = h_px + 50

    cv2.line(img, (dim_start, dim_y), (dim_end, dim_y), (0, 0, 0), 1)
    cv2.line(img, (dim_start, dim_y - 5), (dim_start, dim_y + 5), (0, 0, 0), 1)
    cv2.line(img, (dim_end, dim_y - 5), (dim_end, dim_y + 5), (0, 0, 0), 1)
    cv2.putText(img, "4000", ((dim_start + dim_end) // 2 - 20, dim_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    plan = SyntheticPlan(
        name="simple_1bhk",
        width_mm=plan_width_mm,
        height_mm=plan_height_mm,
        rooms=rooms,
        scale=scale,
        dpi=dpi
    )

    return img, plan


def generate_2bhk_with_utility(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """
    Generate a 2BHK apartment with utility room.

    More complex layout with passage and utility.
    """

    rooms = [
        SyntheticRoom("Master Bedroom", (0, 0, 4500, 4000), 18.0),
        SyntheticRoom("Bedroom 2", (4500, 0, 3500, 3500), 12.25),
        SyntheticRoom("Living", (0, 4000, 5000, 4500), 22.5),
        SyntheticRoom("Dining", (5000, 4000, 3000, 3000), 9.0),
        SyntheticRoom("Kitchen", (5000, 7000, 3000, 3000), 9.0),
        SyntheticRoom("Toilet 1", (4500, 0, 2000, 2000), 4.0),
        SyntheticRoom("Toilet 2", (6500, 0, 1500, 2000), 3.0),
        SyntheticRoom("Utility", (0, 8500, 2000, 1500), 3.0),
        SyntheticRoom("Balcony 1", (0, 10000, 2500, 1500), 3.75),
        SyntheticRoom("Balcony 2", (5000, 10000, 3000, 1500), 4.5),
        SyntheticRoom("Passage", (4500, 2000, 1000, 2000), 2.0),
    ]

    plan_width_mm = 8000
    plan_height_mm = 11500

    w_px = mm_to_pixels(plan_width_mm, scale, dpi)
    h_px = mm_to_pixels(plan_height_mm, scale, dpi)

    img = np.ones((h_px + 100, w_px + 100, 3), dtype=np.uint8) * 255

    offset = 50
    wall_thickness = max(2, mm_to_pixels(150, scale, dpi))

    for room in rooms:
        x = mm_to_pixels(room.bbox[0], scale, dpi) + offset
        y = mm_to_pixels(room.bbox[1], scale, dpi) + offset
        w = mm_to_pixels(room.bbox[2], scale, dpi)
        h = mm_to_pixels(room.bbox[3], scale, dpi)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), wall_thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Truncate long labels
        label = room.name if len(room.name) <= 10 else room.name[:10]
        label_size = cv2.getTextSize(label, font, 0.4, 1)[0]
        label_x = max(x + 5, x + w // 2 - label_size[0] // 2)
        label_y = y + h // 2

        cv2.putText(img, label, (label_x, label_y), font, 0.4, (0, 0, 0), 1)

    scale_text = f"Scale 1:{scale}"
    cv2.putText(img, scale_text, (10, h_px + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    plan = SyntheticPlan(
        name="2bhk_utility",
        width_mm=plan_width_mm,
        height_mm=plan_height_mm,
        rooms=rooms,
        scale=scale,
        dpi=dpi
    )

    return img, plan


def generate_simple_rectangle(scale: int = 100, dpi: int = 300) -> Tuple[np.ndarray, SyntheticPlan]:
    """Generate simplest test case - single rectangle."""

    rooms = [
        SyntheticRoom("Room", (0, 0, 10000, 10000), 100.0),
    ]

    plan_width_mm = 10000
    plan_height_mm = 10000

    w_px = mm_to_pixels(plan_width_mm, scale, dpi)
    h_px = mm_to_pixels(plan_height_mm, scale, dpi)

    img = np.ones((h_px + 100, w_px + 100, 3), dtype=np.uint8) * 255

    offset = 50
    wall_thickness = max(3, mm_to_pixels(200, scale, dpi))

    x = offset
    y = offset
    cv2.rectangle(img, (x, y), (x + w_px, y + h_px), (0, 0, 0), wall_thickness)

    cv2.putText(img, "Room", (w_px // 2, h_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "100 sqm", (w_px // 2, h_px // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(img, f"Scale 1:{scale}", (10, h_px + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add dimension
    cv2.putText(img, "10000", (w_px // 2 - 30, h_px + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    plan = SyntheticPlan(
        name="simple_rectangle",
        width_mm=plan_width_mm,
        height_mm=plan_height_mm,
        rooms=rooms,
        scale=scale,
        dpi=dpi
    )

    return img, plan


def save_synthetic_plans(output_dir: Path):
    """Generate and save all synthetic test plans."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = []

    # Simple rectangle test
    img, plan = generate_simple_rectangle()
    cv2.imwrite(str(output_dir / "test_simple_rectangle.png"), img)
    plans.append(plan)
    print(f"Generated: test_simple_rectangle.png")

    # 1BHK test
    img, plan = generate_simple_1bhk()
    cv2.imwrite(str(output_dir / "test_1bhk.png"), img)
    plans.append(plan)
    print(f"Generated: test_1bhk.png")

    # 2BHK test
    img, plan = generate_2bhk_with_utility()
    cv2.imwrite(str(output_dir / "test_2bhk.png"), img)
    plans.append(plan)
    print(f"Generated: test_2bhk.png")

    # Save ground truth
    ground_truth = {
        plan.name: {
            'width_mm': plan.width_mm,
            'height_mm': plan.height_mm,
            'scale': plan.scale,
            'dpi': plan.dpi,
            'rooms': [
                {
                    'name': room.name,
                    'bbox_mm': room.bbox,
                    'area_sqm': room.area_sqm
                }
                for room in plan.rooms
            ]
        }
        for plan in plans
    }

    with open(output_dir / "ground_truth.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nSaved ground truth to: {output_dir / 'ground_truth.json'}")
    return plans


if __name__ == "__main__":
    import sys

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/plans")
    save_synthetic_plans(output_dir)
