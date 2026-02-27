"""
Synthetic Structural Drawing Generator
Creates test drawings for column layouts, beam layouts, and foundation plans.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict


def create_column_layout(
    width: int = 1000,
    height: int = 1000,
    grid_size_px: int = 200,
    column_size_px: int = 20
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Create a synthetic column layout drawing.

    Returns:
        (image, column_data)
    """
    # White background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    columns = []
    col_num = 1

    # Draw grid lines (light gray)
    for x in range(grid_size_px, width, grid_size_px):
        cv2.line(image, (x, 0), (x, height), (200, 200, 200), 1)
    for y in range(grid_size_px, height, grid_size_px):
        cv2.line(image, (0, y), (width, y), (200, 200, 200), 1)

    # Draw columns at grid intersections
    for y in range(grid_size_px, height - grid_size_px//2, grid_size_px):
        for x in range(grid_size_px, width - grid_size_px//2, grid_size_px):
            # Column rectangle (filled black)
            x1 = x - column_size_px // 2
            y1 = y - column_size_px // 2
            x2 = x + column_size_px // 2
            y2 = y + column_size_px // 2

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # Label
            label = f"C{col_num}"
            cv2.putText(
                image, label,
                (x - 15, y - column_size_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )

            columns.append({
                'label': label,
                'center': (x, y),
                'bbox': (x1, y1, column_size_px, column_size_px)
            })

            col_num += 1

    # Add scale note
    cv2.putText(
        image, "SCALE 1:100",
        (50, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )

    # Add title
    cv2.putText(
        image, "COLUMN LAYOUT PLAN",
        (width // 2 - 100, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

    return image, columns


def create_beam_layout(
    width: int = 1000,
    height: int = 1000,
    grid_size_px: int = 200,
    column_size_px: int = 20,
    beam_thickness_px: int = 10
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    Create a synthetic beam layout drawing.

    Returns:
        (image, column_data, beam_data)
    """
    # Start with column layout
    image, columns = create_column_layout(width, height, grid_size_px, column_size_px)

    beams = []
    beam_num = 1

    # Draw horizontal beams between columns
    for y in range(grid_size_px, height - grid_size_px//2, grid_size_px):
        for x in range(grid_size_px, width - grid_size_px - grid_size_px//2, grid_size_px):
            x_end = x + grid_size_px

            # Draw beam line
            y1 = y - beam_thickness_px // 2
            y2 = y + beam_thickness_px // 2

            cv2.rectangle(image, (x + column_size_px//2, y1),
                         (x_end - column_size_px//2, y2), (100, 100, 100), -1)

            # Label at midpoint
            mid_x = (x + x_end) // 2
            label = f"B{beam_num}"
            cv2.putText(
                image, label,
                (mid_x - 10, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1
            )

            beams.append({
                'label': label,
                'start': (x, y),
                'end': (x_end, y),
                'direction': 'horizontal'
            })
            beam_num += 1

    # Draw vertical beams
    for x in range(grid_size_px, width - grid_size_px//2, grid_size_px):
        for y in range(grid_size_px, height - grid_size_px - grid_size_px//2, grid_size_px):
            y_end = y + grid_size_px

            x1 = x - beam_thickness_px // 2
            x2 = x + beam_thickness_px // 2

            cv2.rectangle(image, (x1, y + column_size_px//2),
                         (x2, y_end - column_size_px//2), (100, 100, 100), -1)

            mid_y = (y + y_end) // 2
            label = f"B{beam_num}"
            cv2.putText(
                image, label,
                (x + 15, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1
            )

            beams.append({
                'label': label,
                'start': (x, y),
                'end': (x, y_end),
                'direction': 'vertical'
            })
            beam_num += 1

    # Update title
    cv2.rectangle(image, (width//2 - 120, 10), (width//2 + 120, 45), (255, 255, 255), -1)
    cv2.putText(
        image, "BEAM LAYOUT PLAN",
        (width // 2 - 90, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

    return image, columns, beams


def create_foundation_plan(
    width: int = 1000,
    height: int = 1000,
    grid_size_px: int = 200,
    column_size_px: int = 20,
    footing_size_px: int = 80
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    Create a synthetic foundation plan.

    Returns:
        (image, column_data, footing_data)
    """
    # White background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    columns = []
    footings = []
    col_num = 1
    ftg_num = 1

    # Draw footings with hatching at grid intersections
    for y in range(grid_size_px, height - grid_size_px//2, grid_size_px):
        for x in range(grid_size_px, width - grid_size_px//2, grid_size_px):
            # Footing rectangle (larger, hatched)
            fx1 = x - footing_size_px // 2
            fy1 = y - footing_size_px // 2
            fx2 = x + footing_size_px // 2
            fy2 = y + footing_size_px // 2

            # Footing outline
            cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 0, 0), 2)

            # Hatching pattern
            for i in range(fx1, fx2, 10):
                cv2.line(image, (i, fy1), (i + (fy2-fy1), fy2), (150, 150, 150), 1)

            # Footing label
            label = f"F{ftg_num}"
            cv2.putText(
                image, label,
                (fx1, fy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )

            footings.append({
                'label': label,
                'center': (x, y),
                'bbox': (fx1, fy1, footing_size_px, footing_size_px),
                'type': 'isolated'
            })
            ftg_num += 1

            # Column rectangle inside footing (smaller, filled)
            cx1 = x - column_size_px // 2
            cy1 = y - column_size_px // 2
            cx2 = x + column_size_px // 2
            cy2 = y + column_size_px // 2

            cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)

            columns.append({
                'label': f"C{col_num}",
                'center': (x, y),
                'bbox': (cx1, cy1, column_size_px, column_size_px)
            })
            col_num += 1

    # Add scale note
    cv2.putText(
        image, "SCALE 1:100",
        (50, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )

    # Add title
    cv2.putText(
        image, "FOUNDATION PLAN",
        (width // 2 - 80, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

    return image, columns, footings


def create_column_schedule_image(
    columns: List[Dict],
    width: int = 600,
    row_height: int = 30
) -> np.ndarray:
    """
    Create a column schedule table image.
    """
    height = (len(columns) + 2) * row_height + 50

    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(
        image, "COLUMN SCHEDULE",
        (width // 2 - 80, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

    # Header
    y = 60
    headers = ['MARK', 'SIZE (mm)', 'CONCRETE', 'REMARKS']
    x_positions = [20, 100, 250, 400]

    for i, header in enumerate(headers):
        cv2.putText(image, header, (x_positions[i], y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.line(image, (10, y + 5), (width - 10, y + 5), (0, 0, 0), 1)

    # Data rows
    sizes = ['230x450', '300x600', '230x230']  # Cycle through sizes

    for i, col in enumerate(columns):
        y = 60 + (i + 1) * row_height + 10

        label = col.get('label', f'C{i+1}')
        size = sizes[i % len(sizes)]

        cv2.putText(image, label, (x_positions[0], y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, size, (x_positions[1], y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, 'M25', (x_positions[2], y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return image


def save_synthetic_structural(output_dir: Path):
    """Save all synthetic structural drawings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Column layout
    col_image, columns = create_column_layout()
    cv2.imwrite(str(output_dir / "test_column_layout.png"), col_image)
    print(f"Created column layout with {len(columns)} columns")

    # Beam layout
    beam_image, _, beams = create_beam_layout()
    cv2.imwrite(str(output_dir / "test_beam_layout.png"), beam_image)
    print(f"Created beam layout with {len(beams)} beams")

    # Foundation plan
    ftg_image, _, footings = create_foundation_plan()
    cv2.imwrite(str(output_dir / "test_foundation_plan.png"), ftg_image)
    print(f"Created foundation plan with {len(footings)} footings")

    # Column schedule
    sched_image = create_column_schedule_image(columns)
    cv2.imwrite(str(output_dir / "test_column_schedule.png"), sched_image)
    print("Created column schedule")

    return {
        'columns': columns,
        'beams': beams,
        'footings': footings
    }


if __name__ == "__main__":
    import sys

    output_dir = Path(__file__).parent.parent / "data" / "plans"
    data = save_synthetic_structural(output_dir)

    print(f"\nGenerated {len(data['columns'])} columns, "
          f"{len(data['beams'])} beams, "
          f"{len(data['footings'])} footings")
