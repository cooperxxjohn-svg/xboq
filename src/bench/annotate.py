"""
Annotation Helper - CLI tool for creating ground truth annotations.

Supports:
1. Interactive OpenCV window for clicking points
2. CSV import for batch annotations
3. Validation of existing annotations

NOT a full UI - just essential tools for creating annotations.
"""

import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .annotation import (
    AnnotationSchema,
    RoomAnnotation,
    OpeningAnnotation,
    ScaleAnnotation,
    save_annotations,
    load_annotations,
    INDIA_ROOM_ALIASES,
)

logger = logging.getLogger(__name__)


class AnnotationHelper:
    """
    CLI-based annotation helper for floor plans.
    """

    def __init__(
        self,
        benchmark_dir: Path = Path("data/benchmark"),
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.raw_dir = self.benchmark_dir / "raw"
        self.annotations_dir = self.benchmark_dir / "annotations"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        # State for interactive mode
        self.current_points: List[Tuple[int, int]] = []
        self.current_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None

    def annotate_image(self, image_path: Path) -> AnnotationSchema:
        """
        Interactive annotation using OpenCV window.

        Controls:
        - Left click: Add point to current polygon
        - Right click: Complete current polygon
        - 'r': Start new room polygon
        - 'o': Start new opening bbox
        - 's': Set scale (click two points)
        - 'u': Undo last point
        - 'c': Clear current polygon
        - 'q': Quit and save
        - ESC: Quit without saving

        Args:
            image_path: Path to image

        Returns:
            AnnotationSchema
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = self.current_image.shape[:2]

        # Check for existing annotation
        ann_path = self.annotations_dir / f"{image_path.stem}.json"
        if ann_path.exists():
            annotation = load_annotations(ann_path)
            logger.info(f"Loaded existing annotation with {len(annotation.rooms)} rooms")
        else:
            annotation = AnnotationSchema(
                image_id=image_path.stem,
                image_path=str(image_path),
                image_width=w,
                image_height=h,
            )

        # Interactive mode
        mode = "room"  # room, opening, scale
        self.current_points = []
        room_count = len(annotation.rooms)
        opening_count = len(annotation.openings)

        def mouse_callback(event, x, y, flags, param):
            nonlocal mode, room_count, opening_count

            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append((x, y))
                self._update_display(annotation, mode)

            elif event == cv2.EVENT_RBUTTONDOWN:
                # Complete current annotation
                if len(self.current_points) >= 3 and mode == "room":
                    label = self._prompt_label("room")
                    room_count += 1
                    annotation.rooms.append(RoomAnnotation(
                        id=f"R{room_count}",
                        label=label,
                        polygon=self.current_points.copy(),
                    ))
                    logger.info(f"Added room: {label}")
                    self.current_points = []

                elif len(self.current_points) >= 2 and mode == "opening":
                    # Create bbox from first and last point
                    x1 = min(p[0] for p in self.current_points)
                    y1 = min(p[1] for p in self.current_points)
                    x2 = max(p[0] for p in self.current_points)
                    y2 = max(p[1] for p in self.current_points)

                    otype = self._prompt_label("opening")
                    opening_count += 1
                    annotation.openings.append(OpeningAnnotation(
                        id=f"O{opening_count}",
                        type=otype,
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                    ))
                    logger.info(f"Added opening: {otype}")
                    self.current_points = []

                elif len(self.current_points) >= 2 and mode == "scale":
                    length = self._prompt_length()
                    annotation.scale = ScaleAnnotation(
                        point1=self.current_points[0],
                        point2=self.current_points[1],
                        length_mm=length,
                    )
                    logger.info(f"Set scale: {length}mm")
                    self.current_points = []
                    mode = "room"

                self._update_display(annotation, mode)

        cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotation", mouse_callback)

        print("\n=== Annotation Mode ===")
        print("Left click: Add point")
        print("Right click: Complete polygon/box")
        print("'r': Room mode | 'o': Opening mode | 's': Scale mode")
        print("'u': Undo | 'c': Clear | 'q': Save & quit | ESC: Quit")
        print("=" * 25)

        while True:
            self._update_display(annotation, mode)
            cv2.imshow("Annotation", self.display_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                save_annotations(annotation, ann_path)
                print(f"\nSaved: {ann_path}")
                break
            elif key == 27:  # ESC
                print("\nQuit without saving")
                break
            elif key == ord('r'):
                mode = "room"
                self.current_points = []
                print("Mode: Room")
            elif key == ord('o'):
                mode = "opening"
                self.current_points = []
                print("Mode: Opening")
            elif key == ord('s'):
                mode = "scale"
                self.current_points = []
                print("Mode: Scale (click 2 points)")
            elif key == ord('u'):
                if self.current_points:
                    self.current_points.pop()
            elif key == ord('c'):
                self.current_points = []

        cv2.destroyAllWindows()
        return annotation

    def _update_display(self, annotation: AnnotationSchema, mode: str) -> None:
        """Update display image with annotations."""
        self.display_image = self.current_image.copy()

        # Draw existing rooms
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for i, room in enumerate(annotation.rooms):
            color = colors[i % len(colors)]
            if room.polygon:
                pts = np.array(room.polygon, dtype=np.int32)
                cv2.polylines(self.display_image, [pts], True, color, 2)
                cx, cy = room.centroid
                cv2.putText(
                    self.display_image, room.label,
                    (int(cx) - 20, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        # Draw existing openings
        for opening in annotation.openings:
            x, y, w, h = opening.bbox
            color = (0, 128, 255) if opening.type == "door" else (255, 128, 0)
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), color, 2)

        # Draw scale
        if annotation.scale:
            p1 = tuple(map(int, annotation.scale.point1))
            p2 = tuple(map(int, annotation.scale.point2))
            cv2.line(self.display_image, p1, p2, (0, 255, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                self.display_image, f"{annotation.scale.length_mm}mm",
                mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # Draw current points
        for pt in self.current_points:
            cv2.circle(self.display_image, pt, 5, (0, 255, 0), -1)

        if len(self.current_points) >= 2:
            pts = np.array(self.current_points, dtype=np.int32)
            if mode == "room":
                cv2.polylines(self.display_image, [pts], False, (0, 255, 0), 1)
            else:
                x1 = min(p[0] for p in self.current_points)
                y1 = min(p[1] for p in self.current_points)
                x2 = max(p[0] for p in self.current_points)
                y2 = max(p[1] for p in self.current_points)
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Mode indicator
        mode_text = f"Mode: {mode.upper()}"
        cv2.putText(
            self.display_image, mode_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    def _prompt_label(self, annotation_type: str) -> str:
        """Prompt user for label."""
        if annotation_type == "room":
            print("\nRoom types: Bedroom, Living, Kitchen, Toilet, Dining, Balcony, Pooja, Utility, Store, Passage")
            label = input("Enter room label: ").strip()
            return label if label else "Room"
        else:
            print("\nOpening types: door, window, ventilator")
            otype = input("Enter opening type: ").strip().lower()
            return otype if otype in ["door", "window", "ventilator"] else "door"

    def _prompt_length(self) -> float:
        """Prompt user for scale length."""
        try:
            length = input("Enter length in mm: ").strip()
            return float(length)
        except ValueError:
            return 1000.0

    def import_from_csv(self, csv_path: Path) -> None:
        """
        Import annotations from CSV file.

        CSV format:
        image_id,room_id,label,polygon,area_sqm
        test_1bhk,R1,Bedroom,"[(0,0),(100,0),(100,100),(0,100)]",16.0

        Args:
            csv_path: Path to CSV file
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Group by image
        image_data: Dict[str, List[Dict]] = {}

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id", "")
                if image_id not in image_data:
                    image_data[image_id] = []
                image_data[image_id].append(row)

        # Create annotations
        for image_id, rows in image_data.items():
            # Find image
            img_path = self.raw_dir / f"{image_id}.png"
            if not img_path.exists():
                img_path = self.raw_dir / f"{image_id}.jpg"
            if not img_path.exists():
                logger.warning(f"Image not found: {image_id}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            rooms = []
            for row in rows:
                # Parse polygon from string
                polygon_str = row.get("polygon", "[]")
                try:
                    polygon = eval(polygon_str)  # Simple parsing
                except:
                    polygon = []

                rooms.append(RoomAnnotation(
                    id=row.get("room_id", f"R{len(rooms)+1}"),
                    label=row.get("label", "Room"),
                    polygon=polygon,
                    area_sqm=float(row.get("area_sqm", 0)) if row.get("area_sqm") else None,
                ))

            annotation = AnnotationSchema(
                image_id=image_id,
                image_path=str(img_path),
                image_width=w,
                image_height=h,
                rooms=rooms,
            )

            ann_path = self.annotations_dir / f"{image_id}.json"
            save_annotations(annotation, ann_path)
            logger.info(f"Imported annotation: {image_id} ({len(rooms)} rooms)")

    def validate_annotations(self) -> List[str]:
        """
        Validate all annotations in benchmark.

        Returns:
            List of validation warnings
        """
        warnings = []

        for ann_path in self.annotations_dir.glob("*.json"):
            try:
                annotation = load_annotations(ann_path)

                # Check image exists
                if not Path(annotation.image_path).exists():
                    warnings.append(f"{ann_path.stem}: Image not found")

                # Check rooms
                for room in annotation.rooms:
                    if len(room.polygon) < 3:
                        warnings.append(f"{ann_path.stem}/{room.id}: Invalid polygon")
                    if not room.label:
                        warnings.append(f"{ann_path.stem}/{room.id}: Missing label")

                # Check openings
                for opening in annotation.openings:
                    x, y, w, h = opening.bbox
                    if w <= 0 or h <= 0:
                        warnings.append(f"{ann_path.stem}/{opening.id}: Invalid bbox")

            except Exception as e:
                warnings.append(f"{ann_path.stem}: Load error - {e}")

        return warnings

    def interactive_mode(self) -> None:
        """Run interactive annotation session."""
        print("\n=== Annotation Helper ===")
        print("Commands:")
        print("  list     - List unannotated images")
        print("  annotate <id> - Annotate specific image")
        print("  validate - Validate all annotations")
        print("  stats    - Show annotation statistics")
        print("  quit     - Exit")
        print("=" * 25)

        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == "quit":
                    break
                elif cmd[0] == "list":
                    self._list_unannotated()
                elif cmd[0] == "annotate" and len(cmd) > 1:
                    img_path = self.raw_dir / f"{cmd[1]}.png"
                    if not img_path.exists():
                        img_path = self.raw_dir / f"{cmd[1]}.jpg"
                    if img_path.exists():
                        self.annotate_image(img_path)
                    else:
                        print(f"Image not found: {cmd[1]}")
                elif cmd[0] == "validate":
                    warnings = self.validate_annotations()
                    if warnings:
                        print(f"Found {len(warnings)} warnings:")
                        for w in warnings[:10]:
                            print(f"  - {w}")
                    else:
                        print("All annotations valid!")
                elif cmd[0] == "stats":
                    self._show_stats()
                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _list_unannotated(self) -> None:
        """List images without annotations."""
        annotated = set(p.stem for p in self.annotations_dir.glob("*.json"))
        raw_images = list(self.raw_dir.glob("*.png")) + list(self.raw_dir.glob("*.jpg"))

        unannotated = [p for p in raw_images if p.stem not in annotated]

        if unannotated:
            print(f"Unannotated images ({len(unannotated)}):")
            for img in unannotated[:20]:
                print(f"  - {img.stem}")
        else:
            print("All images annotated!")

    def _show_stats(self) -> None:
        """Show annotation statistics."""
        total_images = len(list(self.raw_dir.glob("*.png"))) + len(list(self.raw_dir.glob("*.jpg")))
        total_annotations = len(list(self.annotations_dir.glob("*.json")))

        total_rooms = 0
        total_openings = 0
        label_counts = {}

        for ann_path in self.annotations_dir.glob("*.json"):
            try:
                annotation = load_annotations(ann_path)
                total_rooms += len(annotation.rooms)
                total_openings += len(annotation.openings)

                for room in annotation.rooms:
                    label_counts[room.label] = label_counts.get(room.label, 0) + 1
            except:
                pass

        print(f"\nAnnotation Statistics:")
        print(f"  Images: {total_annotations}/{total_images}")
        print(f"  Rooms: {total_rooms}")
        print(f"  Openings: {total_openings}")
        print(f"\nRoom labels:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {label}: {count}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Annotation helper")
    parser.add_argument("--benchmark-dir", default="data/benchmark")
    parser.add_argument("--image", help="Image to annotate")
    parser.add_argument("--from-csv", help="Import from CSV")
    parser.add_argument("--validate", action="store_true", help="Validate annotations")

    args = parser.parse_args()

    helper = AnnotationHelper(benchmark_dir=Path(args.benchmark_dir))

    if args.validate:
        warnings = helper.validate_annotations()
        for w in warnings:
            print(w)
    elif args.from_csv:
        helper.import_from_csv(Path(args.from_csv))
    elif args.image:
        helper.annotate_image(Path(args.image))
    else:
        helper.interactive_mode()
