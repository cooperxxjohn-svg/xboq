"""
Benchmark Dataset Builder.

Creates benchmark dataset from:
1. Existing plans in data/plans/
2. Downloaded public floor plan datasets
3. Synthetic test images

Generates manifest.json with image metadata.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkBuilder:
    """
    Builds benchmark dataset for floor plan analysis.
    """

    def __init__(
        self,
        benchmark_dir: Path = Path("data/benchmark"),
        use_existing: bool = True,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.use_existing = use_existing

        # Create directories
        self.raw_dir = self.benchmark_dir / "raw"
        self.processed_dir = self.benchmark_dir / "processed"
        self.annotations_dir = self.benchmark_dir / "annotations"

        for d in [self.raw_dir, self.processed_dir, self.annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        n_images: int = 30,
        download_if_needed: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build benchmark dataset.

        Args:
            n_images: Target number of images
            download_if_needed: Download public dataset if not enough local images

        Returns:
            List of manifest entries
        """
        manifest = []

        # Step 1: Collect existing plans
        if self.use_existing:
            existing = self._collect_existing_plans()
            manifest.extend(existing)
            logger.info(f"Found {len(existing)} existing plans")

        # Step 2: Convert existing ground truth
        self._convert_existing_ground_truth()

        # Step 3: Download if needed
        if len(manifest) < n_images and download_if_needed:
            needed = n_images - len(manifest)
            logger.info(f"Need {needed} more images, attempting download...")
            downloaded = self._download_public_dataset(needed)
            manifest.extend(downloaded)

        # Step 4: Generate synthetic if still not enough
        if len(manifest) < n_images:
            needed = n_images - len(manifest)
            logger.info(f"Generating {needed} synthetic images...")
            synthetic = self._generate_synthetic_plans(needed)
            manifest.extend(synthetic)

        # Save manifest
        self._save_manifest(manifest)

        return manifest

    def _collect_existing_plans(self) -> List[Dict[str, Any]]:
        """Collect existing plans from data/plans/."""
        existing_dir = Path("data/plans")
        manifest = []

        if not existing_dir.exists():
            return manifest

        for img_path in existing_dir.glob("*.png"):
            if img_path.stem.startswith("test_") or "schedule" not in img_path.stem:
                # Copy to raw dir
                dest = self.raw_dir / img_path.name
                if not dest.exists():
                    shutil.copy(img_path, dest)

                # Get image dimensions
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    manifest.append({
                        "id": img_path.stem,
                        "path": str(dest),
                        "width": w,
                        "height": h,
                        "source": "existing",
                    })

        for img_path in existing_dir.glob("*.jpg"):
            dest = self.raw_dir / img_path.name
            if not dest.exists():
                shutil.copy(img_path, dest)

            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                manifest.append({
                    "id": img_path.stem,
                    "path": str(dest),
                    "width": w,
                    "height": h,
                    "source": "existing",
                })

        return manifest

    def _convert_existing_ground_truth(self) -> None:
        """Convert existing ground_truth.json to new annotation format."""
        gt_path = Path("data/plans/ground_truth.json")
        if not gt_path.exists():
            return

        with open(gt_path) as f:
            ground_truth = json.load(f)

        for plan_id, gt_data in ground_truth.items():
            ann_path = self.annotations_dir / f"test_{plan_id}.json"

            # Find corresponding image
            img_name = f"test_{plan_id}.png"
            img_path = self.raw_dir / img_name

            if not img_path.exists():
                # Try to copy from existing
                src = Path("data/plans") / img_name
                if src.exists():
                    shutil.copy(src, img_path)
                else:
                    continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Convert rooms to new format
            rooms = []
            for i, room_data in enumerate(gt_data.get("rooms", [])):
                # Convert bbox_mm to pixel polygon (simplified - just bbox corners)
                bbox_mm = room_data.get("bbox_mm", [0, 0, 0, 0])
                scale = gt_data.get("scale", 100)
                dpi = gt_data.get("dpi", 300)

                # mm to pixels
                px_per_mm = dpi / 25.4 / scale

                x_mm, y_mm, w_mm, h_mm = bbox_mm
                x = x_mm * px_per_mm
                y = y_mm * px_per_mm
                width = w_mm * px_per_mm
                height = h_mm * px_per_mm

                polygon = [
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                ]

                rooms.append({
                    "id": f"R{i+1}",
                    "label": room_data.get("name", "Room"),
                    "polygon": polygon,
                    "area_sqm": room_data.get("area_sqm"),
                    "aliases": [],
                })

            # Build scale annotation
            scale_data = None
            if gt_data.get("scale") and gt_data.get("width_mm"):
                # Use top edge as scale reference
                px_per_mm = dpi / 25.4 / gt_data["scale"]
                scale_data = {
                    "point1": [0, 0],
                    "point2": [gt_data["width_mm"] * px_per_mm, 0],
                    "length_mm": gt_data["width_mm"],
                }

            annotation = {
                "image_id": f"test_{plan_id}",
                "image_path": str(img_path),
                "image_width": w,
                "image_height": h,
                "rooms": rooms,
                "openings": [],
                "scale": scale_data,
                "metadata": {
                    "source": "ground_truth.json",
                    "original_scale": gt_data.get("scale"),
                },
            }

            with open(ann_path, "w") as f:
                json.dump(annotation, f, indent=2)

            logger.info(f"Converted annotation: {ann_path.name}")

    def _download_public_dataset(self, n_images: int) -> List[Dict[str, Any]]:
        """
        Download public floor plan dataset.

        Uses permissive sources:
        - Wikimedia Commons floor plans (CC-BY-SA)
        - Public domain architectural drawings
        """
        manifest = []

        try:
            manifest = self._download_wikimedia_floorplans(n_images)
        except Exception as e:
            logger.warning(f"Wikimedia download failed: {e}")

        # Create a README with additional manual sources
        readme_path = self.benchmark_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("# Benchmark Dataset\n\n")
            f.write("## Adding Images\n\n")
            f.write("Place floor plan images (PNG/JPG) in the `raw/` directory.\n\n")
            f.write("## Public Datasets\n\n")
            f.write("Consider downloading from:\n")
            f.write("- CubiCasa5k: https://github.com/CubiCasa/CubiCasa5k\n")
            f.write("- R2V: https://github.com/art-programmer/FloorplanTransformation\n")
            f.write("- LIFULL: https://www.nii.ac.jp/dsc/idr/lifull/\n\n")
            f.write("## India-specific Plans\n\n")
            f.write("For India-specific testing, collect plans that include:\n")
            f.write("- Indian room names (Toilet/WC, Pooja, etc.)\n")
            f.write("- Mixed units (mm/feet)\n")
            f.write("- Typical Indian apartment layouts\n")

        return manifest

    def _download_wikimedia_floorplans(self, n_images: int) -> List[Dict[str, Any]]:
        """Download floor plans from Wikimedia Commons (CC licensed)."""
        import urllib.request
        import urllib.parse

        manifest = []

        # Wikimedia Commons API to search for floor plans
        # These are architectural floor plan images with permissive licenses
        search_terms = [
            "floor plan apartment",
            "floor plan residential",
            "architectural floor plan",
            "house floor plan",
        ]

        base_api = "https://commons.wikimedia.org/w/api.php"

        downloaded = 0
        for term in search_terms:
            if downloaded >= n_images:
                break

            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": f'"{term}"',
                "srnamespace": "6",  # File namespace
                "srlimit": min(10, n_images - downloaded),
            }

            url = f"{base_api}?{urllib.parse.urlencode(params)}"

            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = json.loads(response.read().decode())

                results = data.get("query", {}).get("search", [])

                for result in results:
                    if downloaded >= n_images:
                        break

                    title = result.get("title", "")
                    if not title.startswith("File:"):
                        continue

                    # Get image info
                    info_params = {
                        "action": "query",
                        "format": "json",
                        "titles": title,
                        "prop": "imageinfo",
                        "iiprop": "url|size|mime",
                    }
                    info_url = f"{base_api}?{urllib.parse.urlencode(info_params)}"

                    with urllib.request.urlopen(info_url, timeout=30) as info_response:
                        info_data = json.loads(info_response.read().decode())

                    pages = info_data.get("query", {}).get("pages", {})
                    for page_id, page_data in pages.items():
                        if page_id == "-1":
                            continue

                        image_info = page_data.get("imageinfo", [{}])[0]
                        img_url = image_info.get("url")
                        mime = image_info.get("mime", "")

                        if not img_url or mime not in ["image/png", "image/jpeg"]:
                            continue

                        # Download the image
                        file_ext = ".png" if "png" in mime else ".jpg"
                        file_id = f"public_{downloaded:03d}"
                        dest_path = self.raw_dir / f"{file_id}{file_ext}"

                        logger.info(f"Downloading: {title[:50]}...")

                        try:
                            urllib.request.urlretrieve(img_url, str(dest_path))

                            # Verify it's a valid image
                            img = cv2.imread(str(dest_path))
                            if img is not None and img.size > 0:
                                h, w = img.shape[:2]
                                manifest.append({
                                    "id": file_id,
                                    "path": str(dest_path),
                                    "width": w,
                                    "height": h,
                                    "source": "wikimedia",
                                    "source_url": img_url,
                                })
                                downloaded += 1
                                logger.info(f"Downloaded: {file_id} ({w}x{h})")
                            else:
                                dest_path.unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to download {title}: {e}")
                            if dest_path.exists():
                                dest_path.unlink()

            except Exception as e:
                logger.warning(f"Search failed for '{term}': {e}")

        logger.info(f"Downloaded {len(manifest)} images from Wikimedia Commons")
        return manifest

    def _generate_synthetic_plans(self, n_images: int) -> List[Dict[str, Any]]:
        """Generate synthetic floor plan images for testing."""
        manifest = []

        try:
            # Import the synthetic generator functions
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
            from synthetic_generator import (
                generate_simple_rectangle,
                generate_simple_1bhk,
                generate_2bhk_with_utility,
            )

            generators = [
                generate_simple_rectangle,
                generate_simple_1bhk,
                generate_2bhk_with_utility,
            ]

            for i in range(n_images):
                plan_id = f"synthetic_{i:03d}"
                img_path = self.raw_dir / f"{plan_id}.png"

                # Cycle through generators
                gen_func = generators[i % len(generators)]
                img, plan = gen_func()
                cv2.imwrite(str(img_path), img)

                h, w = img.shape[:2]

                # Create annotation from plan
                ann_rooms = []
                scale = plan.scale
                dpi = plan.dpi

                for j, room in enumerate(plan.rooms):
                    # Convert bbox_mm to pixel polygon
                    x_mm, y_mm, w_mm, h_mm = room.bbox
                    px_per_mm = dpi / 25.4 / scale
                    offset = 50  # Same offset used in generator

                    x = x_mm * px_per_mm + offset
                    y = y_mm * px_per_mm + offset
                    width = w_mm * px_per_mm
                    height = h_mm * px_per_mm

                    polygon = [
                        (x, y),
                        (x + width, y),
                        (x + width, y + height),
                        (x, y + height),
                    ]

                    ann_rooms.append({
                        "id": f"R{j+1}",
                        "label": room.name,
                        "polygon": polygon,
                        "area_sqm": room.area_sqm,
                        "aliases": [],
                    })

                annotation = {
                    "image_id": plan_id,
                    "image_path": str(img_path),
                    "image_width": w,
                    "image_height": h,
                    "rooms": ann_rooms,
                    "openings": [],
                    "scale": {
                        "point1": [50, 50],
                        "point2": [50 + plan.width_mm * dpi / 25.4 / scale, 50],
                        "length_mm": plan.width_mm,
                    },
                    "metadata": {"source": "synthetic", "original_scale": scale},
                }

                ann_path = self.annotations_dir / f"{plan_id}.json"
                with open(ann_path, "w") as f:
                    json.dump(annotation, f, indent=2)

                manifest.append({
                    "id": plan_id,
                    "path": str(img_path),
                    "width": w,
                    "height": h,
                    "source": "synthetic",
                })

                logger.info(f"Generated synthetic plan: {plan_id}")

        except ImportError as e:
            logger.warning(f"Synthetic generator import failed: {e}")
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")

        return manifest

    def _save_manifest(self, manifest: List[Dict[str, Any]]) -> None:
        """Save manifest to JSON."""
        manifest_path = self.benchmark_dir / "manifest.json"

        manifest_data = {
            "version": "1.0",
            "created": str(Path.cwd()),
            "total_images": len(manifest),
            "images": manifest,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Saved manifest with {len(manifest)} images")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build benchmark dataset")
    parser.add_argument("--n", type=int, default=30, help="Number of images")
    parser.add_argument("--benchmark-dir", default="data/benchmark", help="Output directory")
    parser.add_argument("--use-existing", action="store_true", default=True)
    parser.add_argument("--download", action="store_true", help="Download if needed")

    args = parser.parse_args()

    builder = BenchmarkBuilder(
        benchmark_dir=Path(args.benchmark_dir),
        use_existing=args.use_existing,
    )

    manifest = builder.build(n_images=args.n, download_if_needed=args.download)
    print(f"\nBuilt benchmark with {len(manifest)} images")
