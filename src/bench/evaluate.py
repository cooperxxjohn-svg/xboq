"""
Benchmark Evaluator - Runs pipeline on benchmark dataset and computes metrics.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from .annotation import AnnotationSchema, load_annotations
from .metrics import (
    RoomMetrics,
    OpeningMetrics,
    ScaleMetrics,
    compute_room_metrics,
    compute_opening_metrics,
    compute_scale_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single image."""
    image_id: str
    image_path: str
    success: bool
    run_time_sec: float

    # Metrics
    room_metrics: Optional[RoomMetrics] = None
    opening_metrics: Optional[OpeningMetrics] = None
    scale_metrics: Optional[ScaleMetrics] = None

    # Raw predictions
    pred_rooms: List[Dict] = field(default_factory=list)
    pred_openings: List[Dict] = field(default_factory=list)
    pred_scale: Optional[Dict] = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "success": self.success,
            "run_time_sec": self.run_time_sec,
            "room_metrics": self.room_metrics.to_dict() if self.room_metrics else None,
            "opening_metrics": self.opening_metrics.to_dict() if self.opening_metrics else None,
            "scale_metrics": self.scale_metrics.to_dict() if self.scale_metrics else None,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class BenchmarkResults:
    """Aggregated results across benchmark dataset."""
    num_images: int = 0
    num_success: int = 0
    total_time_sec: float = 0.0

    # Aggregated room metrics
    mean_room_f1_50: float = 0.0
    mean_room_f1_75: float = 0.0
    mean_label_accuracy: float = 0.0
    mean_area_error: float = 0.0

    # Aggregated opening metrics
    mean_opening_f1: float = 0.0
    mean_door_f1: float = 0.0
    mean_window_f1: float = 0.0

    # Scale metrics
    mean_scale_error: float = 0.0
    scale_detection_rate: float = 0.0

    # Segmentation issues
    total_oversegmented: int = 0
    total_undersegmented: int = 0
    total_missed: int = 0
    total_spurious: int = 0

    # Individual results
    results: List[EvaluationResult] = field(default_factory=list)

    # Config used
    params_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_images": self.num_images,
            "num_success": self.num_success,
            "success_rate": self.num_success / max(self.num_images, 1),
            "total_time_sec": self.total_time_sec,
            "mean_room_f1_50": self.mean_room_f1_50,
            "mean_room_f1_75": self.mean_room_f1_75,
            "mean_label_accuracy": self.mean_label_accuracy,
            "mean_area_error": self.mean_area_error,
            "mean_opening_f1": self.mean_opening_f1,
            "mean_door_f1": self.mean_door_f1,
            "mean_window_f1": self.mean_window_f1,
            "mean_scale_error": self.mean_scale_error,
            "scale_detection_rate": self.scale_detection_rate,
            "total_oversegmented": self.total_oversegmented,
            "total_undersegmented": self.total_undersegmented,
            "total_missed": self.total_missed,
            "total_spurious": self.total_spurious,
            "params_used": self.params_used,
        }

    @property
    def weighted_score(self) -> float:
        """Compute weighted score for parameter tuning."""
        # Higher is better
        # F1@0.5 + 0.5*label_acc - 0.2*overseg_rate
        overseg_rate = self.total_oversegmented / max(self.num_images, 1)
        return (
            self.mean_room_f1_50
            + 0.5 * self.mean_label_accuracy
            - 0.2 * overseg_rate
        )


class BenchmarkEvaluator:
    """
    Evaluates floor plan analysis pipeline on benchmark dataset.
    """

    def __init__(
        self,
        benchmark_dir: Path = Path("data/benchmark"),
        output_dir: Path = Path("out/bench"),
        params_path: Optional[Path] = None,
    ):
        """
        Initialize evaluator.

        Args:
            benchmark_dir: Path to benchmark directory
            output_dir: Output directory for results
            params_path: Path to segmentation_params.yaml (optional)
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.params_path = params_path

        self.annotations_dir = self.benchmark_dir / "annotations"
        self.raw_dir = self.benchmark_dir / "raw"
        self.processed_dir = self.benchmark_dir / "processed"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load parameters
        self.params = self._load_params()

    def _load_params(self) -> Dict[str, Any]:
        """Load segmentation parameters."""
        default_params = {
            "preprocess": {
                "adaptive_block_size": 35,
                "adaptive_c": 10,
                "enable_deskew": True,
                "enable_denoise": True,
                "denoise_strength": 10,
            },
            "walls": {
                "gap_close_size": 15,
                "min_wall_length": 30,
            },
            "regions": {
                "min_room_area_ratio": 0.001,
                "max_room_area_ratio": 0.5,
                "min_aspect_ratio": 0.1,
                "max_aspect_ratio": 10.0,
                "min_solidity": 0.3,
            },
            "polygons": {
                "simplification_epsilon": 2.0,
                "min_vertices": 4,
                "max_vertices": 100,
            },
        }

        if self.params_path and self.params_path.exists():
            import yaml
            with open(self.params_path) as f:
                loaded = yaml.safe_load(f)
                # Merge with defaults
                for key in loaded:
                    if key in default_params:
                        default_params[key].update(loaded[key])
                    else:
                        default_params[key] = loaded[key]

        return default_params

    def load_manifest(self) -> List[Dict[str, Any]]:
        """Load benchmark manifest."""
        manifest_path = self.benchmark_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f).get("images", [])
        return []

    def evaluate_single(
        self,
        image_path: Path,
        annotation: Optional[AnnotationSchema] = None,
    ) -> EvaluationResult:
        """
        Evaluate pipeline on a single image.

        Args:
            image_path: Path to image
            annotation: Ground truth annotation

        Returns:
            EvaluationResult
        """
        # Import dynamically to avoid package conflicts
        import subprocess
        import json as json_module

        # Run pipeline via subprocess to avoid import issues
        return self._run_pipeline_subprocess(image_path, annotation)

    def _run_pipeline_subprocess(
        self,
        image_path: Path,
        annotation: Optional[AnnotationSchema] = None,
    ) -> EvaluationResult:
        """Run pipeline in subprocess to avoid import issues."""
        import subprocess
        import json as json_module

        result = EvaluationResult(
            image_id=image_path.stem,
            image_path=str(image_path),
            success=False,
            run_time_sec=0.0,
        )

        start_time = time.time()

        try:
            # Run pipeline via standalone bench_runner script
            params_json = json.dumps({
                "adaptive_block_size": self.params["preprocess"]["adaptive_block_size"],
                "enable_deskew": self.params["preprocess"]["enable_deskew"],
                "enable_denoise": self.params["preprocess"]["enable_denoise"],
                "gap_close_size": self.params["walls"]["gap_close_size"],
                "min_room_area_ratio": self.params["regions"]["min_room_area_ratio"],
                "max_room_area_ratio": self.params["regions"]["max_room_area_ratio"],
            })

            cmd = [
                "python3", "scripts/bench_runner.py",
                str(image_path),
                str(self.output_dir / "runs" / image_path.stem),
                params_json,
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if proc.returncode != 0:
                result.errors.append(proc.stderr[:500] if proc.stderr else "Pipeline failed")
                result.run_time_sec = time.time() - start_time
                return result

            # Parse output
            output = json_module.loads(proc.stdout)

            if not output.get("success"):
                result.errors = output.get("errors", ["Pipeline failed"])
                result.run_time_sec = time.time() - start_time
                return result

            result.pred_rooms = output.get("rooms", [])
            result.pred_scale = output.get("scale")

            # Compute metrics if we have annotations
            if annotation:
                gt_rooms = [
                    {
                        "id": r.id,
                        "polygon": r.polygon,
                        "label": r.label,
                        "area": r.area_sqm or 0,
                    }
                    for r in annotation.rooms
                ]
                result.room_metrics = compute_room_metrics(result.pred_rooms, gt_rooms)

                if annotation.openings:
                    gt_openings = [
                        {"id": o.id, "bbox": o.bbox, "type": o.type}
                        for o in annotation.openings
                    ]
                    result.opening_metrics = compute_opening_metrics([], gt_openings)

                if annotation.scale:
                    result.scale_metrics = compute_scale_metrics(
                        result.pred_scale,
                        annotation.scale.to_dict(),
                    )

            result.success = True

        except subprocess.TimeoutExpired:
            result.errors.append("Pipeline timed out")
        except Exception as e:
            logger.error(f"Evaluation failed for {image_path}: {e}", exc_info=True)
            result.errors.append(str(e))

        result.run_time_sec = time.time() - start_time
        return result

    def evaluate_benchmark(
        self,
        max_images: Optional[int] = None,
    ) -> BenchmarkResults:
        """
        Evaluate pipeline on entire benchmark dataset.

        Args:
            max_images: Maximum number of images to evaluate

        Returns:
            BenchmarkResults
        """
        manifest = self.load_manifest()
        if not manifest:
            logger.warning("No images in manifest, searching for images...")
            # Fall back to finding images directly
            images = list(self.raw_dir.glob("*.png")) + list(self.raw_dir.glob("*.jpg"))
            manifest = [{"id": img.stem, "path": str(img)} for img in images]

        if max_images:
            manifest = manifest[:max_images]

        logger.info(f"Evaluating {len(manifest)} images")

        results = BenchmarkResults(
            num_images=len(manifest),
            params_used=self.params,
        )

        # Metrics accumulators
        room_f1_50s = []
        room_f1_75s = []
        label_accs = []
        area_errors = []
        opening_f1s = []
        door_f1s = []
        window_f1s = []
        scale_errors = []

        for i, entry in enumerate(manifest):
            image_id = entry.get("id", f"image_{i}")
            image_path = Path(entry.get("path", ""))

            if not image_path.exists():
                # Try finding in raw dir
                image_path = self.raw_dir / f"{image_id}.png"
                if not image_path.exists():
                    image_path = self.raw_dir / f"{image_id}.jpg"

            if not image_path.exists():
                logger.warning(f"Image not found: {image_id}")
                continue

            # Load annotation if exists
            annotation = None
            ann_path = self.annotations_dir / f"{image_id}.json"
            if ann_path.exists():
                annotation = load_annotations(ann_path)

            logger.info(f"[{i+1}/{len(manifest)}] Evaluating {image_id}")
            eval_result = self.evaluate_single(image_path, annotation)
            results.results.append(eval_result)

            if eval_result.success:
                results.num_success += 1
                results.total_time_sec += eval_result.run_time_sec

                if eval_result.room_metrics:
                    rm = eval_result.room_metrics
                    room_f1_50s.append(rm.f1_50)
                    room_f1_75s.append(rm.f1_75)
                    label_accs.append(rm.label_accuracy_with_alias)
                    area_errors.append(rm.mean_area_error_pct)
                    results.total_oversegmented += rm.oversegmented
                    results.total_undersegmented += rm.undersegmented
                    results.total_missed += rm.missed
                    results.total_spurious += rm.spurious

                if eval_result.opening_metrics:
                    om = eval_result.opening_metrics
                    opening_f1s.append(om.f1)
                    door_f1s.append(om.door_f1)
                    window_f1s.append(om.window_f1)

                if eval_result.scale_metrics:
                    sm = eval_result.scale_metrics
                    if sm.has_gt_scale:
                        scale_errors.append(sm.scale_error_pct)

        # Aggregate metrics
        results.mean_room_f1_50 = np.mean(room_f1_50s) if room_f1_50s else 0
        results.mean_room_f1_75 = np.mean(room_f1_75s) if room_f1_75s else 0
        results.mean_label_accuracy = np.mean(label_accs) if label_accs else 0
        results.mean_area_error = np.mean(area_errors) if area_errors else 0

        results.mean_opening_f1 = np.mean(opening_f1s) if opening_f1s else 0
        results.mean_door_f1 = np.mean(door_f1s) if door_f1s else 0
        results.mean_window_f1 = np.mean(window_f1s) if window_f1s else 0

        results.mean_scale_error = np.mean(scale_errors) if scale_errors else 0
        results.scale_detection_rate = len([s for s in scale_errors if s < 10]) / max(len(scale_errors), 1)

        return results

    def save_results(self, results: BenchmarkResults, filename: str = "results.json") -> Path:
        """Save benchmark results to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Saved results to {output_path}")

        # Also save CSV for easy analysis
        csv_path = self.output_dir / "results.csv"
        self._save_results_csv(results, csv_path)

        return output_path

    def _save_results_csv(self, results: BenchmarkResults, csv_path: Path) -> None:
        """Save per-image results to CSV for easy analysis."""
        import csv

        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "image_id",
                "success",
                "run_time_sec",
                "room_f1_50",
                "room_f1_75",
                "label_accuracy",
                "area_error_pct",
                "num_pred_rooms",
                "num_gt_rooms",
                "oversegmented",
                "undersegmented",
                "missed",
                "spurious",
                "errors",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results.results:
                row = {
                    "image_id": result.image_id,
                    "success": result.success,
                    "run_time_sec": f"{result.run_time_sec:.2f}",
                    "room_f1_50": "",
                    "room_f1_75": "",
                    "label_accuracy": "",
                    "area_error_pct": "",
                    "num_pred_rooms": len(result.pred_rooms),
                    "num_gt_rooms": "",
                    "oversegmented": "",
                    "undersegmented": "",
                    "missed": "",
                    "spurious": "",
                    "errors": "; ".join(result.errors) if result.errors else "",
                }

                if result.room_metrics:
                    rm = result.room_metrics
                    row.update({
                        "room_f1_50": f"{rm.f1_50:.3f}",
                        "room_f1_75": f"{rm.f1_75:.3f}",
                        "label_accuracy": f"{rm.label_accuracy_with_alias:.3f}",
                        "area_error_pct": f"{rm.mean_area_error_pct:.1f}",
                        "num_gt_rooms": rm.num_gt,
                        "oversegmented": rm.oversegmented,
                        "undersegmented": rm.undersegmented,
                        "missed": rm.missed,
                        "spurious": rm.spurious,
                    })

                writer.writerow(row)

        logger.info(f"Saved CSV results to {csv_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate floor plan pipeline")
    parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    parser.add_argument("--output-dir", default="out/bench", help="Output directory")
    parser.add_argument("--params", help="Path to segmentation_params.yaml")
    parser.add_argument("--max-images", type=int, help="Maximum images to evaluate")

    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir),
        params_path=Path(args.params) if args.params else None,
    )

    results = evaluator.evaluate_benchmark(max_images=args.max_images)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Images: {results.num_success}/{results.num_images} successful")
    print(f"Total time: {results.total_time_sec:.1f}s")
    print(f"\nRoom Segmentation:")
    print(f"  F1@0.5: {results.mean_room_f1_50:.2%}")
    print(f"  F1@0.75: {results.mean_room_f1_75:.2%}")
    print(f"  Label Accuracy: {results.mean_label_accuracy:.2%}")
    print(f"  Mean Area Error: {results.mean_area_error:.1f}%")
    print(f"\nSegmentation Issues:")
    print(f"  Over-segmented: {results.total_oversegmented}")
    print(f"  Under-segmented: {results.total_undersegmented}")
    print(f"  Missed: {results.total_missed}")
    print(f"  Spurious: {results.total_spurious}")
    print(f"\nWeighted Score: {results.weighted_score:.3f}")
    print("=" * 60)

    evaluator.save_results(results)
