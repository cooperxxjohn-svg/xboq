"""
Failure Mining - Automatically identify and analyze worst cases.

Identifies:
- Top N worst pages by room F1
- Top N worst by area error
- Over-segmentation cases (one room split into many)
- Under-segmentation cases (multiple rooms merged)
- Missing/low-confidence scale cases

Outputs:
- Overlay visualizations
- Diff images
- Analysis notes with suspected causes
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from .annotation import load_annotations, AnnotationSchema
from .evaluate import EvaluationResult, BenchmarkResults

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    """A single failure case with analysis."""
    image_id: str
    image_path: str
    failure_type: str  # low_f1, area_error, overseg, underseg, scale
    severity: float  # 0-1, higher = worse

    # Metrics
    room_f1: float = 0.0
    area_error: float = 0.0

    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    suspected_causes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "room_f1": self.room_f1,
            "area_error": self.area_error,
            "details": self.details,
            "suspected_causes": self.suspected_causes,
            "recommendations": self.recommendations,
        }


class FailureMiner:
    """
    Mines failure cases from benchmark results.
    """

    def __init__(
        self,
        benchmark_dir: Path = Path("data/benchmark"),
        output_dir: Path = Path("out/bench/failures"),
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotations_dir = self.benchmark_dir / "annotations"
        self.raw_dir = self.benchmark_dir / "raw"

    def mine_failures(
        self,
        results: BenchmarkResults,
        top_n: int = 20,
    ) -> List[FailureCase]:
        """
        Extract top failure cases from benchmark results.

        Args:
            results: Benchmark evaluation results
            top_n: Number of top failures to extract per category

        Returns:
            List of FailureCase objects
        """
        failures = []

        # Sort by room F1 (worst first)
        results_with_metrics = [r for r in results.results if r.room_metrics]
        sorted_by_f1 = sorted(results_with_metrics, key=lambda r: r.room_metrics.f1_50)

        # Top worst by F1
        for r in sorted_by_f1[:top_n]:
            failure = self._analyze_low_f1(r)
            failures.append(failure)

        # Sort by area error (worst first)
        sorted_by_area = sorted(
            results_with_metrics,
            key=lambda r: r.room_metrics.mean_area_error_pct,
            reverse=True
        )

        # Top worst by area error
        for r in sorted_by_area[:top_n]:
            failure = self._analyze_area_error(r)
            if failure not in failures:
                failures.append(failure)

        # Over-segmentation cases
        overseg_cases = [r for r in results_with_metrics if r.room_metrics.oversegmented > 0]
        overseg_cases.sort(key=lambda r: r.room_metrics.oversegmented, reverse=True)

        for r in overseg_cases[:top_n]:
            failure = self._analyze_oversegmentation(r)
            if failure not in failures:
                failures.append(failure)

        # Under-segmentation cases
        underseg_cases = [r for r in results_with_metrics if r.room_metrics.undersegmented > 0]
        underseg_cases.sort(key=lambda r: r.room_metrics.undersegmented, reverse=True)

        for r in underseg_cases[:top_n]:
            failure = self._analyze_undersegmentation(r)
            if failure not in failures:
                failures.append(failure)

        # Scale issues
        scale_issues = [r for r in results.results if r.scale_metrics and r.scale_metrics.scale_error_pct > 20]
        scale_issues.sort(key=lambda r: r.scale_metrics.scale_error_pct, reverse=True)

        for r in scale_issues[:top_n]:
            failure = self._analyze_scale_error(r)
            if failure not in failures:
                failures.append(failure)

        logger.info(f"Mined {len(failures)} failure cases")
        return failures

    def _analyze_low_f1(self, result: EvaluationResult) -> FailureCase:
        """Analyze low F1 failure case."""
        rm = result.room_metrics

        causes = []
        recommendations = []

        # Determine causes
        if rm.missed > rm.num_gt * 0.3:
            causes.append("Many rooms missed - may need lower gap_close_size")
            recommendations.append("Reduce gap_close_size to preserve room boundaries")

        if rm.spurious > rm.num_pred * 0.3:
            causes.append("Many spurious detections - likely noise or furniture")
            recommendations.append("Increase min_room_area_ratio threshold")

        if rm.label_accuracy < 0.5:
            causes.append("Poor label matching - OCR or text placement issues")
            recommendations.append("Check text extraction and room alias rules")

        if not causes:
            causes.append("Complex floor plan geometry")
            recommendations.append("Manual review needed")

        return FailureCase(
            image_id=result.image_id,
            image_path=result.image_path,
            failure_type="low_f1",
            severity=1 - rm.f1_50,
            room_f1=rm.f1_50,
            area_error=rm.mean_area_error_pct,
            details={
                "matched": rm.num_matched_50,
                "missed": rm.missed,
                "spurious": rm.spurious,
                "num_pred": rm.num_pred,
                "num_gt": rm.num_gt,
            },
            suspected_causes=causes,
            recommendations=recommendations,
        )

    def _analyze_area_error(self, result: EvaluationResult) -> FailureCase:
        """Analyze high area error failure case."""
        rm = result.room_metrics

        causes = []
        recommendations = []

        if rm.mean_area_error_pct > 30:
            causes.append("High area errors - likely scale detection issue")
            recommendations.append("Check scale inference, verify DPI setting")

        if rm.max_area_error_pct > 50:
            causes.append("Some rooms have very high area errors")
            recommendations.append("Check for partial room detection or merged rooms")

        return FailureCase(
            image_id=result.image_id,
            image_path=result.image_path,
            failure_type="area_error",
            severity=min(rm.mean_area_error_pct / 100, 1.0),
            room_f1=rm.f1_50,
            area_error=rm.mean_area_error_pct,
            details={
                "mean_error": rm.mean_area_error_pct,
                "max_error": rm.max_area_error_pct,
            },
            suspected_causes=causes,
            recommendations=recommendations,
        )

    def _analyze_oversegmentation(self, result: EvaluationResult) -> FailureCase:
        """Analyze over-segmentation failure case."""
        rm = result.room_metrics

        causes = [
            "Room split into multiple regions",
            "Likely caused by internal lines (furniture, dimensions)",
        ]
        recommendations = [
            "Increase gap_close_size",
            "Add morphological closing passes",
            "Filter thin line elements",
        ]

        return FailureCase(
            image_id=result.image_id,
            image_path=result.image_path,
            failure_type="oversegmentation",
            severity=min(rm.oversegmented / max(rm.num_gt, 1), 1.0),
            room_f1=rm.f1_50,
            area_error=rm.mean_area_error_pct,
            details={
                "oversegmented_count": rm.oversegmented,
                "num_gt": rm.num_gt,
                "num_pred": rm.num_pred,
            },
            suspected_causes=causes,
            recommendations=recommendations,
        )

    def _analyze_undersegmentation(self, result: EvaluationResult) -> FailureCase:
        """Analyze under-segmentation failure case."""
        rm = result.room_metrics

        causes = [
            "Multiple rooms merged into one",
            "Walls not detected - may be too thin or low contrast",
        ]
        recommendations = [
            "Decrease gap_close_size",
            "Check wall detection thresholds",
            "Improve binarization for thin walls",
        ]

        return FailureCase(
            image_id=result.image_id,
            image_path=result.image_path,
            failure_type="undersegmentation",
            severity=min(rm.undersegmented / max(rm.num_pred, 1), 1.0),
            room_f1=rm.f1_50,
            area_error=rm.mean_area_error_pct,
            details={
                "undersegmented_count": rm.undersegmented,
                "num_gt": rm.num_gt,
                "num_pred": rm.num_pred,
            },
            suspected_causes=causes,
            recommendations=recommendations,
        )

    def _analyze_scale_error(self, result: EvaluationResult) -> FailureCase:
        """Analyze scale detection failure case."""
        sm = result.scale_metrics

        causes = []
        recommendations = []

        if not sm.has_pred_scale:
            causes.append("Scale not detected at all")
            recommendations.append("Check for scale bar in image")
            recommendations.append("Verify text extraction working")
        else:
            causes.append(f"Scale error: {sm.scale_error_pct:.1f}%")
            if sm.confidence < 0.5:
                causes.append("Low confidence scale detection")
            recommendations.append("Verify scale inference algorithm")
            recommendations.append("Check for multiple scale indicators")

        return FailureCase(
            image_id=result.image_id,
            image_path=result.image_path,
            failure_type="scale",
            severity=min(sm.scale_error_pct / 100, 1.0),
            room_f1=result.room_metrics.f1_50 if result.room_metrics else 0,
            area_error=result.room_metrics.mean_area_error_pct if result.room_metrics else 0,
            details={
                "scale_error_pct": sm.scale_error_pct,
                "gt_px_per_mm": sm.gt_px_per_mm,
                "pred_px_per_mm": sm.pred_px_per_mm,
                "confidence": sm.confidence,
            },
            suspected_causes=causes,
            recommendations=recommendations,
        )

    def generate_failure_outputs(
        self,
        failures: List[FailureCase],
        results: BenchmarkResults,
    ) -> None:
        """
        Generate output artifacts for each failure case.

        Creates:
        - overlay_pred.png - predictions overlaid on image
        - overlay_gt.png - ground truth overlaid (if exists)
        - diff.png - difference visualization
        - notes.md - analysis notes
        """
        for failure in failures:
            failure_dir = self.output_dir / failure.image_id
            failure_dir.mkdir(parents=True, exist_ok=True)

            # Find the evaluation result
            eval_result = next(
                (r for r in results.results if r.image_id == failure.image_id),
                None
            )

            if not eval_result:
                continue

            # Load image
            image_path = Path(failure.image_path)
            if not image_path.exists():
                image_path = self.raw_dir / f"{failure.image_id}.png"
            if not image_path.exists():
                image_path = self.raw_dir / f"{failure.image_id}.jpg"

            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    # Generate prediction overlay
                    pred_overlay = self._draw_predictions(image.copy(), eval_result.pred_rooms)
                    cv2.imwrite(str(failure_dir / "overlay_pred.png"), pred_overlay)

                    # Generate GT overlay if annotation exists
                    ann_path = self.annotations_dir / f"{failure.image_id}.json"
                    if ann_path.exists():
                        annotation = load_annotations(ann_path)
                        gt_overlay = self._draw_ground_truth(image.copy(), annotation)
                        cv2.imwrite(str(failure_dir / "overlay_gt.png"), gt_overlay)

                        # Generate diff
                        diff = self._generate_diff(pred_overlay, gt_overlay)
                        cv2.imwrite(str(failure_dir / "diff.png"), diff)

            # Generate notes markdown
            self._write_notes(failure_dir / "notes.md", failure, eval_result)

        logger.info(f"Generated outputs for {len(failures)} failure cases")

    def _draw_predictions(
        self,
        image: np.ndarray,
        pred_rooms: List[Dict],
    ) -> np.ndarray:
        """Draw predicted rooms on image."""
        overlay = image.copy()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (0, 128, 128), (128, 128, 0),
        ]

        for i, room in enumerate(pred_rooms):
            color = colors[i % len(colors)]
            polygon = room.get("polygon", [])

            if polygon:
                pts = np.array(polygon, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, color, 2)

                # Draw centroid with label
                cx = int(np.mean([p[0] for p in polygon]))
                cy = int(np.mean([p[1] for p in polygon]))
                label = room.get("label", f"R{i}")

                cv2.putText(
                    overlay, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        return overlay

    def _draw_ground_truth(
        self,
        image: np.ndarray,
        annotation: AnnotationSchema,
    ) -> np.ndarray:
        """Draw ground truth rooms on image."""
        overlay = image.copy()

        colors = [
            (0, 128, 0), (0, 0, 128), (128, 0, 0),
            (0, 128, 128), (128, 0, 128), (128, 128, 0),
        ]

        for i, room in enumerate(annotation.rooms):
            color = colors[i % len(colors)]

            if room.polygon:
                pts = np.array(room.polygon, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, color, 3)

                cx, cy = room.centroid
                cv2.putText(
                    overlay, room.label, (int(cx) - 20, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        return overlay

    def _generate_diff(
        self,
        pred_overlay: np.ndarray,
        gt_overlay: np.ndarray,
    ) -> np.ndarray:
        """Generate diff visualization between pred and GT."""
        # Simple approach: blend with different colors
        diff = cv2.addWeighted(pred_overlay, 0.5, gt_overlay, 0.5, 0)
        return diff

    def _write_notes(
        self,
        path: Path,
        failure: FailureCase,
        result: EvaluationResult,
    ) -> None:
        """Write analysis notes markdown."""
        with open(path, "w") as f:
            f.write(f"# Failure Analysis: {failure.image_id}\n\n")

            f.write(f"**Type:** {failure.failure_type}\n")
            f.write(f"**Severity:** {failure.severity:.2f}\n\n")

            f.write("## Metrics\n\n")
            f.write(f"- Room F1@0.5: {failure.room_f1:.2%}\n")
            f.write(f"- Area Error: {failure.area_error:.1f}%\n\n")

            if failure.details:
                f.write("## Details\n\n")
                for key, val in failure.details.items():
                    f.write(f"- {key}: {val}\n")
                f.write("\n")

            f.write("## Suspected Causes\n\n")
            for cause in failure.suspected_causes:
                f.write(f"- {cause}\n")
            f.write("\n")

            f.write("## Recommendations\n\n")
            for rec in failure.recommendations:
                f.write(f"- {rec}\n")
            f.write("\n")

            if result.errors:
                f.write("## Errors\n\n")
                for err in result.errors:
                    f.write(f"- {err}\n")

    def save_failures(self, failures: List[FailureCase], filename: str = "failures.json") -> Path:
        """Save failures to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump([f.to_dict() for f in failures], f, indent=2)
        logger.info(f"Saved {len(failures)} failures to {output_path}")
        return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Mine failure cases from benchmark results")
    parser.add_argument("--results", required=True, help="Path to benchmark results JSON")
    parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    parser.add_argument("--output-dir", default="out/bench/failures", help="Output directory")
    parser.add_argument("--top", type=int, default=20, help="Top N failures per category")

    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        results_dict = json.load(f)

    # Convert back to BenchmarkResults (simplified)
    results = BenchmarkResults(
        num_images=results_dict.get("num_images", 0),
        num_success=results_dict.get("num_success", 0),
    )

    # Mine failures
    miner = FailureMiner(
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir),
    )

    failures = miner.mine_failures(results, top_n=args.top)

    print(f"\nMined {len(failures)} failure cases:")
    for failure in failures[:10]:
        print(f"  {failure.image_id}: {failure.failure_type} (severity: {failure.severity:.2f})")

    miner.save_failures(failures)
