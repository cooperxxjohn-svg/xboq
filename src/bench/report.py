"""
Report Generator - Creates comprehensive benchmark reports.

Generates:
- Markdown report with metrics, failure patterns, recommendations
- Summary statistics
- Parameter comparison
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .evaluate import BenchmarkResults, EvaluationResult
from .failures import FailureCase
from .sweep import SweepResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive benchmark reports.
    """

    def __init__(
        self,
        output_dir: Path = Path("out/bench"),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: BenchmarkResults,
        failures: Optional[List[FailureCase]] = None,
        sweep_results: Optional[List[SweepResult]] = None,
        baseline_results: Optional[BenchmarkResults] = None,
    ) -> Path:
        """
        Generate comprehensive markdown report.

        Args:
            results: Current benchmark results
            failures: Failure cases
            sweep_results: Parameter sweep results
            baseline_results: Baseline for comparison

        Returns:
            Path to generated report
        """
        report_path = self.output_dir / "report.md"

        with open(report_path, "w") as f:
            self._write_header(f, results)
            self._write_dataset_summary(f, results)
            self._write_metrics(f, results)

            if baseline_results:
                self._write_comparison(f, results, baseline_results)

            if failures:
                self._write_failure_patterns(f, failures)

            if sweep_results:
                self._write_sweep_summary(f, sweep_results)

            self._write_params_used(f, results)
            self._write_recommendations(f, results, failures)

        logger.info(f"Generated report: {report_path}")
        return report_path

    def _write_header(self, f, results: BenchmarkResults) -> None:
        """Write report header."""
        f.write("# Floor Plan Analysis Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    def _write_dataset_summary(self, f, results: BenchmarkResults) -> None:
        """Write dataset summary section."""
        f.write("## Dataset Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Images | {results.num_images} |\n")
        f.write(f"| Successful | {results.num_success} |\n")
        f.write(f"| Success Rate | {results.num_success/max(results.num_images, 1):.1%} |\n")
        f.write(f"| Total Time | {results.total_time_sec:.1f}s |\n")
        f.write(f"| Avg Time/Image | {results.total_time_sec/max(results.num_success, 1):.2f}s |\n")
        f.write("\n")

    def _write_metrics(self, f, results: BenchmarkResults) -> None:
        """Write metrics section."""
        f.write("## Room Segmentation Metrics\n\n")

        # Main metrics table
        f.write("| Metric | Value | Target |\n")
        f.write("|--------|-------|--------|\n")
        f.write(f"| **F1@0.5** | **{results.mean_room_f1_50:.2%}** | >80% |\n")
        f.write(f"| F1@0.75 | {results.mean_room_f1_75:.2%} | >60% |\n")
        f.write(f"| Label Accuracy | {results.mean_label_accuracy:.2%} | >85% |\n")
        f.write(f"| Mean Area Error | {results.mean_area_error:.1f}% | <15% |\n")
        f.write("\n")

        # Segmentation issues
        f.write("### Segmentation Issues\n\n")
        f.write("| Issue | Count |\n")
        f.write("|-------|-------|\n")
        f.write(f"| Over-segmented | {results.total_oversegmented} |\n")
        f.write(f"| Under-segmented | {results.total_undersegmented} |\n")
        f.write(f"| Missed rooms | {results.total_missed} |\n")
        f.write(f"| Spurious detections | {results.total_spurious} |\n")
        f.write("\n")

        # Scale metrics
        f.write("### Scale Detection\n\n")
        f.write(f"- Detection Rate: {results.scale_detection_rate:.1%}\n")
        f.write(f"- Mean Error: {results.mean_scale_error:.1f}%\n")
        f.write("\n")

        # Opening metrics if available
        if results.mean_opening_f1 > 0:
            f.write("### Opening Detection\n\n")
            f.write("| Type | F1 |\n")
            f.write("|------|----|\n")
            f.write(f"| Overall | {results.mean_opening_f1:.2%} |\n")
            f.write(f"| Doors | {results.mean_door_f1:.2%} |\n")
            f.write(f"| Windows | {results.mean_window_f1:.2%} |\n")
            f.write("\n")

        # Weighted score
        f.write(f"### Weighted Score: **{results.weighted_score:.3f}**\n\n")
        f.write("*(score = F1@0.5 + 0.5×label_acc - 0.2×overseg_rate)*\n\n")

    def _write_comparison(self, f, current: BenchmarkResults, baseline: BenchmarkResults) -> None:
        """Write comparison with baseline."""
        f.write("## Comparison with Baseline\n\n")

        def delta_str(curr, base, higher_is_better=True):
            diff = curr - base
            sign = "+" if diff > 0 else ""
            indicator = "↑" if (diff > 0) == higher_is_better else "↓"
            color = "green" if (diff > 0) == higher_is_better else "red"
            return f"{sign}{diff:.2%} {indicator}"

        f.write("| Metric | Baseline | Current | Change |\n")
        f.write("|--------|----------|---------|--------|\n")
        f.write(f"| F1@0.5 | {baseline.mean_room_f1_50:.2%} | {current.mean_room_f1_50:.2%} | {delta_str(current.mean_room_f1_50, baseline.mean_room_f1_50)} |\n")
        f.write(f"| F1@0.75 | {baseline.mean_room_f1_75:.2%} | {current.mean_room_f1_75:.2%} | {delta_str(current.mean_room_f1_75, baseline.mean_room_f1_75)} |\n")
        f.write(f"| Label Acc | {baseline.mean_label_accuracy:.2%} | {current.mean_label_accuracy:.2%} | {delta_str(current.mean_label_accuracy, baseline.mean_label_accuracy)} |\n")
        f.write(f"| Area Error | {baseline.mean_area_error:.1f}% | {current.mean_area_error:.1f}% | {delta_str(current.mean_area_error, baseline.mean_area_error, False)} |\n")
        f.write(f"| Weighted | {baseline.weighted_score:.3f} | {current.weighted_score:.3f} | {delta_str(current.weighted_score, baseline.weighted_score)} |\n")
        f.write("\n")

    def _write_failure_patterns(self, f, failures: List[FailureCase]) -> None:
        """Write failure patterns section."""
        f.write("## Failure Analysis\n\n")

        # Count by type
        type_counts = {}
        for failure in failures:
            type_counts[failure.failure_type] = type_counts.get(failure.failure_type, 0) + 1

        f.write("### Failure Types\n\n")
        f.write("| Type | Count |\n")
        f.write("|------|-------|\n")
        for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {ftype} | {count} |\n")
        f.write("\n")

        # Common causes
        f.write("### Common Suspected Causes\n\n")
        cause_counts = {}
        for failure in failures:
            for cause in failure.suspected_causes:
                cause_counts[cause] = cause_counts.get(cause, 0) + 1

        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1])[:10]:
            f.write(f"- **{cause}** ({count} occurrences)\n")
        f.write("\n")

        # Top failures
        f.write("### Top 10 Worst Cases\n\n")
        f.write("| Image | Type | F1 | Severity |\n")
        f.write("|-------|------|-----|----------|\n")
        for failure in sorted(failures, key=lambda x: -x.severity)[:10]:
            f.write(f"| {failure.image_id} | {failure.failure_type} | {failure.room_f1:.2%} | {failure.severity:.2f} |\n")
        f.write("\n")

    def _write_sweep_summary(self, f, sweep_results: List[SweepResult]) -> None:
        """Write parameter sweep summary."""
        f.write("## Parameter Sweep Results\n\n")

        successful = [r for r in sweep_results if r.success]
        f.write(f"- Trials: {len(sweep_results)}\n")
        f.write(f"- Successful: {len(successful)}\n")

        if successful:
            best = max(successful, key=lambda x: x.weighted_score)
            worst = min(successful, key=lambda x: x.weighted_score)

            f.write(f"- Best score: {best.weighted_score:.3f}\n")
            f.write(f"- Worst score: {worst.weighted_score:.3f}\n")
            f.write(f"- Score range: {best.weighted_score - worst.weighted_score:.3f}\n\n")

            f.write("### Best Parameters Found\n\n")
            f.write("```yaml\n")
            for k, v in best.params.items():
                f.write(f"{k}: {v}\n")
            f.write("```\n\n")

    def _write_params_used(self, f, results: BenchmarkResults) -> None:
        """Write parameters used section."""
        f.write("## Parameters Used\n\n")
        f.write("```yaml\n")

        if results.params_used:
            for category, params in results.params_used.items():
                f.write(f"{category}:\n")
                if isinstance(params, dict):
                    for k, v in params.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {params}\n")

        f.write("```\n\n")

    def _write_recommendations(
        self,
        f,
        results: BenchmarkResults,
        failures: Optional[List[FailureCase]] = None
    ) -> None:
        """Write recommendations section."""
        f.write("## Recommendations\n\n")

        recommendations = []

        # Based on metrics
        if results.mean_room_f1_50 < 0.7:
            recommendations.append("**Critical:** Room F1 below 70% - review wall detection and gap closing parameters")

        if results.total_oversegmented > results.num_images * 0.2:
            recommendations.append("**High over-segmentation:** Increase `gap_close_size` to merge fragmented rooms")

        if results.total_undersegmented > results.num_images * 0.2:
            recommendations.append("**High under-segmentation:** Decrease `gap_close_size` or improve wall detection")

        if results.mean_label_accuracy < 0.7:
            recommendations.append("**Low label accuracy:** Review text extraction and room alias rules")

        if results.mean_area_error > 20:
            recommendations.append("**High area errors:** Check scale detection - verify DPI setting")

        if results.scale_detection_rate < 0.5:
            recommendations.append("**Low scale detection:** Improve scale inference or provide manual scale")

        # Based on failures if available
        if failures:
            cause_counts = {}
            for failure in failures:
                for cause in failure.suspected_causes:
                    cause_counts[cause] = cause_counts.get(cause, 0) + 1

            if cause_counts:
                top_cause = max(cause_counts.items(), key=lambda x: x[1])
                recommendations.append(f"**Most common failure cause:** {top_cause[0]}")

        if not recommendations:
            recommendations.append("✓ All metrics within acceptable ranges")

        f.write("### Next Steps\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        f.write("\n")

        # Improvement workflow
        f.write("### Improvement Workflow\n\n")
        f.write("1. Review failure cases in `out/bench/failures/`\n")
        f.write("2. Run parameter sweep: `python -m src.bench.sweep --trials 50`\n")
        f.write("3. Apply best params from `rules/segmentation_params.yaml`\n")
        f.write("4. Add more annotated examples to `data/benchmark/annotations/`\n")
        f.write("5. Re-run benchmark: `python -m src.bench.run`\n")
        f.write("\n")


def generate_summary_report(
    results_path: Path,
    failures_path: Optional[Path] = None,
    sweep_path: Optional[Path] = None,
    output_dir: Path = Path("out/bench"),
) -> Path:
    """
    Generate report from saved results files.

    Args:
        results_path: Path to results JSON
        failures_path: Path to failures JSON
        sweep_path: Path to sweep results CSV
        output_dir: Output directory

    Returns:
        Path to generated report
    """
    # Load results
    with open(results_path) as f:
        results_dict = json.load(f)

    results = BenchmarkResults(**{
        k: v for k, v in results_dict.items()
        if k not in ["results"]
    })

    # Load failures if provided
    failures = None
    if failures_path and failures_path.exists():
        with open(failures_path) as f:
            failures_data = json.load(f)
            failures = [FailureCase(**d) for d in failures_data]

    # Load sweep results if provided
    sweep_results = None
    if sweep_path and sweep_path.exists():
        import csv
        sweep_results = []
        with open(sweep_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sweep_results.append(SweepResult(
                    trial_id=int(row["trial_id"]),
                    params={},  # Would need to reconstruct
                    weighted_score=float(row["weighted_score"]),
                    room_f1_50=float(row["room_f1_50"]),
                    room_f1_75=float(row["room_f1_75"]),
                    label_accuracy=float(row["label_accuracy"]),
                    oversegmented=int(row["oversegmented"]),
                    success=row["success"] == "True",
                    run_time_sec=float(row["run_time_sec"]),
                ))

    generator = ReportGenerator(output_dir)
    return generator.generate_report(results, failures, sweep_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--failures", help="Path to failures JSON")
    parser.add_argument("--sweep", help="Path to sweep CSV")
    parser.add_argument("--output-dir", default="out/bench", help="Output directory")

    args = parser.parse_args()

    report_path = generate_summary_report(
        results_path=Path(args.results),
        failures_path=Path(args.failures) if args.failures else None,
        sweep_path=Path(args.sweep) if args.sweep else None,
        output_dir=Path(args.output_dir),
    )

    print(f"Generated report: {report_path}")
