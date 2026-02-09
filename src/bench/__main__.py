"""
CLI entry point for benchmarking commands.

Usage:
    python -m src.bench build_benchmark --n 30
    python -m src.bench build_public_set --n 40
    python -m src.bench run --params rules/segmentation_params.yaml
    python -m src.bench sweep --trials 50
    python -m src.bench failures --top 20
    python -m src.bench annotate --image <path>

Output:
    out/bench/results.json  - Full benchmark results
    out/bench/results.csv   - Per-image results CSV
    out/bench/report.md     - Human-readable report
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def cmd_build_benchmark(args):
    """Build benchmark dataset."""
    from .build import BenchmarkBuilder

    builder = BenchmarkBuilder(
        benchmark_dir=Path(args.benchmark_dir),
        use_existing=args.use_existing,
    )

    manifest = builder.build(
        n_images=args.n,
        download_if_needed=args.download,
    )

    print(f"\nBuilt benchmark with {len(manifest)} images")
    print(f"Location: {args.benchmark_dir}")


def cmd_build_public_set(args):
    """Download public floor plan images for benchmark."""
    from .build import BenchmarkBuilder

    builder = BenchmarkBuilder(
        benchmark_dir=Path(args.benchmark_dir),
        use_existing=False,
    )

    # Ensure directories exist
    builder.raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.n} public floor plan images...")
    print("Sources: Wikimedia Commons (CC licensed)")

    manifest = builder._download_wikimedia_floorplans(args.n)

    # Also generate synthetic if needed
    if len(manifest) < args.n and not args.no_synthetic:
        needed = args.n - len(manifest)
        print(f"\nGenerating {needed} synthetic plans to fill gap...")
        synthetic = builder._generate_synthetic_plans(needed)
        manifest.extend(synthetic)

    # Save manifest
    builder._save_manifest(manifest)

    print(f"\nBuilt public dataset with {len(manifest)} images")
    print(f"  - Downloaded: {len([m for m in manifest if m.get('source') == 'wikimedia'])}")
    print(f"  - Synthetic: {len([m for m in manifest if m.get('source') == 'synthetic'])}")
    print(f"Location: {args.benchmark_dir}/raw/")


def cmd_run(args):
    """Run benchmark evaluation."""
    from .evaluate import BenchmarkEvaluator

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

    # Generate report
    if not args.no_report:
        from .report import ReportGenerator
        generator = ReportGenerator(Path(args.output_dir))
        generator.generate_report(results)


def cmd_sweep(args):
    """Run parameter sweep."""
    from .sweep import ParameterSweeper

    sweeper = ParameterSweeper(
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir),
    )

    if args.method == "grid" and args.grid_params:
        best = sweeper.grid_search(
            params_to_sweep=args.grid_params,
            max_images=args.max_images,
        )
    else:
        best = sweeper.random_search(
            n_trials=args.trials,
            max_images=args.max_images,
        )

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Best weighted score: {best.weighted_score:.3f}")
    print(f"Best F1@0.5: {best.room_f1_50:.2%}")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    sweeper.save_best_params()
    sweeper.save_sweep_results()


def cmd_failures(args):
    """Mine failure cases."""
    import json
    from .failures import FailureMiner
    from .evaluate import BenchmarkResults, EvaluationResult

    # Load results
    results_path = Path(args.output_dir) / "results.json"
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Run 'python -m src.bench run' first")
        sys.exit(1)

    with open(results_path) as f:
        results_dict = json.load(f)

    # Reconstruct results (simplified)
    results = BenchmarkResults(
        num_images=results_dict.get("num_images", 0),
        num_success=results_dict.get("num_success", 0),
    )

    miner = FailureMiner(
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir) / "failures",
    )

    failures = miner.mine_failures(results, top_n=args.top)
    miner.generate_failure_outputs(failures, results)
    miner.save_failures(failures)

    print(f"\nMined {len(failures)} failure cases")
    print(f"Output: {args.output_dir}/failures/")


def cmd_annotate(args):
    """Interactive annotation helper."""
    from .annotate import AnnotationHelper

    helper = AnnotationHelper(
        benchmark_dir=Path(args.benchmark_dir),
    )

    if args.image:
        helper.annotate_image(Path(args.image))
    elif args.from_csv:
        helper.import_from_csv(Path(args.from_csv))
    else:
        helper.interactive_mode()


def main():
    parser = argparse.ArgumentParser(
        description="Floor Plan Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build_benchmark   Build benchmark dataset from existing or downloaded plans
  run               Run pipeline evaluation on benchmark
  sweep             Parameter sweep to find optimal config
  failures          Mine and analyze failure cases
  annotate          Interactive annotation helper

Examples:
  python -m src.bench build_benchmark --n 30
  python -m src.bench run --params rules/segmentation_params.yaml
  python -m src.bench sweep --trials 50
  python -m src.bench failures --top 20
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build_benchmark
    build_parser = subparsers.add_parser("build_benchmark", help="Build benchmark dataset")
    build_parser.add_argument("--n", type=int, default=30, help="Number of images")
    build_parser.add_argument("--benchmark-dir", default="data/benchmark", help="Output directory")
    build_parser.add_argument("--use-existing", action="store_true", help="Use existing plans in data/plans")
    build_parser.add_argument("--download", action="store_true", help="Download public dataset if needed")

    # build_public_set - dedicated command for downloading public data
    public_parser = subparsers.add_parser("build_public_set", help="Download public floor plan images")
    public_parser.add_argument("--n", type=int, default=40, help="Number of images to download")
    public_parser.add_argument("--benchmark-dir", default="data/benchmark", help="Output directory")
    public_parser.add_argument("--no-synthetic", action="store_true", help="Don't generate synthetic plans")

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation")
    run_parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    run_parser.add_argument("--output-dir", default="out/bench", help="Output directory")
    run_parser.add_argument("--params", help="Path to segmentation_params.yaml")
    run_parser.add_argument("--max-images", type=int, help="Maximum images to evaluate")
    run_parser.add_argument("--no-report", action="store_true", help="Skip report generation")

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="Parameter sweep")
    sweep_parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    sweep_parser.add_argument("--output-dir", default="out/bench", help="Output directory")
    sweep_parser.add_argument("--trials", type=int, default=50, help="Number of random trials")
    sweep_parser.add_argument("--max-images", type=int, help="Max images per trial")
    sweep_parser.add_argument("--method", choices=["random", "grid"], default="random", help="Search method")
    sweep_parser.add_argument("--grid-params", nargs="+", help="Params for grid search")

    # failures
    failures_parser = subparsers.add_parser("failures", help="Mine failure cases")
    failures_parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    failures_parser.add_argument("--output-dir", default="out/bench", help="Output directory")
    failures_parser.add_argument("--top", type=int, default=20, help="Top N failures per category")

    # annotate
    annotate_parser = subparsers.add_parser("annotate", help="Annotation helper")
    annotate_parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    annotate_parser.add_argument("--image", help="Path to image to annotate")
    annotate_parser.add_argument("--from-csv", help="Import annotations from CSV")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    if args.command == "build_benchmark":
        cmd_build_benchmark(args)
    elif args.command == "build_public_set":
        cmd_build_public_set(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "failures":
        cmd_failures(args)
    elif args.command == "annotate":
        cmd_annotate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
