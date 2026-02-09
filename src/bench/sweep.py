"""
Parameter Sweep - Automated parameter tuning for segmentation.

Supports:
- Grid search over key parameters
- Random search with configurable trials
- Evaluation using weighted score

Parameters:
- Binarization: adaptive_block_size, adaptive_c
- Line dilation: kernel size
- Gap closing: iterations, size
- Min room area threshold
- Hole fill threshold
- Contour simplification epsilon
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import yaml

from .evaluate import BenchmarkEvaluator, BenchmarkResults

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for a single parameter."""
    name: str
    category: str  # preprocess, walls, regions, polygons
    min_val: float
    max_val: float
    step: Optional[float] = None  # For grid search
    type: str = "int"  # int, float

    def sample(self) -> Any:
        """Sample a random value."""
        if self.type == "int":
            return random.randint(int(self.min_val), int(self.max_val))
        else:
            return random.uniform(self.min_val, self.max_val)

    def grid_values(self) -> List[Any]:
        """Get grid of values."""
        if self.step:
            vals = np.arange(self.min_val, self.max_val + self.step, self.step)
        else:
            vals = np.linspace(self.min_val, self.max_val, 5)

        if self.type == "int":
            return [int(v) for v in vals]
        return list(vals)


@dataclass
class SweepResult:
    """Result of a single parameter configuration trial."""
    trial_id: int
    params: Dict[str, Any]
    weighted_score: float
    room_f1_50: float
    room_f1_75: float
    label_accuracy: float
    oversegmented: int
    success: bool
    run_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "weighted_score": self.weighted_score,
            "room_f1_50": self.room_f1_50,
            "room_f1_75": self.room_f1_75,
            "label_accuracy": self.label_accuracy,
            "oversegmented": self.oversegmented,
            "success": self.success,
            "run_time_sec": self.run_time_sec,
        }


# Default parameter registry
DEFAULT_PARAMETER_REGISTRY = [
    # Preprocessing
    ParameterConfig("adaptive_block_size", "preprocess", 15, 55, step=10, type="int"),
    ParameterConfig("adaptive_c", "preprocess", 5, 20, step=5, type="int"),
    ParameterConfig("denoise_strength", "preprocess", 5, 20, step=5, type="int"),

    # Walls
    ParameterConfig("gap_close_size", "walls", 5, 30, step=5, type="int"),
    ParameterConfig("min_wall_length", "walls", 15, 50, step=10, type="int"),

    # Regions
    ParameterConfig("min_room_area_ratio", "regions", 0.0005, 0.005, type="float"),
    ParameterConfig("max_room_area_ratio", "regions", 0.3, 0.7, type="float"),
    ParameterConfig("min_solidity", "regions", 0.2, 0.5, type="float"),

    # Polygons
    ParameterConfig("simplification_epsilon", "polygons", 1.0, 5.0, type="float"),
]


class ParameterSweeper:
    """
    Sweeps over parameters to find optimal configuration.
    """

    def __init__(
        self,
        benchmark_dir: Path = Path("data/benchmark"),
        output_dir: Path = Path("out/bench"),
        parameter_registry: Optional[List[ParameterConfig]] = None,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.parameter_registry = parameter_registry or DEFAULT_PARAMETER_REGISTRY

        self.results: List[SweepResult] = []
        self.best_result: Optional[SweepResult] = None

    def _params_to_dict(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Convert flat params to nested dict by category."""
        result = {
            "preprocess": {},
            "walls": {},
            "regions": {},
            "polygons": {},
        }

        for param in self.parameter_registry:
            if param.name in params:
                result[param.category][param.name] = params[param.name]

        return result

    def _write_params_yaml(self, params: Dict[str, Any], path: Path) -> None:
        """Write parameters to YAML file."""
        nested_params = self._params_to_dict(params)
        with open(path, "w") as f:
            yaml.dump(nested_params, f, default_flow_style=False)

    def sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameter configuration."""
        params = {}
        for param in self.parameter_registry:
            params[param.name] = param.sample()
        return params

    def get_baseline_params(self) -> Dict[str, Any]:
        """Get baseline (default) parameters."""
        return {
            "adaptive_block_size": 35,
            "adaptive_c": 10,
            "denoise_strength": 10,
            "gap_close_size": 15,
            "min_wall_length": 30,
            "min_room_area_ratio": 0.001,
            "max_room_area_ratio": 0.5,
            "min_solidity": 0.3,
            "simplification_epsilon": 2.0,
        }

    def evaluate_params(
        self,
        params: Dict[str, Any],
        trial_id: int,
        max_images: Optional[int] = None,
    ) -> SweepResult:
        """
        Evaluate a parameter configuration.

        Args:
            params: Parameter configuration
            trial_id: Trial number
            max_images: Max images to evaluate

        Returns:
            SweepResult
        """
        import time

        # Write params to temp file
        temp_params_path = self.output_dir / f"temp_params_{trial_id}.yaml"
        self._write_params_yaml(params, temp_params_path)

        start_time = time.time()

        try:
            evaluator = BenchmarkEvaluator(
                benchmark_dir=self.benchmark_dir,
                output_dir=self.output_dir / f"trial_{trial_id}",
                params_path=temp_params_path,
            )

            results = evaluator.evaluate_benchmark(max_images=max_images)

            sweep_result = SweepResult(
                trial_id=trial_id,
                params=params,
                weighted_score=results.weighted_score,
                room_f1_50=results.mean_room_f1_50,
                room_f1_75=results.mean_room_f1_75,
                label_accuracy=results.mean_label_accuracy,
                oversegmented=results.total_oversegmented,
                success=True,
                run_time_sec=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            sweep_result = SweepResult(
                trial_id=trial_id,
                params=params,
                weighted_score=0.0,
                room_f1_50=0.0,
                room_f1_75=0.0,
                label_accuracy=0.0,
                oversegmented=0,
                success=False,
                run_time_sec=time.time() - start_time,
            )

        finally:
            # Clean up temp file
            if temp_params_path.exists():
                temp_params_path.unlink()

        return sweep_result

    def random_search(
        self,
        n_trials: int = 50,
        max_images: Optional[int] = None,
    ) -> SweepResult:
        """
        Random search over parameter space.

        Args:
            n_trials: Number of random trials
            max_images: Max images per trial

        Returns:
            Best SweepResult
        """
        logger.info(f"Starting random search with {n_trials} trials")

        # Start with baseline
        baseline_params = self.get_baseline_params()
        baseline_result = self.evaluate_params(baseline_params, 0, max_images)
        self.results.append(baseline_result)
        self.best_result = baseline_result

        logger.info(f"Baseline score: {baseline_result.weighted_score:.3f}")

        for i in range(1, n_trials + 1):
            params = self.sample_random_params()
            result = self.evaluate_params(params, i, max_images)
            self.results.append(result)

            if result.success and result.weighted_score > self.best_result.weighted_score:
                self.best_result = result
                logger.info(f"Trial {i}: NEW BEST score = {result.weighted_score:.3f}")
            else:
                logger.info(f"Trial {i}: score = {result.weighted_score:.3f}")

        return self.best_result

    def grid_search(
        self,
        params_to_sweep: List[str],
        max_images: Optional[int] = None,
    ) -> SweepResult:
        """
        Grid search over specified parameters.

        Args:
            params_to_sweep: List of parameter names to sweep
            max_images: Max images per trial

        Returns:
            Best SweepResult
        """
        # Get grid values for each parameter
        grids = {}
        for param in self.parameter_registry:
            if param.name in params_to_sweep:
                grids[param.name] = param.grid_values()

        # Calculate total combinations
        total = 1
        for vals in grids.values():
            total *= len(vals)

        logger.info(f"Starting grid search with {total} combinations")

        # Generate all combinations
        from itertools import product

        keys = list(grids.keys())
        values = [grids[k] for k in keys]

        baseline = self.get_baseline_params()
        trial_id = 0

        for combo in product(*values):
            params = baseline.copy()
            for k, v in zip(keys, combo):
                params[k] = v

            result = self.evaluate_params(params, trial_id, max_images)
            self.results.append(result)

            if self.best_result is None or (result.success and result.weighted_score > self.best_result.weighted_score):
                self.best_result = result
                logger.info(f"Trial {trial_id}: NEW BEST score = {result.weighted_score:.3f}")
            else:
                logger.info(f"Trial {trial_id}: score = {result.weighted_score:.3f}")

            trial_id += 1

        return self.best_result

    def save_best_params(self, path: Optional[Path] = None) -> Path:
        """Save best parameters to YAML."""
        if path is None:
            path = Path("rules/segmentation_params.yaml")

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.best_result:
            self._write_params_yaml(self.best_result.params, path)
            logger.info(f"Saved best params to {path}")
        return path

    def save_sweep_results(self, filename: str = "sweep_results.csv") -> Path:
        """Save all sweep results to CSV."""
        import csv

        output_path = self.output_dir / filename

        with open(output_path, "w", newline="") as f:
            fieldnames = [
                "trial_id", "weighted_score", "room_f1_50", "room_f1_75",
                "label_accuracy", "oversegmented", "run_time_sec", "success"
            ]
            # Add param columns
            param_names = [p.name for p in self.parameter_registry]
            fieldnames.extend(param_names)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "trial_id": result.trial_id,
                    "weighted_score": result.weighted_score,
                    "room_f1_50": result.room_f1_50,
                    "room_f1_75": result.room_f1_75,
                    "label_accuracy": result.label_accuracy,
                    "oversegmented": result.oversegmented,
                    "run_time_sec": result.run_time_sec,
                    "success": result.success,
                }
                for pname in param_names:
                    row[pname] = result.params.get(pname, "")
                writer.writerow(row)

        logger.info(f"Saved sweep results to {output_path}")
        return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Parameter sweep for floor plan segmentation")
    parser.add_argument("--benchmark-dir", default="data/benchmark", help="Benchmark directory")
    parser.add_argument("--output-dir", default="out/bench", help="Output directory")
    parser.add_argument("--trials", type=int, default=50, help="Number of random trials")
    parser.add_argument("--max-images", type=int, help="Max images per trial")
    parser.add_argument("--method", choices=["random", "grid"], default="random", help="Search method")
    parser.add_argument("--grid-params", nargs="+", help="Params to grid search")

    args = parser.parse_args()

    sweeper = ParameterSweeper(
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir),
    )

    if args.method == "random":
        best = sweeper.random_search(n_trials=args.trials, max_images=args.max_images)
    else:
        grid_params = args.grid_params or ["gap_close_size", "adaptive_block_size"]
        best = sweeper.grid_search(params_to_sweep=grid_params, max_images=args.max_images)

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
