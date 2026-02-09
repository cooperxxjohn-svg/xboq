"""
Benchmarking and Evaluation Framework for Floor Plan Analysis.

This module provides:
- Annotation format and helpers
- Evaluation metrics (IoU, precision/recall, F1)
- Failure mining
- Parameter sweeps
- Report generation

India context:
- Supports Indian room names (Toilet/WC/Bath/OTS/Pooja etc.)
- Mixed units handling
"""

from .annotation import AnnotationSchema, load_annotations, save_annotations
from .metrics import compute_room_metrics, compute_opening_metrics
from .evaluate import BenchmarkEvaluator
from .sweep import ParameterSweeper
from .failures import FailureMiner
from .report import ReportGenerator

__all__ = [
    "AnnotationSchema",
    "load_annotations",
    "save_annotations",
    "compute_room_metrics",
    "compute_opening_metrics",
    "BenchmarkEvaluator",
    "ParameterSweeper",
    "FailureMiner",
    "ReportGenerator",
]
