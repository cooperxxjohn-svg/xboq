"""
XBOQ Openings Detection Module
Door and Window detection for Indian architectural plans.

Multi-signal detection approach:
- Swing arcs (doors)
- Wall gaps/openings
- Symbol patterns (windows: double parallel lines)
- OCR tag association

Indian conventions: D, D1, W1, V, SD, MD, etc.
"""

from .detect_doors import DoorDetector, DetectedDoor
from .detect_windows import WindowDetector, DetectedWindow
from .tags import TagExtractor, OpeningTag
from .sizes import SizeInferrer, OpeningSize
from .assign import RoomAssigner, OpeningAssignment
from .export import OpeningsExporter
from .pipeline import OpeningsPipeline, OpeningsResult, run_openings_pipeline

__all__ = [
    "DoorDetector",
    "DetectedDoor",
    "WindowDetector",
    "DetectedWindow",
    "TagExtractor",
    "OpeningTag",
    "SizeInferrer",
    "OpeningSize",
    "RoomAssigner",
    "OpeningAssignment",
    "OpeningsExporter",
    "OpeningsPipeline",
    "OpeningsResult",
    "run_openings_pipeline",
]
