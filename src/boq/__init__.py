"""
XBOQ - Bill of Quantities Engine
India-first BOQ generation for residential RCC projects.

Modules:
- wall_boq: Wall thickness clustering, brick/plaster quantities
- finish_boq: Room finish mapping and quantities
- slab_boq: Slab area and concrete volume
- steel_boq: Reinforcement steel estimation
- openings_boq: Door/window schedules
- export: BOQ CSV and report generation
- confidence: Confidence scoring and heatmap
- schema: Standardized BOQ data structures and validators
"""

from .wall_boq import WallBOQCalculator, WallBOQItem
from .finish_boq import FinishBOQCalculator, FinishBOQItem
from .slab_boq import SlabBOQCalculator, SlabBOQItem
from .steel_boq import SteelBOQCalculator, SteelBOQItem
from .openings_boq import OpeningsBOQCalculator, OpeningBOQItem
from .export import BOQExporter, BOQPackage
from .confidence import ConfidenceCalculator, ConfidenceHeatmap
from .engine import BOQEngine, BOQResult
from .schema import BOQItem, BOQValidator, BOQPackageSchema, load_profile, merge_boq_items

__all__ = [
    "WallBOQCalculator",
    "WallBOQItem",
    "FinishBOQCalculator",
    "FinishBOQItem",
    "SlabBOQCalculator",
    "SlabBOQItem",
    "SteelBOQCalculator",
    "SteelBOQItem",
    "OpeningsBOQCalculator",
    "OpeningBOQItem",
    "BOQExporter",
    "BOQPackage",
    "ConfidenceCalculator",
    "ConfidenceHeatmap",
    "BOQEngine",
    "BOQResult",
    "BOQItem",
    "BOQValidator",
    "BOQPackageSchema",
    "load_profile",
    "merge_boq_items",
]
