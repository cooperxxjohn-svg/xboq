"""
Structural Takeoff & Material Quantity Estimator
For Indian RCC Buildings

Modules:
- detect_columns: Column detection from structural drawings
- detect_beams: Beam detection with column-beam graph
- detect_footings: Footing detection from foundation plans
- schedule_extractor: Extract sizes from schedules/tables
- quantity_engine: Compute concrete volumes and steel quantities
- steel_estimator: Detailed BBS or kg/m³ estimation
- qc_structural: Quality control and explainability
- export_structural: Export to JSON/CSV/overlays
"""

__version__ = "1.0.0"

from .detect_columns import (
    DetectedColumn,
    ColumnDetectionResult,
    ColumnDetector,
    detect_columns
)

from .detect_beams import (
    DetectedBeam,
    ColumnBeamGraph,
    BeamDetectionResult,
    BeamDetector,
    detect_beams
)

from .detect_footings import (
    DetectedFooting,
    FootingDetectionResult,
    FootingDetector,
    detect_footings
)

from .schedule_extractor import (
    ExtractedSchedule,
    ScheduleExtractionResult,
    ScheduleExtractor,
    extract_schedules
)

from .quantity_engine import (
    ElementQuantity,
    QuantitySummary,
    QuantityResult,
    QuantityEngine,
    compute_quantities
)

from .steel_estimator import (
    RebarDetail,
    BBSEntry,
    SteelEstimationResult,
    SteelEstimator,
    estimate_steel,
    BAR_WEIGHTS
)

from .qc_structural import (
    QCIssue,
    AssumptionLog,
    StructuralQCReport,
    StructuralQC,
    generate_qc_report,
    Severity,
    QCCode
)

from .export_structural import (
    StructuralExporter,
    export_structural
)

from .units import (
    feet_inch_to_mm,
    mm_to_feet_inch,
    parse_footing_size,
    parse_dimension,
    extract_column_label,
    extract_footing_label,
    FOOTING_TYPES
)

from .foundation_extractor import (
    FoundationPlanData,
    FoundationPlanExtractor,
    extract_foundation_plan
)

__all__ = [
    # Version
    '__version__',

    # Column detection
    'DetectedColumn',
    'ColumnDetectionResult',
    'ColumnDetector',
    'detect_columns',

    # Beam detection
    'DetectedBeam',
    'ColumnBeamGraph',
    'BeamDetectionResult',
    'BeamDetector',
    'detect_beams',

    # Footing detection
    'DetectedFooting',
    'FootingDetectionResult',
    'FootingDetector',
    'detect_footings',

    # Schedule extraction
    'ExtractedSchedule',
    'ScheduleExtractionResult',
    'ScheduleExtractor',
    'extract_schedules',

    # Quantity computation
    'ElementQuantity',
    'QuantitySummary',
    'QuantityResult',
    'QuantityEngine',
    'compute_quantities',

    # Steel estimation
    'RebarDetail',
    'BBSEntry',
    'SteelEstimationResult',
    'SteelEstimator',
    'estimate_steel',
    'BAR_WEIGHTS',

    # QC
    'QCIssue',
    'AssumptionLog',
    'StructuralQCReport',
    'StructuralQC',
    'generate_qc_report',
    'Severity',
    'QCCode',

    # Export
    'StructuralExporter',
    'export_structural',

    # Units
    'feet_inch_to_mm',
    'mm_to_feet_inch',
    'parse_footing_size',
    'parse_dimension',
    'extract_column_label',
    'extract_footing_label',
    'FOOTING_TYPES',

    # Foundation plan extraction
    'FoundationPlanData',
    'FoundationPlanExtractor',
    'extract_foundation_plan'
]
