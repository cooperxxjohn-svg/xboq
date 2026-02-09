"""
Estimator Output Schema - Enhanced Version
Structured data format for construction quantity estimation.

Includes:
- Project and material info
- Element schedules (columns, footings, beams)
- Scope checklist (detected vs missing)
- Requirements extracted from notes
- BOQ items with dependencies
- Assumptions log
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class QtyStatus(Enum):
    """Quantity computation status."""
    COMPUTED = "computed"      # Fully calculated
    PARTIAL = "partial"        # Some inputs missing
    UNKNOWN = "unknown"        # Cannot compute
    MANUAL = "manual"          # Requires manual input


class ConfidenceLevel(Enum):
    """Confidence level for extracted data."""
    HIGH = "high"       # > 80%
    MEDIUM = "medium"   # 50-80%
    LOW = "low"         # < 50%


# =============================================================================
# PROJECT INFO
# =============================================================================

@dataclass
class ProjectInfo:
    """Project metadata extracted from drawing."""
    sheet_name: str = ""
    sheet_number: str = ""
    scale: str = "1:100"
    drawing_title: str = ""
    project_name: str = ""
    date: str = ""
    revision: str = ""
    drawn_by: str = ""
    checked_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialSpecs:
    """Material specifications from drawing."""
    concrete_grade: str = "M25"
    steel_grade: str = "Fe500"
    cover_mm: int = 50
    soil_bearing_capacity: Optional[float] = None  # t/sqm
    exposure_class: Optional[str] = None
    mix_design: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# ELEMENT ENTRIES
# =============================================================================

@dataclass
class ColumnEntry:
    """Single column entry for schedule."""
    mark: str  # e.g., "C1", "C2"
    column_type: str = "rectangular"  # rectangular, circular
    size_mm: Tuple[int, int] = (300, 300)  # width x depth
    level: str = "Foundation"  # Foundation, Plinth, Ground, etc.
    count: int = 1
    height_mm: Optional[int] = None  # None if not detected
    confidence: float = 0.8
    evidence: Dict[str, Any] = field(default_factory=dict)
    source: str = "detection"  # detection, schedule, callout

    # Computed quantities (in SI units)
    concrete_volume_m3: float = 0.0
    steel_kg: float = 0.0
    formwork_sqm: float = 0.0

    # Quantity status
    qty_status: str = "unknown"  # computed, partial, unknown
    dependencies: List[str] = field(default_factory=list)

    def compute_quantities(self, steel_ratio_kg_m3: float = 120.0):
        """Calculate concrete volume, steel, and formwork."""
        if self.height_mm is None:
            self.qty_status = "unknown"
            self.dependencies = ["column height/level not available"]
            return

        # Volume = width × depth × height × count (convert mm to m)
        w_m = self.size_mm[0] / 1000
        d_m = self.size_mm[1] / 1000
        h_m = self.height_mm / 1000

        self.concrete_volume_m3 = w_m * d_m * h_m * self.count
        self.steel_kg = self.concrete_volume_m3 * steel_ratio_kg_m3

        # Formwork = perimeter × height × count
        perimeter_m = 2 * (w_m + d_m)
        self.formwork_sqm = perimeter_m * h_m * self.count

        self.qty_status = "computed"
        self.dependencies = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mark': self.mark,
            'type': self.column_type,
            'size_mm': list(self.size_mm),
            'size_m': [self.size_mm[0]/1000, self.size_mm[1]/1000],
            'level': self.level,
            'count': self.count,
            'height_mm': self.height_mm,
            'height_m': self.height_mm / 1000 if self.height_mm else None,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'source': self.source,
            'concrete_m3': round(self.concrete_volume_m3, 4),
            'steel_kg': round(self.steel_kg, 2),
            'formwork_sqm': round(self.formwork_sqm, 2),
            'qty_status': self.qty_status,
            'dependencies': self.dependencies
        }


@dataclass
class FootingEntry:
    """Single footing entry for schedule."""
    mark: str  # e.g., "F1", "F2"
    shape: str = "isolated"  # isolated, combined, strip
    L_mm: int = 1500  # Length
    B_mm: int = 1500  # Breadth
    D_mm: Optional[int] = None  # Depth - may not be detected
    count: int = 1
    confidence: float = 0.8
    column_marks: List[str] = field(default_factory=list)
    source: str = "detection"

    # Computed quantities (SI units)
    concrete_volume_m3: float = 0.0
    steel_kg: float = 0.0
    formwork_sqm: float = 0.0
    excavation_m3: float = 0.0
    pcc_m3: float = 0.0

    # Quantity status
    qty_status: str = "unknown"
    dependencies: List[str] = field(default_factory=list)

    def compute_quantities(
        self,
        steel_ratio_kg_m3: float = 80.0,
        pcc_thickness_mm: Optional[int] = 100,
        include_pcc: bool = True
    ):
        """Calculate concrete volume, steel, formwork, excavation, and PCC."""
        self.dependencies = []

        L_m = self.L_mm / 1000
        B_m = self.B_mm / 1000

        if self.D_mm is None:
            # Estimate depth from size (typical D/B ratio 0.25-0.35)
            self.D_mm = max(300, min(600, int(min(self.L_mm, self.B_mm) * 0.3)))
            self.dependencies.append("footing depth estimated from size")

        D_m = self.D_mm / 1000

        # RCC Volume = L × B × D × count
        self.concrete_volume_m3 = L_m * B_m * D_m * self.count

        # Steel from ratio (footings typically 60-100 kg/m³)
        self.steel_kg = self.concrete_volume_m3 * steel_ratio_kg_m3

        # Formwork = perimeter × depth × count (4 sides)
        perimeter_m = 2 * (L_m + B_m)
        self.formwork_sqm = perimeter_m * D_m * self.count

        # Excavation = (L + 0.6) × (B + 0.6) × (D + pcc_depth) × count
        pcc_m = (pcc_thickness_mm or 100) / 1000
        self.excavation_m3 = (L_m + 0.6) * (B_m + 0.6) * (D_m + pcc_m) * self.count

        # PCC = (L + 0.3) × (B + 0.3) × thickness × count
        if include_pcc and pcc_thickness_mm:
            self.pcc_m3 = (L_m + 0.3) * (B_m + 0.3) * pcc_m * self.count
        else:
            self.pcc_m3 = 0.0
            if not pcc_thickness_mm:
                self.dependencies.append("PCC thickness not specified")

        self.qty_status = "computed" if not self.dependencies else "partial"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mark': self.mark,
            'shape': self.shape,
            'L_mm': self.L_mm,
            'B_mm': self.B_mm,
            'D_mm': self.D_mm,
            'L_m': self.L_mm / 1000,
            'B_m': self.B_mm / 1000,
            'D_m': self.D_mm / 1000 if self.D_mm else None,
            'count': self.count,
            'confidence': self.confidence,
            'column_marks': self.column_marks,
            'source': self.source,
            'concrete_m3': round(self.concrete_volume_m3, 4),
            'steel_kg': round(self.steel_kg, 2),
            'formwork_sqm': round(self.formwork_sqm, 2),
            'excavation_m3': round(self.excavation_m3, 3),
            'pcc_m3': round(self.pcc_m3, 4),
            'qty_status': self.qty_status,
            'dependencies': self.dependencies
        }


@dataclass
class BeamEntry:
    """Single beam entry for schedule."""
    mark: str  # e.g., "B1", "PB1"
    beam_type: str = "plinth"  # plinth, tie, lintel, roof
    width_mm: int = 230
    depth_mm: int = 450
    span_mm: Optional[int] = None
    count: int = 1
    confidence: float = 0.5
    source: str = "detection"

    # Computed quantities
    concrete_volume_m3: float = 0.0
    steel_kg: float = 0.0
    formwork_sqm: float = 0.0

    qty_status: str = "unknown"
    dependencies: List[str] = field(default_factory=list)

    def compute_quantities(self, steel_ratio_kg_m3: float = 100.0):
        """Calculate beam quantities."""
        if self.span_mm is None:
            self.qty_status = "unknown"
            self.dependencies = ["beam span not available"]
            return

        w_m = self.width_mm / 1000
        d_m = self.depth_mm / 1000
        span_m = self.span_mm / 1000

        self.concrete_volume_m3 = w_m * d_m * span_m * self.count
        self.steel_kg = self.concrete_volume_m3 * steel_ratio_kg_m3

        # Formwork = (2×depth + width) × span × count
        self.formwork_sqm = (2 * d_m + w_m) * span_m * self.count

        self.qty_status = "computed"
        self.dependencies = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mark': self.mark,
            'type': self.beam_type,
            'width_mm': self.width_mm,
            'depth_mm': self.depth_mm,
            'span_mm': self.span_mm,
            'count': self.count,
            'confidence': self.confidence,
            'concrete_m3': round(self.concrete_volume_m3, 4),
            'steel_kg': round(self.steel_kg, 2),
            'formwork_sqm': round(self.formwork_sqm, 2),
            'qty_status': self.qty_status,
            'dependencies': self.dependencies
        }


# =============================================================================
# BOQ ITEM
# =============================================================================

@dataclass
class BOQItem:
    """Bill of Quantities line item with dependency tracking."""
    item_name: str
    unit: str
    qty: Optional[float] = None
    qty_status: str = "unknown"  # computed, partial, unknown
    basis: str = ""  # Calculation basis description
    dependencies: List[str] = field(default_factory=list)
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    trade: str = ""  # civil, rcc, steel, etc.
    element_type: str = ""  # Column, Footing, Beam

    # 3-Phase extraction fields
    source: str = "explicit"  # explicit, synonym, inferred
    rule_fired: str = ""  # Rule ID that generated this item
    keywords_matched: List[str] = field(default_factory=list)  # Matched synonyms/keywords
    evidence_text: str = ""  # Raw text evidence
    bbox: Optional[List[float]] = None  # Location on page [x1, y1, x2, y2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_name': self.item_name,
            'unit': self.unit,
            'qty': round(self.qty, 3) if self.qty is not None else None,
            'qty_status': self.qty_status,
            'basis': self.basis,
            'dependencies': self.dependencies,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'trade': self.trade,
            'element_type': self.element_type,
            'source': self.source,
            'rule_fired': self.rule_fired,
            'keywords_matched': self.keywords_matched,
            'evidence_text': self.evidence_text,
            'bbox': self.bbox
        }


# =============================================================================
# SCOPE CHECKLIST
# =============================================================================

@dataclass
class ScopeChecklist:
    """Tracks what was detected vs what's missing."""
    detected: List[str] = field(default_factory=list)
    missing_or_unclear: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'detected': self.detected,
            'missing_or_unclear': self.missing_or_unclear
        }


# =============================================================================
# REQUIREMENT
# =============================================================================

@dataclass
class Requirement:
    """Extracted requirement from notes/legends."""
    category: str  # Material, Execution, QA/QC, Code, etc.
    requirement: str  # Description
    value: Optional[str] = None  # Specific value if any
    source_text: str = ""  # Original text
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# ASSUMPTION
# =============================================================================

@dataclass
class AssumptionEntry:
    """Assumption made during estimation."""
    category: str  # Geometry, Material, Reinforcement, etc.
    description: str
    assumed_value: str
    source: str = "IS 456:2000"
    impact: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# MAIN ESTIMATOR OUTPUT
# =============================================================================

@dataclass
class EstimatorOutput:
    """
    Complete estimator output pack.
    Contains all data needed for BOQ generation and Excel export.
    """
    # Project info
    project: ProjectInfo = field(default_factory=ProjectInfo)
    materials: MaterialSpecs = field(default_factory=MaterialSpecs)

    # Element schedules
    columns: List[ColumnEntry] = field(default_factory=list)
    footings: List[FootingEntry] = field(default_factory=list)
    beams: List[BeamEntry] = field(default_factory=list)

    # Scope tracking
    scope_checklist: ScopeChecklist = field(default_factory=ScopeChecklist)

    # Requirements from notes
    requirements: List[Requirement] = field(default_factory=list)

    # BOQ items
    boq_items: List[BOQItem] = field(default_factory=list)

    # Assumptions/notes
    assumptions: List[AssumptionEntry] = field(default_factory=list)

    # Summary totals
    total_columns: int = 0
    total_footings: int = 0
    total_beams: int = 0
    total_concrete_m3: float = 0.0
    total_steel_kg: float = 0.0
    total_formwork_sqm: float = 0.0
    total_excavation_m3: float = 0.0

    # Metadata
    confidence: float = 0.0
    processing_mode: str = "structural"
    has_bar_schedule: bool = False

    def compute_all_quantities(self):
        """Compute quantities for all elements."""
        logger.info("Computing quantities for all elements...")

        # Compute column quantities
        for col in self.columns:
            col.compute_quantities(steel_ratio_kg_m3=120.0)

        # Compute footing quantities
        for ftg in self.footings:
            ftg.compute_quantities(steel_ratio_kg_m3=80.0)

        # Compute beam quantities
        for beam in self.beams:
            beam.compute_quantities(steel_ratio_kg_m3=100.0)

        # Calculate totals
        self._calculate_totals()

        logger.info(f"Computed: {self.total_concrete_m3:.2f} m³ concrete, "
                   f"{self.total_steel_kg:.0f} kg steel")

    def _calculate_totals(self):
        """Sum up all quantities."""
        self.total_columns = sum(c.count for c in self.columns)
        self.total_footings = sum(f.count for f in self.footings)
        self.total_beams = sum(b.count for b in self.beams)

        # Concrete
        col_concrete = sum(c.concrete_volume_m3 for c in self.columns)
        ftg_concrete = sum(f.concrete_volume_m3 for f in self.footings)
        beam_concrete = sum(b.concrete_volume_m3 for b in self.beams)
        self.total_concrete_m3 = col_concrete + ftg_concrete + beam_concrete

        # Steel
        col_steel = sum(c.steel_kg for c in self.columns)
        ftg_steel = sum(f.steel_kg for f in self.footings)
        beam_steel = sum(b.steel_kg for b in self.beams)
        self.total_steel_kg = col_steel + ftg_steel + beam_steel

        # Formwork
        col_formwork = sum(c.formwork_sqm for c in self.columns)
        ftg_formwork = sum(f.formwork_sqm for f in self.footings)
        beam_formwork = sum(b.formwork_sqm for b in self.beams)
        self.total_formwork_sqm = col_formwork + ftg_formwork + beam_formwork

        # Excavation
        self.total_excavation_m3 = sum(f.excavation_m3 for f in self.footings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'project': self.project.to_dict(),
            'materials': self.materials.to_dict(),
            'columns': [c.to_dict() for c in self.columns],
            'footings': [f.to_dict() for f in self.footings],
            'beams': [b.to_dict() for b in self.beams],
            'scope_checklist': self.scope_checklist.to_dict(),
            'requirements': [r.to_dict() for r in self.requirements],
            'boq_items': [b.to_dict() for b in self.boq_items],
            'assumptions': [a.to_dict() for a in self.assumptions],
            'summary': {
                'total_columns': self.total_columns,
                'total_footings': self.total_footings,
                'total_beams': self.total_beams,
                'total_concrete_m3': round(self.total_concrete_m3, 3),
                'total_steel_kg': round(self.total_steel_kg, 1),
                'total_steel_tonnes': round(self.total_steel_kg / 1000, 3),
                'total_formwork_sqm': round(self.total_formwork_sqm, 2),
                'total_excavation_m3': round(self.total_excavation_m3, 2)
            },
            'confidence': self.confidence,
            'processing_mode': self.processing_mode,
            'has_bar_schedule': self.has_bar_schedule
        }


# =============================================================================
# BUILDER FUNCTION
# =============================================================================

def build_estimator_output(
    foundation_data: Any,
    quantity_result: Any = None,
    floors: int = 1,
    storey_height_mm: int = 3000,
    ocr_notes_text: str = "",
    sheet_meta: Dict[str, Any] = None
) -> EstimatorOutput:
    """
    Build EstimatorOutput from foundation extraction results.

    Args:
        foundation_data: FoundationPlanData from foundation_extractor
        quantity_result: Optional QuantityResult for additional data
        floors: Number of floors
        storey_height_mm: Height per storey
        ocr_notes_text: Raw OCR text from notes/legends
        sheet_meta: Optional sheet metadata dict

    Returns:
        EstimatorOutput with computed quantities
    """
    output = EstimatorOutput()

    # Set project info
    sheet_meta = sheet_meta or {}
    output.project = ProjectInfo(
        sheet_name=sheet_meta.get('sheet_name', getattr(foundation_data, 'sheet_name', 'Foundation Plan')),
        sheet_number=sheet_meta.get('sheet_number', ''),
        scale=getattr(foundation_data, 'scale', '1:100'),
        drawing_title=sheet_meta.get('title', 'Foundation Layout'),
        project_name=sheet_meta.get('project_name', ''),
    )

    # Set materials
    output.materials = MaterialSpecs(
        concrete_grade=getattr(foundation_data, 'concrete_grade', 'M25'),
        steel_grade=getattr(foundation_data, 'steel_grade', 'Fe500'),
        soil_bearing_capacity=getattr(foundation_data, 'soil_bearing', None)
    )

    # =========================================================================
    # GET TABLE-EXTRACTED DATA (if available)
    # =========================================================================
    table_columns = getattr(foundation_data, 'table_columns', [])
    table_footings = getattr(foundation_data, 'table_footings', [])
    bar_schedule = getattr(foundation_data, 'bar_schedule', [])

    logger.info(f"Table data: {len(table_columns)} columns, {len(table_footings)} footings, {len(bar_schedule)} bars")

    # =========================================================================
    # BUILD COLUMN ENTRIES
    # =========================================================================
    # Priority: table data > visual detection
    # Build column entries - group by size
    column_groups: Dict[Tuple[int, int], List[Any]] = {}

    # First, add table-extracted columns (higher confidence)
    for tc in table_columns:
        size = tc.get('size_mm')
        if size and isinstance(size, (list, tuple)) and len(size) >= 2:
            size = (int(size[0]), int(size[1]))
        else:
            continue  # Skip if no valid size

        if size not in column_groups:
            column_groups[size] = []

        # Create a mock column object for consistency
        class TableColumn:
            def __init__(self, data):
                self.size_mm = data.get('size_mm')
                self.label = data.get('mark', '')
                self.confidence = data.get('confidence', 0.9)
                self.source = 'table'

        column_groups[size].append(TableColumn(tc))

    # Then add visual detection columns (only if not already found in table)
    for col in getattr(foundation_data, 'columns', []):
        if hasattr(col, 'size_mm') and col.size_mm:
            size = tuple(col.size_mm) if isinstance(col.size_mm, (list, tuple)) else (300, 300)
        else:
            size = (300, 300)

        # Check if this size already has table data
        has_table_data = any(
            tc.get('size_mm') == list(size) or tc.get('size_mm') == size
            for tc in table_columns
        )

        if not has_table_data:
            if size not in column_groups:
                column_groups[size] = []
            column_groups[size].append(col)

    for size, cols in column_groups.items():
        marks = [c.label for c in cols if hasattr(c, 'label') and c.label]
        mark = marks[0] if marks else f"C-{size[0]}x{size[1]}"

        # Determine source
        source = 'detection'
        confidence = 0.8
        if cols and hasattr(cols[0], 'source') and cols[0].source == 'table':
            source = 'table'
            confidence = 0.9

        entry = ColumnEntry(
            mark=mark,
            column_type="rectangular",
            size_mm=size,
            level="Foundation to Plinth",
            count=len(cols),
            height_mm=storey_height_mm,  # Use provided height
            confidence=confidence,
            evidence={'column_labels': marks[:10], 'source': source},
            source=source
        )
        output.columns.append(entry)

    # =========================================================================
    # BUILD FOOTING ENTRIES
    # =========================================================================
    # Priority: table data > visual detection
    footing_types = getattr(foundation_data, 'footing_types', {})
    column_footing_map = getattr(foundation_data, 'column_footing_map', {})

    # First, add table-extracted footings (higher confidence, may have depth)
    footings_added = set()

    for tf in table_footings:
        mark = tf.get('mark', '')
        if not mark:
            continue

        L_mm = tf.get('L_mm', 1500)
        B_mm = tf.get('B_mm', 1500)
        D_mm = tf.get('D_mm')  # May be None or actual value

        entry = FootingEntry(
            mark=mark,
            shape="isolated",
            L_mm=int(L_mm),
            B_mm=int(B_mm),
            D_mm=int(D_mm) if D_mm else None,
            count=1,  # Table entries are usually per footing type
            confidence=tf.get('confidence', 0.9),
            column_marks=[tf.get('column_mark', '')] if tf.get('column_mark') else [],
            source="table"
        )
        output.footings.append(entry)
        footings_added.add(mark.upper())

    # Count footings from visual detection (only if not already from table)
    footing_counts: Dict[str, int] = {}
    footing_columns: Dict[str, List[str]] = {}

    for col_mark, ftg_type in column_footing_map.items():
        if ftg_type.upper() in footings_added:
            continue  # Skip if already added from table

        if ftg_type not in footing_counts:
            footing_counts[ftg_type] = 0
            footing_columns[ftg_type] = []
        footing_counts[ftg_type] += 1
        footing_columns[ftg_type].append(col_mark)

    # Also count from footings list
    for ftg in getattr(foundation_data, 'footings', []):
        label = getattr(ftg, 'label', '')
        if label and label.upper() not in footings_added and label not in footing_counts:
            footing_counts[label] = 1
            footing_columns[label] = []

    for ftg_type, count in footing_counts.items():
        if ftg_type in footing_types:
            size = footing_types[ftg_type]
            L_mm = int(size[0]) if isinstance(size, (list, tuple)) else 1500
            B_mm = int(size[1]) if isinstance(size, (list, tuple)) else 1500
        else:
            L_mm, B_mm = 1500, 1500

        # Depth may not be available
        D_mm = None  # Will be estimated in compute_quantities

        entry = FootingEntry(
            mark=ftg_type,
            shape="isolated",
            L_mm=L_mm,
            B_mm=B_mm,
            D_mm=D_mm,
            count=count,
            confidence=0.85,
            column_marks=footing_columns.get(ftg_type, [])[:5],
            source="detection"
        )
        output.footings.append(entry)

    # Build beam entries (from tie beams if available)
    for tb in getattr(foundation_data, 'tie_beams', []):
        entry = BeamEntry(
            mark=tb.get('label', 'TB1'),
            beam_type="tie",
            width_mm=int(tb.get('width_mm', 230)),
            depth_mm=int(tb.get('depth_mm', 300)),
            span_mm=None,  # Usually not detected
            count=1,
            confidence=0.6,
            source="detection"
        )
        output.beams.append(entry)

    # Set confidence
    output.confidence = getattr(foundation_data, 'confidence', 0.8)
    output.processing_mode = "structural"

    # Check for bar schedule (from table extraction or text keywords)
    notes = getattr(foundation_data, 'notes', [])
    notes_text = ' '.join(notes) + ' ' + ocr_notes_text

    # Bar schedule detected if we have table-extracted bar data or keywords in text
    output.has_bar_schedule = (
        len(bar_schedule) > 0 or
        any(
            kw in notes_text.upper()
            for kw in ['BAR BENDING', 'BBS', 'REINFORCEMENT SCHEDULE', 'REBAR SCHEDULE']
        )
    )

    if bar_schedule:
        logger.info(f"Bar schedule detected with {len(bar_schedule)} entries from table extraction")

    # Add standard assumptions
    output.assumptions = [
        AssumptionEntry(
            category="Geometry",
            description="Column height taken as foundation to plinth level",
            assumed_value=f"{storey_height_mm} mm",
            source="User input",
            impact="high"
        ),
        AssumptionEntry(
            category="Geometry",
            description="Footing depth estimated from size (D/B ratio)",
            assumed_value="0.30 × min(L, B)",
            source="Thumb rule",
            impact="medium"
        ),
        AssumptionEntry(
            category="Reinforcement",
            description="Column steel ratio",
            assumed_value="120 kg/m³",
            source="IS 456:2000 typical",
            impact="medium"
        ),
        AssumptionEntry(
            category="Reinforcement",
            description="Footing steel ratio",
            assumed_value="80 kg/m³",
            source="IS 456:2000 typical",
            impact="medium"
        ),
        AssumptionEntry(
            category="Excavation",
            description="Working space around footings",
            assumed_value="300 mm each side",
            source="Standard practice",
            impact="low"
        ),
        AssumptionEntry(
            category="PCC",
            description="Lean concrete below footing",
            assumed_value="100 mm M10 grade",
            source="IS 456:2000",
            impact="low"
        )
    ]

    # Compute all quantities
    output.compute_all_quantities()

    return output
