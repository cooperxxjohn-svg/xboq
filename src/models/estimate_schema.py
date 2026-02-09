"""
XBOQ Estimate Schema
Strict pydantic models for India-first preconstruction BOQ & scope tool.

This schema enforces data integrity and prevents hallucination by:
- Using strict enums for categorical fields
- Validating quantity/confidence ranges
- Requiring evidence for all claims
- Tracking dependencies explicitly

NO PRICING - pure scope and quantity extraction.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


# =============================================================================
# ENUMS - Strict categorical values
# =============================================================================

class Discipline(str, Enum):
    """Drawing discipline type."""
    STRUCTURAL = "structural"
    ARCHITECTURAL = "architectural"
    MEP = "mep"
    CIVIL = "civil"
    UNKNOWN = "unknown"


class EvidenceSource(str, Enum):
    """Source of extracted evidence."""
    OCR = "ocr"
    CAMELOT = "camelot"
    PDF_TEXT = "pdf_text"
    PDFPLUMBER = "pdfplumber"
    HEURISTIC = "heuristic"
    USER_INPUT = "user_input"
    INFERRED = "inferred"  # For items derived from other evidence


class ScopeCategory(str, Enum):
    """Standard scope categories for Indian construction."""
    EARTHWORK = "earthwork"
    RCC = "rcc"
    MASONRY = "masonry"
    FINISHES = "finishes"
    WATERPROOFING = "waterproofing"
    DOORS_WINDOWS = "doors_windows"
    PLUMBING = "plumbing"
    ELECTRICAL = "electrical"
    SITEWORKS = "siteworks"
    MISC = "misc"


class ScopeStatus(str, Enum):
    """Status of scope item detection."""
    DETECTED = "detected"      # Found in drawing with evidence
    INFERRED = "inferred"      # Logically implied (e.g., PCC below footings)
    MISSING = "missing"        # Expected but not found
    NEEDS_REVIEW = "needs_review"  # Ambiguous or low confidence


class QtyStatus(str, Enum):
    """Quantity computation status."""
    COMPUTED = "computed"      # Fully calculated
    PARTIAL = "partial"        # Some inputs missing
    UNKNOWN = "unknown"        # Cannot compute without more data


class ConflictType(str, Enum):
    """Types of conflicts/issues detected."""
    SCHEDULE_VS_PLAN = "schedule_vs_plan"
    NOTES_VS_SCHEDULE = "notes_vs_schedule"
    MISSING_SCALE = "missing_scale"
    MISSING_LABELS = "missing_labels"
    AMBIGUOUS_MARK = "ambiguous_mark"
    DUPLICATE_MARK = "duplicate_mark"
    GRADE_MISMATCH = "grade_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"


class Severity(str, Enum):
    """Issue severity level."""
    LOW = "low"
    MED = "med"
    HIGH = "high"


# Allowed units in Indian construction
ALLOWED_UNITS = {
    "cum", "sqm", "rmt", "kg", "nos", "ls", "mt",
    "m3", "m2", "m", "kg/m", "quintal", "tonne",
    "sqft", "cft", "rft",  # Imperial still used
    None  # Unknown/not applicable
}


# =============================================================================
# CORE MODELS
# =============================================================================

class Evidence(BaseModel):
    """
    Evidence linking a claim to its source in the drawing.
    Every extracted fact must have evidence.
    """
    page: int = Field(ge=0, description="0-indexed page number")
    source: EvidenceSource = Field(description="Extraction method")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="Bounding box (x0, y0, x1, y1) if available"
    )
    snippet: str = Field(default="", max_length=500, description="Text snippet or description")
    table_id: Optional[str] = Field(default=None, description="Table identifier if from schedule")

    class Config:
        frozen = True  # Evidence is immutable


class DrawingMeta(BaseModel):
    """
    Metadata about the source drawing.
    Captures what we know about the document.
    """
    file_name: str
    sheet_name: Optional[str] = Field(default=None, description="Sheet name if detected")
    discipline: Discipline = Field(default=Discipline.UNKNOWN)
    scale: Optional[str] = Field(default=None, description="e.g., '1:100', '1:50'")
    units: str = Field(default="mm", description="Drawing units")
    revision: Optional[str] = Field(default=None)
    date_detected: Optional[str] = Field(default=None)

    # What we found in the drawing
    detected_keywords: List[str] = Field(default_factory=list)
    page_count: int = Field(default=1, ge=1)
    has_schedule_tables: bool = Field(default=False)
    has_plan_view: bool = Field(default=False)
    has_notes_section: bool = Field(default=False)

    # Quality indicators
    confidence_overall: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator('confidence_overall')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        return round(v, 3)


class ScopeItem(BaseModel):
    """
    A scope item representing work to be done.
    Uses Indian construction terminology.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: ScopeCategory
    trade: str = Field(description="e.g., 'RCC Columns', 'PCC below footings', 'Brickwork 230mm'")
    status: ScopeStatus
    reason: str = Field(description="Why this status was assigned")
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_reason: str = Field(
        default="",
        description="Explainable reason for confidence score"
    )

    # Derived from evidence
    pages_found: Set[int] = Field(default_factory=set)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        return round(v, 3)

    @model_validator(mode='after')
    def populate_pages(self):
        if self.evidence and not self.pages_found:
            self.pages_found = {e.page for e in self.evidence}
        return self

    class Config:
        # Allow set serialization
        json_encoders = {set: list}


class BOQItem(BaseModel):
    """
    Bill of Quantities item.
    NO PRICING - just scope, quantity, and evidence.

    Uses Indian construction terminology:
    - RCC (Reinforced Cement Concrete)
    - PCC (Plain Cement Concrete)
    - Shuttering/Centering (Formwork)
    - Khudai (Excavation)
    - Saria (Reinforcement Steel)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Classification
    system: str = Field(description="e.g., 'structural', 'architectural', 'mep'")
    subsystem: str = Field(description="e.g., 'rcc', 'masonry', 'finishes'")

    # Item details (India-first wording)
    item_name: str = Field(description="Normalized item name in Indian terminology")
    description: Optional[str] = Field(default=None)

    # Quantity
    unit: Optional[str] = Field(default=None)
    qty: Optional[float] = Field(default=None, ge=0)
    qty_status: QtyStatus = Field(default=QtyStatus.UNKNOWN)
    measurement_rule: str = Field(
        default="",
        description="How qty was derived or what inputs are missing"
    )

    # Dependencies - what's needed to compute qty
    dependencies: List[str] = Field(
        default_factory=list,
        description="e.g., ['need storey height', 'need column count']"
    )

    # Source tracking
    source: str = Field(default="explicit", description="'explicit', 'synonym', 'inferred'")
    rule_fired: Optional[str] = Field(default=None, description="Which rule generated this")
    keywords_matched: List[str] = Field(default_factory=list)

    # Evidence
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_text: Optional[str] = Field(default=None, max_length=1000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_reason: str = Field(
        default="",
        description="Explainable reason for confidence score"
    )

    @field_validator('unit')
    @classmethod
    def validate_unit(cls, v):
        if v is not None and v.lower() not in ALLOWED_UNITS:
            # Don't reject, just warn - allow flexibility
            pass
        return v.lower() if v else None

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        return round(v, 3)

    @model_validator(mode='after')
    def validate_qty_status(self):
        """Qty must be numeric if qty_status != 'unknown'."""
        if self.qty_status == QtyStatus.COMPUTED:
            if self.qty is None:
                raise ValueError("qty must be set when qty_status is 'computed'")
        if self.qty_status == QtyStatus.UNKNOWN:
            # qty can be None or a placeholder
            pass
        return self

    def to_export_dict(self) -> Dict[str, Any]:
        """Format for Excel/CSV export."""
        return {
            "S.No": self.id,
            "System": self.system.title(),
            "Subsystem": self.subsystem.title(),
            "Description": self.item_name,
            "Unit": self.unit or "-",
            "Qty": self.qty if self.qty is not None else "-",
            "Qty Status": self.qty_status.value,
            "Dependencies": ", ".join(self.dependencies) if self.dependencies else "-",
            "Confidence": f"{self.confidence:.0%}",
            "Confidence Reason": self.confidence_reason or "-",
            "Evidence Pages": ", ".join(str(e.page + 1) for e in self.evidence) if self.evidence else "-",
            "Source": self.source,
            "Measurement Rule": self.measurement_rule or "-",
        }


class CoverageRecord(BaseModel):
    """
    Tracks how well a BOQ item is supported by evidence.
    """
    boq_item_id: str
    contributed_by: List[Evidence] = Field(default_factory=list)
    pages_used: Set[int] = Field(default_factory=set)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # What sources contributed
    sources_used: Set[EvidenceSource] = Field(default_factory=set)

    # Breakdown of how score was computed
    coverage_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of score components"
    )

    @model_validator(mode='after')
    def compute_coverage(self):
        if self.contributed_by and not self.pages_used:
            self.pages_used = {e.page for e in self.contributed_by}
        if self.contributed_by and not self.sources_used:
            self.sources_used = {e.source for e in self.contributed_by}
        return self

    class Config:
        json_encoders = {set: list}


class Conflict(BaseModel):
    """
    A detected conflict or issue requiring attention.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ConflictType
    description: str
    severity: Severity
    evidence: List[Evidence] = Field(default_factory=list)
    suggested_resolution: str = Field(default="")

    # Status
    resolved: bool = Field(default=False)
    resolution_notes: Optional[str] = Field(default=None)


class EstimatePackage(BaseModel):
    """
    Complete takeoff package for a drawing.
    This is the main output of the extraction pipeline.
    """
    # Metadata
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")

    # Drawing info
    drawing: DrawingMeta

    # Extracted data
    scope: List[ScopeItem] = Field(default_factory=list)
    boq: List[BOQItem] = Field(default_factory=list)
    coverage: List[CoverageRecord] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)

    # Export paths (filled after export)
    exports: Dict[str, str] = Field(default_factory=dict)

    # Summary stats
    @property
    def stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        scope_by_status = {}
        for s in self.scope:
            scope_by_status[s.status.value] = scope_by_status.get(s.status.value, 0) + 1

        boq_by_status = {}
        for b in self.boq:
            boq_by_status[b.qty_status.value] = boq_by_status.get(b.qty_status.value, 0) + 1

        conflicts_by_severity = {}
        for c in self.conflicts:
            conflicts_by_severity[c.severity.value] = conflicts_by_severity.get(c.severity.value, 0) + 1

        return {
            "total_scope_items": len(self.scope),
            "scope_by_status": scope_by_status,
            "total_boq_items": len(self.boq),
            "boq_by_qty_status": boq_by_status,
            "total_conflicts": len(self.conflicts),
            "conflicts_by_severity": conflicts_by_severity,
            "high_severity_conflicts": conflicts_by_severity.get("high", 0),
            "drawing_confidence": self.drawing.confidence_overall,
        }

    def get_items_needing_review(self) -> List[BOQItem]:
        """Get BOQ items that need human review."""
        return [
            b for b in self.boq
            if b.confidence < 0.6 or b.qty_status == QtyStatus.UNKNOWN
        ]

    def get_missing_scope(self) -> List[ScopeItem]:
        """Get scope items marked as missing."""
        return [s for s in self.scope if s.status == ScopeStatus.MISSING]

    def to_json(self) -> str:
        """Export to JSON string."""
        return self.model_dump_json(indent=2)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: list,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_evidence(
    page: int,
    source: str,
    snippet: str = "",
    bbox: Optional[Tuple[float, float, float, float]] = None,
    table_id: Optional[str] = None
) -> Evidence:
    """Factory function to create Evidence with validation."""
    return Evidence(
        page=page,
        source=EvidenceSource(source),
        snippet=snippet[:500] if snippet else "",
        bbox=bbox,
        table_id=table_id
    )


def create_boq_item(
    item_name: str,
    system: str = "structural",
    subsystem: str = "rcc",
    unit: Optional[str] = None,
    qty: Optional[float] = None,
    evidence: Optional[List[Evidence]] = None,
    **kwargs
) -> BOQItem:
    """Factory function to create BOQItem with sensible defaults."""
    qty_status = QtyStatus.COMPUTED if qty is not None else QtyStatus.UNKNOWN

    return BOQItem(
        item_name=item_name,
        system=system,
        subsystem=subsystem,
        unit=unit,
        qty=qty,
        qty_status=qty_status,
        evidence=evidence or [],
        **kwargs
    )


def create_scope_item(
    category: str,
    trade: str,
    status: str = "detected",
    reason: str = "",
    evidence: Optional[List[Evidence]] = None,
    confidence: float = 0.7
) -> ScopeItem:
    """Factory function to create ScopeItem."""
    return ScopeItem(
        category=ScopeCategory(category),
        trade=trade,
        status=ScopeStatus(status),
        reason=reason,
        evidence=evidence or [],
        confidence=confidence
    )


def create_conflict(
    conflict_type: str,
    description: str,
    severity: str = "med",
    suggested_resolution: str = "",
    evidence: Optional[List[Evidence]] = None
) -> Conflict:
    """Factory function to create Conflict."""
    return Conflict(
        type=ConflictType(conflict_type),
        description=description,
        severity=Severity(severity),
        suggested_resolution=suggested_resolution,
        evidence=evidence or []
    )
