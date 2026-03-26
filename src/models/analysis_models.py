"""
Deep Analysis Models for XBOQ

Structured evidence and dependency tracking for auditable, deeper analysis.
Moves from "LLM surface scan" to "plan-set intelligence" with:
- Explicit evidence references (pages, sheets, detected entities)
- Dependency reasoning (what's missing, what it blocks)
- Impact scoring (cost, schedule, bid impact)
- BOQ skeleton mapping (what becomes priceable when fixed)

These models ensure every blocker/RFI can answer:
- What exactly did we detect? (entities + counts + tags)
- Where did we detect it? (pages/sheets)
- What did we search for and not find? (schedule/legend/spec)
- Why does it matter? (cost risk, schedule risk, trade impact)
- How can the user fix it? (missing dependency + acceptable substitutes)
- What becomes priceable if fixed? (BOQ skeleton mapping)
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Set
from enum import Enum
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class Severity(str, Enum):
    """Blocker severity levels."""
    CRITICAL = "critical"  # Blocks pricing entirely
    HIGH = "high"          # Major impact, must resolve
    MEDIUM = "medium"      # Should clarify
    LOW = "low"            # Nice to have


class BidImpact(str, Enum):
    """How the blocker affects bid submission."""
    BLOCKS_PRICING = "blocks_pricing"       # Cannot price without resolution
    FORCES_ALLOWANCE = "forces_allowance"   # Must add allowance/provisional
    CLARIFICATION_NEEDED = "clarification"  # Need RFI but can proceed
    INFORMATIONAL = "informational"         # FYI only


class RiskLevel(str, Enum):
    """Risk level for cost/schedule impact."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Trade(str, Enum):
    """Construction trades."""
    CIVIL = "civil"
    STRUCTURAL = "structural"
    ARCHITECTURAL = "architectural"
    MEP = "mep"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    FINISHES = "finishes"
    GENERAL = "general"
    COMMERCIAL = "commercial"


class SheetType(str, Enum):
    """Types of drawing sheets."""
    FLOOR_PLAN = "floor_plan"
    SITE_PLAN = "site_plan"
    STRUCTURAL = "structural"
    FOUNDATION = "foundation"
    SECTION = "section"
    ELEVATION = "elevation"
    DETAIL = "detail"
    SCHEDULE = "schedule"
    LEGEND = "legend"
    COVER = "cover"
    INDEX = "index"
    NOTES = "notes"
    MEP = "mep"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    UNKNOWN = "unknown"


class Discipline(str, Enum):
    """Drawing disciplines based on sheet prefix."""
    ARCHITECTURAL = "A"      # A-xxx sheets
    STRUCTURAL = "S"         # S-xxx sheets
    MECHANICAL = "M"         # M-xxx sheets
    ELECTRICAL = "E"         # E-xxx sheets
    PLUMBING = "P"           # P-xxx sheets
    CIVIL = "C"              # C-xxx sheets
    LANDSCAPE = "L"          # L-xxx sheets
    FIRE = "FP"              # FP-xxx sheets
    INTERIOR = "ID"          # ID-xxx sheets
    GENERAL = "G"            # G-xxx sheets
    UNKNOWN = "?"


class BOQItemStatus(str, Enum):
    """Status of a BOQ line item."""
    PRICEABLE = "priceable"           # Has all info to price
    BLOCKED = "blocked"               # Missing dependency
    ASSUMED = "assumed"               # Using assumption/allowance
    NEEDS_VERIFICATION = "needs_verification"


class ConfidenceLevel(str, Enum):
    """Confidence bands for findings."""
    HIGH = "high"       # >= 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5


class CoverageStatus(str, Enum):
    """Whether a check actually covered the relevant pages."""
    FOUND = "found"
    NOT_FOUND_AFTER_SEARCH = "not_found_after_search"
    UNKNOWN_NOT_PROCESSED = "unknown_not_processed"


class SelectionMode(str, Enum):
    """How pages were selected for deep processing."""
    FULL_READ = "full_read"
    FAST_BUDGET = "fast_budget"


class RunMode(str, Enum):
    """User-selectable run mode controlling deep-processing page budget.

    Sprint 20G: Allows users to choose how many pages are deep-processed.
    """
    DEMO_FAST = "demo_fast"              # deep_cap=80  (current default)
    STANDARD_REVIEW = "standard_review"  # deep_cap=220
    FULL_AUDIT = "full_audit"            # deep_cap=None (all pages)

    @property
    def deep_cap(self) -> int | None:
        """Return the page budget for this mode (None = unlimited)."""
        return _RUN_MODE_CAPS[self]

    @property
    def label(self) -> str:
        return _RUN_MODE_LABELS[self]


_RUN_MODE_CAPS: dict["RunMode", int | None] = {
    RunMode.DEMO_FAST: 80,
    RunMode.STANDARD_REVIEW: 220,
    RunMode.FULL_AUDIT: None,
}

_RUN_MODE_LABELS: dict["RunMode", str] = {
    RunMode.DEMO_FAST: "Demo Fast (80 pages)",
    RunMode.STANDARD_REVIEW: "Standard Review (220 pages)",
    RunMode.FULL_AUDIT: "Full Audit (All pages)",
}


class RunCoverage(BaseModel):
    """
    Global coverage snapshot for the entire analysis run.

    Built in pipeline.py after extraction, passed to dependency_reasoner
    and rfi_engine so checks can gate "missing" assertions on whether
    the relevant pages were actually deep-processed.
    """
    pages_total: int = 0
    pages_indexed: int = 0
    pages_deep_processed: int = 0
    pages_skipped: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Each entry: {page_idx, doc_type, discipline, reason}"
    )

    doc_types_detected: Dict[str, int] = Field(default_factory=dict)
    disciplines_detected: Dict[str, int] = Field(default_factory=dict)

    doc_types_fully_covered: List[str] = Field(
        default_factory=list,
        description="Every page of this type was deep-processed"
    )
    doc_types_partially_covered: List[str] = Field(
        default_factory=list,
        description="Some pages of this type were deep-processed"
    )
    doc_types_not_covered: List[str] = Field(
        default_factory=list,
        description="Zero pages of this type were deep-processed"
    )

    selection_mode: SelectionMode = SelectionMode.FAST_BUDGET
    ocr_budget_pages: int = 80
    coverage_by_check: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def is_doc_type_covered(self, doc_type: str) -> CoverageStatus:
        """Check if a doc_type was fully covered by deep processing."""
        if doc_type in self.doc_types_fully_covered:
            return CoverageStatus.NOT_FOUND_AFTER_SEARCH
        # If doc_type was never detected in indexing, it genuinely doesn't exist
        if self.doc_types_detected.get(doc_type, 0) == 0:
            return CoverageStatus.NOT_FOUND_AFTER_SEARCH
        # Partially or not covered
        return CoverageStatus.UNKNOWN_NOT_PROCESSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages_total": self.pages_total,
            "pages_indexed": self.pages_indexed,
            "pages_deep_processed": self.pages_deep_processed,
            "pages_skipped": self.pages_skipped,
            "doc_types_detected": self.doc_types_detected,
            "disciplines_detected": self.disciplines_detected,
            "doc_types_fully_covered": self.doc_types_fully_covered,
            "doc_types_partially_covered": self.doc_types_partially_covered,
            "doc_types_not_covered": self.doc_types_not_covered,
            "selection_mode": self.selection_mode.value,
            "ocr_budget_pages": self.ocr_budget_pages,
            "coverage_by_check": self.coverage_by_check,
        }


# =============================================================================
# EVIDENCE MODEL (Core - used by all findings)
# =============================================================================

class EvidenceRef(BaseModel):
    """
    Structured evidence reference for any finding.

    Every blocker/RFI must have evidence that shows:
    - Where it was found (pages/sheets)
    - What was detected (entities, counts)
    - What was searched for (search attempts)
    """
    # Location
    pages: List[int] = Field(default_factory=list, description="0-indexed page numbers")
    sheets: List[str] = Field(default_factory=list, description="Sheet numbers like 'A-101'")

    # Detected content
    snippets: List[str] = Field(
        default_factory=list,
        description="Text excerpts (max 200 chars each)"
    )
    detected_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured entities: door_tags, window_tags, room_names, callouts, etc."
    )

    # Search audit trail
    search_attempts: Dict[str, Any] = Field(
        default_factory=dict,
        description="What we searched for: keywords checked, patterns matched, table scans"
    )

    # Confidence
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score 0-1"
    )
    confidence_reason: str = Field(
        default="",
        description="Why this confidence level"
    )

    # Spatial evidence — per-page bounding boxes with confidence + bbox_id
    bbox: Optional[List[List[List[Any]]]] = Field(
        default=None,
        description=(
            "Per-evidence-page bounding boxes. Outer list parallels self.pages. "
            "Each inner list: [[x0_rel, y0_rel, x1_rel, y1_rel, confidence, bbox_id], ...] "
            "where coords are 0.0-1.0 page-relative, confidence is 0.0-1.0, "
            "and bbox_id is an optional string like 'BLK-0010-P0-0'."
        ),
    )

    def assign_bbox_ids(self, item_id: str) -> None:
        """Assign bbox_id (6th element) to every box in self.bbox.

        Convention: '{item_id}-P{page_pos}-{box_idx}'
        e.g. 'BLK-0010-P0-0', 'BLK-0010-P1-1'.
        """
        if not self.bbox:
            return
        for page_pos, page_boxes in enumerate(self.bbox):
            for box_idx, box in enumerate(page_boxes):
                bbox_id = f"{item_id}-P{page_pos}-{box_idx}"
                if len(box) >= 6:
                    box[5] = bbox_id  # overwrite existing
                elif len(box) >= 5:
                    box.append(bbox_id)  # append 6th element

    # NOT_FOUND proof
    searched_pages: Optional[List[int]] = Field(
        default=None,
        description="0-indexed pages that were actually searched for this item"
    )
    text_coverage_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Pct of text on searched pages that was actually analyzed"
    )

    @field_validator('snippets')
    @classmethod
    def truncate_snippets(cls, v):
        """Ensure snippets are max 200 chars."""
        return [s[:200] for s in v]

    @property
    def confidence_band(self) -> ConfidenceLevel:
        """Get confidence as a band."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @property
    def has_evidence(self) -> bool:
        """Check if any evidence exists."""
        return bool(self.pages or self.sheets or self.detected_entities or self.search_attempts)

    def summary(self) -> str:
        """Human-readable evidence summary."""
        parts = []
        if self.pages:
            pages_str = ", ".join(str(p+1) for p in self.pages[:5])  # 1-indexed for display
            if len(self.pages) > 5:
                pages_str += f" (+{len(self.pages)-5} more)"
            parts.append(f"Pages: {pages_str}")
        if self.detected_entities:
            for key, val in self.detected_entities.items():
                if isinstance(val, list) and val:
                    parts.append(f"{key}: {len(val)} found")
                elif isinstance(val, (int, float)):
                    parts.append(f"{key}: {val}")
        if self.search_attempts:
            searched = list(self.search_attempts.keys())[:3]
            parts.append(f"Searched: {', '.join(searched)}")
        if self.searched_pages:
            parts.append(f"Searched {len(self.searched_pages)} pages")
        return " | ".join(parts) if parts else "No evidence"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "pages": self.pages,
            "sheets": self.sheets,
            "snippets": self.snippets,
            "detected_entities": self.detected_entities,
            "search_attempts": self.search_attempts,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "bbox": self.bbox,
            "searched_pages": self.searched_pages,
            "text_coverage_pct": self.text_coverage_pct,
        }


# =============================================================================
# BLOCKER MODEL (Deep blockers with impact analysis)
# =============================================================================

class Blocker(BaseModel):
    """
    A bid blocker with full evidence and impact analysis.

    Every blocker must answer:
    - What's the issue? (title, description)
    - What's missing? (missing_dependency)
    - What's the evidence? (evidence)
    - What's the impact? (cost, schedule, bid)
    - How to fix? (fix_actions)
    - What becomes priceable? (unlocks_boq_categories)

    Blockers are now TRADE/CATEGORY-SPECIFIC:
    - affected_trades: List of trades this blocker impacts (not global)
    - A blocker only affects coverage/pricing for its affected_trades
    - Global NO-GO only happens when critical trades have no coverage
    """
    # Identity
    id: str = Field(description="Unique blocker ID like 'BLK-0001'")
    title: str = Field(description="Short title")
    trade: Trade = Field(default=Trade.GENERAL)
    severity: Severity = Field(default=Severity.MEDIUM)

    # Affected trades (NEW: for granular impact tracking)
    affected_trades: List[Trade] = Field(
        default_factory=list,
        description="Specific trades affected by this blocker. Empty = affects only primary trade."
    )

    # Description
    description: str = Field(description="Detailed description of the issue")

    # Dependencies
    missing_dependency: List[str] = Field(
        default_factory=list,
        description="What's missing: door_schedule, window_schedule, finish_schedule, mep_drawings, etc."
    )

    # Impact analysis
    impact_cost: RiskLevel = Field(default=RiskLevel.MEDIUM)
    impact_schedule: RiskLevel = Field(default=RiskLevel.LOW)
    bid_impact: BidImpact = Field(default=BidImpact.CLARIFICATION_NEEDED)

    # Evidence (required)
    evidence: EvidenceRef = Field(default_factory=EvidenceRef)

    # Resolution
    fix_actions: List[str] = Field(
        default_factory=list,
        description="Ordered list of fix options"
    )
    score_delta_estimate: int = Field(
        default=0,
        description="Estimated score improvement if fixed"
    )

    # BOQ mapping
    unlocks_boq_categories: List[str] = Field(
        default_factory=list,
        description="BOQ categories that become priceable if fixed"
    )

    # Metadata
    issue_type: str = Field(default="", description="Category: missing_schedule, missing_drawing, scale_issue, etc.")
    coverage_status: Optional[str] = Field(default=None, description="CoverageStatus value: found, not_found_after_search, unknown_not_processed")
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_critical(self) -> bool:
        return self.severity == Severity.CRITICAL

    @property
    def blocks_pricing(self) -> bool:
        return self.bid_impact == BidImpact.BLOCKS_PRICING

    @property
    def all_affected_trades(self) -> List[Trade]:
        """Get all trades affected by this blocker (primary + explicit)."""
        trades = set([self.trade])
        trades.update(self.affected_trades)
        return list(trades)

    @property
    def is_global_blocker(self) -> bool:
        """Check if this blocker affects all trades (like missing scale)."""
        # Scale blockers and certain critical issues affect all trades
        return (
            'all_measured_items' in self.unlocks_boq_categories or
            self.trade == Trade.GENERAL and self.bid_impact == BidImpact.BLOCKS_PRICING
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON export."""
        return {
            "id": self.id,
            "title": self.title,
            "trade": self.trade.value,
            "severity": self.severity.value,
            "affected_trades": [t.value for t in self.affected_trades],
            "description": self.description,
            "missing_dependency": self.missing_dependency,
            "impact_cost": self.impact_cost.value,
            "impact_schedule": self.impact_schedule.value,
            "bid_impact": self.bid_impact.value,
            "evidence": self.evidence.to_dict(),
            "fix_actions": self.fix_actions,
            "score_delta_estimate": self.score_delta_estimate,
            "unlocks_boq_categories": self.unlocks_boq_categories,
            "issue_type": self.issue_type,
            "coverage_status": self.coverage_status,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# RFI MODEL (Structured RFI with evidence)
# =============================================================================

class RFIItem(BaseModel):
    """
    Request for Information with structured evidence.

    Each RFI is linked to evidence and explains:
    - What's the question?
    - Why does it matter?
    - What evidence supports this?
    - What's the suggested resolution?
    """
    id: str = Field(description="Unique RFI ID like 'RFI-0001'")
    trade: Trade = Field(default=Trade.GENERAL)
    priority: Severity = Field(default=Severity.MEDIUM)

    # Content
    question: str = Field(description="The RFI question")
    why_it_matters: str = Field(description="Impact if not resolved")

    # Evidence
    evidence: EvidenceRef = Field(default_factory=EvidenceRef)

    # Resolution
    suggested_resolution: str = Field(default="")
    acceptable_alternatives: List[str] = Field(default_factory=list)

    # Linking
    related_blocker_id: Optional[str] = Field(default=None)

    # Metadata
    issue_type: str = Field(default="")
    package: str = Field(default="")
    coverage_status: Optional[str] = Field(default=None, description="CoverageStatus value")
    created_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trade": self.trade.value,
            "priority": self.priority.value,
            "question": self.question,
            "why_it_matters": self.why_it_matters,
            "evidence": self.evidence.to_dict(),
            "suggested_resolution": self.suggested_resolution,
            "acceptable_alternatives": self.acceptable_alternatives,
            "related_blocker_id": self.related_blocker_id,
            "issue_type": self.issue_type,
            "package": self.package,
            "coverage_status": self.coverage_status,
            "created_at": self.created_at.isoformat(),
        }

    def to_email_text(self) -> str:
        """Format for email."""
        return f"""RFI: {self.question}
Priority: {self.priority.value.upper()}

Issue:
{self.why_it_matters}

Evidence:
- Pages: {', '.join(str(p+1) for p in self.evidence.pages) or 'N/A'}
- Items detected: {', '.join(str(v) for v in list(self.evidence.detected_entities.values())[:3]) if self.evidence.detected_entities else 'N/A'}

Suggested Resolution:
{self.suggested_resolution}
"""


# =============================================================================
# TRADE COVERAGE MODEL
# =============================================================================

class TradeCoverage(BaseModel):
    """
    Trade-level coverage analysis.

    Shows what's priceable vs blocked for each trade.
    """
    trade: Trade
    coverage_pct: float = Field(ge=0.0, le=100.0, description="Percentage of scope covered")

    # Counts
    total_categories: int = Field(default=0)
    priceable_count: int = Field(default=0)
    blocked_count: int = Field(default=0)
    assumed_count: int = Field(default=0)

    # Dependencies
    missing_dependencies: List[str] = Field(default_factory=list)

    # Risk
    cost_risk: RiskLevel = Field(default=RiskLevel.MEDIUM)
    schedule_risk: RiskLevel = Field(default=RiskLevel.LOW)

    # Actions
    next_action: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade": self.trade.value,
            "coverage_pct": self.coverage_pct,
            "total_categories": self.total_categories,
            "priceable_count": self.priceable_count,
            "blocked_count": self.blocked_count,
            "assumed_count": self.assumed_count,
            "missing_dependencies": self.missing_dependencies,
            "cost_risk": self.cost_risk.value,
            "schedule_risk": self.schedule_risk.value,
            "next_action": self.next_action,
        }


# =============================================================================
# BOQ SKELETON ITEM
# =============================================================================

class BOQSkeletonItem(BaseModel):
    """
    A BOQ skeleton item showing what's priceable.

    Maps blockers to BOQ line items.
    """
    trade: Trade
    category: str = Field(description="BOQ category like 'doors', 'flooring'")
    item_name: str

    # Status
    status: BOQItemStatus = Field(default=BOQItemStatus.BLOCKED)
    blocked_by: List[str] = Field(default_factory=list, description="Blocker IDs")

    # Evidence
    evidence: EvidenceRef = Field(default_factory=EvidenceRef)

    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade": self.trade.value,
            "category": self.category,
            "item_name": self.item_name,
            "status": self.status.value,
            "blocked_by": self.blocked_by,
            "evidence": self.evidence.to_dict(),
            "confidence": self.confidence,
        }


# =============================================================================
# PLAN SHEET MODEL (For Plan Graph)
# =============================================================================

class SheetReference(BaseModel):
    """Reference to another sheet (section callout, detail marker, etc.)."""
    ref_type: str = Field(description="section, detail, elevation, plan")
    ref_id: str = Field(description="The reference like 'A-A', '1/A-5.01'")
    target_sheet: Optional[str] = Field(default=None)


class PlanSheet(BaseModel):
    """
    A single sheet in the plan set graph.

    Captures structure: sheet number, discipline, type, references, detected entities.
    """
    # Identity
    page_index: int = Field(description="0-indexed page number in PDF")
    sheet_no: Optional[str] = Field(default=None, description="Sheet number like 'A-101'")
    title: Optional[str] = Field(default=None, description="Sheet title")

    # Classification
    discipline: Discipline = Field(default=Discipline.UNKNOWN)
    sheet_type: SheetType = Field(default=SheetType.UNKNOWN)

    # References to other sheets
    references: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Outgoing references: sections, elevations, details"
    )

    # Detected entities on this sheet
    detected: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detected items: door_tags, window_tags, room_names, has_scale, has_legend"
    )

    # Text content summary
    text_preview: str = Field(default="", description="First 500 chars of extracted text")

    # Confidence
    classification_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "sheet_no": self.sheet_no,
            "title": self.title,
            "discipline": self.discipline.value,
            "sheet_type": self.sheet_type.value,
            "references": self.references,
            "detected": self.detected,
            "text_preview": self.text_preview[:500],
            "classification_confidence": self.classification_confidence,
        }


# =============================================================================
# PLAN SET GRAPH
# =============================================================================

class PlanSetGraph(BaseModel):
    """
    The complete plan set structure.

    A graph of all sheets with their classifications, references, and detected entities.
    Used for dependency reasoning.
    """
    project_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Sheets
    sheets: List[PlanSheet] = Field(default_factory=list)

    # Aggregates
    total_pages: int = Field(default=0)
    disciplines_found: List[str] = Field(default_factory=list)
    sheet_types_found: Dict[str, int] = Field(default_factory=dict)

    # Entity aggregates
    all_door_tags: List[str] = Field(default_factory=list)
    all_window_tags: List[str] = Field(default_factory=list)
    all_room_names: List[str] = Field(default_factory=list)

    # Schedule detection
    has_door_schedule: bool = Field(default=False)
    has_window_schedule: bool = Field(default=False)
    has_finish_schedule: bool = Field(default=False)
    has_legend: bool = Field(default=False)

    # Scale status
    pages_with_scale: int = Field(default=0)
    pages_without_scale: int = Field(default=0)

    # Sheet-number coverage (pages that have a proper engineering sheet number A-101 etc.)
    # Used to distinguish drawing sets (high ratio) from spec documents (low ratio).
    pages_with_sheet_no: int = Field(default=0)

    # Package classification (set by pipeline after classify_package())
    package_type: Optional[str] = Field(
        default=None,
        description="PackageType value: 'drawing_set', 'tender', 'mixed', 'incomplete'",
    )
    package_type_confidence: float = Field(default=0.0)
    boq_trades_detected: List[str] = Field(
        default_factory=list,
        description="Trades inferred from BOQ text keywords e.g. ['civil', 'general']",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "sheets": [s.to_dict() for s in self.sheets],
            "total_pages": self.total_pages,
            "disciplines_found": self.disciplines_found,
            "sheet_types_found": self.sheet_types_found,
            "all_door_tags": self.all_door_tags,
            "all_window_tags": self.all_window_tags,
            "all_room_names": self.all_room_names,
            "has_door_schedule": self.has_door_schedule,
            "has_window_schedule": self.has_window_schedule,
            "has_finish_schedule": self.has_finish_schedule,
            "has_legend": self.has_legend,
            "pages_with_scale": self.pages_with_scale,
            "pages_without_scale": self.pages_without_scale,
            "pages_with_sheet_no": self.pages_with_sheet_no,
            "package_type": self.package_type,
            "package_type_confidence": self.package_type_confidence,
            "boq_trades_detected": self.boq_trades_detected,
        }


# =============================================================================
# READINESS SCORE (Multi-component)
# =============================================================================

class ReadinessScore(BaseModel):
    """
    Multi-component readiness score.

    Replaces single score with weighted components for deeper insight.
    """
    # Overall
    total_score: int = Field(ge=0, le=100)
    status: str = Field(description="GO, NO-GO, REVIEW")

    # Component scores (0-100 each)
    completeness_score: int = Field(ge=0, le=100, description="Missing dependencies score")
    measurement_score: int = Field(ge=0, le=100, description="Scale/geometry reliability")
    coverage_score: int = Field(ge=0, le=100, description="Trade coverage score")
    blocker_score: int = Field(ge=0, le=100, description="Inverse of critical blockers")

    # Weights (should sum to 1.0)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "completeness": 0.30,
            "measurement": 0.25,
            "coverage": 0.25,
            "blocker": 0.20,
        }
    )

    # Explanation
    score_breakdown: Dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "status": self.status,
            "completeness_score": self.completeness_score,
            "measurement_score": self.measurement_score,
            "coverage_score": self.coverage_score,
            "blocker_score": self.blocker_score,
            "weights": self.weights,
            "score_breakdown": self.score_breakdown,
        }


# =============================================================================
# DEEP ANALYSIS RESULT (Complete output)
# =============================================================================

class DeepAnalysisResult(BaseModel):
    """
    Complete deep analysis output.

    Contains all structured findings with evidence.
    """
    project_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Plan structure
    plan_graph: Optional[PlanSetGraph] = None

    # Findings
    blockers: List[Blocker] = Field(default_factory=list)
    rfis: List[RFIItem] = Field(default_factory=list)

    # Coverage
    trade_coverage: List[TradeCoverage] = Field(default_factory=list)
    boq_skeleton: List[BOQSkeletonItem] = Field(default_factory=list)

    # Score
    readiness_score: Optional[ReadinessScore] = None

    # Aggregates
    total_blockers: int = Field(default=0)
    critical_blockers: int = Field(default=0)
    total_rfis: int = Field(default=0)

    # Metadata
    analysis_version: str = Field(default="2.0.0")

    def compute_aggregates(self):
        """Recompute aggregates from findings."""
        self.total_blockers = len(self.blockers)
        self.critical_blockers = sum(1 for b in self.blockers if b.is_critical)
        self.total_rfis = len(self.rfis)

    def to_dict(self) -> Dict[str, Any]:
        self.compute_aggregates()
        return {
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "plan_graph": self.plan_graph.to_dict() if self.plan_graph else None,
            "blockers": [b.to_dict() for b in self.blockers],
            "rfis": [r.to_dict() for r in self.rfis],
            "trade_coverage": [t.to_dict() for t in self.trade_coverage],
            "boq_skeleton": [b.to_dict() for b in self.boq_skeleton],
            "readiness_score": self.readiness_score.to_dict() if self.readiness_score else None,
            "total_blockers": self.total_blockers,
            "critical_blockers": self.critical_blockers,
            "total_rfis": self.total_rfis,
            "analysis_version": self.analysis_version,
        }


# =============================================================================
# EVALUATION LOG (For continuous improvement)
# =============================================================================

class EvaluationLog(BaseModel):
    """
    Evaluation log for tracking analysis quality over time.

    Captures metrics and user feedback for iteration.
    """
    project_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Detection metrics
    detected_sheets_count: int = Field(default=0)
    schedules_detected_count: int = Field(default=0)
    scale_missing_pages_count: int = Field(default=0)

    # Finding counts
    blockers_by_type: Dict[str, int] = Field(default_factory=dict)
    rfis_by_trade: Dict[str, int] = Field(default_factory=dict)
    coverage_by_trade: Dict[str, float] = Field(default_factory=dict)

    # Scores
    final_score: int = Field(default=0)
    component_scores: Dict[str, int] = Field(default_factory=dict)

    # User feedback (to be populated by UI)
    user_marked_false_positive: List[str] = Field(default_factory=list)
    user_marked_resolved: List[str] = Field(default_factory=list)
    user_corrections: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "detected_sheets_count": self.detected_sheets_count,
            "schedules_detected_count": self.schedules_detected_count,
            "scale_missing_pages_count": self.scale_missing_pages_count,
            "blockers_by_type": self.blockers_by_type,
            "rfis_by_trade": self.rfis_by_trade,
            "coverage_by_trade": self.coverage_by_trade,
            "final_score": self.final_score,
            "component_scores": self.component_scores,
            "user_marked_false_positive": self.user_marked_false_positive,
            "user_marked_resolved": self.user_marked_resolved,
            "user_corrections": self.user_corrections,
        }


# =============================================================================
# UNIFIED LINE ITEM  (Sprint 21 — 100% extraction feature)
# =============================================================================

class UnifiedLineItem(BaseModel):
    """
    A normalised, trade-tagged line item assembled from:
      - BOQ items            (source="boq")
      - Spec/notes clauses   (source="spec_item")
      - Schedule stubs       (source="schedule_stub")

    All items run through taxonomy matching; the result is the canonical
    line-items list used for pricing, coverage scoring, and export.
    """
    id: str                                   # "LI-0001", "LI-0002", …
    source: str                               # "boq" | "spec_item" | "schedule_stub"
    item_no: Optional[str]  = None

    @field_validator("item_no", mode="before")
    @classmethod
    def _coerce_item_no_to_str(cls, v):
        # BUG-14 FIX: BOQ parsers sometimes pass integer item numbers (1, 2, 3).
        # Coerce any non-None value to str so Pydantic validation passes.
        if v is None:
            return v
        return str(v) if not isinstance(v, str) else v
    description: str        = ""
    unit: Optional[str]     = None
    unit_family: Optional[str] = None         # AREA|VOLUME|LINEAR|COUNT|WEIGHT|LUMP
    qty: Optional[float]    = None
    rate: Optional[float]   = None
    trade: str              = "general"
    section: Optional[str]  = None
    source_page: int        = 0

    # Taxonomy match results
    taxonomy_id: Optional[str]          = None
    taxonomy_discipline: Optional[str]  = None
    taxonomy_unit: Optional[str]        = None
    match_confidence: float             = 0.0
    match_method: str                   = ""
    taxonomy_matched: bool              = False

    # Quality flags
    unit_inferred: bool  = False
    qty_missing: bool    = False
    rate_missing: bool   = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                   self.id,
            "source":               self.source,
            "item_no":              self.item_no,
            "description":          self.description,
            "unit":                 self.unit,
            "unit_family":          self.unit_family,
            "qty":                  self.qty,
            "rate":                 self.rate,
            "trade":                self.trade,
            "section":              self.section,
            "source_page":          self.source_page,
            "taxonomy_id":          self.taxonomy_id,
            "taxonomy_discipline":  self.taxonomy_discipline,
            "taxonomy_unit":        self.taxonomy_unit,
            "match_confidence":     self.match_confidence,
            "match_method":         self.match_method,
            "taxonomy_matched":     self.taxonomy_matched,
            "unit_inferred":        self.unit_inferred,
            "qty_missing":          self.qty_missing,
            "rate_missing":         self.rate_missing,
        }


# =============================================================================
# BOQ SKELETON TEMPLATE
# =============================================================================

# Default BOQ skeleton by trade (8-15 categories each)
BOQ_SKELETON_TEMPLATE: Dict[str, List[str]] = {
    "civil": [
        "earthwork_excavation",
        "pcc_below_footings",
        "backfilling",
        "sand_filling",
        "waterproofing_foundation",
        "site_drainage",
        "compound_wall",
        "external_paving",
    ],
    "structural": [
        "rcc_footings",
        "rcc_columns",
        "rcc_beams",
        "rcc_slabs",
        "rcc_staircase",
        "rcc_chajja",
        "reinforcement_steel",
        "formwork",
        "structural_steel",
    ],
    "architectural": [
        "brick_masonry",
        "block_masonry",
        "internal_plaster",
        "external_plaster",
        "doors",
        "windows",
        "ventilators",
        "railing_balustrade",
        "false_ceiling",
    ],
    "mep": [
        "electrical_wiring",
        "electrical_fixtures",
        "plumbing_supply",
        "plumbing_drainage",
        "sanitary_fixtures",
        "fire_fighting",
        "hvac",
        "elevator",
    ],
    "finishes": [
        "wall_painting",
        "ceiling_painting",
        "wall_tiling",
        "floor_tiling",
        "dado_tiling",
        "skirting",
        "polishing",
        "texture_coating",
        "waterproofing_wet_areas",
    ],
    "general": [
        "scaffolding",
        "water_curing",
        "site_clearance",
        "temporary_works",
        "testing_commissioning",
    ],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_blocker_id(index: int) -> str:
    """Generate a blocker ID."""
    return f"BLK-{index:04d}"


def create_rfi_id(index: int) -> str:
    """Generate an RFI ID."""
    return f"RFI-{index:04d}"


def assign_all_bbox_ids(blockers: List[Dict], rfis: List[Dict]) -> None:
    """Walk all blockers and RFIs (as dicts) and assign bbox_ids in-place.

    Each bbox entry gets a 6th element: '{item_id}-P{page_pos}-{box_idx}'.
    Works on serialized dicts (post to_dict()), not Pydantic models.
    """
    for item in blockers:
        item_id = item.get("id", "UNK")
        ev = item.get("evidence", {})
        bbox = ev.get("bbox")
        if not bbox:
            continue
        for page_pos, page_boxes in enumerate(bbox):
            for box_idx, box in enumerate(page_boxes):
                bbox_id = f"{item_id}-P{page_pos}-{box_idx}"
                if len(box) >= 6:
                    box[5] = bbox_id
                elif len(box) >= 5:
                    box.append(bbox_id)

    for item in rfis:
        item_id = item.get("id", "UNK")
        ev = item.get("evidence", {})
        bbox = ev.get("bbox")
        if not bbox:
            continue
        for page_pos, page_boxes in enumerate(bbox):
            for box_idx, box in enumerate(page_boxes):
                bbox_id = f"{item_id}-P{page_pos}-{box_idx}"
                if len(box) >= 6:
                    box[5] = bbox_id
                elif len(box) >= 5:
                    box.append(bbox_id)


def severity_from_confidence(confidence: float) -> Severity:
    """Downgrade severity if low confidence."""
    if confidence < 0.5:
        return Severity.LOW
    elif confidence < 0.7:
        return Severity.MEDIUM
    return Severity.HIGH
