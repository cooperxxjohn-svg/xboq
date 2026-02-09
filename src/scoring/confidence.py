"""
XBOQ Confidence Scoring Module
Rule-based confidence computation with explainable factors.

NO magic numbers without reason - every factor is documented and deterministic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# BASE SCORES BY STATUS
# =============================================================================

CONFIDENCE_BASE_SCORES = {
    "detected": 0.85,      # Found with direct evidence
    "inferred": 0.65,      # Logically implied from other data
    "missing": 0.20,       # Expected but not found
    "needs_review": 0.45,  # Ambiguous or conflicting
    "computed": 0.85,      # Quantity computed
    "partial": 0.60,       # Partial quantity
    "unknown": 0.35,       # Cannot compute
}

# Evidence source weights (more diverse = better)
EVIDENCE_SOURCE_WEIGHTS = {
    "camelot": 0.06,      # Schedule table extraction
    "pdfplumber": 0.05,   # Alternative table extraction
    "pdf_text": 0.04,     # Vector text from PDF
    "ocr": 0.03,          # OCR text (less reliable)
    "heuristic": 0.05,    # Visual/pattern detection
    "user_input": 0.08,   # User confirmed
}

# Required fields for different item types
REQUIRED_FIELDS = {
    "column": ["scale", "storey_height", "column_marks_count"],
    "footing": ["scale", "footing_marks_count"],
    "beam": ["scale", "beam_schedule"],
    "slab": ["scale", "slab_thickness"],
    "rcc_general": ["concrete_grade", "steel_grade"],
    "reinforcement": ["bar_schedule", "storey_height"],
    "earthwork": ["footing_depths"],
    "masonry": ["wall_layout"],
    "default": ["scale"],
}


@dataclass
class ConfidenceResult:
    """Result of confidence computation with full breakdown."""
    score: float
    base_score: float
    factors: List[str] = field(default_factory=list)
    adjustments: Dict[str, float] = field(default_factory=dict)

    @property
    def reason(self) -> str:
        """Human-readable explanation of confidence score."""
        parts = [f"base={self.base_score:.2f}"]
        for name, adj in self.adjustments.items():
            sign = "+" if adj >= 0 else ""
            parts.append(f"{name}={sign}{adj:.2f}")
        parts.append(f"→ final={self.score:.2f}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "base_score": self.base_score,
            "factors": self.factors,
            "adjustments": self.adjustments,
            "reason": self.reason,
        }


def compute_confidence(
    item_type: str,
    status: str,
    evidence_sources: Optional[Set[str]] = None,
    evidence_count: int = 0,
    required_fields_present: Optional[Dict[str, bool]] = None,
    conflicts_count: int = 0,
    qty_status: Optional[str] = None,
    has_schedule_data: bool = False,
    has_plan_data: bool = False,
) -> ConfidenceResult:
    """
    Compute confidence score with explainable factors.

    Args:
        item_type: Type of item (column, footing, beam, reinforcement, etc.)
        status: Status (detected, inferred, missing, needs_review)
        evidence_sources: Set of evidence sources used
        evidence_count: Number of evidence items
        required_fields_present: Dict of required field -> present bool
        conflicts_count: Number of related conflicts
        qty_status: Quantity status (computed, partial, unknown)
        has_schedule_data: Whether schedule table data exists
        has_plan_data: Whether plan/layout data exists

    Returns:
        ConfidenceResult with score and breakdown
    """
    evidence_sources = evidence_sources or set()
    required_fields_present = required_fields_present or {}

    factors = []
    adjustments = {}

    # 1. Base score from status
    base_score = CONFIDENCE_BASE_SCORES.get(status.lower(), 0.50)
    factors.append(f"status={status}")

    # 2. Evidence source diversity bonus (up to +0.15)
    evidence_bonus = 0.0
    if evidence_sources:
        for source in evidence_sources:
            source_weight = EVIDENCE_SOURCE_WEIGHTS.get(source.lower(), 0.02)
            evidence_bonus += source_weight
        evidence_bonus = min(evidence_bonus, 0.15)  # Cap at +0.15

        if evidence_bonus > 0:
            adjustments["evidence_sources"] = evidence_bonus
            factors.append(f"{len(evidence_sources)} evidence sources")

    # 3. Evidence count bonus (diminishing returns)
    if evidence_count > 0:
        count_bonus = min(evidence_count * 0.02, 0.08)  # Up to +0.08
        adjustments["evidence_count"] = count_bonus
        factors.append(f"{evidence_count} evidence items")

    # 4. Required fields check
    item_required = REQUIRED_FIELDS.get(item_type.lower(), REQUIRED_FIELDS["default"])
    fields_present = sum(1 for f in item_required if required_fields_present.get(f, False))
    fields_total = len(item_required)

    if fields_total > 0:
        field_ratio = fields_present / fields_total
        if field_ratio >= 0.8:
            adjustments["required_fields"] = 0.10
            factors.append("required fields present")
        elif field_ratio >= 0.5:
            adjustments["required_fields"] = 0.05
            factors.append("some required fields")
        elif field_ratio < 0.3:
            adjustments["missing_fields"] = -0.15
            factors.append("key fields missing")

    # 5. Conflicts penalty
    if conflicts_count >= 3:
        adjustments["high_conflicts"] = -0.20
        factors.append(f"{conflicts_count} conflicts")
    elif conflicts_count >= 1:
        adjustments["conflicts"] = -0.10
        factors.append(f"{conflicts_count} conflict(s)")

    # 6. Schedule data bonus
    if has_schedule_data:
        adjustments["schedule_data"] = 0.08
        factors.append("schedule data")

    # 7. Plan data bonus
    if has_plan_data:
        adjustments["plan_data"] = 0.05
        factors.append("plan data")

    # 8. Quantity status adjustment
    if qty_status:
        if qty_status.lower() == "computed":
            adjustments["qty_computed"] = 0.10
            factors.append("qty computed")
        elif qty_status.lower() == "partial":
            adjustments["qty_partial"] = 0.05
            factors.append("qty partial")
        elif qty_status.lower() == "unknown":
            adjustments["qty_unknown"] = -0.10
            factors.append("qty unknown")

    # 9. Inferred without evidence penalty
    if status.lower() == "inferred" and evidence_count == 0:
        adjustments["inferred_no_evidence"] = -0.15
        factors.append("inferred without evidence")

    # Calculate final score
    final_score = base_score + sum(adjustments.values())

    # Clamp to valid range
    final_score = max(0.05, min(0.98, final_score))

    return ConfidenceResult(
        score=round(final_score, 3),
        base_score=base_score,
        factors=factors,
        adjustments=adjustments,
    )


def compute_boq_confidence(
    boq_item: Any,
    conflicts_count: int = 0,
    required_fields: Optional[Dict[str, bool]] = None,
) -> ConfidenceResult:
    """
    Compute confidence for a BOQItem.

    Args:
        boq_item: BOQItem object
        conflicts_count: Number of related conflicts
        required_fields: Dict of required field -> present

    Returns:
        ConfidenceResult
    """
    # Extract evidence sources
    evidence_sources = set()
    evidence_count = 0

    if hasattr(boq_item, 'evidence') and boq_item.evidence:
        evidence_count = len(boq_item.evidence)
        for ev in boq_item.evidence:
            if hasattr(ev, 'source'):
                source = ev.source.value if hasattr(ev.source, 'value') else str(ev.source)
                evidence_sources.add(source.lower())

    # Determine item type from subsystem
    item_type = "default"
    if hasattr(boq_item, 'subsystem'):
        subsystem = boq_item.subsystem.lower()
        if "column" in boq_item.item_name.lower():
            item_type = "column"
        elif "footing" in boq_item.item_name.lower():
            item_type = "footing"
        elif "beam" in boq_item.item_name.lower():
            item_type = "beam"
        elif "slab" in boq_item.item_name.lower():
            item_type = "slab"
        elif "steel" in boq_item.item_name.lower() or "reinf" in boq_item.item_name.lower():
            item_type = "reinforcement"
        elif subsystem == "earthwork":
            item_type = "earthwork"
        elif subsystem == "masonry":
            item_type = "masonry"

    # Get status
    source = getattr(boq_item, 'source', 'explicit')
    if source == 'inferred':
        status = 'inferred'
    elif source == 'synonym':
        status = 'detected'
    else:
        status = 'detected'

    # Check qty_status
    qty_status = None
    if hasattr(boq_item, 'qty_status'):
        qty_status = boq_item.qty_status.value if hasattr(boq_item.qty_status, 'value') else str(boq_item.qty_status)

    return compute_confidence(
        item_type=item_type,
        status=status,
        evidence_sources=evidence_sources,
        evidence_count=evidence_count,
        required_fields_present=required_fields or {},
        conflicts_count=conflicts_count,
        qty_status=qty_status,
        has_schedule_data="camelot" in evidence_sources or "pdfplumber" in evidence_sources,
        has_plan_data="heuristic" in evidence_sources,
    )


def compute_scope_confidence(
    scope_item: Any,
    conflicts_count: int = 0,
    required_fields: Optional[Dict[str, bool]] = None,
) -> ConfidenceResult:
    """
    Compute confidence for a ScopeItem.

    Args:
        scope_item: ScopeItem object
        conflicts_count: Number of related conflicts
        required_fields: Dict of required field -> present

    Returns:
        ConfidenceResult
    """
    # Extract evidence sources
    evidence_sources = set()
    evidence_count = 0

    if hasattr(scope_item, 'evidence') and scope_item.evidence:
        evidence_count = len(scope_item.evidence)
        for ev in scope_item.evidence:
            if hasattr(ev, 'source'):
                source = ev.source.value if hasattr(ev.source, 'value') else str(ev.source)
                evidence_sources.add(source.lower())

    # Get status
    status = "detected"
    if hasattr(scope_item, 'status'):
        status = scope_item.status.value if hasattr(scope_item.status, 'value') else str(scope_item.status)

    # Determine item type from category/trade
    item_type = "default"
    if hasattr(scope_item, 'category'):
        category = scope_item.category.value if hasattr(scope_item.category, 'value') else str(scope_item.category)
        item_type = category.lower()

    return compute_confidence(
        item_type=item_type,
        status=status,
        evidence_sources=evidence_sources,
        evidence_count=evidence_count,
        required_fields_present=required_fields or {},
        conflicts_count=conflicts_count,
        qty_status=None,
        has_schedule_data="camelot" in evidence_sources,
        has_plan_data="heuristic" in evidence_sources,
    )
