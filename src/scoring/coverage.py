"""
XBOQ Coverage Scoring Module
Computes how well each BOQ item is supported by evidence.

Coverage reflects the quality and breadth of supporting data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class CoverageResult:
    """Result of coverage computation with full breakdown."""
    score: float
    factors: List[str] = field(default_factory=list)
    adjustments: Dict[str, float] = field(default_factory=dict)
    pages_used: Set[int] = field(default_factory=set)
    sources_used: Set[str] = field(default_factory=set)
    evidence_count: int = 0

    @property
    def breakdown(self) -> str:
        """Human-readable breakdown of coverage score."""
        parts = []
        for name, adj in self.adjustments.items():
            sign = "+" if adj >= 0 else ""
            parts.append(f"{name}={sign}{adj:.2f}")
        parts.append(f"→ {self.score:.0%}")
        return " | ".join(parts) if parts else "no evidence"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "factors": self.factors,
            "adjustments": self.adjustments,
            "pages_used": list(self.pages_used),
            "sources_used": list(self.sources_used),
            "evidence_count": self.evidence_count,
            "breakdown": self.breakdown,
        }


def compute_coverage_score(
    evidence_list: Optional[List[Any]] = None,
    qty_status: Optional[str] = None,
    source: str = "explicit",
    has_snippet: bool = False,
    item_name: str = "",
) -> CoverageResult:
    """
    Compute coverage score based on evidence quality.

    Rules:
    - Start at 0
    - +0.40 if evidence_count >= 1
    - +0.20 if evidence_sources >= 2 (diverse sources)
    - +0.30 if qty_status == "computed"
    - +0.15 if qty_status == "partial"
    - -0.20 if item is inferred AND has no evidence
    - Clamp 0..1

    Args:
        evidence_list: List of Evidence objects
        qty_status: Quantity status (computed, partial, unknown)
        source: Item source (explicit, inferred, synonym)
        has_snippet: Whether evidence has meaningful snippets
        item_name: Item name for context

    Returns:
        CoverageResult with score and breakdown
    """
    evidence_list = evidence_list or []

    factors = []
    adjustments = {}
    pages_used = set()
    sources_used = set()
    evidence_count = len(evidence_list)

    # Extract evidence metadata
    for ev in evidence_list:
        if hasattr(ev, 'page'):
            pages_used.add(ev.page)
        if hasattr(ev, 'source'):
            source_val = ev.source.value if hasattr(ev.source, 'value') else str(ev.source)
            sources_used.add(source_val.lower())
        # Check for meaningful snippet
        if hasattr(ev, 'snippet') and ev.snippet and len(ev.snippet) > 10:
            has_snippet = True

    score = 0.0

    # 1. Evidence presence (+0.40)
    if evidence_count >= 1:
        adjustments["has_evidence"] = 0.40
        factors.append(f"{evidence_count} evidence item(s)")
        score += 0.40

    # 2. Evidence source diversity (+0.20)
    if len(sources_used) >= 2:
        adjustments["diverse_sources"] = 0.20
        factors.append(f"{len(sources_used)} source types")
        score += 0.20
    elif len(sources_used) == 1:
        adjustments["single_source"] = 0.10
        factors.append("1 source type")
        score += 0.10

    # 3. Quantity status
    if qty_status:
        qty_lower = qty_status.lower() if isinstance(qty_status, str) else str(qty_status).lower()
        if qty_lower == "computed":
            adjustments["qty_computed"] = 0.30
            factors.append("quantity computed")
            score += 0.30
        elif qty_lower == "partial":
            adjustments["qty_partial"] = 0.15
            factors.append("quantity partial")
            score += 0.15
        elif qty_lower == "unknown":
            # No bonus, but don't penalize here
            factors.append("quantity unknown")

    # 4. Snippet quality bonus
    if has_snippet and evidence_count > 0:
        adjustments["has_snippets"] = 0.05
        factors.append("evidence snippets")
        score += 0.05

    # 5. Multiple pages bonus
    if len(pages_used) >= 2:
        adjustments["multi_page"] = 0.05
        factors.append(f"{len(pages_used)} pages")
        score += 0.05

    # 6. Inferred without evidence penalty
    if source.lower() == "inferred" and evidence_count == 0:
        adjustments["inferred_no_evidence"] = -0.20
        factors.append("inferred without evidence")
        score -= 0.20

    # 7. Schedule-derived bonus
    if "camelot" in sources_used or "pdfplumber" in sources_used:
        adjustments["schedule_derived"] = 0.05
        factors.append("from schedule")
        score += 0.05

    # Clamp to valid range
    score = max(0.0, min(1.0, score))

    return CoverageResult(
        score=round(score, 3),
        factors=factors,
        adjustments=adjustments,
        pages_used=pages_used,
        sources_used=sources_used,
        evidence_count=evidence_count,
    )


def compute_boq_coverage(boq_item: Any) -> CoverageResult:
    """
    Compute coverage for a BOQItem.

    Args:
        boq_item: BOQItem object

    Returns:
        CoverageResult
    """
    evidence_list = getattr(boq_item, 'evidence', []) or []
    qty_status = None
    if hasattr(boq_item, 'qty_status'):
        qty_status = boq_item.qty_status.value if hasattr(boq_item.qty_status, 'value') else str(boq_item.qty_status)

    source = getattr(boq_item, 'source', 'explicit')
    item_name = getattr(boq_item, 'item_name', '')

    return compute_coverage_score(
        evidence_list=evidence_list,
        qty_status=qty_status,
        source=source,
        item_name=item_name,
    )


def build_coverage_records(boq_items: List[Any]) -> List[Any]:
    """
    Build CoverageRecord objects for all BOQ items.

    Args:
        boq_items: List of BOQItem objects

    Returns:
        List of CoverageRecord objects
    """
    # Import here to avoid circular imports
    from src.models.estimate_schema import CoverageRecord, EvidenceSource

    records = []

    for item in boq_items:
        coverage = compute_boq_coverage(item)

        # Convert sources to EvidenceSource enum
        sources_enum = set()
        for s in coverage.sources_used:
            try:
                sources_enum.add(EvidenceSource(s))
            except ValueError:
                # Unknown source, skip
                pass

        record = CoverageRecord(
            boq_item_id=getattr(item, 'id', ''),
            contributed_by=getattr(item, 'evidence', []) or [],
            pages_used=coverage.pages_used,
            coverage_score=coverage.score,
            sources_used=sources_enum,
            coverage_breakdown=coverage.adjustments,
        )

        records.append(record)

    return records
