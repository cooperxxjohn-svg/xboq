"""
Scope Completeness Engine - Ensure no scope is missed or silently assumed.

This module provides:
- Evidence extraction from notes, specs, legends
- Scope register generation with status tracking
- Scope gaps and estimator checklist
- Provisional BOQ items for unclear scope
- Completeness scoring per package

India-specific terminology and construction scopes are used throughout.

Key principle: NEVER silently assume major scope. Missing evidence becomes explicit gaps.
"""

from .evidence import EvidenceExtractor, EvidenceStore, extract_evidence
from .register import ScopeRegisterGenerator, ScopeRegister, ScopeStatus, generate_scope_register
from .checklist import ScopeGapsGenerator, generate_checklist
from .provisionals import ProvisionalItemsGenerator, generate_provisionals
from .completeness import CompletenessScorer, CompletenessReport, score_completeness

__all__ = [
    # Evidence
    "EvidenceExtractor",
    "EvidenceStore",
    "extract_evidence",
    # Register
    "ScopeRegisterGenerator",
    "ScopeRegister",
    "ScopeStatus",
    "generate_scope_register",
    # Checklist
    "ScopeGapsGenerator",
    "generate_checklist",
    # Provisionals
    "ProvisionalItemsGenerator",
    "generate_provisionals",
    # Completeness
    "CompletenessScorer",
    "CompletenessReport",
    "score_completeness",
]


def run_scope_engine(
    project_id: str,
    page_index: list,
    routing_manifest: dict,
    extraction_results: list,
    project_graph: dict,
    boq_entries: list,
    output_dir,
) -> dict:
    """
    Run the complete scope engine pipeline.

    Args:
        project_id: Project identifier
        page_index: List of indexed pages
        routing_manifest: Page routing manifest
        extraction_results: Extraction results list
        project_graph: Project graph dict
        boq_entries: BOQ entries list
        output_dir: Output directory (Path)

    Returns:
        Dict with scope engine results
    """
    from pathlib import Path
    output_dir = Path(output_dir)

    # 1. Extract evidence
    evidence_store = extract_evidence(
        project_id, page_index, routing_manifest, extraction_results, output_dir
    )

    # 2. Generate scope register
    register = generate_scope_register(
        project_id,
        evidence_store.to_dict(),
        extraction_results,
        project_graph,
        boq_entries,
        output_dir,
    )

    # 3. Generate gaps and checklist
    gaps, checklist = generate_checklist(
        register,
        evidence_store.to_dict(),
        output_dir,
        project_id,
    )

    # 4. Generate provisional items
    provisionals = generate_provisionals(register, output_dir)

    # 5. Score completeness
    completeness = score_completeness(
        register,
        evidence_store.to_dict(),
        output_dir,
    )

    return {
        "evidence_count": len(evidence_store.items),
        "scope_items": len(register.items),
        "gaps_count": len(gaps),
        "checklist_count": len(checklist),
        "provisionals_count": len(provisionals),
        "completeness_score": completeness.overall_score,
        "completeness_grade": completeness.grade,
        "top_risks": [r.to_dict() for r in completeness.top_risks[:5]],
    }
