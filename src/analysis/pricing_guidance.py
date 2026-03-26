"""
Pricing Guidance — contingency range, exclusions, clarifications, VE suggestions.

Input: QA score, addendum index, conflicts, owner profile, run coverage.
Output: contingency_range, recommended_exclusions, recommended_clarifications,
        suggested_alternates_ve.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONTINGENCY LOGIC
# =============================================================================

def _compute_contingency(
    qa_score: Optional[dict],
    addendum_count: int,
    unresolved_conflict_count: int,
    doc_types_not_covered: int,
    owner_profile: Optional[dict],
) -> dict:
    """
    Compute contingency percentage range based on risk signals.

    Base ranges:
        QA > 80  →  3–5%
        QA 50–80 →  5–8%
        QA < 50  →  8–15%

    Adjustments:
        +1% per 2 unresolved conflicts
        +1% per addendum
        +2% if doc_types_not_covered > 0
        -1% if owner is Preferred + past_work > 3
        +2% if owner has disputes
    """
    score = (qa_score or {}).get("score", 50)

    # Base range
    if score > 80:
        low, high = 3.0, 5.0
        rationale_base = "QA score above 80 — low document risk"
    elif score >= 50:
        low, high = 5.0, 8.0
        rationale_base = "QA score 50–80 — moderate document risk"
    else:
        low, high = 8.0, 15.0
        rationale_base = "QA score below 50 — high document risk"

    adjustments = []

    # Conflict adjustment
    conflict_adj = unresolved_conflict_count // 2
    if conflict_adj > 0:
        low += conflict_adj
        high += conflict_adj
        adjustments.append(f"+{conflict_adj}% for {unresolved_conflict_count} unresolved conflicts")

    # Addendum adjustment
    if addendum_count > 0:
        low += addendum_count
        high += addendum_count
        adjustments.append(f"+{addendum_count}% for {addendum_count} addenda")

    # Coverage gap adjustment
    if doc_types_not_covered > 0:
        low += 2
        high += 2
        adjustments.append(f"+2% for {doc_types_not_covered} uncovered document types")

    # Owner profile adjustments
    if owner_profile and isinstance(owner_profile, dict):
        relationship = owner_profile.get("relationship", "").lower()
        past_work = owner_profile.get("past_work", 0)
        disputes = owner_profile.get("disputes", False)

        if relationship == "preferred" and past_work > 3:
            low = max(1, low - 1)
            high = max(2, high - 1)
            adjustments.append("-1% for preferred owner with strong history")

        if disputes:
            low += 2
            high += 2
            adjustments.append("+2% for owner with dispute history")

    # Recommended = midpoint
    recommended = round((low + high) / 2, 1)

    # Build rationale
    rationale_parts = [rationale_base]
    rationale_parts.extend(adjustments)
    rationale = ". ".join(rationale_parts) + "."

    return {
        "low_pct": round(low, 1),
        "high_pct": round(high, 1),
        "recommended_pct": recommended,
        "rationale": rationale,
    }


# =============================================================================
# EXCLUSIONS
# =============================================================================

def _compute_exclusions(
    run_coverage: Optional[dict],
    doc_types_not_covered: List[str],
) -> List[str]:
    """
    Recommend exclusions based on document coverage gaps.

    If certain document types were not covered in the analysis,
    recommend excluding those scopes from the bid.
    """
    exclusions = []

    for dtype in doc_types_not_covered:
        dtype_clean = dtype.replace("_", " ").title()
        exclusions.append(
            f"Scope related to '{dtype_clean}' documents — not covered in analysis"
        )

    # Check for missing key doc types
    if run_coverage and isinstance(run_coverage, dict):
        doc_types_found = run_coverage.get("doc_types_found", [])
        critical_types = ["boq", "schedule", "specification"]
        for ctype in critical_types:
            if ctype not in doc_types_found:
                exclusions.append(
                    f"Quantities/scope from {ctype} documents — not found in tender package"
                )

    return exclusions


# =============================================================================
# CLARIFICATIONS
# =============================================================================

def _compute_clarifications(
    conflicts: List[dict],
) -> List[str]:
    """
    Recommend clarifications based on unresolved conflicts.

    Each unresolved conflict should prompt a clarification in the bid.
    """
    clarifications = []

    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        resolution = conflict.get("resolution", "")
        if resolution == "intentional_revision":
            continue  # Already resolved

        description = conflict.get("description", "") or conflict.get("title", "")
        if description:
            # Truncate long descriptions
            desc_short = description[:120].strip()
            if len(description) > 120:
                desc_short += "..."
            clarifications.append(
                f"Clarify: {desc_short}"
            )

    # Cap at 15 clarifications
    return clarifications[:15]


# =============================================================================
# VE SUGGESTIONS (Value Engineering / Alternates)
# =============================================================================

def _compute_ve_suggestions(
    qa_score: Optional[dict],
) -> List[dict]:
    """
    Suggest VE items from low-scoring QA sub-components.

    When a sub-component scores low, it implies an area where
    the tender package is weak → opportunity to propose alternates.
    """
    suggestions = []
    if not qa_score or not isinstance(qa_score, dict):
        return suggestions

    breakdown = qa_score.get("breakdown", {})
    if not breakdown:
        return suggestions

    # Map low sub-scores to VE suggestions
    _VE_MAP = {
        "coverage_completeness": {
            "threshold": 12,
            "item": "Partial scope definition — propose phased delivery alternate",
            "reason": "Low coverage means many pages unprocessed; scope may be incomplete",
        },
        "conflict_density": {
            "threshold": 12,
            "item": "Multiple conflicting specifications — propose material alternates",
            "reason": "High conflict density suggests ambiguous specifications",
        },
        "addenda_churn": {
            "threshold": 10,
            "item": "High addenda churn — propose design-assist alternate",
            "reason": "Multiple addenda indicate evolving design; design-assist reduces risk",
        },
        "parse_completeness": {
            "threshold": 10,
            "item": "Incomplete tender documents — propose provisional sums for missing scopes",
            "reason": "Missing BOQ/schedules/specs suggest undefined scope areas",
        },
        "toxic_penalty": {
            "threshold": 15,
            "item": "Unreadable document pages — request re-issue of affected sheets",
            "reason": "Toxic pages may contain critical scope information",
        },
    }

    for component, config in _VE_MAP.items():
        score = breakdown.get(component, 20)
        if score < config["threshold"]:
            suggestions.append({
                "item": config["item"],
                "reason": config["reason"],
            })

    return suggestions


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_pricing_guidance(
    qa_score: Optional[dict] = None,
    addendum_index: Optional[List[dict]] = None,
    conflicts: Optional[List[dict]] = None,
    owner_profile: Optional[dict] = None,
    run_coverage: Optional[dict] = None,
    estimating_playbook: Optional[dict] = None,
) -> dict:
    """
    Compute pricing guidance for the bid.

    Args:
        qa_score: QA score dict from compute_qa_score() — {score, breakdown, ...}.
        addendum_index: List of addendum dicts.
        conflicts: List of conflict dicts.
        owner_profile: Optional owner profile dict {relationship, past_work, disputes}.
        run_coverage: Run coverage dict {doc_types_found, doc_types_not_covered, ...}.
        estimating_playbook: Optional estimating playbook dict (Sprint 20C).

    Returns:
        Dict with:
            contingency_range: {low_pct, high_pct, recommended_pct, rationale}
            recommended_exclusions: [str]
            recommended_clarifications: [str]
            suggested_alternates_ve: [{item, reason}]
            basis_of_recommendation: [str] (Sprint 20C, if playbook active)
    """
    addendum_index = addendum_index or []
    conflicts = conflicts or []

    # Count unresolved conflicts
    unresolved_conflicts = [
        c for c in conflicts
        if isinstance(c, dict) and c.get("resolution") != "intentional_revision"
    ]

    # Get doc types not covered
    doc_types_not_covered = []
    if run_coverage and isinstance(run_coverage, dict):
        doc_types_not_covered = run_coverage.get("doc_types_not_covered", [])

    # Compute each section
    contingency = _compute_contingency(
        qa_score=qa_score,
        addendum_count=len(addendum_index),
        unresolved_conflict_count=len(unresolved_conflicts),
        doc_types_not_covered=len(doc_types_not_covered),
        owner_profile=owner_profile,
    )

    # Sprint 20C: Apply playbook adjustments to contingency
    basis_of_recommendation = []
    if estimating_playbook and isinstance(estimating_playbook, dict):
        try:
            from src.analysis.estimating_playbook import (
                compute_playbook_contingency_adjustments,
            )
            _pb_adj = compute_playbook_contingency_adjustments(estimating_playbook)
            _pb_rec = _pb_adj.get("recommended_pct", 0)
            _pb_basis = _pb_adj.get("basis", [])

            # Use playbook as baseline; document-risk adjustments add on top
            _doc_adj = contingency["recommended_pct"] - 5.0  # delta from default 5%
            _combined = round(_pb_rec + max(0, _doc_adj), 1)
            contingency["recommended_pct"] = max(1.0, _combined)
            contingency["low_pct"] = max(1.0, round(_pb_rec - 1.5, 1))
            contingency["high_pct"] = round(_combined + 2.0, 1)

            basis_of_recommendation = list(_pb_basis)
            if _doc_adj > 0:
                basis_of_recommendation.append(
                    f"+{_doc_adj:.1f}% from document risk analysis"
                )
            contingency["rationale"] = (
                contingency.get("rationale", "") +
                " Adjusted by estimating playbook."
            )
        except Exception as e:
            logger.warning(f"Playbook contingency adjustment failed: {e} — using document-risk only")

    exclusions = _compute_exclusions(
        run_coverage=run_coverage,
        doc_types_not_covered=doc_types_not_covered,
    )

    clarifications = _compute_clarifications(conflicts)

    ve_suggestions = _compute_ve_suggestions(qa_score)

    result = {
        "contingency_range": contingency,
        "recommended_exclusions": exclusions,
        "recommended_clarifications": clarifications,
        "suggested_alternates_ve": ve_suggestions,
    }

    if basis_of_recommendation:
        result["basis_of_recommendation"] = basis_of_recommendation

    return result
