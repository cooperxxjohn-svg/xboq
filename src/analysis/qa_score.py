"""
Bid Pack QA Score — 0–100 confidence score for bid pack quality.

Independent from readiness_score. Measures how well the tender documents
have been processed and how clean the analysis outputs are.

Pure module, no Streamlit dependency. Can be tested independently.
"""

from typing import Dict, List, Any, Optional


# =============================================================================
# SUB-COMPONENT SCORING (each 0–20, sum = 0–100)
# =============================================================================

def _score_coverage_completeness(payload: dict) -> int:
    """
    Coverage completeness: how much of the document was deep-processed.
    Score: (pages_deep_processed / pages_total) * 20, capped at 20.
    """
    run_coverage = payload.get("run_coverage") or {}
    total = run_coverage.get("pages_total", 0)
    processed = run_coverage.get("pages_deep_processed", 0)

    if total <= 0:
        return 0
    ratio = min(1.0, processed / total)
    return round(ratio * 20)


def _score_conflict_density(payload: dict) -> int:
    """
    Conflict density: fewer unresolved conflicts = higher score.
    Score: 20 - min(20, unresolved_count * 2).
    Intentional revisions don't count against the score.
    """
    conflicts = payload.get("conflicts", [])
    unresolved = [
        c for c in conflicts
        if c.get("resolution") != "intentional_revision"
    ]
    penalty = min(20, len(unresolved) * 2)
    return 20 - penalty


def _score_addenda_churn(payload: dict) -> int:
    """
    Addenda churn: more addenda = more complexity/risk.
    Score: 20 - min(20, addenda_count * 4).
    """
    addenda = payload.get("addendum_index", [])
    penalty = min(20, len(addenda) * 4)
    return 20 - penalty


def _score_parse_completeness(payload: dict) -> int:
    """
    Parse completeness: have we successfully extracted BOQ, schedules, and requirements?
    Score: (has_boq + has_schedule + has_requirements) / N * 20.

    BUG-8 FIX: For drawing-only sets (is_drawing_set=True, boq_items=0) the BOQ is
    expected to arrive as a separate document.  Penalising these documents for a missing
    BOQ produces an artificially low score and a misleading "Upload BOQ" action.  When a
    confirmed drawing set has no BOQ, we score over 2 components (schedule +
    requirements) rather than 3.
    """
    ext = payload.get("extraction_summary", {})
    counts = ext.get("counts", {})

    has_boq = 1 if counts.get("boq_items", 0) > 0 else 0
    has_schedule = 1 if counts.get("schedules", 0) > 0 else 0
    has_requirements = 1 if counts.get("requirements", 0) > 0 else 0

    # Fallback: check top-level extraction_summary keys
    if not counts:
        has_boq = 1 if ext.get("boq_items", 0) > 0 else 0
        has_schedule = 1 if ext.get("schedules", 0) > 0 else 0
        has_requirements = 1 if ext.get("requirements", 0) > 0 else 0

    # Drawing-only set: BOQ is a separate document, do not penalise for its absence
    drawing_overview = payload.get("drawing_overview", {})
    is_confirmed_drawing_set = drawing_overview.get("is_drawing_set", True)
    if is_confirmed_drawing_set and has_boq == 0:
        denom = 2  # score over schedule + requirements only
        total = has_schedule + has_requirements
    else:
        denom = 3
        total = has_boq + has_schedule + has_requirements

    return round(total / denom * 20)


def _score_toxic_penalty(payload: dict) -> int:
    """
    Toxic page penalty: pages that could not be OCR'd at all.
    Score: 20 - min(20, toxic_count * 5).
    """
    toxic_data = payload.get("toxic_pages") or {}
    toxic_count = toxic_data.get("toxic_count", 0)
    penalty = min(20, toxic_count * 5)
    return 20 - penalty


# =============================================================================
# IMPROVEMENT ACTIONS
# =============================================================================

_ACTION_TEMPLATES = {
    "coverage_completeness": [
        "Run full-read mode to process all {total} pages (currently {processed}/{total})",
        "Increase OCR budget to cover more document types",
    ],
    "conflict_density": [
        "Resolve {count} open conflicts to reduce ambiguity",
        "Review conflicts and tag intentional revisions",
    ],
    "addenda_churn": [
        "Consolidate {count} addenda — check for superseded content",
        "Verify all addendum changes are reflected in base documents",
    ],
    "parse_completeness": [
        "Upload BOQ document alongside drawings to enable quantity extraction",
        "Upload schedule documents for schedule parsing",
        "Check that specifications are in text-searchable format",
    ],
    "toxic_penalty": [
        "Re-scan {count} toxic page(s) at higher quality",
        "Provide text-layer PDFs instead of scanned images",
    ],
}


def _generate_actions(breakdown: dict, payload: dict) -> List[str]:
    """Generate top 5 improvement actions from lowest sub-scores."""
    # Sort sub-components by score (lowest first)
    scored = sorted(breakdown.items(), key=lambda x: x[1])

    actions = []
    for component, score in scored:
        if score >= 18:  # Already near-perfect
            continue
        templates = _ACTION_TEMPLATES.get(component, [])
        for template in templates:
            # Fill in template variables
            try:
                run_cov = payload.get("run_coverage") or {}
                conflicts = payload.get("conflicts", [])
                addenda = payload.get("addendum_index", [])
                toxic = payload.get("toxic_pages") or {}
                unresolved = [c for c in conflicts if c.get("resolution") != "intentional_revision"]
                action = template.format(
                    total=run_cov.get("pages_total", 0),
                    processed=run_cov.get("pages_deep_processed", 0),
                    count=len(unresolved) if "conflict" in component
                          else len(addenda) if "addenda" in component
                          else toxic.get("toxic_count", 0),
                )
                actions.append(action)
            except (KeyError, IndexError, ValueError):
                actions.append(template)

            if len(actions) >= 5:
                break
        if len(actions) >= 5:
            break

    return actions[:5]


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def compute_qa_score(payload: dict) -> dict:
    """
    Compute Bid Pack QA confidence score (0–100).

    Independent from readiness_score. Measures document processing quality.

    Sub-components (each 0–20, sum = 0–100):
        coverage_completeness: pages_deep_processed / pages_total
        conflict_density:      fewer unresolved conflicts = higher
        addenda_churn:         fewer addenda = higher
        parse_completeness:    BOQ + schedule + requirements parsed
        toxic_penalty:         fewer toxic pages = higher

    Args:
        payload: Full or partial analysis payload dict.

    Returns:
        Dict with: score, breakdown, top_actions, confidence.
    """
    breakdown = {
        "coverage_completeness": _score_coverage_completeness(payload),
        "conflict_density": _score_conflict_density(payload),
        "addenda_churn": _score_addenda_churn(payload),
        "parse_completeness": _score_parse_completeness(payload),
        "toxic_penalty": _score_toxic_penalty(payload),
    }

    total_score = sum(breakdown.values())
    total_score = max(0, min(100, total_score))

    # Confidence based on data availability
    run_cov = payload.get("run_coverage")
    ext = payload.get("extraction_summary", {})
    if run_cov and ext.get("counts"):
        confidence = "HIGH"
    elif run_cov or ext:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    top_actions = _generate_actions(breakdown, payload)

    return {
        "score": total_score,
        "breakdown": breakdown,
        "top_actions": top_actions,
        "confidence": confidence,
    }
