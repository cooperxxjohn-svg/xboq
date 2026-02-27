"""
Reconciliation Actions — convert reconciliation findings into
proposed RFIs and proposed assumptions.

Provides both proposal generation (pure preview) and finalization
functions (create full RFI/assumption dicts ready for payload).

All functions are pure (no Streamlit, no I/O).
"""

import csv
import io
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# QUESTION TEMPLATES
# =============================================================================

_RFI_QUESTION_TEMPLATES = {
    ("missing", "req_vs_schedule"): (
        "Clarify {impacted_scope}: requirements reference {impacted_scope} "
        "but no corresponding schedule was found in the tender documents. "
        "Please provide the applicable schedule or confirm exclusion."
    ),
    ("missing", "boq_vs_schedule"): (
        "Schedule mark '{item}' ({schedule_type}) has no corresponding BOQ line item. "
        "Please confirm whether this item is included in the scope and provide pricing guidance."
    ),
    ("conflict", "boq_vs_schedule"): (
        "Quantity mismatch: {description}. "
        "Please confirm the correct quantity for pricing."
    ),
    ("conflict", "req_vs_boq"): (
        "Material/spec conflict: {description}. "
        "Please clarify which specification governs for pricing."
    ),
    ("ambiguity", "req_vs_schedule"): (
        "Specification ambiguity: {description}. "
        "Please confirm the applicable specification for tender pricing."
    ),
    ("ambiguity", "req_vs_boq"): (
        "Specification not reflected in BOQ: {description}. "
        "Please confirm if this specification affects pricing."
    ),
}

_ASSUMPTION_TEXT_TEMPLATES = {
    ("missing", "req_vs_schedule"): (
        "In absence of a {impacted_scope} schedule, {impacted_scope} are assumed as "
        "standard commercial grade per IS specifications. Bidder will qualify this "
        "assumption in the cover letter."
    ),
    ("missing", "boq_vs_schedule"): (
        "Schedule item '{item}' is assumed to be included in the nearest matching "
        "BOQ line item. Rate has been distributed proportionally."
    ),
    ("conflict", "boq_vs_schedule"): (
        "Where BOQ and schedule quantities conflict, the higher quantity has been "
        "adopted for pricing to avoid under-estimation. {description}"
    ),
    ("conflict", "req_vs_boq"): (
        "The stricter specification has been assumed for pricing purposes. "
        "{description}"
    ),
    ("ambiguity", "req_vs_schedule"): (
        "The specification '{item}' is assumed to match the nearest standard grade. "
        "This assumption will be qualified in the bid submission."
    ),
    ("ambiguity", "req_vs_boq"): (
        "Material grade '{item}' mentioned in requirements but absent from BOQ "
        "is assumed to be included in related line items at standard rates."
    ),
}

_IMPACT_IF_WRONG = {
    "high": "Significant cost/schedule impact — 15-40% variance on affected items",
    "medium": "Moderate cost impact — 5-15% variance on affected items",
    "low": "Minor cost impact — under 5% variance on affected items",
}

_SCOPE_LABELS = {
    "req_vs_schedule": "Schedule Specification",
    "boq_vs_schedule": "BOQ vs Schedule",
    "req_vs_boq": "Requirement vs BOQ",
}


# =============================================================================
# PROPOSAL GENERATION
# =============================================================================

def _extract_impacted_scope(finding: dict) -> str:
    """Derive a short scope label from finding evidence and description."""
    desc = finding.get("description", "").lower()
    items = finding.get("evidence", {}).get("items", [])

    if "door" in desc:
        return "doors"
    if "window" in desc:
        return "windows"
    if items:
        return items[0][:40]
    # Fallback: first noun-phrase from description
    return desc[:40] if desc else "unspecified scope"


def finding_to_proposed_rfi(finding: dict) -> dict:
    """
    Convert a reconciliation finding into a proposed RFI dict.

    Returns:
        {
            "question": str,
            "category": str,
            "impacted_scope": str,
            "evidence_refs": {"pages": [int], "items": [str], "snippets": [str]},
            "confidence": float,
            "suggested_resolution": str,
        }
    """
    ftype = finding.get("type", "missing")
    fcat = finding.get("category", "req_vs_schedule")
    evidence = finding.get("evidence", {})
    items = evidence.get("items", [])
    impacted_scope = _extract_impacted_scope(finding)

    # Pick template
    template_key = (ftype, fcat)
    template = _RFI_QUESTION_TEMPLATES.get(
        template_key,
        "Please clarify: {description}",
    )

    # Format question
    fmt_kwargs = {
        "impacted_scope": impacted_scope,
        "description": finding.get("description", ""),
        "item": items[0] if items else "",
        "schedule_type": "",
    }
    question = template.format(**fmt_kwargs) if isinstance(template, str) else template

    return {
        "question": question,
        "category": fcat,
        "impacted_scope": impacted_scope,
        "evidence_refs": {
            "pages": evidence.get("pages", []),
            "items": items,
            "snippets": [finding.get("description", "")],
        },
        "confidence": finding.get("confidence", 0.5),
        "suggested_resolution": finding.get("suggested_action", ""),
    }


def finding_to_proposed_assumption(finding: dict) -> dict:
    """
    Convert a reconciliation finding into a proposed assumption dict.

    Returns:
        {
            "assumption_text": str,
            "scope": str,
            "risk_level": str,
            "basis_pages": [int],
        }
    """
    ftype = finding.get("type", "missing")
    fcat = finding.get("category", "req_vs_schedule")
    evidence = finding.get("evidence", {})
    items = evidence.get("items", [])
    impacted_scope = _extract_impacted_scope(finding)

    template_key = (ftype, fcat)
    template = _ASSUMPTION_TEXT_TEMPLATES.get(
        template_key,
        "Assumed standard practice applies for: {description}",
    )

    fmt_kwargs = {
        "impacted_scope": impacted_scope,
        "description": finding.get("description", ""),
        "item": items[0] if items else "",
    }
    assumption_text = template.format(**fmt_kwargs) if isinstance(template, str) else template

    scope_label = _SCOPE_LABELS.get(fcat, fcat.replace("_", " ").title())

    return {
        "assumption_text": assumption_text,
        "scope": scope_label,
        "risk_level": finding.get("impact", "medium"),
        "basis_pages": evidence.get("pages", []),
    }


def generate_proposals(findings: List[dict]) -> List[dict]:
    """
    For each finding, generate both a proposed RFI and proposed assumption.

    Returns:
        List of {finding, proposed_rfi, proposed_assumption} dicts.
    """
    proposals = []
    for finding in findings:
        proposals.append({
            "finding": finding,
            "proposed_rfi": finding_to_proposed_rfi(finding),
            "proposed_assumption": finding_to_proposed_assumption(finding),
        })
    return proposals


# =============================================================================
# RFI / ASSUMPTION CREATION
# =============================================================================

_RECON_RFI_RE = re.compile(r'^RFI-R-(\d+)$')
_RECON_ASMP_RE = re.compile(r'^ASMP-R-(\d+)$')


def _next_recon_rfi_id(existing_rfis: List[dict]) -> str:
    """Find max index among existing RFI-R-XXXX IDs, return next one."""
    max_idx = 0
    for rfi in existing_rfis:
        m = _RECON_RFI_RE.match(rfi.get("id", ""))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return f"RFI-R-{max_idx + 1:04d}"


def _next_recon_assumption_id(existing: List[dict]) -> str:
    """Find max index among existing ASMP-R-XXXX IDs, return next one."""
    max_idx = 0
    for a in existing:
        m = _RECON_ASMP_RE.match(a.get("id", ""))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return f"ASMP-R-{max_idx + 1:04d}"


def create_recon_rfi(
    proposed_rfi: dict,
    existing_rfis: List[dict],
    finding: dict,
) -> dict:
    """
    Convert a proposed_rfi into a full RFI payload dict ready for appending.

    Uses RFI-R-XXXX prefix to avoid collision with pipeline RFI-XXXX IDs.
    Adds source="reconciler" to distinguish from pipeline-generated RFIs.

    Returns:
        Full RFI dict matching the pipeline RFI payload schema.
    """
    rfi_id = _next_recon_rfi_id(existing_rfis)

    # Map finding.impact to RFI priority
    impact_to_priority = {"high": "high", "medium": "medium", "low": "low"}
    priority = impact_to_priority.get(finding.get("impact", "medium"), "medium")

    return {
        "id": rfi_id,
        "trade": "general",
        "priority": priority,
        "question": proposed_rfi["question"],
        "why_it_matters": finding.get("description", ""),
        "evidence": {
            "pages": proposed_rfi["evidence_refs"]["pages"],
            "sheets": [],
            "snippets": proposed_rfi["evidence_refs"].get("snippets", []),
            "detected_entities": {},
            "search_attempts": {},
            "confidence": proposed_rfi["confidence"],
            "confidence_reason": "Generated from scope reconciliation",
        },
        "suggested_resolution": proposed_rfi["suggested_resolution"],
        "acceptable_alternatives": [],
        "related_blocker_id": None,
        "issue_type": f"reconciliation_{finding.get('category', 'unknown')}",
        "package": "",
        "coverage_status": None,
        "created_at": datetime.now().isoformat(),
        "source": "reconciler",
    }


def create_recon_assumption(
    proposed_assumption: dict,
    existing_assumptions: List[dict],
    finding: dict,
) -> dict:
    """
    Convert a proposed_assumption into a full assumption dict for the log.

    Uses ASMP-R-XXXX prefix to distinguish from template-generated assumptions.

    Returns:
        Full assumption dict with stable schema.
    """
    asmp_id = _next_recon_assumption_id(existing_assumptions)

    impact = finding.get("impact", "medium")
    impact_text = _IMPACT_IF_WRONG.get(impact, "Cost impact unknown")

    return {
        "id": asmp_id,
        "title": f"{proposed_assumption['scope']} Assumption",
        "text": proposed_assumption["assumption_text"],
        "impact_if_wrong": impact_text,
        "risk_level": proposed_assumption["risk_level"],
        "basis_pages": proposed_assumption["basis_pages"],
        "linked_blocker_ids": [],
        "source": "reconciler",
        "created_at": datetime.now().isoformat(),
        # Sprint 9: Bid control fields
        "status": "draft",          # "draft" | "accepted" | "rejected"
        "approved_by": None,        # str or None
        "approved_at": None,        # ISO timestamp or None
        "cost_impact": None,        # float or None
        "scope_tag": "",            # free-text scope tag
    }


# =============================================================================
# ASSUMPTION STATUS MANAGEMENT
# =============================================================================

_VALID_STATUSES = {"draft", "accepted", "rejected"}


def update_assumption_status(
    assumption: dict,
    new_status: str,
    approved_by: str = "",
    cost_impact: Optional[float] = None,
    scope_tag: Optional[str] = None,
) -> dict:
    """
    Return a copy of the assumption with updated status fields.

    Pure function — no side effects.

    Args:
        assumption: Existing assumption dict.
        new_status: One of "draft", "accepted", "rejected".
        approved_by: Who approved/rejected (ignored for draft).
        cost_impact: Optional cost impact value to set.
        scope_tag: Optional scope tag to set.

    Returns:
        New dict with updated status fields.

    Raises:
        ValueError: If new_status is not valid.
    """
    if new_status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}', must be one of {_VALID_STATUSES}")

    updated = {**assumption}
    updated["status"] = new_status

    if new_status in ("accepted", "rejected"):
        updated["approved_by"] = approved_by or assumption.get("approved_by")
        updated["approved_at"] = datetime.now().isoformat()
    else:
        # Revert to draft — clear approval fields
        updated["approved_by"] = None
        updated["approved_at"] = None

    if cost_impact is not None:
        updated["cost_impact"] = cost_impact
    if scope_tag is not None:
        updated["scope_tag"] = scope_tag

    return updated


# =============================================================================
# EXCLUSIONS & CLARIFICATIONS EXPORT
# =============================================================================

def generate_exclusions_clarifications(
    assumptions: List[dict],
) -> Tuple[str, str]:
    """
    Generate exclusions_clarifications.txt and .csv content from assumptions.

    Only includes assumptions with status 'accepted' or 'rejected'.
    Accepted → listed under CLARIFICATIONS.
    Rejected → listed under EXCLUSIONS.

    Args:
        assumptions: List of assumption dicts (may include drafts).

    Returns:
        (txt_content, csv_content) tuple.
    """
    accepted = [a for a in assumptions if a.get("status") == "accepted"]
    rejected = [a for a in assumptions if a.get("status") == "rejected"]

    # ── TXT ───────────────────────────────────────────────────────────
    lines = [
        "EXCLUSIONS & CLARIFICATIONS",
        "=" * 28,
        "",
    ]

    if rejected:
        lines.append("EXCLUSIONS (Rejected Assumptions)")
        lines.append("-" * 34)
        for i, a in enumerate(rejected, 1):
            cost = f"${a.get('cost_impact', 0):,.0f}" if a.get("cost_impact") else "N/A"
            scope = a.get("scope_tag", "") or "—"
            lines.append(
                f"{i}. [{a.get('id', '')}] {a.get('title', '')} — "
                f"{a.get('text', '')[:120]} — Cost impact: {cost} — Scope: {scope}"
            )
        lines.append("")

    if accepted:
        lines.append("CLARIFICATIONS (Accepted Assumptions)")
        lines.append("-" * 38)
        for i, a in enumerate(accepted, 1):
            cost = f"${a.get('cost_impact', 0):,.0f}" if a.get("cost_impact") else "N/A"
            scope = a.get("scope_tag", "") or "—"
            lines.append(
                f"{i}. [{a.get('id', '')}] {a.get('title', '')} — "
                f"{a.get('text', '')[:120]} — Cost impact: {cost} — Scope: {scope}"
            )
        lines.append("")

    if not accepted and not rejected:
        lines.append("No accepted or rejected assumptions.")

    txt_content = "\n".join(lines)

    # ── CSV ───────────────────────────────────────────────────────────
    fieldnames = [
        "id", "title", "text", "status", "cost_impact", "scope_tag",
        "risk_level", "approved_by", "approved_at",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for a in rejected + accepted:
        writer.writerow({
            "id": a.get("id", ""),
            "title": a.get("title", ""),
            "text": a.get("text", ""),
            "status": a.get("status", ""),
            "cost_impact": a.get("cost_impact", ""),
            "scope_tag": a.get("scope_tag", ""),
            "risk_level": a.get("risk_level", ""),
            "approved_by": a.get("approved_by", ""),
            "approved_at": a.get("approved_at", ""),
        })
    csv_content = buf.getvalue()

    return txt_content, csv_content
