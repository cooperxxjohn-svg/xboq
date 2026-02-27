"""
Email Draft Generator — creates email draft .txt files for approved RFIs
and exclusions/clarifications.

Wraps existing rfi_pack.py functions with approval state filtering.
Pure function, no Streamlit dependency. Can be tested independently.

Sprint 15: Packaging + Proof + Meeting Workflow.
"""

from typing import Dict, List, Optional
from datetime import datetime

from src.analysis.approval_states import filter_rfis_for_export


# =============================================================================
# RFI EMAIL DRAFTS
# =============================================================================

def generate_rfi_email_drafts(
    rfis: List[dict],
    include_drafts: bool = False,
) -> Dict[str, str]:
    """
    Generate email drafts for approved RFIs, grouped by trade.

    Filters by approval state, then uses existing rfi_pack.py logic.

    Args:
        rfis: List of RFI dicts from analysis payload.
        include_drafts: If True, include draft RFIs too.

    Returns:
        Dict mapping filename -> email text content.
        e.g. {"rfi_email_all.txt": "Subject: ...", "rfi_email_structural.txt": "..."}
    """
    filtered = filter_rfis_for_export(rfis, include_drafts=include_drafts)
    if not filtered:
        return {}

    try:
        from src.exports.rfi_pack import (
            parse_rfis_to_items, generate_email_draft, group_by_trade,
        )
        items = parse_rfis_to_items(filtered)
    except Exception:
        # Fallback: generate simple email without rfi_pack
        items = None

    drafts = {}

    if items:
        # Combined email for all trades
        all_text = generate_email_draft(items)
        if all_text:
            drafts["rfi_email_all.txt"] = all_text

        # Per-trade emails
        grouped = group_by_trade(items)
        for trade, trade_items in grouped.items():
            if trade_items:
                trade_text = generate_email_draft(trade_items, trade=trade)
                if trade_text:
                    trade_key = trade.lower().replace(" ", "_")
                    drafts[f"rfi_email_{trade_key}.txt"] = trade_text
    else:
        # Fallback: generate plain-text email directly from dicts
        drafts["rfi_email_all.txt"] = _generate_simple_rfi_email(filtered)

    return drafts


def _generate_simple_rfi_email(rfis: List[dict]) -> str:
    """Fallback: generate a simple email from raw RFI dicts."""
    lines = [
        f"Subject: RFIs — {len(rfis)} Items Requiring Clarification",
        "",
        "Dear Consultant,",
        "",
        f"Please find below {len(rfis)} Request(s) for Information (RFIs) "
        "that require clarification before we can finalize our pricing.",
        "",
        "=" * 60,
    ]

    for i, rfi in enumerate(rfis, 1):
        rfi_id = rfi.get("id", f"RFI-{i:04d}")
        question = rfi.get("question", rfi.get("title", ""))
        trade = rfi.get("trade", "general").title()
        priority = rfi.get("priority", "medium").upper()

        lines.extend([
            "",
            f"RFI #{i}: {rfi_id}",
            "-" * 40,
            f"Trade: {trade}",
            f"Priority: {priority}",
            "",
            f"Question:",
            question,
            "",
        ])

        resolution = rfi.get("suggested_resolution", rfi.get("suggested_response", ""))
        if resolution:
            lines.extend([f"Requested Action:", resolution, ""])

    lines.extend([
        "=" * 60,
        "",
        "Please respond at your earliest convenience.",
        "",
        "Best regards,",
        "[Your Name]",
        "[Company]",
    ])

    return "\n".join(lines)


# =============================================================================
# EXCLUSION / CLARIFICATION EMAIL DRAFT
# =============================================================================

def generate_exclusion_email_draft(
    assumptions: List[dict],
    project_name: str = "",
) -> str:
    """
    Generate email draft for exclusions and clarifications.

    Args:
        assumptions: List of assumption dicts with 'status' field.
                     status='rejected' → Exclusions, status='accepted' → Clarifications.
        project_name: Project name for the email subject.

    Returns:
        Email text for exclusions/clarifications notification.
    """
    if not assumptions:
        return ""

    exclusions = [a for a in assumptions if a.get("status") == "rejected"]
    clarifications = [a for a in assumptions if a.get("status") == "accepted"]

    if not exclusions and not clarifications:
        return ""

    project_label = f" — {project_name}" if project_name else ""
    total = len(exclusions) + len(clarifications)

    lines = [
        f"Subject: Exclusions & Clarifications{project_label} — {total} Items",
        "",
        "Dear Project Team,",
        "",
        "Please find below our exclusions and clarifications for this tender submission.",
        "",
    ]

    if exclusions:
        lines.extend([
            "=" * 60,
            "EXCLUSIONS",
            "=" * 60,
            "",
            "The following items are EXCLUDED from our pricing:",
            "",
        ])
        for i, exc in enumerate(exclusions, 1):
            title = exc.get("title", exc.get("assumption", f"Exclusion #{i}"))
            cost_impact = exc.get("cost_impact", "")
            lines.append(f"  {i}. {title}")
            if cost_impact:
                lines.append(f"     Cost Impact: {cost_impact}")
            lines.append("")

    if clarifications:
        lines.extend([
            "=" * 60,
            "CLARIFICATIONS",
            "=" * 60,
            "",
            "The following clarifications apply to our pricing:",
            "",
        ])
        for i, clar in enumerate(clarifications, 1):
            title = clar.get("title", clar.get("assumption", f"Clarification #{i}"))
            text = clar.get("text", clar.get("description", ""))
            lines.append(f"  {i}. {title}")
            if text:
                lines.append(f"     {text[:150]}")
            lines.append("")

    lines.extend([
        "=" * 60,
        "",
        "Please confirm receipt and advise if any items require further discussion.",
        "",
        "Best regards,",
        "[Your Name]",
        "[Company]",
    ])

    return "\n".join(lines)


# =============================================================================
# CONVENIENCE: ALL EMAIL DRAFTS
# =============================================================================

def generate_all_email_drafts(
    rfis: List[dict],
    assumptions: Optional[List[dict]] = None,
    include_drafts: bool = False,
    project_name: str = "",
) -> Dict[str, str]:
    """
    Generate all email drafts and return as a single dict.

    Convenience function that merges RFI email drafts and exclusion email draft.

    Args:
        rfis: List of RFI dicts.
        assumptions: Optional list of assumption dicts.
        include_drafts: If True, include draft RFIs.
        project_name: Project name for email subjects.

    Returns:
        Dict mapping filename -> email text content.
    """
    drafts = {}

    # RFI emails
    rfi_drafts = generate_rfi_email_drafts(rfis, include_drafts=include_drafts)
    drafts.update(rfi_drafts)

    # Exclusions email
    if assumptions:
        excl_text = generate_exclusion_email_draft(assumptions, project_name=project_name)
        if excl_text:
            drafts["exclusions_email.txt"] = excl_text

    return drafts
