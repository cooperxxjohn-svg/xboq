"""
Bulk Actions — batch operations on review queue items.

Actions:
  1. prefer_schedule_for_all_mismatches — resolve door/window recon via schedule
  2. generate_rfis_for_high_mismatches — create RFIs for HIGH impact items
  3. mark_intentional_revisions_reviewed — accept intentional revisions

Pure module, no Streamlit dependency. Can be tested independently.
"""

from datetime import datetime
from typing import Dict, List, Tuple

from .quantity_reconciliation import apply_reconciliation_action
from .approval_states import set_conflict_status


# ─── Bulk Action Functions ────────────────────────────────────────────────

def prefer_schedule_for_mismatches(
    recon_rows: List[dict],
    categories: List[str] = None,
) -> Tuple[List[dict], int]:
    """
    Apply 'prefer_schedule' to all mismatched recon rows in given categories.

    Args:
        recon_rows: Full reconciliation list.
        categories: Filter to these categories (default: ["doors", "windows"]).

    Returns:
        Tuple of (updated_rows, actions_taken_count).
    """
    categories = categories or ["doors", "windows"]
    updated = []
    count = 0
    for row in recon_rows:
        if row.get("mismatch") and row.get("category") in categories:
            updated.append(apply_reconciliation_action(
                row, "prefer_schedule",
                note=f"Bulk action: prefer schedule for {row.get('category', '')}",
            ))
            count += 1
        else:
            updated.append(dict(row))
    return updated, count


def generate_rfis_for_high_mismatches(
    recon_rows: List[dict],
    existing_rfis: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """
    Create RFIs for all HIGH-impact reconciliation mismatches.

    A mismatch is HIGH impact when max_delta >= 5.

    Args:
        recon_rows: Full reconciliation list.
        existing_rfis: Current RFI list (for ID generation).

    Returns:
        Tuple of (new_rfis_created, updated_recon_rows).
    """
    from .recon_actions import create_recon_rfi, finding_to_proposed_rfi

    new_rfis: List[dict] = []
    updated_rows: List[dict] = []
    running_rfis = list(existing_rfis)

    for row in recon_rows:
        if row.get("mismatch") and row.get("max_delta", 0) >= 5:
            # Build description based on row type (structural vs count-based)
            cat = row.get("category", "").title()
            delta_pct = row.get("delta_pct")
            if delta_pct is not None:
                # Structural reconciliation row (BOQ vs drawing qty)
                unit = row.get("boq_unit", "")
                desc = (
                    f"{cat} quantity mismatch: "
                    f"BOQ={row.get('boq_count')} {unit}, "
                    f"drawing={row.get('drawing_count')} {unit}, "
                    f"delta={delta_pct:+.1f}%"
                )
            else:
                # Count-based reconciliation row (doors/windows/finishes)
                desc = (
                    f"{cat} mismatch: "
                    f"schedule={row.get('schedule_count')}, "
                    f"boq={row.get('boq_count')}, "
                    f"drawing={row.get('drawing_count')}"
                )
            finding = {
                "type": "conflict",
                "category": "boq_vs_schedule",
                "impact": "high",
                "description": desc,
                "evidence": {"pages": [], "items": [row.get("category", "")]},
                "confidence": 0.8,
            }
            proposed_rfi = finding_to_proposed_rfi(finding)
            new_rfi = create_recon_rfi(proposed_rfi, running_rfis, finding)
            new_rfi["source"] = "bulk_action"
            new_rfi["bulk_action"] = "generate_rfis_for_high_mismatches"
            new_rfis.append(new_rfi)
            running_rfis.append(new_rfi)
            updated_rows.append(apply_reconciliation_action(row, "create_rfi"))
        else:
            updated_rows.append(dict(row))

    return new_rfis, updated_rows


def mark_intentional_revisions_reviewed(
    conflicts: List[dict],
) -> Tuple[List[dict], int]:
    """
    Mark all conflicts with resolution='intentional_revision' as reviewed.

    Args:
        conflicts: Full conflicts list.

    Returns:
        Tuple of (updated_conflicts, count_marked).
    """
    updated = []
    count = 0
    for c in conflicts:
        if c.get("resolution") == "intentional_revision":
            updated.append(set_conflict_status(c, "reviewed"))
            count += 1
        else:
            updated.append(dict(c))
    return updated, count
