"""
Approval States — status management for RFIs, quantities, and conflicts.

Status fields:
  - RFIs: draft / approved / sent
  - Quantities: draft / accepted
  - Conflicts: unreviewed / reviewed

Pure module, no Streamlit dependency. Can be tested independently.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any


# ─── Valid Statuses ───────────────────────────────────────────────────────

RFI_STATUSES = {"draft", "approved", "sent"}
QUANTITY_STATUSES = {"draft", "accepted"}
CONFLICT_STATUSES = {"unreviewed", "reviewed"}


# ─── Status Setters (pure, return new dict) ───────────────────────────────

def set_rfi_status(rfi: dict, status: str, approved_by: str = "") -> dict:
    """Return copy of RFI with updated status.

    Args:
        rfi: Original RFI dict.
        status: One of 'draft', 'approved', 'sent'.
        approved_by: Who changed the status (optional).

    Returns:
        New dict with updated status fields.

    Raises:
        ValueError: If status is not a valid RFI status.
    """
    if status not in RFI_STATUSES:
        raise ValueError(f"Invalid RFI status '{status}', must be one of {RFI_STATUSES}")
    updated = {**rfi}
    updated["status"] = status
    updated["status_changed_at"] = datetime.now().isoformat()
    if status in ("approved", "sent"):
        updated["status_changed_by"] = approved_by or rfi.get("status_changed_by", "")
    return updated


def set_quantity_status(qty: dict, status: str, accepted_by: str = "") -> dict:
    """Return copy of quantity with updated status.

    Args:
        qty: Original quantity dict.
        status: One of 'draft', 'accepted'.
        accepted_by: Who changed the status (optional).

    Returns:
        New dict with updated status fields.

    Raises:
        ValueError: If status is not a valid quantity status.
    """
    if status not in QUANTITY_STATUSES:
        raise ValueError(f"Invalid quantity status '{status}', must be one of {QUANTITY_STATUSES}")
    updated = {**qty}
    updated["status"] = status
    updated["status_changed_at"] = datetime.now().isoformat()
    if status == "accepted":
        updated["status_changed_by"] = accepted_by
    return updated


def set_conflict_status(conflict: dict, status: str) -> dict:
    """Return copy of conflict with updated review status.

    Args:
        conflict: Original conflict dict.
        status: One of 'unreviewed', 'reviewed'.

    Returns:
        New dict with updated review_status fields.

    Raises:
        ValueError: If status is not a valid conflict status.
    """
    if status not in CONFLICT_STATUSES:
        raise ValueError(f"Invalid conflict status '{status}', must be one of {CONFLICT_STATUSES}")
    updated = {**conflict}
    updated["review_status"] = status
    updated["review_status_changed_at"] = datetime.now().isoformat()
    return updated


# ─── Export Filters ───────────────────────────────────────────────────────

def filter_rfis_for_export(
    rfis: List[dict],
    include_drafts: bool = False,
) -> List[dict]:
    """Filter RFIs for export: approved/sent by default, optionally include drafts.

    Items without a 'status' field are treated as 'draft' (backward compat).
    """
    if include_drafts:
        return list(rfis)
    return [r for r in rfis if r.get("status", "draft") in ("approved", "sent")]


def filter_quantities_for_export(
    quantities: List[dict],
    include_drafts: bool = False,
) -> List[dict]:
    """Filter quantities: accepted by default, optionally include drafts.

    Items without a 'status' field are treated as 'draft' (backward compat).
    """
    if include_drafts:
        return list(quantities)
    return [q for q in quantities if q.get("status", "draft") == "accepted"]


def filter_conflicts_for_export(
    conflicts: List[dict],
    include_unreviewed: bool = False,
) -> List[dict]:
    """Filter conflicts: reviewed by default, optionally include unreviewed.

    Items without a 'review_status' field are treated as 'unreviewed' (backward compat).
    """
    if include_unreviewed:
        return list(conflicts)
    return [c for c in conflicts if c.get("review_status", "unreviewed") == "reviewed"]


# ─── Bulk Operations ─────────────────────────────────────────────────────

def bulk_set_status(
    items: List[dict],
    status: str,
    entity_type: str,
    changed_by: str = "",
) -> List[dict]:
    """Apply status to all items of given entity type. Returns new list.

    Args:
        items: List of dicts to update.
        status: New status string.
        entity_type: One of 'rfi', 'quantity', 'conflict'.
        changed_by: Who made the change (optional).

    Returns:
        New list of updated dicts.

    Raises:
        ValueError: If entity_type is unknown or status is invalid.
    """
    if entity_type == "rfi":
        return [set_rfi_status(i, status, changed_by) for i in items]
    elif entity_type == "quantity":
        return [set_quantity_status(i, status, changed_by) for i in items]
    elif entity_type == "conflict":
        return [set_conflict_status(i, status) for i in items]
    raise ValueError(f"Unknown entity_type: {entity_type}")
