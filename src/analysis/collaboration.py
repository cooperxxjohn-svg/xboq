"""
Collaboration Primitives — comments, assignments, due dates for any entity.

JSONL storage per project. Each line is one action (comment, assign, due_date).
Follows the feedback.py pattern: append-only JSONL, graceful reads.

Pure module except for append/load which do file I/O.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Constants ────────────────────────────────────────────────────────────

COLLABORATABLE_TYPES = {"rfi", "conflict", "assumption", "quantity", "review_item"}

ACTION_COMMENT = "comment"
ACTION_ASSIGN = "assign"
ACTION_DUE_DATE = "due_date"
ACTION_STATUS = "status_change"

_VALID_ACTIONS = {ACTION_COMMENT, ACTION_ASSIGN, ACTION_DUE_DATE, ACTION_STATUS}


# ─── Entry Builder ────────────────────────────────────────────────────────

def make_collaboration_entry(
    entity_type: str,
    entity_id: str,
    action_type: str,
    data: Dict[str, Any],
    author: str = "",
) -> dict:
    """
    Build a collaboration entry dict (does NOT write to disk).

    Args:
        entity_type: One of COLLABORATABLE_TYPES ("rfi", "conflict", etc.)
        entity_id: Unique identifier for the entity (e.g., "RFI-0001")
        action_type: One of "comment", "assign", "due_date", "status_change"
        data: Action-specific data dict:
            - comment: {"text": str}
            - assign: {"assigned_to": str}
            - due_date: {"due_date": str}  (ISO format date)
            - status_change: {"old_status": str, "new_status": str}
        author: Who performed the action (free text)

    Returns:
        Dict with all fields + timestamp.
    """
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "action_type": action_type,
        "data": dict(data),
        "author": author,
        "timestamp": datetime.now().isoformat(),
    }


# ─── Persistence ──────────────────────────────────────────────────────────

def append_collaboration(
    entry: dict,
    project_dir: Path,
) -> Path:
    """
    Append a collaboration entry as JSON line to collaboration.jsonl.

    Args:
        entry: Dict from make_collaboration_entry().
        project_dir: Project directory (e.g., ~/.xboq/projects/<project_id>/)

    Returns:
        Path to the collaboration.jsonl file.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    filepath = project_dir / "collaboration.jsonl"
    with open(filepath, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return filepath


def load_collaboration(
    project_dir: Path,
) -> List[dict]:
    """
    Load all collaboration entries from collaboration.jsonl.
    Returns empty list if file missing or empty.
    """
    filepath = Path(project_dir) / "collaboration.jsonl"
    if not filepath.exists():
        return []

    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


# ─── Aggregation (pure functions) ─────────────────────────────────────────

def get_entity_collaboration(
    entries: List[dict],
    entity_type: str,
    entity_id: str,
) -> Dict[str, Any]:
    """
    Aggregate all collaboration data for a single entity.

    Returns:
        {
            "comments": [{"text": str, "author": str, "timestamp": str}, ...],
            "assigned_to": str,    # latest assignment, or ""
            "due_date": str,       # latest due date, or ""
        }
    """
    comments: List[dict] = []
    assigned_to = ""
    due_date = ""

    for e in entries:
        if e.get("entity_type") != entity_type or e.get("entity_id") != entity_id:
            continue
        action = e.get("action_type", "")
        data = e.get("data", {})

        if action == ACTION_COMMENT:
            comments.append({
                "text": data.get("text", ""),
                "author": e.get("author", ""),
                "timestamp": e.get("timestamp", ""),
            })
        elif action == ACTION_ASSIGN:
            assigned_to = data.get("assigned_to", "")
        elif action == ACTION_DUE_DATE:
            due_date = data.get("due_date", "")

    return {
        "comments": comments,
        "assigned_to": assigned_to,
        "due_date": due_date,
    }


def get_all_assignments(
    entries: List[dict],
    entity_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return all current assignments, optionally filtered by entity_type.

    Returns list of {entity_type, entity_id, assigned_to, due_date}.
    """
    # Track latest assignment per entity
    latest: Dict[str, Dict[str, str]] = {}

    for e in entries:
        if entity_type and e.get("entity_type") != entity_type:
            continue
        action = e.get("action_type", "")
        key = f"{e.get('entity_type', '')}:{e.get('entity_id', '')}"

        if action == ACTION_ASSIGN:
            latest.setdefault(key, {
                "entity_type": e.get("entity_type", ""),
                "entity_id": e.get("entity_id", ""),
                "assigned_to": "",
                "due_date": "",
            })
            latest[key]["assigned_to"] = e.get("data", {}).get("assigned_to", "")
        elif action == ACTION_DUE_DATE:
            latest.setdefault(key, {
                "entity_type": e.get("entity_type", ""),
                "entity_id": e.get("entity_id", ""),
                "assigned_to": "",
                "due_date": "",
            })
            latest[key]["due_date"] = e.get("data", {}).get("due_date", "")

    return [v for v in latest.values() if v.get("assigned_to")]


def build_collaboration_appendix(
    entries: List[dict],
) -> str:
    """
    Build a text appendix summarizing all comments/assignments for export.
    Returns formatted string suitable for inclusion in DOCX/submission pack.
    """
    if not entries:
        return ""

    # Group by entity
    by_entity: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        key = f"{e.get('entity_type', '').upper()}: {e.get('entity_id', '')}"
        by_entity[key].append(e)

    lines = ["COLLABORATION NOTES", "=" * 40, ""]
    for key in sorted(by_entity.keys()):
        lines.append(key)
        lines.append("-" * len(key))
        for e in by_entity[key]:
            action = e.get("action_type", "")
            data = e.get("data", {})
            author = e.get("author", "Unknown")
            ts = e.get("timestamp", "")[:16]

            if action == ACTION_COMMENT:
                lines.append(f"  [{ts}] {author}: {data.get('text', '')}")
            elif action == ACTION_ASSIGN:
                lines.append(f"  [{ts}] Assigned to: {data.get('assigned_to', '')}")
            elif action == ACTION_DUE_DATE:
                lines.append(f"  [{ts}] Due: {data.get('due_date', '')}")
            elif action == ACTION_STATUS:
                lines.append(
                    f"  [{ts}] Status: {data.get('old_status', '')} → {data.get('new_status', '')}"
                )
        lines.append("")

    return "\n".join(lines)


def enrich_items_with_collaboration(
    items: List[dict],
    entries: List[dict],
    entity_type: str,
    id_field: str = "id",
) -> List[dict]:
    """
    Augment a list of entity dicts with collaboration fields.

    Adds comments[], assigned_to, due_date to each item.
    Returns new list of new dicts (no mutation). Uses .get() with defaults
    for backward compatibility.

    Args:
        items: List of entity dicts (e.g., RFIs, quantities).
        entries: All collaboration entries from load_collaboration().
        entity_type: Type string matching the entries (e.g., "rfi").
        id_field: Key in each item dict that holds the entity ID.

    Returns:
        New list of enriched dicts.
    """
    # Pre-build lookup
    collab_by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        eid = item.get(id_field, "")
        if eid:
            collab_by_id[eid] = get_entity_collaboration(entries, entity_type, eid)

    enriched = []
    for item in items:
        new_item = {**item}
        eid = item.get(id_field, "")
        collab = collab_by_id.get(eid, {})
        new_item["comments"] = collab.get("comments", [])
        new_item["assigned_to"] = collab.get("assigned_to", "")
        new_item["due_date"] = collab.get("due_date", "")
        enriched.append(new_item)

    return enriched
