"""
Feedback Capture — persist user feedback on RFIs and quantities as JSONL.

Schema: one JSON object per line in feedback.jsonl.
First JSONL write in the project.

Pure module except for append_feedback() which writes to disk.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def make_feedback_entry(
    feedback_type: str,
    item_id: str,
    verdict: str,
    doc_id: Optional[str] = None,
    page_refs: Optional[List[int]] = None,
    corrected_value: Optional[str] = None,
    original_value: Optional[str] = None,
    notes: str = "",
) -> dict:
    """
    Build a feedback entry dict (does NOT write to disk).

    Args:
        feedback_type: "rfi" or "quantity".
        item_id: RFI ID or quantity evidence_bundle_id.
        verdict: "correct", "wrong", or "edited".
        doc_id: Optional document identifier.
        page_refs: Optional list of page indices.
        corrected_value: Optional corrected value (for "edited" verdicts).
        original_value: Optional original value being corrected.
        notes: Optional free-text notes.

    Returns:
        Dict with all fields + timestamp.
    """
    return {
        "feedback_type": feedback_type,
        "item_id": item_id,
        "verdict": verdict,
        "doc_id": doc_id,
        "page_refs": page_refs or [],
        "corrected_value": corrected_value,
        "original_value": original_value,
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }


def append_feedback(
    feedback_entry: dict,
    output_dir: Path,
) -> Path:
    """
    Append a feedback entry as a JSON line to feedback.jsonl.

    Args:
        feedback_entry: Dict from make_feedback_entry().
        output_dir: Project output directory (e.g., out/<project_id>/).

    Returns:
        Path to the feedback.jsonl file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "feedback.jsonl"
    with open(filepath, "a") as f:
        f.write(json.dumps(feedback_entry, default=str) + "\n")
    return filepath


def load_feedback(output_dir: Path) -> List[dict]:
    """
    Load all feedback entries from feedback.jsonl.

    Returns empty list if file missing or empty.
    """
    filepath = Path(output_dir) / "feedback.jsonl"
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
