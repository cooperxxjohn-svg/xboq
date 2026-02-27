"""
Supersedes Detector — detect supersede/replace language in addendum text.

Tags conflicts whose addendum pages contain supersede language as
'intentional_revision' so the UI can separate true conflicts from
deliberate revisions.

All functions are pure (no Streamlit, no I/O).
"""

import re
from typing import List, Dict, Optional, Union


# =============================================================================
# SUPERSEDE PATTERNS
# =============================================================================

_SUPERSEDE_PATTERNS_RAW = [
    r'this\s+supersedes',
    r'(?:hereby\s+)?replaces?\s+(?:clause|section|item|paragraph)\s+[\w\d.]+',
    r'in\s+lieu\s+of',
    r'previous\s+version\s+(?:stands?\s+)?cancelled',
    r'delete\s+and\s+substitute',
    r'is\s+hereby\s+revised',
    r'shall\s+(?:supersede|replace|override)',
    r'(?:this|the)\s+(?:addendum|amendment|corrigendum)\s+(?:supersedes|replaces)',
    r'notwithstanding\s+(?:the\s+)?(?:earlier|previous|original)',
    r'(?:stands?\s+)?(?:deleted|cancelled|withdrawn)\s+and\s+(?:replaced|substituted)',
]

SUPERSEDE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _SUPERSEDE_PATTERNS_RAW
]


# =============================================================================
# DETECTION
# =============================================================================

def detect_supersede_language(text: str) -> List[dict]:
    """
    Scan text for supersede/replace patterns.

    Args:
        text: OCR text from an addendum page.

    Returns:
        List of match dicts:
        [{
            "pattern": str,        # the regex pattern that matched
            "matched_text": str,   # the actual matched substring
            "start": int,          # char offset in text
            "end": int,            # char offset end
        }]
    """
    if not text:
        return []

    matches = []
    for i, compiled in enumerate(SUPERSEDE_PATTERNS):
        for m in compiled.finditer(text):
            matches.append({
                "pattern": _SUPERSEDE_PATTERNS_RAW[i],
                "matched_text": m.group(),
                "start": m.start(),
                "end": m.end(),
            })
    return matches


def classify_conflict_as_supersede(
    conflict: dict,
    addendum_page_texts: Dict[Union[int, str], str],
) -> Optional[str]:
    """
    Check if a conflict's addendum_page contains supersede language.

    Args:
        conflict: A conflict dict with 'addendum_page' key (int, 0-indexed).
        addendum_page_texts: Mapping of page_idx -> OCR text.

    Returns:
        "intentional_revision" if supersede language found on that page,
        else None.
    """
    page_idx = conflict.get("addendum_page")
    if page_idx is None:
        return None

    text = addendum_page_texts.get(page_idx, "")
    if not text:
        # Try string key
        text = addendum_page_texts.get(str(page_idx), "")

    if not text:
        return None

    matches = detect_supersede_language(text)
    if matches:
        return "intentional_revision"
    return None


def tag_conflicts_with_supersedes(
    conflicts: List[dict],
    addendum_page_texts: Dict[Union[int, str], str],
) -> List[dict]:
    """
    Add 'resolution' field to each conflict based on supersede detection.

    For each conflict:
    - If addendum_page has supersede language -> resolution = "intentional_revision"
    - Otherwise -> resolution = None (unchanged/open conflict)

    Returns new list of shallow-copied dicts (does not mutate input).
    """
    result = []
    for c in conflicts:
        c_copy = {**c}
        resolution = classify_conflict_as_supersede(c, addendum_page_texts)
        c_copy["resolution"] = resolution
        result.append(c_copy)
    return result
