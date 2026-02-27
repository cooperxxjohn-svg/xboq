"""
Determinism Helpers — stable sorting and hashing for reproducible output ordering.

Used as tiebreaker augmentations to existing sort keys. Does NOT replace
existing sort logic — adds a final stable tiebreaker based on content hash.

Pure module, no Streamlit dependency.

Sprint 17: Demo Hardening + Scripted Dataset.
"""

import hashlib
import json
from typing import Any, Callable, List, Sequence


def normalize_for_hashing(value: Any) -> str:
    """
    Normalize a value to a stable string representation for hashing.

    - dicts: JSON-serialized with sorted keys
    - lists/tuples: JSON-serialized (order preserved)
    - None: empty string
    - other: str()
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, default=str)
    return str(value)


def stable_hash_id(item: dict, keys: Sequence[str]) -> str:
    """
    Compute a stable hash from selected keys of a dict.

    Args:
        item: The dict to hash.
        keys: Ordered sequence of keys to include in the hash.

    Returns:
        12-character hex string (SHA-256 prefix).
    """
    hasher = hashlib.sha256()
    for k in keys:
        hasher.update(normalize_for_hashing(item.get(k)).encode("utf-8"))
    return hasher.hexdigest()[:12]


def stable_sort_key(item: dict, keys: Sequence[str]) -> str:
    """
    Build a stable string sort key from selected dict fields.

    Use as the final element in a sort tuple to break ties deterministically.

    Args:
        item: The dict to build key from.
        keys: Fields to concatenate (in order).

    Returns:
        Concatenated normalized string for lexicographic comparison.
    """
    parts = []
    for k in keys:
        parts.append(normalize_for_hashing(item.get(k)))
    return "|".join(parts)


def stable_sort(
    items: List[dict],
    primary_key: Callable[[dict], Any],
    tiebreaker_fields: Sequence[str],
) -> List[dict]:
    """
    Sort a list of dicts with a primary key function and stable tiebreaker.

    Preserves existing sort semantics while adding a deterministic tiebreaker
    based on content hashing.

    Args:
        items: List of dicts to sort.
        primary_key: Existing sort key function (produces a comparable value).
        tiebreaker_fields: Dict keys to use for tiebreaking when primary keys match.

    Returns:
        New sorted list (does not mutate input).
    """
    return sorted(
        items,
        key=lambda x: (primary_key(x), stable_sort_key(x, tiebreaker_fields)),
    )
