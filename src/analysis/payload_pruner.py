"""
Payload Pruner — strip large/redundant data from the pipeline payload
before Streamlit session state serialisation or Supabase storage.

Streamlit Cloud serialises session_state to ~50 MB per session.
The raw payload can exceed this if it includes all_page_texts, rendered
PNG bytes, or TF-IDF index arrays.

Usage:
    from src.analysis.payload_pruner import prune_payload, estimate_payload_mb

    slim = prune_payload(payload)                  # default: safe pruning
    slim = prune_payload(payload, level="minimal") # keep almost everything
    slim = prune_payload(payload, level="full")    # strip everything heavy
    mb   = estimate_payload_mb(payload)            # quick size estimate
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Keys that are almost always safe to drop ────────────────────────────────
_ALWAYS_DROP: Set[str] = {
    "all_page_texts",       # raw full-page OCR text (can be MB each)
    "raw_text",             # concatenated full-text dump
    "page_images",          # base64 PNG blobs
    "tfidf_index",          # serialised TF-IDF matrix
    "chroma_index",         # ChromaDB in-memory index
    "rendered_pages",       # list of PIL images / numpy arrays
    "_debug",               # internal debug dump
}

# ── Keys that are large but useful; only dropped at "standard" / "full" ──────
_STANDARD_DROP: Set[str] = {
    "page_profiles",        # per-page OCR profile (can be 100+ items)
    "ocr_raw_results",      # raw Tesseract output per page
    "extraction_raw",       # raw extractor output before normalisation
}

# ── Keys only dropped at "full" pruning ──────────────────────────────────────
_FULL_DROP: Set[str] = {
    "boq_items_raw",        # pre-normalised BOQ rows (boq_items has the clean version)
    "page_index_raw",       # full PageIndex object dump
    "schedule_raw_tables",  # raw schedule table arrays
    "drawing_raw",          # raw drawing extractor output
}

# ── Keys to truncate (keep first N items) ─────────────────────────────────────
_TRUNCATE_LISTS: Dict[str, int] = {
    "boq_items":    2000,   # practical UI limit
    "line_items":   2000,
    "rfis":         500,
    "blockers":     200,
}


def estimate_payload_mb(payload: Dict[str, Any]) -> float:
    """Estimate payload size in MB via JSON serialisation (approximate)."""
    try:
        return sys.getsizeof(json.dumps(payload, default=str)) / (1024 * 1024)
    except Exception:
        return -1.0


def prune_payload(
    payload: Dict[str, Any],
    level: str = "standard",
    truncate_lists: bool = True,
    max_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return a pruned copy of the payload dict suitable for session state or storage.

    Parameters
    ----------
    payload : dict
        Full pipeline payload.
    level : str
        ``"minimal"`` — only drop binary blobs (page_images, raw_text).
        ``"standard"`` — drop page_profiles, ocr_raw_results (default).
        ``"full"``     — also drop pre-normalised intermediate keys.
    truncate_lists : bool
        If True, truncate long list fields to their practical UI limits.
    max_mb : float or None
        If set and the pruned payload is still larger than this, escalate
        to the next pruning level automatically.

    Returns
    -------
    dict
        Shallow copy with heavy keys removed and lists truncated.
    """
    if not isinstance(payload, dict):
        return payload

    level = level.lower()
    if level not in ("minimal", "standard", "full"):
        level = "standard"

    # Determine which keys to drop
    drop_keys = set(_ALWAYS_DROP)
    if level in ("standard", "full"):
        drop_keys |= _STANDARD_DROP
    if level == "full":
        drop_keys |= _FULL_DROP

    slim = {k: v for k, v in payload.items() if k not in drop_keys}

    # Truncate long lists
    if truncate_lists:
        for key, limit in _TRUNCATE_LISTS.items():
            if key in slim and isinstance(slim[key], list) and len(slim[key]) > limit:
                logger.debug(
                    "payload_pruner: truncated %s from %d → %d items",
                    key, len(slim[key]), limit,
                )
                slim[key] = slim[key][:limit]

    # Auto-escalate if still too large
    if max_mb is not None and max_mb > 0:
        actual_mb = estimate_payload_mb(slim)
        if actual_mb > max_mb:
            if level == "minimal":
                logger.warning(
                    "Payload %.1f MB > %.1f MB target; escalating to standard pruning",
                    actual_mb, max_mb,
                )
                return prune_payload(payload, level="standard", truncate_lists=True, max_mb=max_mb)
            elif level == "standard":
                logger.warning(
                    "Payload %.1f MB > %.1f MB target; escalating to full pruning",
                    actual_mb, max_mb,
                )
                return prune_payload(payload, level="full", truncate_lists=True, max_mb=None)
            else:
                logger.error(
                    "Payload %.1f MB still exceeds target %.1f MB after full pruning",
                    actual_mb, max_mb,
                )

    return slim


def prune_for_session_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prune payload for Streamlit session state storage (50 MB budget).

    Uses standard pruning with auto-escalation to 40 MB limit.
    """
    return prune_payload(payload, level="standard", truncate_lists=True, max_mb=40.0)


def prune_for_supabase(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prune payload for Supabase JSONB storage (1 MB row limit).

    Uses full pruning with auto-escalation to 0.8 MB limit.
    """
    return prune_payload(payload, level="full", truncate_lists=True, max_mb=0.8)
