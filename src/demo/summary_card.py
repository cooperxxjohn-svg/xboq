"""
Summary Card Builder — extract YC demo summary stats from analysis payload.

Pure module, no Streamlit dependency. Can be tested independently.

Sprint 18: Final Demo Polish.
"""

from typing import Any, Dict, List


def build_summary_card(
    payload: dict,
    project_name: str = "",
    cache_used: bool = False,
) -> Dict[str, Any]:
    """Extract YC summary card data from an analysis payload.

    All fields default to safe fallbacks (0, "N/A", False) so the card
    never crashes regardless of payload shape.

    Args:
        payload: Full analysis payload dict (or empty dict).
        project_name: Human-readable project name for display.
        cache_used: Whether cached results were used (affects time saved).

    Returns:
        Dict with keys:
            total_pages (int), deep_pages (int), ocr_pages (int),
            text_layer_pages (int), skipped_pages (int),
            cache_time_saved (str), qa_score (int), top_actions (List[str]),
            approved_rfis (int), accepted_quantities (int),
            accepted_assumptions (int), submission_pack_ready (bool),
            decision (str), readiness_score (int), project_name (str).
    """
    if not isinstance(payload, dict):
        payload = {}

    # ── Pages (Sprint 18: prefer processing_stats, fallback chain) ────
    pstats = payload.get("processing_stats") or {}
    if isinstance(pstats, dict) and pstats.get("total_pages", 0) > 0:
        total_pages = _safe_int(pstats.get("total_pages", 0))
        deep_pages = _safe_int(pstats.get("deep_processed_pages", 0))
        ocr_pages = _safe_int(pstats.get("ocr_pages", 0))
        text_layer_pages = _safe_int(pstats.get("text_layer_pages", 0))
        skipped_pages = _safe_int(pstats.get("skipped_pages", 0))
    else:
        # Fallback: try run_coverage, then drawing_overview
        rc = payload.get("run_coverage") or {}
        overview = payload.get("drawing_overview") or payload.get("overview") or {}
        total_pages = _safe_int(
            rc.get("pages_total") or overview.get("pages_total", 0))
        deep_pages = _safe_int(
            rc.get("pages_deep_processed") or overview.get("pages_deep", 0))
        ocr_pages = _safe_int(
            overview.get("ocr_pages_count") or overview.get("pages_ocr", 0))
        text_layer_pages = max(0, deep_pages - ocr_pages)
        skipped_pages = max(0, total_pages - deep_pages)

    # ── Cache time saved ───────────────────────────────────────────────
    timings = payload.get("timings", {})
    if cache_used and isinstance(timings, dict):
        elapsed = _safe_int(timings.get("total_seconds", 0))
        cache_time_saved = f"~{elapsed}s saved" if elapsed > 0 else "cache hit"
    else:
        cache_time_saved = "N/A"

    # ── QA Score + top actions ─────────────────────────────────────────
    qa_data = payload.get("qa_score") or {}
    qa_score = _safe_int(qa_data.get("score", 0)) if isinstance(qa_data, dict) else 0
    top_actions_raw = qa_data.get("top_actions", []) if isinstance(qa_data, dict) else []
    top_actions = [str(a) for a in top_actions_raw[:2]] if isinstance(top_actions_raw, list) else []

    # ── Counts ─────────────────────────────────────────────────────────
    rfis = payload.get("rfis", [])
    rfis = rfis if isinstance(rfis, list) else []
    approved_rfis = len([r for r in rfis if isinstance(r, dict) and r.get("status") == "approved"])

    quantities = payload.get("quantities", [])
    quantities = quantities if isinstance(quantities, list) else []
    accepted_quantities = len(quantities)

    assumptions = payload.get("assumptions", [])
    assumptions = assumptions if isinstance(assumptions, list) else []
    accepted_assumptions = len([
        a for a in assumptions
        if isinstance(a, dict) and a.get("status") in ("accepted", "approved")
    ])

    # ── Decision ───────────────────────────────────────────────────────
    decision = str(payload.get("decision", "N/A"))
    readiness_score = _safe_int(payload.get("readiness_score", 0))

    # ── Submission pack ────────────────────────────────────────────────
    # Check if export buffers exist in payload (set by export section)
    submission_pack_ready = bool(payload.get("submission_pack_ready", False))

    return {
        "total_pages": total_pages,
        "deep_pages": deep_pages,
        "ocr_pages": ocr_pages,
        "text_layer_pages": text_layer_pages,
        "skipped_pages": skipped_pages,
        "cache_time_saved": cache_time_saved,
        "qa_score": qa_score,
        "top_actions": top_actions,
        "approved_rfis": approved_rfis,
        "accepted_quantities": accepted_quantities,
        "accepted_assumptions": accepted_assumptions,
        "submission_pack_ready": submission_pack_ready,
        "decision": decision,
        "readiness_score": readiness_score,
        "project_name": project_name or str(payload.get("project_id", "")),
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_int(value: Any) -> int:
    """Coerce value to int. Returns 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
