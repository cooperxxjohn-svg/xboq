"""
Narration Script Builder — generate a tight ~60s demo narration from payload.

Pure module, no Streamlit dependency. Can be tested independently.
Deterministic: same payload always produces the same narration.

Sprint 18: Final Demo Polish.
"""

from typing import Dict, List, Optional


def build_narration_script(
    payload: dict,
    project_name: str = "",
    highlights: Optional[List[dict]] = None,
) -> str:
    """Generate a ~60-second narration script for YC demo.

    Covers: project intro, page count, key findings (from highlights),
    counts (RFIs, quantities, blockers), time saved, deliverables.

    Args:
        payload: Full analysis payload dict (or empty dict).
        project_name: Human-readable project name.
        highlights: Pre-built highlights list (from build_highlights).
                    If None, builds internally.

    Returns:
        Multi-line narration script string. Deterministic.
    """
    if not isinstance(payload, dict):
        payload = {}

    # ── Build highlights if not provided ───────────────────────────────
    if highlights is None:
        try:
            from src.analysis.highlights import build_highlights
            highlights = build_highlights(payload)
        except Exception:
            highlights = []

    # ── Extract key numbers ────────────────────────────────────────────
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    page_count = _safe_int(overview.get("pages_total", 0))
    deep_pages = _safe_int(overview.get("pages_deep", 0))

    rfis = payload.get("rfis", []) if isinstance(payload.get("rfis"), list) else []
    rfi_count = len(rfis)
    approved_rfis = len([r for r in rfis if isinstance(r, dict) and r.get("status") == "approved"])

    blockers = payload.get("blockers", []) if isinstance(payload.get("blockers"), list) else []
    blocker_count = len(blockers)

    quantities = payload.get("quantities", []) if isinstance(payload.get("quantities"), list) else []
    qty_count = len(quantities)

    decision = str(payload.get("decision", "N/A"))
    readiness = _safe_int(payload.get("readiness_score", 0))

    qa_data = payload.get("qa_score", {})
    qa_score = _safe_int(qa_data.get("score", 0)) if isinstance(qa_data, dict) else 0

    timings = payload.get("timings", {})
    total_sec = _safe_int(timings.get("total_seconds", 0)) if isinstance(timings, dict) else 0

    name = project_name or str(payload.get("project_id", "this project"))

    # ── Build narration lines ──────────────────────────────────────────
    lines: List[str] = []

    # Section 1: Intro (5-8s)
    lines.append(f"[INTRO] This is {name}.")
    if page_count > 0:
        lines.append(
            f"A {page_count}-page tender document"
            + (f" with {deep_pages} pages deep-processed." if deep_pages > 0 else ".")
        )
    else:
        lines.append("A tender document uploaded for analysis.")

    # Section 2: Processing (5-8s)
    if total_sec > 0:
        lines.append(f"\n[PROCESSING] xBOQ analyzed this in {total_sec} seconds.")
    else:
        lines.append("\n[PROCESSING] xBOQ analyzed this automatically.")
    lines.append("Every page was read, classified, and cross-referenced.")

    # Section 3: Key findings from highlights (15-20s)
    lines.append("\n[FINDINGS]")
    if highlights:
        for hl in highlights[:4]:  # Cap at 4 most important
            label = hl.get("label", "")
            value = hl.get("value", "")
            detail = hl.get("detail", "")
            if label and value:
                line = f"- {label}: {value}"
                if detail:
                    line += f" ({detail})"
                lines.append(line)
    else:
        lines.append(f"- Decision: {decision} (score {readiness}/100)")
        if blocker_count > 0:
            lines.append(f"- {blocker_count} blocker(s) identified")
        lines.append(f"- {rfi_count} RFIs generated")

    # Section 4: Counts summary (10s)
    lines.append("\n[COUNTS]")
    lines.append(f"- {rfi_count} RFIs generated, {approved_rfis} approved")
    lines.append(f"- {qty_count} quantity line items extracted")
    if blocker_count > 0:
        top_blocker = blockers[0].get("title", "N/A") if isinstance(blockers[0], dict) else "N/A"
        lines.append(f"- {blocker_count} blockers — top: {top_blocker}")
    if qa_score > 0:
        lines.append(f"- QA score: {qa_score}/100")

    # Section 5: Deliverables (10s)
    lines.append("\n[DELIVERABLES]")
    lines.append("- Submission pack: 5-folder ZIP with all exports")
    lines.append("- Bid Summary PDF with branded cover page")
    lines.append("- RFI log, quantities, exclusions — all ready to send")

    # Section 6: Close (5s)
    lines.append(f"\n[CLOSE] That's xBOQ — bid risk in seconds, not days.")

    return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_int(value) -> int:
    """Coerce value to int. Returns 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
