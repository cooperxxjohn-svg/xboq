"""
Bid Summary Generator — produces 1-2 page markdown summary from analysis payload.

Pure function, no Streamlit dependency. Can be tested independently.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_bid_summary_markdown(
    payload: Dict[str, Any],
    bid_strategy: Optional[Dict[str, Any]] = None,
    assumptions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generate a markdown bid summary from analysis payload.

    All sections use .get() with defaults for graceful fallback
    when fields are missing (backward compatibility).

    Args:
        payload: Full analysis payload dict.
        bid_strategy: Optional output from compute_bid_strategy().
        assumptions: Optional list of assumption dicts with status fields.

    Returns:
        Markdown string (1-2 pages).
    """
    lines: List[str] = []

    # ── Header ──────────────────────────────────────────────────────
    project_id = payload.get("project_id", "Unknown")
    timestamp = payload.get("timestamp", datetime.now().isoformat())
    lines.append(f"# Bid Summary: {project_id}\n")
    lines.append(f"**Generated**: {timestamp}\n\n")

    # ── Document Coverage ───────────────────────────────────────────
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    lines.append("## Document Coverage\n\n")
    files = overview.get("files", ["N/A"])
    lines.append(f"- **Files**: {', '.join(files)}\n")
    lines.append(f"- **Total Pages**: {overview.get('pages_total', 0)}\n")
    disciplines = overview.get("disciplines_detected", [])
    if disciplines:
        lines.append(f"- **Disciplines**: {', '.join(disciplines)}\n")
    readiness = payload.get("readiness_score", 0)
    decision = payload.get("decision", "N/A")
    lines.append(f"- **Readiness Score**: {readiness}/100\n")
    lines.append(f"- **Decision**: {decision}\n\n")

    # ── Extracted Items ─────────────────────────────────────────────
    ext = payload.get("extraction_summary", {})
    counts = ext.get("counts", {})
    if counts:
        lines.append("## Extracted Items\n\n")
        lines.append("| Category | Count |\n|----------|-------|\n")
        for k, v in counts.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |\n")
        lines.append("\n")

    # ── Key Requirements ────────────────────────────────────────────
    reqs = ext.get("requirements", [])
    if reqs:
        lines.append(f"## Key Requirements ({len(reqs)} total)\n\n")
        for r in reqs[:10]:
            text = r.get("text", "")[:120]
            cat = r.get("category", "")
            if cat:
                lines.append(f"- [{cat}] {text}\n")
            else:
                lines.append(f"- {text}\n")
        if len(reqs) > 10:
            lines.append(f"\n*...and {len(reqs) - 10} more*\n")
        lines.append("\n")

    # ── Top RFIs ────────────────────────────────────────────────────
    rfis = payload.get("rfis", [])
    if rfis:
        lines.append(f"## Top RFIs ({len(rfis)} total)\n\n")
        lines.append("| # | Trade | Priority | Question |\n")
        lines.append("|---|-------|----------|----------|\n")
        for i, rfi in enumerate(rfis[:10], 1):
            trade = rfi.get("trade", "").title()
            pri = rfi.get("priority", "").upper()
            q = rfi.get("question", "")[:80]
            lines.append(f"| {i} | {trade} | {pri} | {q} |\n")
        if len(rfis) > 10:
            lines.append(f"\n*...and {len(rfis) - 10} more*\n")
        lines.append("\n")

    # ── Blockers ────────────────────────────────────────────────────
    blockers = payload.get("blockers", [])
    if blockers:
        lines.append(f"## Blockers ({len(blockers)})\n\n")
        for b in blockers[:8]:
            sev = b.get("severity", "").upper()
            title = b.get("title", "")[:100]
            trade = b.get("trade", "").title()
            lines.append(f"- **[{sev}]** {trade}: {title}\n")
        if len(blockers) > 8:
            lines.append(f"\n*...and {len(blockers) - 8} more*\n")
        lines.append("\n")

    # ── Addenda ─────────────────────────────────────────────────────
    addenda = payload.get("addendum_index", [])
    if addenda:
        lines.append(f"## Addenda ({len(addenda)})\n\n")
        for a in addenda:
            no = a.get("addendum_no", "?")
            date = a.get("date", "")
            title = a.get("title", "Untitled")
            date_str = f" ({date})" if date else ""
            lines.append(f"- **Addendum {no}**{date_str}: {title}\n")
            boq_ch = a.get("boq_changes", [])
            date_ch = a.get("date_changes", [])
            changes = a.get("changes", [])
            if boq_ch:
                lines.append(f"  - BOQ changes: {len(boq_ch)}\n")
            if date_ch:
                lines.append(f"  - Date changes: {len(date_ch)}\n")
            if changes:
                lines.append(f"  - Other changes: {len(changes)}\n")
        lines.append("\n")

    # ── Conflicts (Sprint 9: resolution labels) ─────────────────────
    conflicts = payload.get("conflicts", [])
    if conflicts:
        lines.append(f"## Conflicts ({len(conflicts)})\n\n")
        for c in conflicts[:10]:
            ctype = c.get("type", "")
            # Sprint 9: append (Revision) label for intentional revisions
            _suffix = " (Revision)" if c.get("resolution") == "intentional_revision" else ""
            if ctype == "boq_change":
                item_no = c.get("item_no", "?")
                n_changes = len(c.get("changes", []))
                lines.append(
                    f"- BOQ item {item_no}: {n_changes} field(s) changed "
                    f"(p.{(c.get('base_page') or 0) + 1} vs p.{(c.get('addendum_page') or 0) + 1}){_suffix}\n"
                )
            elif ctype == "boq_new_item":
                item_no = c.get("item_no", "?")
                desc = c.get("description", "")[:60]
                lines.append(f"- New BOQ item {item_no}: {desc}{_suffix}\n")
            elif ctype == "schedule_change":
                mark = c.get("mark", "?")
                n_changes = len(c.get("changes", []))
                lines.append(f"- Schedule {mark}: {n_changes} field(s) changed{_suffix}\n")
            elif ctype == "requirement_modified":
                sim = c.get("similarity", 0)
                lines.append(f"- Requirement modified ({sim:.0%} similar){_suffix}\n")
            elif ctype == "requirement_new":
                text = c.get("text", "")[:80]
                lines.append(f"- New requirement: {text}{_suffix}\n")
            else:
                lines.append(f"- {ctype}{_suffix}\n")
        if len(conflicts) > 10:
            lines.append(f"\n*...and {len(conflicts) - 10} more*\n")
        lines.append("\n")

    # ── Exclusions & Clarifications (Sprint 9) ─────────────────────
    if assumptions:
        rejected = [a for a in assumptions if a.get("status") == "rejected"]
        accepted = [a for a in assumptions if a.get("status") == "accepted"]
        if rejected or accepted:
            lines.append("## Exclusions & Clarifications\n\n")
            if rejected:
                lines.append("### Exclusions\n\n")
                for a in rejected:
                    title = a.get("title", "")[:100]
                    cost = a.get("cost_impact")
                    cost_str = f" [Cost: {cost}]" if cost is not None else ""
                    lines.append(f"- **{title}**{cost_str}\n")
                lines.append("\n")
            if accepted:
                lines.append("### Clarifications\n\n")
                for a in accepted:
                    title = a.get("title", "")[:100]
                    cost = a.get("cost_impact")
                    cost_str = f" [Cost: {cost}]" if cost is not None else ""
                    lines.append(f"- **{title}**{cost_str}\n")
                lines.append("\n")

    # ── Multi-Document Listing (Sprint 9) ──────────────────────────
    mdi_data = payload.get("multi_doc_index")
    if mdi_data and len(mdi_data.get("docs", [])) > 1:
        lines.append("## Document Set\n\n")
        for doc in mdi_data["docs"]:
            lines.append(
                f"- **{doc.get('filename', '?')}** — {doc.get('page_count', 0)} pages\n"
            )
        lines.append("\n")

    # ── Bid Strategy ────────────────────────────────────────────────
    if bid_strategy:
        lines.append("## Bid Strategy Assessment\n\n")
        for key in ("client_fit", "risk_score", "competition_score", "readiness_score"):
            dial = bid_strategy.get(key, {})
            name = dial.get("name", key.replace("_", " ").title())
            score = dial.get("score")
            conf = dial.get("confidence", "")
            based_on = dial.get("based_on", [])
            if score is not None:
                lines.append(f"- **{name}**: {score}/100 ({conf})\n")
            else:
                lines.append(f"- **{name}**: Not computed\n")
            if based_on:
                for reason in based_on[:3]:
                    lines.append(f"  - {reason}\n")

        recs = bid_strategy.get("recommendations", [])
        if recs:
            lines.append("\n**Recommendations**:\n\n")
            for r in recs:
                lines.append(f"- {r}\n")
        lines.append("\n")

    # ── Bid Pack QA Score (Sprint 10) ─────────────────────────────
    qa_score = payload.get("qa_score")
    if qa_score and isinstance(qa_score, dict):
        score = qa_score.get("score", 0)
        confidence = qa_score.get("confidence", "")
        lines.append(f"## Bid Pack QA Score: {score}/100 ({confidence})\n\n")

        breakdown = qa_score.get("breakdown", {})
        if breakdown:
            lines.append("| Component | Score |\n|-----------|-------|\n")
            for comp_key, comp_val in breakdown.items():
                label = comp_key.replace("_", " ").title()
                lines.append(f"| {label} | {comp_val}/20 |\n")
            lines.append("\n")

        top_actions = qa_score.get("top_actions", [])
        if top_actions:
            lines.append("**Top Actions to Improve:**\n\n")
            for action in top_actions[:5]:
                lines.append(f"- {action}\n")
            lines.append("\n")

    # ── Pricing Guidance (Sprint 11) ──────────────────────────────
    pricing = payload.get("pricing_guidance")
    if pricing and isinstance(pricing, dict):
        lines.append("## Pricing Guidance\n\n")

        # Contingency range
        cont = pricing.get("contingency_range", {})
        if cont:
            lines.append("### Contingency Range\n\n")
            lines.append("| Metric | Value |\n|--------|-------|\n")
            lines.append(f"| Low | {cont.get('low_pct', 0)}% |\n")
            lines.append(f"| High | {cont.get('high_pct', 0)}% |\n")
            lines.append(f"| **Recommended** | **{cont.get('recommended_pct', 0)}%** |\n")
            lines.append("\n")
            rationale = cont.get("rationale", "")
            if rationale:
                lines.append(f"*{rationale}*\n\n")

        # Recommended Exclusions
        excl = pricing.get("recommended_exclusions", [])
        if excl:
            lines.append("### Recommended Exclusions\n\n")
            for e in excl:
                lines.append(f"- {e}\n")
            lines.append("\n")

        # Recommended Clarifications
        clar = pricing.get("recommended_clarifications", [])
        if clar:
            lines.append("### Recommended Clarifications\n\n")
            for c in clar:
                lines.append(f"- {c}\n")
            lines.append("\n")

        # Suggested Alternates / VE
        ve = pricing.get("suggested_alternates_ve", [])
        if ve:
            lines.append("### Suggested Alternates / Value Engineering\n\n")
            for v in ve:
                item = v.get("item", "")
                reason = v.get("reason", "")
                lines.append(f"- **{item}**\n")
                if reason:
                    lines.append(f"  - {reason}\n")
            lines.append("\n")

    # ── Estimating Playbook / Basis of Estimate (Sprint 20C) ────────
    _ep = payload.get("estimating_playbook")
    if _ep and isinstance(_ep, dict):
        try:
            from src.analysis.estimating_playbook import summarize_playbook_for_exports
            _ep_summary = summarize_playbook_for_exports(_ep)
            if _ep_summary:
                lines.append("## Basis of Estimate\n\n")
                lines.append(f"{_ep_summary}\n\n")

            # Basis of recommendation from pricing guidance
            _bor = (pricing or {}).get("basis_of_recommendation", [])
            if _bor:
                lines.append("**Contingency Basis:**\n\n")
                for _b in _bor:
                    lines.append(f"- {_b}\n")
                lines.append("\n")
        except Exception:
            pass

    # ── Footer ──────────────────────────────────────────────────────
    lines.append("---\n\n")
    lines.append("*Generated by xBOQ Bid Engineer*\n")

    return "".join(lines)
