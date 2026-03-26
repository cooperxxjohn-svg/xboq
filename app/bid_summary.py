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
    readiness = payload.get("readiness_score", 0)
    decision = payload.get("decision", "N/A")
    lines.append(f"# Bid Summary: {project_id}\n")
    lines.append(f"**Decision: {decision}** | Score: {readiness}/100 | {timestamp}\n\n")

    # ── Tender at a Glance (Sprint 22) ────────────────────────────
    _ps = payload.get("processing_stats") or {}
    _ext_glance = payload.get("extraction_summary") or {}
    _ext_counts = _ext_glance.get("counts", {})
    _boq_count = _ext_counts.get("boq_items", 0)
    _rfi_count = len(payload.get("rfis", []))
    _blocker_count = len(payload.get("blockers", []))
    _pages = _ps.get("total_pages", 0)
    _files = len(payload.get("file_info", []))
    _boq_src = payload.get("boq_source", "pdf")
    _comm_count = len(payload.get("commercial_terms", []))

    # Build concise glance line
    _glance_parts = []
    if _files:
        _glance_parts.append(f"{_files} files, {_pages} pages")
    if _boq_count:
        _glance_parts.append(f"{_boq_count} BOQ items ({_boq_src})")
    if _rfi_count:
        _glance_parts.append(f"{_rfi_count} RFIs")
    if _blocker_count:
        _glance_parts.append(f"{_blocker_count} blockers")
    if _comm_count:
        _glance_parts.append(f"{_comm_count} commercial terms")

    # Top risk (first blocker by severity)
    _blockers_sorted = sorted(
        payload.get("blockers", []),
        key=lambda b: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(
            (b.get("severity") or "").upper(), 3
        ),
    )
    _top_risk = ""
    if _blockers_sorted:
        _tr = _blockers_sorted[0]
        _top_risk = f" | Top risk: {(_tr.get('title') or '')[:60]}"

    # Pricing guidance contingency
    _pg = payload.get("pricing_guidance") or {}
    _cont = _pg.get("contingency_range", {})
    _cont_str = ""
    if _cont.get("recommended_pct"):
        _cont_str = f" | Contingency: {_cont['recommended_pct']}%"

    if _glance_parts:
        lines.append("> **At a Glance**: " + " | ".join(_glance_parts) + _top_risk + _cont_str + "\n\n")

    # ── Document Coverage ───────────────────────────────────────────
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    lines.append("## Document Coverage\n\n")
    files = overview.get("files", ["N/A"])
    lines.append(f"- **Files**: {', '.join(files)}\n")
    lines.append(f"- **Total Pages**: {overview.get('pages_total', 0)}\n")
    disciplines = overview.get("disciplines_detected", [])
    if disciplines:
        lines.append(f"- **Disciplines**: {', '.join(disciplines)}\n")
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

    # ── Key Cost Drivers (Sprint 22) ─────────────────────────────────
    boq_items = ext.get("boq_items", [])
    boq_source = payload.get("boq_source", "pdf")
    if boq_items:
        # Compute amounts and sort
        priced_items = []
        for item in boq_items:
            qty = item.get("qty", 0) or 0
            rate = item.get("rate", 0) or 0
            try:
                amt = float(qty) * float(rate)
            except (TypeError, ValueError):
                amt = 0.0
            priced_items.append({**item, "_amount": amt})

        total_value = sum(it["_amount"] for it in priced_items)
        priced_items.sort(key=lambda x: x["_amount"], reverse=True)

        # Trade breakdown
        trade_totals: Dict[str, float] = {}
        trade_counts: Dict[str, int] = {}
        for item in priced_items:
            trade = (item.get("trade") or "general").title()
            trade_totals[trade] = trade_totals.get(trade, 0) + item["_amount"]
            trade_counts[trade] = trade_counts.get(trade, 0) + 1

        lines.append(f"## Key Cost Drivers ({len(boq_items)} BOQ items, source: {boq_source})\n\n")

        # Top 10 items by value
        top_items = [it for it in priced_items[:10] if it["_amount"] > 0]
        if top_items and total_value > 0:
            lines.append("### Top Items by Value\n\n")
            lines.append("| # | Description | Unit | Qty | Rate | Amount | % |\n")
            lines.append("|---|-------------|------|-----|------|--------|---|\n")
            for i, item in enumerate(top_items, 1):
                desc = (item.get("description") or "")[:50]
                unit = item.get("unit", "")
                qty = item.get("qty", 0)
                rate = item.get("rate", 0)
                amt = item["_amount"]
                pct = (amt / total_value * 100) if total_value > 0 else 0
                lines.append(
                    f"| {i} | {desc} | {unit} | {qty:,.1f} | {rate:,.0f} | {amt:,.0f} | {pct:.1f}% |\n"
                )
            lines.append("\n")

            # Concentration warning
            if top_items[0]["_amount"] > total_value * 0.30:
                desc = (top_items[0].get("description") or "")[:40]
                lines.append(
                    f"**Warning**: Top item ({desc}) represents "
                    f">{top_items[0]['_amount'] / total_value * 100:.0f}% of total — verify quantity independently.\n\n"
                )

        # Trade breakdown table
        if trade_totals:
            sorted_trades = sorted(trade_totals.items(), key=lambda x: x[1], reverse=True)
            lines.append("### Trade Breakdown\n\n")
            lines.append("| Trade | Items | Value | % |\n")
            lines.append("|-------|-------|-------|---|\n")
            for trade, value in sorted_trades:
                count = trade_counts.get(trade, 0)
                pct = (value / total_value * 100) if total_value > 0 else 0
                lines.append(f"| {trade} | {count} | {value:,.0f} | {pct:.1f}% |\n")
            lines.append(f"| **Total** | **{len(boq_items)}** | **{total_value:,.0f}** | **100%** |\n")
            lines.append("\n")

        # Sprint 22: BOQ completeness indicator
        expected_trades = {"civil", "structural", "architectural", "finishes",
                          "electrical", "plumbing", "hvac", "fire"}
        covered_trades = {(item.get("trade") or "general").lower() for item in boq_items}
        covered_expected = expected_trades & covered_trades
        missing_trades = expected_trades - covered_trades
        if covered_expected:
            lines.append(
                f"**BOQ Coverage**: {len(covered_expected)}/{len(expected_trades)} "
                f"expected trades have BOQ items.\n"
            )
            if missing_trades:
                lines.append(
                    f"Missing trades: {', '.join(sorted(t.title() for t in missing_trades))} "
                    f"— verify if in scope or add to exclusions.\n"
                )
            lines.append("\n")

    # ── Commercial Terms (Sprint 22) ─────────────────────────────
    commercial_terms = payload.get("commercial_terms", [])
    if commercial_terms:
        lines.append(f"## Commercial Terms ({len(commercial_terms)})\n\n")
        lines.append("| Term | Value | Unit | Confidence |\n")
        lines.append("|------|-------|------|------------|\n")
        for term in commercial_terms:
            ttype = (term.get("term_type") or "").replace("_", " ").title()
            tval = term.get("value", "")
            tunit = term.get("unit", "")
            tconf = term.get("confidence", 0)
            lines.append(f"| {ttype} | {tval} | {tunit} | {tconf:.0%} |\n")
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

    # ── Exclusions & Clarifications (Sprint 9 + Sprint 22 enhancements) ─────
    has_exclusions = False
    if assumptions:
        rejected = [a for a in assumptions if a.get("status") == "rejected"]
        accepted = [a for a in assumptions if a.get("status") == "accepted"]
        if rejected or accepted:
            has_exclusions = True
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

    # Sprint 22: Auto-generated scope gap exclusions
    _scope_gaps = []
    # Missing BOQ trades
    _ext_excl = payload.get("extraction_summary") or {}
    _boq_excl = _ext_excl.get("boq_items", [])
    if _boq_excl:
        _covered = {(it.get("trade") or "general").lower() for it in _boq_excl}
        _expected = {"civil", "structural", "architectural", "finishes",
                     "electrical", "plumbing", "hvac", "fire"}
        _missing = _expected - _covered
        for t in sorted(_missing):
            _scope_gaps.append(f"{t.title()} works — no BOQ items found for this trade")
    # Missing schedules
    _scheds = _ext_excl.get("schedules", {})
    if isinstance(_scheds, dict):
        if not _scheds.get("doors"):
            _scope_gaps.append("Door schedule — not extracted, verify if in scope")
        if not _scheds.get("windows"):
            _scope_gaps.append("Window schedule — not extracted, verify if in scope")
    # Missing commercial terms
    _terms = payload.get("commercial_terms", [])
    _term_types = {t.get("term_type") for t in _terms if isinstance(t, dict)}
    for _key_term, _label in [
        ("ld_clause", "Liquidated damages clause"),
        ("retention", "Retention clause"),
        ("warranty_dlp", "Warranty / DLP clause"),
        ("emd_bid_security", "EMD / bid security"),
        ("performance_bond", "Performance bank guarantee"),
    ]:
        if _key_term not in _term_types:
            _scope_gaps.append(f"{_label} — not found in tender documents, clarify with client")

    if _scope_gaps:
        if not has_exclusions:
            lines.append("## Exclusions & Clarifications\n\n")
        lines.append("### Suggested Scope Clarifications\n\n")
        lines.append("*Auto-detected gaps — review and confirm before including in bid:*\n\n")
        for gap in _scope_gaps:
            lines.append(f"- {gap}\n")
        lines.append("\n")

    # ── Quantity Reconciliation (Sprint 22 Phase 4) ─────────────────
    _qty_recon = payload.get("quantity_reconciliation", [])
    if _qty_recon and isinstance(_qty_recon, list):
        # Filter to rows with actual data
        _recon_with_data = [
            r for r in _qty_recon
            if isinstance(r, dict) and (
                r.get("schedule_count") is not None or
                r.get("boq_count") is not None or
                r.get("drawing_count") is not None
            )
        ]
        if _recon_with_data:
            _mismatches = [r for r in _recon_with_data if r.get("mismatch")]
            lines.append(f"## Quantity Reconciliation ({len(_recon_with_data)} items")
            if _mismatches:
                lines.append(f", {len(_mismatches)} discrepancies")
            lines.append(")\n\n")

            lines.append("| Category | Schedule | BOQ | Drawing | Delta | Status |\n")
            lines.append("|----------|----------|-----|---------|-------|--------|\n")
            for r in _recon_with_data:
                cat = r.get("category", "").title()
                sched = r.get("schedule_count")
                boq = r.get("boq_count")
                draw = r.get("drawing_count")
                delta = r.get("delta_pct")
                mismatch = r.get("mismatch", False)

                sched_str = str(sched) if sched is not None else "-"
                boq_str = str(boq) if boq is not None else "-"
                draw_str = str(draw) if draw is not None else "-"
                delta_str = f"{delta:+.1f}%" if delta is not None else "-"
                status = "MISMATCH" if mismatch else "OK"

                lines.append(
                    f"| {cat} | {sched_str} | {boq_str} | {draw_str} | {delta_str} | {status} |\n"
                )
            lines.append("\n")

            if _mismatches:
                lines.append("**Discrepancies > 15% require verification:**\n\n")
                for r in _mismatches:
                    notes = r.get("notes", "")
                    if notes:
                        lines.append(f"- {notes}\n")
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
