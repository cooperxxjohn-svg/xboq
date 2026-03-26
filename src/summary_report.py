"""
One-Page Executive Summary Report Generator

Sprint 26: Generates a clean, printable markdown summary from analysis.json
that a contractor can take to a pre-bid meeting.

Usage:
    # CLI
    python -m src.summary_report --input out/sonipat/analysis.json

    # Python
    from src.summary_report import generate_summary
    md = generate_summary(payload)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_summary(payload: dict) -> str:
    """
    Generate a one-page executive summary from an analysis payload.

    Returns:
        Markdown string suitable for printing / display.
    """
    project_id = payload.get("project_id", "Unknown Project")
    decision = payload.get("decision", "UNKNOWN")
    score = payload.get("readiness_score", 0)
    sub_scores = payload.get("sub_scores", {})
    blockers = payload.get("blockers", [])
    rfis = payload.get("rfis", [])
    commercial_terms = payload.get("commercial_terms", [])
    pricing = payload.get("pricing_guidance", {})
    timings = payload.get("timings", {})
    processing_stats = payload.get("processing_stats", {})
    overview = payload.get("drawing_overview", {})
    pi = payload.get("diagnostics", {}).get("page_index", {})
    plan_graph = payload.get("diagnostics", {}).get("plan_graph", {}) or {}
    es = payload.get("extraction_summary", {})
    counts = es.get("counts", {}) if isinstance(es, dict) else {}
    req_by_trade = payload.get("requirements_by_trade", {})
    trade_coverage = payload.get("trade_coverage", [])
    timestamp = payload.get("timestamp", datetime.now().isoformat())

    # ── Header
    lines = []
    lines.append(f"# Tender Analysis: {project_id}")
    lines.append(f"**Generated:** {timestamp[:19].replace('T', ' ')}")
    lines.append("")

    # ── Decision badge
    badge = {"PASS": "GO", "CONDITIONAL": "CONDITIONAL", "NO-GO": "NO-GO",
             "NO_DRAWINGS": "NO DRAWINGS"}.get(decision, decision)
    lines.append(f"## Decision: {badge} ({score}/100)")
    lines.append("")

    # Sub-scores
    if sub_scores:
        parts = []
        for key in ["completeness", "coverage", "measurement", "blocker"]:
            val = sub_scores.get(key)
            if val is not None:
                parts.append(f"{key.title()}: {val}")
        if parts:
            lines.append(f"**Sub-scores:** {' | '.join(parts)}")
            lines.append("")

    # ── Drawing Set Overview
    total_pages = pi.get("total_pages", overview.get("total_pages", 0))
    disciplines = overview.get("disciplines_detected", [])
    counts_by_type = pi.get("counts_by_type", {})

    lines.append("## Drawing Set")
    lines.append("")
    lines.append(f"- **Pages:** {total_pages}")
    if disciplines:
        lines.append(f"- **Disciplines:** {', '.join(disciplines)}")

    # Page type summary (compact)
    if counts_by_type:
        major_types = [(k, v) for k, v in sorted(counts_by_type.items(), key=lambda x: -x[1]) if v > 0]
        type_str = ", ".join(f"{k} ({v})" for k, v in major_types[:6])
        lines.append(f"- **Page types:** {type_str}")

    # Plan graph summary
    doors = plan_graph.get("all_door_tags", [])
    windows = plan_graph.get("all_window_tags", [])
    rooms = plan_graph.get("all_room_names", [])
    if doors or windows or rooms:
        parts = []
        if doors:
            parts.append(f"{len(doors)} door types ({', '.join(doors[:5])})")
        if windows:
            parts.append(f"{len(windows)} window types ({', '.join(windows[:5])})")
        if rooms:
            parts.append(f"{len(rooms)} room types")
        lines.append(f"- **Detected:** {'; '.join(parts)}")

    scale_with = plan_graph.get("pages_with_scale", overview.get("pages_with_scale", 0))
    scale_without = plan_graph.get("pages_without_scale", overview.get("pages_without_scale", 0))
    if scale_with or scale_without:
        total_scale = scale_with + scale_without
        if total_scale > 0:
            pct = scale_with / total_scale * 100
            lines.append(f"- **Scale coverage:** {scale_with}/{total_scale} pages ({pct:.0f}%)")
    lines.append("")

    # ── Extraction Summary
    boq_items = counts.get("boq_items", 0)
    req_count = counts.get("requirements", 0)
    sched_count = counts.get("schedules", 0)
    comm_count = len(commercial_terms) if isinstance(commercial_terms, list) else counts.get("commercial_terms", 0)

    lines.append("## Extraction Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| BOQ items | {boq_items} |")
    lines.append(f"| Requirements/specs | {req_count} |")
    lines.append(f"| Schedule rows | {sched_count} |")
    lines.append(f"| Commercial terms | {comm_count} |")
    lines.append(f"| RFIs generated | {len(rfis)} |")
    lines.append(f"| Blockers | {len(blockers)} |")
    lines.append("")

    # ── Critical Blockers (top 5)
    high_blockers = [b for b in blockers if b.get("severity", "").lower() in ("high", "critical")]
    med_blockers = [b for b in blockers if b.get("severity", "").lower() == "medium"]

    if high_blockers or med_blockers:
        lines.append("## Blockers")
        lines.append("")
        for b in (high_blockers + med_blockers)[:5]:
            sev = b.get("severity", "MEDIUM").upper()
            title = b.get("title", "Blocker")
            fix = b.get("fix_actions", "")
            if isinstance(fix, list):
                fix = fix[0] if fix else ""
            line = f"- **[{sev}]** {title}"
            if fix:
                line += f" — *{fix[:100]}*"
            lines.append(line)
        if len(high_blockers) + len(med_blockers) > 5:
            lines.append(f"- *...and {len(high_blockers) + len(med_blockers) - 5} more*")
        lines.append("")

    # ── RFIs by Trade (compact)
    if rfis:
        rfi_by_trade = {}
        for r in rfis:
            t = r.get("trade", "general")
            rfi_by_trade[t] = rfi_by_trade.get(t, 0) + 1

        lines.append("## RFIs by Trade")
        lines.append("")
        lines.append("| Trade | Count |")
        lines.append("|-------|-------|")
        for trade, count in sorted(rfi_by_trade.items(), key=lambda x: -x[1]):
            lines.append(f"| {trade} | {count} |")
        lines.append("")

    # ── Commercial Terms (compact)
    if commercial_terms and isinstance(commercial_terms, list):
        lines.append("## Key Commercial Terms")
        lines.append("")
        for t in commercial_terms[:8]:
            if not isinstance(t, dict):
                continue
            term_type = t.get("term_type", "unknown")
            value = t.get("value", "")
            unit = t.get("unit", "")
            label = term_type.replace("_", " ").title()
            lines.append(f"- **{label}:** {value} {unit}".strip())
        lines.append("")

    # ── Trade Coverage
    if trade_coverage:
        lines.append("## Trade Coverage")
        lines.append("")
        lines.append("| Trade | Coverage | Status |")
        lines.append("|-------|----------|--------|")
        for tc in trade_coverage:
            trade = tc.get("trade", "?")
            cov = tc.get("coverage_pct", 0)
            if cov >= 80:
                status = "OK"
            elif cov >= 50:
                status = "PARTIAL"
            else:
                status = "BLOCKED"
            lines.append(f"| {trade} | {cov:.0f}% | {status} |")
        lines.append("")

    # ── Pricing Guidance
    if pricing and isinstance(pricing, dict):
        contingency = pricing.get("contingency_range", {})
        rec = contingency.get("recommended_pct",
              pricing.get("recommended_contingency_pct"))
        if rec:
            lines.append("## Pricing Guidance")
            lines.append("")
            lines.append(f"- **Recommended contingency:** {rec}%")
            reason = contingency.get("reason", pricing.get("contingency_reason", ""))
            if reason:
                lines.append(f"- **Reason:** {reason}")
            lines.append("")

    # ── Recommended Next Steps (synthesized)
    lines.append("## Recommended Next Steps")
    lines.append("")
    step_num = 1
    if high_blockers:
        lines.append(f"{step_num}. Resolve {len(high_blockers)} HIGH-severity blocker(s)")
        step_num += 1
    if rfis:
        lines.append(f"{step_num}. Send {len(rfis)} RFIs to architect/client for clarification")
        step_num += 1
    if med_blockers:
        lines.append(f"{step_num}. Address {len(med_blockers)} MEDIUM-severity issue(s)")
        step_num += 1
    if req_count > 0:
        lines.append(f"{step_num}. Review {req_count} specification requirements for compliance")
        step_num += 1
    if pricing and isinstance(pricing, dict):
        lines.append(f"{step_num}. Apply contingency and prepare pricing estimate")
        step_num += 1
    if step_num == 1:
        lines.append("1. Review complete analysis and proceed with bid preparation")
    lines.append("")

    # ── Processing Info (footer)
    total_time = timings.get("total_s", 0)
    if total_time:
        lines.append("---")
        lines.append(f"*Analysis completed in {total_time:.1f}s*")
        lines.append("")

    return "\n".join(lines)


def generate_rfi_csv(payload: dict) -> str:
    """Generate a CSV export of all RFIs."""
    rfis = payload.get("rfis", [])
    if not rfis:
        return "trade,priority,question,why_it_matters,pages\n"

    lines = ["trade,priority,confidence,question,why_it_matters,pages"]
    for r in rfis:
        trade = r.get("trade", "general")
        priority = r.get("priority", "MEDIUM")
        question = r.get("question", r.get("rfi_text", "")).replace('"', '""')
        why = r.get("why_it_matters", "").replace('"', '""')
        ev = r.get("evidence", {})
        # Confidence lives in evidence dict
        confidence = ev.get("confidence", 0) if isinstance(ev, dict) else 0
        pages = ""
        if isinstance(ev, dict):
            page_list = ev.get("pages", [])
            pages = ";".join(str(p) for p in page_list[:10])
        lines.append(f'"{trade}","{priority}","{confidence}","{question}","{why}","{pages}"')

    return "\n".join(lines)


def generate_blocker_csv(payload: dict) -> str:
    """Generate a CSV export of all blockers."""
    blockers = payload.get("blockers", [])
    if not blockers:
        return "severity,trade,title,description,fix_actions\n"

    lines = ["severity,trade,title,description,fix_actions"]
    for b in blockers:
        sev = b.get("severity", "MEDIUM")
        trade = b.get("trade", "general")
        title = b.get("title", "").replace('"', '""')
        desc = b.get("description", "").replace('"', '""')[:200]
        fix = b.get("fix_actions", "")
        if isinstance(fix, list):
            fix = "; ".join(fix)
        fix = fix.replace('"', '""')[:200]
        lines.append(f'"{sev}","{trade}","{title}","{desc}","{fix}"')

    return "\n".join(lines)


def save_exports(payload: dict, output_dir: str) -> List[str]:
    """
    Generate and save all export files to the output directory.

    Returns:
        List of file paths written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. Executive summary
    summary_md = generate_summary(payload)
    summary_path = out / "summary.md"
    summary_path.write_text(summary_md)
    written.append(str(summary_path))

    # 2. RFI CSV
    rfi_csv = generate_rfi_csv(payload)
    rfi_dir = out / "rfi"
    rfi_dir.mkdir(exist_ok=True)
    rfi_path = rfi_dir / "rfis.csv"
    rfi_path.write_text(rfi_csv)
    written.append(str(rfi_path))

    # 3. RFI JSON
    rfis = payload.get("rfis", [])
    rfi_json_path = rfi_dir / "rfis.json"
    rfi_json_path.write_text(json.dumps(rfis, indent=2, default=str))
    written.append(str(rfi_json_path))

    # 4. Blocker CSV
    blocker_csv = generate_blocker_csv(payload)
    blocker_path = out / "blockers.csv"
    blocker_path.write_text(blocker_csv)
    written.append(str(blocker_path))

    # 5. Commercial terms JSON
    terms = payload.get("commercial_terms", [])
    if terms:
        terms_path = out / "commercial_terms.json"
        terms_path.write_text(json.dumps(terms, indent=2, default=str))
        written.append(str(terms_path))

    # 6. Bid gate report (decision doc)
    gate_md = _generate_bid_gate_report(payload)
    gate_path = out / "bid_gate_report.md"
    gate_path.write_text(gate_md)
    written.append(str(gate_path))

    return written


def _generate_bid_gate_report(payload: dict) -> str:
    """Generate a bid gate GO / NO-GO report."""
    project_id = payload.get("project_id", "Unknown")
    decision = payload.get("decision", "UNKNOWN")
    score = payload.get("readiness_score", 0)
    sub_scores = payload.get("sub_scores", {})
    blockers = payload.get("blockers", [])
    rfis = payload.get("rfis", [])

    lines = []
    lines.append(f"# Bid Gate Report: {project_id}")
    lines.append("")
    lines.append(f"## Verdict: {decision}")
    lines.append(f"**Readiness Score:** {score}/100")
    lines.append("")

    if sub_scores:
        lines.append("### Score Breakdown")
        lines.append("")
        for k, v in sub_scores.items():
            bar_len = int(v / 5)  # 0-20 chars
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"- **{k.title()}:** [{bar}] {v}/100")
        lines.append("")

    # Risk assessment
    high_count = sum(1 for b in blockers if b.get("severity", "").lower() in ("high", "critical"))
    med_count = sum(1 for b in blockers if b.get("severity", "").lower() == "medium")
    low_count = sum(1 for b in blockers if b.get("severity", "").lower() == "low")

    lines.append("### Risk Summary")
    lines.append("")
    lines.append(f"| Severity | Count |")
    lines.append(f"|----------|-------|")
    lines.append(f"| HIGH / CRITICAL | {high_count} |")
    lines.append(f"| MEDIUM | {med_count} |")
    lines.append(f"| LOW | {low_count} |")
    lines.append(f"| **Total blockers** | **{len(blockers)}** |")
    lines.append(f"| RFIs to send | {len(rfis)} |")
    lines.append("")

    # Decision rationale
    lines.append("### Decision Rationale")
    lines.append("")
    if decision == "PASS":
        lines.append("All critical dependencies are met. Proceed with bid preparation.")
    elif decision == "CONDITIONAL":
        lines.append("Bid can proceed **conditionally** with the following actions required:")
        lines.append("")
        for b in blockers[:5]:
            if b.get("severity", "").lower() in ("high", "critical"):
                lines.append(f"- {b.get('title', 'Issue')}")
    elif decision == "NO-GO":
        lines.append("**Critical gaps prevent bid submission.** Resolve the following:")
        lines.append("")
        for b in blockers[:5]:
            lines.append(f"- {b.get('title', 'Issue')}")
    elif decision == "NO_DRAWINGS":
        lines.append("**No construction drawings detected.** Upload drawing PDFs to proceed.")
    else:
        lines.append(f"Decision: {decision}")
    lines.append("")

    return "\n".join(lines)


# ── CLI ──

def cmd_summary(args) -> int:
    """CLI handler for summary command."""
    from src.ask_tender import load_payload
    payload = load_payload(args.input)

    if getattr(args, "export_dir", None):
        files = save_exports(payload, args.export_dir)
        print(f"Exported {len(files)} files to {args.export_dir}/:")
        for f in files:
            print(f"  - {Path(f).name}")
    else:
        md = generate_summary(payload)
        print(md)

    return 0


def main():
    """Standalone CLI."""
    import argparse
    parser = argparse.ArgumentParser(description="One-page executive summary")
    parser.add_argument("--input", "-i", required=True, help="Path to analysis.json")
    parser.add_argument("--export-dir", "-o", help="Output directory for all exports")
    args = parser.parse_args()
    return cmd_summary(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
