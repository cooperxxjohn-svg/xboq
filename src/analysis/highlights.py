"""
Highlights — extract top-level executive insights from analysis payload.

Produces a short, ordered list of "highlight cards" for the demo dashboard.
Each highlight has: icon, label, value, severity, detail.

Pure module, no Streamlit dependency. Can be tested independently.

Sprint 17: Demo Hardening + Scripted Dataset.
"""

from collections import Counter
from typing import List


def build_highlights(payload: dict) -> List[dict]:
    """
    Extract executive highlights from an analysis payload.

    Args:
        payload: Full analysis payload dict (same as analysis.json).

    Returns:
        Ordered list of highlight dicts:
        [{
            "icon": str,      # emoji
            "label": str,     # short label
            "value": str,     # primary value text
            "severity": str,  # "good" | "warn" | "bad" — for color coding
            "detail": str,    # one-line supporting detail
        }]
    """
    highlights = []

    # 1. Decision / Readiness
    decision = payload.get("decision", "UNKNOWN")
    score = payload.get("readiness_score", 0)
    _sev = {"PASS": "good", "CONDITIONAL": "warn", "NO-GO": "bad"}.get(decision, "warn")
    highlights.append({
        "icon": _decision_icon(decision),
        "label": "Bid Readiness",
        "value": f"{score}/100 — {decision}",
        "severity": _sev,
        "detail": _decision_detail(payload),
    })

    # 2. Blockers
    blockers = payload.get("blockers", [])
    critical = [b for b in blockers if b.get("severity") == "high"]
    if blockers:
        highlights.append({
            "icon": "\U0001f6ab",
            "label": "Blockers",
            "value": f"{len(critical)} critical, {len(blockers)} total",
            "severity": "bad" if critical else "warn",
            "detail": critical[0].get("title", "")[:80] if critical else "",
        })

    # 3. RFIs
    rfis = payload.get("rfis", [])
    approved = [r for r in rfis if r.get("status") == "approved"]
    if rfis:
        highlights.append({
            "icon": "\U0001f4cb",
            "label": "RFIs",
            "value": f"{len(rfis)} generated ({len(approved)} approved)",
            "severity": "warn" if len(rfis) > 5 else "good",
            "detail": f"Top trade: {_top_rfi_trade(rfis)}" if rfis else "",
        })

    # 4. Trade Coverage
    coverage = payload.get("trade_coverage", [])
    if coverage:
        ready_count = sum(1 for t in coverage if t.get("coverage_pct", 0) >= 80)
        total_trades = len(coverage)
        highlights.append({
            "icon": "\U0001f4ca",
            "label": "Trade Coverage",
            "value": f"{ready_count}/{total_trades} trades ready",
            "severity": "good" if ready_count == total_trades else "warn",
            "detail": _worst_trade(coverage),
        })

    # 5. Conflicts
    conflicts = payload.get("conflicts", [])
    unresolved = [c for c in conflicts
                  if c.get("resolution") != "intentional_revision"]
    if unresolved:
        highlights.append({
            "icon": "\u26a0\ufe0f",
            "label": "Conflicts",
            "value": f"{len(unresolved)} unresolved",
            "severity": "bad" if len(unresolved) > 3 else "warn",
            "detail": "",
        })

    # 6. QA Score
    qa = payload.get("qa_score")
    if qa and isinstance(qa, dict):
        qa_val = qa.get("score", 0)
        highlights.append({
            "icon": "\u2705",
            "label": "QA Score",
            "value": f"{qa_val}/100",
            "severity": "good" if qa_val >= 70 else ("warn" if qa_val >= 40 else "bad"),
            "detail": "; ".join(qa.get("top_actions", [])[:2]),
        })

    # 7. Risk Items
    risk_results = payload.get("risk_results", [])
    high_risks = [r for r in risk_results
                  if r.get("impact") == "high" and r.get("found")]
    if high_risks:
        highlights.append({
            "icon": "\U0001f534",
            "label": "Risk Items",
            "value": f"{len(high_risks)} high-impact risks found",
            "severity": "bad",
            "detail": high_risks[0].get("label", "")[:60] if high_risks else "",
        })

    # 8. Quantities
    quantities = payload.get("quantities", [])
    priced = [q for q in quantities if q.get("rate") is not None]
    if quantities:
        highlights.append({
            "icon": "\U0001f4b0",
            "label": "Quantities",
            "value": f"{len(quantities)} items ({len(priced)} with rates)",
            "severity": "good" if len(priced) > len(quantities) * 0.5 else "warn",
            "detail": "",
        })

    return highlights


# ── Helpers ─────────────────────────────────────────────────────────────


def _decision_icon(decision: str) -> str:
    return {
        "PASS": "\u2705",
        "CONDITIONAL": "\u26a0\ufe0f",
        "NO-GO": "\U0001f6ab",
    }.get(decision, "\u2753")


def _decision_detail(payload: dict) -> str:
    sub = payload.get("sub_scores", {})
    if not sub:
        return ""
    weakest = min(sub, key=lambda k: sub[k]) if sub else ""
    return f"Weakest: {weakest} ({sub.get(weakest, '?')}/100)" if weakest else ""


def _top_rfi_trade(rfis: list) -> str:
    trades = Counter(r.get("trade", "general") for r in rfis)
    if trades:
        top, count = trades.most_common(1)[0]
        return f"{top} ({count})"
    return ""


def _worst_trade(coverage: list) -> str:
    if not coverage:
        return ""
    worst = min(coverage, key=lambda t: t.get("coverage_pct", 0))
    return f"Lowest: {worst.get('trade', '?')} ({worst.get('coverage_pct', 0):.0f}%)"
