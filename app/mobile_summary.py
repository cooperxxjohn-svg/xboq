"""
Mobile-first summary view — single-page PWA.

Renders a self-contained HTML page with:
  • Bid readiness score (large gauge)
  • Top 5 RFIs
  • Cost estimate (total BOQ + autofill)
  • Key deadlines
  • No 31-tab complexity — optimised for phones

Usage (standalone):
    python app/mobile_summary.py --job_id <id> --out summary.html

Usage (from Streamlit demo_page.py):
    from app.mobile_summary import render_mobile_summary
    html = render_mobile_summary(payload, project_name="Hospital Block")
    st.components.v1.html(html, height=900, scrolling=True)

Usage (API):
    GET /api/mobile-summary/{job_id}   → returns HTML
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (matches xBOQ brand — deep navy + amber)
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 0 0 32px 0;
}
header {
  background: #1e293b;
  padding: 16px 20px 12px;
  border-bottom: 2px solid #f59e0b;
  display: flex;
  align-items: center;
  gap: 12px;
}
header .logo { font-size: 22px; font-weight: 800; color: #f59e0b; }
header .proj { font-size: 14px; color: #94a3b8; }
.score-ring {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 20px 16px;
  background: #1e293b;
  margin: 16px;
  border-radius: 16px;
}
.ring-svg { width: 140px; height: 140px; }
.score-val { font-size: 36px; font-weight: 800; }
.score-label { font-size: 13px; color: #94a3b8; margin-top: 4px; letter-spacing: 1px; }
.section {
  margin: 0 16px 16px;
  background: #1e293b;
  border-radius: 12px;
  overflow: hidden;
}
.section-head {
  padding: 14px 16px 10px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #94a3b8;
  text-transform: uppercase;
  border-bottom: 1px solid #334155;
}
.stat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #1e3a5f22;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { font-size: 14px; color: #cbd5e1; }
.stat-val { font-size: 14px; font-weight: 700; color: #f8fafc; }
.rfi-item {
  padding: 12px 16px;
  border-bottom: 1px solid #334155;
  cursor: pointer;
}
.rfi-item:last-child { border-bottom: none; }
.rfi-num { font-size: 11px; color: #64748b; margin-bottom: 3px; }
.rfi-q { font-size: 13px; color: #e2e8f0; line-height: 1.4; }
.rfi-trade {
  display: inline-block;
  margin-top: 6px;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  background: #1d4ed8;
  color: #bfdbfe;
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 700;
}
.badge-green  { background: #14532d; color: #86efac; }
.badge-amber  { background: #78350f; color: #fcd34d; }
.badge-red    { background: #7f1d1d; color: #fca5a5; }
.deadline-row {
  padding: 12px 16px;
  border-bottom: 1px solid #334155;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.deadline-row:last-child { border-bottom: none; }
.dl-label { font-size: 13px; color: #cbd5e1; }
.dl-val { font-size: 13px; font-weight: 700; color: #fbbf24; }
.footer {
  text-align: center;
  padding: 24px 16px 8px;
  font-size: 11px;
  color: #475569;
}
"""

# ---------------------------------------------------------------------------
# SVG ring gauge
# ---------------------------------------------------------------------------

def _ring_gauge(score: int, label: str) -> str:
    """Return SVG + score number for the readiness ring."""
    radius = 52
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - score / 100)

    if score >= 75:
        colour = "#22c55e"
    elif score >= 50:
        colour = "#f59e0b"
    else:
        colour = "#ef4444"

    badge_class = "badge-green" if score >= 75 else ("badge-amber" if score >= 50 else "badge-red")

    return f"""
<div class="score-ring">
  <svg class="ring-svg" viewBox="0 0 140 140">
    <circle cx="70" cy="70" r="{radius}" fill="none" stroke="#334155" stroke-width="12"/>
    <circle cx="70" cy="70" r="{radius}" fill="none"
      stroke="{colour}" stroke-width="12"
      stroke-dasharray="{circumference:.1f}"
      stroke-dashoffset="{offset:.1f}"
      stroke-linecap="round"
      transform="rotate(-90 70 70)"
    />
    <text x="70" y="67" text-anchor="middle" font-size="28" font-weight="800"
          fill="{colour}" font-family="sans-serif">{score}</text>
    <text x="70" y="87" text-anchor="middle" font-size="11" fill="#94a3b8"
          font-family="sans-serif">/ 100</text>
  </svg>
  <span class="badge {badge_class}">{label}</span>
</div>
"""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _cost_section(payload: dict) -> str:
    boq_items = payload.get("boq_items") or []
    boq_total = sum(
        float(i.get("total_inr") or 0) or
        float(i.get("rate_inr") or 0) * float(i.get("quantity") or 0)
        for i in boq_items
    )

    autofill = payload.get("boq_autofill") or {}
    af_total = float(autofill.get("total_inr") or 0)

    grand = boq_total + af_total

    def _fmt(n: float) -> str:
        if n >= 1e7:
            return f"₹{n/1e7:.2f} Cr"
        if n >= 1e5:
            return f"₹{n/1e5:.2f} L"
        return f"₹{n:,.0f}"

    rows = ""
    if boq_total:
        rows += f"""<div class="stat-row"><span class="stat-label">BOQ Total</span>
                    <span class="stat-val">{_fmt(boq_total)}</span></div>"""
    if af_total:
        rows += f"""<div class="stat-row"><span class="stat-label">Auto-fill (est.)</span>
                    <span class="stat-val">{_fmt(af_total)}</span></div>"""
    if grand:
        rows += f"""<div class="stat-row"><span class="stat-label" style="font-weight:700">Grand Total</span>
                    <span class="stat-val" style="color:#f59e0b">{_fmt(grand)}</span></div>"""

    if not rows:
        rows = '<div class="stat-row"><span class="stat-label">No cost data available</span></div>'

    return f"""<div class="section">
  <div class="section-head">Cost Estimate</div>
  {rows}
</div>"""


def _rfi_section(payload: dict) -> str:
    rfis: List[dict] = payload.get("rfis") or []
    # Sort by priority: critical > high > medium > low
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rfis_sorted = sorted(rfis, key=lambda r: priority_order.get(
        (r.get("priority") or "low").lower(), 3
    ))
    top5 = rfis_sorted[:5]

    if not top5:
        return f"""<div class="section">
  <div class="section-head">Top RFIs</div>
  <div class="stat-row"><span class="stat-label">No RFIs generated</span></div>
</div>"""

    items_html = ""
    for i, rfi in enumerate(top5, 1):
        q = (rfi.get("question") or rfi.get("title") or "")[:120]
        trade = rfi.get("trade") or ""
        items_html += f"""<div class="rfi-item">
  <div class="rfi-num">RFI {i} of {len(rfis)}</div>
  <div class="rfi-q">{q}</div>
  {f'<span class="rfi-trade">{trade.upper()}</span>' if trade else ''}
</div>"""

    return f"""<div class="section">
  <div class="section-head">Top {len(top5)} RFIs <span style="font-weight:400;color:#64748b">({len(rfis)} total)</span></div>
  {items_html}
</div>"""


def _deadline_section(payload: dict) -> str:
    terms = payload.get("commercial_terms") or []
    deadlines = [t for t in terms if t.get("term_type") == "bid_deadline"]

    if not deadlines:
        return ""

    rows = ""
    for d in deadlines[:3]:
        iso = d.get("iso_date") or d.get("value") or "—"
        snippet = (d.get("snippet") or "Bid deadline")[:60]
        rows += f"""<div class="deadline-row">
  <span class="dl-label">{snippet}</span>
  <span class="dl-val">{iso}</span>
</div>"""

    return f"""<div class="section">
  <div class="section-head">Deadlines</div>
  {rows}
</div>"""


def _stats_section(payload: dict) -> str:
    stats = payload.get("processing_stats") or {}
    boq_items = payload.get("boq_items") or []
    blockers = payload.get("blockers") or []
    rfis = payload.get("rfis") or []

    rows = (
        f'<div class="stat-row"><span class="stat-label">Pages Processed</span>'
        f'<span class="stat-val">{stats.get("deep_processed_pages", "—")}</span></div>'

        f'<div class="stat-row"><span class="stat-label">BOQ Items</span>'
        f'<span class="stat-val">{len(boq_items)}</span></div>'

        f'<div class="stat-row"><span class="stat-label">RFIs Generated</span>'
        f'<span class="stat-val">{len(rfis)}</span></div>'

        f'<div class="stat-row"><span class="stat-label">Blockers</span>'
        f'<span class="stat-val" style="color:{"#ef4444" if blockers else "#22c55e"}">'
        f'{len(blockers)}</span></div>'
    )

    qa = payload.get("qa_score")
    if qa is not None:
        rows += (
            f'<div class="stat-row"><span class="stat-label">QA Score</span>'
            f'<span class="stat-val">{int(float(qa) * 100) if float(qa) <= 1 else int(qa)}%</span></div>'
        )

    return f"""<div class="section">
  <div class="section-head">At a Glance</div>
  {rows}
</div>"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def render_mobile_summary(
    payload: Dict[str, Any],
    project_name: str = "",
    *,
    include_meta_refresh: bool = False,
) -> str:
    """
    Build a complete self-contained HTML page from a pipeline payload.

    Args:
        payload:             Pipeline result dict.
        project_name:        Display name for the project.
        include_meta_refresh: If True, add a 30-second meta-refresh (for live views).

    Returns:
        HTML string.
    """
    if not isinstance(payload, dict):
        payload = {}

    project_name = project_name or payload.get("project_name") or "Tender Analysis"

    # Bid readiness score
    synthesis = payload.get("bid_synthesis") or {}
    score = int(synthesis.get("bid_readiness_score") or payload.get("bid_readiness_score") or 0)
    label_map = {
        "READY": "READY",
        "CONDITIONAL": "CONDITIONAL",
        "NOT READY": "NOT READY",
    }
    bid_label = synthesis.get("bid_readiness_label") or (
        "READY" if score >= 75 else ("CONDITIONAL" if score >= 50 else "NOT READY")
    )
    display_label = label_map.get(bid_label.upper(), bid_label)

    meta_refresh = '<meta http-equiv="refresh" content="30">' if include_meta_refresh else ""
    meta_viewport = '<meta name="viewport" content="width=device-width, initial-scale=1">'
    meta_pwa = (
        '<meta name="mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
        '<meta name="theme-color" content="#0f172a">'
    )

    ring_html = _ring_gauge(score, display_label)
    cost_html = _cost_section(payload)
    rfi_html = _rfi_section(payload)
    deadline_html = _deadline_section(payload)
    stats_html = _stats_section(payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {meta_viewport}
  {meta_pwa}
  {meta_refresh}
  <title>xBOQ — {project_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <span class="logo">xBOQ</span>
    <span class="proj">{project_name}</span>
  </header>

  {ring_html}
  {stats_html}
  {cost_html}
  {rfi_html}
  {deadline_html}

  <div class="footer">
    xBOQ.ai &nbsp;·&nbsp; Mobile Summary &nbsp;·&nbsp; Powered by AI
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# API endpoint helper (imported by main.py)
# ---------------------------------------------------------------------------

def build_mobile_summary_response(job_id: str) -> str:
    """
    Load a job by ID and return the mobile summary HTML.
    Raises HTTPException if job not found.
    """
    from fastapi import HTTPException
    try:
        from src.api.job_store import job_store
        job = job_store.get_job(job_id)
    except Exception:
        job = None
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    payload = job.payload or {}
    project_name = payload.get("project_name") or job_id
    return render_mobile_summary(payload, project_name=project_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate xBOQ mobile summary HTML")
    parser.add_argument("--job_id", help="Job ID (reads from ~/.xboq)")
    parser.add_argument("--payload", help="Path to analysis JSON file")
    parser.add_argument("--out", default="mobile_summary.html", help="Output HTML file")
    parser.add_argument("--project_name", default="", help="Project display name")
    args = parser.parse_args()

    payload: dict = {}
    if args.payload:
        with open(args.payload, encoding="utf-8") as f:
            payload = json.load(f)
    elif args.job_id:
        import sys
        sys.path.insert(0, str(__file__.replace("/app/mobile_summary.py", "")))
        try:
            from src.api.job_store import job_store
            job = job_store.get_job(args.job_id)
            payload = (job.payload or {}) if job else {}
        except ImportError:
            print("job_store not available — pass --payload instead")
            return

    html = render_mobile_summary(payload, project_name=args.project_name)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    _cli()
