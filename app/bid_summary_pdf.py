"""
Bid Summary PDF Generator — produces a professional PDF from analysis payload.

Uses ReportLab (already a project dependency).
Pure function, no Streamlit dependency. Can be tested independently.
"""

import io
from typing import Dict, Any, List, Optional
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# =============================================================================
# STYLES
# =============================================================================

def _build_styles():
    """Build custom paragraph styles for the bid summary PDF."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'BidTitle',
        parent=styles['Heading1'],
        fontSize=22,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d'),
    )

    h1_style = ParagraphStyle(
        'BidH1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceBefore=18,
        spaceAfter=8,
        textColor=colors.HexColor('#2c5282'),
    )

    h2_style = ParagraphStyle(
        'BidH2',
        parent=styles['Heading2'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#2d3748'),
    )

    normal_style = ParagraphStyle(
        'BidNormal',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        'BidBullet',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=3,
        leftIndent=12,
        bulletIndent=0,
    )

    footer_style = ParagraphStyle(
        'BidFooter',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER,
    )

    return {
        'title': title_style,
        'h1': h1_style,
        'h2': h2_style,
        'normal': normal_style,
        'bullet': bullet_style,
        'footer': footer_style,
    }


# =============================================================================
# TABLE HELPERS
# =============================================================================

_TABLE_STYLE_BASE = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
])


def _safe_str(val: Any, max_len: int = 80) -> str:
    """Safely convert to string and truncate."""
    s = str(val) if val is not None else ""
    # Escape ReportLab XML special chars
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


# =============================================================================
# PDF GENERATION
# =============================================================================

def generate_bid_summary_pdf(
    payload: Dict[str, Any],
    bid_strategy: Optional[Dict[str, Any]] = None,
    assumptions: Optional[List[Dict[str, Any]]] = None,
    include_drafts: bool = False,
    project_name: str = "",
    cover_branding: bool = False,
    watermark: str = "",
) -> bytes:
    """
    Generate a professional bid summary PDF from analysis payload.

    Args:
        payload: Full analysis payload dict.
        bid_strategy: Optional output from compute_bid_strategy().
        assumptions: Optional list of assumption dicts with status fields.
        include_drafts: If False, only approved/sent RFIs and reviewed conflicts are included.
        project_name: Human-readable project name for cover page.
        cover_branding: If True, adds a branded cover page with xBOQ logo + xboq.ai.
        watermark: If non-empty, adds diagonal watermark text on every page.

    Returns:
        PDF content as bytes.
    """
    # Sprint 13: Filter payload items by approval state
    from src.analysis.approval_states import filter_rfis_for_export, filter_conflicts_for_export
    payload = dict(payload)  # shallow copy
    payload["rfis"] = filter_rfis_for_export(payload.get("rfis", []), include_drafts=include_drafts)
    payload["conflicts"] = filter_conflicts_for_export(payload.get("conflicts", []), include_unreviewed=include_drafts)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = _build_styles()
    content: List = []

    # ── Sprint 18: Branded Cover Page (optional) ──────────────────────
    if cover_branding:
        content.extend(_build_cover_page(
            styles, project_name or payload.get("project_id", ""),
            payload.get("timestamp", datetime.now().isoformat()),
        ))
        content.append(PageBreak())

    # ── Title Page ─────────────────────────────────────────────────────
    project_id = payload.get("project_id", "Unknown")
    timestamp = payload.get("timestamp", datetime.now().isoformat())

    content.append(Spacer(1, 1.5 * inch))
    content.append(Paragraph("Bid Summary", styles['title']))
    content.append(Spacer(1, 0.2 * inch))
    content.append(HRFlowable(
        width="60%", thickness=2, color=colors.HexColor('#3182ce'),
    ))
    content.append(Spacer(1, 0.3 * inch))
    content.append(Paragraph(
        f"Project: {_safe_str(project_id, 120)}", styles['h2'],
    ))
    content.append(Paragraph(
        f"Generated: {_safe_str(timestamp, 40)}", styles['normal'],
    ))

    readiness = payload.get("readiness_score", 0)
    decision = payload.get("decision", "N/A")
    content.append(Spacer(1, 0.2 * inch))

    # Readiness badge
    summary_data = [
        ['Readiness Score', f'{readiness}/100'],
        ['Decision', str(decision)],
    ]
    summary_table = Table(summary_data, colWidths=[2.0 * inch, 2.5 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#edf2f7')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(summary_table)
    content.append(Spacer(1, 0.5 * inch))

    # ── Document Coverage ──────────────────────────────────────────────
    overview = payload.get("drawing_overview") or payload.get("overview") or {}
    content.append(Paragraph("Document Coverage", styles['h1']))
    files = overview.get("files", ["N/A"])
    content.append(Paragraph(f"<b>Files:</b> {', '.join(_safe_str(f, 60) for f in files[:5])}", styles['normal']))
    content.append(Paragraph(f"<b>Total Pages:</b> {overview.get('pages_total', 0)}", styles['normal']))
    disciplines = overview.get("disciplines_detected", [])
    if disciplines:
        content.append(Paragraph(f"<b>Disciplines:</b> {', '.join(disciplines)}", styles['normal']))
    content.append(Spacer(1, 0.15 * inch))

    # ── Extracted Items ────────────────────────────────────────────────
    ext = payload.get("extraction_summary", {})
    counts = ext.get("counts", {})
    if counts:
        content.append(Paragraph("Extracted Items", styles['h1']))
        count_data = [['Category', 'Count']]
        for k, v in counts.items():
            count_data.append([k.replace('_', ' ').title(), str(v)])
        count_table = Table(count_data, colWidths=[3.0 * inch, 1.5 * inch])
        count_table.setStyle(_TABLE_STYLE_BASE)
        content.append(count_table)
        content.append(Spacer(1, 0.15 * inch))

    # ── Key Requirements ───────────────────────────────────────────────
    reqs = ext.get("requirements", [])
    if reqs:
        content.append(Paragraph(f"Key Requirements ({len(reqs)} total)", styles['h1']))
        for r in reqs[:10]:
            text = _safe_str(r.get("text", ""), 120)
            cat = r.get("category", "")
            prefix = f"[{cat}] " if cat else ""
            content.append(Paragraph(f"\u2022 {prefix}{text}", styles['bullet']))
        if len(reqs) > 10:
            content.append(Paragraph(
                f"<i>...and {len(reqs) - 10} more</i>", styles['normal'],
            ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Top RFIs ───────────────────────────────────────────────────────
    rfis = payload.get("rfis", [])
    if rfis:
        content.append(Paragraph(f"Top RFIs ({len(rfis)} total)", styles['h1']))
        rfi_data = [['#', 'Trade', 'Priority', 'Question']]
        for i, rfi in enumerate(rfis[:10], 1):
            rfi_data.append([
                str(i),
                _safe_str(rfi.get("trade", ""), 20).title(),
                _safe_str(rfi.get("priority", ""), 10).upper(),
                _safe_str(rfi.get("question", ""), 70),
            ])
        rfi_table = Table(rfi_data, colWidths=[0.4 * inch, 1.0 * inch, 0.8 * inch, 3.5 * inch])
        rfi_table.setStyle(_TABLE_STYLE_BASE)
        content.append(rfi_table)
        if len(rfis) > 10:
            content.append(Paragraph(
                f"<i>...and {len(rfis) - 10} more</i>", styles['normal'],
            ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Blockers ───────────────────────────────────────────────────────
    blockers = payload.get("blockers", [])
    if blockers:
        content.append(Paragraph(f"Blockers ({len(blockers)})", styles['h1']))
        for b in blockers[:8]:
            sev = (b.get("severity", "") or "").upper()
            title = _safe_str(b.get("title", ""), 100)
            trade = _safe_str(b.get("trade", ""), 20).title()
            content.append(Paragraph(
                f"\u2022 <b>[{sev}]</b> {trade}: {title}", styles['bullet'],
            ))
        if len(blockers) > 8:
            content.append(Paragraph(
                f"<i>...and {len(blockers) - 8} more</i>", styles['normal'],
            ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Addenda ────────────────────────────────────────────────────────
    addenda = payload.get("addendum_index", [])
    if addenda:
        content.append(Paragraph(f"Addenda ({len(addenda)})", styles['h1']))
        for a in addenda:
            no = a.get("addendum_no", "?")
            date = a.get("date", "")
            title = _safe_str(a.get("title", "Untitled"), 80)
            date_str = f" ({date})" if date else ""
            content.append(Paragraph(
                f"\u2022 <b>Addendum {no}</b>{date_str}: {title}", styles['bullet'],
            ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Conflicts (Sprint 9: resolution labels) ────────────────────────
    conflicts = payload.get("conflicts", [])
    if conflicts:
        content.append(Paragraph(f"Conflicts ({len(conflicts)})", styles['h1']))
        conf_data = [['Type', 'Item', 'Detail', 'Confidence', 'Resolution']]
        for c in conflicts[:10]:
            ctype = c.get("type", "").replace("_", " ").title()
            # Sprint 9: append (Rev) for intentional revisions
            if c.get("resolution") == "intentional_revision":
                ctype += " (Rev)"
            item_id = _safe_str(c.get("item_no") or c.get("mark") or "", 15)
            changes = c.get("changes", [])
            if changes:
                detail = "; ".join(
                    f"{ch['field']}: {ch['base_value']} \u2192 {ch['addendum_value']}"
                    for ch in changes[:2]
                )
            elif c.get("text"):
                detail = _safe_str(c["text"], 60)
            elif c.get("addendum_text"):
                detail = _safe_str(c["addendum_text"], 60)
            else:
                detail = ""
            conf_val = c.get("delta_confidence")
            conf_str = f"{conf_val:.0%}" if conf_val is not None else "—"
            res_str = c.get("resolution", "") or ""
            conf_data.append([ctype, item_id, _safe_str(detail, 50), conf_str, _safe_str(res_str, 20)])
        conf_table = Table(conf_data, colWidths=[1.1 * inch, 0.7 * inch, 2.2 * inch, 0.7 * inch, 1.0 * inch])
        conf_table.setStyle(_TABLE_STYLE_BASE)
        content.append(conf_table)
        if len(conflicts) > 10:
            content.append(Paragraph(
                f"<i>...and {len(conflicts) - 10} more</i>", styles['normal'],
            ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Exclusions & Clarifications (Sprint 9) ────────────────────────
    if assumptions:
        rejected = [a for a in assumptions if a.get("status") == "rejected"]
        accepted = [a for a in assumptions if a.get("status") == "accepted"]
        if rejected or accepted:
            content.append(Paragraph("Exclusions &amp; Clarifications", styles['h1']))
            if rejected:
                content.append(Paragraph("Exclusions", styles['h2']))
                for a in rejected:
                    title = _safe_str(a.get("title", ""), 100)
                    cost = a.get("cost_impact")
                    cost_str = f" [Cost: {cost}]" if cost is not None else ""
                    content.append(Paragraph(
                        f"\u2022 <b>{title}</b>{cost_str}", styles['bullet'],
                    ))
            if accepted:
                content.append(Paragraph("Clarifications", styles['h2']))
                for a in accepted:
                    title = _safe_str(a.get("title", ""), 100)
                    cost = a.get("cost_impact")
                    cost_str = f" [Cost: {cost}]" if cost is not None else ""
                    content.append(Paragraph(
                        f"\u2022 <b>{title}</b>{cost_str}", styles['bullet'],
                    ))
            content.append(Spacer(1, 0.15 * inch))

    # ── Multi-Document Listing (Sprint 9) ─────────────────────────────
    mdi_data = payload.get("multi_doc_index")
    if mdi_data and len(mdi_data.get("docs", [])) > 1:
        content.append(Paragraph("Document Set", styles['h1']))
        for _doc_entry in mdi_data["docs"]:
            fname = _safe_str(_doc_entry.get("filename", "?"), 60)
            pc = _doc_entry.get("page_count", 0)
            content.append(Paragraph(
                f"\u2022 <b>{fname}</b> \u2014 {pc} pages", styles['bullet'],
            ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Bid Strategy ───────────────────────────────────────────────────
    if bid_strategy:
        content.append(Paragraph("Bid Strategy Assessment", styles['h1']))
        for key in ("client_fit", "risk_score", "competition_score", "readiness_score"):
            dial = bid_strategy.get(key, {})
            name = dial.get("name", key.replace("_", " ").title())
            score = dial.get("score")
            conf = dial.get("confidence", "")
            based_on = dial.get("based_on", [])
            if score is not None:
                content.append(Paragraph(
                    f"\u2022 <b>{name}:</b> {score}/100 ({conf})", styles['bullet'],
                ))
            else:
                content.append(Paragraph(
                    f"\u2022 <b>{name}:</b> Not computed", styles['bullet'],
                ))
            for reason in based_on[:3]:
                content.append(Paragraph(
                    f"    \u2013 {_safe_str(reason, 100)}", styles['bullet'],
                ))

        recs = bid_strategy.get("recommendations", [])
        if recs:
            content.append(Spacer(1, 0.1 * inch))
            content.append(Paragraph("<b>Recommendations:</b>", styles['normal']))
            for r in recs:
                content.append(Paragraph(f"\u2022 {_safe_str(r, 120)}", styles['bullet']))
        content.append(Spacer(1, 0.15 * inch))

    # ── Bid Pack QA Score (Sprint 10) ────────────────────────────────
    qa_score = payload.get("qa_score")
    if qa_score and isinstance(qa_score, dict):
        score_val = qa_score.get("score", 0)
        confidence = qa_score.get("confidence", "")
        content.append(Paragraph(
            f"Bid Pack QA Score: {score_val}/100 ({confidence})", styles['h1'],
        ))

        breakdown = qa_score.get("breakdown", {})
        if breakdown:
            qa_table_data = [["Component", "Score"]]
            for comp_key, comp_val in breakdown.items():
                label = comp_key.replace("_", " ").title()
                qa_table_data.append([label, f"{comp_val}/20"])
            qa_table = Table(qa_table_data, colWidths=[3.5 * inch, 1.5 * inch])
            qa_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ]))
            content.append(qa_table)
            content.append(Spacer(1, 0.1 * inch))

        top_actions = qa_score.get("top_actions", [])
        if top_actions:
            content.append(Paragraph("<b>Top Actions to Improve:</b>", styles['normal']))
            for action in top_actions[:5]:
                content.append(Paragraph(
                    f"\u2022 {_safe_str(action, 120)}", styles['bullet'],
                ))
        content.append(Spacer(1, 0.15 * inch))

    # ── Pricing Guidance (Sprint 11) ──────────────────────────────────
    pricing = payload.get("pricing_guidance")
    if pricing and isinstance(pricing, dict):
        content.append(Paragraph("Pricing Guidance", styles['h1']))

        # Contingency range table
        cont = pricing.get("contingency_range", {})
        if cont:
            content.append(Paragraph("Contingency Range", styles['h2']))
            cont_data = [
                ["Metric", "Value"],
                ["Low", f"{cont.get('low_pct', 0)}%"],
                ["High", f"{cont.get('high_pct', 0)}%"],
                ["Recommended", f"{cont.get('recommended_pct', 0)}%"],
            ]
            cont_table = Table(cont_data, colWidths=[2.5 * inch, 2.0 * inch])
            cont_table.setStyle(_TABLE_STYLE_BASE)
            content.append(cont_table)

            rationale = cont.get("rationale", "")
            if rationale:
                content.append(Spacer(1, 0.05 * inch))
                content.append(Paragraph(
                    f"<i>{_safe_str(rationale, 200)}</i>", styles['normal'],
                ))
            content.append(Spacer(1, 0.1 * inch))

        # Recommended Exclusions
        excl = pricing.get("recommended_exclusions", [])
        if excl:
            content.append(Paragraph("Recommended Exclusions", styles['h2']))
            for e in excl:
                content.append(Paragraph(
                    f"\u2022 {_safe_str(e, 120)}", styles['bullet'],
                ))
            content.append(Spacer(1, 0.1 * inch))

        # Recommended Clarifications
        clar = pricing.get("recommended_clarifications", [])
        if clar:
            content.append(Paragraph("Recommended Clarifications", styles['h2']))
            for c in clar[:15]:
                content.append(Paragraph(
                    f"\u2022 {_safe_str(c, 120)}", styles['bullet'],
                ))
            content.append(Spacer(1, 0.1 * inch))

        # Suggested Alternates / VE
        ve = pricing.get("suggested_alternates_ve", [])
        if ve:
            content.append(Paragraph("Suggested Alternates / Value Engineering", styles['h2']))
            for v in ve:
                item = _safe_str(v.get("item", ""), 100)
                reason = _safe_str(v.get("reason", ""), 100)
                content.append(Paragraph(
                    f"\u2022 <b>{item}</b>", styles['bullet'],
                ))
                if reason:
                    content.append(Paragraph(
                        f"    \u2013 {reason}", styles['bullet'],
                    ))
            content.append(Spacer(1, 0.15 * inch))

    # ── Footer ─────────────────────────────────────────────────────────
    content.append(Spacer(1, 0.3 * inch))
    content.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor('#cbd5e0'),
    ))
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph(
        "Generated by xBOQ Bid Engineer", styles['footer'],
    ))

    # Build PDF
    # Sprint 18: Optional watermark on every page
    if watermark:
        _wm_text = watermark

        def _watermark_callback(canvas_obj, doc_obj):
            canvas_obj.saveState()
            canvas_obj.setFont("Helvetica", 60)
            canvas_obj.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.3))
            canvas_obj.translate(A4[0] / 2, A4[1] / 2)
            canvas_obj.rotate(45)
            canvas_obj.drawCentredString(0, 0, _wm_text)
            canvas_obj.restoreState()

        doc.build(content, onFirstPage=_watermark_callback, onLaterPages=_watermark_callback)
    else:
        doc.build(content)
    return buffer.getvalue()


# =============================================================================
# SPRINT 18: BRANDED COVER PAGE
# =============================================================================

def _build_cover_page(styles, project_name: str, timestamp: str) -> list:
    """Build a branded cover page with xBOQ branding.

    Returns list of flowable elements for the cover page.
    """
    cover: list = []

    # Top spacer
    cover.append(Spacer(1, 2.5 * inch))

    # Logo text (large)
    logo_style = ParagraphStyle(
        'CoverLogo',
        parent=styles['title'],
        fontSize=48,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d'),
    )
    cover.append(Paragraph("xBOQ", logo_style))

    # Subtitle
    sub_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['normal'],
        fontSize=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=30,
    )
    cover.append(Paragraph("Pre-Construction Scope &amp; Risk Report", sub_style))

    # Horizontal rule
    cover.append(Spacer(1, 0.2 * inch))
    cover.append(HRFlowable(
        width="40%", thickness=2, color=colors.HexColor('#3182ce'),
    ))
    cover.append(Spacer(1, 0.5 * inch))

    # Project name
    if project_name:
        proj_style = ParagraphStyle(
            'CoverProject',
            parent=styles['normal'],
            fontSize=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=10,
        )
        cover.append(Paragraph(_safe_str(project_name, 100), proj_style))

    # Timestamp
    ts_style = ParagraphStyle(
        'CoverTimestamp',
        parent=styles['normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#718096'),
        spaceAfter=20,
    )
    cover.append(Paragraph(f"Generated: {_safe_str(timestamp, 40)}", ts_style))

    # Footer with URL
    cover.append(Spacer(1, 2.5 * inch))
    url_style = ParagraphStyle(
        'CoverURL',
        parent=styles['normal'],
        fontSize=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#3182ce'),
    )
    cover.append(Paragraph("xboq.ai", url_style))

    return cover


# =============================================================================
# EXECUTIVE SUMMARY PDF (one-page)
# =============================================================================

def _one_page_summary(payload: Dict[str, Any]) -> str:
    """Generate plain-text content for a one-page executive summary.

    Returns a multi-line string summarising key metrics, top risks, and a footer.
    Useful as a fallback or for logging when PDF generation is not needed.
    """
    project_id = payload.get("project_id") or payload.get("project_name") or "Unknown Project"
    timestamp = payload.get("timestamp", datetime.now().isoformat())[:10]
    total_cost = payload.get("total_cost") or payload.get("boq_stats", {}).get("total_cost") or 0
    readiness = payload.get("readiness_score", 0)
    rfis = payload.get("rfis", [])
    blockers = payload.get("blockers", [])
    critical_gaps = [b for b in blockers if (b.get("severity") or "").lower() in ("critical", "high")]

    lines = [
        "=" * 60,
        f"EXECUTIVE SUMMARY — {_safe_str(project_id, 60)}",
        f"Analysis date: {timestamp}",
        "=" * 60,
        "",
        f"Total Cost Estimate : ₹{total_cost/1e5:.1f}L" if total_cost else "Total Cost Estimate : N/A",
        f"Readiness Score     : {readiness}/100",
        f"RFIs Generated      : {len(rfis)}",
        f"Critical Gaps       : {len(critical_gaps)}",
        "",
        "TOP RISKS:",
    ]
    for b in blockers[:3]:
        sev = (b.get("severity") or "").upper()
        title = _safe_str(b.get("title", ""), 80)
        lines.append(f"  [{sev}] {title}")
    if not blockers:
        lines.append("  None identified")
    lines += ["", "-" * 60, "Generated by xBOQ.ai", "-" * 60]
    return "\n".join(lines)


def generate_executive_summary_pdf(
    payload: Dict[str, Any],
    output_path: Optional[str] = None,
) -> bytes:
    """Generate a clean one-page executive summary PDF.

    Args:
        payload: Full analysis payload dict.
        output_path: If provided, write PDF bytes to this path as well.

    Returns:
        PDF content as bytes.
    """
    project_id = payload.get("project_id") or payload.get("project_name") or "Unknown Project"
    timestamp = payload.get("timestamp", datetime.now().isoformat())[:10]
    total_cost = payload.get("total_cost") or (payload.get("boq_stats") or {}).get("total_cost") or 0
    readiness = payload.get("readiness_score", 0)
    rfis = payload.get("rfis", [])
    blockers = payload.get("blockers", [])
    critical_gaps = [b for b in blockers if (b.get("severity") or "").lower() in ("critical", "high")]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = _build_styles()

    # Custom styles for exec summary
    exec_title_style = ParagraphStyle(
        'ExecTitle',
        parent=styles['title'],
        fontSize=20,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d'),
    )
    exec_sub_style = ParagraphStyle(
        'ExecSub',
        parent=styles['normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=16,
    )
    metric_label_style = ParagraphStyle(
        'MetricLabel',
        parent=styles['normal'],
        fontSize=8,
        textColor=colors.HexColor('#718096'),
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    metric_value_style = ParagraphStyle(
        'MetricValue',
        parent=styles['h1'],
        fontSize=18,
        spaceBefore=0,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c5282'),
    )

    content: List = []

    # Header
    content.append(Spacer(1, 0.3 * inch))
    content.append(Paragraph("Executive Summary", exec_title_style))
    content.append(Paragraph(
        f"{_safe_str(project_id, 80)} &nbsp;|&nbsp; {_safe_str(timestamp, 20)}",
        exec_sub_style,
    ))
    content.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#3182ce')))
    content.append(Spacer(1, 0.25 * inch))

    # 4 key metrics in a 2x2 table
    total_cost_str = f"₹{total_cost/1e5:.1f}L" if total_cost else "N/A"
    metrics_data = [
        [
            Paragraph("TOTAL COST ESTIMATE", metric_label_style),
            Paragraph("READINESS SCORE", metric_label_style),
            Paragraph("RFIs GENERATED", metric_label_style),
            Paragraph("CRITICAL GAPS", metric_label_style),
        ],
        [
            Paragraph(_safe_str(total_cost_str, 20), metric_value_style),
            Paragraph(f"{readiness}/100", metric_value_style),
            Paragraph(str(len(rfis)), metric_value_style),
            Paragraph(str(len(critical_gaps)), metric_value_style),
        ],
    ]
    metrics_table = Table(
        metrics_data,
        colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch],
    )
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(metrics_table)
    content.append(Spacer(1, 0.3 * inch))

    # Top 3 risks
    content.append(Paragraph("Top Risks", styles['h1']))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0')))
    content.append(Spacer(1, 0.1 * inch))
    top_risks = blockers[:3]
    if top_risks:
        risk_data = [["Severity", "Trade", "Description"]]
        for b in top_risks:
            sev = _safe_str(b.get("severity", ""), 10).upper()
            trade = _safe_str(b.get("trade", ""), 20).title()
            title = _safe_str(b.get("title", ""), 100)
            risk_data.append([sev, trade, title])
        risk_table = Table(
            risk_data,
            colWidths=[0.9 * inch, 1.2 * inch, 5.1 * inch],
        )
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(risk_table)
    else:
        content.append(Paragraph("No blockers identified.", styles['normal']))

    # Footer
    content.append(Spacer(1, 0.5 * inch))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e0')))
    content.append(Spacer(1, 0.08 * inch))
    content.append(Paragraph("Generated by xBOQ.ai", styles['footer']))

    doc.build(content)
    pdf_bytes = buffer.getvalue()

    if output_path:
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as _f:
            _f.write(pdf_bytes)

    return pdf_bytes
