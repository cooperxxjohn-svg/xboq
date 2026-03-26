"""PDF summary export — single/two-page A4 report using ReportLab."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
COLOR_GREEN = colors.HexColor("#375623")
COLOR_AMBER = colors.HexColor("#C55A11")
COLOR_RED = colors.HexColor("#C00000")
COLOR_BLUE_DARK = colors.HexColor("#1F3864")
COLOR_BLUE_LIGHT = colors.HexColor("#B8CCE4")
COLOR_GREY_LIGHT = colors.HexColor("#F2F2F2")
COLOR_TABLE_HDR = colors.HexColor("#1F3864")

DECISION_COLORS = {
    "PROCEED": COLOR_GREEN,
    "CONDITIONAL_PROCEED": COLOR_AMBER,
    "DO_NOT_PROCEED": COLOR_RED,
}


def _score_color(score: float) -> colors.Color:
    if score >= 75:
        return COLOR_GREEN
    if score >= 50:
        return COLOR_AMBER
    return COLOR_RED


def _coverage_color(score: float) -> colors.Color:
    if score >= 0.7:
        return COLOR_GREEN
    if score >= 0.4:
        return COLOR_AMBER
    return COLOR_RED


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------
def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=20,
            leading=24,
            textColor=COLOR_BLUE_DARK,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#444444"),
            fontName="Helvetica",
        ),
        "meta_right": ParagraphStyle(
            "meta_right",
            parent=base["Normal"],
            fontSize=9,
            alignment=TA_RIGHT,
            textColor=colors.HexColor("#666666"),
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            textColor=COLOR_BLUE_DARK,
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#888888"),
        ),
        "score_big": ParagraphStyle(
            "score_big",
            parent=base["Normal"],
            fontSize=36,
            leading=40,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "score_label": ParagraphStyle(
            "score_label",
            parent=base["Normal"],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
        ),
        "rfi_item": ParagraphStyle(
            "rfi_item",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            leftIndent=12,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _header_section(payload: dict, styles: dict) -> list:
    """Title block + meta line."""
    elements = []

    elements.append(
        Paragraph("xBOQ.ai \u2014 Tender Analysis Report", styles["title"])
    )

    project_id = payload.get("project_id", "—")
    timestamp = payload.get("timestamp", "—")
    run_date = date.today().isoformat()

    meta_html = (
        f"<b>Project ID:</b> {project_id}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<b>Timestamp:</b> {timestamp}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<b>Run Date:</b> {run_date}"
    )
    elements.append(Paragraph(meta_html, styles["meta_right"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_BLUE_DARK))
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def _scorecard_section(payload: dict, styles: dict) -> list:
    """Big readiness score number + decision badge."""
    elements = []
    elements.append(Paragraph("Readiness Scorecard", styles["section_heading"]))

    score = payload.get("readiness_score", 0.0)
    decision = payload.get("decision", "—")
    score_color = _score_color(score)
    decision_color = DECISION_COLORS.get(decision, colors.black)

    # Score and decision as a 2-column table for layout
    score_para = Paragraph(
        f'<font color="{score_color.hexval()}" size="36"><b>{score:.1f} / 100</b></font>',
        styles["score_label"],
    )
    decision_para = Paragraph(
        f'<font color="{decision_color.hexval()}" size="18"><b>{decision}</b></font>',
        styles["score_label"],
    )

    tbl = Table(
        [[score_para, decision_para]],
        colWidths=[9 * cm, 9 * cm],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_GREY_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def _trade_coverage_section(payload: dict, styles: dict) -> list:
    """Trade coverage 2-column table."""
    elements = []
    elements.append(Paragraph("Trade Coverage", styles["section_heading"]))

    trade_coverage = payload.get("trade_coverage") or {}
    if not trade_coverage:
        elements.append(Paragraph("No trade coverage data available.", styles["body"]))
        return elements

    header = [
        Paragraph("<b>Trade</b>", styles["body"]),
        Paragraph("<b>Coverage Score</b>", styles["body"]),
    ]
    rows = [header]
    tbl_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HDR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]

    for row_idx, (trade, info) in enumerate(sorted(trade_coverage.items()), start=1):
        cov_score = info.get("score", 0.0) if isinstance(info, dict) else 0.0
        cov_color = _coverage_color(cov_score)
        score_text = Paragraph(
            f'<font color="{cov_color.hexval()}"><b>{cov_score:.2f}</b></font>',
            styles["body"],
        )
        rows.append([trade.capitalize(), score_text])
        if row_idx % 2 == 0:
            tbl_styles.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx), COLOR_GREY_LIGHT)
            )

    tbl = Table(rows, colWidths=[9 * cm, 9 * cm])
    tbl.setStyle(TableStyle(tbl_styles))
    elements.append(tbl)
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def _boq_summary_section(payload: dict, styles: dict) -> list:
    """BOQ summary box."""
    elements = []
    elements.append(Paragraph("BOQ Summary", styles["section_heading"]))

    li_summary = payload.get("line_items_summary") or {}
    boq_stats = payload.get("boq_stats") or {}

    total = li_summary.get("total") or boq_stats.get("total_items") or 0
    tax_matched = li_summary.get("taxonomy_matched") or 0
    qty_missing = li_summary.get("qty_missing") or 0
    rate_missing = li_summary.get("rate_missing") or 0
    tax_pct = (tax_matched / total * 100) if total else 0.0

    rows = [
        [
            Paragraph("<b>Metric</b>", styles["body"]),
            Paragraph("<b>Value</b>", styles["body"]),
        ],
        ["Total Line Items", str(total)],
        ["Taxonomy Matched", f"{tax_matched} ({tax_pct:.1f}%)"],
        ["Qty Missing", str(qty_missing)],
        ["Rate Missing", str(rate_missing)],
    ]

    tbl = Table(rows, colWidths=[9 * cm, 9 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HDR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
                ("BACKGROUND", (0, 2), (-1, 2), COLOR_GREY_LIGHT),
                ("BACKGROUND", (0, 4), (-1, 4), COLOR_GREY_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def _top_rfis_section(payload: dict, styles: dict) -> list:
    """Top 5 RFIs (P1 first)."""
    elements = []
    elements.append(Paragraph("Top RFIs", styles["section_heading"]))

    rfis = payload.get("rfis") or []
    if not rfis:
        elements.append(Paragraph("No RFIs generated.", styles["body"]))
        return elements

    priority_order = {"P1": 0, "P2": 1, "P3": 2}
    sorted_rfis = sorted(
        rfis,
        key=lambda r: priority_order.get((r.get("priority") or "P3").upper(), 99),
    )
    top5 = sorted_rfis[:5]

    for idx, rfi in enumerate(top5, start=1):
        priority = (rfi.get("priority") or "P3").upper()
        trade = rfi.get("trade", "—")
        question = rfi.get("question", "")
        rfi_id = rfi.get("id", f"RFI-{idx:03d}")

        p_color = {"P1": COLOR_RED, "P2": COLOR_AMBER, "P3": COLOR_GREEN}.get(
            priority, colors.black
        )
        text = (
            f'{idx}. <b>[{rfi_id}]</b> '
            f'<font color="{p_color.hexval()}"><b>{priority}</b></font> '
            f'[{trade}] — {question}'
        )
        elements.append(Paragraph(text, styles["rfi_item"]))
        elements.append(Spacer(1, 0.15 * cm))

    elements.append(Spacer(1, 0.3 * cm))
    return elements


def _footer_section(payload: dict, styles: dict) -> list:
    timestamp = payload.get("timestamp", "—")
    elements = [
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")),
        Spacer(1, 0.15 * cm),
        Paragraph(
            f"Generated by xBOQ.ai | {timestamp}",
            styles["footer"],
        ),
    ]
    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_pdf_summary(payload: dict, output_path: Path) -> Path:
    """Export a professional A4 PDF summary of the pipeline payload.

    Args:
        payload: Pipeline output dict.
        output_path: Destination .pdf file path.

    Returns:
        output_path after writing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2 * cm,
        title="xBOQ.ai Tender Analysis Report",
        author="xBOQ.ai",
    )

    styles = _build_styles()
    story: list = []

    story.extend(_header_section(payload, styles))
    story.extend(_scorecard_section(payload, styles))
    story.extend(_trade_coverage_section(payload, styles))
    story.extend(_boq_summary_section(payload, styles))
    story.extend(_top_rfis_section(payload, styles))
    story.extend(_footer_section(payload, styles))

    doc.build(story)
    return output_path
