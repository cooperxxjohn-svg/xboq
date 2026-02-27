"""
Evidence Appendix PDF Generator — produces a professional evidence appendix from
approved RFIs, reviewed conflicts, and accepted assumptions.

Uses ReportLab (already a project dependency).
Pure function, no Streamlit dependency. Can be tested independently.

Sprint 15: Packaging + Proof + Meeting Workflow.
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
# STYLES (matching bid_summary_pdf.py conventions)
# =============================================================================

def _build_styles():
    """Build custom paragraph styles for the evidence appendix."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'EvidenceTitle',
        parent=styles['Heading1'],
        fontSize=22,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d'),
    )

    h1_style = ParagraphStyle(
        'EvidenceH1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceBefore=18,
        spaceAfter=8,
        textColor=colors.HexColor('#2c5282'),
    )

    h2_style = ParagraphStyle(
        'EvidenceH2',
        parent=styles['Heading2'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#2d3748'),
    )

    normal_style = ParagraphStyle(
        'EvidenceNormal',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        'EvidenceBullet',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=3,
        leftIndent=12,
        bulletIndent=0,
    )

    footer_style = ParagraphStyle(
        'EvidenceFooter',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER,
    )

    toc_entry = ParagraphStyle(
        'TocEntry',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        leftIndent=12,
        textColor=colors.HexColor('#2c5282'),
    )

    return {
        'title': title_style,
        'h1': h1_style,
        'h2': h2_style,
        'normal': normal_style,
        'bullet': bullet_style,
        'footer': footer_style,
        'toc_entry': toc_entry,
    }


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
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


# =============================================================================
# BOOKMARK-ENABLED DOC TEMPLATE
# =============================================================================

class _BookmarkDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate subclass that adds PDF bookmarks for section headings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bookmarks = []  # [(title, key, level)]
        self._toc_entries = []  # [(title, page_num)]

    def afterFlowable(self, flowable):
        """Hook called after each flowable is rendered — adds bookmarks."""
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in ('EvidenceH1', 'EvidenceH2'):
                level = 0 if style_name == 'EvidenceH1' else 1
                title = flowable.getPlainText()
                key = f"bm_{len(self._bookmarks)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(title, key, level=level)
                self._bookmarks.append((title, key, level))
                self._toc_entries.append((title, self.page))


# =============================================================================
# EVIDENCE ROW BUILDERS
# =============================================================================

def _build_rfi_evidence(rfi: dict, idx: int, styles: dict) -> list:
    """Build flowables for a single RFI evidence item."""
    flowables = []
    rfi_id = _safe_str(rfi.get("id", f"RFI-{idx + 1:04d}"), 30)
    trade = _safe_str(rfi.get("trade", "general"), 30).title()
    priority = _safe_str(rfi.get("priority", "medium"), 20).upper()
    question = _safe_str(rfi.get("question", ""), 200)
    status = _safe_str(rfi.get("status", "draft"), 20)

    flowables.append(Paragraph(f"<b>{rfi_id}</b> [{trade}] [{priority}] — {status}", styles['h2']))

    # Question
    flowables.append(Paragraph(f"<b>Question:</b> {question}", styles['normal']))

    # Evidence pages and sheets
    ev_pages = rfi.get("evidence_pages", [])
    sheets = rfi.get("evidence", {}).get("sheets", []) if isinstance(rfi.get("evidence"), dict) else []
    if not sheets:
        sheets = rfi.get("sheets", [])

    if ev_pages:
        page_str = ", ".join(str(p + 1) for p in ev_pages[:10])
        flowables.append(Paragraph(f"<b>Pages:</b> {page_str}", styles['normal']))
    if sheets:
        sheet_str = ", ".join(_safe_str(s, 20) for s in sheets[:10])
        flowables.append(Paragraph(f"<b>Sheets:</b> {sheet_str}", styles['normal']))

    # Bounding box coordinates (text reference, not image)
    bbox_data = rfi.get("evidence", {}).get("bbox", []) if isinstance(rfi.get("evidence"), dict) else []
    if not bbox_data:
        bbox_data = rfi.get("bbox", [])
    if bbox_data:
        for page_idx, page_boxes in enumerate(bbox_data[:5]):
            if not page_boxes:
                continue
            for box in page_boxes[:3]:
                if len(box) >= 4:
                    x0, y0, x1, y1 = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in box[:4]]
                    conf = f", conf={box[4]:.2f}" if len(box) >= 5 and isinstance(box[4], (int, float)) else ""
                    bid = f" [{box[5]}]" if len(box) >= 6 else ""
                    flowables.append(Paragraph(
                        f"<b>Region:</b> ({x0}, {y0}) to ({x1}, {y1}){conf}{bid}",
                        styles['bullet'],
                    ))

    # Snippets
    snippets = rfi.get("evidence", {}).get("snippets", []) if isinstance(rfi.get("evidence"), dict) else []
    if not snippets:
        snippet_text = rfi.get("description", "") or rfi.get("missing_info", "")
        if snippet_text:
            snippets = [snippet_text]
    for snip in snippets[:3]:
        flowables.append(Paragraph(f"<i>Snippet:</i> \"{_safe_str(snip, 150)}\"", styles['bullet']))

    # Suggested resolution
    resolution = rfi.get("suggested_resolution", "") or rfi.get("suggested_response", "")
    if resolution:
        flowables.append(Paragraph(f"<b>Suggested Resolution:</b> {_safe_str(resolution, 150)}", styles['normal']))

    flowables.append(Spacer(1, 8))
    return flowables


def _build_conflict_evidence(conflict: dict, idx: int, styles: dict) -> list:
    """Build flowables for a single conflict evidence item."""
    flowables = []
    ctype = _safe_str(conflict.get("type", "unknown"), 30)
    item_no = _safe_str(conflict.get("item_no", conflict.get("item", "")), 30)
    delta_conf = conflict.get("delta_confidence", 0)
    review_status = _safe_str(conflict.get("review_status", "unreviewed"), 20)

    title = f"Conflict #{idx + 1}: {ctype}"
    if item_no:
        title += f" — Item {item_no}"
    flowables.append(Paragraph(f"<b>{title}</b> [{review_status}]", styles['h2']))

    # Confidence
    if delta_conf:
        flowables.append(Paragraph(f"<b>Delta Confidence:</b> {delta_conf:.2f}", styles['normal']))

    # Pages
    base_page = conflict.get("base_page")
    add_page = conflict.get("addendum_page")
    if base_page is not None or add_page is not None:
        page_info = []
        if base_page is not None:
            page_info.append(f"Base: p{base_page + 1}")
        if add_page is not None:
            page_info.append(f"Addendum: p{add_page + 1}")
        flowables.append(Paragraph(f"<b>Pages:</b> {', '.join(page_info)}", styles['normal']))

    # Changes
    changes = conflict.get("changes", [])
    for ch in changes[:5]:
        field = _safe_str(ch.get("field", ""), 30)
        base_val = _safe_str(ch.get("base_value", ""), 40)
        add_val = _safe_str(ch.get("addendum_value", ""), 40)
        flowables.append(Paragraph(
            f"<b>{field}:</b> {base_val} &rarr; {add_val}",
            styles['bullet'],
        ))

    # Description
    desc = conflict.get("description", "")
    if desc:
        flowables.append(Paragraph(f"<i>{_safe_str(desc, 200)}</i>", styles['normal']))

    flowables.append(Spacer(1, 8))
    return flowables


def _build_assumption_evidence(assumption: dict, idx: int, styles: dict) -> list:
    """Build flowables for a single assumption evidence item."""
    flowables = []
    title = _safe_str(assumption.get("title", assumption.get("assumption", f"Assumption #{idx + 1}")), 100)
    status = _safe_str(assumption.get("status", ""), 20)
    cost_impact = _safe_str(assumption.get("cost_impact", ""), 50)

    flowables.append(Paragraph(f"<b>{title}</b> [{status}]", styles['h2']))

    if cost_impact:
        flowables.append(Paragraph(f"<b>Cost Impact:</b> {cost_impact}", styles['normal']))

    basis_pages = assumption.get("basis_pages", [])
    if basis_pages:
        page_str = ", ".join(str(p + 1) if isinstance(p, int) else str(p) for p in basis_pages[:10])
        flowables.append(Paragraph(f"<b>Basis Pages:</b> {page_str}", styles['normal']))

    text = assumption.get("text", assumption.get("description", ""))
    if text:
        flowables.append(Paragraph(f"<i>{_safe_str(text, 200)}</i>", styles['normal']))

    flowables.append(Spacer(1, 8))
    return flowables


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_evidence_appendix_pdf(
    rfis: List[dict],
    conflicts: List[dict],
    assumptions: Optional[List[dict]] = None,
    include_drafts: bool = False,
    config: Optional[dict] = None,
) -> bytes:
    """
    Generate an evidence appendix PDF containing approved RFIs, reviewed conflicts,
    and accepted assumptions with document/page references, bbox coordinates,
    snippets, citations, and item IDs.

    Args:
        rfis: List of RFI dicts from analysis payload.
        conflicts: List of conflict dicts from analysis payload.
        assumptions: Optional list of assumption dicts.
        include_drafts: If False, only approved/sent RFIs and reviewed conflicts.
        config: Optional dict to control sections:
                {"include_rfis": True, "include_conflicts": True, "include_assumptions": True}

    Returns:
        PDF content as bytes.
    """
    config = config or {}
    include_rfis = config.get("include_rfis", True)
    include_conflicts = config.get("include_conflicts", True)
    include_assumptions = config.get("include_assumptions", True)

    # Filter by approval state
    from src.analysis.approval_states import filter_rfis_for_export, filter_conflicts_for_export
    filtered_rfis = filter_rfis_for_export(rfis, include_drafts=include_drafts) if include_rfis else []
    filtered_conflicts = filter_conflicts_for_export(conflicts, include_unreviewed=include_drafts) if include_conflicts else []
    filtered_assumptions = []
    if assumptions and include_assumptions:
        if include_drafts:
            filtered_assumptions = list(assumptions)
        else:
            filtered_assumptions = [a for a in assumptions if a.get("status") == "accepted"]

    styles = _build_styles()
    content = []

    # ── Cover page ────────────────────────────────────────────────────
    content.append(Spacer(1, 40))
    content.append(Paragraph("Evidence Appendix", styles['title']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['normal'],
    ))
    content.append(Spacer(1, 10))

    total_items = len(filtered_rfis) + len(filtered_conflicts) + len(filtered_assumptions)
    content.append(Paragraph(
        f"<b>{total_items} evidence items</b> | "
        f"{len(filtered_rfis)} RFIs | "
        f"{len(filtered_conflicts)} Conflicts | "
        f"{len(filtered_assumptions)} Assumptions",
        styles['normal'],
    ))

    # ── Table of Contents ─────────────────────────────────────────────
    content.append(Spacer(1, 20))
    content.append(Paragraph("Table of Contents", styles['h1']))
    toc_items = []
    if filtered_rfis:
        toc_items.append("1. RFI Evidence")
        for i, rfi in enumerate(filtered_rfis):
            rfi_id = rfi.get("id", f"RFI-{i + 1:04d}")
            toc_items.append(f"    1.{i + 1} {rfi_id}")
    if filtered_conflicts:
        toc_items.append(f"{2 if filtered_rfis else 1}. Conflict Evidence")
    if filtered_assumptions:
        sec_num = 1 + bool(filtered_rfis) + bool(filtered_conflicts)
        toc_items.append(f"{sec_num}. Assumption Evidence")

    for entry in toc_items:
        content.append(Paragraph(entry, styles['toc_entry']))

    if not total_items:
        content.append(Spacer(1, 20))
        content.append(Paragraph("No evidence items matching the current filter criteria.", styles['normal']))

    # ── RFI Evidence Section ──────────────────────────────────────────
    if filtered_rfis:
        content.append(PageBreak())
        content.append(Paragraph("RFI Evidence", styles['h1']))
        content.append(Spacer(1, 6))

        # Summary table
        summary_data = [["#", "ID", "Trade", "Priority", "Status"]]
        for i, rfi in enumerate(filtered_rfis):
            summary_data.append([
                str(i + 1),
                _safe_str(rfi.get("id", ""), 20),
                _safe_str(rfi.get("trade", ""), 20).title(),
                _safe_str(rfi.get("priority", ""), 10).upper(),
                _safe_str(rfi.get("status", "draft"), 10),
            ])
        if len(summary_data) > 1:
            t = Table(summary_data, colWidths=[30, 80, 80, 60, 60])
            t.setStyle(_TABLE_STYLE_BASE)
            content.append(t)
            content.append(Spacer(1, 12))

        # Detailed evidence for each RFI
        for i, rfi in enumerate(filtered_rfis):
            content.extend(_build_rfi_evidence(rfi, i, styles))
            content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e0')))

    # ── Conflict Evidence Section ─────────────────────────────────────
    if filtered_conflicts:
        content.append(PageBreak())
        content.append(Paragraph("Conflict Evidence", styles['h1']))
        content.append(Spacer(1, 6))

        # Summary table
        summary_data = [["#", "Type", "Item", "Confidence", "Status"]]
        for i, c in enumerate(filtered_conflicts):
            summary_data.append([
                str(i + 1),
                _safe_str(c.get("type", ""), 20),
                _safe_str(c.get("item_no", c.get("item", "")), 20),
                f"{c.get('delta_confidence', 0):.2f}" if c.get("delta_confidence") else "—",
                _safe_str(c.get("review_status", "unreviewed"), 15),
            ])
        if len(summary_data) > 1:
            t = Table(summary_data, colWidths=[30, 80, 80, 60, 60])
            t.setStyle(_TABLE_STYLE_BASE)
            content.append(t)
            content.append(Spacer(1, 12))

        for i, c in enumerate(filtered_conflicts):
            content.extend(_build_conflict_evidence(c, i, styles))
            content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e0')))

    # ── Assumption Evidence Section ───────────────────────────────────
    if filtered_assumptions:
        content.append(PageBreak())
        content.append(Paragraph("Assumption Evidence", styles['h1']))
        content.append(Spacer(1, 6))

        for i, a in enumerate(filtered_assumptions):
            content.extend(_build_assumption_evidence(a, i, styles))
            content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e0')))

    # ── Footer ────────────────────────────────────────────────────────
    content.append(Spacer(1, 20))
    content.append(Paragraph("Evidence Appendix — Generated by xBOQ Bid Engineer", styles['footer']))

    # ── Build PDF ─────────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc = _BookmarkDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(content)
    return buffer.getvalue()
