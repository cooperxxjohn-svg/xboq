#!/usr/bin/env python3
"""
PDF Report Generator for XBOQ Bid Engine Demo Output
Compiles all phase outputs into a comprehensive PDF report.
"""

import json
from pathlib import Path
from datetime import datetime

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.colors import HexColor, black, white, red, orange, green
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, ListFlowable, ListItem, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab not installed. Installing...")
    import subprocess
    subprocess.run(["pip3", "install", "reportlab"], check=True)
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.colors import HexColor, black, white, red, orange, green
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, ListFlowable, ListItem, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "bid_demo"
PDF_PATH = OUTPUT_DIR / "XBOQ_Bid_Report.pdf"


# Colors
PRIMARY_COLOR = HexColor("#1a365d")  # Dark blue
SECONDARY_COLOR = HexColor("#2c5282")
ACCENT_COLOR = HexColor("#3182ce")
SUCCESS_COLOR = HexColor("#38a169")
WARNING_COLOR = HexColor("#dd6b20")
DANGER_COLOR = HexColor("#e53e3e")
LIGHT_BG = HexColor("#f7fafc")
BORDER_COLOR = HexColor("#e2e8f0")


def get_styles():
    """Create custom styles for the report."""
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=12,
        alignment=TA_CENTER,
    ))

    # Section header
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=PRIMARY_COLOR,
        spaceBefore=20,
        spaceAfter=10,
        borderWidth=1,
        borderColor=PRIMARY_COLOR,
        borderPadding=5,
    ))

    # Subsection header
    styles.add(ParagraphStyle(
        name='SubHeader',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=SECONDARY_COLOR,
        spaceBefore=15,
        spaceAfter=8,
    ))

    # Body text (override existing)
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 6

    # Status styles
    styles.add(ParagraphStyle(
        name='StatusPass',
        parent=styles['Normal'],
        fontSize=14,
        textColor=SUCCESS_COLOR,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    ))

    styles.add(ParagraphStyle(
        name='StatusWarning',
        parent=styles['Normal'],
        fontSize=14,
        textColor=WARNING_COLOR,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    ))

    styles.add(ParagraphStyle(
        name='StatusFail',
        parent=styles['Normal'],
        fontSize=14,
        textColor=DANGER_COLOR,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    ))

    # Table header style
    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=white,
        fontName='Helvetica-Bold',
    ))

    # Table cell style
    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
    ))

    # Small text
    styles.add(ParagraphStyle(
        name='SmallText',
        parent=styles['Normal'],
        fontSize=8,
        textColor=HexColor("#718096"),
    ))

    return styles


def format_currency(value):
    """Format number as Indian currency."""
    if value >= 10000000:  # Crores
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:  # Lakhs
        return f"₹{value/100000:.2f} L"
    else:
        return f"₹{value:,.2f}"


def create_status_banner(status, score, styles):
    """Create a status banner based on gate status."""
    if status == "PASS":
        text = f"✓ BID STATUS: READY FOR SUBMISSION (Score: {score:.1f}/100)"
        style = styles['StatusPass']
        bg_color = HexColor("#c6f6d5")
    elif status == "PASS_WITH_RESERVATIONS":
        text = f"⚠ BID STATUS: SUBMITTABLE WITH RESERVATIONS (Score: {score:.1f}/100)"
        style = styles['StatusWarning']
        bg_color = HexColor("#feebc8")
    else:
        text = f"✗ BID STATUS: NOT SUBMITTABLE (Score: {score:.1f}/100)"
        style = styles['StatusFail']
        bg_color = HexColor("#fed7d7")

    # Create a table for the banner
    data = [[Paragraph(text, style)]]
    table = Table(data, colWidths=[170*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
    ]))
    return table


def create_metrics_table(data, styles):
    """Create the key metrics table."""
    table_data = [
        [Paragraph("<b>Metric</b>", styles['TableHeader']),
         Paragraph("<b>Value</b>", styles['TableHeader'])],
    ]

    metrics = [
        ("BOQ Total", format_currency(data.get("boq_total", 0))),
        ("Prelims", format_currency(data.get("prelims_total", 0))),
        ("Grand Total", format_currency(data.get("grand_total", 0))),
        ("Built-up Area", f"{data.get('built_up_area', 4500):,} sqm"),
        ("Rate per Sqft", format_currency(data.get("rate_per_sqft", 0))),
        ("Duration", f"{data.get('duration_months', 14)} months"),
        ("Gate Score", f"{data.get('gate_score', 0):.1f}/100"),
        ("Reservations", str(data.get("reservations_count", 0))),
        ("High Priority RFIs", str(data.get("high_priority_rfis", 0))),
    ]

    for metric, value in metrics:
        table_data.append([
            Paragraph(metric, styles['TableCell']),
            Paragraph(f"<b>{value}</b>", styles['TableCell']),
        ])

    table = Table(table_data, colWidths=[80*mm, 90*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    return table


def create_boq_summary_table(priced_boq, styles):
    """Create BOQ summary by package."""
    # Aggregate by package
    packages = {}
    total = 0
    for item in priced_boq:
        pkg = item.get("package", "misc").upper()
        if pkg not in packages:
            packages[pkg] = {"items": 0, "value": 0}
        packages[pkg]["items"] += 1
        packages[pkg]["value"] += item.get("amount", 0)
        total += item.get("amount", 0)

    table_data = [
        [Paragraph("<b>Package</b>", styles['TableHeader']),
         Paragraph("<b>Items</b>", styles['TableHeader']),
         Paragraph("<b>Value (INR)</b>", styles['TableHeader']),
         Paragraph("<b>%</b>", styles['TableHeader'])],
    ]

    for pkg, data in sorted(packages.items(), key=lambda x: -x[1]["value"]):
        pct = (data["value"] / total * 100) if total > 0 else 0
        table_data.append([
            Paragraph(pkg, styles['TableCell']),
            Paragraph(str(data["items"]), styles['TableCell']),
            Paragraph(format_currency(data["value"]), styles['TableCell']),
            Paragraph(f"{pct:.1f}%", styles['TableCell']),
        ])

    # Total row
    table_data.append([
        Paragraph("<b>TOTAL</b>", styles['TableCell']),
        Paragraph(f"<b>{len(priced_boq)}</b>", styles['TableCell']),
        Paragraph(f"<b>{format_currency(total)}</b>", styles['TableCell']),
        Paragraph("<b>100%</b>", styles['TableCell']),
    ])

    table = Table(table_data, colWidths=[50*mm, 25*mm, 55*mm, 25*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return table


def create_reservations_table(reservations, styles):
    """Create reservations table."""
    table_data = [
        [Paragraph("<b>#</b>", styles['TableHeader']),
         Paragraph("<b>Code</b>", styles['TableHeader']),
         Paragraph("<b>Description</b>", styles['TableHeader']),
         Paragraph("<b>Severity</b>", styles['TableHeader'])],
    ]

    severity_colors = {
        "high": DANGER_COLOR,
        "medium": WARNING_COLOR,
        "low": HexColor("#ecc94b"),
    }

    for i, res in enumerate(reservations, 1):
        severity = res.get("severity", "medium")
        sev_text = f"<font color='{severity_colors.get(severity, black)}'><b>{severity.upper()}</b></font>"

        table_data.append([
            Paragraph(str(i), styles['TableCell']),
            Paragraph(f"<b>{res.get('code', '')}</b>", styles['TableCell']),
            Paragraph(res.get("description", ""), styles['TableCell']),
            Paragraph(sev_text, styles['TableCell']),
        ])

    table = Table(table_data, colWidths=[10*mm, 25*mm, 100*mm, 25*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
    ]))
    return table


def create_rfi_table(rfis, styles):
    """Create RFI table."""
    table_data = [
        [Paragraph("<b>ID</b>", styles['TableHeader']),
         Paragraph("<b>Question</b>", styles['TableHeader']),
         Paragraph("<b>Priority</b>", styles['TableHeader']),
         Paragraph("<b>Status</b>", styles['TableHeader'])],
    ]

    priority_colors = {
        "high": DANGER_COLOR,
        "medium": WARNING_COLOR,
        "low": SUCCESS_COLOR,
    }

    for rfi in rfis:
        priority = rfi.get("priority", "medium")
        pri_text = f"<font color='{priority_colors.get(priority, black)}'><b>{priority.upper()}</b></font>"

        table_data.append([
            Paragraph(rfi.get("rfi_id", ""), styles['TableCell']),
            Paragraph(rfi.get("question", ""), styles['TableCell']),
            Paragraph(pri_text, styles['TableCell']),
            Paragraph(rfi.get("status", "open").upper(), styles['TableCell']),
        ])

    table = Table(table_data, colWidths=[20*mm, 100*mm, 20*mm, 20*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
    ]))
    return table


def create_phase_status_table(phases, styles):
    """Create phase status table."""
    phase_names = [
        ("phase16", "Owner Docs Parser"),
        ("phase17", "Owner Inputs Engine"),
        ("phase18", "BOQ Alignment"),
        ("phase19", "Pricing Engine"),
        ("phase20", "Quote Leveling"),
        ("phase21", "Prelims Generator"),
        ("phase22", "Bid Book Export"),
        ("phase23", "Bid Gate"),
        ("phase24", "Clarifications Letter"),
        ("phase25", "Package Outputs"),
    ]

    table_data = [
        [Paragraph("<b>Phase</b>", styles['TableHeader']),
         Paragraph("<b>Description</b>", styles['TableHeader']),
         Paragraph("<b>Status</b>", styles['TableHeader'])],
    ]

    status_icons = {
        "completed": ("✓", SUCCESS_COLOR),
        "skipped": ("○", HexColor("#a0aec0")),
        "error": ("✗", DANGER_COLOR),
    }

    for key, name in phase_names:
        phase_data = phases.get(key, {})
        status = phase_data.get("status", "not_run")
        icon, color = status_icons.get(status, ("?", black))

        table_data.append([
            Paragraph(key.upper(), styles['TableCell']),
            Paragraph(name, styles['TableCell']),
            Paragraph(f"<font color='{color}'><b>{icon} {status.title()}</b></font>", styles['TableCell']),
        ])

    table = Table(table_data, colWidths=[30*mm, 90*mm, 40*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
    ]))
    return table


def create_boq_detail_table(priced_boq, styles):
    """Create detailed BOQ table."""
    table_data = [
        [Paragraph("<b>SL</b>", styles['TableHeader']),
         Paragraph("<b>Description</b>", styles['TableHeader']),
         Paragraph("<b>Qty</b>", styles['TableHeader']),
         Paragraph("<b>Unit</b>", styles['TableHeader']),
         Paragraph("<b>Rate</b>", styles['TableHeader']),
         Paragraph("<b>Amount</b>", styles['TableHeader'])],
    ]

    for i, item in enumerate(priced_boq, 1):
        desc = item.get("description", "")
        if item.get("is_provisional"):
            desc = f"<i>{desc}</i> *"

        table_data.append([
            Paragraph(str(i), styles['TableCell']),
            Paragraph(desc, styles['TableCell']),
            Paragraph(f"{item.get('quantity', 0):,.2f}", styles['TableCell']),
            Paragraph(item.get("unit", ""), styles['TableCell']),
            Paragraph(f"₹{item.get('rate', 0):,.2f}", styles['TableCell']),
            Paragraph(f"₹{item.get('amount', 0):,.2f}", styles['TableCell']),
        ])

    # Total row
    total = sum(item.get("amount", 0) for item in priced_boq)
    table_data.append([
        Paragraph("", styles['TableCell']),
        Paragraph("<b>TOTAL</b>", styles['TableCell']),
        Paragraph("", styles['TableCell']),
        Paragraph("", styles['TableCell']),
        Paragraph("", styles['TableCell']),
        Paragraph(f"<b>₹{total:,.2f}</b>", styles['TableCell']),
    ])

    table = Table(table_data, colWidths=[12*mm, 70*mm, 20*mm, 15*mm, 25*mm, 28*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [white, LIGHT_BG]),
        ('BACKGROUND', (0, -1), (-1, -1), LIGHT_BG),
    ]))
    return table


def generate_pdf():
    """Generate the complete PDF report."""
    print("Loading data...")

    # Load main results
    with open(OUTPUT_DIR / "bid_engine_results.json") as f:
        results = json.load(f)

    # Load priced BOQ
    with open(OUTPUT_DIR / "phase19_pricing" / "priced_boq.json") as f:
        priced_boq = json.load(f)

    # Load gate result
    with open(OUTPUT_DIR / "phase23_bid_gate" / "bid_gate_report.json") as f:
        gate_result = json.load(f)

    # Load prelims
    with open(OUTPUT_DIR / "phase21_prelims" / "prelims_items.json") as f:
        prelims_items = json.load(f)

    # Get RFIs from gate result or create sample
    rfis = [
        {"rfi_id": "RFI-001", "question": "Confirm column C5 dimensions - mismatch between structural and architectural", "priority": "high", "status": "open"},
        {"rfi_id": "RFI-002", "question": "Provide MEP drawings for plumbing and electrical layout", "priority": "high", "status": "open"},
        {"rfi_id": "RFI-003", "question": "Clarify external development scope - driveway extent unclear", "priority": "medium", "status": "open"},
        {"rfi_id": "RFI-004", "question": "Confirm tile specifications for living areas", "priority": "low", "status": "open"},
        {"rfi_id": "RFI-005", "question": "Provide soil investigation report for foundation design", "priority": "high", "status": "open"},
        {"rfi_id": "RFI-006", "question": "Confirm ceiling height - 3.0m or 3.15m", "priority": "medium", "status": "open"},
    ]

    styles = get_styles()

    # Calculate metrics
    boq_total = sum(item.get("amount", 0) for item in priced_boq)
    prelims_total = sum(item.get("amount", 0) for item in prelims_items)
    grand_total = boq_total + prelims_total

    metrics = {
        "boq_total": boq_total,
        "prelims_total": prelims_total,
        "grand_total": grand_total,
        "built_up_area": 4500,
        "rate_per_sqft": grand_total / 48438,
        "duration_months": 14,
        "gate_score": gate_result.get("score", 0),
        "reservations_count": len(gate_result.get("reservations", [])),
        "high_priority_rfis": len([r for r in rfis if r["priority"] == "high"]),
    }

    # Build document
    print("Generating PDF...")
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("XBOQ", styles['ReportTitle']))
    story.append(Paragraph("India-First Preconstruction BOQ & Scope Tool", styles['BodyText']))
    story.append(Spacer(1, 20*mm))

    story.append(Paragraph("<b>BID SUBMISSION REPORT</b>", ParagraphStyle(
        name='CoverSubtitle',
        parent=styles['Normal'],
        fontSize=18,
        textColor=SECONDARY_COLOR,
        alignment=TA_CENTER,
    )))
    story.append(Spacer(1, 15*mm))

    story.append(Paragraph(f"<b>Project:</b> {results.get('project_name', 'Demo Project')}", styles['BodyText']))
    story.append(Paragraph(f"<b>Project ID:</b> {results.get('project_id', 'DEMO-2024-001')}", styles['BodyText']))
    story.append(Paragraph(f"<b>Location:</b> Whitefield, Bangalore", styles['BodyText']))
    story.append(Paragraph(f"<b>Built-up Area:</b> 4,500 sqm (48,438 sqft)", styles['BodyText']))
    story.append(Spacer(1, 10*mm))

    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %B %Y, %H:%M')}", styles['BodyText']))
    story.append(Spacer(1, 30*mm))

    # Status banner on cover
    story.append(create_status_banner(
        gate_result.get("status", "UNKNOWN"),
        gate_result.get("score", 0),
        styles
    ))

    story.append(PageBreak())

    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. EXECUTIVE SUMMARY", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("Key Metrics", styles['SubHeader']))
    story.append(create_metrics_table(metrics, styles))
    story.append(Spacer(1, 10*mm))

    story.append(Paragraph("BOQ Summary by Package", styles['SubHeader']))
    story.append(create_boq_summary_table(priced_boq, styles))
    story.append(Spacer(1, 10*mm))

    # Provisional items note
    provisional_items = [i for i in priced_boq if i.get("is_provisional")]
    if provisional_items:
        prov_total = sum(i.get("amount", 0) for i in provisional_items)
        story.append(Paragraph(
            f"<i>* {len(provisional_items)} provisional items totaling {format_currency(prov_total)} "
            f"({prov_total/boq_total*100:.1f}% of BOQ)</i>",
            styles['SmallText']
        ))

    story.append(PageBreak())

    # =========================================================================
    # BID GATE RESULTS
    # =========================================================================
    story.append(Paragraph("2. BID GATE RESULTS", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    story.append(create_status_banner(
        gate_result.get("status", "UNKNOWN"),
        gate_result.get("score", 0),
        styles
    ))
    story.append(Spacer(1, 10*mm))

    # Gate checks table
    story.append(Paragraph("Gate Checks", styles['SubHeader']))

    checks = gate_result.get("checks", {})
    check_data = [
        [Paragraph("<b>Check</b>", styles['TableHeader']),
         Paragraph("<b>Value</b>", styles['TableHeader']),
         Paragraph("<b>Threshold</b>", styles['TableHeader']),
         Paragraph("<b>Score</b>", styles['TableHeader'])],
    ]

    for name, check in checks.items():
        score = check.get("score", 0)
        if score >= 80:
            score_text = f"<font color='{SUCCESS_COLOR}'><b>✓ {score}</b></font>"
        elif score >= 60:
            score_text = f"<font color='{WARNING_COLOR}'><b>○ {score}</b></font>"
        else:
            score_text = f"<font color='{DANGER_COLOR}'><b>✗ {score}</b></font>"

        check_data.append([
            Paragraph(name.replace("_", " ").title(), styles['TableCell']),
            Paragraph(str(check.get("value", "")), styles['TableCell']),
            Paragraph(str(check.get("threshold", "")), styles['TableCell']),
            Paragraph(score_text, styles['TableCell']),
        ])

    check_table = Table(check_data, colWidths=[50*mm, 35*mm, 35*mm, 30*mm])
    check_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
    ]))
    story.append(check_table)
    story.append(Spacer(1, 10*mm))

    # Reservations
    reservations = gate_result.get("reservations", [])
    if reservations:
        story.append(Paragraph("Reservations", styles['SubHeader']))
        story.append(create_reservations_table(reservations, styles))

    story.append(PageBreak())

    # =========================================================================
    # RFIs
    # =========================================================================
    story.append(Paragraph("3. REQUESTS FOR INFORMATION (RFIs)", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    high_rfis = [r for r in rfis if r["priority"] == "high"]
    story.append(Paragraph(
        f"<b>{len(high_rfis)} High Priority RFIs</b> require resolution before bid finalization.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 5*mm))

    story.append(create_rfi_table(rfis, styles))
    story.append(Spacer(1, 10*mm))

    # =========================================================================
    # PHASE STATUS
    # =========================================================================
    story.append(Paragraph("4. PHASE STATUS", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph(
        "The XBOQ Bid Engine processed the following phases (16-25):",
        styles['BodyText']
    ))
    story.append(Spacer(1, 5*mm))

    story.append(create_phase_status_table(results.get("phases", {}), styles))

    story.append(PageBreak())

    # =========================================================================
    # DETAILED BOQ
    # =========================================================================
    story.append(Paragraph("5. DETAILED BILL OF QUANTITIES", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph(
        f"<b>{len(priced_boq)} line items</b> | Total: <b>{format_currency(boq_total)}</b>",
        styles['BodyText']
    ))
    story.append(Spacer(1, 5*mm))

    story.append(create_boq_detail_table(priced_boq, styles))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("<i>* Provisional items - rates based on allowances pending detailed drawings</i>", styles['SmallText']))

    story.append(PageBreak())

    # =========================================================================
    # PRELIMINARIES
    # =========================================================================
    story.append(Paragraph("6. PRELIMINARY & GENERAL ITEMS", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    prelim_data = [
        [Paragraph("<b>Item</b>", styles['TableHeader']),
         Paragraph("<b>Amount (INR)</b>", styles['TableHeader'])],
    ]

    for item in prelims_items:
        prelim_data.append([
            Paragraph(item.get("item", ""), styles['TableCell']),
            Paragraph(f"₹{item.get('amount', 0):,.2f}", styles['TableCell']),
        ])

    prelim_data.append([
        Paragraph("<b>TOTAL PRELIMS</b>", styles['TableCell']),
        Paragraph(f"<b>₹{prelims_total:,.2f}</b>", styles['TableCell']),
    ])

    prelim_table = Table(prelim_data, colWidths=[120*mm, 40*mm])
    prelim_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [white, LIGHT_BG]),
        ('BACKGROUND', (0, -1), (-1, -1), LIGHT_BG),
    ]))
    story.append(prelim_table)
    story.append(Spacer(1, 10*mm))

    story.append(Paragraph(
        f"Prelims as % of BOQ: <b>{prelims_total/boq_total*100:.1f}%</b>",
        styles['BodyText']
    ))

    story.append(PageBreak())

    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    story.append(Paragraph("7. NEXT STEPS", styles['SectionHeader']))
    story.append(Spacer(1, 5*mm))

    status = gate_result.get("status", "UNKNOWN")

    if status == "FAIL":
        steps = [
            "Review bid_gate_report.md for critical failures",
            "Resolve all high-priority RFIs",
            "Obtain missing information (MEP drawings, soil report)",
            "Re-run bid engine after resolution",
            "Do NOT submit until gate status improves",
        ]
    elif status == "PASS_WITH_RESERVATIONS":
        steps = [
            "Review all reservations in bid_gate_report.md",
            "Document reservations in clarifications letter",
            "Obtain subcontractor quotes using RFQ sheets from packages/",
            "Get management approval for submission with reservations",
            "Submit bid with clarifications letter attached",
        ]
    else:
        steps = [
            "Final review of all bid documents",
            "Obtain management approval",
            "Submit bid before deadline",
        ]

    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{i}.</b> {step}", styles['BodyText']))

    story.append(Spacer(1, 20*mm))

    # Footer
    story.append(Paragraph(
        "---",
        styles['BodyText']
    ))
    story.append(Paragraph(
        f"<i>Report generated by XBOQ Bid Engine v1.0</i>",
        styles['SmallText']
    ))
    story.append(Paragraph(
        f"<i>India-First Preconstruction BOQ & Scope Tool</i>",
        styles['SmallText']
    ))

    # Build PDF
    doc.build(story)
    print(f"PDF generated: {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    generate_pdf()
