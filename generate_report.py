#!/usr/bin/env python3
"""
Generate PDF Report for XBOQ Demo Run

Creates a professional PDF report with all pipeline outputs.
"""

import json
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import yaml


def create_pdf_report():
    """Generate comprehensive PDF report."""

    # Setup paths
    project_dir = Path(__file__).parent / "data" / "projects" / "demo_villa"
    rules_dir = Path(__file__).parent / "rules"
    output_path = Path(__file__).parent / "out" / "demo_villa" / "XBOQ_Demo_Report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(project_dir / "rooms.json") as f:
        rooms_data = json.load(f)
    with open(project_dir / "openings.json") as f:
        openings_data = json.load(f)
    with open(rules_dir / "measurement_rules.yaml") as f:
        measurement_rules = yaml.safe_load(f)

    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d')
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2c5282')
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2d3748')
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )

    # Build content
    content = []

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph("XBOQ", title_style))
    content.append(Paragraph("Pre-Construction BOQ Engine", ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568')
    )))
    content.append(Spacer(1, 0.5*inch))
    content.append(HRFlowable(width="50%", thickness=2, color=colors.HexColor('#3182ce')))
    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph("Demo Project Report", ParagraphStyle(
        'ProjectTitle', parent=styles['Heading2'], fontSize=18, alignment=TA_CENTER
    )))
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph(f"Project: demo_villa", ParagraphStyle(
        'Info', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER
    )))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ParagraphStyle(
        'Info', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER,
        textColor=colors.HexColor('#718096')
    )))
    content.append(Spacer(1, 1*inch))

    # Summary box
    summary_data = [
        ['Total Area', '102.2 sqm (1,100 sqft)'],
        ['Rooms', '10'],
        ['Openings', '16 (9 doors, 5 windows, 2 vents)'],
        ['Estimated Value', '₹91.0 Lakhs'],
        ['Contingency', '₹6.85 Lakhs (7.5%)'],
        ['Bid Status', '✓ GO'],
    ]

    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#edf2f7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2d3748')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    content.append(summary_table)

    content.append(PageBreak())

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    content.append(Paragraph("Table of Contents", h1_style))
    content.append(Spacer(1, 0.2*inch))

    toc_items = [
        "1. Project Overview",
        "2. Room Schedule",
        "3. Openings Schedule",
        "4. Finish Specifications",
        "5. Measurement Rules (IS 1200)",
        "6. Bill of Quantities",
        "7. Risk Analysis",
        "8. Bid Strategy",
        "9. Exclusions & Assumptions",
    ]

    for item in toc_items:
        content.append(Paragraph(item, normal_style))

    content.append(PageBreak())

    # =========================================================================
    # 1. PROJECT OVERVIEW
    # =========================================================================
    content.append(Paragraph("1. Project Overview", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Project Details", h2_style))

    project_info = [
        ['Project ID', 'demo_villa'],
        ['Type', 'Residential Villa'],
        ['Profile', 'Typical (Standard Indian Residential)'],
        ['Scale', '1:100'],
        ['Total Built-up Area', '102.2 sqm'],
        ['Processing Date', datetime.now().strftime('%Y-%m-%d')],
    ]

    info_table = Table(project_info, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    content.append(info_table)
    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph("Pipeline Phases Executed", h2_style))

    phases = [
        ['Phase', 'Status', 'Description'],
        ['1. Index', '✓', 'Load project data'],
        ['2. Classify', '✓', 'Room type classification'],
        ['3. Finishes', '✓', 'Apply finish templates'],
        ['4. Measure', '✓', 'IS 1200 measurement rules'],
        ['5. BOQ', '✓', 'Generate quantities'],
        ['6. Risk', '✓', 'Package risk assessment'],
        ['7. Sensitivity', '✓', 'Rate sensitivity analysis'],
        ['8. Strategy', '✓', 'Bid strategy generation'],
        ['9. Documents', '✓', 'Exclusions & assumptions'],
    ]

    phase_table = Table(phases, colWidths=[1.5*inch, 0.7*inch, 3.5*inch])
    phase_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    content.append(phase_table)

    content.append(PageBreak())

    # =========================================================================
    # 2. ROOM SCHEDULE
    # =========================================================================
    content.append(Paragraph("2. Room Schedule", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    room_headers = ['ID', 'Label', 'Type', 'Area (sqm)', 'Perimeter (m)', 'Confidence']
    room_rows = [room_headers]

    for room in rooms_data['rooms']:
        room_rows.append([
            room['id'],
            room['label'],
            room['room_type'].title(),
            f"{room['area_sqm']:.1f}",
            f"{room['perimeter_m']:.1f}",
            f"{room['confidence']:.0%}"
        ])

    # Add total row
    total_area = sum(r['area_sqm'] for r in rooms_data['rooms'])
    room_rows.append(['', 'TOTAL', '', f"{total_area:.1f}", '', ''])

    room_table = Table(room_rows, colWidths=[0.6*inch, 1.5*inch, 1*inch, 1*inch, 1*inch, 0.9*inch])
    room_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#edf2f7')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(room_table)

    content.append(PageBreak())

    # =========================================================================
    # 3. OPENINGS SCHEDULE
    # =========================================================================
    content.append(Paragraph("3. Openings Schedule", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Doors", h2_style))

    door_headers = ['Tag', 'Description', 'Size (W×H)', 'Material', 'Conf.']
    door_rows = [door_headers]

    for op in openings_data['openings']:
        if op['type'] == 'door':
            door_rows.append([
                op['tag'],
                op['description'],
                f"{op['width_m']:.2f} × {op['height_m']:.2f}m",
                op['material'].replace('_', ' ').title(),
                f"{op['confidence']:.0%}"
            ])

    door_table = Table(door_rows, colWidths=[0.6*inch, 2*inch, 1.2*inch, 1.5*inch, 0.7*inch])
    door_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c05621')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    content.append(door_table)
    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph("Windows & Ventilators", h2_style))

    window_headers = ['Tag', 'Description', 'Size (W×H)', 'Material', 'Conf.']
    window_rows = [window_headers]

    for op in openings_data['openings']:
        if op['type'] in ['window', 'ventilator']:
            window_rows.append([
                op['tag'],
                op['description'],
                f"{op['width_m']:.2f} × {op['height_m']:.2f}m",
                op.get('material', 'N/A').replace('_', ' ').title(),
                f"{op['confidence']:.0%}"
            ])

    window_table = Table(window_rows, colWidths=[0.6*inch, 2*inch, 1.2*inch, 1.5*inch, 0.7*inch])
    window_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    content.append(window_table)

    content.append(PageBreak())

    # =========================================================================
    # 4. FINISH SPECIFICATIONS
    # =========================================================================
    content.append(Paragraph("4. Finish Specifications", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    finish_map = {
        'living': {'floor': 'Vitrified 600×600', 'wall': 'Plastic Emulsion', 'dado': '—', 'wp': '—'},
        'dining': {'floor': 'Vitrified 600×600', 'wall': 'Plastic Emulsion', 'dado': '—', 'wp': '—'},
        'bedroom': {'floor': 'Vitrified 600×600', 'wall': 'Plastic Emulsion', 'dado': '—', 'wp': '—'},
        'kitchen': {'floor': 'Vitrified 600×600', 'wall': 'Plastic Emulsion', 'dado': '600mm Ceramic', 'wp': '—'},
        'toilet': {'floor': 'Anti-skid 300×300', 'wall': 'Ceramic Tiles', 'dado': '2100mm', 'wp': '✓'},
        'balcony': {'floor': 'Anti-skid 300×300', 'wall': 'Exterior Emulsion', 'dado': '—', 'wp': '✓'},
        'pooja': {'floor': 'Marble', 'wall': 'Plastic Emulsion', 'dado': '—', 'wp': '—'},
        'utility': {'floor': 'IPS', 'wall': 'Cement Paint', 'dado': '—', 'wp': '—'},
    }

    finish_headers = ['Room', 'Floor', 'Wall', 'Dado', 'W/P']
    finish_rows = [finish_headers]

    for room in rooms_data['rooms']:
        rt = room['room_type']
        f = finish_map.get(rt, {'floor': '—', 'wall': '—', 'dado': '—', 'wp': '—'})
        finish_rows.append([
            room['label'],
            f['floor'],
            f['wall'],
            f['dado'],
            f['wp']
        ])

    finish_table = Table(finish_rows, colWidths=[1.5*inch, 1.3*inch, 1.3*inch, 1.2*inch, 0.5*inch])
    finish_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(finish_table)

    content.append(PageBreak())

    # =========================================================================
    # 5. MEASUREMENT RULES
    # =========================================================================
    content.append(Paragraph("5. Measurement Rules (IS 1200 / CPWD)", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Deduction Thresholds", h2_style))

    ded_data = [
        ['Item', 'Threshold', 'Rule'],
        ['Plaster', '> 0.5 sqm', 'IS 1200 Part 12 - Openings > 0.5 sqm deducted'],
        ['Paint', '> 0.1 sqm', 'IS 1200 Part 13 - Small openings not deducted'],
        ['Masonry', '> 0.1 sqm', 'IS 1200 Part 6 - Openings > 0.1 sqm deducted'],
        ['Tiles', '> 0.0 sqm', 'All openings deducted'],
    ]

    ded_table = Table(ded_data, colWidths=[1.2*inch, 1*inch, 3.5*inch])
    ded_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#744210')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    content.append(ded_table)
    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph("Opening Deductions Applied", h2_style))

    opening_ded_headers = ['Opening', 'Type', 'Size', 'Area (sqm)', 'Plaster Ded.']
    opening_ded_rows = [opening_ded_headers]

    total_door_area = 0
    total_window_area = 0

    for op in openings_data['openings']:
        area = op['width_m'] * op['height_m']
        deduct = '✓' if area > 0.5 else '✗'
        opening_ded_rows.append([
            op['tag'],
            op['type'].title(),
            f"{op['width_m']:.2f} × {op['height_m']:.2f}m",
            f"{area:.2f}",
            deduct
        ])
        if op['type'] == 'door':
            total_door_area += area
        elif op['type'] == 'window':
            total_window_area += area

    opening_ded_rows.append(['', 'Total Doors', '', f"{total_door_area:.2f}", ''])
    opening_ded_rows.append(['', 'Total Windows', '', f"{total_window_area:.2f}", ''])

    opening_ded_table = Table(opening_ded_rows, colWidths=[0.8*inch, 1*inch, 1.3*inch, 1*inch, 1*inch])
    opening_ded_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#553c9a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -2), (-1, -1), colors.HexColor('#f7fafc')),
        ('FONTNAME', (0, -2), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(opening_ded_table)

    content.append(PageBreak())

    # =========================================================================
    # 6. BILL OF QUANTITIES
    # =========================================================================
    content.append(Paragraph("6. Bill of Quantities", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Flooring Quantities", h2_style))

    floor_headers = ['Item', 'Description', 'Qty', 'Unit']
    floor_rows = [floor_headers]

    item_counter = 1
    for room in rooms_data['rooms']:
        floor_type = finish_map.get(room['room_type'], {}).get('floor', 'Default')
        qty = room['area_sqm'] * 1.05  # 5% wastage
        floor_rows.append([
            f'FLR-{item_counter:03d}',
            f"{floor_type} in {room['label']}",
            f"{qty:.2f}",
            'sqm'
        ])
        item_counter += 1

    floor_table = Table(floor_rows, colWidths=[1*inch, 3.5*inch, 0.8*inch, 0.6*inch])
    floor_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(floor_table)
    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph("Painting Quantities (with deductions)", h2_style))

    paint_headers = ['Item', 'Room', 'Gross', 'Deduction', 'Net', 'Unit']
    paint_rows = [paint_headers]

    for room in rooms_data['rooms']:
        gross = room['perimeter_m'] * 3.0
        room_openings = [o for o in openings_data['openings']
                        if o.get('room_left_id') == room['id'] or o.get('room_right_id') == room['id']]
        deduction = sum(o['width_m'] * o['height_m'] for o in room_openings if o['width_m'] * o['height_m'] > 0.1)
        net = gross - deduction
        paint_rows.append([
            f'PNT-{item_counter:03d}',
            room['label'],
            f"{gross:.1f}",
            f"{deduction:.1f}",
            f"{net:.1f}",
            'sqm'
        ])
        item_counter += 1

    paint_table = Table(paint_rows, colWidths=[1*inch, 1.5*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.6*inch])
    paint_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(paint_table)

    content.append(PageBreak())

    # =========================================================================
    # 7. RISK ANALYSIS
    # =========================================================================
    content.append(Paragraph("7. Risk Analysis", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Package-wise Risk Assessment", h2_style))

    risk_headers = ['Package', 'Value (₹L)', 'Risk Level', 'Contingency', 'Amount (₹L)']
    risk_rows = [risk_headers]

    packages = [
        ('Civil & Structure', 45.0, 'Medium', 0.08),
        ('Flooring', 12.0, 'Low', 0.05),
        ('Painting', 8.0, 'Low', 0.05),
        ('Plumbing', 6.0, 'Medium', 0.08),
        ('Electrical', 7.0, 'Medium', 0.08),
        ('Doors & Windows', 5.0, 'Low', 0.05),
        ('External Works', 8.0, 'High', 0.12),
    ]

    total_value = 0
    total_contingency = 0

    for pkg_name, value, risk, cont_rate in packages:
        cont_amt = value * cont_rate
        total_value += value
        total_contingency += cont_amt
        risk_rows.append([
            pkg_name,
            f"{value:.1f}",
            risk,
            f"{cont_rate*100:.0f}%",
            f"{cont_amt:.2f}"
        ])

    risk_rows.append(['TOTAL', f"{total_value:.1f}", '', '', f"{total_contingency:.2f}"])

    risk_table = Table(risk_rows, colWidths=[1.8*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c53030')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fed7d7')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(risk_table)
    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph("Rate Sensitivity Analysis", h2_style))

    sens_headers = ['Factor', 'Change', 'Impact on Cost', 'New Total (₹L)']
    sens_rows = [sens_headers]

    sensitivity = [
        ('Steel', '+10%', '+15.0%', 104.6),
        ('Cement', '+10%', '+8.0%', 98.3),
        ('Labour', '+10%', '+12.0%', 101.9),
        ('Tiles', '-10%', '-4.0%', 87.4),
    ]

    for factor, change, impact, new_total in sensitivity:
        sens_rows.append([factor, change, impact, f"{new_total:.1f}"])

    sens_table = Table(sens_rows, colWidths=[1.5*inch, 1*inch, 1.2*inch, 1.5*inch])
    sens_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d69e2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    content.append(sens_table)

    content.append(PageBreak())

    # =========================================================================
    # 8. BID STRATEGY
    # =========================================================================
    content.append(Paragraph("8. Bid Strategy", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Safe Packages (Price Aggressively - 5% Margin)", h2_style))
    safe_pkgs = [
        "• Flooring - Well-defined scope, low variation risk",
        "• Painting - Standard specs, predictable quantities",
        "• Doors & Windows - Schedule available, fixed sizes"
    ]
    for p in safe_pkgs:
        content.append(Paragraph(p, normal_style))

    content.append(Paragraph("Risky Packages (Protect Margin - 12-15%)", h2_style))
    risky_pkgs = [
        "• External Works - Site conditions unknown",
        "• Plumbing - Hidden work, coordination risk",
        "• Electrical - Specification gaps possible"
    ]
    for p in risky_pkgs:
        content.append(Paragraph(p, normal_style))

    content.append(Paragraph("Quote Requirements (Get SC quotes before submission)", h2_style))
    quotes = [
        "• Waterproofing specialist",
        "• Aluminium windows fabricator",
        "• Electrical contractor"
    ]
    for q in quotes:
        content.append(Paragraph(q, normal_style))

    content.append(Paragraph("Top Risk Drivers", h2_style))
    risks = [
        "1. Steel price volatility (no escalation clause)",
        "2. Site access restrictions (urban location)",
        "3. Weather delays (monsoon overlap)"
    ]
    for r in risks:
        content.append(Paragraph(r, normal_style))

    content.append(Spacer(1, 0.3*inch))

    # Go/No-Go box
    gonogo_data = [
        ['RECOMMENDED MARGIN', '8-10% overall'],
        ['GO/NO-GO STATUS', '✓ GO - Standard residential, manageable risk'],
    ]

    gonogo_table = Table(gonogo_data, colWidths=[2*inch, 3.5*inch])
    gonogo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#c6f6d5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#38a169')),
    ]))
    content.append(gonogo_table)

    content.append(PageBreak())

    # =========================================================================
    # 9. EXCLUSIONS & ASSUMPTIONS
    # =========================================================================
    content.append(Paragraph("9. Exclusions & Assumptions", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))

    content.append(Paragraph("Exclusions (Items NOT included in this estimate)", h2_style))

    exclusions = [
        "1. Architect/consultant fees",
        "2. Statutory approvals and permits",
        "3. Soil testing and investigation",
        "4. Furniture and furnishings",
        "5. Landscaping beyond basic planting",
        "6. Security systems",
        "7. HVAC system (provision only included)",
        "8. Generator set",
        "9. Solar system",
        "10. Water treatment plant"
    ]

    for exc in exclusions:
        content.append(Paragraph(exc, normal_style))

    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph("Assumptions (Basis of estimate)", h2_style))

    assumptions = [
        "1. 8-hour working day, 26 days/month",
        "2. Water and electricity available at site",
        "3. Clear site access for material delivery",
        "4. No rock excavation required",
        "5. Soil bearing capacity ≥ 200 kN/sqm",
        "6. No dewatering required",
        "7. Standard floor height 3.0m",
        "8. All dimensions as per architectural drawings",
        "9. Finish specifications as per tender document",
        "10. Rates valid for 90 days from bid date"
    ]

    for asm in assumptions:
        content.append(Paragraph(asm, normal_style))

    # =========================================================================
    # BUILD PDF
    # =========================================================================
    doc.build(content)

    print(f"✅ PDF Report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    create_pdf_report()
