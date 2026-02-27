#!/usr/bin/env python3
"""
XBOQ Demo PDF Generator - STRICTLY DATA-DRIVEN
Generates a PDF report based ONLY on actual pipeline outputs.
No placeholders, no fallbacks, no fake data.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.legends import Legend
except ImportError:
    print("Installing reportlab...")
    os.system("pip install reportlab")
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.legends import Legend


class XBOQReportGenerator:
    """
    Generates STRICTLY DATA-DRIVEN PDF reports.
    Only shows what actually exists in the output directory.
    """

    # Colors
    PRIMARY = colors.HexColor('#1a365d')
    SECONDARY = colors.HexColor('#2b6cb0')
    SUCCESS = colors.HexColor('#38a169')
    WARNING = colors.HexColor('#dd6b20')
    DANGER = colors.HexColor('#c53030')
    LIGHT_BG = colors.HexColor('#f7fafc')
    MUTED = colors.HexColor('#718096')

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.story = []

        # Load core metadata FIRST
        self.metadata = self._load_json('run_metadata.json')
        self.pages_processed = 0
        self.rooms_found = 0
        self.boq_items = 0
        self.has_valid_data = False

        self._extract_core_stats()

    def _setup_styles(self):
        """Setup report styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle', parent=self.styles['Heading1'],
            fontSize=24, textColor=self.PRIMARY, alignment=TA_CENTER, spaceAfter=20
        ))
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle', parent=self.styles['Normal'],
            fontSize=12, textColor=self.SECONDARY, alignment=TA_CENTER, spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHead', parent=self.styles['Heading2'],
            fontSize=14, textColor=self.PRIMARY, spaceBefore=15, spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            name='SubHead', parent=self.styles['Heading3'],
            fontSize=11, textColor=self.SECONDARY, spaceBefore=10, spaceAfter=5
        ))
        self.styles.add(ParagraphStyle(
            name='Body', parent=self.styles['Normal'], fontSize=9, spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name='Small', parent=self.styles['Normal'],
            fontSize=8, textColor=self.MUTED
        ))
        self.styles.add(ParagraphStyle(
            name='NoData', parent=self.styles['Normal'],
            fontSize=10, textColor=self.DANGER, backColor=colors.HexColor('#fed7d7'),
            borderPadding=8, spaceBefore=5, spaceAfter=5
        ))
        self.styles.add(ParagraphStyle(
            name='Warning', parent=self.styles['Normal'],
            fontSize=10, textColor=self.WARNING, backColor=colors.HexColor('#feebc8'),
            borderPadding=8, spaceBefore=5, spaceAfter=5
        ))
        self.styles.add(ParagraphStyle(
            name='Success', parent=self.styles['Normal'],
            fontSize=10, textColor=self.SUCCESS, backColor=colors.HexColor('#c6f6d5'),
            borderPadding=8, spaceBefore=5, spaceAfter=5
        ))

    def _load_json(self, rel_path: str) -> Optional[Dict]:
        """Load JSON file. Returns None if not found or invalid."""
        filepath = self.output_dir / rel_path
        if not filepath.exists():
            return None
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _load_csv(self, rel_path: str, max_rows: int = 20) -> List[List[str]]:
        """Load CSV file. Returns empty list if not found."""
        import csv
        filepath = self.output_dir / rel_path
        if not filepath.exists():
            return []
        try:
            rows = []
            with open(filepath) as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    # Filter out empty rows
                    if any(cell.strip() for cell in row):
                        rows.append(row)
            return rows
        except IOError:
            return []

    def _file_exists(self, rel_path: str) -> bool:
        """Check if file exists and is non-empty."""
        filepath = self.output_dir / rel_path
        return filepath.exists() and filepath.stat().st_size > 0

    def _extract_core_stats(self):
        """Extract core statistics from metadata by parsing phase messages."""
        if not self.metadata:
            return

        # Try final_stats first (if available)
        stats = self.metadata.get('final_stats', {})
        if stats:
            self.pages_processed = stats.get('pages_processed', 0)
            self.rooms_found = stats.get('rooms_found', 0)
            self.boq_items = stats.get('boq_items', 0)
        else:
            # Parse from phase messages in summary
            summary = self.metadata.get('summary', {})
            phases = summary.get('phases', [])

            for phase in phases:
                msg = phase.get('message', '')
                phase_id = phase.get('phase', '')

                # Parse "Found X pages from Y files"
                if phase_id == '01_index' and 'pages' in msg:
                    import re
                    match = re.search(r'Found (\d+) pages', msg)
                    if match:
                        self.pages_processed = int(match.group(1))

                # Parse "Processed X pages, Y rooms"
                elif phase_id == '03_extract' and 'rooms' in msg:
                    import re
                    match = re.search(r'(\d+) rooms', msg)
                    if match:
                        self.rooms_found = int(match.group(1))

                # Parse "Generated X BOQ items"
                elif phase_id == '05_takeoff' and 'BOQ' in msg:
                    import re
                    match = re.search(r'Generated (\d+) BOQ', msg)
                    if match:
                        self.boq_items = int(match.group(1))

        # Check if we have valid data
        self.has_valid_data = self.pages_processed > 0

    def _make_table(self, data: List[List[str]], col_widths: List[float] = None,
                    header: bool = True) -> Optional[Table]:
        """Create a styled table from data."""
        if not data or len(data) < 1:
            return None

        # Truncate cell content
        truncated = []
        for row in data:
            truncated.append([str(c)[:40] if c else '' for c in row])

        table = Table(truncated, colWidths=col_widths)
        style_commands = [
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.LIGHT_BG]),
        ]
        if header and len(data) > 0:
            style_commands.extend([
                ('BACKGROUND', (0, 0), (-1, 0), self.PRIMARY),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ])
        table.setStyle(TableStyle(style_commands))
        return table

    def _no_data(self, section: str) -> Paragraph:
        """Return a 'no data' message."""
        return Paragraph(f"NOT GENERATED — NO DATA FROM PIPELINE ({section})", self.styles['NoData'])

    def _add_header(self):
        """Add report header with core metrics."""
        self.story.append(Paragraph("XBOQ Pipeline Report", self.styles['ReportTitle']))

        project_id = self.metadata.get('project_id', 'unknown') if self.metadata else 'unknown'
        timestamp = self.metadata.get('start_time', 'unknown') if self.metadata else 'unknown'

        self.story.append(Paragraph(f"Project: {project_id}", self.styles['ReportSubtitle']))
        self.story.append(Paragraph(f"Generated: {timestamp[:19] if len(str(timestamp)) > 19 else timestamp}", self.styles['Small']))
        self.story.append(Spacer(1, 0.3*inch))

        # CRITICAL METRICS BOX
        self.story.append(Paragraph("Pipeline Results", self.styles['SectionHead']))

        # Load measurement gate
        gate_result = self._load_json('measurement_gate_result.json')
        gate_decision = 'N/A'
        geometry_score = 'N/A'
        measured_pct = 'N/A'

        if gate_result:
            summary = gate_result.get('summary', {})
            gate_decision = summary.get('gate_decision', 'N/A')
            geometry_score = f"{summary.get('geometry_score', 0):.2f}"
            measured_pct = f"{summary.get('measured_percentage', 0):.1f}%"

        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Pages Processed', str(self.pages_processed),
             'OK' if self.pages_processed > 0 else 'NONE'],
            ['Rooms Extracted', str(self.rooms_found),
             'OK' if self.rooms_found > 0 else 'NONE'],
            ['BOQ Items', str(self.boq_items),
             'OK' if self.boq_items > 0 else 'NONE'],
            ['Geometry Score', geometry_score,
             'OK' if geometry_score not in ['N/A', '0.00'] else 'LOW'],
            ['Measured %', measured_pct,
             'OK' if measured_pct not in ['N/A', '0.0%'] else 'LOW'],
            ['Measurement Gate', gate_decision,
             gate_decision if gate_decision != 'N/A' else 'N/A'],
        ]

        table = self._make_table(metrics_data, col_widths=[2*inch, 1.5*inch, 1*inch])
        if table:
            self.story.append(table)

        self.story.append(Spacer(1, 0.2*inch))

        # EARLY EXIT CHECK
        if self.pages_processed == 0:
            self.story.append(Paragraph(
                "⚠️ NO DRAWINGS WERE PROCESSED. Cannot generate BOQ or estimate. "
                "Check input directory and file formats.",
                self.styles['NoData']
            ))
            self.story.append(Spacer(1, 0.3*inch))
            self.story.append(Paragraph(
                "Report truncated. Fix input issues and re-run pipeline.",
                self.styles['Warning']
            ))
            return False  # Signal to stop report

        return True  # Continue with full report

    def _add_phase_summary(self):
        """Add phase execution summary."""
        self.story.append(Paragraph("Phase Execution", self.styles['SectionHead']))

        if not self.metadata:
            self.story.append(self._no_data('run_metadata.json'))
            return

        # Try both formats: direct phases list or summary.phases
        phases = self.metadata.get('phases', [])
        if not phases:
            summary = self.metadata.get('summary', {})
            phases = summary.get('phases', [])

        if not phases:
            self.story.append(self._no_data('phases list'))
            return

        # Count statuses - handle both formats
        completed = 0
        skipped = 0
        failed = 0

        for p in phases:
            # New format uses 'success' and 'skipped' booleans
            if p.get('skipped'):
                skipped += 1
            elif p.get('success') == False:
                failed += 1
            elif p.get('success') == True:
                completed += 1
            # Old format uses 'status' string
            elif p.get('status') == 'completed':
                completed += 1
            elif p.get('status') == 'skipped':
                skipped += 1
            elif p.get('status') == 'failed':
                failed += 1

        self.story.append(Paragraph(
            f"Completed: {completed} | Skipped: {skipped} | Failed: {failed}",
            self.styles['Body']
        ))

        # Phase table
        table_data = [['Phase', 'Status', 'Result']]
        for p in phases[:25]:  # Limit rows
            # Handle both formats
            phase_id = p.get('phase', p.get('id', 'unknown'))
            message = p.get('message', p.get('result', ''))

            if p.get('skipped'):
                status = 'skipped'
            elif p.get('success') == False:
                status = 'failed'
            elif p.get('success') == True:
                status = 'completed'
            else:
                status = p.get('status', 'unknown')

            result = (message[:50] if message else '')

            table_data.append([
                phase_id[:20],
                status,
                result
            ])

        table = self._make_table(table_data, col_widths=[1.5*inch, 1*inch, 3*inch])
        if table:
            self.story.append(table)

    def _add_rooms_section(self):
        """Add rooms section - ONLY from actual data."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Room Extraction", self.styles['SectionHead']))

        combined = self._load_json('combined/combined_rooms.json')

        if not combined:
            self.story.append(self._no_data('combined/combined_rooms.json'))
            return

        rooms = combined.get('rooms', [])
        if not rooms:
            self.story.append(Paragraph("0 rooms extracted from drawings.", self.styles['Warning']))
            return

        self.story.append(Paragraph(f"Extracted {len(rooms)} rooms:", self.styles['Body']))

        # Room table from actual data
        table_data = [['Room Name', 'Type', 'Area (m²)', 'Perimeter (m)']]
        total_area = 0

        for room in rooms[:20]:  # Limit to 20 rows
            area = room.get('area_m2', 0)
            total_area += area
            table_data.append([
                room.get('name', 'Unknown')[:25],
                room.get('room_type', 'unknown'),
                f"{area:.1f}" if area else '0',
                f"{room.get('perimeter_m', 0):.1f}"
            ])

        if len(rooms) > 20:
            table_data.append(['...', f'+{len(rooms)-20} more', '', ''])

        table = self._make_table(table_data, col_widths=[2*inch, 1.2*inch, 1*inch, 1.2*inch])
        if table:
            self.story.append(table)

        self.story.append(Spacer(1, 0.1*inch))
        self.story.append(Paragraph(f"Total area: {total_area:.1f} m²", self.styles['Body']))

    def _add_boq_section(self):
        """Add BOQ section - ONLY from actual CSVs."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("BOQ Takeoff", self.styles['SectionHead']))

        # Check for measured BOQ
        boq_measured = self._load_csv('boq/boq_measured.csv', 15)
        boq_inferred = self._load_csv('boq/boq_inferred.csv', 10)
        boq_combined = self._load_csv('boq/boq_quantities.csv', 15)

        has_any_boq = bool(boq_measured) or bool(boq_inferred) or bool(boq_combined)

        if not has_any_boq:
            self.story.append(self._no_data('boq/*.csv files'))
            return

        # Measured quantities
        self.story.append(Paragraph("Measured Quantities (from drawings):", self.styles['SubHead']))
        if boq_measured and len(boq_measured) > 1:
            table = self._make_table(boq_measured)
            if table:
                self.story.append(table)
            self.story.append(Paragraph(f"({len(boq_measured)-1} measured items)", self.styles['Small']))
        else:
            self.story.append(Paragraph("No measured quantities. Scale may be unknown.", self.styles['Warning']))

        self.story.append(Spacer(1, 0.2*inch))

        # Inferred quantities
        self.story.append(Paragraph("Inferred Quantities (estimated/defaults):", self.styles['SubHead']))
        if boq_inferred and len(boq_inferred) > 1:
            table = self._make_table(boq_inferred)
            if table:
                self.story.append(table)
            self.story.append(Paragraph(f"({len(boq_inferred)-1} inferred items - require verification)", self.styles['Small']))
        else:
            self.story.append(Paragraph("No inferred quantities.", self.styles['Body']))

    def _add_measurement_gate_section(self):
        """Add measurement gate section."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Measurement Gate", self.styles['SectionHead']))

        gate_result = self._load_json('measurement_gate_result.json')

        if not gate_result:
            self.story.append(self._no_data('measurement_gate_result.json'))
            return

        summary = gate_result.get('summary', {})
        decision = summary.get('gate_decision', 'UNKNOWN')

        # Decision display
        if decision == 'PASS':
            self.story.append(Paragraph(f"Gate Decision: {decision}", self.styles['Success']))
        elif decision == 'WARN':
            self.story.append(Paragraph(f"Gate Decision: {decision}", self.styles['Warning']))
        else:
            self.story.append(Paragraph(f"Gate Decision: {decision}", self.styles['NoData']))

        # Metrics
        metrics = [
            ['Metric', 'Value'],
            ['Scale Status', summary.get('scale_status', 'N/A')],
            ['Measured %', f"{summary.get('measured_percentage', 0):.1f}%"],
            ['Geometry Score', f"{summary.get('geometry_score', 0):.2f}"],
            ['Total Items', str(summary.get('total_items', 0))],
            ['Measured Items', str(summary.get('measured_count', 0))],
        ]

        table = self._make_table(metrics, col_widths=[2*inch, 2*inch])
        if table:
            self.story.append(table)

        # Warnings
        warnings = gate_result.get('warnings', [])
        if warnings:
            self.story.append(Spacer(1, 0.1*inch))
            self.story.append(Paragraph("Warnings:", self.styles['SubHead']))
            for w in warnings[:5]:
                self.story.append(Paragraph(f"• {w}", self.styles['Body']))

    def _add_pricing_section(self):
        """Add pricing section."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Pricing Estimate", self.styles['SectionHead']))

        pricing_csv = self._load_csv('pricing/estimate_priced.csv', 15)

        if not pricing_csv or len(pricing_csv) < 2:
            self.story.append(self._no_data('pricing/estimate_priced.csv'))
            return

        # Show pricing table
        table = self._make_table(pricing_csv)
        if table:
            self.story.append(table)

        # Load JSON for totals
        pricing_json = self._load_json('pricing/estimate_priced.json')
        if pricing_json:
            total = pricing_json.get('total_estimate', 0)
            measured = pricing_json.get('measured_total', 0)
            self.story.append(Spacer(1, 0.1*inch))
            self.story.append(Paragraph(
                f"Total (measured only): ₹{measured/100000:.2f}L | "
                f"Including inferred: ₹{total/100000:.2f}L",
                self.styles['Body']
            ))

    def _add_mep_section(self):
        """Add MEP section - ONLY if MEP was run."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("MEP Device Takeoff", self.styles['SectionHead']))

        # Check if MEP directory exists
        mep_dir = self.output_dir / 'mep'
        if not mep_dir.exists():
            self.story.append(Paragraph("MEP phases not enabled (use --enable-mep)", self.styles['Warning']))
            return

        devices = self._load_json('mep/devices.json')
        takeoff_csv = self._load_csv('mep/mep_takeoff.csv', 15)

        if not devices and not takeoff_csv:
            self.story.append(self._no_data('mep/devices.json or mep/mep_takeoff.csv'))
            return

        # Device count
        device_list = devices.get('devices', []) if devices else []
        self.story.append(Paragraph(f"Detected {len(device_list)} devices", self.styles['Body']))

        if not device_list:
            self.story.append(Paragraph("No MEP devices detected in drawings.", self.styles['Warning']))
            return

        # Takeoff table
        if takeoff_csv and len(takeoff_csv) > 1:
            # Limit columns for readability
            limited = []
            for row in takeoff_csv:
                limited.append(row[:7] if len(row) > 7 else row)

            table = self._make_table(limited)
            if table:
                self.story.append(table)

        # MEP RFIs
        mep_rfis = self._load_csv('mep/mep_takeoff_rfis.csv', 10)
        if mep_rfis and len(mep_rfis) > 1:
            self.story.append(Spacer(1, 0.1*inch))
            self.story.append(Paragraph(f"MEP RFIs ({len(mep_rfis)-1} items need specification):", self.styles['SubHead']))
            table = self._make_table(mep_rfis)
            if table:
                self.story.append(table)

    def _add_rfi_section(self):
        """Add RFI section."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("RFI Log", self.styles['SectionHead']))

        rfi_log = self._load_json('rfi/rfi_log.json')

        if not rfi_log:
            self.story.append(self._no_data('rfi/rfi_log.json'))
            return

        rfis = rfi_log.get('rfis', [])
        if not rfis:
            self.story.append(Paragraph("No RFIs generated.", self.styles['Body']))
            return

        self.story.append(Paragraph(f"{len(rfis)} RFIs generated:", self.styles['Body']))

        table_data = [['ID', 'Category', 'Question', 'Priority']]
        for rfi in rfis[:15]:
            table_data.append([
                str(rfi.get('id', ''))[:10],
                str(rfi.get('category', ''))[:15],
                str(rfi.get('question', ''))[:40],
                str(rfi.get('priority', 'medium'))
            ])

        if len(rfis) > 15:
            table_data.append(['...', f'+{len(rfis)-15} more', '', ''])

        table = self._make_table(table_data, col_widths=[0.8*inch, 1*inch, 2.8*inch, 0.8*inch])
        if table:
            self.story.append(table)

    def _add_bid_gate_section(self):
        """Add bid gate section."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Bid Gate Assessment", self.styles['SectionHead']))

        gate_result = self._load_json('bid_gate_result.json')

        if not gate_result:
            self.story.append(self._no_data('bid_gate_result.json'))
            return

        decision = gate_result.get('decision', 'UNKNOWN')
        score = gate_result.get('score', 0)

        # Decision display
        if decision == 'GO':
            self.story.append(Paragraph(f"Decision: {decision} (Score: {score}/100)", self.styles['Success']))
        else:
            self.story.append(Paragraph(f"Decision: {decision} (Score: {score}/100)", self.styles['NoData']))

        # Blockers
        blockers = gate_result.get('blockers', [])
        if blockers:
            self.story.append(Spacer(1, 0.1*inch))
            self.story.append(Paragraph("Blockers:", self.styles['SubHead']))
            for b in blockers[:5]:
                self.story.append(Paragraph(f"🚫 {b}", self.styles['Body']))

        # Recommendations
        recommendations = gate_result.get('recommendations', [])
        if recommendations:
            self.story.append(Spacer(1, 0.1*inch))
            self.story.append(Paragraph("Recommendations:", self.styles['SubHead']))
            for r in recommendations[:5]:
                self.story.append(Paragraph(f"→ {r}", self.styles['Body']))

    def _add_files_reference(self):
        """Add output files reference - show what actually exists."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Output Files", self.styles['SectionHead']))

        # Check which files actually exist
        files_to_check = [
            ('summary.md', 'Run summary'),
            ('run_metadata.json', 'Pipeline metadata'),
            ('boq/boq_measured.csv', 'Measured quantities'),
            ('boq/boq_inferred.csv', 'Inferred quantities'),
            ('boq/boq_quantities.csv', 'Combined BOQ'),
            ('pricing/estimate_priced.csv', 'Priced estimate'),
            ('mep/mep_takeoff.csv', 'MEP takeoff'),
            ('mep/mep_takeoff.xlsx', 'MEP Excel'),
            ('mep/devices.json', 'Detected devices'),
            ('rfi/rfi_log.json', 'RFI list'),
            ('rfi/rfi_log.md', 'RFI markdown'),
            ('measurement_gate_result.json', 'Measurement gate'),
            ('bid_gate_result.json', 'Bid gate'),
            ('bid_gate_report.md', 'Bid gate report'),
            ('proof/proof_pack.md', 'Proof pack'),
            ('provenance/quantity_provenance.json', 'Quantity provenance'),
        ]

        table_data = [['File', 'Description', 'Status']]
        for filepath, desc in files_to_check:
            exists = self._file_exists(filepath)
            table_data.append([
                filepath,
                desc,
                '✓ EXISTS' if exists else '✗ MISSING'
            ])

        table = self._make_table(table_data, col_widths=[2.5*inch, 1.8*inch, 1*inch])
        if table:
            self.story.append(table)

    def _add_footer(self):
        """Add report footer."""
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Small']
        ))
        self.story.append(Paragraph(
            "XBOQ - India-First Preconstruction BOQ & Scope Tool",
            self.styles['Small']
        ))
        self.story.append(Paragraph(
            "This report shows ONLY actual pipeline outputs. No placeholders or demo data.",
            self.styles['Small']
        ))

    def generate(self, output_path: str = None) -> str:
        """Generate the PDF report."""
        if output_path is None:
            output_path = str(self.output_dir / 'XBOQ_Report.pdf')

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
        )

        # Add header and check if we should continue
        should_continue = self._add_header()

        if should_continue:
            # Full report
            self._add_phase_summary()
            self._add_rooms_section()
            self._add_boq_section()
            self._add_measurement_gate_section()
            self._add_pricing_section()
            self._add_mep_section()
            self._add_rfi_section()
            self._add_bid_gate_section()
            self._add_files_reference()

        self._add_footer()

        # Build PDF
        doc.build(self.story)

        return output_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate XBOQ Report (Strictly Data-Driven)')
    parser.add_argument('--output_dir', '-o', required=True,
                       help='Path to XBOQ output directory')
    parser.add_argument('--pdf_path', '-p', default=None,
                       help='Output PDF path')

    args = parser.parse_args()

    try:
        generator = XBOQReportGenerator(args.output_dir)
        pdf_path = generator.generate(args.pdf_path)
        print(f"✅ Report generated: {pdf_path}")
        print(f"   Pages processed: {generator.pages_processed}")
        print(f"   Rooms found: {generator.rooms_found}")
        print(f"   BOQ items: {generator.boq_items}")
        return pdf_path
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
