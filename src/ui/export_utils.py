"""
XBOQ Export Utilities
Excel, CSV, and ZIP package exports for takeoff data.
"""

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.estimate_schema import EstimatePackage, BOQItem, ScopeItem, Conflict

# Standard rebar diameters for Indian construction (IS 1786)
STANDARD_DIAMETERS = [8, 10, 12, 16, 20, 25, 28, 32, 36, 40]


# =============================================================================
# EXCEL EXPORT
# =============================================================================

def export_boq_to_excel(
    boq_items: List[BOQItem],
    drawing_name: str = "Takeoff"
) -> io.BytesIO:
    """
    Export BOQ items to Excel in CPWD-style format.

    Columns: S.No | Description | Unit | Qty | Qty Status | Dependencies | Confidence | Evidence Pages

    Args:
        boq_items: List of BOQItem objects
        drawing_name: Name for the sheet

    Returns:
        BytesIO buffer containing Excel file
    """
    rows = []
    for i, item in enumerate(boq_items, 1):
        rows.append({
            "S.No": i,
            "System": item.system.title(),
            "Subsystem": item.subsystem.title(),
            "Description": item.item_name,
            "Unit": item.unit or "-",
            "Qty": item.qty if item.qty is not None else "-",
            "Qty Status": item.qty_status.value if hasattr(item.qty_status, 'value') else str(item.qty_status),
            "Measurement Rule": item.measurement_rule or "-",
            "Dependencies": ", ".join(item.dependencies) if item.dependencies else "-",
            "Confidence": f"{item.confidence:.0%}",
            "Source": item.source,
            "Evidence Pages": ", ".join(str(e.page + 1) for e in item.evidence) if item.evidence else "-",
        })

    df = pd.DataFrame(rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='BOQ', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['BOQ']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

    buffer.seek(0)
    return buffer


def export_scope_to_excel(
    scope_items: List[ScopeItem],
    drawing_name: str = "Takeoff"
) -> io.BytesIO:
    """
    Export scope checklist to Excel.

    Args:
        scope_items: List of ScopeItem objects
        drawing_name: Name for the sheet

    Returns:
        BytesIO buffer containing Excel file
    """
    rows = []
    for item in scope_items:
        rows.append({
            "Category": item.category.value.title() if hasattr(item.category, 'value') else str(item.category),
            "Trade": item.trade,
            "Status": item.status.value.title() if hasattr(item.status, 'value') else str(item.status),
            "Reason": item.reason,
            "Confidence": f"{item.confidence:.0%}",
            "Evidence Pages": ", ".join(str(p + 1) for p in item.pages_found) if item.pages_found else "-",
        })

    df = pd.DataFrame(rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Scope Checklist', index=False)

        worksheet = writer.sheets['Scope Checklist']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

    buffer.seek(0)
    return buffer


def export_full_package_to_excel(package: EstimatePackage) -> io.BytesIO:
    """
    Export complete EstimatePackage to multi-sheet Excel.

    Sheets:
    - Summary
    - BOQ
    - Scope Checklist
    - Conflicts
    - Coverage

    Args:
        package: EstimatePackage object

    Returns:
        BytesIO buffer containing Excel file
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Property": [
                "File Name",
                "Discipline",
                "Scale",
                "Concrete Grade",
                "Drawing Confidence",
                "Total BOQ Items",
                "Computed Qty Items",
                "Unknown Qty Items",
                "Total Scope Items",
                "Detected Scope",
                "Missing Scope",
                "Conflicts",
                "High Severity Conflicts",
                "Generated At",
            ],
            "Value": [
                package.drawing.file_name,
                package.drawing.discipline.value if hasattr(package.drawing.discipline, 'value') else str(package.drawing.discipline),
                package.drawing.scale or "Not detected",
                "M25",  # Default
                f"{package.drawing.confidence_overall:.0%}",
                len(package.boq),
                sum(1 for b in package.boq if b.qty_status.value == 'computed'),
                sum(1 for b in package.boq if b.qty_status.value == 'unknown'),
                len(package.scope),
                sum(1 for s in package.scope if s.status.value == 'detected'),
                sum(1 for s in package.scope if s.status.value == 'missing'),
                len(package.conflicts),
                sum(1 for c in package.conflicts if c.severity.value == 'high'),
                package.created_at.isoformat(),
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # BOQ sheet
        boq_rows = []
        for i, item in enumerate(package.boq, 1):
            boq_rows.append({
                "S.No": i,
                "System": item.system.title(),
                "Subsystem": item.subsystem.title(),
                "Description": item.item_name,
                "Unit": item.unit or "-",
                "Qty": item.qty if item.qty is not None else "-",
                "Qty Status": item.qty_status.value,
                "Measurement Rule": item.measurement_rule or "-",
                "Dependencies": ", ".join(item.dependencies) if item.dependencies else "-",
                "Confidence": f"{item.confidence:.0%}",
                "Source": item.source,
            })
        if boq_rows:
            pd.DataFrame(boq_rows).to_excel(writer, sheet_name='BOQ', index=False)

        # Scope sheet
        scope_rows = []
        for item in package.scope:
            scope_rows.append({
                "Category": item.category.value.title(),
                "Trade": item.trade,
                "Status": item.status.value.title(),
                "Reason": item.reason,
                "Confidence": f"{item.confidence:.0%}",
            })
        if scope_rows:
            pd.DataFrame(scope_rows).to_excel(writer, sheet_name='Scope Checklist', index=False)

        # Conflicts sheet
        conflict_rows = []
        for c in package.conflicts:
            conflict_rows.append({
                "Type": c.type.value.replace('_', ' ').title(),
                "Severity": c.severity.value.upper(),
                "Description": c.description,
                "Suggested Resolution": c.suggested_resolution,
            })
        if conflict_rows:
            pd.DataFrame(conflict_rows).to_excel(writer, sheet_name='Conflicts', index=False)

        # Coverage sheet
        coverage_rows = []
        boq_lookup = {b.id: b.item_name for b in package.boq}
        for c in package.coverage:
            coverage_rows.append({
                "BOQ Item": boq_lookup.get(c.boq_item_id, c.boq_item_id)[:60],
                "Coverage Score": f"{c.coverage_score:.0%}",
                "Pages Used": ", ".join(str(p + 1) for p in c.pages_used) if c.pages_used else "-",
                "Evidence Count": len(c.contributed_by),
            })
        if coverage_rows:
            pd.DataFrame(coverage_rows).to_excel(writer, sheet_name='Coverage', index=False)

    buffer.seek(0)
    return buffer


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_boq_to_csv(boq_items: List[BOQItem]) -> str:
    """Export BOQ items to CSV string."""
    rows = []
    for i, item in enumerate(boq_items, 1):
        rows.append({
            "S.No": i,
            "System": item.system,
            "Subsystem": item.subsystem,
            "Description": item.item_name,
            "Unit": item.unit or "",
            "Qty": item.qty if item.qty is not None else "",
            "Qty Status": item.qty_status.value,
            "Dependencies": "; ".join(item.dependencies) if item.dependencies else "",
            "Confidence": f"{item.confidence:.2f}",
            "Source": item.source,
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_scope_to_csv(scope_items: List[ScopeItem]) -> str:
    """Export scope items to CSV string."""
    rows = []
    for item in scope_items:
        rows.append({
            "Category": item.category.value,
            "Trade": item.trade,
            "Status": item.status.value,
            "Reason": item.reason,
            "Confidence": f"{item.confidence:.2f}",
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# =============================================================================
# ZIP PACKAGE EXPORT
# =============================================================================

# =============================================================================
# VENDOR COLUMN SCHEDULE EXPORT
# =============================================================================

def export_vendor_column_schedule_to_excel(
    column_schedule: Any,
    drawing_name: str = "ColumnSchedule"
) -> io.BytesIO:
    """
    Export column schedule to vendor-ready Excel format.

    Normalizes:
    - Column marks (unique list)
    - Section sizes
    - Longitudinal bars (qty, diameter)
    - Ties (legs, diameter, spacing)
    - Confidence per entry

    Args:
        column_schedule: ColumnScheduleResult object
        drawing_name: Name for the file

    Returns:
        BytesIO buffer containing Excel file
    """
    rows = []

    if hasattr(column_schedule, 'entries') and column_schedule.entries:
        for entry in column_schedule.entries:
            # Parse longitudinal bars
            long_bars = []
            if entry.longitudinal_parsed:
                for rebar in entry.longitudinal_parsed:
                    long_bars.append(f"{rebar.quantity}Y{rebar.diameter_mm}")

            # Parse ties
            tie_str = "-"
            if entry.ties_parsed:
                t = entry.ties_parsed
                tie_str = f"{t.legs}L Y{t.diameter_mm}@{t.spacing_mm}"

            rows.append({
                "Row": entry.row_number,
                "Column Marks": ", ".join(entry.column_marks),
                "Count": len(entry.column_marks),
                "Section Size": entry.section_size or "-",
                "Longitudinal Bars": " + ".join(long_bars) if long_bars else entry.longitudinal_raw or "-",
                "Ties/Stirrups": tie_str,
                "Raw Longitudinal": entry.longitudinal_raw or "-",
                "Raw Ties": entry.ties_raw or "-",
                "Confidence": f"{entry.confidence:.0%}",
            })

    if not rows:
        # Return empty template
        rows = [{
            "Row": "-",
            "Column Marks": "-",
            "Count": "-",
            "Section Size": "-",
            "Longitudinal Bars": "-",
            "Ties/Stirrups": "-",
            "Raw Longitudinal": "-",
            "Raw Ties": "-",
            "Confidence": "-",
        }]

    df = pd.DataFrame(rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Column Schedule', index=False)

        # Adjust column widths
        worksheet = writer.sheets['Column Schedule']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 40)

    buffer.seek(0)
    return buffer


def export_vendor_column_schedule_to_csv(column_schedule: Any) -> str:
    """
    Export column schedule to vendor-ready CSV format.

    Args:
        column_schedule: ColumnScheduleResult object

    Returns:
        CSV string
    """
    rows = []

    if hasattr(column_schedule, 'entries') and column_schedule.entries:
        for entry in column_schedule.entries:
            # Parse longitudinal bars
            long_bars = []
            if entry.longitudinal_parsed:
                for rebar in entry.longitudinal_parsed:
                    long_bars.append(f"{rebar.quantity}Y{rebar.diameter_mm}")

            # Parse ties
            tie_str = ""
            if entry.ties_parsed:
                t = entry.ties_parsed
                tie_str = f"{t.legs}L Y{t.diameter_mm}@{t.spacing_mm}"

            rows.append({
                "Row": entry.row_number,
                "Column_Marks": ";".join(entry.column_marks),
                "Count": len(entry.column_marks),
                "Section_Size": entry.section_size or "",
                "Longitudinal_Bars": "+".join(long_bars) if long_bars else entry.longitudinal_raw or "",
                "Ties_Stirrups": tie_str,
                "Confidence": f"{entry.confidence:.2f}",
            })

    if not rows:
        return "Row,Column_Marks,Count,Section_Size,Longitudinal_Bars,Ties_Stirrups,Confidence\n"

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_vendor_schedule_summary(column_schedule: Any) -> Dict[str, Any]:
    """
    Generate summary statistics for vendor column schedule.

    Args:
        column_schedule: ColumnScheduleResult object

    Returns:
        Dictionary with summary stats
    """
    summary = {
        "total_entries": 0,
        "total_columns": 0,
        "section_sizes": set(),
        "diameters_used": set(),
        "tie_spacings": set(),
        "avg_confidence": 0.0,
        "all_column_marks": [],
    }

    if not hasattr(column_schedule, 'entries') or not column_schedule.entries:
        return summary

    summary["total_entries"] = len(column_schedule.entries)

    confidences = []
    for entry in column_schedule.entries:
        summary["total_columns"] += len(entry.column_marks)
        summary["all_column_marks"].extend(entry.column_marks)

        if entry.section_size:
            summary["section_sizes"].add(entry.section_size)

        if entry.longitudinal_parsed:
            for rebar in entry.longitudinal_parsed:
                summary["diameters_used"].add(rebar.diameter_mm)

        if entry.ties_parsed:
            summary["tie_spacings"].add(entry.ties_parsed.spacing_mm)
            summary["diameters_used"].add(entry.ties_parsed.diameter_mm)

        confidences.append(entry.confidence)

    if confidences:
        summary["avg_confidence"] = sum(confidences) / len(confidences)

    # Convert sets to sorted lists for JSON serialization
    summary["section_sizes"] = sorted(list(summary["section_sizes"]))
    summary["diameters_used"] = sorted(list(summary["diameters_used"]))
    summary["tie_spacings"] = sorted(list(summary["tie_spacings"]))

    return summary


def export_takeoff_package_zip(
    package: EstimatePackage,
    include_json: bool = True
) -> io.BytesIO:
    """
    Export complete takeoff package as ZIP file.

    Contents:
    - BOQ.xlsx
    - BOQ.csv
    - ScopeChecklist.xlsx
    - ScopeChecklist.csv
    - Conflicts.csv
    - EstimatePackage.json (full data dump)
    - README.txt

    Args:
        package: EstimatePackage object
        include_json: Include JSON dump of full package

    Returns:
        BytesIO buffer containing ZIP file
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # BOQ Excel
        boq_excel = export_boq_to_excel(package.boq, package.drawing.file_name)
        zf.writestr('BOQ.xlsx', boq_excel.getvalue())

        # BOQ CSV
        boq_csv = export_boq_to_csv(package.boq)
        zf.writestr('BOQ.csv', boq_csv)

        # Scope Excel
        scope_excel = export_scope_to_excel(package.scope, package.drawing.file_name)
        zf.writestr('ScopeChecklist.xlsx', scope_excel.getvalue())

        # Scope CSV
        scope_csv = export_scope_to_csv(package.scope)
        zf.writestr('ScopeChecklist.csv', scope_csv)

        # Conflicts CSV
        if package.conflicts:
            conflict_rows = []
            for c in package.conflicts:
                conflict_rows.append({
                    "Type": c.type.value,
                    "Severity": c.severity.value,
                    "Description": c.description,
                    "Suggested Resolution": c.suggested_resolution,
                })
            conflicts_csv = pd.DataFrame(conflict_rows).to_csv(index=False)
            zf.writestr('Conflicts.csv', conflicts_csv)

        # Full package Excel
        full_excel = export_full_package_to_excel(package)
        zf.writestr('FullTakeoff.xlsx', full_excel.getvalue())

        # JSON dump
        if include_json:
            json_data = package.model_dump_json(indent=2)
            zf.writestr('EstimatePackage.json', json_data)

        # README
        readme = f"""XBOQ Takeoff Package
====================

Generated: {datetime.now().isoformat()}
Drawing: {package.drawing.file_name}
Package ID: {package.package_id}

Contents:
---------
- BOQ.xlsx / BOQ.csv: Bill of Quantities
- ScopeChecklist.xlsx / ScopeChecklist.csv: Scope items
- Conflicts.csv: Detected issues
- FullTakeoff.xlsx: Complete multi-sheet workbook
- EstimatePackage.json: Machine-readable full data

Summary:
--------
- BOQ Items: {len(package.boq)}
- Scope Items: {len(package.scope)}
- Conflicts: {len(package.conflicts)}
- Drawing Confidence: {package.drawing.confidence_overall:.0%}

Notes:
------
- This is a scope and quantity takeoff only (NO PRICING)
- Items with "unknown" qty status need additional inputs
- Review all "needs_review" and "missing" items
- Verify measurements against original drawings

Generated by XBOQ - India-first Preconstruction Platform
"""
        zf.writestr('README.txt', readme)

    buffer.seek(0)
    return buffer
