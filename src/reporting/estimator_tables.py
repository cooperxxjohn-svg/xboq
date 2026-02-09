"""
Estimator Tables Module
Builds pandas DataFrames and exports to Excel for BOQ reporting.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO
import logging

from .estimator_output import EstimatorOutput

logger = logging.getLogger(__name__)


def build_columns_df(output: EstimatorOutput) -> pd.DataFrame:
    """
    Build columns schedule DataFrame.

    Args:
        output: EstimatorOutput with column data

    Returns:
        DataFrame with columns schedule
    """
    if not output.columns:
        return pd.DataFrame(columns=[
            'Mark', 'Size (mm)', 'Height (mm)', 'Count',
            'Concrete (m³)', 'Steel (kg)', 'Formwork (sqm)', 'Confidence'
        ])

    data = []
    for col in output.columns:
        data.append({
            'Mark': col.mark,
            'Size (mm)': f"{col.size_mm[0]} × {col.size_mm[1]}",
            'Height (mm)': col.height_mm,
            'Count': col.count,
            'Concrete (m³)': round(col.concrete_volume_m3, 4),
            'Steel (kg)': round(col.steel_kg, 1),
            'Formwork (sqm)': round(col.formwork_sqm, 2),
            'Confidence': f"{col.confidence:.0%}"
        })

    df = pd.DataFrame(data)

    # Add totals row
    totals = pd.DataFrame([{
        'Mark': 'TOTAL',
        'Size (mm)': '',
        'Height (mm)': '',
        'Count': df['Count'].sum(),
        'Concrete (m³)': round(df['Concrete (m³)'].sum(), 3),
        'Steel (kg)': round(df['Steel (kg)'].sum(), 1),
        'Formwork (sqm)': round(df['Formwork (sqm)'].sum(), 2),
        'Confidence': ''
    }])

    df = pd.concat([df, totals], ignore_index=True)
    return df


def build_footings_df(output: EstimatorOutput) -> pd.DataFrame:
    """
    Build footings schedule DataFrame.

    Args:
        output: EstimatorOutput with footing data

    Returns:
        DataFrame with footings schedule
    """
    if not output.footings:
        return pd.DataFrame(columns=[
            'Mark', 'Size L×B×D (mm)', 'Count', 'Concrete (m³)',
            'Steel (kg)', 'Formwork (sqm)', 'Excavation (m³)', 'PCC (m³)'
        ])

    data = []
    for ftg in output.footings:
        data.append({
            'Mark': ftg.mark,
            'Size L×B×D (mm)': f"{ftg.L_mm} × {ftg.B_mm} × {ftg.D_mm}",
            'Count': ftg.count,
            'Concrete (m³)': round(ftg.concrete_volume_m3, 4),
            'Steel (kg)': round(ftg.steel_kg, 1),
            'Formwork (sqm)': round(ftg.formwork_sqm, 2),
            'Excavation (m³)': round(ftg.excavation_m3, 3),
            'PCC (m³)': round(ftg.pcc_m3, 4)
        })

    df = pd.DataFrame(data)

    # Add totals row
    totals = pd.DataFrame([{
        'Mark': 'TOTAL',
        'Size L×B×D (mm)': '',
        'Count': df['Count'].sum(),
        'Concrete (m³)': round(df['Concrete (m³)'].sum(), 3),
        'Steel (kg)': round(df['Steel (kg)'].sum(), 1),
        'Formwork (sqm)': round(df['Formwork (sqm)'].sum(), 2),
        'Excavation (m³)': round(df['Excavation (m³)'].sum(), 2),
        'PCC (m³)': round(df['PCC (m³)'].sum(), 3)
    }])

    df = pd.concat([df, totals], ignore_index=True)
    return df


def build_boq_df(output: EstimatorOutput) -> pd.DataFrame:
    """
    Build BOQ DataFrame.

    Args:
        output: EstimatorOutput with BOQ lines

    Returns:
        DataFrame with BOQ
    """
    if not output.boq:
        return pd.DataFrame(columns=[
            'Item No', 'Description', 'Unit', 'Quantity', 'Remarks'
        ])

    data = []
    for boq in output.boq:
        data.append({
            'Item No': boq.item_no,
            'Description': boq.description,
            'Unit': boq.unit,
            'Quantity': boq.quantity,
            'Remarks': boq.remarks
        })

    return pd.DataFrame(data)


def build_assumptions_df(output: EstimatorOutput) -> pd.DataFrame:
    """
    Build assumptions DataFrame.

    Args:
        output: EstimatorOutput with assumptions

    Returns:
        DataFrame with assumptions
    """
    if not output.assumptions:
        return pd.DataFrame(columns=[
            'Category', 'Description', 'Assumed Value', 'Source', 'Impact'
        ])

    data = []
    for assum in output.assumptions:
        data.append({
            'Category': assum.category,
            'Description': assum.description,
            'Assumed Value': assum.assumed_value,
            'Source': assum.source,
            'Impact': assum.impact.upper()
        })

    return pd.DataFrame(data)


def build_summary_df(output: EstimatorOutput) -> pd.DataFrame:
    """
    Build summary DataFrame.

    Args:
        output: EstimatorOutput with summary

    Returns:
        DataFrame with summary
    """
    data = [
        {'Item': 'Total Columns', 'Value': output.total_columns, 'Unit': 'nos'},
        {'Item': 'Total Footings', 'Value': output.total_footings, 'Unit': 'nos'},
        {'Item': 'Total Concrete', 'Value': round(output.total_concrete_m3, 3), 'Unit': 'm³'},
        {'Item': 'Total Steel', 'Value': round(output.total_steel_kg, 1), 'Unit': 'kg'},
        {'Item': 'Total Steel', 'Value': round(output.total_steel_kg / 1000, 3), 'Unit': 'tonnes'},
        {'Item': 'Total Formwork', 'Value': round(output.total_formwork_sqm, 2), 'Unit': 'sqm'},
        {'Item': 'Total Excavation', 'Value': round(output.total_excavation_m3, 2), 'Unit': 'm³'},
        {'Item': 'Confidence Level', 'Value': f"{output.confidence:.0%}", 'Unit': ''},
    ]
    return pd.DataFrame(data)


def export_to_excel(output: EstimatorOutput, filepath: Optional[Path] = None) -> BytesIO:
    """
    Export estimator output to Excel file with multiple sheets.

    Args:
        output: EstimatorOutput to export
        filepath: Optional file path to save (if None, returns BytesIO)

    Returns:
        BytesIO buffer with Excel file
    """
    # Build all DataFrames
    columns_df = build_columns_df(output)
    footings_df = build_footings_df(output)
    boq_df = build_boq_df(output)
    assumptions_df = build_assumptions_df(output)
    summary_df = build_summary_df(output)

    # Project info DataFrame
    project_df = pd.DataFrame([
        {'Field': 'Sheet Name', 'Value': output.project.sheet_name},
        {'Field': 'Scale', 'Value': output.project.scale},
        {'Field': 'Concrete Grade', 'Value': output.materials.concrete_grade},
        {'Field': 'Steel Grade', 'Value': output.materials.steel_grade},
        {'Field': 'SBC (t/sqm)', 'Value': output.materials.soil_bearing_capacity or 'N/A'},
        {'Field': 'Processing Mode', 'Value': output.processing_mode.upper()},
    ])

    # Create Excel writer
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Write each sheet
        project_df.to_excel(writer, sheet_name='Project Info', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        columns_df.to_excel(writer, sheet_name='Columns Schedule', index=False)
        footings_df.to_excel(writer, sheet_name='Footings Schedule', index=False)
        boq_df.to_excel(writer, sheet_name='BOQ', index=False)
        assumptions_df.to_excel(writer, sheet_name='Assumptions', index=False)

        # Format columns width
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    buffer.seek(0)

    # Optionally save to file
    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(buffer.getvalue())
        buffer.seek(0)
        logger.info(f"Excel exported to: {filepath}")

    return buffer


def format_size_feet_inch(mm: float) -> str:
    """Convert mm to feet-inches string."""
    total_inches = mm / 25.4
    feet = int(total_inches // 12)
    inches = int(round(total_inches % 12))
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}'-{inches}\""


def build_footings_df_with_feet(output: EstimatorOutput) -> pd.DataFrame:
    """
    Build footings schedule with feet-inch dimensions.

    Args:
        output: EstimatorOutput with footing data

    Returns:
        DataFrame with footings in feet-inch format
    """
    if not output.footings:
        return pd.DataFrame()

    data = []
    for ftg in output.footings:
        L_ft = format_size_feet_inch(ftg.L_mm)
        B_ft = format_size_feet_inch(ftg.B_mm)
        D_ft = format_size_feet_inch(ftg.D_mm)

        data.append({
            'Mark': ftg.mark,
            'Size (ft-in)': f"{L_ft} × {B_ft}",
            'Depth': D_ft,
            'Size (mm)': f"{ftg.L_mm} × {ftg.B_mm} × {ftg.D_mm}",
            'Count': ftg.count,
            'RCC (m³)': round(ftg.concrete_volume_m3, 3),
            'Steel (kg)': round(ftg.steel_kg, 1),
        })

    df = pd.DataFrame(data)

    # Add totals
    totals = pd.DataFrame([{
        'Mark': 'TOTAL',
        'Size (ft-in)': '',
        'Depth': '',
        'Size (mm)': '',
        'Count': df['Count'].sum(),
        'RCC (m³)': round(df['RCC (m³)'].sum(), 3),
        'Steel (kg)': round(df['Steel (kg)'].sum(), 1),
    }])

    return pd.concat([df, totals], ignore_index=True)
