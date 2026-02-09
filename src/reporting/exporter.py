"""
Exporter Module
Exports EstimatorOutput to Excel and JSON formats.

Excel sheets:
- Elements_Columns
- Elements_Footings
- Elements_Beams
- Requirements
- Scope_Checklist
- BOQ_Items
- Assumptions
"""

import json
from pathlib import Path
from typing import Optional
from io import BytesIO
import pandas as pd
import logging

from .estimator_output import EstimatorOutput

logger = logging.getLogger(__name__)


def build_columns_df(output: EstimatorOutput) -> pd.DataFrame:
    """Build columns schedule DataFrame."""
    if not output.columns:
        return pd.DataFrame(columns=[
            'Mark', 'Type', 'Size (mm)', 'Size (m)', 'Level', 'Count',
            'Height (mm)', 'Concrete (m³)', 'Steel (kg)', 'Formwork (m²)',
            'Qty Status', 'Dependencies', 'Confidence'
        ])

    data = []
    for col in output.columns:
        data.append({
            'Mark': col.mark,
            'Type': col.column_type,
            'Size (mm)': f"{col.size_mm[0]} × {col.size_mm[1]}",
            'Size (m)': f"{col.size_mm[0]/1000:.3f} × {col.size_mm[1]/1000:.3f}",
            'Level': col.level,
            'Count': col.count,
            'Height (mm)': col.height_mm or '-',
            'Concrete (m³)': round(col.concrete_volume_m3, 4),
            'Steel (kg)': round(col.steel_kg, 1),
            'Formwork (m²)': round(col.formwork_sqm, 2),
            'Qty Status': col.qty_status,
            'Dependencies': '; '.join(col.dependencies) if col.dependencies else '-',
            'Confidence': f"{col.confidence:.0%}"
        })

    df = pd.DataFrame(data)

    # Add totals row
    if len(df) > 0:
        totals = pd.DataFrame([{
            'Mark': 'TOTAL',
            'Type': '',
            'Size (mm)': '',
            'Size (m)': '',
            'Level': '',
            'Count': df['Count'].sum(),
            'Height (mm)': '',
            'Concrete (m³)': round(df['Concrete (m³)'].sum(), 3),
            'Steel (kg)': round(df['Steel (kg)'].sum(), 1),
            'Formwork (m²)': round(df['Formwork (m²)'].sum(), 2),
            'Qty Status': '',
            'Dependencies': '',
            'Confidence': ''
        }])
        df = pd.concat([df, totals], ignore_index=True)

    return df


def build_footings_df(output: EstimatorOutput) -> pd.DataFrame:
    """Build footings schedule DataFrame."""
    if not output.footings:
        return pd.DataFrame(columns=[
            'Mark', 'Shape', 'L (mm)', 'B (mm)', 'D (mm)',
            'L (m)', 'B (m)', 'D (m)', 'Count',
            'Concrete (m³)', 'Steel (kg)', 'Formwork (m²)',
            'Excavation (m³)', 'PCC (m³)',
            'Qty Status', 'Dependencies', 'Confidence'
        ])

    data = []
    for ftg in output.footings:
        data.append({
            'Mark': ftg.mark,
            'Shape': ftg.shape,
            'L (mm)': ftg.L_mm,
            'B (mm)': ftg.B_mm,
            'D (mm)': ftg.D_mm or '-',
            'L (m)': round(ftg.L_mm / 1000, 3),
            'B (m)': round(ftg.B_mm / 1000, 3),
            'D (m)': round(ftg.D_mm / 1000, 3) if ftg.D_mm else '-',
            'Count': ftg.count,
            'Concrete (m³)': round(ftg.concrete_volume_m3, 4),
            'Steel (kg)': round(ftg.steel_kg, 1),
            'Formwork (m²)': round(ftg.formwork_sqm, 2),
            'Excavation (m³)': round(ftg.excavation_m3, 3),
            'PCC (m³)': round(ftg.pcc_m3, 4),
            'Qty Status': ftg.qty_status,
            'Dependencies': '; '.join(ftg.dependencies) if ftg.dependencies else '-',
            'Confidence': f"{ftg.confidence:.0%}"
        })

    df = pd.DataFrame(data)

    # Add totals row
    if len(df) > 0:
        totals = pd.DataFrame([{
            'Mark': 'TOTAL',
            'Shape': '',
            'L (mm)': '',
            'B (mm)': '',
            'D (mm)': '',
            'L (m)': '',
            'B (m)': '',
            'D (m)': '',
            'Count': df['Count'].sum(),
            'Concrete (m³)': round(df['Concrete (m³)'].sum(), 3),
            'Steel (kg)': round(df['Steel (kg)'].sum(), 1),
            'Formwork (m²)': round(df['Formwork (m²)'].sum(), 2),
            'Excavation (m³)': round(df['Excavation (m³)'].sum(), 2),
            'PCC (m³)': round(df['PCC (m³)'].sum(), 3),
            'Qty Status': '',
            'Dependencies': '',
            'Confidence': ''
        }])
        df = pd.concat([df, totals], ignore_index=True)

    return df


def build_beams_df(output: EstimatorOutput) -> pd.DataFrame:
    """Build beams schedule DataFrame."""
    if not output.beams:
        return pd.DataFrame(columns=[
            'Mark', 'Type', 'Width (mm)', 'Depth (mm)', 'Span (mm)',
            'Count', 'Concrete (m³)', 'Steel (kg)', 'Formwork (m²)',
            'Qty Status', 'Dependencies', 'Confidence'
        ])

    data = []
    for beam in output.beams:
        data.append({
            'Mark': beam.mark,
            'Type': beam.beam_type,
            'Width (mm)': beam.width_mm,
            'Depth (mm)': beam.depth_mm,
            'Span (mm)': beam.span_mm or '-',
            'Count': beam.count,
            'Concrete (m³)': round(beam.concrete_volume_m3, 4),
            'Steel (kg)': round(beam.steel_kg, 1),
            'Formwork (m²)': round(beam.formwork_sqm, 2),
            'Qty Status': beam.qty_status,
            'Dependencies': '; '.join(beam.dependencies) if beam.dependencies else '-',
            'Confidence': f"{beam.confidence:.0%}"
        })

    return pd.DataFrame(data)


def build_requirements_df(output: EstimatorOutput) -> pd.DataFrame:
    """Build requirements DataFrame."""
    if not output.requirements:
        return pd.DataFrame(columns=[
            'Category', 'Requirement', 'Value', 'Confidence', 'Source Text'
        ])

    data = []
    for req in output.requirements:
        data.append({
            'Category': req.category,
            'Requirement': req.requirement,
            'Value': req.value or '-',
            'Confidence': f"{req.confidence:.0%}",
            'Source Text': req.source_text[:100] if req.source_text else '-'
        })

    return pd.DataFrame(data)


def build_scope_df(output: EstimatorOutput) -> pd.DataFrame:
    """Build scope checklist DataFrame."""
    detected = output.scope_checklist.detected or []
    missing = output.scope_checklist.missing_or_unclear or []

    # Create two-column layout
    max_len = max(len(detected), len(missing))
    detected_padded = detected + [''] * (max_len - len(detected))
    missing_padded = missing + [''] * (max_len - len(missing))

    return pd.DataFrame({
        'Detected Scope': detected_padded,
        'Missing / Unclear': missing_padded
    })


def build_boq_df(output: EstimatorOutput, include_debug: bool = False) -> pd.DataFrame:
    """
    Build BOQ items DataFrame.

    Args:
        output: EstimatorOutput with BOQ items
        include_debug: Include source, rule_fired, keywords columns

    Returns:
        DataFrame with BOQ items
    """
    if not output.boq_items:
        columns = [
            'Item Name', 'Unit', 'Qty', 'Qty Status', 'Trade',
            'Element Type', 'Basis', 'Dependencies', 'Confidence'
        ]
        if include_debug:
            columns.extend(['Source', 'Rule', 'Keywords'])
        return pd.DataFrame(columns=columns)

    data = []
    for item in output.boq_items:
        row = {
            'Item Name': item.item_name,
            'Unit': item.unit,
            'Qty': round(item.qty, 3) if item.qty is not None else '-',
            'Qty Status': item.qty_status,
            'Trade': item.trade,
            'Element Type': item.element_type,
            'Basis': item.basis,
            'Dependencies': '; '.join(item.dependencies) if item.dependencies else '-',
            'Confidence': f"{item.confidence:.0%}"
        }

        if include_debug:
            row['Source'] = getattr(item, 'source', 'explicit')
            row['Rule'] = getattr(item, 'rule_fired', '-')
            keywords = getattr(item, 'keywords_matched', [])
            row['Keywords'] = ', '.join(keywords[:3]) if keywords else '-'

        data.append(row)

    return pd.DataFrame(data)


def build_assumptions_df(output: EstimatorOutput) -> pd.DataFrame:
    """Build assumptions DataFrame."""
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
    """Build summary DataFrame."""
    data = [
        {'Item': 'Total Columns', 'Value': output.total_columns, 'Unit': 'nos'},
        {'Item': 'Total Footings', 'Value': output.total_footings, 'Unit': 'nos'},
        {'Item': 'Total Beams', 'Value': output.total_beams, 'Unit': 'nos'},
        {'Item': 'Total Concrete', 'Value': round(output.total_concrete_m3, 3), 'Unit': 'm³'},
        {'Item': 'Total Steel', 'Value': round(output.total_steel_kg, 1), 'Unit': 'kg'},
        {'Item': 'Total Steel', 'Value': round(output.total_steel_kg / 1000, 3), 'Unit': 'tonnes'},
        {'Item': 'Total Formwork', 'Value': round(output.total_formwork_sqm, 2), 'Unit': 'm²'},
        {'Item': 'Total Excavation', 'Value': round(output.total_excavation_m3, 2), 'Unit': 'm³'},
        {'Item': 'Confidence Level', 'Value': f"{output.confidence:.0%}", 'Unit': ''},
        {'Item': 'Bar Schedule', 'Value': 'Yes' if output.has_bar_schedule else 'No', 'Unit': ''},
    ]
    return pd.DataFrame(data)


def export_to_excel(
    output: EstimatorOutput,
    filepath: Optional[Path] = None
) -> BytesIO:
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
    beams_df = build_beams_df(output)
    requirements_df = build_requirements_df(output)
    scope_df = build_scope_df(output)
    boq_df = build_boq_df(output)
    assumptions_df = build_assumptions_df(output)
    summary_df = build_summary_df(output)

    # Project info DataFrame
    project_df = pd.DataFrame([
        {'Field': 'Sheet Name', 'Value': output.project.sheet_name},
        {'Field': 'Sheet Number', 'Value': output.project.sheet_number or '-'},
        {'Field': 'Scale', 'Value': output.project.scale},
        {'Field': 'Drawing Title', 'Value': output.project.drawing_title},
        {'Field': 'Concrete Grade', 'Value': output.materials.concrete_grade},
        {'Field': 'Steel Grade', 'Value': output.materials.steel_grade},
        {'Field': 'Cover (mm)', 'Value': output.materials.cover_mm},
        {'Field': 'SBC (t/sqm)', 'Value': output.materials.soil_bearing_capacity or '-'},
        {'Field': 'Processing Mode', 'Value': output.processing_mode.upper()},
    ])

    # Create Excel writer
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Write each sheet
        project_df.to_excel(writer, sheet_name='Project_Info', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        columns_df.to_excel(writer, sheet_name='Elements_Columns', index=False)
        footings_df.to_excel(writer, sheet_name='Elements_Footings', index=False)
        beams_df.to_excel(writer, sheet_name='Elements_Beams', index=False)
        requirements_df.to_excel(writer, sheet_name='Requirements', index=False)
        scope_df.to_excel(writer, sheet_name='Scope_Checklist', index=False)
        boq_df.to_excel(writer, sheet_name='BOQ_Items', index=False)
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
                adjusted_width = min(max_length + 2, 60)
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


def export_to_json(
    output: EstimatorOutput,
    filepath: Optional[Path] = None,
    indent: int = 2
) -> str:
    """
    Export estimator output to JSON.

    Args:
        output: EstimatorOutput to export
        filepath: Optional file path to save
        indent: JSON indentation

    Returns:
        JSON string
    """
    json_data = output.to_dict()
    json_str = json.dumps(json_data, indent=indent, ensure_ascii=False)

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        logger.info(f"JSON exported to: {filepath}")

    return json_str
