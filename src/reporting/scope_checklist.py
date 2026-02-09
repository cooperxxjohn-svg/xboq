"""
Scope Checklist Module
Tracks what was detected vs what's missing or unclear.

Populates scope_checklist.detected and scope_checklist.missing_or_unclear
based on extraction results.
"""

from typing import List
from .estimator_output import EstimatorOutput, ScopeChecklist
import logging

logger = logging.getLogger(__name__)


def build_scope_checklist(output: EstimatorOutput) -> ScopeChecklist:
    """
    Build scope checklist from estimator output.

    Analyzes what was extracted and what's missing.

    Args:
        output: EstimatorOutput with extraction results

    Returns:
        ScopeChecklist with detected and missing items
    """
    detected = []
    missing = []

    # =========================================================================
    # PROJECT/DRAWING INFO
    # =========================================================================
    if output.project.scale and output.project.scale != "1:100":
        detected.append(f"Scale: {output.project.scale}")
    elif output.project.scale == "1:100":
        detected.append("Scale: 1:100 (default or detected)")

    if output.project.sheet_number:
        detected.append(f"Sheet number: {output.project.sheet_number}")
    else:
        missing.append("Sheet number not detected")

    if output.project.drawing_title:
        detected.append(f"Drawing title: {output.project.drawing_title}")

    # =========================================================================
    # MATERIALS
    # =========================================================================
    if output.materials.concrete_grade:
        detected.append(f"Concrete grade: {output.materials.concrete_grade}")
    else:
        missing.append("Concrete grade not specified")

    if output.materials.steel_grade:
        detected.append(f"Steel grade: {output.materials.steel_grade}")
    else:
        missing.append("Steel/rebar grade not specified")

    if output.materials.soil_bearing_capacity:
        detected.append(f"SBC: {output.materials.soil_bearing_capacity} t/sqm")
    else:
        missing.append("Soil bearing capacity (SBC) not found")

    if output.materials.cover_mm and output.materials.cover_mm != 50:
        detected.append(f"Cover: {output.materials.cover_mm}mm")
    else:
        missing.append("Cover not specified (assuming 50mm)")

    if output.materials.exposure_class:
        detected.append(f"Exposure class: {output.materials.exposure_class}")
    else:
        missing.append("Exposure class not specified")

    # =========================================================================
    # COLUMNS
    # =========================================================================
    if output.columns:
        detected.append(f"Columns: {output.total_columns} nos ({len(output.columns)} sizes)")

        # Check for column sizes
        cols_with_size = [c for c in output.columns if c.size_mm and c.size_mm != (300, 300)]
        if cols_with_size:
            detected.append("Column sizes extracted")
        else:
            missing.append("Column sizes not found (using default 300×300)")

        # Check for column heights
        cols_with_height = [c for c in output.columns if c.height_mm is not None]
        if cols_with_height:
            detected.append("Column heights available")
        else:
            missing.append("Column heights/levels not available")

        # Check column labels
        cols_with_labels = sum(
            1 for c in output.columns
            if c.evidence.get('column_labels')
        )
        if cols_with_labels:
            detected.append(f"Column labels: {cols_with_labels} groups labeled")
    else:
        missing.append("No columns detected")

    # =========================================================================
    # FOOTINGS
    # =========================================================================
    if output.footings:
        detected.append(f"Footings: {output.total_footings} nos ({len(output.footings)} types)")

        # Check for footing sizes
        ftgs_with_size = [f for f in output.footings if f.L_mm > 0 and f.B_mm > 0]
        if ftgs_with_size:
            detected.append("Footing plan sizes (L×B) extracted")
        else:
            missing.append("Footing plan sizes not found")

        # Check for footing depths
        ftgs_with_depth = [f for f in output.footings if f.D_mm is not None]
        if len(ftgs_with_depth) == len(output.footings):
            detected.append("Footing depths available")
        elif ftgs_with_depth:
            missing.append(f"Footing depth missing for {len(output.footings) - len(ftgs_with_depth)} types")
        else:
            missing.append("Footing depths not specified (estimated from size)")

        # Check footing-column mapping
        ftgs_with_cols = [f for f in output.footings if f.column_marks]
        if ftgs_with_cols:
            detected.append("Column-footing mapping available")
        else:
            missing.append("Column-footing mapping not detected")
    else:
        missing.append("No footings detected")

    # =========================================================================
    # BEAMS
    # =========================================================================
    if output.beams:
        detected.append(f"Beams: {output.total_beams} nos ({len(output.beams)} types)")

        # Check for beam sizes
        beams_with_size = [b for b in output.beams if b.width_mm > 0 and b.depth_mm > 0]
        if beams_with_size:
            detected.append("Beam sizes (W×D) extracted")

        # Check for beam spans
        beams_with_span = [b for b in output.beams if b.span_mm is not None]
        if beams_with_span:
            detected.append("Beam spans available")
        else:
            missing.append("Beam spans not available")
    else:
        missing.append("No beams detected (may be on separate drawing)")

    # =========================================================================
    # REINFORCEMENT
    # =========================================================================
    if output.has_bar_schedule:
        detected.append("Bar bending schedule detected")
    else:
        missing.append("Bar bending schedule not found (using kg/m³ estimates)")

    # =========================================================================
    # REQUIREMENTS/NOTES
    # =========================================================================
    if output.requirements:
        req_categories = set(r.category for r in output.requirements)
        detected.append(f"Requirements: {len(output.requirements)} items in {len(req_categories)} categories")
    else:
        missing.append("No special requirements extracted from notes")

    # =========================================================================
    # BOQ ITEMS STATUS
    # =========================================================================
    if output.boq_items:
        computed = len([b for b in output.boq_items if b.qty_status == "computed"])
        partial = len([b for b in output.boq_items if b.qty_status == "partial"])
        unknown = len([b for b in output.boq_items if b.qty_status == "unknown"])

        if computed > 0:
            detected.append(f"BOQ: {computed} items fully computed")
        if partial > 0:
            missing.append(f"BOQ: {partial} items partially computed (need verification)")
        if unknown > 0:
            missing.append(f"BOQ: {unknown} items need manual input")

    # =========================================================================
    # ADDITIONAL SCOPE ITEMS TO CHECK
    # =========================================================================
    # These are common scope items that may not be detected
    standard_missing = [
        "Plinth beam schedule",
        "Slab thickness and reinforcement",
        "Staircase details",
        "Water tank/sump details",
        "Retaining wall details",
        "Expansion joint locations",
        "Construction sequence",
        "Curing requirements",
    ]

    # Only add if not already detected something similar
    for item in standard_missing:
        item_lower = item.lower()
        if not any(item_lower in d.lower() for d in detected):
            # Check if it's mentioned in requirements
            if not any(item_lower in r.requirement.lower() for r in output.requirements):
                missing.append(f"{item} (not in current drawing)")

    logger.info(f"Scope checklist: {len(detected)} detected, {len(missing)} missing/unclear")

    return ScopeChecklist(
        detected=detected,
        missing_or_unclear=missing
    )


def attach_scope_checklist(output: EstimatorOutput) -> EstimatorOutput:
    """
    Build and attach scope checklist to estimator output.

    Args:
        output: EstimatorOutput to modify

    Returns:
        Modified EstimatorOutput with scope_checklist populated
    """
    output.scope_checklist = build_scope_checklist(output)
    return output
