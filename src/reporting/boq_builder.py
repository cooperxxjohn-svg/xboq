"""
BOQ Builder Module
Generates Bill of Quantities line items with dependency tracking.

3-Phase Extraction Approach:
- Phase 1 (Explicit): Direct extraction from detected elements (columns, footings, beams)
- Phase 2 (Synonym-derived): Items detected via Indian synonyms in OCR text
- Phase 3 (Inferred): Items added via scope completion rules (if footings -> excavation, PCC, etc.)

Rules:
- Excavation: Only if foundation depth exists, else dependency flag
- PCC: Only if thickness found in notes, else placeholder
- RCC: Computed from L*B*D*count where D exists
- Formwork: Computed from perimeter*D*count
- Reinforcement: Only if bar schedule detected, else placeholder
"""

from typing import List, Optional, Dict, Any, Tuple
from .estimator_output import EstimatorOutput, BOQItem
from .synonyms_india import (
    detect_terms_in_text,
    get_probable_items_from_text,
    BOQ_TEMPLATES,
    INDIAN_SYNONYMS
)
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 1: EXPLICIT EXTRACTION FROM DETECTED ELEMENTS
# =============================================================================

def _build_explicit_boq_items(output: EstimatorOutput) -> List[BOQItem]:
    """
    Phase 1: Generate BOQ items directly from detected elements.

    Args:
        output: EstimatorOutput with element data

    Returns:
        List of BOQItem with source='explicit'
    """
    boq_items = []

    # =========================================================================
    # A. EARTHWORK - EXCAVATION
    # =========================================================================
    if output.footings:
        total_excavation = sum(f.excavation_m3 for f in output.footings)
        has_depth = all(f.D_mm is not None for f in output.footings)

        # Check if any footing has depth missing originally
        depth_estimated = any(
            "footing depth estimated" in ' '.join(f.dependencies)
            for f in output.footings
        )

        if total_excavation > 0:
            boq_items.append(BOQItem(
                item_name="Excavation for foundations in all types of soil (excluding rock)",
                unit="m3",
                qty=round(total_excavation, 2) if has_depth or depth_estimated else None,
                qty_status="computed" if has_depth else ("partial" if depth_estimated else "unknown"),
                basis="(L+0.6m) × (B+0.6m) × (D+PCC) × count per footing type",
                dependencies=[] if has_depth else ["foundation depth missing or estimated"],
                confidence=0.9 if has_depth else 0.6,
                evidence=[f"Total footings: {output.total_footings}"],
                trade="civil",
                element_type="Footing",
                source="explicit",
                rule_fired="EXPLICIT_FTG_EXCAVATION"
            ))

    # =========================================================================
    # B. CONCRETE WORK
    # =========================================================================

    # B.1 PCC Below Footings
    if output.footings:
        total_pcc = sum(f.pcc_m3 for f in output.footings)
        pcc_thickness_specified = not any(
            "PCC thickness not specified" in ' '.join(f.dependencies)
            for f in output.footings
        )

        if pcc_thickness_specified and total_pcc > 0:
            boq_items.append(BOQItem(
                item_name="PCC 1:4:8 (M10) below footings, 100mm thick",
                unit="m3",
                qty=round(total_pcc, 3),
                qty_status="computed",
                basis="(L+0.3m) × (B+0.3m) × 0.1m × count",
                dependencies=[],
                confidence=0.9,
                evidence=[f"{output.total_footings} footings"],
                trade="rcc",
                element_type="Footing",
                source="explicit",
                rule_fired="EXPLICIT_FTG_PCC"
            ))
        else:
            boq_items.append(BOQItem(
                item_name="PCC (blinding) below footings - thickness not specified",
                unit="m3",
                qty=None,
                qty_status="unknown",
                basis="Requires PCC thickness from drawing notes",
                dependencies=["PCC thickness not specified in notes"],
                confidence=0.4,
                evidence=[],
                trade="rcc",
                element_type="Footing",
                source="explicit",
                rule_fired="EXPLICIT_FTG_PCC_UNKNOWN"
            ))

    # B.2 RCC in Footings
    if output.footings:
        total_ftg_concrete = sum(f.concrete_volume_m3 for f in output.footings)
        ftg_deps = []
        for f in output.footings:
            ftg_deps.extend(f.dependencies)

        has_all_dims = all(f.D_mm is not None for f in output.footings)

        boq_items.append(BOQItem(
            item_name=f"RCC {output.materials.concrete_grade} in isolated footings "
                     f"including centering, shuttering, and finishing",
            unit="m3",
            qty=round(total_ftg_concrete, 3) if total_ftg_concrete > 0 else None,
            qty_status="computed" if has_all_dims and not ftg_deps else "partial",
            basis="L × B × D × count per footing type",
            dependencies=list(set(ftg_deps)) if ftg_deps else [],
            confidence=0.9 if has_all_dims else 0.65,
            evidence=[f"{output.total_footings} nos footings", f"{len(output.footings)} types"],
            trade="rcc",
            element_type="Footing",
            source="explicit",
            rule_fired="EXPLICIT_FTG_RCC"
        ))

    # B.3 RCC in Columns
    if output.columns:
        total_col_concrete = sum(c.concrete_volume_m3 for c in output.columns)
        col_deps = []
        for c in output.columns:
            col_deps.extend(c.dependencies)

        has_height = all(c.height_mm is not None for c in output.columns)

        if has_height and total_col_concrete > 0:
            boq_items.append(BOQItem(
                item_name=f"RCC {output.materials.concrete_grade} in columns "
                         f"including centering, shuttering, and finishing",
                unit="m3",
                qty=round(total_col_concrete, 3),
                qty_status="computed" if not col_deps else "partial",
                basis="W × D × H × count per column size",
                dependencies=list(set(col_deps)) if col_deps else [],
                confidence=0.9 if not col_deps else 0.65,
                evidence=[f"{output.total_columns} nos columns"],
                trade="rcc",
                element_type="Column",
                source="explicit",
                rule_fired="EXPLICIT_COL_RCC"
            ))
        else:
            boq_items.append(BOQItem(
                item_name=f"RCC {output.materials.concrete_grade} in columns",
                unit="m3",
                qty=None,
                qty_status="unknown",
                basis="W × D × H × count",
                dependencies=["column height/level not available"],
                confidence=0.4,
                evidence=[f"{output.total_columns} columns detected"],
                trade="rcc",
                element_type="Column",
                source="explicit",
                rule_fired="EXPLICIT_COL_RCC_UNKNOWN"
            ))

    # B.4 RCC in Beams (if detected)
    if output.beams:
        total_beam_concrete = sum(b.concrete_volume_m3 for b in output.beams)
        beam_deps = []
        for b in output.beams:
            beam_deps.extend(b.dependencies)

        has_span = all(b.span_mm is not None for b in output.beams)

        if has_span and total_beam_concrete > 0:
            boq_items.append(BOQItem(
                item_name=f"RCC {output.materials.concrete_grade} in beams "
                         f"including centering, shuttering, and finishing",
                unit="m3",
                qty=round(total_beam_concrete, 3),
                qty_status="computed",
                basis="W × D × Span × count",
                dependencies=[],
                confidence=0.85,
                evidence=[f"{output.total_beams} beams"],
                trade="rcc",
                element_type="Beam",
                source="explicit",
                rule_fired="EXPLICIT_BEAM_RCC"
            ))
        elif output.beams:
            boq_items.append(BOQItem(
                item_name=f"RCC {output.materials.concrete_grade} in beams",
                unit="m3",
                qty=None,
                qty_status="unknown",
                basis="W × D × Span × count",
                dependencies=["beam span not available"],
                confidence=0.4,
                evidence=[f"{len(output.beams)} beam types detected"],
                trade="rcc",
                element_type="Beam",
                source="explicit",
                rule_fired="EXPLICIT_BEAM_RCC_UNKNOWN"
            ))

    # =========================================================================
    # C. REINFORCEMENT STEEL
    # =========================================================================

    if output.has_bar_schedule:
        # Bar schedule detected - can compute detailed steel
        if output.footings:
            total_ftg_steel = sum(f.steel_kg for f in output.footings)
            boq_items.append(BOQItem(
                item_name=f"Steel reinforcement {output.materials.steel_grade} for footings "
                         f"including cutting, bending, binding",
                unit="kg",
                qty=round(total_ftg_steel, 1),
                qty_status="computed",
                basis="From bar bending schedule",
                dependencies=[],
                confidence=0.95,
                evidence=["Bar schedule detected"],
                trade="steel",
                element_type="Footing",
                source="explicit",
                rule_fired="EXPLICIT_FTG_STEEL_BBS"
            ))

        if output.columns:
            total_col_steel = sum(c.steel_kg for c in output.columns)
            boq_items.append(BOQItem(
                item_name=f"Steel reinforcement {output.materials.steel_grade} for columns "
                         f"including cutting, bending, binding",
                unit="kg",
                qty=round(total_col_steel, 1),
                qty_status="computed",
                basis="From bar bending schedule",
                dependencies=[],
                confidence=0.95,
                evidence=["Bar schedule detected"],
                trade="steel",
                element_type="Column",
                source="explicit",
                rule_fired="EXPLICIT_COL_STEEL_BBS"
            ))
    else:
        # No bar schedule - add placeholder with estimated qty from ratios
        if output.footings:
            total_ftg_steel = sum(f.steel_kg for f in output.footings)
            boq_items.append(BOQItem(
                item_name=f"Steel reinforcement {output.materials.steel_grade} for footings "
                         f"including cutting, bending, binding",
                unit="kg",
                qty=round(total_ftg_steel, 1) if total_ftg_steel > 0 else None,
                qty_status="partial" if total_ftg_steel > 0 else "unknown",
                basis="Estimated @ 80 kg/m³ (requires bar schedule for accuracy)",
                dependencies=["requires bar schedule for accurate quantity"],
                confidence=0.55,
                evidence=["Estimated from kg/m³ ratio"],
                trade="steel",
                element_type="Footing",
                source="explicit",
                rule_fired="EXPLICIT_FTG_STEEL_RATIO"
            ))

        if output.columns:
            total_col_steel = sum(c.steel_kg for c in output.columns)
            boq_items.append(BOQItem(
                item_name=f"Steel reinforcement {output.materials.steel_grade} for columns "
                         f"including cutting, bending, binding",
                unit="kg",
                qty=round(total_col_steel, 1) if total_col_steel > 0 else None,
                qty_status="partial" if total_col_steel > 0 else "unknown",
                basis="Estimated @ 120 kg/m³ (requires bar schedule for accuracy)",
                dependencies=["requires bar schedule for accurate quantity"],
                confidence=0.55,
                evidence=["Estimated from kg/m³ ratio"],
                trade="steel",
                element_type="Column",
                source="explicit",
                rule_fired="EXPLICIT_COL_STEEL_RATIO"
            ))

    # =========================================================================
    # D. FORMWORK
    # =========================================================================

    # D.1 Formwork for Footings
    if output.footings:
        total_ftg_formwork = sum(f.formwork_sqm for f in output.footings)
        has_depth = all(f.D_mm is not None for f in output.footings)

        if total_ftg_formwork > 0:
            boq_items.append(BOQItem(
                item_name="Centering and shuttering for footings (including stripping)",
                unit="m2",
                qty=round(total_ftg_formwork, 2),
                qty_status="computed" if has_depth else "partial",
                basis="Perimeter × Depth × count (side formwork)",
                dependencies=[] if has_depth else ["footing depth estimated"],
                confidence=0.85 if has_depth else 0.65,
                evidence=[f"Side formwork for {output.total_footings} footings"],
                trade="formwork",
                element_type="Footing",
                source="explicit",
                rule_fired="EXPLICIT_FTG_FORMWORK"
            ))

    # D.2 Formwork for Columns
    if output.columns:
        total_col_formwork = sum(c.formwork_sqm for c in output.columns)
        has_height = all(c.height_mm is not None for c in output.columns)

        if has_height and total_col_formwork > 0:
            boq_items.append(BOQItem(
                item_name="Centering and shuttering for columns (including stripping)",
                unit="m2",
                qty=round(total_col_formwork, 2),
                qty_status="computed",
                basis="Perimeter × Height × count (all 4 sides)",
                dependencies=[],
                confidence=0.9,
                evidence=[f"All sides for {output.total_columns} columns"],
                trade="formwork",
                element_type="Column",
                source="explicit",
                rule_fired="EXPLICIT_COL_FORMWORK"
            ))
        elif output.columns:
            boq_items.append(BOQItem(
                item_name="Centering and shuttering for columns (including stripping)",
                unit="m2",
                qty=None,
                qty_status="unknown",
                basis="Perimeter × Height × count",
                dependencies=["column height/level not available"],
                confidence=0.4,
                evidence=[f"{output.total_columns} columns detected"],
                trade="formwork",
                element_type="Column",
                source="explicit",
                rule_fired="EXPLICIT_COL_FORMWORK_UNKNOWN"
            ))

    # D.3 Formwork for Beams
    if output.beams:
        total_beam_formwork = sum(b.formwork_sqm for b in output.beams)
        has_span = all(b.span_mm is not None for b in output.beams)

        if has_span and total_beam_formwork > 0:
            boq_items.append(BOQItem(
                item_name="Centering and shuttering for beams (including stripping)",
                unit="m2",
                qty=round(total_beam_formwork, 2),
                qty_status="computed",
                basis="(2×Depth + Width) × Span × count",
                dependencies=[],
                confidence=0.85,
                evidence=[f"{output.total_beams} beams"],
                trade="formwork",
                element_type="Beam",
                source="explicit",
                rule_fired="EXPLICIT_BEAM_FORMWORK"
            ))

    return boq_items


# =============================================================================
# PHASE 2: SYNONYM-DERIVED ITEMS FROM OCR TEXT
# =============================================================================

def _build_synonym_derived_items(
    notes_text: str,
    existing_items: List[BOQItem],
    output: EstimatorOutput
) -> List[BOQItem]:
    """
    Phase 2: Detect additional items from OCR text using Indian synonyms.

    Args:
        notes_text: Combined OCR text from notes/legends/schedules
        existing_items: Items already generated from Phase 1
        output: EstimatorOutput for context

    Returns:
        List of additional BOQItem with source='synonym'
    """
    if not notes_text:
        return []

    boq_items = []

    # Get probable items from text
    probable_items = get_probable_items_from_text(notes_text)

    # Get existing item keys to avoid duplicates
    existing_keys = set()
    for item in existing_items:
        # Create key from trade + element type
        key = f"{item.trade}_{item.element_type}".lower()
        existing_keys.add(key)
        # Also add by item name keywords
        for keyword in ['excavation', 'pcc', 'rcc', 'steel', 'formwork', 'shuttering',
                       'waterproof', 'anti_termite', 'backfill', 'curing', 'disposal']:
            if keyword in item.item_name.lower():
                existing_keys.add(keyword)

    # Map detected terms to BOQ items
    term_to_template = {
        'excavation': 'excavation_footing',
        'pcc': 'pcc_footing',
        'backfill': 'backfill',
        'disposal': 'disposal',
        'anti_termite': 'anti_termite',
        'waterproofing': 'waterproofing_dpc',
        'curing': 'curing',
    }

    for term, confidence in probable_items.items():
        # Skip if already covered by explicit items
        if term in existing_keys:
            continue

        # Get template if available
        template_key = term_to_template.get(term)
        if template_key and template_key in BOQ_TEMPLATES:
            template = BOQ_TEMPLATES[template_key]

            # Get matched keywords for evidence
            matches = detect_terms_in_text(notes_text)
            matched_keywords = [m.matched_text for m in matches if m.standard_term == term]

            boq_items.append(BOQItem(
                item_name=template['item_name'],
                unit=template['unit'],
                qty=None,  # Quantity not available from text
                qty_status="unknown",
                basis=template.get('basis', 'From drawing notes'),
                dependencies=["quantity not extractable from text"],
                confidence=min(0.75, confidence),  # Cap at 0.75 for synonym-derived
                evidence=[f"Keyword detected: {', '.join(matched_keywords[:3])}"],
                trade=template.get('trade', 'civil'),
                element_type=template.get('element_type', 'General'),
                source="synonym",
                rule_fired=f"SYNONYM_{term.upper()}",
                keywords_matched=matched_keywords[:5]
            ))

            existing_keys.add(term)

    # Check for special items mentioned in notes
    notes_upper = notes_text.upper()

    # Waterproofing
    if 'waterproof' not in existing_keys:
        if any(kw in notes_upper for kw in ['WATERPROOF', 'WATER PROOF', 'DPC', 'DAMP PROOF', 'MEMBRANE']):
            matched = [kw for kw in ['waterproofing', 'DPC', 'membrane'] if kw.upper() in notes_upper]
            boq_items.append(BOQItem(
                item_name="Waterproofing / DPC at plinth level",
                unit="m2",
                qty=None,
                qty_status="unknown",
                basis="Wall length × thickness",
                dependencies=["area to be measured from plinth plan"],
                confidence=0.70,
                evidence=[f"Keyword detected: {', '.join(matched)}"],
                trade="special",
                element_type="General",
                source="synonym",
                rule_fired="SYNONYM_WATERPROOFING",
                keywords_matched=matched
            ))
            existing_keys.add('waterproof')

    # Anti-termite
    if 'anti_termite' not in existing_keys:
        if any(kw in notes_upper for kw in ['ANTI TERMITE', 'ANTI-TERMITE', 'TERMITE TREATMENT', 'SOIL TREATMENT']):
            matched = [kw for kw in ['anti-termite', 'termite treatment'] if kw.upper().replace('-', ' ') in notes_upper.replace('-', ' ')]
            boq_items.append(BOQItem(
                item_name="Anti-termite treatment in foundation trenches and plinth as per IS 6313",
                unit="m2",
                qty=None,
                qty_status="unknown",
                basis="Plan area of foundations + plinth",
                dependencies=["area to be measured from plan"],
                confidence=0.70,
                evidence=[f"Keyword detected: anti-termite"],
                trade="special",
                element_type="General",
                source="synonym",
                rule_fired="SYNONYM_ANTI_TERMITE",
                keywords_matched=matched or ['anti-termite']
            ))
            existing_keys.add('anti_termite')

    # Curing
    if 'curing' not in existing_keys:
        if any(kw in notes_upper for kw in ['CURING', 'WATER CURING', 'WET CURING', 'PONDING']):
            matched = [kw for kw in ['curing', 'water curing', 'ponding'] if kw.upper() in notes_upper]
            boq_items.append(BOQItem(
                item_name="Curing of concrete surfaces for minimum 7 days",
                unit="m2",
                qty=None,
                qty_status="unknown",
                basis="Total concrete surface area",
                dependencies=["surface area derived from concrete quantities"],
                confidence=0.65,
                evidence=[f"Keyword detected: {', '.join(matched)}"],
                trade="rcc",
                element_type="General",
                source="synonym",
                rule_fired="SYNONYM_CURING",
                keywords_matched=matched
            ))

    logger.info(f"Phase 2: Generated {len(boq_items)} synonym-derived items")
    return boq_items


# =============================================================================
# PHASE 3: INFERRED ITEMS FROM SCOPE COMPLETION RULES
# =============================================================================

def _build_inferred_items(
    output: EstimatorOutput,
    existing_items: List[BOQItem]
) -> List[BOQItem]:
    """
    Phase 3: Add inferred items based on scope completion rules.

    Rules:
    - If footings detected -> excavation, PCC, backfill likely
    - If columns detected -> formwork required
    - Any RCC work -> curing required

    Args:
        output: EstimatorOutput with element data
        existing_items: Items from Phase 1 + Phase 2

    Returns:
        List of additional BOQItem with source='inferred'
    """
    boq_items = []

    # Get existing item signatures
    existing_sigs = set()
    for item in existing_items:
        sig = f"{item.trade}_{item.element_type}_{item.item_name[:30]}".lower()
        existing_sigs.add(sig)
        # Also track by type
        if 'excavation' in item.item_name.lower():
            existing_sigs.add('has_excavation')
        if 'pcc' in item.item_name.lower() or 'blinding' in item.item_name.lower():
            existing_sigs.add('has_pcc')
        if 'backfill' in item.item_name.lower() or 'filling' in item.item_name.lower():
            existing_sigs.add('has_backfill')
        if 'disposal' in item.item_name.lower():
            existing_sigs.add('has_disposal')
        if 'curing' in item.item_name.lower():
            existing_sigs.add('has_curing')
        if 'anti' in item.item_name.lower() and 'termite' in item.item_name.lower():
            existing_sigs.add('has_anti_termite')

    # Rule: If footings detected -> backfill likely
    if output.footings and 'has_backfill' not in existing_sigs:
        total_excavation = sum(f.excavation_m3 for f in output.footings)
        total_concrete = sum(f.concrete_volume_m3 + f.pcc_m3 for f in output.footings)
        estimated_backfill = max(0, total_excavation - total_concrete)

        boq_items.append(BOQItem(
            item_name="Filling in plinth/foundation with excavated earth including watering, ramming and compaction",
            unit="m3",
            qty=round(estimated_backfill, 2) if estimated_backfill > 0 else None,
            qty_status="partial" if estimated_backfill > 0 else "unknown",
            basis="Excavation volume - Concrete volume",
            dependencies=["exact backfill volume requires site measurement"],
            confidence=0.55,
            evidence=["Inferred: footings detected -> backfill required"],
            trade="civil",
            element_type="General",
            source="inferred",
            rule_fired="INFER_FTG_BACKFILL"
        ))
        existing_sigs.add('has_backfill')

    # Rule: If backfill exists -> disposal likely for surplus
    if output.footings and 'has_disposal' not in existing_sigs:
        boq_items.append(BOQItem(
            item_name="Carting away and disposal of surplus excavated earth beyond 50m lead",
            unit="m3",
            qty=None,
            qty_status="unknown",
            basis="Excavation volume - Backfill volume",
            dependencies=["surplus earth quantity depends on actual backfill"],
            confidence=0.50,
            evidence=["Inferred: excavation -> surplus earth disposal"],
            trade="civil",
            element_type="General",
            source="inferred",
            rule_fired="INFER_DISPOSAL"
        ))
        existing_sigs.add('has_disposal')

    # Rule: If any RCC work -> curing required
    has_rcc = any(
        'rcc' in item.item_name.lower() or 'concrete' in item.item_name.lower()
        for item in existing_items
    )
    if has_rcc and 'has_curing' not in existing_sigs:
        boq_items.append(BOQItem(
            item_name="Curing of concrete surfaces for minimum 7 days by water spraying/ponding",
            unit="m2",
            qty=None,
            qty_status="unknown",
            basis="Total concrete surface area",
            dependencies=["surface area derived from concrete work"],
            confidence=0.50,
            evidence=["Inferred: RCC work detected -> curing required"],
            trade="rcc",
            element_type="General",
            source="inferred",
            rule_fired="INFER_CURING"
        ))
        existing_sigs.add('has_curing')

    # Rule: If foundation work -> anti-termite treatment often required (regional)
    # Only add if not already present and if we have footings
    if output.footings and 'has_anti_termite' not in existing_sigs:
        boq_items.append(BOQItem(
            item_name="Anti-termite treatment in foundation trenches and plinth (if applicable as per site conditions)",
            unit="m2",
            qty=None,
            qty_status="unknown",
            basis="Plan area of foundations + plinth",
            dependencies=["applicability based on site/soil conditions"],
            confidence=0.40,
            evidence=["Inferred: foundation work -> anti-termite may be required"],
            trade="special",
            element_type="General",
            source="inferred",
            rule_fired="INFER_ANTI_TERMITE"
        ))

    logger.info(f"Phase 3: Generated {len(boq_items)} inferred items")
    return boq_items


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_boq_items(
    output: EstimatorOutput,
    notes_text: str = "",
    high_recall: bool = False
) -> List[BOQItem]:
    """
    Generate BOQ-ready line items using 3-phase extraction.

    Phase 1: Explicit extraction from detected elements
    Phase 2: Synonym-derived items from OCR text (if high_recall)
    Phase 3: Inferred items from scope completion rules (if high_recall)

    Args:
        output: EstimatorOutput with element data
        notes_text: Combined OCR text for synonym detection
        high_recall: Enable phases 2 and 3 for maximum coverage

    Returns:
        List of BOQItem with dependencies tracked
    """
    all_items = []

    # Phase 1: Always run - explicit extraction
    logger.info("BOQ Builder: Phase 1 - Explicit extraction")
    explicit_items = _build_explicit_boq_items(output)
    all_items.extend(explicit_items)
    logger.info(f"Phase 1 complete: {len(explicit_items)} explicit items")

    if high_recall:
        # Phase 2: Synonym-derived items
        logger.info("BOQ Builder: Phase 2 - Synonym detection")
        synonym_items = _build_synonym_derived_items(notes_text, all_items, output)
        all_items.extend(synonym_items)
        logger.info(f"Phase 2 complete: {len(synonym_items)} synonym-derived items")

        # Phase 3: Inferred items
        logger.info("BOQ Builder: Phase 3 - Inference rules")
        inferred_items = _build_inferred_items(output, all_items)
        all_items.extend(inferred_items)
        logger.info(f"Phase 3 complete: {len(inferred_items)} inferred items")

    # Sort by confidence (highest first), then by element type
    element_order = {'Footing': 1, 'Column': 2, 'Beam': 3, 'General': 4}
    all_items.sort(key=lambda x: (element_order.get(x.element_type, 5), -x.confidence))

    logger.info(f"BOQ Builder: Total {len(all_items)} items generated")
    return all_items


def attach_boq_items(
    output: EstimatorOutput,
    notes_text: str = "",
    high_recall: bool = False
) -> EstimatorOutput:
    """
    Build and attach BOQ items to estimator output.

    Args:
        output: EstimatorOutput to modify
        notes_text: Combined OCR text for synonym detection
        high_recall: Enable high recall mode

    Returns:
        Modified EstimatorOutput with boq_items populated
    """
    output.boq_items = build_boq_items(output, notes_text, high_recall)
    return output


def get_boq_extraction_stats(boq_items: List[BOQItem]) -> Dict[str, Any]:
    """
    Get statistics about BOQ extraction by source.

    Args:
        boq_items: List of BOQ items

    Returns:
        Dict with counts by source, confidence distribution, etc.
    """
    stats = {
        'total': len(boq_items),
        'by_source': {'explicit': 0, 'synonym': 0, 'inferred': 0},
        'by_status': {'computed': 0, 'partial': 0, 'unknown': 0},
        'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        'by_trade': {},
        'by_element': {}
    }

    for item in boq_items:
        # By source
        source = getattr(item, 'source', 'explicit')
        if source in stats['by_source']:
            stats['by_source'][source] += 1

        # By status
        if item.qty_status in stats['by_status']:
            stats['by_status'][item.qty_status] += 1

        # By confidence
        if item.confidence >= 0.85:
            stats['by_confidence']['high'] += 1
        elif item.confidence >= 0.60:
            stats['by_confidence']['medium'] += 1
        else:
            stats['by_confidence']['low'] += 1

        # By trade
        if item.trade not in stats['by_trade']:
            stats['by_trade'][item.trade] = 0
        stats['by_trade'][item.trade] += 1

        # By element
        if item.element_type not in stats['by_element']:
            stats['by_element'][item.element_type] = 0
        stats['by_element'][item.element_type] += 1

    return stats
