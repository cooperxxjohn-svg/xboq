"""
Scope Completion Engine
Ensures all expected BOQ items are present based on detected elements.

Rules-based inference to maximize recall:
- If footings detected -> excavation, PCC, RCC, formwork, steel, backfill
- If columns detected -> RCC, formwork, steel
- If concrete detected -> curing item

Each inferred item has:
- qty_status: "unknown" or "inferred"
- confidence: <= 0.55
- dependencies: what's missing to compute qty
- rule_fired: which rule triggered this item
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .estimator_output import EstimatorOutput, BOQItem
from .synonyms_india import BOQ_TEMPLATES, detect_terms_in_text
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# INFERENCE RULES
# =============================================================================

@dataclass
class InferenceRule:
    """Rule for inferring missing BOQ items."""
    rule_id: str
    description: str
    trigger_condition: str  # What must be detected
    required_items: List[str]  # BOQ template keys that must exist
    priority: int = 1  # Lower = higher priority


INFERENCE_RULES: List[InferenceRule] = [
    # Footing rules
    InferenceRule(
        rule_id="FTG_001",
        description="Footings detected -> excavation required",
        trigger_condition="footings",
        required_items=["excavation_footing"],
        priority=1
    ),
    InferenceRule(
        rule_id="FTG_002",
        description="Footings detected -> PCC blinding required",
        trigger_condition="footings",
        required_items=["pcc_footing"],
        priority=2
    ),
    InferenceRule(
        rule_id="FTG_003",
        description="Footings detected -> RCC in footings required",
        trigger_condition="footings",
        required_items=["rcc_footing"],
        priority=3
    ),
    InferenceRule(
        rule_id="FTG_004",
        description="Footings detected -> formwork for footings required",
        trigger_condition="footings",
        required_items=["formwork_footing"],
        priority=4
    ),
    InferenceRule(
        rule_id="FTG_005",
        description="Footings detected -> reinforcement for footings required",
        trigger_condition="footings",
        required_items=["steel_footing"],
        priority=5
    ),
    InferenceRule(
        rule_id="FTG_006",
        description="Footings detected -> backfilling required",
        trigger_condition="footings",
        required_items=["backfill"],
        priority=6
    ),

    # Column rules
    InferenceRule(
        rule_id="COL_001",
        description="Columns detected -> RCC in columns required",
        trigger_condition="columns",
        required_items=["rcc_column"],
        priority=1
    ),
    InferenceRule(
        rule_id="COL_002",
        description="Columns detected -> formwork for columns required",
        trigger_condition="columns",
        required_items=["formwork_column"],
        priority=2
    ),
    InferenceRule(
        rule_id="COL_003",
        description="Columns detected -> reinforcement for columns required",
        trigger_condition="columns",
        required_items=["steel_column"],
        priority=3
    ),

    # General rules
    InferenceRule(
        rule_id="GEN_001",
        description="Concrete work detected -> curing required",
        trigger_condition="concrete",
        required_items=["curing"],
        priority=10
    ),
    InferenceRule(
        rule_id="GEN_002",
        description="Excavation detected -> disposal may be required",
        trigger_condition="excavation",
        required_items=["disposal"],
        priority=11
    ),
]


# =============================================================================
# EVIDENCE TRACKING
# =============================================================================

@dataclass
class ItemEvidence:
    """Evidence for why a BOQ item exists."""
    rule_fired: str
    keywords_matched: List[str] = field(default_factory=list)
    evidence_text: List[str] = field(default_factory=list)
    bbox_locations: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_fired': self.rule_fired,
            'keywords_matched': self.keywords_matched,
            'evidence_text': self.evidence_text,
            'bbox_locations': self.bbox_locations,
            'confidence_breakdown': self.confidence_breakdown
        }


@dataclass
class EnhancedBOQItem:
    """BOQ item with full evidence tracking."""
    item: BOQItem
    evidence: ItemEvidence
    category: str = "confirmed"  # confirmed, probable, inferred

    def to_dict(self) -> Dict[str, Any]:
        result = self.item.to_dict()
        result['evidence_details'] = self.evidence.to_dict()
        result['category'] = self.category
        return result


# =============================================================================
# SCOPE COMPLETION FUNCTIONS
# =============================================================================

def check_item_exists(
    boq_items: List[BOQItem],
    template_key: str,
    element_type: Optional[str] = None
) -> bool:
    """
    Check if a BOQ item matching template already exists.

    Args:
        boq_items: List of existing BOQ items
        template_key: Key from BOQ_TEMPLATES
        element_type: Optional element type filter

    Returns:
        True if similar item exists
    """
    template = BOQ_TEMPLATES.get(template_key, {})
    if not template:
        return False

    # Keywords to match
    keywords = template_key.split('_')

    for item in boq_items:
        name_lower = item.item_name.lower()

        # Check if keywords match
        matches = sum(1 for kw in keywords if kw in name_lower)

        # Also check element type
        if element_type and item.element_type:
            if item.element_type.lower() == element_type.lower():
                matches += 1

        if matches >= 2:  # At least 2 keywords match
            return True

    return False


def get_missing_dependencies(
    output: EstimatorOutput,
    template_key: str
) -> Tuple[List[str], Optional[float]]:
    """
    Determine what's missing to compute quantity for an item.

    Args:
        output: EstimatorOutput
        template_key: BOQ template key

    Returns:
        (list of dependencies, computed qty or None)
    """
    dependencies = []
    qty = None

    if "footing" in template_key:
        # Check footing data
        if not output.footings:
            dependencies.append("no footings detected")
        else:
            depths_missing = sum(1 for f in output.footings if f.D_mm is None)
            if depths_missing > 0:
                dependencies.append(f"footing depth missing for {depths_missing} types")

            if "excavation" in template_key:
                qty = output.total_excavation_m3 if output.total_excavation_m3 > 0 else None

            elif "pcc" in template_key:
                total_pcc = sum(f.pcc_m3 for f in output.footings)
                qty = total_pcc if total_pcc > 0 else None
                if not any("PCC" in r.requirement for r in output.requirements):
                    dependencies.append("PCC thickness not specified in notes")

            elif "rcc" in template_key and "footing" in template_key:
                qty = sum(f.concrete_volume_m3 for f in output.footings)
                if qty == 0:
                    dependencies.append("footing dimensions incomplete")

            elif "formwork" in template_key and "footing" in template_key:
                qty = sum(f.formwork_sqm for f in output.footings)
                if qty == 0:
                    dependencies.append("footing dimensions incomplete")

            elif "steel" in template_key and "footing" in template_key:
                if not output.has_bar_schedule:
                    dependencies.append("requires bar bending schedule for accurate quantity")
                qty = sum(f.steel_kg for f in output.footings)

    elif "column" in template_key:
        if not output.columns:
            dependencies.append("no columns detected")
        else:
            heights_missing = sum(1 for c in output.columns if c.height_mm is None)
            if heights_missing > 0:
                dependencies.append(f"column height/level missing for {heights_missing} sizes")

            if "rcc" in template_key:
                qty = sum(c.concrete_volume_m3 for c in output.columns)
                if qty == 0:
                    dependencies.append("column dimensions incomplete")

            elif "formwork" in template_key:
                qty = sum(c.formwork_sqm for c in output.columns)
                if qty == 0:
                    dependencies.append("column dimensions incomplete")

            elif "steel" in template_key:
                if not output.has_bar_schedule:
                    dependencies.append("requires bar bending schedule for accurate quantity")
                qty = sum(c.steel_kg for c in output.columns)

    elif "backfill" in template_key:
        if output.total_excavation_m3 > 0 and output.total_concrete_m3 > 0:
            qty = output.total_excavation_m3 - output.total_concrete_m3
            if qty < 0:
                qty = None
                dependencies.append("excavation less than concrete (check dimensions)")
        else:
            dependencies.append("excavation and concrete volumes needed")

    elif "disposal" in template_key:
        dependencies.append("depends on backfill volume calculation")
        qty = None  # Always unknown initially

    elif "curing" in template_key:
        dependencies.append("requires total concrete surface area")
        qty = None

    return dependencies, qty


def create_inferred_item(
    template_key: str,
    output: EstimatorOutput,
    rule: InferenceRule,
    notes_text: str = ""
) -> Optional[EnhancedBOQItem]:
    """
    Create an inferred BOQ item from template.

    Args:
        template_key: Key from BOQ_TEMPLATES
        output: EstimatorOutput
        rule: The rule that triggered this
        notes_text: OCR notes for evidence

    Returns:
        EnhancedBOQItem or None
    """
    template = BOQ_TEMPLATES.get(template_key)
    if not template:
        return None

    # Get dependencies and quantity
    dependencies, qty = get_missing_dependencies(output, template_key)

    # Determine qty_status
    if qty is not None and qty > 0:
        qty_status = "partial" if dependencies else "computed"
        confidence = 0.70 if dependencies else 0.85
    else:
        qty_status = "unknown"
        confidence = 0.50

    # Look for evidence in notes
    evidence_text = []
    keywords_matched = []

    # Check notes for related terms
    if notes_text:
        key_terms = template_key.replace('_', ' ').split()
        for term in key_terms:
            if term.lower() in notes_text.lower():
                keywords_matched.append(term)
                # Find context
                idx = notes_text.lower().find(term.lower())
                if idx >= 0:
                    start = max(0, idx - 30)
                    end = min(len(notes_text), idx + len(term) + 30)
                    evidence_text.append(notes_text[start:end].strip())

    # Create BOQ item
    boq_item = BOQItem(
        item_name=template["item_name"],
        unit=template["unit"],
        qty=round(qty, 3) if qty else None,
        qty_status=qty_status,
        basis=template["basis"],
        dependencies=dependencies,
        confidence=confidence,
        evidence=evidence_text[:3],  # Limit evidence
        trade=template["trade"],
        element_type=template["element_type"]
    )

    # Create evidence
    evidence = ItemEvidence(
        rule_fired=f"{rule.rule_id}: {rule.description}",
        keywords_matched=keywords_matched,
        evidence_text=evidence_text[:5],
        confidence_breakdown={
            "base_confidence": 0.50,
            "qty_available": 0.20 if qty else 0.0,
            "no_dependencies": 0.15 if not dependencies else 0.0
        }
    )

    return EnhancedBOQItem(
        item=boq_item,
        evidence=evidence,
        category="inferred"
    )


def complete_scope(
    output: EstimatorOutput,
    notes_text: str = "",
    high_recall: bool = True
) -> Tuple[EstimatorOutput, List[EnhancedBOQItem], Dict[str, Any]]:
    """
    Complete the scope by adding missing-but-expected items.

    Args:
        output: EstimatorOutput to enhance
        notes_text: Combined notes text for evidence
        high_recall: Enable high recall mode

    Returns:
        (enhanced output, list of all enhanced items, coverage report)
    """
    logger.info("Running scope completion engine...")

    # Categorize existing items
    confirmed_items: List[EnhancedBOQItem] = []
    probable_items: List[EnhancedBOQItem] = []
    inferred_items: List[EnhancedBOQItem] = []

    # Convert existing BOQ items to enhanced items
    for item in output.boq_items:
        # Determine category based on confidence and status
        if item.confidence >= 0.85 and item.qty_status == "computed":
            category = "confirmed"
        elif item.confidence >= 0.60:
            category = "probable"
        else:
            category = "inferred"

        evidence = ItemEvidence(
            rule_fired="EXPLICIT: Evidence-based extraction",
            keywords_matched=[],
            evidence_text=item.evidence if isinstance(item.evidence, list) else [],
            confidence_breakdown={"extraction_confidence": item.confidence}
        )

        enhanced = EnhancedBOQItem(item=item, evidence=evidence, category=category)

        if category == "confirmed":
            confirmed_items.append(enhanced)
        elif category == "probable":
            probable_items.append(enhanced)
        else:
            inferred_items.append(enhanced)

    # Track what triggers are active
    active_triggers = set()

    if output.footings:
        active_triggers.add("footings")
    if output.columns:
        active_triggers.add("columns")
    if output.total_concrete_m3 > 0:
        active_triggers.add("concrete")
    if output.total_excavation_m3 > 0:
        active_triggers.add("excavation")

    # Apply inference rules
    if high_recall:
        for rule in sorted(INFERENCE_RULES, key=lambda r: r.priority):
            if rule.trigger_condition not in active_triggers:
                continue

            for template_key in rule.required_items:
                # Check if item already exists
                all_items = [e.item for e in confirmed_items + probable_items + inferred_items]
                if check_item_exists(all_items, template_key):
                    continue

                # Create inferred item
                enhanced = create_inferred_item(template_key, output, rule, notes_text)
                if enhanced:
                    inferred_items.append(enhanced)
                    logger.debug(f"Added inferred item: {template_key} via {rule.rule_id}")

    # Check for probable items from notes text (using synonyms)
    if notes_text and high_recall:
        detected_terms = detect_terms_in_text(notes_text)

        for match in detected_terms:
            # Map to template keys
            term_to_template = {
                "excavation": "excavation_footing",
                "pcc": "pcc_footing",
                "backfill": "backfill",
                "anti_termite": "anti_termite",
                "waterproofing": "waterproofing_dpc",
                "curing": "curing",
            }

            template_key = term_to_template.get(match.standard_term)
            if not template_key:
                continue

            all_items = [e.item for e in confirmed_items + probable_items + inferred_items]
            if check_item_exists(all_items, template_key):
                continue

            # Create probable item
            template = BOQ_TEMPLATES.get(template_key, {})
            if template:
                deps, qty = get_missing_dependencies(output, template_key)

                boq_item = BOQItem(
                    item_name=template["item_name"],
                    unit=template["unit"],
                    qty=round(qty, 3) if qty else None,
                    qty_status="partial" if qty else "unknown",
                    basis=template["basis"],
                    dependencies=deps,
                    confidence=match.confidence * 0.9,  # Reduce slightly
                    evidence=[match.matched_text],
                    trade=template["trade"],
                    element_type=template.get("element_type", "General")
                )

                evidence = ItemEvidence(
                    rule_fired=f"SYNONYM: Matched '{match.matched_text}' -> {match.standard_term}",
                    keywords_matched=[match.matched_text],
                    evidence_text=[match.matched_text],
                    confidence_breakdown={"synonym_confidence": match.confidence}
                )

                probable_items.append(EnhancedBOQItem(
                    item=boq_item,
                    evidence=evidence,
                    category="probable"
                ))

    # Build coverage report
    all_enhanced = confirmed_items + probable_items + inferred_items
    missing_deps = []
    for e in all_enhanced:
        missing_deps.extend(e.item.dependencies)

    # Count unique dependencies
    dep_counts = {}
    for dep in missing_deps:
        dep_counts[dep] = dep_counts.get(dep, 0) + 1

    coverage_report = {
        "detected_count": len(confirmed_items),
        "probable_count": len(probable_items),
        "inferred_count": len(inferred_items),
        "total_items": len(all_enhanced),
        "computed_qty_count": len([e for e in all_enhanced if e.item.qty_status == "computed"]),
        "partial_qty_count": len([e for e in all_enhanced if e.item.qty_status == "partial"]),
        "unknown_qty_count": len([e for e in all_enhanced if e.item.qty_status == "unknown"]),
        "top_missing_dependencies": sorted(dep_counts.items(), key=lambda x: -x[1])[:10],
        "active_triggers": list(active_triggers),
        "rules_fired": len([e for e in inferred_items if "RULE" in e.evidence.rule_fired or e.evidence.rule_fired.startswith("FTG") or e.evidence.rule_fired.startswith("COL") or e.evidence.rule_fired.startswith("GEN")])
    }

    # Update output with all items
    output.boq_items = [e.item for e in all_enhanced]

    logger.info(f"Scope completion: {coverage_report['detected_count']} confirmed, "
               f"{coverage_report['probable_count']} probable, "
               f"{coverage_report['inferred_count']} inferred")

    return output, all_enhanced, coverage_report


def generate_missing_items_report(
    output: EstimatorOutput,
    enhanced_items: List[EnhancedBOQItem],
    coverage_report: Dict[str, Any]
) -> str:
    """
    Generate a markdown report of missing items and reasons.

    Args:
        output: EstimatorOutput
        enhanced_items: List of enhanced BOQ items
        coverage_report: Coverage statistics

    Returns:
        Markdown string
    """
    lines = [
        "# Missing Items Report",
        "",
        f"Generated for: {output.project.sheet_name or 'Unknown'}",
        f"Confidence: {output.confidence:.0%}",
        "",
        "## Coverage Summary",
        "",
        f"- Confirmed items: {coverage_report['detected_count']}",
        f"- Probable items: {coverage_report['probable_count']}",
        f"- Inferred items: {coverage_report['inferred_count']}",
        f"- Items with computed qty: {coverage_report['computed_qty_count']}",
        f"- Items with unknown qty: {coverage_report['unknown_qty_count']}",
        "",
        "## Top Missing Dependencies",
        ""
    ]

    for dep, count in coverage_report['top_missing_dependencies']:
        lines.append(f"- **{dep}** (affects {count} items)")

    lines.extend([
        "",
        "## Items Needing Manual Input",
        "",
        "| Item | Unit | Status | Dependencies |",
        "|------|------|--------|--------------|"
    ])

    for e in enhanced_items:
        if e.item.qty_status in ("unknown", "partial"):
            deps = "; ".join(e.item.dependencies[:2]) if e.item.dependencies else "-"
            name = e.item.item_name[:50] + "..." if len(e.item.item_name) > 50 else e.item.item_name
            lines.append(f"| {name} | {e.item.unit} | {e.item.qty_status} | {deps} |")

    lines.extend([
        "",
        "## Expected Scope Items Not Found",
        ""
    ])

    # List standard items that should exist but don't
    expected_if_footings = ["excavation_footing", "pcc_footing", "rcc_footing", "formwork_footing", "steel_footing"]
    expected_if_columns = ["rcc_column", "formwork_column", "steel_column"]

    if output.footings:
        for template_key in expected_if_footings:
            exists = check_item_exists(output.boq_items, template_key)
            if not exists:
                lines.append(f"- **{template_key}**: Expected but not found")

    if output.columns:
        for template_key in expected_if_columns:
            exists = check_item_exists(output.boq_items, template_key)
            if not exists:
                lines.append(f"- **{template_key}**: Expected but not found")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "1. Check drawing notes for PCC/blinding thickness specification",
        "2. Verify column heights are specified in section/elevation",
        "3. Look for bar bending schedule (BBS) for accurate steel quantities",
        "4. Check footing details for depth dimensions",
        ""
    ])

    return "\n".join(lines)
