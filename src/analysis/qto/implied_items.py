"""
Implied Items Rule Engine — BOQ items that every estimator adds from experience.

These are items NOT explicitly stated in the drawings or specifications, but
logically required given what has been extracted. Examples:
  - PCC M10 lean concrete bed is always required below any footing
  - Anti-termite treatment is always needed at sub-grade level
  - Earthwork excavation and backfilling follow from the footing schedule
  - Roof waterproofing is needed wherever a top slab exists
  - Wet area waterproofing follows from toilet/bathroom room detection

Design:
  - Each rule is a plain function: RuleContext → List[dict] (BOQ line items)
  - Items carry rule_name, confidence 0.40–0.65, source="implied_rule"
  - Rules are fully independent; order matters only where items chain (e.g.
    backfilling references the qty computed by earthwork_excavation).
  - All quantities are labelled with the assumption they rest on so the
    estimator can audit / override.

Usage:
    ctx = RuleContext(
        structural_elements=[...],
        structural_items=[...],   # already-generated BOQ items
        rooms=[...],
        total_area_sqm=450.0,
        floors=4,
        building_type="residential",
        storey_height_mm=3000,
    )
    items, triggered = run_implied_rules(ctx)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# ---------------------------------------------------------------------------
# We import these lightly; modules must already be on path.
# ---------------------------------------------------------------------------
try:
    from .finish_takeoff import RoomData  # type: ignore
    from .structural_takeoff import ScheduledElement  # type: ignore
except ImportError:
    RoomData = object  # type: ignore
    ScheduledElement = object  # type: ignore


# =============================================================================
# RULE CONTEXT — packages all extracted data rules can read
# =============================================================================

@dataclass
class RuleContext:
    """All QTO data available to the implied-item rules."""

    # From structural_takeoff
    structural_elements: List  # List[ScheduledElement]
    structural_items: List[dict]   # already-generated structural BOQ items

    # From finish_takeoff
    rooms: List                    # List[RoomData]

    # Building parameters
    total_area_sqm: float          # built-up area per floor
    floors: int                    # number of floors (G = 1, G+4 = 5)
    building_type: str = "residential"  # residential | commercial | industrial
    storey_height_mm: int = 3000   # mm

    # Optional drawing callout data
    door_count: int = 0            # door tags found in drawings
    window_count: int = 0          # window tags found in drawings
    drawing_callouts: List[dict] = field(default_factory=list)


# =============================================================================
# HELPERS
# =============================================================================

def _footings(ctx: RuleContext) -> List:
    """Return footing elements from structural_elements."""
    return [e for e in ctx.structural_elements if getattr(e, 'element_type', '') == 'footing']


def _columns(ctx: RuleContext) -> List:
    return [e for e in ctx.structural_elements if getattr(e, 'element_type', '') == 'column']


def _slabs(ctx: RuleContext) -> List:
    return [e for e in ctx.structural_elements if getattr(e, 'element_type', '') == 'slab']


def _wet_rooms(ctx: RuleContext) -> List:
    """Return rooms classified as wet areas (toilets, bathrooms, kitchen)."""
    WET_KEYWORDS = {"TOILET", "BATHROOM", "BATH", "WC", "WASH", "KITCHEN", "KIT"}
    result = []
    for room in ctx.rooms:
        name = getattr(room, 'name', '') or ''
        if any(k in name.upper() for k in WET_KEYWORDS):
            result.append(room)
    return result


def _total_steel_kg(ctx: RuleContext) -> float:
    """Sum all steel (kg) items already generated."""
    return sum(
        item.get('qty', 0.0)
        for item in ctx.structural_items
        if item.get('unit', '') == 'kg'
    )


def _room_area(room) -> float:
    return float(getattr(room, 'area_sqm', 0) or 0)


def _room_perimeter_estimate(room) -> float:
    """Estimate room perimeter from floor area (assumes square-ish room)."""
    area = _room_area(room)
    if area <= 0:
        return 0.0
    return 4.0 * math.sqrt(area)


def _item(
    description: str,
    unit: str,
    qty: float,
    trade: str,
    section: str,
    rule_name: str,
    source_page: int = 0,
    confidence: float = 0.50,
    source: str = "implied_rule",
) -> dict:
    return {
        "item_no":          None,
        "description":      description,
        "unit":             unit,
        "unit_inferred":    False,
        "qty":              round(qty, 3),
        "rate":             None,
        "trade":            trade,
        "section":          section,
        "source_page":      source_page,
        "source":           source,
        "confidence":       round(confidence, 2),
        "is_priceable":     True,
        "priceable_reason": "priceable",
        "qto_method":       f"implied:{rule_name}",
        "rule_name":        rule_name,
    }


# =============================================================================
# RULE IMPLEMENTATIONS
# =============================================================================

# ── Rule 1: Earthwork excavation ────────────────────────────────────────────

def rule_earthwork_excavation(ctx: RuleContext) -> Tuple[List[dict], float]:
    """
    Generates earthwork excavation volume based on footing footprints.

    qty = sum(footing_L × footing_W × (depth + 300mm working) × count) × 1.25 bulking

    Returns: (items, excavation_vol_m3) so backfill rule can chain from it.
    """
    ftgs = _footings(ctx)
    if not ftgs:
        return [], 0.0

    excavation_vol = 0.0
    for ftg in ftgs:
        L = (getattr(ftg, 'width_mm', None) or 1500) / 1000
        W = (getattr(ftg, 'depth_mm', None) or 1500) / 1000
        D = (getattr(ftg, 'length_mm', None) or 450) / 1000
        working_depth_m = D + 0.30   # 300mm extra below footing for working space
        count = getattr(ftg, 'count', 1) or 1
        excavation_vol += L * W * working_depth_m * count

    # Add 25% bulkage factor (loose excavated earth occupies more volume)
    excavation_vol *= 1.25
    excavation_vol = round(excavation_vol, 3)

    if excavation_vol <= 0:
        return [], 0.0

    items = [_item(
        description=(
            "Earthwork in excavation by mechanical means for isolated column "
            "footings including disposal of excavated earth within 50m lead and "
            "1.5m lift, dressing of sides and bottom"
        ),
        unit="cum",
        qty=excavation_vol,
        trade="civil",
        section="SUBSTRUCTURE — EARTHWORK",
        rule_name="earthwork_excavation",
        confidence=0.55,
    )]
    return items, excavation_vol


# ── Rule 2: Backfilling ──────────────────────────────────────────────────────

def rule_backfilling(ctx: RuleContext, excavation_vol_m3: float) -> List[dict]:
    """
    Earth backfilling in trenches = excavation vol − concrete footing vol.
    Approximately 55% of excavation is returned as backfill.
    """
    if excavation_vol_m3 <= 0:
        return []

    # Footing concrete volume (already computed by structural_takeoff)
    footing_concrete_vol = sum(
        item.get('qty', 0.0)
        for item in ctx.structural_items
        if item.get('unit', '') == 'cum'
        and 'footing' in item.get('section', '').lower()
    )

    if footing_concrete_vol > 0:
        backfill_vol = excavation_vol_m3 - footing_concrete_vol
        if backfill_vol <= 0:
            backfill_vol = excavation_vol_m3 * 0.55
    else:
        # Footing concrete not tracked — use assumption: 55% of excavation returned
        backfill_vol = excavation_vol_m3 * 0.55

    backfill_vol = round(backfill_vol, 3)

    return [_item(
        description=(
            "Earth filling in foundation trenches/plinth with available excavated "
            "earth in 150mm layers, watering and compacting with mechanical rammer "
            "to achieve 95% proctor density"
        ),
        unit="cum",
        qty=backfill_vol,
        trade="civil",
        section="SUBSTRUCTURE — EARTHWORK",
        rule_name="backfilling",
        confidence=0.50,
    )]


# ── Rule 3: PCC lean concrete bed ───────────────────────────────────────────

def rule_pcc_bed_below_footings(ctx: RuleContext) -> List[dict]:
    """
    PCC M10 / 1:3:6 lean concrete 75mm thick below every footing.
    qty = sum(footing_L × footing_W × count × 0.075)
    """
    ftgs = _footings(ctx)
    if not ftgs:
        return []

    pcc_vol = 0.0
    for ftg in ftgs:
        L = (getattr(ftg, 'width_mm', None) or 1500) / 1000
        W = (getattr(ftg, 'depth_mm', None) or 1500) / 1000
        count = getattr(ftg, 'count', 1) or 1
        pcc_vol += L * W * 0.075 * count   # 75mm thick

    pcc_vol = round(pcc_vol, 3)
    if pcc_vol <= 0:
        return []

    return [_item(
        description=(
            "Providing and laying cement concrete 1:3:6 (M10) lean concrete bed "
            "75mm thick below footings, well watered and tamped"
        ),
        unit="cum",
        qty=pcc_vol,
        trade="civil",
        section="SUBSTRUCTURE — PCC BED",
        rule_name="pcc_bed_below_footings",
        confidence=0.65,
    )]


# ── Rule 4: Anti-termite treatment ──────────────────────────────────────────

def rule_anti_termite_treatment(ctx: RuleContext) -> List[dict]:
    """
    Anti-termite soil treatment required at sub-grade level for any building.
    qty = building footprint area (sqm) = total_area_sqm (ground floor only).
    """
    area = ctx.total_area_sqm
    if area <= 0:
        return []

    return [_item(
        description=(
            "Anti-termite treatment at pre-construction stage — soil poisoning to "
            "sub-grade surfaces under floors, sides of all wall/column pits with "
            "approved chlorpyrifos 1% emulsion as per IS 6313 Part 2"
        ),
        unit="sqm",
        qty=round(area, 2),
        trade="civil",
        section="SUBSTRUCTURE — ANTI-TERMITE",
        rule_name="anti_termite_treatment",
        confidence=0.70,
    )]


# ── Rule 5: DPC at plinth level ─────────────────────────────────────────────

def rule_dpc_at_plinth(ctx: RuleContext) -> List[dict]:
    """
    Damp proof course 50mm thick at plinth level across entire building footprint.
    Required for any building with a ground floor.
    """
    area = ctx.total_area_sqm
    if area <= 0:
        return []

    return [_item(
        description=(
            "Providing and laying damp proof course 50mm thick with M20 grade "
            "waterproof concrete (2% waterproofing compound by weight of cement) "
            "at plinth level"
        ),
        unit="sqm",
        qty=round(area, 2),
        trade="civil",
        section="SUBSTRUCTURE — DPC",
        rule_name="dpc_at_plinth",
        confidence=0.65,
    )]


# ── Rule 6: Brick masonry walls ─────────────────────────────────────────────

def rule_brick_masonry_walls(ctx: RuleContext) -> List[dict]:
    """
    Brick masonry volume estimated from room perimeters.

    Strategy:
    - Sum room perimeters to get total wall centerline length
    - Divide by 2 to approximate shared-wall de-duplication
    - Multiply by storey height and wall thickness (230mm external, 115mm internal)
    - Deduct ~15% for door/window openings

    Returns TWO items: external 230mm and internal 115mm walls.
    """
    if not ctx.rooms:
        return []

    total_perimeter = sum(_room_perimeter_estimate(r) for r in ctx.rooms)
    if total_perimeter <= 0:
        return []

    h_m = ctx.storey_height_mm / 1000
    opening_deduct = 0.85   # 15% for openings

    # Rough split: ~30% external 230mm walls, ~70% internal 115mm walls
    external_length = total_perimeter * 0.30 / 2   # /2 for shared wall correction
    internal_length = total_perimeter * 0.70 / 2

    external_vol = round(external_length * h_m * 0.230 * ctx.floors * opening_deduct, 3)
    internal_vol = round(internal_length * h_m * 0.115 * ctx.floors * opening_deduct, 3)

    items = []
    if external_vol > 0:
        items.append(_item(
            description=(
                "Brick masonry with class A burnt clay bricks (7.5 N/mm²) in "
                "cement mortar 1:6 in superstructure above plinth level, "
                "230mm (1 brick) thick external walls"
            ),
            unit="cum",
            qty=external_vol,
            trade="civil",
            section="SUPERSTRUCTURE — MASONRY",
            rule_name="brick_masonry_external",
            confidence=0.45,
        ))
    if internal_vol > 0:
        items.append(_item(
            description=(
                "Brick masonry with class A burnt clay bricks (7.5 N/mm²) in "
                "cement mortar 1:6 in superstructure, 115mm (half brick) thick "
                "internal partition walls"
            ),
            unit="cum",
            qty=internal_vol,
            trade="civil",
            section="SUPERSTRUCTURE — MASONRY",
            rule_name="brick_masonry_internal",
            confidence=0.45,
        ))
    return items


# ── Rule 7: Internal plaster ────────────────────────────────────────────────

def rule_internal_plastering(ctx: RuleContext) -> List[dict]:
    """
    Internal cement plaster 12mm 1:6 on brick/RCC surfaces.
    Derived from room wall areas (same calculation as finish_takeoff wall area).
    """
    if not ctx.rooms:
        return []

    total_wall_sqm = 0.0
    h_m = ctx.storey_height_mm / 1000
    for room in ctx.rooms:
        perimeter = _room_perimeter_estimate(room)
        wall_sqm = perimeter * h_m * 0.85   # 15% deduct for openings
        total_wall_sqm += wall_sqm

    total_wall_sqm = round(total_wall_sqm, 2)
    if total_wall_sqm <= 0:
        return []

    return [_item(
        description=(
            "Providing and applying cement plaster 12mm thick in two coats in "
            "proportion 1:6 (1 cement:6 coarse sand) on internal brick/RCC "
            "surfaces including finishing smooth with cement slurry"
        ),
        unit="sqm",
        qty=total_wall_sqm,
        trade="civil",
        section="SUPERSTRUCTURE — PLASTERING",
        rule_name="internal_plastering",
        confidence=0.45,
    )]


# ── Rule 8: External plaster ────────────────────────────────────────────────

def rule_external_plastering(ctx: RuleContext) -> List[dict]:
    """
    External cement plaster 20mm 1:4 on external faces.
    Estimated from building perimeter × total height.
    """
    area = ctx.total_area_sqm
    if area <= 0:
        return []

    # Approximate building perimeter from footprint area (assume square-ish)
    building_perimeter = 4.0 * math.sqrt(area)
    total_height = (ctx.storey_height_mm / 1000) * ctx.floors
    external_sqm = round(building_perimeter * total_height * 0.85, 2)

    if external_sqm <= 0:
        return []

    return [_item(
        description=(
            "Providing and applying cement plaster 20mm thick in two coats in "
            "proportion 1:4 (1 cement:4 coarse sand) on external surfaces "
            "including keyways, curing and finishing to true and even surface"
        ),
        unit="sqm",
        qty=external_sqm,
        trade="civil",
        section="SUPERSTRUCTURE — PLASTERING",
        rule_name="external_plastering",
        confidence=0.40,
    )]


# ── Rule 9: Wet area waterproofing ──────────────────────────────────────────

def rule_wet_area_waterproofing(ctx: RuleContext) -> List[dict]:
    """
    Waterproofing for toilets, bathrooms, kitchens.
    qty = floor area + perimeter × 0.3m (300mm upstand on walls).
    """
    wet_rooms = _wet_rooms(ctx)
    if not wet_rooms:
        return []

    total_wp_sqm = 0.0
    for room in wet_rooms:
        floor_area = _room_area(room)
        if floor_area <= 0:
            continue
        perimeter = _room_perimeter_estimate(room)
        upstand_sqm = perimeter * 0.30   # 300mm up the walls
        total_wp_sqm += floor_area + upstand_sqm

    total_wp_sqm = round(total_wp_sqm, 2)
    if total_wp_sqm <= 0:
        return []

    return [_item(
        description=(
            "Waterproofing treatment to toilet/kitchen floors and walls — 2 coats "
            "of cementitious polymer-modified waterproofing slurry (acrylic based) "
            "applied on floor and 300mm high upstand on walls including protection "
            "screed as per manufacturer specification"
        ),
        unit="sqm",
        qty=total_wp_sqm,
        trade="waterproofing",
        section="FINISHES — WATERPROOFING",
        rule_name="wet_area_waterproofing",
        confidence=0.60,
    )]


# ── Rule 10: Roof waterproofing ─────────────────────────────────────────────

def rule_roof_waterproofing(ctx: RuleContext) -> List[dict]:
    """
    APP/SBS bituminous membrane waterproofing to roof terrace slab.
    Triggered by: slab elements detected OR total_area_sqm > 0 for top floor.
    qty = total_area_sqm × 1.10 (10% extra for parapet and overlaps).
    """
    slab_elements = _slabs(ctx)
    area = ctx.total_area_sqm
    if area <= 0 and not slab_elements:
        return []

    wp_area = round(max(area, 0) * 1.10, 2)
    if wp_area <= 0:
        return []

    return [_item(
        description=(
            "Providing and fixing waterproofing treatment to roof terrace slab — "
            "2 coats of APP modified bituminous membrane 4mm thick including "
            "primer coat, 50mm mud phuska over membrane, and 25mm protective "
            "screed with slope to drains"
        ),
        unit="sqm",
        qty=wp_area,
        trade="waterproofing",
        section="FINISHES — WATERPROOFING",
        rule_name="roof_waterproofing",
        confidence=0.55,
    )]


# ── Rule 11: Binding wire ────────────────────────────────────────────────────

def rule_binding_wire(ctx: RuleContext) -> List[dict]:
    """
    Annealed binding wire @ 1% of total steel weight.
    Only generated if structural steel items exist.
    """
    steel_kg = _total_steel_kg(ctx)
    if steel_kg <= 0:
        return []

    wire_kg = round(steel_kg * 0.01, 2)
    if wire_kg <= 0:
        return []

    return [_item(
        description=(
            "Providing and fixing 16 gauge (1.6mm dia) annealed mild steel binding "
            "wire for tying and binding reinforcement bars at intersections "
            "including all wastage"
        ),
        unit="kg",
        qty=wire_kg,
        trade="structural",
        section="STRUCTURAL — REINFORCEMENT",
        rule_name="binding_wire",
        confidence=0.75,
    )]


# ── Rule 12: Lintels over openings ──────────────────────────────────────────

def rule_lintels_over_openings(ctx: RuleContext) -> List[dict]:
    """
    RCC lintels above door and window openings.
    Triggered by: door_count or window_count > 0 in RuleContext.

    Assumptions:
    - Door lintel: 230mm wide × 150mm deep × (door_width + 600mm bearing)
      Typical door clear width 1.0m → total lintel span 1.6m
    - Window lintel: 230mm wide × 150mm deep × (window_width + 600mm bearing)
      Typical window clear width 1.2m → total lintel span 1.8m
    """
    d_count = ctx.door_count or 0
    w_count = ctx.window_count or 0

    if d_count == 0 and w_count == 0:
        # If no tag counts available, estimate from room count: assume 1 door + 1 window per room
        if ctx.rooms:
            d_count = len(ctx.rooms)
            w_count = len(ctx.rooms)
        else:
            return []

    door_lintel_vol = d_count * (0.230 * 0.150 * 1.60)   # 230×150mm × 1.6m span
    window_lintel_vol = w_count * (0.230 * 0.150 * 1.80) # 230×150mm × 1.8m span
    total_lintel_vol = round((door_lintel_vol + window_lintel_vol) * ctx.floors, 3)
    lintel_steel_kg = round(total_lintel_vol * 120.0, 1)   # 120 kg/m³ for lintels

    items = []
    if total_lintel_vol > 0:
        items.append(_item(
            description=(
                "Providing RCC M20 grade precast/cast-in-situ lintels "
                "230mm wide × 150mm deep over door and window openings "
                "including centering, shuttering, Fe500 reinforcement and curing"
            ),
            unit="cum",
            qty=total_lintel_vol,
            trade="structural",
            section="SUPERSTRUCTURE — LINTELS",
            rule_name="lintels_concrete",
            confidence=0.45,
        ))
        items.append(_item(
            description=(
                "Reinforcement Fe500 deformed bars for lintels including "
                "cutting, bending and binding"
            ),
            unit="kg",
            qty=lintel_steel_kg,
            trade="structural",
            section="SUPERSTRUCTURE — LINTELS",
            rule_name="lintels_steel",
            confidence=0.42,
        ))
    return items


# ── Rule 13: Sunken slab waterproofing (for toilets) ────────────────────────

def rule_sunken_slab_filling(ctx: RuleContext) -> List[dict]:
    """
    Lightweight concrete filling in sunken slab area of toilets.
    Indian practice: toilet slabs are sunken 300–450mm for drainage.
    qty = wet room floor area × 0.35m average fill depth.
    """
    wet_rooms = _wet_rooms(ctx)
    if not wet_rooms:
        return []

    total_sunken_vol = sum(
        _room_area(r) * 0.350   # 350mm average fill (CLC/brick bat/lightweight)
        for r in wet_rooms
        if _room_area(r) > 0
    )
    total_sunken_vol = round(total_sunken_vol, 3)
    if total_sunken_vol <= 0:
        return []

    return [_item(
        description=(
            "Providing and laying light-weight CLC (cellular lightweight concrete) "
            "filling in sunken slab of toilets/bathrooms to bring to floor level "
            "over waterproofing membrane, in average 300–450mm depth"
        ),
        unit="cum",
        qty=total_sunken_vol,
        trade="civil",
        section="SUPERSTRUCTURE — SUNKEN FILLING",
        rule_name="sunken_slab_filling",
        confidence=0.50,
    )]


# =============================================================================
# RULE REGISTRY
# =============================================================================

# Ordered list — earthwork must run before backfilling (chaining)
_RULE_NAMES_ORDERED = [
    "earthwork_excavation",
    "backfilling",
    "pcc_bed_below_footings",
    "anti_termite_treatment",
    "dpc_at_plinth",
    "brick_masonry_walls",
    "internal_plastering",
    "external_plastering",
    "wet_area_waterproofing",
    "sunken_slab_filling",
    "roof_waterproofing",
    "binding_wire",
    "lintels_over_openings",
]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_implied_rules(
    ctx: RuleContext,
    enabled_rules: Optional[List[str]] = None,
) -> Tuple[List[dict], List[str]]:
    """
    Run all enabled implied-item rules against the given context.

    Args:
        ctx:            full rule context (structural elements, rooms, etc.)
        enabled_rules:  if provided, only run these rule names; else run all

    Returns:
        (items, triggered_rules) where triggered_rules lists rule names that
        produced at least one item.
    """
    rules_to_run = enabled_rules if enabled_rules is not None else _RULE_NAMES_ORDERED

    all_items: List[dict] = []
    triggered: List[str] = []

    # earthwork runs first; its volume is passed to backfilling
    excavation_vol_m3 = 0.0

    for rule_name in rules_to_run:
        try:
            new_items: List[dict] = []

            if rule_name == "earthwork_excavation":
                new_items, excavation_vol_m3 = rule_earthwork_excavation(ctx)

            elif rule_name == "backfilling":
                new_items = rule_backfilling(ctx, excavation_vol_m3)

            elif rule_name == "pcc_bed_below_footings":
                new_items = rule_pcc_bed_below_footings(ctx)

            elif rule_name == "anti_termite_treatment":
                new_items = rule_anti_termite_treatment(ctx)

            elif rule_name == "dpc_at_plinth":
                new_items = rule_dpc_at_plinth(ctx)

            elif rule_name == "brick_masonry_walls":
                new_items = rule_brick_masonry_walls(ctx)

            elif rule_name == "internal_plastering":
                new_items = rule_internal_plastering(ctx)

            elif rule_name == "external_plastering":
                new_items = rule_external_plastering(ctx)

            elif rule_name == "wet_area_waterproofing":
                new_items = rule_wet_area_waterproofing(ctx)

            elif rule_name == "sunken_slab_filling":
                new_items = rule_sunken_slab_filling(ctx)

            elif rule_name == "roof_waterproofing":
                new_items = rule_roof_waterproofing(ctx)

            elif rule_name == "binding_wire":
                new_items = rule_binding_wire(ctx)

            elif rule_name == "lintels_over_openings":
                new_items = rule_lintels_over_openings(ctx)

            if new_items:
                all_items.extend(new_items)
                triggered.append(rule_name)

        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Implied rule '%s' raised: %s", rule_name, exc
            )

    return all_items, triggered


# =============================================================================
# CONVENIENCE FACTORY — build RuleContext from pipeline state
# =============================================================================

def build_rule_context(
    structural_elements: list,
    structural_items: list,
    rooms: list,
    total_area_sqm: float,
    floors: int,
    building_type: str = "residential",
    storey_height_mm: int = 3000,
    drawing_callouts: Optional[List[dict]] = None,
) -> RuleContext:
    """
    Build a RuleContext from the data available inside pipeline.py.
    Also infers door/window counts from drawing callouts if available.
    """
    callouts = drawing_callouts or []
    door_count = sum(
        1 for c in callouts
        if c.get('callout_type') == 'tag' and c.get('text', '').startswith('D')
    )
    window_count = sum(
        1 for c in callouts
        if c.get('callout_type') == 'tag' and c.get('text', '').startswith('W')
    )

    return RuleContext(
        structural_elements=structural_elements,
        structural_items=structural_items,
        rooms=rooms,
        total_area_sqm=total_area_sqm,
        floors=floors,
        building_type=building_type,
        storey_height_mm=storey_height_mm,
        door_count=door_count,
        window_count=window_count,
        drawing_callouts=callouts,
    )


# =============================================================================
# STANDARD RESULT WRAPPER (mirrors other QTO module patterns)
# =============================================================================

@dataclass
class ImpliedItemsResult:
    """Standard result object for implied items (mirrors other QTO Result types)."""
    line_items: List[dict] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    mode: str = "implied_rules"
    warnings: List[str] = field(default_factory=list)

    # Convenience accessor used by pipeline destructuring shim
    def as_tuple(self) -> Tuple[List[dict], List[str]]:
        return self.line_items, self.triggered_rules


def run_implied_rules_result(
    ctx: RuleContext,
    enabled_rules: Optional[List[str]] = None,
) -> ImpliedItemsResult:
    """Same as run_implied_rules() but returns an ImpliedItemsResult (standard QTO pattern)."""
    items, triggered = run_implied_rules(ctx, enabled_rules)
    return ImpliedItemsResult(line_items=items, triggered_rules=triggered)
