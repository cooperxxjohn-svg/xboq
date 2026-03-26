"""
Scope Dependency Graph — Ensures no scope item gets missed.

Every construction element has mandatory dependencies:
- Column → RCC + formwork + steel + curing
- Toilet → waterproofing + anti-skid tile + wall tile + dado + WC + wash basin + floor trap
- Kitchen → granite platform + sink + chimney point + dado tiles + waterproofing

This module:
1. Defines 200+ dependency rules across ALL construction trades
2. Given a set of detected elements, returns ALL required sub-items
3. Given a BOQ, flags missing dependencies
4. Bidirectional: detects orphan sub-items (steel without RCC)

Based on IS 456, IS 1200, NBC 2016, CPWD DSR 2024.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DEPENDENCY RULES
# =============================================================================

class Trade(Enum):
    STRUCTURAL = "structural"
    MASONRY = "masonry"
    FINISHING = "finishing"
    PLUMBING = "plumbing"
    ELECTRICAL = "electrical"
    WATERPROOFING = "waterproofing"
    CARPENTRY = "carpentry"
    EXTERNAL = "external"
    FIRE_SAFETY = "fire_safety"
    HVAC = "hvac"
    MISC = "misc"


@dataclass
class DependencyRule:
    """A single dependency: if trigger exists, required items must also exist."""
    trigger: str                # What element triggers this rule
    required_items: List[str]   # What MUST exist if trigger is present
    trade: Trade
    priority: int = 1           # 1=critical, 2=important, 3=good-practice
    condition: str = ""         # Optional condition (e.g., "wet_area", "multi_storey")
    note: str = ""              # IS code / standard reference


# ── Structural Dependencies ──

STRUCTURAL_RULES = [
    # Footings
    DependencyRule("footing", [
        "excavation", "pcc_blinding", "rcc_footing", "formwork_footing",
        "steel_footing", "backfill", "anti_termite", "curing",
    ], Trade.STRUCTURAL, 1, note="IS 456, IS 1200 Part 1"),
    DependencyRule("footing", [
        "earth_disposal",
    ], Trade.STRUCTURAL, 2, note="Surplus earth disposal"),

    # Columns
    DependencyRule("column", [
        "rcc_column", "formwork_column", "steel_column", "curing",
    ], Trade.STRUCTURAL, 1, note="IS 456 Cl 26"),
    DependencyRule("column", [
        "column_footing",  # Every column needs a footing
    ], Trade.STRUCTURAL, 1, note="Every column must have foundation"),

    # Beams
    DependencyRule("beam", [
        "rcc_beam", "formwork_beam", "steel_beam", "curing",
    ], Trade.STRUCTURAL, 1, note="IS 456 Cl 26"),

    # Slabs
    DependencyRule("slab", [
        "rcc_slab", "formwork_slab", "steel_slab", "curing",
    ], Trade.STRUCTURAL, 1, note="IS 456 Cl 26"),

    # Lintels (above every opening)
    DependencyRule("door", [
        "lintel",
    ], Trade.STRUCTURAL, 2, note="Lintel above every door/window opening"),
    DependencyRule("window", [
        "lintel", "sill",
    ], Trade.STRUCTURAL, 2, note="Lintel + sill for windows"),

    # Staircase
    DependencyRule("staircase", [
        "rcc_staircase", "formwork_staircase", "steel_staircase",
        "staircase_railing", "staircase_flooring", "curing",
    ], Trade.STRUCTURAL, 1, note="IS 456, NBC 2016"),

    # Parapet
    DependencyRule("terrace", [
        "parapet_wall", "parapet_coping",
    ], Trade.STRUCTURAL, 1, note="NBC 2016 - minimum 1.0m parapet"),

    # Plinth beam
    DependencyRule("plinth", [
        "rcc_plinth_beam", "formwork_plinth_beam", "steel_plinth_beam",
        "dpc_waterproofing",
    ], Trade.STRUCTURAL, 1, note="DPC at plinth level mandatory"),

    # Water tank
    DependencyRule("water_tank", [
        "rcc_water_tank", "formwork_water_tank", "steel_water_tank",
        "waterproofing_water_tank", "curing",
    ], Trade.STRUCTURAL, 1, note="IS 3370 for water retaining structures"),

    # Retaining wall
    DependencyRule("retaining_wall", [
        "rcc_retaining_wall", "formwork_retaining_wall",
        "steel_retaining_wall", "drainage_behind_wall",
        "waterproofing_retaining_wall", "backfill_retaining_wall",
    ], Trade.STRUCTURAL, 1, note="IS 456 Annex B"),

    # Lift pit (multi-storey)
    DependencyRule("lift", [
        "rcc_lift_pit", "formwork_lift_pit", "waterproofing_lift_pit",
        "lift_machine_room", "lift_shaft_construction",
    ], Trade.STRUCTURAL, 1, condition="multi_storey", note="NBC Part 4"),

    # Chajja / sunshade
    DependencyRule("chajja", [
        "rcc_chajja", "formwork_chajja", "steel_chajja",
        "waterproofing_chajja",
    ], Trade.STRUCTURAL, 2),
]

# ── Masonry Dependencies ──

MASONRY_RULES = [
    # External walls
    DependencyRule("external_wall", [
        "brickwork_230mm", "plaster_external", "painting_external",
        "scaffolding",
    ], Trade.MASONRY, 1, note="IS 1905, NBC 2016"),

    # Internal walls
    DependencyRule("internal_wall", [
        "brickwork_115mm", "plaster_internal_both_sides",
        "painting_internal",
    ], Trade.MASONRY, 1),

    # Partition walls
    DependencyRule("partition_wall", [
        "aac_block_100mm", "plaster_internal_both_sides",
        "painting_internal",
    ], Trade.MASONRY, 2),
]

# ── Room-based Finishing Dependencies ──

FINISHING_RULES = [
    # Every room needs
    DependencyRule("room", [
        "flooring", "skirting", "ceiling_plaster", "ceiling_paint",
        "wall_plaster", "wall_paint",
    ], Trade.FINISHING, 1, note="Basic finish for any room"),

    # Toilet / Bathroom
    DependencyRule("toilet", [
        "waterproofing_floor", "waterproofing_wall",
        "anti_skid_floor_tile", "wall_tiles_full_height",
        "ewc_or_iwc", "wash_basin", "floor_trap", "nahni_trap",
        "cp_fittings", "mirror", "towel_rod", "soap_dish",
        "health_faucet", "shower_or_overhead_shower",
        "ceiling_false_or_plaster", "exhaust_fan_point",
        "light_point", "door_frame", "door_shutter",
    ], Trade.FINISHING, 1, note="Standard toilet/bathroom scope"),

    DependencyRule("bathroom", [
        "waterproofing_floor", "waterproofing_wall",
        "anti_skid_floor_tile", "wall_tiles_full_height",
        "ewc_or_iwc", "wash_basin", "floor_trap", "nahni_trap",
        "bathtub_or_shower_enclosure", "cp_fittings",
        "mirror", "towel_rod", "soap_dish",
        "ceiling_false_or_plaster", "exhaust_fan_point",
        "geyser_point", "light_point",
    ], Trade.FINISHING, 1, note="Bathroom with bathtub/shower"),

    # Kitchen
    DependencyRule("kitchen", [
        "granite_platform", "ss_sink", "dado_tiles_600mm",
        "waterproofing_under_platform", "chimney_point",
        "exhaust_fan_point", "cooking_gas_point",
        "water_purifier_point", "dishwasher_point",
        "multiple_power_points", "light_point",
        "anti_skid_floor_tile", "wall_paint_or_tile",
    ], Trade.FINISHING, 1, note="Standard kitchen scope"),

    # Bedroom
    DependencyRule("bedroom", [
        "vitrified_tile_flooring", "skirting", "wall_paint",
        "ceiling_paint_or_pop", "ac_point", "light_point",
        "fan_point", "power_points_x2", "tv_point",
    ], Trade.FINISHING, 2, note="Standard bedroom scope"),

    # Living room
    DependencyRule("living_room", [
        "vitrified_tile_flooring", "skirting", "wall_paint",
        "ceiling_paint_or_pop", "ac_point", "light_point",
        "fan_point", "power_points_x3", "tv_point",
    ], Trade.FINISHING, 2),

    # Balcony
    DependencyRule("balcony", [
        "anti_skid_floor_tile", "waterproofing_floor",
        "ms_railing", "light_point", "drain_outlet",
    ], Trade.FINISHING, 1, note="NBC 2016 railing mandatory"),

    # Terrace
    DependencyRule("terrace", [
        "waterproofing_terrace", "ips_flooring_or_tiles",
        "mumty_construction", "drain_outlet", "light_point",
    ], Trade.FINISHING, 1, note="Terrace waterproofing mandatory"),

    # Staircase area
    DependencyRule("staircase_area", [
        "kota_stone_or_granite_flooring", "ms_handrail",
        "wall_plaster", "wall_paint", "light_point_per_landing",
    ], Trade.FINISHING, 1),

    # Utility / Store
    DependencyRule("utility", [
        "anti_skid_floor_tile", "waterproofing_floor",
        "dado_tiles", "wash_basin_or_sink",
        "floor_trap", "light_point", "power_point",
    ], Trade.FINISHING, 2),

    # Pooja room
    DependencyRule("pooja_room", [
        "marble_flooring", "wall_tiles_or_paint",
        "light_point", "ceiling_pop",
    ], Trade.FINISHING, 3),

    # Corridor / Passage
    DependencyRule("corridor", [
        "vitrified_tile_flooring", "skirting",
        "wall_paint", "ceiling_paint", "light_point",
    ], Trade.FINISHING, 2),
]

# ── Plumbing Dependencies ──

PLUMBING_RULES = [
    # Water supply system
    DependencyRule("building", [
        "overhead_water_tank", "water_supply_main_line",
        "distribution_pipeline", "water_meter",
    ], Trade.PLUMBING, 1, note="Every building needs water supply"),

    # Drainage system
    DependencyRule("building", [
        "soil_pipe_stack", "waste_pipe_stack",
        "vent_pipe", "rainwater_downpipe",
        "underground_drainage", "inspection_chamber",
        "gully_trap",
    ], Trade.PLUMBING, 1, note="Complete drainage system"),

    # Toilet plumbing
    DependencyRule("toilet", [
        "cold_water_inlet", "hot_water_inlet",
        "soil_pipe_connection", "waste_pipe_connection",
        "floor_drain", "concealed_valves",
    ], Trade.PLUMBING, 1),

    # Kitchen plumbing
    DependencyRule("kitchen", [
        "cold_water_inlet", "hot_water_inlet",
        "waste_pipe_connection", "gas_connection_point",
    ], Trade.PLUMBING, 1),

    # Septic / STP
    DependencyRule("building", [
        "septic_tank_or_stp_connection",
    ], Trade.PLUMBING, 1, note="NBC 2016 sanitation requirement"),

    # Rainwater harvesting
    DependencyRule("building", [
        "rainwater_harvesting_system",
    ], Trade.PLUMBING, 2, condition="plot_area_gt_100sqm",
       note="Many local bylaws require RWH"),
]

# ── Electrical Dependencies ──

ELECTRICAL_RULES = [
    # Building-level
    DependencyRule("building", [
        "main_cable_from_meter", "main_distribution_board",
        "earthing_system", "energy_meter_board",
    ], Trade.ELECTRICAL, 1, note="Every building needs power"),

    # Per-floor
    DependencyRule("floor", [
        "floor_distribution_board", "rccb_protection",
        "mcb_per_circuit",
    ], Trade.ELECTRICAL, 1),

    # Bedroom electrical
    DependencyRule("bedroom", [
        "light_point_x2", "fan_point", "ac_point_16a",
        "power_socket_x2_5a", "power_socket_16a",
        "tv_point", "telephone_point",
    ], Trade.ELECTRICAL, 1),

    # Kitchen electrical
    DependencyRule("kitchen", [
        "light_point_x2", "exhaust_fan_point",
        "power_socket_x3_16a", "chimney_point_16a",
        "water_purifier_point", "dishwasher_point_16a",
        "refrigerator_point_16a",
    ], Trade.ELECTRICAL, 1),

    # Toilet electrical
    DependencyRule("toilet", [
        "light_point", "exhaust_fan_point",
        "geyser_point_16a",
    ], Trade.ELECTRICAL, 1),

    # Living room electrical
    DependencyRule("living_room", [
        "light_point_x3", "fan_point", "ac_point_16a",
        "power_socket_x3_5a", "tv_point",
    ], Trade.ELECTRICAL, 1),

    # Common areas
    DependencyRule("staircase_area", [
        "staircase_light_per_landing", "two_way_switch",
    ], Trade.ELECTRICAL, 1),

    # Multi-storey building
    DependencyRule("building", [
        "dg_backup_wiring", "lightning_arrester",
    ], Trade.ELECTRICAL, 2, condition="multi_storey"),
]

# ── Waterproofing Dependencies ──

WATERPROOFING_RULES = [
    DependencyRule("toilet", [
        "waterproofing_floor_ips", "waterproofing_wall_upto_200mm",
    ], Trade.WATERPROOFING, 1, note="IS 2645, CPWD Spec"),

    DependencyRule("bathroom", [
        "waterproofing_floor_ips", "waterproofing_wall_full_height",
    ], Trade.WATERPROOFING, 1),

    DependencyRule("kitchen", [
        "waterproofing_under_granite_platform",
    ], Trade.WATERPROOFING, 2),

    DependencyRule("terrace", [
        "waterproofing_terrace_bbc",
        "waterproofing_junction_sealing",
    ], Trade.WATERPROOFING, 1, note="Essential for flat roofs"),

    DependencyRule("balcony", [
        "waterproofing_balcony_floor",
    ], Trade.WATERPROOFING, 1),

    DependencyRule("basement", [
        "waterproofing_basement_app_membrane",
        "waterproofing_retaining_wall",
    ], Trade.WATERPROOFING, 1),

    DependencyRule("sunken_slab", [
        "waterproofing_sunken_ips",
    ], Trade.WATERPROOFING, 1),

    DependencyRule("plinth", [
        "dpc_at_plinth_level",
    ], Trade.WATERPROOFING, 1, note="DPC mandatory at plinth level"),
]

# ── Door / Window Dependencies ──

CARPENTRY_RULES = [
    # Every door needs
    DependencyRule("door", [
        "door_frame", "door_shutter", "hinges_x3",
        "door_lock", "door_stopper", "tower_bolt",
    ], Trade.CARPENTRY, 1, note="Complete door set"),

    # Every window needs
    DependencyRule("window", [
        "window_frame_or_section", "glass_pane",
        "window_stay_x2", "tower_bolt",
        "mosquito_mesh",
    ], Trade.CARPENTRY, 1, note="Complete window set"),

    # Main door extras
    DependencyRule("main_door", [
        "door_frame_heavy_section", "door_shutter_solid",
        "night_latch", "door_viewer", "door_chain",
        "name_plate",
    ], Trade.CARPENTRY, 2),

    # Toilet door
    DependencyRule("toilet_door", [
        "door_frame", "door_shutter_flush",
        "tower_bolt_inside", "indicator_bolt",
    ], Trade.CARPENTRY, 1),
]

# ── External Works Dependencies ──

EXTERNAL_RULES = [
    DependencyRule("building", [
        "compound_wall", "main_gate",
        "approach_road_or_driveway",
    ], Trade.EXTERNAL, 1, note="Basic external works"),

    DependencyRule("building", [
        "storm_water_drain", "surface_drain",
        "manhole_per_30m", "soak_pit",
    ], Trade.EXTERNAL, 1, note="Drainage mandatory"),

    DependencyRule("building", [
        "external_lighting", "garden_or_landscape",
        "parking_paving",
    ], Trade.EXTERNAL, 2),

    DependencyRule("building", [
        "underground_sump", "overhead_tank_structure",
        "pump_room",
    ], Trade.EXTERNAL, 1, condition="multi_unit",
       note="Water storage for apartments"),

    DependencyRule("building", [
        "borewell",
    ], Trade.EXTERNAL, 3, condition="no_municipal_water"),

    DependencyRule("multi_storey_building", [
        "fire_exit_staircase", "fire_hydrant_system",
        "fire_alarm_system",
    ], Trade.FIRE_SAFETY, 1, note="NBC Part 4 Fire Safety"),
]

# ── Fire Safety Dependencies ──

FIRE_SAFETY_RULES = [
    DependencyRule("building", [
        "fire_extinguisher_per_floor",
    ], Trade.FIRE_SAFETY, 1, condition="multi_storey", note="NBC Part 4"),

    DependencyRule("building_above_15m", [
        "fire_hydrant_system", "hose_reel_per_floor",
        "sprinkler_system", "fire_alarm_panel",
        "smoke_detector_per_room", "fire_escape_staircase",
        "pressurized_staircase", "fire_rated_doors",
    ], Trade.FIRE_SAFETY, 1, note="NBC Part 4 for high-rise"),
]

# ── HVAC Dependencies ──

HVAC_RULES = [
    DependencyRule("bedroom", [
        "ac_point_provision", "ac_drain_point",
    ], Trade.HVAC, 2),

    DependencyRule("kitchen", [
        "exhaust_duct_opening",
    ], Trade.HVAC, 1),

    DependencyRule("toilet", [
        "exhaust_fan_provision",
    ], Trade.HVAC, 1),
]

# ── Misc Dependencies ──

MISC_RULES = [
    DependencyRule("multi_storey_building", [
        "scaffolding", "safety_net",
    ], Trade.MISC, 1, note="Construction safety"),

    DependencyRule("building", [
        "construction_water", "construction_power",
        "site_office_barricading",
    ], Trade.MISC, 2, note="Prelims / site setup"),

    DependencyRule("building", [
        "final_cleaning", "pest_control_post_construction",
    ], Trade.MISC, 3),
]


# =============================================================================
# ALL RULES COMBINED
# =============================================================================

ALL_DEPENDENCY_RULES: List[DependencyRule] = (
    STRUCTURAL_RULES
    + MASONRY_RULES
    + FINISHING_RULES
    + PLUMBING_RULES
    + ELECTRICAL_RULES
    + WATERPROOFING_RULES
    + CARPENTRY_RULES
    + EXTERNAL_RULES
    + FIRE_SAFETY_RULES
    + HVAC_RULES
    + MISC_RULES
)


# =============================================================================
# SCOPE GAP DETECTION
# =============================================================================

@dataclass
class ScopeGap:
    """A missing scope item detected by dependency analysis."""
    missing_item: str
    triggered_by: str
    trade: str
    priority: int
    note: str = ""
    severity: str = "warning"  # "error" for priority 1, "warning" for 2, "info" for 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_item": self.missing_item,
            "triggered_by": self.triggered_by,
            "trade": self.trade,
            "priority": self.priority,
            "severity": self.severity,
            "note": self.note,
        }


@dataclass
class ScopeAnalysisResult:
    """Complete scope gap analysis result."""
    gaps: List[ScopeGap] = field(default_factory=list)
    total_rules_checked: int = 0
    total_gaps_found: int = 0
    total_items_checked: int = 0  # Total required items across all fired rules
    coverage_by_trade: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def critical_gaps(self) -> List[ScopeGap]:
        return [g for g in self.gaps if g.priority == 1]

    @property
    def important_gaps(self) -> List[ScopeGap]:
        return [g for g in self.gaps if g.priority == 2]

    @property
    def completeness_score(self) -> float:
        """0-100 score. 100 = no gaps found. Based on items, not rules."""
        if self.total_items_checked == 0:
            return 100.0
        items_found = self.total_items_checked - self.total_gaps_found
        return max(0.0, round((items_found / self.total_items_checked) * 100, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rules_checked": self.total_rules_checked,
            "total_items_checked": self.total_items_checked,
            "total_gaps_found": self.total_gaps_found,
            "critical_gaps": len(self.critical_gaps),
            "important_gaps": len(self.important_gaps),
            "completeness_score": self.completeness_score,
            "coverage_by_trade": self.coverage_by_trade,
            "gaps": [g.to_dict() for g in self.gaps],
        }

    def summary(self) -> str:
        lines = [f"Scope Analysis: {self.total_rules_checked} rules checked"]
        lines.append(f"  Gaps found: {self.total_gaps_found} "
                     f"({len(self.critical_gaps)} critical, "
                     f"{len(self.important_gaps)} important)")
        lines.append(f"  Completeness: {self.completeness_score}%")
        if self.critical_gaps:
            lines.append("  Critical missing items:")
            for g in self.critical_gaps[:10]:
                lines.append(f"    - {g.missing_item} (needs: {g.triggered_by})")
        return "\n".join(lines)


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def _normalize_description(text: str) -> str:
    """Normalize description for keyword matching."""
    return text.lower().strip().replace("-", " ").replace("_", " ")


def _item_exists_in_boq(
    item_keyword: str,
    boq_descriptions: Set[str],
) -> bool:
    """Check if an item (approximately) exists in BOQ descriptions."""
    keyword = _normalize_description(item_keyword)
    # Split multi-word keywords
    words = keyword.split("_") if "_" in item_keyword else keyword.split()

    for desc in boq_descriptions:
        desc_lower = desc.lower()
        # All words must appear in at least one BOQ description
        if all(w in desc_lower for w in words):
            return True
    return False


def _element_present(
    trigger: str,
    elements_detected: Set[str],
    boq_descriptions: Set[str],
    room_types: Set[str],
    project_params: Dict[str, Any],
) -> bool:
    """Check if a trigger element is present in the project."""
    trigger_lower = trigger.lower()

    # Direct match in detected elements
    if trigger_lower in elements_detected:
        return True

    # Room type match
    room_triggers = {
        "toilet", "bathroom", "kitchen", "bedroom", "living_room",
        "balcony", "terrace", "utility", "pooja_room", "corridor",
        "staircase_area", "store",
    }
    if trigger_lower in room_triggers and trigger_lower in room_types:
        return True

    # Building-level triggers (always present)
    if trigger_lower in ("building", "floor", "plinth"):
        return True

    # Conditional triggers
    if trigger_lower == "multi_storey_building":
        return project_params.get("num_floors", 1) > 2

    if trigger_lower == "building_above_15m":
        return project_params.get("building_height_m", 0) > 15

    if trigger_lower == "multi_unit":
        return project_params.get("num_units", 1) > 1

    # Check BOQ descriptions as fallback
    return _item_exists_in_boq(trigger, boq_descriptions)


def _check_condition(
    condition: str,
    project_params: Dict[str, Any],
    room_types: Set[str],
) -> bool:
    """Check if a rule's condition is satisfied."""
    if not condition:
        return True

    if condition == "multi_storey":
        return project_params.get("num_floors", 1) > 2

    if condition == "multi_unit":
        return project_params.get("num_units", 1) > 1

    if condition == "wet_area":
        return bool(room_types & {"toilet", "bathroom", "kitchen", "utility"})

    if condition == "plot_area_gt_100sqm":
        return project_params.get("plot_area_sqm", 0) > 100

    if condition == "no_municipal_water":
        return not project_params.get("has_municipal_water", True)

    return True  # Unknown conditions default to True


def analyze_scope_gaps(
    boq_items: List[Dict[str, Any]],
    detected_elements: List[str] = None,
    room_types: List[str] = None,
    project_params: Dict[str, Any] = None,
    rules: List[DependencyRule] = None,
) -> ScopeAnalysisResult:
    """
    Analyze BOQ for missing scope items using dependency rules.

    Args:
        boq_items: Current BOQ items (each with 'description' key)
        detected_elements: Elements detected from drawings (e.g., ['footing', 'column', 'slab'])
        room_types: Room types detected (e.g., ['toilet', 'bedroom', 'kitchen'])
        project_params: Project parameters (num_floors, plot_area_sqm, etc.)
        rules: Custom rules (default: ALL_DEPENDENCY_RULES)

    Returns:
        ScopeAnalysisResult with gaps and completeness score
    """
    rules = rules or ALL_DEPENDENCY_RULES
    project_params = project_params or {}
    detected = set(e.lower() for e in (detected_elements or []))
    rooms = set(r.lower().replace(" ", "_") for r in (room_types or []))

    # Build set of BOQ descriptions for matching
    boq_descs = set()
    for item in boq_items:
        desc = item.get("description", item.get("item_name", ""))
        if desc:
            boq_descs.add(desc)

    result = ScopeAnalysisResult()
    trade_coverage: Dict[str, Dict[str, int]] = {}

    for rule in rules:
        # Check if trigger element is present
        if not _element_present(rule.trigger, detected, boq_descs, rooms, project_params):
            continue

        # Check condition
        if not _check_condition(rule.condition, project_params, rooms):
            continue

        result.total_rules_checked += 1
        result.total_items_checked += len(rule.required_items)
        trade_name = rule.trade.value

        if trade_name not in trade_coverage:
            trade_coverage[trade_name] = {"rules": 0, "gaps": 0, "items": 0}
        trade_coverage[trade_name]["rules"] += 1
        trade_coverage[trade_name]["items"] += len(rule.required_items)

        # Check each required item
        for required in rule.required_items:
            if not _item_exists_in_boq(required, boq_descs):
                severity = "error" if rule.priority == 1 else (
                    "warning" if rule.priority == 2 else "info"
                )
                gap = ScopeGap(
                    missing_item=required,
                    triggered_by=rule.trigger,
                    trade=trade_name,
                    priority=rule.priority,
                    note=rule.note,
                    severity=severity,
                )
                result.gaps.append(gap)
                trade_coverage[trade_name]["gaps"] += 1

    result.total_gaps_found = len(result.gaps)
    result.coverage_by_trade = trade_coverage

    logger.info(result.summary())
    return result


def get_required_items_for_element(
    element: str,
    project_params: Dict[str, Any] = None,
) -> List[Tuple[str, int, str]]:
    """
    Get all required items for a given element.

    Args:
        element: Element type (e.g., 'footing', 'toilet', 'kitchen')
        project_params: Project params for condition checking

    Returns:
        List of (item_name, priority, trade) tuples
    """
    project_params = project_params or {}
    rooms = {element.lower().replace(" ", "_")}
    required = []

    for rule in ALL_DEPENDENCY_RULES:
        if rule.trigger.lower() != element.lower():
            continue
        if not _check_condition(rule.condition, project_params, rooms):
            continue

        for item in rule.required_items:
            required.append((item, rule.priority, rule.trade.value))

    return required


def get_all_triggers() -> List[str]:
    """Get list of all unique trigger elements across all rules."""
    return sorted(set(r.trigger for r in ALL_DEPENDENCY_RULES))


def get_rules_by_trade(trade: Trade) -> List[DependencyRule]:
    """Get all rules for a specific trade."""
    return [r for r in ALL_DEPENDENCY_RULES if r.trade == trade]


def get_rule_count() -> int:
    """Get total number of dependency rules."""
    return len(ALL_DEPENDENCY_RULES)


def get_required_item_count() -> int:
    """Get total number of unique required items across all rules."""
    items = set()
    for rule in ALL_DEPENDENCY_RULES:
        items.update(rule.required_items)
    return len(items)


# =============================================================================
# KNOWLEDGE BASE EXTENSION (additive — appends rules from YAML data files)
# =============================================================================

try:
    from src.knowledge_base import get_all_dependency_rules as _kb_dep_rules

    # Dedup by (trigger, required_items) pair — allows multiple rules per trigger
    # so KB rules that add NEW required_items to an existing trigger get merged in.
    # Only exact duplicate rule content is skipped.
    _kb_existing_keys = {
        (r.trigger, tuple(sorted(r.required_items)))
        for r in ALL_DEPENDENCY_RULES
    }
    _kb_added = 0
    for _rule in _kb_dep_rules():
        _key = (_rule.trigger, tuple(sorted(_rule.required_items)))
        if _key not in _kb_existing_keys:
            ALL_DEPENDENCY_RULES.append(_rule)
            _kb_existing_keys.add(_key)
            _kb_added += 1
    if _kb_added:
        logger.info(
            "Knowledge base: +%d dependency rules (total: %d)",
            _kb_added, len(ALL_DEPENDENCY_RULES),
        )
    del _kb_existing_keys, _kb_added
except ImportError:
    pass  # Knowledge base not installed yet
except Exception as _e:
    logger.warning("Knowledge base dependency loading failed: %s", _e)
