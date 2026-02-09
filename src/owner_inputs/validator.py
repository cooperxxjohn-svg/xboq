"""
Owner Inputs Validator - Validate owner inputs against template.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class MissingField:
    """Represents a missing field in owner inputs."""
    path: str  # Dot-separated path: finishes.flooring.living_dining.type
    field_name: str  # Human readable name
    is_mandatory: bool
    priority: str  # critical / important / optional
    why_needed: str
    impact: str
    options: List[str] = field(default_factory=list)
    default_value: Any = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "field_name": self.field_name,
            "is_mandatory": self.is_mandatory,
            "priority": self.priority,
            "why_needed": self.why_needed,
            "impact": self.impact,
            "options": self.options,
            "default_value": self.default_value,
        }


@dataclass
class ValidationResult:
    """Result of owner inputs validation."""
    missing_mandatory: List[MissingField]
    missing_optional: List[MissingField]
    provided_fields: List[str]
    completeness_score: float
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "missing_mandatory": [f.to_dict() for f in self.missing_mandatory],
            "missing_optional": [f.to_dict() for f in self.missing_optional],
            "provided_fields": self.provided_fields,
            "completeness_score": self.completeness_score,
            "validation_errors": self.validation_errors,
        }


class OwnerInputsValidator:
    """Validate owner inputs against template."""

    # Field metadata: why needed, impact, priority, options
    FIELD_METADATA = {
        # Project basics
        "project.name": {
            "priority": "critical",
            "why_needed": "Project identification and documentation",
            "impact": "Cannot generate proper bid documents",
            "options": [],
        },
        "project.location.city": {
            "priority": "critical",
            "why_needed": "Location multiplier for rates, material availability",
            "impact": "Cannot apply correct city/regional rate factors",
            "options": [],
        },
        "project.location.state": {
            "priority": "critical",
            "why_needed": "State-specific regulations, GST registration",
            "impact": "Cannot determine applicable taxes and regulations",
            "options": [],
        },
        "project.type": {
            "priority": "critical",
            "why_needed": "Building typology affects finish specs, MEP requirements",
            "impact": "Cannot determine appropriate specifications",
            "options": ["residential", "commercial", "industrial", "institutional"],
        },
        "project.completion_months": {
            "priority": "critical",
            "why_needed": "Duration affects prelims, supervision, equipment costs",
            "impact": "Cannot calculate time-based preliminary costs",
            "options": [],
        },

        # Finishes
        "finishes.grade": {
            "priority": "critical",
            "why_needed": "Overall finish grade drives material selection and rates",
            "impact": "Cannot determine appropriate material costs",
            "options": ["basic", "standard", "premium", "luxury"],
        },
        "finishes.flooring.living_dining.type": {
            "priority": "critical",
            "why_needed": "Flooring material for main living areas",
            "impact": "Flooring cost can vary 3-10x based on material",
            "options": ["vitrified_tile", "marble", "granite", "wooden", "other"],
        },
        "finishes.flooring.living_dining.size": {
            "priority": "important",
            "why_needed": "Tile size affects wastage and labor rates",
            "impact": "Incorrect wastage and labor calculation",
            "options": ["600x600", "800x800", "1200x600", "as_per_layout"],
        },
        "finishes.flooring.bedrooms.type": {
            "priority": "critical",
            "why_needed": "Flooring material for bedrooms",
            "impact": "Cannot estimate bedroom flooring cost",
            "options": ["vitrified_tile", "marble", "granite", "wooden", "other"],
        },
        "finishes.flooring.bedrooms.size": {
            "priority": "important",
            "why_needed": "Tile size for bedrooms",
            "impact": "Incorrect wastage calculation",
            "options": ["600x600", "800x800", "1200x600", "as_per_layout"],
        },
        "finishes.flooring.kitchen.type": {
            "priority": "critical",
            "why_needed": "Kitchen flooring material",
            "impact": "Cannot estimate kitchen flooring",
            "options": ["vitrified_tile", "ceramic", "granite", "other"],
        },
        "finishes.flooring.kitchen.size": {
            "priority": "important",
            "why_needed": "Kitchen tile size",
            "impact": "Incorrect wastage calculation",
            "options": ["600x600", "300x300", "600x1200"],
        },
        "finishes.flooring.bathrooms.type": {
            "priority": "critical",
            "why_needed": "Bathroom flooring (must be anti-skid)",
            "impact": "Cannot estimate bathroom flooring",
            "options": ["ceramic", "vitrified", "stone"],
        },
        "finishes.flooring.bathrooms.size": {
            "priority": "important",
            "why_needed": "Bathroom tile size",
            "impact": "Incorrect wastage calculation",
            "options": ["300x300", "400x400", "600x600"],
        },
        "finishes.flooring.balcony_terrace.type": {
            "priority": "important",
            "why_needed": "Balcony/terrace flooring (anti-skid)",
            "impact": "Cannot estimate external flooring",
            "options": ["kota", "ceramic", "vitrified", "paver"],
        },
        "finishes.flooring.common_areas.type": {
            "priority": "important",
            "why_needed": "Common area flooring material",
            "impact": "Cannot estimate common area flooring",
            "options": ["vitrified_tile", "marble", "granite", "kota"],
        },
        "finishes.flooring.common_areas.size": {
            "priority": "important",
            "why_needed": "Common area tile size",
            "impact": "Incorrect wastage calculation",
            "options": ["600x600", "800x800", "1200x600"],
        },
        "finishes.flooring.parking.type": {
            "priority": "important",
            "why_needed": "Parking area flooring",
            "impact": "Cannot estimate parking flooring",
            "options": ["paver_blocks", "vdf", "kota", "exposed_concrete"],
        },

        # Wall finishes
        "finishes.wall_finishes.internal_paint.type": {
            "priority": "critical",
            "why_needed": "Internal paint specification",
            "impact": "Paint costs can vary 2-5x based on type",
            "options": ["acrylic_emulsion", "oil_bound_distemper", "texture"],
        },
        "finishes.wall_finishes.external_paint.type": {
            "priority": "critical",
            "why_needed": "External paint specification",
            "impact": "External paint affects durability and cost",
            "options": ["acrylic_exterior", "texture", "elastomeric"],
        },
        "finishes.wall_finishes.bathroom_dado.height_mm": {
            "priority": "critical",
            "why_needed": "Bathroom wall tile height",
            "impact": "Tile quantity depends on dado height",
            "options": ["full_height", "1200", "2100"],
        },
        "finishes.wall_finishes.bathroom_dado.tile_type": {
            "priority": "critical",
            "why_needed": "Bathroom wall tile material",
            "impact": "Cannot estimate bathroom wall tiles",
            "options": ["ceramic", "vitrified", "designer"],
        },
        "finishes.wall_finishes.bathroom_dado.tile_size": {
            "priority": "important",
            "why_needed": "Bathroom wall tile size",
            "impact": "Incorrect wastage calculation",
            "options": ["300x450", "300x600", "600x600"],
        },
        "finishes.wall_finishes.kitchen_dado.height_mm": {
            "priority": "critical",
            "why_needed": "Kitchen dado tile height",
            "impact": "Tile quantity depends on dado height",
            "options": ["600", "750", "platform_top_only"],
        },
        "finishes.wall_finishes.kitchen_dado.tile_type": {
            "priority": "important",
            "why_needed": "Kitchen dado tile material",
            "impact": "Cannot estimate kitchen wall tiles",
            "options": ["ceramic", "vitrified", "designer"],
        },

        # Ceiling
        "finishes.ceiling.type": {
            "priority": "critical",
            "why_needed": "Ceiling finish type",
            "impact": "False ceiling significantly affects cost",
            "options": ["plain_plaster", "pop_finish", "gypsum_false_ceiling", "grid_ceiling"],
        },

        # Waterproofing
        "waterproofing.toilet.system": {
            "priority": "critical",
            "why_needed": "Toilet waterproofing system",
            "impact": "Waterproofing is critical - failure is costly",
            "options": ["cementitious_coating", "integral_compound", "app_membrane", "liquid_membrane"],
        },
        "waterproofing.terrace.system": {
            "priority": "critical",
            "why_needed": "Terrace waterproofing system",
            "impact": "Terrace leakage is common problem",
            "options": ["brick_bat_coba", "app_membrane", "liquid_membrane", "heat_insulation_tiles"],
        },
        "waterproofing.water_tank.system": {
            "priority": "critical",
            "why_needed": "Water tank waterproofing",
            "impact": "Tank seepage causes structural issues",
            "options": ["cementitious_coating", "epoxy_coating", "crystalline"],
        },

        # Doors
        "doors.main_entrance.type": {
            "priority": "critical",
            "why_needed": "Main door material and finish",
            "impact": "Main door cost varies significantly",
            "options": ["teak_solid", "teak_veneer", "flush_laminated", "designer"],
        },
        "doors.main_entrance.hardware_level": {
            "priority": "important",
            "why_needed": "Door hardware quality level",
            "impact": "Hardware costs vary 3-10x",
            "options": ["basic", "standard", "premium"],
        },
        "doors.internal.type": {
            "priority": "critical",
            "why_needed": "Internal door material",
            "impact": "Affects cost per door significantly",
            "options": ["flush_painted", "flush_laminated", "skin_moulded"],
        },
        "doors.internal.frame_type": {
            "priority": "critical",
            "why_needed": "Door frame material",
            "impact": "Frame material affects durability and cost",
            "options": ["sal_wood", "teak", "pressed_steel", "wpc"],
        },
        "doors.internal.hardware_level": {
            "priority": "important",
            "why_needed": "Internal door hardware",
            "impact": "Hardware cost per door",
            "options": ["basic", "standard", "premium"],
        },
        "doors.bathroom.type": {
            "priority": "critical",
            "why_needed": "Bathroom door material (water resistant)",
            "impact": "Must be water resistant",
            "options": ["wpc", "pvc", "upvc", "aluminum"],
        },
        "doors.bathroom.hardware_level": {
            "priority": "important",
            "why_needed": "Bathroom door hardware",
            "impact": "Hardware cost",
            "options": ["basic", "standard", "premium"],
        },
        "doors.service.type": {
            "priority": "important",
            "why_needed": "Service door material",
            "impact": "Service door cost",
            "options": ["ms_painted", "gi", "flush"],
        },

        # Windows
        "windows.type": {
            "priority": "critical",
            "why_needed": "Window frame material",
            "impact": "Window costs vary 2-4x based on material",
            "options": ["aluminum_powder_coated", "upvc", "wood", "aluminum_anodized"],
        },
        "windows.glass_type": {
            "priority": "critical",
            "why_needed": "Glass specification",
            "impact": "Glass affects cost and performance",
            "options": ["clear", "tinted", "reflective", "saint_gobain_equivalent"],
        },
        "windows.glass_thickness_mm": {
            "priority": "important",
            "why_needed": "Glass thickness",
            "impact": "Affects structural and cost",
            "options": ["5", "6", "8"],
        },
        "windows.grill": {
            "priority": "important",
            "why_needed": "Window grill type",
            "impact": "Grill cost varies by material",
            "options": ["ms_grill", "ss_grill", "none"],
        },

        # Sanitary
        "sanitary.brand_level": {
            "priority": "critical",
            "why_needed": "Sanitary ware quality level",
            "impact": "Costs vary 3-10x based on brand",
            "options": ["basic", "standard", "premium", "luxury"],
        },
        "sanitary.wc_type": {
            "priority": "critical",
            "why_needed": "WC type selection",
            "impact": "EWC vs Indian affects cost and plumbing",
            "options": ["ewc", "indian_wc", "both"],
        },
        "sanitary.cistern_type": {
            "priority": "important",
            "why_needed": "Cistern type",
            "impact": "Concealed cistern costs more",
            "options": ["exposed", "concealed"],
        },
        "sanitary.wash_basin.type": {
            "priority": "important",
            "why_needed": "Wash basin type",
            "impact": "Basin type affects cost",
            "options": ["wall_hung", "pedestal", "counter_top"],
        },
        "sanitary.shower.type": {
            "priority": "important",
            "why_needed": "Shower fitting type",
            "impact": "Rain shower costs more",
            "options": ["overhead_only", "overhead_hand", "rain_shower"],
        },
        "sanitary.cp_fittings.brand_level": {
            "priority": "critical",
            "why_needed": "CP fittings quality level",
            "impact": "CP costs vary significantly",
            "options": ["basic", "standard", "premium", "luxury"],
        },
        "sanitary.accessories.included": {
            "priority": "important",
            "why_needed": "Bathroom accessories in scope",
            "impact": "Accessories cost addition",
            "options": ["true", "false"],
        },
        "sanitary.accessories.brand_level": {
            "priority": "important",
            "why_needed": "Accessories quality level",
            "impact": "Accessory cost per bathroom",
            "options": ["basic", "standard", "premium"],
        },

        # Plumbing
        "plumbing.water_supply.pipe_type": {
            "priority": "critical",
            "why_needed": "Water supply pipe material",
            "impact": "Pipe material affects cost and durability",
            "options": ["cpvc", "ppr", "composite"],
        },
        "plumbing.water_supply.cpvc_class": {
            "priority": "important",
            "why_needed": "CPVC pipe class/schedule",
            "impact": "Affects pipe cost",
            "options": ["sch40", "sch80", "sdr11"],
        },
        "plumbing.drainage.pipe_type": {
            "priority": "critical",
            "why_needed": "Drainage pipe material",
            "impact": "Drainage pipe cost",
            "options": ["upvc_swr", "hdpe", "ci"],
        },
        "plumbing.hot_water.required": {
            "priority": "important",
            "why_needed": "Hot water system in scope",
            "impact": "Additional plumbing and equipment",
            "options": ["true", "false"],
        },

        # Electrical
        "electrical.brand_level": {
            "priority": "critical",
            "why_needed": "Electrical fittings quality",
            "impact": "Electrical costs vary significantly",
            "options": ["basic", "standard", "premium"],
        },
        "electrical.switches.type": {
            "priority": "critical",
            "why_needed": "Switch type (modular/conventional)",
            "impact": "Modular costs more but standard now",
            "options": ["modular", "conventional"],
        },
        "electrical.switches.brand_range": {
            "priority": "critical",
            "why_needed": "Switch brand",
            "impact": "Brand affects cost significantly",
            "options": ["anchor", "legrand", "schneider", "havells"],
        },
        "electrical.wiring.type": {
            "priority": "critical",
            "why_needed": "Wire type/insulation",
            "impact": "Fire rating affects cost",
            "options": ["fr_pvc", "frls", "lszh"],
        },
        "electrical.wiring.conduit_type": {
            "priority": "important",
            "why_needed": "Conduit material",
            "impact": "Conduit cost",
            "options": ["pvc", "gi", "concealed_in_slab"],
        },
        "electrical.db_panel.type": {
            "priority": "important",
            "why_needed": "DB panel type",
            "impact": "Panel cost",
            "options": ["sp", "tp"],
        },
        "electrical.db_panel.mcb_brand": {
            "priority": "important",
            "why_needed": "MCB brand",
            "impact": "MCB cost",
            "options": ["havells", "legrand", "schneider", "abb"],
        },
        "electrical.earthing.type": {
            "priority": "important",
            "why_needed": "Earthing system type",
            "impact": "Earthing cost",
            "options": ["plate", "pipe", "chemical"],
        },
        "electrical.generator_backup.required": {
            "priority": "important",
            "why_needed": "Generator/DG in scope",
            "impact": "Major equipment cost",
            "options": ["true", "false"],
        },

        # HVAC
        "hvac.required": {
            "priority": "critical",
            "why_needed": "HVAC system in scope",
            "impact": "HVAC can be 10-15% of cost",
            "options": ["true", "false"],
        },
        "hvac.type": {
            "priority": "important",
            "why_needed": "HVAC system type",
            "impact": "VRV/central costs much more than split",
            "options": ["split_ac_provision", "vrv", "central_ac"],
        },

        # Fire
        "fire.required": {
            "priority": "critical",
            "why_needed": "Fire system in scope (check bylaws)",
            "impact": "Fire safety is mandatory above certain heights",
            "options": ["true", "false"],
        },
        "fire.type": {
            "priority": "important",
            "why_needed": "Fire system type",
            "impact": "Sprinkler costs more than hydrant only",
            "options": ["hydrant", "sprinkler", "both"],
        },

        # Lift
        "lift.required": {
            "priority": "critical",
            "why_needed": "Lift in scope",
            "impact": "Lift is major equipment cost",
            "options": ["true", "false"],
        },
        "lift.count": {
            "priority": "important",
            "why_needed": "Number of lifts",
            "impact": "Per-lift cost",
            "options": [],
        },
        "lift.type": {
            "priority": "important",
            "why_needed": "Lift type",
            "impact": "Service lift costs more",
            "options": ["passenger", "service", "both"],
        },
        "lift.capacity_persons": {
            "priority": "important",
            "why_needed": "Lift capacity",
            "impact": "Larger capacity costs more",
            "options": ["6", "8", "10", "13", "15"],
        },
        "lift.brand_range": {
            "priority": "important",
            "why_needed": "Lift brand tier",
            "impact": "Brand affects cost 2-3x",
            "options": ["otis", "kone", "johnson", "thyssenkrupp", "indian"],
        },

        # External works
        "external_works.compound_wall.required": {
            "priority": "important",
            "why_needed": "Compound wall in scope",
            "impact": "Significant external cost",
            "options": ["true", "false"],
        },
        "external_works.compound_wall.height_m": {
            "priority": "important",
            "why_needed": "Compound wall height",
            "impact": "Height affects quantity",
            "options": [],
        },
        "external_works.compound_wall.type": {
            "priority": "important",
            "why_needed": "Compound wall type",
            "impact": "Material affects cost",
            "options": ["brick", "rcc", "precast", "ms_grill"],
        },
        "external_works.gate.type": {
            "priority": "important",
            "why_needed": "Gate type",
            "impact": "Automatic gates cost much more",
            "options": ["ms_sliding", "ms_swing", "ss", "automatic"],
        },
        "external_works.paving.type": {
            "priority": "important",
            "why_needed": "External paving type",
            "impact": "Paving material cost",
            "options": ["paver_blocks", "kota", "concrete", "interlocking"],
        },

        # Commercial
        "commercial.gst_included": {
            "priority": "critical",
            "why_needed": "GST inclusion in rates",
            "impact": "18% difference in quoted amount",
            "options": ["true", "false"],
        },
        "commercial.escalation.applicable": {
            "priority": "important",
            "why_needed": "Escalation clause applicable",
            "impact": "Risk allocation for price changes",
            "options": ["true", "false"],
        },

        # Testing
        "testing.concrete.cube_test": {
            "priority": "important",
            "why_needed": "Concrete cube testing",
            "impact": "Testing costs",
            "options": ["true", "false"],
        },
        "testing.rebar.test_required": {
            "priority": "important",
            "why_needed": "Rebar testing",
            "impact": "Testing costs",
            "options": ["true", "false"],
        },
        "testing.waterproofing.flood_test": {
            "priority": "important",
            "why_needed": "Waterproofing flood test",
            "impact": "Testing and remediation if fail",
            "options": ["true", "false"],
        },
        "testing.plumbing.pressure_test": {
            "priority": "important",
            "why_needed": "Plumbing pressure test",
            "impact": "Testing costs",
            "options": ["true", "false"],
        },
        "testing.electrical.megger_test": {
            "priority": "important",
            "why_needed": "Electrical megger test",
            "impact": "Testing costs",
            "options": ["true", "false"],
        },
        "testing.electrical.earthing_test": {
            "priority": "important",
            "why_needed": "Earthing resistance test",
            "impact": "Testing costs",
            "options": ["true", "false"],
        },
    }

    def __init__(self):
        self.template = self._load_template()

    def _load_template(self) -> dict:
        """Load owner inputs template."""
        template_path = Path(__file__).parent.parent.parent / "templates" / "owner_inputs_template.yaml"

        if template_path.exists():
            with open(template_path, "r") as f:
                return yaml.safe_load(f) or {}
        else:
            # Return minimal template
            return {}

    def validate(self, owner_inputs: dict) -> ValidationResult:
        """Validate owner inputs against template."""
        missing_mandatory = []
        missing_optional = []
        provided_fields = []
        validation_errors = []

        # Check all fields in metadata
        for field_path, metadata in self.FIELD_METADATA.items():
            value = self._get_nested_value(owner_inputs, field_path)

            if value is None or value == "" or value == []:
                # Field is missing
                is_mandatory = metadata["priority"] in ["critical", "important"]

                missing_field = MissingField(
                    path=field_path,
                    field_name=field_path.split(".")[-1],
                    is_mandatory=is_mandatory,
                    priority=metadata["priority"],
                    why_needed=metadata["why_needed"],
                    impact=metadata["impact"],
                    options=metadata.get("options", []),
                    default_value=None,
                )

                if is_mandatory:
                    missing_mandatory.append(missing_field)
                else:
                    missing_optional.append(missing_field)
            else:
                provided_fields.append(field_path)

                # Validate value against options if provided
                options = metadata.get("options", [])
                if options and str(value).lower() not in [o.lower() for o in options]:
                    validation_errors.append(
                        f"{field_path}: value '{value}' not in allowed options {options}"
                    )

        # Calculate completeness score (only based on mandatory fields)
        total_mandatory = len(missing_mandatory) + len([f for f in provided_fields if self.FIELD_METADATA.get(f, {}).get("priority") in ["critical", "important"]])
        provided_mandatory = total_mandatory - len(missing_mandatory)
        completeness_score = (provided_mandatory / total_mandatory * 100) if total_mandatory > 0 else 100

        return ValidationResult(
            missing_mandatory=missing_mandatory,
            missing_optional=missing_optional,
            provided_fields=provided_fields,
            completeness_score=completeness_score,
            validation_errors=validation_errors,
        )

    def _get_nested_value(self, obj: dict, path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split(".")
        value = obj

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

            if value is None:
                return None

        return value
