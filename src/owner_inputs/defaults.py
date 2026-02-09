"""
Defaults Engine - Apply standard defaults for missing optional fields.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class AppliedDefault:
    """Represents a default value applied to a missing field."""
    field_path: str
    value: Any
    basis: str  # Why this default was chosen
    source: str  # Where this default comes from (IS code, CPWD, industry standard, etc.)

    def to_dict(self) -> dict:
        return {
            "field_path": self.field_path,
            "value": self.value,
            "basis": self.basis,
            "source": self.source,
        }


class DefaultsEngine:
    """Apply standard defaults for missing optional fields."""

    # Default values with basis
    DEFAULTS = {
        # Project defaults
        "project.location.pincode": {
            "value": "",
            "basis": "Not required for estimation",
            "source": "System default",
        },
        "project.built_up_area_sqm": {
            "value": None,
            "basis": "Will be calculated from drawings",
            "source": "Drawing extraction",
        },
        "project.floors_above_ground": {
            "value": None,
            "basis": "Will be calculated from drawings",
            "source": "Drawing extraction",
        },
        "project.floors_below_ground": {
            "value": 0,
            "basis": "Assume no basement unless specified",
            "source": "Standard assumption",
        },

        # Flooring defaults
        "finishes.flooring.living_dining.make_range": {
            "value": "architect_approved",
            "basis": "Standard practice - architect to approve makes",
            "source": "Industry standard",
        },
        "finishes.flooring.bedrooms.make_range": {
            "value": "architect_approved",
            "basis": "Standard practice - architect to approve makes",
            "source": "Industry standard",
        },
        "finishes.flooring.kitchen.anti_skid": {
            "value": True,
            "basis": "Kitchen requires anti-skid as per safety norms",
            "source": "Building safety standards",
        },
        "finishes.flooring.parking.thickness_mm": {
            "value": 60,
            "basis": "Standard paver block thickness for light vehicles",
            "source": "Industry standard",
        },

        # Wall finishes defaults
        "finishes.wall_finishes.internal_paint.coats": {
            "value": 2,
            "basis": "Standard practice - 1 primer + 2 finish coats",
            "source": "Paint manufacturer recommendations",
        },
        "finishes.wall_finishes.internal_paint.make_range": {
            "value": "architect_approved",
            "basis": "Standard practice",
            "source": "Industry standard",
        },
        "finishes.wall_finishes.external_paint.coats": {
            "value": 3,
            "basis": "External requires more coats for durability",
            "source": "Paint manufacturer recommendations",
        },
        "finishes.wall_finishes.external_paint.make_range": {
            "value": "architect_approved",
            "basis": "Standard practice",
            "source": "Industry standard",
        },

        # Ceiling defaults
        "finishes.ceiling.false_ceiling_locations": {
            "value": [],
            "basis": "No false ceiling unless specified",
            "source": "Standard assumption",
        },

        # Waterproofing defaults
        "waterproofing.toilet.brand_range": {
            "value": "architect_approved",
            "basis": "Brand to be approved by architect",
            "source": "Industry standard",
        },
        "waterproofing.toilet.turnup_mm": {
            "value": 150,
            "basis": "Standard turnup height for waterproofing",
            "source": "IS 3036",
        },
        "waterproofing.terrace.heat_insulation": {
            "value": False,
            "basis": "Heat insulation not included unless specified",
            "source": "Standard assumption",
        },
        "waterproofing.terrace.brand_range": {
            "value": "architect_approved",
            "basis": "Brand to be approved by architect",
            "source": "Industry standard",
        },
        "waterproofing.water_tank.brand_range": {
            "value": "architect_approved",
            "basis": "Brand to be approved by architect",
            "source": "Industry standard",
        },
        "waterproofing.basement.required": {
            "value": False,
            "basis": "No basement waterproofing unless basement exists",
            "source": "Standard assumption",
        },
        "waterproofing.podium.required": {
            "value": False,
            "basis": "No podium waterproofing unless podium exists",
            "source": "Standard assumption",
        },

        # Door defaults
        "doors.main_entrance.size": {
            "value": "as_per_schedule",
            "basis": "Size as per door schedule in drawings",
            "source": "Drawing reference",
        },
        "doors.service.hardware_level": {
            "value": "basic",
            "basis": "Service doors use basic hardware",
            "source": "Industry standard",
        },

        # Window defaults
        "windows.mosquito_mesh": {
            "value": True,
            "basis": "Mosquito mesh standard in residential",
            "source": "Industry standard",
        },

        # Plumbing defaults
        "plumbing.water_supply.brand_range": {
            "value": "supreme_equivalent",
            "basis": "Standard ISI marked pipes",
            "source": "IS 15778",
        },
        "plumbing.drainage.brand_range": {
            "value": "supreme_equivalent",
            "basis": "Standard ISI marked pipes",
            "source": "IS 13592",
        },
        "plumbing.rainwater_harvesting.required": {
            "value": False,
            "basis": "RWH not included unless mandated",
            "source": "Standard assumption (check local bylaws)",
        },

        # Electrical defaults
        "electrical.wiring.brand_range": {
            "value": "polycab_equivalent",
            "basis": "Standard FR/FRLS wire",
            "source": "IS 694",
        },
        "electrical.solar.required": {
            "value": False,
            "basis": "Solar not included unless specified",
            "source": "Standard assumption",
        },

        # HVAC defaults
        "hvac.provision_only": {
            "value": True,
            "basis": "Only conduit and drain provision, not units",
            "source": "Standard residential practice",
        },

        # Fire defaults
        "fire.fire_extinguishers": {
            "value": True,
            "basis": "Fire extinguishers mandatory per NBC",
            "source": "NBC 2016",
        },

        # External works defaults
        "external_works.landscaping.required": {
            "value": False,
            "basis": "Landscaping excluded unless specified",
            "source": "Standard assumption",
        },

        # Commercial defaults
        "commercial.gst_rate": {
            "value": 18,
            "basis": "Standard GST rate for construction",
            "source": "GST Act",
        },
        "commercial.payment_terms.milestone_based": {
            "value": True,
            "basis": "Milestone-based payments standard",
            "source": "Industry standard",
        },
        "commercial.payment_terms.retention_percent": {
            "value": 5,
            "basis": "Standard retention percentage",
            "source": "Industry standard",
        },
        "commercial.defect_liability_months": {
            "value": 12,
            "basis": "Standard DLP period",
            "source": "Industry standard",
        },
        "commercial.insurance.car_required": {
            "value": True,
            "basis": "CAR insurance standard for construction",
            "source": "Industry standard",
        },
        "commercial.insurance.wc_required": {
            "value": True,
            "basis": "Workmen compensation mandatory",
            "source": "Workmen Compensation Act",
        },

        # Testing defaults
        "testing.concrete.third_party": {
            "value": False,
            "basis": "Third party testing not mandatory unless specified",
            "source": "Standard assumption",
        },
        "testing.concrete.frequency": {
            "value": "as_per_is",
            "basis": "Testing frequency as per IS 456",
            "source": "IS 456:2000",
        },
        "testing.rebar.mill_test_certificate": {
            "value": True,
            "basis": "Mill TC required for all steel",
            "source": "Industry standard",
        },
        "testing.waterproofing.duration_hours": {
            "value": 24,
            "basis": "24 hour flood test standard",
            "source": "Industry standard",
        },
        "testing.plumbing.test_pressure_bar": {
            "value": 10,
            "basis": "Standard pressure test at 10 bar",
            "source": "IS 3114",
        },

        # Site conditions defaults
        "site_conditions.access_restricted": {
            "value": False,
            "basis": "Normal access assumed",
            "source": "Standard assumption",
        },
        "site_conditions.working_hours_restricted": {
            "value": False,
            "basis": "Normal working hours assumed",
            "source": "Standard assumption",
        },
        "site_conditions.noise_restrictions": {
            "value": False,
            "basis": "No noise restrictions assumed",
            "source": "Standard assumption",
        },
        "site_conditions.existing_structure": {
            "value": False,
            "basis": "Clear site assumed",
            "source": "Standard assumption",
        },
        "site_conditions.soil_type": {
            "value": "normal_soil",
            "basis": "Normal soil assumed (verify with soil report)",
            "source": "Standard assumption",
        },
    }

    def apply_defaults(
        self,
        owner_inputs: dict,
        missing_optional: list,
    ) -> Tuple[List[AppliedDefault], dict]:
        """
        Apply defaults for missing optional fields.

        Returns:
            Tuple of (list of applied defaults, updated inputs dict)
        """
        applied = []
        result = self._deep_copy(owner_inputs)

        for field in missing_optional:
            if field.path in self.DEFAULTS:
                default_info = self.DEFAULTS[field.path]
                value = default_info["value"]

                # Set the value in result
                self._set_nested_value(result, field.path, value)

                applied.append(AppliedDefault(
                    field_path=field.path,
                    value=value,
                    basis=default_info["basis"],
                    source=default_info["source"],
                ))

        return applied, result

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a nested dict/list structure."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def _set_nested_value(self, obj: dict, path: str, value: Any) -> None:
        """Set value in nested dict using dot notation."""
        keys = path.split(".")

        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]

        obj[keys[-1]] = value

    def get_default_for_field(self, field_path: str) -> dict:
        """Get default info for a specific field."""
        return self.DEFAULTS.get(field_path)

    def get_grade_based_defaults(self, grade: str) -> dict:
        """Get defaults based on finish grade (basic/standard/premium/luxury)."""
        grade_defaults = {
            "basic": {
                "finishes.flooring.living_dining.type": "vitrified_tile",
                "finishes.flooring.living_dining.size": "600x600",
                "finishes.wall_finishes.internal_paint.type": "oil_bound_distemper",
                "sanitary.brand_level": "basic",
                "sanitary.cp_fittings.brand_level": "basic",
                "doors.main_entrance.type": "flush_laminated",
                "doors.internal.type": "flush_painted",
                "windows.type": "aluminum_anodized",
                "electrical.brand_level": "basic",
            },
            "standard": {
                "finishes.flooring.living_dining.type": "vitrified_tile",
                "finishes.flooring.living_dining.size": "800x800",
                "finishes.wall_finishes.internal_paint.type": "acrylic_emulsion",
                "sanitary.brand_level": "standard",
                "sanitary.cp_fittings.brand_level": "standard",
                "doors.main_entrance.type": "teak_veneer",
                "doors.internal.type": "flush_laminated",
                "windows.type": "aluminum_powder_coated",
                "electrical.brand_level": "standard",
            },
            "premium": {
                "finishes.flooring.living_dining.type": "marble",
                "finishes.flooring.living_dining.size": "1200x600",
                "finishes.wall_finishes.internal_paint.type": "acrylic_emulsion",
                "sanitary.brand_level": "premium",
                "sanitary.cp_fittings.brand_level": "premium",
                "doors.main_entrance.type": "teak_solid",
                "doors.internal.type": "flush_laminated",
                "windows.type": "upvc",
                "electrical.brand_level": "premium",
            },
            "luxury": {
                "finishes.flooring.living_dining.type": "marble",
                "finishes.flooring.living_dining.size": "1200x1200",
                "finishes.wall_finishes.internal_paint.type": "texture",
                "sanitary.brand_level": "luxury",
                "sanitary.cp_fittings.brand_level": "luxury",
                "doors.main_entrance.type": "designer",
                "doors.internal.type": "flush_laminated",
                "windows.type": "upvc",
                "electrical.brand_level": "premium",
            },
        }

        return grade_defaults.get(grade.lower(), grade_defaults["standard"])
