"""
Owner Inputs RFI Generator - Generate RFIs for missing mandatory fields.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class OwnerInputRFI:
    """RFI generated from missing owner input."""
    rfi_id: str
    field_path: str
    field_name: str
    priority: str
    question: str
    why_needed: str
    impact: str
    options: List[str] = field(default_factory=list)
    default_assumption: str = ""

    def to_dict(self) -> dict:
        return {
            "rfi_id": self.rfi_id,
            "field_path": self.field_path,
            "field_name": self.field_name,
            "priority": self.priority,
            "question": self.question,
            "why_needed": self.why_needed,
            "impact": self.impact,
            "options": self.options,
            "default_assumption": self.default_assumption,
            "issue_type": "missing_owner_input",
        }


class OwnerInputsRFIGenerator:
    """Generate RFIs for missing mandatory owner inputs."""

    # RFI question templates by field category
    RFI_TEMPLATES = {
        # Project
        "project.name": {
            "question": "Please provide the project name for documentation.",
            "default": "Refer to project as per site address",
        },
        "project.location.city": {
            "question": "Please confirm the city/town where the project is located. This is required for applying correct regional rate factors.",
            "default": "Cannot assume - location directly affects rates",
        },
        "project.location.state": {
            "question": "Please confirm the state where the project is located. This is required for GST registration and regulatory compliance.",
            "default": "Cannot assume - affects tax and compliance",
        },
        "project.type": {
            "question": "Please confirm the project type: residential, commercial, industrial, or institutional. This determines applicable finish specifications.",
            "default": "Assume residential unless building form suggests otherwise",
        },
        "project.completion_months": {
            "question": "Please confirm the expected project duration in months. This is required for calculating preliminary costs (supervision, equipment hire, etc.).",
            "default": "Cannot assume - significantly affects prelims",
        },

        # Finishes
        "finishes.grade": {
            "question": "Please specify the overall finish grade: basic, standard, premium, or luxury. This drives material selection across all packages.",
            "default": "Assume 'standard' grade if not specified",
        },
        "finishes.flooring.living_dining.type": {
            "question": "Please specify flooring material for living/dining areas: vitrified tile, marble, granite, or wooden.",
            "default": "Assume vitrified tile (600x600mm) for standard grade",
        },
        "finishes.flooring.bedrooms.type": {
            "question": "Please specify flooring material for bedrooms.",
            "default": "Assume same as living/dining unless otherwise specified",
        },
        "finishes.flooring.kitchen.type": {
            "question": "Please specify kitchen flooring material. Must be anti-skid.",
            "default": "Assume vitrified anti-skid tile",
        },
        "finishes.flooring.bathrooms.type": {
            "question": "Please specify bathroom flooring material. Must be anti-skid ceramic/vitrified/stone.",
            "default": "Assume ceramic anti-skid tiles",
        },
        "finishes.wall_finishes.internal_paint.type": {
            "question": "Please specify internal wall paint type: acrylic emulsion, oil bound distemper, or texture.",
            "default": "Assume acrylic emulsion for standard grade",
        },
        "finishes.wall_finishes.external_paint.type": {
            "question": "Please specify external wall paint type: acrylic exterior, texture, or elastomeric.",
            "default": "Assume acrylic exterior paint",
        },
        "finishes.wall_finishes.bathroom_dado.height_mm": {
            "question": "Please specify bathroom wall tile dado height: full height, 1200mm, or 2100mm.",
            "default": "Assume 2100mm (7 feet) dado height",
        },
        "finishes.wall_finishes.bathroom_dado.tile_type": {
            "question": "Please specify bathroom wall tile type.",
            "default": "Assume ceramic tiles",
        },
        "finishes.wall_finishes.kitchen_dado.height_mm": {
            "question": "Please specify kitchen dado height: 600mm, 750mm, or platform top only.",
            "default": "Assume 600mm dado above counter",
        },
        "finishes.ceiling.type": {
            "question": "Please specify ceiling finish: plain plaster, POP finish, gypsum false ceiling, or grid ceiling.",
            "default": "Assume plain plaster with POP punning",
        },

        # Waterproofing
        "waterproofing.toilet.system": {
            "question": "Please specify toilet waterproofing system: cementitious coating, integral compound, APP membrane, or liquid membrane.",
            "default": "Assume cementitious coating (Dr. Fixit/equivalent)",
        },
        "waterproofing.terrace.system": {
            "question": "Please specify terrace waterproofing system: brick bat coba, APP membrane, liquid membrane, or heat insulation tiles.",
            "default": "Assume APP membrane with brick bat coba protection",
        },
        "waterproofing.water_tank.system": {
            "question": "Please specify water tank waterproofing system: cementitious coating, epoxy coating, or crystalline.",
            "default": "Assume cementitious coating for potable water",
        },

        # Doors
        "doors.main_entrance.type": {
            "question": "Please specify main entrance door type: teak solid, teak veneer, flush laminated, or designer.",
            "default": "Assume teak veneer for standard grade",
        },
        "doors.main_entrance.hardware_level": {
            "question": "Please specify main door hardware level: basic, standard, or premium.",
            "default": "Assume standard hardware",
        },
        "doors.internal.type": {
            "question": "Please specify internal door type: flush painted, flush laminated, or skin moulded.",
            "default": "Assume flush laminated for standard grade",
        },
        "doors.internal.frame_type": {
            "question": "Please specify door frame material: sal wood, teak, pressed steel, or WPC.",
            "default": "Assume sal wood frames",
        },
        "doors.internal.hardware_level": {
            "question": "Please specify internal door hardware level.",
            "default": "Assume standard hardware",
        },
        "doors.bathroom.type": {
            "question": "Please specify bathroom door type: WPC, PVC, UPVC, or aluminum (must be water resistant).",
            "default": "Assume WPC doors",
        },
        "doors.bathroom.hardware_level": {
            "question": "Please specify bathroom door hardware level.",
            "default": "Assume standard hardware",
        },

        # Windows
        "windows.type": {
            "question": "Please specify window type: aluminum powder coated, UPVC, wood, or aluminum anodized.",
            "default": "Assume aluminum powder coated",
        },
        "windows.glass_type": {
            "question": "Please specify glass type: clear, tinted, reflective, or Saint Gobain equivalent.",
            "default": "Assume clear glass",
        },
        "windows.glass_thickness_mm": {
            "question": "Please specify glass thickness: 5mm, 6mm, or 8mm.",
            "default": "Assume 5mm for residential",
        },
        "windows.grill": {
            "question": "Please specify window grill type: MS grill, SS grill, or none.",
            "default": "Assume MS grill",
        },

        # Sanitary
        "sanitary.brand_level": {
            "question": "Please specify sanitary ware brand level: basic (Hindware), standard (Parryware), premium (Kohler/Duravit), or luxury (owner supply).",
            "default": "Assume standard (Parryware/equivalent)",
        },
        "sanitary.wc_type": {
            "question": "Please specify WC type: EWC, Indian WC, or both.",
            "default": "Assume EWC for residential",
        },
        "sanitary.cp_fittings.brand_level": {
            "question": "Please specify CP fittings brand level: basic, standard, premium, or luxury.",
            "default": "Assume standard level",
        },

        # Plumbing
        "plumbing.water_supply.pipe_type": {
            "question": "Please specify water supply pipe type: CPVC, PPR, or composite.",
            "default": "Assume CPVC (Astral/Supreme equivalent)",
        },
        "plumbing.drainage.pipe_type": {
            "question": "Please specify drainage pipe type: UPVC SWR, HDPE, or CI.",
            "default": "Assume UPVC SWR",
        },

        # Electrical
        "electrical.brand_level": {
            "question": "Please specify electrical fittings brand level: basic (Anchor), standard (Legrand), or premium (Schneider).",
            "default": "Assume standard (Legrand/equivalent)",
        },
        "electrical.switches.type": {
            "question": "Please specify switch type: modular or conventional.",
            "default": "Assume modular switches",
        },
        "electrical.switches.brand_range": {
            "question": "Please specify switch brand range.",
            "default": "Assume Legrand or equivalent",
        },
        "electrical.wiring.type": {
            "question": "Please specify wire type: FR PVC, FRLS, or LSZH.",
            "default": "Assume FR PVC (Polycab/equivalent)",
        },

        # HVAC
        "hvac.required": {
            "question": "Please confirm if HVAC/AC system is in scope: yes or no. If yes, specify type: split AC provision, VRV, or central AC.",
            "default": "Assume only AC point provisions (conduit + drain)",
        },

        # Fire
        "fire.required": {
            "question": "Please confirm if fire fighting system is in scope. Note: Fire system is mandatory for buildings above 15m height as per NBC. If yes, specify type: hydrant, sprinkler, or both.",
            "default": "Check local bylaws - exclude if building under 15m height",
        },

        # Lift
        "lift.required": {
            "question": "Please confirm if lift is in scope. Note: Lift is mandatory for buildings above G+3. If yes, provide count, type, and brand preference.",
            "default": "Check local bylaws for mandatory requirements",
        },

        # Commercial
        "commercial.gst_included": {
            "question": "Please confirm if quoted rates should include GST (18%) or be exclusive of GST.",
            "default": "Assume rates exclusive of GST",
        },
    }

    def __init__(self):
        self.rfi_counter = 0

    def generate(self, missing_mandatory: list) -> List[OwnerInputRFI]:
        """Generate RFIs for missing mandatory fields."""
        rfis = []

        for field in missing_mandatory:
            rfi = self._create_rfi(field)
            rfis.append(rfi)

        # Sort by priority
        priority_order = {"critical": 0, "important": 1, "optional": 2}
        rfis.sort(key=lambda r: priority_order.get(r.priority, 3))

        return rfis

    def _create_rfi(self, field) -> OwnerInputRFI:
        """Create RFI for a missing field."""
        self.rfi_counter += 1
        rfi_id = f"OI-RFI-{self.rfi_counter:04d}"

        # Get template
        template = self.RFI_TEMPLATES.get(field.path, {})

        # Build question
        if template:
            question = template.get("question", "")
            default = template.get("default", "")
        else:
            # Generate generic question
            readable_name = field.path.replace(".", " > ").replace("_", " ").title()
            question = f"Please specify {readable_name}."
            default = "No standard assumption available"

        # Map priority
        priority_map = {
            "critical": "high",
            "important": "medium",
            "optional": "low",
        }
        priority = priority_map.get(field.priority, "medium")

        return OwnerInputRFI(
            rfi_id=rfi_id,
            field_path=field.path,
            field_name=field.field_name,
            priority=priority,
            question=question,
            why_needed=field.why_needed,
            impact=field.impact,
            options=field.options,
            default_assumption=default,
        )

    def generate_summary_rfi(self, missing_mandatory: list) -> dict:
        """Generate a summary RFI document for bulk response."""
        # Group by category
        by_category = {}
        for field in missing_mandatory:
            category = field.path.split(".")[0]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(field)

        summary = {
            "title": "Owner Inputs Required",
            "total_fields": len(missing_mandatory),
            "critical_count": len([f for f in missing_mandatory if f.priority == "critical"]),
            "important_count": len([f for f in missing_mandatory if f.priority == "important"]),
            "categories": {},
        }

        for category, fields in by_category.items():
            summary["categories"][category] = {
                "fields": [
                    {
                        "name": f.field_name,
                        "path": f.path,
                        "options": f.options,
                        "priority": f.priority,
                    }
                    for f in fields
                ],
                "count": len(fields),
            }

        return summary
