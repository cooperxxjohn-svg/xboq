"""
Rate Builder - Build rates from material + labor + equipment + overhead.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re


@dataclass
class RateBuildUp:
    """Complete rate build-up for a BOQ item."""
    item_id: str
    description: str
    unit: str
    components: List[Dict] = field(default_factory=list)
    subtotal: float = 0.0
    overhead_percent: float = 0.0
    overheads: float = 0.0
    rate: float = 0.0
    source: str = ""
    confidence: float = 0.0
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "description": self.description,
            "unit": self.unit,
            "components": self.components,
            "subtotal": self.subtotal,
            "overhead_percent": self.overhead_percent,
            "overheads": self.overheads,
            "rate": self.rate,
            "source": self.source,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
        }


class RateBuilder:
    """Build rates for BOQ items from components."""

    # Standard overhead percentages
    OVERHEAD_RATES = {
        "contractor_profit": 10.0,  # 10% profit
        "site_overhead": 5.0,  # 5% site overhead
        "head_office": 2.5,  # 2.5% head office overhead
    }

    # Default overhead by package
    PACKAGE_OVERHEADS = {
        "civil_structural": 15.0,
        "masonry": 15.0,
        "plaster_finishes": 12.0,
        "flooring": 12.0,
        "waterproofing": 15.0,
        "doors_windows": 10.0,
        "plumbing": 12.0,
        "electrical": 12.0,
        "external_works": 15.0,
        "miscellaneous": 15.0,
    }

    # Rate templates for common items (per unit)
    RATE_TEMPLATES = {
        # RCC works (per cum)
        "rcc_m20": {
            "materials": [
                {"item": "cement", "qty": 8.22, "unit": "bag"},
                {"item": "sand", "qty": 0.44, "unit": "cum"},
                {"item": "aggregate_20mm", "qty": 0.88, "unit": "cum"},
            ],
            "labor": [
                {"item": "mason", "qty": 0.5, "unit": "day"},
                {"item": "helper", "qty": 2.0, "unit": "day"},
            ],
            "equipment": [
                {"item": "mixer", "qty": 0.5, "unit": "day"},
                {"item": "vibrator", "qty": 0.5, "unit": "day"},
            ],
        },
        "rcc_m25": {
            "materials": [
                {"item": "cement", "qty": 9.51, "unit": "bag"},
                {"item": "sand", "qty": 0.42, "unit": "cum"},
                {"item": "aggregate_20mm", "qty": 0.84, "unit": "cum"},
            ],
            "labor": [
                {"item": "mason", "qty": 0.5, "unit": "day"},
                {"item": "helper", "qty": 2.0, "unit": "day"},
            ],
            "equipment": [
                {"item": "mixer", "qty": 0.5, "unit": "day"},
                {"item": "vibrator", "qty": 0.5, "unit": "day"},
            ],
        },

        # Brickwork (per cum)
        "brick_230mm": {
            "materials": [
                {"item": "brick", "qty": 500, "unit": "nos"},
                {"item": "cement", "qty": 1.66, "unit": "bag"},
                {"item": "sand", "qty": 0.28, "unit": "cum"},
            ],
            "labor": [
                {"item": "mason", "qty": 1.5, "unit": "day"},
                {"item": "helper", "qty": 1.5, "unit": "day"},
            ],
        },
        "brick_115mm": {
            "materials": [
                {"item": "brick", "qty": 450, "unit": "nos"},
                {"item": "cement", "qty": 1.0, "unit": "bag"},
                {"item": "sand", "qty": 0.17, "unit": "cum"},
            ],
            "labor": [
                {"item": "mason", "qty": 1.8, "unit": "day"},
                {"item": "helper", "qty": 1.8, "unit": "day"},
            ],
        },

        # Plastering (per sqm)
        "plaster_12mm": {
            "materials": [
                {"item": "cement", "qty": 0.07, "unit": "bag"},
                {"item": "sand", "qty": 0.014, "unit": "cum"},
            ],
            "labor": [
                {"item": "mason", "qty": 0.08, "unit": "day"},
                {"item": "helper", "qty": 0.08, "unit": "day"},
            ],
        },
        "plaster_20mm": {
            "materials": [
                {"item": "cement", "qty": 0.10, "unit": "bag"},
                {"item": "sand", "qty": 0.02, "unit": "cum"},
            ],
            "labor": [
                {"item": "mason", "qty": 0.10, "unit": "day"},
                {"item": "helper", "qty": 0.10, "unit": "day"},
            ],
        },

        # Flooring (per sqm)
        "vitrified_600x600": {
            "materials": [
                {"item": "vitrified_tile_600x600", "qty": 1.05, "unit": "sqm"},
                {"item": "cement", "qty": 0.12, "unit": "bag"},
                {"item": "sand", "qty": 0.03, "unit": "cum"},
                {"item": "tile_adhesive", "qty": 0.5, "unit": "kg"},
            ],
            "labor": [
                {"item": "tile_mason", "qty": 0.15, "unit": "day"},
                {"item": "helper", "qty": 0.15, "unit": "day"},
            ],
        },
        "ceramic_tile": {
            "materials": [
                {"item": "ceramic_tile", "qty": 1.05, "unit": "sqm"},
                {"item": "cement", "qty": 0.12, "unit": "bag"},
                {"item": "sand", "qty": 0.03, "unit": "cum"},
            ],
            "labor": [
                {"item": "tile_mason", "qty": 0.12, "unit": "day"},
                {"item": "helper", "qty": 0.12, "unit": "day"},
            ],
        },

        # Painting (per sqm)
        "acrylic_emulsion": {
            "materials": [
                {"item": "wall_putty", "qty": 1.0, "unit": "kg"},
                {"item": "primer", "qty": 0.1, "unit": "ltr"},
                {"item": "acrylic_emulsion", "qty": 0.25, "unit": "ltr"},
            ],
            "labor": [
                {"item": "painter", "qty": 0.05, "unit": "day"},
                {"item": "helper", "qty": 0.03, "unit": "day"},
            ],
        },
        "exterior_paint": {
            "materials": [
                {"item": "cement_primer", "qty": 0.1, "unit": "ltr"},
                {"item": "exterior_emulsion", "qty": 0.3, "unit": "ltr"},
            ],
            "labor": [
                {"item": "painter", "qty": 0.06, "unit": "day"},
                {"item": "helper", "qty": 0.04, "unit": "day"},
            ],
        },

        # Waterproofing (per sqm)
        "wc_waterproofing": {
            "materials": [
                {"item": "cement", "qty": 0.05, "unit": "bag"},
                {"item": "waterproofing_compound", "qty": 0.5, "unit": "kg"},
            ],
            "labor": [
                {"item": "mason", "qty": 0.08, "unit": "day"},
                {"item": "helper", "qty": 0.08, "unit": "day"},
            ],
        },
        "terrace_waterproofing": {
            "materials": [
                {"item": "app_membrane", "qty": 1.1, "unit": "sqm"},
                {"item": "primer", "qty": 0.2, "unit": "ltr"},
                {"item": "brick_bat", "qty": 0.06, "unit": "cum"},
                {"item": "cement", "qty": 0.15, "unit": "bag"},
            ],
            "labor": [
                {"item": "waterproofing_applicator", "qty": 0.15, "unit": "day"},
                {"item": "helper", "qty": 0.20, "unit": "day"},
            ],
        },

        # Doors (per sqm)
        "flush_door": {
            "materials": [
                {"item": "flush_door_shutter", "qty": 1.0, "unit": "sqm"},
                {"item": "door_frame_sal", "qty": 0.05, "unit": "cum"},
                {"item": "hardware_set", "qty": 1.0, "unit": "set"},
            ],
            "labor": [
                {"item": "carpenter", "qty": 0.5, "unit": "day"},
                {"item": "helper", "qty": 0.25, "unit": "day"},
            ],
        },

        # Windows (per sqm)
        "aluminum_window": {
            "materials": [
                {"item": "aluminum_section", "qty": 5.0, "unit": "kg"},
                {"item": "glass_5mm", "qty": 1.05, "unit": "sqm"},
                {"item": "hardware", "qty": 1.0, "unit": "set"},
            ],
            "labor": [
                {"item": "aluminum_fabricator", "qty": 0.3, "unit": "day"},
                {"item": "helper", "qty": 0.15, "unit": "day"},
            ],
        },

        # Plumbing (per point)
        "plumbing_point": {
            "materials": [
                {"item": "cpvc_pipe_15mm", "qty": 3.0, "unit": "rmt"},
                {"item": "cpvc_fittings", "qty": 1.0, "unit": "set"},
            ],
            "labor": [
                {"item": "plumber", "qty": 0.25, "unit": "day"},
                {"item": "helper", "qty": 0.25, "unit": "day"},
            ],
        },

        # Electrical (per point)
        "electrical_point": {
            "materials": [
                {"item": "wire_1.5sqmm", "qty": 12.0, "unit": "rmt"},
                {"item": "conduit_20mm", "qty": 3.0, "unit": "rmt"},
                {"item": "switch_socket", "qty": 1.0, "unit": "nos"},
                {"item": "junction_box", "qty": 1.0, "unit": "nos"},
            ],
            "labor": [
                {"item": "electrician", "qty": 0.2, "unit": "day"},
                {"item": "helper", "qty": 0.15, "unit": "day"},
            ],
        },
    }

    def __init__(
        self,
        material_book,
        labor_book,
        location_factor: float = 1.0,
        finish_grade: str = "standard",
    ):
        self.material_book = material_book
        self.labor_book = labor_book
        self.location_factor = location_factor
        self.finish_grade = finish_grade

    def build_rate(self, item: Dict) -> RateBuildUp:
        """Build rate for a BOQ item."""
        description = item.get("description", "")
        unit = item.get("unit", "")
        package = item.get("package", "miscellaneous")

        # Try to match a rate template
        template_key = self._match_template(description, unit)

        if template_key and template_key in self.RATE_TEMPLATES:
            return self._build_from_template(item, template_key)
        else:
            # Try to estimate from database
            return self._build_from_database(item)

    def _match_template(self, description: str, unit: str) -> Optional[str]:
        """Match description to a rate template."""
        desc_lower = description.lower()

        # RCC matching
        if "rcc" in desc_lower or "reinforced" in desc_lower:
            if "m25" in desc_lower or "m-25" in desc_lower:
                return "rcc_m25"
            elif "m20" in desc_lower or "m-20" in desc_lower:
                return "rcc_m20"
            elif "m30" in desc_lower:
                return "rcc_m25"  # Use M25 as proxy

        # Brickwork matching
        if "brick" in desc_lower:
            if "230" in desc_lower or "9 inch" in desc_lower or "9inch" in desc_lower:
                return "brick_230mm"
            elif "115" in desc_lower or "4.5" in desc_lower or "half" in desc_lower:
                return "brick_115mm"
            return "brick_115mm"  # Default

        # Plastering matching
        if "plaster" in desc_lower:
            if "20mm" in desc_lower or "20 mm" in desc_lower:
                return "plaster_20mm"
            elif "12mm" in desc_lower or "12 mm" in desc_lower:
                return "plaster_12mm"
            return "plaster_12mm"  # Default

        # Flooring matching
        if "vitrified" in desc_lower or "vt" in desc_lower:
            return "vitrified_600x600"
        if "ceramic" in desc_lower:
            return "ceramic_tile"

        # Painting matching
        if "emulsion" in desc_lower or "paint" in desc_lower:
            if "exterior" in desc_lower or "external" in desc_lower:
                return "exterior_paint"
            return "acrylic_emulsion"

        # Waterproofing matching
        if "waterproof" in desc_lower or "wp" in desc_lower:
            if "toilet" in desc_lower or "bathroom" in desc_lower or "wc" in desc_lower:
                return "wc_waterproofing"
            if "terrace" in desc_lower or "roof" in desc_lower:
                return "terrace_waterproofing"

        # Door matching
        if "door" in desc_lower and "flush" in desc_lower:
            return "flush_door"

        # Window matching
        if "window" in desc_lower and ("aluminum" in desc_lower or "aluminium" in desc_lower):
            return "aluminum_window"

        # Plumbing matching
        if "plumbing" in desc_lower or "point" in desc_lower:
            if "cpvc" in desc_lower or "water" in desc_lower:
                return "plumbing_point"

        # Electrical matching
        if "electrical" in desc_lower or "wiring" in desc_lower:
            return "electrical_point"

        return None

    def _build_from_template(self, item: Dict, template_key: str) -> RateBuildUp:
        """Build rate from a template."""
        template = self.RATE_TEMPLATES[template_key]
        package = item.get("package", "miscellaneous")

        components = []
        subtotal = 0.0
        assumptions = []

        # Process materials
        if "materials" in template:
            for mat in template["materials"]:
                mat_price = self.material_book.get_price(
                    mat["item"],
                    grade=self.finish_grade,
                )
                mat_price *= self.location_factor

                amount = mat["qty"] * mat_price

                components.append({
                    "type": "material",
                    "description": mat["item"].replace("_", " ").title(),
                    "quantity": mat["qty"],
                    "unit": mat["unit"],
                    "rate": mat_price,
                    "amount": round(amount, 2),
                })
                subtotal += amount

        # Process labor
        if "labor" in template:
            for lab in template["labor"]:
                lab_rate = self.labor_book.get_rate(lab["item"])
                lab_rate *= self.location_factor

                amount = lab["qty"] * lab_rate

                components.append({
                    "type": "labor",
                    "description": lab["item"].replace("_", " ").title(),
                    "quantity": lab["qty"],
                    "unit": lab["unit"],
                    "rate": lab_rate,
                    "amount": round(amount, 2),
                })
                subtotal += amount

        # Process equipment
        if "equipment" in template:
            for equip in template["equipment"]:
                equip_rate = self._get_equipment_rate(equip["item"])
                equip_rate *= self.location_factor

                amount = equip["qty"] * equip_rate

                components.append({
                    "type": "equipment",
                    "description": equip["item"].replace("_", " ").title(),
                    "quantity": equip["qty"],
                    "unit": equip["unit"],
                    "rate": equip_rate,
                    "amount": round(amount, 2),
                })
                subtotal += amount

        # Apply overheads
        overhead_percent = self.PACKAGE_OVERHEADS.get(package, 15.0)
        overheads = subtotal * (overhead_percent / 100)
        total_rate = subtotal + overheads

        assumptions.append(f"Rate built from template: {template_key}")
        assumptions.append(f"Location factor: {self.location_factor}")
        assumptions.append(f"Finish grade: {self.finish_grade}")

        return RateBuildUp(
            item_id=item.get("unified_item_no", ""),
            description=item.get("description", ""),
            unit=item.get("unit", ""),
            components=components,
            subtotal=round(subtotal, 2),
            overhead_percent=overhead_percent,
            overheads=round(overheads, 2),
            rate=round(total_rate, 2),
            source=f"Template: {template_key}",
            confidence=0.85,
            assumptions=assumptions,
        )

    def _build_from_database(self, item: Dict) -> RateBuildUp:
        """Build rate from database lookup."""
        description = item.get("description", "")
        unit = item.get("unit", "")
        package = item.get("package", "miscellaneous")

        # Try to get a base rate from material book
        base_rate = self.material_book.get_composite_rate(description, unit)

        if base_rate > 0:
            base_rate *= self.location_factor

            # Apply grade factor
            grade_factors = {"basic": 0.8, "standard": 1.0, "premium": 1.3, "luxury": 1.8}
            grade_factor = grade_factors.get(self.finish_grade, 1.0)
            base_rate *= grade_factor

            # Apply overhead
            overhead_percent = self.PACKAGE_OVERHEADS.get(package, 15.0)
            overheads = base_rate * (overhead_percent / 100)
            total_rate = base_rate + overheads

            return RateBuildUp(
                item_id=item.get("unified_item_no", ""),
                description=description,
                unit=unit,
                components=[{
                    "type": "composite",
                    "description": "Composite rate from database",
                    "quantity": 1.0,
                    "unit": unit,
                    "rate": base_rate,
                    "amount": base_rate,
                }],
                subtotal=round(base_rate, 2),
                overhead_percent=overhead_percent,
                overheads=round(overheads, 2),
                rate=round(total_rate, 2),
                source="Database lookup",
                confidence=0.7,
                assumptions=[
                    f"Composite rate from database",
                    f"Grade factor: {grade_factor} ({self.finish_grade})",
                    f"Location factor: {self.location_factor}",
                ],
            )

        # No rate found
        return RateBuildUp(
            item_id=item.get("unified_item_no", ""),
            description=description,
            unit=unit,
            components=[],
            subtotal=0,
            overhead_percent=0,
            overheads=0,
            rate=0,
            source="NOT_FOUND",
            confidence=0,
            assumptions=["No matching rate template or database entry"],
        )

    def _get_equipment_rate(self, equipment: str) -> float:
        """Get equipment rental rate per day."""
        equipment_rates = {
            "mixer": 1500,
            "vibrator": 800,
            "scaffolding": 50,  # per sqm per day
            "crane": 15000,
            "hoist": 3000,
            "pump": 2500,
        }
        return equipment_rates.get(equipment, 1000)
