"""
Material Aggregator - Aggregate materials across BOQ items.

Provides:
- Material consolidation by type
- Procurement quantity calculation
- Category-wise summaries
"""

from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict

from .calculator import MaterialEstimate, MaterialItem


@dataclass
class AggregatedMaterial:
    """Aggregated material quantity."""
    material_name: str
    quantity: float
    unit: str
    category: str = ""
    source_count: int = 0
    source_items: List[str] = field(default_factory=list)
    procurement_quantity: float = 0.0
    procurement_unit: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "material_name": self.material_name,
            "quantity": self.quantity,
            "unit": self.unit,
            "category": self.category,
            "source_count": self.source_count,
            "source_items": self.source_items[:10],  # Limit for readability
            "procurement_quantity": self.procurement_quantity,
            "procurement_unit": self.procurement_unit,
            "notes": self.notes,
        }


class MaterialAggregator:
    """Aggregate materials from multiple estimates."""

    # Material categories
    CATEGORIES = {
        "cement": ["cement", "opc", "ppc", "psc"],
        "aggregates": ["sand", "aggregate", "grit", "bajri", "coarse"],
        "steel": ["steel", "tmt", "bar", "rebar", "reinforcement", "binding wire"],
        "bricks_blocks": ["brick", "block", "aac", "fly ash brick", "cement block"],
        "tiles": ["tile", "vitrified", "ceramic", "anti-skid"],
        "stone": ["granite", "marble", "kota", "shahabad", "tandur"],
        "paint": ["paint", "primer", "emulsion", "enamel", "putty", "distemper"],
        "waterproofing": ["waterproof", "membrane", "bitumen", "coating"],
        "plumbing": ["pipe", "cpvc", "upvc", "fitting", "valve"],
        "electrical": ["wire", "cable", "conduit", "switch", "socket"],
        "wood": ["wood", "plywood", "sal", "teak", "flush door"],
        "hardware": ["hinge", "lock", "handle", "screw", "nail", "bolt"],
        "adhesive": ["adhesive", "grout", "sealant", "bonding", "joint compound"],
        "ceiling": ["gypsum", "board", "grid", "channel", "hanger"],
        "misc": [],  # Default category
    }

    # Procurement conversions
    PROCUREMENT_CONVERSIONS = {
        ("cement", "bags"): (50, "kg", 50),  # 1 bag = 50 kg
        ("sand", "cum"): (1.5, "MT", 1.5),   # 1 cum ≈ 1.5 MT
        ("aggregate", "cum"): (1.6, "MT", 1.6),
        ("steel", "kg"): (1000, "MT", 0.001),
        ("brick", "nos"): (1000, "thousand", 0.001),
        ("tile", "sqm"): (1, "sqm", 1),
        ("paint", "liters"): (1, "liters", 1),
    }

    def __init__(self):
        pass

    def aggregate(
        self, estimates: List[MaterialEstimate]
    ) -> List[AggregatedMaterial]:
        """Aggregate materials from all estimates."""
        # Group materials by normalized name and unit
        grouped: Dict[tuple, Dict] = defaultdict(
            lambda: {"quantity": 0.0, "sources": []}
        )

        for estimate in estimates:
            for material in estimate.materials:
                key = self._normalize_key(material.material_name, material.unit)
                grouped[key]["quantity"] += material.quantity
                if estimate.boq_item_code:
                    grouped[key]["sources"].append(estimate.boq_item_code)

        # Convert to aggregated materials
        aggregated = []
        for (name, unit), data in grouped.items():
            category = self._categorize(name)
            procurement = self._calc_procurement(name, data["quantity"], unit)

            aggregated.append(AggregatedMaterial(
                material_name=name,
                quantity=round(data["quantity"], 2),
                unit=unit,
                category=category,
                source_count=len(data["sources"]),
                source_items=list(set(data["sources"])),
                procurement_quantity=round(procurement[0], 2),
                procurement_unit=procurement[1],
            ))

        # Sort by category and quantity
        aggregated.sort(key=lambda m: (m.category, -m.quantity))

        return aggregated

    def _normalize_key(self, name: str, unit: str) -> tuple:
        """Normalize material name and unit for grouping."""
        # Normalize name
        name = name.lower().strip()
        name = name.replace("_", " ")

        # Normalize common variations
        normalizations = {
            "aggregate 20mm": "coarse aggregate 20mm",
            "aggregate 40mm": "coarse aggregate 40mm",
            "aggregate20mm": "coarse aggregate 20mm",
            "sand bedding": "sand",
            "fine aggregate": "sand",
        }

        for original, normalized in normalizations.items():
            if original in name:
                name = normalized
                break

        # Title case
        name = name.title()

        # Normalize unit
        unit = unit.lower().strip()
        unit_normalizations = {
            "bag": "bags",
            "no": "nos",
            "number": "nos",
            "sqm": "sqm",
            "sq.m": "sqm",
            "cum": "cum",
            "cu.m": "cum",
            "rmt": "rmt",
            "rm": "rmt",
            "mt": "MT",
            "kg": "kg",
            "liter": "liters",
            "litre": "liters",
            "l": "liters",
        }

        unit = unit_normalizations.get(unit, unit)

        return (name, unit)

    def _categorize(self, name: str) -> str:
        """Categorize material."""
        name_lower = name.lower()

        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category

        return "misc"

    def _calc_procurement(
        self, name: str, quantity: float, unit: str
    ) -> tuple:
        """Calculate procurement quantity."""
        name_lower = name.lower()

        # Check for direct conversions
        for (material_key, unit_key), (divisor, proc_unit, multiplier) in self.PROCUREMENT_CONVERSIONS.items():
            if material_key in name_lower and unit_key == unit.lower():
                proc_qty = quantity * multiplier
                return (proc_qty, proc_unit)

        # Default: return as-is
        return (quantity, unit)

    def get_category_summary(
        self, aggregated: List[AggregatedMaterial]
    ) -> Dict[str, Dict]:
        """Get summary by category."""
        summary = defaultdict(lambda: {"items": 0, "materials": []})

        for material in aggregated:
            cat = material.category
            summary[cat]["items"] += 1
            summary[cat]["materials"].append({
                "name": material.material_name,
                "quantity": material.quantity,
                "unit": material.unit,
            })

        return dict(summary)

    def get_major_materials(
        self, aggregated: List[AggregatedMaterial]
    ) -> Dict[str, float]:
        """Get major material totals."""
        totals = {
            "cement_bags": 0.0,
            "sand_cum": 0.0,
            "aggregate_cum": 0.0,
            "steel_kg": 0.0,
            "bricks_nos": 0.0,
            "tiles_sqm": 0.0,
            "paint_liters": 0.0,
        }

        for material in aggregated:
            name_lower = material.material_name.lower()
            unit = material.unit.lower()

            if "cement" in name_lower and unit == "bags":
                totals["cement_bags"] += material.quantity
            elif "sand" in name_lower and unit == "cum":
                totals["sand_cum"] += material.quantity
            elif "aggregate" in name_lower and unit == "cum":
                totals["aggregate_cum"] += material.quantity
            elif "steel" in name_lower and unit == "kg":
                totals["steel_kg"] += material.quantity
            elif "brick" in name_lower and unit == "nos":
                totals["bricks_nos"] += material.quantity
            elif "tile" in name_lower and unit == "sqm":
                totals["tiles_sqm"] += material.quantity
            elif "paint" in name_lower and unit == "liters":
                totals["paint_liters"] += material.quantity

        return {k: round(v, 2) for k, v in totals.items()}
