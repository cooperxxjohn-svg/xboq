"""
Material Calculator - Calculate material quantities from BOQ items.

Uses CPWD/DSR norms for Indian construction.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class MaterialItem:
    """Single material requirement."""
    material_name: str
    quantity: float
    unit: str
    source_boq_item: str = ""
    norm_reference: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "material_name": self.material_name,
            "quantity": self.quantity,
            "unit": self.unit,
            "source_boq_item": self.source_boq_item,
            "norm_reference": self.norm_reference,
            "notes": self.notes,
        }


@dataclass
class MaterialEstimate:
    """Material estimate for a BOQ item."""
    boq_item_code: str
    boq_description: str
    boq_quantity: float
    boq_unit: str
    work_type: str
    materials: List[MaterialItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "boq_item_code": self.boq_item_code,
            "boq_description": self.boq_description,
            "boq_quantity": self.boq_quantity,
            "boq_unit": self.boq_unit,
            "work_type": self.work_type,
            "materials": [m.to_dict() for m in self.materials],
        }


class MaterialCalculator:
    """Calculate material requirements from BOQ items."""

    # Work type detection patterns
    WORK_TYPE_PATTERNS = {
        "rcc": [
            r'rcc\s+(?:m\d+|grade)',
            r'reinforced\s+cement\s+concrete',
            r'rcc\s+(?:slab|beam|column|footing|wall)',
            r'concrete\s+(?:m\d+)',
        ],
        "pcc": [
            r'pcc\s+\d+:\d+:\d+',
            r'plain\s+cement\s+concrete',
            r'lean\s+concrete',
            r'pcc\s+bed',
        ],
        "masonry_brick": [
            r'brick\s*work',
            r'brick\s+masonry',
            r'\d+mm\s+(?:thick\s+)?brick',
            r'half\s+brick',
        ],
        "masonry_aac": [
            r'aac\s+block',
            r'autoclaved\s+aerated',
            r'siporex',
            r'celcon\s+block',
        ],
        "plaster": [
            r'cement\s+plaster',
            r'cm\s+plaster',
            r'\d+mm\s+plaster',
            r'sand\s+cement\s+plaster',
            r'internal\s+plaster',
            r'external\s+plaster',
        ],
        "flooring_tile": [
            r'vitrified\s+tile',
            r'ceramic\s+tile',
            r'floor\s+tile',
            r'anti[- ]?skid\s+tile',
        ],
        "flooring_stone": [
            r'granite\s+floor',
            r'marble\s+floor',
            r'kota\s+stone',
            r'stone\s+floor',
        ],
        "wall_tile": [
            r'wall\s+tile',
            r'dado\s+tile',
            r'kitchen\s+tile',
            r'bathroom\s+tile',
        ],
        "painting": [
            r'paint',
            r'emulsion',
            r'distemper',
            r'enamel',
            r'primer',
            r'putty',
        ],
        "waterproofing": [
            r'waterproof',
            r'membrane',
            r'damp\s+proof',
            r'water\s+tank\s+treatment',
        ],
        "steel": [
            r'reinforcement',
            r'steel\s+bar',
            r'tmt\s+bar',
            r'rebar',
            r'fe\s*\d+',
        ],
        "formwork": [
            r'formwork',
            r'shuttering',
            r'centering',
        ],
        "ceiling": [
            r'false\s+ceiling',
            r'gypsum\s+ceiling',
            r'pop\s+ceiling',
            r'grid\s+ceiling',
        ],
        "external_paving": [
            r'paver',
            r'interlocking',
            r'paving\s+block',
        ],
    }

    def __init__(self, norms_path: Optional[Path] = None):
        """Initialize calculator with material norms."""
        self.norms = self._load_norms(norms_path)
        self.compiled_patterns = {}
        for work_type, patterns in self.WORK_TYPE_PATTERNS.items():
            self.compiled_patterns[work_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def _load_norms(self, norms_path: Optional[Path]) -> Dict:
        """Load material norms from YAML."""
        if norms_path is None:
            # Try default location
            default_path = Path(__file__).parent.parent.parent / "rules" / "material_norms.yaml"
            if default_path.exists():
                norms_path = default_path

        if norms_path and norms_path.exists():
            with open(norms_path) as f:
                return yaml.safe_load(f)

        # Return minimal defaults if no norms file
        return self._get_default_norms()

    def _get_default_norms(self) -> Dict:
        """Get minimal default norms."""
        return {
            "rcc": {
                "M20": {
                    "cement": {"quantity": 8.22, "unit": "bags"},
                    "sand": {"quantity": 0.45, "unit": "cum"},
                    "aggregate_20mm": {"quantity": 0.90, "unit": "cum"},
                }
            },
            "masonry": {
                "brick_230mm": {
                    "bricks": {"quantity": 460, "unit": "nos"},
                    "cement": {"quantity": 1.26, "unit": "bags"},
                    "sand": {"quantity": 0.25, "unit": "cum"},
                }
            },
            "plaster": {
                "cm_12mm": {
                    "cement": {"quantity": 0.12, "unit": "bags"},
                    "sand": {"quantity": 0.012, "unit": "cum"},
                }
            },
        }

    def calculate_all(self, boq_entries: List[Dict]) -> List[MaterialEstimate]:
        """Calculate materials for all BOQ entries."""
        estimates = []

        for entry in boq_entries:
            estimate = self.calculate_for_entry(entry)
            if estimate and estimate.materials:
                estimates.append(estimate)

        return estimates

    def calculate_for_entry(self, entry: Dict) -> Optional[MaterialEstimate]:
        """Calculate materials for a single BOQ entry."""
        description = entry.get("description", "")
        quantity = entry.get("quantity", 0)
        unit = entry.get("unit", "")
        item_code = entry.get("item_code", "")

        if not quantity or quantity <= 0:
            return None

        # Detect work type
        work_type = self._detect_work_type(description)

        if not work_type:
            return None

        # Get material norms for work type
        materials = self._calculate_materials(
            work_type, description, quantity, unit, item_code
        )

        return MaterialEstimate(
            boq_item_code=item_code,
            boq_description=description,
            boq_quantity=quantity,
            boq_unit=unit,
            work_type=work_type,
            materials=materials,
        )

    def _detect_work_type(self, description: str) -> Optional[str]:
        """Detect work type from description."""
        desc_lower = description.lower()

        for work_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(desc_lower):
                    return work_type

        return None

    def _calculate_materials(
        self,
        work_type: str,
        description: str,
        quantity: float,
        unit: str,
        item_code: str,
    ) -> List[MaterialItem]:
        """Calculate materials based on work type."""
        materials = []

        if work_type == "rcc":
            materials = self._calc_rcc(description, quantity)
        elif work_type == "pcc":
            materials = self._calc_pcc(description, quantity)
        elif work_type == "masonry_brick":
            materials = self._calc_brick_masonry(description, quantity)
        elif work_type == "masonry_aac":
            materials = self._calc_aac_masonry(description, quantity)
        elif work_type == "plaster":
            materials = self._calc_plaster(description, quantity)
        elif work_type in ["flooring_tile", "wall_tile"]:
            materials = self._calc_tiles(description, quantity)
        elif work_type == "flooring_stone":
            materials = self._calc_stone(description, quantity)
        elif work_type == "painting":
            materials = self._calc_painting(description, quantity)
        elif work_type == "waterproofing":
            materials = self._calc_waterproofing(description, quantity)
        elif work_type == "steel":
            materials = self._calc_steel(description, quantity)
        elif work_type == "formwork":
            materials = self._calc_formwork(description, quantity)
        elif work_type == "ceiling":
            materials = self._calc_ceiling(description, quantity)
        elif work_type == "external_paving":
            materials = self._calc_paving(description, quantity)

        # Add source reference
        for m in materials:
            m.source_boq_item = item_code

        return materials

    def _calc_rcc(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate RCC materials."""
        # Detect grade
        grade = "M20"  # Default
        grade_match = re.search(r'm\s*(\d+)', description, re.IGNORECASE)
        if grade_match:
            grade = f"M{grade_match.group(1)}"

        norms = self.norms.get("rcc", {}).get(grade, self.norms.get("rcc", {}).get("M20", {}))

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD RCC {grade}",
                    notes=data.get("notes", ""),
                ))

        return materials

    def _calc_pcc(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate PCC materials."""
        # Detect mix
        mix = "1:4:8"  # Default
        mix_match = re.search(r'(\d+:\d+:\d+)', description)
        if mix_match:
            mix = mix_match.group(1)

        norms = self.norms.get("pcc", {}).get(mix, self.norms.get("pcc", {}).get("1:4:8", {}))

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD PCC {mix}",
                ))

        return materials

    def _calc_brick_masonry(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate brick masonry materials."""
        # Detect thickness
        norm_key = "brick_230mm"  # Default
        if "115" in description or "half" in description.lower():
            norm_key = "brick_115mm"

        norms = self.norms.get("masonry", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 0),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD Brickwork {norm_key}",
                ))

        return materials

    def _calc_aac_masonry(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate AAC block masonry materials."""
        # Detect thickness
        norm_key = "aac_200mm"  # Default
        if "150" in description:
            norm_key = "aac_150mm"
        elif "100" in description:
            norm_key = "aac_100mm"

        norms = self.norms.get("masonry", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD AAC {norm_key}",
                ))

        return materials

    def _calc_plaster(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate plaster materials."""
        # Detect thickness
        norm_key = "cm_12mm"  # Default
        if "20" in description or "20mm" in description.lower():
            norm_key = "cm_20mm"
        elif "15" in description or "15mm" in description.lower():
            norm_key = "cm_15mm"

        norms = self.norms.get("plaster", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD Plaster {norm_key}",
                ))

        return materials

    def _calc_tiles(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate tile flooring/wall materials."""
        # Detect tile type
        norm_key = "vitrified_tiles_600x600"  # Default
        if "ceramic" in description.lower() or "300" in description:
            norm_key = "ceramic_tiles_300x300"

        norms = self.norms.get("flooring", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD {norm_key}",
                ))

        return materials

    def _calc_stone(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate stone flooring materials."""
        # Detect stone type
        norm_key = "granite_flooring"  # Default
        if "marble" in description.lower():
            norm_key = "marble_flooring"
        elif "kota" in description.lower():
            norm_key = "kota_stone"

        norms = self.norms.get("flooring", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD {norm_key}",
                ))

        return materials

    def _calc_painting(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate painting materials."""
        # Detect paint type
        norm_key = "acrylic_emulsion"  # Default
        if "exterior" in description.lower() or "external" in description.lower():
            norm_key = "exterior_emulsion"
        elif "enamel" in description.lower():
            norm_key = "enamel_paint"
        elif "distemper" in description.lower():
            norm_key = "oil_bound_distemper"

        norms = self.norms.get("painting", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD {norm_key}",
                ))

        return materials

    def _calc_waterproofing(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate waterproofing materials."""
        # Detect waterproofing type
        norm_key = "cementitious_coating"  # Default
        if "membrane" in description.lower() or "app" in description.lower():
            norm_key = "app_membrane_3mm"
        elif "liquid" in description.lower():
            norm_key = "liquid_membrane"

        norms = self.norms.get("waterproofing", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD {norm_key}",
                ))

        return materials

    def _calc_steel(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate steel reinforcement materials."""
        norms = self.norms.get("steel", {})

        materials = [
            MaterialItem(
                material_name="TMT Steel Bar",
                quantity=round(quantity * 1.03, 2),  # 3% wastage
                unit="kg",
                norm_reference="CPWD Steel",
                notes="Including 3% cutting wastage",
            ),
        ]

        binding_wire = norms.get("binding_wire", {})
        if binding_wire:
            materials.append(MaterialItem(
                material_name="Binding Wire",
                quantity=round(binding_wire.get("quantity", 12) * quantity / 1000, 2),
                unit="kg",
                norm_reference="CPWD Steel",
            ))

        return materials

    def _calc_formwork(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate formwork materials."""
        norms = self.norms.get("formwork", {}).get("plywood_formwork", {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference="CPWD Formwork",
                ))

        return materials

    def _calc_ceiling(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate false ceiling materials."""
        norm_key = "gypsum_board_false_ceiling"
        if "grid" in description.lower():
            norm_key = "grid_ceiling"

        norms = self.norms.get("ceiling", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD {norm_key}",
                ))

        return materials

    def _calc_paving(self, description: str, quantity: float) -> List[MaterialItem]:
        """Calculate external paving materials."""
        norm_key = "paver_blocks_60mm"
        if "80" in description:
            norm_key = "interlock_tiles_80mm"

        norms = self.norms.get("external", {}).get(norm_key, {})

        materials = []
        for material_name, data in norms.items():
            if isinstance(data, dict) and "quantity" in data:
                materials.append(MaterialItem(
                    material_name=material_name.replace("_", " ").title(),
                    quantity=round(data["quantity"] * quantity, 2),
                    unit=data.get("unit", ""),
                    norm_reference=f"CPWD {norm_key}",
                ))

        return materials
