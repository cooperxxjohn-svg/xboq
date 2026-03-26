"""
Material Wastage Calculator — IS-standard wastage norms for Indian construction.

Adds wastage percentages per material type to convert net quantities to gross
(procurement) quantities. Based on CPWD/DSR norms and industry practice.

Wastage covers:
- Handling loss (transport, storage, stacking)
- Cutting/shaping waste (rebar cuts, tile cuts)
- Mixing/application waste (mortar drops, paint overspray)
- Breakage (bricks, tiles, pipes)
- Site conditions (weather, ground, spillage)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# WASTAGE NORMS BY MATERIAL (% of net quantity)
# =============================================================================

# Based on CPWD norms + industry benchmarks (India, 2024)
# Format: material_key -> {standard: %, careful: %, rough: %}
# standard = normal site, careful = well-managed, rough = poor site conditions

WASTAGE_NORMS: Dict[str, Dict[str, float]] = {
    # ── Cement & Concrete ──
    "cement": {"careful": 1.5, "standard": 2.5, "rough": 4.0},
    "sand": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "aggregate_20mm": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "aggregate_10mm": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "ready_mix_concrete": {"careful": 1.0, "standard": 2.0, "rough": 3.0},

    # ── Steel ──
    "steel_rebar": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "steel_structural": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "binding_wire": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "ms_plate": {"careful": 3.0, "standard": 5.0, "rough": 8.0},

    # ── Bricks & Blocks ──
    "brick": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "aac_block": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "fly_ash_brick": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "concrete_block": {"careful": 2.0, "standard": 3.0, "rough": 5.0},

    # ── Tiles & Stone ──
    "vitrified_tile": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "ceramic_tile": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "marble": {"careful": 5.0, "standard": 8.0, "rough": 12.0},
    "granite": {"careful": 5.0, "standard": 8.0, "rough": 12.0},
    "kota_stone": {"careful": 4.0, "standard": 6.0, "rough": 10.0},
    "tile_adhesive": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "tile_grout": {"careful": 5.0, "standard": 8.0, "rough": 12.0},

    # ── Paint & Finishes ──
    "paint": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "primer": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "wall_putty": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "distemper": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "texture_paint": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "varnish": {"careful": 3.0, "standard": 5.0, "rough": 7.0},

    # ── Waterproofing ──
    "waterproofing_compound": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "app_membrane": {"careful": 5.0, "standard": 8.0, "rough": 12.0},
    "liquid_membrane": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "bitumen": {"careful": 2.0, "standard": 3.0, "rough": 5.0},

    # ── Plumbing ──
    "cpvc_pipe": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "upvc_pipe": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "gi_pipe": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "ppr_pipe": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "pipe_fittings": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "sanitary_fittings": {"careful": 1.0, "standard": 2.0, "rough": 3.0},

    # ── Electrical ──
    "wire": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "conduit": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "switch_socket": {"careful": 1.0, "standard": 2.0, "rough": 3.0},
    "mcb": {"careful": 1.0, "standard": 2.0, "rough": 3.0},
    "db_box": {"careful": 0.5, "standard": 1.0, "rough": 2.0},
    "cable": {"careful": 2.0, "standard": 3.0, "rough": 5.0},

    # ── Timber & Wood ──
    "timber": {"careful": 5.0, "standard": 8.0, "rough": 12.0},
    "plywood": {"careful": 5.0, "standard": 8.0, "rough": 12.0},
    "wooden_flooring": {"careful": 5.0, "standard": 8.0, "rough": 12.0},
    "door_frame": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "door_shutter": {"careful": 2.0, "standard": 3.0, "rough": 5.0},

    # ── Misc ──
    "glass": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "aluminium_section": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "pvc_sheet": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
    "gypsum_board": {"careful": 3.0, "standard": 5.0, "rough": 8.0},
    "insulation": {"careful": 2.0, "standard": 3.0, "rough": 5.0},
}

# Default wastage for unknown materials
DEFAULT_WASTAGE: Dict[str, float] = {"careful": 2.0, "standard": 3.0, "rough": 5.0}

# Site condition multipliers (applied on top of base wastage)
SITE_CONDITION_MULTIPLIER: Dict[str, float] = {
    "excellent": 0.8,     # Well-managed, good storage
    "good": 0.9,          # Decent management
    "standard": 1.0,      # Normal conditions
    "poor": 1.2,          # Poor storage, untrained labor
    "very_poor": 1.5,     # Remote site, harsh conditions
}

# Project type adjustments
PROJECT_TYPE_ADJUSTMENT: Dict[str, float] = {
    "residential": 1.0,
    "commercial": 0.95,     # Better management, less wastage
    "industrial": 0.9,      # Professional workforce
    "institutional": 1.0,
    "infrastructure": 1.1,  # Larger scale, more handling
}


# =============================================================================
# MATERIAL KEYWORD MATCHING
# =============================================================================

def _match_material_key(material_name: str) -> str:
    """
    Match a material name/description to a wastage norm key.

    Args:
        material_name: Material name or description

    Returns:
        Matching key from WASTAGE_NORMS or 'general'
    """
    name_lower = material_name.lower().strip()

    # Exact match
    if name_lower in WASTAGE_NORMS:
        return name_lower

    # Partial match patterns (order matters — more specific first)
    _PATTERNS = [
        # Steel
        ("steel_rebar", ["rebar", "tmt", "saria", "reinforcement", "bar bending"]),
        ("steel_structural", ["structural steel", "ms angle", "ms channel", "ismc"]),
        ("binding_wire", ["binding wire", "tie wire"]),
        ("ms_plate", ["ms plate", "steel plate"]),

        # Concrete
        ("cement", ["cement", "opc", "ppc"]),
        ("sand", ["sand", "river sand", "m-sand", "msand"]),
        ("aggregate_20mm", ["aggregate", "jelly", "grit", "metal", "bajri"]),
        ("ready_mix_concrete", ["rmc", "ready mix"]),

        # Bricks
        ("aac_block", ["aac", "autoclaved"]),
        ("fly_ash_brick", ["fly ash"]),
        ("concrete_block", ["concrete block", "hollow block"]),
        ("brick", ["brick"]),

        # Tiles
        ("vitrified_tile", ["vitrified"]),
        ("ceramic_tile", ["ceramic"]),
        ("marble", ["marble"]),
        ("granite", ["granite"]),
        ("kota_stone", ["kota"]),
        ("tile_adhesive", ["adhesive"]),
        ("tile_grout", ["grout"]),

        # Paint
        ("texture_paint", ["texture"]),
        ("wall_putty", ["putty"]),
        ("primer", ["primer"]),
        ("distemper", ["distemper"]),
        ("paint", ["paint", "emulsion"]),

        # Waterproofing
        ("app_membrane", ["app membrane", "sbs membrane"]),
        ("liquid_membrane", ["liquid membrane"]),
        ("waterproofing_compound", ["waterproof"]),

        # Plumbing
        ("cpvc_pipe", ["cpvc"]),
        ("upvc_pipe", ["upvc", "swr"]),
        ("gi_pipe", ["gi pipe", "galvanized"]),
        ("ppr_pipe", ["ppr"]),
        ("pipe_fittings", ["fitting", "elbow", "tee", "coupling"]),
        ("sanitary_fittings", ["sanitary", "wc", "basin", "cistern"]),

        # Electrical
        ("wire", ["wire", "cable"]),
        ("conduit", ["conduit"]),
        ("switch_socket", ["switch", "socket"]),
        ("mcb", ["mcb", "rccb"]),
        ("db_box", ["db box", "distribution board"]),

        # Timber
        ("plywood", ["plywood", "ply"]),
        ("wooden_flooring", ["wooden floor", "laminate"]),
        ("door_frame", ["door frame", "chaukhat"]),
        ("door_shutter", ["door shutter", "flush door"]),
        ("timber", ["timber", "wood"]),

        # Misc
        ("glass", ["glass"]),
        ("aluminium_section", ["aluminium", "aluminum"]),
        ("gypsum_board", ["gypsum", "drywall"]),
    ]

    for key, patterns in _PATTERNS:
        for pattern in patterns:
            if pattern in name_lower:
                return key

    return "general"


# =============================================================================
# WASTAGE CALCULATION
# =============================================================================

@dataclass
class WastageResult:
    """Result of wastage calculation for a single material."""
    material_name: str
    material_key: str
    net_quantity: float
    wastage_pct: float
    wastage_quantity: float
    gross_quantity: float
    unit: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_name": self.material_name,
            "material_key": self.material_key,
            "net_quantity": self.net_quantity,
            "wastage_pct": self.wastage_pct,
            "wastage_quantity": round(self.wastage_quantity, 3),
            "gross_quantity": round(self.gross_quantity, 3),
            "unit": self.unit,
            "notes": self.notes,
        }


def get_wastage_pct(
    material_name: str,
    site_condition: str = "standard",
    project_type: str = "residential",
) -> float:
    """
    Get wastage percentage for a material.

    Args:
        material_name: Material name or description
        site_condition: 'excellent', 'good', 'standard', 'poor', 'very_poor'
        project_type: 'residential', 'commercial', 'industrial', etc.

    Returns:
        Wastage percentage (e.g., 5.0 for 5%)
    """
    key = _match_material_key(material_name)

    norms = WASTAGE_NORMS.get(key, DEFAULT_WASTAGE)
    base_pct = norms.get(site_condition, norms.get("standard", 3.0))

    # Apply site and project multipliers
    site_mult = SITE_CONDITION_MULTIPLIER.get(site_condition, 1.0)
    proj_mult = PROJECT_TYPE_ADJUSTMENT.get(project_type, 1.0)

    return round(base_pct * site_mult * proj_mult, 2)


def calculate_wastage(
    material_name: str,
    net_quantity: float,
    unit: str = "",
    site_condition: str = "standard",
    project_type: str = "residential",
) -> WastageResult:
    """
    Calculate wastage for a single material.

    Args:
        material_name: Material name
        net_quantity: Net (design) quantity
        unit: Unit of measurement
        site_condition: Site condition level
        project_type: Project type

    Returns:
        WastageResult with gross quantity
    """
    key = _match_material_key(material_name)
    pct = get_wastage_pct(material_name, site_condition, project_type)

    wastage_qty = net_quantity * pct / 100.0
    gross_qty = net_quantity + wastage_qty

    return WastageResult(
        material_name=material_name,
        material_key=key,
        net_quantity=net_quantity,
        wastage_pct=pct,
        wastage_quantity=wastage_qty,
        gross_quantity=gross_qty,
        unit=unit,
        notes=f"Wastage {pct}% applied ({site_condition} site, {project_type} project)",
    )


def apply_wastage_to_materials(
    materials: List[Dict[str, Any]],
    site_condition: str = "standard",
    project_type: str = "residential",
) -> Dict[str, Any]:
    """
    Apply wastage to an entire materials list (BOM).

    Args:
        materials: List of material dicts with 'material_name', 'quantity', 'unit'
        site_condition: Site condition
        project_type: Project type

    Returns:
        Dict with adjusted_materials, summary, total_wastage_cost
    """
    adjusted = []
    total_net = 0.0
    total_gross = 0.0

    for mat in materials:
        name = mat.get("material_name", mat.get("name", ""))
        qty = float(mat.get("quantity", mat.get("qty", 0)))
        unit = mat.get("unit", "")

        result = calculate_wastage(name, qty, unit, site_condition, project_type)

        adjusted_mat = mat.copy()
        adjusted_mat["net_quantity"] = qty
        adjusted_mat["gross_quantity"] = round(result.gross_quantity, 3)
        adjusted_mat["wastage_pct"] = result.wastage_pct
        adjusted_mat["wastage_quantity"] = round(result.wastage_quantity, 3)
        adjusted_mat["material_key"] = result.material_key
        adjusted.append(adjusted_mat)

        # Track totals (rough estimate using quantity as proxy for cost)
        total_net += qty
        total_gross += result.gross_quantity

    overall_wastage_pct = round(
        (total_gross - total_net) / total_net * 100, 2
    ) if total_net > 0 else 0.0

    return {
        "adjusted_materials": adjusted,
        "total_materials": len(adjusted),
        "overall_wastage_pct": overall_wastage_pct,
        "site_condition": site_condition,
        "project_type": project_type,
        "note": "Gross quantities include wastage buffer for procurement. "
                "Actual wastage may vary based on workmanship and storage.",
    }
