"""
Material Price Book - India-specific material prices.
"""

from typing import Dict, Optional
import re


class MaterialPriceBook:
    """Material price database with India-specific rates."""

    # Base prices (Delhi, 2024 rates, standard grade)
    # Prices in INR
    BASE_PRICES = {
        # Cement and aggregates
        "cement": {"unit": "bag", "basic": 350, "standard": 380, "premium": 420},  # 50kg bag
        "sand": {"unit": "cum", "basic": 3500, "standard": 4000, "premium": 4500},  # River sand
        "aggregate_20mm": {"unit": "cum", "basic": 2800, "standard": 3200, "premium": 3500},
        "aggregate_10mm": {"unit": "cum", "basic": 3000, "standard": 3400, "premium": 3800},

        # Bricks and blocks
        "brick": {"unit": "nos", "basic": 7, "standard": 8, "premium": 10},  # Red brick
        "aac_block": {"unit": "nos", "basic": 55, "standard": 60, "premium": 70},  # 600x200x200
        "fly_ash_brick": {"unit": "nos", "basic": 6, "standard": 7, "premium": 8},

        # Steel
        "steel_rebar": {"unit": "kg", "basic": 65, "standard": 68, "premium": 72},  # TMT bars
        "steel_structural": {"unit": "kg", "basic": 70, "standard": 75, "premium": 80},
        "ms_plate": {"unit": "kg", "basic": 75, "standard": 80, "premium": 85},

        # Flooring materials
        "vitrified_tile_600x600": {"unit": "sqm", "basic": 450, "standard": 650, "premium": 1200},
        "vitrified_tile_800x800": {"unit": "sqm", "basic": 550, "standard": 800, "premium": 1500},
        "ceramic_tile": {"unit": "sqm", "basic": 300, "standard": 450, "premium": 700},
        "marble": {"unit": "sqm", "basic": 800, "standard": 1500, "premium": 3500},
        "granite": {"unit": "sqm", "basic": 700, "standard": 1200, "premium": 2500},
        "kota_stone": {"unit": "sqm", "basic": 350, "standard": 450, "premium": 600},
        "wooden_flooring": {"unit": "sqm", "basic": 1200, "standard": 2000, "premium": 4000},

        # Tile adhesives
        "tile_adhesive": {"unit": "kg", "basic": 25, "standard": 35, "premium": 50},
        "tile_grout": {"unit": "kg", "basic": 40, "standard": 60, "premium": 100},

        # Painting materials
        "wall_putty": {"unit": "kg", "basic": 25, "standard": 35, "premium": 45},
        "primer": {"unit": "ltr", "basic": 150, "standard": 200, "premium": 300},
        "acrylic_emulsion": {"unit": "ltr", "basic": 200, "standard": 350, "premium": 550},
        "exterior_emulsion": {"unit": "ltr", "basic": 250, "standard": 400, "premium": 600},
        "distemper": {"unit": "ltr", "basic": 80, "standard": 120, "premium": 180},
        "cement_primer": {"unit": "ltr", "basic": 120, "standard": 160, "premium": 220},
        "texture_paint": {"unit": "ltr", "basic": 400, "standard": 600, "premium": 1000},

        # Waterproofing materials
        "waterproofing_compound": {"unit": "kg", "basic": 80, "standard": 120, "premium": 180},
        "app_membrane": {"unit": "sqm", "basic": 250, "standard": 350, "premium": 500},
        "liquid_membrane": {"unit": "ltr", "basic": 200, "standard": 300, "premium": 450},
        "brick_bat": {"unit": "cum", "basic": 1500, "standard": 1800, "premium": 2200},

        # Plumbing materials
        "cpvc_pipe_15mm": {"unit": "rmt", "basic": 50, "standard": 65, "premium": 85},
        "cpvc_pipe_20mm": {"unit": "rmt", "basic": 70, "standard": 90, "premium": 120},
        "cpvc_pipe_25mm": {"unit": "rmt", "basic": 100, "standard": 130, "premium": 170},
        "cpvc_fittings": {"unit": "set", "basic": 200, "standard": 300, "premium": 450},
        "upvc_swr_110mm": {"unit": "rmt", "basic": 180, "standard": 220, "premium": 280},
        "upvc_swr_75mm": {"unit": "rmt", "basic": 120, "standard": 150, "premium": 190},
        "gi_pipe_15mm": {"unit": "rmt", "basic": 120, "standard": 150, "premium": 190},
        "ppr_pipe_20mm": {"unit": "rmt", "basic": 80, "standard": 100, "premium": 130},

        # Electrical materials
        "wire_1.5sqmm": {"unit": "rmt", "basic": 12, "standard": 16, "premium": 22},
        "wire_2.5sqmm": {"unit": "rmt", "basic": 18, "standard": 24, "premium": 32},
        "wire_4sqmm": {"unit": "rmt", "basic": 28, "standard": 36, "premium": 48},
        "conduit_20mm": {"unit": "rmt", "basic": 25, "standard": 35, "premium": 50},
        "conduit_25mm": {"unit": "rmt", "basic": 35, "standard": 45, "premium": 65},
        "switch_socket": {"unit": "nos", "basic": 80, "standard": 180, "premium": 350},
        "junction_box": {"unit": "nos", "basic": 30, "standard": 50, "premium": 80},
        "mcb_sp": {"unit": "nos", "basic": 120, "standard": 200, "premium": 350},
        "mcb_dp": {"unit": "nos", "basic": 250, "standard": 400, "premium": 650},
        "db_box_4way": {"unit": "nos", "basic": 400, "standard": 700, "premium": 1200},
        "db_box_8way": {"unit": "nos", "basic": 700, "standard": 1200, "premium": 2000},

        # Door materials
        "flush_door_shutter": {"unit": "sqm", "basic": 1200, "standard": 1800, "premium": 2800},
        "door_frame_sal": {"unit": "cum", "basic": 45000, "standard": 55000, "premium": 70000},
        "door_frame_teak": {"unit": "cum", "basic": 120000, "standard": 150000, "premium": 200000},
        "hardware_set": {"unit": "set", "basic": 500, "standard": 1200, "premium": 3000},
        "wpc_door": {"unit": "nos", "basic": 4500, "standard": 6500, "premium": 9000},
        "pvc_door": {"unit": "nos", "basic": 3000, "standard": 4500, "premium": 6500},

        # Window materials
        "aluminum_section": {"unit": "kg", "basic": 280, "standard": 320, "premium": 380},
        "glass_5mm": {"unit": "sqm", "basic": 180, "standard": 250, "premium": 350},
        "glass_6mm": {"unit": "sqm", "basic": 220, "standard": 300, "premium": 420},
        "glass_8mm": {"unit": "sqm", "basic": 320, "standard": 420, "premium": 550},
        "upvc_profile": {"unit": "kg", "basic": 350, "standard": 420, "premium": 520},
        "ms_grill": {"unit": "kg", "basic": 80, "standard": 95, "premium": 120},
        "ss_grill": {"unit": "kg", "basic": 280, "standard": 350, "premium": 450},

        # Sanitary fittings
        "ewc_basic": {"unit": "nos", "basic": 3500, "standard": 6000, "premium": 15000},
        "iwc_basic": {"unit": "nos", "basic": 1500, "standard": 2500, "premium": 4500},
        "wash_basin": {"unit": "nos", "basic": 1800, "standard": 4000, "premium": 12000},
        "cistern_exposed": {"unit": "nos", "basic": 1200, "standard": 2500, "premium": 5000},
        "cistern_concealed": {"unit": "nos", "basic": 4000, "standard": 8000, "premium": 18000},
        "shower_mixer": {"unit": "nos", "basic": 2500, "standard": 6000, "premium": 15000},
        "cp_pillar_cock": {"unit": "nos", "basic": 800, "standard": 2000, "premium": 5000},

        # External works
        "paver_block_60mm": {"unit": "sqm", "basic": 450, "standard": 550, "premium": 700},
        "paver_block_80mm": {"unit": "sqm", "basic": 550, "standard": 680, "premium": 850},
        "ms_gate": {"unit": "kg", "basic": 90, "standard": 110, "premium": 140},
    }

    # Composite rates for common items (all-inclusive per unit)
    COMPOSITE_RATES = {
        # RCC works (per cum, including formwork, steel avg 80 kg/cum)
        "rcc_m20_footing": {"unit": "cum", "basic": 7500, "standard": 8500, "premium": 10000},
        "rcc_m25_column": {"unit": "cum", "basic": 12000, "standard": 14000, "premium": 17000},
        "rcc_m25_beam": {"unit": "cum", "basic": 11000, "standard": 13000, "premium": 16000},
        "rcc_m25_slab": {"unit": "cum", "basic": 9500, "standard": 11000, "premium": 13500},
        "rcc_m20_plinth_beam": {"unit": "cum", "basic": 8500, "standard": 10000, "premium": 12000},

        # PCC works
        "pcc_m10": {"unit": "cum", "basic": 4500, "standard": 5000, "premium": 5500},
        "pcc_m15": {"unit": "cum", "basic": 5000, "standard": 5500, "premium": 6200},

        # Masonry works
        "brick_masonry_230mm": {"unit": "cum", "basic": 6500, "standard": 7500, "premium": 9000},
        "brick_masonry_115mm": {"unit": "sqm", "basic": 650, "standard": 750, "premium": 900},
        "aac_block_200mm": {"unit": "cum", "basic": 5500, "standard": 6500, "premium": 8000},

        # Plastering
        "cement_plaster_12mm": {"unit": "sqm", "basic": 150, "standard": 180, "premium": 220},
        "cement_plaster_20mm": {"unit": "sqm", "basic": 200, "standard": 240, "premium": 290},
        "pop_punning": {"unit": "sqm", "basic": 80, "standard": 100, "premium": 130},

        # Flooring complete
        "vitrified_flooring_complete": {"unit": "sqm", "basic": 900, "standard": 1200, "premium": 2000},
        "marble_flooring_complete": {"unit": "sqm", "basic": 1500, "standard": 2500, "premium": 5000},
        "granite_flooring_complete": {"unit": "sqm", "basic": 1300, "standard": 2000, "premium": 4000},
        "ceramic_flooring_complete": {"unit": "sqm", "basic": 700, "standard": 950, "premium": 1400},

        # Wall tiling complete
        "ceramic_dado_complete": {"unit": "sqm", "basic": 750, "standard": 1000, "premium": 1500},

        # Painting complete
        "internal_paint_complete": {"unit": "sqm", "basic": 100, "standard": 150, "premium": 250},
        "external_paint_complete": {"unit": "sqm", "basic": 120, "standard": 180, "premium": 300},

        # Waterproofing complete
        "toilet_wp_complete": {"unit": "sqm", "basic": 300, "standard": 450, "premium": 700},
        "terrace_wp_complete": {"unit": "sqm", "basic": 450, "standard": 650, "premium": 950},

        # Doors complete (per nos)
        "flush_door_complete": {"unit": "nos", "basic": 8000, "standard": 12000, "premium": 20000},
        "main_door_complete": {"unit": "nos", "basic": 25000, "standard": 45000, "premium": 80000},

        # Windows complete (per sqm)
        "aluminum_window_complete": {"unit": "sqm", "basic": 4500, "standard": 6000, "premium": 9000},
        "upvc_window_complete": {"unit": "sqm", "basic": 5500, "standard": 7500, "premium": 11000},

        # Plumbing per point
        "water_supply_point": {"unit": "nos", "basic": 1500, "standard": 2200, "premium": 3500},
        "drainage_point": {"unit": "nos", "basic": 1200, "standard": 1800, "premium": 2800},

        # Electrical per point
        "light_point": {"unit": "nos", "basic": 800, "standard": 1200, "premium": 2000},
        "power_point": {"unit": "nos", "basic": 1000, "standard": 1500, "premium": 2500},
        "ac_point": {"unit": "nos", "basic": 2500, "standard": 3500, "premium": 5500},
    }

    def __init__(self):
        pass

    def get_price(self, item: str, grade: str = "standard") -> float:
        """Get material price for specified grade."""
        item_key = self._normalize_item_name(item)

        if item_key in self.BASE_PRICES:
            prices = self.BASE_PRICES[item_key]
            return prices.get(grade, prices.get("standard", 0))

        # Try partial matching
        for key, prices in self.BASE_PRICES.items():
            if item_key in key or key in item_key:
                return prices.get(grade, prices.get("standard", 0))

        return 0

    def get_composite_rate(self, description: str, unit: str) -> float:
        """Get composite rate for an item description."""
        desc_lower = description.lower()

        # Try to match composite rates
        for key, rate_info in self.COMPOSITE_RATES.items():
            keywords = key.replace("_", " ").split()
            if all(kw in desc_lower for kw in keywords):
                return rate_info.get("standard", 0)

        # Fallback: try base prices if unit matches
        for key, prices in self.BASE_PRICES.items():
            keywords = key.replace("_", " ").split()
            if all(kw in desc_lower for kw in keywords):
                return prices.get("standard", 0)

        return 0

    def _normalize_item_name(self, item: str) -> str:
        """Normalize item name for lookup."""
        return item.lower().strip().replace(" ", "_").replace("-", "_")

    def get_all_materials(self) -> Dict:
        """Get all materials with prices."""
        return self.BASE_PRICES.copy()
