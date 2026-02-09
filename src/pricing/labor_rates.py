"""
Labor Rate Book - India-specific labor rates.
"""

from typing import Dict


class LaborRateBook:
    """Labor rate database with India-specific daily wages."""

    # Base daily wages (Delhi, 2024 rates)
    # Rates in INR per day (8-hour day)
    BASE_RATES = {
        # Skilled labor
        "mason": 900,
        "tile_mason": 950,
        "carpenter": 850,
        "plumber": 850,
        "electrician": 850,
        "painter": 750,
        "welder": 900,
        "fabricator": 900,
        "aluminum_fabricator": 950,
        "waterproofing_applicator": 900,
        "polisher": 800,
        "fitter": 850,

        # Semi-skilled labor
        "helper": 550,
        "bar_bender": 750,
        "mixer_operator": 650,
        "crane_operator": 1000,
        "driver": 700,

        # Unskilled labor
        "coolie": 500,
        "mazdoor": 500,
        "beldar": 500,

        # Specialized trades
        "false_ceiling_worker": 900,
        "gypsum_worker": 900,
        "hvac_technician": 1000,
        "fire_system_tech": 1000,
        "lift_technician": 1200,

        # Supervisory
        "foreman": 1200,
        "site_supervisor": 1500,
        "safety_officer": 1800,
    }

    # Productivity factors by trade (output per day)
    PRODUCTIVITY = {
        "mason": {
            "brick_masonry_cum": 0.8,  # cum per day
            "plastering_sqm": 12,  # sqm per day
            "pointing_sqm": 8,
        },
        "tile_mason": {
            "flooring_sqm": 8,  # sqm per day
            "dado_sqm": 6,
        },
        "carpenter": {
            "formwork_sqm": 6,  # sqm per day
            "door_fixing_nos": 2,
        },
        "painter": {
            "painting_sqm": 25,  # sqm per day (2 coats)
            "putty_sqm": 15,
        },
        "plumber": {
            "pipe_laying_rmt": 15,  # rmt per day
            "fixture_nos": 4,
        },
        "electrician": {
            "wiring_rmt": 30,  # rmt per day
            "point_nos": 8,
        },
    }

    # Minimum wages by state (as per 2024 notifications)
    STATE_MIN_WAGES = {
        "delhi": {"skilled": 738, "semi_skilled": 672, "unskilled": 609},
        "maharashtra": {"skilled": 650, "semi_skilled": 585, "unskilled": 525},
        "karnataka": {"skilled": 598, "semi_skilled": 548, "unskilled": 500},
        "tamil_nadu": {"skilled": 620, "semi_skilled": 560, "unskilled": 505},
        "telangana": {"skilled": 610, "semi_skilled": 555, "unskilled": 500},
        "andhra_pradesh": {"skilled": 590, "semi_skilled": 540, "unskilled": 490},
        "kerala": {"skilled": 680, "semi_skilled": 620, "unskilled": 560},
        "west_bengal": {"skilled": 560, "semi_skilled": 510, "unskilled": 460},
        "gujarat": {"skilled": 590, "semi_skilled": 540, "unskilled": 490},
        "rajasthan": {"skilled": 530, "semi_skilled": 485, "unskilled": 440},
        "uttar_pradesh": {"skilled": 520, "semi_skilled": 475, "unskilled": 430},
        "madhya_pradesh": {"skilled": 510, "semi_skilled": 465, "unskilled": 420},
        "punjab": {"skilled": 620, "semi_skilled": 565, "unskilled": 510},
        "haryana": {"skilled": 650, "semi_skilled": 590, "unskilled": 535},
    }

    def __init__(self):
        pass

    def get_rate(self, trade: str, state: str = "delhi") -> float:
        """Get daily wage for a trade."""
        trade_key = self._normalize_trade_name(trade)

        if trade_key in self.BASE_RATES:
            base_rate = self.BASE_RATES[trade_key]

            # Adjust for state if different from Delhi
            if state.lower() != "delhi":
                state_factor = self._get_state_factor(state.lower())
                return base_rate * state_factor

            return base_rate

        # Try partial matching
        for key, rate in self.BASE_RATES.items():
            if trade_key in key or key in trade_key:
                return rate

        # Default to helper rate
        return self.BASE_RATES.get("helper", 550)

    def _normalize_trade_name(self, trade: str) -> str:
        """Normalize trade name for lookup."""
        return trade.lower().strip().replace(" ", "_").replace("-", "_")

    def _get_state_factor(self, state: str) -> float:
        """Get wage adjustment factor for state vs Delhi."""
        if state not in self.STATE_MIN_WAGES:
            return 1.0

        delhi_avg = (738 + 672 + 609) / 3
        state_wages = self.STATE_MIN_WAGES[state]
        state_avg = (state_wages["skilled"] + state_wages["semi_skilled"] + state_wages["unskilled"]) / 3

        return state_avg / delhi_avg

    def get_productivity(self, trade: str, work_type: str) -> float:
        """Get productivity (output per day) for a trade and work type."""
        trade_key = self._normalize_trade_name(trade)

        if trade_key in self.PRODUCTIVITY:
            return self.PRODUCTIVITY[trade_key].get(work_type, 1.0)

        return 1.0

    def get_labor_cost_per_unit(
        self,
        trade: str,
        work_type: str,
        helpers: int = 1,
        state: str = "delhi",
    ) -> float:
        """Calculate labor cost per unit of work."""
        trade_rate = self.get_rate(trade, state)
        helper_rate = self.get_rate("helper", state)

        productivity = self.get_productivity(trade, work_type)
        if productivity == 0:
            productivity = 1.0

        # Total daily cost
        daily_cost = trade_rate + (helper_rate * helpers)

        # Cost per unit
        return daily_cost / productivity

    def get_all_trades(self) -> Dict:
        """Get all trades with rates."""
        return self.BASE_RATES.copy()

    def calculate_gang_cost(
        self,
        gang_composition: Dict[str, int],
        state: str = "delhi",
    ) -> float:
        """Calculate daily cost for a gang of workers."""
        total = 0
        for trade, count in gang_composition.items():
            rate = self.get_rate(trade, state)
            total += rate * count
        return total
