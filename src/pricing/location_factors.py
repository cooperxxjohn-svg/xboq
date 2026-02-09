"""
Location Multipliers - City-wise cost factors for India.
"""

from typing import Dict, Optional


class LocationMultiplier:
    """Location-based cost multipliers for different cities in India."""

    # City multipliers relative to Delhi (base = 1.0)
    # Factors account for: material transport, labor costs, logistics
    CITY_MULTIPLIERS = {
        # Metro cities
        "delhi": 1.00,
        "new delhi": 1.00,
        "gurgaon": 1.02,
        "gurugram": 1.02,
        "noida": 0.98,
        "greater noida": 0.96,
        "faridabad": 0.97,
        "ghaziabad": 0.96,

        "mumbai": 1.15,
        "navi mumbai": 1.10,
        "thane": 1.08,
        "pune": 1.05,

        "bangalore": 1.08,
        "bengaluru": 1.08,

        "chennai": 1.02,
        "hyderabad": 1.00,
        "kolkata": 0.95,

        # Tier-2 cities
        "ahmedabad": 0.95,
        "surat": 0.92,
        "vadodara": 0.90,
        "rajkot": 0.88,

        "jaipur": 0.90,
        "jodhpur": 0.88,
        "udaipur": 0.92,

        "lucknow": 0.85,
        "kanpur": 0.83,
        "varanasi": 0.82,
        "agra": 0.84,

        "chandigarh": 0.98,
        "ludhiana": 0.92,
        "amritsar": 0.88,

        "indore": 0.88,
        "bhopal": 0.86,
        "nagpur": 0.90,

        "patna": 0.82,
        "ranchi": 0.85,
        "bhubaneswar": 0.85,

        "kochi": 1.00,
        "thiruvananthapuram": 0.98,
        "coimbatore": 0.95,
        "madurai": 0.90,
        "trichy": 0.88,

        "visakhapatnam": 0.92,
        "vijayawada": 0.88,

        # Hill stations / difficult terrain
        "shimla": 1.15,
        "dehradun": 1.05,
        "mussoorie": 1.20,
        "nainital": 1.18,
        "manali": 1.25,
        "srinagar": 1.30,
        "leh": 1.50,
        "darjeeling": 1.15,
        "shillong": 1.20,
        "gangtok": 1.25,

        # Northeast
        "guwahati": 1.08,
        "imphal": 1.25,
        "aizawl": 1.30,
        "kohima": 1.35,
        "agartala": 1.15,

        # Coastal / remote
        "port blair": 1.40,
        "daman": 0.95,
        "goa": 1.05,
        "panaji": 1.05,
        "pondicherry": 0.95,
        "puducherry": 0.95,
    }

    # State-level multipliers (fallback when city not found)
    STATE_MULTIPLIERS = {
        "delhi": 1.00,
        "haryana": 0.98,
        "uttar pradesh": 0.85,
        "maharashtra": 1.08,
        "karnataka": 1.05,
        "tamil nadu": 0.98,
        "telangana": 0.98,
        "andhra pradesh": 0.90,
        "kerala": 1.00,
        "west bengal": 0.92,
        "gujarat": 0.92,
        "rajasthan": 0.88,
        "madhya pradesh": 0.85,
        "punjab": 0.95,
        "bihar": 0.80,
        "jharkhand": 0.82,
        "odisha": 0.85,
        "chhattisgarh": 0.85,
        "assam": 1.05,
        "himachal pradesh": 1.12,
        "uttarakhand": 1.05,
        "goa": 1.05,
        "jammu and kashmir": 1.25,
        "ladakh": 1.45,
        "sikkim": 1.20,
        "arunachal pradesh": 1.30,
        "nagaland": 1.30,
        "manipur": 1.25,
        "mizoram": 1.28,
        "tripura": 1.12,
        "meghalaya": 1.18,
    }

    # Package-specific adjustments (some materials more affected by location)
    PACKAGE_ADJUSTMENTS = {
        "civil_structural": {
            "remote": 1.10,  # Higher transport for heavy materials
            "metro": 1.00,
        },
        "masonry": {
            "remote": 1.08,
            "metro": 1.00,
        },
        "flooring": {
            "remote": 1.05,  # Tiles can be transported efficiently
            "metro": 1.00,
        },
        "plumbing": {
            "remote": 1.05,
            "metro": 1.00,
        },
        "electrical": {
            "remote": 1.03,  # Lightweight materials
            "metro": 1.00,
        },
        "doors_windows": {
            "remote": 1.08,
            "metro": 1.00,
        },
    }

    def __init__(self):
        pass

    def get_multiplier(self, city: str, state: str = None) -> float:
        """Get location multiplier for a city."""
        city_key = self._normalize_location(city)

        # Try exact city match
        if city_key in self.CITY_MULTIPLIERS:
            return self.CITY_MULTIPLIERS[city_key]

        # Try state match
        if state:
            state_key = self._normalize_location(state)
            if state_key in self.STATE_MULTIPLIERS:
                return self.STATE_MULTIPLIERS[state_key]

        # Try to infer state from city name patterns
        inferred_state = self._infer_state(city_key)
        if inferred_state and inferred_state in self.STATE_MULTIPLIERS:
            return self.STATE_MULTIPLIERS[inferred_state]

        # Default multiplier
        return 1.0

    def get_package_multiplier(
        self,
        city: str,
        package: str,
        state: str = None,
    ) -> float:
        """Get location multiplier adjusted for package type."""
        base_mult = self.get_multiplier(city, state)

        # Determine if remote or metro
        if base_mult > 1.15:
            location_type = "remote"
        else:
            location_type = "metro"

        # Apply package adjustment
        if package in self.PACKAGE_ADJUSTMENTS:
            pkg_adj = self.PACKAGE_ADJUSTMENTS[package].get(location_type, 1.0)
            return base_mult * pkg_adj

        return base_mult

    def _normalize_location(self, location: str) -> str:
        """Normalize location name for lookup."""
        return location.lower().strip()

    def _infer_state(self, city: str) -> Optional[str]:
        """Try to infer state from city characteristics."""
        # Known city-state mappings for common cities not in list
        city_state_map = {
            "mysore": "karnataka",
            "mysuru": "karnataka",
            "mangalore": "karnataka",
            "mangaluru": "karnataka",
            "hubli": "karnataka",
            "belgaum": "karnataka",
            "nashik": "maharashtra",
            "aurangabad": "maharashtra",
            "solapur": "maharashtra",
            "kolhapur": "maharashtra",
            "meerut": "uttar pradesh",
            "allahabad": "uttar pradesh",
            "prayagraj": "uttar pradesh",
            "gorakhpur": "uttar pradesh",
            "bareilly": "uttar pradesh",
            "aligarh": "uttar pradesh",
            "moradabad": "uttar pradesh",
            "jabalpur": "madhya pradesh",
            "gwalior": "madhya pradesh",
            "ujjain": "madhya pradesh",
            "raipur": "chhattisgarh",
            "bilaspur": "chhattisgarh",
            "cuttack": "odisha",
            "rourkela": "odisha",
            "jamshedpur": "jharkhand",
            "dhanbad": "jharkhand",
            "asansol": "west bengal",
            "durgapur": "west bengal",
            "siliguri": "west bengal",
            "howrah": "west bengal",
        }

        return city_state_map.get(city)

    def get_all_cities(self) -> Dict:
        """Get all cities with multipliers."""
        return self.CITY_MULTIPLIERS.copy()

    def get_all_states(self) -> Dict:
        """Get all states with multipliers."""
        return self.STATE_MULTIPLIERS.copy()

    def estimate_transport_cost(
        self,
        from_city: str,
        to_city: str,
        weight_mt: float,
    ) -> float:
        """Estimate transport cost between cities."""
        # Simplified estimation based on distance proxy (multiplier difference)
        from_mult = self.get_multiplier(from_city)
        to_mult = self.get_multiplier(to_city)

        # Base rate per MT per "distance unit"
        base_rate = 2000  # INR per MT

        # Distance proxy (difference in multipliers * 10)
        distance_factor = abs(to_mult - from_mult) * 10 + 1

        return weight_mt * base_rate * distance_factor
