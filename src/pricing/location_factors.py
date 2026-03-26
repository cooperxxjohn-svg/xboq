"""
Location Factors — City-level cost adjustment for Indian construction.

Provides:
- CITY_FACTORS: Cost multiplier relative to Delhi (base=1.0) for 30+ Indian cities
- MATERIAL_CITY_ADJUSTMENTS: Per-material per-city overrides
- Functions: get_city_factor, get_material_city_factor, adjust_rate_for_location,
  get_all_cities, get_nearest_city

All factors are base = Delhi = 1.0.  Tier classification follows Census 2011 UA
population cutoffs: Tier-1 > 4M, Tier-2 1-4M, Tier-3 < 1M.

Can be used standalone or called from escalation.py via the `location` parameter.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CITY FACTORS  (relative to Delhi = 1.0)
# =============================================================================

CITY_FACTORS: Dict[str, float] = {
    # ── Tier 1 (Metros) ───────────────────────────────────────────────
    "delhi": 1.00,
    "new delhi": 1.00,
    "mumbai": 1.15,
    "bangalore": 1.08,
    "bengaluru": 1.08,
    "chennai": 1.05,
    "kolkata": 0.95,
    "hyderabad": 1.02,
    "pune": 1.10,
    "ahmedabad": 0.98,

    # ── Tier 2 ────────────────────────────────────────────────────────
    "lucknow": 0.88,
    "jaipur": 0.90,
    "chandigarh": 0.95,
    "bhopal": 0.85,
    "nagpur": 0.88,
    "indore": 0.87,
    "visakhapatnam": 0.92,
    "vizag": 0.92,
    "kochi": 1.05,
    "coimbatore": 0.95,
    "vadodara": 0.95,
    "surat": 0.92,
    "rajkot": 0.88,
    "jodhpur": 0.88,
    "udaipur": 0.92,
    "kanpur": 0.83,
    "ludhiana": 0.92,
    "amritsar": 0.88,
    "madurai": 0.90,
    "trichy": 0.88,
    "vijayawada": 0.88,
    "thiruvananthapuram": 0.98,

    # ── Tier 3 ────────────────────────────────────────────────────────
    "dehradun": 0.92,
    "ranchi": 0.82,
    "patna": 0.80,
    "guwahati": 0.90,
    "bhubaneswar": 0.85,
    "mysore": 0.95,
    "mysuru": 0.95,
    "mangalore": 0.98,
    "mangaluru": 0.98,
    "raipur": 0.82,
    "jammu": 0.95,
    "shimla": 1.05,
    "rishikesh": 0.95,
    "varanasi": 0.85,
    "agra": 0.88,

    # ── NCR / Parent-city satellites ──────────────────────────────────
    "navi mumbai": 1.15,      # parent: Mumbai
    "thane": 1.15,            # parent: Mumbai
    "noida": 1.00,            # parent: Delhi
    "greater noida": 0.98,
    "gurgaon": 1.00,          # parent: Delhi
    "gurugram": 1.00,
    "faridabad": 0.97,
    "ghaziabad": 0.96,

    # ── Hill stations / difficult terrain ─────────────────────────────
    "mussoorie": 1.20,
    "nainital": 1.18,
    "manali": 1.25,
    "srinagar": 1.30,
    "leh": 1.50,
    "darjeeling": 1.15,

    # ── Northeast ─────────────────────────────────────────────────────
    "shillong": 1.20,
    "gangtok": 1.25,
    "imphal": 1.25,
    "aizawl": 1.30,
    "kohima": 1.35,
    "agartala": 1.15,

    # ── Coastal / UT ──────────────────────────────────────────────────
    "port blair": 1.40,
    "daman": 0.95,
    "goa": 1.05,
    "panaji": 1.05,
    "pondicherry": 0.95,
    "puducherry": 0.95,
}


# =============================================================================
# MATERIAL-CITY ADJUSTMENTS
# =============================================================================
# Per-material per-city (or region) overrides expressed as *additive* percentage
# adjustments to the base CITY_FACTORS multiplier.
#
# Usage: effective_factor = city_factor * (1 + adjustment/100)
# E.g. Steel in Mumbai: 1.15 * (1 + 5/100) = 1.2075
#
# "south_india" / "ne_india" are region groups resolved at lookup time.
# =============================================================================

# Region membership for material adjustments
_SOUTH_INDIA_CITIES = {
    "chennai", "bangalore", "bengaluru", "hyderabad", "kochi",
    "coimbatore", "madurai", "trichy", "mangalore", "mangaluru",
    "mysore", "mysuru", "thiruvananthapuram", "vijayawada",
    "visakhapatnam", "vizag", "pondicherry", "puducherry",
}

_NE_INDIA_CITIES = {
    "guwahati", "shillong", "gangtok", "imphal", "aizawl",
    "kohima", "agartala", "darjeeling",
}

MATERIAL_CITY_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # ── Steel ─────────────────────────────────────────────────────────
    "steel": {
        "mumbai": 5.0,
        "navi mumbai": 5.0,
        "thane": 5.0,
        "kolkata": -3.0,
        "chennai": 2.0,
    },

    # ── Cement ────────────────────────────────────────────────────────
    # South India has proximity to major cement plants (Andhra/TN belt)
    "cement": {
        # All south_india cities get -5%; populated at module load below
    },

    # ── Aggregates (sand, stone, gravel) ──────────────────────────────
    "aggregates": {
        "bangalore": 15.0,   # Sand mining restrictions in Karnataka
        "bengaluru": 15.0,
        "mumbai": 10.0,
        "navi mumbai": 10.0,
        "thane": 10.0,
    },

    # ── Labour ────────────────────────────────────────────────────────
    "labor": {
        "mumbai": 20.0,
        "navi mumbai": 18.0,
        "thane": 18.0,
        "bangalore": 15.0,
        "bengaluru": 15.0,
        "chennai": 10.0,
        "pune": 12.0,
        "kolkata": -5.0,
        "gurgaon": 8.0,
        "gurugram": 8.0,
        "noida": 5.0,
    },

    # ── Timber ────────────────────────────────────────────────────────
    # NE India has better timber availability; South has less supply
    "timber": {
        # NE cities get -10%, South cities get +5%; populated below
    },

    # ── Bricks ────────────────────────────────────────────────────────
    "bricks": {
        "mumbai": 8.0,
        "bangalore": 5.0,
        "bengaluru": 5.0,
    },

    # ── Fuel / Transport ──────────────────────────────────────────────
    "fuel_transport": {
        "mumbai": 5.0,
        "leh": 30.0,
        "srinagar": 15.0,
        "port blair": 25.0,
        "manali": 12.0,
        "shimla": 8.0,
    },
}

# Populate region-based cement adjustments (South India -5%)
for _city in _SOUTH_INDIA_CITIES:
    MATERIAL_CITY_ADJUSTMENTS["cement"].setdefault(_city, -5.0)

# Populate region-based timber adjustments
for _city in _NE_INDIA_CITIES:
    MATERIAL_CITY_ADJUSTMENTS["timber"].setdefault(_city, -10.0)
for _city in _SOUTH_INDIA_CITIES:
    MATERIAL_CITY_ADJUSTMENTS["timber"].setdefault(_city, 5.0)


# =============================================================================
# STATE-LEVEL FALLBACKS
# =============================================================================

STATE_MULTIPLIERS: Dict[str, float] = {
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


# City-to-state inference table (for cities not in CITY_FACTORS)
_CITY_STATE_MAP: Dict[str, str] = {
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


# Common aliases / alternate spellings
_CITY_ALIASES: Dict[str, str] = {
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
    "mangaluru": "mangalore",
    "mysuru": "mysore",
    "thiruvananthapuram": "thiruvananthapuram",
    "trivandrum": "thiruvananthapuram",
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
    "vizag": "visakhapatnam",
    "pondicherry": "puducherry",
    "benares": "varanasi",
    "kashi": "varanasi",
    "prayagraj": "allahabad",
    "cochin": "kochi",
    "calicut": "kochi",
    "poona": "pune",
    "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram",
}


# =============================================================================
# HELPER: Normalize and match city names
# =============================================================================

def _normalize(name: str) -> str:
    """Lowercase, strip whitespace and common suffixes."""
    return name.lower().strip().replace(".", "").replace(",", "")


def _fuzzy_score(a: str, b: str) -> float:
    """Return similarity ratio 0..1 between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def _resolve_city_key(city: str) -> Optional[str]:
    """
    Resolve a user-supplied city name to a canonical key in CITY_FACTORS.

    Resolution order:
    1. Exact match (normalized)
    2. Alias lookup
    3. Best fuzzy match (>= 0.80 threshold)

    Returns None if no confident match found.
    """
    norm = _normalize(city)

    # Exact match
    if norm in CITY_FACTORS:
        return norm

    # Alias
    alias_target = _CITY_ALIASES.get(norm)
    if alias_target:
        canon = _normalize(alias_target)
        if canon in CITY_FACTORS:
            return canon

    # Fuzzy match
    best_key = None
    best_score = 0.0
    for key in CITY_FACTORS:
        score = _fuzzy_score(norm, key)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= 0.80 and best_key is not None:
        return best_key

    return None


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def get_city_factor(city: str) -> float:
    """
    Return overall cost multiplier for *city* relative to Delhi (1.0).

    Uses fuzzy matching on the city name.  Falls back to state-level
    multiplier if the city can be mapped to a known state, otherwise
    returns 1.0 (Delhi baseline).

    Args:
        city: City name (case-insensitive, common aliases accepted).

    Returns:
        Cost multiplier float, e.g. 1.15 for Mumbai.
    """
    key = _resolve_city_key(city)
    if key is not None:
        return CITY_FACTORS[key]

    # Try state-level fallback via inference map
    norm = _normalize(city)
    state = _CITY_STATE_MAP.get(norm)
    if state and state in STATE_MULTIPLIERS:
        return STATE_MULTIPLIERS[state]

    logger.warning("City '%s' not found in location factors; using base 1.0", city)
    return 1.0


def get_material_city_factor(city: str, material_category: str) -> float:
    """
    Return material-specific cost factor for *city*.

    Combines the overall city factor with any per-material per-city
    adjustment from MATERIAL_CITY_ADJUSTMENTS.

    Args:
        city: City name.
        material_category: Material key — e.g. 'steel', 'cement', 'labor',
            'aggregates', 'timber', 'bricks', 'fuel_transport'.  Also
            accepts MaterialCategory enum values.

    Returns:
        Adjusted cost multiplier float.
    """
    base_factor = get_city_factor(city)

    # Normalize material key (handle enum .value or plain string)
    mat_key = str(material_category).lower().strip()
    if "." in mat_key:
        mat_key = mat_key.split(".")[-1]

    adjustments = MATERIAL_CITY_ADJUSTMENTS.get(mat_key)
    if not adjustments:
        return base_factor

    # Resolve city key for adjustment lookup
    city_key = _resolve_city_key(city) or _normalize(city)
    adj_pct = adjustments.get(city_key, 0.0)

    if adj_pct != 0.0:
        return round(base_factor * (1 + adj_pct / 100.0), 4)

    return base_factor


def adjust_rate_for_location(
    base_rate: float,
    city: str,
    material_category: Optional[str] = None,
) -> float:
    """
    Apply location factor to a base rate (Delhi 2024-Q1).

    If *material_category* is given, applies the material-specific
    city adjustment; otherwise uses the overall city factor.

    Args:
        base_rate: Rate in INR (Delhi basis).
        city: Target city.
        material_category: Optional material key for finer adjustment.

    Returns:
        Adjusted rate in INR.
    """
    if material_category:
        factor = get_material_city_factor(city, material_category)
    else:
        factor = get_city_factor(city)

    return round(base_rate * factor, 2)


def get_all_cities() -> List[str]:
    """
    Return a sorted list of all supported city names.

    Returns:
        List of city name strings (canonical, lowercase).
    """
    return sorted(CITY_FACTORS.keys())


def get_nearest_city(city: str) -> Tuple[str, float]:
    """
    Fuzzy-match *city* to the nearest supported city name.

    Args:
        city: User-supplied city name.

    Returns:
        (matched_city, confidence) where confidence is 0.0 .. 1.0.
        If exact match found, confidence is 1.0.
        If no reasonable match, returns ('delhi', 0.0).
    """
    norm = _normalize(city)

    # Exact hit
    if norm in CITY_FACTORS:
        return (norm, 1.0)

    # Alias hit
    alias_target = _CITY_ALIASES.get(norm)
    if alias_target:
        canon = _normalize(alias_target)
        if canon in CITY_FACTORS:
            return (canon, 0.98)

    # Fuzzy scan
    best_key = "delhi"
    best_score = 0.0
    for key in CITY_FACTORS:
        score = _fuzzy_score(norm, key)
        if score > best_score:
            best_score = score
            best_key = key

    return (best_key, round(best_score, 2))


# =============================================================================
# BACKWARD-COMPATIBLE CLASS (used by __init__.py / rate_builder.py)
# =============================================================================

class LocationMultiplier:
    """
    Location-based cost multipliers for different cities in India.

    Preserved for backward compatibility with existing callers
    (``__init__.py``, ``rate_builder.py``).  New code should prefer the
    module-level functions above.
    """

    # Expose module-level dicts as class attributes for legacy access
    CITY_MULTIPLIERS = CITY_FACTORS
    STATE_MULTIPLIERS = STATE_MULTIPLIERS

    # Package-specific adjustments (some materials more affected by location)
    PACKAGE_ADJUSTMENTS = {
        "civil_structural": {
            "remote": 1.10,
            "metro": 1.00,
        },
        "masonry": {
            "remote": 1.08,
            "metro": 1.00,
        },
        "flooring": {
            "remote": 1.05,
            "metro": 1.00,
        },
        "plumbing": {
            "remote": 1.05,
            "metro": 1.00,
        },
        "electrical": {
            "remote": 1.03,
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
        factor = get_city_factor(city)

        # If module-level lookup returned default 1.0 and we have a state,
        # try state-level fallback explicitly.
        if factor == 1.0 and state:
            state_key = _normalize(state)
            if state_key in STATE_MULTIPLIERS:
                return STATE_MULTIPLIERS[state_key]

        return factor

    def get_package_multiplier(
        self,
        city: str,
        package: str,
        state: str = None,
    ) -> float:
        """Get location multiplier adjusted for package type."""
        base_mult = self.get_multiplier(city, state)

        if base_mult > 1.15:
            location_type = "remote"
        else:
            location_type = "metro"

        if package in self.PACKAGE_ADJUSTMENTS:
            pkg_adj = self.PACKAGE_ADJUSTMENTS[package].get(location_type, 1.0)
            return base_mult * pkg_adj

        return base_mult

    def _normalize_location(self, location: str) -> str:
        return _normalize(location)

    def _infer_state(self, city: str) -> Optional[str]:
        return _CITY_STATE_MAP.get(city)

    def get_all_cities(self) -> Dict:
        return CITY_FACTORS.copy()

    def get_all_states(self) -> Dict:
        return STATE_MULTIPLIERS.copy()

    def estimate_transport_cost(
        self,
        from_city: str,
        to_city: str,
        weight_mt: float,
    ) -> float:
        """Estimate transport cost between cities."""
        from_mult = self.get_multiplier(from_city)
        to_mult = self.get_multiplier(to_city)
        base_rate = 2000  # INR per MT
        distance_factor = abs(to_mult - from_mult) * 10 + 1
        return weight_mt * base_rate * distance_factor
