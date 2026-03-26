"""
Rate Engine — Apply Indian market unit rates (INR) to BOQ line items.

Rates are Q1 2025 Mumbai / Tier-1 city benchmark rates.
Regional multipliers scale rates for Tier-2 and Tier-3 cities.

Usage:
    from src.analysis.qto.rate_engine import apply_rates, compute_trade_summary

    rated_items = apply_rates(items, region="tier2")
    summary     = compute_trade_summary(rated_items)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# RATE DATABASE  — Q1 2025 Mumbai / Tier-1 benchmark
# Key:   lowercase keyword (longer key = more specific match, wins over shorter)
# Value: (rate_inr_per_unit, unit_description)
# =============================================================================

_RATES: Dict[str, Tuple[float, str]] = {

    # ── Earthwork / Civil ────────────────────────────────────────────────────
    "site clearance":                   (25.0,      "sqm"),
    "grubbing":                         (25.0,      "sqm"),
    "topsoil excavation":               (165.0,     "cum"),
    "bulk excavation":                  (210.0,     "cum"),
    "excavation":                       (185.0,     "cum"),
    "disposal of excavated":            (105.0,     "cum"),
    "disposal":                         (105.0,     "cum"),
    "hardcore filling":                 (1200.0,    "cum"),
    "hardcore":                         (1100.0,    "cum"),
    "anti-termite":                     (95.0,      "sqm"),
    "anti termite":                     (95.0,      "sqm"),
    "pcc 1:4:8":                        (4550.0,    "cum"),
    "plain cement concrete":            (4850.0,    "cum"),
    "pcc":                              (4850.0,    "cum"),
    "interlocking concrete block paving": (750.0,   "sqm"),
    "interlocking":                     (720.0,     "sqm"),
    "external paving":                  (680.0,     "sqm"),
    "paving":                           (650.0,     "sqm"),
    "compound / boundary wall":         (3200.0,    "lm"),

    # ── Structural — Concrete ─────────────────────────────────────────────────
    "rcc m30":                          (9200.0,    "cum"),
    "rcc m25":                          (8800.0,    "cum"),
    "rcc m20":                          (8200.0,    "cum"),
    "reinforced cement concrete":       (8500.0,    "cum"),
    "concrete slab":                    (8500.0,    "cum"),
    "solid flat slab":                  (8500.0,    "cum"),
    "concrete column":                  (9200.0,    "cum"),
    "columns":                          (9200.0,    "cum"),
    "concrete beam":                    (9000.0,    "cum"),
    "beams":                            (9000.0,    "cum"),
    "isolated footings":                (8200.0,    "cum"),
    "footing":                          (8200.0,    "cum"),
    "staircase":                        (9000.0,    "cum"),
    "rcc wall":                         (8500.0,    "cum"),
    "rcc":                              (8500.0,    "cum"),

    # ── Structural — Steel ────────────────────────────────────────────────────
    "fe500 deformed bars":              (88000.0,   "MT"),
    "fe500":                            (88000.0,   "MT"),
    "fe415":                            (84000.0,   "MT"),
    "deformed bars":                    (87000.0,   "MT"),
    "steel reinforcement":              (88000.0,   "MT"),
    "tor steel":                        (88000.0,   "MT"),
    "rebar":                            (88000.0,   "MT"),

    # ── Structural — Formwork ─────────────────────────────────────────────────
    "shuttering for columns":           (950.0,     "sqm"),
    "shuttering for beams":             (950.0,     "sqm"),
    "centering, shuttering":            (900.0,     "sqm"),
    "shuttering":                       (900.0,     "sqm"),
    "formwork slab":                    (850.0,     "sqm"),
    "formwork column":                  (950.0,     "sqm"),
    "formwork beam":                    (950.0,     "sqm"),
    "formwork":                         (900.0,     "sqm"),

    # ── Masonry ───────────────────────────────────────────────────────────────
    "230mm thick brick":                (4100.0,    "cum"),
    "brickwork":                        (4100.0,    "cum"),
    "aac block":                        (2800.0,    "sqm"),
    "cfc block":                        (2800.0,    "sqm"),
    "block work":                       (2800.0,    "sqm"),

    # ── Finishes — Floor ──────────────────────────────────────────────────────
    "800x800":                          (850.0,     "sqm"),
    "vitrified tiles in flooring":      (720.0,     "sqm"),
    "vitrified tile dado":              (680.0,     "sqm"),
    "vitrified tile":                   (720.0,     "sqm"),
    "vitrified":                        (700.0,     "sqm"),
    "non-slip ceramic tile":            (520.0,     "sqm"),
    "ceramic tiles in flooring":        (480.0,     "sqm"),
    "ceramic tile dado":                (580.0,     "sqm"),
    "ceramic tile":                     (480.0,     "sqm"),
    "ceramic":                          (480.0,     "sqm"),
    "marble flooring":                  (1800.0,    "sqm"),
    "marble":                           (1800.0,    "sqm"),
    "granite flooring":                 (1400.0,    "sqm"),
    "granite floor":                    (1400.0,    "sqm"),
    "kota stone flooring":              (650.0,     "sqm"),
    "kota stone":                       (650.0,     "sqm"),
    "kota":                             (620.0,     "sqm"),
    "anti-skid":                        (550.0,     "sqm"),
    "anti skid":                        (550.0,     "sqm"),
    "epoxy flooring":                   (1200.0,    "sqm"),
    "epoxy":                            (1100.0,    "sqm"),
    "ips flooring":                     (350.0,     "sqm"),
    "cc flooring":                      (280.0,     "sqm"),

    # ── Finishes — Wall ───────────────────────────────────────────────────────
    "cement plaster":                   (165.0,     "sqm"),
    "white cement putty":               (120.0,     "sqm"),
    "gypsum putty":                     (140.0,     "sqm"),
    "putty":                            (90.0,      "sqm"),

    # ── Painting ──────────────────────────────────────────────────────────────
    "weather shield":                   (140.0,     "sqm"),
    "exterior":                         (140.0,     "sqm"),
    "acrylic emulsion":                 (80.0,      "sqm"),
    "oil bound distemper":              (75.0,      "sqm"),
    "obd":                              (75.0,      "sqm"),
    "anti-carbonation coating":         (350.0,     "sqm"),
    "exterior weather coat":            (160.0,     "sqm"),
    "enamel paint":                     (160.0,     "sqm"),
    "enamel":                           (160.0,     "sqm"),
    "epoxy primer":                     (85.0,      "sqm"),
    "acrylic primer coat":              (45.0,      "sqm"),
    "texture paint":                    (220.0,     "sqm"),
    "texture":                          (200.0,     "sqm"),
    "anti-corrosive paint":             (180.0,     "sqm"),
    "snowcem":                          (65.0,      "sqm"),
    "distemper":                        (75.0,      "sqm"),
    "primer":                           (45.0,      "sqm"),
    "painting":                         (85.0,      "sqm"),
    "paint":                            (85.0,      "sqm"),

    # ── Waterproofing ─────────────────────────────────────────────────────────
    "crystalline waterproofing":        (380.0,     "sqm"),
    "app membrane":                     (520.0,     "sqm"),
    "bituminous tanking":               (620.0,     "sqm"),
    "bituminous":                       (580.0,     "sqm"),
    "polymer modified waterproofing":   (290.0,     "sqm"),
    "waterproofing":                    (350.0,     "sqm"),
    "sealant to joints":                (180.0,     "lm"),
    "sealant":                          (160.0,     "lm"),
    "grouting":                         (120.0,     "lm"),

    # ── Doors & Windows ───────────────────────────────────────────────────────
    "hw flush door":                    (4200.0,    "sqm"),
    "flush door":                       (4200.0,    "sqm"),
    "ms rolling shutter":               (3500.0,    "sqm"),
    "rolling shutter":                  (3200.0,    "sqm"),
    "upvc window":                      (2200.0,    "sqm"),
    "aluminium window":                 (2800.0,    "sqm"),
    "steel window":                     (1800.0,    "sqm"),
    "window":                           (2200.0,    "sqm"),
    "timber door frame":                (2800.0,    "No"),
    "ms door frame":                    (3500.0,    "No"),
    "door frame":                       (2800.0,    "No"),
    "ironmongery":                      (2500.0,    "No"),
    "ms grille":                        (1200.0,    "sqm"),
    "grille":                           (1100.0,    "sqm"),
    "granite sill":                     (350.0,     "lm"),
    "window sill":                      (300.0,     "lm"),

    # ── MEP — Electrical ──────────────────────────────────────────────────────
    "led lighting fixture":             (1800.0,    "No"),
    "lighting fixture":                 (1800.0,    "No"),
    "ceiling fan":                      (2200.0,    "No"),
    "exhaust fan":                      (1200.0,    "No"),
    "power socket":                     (850.0,     "No"),
    "distribution board":               (18000.0,   "No"),
    "pvc conduit 25mm":                 (120.0,     "lm"),
    "conduit":                          (110.0,     "lm"),
    "wiring 4 sqmm":                    (145.0,     "lm"),
    "wiring 1.5 sqmm":                  (95.0,      "lm"),
    "wiring":                           (95.0,      "lm"),
    "earthing":                         (8500.0,    "No"),

    # ── MEP — Plumbing ────────────────────────────────────────────────────────
    "wc pan":                           (5500.0,    "No"),
    "wash basin":                       (4500.0,    "No"),
    "kitchen sink":                     (4200.0,    "No"),
    "shower unit":                      (3800.0,    "No"),
    "cpvc pipe 40mm":                   (380.0,     "lm"),
    "cpvc pipe 25mm":                   (280.0,     "lm"),
    "cpvc pipe":                        (280.0,     "lm"),
    "soil/waste pipe":                  (420.0,     "lm"),
    "soil pipe":                        (400.0,     "lm"),
    "cold water pipe":                  (180.0,     "lm"),
    "overhead tank connection":         (8500.0,    "No"),
    "sump pump":                        (22000.0,   "No"),

    # ── MEP — HVAC ────────────────────────────────────────────────────────────
    "cassette ac":                      (72000.0,   "No"),
    "split ac 1.5 ton":                 (48000.0,   "No"),
    "split ac 1 ton":                   (38000.0,   "No"),
    "split ac":                         (43000.0,   "No"),
    "air handling unit":                (180000.0,  "No"),
    "ahu":                              (180000.0,  "No"),
    "refrigerant piping":               (850.0,     "lm"),

    # ── Site / Utilities ──────────────────────────────────────────────────────
    "underground rcc sump":             (85000.0,   "No"),
    "hdpe water storage tank":          (18000.0,   "No"),
    "overhead hdpe":                    (18000.0,   "No"),
    "septic tank":                      (45000.0,   "No"),
    "main entrance gate":               (38000.0,   "No"),
    "ms fabricated":                    (38000.0,   "No"),
    "stormwater drainage":              (1800.0,    "lm"),
    "external stormwater":              (1800.0,    "lm"),
    "external drainage":                (1800.0,    "lm"),
    "drainage":                         (1600.0,    "lm"),
    "landscaping":                      (350.0,     "sqm"),
    "landscape":                        (320.0,     "sqm"),
    "turfing":                          (280.0,     "sqm"),

    # ── ELV — Fire Alarm ─────────────────────────────────────────────────────
    "addressable smoke detector":       (3200.0,    "Nos"),
    "smoke detector":                   (2800.0,    "Nos"),
    "heat detector":                    (3000.0,    "Nos"),
    "manual call point":                (2200.0,    "Nos"),
    "break-glass":                      (2000.0,    "Nos"),
    "fire alarm control panel":         (85000.0,   "Nos"),
    "facp":                             (75000.0,   "Nos"),
    "fire alarm panel":                 (75000.0,   "Nos"),
    "hooter":                           (1800.0,    "Nos"),
    "sounder":                          (1800.0,    "Nos"),
    "frls 1.5":                         (75.0,      "rm"),
    "fire alarm cable":                 (75.0,      "rm"),

    # ── ELV — CCTV ───────────────────────────────────────────────────────────
    "ip/hd cctv camera":                (18000.0,   "Nos"),
    "ip cctv camera":                   (17000.0,   "Nos"),
    "hd cctv camera":                   (16000.0,   "Nos"),
    "cctv camera":                      (15000.0,   "Nos"),
    "nvr/dvr":                          (22000.0,   "Nos"),
    "nvr":                              (25000.0,   "Nos"),
    "dvr":                              (20000.0,   "Nos"),
    "led monitor":                      (12000.0,   "Nos"),
    "surveillance monitor":             (12000.0,   "Nos"),

    # ── ELV — PA / Public Address ─────────────────────────────────────────────
    "pa speaker":                       (2500.0,    "Nos"),
    "ceiling speaker":                  (2200.0,    "Nos"),
    "pa amplifier":                     (18000.0,   "Nos"),
    "pa control":                       (12000.0,   "Nos"),
    "pa cable":                         (60.0,      "rm"),
    "2-core 1.5 sqmm":                  (65.0,      "rm"),

    # ── ELV — Data Networking ────────────────────────────────────────────────
    "cat6 data outlet":                 (1800.0,    "Nos"),
    "data outlet":                      (1500.0,    "Nos"),
    "rj45":                             (1500.0,    "Nos"),
    "gigabit ethernet network switch":  (35000.0,   "Nos"),
    "managed gigabit":                  (30000.0,   "Nos"),
    "network switch":                   (28000.0,   "Nos"),
    "cat6 utp 4-pair":                  (65.0,      "rm"),
    "cat6 utp":                         (60.0,      "rm"),
    "cat6 cable":                       (60.0,      "rm"),
    "ofc backbone":                     (180.0,     "rm"),
    "single-mode ofc":                  (180.0,     "rm"),
    "optical fibre":                    (150.0,     "rm"),

    # ── ELV — Telephone / Structured Wiring ──────────────────────────────────
    "telephone outlet":                 (1200.0,    "Nos"),
    "rj11":                             (1200.0,    "Nos"),
    "telephone cable":                  (45.0,      "rm"),
    "cw1308":                           (45.0,      "rm"),
    "structured cabling":               (850.0,     "point"),

    # ── ELV — Access Control ─────────────────────────────────────────────────
    "access control reader":            (8500.0,    "Nos"),
    "proximity card reader":            (8000.0,    "Nos"),
    "biometric":                        (12000.0,   "Nos"),
    "access control controller":        (22000.0,   "Nos"),
    "electric strike":                  (5500.0,    "Nos"),
    "electromagnetic lock":             (6500.0,    "Nos"),
    "door closer":                      (2500.0,    "Nos"),

    # ── ELV — Nurse Call ─────────────────────────────────────────────────────
    "nurse call push-button":           (4500.0,    "Nos"),
    "patient call":                     (4500.0,    "Nos"),
    "nurse call master station":        (28000.0,   "Nos"),
    "nurse call":                       (4000.0,    "Nos"),

    # ── External Development ──────────────────────────────────────────────────
    "overhead water tank":              (18000.0,   "KL"),
    "overhead tank":                    (18000.0,   "KL"),
    "oht":                              (18000.0,   "KL"),
    "underground sump":                 (22000.0,   "KL"),
    "sump tank":                        (22000.0,   "KLD"),
    "sewage treatment plant":           (85000.0,   "KLD"),
    "stp":                              (80000.0,   "KLD"),
    "gi class-c pipe":                  (380.0,     "rm"),
    "gi pipe":                          (350.0,     "rm"),
    "pvc swr pipe":                     (280.0,     "rm"),
    "swr pipe":                         (260.0,     "rm"),
    "hdpe pipe":                        (320.0,     "rm"),
    "distribution transformer":         (850000.0,  "No"),
    "oil-cooled transformer":           (800000.0,  "No"),
    "transformer":                      (750000.0,  "No"),
    "lt cable trench":                  (2800.0,    "rm"),
    "cable trench":                     (2500.0,    "rm"),
    "street light pole":                (25000.0,   "No"),
    "street light":                     (22000.0,   "No"),
    "water bound macadam":              (420.0,     "sqm"),
    "wbm":                              (400.0,     "sqm"),
    "bituminous macadam":               (680.0,     "sqm"),
    "dbm":                              (650.0,     "sqm"),
    "bituminous concrete":              (720.0,     "sqm"),
    "concrete road":                    (1200.0,    "sqm"),
    "brick masonry u-drain":            (3500.0,    "rm"),
    "u-drain":                          (3200.0,    "rm"),
    "storm water drain":                (3200.0,    "rm"),
    "topsoil filling":                  (180.0,     "sqm"),
    "horticulture":                     (320.0,     "sqm"),
    "plantation":                       (280.0,     "sqm"),
    "grass turf":                       (120.0,     "sqm"),
    "compound wall":                    (3500.0,    "rm"),
    "boundary wall":                    (3500.0,    "rm"),
    "main gate":                        (150000.0,  "No"),
    "pedestrian gate":                  (45000.0,   "No"),
    "solar pv panel":                   (45000.0,   "kWp"),
    "solar rooftop":                    (45000.0,   "kWp"),
}

# ── Composite item BOM expansion ──────────────────────────────────────────────
# Maps lowercase trigger substrings → list of (component_desc, qty_fraction, unit_override)
# qty_fraction: component_qty = parent_qty * fraction
# unit_override: replaces parent unit for this component (None = inherit parent unit)
_BOM_EXPANSION: dict = {
    "flush door": [
        ("timber door frame 100x75mm",        1.0,  "Nos"),
        ("painting doors and frames",          3.5,  "sqm"),
        ("ironmongery door fittings",          1.0,  "Nos"),
        ("hydraulic door closer",              1.0,  "Nos"),
    ],
    "hw door": [
        ("hardwood timber door frame",         1.0,  "Nos"),
        ("painting doors and frames",          3.5,  "sqm"),
        ("ironmongery door fittings",          1.0,  "Nos"),
    ],
    "aluminium window": [
        ("aluminium sliding window",           1.0,  None),
        ("silicone sealant perimeter",         4.0,  "lm"),
        ("kota stone window sill",             1.0,  "lm"),
    ],
    "upvc window": [
        ("upvc casement window",               1.0,  None),
        ("silicone sealant perimeter",         3.5,  "lm"),
    ],
    "rcc column": [
        ("rcc m25 in columns",                 1.0,  "cum"),
        ("centering and shuttering columns",   8.0,  "sqm"),
        ("reinforcement fe500 columns",        0.12, "MT"),
    ],
    "rcc beam": [
        ("rcc m25 in beams",                   1.0,  "cum"),
        ("centering and shuttering beams",     6.0,  "sqm"),
        ("reinforcement fe500 beams",          0.10, "MT"),
    ],
    "rcc slab": [
        ("rcc m20 in slabs",                   1.0,  "cum"),
        ("centering and shuttering slabs",     1.0,  "sqm"),
        ("reinforcement fe415 slabs",          0.08, "MT"),
    ],
    "isolated footing": [
        ("pcc m10 bed",                        0.25, "cum"),
        ("rcc m25 in footings",                1.0,  "cum"),
        ("centering shuttering footings",      3.0,  "sqm"),
        ("reinforcement fe500 footings",       0.09, "MT"),
    ],
    "brickwork in cm": [
        ("brickwork 230mm cm 1:6",             1.0,  None),
        ("cement plaster 12mm internal",       1.6,  "sqm"),
    ],
    "aac block masonry": [
        ("aac block masonry 200mm",            1.0,  None),
        ("cement plaster 12mm internal",       1.6,  "sqm"),
    ],
    "vitrified tile flooring": [
        ("vitrified tile 600x600",             1.0,  None),
        ("cement plaster bed 25mm",            1.0,  "sqm"),
        ("tile grout joints",                  1.0,  "sqm"),
    ],
    "marble flooring": [
        ("marble flooring 18mm",               1.0,  None),
        ("cement mortar base 25mm",            1.0,  "sqm"),
        ("white cement grouting",              1.0,  "sqm"),
    ],
    "ms rolling shutter": [
        ("ms rolling shutter",                 1.0,  None),
        ("painting rolling shutter enamel",    2.0,  "sqm"),
        ("guide channel anchor bolts",         2.0,  "lm"),
    ],
    "compound wall brick": [
        ("brickwork 230mm compound wall",      1.0,  "cum"),
        ("cement plaster both sides",          2.0,  "sqm"),
        ("painting exterior two coats",        2.0,  "sqm"),
    ],
    "underground sump": [
        ("pcc m10 bed sump",                   0.20, "cum"),
        ("rcc m25 walls sump",                 1.0,  "cum"),
        ("centering shuttering sump",          5.0,  "sqm"),
        ("reinforcement fe500 sump",           0.12, "MT"),
        ("waterproofing cementitious sump",    1.0,  "sqm"),
    ],
    "overhead water tank": [
        ("rcc m25 overhead tank",              1.0,  "cum"),
        ("centering shuttering tank",          5.0,  "sqm"),
        ("reinforcement fe500 tank",           0.12, "MT"),
        ("cement plaster waterproofed",        1.0,  "sqm"),
    ],
    "septic tank": [
        ("brick masonry septic tank",          1.0,  "cum"),
        ("rcc m20 cover slab",                 0.15, "cum"),
        ("waterproofing cementitious",         1.0,  "sqm"),
    ],
    "epoxy flooring": [
        ("epoxy primer coat",                  1.0,  "sqm"),
        ("epoxy flooring 3mm",                 1.0,  None),
        ("polyurethane topcoat",               1.0,  "sqm"),
    ],
    "staircase rcc": [
        ("rcc m25 staircase waist slab",       1.0,  "cum"),
        ("centering shuttering staircase",     6.0,  "sqm"),
        ("reinforcement fe500 staircase",      0.11, "MT"),
        ("marble flooring 18mm",               3.5,  "sqm"),
    ],
    "external paving": [
        ("compacted hardcore bed 150mm",       1.0,  "sqm"),
        ("pcc m10 base 75mm",                  0.075,"cum"),
        ("interlocking paving block 80mm",     1.0,  None),
        ("sand bedding 25mm",                  0.025,"cum"),
    ],
}

# =============================================================================
# REGIONAL MULTIPLIERS
# =============================================================================

_REGION_MULTIPLIERS: Dict[str, float] = {
    "tier1": 1.00,   # Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Pune
    "tier2": 0.85,   # Ahmedabad, Kolkata, Jaipur, Lucknow, Nagpur, Coimbatore
    "tier3": 0.72,   # Smaller cities, semi-urban, rural
}

_DEFAULT_REGION = "tier1"

_RATE_DB_VERSION = "rate_db_v2_dsr2023_calibrated"


# =============================================================================
# MATCHING LOGIC
# =============================================================================

def _match_rate(description: str) -> Tuple[Optional[float], str, float]:
    """
    Find best matching rate for a BOQ item description.

    Matching strategy: substring search — longer keyword wins (more specific).

    Returns:
        (rate_inr, unit_description, confidence) or (None, "", 0.0) if no match.
    """
    desc_lower = description.lower()
    best_match: Optional[Tuple[float, str, float]] = None
    best_score: int = 0

    for keyword, (rate, unit_desc) in _RATES.items():
        if keyword in desc_lower:
            score = len(keyword)   # longer match = more specific = wins
            if score > best_score:
                best_score = score
                # Confidence: base 0.60, +0.02 per character of matched keyword
                confidence = min(0.95, 0.60 + score * 0.02)
                best_match = (rate, unit_desc, confidence)

    return best_match or (None, "", 0.0)


def _match_rate_with_dsr_fallback(
    description: str,
    unit: str = "",
) -> tuple:
    """
    Two-tier rate lookup:
    1. In-memory _RATES keyword match (fast).
    2. dsr_lookup.RateLookup fallback if confidence < 0.70.
    Returns (rate_inr, unit_desc, confidence).
    """
    rate, unit_desc, confidence = _match_rate(description)
    if confidence >= 0.70:
        return rate, unit_desc, confidence
    # Tier-2: best-effort DSR file lookup
    try:
        from src.analysis.rate_intelligence.dsr_lookup import RateLookup
        _lookup = RateLookup()
        match = _lookup.find_best_match(description, unit=unit, min_score=0.30)
        if match:
            dsr_item, dsr_score = match
            dsr_rate = float(dsr_item.get("rate", 0))
            if dsr_rate > 0:
                return dsr_rate, dsr_item.get("unit", ""), min(0.75, dsr_score)
    except Exception:
        pass  # DSR lookup is best-effort; never crash
    return rate, unit_desc, confidence


# =============================================================================
# BOM EXPANSION
# =============================================================================

def _expand_bom(item: dict, region_multiplier: float = 1.0) -> Optional[List[dict]]:
    """
    If item description matches a composite BOM trigger, return a list of
    expanded component dicts with individual rates applied.
    Returns None when no BOM match found (caller uses single-rate path).
    Each component emits both 'qty' and 'quantity' keys for compatibility.
    """
    desc_lower = (item.get("description") or "").lower()
    for trigger, components in _BOM_EXPANSION.items():
        if trigger in desc_lower:
            parent_qty = float(item.get("qty") or item.get("quantity") or 0.0)
            if parent_qty == 0:
                return None  # zero-qty: skip expansion
            expanded = []
            for comp_desc, qty_fraction, unit_override in components:
                comp_qty = round(parent_qty * qty_fraction, 4)
                rate, _, confidence = _match_rate(comp_desc)
                adjusted_rate = round((rate or 0.0) * region_multiplier, 2)
                # Build component item — inherit parent metadata, override specific fields
                comp = {k: v for k, v in item.items()
                        if k not in ("description", "qty", "quantity", "unit",
                                     "rate_inr", "amount_inr", "rate_source",
                                     "rate_confidence", "rate_region")}
                comp.update({
                    "description":     comp_desc,
                    "qty":             comp_qty,
                    "quantity":        comp_qty,
                    "unit":            unit_override if unit_override is not None else item.get("unit", ""),
                    "rate_inr":        adjusted_rate,
                    "amount_inr":      round(adjusted_rate * comp_qty, 2),
                    "rate_source":     _RATE_DB_VERSION + "_bom",
                    "rate_confidence": round(min(confidence, 0.85), 2),
                    "rate_region":     item.get("rate_region", "tier1"),
                    "bom_parent":      item.get("description", ""),
                    "bom_expanded":    True,
                })
                expanded.append(comp)
            return expanded
    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def apply_rates(
    items: List[dict],
    region: str = "tier1",
    use_dsr_fallback: bool = True,
    expand_composites: bool = True,
    project_rates: Optional[dict] = None,
) -> List[dict]:
    """
    Apply Indian market unit rates to a list of BOQ line items.

    Each item dict is extended in-place with:
        rate_inr          float  — matched unit rate (0 if no match)
        amount_inr        float  — rate × qty
        rate_source       str    — "rate_db_v2_dsr2023_calibrated" | "project_override" | ""
        rate_confidence   float  — match confidence (0.0–0.95)
        rate_region       str    — region code used

    When expand_composites=True (default), composite items (e.g. flush doors,
    RCC columns) are replaced by their BOM component breakdown.

    Args:
        items:             List of BOQ item dicts (from any QTO module).
        region:            "tier1" | "tier2" | "tier3"
        use_dsr_fallback:  Fall back to DSR lookup for low-confidence matches.
        expand_composites: If True, expand composite items into BOM components.
        project_rates:     Optional dict of project/org-level rate overrides
                           (from src.analysis.project_rates.load_rates).
                           These take priority over all built-in rates.

    Returns:
        List of items with rate fields added.  Composite items are replaced by
        their expanded components when expand_composites=True.
    """
    # Build keyword → rate lookup from project_rates overrides (P0-6)
    _override_lookup: Dict[str, float] = {}
    if project_rates:
        try:
            from src.analysis.project_rates import rates_to_lookup
            _override_lookup = rates_to_lookup(project_rates)
        except Exception as _e:
            logger.warning("project_rates lookup build failed: %s", _e)

    # Try pluggable RateSource first (R5 abstraction)
    _rate_src = None
    try:
        from src.analysis.rate_source import RateSourceRegistry as _RSR
        _rate_src = _RSR  # use registry for cross-source lookup
    except Exception:
        pass

    region = region.lower().strip()
    multiplier = _REGION_MULTIPLIERS.get(region, _REGION_MULTIPLIERS[_DEFAULT_REGION])

    if region not in _REGION_MULTIPLIERS:
        logger.warning(
            "Unknown region '%s' — defaulting to tier1 rates (multiplier=1.0)", region
        )
        region = _DEFAULT_REGION

    unmatched: List[str] = []
    result_items: List[dict] = []

    for item in items:
        # BOM expansion path: composite items → multiple component rows
        if expand_composites:
            expanded = _expand_bom(item, region_multiplier=multiplier)
            if expanded is not None:
                result_items.extend(expanded)
                continue

        # Single-rate path (original logic)
        description: str = item.get("description") or ""
        qty: float = float(item.get("qty") or 0.0)

        # ── Project/org override lookup — highest priority (P0-6) ────────
        _override_rate: Optional[float] = None
        if _override_lookup:
            _desc_lower = description.lower()
            for _kw, _kw_rate in _override_lookup.items():
                if _kw in _desc_lower:
                    _override_rate = _kw_rate
                    break  # first (longest) match wins

        if _override_rate is not None:
            # Project override rates are NOT scaled by regional multiplier
            # (they are actual procurement rates, already region-specific)
            amount = round(_override_rate * qty, 2)
            item["rate_inr"] = _override_rate
            item["amount_inr"] = amount
            item["rate_source"] = "project_override"
            item["rate_confidence"] = 1.0
            item["rate_region"] = region
            result_items.append(item)
            continue

        if use_dsr_fallback:
            rate, _unit_desc, confidence = _match_rate_with_dsr_fallback(
                description, unit=item.get("unit", "")
            )
        else:
            rate, _unit_desc, confidence = _match_rate(description)

        if rate is not None:
            adjusted_rate = round(rate * multiplier, 2)
            amount = round(adjusted_rate * qty, 2)
            item["rate_inr"] = adjusted_rate
            item["amount_inr"] = amount
            item["rate_source"] = _RATE_DB_VERSION
            item["rate_confidence"] = round(confidence, 2)
        else:
            item["rate_inr"] = 0.0
            item["amount_inr"] = 0.0
            item["rate_source"] = ""
            item["rate_confidence"] = 0.0
            unmatched.append(description[:80])

        # Fallback to pluggable RateSource registry (tries DSR → MH-PWD → DL-CPWD)
        if _rate_src and (not item.get("rate_inr") or item.get("rate_inr") == 0):
            try:
                _rs_result = _rate_src.lookup_across_sources(
                    description=item.get("description", ""),
                    unit=item.get("unit", ""),
                    trade=item.get("trade", ""),
                    region=region or "tier1",
                )
                if _rs_result and _rs_result.get("rate", 0) > 0:
                    item["rate_inr"] = _rs_result["rate"]
                    item["rate_source"] = _rs_result.get("source", "rate_registry")
                    item["rate_confidence"] = _rs_result.get("confidence", 0.5)
            except Exception:
                pass

        item["rate_region"] = region
        result_items.append(item)

    if unmatched:
        logger.debug(
            "apply_rates: %d items unmatched — first few: %s",
            len(unmatched), unmatched[:3],
        )

    return result_items


def compute_trade_summary(rated_items: List[dict]) -> Dict[str, dict]:
    """
    Group rated BOQ items by trade and compute subtotals.

    Args:
        rated_items: Items with rate_inr / amount_inr fields (output of apply_rates).

    Returns:
        Dict keyed by trade name:
        {
            trade_name: {
                "item_count":    int,
                "total_amount":  float,   # INR
                "items":         List[dict],
            }
        }
    """
    summary: Dict[str, dict] = {}

    for item in rated_items:
        trade: str = str(item.get("trade") or "Unclassified").strip()
        amount: float = float(item.get("amount_inr") or 0.0)

        if trade not in summary:
            summary[trade] = {
                "item_count":   0,
                "total_amount": 0.0,
                "items":        [],
            }

        summary[trade]["item_count"] += 1
        summary[trade]["total_amount"] = round(
            summary[trade]["total_amount"] + amount, 2
        )
        summary[trade]["items"].append(item)

    # Sort summary by total_amount descending for easy scanning
    summary = dict(
        sorted(summary.items(), key=lambda kv: kv[1]["total_amount"], reverse=True)
    )

    return summary


def rate_summary_text(summary: Dict[str, dict]) -> str:
    """
    Format a trade summary as a readable text block (for logging / debug output).

    Args:
        summary: Output of compute_trade_summary().

    Returns:
        Multi-line string with trade totals and grand total.
    """
    lines = ["Rate Summary by Trade", "=" * 52]
    grand_total = 0.0

    for trade, data in summary.items():
        total = data["total_amount"]
        count = data["item_count"]
        grand_total += total
        lines.append(
            f"  {trade:<30}  {count:>3} items   INR {total:>14,.0f}"
        )

    lines.append("-" * 52)
    lines.append(f"  {'GRAND TOTAL':<30}           INR {grand_total:>14,.0f}")
    return "\n".join(lines)
