"""
BOQ Text -> Taxonomy Item Matcher

Matches free-text BOQ line item descriptions to taxonomy items
using a combination of:
1. Exact alias matching (fastest, highest precision)
2. Token overlap scoring with TF-IDF-like weighting (primary strategy)

Design principles:
- Prefer NO match over a WRONG match (precision > recall)
- Only use taxonomy item names + aliases for matching (NOT synonyms — those
  have ambiguous many-to-many mappings that cause false positives)
- Score multiple candidates, pick the best, require minimum threshold
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a BOQ text to taxonomy items."""
    input_text: str
    matched: bool
    taxonomy_id: str = ""           # e.g. "STR.RCC.FTG.001"
    canonical_name: str = ""        # e.g. "RCC M20 Isolated Footing"
    discipline: str = ""            # e.g. "structural"
    trade: str = ""                 # e.g. "rcc_concrete"
    unit: str = ""                  # e.g. "cum"
    confidence: float = 0.0         # 0.0 - 1.0
    match_method: str = ""          # "exact_alias", "token_overlap", "best_guess", "keyword_fallback"
    matched_alias: str = ""         # The specific alias/tokens that matched
    suggested: bool = False         # True = below-threshold best-guess or keyword fallback


def _item_get(item, key, default=""):
    """Access a taxonomy item field by name, supporting both dataclass and dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


# Tokens that appear in nearly every BOQ description — zero matching value
_STOPWORDS = frozenset({
    # English function words
    'and', 'or', 'the', 'of', 'in', 'for', 'to', 'with', 'at', 'by', 'on',
    'as', 'per', 'all', 'from', 'this', 'that', 'shall', 'be', 'is', 'are',
    'a', 'an', 'its', 'their', 'any', 'each', 'not', 'no', 'do',
    # Construction action verbs (appear in 60-80% of BOQ line items)
    'providing', 'laying', 'fixing', 'supplying', 'including', 'etc',
    'provide', 'supply', 'fix', 'lay', 'install', 'erect', 'construct',
    'apply', 'finish', 'coat', 'prepare', 'clean', 'treat', 'seal',
    'using', 'making', 'forming', 'cutting', 'fitting', 'placing',
    # BOQ boilerplate (appear in 40-70% of descriptions)
    'work', 'works', 'item', 'items', 'type', 'types', 'grade',
    'complete', 'necessary', 'required', 'specified', 'approved',
    'make', 'brand', 'equivalent', 'similar', 'quality',
    # Condition / state descriptors (low matching signal)
    'new', 'existing', 'old', 'original', 'temporary', 'permanent',
    'above', 'below', 'over', 'under', 'between',
    # Location noise words
    'site', 'area', 'zone', 'location', 'floor', 'level', 'place',
    # Financial noise (sometimes appear in description field)
    'rate', 'cost', 'price', 'amount',
    # Directional (no signal)
    'both', 'either', 'wherever', 'throughout', 'wherever',
})

# High-value domain tokens — these carry strong signal when matched
_DOMAIN_TOKENS = frozenset({
    # Structural
    'rcc', 'pcc', 'concrete', 'footing', 'column', 'beam', 'slab', 'raft',
    'lintel', 'chajja', 'sunshade', 'parapet', 'staircase', 'retaining',
    'reinforcement', 'tmt', 'fe500d', 'fe500', 'formwork', 'shuttering',
    # Civil
    'excavation', 'earthwork', 'filling', 'compaction', 'dewatering',
    'piling', 'termite', 'demolition', 'shoring',
    # Masonry
    'brick', 'block', 'aac', 'masonry', 'mortar', 'partition',
    # Waterproofing
    'waterproofing', 'membrane', 'app', 'sbs', 'bitumen', 'cementitious',
    'dpc', 'tanking', 'crystalline',
    # Finishes
    'plaster', 'tile', 'tiles', 'vitrified', 'ceramic', 'granite', 'marble',
    'terrazzo', 'epoxy', 'flooring', 'skirting', 'dado', 'cladding',
    'painting', 'emulsion', 'primer', 'putty', 'distemper', 'texture',
    'wallpaper', 'laminate', 'veneer',
    # Ceiling
    'ceiling', 'gypsum', 'acoustic', 'mineral', 'baffle',
    # Doors/Windows
    'door', 'window', 'flush', 'panel', 'shutter', 'grille', 'grill',
    'aluminium', 'aluminum', 'upvc', 'sliding', 'casement',
    'glass', 'toughened', 'laminated', 'dgu',
    'hinge', 'lock', 'closer', 'handle', 'bolt',
    # Plumbing
    'pipe', 'cpvc', 'ppr', 'hdpe', 'pvc', 'upvc',
    'ewc', 'basin', 'urinal', 'bathtub', 'shower', 'tap', 'mixer',
    'valve', 'cistern', 'faucet', 'geyser', 'pump', 'stp',
    # Electrical
    'wiring', 'conduit', 'cable', 'mcb', 'mccb', 'rccb', 'panel',
    'switchboard', 'earthing', 'lightning',
    'led', 'light', 'fan', 'exhaust', 'socket', 'switch',
    'solar', 'inverter', 'ups', 'generator', 'transformer',
    'cctv', 'intercom', 'bms',
    # Fire
    'sprinkler', 'hydrant', 'detector', 'alarm', 'extinguisher',
    'suppression', 'riser', 'hose',
    # HVAC
    'hvac', 'chiller', 'ahu', 'duct', 'diffuser', 'thermostat',
    'split', 'cassette', 'vrf', 'vrv', 'ventilation',
    # Elevator
    'elevator', 'lift', 'escalator', 'dumbwaiter',
    # External
    'road', 'paving', 'kerb', 'drain', 'manhole', 'fencing',
    'gate', 'landscape', 'irrigation',
    # Facade
    'acp', 'curtain', 'glazing', 'spider', 'louver',
})

# Maps domain tokens to their discipline — used for keyword fallback when no
# scored match is found. Ordered: first hit wins (most specific listed first).
_DOMAIN_TO_DISCIPLINE: Dict[str, str] = {
    # Structural
    'footing': 'structural', 'foundation': 'structural', 'raft': 'structural',
    'pile': 'structural', 'piling': 'structural', 'rcc': 'structural',
    'pcc': 'structural', 'concrete': 'structural', 'reinforcement': 'structural',
    'tmt': 'structural', 'fe500': 'structural', 'fe500d': 'structural',
    'formwork': 'structural', 'shuttering': 'structural',
    'column': 'structural', 'beam': 'structural', 'slab': 'structural',
    'lintel': 'structural', 'chajja': 'structural', 'parapet': 'structural',
    'staircase': 'structural', 'retaining': 'structural',
    # Civil / Earthwork
    'excavation': 'civil', 'earthwork': 'civil', 'filling': 'civil',
    'compaction': 'civil', 'dewatering': 'civil', 'termite': 'civil',
    'demolition': 'civil', 'shoring': 'civil',
    'road': 'civil', 'paving': 'civil', 'kerb': 'civil',
    'drain': 'civil', 'manhole': 'civil', 'fencing': 'civil',
    'gate': 'civil', 'landscape': 'civil', 'irrigation': 'civil',
    # Masonry
    'brick': 'masonry', 'block': 'masonry', 'aac': 'masonry',
    'masonry': 'masonry', 'mortar': 'masonry', 'partition': 'masonry',
    # Waterproofing
    'waterproofing': 'waterproofing', 'membrane': 'waterproofing',
    'app': 'waterproofing', 'sbs': 'waterproofing', 'bitumen': 'waterproofing',
    'cementitious': 'waterproofing', 'dpc': 'waterproofing',
    'tanking': 'waterproofing', 'crystalline': 'waterproofing',
    # Finishes / Flooring
    'plaster': 'finishes', 'tile': 'finishes', 'tiles': 'finishes',
    'vitrified': 'finishes', 'ceramic': 'finishes', 'granite': 'finishes',
    'marble': 'finishes', 'terrazzo': 'finishes', 'epoxy': 'finishes',
    'flooring': 'finishes', 'skirting': 'finishes', 'dado': 'finishes',
    'cladding': 'finishes', 'painting': 'finishes', 'emulsion': 'finishes',
    'primer': 'finishes', 'putty': 'finishes', 'distemper': 'finishes',
    'texture': 'finishes', 'wallpaper': 'finishes', 'laminate': 'finishes',
    'veneer': 'finishes',
    # Ceiling
    'ceiling': 'ceiling', 'gypsum': 'ceiling', 'acoustic': 'ceiling',
    'mineral': 'ceiling', 'baffle': 'ceiling',
    # Doors / Windows / Glazing
    'door': 'architectural', 'window': 'architectural', 'flush': 'architectural',
    'shutter': 'architectural', 'grille': 'architectural', 'grill': 'architectural',
    'aluminium': 'architectural', 'aluminum': 'architectural',
    'upvc': 'architectural', 'sliding': 'architectural', 'casement': 'architectural',
    'glass': 'architectural', 'toughened': 'architectural',
    'laminated': 'architectural', 'dgu': 'architectural',
    'hinge': 'architectural', 'lock': 'architectural', 'closer': 'architectural',
    # Facade
    'acp': 'facade', 'curtain': 'facade', 'glazing': 'facade',
    'spider': 'facade', 'louver': 'facade',
    # Plumbing / Sanitary
    'pipe': 'plumbing', 'cpvc': 'plumbing', 'ppr': 'plumbing',
    'hdpe': 'plumbing', 'pvc': 'plumbing',
    'ewc': 'plumbing', 'basin': 'plumbing', 'urinal': 'plumbing',
    'bathtub': 'plumbing', 'shower': 'plumbing', 'tap': 'plumbing',
    'mixer': 'plumbing', 'valve': 'plumbing', 'cistern': 'plumbing',
    'faucet': 'plumbing', 'geyser': 'plumbing', 'pump': 'plumbing',
    'stp': 'plumbing',
    # Electrical
    'wiring': 'electrical', 'conduit': 'electrical', 'cable': 'electrical',
    'mcb': 'electrical', 'mccb': 'electrical', 'rccb': 'electrical',
    'switchboard': 'electrical', 'earthing': 'electrical', 'lightning': 'electrical',
    'led': 'electrical', 'light': 'electrical', 'fan': 'electrical',
    'exhaust': 'electrical', 'socket': 'electrical', 'switch': 'electrical',
    'solar': 'electrical', 'inverter': 'electrical', 'ups': 'electrical',
    'generator': 'electrical', 'transformer': 'electrical',
    'cctv': 'electrical', 'intercom': 'electrical', 'bms': 'electrical',
    # Fire Protection
    'sprinkler': 'fire', 'hydrant': 'fire', 'detector': 'fire',
    'alarm': 'fire', 'extinguisher': 'fire', 'suppression': 'fire',
    'riser': 'fire', 'hose': 'fire',
    # HVAC
    'hvac': 'hvac', 'chiller': 'hvac', 'ahu': 'hvac', 'duct': 'hvac',
    'diffuser': 'hvac', 'thermostat': 'hvac', 'split': 'hvac',
    'cassette': 'hvac', 'vrf': 'hvac', 'vrv': 'hvac', 'ventilation': 'hvac',
    # Elevator
    'elevator': 'elevator', 'lift': 'elevator',
    'escalator': 'elevator', 'dumbwaiter': 'elevator',
}

# ---------------------------------------------------------------------------
# Unit-family compatibility for BOQ unit-context enrichment
# ---------------------------------------------------------------------------

_UNIT_FAMILIES: Dict[str, str] = {
    # VOLUME
    "cum": "VOLUME",   "m3": "VOLUME",    "cuft": "VOLUME",  "cft": "VOLUME",
    # AREA
    "sqm": "AREA",     "m2": "AREA",      "sqft": "AREA",    "sft": "AREA",
    "sq.m": "AREA",    "sq.ft": "AREA",
    # LINEAR
    "rmt": "LINEAR",   "rm": "LINEAR",    "rft": "LINEAR",   "m": "LINEAR",
    "mtr": "LINEAR",   "lm": "LINEAR",    "lft": "LINEAR",   "rlm": "LINEAR",
    # COUNT
    "nos": "COUNT",    "no": "COUNT",     "nr": "COUNT",     "each": "COUNT",
    "ea": "COUNT",     "set": "COUNT",    "pair": "COUNT",   "point": "COUNT",
    "floor": "COUNT",
    # WEIGHT
    "kg": "WEIGHT",    "mt": "WEIGHT",    "tonne": "WEIGHT", "ton": "WEIGHT",
    "quintal": "WEIGHT",
    # LUMP — ambiguous by nature; neither boost nor penalty
    "ls": "LUMP",      "lump": "LUMP",    "lumpsum": "LUMP", "job": "LUMP",
    "lot": "LUMP",     "allow": "LUMP",   "day": "LUMP",     "month": "LUMP",
    "trip": "LUMP",    "roll": "LUMP",    "kwp": "LUMP",     "acre": "LUMP",
    "litre": "LUMP",   "ltr": "LUMP",
    # ── Phase-1 additions: wild / regional unit variants ──────────────────
    # AREA extras
    "m²":   "AREA",    "sqmt": "AREA",    "sqmtr": "AREA",
    "sqyd": "AREA",    "syd":  "AREA",
    # COUNT extras
    "pcs":  "COUNT",   "pc":   "COUNT",
    "unit": "COUNT",   "units":"COUNT",
    "bundle":"COUNT",  "bag":  "COUNT",   "bags": "COUNT",
    # LINEAR extras
    "mtrs": "LINEAR",  "r/f":  "LINEAR",  "rf":   "LINEAR",
    # WEIGHT extras
    "kgs":  "WEIGHT",  "gm":   "WEIGHT",  "gms":  "WEIGHT",
    # LUMP extras (electrical / special measures)
    "kva":  "LUMP",    "kw":   "LUMP",    "hp":   "LUMP",
    "liters":"LUMP",   "litres":"LUMP",
}

_UNIT_BOOST_SAME    = 1.4   # explicit unit, same family  → strong positive signal
_UNIT_PENALTY_DIFF  = 0.5   # explicit unit, diff family  → strong negative signal
_UNIT_BOOST_INFER   = 1.2   # inferred family, same       → soft  positive signal
_UNIT_PENALTY_INFER = 0.7   # inferred family, diff       → soft  negative signal
_SECTION_BOOST_SAME   = 1.15  # section discipline matches taxonomy → very soft boost
_SECTION_PENALTY_DIFF = 0.85  # section discipline conflicts        → very soft penalty


def _unit_family(unit: str) -> str:
    """Return family name for a unit string, or '' if unknown/missing.

    Normalises to lowercase and strips trailing punctuation
    (e.g. ``"rmt."`` → ``"rmt"``).  Returns ``""`` for empty or
    unrecognised units — callers treat ``""`` as "no information"
    and apply a factor of 1.0 (backward-compatible).
    """
    if not unit:
        return ""
    return _UNIT_FAMILIES.get(unit.lower().strip().rstrip("."), "")


def _unit_compat(boq_unit: str, tax_unit: str) -> float:
    """Return the score multiplier for a BOQ-unit vs taxonomy-unit pair.

    Rules (in priority order):

    1. Either side unknown/missing  → 1.0  (no information, no change)
    2. Either family is LUMP        → 1.0  (lump items are inherently vague)
    3. Same family                  → ``_UNIT_BOOST_SAME``   (1.4)
    4. Different family             → ``_UNIT_PENALTY_DIFF`` (0.5)
    """
    fam_boq = _unit_family(boq_unit)
    fam_tax = _unit_family(tax_unit)

    if not fam_boq or not fam_tax:              # Rule 1: unknown
        return 1.0
    if fam_boq == "LUMP" or fam_tax == "LUMP":  # Rule 2: lump = ambiguous
        return 1.0
    return _UNIT_BOOST_SAME if fam_boq == fam_tax else _UNIT_PENALTY_DIFF


# ---------------------------------------------------------------------------
# Description-token → unit-family inference (Phase 2)
# ---------------------------------------------------------------------------
# Maps individual construction-domain tokens to the unit family most commonly
# used to measure that type of work.  Used when a BOQ item carries no explicit
# unit — the inferred family guides scoring with softer multipliers (1.2/0.7)
# so it merely tips ambiguous candidates rather than overriding text ranking.
#
# Design principles:
#   • Only include tokens with a *clear, dominant* unit family.  Ambiguous
#     tokens (e.g. "insulation" → wall AREA *or* pipe LINEAR) are omitted.
#   • Plural/singular both listed for noisy real-world descriptions.
#   • Votes are additive — "concrete rcc m25" scores 2× VOLUME which beats
#     a single AREA vote from an overlapping generic token.

_TOKEN_TO_UNIT_FAMILY: Dict[str, str] = {
    # VOLUME — items typically measured in cubic metres (cum / m³)
    "concrete":    "VOLUME", "concreting":  "VOLUME", "rcc":         "VOLUME",
    "pcc":         "VOLUME", "excavation":  "VOLUME", "earthwork":   "VOLUME",
    "earthworks":  "VOLUME", "backfill":    "VOLUME", "backfilling": "VOLUME",
    "embankment":  "VOLUME", "grouting":    "VOLUME", "compaction":  "VOLUME",
    "compacted":   "VOLUME",
    # AREA — items typically measured in square metres (sqm / m²)
    "painting":    "AREA",   "waterproofing":"AREA",  "tiling":      "AREA",
    "tile":        "AREA",   "tiles":        "AREA",  "flooring":    "AREA",
    "plaster":     "AREA",   "plastering":   "AREA",  "screed":      "AREA",
    "carpet":      "AREA",   "vinyl":        "AREA",  "cladding":    "AREA",
    "lining":      "AREA",   "membrane":     "AREA",  "dpc":         "AREA",
    "panelling":   "AREA",   "paneling":     "AREA",  "rendering":   "AREA",
    "render":      "AREA",   "whitewash":    "AREA",  "distemper":   "AREA",
    "glazing":     "AREA",   "primer":       "AREA",  "enamel":      "AREA",
    "stucco":      "AREA",   "coating":      "AREA",
    # LINEAR — items typically measured in running metres (rmt / m)
    "pipe":        "LINEAR", "piping":      "LINEAR", "pipeline":    "LINEAR",
    "cable":       "LINEAR", "cables":      "LINEAR", "conduit":     "LINEAR",
    "duct":        "LINEAR", "ducting":     "LINEAR", "drainage":    "LINEAR",
    "gutter":      "LINEAR", "handrail":    "LINEAR", "railing":     "LINEAR",
    "wire":        "LINEAR", "wiring":      "LINEAR", "tube":        "LINEAR",
    "fencing":     "LINEAR", "kerb":        "LINEAR", "dado":        "LINEAR",
    "skirting":    "LINEAR", "trunking":    "LINEAR", "channel":     "LINEAR",
    # COUNT — items typically measured in numbers / each (nos / ea)
    "door":        "COUNT",  "doors":       "COUNT",  "window":      "COUNT",
    "windows":     "COUNT",  "fitting":     "COUNT",  "fittings":    "COUNT",
    "fixture":     "COUNT",  "fixtures":    "COUNT",  "fan":         "COUNT",
    "fans":        "COUNT",  "switch":      "COUNT",  "socket":      "COUNT",
    "tap":         "COUNT",  "valve":       "COUNT",  "valves":      "COUNT",
    "manhole":     "COUNT",  "manholes":    "COUNT",  "pump":        "COUNT",
    "pumps":       "COUNT",  "cabinet":     "COUNT",  "sign":        "COUNT",
    "gully":       "COUNT",  "light":       "COUNT",  "trap":        "COUNT",
    # WEIGHT — items typically measured in kg or tonne
    "steel":       "WEIGHT", "rebar":       "WEIGHT", "reinforcement":"WEIGHT",
    "tmt":         "WEIGHT", "structural":  "WEIGHT", "bar":         "WEIGHT",
    "bars":        "WEIGHT",
}


def _infer_unit_family(text_tokens: Set[str]) -> str:
    """Infer the most likely unit family from BOQ description tokens.

    Counts votes from :data:`_TOKEN_TO_UNIT_FAMILY` for each token present in
    *text_tokens*.  Returns the winning family when there is an unambiguous
    majority, or ``""`` on a tie / no votes.

    Only called when the BOQ item carries no explicit unit.  The result is used
    with softer multipliers (1.2 / 0.7) rather than the explicit-unit factors
    (1.4 / 0.5) so inference merely nudges ambiguous candidates.
    """
    votes: Dict[str, int] = {}
    for tok in text_tokens:
        fam = _TOKEN_TO_UNIT_FAMILY.get(tok)
        if fam:
            votes[fam] = votes.get(fam, 0) + 1
    if not votes:
        return ""
    max_votes = max(votes.values())
    winners = [f for f, v in votes.items() if v == max_votes]
    return winners[0] if len(winners) == 1 else ""


# ---------------------------------------------------------------------------
# Section-header → taxonomy-discipline mapping (Phase 3)
# ---------------------------------------------------------------------------
# Maps single tokens found in BOQ section headers to the taxonomy discipline
# codes used in _DOMAIN_TO_DISCIPLINE.  Kept deliberately conservative:
# only tokens with a *single*, *unambiguous* discipline mapping are listed
# (e.g. "plumbing" → "plumbing") to avoid false penalisation.
#
# Multipliers are the softest of the three signals (1.15 / 0.85) because
# section headers are broad — a "CIVIL WORKS" section may include items
# from structural, masonry or waterproofing sub-trades.

_SECTION_KEYWORDS: Dict[str, str] = {
    # ── structural (02 structural_rcc, 03 structural_steel, 04 structural_precast)
    "structural":    "structural",  "structure":     "structural",
    # ── civil (01 civil_earthwork)
    "civil":         "civil",       "earthwork":     "civil",
    "earthworks":    "civil",       "sitework":      "civil",
    "excavation":    "civil",
    # ── architectural — masonry, waterproofing, wall/ceiling finishes,
    #                    doors/windows, facade, interior fitout
    #    (05 masonry, 06 waterproofing, 08 finishes_wall, 09 finishes_ceiling,
    #     10 doors_windows, 18 facade, 19 interior_fitout)
    "masonry":       "architectural", "brickwork":   "architectural",
    "blockwork":     "architectural",
    "waterproofing": "architectural",
    "ceiling":       "architectural",
    "architectural": "architectural", "joinery":     "architectural",
    "carpentry":     "architectural", "glazing":     "architectural",
    "facade":        "architectural", "cladding":    "architectural",
    "interior":      "architectural", "fitout":      "architectural",
    "painting":      "architectural",  # wall painting → architectural
    # ── finishing — floor finishes ONLY (07 finishes_floor)
    "finishing":     "finishing",   "finishes":      "finishing",
    "flooring":      "finishing",
    # ── mep — plumbing, electrical, fire safety, HVAC, elevator
    #    (11 plumbing, 12 electrical, 13 fire_safety, 14 hvac, 15 elevator)
    "plumbing":      "mep",         "sanitary":      "mep",
    "drainage":      "mep",
    "electrical":    "mep",         "power":         "mep",
    "lighting":      "mep",
    "fire":          "mep",
    "hvac":          "mep",         "mechanical":    "mep",
    "ventilation":   "mep",         "ductwork":      "mep",
    "elevator":      "mep",         "lift":          "mep",
    "escalator":     "mep",
}


def _section_discipline(section_text: str) -> str:
    """Normalise a BOQ section-header string to a taxonomy discipline code.

    Tokenises *section_text*, counts votes from :data:`_SECTION_KEYWORDS`,
    and returns the clear single winner, or ``""`` on tie / no votes.

    Examples::

        _section_discipline("3.0 STRUCTURAL WORKS")  → "structural"
        _section_discipline("PART B – PLUMBING")     → "plumbing"
        _section_discipline("GENERAL ITEMS")         → ""   (no keyword)
        _section_discipline("pipe tile door")        → ""   (three-way tie)
    """
    if not section_text:
        return ""
    votes: Dict[str, int] = {}
    for tok in _tokenize(section_text.lower()):
        disc = _SECTION_KEYWORDS.get(tok)
        if disc:
            votes[disc] = votes.get(disc, 0) + 1
    if not votes:
        return ""
    max_v = max(votes.values())
    winners = [d for d, v in votes.items() if v == max_v]
    return winners[0] if len(winners) == 1 else ""


class TaxonomyMatcher:
    """Matches BOQ text descriptions to taxonomy items."""

    def __init__(self):
        self._alias_index: Dict[str, Any] = {}      # lowercase alias -> taxonomy item
        self._token_index: Dict[str, Set[str]] = {}  # token -> set of item IDs
        self._items_by_id: Dict[str, Any] = {}       # ID -> taxonomy item
        self._item_token_sets: Dict[str, Set[str]] = {}  # ID -> set of all tokens
        self._token_doc_freq: Dict[str, int] = {}    # token -> number of items containing it
        self._total_items: int = 0
        self._loaded = False

    def load(self):
        """Load taxonomy items, build indexes. No synonym mapping (too noisy)."""
        if self._loaded:
            return

        from src.knowledge_base import get_taxonomy_items

        items = get_taxonomy_items()
        self._total_items = len(items)

        for item in items:
            item_id = _item_get(item, "id", "")
            if not item_id:
                continue
            self._items_by_id[item_id] = item

            # Collect all text representations of this item
            all_text_sources = []

            name = _item_get(item, "standard_name", "").lower().strip()
            if name:
                self._alias_index[name] = item
                all_text_sources.append(name)

            for alias in _item_get(item, "aliases", []):
                alias_lower = alias.lower().strip()
                if alias_lower:
                    # Only set if not already taken (first item wins)
                    if alias_lower not in self._alias_index:
                        self._alias_index[alias_lower] = item
                    all_text_sources.append(alias_lower)

            # Build token set for this item (from name + all aliases)
            item_tokens = set()
            for src_text in all_text_sources:
                for token in _tokenize(src_text):
                    if len(token) > 1:
                        item_tokens.add(token)

            self._item_token_sets[item_id] = item_tokens

            # Update token index and doc frequency
            for token in item_tokens:
                self._token_index.setdefault(token, set()).add(item_id)
                self._token_doc_freq[token] = self._token_doc_freq.get(token, 0) + 1

        self._loaded = True
        logger.info("TaxonomyMatcher loaded: %d aliases, %d tokens, %d items",
                     len(self._alias_index), len(self._token_index), len(self._items_by_id))

    def match(self, text: str, min_confidence: float = 0.5,
              unit: str = "", section: str = "") -> MatchResult:
        """
        Match a single BOQ text description to the best taxonomy item.

        Strategy:
        1. Try exact alias match (full cleaned text == an alias)
        2. Score ALL candidate items by weighted token overlap, pick the best

        Returns MatchResult with matched=False if nothing exceeds min_confidence.
        """
        self.load()

        if not text or not text.strip():
            return MatchResult(input_text=text, matched=False)

        text_clean = _clean_text(text)
        text_lower = text_clean.lower().strip()

        if not text_lower:
            return MatchResult(input_text=text, matched=False)

        # ── Strategy 1: Exact alias match ──
        if text_lower in self._alias_index:
            item = self._alias_index[text_lower]
            return _make_result(text, item, 0.95, "exact_alias", text_lower)

        # ── Strategy 2: Weighted token overlap ──
        text_tokens = set(_tokenize(text_lower))
        if not text_tokens:
            return MatchResult(input_text=text, matched=False)

        # When the caller supplies no explicit unit, try to infer the unit
        # family from the description tokens.  The inferred family uses softer
        # multipliers (1.2 / 0.7) than an explicit unit (1.4 / 0.5) so it
        # only nudges ambiguous candidates rather than overriding text ranking.
        inferred_unit_fam: str = "" if unit else _infer_unit_family(text_tokens)

        # Resolve BOQ section header to a taxonomy discipline code (once, pre-loop).
        # Used with very soft multipliers (1.15 / 0.85) applied after the unit signal.
        section_disc: str = _section_discipline(section)

        # Gather candidate items: any item sharing at least one token
        candidate_ids: Dict[str, float] = {}  # item_id -> raw overlap count
        for token in text_tokens:
            if token in self._token_index:
                for item_id in self._token_index[token]:
                    candidate_ids[item_id] = candidate_ids.get(item_id, 0) + 1

        if not candidate_ids:
            return MatchResult(input_text=text, matched=False)

        # Score top candidates properly (only score top 20 by raw overlap)
        top_candidates = sorted(candidate_ids.items(), key=lambda x: -x[1])[:20]

        best_score = 0.0
        best_item = None
        best_overlap_tokens = set()

        for item_id, raw_count in top_candidates:
            item_tokens = self._item_token_sets.get(item_id, set())
            if not item_tokens:
                continue

            overlap = text_tokens & item_tokens
            if not overlap:
                continue

            # Weighted overlap: domain tokens count more, rare tokens count more
            weighted_overlap = 0.0
            for token in overlap:
                # IDF-like weight: rare tokens are more informative
                doc_freq = self._token_doc_freq.get(token, 1)
                idf = math.log(max(self._total_items, 1) / max(doc_freq, 1))
                # Domain boost: known construction terms get 2x
                domain_boost = 2.0 if token in _DOMAIN_TOKENS else 1.0
                weighted_overlap += idf * domain_boost

            # Normalize by the size of both token sets
            weighted_text_total = 0.0
            for token in text_tokens:
                doc_freq = self._token_doc_freq.get(token, 1)
                idf = math.log(max(self._total_items, 1) / max(doc_freq, 1))
                domain_boost = 2.0 if token in _DOMAIN_TOKENS else 1.0
                weighted_text_total += idf * domain_boost

            weighted_item_total = 0.0
            for token in item_tokens:
                doc_freq = self._token_doc_freq.get(token, 1)
                idf = math.log(max(self._total_items, 1) / max(doc_freq, 1))
                domain_boost = 2.0 if token in _DOMAIN_TOKENS else 1.0
                weighted_item_total += idf * domain_boost

            # Asymmetric scoring: we care more about covering the input text
            # than covering the item (input is short, item name is detailed)
            text_recall = weighted_overlap / max(weighted_text_total, 0.01)
            item_precision = weighted_overlap / max(weighted_item_total, 0.01)

            # F-beta with beta=2 (favor recall of input tokens)
            beta = 2.0
            if text_recall + item_precision > 0:
                score = (1 + beta**2) * (item_precision * text_recall) / (beta**2 * item_precision + text_recall)
            else:
                score = 0.0

            # Require at least 2 overlapping domain tokens, or 1 domain + 2 others
            domain_overlap = overlap & _DOMAIN_TOKENS
            non_domain_overlap = overlap - _DOMAIN_TOKENS
            if len(domain_overlap) == 0 and len(non_domain_overlap) < 3:
                score *= 0.3  # Heavy penalty for no domain token match

            # Unit-family compatibility multiplier.
            # Runs after all text-based scoring so unit never creates a
            # match from nothing — it only amplifies or suppresses an
            # existing text-score signal.
            #
            # Two signal strengths:
            #   • Explicit BOQ unit  → strong  (1.4 boost / 0.5 penalty)  via _unit_compat()
            #   • Inferred from desc → soft    (1.2 boost / 0.7 penalty)  computed here
            tax_unit = _item_get(self._items_by_id.get(item_id, {}), "unit", "")
            if unit:
                # Caller supplied an explicit unit — use strong multiplier.
                score = score * _unit_compat(unit, tax_unit)
            elif inferred_unit_fam:
                # No explicit unit; apply softer inference-based multiplier.
                fam_tax = _unit_family(tax_unit)
                if fam_tax and fam_tax != "LUMP" and inferred_unit_fam != "LUMP":
                    factor = (
                        _UNIT_BOOST_INFER
                        if inferred_unit_fam == fam_tax
                        else _UNIT_PENALTY_INFER
                    )
                    score = score * factor

            # Section-discipline multiplier (softest signal — applied last).
            # Only active when the caller supplied a recognisable section header.
            if section_disc:
                tax_disc = _item_get(self._items_by_id.get(item_id, {}), "discipline", "")
                if tax_disc:
                    factor = (
                        _SECTION_BOOST_SAME
                        if section_disc == tax_disc
                        else _SECTION_PENALTY_DIFF
                    )
                    score = score * factor

            if score > best_score:
                best_score = score
                best_item = self._items_by_id.get(item_id)
                best_overlap_tokens = overlap

        # ── 100% coverage: always return something useful ──────────────
        # Three tiers below exact alias match:
        #   1. token_overlap  — score >= min_confidence  → confirmed match
        #   2. best_guess     — 0 < score < min_confidence → best available
        #   3. keyword_fallback — no candidate items at all → discipline only

        if best_item is not None and best_score >= min_confidence:
            # Confirmed match — cap at 0.90 (never as certain as exact alias)
            confidence = min(0.90, best_score)
            return _make_result(
                text, best_item, confidence, "token_overlap",
                ", ".join(sorted(best_overlap_tokens)),
            )

        if best_item is not None and best_score > 0:
            # Below threshold but still the closest taxonomy item found.
            # Mark as suggested so downstream can decide how to use it.
            confidence = min(0.45, best_score)   # cap below confirmed range
            result = _make_result(
                text, best_item, confidence, "best_guess",
                ", ".join(sorted(best_overlap_tokens)),
            )
            result.suggested = True
            return result

        # No token overlap at all — fall back to keyword-based discipline detection.
        # Scan the input tokens for known domain keywords and vote on discipline.
        discipline_votes: Dict[str, int] = {}
        for token in text_tokens:
            disc = _DOMAIN_TO_DISCIPLINE.get(token)
            if disc:
                discipline_votes[disc] = discipline_votes.get(disc, 0) + 1

        if discipline_votes:
            top_disc = max(discipline_votes, key=lambda d: discipline_votes[d])
            return MatchResult(
                input_text=text,
                matched=True,
                discipline=top_disc,
                confidence=0.15,
                match_method="keyword_fallback",
                matched_alias=", ".join(
                    t for t in text_tokens if _DOMAIN_TO_DISCIPLINE.get(t) == top_disc
                ),
                suggested=True,
            )

        # Truly unrecognisable text (no domain tokens, no token overlap)
        return MatchResult(input_text=text, matched=False)

    def match_batch(
        self,
        texts: List[str],
        min_confidence: float = 0.5,
        units: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
    ) -> List[MatchResult]:
        """Match a batch of BOQ text descriptions.

        Args:
            texts:          List of BOQ description strings.
            min_confidence: Minimum score to treat as a confirmed match.
            units:          Optional list of unit strings parallel to *texts*
                            (e.g. ``["cum", "sqm", "nos"]``).  If provided,
                            must be the same length as *texts*.  Missing or
                            ``None`` entries default to ``""`` (no unit context).
            sections:       Optional list of BOQ section-header strings parallel
                            to *texts* (e.g. ``["STRUCTURAL WORKS", "", ...]``).
                            Used as a soft discipline hint (×1.15 / ×0.85).
                            Omit or pass ``None`` to disable section context.
        """
        self.load()
        _units    = units    if units    is not None else [""] * len(texts)
        _sections = sections if sections is not None else [""] * len(texts)
        return [
            self.match(t, min_confidence, u or "", s or "")
            for t, u, s in zip(texts, _units, _sections)
        ]

    def get_match_stats(self, results: List[MatchResult]) -> Dict[str, Any]:
        """Get summary statistics for a batch of match results."""
        total = len(results)
        matched = sum(1 for r in results if r.matched)
        by_method: Dict[str, int] = {}
        by_discipline: Dict[str, int] = {}
        by_trade: Dict[str, int] = {}
        confidences: List[float] = []

        for r in results:
            if r.matched:
                confidences.append(r.confidence)
                by_method[r.match_method] = by_method.get(r.match_method, 0) + 1
                if r.discipline:
                    by_discipline[r.discipline] = by_discipline.get(r.discipline, 0) + 1
                if r.trade:
                    by_trade[r.trade] = by_trade.get(r.trade, 0) + 1

        return {
            "total": total,
            "matched": matched,
            "unmatched": total - matched,
            "match_rate": matched / max(total, 1),
            "avg_confidence": sum(confidences) / max(len(confidences), 1),
            "by_method": by_method,
            "by_discipline": by_discipline,
            "by_trade": by_trade,
        }


# ── Helpers ──

def _clean_text(text: str) -> str:
    """Clean BOQ text for matching — strip quantities, dimensions, rates."""
    # Remove dimensions with units
    text = re.sub(
        r'\b\d+(\.\d+)?\s*(sqm|sqft|cum|rmt|rm|nos|kg|mt|ltr|m2|m3|mm|cm|m)\b',
        '', text, flags=re.IGNORECASE,
    )
    # Remove standalone numbers (quantities)
    text = re.sub(r'\b\d+(\.\d+)?\b', '', text)
    # Remove rate info
    text = re.sub(r'@?\s*rs\.?\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'rate\s*:\s*\d+', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenize text into meaningful words, filtering stopwords."""
    tokens = re.split(r'[^a-z0-9]+', text.lower())
    return [t for t in tokens if t and len(t) > 1 and t not in _STOPWORDS]


def _make_result(text: str, item, confidence: float, method: str, matched_alias: str) -> MatchResult:
    """Create a MatchResult from a taxonomy item (dataclass or dict)."""
    return MatchResult(
        input_text=text,
        matched=True,
        taxonomy_id=_item_get(item, "id", ""),
        canonical_name=_item_get(item, "standard_name", ""),
        discipline=_item_get(item, "discipline", ""),
        trade=_item_get(item, "trade", ""),
        unit=_item_get(item, "unit", ""),
        confidence=round(confidence, 3),
        match_method=method,
        matched_alias=matched_alias,
    )


# ── Module-level convenience ──

_matcher_singleton = None


def get_matcher() -> TaxonomyMatcher:
    """Get the singleton TaxonomyMatcher instance."""
    global _matcher_singleton
    if _matcher_singleton is None:
        _matcher_singleton = TaxonomyMatcher()
    return _matcher_singleton


def match_boq_text(
    text: str,
    min_confidence: float = 0.3,
    unit: str = "",
    section: str = "",
) -> MatchResult:
    """Convenience function: match a single BOQ text.

    Args:
        text:           BOQ item description.
        min_confidence: Minimum confidence threshold.
        unit:           BOQ item unit of measurement (e.g. ``"cum"``, ``"sqm"``).
                        Omit or pass ``""`` to preserve backward-compatible behaviour.
        section:        BOQ section-header string above this item
                        (e.g. ``"3.0 STRUCTURAL WORKS"``).  Used as a very soft
                        discipline hint (×1.15 / ×0.85).  Omit or pass ``""``
                        to preserve backward-compatible behaviour.
    """
    return get_matcher().match(text, min_confidence, unit, section)


def match_boq_batch(
    texts: List[str],
    min_confidence: float = 0.3,
    units: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
) -> List[MatchResult]:
    """Convenience function: match a batch of BOQ texts.

    Args:
        texts:          List of BOQ descriptions.
        min_confidence: Minimum confidence threshold.
        units:          Optional list of unit strings parallel to *texts*.
                        Omit to preserve backward-compatible behaviour.
        sections:       Optional list of section-header strings parallel to
                        *texts*.  Omit to preserve backward-compatible behaviour.
    """
    return get_matcher().match_batch(texts, min_confidence, units, sections)
