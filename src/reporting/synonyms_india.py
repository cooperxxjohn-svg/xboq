"""
Indian Construction Synonyms Module
Maps standard BOQ terms to regional/colloquial variations used in India.

Used to increase recall when detecting scope items from OCR text.
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SYNONYM DICTIONARIES
# =============================================================================

# Standard term -> list of synonyms/variants
INDIAN_SYNONYMS: Dict[str, List[str]] = {
    # Earthwork
    "excavation": [
        "khudai", "khodai", "earthwork", "earth work", "foundation trench",
        "foundation trenches", "pit excavation", "pit digging", "cutting",
        "trench cutting", "foundation cutting", "earth cutting", "digging"
    ],

    # PCC / Blinding
    "pcc": [
        "blinding", "blinding concrete", "levelling course", "leveling course",
        "mud mat", "lean concrete", "bed concrete", "plain cement concrete",
        "plain concrete", "foundation bed", "base concrete", "pcc bed",
        "levelling bed", "leveling bed", "plain cc", "1:4:8", "1:3:6", "m10", "m7.5"
    ],

    # RCC Work
    "rcc": [
        "reinforced cement concrete", "reinforced concrete", "rcc work",
        "rcc concrete", "structural concrete", "cement concrete reinforced",
        "r.c.c.", "r.c.c"
    ],

    # Formwork
    "formwork": [
        "shuttering", "centering", "centering and shuttering",
        "shuttering and centering", "form work", "mould", "boxing",
        "timber shuttering", "plywood shuttering", "steel shuttering",
        "side shuttering", "column shuttering", "beam shuttering"
    ],

    # Reinforcement
    "reinforcement": [
        "saria", "sariya", "rebar", "rebars", "steel bar", "steel bars",
        "tmt", "tmt bar", "tmt bars", "fe500", "fe 500", "fe500d",
        "fe415", "fe 415", "hsd", "hsd bar", "hsd bars", "crs",
        "tor steel", "torsteel", "binding wire", "bar bending",
        "bbs", "bar bending schedule", "rebar schedule",
        "reinforcement steel", "steel reinforcement", "main steel",
        "distribution steel", "stirrups", "ties", "rings"
    ],

    # Backfill
    "backfill": [
        "refilling", "earth filling", "filling", "compaction",
        "ramming", "consolidation", "back fill", "back-fill",
        "soil filling", "sand filling", "murum filling",
        "earth compaction", "watering and compaction",
        "filling in foundation", "filling in plinth"
    ],

    # Disposal
    "disposal": [
        "carting away", "removal", "debris removal", "earth disposal",
        "surplus earth", "excess earth", "lead and lift", "transportation",
        "dumping", "disposing", "waste disposal"
    ],

    # Waterproofing
    "waterproofing": [
        "dpc", "damp proof course", "damp proofing", "dampproofing",
        "membrane", "app membrane", "sbs membrane", "bituminous",
        "water proofing", "water-proofing", "moisture barrier",
        "tar felt", "polythene sheet", "integral waterproofing"
    ],

    # Anti-termite
    "anti_termite": [
        "anti termite", "anti-termite", "termite treatment",
        "termite proofing", "soil treatment", "chemical treatment",
        "pre-construction anti termite", "post-construction anti termite",
        "termite barrier"
    ],

    # Curing
    "curing": [
        "water curing", "wet curing", "ponding", "curing compound",
        "membrane curing", "spraying", "jute curing", "gunny bag curing"
    ],

    # Columns
    "column": [
        "pillar", "post", "stanchion", "vertical member",
        "rcc column", "concrete column"
    ],

    # Footings
    "footing": [
        "foundation", "isolated footing", "isolated foundation",
        "spread footing", "pad footing", "individual footing",
        "combined footing", "strip footing", "raft", "raft foundation",
        "mat foundation", "pile cap"
    ],

    # Beams
    "beam": [
        "plinth beam", "tie beam", "ground beam", "grade beam",
        "lintel", "lintel beam", "roof beam", "floor beam",
        "rcc beam", "concrete beam"
    ],

    # Slab
    "slab": [
        "rcc slab", "roof slab", "floor slab", "chajja", "canopy",
        "sunshade", "waist slab", "landing slab", "concrete slab"
    ],

    # Finishes
    "plaster": [
        "plastering", "cement plaster", "sand facing", "neeru finish",
        "punning", "rendering", "internal plaster", "external plaster"
    ],

    # Masonry
    "masonry": [
        "brick work", "brickwork", "block work", "blockwork",
        "stone masonry", "rubble masonry", "ashlar masonry"
    ],

    # Flooring
    "flooring": [
        "floor", "tiles", "tiling", "floor tiles", "floor finish",
        "vitrified", "ceramic", "marble", "granite", "kota stone",
        "IPS", "terrazzo", "shahabad", "paver"
    ],

    # Painting
    "painting": [
        "paint", "emulsion", "distemper", "texture paint",
        "enamel paint", "primer", "putty", "wall paint", "oil paint"
    ],

    # Plumbing
    "plumbing": [
        "pipe", "piping", "pipeline", "plumbing work", "water supply",
        "drainage", "sewerage", "cpvc", "upvc", "gi pipe", "ppr pipe",
        "swr pipe", "sanitary", "sanitary fittings"
    ],

    # Electrical
    "electrical": [
        "wiring", "electrification", "electric", "electrical work",
        "conduit", "switch", "socket", "mcb", "db", "distribution board",
        "earthing", "lighting", "light point", "power point"
    ],

    # Doors
    "door": [
        "door", "doorway", "flush door", "panel door", "fire door",
        "door frame", "door shutter", "chaukhat", "wooden door",
        "ms door", "glass door", "automatic door"
    ],

    # Windows
    "window": [
        "window", "ventilator", "glazing", "glass pane",
        "aluminium window", "upvc window", "steel window",
        "sliding window", "casement window", "fixed window"
    ],

    # False ceiling
    "false_ceiling": [
        "false ceiling", "suspended ceiling", "drop ceiling",
        "pop ceiling", "gypsum ceiling", "mineral fiber ceiling",
        "grid ceiling", "metal ceiling"
    ],

    # External works
    "external_works": [
        "external", "site work", "compound wall", "boundary wall",
        "gate", "road", "driveway", "drain", "drainage", "landscape",
        "landscaping", "paving", "parking", "retaining wall"
    ],

    # Kitchen
    "kitchen_work": [
        "kitchen", "kitchen platform", "cooking platform",
        "granite platform", "ss sink", "kitchen counter",
        "chimney", "exhaust"
    ],

    # Fire safety
    "fire_safety": [
        "fire", "fire fighting", "fire extinguisher", "hydrant",
        "sprinkler", "smoke detector", "fire alarm", "fire escape"
    ],

    # Dewatering
    "dewatering": [
        "dewatering", "sumps", "pumping out", "dewatering arrangement",
        "well point", "sheet piling"
    ],

    # Grouting
    "grouting": [
        "grouting", "cement grouting", "epoxy grouting", "crack grouting",
        "PU grouting", "non-shrink grout", "pressure grouting"
    ],

    # Screed
    "screed": [
        "screed", "cement screed", "sand-cement screed",
        "self-levelling screed", "screeding", "floor screed"
    ],

    # Parapet
    "parapet": [
        "parapet", "parapet wall", "parapet railing", "coping",
        "parapet coping"
    ],

    # Retaining wall
    "retaining_wall": [
        "retaining wall", "breast wall", "RCC retaining", "gravity wall",
        "gabion wall", "retaining structure"
    ],

    # Lift / Elevator
    "lift": [
        "lift", "elevator", "lift pit", "machine room", "lift shaft",
        "passenger lift", "goods lift"
    ],

    # STP / Sewage treatment
    "stp": [
        "STP", "sewage treatment", "effluent treatment", "MBR", "MBBR",
        "septic tank", "soak pit"
    ],

    # Solar / EV
    "solar": [
        "solar panel", "solar water heater", "EV charging", "EVSE",
        "solar PV", "rooftop solar"
    ],

    # Insulation
    "insulation": [
        "insulation", "thermal insulation", "XPS board", "rock wool",
        "glasswool", "PIR", "EPS", "acoustic insulation"
    ],

    # Scaffolding
    "scaffolding": [
        "scaffolding", "staging", "putlog", "H-frame", "cup lock",
        "tubular scaffold", "double scaffolding"
    ],

    # Expansion joint
    "expansion_joint": [
        "expansion joint", "construction joint", "movement joint",
        "joint sealant", "polysulphide sealant"
    ],

    # Precast
    "precast": [
        "precast", "pre-cast", "prefab", "prefabricated",
        "hollow core slab", "precast wall"
    ],

    # Road work
    "road_work": [
        "road marking", "thermoplastic paint", "speed breaker", "kerb",
        "kerb stone", "bituminous road", "WBM", "WMM", "GSB"
    ]
}


# Reverse mapping: synonym -> standard term
SYNONYM_TO_STANDARD: Dict[str, str] = {}
for standard, synonyms in INDIAN_SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_STANDARD[syn.lower()] = standard


# =============================================================================
# REGEX PATTERNS FOR FUZZY MATCHING
# =============================================================================

# Compiled regex patterns for faster matching
PATTERNS: Dict[str, re.Pattern] = {
    "excavation": re.compile(
        r'\b(excavat|khudai|khodai|earthwork|earth\s*work|trench|pit\s*dig|cutting)\w*\b',
        re.IGNORECASE
    ),
    "pcc": re.compile(
        r'\b(pcc|blinding|lean\s*concrete|plain\s*(cement\s*)?concrete|'
        r'levell?ing\s*(course|bed)|mud\s*mat|bed\s*concrete|'
        r'1\s*:\s*[34]\s*:\s*[68]|m\s*10|m\s*7\.?5)\b',
        re.IGNORECASE
    ),
    "rcc": re.compile(
        r'\b(rcc|r\.?c\.?c\.?|reinforced\s*(cement\s*)?concrete|'
        r'structural\s*concrete)\b',
        re.IGNORECASE
    ),
    "formwork": re.compile(
        r'\b(formwork|form\s*work|shuttering|centering|'
        r'center?ing\s*(and|&)?\s*shuttering|mould|boxing)\b',
        re.IGNORECASE
    ),
    "reinforcement": re.compile(
        r'\b(reinforc|saria|sariya|rebar|steel\s*bar|'
        r'tmt|fe\s*500|fe\s*415|hsd|tor\s*steel|bbs|'
        r'bar\s*bending|stirrup|distribution\s*steel)\w*\b',
        re.IGNORECASE
    ),
    "backfill": re.compile(
        r'\b(backfill|back\s*fill|refill|earth\s*fill|'
        r'filling|compaction|ramming|consolidat)\w*\b',
        re.IGNORECASE
    ),
    "waterproofing": re.compile(
        r'\b(waterproof|water\s*proof|dpc|damp\s*proof|'
        r'membrane|bituminous|moisture\s*barrier)\w*\b',
        re.IGNORECASE
    ),
    "anti_termite": re.compile(
        r'\b(anti[\s\-]?termite|termite\s*(treat|proof)|'
        r'soil\s*treatment|chemical\s*treatment)\w*\b',
        re.IGNORECASE
    ),
    "curing": re.compile(
        r'\b(curing|cur(ed|ing)|ponding|water\s*curing|'
        r'wet\s*curing|membrane\s*curing)\b',
        re.IGNORECASE
    ),
    "footing": re.compile(
        r'\b(footing|foundation|isolated|spread|pad|'
        r'combined|strip|raft|pile\s*cap)\b',
        re.IGNORECASE
    ),
    "column": re.compile(
        r'\b(column|pillar|post|stanchion|vertical\s*member)\b',
        re.IGNORECASE
    ),
    "beam": re.compile(
        r'\b(beam|plinth\s*beam|tie\s*beam|lintel|'
        r'grade\s*beam|ground\s*beam)\b',
        re.IGNORECASE
    ),
    "slab": re.compile(
        r'\b(slab|roof\s*slab|floor\s*slab|chajja|canopy|'
        r'sunshade|waist\s*slab|landing\s*slab)\b',
        re.IGNORECASE
    ),
    "masonry": re.compile(
        r'\b(masonry|brick\s*work|brickwork|block\s*work|blockwork|'
        r'stone\s*masonry|rubble|ashlar)\w*\b',
        re.IGNORECASE
    ),
    "plaster": re.compile(
        r'\b(plaster|plastering|rendering|neeru|punning|'
        r'sand\s*facing|cement\s*plaster)\w*\b',
        re.IGNORECASE
    ),
    "flooring": re.compile(
        r'\b(floor|flooring|tiles?|tiling|vitrified|ceramic|'
        r'marble|granite|kota\s*stone|ips|terrazzo|shahabad|paver)\w*\b',
        re.IGNORECASE
    ),
    "painting": re.compile(
        r'\b(paint|painting|emulsion|distemper|texture\s*paint|'
        r'enamel|primer|putty|white\s*wash|oil\s*paint)\w*\b',
        re.IGNORECASE
    ),
    "plumbing": re.compile(
        r'\b(plumb|piping|pipeline|water\s*supply|drainage|sewerage|'
        r'cpvc|upvc|gi\s*pipe|ppr|swr|sanitary)\w*\b',
        re.IGNORECASE
    ),
    "electrical": re.compile(
        r'\b(electric|wiring|electrification|conduit|switch|socket|'
        r'mcb|distribution\s*board|earthing|light\s*point|power\s*point)\w*\b',
        re.IGNORECASE
    ),
    "door": re.compile(
        r'\b(door|doorway|flush\s*door|panel\s*door|fire\s*door|'
        r'door\s*frame|door\s*shutter|chaukhat)\w*\b',
        re.IGNORECASE
    ),
    "window": re.compile(
        r'\b(window|ventilator|glazing|glass\s*pane|'
        r'aluminium\s*window|upvc\s*window|sliding\s*window|casement)\w*\b',
        re.IGNORECASE
    ),
    "false_ceiling": re.compile(
        r'\b(false\s*ceiling|suspended\s*ceiling|drop\s*ceiling|'
        r'pop\s*ceiling|gypsum\s*ceiling|grid\s*ceiling|metal\s*ceiling)\b',
        re.IGNORECASE
    ),
    "external_works": re.compile(
        r'\b(compound\s*wall|boundary\s*wall|gate|driveway|'
        r'retaining\s*wall|landscap|paving|parking)\w*\b',
        re.IGNORECASE
    ),
    "kitchen_work": re.compile(
        r'\b(kitchen|cooking\s*platform|granite\s*platform|'
        r'ss\s*sink|kitchen\s*counter|chimney|exhaust)\w*\b',
        re.IGNORECASE
    ),
    "fire_safety": re.compile(
        r'\b(fire\s*fight|fire\s*extinguisher|hydrant|sprinkler|'
        r'smoke\s*detector|fire\s*alarm|fire\s*escape)\w*\b',
        re.IGNORECASE
    ),
    "dewatering": re.compile(
        r'\b(dewater|sumps?|pumping\s*out|well\s*point|sheet\s*piling)\w*\b',
        re.IGNORECASE
    ),
    "grouting": re.compile(
        r'\b(grout|cement\s*grout|epoxy\s*grout|crack\s*grout|'
        r'pu\s*grout|non[\s\-]?shrink\s*grout|pressure\s*grout)\w*\b',
        re.IGNORECASE
    ),
    "screed": re.compile(
        r'\b(screed|cement\s*screed|sand[\s\-]cement\s*screed|'
        r'self[\s\-]levell?ing\s*screed|screeding|floor\s*screed)\b',
        re.IGNORECASE
    ),
    "parapet": re.compile(
        r'\b(parapet|parapet\s*wall|parapet\s*railing|coping|'
        r'parapet\s*coping)\b',
        re.IGNORECASE
    ),
    "retaining_wall": re.compile(
        r'\b(retaining\s*wall|breast\s*wall|rcc\s*retaining|'
        r'gravity\s*wall|gabion\s*wall|retaining\s*structure)\b',
        re.IGNORECASE
    ),
    "lift": re.compile(
        r'\b(lift|elevator|lift\s*pit|machine\s*room|lift\s*shaft|'
        r'passenger\s*lift|goods\s*lift)\b',
        re.IGNORECASE
    ),
    "stp": re.compile(
        r'\b(stp|sewage\s*treatment|effluent\s*treatment|mbr|mbbr|'
        r'septic\s*tank|soak\s*pit)\b',
        re.IGNORECASE
    ),
    "solar": re.compile(
        r'\b(solar\s*panel|solar\s*water\s*heater|ev\s*charg|evse|'
        r'solar\s*pv|rooftop\s*solar)\w*\b',
        re.IGNORECASE
    ),
    "insulation": re.compile(
        r'\b(insulation|thermal\s*insulation|xps\s*board|rock\s*wool|'
        r'glass\s*wool|pir|eps|acoustic\s*insulation)\b',
        re.IGNORECASE
    ),
    "scaffolding": re.compile(
        r'\b(scaffolding|staging|putlog|h[\s\-]?frame|cup\s*lock|'
        r'tubular\s*scaffold|double\s*scaffolding)\w*\b',
        re.IGNORECASE
    ),
    "expansion_joint": re.compile(
        r'\b(expansion\s*joint|construction\s*joint|movement\s*joint|'
        r'joint\s*sealant|polysulphide\s*sealant)\b',
        re.IGNORECASE
    ),
    "precast": re.compile(
        r'\b(pre[\s\-]?cast|prefab|prefabricat|hollow\s*core\s*slab|'
        r'precast\s*wall)\w*\b',
        re.IGNORECASE
    ),
    "road_work": re.compile(
        r'\b(road\s*mark|thermoplastic\s*paint|speed\s*breaker|'
        r'kerb|kerb\s*stone|bituminous\s*road|wbm|wmm|gsb)\w*\b',
        re.IGNORECASE
    )
}


# =============================================================================
# FUNCTIONS
# =============================================================================

@dataclass
class SynonymMatch:
    """Result of synonym matching."""
    standard_term: str
    matched_text: str
    confidence: float
    is_exact: bool


def normalize_term(text: str) -> Optional[str]:
    """
    Normalize a term to its standard form.

    Args:
        text: Input text (possibly Indian variant)

    Returns:
        Standard term or None if not found
    """
    text_lower = text.lower().strip()

    # Direct lookup
    if text_lower in SYNONYM_TO_STANDARD:
        return SYNONYM_TO_STANDARD[text_lower]

    # Check if it's already a standard term
    if text_lower in INDIAN_SYNONYMS:
        return text_lower

    return None


def find_synonyms(standard_term: str) -> List[str]:
    """
    Get all synonyms for a standard term.

    Args:
        standard_term: Standard BOQ term

    Returns:
        List of synonyms including regional variations
    """
    return INDIAN_SYNONYMS.get(standard_term.lower(), [])


def detect_terms_in_text(text: str) -> List[SynonymMatch]:
    """
    Detect all construction terms in text using patterns and synonyms.

    Args:
        text: Input text (OCR, notes, etc.)

    Returns:
        List of SynonymMatch results
    """
    matches = []
    text_lower = text.lower()

    # Pattern-based detection
    for term, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append(SynonymMatch(
                standard_term=term,
                matched_text=match.group(0),
                confidence=0.85 if match.group(0).lower() == term else 0.70,
                is_exact=match.group(0).lower() == term
            ))

    # Direct synonym lookup
    for word in re.findall(r'\b\w+\b', text_lower):
        if word in SYNONYM_TO_STANDARD:
            std = SYNONYM_TO_STANDARD[word]
            # Avoid duplicates
            if not any(m.matched_text.lower() == word and m.standard_term == std for m in matches):
                matches.append(SynonymMatch(
                    standard_term=std,
                    matched_text=word,
                    confidence=0.80,
                    is_exact=False
                ))

    return matches


def get_probable_items_from_text(text: str) -> Dict[str, float]:
    """
    Extract probable BOQ item categories from text.

    Args:
        text: Input text

    Returns:
        Dict of standard_term -> max confidence
    """
    matches = detect_terms_in_text(text)

    # Aggregate by term, keep max confidence
    result: Dict[str, float] = {}
    for match in matches:
        if match.standard_term not in result:
            result[match.standard_term] = match.confidence
        else:
            result[match.standard_term] = max(result[match.standard_term], match.confidence)

    return result


def expand_search_terms(term: str) -> List[str]:
    """
    Expand a search term to include all synonyms.

    Args:
        term: Standard term or synonym

    Returns:
        List of all related terms to search for
    """
    # Normalize to standard
    std = normalize_term(term)
    if std:
        return [std] + INDIAN_SYNONYMS.get(std, [])

    # If not found, check if it's already standard
    if term.lower() in INDIAN_SYNONYMS:
        return [term.lower()] + INDIAN_SYNONYMS[term.lower()]

    return [term]


# =============================================================================
# BOQ ITEM TEMPLATES WITH INDIAN DESCRIPTIONS
# =============================================================================

BOQ_TEMPLATES: Dict[str, Dict] = {
    "excavation_footing": {
        "item_name": "Excavation for foundation in all types of soil (excluding rock) including disposal of excavated earth within 50m lead",
        "unit": "m3",
        "trade": "civil",
        "element_type": "Footing",
        "basis": "(L+working space) × (B+working space) × depth"
    },
    "pcc_footing": {
        "item_name": "Providing and laying in position plain cement concrete 1:4:8 (1 cement: 4 coarse sand: 8 graded stone aggregate 40mm nominal size) in foundation including compacting, curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Footing",
        "basis": "(L+projection) × (B+projection) × thickness"
    },
    "rcc_footing": {
        "item_name": "Providing and laying in position reinforced cement concrete in isolated footings including centering, shuttering, compacting, finishing and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Footing",
        "basis": "L × B × D"
    },
    "formwork_footing": {
        "item_name": "Providing and fixing centering and shuttering for footings including stripping, cleaning and stacking",
        "unit": "m2",
        "trade": "formwork",
        "element_type": "Footing",
        "basis": "Perimeter × Depth (side formwork)"
    },
    "steel_footing": {
        "item_name": "Providing and fixing steel reinforcement for RCC work in footings including cutting, bending, binding with annealed wire",
        "unit": "kg",
        "trade": "steel",
        "element_type": "Footing",
        "basis": "As per bar bending schedule or @ kg/m³ ratio"
    },
    "rcc_column": {
        "item_name": "Providing and laying in position reinforced cement concrete in columns including centering, shuttering, compacting, finishing and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Column",
        "basis": "Width × Depth × Height"
    },
    "formwork_column": {
        "item_name": "Providing and fixing centering and shuttering for columns including stripping, cleaning and stacking",
        "unit": "m2",
        "trade": "formwork",
        "element_type": "Column",
        "basis": "Perimeter × Height (all 4 sides)"
    },
    "steel_column": {
        "item_name": "Providing and fixing steel reinforcement for RCC work in columns including cutting, bending, binding with annealed wire",
        "unit": "kg",
        "trade": "steel",
        "element_type": "Column",
        "basis": "As per bar bending schedule or @ kg/m³ ratio"
    },
    "backfill": {
        "item_name": "Filling in plinth/foundation with excavated earth in layers not exceeding 200mm including watering, ramming and compaction",
        "unit": "m3",
        "trade": "civil",
        "element_type": "General",
        "basis": "Excavation volume - Concrete volume"
    },
    "disposal": {
        "item_name": "Carting away and disposal of surplus excavated earth beyond 50m lead",
        "unit": "m3",
        "trade": "civil",
        "element_type": "General",
        "basis": "Excavation volume - Backfill volume"
    },
    "anti_termite_detailed": {
        "item_name": "Anti-termite treatment in foundation trenches and plinth with approved chemical as per IS 6313",
        "unit": "m2",
        "trade": "special",
        "element_type": "General",
        "basis": "Plan area of foundations + plinth"
    },
    "waterproofing_dpc": {
        "item_name": "Providing and laying damp proof course (DPC) at plinth level with cement mortar 1:3 mixed with waterproofing compound",
        "unit": "m2",
        "trade": "special",
        "element_type": "General",
        "basis": "Wall length × thickness"
    },
    "curing_detailed": {
        "item_name": "Curing of concrete surfaces for minimum 7 days by water spraying/ponding",
        "unit": "m2",
        "trade": "rcc",
        "element_type": "General",
        "basis": "Total concrete surface area"
    },

    # =========================================================================
    # RCC Elements — Beams
    # =========================================================================
    "rcc_beam": {
        "item_name": "Providing and laying in position reinforced cement concrete in beams including centering, shuttering, compacting, finishing and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Beam",
        "basis": "Width × Depth × Length"
    },
    "formwork_beam": {
        "item_name": "Providing and fixing centering and shuttering for beams including stripping, cleaning and stacking",
        "unit": "m2",
        "trade": "formwork",
        "element_type": "Beam",
        "basis": "(2 × Depth + Width) × Length (bottom + two sides)"
    },
    "steel_beam": {
        "item_name": "Providing and fixing steel reinforcement for RCC work in beams including cutting, bending, binding with annealed wire",
        "unit": "kg",
        "trade": "steel",
        "element_type": "Beam",
        "basis": "As per bar bending schedule or @ kg/m3 ratio"
    },

    # =========================================================================
    # RCC Elements — Slabs
    # =========================================================================
    "rcc_slab": {
        "item_name": "Providing and laying in position reinforced cement concrete in slabs including centering, shuttering, compacting, finishing and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Slab",
        "basis": "Length × Width × Thickness"
    },
    "formwork_slab": {
        "item_name": "Providing and fixing centering and shuttering for slabs including stripping, cleaning and stacking",
        "unit": "m2",
        "trade": "formwork",
        "element_type": "Slab",
        "basis": "Length × Width (soffit area)"
    },
    "steel_slab": {
        "item_name": "Providing and fixing steel reinforcement for RCC work in slabs including cutting, bending, binding with annealed wire",
        "unit": "kg",
        "trade": "steel",
        "element_type": "Slab",
        "basis": "As per bar bending schedule or @ kg/m3 ratio"
    },

    # =========================================================================
    # RCC Elements — Lintels
    # =========================================================================
    "rcc_lintel": {
        "item_name": "Providing and laying in position reinforced cement concrete in lintels including centering, shuttering, compacting, finishing and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Lintel",
        "basis": "Width × Depth × Length (opening + bearing)"
    },
    "formwork_lintel": {
        "item_name": "Providing and fixing centering and shuttering for lintels including stripping, cleaning and stacking",
        "unit": "m2",
        "trade": "formwork",
        "element_type": "Lintel",
        "basis": "(2 × Depth + Width) × Length"
    },
    "steel_lintel": {
        "item_name": "Providing and fixing steel reinforcement for RCC work in lintels including cutting, bending, binding with annealed wire",
        "unit": "kg",
        "trade": "steel",
        "element_type": "Lintel",
        "basis": "As per bar bending schedule or @ kg/m3 ratio"
    },

    # =========================================================================
    # RCC Elements — Staircase
    # =========================================================================
    "rcc_staircase": {
        "item_name": "Providing and laying in position reinforced cement concrete in staircase waist slab, landing and steps including centering, shuttering, compacting, finishing and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "Staircase",
        "basis": "Inclined slab area × waist thickness + step volume"
    },
    "formwork_staircase": {
        "item_name": "Providing and fixing centering and shuttering for staircase including waist slab, risers and landing",
        "unit": "m2",
        "trade": "formwork",
        "element_type": "Staircase",
        "basis": "Inclined soffit area + riser shuttering + sides"
    },
    "steel_staircase": {
        "item_name": "Providing and fixing steel reinforcement for RCC work in staircase including cutting, bending, binding with annealed wire",
        "unit": "kg",
        "trade": "steel",
        "element_type": "Staircase",
        "basis": "As per bar bending schedule or @ kg/m3 ratio"
    },

    # =========================================================================
    # Masonry
    # =========================================================================
    "masonry_230mm": {
        "item_name": "Providing and constructing 230mm (9-inch) thick external brick wall in cement mortar 1:6 using first class burnt clay bricks",
        "unit": "m3",
        "trade": "masonry",
        "element_type": "Wall",
        "basis": "Length × Height × 0.23 (deduct openings)"
    },
    "masonry_115mm": {
        "item_name": "Providing and constructing 115mm (4.5-inch) thick internal brick wall in cement mortar 1:4 using first class burnt clay bricks",
        "unit": "m2",
        "trade": "masonry",
        "element_type": "Wall",
        "basis": "Length × Height (deduct openings)"
    },
    "masonry_aac_200mm": {
        "item_name": "Providing and constructing 200mm thick AAC block masonry wall in cement polymer adhesive mortar as per IS 6441",
        "unit": "m3",
        "trade": "masonry",
        "element_type": "Wall",
        "basis": "Length × Height × 0.20 (deduct openings)"
    },
    "masonry_aac_100mm": {
        "item_name": "Providing and constructing 100mm thick AAC block partition wall in cement polymer adhesive mortar",
        "unit": "m2",
        "trade": "masonry",
        "element_type": "Wall",
        "basis": "Length × Height (deduct openings)"
    },
    "masonry_concrete_block": {
        "item_name": "Providing and constructing 150mm thick solid concrete block masonry in cement mortar 1:4 using precast blocks 400x200x150mm",
        "unit": "m3",
        "trade": "masonry",
        "element_type": "Wall",
        "basis": "Length × Height × 0.15 (deduct openings)"
    },

    # =========================================================================
    # Plaster
    # =========================================================================
    "plaster_internal_12mm": {
        "item_name": "Providing and applying 12mm thick internal cement plaster in single coat in cement mortar 1:4 on walls including curing",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "Wall area (deduct openings, add jambs/soffits)"
    },
    "plaster_external_15mm": {
        "item_name": "Providing and applying 15mm thick external cement plaster in two coats (10mm base + 5mm finish) in cement mortar 1:4 on walls including curing",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "External wall area (deduct openings, add jambs/soffits)"
    },
    "plaster_neeru_finish": {
        "item_name": "Providing and applying 6mm thick POP/neeru finish coat over plastered surface on walls and ceiling",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "Same as plastered area"
    },
    "plaster_ceiling": {
        "item_name": "Providing and applying 12mm thick cement plaster in single coat in cement mortar 1:4 on ceiling including curing",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Slab",
        "basis": "Ceiling area (room L × W)"
    },

    # =========================================================================
    # Flooring & Skirting
    # =========================================================================
    "flooring_vitrified": {
        "item_name": "Providing and laying vitrified tiles 600x600mm in required pattern over 20mm thick cement mortar 1:4 bed including grouting of joints",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Floor",
        "basis": "Room area (L × W)"
    },
    "flooring_ceramic": {
        "item_name": "Providing and laying ceramic floor tiles 300x300mm over 20mm thick cement mortar 1:4 bed including grouting of joints",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Floor",
        "basis": "Room area (L × W)"
    },
    "flooring_marble": {
        "item_name": "Providing and laying polished marble stone flooring 18mm thick in required pattern over 25mm thick cement mortar 1:3 bed including rubbing and polishing",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Floor",
        "basis": "Room area (L × W)"
    },
    "flooring_granite": {
        "item_name": "Providing and laying polished granite stone flooring 18mm thick in required pattern over 25mm thick cement mortar 1:3 bed including rubbing and polishing",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Floor",
        "basis": "Room area (L × W)"
    },
    "flooring_ips": {
        "item_name": "Providing and laying 25mm thick IPS (Indian Patent Stone) flooring in cement mortar 1:2 mixed with approved hardener including curing",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Floor",
        "basis": "Room area (L × W)"
    },
    "skirting_tile": {
        "item_name": "Providing and fixing tile skirting 100mm height with matching/contrasting tiles over 12mm thick cement mortar 1:3 bed",
        "unit": "m",
        "trade": "finishing",
        "element_type": "Floor",
        "basis": "Room perimeter (deduct door openings)"
    },
    "dado_tiles": {
        "item_name": "Providing and fixing ceramic tile dado in kitchen/toilet areas over 12mm thick cement mortar 1:3 bed including grouting",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "Wall area up to dado height (deduct openings)"
    },
    "wall_tiles": {
        "item_name": "Providing and fixing ceramic wall tiles up to 2100mm height over 12mm thick cement mortar 1:3 bed including grouting of joints",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "Wall area up to tile height (deduct openings)"
    },

    # =========================================================================
    # Painting
    # =========================================================================
    "painting_internal": {
        "item_name": "Providing and applying two coats of plastic emulsion paint of approved brand and shade on internal walls over one coat of primer after surface preparation",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "Internal wall area (deduct openings, add jambs)"
    },
    "painting_external": {
        "item_name": "Providing and applying two coats of exterior grade emulsion paint of approved brand and shade on external walls over one coat of primer after surface preparation",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Wall",
        "basis": "External wall area (deduct openings, add jambs)"
    },
    "painting_ceiling": {
        "item_name": "Providing and applying two coats of plastic emulsion paint of approved brand and shade on ceiling over one coat of primer after surface preparation",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "Slab",
        "basis": "Ceiling area (room L × W)"
    },
    "painting_woodwork": {
        "item_name": "Providing and applying two coats of synthetic enamel paint of approved brand and shade on woodwork over one coat of primer after surface preparation",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "General",
        "basis": "Surface area of woodwork (both faces + edges)"
    },

    # =========================================================================
    # Waterproofing (additional)
    # =========================================================================
    "waterproofing_toilet": {
        "item_name": "Providing and applying waterproofing treatment in toilets/bathrooms with integral waterproofing compound in IPS with coba including sealing of junctions",
        "unit": "m2",
        "trade": "special",
        "element_type": "Floor",
        "basis": "Toilet floor area + wall up to 200mm height"
    },
    "waterproofing_terrace": {
        "item_name": "Providing and laying terrace waterproofing treatment with brick bat coba and integral waterproofing compound including protective screed and junction sealing",
        "unit": "m2",
        "trade": "special",
        "element_type": "Slab",
        "basis": "Terrace slab area + parapet turn-up"
    },
    "waterproofing_basement": {
        "item_name": "Providing and laying basement waterproofing with APP modified bituminous membrane of approved make including primer coat and protective screed",
        "unit": "m2",
        "trade": "special",
        "element_type": "General",
        "basis": "Basement floor area + retaining wall area"
    },
    "waterproofing_sunken": {
        "item_name": "Providing and applying waterproofing treatment for sunken slab portion in toilets/bathrooms including IPS with waterproofing compound and junction sealing",
        "unit": "m2",
        "trade": "special",
        "element_type": "Slab",
        "basis": "Sunken slab area (floor + walls up to slab level)"
    },

    # =========================================================================
    # Doors & Windows
    # =========================================================================
    "door_frame_sal": {
        "item_name": "Providing and fixing sal wood door frame of section 100x75mm with hold fast including wrought iron hold fasts, priming coat and necessary hardware",
        "unit": "m",
        "trade": "carpentry",
        "element_type": "Door",
        "basis": "Frame perimeter (2 × height + width)"
    },
    "door_shutter_flush": {
        "item_name": "Providing and fixing 35mm thick flush door shutter of approved make with commercial ply veneered both sides including necessary hardware and fittings",
        "unit": "m2",
        "trade": "carpentry",
        "element_type": "Door",
        "basis": "Width × Height of shutter"
    },
    "door_shutter_panel": {
        "item_name": "Providing and fixing solid panel door shutter of seasoned wood with styles and rails including hardware, fittings and priming coat",
        "unit": "m2",
        "trade": "carpentry",
        "element_type": "Door",
        "basis": "Width × Height of shutter"
    },
    "window_aluminium": {
        "item_name": "Providing and fixing aluminium sliding window of approved section and make with 5mm clear glass including all hardware, rubber beading and weather sealing",
        "unit": "m2",
        "trade": "aluminium",
        "element_type": "Window",
        "basis": "Width × Height of opening"
    },
    "window_upvc": {
        "item_name": "Providing and fixing UPVC sliding window of approved make and profile with 5mm clear glass including all hardware, beading, weather sealing and mosquito mesh",
        "unit": "m2",
        "trade": "aluminium",
        "element_type": "Window",
        "basis": "Width × Height of opening"
    },
    "ventilator": {
        "item_name": "Providing and fixing aluminium louvered ventilator of approved make and section with mosquito mesh including all hardware and weather sealing",
        "unit": "m2",
        "trade": "aluminium",
        "element_type": "Window",
        "basis": "Width × Height of opening"
    },

    # =========================================================================
    # Plumbing & Sanitary
    # =========================================================================
    "plumbing_cpvc_hot": {
        "item_name": "Providing and fixing CPVC hot and cold water supply pipeline 15-25mm diameter of approved make with fittings, clamps and testing",
        "unit": "m",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Running length of pipeline as per layout"
    },
    "plumbing_upvc_swr": {
        "item_name": "Providing and fixing UPVC SWR (soil, waste, rain) pipe 110mm diameter of approved make with fittings, clamps and testing",
        "unit": "m",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Running length of pipeline as per layout"
    },
    "plumbing_gi_supply": {
        "item_name": "Providing and fixing GI water supply pipe 15-50mm diameter (medium class) with GI fittings including cutting, threading, jointing and testing",
        "unit": "m",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Running length of pipeline as per layout"
    },
    "sanitary_ewc": {
        "item_name": "Providing and fixing European type white vitreous china water closet with flush valve/cistern of approved make including seat cover, connection to drain and water supply",
        "unit": "no",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Number of WCs as per drawing"
    },
    "sanitary_wash_basin": {
        "item_name": "Providing and fixing white vitreous china wash basin with pedestal of approved make including CP brass waste coupling, bottle trap and connection to drain",
        "unit": "no",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Number of wash basins as per drawing"
    },
    "sanitary_kitchen_sink": {
        "item_name": "Providing and fixing stainless steel kitchen sink of approved make and size with waste coupling, connection to drain and water supply",
        "unit": "no",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Number of sinks as per drawing"
    },
    "water_tank_overhead": {
        "item_name": "Providing and fixing overhead water storage tank of approved make in HDPE/FRP/SS of required capacity including supports, piping connections and accessories",
        "unit": "litre",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Tank capacity as per design"
    },
    "rainwater_harvesting": {
        "item_name": "Providing and constructing rainwater harvesting system with collection pit, filter bed, recharge well including all piping, chamber and accessories as per local authority norms",
        "unit": "no",
        "trade": "plumbing",
        "element_type": "General",
        "basis": "Number of units as per site plan"
    },

    # =========================================================================
    # Electrical
    # =========================================================================
    "electrical_wiring_light": {
        "item_name": "Providing and wiring for one light point with 1.5 sq.mm FR PVC insulated copper conductor wire in 20mm UPVC conduit with modular switch and accessories",
        "unit": "point",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of light points as per drawing"
    },
    "electrical_wiring_power": {
        "item_name": "Providing and wiring for one power point (5/15A combined) with 4 sq.mm FR PVC insulated copper conductor wire in 25mm UPVC conduit with modular socket and accessories",
        "unit": "point",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of power points as per drawing"
    },
    "electrical_wiring_ac": {
        "item_name": "Providing and wiring for one AC point (20A) with 4 sq.mm FR PVC insulated copper conductor wire in 25mm UPVC conduit including isolator switch and accessories",
        "unit": "point",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of AC points as per drawing"
    },
    "electrical_switch_board": {
        "item_name": "Providing and fixing modular switch plate of approved make with required number of switches, sockets, blanks and accessories on flush-mounted GI box",
        "unit": "no",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of switch boards as per drawing"
    },
    "electrical_mcb_db": {
        "item_name": "Providing and fixing MCB distribution board of approved make in sheet steel enclosure with required number of MCBs, RCCB, bus bars and accessories",
        "unit": "no",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of DBs as per SLD"
    },
    "electrical_earthing": {
        "item_name": "Providing and making earthing with GI plate/pipe electrode in charcoal and salt pit including 8 SWG GI earth wire, earth lead and accessories as per IS 3043",
        "unit": "no",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of earth pits as per design"
    },
    "electrical_main_cable": {
        "item_name": "Providing and laying 3.5/4 core armoured XLPE copper cable of required size from energy meter to main distribution board including cable tray/trench and terminations",
        "unit": "m",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Running length from meter to DB"
    },
    "electrical_light_fixture": {
        "item_name": "Providing and installing LED light fixture of approved make, wattage and type (surface/recessed/pendant) including all accessories, wiring and connection",
        "unit": "no",
        "trade": "electrical",
        "element_type": "General",
        "basis": "Number of fixtures as per lighting layout"
    },

    # =========================================================================
    # External Works
    # =========================================================================
    "external_compound_wall": {
        "item_name": "Providing and constructing compound wall in brick/RCC with plinth beam, columns at intervals, coping and plastering both sides including foundation",
        "unit": "m",
        "trade": "civil",
        "element_type": "General",
        "basis": "Running length of compound wall"
    },
    "external_gate": {
        "item_name": "Providing and fixing MS gate with MS tubular/box section frame including hinges, stopper, locking arrangement, primer and enamel paint",
        "unit": "kg",
        "trade": "civil",
        "element_type": "General",
        "basis": "Weight of gate assembly"
    },
    "external_road_cc": {
        "item_name": "Providing and laying CC road/driveway with 150mm thick PCC M15 over 150mm thick compacted sub-base including formwork, joints, finishing and curing",
        "unit": "m2",
        "trade": "civil",
        "element_type": "General",
        "basis": "Plan area of road/driveway"
    },
    "external_drain": {
        "item_name": "Providing and constructing open/covered surface drain in brick/RCC for storm water and waste water disposal including plastering, cover slabs and connections",
        "unit": "m",
        "trade": "civil",
        "element_type": "General",
        "basis": "Running length of drain"
    },
    "external_stp": {
        "item_name": "Providing and constructing septic tank/STP of required capacity in RCC with inlet/outlet chambers, baffles, manhole covers and all connections as per local norms",
        "unit": "no",
        "trade": "civil",
        "element_type": "General",
        "basis": "Number of units as per site plan"
    },
    "external_landscaping": {
        "item_name": "Providing and executing landscaping work including lawn development with turf grass, planting, paver block paving, garden edging and topsoil spreading",
        "unit": "m2",
        "trade": "civil",
        "element_type": "General",
        "basis": "Plan area of landscaped zone"
    },

    # =========================================================================
    # Miscellaneous
    # =========================================================================
    "scaffolding": {
        "item_name": "Providing and erecting double scaffolding (cup-lock/H-frame) for multi-storey work including dismantling, as required for construction and finishing activities",
        "unit": "m2",
        "trade": "civil",
        "element_type": "General",
        "basis": "Elevation area requiring scaffolding"
    },
    "pcc_grade_slab": {
        "item_name": "Providing and laying in position plain cement concrete 1:3:6 (1 cement: 3 coarse sand: 6 graded stone aggregate 20mm nominal size) for grade slab/apron including compacting and curing",
        "unit": "m3",
        "trade": "rcc",
        "element_type": "General",
        "basis": "Plan area × thickness"
    },
    "false_ceiling_pop": {
        "item_name": "Providing and fixing POP (Plaster of Paris) false ceiling with GI frame suspension system including all accessories, finishing and painting",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "General",
        "basis": "Ceiling area (room L × W)"
    },
    "false_ceiling_gypsum": {
        "item_name": "Providing and fixing gypsum board false ceiling with GI frame suspension system of approved make including all accessories, jointing, finishing and painting",
        "unit": "m2",
        "trade": "finishing",
        "element_type": "General",
        "basis": "Ceiling area (room L × W)"
    },

    # =========================================================================
    # Additional Templates — Parapet, Staircase, Termite, Curing, Joints, etc.
    # =========================================================================
    "parapet_wall": {
        "item_name": "Brick masonry parapet wall",
        "unit": "cum",
        "trade": "masonry",
        "element_type": "parapet",
        "basis": "IS 1200 Part 3"
    },
    "coping": {
        "item_name": "CC coping on parapet",
        "unit": "rmt",
        "trade": "masonry",
        "element_type": "parapet",
        "basis": "IS 1200"
    },
    "staircase_railing": {
        "item_name": "MS staircase railing",
        "unit": "rmt",
        "trade": "metalwork",
        "element_type": "staircase",
        "basis": "IS 1148"
    },
    "anti_termite": {
        "item_name": "Anti-termite treatment to soil",
        "unit": "sqm",
        "trade": "termite",
        "element_type": "foundation",
        "basis": "IS 6313"
    },
    "curing": {
        "item_name": "Curing of concrete",
        "unit": "sqm",
        "trade": "structural",
        "element_type": "concrete",
        "basis": "IS 456"
    },
    "expansion_joint": {
        "item_name": "Expansion joint with sealant",
        "unit": "rmt",
        "trade": "structural",
        "element_type": "joint",
        "basis": "IS 456"
    },
    "earth_filling": {
        "item_name": "Earth filling in plinth",
        "unit": "cum",
        "trade": "earthwork",
        "element_type": "plinth",
        "basis": "IS 1200 Part 1"
    },
    "sand_filling": {
        "item_name": "Sand filling under floor",
        "unit": "cum",
        "trade": "earthwork",
        "element_type": "floor",
        "basis": "IS 1200 Part 1"
    },
    "dpc": {
        "item_name": "DPC with cement concrete",
        "unit": "sqm",
        "trade": "waterproofing",
        "element_type": "plinth",
        "basis": "IS 2645"
    },
    "plinth_beam": {
        "item_name": "RCC plinth beam",
        "unit": "cum",
        "trade": "structural",
        "element_type": "plinth",
        "basis": "IS 456"
    },
    "compound_wall": {
        "item_name": "Brick compound wall",
        "unit": "rmt",
        "trade": "external",
        "element_type": "boundary",
        "basis": "IS 1200 Part 3"
    },
    "septic_tank": {
        "item_name": "RCC septic tank",
        "unit": "no",
        "trade": "external",
        "element_type": "drainage",
        "basis": "IS 2470"
    },
    "overhead_tank": {
        "item_name": "RCC overhead water tank",
        "unit": "no",
        "trade": "structural",
        "element_type": "water_supply",
        "basis": "IS 3370"
    }
}


# ---------------------------------------------------------------------------
# Knowledge Base Extension (additive — extends INDIAN_SYNONYMS from YAML data)
# ---------------------------------------------------------------------------
try:
    from src.knowledge_base import get_all_synonyms as _kb_syn

    for _canonical, _aliases in _kb_syn().items():
        if _canonical in INDIAN_SYNONYMS:
            _existing = set(INDIAN_SYNONYMS[_canonical])
            INDIAN_SYNONYMS[_canonical].extend(
                [a for a in _aliases if a not in _existing]
            )
        else:
            INDIAN_SYNONYMS[_canonical] = _aliases

    del _canonical, _aliases, _existing
except ImportError:
    pass  # Knowledge base not installed yet
except Exception as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Knowledge base synonym loading failed: %s", _e
    )
