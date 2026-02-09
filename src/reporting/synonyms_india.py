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
        "pre-construction anti termite", "post-construction anti termite"
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
    "anti_termite": {
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
    "curing": {
        "item_name": "Curing of concrete surfaces for minimum 7 days by water spraying/ponding",
        "unit": "m2",
        "trade": "rcc",
        "element_type": "General",
        "basis": "Total concrete surface area"
    }
}
