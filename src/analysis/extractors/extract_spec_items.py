"""
Spec Item Extractor — Parse numbered line items from spec/notes/conditions pages.

Converts structured spec text (numbered clauses, lettered items) into priceable
line items with inferred units, quantity hints, and trade classification.

These complement BOQ items: BOQ provides rates/quantities, spec items provide
scope descriptions that may not have been priced or that carry different detail.

Confidence is intentionally 0.45 (lower than BOQ items at 0.75+) because the
unit and quantity inference is heuristic.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Unit inference
# ---------------------------------------------------------------------------

# Maps our canonical unit codes to trigger keywords.
# First match wins in order of specificity (longer keywords first within each).
_UNIT_INFER: Dict[str, List[str]] = {
    "rmt": [
        "running metre", "running meter", "r.m", "per metre", "per meter",
        "pipe", "cable", "conduit", "duct", "railing", "skirting",
        "cornice", "channel", "gutter", "drain", "kerb", "sewer",
        "rmt",
    ],
    "sqm": [
        "waterproofing", "painting", "tiling", "flooring", "plaster",
        "plastering", "rendering", "cladding", "membrane", "coating",
        "shuttering", "formwork", "dado",
        "sqm", "sq.m",
        "tile", "floor",  # shorter — check AFTER multi-word
        # NOTE: "m2" / "per m2" intentionally omitted — explicit unit regex
        # handles those with word boundaries (avoids "m25" false positive).
    ],
    "cum": [
        "excavation", "earthwork", "filling", "backfill",
        "concrete", "concreting", "masonry", "brick", "block",
        "cum", "cu.m",
        # NOTE: "m3" / "per m3" / "per cum" handled by explicit unit regex only.
    ],
    "nos": [
        "door", "window", "fixture", "light fitting", "fan", "switch",
        "socket", "valve", "manhole", "pump", "unit", "set",
        "point", "outlet",
        "nos", "no.", "each", "numbers",
    ],
    "kg": ["rebar", "reinforcement", "tmt", "tor", "mild steel bar"],
    "mt": ["structural steel", "fabrication", "rolled section"],
    "bag": ["cement bag"],
    "ls": ["lump sum", "provisional", "allowance", "l.s", "p.s"],
}

# Regex to find an explicit unit token in text (check first; beats inference).
_EXPLICIT_UNIT_RE = re.compile(
    r'\b(sqm|cum|rmt|nos|kg|mt|ls|bag|liter|litre|m2|m3)\b',
    re.IGNORECASE,
)

_UNIT_ALIASES: Dict[str, str] = {
    "m2": "sqm", "m3": "cum", "liter": "ltr", "litre": "ltr",
}


def _normalize_unit(raw: str) -> str:
    return _UNIT_ALIASES.get(raw.lower(), raw.lower())


def _infer_unit(text: str) -> Tuple[Optional[str], bool]:
    """
    Return (unit, unit_inferred).

    unit_inferred=False means an explicit token was found in the text.
    unit_inferred=True  means we guessed from keywords.
    """
    m = _EXPLICIT_UNIT_RE.search(text)
    if m:
        return _normalize_unit(m.group(1)), False

    text_lower = text.lower()
    for unit, keywords in _UNIT_INFER.items():
        if any(kw in text_lower for kw in keywords):
            return unit, True

    return None, False


# ---------------------------------------------------------------------------
# Quantity extraction
# ---------------------------------------------------------------------------

# Patterns like "5 nos", "supply 3 pumps", "install 10 light"
_QTY_RE = re.compile(
    r'(?:supply|provide|install|furnish|fix)?\s*(\d+(?:\.\d+)?)\s*'
    r'(?:nos?|no\.|each|sets?|units?|'
    r'sqm|sq\.m|cum|cu\.m|rmt|kg|mt|bags?|ltr)',
    re.IGNORECASE,
)


def _extract_qty(text: str) -> Optional[float]:
    m = _QTY_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Standards / code references
# ---------------------------------------------------------------------------

_STANDARDS_RE = re.compile(
    r'\b(IS[:\s]\d{3,6}|BS[:\s]\d{3,6}|ASTM\s+[A-Z]\d{2,4}|'
    r'IS\s+\d{3,6}(?:[-:]\d+)?|BIS\s+\d+)\b',
    re.IGNORECASE,
)


def _extract_standards(text: str) -> List[str]:
    return list({m.strip() for m in _STANDARDS_RE.findall(text)})


# ---------------------------------------------------------------------------
# Trade classification (mirrors extract_notes.py pattern)
# ---------------------------------------------------------------------------

_TRADE_KEYWORDS: Dict[str, List[str]] = {
    "civil": [
        "earthwork", "excavation", "backfill", "pcc", "soil",
        "grading", "site clearance", "dewatering", "subgrade",
        "culvert", "retaining wall", "compound wall", "boundary wall",
        "road work", "pavement", "soling", "base course", "sub-base",
        "filling", "embankment", "levelling", "anti-termite",
        "underpinning", "shoring", "sheeting", "well foundation",
        "hard rock", "soft rock", "murrum", "laterite",
        "demolished", "demolition", "dismantling",
        "rubble", "disposal of", "carting away",
    ],
    "structural": [
        "rcc", "reinforcement", "formwork", "shuttering",
        "footing", "column", "beam", "slab", "raft",
        "staircase", "rebar", "bar bending", "foundation",
        "pile", "shear wall", "lintel", "sunshade", "chajja",
        "concrete m20", "concrete m25", "concrete m30", "concrete m15",
        "concrete m10", "m20 grade", "m25 grade", "m30 grade",
        "grade of concrete", "precast", "prestressed",
        "tmt bar", "tmt rod", "fe415", "fe500",
        "structural steel", "ms section", "rolled section",
        "fabrication", "erection of steel", "lattice", "truss",
        "gusset", "splice", "base plate",
    ],
    "architectural": [
        "brick", "block", "masonry", "plaster", "door",
        "window", "railing", "partition", "false ceiling",
        "cladding", "facade", "joinery", "shutter", "frame",
        "fly ash brick", "aac block", "hollow block", "solid block",
        "pointing", "grouting of joints", "mortar",
        "dpc", "damp proof", "waterproof course",
        "gypsum plaster", "sand faced", "rough cast",
        "sill", "lintel level", "band", "facia",
        "precast block", "precast slab",
        "aluminium window", "upvc window", "steel window",
        "flush door", "panel door", "fire door", "rolling shutter",
        "glass", "glazing", "curtain wall",
        "insulation", "thermal insulation", "acoustic",
        "skylight", "roofing sheet", "metal deck",
        "handrail", "balustrade", "grille", "jali",
        "false ceiling", "grid ceiling", "lay-in tile ceiling",
        "suspended ceiling",
    ],
    "finishes": [
        "paint", "painting", "tile", "tiling", "polish",
        "marble", "granite", "laminate", "flooring",
        "dado", "skirting", "putty", "primer",
        "waterproofing", "rendering", "texture",
        "vitrified", "ceramic tile", "kota stone",
        "ips floor", "mosaic", "terrazzo",
        "stone cladding", "stone flooring",
        "emulsion", "distemper", "whitewash", "lime wash",
        "enamel paint", "oil bound", "obd", "acrylic",
        "epoxy coating", "epoxy flooring", "pvc flooring",
        "carpet", "wooden flooring", "laminated flooring",
        "anti-skid", "non-skid", "anti-slip",
        "elastomeric", "polyurethane",
        "french polish", "melamine",
    ],
    "plumbing": [
        "plumbing", "drainage", "sanitary", "water supply",
        "pipe", "cistern", "pump", "sewage", "faucet",
        "toilet", "basin", "urinal", "gully trap",
        "water tank", "overhead tank", "underground tank",
        "sump", "septic tank", "soak pit", "inspection chamber",
        "manhole", "catch basin", "storm drain",
        "upvc pipe", "cpvc pipe", "gi pipe", "ci pipe",
        "hdpe pipe", "pp-r pipe", "copper pipe",
        "ball valve", "gate valve", "check valve", "butterfly valve",
        "water meter", "pressure gauge",
        "water heater", "geyser", "solar water heater",
        "ewc", "indian wc", "squat plate", "wash basin",
        "kitchen sink", "laboratory sink",
        "shower", "shower tray", "bath tub",
        "floor trap", "nahani trap", "bottle trap",
        "rainwater pipe", "down take pipe",
        "booster pump", "submersible pump", "centrifugal pump",
    ],
    "electrical": [
        "wiring", "cable", "panel", "lighting", "switch",
        "socket", "earthing", "transformer", "generator",
        "conduit", "mcb", "elcb", "rccb", "substation",
        "distribution board", "mdb", "sdb", "db box",
        "power cable", "armoured cable", "xlpe cable",
        "bus duct", "bus bar", "ht cable", "lt cable",
        "street light", "flood light", "led light",
        "ceiling fan", "exhaust fan", "fresh air fan",
        "ug cable", "overhead line", "pole",
        "lightning conductor", "surge arrester",
        "ups", "inverter", "battery bank",
        "solar panel", "solar pv", "net metering",
        "cctv", "access control", "fire alarm panel",
        "public address", "pa system",
        "dg set", "diesel generator",
    ],
    "mep": [
        "hvac", "duct", "air conditioning", "chiller",
        "fire fighting", "sprinkler", "ventilation",
        "ahu", "vrf", "fan coil", "fcu",
        "cooling tower", "condenser", "compressor",
        "split ac", "cassette ac", "precision ac",
        "smoke detector", "heat detector", "fire hydrant",
        "hose reel", "fire extinguisher",
        "bms", "building management", "automation",
        "lift", "elevator", "escalator",
        "compressed air", "medical gas", "nitrogen",
        "diesel storage", "day tank",
    ],
    "external": [
        "external", "compound", "landscape", "garden",
        "fencing", "gate", "boundary", "paver block",
        "car park", "parking area", "approach road",
        "kerb", "edging", "footpath", "walkway",
        "drainage channel", "surface drain",
        "street furniture", "signage",
        "site development", "site grading",
        "tree transplant", "plantation",
    ],
    "waterproofing": [
        "waterproofing", "damp proofing", "crystalline",
        "torch applied", "cold applied membrane",
        "bituminous", "polymer modified",
        "terrace waterproofing", "basement waterproofing",
        "toilet waterproofing", "wet area waterproofing",
        "injection grouting", "crack filling",
    ],
}


def _classify_trade(text: str) -> str:
    text_lower = text.lower()
    best_trade = "general"
    best_score = 0
    for trade, keywords in _TRADE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                # longer keyword = more specific = higher score
                score += len(kw.split())
        if score > best_score:
            best_score = score
            best_trade = trade
    return best_trade


# ---------------------------------------------------------------------------
# Priceable vs contractual classifier
# ---------------------------------------------------------------------------

# Matches descriptions that START with a standards code citation (IS, BS, ASTM, etc.)
# and are therefore primarily a reference note, not a billable work item.
# Examples: "IS: 4912-1978 Safety requirements for ..."
#           "BS 8000 Workmanship on building sites"
#           "NBC 2016 Part 4 Fire and Life Safety"
_STANDARDS_ONLY_RE = re.compile(
    r'^[\'\"]?\s*(?:IS|BS|ASTM|IRC|NBC|BIS|SP)[:\s\-]\s*\d',
    re.IGNORECASE,
)

# Phrases that strongly indicate a contractual / administrative clause —
# NOT a physical work item that belongs on a BOQ.
_CONTRACTUAL_TRIGGERS: List[str] = [
    # Financial / security
    "security deposit", "earnest money", "emd", "performance guarantee",
    "performance security", "bank guarantee", "bid security", "bid bond",
    "retention money", "retention amount", "mobilisation advance",
    "advance payment", "recovery of advance", "refund of", "refund shall",
    "interest on advance", "interest shall be charged",
    # Damages / penalties
    "liquidated damages", "penalty shall", "compensation shall",
    "damages shall", "levy of penalty", "recovery of penalty",
    "time overrun", "delay penalty",
    # Legal / dispute
    "arbitration", "dispute resolution", "adjudication", "conciliation",
    "jurisdiction", "court of law", "legal proceedings",
    "force majeure", "act of god",
    # Administrative
    "defects liability period", "defect liability", "warranty period",
    "maintenance period", "dnlp", "dlp period",
    "completion certificate", "taking over certificate",
    "virtual completion", "final payment certificate",
    "insurance premium", "workmen compensation", "third party insurance",
    "contractor shall insure", "employer shall", "contractor shall maintain",
    "indemnify", "indemnification",
    "income tax", "gst shall", "tds shall", "withholding tax",
    "price escalation", "price adjustment", "price variation clause",
    "variation order", "change order shall",
    "sub-contracting", "sub contractor approval",
    "termination of contract", "suspension of work by employer",
    "notice to proceed", "letter of award", "letter of intent",
]

# Short phrase fragments (checked with `in` after lowercasing)
_CONTRACTUAL_FRAGMENTS: List[str] = [
    "% of contract", "% of bid", "% of quoted", "% per week", "% per month",
    "shall be forfeited", "shall be recovered", "shall be deducted",
    "shall be refunded", "within 30 days", "within 15 days", "within 7 days",
    "working days of", "calendar days of",
    "at the rate of", "at the discretion of",
    "contractor is required to", "contractor shall submit",
    "contractor shall obtain", "contractor shall provide insurance",
    "all disputes", "any dispute", "the engineer's decision",
    "notwithstanding anything", "without prejudice",
    "in the event of", "in case of breach",
]

# Verbs that are strong indicators of physical / billable work
_WORK_VERBS: List[str] = [
    "supply", "supply and", "supply &",
    "provide", "provide and",
    "install", "installation of",
    "construct", "construction of",
    "excavate", "excavation of",
    "lay", "laying of", "laying and",
    "fix", "fixing of",
    "paint", "painting of",
    "plaster", "plastering of",
    "erect", "erection of",
    "fabricate", "fabrication of",
    "dismantle", "demolish", "demolition of",
    "remove", "removal of",
    "backfill", "filling",
    "waterproof", "waterproofing of",
    "tile", "tiling of",
    "pour", "place", "cast",
    "weld", "welding",
    "grout", "grouting",
    "seal", "sealing",
    "reinforce",
]


def _classify_priceable(description: str, doc_type: str = "spec") -> tuple:
    """
    Return (is_priceable: bool, reason: str).

    Logic (in priority order):
    1. Any contractual trigger phrase found → contractual
    2. Any contractual fragment found → contractual
    3. Description is primarily a standards/code citation with no work verb → standards_reference
    4. Conditions/notes doc with no work verb + contains "shall" → likely contractual
    5. Otherwise → priceable (may still be noisy, but let taxonomy filter)
    """
    text_lower = description.lower()

    # Check full trigger phrases
    for trigger in _CONTRACTUAL_TRIGGERS:
        if trigger in text_lower:
            return False, f"contractual_trigger:{trigger}"

    # Check short fragments
    for frag in _CONTRACTUAL_FRAGMENTS:
        if frag in text_lower:
            return False, f"contractual_fragment:{frag}"

    # Check if description is primarily a standards reference (e.g. "IS: 4912-1978 ...")
    # If it starts with a standards code but also contains a work verb it is a real item
    # like "provide as per IS:456" — keep those.
    stripped = description.strip(" '\"")
    if _STANDARDS_ONLY_RE.match(stripped):
        has_work_verb = any(verb in text_lower for verb in _WORK_VERBS)
        if not has_work_verb:
            return False, "standards_reference"

    # For conditions/notes pages: extra skepticism on "shall" items without work verbs
    if doc_type in ("conditions", "notes"):
        has_work_verb = any(verb in text_lower for verb in _WORK_VERBS)
        has_shall = "shall" in text_lower
        has_unit  = bool(_EXPLICIT_UNIT_RE.search(description))
        if has_shall and not has_work_verb and not has_unit:
            return False, "conditions_clause_no_work_verb"

    return True, "priceable"


# ---------------------------------------------------------------------------
# Item numbering patterns
# ---------------------------------------------------------------------------

# Pattern A: decimal-numbered items — separator optional because "X.Y" or
# "X.Y.Z" followed by text is unambiguous in spec documents.
# Matches: "3.4 Supply ...", "3.4. Supply ...", "12.3.1 All pipes ..."
_DECIMAL_RE = re.compile(
    r'(?:^|\n)\s*(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\s*[.):,]?\s+(.{15,300})',
    re.MULTILINE,
)

# Pattern B: simple-numbered or lettered items — separator required to avoid
# matching arbitrary numbers in prose.
# Matches: "3. Supply ...", "a) Provide ...", "iv. Install ..."
_SIMPLE_RE = re.compile(
    r'(?:^|\n)\s*(\d{1,3}|[a-z]|[ivxlc]{1,5})\s*[.)]\s+(.{15,300})',
    re.IGNORECASE | re.MULTILINE,
)

# Section headers: ALL-CAPS lines, optionally preceded by a number.
# We track these but do NOT emit them as items.
_SECTION_RE = re.compile(
    r'(?:^|\n)\s*(?:\d+[.)]\s+)?[A-Z][A-Z\s/&,()–\-]{8,}$',
    re.MULTILINE,
)

# Minimum word count for a valid description (filters noise)
_MIN_WORDS = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_spec_items(
    text: str,
    page_idx: Optional[int] = None,
    doc_type: str = "spec",
    *,
    source_page: Optional[int] = None,
) -> List[dict]:
    """
    Parse numbered line items from spec / notes / conditions / addendum text.

    Args:
        text:        Raw OCR text from one page.
        page_idx:    Zero-based page index (for source_page field).
                     Also accepted as the keyword ``source_page`` for
                     compatibility with the plan spec API.
        doc_type:    Original doc_type (spec, notes, conditions, addendum, legend).

    Returns:
        List of dicts, each representing one priceable line item.
        Empty list if no numbered items found.
    """
    if not text or not text.strip():
        return []

    # Resolve page index: accept either positional page_idx or keyword source_page
    _page = source_page if source_page is not None else (page_idx if page_idx is not None else 0)

    items: List[dict] = []
    current_section = ""

    # --- Detect section headers (for annotation only) ---
    section_positions: List[Tuple[int, str]] = []
    for m in _SECTION_RE.finditer(text):
        header = m.group(0).strip()
        if len(header) > 5:
            section_positions.append((m.start(), header))

    def _section_at(pos: int) -> str:
        """Return the most recent section header before pos."""
        hdr = ""
        for sp, sh in section_positions:
            if sp <= pos:
                hdr = sh
            else:
                break
        return hdr

    # --- Extract numbered items ---
    # Collect matches from both patterns; deduplicate by match start position.
    _seen_positions: set = set()
    _raw_matches: List[Tuple[int, str, str]] = []  # (start, item_no, description)

    for pattern in (_DECIMAL_RE, _SIMPLE_RE):
        for m in pattern.finditer(text):
            pos = m.start()
            if pos in _seen_positions:
                continue
            _seen_positions.add(pos)
            _raw_matches.append((pos, m.group(1).strip(), m.group(2).strip()))

    # Sort by position so items appear in document order
    _raw_matches.sort(key=lambda x: x[0])

    for pos, item_no, description in _raw_matches:
        # Normalise whitespace (OCR sometimes inserts extra spaces)
        description = re.sub(r'\s{2,}', ' ', description)

        # Skip very short or clearly noisy lines
        if len(description.split()) < _MIN_WORDS:
            continue

        # Truncate at 250 chars
        if len(description) > 250:
            description = description[:247] + "..."

        unit, unit_inferred = _infer_unit(description)
        qty = _extract_qty(description)
        trade = _classify_trade(description)
        section = _section_at(pos)
        standards_codes = _extract_standards(description)
        is_priceable, priceable_reason = _classify_priceable(description, doc_type)

        items.append({
            "item_no":          item_no,
            "description":      description,
            "unit":             unit,
            "unit_inferred":    unit_inferred,
            "qty":              qty,
            "trade":            trade,
            "section":          section,
            "standards_codes":  standards_codes,
            "source_page":      _page,
            "source":           "spec_item",
            "confidence":       0.45,
            "doc_type":         doc_type,
            "is_priceable":     is_priceable,
            "priceable_reason": priceable_reason,
        })

    return items
