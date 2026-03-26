"""
Notes/Spec/Legend Extractor — Pull requirements from text-heavy pages.

Handles doc_types: notes, legend, spec, conditions, addendum.

Extracts:
- "All ... shall be ..." clauses
- IS/BS/ASTM/EN code references
- Numbered items and bullet points
- Material/workmanship/standard requirements
"""

import re
from typing import List, Optional


# =============================================================================
# PATTERNS
# =============================================================================

# Standard/code references
CODE_PATTERNS = [
    re.compile(r'\bIS\s*[:\-]?\s*(\d{3,5})(?:\s*[:\-]\s*\d+)?', re.IGNORECASE),
    re.compile(r'\bBS\s*[:\-]?\s*(\d{3,5})', re.IGNORECASE),
    re.compile(r'\bASTM\s+([A-Z]\d+)', re.IGNORECASE),
    re.compile(r'\bEN\s*[:\-]?\s*(\d{3,5})', re.IGNORECASE),
    re.compile(r'\bISO\s*[:\-]?\s*(\d{3,5})', re.IGNORECASE),
    re.compile(r'\bCPWD\s+specifications?', re.IGNORECASE),
    re.compile(r'\bNBC\s+\d+', re.IGNORECASE),
]

# Requirement sentence patterns
REQUIREMENT_PATTERNS = [
    re.compile(r'(?:all|every)\s+.{5,80}\s+shall\s+be\s+.{5,120}', re.IGNORECASE),
    re.compile(r'shall\s+(?:be|have|comply|conform|meet)\s+.{5,120}', re.IGNORECASE),
    re.compile(r'must\s+(?:be|have|comply|conform|meet)\s+.{5,120}', re.IGNORECASE),
    re.compile(r'(?:minimum|maximum|not\s+less\s+than|not\s+more\s+than)\s+.{3,80}', re.IGNORECASE),
    # Sprint 22: India tender patterns
    re.compile(r'(?:as\s+per|conforming\s+to|in\s+accordance\s+with)\s+(?:IS|BS|ASTM|CPWD|NBC).{3,120}', re.IGNORECASE),
    re.compile(r'(?:approved\s+by|subject\s+to\s+approval\s+of)\s+(?:the\s+)?(?:engineer|architect|client|employer).{0,80}', re.IGNORECASE),
    re.compile(r'(?:the\s+contractor\s+shall|contractor\s+is\s+required\s+to)\s+.{5,120}', re.IGNORECASE),
    re.compile(r'(?:work\s+shall\s+be|works?\s+to\s+be)\s+(?:carried\s+out|executed|completed)\s+.{5,120}', re.IGNORECASE),
    re.compile(r'(?:prior\s+approval|written\s+approval|prior\s+permission)\s+.{5,80}', re.IGNORECASE),
    re.compile(r'(?:should\s+not|shall\s+not)\s+(?:exceed|be\s+less\s+than|be\s+more\s+than)\s+.{3,80}', re.IGNORECASE),
    re.compile(r'(?:tolerance|permissible\s+variation)\s+.{3,80}', re.IGNORECASE),
]

# Material specification indicators
MATERIAL_PATTERNS = [
    re.compile(r'\bM-?\d{2,3}\b'),                    # M25, M-30 (concrete grade)
    re.compile(r'\bFe\s*\d{3}\b'),                     # Fe415, Fe500
    re.compile(r'\bgrade\s+[A-Z]?\d+', re.IGNORECASE),
    re.compile(r'\b\d+\s*mm\s+(?:thick|thk|dia)\b', re.IGNORECASE),
    re.compile(r'\bRCC\b'),
    re.compile(r'\bPCC\b'),
    re.compile(r'\b(?:cement|concrete|steel|brick|block|mortar)\s+(?:grade|type|class)', re.IGNORECASE),
]

# Category keywords
CATEGORY_KEYWORDS = {
    "material": ["material", "grade", "cement", "concrete", "steel", "brick",
                  "mortar", "aggregate", "sand", "timber", "glass", "aluminum",
                  "aluminium", "paint", "tile", "marble", "granite"],
    "workmanship": ["workmanship", "finish", "tolerance", "joint", "curing",
                     "compaction", "mixing", "placing", "laying", "installation",
                     "fixing", "welding", "fabrication"],
    "standard": ["is:", "bs:", "astm", "en:", "iso:", "cpwd", "nbc", "code",
                  "standard", "specification clause"],
    "testing": ["test", "testing", "cube", "slump", "sample", "inspection",
                "quality", "check", "compliance", "certified"],
}


# =============================================================================
# EXTRACTION
# =============================================================================

def _categorize(text: str) -> str:
    """Categorize a requirement by keyword matching."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "general"


# Trade classification keywords (maps to Trade enum values)
_TRADE_KEYWORDS = {
    "civil": ["earthwork", "excavation", "backfill", "pcc", "soil",
              "grading", "site clearance", "dewatering", "road", "pavement"],
    "structural": ["rcc", "reinforcement", "formwork", "footing", "column",
                   "beam", "slab", "steel", "concrete", "staircase", "rebar",
                   "bar bending", "foundation"],
    "architectural": ["brick", "block", "masonry", "plaster", "door",
                      "window", "railing", "partition", "false ceiling"],
    "electrical": ["wiring", "cable", "panel", "lighting", "switch",
                   "socket", "earthing", "transformer", "generator"],
    "plumbing": ["plumbing", "drainage", "sanitary", "water supply",
                 "pipe", "cistern", "pump", "sewage"],
    "mep": ["hvac", "duct", "air conditioning", "chiller", "fire fighting",
            "sprinkler", "ventilation"],
    "finishes": ["paint", "tile", "polish", "marble", "granite", "laminate",
                 "flooring", "dado", "skirting", "putty", "primer"],
}


def _classify_trade(text: str) -> str:
    """Classify a requirement into a construction trade by keyword matching."""
    text_lower = text.lower()
    best_trade = "general"
    best_hits = 0
    for trade, keywords in _TRADE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > best_hits:
            best_hits = hits
            best_trade = trade
    return best_trade


# Pattern for standards/code extraction
_STANDARDS_RE = re.compile(
    r'\b('
    r'IS\s*[:\-]?\s*\d{3,5}(?:\s*[:\-]\s*\d{4})?'
    r'|BS\s*[:\-]?\s*\d{3,5}'
    r'|ASTM\s+[A-Z]\d+'
    r'|EN\s*[:\-]?\s*\d{3,5}'
    r'|ISO\s*[:\-]?\s*\d{3,5}'
    r'|NBC\s+\d+'
    r'|CPWD\s+\d+'
    r')\b',
    re.IGNORECASE,
)


def _extract_standards_codes(text: str) -> list:
    """Extract IS/ASTM/BS/EN/NBC code references from text."""
    if not text:
        return []
    codes = []
    seen = set()
    for m in _STANDARDS_RE.finditer(text):
        code = " ".join(m.group(0).split())  # normalize whitespace
        code_key = code.upper().replace(" ", "")
        if code_key not in seen:
            seen.add(code_key)
            codes.append(code)
    return codes


# Pattern for approved makes
_MAKES_RE = re.compile(
    r'(?:approved\s+makes?|make\s*[:\-]|manufacturer\s*[:\-]|brand\s*[:\-])'
    r'\s*(.{5,200})',
    re.IGNORECASE,
)


def _extract_approved_makes(text: str) -> list:
    """Extract approved makes/manufacturer lists from text."""
    if not text:
        return []
    makes = []
    for m in _MAKES_RE.finditer(text):
        raw = m.group(1).strip()
        # Split by / , or
        parts = re.split(r'[/,]|\bor\b', raw)
        for p in parts:
            p = p.strip().strip('.')
            # Keep only reasonable make names (2-40 chars, not just numbers)
            if 2 <= len(p) <= 40 and not p.isdigit():
                makes.append(p)
            if len(makes) >= 10:
                break
        if len(makes) >= 10:
            break
    return makes


def _clean_ocr_text(text: str) -> str:
    """Pre-clean OCR text for better pattern matching.

    - Collapse multiple spaces (OCR artifact)
    - Fix common OCR broken words
    - Normalize line breaks
    """
    # Collapse runs of spaces (but preserve single newlines)
    cleaned = re.sub(r'[ \t]{2,}', ' ', text)
    # Fix OCR broken words: "s h a l l" → "shall" (rare but handle)
    # Fix "con crete" → "concrete" etc. (common OCR errors)
    cleaned = re.sub(r'(?<=[a-z])\s(?=[a-z]{2})', '', cleaned)
    return cleaned


def extract_requirements(
    text: str,
    source_page: int,
    sheet_id: Optional[str],
    doc_type: str,
) -> List[dict]:
    """
    Extract requirements from a notes/spec/legend/conditions page.

    Returns list of dicts:
        [{text, category, trade, standards_codes, approved_makes,
          source_page, sheet_id, confidence, doc_type}]
    """
    requirements = []
    seen_texts = set()

    # Sprint 22: Pre-clean OCR text for better matching
    cleaned = _clean_ocr_text(text)

    def _add(req_text: str, confidence: float = 0.6):
        # Normalize whitespace
        req_text = " ".join(req_text.split())
        # Skip very short or already seen
        if len(req_text) < 15 or req_text in seen_texts:
            return
        seen_texts.add(req_text)
        requirements.append({
            "text": req_text[:300],
            "category": _categorize(req_text),
            "trade": _classify_trade(req_text),
            "standards_codes": _extract_standards_codes(req_text),
            "approved_makes": _extract_approved_makes(req_text),
            "source_page": source_page,
            "sheet_id": sheet_id,
            "confidence": confidence,
            "doc_type": doc_type,
        })

    # 1. Extract "shall be" / "must be" / India-specific requirement sentences
    for pattern in REQUIREMENT_PATTERNS:
        for match in pattern.finditer(cleaned):
            _add(match.group(0), confidence=0.8)

    # 2. Extract code references (IS:456, ASTM C150, etc.)
    for pattern in CODE_PATTERNS:
        for match in pattern.finditer(cleaned):
            # Get surrounding context (whole line)
            start = cleaned.rfind('\n', 0, match.start()) + 1
            end = cleaned.find('\n', match.end())
            if end == -1:
                end = min(match.end() + 150, len(cleaned))
            context = cleaned[start:end].strip()
            _add(context, confidence=0.7)

    # 3. Extract numbered items (1. ... , 2. ... , i) ... , a) ... )
    numbered_pattern = re.compile(
        r'(?:^|\n)\s*(?:\d{1,3}[\.\)]\s+|[a-z][\.\)]\s+|[ivxlc]+[\.\)]\s+)(.{15,200})',
        re.IGNORECASE | re.MULTILINE,
    )
    for match in numbered_pattern.finditer(cleaned):
        _add(match.group(0).strip(), confidence=0.5)

    # 4. Extract material specification lines
    for pattern in MATERIAL_PATTERNS:
        for match in pattern.finditer(cleaned):
            start = cleaned.rfind('\n', 0, match.start()) + 1
            end = cleaned.find('\n', match.end())
            if end == -1:
                end = min(match.end() + 150, len(cleaned))
            context = cleaned[start:end].strip()
            _add(context, confidence=0.65)

    # 5. Sprint 22: Extract clause-structured paragraphs
    # India tenders use "Clause X.Y.Z:" format
    clause_pattern = re.compile(
        r'(?:^|\n)\s*(?:clause|cl\.?)\s*(\d+(?:\.\d+)*)\s*[:\-]\s*(.{15,200})',
        re.IGNORECASE | re.MULTILINE,
    )
    for match in clause_pattern.finditer(cleaned):
        _add(match.group(0).strip(), confidence=0.6)

    return requirements
