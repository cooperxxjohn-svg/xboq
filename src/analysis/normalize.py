"""
Normalization layer for extracted data.

Reuses _normalize_unit() and UNIT_NORMALIZE from extract_boq.py.
All functions are pure (no Streamlit, no I/O).
"""

import re
from typing import List, Dict, Optional, Union


# =============================================================================
# INDIAN NUMBER FORMAT PARSING
# =============================================================================

# Indian lakh/crore format: "1,25,000" or "12,50,000.50"
# Western format: "125,000" or "1,250,000.50"
# We handle both by stripping commas after validation.
INDIAN_NUM_RE = re.compile(
    r'^-?\d{1,2}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?$'
)


def _parse_indian_number(s: str) -> Optional[float]:
    """
    Parse numeric strings including Indian lakh format.

    Examples:
        "1,25,000"   -> 125000.0
        "12,50,000.50" -> 1250000.5
        "450"        -> 450.0
        "12,500"     -> 12500.0
        ""           -> None
        "abc"        -> None
    """
    if not s:
        return None
    cleaned = s.strip()
    if not cleaned:
        return None
    # Strip commas regardless of format (works for both Indian and Western)
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# =============================================================================
# REQUIREMENT TEXT PRE-NORMALIZATION (Sprint 7)
# =============================================================================

# Boilerplate prefixes to strip from requirement text
_BOILERPLATE_PREFIX_RE = re.compile(
    r'^(?:'
    r'note\s*:\s*|'
    r'general\s*:\s*|'
    r'ref\s*:\s*|'
    r'\d+[.)]\s*'
    r')',
    re.IGNORECASE,
)

# Unit abbreviation normalization for requirement text
_REQ_UNIT_MAP = {
    "sq.m": "sqm", "sq. m": "sqm", "sq m": "sqm", "sq.mt": "sqm",
    "cu.m": "cum", "cu. m": "cum", "cu m": "cum", "cu.mt": "cum",
    "r.m": "rmt", "r. m": "rmt", "r.mt": "rmt",
    "sq.ft": "sqft", "sq. ft": "sqft",
    "cu.ft": "cuft", "cu. ft": "cuft",
}

# IS/BS code colon normalization: "IS : 456" → "IS 456"
_CODE_COLON_RE = re.compile(r'\b(IS|BS|EN|ASTM|IRC)\s*:\s*', re.IGNORECASE)

# Collapse multiple whitespace
_MULTI_SPACE_RE = re.compile(r'\s+')

# Indian number format in text: replace with plain number
_INDIAN_NUM_IN_TEXT_RE = re.compile(r'\b(\d{1,2}(?:,\d{2})*(?:,\d{3}))\b')


def normalize_requirement_text(text: str) -> str:
    """
    Normalize requirement text for comparison (reduces false deltas).

    Steps:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces/newlines to single space
    - Remove common boilerplate prefixes
    - Normalize unit abbreviations (sq.m → sqm, cu.m → cum, etc.)
    - Normalize Indian-format numbers in text (1,25,000 → 125000)
    - Normalize IS/BS code colons (IS : 456 → IS 456)
    - Strip trailing punctuation (periods, colons, semicolons)
    """
    if not text:
        return ""
    result = text.lower().strip()
    # Collapse whitespace
    result = _MULTI_SPACE_RE.sub(" ", result)
    # Strip boilerplate prefix
    result = _BOILERPLATE_PREFIX_RE.sub("", result).strip()
    # Normalize unit abbreviations
    for old, new in _REQ_UNIT_MAP.items():
        result = result.replace(old, new)
    # Normalize Indian numbers in text
    def _replace_indian(m: re.Match) -> str:
        return m.group(0).replace(",", "")
    result = _INDIAN_NUM_IN_TEXT_RE.sub(_replace_indian, result)
    # Normalize IS/BS code colons
    result = _CODE_COLON_RE.sub(lambda m: m.group(1).upper() + " ", result)
    # Strip trailing punctuation
    result = result.rstrip(".:;,")
    return result.strip()


# =============================================================================
# BOQ ITEM NORMALIZATION
# =============================================================================

# Keywords that indicate a row is administrative/contractual, NOT a BOQ work item.
# Only applied when BOTH qty and unit are missing — genuine BOQ items rarely lack both.
_NON_BOQ_KEYWORDS = re.compile(
    r'\b('
    r'clause|condition|article|schedule of|annexure|appendix|'
    r'tender notice|notice inviting|nit|general condition|special condition|'
    r'instruction to bidder|bid document|technical specification|'
    r'penalty|liquidated damage|retention|earnest money|emd|'
    r'time limit|period of completion|defect liability|'
    r'contractor shall|engineer in charge|superintending engineer|'
    r'the contractor|the employer|the engineer|work order|'
    r'bank guarantee|performance security|arbitration|dispute|jurisdiction'
    r')\b',
    re.IGNORECASE,
)

# Minimum description length for a real BOQ work item (very short = header/page noise)
_MIN_DESC_LEN = 10


def _is_non_boq_row(item: dict) -> bool:
    """Return True if this row is administrative/contractual content, not a BOQ work item.

    Criterion: both qty AND unit are missing, AND description matches non-BOQ keywords.
    We never reject rows that have either qty or unit — they may be valid provisional items.
    """
    flags = item.get("flags") or []
    if "qty_missing" not in flags or "unit_missing" not in flags:
        return False  # has qty or unit → keep regardless
    desc = (item.get("description") or "").strip()
    if len(desc) < _MIN_DESC_LEN:
        return True  # too short to be a work item
    return bool(_NON_BOQ_KEYWORDS.search(desc))


def normalize_boq_items(items: List[dict]) -> List[dict]:
    """
    Normalize BOQ items in-place (returns new list of dicts):

    - Unit: canonical form via _normalize_unit (reused from extract_boq.py)
    - Qty/Rate: parse Indian-format strings to float
    - Description: strip trailing whitespace and dots
    - Non-BOQ rows (both qty+unit missing AND administrative keywords) are excluded.
    """
    # Late import to avoid circular dependency at module level
    from .extractors.extract_boq import _normalize_unit, infer_boq_trade, flag_boq_item

    normalized = []
    for item in items:
        new = dict(item)
        # Unit normalization
        if new.get("unit"):
            new["unit"] = _normalize_unit(new["unit"])
        # Qty: parse string numbers
        if new.get("qty") is not None and isinstance(new["qty"], str):
            parsed = _parse_indian_number(new["qty"])
            if parsed is not None:
                new["qty"] = parsed
        # Rate: parse string numbers
        if new.get("rate") is not None and isinstance(new["rate"], str):
            parsed = _parse_indian_number(new["rate"])
            if parsed is not None:
                new["rate"] = parsed
        # Description cleanup
        if new.get("description"):
            new["description"] = new["description"].strip().rstrip(".")
        # Trade inference (new)
        if "trade" not in new:
            new["trade"] = infer_boq_trade(new.get("description", ""))
        # Item flags (new)
        if "flags" not in new:
            new["flags"] = flag_boq_item(new)
        # Drop administrative/contractual rows that are not work items
        if _is_non_boq_row(new):
            continue
        normalized.append(new)
    return normalized


# =============================================================================
# SCHEDULE ROW NORMALIZATION
# =============================================================================

# Size pattern: "900x2100" or "900X2100" or "900 x 2100" → "900 x 2100"
SIZE_NORMALIZE_RE = re.compile(r'(\d+)\s*[xX\u00d7]\s*(\d+)')


def normalize_schedule_rows(rows: List[dict]) -> List[dict]:
    """
    Normalize schedule rows:

    - Size: "900x2100" → "900 x 2100"
    - Mark: uppercase + strip
    - De-dup by (mark, schedule_type) keeping highest confidence
    """
    normalized = []
    for row in rows:
        new = dict(row)
        # Size normalization
        if new.get("size"):
            new["size"] = SIZE_NORMALIZE_RE.sub(r'\1 x \2', new["size"])
        # Mark normalization
        if new.get("mark"):
            new["mark"] = new["mark"].upper().strip()
        # Also normalize inside fields dict if present
        fields = new.get("fields")
        if isinstance(fields, dict):
            new_fields = dict(fields)
            if new_fields.get("size"):
                new_fields["size"] = SIZE_NORMALIZE_RE.sub(
                    r'\1 x \2', new_fields["size"]
                )
            new["fields"] = new_fields
        normalized.append(new)

    # De-dup by (mark, schedule_type) keeping highest confidence
    seen: Dict[tuple, int] = {}  # key -> index in deduped
    deduped: List[dict] = []
    for row in normalized:
        key = (row.get("mark", ""), row.get("schedule_type", ""))
        if not key[0]:
            # No mark — keep as-is (can't de-dup)
            deduped.append(row)
            continue
        if key in seen:
            idx = seen[key]
            if row.get("confidence", 0) > deduped[idx].get("confidence", 0):
                deduped[idx] = row
        else:
            seen[key] = len(deduped)
            deduped.append(row)
    return deduped


# =============================================================================
# REQUIREMENT NORMALIZATION
# =============================================================================

def normalize_commercial_terms(terms: List[dict]) -> List[dict]:
    """
    Normalize and deduplicate commercial terms.

    - Parse Indian number format in value fields
    - Dedup by term_type, keeping highest confidence per type
    - Normalize unit strings (months/years → month/year)
    """
    if not terms:
        return []

    # Group by term_type, keep highest confidence per type
    best: Dict[str, dict] = {}
    for t in terms:
        tt = t.get("term_type", "")
        if not tt:
            continue
        # Try to parse value if it's a string with Indian formatting
        val = t.get("value")
        if isinstance(val, str):
            parsed = _parse_indian_number(val)
            if parsed is not None:
                t = dict(t, value=parsed)
        # Normalize unit
        unit = (t.get("unit") or "").strip().rstrip("s").lower()
        if unit:
            t = dict(t, unit=unit)
        # Keep highest confidence
        existing = best.get(tt)
        if existing is None or t.get("confidence", 0) > existing.get("confidence", 0):
            best[tt] = t

    return list(best.values())


def normalize_requirements(reqs: List[dict]) -> List[dict]:
    """
    Normalize requirements:

    - De-dup by exact match after lowercase + strip
    - Keeps first occurrence (preserves original ordering)
    """
    seen: set = set()
    deduped: List[dict] = []
    for req in reqs:
        text = req.get("text", "").lower().strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(req)
        elif not text:
            # Keep items with empty text (shouldn't happen, but don't drop)
            deduped.append(req)
    return deduped
