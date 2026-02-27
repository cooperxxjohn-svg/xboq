"""
Commercial / Contract-clause extractor.

Parses OCR text from conditions, spec, and addendum pages to identify
key commercial terms needed for bid pricing:
    LD, retention, warranty/DLP, bid validity, EMD, PBG,
    mobilization advance, insurance, escalation.

Pure module — no Streamlit, no I/O.
"""

import re
from typing import List, Optional


# =============================================================================
# TERM PATTERNS  (case-insensitive)
# Each tuple: (term_type, compiled regex, value_group, unit, cadence_group)
# =============================================================================

_PATTERNS = [
    # Liquidated Damages — percentage + cadence
    (
        "ld_clause",
        re.compile(
            r'(?:liquidated\s+damages?|[^a-zA-Z]LD\s*[:\-])'
            r'[^.]{0,80}?'
            r'(\d+(?:\.\d+)?)\s*%'
            r'(?:\s*(?:per|\/|each)\s*(day|week|month))?',
            re.IGNORECASE,
        ),
        1,   # value group
        "%",
        2,   # cadence group (may be None)
    ),
    # Retention percentage
    (
        "retention",
        re.compile(
            r'retention\s*(?:money|amount)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Warranty / Defect Liability Period
    (
        "warranty_dlp",
        re.compile(
            r'(?:warranty|defect\s+liability\s+period|DLP|maintenance\s+period)'
            r'\s*(?:of|[:\-])?\s*(\d+)\s*(month|year|months|years)',
            re.IGNORECASE,
        ),
        1, None, 2,  # unit comes from group 2
    ),
    # Bid / tender validity
    (
        "bid_validity",
        re.compile(
            r'(?:bid|tender)\s+(?:validity|shall\s+remain\s+valid\s+for)'
            r'\s*(?:of|[:\-])?\s*(\d+)\s*(day|month|days|months)',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # EMD / Bid Security (amount in Rs/INR)
    (
        "emd_bid_security",
        re.compile(
            r'(?:EMD|earnest\s+money\s*(?:deposit)?|bid\s+security)'
            r'\s*(?:of|[:\-])?\s*(?:Rs\.?\s*|INR\s*|₹\s*)?'
            r'(\d[\d,]*(?:\.\d+)?)\s*(?:/[\-])?'
            r'(?:\s*(?:lakh|lac|crore))?',
            re.IGNORECASE,
        ),
        1, "Rs", None,
    ),
    # Performance Bank Guarantee / PBG
    (
        "performance_bond",
        re.compile(
            r'(?:performance\s+(?:bank\s+)?guarantee|PBG|performance\s+bond'
            r'|performance\s+security)'
            r'\s*(?:of|[:\-])?\s*(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Mobilization advance
    (
        "mobilization_advance",
        re.compile(
            r'(?:mobilization|mobilisation)\s+advance\s*(?:of|[:\-])?\s*(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Insurance / CAR policy
    (
        "insurance",
        re.compile(
            r'(?:insurance|CAR\s+policy|contractor.?s?\s+all\s+risk)'
            r'[^.]{0,120}?'
            r'(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore|%)',
            re.IGNORECASE,
        ),
        1, None, 2,  # unit from group 2
    ),
    # Escalation / price variation clause (boolean detection)
    (
        "escalation",
        re.compile(
            r'(?:escalation|price\s+(?:variation|adjustment|escalation))\s+clause',
            re.IGNORECASE,
        ),
        None, None, None,  # boolean — no numeric value
    ),
]

# Broader fallback patterns for terms that may appear in different phrasing
_FALLBACK_PATTERNS = [
    # LD as standalone "LD @ X%"
    (
        "ld_clause",
        re.compile(
            r'\bLD\s*@\s*(\d+(?:\.\d+)?)\s*%\s*per\s*(day|week|month)',
            re.IGNORECASE,
        ),
        1, "%", 2,
    ),
    # Retention as "X% retention"
    (
        "retention",
        re.compile(
            r'(\d+(?:\.\d+)?)\s*%\s*retention',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Security deposit (alias for EMD/PBG)
    (
        "emd_bid_security",
        re.compile(
            r'security\s+deposit\s*[:\-]?\s*(?:Rs\.?\s*|INR\s*)?'
            r'(\d[\d,]*(?:\.\d+)?)',
            re.IGNORECASE,
        ),
        1, "Rs", None,
    ),
    # DLP standalone "DLP X months"
    (
        "warranty_dlp",
        re.compile(
            r'\bDLP\s*[:\-]?\s*(\d+)\s*(month|year)s?',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # Bid validity fallback: "remain valid for X days/months"
    (
        "bid_validity",
        re.compile(
            r'(?:remain|be)\s+valid\s+for\s+(\d+)\s*(day|month|days|months)',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # PBG fallback: "PBG of 10%"
    (
        "performance_bond",
        re.compile(
            r'\bPBG\s+of\s+(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Mobilization advance fallback: "mobilization advance of X%"
    (
        "mobilization_advance",
        re.compile(
            r'(?:mobilization|mobilisation)\s+advance\s+of\s+(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
]


def _snippet_around(text: str, start: int, end: int, window: int = 100) -> str:
    """Extract a snippet of text around the match region."""
    s = max(0, start - window)
    e = min(len(text), end + window)
    snippet = text[s:e].strip()
    # Clean up newlines and excessive whitespace
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet[:200]


def _parse_value(raw: str) -> Optional[float]:
    """Parse a numeric value, handling Indian comma format."""
    if not raw:
        return None
    cleaned = raw.replace(',', '')
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def extract_commercial_terms(
    text: str,
    source_page: int,
    sheet_id: Optional[str] = None,
    doc_type: str = "conditions",
) -> List[dict]:
    """
    Extract commercial/contract terms from OCR text.

    Args:
        text: Raw OCR text from a single page.
        source_page: 0-indexed page number.
        sheet_id: Drawing sheet ID (usually None for conditions pages).
        doc_type: Page type from classifier.

    Returns:
        List of dicts, each with:
            term_type, value, unit, cadence, snippet,
            source_page, sheet_id, confidence, doc_type
    """
    if not text or not text.strip():
        return []

    results = []
    seen = set()  # (term_type, source_page) dedup

    for patterns, confidence in [(_PATTERNS, 0.75), (_FALLBACK_PATTERNS, 0.55)]:
        for entry in patterns:
            term_type, pattern, val_group, unit, cadence_group = entry

            for m in pattern.finditer(text):
                key = (term_type, source_page)
                if key in seen:
                    continue
                seen.add(key)

                # Extract value
                value = None
                if val_group is not None:
                    try:
                        raw_val = m.group(val_group)
                        value = _parse_value(raw_val)
                    except (IndexError, TypeError):
                        pass

                # Boolean detection (escalation)
                if val_group is None:
                    value = True

                # Extract unit from group if needed
                actual_unit = unit
                if actual_unit is None and cadence_group is not None:
                    try:
                        actual_unit = m.group(cadence_group)
                    except (IndexError, TypeError):
                        actual_unit = ""
                if actual_unit:
                    actual_unit = actual_unit.strip().rstrip('s')  # months -> month

                # Extract cadence
                cadence = None
                if cadence_group is not None:
                    try:
                        cadence = m.group(cadence_group)
                        if cadence:
                            cadence = cadence.strip().rstrip('s')
                    except (IndexError, TypeError):
                        pass

                # For warranty_dlp and bid_validity, unit is the time unit
                if term_type in ("warranty_dlp", "bid_validity"):
                    actual_unit = cadence or actual_unit

                snippet = _snippet_around(text, m.start(), m.end())

                results.append({
                    "term_type": term_type,
                    "value": value,
                    "unit": actual_unit or "",
                    "cadence": cadence,
                    "snippet": snippet,
                    "source_page": source_page,
                    "sheet_id": sheet_id,
                    "confidence": confidence,
                    "doc_type": doc_type,
                })

    return results
