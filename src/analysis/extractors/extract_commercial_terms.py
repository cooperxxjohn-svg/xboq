"""
Commercial / Contract-clause extractor.

Parses OCR text from conditions, spec, and addendum pages to identify
key commercial terms needed for bid pricing:
    LD, retention, warranty/DLP, bid validity, EMD, PBG,
    mobilization advance, insurance, escalation, bid_deadline.

Pure module — no Streamlit, no I/O.
"""

import re
from datetime import date, datetime
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
    # Insurance / CAR policy — Indian (lakh/crore) AND US/international (million/$)
    (
        "insurance",
        re.compile(
            r'(?:insurance|CAR\s+policy|contractor.?s?\s+all\s+risk'
            r'|general\s+liability|commercial\s+general\s+liability)'
            r'[^.]{0,120}?'
            r'(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore|million|%)',
            re.IGNORECASE,
        ),
        1, None, 2,  # unit from group 2
    ),
    # Retainage (US equivalent of retention)
    (
        "retention",
        re.compile(
            r'retainage\s*(?:of|[:\-])?\s*(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Bid bond / bid security (US) — percentage or dollar amount
    (
        "emd_bid_security",
        re.compile(
            r'(?:bid\s+bond|bid\s+security|bidder.?s?\s+bond)'
            r'\s*(?:of|[:\-])?\s*(?:\$\s*)?(\d[\d,]*(?:\.\d+)?)\s*(%|percent|million|thousand)?',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # Payment terms — net days (US convention: Net 30, Net 45)
    (
        "payment_terms",
        re.compile(
            r'(?:net|payment\s+within|pay\s+within)\s+(\d+)\s*(days?|calendar\s+days?)',
            re.IGNORECASE,
        ),
        1, None, 2,
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
    # NIT Estimated Cost / Put-to-Tender Amount (India: "Estimated Cost Rs. X Crore")
    (
        "nit_estimated_cost",
        re.compile(
            r'(?:estimated\s+cost(?:\s+put\s+to\s+tender)?'
            r'|cost\s+put\s+to\s+tender'
            r'|tender\s+value'
            r'|approximate\s+(?:estimated\s+)?cost)'
            r'\s*(?:of\s+work\s*)?'
            r'[^.]{0,200}?'
            r'(?:Rs\.?\s*|INR\s*|₹\s*)?'
            r'(\d[\d,]*(?:\.\d+)?)\s*(crore|lakh|lac|million|billion)?',
            re.IGNORECASE,
        ),
        1, None, 2,  # value group=1, unit from group 2
    ),
    # Sprint 22: Completion time / period
    (
        "completion_time",
        re.compile(
            r'(?:time\s+for\s+completion|completion\s+period|period\s+of\s+completion'
            r'|contract\s+period|project\s+duration)'
            r'\s*(?:of|[:\-])?\s*(\d+)\s*(day|week|month|year|days|weeks|months|years)',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # Sprint 22: Payment terms
    (
        "payment_terms",
        re.compile(
            r'(?:payment\s+(?:shall\s+be\s+made|within|of\s+running\s+account)|running\s+account\s+bill)'
            r'[^.]{0,80}?(\d+)\s*(day|days|month|months)',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # Sprint 22: Dispute resolution / Arbitration
    (
        "dispute_resolution",
        re.compile(
            r'(?:arbitration|dispute\s+resolution|settlement\s+of\s+disputes?)'
            r'\s*(?:clause|under|as\s+per)',
            re.IGNORECASE,
        ),
        None, None, None,  # boolean
    ),
    # Sprint 22: Security deposit (percentage)
    (
        "security_deposit",
        re.compile(
            r'security\s+deposit\s*(?:of|[:\-@])?\s*(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ),
        1, "%", None,
    ),
    # Sprint 22: Penalty clause
    (
        "penalty",
        re.compile(
            r'penalty\s+(?:of|@|[:\-])?\s*(\d+(?:\.\d+)?)\s*%'
            r'(?:\s*(?:per|\/|each)\s*(day|week|month))?',
            re.IGNORECASE,
        ),
        1, "%", 2,
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
    # Sprint 22: Completion fallback "complete within X months"
    (
        "completion_time",
        re.compile(
            r'(?:complete|completed)\s+within\s+(\d+)\s*(day|week|month|year)s?',
            re.IGNORECASE,
        ),
        1, None, 2,
    ),
    # Sprint 22: Penalty fallback "LD/penalty @ X%"
    (
        "penalty",
        re.compile(
            r'(?:penalty|LD)\s*@\s*(\d+(?:\.\d+)?)\s*%\s*(?:per\s*(day|week|month))?',
            re.IGNORECASE,
        ),
        1, "%", 2,
    ),
    # Sprint 22: Interest-free advance
    (
        "mobilization_advance",
        re.compile(
            r'(?:interest[\s-]*free\s+advance|advance\s+payment)\s*(?:of|[:\-])?\s*(\d+(?:\.\d+)?)\s*%',
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


# =============================================================================
# BID DEADLINE EXTRACTION  (absolute date, not duration)
# =============================================================================

_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Patterns that capture an absolute bid/submission/closing date
_DEADLINE_LABEL = (
    r'(?:last\s+date\s+(?:for\s+)?(?:submission|receipt)|'
    r'bid\s+(?:submission|due|closing|deadline)|'
    r'tender\s+(?:submission|due|closing|deadline)|'
    r'due\s+date\s+for\s+(?:bid|tender|submission)|'
    r'closing\s+date|'
    r'submission\s+deadline|'
    r'date\s+of\s+(?:bid|tender)\s+(?:submission|opening))'
)

# DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
_RE_DMY_NUMERIC = re.compile(
    _DEADLINE_LABEL + r'\s*[:\-]?\s*'
    r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
    re.IGNORECASE,
)

# DD Month YYYY  (e.g. "15 March 2026")
_RE_DMY_TEXT = re.compile(
    _DEADLINE_LABEL + r'\s*[:\-]?\s*'
    r'(\d{1,2})\s+(' + '|'.join(_MONTH_MAP.keys()) + r')\s+(\d{4})',
    re.IGNORECASE,
)

# Month DD, YYYY  (e.g. "March 15, 2026")
_RE_MDY_TEXT = re.compile(
    _DEADLINE_LABEL + r'\s*[:\-]?\s*'
    r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s+(\d{1,2}),?\s+(\d{4})',
    re.IGNORECASE,
)


def _try_date(day: int, month: int, year: int) -> Optional[str]:
    """Return ISO date string or None if invalid."""
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def extract_bid_deadline(text: str, source_page: int) -> Optional[dict]:
    """
    Extract an absolute bid submission deadline from OCR text.

    Returns a dict with keys:
        term_type="bid_deadline", iso_date (str YYYY-MM-DD),
        snippet, source_page, confidence
    or None if no deadline found.
    """
    if not text or not text.strip():
        return None

    def _make(iso: str, m) -> dict:
        snippet = _snippet_around(text, m.start(), m.end())
        return {
            "term_type":  "bid_deadline",
            "iso_date":   iso,
            "snippet":    snippet,
            "source_page": source_page,
            "confidence": 0.80,
        }

    # DD/MM/YYYY
    m = _RE_DMY_NUMERIC.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        iso = _try_date(d, mo, y)
        if iso:
            return _make(iso, m)

    # DD Month YYYY
    m = _RE_DMY_TEXT.search(text)
    if m:
        d = int(m.group(1))
        mo = _MONTH_MAP.get(m.group(2).lower(), 0)
        y = int(m.group(3))
        iso = _try_date(d, mo, y) if mo else None
        if iso:
            return _make(iso, m)

    # Month DD, YYYY
    m = _RE_MDY_TEXT.search(text)
    if m:
        mo = _MONTH_MAP.get(m.group(1).lower(), 0)
        d = int(m.group(2))
        y = int(m.group(3))
        iso = _try_date(d, mo, y) if mo else None
        if iso:
            return _make(iso, m)

    return None
