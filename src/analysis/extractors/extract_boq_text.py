"""
Text-based BOQ extractor — fallback for scanned/poorly-structured BOQ pages.

When the structured table extractor yields <5 items, this parser attempts to
extract BOQ items from raw OCR text using pattern matching.

Handles formats like:
  1    Earthwork excavation in all types...    cum    250.00    145.00    36,250.00
  2    Providing and laying M25 RCC...         cum     85.00   8,850.00  752,250.00
  a)   Supply and fix UPVC pipe 100mm...       rmt    450.00    185.00    83,250.00
"""
import re
from typing import List, Dict, Optional

# Unit tokens recognized in BOQ text
_UNITS = {
    "sqm", "sqft", "m2", "sft", "sq.m", "sq.ft",
    "cum", "cft", "m3", "cuft",
    "rmt", "m", "rm", "lm", "mtr", "rlm", "rft",
    "nos", "each", "no", "nr", "no.", "nos.",
    "set", "unit", "pcs", "pc",
    "kg", "mt", "kgs", "tonne",
    "ls", "l.s", "lump", "job", "lot",
    "point", "pt", "bag", "litre", "ltr",
}

# Pattern: item_no  description  unit  qty  rate  amount
# Various formats — all columns optional except description
_NUMBER_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(\d{1,3}(?:\.\d{1,3})?|[a-zA-Z]\d{0,2}|[ivxlIVXL]{1,5})\s*[.):]\s*'
    r'(.{10,250})',
    re.MULTILINE,
)

_NUMERIC_RE = re.compile(r'[\d,]+(?:\.\d+)?')


def _parse_unit_qty_rate(tail: str) -> tuple:
    """Extract (unit, qty, rate) from the tail of a BOQ line."""
    unit = None
    qty = None
    rate = None

    tokens = tail.split()
    # Look for unit token in last 6 tokens
    for i, tok in enumerate(reversed(tokens[-6:])):
        if tok.lower().rstrip('.') in _UNITS:
            unit = tok.lower().rstrip('.')
            # Numbers after unit = qty, rate, amount (take first two)
            nums = []
            for t in tokens[-(6-i):]:
                m = re.fullmatch(r'[\d,]+(?:\.\d+)?', t.replace(',', ''))
                if m:
                    try:
                        nums.append(float(t.replace(',', '')))
                    except ValueError:
                        pass
            if len(nums) >= 1:
                qty = nums[0]
            if len(nums) >= 2:
                rate = nums[1]
            break
    return unit, qty, rate


def extract_boq_from_text(
    text: str,
    source_page: int = 0,
    min_desc_len: int = 12,
) -> List[Dict]:
    """
    Parse BOQ items from raw OCR text.
    Returns list of item dicts compatible with BOQ extractor output.
    """
    items = []
    seen_descs = set()

    for m in _NUMBER_RE.finditer(text):
        item_no = m.group(1).strip()
        rest    = m.group(2).strip()

        # Skip if rest is too short or looks like a header/section
        if len(rest) < min_desc_len:
            continue
        if rest.isupper() and len(rest.split()) < 6:
            continue  # likely a section header

        # Split description from unit/qty/rate tail
        # Heuristic: description ends when we hit a known unit token
        desc_parts = []
        tail_start = len(rest)
        tokens = rest.split()
        for i, tok in enumerate(tokens):
            if tok.lower().rstrip('.') in _UNITS:
                tail_start = len(' '.join(tokens[:i]))
                break
            desc_parts.append(tok)

        description = ' '.join(desc_parts).strip()[:250]
        tail = rest[tail_start:].strip()

        if len(description) < min_desc_len:
            continue

        # Dedup
        key = description[:50].lower()
        if key in seen_descs:
            continue
        seen_descs.add(key)

        unit, qty, rate = _parse_unit_qty_rate(tail)

        items.append({
            "item_no":    item_no,
            "description": description,
            "unit":       unit,
            "qty":        qty,
            "rate":       rate,
            "amount":     (qty * rate) if (qty and rate) else None,
            "trade":      "general",
            "section":    "",
            "source_page": source_page,
            "source":     "boq_text",
            "confidence": 0.50,
        })

    return items
