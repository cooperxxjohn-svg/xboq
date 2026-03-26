"""
BOQ Extractor — Parse Bill of Quantities items from OCR text.

Handles doc_type: boq

Sprint 20F: Enhanced with table_router integration and richer item patterns.
Falls back to existing regex/OCR logic when table router returns no rows.
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# UNIT NORMALIZATION (from owner_docs/boq_parser.py)
# =============================================================================

UNIT_NORMALIZE: dict = {
    "sq.m": "sqm", "sq m": "sqm", "square metre": "sqm", "square meter": "sqm",
    "sq.m.": "sqm", "sq.mt": "sqm", "sqmt": "sqm", "m2": "sqm",
    "sqft": "sqft", "sq.ft": "sqft", "sq ft": "sqft", "square feet": "sqft",
    "sft": "sqft", "sq.ft.": "sqft",
    "cu.m": "cum", "cu m": "cum", "cubic metre": "cum", "cubic meter": "cum",
    "cu.m.": "cum", "cu.mt": "cum", "cumt": "cum", "m3": "cum", "cbm": "cum",
    "cft": "cft", "cu.ft": "cft", "cu ft": "cft", "cubic feet": "cft",
    "rm": "rmt", "r.m": "rmt", "r.m.": "rmt", "running metre": "rmt",
    "running meter": "rmt", "rmt.": "rmt", "r.mtr": "rmt", "mtr": "rmt",
    "m": "rmt",
    "no": "nos", "no.": "nos", "numbers": "nos", "each": "nos", "set": "nos",
    "pair": "nos", "sets": "nos", "nos.": "nos", "pcs": "nos", "pc": "nos",
    "kgs": "kg", "kilogram": "kg", "kg.": "kg",
    "mt": "mt", "tonne": "mt", "metric ton": "mt", "t": "mt", "ton": "mt",
    "quintal": "quintal", "qtl": "quintal",
    "l.s": "LS", "l.s.": "LS", "lump sum": "LS", "lumpsum": "LS", "ls": "LS",
    "job": "LS", "l/s": "LS",
    "litre": "liter", "lit": "liter", "ltr": "liter",
    "bag": "bag", "bags": "bag",
    "km": "km", "k.m.": "km",
}

KNOWN_UNITS = {
    "sqm", "sqft", "cum", "cft", "rmt", "nos", "kg", "mt", "LS",
    "liter", "bag", "kl", "quintal", "point", "coat",
}


def _normalize_unit(unit: str) -> str:
    """Normalize a unit string to standard form."""
    u = unit.strip().lower()
    if u in UNIT_NORMALIZE:
        return UNIT_NORMALIZE[u]
    if u in KNOWN_UNITS:
        return u
    # Check if the raw unit (case-preserved) is known
    if unit.strip() in KNOWN_UNITS:
        return unit.strip()
    return unit.strip()


# =============================================================================
# ITEM NUMBER PATTERNS (Sprint 20F: expanded)
# =============================================================================

# Matches: 1, 1.1, 1.1.a, 2.3, 1.01, A-1, 2(a), etc.
ITEM_NO_PATTERN = re.compile(
    r'^[\s]*(\d{1,3}(?:\.\d{1,3})?(?:\.[a-z])?)\s+',
    re.IGNORECASE,
)

# Sprint 20F: Extended item number pattern for Indian BOQ formats
# Matches: "1", "1.1", "1.01", "1.1.a", "A-1", "2(a)", "ii", "iii"
EXTENDED_ITEM_NO = re.compile(
    r'^\s*'
    r'('
    r'\d{1,3}(?:\.\d{1,3}){0,2}(?:\.[a-z])?'  # 1, 1.1, 1.01, 1.1.a
    r'|[A-Z]-?\d{1,4}'                          # A-1, B2, A101, A1011
    r'|\d{1,3}\s*\([a-z]\)'                      # 2(a), 3(b)
    r'|[ivx]{1,4}'                                # roman numerals i-iv
    r')'
    r'[\s\.\)]+',
    re.IGNORECASE,
)

# Full BOQ line: item_no, description, unit, quantity, rate
_UNIT_RE = (
    r'sqm|sq\.?\s*m\.?|cum|cu\.?\s*m\.?|rmt|r\.?m\.?|mtr'
    r'|nos|no\.?|each|set|pair|pcs?'
    r'|kg\.?|mt|tonne?|quintal|qtl'
    r'|sqft|sq\.?\s*ft\.?|sft|cft|cu\.?\s*ft\.?'
    r'|LS|l\.?s\.?|lump\s*sum|job'
    r'|kl|liter|litre|ltr|bag|km|m2|m3|cbm'
)
BOQ_LINE_PATTERN = re.compile(
    r'^\s*(\d{1,3}(?:\.\d{1,3})?(?:\.[a-z])?)'    # item number
    r'\s+(.{10,200}?)'                               # description (greedy but bounded)
    r'\s+(' + _UNIT_RE + r')'                         # unit (expanded)
    r'\s+(\d[\d,]*(?:\.\d+)?)'                        # quantity (Indian comma format)
    r'(?:\s+(\d[\d,]*(?:\.\d+)?))?',                  # optional rate (Indian comma format)
    re.IGNORECASE,
)

# Simpler pattern: just number + description + optional unit/qty
SIMPLE_ITEM_PATTERN = re.compile(
    r'^\s*(\d{1,3}(?:\.\d{1,3})?)\s+'             # item number
    r'(.{10,300})',                                  # rest of line
    re.IGNORECASE,
)

# Quantity detection in line
QTY_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:nos|sqm|cum|rmt|kg|mt|no\.?|each|set)', re.IGNORECASE)

# Rate detection
RATE_PATTERN = re.compile(r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)', re.IGNORECASE)

# Sprint 22: Tabular BOQ line — columns separated by 2+ spaces or tabs
# Matches: "1.1    Earthwork excavation    cum    250.00    450.00    112500.00"
TABULAR_LINE_PATTERN = re.compile(
    r'^\s*(\d{1,3}(?:\.\d{1,3}){0,2}(?:\.[a-z])?)'  # item number
    r'\s{2,}(.{8,200}?)'                                # description (2+ space gap)
    r'\s{2,}(\S{1,15})'                                  # unit (any short token)
    r'\s{2,}(\d[\d,]*(?:\.\d+)?)'                         # quantity
    r'(?:\s{2,}(\d[\d,]*(?:\.\d+)?))?'                    # optional rate
    r'(?:\s{2,}(\d[\d,]*(?:\.\d+)?))?',                   # optional amount
    re.IGNORECASE,
)

# 7-column BOQ: S.No | DSR/SOR Reference | Description | Unit | Qty | Rate | Amount
# Seen in Indian PWD / CPWD tenders (e.g. SPG format with schedule prefix A101, A1011)
# The DSR column may contain "DSR 2023 CIVIL Item No. 5.33.1.1" or simply "-"
TABULAR_7COL_PATTERN = re.compile(
    r'^\s*([A-Z]-?\d{2,4}|\d{1,3}(?:\.\d{1,3}){0,2})'   # item number (A1011 or 1.1.1)
    r'\s{2,}(DSR[^\n]{3,80}?|SOR[^\n]{3,80}?|[-–—])'     # DSR/SOR ref or dash
    r'\s{2,}(.{8,180}?)'                                    # description
    r'\s{2,}(\S{1,12})'                                     # unit
    r'\s{2,}(\d[\d,]*(?:\.\d+)?)'                           # qty
    r'(?:\s{2,}(\d[\d,]*(?:\.\d+)?))?'                      # optional rate
    r'(?:\s{2,}(\d[\d,]*(?:\.\d+)?))?',                     # optional amount
    re.IGNORECASE,
)

# Sprint 20F: Row classification patterns
_HEADER_LIKE = re.compile(
    r'\b(item\s*no|s\.?\s*no|description|unit|qty|quantity|rate|amount)\b',
    re.IGNORECASE,
)
_TOTAL_LIKE = re.compile(
    r'\b(sub\s*total|total|grand\s*total|carried\s*forward|brought\s*forward|carried\s*over)\b',
    re.IGNORECASE,
)

# Section-header detection: lines that label a group of BOQ items but carry
# no qty/rate data.  Two sub-patterns:
#   1. Numbered heading  – "3.0 STRUCTURAL WORKS", "2) CIVIL WORKS"
#   2. ALL-CAPS heading  – "FINISHING WORKS", "PLUMBING & SANITARY"
_BOQ_SECTION_RE = re.compile(
    r'^\s*(?:'
    r'\d+[.)]\s+[A-Z][A-Z\s/&,()–-]{4,}'    # numbered heading
    r'|[A-Z][A-Z\s/&,()–-]{8,}$'             # ALL-CAPS line ≥9 chars
    r')',
)
# A unit token present in the line strongly suggests a data row, not a header
_SECTION_UNIT_GUARD = re.compile(
    r'\b(?:cum|sqm|m3|m2|sqft|rmt|rm|nos|no|nr|each|kg|tonne|ls|lot)\b',
    re.IGNORECASE,
)


# Regex to pull a DSR / SOR reference from a text blob.
# Matches: "DSR 2023 CIVIL Item No. 2.6.1", "SOR 2024 Item 5.33.1.1"
_DSR_EXTRACT_RE = re.compile(
    r'\b((?:DSR|SOR)\s+(?:\d{4}\s+)?(?:[A-Z][A-Z&/ ]{0,20}\s+)?'
    r'(?:Item\s+No\.?\s+)?[\d]{1,3}(?:\.[\d]{1,3}){1,4})',
    re.IGNORECASE,
)

# Alphanumeric parent-item: letter + exactly 3 digits = section header in SPG format
# (e.g. A101, B103, R201)  — these carry no qty/rate data
_ALPHA_PARENT_RE = re.compile(r'^[A-Z]\d{3}$')

# Schedule letter → trade (SPG / CPWD convention)
#   A/A1 = civil, B/B1 = structural, C = masonry/arch, D/D1 = plaster/finishes,
#   E/E1 = tiles/finishes, F = cladding, G = false ceiling, H = doors & windows,
#   I = roof, J = lift/elevators, K/K1 = plumbing, L = drainage/sewage,
#   M = external services, R/S/T = electrical
_SCHEDULE_PREFIX_TRADE: Dict[str, str] = {
    "A": "civil",
    "B": "structural",
    "C": "architectural",
    "D": "finishes",
    "E": "finishes",
    "F": "architectural",
    "G": "architectural",
    "H": "architectural",
    "I": "architectural",
    "J": "mep",
    "K": "plumbing",
    "L": "plumbing",
    "M": "plumbing",
    "R": "electrical",
    "S": "electrical",
    "T": "electrical",
}


def _trade_from_item_no(item_no: str) -> Optional[str]:
    """Return trade from schedule-letter prefix, or None if not applicable."""
    if not item_no:
        return None
    prefix = item_no[0].upper()
    if prefix.isalpha():
        return _SCHEDULE_PREFIX_TRADE.get(prefix)
    return None


# =============================================================================
# ROW CLASSIFICATION HELPERS (Sprint 20F)
# =============================================================================

def _classify_row(text: str) -> Optional[str]:
    """Classify a row/line as header, total, or None (data row).

    Returns: 'header_like', 'subtotal_or_total', or None.
    """
    if not text:
        return None
    text_lower = text.lower()
    header_hits = len(_HEADER_LIKE.findall(text_lower))
    if header_hits >= 2:
        return "header_like"
    if _TOTAL_LIKE.search(text_lower):
        return "subtotal_or_total"
    return None


def _is_continuation_row(text: str) -> bool:
    """Check if a line is likely a continuation of the previous item description.

    Continuation rows: don't start with a number, are short-to-medium length,
    and have no unit/qty pattern.
    """
    stripped = text.strip()
    if not stripped or len(stripped) < 5:
        return False
    # Starts with letter or special char (not a number = no item_no)
    if re.match(r'^\d', stripped):
        return False
    # Should NOT have unit/qty pattern
    if QTY_PATTERN.search(stripped):
        return False
    # Should be medium length (likely a description continuation)
    return len(stripped) < 200


def _is_boq_section_header(line: str) -> bool:
    """Return True if *line* looks like a BOQ section/group heading.

    Criteria:
    - Matches the numbered-heading or ALL-CAPS pattern in ``_BOQ_SECTION_RE``
    - Contains NO unit tokens (unit token → data row, not a heading)
    - Length between 6 and 120 characters
    """
    stripped = line.strip()
    if len(stripped) < 6 or len(stripped) > 120:
        return False
    if _SECTION_UNIT_GUARD.search(stripped):
        return False
    return bool(_BOQ_SECTION_RE.match(stripped))


# =============================================================================
# DEDUP KEY HELPER
# =============================================================================

def _dedup_key(item_no: str, description: str) -> str:
    """Build a normalized dedup key from item_no + description prefix.

    Normalizes by lowercasing, collapsing whitespace, and using an 80-char
    prefix so that items differing only in specs (e.g. M20 vs M25 concrete)
    are not silently merged.
    """
    desc = (description or "").lower().strip()
    desc = re.sub(r'\s+', ' ', desc)  # collapse multiple spaces / tabs
    return f"{(item_no or '').strip()}:{desc[:80]}"


# =============================================================================
# TABLE ROUTER INTEGRATION (Sprint 20F)
# =============================================================================

def _parse_routed_rows(
    rows: List[Any],
    headers: Optional[List[str]],
    source_page: int,
) -> List[dict]:
    """Parse structured table rows from table_router into BOQ items.

    Handles both list[list[str]] and list[dict] formats.
    """
    items = []
    seen = set()
    current_section: str = ""   # tracks the most recent section header row

    # Try to detect column indices from headers
    col_map = _detect_column_map(headers)

    for row in rows:
        # Normalize to list of strings
        if isinstance(row, dict):
            cells = list(row.values())
        elif isinstance(row, (list, tuple)):
            cells = [str(c) for c in row]
        else:
            continue

        if not cells or not any(c.strip() for c in cells if c):
            continue

        # Join all cells for classification
        full_text = " ".join(c.strip() for c in cells if c)

        # Skip header/total rows
        row_class = _classify_row(full_text)
        if row_class:
            continue

        # Extract fields using column map or positional heuristics,
        # carrying the current section header through to the item dict.
        item = _extract_item_from_cells(cells, col_map, source_page, current_section)
        if item:
            key = _dedup_key(item['item_no'], item.get('description'))
            if key not in seen:
                seen.add(key)
                items.append(item)
        else:
            # Row was not a valid data item — try to use it as a section header.
            # Case 1: classic ALL-CAPS / numbered heading
            if _is_boq_section_header(full_text):
                current_section = full_text[:80]
            # Case 2: alpha parent item (A101, B103) — use joined cells as header
            elif cells and _ALPHA_PARENT_RE.match(cells[0].strip()):
                # Pull description text from the appropriate column
                desc_cell = cells[2].strip() if len(cells) >= 7 else (
                    cells[1].strip() if len(cells) >= 2 else ""
                )
                if desc_cell:
                    current_section = f"{cells[0].strip()} — {desc_cell[:70]}"

    return items


def _detect_column_map(headers: Optional[List[str]]) -> Dict[str, int]:
    """Detect which column index maps to which BOQ field from header row."""
    col_map: Dict[str, int] = {}
    if not headers:
        return col_map

    for i, h in enumerate(headers):
        hl = (h or "").lower().strip()
        if re.search(r'\b(item\s*no|s\.?\s*no|sl\.?\s*no)\b', hl):
            col_map["item_no"] = i
        elif re.search(r'\bdescription\b|\bparticulars\b|\bitem\s*of\s*work\b', hl):
            col_map["description"] = i
        elif re.search(r'\bunit\b', hl):
            col_map["unit"] = i
        elif re.search(r'\bqty\b|\bquantity\b', hl):
            col_map["qty"] = i
        elif re.search(r'\brate\b', hl):
            col_map["rate"] = i
        elif re.search(r'\bamount\b|\btotal\b', hl):
            col_map["amount"] = i
        elif re.search(
            r'\bdsr\b|\bsor\b|\breference\s*of\s*rate\b|'
            r'\bschedule\s*of\s*rates?\b|\bdsr\s*ref\b|\brate\s*ref\b',
            hl,
        ):
            col_map["sor_code"] = i

    return col_map


def _extract_item_from_cells(
    cells: List[str],
    col_map: Dict[str, int],
    source_page: int,
    section: str = "",
) -> Optional[dict]:
    """Extract a BOQ item from a row of table cells.

    Uses column map if available, otherwise falls back to positional heuristics.
    Handles 5-column (item/desc/unit/qty/rate) and 7-column (item/DSR/desc/unit/qty/rate/amount).
    Adds sor_code and trade fields to output.
    """
    # Helper to safely get cell value
    def _cell(idx: int) -> str:
        if 0 <= idx < len(cells):
            return (cells[idx] or "").strip()
        return ""

    item_no: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    qty: Optional[float] = None
    rate: Optional[float] = None
    sor_code: Optional[str] = None
    confidence = 0.6

    if col_map:
        # Use header-mapped columns
        if "item_no" in col_map:
            item_no = _cell(col_map["item_no"])
        if "description" in col_map:
            description = _cell(col_map["description"])
        if "unit" in col_map:
            raw_unit = _cell(col_map["unit"])
            unit = _normalize_unit(raw_unit) if raw_unit else None
        if "qty" in col_map:
            qty = _safe_float(_cell(col_map["qty"]))
        if "rate" in col_map:
            rate = _safe_float(_cell(col_map["rate"]))
        if "sor_code" in col_map:
            raw_sor = _cell(col_map["sor_code"])
            sor_code = raw_sor if raw_sor and raw_sor not in ("-", "–", "—") else None
        confidence = 0.75
    else:
        # Positional heuristics — detect 7-column (with DSR ref) vs 5-column layout.
        # 7-col: item_no | DSR_ref | description | unit | qty | rate | amount
        # 5-col: item_no | description | unit | qty | rate
        if len(cells) >= 3:
            item_no = cells[0].strip()
            # Detect if cell[1] is a DSR/SOR reference (or "-") rather than description
            cell1 = cells[1].strip() if len(cells) > 1 else ""
            is_dsr_col = bool(
                re.match(r'^(?:DSR|SOR)\b', cell1, re.IGNORECASE)
                or (cell1 in ("-", "–", "—") and len(cells) >= 7)
            )
            if is_dsr_col and len(cells) >= 4:
                # 7-column layout
                raw_sor = cell1
                sor_code = raw_sor if raw_sor not in ("-", "–", "—") else None
                description = cells[2].strip() if len(cells) > 2 else ""
                if len(cells) >= 5:
                    raw_unit = cells[3].strip()
                    unit = _normalize_unit(raw_unit) if raw_unit else None
                    qty = _safe_float(cells[4].strip())
                if len(cells) >= 6:
                    rate = _safe_float(cells[5].strip())
            else:
                # 5-column layout (standard)
                description = cell1
                if len(cells) >= 4:
                    raw_unit = cells[2].strip()
                    unit = _normalize_unit(raw_unit) if raw_unit else None
                    qty = _safe_float(cells[3].strip())
                if len(cells) >= 5:
                    rate = _safe_float(cells[4].strip())

    # Validate item_no looks like a real item number
    # Now accepts: 1, 1.1, 1.1.a, A-1, A101, A1011, 2(a)
    if not item_no:
        return None
    if not re.match(
        r'^\d{1,3}(?:\.\d{1,3}){0,2}(?:\.[a-z])?$'  # numeric: 1, 1.1, 1.1.a
        r'|^[A-Z]-?\d{1,4}$'                           # alpha: A-1, A101, A1011
        r'|^\d{1,3}\s*\([a-z]\)$',                      # paren: 2(a)
        item_no, re.IGNORECASE,
    ):
        return None

    # Need at least a description
    if not description or len(description) < 5:
        return None

    # Parent-item detection: letter + exactly 3 digits with no qty/unit
    # = section header in SPG/CPWD alphanumeric format (A101, B103, R201)
    # Emit these as None so the caller can use them as section headers.
    if _ALPHA_PARENT_RE.match(item_no) and not unit and qty is None:
        return None  # caller will treat as section header

    # Flag row as incomplete but still useful if it has description
    row_flag = None
    if not unit and not qty:
        row_flag = "likely_item_but_incomplete"
        confidence = 0.35

    # Trade from schedule prefix (letter-prefixed item numbers)
    trade = _trade_from_item_no(item_no) or infer_boq_trade(description or "", item_no)

    item: Dict[str, Any] = {
        "item_no": item_no,
        "description": (description or "")[:200],
        "unit": unit,
        "qty": qty,
        "rate": rate,
        "sor_code": sor_code,
        "trade": trade,
        "source_page": source_page,
        "confidence": confidence,
        "section": section,   # BOQ section header above this item (may be "")
    }
    if row_flag:
        item["row_flag"] = row_flag

    return item


# =============================================================================
# EXTRACTION (original regex path, preserved as fallback)
# =============================================================================

def extract_boq_items(
    text: str,
    source_page: int,
    pdf_path: Optional[str] = None,
    page_meta: Optional[Dict[str, Any]] = None,
    enable_table_router: bool = True,
    enable_debug: bool = False,
) -> List[dict]:
    """
    Extract BOQ line items from OCR text.

    Sprint 20F: First tries table_router for structured extraction,
    then falls back to regex patterns.  Backward-compatible output schema.

    Args:
        text: OCR/text content for the page.
        source_page: 0-indexed page number.
        pdf_path: Optional path to PDF for table_router native extraction.
        page_meta: Optional metadata dict (doc_type, discipline, has_text_layer).
        enable_table_router: If True, try table_router first.
        enable_debug: If True, attach boq_page_debug to returned items.

    Returns:
        list of dicts: [{item_no, description, unit, qty, rate, source_page, confidence}]
    """
    debug_info: Dict[str, Any] = {}
    t0 = time.perf_counter()

    # ── PHASE 1: Try table_router ────────────────────────────────────────
    routed_items = []
    if enable_table_router:
        try:
            from ..table_router import extract_table_rows_from_page
            meta = dict(page_meta or {})
            meta.setdefault("page_number", source_page)
            meta.setdefault("doc_type", "boq")

            table_result = extract_table_rows_from_page(
                page_input=pdf_path,
                page_meta=meta,
                ocr_text=text,
                config={"enable_debug": enable_debug},
            )

            if table_result.rows:
                routed_items = _parse_routed_rows(
                    table_result.rows,
                    table_result.headers,
                    source_page,
                )

            if enable_debug:
                debug_info["table_router"] = table_result.to_dict()
                debug_info["routed_item_count"] = len(routed_items)

        except Exception as e:
            if enable_debug:
                debug_info["table_router_error"] = str(e)

    # ── PHASE 2: Regex fallback (always runs, results merged) ────────────
    regex_items = _extract_boq_items_regex(text, source_page)

    if enable_debug:
        debug_info["regex_item_count"] = len(regex_items)

    # ── PHASE 3: Merge results ───────────────────────────────────────────
    # Prefer routed items if they produced more results; otherwise use regex.
    # Deduplicate by item_no + description prefix.
    if routed_items and len(routed_items) >= len(regex_items):
        items = routed_items
        method = "table_router"
    else:
        items = regex_items
        method = "regex"

    # Merge any unique items from the other source
    other = regex_items if method == "table_router" else routed_items
    seen = {_dedup_key(it['item_no'], it.get('description')) for it in items}
    for it in other:
        key = _dedup_key(it['item_no'], it.get('description'))
        if key not in seen:
            seen.add(key)
            items.append(it)

    # ── PHASE 4: Text-based fallback for scanned/poorly-structured pages ──
    # When both structured extractors together yield fewer than 5 items,
    # the page is likely a scanned image where OCR produced unstructured text.
    # Run the text-pattern fallback and merge any new items found.
    if len(items) < 5 and text and text.strip():
        try:
            from .extract_boq_text import extract_boq_from_text
            text_items = extract_boq_from_text(text, source_page=source_page)
            if text_items:
                seen_text = {_dedup_key(it['item_no'], it.get('description')) for it in items}
                added = 0
                for it in text_items:
                    key = _dedup_key(it['item_no'], it.get('description'))
                    if key not in seen_text:
                        seen_text.add(key)
                        items.append(it)
                        added += 1
                if enable_debug:
                    debug_info["text_fallback_items_added"] = added
                    debug_info["text_fallback_triggered"] = True
        except Exception as e:
            if enable_debug:
                debug_info["text_fallback_error"] = str(e)

    if enable_debug:
        debug_info["method_used"] = method
        debug_info["final_item_count"] = len(items)
        debug_info["parse_time_s"] = round(time.perf_counter() - t0, 4)
        # Attach debug to each item (only in debug mode)
        for it in items:
            it["_boq_page_debug"] = debug_info

    return items


def _extract_boq_items_regex(
    text: str,
    source_page: int,
) -> List[dict]:
    """Original regex-based BOQ extraction (preserved as fallback).

    Returns list of dicts:
        [{item_no, description, unit, qty, rate, source_page, confidence}]
    """
    items = []
    lines = text.split('\n')
    seen_items = set()
    current_section: str = ""   # most-recent section header seen on this page

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 10:
            continue

        # Capture BOQ section headers; they carry no numeric data so skip them
        # as data rows but remember the text for child items below.
        if _is_boq_section_header(line_stripped):
            current_section = line_stripped[:80]
            continue

        # Skip column header / total lines
        row_class = _classify_row(line_stripped)
        if row_class:
            continue

        # ── Pattern A: 7-column tabular (item | DSR ref | desc | unit | qty | rate | amt)
        tab7_match = TABULAR_7COL_PATTERN.match(line_stripped)
        if tab7_match:
            item_no = tab7_match.group(1).strip()
            raw_sor = tab7_match.group(2).strip()
            sor_code = raw_sor if raw_sor not in ("-", "–", "—") else None
            description = tab7_match.group(3).strip()
            raw_unit = tab7_match.group(4).strip()
            unit = _normalize_unit(raw_unit)
            qty = _safe_float(tab7_match.group(5))
            rate = _safe_float(tab7_match.group(6)) if tab7_match.group(6) else None

            # Alpha parent detection: A101 without qty = section header
            if _ALPHA_PARENT_RE.match(item_no) and not unit and qty is None:
                current_section = f"{item_no} — {description[:70]}"
                continue

            if description and len(description) >= 5:
                key = _dedup_key(item_no, description)
                if key not in seen_items:
                    seen_items.add(key)
                    items.append({
                        "item_no": item_no,
                        "description": description[:200],
                        "unit": unit,
                        "qty": qty,
                        "rate": rate,
                        "sor_code": sor_code,
                        "trade": infer_boq_trade(description, item_no),
                        "source_page": source_page,
                        "confidence": 0.85,
                        "section": current_section,
                    })
            continue

        # ── Pattern B: full inline pattern (item desc unit qty rate on one line)
        match = BOQ_LINE_PATTERN.match(line_stripped)
        if match:
            item_no = match.group(1)
            description = match.group(2).strip()
            unit = _normalize_unit(match.group(3))
            qty_str = match.group(4)
            rate_str = match.group(5)

            qty = _safe_float(qty_str)
            rate = _safe_float(rate_str) if rate_str else None

            # Extract DSR ref if present in description
            dsr_m = _DSR_EXTRACT_RE.search(description)
            sor_code: Optional[str] = dsr_m.group(1) if dsr_m else None
            if sor_code:
                description = description.replace(sor_code, "").strip(" ,;—-")

            key = _dedup_key(item_no, description)
            if key not in seen_items:
                seen_items.add(key)
                items.append({
                    "item_no": item_no,
                    "description": description[:200],
                    "unit": unit,
                    "qty": qty,
                    "rate": rate,
                    "sor_code": sor_code,
                    "trade": infer_boq_trade(description, item_no),
                    "source_page": source_page,
                    "confidence": 0.8,
                    "section": current_section,
                })
            continue

        # ── Pattern C: 5-column tabular (columns separated by 2+ spaces)
        tab_match = TABULAR_LINE_PATTERN.match(line_stripped)
        if tab_match:
            item_no = tab_match.group(1)
            description = tab_match.group(2).strip()
            raw_unit = tab_match.group(3).strip()
            unit = _normalize_unit(raw_unit)
            qty = _safe_float(tab_match.group(4))
            rate = _safe_float(tab_match.group(5)) if tab_match.group(5) else None

            # Extract DSR ref from description if present
            dsr_m = _DSR_EXTRACT_RE.search(description)
            sor_code = dsr_m.group(1) if dsr_m else None
            if sor_code:
                description = description.replace(sor_code, "").strip(" ,;—-")

            # Validate: unit should be recognized or short
            if len(raw_unit) <= 10 and description and len(description) >= 5:
                key = _dedup_key(item_no, description)
                if key not in seen_items:
                    seen_items.add(key)
                    items.append({
                        "item_no": item_no,
                        "description": description[:200],
                        "unit": unit,
                        "qty": qty,
                        "rate": rate,
                        "sor_code": sor_code,
                        "trade": infer_boq_trade(description, item_no),
                        "source_page": source_page,
                        "confidence": 0.7,
                        "section": current_section,
                    })
                continue

        # ── Pattern D: extended item number (Sprint 20F — handles A-1, A101, A1011)
        ext_match = EXTENDED_ITEM_NO.match(line_stripped)
        if ext_match:
            item_no = ext_match.group(1).strip().rstrip('.')
            rest = line_stripped[ext_match.end():].strip()

            # Alpha parent item detection: A101 (3-digit) with no numeric data → section header
            if _ALPHA_PARENT_RE.match(item_no):
                # Check if there's any numeric content suggesting this is a data row
                has_nums = bool(re.search(r'\d[\d,]*\.\d+', rest))  # decimal = qty/rate
                if not has_nums:
                    current_section = f"{item_no} — {rest[:70]}" if rest else item_no
                    continue

            if rest and len(rest) >= 5:
                unit = None
                qty = None
                rate = None
                sor_code = None

                # Extract DSR/SOR reference first and remove it from rest
                dsr_m = _DSR_EXTRACT_RE.search(rest)
                if dsr_m:
                    sor_code = dsr_m.group(1)
                    rest = rest[:dsr_m.start()].strip() + " " + rest[dsr_m.end():].strip()
                    rest = rest.strip()

                qty_match = QTY_PATTERN.search(rest)
                if qty_match:
                    qty = _safe_float(qty_match.group(1))

                rate_match = RATE_PATTERN.search(rest)
                if rate_match:
                    rate_str = rate_match.group(1).replace(',', '')
                    rate = _safe_float(rate_str)

                for u in KNOWN_UNITS:
                    if re.search(r'\b' + re.escape(u) + r'\b', rest, re.IGNORECASE):
                        unit = u
                        break

                description = rest
                if qty_match:
                    description = rest[:qty_match.start()].strip()
                if not description or len(description) < 5:
                    description = rest[:100]

                key = _dedup_key(item_no, description)
                if key not in seen_items and len(description) >= 5:
                    seen_items.add(key)
                    items.append({
                        "item_no": item_no,
                        "description": description[:200],
                        "unit": unit,
                        "qty": qty,
                        "rate": rate,
                        "sor_code": sor_code,
                        "trade": infer_boq_trade(description, item_no),
                        "source_page": source_page,
                        "confidence": 0.5 if unit else 0.3,
                        "section": current_section,
                    })
                continue

        # ── Pattern E: simple pattern (item_no + description only)
        match = SIMPLE_ITEM_PATTERN.match(line_stripped)
        if match:
            item_no = match.group(1)
            rest = match.group(2).strip()

            # Try to find unit and qty in the rest
            unit = None
            qty = None
            rate = None
            sor_code = None

            # Extract DSR/SOR reference
            dsr_m = _DSR_EXTRACT_RE.search(rest)
            if dsr_m:
                sor_code = dsr_m.group(1)
                rest = rest[:dsr_m.start()].strip() + " " + rest[dsr_m.end():].strip()
                rest = rest.strip()

            qty_match = QTY_PATTERN.search(rest)
            if qty_match:
                qty = _safe_float(qty_match.group(1))

            rate_match = RATE_PATTERN.search(rest)
            if rate_match:
                rate_str = rate_match.group(1).replace(',', '')
                rate = _safe_float(rate_str)

            # Try to detect unit
            for u in KNOWN_UNITS:
                if re.search(r'\b' + re.escape(u) + r'\b', rest, re.IGNORECASE):
                    unit = u
                    break

            # Extract description (everything before the unit/qty portion)
            description = rest
            if qty_match:
                description = rest[:qty_match.start()].strip()
            if not description or len(description) < 5:
                description = rest[:100]

            key = _dedup_key(item_no, description)
            if key not in seen_items and len(description) >= 5:
                seen_items.add(key)
                items.append({
                    "item_no": item_no,
                    "description": description[:200],
                    "unit": unit,
                    "qty": qty,
                    "rate": rate,
                    "sor_code": sor_code,
                    "trade": infer_boq_trade(description, item_no),
                    "source_page": source_page,
                    "confidence": 0.5 if unit else 0.3,
                    "section": current_section,
                })

    return items


def _safe_float(s: Optional[str]) -> Optional[float]:
    """Safely convert string to float, handling Indian number format.

    Indian format: 1,25,000 → 125000 (commas after first 3 digits are every 2 digits).
    Standard format: 125,000 → 125000.
    Both are handled by simply stripping all commas.
    Also handles Rs./INR/₹ prefixes and lakh/crore suffixes.
    """
    if not s:
        return None
    try:
        cleaned = s.strip()
        # Strip currency prefix
        cleaned = re.sub(r'^(?:Rs\.?\s*|INR\s*|₹\s*)', '', cleaned)
        # Strip commas
        cleaned = cleaned.replace(',', '')
        val = float(cleaned)
        return val
    except (ValueError, TypeError):
        return None


# =============================================================================
# BOQ TRADE INFERENCE + FLAGGING
# =============================================================================

BOQ_TRADE_KEYWORDS = {
    "civil": ["earthwork", "excavation", "backfill", "pcc", "levelling",
              "compaction", "grading", "sand filling", "soil", "dewatering"],
    "structural": ["rcc", "reinforcement", "formwork", "steel", "concrete",
                   "footing", "column", "beam", "slab", "staircase",
                   "shuttering", "bar bending", "rebar"],
    "architectural": ["brick", "block", "masonry", "plaster", "door",
                      "window", "railing", "false ceiling", "partition",
                      "aluminium", "glazing", "grille"],
    "finishes": ["paint", "tile", "polish", "marble", "granite", "laminate",
                 "dado", "skirting", "flooring", "wallpaper", "putty",
                 "primer", "texture", "ceramic", "vitrified", "epoxy coat"],
    "hvac": ["hvac", "duct", "air conditioning", "chiller", "ahu", "vrf",
             "split ac", "fan coil", "ventilation"],
    "fire": ["fire", "sprinkler", "extinguisher", "hydrant", "alarm",
             "fire fighting", "hose reel", "smoke detector"],
    "elv": ["cctv", "intercom", "access control", "bms", "ict",
            "networking", "cabling", "pa system", "ups"],
    "electrical": ["wiring", "cable", "panel", "lighting", "switch",
                   "socket", "earthing", "db", "conduit", "mcb",
                   "transformer", "generator", "dg set"],
    "plumbing": ["pipe", "plumbing", "drainage", "sanitary", "water supply",
                 "cistern", "pump", "manhole", "sewage", "cpvc", "upvc",
                 "gi pipe", "valve", "tap", "flush"],
}


def infer_boq_trade(description: str, item_no: Optional[str] = None) -> str:
    """Classify a BOQ line item into a trade.

    Priority order:
    1. Schedule-letter prefix on item_no (A→civil, R→electrical, etc.)
    2. DSR/SOR reference in description (contains trade name)
    3. Keyword matching on description text

    Returns the best-matching trade name, or 'general' if no match.
    """
    if not description and not item_no:
        return "general"

    # 1. Schedule prefix (highest confidence)
    prefix_trade = _trade_from_item_no(item_no or "")
    if prefix_trade:
        return prefix_trade

    # 2. DSR/SOR reference trade keyword
    dsr_m = _DSR_EXTRACT_RE.search(description or "")
    if dsr_m:
        dsr_text = dsr_m.group(1).upper()
        for trade_kw, trade_name in [
            ("CIVIL", "civil"), ("STRUCTURAL", "structural"),
            ("ELECTRICAL", "electrical"), ("PLUMBING", "plumbing"),
            ("FINISHES", "finishes"), ("ARCHITECTURAL", "architectural"),
            ("MEP", "mep"), ("HVAC", "hvac"), ("FIRE", "fire"),
        ]:
            if trade_kw in dsr_text:
                return trade_name

    # 3. Keyword matching
    if not description:
        return "general"
    desc_lower = description.lower()
    best_trade = "general"
    best_hits = 0
    for trade, keywords in BOQ_TRADE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in desc_lower)
        if hits > best_hits:
            best_hits = hits
            best_trade = trade
    return best_trade


def flag_boq_item(item: dict) -> list:
    """Flag potential issues with a BOQ line item.

    Returns list of flag strings:
        zero_rate, provisional_sum, qty_missing, unit_missing
    """
    flags = []
    # Zero rate
    rate = item.get("rate")
    if rate is not None and (isinstance(rate, (int, float)) and rate == 0):
        flags.append("zero_rate")
    # Provisional sum
    desc = (item.get("description") or "").lower()
    if "provisional" in desc or "lump sum" in desc or "allowance" in desc:
        flags.append("provisional_sum")
    # Missing qty
    if item.get("qty") is None:
        flags.append("qty_missing")
    # Missing unit
    if not item.get("unit"):
        flags.append("unit_missing")
    return flags
