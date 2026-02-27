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
    "sqft": "sqft", "sq.ft": "sqft", "sq ft": "sqft", "square feet": "sqft",
    "cu.m": "cum", "cu m": "cum", "cubic metre": "cum", "cubic meter": "cum",
    "cft": "cft", "cu.ft": "cft", "cu ft": "cft", "cubic feet": "cft",
    "rm": "rmt", "r.m": "rmt", "r.m.": "rmt", "running metre": "rmt",
    "running meter": "rmt",
    "no": "nos", "no.": "nos", "numbers": "nos", "each": "nos", "set": "nos",
    "pair": "nos",
    "kgs": "kg", "kilogram": "kg",
    "mt": "mt", "tonne": "mt", "metric ton": "mt",
    "l.s": "LS", "l.s.": "LS", "lump sum": "LS", "lumpsum": "LS", "ls": "LS",
    "job": "LS",
    "litre": "liter", "lit": "liter",
    "bag": "bag", "bags": "bag",
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
    r'|[A-Z]-?\d{1,3}'                          # A-1, B2
    r'|\d{1,3}\s*\([a-z]\)'                      # 2(a), 3(b)
    r'|[ivx]{1,4}'                                # roman numerals i-iv
    r')'
    r'[\s\.\)]+',
    re.IGNORECASE,
)

# Full BOQ line: item_no, description, unit, quantity, rate
BOQ_LINE_PATTERN = re.compile(
    r'^\s*(\d{1,3}(?:\.\d{1,3})?(?:\.[a-z])?)'  # item number
    r'\s+(.{10,200}?)'                             # description (greedy but bounded)
    r'\s+(sqm|cum|rmt|nos|kg|mt|LS|sqft|cft|no\.?|each|set|bag|kl|liter|litre|pair|lump\s*sum|running\s*met[re]{2})'  # unit
    r'\s+(\d+(?:\.\d+)?)'                          # quantity
    r'(?:\s+(\d+(?:\.\d+)?))?',                    # optional rate
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

# Sprint 20F: Row classification patterns
_HEADER_LIKE = re.compile(
    r'\b(item\s*no|s\.?\s*no|description|unit|qty|quantity|rate|amount)\b',
    re.IGNORECASE,
)
_TOTAL_LIKE = re.compile(
    r'\b(sub\s*total|total|grand\s*total|carried\s*forward|brought\s*forward|carried\s*over)\b',
    re.IGNORECASE,
)


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

        # Extract fields using column map or positional heuristics
        item = _extract_item_from_cells(cells, col_map, source_page)
        if item:
            key = f"{item['item_no']}:{(item.get('description') or '')[:30]}"
            if key not in seen:
                seen.add(key)
                items.append(item)

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

    return col_map


def _extract_item_from_cells(
    cells: List[str],
    col_map: Dict[str, int],
    source_page: int,
) -> Optional[dict]:
    """Extract a BOQ item from a row of table cells.

    Uses column map if available, otherwise falls back to positional heuristics.
    """
    # Helper to safely get cell value
    def _cell(idx: int) -> str:
        if 0 <= idx < len(cells):
            return (cells[idx] or "").strip()
        return ""

    item_no = None
    description = None
    unit = None
    qty = None
    rate = None
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
        confidence = 0.75
    else:
        # Positional heuristics for typical BOQ layout:
        # col0=item_no, col1=description, col2=unit, col3=qty, col4=rate
        if len(cells) >= 3:
            item_no = cells[0].strip()
            description = cells[1].strip()
            if len(cells) >= 4:
                raw_unit = cells[2].strip()
                unit = _normalize_unit(raw_unit) if raw_unit else None
                qty = _safe_float(cells[3].strip())
            if len(cells) >= 5:
                rate = _safe_float(cells[4].strip())

    # Validate item_no looks like a real item number
    if not item_no:
        return None
    if not re.match(r'^\d{1,3}(?:\.\d{1,3}){0,2}(?:\.[a-z])?$|^[A-Z]-?\d{1,3}$|^\d{1,3}\s*\([a-z]\)$', item_no, re.IGNORECASE):
        return None

    # Need at least a description
    if not description or len(description) < 5:
        return None

    # Flag row as incomplete but still useful if it has description
    row_flag = None
    if not unit and not qty:
        row_flag = "likely_item_but_incomplete"
        confidence = 0.35

    item = {
        "item_no": item_no,
        "description": description[:200],
        "unit": unit,
        "qty": qty,
        "rate": rate,
        "source_page": source_page,
        "confidence": confidence,
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
    seen = {f"{it['item_no']}:{(it.get('description') or '')[:30]}" for it in items}
    for it in other:
        key = f"{it['item_no']}:{(it.get('description') or '')[:30]}"
        if key not in seen:
            seen.add(key)
            items.append(it)

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

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 10:
            continue

        # Skip header/total lines
        row_class = _classify_row(line_stripped)
        if row_class:
            continue

        # Try full pattern first
        match = BOQ_LINE_PATTERN.match(line_stripped)
        if match:
            item_no = match.group(1)
            description = match.group(2).strip()
            unit = _normalize_unit(match.group(3))
            qty_str = match.group(4)
            rate_str = match.group(5)

            qty = _safe_float(qty_str)
            rate = _safe_float(rate_str) if rate_str else None

            key = f"{item_no}:{description[:30]}"
            if key not in seen_items:
                seen_items.add(key)
                items.append({
                    "item_no": item_no,
                    "description": description[:200],
                    "unit": unit,
                    "qty": qty,
                    "rate": rate,
                    "source_page": source_page,
                    "confidence": 0.8,
                })
            continue

        # Try extended item number pattern (Sprint 20F)
        ext_match = EXTENDED_ITEM_NO.match(line_stripped)
        if ext_match:
            item_no = ext_match.group(1).strip().rstrip('.')
            rest = line_stripped[ext_match.end():].strip()

            if rest and len(rest) >= 5:
                unit = None
                qty = None
                rate = None

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

                key = f"{item_no}:{description[:30]}"
                if key not in seen_items and len(description) >= 5:
                    seen_items.add(key)
                    items.append({
                        "item_no": item_no,
                        "description": description[:200],
                        "unit": unit,
                        "qty": qty,
                        "rate": rate,
                        "source_page": source_page,
                        "confidence": 0.5 if unit else 0.3,
                    })
                continue

        # Try simple pattern (item_no + description)
        match = SIMPLE_ITEM_PATTERN.match(line_stripped)
        if match:
            item_no = match.group(1)
            rest = match.group(2).strip()

            # Try to find unit and qty in the rest
            unit = None
            qty = None
            rate = None

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

            key = f"{item_no}:{description[:30]}"
            if key not in seen_items and len(description) >= 5:
                seen_items.add(key)
                items.append({
                    "item_no": item_no,
                    "description": description[:200],
                    "unit": unit,
                    "qty": qty,
                    "rate": rate,
                    "source_page": source_page,
                    "confidence": 0.5 if unit else 0.3,
                })

    return items


def _safe_float(s: Optional[str]) -> Optional[float]:
    """Safely convert string to float."""
    if not s:
        return None
    try:
        return float(s.replace(',', ''))
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


def infer_boq_trade(description: str) -> str:
    """Classify a BOQ line item into a trade by keyword matching.

    Returns the best-matching trade name, or 'general' if no match.
    """
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
