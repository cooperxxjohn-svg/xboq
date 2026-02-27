"""
Schedule Table Extractor — Parse door/window/finish schedules from OCR text.

Handles doc_type: schedule

Sprint 20F: Enhanced with table_router integration, merged header support,
and richer mark/size/qty parsing.  Falls back to current parser if router
returns no usable rows.
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULE TYPE DETECTION
# =============================================================================

SCHEDULE_TYPE_PATTERNS = {
    "door": [
        re.compile(r'\bdoor\s+schedule\b', re.IGNORECASE),
        re.compile(r'\bschedule\s+of\s+doors\b', re.IGNORECASE),
    ],
    "window": [
        re.compile(r'\bwindow\s+schedule\b', re.IGNORECASE),
        re.compile(r'\bschedule\s+of\s+windows\b', re.IGNORECASE),
    ],
    "finish": [
        re.compile(r'\bfinish\s+schedule\b', re.IGNORECASE),
        re.compile(r'\bschedule\s+of\s+finishes\b', re.IGNORECASE),
        re.compile(r'\broom\s+finish\b', re.IGNORECASE),
    ],
    "fixture": [
        re.compile(r'\bfixture\s+schedule\b', re.IGNORECASE),
        re.compile(r'\bhardware\s+schedule\b', re.IGNORECASE),
    ],
}

# Header column keywords (case-insensitive)
HEADER_KEYWORDS = [
    "mark", "no", "type", "size", "width", "height", "qty", "quantity",
    "material", "finish", "remarks", "description", "location", "hardware",
    "glazing", "frame", "leaf", "panel", "room", "floor", "wall", "ceiling",
    "dado", "skirting",
]

# Tag patterns for marks (Sprint 20F: expanded)
MARK_PATTERNS = [
    re.compile(r'\b(D-?\d{1,3}[A-Z]?)\b'),      # D1, D-01, D1A
    re.compile(r'\b(W-?\d{1,3}[A-Z]?)\b'),      # W1, W-01
    re.compile(r'\b(DR-?\d{1,3})\b'),             # DR01
    re.compile(r'\b(WN-?\d{1,3})\b'),             # WN01
    re.compile(r'\b(F-?\d{1,3})\b'),              # F1 (fixture/finish)
    re.compile(r'\b(FN-?\d{1,3})\b'),             # FN1 (finish)
]

# Size patterns (Sprint 20F: expanded for mm, ft-in, mixed)
SIZE_PATTERNS = [
    re.compile(r'(\d{3,4})\s*[xX×]\s*(\d{3,4})'),           # 900x2100, 900 x 2100
    re.compile(r"(\d{1,2})['\u2032]\s*[-]\s*(\d{1,2})[\"'\u2033]"),  # 3'-0"
    re.compile(r'(\d{3,4})\s*mm\s*[xX×]\s*(\d{3,4})\s*mm'),  # 900mm x 2100mm
    re.compile(r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)'),  # Generic WxH
]


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def _detect_schedule_type(text: str) -> str:
    """Detect schedule type from full page text."""
    for stype, patterns in SCHEDULE_TYPE_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                return stype
    return "unknown"


def _is_header_row(line: str) -> bool:
    """Check if a line looks like a table header."""
    line_lower = line.lower()
    matches = sum(1 for kw in HEADER_KEYWORDS if kw in line_lower)
    return matches >= 3


def _extract_mark(text: str) -> Optional[str]:
    """Extract a door/window/fixture mark from text."""
    for pat in MARK_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(1)
    return None


def _extract_size(text: str) -> Optional[str]:
    """Extract size (e.g., 900x2100) from text."""
    for pat in SIZE_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(0)
    return None


def _extract_quantity(text: str) -> Optional[int]:
    """Extract quantity from text (look for standalone numbers)."""
    # Look for "Qty: 4" or just a small number in context
    qty_match = re.search(r'\bqty[:\s]*(\d{1,3})\b', text, re.IGNORECASE)
    if qty_match:
        return int(qty_match.group(1))
    # Look for standalone small numbers (1-99) that could be quantities
    nums = re.findall(r'\b(\d{1,2})\b', text)
    # Filter out numbers that are likely part of sizes
    for n in nums:
        val = int(n)
        if 1 <= val <= 50:
            return val
    return None


def _split_table_row(line: str) -> List[str]:
    """Split a table row by 2+ spaces or tabs."""
    parts = re.split(r'\s{2,}|\t', line.strip())
    return [p.strip() for p in parts if p.strip()]


# =============================================================================
# TABLE ROUTER INTEGRATION (Sprint 20F)
# =============================================================================

def _parse_routed_schedule_rows(
    rows: List[Any],
    headers: Optional[List[str]],
    schedule_type: str,
    source_page: int,
    sheet_id: Optional[str],
) -> List[dict]:
    """Parse structured table rows from table_router into schedule items."""
    items = []

    # Detect header columns for marks, size, qty, etc.
    col_map = _detect_schedule_col_map(headers)

    for row in rows:
        # Normalize row to list of strings
        if isinstance(row, dict):
            cells = list(row.values())
        elif isinstance(row, (list, tuple)):
            cells = [str(c) for c in row]
        else:
            continue

        if not cells or not any(c.strip() for c in cells if c):
            continue

        full_text = " ".join(c.strip() for c in cells if c)

        # Skip header-like rows
        if _is_header_row(full_text):
            continue

        # Try to extract mark from mapped column or full text
        mark = None
        if "mark" in col_map and col_map["mark"] < len(cells):
            mark = _extract_mark(cells[col_map["mark"]])
        if not mark:
            mark = _extract_mark(full_text)

        if not mark:
            continue

        # Extract size and qty
        size = None
        qty = None

        if "size" in col_map and col_map["size"] < len(cells):
            size = _extract_size(cells[col_map["size"]])
        if not size:
            size = _extract_size(full_text)

        if "qty" in col_map and col_map["qty"] < len(cells):
            qty = _extract_quantity(cells[col_map["qty"]])
        if qty is None:
            qty = _extract_quantity(full_text)

        # Build fields dict from cells
        fields: Dict[str, str] = {}
        if headers:
            for j, cell in enumerate(cells):
                if j < len(headers):
                    fields[headers[j].lower().strip()] = (cell or "").strip()
                else:
                    fields[f"col_{j}"] = (cell or "").strip()
        else:
            for j, cell in enumerate(cells):
                fields[f"col_{j}"] = (cell or "").strip()

        items.append({
            "mark": mark,
            "fields": fields,
            "schedule_type": schedule_type,
            "source_page": source_page,
            "sheet_id": sheet_id,
            "has_size": size is not None,
            "has_qty": qty is not None,
            "size": size,
            "qty": qty,
        })

    return items


def _detect_schedule_col_map(headers: Optional[List[str]]) -> Dict[str, int]:
    """Map header labels to column indices for schedule fields."""
    col_map: Dict[str, int] = {}
    if not headers:
        return col_map

    for i, h in enumerate(headers):
        hl = (h or "").lower().strip()
        if re.search(r'\bmark\b|\bno\b|\btag\b', hl):
            col_map.setdefault("mark", i)
        elif re.search(r'\bsize\b|\bwidth\b|\bdim', hl):
            col_map.setdefault("size", i)
        elif re.search(r'\bqty\b|\bquantity\b', hl):
            col_map.setdefault("qty", i)
        elif re.search(r'\btype\b', hl):
            col_map.setdefault("type", i)
        elif re.search(r'\bmaterial\b|\bfinish\b', hl):
            col_map.setdefault("material", i)

    return col_map


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_schedule_rows(
    text: str,
    source_page: int,
    sheet_id: Optional[str],
    pdf_path: Optional[str] = None,
    page_meta: Optional[Dict[str, Any]] = None,
    enable_table_router: bool = True,
    enable_debug: bool = False,
) -> List[dict]:
    """
    Extract schedule rows from a schedule page.

    Sprint 20F: Tries table_router first, then falls back to current parser.

    Returns list of dicts:
        [{mark, fields, schedule_type, source_page, sheet_id, has_size, has_qty}]
    """
    schedule_type = _detect_schedule_type(text)
    debug_info: Dict[str, Any] = {}
    t0 = time.perf_counter()

    # ── PHASE 1: Try table_router ────────────────────────────────────────
    routed_rows: List[dict] = []
    if enable_table_router:
        try:
            from ..table_router import extract_table_rows_from_page
            meta = dict(page_meta or {})
            meta.setdefault("page_number", source_page)
            meta.setdefault("doc_type", "schedule")

            table_result = extract_table_rows_from_page(
                page_input=pdf_path,
                page_meta=meta,
                ocr_text=text,
                config={"enable_debug": enable_debug},
            )

            if table_result.rows:
                routed_rows = _parse_routed_schedule_rows(
                    table_result.rows,
                    table_result.headers,
                    schedule_type,
                    source_page,
                    sheet_id,
                )

            if enable_debug:
                debug_info["table_router"] = table_result.to_dict()
                debug_info["routed_row_count"] = len(routed_rows)

        except Exception as e:
            if enable_debug:
                debug_info["table_router_error"] = str(e)

    # ── PHASE 2: Regex fallback ──────────────────────────────────────────
    regex_rows = _extract_schedule_rows_regex(text, source_page, sheet_id, schedule_type)

    if enable_debug:
        debug_info["regex_row_count"] = len(regex_rows)

    # ── PHASE 3: Merge ───────────────────────────────────────────────────
    # Use routed rows if they found more items; otherwise regex
    if routed_rows and len(routed_rows) >= len(regex_rows):
        rows = routed_rows
        method = "table_router"
    else:
        rows = regex_rows
        method = "regex"

    # Merge unique marks from the other source
    other = regex_rows if method == "table_router" else routed_rows
    seen_marks = {r["mark"] for r in rows}
    for r in other:
        if r["mark"] not in seen_marks:
            seen_marks.add(r["mark"])
            rows.append(r)

    if enable_debug:
        debug_info["method_used"] = method
        debug_info["final_row_count"] = len(rows)
        debug_info["schedule_type"] = schedule_type
        debug_info["schedule_confidence"] = min(0.9, 0.3 + 0.05 * len(rows))
        debug_info["parse_time_s"] = round(time.perf_counter() - t0, 4)
        for r in rows:
            r["_schedule_page_debug"] = debug_info

    return rows


def _extract_schedule_rows_regex(
    text: str,
    source_page: int,
    sheet_id: Optional[str],
    schedule_type: str,
) -> List[dict]:
    """Original regex-based schedule extraction (preserved as fallback)."""
    rows = []
    lines = text.split('\n')

    in_table = False
    header_parts: List[str] = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if _is_header_row(line_stripped):
            in_table = True
            header_parts = _split_table_row(line_stripped)
            continue

        if in_table:
            # Check if this looks like a data row
            mark = _extract_mark(line_stripped)

            if mark:
                parts = _split_table_row(line_stripped)
                size = _extract_size(line_stripped)
                qty = _extract_quantity(line_stripped)

                # Build fields dict by mapping to header columns
                fields: Dict[str, str] = {}
                for j, part in enumerate(parts):
                    if j < len(header_parts):
                        fields[header_parts[j].lower()] = part
                    else:
                        fields[f"col_{j}"] = part

                rows.append({
                    "mark": mark,
                    "fields": fields,
                    "schedule_type": schedule_type,
                    "source_page": source_page,
                    "sheet_id": sheet_id,
                    "has_size": size is not None,
                    "has_qty": qty is not None,
                    "size": size,
                    "qty": qty,
                })

            elif not re.match(r'^\s*\d', line_stripped) and len(line_stripped) < 10:
                # Short non-data line might end the table
                in_table = False

    # Fallback: even without a header row, look for mark patterns line by line
    if not rows:
        for i, line in enumerate(lines):
            mark = _extract_mark(line.strip())
            if mark:
                size = _extract_size(line)
                qty = _extract_quantity(line)
                rows.append({
                    "mark": mark,
                    "fields": {"raw": line.strip()},
                    "schedule_type": schedule_type,
                    "source_page": source_page,
                    "sheet_id": sheet_id,
                    "has_size": size is not None,
                    "has_qty": qty is not None,
                    "size": size,
                    "qty": qty,
                })

    return rows
