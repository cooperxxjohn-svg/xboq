"""
Scope Reconciliation — cross-check requirements vs schedules vs BOQ.

Detects missing links, quantity mismatches, and ambiguities between
extraction outputs. All functions are pure (no Streamlit, no I/O).

Also provides link_schedules_to_boq() which:
  - Links schedule marks (D-01, W-05) to BOQ item descriptions.
  - Creates stub BOQ items for unmatched schedule marks.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


# =============================================================================
# PATTERNS FOR SCOPE MATCHING
# =============================================================================

_DOOR_RE = re.compile(r'\bdoors?\b', re.IGNORECASE)
_WINDOW_RE = re.compile(r'\bwindows?\b', re.IGNORECASE)

# Common material grade keywords
_MATERIAL_GRADE_RE = re.compile(
    r'\b(teak|sal|deodar|maranti|hardwood|softwood|plywood|mdf|wpc|upvc|aluminium|aluminum|'
    r'steel|stainless|brass|bronze|copper|granite|marble|ceramic|vitrified|porcelain|'
    r'cement|opc|ppc|m\d{2,3}|fe\d{3,4}|tmmt|tmt)\b',
    re.IGNORECASE,
)

# Door schedule types
_DOOR_SCHEDULE_TYPES = {"door", "door_schedule", "doors"}
# Window schedule types
_WINDOW_SCHEDULE_TYPES = {"window", "window_schedule", "windows"}


def _get_schedule_types(schedules: List[dict]) -> Dict[str, List[dict]]:
    """Group schedule rows by normalized schedule_type."""
    by_type: Dict[str, List[dict]] = {}
    for s in schedules:
        stype = (s.get("schedule_type", "") or "").lower().strip()
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(s)
    return by_type


def _has_door_schedules(schedule_types: Dict[str, List[dict]]) -> bool:
    """Check if any schedule type is a door schedule."""
    for stype in schedule_types:
        if stype in _DOOR_SCHEDULE_TYPES or "door" in stype:
            return True
    return False


def _has_window_schedules(schedule_types: Dict[str, List[dict]]) -> bool:
    """Check if any schedule type is a window schedule."""
    for stype in schedule_types:
        if stype in _WINDOW_SCHEDULE_TYPES or "window" in stype:
            return True
    return False


def _get_door_schedule_count(schedule_types: Dict[str, List[dict]]) -> int:
    """Count unique door marks in schedules."""
    marks = set()
    for stype, rows in schedule_types.items():
        if stype in _DOOR_SCHEDULE_TYPES or "door" in stype:
            for r in rows:
                m = r.get("mark")
                if m:
                    marks.add(m)
    return len(marks)


def _get_window_schedule_count(schedule_types: Dict[str, List[dict]]) -> int:
    """Count unique window marks in schedules."""
    marks = set()
    for stype, rows in schedule_types.items():
        if stype in _WINDOW_SCHEDULE_TYPES or "window" in stype:
            for r in rows:
                m = r.get("mark")
                if m:
                    marks.add(m)
    return len(marks)


def _extract_pages(items: List[dict]) -> List[int]:
    """Extract unique source_page values from items."""
    pages = set()
    for i in items:
        pg = i.get("source_page")
        if pg is not None:
            pages.add(pg)
    return sorted(pages)


def _get_boq_door_qty(boq_items: List[dict]) -> int:
    """Sum qty from BOQ items mentioning 'door'."""
    total = 0
    for item in boq_items:
        desc = (item.get("description", "") or "").lower()
        if _DOOR_RE.search(desc):
            qty = item.get("qty")
            if isinstance(qty, (int, float)) and qty > 0:
                total += int(qty)
    return total


def _get_boq_window_qty(boq_items: List[dict]) -> int:
    """Sum qty from BOQ items mentioning 'window'."""
    total = 0
    for item in boq_items:
        desc = (item.get("description", "") or "").lower()
        if _WINDOW_RE.search(desc):
            qty = item.get("qty")
            if isinstance(qty, (int, float)) and qty > 0:
                total += int(qty)
    return total


# =============================================================================
# CROSS-CHECK RULES
# =============================================================================

def _check_reqs_vs_schedules(
    requirements: List[dict],
    schedule_types: Dict[str, List[dict]],
) -> List[dict]:
    """Check requirements against schedule data."""
    findings: List[dict] = []

    req_texts = [r.get("text", "").lower() for r in requirements]
    combined_req_text = " ".join(req_texts)
    req_pages = _extract_pages(requirements)

    # Doors: requirements mention doors but no door schedule
    if _DOOR_RE.search(combined_req_text) and not _has_door_schedules(schedule_types):
        findings.append({
            "type": "missing",
            "category": "req_vs_schedule",
            "description": "Requirements mention doors but no door schedule was found",
            "impact": "high",
            "suggested_action": "Request door schedule from design team or verify inclusion in drawings",
            "evidence": {"pages": req_pages, "items": ["door requirements"]},
            "confidence": 0.85,
        })

    # Windows: requirements mention windows but no window schedule
    if _WINDOW_RE.search(combined_req_text) and not _has_window_schedules(schedule_types):
        findings.append({
            "type": "missing",
            "category": "req_vs_schedule",
            "description": "Requirements mention windows but no window schedule was found",
            "impact": "high",
            "suggested_action": "Request window schedule from design team or verify inclusion in drawings",
            "evidence": {"pages": req_pages, "items": ["window requirements"]},
            "confidence": 0.85,
        })

    # Material grades in requirements not found in any schedule field
    all_schedule_text = ""
    for rows in schedule_types.values():
        for r in rows:
            fields = r.get("fields", {})
            if isinstance(fields, dict):
                all_schedule_text += " ".join(str(v) for v in fields.values()) + " "
            all_schedule_text += (r.get("description", "") or "") + " "
    all_schedule_text = all_schedule_text.lower()

    req_grades = set(_MATERIAL_GRADE_RE.findall(combined_req_text))
    for grade in req_grades:
        if grade.lower() not in all_schedule_text:
            findings.append({
                "type": "ambiguity",
                "category": "req_vs_schedule",
                "description": f"Material grade '{grade}' in requirements but not found in schedules",
                "impact": "medium",
                "suggested_action": f"Verify that '{grade}' specification is reflected in relevant schedules",
                "evidence": {"pages": req_pages, "items": [grade]},
                "confidence": 0.7,
            })

    return findings


def _check_boq_vs_schedules(
    boq_items: List[dict],
    schedule_types: Dict[str, List[dict]],
) -> List[dict]:
    """Check BOQ items against schedule data."""
    findings: List[dict] = []

    # Door qty mismatch: BOQ door qty vs schedule mark count
    boq_door_qty = _get_boq_door_qty(boq_items)
    sched_door_count = _get_door_schedule_count(schedule_types)

    if boq_door_qty > 0 and sched_door_count > 0 and boq_door_qty != sched_door_count:
        findings.append({
            "type": "conflict",
            "category": "boq_vs_schedule",
            "description": (
                f"BOQ door quantity ({boq_door_qty}) does not match "
                f"door schedule marks ({sched_door_count})"
            ),
            "impact": "high",
            "suggested_action": "Reconcile door count between BOQ and door schedule",
            "evidence": {
                "pages": _extract_pages(boq_items),
                "items": [f"BOQ doors: {boq_door_qty}", f"Schedule marks: {sched_door_count}"],
            },
            "confidence": 0.8,
        })

    # Window qty mismatch
    boq_window_qty = _get_boq_window_qty(boq_items)
    sched_window_count = _get_window_schedule_count(schedule_types)

    if boq_window_qty > 0 and sched_window_count > 0 and boq_window_qty != sched_window_count:
        findings.append({
            "type": "conflict",
            "category": "boq_vs_schedule",
            "description": (
                f"BOQ window quantity ({boq_window_qty}) does not match "
                f"window schedule marks ({sched_window_count})"
            ),
            "impact": "high",
            "suggested_action": "Reconcile window count between BOQ and window schedule",
            "evidence": {
                "pages": _extract_pages(boq_items),
                "items": [f"BOQ windows: {boq_window_qty}", f"Schedule marks: {sched_window_count}"],
            },
            "confidence": 0.8,
        })

    # Schedule marks without corresponding BOQ line
    all_boq_desc = " ".join((i.get("description", "") or "").lower() for i in boq_items)
    for stype, rows in schedule_types.items():
        for r in rows:
            mark = r.get("mark", "")
            if mark and mark.lower() not in all_boq_desc:
                # Check if schedule type is mentioned in BOQ at all
                if stype.lower() not in all_boq_desc:
                    findings.append({
                        "type": "missing",
                        "category": "boq_vs_schedule",
                        "description": f"Schedule mark '{mark}' ({stype}) has no corresponding BOQ line item",
                        "impact": "medium",
                        "suggested_action": f"Verify BOQ includes pricing for schedule item {mark}",
                        "evidence": {
                            "pages": [r.get("source_page")] if r.get("source_page") is not None else [],
                            "items": [mark],
                        },
                        "confidence": 0.65,
                    })

    return findings


def _check_reqs_vs_boq(
    requirements: List[dict],
    boq_items: List[dict],
) -> List[dict]:
    """Check requirements against BOQ data."""
    findings: List[dict] = []

    req_texts = [r.get("text", "").lower() for r in requirements]
    combined_req_text = " ".join(req_texts)
    all_boq_desc = " ".join((i.get("description", "") or "").lower() for i in boq_items)

    # Material grades in requirements not found in BOQ descriptions
    req_grades = set(_MATERIAL_GRADE_RE.findall(combined_req_text))
    for grade in req_grades:
        if grade.lower() not in all_boq_desc:
            findings.append({
                "type": "ambiguity",
                "category": "req_vs_boq",
                "description": f"Material grade '{grade}' in requirements but not found in BOQ descriptions",
                "impact": "medium",
                "suggested_action": f"Verify BOQ descriptions include '{grade}' specification",
                "evidence": {
                    "pages": _extract_pages(requirements),
                    "items": [grade],
                },
                "confidence": 0.7,
            })

    # BOQ items with zero rate where requirements specify material
    for item in boq_items:
        rate = item.get("rate")
        desc = (item.get("description", "") or "").lower()
        if rate is not None and (rate == 0 or rate == "0"):
            # Check if requirements mention something related
            matches = _MATERIAL_GRADE_RE.findall(desc)
            if matches:
                findings.append({
                    "type": "ambiguity",
                    "category": "req_vs_boq",
                    "description": f"BOQ item '{item.get('item_no', '?')}' has zero rate but specifies material ({', '.join(matches)})",
                    "impact": "low",
                    "suggested_action": "Verify rate for this item — may need pricing",
                    "evidence": {
                        "pages": [item.get("source_page")] if item.get("source_page") is not None else [],
                        "items": [item.get("item_no", "?")],
                    },
                    "confidence": 0.6,
                })

    return findings


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def reconcile_scope(
    requirements: List[dict],
    schedules: List[dict],
    boq_items: List[dict],
) -> List[dict]:
    """
    Cross-check requirements vs schedules vs BOQ for scope gaps.

    Args:
        requirements: Extracted requirement dicts.
        schedules: Extracted schedule row dicts.
        boq_items: Extracted BOQ item dicts.

    Returns:
        List of finding dicts, each with:
        - type: "missing" | "conflict" | "ambiguity"
        - category: "req_vs_schedule" | "boq_vs_schedule" | "req_vs_boq"
        - description: str
        - impact: "high" | "medium" | "low"
        - suggested_action: str
        - evidence: {pages: [int], items: [str]}
        - confidence: float
    """
    if not requirements and not schedules and not boq_items:
        return []

    schedule_types = _get_schedule_types(schedules)
    findings: List[dict] = []

    if requirements:
        findings.extend(_check_reqs_vs_schedules(requirements, schedule_types))

    if boq_items or schedules:
        findings.extend(_check_boq_vs_schedules(boq_items, schedule_types))

    if requirements and boq_items:
        findings.extend(_check_reqs_vs_boq(requirements, boq_items))

    # Sprint 17: Deterministic sort by impact/type/category
    from src.analysis.determinism import stable_sort_key
    _IMP = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (
        _IMP.get(f.get("impact", "low"), 9),
        f.get("type", ""),
        f.get("category", ""),
        stable_sort_key(f, ("description",)),
    ))

    return findings


# =============================================================================
# SCHEDULE ↔ BOQ LINKER  (Sprint 21 — 100% extraction feature)
# =============================================================================

@dataclass
class ReconciliationResult:
    """Output of link_schedules_to_boq()."""
    linked_pairs: List[Dict] = field(default_factory=list)
    """Each entry: {schedule_mark, boq_item_no, boq_description, confidence}."""

    unmatched_schedule_marks: List[str] = field(default_factory=list)
    """Marks that appeared in schedules but were not found in any BOQ item."""

    stub_items: List[Dict] = field(default_factory=list)
    """Synthesised BOQ-style dicts for each unmatched schedule mark."""


# Maps schedule_type value to trade string
_SCHED_TYPE_TO_TRADE: Dict[str, str] = {
    "door":    "architectural",
    "window":  "architectural",
    "finish":  "finishes",
    "fixture": "mep",
    "hardware":"architectural",
    "room":    "architectural",
}
_DEFAULT_TRADE = "architectural"


def _normalize_mark(mark: str) -> str:
    """Uppercase and strip separators for consistent comparison."""
    return re.sub(r"[\s\-/]", "", mark.upper())


def _build_mark_pattern(marks: List[str]) -> Optional[re.Pattern]:
    """Build a compiled regex that matches any mark as a whole word."""
    if not marks:
        return None
    escaped = [re.escape(m) for m in sorted(marks, key=len, reverse=True)]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)


def link_schedules_to_boq(
    schedules: List[dict],
    boq_items: List[dict],
) -> ReconciliationResult:
    """
    Link schedule marks to BOQ item descriptions.

    Mutates boq_items in-place by adding ``linked_schedule_mark``.
    Mutates schedule items in-place by adding ``matched_boq_item_nos``.

    Args:
        schedules:  Schedule rows from ExtractionResult.schedules.
        boq_items:  BOQ items from ExtractionResult.boq_items.

    Returns:
        ReconciliationResult with linked_pairs, unmatched_schedule_marks,
        and stub_items for unmatched marks.
    """
    result = ReconciliationResult()

    # Initialise BOQ annotations regardless of schedule availability
    for boq in boq_items:
        boq.setdefault("linked_schedule_mark", None)

    if not schedules:
        return result

    # --- Build mark lookup: normalised_mark → schedule_item ---
    mark_lookup: Dict[str, dict] = {}
    norm_to_original: Dict[str, str] = {}
    for sched in schedules:
        raw_mark = sched.get("mark", "")
        if not raw_mark:
            continue
        norm = _normalize_mark(raw_mark)
        if norm:
            mark_lookup[norm] = sched
            norm_to_original[norm] = raw_mark
            sched.setdefault("matched_boq_item_nos", [])

    if not mark_lookup:
        return result

    mark_pattern = _build_mark_pattern(list(norm_to_original.values()))
    matched_norms: Set[str] = set()

    # --- Scan BOQ items for mark appearances ---
    for boq in boq_items:
        if mark_pattern is None:
            break
        desc = boq.get("description", "") or ""
        m = mark_pattern.search(desc)
        if m:
            found_mark = m.group(1)
            norm = _normalize_mark(found_mark)
            sched_item = mark_lookup.get(norm)
            if sched_item:
                boq["linked_schedule_mark"] = found_mark
                item_no = boq.get("item_no") or boq.get("sr_no") or ""
                sched_item["matched_boq_item_nos"].append(item_no)
                matched_norms.add(norm)
                result.linked_pairs.append({
                    "schedule_mark":   found_mark,
                    "boq_item_no":     item_no,
                    "boq_description": desc[:120],
                    "confidence":      0.90,
                })

    # --- Build stub items for unmatched marks ---
    for norm, sched_item in mark_lookup.items():
        if norm in matched_norms:
            continue
        raw_mark = norm_to_original[norm]
        result.unmatched_schedule_marks.append(raw_mark)

        sched_type = (sched_item.get("schedule_type") or "item").lower()
        trade = _SCHED_TYPE_TO_TRADE.get(sched_type, _DEFAULT_TRADE)
        size = (
            sched_item.get("size")
            or sched_item.get("dimensions")
            or "size TBC"
        )
        qty = sched_item.get("qty") or sched_item.get("quantity")
        try:
            qty = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            qty = None

        result.stub_items.append({
            "item_no":              None,
            "description":          f"{sched_type.title()} {raw_mark}, {size}",
            "unit":                 "nos",
            "qty":                  qty,
            "rate":                 None,
            "section":              f"FROM SCHEDULE — {sched_type.upper()}",
            "source":               "schedule_stub",
            "confidence":           0.40,
            "source_page":          sched_item.get("source_page", 0),
            "trade":                trade,
            "linked_schedule_mark": raw_mark,
        })

    return result
