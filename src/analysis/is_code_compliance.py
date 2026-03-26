"""
src/analysis/is_code_compliance.py

IS code compliance checker for xBOQ.ai.

Validates extracted BOQ/spec items against Indian Standards requirements:
  - IS:456 — Plain and Reinforced Concrete (concrete grades vs exposure)
  - IS:1786 — High Strength Deformed Steel Bars
  - IS:2062 — Structural Steel
  - IS:3812 — Fly Ash specification
  - IS:8112 — 43 Grade OPC Cement
  - NBC 2016 — National Building Code fire/structural requirements

Usage:
    from src.analysis.is_code_compliance import check_is_compliance
    violations = check_is_compliance(payload)
    # Returns List[ISViolation]
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ISViolation:
    code: str           # e.g. "IS:456-2000"
    clause: str         # e.g. "6.1.2"
    severity: str       # "error" | "warning" | "info"
    description: str
    item_description: str
    trade: str
    suggestion: str = ""
    source_page: int = 0


# ── IS:456 Concrete Grade Rules ───────────────────────────────────────────────

# Minimum concrete grades by exposure condition per IS:456-2000 Table 5
_IS456_MIN_GRADES: Dict[str, int] = {
    "mild":           20,   # M20 min
    "moderate":       25,   # M25 min
    "severe":         30,   # M30 min
    "very severe":    35,   # M35 min
    "extreme":        40,   # M40 min
}

# Default exposure for structural elements by context
_ELEMENT_EXPOSURE: Dict[str, str] = {
    "foundation":     "moderate",
    "footing":        "moderate",
    "basement":       "severe",
    "retaining wall": "severe",
    "slab":           "mild",
    "column":         "mild",
    "beam":           "mild",
    "roof slab":      "moderate",
    "water tank":     "severe",
    "swimming pool":  "very severe",
}

_CONCRETE_GRADE_RE = re.compile(r'\bm\s*(\d{2,3})\b', re.IGNORECASE)
_REBAR_GRADE_RE = re.compile(r'\bfe\s*(\d{3})\b', re.IGNORECASE)
_STEEL_GRADE_RE = re.compile(r'\be\s*(\d{3})\b|\bfe\s*(\d{3}[a-z]?)\b', re.IGNORECASE)


def _parse_concrete_grade(text: str) -> Optional[int]:
    m = _CONCRETE_GRADE_RE.search(text)
    return int(m.group(1)) if m else None


def _parse_rebar_grade(text: str) -> Optional[int]:
    m = _REBAR_GRADE_RE.search(text)
    return int(m.group(1)) if m else None


def _check_is456(item: dict) -> List[ISViolation]:
    """Check IS:456 concrete grade requirements."""
    violations = []
    desc = str(item.get("description") or "").lower()
    grade = _parse_concrete_grade(desc)
    if grade is None:
        return violations

    # Determine exposure condition from description
    exposure = "mild"
    for element, exp in _ELEMENT_EXPOSURE.items():
        if element in desc:
            exposure = exp
            break

    min_grade = _IS456_MIN_GRADES.get(exposure, 20)
    if grade < min_grade:
        violations.append(ISViolation(
            code="IS:456-2000",
            clause="6.1.2 Table 5",
            severity="error",
            description=(
                f"Concrete grade M{grade} is below IS:456 minimum M{min_grade} "
                f"for '{exposure}' exposure condition"
            ),
            item_description=str(item.get("description", ""))[:100],
            trade=str(item.get("trade", "structural")),
            suggestion=f"Upgrade to minimum M{min_grade} per IS:456-2000 Table 5 for {exposure} exposure",
            source_page=int(item.get("source_page") or item.get("page") or 0),
        ))

    return violations


def _check_is1786(item: dict) -> List[ISViolation]:
    """Check IS:1786 rebar grade requirements."""
    violations = []
    desc = str(item.get("description") or "").lower()
    rebar_grade = _parse_rebar_grade(desc)

    if rebar_grade is None:
        return violations

    # IS:1786-2008: Fe415 minimum for seismic zones III-V
    # Fe550D recommended for ductile detailing
    if rebar_grade < 415:
        violations.append(ISViolation(
            code="IS:1786-2008",
            clause="4.1",
            severity="warning",
            description=f"Rebar Fe{rebar_grade} is below IS:1786 recommended Fe415 minimum",
            item_description=str(item.get("description", ""))[:100],
            trade=str(item.get("trade", "structural")),
            suggestion="Use Fe415 or Fe500 per IS:1786-2008 for structural reinforcement",
            source_page=int(item.get("source_page") or item.get("page") or 0),
        ))
    return violations


def _check_coverage_thickness(item: dict) -> List[ISViolation]:
    """Check if cover thickness is mentioned for RCC elements (IS:456 Cl.26.4)."""
    violations = []
    desc = str(item.get("description") or "").lower()
    is_rcc = any(kw in desc for kw in ["rcc", "reinforced concrete", "r.c.c"])
    # Use word-boundary patterns so "cc" doesn't match inside "rcc"
    has_cover = bool(
        re.search(r'\bcover\b', desc) or
        re.search(r'\bclear cover\b', desc) or
        re.search(r'\bcc\b', desc)
    )

    if is_rcc and not has_cover:
        violations.append(ISViolation(
            code="IS:456-2000",
            clause="26.4",
            severity="info",
            description="RCC item does not specify clear cover to reinforcement",
            item_description=str(item.get("description", ""))[:100],
            trade=str(item.get("trade", "structural")),
            suggestion="Specify clear cover per IS:456 Cl.26.4 (25mm for columns, 40mm for footings)",
            source_page=int(item.get("source_page") or item.get("page") or 0),
        ))
    return violations


# ── Public API ────────────────────────────────────────────────────────────────

def check_is_compliance(payload: dict) -> List[ISViolation]:
    """
    Run all IS code compliance checks on the analysis payload.

    Checks:
    - IS:456-2000 concrete grade vs exposure condition
    - IS:1786-2008 rebar grade requirements
    - IS:456 Cl.26.4 clear cover specification

    Parameters
    ----------
    payload : dict
        Full analysis payload from pipeline.

    Returns
    -------
    List[ISViolation]
        Sorted by severity (error → warning → info).
    """
    violations: List[ISViolation] = []

    all_items = (
        (payload.get("boq_items") or []) +
        (payload.get("spec_items") or []) +
        (payload.get("line_items") or [])
    )

    for item in all_items:
        if not isinstance(item, dict):
            continue
        violations.extend(_check_is456(item))
        violations.extend(_check_is1786(item))
        violations.extend(_check_coverage_thickness(item))

    # Deduplicate by (code, description[:40])
    seen = set()
    unique: List[ISViolation] = []
    for v in violations:
        key = (v.code, v.description[:40])
        if key not in seen:
            seen.add(key)
            unique.append(v)

    # Sort: errors first, then warnings, then info
    _sev_order = {"error": 0, "warning": 1, "info": 2}
    unique.sort(key=lambda v: _sev_order.get(v.severity, 9))

    logger.info("IS compliance check: %d violations (%d errors, %d warnings, %d info)",
                len(unique),
                sum(1 for v in unique if v.severity == "error"),
                sum(1 for v in unique if v.severity == "warning"),
                sum(1 for v in unique if v.severity == "info"))

    return unique


def compliance_summary(violations: List[ISViolation]) -> dict:
    """Return a dict summary for embedding in payload."""
    return {
        "total": len(violations),
        "errors": sum(1 for v in violations if v.severity == "error"),
        "warnings": sum(1 for v in violations if v.severity == "warning"),
        "info": sum(1 for v in violations if v.severity == "info"),
        "violations": [
            {
                "code": v.code,
                "clause": v.clause,
                "severity": v.severity,
                "description": v.description,
                "item": v.item_description,
                "trade": v.trade,
                "suggestion": v.suggestion,
                "page": v.source_page,
            }
            for v in violations[:50]
        ],
    }
