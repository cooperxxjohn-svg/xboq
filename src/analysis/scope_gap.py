"""
Scope Gap Detector — cross-references BOQ items against drawing entities,
page text, and spec requirements to flag potentially missing scope items.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── Trade scope templates ─────────────────────────────────────────────────────
# Each entry: {trade, keywords_in_drawings, expected_boq_keywords}
# If drawing keywords are present but no matching BOQ item → gap flagged

_SCOPE_TEMPLATES = [
    {
        "trade": "civil",
        "gap_id": "CIV-01",
        "drawing_signals": ["excavation", "earthwork", "site clearing", "grading", "cut and fill"],
        "boq_keywords": ["excavation", "earthwork", "earth", "cut", "fill"],
        "description": "Earthwork / Excavation",
        "severity": "high",
    },
    {
        "trade": "civil",
        "gap_id": "CIV-02",
        "drawing_signals": ["footing", "foundation", "pile", "raft", "mat foundation"],
        "boq_keywords": ["footing", "foundation", "pile", "raft", "grade beam"],
        "description": "Foundation Works",
        "severity": "high",
    },
    {
        "trade": "civil",
        "gap_id": "CIV-03",
        "drawing_signals": ["column", "rcc column", "rc column"],
        "boq_keywords": ["column", "rcc col", "rc col"],
        "description": "RCC Columns",
        "severity": "high",
    },
    {
        "trade": "civil",
        "gap_id": "CIV-04",
        "drawing_signals": ["beam", "rcc beam", "rc beam", "lintel"],
        "boq_keywords": ["beam", "lintel"],
        "description": "RCC Beams / Lintels",
        "severity": "medium",
    },
    {
        "trade": "civil",
        "gap_id": "CIV-05",
        "drawing_signals": ["slab", "roof slab", "floor slab"],
        "boq_keywords": ["slab"],
        "description": "RCC Slabs",
        "severity": "high",
    },
    {
        "trade": "civil",
        "gap_id": "CIV-06",
        "drawing_signals": ["staircase", "stair", "steps"],
        "boq_keywords": ["stair", "step", "staircase", "riser", "tread"],
        "description": "Staircase",
        "severity": "medium",
    },
    {
        "trade": "architectural",
        "gap_id": "ARCH-01",
        "drawing_signals": ["brick", "block", "masonry", "wall"],
        "boq_keywords": ["brick", "block", "masonry", "brickwork"],
        "description": "Brickwork / Masonry Walls",
        "severity": "high",
    },
    {
        "trade": "architectural",
        "gap_id": "ARCH-02",
        "drawing_signals": ["plaster", "rendering"],
        "boq_keywords": ["plaster", "render", "skim"],
        "description": "Plastering",
        "severity": "medium",
    },
    {
        "trade": "architectural",
        "gap_id": "ARCH-03",
        "drawing_signals": ["tile", "flooring", "floor finish", "marble", "granite"],
        "boq_keywords": ["tile", "floor", "marble", "granite", "vitrified", "ceramic"],
        "description": "Floor Finishes",
        "severity": "medium",
    },
    {
        "trade": "architectural",
        "gap_id": "ARCH-04",
        "drawing_signals": ["paint", "painting", "emulsion"],
        "boq_keywords": ["paint", "emulsion", "distemper", "primer"],
        "description": "Painting",
        "severity": "medium",
    },
    {
        "trade": "architectural",
        "gap_id": "ARCH-05",
        "drawing_signals": ["door", "d1", "d2", "door schedule"],
        "boq_keywords": ["door", "shutter", "frame"],
        "description": "Doors & Frames",
        "severity": "medium",
    },
    {
        "trade": "architectural",
        "gap_id": "ARCH-06",
        "drawing_signals": ["window", "w1", "w2", "window schedule"],
        "boq_keywords": ["window", "glazing", "shutter"],
        "description": "Windows & Glazing",
        "severity": "medium",
    },
    {
        "trade": "mep",
        "gap_id": "MEP-01",
        "drawing_signals": ["water supply", "plumbing", "cp fitting", "sanitary"],
        "boq_keywords": ["plumbing", "water supply", "sanitary", "cp fitting", "pipe"],
        "description": "Plumbing & Water Supply",
        "severity": "high",
    },
    {
        "trade": "mep",
        "gap_id": "MEP-02",
        "drawing_signals": ["drainage", "sewer", "sewage", "sump"],
        "boq_keywords": ["drain", "sewer", "sewage", "sump", "manhole"],
        "description": "Drainage & Sewerage",
        "severity": "high",
    },
    {
        "trade": "mep",
        "gap_id": "MEP-03",
        "drawing_signals": ["electrical", "wiring", "panel", "distribution board", "db", "mcb"],
        "boq_keywords": ["electrical", "wiring", "cable", "conduit", "switch", "panel"],
        "description": "Electrical Works",
        "severity": "high",
    },
    {
        "trade": "mep",
        "gap_id": "MEP-04",
        "drawing_signals": ["hvac", "air conditioning", "ac", "duct", "ventilation"],
        "boq_keywords": ["hvac", "ac", "air condition", "duct", "ventilation"],
        "description": "HVAC / Air Conditioning",
        "severity": "medium",
    },
    {
        "trade": "mep",
        "gap_id": "MEP-05",
        "drawing_signals": ["fire fighting", "sprinkler", "hydrant", "fire alarm"],
        "boq_keywords": ["fire", "sprinkler", "hydrant", "alarm"],
        "description": "Fire Fighting System",
        "severity": "high",
    },
    {
        "trade": "sitework",
        "gap_id": "SITE-01",
        "drawing_signals": ["compound wall", "boundary wall", "retaining wall"],
        "boq_keywords": ["compound wall", "boundary wall", "retaining wall"],
        "description": "Compound / Boundary Wall",
        "severity": "medium",
    },
    {
        "trade": "sitework",
        "gap_id": "SITE-02",
        "drawing_signals": ["road", "pavement", "paving", "driveway", "parking"],
        "boq_keywords": ["road", "paving", "pavement", "bitumen", "concrete road"],
        "description": "Roads & Paving",
        "severity": "medium",
    },
    {
        "trade": "structural",
        "gap_id": "STR-01",
        "drawing_signals": ["steel structure", "ms structure", "structural steel", "truss"],
        "boq_keywords": ["steel structure", "structural steel", "ms section", "truss"],
        "description": "Structural Steel",
        "severity": "high",
    },
]


@dataclass
class ScopeGap:
    gap_id: str
    trade: str
    description: str
    severity: str           # "high" | "medium" | "low"
    drawing_signals_found: List[str] = field(default_factory=list)
    boq_coverage: str = "missing"   # "missing" | "partial" | "present"
    matched_boq_items: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class ScopeGapResult:
    gaps: List[ScopeGap] = field(default_factory=list)
    total_gaps: int = 0
    high_severity: int = 0
    medium_severity: int = 0
    coverage_pct: float = 0.0
    summary: str = ""


def detect_scope_gaps(payload: dict) -> ScopeGapResult:
    """
    Detect scope gaps by cross-referencing drawing signals against BOQ items.
    Always returns gracefully.
    """
    result = ScopeGapResult()

    # Collect all text signals from payload
    drawing_text = _collect_drawing_text(payload)
    boq_items = payload.get("boq_items", [])
    boq_text = " ".join(
        str(i.get("description", "")) + " " + str(i.get("trade", ""))
        for i in boq_items
    ).lower()

    templates_checked = 0
    templates_covered = 0

    for tmpl in _SCOPE_TEMPLATES:
        templates_checked += 1

        # Check if drawing signals are present
        signals_found = [s for s in tmpl["drawing_signals"] if s.lower() in drawing_text]
        if not signals_found:
            # No drawing signal — can't determine gap, skip
            templates_covered += 1  # don't flag as gap if no signal
            continue

        # Check BOQ coverage
        boq_matches = [kw for kw in tmpl["boq_keywords"] if kw.lower() in boq_text]
        if boq_matches:
            coverage = "present"
            templates_covered += 1
        else:
            coverage = "missing"

        if coverage == "missing":
            gap = ScopeGap(
                gap_id=tmpl["gap_id"],
                trade=tmpl["trade"],
                description=tmpl["description"],
                severity=tmpl["severity"],
                drawing_signals_found=signals_found,
                boq_coverage=coverage,
                matched_boq_items=boq_matches,
                recommendation=f"Drawing/spec references '{signals_found[0]}' but no matching BOQ item found. "
                               f"Add line items for: {tmpl['description']}.",
            )
            result.gaps.append(gap)

    result.total_gaps = len(result.gaps)
    result.high_severity = sum(1 for g in result.gaps if g.severity == "high")
    result.medium_severity = sum(1 for g in result.gaps if g.severity == "medium")
    result.coverage_pct = round(templates_covered / max(templates_checked, 1) * 100, 1)

    if result.total_gaps == 0:
        result.summary = "No scope gaps detected \u2014 BOQ appears to cover all identified drawing elements."
    else:
        result.summary = (
            f"**{result.total_gaps} potential scope gap{'s' if result.total_gaps != 1 else ''}** detected "
            f"({result.high_severity} high, {result.medium_severity} medium severity). "
            f"These items appear in drawings/specs but have no matching BOQ line item."
        )

    return result


def _collect_drawing_text(payload: dict) -> str:
    """Collect all text from payload that could contain drawing/spec signals."""
    parts = []

    # Page index text snippets
    pi = payload.get("diagnostics", {}).get("page_index", {})
    if isinstance(pi, dict):
        for page in pi.get("pages", []):
            parts.append(str(page.get("text_snippet", "")))
            parts.append(str(page.get("discipline", "")))
            parts.append(str(page.get("doc_type", "")))

    # Structural takeoff
    st = payload.get("structural_takeoff", {})
    if isinstance(st, dict):
        parts.append(json.dumps(st))

    # Plan graph
    pg = payload.get("plan_graph", {})
    if isinstance(pg, dict):
        parts.append(json.dumps(pg))

    # Requirements by trade
    rbt = payload.get("requirements_by_trade", {})
    if isinstance(rbt, dict):
        parts.append(json.dumps(rbt))

    # BOQ items descriptions (to check for signals in any attached spec text)
    for item in payload.get("boq_items", []):
        parts.append(str(item.get("description", "")))

    # RFIs often contain scope clues
    for rfi in payload.get("rfis", []):
        parts.append(str(rfi.get("question", rfi.get("rfi_text", ""))))
        parts.append(str(rfi.get("context", "")))

    # Notes / scope
    parts.append(str(payload.get("scope_summary", "")))
    parts.append(str(payload.get("project_name", "")))

    return " ".join(parts).lower()
