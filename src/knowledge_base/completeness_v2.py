"""
KB-Powered Completeness Scoring v2

Enhanced BOQ completeness scoring using the full 6,000+ item taxonomy.
Scores by discipline coverage, trade depth, and item recognition.

Calibrated scoring model:
  1. Discipline Breadth (40%) — are expected disciplines present?
  2. Match Rate (30%) — do BOQ items match taxonomy?
  3. Trade Depth (20%) — enough items per discipline?
  4. Key Items (10%) — are essential items present?

This gives realistic scores: a 10-item structural BOQ scores ~30-40/100,
a 200-item multi-discipline BOQ scores 70-90/100.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DisciplineCoverage:
    """Coverage assessment for a single discipline."""
    discipline: str
    display_name: str
    taxonomy_items: int         # Total items in KB for this discipline
    matched_items: int          # Items matched from BOQ
    coverage_pct: float         # 0-100 (relative to "expected" items, not total)
    status: str                 # "good" (>60%), "partial" (20-60%), "missing" (<20%)
    key_missing: List[str]      # Top missing items/categories


@dataclass
class CompletenessResultV2:
    """Enhanced completeness scoring result."""
    overall_score: float        # 0-100
    grade: str                  # A-F

    # Discipline-level breakdown
    disciplines_found: int
    disciplines_expected: int   # Based on building type
    discipline_coverage: List[DisciplineCoverage]

    # Item-level metrics
    total_boq_items: int
    items_matched: int
    items_unmatched: int
    match_rate: float           # 0-1

    # Trade-level
    trades_found: int
    trades_in_taxonomy: int

    # Sub-scores (0-100 each)
    breadth_score: float        # Discipline breadth
    match_score: float          # BOQ-to-taxonomy match rate
    depth_score: float          # Items per discipline
    key_items_score: float      # Essential items coverage

    # Improvement hints
    top_gaps: List[str]         # Top 10 missing areas


# Essential items that a complete BOQ should have (per building type)
_ESSENTIAL_ITEMS = {
    "all": [
        # Structural basics
        "excavation", "footing", "column", "beam", "slab",
        "reinforcement", "formwork",
        # Masonry
        "brick", "block", "masonry",
        # Waterproofing
        "waterproofing",
        # Finishes
        "plaster", "painting", "flooring", "tile",
        # Doors/Windows
        "door", "window",
        # Plumbing
        "pipe", "sanitary", "fixture",
        # Electrical
        "wiring", "conduit", "switch", "light",
        # External
        "drain", "road", "paving",
    ],
    "commercial": [
        "fire", "sprinkler", "hvac", "chiller", "elevator", "lift",
        "facade", "curtain", "glazing",
    ],
    "hospital": [
        "fire", "sprinkler", "hvac", "elevator", "medical",
        "gas", "nurse", "clean", "autoclave",
    ],
    "hotel": [
        "fire", "sprinkler", "hvac", "elevator",
        "kitchen", "laundry", "pool",
    ],
    "factory": [
        "fire", "crane", "industrial", "loading",
        "ventilation", "compressed", "effluent",
    ],
}

# Realistic target: items per discipline for a "complete" BOQ
# Uses ACTUAL taxonomy discipline groupings (8 disciplines, not 21)
_DEPTH_TARGETS = {
    "structural": 15,           # footings + columns + beams + slabs + reinforcement + formwork
    "civil": 5,                 # excavation + filling + compaction + piling
    "architectural": 25,        # masonry + waterproofing + doors + windows + ceiling (big group)
    "finishing": 10,            # floor tiles + wall tiles + skirting + paving
    "mep": 20,                  # plumbing + electrical + fire + hvac + elevator (big group)
    "general": 8,               # prelims + external works
    "infrastructure": 3,        # bridges + roads + treatment
    "specialised": 2,           # pool + auditorium + lab
}
_DEFAULT_DEPTH_TARGET = 5


def score_completeness_v2(
    boq_items: List[Dict[str, Any]],
    building_type: str = "residential",
    extra_items: List[Dict[str, Any]] = None,
) -> CompletenessResultV2:
    """
    Score BOQ completeness using full KB taxonomy.

    Calibrated to give realistic scores:
    - 10-item BOQ for a small structure: ~25-40/100
    - 50-item multi-trade BOQ: ~50-65/100
    - 200-item comprehensive BOQ: ~75-95/100

    Args:
        boq_items:    List of dicts with 'description' or 'item_name' keys
        building_type: Type of building for context-aware scoring
        extra_items:  Optional additional items (spec_items + schedule_stubs)
                      prepended to the BOQ items before taxonomy matching to
                      improve coverage scores for TENDER/MIXED packages.

    Returns:
        CompletenessResultV2 with discipline-level breakdown
    """
    from src.knowledge_base import get_taxonomy
    from src.knowledge_base.matcher import match_boq_batch, get_matcher

    # Combine BOQ items with any extra items (spec items + stubs)
    all_boq = list(boq_items or [])
    if extra_items:
        all_boq = list(extra_items) + all_boq

    tax = get_taxonomy()
    all_items = tax.all_items()

    # Group taxonomy items by discipline
    disc_items = {}
    for item in all_items:
        disc = getattr(item, 'discipline', 'other')
        disc_items.setdefault(disc, []).append(item)

    # Match BOQ items — build parallel (desc, unit, section) from same filtered set
    # so unit-family and section-discipline context travels with each description.
    _valid_items    = [b for b in all_boq if b.get("description") or b.get("item_name")]
    boq_descs       = [b.get("description") or b.get("item_name", "") for b in _valid_items]
    boq_units       = [b.get("unit", "") for b in _valid_items]
    boq_sections    = [b.get("section", "") for b in _valid_items]

    matches = match_boq_batch(
        boq_descs, min_confidence=0.3, units=boq_units, sections=boq_sections,
    ) if boq_descs else []
    match_stats = get_matcher().get_match_stats(matches)

    # Track which disciplines/trades were matched
    matched_disciplines = set()
    matched_trades = set()
    matched_by_disc: Dict[str, set] = {}

    for m in matches:
        if m.matched:
            matched_disciplines.add(m.discipline)
            matched_trades.add(m.trade)
            matched_by_disc.setdefault(m.discipline, set()).add(m.taxonomy_id)

    # Determine expected disciplines (using ACTUAL taxonomy discipline names)
    # Taxonomy groups: architectural, civil, finishing, general, infrastructure,
    #                  mep, specialised, structural
    core_disciplines = {
        "civil", "structural", "architectural", "finishing", "mep", "general",
    }

    btype_extras = {
        "residential": set(),
        "commercial": {"infrastructure"},
        "hospital": {"specialised", "infrastructure"},
        "school": set(),
        "hotel": {"specialised"},
        "factory": {"specialised", "infrastructure"},
        "data_center": {"specialised", "infrastructure"},
    }

    expected_disciplines = core_disciplines | btype_extras.get(building_type, set())

    # ── Score 1: Discipline Breadth (40%) ──
    # How many expected disciplines have at least 1 matched item?
    found_expected = matched_disciplines & expected_disciplines
    breadth_raw = len(found_expected) / max(len(expected_disciplines), 1)
    breadth_score = min(100, breadth_raw * 100)

    # ── Score 2: Match Rate (30%) ──
    # What % of BOQ items match something in the taxonomy?
    match_rate = match_stats.get("match_rate", 0)
    match_score = min(100, match_rate * 100)

    # ── Score 3: Trade Depth (20%) ──
    # For matched disciplines, how many items vs target?
    depth_scores = []
    for disc in matched_disciplines:
        count = len(matched_by_disc.get(disc, set()))
        target = _DEPTH_TARGETS.get(disc, _DEFAULT_DEPTH_TARGET)
        disc_depth = min(1.0, count / max(target, 1))
        depth_scores.append(disc_depth)

    depth_score = (sum(depth_scores) / max(len(depth_scores), 1)) * 100 if depth_scores else 0

    # ── Score 4: Key Items (10%) ──
    # Are essential items present (checked by keyword in matched items)?
    essential_keywords = set(_ESSENTIAL_ITEMS.get("all", []))
    essential_keywords |= set(_ESSENTIAL_ITEMS.get(building_type, []))

    # Check if any matched taxonomy item name/alias contains the essential keyword
    matched_names_lower = set()
    for m in matches:
        if m.matched:
            matched_names_lower.add(m.canonical_name.lower())
            matched_names_lower.add(m.matched_alias.lower())

    all_matched_text = " ".join(matched_names_lower)
    found_essential = sum(1 for kw in essential_keywords if kw in all_matched_text)
    key_items_score = min(100, (found_essential / max(len(essential_keywords), 1)) * 100)

    # ── Weighted Overall Score ──
    overall = (
        breadth_score * 0.40 +
        match_score * 0.30 +
        depth_score * 0.20 +
        key_items_score * 0.10
    )

    # Grade
    if overall >= 80: grade = "A"
    elif overall >= 65: grade = "B"
    elif overall >= 50: grade = "C"
    elif overall >= 35: grade = "D"
    elif overall >= 20: grade = "E"
    else: grade = "F"

    # ── Discipline Coverage (for display) ──
    discipline_coverage = []
    for disc, items in sorted(disc_items.items()):
        matched_count = len(matched_by_disc.get(disc, set()))
        total_count = len(items)
        target = _DEPTH_TARGETS.get(disc, _DEFAULT_DEPTH_TARGET)

        # Coverage relative to TARGET, not total taxonomy
        coverage = min(100, (matched_count / max(target, 1)) * 100)

        if coverage > 60:
            status = "good"
        elif coverage > 20:
            status = "partial"
        else:
            status = "missing"

        # Key missing categories
        matched_ids = matched_by_disc.get(disc, set())
        missing_cats = set()
        for it in items:
            if getattr(it, 'id', '') not in matched_ids:
                cat = getattr(it, 'category', '') or getattr(it, 'sub_trade', '')
                if cat:
                    missing_cats.add(cat.replace("_", " ").title())

        display_name = disc.replace("_", " ").title()
        discipline_coverage.append(DisciplineCoverage(
            discipline=disc,
            display_name=display_name,
            taxonomy_items=total_count,
            matched_items=matched_count,
            coverage_pct=round(coverage, 1),
            status=status,
            key_missing=sorted(missing_cats)[:5],
        ))

    # Top gaps: missing expected disciplines, then under-covered ones
    top_gaps = []
    for dc in discipline_coverage:
        if dc.discipline in expected_disciplines and dc.matched_items == 0:
            top_gaps.append(f"{dc.display_name}: not found in BOQ")
    for dc in discipline_coverage:
        if dc.discipline in expected_disciplines and dc.status == "partial":
            top_gaps.append(f"{dc.display_name}: partial ({dc.matched_items} items, target {_DEPTH_TARGETS.get(dc.discipline, _DEFAULT_DEPTH_TARGET)})")
    top_gaps = top_gaps[:10]

    # All trades in taxonomy
    all_trades = set()
    for item in all_items:
        t = getattr(item, 'trade', '')
        if t:
            all_trades.add(t)

    return CompletenessResultV2(
        overall_score=round(overall, 1),
        grade=grade,
        disciplines_found=len(matched_disciplines),
        disciplines_expected=len(expected_disciplines),
        discipline_coverage=discipline_coverage,
        total_boq_items=len(boq_descs),
        items_matched=match_stats.get("matched", 0),
        items_unmatched=match_stats.get("unmatched", 0),
        match_rate=match_stats.get("match_rate", 0),
        trades_found=len(matched_trades),
        trades_in_taxonomy=len(all_trades),
        breadth_score=round(breadth_score, 1),
        match_score=round(match_score, 1),
        depth_score=round(depth_score, 1),
        key_items_score=round(key_items_score, 1),
        top_gaps=top_gaps,
    )
