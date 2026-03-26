"""
BOQ Completeness Scorer — Comprehensive assessment of BOQ coverage.

Scores a BOQ on multiple dimensions:
1. Trade coverage: Which construction trades are present
2. Item completeness: Does each item have description + qty + unit + rate
3. Scope coverage: Are dependent items present (formwork for RCC, curing for concrete, etc.)
4. IS 1200 compliance: Are measurement bases correct
5. Rate coverage: Are all items priced

Produces a single 0-100 score with detailed breakdown.

Usage:
    from src.boq.completeness_scorer import score_boq_completeness

    items = [
        {"description": "RCC M25 footing", "qty": 12.5, "unit": "cum", "rate": 8500},
        {"description": "Excavation in all soils", "qty": 45.0, "unit": "cum", "rate": 250},
        ...
    ]
    report = score_boq_completeness(items, project_type="residential")
    print(report.summary())
    print(f"Grade: {report.grade}")

Based on IS 456, IS 1200, NBC 2016, CPWD DSR 2024.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .scope_dependencies import (
    ALL_DEPENDENCY_RULES,
    DependencyRule,
    _normalize_description,
    _item_exists_in_boq,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXPECTED TRADES
# =============================================================================

EXPECTED_TRADES: Dict[str, Dict[str, Any]] = {
    "structural": {
        "weight": 15,
        "required": True,
        "keywords": ["rcc", "concrete", "steel", "formwork"],
    },
    "earthwork": {
        "weight": 8,
        "required": True,
        "keywords": ["excavation", "backfill", "earth"],
    },
    "masonry": {
        "weight": 10,
        "required": True,
        "keywords": ["masonry", "brick", "block"],
    },
    "plaster": {
        "weight": 8,
        "required": True,
        "keywords": ["plaster", "plastering"],
    },
    "flooring": {
        "weight": 8,
        "required": True,
        "keywords": ["flooring", "floor", "tile"],
    },
    "painting": {
        "weight": 5,
        "required": True,
        "keywords": ["paint", "emulsion", "primer"],
    },
    "waterproofing": {
        "weight": 5,
        "required": True,
        "keywords": ["waterproof", "damp proof"],
    },
    "doors_windows": {
        "weight": 7,
        "required": True,
        "keywords": ["door", "window", "frame"],
    },
    "plumbing": {
        "weight": 8,
        "required": True,
        "keywords": ["plumb", "pipe", "sanitary", "cpvc"],
    },
    "electrical": {
        "weight": 8,
        "required": True,
        "keywords": ["electric", "wiring", "switch", "point"],
    },
    "external": {
        "weight": 5,
        "required": False,
        "keywords": ["compound wall", "gate", "road", "drain"],
    },
    "false_ceiling": {
        "weight": 3,
        "required": False,
        "keywords": ["false ceiling", "gypsum", "grid"],
    },
    "fire_safety": {
        "weight": 3,
        "required": False,
        "keywords": ["fire", "extinguisher", "hydrant"],
    },
    "hvac": {
        "weight": 4,
        "required": False,
        "keywords": ["hvac", "ac", "duct", "ventil"],
    },
    "lift": {
        "weight": 3,
        "required": False,
        "keywords": ["lift", "elevator"],
    },
}

# ── Knowledge Base keyword extension (additive) ──
try:
    from src.knowledge_base import get_trade_keywords as _kb_trade_kw
    for _trade, _keywords in _kb_trade_kw().items():
        if _trade in EXPECTED_TRADES:
            _existing_kw = set(EXPECTED_TRADES[_trade]["keywords"])
            EXPECTED_TRADES[_trade]["keywords"].extend(
                [k for k in _keywords if k not in _existing_kw]
            )
        else:
            EXPECTED_TRADES[_trade] = {
                "weight": 3,
                "required": False,
                "keywords": _keywords,
            }
    del _trade, _keywords, _existing_kw
except ImportError:
    pass
except Exception as _e:
    logger.warning("Knowledge base trade keyword loading failed: %s", _e)

# IS 1200 unit compliance rules: maps common item patterns to expected units.
# Each entry is (keyword_pattern, expected_unit, is_code_reference).
IS_1200_UNIT_RULES: List[Tuple[str, str, str]] = [
    ("excavation", "cum", "IS 1200 Part 1"),
    ("backfill", "cum", "IS 1200 Part 1"),
    ("pcc", "cum", "IS 1200 Part 7"),
    ("rcc", "cum", "IS 1200 Part 7"),
    ("concrete", "cum", "IS 1200 Part 7"),
    ("formwork", "sqm", "IS 1200 Part 8"),
    ("shuttering", "sqm", "IS 1200 Part 8"),
    ("reinforcement", "kg", "IS 1200 Part 9"),
    ("steel bar", "kg", "IS 1200 Part 9"),
    ("brickwork", "cum", "IS 1200 Part 4"),
    ("brick masonry", "cum", "IS 1200 Part 4"),
    ("block masonry", "cum", "IS 1200 Part 4"),
    ("plaster", "sqm", "IS 1200 Part 12"),
    ("plastering", "sqm", "IS 1200 Part 12"),
    ("flooring", "sqm", "IS 1200 Part 11"),
    ("tiling", "sqm", "IS 1200 Part 11"),
    ("tile", "sqm", "IS 1200 Part 11"),
    ("painting", "sqm", "IS 1200 Part 13"),
    ("whitewash", "sqm", "IS 1200 Part 13"),
    ("waterproofing", "sqm", "IS 1200 Part 17"),
    ("damp proof", "sqm", "IS 1200 Part 17"),
    ("door frame", "rm", "IS 1200 Part 10"),
    ("window frame", "rm", "IS 1200 Part 10"),
    ("door shutter", "sqm", "IS 1200 Part 10"),
    ("skirting", "rm", "IS 1200 Part 11"),
    ("dado", "sqm", "IS 1200 Part 11"),
    ("pipe", "rm", "IS 1200 Part 15"),
]

# Project-type trade overrides: for commercial/industrial, some trades are
# always expected even if marked optional for residential.
PROJECT_TRADE_OVERRIDES: Dict[str, Dict[str, bool]] = {
    "residential": {},
    "commercial": {
        "fire_safety": True,
        "hvac": True,
        "false_ceiling": True,
    },
    "industrial": {
        "fire_safety": True,
        "hvac": True,
        "external": True,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeScore:
    """Score for a single construction trade within the BOQ."""
    trade: str
    weight: int
    present: bool
    item_count: int
    items_with_qty: int
    items_with_rate: int
    score: float  # 0-100 for this trade

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade": self.trade,
            "weight": self.weight,
            "present": self.present,
            "item_count": self.item_count,
            "items_with_qty": self.items_with_qty,
            "items_with_rate": self.items_with_rate,
            "score": round(self.score, 1),
        }


@dataclass
class CompletenessReport:
    """
    Full completeness assessment report for a BOQ.

    Attributes:
        overall_score: Weighted composite score from 0 to 100.
        trade_scores: Per-trade scoring breakdown.
        missing_trades: Names of required trades not found in the BOQ.
        incomplete_items: Items missing qty, unit, or rate.
        dependency_gaps: Missing dependent items identified via scope rules.
        is_1200_issues: Items with incorrect units per IS 1200.
        rate_coverage_pct: Percentage of items that have a rate.
        qty_coverage_pct: Percentage of items that have a non-zero quantity.
        item_count: Total number of BOQ items analysed.
        trades_found: Number of distinct trades detected.
        trades_expected: Number of trades that should be present.
        grade: Letter grade (A/B/C/D/F) based on overall score.
        component_scores: Individual dimension scores used in weighting.
    """
    overall_score: float = 0.0
    trade_scores: List[TradeScore] = field(default_factory=list)
    missing_trades: List[str] = field(default_factory=list)
    incomplete_items: List[Dict[str, Any]] = field(default_factory=list)
    dependency_gaps: List[Dict[str, Any]] = field(default_factory=list)
    is_1200_issues: List[Dict[str, Any]] = field(default_factory=list)
    rate_coverage_pct: float = 0.0
    qty_coverage_pct: float = 0.0
    item_count: int = 0
    trades_found: int = 0
    trades_expected: int = 0
    grade: str = "F"
    component_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire report to a plain dictionary."""
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "item_count": self.item_count,
            "trades_found": self.trades_found,
            "trades_expected": self.trades_expected,
            "rate_coverage_pct": round(self.rate_coverage_pct, 1),
            "qty_coverage_pct": round(self.qty_coverage_pct, 1),
            "component_scores": {
                k: round(v, 1) for k, v in self.component_scores.items()
            },
            "trade_scores": [ts.to_dict() for ts in self.trade_scores],
            "missing_trades": self.missing_trades,
            "incomplete_items_count": len(self.incomplete_items),
            "incomplete_items": self.incomplete_items[:50],  # cap for readability
            "dependency_gaps_count": len(self.dependency_gaps),
            "dependency_gaps": self.dependency_gaps[:50],
            "is_1200_issues_count": len(self.is_1200_issues),
            "is_1200_issues": self.is_1200_issues[:30],
        }

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"BOQ Completeness Report",
            f"{'=' * 50}",
            f"Overall Score : {self.overall_score:.0f}/100  (Grade {self.grade})",
            f"Items         : {self.item_count}",
            f"Trades Found  : {self.trades_found}/{self.trades_expected}",
            f"Rate Coverage : {self.rate_coverage_pct:.0f}%",
            f"Qty Coverage  : {self.qty_coverage_pct:.0f}%",
            "",
            "Component Scores:",
        ]
        for name, score in self.component_scores.items():
            lines.append(f"  {name:25s}: {score:.0f}/100")

        if self.missing_trades:
            lines.append("")
            lines.append(f"Missing Trades ({len(self.missing_trades)}):")
            for t in self.missing_trades:
                lines.append(f"  - {t}")

        if self.dependency_gaps:
            lines.append("")
            lines.append(f"Scope Gaps ({len(self.dependency_gaps)}):")
            for gap in self.dependency_gaps[:10]:
                lines.append(
                    f"  - Missing '{gap['missing_item']}' "
                    f"(required by {gap['triggered_by']}, "
                    f"priority {gap['priority']})"
                )
            if len(self.dependency_gaps) > 10:
                lines.append(f"  ... and {len(self.dependency_gaps) - 10} more")

        if self.is_1200_issues:
            lines.append("")
            lines.append(f"IS 1200 Unit Issues ({len(self.is_1200_issues)}):")
            for issue in self.is_1200_issues[:5]:
                lines.append(
                    f"  - '{issue['description'][:60]}' "
                    f"has unit '{issue['actual_unit']}', "
                    f"expected '{issue['expected_unit']}' "
                    f"({issue['reference']})"
                )

        if self.incomplete_items:
            lines.append("")
            lines.append(f"Incomplete Items ({len(self.incomplete_items)}):")
            for item in self.incomplete_items[:5]:
                missing = ", ".join(item.get("missing_fields", []))
                desc = item.get("description", "?")[:50]
                lines.append(f"  - '{desc}' missing: {missing}")
            if len(self.incomplete_items) > 5:
                lines.append(
                    f"  ... and {len(self.incomplete_items) - 5} more"
                )

        lines.append("")
        lines.append(_grade_explanation(self.grade))
        return "\n".join(lines)


# =============================================================================
# GRADE BOUNDARIES
# =============================================================================

_GRADE_THRESHOLDS: List[Tuple[str, float, str]] = [
    ("A", 90.0, "Ready to bid"),
    ("B", 75.0, "Minor gaps, can bid with qualifications"),
    ("C", 60.0, "Significant gaps, needs review"),
    ("D", 40.0, "Major gaps, high risk"),
    ("F", 0.0, "Incomplete, do not bid"),
]


def _assign_grade(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    for grade, threshold, _ in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def _grade_explanation(grade: str) -> str:
    """Return the meaning string for a grade."""
    for g, _, explanation in _GRADE_THRESHOLDS:
        if g == grade:
            return f"Grade {g}: {explanation}"
    return f"Grade {grade}: Unknown"


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _extract_description(item: Dict[str, Any]) -> str:
    """Pull description from a BOQ item dict, tolerating multiple key names."""
    for key in ("description", "item_description", "item_name", "desc", "name"):
        val = item.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_qty(item: Dict[str, Any]) -> Optional[float]:
    """Pull quantity from a BOQ item dict and coerce to float."""
    for key in ("qty", "quantity", "Qty", "Quantity"):
        val = item.get(key)
        if val is not None:
            try:
                fval = float(val)
                if fval > 0:
                    return fval
            except (ValueError, TypeError):
                pass
    return None


def _extract_unit(item: Dict[str, Any]) -> Optional[str]:
    """Pull unit from a BOQ item dict."""
    for key in ("unit", "uom", "Unit", "UOM"):
        val = item.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_rate(item: Dict[str, Any]) -> Optional[float]:
    """Pull rate from a BOQ item dict and coerce to float."""
    for key in ("rate", "unit_rate", "Rate", "unit_price", "price"):
        val = item.get(key)
        if val is not None:
            try:
                fval = float(val)
                if fval > 0:
                    return fval
            except (ValueError, TypeError):
                pass
    return None


def _match_trade(description: str, trade_cfg: Dict[str, Any]) -> bool:
    """Check whether a description matches any keyword for a trade."""
    desc_lower = description.lower()
    for kw in trade_cfg["keywords"]:
        if kw in desc_lower:
            return True
    return False


# =============================================================================
# DIMENSION SCORERS
# =============================================================================

def _score_trade_coverage(
    boq_items: List[Dict[str, Any]],
    project_type: str,
    include_external: bool,
) -> Tuple[float, List[TradeScore], List[str]]:
    """
    Score trade coverage: which expected trades are present in the BOQ.

    Returns:
        (score 0-100, list of TradeScore, list of missing trade names)
    """
    overrides = PROJECT_TRADE_OVERRIDES.get(project_type, {})

    # Collect all descriptions for matching.
    descriptions = [_extract_description(item) for item in boq_items]
    descriptions = [d for d in descriptions if d]

    trade_scores: List[TradeScore] = []
    total_weight = 0
    earned_weight = 0
    missing_trades: List[str] = []

    for trade_name, cfg in EXPECTED_TRADES.items():
        # Determine whether this trade is expected for the project type.
        is_required = cfg["required"]
        if trade_name in overrides:
            is_required = overrides[trade_name]
        if trade_name == "external" and not include_external:
            is_required = False

        weight = cfg["weight"]

        # Find items belonging to this trade.
        matched_items = []
        for i, desc in enumerate(descriptions):
            if _match_trade(desc, cfg):
                matched_items.append(boq_items[i])

        present = len(matched_items) > 0
        items_with_qty = sum(
            1 for item in matched_items if _extract_qty(item) is not None
        )
        items_with_rate = sum(
            1 for item in matched_items if _extract_rate(item) is not None
        )

        # Per-trade score: 60% for presence, 25% for having qty, 15% for rate.
        if not present:
            trade_score_val = 0.0
        else:
            presence_part = 60.0
            qty_part = (
                25.0 * (items_with_qty / len(matched_items))
                if matched_items
                else 0.0
            )
            rate_part = (
                15.0 * (items_with_rate / len(matched_items))
                if matched_items
                else 0.0
            )
            trade_score_val = presence_part + qty_part + rate_part

        ts = TradeScore(
            trade=trade_name,
            weight=weight,
            present=present,
            item_count=len(matched_items),
            items_with_qty=items_with_qty,
            items_with_rate=items_with_rate,
            score=trade_score_val,
        )
        trade_scores.append(ts)

        # Accumulate weighted score only for expected trades.
        if is_required:
            total_weight += weight
            earned_weight += weight * (trade_score_val / 100.0)
            if not present:
                missing_trades.append(trade_name)

    # Overall trade coverage score.
    coverage_score = (earned_weight / total_weight * 100.0) if total_weight else 0.0
    return coverage_score, trade_scores, missing_trades


def _score_item_completeness(
    boq_items: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]], float, float]:
    """
    Score individual item completeness.

    Each item should have: description (required), qty (required),
    unit (required), rate (desired).

    Returns:
        (score 0-100, incomplete_items list, qty_coverage_pct, rate_coverage_pct)
    """
    if not boq_items:
        return 0.0, [], 0.0, 0.0

    total = len(boq_items)
    items_with_desc = 0
    items_with_qty = 0
    items_with_unit = 0
    items_with_rate = 0
    incomplete: List[Dict[str, Any]] = []

    for item in boq_items:
        desc = _extract_description(item)
        qty = _extract_qty(item)
        unit = _extract_unit(item)
        rate = _extract_rate(item)

        has_desc = bool(desc)
        has_qty = qty is not None
        has_unit = unit is not None
        has_rate = rate is not None

        if has_desc:
            items_with_desc += 1
        if has_qty:
            items_with_qty += 1
        if has_unit:
            items_with_unit += 1
        if has_rate:
            items_with_rate += 1

        # Track items that are incomplete.
        missing_fields: List[str] = []
        if not has_desc:
            missing_fields.append("description")
        if not has_qty:
            missing_fields.append("qty")
        if not has_unit:
            missing_fields.append("unit")
        if not has_rate:
            missing_fields.append("rate")

        if missing_fields:
            incomplete.append({
                "description": desc or "(no description)",
                "missing_fields": missing_fields,
                "item": {k: v for k, v in item.items() if k != "_raw"},
            })

    # Weight: description 30%, qty 30%, unit 20%, rate 20%.
    desc_pct = items_with_desc / total
    qty_pct = items_with_qty / total
    unit_pct = items_with_unit / total
    rate_pct = items_with_rate / total

    score = (desc_pct * 30.0 + qty_pct * 30.0 + unit_pct * 20.0 + rate_pct * 20.0)

    return score, incomplete, qty_pct * 100.0, rate_pct * 100.0


def _score_scope_dependencies(
    boq_items: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Check scope dependency coverage using rules from scope_dependencies.py.

    Identifies items in the BOQ that trigger dependency rules and checks
    whether the required companion items are also present.

    Returns:
        (score 0-100, list of dependency gap dicts)
    """
    # Build description set for matching.
    boq_descs: Set[str] = set()
    for item in boq_items:
        desc = _extract_description(item)
        if desc:
            boq_descs.add(desc)

    if not boq_descs:
        return 0.0, []

    # We focus on the most actionable rules: those whose trigger is
    # identifiable in the BOQ itself. Building-level and room-level triggers
    # are only checked when such keywords appear in item descriptions.
    total_required = 0
    total_found = 0
    gaps: List[Dict[str, Any]] = []

    triggers_checked: Set[str] = set()

    for rule in ALL_DEPENDENCY_RULES:
        trigger = rule.trigger.lower()

        # Check if trigger is represented in the BOQ descriptions.
        trigger_present = _item_exists_in_boq(rule.trigger, boq_descs)
        if not trigger_present:
            continue

        if trigger in triggers_checked and rule.priority > 1:
            # Avoid double-counting low-priority rules for same trigger.
            pass
        triggers_checked.add(trigger)

        for required_item in rule.required_items:
            total_required += 1
            if _item_exists_in_boq(required_item, boq_descs):
                total_found += 1
            else:
                gaps.append({
                    "missing_item": required_item,
                    "triggered_by": rule.trigger,
                    "trade": rule.trade.value,
                    "priority": rule.priority,
                    "note": rule.note,
                })

    if total_required == 0:
        # No dependency rules fired. If we have items, give partial credit.
        return 70.0 if boq_descs else 0.0, []

    score = (total_found / total_required) * 100.0
    return score, gaps


def _score_is_1200_compliance(
    boq_items: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Check IS 1200 measurement unit compliance.

    For items whose description matches known IS 1200 categories, verify
    the unit is correct.

    Returns:
        (score 0-100, list of issues)
    """
    checked = 0
    correct = 0
    issues: List[Dict[str, Any]] = []

    for item in boq_items:
        desc = _extract_description(item)
        unit = _extract_unit(item)
        if not desc or not unit:
            continue

        desc_lower = desc.lower()
        unit_lower = unit.lower().strip()

        for pattern, expected_unit, reference in IS_1200_UNIT_RULES:
            if pattern in desc_lower:
                checked += 1
                if unit_lower == expected_unit.lower():
                    correct += 1
                else:
                    # Allow common aliases.
                    aliases = _unit_aliases(expected_unit)
                    if unit_lower in aliases:
                        correct += 1
                    else:
                        issues.append({
                            "description": desc,
                            "actual_unit": unit,
                            "expected_unit": expected_unit,
                            "reference": reference,
                        })
                break  # Only match first rule per item.

    if checked == 0:
        # No IS 1200 checkable items; neutral score.
        return 80.0, []

    score = (correct / checked) * 100.0
    return score, issues


def _unit_aliases(canonical: str) -> Set[str]:
    """Return lowercase aliases for a canonical unit."""
    alias_map: Dict[str, Set[str]] = {
        "cum": {"cum", "m3", "cu.m", "cu m", "cubic meter", "cubic metre", "cmt"},
        "sqm": {"sqm", "m2", "sq.m", "sq m", "square meter", "square metre", "smt"},
        "kg": {"kg", "kgs", "kilogram", "kilograms"},
        "rm": {"rm", "rmt", "r.m", "running meter", "running metre", "m"},
        "nos": {"nos", "no", "no.", "number", "numbers", "each", "ea"},
    }
    return alias_map.get(canonical.lower(), {canonical.lower()})


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def score_boq_completeness(
    boq_items: List[Dict[str, Any]],
    project_type: str = "residential",
    include_external: bool = True,
) -> CompletenessReport:
    """
    Score a BOQ for completeness across multiple dimensions.

    This is the primary entry point. It takes a list of BOQ item dicts
    and returns a CompletenessReport with an overall 0-100 score, letter
    grade, and detailed breakdown.

    Args:
        boq_items: List of BOQ item dicts. Each dict should contain at
            minimum a 'description' key. 'qty', 'unit', and 'rate' are
            checked for completeness.
        project_type: One of 'residential', 'commercial', 'industrial'.
            Affects which trades are expected.
        include_external: Whether external works should be expected.
            Useful to disable for interior-only scopes.

    Returns:
        CompletenessReport with overall score, grade, and all details.

    Score formula:
        overall = (trade_coverage * 0.35)
                + (item_completeness * 0.25)
                + (scope_deps * 0.25)
                + (rate_coverage * 0.15)

    Grade boundaries:
        A: 90-100 (Ready to bid)
        B: 75-89  (Minor gaps, can bid with qualifications)
        C: 60-74  (Significant gaps, needs review)
        D: 40-59  (Major gaps, high risk)
        F: 0-39   (Incomplete, do not bid)
    """
    report = CompletenessReport()
    report.item_count = len(boq_items)

    if not boq_items:
        report.grade = "F"
        report.component_scores = {
            "trade_coverage": 0.0,
            "item_completeness": 0.0,
            "scope_dependencies": 0.0,
            "rate_coverage": 0.0,
        }
        logger.info("BOQ completeness: 0 items, grade F")
        return report

    # --- Dimension 1: Trade coverage (35%) ---
    trade_score, trade_scores, missing_trades = _score_trade_coverage(
        boq_items, project_type, include_external
    )
    report.trade_scores = trade_scores
    report.missing_trades = missing_trades
    report.trades_found = sum(1 for ts in trade_scores if ts.present)
    report.trades_expected = sum(
        1
        for name, cfg in EXPECTED_TRADES.items()
        if cfg["required"]
        or PROJECT_TRADE_OVERRIDES.get(project_type, {}).get(name, False)
    )

    # --- Dimension 2: Item completeness (25%) ---
    item_score, incomplete_items, qty_pct, rate_pct = _score_item_completeness(
        boq_items
    )
    report.incomplete_items = incomplete_items
    report.qty_coverage_pct = qty_pct
    report.rate_coverage_pct = rate_pct

    # --- Dimension 3: Scope dependency coverage (25%) ---
    scope_score, dependency_gaps = _score_scope_dependencies(boq_items)
    report.dependency_gaps = dependency_gaps

    # --- Dimension 4: IS 1200 compliance (folded into rate_coverage weight) ---
    # IS 1200 compliance is informational; we use rate_coverage_pct for the
    # fourth weighted component but also report IS 1200 issues.
    is_1200_score, is_1200_issues = _score_is_1200_compliance(boq_items)
    report.is_1200_issues = is_1200_issues

    # The fourth component blends rate coverage (70%) with IS 1200 (30%).
    rate_dimension = rate_pct * 0.70 + is_1200_score * 0.30

    # --- Composite ---
    overall = (
        trade_score * 0.35
        + item_score * 0.25
        + scope_score * 0.25
        + rate_dimension * 0.15
    )

    # Clamp to 0-100.
    overall = max(0.0, min(100.0, overall))

    report.overall_score = overall
    report.grade = _assign_grade(overall)
    report.component_scores = {
        "trade_coverage": trade_score,
        "item_completeness": item_score,
        "scope_dependencies": scope_score,
        "rate_coverage": rate_dimension,
    }

    logger.info(
        "BOQ completeness: score=%.1f, grade=%s, items=%d, trades=%d/%d",
        overall, report.grade, report.item_count,
        report.trades_found, report.trades_expected,
    )

    return report


# =============================================================================
# CONVENIENCE HELPERS
# =============================================================================

def quick_score(boq_items: List[Dict[str, Any]]) -> float:
    """
    Return just the overall score (0-100) without building a full report.

    Useful for dashboards or sorting multiple BOQs by quality.
    """
    report = score_boq_completeness(boq_items)
    return report.overall_score


def identify_missing_trades(
    boq_items: List[Dict[str, Any]],
    project_type: str = "residential",
) -> List[str]:
    """
    Return list of required-but-missing trade names.

    Useful for quick gap analysis without full scoring.
    """
    _, _, missing = _score_trade_coverage(boq_items, project_type, True)
    return missing


def get_improvement_suggestions(
    report: CompletenessReport,
    max_suggestions: int = 10,
) -> List[str]:
    """
    Generate actionable improvement suggestions from a completeness report.

    Returns a list of plain-English suggestions, prioritised by impact.

    Args:
        report: A CompletenessReport from score_boq_completeness.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of suggestion strings.
    """
    suggestions: List[str] = []

    # Missing trades are the highest-impact fix.
    for trade in report.missing_trades:
        weight = EXPECTED_TRADES.get(trade, {}).get("weight", 0)
        suggestions.append(
            f"Add {trade.replace('_', ' ')} items "
            f"(weight {weight}% of trade score)"
        )

    # Critical dependency gaps.
    critical_gaps = [g for g in report.dependency_gaps if g["priority"] == 1]
    seen_items: Set[str] = set()
    for gap in critical_gaps:
        item = gap["missing_item"]
        if item not in seen_items:
            seen_items.add(item)
            suggestions.append(
                f"Add missing scope item: {item.replace('_', ' ')} "
                f"(required by {gap['triggered_by']})"
            )

    # Rate coverage.
    if report.rate_coverage_pct < 80:
        suggestions.append(
            f"Add rates to BOQ items — currently only "
            f"{report.rate_coverage_pct:.0f}% have rates"
        )

    # Quantity coverage.
    if report.qty_coverage_pct < 90:
        suggestions.append(
            f"Add quantities to BOQ items — currently only "
            f"{report.qty_coverage_pct:.0f}% have quantities"
        )

    # IS 1200 issues.
    if report.is_1200_issues:
        suggestions.append(
            f"Fix {len(report.is_1200_issues)} unit(s) that do not "
            f"comply with IS 1200 measurement standards"
        )

    return suggestions[:max_suggestions]
