"""
src/reasoning/cost_impact.py

Estimate financial exposure (INR) for each identified gap.

All estimates are approximate ranges based on construction industry benchmarks
for India (2024 rates). They are NOT quotes — they help the bidder decide
which gaps need subcontractor quotes vs which can use provisional sums.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Benchmark exposure ranges per gap pattern (INR) ───────────────────────────
# Each entry: (min_inr, max_inr, basis_text)

_PATTERN_RULES = [
    # keyword in description        min        max       basis
    ("geotech", "soil report",      200_000,  800_000,  "Geotechnical investigation + foundation risk contingency"),
    ("mep", "3-8% of",              0,        0,        "3-8% of project cost — use grand total for range"),  # special
    ("unmatched schedule",          50_000,   500_000,  "Allowance per unmatched schedule mark × count"),
    ("no boq",                      0,        0,        "Full project cost unknown — cannot price"),
    ("no commercial",               100_000,  500_000,  "Contract risk (LD/retention) — typical 2-5% of contract value"),
    ("finish schedule",             300_000, 1_500_000, "Provisional sum for unspecified finishes"),
    ("unrated",                     200_000, 1_000_000, "Open items requiring subcontractor quotes"),
    ("critical blocker",            500_000, 5_000_000, "Blocker may prevent entire trade from being priced"),
    ("critical rfi",                200_000, 2_000_000, "Unresolved RFI risk — cost depends on scope"),
    ("readiness score",             500_000, 3_000_000, "Multiple information gaps create collective pricing risk"),
    ("mep/services",                0,        0,        "3-8% of project cost for MEP"),
    ("missing drawing",             100_000,  800_000,  "Provisional sum / design risk contingency"),
]

_DEFAULT_MIN = 100_000
_DEFAULT_MAX = 500_000
_DEFAULT_BASIS = "Estimated exposure based on typical construction risk benchmarks"


@dataclass
class CostImpact:
    gap_id: str
    min_inr: float
    max_inr: float
    expected_inr: float      # midpoint (simple average)
    basis: str
    confidence: float        # 0.0-1.0


def estimate_cost_impact(gap: "Gap", payload: dict) -> CostImpact:  # noqa: F821
    """
    Estimate INR cost exposure for a single gap.

    Uses heuristic pattern matching on the gap description.  For percentage-
    based gaps (MEP), derives the range from payload grand_total_inr.
    """
    desc_lower = gap.description.lower()
    grand_total = float(
        (payload.get("qto_summary") or {}).get("grand_total_inr", 0) or 0
    )

    min_inr = _DEFAULT_MIN
    max_inr = _DEFAULT_MAX
    basis = _DEFAULT_BASIS
    confidence = 0.35

    for *keywords, rule_min, rule_max, rule_basis in _PATTERN_RULES:
        if any(kw in desc_lower for kw in keywords):
            if rule_min == 0 and rule_max == 0:
                # Percentage-based rule
                pct_lo, pct_hi = _extract_pct(rule_basis)
                if grand_total > 0:
                    min_inr = grand_total * pct_lo
                    max_inr = grand_total * pct_hi
                    basis = rule_basis
                    confidence = 0.45
                else:
                    min_inr = 500_000
                    max_inr = 5_000_000
                    basis = f"{rule_basis} (project total unknown)"
                    confidence = 0.25
            else:
                min_inr = rule_min
                max_inr = rule_max
                basis = rule_basis
                confidence = 0.40

                # Scale by count if present in description
                count_match = _extract_count(gap.description)
                if count_match and count_match > 1:
                    min_inr = min_inr * min(count_match, 20)
                    max_inr = max_inr * min(count_match, 20)
                    basis += f" (×{min(count_match, 20)} items)"
            break

    # Severity multiplier
    sev_mul = {"CRITICAL": 2.0, "HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}
    mul = sev_mul.get(gap.severity, 1.0)
    min_inr *= mul
    max_inr *= mul

    expected_inr = (min_inr + max_inr) / 2.0

    return CostImpact(
        gap_id=gap.id,
        min_inr=round(min_inr),
        max_inr=round(max_inr),
        expected_inr=round(expected_inr),
        basis=basis,
        confidence=confidence,
    )


def total_exposure(impacts: List[CostImpact]) -> float:
    """Sum of expected_inr across all cost impacts."""
    return sum(ci.expected_inr for ci in impacts)


def _extract_pct(text: str):
    """Extract (lo, hi) percentage from text like '3-8% of project cost'."""
    import re
    m = re.search(r"(\d+)-(\d+)%", text)
    if m:
        return int(m.group(1)) / 100, int(m.group(2)) / 100
    m = re.search(r"(\d+)%", text)
    if m:
        p = int(m.group(1)) / 100
        return p, p * 2
    return 0.03, 0.08


def _extract_count(text: str) -> Optional[int]:
    """Extract leading integer count from gap description."""
    import re
    m = re.match(r"(\d+)\s+\w", text.strip())
    if m:
        return int(m.group(1))
    return None
