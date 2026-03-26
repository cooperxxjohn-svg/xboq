"""
Sprint 46 — Cash Flow / S-Curve Generator
==========================================
Computes a monthly spend profile (S-curve) from BOQ line items.

Each trade is assigned a timing band so the aggregated spend follows
a realistic construction S-curve (slow start → peak mid-project → tail off).

Usage:
    from src.analysis.cash_flow import compute_cash_flow
    result = compute_cash_flow(line_items, duration_months=18)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Trade timing bands ────────────────────────────────────────────────────────
# (start_frac, peak_frac, end_frac) — fractions of project duration
_TRADE_BANDS: Dict[str, Tuple[float, float, float]] = {
    # Early works
    "sitework":     (0.00, 0.10, 0.25),
    "earthwork":    (0.00, 0.10, 0.25),
    "piling":       (0.00, 0.15, 0.30),
    "foundation":   (0.05, 0.20, 0.35),
    # Structure
    "structural":   (0.10, 0.35, 0.60),
    "concrete":     (0.10, 0.35, 0.60),
    "rcc":          (0.10, 0.35, 0.60),
    "steel":        (0.15, 0.40, 0.65),
    "masonry":      (0.20, 0.45, 0.65),
    # Envelope
    "waterproofing": (0.30, 0.50, 0.70),
    "roofing":      (0.35, 0.55, 0.75),
    "cladding":     (0.35, 0.60, 0.80),
    "glazing":      (0.35, 0.60, 0.80),
    # MEP
    "electrical":   (0.35, 0.60, 0.85),
    "plumbing":     (0.35, 0.60, 0.85),
    "hvac":         (0.40, 0.65, 0.88),
    "fire":         (0.40, 0.65, 0.88),
    "mep":          (0.38, 0.62, 0.86),
    # Finishes
    "painting":     (0.55, 0.78, 0.95),
    "flooring":     (0.55, 0.78, 0.95),
    "tiling":       (0.55, 0.78, 0.95),
    "ceiling":      (0.55, 0.80, 0.96),
    "door":         (0.60, 0.80, 0.96),
    "window":       (0.55, 0.75, 0.93),
    "finishing":    (0.55, 0.80, 0.96),
    "finishes":     (0.55, 0.80, 0.96),
    # External
    "landscaping":  (0.70, 0.88, 0.98),
    "external":     (0.70, 0.88, 0.98),
    # Prelims / overhead
    "preliminaries": (0.00, 0.50, 1.00),
    "prelims":      (0.00, 0.50, 1.00),
    "general":      (0.05, 0.50, 0.95),
}

_DEFAULT_BAND: Tuple[float, float, float] = (0.10, 0.50, 0.90)


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class CashFlowMonth:
    month: int                   # 1-based
    label: str                   # "M1", "M2", …
    planned_spend: float         # INR this month
    cumulative_spend: float      # INR cumulative
    cumulative_pct: float        # 0-100
    trade_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class CashFlowResult:
    months: List[CashFlowMonth]
    total_value: float
    duration_months: int
    # Aggregated for charting
    monthly_spends: List[float]        # [spend_m1, spend_m2, …]
    cumulative_pcts: List[float]       # [0, pct_m1, pct_m2, …] (n+1 points)
    trade_totals: Dict[str, float]     # trade → total INR
    trade_monthly: Dict[str, List[float]]  # trade → [spend_m1, …]
    peak_month: int                    # month with highest spend
    front_half_pct: float              # % of value in first half of project


# ── Triangle distribution helper ─────────────────────────────────────────────

def _triangle_weights(n_months: int, start_f: float, peak_f: float, end_f: float) -> List[float]:
    """
    Generate n_months weights following a triangular distribution
    bounded by [start_f, end_f] and peaking at peak_f (all as fractions of n_months).
    Returns weights that sum to 1.0.
    """
    weights = []
    for m in range(n_months):
        t = (m + 0.5) / n_months          # midpoint of month as fraction
        if t < start_f or t > end_f:
            w = 0.0
        elif t <= peak_f:
            span = peak_f - start_f
            w = (t - start_f) / span if span > 0 else 1.0
        else:
            span = end_f - peak_f
            w = (end_f - t) / span if span > 0 else 1.0
        weights.append(max(w, 0.0))

    total = sum(weights)
    if total == 0:
        # Fallback: uniform across full period
        return [1.0 / n_months] * n_months
    return [w / total for w in weights]


# ── Trade classifier ──────────────────────────────────────────────────────────

def _classify_trade(description: str, trade: str = "") -> str:
    """Map a BOQ item to a trade band key."""
    text = (description + " " + trade).lower()
    # Priority order — more specific first
    checks = [
        ("sitework", ["site", "earthwork", "excavat", "backfill", "compaction", "grading"]),
        ("piling",   ["pile", "piling", "bored", "micro pile"]),
        ("foundation",["foundation", "footing", "raft", "pile cap", "plinth"]),
        ("structural",["structural", "rcc", "reinforced", "concrete", "slab", "beam", "column", "shear", "retaining"]),
        ("masonry",  ["masonry", "brick", "block", "stonework"]),
        ("waterproofing", ["waterproof", "damp proof", "membrane", "bitumen", "injection"]),
        ("roofing",  ["roof", "terrace", "insulation"]),
        ("cladding", ["cladding", "facade", "curtain wall", "composite panel"]),
        ("glazing",  ["glazing", "glass", "aluminium window", "aluminum window", "curtain"]),
        ("electrical",["electrical", "wiring", "conduit", "switchboard", "panel", "mcb", "light fixture", "db", "ups"]),
        ("plumbing", ["plumbing", "sanitary", "toilet", "basin", "pipe", "drain", "sewage", "water supply", "fittings"]),
        ("hvac",     ["hvac", "air conditioning", "ac", "ventilation", "ahu", "chiller", "duct"]),
        ("fire",     ["fire", "sprinkler", "fm 200", "hydrant", "detection"]),
        ("painting", ["paint", "primer", "putty", "emulsion", "distemper", "texture"]),
        ("flooring", ["floor", "tile", "marble", "granite", "vinyl", "epoxy", "terrazzo", "kota"]),
        ("ceiling",  ["ceiling", "false ceiling", "gypsum", "mineral fiber"]),
        ("door",     ["door", "frame", "shutter", "hardware"]),
        ("window",   ["window", "casement", "sliding sash"]),
        ("landscaping", ["landscape", "garden", "plantation", "paver", "turf"]),
        ("external", ["external", "compound wall", "gate", "road", "parking", "storm"]),
        ("preliminaries", ["prelim", "mobilisation", "site establish", "insurance", "safety", "overhead"]),
    ]
    for key, keywords in checks:
        if any(kw in text for kw in keywords):
            return key
    return "general"


# ── Main function ─────────────────────────────────────────────────────────────

def compute_cash_flow(
    line_items: List[Dict],
    duration_months: int,
    project_value_override: Optional[float] = None,
) -> CashFlowResult:
    """
    Compute monthly cash flow from BOQ line items.

    Args:
        line_items: List of BOQ dicts, each with keys like:
            description (str), trade (str, optional), amount / rate+qty (float)
        duration_months: Project duration in calendar months (≥1)
        project_value_override: If provided, scale all amounts so total = this value

    Returns:
        CashFlowResult with per-month spend, cumulative S-curve, trade breakdown
    """
    duration_months = max(1, int(duration_months))

    # ── Extract amounts per item ──────────────────────────────────────────────
    trade_items: Dict[str, float] = {}

    for item in line_items:
        desc  = str(item.get("description", item.get("desc", "")))
        trade = str(item.get("trade", item.get("package", "")))
        amt   = _item_amount(item)
        if amt <= 0:
            continue
        band_key = _classify_trade(desc, trade)
        trade_items[band_key] = trade_items.get(band_key, 0.0) + amt

    if not trade_items:
        # Fallback: single lump-sum uniformly distributed
        trade_items["general"] = project_value_override or 1_000_000.0

    total_value = sum(trade_items.values())
    if project_value_override and project_value_override > 0:
        scale = project_value_override / total_value
        trade_items = {k: v * scale for k, v in trade_items.items()}
        total_value = project_value_override

    # ── Build monthly spend per trade ─────────────────────────────────────────
    trade_monthly: Dict[str, List[float]] = {}
    for trade_key, trade_total in trade_items.items():
        band = _TRADE_BANDS.get(trade_key, _DEFAULT_BAND)
        weights = _triangle_weights(duration_months, *band)
        trade_monthly[trade_key] = [w * trade_total for w in weights]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    monthly_spends: List[float] = [0.0] * duration_months
    for spends in trade_monthly.values():
        for i, v in enumerate(spends):
            monthly_spends[i] += v

    cumulative = 0.0
    months: List[CashFlowMonth] = []
    cumulative_pcts = [0.0]  # starts at 0

    for i, spend in enumerate(monthly_spends):
        cumulative += spend
        pct = (cumulative / total_value * 100) if total_value > 0 else 0.0
        cumulative_pcts.append(pct)

        breakdown = {k: v[i] for k, v in trade_monthly.items() if v[i] > 0}
        months.append(CashFlowMonth(
            month=i + 1,
            label=f"M{i + 1}",
            planned_spend=spend,
            cumulative_spend=cumulative,
            cumulative_pct=pct,
            trade_breakdown=breakdown,
        ))

    peak_month = (monthly_spends.index(max(monthly_spends)) + 1) if monthly_spends else 1
    half = duration_months // 2
    front_half_spend = sum(monthly_spends[:half])
    front_half_pct = (front_half_spend / total_value * 100) if total_value > 0 else 50.0

    # Friendly display names for trades
    trade_totals = {k: sum(v) for k, v in trade_monthly.items()}

    return CashFlowResult(
        months=months,
        total_value=total_value,
        duration_months=duration_months,
        monthly_spends=monthly_spends,
        cumulative_pcts=cumulative_pcts,
        trade_totals=trade_totals,
        trade_monthly=trade_monthly,
        peak_month=peak_month,
        front_half_pct=round(front_half_pct, 1),
    )


def _item_amount(item: Dict) -> float:
    """Extract monetary amount from a BOQ line item dict."""
    # Try direct amount keys
    for key in ("amount", "total_amount", "total", "value", "cost", "estimated_cost_inr"):
        v = item.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    # Try rate × qty
    rate = None
    for key in ("rate", "unit_rate", "rate_inr"):
        v = item.get(key)
        if v is not None:
            try:
                rate = float(v)
                break
            except (TypeError, ValueError):
                pass
    qty = None
    for key in ("qty", "quantity", "takeoff_qty"):
        v = item.get(key)
        if v is not None:
            try:
                qty = float(v)
                break
            except (TypeError, ValueError):
                pass
    if rate is not None and qty is not None:
        return rate * qty
    return 0.0


# ── Formatting helpers (used by UI) ──────────────────────────────────────────

def fmt_inr(amount: float) -> str:
    """Format as Indian Rupees (₹X.XX Cr / ₹X.X L / ₹X)."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.1f} L"
    return f"₹{amount:,.0f}"


TRADE_DISPLAY: Dict[str, str] = {
    "sitework":       "Sitework & Earthwork",
    "piling":         "Piling",
    "foundation":     "Foundation",
    "structural":     "Structural / RCC",
    "masonry":        "Masonry",
    "waterproofing":  "Waterproofing",
    "roofing":        "Roofing",
    "cladding":       "Cladding / Facade",
    "glazing":        "Glazing",
    "electrical":     "Electrical",
    "plumbing":       "Plumbing & Sanitary",
    "hvac":           "HVAC",
    "fire":           "Fire Protection",
    "mep":            "MEP (General)",
    "painting":       "Painting",
    "flooring":       "Flooring",
    "ceiling":        "False Ceiling",
    "door":           "Doors & Hardware",
    "window":         "Windows",
    "landscaping":    "Landscaping",
    "external":       "External Works",
    "preliminaries":  "Preliminaries",
    "general":        "General / Unclassified",
}
