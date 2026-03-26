"""
Bid margin analysis: tender value (from extracted BOQ rates OR NIT stated cost)
vs contractor cost (from market/DSR rates).

Tender value sources, in priority order:
  1. Per-item rates extracted from the BOQ table (most accurate — item-rate tenders)
  2. NIT Estimated Cost stated on the cover / preliminary pages (percentage-rate tenders,
     or item-rate tenders where the BOQ rate column wasn't extracted)

Contractor cost: market/DSR rates × extracted quantities (what it actually costs to build).
Margin = Tender Value − Contractor Cost.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeMargin:
    trade: str
    tender_value_inr: float          # extracted BOQ rate × qty
    contractor_cost_inr: float       # market/DSR rate × qty
    margin_inr: float                # tender_value - contractor_cost
    margin_pct: float                # margin_inr / tender_value * 100
    item_count: int
    items_with_tender_rate: int      # how many line items had extracted rates
    coverage_pct: float              # items_with_tender_rate / item_count * 100


@dataclass
class BidMarginResult:
    tender_value_inr: float          # total from BOQ extracted rates × qty
    contractor_cost_inr: float       # total from market rates × qty
    margin_inr: float
    margin_pct: float
    by_trade: List[TradeMargin]
    coverage_pct: float              # % of items that had an extracted tender rate
    note: str                        # e.g. "Based on 78% BOQ coverage (42/54 items)"
    reliable: bool                   # True if coverage_pct >= 50
    scope_coverage_pct: float = 0.0  # % of NIT-stated tender value covered by QTO cost


_CRORE = 1_00_00_000   # 10 million
_LAKH  = 1_00_000      # 100 thousand

_NIT_SNIPPET_RE = re.compile(
    # Require the unit (crore/lakh) to be present so we don't grab small interim numbers
    r'(?:estimated\s+cost(?:\s+put\s+to\s+tender)?'
    r'|cost\s+put\s+to\s+tender'
    r'|tender\s+value'
    r'|approximate\s+(?:estimated\s+)?cost)'
    r'.{0,200}?'
    r'(?:Rs\.?\s*|INR\s*|₹\s*)?'
    r'(\d[\d,]*(?:\.\d+)?)\s+(crore|lakh|lac|million|billion)',  # unit REQUIRED
    re.IGNORECASE,
)


def extract_nit_value(commercial_terms: list) -> Optional[float]:
    """
    Extract the NIT (Notice Inviting Tender) estimated cost in INR from the
    commercial_terms payload list.

    Looks for:
      1. term_type == "nit_estimated_cost"  (new pattern, future runs)
      2. Scans snippet text of all terms for "Estimated Cost X Crore/Lakh" patterns
         (works on existing stored runs without re-running the extractor)

    Returns the value in INR (converts crore/lakh), or None if not found.
    """
    if not commercial_terms:
        return None

    _UNIT_MAP = {
        "crore": _CRORE, "lakh": _LAKH, "lac": _LAKH,
        "million": 1_000_000, "billion": 1_000_000_000,
    }

    # 1. Look for the dedicated term_type first
    for ct in commercial_terms:
        if ct.get("term_type") == "nit_estimated_cost":
            val = _safe_float(ct.get("value"))
            if val and val > 0:
                unit = (ct.get("unit") or ct.get("cadence") or "").lower()
                multiplier = _UNIT_MAP.get(unit, 1.0)
                return round(val * multiplier, 2)

    # 2. Scan snippets of all terms for "Estimated Cost X Crore" patterns
    for ct in commercial_terms:
        snippet = ct.get("snippet") or ""
        m = _NIT_SNIPPET_RE.search(snippet)
        if m:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue
            unit = (m.group(2) or "").lower()
            multiplier = _UNIT_MAP.get(unit, 1.0)
            result = round(val * multiplier, 2)
            if result > 100_000:  # sanity: must be > ₹1 lakh to be plausible
                return result

    return None


def _safe_float(value) -> Optional[float]:
    """Return float if value is a valid non-zero number, else None."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def compute_bid_margin(
    line_items: list,
    boq_items: list = None,
    nit_value_inr: Optional[float] = None,
) -> BidMarginResult:
    """
    Compute the bid margin by comparing tender BOQ rates against market rates.

    line_items: list of dicts with keys:
        trade         (str)   — trade category
        description   (str)   — item description
        unit          (str)   — unit of measurement
        qty           (float) — quantity
        rate_inr      (float) — market/DSR rate applied by rate engine
        amount_inr    (float) — market cost (rate_inr × qty)
        rate          (float) — extracted from PDF BOQ (may be None/0)
        extracted_rate (float) — alternate key for PDF-extracted rate
        boq_rate      (float) — alternate key for PDF-extracted rate

    boq_items: raw BOQ extraction output (optional — used as fallback
        for rate lookup by description matching).

    nit_value_inr: NIT stated "Estimated Cost Put to Tender" in INR.
        Used as the tender value when per-item rate coverage is below 20%.
        Obtain via extract_nit_value(payload["commercial_terms"]).

    Returns BidMarginResult with tender_value and contractor_cost by trade.
    When nit_value_inr is provided and per-item coverage is low, it is used
    as the tender_value and reliable is set to True.
    """
    if not line_items:
        return BidMarginResult(
            tender_value_inr=0.0,
            contractor_cost_inr=0.0,
            margin_inr=0.0,
            margin_pct=0.0,
            by_trade=[],
            coverage_pct=0.0,
            note="No line items provided",
            reliable=False,
        )

    # Build a description → extracted_rate lookup from boq_items for fallback
    _boq_rate_by_desc: Dict[str, float] = {}
    for _bi in (boq_items or []):
        _bd = (_bi.get("description") or "").strip().lower()
        _br = _safe_float(
            _bi.get("rate") or _bi.get("extracted_rate") or _bi.get("boq_rate")
        )
        if _bd and _br:
            _boq_rate_by_desc[_bd] = _br

    # Per-trade accumulators
    _by_trade_data: Dict[str, dict] = {}

    for item in line_items:
        trade = item.get("trade") or "general"
        if trade not in _by_trade_data:
            _by_trade_data[trade] = {
                "tender_value_inr": 0.0,
                "contractor_cost_inr": 0.0,
                "item_count": 0,
                "items_with_tender_rate": 0,
            }

        td = _by_trade_data[trade]
        td["item_count"] += 1

        qty = float(item.get("qty") or 0.0)

        # Contractor cost from market rate engine
        contractor_amount = float(item.get("amount_inr") or 0.0)
        if contractor_amount == 0.0 and qty > 0:
            _rate_inr = _safe_float(item.get("rate_inr"))
            if _rate_inr:
                contractor_amount = _rate_inr * qty
        td["contractor_cost_inr"] += contractor_amount

        # Tender value from PDF-extracted rate
        # Check all possible keys: rate, extracted_rate, boq_rate
        extracted_rate = _safe_float(
            item.get("rate") or item.get("extracted_rate") or item.get("boq_rate")
        )

        # Fallback: look up by description in boq_items
        if extracted_rate is None and _boq_rate_by_desc:
            _desc_key = (item.get("description") or "").strip().lower()
            if _desc_key:
                extracted_rate = _boq_rate_by_desc.get(_desc_key)

        if extracted_rate is not None and qty > 0:
            td["tender_value_inr"] += extracted_rate * qty
            td["items_with_tender_rate"] += 1

    # Build trade-level TradeMargin objects
    by_trade: List[TradeMargin] = []
    total_tender = 0.0
    total_cost = 0.0
    total_items = 0
    total_with_rate = 0

    for trade_name, td in _by_trade_data.items():
        t_val = td["tender_value_inr"]
        c_val = td["contractor_cost_inr"]
        n_items = td["item_count"]
        n_rated = td["items_with_tender_rate"]

        margin_inr = t_val - c_val
        margin_pct = (margin_inr / t_val * 100.0) if t_val > 0 else 0.0
        cov_pct = (n_rated / n_items * 100.0) if n_items > 0 else 0.0

        by_trade.append(TradeMargin(
            trade=trade_name,
            tender_value_inr=round(t_val, 2),
            contractor_cost_inr=round(c_val, 2),
            margin_inr=round(margin_inr, 2),
            margin_pct=round(margin_pct, 2),
            item_count=n_items,
            items_with_tender_rate=n_rated,
            coverage_pct=round(cov_pct, 2),
        ))

        total_tender += t_val
        total_cost += c_val
        total_items += n_items
        total_with_rate += n_rated

    # Sort by tender value descending
    by_trade.sort(key=lambda t: t.tender_value_inr, reverse=True)

    overall_margin_inr = total_tender - total_cost
    overall_margin_pct = (
        (overall_margin_inr / total_tender * 100.0) if total_tender > 0 else 0.0
    )
    overall_coverage = (
        (total_with_rate / total_items * 100.0) if total_items > 0 else 0.0
    )

    # If per-item coverage is low but NIT stated cost is available, use it as tender value
    # Note: total_cost is still the per-item market rate sum
    _used_nit_value = False
    if overall_coverage < 50 and nit_value_inr and nit_value_inr > 0:
        total_tender = nit_value_inr
        overall_margin_inr = total_tender - total_cost
        overall_margin_pct = (
            (overall_margin_inr / total_tender * 100.0) if total_tender > 0 else 0.0
        )
        _used_nit_value = True

    # Scope coverage: what % of the NIT stated value has the QTO covered
    _scope_coverage_pct = 0.0
    if nit_value_inr and nit_value_inr > 0 and total_cost > 0:
        _scope_coverage_pct = round(min(total_cost / nit_value_inr * 100.0, 100.0), 1)

    # Reliability and note
    if _used_nit_value:
        reliable = True
        _nit_cr = total_tender / _CRORE
        note = (
            f"Tender value from NIT estimated cost (₹{_nit_cr:.1f} Cr stated on cover page). "
            f"Contractor cost from market/DSR rates × extracted quantities "
            f"({_scope_coverage_pct:.1f}% of scope captured — run Full Audit for complete estimate)."
        )
    elif overall_coverage < 20:
        reliable = False
        note = (
            f"Low BOQ rate coverage (<20%) — tender value estimate unreliable "
            f"({total_with_rate}/{total_items} items had extracted rates). "
            f"Upload NIT document with 'Estimated Cost' to enable bid margin analysis."
        )
    elif overall_coverage >= 50:
        reliable = True
        note = (
            f"Based on {overall_coverage:.0f}% BOQ coverage "
            f"({total_with_rate}/{total_items} items had extracted rates)"
        )
    else:
        reliable = False
        note = (
            f"Partial BOQ rate coverage ({overall_coverage:.0f}%) — "
            f"{total_with_rate}/{total_items} items had extracted rates; "
            f"tender value may be understated"
        )

    logger.debug(
        "bid_margin: tender=₹%.0f cost=₹%.0f margin=%.1f%% coverage=%.0f%% (%d/%d items)",
        total_tender, total_cost, overall_margin_pct, overall_coverage,
        total_with_rate, total_items,
    )

    return BidMarginResult(
        tender_value_inr=round(total_tender, 2),
        contractor_cost_inr=round(total_cost, 2),
        margin_inr=round(overall_margin_inr, 2),
        margin_pct=round(overall_margin_pct, 2),
        by_trade=by_trade,
        coverage_pct=round(overall_coverage, 2),
        note=note,
        reliable=reliable,
        scope_coverage_pct=_scope_coverage_pct,
    )
