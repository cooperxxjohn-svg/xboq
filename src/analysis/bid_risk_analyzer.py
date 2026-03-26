"""
Bid Risk Analyzer — Comprehensive risk assessment for construction tender bids.

Integrates signals from multiple analysis modules:
1. Scope completeness (missing items = cost overrun risk)
2. Quantity variance (mismatches = re-measurement risk)
3. Pricing coverage (missing rates = unpriced scope)
4. Document completeness (missing drawings/specs = ambiguity risk)
5. Market conditions (escalation risk for long-duration projects)
6. Contractual risk signals (payment terms, defect liability, etc.)
7. Site conditions (logistics, access, seasonal disruptions)

Produces a risk score and actionable recommendations for bidders.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskCategory(Enum):
    SCOPE = "scope"
    QUANTITY = "quantity"
    PRICING = "pricing"
    DOCUMENT = "document"
    MARKET = "market"
    CONTRACTUAL = "contractual"
    SITE = "site"


class RiskLevel(Enum):
    CRITICAL = "critical"   # Show-stopper risks
    HIGH = "high"           # Must address before bidding
    MEDIUM = "medium"       # Address with qualifications
    LOW = "low"             # Acceptable risk
    NONE = "none"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskItem:
    """Single identified risk."""
    category: RiskCategory
    level: RiskLevel
    title: str
    description: str
    financial_impact: str = ""      # e.g., "5-10 lakhs" or "2-5% of bid value"
    mitigation: str = ""
    source: str = ""                # Which analysis module identified this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "level": self.level.value,
            "title": self.title,
            "description": self.description,
            "financial_impact": self.financial_impact,
            "mitigation": self.mitigation,
            "source": self.source,
        }


@dataclass
class BidRiskReport:
    """Complete bid risk assessment."""
    risks: List[RiskItem] = field(default_factory=list)
    overall_risk_score: float = 0.0  # 0-100, higher = more risky
    bid_recommendation: str = ""     # GO | GO_WITH_QUALIFICATIONS | NO_GO | NEEDS_REVIEW

    @property
    def critical_risks(self) -> List[RiskItem]:
        return [r for r in self.risks if r.level == RiskLevel.CRITICAL]

    @property
    def high_risks(self) -> List[RiskItem]:
        return [r for r in self.risks if r.level == RiskLevel.HIGH]

    @property
    def risk_by_category(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.risks:
            cat = r.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk_score": self.overall_risk_score,
            "bid_recommendation": self.bid_recommendation,
            "total_risks": len(self.risks),
            "critical": len(self.critical_risks),
            "high": len(self.high_risks),
            "risk_by_category": self.risk_by_category,
            "risks": [r.to_dict() for r in self.risks],
        }

    def summary(self) -> str:
        return (
            f"Bid Risk: score={self.overall_risk_score:.0f}/100, "
            f"recommendation={self.bid_recommendation}, "
            f"{len(self.critical_risks)} critical, {len(self.high_risks)} high risks"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_DOCUMENTS = [
    "boq",
    "architectural_drawings",
    "structural_drawings",
    "specifications",
    "schedule_of_rates",
    "conditions_of_contract",
]

# Structural BOQ item keywords (case-insensitive matching)
_STRUCTURAL_KEYWORDS = frozenset({
    "rcc", "reinforcement", "concrete", "footing", "foundation",
    "beam", "column", "slab", "pile", "raft", "retaining wall",
    "structural steel", "shuttering", "formwork",
})

# MEP BOQ item keywords (case-insensitive matching)
_MEP_KEYWORDS = frozenset({
    "electrical", "plumbing", "hvac", "fire", "firefighting",
    "sanitary", "drainage", "wiring", "conduit", "switchgear",
    "air conditioning", "ventilation", "pump", "transformer",
    "sprinkler", "cable", "panel", "db", "mep",
})

# Remote / hill-station locations (representative list for India)
_REMOTE_LOCATIONS = frozenset({
    "leh", "ladakh", "kargil", "spiti", "lahaul", "kinnaur",
    "tawang", "itanagar", "kohima", "imphal", "aizawl", "agartala",
    "shillong", "gangtok", "dimapur", "silchar",
    "port blair", "andaman", "nicobar", "lakshadweep", "kavaratti",
    "minicoy",
})

_HILL_STATION_LOCATIONS = frozenset({
    "shimla", "manali", "mussoorie", "nainital", "darjeeling",
    "ooty", "kodaikanal", "munnar", "coorg", "mount abu",
    "pahalgam", "gulmarg", "srinagar", "dalhousie", "dharamshala",
    "mcleodganj", "almora", "ranikhet", "lansdowne",
})

_FLOOD_PRONE_LOCATIONS = frozenset({
    "patna", "guwahati", "kolkata", "chennai", "mumbai",
    "kochi", "allahabad", "prayagraj", "varanasi", "lucknow",
    "dhubri", "jorhat", "dibrugarh", "barpeta", "silchar",
    "sambalpur", "cuttack", "bhagalpur",
})

_URBAN_CONGESTED_LOCATIONS = frozenset({
    "mumbai", "delhi", "kolkata", "chennai", "bangalore",
    "bengaluru", "hyderabad", "ahmedabad", "pune", "jaipur",
    "lucknow", "old delhi", "chandni chowk",
})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_bid_risk(
    boq_items: Optional[List[Dict[str, Any]]] = None,
    scope_gaps: int = 0,
    quantity_mismatches: int = 0,
    missing_rates_pct: float = 0.0,
    completeness_score: float = 100.0,
    project_duration_months: int = 12,
    project_value_lakhs: float = 100.0,
    document_types_available: Optional[List[str]] = None,
    contract_conditions: Optional[Dict[str, Any]] = None,
    location: str = "Delhi",
) -> BidRiskReport:
    """
    Analyze bid risk from multiple signals.

    Parameters
    ----------
    boq_items : list of dict
        Parsed BOQ line items. Each dict should have at least a
        ``description`` key and optionally ``amount``, ``rate``, ``quantity``.
    scope_gaps : int
        Number of scope items identified as missing by the scope matrix.
    quantity_mismatches : int
        Number of items where take-off quantity differs materially from
        the BOQ quantity.
    missing_rates_pct : float
        Percentage (0-100) of line items that have no rate / unit price.
    completeness_score : float
        BOQ completeness score (0-100) from the QA module.
    project_duration_months : int
        Expected project duration in months.
    project_value_lakhs : float
        Estimated project value in lakhs (INR).
    document_types_available : list of str
        List of document type slugs that are present in the tender set.
    contract_conditions : dict
        Key contractual parameters extracted from conditions of contract.
    location : str
        Project site location name.

    Returns
    -------
    BidRiskReport
    """
    report = BidRiskReport()
    boq_items = boq_items or []
    document_types_available = document_types_available or []
    contract_conditions = contract_conditions or {}

    # 1. Scope risks
    report.risks.extend(
        _assess_scope_risks(scope_gaps, completeness_score, boq_items)
    )

    # 2. Quantity risks
    report.risks.extend(
        _assess_quantity_risks(quantity_mismatches, len(boq_items))
    )

    # 3. Pricing risks
    report.risks.extend(
        _assess_pricing_risks(missing_rates_pct, project_value_lakhs)
    )

    # 4. Document risks
    report.risks.extend(
        _assess_document_risks(document_types_available)
    )

    # 5. Market / escalation risks
    report.risks.extend(
        _assess_market_risks(
            project_duration_months, project_value_lakhs, location, boq_items,
        )
    )

    # 6. Contractual risks
    report.risks.extend(
        _assess_contractual_risks(contract_conditions, project_duration_months)
    )

    # 7. Site risks
    report.risks.extend(
        _assess_site_risks(location, project_value_lakhs)
    )

    # Calculate overall risk score
    report.overall_risk_score = _calculate_risk_score(report.risks)

    # Determine recommendation
    report.bid_recommendation = _determine_recommendation(report)

    logger.info(report.summary())
    return report


# ---------------------------------------------------------------------------
# 1. Scope risk assessment
# ---------------------------------------------------------------------------

def _assess_scope_risks(
    scope_gaps: int,
    completeness_score: float,
    boq_items: List[Dict[str, Any]],
) -> List[RiskItem]:
    """Identify risks from missing scope and incomplete BOQ."""
    risks: List[RiskItem] = []

    # --- Gap count thresholds ---
    if scope_gaps > 10:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.CRITICAL,
            title="Significant missing scope items",
            description=(
                f"{scope_gaps} scope items identified as missing from the BOQ. "
                "This level of omission indicates serious cost-overrun risk."
            ),
            financial_impact="10-25% of bid value at risk from uncosted scope",
            mitigation=(
                "Request clarification from the client or issue an RFI. "
                "Include provisional sums for missing work packages."
            ),
            source="scope_matrix",
        ))
    elif scope_gaps > 5:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.HIGH,
            title="Multiple missing scope items",
            description=(
                f"{scope_gaps} scope items appear missing from the BOQ. "
                "These gaps may lead to unpriced work during execution."
            ),
            financial_impact="5-10% of bid value at risk",
            mitigation=(
                "Raise RFIs for the missing items. Consider adding "
                "qualifications or provisional sums in the bid."
            ),
            source="scope_matrix",
        ))
    elif scope_gaps > 0:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.MEDIUM,
            title="Minor scope gaps detected",
            description=(
                f"{scope_gaps} scope item(s) may be missing from the BOQ."
            ),
            financial_impact="1-3% of bid value",
            mitigation="Verify against drawings and raise RFIs if needed.",
            source="scope_matrix",
        ))

    # --- Completeness score thresholds ---
    if completeness_score < 50:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.CRITICAL,
            title="BOQ completeness below 50%",
            description=(
                f"BOQ completeness score is only {completeness_score:.0f}%. "
                "More than half the expected items are missing or incomplete."
            ),
            financial_impact="Potentially 20-40% cost overrun",
            mitigation=(
                "Do not bid without requesting a revised or supplementary BOQ. "
                "Consider a NO-GO if the client cannot provide clarification."
            ),
            source="qa_score",
        ))
    elif completeness_score < 70:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.HIGH,
            title="Low BOQ completeness",
            description=(
                f"BOQ completeness score is {completeness_score:.0f}%. "
                "Significant items may be missing."
            ),
            financial_impact="10-20% cost overrun risk",
            mitigation=(
                "Request a revised BOQ or add qualifications listing "
                "items assumed to be excluded."
            ),
            source="qa_score",
        ))
    elif completeness_score < 85:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.MEDIUM,
            title="Moderate BOQ completeness",
            description=(
                f"BOQ completeness score is {completeness_score:.0f}%. "
                "Some items may need verification."
            ),
            financial_impact="3-8% cost overrun risk",
            mitigation="Cross-check BOQ against drawings and specifications.",
            source="qa_score",
        ))

    # --- Structural items check ---
    has_structural = _boq_contains_category(boq_items, _STRUCTURAL_KEYWORDS)
    if boq_items and not has_structural:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.CRITICAL,
            title="No structural work items in BOQ",
            description=(
                "The BOQ does not appear to contain any structural work items "
                "(RCC, reinforcement, foundations, beams, columns, etc.). "
                "For a building project this is a critical omission."
            ),
            financial_impact="Structural work typically represents 25-35% of project cost",
            mitigation=(
                "Confirm with the client whether structural work is in a "
                "separate contract or has been inadvertently omitted."
            ),
            source="scope_matrix",
        ))

    # --- MEP items check ---
    has_mep = _boq_contains_category(boq_items, _MEP_KEYWORDS)
    if boq_items and not has_mep:
        risks.append(RiskItem(
            category=RiskCategory.SCOPE,
            level=RiskLevel.HIGH,
            title="No MEP items in BOQ",
            description=(
                "The BOQ does not contain any MEP (Mechanical, Electrical, "
                "Plumbing) items. These are usually 15-25% of building cost."
            ),
            financial_impact="15-25% of project value if MEP is expected in scope",
            mitigation=(
                "Confirm whether MEP is a separate package or should be "
                "included. Add qualifications if unclear."
            ),
            source="scope_matrix",
        ))

    return risks


# ---------------------------------------------------------------------------
# 2. Quantity risk assessment
# ---------------------------------------------------------------------------

def _assess_quantity_risks(
    quantity_mismatches: int,
    total_items: int,
) -> List[RiskItem]:
    """Identify risks from quantity discrepancies between BOQ and take-off."""
    risks: List[RiskItem] = []

    if total_items == 0:
        return risks

    mismatch_pct = (quantity_mismatches / total_items) * 100

    if mismatch_pct > 20:
        risks.append(RiskItem(
            category=RiskCategory.QUANTITY,
            level=RiskLevel.CRITICAL,
            title="Severe quantity mismatches",
            description=(
                f"{quantity_mismatches} of {total_items} items "
                f"({mismatch_pct:.0f}%) have material quantity discrepancies "
                "between the BOQ and independent take-off. "
                "Re-measurement during execution will likely cause disputes."
            ),
            financial_impact="10-20% cost variance due to re-measurement",
            mitigation=(
                "Perform a detailed quantity check for all major items. "
                "Consider bidding with a qualification on quantities."
            ),
            source="quantity_crosscheck",
        ))
    elif mismatch_pct > 10:
        risks.append(RiskItem(
            category=RiskCategory.QUANTITY,
            level=RiskLevel.HIGH,
            title="Significant quantity mismatches",
            description=(
                f"{quantity_mismatches} of {total_items} items "
                f"({mismatch_pct:.0f}%) show quantity discrepancies. "
                "Re-measurement risk is elevated."
            ),
            financial_impact="5-10% cost variance possible",
            mitigation=(
                "Verify quantities for high-value items. Flag mismatches "
                "in the bid submission."
            ),
            source="quantity_crosscheck",
        ))
    elif mismatch_pct > 5:
        risks.append(RiskItem(
            category=RiskCategory.QUANTITY,
            level=RiskLevel.MEDIUM,
            title="Moderate quantity discrepancies",
            description=(
                f"{quantity_mismatches} of {total_items} items "
                f"({mismatch_pct:.0f}%) show quantity differences."
            ),
            financial_impact="2-5% cost variance possible",
            mitigation="Spot-check high-value items before bid submission.",
            source="quantity_crosscheck",
        ))
    elif mismatch_pct > 0:
        risks.append(RiskItem(
            category=RiskCategory.QUANTITY,
            level=RiskLevel.LOW,
            title="Minor quantity differences",
            description=(
                f"{quantity_mismatches} of {total_items} items show "
                "small quantity differences. Within normal tolerance."
            ),
            financial_impact="Less than 2% variance",
            mitigation="No immediate action needed. Monitor during execution.",
            source="quantity_crosscheck",
        ))

    return risks


# ---------------------------------------------------------------------------
# 3. Pricing risk assessment
# ---------------------------------------------------------------------------

def _assess_pricing_risks(
    missing_rates_pct: float,
    project_value_lakhs: float,
) -> List[RiskItem]:
    """Identify risks from unpriced or partially priced BOQ items."""
    risks: List[RiskItem] = []

    if missing_rates_pct > 30:
        risks.append(RiskItem(
            category=RiskCategory.PRICING,
            level=RiskLevel.CRITICAL,
            title="Over 30% items unpriced",
            description=(
                f"{missing_rates_pct:.0f}% of BOQ line items do not have "
                "a unit rate or amount. The bid cannot be accurately costed."
            ),
            financial_impact=(
                "Impossible to determine total bid value reliably; "
                "risk of significant under- or over-pricing"
            ),
            mitigation=(
                "Obtain rate analysis or quotations for all unpriced items "
                "before submission. Consider requesting an extension."
            ),
            source="pricing_guidance",
        ))
    elif missing_rates_pct > 15:
        risks.append(RiskItem(
            category=RiskCategory.PRICING,
            level=RiskLevel.HIGH,
            title="Significant unpriced items",
            description=(
                f"{missing_rates_pct:.0f}% of BOQ line items lack rates. "
                "Bid accuracy is compromised."
            ),
            financial_impact="5-15% pricing uncertainty",
            mitigation=(
                "Prioritize rate analysis for high-value unpriced items. "
                "Use schedule of rates as fallback where available."
            ),
            source="pricing_guidance",
        ))
    elif missing_rates_pct > 5:
        risks.append(RiskItem(
            category=RiskCategory.PRICING,
            level=RiskLevel.MEDIUM,
            title="Some items lack pricing",
            description=(
                f"{missing_rates_pct:.0f}% of items are unpriced."
            ),
            financial_impact="2-5% pricing uncertainty",
            mitigation="Complete rate analysis for remaining items.",
            source="pricing_guidance",
        ))

    # High-value project with any missing rates = escalation concern
    if project_value_lakhs > 500 and missing_rates_pct > 0:
        risks.append(RiskItem(
            category=RiskCategory.PRICING,
            level=RiskLevel.HIGH,
            title="Unpriced items on high-value project",
            description=(
                f"Project value is {project_value_lakhs:.0f} lakhs and "
                f"{missing_rates_pct:.1f}% of items remain unpriced. "
                "Even small pricing gaps have large absolute impact."
            ),
            financial_impact=(
                f"At {project_value_lakhs:.0f} lakhs, even 1% error = "
                f"{project_value_lakhs * 0.01:.1f} lakhs"
            ),
            mitigation=(
                "Ensure 100% pricing coverage. Obtain competitive "
                "subcontractor quotes for specialist items."
            ),
            source="pricing_guidance",
        ))

    return risks


# ---------------------------------------------------------------------------
# 4. Document risk assessment
# ---------------------------------------------------------------------------

def _assess_document_risks(
    document_types_available: List[str],
) -> List[RiskItem]:
    """Identify risks from missing tender documents."""
    risks: List[RiskItem] = []
    available_lower = {d.lower().strip() for d in document_types_available}

    _doc_risk_map: List[Tuple[str, str, RiskLevel, str, str]] = [
        (
            "boq",
            "Bill of Quantities (BOQ) not provided",
            RiskLevel.CRITICAL,
            "Cannot price the bid without a BOQ",
            "Request the BOQ from the client immediately. Do not bid without it.",
        ),
        (
            "architectural_drawings",
            "Architectural drawings missing",
            RiskLevel.HIGH,
            "Scope ambiguity — cannot verify areas, layouts, or finishes",
            (
                "Request architectural drawings. Qualify bid stating "
                "assumptions on areas and finishes."
            ),
        ),
        (
            "structural_drawings",
            "Structural drawings missing",
            RiskLevel.HIGH,
            "Cannot verify structural quantities (RCC, steel, foundations)",
            (
                "Request structural drawings. Add risk premium or "
                "provisional sums for structural work."
            ),
        ),
        (
            "specifications",
            "Technical specifications missing",
            RiskLevel.MEDIUM,
            "Material grades, brands, and workmanship standards unclear",
            (
                "Bid based on standard specifications (IS codes). "
                "Qualify any assumptions on material quality."
            ),
        ),
        (
            "schedule_of_rates",
            "Schedule of rates not provided",
            RiskLevel.MEDIUM,
            "No benchmark rates available for deviation items",
            (
                "Use CPWD/state SOR as reference. Note the assumed "
                "SOR in bid qualifications."
            ),
        ),
        (
            "conditions_of_contract",
            "Conditions of contract missing",
            RiskLevel.MEDIUM,
            "Payment terms, variations, and risk allocation unknown",
            (
                "Assume standard CPWD/FIDIC conditions. Qualify the "
                "bid listing assumed contract terms."
            ),
        ),
    ]

    for doc_key, title, level, impact, mitigation in _doc_risk_map:
        if doc_key not in available_lower:
            risks.append(RiskItem(
                category=RiskCategory.DOCUMENT,
                level=level,
                title=title,
                description=(
                    f"The tender set is missing '{doc_key.replace('_', ' ')}'. "
                    "This creates ambiguity and pricing risk."
                ),
                financial_impact=impact,
                mitigation=mitigation,
                source="document_completeness",
            ))

    return risks


# ---------------------------------------------------------------------------
# 5. Market / escalation risk assessment
# ---------------------------------------------------------------------------

def _assess_market_risks(
    project_duration_months: int,
    project_value_lakhs: float,
    location: str,
    boq_items: List[Dict[str, Any]],
) -> List[RiskItem]:
    """Identify risks from market conditions, escalation, and logistics."""
    risks: List[RiskItem] = []

    # --- Duration-based escalation risk ---
    if project_duration_months > 24:
        risks.append(RiskItem(
            category=RiskCategory.MARKET,
            level=RiskLevel.HIGH,
            title="High escalation risk — long duration project",
            description=(
                f"Project duration of {project_duration_months} months "
                "exceeds 24 months. Material costs (cement, steel, aggregates) "
                "can increase 15-20% over this period."
            ),
            financial_impact="15-20% increase in material costs over project life",
            mitigation=(
                "Ensure escalation clause is included in the contract. "
                "Build escalation contingency of 8-12% into the bid. "
                "Lock in bulk material procurement rates early."
            ),
            source="market_analysis",
        ))
    elif project_duration_months > 18:
        risks.append(RiskItem(
            category=RiskCategory.MARKET,
            level=RiskLevel.MEDIUM,
            title="Moderate escalation risk",
            description=(
                f"Project duration of {project_duration_months} months "
                "creates moderate material price escalation risk."
            ),
            financial_impact="8-12% increase in material costs possible",
            mitigation=(
                "Include escalation clause. Build 5-8% contingency "
                "for material price variations."
            ),
            source="market_analysis",
        ))
    elif project_duration_months > 12:
        risks.append(RiskItem(
            category=RiskCategory.MARKET,
            level=RiskLevel.LOW,
            title="Minor escalation exposure",
            description=(
                f"Project duration of {project_duration_months} months "
                "has limited but non-zero escalation exposure."
            ),
            financial_impact="3-5% material cost variation possible",
            mitigation="Standard escalation clause should suffice.",
            source="market_analysis",
        ))

    # --- Steel-heavy project check ---
    steel_fraction = _estimate_steel_fraction(boq_items, project_value_lakhs)
    if steel_fraction > 0.30:
        extra_level = (
            RiskLevel.HIGH
            if project_duration_months > 18
            else RiskLevel.MEDIUM
        )
        risks.append(RiskItem(
            category=RiskCategory.MARKET,
            level=extra_level,
            title="Steel-heavy project — price volatility risk",
            description=(
                f"Estimated steel component is {steel_fraction * 100:.0f}% "
                "of project value. Steel prices are highly volatile in "
                "the Indian market (10-30% swings within a year)."
            ),
            financial_impact=(
                f"{steel_fraction * 100:.0f}% of value exposed to "
                "steel price fluctuations"
            ),
            mitigation=(
                "Negotiate a steel price variation clause. "
                "Consider advance procurement of reinforcement steel. "
                "Get fixed-price quotes from steel suppliers."
            ),
            source="market_analysis",
        ))

    # --- Remote location logistics ---
    loc_lower = location.lower().strip()
    if _is_location_in_set(loc_lower, _REMOTE_LOCATIONS):
        risks.append(RiskItem(
            category=RiskCategory.MARKET,
            level=RiskLevel.HIGH,
            title="Remote location — high logistics cost",
            description=(
                f"Project location '{location}' is classified as remote. "
                "Material transport costs, labour availability, and "
                "working-season limitations will significantly impact cost."
            ),
            financial_impact="20-40% logistics premium on material costs",
            mitigation=(
                "Add logistics premium to all material rates. "
                "Plan for seasonal road closures and limited working months. "
                "Identify local material sources."
            ),
            source="market_analysis",
        ))

    return risks


# ---------------------------------------------------------------------------
# 6. Contractual risk assessment
# ---------------------------------------------------------------------------

def _assess_contractual_risks(
    contract_conditions: Dict[str, Any],
    project_duration_months: int = 12,
) -> List[RiskItem]:
    """Identify risks from unfavourable contract conditions."""
    risks: List[RiskItem] = []

    if not contract_conditions:
        return risks

    # --- Defect liability period ---
    dlp_years = contract_conditions.get("defect_liability_years", 0)
    if dlp_years > 5:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.HIGH,
            title="Extended defect liability period",
            description=(
                f"Defect liability period is {dlp_years} years, "
                "exceeding the standard 1-2 years. This increases "
                "long-term warranty exposure and bank guarantee costs."
            ),
            financial_impact=(
                f"Bank guarantee cost for {dlp_years} years; "
                "potential rectification costs over extended period"
            ),
            mitigation=(
                "Price in extended bank guarantee charges. "
                "Negotiate to reduce DLP to standard 1-2 years. "
                "Ensure adequate defects liability insurance."
            ),
            source="contract_analysis",
        ))
    elif dlp_years > 3:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.MEDIUM,
            title="Above-average defect liability period",
            description=(
                f"Defect liability period of {dlp_years} years "
                "is above the typical 1-2 year standard."
            ),
            financial_impact="Additional bank guarantee and insurance costs",
            mitigation="Include extended BG costs in overheads.",
            source="contract_analysis",
        ))

    # --- Retention percentage ---
    retention_pct = contract_conditions.get("retention_pct", 0)
    if retention_pct > 10:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.MEDIUM,
            title="High retention percentage",
            description=(
                f"Retention is {retention_pct}% against the standard 5%. "
                "This reduces cash flow during execution."
            ),
            financial_impact=(
                f"{retention_pct}% of certified value locked; "
                "increased working capital requirement"
            ),
            mitigation=(
                "Factor in the cost of additional working capital. "
                "Negotiate retention reduction after 50% completion."
            ),
            source="contract_analysis",
        ))

    # --- Payment terms ---
    payment_days = contract_conditions.get("payment_terms_days", 0)
    if payment_days > 90:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.HIGH,
            title="Extended payment terms — cash flow risk",
            description=(
                f"Payment terms are {payment_days} days from certification. "
                "This creates severe cash flow strain, especially for "
                "material-intensive phases."
            ),
            financial_impact=(
                f"Working capital locked for {payment_days} days per bill; "
                "interest cost on bridging finance"
            ),
            mitigation=(
                "Add interest/financing cost to overheads. "
                "Negotiate mobilization advance or interim payments. "
                "Secure adequate credit facilities."
            ),
            source="contract_analysis",
        ))
    elif payment_days > 60:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.MEDIUM,
            title="Slow payment cycle",
            description=(
                f"Payment terms are {payment_days} days. "
                "Standard government terms are 30-45 days."
            ),
            financial_impact="Increased working capital cost",
            mitigation="Factor financing cost into bid overheads.",
            source="contract_analysis",
        ))

    # --- No escalation clause ---
    no_escalation = contract_conditions.get("no_escalation_clause", False)
    if no_escalation and project_duration_months > 12:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.CRITICAL,
            title="No escalation clause on long-duration project",
            description=(
                f"The contract has no price escalation clause for a "
                f"{project_duration_months}-month project. All material "
                "and labour cost increases will be borne by the contractor."
            ),
            financial_impact=(
                "Potential 10-25% unrecoverable cost increase "
                "over project duration"
            ),
            mitigation=(
                "Strongly negotiate for an escalation clause. If refused, "
                "build significant contingency (10-15%) into the bid. "
                "Consider NO-GO if duration exceeds 24 months."
            ),
            source="contract_analysis",
        ))
    elif no_escalation:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.MEDIUM,
            title="No escalation clause",
            description=(
                "The contract does not include a price escalation clause. "
                "For a short-duration project this is manageable."
            ),
            financial_impact="3-5% risk from material price changes",
            mitigation="Add contingency of 3-5% for material variations.",
            source="contract_analysis",
        ))

    # --- Liquidated damages ---
    ld_pct = contract_conditions.get("liquidated_damages_pct", 0)
    if ld_pct > 1:
        level = RiskLevel.HIGH if ld_pct > 2 else RiskLevel.MEDIUM
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=level,
            title="High liquidated damages",
            description=(
                f"Liquidated damages are {ld_pct}% per week/month of delay "
                f"(standard is 0.5-1%). "
                "{'Excessively punitive — ' if ld_pct > 2 else ''}"
                "This amplifies schedule risk."
            ),
            financial_impact=(
                f"LD exposure of {ld_pct}% per period of delay, "
                "potentially capped at 5-10% of contract value"
            ),
            mitigation=(
                "Build schedule buffers. Negotiate LD cap and rate. "
                "Ensure force majeure and extension of time clauses are robust."
            ),
            source="contract_analysis",
        ))

    # --- No variation clause ---
    no_variation = contract_conditions.get("no_variation_clause", False)
    if no_variation:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.HIGH,
            title="No variation clause — scope creep risk",
            description=(
                "The contract does not include a variation/change order "
                "mechanism. Any additional work requested by the client "
                "may not be compensated."
            ),
            financial_impact="Unquantifiable — depends on scope changes during execution",
            mitigation=(
                "Negotiate inclusion of a variation clause. "
                "If refused, bid strictly to scope and document all "
                "client instructions in writing."
            ),
            source="contract_analysis",
        ))

    # --- Performance guarantee ---
    pg_pct = contract_conditions.get("performance_guarantee_pct", 0)
    if pg_pct > 10:
        risks.append(RiskItem(
            category=RiskCategory.CONTRACTUAL,
            level=RiskLevel.MEDIUM,
            title="High performance guarantee requirement",
            description=(
                f"Performance guarantee is {pg_pct}% of contract value "
                "(standard is 5-10%). This ties up additional bank "
                "guarantee limits."
            ),
            financial_impact=f"BG charges on {pg_pct}% of contract value",
            mitigation=(
                "Include additional BG charges in overheads. "
                "Negotiate reduction to 5% or provision of insurance bond."
            ),
            source="contract_analysis",
        ))

    return risks


# ---------------------------------------------------------------------------
# 7. Site risk assessment
# ---------------------------------------------------------------------------

def _assess_site_risks(
    location: str,
    project_value_lakhs: float,
) -> List[RiskItem]:
    """Identify risks from site location and conditions."""
    risks: List[RiskItem] = []
    loc_lower = location.lower().strip()

    # --- Hill station / remote ---
    if _is_location_in_set(loc_lower, _HILL_STATION_LOCATIONS):
        risks.append(RiskItem(
            category=RiskCategory.SITE,
            level=RiskLevel.MEDIUM,
            title="Hill station location — logistics premium",
            description=(
                f"Project is located at '{location}', a hill station. "
                "Material transport is costlier due to winding roads, "
                "load restrictions, and seasonal access limitations."
            ),
            financial_impact="10-20% premium on material transport costs",
            mitigation=(
                "Add hill transport premium to material rates. "
                "Plan stockpiling before monsoon/winter closures. "
                "Identify local aggregate and sand sources."
            ),
            source="site_analysis",
        ))

    # --- Flood-prone area ---
    if _is_location_in_set(loc_lower, _FLOOD_PRONE_LOCATIONS):
        risks.append(RiskItem(
            category=RiskCategory.SITE,
            level=RiskLevel.MEDIUM,
            title="Flood-prone area — seasonal disruption risk",
            description=(
                f"'{location}' is in a flood-prone region. "
                "Construction may be disrupted during monsoon months "
                "(June-September), causing schedule delays."
            ),
            financial_impact=(
                "2-4 months of potential disruption per year; "
                "dewatering and protection costs"
            ),
            mitigation=(
                "Plan critical below-ground work outside monsoon season. "
                "Include dewatering costs. Build schedule float for "
                "monsoon disruptions."
            ),
            source="site_analysis",
        ))

    # --- Urban congested site ---
    if _is_location_in_set(loc_lower, _URBAN_CONGESTED_LOCATIONS):
        risks.append(RiskItem(
            category=RiskCategory.SITE,
            level=RiskLevel.MEDIUM,
            title="Urban congested site — access restrictions",
            description=(
                f"'{location}' is a densely urban area. "
                "Material delivery may be restricted to night hours, "
                "crane operations may have height/time limits, and "
                "labour accommodation may be expensive."
            ),
            financial_impact="5-15% premium on logistics and labour accommodation",
            mitigation=(
                "Plan night-time material deliveries. "
                "Check local municipal restrictions on construction hours. "
                "Budget for higher labour accommodation costs."
            ),
            source="site_analysis",
        ))

    # --- Remote / NE India / island territory ---
    if _is_location_in_set(loc_lower, _REMOTE_LOCATIONS):
        risks.append(RiskItem(
            category=RiskCategory.SITE,
            level=RiskLevel.HIGH,
            title="Remote/island location — limited access and resources",
            description=(
                f"'{location}' has limited road/air/sea connectivity. "
                "Skilled labour availability is low, material lead times "
                "are extended, and working season may be restricted."
            ),
            financial_impact=(
                "25-50% premium on overall project cost compared to "
                "metro locations"
            ),
            mitigation=(
                "Plan material procurement 3-6 months in advance. "
                "Consider pre-fabrication to reduce on-site work. "
                "Arrange labour from nearby metros with accommodation."
            ),
            source="site_analysis",
        ))

    # --- High-value project in any non-metro location ---
    _metro_cities = {"delhi", "mumbai", "bangalore", "bengaluru",
                     "hyderabad", "chennai", "kolkata", "pune", "ahmedabad"}
    if (project_value_lakhs > 1000
            and not _is_location_in_set(loc_lower, _metro_cities)):
        risks.append(RiskItem(
            category=RiskCategory.SITE,
            level=RiskLevel.LOW,
            title="Large project in non-metro location",
            description=(
                f"Project value of {project_value_lakhs:.0f} lakhs "
                f"in {location} may face resource mobilisation challenges."
            ),
            financial_impact="Mobilisation and demobilisation cost premium",
            mitigation=(
                "Plan early mobilisation. Secure accommodation and "
                "offices at site in advance."
            ),
            source="site_analysis",
        ))

    return risks


# ---------------------------------------------------------------------------
# Risk score calculation
# ---------------------------------------------------------------------------

_RISK_WEIGHTS: Dict[RiskLevel, int] = {
    RiskLevel.CRITICAL: 25,
    RiskLevel.HIGH: 15,
    RiskLevel.MEDIUM: 5,
    RiskLevel.LOW: 1,
    RiskLevel.NONE: 0,
}


def _calculate_risk_score(risks: List[RiskItem]) -> float:
    """
    Calculate an overall risk score between 0 and 100.

    Uses weighted sum of risk items normalised against a theoretical maximum
    where every risk would be CRITICAL.
    """
    if not risks:
        return 0.0

    total = sum(_RISK_WEIGHTS[r.level] for r in risks)
    max_possible = len(risks) * _RISK_WEIGHTS[RiskLevel.CRITICAL]

    if max_possible == 0:
        return 0.0

    return min(100.0, round(total / max_possible * 100, 1))


# ---------------------------------------------------------------------------
# Bid recommendation
# ---------------------------------------------------------------------------

def _determine_recommendation(report: BidRiskReport) -> str:
    """
    Determine bid go/no-go recommendation.

    Rules:
    - Any CRITICAL risk     -> NO_GO
    - 3+ HIGH risks         -> NO_GO
    - 1-2 HIGH risks        -> GO_WITH_QUALIFICATIONS
    - Only MEDIUM/LOW risks -> GO
    - No risks              -> GO
    """
    if not report.risks:
        return "GO"

    num_critical = len(report.critical_risks)
    num_high = len(report.high_risks)

    if num_critical > 0:
        return "NO_GO"

    if num_high >= 3:
        return "NO_GO"

    if num_high >= 1:
        return "GO_WITH_QUALIFICATIONS"

    return "GO"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boq_contains_category(
    boq_items: List[Dict[str, Any]],
    keywords: frozenset,
) -> bool:
    """Check whether any BOQ item description matches a keyword set."""
    for item in boq_items:
        desc = str(item.get("description", "")).lower()
        for kw in keywords:
            if kw in desc:
                return True
    return False


def _estimate_steel_fraction(
    boq_items: List[Dict[str, Any]],
    project_value_lakhs: float,
) -> float:
    """
    Estimate what fraction of project value is steel/reinforcement.

    Looks at BOQ item amounts for steel-related items. Falls back to
    a heuristic based on item descriptions if amounts are not available.
    """
    if not boq_items or project_value_lakhs <= 0:
        return 0.0

    steel_keywords = {"steel", "reinforcement", "rebar", "tmt", "structural steel"}
    steel_value = 0.0
    total_value = 0.0

    for item in boq_items:
        amount = _safe_float(item.get("amount", 0))
        total_value += amount
        desc = str(item.get("description", "")).lower()
        for kw in steel_keywords:
            if kw in desc:
                steel_value += amount
                break

    if total_value > 0:
        return steel_value / total_value

    # Fallback: count-based heuristic
    steel_count = sum(
        1 for item in boq_items
        if any(kw in str(item.get("description", "")).lower()
               for kw in steel_keywords)
    )
    if boq_items:
        return steel_count / len(boq_items)

    return 0.0


def _is_location_in_set(location_lower: str, location_set: frozenset) -> bool:
    """
    Check if a location string matches any entry in a location set.

    Supports partial matching so that "new delhi" matches "delhi" etc.
    """
    for loc in location_set:
        if loc in location_lower or location_lower in loc:
            return True
    return False


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
