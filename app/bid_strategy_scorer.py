"""
Bid Strategy Scorer

Pure computation module (no Streamlit dependency) that produces
4 dial scores + confidence levels + document-driven recommendations.

Scores:
  1. Client Fit (0-100) — from user inputs about client relationship
  2. Risk Score (0-100, higher = more risk) — from document findings + user flags
  3. Competition Score (0-100, higher = more pressure) — from competitor/market inputs
  4. Readiness Score (0-100) — directly from document analysis payload

Each score is a dict:
  {"name": str, "score": int|None, "confidence": "HIGH"|"MEDIUM"|"LOW"|"UNKNOWN",
   "based_on": [str, ...]}

None score + "UNKNOWN" confidence means insufficient inputs to compute.
Recommendations are document-derived facts, never financial advice.
"""

from typing import Dict, List, Any, Optional


def compute_bid_strategy(
    inputs: Dict[str, Any],
    payload: Dict[str, Any],
    estimating_playbook: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute bid strategy scores from user inputs and document analysis payload.

    Args:
        inputs: User-provided form inputs:
            - relationship_level: "" | "New" | "Repeat" | "Preferred"
            - past_work_count: int (0+)
            - last_project_date: str (e.g. "2024-06")
            - payment_delays: bool
            - disputes: bool
            - high_co_rate: bool
            - competitors: list[str]
            - market_pressure: int (0-10)
            - target_margin: float (%)
            - win_probability: float (%)
        payload: Analysis result payload dict with:
            - readiness_score, sub_scores, blockers, rfis, trade_coverage
        estimating_playbook: Optional playbook dict (Sprint 20C).
            Seeds inputs with relationship_bid, competition_intensity,
            must_win, risk_posture when corresponding form fields are empty.

    Returns:
        Dict with keys:
            - client_fit: score dict
            - risk_score: score dict
            - competition_score: score dict
            - readiness_score: score dict
            - recommendations: list[str]
            - missing_inputs: list[str]
            - playbook_applied: bool (Sprint 20C)
    """
    # Sprint 20C: Seed inputs from playbook when form fields are empty
    playbook_applied = False
    if estimating_playbook and isinstance(estimating_playbook, dict):
        inputs = dict(inputs)  # shallow copy to avoid mutating original
        project = estimating_playbook.get("project", {})
        company = estimating_playbook.get("company", {})

        # relationship_bid → treat as "Repeat" relationship if set and no form value
        if not inputs.get("relationship_level") and project.get("relationship_bid"):
            inputs["relationship_level"] = "Repeat"
            playbook_applied = True

        # competition_intensity → map to market_pressure (low=2, med=5, high=8)
        if not inputs.get("market_pressure") and project.get("competition_intensity"):
            ci_map = {"low": 2, "med": 5, "high": 8}
            inputs["market_pressure"] = ci_map.get(
                project.get("competition_intensity", ""), 0
            )
            if inputs["market_pressure"]:
                playbook_applied = True

        # must_win → boost win_probability to 80% if not set
        if (inputs.get("win_probability") is None or inputs.get("win_probability") == 0) \
                and project.get("must_win"):
            inputs["win_probability"] = 80.0
            playbook_applied = True

        # risk_posture → adjust target_margin default (conservative=+2, aggressive=-1)
        if (inputs.get("target_margin") is None or inputs.get("target_margin") == 0) \
                and company.get("risk_posture"):
            posture = company.get("risk_posture", "balanced")
            profit = float(company.get("default_profit_pct", 8.0) or 8.0)
            if posture == "conservative":
                inputs["target_margin"] = profit + 2.0
            elif posture == "aggressive":
                inputs["target_margin"] = max(1.0, profit - 1.0)
            else:
                inputs["target_margin"] = profit
            playbook_applied = True

    client_fit = _compute_client_fit(inputs)
    risk_score = _compute_risk_score(inputs, payload)
    competition_score = _compute_competition_score(inputs)
    readiness_score = _compute_readiness_score(payload)
    recommendations = _compute_recommendations(payload)
    missing_inputs = _find_missing_inputs(inputs)

    return {
        "client_fit": client_fit,
        "risk_score": risk_score,
        "competition_score": competition_score,
        "readiness_score": readiness_score,
        "recommendations": recommendations,
        "missing_inputs": missing_inputs,
        "playbook_applied": playbook_applied,
    }


def _compute_client_fit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Client Fit score (0-100). UNKNOWN if relationship_level not provided."""
    relationship = inputs.get("relationship_level", "")
    if not relationship:
        return {
            "name": "Client Fit",
            "score": None,
            "confidence": "UNKNOWN",
            "based_on": ["Provide relationship level to compute"],
        }

    # Base score from relationship
    base_map = {"New": 30, "Repeat": 60, "Preferred": 80}
    score = base_map.get(relationship, 30)
    based_on = [f"{relationship} client (+{score})"]

    # Past work bonus: +5 per project, cap +20
    past_count = inputs.get("past_work_count", 0) or 0
    if past_count > 0:
        bonus = min(past_count * 5, 20)
        score += bonus
        based_on.append(f"{past_count} past project(s) (+{bonus})")

    # Negative flags
    if inputs.get("payment_delays"):
        score -= 20
        based_on.append("Payment delays (-20)")
    if inputs.get("disputes"):
        score -= 25
        based_on.append("Disputes (-25)")
    if inputs.get("high_co_rate"):
        score -= 15
        based_on.append("High CO rate (-15)")

    score = max(0, min(100, score))

    # Confidence based on how much info we have
    flags_provided = sum(1 for k in ("payment_delays", "disputes", "high_co_rate")
                         if k in inputs and inputs[k] is not None)
    confidence = "HIGH" if flags_provided >= 2 else "MEDIUM"

    return {
        "name": "Client Fit",
        "score": score,
        "confidence": confidence,
        "based_on": based_on,
    }


def _compute_risk_score(inputs: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Risk Score (0-100, higher = more risk). Always computed (document-derived base)."""
    score = 0
    based_on = []

    # Document-derived risk (always available)
    blockers = payload.get("blockers", [])
    not_found_count = sum(
        1 for b in blockers
        if b.get("coverage_status") == "not_found_after_search"
    )
    if not_found_count > 0:
        contrib = not_found_count * 10
        score += contrib
        based_on.append(f"{not_found_count} NOT_FOUND blocker(s) (+{contrib})")

    critical_count = sum(
        1 for b in blockers
        if b.get("severity") == "critical"
    )
    if critical_count > 0:
        contrib = critical_count * 15
        score += contrib
        based_on.append(f"{critical_count} critical blocker(s) (+{contrib})")

    # Trades with 0% coverage and assumed items
    trade_coverage = payload.get("trade_coverage", [])
    zero_coverage_trades = [
        t for t in trade_coverage
        if t.get("coverage_pct", 100) == 0 and t.get("assumed_count", 0) > 0
    ]
    if zero_coverage_trades:
        contrib = len(zero_coverage_trades) * 5
        score += contrib
        trade_names = ", ".join(t.get("trade", "?") for t in zero_coverage_trades)
        based_on.append(f"{len(zero_coverage_trades)} trade(s) at 0% coverage: {trade_names} (+{contrib})")

    if not based_on:
        based_on.append("No blockers found (base risk = 0)")

    # User-provided flags
    user_flags_count = 0
    if inputs.get("payment_delays"):
        score += 15
        based_on.append("Payment delays (+15)")
        user_flags_count += 1
    if inputs.get("disputes"):
        score += 20
        based_on.append("Disputes (+20)")
        user_flags_count += 1
    if inputs.get("high_co_rate"):
        score += 10
        based_on.append("High CO rate (+10)")
        user_flags_count += 1

    market_pressure = inputs.get("market_pressure", 0) or 0
    if market_pressure > 7:
        score += 10
        based_on.append(f"High market pressure ({market_pressure}/10) (+10)")
        user_flags_count += 1

    score = max(0, min(100, score))

    # Confidence: HIGH if many inputs, MEDIUM if some, LOW if document-only
    if user_flags_count >= 3:
        confidence = "HIGH"
    elif user_flags_count >= 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "name": "Risk Score",
        "score": score,
        "confidence": confidence,
        "based_on": based_on,
    }


def _compute_competition_score(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Competition Score (0-100, higher = more competitive pressure)."""
    competitors = inputs.get("competitors", []) or []
    market_pressure = inputs.get("market_pressure", 0) or 0

    if not competitors and not market_pressure:
        return {
            "name": "Competition",
            "score": None,
            "confidence": "UNKNOWN",
            "based_on": ["Provide competitors or market pressure to compute"],
        }

    score = 0
    based_on = []

    # Competitors: 15 per competitor, cap at 75
    if competitors:
        contrib = min(len(competitors) * 15, 75)
        score += contrib
        based_on.append(f"{len(competitors)} known competitor(s) (+{contrib})")

    # Market pressure: 5 per point
    if market_pressure:
        contrib = market_pressure * 5
        score += contrib
        based_on.append(f"Market pressure {market_pressure}/10 (+{contrib})")

    score = max(0, min(100, score))

    confidence = "MEDIUM" if competitors else "LOW"

    return {
        "name": "Competition",
        "score": score,
        "confidence": confidence,
        "based_on": based_on,
    }


def _compute_readiness_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Readiness Score (0-100). Always computed from document analysis."""
    score = payload.get("readiness_score", 0)
    if score is None:
        score = 0

    sub_scores = payload.get("sub_scores", {})
    based_on = []

    if sub_scores:
        for key, val in sub_scores.items():
            if val is not None:
                based_on.append(f"{key}: {val}/100")
    else:
        based_on.append("Score not available in payload")

    if score == 0 and not sub_scores:
        confidence = "LOW"
        based_on = ["Score not available"]
    else:
        confidence = "HIGH"

    return {
        "name": "Readiness",
        "score": max(0, min(100, score)),
        "confidence": confidence,
        "based_on": based_on,
    }


def _compute_recommendations(payload: Dict[str, Any]) -> List[str]:
    """Document-driven recommendations. Always shown regardless of user inputs."""
    recs = []

    blockers = payload.get("blockers", [])

    # High scope ambiguity
    not_found_count = sum(
        1 for b in blockers
        if b.get("coverage_status") == "not_found_after_search"
    )
    if not_found_count >= 2:
        recs.append(
            f"{not_found_count} items not found after full search "
            "- recommend submitting RFIs before bid or including explicit exclusions"
        )

    # Missing schedules
    missing_schedules = [
        b for b in blockers
        if b.get("issue_type") == "missing_schedule"
    ]
    if missing_schedules:
        schedule_types = [b.get("title", "?") for b in missing_schedules]
        recs.append(
            f"Missing schedule(s) detected ({len(missing_schedules)}) "
            "- recommend explicit assumptions with qualifications rather than silent exclusions"
        )

    # Trades at 0% coverage
    trade_coverage = payload.get("trade_coverage", [])
    zero_trades = [
        t for t in trade_coverage
        if t.get("coverage_pct", 100) == 0
    ]
    for t in zero_trades:
        trade_name = t.get("trade", "unknown")
        recs.append(
            f'Trade "{trade_name}" has 0% coverage '
            "- highlight risk in bid cover letter"
        )

    # Low readiness
    readiness = payload.get("readiness_score", 100)
    if readiness is not None and readiness < 50:
        recs.append(
            f"Low readiness score ({readiness}/100) "
            "- consider requesting tender extension"
        )

    if not recs:
        recs.append("No significant document-driven risks identified")

    return recs


def _find_missing_inputs(inputs: Dict[str, Any]) -> List[str]:
    """List user inputs that are empty/missing."""
    missing = []

    if not inputs.get("relationship_level"):
        missing.append("relationship_level")
    if not inputs.get("competitors"):
        missing.append("competitors")
    if not inputs.get("market_pressure"):
        missing.append("market_pressure")
    if inputs.get("target_margin") is None or inputs.get("target_margin") == 0:
        missing.append("target_margin")
    if inputs.get("win_probability") is None or inputs.get("win_probability") == 0:
        missing.append("win_probability")

    return missing
