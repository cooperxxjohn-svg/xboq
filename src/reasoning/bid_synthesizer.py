"""
src/reasoning/bid_synthesizer.py

Compile all extracted and reasoned data into a single BidSynthesis object —
the top-level "bid readiness report" that drives the Bid Intelligence UI tab.

BidSynthesis contains:
  - bid_readiness_score   (0-100)
  - executive_summary     (LLM-generated or rule template)
  - scope_summary         per-trade item count + estimated cost
  - critical_gaps         list of CRITICAL Gap objects
  - rfi_list              formatted ready-to-send RFIs from gaps
  - risk_register         structured risk entries
  - cost figures          grand total, per-sqm, gap exposure, contingency %
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BidSynthesis:
    project_name: str
    bid_readiness_score: int           # 0-100
    bid_readiness_label: str           # "READY" | "CONDITIONAL" | "NOT READY"
    executive_summary: str
    scope_summary: Dict[str, Any]      # {trade: {items, estimated_cost_inr}}
    critical_gaps: List[Any]           # Gap objects with severity == CRITICAL
    all_gaps: List[Any]                # all Gap objects
    total_gap_exposure_inr: float
    rfi_list: List[Dict]
    risk_register: List[Dict]
    estimated_cost_inr: float
    cost_per_sqm: float
    recommended_contingency_pct: float
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    award_probability: Optional[int] = None   # 0-100, set by award_predictor after synthesis

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "bid_readiness_score": self.bid_readiness_score,
            "bid_readiness_label": self.bid_readiness_label,
            "executive_summary": self.executive_summary,
            "scope_summary": self.scope_summary,
            "critical_gaps": [_gap_to_dict(g) for g in self.critical_gaps],
            "all_gaps": [_gap_to_dict(g) for g in self.all_gaps],
            "total_gap_exposure_inr": self.total_gap_exposure_inr,
            "rfi_list": self.rfi_list,
            "risk_register": self.risk_register,
            "estimated_cost_inr": self.estimated_cost_inr,
            "cost_per_sqm": self.cost_per_sqm,
            "recommended_contingency_pct": self.recommended_contingency_pct,
            "generated_at": self.generated_at,
            "award_probability": self.award_probability,
        }


def synthesize_bid(
    payload: dict,
    gaps: List[Any],
    cost_impacts: List[Any],
    llm_client: Any = None,
) -> BidSynthesis:
    """
    Build a BidSynthesis from payload + gap/impact analysis.

    Works without an LLM client (uses rule-template for executive_summary).
    """
    project_name = (
        payload.get("project_name")
        or payload.get("filename", "")
        or payload.get("project_id", "Unnamed Project")
    )

    # ── Cost figures ──────────────────────────────────────────────────────
    qto = payload.get("qto_summary") or {}
    estimated_cost_inr = float(qto.get("grand_total_inr", 0) or 0)
    area_sqm = float(qto.get("vmeas_area_sqm", 0) or 0)
    cost_per_sqm = (estimated_cost_inr / area_sqm) if area_sqm > 0 else 0.0

    from .cost_impact import total_exposure
    total_gap_exposure = total_exposure(cost_impacts)

    # ── Readiness score ────────────────────────────────────────────────────
    score = _compute_readiness_score(payload, gaps, estimated_cost_inr)
    label = (
        "READY" if score >= 75
        else "CONDITIONAL" if score >= 50
        else "NOT READY"
    )

    # ── Recommended contingency % ──────────────────────────────────────────
    if estimated_cost_inr > 0 and total_gap_exposure > 0:
        raw_contingency = total_gap_exposure / estimated_cost_inr * 100
        recommended_contingency_pct = min(30.0, max(5.0, round(raw_contingency, 1)))
    else:
        recommended_contingency_pct = 10.0 if score < 70 else 5.0

    # ── Scope summary ──────────────────────────────────────────────────────
    scope_summary = _build_scope_summary(payload)

    # ── RFI list ───────────────────────────────────────────────────────────
    rfi_list = _build_rfi_list(payload, gaps)

    # ── Risk register ──────────────────────────────────────────────────────
    risk_register = _build_risk_register(payload, gaps)

    # ── Executive summary ──────────────────────────────────────────────────
    critical_gaps = [g for g in gaps if g.severity == "CRITICAL"]
    exec_summary = _executive_summary(
        payload, gaps, critical_gaps, score, label,
        estimated_cost_inr, cost_per_sqm, llm_client,
    )

    return BidSynthesis(
        project_name=str(project_name),
        bid_readiness_score=score,
        bid_readiness_label=label,
        executive_summary=exec_summary,
        scope_summary=scope_summary,
        critical_gaps=critical_gaps,
        all_gaps=gaps,
        total_gap_exposure_inr=round(total_gap_exposure),
        rfi_list=rfi_list,
        risk_register=risk_register,
        estimated_cost_inr=round(estimated_cost_inr),
        cost_per_sqm=round(cost_per_sqm),
        recommended_contingency_pct=recommended_contingency_pct,
    )


# ── Scoring ────────────────────────────────────────────────────────────────────

def _compute_readiness_score(payload: dict, gaps: List[Any], estimated_cost_inr: float) -> int:
    score = 100

    # Deduct per gap severity
    for g in gaps:
        sev = g.severity
        if sev == "CRITICAL":
            score -= 20
        elif sev == "HIGH":
            score -= 8
        elif sev == "MEDIUM":
            score -= 2

    # Bonus: rates applied
    if estimated_cost_inr > 0:
        score += 5

    # Bonus: taxonomy matched
    qto = payload.get("qto_summary") or {}
    total_items = qto.get("total_spec_items", 0) or 0
    tax_matched = qto.get("taxonomy_matched", 0) or 0
    if total_items > 0 and (tax_matched / total_items) > 0.6:
        score += 5

    # Existing pipeline readiness score as a signal (weighted 20%)
    pipeline_score = payload.get("readiness_score", payload.get("qa_score"))
    if isinstance(pipeline_score, (int, float)):
        score = int(score * 0.8 + pipeline_score * 0.2)

    return max(0, min(100, score))


# ── Sub-builders ───────────────────────────────────────────────────────────────

def _build_scope_summary(payload: dict) -> dict:
    qto = payload.get("qto_summary") or {}
    trade_summary = qto.get("trade_summary") or {}
    scope: Dict[str, Any] = {}

    for trade, info in trade_summary.items():
        scope[trade] = {
            "items": info.get("item_count", 0),
            "estimated_cost_inr": round(info.get("total_amount", 0)),
        }

    # Add structural QTO stats if present
    if qto.get("st_concrete_cum"):
        scope.setdefault("structural", {})["concrete_cum"] = qto["st_concrete_cum"]
    if qto.get("st_steel_kg"):
        scope.setdefault("structural", {})["steel_kg"] = qto["st_steel_kg"]
    if qto.get("st_floors"):
        scope.setdefault("structural", {})["floors"] = qto["st_floors"]

    return scope


def _build_rfi_list(payload: dict, gaps: List[Any]) -> List[dict]:
    rfis: List[dict] = []

    # From gaps (action_required becomes the RFI question)
    for gap in gaps:
        if gap.severity in ("CRITICAL", "HIGH") and gap.action_required:
            rfis.append({
                "ref": f"RFI-{gap.id}",
                "trade": gap.trade,
                "priority": gap.severity,
                "question": gap.action_required,
                "basis": gap.description,
                "source": "gap_analysis",
            })

    # From existing pipeline RFIs
    for rfi in (payload.get("rfis") or [])[:20]:
        if rfi.get("question"):
            rfis.append({
                "ref": rfi.get("id", "RFI-?"),
                "trade": rfi.get("trade", "general"),
                "priority": rfi.get("severity", "MEDIUM"),
                "question": rfi.get("question", ""),
                "basis": rfi.get("description", ""),
                "source": "pipeline",
            })

    return rfis[:50]  # cap at 50


def _build_risk_register(payload: dict, gaps: List[Any]) -> List[dict]:
    risks: List[dict] = []
    seen: set = set()

    for gap in gaps:
        key = (gap.trade, gap.severity)
        if key in seen:
            continue
        seen.add(key)

        prob = {"CRITICAL": "HIGH", "HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}.get(gap.severity, "MEDIUM")
        impact = {"CRITICAL": "HIGH", "HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}.get(gap.severity, "MEDIUM")
        risks.append({
            "trade": gap.trade,
            "risk": gap.description[:120],
            "probability": prob,
            "impact": impact,
            "mitigation": gap.action_required[:200] if gap.action_required else "Monitor and raise RFI",
            "source": gap.source,
        })

    # Add existing blocker risks
    for blk in (payload.get("blockers") or [])[:10]:
        sev = str(blk.get("severity", "MEDIUM")).upper()
        if (blk.get("trade", "general"), sev) not in seen:
            risks.append({
                "trade": blk.get("trade", "general"),
                "risk": str(blk.get("description") or blk.get("title", ""))[:120],
                "probability": "HIGH" if sev == "CRITICAL" else "MEDIUM",
                "impact": "HIGH",
                "mitigation": str(blk.get("fix_action", "Resolve before bid submission"))[:200],
                "source": "blocker",
            })

    return risks[:30]


# ── Executive summary ──────────────────────────────────────────────────────────

_SUMMARY_TEMPLATE = (
    "This {label} bid package has a readiness score of {score}/100. "
    "{cost_line}"
    "{gap_line}"
    "{action_line}"
)


def _executive_summary(
    payload: dict,
    gaps: List[Any],
    critical_gaps: List[Any],
    score: int,
    label: str,
    estimated_cost_inr: float,
    cost_per_sqm: float,
    llm_client: Any,
) -> str:
    # Try LLM first
    if llm_client is not None:
        try:
            summary = _llm_executive_summary(
                payload, gaps, critical_gaps, score, label, estimated_cost_inr, llm_client
            )
            if summary:
                return summary
        except Exception as e:
            logger.warning("LLM executive summary failed: %s", e)

    # Rule-based fallback
    cost_line = (
        f"Estimated project cost is ₹{estimated_cost_inr/1e7:.1f} Cr (₹{cost_per_sqm:,.0f}/sqm). "
        if estimated_cost_inr > 0
        else "Project cost estimate is not available. "
    )

    n_all = len(gaps)
    n_crit = len(critical_gaps)
    if n_crit > 0:
        gap_line = (
            f"There are {n_crit} critical gap(s) and {n_all - n_crit} additional gaps "
            f"that require resolution before submitting a firm bid. "
        )
    elif n_all > 0:
        gap_line = f"There are {n_all} gap(s) requiring attention before bidding. "
    else:
        gap_line = "No significant gaps identified. "

    action_line = (
        "Resolve critical issues and obtain subcontractor quotes for unpriced trades before submission."
        if n_crit > 0
        else "Review flagged items and confirm scope with the client."
    )

    return _SUMMARY_TEMPLATE.format(
        label=label,
        score=score,
        cost_line=cost_line,
        gap_line=gap_line,
        action_line=action_line,
    )


_LLM_SUMMARY_PROMPT = """\
You are a quantity surveyor writing a 3-sentence executive summary of a construction bid analysis.

Project: {project_name}
Bid Readiness: {label} ({score}/100)
Estimated Cost: {cost}
Critical Gaps: {n_critical}
Total Gaps: {n_gaps}
Key Gap Summaries:
{gap_summaries}

Write exactly 3 concise sentences: (1) project overview and readiness, (2) cost and coverage,
(3) recommended actions. Be direct and professional. No bullet points.
"""


def _llm_executive_summary(
    payload, gaps, critical_gaps, score, label, estimated_cost_inr, llm_client
) -> str:
    project_name = payload.get("project_name") or payload.get("project_id", "Project")
    cost_str = f"₹{estimated_cost_inr/1e7:.1f} Cr" if estimated_cost_inr > 0 else "Unknown"
    gap_summaries = "\n".join(
        f"- [{g.severity}] {g.trade}: {g.description[:80]}"
        for g in gaps[:8]
    )
    prompt = _LLM_SUMMARY_PROMPT.format(
        project_name=project_name,
        label=label,
        score=score,
        cost=cost_str,
        n_critical=len(critical_gaps),
        n_gaps=len(gaps),
        gap_summaries=gap_summaries or "None identified",
    )

    try:
        from src.utils.llm_caller import call_llm
        return call_llm(
            client=llm_client,
            system="You are an expert construction bid analyst.",
            user=prompt,
            max_tokens=800,
            temperature=0.3,
            openai_model="gpt-4o-mini",
            anthropic_model="claude-haiku-4-5-20251001",
        ).strip()
    except Exception:
        return ""


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gap_to_dict(g: Any) -> dict:
    return {
        "id": g.id,
        "trade": g.trade,
        "severity": g.severity,
        "description": g.description,
        "evidence": getattr(g, "evidence", []),
        "action_required": getattr(g, "action_required", ""),
        "cost_impact": getattr(g, "cost_impact", None),
        "source": getattr(g, "source", "rule"),
    }
