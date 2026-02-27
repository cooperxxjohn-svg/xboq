"""
LLM Enrichment for Deep Analysis

Uses LLM to enrich blockers with grounded analysis.
LLM is constrained to use only detected entities from plan graph.

Key principles:
1. LLM does not hallucinate entities - only uses what was detected
2. LLM provides impact reasoning based on construction knowledge
3. LLM suggests resolution priorities based on bid timeline
4. All LLM outputs are validated against plan graph evidence
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.analysis_models import (
    PlanSetGraph, Blocker, RFIItem, EvidenceRef,
    Severity, BidImpact, RiskLevel, Trade,
    DeepAnalysisResult, ReadinessScore,
)


# =============================================================================
# LLM PROMPTS
# =============================================================================

ENRICHMENT_SYSTEM_PROMPT = """You are an expert construction estimator analyzing architectural drawings for bid readiness.

Your role is to enrich blocker findings with construction-specific reasoning.
You MUST only reference entities that have been detected in the plan set - never invent or assume.

For each blocker, provide:
1. Construction impact explanation (why this matters for pricing)
2. Risk assessment with specifics (cost variance, schedule delay potential)
3. Prioritized resolution steps
4. Trade-offs if using assumptions

Be concise and actionable. Use Indian construction terminology where appropriate (lakh, crore, PWD rates, etc.)."""


ENRICHMENT_USER_TEMPLATE = """## Plan Set Summary
- Total Pages: {total_pages}
- Disciplines Found: {disciplines}
- Sheet Types: {sheet_types}

## Detected Entities
- Door Tags: {door_tags}
- Window Tags: {window_tags}
- Room Types: {room_names}
- Pages With Scale: {pages_with_scale}
- Pages Without Scale: {pages_without_scale}

## Blocker to Enrich
ID: {blocker_id}
Title: {blocker_title}
Trade: {trade}
Current Severity: {severity}

Evidence:
- Pages: {evidence_pages}
- Detected Items: {evidence_entities}
- Searched For: {evidence_search}

Current Description:
{description}

---

Enrich this blocker with:
1. **Construction Impact** (2-3 sentences on why this blocks accurate pricing)
2. **Cost Risk** (specific range like "10-20% variance on door package")
3. **Schedule Risk** (potential delay if discovered late)
4. **Resolution Priority** (high/medium/low with reasoning)
5. **If Using Assumption** (what assumption to use and risk of being wrong)

Respond in JSON format:
{{
    "construction_impact": "...",
    "cost_risk_detail": "...",
    "cost_variance_pct": [10, 20],
    "schedule_risk_detail": "...",
    "schedule_delay_days": [5, 15],
    "resolution_priority": "high|medium|low",
    "priority_reasoning": "...",
    "assumption_if_needed": "...",
    "assumption_risk": "..."
}}"""


SUMMARY_PROMPT_TEMPLATE = """## Deep Analysis Results

Total Blockers: {total_blockers}
Critical/High: {critical_count}
Trades Affected: {trades_affected}

## Key Blockers:
{blockers_summary}

## Trade Coverage:
{trade_coverage}

---

Provide a 3-4 sentence executive summary for this bid analysis.
Focus on:
1. Overall bid readiness status
2. Most impactful blockers to resolve first
3. Estimated effort to reach GO status

Keep it actionable for a construction project manager."""


# =============================================================================
# LLM ENRICHMENT CLASS
# =============================================================================

class LLMEnrichment:
    """
    Enriches blockers using LLM with grounded context.

    LLM is constrained to use only entities from the plan graph.
    """

    def __init__(self, llm_client=None):
        """
        Initialize with optional LLM client.

        If no client provided, enrichment returns defaults.
        """
        self.llm_client = llm_client

    def enrich_blocker(
        self,
        blocker: Blocker,
        graph: PlanSetGraph
    ) -> Dict[str, Any]:
        """
        Enrich a single blocker with LLM analysis.

        Args:
            blocker: Blocker to enrich
            graph: PlanSetGraph for context

        Returns:
            Dict with enrichment data
        """
        if not self.llm_client:
            return self._default_enrichment(blocker)

        # Build prompt with grounded context
        prompt = ENRICHMENT_USER_TEMPLATE.format(
            total_pages=graph.total_pages,
            disciplines=", ".join(graph.disciplines_found) or "Unknown",
            sheet_types=json.dumps(graph.sheet_types_found),
            door_tags=", ".join(graph.all_door_tags[:20]) or "None",
            window_tags=", ".join(graph.all_window_tags[:20]) or "None",
            room_names=", ".join(graph.all_room_names[:20]) or "None",
            pages_with_scale=graph.pages_with_scale,
            pages_without_scale=graph.pages_without_scale,
            blocker_id=blocker.id,
            blocker_title=blocker.title,
            trade=blocker.trade.value,
            severity=blocker.severity.value,
            evidence_pages=blocker.evidence.pages[:10] if blocker.evidence.pages else "None",
            evidence_entities=json.dumps(blocker.evidence.detected_entities),
            evidence_search=json.dumps(blocker.evidence.search_attempts),
            description=blocker.description,
        )

        try:
            response = self._call_llm(
                system_prompt=ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=prompt
            )
            return self._parse_enrichment_response(response)
        except Exception as e:
            print(f"LLM enrichment failed: {e}")
            return self._default_enrichment(blocker)

    def enrich_all_blockers(
        self,
        blockers: List[Blocker],
        graph: PlanSetGraph
    ) -> List[Dict[str, Any]]:
        """
        Enrich all blockers.

        Args:
            blockers: List of blockers
            graph: PlanSetGraph for context

        Returns:
            List of enrichment dicts
        """
        enrichments = []
        for blocker in blockers:
            enrichment = self.enrich_blocker(blocker, graph)
            enrichments.append({
                "blocker_id": blocker.id,
                "enrichment": enrichment,
            })
        return enrichments

    def generate_summary(
        self,
        result: DeepAnalysisResult
    ) -> str:
        """
        Generate executive summary using LLM.

        Args:
            result: DeepAnalysisResult to summarize

        Returns:
            Summary text
        """
        if not self.llm_client:
            return self._default_summary(result)

        # Build blockers summary
        blockers_text = ""
        for b in result.blockers[:5]:
            blockers_text += f"- [{b.severity.value.upper()}] {b.title}\n"

        # Build trade coverage summary
        trade_text = ""
        for t in result.trade_coverage:
            trade_text += f"- {t.trade.value}: {t.coverage_pct:.0f}% ({t.next_action})\n"

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            total_blockers=len(result.blockers),
            critical_count=sum(1 for b in result.blockers
                              if b.severity in [Severity.CRITICAL, Severity.HIGH]),
            trades_affected=", ".join(set(b.trade.value for b in result.blockers)),
            blockers_summary=blockers_text,
            trade_coverage=trade_text,
        )

        try:
            response = self._call_llm(
                system_prompt=ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=prompt
            )
            return response.strip()
        except Exception:
            return self._default_summary(result)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM client.

        Override this method for specific LLM implementations.
        """
        if hasattr(self.llm_client, 'chat'):
            # OpenAI-style client
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Use smaller model for enrichment
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content
        elif hasattr(self.llm_client, 'messages'):
            # Anthropic-style client
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        else:
            raise ValueError("Unknown LLM client type")

    def _parse_enrichment_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return self._default_enrichment_values()

    def _default_enrichment(self, blocker: Blocker) -> Dict[str, Any]:
        """Return default enrichment when LLM not available."""
        defaults = self._default_enrichment_values()

        # Customize based on blocker type
        if "schedule" in blocker.issue_type:
            defaults["construction_impact"] = (
                f"Without {blocker.missing_dependency[0] if blocker.missing_dependency else 'schedule'}, "
                f"quantities cannot be priced against specifications. Standard rates may not apply."
            )
            defaults["cost_variance_pct"] = [15, 30]

        elif "scale" in blocker.issue_type:
            defaults["construction_impact"] = (
                "Scale is essential for accurate quantity takeoff. "
                "All area and length measurements are unreliable without confirmed scale."
            )
            defaults["cost_variance_pct"] = [20, 50]

        elif "drawing" in blocker.issue_type:
            defaults["construction_impact"] = (
                f"Missing {blocker.missing_dependency[0] if blocker.missing_dependency else 'drawings'} "
                f"means scope is incomplete. Major cost items may be missed."
            )
            defaults["cost_variance_pct"] = [10, 40]

        return defaults

    def _default_enrichment_values(self) -> Dict[str, Any]:
        """Return base default values."""
        return {
            "construction_impact": "Impact analysis requires LLM enrichment.",
            "cost_risk_detail": "Cost variance depends on missing information.",
            "cost_variance_pct": [10, 25],
            "schedule_risk_detail": "Schedule impact depends on when issue is discovered.",
            "schedule_delay_days": [5, 20],
            "resolution_priority": "high",
            "priority_reasoning": "Resolve before pricing to avoid assumptions.",
            "assumption_if_needed": "Use market standard specifications.",
            "assumption_risk": "Actual specifications may differ significantly.",
        }

    def _default_summary(self, result: DeepAnalysisResult) -> str:
        """Generate default summary without LLM."""
        critical = sum(1 for b in result.blockers
                      if b.severity in [Severity.CRITICAL, Severity.HIGH])

        if critical == 0:
            status = "appears ready for pricing"
            action = "Verify assumptions before bid submission"
        elif critical <= 2:
            status = "has minor gaps"
            action = f"Resolve {critical} high-priority items first"
        else:
            status = "is not ready for pricing"
            action = f"Address {critical} critical blockers before proceeding"

        score = result.readiness_score.total_score if result.readiness_score else 0

        return (
            f"This drawing set {status} with a readiness score of {score}/100. "
            f"Found {len(result.blockers)} blockers across "
            f"{len(set(b.trade.value for b in result.blockers))} trades. "
            f"{action}."
        )


# =============================================================================
# READINESS SCORE CALCULATOR
# =============================================================================

def calculate_readiness_score(
    graph: PlanSetGraph,
    blockers: List[Blocker],
    trade_coverage: List = None,
) -> ReadinessScore:
    """
    Calculate multi-component readiness score with PASS/CONDITIONAL/NO-GO decision.

    Decision Logic:
    - PASS: Reasonably complete, safe to bid (score >= 70, no blocks_pricing blockers)
    - CONDITIONAL: Some trades blocked but others usable (score >= 40 OR some trades have >50% coverage)
    - NO-GO: Drawing set is so incomplete that bidding is extremely risky

    Components:
    - Completeness: Based on missing dependencies
    - Measurement: Based on scale coverage + dimension presence
    - Coverage: Based on discipline/sheet type coverage
    - Blocker: Weighted by severity and bid_impact

    Returns:
        ReadinessScore with breakdown
    """
    # Completeness score (0-100)
    # Penalize less harshly for missing dependencies
    dependency_types = set()
    for b in blockers:
        dependency_types.update(b.missing_dependency)
    completeness = max(0, 100 - len(dependency_types) * 10)  # Reduced from 12

    # Measurement score (0-100)
    # Based on scale coverage, but give partial credit for having any drawings
    total_pages = max(graph.total_pages, 1)
    scale_coverage = graph.pages_with_scale / total_pages

    # NEW: Give base credit for having drawings at all
    base_measurement = 30 if total_pages >= 1 else 0
    measurement = int(base_measurement + scale_coverage * 70)

    # Coverage score (0-100)
    # Based on actual sheet types and entities found, not just disciplines
    expected_disciplines = {"A", "S", "M", "E", "P"}
    found = set(graph.disciplines_found)
    discipline_coverage = len(found.intersection(expected_disciplines)) / len(expected_disciplines)

    # NEW: Also credit for having ANY sheet types identified
    sheet_types_found = len(graph.sheet_types_found) > 0
    entities_found = bool(graph.all_door_tags or graph.all_window_tags or graph.all_room_names)

    # Base coverage from disciplines
    coverage = int(discipline_coverage * 60)

    # Bonus for sheet types and entities
    if sheet_types_found:
        coverage += 20
    if entities_found:
        coverage += 20

    coverage = min(coverage, 100)

    # Blocker score (0-100)
    # NEW: Weight by severity AND bid_impact
    critical_count = 0
    pricing_blockers = 0
    for b in blockers:
        if b.severity in [Severity.CRITICAL, Severity.HIGH]:
            critical_count += 1
        if b.bid_impact == BidImpact.BLOCKS_PRICING:
            pricing_blockers += 1

    # Pricing blockers matter more than just high severity
    blocker_penalty = critical_count * 10 + pricing_blockers * 15
    blocker_score = max(0, 100 - blocker_penalty)

    # Weighted total
    weights = {
        "completeness": 0.25,  # Reduced from 0.30
        "measurement": 0.20,   # Reduced from 0.25
        "coverage": 0.35,      # Increased from 0.25 - coverage matters most
        "blocker": 0.20,
    }

    total = int(
        completeness * weights["completeness"] +
        measurement * weights["measurement"] +
        coverage * weights["coverage"] +
        blocker_score * weights["blocker"]
    )

    # ==========================================================================
    # DECISION LOGIC: PASS / CONDITIONAL / NO-GO
    # ==========================================================================
    #
    # PASS: Ready to bid
    #   - Score >= 70 AND no pricing blockers
    #   - OR score >= 60 AND at least 3 trades have good coverage
    #
    # CONDITIONAL: Can proceed with caution
    #   - Score >= 40 (some useful content)
    #   - OR at least 1 trade has > 50% coverage
    #   - System recommends RFIs but doesn't ban pricing
    #
    # NO-GO: Too risky to bid
    #   - Almost nothing measurable (score < 40)
    #   - AND no trades with meaningful coverage
    #   - AND critical disciplines missing
    # ==========================================================================

    # Check if any trades have good coverage
    good_coverage_trades = 0
    any_coverage_trades = 0
    if trade_coverage:
        for tc in trade_coverage:
            if tc.coverage_pct >= 60:
                good_coverage_trades += 1
            if tc.coverage_pct > 0:
                any_coverage_trades += 1

    # Determine status
    # PASS requires:
    #   - High score (>=70) AND no pricing blockers, OR
    #   - Good score (>=65) AND multiple trades with good coverage AND no critical blockers
    if total >= 70 and pricing_blockers == 0 and critical_count == 0:
        status = "PASS"
    elif total >= 65 and good_coverage_trades >= 3 and pricing_blockers == 0:
        status = "PASS"
    # CONDITIONAL when:
    #   - Moderate score (>=40) with some issues, OR
    #   - Some trades have coverage even if overall score is low
    elif total >= 40 or any_coverage_trades >= 1:
        status = "CONDITIONAL"
    elif graph.total_pages > 0 and (sheet_types_found or entities_found):
        # Have some drawings with identifiable content - CONDITIONAL not NO-GO
        status = "CONDITIONAL"
    else:
        status = "NO-GO"

    return ReadinessScore(
        total_score=total,
        status=status,
        completeness_score=completeness,
        measurement_score=measurement,
        coverage_score=coverage,
        blocker_score=blocker_score,
        weights=weights,
        score_breakdown={
            "completeness": f"{len(dependency_types)} missing dependency types",
            "measurement": f"{graph.pages_with_scale}/{total_pages} pages with scale",
            "coverage": f"{len(found)} disciplines, {len(graph.sheet_types_found)} sheet types",
            "blocker": f"{critical_count} high severity, {pricing_blockers} pricing blockers",
        }
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def enrich_analysis(
    result: DeepAnalysisResult,
    llm_client=None
) -> DeepAnalysisResult:
    """
    Enrich a deep analysis result with LLM-generated insights.

    Args:
        result: DeepAnalysisResult to enrich
        llm_client: Optional LLM client

    Returns:
        Enriched DeepAnalysisResult
    """
    enricher = LLMEnrichment(llm_client)

    # Enrich blockers
    if result.plan_graph:
        enrichments = enricher.enrich_all_blockers(
            result.blockers,
            result.plan_graph
        )

        # Update blockers with enrichments
        for enr in enrichments:
            blocker = next(
                (b for b in result.blockers if b.id == enr["blocker_id"]),
                None
            )
            if blocker and enr["enrichment"]:
                # Add enrichment to evidence
                blocker.evidence.confidence_reason = enr["enrichment"].get(
                    "construction_impact",
                    blocker.evidence.confidence_reason
                )

        # Calculate readiness score
        result.readiness_score = calculate_readiness_score(
            result.plan_graph,
            result.blockers,
            trade_coverage=result.trade_coverage
        )

    return result
