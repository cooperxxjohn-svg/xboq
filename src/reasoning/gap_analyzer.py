"""
src/reasoning/gap_analyzer.py

Two-pass gap detection over a bid package payload.

Pass 1 (rule-based, always runs — no LLM required):
  - Missing BOQ for detected trades
  - Unmatched schedule marks (doors/windows with no BOQ item)
  - CRITICAL-severity RFIs/blockers
  - Missing soil/geotech report when structural work detected
  - High-value items with no rate applied

Pass 2 (LLM, runs only if llm_client is provided):
  - Semantic context built from top ChromaDB chunks
  - Prompt Claude/GPT for additional gaps not caught by rules
  - JSON-parse response into Gap objects
  - Deduplicate against Pass 1 results
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    id: str                          # "GAP-001"
    trade: str
    severity: str                    # CRITICAL | HIGH | MEDIUM | LOW
    description: str
    evidence: List[str] = field(default_factory=list)
    action_required: str = ""
    cost_impact: Optional[str] = None
    source: str = "rule"             # "rule" | "llm"


# ── Trade keywords for classification ─────────────────────────────────────────
_TRADE_KEYWORDS: Dict[str, List[str]] = {
    # More-specific trades first so they win over broader ones
    "waterproofing": ["waterproof", "membrane", "crystalline", "tanking", "sump"],
    "painting": ["paint", "odb", "weather shield", "primer", "coating"],
    "structural": ["rcc", "concrete", "footing", "column", "beam", "slab", "pile",
                   "structural", "foundation", "reinforcement", "rebar"],
    "civil": ["excavation", "earthwork", "filling", "site", "compound", "pcc",
              "plinth", "hard core", "anti-termite"],
    "mep": ["electrical", "plumbing", "hvac", "fire", "mep", "duct", "pipe",
            "conduit", "cable", "sanitary"],
    "architectural": ["door", "window", "finish", "plaster", "tile", "floor",
                      "ceiling", "dado", "wall", "cladding"],
}


def _classify_trade(text: str) -> str:
    t = text.lower()
    for trade, kws in _TRADE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return trade
    return "general"


# ── Rule-based gap detection ───────────────────────────────────────────────────

def _rule_gaps(payload: dict) -> List[Gap]:
    gaps: List[Gap] = []
    counter = [0]

    def _g(trade, severity, description, evidence=None, action="", cost_impact=None):
        counter[0] += 1
        return Gap(
            id=f"GAP-{counter[0]:03d}",
            trade=trade,
            severity=severity,
            description=description,
            evidence=evidence or [],
            action_required=action,
            cost_impact=cost_impact,
            source="rule",
        )

    qto = payload.get("qto_summary", {})
    boq_items = payload.get("boq_items") or []
    spec_items = payload.get("spec_items") or []
    rfis = payload.get("rfis") or []
    blockers = payload.get("blockers") or []
    extraction = payload.get("extraction_summary", {})
    schedules = extraction.get("schedules", [])
    schedule_boq = payload.get("schedule_boq_links", {})

    all_items = boq_items + spec_items
    trades_in_items = {_classify_trade(i.get("description", "")) for i in all_items}

    # ── R1: CRITICAL RFIs become CRITICAL gaps ─────────────────────────────
    for rfi in rfis:
        sev = str(rfi.get("severity", "")).upper()
        if sev == "CRITICAL":
            gaps.append(_g(
                trade=str(rfi.get("trade", "general")),
                severity="CRITICAL",
                description=f"Unresolved critical RFI: {rfi.get('question', rfi.get('description',''))}",
                evidence=[f"Page {rfi.get('source_page', '?')}"],
                action="Raise formal RFI to client and await clarification before pricing",
            ))

    # ── R2: CRITICAL blockers ──────────────────────────────────────────────
    for blk in blockers:
        sev = str(blk.get("severity", "")).upper()
        if sev == "CRITICAL":
            gaps.append(_g(
                trade=str(blk.get("trade", "general")),
                severity="CRITICAL",
                description=f"Critical blocker: {blk.get('description', blk.get('title', ''))}",
                evidence=[f"Page {blk.get('source_page', '?')}"],
                action=str(blk.get("fix_action") or "Resolve before submitting bid"),
            ))

    # ── R3: Structural work detected but no geotech/soil report ───────────
    has_structural = (
        qto.get("st_concrete_cum", 0) > 0
        or qto.get("st_steel_kg", 0) > 0
        or "structural" in trades_in_items
    )
    page_texts = payload.get("ocr_text_by_page", {})
    geotech_found = any(
        re.search(r"geotech|soil\s+report|borehole|spt\s+test|sub.?soil", t, re.IGNORECASE)
        for t in page_texts.values()
    )
    if has_structural and not geotech_found:
        gaps.append(_g(
            trade="structural",
            severity="HIGH",
            description="No geotechnical / soil investigation report found in bid package",
            action="Request soil report from client. Add ₹2-8L provisional sum for investigation if not provided.",
            cost_impact="₹2-8L provisional sum + risk contingency on foundation",
        ))

    # ── R4: Unmatched schedule marks (door/window schedule items with no BOQ) ─
    unmatched = schedule_boq.get("unmatched_schedule_marks", [])
    if unmatched:
        gaps.append(_g(
            trade="architectural",
            severity="HIGH",
            description=f"{len(unmatched)} schedule mark(s) have no matching BOQ item: "
                        f"{', '.join(str(m) for m in unmatched[:5])}{'...' if len(unmatched) > 5 else ''}",
            action="Price each unmatched mark separately or raise RFI for missing BOQ line items",
            cost_impact="Allowance required per unmatched mark",
        ))

    # ── R5: High-value items with no rate applied ──────────────────────────
    unrated_high = [
        i for i in all_items
        if i.get("qty") and float(i.get("qty") or 0) > 0
        and (i.get("rate_inr") or 0) == 0
        and float(i.get("qty") or 0) * 5000 > 500_000  # ~₹5L threshold
    ]
    if unrated_high:
        gaps.append(_g(
            trade="general",
            severity="MEDIUM",
            description=f"{len(unrated_high)} high-quantity items have no rate applied (potential ₹5L+ exposure each)",
            action="Obtain subcontractor quotes or apply market rates for unrated items",
        ))

    # ── R6: No BOQ / spec items at all ────────────────────────────────────
    if not all_items:
        gaps.append(_g(
            trade="general",
            severity="CRITICAL",
            description="No BOQ or specification items extracted from this bid package",
            action="Verify that the uploaded document is a valid tender/bid package",
        ))

    # ── R7: MEP detected in drawings but no MEP BOQ items ─────────────────
    plan_graph = payload.get("diagnostics", {}).get("plan_graph") or {}
    disciplines = plan_graph.get("disciplines_found", [])
    mep_disciplines = {"electrical", "plumbing", "hvac", "mechanical", "fire"}
    mep_in_drawings = bool(mep_disciplines & {d.lower() for d in disciplines})
    mep_in_items = "mep" in trades_in_items
    if mep_in_drawings and not mep_in_items:
        gaps.append(_g(
            trade="mep",
            severity="HIGH",
            description="MEP/services drawings detected but no MEP items in BOQ/specifications",
            action="Request MEP BOQ from specialist sub-contractor or raise RFI for MEP scope",
            cost_impact="3-8% of project cost typical for MEP services",
        ))

    # ── R8: Missing finishing schedule ────────────────────────────────────
    has_finish_schedule = plan_graph.get("has_finish_schedule", False)
    has_finish_items = any(
        "finish" in i.get("trade", "").lower() or "tile" in i.get("description", "").lower()
        for i in all_items
    )
    if not has_finish_schedule and not has_finish_items and all_items:
        gaps.append(_g(
            trade="finishes",
            severity="MEDIUM",
            description="No finish schedule found; finishing items may be under-specified",
            action="Request finishes schedule from architect or use provisional sums for flooring/wall finishes",
        ))

    # ── R9: No commercial terms extracted ─────────────────────────────────
    commercial = payload.get("contractual_items") or []
    if not commercial:
        gaps.append(_g(
            trade="commercial",
            severity="MEDIUM",
            description="No commercial terms (LD, retention, payment terms) extracted from package",
            action="Review tender conditions document manually to identify penalty and payment risk",
        ))

    # ── R10: Low bid readiness score ──────────────────────────────────────
    readiness = payload.get("readiness_score", payload.get("qa_score", 100))
    if isinstance(readiness, (int, float)) and readiness < 50:
        gaps.append(_g(
            trade="general",
            severity="HIGH",
            description=f"Overall bid readiness score is low ({readiness}/100) — significant information is missing",
            action="Review all HIGH/CRITICAL gaps before proceeding. Consider conditional bid or requesting more information.",
        ))

    return gaps


# ── LLM-based gap detection ────────────────────────────────────────────────────

_LLM_GAP_PROMPT = """\
You are a senior quantity surveyor reviewing a construction bid package.
Your job: identify ALL gaps that would prevent accurate pricing.

Extracted bid content:
---
{context}
---

Return ONLY valid JSON (no prose), in this format:
{{
  "gaps": [
    {{
      "trade": "structural",
      "severity": "HIGH",
      "description": "Concrete grade not specified for columns",
      "action": "Raise RFI to confirm concrete grade before pricing"
    }}
  ]
}}

Severity levels: CRITICAL (blocks pricing), HIGH (significant risk), MEDIUM (clarification needed), LOW (minor).
Focus on: missing drawings, unspecified materials, ambiguous quantities, conflicting dimensions,
missing schedule items, unclear scope boundaries between trades.
Return at most 15 gaps. Return empty list if none found.
"""


def _llm_gaps(payload: dict, store: Any, embedder: Any, llm_client: Any) -> List[Gap]:
    """Run LLM-based gap detection using ChromaDB context."""
    # Trade-scoped gap queries for better precision (R5)
    _gap_query_config = [
        ("structural system, columns, beams, RCC grades, steel reinforcement", "structural"),
        ("MEP mechanical electrical plumbing services drawings specifications", "mep"),
        ("finishes flooring tiles painting waterproofing specifications", "finishing"),
        ("external development roads compound wall site utilities", "external"),
        ("commercial terms payment milestone retention performance bank guarantee", "commercial"),
        # Cross-cutting broad queries (no trade filter)
        ("scope of work inclusions exclusions boundaries contractor responsibility", None),
        ("missing drawings schedule incomplete specification ambiguous clause", None),
        ("geotechnical soil report foundation design criteria seismic zone", None),
    ]

    _tender_chunks: List[str] = []
    for _q, _trade in _gap_query_config:
        if _trade:
            _hits = store.search_by_trade(_q, _trade, embedder, n_results=15) if hasattr(store, 'search_by_trade') else store.search(_q, embedder, n_results=15)
        else:
            _hits = store.search(_q, embedder, n_results=15)
        for _h in (_hits or []):
            _score = _h.get("score", 0) if isinstance(_h, dict) else getattr(_h, "score", 0)
            _text  = _h.get("text", "") or _h.get("document", "") if isinstance(_h, dict) else getattr(_h, "text", "")
            if _score >= 0.20 and _text not in _tender_chunks:
                _tender_chunks.append(_text)
    _tender_chunks = _tender_chunks[:40]  # increased from 30

    # Augment with domain knowledge (DSR rates, IS codes, taxonomy) using R4 context assembly
    _domain_chunks: list = []
    try:
        from src.embeddings.kb_interface import get_kb as _get_kb
        _domain_store = _get_kb()
        if _domain_store.count() > 0:
            # Build trade-scoped queries from top gap query config
            _trade_queries = {
                _trade: _dq
                for _dq, _trade in _gap_query_config[:5]
                if _trade is not None
            }
            if not _trade_queries:
                _trade_queries = {"structural": _gap_query_config[0][0]}
            # Use assemble_context for structured multi-trade retrieval (R4)
            _ctx_frames = _domain_store.assemble_context(
                _trade_queries,
                n_per_trade=4,
                include_cross_deps=True,
                max_chars=3000,
            )
            _formatted = _domain_store.format_context_for_llm(_ctx_frames, max_chars=2000)
            if _formatted:
                _domain_chunks = [_formatted]
            # Also do direct searches for critical queries (no trade filter)
            for _dq, _ in _gap_query_config[5:]:  # cross-cutting queries
                _dk_hits = _domain_store.search(_dq, n_results=3)
                for _h in _dk_hits:
                    if _h.get("score", 0) >= 0.30:
                        _domain_chunks.append(f"[DOMAIN: {_h.get('source','kb')}] {_h['text']}")
            _domain_chunks = list(dict.fromkeys(_domain_chunks))[:8]  # dedup + cap
    except Exception as _dk_err:
        pass  # domain KB is best-effort

    if not _tender_chunks:
        return []

    # Structured context assembly (R4)
    _context_parts = []
    if _tender_chunks:
        _context_parts.append("=== TENDER DOCUMENT EXTRACTS ===")
        _context_parts.extend(f"[{i+1}] {c}" for i, c in enumerate(_tender_chunks[:20]))
    if _domain_chunks:
        _context_parts.append("\n=== DOMAIN KNOWLEDGE (DSR rates / IS codes / taxonomy) ===")
        _context_parts.extend(f"[KB] {c}" for c in _domain_chunks[:10])
    _context_str = "\n".join(_context_parts)

    prompt = _LLM_GAP_PROMPT.format(context=_context_str)

    raw = ""
    try:
        from src.utils.llm_caller import call_llm
        raw = call_llm(
            client=llm_client,
            system="You are an expert construction estimator analyzing tender documents.",
            user=prompt,
            max_tokens=4000,
            temperature=0.1,
            openai_model="gpt-4o-mini",
            anthropic_model="claude-haiku-4-5-20251001",
        )
    except Exception as e:
        logger.warning("LLM gap detection failed: %s", e)
        return []

    # Parse JSON
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        raw_gaps = data.get("gaps", [])
    except Exception:
        logger.warning("Failed to parse LLM gap response")
        return []

    gaps: List[Gap] = []
    for i, g in enumerate(raw_gaps[:15]):
        desc = str(g.get("description") or "").strip()
        if not desc:
            continue
        gaps.append(Gap(
            id=f"GLLM-{i + 1:03d}",
            trade=str(g.get("trade", "general")),
            severity=str(g.get("severity", "MEDIUM")).upper(),
            description=desc,
            action_required=str(g.get("action", "")),
            source="llm",
        ))
    return gaps


def _dedup_gaps(rule_gaps: List[Gap], llm_gaps: List[Gap]) -> List[Gap]:
    """
    Remove LLM gaps that are near-duplicates of rule gaps.
    Similarity: same trade and >40% word overlap in description.
    """
    def _words(s: str) -> set:
        return set(re.findall(r"\w{4,}", s.lower()))

    merged = list(rule_gaps)
    for lg in llm_gaps:
        lg_words = _words(lg.description)
        is_dup = False
        for rg in rule_gaps:
            if rg.trade != lg.trade:
                continue
            rg_words = _words(rg.description)
            if not rg_words:
                continue
            overlap = len(lg_words & rg_words) / max(len(rg_words), 1)
            if overlap > 0.4:
                is_dup = True
                break
        if not is_dup:
            merged.append(lg)

    # Renumber all gaps sequentially
    for i, g in enumerate(merged):
        g.id = f"GAP-{i + 1:03d}"

    return merged


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_gaps(
    payload: dict,
    store: Any = None,
    embedder: Any = None,
    llm_client: Any = None,
) -> List[Gap]:
    """
    Identify bid gaps in two passes.

    Pass 1: rule-based (always runs).
    Pass 2: LLM-based (only if llm_client and store/embedder available).

    Returns list of Gap objects sorted by severity.
    """
    _SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    rule_gaps = _rule_gaps(payload)

    llm_gaps: List[Gap] = []
    if llm_client is not None and store is not None and embedder is not None:
        try:
            llm_gaps = _llm_gaps(payload, store, embedder, llm_client)
        except Exception as e:
            logger.warning("LLM gap pass failed: %s", e)

    all_gaps = _dedup_gaps(rule_gaps, llm_gaps)
    all_gaps.sort(key=lambda g: (_SEV_ORDER.get(g.severity, 9), g.trade))

    logger.info(
        "Gap analysis: %d rule gaps + %d LLM gaps → %d total after dedup",
        len(rule_gaps), len(llm_gaps), len(all_gaps),
    )
    return all_gaps
