"""
Ask Tender — Hybrid RAG for construction tender analysis.

Sprint 24 Phase 4: Deterministic keyword/pattern matching against the
analysis payload.  Structured lookups with a thin intent-detection layer
that maps natural-language questions to the right payload sections.

Sprint 29: Added TF-IDF vector embedding + semantic search layer.
When structured handlers return answers, RAG supplements with semantically
relevant context.  When handlers return nothing, RAG provides vector-based
semantic fallback before basic OCR keyword search.

Usage:
    # CLI
    python -m src ask --input out/sonipat/analysis.json -q "What trades are blocked?"

    # Python
    from src.ask_tender import ask, load_payload
    payload = load_payload("out/sonipat/analysis.json")
    results = ask(payload, "How many RFIs?")
"""

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ANSWER SCHEMA
# =============================================================================

@dataclass
class Answer:
    """A single answer chunk returned by the ask engine."""
    source: str          # e.g., "rfis", "blockers", "trade_coverage"
    title: str           # short heading
    body: str            # human-readable answer text
    confidence: float    # 0.0 – 1.0
    evidence: List[str] = field(default_factory=list)   # page refs, snippets

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AskResult:
    """Full result from an ask() call."""
    query: str
    intent: str               # detected intent label
    answers: List[Answer] = field(default_factory=list)
    fallback_search: bool = False   # True when we fell back to OCR text search

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "answer_count": len(self.answers),
            "fallback_search": self.fallback_search,
            "answers": [a.to_dict() for a in self.answers],
        }

    def summary(self) -> str:
        """One-line summary for CLI output."""
        if not self.answers:
            return f"No answers found for: {self.query}"
        top = self.answers[0]
        return f"[{top.source}] {top.title}: {top.body[:120]}"


# =============================================================================
# INTENT DETECTION — map question → payload sections
# =============================================================================

# Each intent: (label, regex pattern, handler function name, payload keys used)
INTENT_PATTERNS: List[Tuple[str, str, str]] = [
    # Blocker / blocking questions
    ("blockers",
     r"(block|blocker|blocking|what.*block|critical\s+issue|showstopper)",
     "_answer_blockers"),

    # RFI questions
    ("rfis",
     r"(rfi|request\s+for\s+information|clarification|question.*tender|how\s+many\s+rfi)",
     "_answer_rfis"),

    # Trade / coverage questions
    ("trade_coverage",
     r"(trade|coverage|discipline|what\s+trade|which\s+trade|how.*cover|mep|civil|structural|architect|electri|plumb|finish)",
     "_answer_trade_coverage"),

    # BOQ / quantity questions
    ("quantities",
     r"(boq|bill\s+of\s+quantit|quantity|quantities|how\s+many\s+item|line\s+item|measure)",
     "_answer_quantities"),

    # Commercial / contract terms
    ("commercial",
     r"(commercial|contract\s+term|payment|retention|warranty|defect|liabilit|\bld\b|liquidat|penalty|\badvance\b|\bemd\b|pbg|bank\s+guarantee)",
     "_answer_commercial"),

    # Drawing / page overview
    ("drawings",
     r"(drawing|page|sheet|plan|elevation|section|how\s+many\s+page|document|pdf|scan|floor\s+plan)",
     "_answer_drawings"),

    # Schedule / door / window
    ("schedules",
     r"(schedule|door|window|finish|room|opening|mark|tag|D-?\d|W-?\d)",
     "_answer_schedules"),

    # Structural
    ("structural",
     r"(structural|rcc|reinforc|concrete|steel|rebar|foundation|footing|beam|column|slab)",
     "_answer_structural"),

    # Score / readiness / decision
    ("readiness",
     r"(score|readiness|ready|decision|go\s*\/?\s*no.?go|conditional|bid.*ready|can\s+we\s+bid|should\s+we\s+bid|qa|quality)",
     "_answer_readiness"),

    # Requirements / specs
    ("requirements",
     r"(requirement|specification|spec|standard|approved\s+make|material|is\s+200|is\s+456|nbc|isi)",
     "_answer_requirements"),

    # Next steps / action items
    ("next_steps",
     r"(what.*do|next\s+step|priorit|action|todo|recommend|first\s+thing|where.*start|how.*proceed)",
     "_answer_next_steps"),

    # Project overview / summary
    ("overview",
     r"(overview|summary|tell\s+me\s+about|what\s+is\s+this|project\s+info|tender\s+info|at\s+a\s+glance)",
     "_answer_overview"),
]


def detect_intent(query: str) -> Tuple[str, str]:
    """
    Detect the user's intent from their natural language query.

    Returns:
        (intent_label, handler_function_name)
    """
    q = query.lower().strip()
    for label, pattern, handler in INTENT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return label, handler
    return "general", "_answer_general"


# =============================================================================
# ANSWER HANDLERS — one per intent
# =============================================================================

def _answer_blockers(payload: dict, query: str) -> List[Answer]:
    blockers = payload.get("blockers", [])
    if not blockers:
        return [Answer(
            source="blockers", title="No blockers found",
            body="The analysis did not detect any blocking issues.",
            confidence=0.9,
        )]
    answers = []
    for b in blockers:
        sev = b.get("severity", "MEDIUM")
        title = b.get("title", b.get("blocker_id", "Blocker"))
        desc = b.get("description", "")
        fix = b.get("fix_actions", "")
        if isinstance(fix, list):
            fix = "; ".join(fix)
        body = f"[{sev}] {desc}"
        if fix:
            body += f"\nFix: {fix}"
        evidence = []
        ev = b.get("evidence", {})
        if isinstance(ev, dict):
            pages = ev.get("pages", [])
            if pages:
                evidence.append(f"Pages: {', '.join(str(p) for p in pages[:5])}")
        answers.append(Answer(
            source="blockers", title=title, body=body,
            confidence=0.9, evidence=evidence,
        ))
    return answers


def _answer_rfis(payload: dict, query: str) -> List[Answer]:
    rfis = payload.get("rfis", [])
    if not rfis:
        return [Answer(
            source="rfis", title="No RFIs generated",
            body="The analysis did not generate any Requests for Information.",
            confidence=0.9,
        )]

    # If asking "how many", give a count answer first
    q = query.lower()
    answers = []
    if re.search(r"how\s+many|count|total|number", q):
        by_trade = {}
        for r in rfis:
            t = r.get("trade", "general")
            by_trade[t] = by_trade.get(t, 0) + 1
        breakdown = ", ".join(f"{t}: {c}" for t, c in sorted(by_trade.items()))
        answers.append(Answer(
            source="rfis", title=f"{len(rfis)} RFIs generated",
            body=f"Total: {len(rfis)} RFIs across {len(by_trade)} trades.\nBreakdown: {breakdown}",
            confidence=1.0,
        ))

    # Filter by trade if mentioned
    trade_filter = None
    for t in ["civil", "structural", "architectural", "mep", "electrical",
              "plumbing", "finishes", "general", "commercial"]:
        if t in q:
            trade_filter = t
            break

    shown = 0
    for r in rfis:
        if trade_filter and r.get("trade", "").lower() != trade_filter:
            continue
        if shown >= 5:
            break
        question = r.get("question", r.get("rfi_text", ""))
        trade = r.get("trade", "general")
        priority = r.get("priority", "MEDIUM")
        why = r.get("why_it_matters", "")
        body = question
        if why:
            body += f"\nWhy it matters: {why}"
        evidence = []
        ev = r.get("evidence", {})
        if isinstance(ev, dict):
            pages = ev.get("pages", [])
            if pages:
                evidence.append(f"Pages: {', '.join(str(p) for p in pages[:5])}")
        answers.append(Answer(
            source="rfis", title=f"[{trade}/{priority}] RFI",
            body=body, confidence=0.85, evidence=evidence,
        ))
        shown += 1

    if trade_filter and shown == 0:
        answers.append(Answer(
            source="rfis", title=f"No {trade_filter} RFIs",
            body=f"No RFIs found for {trade_filter} trade.",
            confidence=0.8,
        ))

    return answers


def _answer_trade_coverage(payload: dict, query: str) -> List[Answer]:
    tc = payload.get("trade_coverage", [])
    if not tc:
        return [Answer(
            source="trade_coverage", title="No trade coverage data",
            body="Trade coverage analysis was not performed.",
            confidence=0.5,
        )]

    answers = []
    q = query.lower()

    # Check if asking about a specific trade
    specific_trade = None
    for t in ["civil", "structural", "architectural", "mep", "electrical",
              "plumbing", "finishes", "general"]:
        if t in q:
            specific_trade = t
            break

    for tc_item in tc:
        trade = tc_item.get("trade", "")
        if specific_trade and specific_trade not in trade.lower():
            continue
        cov = tc_item.get("coverage_pct", 0)
        status = "OK" if cov >= 80 else "NEEDS ATTENTION" if cov >= 50 else "BLOCKED"
        missing = tc_item.get("missing_dependencies", [])
        if isinstance(missing, list):
            missing_str = ", ".join(missing) if missing else "none"
        else:
            missing_str = str(missing)
        body = f"Coverage: {cov:.0f}% [{status}]"
        if missing_str != "none":
            body += f"\nMissing: {missing_str}"
        answers.append(Answer(
            source="trade_coverage", title=f"Trade: {trade}",
            body=body, confidence=0.9,
        ))

    if not answers:
        return [Answer(
            source="trade_coverage", title="Trade not found",
            body=f"No coverage data for requested trade.",
            confidence=0.5,
        )]
    return answers


def _answer_quantities(payload: dict, query: str) -> List[Answer]:
    answers = []
    es = payload.get("extraction_summary") or {}
    counts = es.get("counts") or {}
    boq_items = counts.get("boq_items", 0)

    # Summary count
    answers.append(Answer(
        source="quantities", title=f"{boq_items} BOQ items extracted",
        body=f"BOQ items: {boq_items}, Schedules: {counts.get('schedules', 0)}",
        confidence=0.95,
    ))

    # BOQ stats if available
    boq_stats = payload.get("boq_stats", {})
    if boq_stats:
        by_trade = boq_stats.get("by_trade", {})
        if by_trade:
            breakdown = ", ".join(f"{t}: {c}" for t, c in by_trade.items())
            answers.append(Answer(
                source="quantities", title="BOQ by trade",
                body=breakdown, confidence=0.85,
            ))

    # Quantities list (top items)
    quantities = payload.get("quantities", [])
    if quantities and isinstance(quantities, list):
        q_lower = query.lower()
        # Filter by keyword if present
        filtered = quantities
        search_terms = re.findall(r'\b\w{4,}\b', q_lower)
        for term in search_terms:
            if term in ("many", "item", "quantity", "total", "list",
                        "what", "show", "give", "tell"):
                continue
            matched = [q for q in filtered
                       if term in str(q.get("item", "")).lower()
                       or term in str(q.get("trade", "")).lower()]
            if matched:
                filtered = matched

        for qi in filtered[:5]:
            item = qi.get("item", "Unknown")
            qty = qi.get("qty", "?")
            unit = qi.get("unit", "")
            trade = qi.get("trade", "")
            conf = qi.get("confidence", 0.5)
            answers.append(Answer(
                source="quantities", title=f"{item}",
                body=f"Qty: {qty} {unit} | Trade: {trade}",
                confidence=conf,
            ))

    return answers


def _answer_commercial(payload: dict, query: str) -> List[Answer]:
    # Sprint 26: Check both top-level and extraction_summary locations
    terms = payload.get("commercial_terms") or []
    if not isinstance(terms, list):
        terms = []
    if not terms:
        es = payload.get("extraction_summary") or {}
        terms = es.get("commercial_terms") or []
        if not isinstance(terms, list):
            terms = []
    if not terms:
        return [Answer(
            source="commercial", title="No commercial terms extracted",
            body="The analysis did not extract any commercial/contract terms.",
            confidence=0.7,
        )]

    answers = []
    q = query.lower()

    # Add summary count if asking generally
    if not re.search(r"\bld\b|liquidat|retention|defect|liabilit|warranty|dlp|\badvance\b|mobili|\bemd\b|penalty|insurance", q):
        term_types = [t.get("term_type", "?") for t in terms if isinstance(t, dict)]
        answers.append(Answer(
            source="commercial",
            title=f"{len(term_types)} commercial terms found",
            body=f"Types: {', '.join(t.replace('_', ' ').title() for t in term_types)}",
            confidence=0.95,
        ))

    for t in terms:
        if not isinstance(t, dict):
            continue
        term_type = t.get("term_type", "unknown")
        value = t.get("value", "")
        snippet = t.get("snippet", "")
        page = t.get("page_idx", "?")

        # If user asked about specific term type, filter
        if re.search(r"\bld\b|liquidat", q) and "ld" not in term_type:
            continue
        if re.search(r"retention", q) and "retention" not in term_type:
            continue
        if re.search(r"defect|liabilit|warranty|dlp", q) and "defect" not in term_type and "warranty" not in term_type and "dlp" not in term_type:
            continue
        if re.search(r"\badvance\b|mobili", q) and "advance" not in term_type and "mobili" not in term_type:
            continue

        body = f"{term_type}: {value}"
        evidence = []
        if snippet:
            evidence.append(snippet[:150])
        if page != "?":
            evidence.append(f"Page {page}")
        answers.append(Answer(
            source="commercial", title=term_type.replace("_", " ").title(),
            body=body, confidence=t.get("confidence", 0.7),
            evidence=evidence,
        ))

    if not answers:
        return [Answer(
            source="commercial", title=f"{len(terms)} commercial terms found",
            body="Terms found but none match your specific query. Try a broader question.",
            confidence=0.5,
        )]
    return answers


def _answer_drawings(payload: dict, query: str) -> List[Answer]:
    answers = []
    overview = payload.get("drawing_overview") or {}
    pi = (payload.get("diagnostics") or {}).get("page_index", {})

    total = pi.get("total_pages", overview.get("total_pages", 0))
    counts_by_type = pi.get("counts_by_type", {})
    disciplines = overview.get("disciplines_detected", [])

    # Summary
    type_breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(counts_by_type.items()) if v > 0)
    answers.append(Answer(
        source="drawings", title=f"{total} pages analyzed",
        body=f"Page types: {type_breakdown}",
        confidence=0.95,
    ))

    if disciplines:
        answers.append(Answer(
            source="drawings", title=f"{len(disciplines)} disciplines detected",
            body=f"Disciplines: {', '.join(disciplines)}",
            confidence=0.9,
        ))

    # Sheet types
    sheet_types = overview.get("sheet_types_count", {})
    if sheet_types:
        st_str = ", ".join(f"{k}: {v}" for k, v in sorted(sheet_types.items()) if v > 0)
        answers.append(Answer(
            source="drawings", title="Sheet types",
            body=st_str, confidence=0.85,
        ))

    # Scale info
    scale_with = overview.get("pages_with_scale", 0)
    scale_without = overview.get("pages_without_scale", 0)
    if scale_with or scale_without:
        answers.append(Answer(
            source="drawings", title="Scale detection",
            body=f"{scale_with} pages with scale, {scale_without} without",
            confidence=0.85,
        ))

    return answers


def _answer_schedules(payload: dict, query: str) -> List[Answer]:
    answers = []

    # Plan graph data (doors, windows, rooms)
    # Sprint 25: now populated in diagnostics.plan_graph via pipeline fix
    pg = payload.get("diagnostics", {}).get("plan_graph", {})
    if not pg:
        pg = payload.get("deep_analysis", {})

    door_tags = pg.get("all_door_tags", [])
    window_tags = pg.get("all_window_tags", [])
    room_names = pg.get("all_room_names", [])

    # Sprint 25: Cross-reference blockers for door/window counts when plan_graph empty
    if not door_tags or not window_tags:
        blockers = payload.get("blockers", [])
        for b in blockers:
            title = (b.get("title", "") or "").lower()
            desc = (b.get("description", "") or "").lower()
            if not door_tags and "door" in title:
                # Extract count from blocker description
                m = re.search(r'(\d+)\s+door\s+(?:type|tag|instance)', desc)
                count = m.group(1) if m else "?"
                answers.append(Answer(
                    source="blockers", title=f"Door schedule blocker",
                    body=f"{count} door types detected by blocker analysis. "
                         f"Door schedule is missing — see blockers for details.",
                    confidence=0.8,
                    evidence=[f"Blocker: {b.get('title', '')}"],
                ))
            if not window_tags and "window" in title:
                m = re.search(r'(\d+)\s+window\s+(?:type|tag|instance)', desc)
                count = m.group(1) if m else "?"
                answers.append(Answer(
                    source="blockers", title=f"Window schedule blocker",
                    body=f"{count} window types detected by blocker analysis. "
                         f"Window schedule is missing — see blockers for details.",
                    confidence=0.8,
                    evidence=[f"Blocker: {b.get('title', '')}"],
                ))

    q = query.lower()

    if re.search(r"door", q) or not re.search(r"window|room|finish", q):
        if door_tags:
            answers.append(Answer(
                source="schedules", title=f"{len(door_tags)} door types found",
                body=f"Door tags: {', '.join(door_tags)}",
                confidence=0.9,
            ))
        elif not any("Door" in a.title for a in answers):
            answers.append(Answer(
                source="schedules", title="No door tags in plan graph",
                body="No door types were detected in the plan graph.",
                confidence=0.5,
            ))

    if re.search(r"window", q) or not re.search(r"door|room|finish", q):
        if window_tags:
            answers.append(Answer(
                source="schedules", title=f"{len(window_tags)} window types found",
                body=f"Window tags: {', '.join(window_tags)}",
                confidence=0.9,
            ))

    if re.search(r"room|finish", q) or not re.search(r"door|window", q):
        if room_names:
            answers.append(Answer(
                source="schedules", title=f"{len(room_names)} room types found",
                body=f"Rooms: {', '.join(room_names[:15])}",
                confidence=0.85,
            ))

    # Schedule extraction counts
    es = payload.get("extraction_summary", {})
    sched_count = es.get("counts", {}).get("schedules", 0)
    answers.append(Answer(
        source="schedules", title=f"{sched_count} schedule rows extracted",
        body=f"Schedule rows: {sched_count}",
        confidence=0.9,
    ))

    return answers


def _answer_structural(payload: dict, query: str) -> List[Answer]:
    st_data = payload.get("structural_takeoff", {})
    if not st_data:
        return [Answer(
            source="structural", title="No structural takeoff data",
            body="Structural analysis was not performed or produced no results.",
            confidence=0.6,
        )]

    answers = []
    summary = st_data.get("summary", {})
    if summary:
        concrete = summary.get("total_concrete_m3", 0)
        steel = summary.get("total_steel_kg", 0)
        elements = summary.get("element_count", 0)
        body = f"Concrete: {concrete:.1f} m\u00b3, Steel: {steel:.0f} kg ({steel/1000:.1f} MT)"
        if elements:
            body += f", Elements: {elements}"
        answers.append(Answer(
            source="structural", title="Structural takeoff summary",
            body=body, confidence=0.85,
        ))

    warnings = st_data.get("warnings", [])
    if warnings:
        answers.append(Answer(
            source="structural", title=f"{len(warnings)} structural warnings",
            body="\n".join(f"- {w}" for w in warnings[:5]),
            confidence=0.75,
        ))

    return answers


def _answer_readiness(payload: dict, query: str) -> List[Answer]:
    answers = []
    score = payload.get("readiness_score", 0)
    if not isinstance(score, (int, float)):
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0
    decision = payload.get("decision", "UNKNOWN")
    sub = payload.get("sub_scores") or {}

    body = f"Score: {score}/100 \u2192 {decision}"
    if sub:
        parts = [f"{k}: {v}" for k, v in sub.items()]
        body += f"\nSub-scores: {', '.join(parts)}"
    answers.append(Answer(
        source="readiness", title=f"{decision} ({score}/100)",
        body=body, confidence=1.0,
    ))

    # QA score
    qa = payload.get("qa_score", {})
    if isinstance(qa, dict) and qa.get("score") is not None:
        qa_body = f"QA Score: {qa['score']}/100"
        components = qa.get("components", {})
        if components:
            qa_body += " | " + ", ".join(f"{k}: {v}" for k, v in components.items())
        answers.append(Answer(
            source="readiness", title="QA Score",
            body=qa_body, confidence=0.9,
        ))

    return answers


def _answer_requirements(payload: dict, query: str) -> List[Answer]:
    answers = []
    es = payload.get("extraction_summary", {})
    req_count = es.get("counts", {}).get("requirements", 0)

    answers.append(Answer(
        source="requirements", title=f"{req_count} requirements extracted",
        body=f"Total extracted requirements/specs: {req_count}",
        confidence=0.9,
    ))

    # Requirements by trade
    rbt = payload.get("requirements_by_trade") or {}
    if isinstance(rbt, dict) and rbt:
        parts = [f"{t}: {len(items) if isinstance(items, list) else items}"
                 for t, items in rbt.items()]
        answers.append(Answer(
            source="requirements", title="Requirements by trade",
            body=", ".join(parts), confidence=0.85,
        ))

    return answers


def _answer_next_steps(payload: dict, query: str) -> List[Answer]:
    """Synthesize a prioritized action list from blockers, RFIs, and guidance."""
    answers = []
    blockers = payload.get("blockers") or []
    if not isinstance(blockers, list):
        blockers = []
    rfis = payload.get("rfis") or []
    if not isinstance(rfis, list):
        rfis = []
    pricing = payload.get("pricing_guidance", {})
    decision = payload.get("decision", "UNKNOWN")
    score = payload.get("readiness_score", 0)

    # Step 1: Decision context
    answers.append(Answer(
        source="next_steps", title=f"Decision: {decision} ({score}/100)",
        body=f"Current readiness: {score}/100 → {decision}. "
             f"Focus on resolving blockers to improve score.",
        confidence=1.0,
    ))

    # Step 2: HIGH blockers
    high_blockers = [b for b in blockers if b.get("severity", "").lower() == "high"]
    if high_blockers:
        items = []
        for i, b in enumerate(high_blockers, 1):
            fix = b.get("fix_actions", "")
            if isinstance(fix, list):
                fix = fix[0] if fix else ""
            items.append(f"{i}. {b.get('title', 'Blocker')}: {fix}")
        answers.append(Answer(
            source="next_steps",
            title=f"Priority 1: Resolve {len(high_blockers)} HIGH blockers",
            body="\n".join(items),
            confidence=0.95,
        ))

    # Step 3: Send RFIs
    if rfis:
        by_trade = {}
        for r in rfis:
            t = r.get("trade", "general")
            by_trade[t] = by_trade.get(t, 0) + 1
        breakdown = ", ".join(f"{t}: {c}" for t, c in sorted(by_trade.items()))
        answers.append(Answer(
            source="next_steps",
            title=f"Priority 2: Send {len(rfis)} RFIs to client",
            body=f"Trades: {breakdown}.\nSend before pricing to get clarifications back in time.",
            confidence=0.9,
        ))

    # Step 4: Medium blockers
    med_blockers = [b for b in blockers if b.get("severity", "").lower() == "medium"]
    if med_blockers:
        titles = [b.get("title", "Issue") for b in med_blockers]
        answers.append(Answer(
            source="next_steps",
            title=f"Priority 3: Address {len(med_blockers)} MEDIUM issues",
            body=", ".join(titles),
            confidence=0.8,
        ))

    # Step 5: Contingency
    if pricing:
        contingency = pricing.get("contingency_range", {})
        rec = contingency.get("recommended_pct", pricing.get("recommended_contingency_pct", "?"))
        answers.append(Answer(
            source="next_steps",
            title=f"Priority 4: Apply {rec}% contingency",
            body=f"Recommended contingency: {rec}% based on document quality and blockers.",
            confidence=0.85,
        ))

    if not answers:
        answers.append(Answer(
            source="next_steps", title="No action items identified",
            body="Analysis appears complete. Review the bid pack and proceed.",
            confidence=0.7,
        ))

    return answers


def _answer_overview(payload: dict, query: str) -> List[Answer]:
    answers = []
    project_id = payload.get("project_id", "Unknown")
    pi = (payload.get("diagnostics") or {}).get("page_index", {})
    total_pages = pi.get("total_pages", 0)
    es = payload.get("extraction_summary") or {}
    counts = es.get("counts") or {}
    score = payload.get("readiness_score", 0)
    if not isinstance(score, (int, float)):
        score = 0
    decision = payload.get("decision", "UNKNOWN")
    rfis = payload.get("rfis") or []
    if not isinstance(rfis, list):
        rfis = []
    blockers = payload.get("blockers") or []
    if not isinstance(blockers, list):
        blockers = []

    body = (
        f"Project: {project_id}\n"
        f"Pages: {total_pages}\n"
        f"Decision: {decision} ({score}/100)\n"
        f"BOQ items: {counts.get('boq_items', 0)}, "
        f"Requirements: {counts.get('requirements', 0)}, "
        f"Commercial terms: {counts.get('commercial_terms', 0)}\n"
        f"RFIs: {len(rfis)}, Blockers: {len(blockers)}"
    )
    answers.append(Answer(
        source="overview", title=f"Tender Overview: {project_id}",
        body=body, confidence=1.0,
    ))
    return answers


def _answer_general(payload: dict, query: str) -> List[Answer]:
    """Fallback: search OCR text cache for the query terms."""
    ocr_cache = payload.get("ocr_text_cache", {})
    if not ocr_cache:
        return [Answer(
            source="ocr_search", title="No OCR text available",
            body="No OCR text cache in this analysis. Try a more specific question.",
            confidence=0.3,
        )]

    # Extract meaningful search terms (3+ chars, not stopwords)
    stopwords = {
        "what", "which", "where", "when", "does", "have", "this",
        "that", "about", "from", "with", "they", "there", "their",
        "been", "were", "will", "would", "could", "should",
        "tell", "show", "give", "find", "many", "much",
    }
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())
    terms = [w for w in words if w not in stopwords]

    if not terms:
        return [Answer(
            source="ocr_search", title="Could not extract search terms",
            body="Please rephrase your question with more specific terms.",
            confidence=0.2,
        )]

    # Search OCR cache
    results = []
    search_re = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)

    for page_key, text in ocr_cache.items():
        if not text:
            continue
        matches = list(search_re.finditer(text))
        if not matches:
            continue
        # Take first match with context
        m = matches[0]
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        snippet = text[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        results.append((int(page_key) if str(page_key).isdigit() else 0,
                        snippet, len(matches)))
        if len(results) >= 5:
            break

    if not results:
        return [Answer(
            source="ocr_search",
            title="No matches found",
            body=f"No text matches found for: {', '.join(terms)}",
            confidence=0.3,
        )]

    answers = []
    for page_idx, snippet, hit_count in results:
        answers.append(Answer(
            source="ocr_search",
            title=f"Page {page_idx + 1} ({hit_count} hits)",
            body=snippet,
            confidence=0.5,
            evidence=[f"Page {page_idx + 1}"],
        ))
    return answers


# =============================================================================
# SPRINT 29: RAG SEMANTIC SEARCH
# =============================================================================

# Module-level cache for the RAG pipeline (rebuilt per payload)
_rag_cache: Dict[int, Any] = {}  # id(payload) → RAGPipeline


def _get_rag_pipeline(payload: dict):
    """Get or create a RAG pipeline for the given payload.

    Caches by payload identity to avoid rebuilding on every query.
    """
    payload_id = id(payload)
    if payload_id in _rag_cache:
        return _rag_cache[payload_id]
    try:
        from src.rag import RAGPipeline
        pipeline = RAGPipeline(payload)
        _rag_cache[payload_id] = pipeline
        logger.info("RAG pipeline built: %d chunks", pipeline.index.size)
        return pipeline
    except Exception as e:
        logger.warning("RAG pipeline init failed: %s", e)
        _rag_cache[payload_id] = None
        return None


def _answer_rag(payload: dict, query: str) -> List[Answer]:
    """RAG semantic search — retrieves chunks by TF-IDF cosine similarity.

    Returns Answer objects compatible with the structured handler format.

    Args:
        payload: The analysis.json payload dict.
        query: Natural language query string.

    Returns:
        List of Answer objects from semantic search, empty on failure.
    """
    pipeline = _get_rag_pipeline(payload)
    if pipeline is None:
        return []

    try:
        hits = pipeline.semantic_search(query, top_k=5, min_score=0.1)
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return []

    answers = []
    for hit in hits:
        answers.append(Answer(
            source=hit.get("source", "rag"),
            title=hit.get("title", "Semantic match"),
            body=hit.get("body", ""),
            confidence=hit.get("confidence", 0.3),
            evidence=hit.get("evidence", []),
        ))
    return answers


# Map handler names to functions
_HANDLERS = {
    "_answer_blockers": _answer_blockers,
    "_answer_rfis": _answer_rfis,
    "_answer_trade_coverage": _answer_trade_coverage,
    "_answer_quantities": _answer_quantities,
    "_answer_commercial": _answer_commercial,
    "_answer_drawings": _answer_drawings,
    "_answer_schedules": _answer_schedules,
    "_answer_structural": _answer_structural,
    "_answer_readiness": _answer_readiness,
    "_answer_requirements": _answer_requirements,
    "_answer_next_steps": _answer_next_steps,
    "_answer_overview": _answer_overview,
    "_answer_general": _answer_general,
}


# =============================================================================
# MAIN API
# =============================================================================

def ask(payload: dict, query: str, use_rag: bool = True) -> AskResult:
    """
    Ask a natural language question about a tender.

    Args:
        payload: The analysis.json payload dict.
        query: Natural language question string.
        use_rag: If True, uses RAG semantic search as supplement/fallback.
                 Set False for deterministic-only mode (Sprint 24 behavior).

    Returns:
        AskResult with answers sorted by confidence.
    """
    if not query or not query.strip():
        return AskResult(query=query, intent="empty")

    intent, handler_name = detect_intent(query)
    handler = _HANDLERS.get(handler_name, _answer_general)

    answers = handler(payload, query)

    # If the handler returned nothing, fall back to semantic/OCR search
    fallback = False
    if not answers and handler_name != "_answer_general":
        # Sprint 29: Try RAG semantic search before basic OCR fallback
        if use_rag:
            rag_answers = _answer_rag(payload, query)
            if rag_answers:
                answers = rag_answers
                fallback = True
        # If RAG found nothing, fall back to basic OCR keyword search
        if not answers:
            answers = _answer_general(payload, query)
            fallback = True

    # Sprint 29: RAG semantic supplement — when a structured handler answered,
    # also search via TF-IDF vectors for semantically relevant context.
    # Supersedes Sprint 25 B2 OCR supplement (RAG includes OCR chunks).
    if (handler_name != "_answer_general"
            and not fallback
            and intent not in ("overview", "readiness", "next_steps", "empty")
            and answers):
        if use_rag:
            rag_hits = _answer_rag(payload, query)
            # Add top 3 RAG hits as semantic evidence
            for hit in rag_hits[:3]:
                if hit.confidence > 0.3:
                    hit.title = f"[RAG] {hit.title}"
                    hit.confidence = min(hit.confidence, 0.45)  # Cap below structured
                    answers.append(hit)
        else:
            # Legacy Sprint 25 B2: OCR-only supplement
            ocr_hits = _answer_general(payload, query)
            for hit in ocr_hits[:2]:
                if hit.confidence > 0.3:
                    hit.title = f"[OCR] {hit.title}"
                    hit.confidence = 0.4
                    answers.append(hit)

    # Sort by confidence descending
    answers.sort(key=lambda a: a.confidence, reverse=True)

    return AskResult(
        query=query,
        intent=intent,
        answers=answers,
        fallback_search=fallback,
    )


def load_payload(path: str) -> dict:
    """Load an analysis.json payload from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Analysis file not found: {path}")
    with open(p) as f:
        return json.load(f)


# =============================================================================
# CLI
# =============================================================================

def cmd_ask(args) -> int:
    """CLI handler for 'ask' command."""
    payload = load_payload(args.input)
    use_rag = not getattr(args, "no_rag", False)
    result = ask(payload, args.query, use_rag=use_rag)

    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\nIntent: {result.intent}")
    print(f"Answers: {len(result.answers)}")
    if result.fallback_search:
        print("(Fell back to OCR text search)")
    print("-" * 60)

    for i, ans in enumerate(result.answers, 1):
        print(f"\n{i}. [{ans.source}] {ans.title}")
        print(f"   {ans.body}")
        if ans.evidence:
            print(f"   Evidence: {'; '.join(ans.evidence)}")
        print(f"   Confidence: {ans.confidence:.0%}")

    print()
    return 0


def main():
    """Standalone CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Ask Tender — Hybrid RAG")
    parser.add_argument("--input", "-i", required=True, help="Path to analysis.json")
    parser.add_argument("--query", "-q", required=True, help="Natural language question")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG semantic search (deterministic-only mode)")
    args = parser.parse_args()
    return cmd_ask(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
