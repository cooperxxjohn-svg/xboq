"""
Natural language query interface for tender payloads.
Allows users to ask questions about a loaded tender in plain English.
"""
from __future__ import annotations
import re
import json
from typing import Optional

# ── Heuristic query engine (no LLM dependency) ────────────────────────────────

_TRADE_KEYWORDS = {
    "civil": ["civil", "concrete", r"\brcc\b", "foundation", "footing", "column", "beam", "slab"],
    "structural": ["structural", "steel", "rebar", "reinforcement", r"\bms\b"],
    "architectural": ["architectural", r"\barch\b", "brick", "masonry", "plaster", "paint", "tile", "flooring"],
    "mep": [r"\bmep\b", "plumbing", "hvac", "electrical", "fire", "sanitary"],
    "finishing": ["finish", r"\bdoor\b", r"\bwindow\b", "glazing"],
    "sitework": ["site", "earthwork", "excavation", r"\broad\b", "drainage"],
}

_INTENT_PATTERNS = [
    (r"\b(rfi|rfis|clarifications?|question)\b", "rfis"),
    (r"\b(boq|bill of quantity|line items?)\b", "boq"),
    (r"\b(blockers?|blocker|risk|issue)\b", "blockers"),
    (r"\b(cost|total|amount|value|price|\u20b9|inr|rupee)\b", "cost"),
    (r"\b(area|sqm|square metre|floor area)\b", "area"),
    (r"\b(page|pages|sheet|sheets)\b", "pages"),
    (r"\b(deadline|submission|due date|bid date)\b", "deadline"),
    (r"\b(trade|discipline|work type)\b", "trades"),
    (r"\b(quality|qa|score|confidence)\b", "quality"),
    (r"\b(summary|overview|brief)\b", "summary"),
]


def _detect_intent(query: str) -> str:
    q = query.lower()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, q):
            return intent
    return "summary"


def _detect_trade(query: str) -> Optional[str]:
    q = query.lower()
    for trade, kws in _TRADE_KEYWORDS.items():
        for kw in kws:
            # Keywords starting with \b are regex patterns; others use substring match
            if kw.startswith(r"\b"):
                if re.search(kw, q):
                    return trade
            elif kw in q:
                return trade
    return None


def _detect_threshold(query: str) -> Optional[float]:
    """Extract a numeric threshold from query like 'over \u20b910L' or 'above 100'."""
    m = re.search(r"(?:over|above|more than|>)\s*[\u20b9rs.]?\s*([\d,.]+)\s*([lLkKcC]?)", query.lower())
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    suffix = m.group(2).lower()
    if suffix == "l":
        val *= 100_000
    elif suffix == "k":
        val *= 1_000
    elif suffix == "c":
        val *= 10_000_000
    return val


def answer_query(query: str, payload: dict) -> dict:
    """
    Answer a natural language query about a tender payload.
    Returns {answer: str, intent: str, results: list, count: int}.
    Always returns gracefully — never raises.
    """
    if not payload:
        return {"answer": "No tender loaded. Please upload and analyse a tender first.", "intent": "none", "results": [], "count": 0}

    intent = _detect_intent(query)
    trade_filter = _detect_trade(query)
    threshold = _detect_threshold(query)

    try:
        if intent == "rfis":
            return _answer_rfis(query, payload, trade_filter)
        elif intent == "boq":
            return _answer_boq(query, payload, trade_filter, threshold)
        elif intent == "blockers":
            return _answer_blockers(payload, trade_filter)
        elif intent == "cost":
            return _answer_cost(payload, trade_filter, threshold)
        elif intent == "area":
            return _answer_area(payload)
        elif intent == "pages":
            return _answer_pages(payload)
        elif intent == "deadline":
            return _answer_deadline(payload)
        elif intent == "trades":
            return _answer_trades(payload)
        elif intent == "quality":
            return _answer_quality(payload)
        else:
            return _answer_summary(payload)
    except Exception as exc:
        return {"answer": f"Could not process query: {exc}", "intent": intent, "results": [], "count": 0}


def _answer_rfis(query, payload, trade_filter):
    rfis = payload.get("rfis", [])
    if trade_filter:
        rfis = [r for r in rfis if trade_filter.lower() in str(r.get("trade", "")).lower()]
    q = query.lower()
    if "high" in q or "critical" in q or "urgent" in q:
        rfis = [r for r in rfis if str(r.get("priority", "")).lower() in ("high", "critical")]
    count = len(rfis)
    if count == 0:
        answer = f"No RFIs found{' for ' + trade_filter if trade_filter else ''}."
    else:
        sample = rfis[:5]
        lines = [f"- [{r.get('priority','?').upper()}] {r.get('question', r.get('rfi_text', ''))[:120]}" for r in sample]
        answer = f"Found **{count} RFI{'s' if count != 1 else ''}**{' for ' + trade_filter if trade_filter else ''}:\n" + "\n".join(lines)
        if count > 5:
            answer += f"\n_\u2026and {count-5} more_"
    return {"answer": answer, "intent": "rfis", "results": rfis, "count": count}


def _answer_boq(query, payload, trade_filter, threshold):
    items = payload.get("boq_items", [])
    if trade_filter:
        items = [i for i in items if trade_filter.lower() in str(i.get("trade", "")).lower()]
    if threshold is not None:
        items = [i for i in items if float(i.get("total_inr", i.get("total", 0)) or 0) > threshold]
    count = len(items)
    total = sum(float(i.get("total_inr", i.get("total", 0)) or 0) for i in items)
    if count == 0:
        answer = "No BOQ items match your query."
    else:
        sample = items[:5]
        lines = [f"- {i.get('description','?')[:80]} \u2014 {i.get('quantity','')} {i.get('unit','')} @ \u20b9{i.get('rate_inr','?')}" for i in sample]
        answer = f"Found **{count} item{'s' if count != 1 else ''}** (total \u20b9{total:,.0f}):\n" + "\n".join(lines)
        if count > 5:
            answer += f"\n_\u2026and {count-5} more_"
    return {"answer": answer, "intent": "boq", "results": items, "count": count}


def _answer_blockers(payload, trade_filter):
    blockers = payload.get("blockers", [])
    if trade_filter:
        blockers = [b for b in blockers if trade_filter.lower() in str(b.get("trade", "") + b.get("description", "")).lower()]
    count = len(blockers)
    if count == 0:
        answer = "No blockers found."
    else:
        lines = [f"- {b.get('description', b.get('issue', ''))[:120]}" for b in blockers[:5]]
        answer = f"Found **{count} blocker{'s' if count != 1 else ''}**:\n" + "\n".join(lines)
    return {"answer": answer, "intent": "blockers", "results": blockers, "count": count}


def _answer_cost(payload, trade_filter, threshold):
    items = payload.get("boq_items", [])
    if trade_filter:
        items = [i for i in items if trade_filter.lower() in str(i.get("trade", "")).lower()]
    if threshold is not None:
        items = [i for i in items if float(i.get("total_inr", i.get("total", 0)) or 0) > threshold]
    total = sum(float(i.get("total_inr", i.get("total", 0)) or 0) for i in items)
    count = len(items)
    if total == 0 and not items:
        answer = "No cost data available in this tender."
    else:
        scope = f" ({trade_filter} trade)" if trade_filter else ""
        answer = f"Total estimated cost{scope}: **\u20b9{total:,.0f}** across {count} line items."
        # breakdown by trade
        if not trade_filter:
            trade_totals: dict = {}
            for i in items:
                t = i.get("trade", "other")
                trade_totals[t] = trade_totals.get(t, 0) + float(i.get("total_inr", i.get("total", 0)) or 0)
            top = sorted(trade_totals.items(), key=lambda x: -x[1])[:5]
            if top:
                answer += "\n\nBy trade:\n" + "\n".join(f"- {t}: \u20b9{v:,.0f}" for t, v in top)
    return {"answer": answer, "intent": "cost", "results": items, "count": count}


def _answer_area(payload):
    area = payload.get("total_area_sqm") or payload.get("project_area_sqm")
    plan = payload.get("plan_graph", {})
    rooms = plan.get("rooms", []) if isinstance(plan, dict) else []
    if area:
        answer = f"Total project area: **{area:,.1f} sqm**"
        if rooms:
            answer += f" across {len(rooms)} room{'s' if len(rooms) != 1 else ''}."
    else:
        answer = "Area data not available \u2014 ensure drawings pages were processed."
    return {"answer": answer, "intent": "area", "results": [], "count": 0}


def _answer_pages(payload):
    stats = payload.get("processing_stats", {})
    total = stats.get("total_pages", 0)
    deep = stats.get("deep_processed_pages", 0)
    ocr = stats.get("ocr_pages", 0)
    pi = payload.get("diagnostics", {}).get("page_index", {})
    pages = pi.get("pages", []) if isinstance(pi, dict) else []
    answer = f"Processed **{total} pages** ({deep} deep, {ocr} OCR)."
    if pages:
        doc_types: dict = {}
        for p in pages:
            dt = p.get("doc_type", "unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1
        answer += "\n\nPage types: " + ", ".join(f"{dt}: {n}" for dt, n in sorted(doc_types.items()))
    return {"answer": answer, "intent": "pages", "results": pages, "count": total}


def _answer_deadline(payload):
    deadline = payload.get("bid_deadline") or payload.get("submission_date")
    commercial = payload.get("commercial_terms", {})
    if isinstance(commercial, dict):
        deadline = deadline or commercial.get("bid_submission_date") or commercial.get("submission_deadline")
    if deadline:
        answer = f"Bid submission deadline: **{deadline}**"
    else:
        answer = "No submission deadline found in this tender. Check the commercial terms tab."
    return {"answer": answer, "intent": "deadline", "results": [], "count": 0}


def _answer_trades(payload):
    items = payload.get("boq_items", [])
    trade_set: dict = {}
    for i in items:
        t = i.get("trade", "other")
        trade_set[t] = trade_set.get(t, 0) + 1
    if not trade_set:
        answer = "No trade breakdown available."
    else:
        answer = f"Found **{len(trade_set)} trades**:\n" + "\n".join(f"- {t}: {n} items" for t, n in sorted(trade_set.items(), key=lambda x: -x[1]))
    return {"answer": answer, "intent": "trades", "results": list(trade_set.keys()), "count": len(trade_set)}


def _answer_quality(payload):
    qa = payload.get("qa_score") or payload.get("quality_score", {})
    if isinstance(qa, dict):
        score = qa.get("overall_score") or qa.get("score")
    elif isinstance(qa, (int, float)):
        score = qa
    else:
        score = None
    trade_conf = payload.get("trade_confidence", {})
    if score is not None:
        answer = f"QA score: **{score}/100**"
    else:
        answer = "No QA score available for this tender."
    if trade_conf:
        low = [t for t, v in trade_conf.items() if isinstance(v, (int, float)) and v < 0.6]
        if low:
            answer += f"\n\nLow-confidence trades: {', '.join(low)}"
    return {"answer": answer, "intent": "quality", "results": [], "count": 0}


def _answer_summary(payload):
    name = payload.get("project_name", "Unknown project")
    stats = payload.get("processing_stats", {})
    total_pages = stats.get("total_pages", 0)
    boq_items = payload.get("boq_items", [])
    rfis = payload.get("rfis", [])
    blockers = payload.get("blockers", [])
    total_cost = sum(float(i.get("total_inr", i.get("total", 0)) or 0) for i in boq_items)
    answer = (
        f"**{name}**\n\n"
        f"- Pages processed: {total_pages}\n"
        f"- BOQ items: {len(boq_items)} (total \u20b9{total_cost:,.0f})\n"
        f"- RFIs: {len(rfis)}\n"
        f"- Blockers: {len(blockers)}\n\n"
        f"Try asking: _'show civil RFIs'_, _'BOQ items over \u20b910L'_, _'what are the blockers?'_, _'total cost by trade'_"
    )
    return {"answer": answer, "intent": "summary", "results": [], "count": 0}
