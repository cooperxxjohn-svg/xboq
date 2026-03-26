"""
src/bidagent/agent.py

Stateful conversational Q&A agent for a bid package.

Routing logic (in order):
1. ask_tender structured handler  — fast, deterministic, intent-matched
2. ChromaDB semantic search       — retrieve top-8 relevant chunks
3. LLM synthesis (if client)      — compose answer with history + context
4. Fallback text                  — combine structured answer + top chunk texts

Suggested follow-up questions are generated per answer to guide the user
through a complete bid review workflow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are xBOQ, an AI quantity surveyor analysing a construction bid package.
Answer using ONLY the context provided below. Be concise, specific, and professional.
If information is missing from the context, say so clearly.
Always reference page numbers when available.
Respond in plain text (no markdown), maximum 4 sentences.

Bid readiness: {label} ({score}/100)
Critical gaps: {n_critical}
Estimated cost: {cost}
"""

_SUGGESTED_QUESTIONS: List[str] = [
    "What are the critical blockers?",
    "Explain the structural scope",
    "What RFIs should I send first?",
    "How complete is the MEP specification?",
    "What is the recommended contingency?",
    "Which trades have missing BOQ items?",
    "What are the main commercial risks?",
    "Summarise the door and window schedule",
]


@dataclass
class AgentAnswer:
    answer: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    suggested_follow_ups: List[str] = field(default_factory=list)
    from_handler: bool = False
    from_llm: bool = False
    from_rag: bool = False


class BidAgent:
    """
    Stateful conversational agent over a bid package.

    Parameters
    ----------
    payload : dict
        Full analysis payload from pipeline.py.
    synthesis : BidSynthesis
        Compiled bid synthesis object (from bid_synthesizer.py).
    store : BidChromaStore, optional
        ChromaDB store for semantic retrieval.
    embedder : Embedder, optional
        Text embedder (required if store is provided).
    llm_client : optional
        OpenAI or Anthropic client for answer synthesis.
    history_limit : int
        Maximum number of Q&A turns retained in context.
    """

    def __init__(
        self,
        payload: dict,
        synthesis: Any,
        store: Optional[Any] = None,
        embedder: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        history_limit: int = 10,
    ):
        self._payload = payload
        self._synthesis = synthesis
        self._store = store
        self._embedder = embedder
        self._llm = llm_client
        self._history_limit = history_limit
        self._history: List[Dict[str, str]] = []   # [{role, content}]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> AgentAnswer:
        """Answer a question about the bid package."""
        question = question.strip()
        if not question:
            return AgentAnswer(
                answer="Please enter a question about the bid package.",
                confidence=0.0,
                suggested_follow_ups=_SUGGESTED_QUESTIONS[:3],
            )

        structured_answer: Optional[str] = None
        from_handler = False

        # ── Step 1: structured handler ─────────────────────────────────
        try:
            from src.ask_tender import ask as _ask_tender
            result = _ask_tender(self._payload, question, use_rag=False)
            if result and result.body and result.confidence > 0.4:
                structured_answer = result.body
                from_handler = True
        except Exception as e:
            logger.debug("ask_tender handler failed: %s", e)

        # ── Step 2: semantic search ────────────────────────────────────
        chunks: List[str] = []
        sources: List[str] = []
        from_rag = False
        if self._store is not None and self._embedder is not None:
            try:
                hits = self._store.search(question, self._embedder, n_results=15)
                for h in hits:
                    if h.score > 0.25:
                        chunks.append(h.text)
                        pg = h.metadata.get("page")
                        src = h.metadata.get("source", "")
                        if pg:
                            sources.append(f"p{pg} ({src})")
                        else:
                            sources.append(src)
                from_rag = bool(chunks)
            except Exception as e:
                logger.debug("ChromaDB search failed: %s", e)

        # Augment with domain knowledge for all queries (R4/R5)
        # Use more results for rate/spec queries, fewer for general queries
        _agent_domain_chunks: list = []
        try:
            from src.embeddings.kb_interface import get_kb as _get_kb
            _agent_domain_store = _get_kb()
            if _agent_domain_store.count() > 0:
                _rate_keywords = {"rate", "cost", "price", "amount", "dsr", "sor", "₹", "rs",
                                  "specification", "spec", "is code", "nbc", "grade", "mix"}
                _q_lower = question.lower()
                _is_rate_query = any(kw in _q_lower for kw in _rate_keywords)
                _n_kb = 5 if _is_rate_query else 3   # more results for rate queries
                _dkb_hits = _agent_domain_store.search(question, self._embedder, n_results=_n_kb)
                _agent_domain_chunks = [
                    f"[DOMAIN: {h.get('source','kb')}] {h['text']}"
                    for h in _dkb_hits if h.get("score", 0) >= (0.25 if _is_rate_query else 0.35)
                ][:_n_kb]
        except Exception:
            pass

        # ── Step 3: LLM synthesis ──────────────────────────────────────
        from_llm = False
        answer = ""
        if self._llm is not None and (chunks or structured_answer):
            try:
                answer = self._llm_answer(question, structured_answer, chunks, _agent_domain_chunks)
                from_llm = bool(answer)
            except Exception as e:
                logger.warning("LLM answer failed: %s", e)

        # ── Step 4: fallback combination ───────────────────────────────
        if not answer:
            parts = []
            if structured_answer:
                parts.append(structured_answer)
            if chunks:
                parts.append("Relevant context: " + " | ".join(c[:120] for c in chunks[:3]))
            answer = " ".join(parts) if parts else (
                "I don't have enough information in this bid package to answer that question. "
                "Please check the source documents directly."
            )

        # ── Update history ─────────────────────────────────────────────
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > self._history_limit * 2:
            self._history = self._history[-(self._history_limit * 2):]

        follow_ups = _follow_up_suggestions(question, self._synthesis)
        confidence = self._confidence_score(from_handler, from_llm, from_rag, chunks)

        return AgentAnswer(
            answer=answer,
            confidence=confidence,
            sources=list(dict.fromkeys(sources))[:8],   # deduplicated
            suggested_follow_ups=follow_ups,
            from_handler=from_handler,
            from_llm=from_llm,
            from_rag=from_rag,
        )

    def reset_history(self) -> None:
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _llm_answer(
        self,
        question: str,
        structured_answer: Optional[str],
        chunks: List[str],
        domain_chunks: Optional[List[str]] = None,
    ) -> str:
        synth = self._synthesis
        cost = (
            f"₹{synth.estimated_cost_inr/1e7:.1f} Cr"
            if synth.estimated_cost_inr > 0
            else "Unknown"
        )
        system = _SYSTEM_PROMPT.format(
            label=synth.bid_readiness_label,
            score=synth.bid_readiness_score,
            n_critical=len(synth.critical_gaps),
            cost=cost,
        )

        context_parts = []
        if structured_answer:
            context_parts.append(f"Structured analysis:\n{structured_answer}")
        if chunks:
            context_parts.append("Relevant document excerpts:\n" + "\n".join(f"- {c[:600]}" for c in chunks[:6]))
        if domain_chunks:
            context_parts.append("Domain knowledge:\n" + "\n".join(f"- {c}" for c in domain_chunks))

        context = "\n\n".join(context_parts)

        messages = []
        # Include recent history (up to last 4 turns)
        for turn in self._history[-8:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        # Current question with context
        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        })

        if hasattr(self._llm, "chat"):
            resp = self._llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}] + messages,
                temperature=0.2,
                max_tokens=2000,
            )
            return resp.choices[0].message.content.strip()

        if hasattr(self._llm, "messages"):
            resp = self._llm.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=system,
                messages=messages,
            )
            return resp.content[0].text.strip() if resp.content else ""

        return ""

    @staticmethod
    def _confidence_score(from_handler: bool, from_llm: bool, from_rag: bool, chunks: List) -> float:
        score = 0.3
        if from_handler:
            score += 0.3
        if from_rag and chunks:
            score += 0.2
        if from_llm:
            score += 0.15
        return min(1.0, score)


# ------------------------------------------------------------------
# Suggested follow-ups
# ------------------------------------------------------------------

def _follow_up_suggestions(question: str, synthesis: Any) -> List[str]:
    q_lower = question.lower()

    if any(kw in q_lower for kw in ["blocker", "critical", "ready"]):
        return [
            "What RFIs should I send first?",
            "What is the recommended contingency?",
            "Which trades are highest risk?",
        ]
    if any(kw in q_lower for kw in ["structural", "concrete", "rcc", "steel"]):
        return [
            "What is the estimated structural cost?",
            "Are there missing structural drawings?",
            "What concrete grades are specified?",
        ]
    if any(kw in q_lower for kw in ["mep", "electrical", "plumbing", "hvac"]):
        return [
            "How complete is the MEP specification?",
            "Are MEP BOQ items extracted?",
            "What specialist subcontractors are needed?",
        ]
    if any(kw in q_lower for kw in ["cost", "rate", "price", "estimate"]):
        return [
            "What is the recommended contingency?",
            "Which trades have unrated items?",
            "What is the cost per sqm?",
        ]
    if any(kw in q_lower for kw in ["rfi", "clarification", "question"]):
        return [
            "What are the critical blockers?",
            "Which gaps have the highest cost impact?",
            "Summarise the door and window schedule",
        ]

    # Generic follow-ups
    return [
        "What are the critical blockers?",
        "What is the recommended contingency?",
        "What RFIs should I send first?",
    ]
