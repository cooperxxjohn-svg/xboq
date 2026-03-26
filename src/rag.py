"""
RAG Pipeline — Retrieval-Augmented Generation for Tender Analysis.

Sprint 29: Semantic search layer over the analysis payload.  Uses the
VectorIndex (TF-IDF + cosine similarity) from vectorstore.py for retrieval,
then synthesizes structured answers from retrieved context.

Three retrieval modes:
  1. **Vector-only**: Pure cosine similarity search.
  2. **Hybrid**: Vector search + structured payload lookup (combines both).
  3. **Reranked**: Retrieves 2×top_k candidates, reranks by relevance signals.

Integrates with Ask Tender as an enhanced fallback — when the deterministic
intent handler finds nothing or has low confidence, RAG supplements with
semantically-relevant context from the full corpus.

Usage:
    from src.rag import RAGPipeline

    rag = RAGPipeline(payload)
    results = rag.query("What concrete grade is specified for columns?")
    # → RAGResult(query=..., chunks=[...], answer=..., confidence=0.82)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.vectorstore import VectorIndex, build_index_from_payload, tokenize

logger = logging.getLogger(__name__)


# =============================================================================
# RAG RESULT SCHEMA
# =============================================================================

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with relevance score."""
    chunk_id: str
    text: str
    score: float              # cosine similarity (0.0–1.0)
    source: str               # "ocr", "blocker", "rfi", "requirement", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:300],
            "score": round(self.score, 4),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class RAGResult:
    """Full result from a RAG query."""
    query: str
    chunks: List[RetrievedChunk]    # retrieved context chunks
    answer: str                      # synthesized answer text
    confidence: float                # overall confidence (0.0–1.0)
    sources_used: List[str] = field(default_factory=list)  # source types used
    retrieval_mode: str = "hybrid"   # "vector", "hybrid", or "reranked"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "chunk_count": len(self.chunks),
            "sources_used": self.sources_used,
            "retrieval_mode": self.retrieval_mode,
            "chunks": [c.to_dict() for c in self.chunks],
        }


# =============================================================================
# RAG PIPELINE
# =============================================================================

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for tender analysis.

    Builds a vector index from the analysis payload and provides
    semantic search with answer synthesis.

    Args:
        payload: The analysis.json payload dict.
        index: Optional pre-built VectorIndex. If None, builds from payload.
    """

    def __init__(self, payload: dict, index: Optional[VectorIndex] = None) -> None:
        self.payload = payload
        self.index = index or build_index_from_payload(payload)

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.05,
        mode: str = "hybrid",
        source_filter: Optional[List[str]] = None,
    ) -> RAGResult:
        """Query the RAG pipeline.

        Args:
            question: Natural language question.
            top_k: Maximum number of chunks to retrieve.
            min_score: Minimum similarity threshold.
            mode: Retrieval mode — "vector", "hybrid", or "reranked".
            source_filter: Optional list of source types to filter
                          (e.g., ["ocr", "blocker"]).

        Returns:
            RAGResult with retrieved chunks and synthesized answer.
        """
        if not question or not question.strip():
            return RAGResult(
                query=question, chunks=[], answer="", confidence=0.0,
                retrieval_mode=mode,
            )

        # Build filter function from source_filter
        filter_fn = None
        if source_filter:
            allowed = set(source_filter)
            filter_fn = lambda meta: meta.get("source", "") in allowed

        if mode == "reranked":
            chunks = self._retrieve_reranked(question, top_k, min_score, filter_fn)
        elif mode == "hybrid":
            chunks = self._retrieve_hybrid(question, top_k, min_score, filter_fn)
        else:
            chunks = self._retrieve_vector(question, top_k, min_score, filter_fn)

        # Synthesize answer from retrieved chunks
        answer, confidence = self._synthesize(question, chunks)

        # Collect source types
        sources_used = list(set(c.source for c in chunks))

        return RAGResult(
            query=question,
            chunks=chunks,
            answer=answer,
            confidence=confidence,
            sources_used=sources_used,
            retrieval_mode=mode,
        )

    # ----- Retrieval strategies -----

    def _retrieve_vector(
        self,
        query: str,
        top_k: int,
        min_score: float,
        filter_fn: Optional[Any],
    ) -> List[RetrievedChunk]:
        """Pure vector similarity search."""
        results = self.index.search(query, top_k=top_k, min_score=min_score,
                                     filter_fn=filter_fn)
        chunks = []
        for chunk_id, score, metadata in results:
            # Find original text
            idx = self.index._id_to_idx.get(chunk_id)
            text = self.index.chunks[idx].text if idx is not None else ""
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                score=score,
                source=metadata.get("source", "unknown"),
                metadata=metadata,
            ))
        return chunks

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        min_score: float,
        filter_fn: Optional[Any],
    ) -> List[RetrievedChunk]:
        """Hybrid: vector search + keyword boosting.

        Retrieves via vector similarity, then boosts scores for chunks
        that have exact keyword matches from the query.
        """
        # Get 2×top_k from vector search for reranking headroom
        vector_results = self.index.search(query, top_k=top_k * 2,
                                            min_score=min_score * 0.5,
                                            filter_fn=filter_fn)

        if not vector_results:
            return []

        # Extract query keywords for boosting
        query_tokens = set(tokenize(query))

        chunks = []
        for chunk_id, score, metadata in vector_results:
            idx = self.index._id_to_idx.get(chunk_id)
            if idx is None:
                continue
            chunk = self.index.chunks[idx]

            # Keyword boost: check for exact token matches
            chunk_tokens = set(chunk.tokens)
            overlap = query_tokens & chunk_tokens
            keyword_boost = len(overlap) / max(len(query_tokens), 1) * 0.15

            # Source boost: structured findings are more reliable than raw OCR
            source = metadata.get("source", "")
            source_boost = 0.0
            if source in ("blocker", "rfi", "requirement"):
                source_boost = 0.10
            elif source in ("commercial", "quantity", "trade_coverage"):
                source_boost = 0.05

            final_score = min(1.0, score + keyword_boost + source_boost)

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk.text,
                score=final_score,
                source=source or "unknown",
                metadata=metadata,
            ))

        # Re-sort by boosted score and take top_k
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    def _retrieve_reranked(
        self,
        query: str,
        top_k: int,
        min_score: float,
        filter_fn: Optional[Any],
    ) -> List[RetrievedChunk]:
        """Reranked: retrieve 3×top_k, rerank by multi-signal scoring.

        Signals: vector similarity, keyword overlap, source reliability,
        text length quality, and query-chunk relevance alignment.
        """
        # Retrieve broad candidate set
        vector_results = self.index.search(query, top_k=top_k * 3,
                                            min_score=min_score * 0.3,
                                            filter_fn=filter_fn)
        if not vector_results:
            return []

        query_tokens = set(tokenize(query))
        candidates = []

        for chunk_id, vec_score, metadata in vector_results:
            idx = self.index._id_to_idx.get(chunk_id)
            if idx is None:
                continue
            chunk = self.index.chunks[idx]
            chunk_tokens = set(chunk.tokens)

            # Signal 1: Vector similarity (0–1)
            s_vector = vec_score

            # Signal 2: Keyword overlap ratio (0–1)
            overlap = query_tokens & chunk_tokens
            s_keyword = len(overlap) / max(len(query_tokens), 1)

            # Signal 3: Source reliability (0–1)
            source = metadata.get("source", "")
            s_source = {
                "blocker": 0.9, "rfi": 0.85, "requirement": 0.85,
                "commercial": 0.8, "quantity": 0.75,
                "trade_coverage": 0.7, "structural": 0.7,
                "ocr": 0.5,
            }.get(source, 0.4)

            # Signal 4: Text quality — moderate length is better
            text_len = len(chunk.text)
            if 50 <= text_len <= 500:
                s_quality = 1.0
            elif text_len < 50:
                s_quality = text_len / 50
            else:
                s_quality = max(0.5, 1.0 - (text_len - 500) / 2000)

            # Weighted combination
            final_score = (
                0.50 * s_vector +
                0.20 * s_keyword +
                0.15 * s_source +
                0.15 * s_quality
            )

            candidates.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk.text,
                score=min(1.0, final_score),
                source=source or "unknown",
                metadata=metadata,
            ))

        # Sort by reranked score
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Deduplicate: if two chunks from same page, keep higher score
        seen_pages = set()
        deduped = []
        for c in candidates:
            page = c.metadata.get("page", None)
            page_key = (c.source, page) if page is not None else None
            if page_key and page_key in seen_pages:
                # Allow max 2 chunks from same page
                page_count = sum(1 for d in deduped
                                if d.metadata.get("page") == page
                                and d.source == c.source)
                if page_count >= 2:
                    continue
            if page_key:
                seen_pages.add(page_key)
            deduped.append(c)
            if len(deduped) >= top_k:
                break

        return deduped

    # ----- Answer synthesis -----

    def _synthesize(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> Tuple[str, float]:
        """Synthesize an answer from retrieved chunks.

        Uses extractive synthesis — selects and combines relevant
        snippets from retrieved chunks into a coherent answer.

        Args:
            query: Original query string.
            chunks: Retrieved context chunks.

        Returns:
            (answer_text, confidence_score) tuple.
        """
        if not chunks:
            return ("No relevant information found for this query.", 0.2)

        # Build answer from top chunks
        parts = []
        sources = set()
        max_conf = 0.0

        for i, chunk in enumerate(chunks[:5]):  # Cap at 5 chunks in answer
            source_label = _source_display(chunk.source)
            page_info = ""
            if "page" in chunk.metadata:
                page_num = chunk.metadata["page"]
                if isinstance(page_num, int) and page_num >= 0:
                    page_info = f" (p.{page_num + 1})"

            # Truncate long chunks for answer readability
            text = chunk.text.strip()
            if len(text) > 300:
                # Find sentence boundary near 300 chars
                end = text.find(". ", 200, 350)
                if end > 0:
                    text = text[:end + 1]
                else:
                    text = text[:300] + "..."

            relevance = f"{chunk.score:.0%}"
            parts.append(f"**{source_label}{page_info}** [{relevance}]: {text}")
            sources.add(chunk.source)
            max_conf = max(max_conf, chunk.score)

        answer = "\n\n".join(parts)

        # Confidence: weighted by top score and number of supporting chunks
        n_support = min(len(chunks), 5)
        avg_score = sum(c.score for c in chunks[:5]) / n_support if n_support else 0
        confidence = 0.6 * max_conf + 0.3 * avg_score + 0.1 * min(n_support / 3, 1.0)
        confidence = min(1.0, max(0.0, confidence))

        return (answer, confidence)

    # ----- Batch relevance scoring -----

    def score_items(
        self,
        items: List[Dict[str, Any]],
        context_query: str,
        text_key: str = "description",
    ) -> List[Tuple[int, float]]:
        """Score a list of items by relevance to a context query.

        Assigns a relevance value (0.0–1.0) to each item based on
        how semantically similar it is to the query context.

        Args:
            items: List of dicts with text content.
            context_query: Context text to score against.
            text_key: Key in item dicts containing text to compare.

        Returns:
            List of (item_index, relevance_score) sorted by score desc.
        """
        if not items or not context_query:
            return []

        scored = []
        for i, item in enumerate(items):
            text = ""
            if isinstance(item, dict):
                text = str(item.get(text_key, ""))
                # Also include other text fields
                for k in ("title", "question", "trade"):
                    v = item.get(k, "")
                    if v:
                        text += f" {v}"
            elif isinstance(item, str):
                text = item

            if text:
                score = self.index.relevance_score(context_query, text)
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ----- Semantic search for Ask Tender -----

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Simplified semantic search interface for Ask Tender integration.

        Returns results in a format compatible with the Answer dataclass.

        Args:
            query: Search query string.
            top_k: Max results.
            min_score: Minimum similarity threshold.

        Returns:
            List of dicts with keys: source, title, body, confidence, evidence.
        """
        result = self.query(question=query, top_k=top_k, min_score=min_score,
                           mode="hybrid")

        answers = []
        for chunk in result.chunks:
            page_refs = []
            if "page" in chunk.metadata:
                page_num = chunk.metadata["page"]
                if isinstance(page_num, int) and page_num >= 0:
                    page_refs.append(f"Page {page_num + 1}")

            answers.append({
                "source": f"rag_{chunk.source}",
                "title": _make_title(chunk),
                "body": chunk.text[:300],
                "confidence": round(chunk.score, 2),
                "evidence": page_refs,
            })

        return answers


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _source_display(source: str) -> str:
    """Human-readable source label."""
    return {
        "ocr": "OCR Text",
        "blocker": "Blocker",
        "rfi": "RFI",
        "requirement": "Requirement",
        "commercial": "Commercial Term",
        "quantity": "Quantity",
        "trade_coverage": "Trade Coverage",
        "structural": "Structural",
    }.get(source, source.title())


def _make_title(chunk: RetrievedChunk) -> str:
    """Generate a short title for a retrieved chunk."""
    source = _source_display(chunk.source)
    meta = chunk.metadata

    if chunk.source == "ocr":
        page = meta.get("page", -1)
        seg = meta.get("segment", None)
        if page >= 0:
            title = f"{source} — Page {page + 1}"
            if seg is not None:
                title += f" (segment {seg})"
            return title
        return source

    if chunk.source == "blocker":
        return f"{source}: {meta.get('id', 'unknown')} [{meta.get('severity', '')}]"

    if chunk.source == "rfi":
        return f"{source}: {meta.get('id', 'unknown')} ({meta.get('trade', '')})"

    if chunk.source == "requirement":
        return f"{source} — {meta.get('trade', 'general')}"

    if chunk.source == "commercial":
        return f"{source}: {meta.get('field', '')}"

    if chunk.source == "quantity":
        return f"{source} ({meta.get('trade', '')})"

    if chunk.source == "trade_coverage":
        return f"{source}: {meta.get('trade', '')} — {meta.get('coverage_pct', 0)}%"

    return source
