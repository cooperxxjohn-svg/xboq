"""
HybridRetriever — combines BM25 sparse + ChromaDB dense retrieval
using Reciprocal Rank Fusion (RRF) for xBOQ.ai.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))   where k=60

Usage:
    from src.embeddings.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(chroma_store, bm25_index, embedder)
    results = retriever.search("M25 RCC columns Fe500 stirrups", n_results=10)
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_RRF_K = 60   # standard RRF constant


class HybridRetriever:
    """
    Combines BM25 sparse retrieval and ChromaDB dense retrieval via RRF fusion.

    Falls back gracefully:
    - If BM25 index is not built → returns dense-only results
    - If ChromaDB is unavailable → returns BM25-only results
    - If neither → returns []
    """

    def __init__(
        self,
        chroma_store,     # BidChromaStore instance
        bm25_index=None,  # BM25Index instance (optional)
        embedder=None,    # Embedder instance
        domain_kb_store=None,  # DomainKBStore for background knowledge
    ):
        self.chroma_store = chroma_store
        self.bm25 = bm25_index
        self.embedder = embedder
        self.domain_kb = domain_kb_store

    def search(
        self,
        query: str,
        n_results: int = 15,
        trade_filter: Optional[str] = None,
        include_domain_kb: bool = True,
        domain_kb_n: int = 5,
    ) -> List[dict]:
        """
        Hybrid search: BM25 + dense via RRF.

        Returns list of result dicts with keys:
          text, score (RRF score), source, [metadata fields]
        sorted by descending RRF score.
        """
        candidate_pool: int = n_results * 3   # retrieve more, then re-rank

        # ── Dense retrieval (ChromaDB) ────────────────────────────────────────
        dense_results: List[dict] = []
        try:
            search_kwargs = dict(n_results=candidate_pool)
            if trade_filter:
                search_kwargs["where"] = {"trade": trade_filter}
            raw = self.chroma_store.search(query, self.embedder, **search_kwargs)
            for item in (raw or []):
                dense_results.append({
                    "text":   item.get("text") or item.get("document", ""),
                    "score":  float(item.get("score", 0)),
                    "source": item.get("source", "tender"),
                    **{k: v for k, v in item.items()
                       if k not in ("text", "document", "score")},
                })
        except Exception as e:
            logger.debug("HybridRetriever: dense search failed — %s", e)

        # ── Sparse retrieval (BM25) ───────────────────────────────────────────
        bm25_results: List[dict] = []
        if self.bm25 and self.bm25.doc_count > 0:
            try:
                raw_bm25 = self.bm25.search(query, n_results=candidate_pool)
                # Map BM25 results back to metadata using document text as key
                for doc_idx, score, text in raw_bm25:
                    bm25_results.append({
                        "text":   text,
                        "score":  score,
                        "source": "tender",
                    })
            except Exception as e:
                logger.debug("HybridRetriever: BM25 search failed — %s", e)

        # ── RRF Fusion ────────────────────────────────────────────────────────
        # Build text → RRF score mapping
        rrf_scores: dict = {}
        text_to_meta: dict = {}

        for rank, result in enumerate(dense_results):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (_RRF_K + rank + 1)
            text_to_meta[text] = result

        for rank, result in enumerate(bm25_results):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if text not in text_to_meta:
                text_to_meta[text] = result

        # Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        results = []
        for text, rrf_score in fused:
            meta = text_to_meta.get(text, {}).copy()
            meta["text"] = text
            meta["rrf_score"] = round(rrf_score, 6)
            meta["score"] = round(rrf_score, 6)
            results.append(meta)

        # ── Domain KB augmentation ────────────────────────────────────────────
        if include_domain_kb and self.domain_kb and domain_kb_n > 0:
            try:
                kb_results = self.domain_kb.search(
                    query, self.embedder, n_results=domain_kb_n,
                    trade_filter=trade_filter,
                )
                for kr in kb_results:
                    if kr.get("score", 0) >= 0.30:   # only add relevant KB hits
                        results.append({
                            "text":   kr["text"],
                            "score":  kr.get("score", 0),
                            "rrf_score": 0.0,
                            "source": kr.get("source", "domain_kb"),
                            **{k: v for k, v in kr.items()
                               if k not in ("text", "score")},
                        })
            except Exception as e:
                logger.debug("HybridRetriever: domain KB search failed — %s", e)

        return results
