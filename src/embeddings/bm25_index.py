"""
BM25 sparse index for hybrid retrieval in xBOQ.ai.

Provides keyword-based retrieval to complement dense embeddings.
Especially effective for BOQ item codes, IS grade references,
and exact material/trade terms that embeddings may miss.

Usage:
    from src.embeddings.bm25_index import BM25Index
    idx = BM25Index()
    idx.build(documents)           # list of str
    results = idx.search("M25 RCC columns", n_results=10)
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Domain-specific stopwords for construction text
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with",
    "on", "at", "by", "as", "is", "are", "was", "be", "been", "being",
    "all", "any", "from", "this", "that", "it", "its", "per", "no", "not",
    "shall", "should", "will", "may", "including", "such", "each", "above",
    "below", "as per", "upto", "up", "etc", "complete", "work", "works",
})


def _tokenize(text: str) -> List[str]:
    """Tokenize construction text preserving domain tokens like M25, Fe500, IS:456."""
    text_lower = text.lower()
    # Keep alphanumeric + colon (for IS:456) and hyphen (for M-25)
    tokens = re.findall(r'[a-z0-9][a-z0-9:.\-]*[a-z0-9]|[a-z0-9]', text_lower)
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


class BM25Index:
    """
    Lightweight BM25 index over a list of text documents.
    Backed by rank_bm25.BM25Okapi.
    """

    def __init__(self):
        self._bm25 = None
        self._documents: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, documents: List[str]) -> None:
        """Build BM25 index from a list of text strings."""
        if not documents:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 index unavailable. "
                           "Install with: pip install rank-bm25")
            return

        self._documents = list(documents)
        self._tokenized_corpus = [_tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.debug("BM25Index: built with %d documents", len(self._documents))

    def search(
        self,
        query: str,
        n_results: int = 20,
    ) -> List[Tuple[int, float, str]]:
        """
        Search BM25 index.
        Returns list of (document_index, bm25_score, document_text) tuples,
        sorted by descending score.
        """
        if self._bm25 is None or not self._documents:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        try:
            scores = self._bm25.get_scores(query_tokens)
            # Get top-n indices by score
            top_n = min(n_results, len(self._documents))
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            return [(idx, float(scores[idx]), self._documents[idx]) for idx in indices]
        except Exception as e:
            logger.warning("BM25Index.search failed: %s", e)
            return []

    @property
    def doc_count(self) -> int:
        return len(self._documents)
