"""
Vector Store — TF-IDF + Cosine Similarity Index for Tender Analysis.

Sprint 29: Provides a lightweight vector embedding layer using TF-IDF
(Term Frequency–Inverse Document Frequency) and cosine similarity.
Zero new dependencies — uses only numpy and scipy already in requirements.

Indexes text chunks from OCR pages, extracted findings (blockers, RFIs,
requirements), and structured fields.  Supports fast top-k nearest
neighbour search for semantic retrieval.

Usage:
    from src.vectorstore import VectorIndex

    idx = VectorIndex()
    idx.add("chunk-0", "RCC grade M25 for all columns", {"page": 12, "type": "ocr"})
    idx.add("chunk-1", "Provide approved make list for MEP", {"page": 4, "type": "rfi"})
    idx.build()
    results = idx.search("what concrete grade is specified?", top_k=5)
    # → [(chunk_id, score, metadata), ...]
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

# Common English stopwords + construction-domain noise words
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "be", "as", "by", "so", "up", "if", "no", "do",
    "my", "we", "he", "me", "am", "us", "this", "that", "with", "from",
    "are", "was", "were", "been", "will", "have", "has", "had", "may",
    "can", "not", "its", "than", "all", "each", "any", "our", "who",
    "how", "what", "when", "where", "which", "would", "could", "should",
    "shall", "into", "also", "per", "etc", "viz", "via", "ie", "eg",
    "above", "below", "over", "under", "after", "before",
    # Minimal construction noise (keep domain terms!)
    "page", "refer", "see", "note", "nos",
})


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens.

    Preserves construction-domain tokens like 'M25', 'D-101', 'IS:456'.
    Strips stopwords and single-character tokens.

    Args:
        text: Raw text string.

    Returns:
        List of normalized tokens.
    """
    if not text:
        return []
    # Lowercase and split on non-alphanumeric (preserving hyphens in codes)
    tokens = re.findall(r'[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]


# =============================================================================
# CHUNK — a single indexed unit
# =============================================================================

@dataclass
class Chunk:
    """A text chunk in the vector store with metadata."""
    chunk_id: str        # unique identifier
    text: str            # original text
    tokens: List[str]    # tokenized form
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:500],  # Truncate for serialization
            "metadata": self.metadata,
        }


# =============================================================================
# TF-IDF VECTOR INDEX
# =============================================================================

class VectorIndex:
    """TF-IDF vector index with cosine similarity search.

    Builds a sparse TF-IDF matrix from text chunks and supports
    top-k nearest neighbour search via cosine similarity.

    Uses numpy for vector math — no heavy ML frameworks needed.

    Example:
        idx = VectorIndex()
        idx.add("c0", "RCC column M25 grade", {"page": 5})
        idx.add("c1", "MEP plumbing layout", {"page": 12})
        idx.build()
        results = idx.search("concrete grade for columns", top_k=3)
    """

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._id_to_idx: Dict[str, int] = {}  # chunk_id → list index
        self._built = False

        # TF-IDF state (populated by build())
        self._vocab: Dict[str, int] = {}  # token → column index
        self._idf: Any = None             # numpy array (vocab_size,)
        self._tfidf_matrix: Any = None    # numpy array (n_chunks, vocab_size)
        self._norms: Any = None           # numpy array (n_chunks,) — L2 norms

    # ----- Adding chunks -----

    def add(self, chunk_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a text chunk to the index.

        Args:
            chunk_id: Unique identifier for this chunk.
            text: Raw text content.
            metadata: Optional metadata dict (page, doc_type, source, etc.).
        """
        if chunk_id in self._id_to_idx:
            logger.debug("Duplicate chunk_id '%s' — skipping", chunk_id)
            return
        tokens = tokenize(text)
        if not tokens:
            return
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            tokens=tokens,
            metadata=metadata or {},
        )
        self._id_to_idx[chunk_id] = len(self.chunks)
        self.chunks.append(chunk)
        self._built = False  # Invalidate

    def add_batch(self, items: List[Tuple[str, str, Dict[str, Any]]]) -> int:
        """Add multiple chunks at once.

        Args:
            items: List of (chunk_id, text, metadata) tuples.

        Returns:
            Number of chunks actually added (excludes duplicates/empty).
        """
        before = len(self.chunks)
        for chunk_id, text, meta in items:
            self.add(chunk_id, text, meta)
        return len(self.chunks) - before

    # ----- Building the index -----

    def build(self) -> None:
        """Build the TF-IDF matrix from all added chunks.

        Must be called after adding chunks and before searching.
        Re-calling build() after adding more chunks is safe.
        """
        import numpy as np

        n = len(self.chunks)
        if n == 0:
            self._built = True
            return

        # Step 1: Build vocabulary from all tokens
        vocab: Dict[str, int] = {}
        for chunk in self.chunks:
            for token in set(chunk.tokens):  # unique per doc for DF
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab
        vocab_size = len(vocab)

        if vocab_size == 0:
            self._built = True
            return

        # Step 2: Compute document frequency (DF) for each term
        df = np.zeros(vocab_size, dtype=np.float64)
        for chunk in self.chunks:
            seen = set()
            for token in chunk.tokens:
                if token not in seen:
                    df[vocab[token]] += 1
                    seen.add(token)

        # Step 3: Compute IDF = log(N / df) + 1  (smooth IDF)
        self._idf = np.log(n / (df + 1e-10)) + 1.0

        # Step 4: Build TF-IDF matrix (n_chunks × vocab_size)
        tfidf = np.zeros((n, vocab_size), dtype=np.float64)
        for i, chunk in enumerate(self.chunks):
            tf_counts = Counter(chunk.tokens)
            total_tokens = len(chunk.tokens)
            for token, count in tf_counts.items():
                col = vocab[token]
                tf = count / total_tokens if total_tokens > 0 else 0.0  # Normalized TF
                tfidf[i, col] = tf * self._idf[col]

        # Step 5: Compute L2 norms for fast cosine similarity
        self._norms = np.linalg.norm(tfidf, axis=1)
        self._norms[self._norms == 0] = 1e-10  # Avoid division by zero

        self._tfidf_matrix = tfidf
        self._built = True
        logger.info("VectorIndex built: %d chunks, %d vocab terms", n, vocab_size)

    # ----- Searching -----

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
        filter_fn: Optional[Any] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for chunks most similar to the query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            min_score: Minimum cosine similarity score (0.0–1.0).
            filter_fn: Optional callable(metadata) → bool to pre-filter chunks.

        Returns:
            List of (chunk_id, similarity_score, metadata) tuples,
            sorted by score descending.
        """
        import numpy as np

        if not self._built:
            self.build()

        if not self.chunks or self._tfidf_matrix is None:
            return []

        # Vectorize query using same TF-IDF vocabulary
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        vocab_size = len(self._vocab)
        q_vec = np.zeros(vocab_size, dtype=np.float64)
        q_tf = Counter(query_tokens)
        q_total = len(query_tokens)

        has_vocab_match = False
        for token, count in q_tf.items():
            if token in self._vocab:
                col = self._vocab[token]
                tf = count / q_total
                q_vec[col] = tf * self._idf[col]
                has_vocab_match = True

        if not has_vocab_match:
            return []

        # Cosine similarity: dot(q, doc) / (||q|| * ||doc||)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-10:
            return []

        similarities = self._tfidf_matrix.dot(q_vec) / (self._norms * q_norm)

        # Apply filter function if provided
        if filter_fn:
            for i, chunk in enumerate(self.chunks):
                if not filter_fn(chunk.metadata):
                    similarities[i] = 0.0

        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            chunk = self.chunks[idx]
            results.append((chunk.chunk_id, score, chunk.metadata))

        return results

    # ----- Relevance scoring -----

    def relevance_score(self, text_a: str, text_b: str) -> float:
        """Compute TF-IDF cosine similarity between two text strings.

        Utility method for pairwise comparison without needing to add
        chunks to the index. Uses the index's vocabulary if built,
        otherwise builds a temporary mini-vocabulary.

        Args:
            text_a: First text string.
            text_b: Second text string.

        Returns:
            Cosine similarity score (0.0–1.0).
        """
        import numpy as np

        tokens_a = tokenize(text_a)
        tokens_b = tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0

        # Build mini vocabulary from both texts
        all_tokens = set(tokens_a) | set(tokens_b)
        mini_vocab = {t: i for i, t in enumerate(all_tokens)}
        v_size = len(mini_vocab)

        # Simple TF vectors (no IDF for pairwise — just term frequency)
        vec_a = np.zeros(v_size, dtype=np.float64)
        vec_b = np.zeros(v_size, dtype=np.float64)

        for t in tokens_a:
            vec_a[mini_vocab[t]] += 1
        for t in tokens_b:
            vec_b[mini_vocab[t]] += 1

        # Normalize
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    # ----- Persistence -----

    def save(self, path: str) -> None:
        """Save the index to a JSON file.

        Saves chunk data and metadata for reconstruction. The TF-IDF
        matrix is NOT serialized — call build() after load().

        Args:
            path: File path to save to.
        """
        data = {
            "version": "1.0",
            "chunk_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("VectorIndex saved: %d chunks → %s", len(self.chunks), path)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load an index from a JSON file and rebuild.

        Args:
            path: File path to load from.

        Returns:
            Rebuilt VectorIndex instance.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vector index not found: {path}")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        idx = cls()
        for chunk_data in data.get("chunks", []):
            idx.add(
                chunk_id=chunk_data["chunk_id"],
                text=chunk_data["text"],
                metadata=chunk_data.get("metadata", {}),
            )
        idx.build()
        logger.info("VectorIndex loaded: %d chunks from %s", len(idx.chunks), path)
        return idx

    # ----- Info -----

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)

    @property
    def vocab_size(self) -> int:
        """Number of unique terms in the vocabulary."""
        return len(self._vocab)

    @property
    def is_built(self) -> bool:
        """Whether the TF-IDF matrix has been computed."""
        return self._built

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        source_counts: Dict[str, int] = {}
        for chunk in self.chunks:
            src = chunk.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        return {
            "chunk_count": len(self.chunks),
            "vocab_size": len(self._vocab),
            "built": self._built,
            "sources": source_counts,
        }

    def __repr__(self) -> str:
        return f"VectorIndex(chunks={len(self.chunks)}, vocab={len(self._vocab)}, built={self._built})"


# =============================================================================
# INDEX BUILDER — create index from analysis payload
# =============================================================================

def build_index_from_payload(payload: dict) -> VectorIndex:
    """Build a VectorIndex from a complete analysis payload.

    Indexes all text sources in the payload:
    - OCR text cache (page-level chunks)
    - Blockers (title + description + evidence)
    - RFIs (question + why_it_matters)
    - Requirements by trade
    - Commercial terms
    - Quantities / BOQ items
    - Structural takeoff elements

    Args:
        payload: The analysis.json payload dict.

    Returns:
        Built VectorIndex ready for search.
    """
    idx = VectorIndex()
    chunk_n = 0

    # --- 1. OCR text cache (largest source — one chunk per page) ---
    ocr_cache = payload.get("ocr_text_cache", {})
    for page_key, text in ocr_cache.items():
        if not text or not isinstance(text, str):
            continue
        page_idx = int(page_key) if str(page_key).isdigit() else -1
        # Split long pages into ~500-char sub-chunks for better retrieval
        if len(text) > 600:
            segments = _split_text(text, max_len=500, overlap=50)
            for seg_i, seg in enumerate(segments):
                cid = f"ocr-p{page_idx}-s{seg_i}"
                idx.add(cid, seg, {
                    "source": "ocr",
                    "page": page_idx,
                    "segment": seg_i,
                })
                chunk_n += 1
        else:
            cid = f"ocr-p{page_idx}"
            idx.add(cid, text, {"source": "ocr", "page": page_idx})
            chunk_n += 1

    # --- 2. Blockers ---
    for b in payload.get("blockers", []):
        bid = b.get("id", b.get("blocker_id", f"blocker-{chunk_n}"))
        parts = [
            b.get("title", ""),
            b.get("description", ""),
            b.get("missing_dependency", ""),
        ]
        # Include fix actions
        for fa in b.get("fix_actions", []):
            if isinstance(fa, str):
                parts.append(fa)
            elif isinstance(fa, dict):
                parts.append(fa.get("action", ""))
        # Coerce to str to guard against list/non-str values in payload fields
        text = " ".join(str(p) for p in parts if p)
        if text:
            idx.add(f"blocker-{bid}", text, {
                "source": "blocker",
                "id": bid,
                "severity": b.get("severity", "MEDIUM"),
                "trade": b.get("trade", ""),
            })
            chunk_n += 1

    # --- 3. RFIs ---
    for r in payload.get("rfis", []):
        rid = r.get("id", r.get("rfi_id", f"rfi-{chunk_n}"))
        parts = [
            r.get("question", ""),
            r.get("why_it_matters", ""),
            r.get("suggested_resolution", ""),
        ]
        # Coerce to str to guard against list/non-str values in payload fields
        text = " ".join(str(p) for p in parts if p)
        if text:
            idx.add(f"rfi-{rid}", text, {
                "source": "rfi",
                "id": rid,
                "trade": r.get("trade", ""),
                "priority": r.get("priority", ""),
            })
            chunk_n += 1

    # --- 4. Requirements by trade ---
    reqs = payload.get("requirements_by_trade", {})
    if isinstance(reqs, dict):
        for trade, req_list in reqs.items():
            if not isinstance(req_list, list):
                continue
            for ri, req in enumerate(req_list):
                req_text = req if isinstance(req, str) else str(req.get("text", req))
                if req_text:
                    idx.add(f"req-{trade}-{ri}", req_text, {
                        "source": "requirement",
                        "trade": trade,
                    })
                    chunk_n += 1

    # --- 5. Commercial terms ---
    commercial = payload.get("commercial_terms", {})
    if not commercial:
        # Sprint 26: also check extraction_summary
        commercial = payload.get("extraction_summary", {}).get("commercial_terms", {})
    if isinstance(commercial, dict):
        for key, val in commercial.items():
            val_str = str(val) if val else ""
            if val_str and len(val_str) > 3:
                idx.add(f"commercial-{key}", f"{key}: {val_str}", {
                    "source": "commercial",
                    "field": key,
                })
                chunk_n += 1

    # --- 6. Quantities ---
    quantities = payload.get("quantities", [])
    if isinstance(quantities, list):
        for qi, q in enumerate(quantities):
            if isinstance(q, dict):
                parts = [q.get("description", ""), q.get("trade", ""),
                         str(q.get("quantity", "")), q.get("unit", "")]
                text = " ".join(p for p in parts if p)
            else:
                text = str(q)
            if text and len(text) > 5:
                idx.add(f"qty-{qi}", text, {
                    "source": "quantity",
                    "trade": q.get("trade", "") if isinstance(q, dict) else "",
                })
                chunk_n += 1

    # --- 7. Structural takeoff ---
    structural = payload.get("structural_takeoff", {})
    if isinstance(structural, dict):
        for element_type, data in structural.items():
            if isinstance(data, dict):
                text = f"{element_type}: {json.dumps(data)}"
            elif isinstance(data, list):
                text = f"{element_type}: " + "; ".join(str(d) for d in data)
            else:
                text = f"{element_type}: {data}"
            if text and len(text) > 5:
                idx.add(f"structural-{element_type}", text, {
                    "source": "structural",
                    "element_type": element_type,
                })
                chunk_n += 1

    # --- 8. Trade coverage ---
    for tc in payload.get("trade_coverage", []):
        if isinstance(tc, dict):
            trade = tc.get("trade", "")
            text = f"{trade} coverage {tc.get('coverage_pct', 0)}%"
            if tc.get("blocked_count"):
                text += f", {tc['blocked_count']} blocked items"
            idx.add(f"trade-{trade}", text, {
                "source": "trade_coverage",
                "trade": trade,
                "coverage_pct": tc.get("coverage_pct", 0),
            })
            chunk_n += 1

    idx.build()
    logger.info("Payload index built: %d chunks from %d sources",
                idx.size, len(idx.stats().get("sources", {})))
    return idx


def _split_text(text: str, max_len: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping segments for chunked indexing.

    Tries to break at sentence boundaries. Falls back to word boundaries.

    Args:
        text: Text to split.
        max_len: Maximum characters per segment.
        overlap: Character overlap between segments.

    Returns:
        List of text segments.
    """
    if len(text) <= max_len:
        return [text]

    segments = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            # Look for period/newline near end
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                last_sep = text.rfind(sep, start + max_len // 2, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end - overlap if end < len(text) else len(text)

    return segments if segments else [text]
