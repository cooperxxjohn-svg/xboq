"""
src/embeddings/chroma_store.py

ChromaDB-backed persistent vector store for xBOQ bid packages.

One ChromaDB collection per project_id.  Each document = one indexable chunk
derived from the analysis payload.  Metadata filters allow trade- or source-
scoped retrieval.

Chunk sources indexed:
  - page_texts  (OCR text per page)
  - boq_items   (description + unit + section)
  - spec_items  (description + section + standards_codes)
  - rfi_items   (description + trade + severity)
  - structural  (description + unit)
  - commercial  (payment/LD/warranty terms)
  - room_schedule (room name + area + dimensions)
  - qto_summary  (free-text blurb of QTO totals)
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Max chars per chunk for OCR pages (long pages are split with overlap)
_PAGE_CHUNK_CHARS = 1500
_OVERLAP_CHARS = 150


@dataclass
class SearchResult:
    text: str
    score: float          # cosine similarity 0-1 (higher = more relevant)
    source: str           # "boq" | "spec" | "rfi" | "ocr" | "structural" | ...
    metadata: Dict[str, Any] = field(default_factory=dict)


class BidChromaStore:
    """
    ChromaDB collection per project.

    Parameters
    ----------
    project_id : str
        Used as the ChromaDB collection name (sanitised to alphanumeric).
    persist_dir : str
        Directory where ChromaDB persists data on disk.
    """

    def __init__(self, project_id: str, persist_dir: str = ".chroma"):
        import chromadb  # type: ignore
        self._project_id = project_id
        self._collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:63] or "xboq"
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "BidChromaStore: collection=%s, existing_docs=%d",
            self._collection_name, self._col.count(),
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_payload(self, payload: dict, embedder) -> int:
        """
        Index all payload sections into ChromaDB.

        Clears the collection first (re-index on every pipeline run so
        content always reflects the latest analysis).

        Returns
        -------
        int
            Number of chunks indexed.
        """
        # Clear existing documents for this project
        self._clear()

        chunks: List[tuple] = []  # (id, text, metadata)

        # ── OCR page texts ──────────────────────────────────────────
        for page_idx, text in payload.get("ocr_text_by_page", {}).items():
            if not text or not text.strip():
                continue
            doc_type = self._page_doc_type(page_idx, payload)
            for i, chunk_text in enumerate(BidChromaStore._split_text(text, target_chunk_size=_PAGE_CHUNK_CHARS)):
                cid = _make_id(f"ocr:{page_idx}:{i}")
                chunks.append((
                    cid,
                    chunk_text,
                    {"source": "ocr", "page": int(page_idx), "doc_type": doc_type},
                ))

        # ── BOQ items ────────────────────────────────────────────────
        for i, item in enumerate(payload.get("boq_items", [])):
            text = _boq_text(item)
            if not text:
                continue
            cid = _make_id(f"boq:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {
                    "source": "boq",
                    "trade": str(item.get("trade", "general")),
                    "page": int(item.get("source_page") or 0),
                    "unit": str(item.get("unit") or ""),
                },
            ))

        # ── Spec / line items ────────────────────────────────────────
        for i, item in enumerate(payload.get("spec_items", [])):
            text = _spec_text(item)
            if not text:
                continue
            cid = _make_id(f"spec:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {
                    "source": "spec",
                    "trade": str(item.get("trade", "general")),
                    "page": int(item.get("source_page") or 0),
                },
            ))

        # ── RFI items ────────────────────────────────────────────────
        for i, rfi in enumerate(payload.get("rfis", [])):
            text = _rfi_text(rfi)
            if not text:
                continue
            cid = _make_id(f"rfi:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {
                    "source": "rfi",
                    "trade": str(rfi.get("trade", "general")),
                    "severity": str(rfi.get("severity", "MEDIUM")),
                    "page": int(rfi.get("source_page") or 0),
                },
            ))

        # ── Structural items (from qto_summary) ─────────────────────
        qto = payload.get("qto_summary", {})
        for i, item in enumerate(qto.get("structural_items", [])):
            text = _boq_text(item)
            if not text:
                continue
            cid = _make_id(f"struct:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {"source": "structural", "trade": "structural", "page": int(item.get("source_page") or 0)},
            ))

        # ── Extraction requirements ──────────────────────────────────
        extraction = payload.get("extraction_summary", {})
        for i, req in enumerate(extraction.get("requirements", [])):
            text = str(req.get("description") or req.get("text") or "").strip()
            if not text:
                continue
            cid = _make_id(f"req:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {
                    "source": "requirement",
                    "trade": str(req.get("trade", "general")),
                    "page": int(req.get("source_page") or 0),
                },
            ))

        # ── Blockers ────────────────────────────────────────────────
        for i, blk in enumerate(payload.get("blockers", [])):
            text = str(blk.get("description") or blk.get("title") or "").strip()
            if not text:
                continue
            cid = _make_id(f"blk:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {
                    "source": "blocker",
                    "trade": str(blk.get("trade", "general")),
                    "severity": str(blk.get("severity", "MEDIUM")),
                    "page": int(blk.get("source_page") or 0),
                },
            ))

        # ── Commercial terms ─────────────────────────────────────────
        for i, term in enumerate(payload.get("contractual_items", [])):
            text = str(term.get("description") or term.get("clause") or "").strip()
            if not text:
                continue
            cid = _make_id(f"comm:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {"source": "commercial", "page": int(term.get("source_page") or 0)},
            ))

        # ── Room schedule (visual measurement) ──────────────────────
        room_sched = qto.get("vmeas_room_schedule") or []
        for i, room in enumerate(room_sched):
            text = _room_text(room)
            if not text:
                continue
            cid = _make_id(f"room:{i}:{text[:40]}")
            chunks.append((
                cid,
                text,
                {"source": "room_schedule", "page": int(room.get("source_page") or 0)},
            ))

        # ── QTO summary blurb ────────────────────────────────────────
        qto_blurb = _qto_blurb(payload)
        if qto_blurb:
            chunks.append((_make_id("qto_blurb"), qto_blurb, {"source": "qto_summary"}))

        if not chunks:
            logger.warning("BidChromaStore.index_payload: no chunks to index")
            return 0

        # Embed + store in batches
        ids, texts, metadatas = zip(*chunks)
        embeddings = embedder.embed(list(texts)).tolist()

        _BATCH = 500
        for start in range(0, len(ids), _BATCH):
            self._col.add(
                ids=list(ids[start:start + _BATCH]),
                documents=list(texts[start:start + _BATCH]),
                embeddings=embeddings[start:start + _BATCH],
                metadatas=list(metadatas[start:start + _BATCH]),
            )

        logger.info("BidChromaStore: indexed %d chunks for project %s", len(chunks), self._project_id)
        return len(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        embedder,
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        Semantic search.

        Parameters
        ----------
        query : str
            Natural language question or keyword string.
        embedder : Embedder
            Used to embed the query.
        n_results : int
            Top-k results to return.
        where : dict, optional
            ChromaDB metadata filter, e.g. {"source": "boq"} or
            {"trade": {"$in": ["structural", "civil"]}}.

        Returns
        -------
        List[SearchResult] ordered by descending similarity.
        """
        if self._col.count() == 0:
            return []

        q_vec = embedder.embed_one(query).tolist()
        n = min(n_results, self._col.count())

        kwargs: dict = dict(
            query_embeddings=[q_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        try:
            res = self._col.query(**kwargs)
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            return []

        results: List[SearchResult] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance = 1 - cosine_similarity
            score = max(0.0, 1.0 - float(dist))
            results.append(SearchResult(
                text=doc,
                score=score,
                source=str(meta.get("source", "unknown")),
                metadata=dict(meta),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search_by_trade(
        self,
        query: str,
        trade: str,
        embedder,
        n_results: int = 10,
    ) -> List[SearchResult]:
        """Trade-scoped semantic search — only returns chunks tagged with the given trade."""
        try:
            return self.search(query, embedder, n_results=n_results,
                              where={"trade": trade})
        except Exception:
            # Fall back to unfiltered if trade filter fails (e.g., no chunks with that trade)
            return self.search(query, embedder, n_results=n_results)

    def search_by_source(
        self,
        query: str,
        source_types: list,
        embedder,
        n_results: int = 10,
    ) -> List[SearchResult]:
        """Source-filtered semantic search."""
        if len(source_types) == 1:
            where = {"source": source_types[0]}
        else:
            where = {"source": {"$in": source_types}}
        try:
            return self.search(query, embedder, n_results=n_results, where=where)
        except Exception:
            return self.search(query, embedder, n_results=n_results)

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_text(
        text: str,
        target_chunk_size: int = 800,
        overlap_sentences: int = 2,
    ) -> List[str]:
        """
        Semantic paragraph-aware text splitter.

        Strategy:
        1. Split on paragraph breaks (double newline / blank line)
        2. Merge short paragraphs until target_chunk_size reached
        3. Split long paragraphs at sentence boundaries (. ; \\n)
        4. Prepend last `overlap_sentences` sentences of previous chunk to next

        This keeps BOQ table rows together and avoids splitting
        specification clauses mid-sentence.
        """
        if not text or not text.strip():
            return []

        import re

        # ── 1. Split into paragraphs ─────────────────────────────────────────────
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\r\n\s*\r\n', text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        # ── 2. Split long paragraphs at sentence boundaries ──────────────────────
        def split_sentences(para: str) -> List[str]:
            # Split on ". ", "; ", ".\n", ";\n" but keep the punctuation
            parts = re.split(r'(?<=[.;])\s+(?=[A-Z0-9\(])', para)
            return [p.strip() for p in parts if p.strip()]

        sentences: List[str] = []
        for para in paragraphs:
            if len(para) <= target_chunk_size:
                sentences.append(para)
            else:
                sentences.extend(split_sentences(para))

        if not sentences:
            return [text[:target_chunk_size]] if text else []

        # ── 3. Merge sentences into chunks of ~target_chunk_size ─────────────────
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > target_chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Keep last `overlap_sentences` for overlap
                if overlap_sentences > 0:
                    current_sentences = current_sentences[-overlap_sentences:]
                    current_len = sum(len(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_len = 0
            current_sentences.append(sent)
            current_len += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # ── 4. Hard-split any chunk still over 2× target (safety net) ────────────
        final_chunks: List[str] = []
        hard_limit = target_chunk_size * 2
        for chunk in chunks:
            if len(chunk) <= hard_limit:
                final_chunks.append(chunk)
            else:
                # Fall back to word-boundary split
                words = chunk.split()
                part: List[str] = []
                part_len = 0
                for word in words:
                    if part_len + len(word) + 1 > target_chunk_size and part:
                        final_chunks.append(" ".join(part))
                        part = part[-overlap_sentences * 5:] if overlap_sentences else []
                        part_len = sum(len(w) for w in part)
                    part.append(word)
                    part_len += len(word) + 1
                if part:
                    final_chunks.append(" ".join(part))

        return [c for c in final_chunks if c.strip()]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._col.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection_name)
        logger.info("BidChromaStore: deleted collection %s", self._collection_name)

    def _clear(self) -> None:
        """Remove all documents from the collection (keep collection itself)."""
        n = self._col.count()
        if n == 0:
            return
        # Get all IDs and delete
        all_ids = self._col.get(include=[])["ids"]
        if all_ids:
            self._col.delete(ids=all_ids)
        logger.debug("BidChromaStore: cleared %d existing docs", n)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_id(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _split_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _boq_text(item: dict) -> str:
    parts = [
        str(item.get("description") or ""),
        str(item.get("section") or ""),
        str(item.get("unit") or ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def _spec_text(item: dict) -> str:
    parts = [
        str(item.get("description") or ""),
        str(item.get("section") or ""),
        " ".join(item.get("standards_codes") or []),
    ]
    return " | ".join(p for p in parts if p).strip()


def _rfi_text(rfi: dict) -> str:
    parts = [
        str(rfi.get("question") or rfi.get("description") or ""),
        str(rfi.get("trade") or ""),
        str(rfi.get("severity") or ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def _room_text(room: dict) -> str:
    name = room.get("name") or room.get("room_name") or ""
    area = room.get("area_sqm")
    dims = f"{room.get('dim_l', '')}×{room.get('dim_w', '')}" if room.get("dim_l") else ""
    parts = [str(name), f"area {area} sqm" if area else "", dims]
    return " ".join(p for p in parts if p).strip()


def _qto_blurb(payload: dict) -> str:
    qto = payload.get("qto_summary", {})
    if not qto:
        return ""
    lines = []
    if qto.get("vmeas_area_sqm"):
        lines.append(f"Total measured area: {qto['vmeas_area_sqm']:.0f} sqm")
    if qto.get("grand_total_inr"):
        lines.append(f"Estimated cost: ₹{qto['grand_total_inr']:,.0f}")
    if qto.get("st_concrete_cum"):
        lines.append(f"Concrete: {qto['st_concrete_cum']:.1f} cum")
    if qto.get("st_steel_kg"):
        lines.append(f"Steel: {qto['st_steel_kg']:.0f} kg")
    if qto.get("total_spec_items"):
        lines.append(f"QTO line items: {qto['total_spec_items']}")
    return ". ".join(lines)


def _page_doc_type(page_idx, payload: dict) -> str:
    """Look up doc_type for a page from diagnostics.page_index."""
    try:
        pages = payload.get("diagnostics", {}).get("page_index", {}).get("pages", [])
        for p in pages:
            if p.get("page_idx") == int(page_idx):
                return str(p.get("doc_type", "unknown"))
    except Exception:
        pass
    return "unknown"
