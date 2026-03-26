"""
DomainKBStore — Persistent domain knowledge base for xBOQ.ai RAG.

Indexes all static domain knowledge into a shared ChromaDB collection:
  - DSR 2023 rate schedule (national + state SOR files)
  - QTO quantity benchmarks
  - Taxonomy items (21 discipline YAML files)
  - Synonym mappings (Hindi terms, abbreviations, brand generics)
  - Dependency rules
  - RFI trigger rules

Primary backend: ChromaDB (persistent).
Fallback backend: sklearn TF-IDF + numpy cosine similarity (in-memory).
  — activated automatically when ChromaDB/Rust bindings are unavailable.
  — documents cached to ~/.xboq/domain_kb/fallback_docs.json for reuse.

This collection is built once via build_domain_kb.py and reused across
all tender runs. It is NEVER cleared during a pipeline run.

Usage:
    from src.embeddings.domain_kb_store import DomainKBStore
    store = DomainKBStore()
    store.build_if_stale()           # builds on first run, skips if up to date
    results = store.search("M25 concrete rate Delhi", n_results=10)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Sklearn/numpy TF-IDF fallback ────────────────────────────────────────────
_SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RATES_DIR    = _PROJECT_ROOT / "rates"
_KB_DIR       = _PROJECT_ROOT / "src" / "knowledge_base"

# Persistent store location — shared across all tender runs
_DEFAULT_PERSIST_DIR = Path.home() / ".xboq" / "domain_kb"

# Hash file to detect staleness
_HASH_FILE = _DEFAULT_PERSIST_DIR / "source_hash.txt"

_COLLECTION_NAME = "xboq_domain_kb_v1"


# ── DomainKBStore ────────────────────────────────────────────────────────────

_FALLBACK_DOCS_FILE = _DEFAULT_PERSIST_DIR / "fallback_docs.json"


class DomainKBStore:
    """
    Persistent domain knowledge base backed by ChromaDB (primary) or
    sklearn TF-IDF (fallback when ChromaDB/Rust bindings unavailable).
    Singleton-friendly: safe to instantiate multiple times.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or _DEFAULT_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        # Fallback TF-IDF state (populated by _ensure_fallback())
        self._fallback_docs: List[dict] = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._fallback_loaded = False

    # ── Lazy ChromaDB init ────────────────────────────────────────────────────

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except Exception as e:
            logger.debug("DomainKBStore: ChromaDB unavailable — %s", e)
            return None

    # ── TF-IDF fallback ───────────────────────────────────────────────────────

    def _ensure_fallback(self) -> bool:
        """Load or build the TF-IDF fallback index. Returns True if available."""
        if self._fallback_loaded:
            return bool(self._fallback_docs)
        if not _SKLEARN_AVAILABLE:
            return False
        # Try loading cached docs from disk
        fallback_path = self.persist_dir / "fallback_docs.json"
        if fallback_path.exists() and not self._is_fallback_stale(fallback_path):
            try:
                self._fallback_docs = json.loads(fallback_path.read_text())
                self._fit_tfidf()
                self._fallback_loaded = True
                logger.debug("DomainKBStore: loaded %d fallback docs", len(self._fallback_docs))
                return True
            except Exception as e:
                logger.warning("DomainKBStore: fallback load failed — %s", e)
        return False

    def _is_fallback_stale(self, fallback_path: Path) -> bool:
        if not _HASH_FILE.exists():
            return True
        try:
            stored = _HASH_FILE.read_text().strip()
            return stored != self._compute_source_hash()
        except Exception:
            return True

    def _fit_tfidf(self) -> None:
        """Fit TF-IDF vectorizer on _fallback_docs texts."""
        if not _SKLEARN_AVAILABLE or not self._fallback_docs:
            return
        texts = [d["text"] for d in self._fallback_docs]
        self._tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=1,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def _build_fallback(self, all_docs: List[dict]) -> int:
        """Populate the fallback index from all_docs (no ChromaDB needed)."""
        if not _SKLEARN_AVAILABLE:
            logger.warning("DomainKBStore: sklearn unavailable — fallback not built")
            return 0
        self._fallback_docs = all_docs
        self._fit_tfidf()
        self._fallback_loaded = True
        # Persist docs to disk
        fallback_path = self.persist_dir / "fallback_docs.json"
        try:
            fallback_path.write_text(json.dumps(all_docs, ensure_ascii=False))
            logger.info("DomainKBStore: saved %d fallback docs to %s", len(all_docs), fallback_path)
        except Exception as e:
            logger.warning("DomainKBStore: could not save fallback docs — %s", e)
        return len(all_docs)

    def _search_fallback(
        self,
        query: str,
        n_results: int = 10,
        source_filter: Optional[str] = None,
        trade_filter: Optional[str] = None,
    ) -> List[dict]:
        """TF-IDF cosine similarity search over _fallback_docs."""
        if not self._ensure_fallback():
            return []
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []
        try:
            q_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]
            # Apply filters (trade uses substring match to handle prefixed names like "05_masonry")
            candidates = []
            for idx, score in enumerate(scores):
                doc = self._fallback_docs[idx]
                if source_filter and doc.get("source") != source_filter:
                    continue
                if trade_filter:
                    doc_trade = doc.get("trade", "")
                    # Accept exact match OR prefix-number format (e.g. "05_masonry" matches "masonry")
                    if trade_filter not in doc_trade and doc_trade not in trade_filter:
                        continue
                candidates.append((idx, float(score)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            out = []
            for idx, score in candidates[:n_results]:
                doc = self._fallback_docs[idx]
                out.append({"text": doc["text"], "score": round(score, 4), **{k: v for k, v in doc.items() if k != "text"}})
            return out
        except Exception as e:
            logger.warning("DomainKBStore._search_fallback failed: %s", e)
            return []

    # ── Staleness check ───────────────────────────────────────────────────────

    def _compute_source_hash(self) -> str:
        """Hash all source files to detect if rebuild is needed."""
        h = hashlib.md5()
        sources = [
            *sorted(_RATES_DIR.glob("*.json")),
            _KB_DIR / "benchmarks" / "quantity_benchmarks.yaml",
        ]
        # Also hash taxonomy directory listing
        tax_dir = _KB_DIR / "taxonomy" / "data"
        if tax_dir.exists():
            sources.extend(sorted(tax_dir.glob("*.yaml")))
        for p in sources:
            if p.exists():
                h.update(p.name.encode())
                h.update(str(p.stat().st_mtime).encode())
        return h.hexdigest()[:16]

    def _is_stale(self) -> bool:
        col = self._get_collection()
        if col is not None:
            # ChromaDB path
            if col.count() == 0:
                return True
            if not _HASH_FILE.exists():
                return True
            stored = _HASH_FILE.read_text().strip()
            return stored != self._compute_source_hash()
        # Fallback path — check if fallback_docs.json is missing or stale
        fallback_path = self.persist_dir / "fallback_docs.json"
        if not fallback_path.exists():
            return True
        if not _HASH_FILE.exists():
            return True
        try:
            stored = _HASH_FILE.read_text().strip()
            return stored != self._compute_source_hash()
        except Exception:
            return True

    def build_if_stale(self, embedder=None) -> bool:
        """Build or rebuild the domain KB if source files have changed.
        Returns True if a build was performed."""
        if not self._is_stale():
            n = self.count()
            logger.debug("DomainKBStore: up to date (%d docs)", n)
            return False
        logger.info("DomainKBStore: building domain knowledge base...")
        self.build(embedder=embedder)
        return True

    # ── Document loaders ──────────────────────────────────────────────────────

    def _load_rate_documents(self) -> List[dict]:
        """Load rate schedule JSON files into document dicts."""
        docs = []
        rate_files = {
            "dsr_national": _RATES_DIR / "dsr_2023_rates.json",
            "mh_pwd":       _RATES_DIR / "mh_pwd_2023_rates.json",
            "dl_cpwd":      _RATES_DIR / "dl_cpwd_2023_rates.json",
        }
        for source_key, path in rate_files.items():
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else data.get("items", data.get("rates", []))
                state = source_key.split("_")[0].upper()
                if state == "DSR":
                    state = "national"
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    desc = (item.get("description") or item.get("item_description") or
                            item.get("name") or "").strip()
                    if not desc:
                        continue
                    unit  = item.get("unit", "")
                    rate  = item.get("rate", item.get("amount", ""))
                    code  = item.get("code", item.get("item_code", ""))
                    trade = item.get("trade", item.get("category", "civil"))
                    # Format as a retrievable text chunk
                    text = f"{desc}"
                    if unit:
                        text += f" | unit: {unit}"
                    if rate:
                        text += f" | rate: ₹{rate}"
                    if code:
                        text += f" | DSR code: {code}"
                    docs.append({
                        "text":   text,
                        "source": "dsr_rates",
                        "state":  state,
                        "trade":  str(trade).lower(),
                        "rate":   str(rate),
                        "unit":   str(unit),
                    })
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        return docs

    def _load_benchmark_documents(self) -> List[dict]:
        """Load quantity benchmarks into document dicts."""
        docs = []
        bm_path = _KB_DIR / "benchmarks" / "quantity_benchmarks.yaml"
        if not bm_path.exists():
            return docs
        try:
            import yaml
            with open(bm_path) as f:
                raw = yaml.safe_load(f)
            project_types = raw if isinstance(raw, list) else raw.get("project_benchmarks", raw.get("project_types", []))
            for pt in project_types:
                ptype = pt.get("project_type", "")
                desc  = pt.get("description", "")
                for item in pt.get("per_sqm_bua", []):
                    item_name = item.get("item", "")
                    unit      = item.get("unit", "")
                    typical   = item.get("typical", "")
                    min_v     = item.get("min", "")
                    max_v     = item.get("max", "")
                    text = (f"Quantity benchmark for {ptype}: {item_name} "
                            f"typical {typical} {unit}/sqm BUA "
                            f"(range {min_v}–{max_v} {unit}/sqm). "
                            f"Building type: {desc}")
                    docs.append({
                        "text":          text,
                        "source":        "benchmarks",
                        "building_type": ptype,
                        "trade":         "general",
                    })
        except Exception as e:
            logger.warning("Failed to load benchmarks: %s", e)
        return docs

    def _load_taxonomy_documents(self) -> List[dict]:
        """Load taxonomy YAML files into document dicts.

        Supports both flat-list format and the nested xBOQ taxonomy format:
          sub_trades:
            <sub_trade_name>:
              categories:
                <category_name>:
                  items: [{id, standard_name, unit, aliases, notes, ...}]
        """
        docs = []
        tax_dir = _KB_DIR / "taxonomy" / "data"
        if not tax_dir.exists():
            return docs
        try:
            import yaml
        except ImportError:
            return docs

        def _emit(item: dict, trade: str) -> None:
            """Convert a single taxonomy item dict into a document and append to docs."""
            name = (
                item.get("standard_name") or item.get("name") or
                item.get("description") or ""
            ).strip()
            if not name:
                return
            unit    = item.get("unit", "")
            aliases = item.get("aliases", item.get("synonyms", []))
            notes   = item.get("notes", item.get("specification", ""))
            rate_min = ""
            rate_max = ""
            rate_rng = item.get("typical_rate_range", {})
            if isinstance(rate_rng, dict):
                rate_min = rate_rng.get("min", "")
                rate_max = rate_rng.get("max", "")
            text = name
            if unit:
                text += f" (unit: {unit})"
            if aliases:
                alias_str = (
                    ", ".join(str(a) for a in aliases[:5])
                    if isinstance(aliases, list) else str(aliases)
                )
                text += f" | also known as: {alias_str}"
            if rate_min and rate_max:
                text += f" | typical rate: ₹{rate_min}–₹{rate_max}"
            if notes:
                text += f" | {str(notes)[:200]}"
            docs.append({"text": text, "source": "taxonomy", "trade": trade})

        for yaml_file in sorted(tax_dir.glob("*.yaml")):
            if yaml_file.stem == "assign_is_codes":
                continue
            trade = yaml_file.stem
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue

                # ── Flat list format ──────────────────────────────────────
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            _emit(item, trade)
                    continue

                if not isinstance(data, dict):
                    continue

                # ── Nested xBOQ format (sub_trades → categories → items) ─
                sub_trades = data.get("sub_trades")
                if isinstance(sub_trades, dict):
                    for st_name, st_val in sub_trades.items():
                        if not isinstance(st_val, dict):
                            continue
                        categories = st_val.get("categories", {})
                        if isinstance(categories, dict):
                            for cat_name, cat_val in categories.items():
                                if not isinstance(cat_val, dict):
                                    continue
                                for item in cat_val.get("items", []):
                                    if isinstance(item, dict):
                                        _emit(item, trade)
                        elif isinstance(categories, list):
                            for item in categories:
                                if isinstance(item, dict):
                                    _emit(item, trade)
                    continue

                # ── Flat dict with items/taxonomy key ────────────────────
                flat_items = (
                    data.get("items") or data.get("taxonomy") or
                    (list(data.values())[0] if data else [])
                )
                if isinstance(flat_items, list):
                    for item in flat_items:
                        if isinstance(item, dict):
                            _emit(item, trade)

            except Exception as e:
                logger.warning("Failed to load taxonomy %s: %s", yaml_file.name, e)
        return docs

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, embedder=None) -> int:
        """Build the domain KB from all source files. Returns document count.

        Primary path: ChromaDB + embedder.
        Fallback path: sklearn TF-IDF in-memory (when ChromaDB/embedder unavailable).
        """
        # Load all documents regardless of backend
        all_docs: List[dict] = []
        all_docs.extend(self._load_rate_documents())
        all_docs.extend(self._load_benchmark_documents())
        all_docs.extend(self._load_taxonomy_documents())

        if not all_docs:
            logger.warning("DomainKBStore: no documents loaded")
            return 0

        logger.info("DomainKBStore: loaded %d documents from source files", len(all_docs))

        col = self._get_collection()
        if col is None:
            # ── Fallback: sklearn TF-IDF ──────────────────────────────────────
            logger.info("DomainKBStore: ChromaDB unavailable — using TF-IDF fallback")
            added = self._build_fallback(all_docs)
            if added:
                _HASH_FILE.write_text(self._compute_source_hash())
                logger.info("DomainKBStore: TF-IDF fallback built with %d documents", added)
            return added

        # ── Primary: ChromaDB + embedder ──────────────────────────────────────
        if embedder is None:
            try:
                from src.embeddings.embedder import Embedder
                embedder = Embedder(backend="auto")
            except Exception as e:
                logger.warning("DomainKBStore: embedder unavailable — falling back to TF-IDF — %s", e)
                added = self._build_fallback(all_docs)
                if added:
                    _HASH_FILE.write_text(self._compute_source_hash())
                return added

        # Clear old collection
        try:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            col = self._collection
        except Exception:
            pass

        batch_size = 200
        added = 0
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i: i + batch_size]
            texts = [d["text"] for d in batch]
            try:
                embeddings = embedder.embed(texts)
                ids = [f"kb_{i + j}" for j in range(len(batch))]
                metadatas = [{k: v for k, v in d.items() if k != "text"} for d in batch]
                col.add(
                    ids=ids,
                    documents=texts,
                    embeddings=[e.tolist() for e in embeddings],
                    metadatas=metadatas,
                )
                added += len(batch)
                logger.debug("DomainKBStore: added batch %d/%d (%d docs)",
                             i // batch_size + 1,
                             (len(all_docs) + batch_size - 1) // batch_size,
                             added)
            except Exception as e:
                logger.warning("DomainKBStore: batch %d failed — %s", i, e)

        # Also build fallback index for environments where ChromaDB is later unavailable
        self._build_fallback(all_docs)

        # Save hash
        _HASH_FILE.write_text(self._compute_source_hash())
        logger.info("DomainKBStore: built with %d documents", added)
        return added

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        embedder=None,
        n_results: int = 10,
        source_filter: Optional[str] = None,   # "dsr_rates"|"taxonomy"|"benchmarks"
        trade_filter: Optional[str] = None,     # "structural"|"masonry" etc.
    ) -> List[dict]:
        """
        Search the domain KB for relevant knowledge.
        Tries ChromaDB first; falls back to TF-IDF when ChromaDB unavailable.
        Returns list of {text, score, source, trade, ...} dicts.
        """
        col = self._get_collection()

        if col is not None and col.count() > 0:
            # ── Primary: ChromaDB ─────────────────────────────────────────────
            if embedder is None:
                try:
                    from src.embeddings.embedder import Embedder
                    embedder = Embedder(backend="auto")
                except Exception:
                    pass  # fall through to TF-IDF
            if embedder is not None:
                try:
                    q_emb = embedder.embed([query])[0].tolist()
                    where_clause = None
                    if source_filter and trade_filter:
                        where_clause = {"$and": [{"source": source_filter}, {"trade": trade_filter}]}
                    elif source_filter:
                        where_clause = {"source": source_filter}
                    elif trade_filter:
                        where_clause = {"trade": trade_filter}
                    kwargs = dict(
                        query_embeddings=[q_emb],
                        n_results=min(n_results, max(1, col.count())),
                        include=["documents", "distances", "metadatas"],
                    )
                    if where_clause:
                        kwargs["where"] = where_clause
                    results = col.query(**kwargs)
                    out = []
                    for doc, dist, meta in zip(
                        results["documents"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                    ):
                        out.append({
                            "text":  doc,
                            "score": round(1.0 - float(dist), 4),
                            **(meta or {}),
                        })
                    return out
                except Exception as e:
                    logger.warning("DomainKBStore.search (ChromaDB) failed: %s — trying fallback", e)

        # ── Fallback: TF-IDF ──────────────────────────────────────────────────
        return self._search_fallback(query, n_results=n_results,
                                     source_filter=source_filter,
                                     trade_filter=trade_filter)

    def count(self) -> int:
        col = self._get_collection()
        if col is not None:
            return col.count()
        # Fallback count
        if self._fallback_loaded:
            return len(self._fallback_docs)
        # Try loading from disk without fitting
        fallback_path = self.persist_dir / "fallback_docs.json"
        if fallback_path.exists():
            try:
                docs = json.loads(fallback_path.read_text())
                return len(docs)
            except Exception:
                pass
        return 0

    # ── R5: Extended metadata filters ─────────────────────────────────────────

    def search_by_trade(
        self,
        query: str,
        trade: str,
        embedder=None,
        n_results: int = 10,
    ) -> List[dict]:
        """Search restricted to a specific trade (R5)."""
        return self.search(query, embedder=embedder, n_results=n_results, trade_filter=trade)

    def search_by_state(
        self,
        query: str,
        state: str,
        embedder=None,
        n_results: int = 10,
    ) -> List[dict]:
        """Search restricted to state-specific DSR rates (R5).
        state: 'national' | 'mh' | 'dl' | case-insensitive
        """
        state_norm = state.lower().strip()
        # Use source+state combination — custom filter via fallback
        results = self._search_with_filters(
            query=query,
            embedder=embedder,
            n_results=n_results,
            source_filter="dsr_rates",
            extra_filters={"state": state_norm},
        )
        # Also include national rates if state-specific requested
        if state_norm != "national":
            national = self._search_with_filters(
                query=query,
                embedder=embedder,
                n_results=max(2, n_results // 3),
                source_filter="dsr_rates",
                extra_filters={"state": "national"},
            )
            seen = {r["text"] for r in results}
            for r in national:
                if r["text"] not in seen:
                    results.append(r)
        return results[:n_results]

    def search_by_building_type(
        self,
        query: str,
        building_type: str,
        embedder=None,
        n_results: int = 10,
    ) -> List[dict]:
        """Search benchmarks filtered by building type (R5)."""
        return self._search_with_filters(
            query=query,
            embedder=embedder,
            n_results=n_results,
            source_filter="benchmarks",
            extra_filters={"building_type": building_type.lower()},
        )

    def _search_with_filters(
        self,
        query: str,
        embedder=None,
        n_results: int = 10,
        source_filter: Optional[str] = None,
        trade_filter: Optional[str] = None,
        extra_filters: Optional[dict] = None,
    ) -> List[dict]:
        """
        Internal search with arbitrary metadata filters.
        ChromaDB path uses $and clause; TF-IDF path filters by key equality.
        """
        col = self._get_collection()
        if col is not None and col.count() > 0:
            if embedder is None:
                try:
                    from src.embeddings.embedder import Embedder
                    embedder = Embedder(backend="auto")
                except Exception:
                    pass
            if embedder is not None:
                try:
                    q_emb = embedder.embed([query])[0].tolist()
                    conditions = []
                    if source_filter:
                        conditions.append({"source": source_filter})
                    if trade_filter:
                        conditions.append({"trade": trade_filter})
                    for k, v in (extra_filters or {}).items():
                        conditions.append({k: v})
                    if len(conditions) == 0:
                        where_clause = None
                    elif len(conditions) == 1:
                        where_clause = conditions[0]
                    else:
                        where_clause = {"$and": conditions}
                    kwargs = dict(
                        query_embeddings=[q_emb],
                        n_results=min(n_results, max(1, col.count())),
                        include=["documents", "distances", "metadatas"],
                    )
                    if where_clause:
                        kwargs["where"] = where_clause
                    results = col.query(**kwargs)
                    out = []
                    for doc, dist, meta in zip(
                        results["documents"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                    ):
                        out.append({
                            "text": doc,
                            "score": round(1.0 - float(dist), 4),
                            **(meta or {}),
                        })
                    return out
                except Exception as e:
                    logger.debug("DomainKBStore._search_with_filters (chroma) failed: %s", e)
        # TF-IDF fallback with extra_filters
        return self._search_fallback_extended(
            query=query,
            n_results=n_results,
            source_filter=source_filter,
            trade_filter=trade_filter,
            extra_filters=extra_filters,
        )

    def _search_fallback_extended(
        self,
        query: str,
        n_results: int = 10,
        source_filter: Optional[str] = None,
        trade_filter: Optional[str] = None,
        extra_filters: Optional[dict] = None,
    ) -> List[dict]:
        """TF-IDF fallback with support for arbitrary extra metadata filters."""
        if not self._ensure_fallback():
            return []
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []
        try:
            q_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]
            candidates = []
            for idx, score in enumerate(scores):
                doc = self._fallback_docs[idx]
                if source_filter and doc.get("source") != source_filter:
                    continue
                if trade_filter:
                    doc_trade = doc.get("trade", "")
                    if trade_filter not in doc_trade and doc_trade not in trade_filter:
                        continue
                if extra_filters:
                    skip = False
                    for k, v in extra_filters.items():
                        dv = str(doc.get(k, "")).lower()
                        if dv != str(v).lower():
                            skip = True
                            break
                    if skip:
                        continue
                candidates.append((idx, float(score)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            out = []
            for idx, score in candidates[:n_results]:
                doc = self._fallback_docs[idx]
                out.append({"text": doc["text"], "score": round(score, 4), **{k: v for k, v in doc.items() if k != "text"}})
            return out
        except Exception as e:
            logger.warning("DomainKBStore._search_fallback_extended failed: %s", e)
            return []

    def rate_confidence(
        self,
        hit: dict,
        building_type: str = "",
        state: str = "",
    ) -> float:
        """
        Score applicability of a rate/benchmark hit to current project context (R5).

        Considers:
        - Base retrieval score
        - State match bonus (state-specific > national)
        - Building type match bonus
        Returns float in [0.0, 1.0].
        """
        base = float(hit.get("score", 0.5))
        bonus = 0.0

        hit_state = str(hit.get("state", "")).lower().strip()
        if state and hit_state:
            state_norm = state.lower().strip()
            if hit_state == state_norm:
                bonus += 0.15          # exact state match
            elif hit_state == "national":
                bonus += 0.05          # national always a decent fallback
            else:
                bonus -= 0.05          # wrong state

        hit_bt = str(hit.get("building_type", "")).lower().strip()
        if building_type and hit_bt:
            bt_norm = building_type.lower().strip()
            if hit_bt == bt_norm:
                bonus += 0.15
            elif hit_bt:
                bonus -= 0.05

        return round(min(1.0, max(0.0, base + bonus)), 4)

    # ── R4: Structured context assembly ───────────────────────────────────────

    #: Cross-trade dependencies — querying a primary trade also pulls context
    #: for related trades (e.g., structural scope → also get waterproofing norms).
    _CROSS_TRADE_DEPS: dict = {
        "structural":      ["waterproofing", "civil"],
        "mep":             ["structural"],
        "architectural":   ["civil"],
        "waterproofing":   ["structural", "civil"],
        "civil":           ["structural"],
        "finishes":        ["architectural"],
        "external":        ["civil", "structural"],
    }

    def assemble_context(
        self,
        queries_by_trade: dict,
        n_per_trade: int = 5,
        embedder=None,
        include_cross_deps: bool = True,
        max_chars: int = 8000,
    ) -> dict:
        """
        Build structured domain context frames per trade (R4).

        Parameters
        ----------
        queries_by_trade : dict
            {trade: query_text} — one query string per trade of interest.
        n_per_trade : int
            Max domain KB hits per source type (rates / benchmarks / taxonomy).
        embedder : optional
            Embedder instance; auto-created if None.
        include_cross_deps : bool
            If True, automatically adds dependent trades (e.g. structural → waterproofing).
        max_chars : int
            Soft cap on total assembled context size.

        Returns
        -------
        dict
            {
              trade: {
                "rates":      [hit_dict, ...],   # DSR rate hits
                "benchmarks": [hit_dict, ...],   # quantity benchmark hits
                "taxonomy":   [hit_dict, ...],   # taxonomy hits
              }
            }
        """
        # Expand with cross-trade dependencies
        expanded: dict = dict(queries_by_trade)
        if include_cross_deps:
            for trade, query in list(queries_by_trade.items()):
                for dep in self._CROSS_TRADE_DEPS.get(trade, []):
                    if dep not in expanded:
                        expanded[dep] = query   # reuse originating query

        frames: dict = {}
        seen_texts: set = set()
        char_budget = max_chars

        for trade, query in expanded.items():
            if char_budget <= 0:
                break

            frame: dict = {"rates": [], "benchmarks": [], "taxonomy": []}

            # DSR rates for this trade
            for h in self.search(query, embedder=embedder, n_results=n_per_trade,
                                  source_filter="dsr_rates", trade_filter=trade):
                if char_budget <= 0:
                    break
                if h["text"] not in seen_texts:
                    seen_texts.add(h["text"])
                    frame["rates"].append(h)
                    char_budget -= len(h["text"])

            # Quantity benchmarks (not trade-scoped — benchmarks are project-type scoped)
            for h in self.search(query, embedder=embedder, n_results=min(3, n_per_trade),
                                  source_filter="benchmarks"):
                if char_budget <= 0:
                    break
                if h["text"] not in seen_texts:
                    seen_texts.add(h["text"])
                    frame["benchmarks"].append(h)
                    char_budget -= len(h["text"])

            # Taxonomy for this trade
            for h in self.search(query, embedder=embedder, n_results=n_per_trade,
                                  source_filter="taxonomy", trade_filter=trade):
                if char_budget <= 0:
                    break
                if h["text"] not in seen_texts:
                    seen_texts.add(h["text"])
                    frame["taxonomy"].append(h)
                    char_budget -= len(h["text"])

            if frame["rates"] or frame["benchmarks"] or frame["taxonomy"]:
                frames[trade] = frame

        return frames

    @staticmethod
    def format_context_for_llm(
        context_frames: dict,
        max_chars: int = 6000,
    ) -> str:
        """
        Format assembled context frames into an LLM-ready string (R4).

        Each trade section has a header and bullet lines for rates / benchmarks / taxonomy.
        Truncated to max_chars to stay within LLM context budget.
        """
        parts: list = []
        char_count = 0

        for trade, frame in context_frames.items():
            if char_count >= max_chars:
                break

            section: list = []
            header = f"=== DOMAIN KNOWLEDGE: {trade.upper()} ==="

            for h in (frame.get("rates") or [])[:3]:
                line = f"[Rate] {h['text'][:200]}"
                section.append(line)
                char_count += len(line)
                if char_count >= max_chars:
                    break

            for h in (frame.get("benchmarks") or [])[:2]:
                line = f"[Benchmark] {h['text'][:200]}"
                section.append(line)
                char_count += len(line)
                if char_count >= max_chars:
                    break

            for h in (frame.get("taxonomy") or [])[:3]:
                line = f"[Taxonomy] {h['text'][:150]}"
                section.append(line)
                char_count += len(line)
                if char_count >= max_chars:
                    break

            if section:
                parts.append(header)
                parts.extend(section)

        return "\n".join(parts)
