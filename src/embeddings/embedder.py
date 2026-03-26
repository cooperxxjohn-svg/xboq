"""
src/embeddings/embedder.py

Dense text embedding for xBOQ bid intelligence.

Backend priority:
  1. sentence-transformers (all-MiniLM-L6-v2) — offline, fast, 384-dim
  2. OpenAI text-embedding-3-small — requires OPENAI_API_KEY env var
  3. Raises ImportError if neither is available

Falls back gracefully: if chromadb / sentence-transformers are not installed the
rest of the pipeline still works (intelligence layer is skipped via try/except).
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ST_MODEL_NAME = "all-MiniLM-L6-v2"       # 384-dim, ~22MB, works offline
_OAI_MODEL_NAME = "text-embedding-3-small" # 1536-dim, needs API key
_DIM_ST = 384
_DIM_OAI = 1536


class Embedder:
    """
    Embed text to dense float32 vectors.

    Parameters
    ----------
    backend : str
        "auto"  — try sentence-transformers, fall back to OpenAI
        "st"    — sentence-transformers only (raises if not installed)
        "openai"— OpenAI only (raises if key missing)
    """

    def __init__(self, backend: str = "auto"):
        self._backend: str = ""
        self._st_model = None
        self._oai_client = None
        self.dim: int = 0

        if backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st_model = SentenceTransformer(_ST_MODEL_NAME)
                self._backend = "st"
                self.dim = _DIM_ST
                logger.info("Embedder: sentence-transformers backend loaded (%s)", _ST_MODEL_NAME)
            except ImportError:
                if backend == "st":
                    raise
                logger.debug("sentence-transformers not available, trying OpenAI")

        if not self._backend and backend in ("auto", "openai"):
            try:
                import openai as _openai  # type: ignore
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY not set")
                self._oai_client = _openai.OpenAI(api_key=api_key)
                self._backend = "openai"
                self.dim = _DIM_OAI
                logger.info("Embedder: OpenAI backend loaded (%s)", _OAI_MODEL_NAME)
            except Exception as e:
                if backend == "openai":
                    raise
                logger.debug("OpenAI embedding unavailable: %s", e)

        if not self._backend:
            raise ImportError(
                "No embedding backend available. Install sentence-transformers:\n"
                "  pip install sentence-transformers\n"
                "or set OPENAI_API_KEY for the OpenAI fallback."
            )

    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns
        -------
        np.ndarray of shape (N, dim), dtype float32
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Replace empty strings with a single space (encoders choke on empty)
        cleaned = [t.strip() or " " for t in texts]

        if self._backend == "st":
            vecs = self._st_model.encode(
                cleaned,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2 normalised → dot = cosine
            )
            return vecs.astype(np.float32)

        if self._backend == "openai":
            response = self._oai_client.embeddings.create(
                model=_OAI_MODEL_NAME,
                input=cleaned,
            )
            vecs = np.array(
                [item.embedding for item in response.data],
                dtype=np.float32,
            )
            # Normalise to unit vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return vecs / norms

        raise RuntimeError(f"Unknown backend: {self._backend}")

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string → 1-D array of shape (dim,)."""
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit vectors."""
        return float(np.dot(a, b))

    def batch_similarity(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        Cosine similarities of query (dim,) against corpus (N, dim).
        Returns 1-D array of shape (N,).
        """
        return corpus @ query

    def __repr__(self) -> str:
        return f"Embedder(backend={self._backend!r}, dim={self.dim})"
