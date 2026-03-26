"""
KnowledgeBase interface for xBOQ.ai RAG layer.

Provides:
  - ``KnowledgeBase`` — Protocol that all KB implementations satisfy.
  - ``get_kb()`` — Lazy singleton that returns the default DomainKBStore.
  - ``reset_kb()`` — Test helper to clear the singleton (e.g. between test runs).

Usage (all call sites should prefer this over direct instantiation):

    from src.embeddings.kb_interface import get_kb

    kb = get_kb()
    kb.build_if_stale()
    results = kb.search("M25 concrete rate Delhi", n_results=10)
    context = kb.format_context_for_llm(
        kb.assemble_context(["civil"], query="RCC column estimate"),
        max_chars=2000,
    )
    results_by_trade = kb.search_by_trade("waterproofing", "waterproofing", n_results=8)

The singleton is initialised on the first ``get_kb()`` call and reused for the
lifetime of the process.  It is never rebuilt mid-run unless ``build_if_stale()``
detects source changes.

In tests, inject a mock via ``reset_kb(mock_kb_instance)`` before the test and
call ``reset_kb()`` (no arg) in teardown to restore the default.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol — the stable interface any KB implementation must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class KnowledgeBase(Protocol):
    """
    Stable contract for domain knowledge bases used in RAG contexts.

    All methods must degrade gracefully (return [] / empty string / False)
    when the underlying store is unavailable — never raise at call-site.
    """

    def build_if_stale(self, embedder=None) -> bool:
        """Build or rebuild the index if the source data has changed.

        Returns True if a build was triggered, False if already up to date.
        """
        ...

    def search(
        self,
        query: str,
        embedder=None,
        n_results: int = 10,
        trade_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text / semantic search.  Returns list of result dicts with
        at minimum ``text`` and ``score`` keys."""
        ...

    def search_by_trade(
        self,
        query: str,
        trade: str,
        embedder=None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search filtered to a specific trade (e.g. ``'civil'``, ``'mep'``)."""
        ...

    def assemble_context(
        self,
        trades: List[str],
        query: str = "",
        n_per_trade: int = 5,
        embedder=None,
    ) -> List[Dict[str, Any]]:
        """Retrieve structured context frames across multiple trades."""
        ...

    def format_context_for_llm(
        self,
        context_frames: List[Dict[str, Any]],
        max_chars: int = 3000,
    ) -> str:
        """Format retrieved context into a compact string for LLM prompts."""
        ...

    def count(self) -> int:
        """Return total number of indexed documents."""
        ...

    def rate_confidence(
        self,
        item_description: str,
        trade: str,
        n_results: int = 5,
        embedder=None,
    ) -> Dict[str, Any]:
        """Return rate confidence dict for an item against the KB."""
        ...


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_kb_singleton: Optional[Any] = None


def get_kb(persist_dir=None) -> "KnowledgeBase":  # type: ignore[return]
    """Return the process-wide DomainKBStore singleton.

    Thread-safe at the Python GIL level; safe for multi-threaded FastAPI use.
    The first call may be slow (index initialisation check); subsequent calls
    are instant (cached object reference).

    Args:
        persist_dir: Override the default ``~/.xboq/domain_kb`` path.
                     Only honoured on the *first* call; ignored thereafter.
    """
    global _kb_singleton
    if _kb_singleton is None:
        from src.embeddings.domain_kb_store import DomainKBStore
        _kb_singleton = DomainKBStore(persist_dir=persist_dir)
    return _kb_singleton  # type: ignore[return-value]


def reset_kb(replacement=None) -> None:
    """Replace (or clear) the singleton.

    In tests::

        from src.embeddings.kb_interface import reset_kb

        class _StubKB:
            def build_if_stale(self, embedder=None): return False
            def search(self, query, embedder=None, n_results=10, trade_filter=None): return []
            def search_by_trade(self, query, trade, embedder=None, n_results=10): return []
            def assemble_context(self, trades, query="", n_per_trade=5, embedder=None): return []
            def format_context_for_llm(self, frames, max_chars=3000): return ""
            def count(self): return 0
            def rate_confidence(self, desc, trade, n_results=5, embedder=None): return {}

        reset_kb(_StubKB())
        # ... run tests ...
        reset_kb()   # restore default

    Args:
        replacement: A ``KnowledgeBase``-compatible object, or ``None`` to
                     clear the singleton so it will be recreated on next
                     ``get_kb()`` call.
    """
    global _kb_singleton
    _kb_singleton = replacement
