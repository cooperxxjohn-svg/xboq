"""
Page-level LLM result cache — avoids re-extracting identical page content.

Cache key: SHA-256 of (page_text + doc_type + extractor_version).
Cache location: ~/.xboq/page_cache/ (configurable via XBOQ_CACHE_DIR env var).
Cache format: one JSON file per page, named by hash.
TTL: 30 days (configurable).
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Any

_DEFAULT_CACHE_DIR = Path.home() / ".xboq" / "page_cache"
_EXTRACTOR_VERSION = "v1"  # bump when extractor logic changes
_TTL_SECONDS = 30 * 24 * 3600  # 30 days


class PageCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self._dir = Path(os.environ.get("XBOQ_CACHE_DIR", str(cache_dir or _DEFAULT_CACHE_DIR)))
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, page_text: str, doc_type: str) -> str:
        h = hashlib.sha256(
            f"{_EXTRACTOR_VERSION}|{doc_type}|{page_text}".encode()
        ).hexdigest()
        return h

    def get(self, page_text: str, doc_type: str) -> Optional[dict]:
        """Return cached extraction result or None if miss/expired."""
        key = self._key(page_text, doc_type)
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                entry = json.load(f)
            if time.time() - entry.get("cached_at", 0) > _TTL_SECONDS:
                path.unlink(missing_ok=True)
                return None
            return entry.get("result")
        except Exception:
            return None

    def put(self, page_text: str, doc_type: str, result: dict) -> None:
        """Store extraction result in cache."""
        key = self._key(page_text, doc_type)
        path = self._dir / f"{key}.json"
        try:
            with open(path, "w") as f:
                json.dump({"cached_at": time.time(), "result": result}, f)
        except Exception:
            pass

    def stats(self) -> dict:
        """Return cache statistics."""
        files = list(self._dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {"entries": len(files), "size_mb": round(total_size / 1024 / 1024, 2)}

    def clear(self, older_than_days: int = 30) -> int:
        """Remove entries older than N days. Returns count deleted."""
        cutoff = time.time() - older_than_days * 86400
        deleted = 0
        for f in self._dir.glob("*.json"):
            try:
                entry = json.loads(f.read_text())
                if entry.get("cached_at", 0) < cutoff:
                    f.unlink()
                    deleted += 1
            except Exception:
                pass
        return deleted
