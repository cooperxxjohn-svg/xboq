"""
Pipeline Cache — persistent cross-run caching for analysis pipeline stages.

Cache key = sha256(pdf_bytes) + config hash.
Each stage saves a JSON sidecar to .xboq_cache/{key_prefix}/.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CACHE KEY COMPUTATION
# =============================================================================

def compute_cache_key(pdf_paths: List[Path], config: dict) -> str:
    """
    Compute a deterministic cache key from PDF file contents + config.

    Args:
        pdf_paths: List of PDF file paths (order matters).
        config: Dict of pipeline config values (dpi, budget, etc.).

    Returns:
        Hex string (SHA-256) uniquely identifying this run configuration.
    """
    hasher = hashlib.sha256()

    # Hash PDF file contents (bytes)
    for pdf_path in sorted(pdf_paths, key=lambda p: str(p)):
        p = Path(pdf_path)
        if p.exists():
            # Hash file size + first 64KB + last 64KB for speed on large files
            file_size = p.stat().st_size
            hasher.update(f"file:{p.name}:{file_size}".encode())
            with open(p, "rb") as f:
                head = f.read(65536)
                hasher.update(head)
                if file_size > 131072:
                    f.seek(-65536, 2)
                    tail = f.read(65536)
                    hasher.update(tail)
        else:
            hasher.update(f"missing:{p.name}".encode())

    # Hash config values
    config_str = json.dumps(config, sort_keys=True, default=str)
    hasher.update(config_str.encode())

    return hasher.hexdigest()


# =============================================================================
# CACHE DIRECTORY MANAGEMENT
# =============================================================================

def get_cache_dir(output_dir: Path, cache_key: str) -> Path:
    """
    Get or create the cache directory for a given cache key.

    Structure: {output_dir}/.xboq_cache/{key_prefix}/

    Args:
        output_dir: Base output directory for the project.
        cache_key: Full SHA-256 hex string.

    Returns:
        Path to the cache directory (created if needed).
    """
    prefix = cache_key[:16]
    cache_path = Path(output_dir) / ".xboq_cache" / prefix
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


# =============================================================================
# STAGE LOAD / SAVE
# =============================================================================

def load_cached_stage(cache_dir: Path, stage: str) -> Optional[dict]:
    """
    Load a cached stage result from disk.

    Args:
        cache_dir: Path to cache directory.
        stage: Stage name (e.g., 'page_index', 'ocr_text', 'extraction').

    Returns:
        Dict with cached data, or None if cache miss.
    """
    stage_file = Path(cache_dir) / f"{stage}.json"
    if not stage_file.exists():
        return None

    try:
        with open(stage_file, "r") as f:
            data = json.load(f)
        # Validate it has content
        if isinstance(data, dict) and data:
            return data
    except (json.JSONDecodeError, IOError, OSError):
        pass

    return None


def save_cached_stage(cache_dir: Path, stage: str, data: dict) -> Path:
    """
    Save a stage result to the cache directory.

    Args:
        cache_dir: Path to cache directory.
        stage: Stage name.
        data: Dict to serialize as JSON.

    Returns:
        Path to the saved file.
    """
    stage_file = Path(cache_dir) / f"{stage}.json"
    with open(stage_file, "w") as f:
        json.dump(data, f, default=str)
    return stage_file


# =============================================================================
# CACHE SIZE MEASUREMENT
# =============================================================================

def _measure_cache_bytes(cache_dir: Path) -> int:
    """Compute total bytes used by cache directory."""
    total = 0
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for f in cache_path.iterdir():
            if f.is_file():
                total += f.stat().st_size
    return total


# =============================================================================
# CACHE STATS
# =============================================================================

def build_cache_stats(
    hits: List[str],
    misses: List[str],
    time_saved_s: float,
    cache_bytes: int,
) -> dict:
    """
    Build a cache stats summary for the payload.

    Args:
        hits: List of stage names that were cache hits.
        misses: List of stage names that were cache misses.
        time_saved_s: Estimated seconds saved by cache hits.
        cache_bytes: Total bytes of cached data on disk.

    Returns:
        Dict with cache stats: hits, misses, hit_rate, time_saved_s, cache_bytes.
    """
    total = len(hits) + len(misses)
    hit_rate = len(hits) / total if total > 0 else 0.0

    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hit_rate, 2),
        "time_saved_s": round(time_saved_s, 2),
        "cache_bytes": cache_bytes,
    }
