"""
Local project history — stores pipeline results in ~/.xboq/history/ as JSON files.
Used as fallback when Supabase is not configured, or alongside it.
"""
from __future__ import annotations
import json, time, hashlib, os
from pathlib import Path
from typing import List, Dict, Optional

_HISTORY_DIR = Path.home() / ".xboq" / "history"
_MAX_ENTRIES = 50  # keep last 50 analyses


def _dir() -> Path:
    d = Path(os.environ.get("XBOQ_HISTORY_DIR", str(_HISTORY_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(filename: str, summary: dict, payload: dict) -> str:
    """Save a pipeline run. Returns the run_id."""
    run_id = hashlib.sha256(f"{filename}{time.time()}".encode()).hexdigest()[:12]
    entry = {
        "run_id":     run_id,
        "filename":   filename,
        "saved_at":   time.time(),
        "summary":    summary,
        # Slim payload — strip large text blobs
        "payload":    {k: v for k, v in payload.items()
                       if k not in ("raw_text", "all_page_texts", "page_images")},
    }
    path = _dir() / f"{run_id}.json"
    path.write_text(json.dumps(entry, default=str))
    _prune()
    return run_id


def list_runs(limit: int = 20) -> List[Dict]:
    """Return recent runs, newest first."""
    files = sorted(_dir().glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    runs = []
    for f in files[:limit]:
        try:
            e = json.loads(f.read_text())
            runs.append({
                "run_id":   e["run_id"],
                "filename": e.get("filename", "unknown"),
                "saved_at": e.get("saved_at", 0),
                "summary":  e.get("summary", {}),
            })
        except Exception:
            pass
    return runs


def get_run(run_id: str) -> Optional[Dict]:
    """Load a full run by ID."""
    path = _dir() / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def delete_run(run_id: str) -> bool:
    path = _dir() / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def _prune():
    """Keep only _MAX_ENTRIES most recent runs."""
    files = sorted(_dir().glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[_MAX_ENTRIES:]:
        try:
            f.unlink()
        except Exception:
            pass
