#!/usr/bin/env python3
"""
Run Artifacts — write standardized debug artifacts after a pipeline run.

This module provides lightweight helpers that dump structured JSON files
alongside normal pipeline output.  It does **not** alter core behaviour;
callers opt-in by calling individual ``write_*`` helpers.

Typical usage
-------------
>>> from src.ops.run_artifacts import write_run_summary, write_run_manifest
>>> write_run_manifest(payload, output_dir / "run_artifacts")
>>> write_run_summary(payload, output_dir / "run_artifacts")
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_json(obj: Any) -> Any:
    """Make *obj* JSON-serialisable (best-effort)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError, OverflowError):
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)


def _write_json(data: Any, path: Path) -> Path:
    """Serialise *data* to *path* as pretty-printed JSON."""
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_safe_json(data), fh, indent=2, ensure_ascii=False, default=str)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_run_manifest(
    payload: Dict[str, Any],
    output_dir: Union[str, Path],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``run_manifest.json`` — lightweight envelope about the run.

    Contains run timestamp, payload keys present, and optional extra metadata.
    """
    output_dir = Path(output_dir)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "payload_keys": sorted(payload.keys()) if payload else [],
        "payload_key_count": len(payload) if payload else 0,
    }
    if extra:
        manifest["extra"] = extra
    return _write_json(manifest, output_dir / "run_manifest.json")


def write_processing_stats(
    payload: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Optional[Path]:
    """Write ``processing_stats.json`` extracted from the payload."""
    stats = payload.get("processing_stats")
    if not stats:
        return None
    return _write_json(stats, Path(output_dir) / "processing_stats.json")


def write_extraction_diagnostics(
    payload: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Optional[Path]:
    """Write ``extraction_diagnostics.json`` extracted from the payload."""
    diag = payload.get("extraction_diagnostics")
    if not diag:
        return None
    return _write_json(diag, Path(output_dir) / "extraction_diagnostics.json")


def write_run_summary(
    payload: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Path:
    """Write ``run_summary.json`` — compact quality-focused summary.

    Pulls the most important KPIs into a single small file for quick review.
    """
    output_dir = Path(output_dir)
    ps = payload.get("processing_stats", {})
    ed = payload.get("extraction_diagnostics", {})

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages_total": ps.get("total_pages", 0),
        "deep_processed_pages": ps.get("deep_processed_pages", 0),
        "ocr_pages": ps.get("ocr_pages", 0),
        "selection_mode": ps.get("selection_mode", "unknown"),
        "boq_items": len(payload.get("boq_items", [])),
        "rfis": len(payload.get("rfis", [])),
        "blockers": len(payload.get("blockers", [])),
        "commercial_terms": len(payload.get("commercial_terms", [])) if isinstance(payload.get("commercial_terms"), list) else (1 if payload.get("commercial_terms") else 0),
        "qa_score": payload.get("qa_score"),
        "boq_pages_attempted": ed.get("boq_pages_attempted", 0),
        "boq_pages_parsed": ed.get("boq_pages_parsed", 0),
        "schedule_pages_attempted": ed.get("schedule_pages_attempted", 0),
        "schedule_pages_parsed": ed.get("schedule_pages_parsed", 0),
    }
    return _write_json(summary, output_dir / "run_summary.json")


def write_error_report(
    error: Union[Exception, str],
    output_dir: Union[str, Path],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append a single error entry to ``errors.jsonl``.

    Each call appends one JSON line (JSONL format) so multiple errors
    accumulate across retries.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(error, Exception):
        entry["error_type"] = type(error).__name__
        entry["error_message"] = str(error)
        entry["traceback"] = traceback.format_exception(type(error), error, error.__traceback__)
    else:
        entry["error_type"] = "str"
        entry["error_message"] = str(error)

    if context:
        entry["context"] = context

    errors_path = output_dir / "errors.jsonl"
    with open(errors_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    return errors_path


def write_all_artifacts(
    payload: Dict[str, Any],
    output_dir: Union[str, Path],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Convenience: write all standard artifacts and return list of paths."""
    output_dir = Path(output_dir)
    written: List[Path] = []
    written.append(write_run_manifest(payload, output_dir, extra=extra))
    p = write_processing_stats(payload, output_dir)
    if p:
        written.append(p)
    p = write_extraction_diagnostics(payload, output_dir)
    if p:
        written.append(p)
    written.append(write_run_summary(payload, output_dir))
    return written
