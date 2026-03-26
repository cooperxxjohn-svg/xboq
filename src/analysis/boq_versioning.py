"""
src/analysis/boq_versioning.py

BOQ version snapshot storage and diff.

Each pipeline run can save a BOQ snapshot keyed by (project_id, run_id).
snapshots stored at ~/.xboq/boq_history/{project_id}/

Usage:
    from src.analysis.boq_versioning import BOQVersionStore
    store = BOQVersionStore()
    run_id = store.save_snapshot(project_id, payload)
    diff = store.diff_snapshots(project_id, run_id_a, run_id_b)
    history = store.list_runs(project_id)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = Path.home() / ".xboq" / "boq_history"

# Keys to exclude from snapshot (large/binary/noise)
_SNAPSHOT_EXCLUDE_KEYS = {
    "ocr_text_by_page", "ocr_text_cache", "page_images",
    "raw_tables", "_validation_warnings",
}


@dataclass
class BOQDiffItem:
    item_id: str
    description: str
    trade: str
    change_type: str         # "added" | "removed" | "qty_changed" | "rate_changed"
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    pct_change: Optional[float] = None


@dataclass
class BOQDiff:
    project_id: str
    run_a: str
    run_b: str
    changes: List[BOQDiffItem] = field(default_factory=list)
    total_cost_a: float = 0.0
    total_cost_b: float = 0.0

    @property
    def cost_delta(self) -> float:
        return self.total_cost_b - self.total_cost_a

    @property
    def cost_delta_pct(self) -> float:
        if self.total_cost_a == 0:
            return 0.0
        return round((self.cost_delta / self.total_cost_a) * 100, 1)

    @property
    def n_added(self) -> int:
        return sum(1 for c in self.changes if c.change_type == "added")

    @property
    def n_removed(self) -> int:
        return sum(1 for c in self.changes if c.change_type == "removed")

    @property
    def n_changed(self) -> int:
        return sum(1 for c in self.changes if c.change_type in ("qty_changed", "rate_changed"))

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "run_a": self.run_a,
            "run_b": self.run_b,
            "total_cost_a": self.total_cost_a,
            "total_cost_b": self.total_cost_b,
            "cost_delta": self.cost_delta,
            "cost_delta_pct": self.cost_delta_pct,
            "n_added": self.n_added,
            "n_removed": self.n_removed,
            "n_changed": self.n_changed,
            "changes": [
                {
                    "item_id": c.item_id,
                    "description": c.description,
                    "trade": c.trade,
                    "change_type": c.change_type,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "pct_change": c.pct_change,
                }
                for c in self.changes
            ],
        }


def _item_key(item: dict) -> str:
    """Stable key for a BOQ item based on description + trade."""
    desc = str(item.get("description") or "").lower().strip()
    trade = str(item.get("trade") or "").lower().strip()
    return hashlib.md5(f"{trade}::{desc}".encode()).hexdigest()[:12]


def _total_cost(items: list) -> float:
    total = 0.0
    for item in items:
        try:
            qty = float(item.get("qty") or item.get("quantity") or 0)
            rate = float(item.get("rate_inr") or item.get("rate") or 0)
            amount = float(item.get("total_inr") or item.get("amount") or (qty * rate))
            total += amount
        except (TypeError, ValueError):
            pass
    return round(total, 2)


class BOQVersionStore:
    """Stores and diffs BOQ snapshots across pipeline runs."""

    def __init__(self, history_dir: Optional[Path] = None):
        self.history_dir = Path(history_dir or _DEFAULT_HISTORY_DIR)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        safe_id = "".join(c for c in project_id if c.isalnum() or c in "_-")[:40]
        d = self.history_dir / safe_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_snapshot(self, project_id: str, payload: dict) -> str:
        """
        Save a BOQ snapshot for this pipeline run.

        Returns
        -------
        str
            run_id (timestamp-based)
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        snap = {
            "run_id": run_id,
            "project_id": project_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "boq_items": payload.get("boq_items") or [],
            "total_cost": _total_cost(payload.get("boq_items") or []),
            "n_items": len(payload.get("boq_items") or []),
            "n_rfis": len(payload.get("rfis") or []),
            "qa_score": payload.get("qa_score"),
            "processing_stats": payload.get("processing_stats") or {},
        }
        snap_path = self._project_dir(project_id) / f"{run_id}.json"
        snap_path.write_text(json.dumps(snap, ensure_ascii=False, default=str))
        logger.info("BOQ snapshot saved: %s / %s (%d items)", project_id, run_id, snap["n_items"])

        # Update index
        self._update_index(project_id, run_id, snap)
        return run_id

    def _update_index(self, project_id: str, run_id: str, snap: dict) -> None:
        idx_path = self._project_dir(project_id) / "index.json"
        index = []
        if idx_path.exists():
            try:
                index = json.loads(idx_path.read_text())
            except Exception:
                index = []
        index.append({
            "run_id": run_id,
            "saved_at": snap["saved_at"],
            "n_items": snap["n_items"],
            "total_cost": snap["total_cost"],
            "n_rfis": snap["n_rfis"],
            "qa_score": snap.get("qa_score"),
        })
        # Keep last 50 runs
        index = index[-50:]
        idx_path.write_text(json.dumps(index, ensure_ascii=False))

    def list_runs(self, project_id: str) -> List[dict]:
        """List all saved runs for a project (newest first)."""
        idx_path = self._project_dir(project_id) / "index.json"
        if not idx_path.exists():
            return []
        try:
            return list(reversed(json.loads(idx_path.read_text())))
        except Exception as e:
            logger.warning("BOQVersionStore: could not read index for %s: %s", project_id, e)
            return []

    def load_snapshot(self, project_id: str, run_id: str) -> Optional[dict]:
        """Load a specific snapshot."""
        snap_path = self._project_dir(project_id) / f"{run_id}.json"
        if not snap_path.exists():
            return None
        try:
            return json.loads(snap_path.read_text())
        except Exception as e:
            logger.warning("BOQVersionStore: could not load %s/%s: %s", project_id, run_id, e)
            return None

    def diff_snapshots(
        self,
        project_id: str,
        run_id_a: str,
        run_id_b: str,
    ) -> Optional[BOQDiff]:
        """
        Compute item-level diff between two BOQ snapshots.

        Parameters
        ----------
        run_id_a : str   Older run (baseline)
        run_id_b : str   Newer run (comparison)

        Returns
        -------
        BOQDiff | None
        """
        snap_a = self.load_snapshot(project_id, run_id_a)
        snap_b = self.load_snapshot(project_id, run_id_b)
        if snap_a is None or snap_b is None:
            logger.warning("BOQVersionStore.diff: snapshot(s) not found (%s, %s)", run_id_a, run_id_b)
            return None

        items_a: dict = {_item_key(i): i for i in (snap_a.get("boq_items") or [])}
        items_b: dict = {_item_key(i): i for i in (snap_b.get("boq_items") or [])}

        changes: List[BOQDiffItem] = []

        # Added items (in B but not A)
        for key, item in items_b.items():
            if key not in items_a:
                changes.append(BOQDiffItem(
                    item_id=key,
                    description=str(item.get("description", ""))[:80],
                    trade=str(item.get("trade", "")),
                    change_type="added",
                    new_value=float(item.get("total_inr") or 0),
                ))

        # Removed items (in A but not B)
        for key, item in items_a.items():
            if key not in items_b:
                changes.append(BOQDiffItem(
                    item_id=key,
                    description=str(item.get("description", ""))[:80],
                    trade=str(item.get("trade", "")),
                    change_type="removed",
                    old_value=float(item.get("total_inr") or 0),
                ))

        # Changed items (in both — check qty and rate)
        for key in set(items_a) & set(items_b):
            ia, ib = items_a[key], items_b[key]
            qty_a = float(ia.get("qty") or ia.get("quantity") or 0)
            qty_b = float(ib.get("qty") or ib.get("quantity") or 0)
            rate_a = float(ia.get("rate_inr") or ia.get("rate") or 0)
            rate_b = float(ib.get("rate_inr") or ib.get("rate") or 0)

            if abs(qty_a - qty_b) > 0.001:
                pct = round(((qty_b - qty_a) / max(qty_a, 0.001)) * 100, 1)
                changes.append(BOQDiffItem(
                    item_id=key,
                    description=str(ia.get("description", ""))[:80],
                    trade=str(ia.get("trade", "")),
                    change_type="qty_changed",
                    old_value=qty_a,
                    new_value=qty_b,
                    pct_change=pct,
                ))
            elif abs(rate_a - rate_b) > 0.01:
                pct = round(((rate_b - rate_a) / max(rate_a, 0.01)) * 100, 1)
                changes.append(BOQDiffItem(
                    item_id=key,
                    description=str(ia.get("description", ""))[:80],
                    trade=str(ia.get("trade", "")),
                    change_type="rate_changed",
                    old_value=rate_a,
                    new_value=rate_b,
                    pct_change=pct,
                ))

        return BOQDiff(
            project_id=project_id,
            run_a=run_id_a,
            run_b=run_id_b,
            changes=changes,
            total_cost_a=snap_a.get("total_cost", 0.0),
            total_cost_b=snap_b.get("total_cost", 0.0),
        )

    def latest_run_id(self, project_id: str) -> Optional[str]:
        """Return the most recent run_id for a project, or None."""
        runs = self.list_runs(project_id)
        return runs[0]["run_id"] if runs else None
