"""
Addendum / Corrigendum Tracker — compares two tender payload runs and
surfaces added, deleted, and changed BOQ items, RFIs, and quantities.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import hashlib


@dataclass
class ItemDiff:
    item_id: str
    description: str
    trade: str
    change_type: str        # "added" | "deleted" | "qty_changed" | "rate_changed" | "unchanged"
    old_quantity: Optional[float] = None
    new_quantity: Optional[float] = None
    old_rate: Optional[float] = None
    new_rate: Optional[float] = None
    old_total: Optional[float] = None
    new_total: Optional[float] = None
    pct_qty_change: Optional[float] = None
    pct_rate_change: Optional[float] = None


@dataclass
class AddendumResult:
    base_run_id: str
    new_run_id: str
    added_items: List[ItemDiff] = field(default_factory=list)
    deleted_items: List[ItemDiff] = field(default_factory=list)
    changed_items: List[ItemDiff] = field(default_factory=list)
    unchanged_items: List[ItemDiff] = field(default_factory=list)
    new_rfis: List[dict] = field(default_factory=list)
    resolved_rfis: List[dict] = field(default_factory=list)
    cost_delta_inr: float = 0.0
    cost_delta_pct: float = 0.0
    summary: str = ""

    @property
    def total_changes(self) -> int:
        return len(self.added_items) + len(self.deleted_items) + len(self.changed_items)

    @property
    def has_changes(self) -> bool:
        return self.total_changes > 0 or bool(self.new_rfis) or bool(self.resolved_rfis)


def _item_key(item: dict) -> str:
    """Stable key for matching items across runs."""
    desc = str(item.get("description", "")).strip().lower()[:80]
    trade = str(item.get("trade", "")).strip().lower()
    unit = str(item.get("unit", "")).strip().lower()
    return hashlib.md5(f"{trade}|{desc}|{unit}".encode()).hexdigest()[:12]


def _safe_float(val) -> float:
    try:
        return float(val or 0)
    except (TypeError, ValueError):
        return 0.0


def compare_payloads(base_payload: dict, new_payload: dict,
                     base_run_id: str = "base", new_run_id: str = "new") -> AddendumResult:
    """
    Compare two payload dicts and return an AddendumResult with all changes.
    Safe to call with empty/partial payloads.
    """
    result = AddendumResult(base_run_id=base_run_id, new_run_id=new_run_id)

    base_items = base_payload.get("boq_items", [])
    new_items = new_payload.get("boq_items", [])

    # Build keyed dicts
    base_keyed: Dict[str, dict] = {_item_key(i): i for i in base_items}
    new_keyed: Dict[str, dict] = {_item_key(i): i for i in new_items}

    base_cost = sum(_safe_float(i.get("total_inr", i.get("total"))) for i in base_items)
    new_cost  = sum(_safe_float(i.get("total_inr", i.get("total"))) for i in new_items)

    # Added items
    for key, item in new_keyed.items():
        if key not in base_keyed:
            result.added_items.append(ItemDiff(
                item_id=key,
                description=item.get("description", "")[:120],
                trade=item.get("trade", ""),
                change_type="added",
                new_quantity=_safe_float(item.get("quantity")),
                new_rate=_safe_float(item.get("rate_inr", item.get("rate"))),
                new_total=_safe_float(item.get("total_inr", item.get("total"))),
            ))

    # Deleted / changed / unchanged
    for key, base_item in base_keyed.items():
        if key not in new_keyed:
            result.deleted_items.append(ItemDiff(
                item_id=key,
                description=base_item.get("description", "")[:120],
                trade=base_item.get("trade", ""),
                change_type="deleted",
                old_quantity=_safe_float(base_item.get("quantity")),
                old_rate=_safe_float(base_item.get("rate_inr", base_item.get("rate"))),
                old_total=_safe_float(base_item.get("total_inr", base_item.get("total"))),
            ))
        else:
            new_item = new_keyed[key]
            old_qty  = _safe_float(base_item.get("quantity"))
            new_qty  = _safe_float(new_item.get("quantity"))
            old_rate = _safe_float(base_item.get("rate_inr", base_item.get("rate")))
            new_rate = _safe_float(new_item.get("rate_inr", new_item.get("rate")))
            old_tot  = _safe_float(base_item.get("total_inr", base_item.get("total")))
            new_tot  = _safe_float(new_item.get("total_inr", new_item.get("total")))

            qty_changed  = abs(old_qty - new_qty) > 0.001 * max(old_qty, 1)
            rate_changed = abs(old_rate - new_rate) > 0.001 * max(old_rate, 1)

            if qty_changed or rate_changed:
                change_type = "qty_changed" if qty_changed else "rate_changed"
                diff = ItemDiff(
                    item_id=key,
                    description=base_item.get("description", "")[:120],
                    trade=base_item.get("trade", ""),
                    change_type=change_type,
                    old_quantity=old_qty, new_quantity=new_qty,
                    old_rate=old_rate,    new_rate=new_rate,
                    old_total=old_tot,    new_total=new_tot,
                )
                if qty_changed and old_qty > 0:
                    diff.pct_qty_change = round((new_qty - old_qty) / old_qty * 100, 1)
                if rate_changed and old_rate > 0:
                    diff.pct_rate_change = round((new_rate - old_rate) / old_rate * 100, 1)
                result.changed_items.append(diff)
            else:
                result.unchanged_items.append(ItemDiff(
                    item_id=key,
                    description=base_item.get("description", "")[:120],
                    trade=base_item.get("trade", ""),
                    change_type="unchanged",
                    old_quantity=old_qty, new_quantity=new_qty,
                    old_rate=old_rate,    new_rate=new_rate,
                    old_total=old_tot,    new_total=new_tot,
                ))

    # RFI diff
    base_rfi_qs = {str(r.get("question", r.get("rfi_text", "")))[:80].lower() for r in base_payload.get("rfis", [])}
    for rfi in new_payload.get("rfis", []):
        q = str(rfi.get("question", rfi.get("rfi_text", "")))[:80].lower()
        if q not in base_rfi_qs:
            result.new_rfis.append(rfi)

    new_rfi_qs = {str(r.get("question", r.get("rfi_text", "")))[:80].lower() for r in new_payload.get("rfis", [])}
    for rfi in base_payload.get("rfis", []):
        q = str(rfi.get("question", rfi.get("rfi_text", "")))[:80].lower()
        if q not in new_rfi_qs:
            result.resolved_rfis.append(rfi)

    # Cost delta
    result.cost_delta_inr = new_cost - base_cost
    result.cost_delta_pct = round((new_cost - base_cost) / max(base_cost, 1) * 100, 1) if base_cost else 0.0

    # Summary
    parts = []
    if result.added_items:
        parts.append(f"{len(result.added_items)} items added")
    if result.deleted_items:
        parts.append(f"{len(result.deleted_items)} items deleted")
    if result.changed_items:
        parts.append(f"{len(result.changed_items)} items changed")
    if result.new_rfis:
        parts.append(f"{len(result.new_rfis)} new RFIs")
    if result.resolved_rfis:
        parts.append(f"{len(result.resolved_rfis)} RFIs resolved")
    if result.cost_delta_inr:
        sign = "+" if result.cost_delta_inr > 0 else ""
        parts.append(f"cost {sign}\u20b9{result.cost_delta_inr:,.0f} ({sign}{result.cost_delta_pct}%)")
    result.summary = "; ".join(parts) if parts else "No changes detected"

    return result


def save_addendum_result(result: AddendumResult, output_dir: str = None) -> Path:
    """Persist result to ~/.xboq/addendums/{new_run_id}.json."""
    import dataclasses
    if output_dir is None:
        output_dir = Path.home() / ".xboq" / "addendums"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"{result.new_run_id}.json"
    fpath.write_text(json.dumps(dataclasses.asdict(result), indent=2, default=str))
    return fpath
