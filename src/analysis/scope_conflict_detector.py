"""
src/analysis/scope_conflict_detector.py

Detects overlapping/duplicate scope between BOQ items.

Common conflicts in Indian construction tenders:
- "PCC in foundation" + "Lean concrete in footings" (same work, different names)
- "Excavation in foundation" + "Earth excavation for footings" (same)
- "Plastering in CM 1:6" appearing in both wall and ceiling items for same room
- Same door/window mark billed twice under different schedule items
- "Anti-termite treatment" in both civil and prelims BOQ

Usage:
    from src.analysis.scope_conflict_detector import detect_scope_conflicts
    conflicts = detect_scope_conflicts(payload)
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScopeConflict:
    conflict_id: str
    conflict_type: str       # "duplicate" | "overlap" | "double_count"
    severity: str            # "high" | "medium" | "low"
    description: str
    item_a_desc: str
    item_b_desc: str
    item_a_trade: str
    item_b_trade: str
    estimated_overcount: Optional[float] = None
    suggestion: str = ""


# ── Conflict patterns ─────────────────────────────────────────────────────────

# Groups of synonymous terms that indicate the same work
_SYNONYM_GROUPS: List[List[str]] = [
    ["pcc", "plain cement concrete", "lean concrete", "blinding concrete"],
    ["anti termite", "anti-termite", "termite treatment", "termicide"],
    ["dpc", "damp proof course", "damp proofing course"],
    ["excavation", "earth work", "earthwork", "digging"],
    ["backfilling", "back filling", "filling in excavation"],
    ["plaster", "plastering", "rendering"],
    ["pointing", "repointing", "mortar pointing"],
    ["waterproofing", "water proofing", "wp treatment"],
    ["false ceiling", "false roof", "suspended ceiling", "drop ceiling"],
    ["paving", "pavement", "flooring", "floor finish"],
]

_SYNONYM_MAP: Dict[str, str] = {}
for group in _SYNONYM_GROUPS:
    canonical = group[0]
    for term in group:
        _SYNONYM_MAP[term] = canonical


def _normalize(text: str) -> str:
    """Normalize description for comparison."""
    text = text.lower().strip()
    # Remove filler words
    text = re.sub(r'\b(in|of|for|to|the|a|an|and|with|including|complete)\b', ' ', text)
    # Apply synonym normalization
    for term, canonical in _SYNONYM_MAP.items():
        text = re.sub(r'\b' + re.escape(term) + r'\b', canonical, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word tokens (ignoring single-char tokens)."""
    ta = set(w for w in a.split() if len(w) >= 2)
    tb = set(w for w in b.split() if len(w) >= 2)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def _find_duplicates(items: List[dict], similarity_threshold: float = 0.40) -> List[ScopeConflict]:
    """Find near-duplicate BOQ items (same work described differently)."""
    conflicts = []
    counter = [0]

    normalized = [(_normalize(str(i.get("description", ""))), i) for i in items]

    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            na, ia = normalized[i]
            nb, ib = normalized[j]

            sim = _token_overlap(na, nb)
            if sim < similarity_threshold:
                continue

            # Check if same unit (strong duplicate signal)
            ua = str(ia.get("unit", "")).lower().strip()
            ub = str(ib.get("unit", "")).lower().strip()
            same_unit = (ua == ub and ua != "")

            severity = "high" if (sim >= 0.70 and same_unit) else ("medium" if sim >= 0.55 else "low")

            # Estimate overcount
            try:
                qty_a = float(ia.get("qty") or ia.get("quantity") or 0)
                qty_b = float(ib.get("qty") or ib.get("quantity") or 0)
                overcount = min(qty_a, qty_b)
            except (TypeError, ValueError):
                overcount = None

            counter[0] += 1
            conflicts.append(ScopeConflict(
                conflict_id=f"CONF-{counter[0]:03d}",
                conflict_type="duplicate",
                severity=severity,
                description=f"Possible duplicate: {sim:.0%} similarity between items",
                item_a_desc=str(ia.get("description", ""))[:80],
                item_b_desc=str(ib.get("description", ""))[:80],
                item_a_trade=str(ia.get("trade", "")),
                item_b_trade=str(ib.get("trade", "")),
                estimated_overcount=overcount,
                suggestion="Verify these are not the same work billed twice. "
                           "If different trades, clarify scope boundary.",
            ))

    return conflicts


def _find_cross_trade_overlaps(items: List[dict]) -> List[ScopeConflict]:
    """Find items that appear in multiple trade sections (common double-counting source)."""
    conflicts = []
    counter = [0]

    # Group by normalized description
    from collections import defaultdict
    desc_to_items: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        key = _normalize(str(item.get("description", "")))[:60]
        if len(key) > 10:
            desc_to_items[key].append(item)

    for desc, group in desc_to_items.items():
        if len(group) < 2:
            continue
        trades = list({str(i.get("trade", "general")) for i in group})
        if len(trades) < 2:
            continue  # same trade repeated — handled by duplicate check

        counter[0] += 1
        conflicts.append(ScopeConflict(
            conflict_id=f"XCONF-{counter[0]:03d}",
            conflict_type="double_count",
            severity="high",
            description=f"Item appears in {len(trades)} different trades: {', '.join(trades)}",
            item_a_desc=str(group[0].get("description", ""))[:80],
            item_b_desc=str(group[1].get("description", ""))[:80],
            item_a_trade=str(group[0].get("trade", "")),
            item_b_trade=str(group[1].get("trade", "")),
            suggestion=f"Same work billed under {len(group)} items. "
                       "Remove duplicates or clarify trade boundary.",
        ))

    return conflicts


def detect_scope_conflicts(
    payload: dict,
    similarity_threshold: float = 0.40,
    max_conflicts: int = 50,
) -> List[ScopeConflict]:
    """
    Detect overlapping/duplicate scope between BOQ items.

    Parameters
    ----------
    payload : dict
        Full analysis payload.
    similarity_threshold : float
        Minimum token overlap to flag as conflict (default 0.60).
    max_conflicts : int
        Cap on returned conflicts.

    Returns
    -------
    List[ScopeConflict] sorted by severity (high first).
    """
    all_items = (payload.get("boq_items") or []) + (payload.get("spec_items") or [])
    if not all_items:
        return []

    conflicts: List[ScopeConflict] = []
    conflicts.extend(_find_duplicates(all_items, similarity_threshold=similarity_threshold))
    conflicts.extend(_find_cross_trade_overlaps(all_items))

    # Sort: high severity first
    _sev = {"high": 0, "medium": 1, "low": 2}
    conflicts.sort(key=lambda c: _sev.get(c.severity, 9))

    logger.info("Scope conflict detection: %d conflicts (%d high, %d medium, %d low)",
                len(conflicts),
                sum(1 for c in conflicts if c.severity == "high"),
                sum(1 for c in conflicts if c.severity == "medium"),
                sum(1 for c in conflicts if c.severity == "low"))

    return conflicts[:max_conflicts]


def conflict_summary(conflicts: List[ScopeConflict]) -> dict:
    """Return a dict summary for embedding in payload."""
    return {
        "total": len(conflicts),
        "high": sum(1 for c in conflicts if c.severity == "high"),
        "medium": sum(1 for c in conflicts if c.severity == "medium"),
        "low": sum(1 for c in conflicts if c.severity == "low"),
        "conflicts": [
            {
                "id": c.conflict_id,
                "type": c.conflict_type,
                "severity": c.severity,
                "description": c.description,
                "item_a": c.item_a_desc,
                "item_b": c.item_b_desc,
                "trade_a": c.item_a_trade,
                "trade_b": c.item_b_trade,
                "overcount": c.estimated_overcount,
                "suggestion": c.suggestion,
            }
            for c in conflicts[:30]
        ],
    }
