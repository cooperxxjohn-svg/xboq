"""
RFI & Assumption Clustering — group by page proximity, scope, text similarity.

Reduces noise by merging similar RFIs and assumptions into clusters
with references to originals.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import csv
import io
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# TEXT SIMILARITY
# =============================================================================

def _normalize_text(text: str) -> str:
    """Normalize text for comparison — lowercase, collapse whitespace, strip."""
    if not text:
        return ""
    # Try to use normalize_requirement_text if available
    try:
        from .normalize import normalize_requirement_text
        return normalize_requirement_text(text)
    except (ImportError, ValueError):
        # Fallback: basic normalization
        return " ".join(text.lower().split()).strip()


def _text_similarity(a: str, b: str) -> float:
    """Compute text similarity ratio using SequenceMatcher (0.0–1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _pages_overlap(pages_a: List[int], pages_b: List[int]) -> bool:
    """Check if two page lists have any overlap."""
    if not pages_a or not pages_b:
        return False
    return bool(set(pages_a) & set(pages_b))


# =============================================================================
# RFI CLUSTERING
# =============================================================================

def cluster_rfis(
    rfis: List[dict],
    multi_doc_index: Optional[dict] = None,
    similarity_threshold: float = 0.65,
) -> List[dict]:
    """
    Cluster RFIs by trade + page overlap + text similarity.

    Two RFIs are grouped if:
      1) Same trade AND overlapping evidence pages, OR
      2) Normalized question text similarity > threshold

    Args:
        rfis: List of RFI dicts (from payload["rfis"]).
        multi_doc_index: Optional multi-doc index dict for doc_id enrichment.
        similarity_threshold: Minimum SequenceMatcher ratio to merge (0.0–1.0).

    Returns:
        List of cluster dicts:
        [{
            cluster_id: str,    # "RFIC-001"
            label: str,         # Summary label for the cluster
            rfis: [dict],       # Original RFI dicts in this cluster
            merged_question: str,  # Best representative question
            trade: str,
            priority: str,      # Highest priority in cluster
            evidence_pages: [int],  # Union of all evidence pages
            doc_ids: [int],     # Unique doc_ids (when multi-doc)
            count: int,
        }]
    """
    if not rfis:
        return []

    # Normalize all questions for comparison
    normalized = []
    for rfi in rfis:
        q = rfi.get("question", "") or ""
        normalized.append(_normalize_text(q))

    # Build adjacency: which RFIs should be in the same cluster
    n = len(rfis)
    parent = list(range(n))  # Union-Find

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = rfis[i], rfis[j]
            trade_i = ri.get("trade", "")
            trade_j = rj.get("trade", "")

            # Criterion 1: Same trade + overlapping pages
            if trade_i and trade_i == trade_j:
                pages_i = _get_evidence_pages(ri)
                pages_j = _get_evidence_pages(rj)
                if _pages_overlap(pages_i, pages_j):
                    union(i, j)
                    continue

            # Criterion 2: Text similarity
            if normalized[i] and normalized[j]:
                sim = _text_similarity(normalized[i], normalized[j])
                if sim >= similarity_threshold:
                    union(i, j)

    # Build clusters from union-find
    clusters_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(i)

    # Priority ordering for selecting highest
    _PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "fyi": 4}

    clusters = []
    for cluster_idx, (root, members) in enumerate(sorted(clusters_map.items())):
        member_rfis = [rfis[i] for i in members]

        # Merged fields
        all_pages: Set[int] = set()
        all_doc_ids: Set[int] = set()
        best_priority = "low"
        trade = member_rfis[0].get("trade", "general")

        for rfi in member_rfis:
            pages = _get_evidence_pages(rfi)
            all_pages.update(pages)

            # Track doc_ids from multi-doc
            if multi_doc_index and multi_doc_index.get("docs"):
                docs = multi_doc_index["docs"]
                if len(docs) > 1:
                    for p in pages:
                        for d in reversed(docs):
                            if p >= d.get("global_page_start", 0):
                                all_doc_ids.add(d.get("doc_id", -1))
                                break

            rfi_priority = rfi.get("priority", "low")
            if _PRIORITY_ORDER.get(rfi_priority, 4) < _PRIORITY_ORDER.get(best_priority, 4):
                best_priority = rfi_priority

        # Use longest question as representative
        merged_question = max(
            (rfi.get("question", "") for rfi in member_rfis),
            key=len,
            default="",
        )

        # Label: trade + count
        label = f"{trade.title()} ({len(member_rfis)} RFI{'s' if len(member_rfis) > 1 else ''})"

        clusters.append({
            "cluster_id": f"RFIC-{cluster_idx + 1:03d}",
            "label": label,
            "rfis": member_rfis,
            "merged_question": merged_question,
            "trade": trade,
            "priority": best_priority,
            "evidence_pages": sorted(all_pages),
            "doc_ids": sorted(all_doc_ids),
            "count": len(member_rfis),
        })

    # Sprint 17: Stable sort by priority/trade + re-number cluster_ids
    from src.analysis.determinism import stable_sort_key
    _PRI = {"critical": 0, "high": 1, "medium": 2, "low": 3, "fyi": 4}
    clusters.sort(key=lambda c: (
        _PRI.get(c.get("priority", "low"), 4),
        c.get("trade", ""),
        stable_sort_key(c, ("merged_question",)),
    ))
    for _ci, _c in enumerate(clusters):
        _c["cluster_id"] = f"RFIC-{_ci + 1:03d}"

    return clusters


def _get_evidence_pages(rfi: dict) -> List[int]:
    """Extract evidence page indices from an RFI dict."""
    evidence = rfi.get("evidence", {})
    if isinstance(evidence, dict):
        pages = evidence.get("pages", []) or evidence.get("page_numbers", [])
        return [p for p in pages if isinstance(p, int)]
    return []


# =============================================================================
# ASSUMPTION CLUSTERING
# =============================================================================

def cluster_assumptions(
    assumptions: List[dict],
    similarity_threshold: float = 0.65,
) -> List[dict]:
    """
    Cluster assumptions by scope_tag + text similarity.

    Two assumptions are grouped if:
      1) Same non-empty scope_tag, OR
      2) Normalized text similarity > threshold

    Args:
        assumptions: List of assumption dicts.
        similarity_threshold: Minimum SequenceMatcher ratio to merge.

    Returns:
        List of cluster dicts:
        [{
            cluster_id: str,    # "ASMPC-001"
            label: str,
            assumptions: [dict],  # Original assumption dicts
            merged_text: str,
            scope_tag: str,
            risk_level: str,    # Highest risk level
            count: int,
        }]
    """
    if not assumptions:
        return []

    # Normalize texts
    normalized = []
    for a in assumptions:
        text = a.get("text", "") or a.get("title", "") or ""
        normalized.append(_normalize_text(text))

    # Union-Find
    n = len(assumptions)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = assumptions[i], assumptions[j]

            # Criterion 1: Same non-empty scope_tag
            tag_i = ai.get("scope_tag", "").strip()
            tag_j = aj.get("scope_tag", "").strip()
            if tag_i and tag_i == tag_j:
                union(i, j)
                continue

            # Criterion 2: Text similarity
            if normalized[i] and normalized[j]:
                sim = _text_similarity(normalized[i], normalized[j])
                if sim >= similarity_threshold:
                    union(i, j)

    # Build clusters
    clusters_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(i)

    _RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    clusters = []
    for cluster_idx, (root, members) in enumerate(sorted(clusters_map.items())):
        member_assumptions = [assumptions[i] for i in members]

        # Highest risk
        best_risk = "low"
        scope_tag = ""
        for a in member_assumptions:
            risk = a.get("risk_level", "low")
            if _RISK_ORDER.get(risk, 3) < _RISK_ORDER.get(best_risk, 3):
                best_risk = risk
            tag = a.get("scope_tag", "").strip()
            if tag and not scope_tag:
                scope_tag = tag

        # Longest text as representative
        merged_text = max(
            (a.get("text", "") or a.get("title", "") for a in member_assumptions),
            key=len,
            default="",
        )

        label = f"{scope_tag or 'General'} ({len(member_assumptions)} assumption{'s' if len(member_assumptions) > 1 else ''})"

        clusters.append({
            "cluster_id": f"ASMPC-{cluster_idx + 1:03d}",
            "label": label,
            "assumptions": member_assumptions,
            "merged_text": merged_text,
            "scope_tag": scope_tag,
            "risk_level": best_risk,
            "count": len(member_assumptions),
        })

    return clusters


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_grouped_csv(clusters: List[dict], item_type: str = "rfi") -> str:
    """
    Export clustered items to CSV string.

    Args:
        clusters: List of cluster dicts from cluster_rfis() or cluster_assumptions().
        item_type: "rfi" or "assumption" (affects column names).

    Returns:
        CSV string with cluster_id, label, count, merged text, original IDs.
    """
    buf = io.StringIO()

    if item_type == "rfi":
        fieldnames = ["cluster_id", "label", "count", "trade", "priority",
                       "merged_question", "original_ids", "evidence_pages"]
    else:
        fieldnames = ["cluster_id", "label", "count", "scope_tag", "risk_level",
                       "merged_text", "original_ids"]

    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for cluster in clusters:
        items = cluster.get("rfis", cluster.get("assumptions", []))
        original_ids = [item.get("id", "") for item in items]

        row = {
            "cluster_id": cluster.get("cluster_id", ""),
            "label": cluster.get("label", ""),
            "count": cluster.get("count", 0),
            "original_ids": "; ".join(original_ids),
        }

        if item_type == "rfi":
            row["trade"] = cluster.get("trade", "")
            row["priority"] = cluster.get("priority", "")
            row["merged_question"] = cluster.get("merged_question", "")
            row["evidence_pages"] = ", ".join(str(p + 1) for p in cluster.get("evidence_pages", []))
        else:
            row["scope_tag"] = cluster.get("scope_tag", "")
            row["risk_level"] = cluster.get("risk_level", "")
            row["merged_text"] = cluster.get("merged_text", "")

        writer.writerow(row)

    return buf.getvalue()
