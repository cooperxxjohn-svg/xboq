"""
Quality Dashboard Metrics — computes trends from feedback.jsonl and run history.

Pure function. No external dependencies beyond the standard library.

Sprint 15: Packaging + Proof + Meeting Workflow.
"""

from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict


# =============================================================================
# NOISY CHECK IDENTIFICATION
# =============================================================================

def identify_noisy_checks(
    feedback_entries: List[dict],
    top_n: int = 5,
) -> List[dict]:
    """
    Find checks that are most frequently marked 'wrong'.

    Groups feedback by item_id, counts wrong verdicts, sorts descending.

    Args:
        feedback_entries: List of feedback entry dicts with 'item_id' and 'verdict'.
        top_n: Number of top noisy checks to return.

    Returns:
        List of dicts sorted by wrong count descending:
        [{"check_id": str, "wrong_count": int, "total_count": int}]
    """
    wrong_counts = Counter()
    total_counts = Counter()

    for entry in feedback_entries:
        item_id = entry.get("item_id", "unknown")
        verdict = entry.get("verdict", "")
        total_counts[item_id] += 1
        if verdict == "wrong":
            wrong_counts[item_id] += 1

    # Build sorted list
    results = []
    for item_id in total_counts:
        if wrong_counts[item_id] > 0:
            results.append({
                "check_id": item_id,
                "wrong_count": wrong_counts[item_id],
                "total_count": total_counts[item_id],
            })

    results.sort(key=lambda x: (-x["wrong_count"], x["check_id"]))
    return results[:top_n]


# =============================================================================
# MAIN METRICS COMPUTATION
# =============================================================================

def compute_quality_metrics(
    feedback_entries: List[dict],
    run_history: List[dict],
    cache_stats: Optional[dict] = None,
) -> dict:
    """
    Compute quality dashboard metrics from feedback and run history.

    Args:
        feedback_entries: From load_feedback() — list of dicts with:
            {feedback_type, item_id, verdict, timestamp, ...}
            Verdicts: "correct", "wrong", "edited".
        run_history: From list_runs() — list of dicts with:
            {run_id, timestamp, readiness_score, rfis_count, blockers_count, ...}
        cache_stats: Optional dict with:
            {hits, misses, time_saved_seconds}

    Returns:
        Dict with all quality metrics.
    """
    cache_stats = cache_stats or {}

    # ── Feedback breakdown ─────────────────────────────────────────────
    rfi_correct = 0
    rfi_wrong = 0
    rfi_edited = 0
    qty_correct = 0
    qty_wrong = 0
    qty_edited = 0
    total_feedback = len(feedback_entries)

    for entry in feedback_entries:
        fb_type = entry.get("feedback_type", "")
        verdict = entry.get("verdict", "")

        if fb_type == "rfi":
            if verdict == "correct":
                rfi_correct += 1
            elif verdict == "wrong":
                rfi_wrong += 1
            elif verdict == "edited":
                rfi_edited += 1
        elif fb_type == "quantity":
            if verdict == "correct":
                qty_correct += 1
            elif verdict == "wrong":
                qty_wrong += 1
            elif verdict == "edited":
                qty_edited += 1

    rfi_total = rfi_correct + rfi_wrong + rfi_edited
    qty_total = qty_correct + qty_wrong + qty_edited
    total_wrong_edited = sum(
        1 for e in feedback_entries if e.get("verdict") in ("wrong", "edited")
    )

    rfi_acceptance_rate = rfi_correct / rfi_total if rfi_total > 0 else 0.0
    qty_acceptance_rate = qty_correct / qty_total if qty_total > 0 else 0.0
    correction_rate = total_wrong_edited / total_feedback if total_feedback > 0 else 0.0

    # ── Cache stats ────────────────────────────────────────────────────
    cache_hits = cache_stats.get("hits", 0)
    cache_misses = cache_stats.get("misses", 0)
    cache_total = cache_hits + cache_misses
    cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0
    cache_time_saved = cache_stats.get("time_saved_seconds", 0.0)

    # ── Top noisy checks ──────────────────────────────────────────────
    top_noisy = identify_noisy_checks(feedback_entries, top_n=5)

    # ── Trend data from run history ───────────────────────────────────
    trend_data = []
    for run in sorted(run_history, key=lambda r: r.get("timestamp", "")):
        trend_data.append({
            "run_id": run.get("run_id", ""),
            "timestamp": run.get("timestamp", ""),
            "readiness_score": run.get("readiness_score", 0),
            "rfis_count": run.get("rfis_count", 0),
        })

    # ── Feedback volume by day ────────────────────────────────────────
    day_counts = defaultdict(lambda: {"correct": 0, "wrong": 0, "edited": 0})
    for entry in feedback_entries:
        ts = entry.get("timestamp", "")
        date_key = ts[:10] if len(ts) >= 10 else "unknown"
        verdict = entry.get("verdict", "")
        if verdict in ("correct", "wrong", "edited"):
            day_counts[date_key][verdict] += 1

    feedback_by_day = sorted(
        [
            {"date": d, **counts}
            for d, counts in day_counts.items()
        ],
        key=lambda x: x["date"],
    )

    return {
        "rfi_acceptance_rate": rfi_acceptance_rate,
        "rfi_correct_count": rfi_correct,
        "rfi_wrong_count": rfi_wrong,
        "rfi_edited_count": rfi_edited,
        "quantity_acceptance_rate": qty_acceptance_rate,
        "correction_rate": correction_rate,
        "total_feedback_count": total_feedback,
        "cache_hit_rate": cache_hit_rate,
        "cache_time_saved_seconds": cache_time_saved,
        "top_noisy_checks": top_noisy,
        "trend_data": trend_data,
        "feedback_by_day": feedback_by_day,
    }
