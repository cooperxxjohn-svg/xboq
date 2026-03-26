"""
Pipeline accuracy report generator.

Generates a structured quality report from a pipeline payload, covering:
- Extraction coverage (pages processed, types found)
- Line item quality (taxonomy match rate, qty/unit completeness)
- Rate intelligence (benchmark match rate, deviation distribution)
- QTO coverage (rooms detected, finish items generated)
- Priceable vs contractual split

Usage:
    from src.reporting.accuracy_report import generate_report, format_report_text
    report = generate_report(payload)
    print(format_report_text(report))
"""
from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter
import time


def generate_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured accuracy report from a pipeline payload."""

    line_items: List[dict] = payload.get("line_items", [])
    contractual: List[dict] = payload.get("contractual_items", [])
    summary: dict = payload.get("line_items_summary", {})
    qto: dict = payload.get("qto_summary", {})
    page_index = payload.get("page_index", {})
    cache_stats = payload.get("cache_stats", {})

    total_items = len(line_items)
    total_contractual = len(contractual)

    # ── Taxonomy & completeness ──────────────────────────────────
    taxonomy_matched = sum(1 for i in line_items if i.get("taxonomy_matched"))
    qty_present      = sum(1 for i in line_items if not i.get("qty_missing", True))
    unit_present     = sum(1 for i in line_items if i.get("unit"))

    # ── Trade breakdown ──────────────────────────────────────────
    trade_counts = Counter(i.get("trade", "general") for i in line_items)

    # ── Source breakdown ─────────────────────────────────────────
    source_counts = Counter(i.get("source", "unknown") for i in line_items)

    # ── Rate intelligence ────────────────────────────────────────
    rate_statuses = Counter()
    deviations = []
    for item in line_items:
        bm = item.get("rate_benchmark", {})
        status = bm.get("status", "NO_MATCH")
        rate_statuses[status] += 1
        dev = bm.get("deviation_pct")
        if dev is not None:
            deviations.append(dev)

    benchmarked = total_items - rate_statuses.get("NO_MATCH", 0) - rate_statuses.get("UNRATED", 0)
    avg_deviation = round(sum(deviations) / len(deviations), 1) if deviations else None
    p75_deviation = None
    if deviations:
        deviations_sorted = sorted(deviations)
        p75_idx = int(len(deviations_sorted) * 0.75)
        p75_deviation = round(deviations_sorted[p75_idx], 1)

    # ── Page coverage ────────────────────────────────────────────
    pages = page_index.get("pages", []) if isinstance(page_index, dict) else []
    page_type_counts = Counter(p.get("doc_type", "unknown") for p in pages)
    total_pages = len(pages)

    # ── Scores ───────────────────────────────────────────────────
    completeness_score = round(
        (
            (taxonomy_matched / total_items if total_items else 0) * 0.4 +
            (qty_present      / total_items if total_items else 0) * 0.3 +
            (unit_present     / total_items if total_items else 0) * 0.3
        ) * 100, 1
    )

    rate_coverage_score = round(
        (benchmarked / total_items * 100) if total_items else 0, 1
    )

    overall_score = round((completeness_score * 0.6 + rate_coverage_score * 0.4), 1)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scores": {
            "overall": overall_score,
            "completeness": completeness_score,
            "rate_coverage": rate_coverage_score,
        },
        "extraction": {
            "total_pages": total_pages,
            "page_types": dict(page_type_counts),
            "priceable_items": total_items,
            "contractual_items": total_contractual,
            "priceable_ratio": round(
                total_items / (total_items + total_contractual) * 100, 1
            ) if (total_items + total_contractual) else 0,
        },
        "item_quality": {
            "taxonomy_matched": taxonomy_matched,
            "taxonomy_match_rate": round(taxonomy_matched / total_items * 100, 1) if total_items else 0,
            "qty_present": qty_present,
            "qty_completeness": round(qty_present / total_items * 100, 1) if total_items else 0,
            "unit_present": unit_present,
            "unit_completeness": round(unit_present / total_items * 100, 1) if total_items else 0,
        },
        "trade_breakdown": dict(trade_counts.most_common()),
        "source_breakdown": dict(source_counts),
        "rate_intelligence": {
            "total_benchmarked": benchmarked,
            "benchmark_coverage": round(benchmarked / total_items * 100, 1) if total_items else 0,
            "status_breakdown": dict(rate_statuses),
            "avg_deviation_pct": avg_deviation,
            "p75_deviation_pct": p75_deviation,
            "above_schedule_count": rate_statuses.get("ABOVE_SCHEDULE", 0),
            "below_schedule_count": rate_statuses.get("BELOW_SCHEDULE", 0),
        },
        "qto": {
            "rooms_detected": qto.get("rooms_detected", 0),
            "finish_items_generated": qto.get("finish_items_generated", 0),
        },
        "cache": cache_stats,
    }


def format_report_text(report: Dict[str, Any]) -> str:
    """Render a human-readable accuracy report."""
    s = report["scores"]
    e = report["extraction"]
    q = report["item_quality"]
    r = report["rate_intelligence"]
    qto = report["qto"]

    lines = [
        "=" * 60,
        "  xBOQ.ai Pipeline Accuracy Report",
        f"  Generated: {report['generated_at']}",
        "=" * 60,
        "",
        f"  Overall Score:    {s['overall']:>5.1f} / 100",
        f"  Completeness:     {s['completeness']:>5.1f} / 100",
        f"  Rate Coverage:    {s['rate_coverage']:>5.1f} / 100",
        "",
        "-- EXTRACTION -------------------------------------------------",
        f"  Pages processed:  {e['total_pages']}",
        f"  Priceable items:  {e['priceable_items']}",
        f"  Contractual:      {e['contractual_items']}  (excluded from BOQ)",
        f"  Priceable ratio:  {e['priceable_ratio']}%",
        "",
        "-- ITEM QUALITY -----------------------------------------------",
        f"  Taxonomy matched: {q['taxonomy_matched']} / {e['priceable_items']}  ({q['taxonomy_match_rate']}%)",
        f"  Qty completeness: {q['qty_completeness']}%",
        f"  Unit completeness:{q['unit_completeness']}%",
        "",
        "-- TRADE BREAKDOWN --------------------------------------------",
    ]
    for trade, count in sorted(report["trade_breakdown"].items(), key=lambda x: -x[1]):
        lines.append(f"  {trade:<20} {count:>5} items")
    lines += [
        "",
        "-- RATE INTELLIGENCE -----------------------------------------",
        f"  Benchmarked:      {r['total_benchmarked']} / {e['priceable_items']}  ({r['benchmark_coverage']}%)",
        f"  Above schedule:   {r['above_schedule_count']}",
        f"  Below schedule:   {r['below_schedule_count']}",
        f"  Avg deviation:    {r['avg_deviation_pct']}%",
        f"  P75 deviation:    {r['p75_deviation_pct']}%",
        "",
        "-- QTO --------------------------------------------------------",
        f"  Rooms detected:   {qto['rooms_detected']}",
        f"  Finish items:     {qto['finish_items_generated']}",
        "",
        "=" * 60,
    ]
    return "\n".join(str(l) for l in lines)


def save_report(report: Dict[str, Any], output_path: str) -> str:
    """Save report as JSON. Returns path."""
    import json
    from pathlib import Path
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str))
    return str(p)
