"""
Line items API — query, filter, and export line items from a completed job.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from src.api.job_store import get_job

router = APIRouter(prefix="/api/line-items", tags=["line-items"])


@router.get("/{job_id}")
async def get_line_items(
    job_id: str,
    trade: Optional[str] = Query(None, description="Filter by trade (civil, structural, architectural, etc.)"),
    status: Optional[str] = Query(None, description="Filter by rate benchmark status (ABOVE_SCHEDULE, AT_SCHEDULE, etc.)"),
    min_confidence: Optional[float] = Query(None, description="Minimum taxonomy match confidence"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return paginated line items for a completed job, with optional filtering."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    payload = job.get("result", {}) or {}
    items = payload.get("line_items", [])

    # Apply filters
    if trade:
        items = [i for i in items if i.get("trade", "").lower() == trade.lower()]
    if status:
        items = [i for i in items
                 if i.get("rate_benchmark", {}).get("status", "").upper() == status.upper()]
    if min_confidence is not None:
        items = [i for i in items if i.get("rate_benchmark", {}).get("match_confidence", 0) >= min_confidence]

    total = len(items)
    page = items[offset: offset + limit]

    return {
        "job_id": job_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": page,
    }


@router.get("/{job_id}/rate-report")
async def get_rate_report(job_id: str):
    """Return rate benchmark summary for a completed job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    payload = job.get("result", {}) or {}
    items = payload.get("line_items", [])

    from collections import Counter, defaultdict
    status_counts = Counter()
    trade_deviations = defaultdict(list)
    above_items = []

    for item in items:
        bm = item.get("rate_benchmark", {})
        status = bm.get("status", "NO_MATCH")
        status_counts[status] += 1
        dev = bm.get("deviation_pct")
        trade = item.get("trade", "general")
        if dev is not None:
            trade_deviations[trade].append(dev)
        if status == "ABOVE_SCHEDULE":
            above_items.append({
                "description": item.get("description", "")[:80],
                "trade": trade,
                "item_rate": bm.get("item_rate"),
                "dsr_rate": bm.get("dsr_rate"),
                "deviation_pct": round(dev, 1) if dev else None,
                "dsr_id": bm.get("dsr_id"),
            })

    # Sort above_items by deviation descending
    above_items.sort(key=lambda x: x.get("deviation_pct") or 0, reverse=True)

    avg_by_trade = {
        t: round(sum(devs) / len(devs), 1)
        for t, devs in trade_deviations.items() if devs
    }

    total = len(items)
    benchmarked = total - status_counts.get("NO_MATCH", 0) - status_counts.get("UNRATED", 0)

    return {
        "job_id": job_id,
        "total_items": total,
        "benchmarked": benchmarked,
        "status_breakdown": dict(status_counts),
        "avg_deviation_by_trade": avg_by_trade,
        "top_above_schedule": above_items[:20],
        "summary": payload.get("line_items_summary", {}),
        "qto_summary": payload.get("qto_summary", {}),
    }


@router.get("/{job_id}/contractual")
async def get_contractual_items(job_id: str):
    """Return contractual / administrative clauses (non-priceable items)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    payload = job.get("result", {}) or {}
    items = payload.get("contractual_items", [])

    return {
        "job_id": job_id,
        "total": len(items),
        "items": items,
    }
