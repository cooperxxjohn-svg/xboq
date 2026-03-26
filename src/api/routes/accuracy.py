"""
Accuracy benchmark dashboard — GET /api/accuracy/*.

Surfaces ground-truth comparison results for processed tenders.
This is the "proof page" — what you show an L&T evaluator to demonstrate
accuracy on real comparable projects.

Routes
------
GET  /api/accuracy/summary          — aggregate stats across all benchmarks
GET  /api/accuracy/benchmarks       — list of individual tender results
GET  /api/accuracy/benchmarks/{id}  — single tender breakdown by trade
POST /api/accuracy/benchmarks       — record a new benchmark result (internal)
GET  /api/accuracy/trends           — accuracy over time (improvement tracking)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["accuracy"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BenchmarkCreate(BaseModel):
    job_id:          str
    tender_ref:      str              = ""
    tender_type:     str              = ""   # hospital | residential | industrial | infrastructure
    qty_delta_pct:   float            = 0.0
    rate_delta_pct:  float            = 0.0
    item_recall:     float            = 0.0  # 0–1
    item_precision:  float            = 0.0  # 0–1
    trade_deltas:    dict             = Field(default_factory=dict)
    run_mode:        str              = ""
    page_count:      int              = 0
    processing_secs: float            = 0.0


# ---------------------------------------------------------------------------
# GET /api/accuracy/summary
# ---------------------------------------------------------------------------

@router.get("/api/accuracy/summary")
async def accuracy_summary(request: Request, org_id: Optional[str] = None) -> JSONResponse:
    """
    High-level accuracy overview for the dashboard.

    Returns:
      - total_tenders_benchmarked
      - median_qty_delta_pct      (lower is better)
      - median_rate_delta_pct
      - median_item_recall_pct    (higher is better)
      - by_tender_type: stats per building category
      - best / worst performing tender
    """
    resolved_org = org_id or "local"
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        ctx = await get_tenant_context(request)
        if ctx.authenticated:
            resolved_org = org_id or ctx.org_id
    except Exception:
        pass

    try:
        from src.api.db import SessionLocal
        from src.api.models import AccuracyBenchmarkModel
        from sqlalchemy import select

        with SessionLocal() as db:
            rows = db.execute(
                select(AccuracyBenchmarkModel)
                .where(AccuracyBenchmarkModel.org_id == resolved_org)
                .order_by(AccuracyBenchmarkModel.created_at.desc())
            ).scalars().all()
    except Exception as exc:
        logger.error("Failed to query accuracy benchmarks: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    if not rows:
        return JSONResponse({
            "total_tenders_benchmarked": 0,
            "message": "No benchmark results recorded yet. Process tenders with ground-truth BOQs to populate.",
        })

    import statistics

    qty_deltas  = [r.qty_delta_pct  for r in rows]
    rate_deltas = [r.rate_delta_pct for r in rows]
    recalls     = [r.item_recall    for r in rows]
    precisions  = [r.item_precision for r in rows]

    # Per-type breakdown
    by_type: dict[str, list] = {}
    for r in rows:
        t = r.tender_type or "unknown"
        by_type.setdefault(t, []).append(r)

    def _type_stats(rlist):
        return {
            "count": len(rlist),
            "median_qty_delta_pct":  round(statistics.median([r.qty_delta_pct  for r in rlist]), 1),
            "median_rate_delta_pct": round(statistics.median([r.rate_delta_pct for r in rlist]), 1),
            "median_recall_pct":     round(statistics.median([r.item_recall     for r in rlist]) * 100, 1),
        }

    # Best / worst
    best  = min(rows, key=lambda r: r.qty_delta_pct)
    worst = max(rows, key=lambda r: r.qty_delta_pct)

    return JSONResponse({
        "total_tenders_benchmarked": len(rows),
        "org_id": resolved_org,
        "median_qty_delta_pct":   round(statistics.median(qty_deltas), 1),
        "median_rate_delta_pct":  round(statistics.median(rate_deltas), 1),
        "median_item_recall_pct": round(statistics.median(recalls) * 100, 1),
        "median_item_precision_pct": round(statistics.median(precisions) * 100, 1),
        "by_tender_type":         {t: _type_stats(v) for t, v in sorted(by_type.items())},
        "best_tender":  {"ref": best.tender_ref,  "qty_delta_pct": best.qty_delta_pct},
        "worst_tender": {"ref": worst.tender_ref, "qty_delta_pct": worst.qty_delta_pct},
    })


# ---------------------------------------------------------------------------
# GET /api/accuracy/benchmarks
# ---------------------------------------------------------------------------

@router.get("/api/accuracy/benchmarks")
async def list_benchmarks(
    request: Request,
    org_id: Optional[str] = None,
    tender_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> JSONResponse:
    """List benchmark results, newest first. Filterable by tender type."""
    resolved_org = org_id or "local"
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        ctx = await get_tenant_context(request)
        if ctx.authenticated:
            resolved_org = org_id or ctx.org_id
    except Exception:
        pass

    try:
        from src.api.db import SessionLocal
        from src.api.models import AccuracyBenchmarkModel
        from sqlalchemy import select

        with SessionLocal() as db:
            q = select(AccuracyBenchmarkModel).where(
                AccuracyBenchmarkModel.org_id == resolved_org
            )
            if tender_type:
                q = q.where(AccuracyBenchmarkModel.tender_type == tender_type)
            q = q.order_by(AccuracyBenchmarkModel.created_at.desc()).limit(limit).offset(offset)
            rows = db.execute(q).scalars().all()
    except Exception as exc:
        logger.exception("DB error in %s", __name__)
        raise HTTPException(status_code=500, detail="Internal server error")

    return JSONResponse({"benchmarks": [r.to_dict() for r in rows], "total": len(rows)})


# ---------------------------------------------------------------------------
# GET /api/accuracy/benchmarks/{benchmark_id}
# ---------------------------------------------------------------------------

@router.get("/api/accuracy/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: int, request: Request) -> JSONResponse:
    """Get a single benchmark result with full per-trade breakdown."""
    try:
        from src.api.db import SessionLocal
        from src.api.models import AccuracyBenchmarkModel

        with SessionLocal() as db:
            row = db.get(AccuracyBenchmarkModel, benchmark_id)
    except Exception as exc:
        logger.exception("DB error in %s", __name__)
        raise HTTPException(status_code=500, detail="Internal server error")

    if row is None:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")

    return JSONResponse(row.to_dict())


# ---------------------------------------------------------------------------
# POST /api/accuracy/benchmarks  (internal — called by pipeline after run)
# ---------------------------------------------------------------------------

@router.post("/api/accuracy/benchmarks", status_code=201)
async def record_benchmark(body: BenchmarkCreate, request: Request) -> JSONResponse:
    """
    Record a new accuracy benchmark result.

    Called internally by the pipeline when ground-truth BOQ is available.
    Requires editor role.
    """
    org_id = "local"
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        from src.auth.rbac import require_role
        ctx = await get_tenant_context(request)
        require_role(ctx, "editor")
        if ctx.authenticated:
            org_id = ctx.org_id
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        import json
        from src.api.db import SessionLocal
        from src.api.models import AccuracyBenchmarkModel

        record = AccuracyBenchmarkModel(
            job_id=body.job_id,
            org_id=org_id,
            tender_ref=body.tender_ref,
            tender_type=body.tender_type,
            qty_delta_pct=body.qty_delta_pct,
            rate_delta_pct=body.rate_delta_pct,
            item_recall=body.item_recall,
            item_precision=body.item_precision,
            trade_deltas_json=json.dumps(body.trade_deltas, default=str),
            run_mode=body.run_mode,
            page_count=body.page_count,
            processing_secs=body.processing_secs,
            created_at=datetime.now(timezone.utc),
        )

        with SessionLocal() as db:
            db.add(record)
            db.commit()
            db.refresh(record)
            new_id = record.id

    except Exception as exc:
        logger.exception("DB error in %s", __name__)
        raise HTTPException(status_code=500, detail="Internal server error")

    return JSONResponse(status_code=201, content={"id": new_id, "job_id": body.job_id})


# ---------------------------------------------------------------------------
# GET /api/accuracy/trends
# ---------------------------------------------------------------------------

@router.get("/api/accuracy/trends")
async def accuracy_trends(request: Request, org_id: Optional[str] = None, days: int = 90) -> JSONResponse:
    """
    Accuracy over time — shows whether the system is improving.

    Groups results by week, returns median qty_delta_pct per week.
    """
    from datetime import timedelta
    resolved_org = org_id or "local"
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        ctx = await get_tenant_context(request)
        if ctx.authenticated:
            resolved_org = org_id or ctx.org_id
    except Exception:
        pass

    try:
        from src.api.db import SessionLocal
        from src.api.models import AccuracyBenchmarkModel
        from sqlalchemy import select

        with SessionLocal() as db:
            rows = db.execute(
                select(AccuracyBenchmarkModel)
                .where(
                    AccuracyBenchmarkModel.org_id == resolved_org,
                    AccuracyBenchmarkModel.created_at >= cutoff,
                )
                .order_by(AccuracyBenchmarkModel.created_at)
            ).scalars().all()
    except Exception as exc:
        logger.exception("DB error in %s", __name__)
        raise HTTPException(status_code=500, detail="Internal server error")

    import statistics
    from collections import defaultdict

    # Group by ISO week
    weekly: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.created_at:
            week_key = r.created_at.strftime("%Y-W%W")
            weekly[week_key].append(r.qty_delta_pct)

    trend = [
        {"week": week, "median_qty_delta_pct": round(statistics.median(vals), 1), "count": len(vals)}
        for week, vals in sorted(weekly.items())
    ]

    return JSONResponse({
        "org_id": resolved_org,
        "days": days,
        "trend": trend,
        "message": "Each data point is one week of processed tenders vs ground truth.",
    })
