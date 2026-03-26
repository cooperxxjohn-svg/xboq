"""
Tender Benchmark REST endpoints — T3-2.

GET  /api/benchmarks/cost-per-sqm?building_type=residential&region=delhi_ncr
GET  /api/benchmarks/trades?trade=civil&region=tier1
POST /api/benchmarks/compare   — compare job's cost to market
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["benchmarks"])


class CompareRequest(BaseModel):
    job_id: str
    building_type: str = ""
    region: str = "tier1"


@router.get("/api/benchmarks/cost-per-sqm")
def get_cost_per_sqm(building_type: str = "residential", region: str = "tier1") -> JSONResponse:
    """Return market cost/sqm statistics for a building type and region."""
    from src.analysis.tender_benchmarks import benchmark_cost_per_sqm
    result = benchmark_cost_per_sqm(building_type, region)
    return JSONResponse(content=result)


@router.get("/api/benchmarks/trades")
def get_trade_benchmark(trade: str = "civil", region: str = "tier1") -> JSONResponse:
    """Return cost/sqm statistics for a specific trade."""
    from src.analysis.tender_benchmarks import benchmark_trade_rates
    result = benchmark_trade_rates(trade, region)
    return JSONResponse(content=result)


@router.post("/api/benchmarks/compare")
def compare_to_market(req: CompareRequest) -> JSONResponse:
    """Compare a job's cost/sqm to market benchmarks."""
    from src.api.job_store import get_job
    from src.analysis.tender_benchmarks import (
        compare_tender_to_market,
        detect_building_type,
    )

    job = get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{req.job_id}' not found")

    payload = job.get("payload") or job.get("result", {})
    building_type = req.building_type or detect_building_type(payload)
    comparison = compare_tender_to_market(payload, building_type, req.region)
    return JSONResponse(content={
        "job_id": req.job_id,
        "building_type": building_type,
        "region": req.region,
        **comparison,
    })
