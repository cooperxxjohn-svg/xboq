"""
Rate History REST endpoints.

GET  /api/rate-history/trades                   — list trades with data
GET  /api/rate-history/{trade}/benchmarks       — statistics for a trade
POST /api/rate-history/{trade}/record           — record a new bid rate
POST /api/rate-history/{trade}/outcome          — update win/loss for a project
POST /api/rate-history/compare                  — compare BOQ items to history
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rates"])


class RecordRateRequest(BaseModel):
    project_id: str
    description: str
    unit: str
    rate_inr: float
    project_name: Optional[str] = ""
    market_rate: Optional[float] = 0.0
    won: Optional[bool] = None
    region: Optional[str] = "tier1"


class OutcomeRequest(BaseModel):
    project_id: str
    won: bool


class CompareRequest(BaseModel):
    items: List[dict]
    trade: str
    threshold_pct: Optional[float] = 15.0


@router.get("/api/rate-history/trades")
def list_trades() -> JSONResponse:
    """List all trades that have historical rate data."""
    from src.analysis.rate_history import list_trades as _lt
    trades = _lt()
    return JSONResponse(content={"trades": trades, "count": len(trades)})


@router.get("/api/rate-history/{trade}/benchmarks")
def trade_benchmarks(trade: str, region: Optional[str] = None) -> JSONResponse:
    """Return statistical benchmarks for a trade (mean, median, p25/p75, win rate)."""
    from src.analysis.rate_history import trade_benchmarks as _tb
    return JSONResponse(content=_tb(trade, region=region))


@router.post("/api/rate-history/{trade}/record", status_code=201)
def record_rate(trade: str, req: RecordRateRequest) -> JSONResponse:
    """Record a submitted bid rate for a BOQ item."""
    from src.analysis.rate_history import record as _rec
    entry = _rec(
        project_id=req.project_id,
        trade=trade,
        description=req.description,
        unit=req.unit,
        rate_inr=req.rate_inr,
        project_name=req.project_name or "",
        market_rate=req.market_rate or 0.0,
        won=req.won,
        region=req.region or "tier1",
    )
    return JSONResponse(content={"entry": entry}, status_code=201)


@router.post("/api/rate-history/{trade}/outcome")
def update_outcome(trade: str, req: OutcomeRequest) -> JSONResponse:
    """Update win/loss outcome for all entries in a project + trade."""
    from src.analysis.rate_history import update_outcome as _uo
    updated = _uo(req.project_id, trade, req.won)
    return JSONResponse(content={"trade": trade, "project_id": req.project_id,
                                  "updated": updated, "won": req.won})


@router.post("/api/rate-history/compare")
def compare_items(req: CompareRequest) -> JSONResponse:
    """Compare BOQ items to historical rate data; flags above/below/ok/no_data."""
    from src.analysis.rate_history import compare_to_history as _cth
    flagged = _cth(req.items, req.trade, threshold_pct=req.threshold_pct or 15.0)
    return JSONResponse(content={
        "trade":  req.trade,
        "items":  flagged,
        "total":  len(flagged),
        "above":  sum(1 for i in flagged if i.get("hist_flag") == "above"),
        "below":  sum(1 for i in flagged if i.get("hist_flag") == "below"),
        "ok":     sum(1 for i in flagged if i.get("hist_flag") == "ok"),
        "no_data": sum(1 for i in flagged if i.get("hist_flag") == "no_data"),
    })
