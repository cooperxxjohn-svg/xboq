"""
Award Probability REST endpoints — T3-4.

POST /api/award-predict/{job_id}
    body: {competitor_count: int, price_delta_pct: float (optional)}
    returns: AwardPrediction

POST /api/award-predict/train
    — retrain model from rate_history records with won field
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["award-prediction"])


class AwardPredictRequest(BaseModel):
    competitor_count: int = 3
    price_delta_pct: Optional[float] = None   # override benchmark comparison


@router.post("/api/award-predict/{job_id}")
def predict_award(job_id: str, req: AwardPredictRequest) -> JSONResponse:
    """Predict win probability for a job."""
    from src.api.job_store import get_job
    from src.reasoning.award_predictor import AwardPredictor

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    payload = job.get("payload") or job.get("result") or {}
    predictor = AwardPredictor()
    features = predictor.extract_features(payload, req.competitor_count)

    # Allow caller to override price_delta_pct
    if req.price_delta_pct is not None:
        features["price_delta_pct"] = req.price_delta_pct

    prediction = predictor.predict(features)
    return JSONResponse(content={
        "job_id": job_id,
        "features": features,
        **prediction.to_dict(),
    })


@router.post("/api/award-predict/train")
async def retrain_model(request: Request) -> JSONResponse:
    """Retrain the award probability model from historical bid outcomes. Requires admin role."""
    from src.api.middleware.tenant_auth import get_tenant_context, TenantContext
    ctx: TenantContext = TenantContext(org_id="local", role="viewer", plan="free", authenticated=False)
    try:
        ctx = await get_tenant_context(request)
    except Exception:
        pass
    try:
        from src.auth.rbac import require_role
        require_role(ctx, "admin")
    except HTTPException:
        raise
    except Exception:
        pass  # non-fatal in dev mode

    from src.reasoning.award_predictor import train_from_history

    n = train_from_history()
    return JSONResponse(content={
        "status": "ok",
        "records_used": n,
        "message": f"Model retrained on {n} labelled bids." if n > 0 else "No labelled records found.",
    })
