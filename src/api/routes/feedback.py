"""
Estimator feedback loop — POST /api/feedback, GET /api/feedback/summary.

Every time an estimator corrects a quantity, rate, unit, or description,
the correction is stored in the estimator_feedback table alongside the
original AI value.

This builds the ground-truth dataset that will be used to:
  1. Calibrate AI confidence scores (are "90% confident" items right 90% of the time?)
  2. Detect systematic biases (always over-estimates earthwork by 18%)
  3. Seed future fine-tuning once enough corrections accumulate

API
---
POST /api/feedback          — submit a correction
GET  /api/feedback/summary  — aggregated accuracy stats per trade/field
GET  /api/feedback/export   — raw corrections as CSV (admin only)
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.analytics import log_event

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    job_id:          str
    field_type:      str              = Field(..., pattern="^(quantity|rate|unit|description)$")
    trade:           str              = ""
    item_ref:        str              = ""
    ai_value:        str              = ""
    human_value:     str
    unit:            str              = ""
    confidence_was:  float            = 0.0
    drawing_ref:     str              = ""
    rule_applied:    str              = ""
    correction_note: str              = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_error(ai_val: str, human_val: str) -> Optional[float]:
    """Return percentage error if both values are numeric, else None."""
    try:
        ai = float(ai_val)
        hv = float(human_val)
        if hv == 0:
            return None
        return abs(ai - hv) / abs(hv) * 100.0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# POST /api/feedback
# ---------------------------------------------------------------------------

@router.post("/api/feedback", status_code=201)
async def submit_feedback(body: FeedbackRequest, request: Request) -> JSONResponse:
    """
    Record an estimator correction.

    Stores the original AI value alongside the human correction so we can
    measure accuracy drift and detect systematic biases over time.
    """
    # Resolve org_id + user_id from auth context
    org_id = "local"
    user_id = ""
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        ctx = await get_tenant_context(request)
        if ctx.authenticated:
            org_id = ctx.org_id
            user_id = ctx.user_id
    except Exception:
        pass

    try:
        from src.api.db import SessionLocal
        from src.api.models import EstimatorFeedbackModel

        record = EstimatorFeedbackModel(
            job_id=body.job_id,
            org_id=org_id,
            user_id=user_id,
            field_type=body.field_type,
            trade=body.trade,
            item_ref=body.item_ref,
            ai_value=str(body.ai_value),
            human_value=str(body.human_value),
            unit=body.unit,
            confidence_was=body.confidence_was,
            drawing_ref=body.drawing_ref,
            rule_applied=body.rule_applied,
            correction_note=body.correction_note,
            created_at=datetime.now(timezone.utc),
        )

        with SessionLocal() as db:
            db.add(record)
            db.commit()
            db.refresh(record)
            feedback_id = record.id

    except Exception as exc:
        logger.error("Failed to store feedback: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {exc}")

    pct = _pct_error(body.ai_value, body.human_value)

    log_event("feedback_submitted", job_id=body.job_id, org_id=org_id,
              extra={"field_type": body.field_type, "trade": body.trade})

    return JSONResponse(
        status_code=201,
        content={
            "id": feedback_id,
            "job_id": body.job_id,
            "field_type": body.field_type,
            "pct_error": round(pct, 2) if pct is not None else None,
            "message": "Correction recorded. Thank you — this improves future estimates.",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/feedback/summary
# ---------------------------------------------------------------------------

@router.get("/api/feedback/summary")
async def feedback_summary(request: Request, org_id: Optional[str] = None) -> JSONResponse:
    """
    Aggregated accuracy statistics per trade and field type.

    Returns:
      - corrections_total: total corrections stored
      - by_trade: {trade: {count, median_pct_error, mean_pct_error}}
      - by_field: {field_type: {count, median_pct_error}}
      - confidence_calibration: {bucket: {expected_accuracy, actual_accuracy}}
    """
    # Resolve org from auth if not supplied
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
        from src.api.models import EstimatorFeedbackModel
        from sqlalchemy import select

        with SessionLocal() as db:
            rows = db.execute(
                select(EstimatorFeedbackModel)
                .where(EstimatorFeedbackModel.org_id == resolved_org)
                .order_by(EstimatorFeedbackModel.created_at.desc())
                .limit(5000)
            ).scalars().all()

    except Exception as exc:
        logger.error("Failed to query feedback: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    if not rows:
        return JSONResponse({
            "corrections_total": 0,
            "by_trade": {},
            "by_field": {},
            "confidence_calibration": {},
            "message": "No corrections recorded yet.",
        })

    # ── Aggregate ────────────────────────────────────────────────────────
    import statistics

    by_trade: dict[str, list[float]] = {}
    by_field: dict[str, list[float]] = {}
    conf_buckets: dict[str, list[bool]] = {
        "0-25": [], "25-50": [], "50-75": [], "75-100": []
    }

    for row in rows:
        pct = _pct_error(row.ai_value, row.human_value)
        trade = row.trade or "unknown"
        field = row.field_type

        if pct is not None:
            by_trade.setdefault(trade, []).append(pct)
            by_field.setdefault(field, []).append(pct)

        # Confidence calibration: was the AI right when it said it was confident?
        conf = row.confidence_was * 100  # 0–100
        correct = pct is not None and pct < 10.0  # within 10% = "correct"
        if conf < 25:
            conf_buckets["0-25"].append(correct)
        elif conf < 50:
            conf_buckets["25-50"].append(correct)
        elif conf < 75:
            conf_buckets["50-75"].append(correct)
        else:
            conf_buckets["75-100"].append(correct)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "median_pct_error": None, "mean_pct_error": None}
        return {
            "count": len(values),
            "median_pct_error": round(statistics.median(values), 1),
            "mean_pct_error": round(statistics.mean(values), 1),
        }

    calibration = {}
    for bucket, results in conf_buckets.items():
        if results:
            calibration[bucket] = {
                "sample_count": len(results),
                "actual_accuracy_pct": round(sum(results) / len(results) * 100, 1),
            }

    return JSONResponse({
        "corrections_total": len(rows),
        "org_id": resolved_org,
        "by_trade": {t: _stats(v) for t, v in sorted(by_trade.items())},
        "by_field": {f: _stats(v) for f, v in sorted(by_field.items())},
        "confidence_calibration": calibration,
    })


# ---------------------------------------------------------------------------
# GET /api/feedback/export  (admin only — CSV dump)
# ---------------------------------------------------------------------------

@router.get("/api/feedback/export")
async def export_feedback(request: Request) -> StreamingResponse:
    """Export all corrections as CSV. Admin role required."""
    try:
        from src.api.middleware.tenant_auth import get_tenant_context
        from src.auth.rbac import require_role
        ctx = await get_tenant_context(request)
        require_role(ctx, "admin")
        org_id = ctx.org_id
    except HTTPException:
        raise
    except Exception:
        org_id = "local"

    try:
        from src.api.db import SessionLocal
        from src.api.models import EstimatorFeedbackModel
        from sqlalchemy import select

        with SessionLocal() as db:
            rows = db.execute(
                select(EstimatorFeedbackModel)
                .where(EstimatorFeedbackModel.org_id == org_id)
                .order_by(EstimatorFeedbackModel.created_at)
            ).scalars().all()
    except Exception as exc:
        logger.exception("DB error in %s", __name__)
        raise HTTPException(status_code=500, detail="Internal server error")

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "id", "job_id", "field_type", "trade", "item_ref",
        "ai_value", "human_value", "unit", "confidence_was",
        "drawing_ref", "rule_applied", "correction_note",
        "pct_error", "user_id", "created_at",
    ])
    writer.writeheader()
    for row in rows:
        pct = _pct_error(row.ai_value, row.human_value)
        writer.writerow({
            "id": row.id,
            "job_id": row.job_id,
            "field_type": row.field_type,
            "trade": row.trade,
            "item_ref": row.item_ref,
            "ai_value": row.ai_value,
            "human_value": row.human_value,
            "unit": row.unit,
            "confidence_was": row.confidence_was,
            "drawing_ref": row.drawing_ref,
            "rule_applied": row.rule_applied,
            "correction_note": row.correction_note,
            "pct_error": round(pct, 2) if pct is not None else "",
            "user_id": row.user_id,
            "created_at": row.created_at.isoformat() if row.created_at else "",
        })

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=xboq_feedback_{org_id}.csv"},
    )
