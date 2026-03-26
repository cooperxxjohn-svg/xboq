"""
Aerial Site Measurement REST endpoint — T4-5.

POST /api/aerial-measure
  body: {address: str, zoom: int = 18}
  returns: SiteMeasurement as JSON

GOOGLE_MAPS_API_KEY read from env; graceful fallback when absent.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["aerial-measurement"])


class AerialMeasureRequest(BaseModel):
    address: str
    zoom: int = 18


@router.post("/api/aerial-measure")
def aerial_measure(req: AerialMeasureRequest) -> JSONResponse:
    """Measure site footprint, roads, and laydown areas from satellite imagery."""
    from src.analysis.aerial_measurement import measure_site

    result = measure_site(address=req.address, zoom=req.zoom)
    return JSONResponse(content=result.to_dict())
