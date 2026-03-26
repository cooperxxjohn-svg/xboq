"""
Admin operations — requires X-Admin-Key header.

POST /api/admin/cleanup   — expire jobs older than TTL, delete output dirs
GET  /api/admin/disk      — disk usage summary for job outputs
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


def _require_admin(request: Request) -> None:
    """Raise 403 unless X-Admin-Key matches XBOQ_ADMIN_KEY env var."""
    admin_key = os.environ.get("XBOQ_ADMIN_KEY", "")
    if not admin_key:
        raise HTTPException(status_code=503, detail="XBOQ_ADMIN_KEY not configured")
    provided = request.headers.get("X-Admin-Key", "")
    if provided != admin_key:
        raise HTTPException(status_code=403, detail="Invalid X-Admin-Key")


@router.post("/api/admin/cleanup", include_in_schema=False)
async def trigger_cleanup(
    request: Request,
    ttl_days: int = Query(default=0, ge=0, description="Override TTL (days). 0 = use XBOQ_JOB_TTL_DAYS env default."),
) -> JSONResponse:
    """
    Manually trigger job TTL cleanup.

    Deletes jobs older than ttl_days from the database and removes their
    output directories from disk. Requires X-Admin-Key header.
    """
    _require_admin(request)

    from src.api.job_store import job_store
    deleted = job_store.expire_old_jobs(ttl_days=ttl_days)

    return JSONResponse(content={
        "deleted": deleted,
        "ttl_days_used": ttl_days or int(os.environ.get("XBOQ_JOB_TTL_DAYS", "90")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": f"Cleanup complete — {deleted} job(s) expired",
    })


@router.get("/api/admin/disk", include_in_schema=False)
async def disk_usage(request: Request) -> JSONResponse:
    """
    Return disk usage summary for job output directories.

    Requires X-Admin-Key header.
    """
    _require_admin(request)

    from src.api.job_db import OUTPUTS_DIR
    import shutil

    total_bytes = 0
    job_count = 0
    largest: list[dict] = []

    if OUTPUTS_DIR.exists():
        for job_dir in OUTPUTS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                size = sum(f.stat().st_size for f in job_dir.rglob("*") if f.is_file())
                total_bytes += size
                job_count += 1
                largest.append({"job_id": job_dir.name, "size_mb": round(size / 1024**2, 2)})
            except Exception:
                pass

    largest.sort(key=lambda x: x["size_mb"], reverse=True)

    try:
        disk = shutil.disk_usage(str(OUTPUTS_DIR))
        disk_info = {
            "total_gb": round(disk.total / 1024**3, 1),
            "used_gb": round(disk.used / 1024**3, 1),
            "free_gb": round(disk.free / 1024**3, 1),
        }
    except Exception:
        disk_info = {}

    return JSONResponse(content={
        "job_count": job_count,
        "total_size_mb": round(total_bytes / 1024**2, 1),
        "largest_jobs": largest[:10],
        "outputs_dir": str(OUTPUTS_DIR),
        "disk": disk_info,
    })
