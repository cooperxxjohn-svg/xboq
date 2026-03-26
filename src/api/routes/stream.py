"""
GET /api/stream/{job_id}  — Server-Sent Events (SSE) live pipeline progress.

Streams progress updates as text/event-stream until the job reaches
a terminal state ("complete" or "error"). Clients can connect immediately
after POST /api/analyze returns a job_id and receive real-time updates.

Event format (one per SSE data block):
    {
        "job_id":  "...",
        "status":  "queued" | "processing" | "complete" | "error",
        "progress": 0.0 – 1.0,
        "message": "Extracting BOQ items…",
        "elapsed_s": 4.2,          // seconds since job was created
        "stage_summary": {...}     // present only when complete
    }

Clients should stop reading once status is "complete" or "error".
The stream ends with a final event then closes.

Example (JavaScript):
    const es = new EventSource("/api/stream/job-123");
    es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        console.log(data.progress, data.message);
        if (data.status === "complete" || data.status === "error") es.close();
    };
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

_POLL_INTERVAL = 0.4   # seconds between checks
_TIMEOUT_S     = 600   # max stream duration (10 min)


async def _event_stream(job_id: str):
    """Async generator yielding SSE-formatted data events."""
    start_time = datetime.now(timezone.utc)

    # Verify job exists before entering loop
    if job_store.get_job(job_id) is None:
        payload = json.dumps({"error": "not_found", "job_id": job_id})
        yield f"data: {payload}\n\n"
        return

    elapsed = 0.0
    while elapsed < _TIMEOUT_S:
        job = job_store.get_job(job_id)

        if job is None:
            # Job disappeared — stream a final error event
            payload = json.dumps({"error": "job_disappeared", "job_id": job_id})
            yield f"data: {payload}\n\n"
            return

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        data: dict = {
            "job_id":    job.job_id,
            "status":    job.status,
            "progress":  round(job.progress, 3),
            "message":   job.progress_message or "",
            "elapsed_s": round(elapsed, 1),
        }

        # Include summary data when complete
        if job.status == "complete" and job.payload:
            p = job.payload
            data["stage_summary"] = {
                "total_pages":    p.get("processing_stats", {}).get("total_pages", 0),
                "boq_items":      len(p.get("boq_items", [])),
                "rfis":           len(p.get("rfis", [])),
                "blockers":       len(p.get("blockers", [])),
                "qa_score":       p.get("qa_score", {}).get("overall_score"),
            }

        if job.errors:
            data["errors"] = job.errors

        yield f"data: {json.dumps(data)}\n\n"

        # Terminal states — emit once and close
        if job.status in ("complete", "error"):
            return

        await asyncio.sleep(_POLL_INTERVAL)

    # Timeout reached — inform client
    timeout_data = json.dumps({
        "job_id":    job_id,
        "status":    "timeout",
        "elapsed_s": round(elapsed, 1),
        "message":   "Stream timed out; poll /api/jobs/{job_id} for status",
    })
    yield f"data: {timeout_data}\n\n"


@router.get("/api/stream/{job_id}")
async def stream_job_progress(job_id: str) -> StreamingResponse:
    """
    Stream live progress events for a running analysis job via SSE.

    Connect immediately after starting an analysis. The stream closes once
    the job reaches "complete" or "error" status.
    """
    # Quick existence check so we can return 404 instead of an SSE error
    if job_store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return StreamingResponse(
        _event_stream(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
            "Connection":      "keep-alive",
        },
    )
