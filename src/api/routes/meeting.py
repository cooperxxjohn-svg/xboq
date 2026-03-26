"""
Pre-Bid Meeting Assistant REST endpoints — T3-3.

POST /api/meeting/{job_id}/transcript  — text body → Q&A → RFIs
POST /api/meeting/{job_id}/audio       — multipart audio → Whisper → Q&A → RFIs
GET  /api/meeting/{job_id}/rfis        — list meeting-sourced RFIs
"""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["meeting"])


class TranscriptRequest(BaseModel):
    transcript: str


@router.post("/api/meeting/{job_id}/transcript")
async def process_transcript(job_id: str, req: TranscriptRequest) -> JSONResponse:
    """Process plain-text transcript → extract Q&A → create RFIs."""
    from src.analysis.meeting_assistant import process_meeting
    result = process_meeting(req.transcript, job_id)
    return JSONResponse(content={
        "job_id": job_id,
        **result,
    })


@router.post("/api/meeting/{job_id}/audio")
async def process_audio(
    job_id: str,
    audio: UploadFile = File(...),
) -> JSONResponse:
    """Upload an audio recording → Whisper transcription → Q&A → RFIs."""
    from src.analysis.meeting_assistant import process_meeting

    ALLOWED_TYPES = {"audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a",
                     "audio/webm", "audio/ogg", "audio/flac", "video/mp4"}
    content_type = audio.content_type or ""
    if content_type and content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {content_type}")

    # Save to temp file (Whisper API needs a file path / file object)
    suffix = os.path.splitext(audio.filename or "audio.mp3")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = process_meeting(tmp_path, job_id)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return JSONResponse(content={
        "job_id": job_id,
        "audio_filename": audio.filename,
        **result,
    })


@router.get("/api/meeting/{job_id}/rfis")
def get_meeting_rfis(job_id: str) -> JSONResponse:
    """List RFIs that originated from a meeting transcript."""
    from src.api.job_store import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    payload = job.get("payload") or job.get("result") or {}
    all_rfis = payload.get("rfis", [])
    meeting_rfis = [r for r in all_rfis if r.get("source") == "meeting"]

    return JSONResponse(content={
        "job_id": job_id,
        "rfis": meeting_rfis,
        "count": len(meeting_rfis),
    })
