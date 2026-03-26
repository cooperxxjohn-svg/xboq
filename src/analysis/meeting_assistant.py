"""
Pre-Bid Meeting Assistant — T3-3.

Accepts an audio file or text transcript from a pre-bid meeting,
extracts Q&A pairs, then maps unanswered questions to new RFIs.

Usage paths:
  1. paste transcript text → extract_qa_pairs() → qa_to_rfis()
  2. upload audio file   → transcribe_audio() → extract_qa_pairs() → qa_to_rfis()

Future: Recall.ai real-time bot for live call integration (noted in docstring only; MVP = post-meeting).

OpenAI Whisper is used for audio transcription when available;
gracefully degrades to "" (no crash) when OPENAI_API_KEY is absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MeetingQA:
    question: str
    answer: str
    source_text: str
    timestamp_hint: str = ""   # "00:12:34" if parseable from transcript

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeetingRFI:
    rfi_id: str
    question: str
    source: str = "meeting"
    trade: str = "general"
    priority: str = "medium"      # "high" | "medium"
    context_snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trade keyword classifier (same vocabulary as rfi_engine._classify_trade)
# ---------------------------------------------------------------------------

_TRADE_KEYWORDS: Dict[str, List[str]] = {
    "waterproofing": ["waterproof", "dampproof", "seepage", "leakage", "tanking"],
    "painting": ["paint", "distemper", "primer", "coating", "finish"],
    "civil": ["concrete", "rcc", "brickwork", "masonry", "excavation", "foundation", "plinth"],
    "structural": ["steel", "rebar", "beam", "column", "slab", "footing", "load", "structural"],
    "mep": ["electrical", "plumbing", "hvac", "fire", "mep", "drainage", "sanitary", "wiring"],
    "finishing": ["tile", "flooring", "ceiling", "plaster", "dado", "cladding", "granite", "marble"],
    "boq": ["rate", "quantity", "item", "boq", "bill of quantities", "tender item"],
    "schedule": ["milestone", "duration", "completion", "mobilisation", "handover", "programme"],
}


def _classify_trade(text: str) -> str:
    lower = text.lower()
    for trade, keywords in _TRADE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return trade
    return "general"


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, llm_client: Any = None) -> str:
    """
    Transcribe an audio file using OpenAI Whisper API.

    Returns empty string if OPENAI_API_KEY is not set or transcription fails.
    Does NOT raise — graceful fallback.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("transcribe_audio: no OPENAI_API_KEY — returning empty transcript")
        return ""

    try:
        import openai
        client = llm_client or openai.OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
        return str(result)
    except Exception as exc:
        logger.warning("transcribe_audio failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Q&A extraction
# ---------------------------------------------------------------------------

# Timestamp pattern e.g. "[00:12:34]" or "(12:34)"
_TIMESTAMP_RE = re.compile(r"[\[\(](\d{1,2}:\d{2}(?::\d{2})?)\s*[\]\)]")

# Patterns: "Q: ...", "Question: ...", "QUESTION ...", "Q1.", etc.
_Q_RE = re.compile(r"^\s*(?:Q(?:uestion)?[\d\s\.\:]+|Q\s*:)\s*(.+)", re.IGNORECASE)
_A_RE = re.compile(r"^\s*(?:A(?:nswer)?[\d\s\.\:]+|A\s*:)\s*(.+)", re.IGNORECASE)


def _regex_extract(transcript: str) -> List[MeetingQA]:
    """Heuristic extraction using Q:/A: line patterns."""
    pairs: List[MeetingQA] = []
    lines = transcript.splitlines()
    current_q: Optional[str] = None
    current_q_raw = ""
    current_ts = ""

    for line in lines:
        ts_match = _TIMESTAMP_RE.search(line)
        if ts_match:
            current_ts = ts_match.group(1)

        q_match = _Q_RE.match(line)
        a_match = _A_RE.match(line)

        if q_match:
            current_q = q_match.group(1).strip()
            current_q_raw = line.strip()
        elif a_match and current_q:
            answer = a_match.group(1).strip()
            pairs.append(MeetingQA(
                question=current_q,
                answer=answer,
                source_text=current_q_raw + "\n" + line.strip(),
                timestamp_hint=current_ts,
            ))
            current_q = None
            current_q_raw = ""
            current_ts = ""

    return pairs[:50]   # cap at 50


def _llm_extract(transcript: str, llm_client: Any) -> List[MeetingQA]:
    """Use LLM to extract Q&A pairs from transcript."""
    prompt = (
        "Extract question-answer pairs from this pre-bid meeting transcript. "
        "Return a JSON array of objects with keys: question, answer, source_text, timestamp_hint. "
        "Include only genuine questions and their answers. Maximum 50 pairs.\n\n"
        f"TRANSCRIPT:\n{transcript[:8000]}"
    )
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        return [
            MeetingQA(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                source_text=str(item.get("source_text", "")),
                timestamp_hint=str(item.get("timestamp_hint", "")),
            )
            for item in data
            if item.get("question")
        ][:50]
    except Exception as exc:
        logger.warning("_llm_extract failed: %s — falling back to regex", exc)
        return _regex_extract(transcript)


def extract_qa_pairs(transcript: str, llm_client: Any = None) -> List[MeetingQA]:
    """
    Extract Q&A pairs from transcript text.

    Uses LLM when llm_client is provided, otherwise regex heuristic.
    """
    if not transcript or not transcript.strip():
        return []
    if llm_client is not None:
        return _llm_extract(transcript, llm_client)
    return _regex_extract(transcript)


# ---------------------------------------------------------------------------
# Q&A → RFI conversion
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity (Jaccard-ish), 0-1."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def qa_to_rfis(
    qa_pairs: List[MeetingQA],
    existing_rfis: List[dict],
    job_payload: dict,
) -> List[MeetingRFI]:
    """
    Convert Q&A pairs to MeetingRFI objects.

    Deduplication: skip if a similar question (≥60% overlap) already exists
    in existing_rfis.
    """
    new_rfis: List[MeetingRFI] = []
    existing_questions = [str(r.get("question", "")) for r in existing_rfis]

    for i, qa in enumerate(qa_pairs):
        # Dedup check
        is_dup = any(
            _similarity(qa.question, eq) >= 0.60
            for eq in existing_questions
        )
        if is_dup:
            continue

        trade = _classify_trade(qa.question + " " + qa.answer)
        priority = "high" if any(
            kw in qa.question.lower()
            for kw in ["urgent", "critical", "clarif", "confirm", "mandatory", "required"]
        ) else "medium"

        rfi = MeetingRFI(
            rfi_id=f"MTG-{i+1:03d}",
            question=qa.question,
            source="meeting",
            trade=trade,
            priority=priority,
            context_snippet=(qa.source_text or qa.question)[:200],
        )
        new_rfis.append(rfi)
        existing_questions.append(qa.question)   # prevent self-dedup in same batch

    return new_rfis


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def process_meeting(
    transcript_or_path: str,
    job_id: str,
    llm_client: Any = None,
) -> dict:
    """
    Process a meeting transcript or audio file.

    Parameters
    ----------
    transcript_or_path : str
        Either plain transcript text, or a file path ending in
        .mp3/.wav/.m4a/.webm/.ogg/.mp4 for audio transcription.
    job_id : str
        Job to attach new RFIs to.  Existing RFIs are fetched from the job
        payload to enable deduplication.
    llm_client : optional
        OpenAI-compatible client for LLM Q&A extraction and Whisper.

    Returns
    -------
    dict with keys: transcript, qa_pairs, new_rfis, duplicate_count, total_qa
    """
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".mp4", ".flac"}
    import pathlib

    # Determine if input is a file path
    transcript = transcript_or_path
    if pathlib.Path(transcript_or_path).suffix.lower() in AUDIO_EXTENSIONS:
        transcript = transcribe_audio(transcript_or_path, llm_client)

    # Load existing RFIs from job
    existing_rfis: List[dict] = []
    try:
        from src.api.job_store import get_job
        job = get_job(job_id)
        if job:
            payload = job.get("payload") or job.get("result") or {}
            existing_rfis = payload.get("rfis", [])
    except Exception as exc:
        logger.debug("process_meeting: could not load job %s: %s", job_id, exc)

    qa_pairs = extract_qa_pairs(transcript, llm_client)
    new_rfis = qa_to_rfis(qa_pairs, existing_rfis, {})

    duplicate_count = len(qa_pairs) - len(new_rfis)

    # Write new RFIs back to job payload
    if new_rfis and job_id:
        try:
            from src.api.job_store import get_job, update_job
            job = get_job(job_id)
            if job:
                payload = job.get("payload") or job.get("result") or {}
                existing_rfis_list = payload.get("rfis", [])
                existing_rfis_list.extend([r.to_dict() for r in new_rfis])
                payload["rfis"] = existing_rfis_list
                if "payload" in job:
                    job["payload"] = payload
                else:
                    job["result"] = payload
                update_job(job_id, job)
        except Exception as exc:
            logger.debug("process_meeting: could not save RFIs for job %s: %s", job_id, exc)

    return {
        "transcript": transcript,
        "qa_pairs": [q.to_dict() for q in qa_pairs],
        "new_rfis": [r.to_dict() for r in new_rfis],
        "duplicate_count": duplicate_count,
        "total_qa": len(qa_pairs),
    }
