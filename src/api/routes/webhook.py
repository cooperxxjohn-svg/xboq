"""
WhatsApp / Telegram bot webhook endpoints.

Both platforms POST incoming messages to a registered webhook URL.
This module handles both, routes the question through BidAgent, and
sends a reply via the respective platform API.

Environment variables:
    TELEGRAM_BOT_TOKEN      — from @BotFather
    TWILIO_ACCOUNT_SID      — Twilio account SID (for WhatsApp via Twilio)
    TWILIO_AUTH_TOKEN       — Twilio auth token
    TWILIO_WHATSAPP_FROM    — "whatsapp:+14155238886" (Twilio sandbox number)
    XBOQ_DEFAULT_JOB_ID     — fallback job ID when no context set per chat
    WEBHOOK_SECRET          — optional shared secret for Telegram verification

Telegram setup:
    POST https://api.telegram.org/bot{TOKEN}/setWebhook
         ?url=https://your-domain.com/api/webhook/telegram

WhatsApp (Twilio) setup:
    In Twilio console → Messaging → WhatsApp → Sandbox:
        When a message comes in: POST https://your-domain.com/api/webhook/whatsapp

Supported commands (both platforms):
    /set_job <job_id>       — bind this chat to a specific analysis job
    /help                   — show available commands
    <any text>              — treated as a question answered by BidAgent
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["webhooks"])

# ---------------------------------------------------------------------------
# Per-chat state (in-memory; replace with Redis/DB for multi-process)
# ---------------------------------------------------------------------------

# Maps chat_id (str) → job_id (str)
_chat_job_map: Dict[str, str] = {}


def _get_job_id(chat_id: str) -> Optional[str]:
    return _chat_job_map.get(chat_id) or os.environ.get("XBOQ_DEFAULT_JOB_ID")


def _set_job_id(chat_id: str, job_id: str) -> None:
    _chat_job_map[chat_id] = job_id


# ---------------------------------------------------------------------------
# BidAgent routing
# ---------------------------------------------------------------------------

def _ask_agent(question: str, job_id: str) -> str:
    """Route question through BidAgent; return answer text."""
    try:
        from src.api.job_store import job_store
        job = job_store.get_job(job_id)
        if job is None or job.status != "complete" or not job.payload:
            return f"Job '{job_id}' is not ready. Upload a tender first."

        from src.bidagent.agent import BidAgent
        from src.reasoning.bid_synthesizer import BidSynthesis

        payload = job.payload
        synthesis = BidSynthesis()  # minimal — agent handles missing fields gracefully
        agent = BidAgent(
            payload=payload,
            synthesis=synthesis,
            store=None,
            embedder=None,
            llm_client=None,   # LLM disabled for bot (keeps costs predictable)
        )
        answer = agent.ask(question)
        text = answer.answer or "No answer available."
        if answer.suggested_follow_ups:
            text += "\n\n_Try asking:_\n" + "\n".join(
                f"• {q}" for q in answer.suggested_follow_ups[:3]
            )
        return text
    except ImportError:
        return "BidAgent module unavailable."
    except Exception as exc:
        logger.exception("BidAgent failed: %s", exc)
        return f"Could not process question: {exc}"


def _handle_message(chat_id: str, text: str) -> str:
    """Process incoming message; return reply text."""
    text = (text or "").strip()

    if text.lower() in ("/start", "/help"):
        return (
            "xBOQ Bid Assistant\n\n"
            "Commands:\n"
            "  /set_job <job_id>  — bind this chat to a tender analysis\n"
            "  /status            — show current job status\n"
            "  <question>         — ask anything about your tender\n\n"
            "Example: _What are the top 3 RFIs?_"
        )

    if text.lower().startswith("/set_job "):
        job_id = text.split(maxsplit=1)[1].strip()
        _set_job_id(chat_id, job_id)
        return f"Bound to job: {job_id}\nNow ask me anything about this tender."

    if text.lower() == "/status":
        job_id = _get_job_id(chat_id)
        if not job_id:
            return "No job set. Use /set_job <job_id> first."
        try:
            from src.api.job_store import job_store
            job = job_store.get_job(job_id)
            if job is None:
                return f"Job '{job_id}' not found."
            return (
                f"Job: {job_id}\n"
                f"Status: {job.status}\n"
                f"Progress: {int(job.progress * 100)}%\n"
                f"{job.progress_message}"
            )
        except Exception as exc:
            return f"Status check failed: {exc}"

    job_id = _get_job_id(chat_id)
    if not job_id:
        return (
            "No job set. Use:\n  /set_job <job_id>\n"
            "to bind this chat to an analysis."
        )

    return _ask_agent(text, job_id)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

async def _send_telegram(chat_id: str, text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json=payload)
    except Exception as exc:
        logger.warning("Telegram sendMessage failed: %s", exc)


@router.post("/api/webhook/telegram")
async def telegram_webhook(request: Request) -> Response:
    """Handle incoming Telegram updates."""
    try:
        update: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Verify optional secret (set in Telegram webhook URL as ?secret=...)
    secret = os.environ.get("WEBHOOK_SECRET", "")
    if secret:
        provided = request.query_params.get("secret", "")
        if provided != secret:
            raise HTTPException(status_code=403, detail="Invalid secret")

    msg = update.get("message") or update.get("edited_message", {})
    if not msg:
        return Response(status_code=200)

    chat_id = str(msg.get("chat", {}).get("id", ""))
    text = msg.get("text", "")

    if not chat_id or not text:
        return Response(status_code=200)

    reply = _handle_message(chat_id, text)
    await _send_telegram(chat_id, reply)
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# WhatsApp (Twilio)
# ---------------------------------------------------------------------------

def _send_whatsapp(to: str, body: str) -> None:
    """Send WhatsApp reply via Twilio REST API."""
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token  = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    if not account_sid or not auth_token:
        logger.debug("Twilio not configured — WhatsApp reply dropped")
        return
    try:
        from twilio.rest import Client as TwilioClient
        client = TwilioClient(account_sid, auth_token)
        client.messages.create(body=body, from_=from_number, to=to)
    except ImportError:
        logger.warning("twilio package not installed")
    except Exception as exc:
        logger.warning("Twilio sendMessage failed: %s", exc)


@router.post("/api/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> JSONResponse:
    """Handle incoming WhatsApp messages via Twilio."""
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid form data")

    from_number = str(form.get("From", ""))        # e.g. "whatsapp:+919999888877"
    body        = str(form.get("Body", "")).strip()
    chat_id     = from_number.replace("whatsapp:", "")

    if not from_number or not body:
        return JSONResponse(content={"ok": True})

    reply = _handle_message(chat_id, body)
    _send_whatsapp(from_number, reply)
    return JSONResponse(content={"ok": True, "reply_length": len(reply)})


# ---------------------------------------------------------------------------
# Internal: inspect chat state (testing / debugging)
# ---------------------------------------------------------------------------

@router.get("/api/webhook/state/{chat_id}")
def get_chat_state(chat_id: str) -> JSONResponse:
    """Return current job binding for a chat ID (debugging endpoint)."""
    return JSONResponse(content={
        "chat_id": chat_id,
        "job_id":  _get_job_id(chat_id),
    })
