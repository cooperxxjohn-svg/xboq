"""
Email Sender — SMTP and SendGrid transport for xBOQ notifications.

Supports three transport modes (resolved in order):
    1. SendGrid  — if SENDGRID_API_KEY env var is set
    2. SMTP      — if SMTP_HOST is set (uses starttls by default)
    3. Log-only  — no real sending; logs email body for development

Usage:
    from src.notifications.email_sender import send_email, EmailMessage

    msg = EmailMessage(
        to=["engineer@contractor.com"],
        subject="RFI Batch — Sonipat Hospital Tender",
        body="Dear Team,\n\nPlease find the RFIs below…",
    )
    result = send_email(msg)
    print(result.success, result.transport_used)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EmailMessage:
    """An email to be sent."""
    to: List[str]
    subject: str
    body: str
    from_addr: str = ""           # defaults to XBOQ_EMAIL_FROM env var
    reply_to: str = ""
    cc: List[str] = field(default_factory=list)
    html_body: str = ""           # optional HTML version


@dataclass
class SendResult:
    """Outcome of a send_email() call."""
    success: bool
    transport_used: str            # "sendgrid" | "smtp" | "log_only"
    error: str = ""
    message_id: str = ""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _from_addr() -> str:
    return os.environ.get("XBOQ_EMAIL_FROM", "noreply@xboq.ai")


def _default_subject_prefix() -> str:
    return os.environ.get("XBOQ_EMAIL_PREFIX", "[xBOQ]")


# ---------------------------------------------------------------------------
# SendGrid transport
# ---------------------------------------------------------------------------

def _send_via_sendgrid(msg: EmailMessage) -> SendResult:
    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, To, Content, Email
    except ImportError:
        return SendResult(False, "sendgrid", error="sendgrid package not installed")

    api_key = os.environ.get("SENDGRID_API_KEY", "")
    if not api_key:
        return SendResult(False, "sendgrid", error="SENDGRID_API_KEY not set")

    from_email = msg.from_addr or _from_addr()
    subject = f"{_default_subject_prefix()} {msg.subject}"

    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    to_list = [{"email": addr} for addr in msg.to]
    cc_list = [{"email": addr} for addr in msg.cc] if msg.cc else []

    mail_data = {
        "personalizations": [{"to": to_list}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/plain", "value": msg.body}],
    }
    if msg.html_body:
        mail_data["content"].append({"type": "text/html", "value": msg.html_body})
    if cc_list:
        mail_data["personalizations"][0]["cc"] = cc_list
    if msg.reply_to:
        mail_data["reply_to"] = {"email": msg.reply_to}

    try:
        response = sg.client.mail.send.post(request_body=mail_data)
        success = response.status_code in (200, 202)
        msg_id = response.headers.get("X-Message-Id", "")
        if not success:
            logger.warning("SendGrid returned %d", response.status_code)
        return SendResult(success=success, transport_used="sendgrid", message_id=msg_id)
    except Exception as exc:
        logger.error("SendGrid send failed: %s", exc)
        return SendResult(False, "sendgrid", error=str(exc))


# ---------------------------------------------------------------------------
# SMTP transport
# ---------------------------------------------------------------------------

def _send_via_smtp(msg: EmailMessage) -> SendResult:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    host = os.environ.get("SMTP_HOST", "")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    use_tls = os.environ.get("SMTP_TLS", "1") == "1"

    from_addr = msg.from_addr or os.environ.get("XBOQ_EMAIL_FROM", user or "noreply@xboq.ai")
    subject = f"{_default_subject_prefix()} {msg.subject}"

    mime = MIMEMultipart("alternative")
    mime["Subject"] = subject
    mime["From"] = from_addr
    mime["To"] = ", ".join(msg.to)
    if msg.cc:
        mime["Cc"] = ", ".join(msg.cc)
    if msg.reply_to:
        mime["Reply-To"] = msg.reply_to

    mime.attach(MIMEText(msg.body, "plain", "utf-8"))
    if msg.html_body:
        mime.attach(MIMEText(msg.html_body, "html", "utf-8"))

    all_recipients = list(msg.to) + list(msg.cc)

    try:
        if use_tls:
            server = smtplib.SMTP(host, port, timeout=10)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(host, port, timeout=10)

        if user and password:
            server.login(user, password)

        server.sendmail(from_addr, all_recipients, mime.as_string())
        server.quit()
        return SendResult(success=True, transport_used="smtp")
    except Exception as exc:
        logger.error("SMTP send failed: %s", exc)
        return SendResult(False, "smtp", error=str(exc))


# ---------------------------------------------------------------------------
# Log-only fallback (dev / test)
# ---------------------------------------------------------------------------

def _send_log_only(msg: EmailMessage) -> SendResult:
    logger.info(
        "EMAIL (log-only) to=%s subject=%r\n%s",
        msg.to, msg.subject, msg.body[:300],
    )
    return SendResult(success=True, transport_used="log_only")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_email(msg: EmailMessage) -> SendResult:
    """
    Send an email using the best available transport.

    Transport priority:
    1. SendGrid  (SENDGRID_API_KEY env var)
    2. SMTP      (SMTP_HOST env var)
    3. Log-only  (development / CI fallback)
    """
    if os.environ.get("SENDGRID_API_KEY"):
        result = _send_via_sendgrid(msg)
        if result.success:
            return result
        logger.warning("SendGrid failed (%s) — trying SMTP", result.error)

    if os.environ.get("SMTP_HOST"):
        result = _send_via_smtp(msg)
        if result.success:
            return result
        logger.warning("SMTP failed (%s) — falling back to log-only", result.error)

    return _send_log_only(msg)


def send_rfi_batch(
    rfis: List[dict],
    to: List[str],
    project_name: str = "",
    from_addr: str = "",
    include_drafts: bool = False,
) -> SendResult:
    """
    Convenience wrapper: generates RFI email drafts and sends them.

    Parameters
    ----------
    rfis : list
        RFI dicts from the pipeline payload.
    to : list
        Recipient email addresses.
    project_name : str
        Used in the email subject line.
    from_addr : str
        Sender address (defaults to XBOQ_EMAIL_FROM env).
    include_drafts : bool
        If True, include draft (unapproved) RFIs.

    Returns
    -------
    SendResult
    """
    try:
        from src.exports.email_drafts import generate_rfi_email_drafts
        drafts = generate_rfi_email_drafts(rfis, include_drafts=include_drafts)
    except Exception as exc:
        logger.error("generate_rfi_email_drafts failed: %s", exc)
        drafts = {}

    if not drafts:
        # No approved RFIs to send
        return SendResult(False, "none", error="No approved RFIs to email")

    # Use the combined "all trades" draft; fall back to first available
    body = drafts.get("rfi_email_all.txt") or next(iter(drafts.values()))

    subject = f"RFI Queries — {project_name}" if project_name else "RFI Queries"

    msg = EmailMessage(
        to=to,
        subject=subject,
        body=body,
        from_addr=from_addr,
    )
    return send_email(msg)
