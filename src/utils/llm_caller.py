"""
LLM call wrapper with automatic retry and exponential backoff.

Handles transient failures from Anthropic and OpenAI:
  - 429 RateLimitError   — exponential backoff, up to MAX_RETRIES attempts
  - 500 / 503 API errors — same backoff
  - Network errors       — same backoff

Usage
-----
    from src.utils.llm_caller import call_llm

    text = call_llm(
        client=my_anthropic_or_openai_client,
        system="You are a construction estimator.",
        user="List the key risks in this BOQ.",
        model="claude-3-5-haiku-20241022",
        max_tokens=500,
    )

The function auto-detects whether `client` is Anthropic or OpenAI.
Returns the response text string on success; raises the final exception
after all retries are exhausted.

Environment overrides
---------------------
  XBOQ_LLM_MAX_RETRIES    — integer, default 4
  XBOQ_LLM_BASE_DELAY_S   — float, base backoff delay in seconds, default 1.0
  XBOQ_LLM_MAX_DELAY_S    — float, cap for backoff delay, default 60.0
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_MAX_RETRIES:   int   = int(os.environ.get("XBOQ_LLM_MAX_RETRIES", "4"))
_BASE_DELAY:    float = float(os.environ.get("XBOQ_LLM_BASE_DELAY_S", "1.0"))
_MAX_DELAY:     float = float(os.environ.get("XBOQ_LLM_MAX_DELAY_S", "60.0"))

# HTTP status codes / error names that indicate a transient failure worth retrying
_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is likely transient and worth retrying."""
    name = type(exc).__name__
    # Anthropic: RateLimitError, InternalServerError, APIConnectionError
    # OpenAI:    RateLimitError, APIConnectionError, APIStatusError
    if any(s in name for s in ("RateLimit", "ServerError", "APIConnection", "Timeout")):
        return True
    # Check for status code attribute (present on most SDK error classes)
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status and int(status) in _RETRYABLE_STATUS:
        return True
    # Also retry on generic ConnectionError / TimeoutError from urllib3 / httpx
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def call_llm(
    client,
    system: str,
    user: str,
    model: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    openai_model: str = "gpt-4o-mini",
    anthropic_model: str = "claude-3-5-haiku-20241022",
) -> str:
    """
    Call the LLM and return the response text.

    Auto-detects Anthropic vs OpenAI client. Retries on transient errors
    with exponential backoff + jitter.

    Parameters
    ----------
    client          Anthropic or OpenAI client instance.
    system          System prompt string.
    user            User message string.
    model           Explicit model ID. If empty, uses openai_model / anthropic_model
                    depending on client type.
    max_tokens      Maximum output tokens.
    temperature     Sampling temperature.
    openai_model    Default model for OpenAI clients.
    anthropic_model Default model for Anthropic clients.
    """
    import random

    last_exc: Exception = RuntimeError("No LLM call attempted")

    for attempt in range(_MAX_RETRIES + 1):
        try:
            if hasattr(client, "messages"):
                # Anthropic
                _model = model or anthropic_model
                resp = client.messages.create(
                    model=_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text

            elif hasattr(client, "chat"):
                # OpenAI
                _model = model or openai_model
                resp = client.chat.completions.create(
                    model=_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            else:
                raise ValueError(f"Unknown LLM client type: {type(client)}")

        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                logger.warning("LLM call failed (non-retryable %s): %s", type(exc).__name__, exc)
                raise

            if attempt >= _MAX_RETRIES:
                logger.error(
                    "LLM call failed after %d retries (%s): %s",
                    _MAX_RETRIES, type(exc).__name__, exc
                )
                raise

            # Exponential backoff with full jitter
            delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
            jitter = random.uniform(0, delay * 0.2)
            sleep_s = delay + jitter
            logger.warning(
                "LLM transient error (attempt %d/%d, sleeping %.1fs): %s: %s",
                attempt + 1, _MAX_RETRIES + 1, sleep_s, type(exc).__name__, exc
            )
            time.sleep(sleep_s)

    raise last_exc
