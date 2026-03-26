"""
POST /api/analyze — upload files and start analysis pipeline.

Security / reliability improvements (P0 pre-pilot):
  - P0-1: Job output written to ~/.xboq/job_outputs/{job_id}/ (persistent)
  - P0-2: Pipeline runs in ThreadPoolExecutor(max_workers=3) — no unbounded threads
  - P0-3: File upload hardening — basename-only filenames, extension whitelist,
           per-request 500 MB size cap
  - P0-7: check_quota / increment_usage wired for tenant plan enforcement
"""

import os
import sys
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.job_store import job_store, set_job as _set_flat_job
from src.api.job_db import OUTPUTS_DIR
from src.api.audit_log import audit, AuditEvent
from src.api.analytics import log_event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])

# ---------------------------------------------------------------------------
# Job queue — Celery preferred, ThreadPoolExecutor fallback
#
# Celery (Redis-backed):
#   - Jobs survive API/worker restarts (tasks re-queued on worker crash)
#   - Multiple workers can process in parallel across machines
#   - Queue depth visible in Flower UI (:5555) and /api/metrics
#
# ThreadPoolExecutor fallback (no Redis):
#   - Used in dev/test when Redis is not running
#   - CELERY_ALWAYS_EAGER=true makes Celery run inline (used in tests)
# ---------------------------------------------------------------------------

try:
    from src.api.worker import celery_app, CELERY_AVAILABLE
    from src.api.tasks import run_pipeline_task, run_pipeline_fast_task, run_pipeline_full_task
    _USE_CELERY = CELERY_AVAILABLE
except ImportError:
    _USE_CELERY = False
    celery_app = None
    run_pipeline_task = None

# ThreadPoolExecutor — fallback when Celery unavailable
_pipeline_executor = ThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="pipeline",
)

# ---------------------------------------------------------------------------
# In-process rate limiter — sliding window per IP
#
# Limits /api/analyze to _RATE_LIMIT_MAX submissions per _RATE_LIMIT_WINDOW_S
# seconds per client IP. Uses a simple deque-per-IP approach with no external
# dependencies. Resets on server restart (acceptable for pilot scale).
# ---------------------------------------------------------------------------

import collections
import threading

_RATE_LIMIT_MAX      = int(os.environ.get("XBOQ_RATE_LIMIT_ANALYZE", "10"))
_RATE_LIMIT_WINDOW_S = 60   # sliding window in seconds

_rate_lock: threading.Lock = threading.Lock()
_rate_buckets: dict = {}    # ip → collections.deque of timestamps


def _check_rate_limit(ip: str) -> None:
    """
    Raise HTTP 429 if `ip` has exceeded _RATE_LIMIT_MAX requests in the last
    _RATE_LIMIT_WINDOW_S seconds. Thread-safe.
    """
    import time
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(ip, collections.deque())
        # Evict timestamps outside the window
        while bucket and bucket[0] < now - _RATE_LIMIT_WINDOW_S:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: max {_RATE_LIMIT_MAX} analyses per "
                    f"{_RATE_LIMIT_WINDOW_S}s per IP. Try again shortly."
                ),
            )
        bucket.append(now)


# ---------------------------------------------------------------------------
# Upload security constants
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".dxf"}
_ALLOWED_ZIP_EXTENSIONS = {".zip"}
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB per upload file
_MAX_TOTAL_BYTES  = 1024 * 1024 * 1024  # 1 GB total per request
_MAX_ZIP_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB extracted (zip bomb guard)
_MAX_FILES_PER_REQUEST = 50             # prevent resource exhaustion via many tiny files

# Magic-byte signatures for supported file types.
# We check the first bytes of the file content to reject files that have been
# renamed to a different extension (e.g. .exe renamed to .pdf).
#
# References:
#   PDF   — b'%PDF'
#   ZIP   — b'PK\x03\x04' (PKZIP local file header)
#   XLSX/XLS/DXF are all ZIP-based or text-based — we use relaxed validation
_MAGIC_BYTES: dict = {
    ".pdf":  [(0, b"%PDF")],
    ".zip":  [(0, b"PK\x03\x04")],
    ".xlsx": [(0, b"PK\x03\x04")],      # XLSX is a ZIP container
    ".xls":  [(0, b"\xd0\xcf\x11\xe0"), # OLE2 (legacy .xls)
              (0, b"PK\x03\x04")],       # Some XLS saved as XLSX
    ".dxf":  [],                          # DXF is plain text — skip magic check
}


def _check_magic_bytes(data: bytes, suffix: str, filename: str) -> None:
    """
    Validate that file bytes match the expected magic signature.
    Raises HTTPException(400) if the content doesn't match the extension.
    """
    checks = _MAGIC_BYTES.get(suffix, [])
    if not checks:
        return  # no magic check for this type (e.g. DXF)
    for offset, magic in checks:
        if data[offset: offset + len(magic)] == magic:
            return  # at least one signature matched
    raise HTTPException(
        status_code=400,
        detail=(
            f"File '{filename}' content does not match its extension '{suffix}'. "
            "Please upload a genuine file."
        ),
    )

# ---------------------------------------------------------------------------
# Defensive imports
# ---------------------------------------------------------------------------

try:
    from src.analysis.pipeline import run_analysis_pipeline, generate_project_id
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logger.warning("src.analysis.pipeline not importable — pipeline will fail gracefully")

try:
    from src.export import export_boq_excel, export_pdf_summary, export_rfi_word
    EXPORTS_AVAILABLE = True
except ImportError:
    EXPORTS_AVAILABLE = False
    logger.warning("src.export not importable — export generation disabled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(raw: str) -> str:
    """
    Strip directory components and reject disallowed extensions.

    Returns a safe filename (basename only).
    Raises HTTPException(400) if extension is not in _ALLOWED_EXTENSIONS.
    """
    name = Path(raw).name  # basename only — strips any ../ traversal
    if not name:
        name = "upload.pdf"
    suffix = Path(name).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{name}' has unsupported extension '{suffix}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))} or .zip"
            ),
        )
    return name


def _extract_zip(zip_bytes: bytes, dest_dir: Path) -> List[Path]:
    """
    Extract a ZIP archive to dest_dir, returning paths of all accepted files.

    Security hardening:
    - Rejects path traversal (e.g. ../../etc/passwd)
    - Rejects entries with disallowed extensions
    - Enforces total extracted size limit (_MAX_ZIP_EXTRACTED_BYTES)
    - Skips macOS metadata files (__MACOSX, .DS_Store)
    """
    import zipfile, io

    extracted: List[Path] = []
    total_extracted = 0

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded .zip file is corrupted or not a valid ZIP archive")

    for entry in zf.infolist():
        # Skip macOS metadata and hidden entries
        name = entry.filename
        if "__MACOSX" in name or name.startswith(".") or name.endswith(".DS_Store"):
            continue
        # Skip directories
        if name.endswith("/"):
            continue
        # Strip directory components — use only basename
        safe_name = Path(name).name
        if not safe_name:
            continue
        # Path traversal guard — ensure basename didn't come from traversal
        suffix = Path(safe_name).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            logger.debug("Skipping ZIP entry with disallowed extension: %s", safe_name)
            continue
        # Zip bomb guard
        total_extracted += entry.file_size
        if total_extracted > _MAX_ZIP_EXTRACTED_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"ZIP contents exceed the {_MAX_ZIP_EXTRACTED_BYTES // (1024**3)} GB extraction limit",
            )
        # Write to dest
        data = zf.read(entry.filename)
        dest = dest_dir / safe_name
        # Handle duplicate filenames within the ZIP
        if dest.exists():
            stem = dest.stem
            suffix_str = dest.suffix
            dest = dest_dir / f"{stem}_{len(extracted)}{suffix_str}"
        dest.write_bytes(data)
        extracted.append(dest)
        logger.debug("Extracted from ZIP: %s (%d bytes)", safe_name, len(data))

    zf.close()
    return extracted


def _build_llm_client():
    """
    Build LLM client from environment — Claude preferred, OpenAI fallback.

    Returns None when XBOQ_OFFLINE_MODE=true (no external calls allowed).
    """
    import os
    if os.environ.get("XBOQ_OFFLINE_MODE", "").lower() in ("1", "true", "yes"):
        logger.info("XBOQ_OFFLINE_MODE=true — skipping LLM client (no external API calls)")
        return None
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    if anthropic_key:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=anthropic_key)
        except ImportError:
            pass
    if openai_key:
        try:
            import openai
            return openai.OpenAI(api_key=openai_key)
        except ImportError:
            pass
    return None


def _queue_depth() -> int:
    """Approximate number of jobs waiting in the executor queue."""
    try:
        return _pipeline_executor._work_queue.qsize()  # type: ignore[attr-defined]
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Background pipeline worker
# ---------------------------------------------------------------------------

def _run_pipeline_in_thread(
    job_id: str,
    input_paths: List[Path],
    excel_paths: List[Path],
    run_mode: str,
    project_name: str,
    job_dir: Path,
    org_id: str = "local",
) -> None:
    """Execute the analysis pipeline and persist results to disk + SQLite."""
    try:
        job_store.update_job(
            job_id,
            status="processing",
            progress=0.05,
            progress_message="Starting pipeline",
            queue_position=0,
        )

        if not PIPELINE_AVAILABLE:
            raise RuntimeError("Analysis pipeline is not available (import failed)")

        project_id = generate_project_id()
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        def progress_callback(stage_id: str, message: str, pct: float) -> None:
            # Honour cancellation — raise to abort the pipeline
            current = job_store.get_job(job_id)
            if current and current.status == "cancelled":
                raise InterruptedError(f"Job {job_id} was cancelled")
            job_store.update_job(
                job_id,
                progress=min(float(pct), 0.95),
                progress_message=message,
            )

        result = run_analysis_pipeline(
            input_files=input_paths,
            project_id=project_id,
            output_dir=output_dir,
            progress_callback=progress_callback,
            run_mode=run_mode if run_mode else "demo_fast",
            boq_excel_paths=excel_paths if excel_paths else None,
            llm_client=_build_llm_client(),
            tenant_id=org_id,
        )

        errors: List[str] = []
        if result.error_message:
            errors.append(result.error_message)

        output_files: dict = {}
        if result.success and EXPORTS_AVAILABLE and result.payload:
            _try_generate_exports(result.payload, output_dir, output_files)

        if result.success:
            # Sanitize NaN/Inf before storing — numpy rate lookups can produce NaN
            _clean_payload = _sanitize_nan(result.payload) if result.payload else result.payload
            # Persist payload to disk via job_db, update metadata in SQLite
            job_store.update_job(
                job_id,
                status="complete",
                completed_at=datetime.now(timezone.utc),
                payload=_clean_payload,
                errors=errors,
                output_files=output_files,
                progress=1.0,
                progress_message="Done",
            )
            # Populate in-memory flat store for line-items routes (thread-safe)
            _set_flat_job(job_id, {
                "job_id": job_id,
                "status": "complete",
                "result": _clean_payload or {},
            })

            # Metrics: record completion
            try:
                from src.api.routes.metrics import record_job_complete
                job_rec = job_store.get_job(job_id)
                if job_rec and job_rec.created_at:
                    elapsed = (datetime.now(timezone.utc) - job_rec.created_at).total_seconds()
                    record_job_complete(elapsed)
                else:
                    record_job_complete()
            except Exception:
                pass

            # Quota: count this as a successful run
            try:
                from src.auth.tenant_manager import increment_usage
                increment_usage(org_id)
            except Exception as _qe:
                logger.debug("increment_usage failed for %s: %s", org_id, _qe)

            # Analytics
            log_event("analysis_complete", job_id=job_id, org_id=org_id, run_mode=run_mode)

            # Audit: job completed
            audit(AuditEvent.JOB_COMPLETED, None,
                  resource_type="job", resource_id=job_id,
                  detail={"org_id": org_id, "run_mode": run_mode,
                          "export_count": len(output_files)})

            # Outbound completion webhook + Slack notification (non-blocking, best-effort)
            _fire_completion_webhook(job_id, org_id, run_mode, result.payload or {})
            _fire_slack_notification(job_id, org_id, run_mode, result.payload or {})

        else:
            job_store.update_job(
                job_id,
                status="error",
                completed_at=datetime.now(timezone.utc),
                errors=errors or ["Pipeline failed with no error message"],
                output_files=output_files,
                progress=0.0,
                progress_message="Pipeline failed",
            )
            # Analytics
            log_event("analysis_failed", job_id=job_id, org_id=org_id, run_mode=run_mode,
                      extra={"errors": errors[:3]})

            # Audit: job failed
            audit(AuditEvent.JOB_FAILED, None,
                  resource_type="job", resource_id=job_id,
                  detail={"org_id": org_id, "errors": errors[:3]})

    except InterruptedError:
        # Raised by progress_callback when status is set to "cancelled"
        # by DELETE /api/jobs/{job_id}. Job already has status="cancelled"
        # in the DB — just log and exit cleanly.
        logger.info("Pipeline thread exiting cleanly: job %s was cancelled", job_id)

    except Exception as exc:
        _handle_pipeline_failure(exc, job_id, org_id)

    except BaseException as exc:
        # Catches BaseExceptionGroup (Python 3.11+ anyio task groups) and other
        # BaseException subclasses that bypass `except Exception`.
        # Always re-raise SystemExit / KeyboardInterrupt.
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        _handle_pipeline_failure(exc, job_id, org_id)


def _handle_pipeline_failure(exc: BaseException, job_id: str, org_id: str) -> None:
    """Mark a job as failed and emit analytics/audit. Called from both except branches."""
    logger.exception("Pipeline thread failed for job %s: %s", job_id, exc)
    # Unwrap ExceptionGroup to get a readable message
    msg = str(exc)
    if hasattr(exc, "exceptions"):  # ExceptionGroup / BaseExceptionGroup
        inner = [str(e) for e in exc.exceptions]  # type: ignore[attr-defined]
        msg = " | ".join(inner) or msg

    job_store.update_job(
        job_id,
        status="error",
        completed_at=datetime.now(timezone.utc),
        errors=[msg],
        progress=0.0,
        progress_message=f"Error: {msg[:200]}",
    )
    log_event("analysis_failed", job_id=job_id, org_id=org_id,
              extra={"error": msg[:500]})
    audit(AuditEvent.JOB_FAILED, None,
          resource_type="job", resource_id=job_id,
          detail={"org_id": org_id, "error": msg[:500]})


def _sanitize_nan(obj):
    """Recursively replace NaN/Inf floats with None. Prevents JSON serialization errors."""
    import math as _math
    if isinstance(obj, float):
        return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _fire_completion_webhook(job_id: str, org_id: str, run_mode: str, payload: dict) -> None:
    """
    POST a completion notification to XBOQ_COMPLETION_WEBHOOK_URL if configured.

    Fires in the calling thread (already a background thread) — non-fatal on error.
    The webhook body is a minimal job summary; NOT the full payload (may be large).

    Set XBOQ_COMPLETION_WEBHOOK_URL=https://... in environment to enable.
    Optional XBOQ_COMPLETION_WEBHOOK_SECRET adds an X-Webhook-Secret header.
    """
    url = os.environ.get("XBOQ_COMPLETION_WEBHOOK_URL", "").strip()
    if not url:
        return

    import json as _json
    import urllib.request

    p = payload
    body = {
        "event":        "job.complete",
        "job_id":       job_id,
        "org_id":       org_id,
        "run_mode":     run_mode,
        "boq_items":    len(p.get("boq_items", [])),
        "rfis":         len(p.get("rfis", [])),
        "blockers":     len(p.get("blockers", [])),
        "qa_score":     p.get("qa_score", {}).get("overall_score") if isinstance(p.get("qa_score"), dict) else p.get("qa_score"),
        "result_url":   f"/api/jobs/{job_id}",
    }

    try:
        data = _json.dumps(body).encode("utf-8")
        headers: dict = {"Content-Type": "application/json", "User-Agent": "xboq-webhook/1.0"}
        secret = os.environ.get("XBOQ_COMPLETION_WEBHOOK_SECRET", "")
        if secret:
            headers["X-Webhook-Secret"] = secret

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Completion webhook delivered to %s (status %d)", url, resp.status)
    except Exception as exc:
        logger.warning("Completion webhook failed (non-fatal): %s", exc)


def _fire_slack_notification(job_id: str, org_id: str, run_mode: str, payload: dict) -> None:
    """
    Post a Slack message when a job completes.

    Set XBOQ_SLACK_WEBHOOK_URL to a Slack Incoming Webhook URL to enable.
    Optional XBOQ_SLACK_CHANNEL overrides the webhook's default channel.

    Message format: Slack Block Kit with summary stats.
    Non-fatal on error.
    """
    url = os.environ.get("XBOQ_SLACK_WEBHOOK_URL", "").strip()
    if not url:
        return

    import json as _json
    import urllib.request

    p = payload
    boq_items = len(p.get("boq_items", []))
    rfis      = len(p.get("rfis", []))
    blockers  = len(p.get("blockers", []))
    qa        = p.get("qa_score", {})
    qa_score  = qa.get("overall_score") if isinstance(qa, dict) else qa
    channel   = os.environ.get("XBOQ_SLACK_CHANNEL", "")

    text = (
        f":white_check_mark: *xBOQ job complete* | `{job_id}`\n"
        f"Org: `{org_id}` | Mode: `{run_mode}`\n"
        f"BOQ items: *{boq_items}* | RFIs: *{rfis}* | Blockers: *{blockers}*"
        + (f" | QA: *{qa_score}*" if qa_score is not None else "")
    )

    body: dict = {"text": text}
    if channel:
        body["channel"] = channel

    try:
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        logger.info("Slack notification sent for job %s", job_id)
    except Exception as exc:
        logger.warning("Slack notification failed (non-fatal): %s", exc)


def _try_generate_exports(payload: dict, output_dir: Path, output_files: dict) -> None:
    """Attempt to generate Excel / PDF / Word exports; silently skip on failure."""
    if not EXPORTS_AVAILABLE:
        return

    try:
        excel_path = output_dir / "boq.xlsx"
        export_boq_excel(payload, excel_path)
        if excel_path.exists():
            output_files["excel"] = str(excel_path)
    except Exception as exc:
        logger.warning("Excel export failed: %s", exc)

    try:
        from src.exports.xlsx_boq_export import export_boq_xlsx
        xlsx_path = output_dir / "boq.xlsx"
        export_boq_xlsx(payload, xlsx_path)
        if xlsx_path.exists():
            output_files["xlsx"] = str(xlsx_path)
    except Exception as exc:
        logger.warning("xlsx export failed: %s", exc)

    try:
        pdf_path = output_dir / "summary.pdf"
        export_pdf_summary(payload, pdf_path)
        if pdf_path.exists():
            output_files["pdf"] = str(pdf_path)
    except Exception as exc:
        logger.warning("PDF export failed: %s", exc)

    try:
        word_path = output_dir / "rfis.docx"
        export_rfi_word(payload, word_path)
        if word_path.exists():
            output_files["word"] = str(word_path)
    except Exception as exc:
        logger.warning("Word export failed: %s", exc)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/api/analyze",
    status_code=202,
    summary="Start analysis pipeline",
    response_description="202 Accepted with job_id; poll GET /api/jobs/{job_id} for progress",
)
async def analyze(
    request: Request,
    files: List[UploadFile] = File(...),
    excel_files: Optional[List[UploadFile]] = File(default=None),
    run_mode: str = Form(default="demo_fast"),
    project_name: str = Form(default=""),
    tenant_id: Optional[str] = Form(default=None),
) -> JSONResponse:
    """
    Upload PDF / Excel / DXF files and start analysis pipeline.

    Returns 202 Accepted with job_id for polling.
    Enforces tenant quota before accepting the job.
    """
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")

    # Resolve org_id from form parameter → tenant context → fallback "local"
    org_id = tenant_id or "local"
    from src.api.middleware.tenant_auth import get_tenant_context, TenantContext
    ctx: TenantContext = TenantContext(org_id=org_id, role="viewer", plan="free", authenticated=False)
    try:
        ctx = await get_tenant_context(request)
        if ctx.authenticated:
            org_id = ctx.org_id
    except Exception:
        pass  # non-fatal — use form-supplied or "local"

    # ── Role check — must be editor+ to submit analysis (P0-5) ──────────
    try:
        from src.auth.rbac import require_role
        require_role(ctx, "editor")
    except HTTPException:
        raise
    except Exception:
        pass  # non-fatal in dev mode

    # ── Quota check (P0-7) ────────────────────────────────────────────────
    try:
        from src.auth.tenant_manager import check_quota, QuotaExceededError
        check_quota(org_id)
    except Exception as qe:
        # Import of QuotaExceededError for isinstance check
        try:
            from src.auth.tenant_manager import QuotaExceededError as _QE
            if isinstance(qe, _QE):
                raise HTTPException(
                    status_code=402,
                    detail=str(qe),
                )
        except ImportError:
            pass
        # Other errors (tenant not registered etc.) — allow through in dev mode
        logger.debug("quota check skipped for %s: %s", org_id, qe)

    # ── Per-IP rate limit ─────────────────────────────────────────────────
    _client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(_client_ip)

    # ── Generate job ID + persistent directory ────────────────────────────
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_dir = job_store.job_output_dir(job_id)

    # ── Save uploaded PDF/DXF/ZIP files (P0-3: security hardened) ───────
    input_paths: List[Path] = []
    total_bytes = 0

    if len(files) > _MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: maximum {_MAX_FILES_PER_REQUEST} files per request, got {len(files)}",
        )

    for upload in files:
        raw_name = upload.filename or f"file_{len(input_paths)}.pdf"
        suffix = Path(raw_name).suffix.lower()
        contents = await upload.read()
        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{Path(raw_name).name}' exceeds the 500 MB per-file limit",
            )
        total_bytes += len(contents)
        if total_bytes > _MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Total upload size exceeds the 1 GB per-request limit",
            )
        if suffix == ".zip":
            # Validate magic bytes before extracting
            _check_magic_bytes(contents, ".zip", Path(raw_name).name)
            # Extract ZIP — adds all accepted files directly to input_paths
            extracted = _extract_zip(contents, job_dir)
            if not extracted:
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP archive '{Path(raw_name).name}' contained no supported files (.pdf, .xlsx, .xls, .dxf)",
                )
            logger.info("Extracted %d files from ZIP '%s'", len(extracted), Path(raw_name).name)
            input_paths.extend(extracted)
        else:
            safe_name = _safe_filename(raw_name)
            # Validate magic bytes before writing to disk
            _check_magic_bytes(contents, suffix, safe_name)
            dest = job_dir / safe_name
            dest.write_bytes(contents)
            input_paths.append(dest)

    # ── Save uploaded Excel files (optional, P0-3 hardened) ──────────────
    excel_paths: List[Path] = []
    if excel_files:
        for upload in excel_files:
            if upload and upload.filename:
                safe_name = _safe_filename(upload.filename)
                contents = await upload.read()
                if len(contents) > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Excel file '{safe_name}' exceeds the 500 MB limit",
                    )
                dest = job_dir / safe_name
                dest.write_bytes(contents)
                excel_paths.append(dest)

    # ── Register job in DB ────────────────────────────────────────────────
    queue_pos = _queue_depth()
    job_store.create_job(
        job_id,
        org_id=org_id,
        project_name=project_name,
        run_mode=run_mode,
    )
    if queue_pos > 0:
        job_store.update_job(job_id, queue_position=queue_pos)

    # Analytics: job submitted
    log_event("analysis_started", job_id=job_id, org_id=org_id, run_mode=run_mode,
              extra={"project_name": project_name, "file_count": len(input_paths)})

    # Metrics counter
    try:
        from src.api.routes.metrics import record_job_queued
        record_job_queued()
    except Exception:
        pass

    # Audit: job accepted
    audit(AuditEvent.JOB_CREATED, ctx if "ctx" in locals() else None,
          resource_type="job", resource_id=job_id,
          detail={"run_mode": run_mode, "project_name": project_name,
                  "file_count": len(input_paths), "queue_position": queue_pos},
          request=request)

    # ── Submit to queue ────────────────────────────────────────────────────
    # Celery (Redis-backed) is preferred — tasks survive restarts and can
    # be distributed across multiple workers.  Falls back to the bounded
    # ThreadPoolExecutor when Redis is unavailable (dev / test).
    # CELERY_ALWAYS_EAGER=true means test/dev mode — bypass Celery entirely
    # because Celery still tries to connect to the Redis broker even in eager
    # mode, which fails when Redis isn't running locally.
    _eager = os.environ.get("CELERY_ALWAYS_EAGER", "false").lower() in ("1", "true", "yes")
    _celery_ok = False
    if _USE_CELERY and run_pipeline_task is not None and not _eager:
        try:
            # Route to fast or full queue based on run_mode
            _fast_modes = {"demo_fast", "demo", "fast"}
            _task = (
                run_pipeline_fast_task
                if run_mode in _fast_modes
                else run_pipeline_full_task
            )
            _task.delay(
                job_id,
                [str(p) for p in input_paths],
                [str(p) for p in excel_paths],
                run_mode,
                project_name,
                str(job_dir),
                org_id,
            )
            _queue = "xboq_fast" if run_mode in _fast_modes else "xboq_full"
            _celery_ok = True
            logger.info("Job %s queued via Celery queue=%s", job_id, _queue)
        except Exception as _ce:
            logger.warning("Celery unavailable (%s) — falling back to ThreadPoolExecutor", _ce)

    if not _celery_ok:
        _pipeline_executor.submit(
            _run_pipeline_in_thread,
            job_id, input_paths, excel_paths, run_mode, project_name, job_dir, org_id,
        )
        logger.info("Job %s queued via ThreadPoolExecutor%s", job_id,
                    " (test/eager mode)" if _eager else " (Celery unavailable)")

    eta_mins = max(1, queue_pos * 5)  # rough ~5 min per queued job ahead
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "queue_position": queue_pos,
            "eta_minutes": eta_mins if queue_pos > 0 else None,
            "status_url": f"/api/jobs/{job_id}",
        },
    )
