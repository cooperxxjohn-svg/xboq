"""
Project / org rate override management API.

GET  /api/rates                          — current org-level overrides + defaults
PUT  /api/rates                          — set/update org-level overrides
DELETE /api/rates/{material_key}         — remove a specific override
GET  /api/rates/keys                     — list all canonical material keys
POST /api/rates/import                   — bulk import from CSV or XLSX

All routes require editor+ role.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.audit_log import audit, AuditEvent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["projects"])


class RateOverrideRequest(BaseModel):
    overrides: dict   # {material_key: {rate_inr, unit?, notes?}}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/rates/keys")
async def list_keys() -> JSONResponse:
    """Return all canonical material keys with labels, units, and default rates."""
    from src.analysis.project_rates import CANONICAL_KEYS
    return JSONResponse(content={
        "keys": {
            k: {
                "label":            v["label"],
                "unit":             v["unit"],
                "default_rate_inr": v["default_rate_inr"],
            }
            for k, v in CANONICAL_KEYS.items()
        }
    })


@router.get("/api/rates")
async def get_rates(request: Request) -> JSONResponse:
    """
    Return current org-level rate overrides merged with canonical defaults.

    Response includes both overridden and default rates so the UI can render a
    complete editable table.
    """
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import require_role
    from src.analysis.project_rates import load_rates, CANONICAL_KEYS

    ctx = await get_tenant_context(request)
    require_role(ctx, "viewer")

    overrides = load_rates(org_id=ctx.org_id)

    # Merge with canonical defaults for the full table
    rows = {}
    for key, meta in CANONICAL_KEYS.items():
        override = overrides.get(key, {})
        rows[key] = {
            "label":            meta["label"],
            "unit":             meta["unit"],
            "default_rate_inr": meta["default_rate_inr"],
            "rate_inr":         override.get("rate_inr"),   # None = not overridden
            "notes":            override.get("notes", ""),
            "updated_by":       override.get("updated_by", ""),
            "updated_at":       override.get("updated_at", ""),
            "is_overridden":    key in overrides,
        }

    # Sanitize: replace NaN/Inf floats with None (not JSON-serializable)
    import math
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    safe_rows = {
        k: {fk: _clean(fv) for fk, fv in row.items()}
        for k, row in rows.items()
    }
    return JSONResponse(content={"org_id": ctx.org_id, "rates": safe_rows})


@router.put("/api/rates")
async def set_rates(body: RateOverrideRequest, request: Request) -> JSONResponse:
    """
    Set or update one or more org-level rate overrides.

    Body: {"overrides": {"fe500": {"rate_inr": 82000, "notes": "Q2 2025 procurement"}}}
    """
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import require_role
    from src.analysis.project_rates import save_org_rates

    ctx = await get_tenant_context(request)
    require_role(ctx, "editor")

    # Inject updated_by
    overrides = {}
    for k, v in (body.overrides or {}).items():
        overrides[k] = {**v, "updated_by": ctx.org_id}

    try:
        saved = save_org_rates(ctx.org_id, overrides)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Audit each key that was set
    for key in (saved or {}):
        rate_val = saved[key].get("rate_inr") if isinstance(saved.get(key), dict) else None
        audit(AuditEvent.RATE_OVERRIDE_SET, ctx,
              resource_type="rate", resource_id=key,
              detail={"rate_inr": rate_val}, request=request)

    return JSONResponse(content={"org_id": ctx.org_id, "saved": saved})


@router.delete("/api/rates/{material_key}")
async def delete_rate(material_key: str, request: Request) -> JSONResponse:
    """Remove an org-level rate override (reverts to engine default)."""
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import require_role
    from src.analysis.project_rates import delete_rate_override

    ctx = await get_tenant_context(request)
    require_role(ctx, "editor")

    removed = delete_rate_override(ctx.org_id, material_key)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No override found for '{material_key}'",
        )
    audit(AuditEvent.RATE_OVERRIDE_DELETED, ctx,
          resource_type="rate", resource_id=material_key, request=request)
    return JSONResponse(content={"removed": True, "key": material_key})

@router.post("/api/rates/import", status_code=200)
async def import_rates(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Bulk import org-level rate overrides from CSV or XLSX.

    Expected columns (case-insensitive):
      material_key   — canonical key (e.g. "fe500", "cement_opc53")
      rate_inr       — rate in Indian Rupees (numeric)
      notes          — optional description / source

    Unknown material_key values are rejected with a 422 listing the bad keys.
    Rows with zero or missing rate_inr are skipped.

    Returns a summary: {imported, skipped, errors}
    """
    from src.api.middleware.tenant_auth import get_tenant_context
    from src.auth.rbac import require_role
    from src.analysis.project_rates import save_org_rates, CANONICAL_KEYS

    ctx = await get_tenant_context(request)
    require_role(ctx, "editor")

    filename = (file.filename or "upload").lower()
    contents = await file.read()

    # Parse into rows: list of dicts
    rows = []
    try:
        import pandas as pd
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a CSV or XLSX file.",
            )
        # Normalise column names: strip + lower
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        rows = df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}")

    if not rows:
        raise HTTPException(status_code=422, detail="File is empty or has no data rows")

    # Validate + build overrides dict
    valid_keys = set(CANONICAL_KEYS.keys())
    overrides = {}
    skipped = []
    bad_keys = []

    for i, row in enumerate(rows, start=2):   # row 2 = first data row (row 1 = header)
        key = str(row.get("material_key", "") or "").strip().lower()
        if not key:
            skipped.append({"row": i, "reason": "missing material_key"})
            continue
        if key not in valid_keys:
            bad_keys.append(key)
            continue
        raw_rate = row.get("rate_inr", "")
        # Treat NaN, None, empty string as missing
        raw_str = str(raw_rate).strip().lower() if raw_rate is not None else ""
        if raw_str in ("", "nan", "none", "null", "-"):
            skipped.append({"row": i, "key": key, "reason": "rate_inr is empty"})
            continue
        try:
            rate = float(raw_rate)
        except (TypeError, ValueError):
            skipped.append({"row": i, "key": key, "reason": "invalid rate_inr"})
            continue
        if rate <= 0:
            skipped.append({"row": i, "key": key, "reason": "rate_inr is zero or negative"})
            continue
        notes = str(row.get("notes", "") or "").strip()
        overrides[key] = {"rate_inr": rate, "notes": notes, "updated_by": ctx.org_id}

    if bad_keys:
        valid_list = sorted(valid_keys)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unknown material_key values in import",
                "unknown_keys": bad_keys,
                "valid_keys": valid_list,
            },
        )

    if not overrides:
        return JSONResponse(content={
            "imported": 0,
            "skipped": len(skipped),
            "skipped_detail": skipped,
            "message": "No valid rates found to import",
        })

    save_org_rates(ctx.org_id, overrides)
    imported_keys = list(overrides.keys())

    # Audit the bulk import
    audit(AuditEvent.RATE_BULK_IMPORT, ctx,
          resource_type="rate", resource_id="bulk",
          detail={"imported": len(imported_keys), "keys": imported_keys},
          request=request)

    return JSONResponse(content={
        "imported": len(imported_keys),
        "skipped": len(skipped),
        "skipped_detail": skipped,
        "keys_imported": imported_keys,
        "message": f"Successfully imported {len(imported_keys)} rate override(s)",
    })
