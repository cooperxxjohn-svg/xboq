"""
Project / org-level rate overrides for xBOQ.ai.

Allows estimators to replace the default Q1-2025 benchmark rates with their
organisation's actual procurement rates (e.g. ₹82,000/MT for steel vs default
₹88,000/MT).

Storage hierarchy (first non-null wins during lookup):
    1. Project-specific overrides   ~/.xboq/project_rates/{project_id}.json
    2. Org-wide defaults            ~/.xboq/org_rates/{org_id}.json
    3. Rate engine defaults         src/analysis/qto/rate_engine.py _RATES dict

Canonical material keys (v1 — the 8 highest-impact items):
    fe500            — Fe500 rebar (MT)
    fe415            — Fe415 rebar (MT)
    cement_opc53     — OPC 53-grade cement (bag or MT)
    river_sand       — River / M-sand fine aggregate (cum)
    aggregate_20mm   — 20mm coarse aggregate (cum)
    shuttering       — Centering & shuttering (sqm)
    upvc_pipe        — UPVC plumbing pipe (rmt)
    electrical_conduit — PVC conduit (rmt)
    common_brick     — Country / fly-ash brick (nos or thousand)
    labour_mason     — Mason day rate (day)
    labour_helper    — Helper day rate (day)

Usage:
    from src.analysis.project_rates import load_rates, save_org_rates, CANONICAL_KEYS

    rates = load_rates(org_id="acme", project_id="proj_abc")
    # → {"fe500": {"rate_inr": 82000, "unit": "MT", ...}, ...}

    apply_rates(items, project_rates=rates)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

_XBOQ_HOME = Path.home() / ".xboq"
_ORG_RATES_DIR     = _XBOQ_HOME / "org_rates"
_PROJECT_RATES_DIR = _XBOQ_HOME / "project_rates"

_ORG_RATES_DIR.mkdir(parents=True, exist_ok=True)
_PROJECT_RATES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DB-backed storage helpers (with file fallback for local dev)
# ---------------------------------------------------------------------------

def _load_from_db(org_id: str = "", project_id: str = "") -> dict:
    """
    Load rate overrides from the DB.
    Returns {} if DB is unavailable or no rows found.
    """
    try:
        from src.api.db import SessionLocal
        from src.api.models import OrgRateModel, ProjectRateModel
        from sqlalchemy import select

        with SessionLocal() as db:
            if project_id:
                rows = db.execute(
                    select(ProjectRateModel).where(
                        ProjectRateModel.project_id == project_id
                    )
                ).scalars().all()
                if rows:
                    return {
                        r.material_key: {
                            "rate_inr":   r.rate_inr,
                            "unit":       r.unit,
                            "notes":      r.notes,
                            "updated_by": r.updated_by,
                            "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                        }
                        for r in rows
                    }
            elif org_id:
                rows = db.execute(
                    select(OrgRateModel).where(OrgRateModel.org_id == org_id)
                ).scalars().all()
                if rows:
                    return {
                        r.material_key: {
                            "rate_inr":   r.rate_inr,
                            "unit":       r.unit,
                            "notes":      r.notes,
                            "updated_by": r.updated_by,
                            "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                        }
                        for r in rows
                    }
    except Exception as exc:
        logger.debug("DB rate load failed (will try file fallback): %s", exc)
    return {}


def _save_to_db(
    data: dict,
    org_id: str = "",
    project_id: str = "",
    updated_by: str = "",
) -> bool:
    """
    Persist rate overrides to DB.
    Returns True on success, False on failure.
    """
    try:
        from src.api.db import SessionLocal
        from src.api.models import OrgRateModel, ProjectRateModel
        from sqlalchemy import delete as sa_delete
        from datetime import datetime, timezone

        with SessionLocal() as db:
            if project_id:
                db.execute(sa_delete(ProjectRateModel).where(
                    ProjectRateModel.project_id == project_id
                ))
                for key, vals in data.items():
                    db.add(ProjectRateModel(
                        project_id=project_id,
                        org_id=org_id or "local",
                        material_key=key,
                        rate_inr=float(vals.get("rate_inr", 0)),
                        unit=str(vals.get("unit", "")),
                        notes=str(vals.get("notes", "")),
                        updated_by=updated_by or str(vals.get("updated_by", "")),
                        updated_at=datetime.now(timezone.utc),
                    ))
            elif org_id:
                db.execute(sa_delete(OrgRateModel).where(
                    OrgRateModel.org_id == org_id
                ))
                for key, vals in data.items():
                    db.add(OrgRateModel(
                        org_id=org_id,
                        material_key=key,
                        rate_inr=float(vals.get("rate_inr", 0)),
                        unit=str(vals.get("unit", "")),
                        notes=str(vals.get("notes", "")),
                        updated_by=updated_by or str(vals.get("updated_by", "")),
                        updated_at=datetime.now(timezone.utc),
                    ))
            db.commit()
        return True
    except Exception as exc:
        logger.warning("DB rate save failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Canonical keys + default labels
# ---------------------------------------------------------------------------

CANONICAL_KEYS: dict[str, dict] = {
    "fe500": {
        "label": "Steel Rebar Fe500",
        "unit":  "MT",
        "default_rate_inr": 88000.0,
        "keywords": ["fe500", "fe 500", "deformed bar", "tor steel", "rebar", "steel reinforcement"],
    },
    "fe415": {
        "label": "Steel Rebar Fe415",
        "unit":  "MT",
        "default_rate_inr": 84000.0,
        "keywords": ["fe415", "fe 415", "mild steel"],
    },
    "cement_opc53": {
        "label": "Cement OPC 53 Grade",
        "unit":  "bag",
        "default_rate_inr": 400.0,
        "keywords": ["cement opc 53", "opc 53", "cement 53", "cement bag"],
    },
    "river_sand": {
        "label": "River Sand / M-Sand",
        "unit":  "cum",
        "default_rate_inr": 2200.0,
        "keywords": ["river sand", "fine sand", "m-sand", "manufactured sand", "sand"],
    },
    "aggregate_20mm": {
        "label": "20mm Coarse Aggregate",
        "unit":  "cum",
        "default_rate_inr": 1400.0,
        "keywords": ["20mm aggregate", "coarse aggregate", "jelly", "aggregate 20"],
    },
    "shuttering": {
        "label": "Centering & Shuttering",
        "unit":  "sqm",
        "default_rate_inr": 900.0,
        "keywords": ["shuttering", "centering", "formwork", "centering & shuttering"],
    },
    "upvc_pipe": {
        "label": "UPVC Plumbing Pipe",
        "unit":  "rmt",
        "default_rate_inr": 320.0,
        "keywords": ["upvc pipe", "pvc pipe", "cpvc pipe", "plumbing pipe"],
    },
    "electrical_conduit": {
        "label": "PVC Electrical Conduit",
        "unit":  "rmt",
        "default_rate_inr": 95.0,
        "keywords": ["electrical conduit", "pvc conduit", "rigid conduit", "conduit pipe"],
    },
    "common_brick": {
        "label": "Common / Fly-Ash Brick",
        "unit":  "thousand",
        "default_rate_inr": 8500.0,
        "keywords": ["fly ash brick", "common brick", "country brick", "modular brick", "brick"],
    },
    "labour_mason": {
        "label": "Mason Day Rate",
        "unit":  "day",
        "default_rate_inr": 900.0,
        "keywords": ["mason", "bricklayer", "skilled worker"],
    },
    "labour_helper": {
        "label": "Helper / Unskilled Day Rate",
        "unit":  "day",
        "default_rate_inr": 550.0,
        "keywords": ["helper", "unskilled", "labour", "mazdoor"],
    },
}


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _safe_id(id_str: str) -> str:
    """Sanitise an ID string to prevent path traversal."""
    return "".join(c for c in id_str if c.isalnum() or c in "-_.")[:64]


def _org_path(org_id: str) -> Path:
    return _ORG_RATES_DIR / f"{_safe_id(org_id)}.json"


def _project_path(project_id: str) -> Path:
    return _PROJECT_RATES_DIR / f"{_safe_id(project_id)}.json"


def _load(path: Path, *, org_id: str = "", project_id: str = "") -> dict:
    """Load rates — DB first, then file fallback."""
    # Try DB
    db_data = _load_from_db(org_id=org_id, project_id=project_id)
    if db_data:
        return db_data
    # File fallback
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load rates from %s: %s", path, exc)
        return {}


def _save(path: Path, data: dict, *, org_id: str = "", project_id: str = "", updated_by: str = "") -> None:
    """Save rates — DB primary, file mirror."""
    _save_to_db(data, org_id=org_id, project_id=project_id, updated_by=updated_by)
    try:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("File rate mirror failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rates(
    org_id: str = "local",
    project_id: str = "",
) -> dict[str, dict]:
    """
    Load effective rate overrides for an org/project, merging both layers.

    Returns a dict: {material_key: {rate_inr, unit, updated_by, updated_at}}
    Only keys with actual overrides are returned — defaults are not included.
    """
    org_rates = _load(_org_path(org_id), org_id=org_id)
    if not project_id:
        return org_rates

    # Project rates take precedence over org rates
    proj_rates = _load(_project_path(project_id), project_id=project_id, org_id=org_id)
    merged = {**org_rates, **proj_rates}
    return merged


def save_org_rates(org_id: str, overrides: dict[str, dict], updated_by: str = "") -> dict:
    """
    Persist org-level rate overrides.

    overrides: {material_key: {rate_inr, unit, notes}}
    Returns the saved overrides dict.
    Raises ValueError on unknown material_key or invalid rate.
    """
    current = _load(_org_path(org_id), org_id=org_id)
    now = datetime.now(timezone.utc).isoformat()

    for key, vals in overrides.items():
        if key not in CANONICAL_KEYS:
            raise ValueError(
                f"Unknown material key '{key}'. "
                f"Valid keys: {', '.join(sorted(CANONICAL_KEYS))}"
            )
        rate = float(vals.get("rate_inr") or 0)
        if rate <= 0:
            raise ValueError(f"rate_inr for '{key}' must be > 0")
        current[key] = {
            "rate_inr":   rate,
            "unit":       vals.get("unit") or CANONICAL_KEYS[key]["unit"],
            "notes":      vals.get("notes", ""),
            "updated_by": updated_by or vals.get("updated_by", ""),
            "updated_at": now,
        }

    _save(_org_path(org_id), current, org_id=org_id, updated_by=updated_by)
    return current


def save_project_rates(project_id: str, overrides: dict[str, dict], org_id: str = "local", updated_by: str = "") -> dict:
    """Persist project-specific rate overrides (same schema as save_org_rates)."""
    current = _load(_project_path(project_id), project_id=project_id, org_id=org_id)
    now = datetime.now(timezone.utc).isoformat()

    for key, vals in overrides.items():
        if key not in CANONICAL_KEYS:
            raise ValueError(f"Unknown material key '{key}'")
        rate = float(vals.get("rate_inr") or 0)
        if rate <= 0:
            raise ValueError(f"rate_inr for '{key}' must be > 0")
        current[key] = {
            "rate_inr":   rate,
            "unit":       vals.get("unit") or CANONICAL_KEYS[key]["unit"],
            "notes":      vals.get("notes", ""),
            "updated_by": updated_by or vals.get("updated_by", ""),
            "updated_at": now,
        }

    _save(_project_path(project_id), current, org_id=org_id, project_id=project_id, updated_by=updated_by)
    return current


def delete_rate_override(
    org_id: str,
    key: str,
    project_id: str = "",
) -> bool:
    """Remove a single material override. Returns True if it existed."""
    if project_id:
        data = _load(_project_path(project_id), project_id=project_id, org_id=org_id)
        if key not in data:
            return False
        del data[key]
        _save(_project_path(project_id), data, org_id=org_id, project_id=project_id)
    else:
        data = _load(_org_path(org_id), org_id=org_id)
        if key not in data:
            return False
        del data[key]
        _save(_org_path(org_id), data, org_id=org_id)
    return True


def rates_to_lookup(
    overrides: dict[str, dict],
) -> dict[str, float]:
    """
    Convert overrides dict to a flat {keyword: rate_inr} lookup suitable for
    injecting into the rate engine's keyword-matching algorithm.

    All keywords for each canonical key are mapped to the same rate.
    """
    lookup: dict[str, float] = {}
    for key, vals in overrides.items():
        if key not in CANONICAL_KEYS:
            continue
        rate = float(vals.get("rate_inr") or 0)
        if rate <= 0:
            continue
        for kw in CANONICAL_KEYS[key]["keywords"]:
            lookup[kw] = rate
    return lookup
