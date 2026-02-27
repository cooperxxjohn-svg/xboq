"""
Demo Mode Configuration — env var gating, project configs, UI defaults.

Pure module, no Streamlit dependency. Can be tested independently.

Sprint 17: Demo Hardening + Scripted Dataset.
Sprint 18: Final Demo Polish — added JUMP_TARGETS, DEMO_FREEZE_DEFAULTS.
"""

import os
from typing import Any, Dict, List, Optional


def is_demo_mode() -> bool:
    """Check if XBOQ_DEMO_MODE=true in environment."""
    return os.environ.get("XBOQ_DEMO_MODE", "").lower() == "true"


# ── Demo Project Configs ────────────────────────────────────────────────

DEMO_PROJECTS: List[Dict[str, str]] = [
    {
        "project_id": "pwd_garage",
        "name": "PWD Garage Construction",
        "tender_ref": "PWD/2024/GAR-001",
        "description": "7-page garage construction (structural + electrical)",
        "asset_filename": "pwd_garage.pdf",
    },
    {
        "project_id": "demo_school",
        "name": "Municipal School Block",
        "tender_ref": "MCB/2024/EDU-042",
        "description": "15-page school construction tender (civil + MEP)",
        "asset_filename": "demo_school.pdf",
    },
    {
        "project_id": "demo_hospital",
        "name": "District Hospital OPD Wing",
        "tender_ref": "DHS/2024/HOS-017",
        "description": "22-page hospital OPD wing (full multidisciplinary)",
        "asset_filename": "demo_hospital.pdf",
    },
]


def get_demo_project(project_id: str) -> Optional[dict]:
    """Look up a demo project config by project_id. Returns None if not found."""
    for p in DEMO_PROJECTS:
        if p["project_id"] == project_id:
            return dict(p)
    return None


def get_demo_project_ids() -> List[str]:
    """Return list of all demo project_ids."""
    return [p["project_id"] for p in DEMO_PROJECTS]


# ── UI Defaults ─────────────────────────────────────────────────────────

DEMO_UI_DEFAULTS: Dict[str, object] = {
    "confidence_threshold": 0.85,
    "max_rfis_display": 10,
    "show_debug_panel": False,
    "auto_expand_highlights": True,
    "narration_template": (
        "This is {project_name}, a {page_count}-page tender. "
        "xBOQ found {blocker_count} blockers, {rfi_count} RFIs, "
        "and rated bid readiness at {readiness_score}/100 ({decision}). "
        "Key risk: {top_risk}."
    ),
}


# ── Sprint 18: Jump Targets ───────────────────────────────────────────────

JUMP_TARGETS: List[Dict[str, Any]] = [
    {"key": "1", "label": "Review Queue", "tab_index": 0},
    {"key": "2", "label": "Bid Pack", "tab_index": 3},
    {"key": "3", "label": "Quantities", "tab_index": 7},
    {"key": "4", "label": "Evidence", "tab_index": 10},
    {"key": "5", "label": "Export", "tab_index": None},
    {"key": "E", "label": "Export", "tab_index": None},
]


# ── Sprint 18: Freeze Defaults ────────────────────────────────────────────

DEMO_FREEZE_DEFAULTS: Dict[str, Any] = {
    "confidence_threshold": 0.85,
    "show_all_rfis": True,
    "max_rfis_display": 999,
    "expand_all_sections": True,
}
