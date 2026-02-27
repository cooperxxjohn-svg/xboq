"""
Demo Asset Resolution — find and validate demo file paths.

Pure module, no Streamlit dependency. Can be tested independently.

Sprint 17: Demo Hardening + Scripted Dataset.
"""

from pathlib import Path
from typing import Dict, List, Optional

# Project root: two levels up from src/demo/
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def resolve_demo_pdf(project_id: str, asset_filename: str) -> Optional[Path]:
    """
    Resolve path to a demo PDF file.

    Search order:
      1. demo_inputs/<asset_filename>
      2. data/projects/<project_id>/<asset_filename>

    Returns Path if found, None otherwise.
    """
    candidates = [
        _PROJECT_ROOT / "demo_inputs" / asset_filename,
        _PROJECT_ROOT / "data" / "projects" / project_id / asset_filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def resolve_demo_cache(project_id: str) -> Optional[Path]:
    """
    Resolve path to a cached analysis.json for a demo project.

    Search order:
      1. demo_cache/<project_id>/analysis.json
      2. out/<project_id>/analysis.json

    Returns Path if found, None otherwise.
    """
    candidates = [
        _PROJECT_ROOT / "demo_cache" / project_id / "analysis.json",
        _PROJECT_ROOT / "out" / project_id / "analysis.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def validate_demo_assets() -> List[Dict[str, object]]:
    """
    Validate all demo project assets and return status report.

    Returns list of dicts:
    [{
        "project_id": str,
        "name": str,
        "pdf_found": bool,
        "pdf_path": str | None,
        "cache_found": bool,
        "cache_path": str | None,
    }]
    """
    from .demo_config import DEMO_PROJECTS

    report = []
    for proj in DEMO_PROJECTS:
        pid = proj["project_id"]
        pdf = resolve_demo_pdf(pid, proj["asset_filename"])
        cache = resolve_demo_cache(pid)
        report.append({
            "project_id": pid,
            "name": proj["name"],
            "pdf_found": pdf is not None,
            "pdf_path": str(pdf) if pdf else None,
            "cache_found": cache is not None,
            "cache_path": str(cache) if cache else None,
        })
    return report
