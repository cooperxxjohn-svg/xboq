"""
Projects — create/list/load projects and manage run history.

Storage: ~/.xboq/projects/<project_id>/
    metadata.json      — project name, owner, bid_date, notes, timestamps
    runs/
        <run_id>.json  — timestamp, payload_path, export_paths, summary metrics

All functions are pure (no Streamlit, no I/O beyond file read/write).
Follows the owner_profiles.py pattern for JSON persistence.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Constants ────────────────────────────────────────────────────────────

DEFAULT_PROJECTS_DIR = Path.home() / ".xboq" / "projects"

_SAFE_NAME_RE = re.compile(r'[^a-zA-Z0-9_\-\s]')


# ─── Helpers ──────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """Convert name to a filesystem-safe string."""
    cleaned = _SAFE_NAME_RE.sub("", name.strip())
    return cleaned.replace(" ", "_").lower()[:100]


def project_dir(
    project_id: str,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> Path:
    """Return path to a project directory."""
    return Path(projects_dir) / project_id


def _generate_project_id() -> str:
    """Generate a unique project ID. Avoids importing pipeline to keep lightweight."""
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"project_{timestamp}_{short_uuid}"


# ─── CRUD ─────────────────────────────────────────────────────────────────

def create_project(
    name: str,
    owner: str = "",
    bid_date: str = "",
    notes: str = "",
    # Sprint 20: Pilot fields
    company_name: str = "",
    trades_in_scope: Optional[List[str]] = None,
    output_preferences: Optional[Dict[str, Any]] = None,
    pilot_mode: bool = False,
    project_id: Optional[str] = None,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> Dict[str, Any]:
    """
    Create a new project. Returns metadata dict with project_id.

    Creates:
        projects_dir/<project_id>/metadata.json
        projects_dir/<project_id>/runs/

    Args:
        name: Human-readable project name.
        owner: Client/owner name.
        bid_date: Bid submission date (free text, typically ISO date).
        notes: Any notes about the project.
        company_name: Contractor/company name (Sprint 20 pilot).
        trades_in_scope: List of trades relevant to this project.
        output_preferences: Dict of output preferences.
        pilot_mode: Whether this is a pilot engagement.
        project_id: Explicit ID, or auto-generated if None.
        projects_dir: Root directory for all projects.

    Returns:
        Metadata dict with all fields + project_id, created_at, updated_at.
    """
    if project_id is None:
        project_id = _generate_project_id()

    proj_dir = project_dir(project_id, projects_dir)
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "runs").mkdir(exist_ok=True)

    now = datetime.now().isoformat()
    metadata = {
        "project_id": project_id,
        "name": name.strip(),
        "owner": owner.strip(),
        "bid_date": bid_date.strip(),
        "notes": notes.strip(),
        # Sprint 20: Pilot fields
        "company_name": company_name.strip() if company_name else "",
        "trades_in_scope": trades_in_scope or [],
        "output_preferences": output_preferences or {},
        "pilot_mode": pilot_mode,
        "created_at": now,
        "updated_at": now,
    }

    meta_path = proj_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def load_project(
    project_id: str,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> Optional[Dict[str, Any]]:
    """
    Load project metadata. Returns None if not found.
    """
    meta_path = project_dir(project_id, projects_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def update_project(
    project_id: str,
    updates: Dict[str, Any],
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> Optional[Dict[str, Any]]:
    """
    Update project metadata fields. Returns updated metadata or None if not found.

    Only updates keys present in `updates` dict. Always refreshes `updated_at`.
    """
    metadata = load_project(project_id, projects_dir)
    if metadata is None:
        return None

    # Only allow updating safe fields
    for key in ("name", "owner", "bid_date", "notes",
                "company_name", "trades_in_scope", "output_preferences", "pilot_mode"):
        if key in updates:
            metadata[key] = updates[key]

    metadata["updated_at"] = datetime.now().isoformat()

    meta_path = project_dir(project_id, projects_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def list_projects(
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> List[Dict[str, Any]]:
    """
    Return list of project metadata dicts, sorted newest first by updated_at.

    Scans all subdirs of projects_dir for metadata.json.
    """
    projects_dir = Path(projects_dir)
    if not projects_dir.exists():
        return []

    projects = []
    for child in projects_dir.iterdir():
        if child.is_dir():
            meta_path = child / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    projects.append(meta)
                except (json.JSONDecodeError, IOError):
                    continue

    # Sort newest first
    projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return projects


# ─── Run History ──────────────────────────────────────────────────────────

def save_run(
    project_id: str,
    run_id: str,
    payload_path: str,
    export_paths: List[str] = None,
    run_metadata: Dict[str, Any] = None,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> Path:
    """
    Save a run record for a project.

    Writes to: projects_dir/<project_id>/runs/<run_id>.json

    Args:
        project_id: Project identifier.
        run_id: Unique run identifier (e.g., "run_20260219_143022").
        payload_path: Path to the analysis.json file.
        export_paths: List of exported file paths.
        run_metadata: Additional metadata (readiness_score, decision, etc.).
        projects_dir: Root directory for all projects.

    Returns:
        Path to the run JSON file.
    """
    runs_dir = project_dir(project_id, projects_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_data = {
        "run_id": run_id,
        "project_id": project_id,
        "timestamp": datetime.now().isoformat(),
        "payload_path": payload_path,
        "export_paths": export_paths or [],
    }

    # Merge in additional metadata
    if run_metadata:
        for key, val in run_metadata.items():
            run_data.setdefault(key, val)

    run_path = runs_dir / f"{run_id}.json"
    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)

    # Touch project updated_at
    try:
        meta = load_project(project_id, projects_dir)
        if meta:
            meta["updated_at"] = datetime.now().isoformat()
            meta_path = project_dir(project_id, projects_dir) / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
    except Exception:
        pass

    return run_path


def list_runs(
    project_id: str,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
) -> List[Dict[str, Any]]:
    """
    Return list of run records for a project, sorted newest first by timestamp.
    """
    runs_dir = project_dir(project_id, projects_dir) / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for run_file in runs_dir.glob("*.json"):
        try:
            with open(run_file, "r") as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue

    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs
