"""
Company Playbook Persistence — save/load/list estimating playbooks
by company name under ~/.xboq/playbooks/.

Pure functions, no Streamlit dependency.  Follows the same pattern as
owner_profiles.py and projects.py.

Sprint 20C.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


# Default storage directory (mirrors DEFAULT_PROJECTS_DIR / DEFAULT_PROFILE_DIR)
DEFAULT_PLAYBOOKS_DIR = Path.home() / ".xboq" / "playbooks"

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-\s]")


def _sanitize_name(name: str) -> str:
    """Convert company name to a filesystem-safe stem."""
    cleaned = _SAFE_NAME_RE.sub("", name.strip())
    return cleaned.replace(" ", "_").lower()[:120]


def _playbook_path(
    company_name: str,
    playbooks_dir: Path = DEFAULT_PLAYBOOKS_DIR,
) -> Path:
    safe = _sanitize_name(company_name)
    if not safe:
        safe = "unnamed_company"
    return Path(playbooks_dir) / f"{safe}.json"


# =========================================================================
# CRUD
# =========================================================================

def save_playbook(
    company_name: str,
    playbook: Dict[str, Any],
    playbooks_dir: Path = DEFAULT_PLAYBOOKS_DIR,
) -> Path:
    """
    Save / overwrite a company playbook.  Returns path written.
    """
    playbooks_dir = Path(playbooks_dir)
    playbooks_dir.mkdir(parents=True, exist_ok=True)

    playbook = dict(playbook)
    playbook["updated_at"] = datetime.now().isoformat()

    # Ensure company name is recorded inside the payload
    playbook.setdefault("company", {})["name"] = company_name.strip()

    path = _playbook_path(company_name, playbooks_dir)
    with open(path, "w") as f:
        json.dump(playbook, f, indent=2, default=str)
    return path


def load_playbook(
    company_name: str,
    playbooks_dir: Path = DEFAULT_PLAYBOOKS_DIR,
) -> Optional[Dict[str, Any]]:
    """
    Load a company playbook.  Returns None if not found.
    """
    path = _playbook_path(company_name, Path(playbooks_dir))
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def list_playbooks(
    playbooks_dir: Path = DEFAULT_PLAYBOOKS_DIR,
) -> List[Dict[str, Any]]:
    """
    Return a list of saved playbooks (lightweight summaries).

    Each entry: {company_name, updated_at, path}
    """
    playbooks_dir = Path(playbooks_dir)
    if not playbooks_dir.exists():
        return []

    results = []
    for p in sorted(playbooks_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            results.append({
                "company_name": data.get("company", {}).get("name", p.stem),
                "updated_at": data.get("updated_at", ""),
                "path": str(p),
            })
        except Exception:
            continue
    return results


def delete_playbook(
    company_name: str,
    playbooks_dir: Path = DEFAULT_PLAYBOOKS_DIR,
) -> bool:
    """
    Delete a company playbook.  Returns True if deleted.
    """
    path = _playbook_path(company_name, Path(playbooks_dir))
    if path.exists():
        path.unlink()
        return True
    return False
