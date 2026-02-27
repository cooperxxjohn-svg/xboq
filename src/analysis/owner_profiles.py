"""
Owner/Client Profile Persistence — save and load bid strategy
form inputs per owner/client as JSON files.

All functions are pure (no Streamlit, no I/O beyond file read/write).
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


# Default profile directory
DEFAULT_PROFILE_DIR = Path.home() / ".xboq" / "profiles"

# Sanitize owner name for filesystem
_SAFE_NAME_RE = re.compile(r'[^a-zA-Z0-9_\-\s]')


def _sanitize_name(name: str) -> str:
    """Convert owner name to a filesystem-safe filename stem."""
    cleaned = _SAFE_NAME_RE.sub("", name.strip())
    return cleaned.replace(" ", "_").lower()[:100]


def _profile_path(owner_name: str, profile_dir: Path = DEFAULT_PROFILE_DIR) -> Path:
    """Return path for an owner profile JSON file."""
    safe = _sanitize_name(owner_name)
    if not safe:
        safe = "unnamed"
    return profile_dir / f"{safe}.json"


def save_profile(
    owner_name: str,
    inputs: Dict[str, Any],
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> Path:
    """
    Save/overwrite profile JSON for an owner. Returns path written.

    Args:
        owner_name: Display name for the owner/client.
        inputs: Bid strategy form inputs dict (same keys as compute_bid_strategy).
        profile_dir: Directory to store profiles.

    Returns:
        Path to the written profile JSON file.
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "owner_name": owner_name.strip(),
        "inputs": dict(inputs),
        "updated_at": datetime.now().isoformat(),
    }

    path = _profile_path(owner_name, profile_dir)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    return path


def load_profile(
    owner_name: str,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> Optional[Dict[str, Any]]:
    """
    Load owner profile. Returns None if not found.

    Returns:
        Dict with keys: owner_name, inputs, updated_at — or None.
    """
    path = _profile_path(owner_name, Path(profile_dir))
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def list_profiles(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> List[str]:
    """
    Return sorted list of owner names with saved profiles.

    Returns:
        List of owner_name strings, sorted alphabetically.
    """
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        return []

    names = []
    for path in profile_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                name = data.get("owner_name", "")
                if name:
                    names.append(name)
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(names)


def diff_inputs(
    saved_inputs: Dict[str, Any],
    current_inputs: Dict[str, Any],
) -> Dict[str, dict]:
    """
    Compare saved vs current inputs. Returns dict of changed keys.

    Returns:
        Dict of changed keys, each with:
        {"saved": <old_value>, "current": <new_value>}
        Only includes keys where values differ.
    """
    changes: Dict[str, dict] = {}
    all_keys = set(list(saved_inputs.keys()) + list(current_inputs.keys()))

    for key in sorted(all_keys):
        saved_val = saved_inputs.get(key)
        current_val = current_inputs.get(key)
        if saved_val != current_val:
            changes[key] = {
                "saved": saved_val,
                "current": current_val,
            }

    return changes
