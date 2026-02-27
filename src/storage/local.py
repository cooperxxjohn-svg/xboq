"""
LocalStorage — filesystem-based StorageBackend implementation.

Delegates to existing pure functions in projects.py, owner_profiles.py,
collaboration.py, and feedback.py. Each function already accepts a
configurable directory parameter, so LocalStorage simply passes its
own base_dir-derived paths.

Sprint 16: Hosted pilot readiness.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .interface import StorageBackend


class LocalStorage(StorageBackend):
    """Filesystem-based storage backend for xBOQ.

    Directory layout under base_dir:
        base_dir/
            projects/         — project metadata + run history
            profiles/         — owner/client bid strategy profiles
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self._base_dir = Path.home() / ".xboq"
        else:
            self._base_dir = Path(base_dir)

        self._projects_dir = self._base_dir / "projects"
        self._profiles_dir = self._base_dir / "profiles"

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def projects_dir(self) -> Path:
        return self._projects_dir

    @property
    def profiles_dir(self) -> Path:
        return self._profiles_dir

    # ── Projects ────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        owner: str = "",
        bid_date: str = "",
        notes: str = "",
        project_id: Optional[str] = None,
    ) -> dict:
        from src.analysis.projects import create_project
        return create_project(
            name=name, owner=owner, bid_date=bid_date, notes=notes,
            project_id=project_id, projects_dir=self._projects_dir,
        )

    def load_project(self, project_id: str) -> Optional[dict]:
        from src.analysis.projects import load_project
        return load_project(project_id, projects_dir=self._projects_dir)

    def update_project(self, project_id: str, updates: dict) -> Optional[dict]:
        from src.analysis.projects import update_project
        return update_project(project_id, updates, projects_dir=self._projects_dir)

    def list_projects(self) -> List[dict]:
        from src.analysis.projects import list_projects
        return list_projects(projects_dir=self._projects_dir)

    def save_run(
        self,
        project_id: str,
        run_id: str,
        payload_path: str,
        export_paths: Optional[List[str]] = None,
        run_metadata: Optional[dict] = None,
    ) -> dict:
        from src.analysis.projects import save_run
        result = save_run(
            project_id=project_id, run_id=run_id,
            payload_path=payload_path,
            export_paths=export_paths or [],
            run_metadata=run_metadata or {},
            projects_dir=self._projects_dir,
        )
        # save_run returns a Path; load the JSON to return a dict
        if isinstance(result, Path) and result.exists():
            import json
            with open(result) as f:
                return json.load(f)
        return {"run_id": run_id, "project_id": project_id}

    def list_runs(self, project_id: str) -> List[dict]:
        from src.analysis.projects import list_runs
        return list_runs(project_id, projects_dir=self._projects_dir)

    # ── Profiles ────────────────────────────────────────────────────────

    def save_profile(self, owner_name: str, inputs: dict) -> str:
        from src.analysis.owner_profiles import save_profile
        path = save_profile(owner_name, inputs, profile_dir=self._profiles_dir)
        return str(path)

    def load_profile(self, owner_name: str) -> Optional[dict]:
        from src.analysis.owner_profiles import load_profile
        return load_profile(owner_name, profile_dir=self._profiles_dir)

    def list_profiles(self) -> List[str]:
        from src.analysis.owner_profiles import list_profiles
        return list_profiles(profile_dir=self._profiles_dir)

    # ── Collaboration ───────────────────────────────────────────────────

    def append_collaboration(self, entry: dict, project_id: str) -> str:
        from src.analysis.collaboration import append_collaboration
        from src.analysis.projects import project_dir
        pdir = project_dir(project_id, self._projects_dir)
        path = append_collaboration(entry, project_dir=pdir)
        return str(path)

    def load_collaboration(self, project_id: str) -> List[dict]:
        from src.analysis.collaboration import load_collaboration
        from src.analysis.projects import project_dir
        pdir = project_dir(project_id, self._projects_dir)
        return load_collaboration(project_dir=pdir)

    # ── Feedback ────────────────────────────────────────────────────────

    def append_feedback(self, entry: dict, output_dir: str) -> str:
        from src.analysis.feedback import append_feedback
        path = append_feedback(entry, output_dir=Path(output_dir))
        return str(path)

    def load_feedback(self, output_dir: str) -> List[dict]:
        from src.analysis.feedback import load_feedback
        return load_feedback(output_dir=Path(output_dir))

    # ── Generic file I/O ────────────────────────────────────────────────

    def save_file(self, path: str, content: bytes) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return str(p)

    def load_file(self, path: str) -> Optional[bytes]:
        p = Path(path)
        if not p.exists():
            return None
        return p.read_bytes()

    def list_files(self, prefix: str) -> List[str]:
        p = Path(prefix)
        if not p.exists():
            return []
        if p.is_file():
            return [str(p)]
        return sorted(str(f) for f in p.rglob("*") if f.is_file())

    def file_exists(self, path: str) -> bool:
        return Path(path).exists()
