"""
StorageBackend — abstract interface for all xBOQ persistence.

Defines the contract that LocalStorage and future S3Storage must satisfy.
Methods mirror the existing pure-function APIs in projects.py, owner_profiles.py,
collaboration.py, and feedback.py.

Sprint 16: Hosted pilot readiness.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageBackend(ABC):
    """Abstract storage interface for xBOQ persistence."""

    # ── Projects ────────────────────────────────────────────────────────

    @abstractmethod
    def create_project(
        self,
        name: str,
        owner: str = "",
        bid_date: str = "",
        notes: str = "",
        project_id: Optional[str] = None,
    ) -> dict:
        """Create a new project. Returns metadata dict with project_id."""
        ...

    @abstractmethod
    def load_project(self, project_id: str) -> Optional[dict]:
        """Load project metadata. Returns None if not found."""
        ...

    @abstractmethod
    def update_project(self, project_id: str, updates: dict) -> Optional[dict]:
        """Update project metadata. Returns updated dict or None."""
        ...

    @abstractmethod
    def list_projects(self) -> List[dict]:
        """Return list of project metadata dicts, sorted newest first."""
        ...

    @abstractmethod
    def save_run(
        self,
        project_id: str,
        run_id: str,
        payload_path: str,
        export_paths: Optional[List[str]] = None,
        run_metadata: Optional[dict] = None,
    ) -> dict:
        """Save a run record for a project. Returns run data dict."""
        ...

    @abstractmethod
    def list_runs(self, project_id: str) -> List[dict]:
        """Return list of run records, sorted newest first."""
        ...

    # ── Profiles ────────────────────────────────────────────────────────

    @abstractmethod
    def save_profile(self, owner_name: str, inputs: dict) -> str:
        """Save owner profile. Returns path/key written."""
        ...

    @abstractmethod
    def load_profile(self, owner_name: str) -> Optional[dict]:
        """Load owner profile. Returns dict or None."""
        ...

    @abstractmethod
    def list_profiles(self) -> List[str]:
        """Return sorted list of owner names with saved profiles."""
        ...

    # ── Collaboration ───────────────────────────────────────────────────

    @abstractmethod
    def append_collaboration(self, entry: dict, project_id: str) -> str:
        """Append a collaboration entry. Returns path/key written."""
        ...

    @abstractmethod
    def load_collaboration(self, project_id: str) -> List[dict]:
        """Load all collaboration entries for a project."""
        ...

    # ── Feedback ────────────────────────────────────────────────────────

    @abstractmethod
    def append_feedback(self, entry: dict, output_dir: str) -> str:
        """Append a feedback entry. Returns path/key written."""
        ...

    @abstractmethod
    def load_feedback(self, output_dir: str) -> List[dict]:
        """Load all feedback entries from output dir."""
        ...

    # ── Generic file I/O ────────────────────────────────────────────────

    @abstractmethod
    def save_file(self, path: str, content: bytes) -> str:
        """Save raw bytes to a path. Returns the path/key written."""
        ...

    @abstractmethod
    def load_file(self, path: str) -> Optional[bytes]:
        """Load raw bytes from a path. Returns None if not found."""
        ...

    @abstractmethod
    def list_files(self, prefix: str) -> List[str]:
        """List files matching a prefix/directory."""
        ...

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        ...
