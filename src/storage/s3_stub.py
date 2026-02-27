"""
S3Storage — stub implementation of StorageBackend for AWS S3.

Validates configuration but raises NotImplementedError on all operations.
Placeholder for Sprint 17 cloud deployment.

Sprint 16: Hosted pilot readiness.
"""

from typing import Any, Dict, List, Optional

from .interface import StorageBackend


class S3Storage(StorageBackend):
    """AWS S3 storage backend stub.

    Accepts configuration parameters for future implementation.
    All operations raise NotImplementedError.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "xboq/",
        region: str = "ap-south-1",
    ):
        if not bucket:
            raise ValueError("S3Storage requires a non-empty bucket name")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._region = region

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def region(self) -> str:
        return self._region

    # ── Projects ────────────────────────────────────────────────────────

    def create_project(self, name: str, owner: str = "", bid_date: str = "",
                       notes: str = "", project_id: Optional[str] = None) -> dict:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def load_project(self, project_id: str) -> Optional[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def update_project(self, project_id: str, updates: dict) -> Optional[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def list_projects(self) -> List[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def save_run(self, project_id: str, run_id: str, payload_path: str,
                 export_paths: Optional[List[str]] = None,
                 run_metadata: Optional[dict] = None) -> dict:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def list_runs(self, project_id: str) -> List[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    # ── Profiles ────────────────────────────────────────────────────────

    def save_profile(self, owner_name: str, inputs: dict) -> str:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def load_profile(self, owner_name: str) -> Optional[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def list_profiles(self) -> List[str]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    # ── Collaboration ───────────────────────────────────────────────────

    def append_collaboration(self, entry: dict, project_id: str) -> str:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def load_collaboration(self, project_id: str) -> List[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    # ── Feedback ────────────────────────────────────────────────────────

    def append_feedback(self, entry: dict, output_dir: str) -> str:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def load_feedback(self, output_dir: str) -> List[dict]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    # ── Generic file I/O ────────────────────────────────────────────────

    def save_file(self, path: str, content: bytes) -> str:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def load_file(self, path: str) -> Optional[bytes]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def list_files(self, prefix: str) -> List[str]:
        raise NotImplementedError("S3Storage coming in Sprint 17")

    def file_exists(self, path: str) -> bool:
        raise NotImplementedError("S3Storage coming in Sprint 17")
