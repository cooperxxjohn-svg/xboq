"""
S3Storage — AWS S3 implementation of StorageBackend.

Stores projects, profiles, collaboration logs, and arbitrary files
in an S3 bucket under a configurable prefix.

Layout:
    {prefix}projects/{project_id}/meta.json
    {prefix}projects/{project_id}/runs/{run_id}.json
    {prefix}profiles/{owner_name}.json
    {prefix}collab/{project_id}/collaboration.jsonl
    {prefix}feedback/{output_dir_hash}.jsonl
    {prefix}files/{path}

Requires: boto3 >= 1.26
Install: pip install boto3

Sprint 47: Full S3 implementation (was stub in Sprint 16/17).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .interface import StorageBackend

logger = logging.getLogger(__name__)


def _require_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3Storage. "
            "Install it with: pip install boto3"
        )


class S3Storage(StorageBackend):
    """AWS S3 storage backend for xBOQ.

    Parameters
    ----------
    bucket : str
        S3 bucket name (must already exist).
    prefix : str
        Key prefix under which all xBOQ objects are stored (default: "xboq/").
    region : str
        AWS region for the bucket (default: "ap-south-1").
    org_id : str, optional
        Tenant organisation ID.  When provided, all keys are prefixed with
        ``{prefix}{org_id}/…`` so each tenant gets an isolated namespace.
    aws_access_key_id / aws_secret_access_key : str, optional
        Explicit credentials.  If omitted, boto3 uses the standard credential
        chain (env vars → ~/.aws/credentials → IAM role).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "xboq/",
        region: str = "ap-south-1",
        org_id: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        if not bucket:
            raise ValueError("S3Storage requires a non-empty bucket name")
        self._bucket = bucket
        base = prefix.rstrip("/") + "/"
        if org_id:
            self._prefix = base + org_id.strip("/") + "/"
        else:
            self._prefix = base
        self._org_id = org_id
        self._region = region
        self._access_key = aws_access_key_id
        self._secret_key = aws_secret_access_key
        self._client = None  # lazy-init

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def region(self) -> str:
        return self._region

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _s3(self):
        """Return (lazily created) boto3 S3 client."""
        if self._client is None:
            boto3 = _require_boto3()
            kwargs: Dict[str, Any] = {"region_name": self._region}
            if self._access_key:
                kwargs["aws_access_key_id"] = self._access_key
            if self._secret_key:
                kwargs["aws_secret_access_key"] = self._secret_key
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _key(self, *parts: str) -> str:
        """Build an S3 key from the prefix and path parts."""
        return self._prefix + "/".join(p.strip("/") for p in parts if p)

    def _put_json(self, key: str, data: Any) -> None:
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self._s3().put_object(
            Bucket=self._bucket, Key=key, Body=body,
            ContentType="application/json",
        )

    def _get_json(self, key: str) -> Optional[Any]:
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as exc:
            if "NoSuchKey" in str(exc) or "404" in str(exc):
                return None
            logger.error("S3 get_json failed for %s: %s", key, exc)
            raise

    def _list_keys(self, prefix: str) -> List[str]:
        """Return all S3 keys under the given prefix (handles pagination)."""
        paginator = self._s3().get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    # ── Projects ─────────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        owner: str = "",
        bid_date: str = "",
        notes: str = "",
        project_id: Optional[str] = None,
    ) -> dict:
        project_id = project_id or f"proj_{uuid4().hex[:12]}"
        meta = {
            "project_id": project_id,
            "name": name,
            "owner": owner,
            "bid_date": bid_date,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._put_json(self._key("projects", project_id, "meta.json"), meta)
        return meta

    def load_project(self, project_id: str) -> Optional[dict]:
        return self._get_json(self._key("projects", project_id, "meta.json"))

    def update_project(self, project_id: str, updates: dict) -> Optional[dict]:
        meta = self.load_project(project_id)
        if meta is None:
            return None
        meta.update(updates)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._put_json(self._key("projects", project_id, "meta.json"), meta)
        return meta

    def list_projects(self) -> List[dict]:
        prefix = self._key("projects") + "/"
        keys = [k for k in self._list_keys(prefix) if k.endswith("/meta.json")]
        projects: List[dict] = []
        for key in keys:
            data = self._get_json(key)
            if data:
                projects.append(data)
        projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
        return projects

    def save_run(
        self,
        project_id: str,
        run_id: str,
        payload_path: str,
        export_paths: Optional[List[str]] = None,
        run_metadata: Optional[dict] = None,
    ) -> dict:
        run_data = {
            "run_id": run_id,
            "project_id": project_id,
            "payload_path": payload_path,
            "export_paths": export_paths or [],
            "run_metadata": run_metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._put_json(
            self._key("projects", project_id, "runs", f"{run_id}.json"),
            run_data,
        )
        return run_data

    def list_runs(self, project_id: str) -> List[dict]:
        prefix = self._key("projects", project_id, "runs") + "/"
        keys = [k for k in self._list_keys(prefix) if k.endswith(".json")]
        runs: List[dict] = []
        for key in keys:
            data = self._get_json(key)
            if data:
                runs.append(data)
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return runs

    # ── Profiles ─────────────────────────────────────────────────────────────

    def save_profile(self, owner_name: str, inputs: dict) -> str:
        key = self._key("profiles", f"{owner_name}.json")
        self._put_json(key, {
            "owner_name": owner_name,
            "inputs": inputs,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return key

    def load_profile(self, owner_name: str) -> Optional[dict]:
        data = self._get_json(self._key("profiles", f"{owner_name}.json"))
        if data:
            return data.get("inputs", data)
        return None

    def list_profiles(self) -> List[str]:
        prefix = self._key("profiles") + "/"
        keys = self._list_keys(prefix)
        return sorted(
            Path(k).stem for k in keys
            if k.endswith(".json")
        )

    # ── Collaboration ─────────────────────────────────────────────────────────

    def append_collaboration(self, entry: dict, project_id: str) -> str:
        """Append a JSONL entry to the project's collaboration log."""
        key = self._key("collab", project_id, "collaboration.jsonl")
        existing = b""
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
            existing = resp["Body"].read()
        except Exception:
            pass
        line = json.dumps(entry, default=str).encode("utf-8") + b"\n"
        self._s3().put_object(
            Bucket=self._bucket, Key=key,
            Body=existing + line,
            ContentType="application/x-ndjson",
        )
        return key

    def load_collaboration(self, project_id: str) -> List[dict]:
        key = self._key("collab", project_id, "collaboration.jsonl")
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
            raw = resp["Body"].read().decode("utf-8")
        except Exception:
            return []
        entries: List[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    # ── Feedback ─────────────────────────────────────────────────────────────

    def append_feedback(self, entry: dict, output_dir: str) -> str:
        dir_hash = hashlib.md5(output_dir.encode()).hexdigest()[:12]
        key = self._key("feedback", f"{dir_hash}.jsonl")
        existing = b""
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
            existing = resp["Body"].read()
        except Exception:
            pass
        line = json.dumps(entry, default=str).encode("utf-8") + b"\n"
        self._s3().put_object(
            Bucket=self._bucket, Key=key,
            Body=existing + line,
            ContentType="application/x-ndjson",
        )
        return key

    def load_feedback(self, output_dir: str) -> List[dict]:
        dir_hash = hashlib.md5(output_dir.encode()).hexdigest()[:12]
        key = self._key("feedback", f"{dir_hash}.jsonl")
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
            raw = resp["Body"].read().decode("utf-8")
        except Exception:
            return []
        entries: List[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    # ── Generic file I/O ─────────────────────────────────────────────────────

    def save_file(self, path: str, content: bytes) -> str:
        key = self._key("files", path)
        self._s3().put_object(Bucket=self._bucket, Key=key, Body=content)
        return key

    def load_file(self, path: str) -> Optional[bytes]:
        key = self._key("files", path)
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except Exception:
            return None

    def list_files(self, prefix: str) -> List[str]:
        return self._list_keys(self._key("files", prefix))

    def file_exists(self, path: str) -> bool:
        key = self._key("files", path)
        try:
            self._s3().head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False
