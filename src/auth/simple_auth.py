"""
SimpleAuth — password-based authentication with tenant isolation.

Stores tenant configs in <auth_dir>/tenants.json. Each tenant gets an
isolated LocalStorage directory under ~/.xboq/tenant_<id>/.

Sprint 16: Hosted pilot readiness.
"""

import hashlib
import hmac
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SimpleAuth:
    """Simple password-based auth for pilot deployment.

    Tenant config schema (tenants.json):
    {
        "<tenant_id>": {
            "name": "Acme Builders",
            "password_hash": "<sha256_hex>",
            "salt": "<hex_salt>",
            "created_at": "<iso_timestamp>",
            "storage_prefix": "tenant_<tenant_id>"
        }
    }
    """

    def __init__(self, auth_dir: Optional[str] = None):
        if auth_dir is None:
            self._auth_dir = Path.home() / ".xboq" / "auth"
        else:
            self._auth_dir = Path(auth_dir)

        self._auth_dir.mkdir(parents=True, exist_ok=True)
        self._tenants_path = self._auth_dir / "tenants.json"

        if not self._tenants_path.exists():
            self._save_tenants({})

    # ── Public API ──────────────────────────────────────────────────────

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        password: str,
    ) -> dict:
        """Create a new tenant. Returns tenant config (without password_hash).

        Args:
            tenant_id: Unique tenant identifier (alphanumeric + underscores).
            name: Human-readable tenant/company name.
            password: Plaintext password (will be hashed).

        Returns:
            Dict with tenant_id, name, created_at, storage_prefix.

        Raises:
            ValueError: If tenant_id already exists or is empty.
        """
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")

        tenants = self._load_tenants()
        if tenant_id in tenants:
            raise ValueError(f"Tenant '{tenant_id}' already exists")

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        storage_prefix = f"tenant_{tenant_id}"

        tenant_data = {
            "name": name.strip(),
            "password_hash": password_hash,
            "salt": salt,
            "created_at": datetime.now().isoformat(),
            "storage_prefix": storage_prefix,
        }

        tenants[tenant_id] = tenant_data
        self._save_tenants(tenants)

        return {
            "tenant_id": tenant_id,
            "name": tenant_data["name"],
            "created_at": tenant_data["created_at"],
            "storage_prefix": storage_prefix,
        }

    def authenticate(
        self,
        tenant_id: str,
        password: str,
    ) -> Optional[dict]:
        """Verify credentials. Returns tenant config (no hash) or None.

        Args:
            tenant_id: Tenant identifier.
            password: Plaintext password to verify.

        Returns:
            Tenant config dict if valid, None if invalid.
        """
        tenants = self._load_tenants()
        tenant = tenants.get(tenant_id)
        if tenant is None:
            return None

        expected_hash = tenant.get("password_hash", "")
        salt = tenant.get("salt", "")
        computed_hash = self._hash_password(password, salt)

        if not hmac.compare_digest(expected_hash, computed_hash):
            return None

        return {
            "tenant_id": tenant_id,
            "name": tenant.get("name", ""),
            "created_at": tenant.get("created_at", ""),
            "storage_prefix": tenant.get("storage_prefix", ""),
        }

    def get_tenant(self, tenant_id: str) -> Optional[dict]:
        """Load tenant config (no password_hash). Returns None if not found."""
        tenants = self._load_tenants()
        tenant = tenants.get(tenant_id)
        if tenant is None:
            return None

        return {
            "tenant_id": tenant_id,
            "name": tenant.get("name", ""),
            "created_at": tenant.get("created_at", ""),
            "storage_prefix": tenant.get("storage_prefix", ""),
        }

    def list_tenants(self) -> List[dict]:
        """Return list of tenant configs (no password hashes), sorted by name."""
        tenants = self._load_tenants()
        result = []
        for tid, data in tenants.items():
            result.append({
                "tenant_id": tid,
                "name": data.get("name", ""),
                "created_at": data.get("created_at", ""),
            })
        return sorted(result, key=lambda t: t.get("name", "").lower())

    def reset_password(self, tenant_id: str, new_password: str) -> bool:
        """Reset password for an existing tenant. Returns True on success, False if not found."""
        tenants = self._load_tenants()
        if tenant_id not in tenants:
            return False

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(new_password, salt)
        tenants[tenant_id]["password_hash"] = password_hash
        tenants[tenant_id]["salt"] = salt
        tenants[tenant_id]["password_reset_at"] = datetime.now().isoformat()
        self._save_tenants(tenants)
        return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant from the registry. Returns True on success, False if not found.
        NOTE: Does NOT delete the tenant's data directory."""
        tenants = self._load_tenants()
        if tenant_id not in tenants:
            return False
        del tenants[tenant_id]
        self._save_tenants(tenants)
        return True

    def get_storage_for_tenant(self, tenant_id: str) -> "LocalStorage":
        """Return a tenant-scoped LocalStorage instance.

        Each tenant gets its own base_dir: ~/.xboq/<storage_prefix>/

        Returns:
            LocalStorage with isolated base_dir.

        Raises:
            ValueError: If tenant_id not found.
        """
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        from src.storage.local import LocalStorage
        base_dir = self._auth_dir.parent / tenant["storage_prefix"]
        return LocalStorage(base_dir=str(base_dir))

    # ── Internal ────────────────────────────────────────────────────────

    def _load_tenants(self) -> dict:
        """Load tenants.json. Returns empty dict on error."""
        if not self._tenants_path.exists():
            return {}
        try:
            with open(self._tenants_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_tenants(self, data: dict) -> None:
        """Write tenants.json."""
        self._auth_dir.mkdir(parents=True, exist_ok=True)
        with open(self._tenants_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2-SHA256 (100k iterations). Returns hex digest."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        ).hex()
