#!/usr/bin/env python3
"""
Rules Snapshot Tool

Creates versioned snapshots of all rule files for reproducibility.

Usage:
    python scripts/snapshot_rules.py --tag v1
    python scripts/snapshot_rules.py --tag 2024-02-03_pre_villa

Creates:
    rules/snapshots/<tag>/
        - copies of all .yaml files from rules/
        - snapshot_manifest.json with metadata
"""

import argparse
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
RULES_DIR = PROJECT_ROOT / "rules"
SNAPSHOTS_DIR = RULES_DIR / "snapshots"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:12]


def create_snapshot(tag: str, overwrite: bool = False) -> Path:
    """
    Create a snapshot of all rule files.

    Args:
        tag: Snapshot tag (e.g., v1, v2, 2024-02-03)
        overwrite: Overwrite existing snapshot

    Returns:
        Path to snapshot directory
    """
    snapshot_dir = SNAPSHOTS_DIR / tag

    if snapshot_dir.exists() and not overwrite:
        print(f"ERROR: Snapshot '{tag}' already exists at {snapshot_dir}")
        print("Use --overwrite to replace existing snapshot")
        sys.exit(1)

    # Create snapshot directory
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Find all rule files
    rule_files = list(RULES_DIR.glob("*.yaml")) + list(RULES_DIR.glob("*.yml"))

    # Copy files and build manifest
    manifest = {
        "tag": tag,
        "created_at": datetime.now().isoformat(),
        "source_dir": str(RULES_DIR),
        "files": [],
    }

    for rule_file in rule_files:
        # Skip snapshots directory
        if "snapshots" in str(rule_file):
            continue

        dest = snapshot_dir / rule_file.name
        shutil.copy2(rule_file, dest)

        file_hash = compute_file_hash(rule_file)
        file_info = {
            "name": rule_file.name,
            "hash": file_hash,
            "size_bytes": rule_file.stat().st_size,
            "modified": datetime.fromtimestamp(rule_file.stat().st_mtime).isoformat(),
        }
        manifest["files"].append(file_info)

        print(f"  ✓ {rule_file.name} ({file_hash})")

    # Write manifest
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"✅ Snapshot created: {snapshot_dir}")
    print(f"   Files: {len(manifest['files'])}")
    print(f"   Manifest: {manifest_path}")

    return snapshot_dir


def list_snapshots() -> None:
    """List all existing snapshots."""
    if not SNAPSHOTS_DIR.exists():
        print("No snapshots found.")
        return

    snapshots = sorted(SNAPSHOTS_DIR.iterdir())
    if not snapshots:
        print("No snapshots found.")
        return

    print("Available snapshots:")
    for snap in snapshots:
        if snap.is_dir():
            manifest_path = snap / "snapshot_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                files_count = len(manifest.get("files", []))
                created = manifest.get("created_at", "unknown")[:10]
                print(f"  - {snap.name} ({files_count} files, {created})")
            else:
                print(f"  - {snap.name} (no manifest)")


def main():
    parser = argparse.ArgumentParser(
        description="Create versioned snapshots of rule files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/snapshot_rules.py --tag v1
    python scripts/snapshot_rules.py --tag 2024-02-03_baseline
    python scripts/snapshot_rules.py --list
        """
    )

    parser.add_argument(
        "--tag", "-t",
        help="Snapshot tag name"
    )

    parser.add_argument(
        "--overwrite", "-o",
        action="store_true",
        help="Overwrite existing snapshot"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List existing snapshots"
    )

    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return

    if not args.tag:
        print("ERROR: --tag is required (or use --list to see existing snapshots)")
        sys.exit(1)

    print(f"Creating snapshot: {args.tag}")
    print("-" * 40)
    create_snapshot(args.tag, args.overwrite)


if __name__ == "__main__":
    main()
