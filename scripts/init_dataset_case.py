#!/usr/bin/env python3
"""
Initialize a dataset case directory for ground-truth capture.

Creates the standard folder structure:

    datasets/<tenant>/<project>/<run>/
        inputs/
        outputs/
        playbook/
        ground_truth/
        feedback/

Does NOT move or copy files — just scaffolds the directories.

Usage:
    python scripts/init_dataset_case.py --tenant acme --project hospital_300pg
    python scripts/init_dataset_case.py --tenant acme --project hospital_300pg --run 2026-02-27_full_audit
    python scripts/init_dataset_case.py --help
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"

SUBDIRS = ["inputs", "outputs", "playbook", "ground_truth", "feedback"]


def init_case(
    tenant: str,
    project: str,
    run: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Create dataset case directory structure. Returns the case root path."""
    base = base_dir or DATASETS_DIR
    if run is None:
        run = date.today().isoformat()

    case_root = base / tenant / project / run
    for subdir in SUBDIRS:
        (case_root / subdir).mkdir(parents=True, exist_ok=True)

    return case_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a dataset case directory for ground-truth capture."
    )
    parser.add_argument("--tenant", required=True, help="Tenant / company name")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument(
        "--run",
        default=None,
        help="Run identifier (default: today's date YYYY-MM-DD)",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DATASETS_DIR),
        help=f"Base directory for datasets (default: {DATASETS_DIR})",
    )
    args = parser.parse_args()

    case_root = init_case(
        tenant=args.tenant,
        project=args.project,
        run=args.run,
        base_dir=Path(args.base_dir),
    )

    print(f"Dataset case initialized at: {case_root}")
    print()
    print("Directory structure:")
    for subdir in SUBDIRS:
        print(f"  {case_root / subdir}/")
    print()
    print("Next steps:")
    print(f"  1. Copy/link input tender files to:   {case_root / 'inputs'}/")
    print(f"  2. Copy pipeline output to:           {case_root / 'outputs'}/")
    print(f"  3. Copy playbook snapshot to:         {case_root / 'playbook'}/")
    print(f"  4. Add ground truth (official BOQ) to: {case_root / 'ground_truth'}/")
    print(f"  5. Copy feedback.jsonl to:            {case_root / 'feedback'}/")


if __name__ == "__main__":
    main()
