#!/usr/bin/env python3
"""
Prewarm demo cache — run pipeline on all demo PDFs and save to demo_cache/.

Usage:
    python3 scripts/prewarm_demo_cache.py [--project_id <id>] [--dry-run]

If --project_id is given, only prewarm that project.
Otherwise, prewarm all demo projects.

Exit code:
    0 = all projects prewarmed (or dry-run)
    1 = at least one failure
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def prewarm_project(project_id: str, dry_run: bool = False) -> bool:
    """
    Run pipeline for a single demo project and save to demo_cache/.

    Returns True on success, False on failure.
    """
    from src.demo.demo_config import get_demo_project
    from src.demo.demo_assets import resolve_demo_pdf, resolve_demo_cache

    proj = get_demo_project(project_id)
    if not proj:
        print(f"  ERROR: Unknown project_id: {project_id}")
        return False

    # Check if already cached
    existing = resolve_demo_cache(project_id)
    if existing:
        print(f"  CACHED: {existing}")
        return True

    pdf_path = resolve_demo_pdf(project_id, proj["asset_filename"])
    if not pdf_path:
        print(f"  SKIP: PDF not found for {project_id} ({proj['asset_filename']})")
        print(f"         Expected at: demo_inputs/{proj['asset_filename']}")
        return False

    output_dir = PROJECT_ROOT / "demo_cache" / project_id

    if dry_run:
        print(f"  DRY-RUN: would run pipeline on {pdf_path.name} -> {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Running pipeline: {pdf_path.name} -> {output_dir}")
    start = time.time()

    try:
        from src.analysis.pipeline import run_analysis_pipeline
        result = run_analysis_pipeline(
            input_files=[pdf_path],
            project_id=project_id,
            output_dir=output_dir,
        )
        elapsed = time.time() - start

        if result.success and result.payload:
            analysis_path = output_dir / "analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(result.payload, f, indent=2, default=str)
            print(f"  OK: {analysis_path} ({elapsed:.1f}s)")
            return True
        else:
            print(f"  FAIL: Pipeline returned success={result.success}")
            if result.error_message:
                print(f"        Error: {result.error_message[:200]}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prewarm demo cache")
    parser.add_argument("--project_id", type=str, default=None,
                        help="Specific project to prewarm (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check assets, don't run pipeline")
    args = parser.parse_args()

    from src.demo.demo_config import get_demo_project_ids

    project_ids = [args.project_id] if args.project_id else get_demo_project_ids()
    results = {}

    print(f"Prewarming {len(project_ids)} demo project(s){'  [DRY RUN]' if args.dry_run else ''}...")
    print()

    for pid in project_ids:
        print(f"[{pid}]")
        results[pid] = prewarm_project(pid, dry_run=args.dry_run)
        print()

    # Summary
    ok = sum(1 for v in results.values() if v)
    fail = sum(1 for v in results.values() if not v)
    print(f"Done: {ok} ok, {fail} skipped/failed")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
