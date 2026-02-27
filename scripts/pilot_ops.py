#!/usr/bin/env python3
"""
Pilot Ops — thin orchestrator around existing pilot tools.

Supports four modes:
    inventory   — scan folders for metadata (no OCR / no pipeline)
    dry-run     — classify files without running the pipeline
    run         — run pilot batch ingest in selected mode
    summary     — read pilot_scorecard.csv and print top findings

Usage:
    python scripts/pilot_ops.py --input ~/tenders/ --mode inventory
    python scripts/pilot_ops.py --input ~/tenders/ --mode dry-run
    python scripts/pilot_ops.py --input ~/tenders/ --mode full_audit --output ~/output/
    python scripts/pilot_ops.py --input ~/output/ --mode summary
    python scripts/pilot_ops.py --help

This script does NOT replace existing scripts; it orchestrates them.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Mode: inventory
# ---------------------------------------------------------------------------

def run_inventory(args: argparse.Namespace) -> int:
    """Run pilot_inventory.py on the input directory."""
    inventory_script = SCRIPTS_DIR / "pilot_inventory.py"

    if not inventory_script.exists():
        # Fallback: simple inline inventory
        print("pilot_inventory.py not found — running inline folder scan.")
        return _inline_inventory(args)

    cmd = [sys.executable, str(inventory_script), "--input", str(args.input)]
    # pilot_inventory.py requires --output; default to <input>_inventory if not set
    output = args.output or str(Path(args.input).resolve().parent / (Path(args.input).name + "_inventory"))
    cmd += ["--output", output]
    if args.include:
        for inc in args.include:
            cmd += ["--include", inc]

    print(f"Running: {' '.join(cmd)}")
    print()
    return subprocess.call(cmd)


def _inline_inventory(args: argparse.Namespace) -> int:
    """Minimal inline inventory when pilot_inventory.py is unavailable."""
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return 1

    print(f"Scanning: {input_dir}")
    print()

    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        files = list(folder.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        pdf_count = sum(1 for f in files if f.is_file() and f.suffix.lower() == ".pdf")
        xls_count = sum(1 for f in files if f.is_file() and f.suffix.lower() in (".xls", ".xlsx"))
        print(f"  {folder.name}: {file_count} files, {total_mb:.1f} MB, {pdf_count} PDFs, {xls_count} Excel")

    print()
    return 0


# ---------------------------------------------------------------------------
# Mode: dry-run
# ---------------------------------------------------------------------------

def run_dry_run(args: argparse.Namespace) -> int:
    """Run pilot_batch_ingest.py in dry-run mode."""
    ingest_script = SCRIPTS_DIR / "pilot_batch_ingest.py"

    if not ingest_script.exists():
        print(f"Error: {ingest_script} not found")
        return 1

    cmd = [
        sys.executable, str(ingest_script),
        "--input", str(args.input),
        "--input-type", "folder",
        "--dry-run",
    ]
    if args.tenant:
        cmd += ["--tenant", args.tenant]
    if args.include:
        for inc in args.include:
            cmd += ["--include", inc]

    print(f"Running: {' '.join(cmd)}")
    print()
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Mode: run (full pipeline)
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    """Run pilot_batch_ingest.py in the selected mode."""
    ingest_script = SCRIPTS_DIR / "pilot_batch_ingest.py"

    if not ingest_script.exists():
        print(f"Error: {ingest_script} not found")
        return 1

    run_mode = args.mode if args.mode not in ("run", "inventory", "dry-run", "summary") else "full_audit"

    cmd = [
        sys.executable, str(ingest_script),
        "--input", str(args.input),
        "--input-type", "folder",
        "--mode", run_mode,
    ]
    if args.output:
        cmd += ["--output", str(args.output)]
    if args.tenant:
        cmd += ["--tenant", args.tenant]
    if args.include:
        for inc in args.include:
            cmd += ["--include", inc]

    print(f"Running: {' '.join(cmd)}")
    print()
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Mode: summary
# ---------------------------------------------------------------------------

def run_summary(args: argparse.Namespace) -> int:
    """Read pilot_scorecard.csv and print top findings."""
    input_dir = Path(args.input)

    # Find scorecard
    scorecard_path = None
    for candidate in [
        input_dir / "pilot_scorecard.csv",
        input_dir / "scorecard.csv",
    ]:
        if candidate.exists():
            scorecard_path = candidate
            break

    if not scorecard_path:
        # Search recursively
        found = list(input_dir.rglob("pilot_scorecard.csv"))
        if found:
            scorecard_path = found[0]

    if not scorecard_path:
        print(f"No pilot_scorecard.csv found in {input_dir}")
        print("Run the pipeline first with: python scripts/pilot_ops.py --input <dir> --mode full_audit")
        return 0

    print(f"Reading: {scorecard_path}")
    print()

    # Parse scorecard
    rows = []
    try:
        with open(scorecard_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error reading scorecard: {e}")
        return 1

    if not rows:
        print("Scorecard is empty.")
        return 0

    # Print summary table
    print("=" * 70)
    print(f"Pilot Scorecard — {len(rows)} tender(s)")
    print("=" * 70)
    print()

    header = f"{'Tender':<30} {'Status':<10} {'BOQ':>5} {'RFIs':>5} {'QA':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))

    for row in rows:
        name = row.get("tender_name", row.get("zip_name", "?"))[:30]
        status = row.get("status", "?")
        boq = row.get("boq_items", "?")
        rfis = row.get("rfis", "?")
        qa = row.get("qa_score", "?")
        if qa and qa != "?" and qa != "None":
            try:
                qa = f"{float(qa):.2f}"
            except (ValueError, TypeError):
                pass
        time_s = row.get("duration_sec", "?")
        if time_s and time_s != "?" and time_s != "None":
            try:
                time_s = f"{float(time_s):.0f}s"
            except (ValueError, TypeError):
                pass

        print(f"{name:<30} {status:<10} {str(boq):>5} {str(rfis):>5} {str(qa):>6} {str(time_s):>6}")

    print()

    # Top failure patterns
    errors = [r for r in rows if r.get("status") in ("ERROR", "PIPELINE_FAIL")]
    zero_boq = [r for r in rows if r.get("boq_items") in ("0", 0, "")]

    if errors:
        print(f"Top Issue 1: {len(errors)} tender(s) with errors")
        for r in errors[:3]:
            name = r.get("tender_name", r.get("zip_name", "?"))
            err = r.get("error", "unknown")[:80]
            print(f"  - {name}: {err}")
        print()

    if zero_boq:
        print(f"Top Issue 2: {len(zero_boq)} tender(s) with 0 BOQ items")
        for r in zero_boq[:3]:
            name = r.get("tender_name", r.get("zip_name", "?"))
            print(f"  - {name}")
        print()

    # Ranking by QA score
    scored = []
    for r in rows:
        qa_val = r.get("qa_score")
        if qa_val and qa_val not in ("?", "None", ""):
            try:
                scored.append((float(qa_val), r))
            except (ValueError, TypeError):
                pass

    if scored:
        scored.sort(key=lambda x: x[0])
        print("Tender ranking by QA score:")
        for i, (qa_val, r) in enumerate(scored, 1):
            name = r.get("tender_name", r.get("zip_name", "?"))
            print(f"  {i}. {name} — QA: {qa_val:.2f}")
        print()

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot Ops — orchestrator for pilot tender workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pilot_ops.py --input ~/tenders/ --mode inventory
  python scripts/pilot_ops.py --input ~/tenders/ --mode dry-run
  python scripts/pilot_ops.py --input ~/tenders/ --output ~/output/ --mode full_audit
  python scripts/pilot_ops.py --input ~/output/ --mode summary
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory (tender folders for inventory/dry-run/run, or output dir for summary)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (for inventory and run modes)",
    )
    parser.add_argument(
        "--tenant", default=None,
        help="Tenant identifier for pilot runs",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["inventory", "dry-run", "run", "summary",
                 "demo_fast", "standard_review", "full_audit"],
        help="Operation mode",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="force_dry_run",
        help="Force dry-run (equivalent to --mode dry-run)",
    )
    parser.add_argument(
        "--include", action="append", default=[],
        help="Include only specified tender folder names (can repeat)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary after run (equivalent to --mode summary)",
    )

    args = parser.parse_args()

    # Handle shortcut flags
    if args.force_dry_run:
        args.mode = "dry-run"

    mode = args.mode

    if mode == "inventory":
        rc = run_inventory(args)
    elif mode == "dry-run":
        rc = run_dry_run(args)
    elif mode == "summary":
        rc = run_summary(args)
    else:
        # run, demo_fast, standard_review, full_audit
        rc = run_pipeline(args)

        # Auto-summary after pipeline run if --summary flag given
        if args.summary and rc == 0:
            summary_args = argparse.Namespace(**vars(args))
            summary_args.input = args.output or args.input
            run_summary(summary_args)

    return rc


if __name__ == "__main__":
    sys.exit(main())
