#!/usr/bin/env python3
"""
Pilot Batch Ingest — process India tender ZIPs or folders end-to-end.

For each tender (ZIP or folder):
  1. Extracts (ZIPs only) or scans (folders directly)
  2. Classifies files (drawings, BOQ, conditions, addenda, XLSX, DOC)
  3. Creates one xBOQ project per tender
  4. Runs pipeline in chosen mode (default: FULL_AUDIT)
  5. Saves run record + exports
  6. Appends row to pilot_scorecard.csv

Usage:
    # Folders (auto-detected)
    python3 scripts/pilot_batch_ingest.py \\
        --input /path/to/parent_dir --output /path/to/output --mode full_audit

    # ZIPs (auto-detected)
    python3 scripts/pilot_batch_ingest.py \\
        --input /path/to/zips/ --output ./out/pilot_batch

    # Dry run (classify only, no pipeline)
    python3 scripts/pilot_batch_ingest.py --input /path/ --dry-run

    # Limit to N tenders
    python3 scripts/pilot_batch_ingest.py --input /path/ --limit 1

    # Process specific tenders by name
    python3 scripts/pilot_batch_ingest.py --input /path/ --include "Tender Documents" "Tender Documents 3"

    # Force input type
    python3 scripts/pilot_batch_ingest.py --input /path/ --input-type folder
"""

import argparse
import csv
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix multiprocessing on macOS (OCR subprocess spawning)
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

# Repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pilot.file_router import (
    classify_directory,
    RoutingSummary,
    ClassifiedFile,
    CATEGORY_DRAWINGS_PDF,
    CATEGORY_BOQ_PDF,
    CATEGORY_CONDITIONS_PDF,
    CATEGORY_ADDENDA_PDF,
    CATEGORY_BOQ_XLSX,
    CATEGORY_CONDITIONS_DOC,
    ALL_CATEGORIES,
)

logger = logging.getLogger("pilot_batch_ingest")


# ═════════════════════════════════════════════════════════════════════════
# SCORECARD
# ═════════════════════════════════════════════════════════════════════════

SCORECARD_FIELDS = [
    "project_id",
    "tender_name",
    "pages_total",
    "deep_processed_pages",
    "ocr_pages",
    "boq_pages_attempted",
    "boq_pages_parsed",
    "boq_items_count",
    "schedule_pages_attempted",
    "schedule_pages_parsed",
    "finish_rows",
    "door_rows",
    "window_rows",
    "commercial_terms_count",
    "requirements_count",
    "rfis_count",
    "blockers_count",
    "conflicts_count",
    "toxic_pages_count",
    "runtime_sec",
    "qa_score",
    "selection_mode",
    "skipped_pages_count",
    "error",
]


def build_scorecard_row(
    project_id: str,
    tender_name: str,
    payload: Optional[Dict[str, Any]],
    runtime_sec: float,
    error: str = "",
) -> Dict[str, Any]:
    """
    Build a single scorecard row from a pipeline result payload.

    Works defensively — all fields default to 0/"" if payload is missing.
    """
    row: Dict[str, Any] = {f: "" for f in SCORECARD_FIELDS}
    row["project_id"] = project_id
    row["tender_name"] = tender_name
    row["runtime_sec"] = f"{runtime_sec:.1f}"
    row["error"] = error

    if not payload:
        return row

    # Processing stats
    ps = payload.get("processing_stats") or {}
    row["pages_total"] = ps.get("total_pages", 0)
    row["deep_processed_pages"] = ps.get("deep_processed_pages", 0)
    row["ocr_pages"] = ps.get("ocr_pages", 0)
    row["skipped_pages_count"] = ps.get("skipped_pages", 0)
    row["toxic_pages_count"] = ps.get("toxic_pages", 0)
    row["selection_mode"] = str(ps.get("selection_mode", ""))

    # Extraction diagnostics
    diag = payload.get("extraction_diagnostics") or {}
    boq_diag = diag.get("boq") or {}
    sched_diag = diag.get("schedules") or {}
    row["boq_pages_attempted"] = boq_diag.get("pages_attempted", 0)
    row["boq_pages_parsed"] = boq_diag.get("pages_parsed", 0)
    row["schedule_pages_attempted"] = sched_diag.get("pages_attempted", 0)
    row["schedule_pages_parsed"] = sched_diag.get("pages_parsed", 0)

    # BOQ items
    boq_items = payload.get("boq_items") or []
    row["boq_items_count"] = len(boq_items) if isinstance(boq_items, list) else 0

    # Schedules
    schedules = payload.get("schedules") or {}
    if isinstance(schedules, dict):
        row["door_rows"] = len(schedules.get("doors", []))
        row["window_rows"] = len(schedules.get("windows", []))
        row["finish_rows"] = len(schedules.get("finishes", []))
    else:
        row["door_rows"] = 0
        row["window_rows"] = 0
        row["finish_rows"] = 0

    # Commercial terms
    terms = payload.get("commercial_terms") or []
    row["commercial_terms_count"] = len(terms) if isinstance(terms, list) else 0

    # Requirements
    reqs = payload.get("requirements") or []
    row["requirements_count"] = len(reqs) if isinstance(reqs, list) else 0

    # RFIs and blockers
    rfis = payload.get("rfis") or []
    row["rfis_count"] = len(rfis) if isinstance(rfis, list) else 0
    blockers = payload.get("blockers") or []
    row["blockers_count"] = len(blockers) if isinstance(blockers, list) else 0

    # Conflicts
    conflicts = payload.get("conflicts") or payload.get("quantity_reconciliation") or []
    if isinstance(conflicts, list):
        row["conflicts_count"] = sum(
            1 for c in conflicts
            if isinstance(c, dict) and c.get("status") in ("conflict", "mismatch")
        )
    else:
        row["conflicts_count"] = 0

    # QA score
    qa = payload.get("qa_score") or {}
    if isinstance(qa, dict):
        row["qa_score"] = qa.get("score", "")
    elif isinstance(qa, (int, float)):
        row["qa_score"] = qa
    else:
        rs = payload.get("readiness_score")
        row["qa_score"] = rs if rs is not None else ""

    return row


def write_scorecard(
    rows: List[Dict[str, Any]],
    output_path: Path,
):
    """Write scorecard CSV (append-safe: writes header only if file is new)."""
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ═════════════════════════════════════════════════════════════════════════
# ZIP EXTRACTION
# ═════════════════════════════════════════════════════════════════════════

def extract_zip_safe(zip_path: Path, dest_dir: Path) -> Path:
    """Extract ZIP to dest_dir, handling nested dirs and encoding issues."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile as e:
        raise ValueError(f"Bad ZIP file: {zip_path.name}: {e}")
    return dest_dir


# ═════════════════════════════════════════════════════════════════════════
# CORE PIPELINE RUNNER (shared by ZIP and folder flows)
# ═════════════════════════════════════════════════════════════════════════

def _run_tender_pipeline(
    workdir: Path,
    tender_name: str,
    project_id: str,
    project_output: Path,
    run_mode: str,
    verbose: bool,
    dry_run: bool,
    source_label: str,
    t0: float,
) -> Dict[str, Any]:
    """
    Core pipeline runner: classify → (optionally) run pipeline → build scorecard.

    Used by both process_single_zip and process_single_folder.
    """
    # Step 1: Classify files in workdir
    routing = classify_directory(workdir, zip_name=tender_name)
    logger.info(f"  Classified {routing.total_files} files")

    if verbose:
        print(routing.summary_table())
        print()

    # Save routing summary
    project_output.mkdir(parents=True, exist_ok=True)
    _save_routing_json(routing, project_output / "file_routing.json")

    if dry_run:
        elapsed = time.perf_counter() - t0
        row = build_scorecard_row(project_id, tender_name, None, elapsed)
        row["pages_total"] = f"(dry-run: {len(routing.all_pdf_files)} PDFs, {routing.total_files} total files)"
        return row

    # Step 2: Collect ALL PDFs for pipeline (don't drop unknown-category PDFs)
    pdf_files = routing.all_pdf_files
    if not pdf_files:
        raise ValueError(f"No PDF files found in {tender_name}")

    # Sprint 21C: Collect Excel BOQ files for structured BOQ parsing
    boq_excel_files = [
        cf.path for cf in routing.by_category.get(CATEGORY_BOQ_XLSX, [])
    ]
    if boq_excel_files:
        logger.info(f"  Found {len(boq_excel_files)} Excel BOQ file(s)")

    logger.info(f"  Running pipeline on {len(pdf_files)} PDFs in {run_mode} mode...")

    # Step 3: Create project record
    from src.analysis.projects import create_project
    proj_meta = create_project(
        name=tender_name,
        project_id=project_id,
        pilot_mode=True,
        notes=f"Pilot batch ingest from {source_label}",
    )

    # Step 4: Run pipeline
    from src.analysis.pipeline import run_analysis_pipeline

    def progress_cb(stage_id: str, message: str, progress: float):
        if verbose and (progress >= 1.0 or "selected" in message.lower()):
            logger.info(f"  [{stage_id}] {message}")

    result = run_analysis_pipeline(
        input_files=pdf_files,
        project_id=project_id,
        output_dir=project_output,
        progress_callback=progress_cb,
        run_mode=run_mode,
        boq_excel_paths=boq_excel_files or None,
    )

    elapsed = time.perf_counter() - t0

    if not result.success:
        row = build_scorecard_row(
            project_id, tender_name, result.payload, elapsed,
            error=result.error_message or "pipeline failed",
        )
        logger.error(f"  FAILED: {result.error_message}")
        return row

    # Step 5: Save run record
    from src.analysis.projects import save_run
    try:
        analysis_path = project_output / "analysis.json"
        save_run(
            project_id=project_id,
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            payload_path=str(analysis_path),
            export_paths=[str(f) for f in result.files_generated],
            run_metadata={
                "run_mode": run_mode,
                "source": source_label,
                "duration_sec": elapsed,
            },
        )
    except Exception:
        pass  # Non-critical

    # Step 6: Write structured run artifacts (non-critical)
    try:
        from src.ops.run_artifacts import write_all_artifacts
        write_all_artifacts(
            result.payload or {},
            project_output / "run_artifacts",
            extra={"tender_name": tender_name, "run_mode": run_mode, "source": source_label},
        )
    except Exception:
        pass  # Non-critical — don't break pipeline on artifact write failure

    # Step 7: Build scorecard row
    row = build_scorecard_row(project_id, tender_name, result.payload, elapsed)
    logger.info(f"  SUCCESS in {elapsed:.1f}s — {row.get('pages_total', '?')} pages")
    return row


# ═════════════════════════════════════════════════════════════════════════
# SINGLE TENDER PROCESSING — ZIP
# ═════════════════════════════════════════════════════════════════════════

def process_single_zip(
    zip_path: Path,
    output_dir: Path,
    run_mode: str = "full_audit",
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Process one tender ZIP end-to-end.

    Returns a scorecard row dict.
    """
    tender_name = zip_path.stem
    project_id = f"pilot_{tender_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_output = output_dir / project_id
    workdir = project_output / "extracted"

    logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Processing ZIP: {zip_path.name}")

    t0 = time.perf_counter()

    try:
        # Extract ZIP
        extract_zip_safe(zip_path, workdir)

        return _run_tender_pipeline(
            workdir=workdir,
            tender_name=tender_name,
            project_id=project_id,
            project_output=project_output,
            run_mode=run_mode,
            verbose=verbose,
            dry_run=dry_run,
            source_label=f"zip:{zip_path.name}",
            t0=t0,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"  ERROR: {error_msg}")
        if verbose:
            traceback.print_exc()
        return build_scorecard_row(project_id, tender_name, None, elapsed, error=error_msg)


# ═════════════════════════════════════════════════════════════════════════
# SINGLE TENDER PROCESSING — FOLDER
# ═════════════════════════════════════════════════════════════════════════

def process_single_folder(
    folder_path: Path,
    output_dir: Path,
    run_mode: str = "full_audit",
    verbose: bool = False,
    dry_run: bool = False,
    tenant: str = "",
) -> Dict[str, Any]:
    """
    Process one tender folder end-to-end (no extraction needed).

    Returns a scorecard row dict.
    """
    tender_name = folder_path.name
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = tender_name.replace(" ", "_").replace("/", "_")
    prefix = f"{tenant}_" if tenant else "pilot_"
    project_id = f"{prefix}{safe_name}_{ts}"
    project_output = output_dir / project_id

    logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Processing folder: {tender_name}")

    t0 = time.perf_counter()

    try:
        # Folder IS the workdir — classify directly (no extraction)
        return _run_tender_pipeline(
            workdir=folder_path,
            tender_name=tender_name,
            project_id=project_id,
            project_output=project_output,
            run_mode=run_mode,
            verbose=verbose,
            dry_run=dry_run,
            source_label=f"folder:{tender_name}",
            t0=t0,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"  ERROR: {error_msg}")
        if verbose:
            traceback.print_exc()
        return build_scorecard_row(project_id, tender_name, None, elapsed, error=error_msg)


def _save_routing_json(routing: RoutingSummary, path: Path):
    """Save routing summary as JSON for audit trail."""
    data = {
        "zip_name": routing.zip_name,
        "total_files": routing.total_files,
        "files": [
            {
                "path": str(cf.path),
                "name": cf.path.name,
                "category": cf.category,
                "reason": cf.reason,
                "size_bytes": cf.size_bytes,
            }
            for cf in routing.classified
        ],
        "summary": {
            cat: len(files)
            for cat, files in routing.by_category.items()
            if files
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════
# INPUT DISCOVERY
# ═════════════════════════════════════════════════════════════════════════

def find_zips(input_dir: Path, only: Optional[str] = None) -> List[Path]:
    """Find ZIP files in input_dir."""
    zips = sorted(input_dir.glob("*.zip"))
    if only:
        zips = [z for z in zips if z.name == only or z.stem == only]
    return zips


def find_folders(
    input_dir: Path,
    only: Optional[List[str]] = None,
) -> List[Path]:
    """Find tender subdirectories in input_dir (non-hidden, non-OS)."""
    _skip = {".git", "__MACOSX", "__pycache__", ".DS_Store", "node_modules"}
    folders = sorted(
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name not in _skip and not p.name.startswith(".")
    )
    if only:
        only_lower = {n.lower() for n in only}
        folders = [f for f in folders if f.name.lower() in only_lower]
    return folders


def find_inputs(
    input_dir: Path,
    input_type: str = "auto",
    only: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> tuple:
    """
    Discover inputs (ZIPs or folders) based on input_type.

    Returns: (items: List[Path], kind: str)  where kind is "zip" or "folder".
    """
    if input_type == "zip":
        only_str = only[0] if only and len(only) == 1 else None
        items = find_zips(input_dir, only_str)
        kind = "zip"
    elif input_type == "folder":
        items = find_folders(input_dir, only)
        kind = "folder"
    else:  # auto
        zips = find_zips(input_dir)
        folders = find_folders(input_dir, only)
        if zips and not folders:
            only_str = only[0] if only and len(only) == 1 else None
            items = find_zips(input_dir, only_str)
            kind = "zip"
        elif folders:
            items = folders
            kind = "folder"
        else:
            items = []
            kind = "none"

    if limit and limit > 0:
        items = items[:limit]

    return items, kind


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pilot Batch Ingest — process India tender ZIPs or folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "--input-dir", "-i", type=Path, required=True, dest="input_dir",
        help="Directory containing tender ZIPs or subfolders",
    )
    parser.add_argument(
        "--output", "--output-dir", "-o", type=Path, default=None, dest="output_dir",
        help="Output directory (default: out/pilot_batch_<timestamp>)",
    )
    parser.add_argument(
        "--mode", "--run-mode", "-m", default="full_audit", dest="run_mode",
        choices=["demo_fast", "standard_review", "full_audit"],
        help="Pipeline run mode (default: full_audit)",
    )
    parser.add_argument(
        "--input-type", default="auto",
        choices=["auto", "zip", "folder"],
        help="Input type: auto-detect, force zip, or force folder (default: auto)",
    )
    parser.add_argument(
        "--tenant", "-t", type=str, default="",
        help="Tenant/client name prefix for project IDs",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Process only this single item (ZIP filename or folder name)",
    )
    parser.add_argument(
        "--include", nargs="+", type=str, default=None,
        help="Process only these items (multiple names allowed)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N tenders",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify files only — skip pipeline analysis",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    # Merge --only and --include into a single filter list
    filter_names = None
    if args.include:
        filter_names = args.include
    elif args.only:
        filter_names = [args.only]

    # Discover inputs
    items, kind = find_inputs(
        args.input_dir,
        input_type=args.input_type,
        only=filter_names,
        limit=args.limit,
    )

    if not items:
        logger.error(f"No inputs found in {args.input_dir} (type={args.input_type})")
        logger.error(f"  Checked for: ZIPs and subdirectories")
        return 1

    logger.info(f"Found {len(items)} {kind}(s) to process")
    for item in items:
        logger.info(f"  → {item.name}")

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tenant_tag = f"_{args.tenant}" if args.tenant else ""
    output_dir = args.output_dir or (PROJECT_ROOT / "out" / f"pilot_batch{tenant_tag}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "pilot_scorecard.csv"

    # Process each tender
    scorecard_rows: List[Dict[str, Any]] = []
    successes = 0
    failures = 0

    for i, item_path in enumerate(items, 1):
        logger.info(f"\n[{i}/{len(items)}] {item_path.name}")
        logger.info("=" * 60)

        if kind == "zip":
            row = process_single_zip(
                zip_path=item_path,
                output_dir=output_dir,
                run_mode=args.run_mode,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
        else:
            row = process_single_folder(
                folder_path=item_path,
                output_dir=output_dir,
                run_mode=args.run_mode,
                verbose=args.verbose,
                dry_run=args.dry_run,
                tenant=args.tenant,
            )

        scorecard_rows.append(row)

        if row.get("error"):
            failures += 1
        else:
            successes += 1

        # Write scorecard incrementally (so partial results survive crashes)
        write_scorecard([row], scorecard_path)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"BATCH COMPLETE: {successes} succeeded, {failures} failed out of {len(items)}")
    logger.info(f"Scorecard: {scorecard_path}")
    logger.info(f"Output:    {output_dir}")

    if args.dry_run:
        logger.info("(Dry-run mode — no pipeline analysis was performed)")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
