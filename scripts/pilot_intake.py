#!/usr/bin/env python3
"""
Pilot intake: unzip a customer zip and run the demo analysis pipeline.

Use when a pilot sends 7 drawing sets + document zip. Extracts the zip,
finds all PDFs, and runs one analysis run (one project_id = one tender).

Usage:
    python scripts/pilot_intake.py --zip /path/to/pilot_delivery.zip --project_id pilot_abc_01
    python scripts/pilot_intake.py --zip /path/to/pilot_delivery.zip --project_id pilot_abc_01 --output ./out/pilot_abc_01
    python scripts/pilot_intake.py --dir /path/to/already_extracted/ --project_id pilot_abc_01

Output: out/<project_id>/analysis.json and full pipeline outputs.
"""

import argparse
import sys
import zipfile
from pathlib import Path

# Repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    """Extract zip into out_dir. Returns out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def collect_pdfs(root: Path, recursive: bool = True) -> list[Path]:
    """Collect all PDF paths under root. If recursive, search subdirs."""
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []
    if recursive:
        return sorted(root.rglob("*.pdf"))
    return sorted(root.glob("*.pdf"))


def main():
    parser = argparse.ArgumentParser(
        description="Pilot intake: unzip and run analysis on all PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--zip", "-z", type=Path, help="Path to pilot zip file")
    parser.add_argument("--dir", "-d", type=Path, help="Path to already-extracted folder (skip unzip)")
    parser.add_argument("--project_id", "-p", required=True, help="Project ID for this run")
    parser.add_argument("--output", "-o", type=Path, help="Output directory (default: out/<project_id>)")
    parser.add_argument("--no-recursive", action="store_true", help="Only look for PDFs in top level of dir")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose progress")
    args = parser.parse_args()

    if not args.zip and not args.dir:
        parser.error("Provide either --zip or --dir")
    if args.zip and args.dir:
        parser.error("Provide only one of --zip or --dir")

    output_dir = args.output or (PROJECT_ROOT / "out" / args.project_id)

    if args.zip:
        if not args.zip.exists():
            print(f"ERROR: Zip not found: {args.zip}")
            return 1
        extract_dir = output_dir / "extracted"
        if args.verbose:
            print(f"Extracting {args.zip} to {extract_dir} ...")
        extract_zip(args.zip, extract_dir)
        pdf_root = extract_dir
    else:
        if not args.dir.exists():
            print(f"ERROR: Dir not found: {args.dir}")
            return 1
        pdf_root = args.dir

    pdfs = collect_pdfs(pdf_root, recursive=not args.no_recursive)
    if not pdfs:
        print("ERROR: No PDFs found.")
        return 1
    if args.verbose:
        print(f"Found {len(pdfs)} PDF(s):")
        for p in pdfs[:20]:
            try:
                rel = p.relative_to(pdf_root)
            except ValueError:
                rel = p.name
            print(f"  - {rel}")
        if len(pdfs) > 20:
            print(f"  ... and {len(pdfs) - 20} more")

    # Run pipeline (multi-PDF supported; run_demo_analysis only globs top-level in dir, so call pipeline directly)
    from src.analysis.pipeline import run_analysis_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)

    def progress_cb(stage_id: str, message: str, progress: float):
        if args.verbose:
            pct = int(progress * 100)
            print(f"  [{stage_id}] {message} ({pct}%)")

    try:
        result = run_analysis_pipeline(
            input_files=pdfs,
            project_id=args.project_id,
            output_dir=output_dir,
            progress_callback=progress_cb,
        )
        if not result.success:
            print(f"ERROR: {result.error_message}")
            return 1
        if args.verbose:
            print(f"Done. Output: {output_dir / 'analysis.json'}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
