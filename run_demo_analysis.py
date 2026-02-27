#!/usr/bin/env python3
"""
XBOQ Demo Analysis CLI - Canonical Single Command for YC Demo

Runs the pre-bid scope & risk analysis pipeline and outputs a stable JSON schema
that the demo frontend can reliably consume.

Usage:
    python run_demo_analysis.py --project_id pwd_garage --input ./demo_inputs/pwd_garage.pdf
    python run_demo_analysis.py --project_id pwd_garage --input ./demo_inputs/pwd_garage.pdf --output ./out/pwd_garage
    python run_demo_analysis.py --project_id pwd_garage --cached  # Use cached result (for demos)

Output Schema (analysis.json):
    {
        "project_id": str,
        "timestamp": str (ISO format),
        "drawing_overview": {
            "files": List[str],
            "pages_total": int,
            "disciplines_detected": List[str],
            "door_tags_found": int,
            "window_tags_found": int,
            "room_names_found": int,
            "scale_found_pages": int,
            "ocr_used": bool,
            "ocr_pages_count": int
        },
        "readiness_score": int (0-100),
        "decision": str ("PASS" | "CONDITIONAL" | "NO-GO"),
        "sub_scores": {
            "completeness": int,
            "coverage": int,
            "measurement": int,
            "blocker": int
        },
        "blockers": List[Blocker],
        "rfis": List[RFI],
        "trade_coverage": List[TradeCoverage],
        "timings": {
            "load_s": float,
            "extract_s": float,
            "graph_s": float,
            "reason_s": float,
            "rfi_s": float,
            "export_s": float,
            "total_s": float
        }
    }
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# STABLE JSON SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION = "1.0.0"

# Fields required in analysis.json for demo frontend
REQUIRED_FIELDS = [
    "project_id",
    "timestamp",
    "drawing_overview",
    "readiness_score",
    "decision",
    "sub_scores",
    "blockers",
    "rfis",
    "trade_coverage",
    "timings",
]


def validate_analysis_json(data: dict) -> tuple[bool, list[str]]:
    """Validate that analysis.json conforms to expected schema."""
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            missing.append(field)

    # Check nested required fields
    overview = data.get("drawing_overview", {})
    overview_required = ["files", "pages_total", "disciplines_detected"]
    for field in overview_required:
        if field not in overview:
            missing.append(f"drawing_overview.{field}")

    timings = data.get("timings", {})
    timing_required = ["load_s", "extract_s", "total_s"]
    for field in timing_required:
        if field not in timings:
            missing.append(f"timings.{field}")

    return len(missing) == 0, missing


def run_analysis(
    project_id: str,
    input_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> dict:
    """
    Run the analysis pipeline and return the result payload.

    Returns:
        dict: The analysis result conforming to stable schema
    """
    from src.analysis.pipeline import (
        run_analysis_pipeline,
        save_uploaded_files,
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    input_files = []
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = list(input_path.glob("*.pdf"))

    if not input_files:
        raise ValueError(f"No PDF files found at: {input_path}")

    if verbose:
        print(f"Found {len(input_files)} PDF file(s)")
        for f in input_files:
            print(f"  - {f.name}")

    # Progress callback for CLI output
    def progress_callback(stage_id: str, message: str, progress: float):
        if verbose:
            pct = int(progress * 100)
            print(f"  [{stage_id}] {message} ({pct}%)")

    # Run the pipeline
    result = run_analysis_pipeline(
        input_files=input_files,
        project_id=project_id,
        output_dir=output_dir,
        progress_callback=progress_callback if verbose else None,
    )

    if not result.success:
        raise RuntimeError(f"Analysis failed: {result.error_message}\n{result.stack_trace}")

    # The payload is the stable JSON output
    payload = result.payload

    # Add schema version
    payload["schema_version"] = SCHEMA_VERSION

    # Validate the output
    is_valid, missing = validate_analysis_json(payload)
    if not is_valid:
        print(f"WARNING: Analysis output missing required fields: {missing}")

    return payload


def load_cached_analysis(project_id: str, cache_dir: Path) -> Optional[dict]:
    """Load cached analysis result if it exists."""
    cache_path = cache_dir / project_id / "analysis.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_analysis(payload: dict, output_dir: Path) -> Path:
    """Save analysis result to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "analysis.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="XBOQ Demo Analysis - Pre-bid scope & risk check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run analysis on a PDF
    python run_demo_analysis.py --project_id garage --input ./drawings/garage.pdf

    # Run analysis with custom output directory
    python run_demo_analysis.py --project_id garage --input ./drawings/garage.pdf --output ./results/garage

    # Use cached result (for reliable demos)
    python run_demo_analysis.py --project_id pwd_garage --cached

    # Cache a new result for future demos
    python run_demo_analysis.py --project_id pwd_garage --input ./drawings/garage.pdf --save-cache
        """
    )

    parser.add_argument(
        "--project_id", "-p",
        required=True,
        help="Project identifier (e.g., pwd_garage, villa_tender)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (default: ./out/<project_id>)"
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use cached analysis result instead of re-running"
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Save result to demo cache for future use"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to stdout (for piping)"
    )

    args = parser.parse_args()

    # Paths
    cache_dir = PROJECT_ROOT / "demo_cache"
    output_dir = args.output or (PROJECT_ROOT / "out" / args.project_id)

    try:
        # Check for cached result
        if args.cached:
            payload = load_cached_analysis(args.project_id, cache_dir)
            if payload:
                if args.verbose:
                    print(f"Loaded cached analysis for {args.project_id}")
            else:
                print(f"ERROR: No cached analysis found for {args.project_id}")
                print(f"       Run with --input to generate one, or --save-cache to cache it")
                return 1
        else:
            # Need input path for fresh analysis
            if not args.input:
                print("ERROR: --input is required unless using --cached")
                return 1

            if not args.input.exists():
                print(f"ERROR: Input not found: {args.input}")
                return 1

            if args.verbose:
                print("=" * 60)
                print("XBOQ Demo Analysis")
                print("=" * 60)
                print(f"Project ID: {args.project_id}")
                print(f"Input: {args.input}")
                print(f"Output: {output_dir}")
                print("=" * 60)
                print()

            # Run analysis
            payload = run_analysis(
                project_id=args.project_id,
                input_path=args.input,
                output_dir=output_dir,
                verbose=args.verbose,
            )

        # Save to output directory
        output_path = save_analysis(payload, output_dir)

        # Save to cache if requested
        if args.save_cache:
            cache_path = cache_dir / args.project_id
            cache_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(output_path, cache_path / "analysis.json")
            if args.verbose:
                print(f"Cached analysis to: {cache_path / 'analysis.json'}")

        # Output
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        else:
            # Print summary
            decision = payload.get("decision", "UNKNOWN")
            score = payload.get("readiness_score", 0)
            blockers = len(payload.get("blockers", []))
            rfis = len(payload.get("rfis", []))
            pages = payload.get("drawing_overview", {}).get("pages_total", 0)
            total_time = payload.get("timings", {}).get("total_s", 0)

            decision_emoji = {"PASS": "✅", "CONDITIONAL": "⚠️", "NO-GO": "❌"}.get(decision, "❓")

            print()
            print("=" * 60)
            print(f"{decision_emoji} Decision: {decision} (Score: {score}/100)")
            print("=" * 60)
            print(f"Pages analyzed: {pages}")
            print(f"Blockers found: {blockers}")
            print(f"RFIs generated: {rfis}")
            print(f"Analysis time:  {total_time:.2f}s")
            print()
            print(f"Output: {output_path}")
            print("=" * 60)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
