#!/usr/bin/env python3
"""
Regression Runner — run benchmark cases and compare KPIs.

Usage:
    python scripts/run_regression.py
    python scripts/run_regression.py --help

If benchmarks/manifest.json is missing, prints setup instructions and exits 0.
"""

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root (one level up from scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
MANIFEST_PATH = BENCHMARKS_DIR / "manifest.json"
EXPECTED_PATH = BENCHMARKS_DIR / "expected_metrics.yaml"
RUNS_DIR = BENCHMARKS_DIR / "_runs"


# ---------------------------------------------------------------------------
# KPI extraction from payload
# ---------------------------------------------------------------------------

def extract_kpis(payload: dict) -> dict:
    """Extract regression KPIs from a pipeline payload dict."""
    ps = payload.get("processing_stats", {})
    ed = payload.get("extraction_diagnostics", {})
    es = payload.get("extraction_summary", {})
    counts = es.get("counts", {})

    # table_methods_used may be a dict of method->count
    table_methods = ed.get("table_methods_used", {})
    if isinstance(table_methods, dict):
        table_methods_str = "; ".join(f"{k}={v}" for k, v in sorted(table_methods.items()))
    elif isinstance(table_methods, list):
        table_methods_str = "; ".join(str(m) for m in table_methods)
    else:
        table_methods_str = str(table_methods) if table_methods else ""

    return {
        "pages_total": ps.get("total_pages", 0),
        "deep_processed_pages": ps.get("deep_processed_pages", 0),
        "ocr_pages": ps.get("ocr_pages", 0),
        "boq_items_count": counts.get("boq_items", len(payload.get("extraction_summary", {}).get("boq_items", payload.get("boq_items", [])))),
        "commercial_terms_count": counts.get("commercial_terms", len(payload.get("commercial_terms", []))),
        "requirements_count": counts.get("requirements", len(payload.get("extraction_summary", {}).get("requirements", []))),
        "rfis_count": len(payload.get("rfis", [])),
        "blockers_count": len(payload.get("blockers", [])),
        "toxic_pages_count": ps.get("toxic_pages", 0),
        "qa_score": payload.get("qa_score", None),
        "boq_pages_attempted": ed.get("boq_pages_attempted", 0),
        "boq_pages_parsed": ed.get("boq_pages_parsed", 0),
        "schedule_pages_attempted": ed.get("schedule_pages_attempted", 0),
        "schedule_pages_parsed": ed.get("schedule_pages_parsed", 0),
        "table_methods_used": table_methods_str,
    }


# ---------------------------------------------------------------------------
# Threshold comparison
# ---------------------------------------------------------------------------

def load_expected_metrics() -> dict:
    """Load expected_metrics.yaml if it exists. Returns {} on any failure."""
    if not EXPECTED_PATH.exists():
        return {}
    try:
        import yaml  # PyYAML is in requirements.txt
        with open(EXPECTED_PATH) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # Fallback: try a simple YAML-like parser for key: value lines
        result = {}
        current_case = None
        try:
            for line in EXPECTED_PATH.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not line.startswith(" ") and stripped.endswith(":"):
                    current_case = stripped[:-1]
                    result[current_case] = {}
                elif current_case and ":" in stripped:
                    key, val = stripped.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        result[current_case][key] = float(val)
                    except ValueError:
                        result[current_case][key] = val
        except Exception:
            pass
        return result
    except Exception:
        return {}


def judge_kpi(kpi_name: str, kpi_value, thresholds: dict) -> str:
    """Return PASS/WARN/FAIL/INFO for a single KPI against thresholds."""
    if not thresholds:
        return "INFO"

    # Skip non-numeric KPIs
    if kpi_value is None or isinstance(kpi_value, str):
        return "INFO"

    kpi_value = float(kpi_value)

    # Check min_* thresholds
    min_key = f"min_{kpi_name}"
    if min_key in thresholds:
        threshold = float(thresholds[min_key])
        if kpi_value >= threshold:
            return "PASS"
        elif kpi_value >= threshold * 0.9:
            return "WARN"
        else:
            return "FAIL"

    # Check max_* thresholds
    max_key = f"max_{kpi_name}"
    if max_key in thresholds:
        threshold = float(thresholds[max_key])
        if kpi_value <= threshold:
            return "PASS"
        elif kpi_value <= threshold * 1.1:
            return "WARN"
        else:
            return "FAIL"

    return "INFO"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_case(case_name: str, case_config: dict, expected: dict) -> dict:
    """Run a single benchmark case. Returns a result dict."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = RUNS_DIR / case_name / timestamp
    case_dir.mkdir(parents=True, exist_ok=True)

    paths_raw = case_config.get("paths", [])
    mode = case_config.get("mode", "standard_review")

    row = {
        "case": case_name,
        "timestamp": timestamp,
        "mode": mode,
        "status": "ERROR",
        "error": "",
    }

    # Validate paths
    input_files = []
    for p in paths_raw:
        fp = Path(p).expanduser()
        if not fp.exists():
            row["error"] = f"File not found: {p}"
            print(f"  [SKIP] {case_name}: {row['error']}")
            return row
        input_files.append(fp)

    if not input_files:
        row["error"] = "No input files specified"
        print(f"  [SKIP] {case_name}: {row['error']}")
        return row

    try:
        # Import pipeline (late import to avoid startup cost if manifest missing)
        sys.path.insert(0, str(REPO_ROOT))
        from src.analysis.pipeline import run_analysis_pipeline

        project_id = f"bench_{case_name}_{timestamp}"
        output_dir = case_dir / "output"

        print(f"  Running {case_name} ({mode}, {len(input_files)} file(s))...")
        result = run_analysis_pipeline(
            input_files=input_files,
            project_id=project_id,
            output_dir=output_dir,
            run_mode=mode,
        )

        payload = getattr(result, "payload", None) or {}

        # Save payload
        payload_path = case_dir / "payload.json"
        try:
            # Make payload JSON-serializable
            serializable = json.loads(json.dumps(payload, default=str))
            with open(payload_path, "w") as f:
                json.dump(serializable, f, indent=2, default=str)
        except Exception:
            # If serialization fails, save what we can
            with open(payload_path, "w") as f:
                f.write("{}")

        # Write run artifacts (non-critical)
        try:
            from src.ops.run_artifacts import write_all_artifacts
            write_all_artifacts(
                payload, case_dir / "run_artifacts",
                extra={"case": case_name, "mode": mode},
            )
        except Exception:
            pass

        # Extract KPIs
        kpis = extract_kpis(payload)
        row.update(kpis)
        row["status"] = "OK" if getattr(result, "success", False) else "PIPELINE_FAIL"

        # Judge KPIs against thresholds
        case_thresholds = expected.get(case_name, {})
        judgements = {}
        for kpi_name, kpi_value in kpis.items():
            judgements[kpi_name] = judge_kpi(kpi_name, kpi_value, case_thresholds)
        row["judgements"] = judgements

        # Print summary
        status_label = "OK" if row["status"] == "OK" else "PIPELINE_FAIL"
        print(f"  [{status_label}] {case_name}: {kpis['boq_items_count']} BOQ items, "
              f"{kpis['rfis_count']} RFIs, {kpis['pages_total']} pages")

        # Print any FAIL/WARN judgements
        for kpi_name, judgement in judgements.items():
            if judgement in ("FAIL", "WARN"):
                print(f"    {judgement}: {kpi_name} = {kpis.get(kpi_name)}")

    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["status"] = "ERROR"
        print(f"  [ERROR] {case_name}: {row['error']}")
        traceback.print_exc()

    return row


def write_report(rows: list, report_path: Path):
    """Write regression report CSV."""
    if not rows:
        return

    # Determine all columns
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys and k != "judgements":
                all_keys.append(k)

    # Add judgement columns
    judgement_keys = set()
    for r in rows:
        for k in r.get("judgements", {}):
            judgement_keys.add(f"judge_{k}")
    all_keys.extend(sorted(judgement_keys))

    # Flatten judgements into row
    flat_rows = []
    for r in rows:
        flat = {k: r.get(k, "") for k in all_keys if k != "judgements"}
        for k, v in r.get("judgements", {}).items():
            flat[f"judge_{k}"] = v
        flat_rows.append(flat)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="xBOQ Regression Runner — benchmark pipeline against known cases."
    )
    parser.add_argument(
        "--manifest", type=str, default=str(MANIFEST_PATH),
        help=f"Path to manifest.json (default: {MANIFEST_PATH})",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    # ------------------------------------------------------------------
    # If manifest is missing, print instructions and exit 0
    # ------------------------------------------------------------------
    if not manifest_path.exists():
        print("=" * 60)
        print("xBOQ Regression Runner")
        print("=" * 60)
        print()
        print("No manifest found. To set up benchmarks:")
        print()
        print("  1. Copy the example manifest:")
        print(f"     cp {BENCHMARKS_DIR}/manifest.example.json {MANIFEST_PATH}")
        print()
        print("  2. Edit manifest.json to point to your local tender files:")
        print('     {"cases": {"my_case": {"paths": ["/path/to/tender.pdf"], "mode": "standard_review"}}}')
        print()
        print("  3. (Optional) Set thresholds in benchmarks/expected_metrics.yaml")
        print()
        print("  4. Run again:")
        print("     python scripts/run_regression.py")
        print()
        print("See benchmarks/README.md for full documentation.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error loading manifest: {e}")
        sys.exit(1)

    cases = manifest.get("cases", {})
    if not cases:
        print("Manifest loaded but contains no cases. Add cases to 'cases' key.")
        sys.exit(0)

    # Load expected metrics
    expected = load_expected_metrics()

    print("=" * 60)
    print(f"xBOQ Regression Runner — {len(cases)} case(s)")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Run each case (deterministic order by case name)
    # ------------------------------------------------------------------
    rows = []
    for case_name in sorted(cases.keys()):
        case_config = cases[case_name]
        row = run_case(case_name, case_config, expected)
        rows.append(row)
        print()

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RUNS_DIR / f"regression_report_{timestamp}.csv"
    write_report(rows, report_path)

    # Also write/overwrite a latest symlink-style copy
    latest_path = RUNS_DIR / "regression_report.csv"
    write_report(rows, latest_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(rows)
    ok_count = sum(1 for r in rows if r.get("status") == "OK")
    fail_count = sum(1 for r in rows if r.get("status") in ("ERROR", "PIPELINE_FAIL"))
    skip_count = total - ok_count - fail_count

    print(f"Summary: {ok_count}/{total} OK, {fail_count} failed, {skip_count} skipped")

    # Count judgement failures
    all_judgements = []
    for r in rows:
        all_judgements.extend(r.get("judgements", {}).values())
    fail_judgements = sum(1 for j in all_judgements if j == "FAIL")
    warn_judgements = sum(1 for j in all_judgements if j == "WARN")
    if fail_judgements:
        print(f"KPI failures: {fail_judgements} FAIL, {warn_judgements} WARN")
    elif warn_judgements:
        print(f"KPI warnings: {warn_judgements} WARN")

    # Exit 0 even on failures (graceful degradation)
    sys.exit(0)


if __name__ == "__main__":
    main()
