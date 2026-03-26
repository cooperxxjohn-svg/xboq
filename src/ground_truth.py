"""
Ground Truth Import/Diff — Measure pipeline accuracy against human-labeled expectations.

Sprint 24 Phase 3: Enables systematic improvement by comparing actual pipeline output
against a "ground truth" JSON file for each tender.

Usage:
    # Create ground truth template from existing analysis
    python -m src.ground_truth template --analysis out/sonipat/analysis.json --output gt/sonipat.json

    # Compare pipeline output against ground truth
    python -m src.ground_truth diff --analysis out/sonipat/analysis.json --truth gt/sonipat.json

    # Batch diff across all tenders
    python -m src.ground_truth batch --truth-dir gt/ --output-dir out/
"""

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# GROUND TRUTH SCHEMA
# =============================================================================

@dataclass
class GroundTruthEntry:
    """Expected output for a single tender."""

    # Tender identity
    tender_name: str
    tender_pages: int = 0
    notes: str = ""  # human reviewer notes

    # Page classification expectations
    expected_page_types: Dict[str, int] = field(default_factory=dict)
    # e.g., {"spec": 140, "conditions": 130, "plan": 75, "boq": 2, "unknown_max": 25}

    # Extraction expectations (counts)
    expected_boq_items_min: int = 0
    expected_schedule_rows_min: int = 0
    expected_requirements_min: int = 0
    expected_commercial_terms_min: int = 0
    expected_rfis_min: int = 0
    expected_blockers_min: int = 0

    # Key items that MUST be found (spot checks)
    expected_door_tags: List[str] = field(default_factory=list)
    expected_window_tags: List[str] = field(default_factory=list)
    expected_room_names: List[str] = field(default_factory=list)
    expected_boq_trades: List[str] = field(default_factory=list)  # e.g., ["civil", "electrical"]
    expected_commercial_term_types: List[str] = field(default_factory=list)  # e.g., ["ld_percent", "retention"]

    # Decision expectations
    expected_decision: str = ""  # "GO", "CONDITIONAL", "NO_GO"
    expected_readiness_min: int = 0
    expected_readiness_max: int = 100

    # BOQ source
    expected_boq_source: str = ""  # "excel", "pdf", "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruthEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# DIFF ENGINE
# =============================================================================

@dataclass
class DiffItem:
    """A single comparison result."""
    field: str
    expected: Any
    actual: Any
    status: str  # "pass", "fail", "warn"
    message: str = ""


@dataclass
class DiffReport:
    """Full diff report between ground truth and actual output."""
    tender_name: str
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    items: List[DiffItem] = field(default_factory=list)
    accuracy_score: float = 0.0  # 0.0 to 1.0

    def add(self, field: str, expected: Any, actual: Any,
            status: str, message: str = ""):
        self.items.append(DiffItem(
            field=field, expected=expected, actual=actual,
            status=status, message=message,
        ))
        self.total_checks += 1
        if status == "pass":
            self.passed += 1
        elif status == "fail":
            self.failed += 1
        elif status == "warn":
            self.warnings += 1

    def compute_score(self):
        if self.total_checks > 0:
            self.accuracy_score = round(self.passed / self.total_checks, 3)

    def to_dict(self) -> dict:
        return {
            "tender_name": self.tender_name,
            "total_checks": self.total_checks,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "accuracy_score": self.accuracy_score,
            "items": [asdict(it) for it in self.items],
        }

    def summary_line(self) -> str:
        return (
            f"{self.tender_name}: "
            f"{self.passed}/{self.total_checks} passed "
            f"({self.accuracy_score:.0%}), "
            f"{self.failed} failed, {self.warnings} warnings"
        )

    def print_report(self):
        print(f"\n{'='*70}")
        print(f"GROUND TRUTH DIFF: {self.tender_name}")
        print(f"{'='*70}")
        print(f"Score: {self.accuracy_score:.0%} ({self.passed}/{self.total_checks} passed)\n")

        # Group by status
        for status_label, status_key in [("FAILURES", "fail"), ("WARNINGS", "warn"), ("PASSES", "pass")]:
            group = [it for it in self.items if it.status == status_key]
            if not group:
                continue
            icon = {"fail": "✗", "warn": "⚠", "pass": "✓"}[status_key]
            print(f"--- {status_label} ({len(group)}) ---")
            for it in group:
                exp = str(it.expected)[:30]
                act = str(it.actual)[:30]
                msg = f" ({it.message})" if it.message else ""
                print(f"  {icon} {it.field:<35s} expected={exp:<15s} actual={act:<15s}{msg}")
            print()


def diff_analysis(
    truth: GroundTruthEntry,
    analysis: dict,
    analysis_dir: Optional[Path] = None,
) -> DiffReport:
    """
    Compare a ground truth entry against an actual analysis payload.

    Args:
        truth: Human-labeled expected output.
        analysis: Pipeline analysis.json payload.
        analysis_dir: Directory containing analysis.json (for loading plan_graph.json).

    Returns:
        DiffReport with pass/fail/warn for each check.
    """
    report = DiffReport(tender_name=truth.tender_name)

    # Extract data from analysis payload
    pi = analysis.get("diagnostics", {}).get("page_index", {})
    es = analysis.get("extraction_summary", {})
    counts = es.get("counts", {})
    page_types = pi.get("counts_by_type", {})
    diag = es.get("extraction_diagnostics", {})

    # ── Page classification checks ──────────────────────────────────────
    for dtype, expected_count in truth.expected_page_types.items():
        if dtype == "unknown_max":
            actual = page_types.get("unknown", 0)
            if actual <= expected_count:
                report.add("pages.unknown", f"<={expected_count}", actual, "pass")
            else:
                report.add("pages.unknown", f"<={expected_count}", actual, "fail",
                           f"{actual - expected_count} over threshold")
        else:
            actual = page_types.get(dtype, 0)
            # Allow ±20% tolerance
            low = int(expected_count * 0.8)
            high = int(expected_count * 1.2) + 1
            if low <= actual <= high:
                report.add(f"pages.{dtype}", expected_count, actual, "pass")
            elif actual > 0:
                report.add(f"pages.{dtype}", expected_count, actual, "warn",
                           f"outside ±20% tolerance")
            else:
                report.add(f"pages.{dtype}", expected_count, actual, "fail",
                           f"expected ~{expected_count}, got 0")

    # ── Extraction count checks (minimums) ──────────────────────────────
    _check_min(report, "boq_items", truth.expected_boq_items_min, counts.get("boq_items", 0))
    _check_min(report, "schedules", truth.expected_schedule_rows_min, counts.get("schedules", 0))
    _check_min(report, "requirements", truth.expected_requirements_min, counts.get("requirements", 0))
    _check_min(report, "commercial_terms", truth.expected_commercial_terms_min, counts.get("commercial_terms", 0))

    rfis = analysis.get("rfis", [])
    _check_min(report, "rfis", truth.expected_rfis_min, len(rfis))

    blockers = analysis.get("blockers", [])
    _check_min(report, "blockers", truth.expected_blockers_min, len(blockers))

    # ── Key item spot checks ────────────────────────────────────────────
    # Door tags
    pg = _load_plan_graph(analysis, analysis_dir)
    actual_door_tags = set(pg.get("all_door_tags", []))
    for tag in truth.expected_door_tags:
        found = tag in actual_door_tags
        report.add(f"door_tag.{tag}", "present", "found" if found else "missing",
                   "pass" if found else "fail")

    # Window tags
    actual_window_tags = set(pg.get("all_window_tags", []))
    for tag in truth.expected_window_tags:
        found = tag in actual_window_tags
        report.add(f"window_tag.{tag}", "present", "found" if found else "missing",
                   "pass" if found else "fail")

    # Room names
    actual_rooms = set(r.upper() for r in pg.get("all_room_names", []))
    for name in truth.expected_room_names:
        found = name.upper() in actual_rooms
        report.add(f"room.{name}", "present", "found" if found else "missing",
                   "pass" if found else "warn")

    # Commercial term types
    actual_terms = es.get("commercial_terms", [])
    actual_term_types = set()
    for t in actual_terms:
        if isinstance(t, dict):
            actual_term_types.add(t.get("term_type", ""))
    for term_type in truth.expected_commercial_term_types:
        found = term_type in actual_term_types
        report.add(f"commercial.{term_type}", "present", "found" if found else "missing",
                   "pass" if found else "warn")

    # ── Decision checks ─────────────────────────────────────────────────
    if truth.expected_decision:
        actual_decision = analysis.get("decision", "")
        match = actual_decision.upper() == truth.expected_decision.upper()
        report.add("decision", truth.expected_decision, actual_decision,
                   "pass" if match else "warn")

    readiness = analysis.get("readiness_score", 0)
    if truth.expected_readiness_min > 0 or truth.expected_readiness_max < 100:
        in_range = truth.expected_readiness_min <= readiness <= truth.expected_readiness_max
        report.add("readiness", f"{truth.expected_readiness_min}-{truth.expected_readiness_max}",
                   readiness, "pass" if in_range else "warn")

    report.compute_score()
    return report


def _check_min(report: DiffReport, field: str, expected_min: int, actual: int):
    """Check that actual meets or exceeds the minimum."""
    if expected_min <= 0:
        return  # No expectation set
    if actual >= expected_min:
        report.add(f"count.{field}", f">={expected_min}", actual, "pass")
    elif actual > 0:
        report.add(f"count.{field}", f">={expected_min}", actual, "warn",
                   f"{expected_min - actual} below minimum")
    else:
        report.add(f"count.{field}", f">={expected_min}", actual, "fail",
                   f"expected >={expected_min}, got 0")


def _load_plan_graph(analysis: dict, analysis_dir: Optional[Path] = None) -> dict:
    """Extract plan_graph data from analysis payload, deep_analysis, or companion file."""
    # Try companion plan_graph.json file first (most reliable)
    if analysis_dir:
        pg_path = Path(analysis_dir) / "plan_graph.json"
        if pg_path.exists():
            try:
                with open(pg_path) as f:
                    return json.load(f)
            except Exception:
                pass

    # Try diagnostics (newer payloads)
    pg = analysis.get("diagnostics", {}).get("plan_graph")
    if pg:
        return pg
    # Try deep_analysis keys
    da = analysis.get("deep_analysis", {})
    return {
        "all_door_tags": da.get("all_door_tags", analysis.get("door_tags_found", [])),
        "all_window_tags": da.get("all_window_tags", analysis.get("window_tags_found", [])),
        "all_room_names": da.get("all_room_names", analysis.get("room_names_found", [])),
    }


# =============================================================================
# TEMPLATE GENERATION
# =============================================================================

def generate_template(analysis_path: Path) -> GroundTruthEntry:
    """
    Generate a ground truth template from an existing analysis.json.

    Pre-fills values from the actual output so the human reviewer only needs
    to verify/correct them rather than typing everything from scratch.
    """
    with open(analysis_path) as f:
        data = json.load(f)

    pi = data.get("diagnostics", {}).get("page_index", {})
    es = data.get("extraction_summary", {})
    counts = es.get("counts", {})
    page_types = pi.get("counts_by_type", {})

    # Try to get plan_graph data
    pg_path = analysis_path.parent / "plan_graph.json"
    pg = {}
    if pg_path.exists():
        with open(pg_path) as f:
            pg = json.load(f)

    tender_name = data.get("project_id", analysis_path.parent.name)
    total_pages = pi.get("total_pages", 0)

    return GroundTruthEntry(
        tender_name=tender_name,
        tender_pages=total_pages,
        notes="AUTO-GENERATED from pipeline output. Review and correct values.",
        expected_page_types={
            "spec": page_types.get("spec", 0),
            "conditions": page_types.get("conditions", 0),
            "plan": page_types.get("plan", 0),
            "boq": page_types.get("boq", 0),
            "schedule": page_types.get("schedule", 0),
            "unknown_max": max(page_types.get("unknown", 0), 25),
        },
        expected_boq_items_min=max(counts.get("boq_items", 0), 0),
        expected_schedule_rows_min=max(counts.get("schedules", 0), 0),
        expected_requirements_min=max(int(counts.get("requirements", 0) * 0.8), 0),
        expected_commercial_terms_min=max(counts.get("commercial_terms", 0), 0),
        expected_rfis_min=max(len(data.get("rfis", [])), 0),
        expected_blockers_min=max(len(data.get("blockers", [])), 0),
        expected_door_tags=pg.get("all_door_tags", []),
        expected_window_tags=pg.get("all_window_tags", []),
        expected_room_names=pg.get("all_room_names", [])[:10],  # top 10
        expected_commercial_term_types=[
            t.get("term_type", "") for t in es.get("commercial_terms", [])
            if isinstance(t, dict) and t.get("term_type")
        ],
        expected_decision=data.get("decision", ""),
        expected_readiness_min=max(data.get("readiness_score", 0) - 15, 0),
        expected_readiness_max=min(data.get("readiness_score", 100) + 15, 100),
        expected_boq_source="pdf",  # default; reviewer should correct
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ground Truth Import/Diff — measure pipeline accuracy",
    )
    sub = parser.add_subparsers(dest="command")

    # template command
    tpl = sub.add_parser("template", help="Generate ground truth template from analysis")
    tpl.add_argument("--analysis", "-a", type=Path, required=True)
    tpl.add_argument("--output", "-o", type=Path, default=None)

    # diff command
    df = sub.add_parser("diff", help="Compare analysis against ground truth")
    df.add_argument("--analysis", "-a", type=Path, required=True)
    df.add_argument("--truth", "-t", type=Path, required=True)
    df.add_argument("--json", action="store_true", help="Output as JSON")

    # batch command
    bt = sub.add_parser("batch", help="Batch diff across all tenders")
    bt.add_argument("--truth-dir", type=Path, required=True)
    bt.add_argument("--output-dir", type=Path, required=True)
    bt.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "template":
        entry = generate_template(args.analysis)
        out_path = args.output or Path(f"gt_{entry.tender_name}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        print(f"Template saved to {out_path}")
        print("Review and correct values, then use 'diff' to compare.")

    elif args.command == "diff":
        with open(args.truth) as f:
            truth = GroundTruthEntry.from_dict(json.load(f))
        with open(args.analysis) as f:
            analysis = json.load(f)
        report = diff_analysis(truth, analysis, analysis_dir=args.analysis.parent)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            report.print_report()

    elif args.command == "batch":
        truth_files = list(args.truth_dir.glob("*.json"))
        if not truth_files:
            print(f"No ground truth files found in {args.truth_dir}")
            return 1

        reports = []
        for tf in sorted(truth_files):
            with open(tf) as f:
                truth = GroundTruthEntry.from_dict(json.load(f))
            # Find matching analysis
            analysis_path = _find_analysis(args.output_dir, truth.tender_name)
            if not analysis_path:
                print(f"  ✗ No analysis found for {truth.tender_name}")
                continue
            with open(analysis_path) as f:
                analysis = json.load(f)
            report = diff_analysis(truth, analysis, analysis_dir=analysis_path.parent)
            reports.append(report)
            print(f"  {report.summary_line()}")

        if reports:
            avg_score = sum(r.accuracy_score for r in reports) / len(reports)
            print(f"\nOverall: {avg_score:.0%} accuracy across {len(reports)} tenders")

    else:
        parser.print_help()

    return 0


def _find_analysis(output_dir: Path, tender_name: str) -> Optional[Path]:
    """Find the latest analysis.json for a tender name."""
    # Try direct match
    direct = output_dir / tender_name / "analysis.json"
    if direct.exists():
        return direct
    # Try glob pattern
    candidates = sorted(output_dir.glob(f"*{tender_name}*/analysis.json"), reverse=True)
    return candidates[0] if candidates else None


if __name__ == "__main__":
    sys.exit(main())
