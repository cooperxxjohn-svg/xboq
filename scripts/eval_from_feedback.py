#!/usr/bin/env python3
"""
Eval Harness — compute metrics from feedback.jsonl.

Metrics:
    - RFI acceptance rate: correct / (correct + wrong + edited) for type=rfi
    - Quantity correction rate: (wrong + edited) / total for type=quantity
    - Common error categories: top-N patterns in edited/wrong entries

Usage:
    python3 scripts/eval_from_feedback.py <project_id>
    python3 scripts/eval_from_feedback.py --all
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_feedback(project_dir: Path) -> list:
    """Load feedback.jsonl from a project output dir."""
    filepath = project_dir / "feedback.jsonl"
    if not filepath.exists():
        return []
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def compute_metrics(entries: list) -> dict:
    """
    Compute eval metrics from feedback entries.

    Returns:
        Dict with:
            total_entries: int
            rfi_acceptance_rate: float | None (correct / total_rfi)
            rfi_counts: {correct, wrong, edited}
            quantity_correction_rate: float | None ((wrong+edited) / total_qty)
            quantity_counts: {correct, wrong, edited}
            common_error_categories: [{category, count}]
    """
    rfi_counts = Counter()
    qty_counts = Counter()
    error_categories = Counter()

    for entry in entries:
        ftype = entry.get("feedback_type", "")
        verdict = entry.get("verdict", "")

        if ftype == "rfi":
            rfi_counts[verdict] += 1
        elif ftype == "quantity":
            qty_counts[verdict] += 1

        # Error categorization for wrong/edited entries
        if verdict in ("wrong", "edited"):
            original = entry.get("original_value", "") or ""
            corrected = entry.get("corrected_value", "") or ""

            if verdict == "edited" and original and corrected:
                # Try to detect category of correction
                try:
                    orig_num = float(original)
                    corr_num = float(corrected)
                    if orig_num != corr_num:
                        error_categories["qty_value_change"] += 1
                    else:
                        error_categories["other_edit"] += 1
                except (ValueError, TypeError):
                    if original.lower() != corrected.lower():
                        error_categories["text_correction"] += 1
                    else:
                        error_categories["other_edit"] += 1
            elif verdict == "wrong":
                error_categories[f"wrong_{ftype}"] += 1

    # Compute rates
    total_rfi = sum(rfi_counts.values())
    total_qty = sum(qty_counts.values())

    rfi_acceptance_rate = (
        rfi_counts["correct"] / total_rfi
        if total_rfi > 0 else None
    )

    qty_correction_rate = (
        (qty_counts["wrong"] + qty_counts["edited"]) / total_qty
        if total_qty > 0 else None
    )

    common_errors = [
        {"category": cat, "count": cnt}
        for cat, cnt in error_categories.most_common(10)
    ]

    return {
        "total_entries": len(entries),
        "rfi_acceptance_rate": rfi_acceptance_rate,
        "rfi_counts": dict(rfi_counts),
        "quantity_correction_rate": qty_correction_rate,
        "quantity_counts": dict(qty_counts),
        "common_error_categories": common_errors,
    }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/eval_from_feedback.py <project_id>")
        print("       python3 scripts/eval_from_feedback.py --all")
        sys.exit(1)

    out_base = PROJECT_ROOT / "out"
    if not out_base.exists():
        print(f"No output directory found at {out_base}")
        sys.exit(1)

    if sys.argv[1] == "--all":
        # Process all projects
        projects = sorted(d for d in out_base.iterdir() if d.is_dir())
    else:
        project_id = sys.argv[1]
        project_dir = out_base / project_id
        if not project_dir.exists():
            print(f"Project directory not found: {project_dir}")
            sys.exit(1)
        projects = [project_dir]

    all_entries = []
    for project_dir in projects:
        entries = load_feedback(project_dir)
        if entries:
            print(f"\n📂 {project_dir.name}: {len(entries)} feedback entries")
            metrics = compute_metrics(entries)
            _print_metrics(metrics)
            all_entries.extend(entries)

    if not all_entries:
        print("No feedback entries found.")
        sys.exit(0)

    if len(projects) > 1:
        print(f"\n{'='*50}")
        print(f"AGGREGATE ({len(all_entries)} total entries across {len(projects)} projects)")
        print(f"{'='*50}")
        metrics = compute_metrics(all_entries)
        _print_metrics(metrics)


def _print_metrics(metrics: dict):
    """Pretty-print metrics."""
    print(f"  Total entries: {metrics['total_entries']}")

    rfi_rate = metrics.get("rfi_acceptance_rate")
    rfi_c = metrics.get("rfi_counts", {})
    if rfi_rate is not None:
        print(f"  RFI acceptance rate: {rfi_rate:.1%}")
        print(f"    correct={rfi_c.get('correct', 0)}, wrong={rfi_c.get('wrong', 0)}, edited={rfi_c.get('edited', 0)}")

    qty_rate = metrics.get("quantity_correction_rate")
    qty_c = metrics.get("quantity_counts", {})
    if qty_rate is not None:
        print(f"  Quantity correction rate: {qty_rate:.1%}")
        print(f"    correct={qty_c.get('correct', 0)}, wrong={qty_c.get('wrong', 0)}, edited={qty_c.get('edited', 0)}")

    errors = metrics.get("common_error_categories", [])
    if errors:
        print(f"  Common error categories:")
        for ec in errors[:5]:
            print(f"    - {ec['category']}: {ec['count']}")


if __name__ == "__main__":
    main()
