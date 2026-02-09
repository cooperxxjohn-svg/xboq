"""
BOQ Alignment Engine - Compare drawings BOQ with owner BOQ.

This module:
- Compares extracted BOQ from drawings with owner-provided BOQ
- Identifies mismatches in quantities, units, descriptions
- Flags missing items (in drawings but not in owner BOQ, and vice versa)
- Generates alignment report with discrepancies
- Outputs: alignment_report.md, discrepancies.json, unified_boq.csv

India-specific BOQ item matching and unit conversions.
"""

from .matcher import BOQMatcher, MatchResult
from .comparator import BOQComparator, Discrepancy
from .reconciler import BOQReconciler

__all__ = [
    "BOQMatcher",
    "MatchResult",
    "BOQComparator",
    "Discrepancy",
    "BOQReconciler",
    "run_alignment_engine",
]


def run_alignment_engine(
    project_id: str,
    drawings_boq: list,
    owner_boq: list,
    output_dir,
    tolerance_percent: float = 10.0,
) -> dict:
    """
    Run the BOQ alignment engine.

    Args:
        project_id: Project identifier
        drawings_boq: BOQ extracted from drawings
        owner_boq: BOQ provided by owner
        output_dir: Output directory
        tolerance_percent: Quantity tolerance for matching (default 10%)

    Returns:
        Dict with alignment results
    """
    from pathlib import Path
    import json
    import csv
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Match BOQ items
    logger.info("Matching BOQ items between drawings and owner BOQ...")
    matcher = BOQMatcher()
    matches = matcher.match(drawings_boq, owner_boq)

    matched_count = len([m for m in matches if m.match_type == "matched"])
    unmatched_drawings = len([m for m in matches if m.match_type == "drawings_only"])
    unmatched_owner = len([m for m in matches if m.match_type == "owner_only"])

    logger.info(f"Matched: {matched_count}, Drawings only: {unmatched_drawings}, Owner only: {unmatched_owner}")

    # 2. Compare quantities for matched items
    logger.info("Comparing quantities for matched items...")
    comparator = BOQComparator(tolerance_percent=tolerance_percent)
    discrepancies = comparator.compare(matches)

    logger.info(f"Found {len(discrepancies)} quantity discrepancies")

    # 3. Generate unified BOQ
    logger.info("Generating unified BOQ...")
    reconciler = BOQReconciler()
    unified_boq, reconciliation_notes = reconciler.reconcile(matches, discrepancies)

    logger.info(f"Unified BOQ has {len(unified_boq)} items")

    # 4. Export results

    # Discrepancies JSON
    with open(output_dir / "discrepancies.json", "w") as f:
        json.dump([d.to_dict() for d in discrepancies], f, indent=2)

    # Matches JSON
    with open(output_dir / "boq_matches.json", "w") as f:
        json.dump([m.to_dict() for m in matches], f, indent=2)

    # Unified BOQ CSV
    with open(output_dir / "unified_boq.csv", "w", newline="") as f:
        if unified_boq:
            writer = csv.DictWriter(f, fieldnames=unified_boq[0].keys())
            writer.writeheader()
            writer.writerows(unified_boq)

    # Alignment report markdown
    _export_alignment_report(
        output_dir / "alignment_report.md",
        project_id,
        matches,
        discrepancies,
        reconciliation_notes,
    )

    # Calculate alignment score
    total_items = matched_count + unmatched_drawings + unmatched_owner
    alignment_score = (matched_count / total_items * 100) if total_items > 0 else 100

    # Calculate quantity accuracy for matched items
    matched_items = [m for m in matches if m.match_type == "matched"]
    within_tolerance = len([m for m in matched_items if not any(d.item_id == m.drawings_item.get("item_id") for d in discrepancies)])
    quantity_accuracy = (within_tolerance / len(matched_items) * 100) if matched_items else 100

    return {
        "total_drawings_items": len(drawings_boq),
        "total_owner_items": len(owner_boq),
        "matched_items": matched_count,
        "drawings_only": unmatched_drawings,
        "owner_only": unmatched_owner,
        "discrepancies": len(discrepancies),
        "alignment_score": alignment_score,
        "quantity_accuracy": quantity_accuracy,
        "unified_boq_items": len(unified_boq),
    }


def _export_alignment_report(
    output_path,
    project_id: str,
    matches: list,
    discrepancies: list,
    reconciliation_notes: list,
) -> None:
    """Export alignment report as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# BOQ Alignment Report: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        matched = [m for m in matches if m.match_type == "matched"]
        drawings_only = [m for m in matches if m.match_type == "drawings_only"]
        owner_only = [m for m in matches if m.match_type == "owner_only"]

        f.write("## Summary\n\n")
        f.write(f"- **Matched Items**: {len(matched)}\n")
        f.write(f"- **In Drawings Only**: {len(drawings_only)} (missing from owner BOQ)\n")
        f.write(f"- **In Owner BOQ Only**: {len(owner_only)} (missing from drawings)\n")
        f.write(f"- **Quantity Discrepancies**: {len(discrepancies)}\n\n")

        # Alignment score
        total = len(matched) + len(drawings_only) + len(owner_only)
        score = (len(matched) / total * 100) if total > 0 else 100
        f.write(f"**Alignment Score**: {score:.1f}%\n\n")

        # Quantity discrepancies
        if discrepancies:
            f.write("## Quantity Discrepancies\n\n")
            f.write("Items where drawing quantities differ from owner BOQ:\n\n")
            f.write("| Item | Description | Drawing Qty | Owner Qty | Difference | % Diff |\n")
            f.write("|------|-------------|-------------|-----------|------------|--------|\n")

            for disc in discrepancies:
                diff = disc.drawings_qty - disc.owner_qty
                diff_pct = (diff / disc.owner_qty * 100) if disc.owner_qty != 0 else 0
                sign = "+" if diff > 0 else ""
                f.write(f"| {disc.item_id} | {disc.description[:40]}... | ")
                f.write(f"{disc.drawings_qty:.2f} {disc.unit} | ")
                f.write(f"{disc.owner_qty:.2f} {disc.unit} | ")
                f.write(f"{sign}{diff:.2f} | {sign}{diff_pct:.1f}% |\n")

            f.write("\n")

        # Items in drawings only
        if drawings_only:
            f.write("## Items in Drawings Only (Missing from Owner BOQ)\n\n")
            f.write("These items were extracted from drawings but not found in owner BOQ:\n\n")
            f.write("| Item | Description | Quantity | Unit | Package |\n")
            f.write("|------|-------------|----------|------|----------|\n")

            for match in drawings_only:
                item = match.drawings_item
                f.write(f"| {item.get('item_id', 'N/A')} | ")
                f.write(f"{item.get('description', 'N/A')[:50]} | ")
                f.write(f"{item.get('quantity', 0):.2f} | ")
                f.write(f"{item.get('unit', 'N/A')} | ")
                f.write(f"{item.get('package', 'N/A')} |\n")

            f.write("\n**Action Required**: Confirm if these items should be added to scope or excluded.\n\n")

        # Items in owner BOQ only
        if owner_only:
            f.write("## Items in Owner BOQ Only (Not Found in Drawings)\n\n")
            f.write("These items are in owner BOQ but could not be extracted from drawings:\n\n")
            f.write("| Item | Description | Quantity | Unit |\n")
            f.write("|------|-------------|----------|------|\n")

            for match in owner_only:
                item = match.owner_item
                f.write(f"| {item.get('item_no', 'N/A')} | ")
                f.write(f"{item.get('description', 'N/A')[:50]} | ")
                f.write(f"{item.get('quantity', 0):.2f} | ")
                f.write(f"{item.get('unit', 'N/A')} |\n")

            f.write("\n**Action Required**: Verify these items in drawings or confirm as provisional.\n\n")

        # Reconciliation notes
        if reconciliation_notes:
            f.write("## Reconciliation Notes\n\n")
            for note in reconciliation_notes:
                f.write(f"- {note}\n")
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if len(drawings_only) > 0:
            f.write(f"1. **{len(drawings_only)} items** found in drawings but missing from owner BOQ. ")
            f.write("Raise RFI to confirm scope inclusion.\n")

        if len(owner_only) > 0:
            f.write(f"2. **{len(owner_only)} items** in owner BOQ but not found in drawings. ")
            f.write("Request additional drawings or treat as provisional.\n")

        if len(discrepancies) > 0:
            high_disc = [d for d in discrepancies if abs(d.difference_percent) > 20]
            if high_disc:
                f.write(f"3. **{len(high_disc)} items** have quantity differences > 20%. ")
                f.write("Review measurement methodology and re-verify.\n")

        if score >= 90:
            f.write("\n✅ **Good alignment** between drawings and owner BOQ.\n")
        elif score >= 70:
            f.write("\n🟠 **Moderate alignment** - several items need reconciliation.\n")
        else:
            f.write("\n🔴 **Poor alignment** - significant discrepancies require attention before bid submission.\n")
