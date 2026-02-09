"""
BOQ Export Module
Export BOQ quantities to CSV, JSON, and report formats.

Output structure:
out/<project_id>/
  boq/
    boq_quantities.csv      - Complete BOQ
    wall_boq.csv            - Wall-specific BOQ
    finishes_boq.csv        - Finish-specific BOQ
    openings_schedule.csv   - Door/window schedule
  overlays/
    walls.png
    rooms.png
    openings.png
    confidence_heatmap.png
  assumptions.json
  report.md
"""

import csv
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BOQLineItem:
    """Standard BOQ line item."""
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    category: Optional[str] = None
    subcategory: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class BOQPackage:
    """Complete BOQ package for export."""
    project_id: str
    generated_at: str
    wall_boq: List[Any]
    finish_boq: List[Any]
    slab_boq: List[Any]
    steel_boq: List[Any]
    openings_boq: List[Any]
    totals: Dict[str, Any]
    assumptions: List[str]
    confidence_summary: Dict[str, float]


class BOQExporter:
    """
    Export BOQ data to various formats.

    Supports:
    - CSV (standard BOQ format)
    - JSON (complete data)
    - Markdown report
    """

    # CSV headers
    BOQ_HEADERS = [
        "Item Code",
        "Description",
        "Qty",
        "Unit",
        "Derived From",
        "Confidence",
        "Category",
        "Notes",
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.boq_dir = self.output_dir / "boq"
        self.boq_dir.mkdir(exist_ok=True)

        self.overlays_dir = self.output_dir / "overlays"
        self.overlays_dir.mkdir(exist_ok=True)

    def export_complete_boq(
        self,
        wall_items: List[Any],
        finish_items: List[Any],
        slab_items: List[Any],
        steel_items: List[Any],
        openings_items: List[Any],
        assumptions: List[str],
        project_id: str = "project",
    ) -> Dict[str, Path]:
        """
        Export complete BOQ package.

        Returns:
            Dict mapping output type to file path
        """
        outputs = {}

        # Combine all items for main BOQ
        all_items = []
        all_items.extend(self._standardize_items(wall_items, "Masonry"))
        all_items.extend(self._standardize_items(finish_items, "Finishes"))
        all_items.extend(self._standardize_items(slab_items, "Structural"))
        all_items.extend(self._standardize_items(steel_items, "Steel"))
        all_items.extend(self._standardize_items(openings_items, "Openings"))

        # Export main BOQ CSV
        main_csv = self.boq_dir / "boq_quantities.csv"
        self._export_csv(all_items, main_csv)
        outputs["main_boq"] = main_csv

        # Export category-specific CSVs
        if wall_items:
            wall_csv = self.boq_dir / "wall_boq.csv"
            self._export_csv(self._standardize_items(wall_items, "Masonry"), wall_csv)
            outputs["wall_boq"] = wall_csv

        if finish_items:
            finish_csv = self.boq_dir / "finishes_boq.csv"
            self._export_csv(self._standardize_items(finish_items, "Finishes"), finish_csv)
            outputs["finishes_boq"] = finish_csv

        if openings_items:
            openings_csv = self.boq_dir / "openings_schedule.csv"
            self._export_csv(self._standardize_items(openings_items, "Openings"), openings_csv)
            outputs["openings_schedule"] = openings_csv

        # Export JSON
        json_path = self.output_dir / "boq_complete.json"
        self._export_json(all_items, assumptions, project_id, json_path)
        outputs["json"] = json_path

        # Export assumptions
        assumptions_path = self.output_dir / "assumptions.json"
        self._export_assumptions(assumptions, assumptions_path)
        outputs["assumptions"] = assumptions_path

        # Export report
        report_path = self.output_dir / "report.md"
        self._export_report(all_items, assumptions, project_id, report_path)
        outputs["report"] = report_path

        logger.info(f"Exported {len(outputs)} files to {self.output_dir}")

        return outputs

    def _standardize_items(
        self,
        items: List[Any],
        category: str,
    ) -> List[BOQLineItem]:
        """Convert various item types to standard BOQ format."""
        standardized = []

        for item in items:
            if isinstance(item, dict):
                line = BOQLineItem(
                    item_code=item.get("item_code", ""),
                    description=item.get("description", ""),
                    qty=item.get("qty", 0),
                    unit=item.get("unit", ""),
                    derived_from=item.get("derived_from", ""),
                    confidence=item.get("confidence", 0.5),
                    category=category,
                    notes=item.get("notes"),
                )
            else:
                # Dataclass object
                line = BOQLineItem(
                    item_code=getattr(item, "item_code", ""),
                    description=getattr(item, "description", ""),
                    qty=getattr(item, "qty", 0),
                    unit=getattr(item, "unit", ""),
                    derived_from=getattr(item, "derived_from", ""),
                    confidence=getattr(item, "confidence", 0.5),
                    category=category,
                    notes=getattr(item, "notes", None),
                )

            standardized.append(line)

        return standardized

    def _export_csv(
        self,
        items: List[BOQLineItem],
        output_path: Path,
    ) -> None:
        """Export items to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.BOQ_HEADERS)

            for item in items:
                writer.writerow([
                    item.item_code,
                    item.description,
                    f"{item.qty:.2f}" if isinstance(item.qty, float) else item.qty,
                    item.unit,
                    item.derived_from,
                    f"{item.confidence:.2f}",
                    item.category or "",
                    item.notes or "",
                ])

    def _export_json(
        self,
        items: List[BOQLineItem],
        assumptions: List[str],
        project_id: str,
        output_path: Path,
    ) -> None:
        """Export complete data as JSON."""
        # Group by category
        by_category: Dict[str, List] = {}
        for item in items:
            cat = item.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "item_code": item.item_code,
                "description": item.description,
                "qty": item.qty,
                "unit": item.unit,
                "derived_from": item.derived_from,
                "confidence": item.confidence,
                "notes": item.notes,
            })

        # Calculate totals
        totals = {}
        for cat, cat_items in by_category.items():
            totals[cat] = {
                "item_count": len(cat_items),
                "avg_confidence": sum(i["confidence"] for i in cat_items) / len(cat_items) if cat_items else 0,
            }

        data = {
            "project_id": project_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_items": len(items),
                "categories": list(by_category.keys()),
                "avg_confidence": sum(i.confidence for i in items) / len(items) if items else 0,
            },
            "items_by_category": by_category,
            "category_totals": totals,
            "assumptions": assumptions,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_assumptions(
        self,
        assumptions: List[str],
        output_path: Path,
    ) -> None:
        """Export assumptions to JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "count": len(assumptions),
            "assumptions": assumptions,
            "note": "These assumptions were used where measured data was not available",
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_report(
        self,
        items: List[BOQLineItem],
        assumptions: List[str],
        project_id: str,
        output_path: Path,
    ) -> None:
        """Export markdown report."""
        # Group by category
        by_category: Dict[str, List] = {}
        for item in items:
            cat = item.category or "Other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        # Calculate confidence stats
        confidences = [i.confidence for i in items]
        high_conf = sum(1 for c in confidences if c >= 0.75)
        med_conf = sum(1 for c in confidences if 0.5 <= c < 0.75)
        low_conf = sum(1 for c in confidences if c < 0.5)

        with open(output_path, "w") as f:
            f.write(f"# BOQ Report - {project_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Items | {len(items)} |\n")
            f.write(f"| Categories | {len(by_category)} |\n")
            f.write(f"| High Confidence (≥75%) | {high_conf} ({high_conf/len(items)*100:.0f}%) |\n")
            f.write(f"| Medium Confidence (50-75%) | {med_conf} ({med_conf/len(items)*100:.0f}%) |\n")
            f.write(f"| Low Confidence (<50%) | {low_conf} ({low_conf/len(items)*100:.0f}%) |\n")
            f.write(f"| Assumptions Made | {len(assumptions)} |\n\n")

            # By category
            for cat, cat_items in by_category.items():
                f.write(f"## {cat}\n\n")
                f.write("| Item Code | Description | Qty | Unit | Confidence |\n")
                f.write("|-----------|-------------|-----|------|------------|\n")

                for item in cat_items:
                    conf_icon = "🟢" if item.confidence >= 0.75 else ("🟡" if item.confidence >= 0.5 else "🔴")
                    f.write(f"| {item.item_code} | {item.description[:40]} | {item.qty:.2f} | {item.unit} | {conf_icon} {item.confidence:.0%} |\n")

                f.write("\n")

            # Assumptions
            if assumptions:
                f.write("## Assumptions\n\n")
                f.write("The following assumptions were made during quantity calculation:\n\n")
                for i, assumption in enumerate(assumptions[:30], 1):
                    f.write(f"{i}. {assumption}\n")
                if len(assumptions) > 30:
                    f.write(f"\n*... and {len(assumptions) - 30} more assumptions*\n")

            # Legend
            f.write("\n---\n\n")
            f.write("**Confidence Legend:**\n")
            f.write("- 🟢 High (≥75%): Measured or detected from drawings\n")
            f.write("- 🟡 Medium (50-75%): Inferred or partially detected\n")
            f.write("- 🔴 Low (<50%): Assumed or estimated\n\n")
            f.write("*Report generated by XBOQ Pre-Construction Engine*\n")

    def save_overlay(
        self,
        image,
        name: str,
    ) -> Path:
        """Save overlay image."""
        import cv2
        output_path = self.overlays_dir / f"{name}.png"
        cv2.imwrite(str(output_path), image)
        return output_path


def create_boq_csv_content(
    items: List[Any],
) -> str:
    """Generate CSV content as string."""
    lines = ["Item Code,Description,Qty,Unit,Derived From,Confidence"]

    for item in items:
        if isinstance(item, dict):
            line = f"{item.get('item_code', '')},{item.get('description', '')}," \
                   f"{item.get('qty', 0)},{item.get('unit', '')}," \
                   f"{item.get('derived_from', '')},{item.get('confidence', 0.5):.2f}"
        else:
            line = f"{getattr(item, 'item_code', '')},{getattr(item, 'description', '')}," \
                   f"{getattr(item, 'qty', 0)},{getattr(item, 'unit', '')}," \
                   f"{getattr(item, 'derived_from', '')},{getattr(item, 'confidence', 0.5):.2f}"

        lines.append(line)

    return "\n".join(lines)
