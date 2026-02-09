"""
BOM Exporter - Export material estimates to various formats.
"""

import csv
import json
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from .calculator import MaterialEstimate
from .aggregator import AggregatedMaterial, MaterialAggregator


class BOMExporter:
    """Export BOM data to files."""

    def __init__(self):
        self.aggregator = MaterialAggregator()

    def export_all(
        self,
        project_id: str,
        estimates: List[MaterialEstimate],
        aggregated: List[AggregatedMaterial],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Export all BOM outputs."""
        output_dir = Path(output_dir)
        boq_dir = output_dir / "boq"
        boq_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Material estimate CSV
        csv_path = boq_dir / "material_estimate.csv"
        self._export_csv(aggregated, csv_path)
        paths["material_csv"] = csv_path

        # Detailed JSON
        json_path = boq_dir / "material_estimate.json"
        self._export_json(project_id, estimates, aggregated, json_path)
        paths["material_json"] = json_path

        # Summary MD
        md_path = boq_dir / "material_summary.md"
        self._export_markdown(project_id, estimates, aggregated, md_path)
        paths["material_md"] = md_path

        return paths

    def _export_csv(
        self, aggregated: List[AggregatedMaterial], output_path: Path
    ) -> None:
        """Export aggregated materials as CSV."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Category",
                "Material",
                "Quantity",
                "Unit",
                "Procurement Qty",
                "Procurement Unit",
                "Source Items",
            ])

            # Data rows
            for material in aggregated:
                writer.writerow([
                    material.category,
                    material.material_name,
                    material.quantity,
                    material.unit,
                    material.procurement_quantity,
                    material.procurement_unit,
                    material.source_count,
                ])

    def _export_json(
        self,
        project_id: str,
        estimates: List[MaterialEstimate],
        aggregated: List[AggregatedMaterial],
        output_path: Path,
    ) -> None:
        """Export detailed material data as JSON."""
        # Get summaries
        category_summary = self.aggregator.get_category_summary(aggregated)
        major_materials = self.aggregator.get_major_materials(aggregated)

        data = {
            "project_id": project_id,
            "generated": datetime.now().isoformat(),
            "summary": {
                "boq_items_processed": len(estimates),
                "material_lines": len(aggregated),
                "categories": len(category_summary),
            },
            "major_materials": major_materials,
            "category_summary": category_summary,
            "aggregated_materials": [m.to_dict() for m in aggregated],
            "detailed_estimates": [e.to_dict() for e in estimates],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _export_markdown(
        self,
        project_id: str,
        estimates: List[MaterialEstimate],
        aggregated: List[AggregatedMaterial],
        output_path: Path,
    ) -> None:
        """Export material summary as markdown."""
        major = self.aggregator.get_major_materials(aggregated)
        category_summary = self.aggregator.get_category_summary(aggregated)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Material Estimate: {project_id}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **BOQ Items Processed**: {len(estimates)}\n")
            f.write(f"- **Material Lines**: {len(aggregated)}\n")
            f.write(f"- **Categories**: {len(category_summary)}\n\n")

            # Major Materials
            f.write("## Major Materials\n\n")
            f.write("| Material | Quantity | Unit |\n")
            f.write("|----------|----------|------|\n")
            f.write(f"| Cement | {major['cement_bags']:,.0f} | bags |\n")
            f.write(f"| Sand | {major['sand_cum']:,.2f} | cum |\n")
            f.write(f"| Aggregate | {major['aggregate_cum']:,.2f} | cum |\n")
            f.write(f"| Steel | {major['steel_kg']:,.0f} | kg |\n")
            f.write(f"| Bricks | {major['bricks_nos']:,.0f} | nos |\n")
            f.write(f"| Tiles | {major['tiles_sqm']:,.2f} | sqm |\n")
            f.write(f"| Paint | {major['paint_liters']:,.2f} | liters |\n")
            f.write("\n")

            # Procurement Summary
            f.write("## Procurement Summary\n\n")
            cement_mt = major['cement_bags'] * 50 / 1000
            sand_mt = major['sand_cum'] * 1.5
            agg_mt = major['aggregate_cum'] * 1.6
            steel_mt = major['steel_kg'] / 1000

            f.write("| Material | Quantity | Unit | Approx Trucks |\n")
            f.write("|----------|----------|------|---------------|\n")
            f.write(f"| Cement | {cement_mt:,.1f} | MT | {cement_mt/10:,.0f} (10MT loads) |\n")
            f.write(f"| Sand | {sand_mt:,.1f} | MT | {sand_mt/12:,.0f} (12MT loads) |\n")
            f.write(f"| Aggregate | {agg_mt:,.1f} | MT | {agg_mt/12:,.0f} (12MT loads) |\n")
            f.write(f"| Steel | {steel_mt:,.1f} | MT | As required |\n")
            f.write("\n")

            # Category-wise Details
            f.write("## Category-wise Materials\n\n")

            for category in sorted(category_summary.keys()):
                cat_data = category_summary[category]
                f.write(f"### {category.replace('_', ' ').title()}\n\n")

                f.write("| Material | Quantity | Unit |\n")
                f.write("|----------|----------|------|\n")

                for mat in cat_data["materials"][:15]:  # Limit per category
                    f.write(f"| {mat['name']} | {mat['quantity']:,.2f} | {mat['unit']} |\n")

                if len(cat_data["materials"]) > 15:
                    f.write(f"| ... | {len(cat_data['materials']) - 15} more items | |\n")

                f.write("\n")

            # Notes
            f.write("## Notes\n\n")
            f.write("1. Material quantities are derived using CPWD/DSR 2024 norms.\n")
            f.write("2. Wastage factors are included as per standard practice.\n")
            f.write("3. Actual procurement quantities may vary based on site conditions.\n")
            f.write("4. Truck capacities are approximate for planning purposes.\n")
            f.write("5. All quantities are for estimation only - verify with detailed BOQ.\n")
