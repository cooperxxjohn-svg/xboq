"""
Estimator Math Engine

Main integration module that applies Indian method-of-measurement rules
to produce estimator-accurate quantities.

Components:
1. Deduction Engine (CPWD-style deductions)
2. Formwork Deriver (RCC -> formwork/staging)
3. Openings Deductor (schedule-based deductions)
4. External Provisionals (auto-generate if no site plan)
5. Prelims from Quantities (derived, not template)
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from .deductions import DeductionEngine, DeductionThresholds
from .formwork import FormworkDeriver, FormworkConstants
from .openings_deduct import OpeningsDeductor
from .external_provisionals import ExternalProvisionals
from .prelims_from_qty import PrelimsFromQuantities

logger = logging.getLogger(__name__)


class EstimatorMathEngine:
    """
    Applies Indian method-of-measurement rules to BOQ.

    Transforms a raw BOQ into estimator-accurate quantities by:
    1. Applying CPWD-style deductions for openings
    2. Deriving formwork from RCC quantities
    3. Using opening schedules for accurate deductions
    4. Adding external works provisionals if missing
    5. Calculating prelims from actual quantities
    """

    def __init__(
        self,
        deduction_thresholds: DeductionThresholds = None,
        formwork_constants: FormworkConstants = None,
    ):
        self.deduction_engine = DeductionEngine(deduction_thresholds)
        self.formwork_deriver = FormworkDeriver(formwork_constants)
        self.openings_deductor = OpeningsDeductor()
        self.external_provisionals = ExternalProvisionals()
        self.prelims_calculator = PrelimsFromQuantities()

        self.results = {}

    def process(
        self,
        boq_items: List[Dict[str, Any]],
        openings: List[Dict[str, Any]] = None,
        openings_csv_path: Path = None,
        structural_elements: Dict[str, Any] = None,
        project_params: Dict[str, Any] = None,
        output_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Process BOQ with estimator math rules.

        Args:
            boq_items: Original BOQ items
            openings: List of openings (doors/windows)
            openings_csv_path: Path to openings_schedule.csv
            structural_elements: Structural info (beams, columns)
            project_params: Project parameters (area, floors, duration, etc.)
            output_dir: Output directory for logs

        Returns:
            Results with adjusted BOQ, derived items, and logs
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        project_params = project_params or {}
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "original_items": len(boq_items),
            "adjustments": [],
        }

        # Step 1: Load openings
        if openings_csv_path and Path(openings_csv_path).exists():
            self.openings_deductor.load_from_csv(openings_csv_path)
        elif openings:
            self.openings_deductor.load_from_list(openings)

        all_openings = [
            {
                "id": o.tag,
                "type": o.opening_type,
                "width_mm": o.width_mm,
                "height_mm": o.height_mm,
                "sill_height_mm": o.sill_height_mm,
                "location": o.location,
            }
            for o in self.openings_deductor.openings
        ]

        # Step 2: Apply CPWD-style deductions
        logger.info("Applying CPWD-style deductions...")
        adjusted_boq, deduction_log = self.deduction_engine.apply_deductions(
            boq_items, all_openings, structural_elements
        )

        if output_dir:
            self.deduction_engine.export_deduction_log(output_dir / "deduction_log.csv")

        self.results["deductions"] = {
            "entries": len(deduction_log),
            "total_deducted": sum(d.deducted_qty for d in deduction_log),
        }

        # Step 3: Apply opening schedule deductions (more accurate)
        logger.info("Applying opening schedule deductions...")
        adjusted_boq = self.openings_deductor.apply_to_boq(adjusted_boq)

        if output_dir:
            self.openings_deductor.export_deductions_csv(output_dir / "opening_deductions.csv")

        self.results["opening_deductions"] = self.openings_deductor.get_summary()

        # Step 4: Derive formwork from RCC
        logger.info("Deriving formwork from RCC quantities...")
        rcc_items = [i for i in adjusted_boq if self._is_rcc_item(i)]
        formwork_items = self.formwork_deriver.derive_formwork(rcc_items, structural_elements)

        self.results["formwork"] = self.formwork_deriver.get_summary()

        # Step 5: Generate external provisionals if needed
        has_site_plan = project_params.get("has_site_plan", False)
        existing_external = [i.get("description", "") for i in adjusted_boq if i.get("package") == "external"]

        logger.info("Checking external works provisionals...")
        external_items = self.external_provisionals.generate_provisionals(
            plot_area_sqm=project_params.get("plot_area_sqm"),
            built_up_area_sqm=project_params.get("built_up_area_sqm"),
            num_floors=project_params.get("num_floors", 1),
            num_units=project_params.get("num_units"),
            has_sewer_connection=project_params.get("has_sewer_connection", True),
            has_site_plan=has_site_plan,
            existing_external_items=existing_external,
        )

        self.results["external_provisionals"] = self.external_provisionals.get_summary()

        # Step 6: Calculate prelims from quantities
        logger.info("Calculating prelims from quantities...")
        rcc_volume = sum(i.get("quantity", 0) for i in rcc_items if i.get("unit", "").lower() == "cum")
        steel_qty_kg = sum(i.get("quantity", 0) for i in adjusted_boq if "steel" in i.get("description", "").lower() and i.get("unit", "").lower() == "kg")
        steel_qty_mt = steel_qty_kg / 1000

        formwork_area = sum(i.get("quantity", 0) for i in formwork_items)
        project_value = sum(i.get("amount", 0) for i in adjusted_boq)

        prelims_items = self.prelims_calculator.calculate_prelims(
            rcc_volume_cum=rcc_volume,
            steel_qty_mt=steel_qty_mt,
            built_up_area_sqm=project_params.get("built_up_area_sqm", 1000),
            plot_area_sqm=project_params.get("plot_area_sqm"),
            num_floors=project_params.get("num_floors", 1),
            project_duration_months=project_params.get("duration_months", 12),
            project_value_inr=project_value,
            formwork_area_sqm=formwork_area,
        )

        self.results["prelims"] = self.prelims_calculator.get_summary()

        # Combine all items
        final_boq = adjusted_boq + formwork_items + external_items

        # Calculate totals
        boq_total = sum(i.get("amount", 0) for i in final_boq)
        prelims_total = sum(i.get("amount", 0) for i in prelims_items)

        self.results["final_boq"] = {
            "items": len(final_boq),
            "boq_total": round(boq_total, 2),
            "prelims_total": round(prelims_total, 2),
            "grand_total": round(boq_total + prelims_total, 2),
        }

        # Export results
        if output_dir:
            self._export_results(output_dir, final_boq, prelims_items)

        return {
            "adjusted_boq": final_boq,
            "prelims_items": prelims_items,
            "formwork_items": formwork_items,
            "external_items": external_items,
            "summary": self.results,
        }

    def _is_rcc_item(self, item: Dict[str, Any]) -> bool:
        """Check if item is an RCC item."""
        desc = item.get("description", "").lower()
        return "rcc" in desc or "concrete" in desc

    def _export_results(
        self,
        output_dir: Path,
        boq: List[Dict],
        prelims: List[Dict],
    ) -> None:
        """Export results to files."""
        # Summary JSON
        with open(output_dir / "estimator_math_summary.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Adjusted BOQ CSV - collect all fieldnames
        if boq:
            all_fields = set()
            for item in boq:
                all_fields.update(item.keys())
            # Remove complex nested fields
            all_fields.discard("deduction_details")

            fieldnames = sorted(all_fields)
            with open(output_dir / "adjusted_boq.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(boq)

        # Prelims CSV
        with open(output_dir / "prelims_from_qty.csv", "w", newline="") as f:
            if prelims:
                writer = csv.DictWriter(f, fieldnames=prelims[0].keys())
                writer.writeheader()
                writer.writerows(prelims)

        # Summary markdown
        self._export_summary_md(output_dir)

        logger.info(f"Estimator math results exported to {output_dir}")

    def _export_summary_md(self, output_dir: Path) -> None:
        """Export summary as markdown."""
        r = self.results

        with open(output_dir / "estimator_math_summary.md", "w") as f:
            f.write("# Estimator Math Engine - Summary\n\n")
            f.write(f"**Generated**: {r.get('timestamp', '')}\n\n")

            f.write("## Processing Summary\n\n")
            f.write(f"- Original BOQ items: {r.get('original_items', 0)}\n")
            f.write(f"- Final BOQ items: {r.get('final_boq', {}).get('items', 0)}\n\n")

            f.write("## Deductions Applied (IS 1200)\n\n")
            ded = r.get("deductions", {})
            f.write(f"- Deduction entries: {ded.get('entries', 0)}\n")
            f.write(f"- Total area deducted: {ded.get('total_deducted', 0):.2f} sqm\n\n")

            f.write("## Opening Deductions\n\n")
            od = r.get("opening_deductions", {})
            f.write(f"- Total openings: {od.get('total_openings', 0)}\n")
            f.write(f"- Doors: {od.get('doors', 0)}\n")
            f.write(f"- Windows: {od.get('windows', 0)}\n")
            f.write(f"- Total door area: {od.get('total_door_area', 0):.2f} sqm\n")
            f.write(f"- Total window area: {od.get('total_window_area', 0):.2f} sqm\n\n")

            f.write("## Formwork Derived\n\n")
            fw = r.get("formwork", {})
            f.write(f"- Items derived: {fw.get('items_derived', 0)}\n")
            f.write(f"- Total formwork: {fw.get('total_formwork_sqm', 0):.2f} sqm\n")
            f.write(f"- Formwork value: {fw.get('total_value', 0):,.2f} INR\n\n")

            f.write("## External Works Provisionals\n\n")
            ext = r.get("external_provisionals", {})
            f.write(f"- Items generated: {ext.get('items_generated', 0)}\n")
            f.write(f"- Provisional value: {ext.get('total_provisional_value', 0):,.2f} INR\n")
            if ext.get("note"):
                f.write(f"- Note: {ext.get('note')}\n\n")

            f.write("## Prelims (Derived from Quantities)\n\n")
            pr = r.get("prelims", {})
            f.write(f"- Total items: {pr.get('total_items', 0)}\n")
            f.write(f"- Total value: {pr.get('total_value', 0):,.2f} INR\n")
            if pr.get("by_category"):
                f.write("\n| Category | Items | Amount |\n")
                f.write("|----------|-------|--------|\n")
                for cat, data in pr.get("by_category", {}).items():
                    f.write(f"| {cat.title()} | {data['items']} | {data['amount']:,.2f} |\n")
            f.write("\n")

            f.write("## Final Totals\n\n")
            final = r.get("final_boq", {})
            f.write(f"| Component | Value (INR) |\n")
            f.write(f"|-----------|-------------|\n")
            f.write(f"| BOQ Total | {final.get('boq_total', 0):,.2f} |\n")
            f.write(f"| Prelims Total | {final.get('prelims_total', 0):,.2f} |\n")
            f.write(f"| **Grand Total** | **{final.get('grand_total', 0):,.2f}** |\n")


def run_estimator_math(
    boq_items: List[Dict[str, Any]],
    openings: List[Dict[str, Any]] = None,
    openings_csv_path: Path = None,
    structural_elements: Dict[str, Any] = None,
    project_params: Dict[str, Any] = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Convenience function to run estimator math engine.

    Args:
        boq_items: Original BOQ items
        openings: List of openings (doors/windows)
        openings_csv_path: Path to openings_schedule.csv
        structural_elements: Structural info (beams, columns)
        project_params: Project parameters
        output_dir: Output directory

    Returns:
        Processing results
    """
    engine = EstimatorMathEngine()
    return engine.process(
        boq_items=boq_items,
        openings=openings,
        openings_csv_path=openings_csv_path,
        structural_elements=structural_elements,
        project_params=project_params,
        output_dir=output_dir,
    )
