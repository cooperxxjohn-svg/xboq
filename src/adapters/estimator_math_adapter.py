"""
Estimator Math Adapter

Maps runner's interface to real measurement_rules modules.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import real modules
from src.measurement_rules.deductions import DeductionEngine
from src.measurement_rules.formwork import FormworkDeriver
from src.measurement_rules.prelims_from_qty import PrelimsFromQuantities
from src.measurement_rules.engine import EstimatorMathEngine, run_estimator_math as _run_estimator_math


def run_estimator_math(
    boq_items: List[Dict] = None,
    project_params: Dict = None,
    output_dir: Path = None,
    rooms: List[Dict] = None,
    openings: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Run estimator math corrections on BOQ items.

    Runner expects this function signature.

    Args:
        boq_items: List of BOQ items
        project_params: Project parameters
        output_dir: Output directory
        rooms: Room data (alternative to boq_items)
        openings: Opening data (alternative to boq_items)

    Returns:
        Dictionary with corrected BOQ and summary
    """
    output_dir = Path(output_dir) if output_dir else None

    result = {
        "summary": {
            "deductions": {"entries": 0, "total_sqm": 0},
            "formwork": {"entries": 0, "total_sqm": 0},
            "prelims": {"entries": 0},
        },
        "corrected_boq": [],
        "deductions_log": [],
        "formwork_items": [],
        "prelims_items": [],
    }

    # If we have boq_items, try to use real engine
    if boq_items:
        try:
            engine_result = _run_estimator_math(boq_items, project_params or {})
            return {
                "summary": engine_result.get("summary", result["summary"]),
                "corrected_boq": engine_result.get("boq_items", boq_items),
            }
        except Exception as e:
            result["error"] = str(e)

    # If we have rooms and openings, generate BOQ items and apply deductions
    if rooms and openings:
        # Generate basic wall-related BOQ items from rooms
        generated_boq = []
        for room in rooms:
            perimeter = room.get("perimeter_m", 0)
            if perimeter > 0:
                wall_area = perimeter * 3.0  # 3m height assumption
                generated_boq.append({
                    "item_id": f"PLT-{room.get('id', 'unknown')}",
                    "description": f"Internal plaster in {room.get('label', 'Room')}",
                    "quantity": wall_area,
                    "unit": "sqm",
                    "room_id": room.get("id"),
                    "rate": 0,
                })
                generated_boq.append({
                    "item_id": f"PNT-{room.get('id', 'unknown')}",
                    "description": f"Painting in {room.get('label', 'Room')}",
                    "quantity": wall_area,
                    "unit": "sqm",
                    "room_id": room.get("id"),
                    "rate": 0,
                })

        # Apply deductions using real engine
        if generated_boq:
            deduction_engine = DeductionEngine()
            adjusted_boq, deduction_log = deduction_engine.apply_deductions(
                boq_items=generated_boq,
                openings=openings,
                structural_elements=None,
            )

            result["corrected_boq"] = adjusted_boq
            result["deductions_log"] = [
                {
                    "deduction_type": d.deduction_type.value,
                    "item_id": d.item_id,
                    "item_description": d.item_description,
                    "opening_id": d.opening_id,
                    "gross_qty": d.gross_qty,
                    "deducted_qty": d.deducted_qty,
                    "net_qty": d.net_qty,
                    "unit": d.unit,
                    "rule_applied": d.rule_applied,
                }
                for d in deduction_log
            ]
            result["summary"]["deductions"]["entries"] = len(deduction_log)
            result["summary"]["deductions"]["total_sqm"] = sum(d.deducted_qty for d in deduction_log)

        # Derive formwork if RCC data exists
        formwork_deriver = FormworkDeriver()
        # This would need RCC quantities - skip if not available

        # Calculate prelims
        prelims_calc = PrelimsFromQuantities()
        if boq_items:
            prelims_result = prelims_calc.calculate(boq_items, project_params or {})
            result["prelims_items"] = prelims_result.get("items", [])
            result["summary"]["prelims"]["entries"] = len(result["prelims_items"])

    # Write output if directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "deductions_applied.json", "w") as f:
            json.dump(result["deductions_log"], f, indent=2)

        with open(output_dir / "estimator_math_summary.json", "w") as f:
            json.dump(result["summary"], f, indent=2)

        # Write deductions CSV
        if result["deductions_log"]:
            import csv
            with open(output_dir / "deductions_applied.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Room ID", "Room Label", "Gross Wall Area", "Total Deduction", "Net Area"])
                for entry in result["deductions_log"]:
                    gross = entry.get("gross_wall_area", 0)
                    ded = sum(d.get("area_sqm", 0) for d in entry.get("deductions", {}).get("items", []))
                    writer.writerow([
                        entry.get("room_id"),
                        entry.get("room_label"),
                        f"{gross:.2f}",
                        f"{ded:.2f}",
                        f"{gross - ded:.2f}",
                    ])

    return result


def apply_measurement_rules(
    rooms: List[Dict],
    openings: List[Dict],
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Apply IS 1200 measurement rules to room/opening data.

    Convenience function that wraps run_estimator_math.

    Args:
        rooms: Room data
        openings: Opening data
        output_dir: Output directory

    Returns:
        Measurement results
    """
    return run_estimator_math(
        rooms=rooms,
        openings=openings,
        output_dir=output_dir,
    )
