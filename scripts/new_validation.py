#!/usr/bin/env python3
"""
New Validation Sheet Generator

Creates a project-specific validation sheet from template.
Used to compare engine outputs against manual takeoff.

Usage:
    python scripts/new_validation.py --project_id villa_001
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


TEMPLATE_PATH = Path(__file__).parent.parent / "validation_template.csv"
OUTPUT_BASE = Path(__file__).parent.parent / "output"


def create_validation_sheet(project_id: str) -> Path:
    """
    Create validation sheet for a project.

    Args:
        project_id: Project identifier

    Returns:
        Path to created validation sheet
    """
    # Create output directory
    validation_dir = OUTPUT_BASE / project_id / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    output_path = validation_dir / "validation.csv"

    # Check if template exists
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: Template not found at {TEMPLATE_PATH}")
        return None

    # Copy template
    shutil.copy(TEMPLATE_PATH, output_path)

    # Also create a header file with project info
    header_path = validation_dir / "validation_info.txt"
    with open(header_path, "w") as f:
        f.write(f"VALIDATION SHEET\n")
        f.write(f"================\n\n")
        f.write(f"Project ID: {project_id}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Instructions:\n")
        f.write(f"1. Run the full pipeline: python run_full_project.py --project_id {project_id}\n")
        f.write(f"2. Fill in Engine_Value column from outputs\n")
        f.write(f"3. Fill in Manual_Value from manual takeoff\n")
        f.write(f"4. Calculate Percent_Error = (Engine - Manual) / Manual × 100\n")
        f.write(f"5. Add notes for any discrepancies > 5%\n\n")
        f.write(f"Key output files to reference:\n")
        f.write(f"- output/{project_id}/boq/boq_output.csv\n")
        f.write(f"- output/{project_id}/scope/room_areas.csv\n")
        f.write(f"- output/{project_id}/scope/openings_schedule.csv\n")
        f.write(f"- output/{project_id}/measurement/deductions_applied.csv\n")

    print(f"Created validation sheet: {output_path}")
    print(f"Created validation info: {header_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create validation sheet for a project"
    )
    parser.add_argument(
        "--project_id",
        required=True,
        help="Project identifier (e.g., villa_001)"
    )

    args = parser.parse_args()

    output_path = create_validation_sheet(args.project_id)

    if output_path:
        print(f"\nNext steps:")
        print(f"1. Run pipeline: python run_full_project.py --project_id {args.project_id}")
        print(f"2. Open {output_path} and fill in values")
        print(f"3. Review any Percent_Error > 5%")


if __name__ == "__main__":
    main()
