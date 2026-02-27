#!/usr/bin/env python3
"""
Project Intake Template Generator

Creates a new project folder structure with all required files.

Usage:
    python scripts/new_project.py --project_id <id>
    python scripts/new_project.py --project_id villa_whitefield_2024

Creates:
    data/projects/<id>/
        drawings/
        owner_docs/
        quotes/
        owner_inputs.yaml
        project_intake.md
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "projects"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


def create_project(project_id: str, overwrite: bool = False) -> Path:
    """
    Create a new project folder with all required files.

    Args:
        project_id: Unique project identifier (e.g., villa_whitefield_2024)
        overwrite: If True, overwrite existing project

    Returns:
        Path to created project directory
    """
    project_dir = DATA_DIR / project_id

    # Check if exists
    if project_dir.exists() and not overwrite:
        print(f"ERROR: Project '{project_id}' already exists at {project_dir}")
        print("Use --overwrite to replace existing project")
        sys.exit(1)

    # Create directory structure
    dirs_to_create = [
        project_dir / "drawings",
        project_dir / "owner_docs",
        project_dir / "quotes",
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # Copy owner_inputs.yaml template
    owner_inputs_template = TEMPLATES_DIR / "owner_inputs_template.yaml"
    owner_inputs_dest = project_dir / "owner_inputs.yaml"

    if owner_inputs_template.exists():
        shutil.copy(owner_inputs_template, owner_inputs_dest)
        # Update project ID in the file
        content = owner_inputs_dest.read_text()
        content = content.replace("name: \"\"", f'name: "{project_id}"')
        owner_inputs_dest.write_text(content)
    else:
        # Create minimal owner_inputs.yaml
        owner_inputs_dest.write_text(generate_minimal_owner_inputs(project_id))

    # Create project_intake.md
    intake_path = project_dir / "project_intake.md"
    intake_path.write_text(generate_project_intake(project_id))

    # Create .gitkeep files in empty dirs
    for d in dirs_to_create:
        gitkeep = d / ".gitkeep"
        if not any(d.iterdir()):
            gitkeep.touch()

    print(f"✅ Project created: {project_dir}")
    print(f"   📁 drawings/     - Place floor plan PDFs/images here")
    print(f"   📁 owner_docs/   - Place tender docs, specs here")
    print(f"   📁 quotes/       - Place subcontractor quotes here")
    print(f"   📄 owner_inputs.yaml - Fill in project specifications")
    print(f"   📄 project_intake.md - Fill in project metadata")

    return project_dir


def generate_minimal_owner_inputs(project_id: str) -> str:
    """Generate minimal owner_inputs.yaml if template missing."""
    return f'''# Owner Inputs for {project_id}
# Fill in known values, leave unknown as null

project:
  name: "{project_id}"
  type: null  # residential / commercial / mixed
  location: null
  built_up_area_sqm: null
  plot_area_sqm: null
  floors: null
  completion_months: null

site:
  address: null
  soil_type: null  # known / unknown / rocky / clay
  water_table_depth_m: null

finishes:
  grade: standard  # economy / standard / premium / luxury
  floor_tile_brand: null
  sanitaryware_brand: null
  paint_brand: null

structural:
  concrete_grade: M25
  steel_grade: Fe500D

mep:
  electrical_load_kw: null
  plumbing_fixtures: null
  hvac_required: false

notes: |
  Add any project-specific notes here.
'''


def generate_project_intake(project_id: str) -> str:
    """Generate project_intake.md template."""
    today = datetime.now().strftime("%Y-%m-%d")

    return f'''# Project Intake: {project_id}

**Created**: {today}
**Status**: Pending

---

## 1. Basic Information

| Field | Value |
|-------|-------|
| **Project ID** | {project_id} |
| **Project Name** |  |
| **Client Name** |  |
| **Location / City** |  |
| **Site Address** |  |

---

## 2. Building Details

| Field | Value |
|-------|-------|
| **Building Type** | ☐ Villa ☐ Apartment ☐ Commercial ☐ Industrial ☐ Mixed |
| **Floors (G + ?)** |  |
| **Built-up Area (sqm)** |  |
| **Plot Area (sqm)** |  |
| **Basement** | ☐ Yes ☐ No |
| **Lift/Elevator** | ☐ Yes ☐ No |

---

## 3. Tender Details

| Field | Value |
|-------|-------|
| **Tender Type** | ☐ Private ☐ CPWD ☐ State PWD ☐ Open ☐ Limited |
| **Bid Deadline** |  |
| **EMD Required** | ☐ Yes ☐ No |
| **Owner BOQ Provided** | ☐ Yes ☐ No |

---

## 4. Drawings Received

| Drawing Type | Received | Sheets | Notes |
|--------------|----------|--------|-------|
| Floor Plans | ☐ Yes ☐ No |  |  |
| Sections | ☐ Yes ☐ No |  |  |
| Elevations | ☐ Yes ☐ No |  |  |
| Structural | ☐ Yes ☐ No |  |  |
| MEP/Plumbing | ☐ Yes ☐ No |  |  |
| Electrical | ☐ Yes ☐ No |  |  |
| Door/Window Schedule | ☐ Yes ☐ No |  |  |
| Finish Schedule | ☐ Yes ☐ No |  |  |
| Site Plan | ☐ Yes ☐ No |  |  |

---

## 5. Known Missing Sheets

List any sheets referenced but not received:
-
-
-

---

## 6. Specifications & Constraints

### Materials / Brands Specified
-
-

### Quality Grade
☐ Economy ☐ Standard ☐ Premium ☐ Luxury

### Special Requirements
-
-

---

## 7. Known Risks / Concerns

-
-
-

---

## 8. Notes

Add any additional context, special instructions, or observations:

```
[Notes here]
```

---

## Checklist Before Processing

- [ ] All floor plans placed in `drawings/`
- [ ] Tender documents in `owner_docs/` (if any)
- [ ] `owner_inputs.yaml` updated with known values
- [ ] Drawing scale verified (typical 1:100)
- [ ] PDF quality checked (not blurry/low-res)

---

*Last Updated: {today}*
'''


def main():
    parser = argparse.ArgumentParser(
        description="Create a new project folder with templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/new_project.py --project_id villa_whitefield_2024
    python scripts/new_project.py --project_id apt_koramangala_phase1 --overwrite
        """
    )

    parser.add_argument(
        "--project_id", "-p",
        required=True,
        help="Unique project identifier (use underscores, no spaces)"
    )

    parser.add_argument(
        "--overwrite", "-o",
        action="store_true",
        help="Overwrite existing project folder"
    )

    args = parser.parse_args()

    # Validate project_id
    project_id = args.project_id.strip()
    if not project_id:
        print("ERROR: project_id cannot be empty")
        sys.exit(1)

    if " " in project_id:
        print("WARNING: Converting spaces to underscores in project_id")
        project_id = project_id.replace(" ", "_")

    create_project(project_id, args.overwrite)


if __name__ == "__main__":
    main()
