"""
Packages Adapter

Maps runner's package export interface to BOQ splitting logic.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import real modules
from src.boq.export import BOQExporter


# Package definitions for Indian construction - match REQUIRED_PACKAGES names
PACKAGE_DEFINITIONS = {
    "rcc_structural": {
        "name": "RCC Structural",
        "file_key": "rcc_structural",
        "prefixes": ["RCC", "CON", "FND", "COL", "BEA", "SLB"],
        "keywords": ["concrete", "rcc", "reinforcement", "foundation", "column", "beam", "slab", "structural"],
    },
    "masonry": {
        "name": "Masonry",
        "file_key": "masonry",
        "prefixes": ["BRK", "MAS", "WAL", "BLK"],
        "keywords": ["brick", "masonry", "block", "aac", "wall construction"],
    },
    "waterproofing": {
        "name": "Waterproofing",
        "file_key": "waterproofing",
        "prefixes": ["WPF", "WP-", "INS"],
        "keywords": ["waterproof", "membrane", "coba", "insulation", "tanking"],
    },
    "flooring": {
        "name": "Flooring",
        "file_key": "flooring",
        "prefixes": ["FLR", "TIL", "MRB", "GRN", "IPS"],
        "keywords": ["floor", "tile", "vitrified", "marble", "granite", "kota", "ips", "anti-skid", "flooring"],
    },
    "doors_windows": {
        "name": "Doors & Windows",
        "file_key": "doors_windows",
        "prefixes": ["DOR", "WIN", "DW-", "D-", "W-", "VNT", "GRL"],
        "keywords": ["door", "window", "ventilator", "grill", "frame", "shutter", "glazing"],
    },
    "wall_finishes": {
        "name": "Wall Finishes",
        "file_key": "wall_finishes",
        "prefixes": ["PLT", "PNT", "DDO", "WAL", "WF-", "CLG"],
        "keywords": ["plaster", "paint", "emulsion", "dado", "wall tile", "texture", "ceiling", "wall finish"],
    },
    "plumbing": {
        "name": "Plumbing",
        "file_key": "plumbing",
        "prefixes": ["PLB", "SAN", "WTR", "DRN"],
        "keywords": ["plumbing", "sanitary", "water", "drainage", "pipe", "fitting", "wc", "basin"],
    },
    "electrical": {
        "name": "Electrical",
        "file_key": "electrical",
        "prefixes": ["ELE", "WIR", "SWT", "LGT"],
        "keywords": ["electrical", "wiring", "switch", "light", "point", "db", "earthing"],
    },
    "external": {
        "name": "External Works",
        "file_key": "external",
        "prefixes": ["EXT", "CMP", "PAV", "LND"],
        "keywords": ["external", "compound", "paving", "landscape", "gate", "boundary"],
    },
    "misc": {
        "name": "Miscellaneous",
        "file_key": "misc",
        "prefixes": ["MSC", "GEN"],
        "keywords": [],
    },
}


def classify_boq_item(item: Dict) -> str:
    """
    Classify a BOQ item into a package.

    Args:
        item: BOQ item dictionary

    Returns:
        Package key
    """
    item_code = item.get("item_code", "").upper()
    description = item.get("description", "").lower()
    category = item.get("category", "").lower()

    # Check by prefix
    for pkg_key, pkg_def in PACKAGE_DEFINITIONS.items():
        for prefix in pkg_def["prefixes"]:
            if item_code.startswith(prefix):
                return pkg_key

    # Check by keywords
    combined_text = f"{description} {category}"
    for pkg_key, pkg_def in PACKAGE_DEFINITIONS.items():
        for keyword in pkg_def["keywords"]:
            if keyword in combined_text:
                return pkg_key

    return "misc"


def run_package_splitter(
    output_dir: Path,
    boq_items: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Split BOQ into trade packages.

    Runner expects this function.

    Args:
        output_dir: Output directory
        boq_items: BOQ items to split (optional, will try to load from output_dir)

    Returns:
        Package split result
    """
    output_dir = Path(output_dir)
    packages_dir = output_dir / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "output_dir": str(packages_dir),
        "packages": {},
        "files_generated": [],
    }

    # Load BOQ items if not provided
    if not boq_items:
        boq_items = []

        # Try to load from various sources
        boq_paths = [
            output_dir / "boq" / "boq_output.json",
            output_dir / "boq" / "boq.json",
        ]

        for bp in boq_paths:
            if bp.exists():
                with open(bp) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        boq_items = data
                    elif isinstance(data, dict):
                        boq_items = data.get("items", data.get("boq_items", []))
                    break

        # If no BOQ file, generate from rooms/openings
        if not boq_items:
            boq_items = _generate_boq_from_scope(output_dir)

    if not boq_items:
        result["message"] = "No BOQ items to split"
        return result

    # Classify items into packages
    packages = {}
    for item in boq_items:
        pkg_key = classify_boq_item(item)
        if pkg_key not in packages:
            packages[pkg_key] = {
                "name": PACKAGE_DEFINITIONS.get(pkg_key, {}).get("name", pkg_key),
                "items": [],
                "total_qty": 0,
                "item_count": 0,
            }
        packages[pkg_key]["items"].append(item)
        packages[pkg_key]["item_count"] += 1

    # Ensure all required packages have at least a placeholder file
    required_pkgs = ["rcc_structural", "masonry", "waterproofing", "flooring", "doors_windows", "wall_finishes"]
    for req_pkg in required_pkgs:
        if req_pkg not in packages:
            packages[req_pkg] = {
                "name": PACKAGE_DEFINITIONS.get(req_pkg, {}).get("name", req_pkg),
                "items": [],
                "total_qty": 0,
                "item_count": 0,
            }

    # Write package CSVs
    for pkg_key, pkg_data in packages.items():
        # Use file_key from definition or pkg_key
        file_key = PACKAGE_DEFINITIONS.get(pkg_key, {}).get("file_key", pkg_key)
        csv_path = packages_dir / f"pkg_{file_key}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Item Code", "Description", "Qty", "Unit", "Room/Location", "Confidence"])

            for item in pkg_data["items"]:
                writer.writerow([
                    item.get("item_code", ""),
                    item.get("description", ""),
                    item.get("qty", 0),
                    item.get("unit", ""),
                    item.get("room_id", item.get("location", "")),
                    item.get("confidence", ""),
                ])

        result["files_generated"].append(f"pkg_{file_key}.csv")
        result["packages"][pkg_key] = {
            "name": pkg_data["name"],
            "item_count": pkg_data["item_count"],
            "file": f"pkg_{file_key}.csv",
        }

    # Write package summary
    with open(packages_dir / "package_summary.json", "w") as f:
        json.dump(result["packages"], f, indent=2)
    result["files_generated"].append("package_summary.json")

    # Write package summary markdown
    with open(packages_dir / "package_summary.md", "w") as f:
        f.write("# Trade Packages Summary\n\n")
        f.write("| Package | Items | File |\n")
        f.write("|---------|-------|------|\n")
        for pkg_key, pkg_info in result["packages"].items():
            f.write(f"| {pkg_info['name']} | {pkg_info['item_count']} | {pkg_info['file']} |\n")
    result["files_generated"].append("package_summary.md")

    return result


def _generate_boq_from_scope(output_dir: Path) -> List[Dict]:
    """Generate basic BOQ items from scope data."""
    boq_items = []
    item_counter = 1

    # Load rooms
    rooms = []
    rooms_path = output_dir / "boq" / "rooms.json"
    if rooms_path.exists():
        with open(rooms_path) as f:
            data = json.load(f)
            rooms = data.get("rooms", [])

    # Load openings
    openings = []
    openings_path = output_dir / "boq" / "openings.json"
    if openings_path.exists():
        with open(openings_path) as f:
            data = json.load(f)
            openings = data.get("openings", [])

    # Generate flooring items
    for room in rooms:
        area = room.get("area_sqm", 0)
        if area > 0:
            boq_items.append({
                "item_code": f"FLR-{item_counter:03d}",
                "description": f"Flooring in {room.get('label', 'Room')}",
                "qty": round(area * 1.05, 2),  # 5% wastage
                "unit": "sqm",
                "room_id": room.get("id"),
                "category": "flooring",
            })
            item_counter += 1

    # Generate painting items
    for room in rooms:
        perimeter = room.get("perimeter_m", 0)
        if perimeter > 0:
            wall_area = perimeter * 3.0  # 3m height
            boq_items.append({
                "item_code": f"PNT-{item_counter:03d}",
                "description": f"Wall painting in {room.get('label', 'Room')}",
                "qty": round(wall_area, 2),
                "unit": "sqm",
                "room_id": room.get("id"),
                "category": "wall_finishes",
            })
            item_counter += 1

    # Generate door items
    for op in openings:
        if op.get("type") == "door":
            boq_items.append({
                "item_code": f"DOR-{item_counter:03d}",
                "description": f"Door {op.get('tag', '')} - {op.get('description', '')}",
                "qty": 1,
                "unit": "nos",
                "room_id": op.get("room_left_id") or op.get("room_right_id"),
                "category": "doors_windows",
            })
            item_counter += 1

    # Generate window items
    for op in openings:
        if op.get("type") == "window":
            area = op.get("width_m", 0) * op.get("height_m", 0)
            boq_items.append({
                "item_code": f"WIN-{item_counter:03d}",
                "description": f"Window {op.get('tag', '')} - {op.get('description', '')}",
                "qty": round(area, 2),
                "unit": "sqm",
                "room_id": op.get("room_left_id"),
                "category": "doors_windows",
            })
            item_counter += 1

    # Generate waterproofing for toilets
    for room in rooms:
        if room.get("room_type") == "toilet":
            area = room.get("area_sqm", 0)
            boq_items.append({
                "item_code": f"WPF-{item_counter:03d}",
                "description": f"Waterproofing in {room.get('label', 'Toilet')}",
                "qty": round(area * 1.1, 2),
                "unit": "sqm",
                "room_id": room.get("id"),
                "category": "waterproofing",
            })
            item_counter += 1

    # Write generated BOQ
    boq_dir = output_dir / "boq"
    boq_dir.mkdir(parents=True, exist_ok=True)

    with open(boq_dir / "boq_output.json", "w") as f:
        json.dump({"items": boq_items}, f, indent=2)

    # Write CSV
    with open(boq_dir / "boq_output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Item Code", "Description", "Qty", "Unit", "Room ID", "Category"])
        for item in boq_items:
            writer.writerow([
                item.get("item_code"),
                item.get("description"),
                item.get("qty"),
                item.get("unit"),
                item.get("room_id"),
                item.get("category"),
            ])

    return boq_items
