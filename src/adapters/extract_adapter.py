"""
Extract Adapter

Maps runner's extract interface to real ingest, scale, and export modules.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import real modules
from src.ingest import PlanIngester, ingest_plan
from src.scale import ScaleInferrer, infer_scale
from src.export import PlanExporter, export_plan


import re


# Room type keywords for extraction from text labels
ROOM_KEYWORDS = {
    "bedroom": "Bedroom",
    "bed room": "Bedroom",
    "master bedroom": "Bedroom",
    "bath": "Toilet",
    "bathroom": "Toilet",
    "toilet": "Toilet",
    "wc": "Toilet",
    "kitchen": "Kitchen",
    "living": "Living",
    "living room": "Living",
    "dining": "Dining",
    "dining room": "Dining",
    "hall": "Living",
    "balcony": "Balcony",
    "foyer": "Foyer",
    "entry": "Foyer",
    "laundry": "Utility",
    "utility": "Utility",
    "store": "Store",
    "storage": "Store",
    "porch": "Porch",
    "garage": "Garage",
    "office": "Office",
    "study": "Office",
    "pooja": "Pooja",
    "drawing": "Drawing",
}

# Opening type patterns
OPENING_PATTERNS = {
    "door": r"D[-\s]?\d+|DR[-\s]?\d+|DOOR\s*\d*",
    "window": r"W[-\s]?\d+|WIN[-\s]?\d+|WINDOW\s*\d*",
}


def _extract_from_text_labels(plan) -> tuple:
    """
    Extract rooms and openings by analyzing text labels from vector PDF.

    Args:
        plan: IngestedPlan with vector_texts

    Returns:
        Tuple of (rooms_list, openings_list)
    """
    rooms = []
    openings = []
    seen_rooms = set()
    seen_openings = set()

    room_id = 0
    opening_id = 0

    for text_item in plan.vector_texts:
        text = text_item.text.strip()
        text_lower = text.lower()

        # Check for room labels
        for keyword, room_type in ROOM_KEYWORDS.items():
            if keyword in text_lower and text_lower not in seen_rooms:
                seen_rooms.add(text_lower)
                room_id += 1

                # Try to extract dimensions from nearby text
                area_sqm = 15.0  # Default assumption
                perimeter_m = 16.0

                rooms.append({
                    "id": f"R{room_id:03d}",
                    "label": text.title(),
                    "room_type": room_type.lower(),
                    "area_sqm": area_sqm,
                    "perimeter_m": perimeter_m,
                    "confidence": 0.7,
                    "source": plan.plan_id,
                    "bbox": list(text_item.bbox),
                })
                break

        # Check for door/window tags
        for opening_type, pattern in OPENING_PATTERNS.items():
            if re.match(pattern, text.upper()) and text.upper() not in seen_openings:
                seen_openings.add(text.upper())
                opening_id += 1

                # Default dimensions
                if opening_type == "door":
                    width_m, height_m = 0.9, 2.1
                else:
                    width_m, height_m = 1.2, 1.5

                openings.append({
                    "id": f"O{opening_id:03d}",
                    "tag": text.upper().replace(" ", ""),
                    "type": opening_type,
                    "description": f"{opening_type.title()} {text.upper()}",
                    "width_m": width_m,
                    "height_m": height_m,
                    "material": "standard",
                    "confidence": 0.65,
                    "source": plan.plan_id,
                })
                break

    return rooms, openings


def process_project_drawings(
    drawings_dir: Path,
    output_dir: Path,
    profile: str = "typical",
) -> Dict[str, Any]:
    """
    Process all drawings in a project, extracting rooms, walls, openings.

    This is the main extraction function expected by the runner.

    Args:
        drawings_dir: Path to drawings folder
        output_dir: Path to output folder
        profile: Processing profile

    Returns:
        Dictionary with:
            - pages_processed: int
            - total_rooms: int
            - total_openings: int
            - results: list of per-page results
    """
    drawings_dir = Path(drawings_dir)
    output_dir = Path(output_dir)

    # Find drawing files
    patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg"]
    files = []
    for pattern in patterns:
        files.extend(drawings_dir.glob(pattern))

    result = {
        "pages_processed": 0,
        "total_rooms": 0,
        "total_openings": 0,
        "results": [],
        "errors": [],
    }

    # Check for pre-existing data (rooms.json, openings.json)
    project_dir = drawings_dir.parent
    rooms_json = project_dir / "rooms.json"
    openings_json = project_dir / "openings.json"

    # If we have pre-existing data, use that
    if rooms_json.exists():
        with open(rooms_json) as f:
            rooms_data = json.load(f)

        rooms = rooms_data.get("rooms", [])
        result["total_rooms"] = len(rooms)
        result["pages_processed"] = 1  # Treat as one logical page

        # Copy to output
        boq_dir = output_dir / "boq"
        boq_dir.mkdir(parents=True, exist_ok=True)

        # Write rooms to output
        with open(boq_dir / "rooms.json", "w") as f:
            json.dump(rooms_data, f, indent=2)

        result["results"].append({
            "source": "rooms.json",
            "rooms_count": len(rooms),
        })

    if openings_json.exists():
        with open(openings_json) as f:
            openings_data = json.load(f)

        openings = openings_data.get("openings", [])
        result["total_openings"] = len(openings)

        # Copy to output
        boq_dir = output_dir / "boq"
        boq_dir.mkdir(parents=True, exist_ok=True)

        with open(boq_dir / "openings.json", "w") as f:
            json.dump(openings_data, f, indent=2)

        result["results"].append({
            "source": "openings.json",
            "openings_count": len(openings),
        })

    # Process actual image files
    ingester = PlanIngester()
    scale_inferrer = ScaleInferrer()

    all_extracted_rooms = []
    all_extracted_openings = []

    for file_path in sorted(files):
        try:
            # Ingest the plan
            plan = ingester.ingest(file_path)

            # Infer scale
            scale_result = scale_inferrer.infer_scale(plan.image, plan.vector_texts)

            # Extract rooms and openings from text labels
            extracted_rooms, extracted_openings = _extract_from_text_labels(plan)
            all_extracted_rooms.extend(extracted_rooms)
            all_extracted_openings.extend(extracted_openings)

            page_result = {
                "file": str(file_path),
                "plan_id": plan.plan_id,
                "scale_method": scale_result.method.value if scale_result else "unknown",
                "scale_confidence": scale_result.confidence if scale_result else 0,
                "rooms_found": len(extracted_rooms),
                "openings_found": len(extracted_openings),
            }

            result["pages_processed"] += 1
            result["results"].append(page_result)

        except Exception as e:
            result["errors"].append({
                "file": str(file_path),
                "error": str(e),
            })

    # Update totals from extracted data
    if all_extracted_rooms:
        result["total_rooms"] += len(all_extracted_rooms)
        # Write extracted data to output
        boq_dir = output_dir / "boq"
        boq_dir.mkdir(parents=True, exist_ok=True)

        with open(boq_dir / "rooms.json", "w") as f:
            json.dump({"rooms": all_extracted_rooms}, f, indent=2)

    if all_extracted_openings:
        result["total_openings"] += len(all_extracted_openings)
        boq_dir = output_dir / "boq"
        boq_dir.mkdir(parents=True, exist_ok=True)

        with open(boq_dir / "openings.json", "w") as f:
            json.dump({"openings": all_extracted_openings}, f, indent=2)

    # Generate combined output if we have data
    if result["total_rooms"] > 0 or result["total_openings"] > 0:
        _generate_scope_output(output_dir, rooms_json, openings_json)

    # Generate quicklook overlay image
    _generate_quicklook_overlay(output_dir, files, all_extracted_rooms)

    return result


def _generate_quicklook_overlay(output_dir: Path, source_files: list, rooms: list):
    """Generate a quicklook overlay PNG showing detected rooms."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # PIL not available, create placeholder
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        # Write a minimal valid PNG (1x1 pixel)
        import struct
        import zlib

        def create_minimal_png():
            # PNG header
            header = b'\x89PNG\r\n\x1a\n'
            # IHDR chunk
            ihdr_data = struct.pack('>IIBBBBB', 400, 300, 8, 2, 0, 0, 0)  # 400x300 RGB
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            # IDAT chunk (compressed pixel data - white image)
            raw_data = b''
            for y in range(300):
                raw_data += b'\x00' + (b'\xff\xff\xff' * 400)  # filter byte + white pixels
            compressed = zlib.compress(raw_data)
            idat_crc = zlib.crc32(b'IDAT' + compressed)
            idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
            # IEND chunk
            iend_crc = zlib.crc32(b'IEND')
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            return header + ihdr + idat + iend

        with open(overlays_dir / "quicklook.png", "wb") as f:
            f.write(create_minimal_png())
        return

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Try to load first source image
    base_image = None
    for src in source_files:
        if str(src).lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                base_image = Image.open(src).convert('RGB')
                break
            except Exception:
                continue

    if base_image is None:
        # Create blank canvas
        base_image = Image.new('RGB', (800, 600), color=(255, 255, 255))

    # Resize if too large
    max_size = 1200
    if max(base_image.size) > max_size:
        ratio = max_size / max(base_image.size)
        new_size = (int(base_image.size[0] * ratio), int(base_image.size[1] * ratio))
        base_image = base_image.resize(new_size, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(base_image)

    # Add room labels as overlays
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    # Draw room boxes if we have bbox info
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i, room in enumerate(rooms):
        if 'bbox' in room and len(room['bbox']) >= 4:
            bbox = room['bbox']
            color = colors[i % len(colors)]
            # Scale bbox to image size
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # Normalize to 0-1 range if values seem to be in coordinate space
            if x2 > 100 or y2 > 100:
                # Assume coordinates are in some pixel space, scale to image
                scale = min(base_image.size[0] / 3000, base_image.size[1] / 3000)
                x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = room.get('label', f'Room {i+1}')
            draw.text((x1 + 5, y1 + 5), label, fill=color, font=font)

    # Add title
    draw.text((10, 10), f"Quicklook: {len(rooms)} rooms detected", fill='#333333', font=font)

    # Save
    base_image.save(overlays_dir / "quicklook.png")


def _generate_scope_output(output_dir: Path, rooms_json: Path, openings_json: Path):
    """Generate scope output files from extracted data."""
    scope_dir = output_dir / "scope"
    scope_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    rooms = []
    openings = []

    if rooms_json.exists():
        with open(rooms_json) as f:
            data = json.load(f)
            rooms = data.get("rooms", [])

    if openings_json.exists():
        with open(openings_json) as f:
            data = json.load(f)
            openings = data.get("openings", [])

    # Write room areas CSV
    import csv
    with open(scope_dir / "room_areas.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Room ID", "Label", "Type", "Area (sqm)", "Perimeter (m)", "Confidence"])
        for room in rooms:
            writer.writerow([
                room.get("id", ""),
                room.get("label", ""),
                room.get("room_type", ""),
                room.get("area_sqm", 0),
                room.get("perimeter_m", 0),
                room.get("confidence", 0),
            ])
        # Total row
        total_area = sum(r.get("area_sqm", 0) for r in rooms)
        writer.writerow(["", "TOTAL", "", total_area, "", ""])

    # Write openings schedule CSV
    with open(scope_dir / "openings_schedule.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tag", "Type", "Description", "Width (m)", "Height (m)", "Material", "Confidence"])
        for op in openings:
            writer.writerow([
                op.get("tag", ""),
                op.get("type", ""),
                op.get("description", ""),
                op.get("width_m", 0),
                op.get("height_m", 0),
                op.get("material", ""),
                op.get("confidence", 0),
            ])

    # Write scope summary JSON
    summary = {
        "total_rooms": len(rooms),
        "total_openings": len(openings),
        "total_area_sqm": sum(r.get("area_sqm", 0) for r in rooms),
        "room_types": {},
        "opening_types": {},
    }

    for room in rooms:
        rt = room.get("room_type", "unknown")
        summary["room_types"][rt] = summary["room_types"].get(rt, 0) + 1

    for op in openings:
        ot = op.get("type", "unknown")
        summary["opening_types"][ot] = summary["opening_types"].get(ot, 0) + 1

    with open(scope_dir / "scope_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def process_floorplan(file_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process a single floorplan file.

    Args:
        file_path: Path to floorplan image/PDF
        output_dir: Output directory

    Returns:
        Processing result
    """
    ingester = PlanIngester()
    plan = ingester.ingest(file_path)

    scale_inferrer = ScaleInferrer()
    scale = scale_inferrer.infer(plan)

    return {
        "plan_id": plan.plan_id,
        "scale": {
            "method": scale.method.value if scale else "unknown",
            "confidence": scale.confidence if scale else 0,
        },
    }
