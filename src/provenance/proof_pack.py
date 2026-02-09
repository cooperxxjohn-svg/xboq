"""
Proof Pack Generator

Generates overlay images and proof documentation for measurement verification.

Outputs:
- walls overlay per page
- rooms overlay per page
- openings overlay per page
- scale overlay (showing dimension used)
- proof_pack.md with BOQ line -> page -> overlay -> method -> confidence table
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OverlaySpec:
    """Specification for an overlay image."""
    page_index: int
    source_file: str
    overlay_type: str  # walls, rooms, openings, scale
    output_path: Path
    items_drawn: int = 0


def generate_proof_pack(
    output_dir: Path,
    boq_items: List[Dict[str, Any]],
    rooms: List[Dict[str, Any]],
    openings: List[Dict[str, Any]],
    source_files: List[Path],
) -> Dict[str, Any]:
    """
    Generate complete proof pack with overlays and documentation.

    Args:
        output_dir: Output directory
        boq_items: BOQ items with provenance
        rooms: Extracted room data
        openings: Extracted opening data
        source_files: Source drawing files

    Returns:
        Summary of generated proof pack
    """
    output_dir = Path(output_dir)
    proof_dir = output_dir / "proof"
    proof_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "overlays_generated": [],
        "proof_pack_path": None,
        "pages_processed": 0,
    }

    # Group rooms and openings by page
    rooms_by_page = _group_by_page(rooms)
    openings_by_page = _group_by_page(openings)

    # Try to generate overlays for each source file
    for i, src_file in enumerate(source_files):
        page_rooms = rooms_by_page.get(i, [])
        page_openings = openings_by_page.get(i, [])

        if page_rooms or page_openings:
            # Generate room overlay
            room_overlay = _generate_room_overlay(
                overlays_dir, i, src_file, page_rooms
            )
            if room_overlay:
                result["overlays_generated"].append(room_overlay)

            # Generate opening overlay
            if page_openings:
                opening_overlay = _generate_opening_overlay(
                    overlays_dir, i, src_file, page_openings
                )
                if opening_overlay:
                    result["overlays_generated"].append(opening_overlay)

            result["pages_processed"] += 1

    # Generate wall overlay (combined from room boundaries)
    wall_overlay = _generate_wall_overlay(overlays_dir, rooms)
    if wall_overlay:
        result["overlays_generated"].append(wall_overlay)

    # Generate scale overlay
    scale_overlay = _generate_scale_overlay(overlays_dir, output_dir)
    if scale_overlay:
        result["overlays_generated"].append(scale_overlay)

    # Generate proof_pack.md
    proof_pack_path = _generate_proof_pack_md(
        proof_dir, boq_items, rooms, openings, result["overlays_generated"]
    )
    result["proof_pack_path"] = str(proof_pack_path)

    return result


def _group_by_page(items: List[Dict]) -> Dict[int, List[Dict]]:
    """Group items by page number."""
    by_page = {}
    for item in items:
        page = item.get("page", 0)
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(item)
    return by_page


def _generate_room_overlay(
    overlays_dir: Path,
    page_index: int,
    source_file: Path,
    rooms: List[Dict],
) -> Optional[str]:
    """Generate room overlay image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Create placeholder text file
        overlay_path = overlays_dir / f"rooms_page_{page_index}.txt"
        with open(overlay_path, "w") as f:
            f.write(f"Room overlay for page {page_index}\n")
            f.write(f"Rooms: {len(rooms)}\n")
            for room in rooms:
                f.write(f"  - {room.get('label', 'Unknown')}: bbox={room.get('bbox', 'N/A')}\n")
        return f"rooms_page_{page_index}.txt"

    # Create overlay image
    overlay_path = overlays_dir / f"rooms_page_{page_index}.png"

    # Try to load source image
    img = None
    if source_file and Path(source_file).exists():
        try:
            ext = str(source_file).lower()
            if ext.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(source_file).convert('RGBA')
        except Exception:
            pass

    if img is None:
        # Create blank canvas
        img = Image.new('RGBA', (800, 600), (255, 255, 255, 255))

    # Create overlay layer
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw room boxes
    colors = [
        (255, 107, 107, 100),  # Red
        (78, 205, 196, 100),   # Teal
        (69, 183, 209, 100),   # Blue
        (150, 206, 180, 100),  # Green
        (255, 234, 167, 100),  # Yellow
    ]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    for i, room in enumerate(rooms):
        bbox = room.get("bbox", [])
        if len(bbox) >= 4:
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Scale if coordinates are large
            if x2 > img.size[0] * 2:
                scale = img.size[0] / 3000
                x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(color[0], color[1], color[2], 255), width=2)

            label = room.get("label", f"Room {i+1}")
            draw.text((x1 + 5, y1 + 5), label, fill=(0, 0, 0, 255), font=font)

    # Composite overlay onto image
    img = Image.alpha_composite(img, overlay)
    img.save(overlay_path)

    return f"rooms_page_{page_index}.png"


def _generate_opening_overlay(
    overlays_dir: Path,
    page_index: int,
    source_file: Path,
    openings: List[Dict],
) -> Optional[str]:
    """Generate opening overlay image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        overlay_path = overlays_dir / f"openings_page_{page_index}.txt"
        with open(overlay_path, "w") as f:
            f.write(f"Opening overlay for page {page_index}\n")
            f.write(f"Openings: {len(openings)}\n")
            for op in openings:
                f.write(f"  - {op.get('type', 'Unknown')}: {op.get('width', 'N/A')}x{op.get('height', 'N/A')}\n")
        return f"openings_page_{page_index}.txt"

    overlay_path = overlays_dir / f"openings_page_{page_index}.png"

    # Create image
    img = Image.new('RGBA', (800, 600), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except Exception:
        font = ImageFont.load_default()

    for i, op in enumerate(openings):
        bbox = op.get("bbox", [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            color = (255, 165, 0, 200)  # Orange
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 12), f"{op.get('type', 'O')}", fill=(255, 165, 0, 255), font=font)

    img.save(overlay_path)
    return f"openings_page_{page_index}.png"


def _generate_wall_overlay(
    overlays_dir: Path,
    rooms: List[Dict],
) -> Optional[str]:
    """Generate combined wall overlay from room boundaries."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        overlay_path = overlays_dir / "walls_combined.txt"
        with open(overlay_path, "w") as f:
            f.write("Wall overlay\n")
            f.write(f"Total rooms: {len(rooms)}\n")
            f.write(f"Estimated wall segments: {len(rooms) * 4}\n")
        return "walls_combined.txt"

    overlay_path = overlays_dir / "walls_combined.png"

    img = Image.new('RGBA', (800, 600), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    wall_color = (50, 50, 50, 255)

    for room in rooms:
        bbox = room.get("bbox", [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            # Scale
            if x2 > 800:
                scale = 800 / 3000
                x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
            draw.rectangle([x1, y1, x2, y2], outline=wall_color, width=2)

    img.save(overlay_path)
    return "walls_combined.png"


def _generate_scale_overlay(
    overlays_dir: Path,
    output_dir: Path,
) -> Optional[str]:
    """Generate scale verification overlay."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        overlay_path = overlays_dir / "scale_info.txt"
        with open(overlay_path, "w") as f:
            f.write("Scale Information\n")
            # Try to load scale info
            meas_file = output_dir / "measurement" / "estimator_math_summary.json"
            if meas_file.exists():
                with open(meas_file) as mf:
                    data = json.load(mf)
                    f.write(f"Scale: 1:{data.get('scale', 'Unknown')}\n")
                    f.write(f"Basis: {data.get('scale_basis', 'Unknown')}\n")
            else:
                f.write("Scale: Unknown\n")
        return "scale_info.txt"

    overlay_path = overlays_dir / "scale_info.png"

    img = Image.new('RGBA', (400, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()
        font_large = font

    # Load scale info
    scale_value = "Unknown"
    scale_basis = "Unknown"
    meas_file = output_dir / "measurement" / "estimator_math_summary.json"
    if meas_file.exists():
        with open(meas_file) as f:
            data = json.load(f)
            scale_value = data.get("scale", "Unknown")
            scale_basis = data.get("scale_basis", "Unknown")

    draw.text((20, 20), "Scale Information", fill=(0, 0, 0), font=font_large)
    draw.text((20, 60), f"Scale: 1:{scale_value}", fill=(0, 0, 150), font=font)
    draw.text((20, 90), f"Basis: {scale_basis}", fill=(100, 100, 100), font=font)

    # Draw scale bar
    if scale_value != "Unknown":
        bar_length = 100  # pixels representing 1m at the scale
        draw.rectangle([20, 140, 20 + bar_length, 150], fill=(0, 0, 0))
        draw.text((20, 155), "1 meter (at scale)", fill=(0, 0, 0), font=font)

    img.save(overlay_path)
    return "scale_info.png"


def _generate_proof_pack_md(
    proof_dir: Path,
    boq_items: List[Dict],
    rooms: List[Dict],
    openings: List[Dict],
    overlays: List[str],
) -> Path:
    """Generate proof_pack.md with provenance table."""
    proof_path = proof_dir / "proof_pack.md"

    lines = [
        "# Measurement Proof Pack",
        "",
        "This document provides traceability from BOQ quantities to drawing evidence.",
        "",
        "## Overlays Generated",
        "",
    ]

    if overlays:
        for ov in overlays:
            lines.append(f"- `overlays/{ov}`")
    else:
        lines.append("- No overlays generated (PIL not available or no geometry)")

    lines.extend([
        "",
        "## Extraction Summary",
        "",
        f"- **Rooms detected:** {len(rooms)}",
        f"- **Openings detected:** {len(openings)}",
        f"- **BOQ items:** {len(boq_items)}",
        "",
        "## BOQ Provenance Table",
        "",
        "| Item | Description | Room | Method | Confidence | Page | Overlay | Measured |",
        "|------|-------------|------|--------|------------|------|---------|----------|",
    ])

    for item in boq_items:
        prov = item.get("provenance", {})
        pages = prov.get("source_pages", [])
        page_str = ",".join(str(p) for p in pages) if pages else "-"

        # Determine overlay reference
        overlay_ref = "-"
        if pages:
            overlay_ref = f"rooms_page_{pages[0]}.png"

        method = prov.get("method", "unknown")
        confidence = prov.get("confidence", 0)
        is_measured = "✅" if prov.get("is_measured", False) else "❌"

        desc = item.get("description", "")[:30]
        room = item.get("room", "-")[:15]

        lines.append(
            f"| {item.get('item_id', '-')[:10]} | {desc} | {room} | {method} | {confidence:.0%} | {page_str} | {overlay_ref} | {is_measured} |"
        )

    lines.extend([
        "",
        "## Legend",
        "",
        "### Methods",
        "- **polygon**: Area calculated from closed polygon detection",
        "- **centerline**: Length from wall centerline extraction",
        "- **symbol_count**: Count from symbol/object detection",
        "- **schedule_table**: Extracted from schedule table in drawing",
        "- **dimension_text**: From dimension annotation",
        "- **text_only**: Room label only, no geometry measured",
        "- **allowance**: Provisional/allowance item",
        "- **inferred**: Inferred from template (no evidence)",
        "",
        "### Measured Status",
        "- ✅ = Geometry-backed with sufficient confidence",
        "- ❌ = Not measured (TBD/allowance)",
        "",
    ])

    with open(proof_path, "w") as f:
        f.write("\n".join(lines))

    return proof_path
