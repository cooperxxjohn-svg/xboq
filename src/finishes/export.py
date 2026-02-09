"""
Finish Export Module
Exports finish takeoff data to BOQ-ready formats.

Outputs:
- finishes_boq.csv: BOQ format (room, item, qty, unit)
- openings_schedule.csv: Door/window schedule
- wall_lengths.csv: Wall centerline lengths by room
- report.md: Assumptions and results summary
"""

import csv
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

from .calculator import RoomFinishes

logger = logging.getLogger(__name__)


class FinishExporter:
    """
    Export finish quantities to BOQ-ready formats.

    Generates:
    - CSV files for BOQ integration
    - Markdown report with assumptions
    - JSON for programmatic access
    """

    # BOQ item descriptions (Indian construction terms)
    BOQ_ITEMS = {
        "floor_tiling": "Providing and laying vitrified tiles ({size}) in flooring",
        "skirting": "Providing and laying skirting tiles (100mm height)",
        "wall_paint": "Applying 2 coats plastic emulsion paint on plastered surface",
        "ceiling_paint": "Applying 2 coats plastic emulsion paint on ceiling",
        "wall_tiles_wet": "Providing and laying ceramic wall tiles in wet areas (up to {height}m)",
        "dado_tiles": "Providing and laying dado tiles in kitchen/utility (up to 900mm)",
        "external_floor": "Providing and laying anti-skid tiles on balcony/terrace",
    }

    def __init__(self, output_dir: Path):
        """
        Initialize exporter.

        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        room_finishes: List[RoomFinishes],
        openings: Optional[List[Dict]] = None,
        wall_lengths: Optional[Dict[str, float]] = None,
        assumptions: Optional[Dict] = None,
        plan_id: str = "plan",
    ) -> Dict[str, Path]:
        """
        Export all finish data.

        Args:
            room_finishes: List of RoomFinishes objects
            openings: List of opening dicts
            wall_lengths: Dict of room_id to wall length
            assumptions: Assumptions dict
            plan_id: Plan identifier

        Returns:
            Dict mapping output type to file path
        """
        outputs = {}

        # Export BOQ CSV
        boq_path = self.export_finishes_boq(room_finishes)
        outputs["finishes_boq"] = boq_path

        # Export wall lengths
        if wall_lengths:
            wall_path = self.export_wall_lengths(room_finishes, wall_lengths)
            outputs["wall_lengths"] = wall_path

        # Export openings schedule
        if openings:
            openings_path = self.export_openings_schedule(openings)
            outputs["openings_schedule"] = openings_path

        # Export report
        report_path = self.export_report(
            room_finishes, openings, wall_lengths, assumptions, plan_id
        )
        outputs["report"] = report_path

        # Export JSON
        json_path = self.export_json(
            room_finishes, openings, wall_lengths, assumptions, plan_id
        )
        outputs["json"] = json_path

        logger.info(f"Exported {len(outputs)} files to {self.output_dir}")

        return outputs

    def export_finishes_boq(
        self,
        room_finishes: List[RoomFinishes],
    ) -> Path:
        """
        Export finishes BOQ in standard format.

        Format: Room, Item, Qty, Unit
        """
        boq_path = self.output_dir / "finishes_boq.csv"

        with open(boq_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["Room", "Item", "Description", "Qty", "Unit"])

            for room in room_finishes:
                # Skip shafts
                if room.room_category == "shaft":
                    continue

                room_name = f"{room.room_label} ({room.room_id})"

                # Floor tiling
                if room.floor_area_sqm > 0:
                    if room.is_external:
                        desc = self.BOQ_ITEMS["external_floor"]
                        item = "External Floor Tiles"
                    else:
                        desc = self.BOQ_ITEMS["floor_tiling"].format(size="600x600")
                        item = "Floor Tiles"

                    writer.writerow([
                        room_name, item, desc,
                        f"{room.floor_area_sqm:.2f}", "sqm"
                    ])

                # Skirting (internal only)
                if room.skirting_length_m > 0 and not room.is_external:
                    writer.writerow([
                        room_name, "Skirting", self.BOQ_ITEMS["skirting"],
                        f"{room.skirting_length_m:.2f}", "rm"
                    ])

                # Wall paint/tiles
                if room.wall_area_sqm > 0:
                    if room.is_wet_area:
                        # Wall tiles for wet area
                        tile_height = room.tile_height_m or 2.1
                        desc = self.BOQ_ITEMS["wall_tiles_wet"].format(height=tile_height)
                        # Tile area up to tile height
                        tile_area = room.perimeter_m * tile_height
                        writer.writerow([
                            room_name, "Wall Tiles", desc,
                            f"{tile_area:.2f}", "sqm"
                        ])

                        # Paint above tile height
                        paint_area = room.wall_area_sqm - tile_area
                        if paint_area > 0:
                            writer.writerow([
                                room_name, "Wall Paint (above tiles)",
                                self.BOQ_ITEMS["wall_paint"],
                                f"{paint_area:.2f}", "sqm"
                            ])
                    elif room.needs_dado:
                        # Dado for kitchen/utility
                        dado_area = room.perimeter_m * 0.9  # 900mm dado
                        writer.writerow([
                            room_name, "Dado Tiles", self.BOQ_ITEMS["dado_tiles"],
                            f"{dado_area:.2f}", "sqm"
                        ])

                        paint_area = room.wall_area_sqm - dado_area
                        if paint_area > 0:
                            writer.writerow([
                                room_name, "Wall Paint (above dado)",
                                self.BOQ_ITEMS["wall_paint"],
                                f"{paint_area:.2f}", "sqm"
                            ])
                    else:
                        # Full wall paint
                        writer.writerow([
                            room_name, "Wall Paint", self.BOQ_ITEMS["wall_paint"],
                            f"{room.wall_area_sqm:.2f}", "sqm"
                        ])

                # Ceiling paint
                if room.ceiling_area_sqm > 0:
                    writer.writerow([
                        room_name, "Ceiling Paint", self.BOQ_ITEMS["ceiling_paint"],
                        f"{room.ceiling_area_sqm:.2f}", "sqm"
                    ])

        return boq_path

    def export_wall_lengths(
        self,
        room_finishes: List[RoomFinishes],
        wall_lengths: Dict[str, float],
    ) -> Path:
        """Export wall centerline lengths by room."""
        csv_path = self.output_dir / "wall_lengths.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["Room ID", "Room Label", "Category", "Perimeter (m)", "Wall Length (m)"])

            for room in room_finishes:
                length = wall_lengths.get(room.room_id, room.perimeter_m)
                writer.writerow([
                    room.room_id,
                    room.room_label,
                    room.room_category,
                    f"{room.perimeter_m:.2f}",
                    f"{length:.2f}",
                ])

        return csv_path

    def export_openings_schedule(
        self,
        openings: List[Dict],
    ) -> Path:
        """Export combined openings schedule."""
        csv_path = self.output_dir / "openings_schedule.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "ID", "Type", "Tag", "Width (mm)", "Height (mm)",
                "Room Left", "Room Right", "Area (sqm)"
            ])

            for opening in openings:
                width_mm = int(opening.get("width_m", 0) * 1000) if opening.get("width_m") else ""
                height_mm = int(opening.get("height_m", 0) * 1000) if opening.get("height_m") else ""

                width_m = opening.get("width_m", 0) or 0
                height_m = opening.get("height_m", 0) or 0
                area = width_m * height_m

                writer.writerow([
                    opening.get("id", ""),
                    opening.get("type", ""),
                    opening.get("tag", ""),
                    width_mm,
                    height_mm,
                    opening.get("room_left_id", ""),
                    opening.get("room_right_id", ""),
                    f"{area:.2f}" if area > 0 else "",
                ])

        return csv_path

    def export_report(
        self,
        room_finishes: List[RoomFinishes],
        openings: Optional[List[Dict]],
        wall_lengths: Optional[Dict[str, float]],
        assumptions: Optional[Dict],
        plan_id: str,
    ) -> Path:
        """Export markdown report with assumptions and results."""
        report_path = self.output_dir / "report.md"

        # Calculate totals
        valid_rooms = [r for r in room_finishes if r.room_category != "shaft"]
        internal = [r for r in valid_rooms if not r.is_external]
        external = [r for r in valid_rooms if r.is_external]
        wet_areas = [r for r in valid_rooms if r.is_wet_area]

        total_floor = sum(r.floor_area_sqm for r in valid_rooms)
        total_skirting = sum(r.skirting_length_m for r in internal)
        total_wall = sum(r.wall_area_sqm for r in valid_rooms)
        total_ceiling = sum(r.ceiling_area_sqm for r in valid_rooms)

        with open(report_path, "w") as f:
            f.write(f"# Finish Takeoff Report\n\n")
            f.write(f"**Plan ID:** {plan_id}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Rooms | {len(valid_rooms)} |\n")
            f.write(f"| Internal Rooms | {len(internal)} |\n")
            f.write(f"| External Rooms | {len(external)} |\n")
            f.write(f"| Wet Areas | {len(wet_areas)} |\n")
            f.write(f"| Total Floor Area | {total_floor:.2f} sqm |\n")
            f.write(f"| Total Skirting | {total_skirting:.2f} rm |\n")
            f.write(f"| Total Wall Area | {total_wall:.2f} sqm |\n")
            f.write(f"| Total Ceiling Area | {total_ceiling:.2f} sqm |\n\n")

            # Openings summary
            if openings:
                doors = [o for o in openings if "door" in o.get("type", "").lower()]
                windows = [o for o in openings if "window" in o.get("type", "").lower()]
                vents = [o for o in openings if "vent" in o.get("type", "").lower()]

                f.write("## Openings Summary\n\n")
                f.write(f"| Type | Count |\n")
                f.write(f"|------|-------|\n")
                f.write(f"| Doors | {len(doors)} |\n")
                f.write(f"| Windows | {len(windows)} |\n")
                f.write(f"| Ventilators | {len(vents)} |\n\n")

            # Room breakdown
            f.write("## Room-wise Breakdown\n\n")
            f.write("| Room | Category | Floor (sqm) | Skirting (m) | Wall (sqm) | Ceiling (sqm) |\n")
            f.write("|------|----------|-------------|--------------|------------|---------------|\n")

            for room in room_finishes:
                f.write(f"| {room.room_label} | {room.room_category} | ")
                f.write(f"{room.floor_area_sqm:.2f} | {room.skirting_length_m:.2f} | ")
                f.write(f"{room.wall_area_sqm:.2f} | {room.ceiling_area_sqm:.2f} |\n")

            f.write("\n")

            # Assumptions
            f.write("## Assumptions Used\n\n")

            if assumptions:
                defaults = assumptions.get("defaults", {})
                f.write("### Default Values\n\n")
                f.write(f"- **Ceiling Height:** {defaults.get('ceiling_height_mm', 3000)} mm\n")
                f.write(f"- **Skirting Height:** {defaults.get('skirting_height_mm', 100)} mm\n")
                f.write(f"- **Bathroom Tile Height:** {defaults.get('bathroom_tile_height_mm', 2100)} mm\n")
                f.write(f"- **Default Door Height:** {defaults.get('door_height_mm', 2100)} mm\n")
                f.write(f"- **Default Window Height:** {defaults.get('window_height_mm', 1200)} mm\n\n")

                room_heights = assumptions.get("room_ceiling_heights", {})
                if room_heights:
                    f.write("### Room-specific Ceiling Heights\n\n")
                    for room_type, height in room_heights.items():
                        f.write(f"- {room_type}: {height} mm\n")
                    f.write("\n")

            # Room-specific assumptions
            all_assumptions = []
            for room in room_finishes:
                for assumption in room.assumptions_used:
                    if assumption not in all_assumptions:
                        all_assumptions.append(assumption)

            if all_assumptions:
                f.write("### Calculation Assumptions\n\n")
                for assumption in all_assumptions:
                    f.write(f"- {assumption}\n")
                f.write("\n")

            # Deduction rules
            f.write("## Deduction Rules Applied\n\n")
            f.write("- Floor area: Shafts and ducts excluded\n")
            f.write("- Skirting: Door widths deducted from perimeter\n")
            f.write("- Wall area: All openings (doors, windows) deducted\n")
            f.write("- External rooms: Treated as separate finish category\n\n")

            f.write("---\n")
            f.write("*Report generated by XBOQ Finish Takeoff Module*\n")

        return report_path

    def export_json(
        self,
        room_finishes: List[RoomFinishes],
        openings: Optional[List[Dict]],
        wall_lengths: Optional[Dict[str, float]],
        assumptions: Optional[Dict],
        plan_id: str,
    ) -> Path:
        """Export complete data as JSON."""
        json_path = self.output_dir / "finishes.json"

        # Convert room finishes to dicts
        rooms_data = []
        for room in room_finishes:
            rooms_data.append({
                "room_id": room.room_id,
                "room_label": room.room_label,
                "room_category": room.room_category,
                "floor_area_sqm": room.floor_area_sqm,
                "carpet_area_sqm": room.carpet_area_sqm,
                "perimeter_m": room.perimeter_m,
                "skirting_length_m": room.skirting_length_m,
                "wall_area_sqm": room.wall_area_sqm,
                "ceiling_area_sqm": room.ceiling_area_sqm,
                "ceiling_height_m": room.ceiling_height_m,
                "tile_height_m": room.tile_height_m,
                "deductions": room.deductions,
                "openings_count": room.openings_count,
                "is_external": room.is_external,
                "is_wet_area": room.is_wet_area,
                "needs_dado": room.needs_dado,
                "assumptions_used": room.assumptions_used,
            })

        # Calculate totals
        valid_rooms = [r for r in room_finishes if r.room_category != "shaft"]
        totals = {
            "total_floor_area_sqm": sum(r.floor_area_sqm for r in valid_rooms),
            "total_skirting_length_m": sum(r.skirting_length_m for r in valid_rooms if not r.is_external),
            "total_wall_area_sqm": sum(r.wall_area_sqm for r in valid_rooms),
            "total_ceiling_area_sqm": sum(r.ceiling_area_sqm for r in valid_rooms),
        }

        data = {
            "plan_id": plan_id,
            "generated_at": datetime.now().isoformat(),
            "totals": totals,
            "rooms": rooms_data,
            "openings": openings or [],
            "wall_lengths": wall_lengths or {},
            "assumptions": assumptions or {},
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return json_path
