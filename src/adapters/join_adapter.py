"""
Join Adapter

Maps runner's ProjectJoiner interface to multi-page joining logic.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class ProjectJoiner:
    """
    Adapter for joining multi-page project data.

    Runner expects:
        joiner = ProjectJoiner()
        result = joiner.join_project(output_dir)

    This combines data from multiple pages into unified project data.
    """

    def __init__(self):
        pass

    def join_project(self, output_dir: Path) -> Dict[str, Any]:
        """
        Join multi-page project data.

        Args:
            output_dir: Output directory with per-page results

        Returns:
            Join result with combined data
        """
        output_dir = Path(output_dir)

        result = {
            "joined": True,
            "pages_joined": 0,
            "combined_rooms": [],
            "combined_openings": [],
        }

        # Look for room data files in multiple locations
        combined_dir = output_dir / "combined"
        boq_dir = output_dir / "boq"
        scope_dir = output_dir / "scope"

        all_rooms = []
        all_openings = []

        # Priority 1: Check combined directory (from multipage_extractor)
        if combined_dir.exists():
            rooms_file = combined_dir / "all_rooms.json"
            if rooms_file.exists():
                with open(rooms_file) as f:
                    data = json.load(f)
                    rooms = data.get("rooms", [])
                    all_rooms.extend(rooms)
                    # Count unique pages
                    pages = set(r.get("source_page", 0) for r in rooms)
                    result["pages_joined"] = len(pages)

            openings_file = combined_dir / "all_openings.json"
            if openings_file.exists():
                with open(openings_file) as f:
                    data = json.load(f)
                    openings = data.get("openings", [])
                    all_openings.extend(openings)

        # Priority 2: Check boq directory (legacy)
        if not all_rooms and boq_dir.exists():
            rooms_file = boq_dir / "rooms.json"
            if rooms_file.exists():
                with open(rooms_file) as f:
                    data = json.load(f)
                    rooms = data.get("rooms", [])
                    all_rooms.extend(rooms)
                    result["pages_joined"] += 1

            openings_file = boq_dir / "openings.json"
            if openings_file.exists():
                with open(openings_file) as f:
                    data = json.load(f)
                    openings = data.get("openings", [])
                    all_openings.extend(openings)

        # Priority 3: Look for per-page results in scope dir
        if not all_rooms and scope_dir.exists():
            for page_file in scope_dir.glob("page_*_rooms.json"):
                with open(page_file) as f:
                    rooms = json.load(f)
                    all_rooms.extend(rooms)
                    result["pages_joined"] += 1

        # Deduplicate rooms by ID
        seen_room_ids = set()
        unique_rooms = []
        for room in all_rooms:
            room_id = room.get("id")
            if room_id not in seen_room_ids:
                seen_room_ids.add(room_id)
                unique_rooms.append(room)

        # Deduplicate openings by ID
        seen_opening_ids = set()
        unique_openings = []
        for opening in all_openings:
            opening_id = opening.get("id")
            if opening_id not in seen_opening_ids:
                seen_opening_ids.add(opening_id)
                unique_openings.append(opening)

        result["combined_rooms"] = unique_rooms
        result["combined_openings"] = unique_openings
        result["total_rooms"] = len(unique_rooms)
        result["total_openings"] = len(unique_openings)

        # Write combined data
        if unique_rooms or unique_openings:
            combined_dir = output_dir / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)

            with open(combined_dir / "all_rooms.json", "w") as f:
                json.dump({"rooms": unique_rooms}, f, indent=2)

            with open(combined_dir / "all_openings.json", "w") as f:
                json.dump({"openings": unique_openings}, f, indent=2)

        return result


def join_project_pages(output_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to join project pages.

    Args:
        output_dir: Output directory

    Returns:
        Join result
    """
    joiner = ProjectJoiner()
    return joiner.join_project(output_dir)
