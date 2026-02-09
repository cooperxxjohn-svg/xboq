"""
RFI Adapter

Maps runner's RFI interfaces to real rfi.generator module.

The real RFIGenerator requires complex objects (SignalCollection, ConflictReport,
ReferenceReport). This adapter provides a simplified interface that generates
RFIs from scope gaps, rooms, and openings data directly.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum


class IssueType(Enum):
    """RFI issue types (mirrors real module)."""
    MISSING_INPUT = "missing_input"
    CONFLICT = "conflict"
    UNCLEAR_SPEC = "unclear_spec"
    MISSING_DIMENSION = "missing_dimension"
    MISSING_SCHEDULE = "missing_schedule"


class Priority(Enum):
    """RFI priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RFIGenerator:
    """
    Adapter for RFI generation.

    Provides simplified interface that works without requiring
    SignalCollection, ConflictReport, ReferenceReport objects.
    """

    def __init__(self):
        self._rfi_counter = 0

    def _generate_rfi_id(self) -> str:
        """Generate unique RFI ID."""
        self._rfi_counter += 1
        return f"RFI-{self._rfi_counter:04d}"

    def generate(
        self,
        scope_gaps: List[Dict] = None,
        rooms: List[Dict] = None,
        openings: List[Dict] = None,
        owner_inputs: Dict = None,
    ) -> Dict[str, Any]:
        """
        Generate RFIs from scope gaps and data.

        Args:
            scope_gaps: List of identified scope gaps
            rooms: Room data
            openings: Opening data
            owner_inputs: Owner input specifications

        Returns:
            RFI generation result
        """
        rfis = []
        scope_gaps = scope_gaps or []
        rooms = rooms or []
        openings = openings or []
        owner_inputs = owner_inputs or {}

        # Generate RFIs from scope gaps
        for gap in scope_gaps:
            priority = Priority.HIGH if gap.get("severity") == "high" else Priority.MEDIUM
            rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_INPUT.value,
                "priority": priority.value,
                "title": gap.get("description", "Scope gap identified"),
                "description": gap.get("details", ""),
                "package": gap.get("package", "general"),
                "suggested_response": gap.get("suggestion", "Please clarify scope"),
            })

        # Check for missing schedules
        door_tags = [o.get("tag") for o in openings if o.get("type") == "door" and o.get("tag")]
        window_tags = [o.get("tag") for o in openings if o.get("type") == "window" and o.get("tag")]

        # Missing door schedule
        if door_tags and not owner_inputs.get("door_schedule_provided"):
            rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_SCHEDULE.value,
                "priority": Priority.HIGH.value,
                "title": f"Door schedule not provided ({len(door_tags)} tags detected)",
                "description": f"Door tags detected: {', '.join(door_tags[:5])}{'...' if len(door_tags) > 5 else ''}",
                "package": "doors_windows",
                "suggested_response": "Provide door schedule with Mark, Size (W x H), Type, Material, Hardware",
            })

        # Missing window schedule
        if window_tags and not owner_inputs.get("window_schedule_provided"):
            rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_SCHEDULE.value,
                "priority": Priority.HIGH.value,
                "title": f"Window schedule not provided ({len(window_tags)} tags detected)",
                "description": f"Window tags detected: {', '.join(window_tags[:5])}{'...' if len(window_tags) > 5 else ''}",
                "package": "doors_windows",
                "suggested_response": "Provide window schedule with Mark, Size (W x H), Type, Frame Material, Glass",
            })

        # Check rooms for missing finish specification
        room_types = set(r.get("room_type", "unknown") for r in rooms)
        wet_rooms = [r for r in rooms if r.get("room_type") in ["toilet", "bathroom", "kitchen", "utility"]]
        if wet_rooms and not owner_inputs.get("finish_schedule_provided"):
            rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.UNCLEAR_SPEC.value,
                "priority": Priority.MEDIUM.value,
                "title": "Finish schedule not provided for wet areas",
                "description": f"Wet areas detected: {len(wet_rooms)} rooms. Tile/waterproofing specs needed.",
                "package": "tiles_finishes",
                "suggested_response": "Provide finish schedule with room-wise floor, wall, ceiling specifications",
            })

        # Summarize by priority
        by_priority = {}
        for rfi in rfis:
            p = rfi["priority"]
            by_priority[p] = by_priority.get(p, 0) + 1

        return {
            "rfis": rfis,
            "total_count": len(rfis),
            "by_priority": by_priority,
        }

    def generate_from_output(self, output_dir: Path) -> Dict[str, Any]:
        """
        Generate RFIs by analyzing output directory data.

        Args:
            output_dir: Output directory with scope data

        Returns:
            RFI generation result
        """
        output_dir = Path(output_dir)

        # Load data from output
        rooms = []
        openings = []
        scope_gaps = []

        # Try to load rooms
        rooms_paths = [
            output_dir / "boq" / "rooms.json",
            output_dir / "scope" / "rooms.json",
            output_dir / "combined" / "all_rooms.json",
        ]
        for rp in rooms_paths:
            if rp.exists():
                with open(rp) as f:
                    data = json.load(f)
                    rooms = data.get("rooms", [])
                    break

        # Try to load openings
        openings_paths = [
            output_dir / "boq" / "openings.json",
            output_dir / "scope" / "openings.json",
            output_dir / "combined" / "all_openings.json",
        ]
        for op in openings_paths:
            if op.exists():
                with open(op) as f:
                    data = json.load(f)
                    openings = data.get("openings", [])
                    break

        # Load scope analysis for gaps
        scope_analysis = output_dir / "scope" / "scope_analysis.json"
        if scope_analysis.exists():
            with open(scope_analysis) as f:
                data = json.load(f)
                scope_gaps = data.get("gaps", [])

        return self.generate(
            scope_gaps=scope_gaps,
            rooms=rooms,
            openings=openings,
        )


def run_rfi_generation(
    output_dir: Path,
    owner_inputs: Dict = None,
) -> Dict[str, Any]:
    """
    Run RFI generation for a project.

    Runner expects this function.

    Args:
        output_dir: Output directory
        owner_inputs: Owner inputs

    Returns:
        RFI result with list of RFIs
    """
    output_dir = Path(output_dir)

    generator = RFIGenerator()
    result = generator.generate_from_output(output_dir)

    # Write RFI outputs
    rfi_dir = output_dir / "rfi"
    rfi_dir.mkdir(parents=True, exist_ok=True)

    # Write RFI JSON
    with open(rfi_dir / "rfis.json", "w") as f:
        json.dump(result, f, indent=2)

    # Write RFI log markdown
    with open(rfi_dir / "rfi_log.md", "w") as f:
        f.write("# RFI Log\n\n")
        f.write(f"**Total RFIs:** {result['total_count']}\n\n")

        if result["rfis"]:
            f.write("## RFI List\n\n")
            f.write("| ID | Priority | Type | Title |\n")
            f.write("|----|----------|------|-------|\n")

            for rfi in result["rfis"]:
                f.write(f"| {rfi['id']} | {rfi['priority']} | {rfi['issue_type']} | {rfi['title']} |\n")

            f.write("\n## RFI Details\n\n")

            for rfi in result["rfis"]:
                f.write(f"### {rfi['id']}: {rfi['title']}\n\n")
                f.write(f"**Priority:** {rfi['priority']}\n")
                f.write(f"**Type:** {rfi['issue_type']}\n\n")
                f.write(f"{rfi['description']}\n\n")
                if rfi.get("suggested_response"):
                    f.write(f"**Suggested Response:** {rfi['suggested_response']}\n\n")
                f.write("---\n\n")
        else:
            f.write("No RFIs generated. All required information appears to be available.\n")

    return {
        "rfi_count": result["total_count"],
        "output": str(rfi_dir / "rfi_log.md"),
        "rfis": result["rfis"],
    }
