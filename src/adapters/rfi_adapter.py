"""
RFI Adapter - Enhanced Version

Maps runner's RFI interfaces to generate estimator-grade RFIs with:
- Drawing-level gap analysis (missing MEP, sections, elevations)
- Specific schedule detection (door tags without door schedule)
- Trade-wise gap summary
- Sheet-specific RFI references

The goal: Show estimators WHAT is missing, not just THAT something is missing.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from collections import Counter, defaultdict
from datetime import datetime


class IssueType(Enum):
    """RFI issue types."""
    MISSING_INPUT = "missing_input"
    MISSING_SCHEDULE = "missing_schedule"
    MISSING_DRAWING = "missing_drawing"
    CONFLICT = "conflict"
    UNCLEAR_SPEC = "unclear_spec"
    MISSING_DIMENSION = "missing_dimension"
    SCALE_ISSUE = "scale_issue"
    SCOPE_GAP = "scope_gap"


class Priority(Enum):
    """RFI priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Trade(Enum):
    """Trade categories for grouping gaps."""
    CIVIL = "civil"
    STRUCTURAL = "structural"
    ARCHITECTURAL = "architectural"
    MEP = "mep"
    FINISHES = "finishes"
    GENERAL = "general"


# Trade mapping for package types
PACKAGE_TO_TRADE = {
    "rcc_structural": Trade.STRUCTURAL,
    "masonry": Trade.CIVIL,
    "waterproofing": Trade.CIVIL,
    "plumbing": Trade.MEP,
    "electrical": Trade.MEP,
    "fire_hvac": Trade.MEP,
    "doors_windows": Trade.ARCHITECTURAL,
    "tiles_finishes": Trade.FINISHES,
    "painting": Trade.FINISHES,
    "external_works": Trade.CIVIL,
    "levels_dimensions": Trade.GENERAL,
}


class EnhancedRFIGenerator:
    """
    Enhanced RFI generation with drawing-level analysis.

    Generates specific, actionable RFIs that estimators can use directly.
    """

    def __init__(self):
        self._rfi_counter = 0
        self.rfis: List[Dict] = []
        self.trade_gaps: Dict[Trade, List[str]] = defaultdict(list)

    def _generate_rfi_id(self) -> str:
        """Generate unique RFI ID."""
        self._rfi_counter += 1
        return f"RFI-{self._rfi_counter:04d}"

    def generate_from_output(self, output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive RFIs by analyzing all output data.

        Args:
            output_dir: Output directory with extraction results

        Returns:
            Complete RFI report with trade-wise summary
        """
        output_dir = Path(output_dir)
        self.rfis = []
        self.trade_gaps = defaultdict(list)
        self._rfi_counter = 0

        # Load all available data
        rooms = self._load_rooms(output_dir)
        openings = self._load_openings(output_dir)
        routing_data = self._load_routing_data(output_dir)
        bid_gate = self._load_bid_gate(output_dir)
        measurement_gate = self._load_measurement_gate(output_dir)

        # 1. Drawing-level gap analysis
        self._analyze_drawing_gaps(routing_data)

        # 2. Schedule detection (tags without schedules)
        self._analyze_schedule_gaps(openings, rooms)

        # 3. Scale and measurement issues
        self._analyze_scale_issues(routing_data, measurement_gate)

        # 4. Scope gaps from bid gate
        self._analyze_scope_gaps(bid_gate)

        # 5. Room-based gaps (wet areas, finishes)
        self._analyze_room_gaps(rooms)

        # Sort RFIs by priority
        self._sort_rfis()

        # Build trade-wise summary
        trade_summary = self._build_trade_summary()

        return {
            "rfis": self.rfis,
            "total_count": len(self.rfis),
            "by_priority": self._count_by_priority(),
            "by_trade": trade_summary,
            "by_issue_type": self._count_by_issue_type(),
            "critical_blockers": self._get_critical_blockers(),
            "generated_at": datetime.now().isoformat(),
        }

    def _load_rooms(self, output_dir: Path) -> List[Dict]:
        """Load room data from output."""
        paths = [
            output_dir / "combined" / "all_rooms.json",
            output_dir / "boq" / "rooms.json",
        ]
        for p in paths:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                    return data.get("rooms", data if isinstance(data, list) else [])
        return []

    def _load_openings(self, output_dir: Path) -> List[Dict]:
        """Load openings data from output."""
        paths = [
            output_dir / "combined" / "all_openings.json",
            output_dir / "boq" / "openings.json",
        ]
        for p in paths:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                    return data.get("openings", data if isinstance(data, list) else [])
        return []

    def _load_routing_data(self, output_dir: Path) -> List[Dict]:
        """Load page routing/classification data."""
        routing_path = output_dir / "routing_debug.csv"
        if routing_path.exists():
            with open(routing_path, newline='') as f:
                return list(csv.DictReader(f))
        return []

    def _load_bid_gate(self, output_dir: Path) -> Dict:
        """Load bid gate results."""
        path = output_dir / "bid_gate_result.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_measurement_gate(self, output_dir: Path) -> Dict:
        """Load measurement gate results."""
        path = output_dir / "measurement_gate_result.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _analyze_drawing_gaps(self, routing_data: List[Dict]) -> None:
        """Analyze what drawing types are missing from the set."""
        if not routing_data:
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_DRAWING.value,
                "priority": Priority.HIGH.value,
                "trade": Trade.GENERAL.value,
                "title": "No drawing classification data available",
                "description": "Could not analyze drawing set - routing data missing.",
                "package": "general",
                "evidence_pages": [],
                "suggested_response": "Ensure drawings are properly uploaded and processed.",
            })
            self.trade_gaps[Trade.GENERAL].append("Drawing classification failed")
            return

        # Count page types
        page_types = Counter(r.get("page_type", "unknown") for r in routing_data)
        total_pages = len(routing_data)

        # Expected drawing types for a complete set
        expected_types = {
            "floor_plan": {"name": "Floor Plans", "min_expected": 1, "trade": Trade.ARCHITECTURAL},
            "elevation": {"name": "Elevations", "min_expected": 2, "trade": Trade.ARCHITECTURAL},
            "section": {"name": "Sections", "min_expected": 1, "trade": Trade.ARCHITECTURAL},
            "structural": {"name": "Structural Drawings", "min_expected": 1, "trade": Trade.STRUCTURAL},
            "mep": {"name": "MEP Drawings", "min_expected": 1, "trade": Trade.MEP},
            "electrical": {"name": "Electrical Drawings", "min_expected": 1, "trade": Trade.MEP},
            "plumbing": {"name": "Plumbing Drawings", "min_expected": 1, "trade": Trade.MEP},
            "detail": {"name": "Detail Drawings", "min_expected": 1, "trade": Trade.ARCHITECTURAL},
            "schedule": {"name": "Schedule Sheets", "min_expected": 1, "trade": Trade.GENERAL},
        }

        # Check for missing types
        for dtype, info in expected_types.items():
            found_count = page_types.get(dtype, 0)
            if found_count < info["min_expected"]:
                priority = Priority.HIGH if dtype in ["floor_plan", "structural", "section"] else Priority.MEDIUM

                self.rfis.append({
                    "id": self._generate_rfi_id(),
                    "issue_type": IssueType.MISSING_DRAWING.value,
                    "priority": priority.value,
                    "trade": info["trade"].value,
                    "title": f"No {info['name']} found in drawing set",
                    "description": f"Drawing set has {total_pages} pages but no {info['name'].lower()} were detected. "
                                   f"This may affect scope coverage for {info['trade'].value} work.",
                    "package": info["trade"].value,
                    "evidence_pages": [],
                    "suggested_response": f"Provide {info['name'].lower()} or confirm if not applicable to project scope.",
                    "impact": f"Cannot verify {info['trade'].value} scope without {info['name'].lower()}",
                })
                self.trade_gaps[info["trade"]].append(f"Missing {info['name']}")

        # Check MEP specifically - common gap
        mep_types = ["mep", "electrical", "plumbing", "fire_hvac"]
        mep_count = sum(page_types.get(t, 0) for t in mep_types)
        if mep_count == 0:
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_DRAWING.value,
                "priority": Priority.HIGH.value,
                "trade": Trade.MEP.value,
                "title": "No MEP drawings found in drawing set",
                "description": f"Drawing set has {total_pages} pages but no electrical, plumbing, or HVAC drawings detected. "
                               "MEP work cannot be scoped from architectural drawings alone.",
                "package": "mep",
                "evidence_pages": [],
                "suggested_response": "Provide MEP drawings or confirm if MEP work is excluded from tender scope.",
                "impact": "Cannot price electrical, plumbing, or HVAC work",
            })
            self.trade_gaps[Trade.MEP].append("No MEP drawings in set")

        # Report unknown/unclassified pages
        unknown_count = page_types.get("unknown", 0) + page_types.get("candidate_plan", 0)
        if unknown_count > total_pages * 0.3:  # More than 30% unclassified
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.UNCLEAR_SPEC.value,
                "priority": Priority.MEDIUM.value,
                "trade": Trade.GENERAL.value,
                "title": f"{unknown_count} of {total_pages} pages could not be classified",
                "description": f"Many pages in the drawing set could not be automatically classified. "
                               "This may indicate poor scan quality, non-standard formatting, or image-only PDFs.",
                "package": "general",
                "evidence_pages": [r.get("page", "") for r in routing_data if r.get("page_type") in ["unknown", "candidate_plan"]][:10],
                "suggested_response": "Provide vector PDFs where possible, or clarify drawing types for unclassified pages.",
            })

    def _analyze_schedule_gaps(self, openings: List[Dict], rooms: List[Dict]) -> None:
        """Analyze schedule gaps based on detected tags."""
        # Extract door tags - check both 'type' and 'opening_type' fields
        door_tags = []
        window_tags = []

        for o in openings:
            tag = o.get("tag") or o.get("mark", "")
            opening_type = o.get("opening_type", o.get("type", "")).lower()

            if "door" in opening_type or "door" in tag.lower():
                if tag:
                    door_tags.append(tag)
            elif "window" in opening_type or "window" in tag.lower():
                if tag:
                    window_tags.append(tag)

        # Deduplicate
        unique_door_tags = sorted(set(door_tags))
        unique_window_tags = sorted(set(window_tags))

        # Door schedule RFI
        if unique_door_tags:
            sample_tags = unique_door_tags[:8]
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_SCHEDULE.value,
                "priority": Priority.HIGH.value,
                "trade": Trade.ARCHITECTURAL.value,
                "title": f"Door schedule required - {len(unique_door_tags)} door types detected",
                "description": f"Detected {len(door_tags)} doors with {len(unique_door_tags)} unique tags: "
                               f"{', '.join(sample_tags)}{'...' if len(unique_door_tags) > 8 else ''}. "
                               f"Door schedule needed to price doors, frames, and hardware.",
                "package": "doors_windows",
                "evidence_pages": list(set(str(o.get("source_page", "")) for o in openings if "door" in o.get("opening_type", "").lower()))[:5],
                "suggested_response": "Provide door schedule with columns: Mark, Size (W x H), Type, Frame Material, Shutter Material, Hardware",
                "detected_tags": unique_door_tags[:20],
                "impact": f"Cannot price {len(door_tags)} doors without schedule",
            })
            self.trade_gaps[Trade.ARCHITECTURAL].append(f"Door schedule missing ({len(unique_door_tags)} types)")

        # Window schedule RFI
        if unique_window_tags:
            sample_tags = unique_window_tags[:8]
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_SCHEDULE.value,
                "priority": Priority.HIGH.value,
                "trade": Trade.ARCHITECTURAL.value,
                "title": f"Window schedule required - {len(unique_window_tags)} window types detected",
                "description": f"Detected {len(window_tags)} windows with {len(unique_window_tags)} unique tags: "
                               f"{', '.join(sample_tags)}{'...' if len(unique_window_tags) > 8 else ''}. "
                               f"Window schedule needed to price frames, glass, and fittings.",
                "package": "doors_windows",
                "evidence_pages": list(set(str(o.get("source_page", "")) for o in openings if "window" in o.get("opening_type", "").lower()))[:5],
                "suggested_response": "Provide window schedule with columns: Mark, Size (W x H), Type, Frame Material, Glass Type & Thickness",
                "detected_tags": unique_window_tags[:20],
                "impact": f"Cannot price {len(window_tags)} windows without schedule",
            })
            self.trade_gaps[Trade.ARCHITECTURAL].append(f"Window schedule missing ({len(unique_window_tags)} types)")

        # Finish schedule RFI based on rooms
        if rooms:
            room_types = Counter(r.get("room_type", r.get("label", "unknown")).lower() for r in rooms)
            total_rooms = len(rooms)

            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.MISSING_SCHEDULE.value,
                "priority": Priority.MEDIUM.value,
                "trade": Trade.FINISHES.value,
                "title": f"Finish schedule required - {total_rooms} rooms detected",
                "description": f"Detected {total_rooms} rooms of {len(room_types)} types. "
                               f"Room types include: {', '.join(list(room_types.keys())[:8])}. "
                               f"Finish schedule needed for flooring, wall, and ceiling specifications.",
                "package": "tiles_finishes",
                "evidence_pages": [],
                "suggested_response": "Provide finish schedule with columns: Room Type, Floor Finish, Wall Finish, Ceiling, Skirting, Dado",
                "room_types_detected": dict(room_types.most_common(15)),
                "impact": f"Cannot price finishes for {total_rooms} rooms",
            })
            self.trade_gaps[Trade.FINISHES].append(f"Finish schedule missing ({total_rooms} rooms)")

    def _analyze_scale_issues(self, routing_data: List[Dict], measurement_gate: Dict) -> None:
        """Analyze scale detection issues."""
        if not routing_data:
            return

        # Count pages with/without scale
        with_scale = [r for r in routing_data if r.get("scale_detected")]
        without_scale = [r for r in routing_data if r.get("is_candidate") == "YES" and not r.get("scale_detected")]

        if without_scale:
            pages_list = [r.get("page", "") for r in without_scale][:10]
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.SCALE_ISSUE.value,
                "priority": Priority.HIGH.value,
                "trade": Trade.GENERAL.value,
                "title": f"Scale not detected on {len(without_scale)} drawing pages",
                "description": f"{len(without_scale)} candidate pages have no detectable scale. "
                               f"Affected pages: {', '.join(pages_list)}{'...' if len(without_scale) > 10 else ''}. "
                               f"Areas cannot be measured reliably without scale.",
                "package": "levels_dimensions",
                "evidence_pages": pages_list,
                "suggested_response": "Add scale notation (e.g., 1:100) to all drawings, or provide scale in tender documents.",
                "impact": "Measurements on these pages are unreliable",
            })
            self.trade_gaps[Trade.GENERAL].append(f"Scale missing on {len(without_scale)} pages")

        # Report measurement gate failures
        if measurement_gate.get("status") in ["FAIL_SCALE", "FAIL_GEOMETRY"]:
            checks = measurement_gate.get("checks", [])
            failed_checks = [c for c in checks if c.get("status") == "FAIL" and c.get("severity") == "blocker"]

            for check in failed_checks:
                self.rfis.append({
                    "id": self._generate_rfi_id(),
                    "issue_type": IssueType.SCALE_ISSUE.value,
                    "priority": Priority.HIGH.value,
                    "trade": Trade.GENERAL.value,
                    "title": f"Measurement blocked: {check.get('check_name', 'Unknown check')}",
                    "description": check.get("message", "Measurement gate check failed"),
                    "package": "general",
                    "evidence_pages": [],
                    "suggested_response": "Provide vector PDFs with clear geometry and dimensions.",
                    "impact": "Cannot produce measured quantities",
                })

    def _analyze_scope_gaps(self, bid_gate: Dict) -> None:
        """Analyze scope gaps from bid gate results."""
        blockers = bid_gate.get("blockers", [])
        warnings = bid_gate.get("warnings", [])

        for blocker in blockers:
            if "measured.json is empty" in blocker:
                # Already covered by measurement gate
                continue

            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.SCOPE_GAP.value,
                "priority": Priority.HIGH.value,
                "trade": Trade.GENERAL.value,
                "title": "Critical scope gap identified",
                "description": blocker,
                "package": "general",
                "evidence_pages": [],
                "suggested_response": "Clarify scope or provide missing information.",
            })
            self.trade_gaps[Trade.GENERAL].append("Critical scope gap")

    def _analyze_room_gaps(self, rooms: List[Dict]) -> None:
        """Analyze room-based gaps (wet areas, etc.)."""
        if not rooms:
            return

        # Identify wet areas
        wet_keywords = ["toilet", "bathroom", "kitchen", "utility", "wash", "wc", "pantry", "laundry"]
        wet_rooms = [r for r in rooms if any(kw in r.get("room_type", r.get("label", "")).lower() for kw in wet_keywords)]

        if wet_rooms:
            wet_room_types = Counter(r.get("room_type", r.get("label", "")).lower() for r in wet_rooms)
            self.rfis.append({
                "id": self._generate_rfi_id(),
                "issue_type": IssueType.UNCLEAR_SPEC.value,
                "priority": Priority.MEDIUM.value,
                "trade": Trade.CIVIL.value,
                "title": f"Waterproofing specification required for {len(wet_rooms)} wet areas",
                "description": f"Detected {len(wet_rooms)} wet area rooms: "
                               f"{', '.join(f'{v}x {k}' for k, v in wet_room_types.most_common(5))}. "
                               f"Waterproofing treatment and tile specifications needed.",
                "package": "waterproofing",
                "evidence_pages": [],
                "suggested_response": "Specify waterproofing system (membrane type, brand) and tile specifications for wet areas.",
                "wet_room_count": len(wet_rooms),
                "wet_room_types": dict(wet_room_types),
            })
            self.trade_gaps[Trade.CIVIL].append(f"Waterproofing spec for {len(wet_rooms)} wet areas")

    def _sort_rfis(self) -> None:
        """Sort RFIs by priority and trade."""
        priority_order = {Priority.HIGH.value: 0, Priority.MEDIUM.value: 1, Priority.LOW.value: 2}
        trade_order = {
            Trade.STRUCTURAL.value: 0,
            Trade.CIVIL.value: 1,
            Trade.ARCHITECTURAL.value: 2,
            Trade.MEP.value: 3,
            Trade.FINISHES.value: 4,
            Trade.GENERAL.value: 5,
        }

        self.rfis.sort(key=lambda r: (
            priority_order.get(r.get("priority"), 99),
            trade_order.get(r.get("trade"), 99),
            r.get("id", "")
        ))

    def _count_by_priority(self) -> Dict[str, int]:
        """Count RFIs by priority."""
        counts = Counter(r.get("priority") for r in self.rfis)
        return dict(counts)

    def _count_by_issue_type(self) -> Dict[str, int]:
        """Count RFIs by issue type."""
        counts = Counter(r.get("issue_type") for r in self.rfis)
        return dict(counts)

    def _build_trade_summary(self) -> Dict[str, Dict]:
        """Build trade-wise gap summary."""
        summary = {}
        for trade in Trade:
            trade_rfis = [r for r in self.rfis if r.get("trade") == trade.value]
            summary[trade.value] = {
                "rfi_count": len(trade_rfis),
                "gaps": self.trade_gaps.get(trade, []),
                "high_priority": sum(1 for r in trade_rfis if r.get("priority") == Priority.HIGH.value),
            }
        return summary

    def _get_critical_blockers(self) -> List[str]:
        """Get list of critical blockers that prevent bidding."""
        blockers = []
        for rfi in self.rfis:
            if rfi.get("priority") == Priority.HIGH.value:
                impact = rfi.get("impact", rfi.get("title", ""))
                if impact:
                    blockers.append(impact)
        return blockers[:10]


# Keep old RFIGenerator for backward compatibility
class RFIGenerator(EnhancedRFIGenerator):
    """Backward compatible RFI generator - now uses enhanced version."""

    def generate(
        self,
        scope_gaps: List[Dict] = None,
        rooms: List[Dict] = None,
        openings: List[Dict] = None,
        owner_inputs: Dict = None,
    ) -> Dict[str, Any]:
        """Legacy generate method - now delegates to enhanced version."""
        # This is kept for backward compatibility but won't be as good
        # as generate_from_output which has access to more data
        self.rfis = []
        self._rfi_counter = 0

        rooms = rooms or []
        openings = openings or []

        self._analyze_schedule_gaps(openings, rooms)
        self._analyze_room_gaps(rooms)
        self._sort_rfis()

        return {
            "rfis": self.rfis,
            "total_count": len(self.rfis),
            "by_priority": self._count_by_priority(),
        }


def run_rfi_generation(
    output_dir: Path,
    owner_inputs: Dict = None,
) -> Dict[str, Any]:
    """
    Run enhanced RFI generation for a project.

    Args:
        output_dir: Output directory
        owner_inputs: Owner inputs (optional)

    Returns:
        RFI result with comprehensive analysis
    """
    output_dir = Path(output_dir)

    generator = EnhancedRFIGenerator()
    result = generator.generate_from_output(output_dir)

    # Write RFI outputs
    rfi_dir = output_dir / "rfi"
    rfi_dir.mkdir(parents=True, exist_ok=True)

    # Write RFI JSON
    with open(rfi_dir / "rfis.json", "w") as f:
        json.dump(result, f, indent=2)

    # Write comprehensive RFI log markdown
    _write_rfi_markdown(rfi_dir / "rfi_log.md", result)

    # Write trade summary
    _write_trade_summary(rfi_dir / "trade_gaps.md", result)

    return {
        "rfi_count": result["total_count"],
        "output": str(rfi_dir / "rfi_log.md"),
        "rfis": result["rfis"],
        "trade_summary": result["by_trade"],
        "critical_blockers": result["critical_blockers"],
    }


def _write_rfi_markdown(path: Path, result: Dict) -> None:
    """Write RFI log as markdown."""
    with open(path, "w") as f:
        f.write("# RFI Log - Pre-Bid Scope Analysis\n\n")
        f.write(f"**Generated:** {result.get('generated_at', 'N/A')}\n")
        f.write(f"**Total RFIs:** {result['total_count']}\n\n")

        # Priority summary
        by_priority = result.get("by_priority", {})
        f.write("## Priority Summary\n\n")
        f.write(f"- **HIGH:** {by_priority.get('high', 0)} (Must resolve before pricing)\n")
        f.write(f"- **MEDIUM:** {by_priority.get('medium', 0)} (Should clarify)\n")
        f.write(f"- **LOW:** {by_priority.get('low', 0)} (Nice to have)\n\n")

        # Critical blockers
        blockers = result.get("critical_blockers", [])
        if blockers:
            f.write("## Critical Blockers\n\n")
            f.write("These issues **must be resolved** before pricing:\n\n")
            for blocker in blockers:
                f.write(f"- {blocker}\n")
            f.write("\n")

        # Trade-wise summary
        f.write("## Trade-Wise Gap Summary\n\n")
        f.write("| Trade | RFIs | High Priority | Key Gaps |\n")
        f.write("|-------|------|---------------|----------|\n")
        for trade, data in result.get("by_trade", {}).items():
            gaps = data.get("gaps", [])[:2]
            gaps_str = ", ".join(gaps) if gaps else "None"
            f.write(f"| {trade.title()} | {data.get('rfi_count', 0)} | {data.get('high_priority', 0)} | {gaps_str} |\n")
        f.write("\n")

        # Detailed RFI list
        if result["rfis"]:
            f.write("## RFI Details\n\n")

            for rfi in result["rfis"]:
                priority_emoji = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(rfi["priority"], "")

                f.write(f"### {rfi['id']}: {rfi['title']}\n\n")
                f.write(f"**Priority:** {priority_emoji}\n")
                f.write(f"**Trade:** {rfi.get('trade', 'general').title()}\n")
                f.write(f"**Type:** {rfi['issue_type']}\n\n")
                f.write(f"{rfi['description']}\n\n")

                # Evidence pages
                evidence = rfi.get("evidence_pages", [])
                if evidence:
                    f.write(f"**Affected Pages:** {', '.join(str(p) for p in evidence[:10])}\n\n")

                # Detected tags (for schedule RFIs)
                tags = rfi.get("detected_tags", [])
                if tags:
                    f.write(f"**Detected Tags:** `{', '.join(tags[:15])}`{'...' if len(tags) > 15 else ''}\n\n")

                # Impact
                impact = rfi.get("impact")
                if impact:
                    f.write(f"**Impact:** {impact}\n\n")

                f.write(f"**Suggested Response:** {rfi.get('suggested_response', 'Please clarify')}\n\n")
                f.write("---\n\n")
        else:
            f.write("## No RFIs Generated\n\n")
            f.write("All required information appears to be available in the drawing set.\n")


def _write_trade_summary(path: Path, result: Dict) -> None:
    """Write trade-wise gap summary."""
    with open(path, "w") as f:
        f.write("# Trade-Wise Gap Summary\n\n")
        f.write("## Overview\n\n")

        for trade, data in result.get("by_trade", {}).items():
            rfi_count = data.get("rfi_count", 0)
            if rfi_count == 0:
                continue

            f.write(f"### {trade.title()}\n\n")
            f.write(f"**RFIs:** {rfi_count} | **High Priority:** {data.get('high_priority', 0)}\n\n")

            gaps = data.get("gaps", [])
            if gaps:
                f.write("**Key Gaps:**\n")
                for gap in gaps:
                    f.write(f"- {gap}\n")
            f.write("\n")
