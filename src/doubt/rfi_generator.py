"""
Doubt RFI Generator - Generate high-priority RFIs from missing sheets.
"""

from dataclasses import dataclass, field
from typing import List, Dict

from .detector import DoubtReport, MissingSheet


@dataclass
class DoubtRFI:
    """RFI generated from doubt analysis."""
    rfi_id: str
    sheet_type: str
    priority: str
    question: str
    why_needed: str
    impacted_packages: List[str] = field(default_factory=list)
    suggested_resolution: str = ""
    workaround: str = ""

    def to_dict(self) -> dict:
        return {
            "rfi_id": self.rfi_id,
            "sheet_type": self.sheet_type,
            "priority": self.priority,
            "question": self.question,
            "why_needed": self.why_needed,
            "impacted_packages": self.impacted_packages,
            "suggested_resolution": self.suggested_resolution,
            "workaround": self.workaround,
            "issue_type": "missing_input",
        }


class DoubtRFIGenerator:
    """Generate RFIs from doubt analysis."""

    # RFI templates for each missing sheet type
    RFI_TEMPLATES = {
        "site_plan": {
            "question": "Please provide site plan showing plot boundaries, building footprint, parking, access roads, and external works areas.",
            "suggested_resolution": "Provide site plan with dimensions and levels",
            "workaround": "Assume standard external works allowance based on building footprint",
        },
        "section": {
            "question": "Please provide building sections showing floor heights, slab thicknesses, staircase dimensions, and parapet heights.",
            "suggested_resolution": "Provide minimum 2 sections (longitudinal and cross)",
            "workaround": "Assume standard floor height 3.0m, slab 150mm, parapet 1.0m",
        },
        "elevation": {
            "question": "Please provide elevations showing external finishes, window/door positions, and facade details.",
            "suggested_resolution": "Provide all 4 elevations with finish specifications",
            "workaround": "Assume external finish same as nearest similar project",
        },
        "foundation_plan": {
            "question": "Please provide foundation plan showing footing sizes, depths, plinth beam layout, and column positions.",
            "suggested_resolution": "Provide foundation plan with structural details",
            "workaround": "Cannot proceed without foundation details - high risk item",
        },
        "structural_plan": {
            "question": "Please provide structural plans showing column grid, beam layout, slab details, and reinforcement schedules.",
            "suggested_resolution": "Provide all floor structural plans with schedules",
            "workaround": "Cannot estimate RCC quantities without structural drawings",
        },
        "plumbing_layout": {
            "question": "Please provide plumbing layouts showing water supply lines, drainage, shaft locations, and fixture positions.",
            "suggested_resolution": "Provide floor-wise plumbing layouts for all services",
            "workaround": "Estimate using standard pipe run factors (3m per fixture point)",
        },
        "electrical_layout": {
            "question": "Please provide electrical layouts showing wiring points, panel locations, conduit runs, and load schedule.",
            "suggested_resolution": "Provide floor-wise electrical layouts with load calculations",
            "workaround": "Estimate using standard point density (1 point per 4 sqm residential)",
        },
        "fire_layout": {
            "question": "Please confirm if fire fighting system is in scope. If yes, provide fire system layout.",
            "suggested_resolution": "Provide fire system layout with specifications",
            "workaround": "Exclude from estimate pending confirmation",
        },
        "door_schedule": {
            "question": "Please provide door schedule showing door types, sizes, materials, and hardware specifications.",
            "suggested_resolution": "Provide complete door schedule with all details",
            "workaround": "Count doors from floor plans, assume standard sizes",
        },
        "window_schedule": {
            "question": "Please provide window schedule showing window types, sizes, materials, and glazing specifications.",
            "suggested_resolution": "Provide complete window schedule with all details",
            "workaround": "Count windows from floor plans/elevations, assume standard sizes",
        },
        "finish_schedule": {
            "question": "Please provide room finish schedule showing floor, wall, and ceiling finishes for each room type.",
            "suggested_resolution": "Provide finish schedule with specifications",
            "workaround": "Assume standard finishes based on room types (vitrified for common, ceramic for wet)",
        },
        "column_schedule": {
            "question": "Please provide column schedule showing sizes and reinforcement for all column types.",
            "suggested_resolution": "Provide column schedule with bar bending schedules",
            "workaround": "Cannot estimate column steel without schedule",
        },
        "toilet_detail": {
            "question": "Please provide toilet detail drawings showing waterproofing treatment, tiling, and CP fixture layout.",
            "suggested_resolution": "Provide enlarged toilet plans with sections",
            "workaround": "Assume standard waterproofing (cementitious coating) and tiling (full height dado)",
        },
        "staircase_detail": {
            "question": "Please provide staircase detail drawings showing tread/riser dimensions, railing details, and finishes.",
            "suggested_resolution": "Provide staircase sections and details",
            "workaround": "Assume standard treads 300mm, risers 150mm, MS railing",
        },
        "terrace_detail": {
            "question": "Please provide terrace detail drawings showing waterproofing treatment, slope, and weathering course.",
            "suggested_resolution": "Provide terrace section with waterproofing specifications",
            "workaround": "Assume standard terrace waterproofing (APP membrane + brick bat coba)",
        },
    }

    def __init__(self):
        self.rfi_counter = 0

    def generate(self, doubt_report: DoubtReport) -> List[DoubtRFI]:
        """Generate RFIs from doubt report."""
        rfis = []

        for missing in doubt_report.missing_sheets:
            rfi = self._create_rfi(missing)
            if rfi:
                rfis.append(rfi)

        # Sort by priority (high first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        rfis.sort(key=lambda r: priority_order.get(r.priority, 3))

        return rfis

    def _create_rfi(self, missing: MissingSheet) -> DoubtRFI:
        """Create RFI from missing sheet."""
        self.rfi_counter += 1
        rfi_id = f"DRFI-{self.rfi_counter:04d}"

        # Get template
        template = self.RFI_TEMPLATES.get(missing.sheet_type, {})

        # Determine priority from severity
        priority_map = {
            "critical": "high",
            "important": "medium",
            "optional": "low",
        }
        priority = priority_map.get(missing.severity, "medium")

        # Build question
        question = template.get("question")
        if not question:
            question = f"Please provide {missing.sheet_type.replace('_', ' ')} drawings."

        return DoubtRFI(
            rfi_id=rfi_id,
            sheet_type=missing.sheet_type,
            priority=priority,
            question=question,
            why_needed=missing.why_needed,
            impacted_packages=missing.impacted_packages,
            suggested_resolution=template.get("suggested_resolution", "Provide requested drawings"),
            workaround=template.get("workaround", "Add contingency for uncertainty"),
        )

    def merge_with_existing_rfis(
        self,
        doubt_rfis: List[DoubtRFI],
        existing_rfis: List[Dict],
    ) -> List[Dict]:
        """Merge doubt RFIs with existing RFI list."""
        merged = []

        # Add doubt RFIs first (they are high priority)
        for drfi in doubt_rfis:
            rfi_dict = drfi.to_dict()
            rfi_dict["source"] = "doubt_engine"
            merged.append(rfi_dict)

        # Add existing RFIs that don't duplicate
        doubt_packages = set()
        for drfi in doubt_rfis:
            doubt_packages.update(drfi.impacted_packages)

        for existing in existing_rfis:
            # Check if this RFI is about the same missing sheet
            existing_package = existing.get("package", "")
            existing_type = existing.get("issue_type", "")

            # Don't add duplicates about missing inputs for same packages
            if existing_type == "missing_input" and existing_package in doubt_packages:
                continue

            merged.append(existing)

        return merged
