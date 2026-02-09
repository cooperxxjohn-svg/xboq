"""
Impact Analyzer - Analyze how revisions affect scope and BOQ.

Provides:
- Scope impact identification
- BOQ quantity impact tracking
- Cost impact estimation
- Recommendations for review
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from .tracker import RevisionHistory
from .comparator import ChangeReport, ChangeType


class ImpactLevel(Enum):
    """Level of impact from change."""
    HIGH = "high"       # Major change, requires review
    MEDIUM = "medium"   # Significant change
    LOW = "low"         # Minor change
    NONE = "none"       # No impact


@dataclass
class ImpactedItem:
    """Item impacted by revision."""
    item_id: str
    item_type: str  # scope, boq, room, door, window, etc.
    package: str = ""
    impact_level: str = "medium"
    reason: str = ""
    source_sheets: List[str] = field(default_factory=list)
    old_quantity: Optional[float] = None
    new_quantity: Optional[float] = None
    quantity_delta: Optional[float] = None
    cost_impact: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "package": self.package,
            "impact_level": self.impact_level,
            "reason": self.reason,
            "source_sheets": self.source_sheets,
            "old_quantity": self.old_quantity,
            "new_quantity": self.new_quantity,
            "quantity_delta": self.quantity_delta,
            "cost_impact": self.cost_impact,
        }


@dataclass
class ImpactReport:
    """Report of revision impacts."""
    impacted_scope_items: List[ImpactedItem] = field(default_factory=list)
    impacted_boq_items: List[ImpactedItem] = field(default_factory=list)
    impacted_rooms: List[ImpactedItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    high_impact_count: int = 0
    medium_impact_count: int = 0
    low_impact_count: int = 0
    total_cost_impact: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "impacted_scope_items": [i.to_dict() for i in self.impacted_scope_items],
            "impacted_boq_items": [i.to_dict() for i in self.impacted_boq_items],
            "impacted_rooms": [i.to_dict() for i in self.impacted_rooms],
            "recommendations": self.recommendations,
            "high_impact_count": self.high_impact_count,
            "medium_impact_count": self.medium_impact_count,
            "low_impact_count": self.low_impact_count,
            "total_cost_impact": self.total_cost_impact,
        }


class ImpactAnalyzer:
    """Analyze impact of revisions on scope and BOQ."""

    # Package keywords for mapping changes
    PACKAGE_KEYWORDS = {
        "rcc_concrete": ["rcc", "concrete", "slab", "column", "beam", "foundation"],
        "masonry_partitions": ["wall", "masonry", "partition", "brick", "block", "aac"],
        "finishes_floor": ["floor", "flooring", "tile", "vitrified", "granite", "marble"],
        "finishes_wall": ["wall finish", "plaster", "putty", "paint"],
        "finishes_ceiling": ["ceiling", "false ceiling", "gypsum"],
        "doors_windows": ["door", "window", "shutter", "frame"],
        "waterproofing": ["waterproof", "damp proof", "water tank", "terrace"],
        "plumbing": ["plumbing", "toilet", "bathroom", "sanitary", "cp fitting"],
        "electrical": ["electrical", "switch", "socket", "light", "wiring"],
        "external_works": ["external", "paving", "compound", "gate"],
    }

    # High-cost items that warrant high impact level
    HIGH_COST_ITEMS = [
        "waterproofing",
        "rcc_concrete",
        "structural",
        "foundation",
        "external_works",
        "facade",
        "aluminum_glazing",
    ]

    def __init__(self):
        pass

    def analyze(
        self,
        revision_history: RevisionHistory,
        change_report: Optional[ChangeReport],
        scope_register: Dict,
        boq_entries: List[Dict],
    ) -> ImpactReport:
        """
        Analyze impact of revisions.

        Args:
            revision_history: Revision history
            change_report: Change report (if comparing versions)
            scope_register: Scope register dict
            boq_entries: BOQ entries list

        Returns:
            ImpactReport with all impacts
        """
        report = ImpactReport()

        # If no changes, analyze based on revision descriptions
        if change_report:
            # Analyze actual changes
            self._analyze_changes(report, change_report, scope_register, boq_entries)
        else:
            # Analyze based on revision descriptions
            self._analyze_revisions(report, revision_history, scope_register, boq_entries)

        # Count impacts
        all_items = (
            report.impacted_scope_items +
            report.impacted_boq_items +
            report.impacted_rooms
        )

        for item in all_items:
            if item.impact_level == "high":
                report.high_impact_count += 1
            elif item.impact_level == "medium":
                report.medium_impact_count += 1
            else:
                report.low_impact_count += 1

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _analyze_changes(
        self,
        report: ImpactReport,
        change_report: ChangeReport,
        scope_register: Dict,
        boq_entries: List[Dict],
    ) -> None:
        """Analyze actual changes from comparison."""
        # Map sheets to scope items
        scope_items = scope_register.get("items", [])
        if isinstance(scope_items, dict):
            scope_items = list(scope_items.values())

        scope_by_sheet = {}
        for item in scope_items:
            sheets = item.get("source_pages", [])
            for sheet in sheets:
                if sheet not in scope_by_sheet:
                    scope_by_sheet[sheet] = []
                scope_by_sheet[sheet].append(item)

        # Map sheets to BOQ items
        boq_by_sheet = {}
        for entry in boq_entries:
            sheet = entry.get("source_page", "")
            if sheet:
                if sheet not in boq_by_sheet:
                    boq_by_sheet[sheet] = []
                boq_by_sheet[sheet].append(entry)

        # Process each changed sheet
        for diff in change_report.diffs:
            if diff.change_type == ChangeType.UNCHANGED:
                continue

            sheet_id = diff.sheet_id

            # Impact on scope items
            if sheet_id in scope_by_sheet:
                for scope_item in scope_by_sheet[sheet_id]:
                    impact_level = self._determine_impact_level(
                        diff, scope_item.get("package", "")
                    )

                    report.impacted_scope_items.append(ImpactedItem(
                        item_id=scope_item.get("item_id", ""),
                        item_type="scope",
                        package=scope_item.get("package", ""),
                        impact_level=impact_level,
                        reason=f"Sheet {sheet_id} changed: {diff.summary}",
                        source_sheets=[sheet_id],
                    ))

            # Impact on BOQ items
            if sheet_id in boq_by_sheet:
                for boq_item in boq_by_sheet[sheet_id]:
                    impact_level = self._determine_impact_level(
                        diff, boq_item.get("category", "")
                    )

                    report.impacted_boq_items.append(ImpactedItem(
                        item_id=boq_item.get("item_code", ""),
                        item_type="boq",
                        package=boq_item.get("category", ""),
                        impact_level=impact_level,
                        reason=f"Sheet {sheet_id} changed: {diff.summary}",
                        source_sheets=[sheet_id],
                        old_quantity=boq_item.get("quantity"),
                        new_quantity=boq_item.get("quantity"),  # Same until re-extracted
                    ))

            # Impact on rooms
            for change in diff.geometry_changes:
                if "Room" in change or "room" in change:
                    report.impacted_rooms.append(ImpactedItem(
                        item_id=change.split(":")[0] if ":" in change else change,
                        item_type="room",
                        impact_level="medium",
                        reason=change,
                        source_sheets=[sheet_id],
                    ))

    def _analyze_revisions(
        self,
        report: ImpactReport,
        revision_history: RevisionHistory,
        scope_register: Dict,
        boq_entries: List[Dict],
    ) -> None:
        """Analyze based on revision descriptions."""
        for sheet_id, sheet_rev in revision_history.sheets.items():
            for rev in sheet_rev.revisions:
                # Parse description for impacted items
                desc = rev.description.lower()

                # Find matching packages
                matched_packages = []
                for package, keywords in self.PACKAGE_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in desc:
                            matched_packages.append(package)
                            break

                for package in matched_packages:
                    impact_level = self._impact_from_revision_type(
                        rev.revision_type.value
                    )

                    report.impacted_scope_items.append(ImpactedItem(
                        item_id=f"{package}_revision_{rev.revision}",
                        item_type="scope",
                        package=package,
                        impact_level=impact_level,
                        reason=f"Revision {rev.revision}: {rev.description[:80]}",
                        source_sheets=[sheet_id],
                    ))

    def _determine_impact_level(self, diff, package: str) -> str:
        """Determine impact level from diff and package."""
        # High impact if geometry changed for high-cost items
        if diff.change_type == ChangeType.GEOMETRY:
            package_lower = package.lower()
            for high_cost in self.HIGH_COST_ITEMS:
                if high_cost in package_lower:
                    return "high"
            return "medium"

        # Medium impact for text changes
        if diff.change_type == ChangeType.TEXT:
            return "medium"

        # Low impact for revision-only changes
        if diff.change_type == ChangeType.REVISION:
            return "low"

        return "medium"

    def _impact_from_revision_type(self, rev_type: str) -> str:
        """Get impact level from revision type."""
        if rev_type in ["structural", "mep"]:
            return "high"
        elif rev_type in ["design", "client", "asi"]:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, report: ImpactReport) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []

        # High impact items
        if report.high_impact_count > 0:
            recommendations.append(
                f"⚠️ {report.high_impact_count} high-impact changes detected. "
                "Re-run takeoff for affected sheets before finalizing BOQ."
            )

        # Structural changes
        structural_impacts = [
            i for i in report.impacted_scope_items
            if "structural" in i.package.lower() or "rcc" in i.package.lower()
        ]
        if structural_impacts:
            recommendations.append(
                f"Structural scope affected by {len(structural_impacts)} changes. "
                "Verify RCC quantities with updated drawings."
            )

        # Waterproofing changes
        wp_impacts = [
            i for i in report.impacted_scope_items
            if "waterproof" in i.package.lower()
        ]
        if wp_impacts:
            recommendations.append(
                "Waterproofing scope may be affected. Verify coverage and specifications."
            )

        # Room area changes
        if report.impacted_rooms:
            recommendations.append(
                f"{len(report.impacted_rooms)} rooms have changed areas. "
                "Update floor finish and plaster quantities."
            )

        # Added sheets
        if report.impacted_boq_items:
            unique_sheets = set()
            for item in report.impacted_boq_items:
                unique_sheets.update(item.source_sheets)
            recommendations.append(
                f"Review {len(unique_sheets)} sheets for BOQ quantity updates."
            )

        if not recommendations:
            recommendations.append(
                "No significant impacts detected. Proceed with current BOQ."
            )

        return recommendations

    def estimate_cost_impact(
        self,
        impacted_items: List[ImpactedItem],
        rate_data: Dict[str, float],
    ) -> float:
        """Estimate cost impact of changes."""
        total_impact = 0.0

        for item in impacted_items:
            if item.quantity_delta and item.item_id in rate_data:
                rate = rate_data[item.item_id]
                impact = item.quantity_delta * rate
                item.cost_impact = impact
                total_impact += impact

        return total_impact
