"""
Scope Gaps and Estimator Checklist Generation.

Generates:
- scope_gaps.md: Detailed gaps analysis
- estimator_checklist.md: Bid-ready confirmation checklist

NO GREENWASHING: If evidence is missing, the checklist MUST say so explicitly.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime

from .register import ScopeRegister, ScopeItem, ScopeStatus

logger = logging.getLogger(__name__)


@dataclass
class ScopeGap:
    """A single scope gap identified."""
    gap_id: str
    category: str  # missing_sheet, missing_detail, missing_spec, ambiguous, unconfirmed
    package: str
    subpackage: str
    description: str
    impact: str  # high, medium, low
    evidence_found: List[str] = field(default_factory=list)
    evidence_missing: List[str] = field(default_factory=list)
    suggested_action: str = ""

    def to_markdown(self) -> str:
        impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(self.impact, "⚪")
        md = f"### {impact_icon} {self.gap_id}: {self.description}\n\n"
        md += f"**Package:** {self.package} > {self.subpackage}\n\n"
        md += f"**Impact:** {self.impact.upper()}\n\n"

        if self.evidence_found:
            md += f"**Evidence Found:** {', '.join(self.evidence_found)}\n\n"

        if self.evidence_missing:
            md += f"**Evidence Missing:** {', '.join(self.evidence_missing)}\n\n"

        md += f"**Action Required:** {self.suggested_action}\n\n"
        md += "---\n\n"
        return md


@dataclass
class ChecklistItem:
    """A single estimator checklist item."""
    item_id: str
    category: str
    question: str
    context: str
    default_assumption: Optional[str] = None
    risk_if_wrong: str = ""
    priority: str = "medium"  # critical, high, medium, low

    def to_markdown(self) -> str:
        priority_icon = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(self.priority, "⚪")

        md = f"### {priority_icon} [{self.item_id}] {self.category}\n\n"
        md += f"**Question:** {self.question}\n\n"
        md += f"**Context:** {self.context}\n\n"

        if self.default_assumption:
            md += f"**Current Assumption:** {self.default_assumption}\n\n"
        else:
            md += f"**Current Assumption:** ⚠️ NONE - Must be confirmed\n\n"

        md += f"**Risk if Wrong:** {self.risk_if_wrong}\n\n"
        md += "- [ ] Confirmed\n- [ ] Excluded\n- [ ] Modified: _____________\n\n"
        md += "---\n\n"
        return md


class ScopeGapsGenerator:
    """
    Generates scope gaps analysis from scope register.
    """

    # High-impact scope categories that must be explicitly confirmed
    CRITICAL_PACKAGES = [
        "waterproofing",
        "fire_hvac",
        "plumbing_sanitary_swr",
        "electrical_power_lighting",
        "external_works_drainage",
    ]

    # Common missing items in Indian projects
    COMMON_GAPS = {
        "waterproofing": [
            ("toilet_waterproofing", "Toilet/bathroom waterproofing system"),
            ("terrace_waterproofing", "Terrace/roof waterproofing system"),
        ],
        "fire_hvac": [
            ("fire_detection", "Fire detection and alarm system"),
            ("fire_fighting", "Fire fighting equipment/system"),
            ("hvac", "HVAC/AC provisions"),
        ],
        "plumbing_storm_rainwater": [
            ("rainwater_harvesting", "Rainwater harvesting system"),
        ],
        "external_works_drainage": [
            ("external_drainage", "Septic tank/soak pit/STP connection"),
        ],
        "finishes_ceiling": [
            ("false_ceiling", "False ceiling provisions"),
        ],
    }

    def __init__(self):
        self._gap_counter = 0
        self._item_counter = 0

    def _generate_gap_id(self) -> str:
        self._gap_counter += 1
        return f"GAP-{self._gap_counter:03d}"

    def _generate_item_id(self) -> str:
        self._item_counter += 1
        return f"CHK-{self._item_counter:03d}"

    def generate_gaps(
        self,
        register: ScopeRegister,
        evidence_store: Dict[str, Any],
    ) -> List[ScopeGap]:
        """
        Generate scope gaps from register.

        Args:
            register: Scope register
            evidence_store: Evidence store dict

        Returns:
            List of ScopeGap
        """
        gaps = []

        # Get all detail references that might be missing
        detail_refs = evidence_store.get("detail_references", [])
        referenced_sheets = {ref.get("reference", "") for ref in detail_refs}

        for item in register.items:
            # Skip DETECTED items
            if item.status == ScopeStatus.DETECTED:
                continue

            # Determine impact based on package criticality
            is_critical = item.package in self.CRITICAL_PACKAGES
            impact = "high" if is_critical else "medium"

            if item.status == ScopeStatus.MISSING_INPUT:
                gap = ScopeGap(
                    gap_id=self._generate_gap_id(),
                    category="missing_sheet",
                    package=item.package_name,
                    subpackage=item.subpackage_name,
                    description=f"Required drawings/schedules not found for {item.subpackage_name}",
                    impact=impact,
                    evidence_found=item.evidence_pages[:3],
                    evidence_missing=item.missing_info,
                    suggested_action=item.recommended_action,
                )
                gaps.append(gap)

            elif item.status == ScopeStatus.UNKNOWN:
                gap = ScopeGap(
                    gap_id=self._generate_gap_id(),
                    category="unconfirmed",
                    package=item.package_name,
                    subpackage=item.subpackage_name,
                    description=f"No evidence found for {item.subpackage_name} - scope unknown",
                    impact=impact,
                    evidence_found=[],
                    evidence_missing=item.missing_info,
                    suggested_action=f"Confirm if {item.subpackage_name} is included in project scope",
                )
                gaps.append(gap)

            elif item.status == ScopeStatus.IMPLIED:
                gap = ScopeGap(
                    gap_id=self._generate_gap_id(),
                    category="ambiguous",
                    package=item.package_name,
                    subpackage=item.subpackage_name,
                    description=f"Incomplete evidence for {item.subpackage_name} - details unclear",
                    impact="medium",
                    evidence_found=item.evidence_pages[:3],
                    evidence_missing=item.missing_info,
                    suggested_action=item.recommended_action or f"Clarify specifications for {item.subpackage_name}",
                )
                gaps.append(gap)

        # Check for referenced but missing details
        # (This would require tracking which detail references were resolved)

        return gaps

    def generate_checklist(
        self,
        register: ScopeRegister,
        gaps: List[ScopeGap],
    ) -> List[ChecklistItem]:
        """
        Generate estimator confirmation checklist.

        Args:
            register: Scope register
            gaps: List of scope gaps

        Returns:
            List of ChecklistItem
        """
        checklist = []

        # Critical items first
        for item in register.items:
            if item.status == ScopeStatus.UNKNOWN and item.package in self.CRITICAL_PACKAGES:
                checklist.append(ChecklistItem(
                    item_id=self._generate_item_id(),
                    category=f"{item.package_name} - {item.subpackage_name}",
                    question=f"Is {item.subpackage_name} included in the scope?",
                    context=f"No evidence found in drawings. Package: {item.package_name}",
                    default_assumption=None,
                    risk_if_wrong=f"Major cost impact if {item.subpackage_name} is required but not priced",
                    priority="critical",
                ))

        # Waterproofing checks (always critical in Indian construction)
        waterproofing_items = [i for i in register.items if i.package == "waterproofing"]
        for item in waterproofing_items:
            if item.status in [ScopeStatus.UNKNOWN, ScopeStatus.IMPLIED]:
                checklist.append(ChecklistItem(
                    item_id=self._generate_item_id(),
                    category="Waterproofing",
                    question=f"Confirm waterproofing specification for {item.subpackage_name}",
                    context=f"Status: {item.status.value}. Evidence: {', '.join(item.evidence_pages[:2]) or 'None'}",
                    default_assumption="APP membrane assumed" if "toilet" in item.subpackage else None,
                    risk_if_wrong="Waterproofing failure leads to major defects and rework",
                    priority="critical" if "toilet" in item.subpackage or "terrace" in item.subpackage else "high",
                ))

        # Fire/HVAC checks
        fire_hvac_items = [i for i in register.items if i.package == "fire_hvac"]
        if all(i.status == ScopeStatus.UNKNOWN for i in fire_hvac_items):
            checklist.append(ChecklistItem(
                item_id=self._generate_item_id(),
                category="Fire & HVAC",
                question="Confirm if Fire/HVAC scope is included",
                context="No MEP/Fire sheets detected in drawing set",
                default_assumption=None,
                risk_if_wrong="Fire systems may be mandated by local authorities",
                priority="critical",
            ))

        # External works checks
        external_items = [i for i in register.items if "external" in i.package]
        for item in external_items:
            if item.status in [ScopeStatus.UNKNOWN, ScopeStatus.MISSING_INPUT]:
                checklist.append(ChecklistItem(
                    item_id=self._generate_item_id(),
                    category="External Works",
                    question=f"Confirm {item.subpackage_name} requirements",
                    context=f"Status: {item.status.value}. {'; '.join(item.missing_info[:2])}",
                    default_assumption=None,
                    risk_if_wrong=f"External works often underestimated",
                    priority="high",
                ))

        # Rainwater harvesting (often mandated in India)
        rwh_items = [i for i in register.items if "rainwater" in i.subpackage]
        for item in rwh_items:
            if item.status != ScopeStatus.DETECTED:
                checklist.append(ChecklistItem(
                    item_id=self._generate_item_id(),
                    category="Rainwater Harvesting",
                    question="Is rainwater harvesting required by local regulations?",
                    context=f"No dedicated RWH drawing/detail found. Status: {item.status.value}",
                    default_assumption=None,
                    risk_if_wrong="Mandated by most municipal bodies in India",
                    priority="high",
                ))

        # Schedule-dependent items
        schedule_items = [
            ("doors_windows", "wood_doors", "Door schedule"),
            ("doors_windows", "windows", "Window schedule"),
            ("finishes_floor", "vitrified_tiles", "Finish schedule"),
            ("reinforcement_steel", "tmt_bars", "BBS schedule"),
        ]

        for pkg, subpkg, schedule_name in schedule_items:
            items = [i for i in register.items if i.package == pkg and i.subpackage == subpkg]
            for item in items:
                if not item.has_schedule:
                    checklist.append(ChecklistItem(
                        item_id=self._generate_item_id(),
                        category=schedule_name,
                        question=f"Confirm quantities for {item.subpackage_name} (no {schedule_name.lower()} found)",
                        context=f"Detected elements but no dedicated schedule. May need to count from drawings.",
                        default_assumption="Counted from floor plans" if item.has_takeoff else None,
                        risk_if_wrong="Quantity accuracy depends on schedule availability",
                        priority="medium",
                    ))

        # False ceiling check
        ceiling_items = [i for i in register.items if i.subpackage == "false_ceiling"]
        for item in ceiling_items:
            if item.status != ScopeStatus.DETECTED:
                checklist.append(ChecklistItem(
                    item_id=self._generate_item_id(),
                    category="False Ceiling",
                    question="Confirm if false ceiling is included in scope",
                    context="No ceiling plan or false ceiling notes detected",
                    default_assumption=None,
                    risk_if_wrong="False ceiling can be 3-5% of finishing cost",
                    priority="medium",
                ))

        # Add items from gaps with high impact
        for gap in gaps:
            if gap.impact == "high" and gap.category == "unconfirmed":
                # Check if already covered
                existing = [c for c in checklist if gap.subpackage in c.category]
                if not existing:
                    checklist.append(ChecklistItem(
                        item_id=self._generate_item_id(),
                        category=f"{gap.package} - {gap.subpackage}",
                        question=gap.description,
                        context=f"Gap: {gap.evidence_missing}",
                        default_assumption=None,
                        risk_if_wrong="Major scope item with no evidence",
                        priority="high",
                    ))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        checklist.sort(key=lambda x: priority_order.get(x.priority, 4))

        return checklist

    def export_gaps(
        self,
        gaps: List[ScopeGap],
        output_path: Path,
        project_id: str,
    ) -> None:
        """Export gaps to Markdown file."""
        with open(output_path, "w") as f:
            f.write(f"# Scope Gaps Analysis: {project_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Summary
            high_count = sum(1 for g in gaps if g.impact == "high")
            medium_count = sum(1 for g in gaps if g.impact == "medium")
            low_count = sum(1 for g in gaps if g.impact == "low")

            f.write("## Summary\n\n")
            f.write(f"| Impact | Count |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| 🔴 High | {high_count} |\n")
            f.write(f"| 🟡 Medium | {medium_count} |\n")
            f.write(f"| 🟢 Low | {low_count} |\n")
            f.write(f"| **Total** | **{len(gaps)}** |\n\n")

            f.write("---\n\n")

            # Group by category
            by_category: Dict[str, List[ScopeGap]] = {}
            for gap in gaps:
                if gap.category not in by_category:
                    by_category[gap.category] = []
                by_category[gap.category].append(gap)

            category_names = {
                "missing_sheet": "Missing Drawings/Schedules",
                "missing_detail": "Missing Details",
                "missing_spec": "Missing Specifications",
                "ambiguous": "Ambiguous/Incomplete Information",
                "unconfirmed": "Unconfirmed Scope Items",
            }

            for category, category_gaps in by_category.items():
                f.write(f"## {category_names.get(category, category.title())}\n\n")
                for gap in category_gaps:
                    f.write(gap.to_markdown())

            if not gaps:
                f.write("## No Gaps Identified\n\n")
                f.write("All scope items have sufficient evidence.\n")

        logger.info(f"Saved scope gaps to: {output_path}")

    def export_checklist(
        self,
        checklist: List[ChecklistItem],
        output_path: Path,
        project_id: str,
    ) -> None:
        """Export checklist to Markdown file."""
        with open(output_path, "w") as f:
            f.write(f"# Estimator Checklist: {project_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Instructions\n\n")
            f.write("Review each item below and confirm scope with the client/architect.\n")
            f.write("**DO NOT ASSUME** - all items marked require explicit confirmation.\n\n")
            f.write("---\n\n")

            # Summary
            critical_count = sum(1 for c in checklist if c.priority == "critical")
            high_count = sum(1 for c in checklist if c.priority == "high")

            f.write("## Summary\n\n")
            f.write(f"| Priority | Count |\n")
            f.write(f"|----------|-------|\n")
            f.write(f"| 🔴 Critical | {critical_count} |\n")
            f.write(f"| 🟠 High | {high_count} |\n")
            f.write(f"| 🟡 Medium | {len(checklist) - critical_count - high_count} |\n")
            f.write(f"| **Total** | **{len(checklist)}** |\n\n")

            if critical_count > 0:
                f.write("⚠️ **WARNING:** Critical items require immediate clarification before bid submission.\n\n")

            f.write("---\n\n")

            # Group by priority
            f.write("## Critical Items (Must Confirm)\n\n")
            for item in checklist:
                if item.priority == "critical":
                    f.write(item.to_markdown())

            f.write("## High Priority Items\n\n")
            for item in checklist:
                if item.priority == "high":
                    f.write(item.to_markdown())

            f.write("## Medium Priority Items\n\n")
            for item in checklist:
                if item.priority == "medium":
                    f.write(item.to_markdown())

            f.write("## Low Priority Items\n\n")
            for item in checklist:
                if item.priority == "low":
                    f.write(item.to_markdown())

            if not checklist:
                f.write("No items require confirmation - all scope is clearly documented.\n")

            # Sign-off section
            f.write("---\n\n")
            f.write("## Sign-Off\n\n")
            f.write("| | |\n")
            f.write("|---|---|\n")
            f.write("| Reviewed By | __________________ |\n")
            f.write("| Date | __________________ |\n")
            f.write("| Client Confirmation | __________________ |\n")

        logger.info(f"Saved estimator checklist to: {output_path}")


def generate_checklist(
    register: ScopeRegister,
    evidence_store: Dict[str, Any],
    output_dir: Path,
    project_id: str,
) -> tuple:
    """
    Convenience function to generate gaps and checklist.

    Args:
        register: Scope register
        evidence_store: Evidence store dict
        output_dir: Output directory
        project_id: Project identifier

    Returns:
        Tuple of (gaps, checklist)
    """
    generator = ScopeGapsGenerator()

    # Generate gaps
    gaps = generator.generate_gaps(register, evidence_store)

    # Generate checklist
    checklist = generator.generate_checklist(register, gaps)

    # Export files
    scope_dir = output_dir / "scope"
    scope_dir.mkdir(parents=True, exist_ok=True)

    generator.export_gaps(gaps, scope_dir / "scope_gaps.md", project_id)
    generator.export_checklist(checklist, scope_dir / "estimator_checklist.md", project_id)

    return gaps, checklist
