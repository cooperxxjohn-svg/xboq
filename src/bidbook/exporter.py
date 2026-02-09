"""
Bid Book Exporter - Main export logic for bid documents.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List


class BidBookExporter:
    """Export complete bid package."""

    def __init__(self, project_id: str, output_dir: Path):
        self.project_id = project_id
        self.output_dir = output_dir

    def generate_assumptions(self, bid_data: dict) -> str:
        """Generate assumptions document."""
        lines = []
        lines.append(f"# Bid Assumptions: {self.project_id}\n")
        lines.append(f"Generated: {datetime.now().isoformat()}\n\n")

        lines.append("## Scope Assumptions\n\n")

        # From drawings
        if bid_data.get("extraction_assumptions"):
            lines.append("### Drawing-based Assumptions\n\n")
            for assumption in bid_data["extraction_assumptions"]:
                lines.append(f"- {assumption}\n")
            lines.append("\n")

        # From owner inputs
        if bid_data.get("owner_input_assumptions"):
            lines.append("### Owner Input Assumptions\n\n")
            for assumption in bid_data["owner_input_assumptions"]:
                lines.append(f"- {assumption}\n")
            lines.append("\n")

        # Applied defaults
        if bid_data.get("applied_defaults"):
            lines.append("### Applied Defaults (Owner Input Not Provided)\n\n")
            lines.append("| Field | Default Value | Basis |\n")
            lines.append("|-------|---------------|-------|\n")
            for default in bid_data["applied_defaults"][:20]:  # Limit to 20
                lines.append(f"| {default.get('field_path', '')} | {default.get('value', '')} | {default.get('basis', '')} |\n")
            if len(bid_data["applied_defaults"]) > 20:
                lines.append(f"\n*...and {len(bid_data['applied_defaults']) - 20} more defaults*\n")
            lines.append("\n")

        lines.append("## Pricing Assumptions\n\n")

        # Location factor
        if bid_data.get("location_factor"):
            lines.append(f"- Location factor applied: {bid_data['location_factor']:.2f}\n")

        # Grade
        if bid_data.get("finish_grade"):
            lines.append(f"- Finish grade assumed: {bid_data['finish_grade']}\n")

        # Base rates
        lines.append("- Base rates as per CPWD DSR 2024 with market adjustments\n")
        lines.append("- Labor rates as per ISR 2024\n")
        lines.append("- All rates exclusive of GST unless specified\n\n")

        lines.append("## Commercial Assumptions\n\n")
        lines.append("- GST @ 18% applicable on all work items\n")
        lines.append("- Prices valid for 90 days from bid submission\n")
        lines.append("- Escalation as per tender conditions (if applicable)\n")
        lines.append("- Retention @ 5% unless otherwise specified\n")
        lines.append("- Defect Liability Period: 12 months from completion\n\n")

        lines.append("## Exclusions from Bid\n\n")
        lines.append("The following are excluded unless specifically mentioned in scope:\n\n")
        standard_exclusions = [
            "Soil investigation and testing",
            "Pile foundation (unless specified)",
            "External development beyond plot boundary",
            "Furniture and loose furnishings",
            "Modular kitchen cabinets (only provision)",
            "Air conditioning units (only provision)",
            "Solar panels and systems",
            "Rainwater harvesting (unless mandated)",
            "Swimming pool and allied works",
            "Basement dewatering if water table high",
            "Rock excavation premium",
            "Night shift work premium",
            "Any work beyond tender drawings",
        ]
        for exc in standard_exclusions:
            lines.append(f"- {exc}\n")

        return "".join(lines)

    def generate_clarifications(self, bid_data: dict) -> str:
        """Generate clarifications and deviations document."""
        lines = []
        lines.append(f"# Clarifications and Deviations: {self.project_id}\n")
        lines.append(f"Generated: {datetime.now().isoformat()}\n\n")

        lines.append("## Clarifications Sought\n\n")
        lines.append("The following clarifications are requested before bid submission:\n\n")

        clarification_count = 1

        # From missing mandatory inputs
        if bid_data.get("missing_mandatory"):
            for field in bid_data["missing_mandatory"][:10]:
                lines.append(f"**C-{clarification_count:03d}**: {field.get('why_needed', 'Specification required')}\n")
                lines.append(f"- Field: {field.get('path', '')}\n")
                lines.append(f"- Impact: {field.get('impact', 'Affects pricing accuracy')}\n\n")
                clarification_count += 1

        # From alignment discrepancies
        if bid_data.get("discrepancies"):
            lines.append("### Quantity Discrepancies\n\n")
            for disc in bid_data["discrepancies"][:5]:
                lines.append(f"**C-{clarification_count:03d}**: Quantity difference for {disc.get('description', '')[:40]}\n")
                lines.append(f"- Drawing qty: {disc.get('drawings_qty', 0):.2f} {disc.get('unit', '')}\n")
                lines.append(f"- Owner BOQ qty: {disc.get('owner_qty', 0):.2f} {disc.get('unit', '')}\n")
                lines.append(f"- Request: Please confirm applicable quantity\n\n")
                clarification_count += 1

        lines.append("## Deviations Proposed\n\n")
        lines.append("The following deviations are proposed from the tender requirements:\n\n")
        lines.append("*No deviations proposed. Bid is fully compliant with tender requirements.*\n\n")

        lines.append("## Alternative Proposals\n\n")
        lines.append("The following value engineering alternatives are offered:\n\n")
        lines.append("*No alternatives proposed at this stage. Can be discussed post-award.*\n\n")

        return "".join(lines)

    def generate_analysis(self, bid_data: dict) -> dict:
        """Generate bid analysis JSON."""
        # Calculate totals
        works_total = sum(item.get("amount", 0) for item in bid_data.get("priced_boq", []))
        prelims_total = sum(item.amount for item in bid_data.get("prelims_items", []))
        subtotal = works_total + prelims_total

        # Add contingency and margin
        contingency = subtotal * 0.03  # 3%
        margin = subtotal * 0.05  # 5%
        total_before_gst = subtotal + contingency + margin
        gst = total_before_gst * 0.18
        grand_total = total_before_gst + gst

        # Package breakdown
        by_package = {}
        for item in bid_data.get("priced_boq", []):
            pkg = item.get("package", "miscellaneous")
            if pkg not in by_package:
                by_package[pkg] = 0
            by_package[pkg] += item.get("amount", 0)

        return {
            "project_id": self.project_id,
            "generated": datetime.now().isoformat(),
            "totals": {
                "works_total": round(works_total, 2),
                "prelims_total": round(prelims_total, 2),
                "subtotal": round(subtotal, 2),
                "contingency_3pct": round(contingency, 2),
                "margin_5pct": round(margin, 2),
                "total_before_gst": round(total_before_gst, 2),
                "gst_18pct": round(gst, 2),
                "grand_total": round(grand_total, 2),
            },
            "by_package": {k: round(v, 2) for k, v in by_package.items()},
            "metrics": {
                "boq_items": len(bid_data.get("priced_boq", [])),
                "prelims_items": len(bid_data.get("prelims_items", [])),
                "rfis_pending": bid_data.get("total_rfis", 0),
                "assumptions_count": len(bid_data.get("assumptions", [])),
                "defaults_applied": len(bid_data.get("applied_defaults", [])),
            },
            "rates": {
                "per_sqft": round(grand_total / max(1, bid_data.get("built_up_area_sqm", 1) * 10.764), 2),
                "per_sqm": round(grand_total / max(1, bid_data.get("built_up_area_sqm", 1)), 2),
            },
            "confidence": {
                "pricing_confidence": bid_data.get("pricing_confidence", 0.7),
                "scope_confidence": bid_data.get("scope_confidence", 0.8),
                "overall": (bid_data.get("pricing_confidence", 0.7) + bid_data.get("scope_confidence", 0.8)) / 2,
            },
        }

    def compile_rfis(self, bid_data: dict) -> str:
        """Compile all RFIs into a single document."""
        lines = []
        lines.append(f"# RFIs to Raise: {self.project_id}\n")
        lines.append(f"Generated: {datetime.now().isoformat()}\n\n")

        rfi_count = 0

        # Doubt-based RFIs (missing drawings)
        if bid_data.get("doubt_rfis"):
            lines.append("## Missing Drawing RFIs\n\n")
            for rfi in bid_data["doubt_rfis"]:
                rfi_count += 1
                priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.get("priority", ""), "⚪")
                lines.append(f"### {priority_emoji} {rfi.get('rfi_id', f'RFI-{rfi_count:04d}')}\n\n")
                lines.append(f"**Question**: {rfi.get('question', '')}\n\n")
                lines.append(f"**Why Needed**: {rfi.get('why_needed', '')}\n\n")
                if rfi.get("workaround"):
                    lines.append(f"**Workaround if not received**: {rfi.get('workaround', '')}\n\n")
                lines.append("---\n\n")

        # Owner input RFIs
        if bid_data.get("owner_input_rfis"):
            lines.append("## Owner Input RFIs\n\n")
            for rfi in bid_data["owner_input_rfis"]:
                rfi_count += 1
                priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.get("priority", ""), "⚪")
                lines.append(f"### {priority_emoji} {rfi.get('rfi_id', f'RFI-{rfi_count:04d}')}\n\n")
                lines.append(f"**Question**: {rfi.get('question', '')}\n\n")
                lines.append(f"**Options**: {', '.join(rfi.get('options', []))}\n\n")
                if rfi.get("default_assumption"):
                    lines.append(f"**Default Assumption**: {rfi.get('default_assumption', '')}\n\n")
                lines.append("---\n\n")

        # Scope register RFIs
        if bid_data.get("scope_rfis"):
            lines.append("## Scope Clarification RFIs\n\n")
            for rfi in bid_data["scope_rfis"]:
                rfi_count += 1
                lines.append(f"### RFI-{rfi_count:04d}: {rfi.get('package', '')}\n\n")
                lines.append(f"**Question**: {rfi.get('question', '')}\n\n")
                lines.append("---\n\n")

        lines.append(f"\n## Summary\n\n")
        lines.append(f"**Total RFIs**: {rfi_count}\n")

        return "".join(lines)

    def generate_submission_checklist(self, bid_data: dict) -> str:
        """Generate submission checklist."""
        lines = []
        lines.append(f"# Bid Submission Checklist: {self.project_id}\n")
        lines.append(f"Generated: {datetime.now().isoformat()}\n\n")

        lines.append("## Technical Documents\n\n")
        technical_items = [
            ("Priced BOQ in prescribed format", "priced_boq.xlsx"),
            ("Technical specifications compliance", "Not included - manual"),
            ("Method statements for key activities", "Not included - manual"),
            ("Project organization chart", "Not included - manual"),
            ("Equipment deployment plan", "prelims_breakdown.md"),
            ("Quality assurance plan", "Not included - manual"),
            ("Safety plan", "Not included - manual"),
            ("Work program / bar chart", "Not included - manual"),
        ]

        for item, status in technical_items:
            check = "✅" if "Not included" not in status else "⬜"
            lines.append(f"- [{check}] {item}: `{status}`\n")

        lines.append("\n## Commercial Documents\n\n")
        commercial_items = [
            ("Bid summary / cover letter", "bid_summary.md"),
            ("Assumptions and exclusions", "assumptions.md"),
            ("Clarifications and deviations", "clarifications.md"),
            ("RFIs / queries", "rfis_to_raise.md"),
            ("EMD / bid bond", "Not included - manual"),
            ("Company profile", "Not included - manual"),
            ("Financial statements", "Not included - manual"),
            ("Experience certificates", "Not included - manual"),
        ]

        for item, status in commercial_items:
            check = "✅" if "Not included" not in status else "⬜"
            lines.append(f"- [{check}] {item}: `{status}`\n")

        lines.append("\n## Pre-Submission Checks\n\n")
        checks = [
            "All BOQ items have rates",
            "GST calculations verified",
            "Arithmetic checked",
            "Validity period confirmed",
            "Authorized signatory details correct",
            "Number of copies as required",
            "Sealed envelopes labeled correctly",
            "Submission deadline noted",
        ]

        for check in checks:
            lines.append(f"- [ ] {check}\n")

        lines.append("\n## Notes\n\n")
        lines.append("- Review all documents before submission\n")
        lines.append("- Ensure compliance with tender instructions\n")
        lines.append("- Keep copy of submitted documents\n")
        lines.append("- Note submission receipt number\n")

        return "".join(lines)
