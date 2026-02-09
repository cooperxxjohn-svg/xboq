"""
Summary Generator - Generate executive summary for bid.
"""

from datetime import datetime


class SummaryGenerator:
    """Generate executive summary for bid submission."""

    def generate(self, bid_data: dict) -> str:
        """Generate executive summary markdown."""
        lines = []

        project_info = bid_data.get("project_info", {})
        project_name = project_info.get("name", bid_data.get("project_id", "Project"))

        lines.append(f"# Bid Summary: {project_name}\n\n")
        lines.append(f"**Submission Date**: {datetime.now().strftime('%d-%b-%Y')}\n\n")

        # Project overview
        lines.append("## Project Overview\n\n")
        lines.append("| Parameter | Value |\n")
        lines.append("|-----------|-------|\n")
        lines.append(f"| Project Name | {project_name} |\n")
        lines.append(f"| Location | {project_info.get('location', {}).get('city', 'N/A')}, {project_info.get('location', {}).get('state', '')} |\n")
        lines.append(f"| Project Type | {project_info.get('type', 'N/A').title()} |\n")
        lines.append(f"| Built-up Area | {bid_data.get('built_up_area_sqm', 0):,.0f} sqm |\n")
        lines.append(f"| Duration | {project_info.get('completion_months', 'N/A')} months |\n")
        lines.append(f"| Finish Grade | {bid_data.get('finish_grade', 'Standard').title()} |\n\n")

        # Bid value
        lines.append("## Bid Value Summary\n\n")

        works_total = sum(item.get("amount", 0) for item in bid_data.get("priced_boq", []))
        prelims_total = sum(item.amount if hasattr(item, 'amount') else item.get('amount', 0)
                          for item in bid_data.get("prelims_items", []))
        subtotal = works_total + prelims_total
        contingency = subtotal * 0.03
        margin = subtotal * 0.05
        total_before_gst = subtotal + contingency + margin
        gst = total_before_gst * 0.18
        grand_total = total_before_gst + gst

        lines.append("| Component | Amount (₹) |\n")
        lines.append("|-----------|------------|\n")
        lines.append(f"| **Works Total** | {works_total:,.2f} |\n")
        lines.append(f"| **Preliminaries** | {prelims_total:,.2f} |\n")
        lines.append(f"| *Sub-total* | *{subtotal:,.2f}* |\n")
        lines.append(f"| Contingency (3%) | {contingency:,.2f} |\n")
        lines.append(f"| Margin (5%) | {margin:,.2f} |\n")
        lines.append(f"| *Total before GST* | *{total_before_gst:,.2f}* |\n")
        lines.append(f"| GST (18%) | {gst:,.2f} |\n")
        lines.append(f"| **GRAND TOTAL** | **₹{grand_total:,.2f}** |\n\n")

        # Rate analysis
        if bid_data.get("built_up_area_sqm"):
            area = bid_data["built_up_area_sqm"]
            rate_per_sqm = grand_total / area
            rate_per_sqft = rate_per_sqm / 10.764
            lines.append("### Rate Analysis\n\n")
            lines.append(f"- Rate per sqm: ₹{rate_per_sqm:,.2f}\n")
            lines.append(f"- Rate per sqft: ₹{rate_per_sqft:,.2f}\n\n")

        # Package-wise breakdown
        lines.append("## Package-wise Breakdown\n\n")
        lines.append("| Package | Amount (₹) | % of Works |\n")
        lines.append("|---------|------------|------------|\n")

        by_package = {}
        for item in bid_data.get("priced_boq", []):
            pkg = item.get("package", "miscellaneous")
            if pkg not in by_package:
                by_package[pkg] = 0
            by_package[pkg] += item.get("amount", 0)

        for pkg, amount in sorted(by_package.items(), key=lambda x: -x[1]):
            pct = (amount / works_total * 100) if works_total > 0 else 0
            lines.append(f"| {pkg.replace('_', ' ').title()} | {amount:,.2f} | {pct:.1f}% |\n")

        lines.append(f"| **Total Works** | **{works_total:,.2f}** | **100%** |\n\n")

        # Key assumptions
        lines.append("## Key Assumptions\n\n")
        assumptions = [
            f"Location factor: {bid_data.get('location_factor', 1.0):.2f} (base: Delhi)",
            f"Finish grade: {bid_data.get('finish_grade', 'standard').title()}",
            "All rates based on CPWD DSR 2024 with market adjustments",
            "Labor rates as per ISR 2024",
            "GST @ 18% included in grand total",
            "Prices valid for 90 days from submission",
        ]
        for assumption in assumptions:
            lines.append(f"- {assumption}\n")
        lines.append("\n")

        # Risk items
        lines.append("## Risk Items / Allowances\n\n")

        allowances = bid_data.get("allowances", [])
        if allowances:
            lines.append("| Category | Allowance | Reason |\n")
            lines.append("|----------|-----------|--------|\n")
            for allow in allowances[:5]:
                lines.append(f"| {allow.get('category', '').title()} | {allow.get('allowance_percent', 0)}% | {allow.get('reason', '')[:50]} |\n")
            lines.append("\n")
        else:
            lines.append("No specific allowances required.\n\n")

        # Pending items
        lines.append("## Pending Items\n\n")

        rfis_count = bid_data.get("total_rfis", 0)
        missing_rates = bid_data.get("missing_rates", 0)

        if rfis_count > 0 or missing_rates > 0:
            lines.append(f"- RFIs to be raised: {rfis_count}\n")
            lines.append(f"- Items with missing rates: {missing_rates}\n")
            lines.append("\nRefer to `rfis_to_raise.md` for details.\n\n")
        else:
            lines.append("All items complete. No pending clarifications.\n\n")

        # Compliance
        lines.append("## Compliance Statement\n\n")
        lines.append("This bid is submitted in compliance with the tender requirements. ")
        lines.append("All scope items as per tender drawings and specifications have been covered. ")
        lines.append("Any deviations or clarifications are documented in `clarifications.md`.\n\n")

        # Validity
        lines.append("## Validity\n\n")
        lines.append("This bid is valid for **90 days** from the date of submission.\n\n")

        # Contact
        lines.append("---\n\n")
        lines.append("*This bid summary is auto-generated by the XBOQ Bid Engine.*\n")

        return "".join(lines)
