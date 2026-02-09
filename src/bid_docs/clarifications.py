"""
Clarifications Letter Generator - Indian contractor bid submission style.
"""

from datetime import datetime
from typing import Dict, List, Any


class ClarificationsGenerator:
    """Generate clarifications/exceptions letter for bid submission."""

    # Standard inclusions for Indian construction bids
    STANDARD_INCLUSIONS = [
        "All materials, labour, tools, and equipment required to complete the work as per tender drawings and specifications",
        "Temporary works including shuttering, scaffolding, and staging",
        "All necessary tests and quality control as per relevant IS codes",
        "Cleaning of work area and removal of debris",
        "Safety measures and PPE for workers as per statutory requirements",
        "Workmen compensation insurance",
        "Supervision by qualified engineers as per tender requirements",
    ]

    # Standard exclusions for Indian construction bids
    STANDARD_EXCLUSIONS = [
        "Work beyond the scope shown in tender drawings",
        "Any work not specifically mentioned in the Bill of Quantities",
        "Premium for night work unless specified in tender",
        "Price escalation beyond tender validity period",
        "Any statutory approvals, NOCs, or permission fees",
        "Third party testing charges unless specified",
        "Work in restricted/confined spaces requiring special permits",
    ]

    def __init__(self):
        pass

    def generate_letter(
        self,
        project_id: str,
        bid_data: dict,
        gate_result: dict,
    ) -> str:
        """Generate complete clarifications letter."""
        lines = []

        # Header
        lines.append(self._generate_header(project_id, bid_data))

        # Subject line
        lines.append(self._generate_subject(bid_data))

        # Opening
        lines.append(self._generate_opening(bid_data))

        # Inclusions section
        lines.append(self._generate_inclusions(bid_data))

        # Exclusions section
        lines.append(self._generate_exclusions(bid_data))

        # Assumptions section
        lines.append(self._generate_assumptions(bid_data))

        # Allowances and provisionals
        lines.append(self._generate_allowances(bid_data))

        # RFIs requiring response
        lines.append(self._generate_rfis(bid_data))

        # Conflicts and pricing notes
        lines.append(self._generate_conflicts(bid_data))

        # Reservations from gate
        if gate_result.get("reservations_count", 0) > 0:
            lines.append(self._generate_reservations(gate_result))

        # Closing
        lines.append(self._generate_closing(bid_data, gate_result))

        return "".join(lines)

    def _generate_header(self, project_id: str, bid_data: dict) -> str:
        """Generate letter header."""
        project_info = bid_data.get("project_info", {})
        project_name = project_info.get("name", project_id)

        return f"""# CLARIFICATIONS, ASSUMPTIONS & EXCEPTIONS

**Project**: {project_name}
**Tender Reference**: {bid_data.get('tender_reference', 'As per tender documents')}
**Date**: {datetime.now().strftime('%d-%b-%Y')}
**From**: [Contractor Name]
**To**: {bid_data.get('owner_name', '[Owner/Client Name]')}

---

"""

    def _generate_subject(self, bid_data: dict) -> str:
        """Generate subject line."""
        project_info = bid_data.get("project_info", {})
        return f"""**Subject**: Clarifications, Assumptions and Exceptions to our Bid for {project_info.get('name', 'the captioned project')}

---

"""

    def _generate_opening(self, bid_data: dict) -> str:
        """Generate opening paragraph."""
        return """Dear Sir/Madam,

With reference to the above tender, we hereby submit our bid along with the following clarifications, assumptions, and exceptions. This document forms an integral part of our bid submission and should be read in conjunction with our priced Bill of Quantities.

We request that any deviations noted herein be considered during bid evaluation and clarified prior to award of contract.

"""

    def _generate_inclusions(self, bid_data: dict) -> str:
        """Generate inclusions section."""
        lines = []
        lines.append("## 1. SCOPE OF WORK - INCLUSIONS\n\n")
        lines.append("Our bid **includes** the following:\n\n")

        # Standard inclusions
        lines.append("### 1.1 Standard Inclusions\n\n")
        for i, inc in enumerate(self.STANDARD_INCLUSIONS, 1):
            lines.append(f"{i}. {inc}\n")
        lines.append("\n")

        # Project-specific inclusions from tender/drawings
        detected_inclusions = bid_data.get("inclusions", [])
        if detected_inclusions:
            lines.append("### 1.2 Scope as per Tender Drawings\n\n")
            for i, inc in enumerate(detected_inclusions, 1):
                lines.append(f"{i}. {inc}\n")
            lines.append("\n")

        # Package-wise scope summary
        packages = bid_data.get("packages_summary", {})
        if packages:
            lines.append("### 1.3 Package-wise Scope Summary\n\n")
            lines.append("| Package | Items | Value (₹) | Status |\n")
            lines.append("|---------|-------|-----------|--------|\n")
            for pkg, data in packages.items():
                status = data.get("status", "Included")
                lines.append(f"| {pkg.replace('_', ' ').title()} | {data.get('items', 0)} | {data.get('value', 0):,.0f} | {status} |\n")
            lines.append("\n")

        return "".join(lines)

    def _generate_exclusions(self, bid_data: dict) -> str:
        """Generate exclusions section."""
        lines = []
        lines.append("## 2. SCOPE EXCLUSIONS\n\n")
        lines.append("Our bid **excludes** the following unless specifically mentioned in BOQ:\n\n")

        # Standard exclusions
        lines.append("### 2.1 Standard Exclusions\n\n")
        for i, exc in enumerate(self.STANDARD_EXCLUSIONS, 1):
            lines.append(f"{i}. {exc}\n")
        lines.append("\n")

        # Project-specific exclusions
        detected_exclusions = bid_data.get("exclusions", [])
        if detected_exclusions:
            lines.append("### 2.2 Project-specific Exclusions\n\n")
            lines.append("The following items are **not included** in our bid:\n\n")
            for i, exc in enumerate(detected_exclusions, 1):
                if isinstance(exc, dict):
                    lines.append(f"{i}. **{exc.get('item', 'Item')}**: {exc.get('reason', 'Not in scope')}\n")
                else:
                    lines.append(f"{i}. {exc}\n")
            lines.append("\n")

        # Inferred exclusions (from missing drawings/scope)
        missing_scope = bid_data.get("missing_scope", [])
        if missing_scope:
            lines.append("### 2.3 Scope Not Covered Due to Missing Information\n\n")
            lines.append("The following items could not be priced due to insufficient information:\n\n")
            for i, item in enumerate(missing_scope, 1):
                lines.append(f"{i}. {item}\n")
            lines.append("\n")
            lines.append("*These items will be quoted separately upon receipt of complete information.*\n\n")

        return "".join(lines)

    def _generate_assumptions(self, bid_data: dict) -> str:
        """Generate assumptions section."""
        lines = []
        lines.append("## 3. ASSUMPTIONS\n\n")
        lines.append("Our bid is based on the following assumptions:\n\n")

        assumptions = bid_data.get("assumptions", [])
        applied_defaults = bid_data.get("applied_defaults", [])

        # Technical assumptions
        lines.append("### 3.1 Technical Assumptions\n\n")

        technical_assumptions = [
            f"Ceiling height: {bid_data.get('ceiling_height_mm', 3000)} mm",
            f"Floor-to-floor height: {bid_data.get('floor_to_floor_mm', 3300)} mm",
            f"Slab thickness: {bid_data.get('slab_thickness_mm', 150)} mm (typical)",
            f"Wall thickness: 230mm external, 115mm internal (unless noted otherwise)",
            f"Plaster: 12mm internal, 20mm external",
            "Clear cover for RCC as per IS 456:2000",
        ]
        for i, assumption in enumerate(technical_assumptions, 1):
            lines.append(f"{i}. {assumption}\n")
        lines.append("\n")

        # Finish assumptions
        if bid_data.get("finish_grade"):
            lines.append("### 3.2 Finish Grade Assumptions\n\n")
            lines.append(f"Finish grade assumed: **{bid_data.get('finish_grade', 'Standard').title()}**\n\n")
            lines.append("Finish specifications based on:\n")
            lines.append("- Owner inputs where provided\n")
            lines.append("- Standard specifications for assumed grade where not provided\n")
            lines.append("- CPWD specifications as reference\n\n")

        # Applied defaults
        if applied_defaults:
            lines.append("### 3.3 Defaults Applied for Missing Owner Inputs\n\n")
            lines.append("| Field | Default Value | Basis |\n")
            lines.append("|-------|---------------|-------|\n")
            for default in applied_defaults[:15]:  # Limit to 15
                if isinstance(default, dict):
                    lines.append(f"| {default.get('field_path', '')} | {default.get('value', '')} | {default.get('basis', '')} |\n")
            if len(applied_defaults) > 15:
                lines.append(f"| ... | *{len(applied_defaults) - 15} more defaults* | See detailed list |\n")
            lines.append("\n")

        # Rate assumptions
        lines.append("### 3.4 Rate Assumptions\n\n")
        rate_assumptions = [
            f"Base rates as per CPWD DSR 2024 with market adjustments",
            f"Location factor applied: {bid_data.get('location_factor', 1.0):.2f}",
            f"Labour rates as per ISR 2024",
            f"Material rates as per current market (validity 90 days)",
            "Overheads and profit included in rates",
        ]
        for i, assumption in enumerate(rate_assumptions, 1):
            lines.append(f"{i}. {assumption}\n")
        lines.append("\n")

        # Other assumptions
        if assumptions:
            lines.append("### 3.5 Other Assumptions\n\n")
            for i, assumption in enumerate(assumptions, 1):
                lines.append(f"{i}. {assumption}\n")
            lines.append("\n")

        return "".join(lines)

    def _generate_allowances(self, bid_data: dict) -> str:
        """Generate allowances and provisional items section."""
        lines = []
        lines.append("## 4. ALLOWANCES AND PROVISIONAL ITEMS\n\n")

        allowances = bid_data.get("allowances", [])
        provisional_items = bid_data.get("provisional_items", [])

        if allowances:
            lines.append("### 4.1 Allowances for Uncertainty\n\n")
            lines.append("The following allowances have been added to account for scope uncertainty:\n\n")
            lines.append("| Category | Allowance % | Reason |\n")
            lines.append("|----------|-------------|--------|\n")
            for allow in allowances:
                if isinstance(allow, dict):
                    lines.append(f"| {allow.get('category', '').replace('_', ' ').title()} | {allow.get('allowance_percent', 0)}% | {allow.get('reason', '')} |\n")
            lines.append("\n")
            lines.append("*Allowances will be adjusted based on actual site conditions and final specifications.*\n\n")

        if provisional_items:
            lines.append("### 4.2 Provisional Items\n\n")
            lines.append("The following items are included as provisional sums:\n\n")
            lines.append("| Item | Provisional Amount (₹) | Remarks |\n")
            lines.append("|------|------------------------|--------|\n")
            for item in provisional_items:
                if isinstance(item, dict):
                    lines.append(f"| {item.get('description', '')} | {item.get('amount', 0):,.0f} | {item.get('remarks', '')} |\n")
            lines.append("\n")
            lines.append("*Provisional items to be finalized based on actual scope/specifications.*\n\n")

        if not allowances and not provisional_items:
            lines.append("No provisional items or special allowances included in this bid.\n\n")

        return "".join(lines)

    def _generate_rfis(self, bid_data: dict) -> str:
        """Generate RFIs section."""
        lines = []
        lines.append("## 5. CLARIFICATIONS REQUIRED (RFIs)\n\n")
        lines.append("We request clarification on the following items before contract finalization:\n\n")

        # Collect RFIs from various sources
        all_rfis = []

        # High priority RFIs
        for rfi in bid_data.get("doubt_rfis", []):
            if isinstance(rfi, dict) and rfi.get("priority") == "high":
                all_rfis.append(rfi)

        for rfi in bid_data.get("owner_input_rfis", []):
            if isinstance(rfi, dict) and rfi.get("priority") == "high":
                all_rfis.append(rfi)

        for rfi in bid_data.get("scope_rfis", []):
            if isinstance(rfi, dict):
                all_rfis.append(rfi)

        # Limit to top 10
        top_rfis = all_rfis[:10]

        if top_rfis:
            lines.append("### 5.1 High Priority Clarifications\n\n")
            for i, rfi in enumerate(top_rfis, 1):
                rfi_id = rfi.get("rfi_id", f"RFI-{i:03d}")
                lines.append(f"**{rfi_id}**: {rfi.get('question', rfi.get('description', 'Clarification required'))}\n\n")
                if rfi.get("impact"):
                    lines.append(f"- *Impact if not clarified*: {rfi.get('impact')}\n")
                if rfi.get("default_assumption"):
                    lines.append(f"- *Current assumption*: {rfi.get('default_assumption')}\n")
                lines.append("\n")

            if len(all_rfis) > 10:
                lines.append(f"*...and {len(all_rfis) - 10} additional RFIs. See detailed RFI list.*\n\n")
        else:
            lines.append("No critical clarifications pending at this time.\n\n")

        return "".join(lines)

    def _generate_conflicts(self, bid_data: dict) -> str:
        """Generate conflicts and resolution section."""
        lines = []
        lines.append("## 6. CONFLICTS DETECTED AND RESOLUTION\n\n")

        discrepancies = bid_data.get("discrepancies", [])
        conflicts = bid_data.get("conflicts", [])

        if discrepancies:
            lines.append("### 6.1 Quantity Discrepancies\n\n")
            lines.append("The following discrepancies were noted between drawings and tender BOQ:\n\n")
            lines.append("| Item | Drawing Qty | Tender Qty | Priced As | Basis |\n")
            lines.append("|------|-------------|------------|-----------|-------|\n")
            for disc in discrepancies[:10]:
                if isinstance(disc, dict):
                    priced_as = "Drawing" if disc.get("drawings_qty", 0) > disc.get("owner_qty", 0) else "Tender"
                    lines.append(f"| {disc.get('description', '')[:30]}... | {disc.get('drawings_qty', 0):.1f} | {disc.get('owner_qty', 0):.1f} | {priced_as} | Conservative |\n")
            lines.append("\n")
            lines.append("*Quantities priced as per more conservative estimate. Actual to be verified at site.*\n\n")

        if conflicts:
            lines.append("### 6.2 Specification Conflicts\n\n")
            for i, conflict in enumerate(conflicts, 1):
                if isinstance(conflict, dict):
                    lines.append(f"{i}. **{conflict.get('item', 'Item')}**: {conflict.get('conflict', 'Conflict noted')}\n")
                    lines.append(f"   - *Resolution*: {conflict.get('resolution', 'As per tender drawings')}\n\n")

        if not discrepancies and not conflicts:
            lines.append("No significant conflicts detected between tender documents.\n\n")

        return "".join(lines)

    def _generate_reservations(self, gate_result: dict) -> str:
        """Generate reservations section from gate result."""
        lines = []
        lines.append("## 7. RESERVATIONS\n\n")
        lines.append("The following reservations apply to this bid:\n\n")

        reservations = gate_result.get("reservations", [])
        if isinstance(reservations, list):
            for i, res in enumerate(reservations, 1):
                if isinstance(res, dict):
                    lines.append(f"**{res.get('code', f'RES-{i:03d}')}**: {res.get('description', '')}\n\n")
                    lines.append(f"- Impact: {res.get('impact', '')}\n")
                    lines.append(f"- Recommendation: {res.get('recommendation', '')}\n\n")

        return "".join(lines)

    def _generate_closing(self, bid_data: dict, gate_result: dict) -> str:
        """Generate closing section."""
        lines = []
        lines.append("## 8. GENERAL CONDITIONS\n\n")

        lines.append("### 8.1 Validity\n\n")
        lines.append("This bid is valid for **90 days** from the date of submission.\n\n")

        lines.append("### 8.2 Payment Terms\n\n")
        lines.append("As per tender conditions, or:\n")
        lines.append("- Mobilization advance: As per tender\n")
        lines.append("- Running bills: Monthly, within 30 days of certification\n")
        lines.append("- Retention: As per tender (standard 5%)\n\n")

        lines.append("### 8.3 Taxes\n\n")
        lines.append("- GST @ 18% applicable on all items\n")
        lines.append("- TDS deductions as applicable\n")
        lines.append("- All other taxes as per prevailing laws\n\n")

        lines.append("### 8.4 Force Majeure\n\n")
        lines.append("Standard force majeure clause as per tender conditions shall apply.\n\n")

        # Status-specific closing
        gate_status = gate_result.get("status", "")
        if gate_status == "FAIL":
            lines.append("---\n\n")
            lines.append("## ⚠️ IMPORTANT NOTICE\n\n")
            lines.append("**This bid is submitted subject to resolution of the critical issues noted above.**\n\n")
            lines.append("We recommend that these clarifications be addressed before contract award.\n\n")
            lines.append("The bid amount may be subject to revision upon receipt of clarifications.\n\n")

        lines.append("---\n\n")
        lines.append("We trust the above clarifications are acceptable. We remain available for any further\n")
        lines.append("discussions or clarifications that may be required.\n\n")
        lines.append("Thanking you,\n\n")
        lines.append("Yours faithfully,\n\n")
        lines.append("**[Authorized Signatory]**  \n")
        lines.append("[Name]  \n")
        lines.append("[Designation]  \n")
        lines.append("[Company Name]  \n")
        lines.append(f"Date: {datetime.now().strftime('%d-%b-%Y')}\n")

        return "".join(lines)
