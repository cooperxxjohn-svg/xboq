"""
Package Splitter - Split BOQ into work packages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import csv


@dataclass
class PackageItem:
    """Single item in a package."""
    item_no: str
    description: str
    unit: str
    quantity: float
    rate: float
    amount: float
    spec_notes: str = ""
    drawing_refs: str = ""
    assumptions: str = ""

    def to_dict(self) -> dict:
        return {
            "item_no": self.item_no,
            "description": self.description,
            "unit": self.unit,
            "quantity": self.quantity,
            "rate": self.rate,
            "amount": self.amount,
            "spec_notes": self.spec_notes,
            "drawing_refs": self.drawing_refs,
            "assumptions": self.assumptions,
        }


@dataclass
class Package:
    """Work package."""
    code: str
    name: str
    description: str
    items: List[PackageItem] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    scope_notes: List[str] = field(default_factory=list)
    inclusions: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)

    @property
    def items_count(self) -> int:
        return len(self.items)

    @property
    def total_value(self) -> float:
        return sum(item.amount for item in self.items)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "items_count": self.items_count,
            "total_value": self.total_value,
            "risks": self.risks,
            "scope_notes": self.scope_notes,
            "inclusions": self.inclusions,
            "exclusions": self.exclusions,
        }


class PackageSplitter:
    """Split BOQ into work packages."""

    # Package definitions
    PACKAGES = {
        "RCC": {
            "name": "RCC / Structure",
            "description": "Reinforced cement concrete works including formwork and reinforcement",
            "keywords": ["rcc", "reinforced", "concrete", "footing", "column", "beam", "slab", "foundation", "lintel", "chajja", "staircase", "parapet"],
            "default_risks": [
                "Structural drawing changes may impact quantities",
                "Foundation depth subject to actual soil conditions",
                "Steel prices subject to market fluctuation",
            ],
            "default_inclusions": [
                "All RCC work as per structural drawings",
                "Shuttering/formwork for all concrete work",
                "Steel reinforcement as per structural drawings",
                "Curing for required period",
            ],
            "default_exclusions": [
                "Pile foundation (if required)",
                "Rock excavation premium",
                "Dewatering (if water table encountered)",
            ],
        },
        "MASONRY": {
            "name": "Masonry / Plaster",
            "description": "Brick/block masonry and plastering works",
            "keywords": ["brick", "block", "aac", "masonry", "wall", "plaster", "plastering", "pointing", "mortar"],
            "default_risks": [
                "Wall locations may change in GFC drawings",
                "Opening sizes subject to door/window schedule",
            ],
            "default_inclusions": [
                "All masonry work as per architectural drawings",
                "Internal and external plastering",
                "Pointing work where specified",
            ],
            "default_exclusions": [
                "Stone cladding",
                "Decorative masonry features",
            ],
        },
        "WATERPROOF": {
            "name": "Waterproofing",
            "description": "Waterproofing treatment for toilets, terrace, tanks, and basement",
            "keywords": ["waterproof", "wp", "membrane", "coba", "treatment", "damp", "proof"],
            "default_risks": [
                "Actual toilet/wet area locations to be confirmed",
                "Terrace area subject to final roof drawing",
                "Product/system approval required before execution",
            ],
            "default_inclusions": [
                "Toilet waterproofing with turn-up",
                "Terrace waterproofing with protection",
                "Water tank waterproofing",
            ],
            "default_exclusions": [
                "Basement waterproofing (unless specified)",
                "Swimming pool waterproofing",
            ],
        },
        "FLOORING": {
            "name": "Flooring / Finishes",
            "description": "Floor and wall finishes including tiles, marble, granite, paint",
            "keywords": ["flooring", "floor", "tile", "vitrified", "ceramic", "marble", "granite", "kota", "wooden", "paint", "painting", "polish", "putty", "finish"],
            "default_risks": [
                "Material selection pending owner approval",
                "Pattern layout to be finalized",
                "Wastage dependent on tile size and pattern",
            ],
            "default_inclusions": [
                "Floor finishes as per schedule",
                "Wall dado/tiles as per schedule",
                "Skirting where specified",
                "Internal and external painting",
            ],
            "default_exclusions": [
                "Imported materials",
                "Custom patterns/designs",
                "Furniture polish",
            ],
        },
        "DOORS_WINDOWS": {
            "name": "Doors / Windows",
            "description": "Doors, windows, ventilators, and associated hardware",
            "keywords": ["door", "window", "shutter", "frame", "ventilator", "grill", "hardware", "lock", "handle"],
            "default_risks": [
                "Door/window schedule to be confirmed",
                "Hardware selection pending approval",
                "MS grill design to be finalized",
            ],
            "default_inclusions": [
                "All doors as per schedule",
                "All windows as per schedule",
                "Door/window frames and shutters",
                "Standard hardware set",
            ],
            "default_exclusions": [
                "Automatic doors",
                "Fire rated doors (unless specified)",
                "Designer/imported hardware",
            ],
        },
        "PLUMBING": {
            "name": "Plumbing",
            "description": "Internal plumbing, sanitary fittings, and drainage",
            "keywords": ["plumb", "pipe", "drainage", "swr", "cpvc", "pvc", "sanitary", "wc", "basin", "tap", "fitting", "cock", "mixer"],
            "default_risks": [
                "MEP drawings not provided - based on norms",
                "Sanitary ware selection pending approval",
                "CP fitting brand/model to be confirmed",
            ],
            "default_inclusions": [
                "Internal water supply piping",
                "Internal drainage piping",
                "Sanitary fixtures as per schedule",
                "CP fittings as per schedule",
            ],
            "default_exclusions": [
                "External water supply from main",
                "External drainage to municipal drain",
                "Septic tank/STP (unless specified)",
                "Solar water heating",
            ],
        },
        "ELECTRICAL": {
            "name": "Electrical",
            "description": "Internal electrical wiring, switches, and fixtures",
            "keywords": ["electric", "wiring", "wire", "cable", "conduit", "switch", "socket", "point", "mcb", "db", "earthing", "light", "fan"],
            "default_risks": [
                "MEP drawings not provided - based on norms",
                "Point locations to be confirmed",
                "Switch/socket brand pending approval",
            ],
            "default_inclusions": [
                "Internal wiring in conduit",
                "Switch boards and distribution boards",
                "Earthing system",
                "Points as per schedule",
            ],
            "default_exclusions": [
                "External electrical connection from EB",
                "Generator/DG set",
                "Solar panels",
                "Home automation",
            ],
        },
        "EXTERNAL": {
            "name": "External Works",
            "description": "Compound wall, gate, paving, landscaping, and external services",
            "keywords": ["compound", "boundary", "gate", "paving", "paver", "landscap", "garden", "external", "site"],
            "default_risks": [
                "External works scope to be confirmed",
                "Site boundary to be verified",
                "Levels subject to site survey",
            ],
            "default_inclusions": [
                "Compound wall as per drawing",
                "Main gate",
                "Internal roads/paving",
            ],
            "default_exclusions": [
                "Landscaping and plantation",
                "External lighting",
                "Swimming pool",
                "Outdoor furniture",
            ],
        },
        "PRELIMS": {
            "name": "Preliminaries",
            "description": "Site establishment, supervision, insurance, and general conditions",
            "keywords": ["prelim", "supervision", "insurance", "scaffold", "temporary", "mobilization"],
            "default_risks": [
                "Duration subject to actual progress",
                "Site conditions may impact costs",
            ],
            "default_inclusions": [
                "Site office and facilities",
                "Site supervision",
                "Insurance (CAR, WC, TPL)",
                "Temporary utilities",
            ],
            "default_exclusions": [
                "Extended duration due to owner delays",
                "Third party testing",
            ],
        },
    }

    def __init__(self):
        pass

    def split(
        self,
        priced_boq: List[Dict],
        prelims_items: List,
        bid_data: dict,
    ) -> List[Package]:
        """Split BOQ into packages."""
        packages = {}

        # Initialize packages
        for code, pkg_def in self.PACKAGES.items():
            packages[code] = Package(
                code=code,
                name=pkg_def["name"],
                description=pkg_def["description"],
                risks=pkg_def["default_risks"].copy(),
                inclusions=pkg_def["default_inclusions"].copy(),
                exclusions=pkg_def["default_exclusions"].copy(),
            )

        # Assign BOQ items to packages
        for item in priced_boq:
            pkg_code = self._classify_item(item)
            if pkg_code in packages:
                packages[pkg_code].items.append(PackageItem(
                    item_no=item.get("unified_item_no", item.get("item_no", "")),
                    description=item.get("description", ""),
                    unit=item.get("unit", ""),
                    quantity=float(item.get("quantity", 0)),
                    rate=float(item.get("rate", 0)),
                    amount=float(item.get("amount", 0)),
                    spec_notes=item.get("spec_notes", ""),
                    drawing_refs=item.get("drawing_refs", ""),
                    assumptions=item.get("assumptions", ""),
                ))

        # Add prelims items to PRELIMS package
        for item in prelims_items:
            if hasattr(item, "description"):
                packages["PRELIMS"].items.append(PackageItem(
                    item_no=f"P-{len(packages['PRELIMS'].items) + 1:03d}",
                    description=item.description,
                    unit=item.unit,
                    quantity=item.quantity,
                    rate=item.rate,
                    amount=item.amount,
                    spec_notes="",
                    drawing_refs="",
                    assumptions=item.basis if hasattr(item, "basis") else "",
                ))
            elif isinstance(item, dict):
                packages["PRELIMS"].items.append(PackageItem(
                    item_no=f"P-{len(packages['PRELIMS'].items) + 1:03d}",
                    description=item.get("description", ""),
                    unit=item.get("unit", ""),
                    quantity=float(item.get("quantity", 0)),
                    rate=float(item.get("rate", 0)),
                    amount=float(item.get("amount", 0)),
                    spec_notes="",
                    drawing_refs="",
                    assumptions=item.get("basis", ""),
                ))

        # Add project-specific risks from bid data
        self._add_project_risks(packages, bid_data)

        # Filter out empty packages
        return [pkg for pkg in packages.values() if pkg.items_count > 0]

    def _classify_item(self, item: Dict) -> str:
        """Classify item into a package."""
        # First check if package is already assigned
        existing_pkg = item.get("package", "").upper()
        if existing_pkg:
            # Map common package names
            pkg_map = {
                "CIVIL_STRUCTURAL": "RCC",
                "CIVIL": "RCC",
                "STRUCTURAL": "RCC",
                "MASONRY_PLASTER": "MASONRY",
                "PLASTER_FINISHES": "MASONRY",
                "FLOORING_FINISHES": "FLOORING",
                "FINISHES": "FLOORING",
                "DOORS_WINDOWS": "DOORS_WINDOWS",
                "WATERPROOFING": "WATERPROOF",
                "PLUMBING": "PLUMBING",
                "ELECTRICAL": "ELECTRICAL",
                "EXTERNAL_WORKS": "EXTERNAL",
                "MISCELLANEOUS": "PRELIMS",
            }
            mapped = pkg_map.get(existing_pkg, existing_pkg)
            if mapped in self.PACKAGES:
                return mapped

        # Classify by description keywords
        desc = item.get("description", "").lower()

        for code, pkg_def in self.PACKAGES.items():
            for keyword in pkg_def["keywords"]:
                if keyword in desc:
                    return code

        # Default to PRELIMS for unclassified items
        return "PRELIMS"

    def _add_project_risks(self, packages: Dict[str, Package], bid_data: dict) -> None:
        """Add project-specific risks from bid data."""
        # Add risks from missing drawings
        if not bid_data.get("structural_drawings_provided", True):
            packages["RCC"].risks.append("CRITICAL: Structural drawings not provided - quantities based on assumptions")

        if not bid_data.get("mep_drawings_provided", True):
            packages["PLUMBING"].risks.append("MEP drawings not provided - quantities based on standard norms")
            packages["ELECTRICAL"].risks.append("MEP drawings not provided - quantities based on standard norms")

        # Add risks from missing owner inputs
        missing_inputs = bid_data.get("missing_mandatory", [])
        for field in missing_inputs[:5]:
            if isinstance(field, dict):
                category = field.get("path", "").split(".")[0]
                if category == "finishes":
                    packages["FLOORING"].risks.append(f"Owner input missing: {field.get('field_name', '')}")
                elif category == "doors":
                    packages["DOORS_WINDOWS"].risks.append(f"Owner input missing: {field.get('field_name', '')}")
                elif category == "sanitary" or category == "plumbing":
                    packages["PLUMBING"].risks.append(f"Owner input missing: {field.get('field_name', '')}")
                elif category == "electrical":
                    packages["ELECTRICAL"].risks.append(f"Owner input missing: {field.get('field_name', '')}")

    def export_package_boq(self, package: Package, output_path: Path) -> None:
        """Export package BOQ to CSV."""
        with open(output_path, "w", newline="") as f:
            fieldnames = ["item_no", "description", "unit", "quantity", "rate", "amount"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in package.items:
                writer.writerow({
                    "item_no": item.item_no,
                    "description": item.description,
                    "unit": item.unit,
                    "quantity": round(item.quantity, 2),
                    "rate": round(item.rate, 2),
                    "amount": round(item.amount, 2),
                })

            # Total row
            writer.writerow({
                "item_no": "",
                "description": "PACKAGE TOTAL",
                "unit": "",
                "quantity": "",
                "rate": "",
                "amount": round(package.total_value, 2),
            })

    def export_package_scope(self, package: Package, output_path: Path, bid_data: dict) -> None:
        """Export package scope document."""
        with open(output_path, "w") as f:
            f.write(f"# {package.name} - Scope of Work\n\n")
            f.write(f"**Package Code**: {package.code}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%d-%b-%Y')}\n\n")

            f.write("## Description\n\n")
            f.write(f"{package.description}\n\n")

            f.write("## Inclusions\n\n")
            for inc in package.inclusions:
                f.write(f"- {inc}\n")
            f.write("\n")

            f.write("## Exclusions\n\n")
            for exc in package.exclusions:
                f.write(f"- {exc}\n")
            f.write("\n")

            f.write("## Scope Notes\n\n")
            if package.scope_notes:
                for note in package.scope_notes:
                    f.write(f"- {note}\n")
            else:
                f.write("- All work as per tender drawings and specifications\n")
                f.write("- Material approvals required before procurement\n")
                f.write("- Quality as per relevant IS codes\n")
            f.write("\n")

            f.write("## Summary\n\n")
            f.write(f"- **Items**: {package.items_count}\n")
            f.write(f"- **Value**: ₹{package.total_value:,.2f}\n")

    def export_package_risks(self, package: Package, output_path: Path) -> None:
        """Export package risks document."""
        with open(output_path, "w") as f:
            f.write(f"# {package.name} - Risk Register\n\n")
            f.write(f"**Package Code**: {package.code}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%d-%b-%Y')}\n\n")

            f.write("## Identified Risks\n\n")

            if package.risks:
                for i, risk in enumerate(package.risks, 1):
                    severity = "HIGH" if "CRITICAL" in risk or "not provided" in risk else "MEDIUM"
                    icon = "🔴" if severity == "HIGH" else "🟠"
                    f.write(f"{i}. {icon} **{severity}**: {risk}\n\n")
            else:
                f.write("No significant risks identified for this package.\n\n")

            f.write("## Risk Mitigation\n\n")
            f.write("- Verify quantities before material ordering\n")
            f.write("- Obtain material approvals before procurement\n")
            f.write("- Raise RFIs for any ambiguities\n")
            f.write("- Document all site variations\n")

    def export_rfq_sheet(self, package: Package, output_path: Path, bid_data: dict) -> None:
        """Export RFQ sheet for subcontractor quotes."""
        with open(output_path, "w", newline="") as f:
            fieldnames = [
                "sl_no",
                "item_description",
                "quantity",
                "unit",
                "specification_notes",
                "drawing_references",
                "assumptions",
                "subcontractor_rate",
                "subcontractor_amount",
                "remarks",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header row with instructions
            f.write(f"# RFQ Sheet - {package.name}\n")
            f.write(f"# Project: {bid_data.get('project_id', 'N/A')}\n")
            f.write(f"# Date: {datetime.now().strftime('%d-%b-%Y')}\n")
            f.write(f"# Instructions: Fill 'subcontractor_rate' column and return\n")
            f.write(f"# \n")

            writer.writeheader()

            for i, item in enumerate(package.items, 1):
                # Get drawing references if available
                drawing_refs = item.drawing_refs
                if not drawing_refs:
                    drawing_refs = bid_data.get("drawing_refs", {}).get(item.item_no, "As per tender drawings")

                writer.writerow({
                    "sl_no": i,
                    "item_description": item.description,
                    "quantity": round(item.quantity, 2),
                    "unit": item.unit,
                    "specification_notes": item.spec_notes or "As per tender specs",
                    "drawing_references": drawing_refs,
                    "assumptions": item.assumptions or "",
                    "subcontractor_rate": "",  # To be filled by subcontractor
                    "subcontractor_amount": "",  # To be filled by subcontractor
                    "remarks": "",
                })

            # Summary rows
            writer.writerow({})
            writer.writerow({
                "sl_no": "",
                "item_description": "PACKAGE TOTAL",
                "quantity": "",
                "unit": "",
                "specification_notes": "",
                "drawing_references": "",
                "assumptions": "",
                "subcontractor_rate": "",
                "subcontractor_amount": "=SUM(I2:I{})".format(len(package.items) + 1),
                "remarks": "",
            })

    def generate_summary(self, packages: List[Package]) -> dict:
        """Generate summary of all packages."""
        total_value = sum(p.total_value for p in packages)

        return {
            "generated_at": datetime.now().isoformat(),
            "total_packages": len(packages),
            "total_items": sum(p.items_count for p in packages),
            "total_value": round(total_value, 2),
            "packages": [
                {
                    "code": p.code,
                    "name": p.name,
                    "items": p.items_count,
                    "value": round(p.total_value, 2),
                    "percentage": round(p.total_value / total_value * 100, 1) if total_value > 0 else 0,
                    "risks_count": len(p.risks),
                }
                for p in packages
            ],
        }

    def generate_index(self, packages: List[Package], project_id: str) -> str:
        """Generate package index markdown."""
        lines = []
        lines.append(f"# Package Index: {project_id}\n\n")
        lines.append(f"Generated: {datetime.now().strftime('%d-%b-%Y')}\n\n")

        total_value = sum(p.total_value for p in packages)

        lines.append("## Summary\n\n")
        lines.append(f"- **Total Packages**: {len(packages)}\n")
        lines.append(f"- **Total Items**: {sum(p.items_count for p in packages)}\n")
        lines.append(f"- **Total Value**: ₹{total_value:,.2f}\n\n")

        lines.append("## Packages\n\n")
        lines.append("| Package | Items | Value (₹) | % | Risks |\n")
        lines.append("|---------|-------|-----------|---|-------|\n")

        for p in sorted(packages, key=lambda x: -x.total_value):
            pct = (p.total_value / total_value * 100) if total_value > 0 else 0
            risk_count = len([r for r in p.risks if "CRITICAL" in r or "not provided" in r])
            risk_indicator = "🔴" if risk_count > 0 else "🟢"
            lines.append(f"| [{p.name}](./{p.code}/) | {p.items_count} | {p.total_value:,.0f} | {pct:.1f}% | {risk_indicator} {len(p.risks)} |\n")

        lines.append(f"| **TOTAL** | **{sum(p.items_count for p in packages)}** | **{total_value:,.0f}** | **100%** | |\n\n")

        lines.append("## Package Contents\n\n")
        lines.append("Each package folder contains:\n\n")
        lines.append("- `pkg_boq.csv` - Package Bill of Quantities\n")
        lines.append("- `pkg_scope.md` - Scope of work, inclusions, exclusions\n")
        lines.append("- `pkg_risks.md` - Risk register for the package\n")
        lines.append("- `rfq_sheet.csv` - RFQ sheet for subcontractor quotes\n\n")

        lines.append("## RFQ Instructions\n\n")
        lines.append("To request subcontractor quotes:\n\n")
        lines.append("1. Send the `rfq_sheet.csv` to potential subcontractors\n")
        lines.append("2. Subcontractor fills `subcontractor_rate` column\n")
        lines.append("3. Collect and compare quotes using Quote Leveling module\n")

        return "".join(lines)
