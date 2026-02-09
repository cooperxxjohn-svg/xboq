"""
Provisional Items Generation - Risk control for unknown/implied scope.

For UNKNOWN or IMPLIED critical packages, add provisional BOQ lines:
- item_code like "PS_WATERPROOFING_SYSTEM"
- qty marked as "TBD" or blank
- notes: "Provisional - confirm spec/sheet"

Outputs:
- out/<project_id>/boq/provisional_items.csv
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from .register import ScopeRegister, ScopeItem, ScopeStatus

logger = logging.getLogger(__name__)


@dataclass
class ProvisionalItem:
    """A provisional BOQ item for unclear scope."""
    item_code: str
    category: str
    description: str
    unit: str
    quantity: str  # "TBD", "LS", or estimated value
    rate: str  # "TBD" or estimated rate
    amount: str  # "TBD" or "PC" (prime cost)
    notes: str
    risk_level: str  # critical, high, medium, low
    source_package: str
    source_subpackage: str
    reason: str

    def to_csv_row(self) -> List[str]:
        return [
            self.item_code,
            self.category,
            self.description,
            self.unit,
            self.quantity,
            self.rate,
            self.amount,
            self.notes,
            self.risk_level,
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_code": self.item_code,
            "category": self.category,
            "description": self.description,
            "unit": self.unit,
            "quantity": self.quantity,
            "rate": self.rate,
            "amount": self.amount,
            "notes": self.notes,
            "risk_level": self.risk_level,
            "source_package": self.source_package,
            "source_subpackage": self.source_subpackage,
            "reason": self.reason,
        }


class ProvisionalItemsGenerator:
    """
    Generates provisional BOQ items for unclear scope.
    """

    # Critical packages that always need provisionals if unclear
    CRITICAL_PACKAGES = {
        "waterproofing": {
            "toilet_waterproofing": {
                "description": "Waterproofing to toilets/bathrooms (sunken area)",
                "unit": "sqm",
                "default_note": "APP/liquid membrane system - confirm specification",
            },
            "terrace_waterproofing": {
                "description": "Waterproofing to terrace/roof",
                "unit": "sqm",
                "default_note": "APP membrane with brick bat coba - confirm system",
            },
            "basement_waterproofing": {
                "description": "Waterproofing to basement/retaining walls",
                "unit": "sqm",
                "default_note": "External/crystalline system - confirm requirement",
            },
        },
        "fire_hvac": {
            "fire_detection": {
                "description": "Fire detection and alarm system",
                "unit": "LS",
                "default_note": "Complete system with smoke detectors, panel, hooters - confirm NBC requirement",
            },
            "fire_fighting": {
                "description": "Fire fighting system (extinguishers/hydrants)",
                "unit": "LS",
                "default_note": "As per local fire NOC requirements - confirm scope",
            },
            "hvac": {
                "description": "HVAC/Air conditioning provisions",
                "unit": "LS",
                "default_note": "Split AC provisions only / Central system - confirm scope",
            },
        },
        "plumbing_storm_rainwater": {
            "rainwater_harvesting": {
                "description": "Rainwater harvesting system",
                "unit": "LS",
                "default_note": "RWH pit with filter chamber - mandated by most municipal bodies",
            },
        },
        "external_works_drainage": {
            "external_drainage": {
                "description": "Septic tank / Soak pit / STP connection",
                "unit": "LS",
                "default_note": "Confirm sewage disposal arrangement",
            },
        },
        "finishes_ceiling": {
            "false_ceiling": {
                "description": "False ceiling (gypsum/POP/grid)",
                "unit": "sqm",
                "default_note": "No ceiling plan found - confirm if included",
            },
        },
    }

    # India-specific standard provisional rates (for reference only)
    REFERENCE_RATES = {
        "waterproofing_toilet": "Rs. 400-600/sqm",
        "waterproofing_terrace": "Rs. 350-500/sqm",
        "fire_system": "Rs. 150-300/sqft built-up",
        "rwh_system": "Rs. 50,000-150,000 lump sum",
        "false_ceiling_gypsum": "Rs. 85-120/sqft",
        "false_ceiling_pop": "Rs. 65-90/sqft",
    }

    def __init__(self):
        self._item_counter = 0

    def _generate_code(self, prefix: str) -> str:
        """Generate provisional item code."""
        self._item_counter += 1
        return f"PS_{prefix}_{self._item_counter:03d}"

    def generate(
        self,
        register: ScopeRegister,
    ) -> List[ProvisionalItem]:
        """
        Generate provisional items from scope register.

        Args:
            register: Scope register

        Returns:
            List of ProvisionalItem
        """
        provisionals = []

        for item in register.items:
            # Only generate for UNKNOWN or IMPLIED status
            if item.status not in [ScopeStatus.UNKNOWN, ScopeStatus.IMPLIED, ScopeStatus.MISSING_INPUT]:
                continue

            # Check if this is a critical package
            if item.package in self.CRITICAL_PACKAGES:
                pkg_config = self.CRITICAL_PACKAGES[item.package]
                if item.subpackage in pkg_config:
                    config = pkg_config[item.subpackage]
                    prov = self._create_provisional(item, config)
                    provisionals.append(prov)
                    continue

            # Generate generic provisional for other UNKNOWN items in critical packages
            if item.status == ScopeStatus.UNKNOWN and item.package in self.CRITICAL_PACKAGES:
                prov = self._create_generic_provisional(item)
                provisionals.append(prov)

        # Add standard provisionals that are often missing
        provisionals.extend(self._add_standard_provisionals(register))

        return provisionals

    def _create_provisional(
        self,
        item: ScopeItem,
        config: Dict[str, str]
    ) -> ProvisionalItem:
        """Create provisional item from config."""
        risk_level = "critical" if item.package in ["waterproofing", "fire_hvac"] else "high"

        return ProvisionalItem(
            item_code=self._generate_code(item.subpackage.upper()[:10]),
            category=f"PROVISIONAL - {item.package_name}",
            description=config["description"],
            unit=config["unit"],
            quantity="TBD",
            rate="TBD",
            amount="PC",  # Prime Cost
            notes=f"PROVISIONAL: {config['default_note']}",
            risk_level=risk_level,
            source_package=item.package,
            source_subpackage=item.subpackage,
            reason=f"Status: {item.status.value}. {'; '.join(item.missing_info[:2])}",
        )

    def _create_generic_provisional(self, item: ScopeItem) -> ProvisionalItem:
        """Create generic provisional for unknown scope."""
        return ProvisionalItem(
            item_code=self._generate_code(item.subpackage.upper()[:10]),
            category=f"PROVISIONAL - {item.package_name}",
            description=f"{item.subpackage_name} (scope unconfirmed)",
            unit="LS",
            quantity="TBD",
            rate="TBD",
            amount="PC",
            notes=f"PROVISIONAL: No evidence found - confirm scope and specification",
            risk_level="high",
            source_package=item.package,
            source_subpackage=item.subpackage,
            reason=f"Status: {item.status.value}. No evidence in drawings.",
        )

    def _add_standard_provisionals(
        self,
        register: ScopeRegister
    ) -> List[ProvisionalItem]:
        """Add standard provisionals that are commonly missed."""
        standard = []

        # Check for testing & commissioning
        testing_items = [i for i in register.items if i.package == "testing_commissioning"]
        has_testing = any(i.status == ScopeStatus.DETECTED for i in testing_items)

        if not has_testing:
            standard.append(ProvisionalItem(
                item_code=self._generate_code("TESTING"),
                category="PROVISIONAL - Testing",
                description="Material testing (cube, slump, rebar)",
                unit="LS",
                quantity="TBD",
                rate="TBD",
                amount="PC",
                notes="PROVISIONAL: Typically 0.5-1% of RCC cost - confirm testing requirements",
                risk_level="medium",
                source_package="testing_commissioning",
                source_subpackage="material_testing",
                reason="Testing scope not explicitly defined",
            ))

        # Check for scaffolding
        scaffolding_items = [i for i in register.items if i.subpackage == "scaffolding"]
        has_scaffolding = any(i.status == ScopeStatus.DETECTED for i in scaffolding_items)

        if not has_scaffolding:
            standard.append(ProvisionalItem(
                item_code=self._generate_code("SCAFFOLD"),
                category="PROVISIONAL - Preliminaries",
                description="Scaffolding and safety provisions",
                unit="LS",
                quantity="TBD",
                rate="TBD",
                amount="PC",
                notes="PROVISIONAL: Typically 2-3% of civil cost - confirm if included in rates",
                risk_level="medium",
                source_package="prelims_general_conditions",
                source_subpackage="scaffolding",
                reason="Scaffolding scope not explicitly defined",
            ))

        # Check for anti-termite
        termite_items = [i for i in register.items if i.subpackage == "anti_termite"]
        has_termite = any(i.status == ScopeStatus.DETECTED for i in termite_items)

        if not has_termite:
            standard.append(ProvisionalItem(
                item_code=self._generate_code("TERMITE"),
                category="PROVISIONAL - Earthwork",
                description="Anti-termite treatment (pre + post construction)",
                unit="sqm",
                quantity="TBD",
                rate="TBD",
                amount="PC",
                notes="PROVISIONAL: As per IS 6313 - confirm if required by local norms",
                risk_level="medium",
                source_package="earthwork",
                source_subpackage="anti_termite",
                reason="Anti-termite treatment not mentioned in notes",
            ))

        return standard

    def export_csv(
        self,
        provisionals: List[ProvisionalItem],
        output_path: Path,
    ) -> None:
        """Export provisionals to CSV."""
        headers = [
            "Item Code", "Category", "Description", "Unit",
            "Quantity", "Rate", "Amount", "Notes", "Risk Level"
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in provisionals:
                writer.writerow(item.to_csv_row())

        logger.info(f"Saved {len(provisionals)} provisional items to: {output_path}")

    def export_json(
        self,
        provisionals: List[ProvisionalItem],
        output_path: Path,
    ) -> None:
        """Export provisionals to JSON."""
        data = {
            "generated": datetime.now().isoformat(),
            "total_items": len(provisionals),
            "by_risk": {
                "critical": len([p for p in provisionals if p.risk_level == "critical"]),
                "high": len([p for p in provisionals if p.risk_level == "high"]),
                "medium": len([p for p in provisionals if p.risk_level == "medium"]),
                "low": len([p for p in provisionals if p.risk_level == "low"]),
            },
            "items": [p.to_dict() for p in provisionals],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved provisional items JSON to: {output_path}")


def generate_provisionals(
    register: ScopeRegister,
    output_dir: Path,
) -> List[ProvisionalItem]:
    """
    Convenience function to generate and save provisional items.

    Args:
        register: Scope register
        output_dir: Output directory

    Returns:
        List of ProvisionalItem
    """
    generator = ProvisionalItemsGenerator()
    provisionals = generator.generate(register)

    # Save files
    boq_dir = output_dir / "boq"
    boq_dir.mkdir(parents=True, exist_ok=True)

    generator.export_csv(provisionals, boq_dir / "provisional_items.csv")
    generator.export_json(provisionals, boq_dir / "provisional_items.json")

    return provisionals
