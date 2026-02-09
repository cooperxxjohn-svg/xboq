"""
Prelims Calculator - Main preliminary cost calculation logic.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class PrelimsItem:
    """Single preliminary cost item."""
    description: str
    unit: str
    quantity: float
    rate: float
    amount: float
    category: str
    basis: str = ""  # Calculation basis

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "unit": self.unit,
            "quantity": self.quantity,
            "rate": self.rate,
            "amount": self.amount,
            "category": self.category,
            "basis": self.basis,
        }


class PrelimsCalculator:
    """Calculate preliminary costs for construction projects."""

    # Typical prelims percentages by project type
    TYPICAL_PRELIMS_PCT = {
        "residential": {"min": 6, "max": 10, "typical": 8},
        "commercial": {"min": 7, "max": 12, "typical": 9},
        "industrial": {"min": 5, "max": 8, "typical": 6},
        "institutional": {"min": 8, "max": 12, "typical": 10},
    }

    # Insurance rates (% of project value)
    INSURANCE_RATES = {
        "car": 0.12,  # Contractor All Risk
        "wc": 0.30,  # Workmen Compensation (% of labor cost, assume 30% of project)
        "tpl": 0.05,  # Third Party Liability
    }

    # Bond rates (% of contract value)
    BOND_RATES = {
        "performance": 0.50,  # 5% bond, 10% bank charge = 0.5%
        "advance": 0.25,  # For mobilization advance
    }

    def __init__(
        self,
        project_value: float,
        duration_months: int,
        built_up_area_sqm: float,
        project_type: str = "residential",
    ):
        self.project_value = project_value
        self.duration_months = duration_months
        self.built_up_area_sqm = built_up_area_sqm
        self.project_type = project_type.lower()

    def calculate_insurance_bonds(
        self,
        project_value: float,
        duration_months: int,
    ) -> List[PrelimsItem]:
        """Calculate insurance and bond costs."""
        items = []

        # CAR Insurance
        car_amount = project_value * (self.INSURANCE_RATES["car"] / 100)
        items.append(PrelimsItem(
            description="Contractor All Risk Insurance (CAR)",
            unit="LS",
            quantity=1,
            rate=car_amount,
            amount=car_amount,
            category="insurance_bonds",
            basis=f"{self.INSURANCE_RATES['car']}% of project value",
        ))

        # Workmen Compensation
        labor_cost_estimate = project_value * 0.30  # Assume 30% of project is labor
        wc_amount = labor_cost_estimate * (self.INSURANCE_RATES["wc"] / 100)
        items.append(PrelimsItem(
            description="Workmen Compensation Insurance",
            unit="LS",
            quantity=1,
            rate=wc_amount,
            amount=wc_amount,
            category="insurance_bonds",
            basis=f"{self.INSURANCE_RATES['wc']}% of estimated labor cost",
        ))

        # Third Party Liability
        tpl_amount = project_value * (self.INSURANCE_RATES["tpl"] / 100)
        items.append(PrelimsItem(
            description="Third Party Liability Insurance",
            unit="LS",
            quantity=1,
            rate=tpl_amount,
            amount=tpl_amount,
            category="insurance_bonds",
            basis=f"{self.INSURANCE_RATES['tpl']}% of project value",
        ))

        # Performance Bank Guarantee
        pbg_amount = project_value * (self.BOND_RATES["performance"] / 100)
        items.append(PrelimsItem(
            description="Performance Bank Guarantee Charges",
            unit="LS",
            quantity=1,
            rate=pbg_amount,
            amount=pbg_amount,
            category="insurance_bonds",
            basis=f"Bank charges @ {self.BOND_RATES['performance']}% of project value",
        ))

        return items

    def calculate_miscellaneous(
        self,
        project_value: float,
        duration_months: int,
    ) -> List[PrelimsItem]:
        """Calculate miscellaneous preliminary costs."""
        items = []

        # Mobilization and demobilization
        mob_amount = project_value * 0.005  # 0.5%
        items.append(PrelimsItem(
            description="Mobilization and Demobilization",
            unit="LS",
            quantity=1,
            rate=mob_amount,
            amount=mob_amount,
            category="miscellaneous",
            basis="0.5% of project value",
        ))

        # Setting out and survey
        survey_amount = min(project_value * 0.002, 200000)  # 0.2% or max 2L
        items.append(PrelimsItem(
            description="Setting Out and Survey",
            unit="LS",
            quantity=1,
            rate=survey_amount,
            amount=survey_amount,
            category="miscellaneous",
            basis="0.2% of project value (max ₹2L)",
        ))

        # As-built drawings
        asbuilt_amount = project_value * 0.001  # 0.1%
        items.append(PrelimsItem(
            description="As-built Drawings and Documentation",
            unit="LS",
            quantity=1,
            rate=asbuilt_amount,
            amount=asbuilt_amount,
            category="miscellaneous",
            basis="0.1% of project value",
        ))

        # Testing and quality control
        testing_amount = project_value * 0.003  # 0.3%
        items.append(PrelimsItem(
            description="Testing and Quality Control",
            unit="LS",
            quantity=1,
            rate=testing_amount,
            amount=testing_amount,
            category="miscellaneous",
            basis="0.3% of project value",
        ))

        # Permits and approvals
        permits_amount = project_value * 0.002  # 0.2%
        items.append(PrelimsItem(
            description="Permits, Approvals and Statutory Charges",
            unit="LS",
            quantity=1,
            rate=permits_amount,
            amount=permits_amount,
            category="miscellaneous",
            basis="0.2% of project value",
        ))

        # Cleaning and debris removal
        cleaning_monthly = self.built_up_area_sqm * 5  # ₹5 per sqm per month
        cleaning_amount = cleaning_monthly * duration_months
        items.append(PrelimsItem(
            description="Site Cleaning and Debris Removal",
            unit="month",
            quantity=duration_months,
            rate=cleaning_monthly,
            amount=cleaning_amount,
            category="miscellaneous",
            basis="₹5 per sqm per month",
        ))

        # Temporary roads and access
        if self.built_up_area_sqm > 2000:
            temp_roads = project_value * 0.003  # 0.3%
            items.append(PrelimsItem(
                description="Temporary Roads and Access",
                unit="LS",
                quantity=1,
                rate=temp_roads,
                amount=temp_roads,
                category="miscellaneous",
                basis="0.3% of project value",
            ))

        # Communication and IT
        it_monthly = 15000 if project_value > 10000000 else 8000
        it_amount = it_monthly * duration_months
        items.append(PrelimsItem(
            description="Communication and IT Infrastructure",
            unit="month",
            quantity=duration_months,
            rate=it_monthly,
            amount=it_amount,
            category="miscellaneous",
            basis="Monthly IT and communication costs",
        ))

        return items

    def get_prelims_benchmark(self) -> dict:
        """Get benchmark prelims percentage for project type."""
        return self.TYPICAL_PRELIMS_PCT.get(
            self.project_type,
            self.TYPICAL_PRELIMS_PCT["residential"]
        )

    def validate_prelims(self, calculated_total: float) -> dict:
        """Validate calculated prelims against benchmarks."""
        benchmark = self.get_prelims_benchmark()
        calculated_pct = (calculated_total / self.project_value * 100) if self.project_value > 0 else 0

        status = "within_range"
        if calculated_pct < benchmark["min"]:
            status = "below_typical"
        elif calculated_pct > benchmark["max"]:
            status = "above_typical"

        return {
            "calculated_percent": round(calculated_pct, 2),
            "benchmark_min": benchmark["min"],
            "benchmark_max": benchmark["max"],
            "benchmark_typical": benchmark["typical"],
            "status": status,
            "recommendation": self._get_recommendation(calculated_pct, benchmark),
        }

    def _get_recommendation(self, calculated_pct: float, benchmark: dict) -> str:
        """Generate recommendation based on prelims validation."""
        if calculated_pct < benchmark["min"]:
            return f"Prelims appear low. Review if all site requirements are covered. Typical range is {benchmark['min']}-{benchmark['max']}%."
        elif calculated_pct > benchmark["max"]:
            return f"Prelims appear high. Review for potential optimization. Typical range is {benchmark['min']}-{benchmark['max']}%."
        else:
            return f"Prelims are within typical range of {benchmark['min']}-{benchmark['max']}%."
