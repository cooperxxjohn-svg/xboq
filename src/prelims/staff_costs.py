"""
Staff Costs Calculator - Calculate site staff and supervision costs.
"""

from typing import List
from .calculator import PrelimsItem


class StaffCostsCalculator:
    """Calculate site staff costs based on project size and duration."""

    # Monthly salaries (CTC) for different positions (Delhi, 2024)
    STAFF_SALARIES = {
        "project_manager": 150000,
        "senior_engineer": 80000,
        "site_engineer": 50000,
        "junior_engineer": 35000,
        "safety_officer": 45000,
        "quality_engineer": 55000,
        "planning_engineer": 60000,
        "surveyor": 40000,
        "store_keeper": 30000,
        "time_keeper": 25000,
        "site_accountant": 35000,
        "office_assistant": 20000,
        "security_guard": 18000,
        "driver": 22000,
        "peon": 15000,
    }

    # Staff requirements by project size (₹ crore)
    STAFF_REQUIREMENTS = {
        "small": {  # < 5 Cr
            "site_engineer": 1,
            "junior_engineer": 1,
            "surveyor": 0.5,
            "store_keeper": 1,
            "time_keeper": 1,
            "security_guard": 2,
            "peon": 1,
        },
        "medium": {  # 5-25 Cr
            "senior_engineer": 1,
            "site_engineer": 2,
            "junior_engineer": 2,
            "safety_officer": 1,
            "surveyor": 1,
            "store_keeper": 1,
            "time_keeper": 1,
            "site_accountant": 1,
            "office_assistant": 1,
            "security_guard": 3,
            "driver": 1,
            "peon": 2,
        },
        "large": {  # 25-100 Cr
            "project_manager": 1,
            "senior_engineer": 2,
            "site_engineer": 4,
            "junior_engineer": 3,
            "safety_officer": 1,
            "quality_engineer": 1,
            "planning_engineer": 1,
            "surveyor": 2,
            "store_keeper": 2,
            "time_keeper": 2,
            "site_accountant": 1,
            "office_assistant": 2,
            "security_guard": 6,
            "driver": 2,
            "peon": 3,
        },
        "mega": {  # > 100 Cr
            "project_manager": 1,
            "senior_engineer": 4,
            "site_engineer": 8,
            "junior_engineer": 6,
            "safety_officer": 2,
            "quality_engineer": 2,
            "planning_engineer": 2,
            "surveyor": 3,
            "store_keeper": 3,
            "time_keeper": 3,
            "site_accountant": 2,
            "office_assistant": 3,
            "security_guard": 10,
            "driver": 3,
            "peon": 5,
        },
    }

    def __init__(self):
        pass

    def calculate(
        self,
        duration_months: int,
        project_value: float,
        project_type: str = "residential",
    ) -> List[PrelimsItem]:
        """Calculate staff costs for the project."""
        items = []

        # Determine project size category
        size_category = self._get_size_category(project_value)

        # Get staff requirements
        requirements = self.STAFF_REQUIREMENTS.get(size_category, self.STAFF_REQUIREMENTS["medium"])

        # Adjust for project type
        type_factor = self._get_type_factor(project_type)

        # Calculate staff costs
        for position, count in requirements.items():
            if count <= 0:
                continue

            monthly_salary = self.STAFF_SALARIES.get(position, 30000)
            adjusted_count = count * type_factor

            # Round to practical numbers
            if adjusted_count < 1:
                adjusted_count = round(adjusted_count, 1)
            else:
                adjusted_count = round(adjusted_count)

            total_cost = monthly_salary * adjusted_count * duration_months

            items.append(PrelimsItem(
                description=position.replace("_", " ").title(),
                unit="month",
                quantity=duration_months * adjusted_count,
                rate=monthly_salary,
                amount=total_cost,
                category="staff_costs",
                basis=f"{adjusted_count} nos × {duration_months} months",
            ))

        # Add head office overhead (typically 3-5% of site staff)
        site_staff_total = sum(i.amount for i in items)
        ho_overhead = site_staff_total * 0.04  # 4%
        items.append(PrelimsItem(
            description="Head Office Overhead (Staff Support)",
            unit="LS",
            quantity=1,
            rate=ho_overhead,
            amount=ho_overhead,
            category="staff_costs",
            basis="4% of site staff costs",
        ))

        return items

    def _get_size_category(self, project_value: float) -> str:
        """Determine project size category from value."""
        value_cr = project_value / 10000000  # Convert to crores

        if value_cr < 5:
            return "small"
        elif value_cr < 25:
            return "medium"
        elif value_cr < 100:
            return "large"
        else:
            return "mega"

    def _get_type_factor(self, project_type: str) -> float:
        """Get staff multiplier based on project type."""
        factors = {
            "residential": 1.0,
            "commercial": 1.1,
            "industrial": 0.9,
            "institutional": 1.15,
            "infrastructure": 1.2,
        }
        return factors.get(project_type.lower(), 1.0)

    def get_recommended_organogram(self, project_value: float) -> dict:
        """Get recommended organization structure."""
        size_category = self._get_size_category(project_value)
        requirements = self.STAFF_REQUIREMENTS.get(size_category, {})

        organogram = {
            "management": [],
            "technical": [],
            "administrative": [],
            "support": [],
        }

        category_mapping = {
            "project_manager": "management",
            "senior_engineer": "management",
            "site_engineer": "technical",
            "junior_engineer": "technical",
            "safety_officer": "technical",
            "quality_engineer": "technical",
            "planning_engineer": "technical",
            "surveyor": "technical",
            "store_keeper": "administrative",
            "time_keeper": "administrative",
            "site_accountant": "administrative",
            "office_assistant": "administrative",
            "security_guard": "support",
            "driver": "support",
            "peon": "support",
        }

        for position, count in requirements.items():
            if count > 0:
                category = category_mapping.get(position, "support")
                organogram[category].append({
                    "position": position.replace("_", " ").title(),
                    "count": count,
                    "monthly_salary": self.STAFF_SALARIES.get(position, 0),
                })

        return organogram
