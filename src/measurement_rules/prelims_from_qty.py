"""
Prelims from Quantities Engine

Derives preliminary & general items from actual project quantities rather than templates.
Uses Indian construction project norms.

Derivation basis:
- RCC volume -> Scaffolding, formwork labor, curing
- Number of floors -> Vertical transport (tower crane, hoist)
- Footprint area -> Site facilities, offices
- Project duration -> Staff costs, temp services
- Project value -> Insurance, contingency
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

logger = logging.getLogger(__name__)


@dataclass
class PrelimsConstants:
    """Indian construction project prelims constants."""

    # STAFF COSTS (monthly salaries INR)
    project_manager_salary: float = 125000
    site_engineer_salary: float = 65000
    supervisor_salary: float = 45000
    surveyor_salary: float = 55000
    quantity_surveyor_salary: float = 70000
    safety_officer_salary: float = 55000
    store_keeper_salary: float = 35000
    accountant_salary: float = 45000
    watchman_salary: float = 18000
    peon_salary: float = 15000

    # Staff ratios per 1000 sqm BUA
    engineers_per_1000sqm: float = 0.5
    supervisors_per_1000sqm: float = 1.0

    # SITE FACILITIES
    site_office_per_sqm_bua: float = 8  # INR per sqm BUA
    stores_per_sqm_bua: float = 5
    labour_hutments_per_sqm_bua: float = 4
    toilet_blocks_per_1000sqm: float = 2

    # TEMPORARY SERVICES
    temp_electricity_per_month: float = 45000
    temp_water_per_month: float = 25000
    internet_per_month: float = 5000

    # EQUIPMENT - derived from RCC volume
    # Tower crane typically for > 10,000 sqm or > 3 floors
    tower_crane_monthly: float = 185000
    tower_crane_mobilization: float = 250000
    hoist_monthly: float = 45000

    # Scaffolding per sqm of formwork
    scaffolding_per_sqm_formwork: float = 85

    # Curing per cum of RCC
    curing_per_cum_rcc: float = 120

    # SAFETY
    safety_equipment_per_sqm: float = 12
    ppe_per_month: float = 35000
    first_aid_initial: float = 25000

    # INSURANCE & STATUTORY
    car_insurance_pct_of_value: float = 0.002  # 0.2% of project value
    workmen_comp_pct_of_labour: float = 0.01  # 1% of labour cost
    statutory_fees_pct: float = 0.005

    # TESTING
    concrete_testing_per_cum: float = 180
    steel_testing_per_mt: float = 850
    soil_testing_initial: float = 45000

    # MISCELLANEOUS
    survey_per_sqm_plot: float = 8
    tools_tackles_pct: float = 0.005  # 0.5% of project value
    mobilization_demob_pct: float = 0.01  # 1% of project value


class PrelimsFromQuantities:
    """
    Derives preliminary items from actual project quantities.

    More accurate than template-based prelims because:
    1. Staff proportional to actual project size
    2. Equipment based on actual RCC/floors
    3. Testing based on actual material quantities
    4. Duration influences time-related costs
    """

    def __init__(self, constants: PrelimsConstants = None):
        self.constants = constants or PrelimsConstants()
        self.prelims_items: List[Dict[str, Any]] = []

    def calculate_prelims(
        self,
        rcc_volume_cum: float,
        steel_qty_mt: float,
        built_up_area_sqm: float,
        plot_area_sqm: float = None,
        num_floors: int = 1,
        project_duration_months: int = 12,
        project_value_inr: float = 0,
        formwork_area_sqm: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Calculate preliminary items from quantities.

        Args:
            rcc_volume_cum: Total RCC volume in cubic meters
            steel_qty_mt: Total steel quantity in metric tonnes
            built_up_area_sqm: Total built-up area
            plot_area_sqm: Plot area (for site facilities)
            num_floors: Number of floors
            project_duration_months: Project duration
            project_value_inr: Total project value
            formwork_area_sqm: Total formwork area (derived if not provided)

        Returns:
            List of preliminary BOQ items
        """
        self.prelims_items = []

        # Default plot area
        if not plot_area_sqm:
            plot_area_sqm = built_up_area_sqm / num_floors / 0.4

        # Default formwork area
        if not formwork_area_sqm:
            formwork_area_sqm = rcc_volume_cum * 10  # Approximate

        # 1. STAFF COSTS
        self._calculate_staff_costs(built_up_area_sqm, project_duration_months)

        # 2. SITE FACILITIES
        self._calculate_site_facilities(built_up_area_sqm, plot_area_sqm)

        # 3. TEMPORARY SERVICES
        self._calculate_temp_services(project_duration_months)

        # 4. EQUIPMENT (based on RCC and floors)
        self._calculate_equipment(
            rcc_volume_cum, formwork_area_sqm, num_floors,
            built_up_area_sqm, project_duration_months
        )

        # 5. SAFETY & PPE
        self._calculate_safety(built_up_area_sqm, project_duration_months)

        # 6. TESTING (based on material quantities)
        self._calculate_testing(rcc_volume_cum, steel_qty_mt)

        # 7. INSURANCE & STATUTORY
        self._calculate_insurance_statutory(project_value_inr)

        # 8. MISCELLANEOUS
        self._calculate_miscellaneous(plot_area_sqm, project_value_inr)

        logger.info(f"Calculated {len(self.prelims_items)} prelims items from quantities")
        return self.prelims_items

    def _add_prelim(
        self,
        item_id: str,
        description: str,
        quantity: float,
        unit: str,
        rate: float,
        category: str,
        derivation: str,
    ) -> None:
        """Add a prelims item."""
        self.prelims_items.append({
            "item_id": item_id,
            "description": description,
            "quantity": round(quantity, 2),
            "unit": unit,
            "rate": round(rate, 2),
            "amount": round(quantity * rate, 2),
            "category": category,
            "derivation": derivation,
        })

    def _calculate_staff_costs(self, bua: float, duration: int) -> None:
        """Calculate staff costs based on project size."""
        c = self.constants

        # Number of staff based on BUA
        num_engineers = max(1, math.ceil(bua / 1000 * c.engineers_per_1000sqm))
        num_supervisors = max(1, math.ceil(bua / 1000 * c.supervisors_per_1000sqm))

        # Project manager for larger projects (>5000 sqm)
        if bua > 5000:
            self._add_prelim(
                "PG-STAFF-001",
                f"Project Manager (1 no x {duration} months)",
                duration, "months", c.project_manager_salary, "staff",
                f"Required for project > 5000 sqm"
            )

        # Site engineers
        self._add_prelim(
            "PG-STAFF-002",
            f"Site Engineers ({num_engineers} nos x {duration} months)",
            duration * num_engineers, "man-months", c.site_engineer_salary, "staff",
            f"{bua:.0f} sqm / 1000 x {c.engineers_per_1000sqm} = {num_engineers} engineers"
        )

        # Supervisors
        self._add_prelim(
            "PG-STAFF-003",
            f"Site Supervisors ({num_supervisors} nos x {duration} months)",
            duration * num_supervisors, "man-months", c.supervisor_salary, "staff",
            f"{bua:.0f} sqm / 1000 x {c.supervisors_per_1000sqm} = {num_supervisors} supervisors"
        )

        # Support staff
        self._add_prelim(
            "PG-STAFF-004",
            f"Storekeeper (1 no x {duration} months)",
            duration, "months", c.store_keeper_salary, "staff",
            "Standard requirement"
        )

        self._add_prelim(
            "PG-STAFF-005",
            f"Security/Watchmen (2 nos x {duration} months)",
            duration * 2, "man-months", c.watchman_salary, "staff",
            "2 watchmen for 24hr coverage"
        )

        # Safety officer for larger projects
        if bua > 3000:
            self._add_prelim(
                "PG-STAFF-006",
                f"Safety Officer (1 no x {duration} months)",
                duration, "months", c.safety_officer_salary, "staff",
                "Required for project > 3000 sqm"
            )

    def _calculate_site_facilities(self, bua: float, plot_area: float) -> None:
        """Calculate site facilities based on areas."""
        c = self.constants

        # Site office
        office_cost = bua * c.site_office_per_sqm_bua
        self._add_prelim(
            "PG-FAC-001",
            "Site office (prefab/container) including furniture",
            1, "LS", office_cost, "facilities",
            f"BUA {bua:.0f} sqm x ₹{c.site_office_per_sqm_bua}/sqm"
        )

        # Stores
        stores_cost = bua * c.stores_per_sqm_bua
        self._add_prelim(
            "PG-FAC-002",
            "Material stores and cement godown",
            1, "LS", stores_cost, "facilities",
            f"BUA {bua:.0f} sqm x ₹{c.stores_per_sqm_bua}/sqm"
        )

        # Labour hutments
        hutments_cost = bua * c.labour_hutments_per_sqm_bua
        self._add_prelim(
            "PG-FAC-003",
            "Labour hutments/accommodation",
            1, "LS", hutments_cost, "facilities",
            f"BUA {bua:.0f} sqm x ₹{c.labour_hutments_per_sqm_bua}/sqm"
        )

        # Toilet blocks
        num_toilets = max(2, math.ceil(bua / 1000 * c.toilet_blocks_per_1000sqm))
        self._add_prelim(
            "PG-FAC-004",
            f"Temporary toilet blocks ({num_toilets} nos)",
            num_toilets, "nos", 35000, "facilities",
            f"{bua:.0f} sqm / 1000 x {c.toilet_blocks_per_1000sqm} = {num_toilets}"
        )

    def _calculate_temp_services(self, duration: int) -> None:
        """Calculate temporary services for project duration."""
        c = self.constants

        self._add_prelim(
            "PG-SVC-001",
            f"Temporary electricity connection and monthly charges ({duration} months)",
            duration, "months", c.temp_electricity_per_month, "services",
            "Monthly charges for construction power"
        )

        self._add_prelim(
            "PG-SVC-002",
            f"Temporary water connection and monthly charges ({duration} months)",
            duration, "months", c.temp_water_per_month, "services",
            "Municipal water for construction"
        )

        self._add_prelim(
            "PG-SVC-003",
            f"Internet/communication ({duration} months)",
            duration, "months", c.internet_per_month, "services",
            "Site office internet and communication"
        )

    def _calculate_equipment(
        self,
        rcc_cum: float,
        formwork_sqm: float,
        num_floors: int,
        bua: float,
        duration: int,
    ) -> None:
        """Calculate equipment based on RCC volume and building height."""
        c = self.constants

        # Tower crane for multi-storey (>G+2) or large projects
        if num_floors > 3 or bua > 8000:
            crane_months = int(duration * 0.7)  # 70% of project duration
            self._add_prelim(
                "PG-EQP-001",
                f"Tower crane hire including operator ({crane_months} months)",
                crane_months, "months", c.tower_crane_monthly, "equipment",
                f"Required for {num_floors} floors / {bua:.0f} sqm"
            )
            self._add_prelim(
                "PG-EQP-002",
                "Tower crane mobilization and demobilization",
                1, "LS", c.tower_crane_mobilization, "equipment",
                "Transport, erection, and dismantling"
            )
        else:
            # Material hoist for smaller projects
            hoist_months = int(duration * 0.6)
            self._add_prelim(
                "PG-EQP-001",
                f"Material hoist hire ({hoist_months} months)",
                hoist_months, "months", c.hoist_monthly, "equipment",
                f"Suitable for {num_floors} floor project"
            )

        # Scaffolding based on formwork area
        self._add_prelim(
            "PG-EQP-003",
            "Scaffolding and staging hire for formwork",
            formwork_sqm, "sqm", c.scaffolding_per_sqm_formwork, "equipment",
            f"Total formwork area: {formwork_sqm:.0f} sqm"
        )

        # Curing based on RCC volume
        self._add_prelim(
            "PG-EQP-004",
            "Concrete curing (water, jute, compound)",
            rcc_cum, "cum", c.curing_per_cum_rcc, "equipment",
            f"Total RCC: {rcc_cum:.0f} cum"
        )

    def _calculate_safety(self, bua: float, duration: int) -> None:
        """Calculate safety requirements."""
        c = self.constants

        self._add_prelim(
            "PG-SAF-001",
            "Safety equipment (nets, barricades, signages)",
            bua, "sqm", c.safety_equipment_per_sqm, "safety",
            f"BUA {bua:.0f} sqm"
        )

        self._add_prelim(
            "PG-SAF-002",
            f"PPE (helmets, boots, jackets, gloves) - {duration} months",
            duration, "months", c.ppe_per_month, "safety",
            "Monthly replenishment"
        )

        self._add_prelim(
            "PG-SAF-003",
            "First aid kit and medical supplies (initial)",
            1, "LS", c.first_aid_initial, "safety",
            "Initial setup"
        )

    def _calculate_testing(self, rcc_cum: float, steel_mt: float) -> None:
        """Calculate testing based on material quantities."""
        c = self.constants

        self._add_prelim(
            "PG-TEST-001",
            f"Concrete cube testing ({int(rcc_cum/50)} sets of cubes)",
            rcc_cum, "cum", c.concrete_testing_per_cum, "testing",
            f"1 set per 50 cum = {int(rcc_cum/50)} sets"
        )

        self._add_prelim(
            "PG-TEST-002",
            f"Steel reinforcement testing ({int(steel_mt/50)} samples)",
            steel_mt, "MT", c.steel_testing_per_mt, "testing",
            f"1 sample per 50 MT = {int(steel_mt/50)} samples"
        )

        self._add_prelim(
            "PG-TEST-003",
            "Soil investigation (initial)",
            1, "LS", c.soil_testing_initial, "testing",
            "SPT and bore log"
        )

    def _calculate_insurance_statutory(self, project_value: float) -> None:
        """Calculate insurance and statutory costs."""
        c = self.constants

        if project_value > 0:
            car_premium = project_value * c.car_insurance_pct_of_value
            self._add_prelim(
                "PG-INS-001",
                "Contractor All Risk (CAR) insurance",
                1, "LS", car_premium, "insurance",
                f"{c.car_insurance_pct_of_value*100}% of project value ₹{project_value:,.0f}"
            )

            # Estimate labour cost as 25% of project value
            labour_value = project_value * 0.25
            workmen_comp = labour_value * c.workmen_comp_pct_of_labour
            self._add_prelim(
                "PG-INS-002",
                "Workmen compensation insurance",
                1, "LS", workmen_comp, "insurance",
                f"{c.workmen_comp_pct_of_labour*100}% of labour value"
            )

            statutory = project_value * c.statutory_fees_pct
            self._add_prelim(
                "PG-STAT-001",
                "Statutory fees and permissions",
                1, "LS", statutory, "statutory",
                f"Estimated {c.statutory_fees_pct*100}% of project value"
            )

    def _calculate_miscellaneous(self, plot_area: float, project_value: float) -> None:
        """Calculate miscellaneous items."""
        c = self.constants

        # Survey
        self._add_prelim(
            "PG-MISC-001",
            "Initial and periodic survey",
            plot_area, "sqm", c.survey_per_sqm_plot, "misc",
            f"Plot area {plot_area:.0f} sqm"
        )

        if project_value > 0:
            tools_cost = project_value * c.tools_tackles_pct
            self._add_prelim(
                "PG-MISC-002",
                "Tools and tackles",
                1, "LS", tools_cost, "misc",
                f"{c.tools_tackles_pct*100}% of project value"
            )

            mob_demob = project_value * c.mobilization_demob_pct
            self._add_prelim(
                "PG-MISC-003",
                "Mobilization and demobilization",
                1, "LS", mob_demob, "misc",
                f"{c.mobilization_demob_pct*100}% of project value"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get prelims summary by category."""
        by_category = {}
        for item in self.prelims_items:
            cat = item.get("category", "other")
            if cat not in by_category:
                by_category[cat] = {"items": 0, "amount": 0}
            by_category[cat]["items"] += 1
            by_category[cat]["amount"] += item.get("amount", 0)

        total = sum(item.get("amount", 0) for item in self.prelims_items)

        return {
            "total_items": len(self.prelims_items),
            "total_value": round(total, 2),
            "by_category": by_category,
        }
