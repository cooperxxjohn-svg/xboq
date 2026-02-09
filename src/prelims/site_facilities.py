"""
Site Facilities Calculator - Calculate temporary facilities costs.
"""

from typing import List
from .calculator import PrelimsItem


class SiteFacilitiesCalculator:
    """Calculate site facilities and temporary works costs."""

    # Monthly rental rates for temporary facilities
    FACILITY_RATES = {
        # Site offices
        "site_office_container": 8000,  # 20ft container per month
        "site_office_prefab": 12000,  # Prefab cabin per month
        "meeting_room": 10000,  # Per month
        "store_room": 5000,  # Per month

        # Labor facilities
        "labor_shed": 4000,  # Per 10 workers per month
        "toilet_block": 3000,  # Per unit per month
        "drinking_water_cooler": 800,  # Per unit per month
        "first_aid_room": 5000,  # Per month

        # Utilities
        "temporary_electricity": 50,  # Per kW per month (load charges)
        "generator_diesel": 25,  # Per kW per month (fuel only)
        "water_tanker": 3000,  # Per trip
        "water_storage_tank": 2000,  # Per 5000L tank per month
    }

    # Setup/dismantling costs (one-time)
    SETUP_COSTS = {
        "boundary_hoarding": 800,  # Per RMT
        "site_signage": 50000,  # LS
        "security_cabin": 25000,  # Each
        "temporary_power_connection": 75000,  # LS
        "temporary_water_connection": 30000,  # LS
        "site_drainage": 100,  # Per sqm
    }

    def __init__(self):
        pass

    def calculate(
        self,
        duration_months: int,
        built_up_area_sqm: float,
        project_type: str = "residential",
    ) -> List[PrelimsItem]:
        """Calculate site facilities costs."""
        items = []

        # Estimate project characteristics
        perimeter = self._estimate_perimeter(built_up_area_sqm)
        peak_workers = self._estimate_peak_workers(built_up_area_sqm, project_type)
        power_requirement = self._estimate_power(built_up_area_sqm, project_type)

        # 1. Site offices
        office_count = self._get_office_count(built_up_area_sqm)

        items.append(PrelimsItem(
            description="Site Office (Prefab/Container)",
            unit="month",
            quantity=duration_months * office_count,
            rate=self.FACILITY_RATES["site_office_prefab"],
            amount=self.FACILITY_RATES["site_office_prefab"] * duration_months * office_count,
            category="site_facilities",
            basis=f"{office_count} offices × {duration_months} months",
        ))

        if built_up_area_sqm > 5000:
            items.append(PrelimsItem(
                description="Meeting Room / Conference",
                unit="month",
                quantity=duration_months,
                rate=self.FACILITY_RATES["meeting_room"],
                amount=self.FACILITY_RATES["meeting_room"] * duration_months,
                category="site_facilities",
                basis=f"1 room × {duration_months} months",
            ))

        # Store rooms
        store_count = max(1, int(built_up_area_sqm / 3000))
        items.append(PrelimsItem(
            description="Store Room / Material Storage",
            unit="month",
            quantity=duration_months * store_count,
            rate=self.FACILITY_RATES["store_room"],
            amount=self.FACILITY_RATES["store_room"] * duration_months * store_count,
            category="site_facilities",
            basis=f"{store_count} stores × {duration_months} months",
        ))

        # 2. Labor facilities
        labor_sheds = max(1, int(peak_workers / 10))
        items.append(PrelimsItem(
            description="Labor Rest Sheds",
            unit="month",
            quantity=duration_months * labor_sheds,
            rate=self.FACILITY_RATES["labor_shed"],
            amount=self.FACILITY_RATES["labor_shed"] * duration_months * labor_sheds,
            category="site_facilities",
            basis=f"For {peak_workers} peak workers",
        ))

        # Toilets (1 per 25 workers as per norms)
        toilets = max(2, int(peak_workers / 25))
        items.append(PrelimsItem(
            description="Toilet Blocks",
            unit="month",
            quantity=duration_months * toilets,
            rate=self.FACILITY_RATES["toilet_block"],
            amount=self.FACILITY_RATES["toilet_block"] * duration_months * toilets,
            category="site_facilities",
            basis=f"{toilets} units (1 per 25 workers)",
        ))

        # Drinking water
        coolers = max(1, int(peak_workers / 50))
        items.append(PrelimsItem(
            description="Drinking Water Coolers",
            unit="month",
            quantity=duration_months * coolers,
            rate=self.FACILITY_RATES["drinking_water_cooler"],
            amount=self.FACILITY_RATES["drinking_water_cooler"] * duration_months * coolers,
            category="site_facilities",
            basis=f"{coolers} units",
        ))

        # First aid
        items.append(PrelimsItem(
            description="First Aid Room",
            unit="month",
            quantity=duration_months,
            rate=self.FACILITY_RATES["first_aid_room"],
            amount=self.FACILITY_RATES["first_aid_room"] * duration_months,
            category="site_facilities",
            basis="Mandatory as per safety norms",
        ))

        # 3. Utilities
        # Temporary electricity
        items.append(PrelimsItem(
            description="Temporary Electricity (Load Charges)",
            unit="month",
            quantity=duration_months * power_requirement,
            rate=self.FACILITY_RATES["temporary_electricity"],
            amount=self.FACILITY_RATES["temporary_electricity"] * duration_months * power_requirement,
            category="utilities",
            basis=f"{power_requirement} kW connected load",
        ))

        # Generator backup (assume 50% of load for 8 hours/day)
        gen_load = power_requirement * 0.5
        items.append(PrelimsItem(
            description="Generator/DG Set Hire (Standby)",
            unit="month",
            quantity=duration_months,
            rate=gen_load * 2000,  # Approx rental per kW per month
            amount=gen_load * 2000 * duration_months,
            category="utilities",
            basis=f"{gen_load:.0f} kW standby",
        ))

        # Water supply
        water_trips_month = max(10, int(peak_workers / 10))
        items.append(PrelimsItem(
            description="Water Supply (Tanker)",
            unit="month",
            quantity=duration_months,
            rate=water_trips_month * self.FACILITY_RATES["water_tanker"],
            amount=water_trips_month * self.FACILITY_RATES["water_tanker"] * duration_months,
            category="utilities",
            basis=f"{water_trips_month} trips/month",
        ))

        # Water storage
        tanks = max(1, int(built_up_area_sqm / 2000))
        items.append(PrelimsItem(
            description="Water Storage Tanks",
            unit="month",
            quantity=duration_months * tanks,
            rate=self.FACILITY_RATES["water_storage_tank"],
            amount=self.FACILITY_RATES["water_storage_tank"] * duration_months * tanks,
            category="utilities",
            basis=f"{tanks} × 5000L tanks",
        ))

        # 4. Setup costs (one-time)
        # Boundary hoarding
        items.append(PrelimsItem(
            description="Boundary Hoarding/Fencing",
            unit="rmt",
            quantity=perimeter,
            rate=self.SETUP_COSTS["boundary_hoarding"],
            amount=self.SETUP_COSTS["boundary_hoarding"] * perimeter,
            category="site_setup",
            basis=f"{perimeter:.0f} rmt perimeter",
        ))

        # Site signage
        items.append(PrelimsItem(
            description="Site Signage and Display Boards",
            unit="LS",
            quantity=1,
            rate=self.SETUP_COSTS["site_signage"],
            amount=self.SETUP_COSTS["site_signage"],
            category="site_setup",
            basis="Project signage, safety boards",
        ))

        # Security cabin
        security_cabins = max(1, int(perimeter / 100))
        items.append(PrelimsItem(
            description="Security Cabins",
            unit="nos",
            quantity=security_cabins,
            rate=self.SETUP_COSTS["security_cabin"],
            amount=self.SETUP_COSTS["security_cabin"] * security_cabins,
            category="site_setup",
            basis=f"{security_cabins} cabins",
        ))

        # Temporary connections
        items.append(PrelimsItem(
            description="Temporary Power Connection",
            unit="LS",
            quantity=1,
            rate=self.SETUP_COSTS["temporary_power_connection"],
            amount=self.SETUP_COSTS["temporary_power_connection"],
            category="site_setup",
            basis="DISCOM connection charges",
        ))

        items.append(PrelimsItem(
            description="Temporary Water Connection",
            unit="LS",
            quantity=1,
            rate=self.SETUP_COSTS["temporary_water_connection"],
            amount=self.SETUP_COSTS["temporary_water_connection"],
            category="site_setup",
            basis="Municipal connection charges",
        ))

        # Site drainage
        drainage_area = built_up_area_sqm * 0.2  # 20% of BUA
        items.append(PrelimsItem(
            description="Temporary Site Drainage",
            unit="sqm",
            quantity=drainage_area,
            rate=self.SETUP_COSTS["site_drainage"],
            amount=self.SETUP_COSTS["site_drainage"] * drainage_area,
            category="site_setup",
            basis="20% of built-up area",
        ))

        return items

    def _estimate_perimeter(self, built_up_area: float) -> float:
        """Estimate site perimeter from built-up area."""
        # Assume plot is roughly 1.5x built-up area and square-ish
        plot_area = built_up_area * 1.5
        side = (plot_area ** 0.5)
        return side * 4  # Perimeter

    def _estimate_peak_workers(self, built_up_area: float, project_type: str) -> int:
        """Estimate peak workforce based on project size."""
        # Rule of thumb: 1 worker per 10-15 sqm during peak
        workers_per_sqm = {
            "residential": 0.08,
            "commercial": 0.10,
            "industrial": 0.06,
            "institutional": 0.09,
        }
        factor = workers_per_sqm.get(project_type.lower(), 0.08)
        return max(20, int(built_up_area * factor))

    def _estimate_power(self, built_up_area: float, project_type: str) -> int:
        """Estimate power requirement in kW."""
        # Rule of thumb: 1 kW per 50-100 sqm
        kw_per_sqm = {
            "residential": 0.015,
            "commercial": 0.020,
            "industrial": 0.025,
            "institutional": 0.018,
        }
        factor = kw_per_sqm.get(project_type.lower(), 0.015)
        return max(25, int(built_up_area * factor))

    def _get_office_count(self, built_up_area: float) -> int:
        """Get number of site offices based on project size."""
        if built_up_area < 2000:
            return 1
        elif built_up_area < 5000:
            return 2
        elif built_up_area < 10000:
            return 3
        else:
            return max(4, int(built_up_area / 3000))
