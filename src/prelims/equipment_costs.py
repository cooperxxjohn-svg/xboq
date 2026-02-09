"""
Equipment Costs Calculator - Calculate construction equipment costs.
"""

from typing import List
from .calculator import PrelimsItem


class EquipmentCostsCalculator:
    """Calculate construction equipment rental and operation costs."""

    # Equipment rental rates (per month, including operator where applicable)
    EQUIPMENT_RATES = {
        # Lifting equipment
        "tower_crane": 350000,  # Per month
        "mobile_crane_20t": 180000,
        "mobile_crane_50t": 300000,
        "material_hoist": 45000,
        "passenger_hoist": 60000,

        # Concrete equipment
        "batching_plant": 200000,  # 30 cum/hr
        "transit_mixer": 80000,  # Per unit per month
        "concrete_pump": 120000,
        "vibrator": 3000,

        # Earthwork equipment
        "excavator_pc200": 150000,
        "jcb": 90000,
        "dumper": 60000,
        "roller": 50000,

        # General equipment
        "scaffolding": 80,  # Per sqm per month
        "formwork": 120,  # Per sqm per use
        "bar_bending_machine": 25000,
        "bar_cutting_machine": 20000,
        "welding_machine": 8000,
        "compressor": 25000,

        # Safety equipment
        "safety_nets": 30,  # Per sqm
        "safety_harness": 500,  # Per unit per month
        "fire_extinguisher": 200,  # Per unit per month
    }

    # Equipment requirements by project characteristics
    EQUIPMENT_MATRIX = {
        "low_rise": {  # G+3 or less
            "material_hoist": 1,
            "scaffolding_factor": 0.3,  # 30% of facade area
            "vibrator": 4,
            "bar_bending_machine": 1,
            "bar_cutting_machine": 1,
            "welding_machine": 2,
        },
        "mid_rise": {  # G+4 to G+10
            "tower_crane": 1,
            "material_hoist": 1,
            "passenger_hoist": 1,
            "scaffolding_factor": 0.4,
            "concrete_pump": 0.5,  # Shared
            "vibrator": 6,
            "bar_bending_machine": 2,
            "bar_cutting_machine": 1,
            "welding_machine": 3,
        },
        "high_rise": {  # > G+10
            "tower_crane": 2,
            "material_hoist": 2,
            "passenger_hoist": 2,
            "scaffolding_factor": 0.5,
            "concrete_pump": 1,
            "vibrator": 10,
            "bar_bending_machine": 3,
            "bar_cutting_machine": 2,
            "welding_machine": 4,
        },
    }

    def __init__(self):
        pass

    def calculate(
        self,
        duration_months: int,
        built_up_area_sqm: float,
        project_type: str = "residential",
        floors: int = 4,
    ) -> List[PrelimsItem]:
        """Calculate equipment costs."""
        items = []

        # Determine building category
        category = self._get_building_category(floors)
        requirements = self.EQUIPMENT_MATRIX.get(category, self.EQUIPMENT_MATRIX["low_rise"])

        # Estimate facade area for scaffolding
        facade_area = self._estimate_facade_area(built_up_area_sqm, floors)

        # Calculate equipment for each requirement
        for equipment, qty_or_factor in requirements.items():
            if equipment == "scaffolding_factor":
                # Scaffolding based on facade area
                scaffolding_area = facade_area * qty_or_factor
                scaffolding_cost = scaffolding_area * self.EQUIPMENT_RATES["scaffolding"] * duration_months

                items.append(PrelimsItem(
                    description="Scaffolding (External)",
                    unit="sqm-month",
                    quantity=scaffolding_area * duration_months,
                    rate=self.EQUIPMENT_RATES["scaffolding"],
                    amount=scaffolding_cost,
                    category="equipment",
                    basis=f"{scaffolding_area:.0f} sqm × {duration_months} months",
                ))
            else:
                if qty_or_factor <= 0:
                    continue

                rate = self.EQUIPMENT_RATES.get(equipment, 50000)
                # Equipment typically needed for 80% of duration
                effective_months = duration_months * 0.8
                quantity = qty_or_factor * effective_months
                amount = quantity * rate

                items.append(PrelimsItem(
                    description=equipment.replace("_", " ").title(),
                    unit="month",
                    quantity=round(quantity, 1),
                    rate=rate,
                    amount=amount,
                    category="equipment",
                    basis=f"{qty_or_factor} units × {effective_months:.0f} months (80% duration)",
                ))

        # Formwork (based on slab area per floor)
        slab_area_per_floor = built_up_area_sqm / max(1, floors)
        formwork_reuses = 8  # Typical reuse factor
        formwork_area = slab_area_per_floor * 1.2  # 20% extra for beams etc.
        formwork_cost = formwork_area * self.EQUIPMENT_RATES["formwork"] * (floors / formwork_reuses)

        items.append(PrelimsItem(
            description="Formwork Material",
            unit="sqm",
            quantity=formwork_area,
            rate=self.EQUIPMENT_RATES["formwork"] * (floors / formwork_reuses),
            amount=formwork_cost,
            category="equipment",
            basis=f"{formwork_area:.0f} sqm with {formwork_reuses} reuses",
        ))

        # Safety equipment
        # Safety nets
        if floors > 3:
            net_area = facade_area * 0.5
            net_cost = net_area * self.EQUIPMENT_RATES["safety_nets"]
            items.append(PrelimsItem(
                description="Safety Nets",
                unit="sqm",
                quantity=net_area,
                rate=self.EQUIPMENT_RATES["safety_nets"],
                amount=net_cost,
                category="safety_equipment",
                basis="50% of facade area",
            ))

        # Safety harnesses (based on workers at height)
        workers_at_height = max(10, int(built_up_area_sqm / 200))
        harness_cost = workers_at_height * self.EQUIPMENT_RATES["safety_harness"] * duration_months

        items.append(PrelimsItem(
            description="Safety Harness and PPE",
            unit="month",
            quantity=duration_months,
            rate=workers_at_height * self.EQUIPMENT_RATES["safety_harness"],
            amount=harness_cost,
            category="safety_equipment",
            basis=f"{workers_at_height} sets",
        ))

        # Fire extinguishers
        extinguisher_count = max(5, int(built_up_area_sqm / 500))
        extinguisher_cost = extinguisher_count * self.EQUIPMENT_RATES["fire_extinguisher"] * duration_months

        items.append(PrelimsItem(
            description="Fire Extinguishers",
            unit="month",
            quantity=duration_months,
            rate=extinguisher_count * self.EQUIPMENT_RATES["fire_extinguisher"],
            amount=extinguisher_cost,
            category="safety_equipment",
            basis=f"{extinguisher_count} units",
        ))

        # Small tools and consumables (typically 0.5-1% of project value)
        # Estimate based on area
        tools_allowance = built_up_area_sqm * 15  # ₹15 per sqm
        items.append(PrelimsItem(
            description="Small Tools and Consumables",
            unit="LS",
            quantity=1,
            rate=tools_allowance,
            amount=tools_allowance,
            category="equipment",
            basis="₹15 per sqm built-up area",
        ))

        return items

    def _get_building_category(self, floors: int) -> str:
        """Determine building category from number of floors."""
        if floors <= 4:
            return "low_rise"
        elif floors <= 10:
            return "mid_rise"
        else:
            return "high_rise"

    def _estimate_facade_area(self, built_up_area: float, floors: int) -> float:
        """Estimate facade area from built-up area and floors."""
        # Assume floor plate is roughly square
        floor_area = built_up_area / max(1, floors)
        side = floor_area ** 0.5

        # Assume average floor height of 3m
        floor_height = 3.0

        # Perimeter × height × floors
        perimeter = side * 4
        facade = perimeter * floor_height * floors

        return facade

    def get_equipment_schedule(
        self,
        duration_months: int,
        floors: int,
    ) -> dict:
        """Generate equipment mobilization schedule."""
        schedule = {}
        category = self._get_building_category(floors)

        # Month-wise equipment presence
        for month in range(1, duration_months + 1):
            schedule[month] = []

            # Excavation equipment (first 15% of duration)
            if month <= duration_months * 0.15:
                schedule[month].extend(["Excavator", "JCB", "Dumper"])

            # Concrete equipment (10% to 80% of duration)
            if duration_months * 0.10 <= month <= duration_months * 0.80:
                schedule[month].extend(["Batching Plant", "Transit Mixer", "Concrete Pump", "Vibrators"])

            # Lifting equipment (15% to 90% of duration)
            if duration_months * 0.15 <= month <= duration_months * 0.90:
                if category in ["mid_rise", "high_rise"]:
                    schedule[month].append("Tower Crane")
                schedule[month].extend(["Material Hoist", "Passenger Hoist"])

            # Scaffolding (40% to 95% of duration)
            if duration_months * 0.40 <= month <= duration_months * 0.95:
                schedule[month].append("Scaffolding")

        return schedule
