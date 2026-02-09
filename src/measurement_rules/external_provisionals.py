"""
External Works Provisionals Engine

Auto-generates provisional BOQ items for external works when site plan is not available.
Uses typical Indian construction project norms.

Items generated:
- Earth filling
- Stormwater drains
- Inspection chambers
- Underground tank / Septic tank
- Kerb stones
- GSB (Granular Sub-Base)
- WMM (Wet Mix Macadam)
- External plumbing
- External electrical
- Boundary wall (if not detailed)
- Landscaping allowance
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExternalWorksConfig:
    """Configuration for external works estimation."""

    # Earth work - filling
    filling_depth_m: float = 0.3  # Average filling depth
    filling_rate_inr_cum: float = 650  # Earth filling rate

    # Stormwater drains
    drain_length_ratio: float = 0.15  # rmt per sqm of plot
    drain_rate_inr_rmt: float = 2800  # RCC drain with cover

    # Inspection chambers
    chambers_per_1000sqm: float = 3  # Number per 1000 sqm plot
    chamber_rate_inr_nos: float = 12500  # Per chamber

    # Underground water tank
    ug_tank_litres_per_person: float = 200  # Litres per person per day
    storage_days: float = 2  # Days of storage
    ug_tank_rate_inr_litre: float = 12  # Per litre capacity

    # Septic tank (if no sewer)
    septic_litres_per_person: float = 70  # Litres per person
    septic_rate_inr_litre: float = 15

    # Soak pit
    soak_pit_rate_inr_nos: float = 18000

    # Kerb stones
    kerb_length_ratio: float = 0.08  # rmt per sqm of plot
    kerb_rate_inr_rmt: float = 450

    # Road layers
    gsb_thickness_mm: float = 150
    gsb_rate_inr_sqm: float = 185
    wmm_thickness_mm: float = 100
    wmm_rate_inr_sqm: float = 220

    # External plumbing
    external_plumb_rate_per_sqm_bua: float = 45  # Per sqm of BUA

    # External electrical
    external_elect_rate_per_sqm_bua: float = 35

    # Compound wall (if not detailed)
    compound_wall_rate_inr_rmt: float = 4500  # Per running meter

    # Landscaping
    landscaping_rate_per_sqm: float = 350  # Soft landscaping

    # Gate
    main_gate_rate_inr: float = 125000
    pedestrian_gate_rate_inr: float = 35000


class ExternalProvisionals:
    """
    Generates provisional BOQ items for external works.

    Uses:
    1. Plot area (if known)
    2. Built-up area (fallback)
    3. Number of floors/units
    """

    def __init__(self, config: ExternalWorksConfig = None):
        self.config = config or ExternalWorksConfig()
        self.provisional_items: List[Dict[str, Any]] = []

    def generate_provisionals(
        self,
        plot_area_sqm: float = None,
        built_up_area_sqm: float = None,
        num_floors: int = None,
        num_units: int = None,
        has_sewer_connection: bool = True,
        has_site_plan: bool = False,
        existing_external_items: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate provisional external works items.

        Args:
            plot_area_sqm: Total plot area
            built_up_area_sqm: Total built-up area
            num_floors: Number of floors
            num_units: Number of residential units
            has_sewer_connection: Municipal sewer available
            has_site_plan: True if detailed site plan provided
            existing_external_items: List of external items already in BOQ

        Returns:
            List of provisional BOQ items
        """
        self.provisional_items = []
        existing = existing_external_items or []

        # If site plan provided, skip auto-generation
        if has_site_plan:
            logger.info("Site plan provided - skipping external provisionals")
            return []

        # Estimate plot area if not provided
        if not plot_area_sqm and built_up_area_sqm:
            # Assume ground coverage ~40%
            plot_area_sqm = built_up_area_sqm / (num_floors or 1) / 0.4

        if not plot_area_sqm:
            logger.warning("Cannot estimate external works - no area provided")
            return []

        # Estimate occupants
        occupants = self._estimate_occupants(built_up_area_sqm, num_units)

        # External area (plot - building footprint)
        footprint = (built_up_area_sqm or plot_area_sqm * 0.4) / (num_floors or 1)
        external_area = max(0, plot_area_sqm - footprint)

        # Plot perimeter (approximate)
        plot_perimeter = 4 * (plot_area_sqm ** 0.5)

        # Generate each item if not already present
        if not self._has_item(existing, ["earth", "filling", "fill"]):
            self._add_earth_filling(external_area)

        if not self._has_item(existing, ["storm", "drain", "sw drain"]):
            self._add_stormwater_drains(plot_area_sqm)

        if not self._has_item(existing, ["chamber", "manhole", "ic"]):
            self._add_inspection_chambers(plot_area_sqm)

        if not self._has_item(existing, ["ug tank", "underground", "sump"]):
            self._add_ug_tank(occupants)

        if not has_sewer_connection:
            if not self._has_item(existing, ["septic"]):
                self._add_septic_tank(occupants)
            if not self._has_item(existing, ["soak pit", "soak"]):
                self._add_soak_pit()

        if not self._has_item(existing, ["kerb", "curb"]):
            self._add_kerb_stones(plot_area_sqm)

        if not self._has_item(existing, ["gsb", "granular"]):
            self._add_gsb(external_area)

        if not self._has_item(existing, ["wmm", "wet mix"]):
            self._add_wmm(external_area)

        if not self._has_item(existing, ["external plumb", "external water", "yard plumb"]):
            self._add_external_plumbing(built_up_area_sqm or plot_area_sqm * 0.4)

        if not self._has_item(existing, ["external elect", "yard elect", "street light"]):
            self._add_external_electrical(built_up_area_sqm or plot_area_sqm * 0.4)

        if not self._has_item(existing, ["compound", "boundary"]):
            self._add_compound_wall(plot_perimeter)

        if not self._has_item(existing, ["landscap", "garden", "soft area"]):
            self._add_landscaping(external_area)

        if not self._has_item(existing, ["gate"]):
            self._add_gates()

        logger.info(f"Generated {len(self.provisional_items)} external works provisionals")
        return self.provisional_items

    def _has_item(self, existing: List[str], keywords: List[str]) -> bool:
        """Check if existing items contain any of the keywords."""
        existing_lower = [e.lower() for e in existing]
        for keyword in keywords:
            for existing_item in existing_lower:
                if keyword in existing_item:
                    return True
        return False

    def _estimate_occupants(
        self,
        built_up_area: float = None,
        num_units: int = None
    ) -> int:
        """Estimate number of occupants."""
        if num_units:
            return num_units * 4  # 4 persons per unit

        if built_up_area:
            # Assume 15 sqm per person
            return max(4, int(built_up_area / 15))

        return 8  # Default

    def _add_provisional(
        self,
        item_id: str,
        description: str,
        quantity: float,
        unit: str,
        rate: float,
        derivation: str,
    ) -> None:
        """Add a provisional item."""
        self.provisional_items.append({
            "item_id": item_id,
            "description": f"{description} - PROVISIONAL",
            "quantity": round(quantity, 2),
            "unit": unit,
            "rate": rate,
            "amount": round(quantity * rate, 2),
            "package": "external",
            "drawing_ref": "N/A",
            "is_provisional": True,
            "derivation_method": derivation,
            "confidence": 0.5,
        })

    def _add_earth_filling(self, external_area: float) -> None:
        """Add earth filling item."""
        volume = external_area * self.config.filling_depth_m
        self._add_provisional(
            "EXT-FILL-001",
            "Earth filling in plinth and external areas including compaction",
            volume,
            "cum",
            self.config.filling_rate_inr_cum,
            f"External area {external_area:.0f} sqm x {self.config.filling_depth_m}m depth",
        )

    def _add_stormwater_drains(self, plot_area: float) -> None:
        """Add stormwater drain item."""
        length = plot_area * self.config.drain_length_ratio
        self._add_provisional(
            "EXT-SWD-001",
            "RCC stormwater drain 300x300mm with precast cover",
            length,
            "rmt",
            self.config.drain_rate_inr_rmt,
            f"Plot area {plot_area:.0f} sqm x {self.config.drain_length_ratio} ratio",
        )

    def _add_inspection_chambers(self, plot_area: float) -> None:
        """Add inspection chambers."""
        count = max(2, int(plot_area / 1000 * self.config.chambers_per_1000sqm))
        self._add_provisional(
            "EXT-IC-001",
            "RCC inspection chamber 450x450mm with CI cover",
            count,
            "nos",
            self.config.chamber_rate_inr_nos,
            f"{self.config.chambers_per_1000sqm} per 1000 sqm plot",
        )

    def _add_ug_tank(self, occupants: int) -> None:
        """Add underground water tank."""
        capacity = occupants * self.config.ug_tank_litres_per_person * self.config.storage_days
        capacity = max(5000, capacity)  # Minimum 5000L

        # Round to nearest 1000
        capacity = round(capacity / 1000) * 1000

        self._add_provisional(
            "EXT-UGT-001",
            f"RCC underground water tank {int(capacity)}L capacity with pump room",
            capacity,
            "litres",
            self.config.ug_tank_rate_inr_litre,
            f"{occupants} persons x {self.config.ug_tank_litres_per_person}L x {self.config.storage_days} days",
        )

    def _add_septic_tank(self, occupants: int) -> None:
        """Add septic tank (when no sewer connection)."""
        capacity = occupants * self.config.septic_litres_per_person
        capacity = max(2000, capacity)  # Minimum 2000L
        capacity = round(capacity / 500) * 500

        self._add_provisional(
            "EXT-SEPT-001",
            f"RCC septic tank {int(capacity)}L capacity (two chambers)",
            capacity,
            "litres",
            self.config.septic_rate_inr_litre,
            f"{occupants} persons x {self.config.septic_litres_per_person}L per person",
        )

    def _add_soak_pit(self) -> None:
        """Add soak pit."""
        self._add_provisional(
            "EXT-SOAK-001",
            "Soak pit 1.5m dia x 2.5m deep with rubble filling",
            1,
            "nos",
            self.config.soak_pit_rate_inr_nos,
            "Standard soak pit for septic tank",
        )

    def _add_kerb_stones(self, plot_area: float) -> None:
        """Add kerb stones."""
        length = plot_area * self.config.kerb_length_ratio
        self._add_provisional(
            "EXT-KERB-001",
            "Precast kerb stones 150x300mm with haunching",
            length,
            "rmt",
            self.config.kerb_rate_inr_rmt,
            f"Plot area {plot_area:.0f} sqm x {self.config.kerb_length_ratio} ratio",
        )

    def _add_gsb(self, external_area: float) -> None:
        """Add GSB layer."""
        # Assume 60% of external area needs paving
        paved_area = external_area * 0.6
        self._add_provisional(
            "EXT-GSB-001",
            f"Granular Sub-Base (GSB) {int(self.config.gsb_thickness_mm)}mm thick compacted",
            paved_area,
            "sqm",
            self.config.gsb_rate_inr_sqm,
            f"Paved area {paved_area:.0f} sqm (60% of external)",
        )

    def _add_wmm(self, external_area: float) -> None:
        """Add WMM layer."""
        paved_area = external_area * 0.6
        self._add_provisional(
            "EXT-WMM-001",
            f"Wet Mix Macadam (WMM) {int(self.config.wmm_thickness_mm)}mm thick",
            paved_area,
            "sqm",
            self.config.wmm_rate_inr_sqm,
            f"Paved area {paved_area:.0f} sqm (60% of external)",
        )

    def _add_external_plumbing(self, bua: float) -> None:
        """Add external plumbing allowance."""
        amount = bua * self.config.external_plumb_rate_per_sqm_bua
        self._add_provisional(
            "EXT-PLUMB-001",
            "External plumbing - water supply and drainage connections",
            1,
            "LS",
            amount,
            f"BUA {bua:.0f} sqm x ₹{self.config.external_plumb_rate_per_sqm_bua}/sqm",
        )

    def _add_external_electrical(self, bua: float) -> None:
        """Add external electrical allowance."""
        amount = bua * self.config.external_elect_rate_per_sqm_bua
        self._add_provisional(
            "EXT-ELECT-001",
            "External electrical - EB connection, meter, street lights, conduits",
            1,
            "LS",
            amount,
            f"BUA {bua:.0f} sqm x ₹{self.config.external_elect_rate_per_sqm_bua}/sqm",
        )

    def _add_compound_wall(self, perimeter: float) -> None:
        """Add compound wall allowance."""
        self._add_provisional(
            "EXT-WALL-001",
            "Compound wall 1.8m height including foundation, plastering and painting",
            perimeter,
            "rmt",
            self.config.compound_wall_rate_inr_rmt,
            f"Plot perimeter {perimeter:.0f} rmt",
        )

    def _add_landscaping(self, external_area: float) -> None:
        """Add landscaping allowance."""
        # Assume 30% soft landscaping
        landscape_area = external_area * 0.3
        self._add_provisional(
            "EXT-LAND-001",
            "Soft landscaping including lawn, plants, irrigation",
            landscape_area,
            "sqm",
            self.config.landscaping_rate_per_sqm,
            f"30% of external area {external_area:.0f} sqm",
        )

    def _add_gates(self) -> None:
        """Add gates."""
        self._add_provisional(
            "EXT-GATE-001",
            "Main entrance gate MS fabricated with automation",
            1,
            "nos",
            self.config.main_gate_rate_inr,
            "Standard main gate allowance",
        )
        self._add_provisional(
            "EXT-GATE-002",
            "Pedestrian gate MS fabricated",
            1,
            "nos",
            self.config.pedestrian_gate_rate_inr,
            "Standard pedestrian gate allowance",
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of generated provisionals."""
        total_value = sum(item.get("amount", 0) for item in self.provisional_items)

        return {
            "items_generated": len(self.provisional_items),
            "total_provisional_value": round(total_value, 2),
            "categories": list(set(item.get("item_id", "").split("-")[1] for item in self.provisional_items)),
            "note": "PROVISIONAL - Subject to site survey and detailed design",
        }
