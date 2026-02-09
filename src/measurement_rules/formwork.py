"""
RCC Formwork Derivation Engine

Derives formwork (centering/shuttering) quantities from RCC quantities.
Uses standard Indian constants for formwork ratios.

Reference: IS 14687 (Formwork for Concrete)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FormworkType(Enum):
    """Types of formwork."""
    SLAB_SOFFIT = "slab_soffit"
    SLAB_EDGE = "slab_edge"
    BEAM_BOTTOM = "beam_bottom"
    BEAM_SIDES = "beam_sides"
    COLUMN = "column"
    STAIRCASE = "staircase"
    FOOTING = "footing"
    LINTEL = "lintel"
    CHAJJA = "chajja"
    SUNSHADE = "sunshade"


@dataclass
class FormworkConstants:
    """
    Standard Indian constants for formwork derivation.

    These are typical ratios used when only RCC volume is known.
    More accurate if actual dimensions are provided.
    """
    # Slab formwork per cum of slab concrete
    slab_formwork_sqm_per_cum: float = 8.0  # For 125mm slab = 1/0.125

    # Beam formwork per cum of beam concrete
    beam_formwork_sqm_per_cum: float = 12.0  # Typical 230x450 beam

    # Column formwork per cum of column concrete
    column_formwork_sqm_per_cum: float = 16.0  # Typical 450x450 column, 3m height

    # Staircase formwork per cum
    staircase_formwork_sqm_per_cum: float = 10.0

    # Footing formwork per cum (sides only)
    footing_formwork_sqm_per_cum: float = 4.0

    # Lintel formwork per cum
    lintel_formwork_sqm_per_cum: float = 14.0

    # Chajja/sunshade formwork per cum
    chajja_formwork_sqm_per_cum: float = 12.0

    # Props/staging per sqm of slab formwork
    staging_sqm_per_formwork_sqm: float = 1.0

    # De-shuttering oil per sqm of formwork
    deshuttering_oil_litre_per_sqm: float = 0.02


class FormworkDeriver:
    """
    Derives formwork quantities from RCC quantities.

    Methods:
    1. From dimensions: Accurate calculation from L x B x D
    2. From volume: Use constants when only cum is known
    """

    def __init__(self, constants: FormworkConstants = None):
        self.constants = constants or FormworkConstants()
        self.derived_items: List[Dict[str, Any]] = []

    def derive_formwork(
        self,
        rcc_items: List[Dict[str, Any]],
        structural_data: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Derive formwork BOQ items from RCC items.

        Args:
            rcc_items: RCC BOQ items with quantities
            structural_data: Optional detailed structural info

        Returns:
            List of formwork BOQ items to add
        """
        self.derived_items = []

        for item in rcc_items:
            item_type = self._classify_rcc_item(item)

            if item_type == "slab":
                self._derive_slab_formwork(item, structural_data)
            elif item_type == "beam":
                self._derive_beam_formwork(item, structural_data)
            elif item_type == "column":
                self._derive_column_formwork(item, structural_data)
            elif item_type == "staircase":
                self._derive_staircase_formwork(item)
            elif item_type == "footing":
                self._derive_footing_formwork(item, structural_data)
            elif item_type == "lintel":
                self._derive_lintel_formwork(item)
            elif item_type == "chajja":
                self._derive_chajja_formwork(item)

        # Add staging/propping
        self._derive_staging()

        return self.derived_items

    def _classify_rcc_item(self, item: Dict[str, Any]) -> str:
        """Classify RCC item type."""
        desc = item.get("description", "").lower()

        if "slab" in desc or "roof" in desc or "floor" in desc:
            return "slab"
        elif "beam" in desc:
            return "beam"
        elif "column" in desc:
            return "column"
        elif "stair" in desc or "waist" in desc:
            return "staircase"
        elif "footing" in desc or "foundation" in desc or "raft" in desc:
            return "footing"
        elif "lintel" in desc:
            return "lintel"
        elif "chajja" in desc or "sunshade" in desc or "canopy" in desc:
            return "chajja"

        return "other"

    def _derive_slab_formwork(
        self,
        item: Dict[str, Any],
        structural: Dict[str, Any] = None
    ) -> None:
        """Derive slab formwork (soffit + edge)."""
        rcc_qty = item.get("quantity", 0)
        drawing_ref = item.get("drawing_ref", "STR")

        # Try to get actual dimensions
        slab_thickness_mm = self._extract_slab_thickness(item)

        if slab_thickness_mm > 0:
            # Accurate: soffit area = volume / thickness
            slab_thickness_m = slab_thickness_mm / 1000
            soffit_area = rcc_qty / slab_thickness_m

            # Edge formwork = perimeter x thickness
            # Estimate perimeter as 4 * sqrt(area)
            perimeter_approx = 4 * (soffit_area ** 0.5)
            edge_area = perimeter_approx * slab_thickness_m
        else:
            # Use constant
            soffit_area = rcc_qty * self.constants.slab_formwork_sqm_per_cum
            edge_area = soffit_area * 0.1  # Approx 10% of soffit

        # Slab soffit formwork
        self.derived_items.append({
            "item_id": f"FW-SLAB-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for RCC slab soffit (derived from {item.get('item_id', 'RCC')})",
            "quantity": round(soffit_area, 2),
            "unit": "sqm",
            "rate": 380,  # Typical rate INR/sqm
            "amount": round(soffit_area * 380, 2),
            "package": "rcc",
            "drawing_ref": drawing_ref,
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"RCC {rcc_qty:.2f} cum / {slab_thickness_mm}mm thickness" if slab_thickness_mm else f"RCC {rcc_qty:.2f} cum x {self.constants.slab_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.SLAB_SOFFIT.value,
        })

        # Slab edge formwork
        if edge_area > 0:
            self.derived_items.append({
                "item_id": f"FW-SLAB-EDGE-{len(self.derived_items)+1:03d}",
                "description": f"Centering and shuttering for RCC slab edges",
                "quantity": round(edge_area, 2),
                "unit": "sqm",
                "rate": 320,
                "amount": round(edge_area * 320, 2),
                "package": "rcc",
                "drawing_ref": drawing_ref,
                "derived_from": item.get("item_id", ""),
                "derivation_method": f"Slab perimeter x {slab_thickness_mm}mm" if slab_thickness_mm else "Estimated 10% of soffit",
                "formwork_type": FormworkType.SLAB_EDGE.value,
            })

    def _derive_beam_formwork(
        self,
        item: Dict[str, Any],
        structural: Dict[str, Any] = None
    ) -> None:
        """Derive beam formwork (bottom + sides)."""
        rcc_qty = item.get("quantity", 0)
        drawing_ref = item.get("drawing_ref", "STR")

        # Try to get beam dimensions
        beam_width_mm, beam_depth_mm = self._extract_beam_dimensions(item)

        if beam_width_mm > 0 and beam_depth_mm > 0:
            beam_width_m = beam_width_mm / 1000
            beam_depth_m = beam_depth_mm / 1000

            # Length = volume / (width x depth)
            beam_length = rcc_qty / (beam_width_m * beam_depth_m) if beam_width_m * beam_depth_m > 0 else 0

            # Bottom formwork
            bottom_area = beam_length * beam_width_m

            # Side formwork (both sides) - depth below slab
            exposed_depth = beam_depth_m - 0.15  # Assume 150mm in slab
            side_area = beam_length * exposed_depth * 2
        else:
            # Use constant
            total_fw = rcc_qty * self.constants.beam_formwork_sqm_per_cum
            bottom_area = total_fw * 0.25  # 25% bottom
            side_area = total_fw * 0.75  # 75% sides

        # Beam bottom formwork
        self.derived_items.append({
            "item_id": f"FW-BEAM-BOT-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for beam bottom (derived from {item.get('item_id', 'RCC')})",
            "quantity": round(bottom_area, 2),
            "unit": "sqm",
            "rate": 420,
            "amount": round(bottom_area * 420, 2),
            "package": "rcc",
            "drawing_ref": drawing_ref,
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"Beam {beam_width_mm}x{beam_depth_mm}mm" if beam_width_mm else f"Constant {self.constants.beam_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.BEAM_BOTTOM.value,
        })

        # Beam side formwork
        self.derived_items.append({
            "item_id": f"FW-BEAM-SIDE-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for beam sides (both sides)",
            "quantity": round(side_area, 2),
            "unit": "sqm",
            "rate": 400,
            "amount": round(side_area * 400, 2),
            "package": "rcc",
            "drawing_ref": drawing_ref,
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"Beam sides 2 x L x {beam_depth_mm-150}mm" if beam_width_mm else "Estimated",
            "formwork_type": FormworkType.BEAM_SIDES.value,
        })

    def _derive_column_formwork(
        self,
        item: Dict[str, Any],
        structural: Dict[str, Any] = None
    ) -> None:
        """Derive column formwork (all sides)."""
        rcc_qty = item.get("quantity", 0)
        drawing_ref = item.get("drawing_ref", "STR")

        # Try to get column dimensions
        col_width_mm, col_depth_mm = self._extract_column_dimensions(item)
        col_height_m = 3.0  # Default floor height

        if col_width_mm > 0:
            col_width_m = col_width_mm / 1000
            col_depth_m = (col_depth_mm or col_width_mm) / 1000

            # Perimeter
            perimeter = 2 * (col_width_m + col_depth_m)

            # Number of columns = volume / (width x depth x height)
            num_columns = rcc_qty / (col_width_m * col_depth_m * col_height_m) if col_width_m * col_depth_m > 0 else 0

            # Total formwork
            fw_area = num_columns * perimeter * col_height_m
        else:
            fw_area = rcc_qty * self.constants.column_formwork_sqm_per_cum

        self.derived_items.append({
            "item_id": f"FW-COL-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for RCC columns (derived from {item.get('item_id', 'RCC')})",
            "quantity": round(fw_area, 2),
            "unit": "sqm",
            "rate": 450,
            "amount": round(fw_area * 450, 2),
            "package": "rcc",
            "drawing_ref": drawing_ref,
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"Column {col_width_mm}x{col_depth_mm}mm x {col_height_m}m" if col_width_mm else f"Constant {self.constants.column_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.COLUMN.value,
        })

    def _derive_staircase_formwork(self, item: Dict[str, Any]) -> None:
        """Derive staircase formwork."""
        rcc_qty = item.get("quantity", 0)
        fw_area = rcc_qty * self.constants.staircase_formwork_sqm_per_cum

        self.derived_items.append({
            "item_id": f"FW-STAIR-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for staircase (waist slab and risers)",
            "quantity": round(fw_area, 2),
            "unit": "sqm",
            "rate": 550,  # Staircases are more complex
            "amount": round(fw_area * 550, 2),
            "package": "rcc",
            "drawing_ref": item.get("drawing_ref", "STR"),
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"RCC {rcc_qty:.2f} cum x {self.constants.staircase_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.STAIRCASE.value,
        })

    def _derive_footing_formwork(
        self,
        item: Dict[str, Any],
        structural: Dict[str, Any] = None
    ) -> None:
        """Derive footing formwork (sides only)."""
        rcc_qty = item.get("quantity", 0)
        fw_area = rcc_qty * self.constants.footing_formwork_sqm_per_cum

        self.derived_items.append({
            "item_id": f"FW-FTG-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for foundation/footing sides",
            "quantity": round(fw_area, 2),
            "unit": "sqm",
            "rate": 280,
            "amount": round(fw_area * 280, 2),
            "package": "rcc",
            "drawing_ref": item.get("drawing_ref", "STR"),
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"RCC {rcc_qty:.2f} cum x {self.constants.footing_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.FOOTING.value,
        })

    def _derive_lintel_formwork(self, item: Dict[str, Any]) -> None:
        """Derive lintel formwork."""
        rcc_qty = item.get("quantity", 0)
        fw_area = rcc_qty * self.constants.lintel_formwork_sqm_per_cum

        self.derived_items.append({
            "item_id": f"FW-LINT-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for lintels",
            "quantity": round(fw_area, 2),
            "unit": "sqm",
            "rate": 380,
            "amount": round(fw_area * 380, 2),
            "package": "rcc",
            "drawing_ref": item.get("drawing_ref", "ARC"),
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"RCC {rcc_qty:.2f} cum x {self.constants.lintel_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.LINTEL.value,
        })

    def _derive_chajja_formwork(self, item: Dict[str, Any]) -> None:
        """Derive chajja/sunshade formwork."""
        rcc_qty = item.get("quantity", 0)
        fw_area = rcc_qty * self.constants.chajja_formwork_sqm_per_cum

        self.derived_items.append({
            "item_id": f"FW-CHAJ-{len(self.derived_items)+1:03d}",
            "description": f"Centering and shuttering for chajja/sunshade",
            "quantity": round(fw_area, 2),
            "unit": "sqm",
            "rate": 420,
            "amount": round(fw_area * 420, 2),
            "package": "rcc",
            "drawing_ref": item.get("drawing_ref", "ARC"),
            "derived_from": item.get("item_id", ""),
            "derivation_method": f"RCC {rcc_qty:.2f} cum x {self.constants.chajja_formwork_sqm_per_cum} sqm/cum",
            "formwork_type": FormworkType.CHAJJA.value,
        })

    def _derive_staging(self) -> None:
        """Derive staging/propping for slabs."""
        # Sum slab formwork area
        slab_fw_area = sum(
            item.get("quantity", 0)
            for item in self.derived_items
            if item.get("formwork_type") == FormworkType.SLAB_SOFFIT.value
        )

        if slab_fw_area > 0:
            staging_area = slab_fw_area * self.constants.staging_sqm_per_formwork_sqm

            self.derived_items.append({
                "item_id": f"STG-{len(self.derived_items)+1:03d}",
                "description": "Staging/propping for slab formwork (steel props/scaffolding)",
                "quantity": round(staging_area, 2),
                "unit": "sqm",
                "rate": 85,  # Monthly rental rate
                "amount": round(staging_area * 85, 2),
                "package": "rcc",
                "drawing_ref": "STR",
                "derived_from": "slab_formwork",
                "derivation_method": f"Slab soffit area {slab_fw_area:.2f} sqm x {self.constants.staging_sqm_per_formwork_sqm}",
                "formwork_type": "staging",
            })

    def _extract_slab_thickness(self, item: Dict[str, Any]) -> float:
        """Extract slab thickness from description."""
        desc = item.get("description", "").lower()

        # Common slab thicknesses
        if "200" in desc or "200mm" in desc:
            return 200
        elif "175" in desc:
            return 175
        elif "150" in desc or "150mm" in desc:
            return 150
        elif "125" in desc:
            return 125
        elif "100" in desc:
            return 100

        # Default 150mm
        return 150

    def _extract_beam_dimensions(self, item: Dict[str, Any]) -> Tuple[float, float]:
        """Extract beam dimensions (width x depth) from description."""
        desc = item.get("description", "").lower()

        # Try to find patterns like "230x450" or "230 x 450"
        import re
        pattern = r"(\d{3})\s*[x×]\s*(\d{3})"
        match = re.search(pattern, desc)

        if match:
            return float(match.group(1)), float(match.group(2))

        # Common beam sizes
        if "230" in desc and "450" in desc:
            return 230, 450
        elif "230" in desc and "600" in desc:
            return 230, 600
        elif "300" in desc and "600" in desc:
            return 300, 600

        return 0, 0

    def _extract_column_dimensions(self, item: Dict[str, Any]) -> Tuple[float, float]:
        """Extract column dimensions from description."""
        desc = item.get("description", "").lower()

        import re
        pattern = r"(\d{3})\s*[x×]\s*(\d{3})"
        match = re.search(pattern, desc)

        if match:
            return float(match.group(1)), float(match.group(2))

        # Common column sizes
        if "450" in desc:
            return 450, 450
        elif "400" in desc:
            return 400, 400
        elif "300" in desc:
            return 300, 300
        elif "230" in desc:
            return 230, 230

        return 0, 0

    def get_summary(self) -> Dict[str, Any]:
        """Get formwork derivation summary."""
        total_formwork = sum(item.get("quantity", 0) for item in self.derived_items if "FW-" in item.get("item_id", ""))
        total_value = sum(item.get("amount", 0) for item in self.derived_items)

        by_type = {}
        for item in self.derived_items:
            fw_type = item.get("formwork_type", "other")
            if fw_type not in by_type:
                by_type[fw_type] = {"qty": 0, "amount": 0}
            by_type[fw_type]["qty"] += item.get("quantity", 0)
            by_type[fw_type]["amount"] += item.get("amount", 0)

        return {
            "total_formwork_sqm": round(total_formwork, 2),
            "total_value": round(total_value, 2),
            "items_derived": len(self.derived_items),
            "by_type": by_type,
        }
