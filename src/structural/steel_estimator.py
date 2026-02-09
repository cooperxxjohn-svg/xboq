"""
Steel Estimation Module
Estimates reinforcement steel quantities using either:
1. Bar Bending Schedule (BBS) - if rebar details available
2. kg/m³ rule-of-thumb factors - for quick estimation

Follows IS 456:2000 guidelines and Indian construction practices.
"""

import logging
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


# Standard TMT bar weights (kg/m) as per IS 1786
BAR_WEIGHTS = {
    6: 0.222,   # Y6/T6
    8: 0.395,   # Y8/T8
    10: 0.617,  # Y10/T10
    12: 0.888,  # Y12/T12
    16: 1.578,  # Y16/T16
    20: 2.466,  # Y20/T20
    25: 3.853,  # Y25/T25
    32: 6.313,  # Y32/T32
}

# Lapping/wastage factors
LAPPING_FACTOR = 1.08  # 8% extra for laps
WASTAGE_FACTOR = 1.03  # 3% wastage


@dataclass
class RebarDetail:
    """Details of a single rebar type."""
    bar_mark: str  # e.g., "A", "B"
    bar_dia: int   # mm
    no_of_bars: int
    length_mm: float
    shape_code: str = "00"  # IS standard shape code
    weight_kg: float = 0.0


@dataclass
class BBSEntry:
    """Bar Bending Schedule entry for an element."""
    element_id: str
    element_type: str
    label: str

    # Main reinforcement
    main_bars: List[RebarDetail] = field(default_factory=list)

    # Secondary/distribution steel (for slabs)
    distribution_bars: List[RebarDetail] = field(default_factory=list)

    # Stirrups/links
    stirrups: List[RebarDetail] = field(default_factory=list)

    # Totals
    main_steel_kg: float = 0.0
    stirrup_steel_kg: float = 0.0
    total_steel_kg: float = 0.0

    # With factors
    with_lap_kg: float = 0.0
    with_wastage_kg: float = 0.0


@dataclass
class SteelEstimationResult:
    """Complete steel estimation result."""
    bbs_entries: List[BBSEntry]

    # Summary by element type
    column_steel_kg: float = 0.0
    beam_steel_kg: float = 0.0
    footing_steel_kg: float = 0.0
    slab_steel_kg: float = 0.0

    # Grand totals
    total_main_kg: float = 0.0
    total_stirrup_kg: float = 0.0
    total_steel_kg: float = 0.0

    # By diameter
    by_diameter: Dict[int, float] = field(default_factory=dict)

    # Method used
    method: str = "kg_per_m3"  # "bbs" or "kg_per_m3"

    # Factors applied
    lap_factor: float = 1.0
    wastage_factor: float = 1.0

    warnings: List[str] = field(default_factory=list)


class SteelEstimator:
    """
    Estimates reinforcement steel for RCC elements.

    Two methods:
    1. BBS Method: When rebar details are available (from drawings/schedules)
    2. kg/m³ Method: Using rule-of-thumb factors per element type
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize estimator."""
        self.config = self._load_config(config_path)

        # Steel factors from assumptions
        self.steel_factors = self.config.get('steel', {}).get('kg_per_m3', {})

        # Default rebar configurations
        self.default_rebar = self.config.get('steel', {}).get('default_rebar', {})

    def _load_config(self, path: Optional[Path]) -> Dict:
        """Load configuration."""
        default_path = Path(__file__).parent.parent.parent / "rules" / "assumptions.yaml"
        try:
            with open(path or default_path) as f:
                return yaml.safe_load(f)
        except Exception:
            return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'steel': {
                'kg_per_m3': {
                    'column': {'min': 150, 'max': 250, 'default': 200},
                    'beam': {'min': 120, 'max': 180, 'default': 150},
                    'footing': {'min': 70, 'max': 120, 'default': 90},
                    'slab': {'min': 80, 'max': 120, 'default': 90}
                },
                'default_rebar': {
                    'column': {
                        'main_dia': 16,
                        'stirrup_dia': 8,
                        'stirrup_spacing': 150
                    },
                    'beam': {
                        'main_dia': 16,
                        'stirrup_dia': 8,
                        'stirrup_spacing': 150
                    },
                    'footing': {
                        'main_dia': 12,
                        'spacing': 150
                    },
                    'slab': {
                        'main_dia': 10,
                        'dist_dia': 8,
                        'main_spacing': 150,
                        'dist_spacing': 200
                    }
                }
            }
        }

    def estimate_from_volume(
        self,
        element_type: str,
        volume_m3: float,
        building_type: str = "residential"
    ) -> Tuple[float, str]:
        """
        Estimate steel using kg/m³ factor.

        Args:
            element_type: "column", "beam", "footing", "slab"
            volume_m3: Concrete volume
            building_type: Building type for factor selection

        Returns:
            (steel_kg, factor_used)
        """
        factors = self.steel_factors.get(element_type, {'default': 150})
        factor = factors.get('default', 150)

        steel_kg = volume_m3 * factor
        return steel_kg, f"{factor} kg/m³"

    def estimate_column_bbs(
        self,
        width_mm: int,
        depth_mm: int,
        height_mm: int,
        main_bars: int = None,
        main_dia: int = None,
        stirrup_dia: int = None,
        stirrup_spacing: int = None,
        label: str = ""
    ) -> BBSEntry:
        """
        Generate BBS for a column.

        Args:
            width_mm, depth_mm, height_mm: Column dimensions
            main_bars: Number of main bars (default: calculated)
            main_dia: Main bar diameter (default: from config)
            stirrup_dia: Stirrup diameter (default: from config)
            stirrup_spacing: Stirrup spacing (default: from config)
            label: Column label

        Returns:
            BBSEntry with detailed breakdown
        """
        config = self.default_rebar.get('column', {})

        main_dia = main_dia or config.get('main_dia', 16)
        stirrup_dia = stirrup_dia or config.get('stirrup_dia', 8)
        stirrup_spacing = stirrup_spacing or config.get('stirrup_spacing', 150)

        # Calculate number of main bars if not provided
        # Minimum: 4 for rectangular, based on code requirements
        if main_bars is None:
            # Rough: 1 bar per 150mm perimeter
            perimeter = 2 * (width_mm + depth_mm)
            main_bars = max(4, int(perimeter / 150))

        # Main bar length = column height + development length
        dev_length = 50 * main_dia  # Approximate development length
        main_length = height_mm + dev_length

        # Main bar weight
        main_weight = main_bars * (main_length / 1000) * BAR_WEIGHTS.get(main_dia, 1.58)

        main_detail = RebarDetail(
            bar_mark="A",
            bar_dia=main_dia,
            no_of_bars=main_bars,
            length_mm=main_length,
            weight_kg=main_weight
        )

        # Stirrup calculation
        # Number of stirrups
        no_stirrups = int(height_mm / stirrup_spacing) + 1

        # Stirrup perimeter (hook length = 10d on each side)
        clear_cover = 40  # mm
        stirrup_width = width_mm - 2 * clear_cover
        stirrup_depth = depth_mm - 2 * clear_cover
        hook_length = 2 * (10 * stirrup_dia + 75)  # Two 135° hooks
        stirrup_length = 2 * (stirrup_width + stirrup_depth) + hook_length

        stirrup_weight = no_stirrups * (stirrup_length / 1000) * BAR_WEIGHTS.get(stirrup_dia, 0.395)

        stirrup_detail = RebarDetail(
            bar_mark="B",
            bar_dia=stirrup_dia,
            no_of_bars=no_stirrups,
            length_mm=stirrup_length,
            weight_kg=stirrup_weight
        )

        total_kg = main_weight + stirrup_weight

        return BBSEntry(
            element_id=label,
            element_type="column",
            label=label,
            main_bars=[main_detail],
            stirrups=[stirrup_detail],
            main_steel_kg=main_weight,
            stirrup_steel_kg=stirrup_weight,
            total_steel_kg=total_kg,
            with_lap_kg=total_kg * LAPPING_FACTOR,
            with_wastage_kg=total_kg * LAPPING_FACTOR * WASTAGE_FACTOR
        )

    def estimate_beam_bbs(
        self,
        width_mm: int,
        depth_mm: int,
        span_mm: int,
        top_bars: int = None,
        bottom_bars: int = None,
        main_dia: int = None,
        stirrup_dia: int = None,
        stirrup_spacing: int = None,
        label: str = ""
    ) -> BBSEntry:
        """
        Generate BBS for a beam.
        """
        config = self.default_rebar.get('beam', {})

        main_dia = main_dia or config.get('main_dia', 16)
        stirrup_dia = stirrup_dia or config.get('stirrup_dia', 8)
        stirrup_spacing = stirrup_spacing or config.get('stirrup_spacing', 150)

        # Default bar counts based on beam width
        if top_bars is None:
            top_bars = max(2, int(width_mm / 100))
        if bottom_bars is None:
            bottom_bars = max(2, int(width_mm / 80))

        # Bar lengths (include anchorage at supports)
        dev_length = 50 * main_dia
        bar_length = span_mm + 2 * dev_length

        # Top bars (negative moment at supports)
        top_weight = top_bars * (bar_length / 1000) * BAR_WEIGHTS.get(main_dia, 1.58)
        top_detail = RebarDetail(
            bar_mark="A",
            bar_dia=main_dia,
            no_of_bars=top_bars,
            length_mm=bar_length,
            weight_kg=top_weight
        )

        # Bottom bars (positive moment at midspan)
        bottom_weight = bottom_bars * (bar_length / 1000) * BAR_WEIGHTS.get(main_dia, 1.58)
        bottom_detail = RebarDetail(
            bar_mark="B",
            bar_dia=main_dia,
            no_of_bars=bottom_bars,
            length_mm=bar_length,
            weight_kg=bottom_weight
        )

        main_weight = top_weight + bottom_weight

        # Stirrups
        clear_cover = 40
        no_stirrups = int(span_mm / stirrup_spacing) + 1

        stirrup_width = width_mm - 2 * clear_cover
        stirrup_depth = depth_mm - 2 * clear_cover
        hook_length = 2 * (10 * stirrup_dia + 75)
        stirrup_length = 2 * (stirrup_width + stirrup_depth) + hook_length

        stirrup_weight = no_stirrups * (stirrup_length / 1000) * BAR_WEIGHTS.get(stirrup_dia, 0.395)

        stirrup_detail = RebarDetail(
            bar_mark="C",
            bar_dia=stirrup_dia,
            no_of_bars=no_stirrups,
            length_mm=stirrup_length,
            weight_kg=stirrup_weight
        )

        total_kg = main_weight + stirrup_weight

        return BBSEntry(
            element_id=label,
            element_type="beam",
            label=label,
            main_bars=[top_detail, bottom_detail],
            stirrups=[stirrup_detail],
            main_steel_kg=main_weight,
            stirrup_steel_kg=stirrup_weight,
            total_steel_kg=total_kg,
            with_lap_kg=total_kg * LAPPING_FACTOR,
            with_wastage_kg=total_kg * LAPPING_FACTOR * WASTAGE_FACTOR
        )

    def estimate_footing_bbs(
        self,
        length_mm: int,
        width_mm: int,
        depth_mm: int,
        main_dia: int = None,
        spacing_mm: int = None,
        label: str = ""
    ) -> BBSEntry:
        """
        Generate BBS for an isolated footing.
        """
        config = self.default_rebar.get('footing', {})

        main_dia = main_dia or config.get('main_dia', 12)
        spacing_mm = spacing_mm or config.get('spacing', 150)

        clear_cover = 75  # Footings need more cover

        # Bars in length direction
        effective_width = width_mm - 2 * clear_cover
        no_bars_length = int(effective_width / spacing_mm) + 1
        bar_length_long = length_mm - 2 * clear_cover + 2 * (50 * main_dia)  # + development

        long_weight = no_bars_length * (bar_length_long / 1000) * BAR_WEIGHTS.get(main_dia, 0.888)

        long_detail = RebarDetail(
            bar_mark="A",
            bar_dia=main_dia,
            no_of_bars=no_bars_length,
            length_mm=bar_length_long,
            weight_kg=long_weight
        )

        # Bars in width direction
        effective_length = length_mm - 2 * clear_cover
        no_bars_width = int(effective_length / spacing_mm) + 1
        bar_length_short = width_mm - 2 * clear_cover + 2 * (50 * main_dia)

        short_weight = no_bars_width * (bar_length_short / 1000) * BAR_WEIGHTS.get(main_dia, 0.888)

        short_detail = RebarDetail(
            bar_mark="B",
            bar_dia=main_dia,
            no_of_bars=no_bars_width,
            length_mm=bar_length_short,
            weight_kg=short_weight
        )

        total_kg = long_weight + short_weight

        return BBSEntry(
            element_id=label,
            element_type="footing",
            label=label,
            main_bars=[long_detail, short_detail],
            main_steel_kg=total_kg,
            total_steel_kg=total_kg,
            with_lap_kg=total_kg * LAPPING_FACTOR,
            with_wastage_kg=total_kg * LAPPING_FACTOR * WASTAGE_FACTOR
        )

    def estimate_slab_bbs(
        self,
        area_sqm: float,
        thickness_mm: int = 125,
        main_dia: int = None,
        dist_dia: int = None,
        main_spacing: int = None,
        dist_spacing: int = None,
        label: str = "Slab"
    ) -> BBSEntry:
        """
        Generate BBS for a slab (simplified).
        """
        config = self.default_rebar.get('slab', {})

        main_dia = main_dia or config.get('main_dia', 10)
        dist_dia = dist_dia or config.get('dist_dia', 8)
        main_spacing = main_spacing or config.get('main_spacing', 150)
        dist_spacing = dist_spacing or config.get('dist_spacing', 200)

        # Simplified: assume square slab
        side_mm = math.sqrt(area_sqm * 1e6)
        clear_cover = 25

        # Main bars
        effective_span = side_mm - 2 * clear_cover
        no_main_bars = int(effective_span / main_spacing) + 1
        main_length = effective_span + 2 * (50 * main_dia)

        main_weight = no_main_bars * (main_length / 1000) * BAR_WEIGHTS.get(main_dia, 0.617)

        main_detail = RebarDetail(
            bar_mark="A",
            bar_dia=main_dia,
            no_of_bars=no_main_bars,
            length_mm=main_length,
            weight_kg=main_weight
        )

        # Distribution bars
        no_dist_bars = int(effective_span / dist_spacing) + 1
        dist_length = effective_span + 2 * (40 * dist_dia)

        dist_weight = no_dist_bars * (dist_length / 1000) * BAR_WEIGHTS.get(dist_dia, 0.395)

        dist_detail = RebarDetail(
            bar_mark="B",
            bar_dia=dist_dia,
            no_of_bars=no_dist_bars,
            length_mm=dist_length,
            weight_kg=dist_weight
        )

        total_kg = main_weight + dist_weight

        return BBSEntry(
            element_id=label,
            element_type="slab",
            label=label,
            main_bars=[main_detail],
            distribution_bars=[dist_detail],
            main_steel_kg=total_kg,
            total_steel_kg=total_kg,
            with_lap_kg=total_kg * LAPPING_FACTOR,
            with_wastage_kg=total_kg * LAPPING_FACTOR * WASTAGE_FACTOR
        )

    def parse_rebar_text(self, text: str) -> Optional[Dict]:
        """
        Parse rebar specification text.

        Patterns:
        - "4Y16" -> 4 nos Y16
        - "Y12@150" -> Y12 at 150mm c/c
        - "8-Y12@150 B/W" -> 8 nos Y12 at 150mm both ways
        """
        result = {}

        # Pattern: NdiaYXX or N-YXX
        bar_count = re.search(r'(\d+)\s*[-]?\s*[YT]\s*(\d+)', text)
        if bar_count:
            result['count'] = int(bar_count.group(1))
            result['diameter'] = int(bar_count.group(2))

        # Pattern: YXX@NNN
        spacing = re.search(r'[YT]\s*(\d+)\s*@\s*(\d+)', text)
        if spacing:
            result['diameter'] = int(spacing.group(1))
            result['spacing'] = int(spacing.group(2))

        # Both ways indicator
        if 'b/w' in text.lower() or 'both' in text.lower():
            result['both_ways'] = True

        return result if result else None

    def generate_summary(
        self,
        bbs_entries: List[BBSEntry],
        include_factors: bool = True
    ) -> SteelEstimationResult:
        """
        Generate summary from BBS entries.
        """
        result = SteelEstimationResult(
            bbs_entries=bbs_entries,
            lap_factor=LAPPING_FACTOR if include_factors else 1.0,
            wastage_factor=WASTAGE_FACTOR if include_factors else 1.0
        )

        by_diameter = {}

        for entry in bbs_entries:
            weight = entry.with_wastage_kg if include_factors else entry.total_steel_kg

            if entry.element_type == "column":
                result.column_steel_kg += weight
            elif entry.element_type == "beam":
                result.beam_steel_kg += weight
            elif entry.element_type == "footing":
                result.footing_steel_kg += weight
            elif entry.element_type == "slab":
                result.slab_steel_kg += weight

            result.total_main_kg += entry.main_steel_kg
            result.total_stirrup_kg += entry.stirrup_steel_kg

            # Track by diameter
            for bar in entry.main_bars + entry.distribution_bars + entry.stirrups:
                dia = bar.bar_dia
                by_diameter[dia] = by_diameter.get(dia, 0) + bar.weight_kg

        result.total_steel_kg = (
            result.column_steel_kg +
            result.beam_steel_kg +
            result.footing_steel_kg +
            result.slab_steel_kg
        )

        result.by_diameter = by_diameter
        result.method = "bbs"

        return result


def estimate_steel(
    element_type: str,
    volume_m3: float
) -> Tuple[float, str]:
    """Convenience function for kg/m³ estimation."""
    estimator = SteelEstimator()
    return estimator.estimate_from_volume(element_type, volume_m3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    estimator = SteelEstimator()

    print("\n=== STEEL ESTIMATION EXAMPLES ===\n")

    # Column BBS
    col_bbs = estimator.estimate_column_bbs(
        width_mm=230,
        depth_mm=450,
        height_mm=3000,
        label="C1"
    )
    print(f"Column C1 (230x450x3000mm):")
    print(f"  Main bars: {col_bbs.main_bars[0].no_of_bars} nos Y{col_bbs.main_bars[0].bar_dia}")
    print(f"  Stirrups: {col_bbs.stirrups[0].no_of_bars} nos Y{col_bbs.stirrups[0].bar_dia}")
    print(f"  Main steel: {col_bbs.main_steel_kg:.2f} kg")
    print(f"  Stirrup steel: {col_bbs.stirrup_steel_kg:.2f} kg")
    print(f"  Total: {col_bbs.total_steel_kg:.2f} kg")
    print(f"  With lap+wastage: {col_bbs.with_wastage_kg:.2f} kg")

    # Beam BBS
    print()
    beam_bbs = estimator.estimate_beam_bbs(
        width_mm=230,
        depth_mm=450,
        span_mm=4000,
        label="B1"
    )
    print(f"Beam B1 (230x450, 4m span):")
    print(f"  Top bars: {beam_bbs.main_bars[0].no_of_bars} nos Y{beam_bbs.main_bars[0].bar_dia}")
    print(f"  Bottom bars: {beam_bbs.main_bars[1].no_of_bars} nos Y{beam_bbs.main_bars[1].bar_dia}")
    print(f"  Stirrups: {beam_bbs.stirrups[0].no_of_bars} nos Y{beam_bbs.stirrups[0].bar_dia}")
    print(f"  Total: {beam_bbs.total_steel_kg:.2f} kg")

    # Footing BBS
    print()
    ftg_bbs = estimator.estimate_footing_bbs(
        length_mm=1500,
        width_mm=1500,
        depth_mm=450,
        label="F1"
    )
    print(f"Footing F1 (1500x1500x450mm):")
    print(f"  Long bars: {ftg_bbs.main_bars[0].no_of_bars} nos Y{ftg_bbs.main_bars[0].bar_dia}")
    print(f"  Short bars: {ftg_bbs.main_bars[1].no_of_bars} nos Y{ftg_bbs.main_bars[1].bar_dia}")
    print(f"  Total: {ftg_bbs.total_steel_kg:.2f} kg")

    # kg/m³ comparison
    print()
    vol = (0.23 * 0.45 * 3.0)  # Column volume
    kg_estimate, factor = estimator.estimate_from_volume("column", vol)
    print(f"Column kg/m³ estimate: {kg_estimate:.2f} kg ({factor})")
    print(f"BBS estimate: {col_bbs.total_steel_kg:.2f} kg")
    print(f"Difference: {abs(kg_estimate - col_bbs.total_steel_kg):.2f} kg")
