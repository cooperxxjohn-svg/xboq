"""
Quantity Computation Engine
Calculates concrete volumes and steel quantities for structural elements.
Supports both exact calculations (from structural drawings) and
assumptions-based estimates (from architectural plans).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import yaml

from .detect_columns import DetectedColumn, ColumnDetectionResult
from .detect_beams import DetectedBeam, BeamDetectionResult
from .detect_footings import DetectedFooting, FootingDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class ElementQuantity:
    """Quantity for a single element."""
    element_id: str
    element_type: str  # "column", "beam", "footing", "slab"
    label: str
    count: int = 1

    # Dimensions (mm)
    width: int = 0
    depth: int = 0
    length: int = 0  # height for columns, span for beams

    # Computed volumes (m³)
    concrete_volume_m3: float = 0.0

    # Steel quantities (kg)
    steel_main_kg: float = 0.0
    steel_stirrup_kg: float = 0.0
    steel_total_kg: float = 0.0

    # Source information
    size_source: str = "assumption"  # "schedule", "callout", "detection", "assumption"
    height_source: str = "assumption"
    steel_source: str = "kg_per_m3"  # "bbs", "kg_per_m3", "manual"

    # Assumptions used
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'element_id': self.element_id,
            'type': self.element_type,
            'label': self.label,
            'count': self.count,
            'dimensions_mm': {
                'width': self.width,
                'depth': self.depth,
                'length': self.length
            },
            'concrete_m3': round(self.concrete_volume_m3, 4),
            'steel_kg': {
                'main': round(self.steel_main_kg, 2),
                'stirrup': round(self.steel_stirrup_kg, 2),
                'total': round(self.steel_total_kg, 2)
            },
            'sources': {
                'size': self.size_source,
                'height': self.height_source,
                'steel': self.steel_source
            },
            'assumptions': self.assumptions
        }


@dataclass
class QuantitySummary:
    """Summary of all quantities."""
    # Element counts
    column_count: int = 0
    beam_count: int = 0
    footing_count: int = 0
    slab_count: int = 0

    # Concrete volumes by type (m³)
    column_concrete_m3: float = 0.0
    beam_concrete_m3: float = 0.0
    footing_concrete_m3: float = 0.0
    slab_concrete_m3: float = 0.0
    total_concrete_m3: float = 0.0

    # Steel quantities by type (kg)
    column_steel_kg: float = 0.0
    beam_steel_kg: float = 0.0
    footing_steel_kg: float = 0.0
    slab_steel_kg: float = 0.0
    total_steel_kg: float = 0.0

    # Conversion to tonnes
    @property
    def total_steel_tonnes(self) -> float:
        return self.total_steel_kg / 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'counts': {
                'columns': self.column_count,
                'beams': self.beam_count,
                'footings': self.footing_count,
                'slabs': self.slab_count
            },
            'concrete_m3': {
                'columns': round(self.column_concrete_m3, 3),
                'beams': round(self.beam_concrete_m3, 3),
                'footings': round(self.footing_concrete_m3, 3),
                'slabs': round(self.slab_concrete_m3, 3),
                'total': round(self.total_concrete_m3, 3)
            },
            'steel_kg': {
                'columns': round(self.column_steel_kg, 2),
                'beams': round(self.beam_steel_kg, 2),
                'footings': round(self.footing_steel_kg, 2),
                'slabs': round(self.slab_steel_kg, 2),
                'total': round(self.total_steel_kg, 2)
            },
            'steel_tonnes': round(self.total_steel_tonnes, 3)
        }


@dataclass
class QuantityResult:
    """Complete quantity computation result."""
    elements: List[ElementQuantity]
    summary: QuantitySummary
    mode: str = "assumption"  # "structural" or "assumption"
    floors: int = 1
    warnings: List[str] = field(default_factory=list)
    assumptions_used: List[Dict[str, Any]] = field(default_factory=list)


class QuantityEngine:
    """
    Computes concrete volumes and steel quantities.

    Two modes:
    1. STRUCTURAL MODE: Uses detected elements with actual sizes from schedules
    2. ASSUMPTION MODE: Uses detected room layout to estimate structure
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize engine with configuration."""
        self.config = self._load_config(config_path)

        # Steel kg/m³ factors from assumptions
        self.steel_factors = self.config.get('steel', {}).get('kg_per_m3', {})

        # Default sizes
        self.default_sizes = self.config.get('default_sizes', {})

        # Storey heights
        self.storey_heights = self.config.get('storey_heights', {})

    def _load_config(self, path: Optional[Path]) -> Dict:
        """Load configuration from assumptions.yaml."""
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
                    'column': {'default': 200},
                    'beam': {'default': 150},
                    'footing': {'default': 90},
                    'slab': {'default': 90}
                }
            },
            'default_sizes': {
                'column': {
                    'residential_small': {'width': 230, 'depth': 230},
                    'residential_medium': {'width': 230, 'depth': 450},
                    'residential_large': {'width': 300, 'depth': 600}
                },
                'beam': {
                    'floor_beam_small': {'width': 230, 'depth': 380},
                    'floor_beam_medium': {'width': 230, 'depth': 450}
                },
                'footing': {
                    'isolated_small': {'length': 1200, 'width': 1200, 'depth': 300},
                    'isolated_medium': {'length': 1500, 'width': 1500, 'depth': 450}
                },
                'slab': {
                    'residential': {'thickness': 125}
                }
            },
            'storey_heights': {
                'default': 3000,
                'ground': 3300,
                'basement': 2700
            }
        }

    def compute_structural(
        self,
        columns: List[DetectedColumn] = None,
        beams: List[DetectedBeam] = None,
        footings: List[DetectedFooting] = None,
        column_sizes: Dict[str, Tuple[int, int]] = None,
        beam_sizes: Dict[str, Tuple[int, int]] = None,
        footing_sizes: Dict[str, Tuple[int, int, int]] = None,
        floors: int = 1,
        storey_height_mm: int = None
    ) -> QuantityResult:
        """
        Compute quantities from structural detection results.

        Args:
            columns: Detected columns
            beams: Detected beams
            footings: Detected footings
            column_sizes: Size mappings from schedule
            beam_sizes: Size mappings from schedule
            footing_sizes: Size mappings from schedule
            floors: Number of floors
            storey_height_mm: Storey height override

        Returns:
            QuantityResult
        """
        logger.info(f"Computing quantities (structural mode) for {floors} floor(s)")

        elements = []
        warnings = []
        assumptions = []

        height = storey_height_mm or self.storey_heights.get('default', 3000)

        # Process columns
        if columns:
            col_elements, col_warnings, col_assumptions = self._compute_column_quantities(
                columns, column_sizes, floors, height
            )
            elements.extend(col_elements)
            warnings.extend(col_warnings)
            assumptions.extend(col_assumptions)

        # Process beams
        if beams:
            beam_elements, beam_warnings, beam_assumptions = self._compute_beam_quantities(
                beams, beam_sizes, floors
            )
            elements.extend(beam_elements)
            warnings.extend(beam_warnings)
            assumptions.extend(beam_assumptions)

        # Process footings (once, not per floor)
        if footings:
            ftg_elements, ftg_warnings, ftg_assumptions = self._compute_footing_quantities(
                footings, footing_sizes
            )
            elements.extend(ftg_elements)
            warnings.extend(ftg_warnings)
            assumptions.extend(ftg_assumptions)

        # Build summary
        summary = self._build_summary(elements)

        return QuantityResult(
            elements=elements,
            summary=summary,
            mode="structural",
            floors=floors,
            warnings=warnings,
            assumptions_used=assumptions
        )

    def compute_assumption(
        self,
        total_area_sqm: float,
        rooms: List[Dict] = None,
        building_type: str = "residential",
        floors: int = 1,
        column_grid_m: float = 4.0
    ) -> QuantityResult:
        """
        Compute quantities using assumption mode (from architectural plan).

        Args:
            total_area_sqm: Total floor area
            rooms: Detected room data (optional)
            building_type: Type of building for assumptions
            floors: Number of floors
            column_grid_m: Assumed column grid spacing

        Returns:
            QuantityResult with estimated quantities
        """
        logger.info(f"Computing quantities (assumption mode) for {total_area_sqm:.1f} sqm, "
                   f"{floors} floor(s)")

        elements = []
        warnings = []
        assumptions = []

        # Estimate number of columns based on area and grid
        grid_mm = column_grid_m * 1000
        area_sqmm = total_area_sqm * 1e6

        # Rough estimation: columns at grid intersections
        # Perimeter ~ 4 * sqrt(area)
        side_m = math.sqrt(total_area_sqm)
        cols_per_side = max(2, int(side_m / column_grid_m) + 1)
        num_columns = cols_per_side * cols_per_side

        assumptions.append({
            'type': 'column_count',
            'value': num_columns,
            'method': f'Grid {column_grid_m}m x {column_grid_m}m over {total_area_sqm:.1f} sqm'
        })

        # Get default column size
        col_size = self.default_sizes.get('column', {}).get(
            f'{building_type}_medium',
            {'width': 230, 'depth': 450}
        )
        col_width = col_size.get('width', 230)
        col_depth = col_size.get('depth', 450)

        assumptions.append({
            'type': 'column_size',
            'value': f'{col_width}x{col_depth}mm',
            'method': f'{building_type}_medium default'
        })

        # Storey height
        height = self.storey_heights.get('default', 3000)

        # Create column elements
        col_steel_factor = self.steel_factors.get('column', {}).get('default', 200)

        for i in range(num_columns):
            vol_m3 = (col_width / 1000) * (col_depth / 1000) * (height / 1000) * floors
            steel_kg = vol_m3 * col_steel_factor

            elements.append(ElementQuantity(
                element_id=f"C{i+1:03d}",
                element_type="column",
                label=f"C{i+1}",
                count=1,
                width=col_width,
                depth=col_depth,
                length=height * floors,
                concrete_volume_m3=vol_m3,
                steel_total_kg=steel_kg,
                size_source="assumption",
                height_source="assumption",
                steel_source="kg_per_m3",
                assumptions=[
                    f"Size: {col_width}x{col_depth}mm (residential default)",
                    f"Height: {height}mm x {floors} floors",
                    f"Steel: {col_steel_factor} kg/m³"
                ]
            ))

        # Estimate beams - roughly 2 beams per column
        num_beams = num_columns * 2
        beam_size = self.default_sizes.get('beam', {}).get(
            'floor_beam_medium',
            {'width': 230, 'depth': 450}
        )
        beam_width = beam_size.get('width', 230)
        beam_depth = beam_size.get('depth', 450)
        beam_length = int(grid_mm * 0.9)  # Slightly less than grid (clear span)

        beam_steel_factor = self.steel_factors.get('beam', {}).get('default', 150)

        for i in range(num_beams):
            vol_m3 = (beam_width / 1000) * (beam_depth / 1000) * (beam_length / 1000) * floors
            steel_kg = vol_m3 * beam_steel_factor

            elements.append(ElementQuantity(
                element_id=f"B{i+1:03d}",
                element_type="beam",
                label=f"B{i+1}",
                count=1,
                width=beam_width,
                depth=beam_depth,
                length=beam_length,
                concrete_volume_m3=vol_m3,
                steel_total_kg=steel_kg,
                size_source="assumption",
                steel_source="kg_per_m3",
                assumptions=[
                    f"Size: {beam_width}x{beam_depth}mm (floor beam default)",
                    f"Span: {beam_length}mm",
                    f"Steel: {beam_steel_factor} kg/m³"
                ]
            ))

        # Estimate footings - one per column
        ftg_size = self.default_sizes.get('footing', {}).get(
            'isolated_medium',
            {'length': 1500, 'width': 1500, 'depth': 450}
        )
        ftg_length = ftg_size.get('length', 1500)
        ftg_width = ftg_size.get('width', 1500)
        ftg_depth = ftg_size.get('depth', 450)

        ftg_steel_factor = self.steel_factors.get('footing', {}).get('default', 90)

        for i in range(num_columns):
            vol_m3 = (ftg_length / 1000) * (ftg_width / 1000) * (ftg_depth / 1000)
            steel_kg = vol_m3 * ftg_steel_factor

            elements.append(ElementQuantity(
                element_id=f"F{i+1:03d}",
                element_type="footing",
                label=f"F{i+1}",
                count=1,
                width=ftg_width,
                depth=ftg_depth,
                length=ftg_length,
                concrete_volume_m3=vol_m3,
                steel_total_kg=steel_kg,
                size_source="assumption",
                steel_source="kg_per_m3",
                assumptions=[
                    f"Size: {ftg_length}x{ftg_width}x{ftg_depth}mm (isolated default)",
                    f"Steel: {ftg_steel_factor} kg/m³"
                ]
            ))

        # Estimate slab
        slab_thickness = self.default_sizes.get('slab', {}).get('residential', {}).get('thickness', 125)
        slab_vol_m3 = total_area_sqm * (slab_thickness / 1000) * floors
        slab_steel_factor = self.steel_factors.get('slab', {}).get('default', 90)
        slab_steel_kg = slab_vol_m3 * slab_steel_factor

        elements.append(ElementQuantity(
            element_id="S001",
            element_type="slab",
            label="Slab",
            count=floors,
            width=int(math.sqrt(total_area_sqm) * 1000),
            depth=int(math.sqrt(total_area_sqm) * 1000),
            length=slab_thickness,
            concrete_volume_m3=slab_vol_m3,
            steel_total_kg=slab_steel_kg,
            size_source="assumption",
            steel_source="kg_per_m3",
            assumptions=[
                f"Area: {total_area_sqm:.1f} sqm x {floors} floors",
                f"Thickness: {slab_thickness}mm",
                f"Steel: {slab_steel_factor} kg/m³"
            ]
        ))

        assumptions.extend([
            {'type': 'beam_count', 'value': num_beams, 'method': '2 beams per column'},
            {'type': 'footing_count', 'value': num_columns, 'method': '1 footing per column'},
            {'type': 'slab_thickness', 'value': f'{slab_thickness}mm', 'method': 'residential default'}
        ])

        # Add warning about assumption mode
        warnings.append(
            "Quantities computed using ASSUMPTION MODE. "
            "For accurate takeoff, use structural drawings."
        )

        summary = self._build_summary(elements)

        return QuantityResult(
            elements=elements,
            summary=summary,
            mode="assumption",
            floors=floors,
            warnings=warnings,
            assumptions_used=assumptions
        )

    def _compute_column_quantities(
        self,
        columns: List[DetectedColumn],
        size_mappings: Dict[str, Tuple[int, int]],
        floors: int,
        height_mm: int
    ) -> Tuple[List[ElementQuantity], List[str], List[Dict]]:
        """Compute quantities for columns."""

        elements = []
        warnings = []
        assumptions = []

        # Default column size
        default_size = self.default_sizes.get('column', {}).get(
            'residential_medium', {'width': 230, 'depth': 450}
        )
        steel_factor = self.steel_factors.get('column', {}).get('default', 200)

        # Group columns by label for counting
        label_counts = {}
        for col in columns:
            label = col.label or 'Unknown'
            label_counts[label] = label_counts.get(label, 0) + 1

        for col in columns:
            label = col.label or 'Unknown'

            # Get size from schedule, detection, or default
            if size_mappings and label in size_mappings:
                w, d = size_mappings[label]
                size_source = "schedule"
            elif col.size_mm:
                w, d = col.size_mm
                size_source = "detection"
            else:
                w, d = default_size['width'], default_size['depth']
                size_source = "assumption"
                assumptions.append({
                    'type': 'column_size',
                    'element': label,
                    'value': f'{w}x{d}mm',
                    'method': 'residential_medium default'
                })

            # Compute volume
            vol_m3 = (w / 1000) * (d / 1000) * (height_mm / 1000) * floors
            steel_kg = vol_m3 * steel_factor

            elem_assumptions = []
            if size_source == "assumption":
                elem_assumptions.append(f"Size: {w}x{d}mm (assumption)")
            elem_assumptions.append(f"Height: {height_mm}mm x {floors} floors")
            elem_assumptions.append(f"Steel: {steel_factor} kg/m³")

            elements.append(ElementQuantity(
                element_id=col.column_id,
                element_type="column",
                label=label,
                count=1,
                width=w,
                depth=d,
                length=height_mm * floors,
                concrete_volume_m3=vol_m3,
                steel_total_kg=steel_kg,
                size_source=size_source,
                height_source="assumption" if height_mm == 3000 else "input",
                steel_source="kg_per_m3",
                assumptions=elem_assumptions
            ))

        return elements, warnings, assumptions

    def _compute_beam_quantities(
        self,
        beams: List[DetectedBeam],
        size_mappings: Dict[str, Tuple[int, int]],
        floors: int
    ) -> Tuple[List[ElementQuantity], List[str], List[Dict]]:
        """Compute quantities for beams."""

        elements = []
        warnings = []
        assumptions = []

        default_size = self.default_sizes.get('beam', {}).get(
            'floor_beam_medium', {'width': 230, 'depth': 450}
        )
        steel_factor = self.steel_factors.get('beam', {}).get('default', 150)

        for beam in beams:
            label = beam.label or 'Unknown'

            # Get size
            if size_mappings and label in size_mappings:
                w, d = size_mappings[label]
                size_source = "schedule"
            elif beam.size_mm:
                w, d = beam.size_mm
                size_source = "detection"
            else:
                w, d = default_size['width'], default_size['depth']
                size_source = "assumption"

            # Get length
            length_mm = beam.length_mm or 3000
            if not beam.length_mm:
                assumptions.append({
                    'type': 'beam_length',
                    'element': label,
                    'value': '3000mm',
                    'method': 'default span'
                })

            # Compute volume
            vol_m3 = (w / 1000) * (d / 1000) * (length_mm / 1000) * floors
            steel_kg = vol_m3 * steel_factor

            elem_assumptions = []
            if size_source == "assumption":
                elem_assumptions.append(f"Size: {w}x{d}mm (assumption)")
            elem_assumptions.append(f"Steel: {steel_factor} kg/m³")

            elements.append(ElementQuantity(
                element_id=beam.beam_id,
                element_type="beam",
                label=label,
                count=1,
                width=w,
                depth=d,
                length=int(length_mm),
                concrete_volume_m3=vol_m3,
                steel_total_kg=steel_kg,
                size_source=size_source,
                steel_source="kg_per_m3",
                assumptions=elem_assumptions
            ))

        return elements, warnings, assumptions

    def _compute_footing_quantities(
        self,
        footings: List[DetectedFooting],
        size_mappings: Dict[str, Tuple[int, int, int]]
    ) -> Tuple[List[ElementQuantity], List[str], List[Dict]]:
        """Compute quantities for footings."""

        elements = []
        warnings = []
        assumptions = []

        default_size = self.default_sizes.get('footing', {}).get(
            'isolated_medium', {'length': 1500, 'width': 1500, 'depth': 450}
        )
        steel_factor = self.steel_factors.get('footing', {}).get('default', 90)

        for ftg in footings:
            label = ftg.label or 'Unknown'

            # Get size
            if size_mappings and label in size_mappings:
                l, w, d = size_mappings[label]
                size_source = "schedule"
            elif ftg.size_mm and len(ftg.size_mm) >= 3:
                l, w, d = ftg.size_mm
                size_source = "detection"
            else:
                l = default_size['length']
                w = default_size['width']
                d = default_size['depth']
                size_source = "assumption"

            # Compute volume
            vol_m3 = (l / 1000) * (w / 1000) * (d / 1000)
            steel_kg = vol_m3 * steel_factor

            elem_assumptions = []
            if size_source == "assumption":
                elem_assumptions.append(f"Size: {l}x{w}x{d}mm (assumption)")
            elem_assumptions.append(f"Steel: {steel_factor} kg/m³")

            elements.append(ElementQuantity(
                element_id=ftg.footing_id,
                element_type="footing",
                label=label,
                count=1,
                width=w,
                depth=d,
                length=l,
                concrete_volume_m3=vol_m3,
                steel_total_kg=steel_kg,
                size_source=size_source,
                steel_source="kg_per_m3",
                assumptions=elem_assumptions
            ))

        return elements, warnings, assumptions

    def _build_summary(self, elements: List[ElementQuantity]) -> QuantitySummary:
        """Build summary from element quantities."""

        summary = QuantitySummary()

        for elem in elements:
            if elem.element_type == "column":
                summary.column_count += elem.count
                summary.column_concrete_m3 += elem.concrete_volume_m3
                summary.column_steel_kg += elem.steel_total_kg

            elif elem.element_type == "beam":
                summary.beam_count += elem.count
                summary.beam_concrete_m3 += elem.concrete_volume_m3
                summary.beam_steel_kg += elem.steel_total_kg

            elif elem.element_type == "footing":
                summary.footing_count += elem.count
                summary.footing_concrete_m3 += elem.concrete_volume_m3
                summary.footing_steel_kg += elem.steel_total_kg

            elif elem.element_type == "slab":
                summary.slab_count += elem.count
                summary.slab_concrete_m3 += elem.concrete_volume_m3
                summary.slab_steel_kg += elem.steel_total_kg

        summary.total_concrete_m3 = (
            summary.column_concrete_m3 +
            summary.beam_concrete_m3 +
            summary.footing_concrete_m3 +
            summary.slab_concrete_m3
        )

        summary.total_steel_kg = (
            summary.column_steel_kg +
            summary.beam_steel_kg +
            summary.footing_steel_kg +
            summary.slab_steel_kg
        )

        return summary


def compute_quantities(
    columns: List[DetectedColumn] = None,
    beams: List[DetectedBeam] = None,
    footings: List[DetectedFooting] = None,
    floors: int = 1
) -> QuantityResult:
    """Convenience function to compute quantities."""
    engine = QuantityEngine()
    return engine.compute_structural(
        columns=columns,
        beams=beams,
        footings=footings,
        floors=floors
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test assumption mode
    engine = QuantityEngine()
    result = engine.compute_assumption(
        total_area_sqm=100,  # 100 sqm apartment
        building_type="residential",
        floors=4,
        column_grid_m=4.0
    )

    print("\n=== QUANTITY SUMMARY (Assumption Mode) ===")
    print(f"Area: 100 sqm x 4 floors")
    print(f"\nElement Counts:")
    print(f"  Columns: {result.summary.column_count}")
    print(f"  Beams: {result.summary.beam_count}")
    print(f"  Footings: {result.summary.footing_count}")

    print(f"\nConcrete Volumes (m³):")
    print(f"  Columns: {result.summary.column_concrete_m3:.2f}")
    print(f"  Beams: {result.summary.beam_concrete_m3:.2f}")
    print(f"  Footings: {result.summary.footing_concrete_m3:.2f}")
    print(f"  Slabs: {result.summary.slab_concrete_m3:.2f}")
    print(f"  TOTAL: {result.summary.total_concrete_m3:.2f}")

    print(f"\nSteel (kg):")
    print(f"  Columns: {result.summary.column_steel_kg:.0f}")
    print(f"  Beams: {result.summary.beam_steel_kg:.0f}")
    print(f"  Footings: {result.summary.footing_steel_kg:.0f}")
    print(f"  Slabs: {result.summary.slab_steel_kg:.0f}")
    print(f"  TOTAL: {result.summary.total_steel_kg:.0f} kg ({result.summary.total_steel_tonnes:.2f} tonnes)")

    print(f"\nWarnings:")
    for w in result.warnings:
        print(f"  ⚠️ {w}")
