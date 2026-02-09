"""
Bill of Materials (BOM) Derivation Engine - Derive material quantities from BOQ.

This module provides:
- Material consumption calculation from BOQ quantities
- CPWD/DSR based norms
- Material aggregation across packages
- Procurement quantity estimation
- India-specific material specifications

Based on:
- CPWD DSR 2024
- IS 456 for concrete
- IS 1200 for measurement
- Standard industry norms
"""

from .calculator import MaterialCalculator, MaterialItem, MaterialEstimate
from .aggregator import MaterialAggregator, AggregatedMaterial
from .exporter import BOMExporter

__all__ = [
    "MaterialCalculator",
    "MaterialItem",
    "MaterialEstimate",
    "MaterialAggregator",
    "AggregatedMaterial",
    "BOMExporter",
]


def run_bom_engine(
    project_id: str,
    boq_entries: list,
    scope_register: dict,
    output_dir,
) -> dict:
    """
    Run the complete BOM derivation pipeline.

    Args:
        project_id: Project identifier
        boq_entries: BOQ entries list
        scope_register: Scope register dict
        output_dir: Output directory

    Returns:
        Dict with BOM results
    """
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)

    # 1. Calculate materials from BOQ
    logger.info("Calculating material requirements...")
    calculator = MaterialCalculator()
    material_estimates = calculator.calculate_all(boq_entries)
    logger.info(f"Calculated materials for {len(material_estimates)} BOQ items")

    # 2. Aggregate materials
    logger.info("Aggregating materials...")
    aggregator = MaterialAggregator()
    aggregated = aggregator.aggregate(material_estimates)
    logger.info(f"Aggregated into {len(aggregated)} material categories")

    # 3. Export results
    logger.info("Exporting BOM...")
    exporter = BOMExporter()
    output_paths = exporter.export_all(
        project_id,
        material_estimates,
        aggregated,
        output_dir,
    )

    # Calculate totals
    total_cement = sum(
        m.quantity for m in aggregated
        if "cement" in m.material_name.lower() and m.unit == "bags"
    )
    total_steel = sum(
        m.quantity for m in aggregated
        if "steel" in m.material_name.lower() and m.unit == "kg"
    )
    total_sand = sum(
        m.quantity for m in aggregated
        if "sand" in m.material_name.lower() and m.unit == "cum"
    )

    return {
        "boq_items_processed": len(material_estimates),
        "material_categories": len(aggregated),
        "total_cement_bags": round(total_cement, 0),
        "total_steel_kg": round(total_steel, 0),
        "total_sand_cum": round(total_sand, 2),
        "output_paths": {k: str(v) for k, v in output_paths.items()},
    }
