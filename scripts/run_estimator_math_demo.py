#!/usr/bin/env python3
"""
Estimator Math Engine Demo

Demonstrates the application of Indian method-of-measurement rules
to produce estimator-accurate quantities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from measurement_rules import EstimatorMathEngine, run_estimator_math

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "estimator_math_demo"


def create_sample_boq():
    """Create sample BOQ with gross quantities (before deductions)."""
    return [
        # RCC Items
        {"item_id": "RCC-001", "description": "RCC M25 grade for columns 450x450mm", "quantity": 45.5, "unit": "cum", "rate": 8500, "amount": 386750, "package": "rcc", "drawing_ref": "STR-01"},
        {"item_id": "RCC-002", "description": "RCC M25 grade for beams 230x450mm", "quantity": 38.2, "unit": "cum", "rate": 8200, "amount": 313240, "package": "rcc", "drawing_ref": "STR-01"},
        {"item_id": "RCC-003", "description": "RCC M25 grade for slab 150mm thick", "quantity": 125.0, "unit": "cum", "rate": 7800, "amount": 975000, "package": "rcc", "drawing_ref": "STR-02"},
        {"item_id": "RCC-004", "description": "Steel reinforcement Fe500D", "quantity": 18500, "unit": "kg", "rate": 85, "amount": 1572500, "package": "rcc", "drawing_ref": "STR-01,STR-02"},

        # Masonry Items (gross quantities - before opening deductions)
        {"item_id": "MAS-001", "description": "AAC Block masonry 200mm thick external walls", "quantity": 1850, "unit": "sqm", "rate": 1450, "amount": 2682500, "package": "masonry", "drawing_ref": "ARC-01,ARC-02"},
        {"item_id": "MAS-002", "description": "AAC Block masonry 100mm thick internal partitions", "quantity": 420, "unit": "sqm", "rate": 980, "amount": 411600, "package": "masonry", "drawing_ref": "ARC-01"},

        # Plaster Items (gross - before opening and beam/column deductions)
        {"item_id": "PLS-001", "description": "Cement plaster 12mm internal walls", "quantity": 4200, "unit": "sqm", "rate": 185, "amount": 777000, "package": "masonry", "drawing_ref": "ARC-01,ARC-02"},
        {"item_id": "PLS-002", "description": "Cement plaster 20mm external walls", "quantity": 1100, "unit": "sqm", "rate": 245, "amount": 269500, "package": "masonry", "drawing_ref": "ARC-03"},

        # Paint Items (gross - before opening deductions)
        {"item_id": "PNT-001", "description": "Interior emulsion paint 2 coats", "quantity": 4200, "unit": "sqm", "rate": 85, "amount": 357000, "package": "finishes", "drawing_ref": "INT-01"},
        {"item_id": "PNT-002", "description": "Exterior weatherproof paint", "quantity": 1100, "unit": "sqm", "rate": 95, "amount": 104500, "package": "finishes", "drawing_ref": "INT-01"},

        # Waterproofing
        {"item_id": "WP-001", "description": "APP membrane waterproofing for terrace", "quantity": 520, "unit": "sqm", "rate": 650, "amount": 338000, "package": "waterproof", "drawing_ref": "ARC-ROOF"},

        # Flooring
        {"item_id": "FLR-001", "description": "Vitrified tiles 600x600mm living/dining", "quantity": 380, "unit": "sqm", "rate": 1850, "amount": 703000, "package": "flooring", "drawing_ref": "INT-01"},
        {"item_id": "FLR-002", "description": "Ceramic tiles 300x300mm anti-skid toilets", "quantity": 95, "unit": "sqm", "rate": 1250, "amount": 118750, "package": "flooring", "drawing_ref": "INT-01"},

        # Doors and Windows (these go in, not deducted)
        {"item_id": "DW-001", "description": "Flush door 900x2100mm with frame", "quantity": 24, "unit": "nos", "rate": 12500, "amount": 300000, "package": "doors_windows", "drawing_ref": "D-SCH"},
        {"item_id": "DW-002", "description": "UPVC sliding window 1800x1500mm", "quantity": 12, "unit": "nos", "rate": 18500, "amount": 222000, "package": "doors_windows", "drawing_ref": "W-SCH"},
        {"item_id": "DW-003", "description": "UPVC casement window 1200x1200mm", "quantity": 18, "unit": "nos", "rate": 12800, "amount": 230400, "package": "doors_windows", "drawing_ref": "W-SCH"},

        # External Works (partial - compound wall only)
        {"item_id": "EXT-001", "description": "Compound wall brick 230mm with foundation", "quantity": 85, "unit": "rmt", "rate": 4500, "amount": 382500, "package": "external", "drawing_ref": "SITE-01"},
    ]


def create_sample_openings():
    """Create sample openings schedule for deductions."""
    return [
        # Doors
        {"tag": "D1", "type": "door", "width_mm": 900, "height_mm": 2100, "sill_height_mm": 0, "location": "internal", "count": 20, "frame_type": "wooden"},
        {"tag": "D2", "type": "door", "width_mm": 1000, "height_mm": 2100, "sill_height_mm": 0, "location": "internal", "count": 4, "frame_type": "wooden"},
        {"tag": "MD", "type": "door", "width_mm": 1200, "height_mm": 2400, "sill_height_mm": 0, "location": "external", "count": 1, "frame_type": "teak"},

        # Windows
        {"tag": "W1", "type": "window", "width_mm": 1800, "height_mm": 1500, "sill_height_mm": 900, "location": "external", "count": 12, "frame_type": "UPVC"},
        {"tag": "W2", "type": "window", "width_mm": 1200, "height_mm": 1200, "sill_height_mm": 900, "location": "external", "count": 18, "frame_type": "UPVC"},
        {"tag": "W3", "type": "window", "width_mm": 600, "height_mm": 600, "sill_height_mm": 1800, "location": "internal", "count": 8, "frame_type": "UPVC"},  # Ventilators - below threshold

        # Toilet windows (small - may be below threshold)
        {"tag": "TW", "type": "window", "width_mm": 450, "height_mm": 450, "sill_height_mm": 1500, "location": "internal", "count": 6, "frame_type": "aluminium"},
    ]


def create_structural_elements():
    """Create sample structural elements for beam/column face deductions."""
    return {
        "columns": [
            {"id": "C1", "width_mm": 450, "depth_mm": 450, "height_m": 3.0, "count": 20},
            {"id": "C2", "width_mm": 300, "depth_mm": 600, "height_m": 3.0, "count": 8},
        ],
        "beams": [
            {"id": "B1", "width_mm": 230, "depth_mm": 450, "length_m": 150, "count": 1},  # Total length
            {"id": "B2", "width_mm": 230, "depth_mm": 600, "length_m": 45, "count": 1},
        ],
    }


def main():
    print("=" * 70)
    print("ESTIMATOR MATH ENGINE DEMO")
    print("Indian Method of Measurement Rules")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sample data
    boq_items = create_sample_boq()
    openings = create_sample_openings()
    structural = create_structural_elements()

    project_params = {
        "built_up_area_sqm": 4500,
        "plot_area_sqm": 800,  # Small urban plot
        "num_floors": 3,
        "num_units": 6,
        "duration_months": 14,
        "has_site_plan": False,  # No site plan - generate provisionals
        "has_sewer_connection": True,
    }

    print(f"Input BOQ items: {len(boq_items)}")
    print(f"Openings (doors/windows): {len(openings)}")
    print(f"Built-up area: {project_params['built_up_area_sqm']} sqm")
    print(f"Floors: {project_params['num_floors']}")
    print(f"Duration: {project_params['duration_months']} months")
    print()

    # Run estimator math engine
    print("Processing...")
    results = run_estimator_math(
        boq_items=boq_items,
        openings=openings,
        structural_elements=structural,
        project_params=project_params,
        output_dir=OUTPUT_DIR,
    )

    # Print summary
    summary = results["summary"]
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n1. DEDUCTIONS APPLIED (IS 1200)")
    ded = summary.get("deductions", {})
    print(f"   Deduction entries: {ded.get('entries', 0)}")
    print(f"   Total area deducted: {ded.get('total_deducted', 0):.2f} sqm")

    print("\n2. OPENING DEDUCTIONS")
    od = summary.get("opening_deductions", {})
    print(f"   Doors: {od.get('doors', 0)}")
    print(f"   Windows: {od.get('windows', 0)}")
    print(f"   Door area: {od.get('total_door_area', 0):.2f} sqm")
    print(f"   Window area: {od.get('total_window_area', 0):.2f} sqm")
    print(f"   Deductions applied: {od.get('deductions_applied', 0)}")

    print("\n3. FORMWORK DERIVED (from RCC)")
    fw = summary.get("formwork", {})
    print(f"   Items derived: {fw.get('items_derived', 0)}")
    print(f"   Total formwork: {fw.get('total_formwork_sqm', 0):.2f} sqm")
    print(f"   Formwork value: Rs.{fw.get('total_value', 0):,.2f}")

    print("\n4. EXTERNAL WORKS PROVISIONALS")
    ext = summary.get("external_provisionals", {})
    print(f"   Items generated: {ext.get('items_generated', 0)}")
    print(f"   Provisional value: Rs.{ext.get('total_provisional_value', 0):,.2f}")

    print("\n5. PRELIMS (Derived from Quantities)")
    pr = summary.get("prelims", {})
    print(f"   Total items: {pr.get('total_items', 0)}")
    print(f"   Total value: Rs.{pr.get('total_value', 0):,.2f}")
    if pr.get("by_category"):
        print("\n   By Category:")
        for cat, data in pr.get("by_category", {}).items():
            print(f"   - {cat.title()}: {data['items']} items, Rs.{data['amount']:,.2f}")

    print("\n6. FINAL TOTALS")
    final = summary.get("final_boq", {})
    print(f"   Final BOQ items: {final.get('items', 0)}")
    print(f"   BOQ Total: Rs.{final.get('boq_total', 0):,.2f}")
    print(f"   Prelims Total: Rs.{final.get('prelims_total', 0):,.2f}")
    print(f"   GRAND TOTAL: Rs.{final.get('grand_total', 0):,.2f}")

    print()
    print("=" * 70)
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 70)

    # Show adjusted items with deductions
    print("\n\nSAMPLE ADJUSTED ITEMS (showing deductions):")
    print("-" * 70)

    adjusted_boq = results["adjusted_boq"]
    for item in adjusted_boq[:10]:
        if item.get("opening_deduction") or item.get("deductions"):
            gross = item.get("gross_quantity", item.get("quantity", 0))
            deduct = item.get("opening_deduction", item.get("deductions", 0))
            net = item.get("quantity", 0)
            print(f"\n{item.get('item_id')}: {item.get('description', '')[:50]}")
            print(f"   Gross: {gross:.2f} {item.get('unit')}")
            print(f"   Deduction: {deduct:.2f} {item.get('unit')}")
            print(f"   Net: {net:.2f} {item.get('unit')}")

    print("\n\nDERIVED FORMWORK ITEMS:")
    print("-" * 70)
    formwork_items = results["formwork_items"]
    for item in formwork_items[:5]:
        print(f"\n{item.get('item_id')}: {item.get('description', '')[:60]}")
        print(f"   Quantity: {item.get('quantity', 0):.2f} {item.get('unit')}")
        print(f"   Amount: Rs.{item.get('amount', 0):,.2f}")
        print(f"   Derivation: {item.get('derivation_method', '')}")

    print("\n\nEXTERNAL PROVISIONALS (auto-generated):")
    print("-" * 70)
    external_items = results["external_items"]
    for item in external_items[:5]:
        print(f"\n{item.get('item_id')}: {item.get('description', '')[:60]}")
        print(f"   Quantity: {item.get('quantity', 0):.2f} {item.get('unit')}")
        print(f"   Amount: Rs.{item.get('amount', 0):,.2f}")
        print(f"   Derivation: {item.get('derivation_method', '')}")

    return results


if __name__ == "__main__":
    main()
