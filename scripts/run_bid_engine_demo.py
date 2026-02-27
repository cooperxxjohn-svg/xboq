#!/usr/bin/env python3
"""
Bid Engine Demo Runner
Generates sample output for all 10 phases (16-25) and creates a PDF report.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "bid_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ID = "DEMO-2024-001"


def create_sample_drawings_boq():
    """Sample BOQ extracted from drawings (simulates Phase 1-11 output)."""
    return [
        # RCC Items
        {"item_id": "RCC-001", "description": "RCC M25 grade for columns 450x450mm", "quantity": 45.5, "unit": "cum", "package": "rcc", "drawing_ref": "STR-01", "confidence": 0.92},
        {"item_id": "RCC-002", "description": "RCC M25 grade for beams 230x450mm", "quantity": 38.2, "unit": "cum", "package": "rcc", "drawing_ref": "STR-01", "confidence": 0.88},
        {"item_id": "RCC-003", "description": "RCC M25 grade for slab 150mm thick", "quantity": 125.0, "unit": "cum", "package": "rcc", "drawing_ref": "STR-02", "confidence": 0.95},
        {"item_id": "RCC-004", "description": "Steel reinforcement Fe500D", "quantity": 18500, "unit": "kg", "package": "rcc", "drawing_ref": "STR-01,STR-02", "confidence": 0.85},

        # Masonry Items
        {"item_id": "MAS-001", "description": "AAC Block masonry 200mm thick", "quantity": 1850, "unit": "sqm", "package": "masonry", "drawing_ref": "ARC-01,ARC-02", "confidence": 0.90},
        {"item_id": "MAS-002", "description": "AAC Block masonry 100mm thick", "quantity": 420, "unit": "sqm", "package": "masonry", "drawing_ref": "ARC-01", "confidence": 0.88},
        {"item_id": "MAS-003", "description": "Cement plaster 12mm internal", "quantity": 4200, "unit": "sqm", "package": "masonry", "drawing_ref": "ARC-01,ARC-02", "confidence": 0.82},
        {"item_id": "MAS-004", "description": "Cement plaster 20mm external", "quantity": 1100, "unit": "sqm", "package": "masonry", "drawing_ref": "ARC-03", "confidence": 0.80},

        # Waterproofing
        {"item_id": "WP-001", "description": "APP membrane waterproofing for terrace", "quantity": 520, "unit": "sqm", "package": "waterproof", "drawing_ref": "ARC-ROOF", "confidence": 0.75},
        {"item_id": "WP-002", "description": "Integral waterproofing for toilets", "quantity": 185, "unit": "sqm", "package": "waterproof", "drawing_ref": "ARC-01", "confidence": 0.70},

        # Flooring
        {"item_id": "FLR-001", "description": "Vitrified tiles 600x600mm living/dining", "quantity": 380, "unit": "sqm", "package": "flooring", "drawing_ref": "INT-01", "confidence": 0.85},
        {"item_id": "FLR-002", "description": "Ceramic tiles 300x300mm anti-skid toilets", "quantity": 95, "unit": "sqm", "package": "flooring", "drawing_ref": "INT-01", "confidence": 0.82},
        {"item_id": "FLR-003", "description": "Granite flooring for lobby", "quantity": 65, "unit": "sqm", "package": "flooring", "drawing_ref": "INT-02", "confidence": 0.78},

        # Doors and Windows
        {"item_id": "DW-001", "description": "Flush door 900x2100mm with frame", "quantity": 24, "unit": "nos", "package": "doors_windows", "drawing_ref": "ARC-01,D-SCH", "confidence": 0.90},
        {"item_id": "DW-002", "description": "Main entrance door teak wood", "quantity": 1, "unit": "nos", "package": "doors_windows", "drawing_ref": "D-SCH", "confidence": 0.88},
        {"item_id": "DW-003", "description": "UPVC sliding window 1800x1500mm", "quantity": 12, "unit": "nos", "package": "doors_windows", "drawing_ref": "W-SCH", "confidence": 0.85},
        {"item_id": "DW-004", "description": "UPVC casement window 1200x1200mm", "quantity": 18, "unit": "nos", "package": "doors_windows", "drawing_ref": "W-SCH", "confidence": 0.82},

        # Plumbing (partial - provisional)
        {"item_id": "PLB-001", "description": "Internal plumbing - provisional", "quantity": 1, "unit": "LS", "package": "plumbing", "drawing_ref": "N/A", "confidence": 0.40, "is_provisional": True},

        # Electrical (partial - provisional)
        {"item_id": "ELE-001", "description": "Internal electrical - provisional", "quantity": 1, "unit": "LS", "package": "electrical", "drawing_ref": "N/A", "confidence": 0.35, "is_provisional": True},

        # External Works
        {"item_id": "EXT-001", "description": "Compound wall brick 230mm", "quantity": 85, "unit": "rmt", "package": "external", "drawing_ref": "SITE-01", "confidence": 0.70},
        {"item_id": "EXT-002", "description": "Paver blocks for driveway", "quantity": 180, "unit": "sqm", "package": "external", "drawing_ref": "SITE-01", "confidence": 0.65},
    ]


def create_sample_scope_register():
    """Sample scope register (simulates Phase 10-15 output)."""
    return {
        "project_id": PROJECT_ID,
        "scale_confidence": 0.78,  # Below threshold - will trigger reservation
        "total_schedule_items": 120,
        "mapped_schedule_items": 95,
        "missing_sheets_count": 2,  # STR-03, MEP-01 referenced but missing
        "structural_drawings_provided": True,
        "mep_drawings_provided": False,  # Will trigger reservation
        "drawings_type": "approval",  # Not GFC - will trigger reservation
        "external_works_status": "partial",
        "mep_coverage_status": "unknown",
        "inclusions": [
            "All RCC structural work as per approved drawings",
            "AAC block masonry with cement plaster",
            "Vitrified/ceramic tile flooring as per schedule",
            "UPVC doors and windows as per schedule",
            "External paving and compound wall",
        ],
        "exclusions": [
            "Furniture and furnishings",
            "Modular kitchen",
            "Air conditioning and HVAC",
            "Landscaping and soft areas",
            "External development beyond site boundary",
        ],
        "missing_scope": [
            {"item": "Lift installation", "reason": "No lift drawings provided"},
            {"item": "Fire fighting system", "reason": "MEP drawings missing"},
            {"item": "STP/WTP", "reason": "Not shown on site plan"},
        ],
        "conflicts": [
            {"type": "dimension_mismatch", "description": "Column C5 shown as 450x450 on STR-01 but 400x400 on ARC-01", "severity": "medium"},
            {"type": "missing_detail", "description": "Staircase handrail detail not provided", "severity": "low"},
        ],
        "rfis": [
            {"id": "RFI-001", "question": "Confirm column C5 dimensions - mismatch between structural and architectural", "priority": "high", "status": "open"},
            {"id": "RFI-002", "question": "Provide MEP drawings for plumbing and electrical layout", "priority": "high", "status": "open"},
            {"id": "RFI-003", "question": "Clarify external development scope - driveway extent unclear", "priority": "medium", "status": "open"},
            {"id": "RFI-004", "question": "Confirm tile specifications for living areas - brand/model", "priority": "low", "status": "open"},
            {"id": "RFI-005", "question": "Provide soil investigation report for foundation design", "priority": "high", "status": "open"},
            {"id": "RFI-006", "question": "Confirm ceiling height - 3.0m or 3.15m", "priority": "medium", "status": "open"},
        ],
    }


def run_demo():
    """Run the bid engine with sample data."""
    print("=" * 70)
    print("XBOQ BID ENGINE DEMO")
    print("=" * 70)
    print(f"Project ID: {PROJECT_ID}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()

    # Create sample data
    drawings_boq = create_sample_drawings_boq()
    scope_register = create_sample_scope_register()

    print(f"Sample BOQ Items: {len(drawings_boq)}")
    print(f"Sample RFIs: {len(scope_register['rfis'])}")
    print()

    # Create project directory with owner_inputs.yaml
    project_dir = OUTPUT_DIR / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Write sample owner_inputs.yaml
    owner_inputs_yaml = """# Owner Inputs for Demo Project
project:
  name: "Residential Villa - Whitefield"
  type: residential
  location: Bangalore
  built_up_area_sqm: 4500
  completion_months: 14

site:
  address: "Survey No. 123, Whitefield, Bangalore - 560066"
  soil_type: unknown  # Soil report not provided
  water_table_depth_m: null

finishes:
  grade: premium
  floor_tile_brand: null  # To be confirmed

structural:
  concrete_grade: M25
  steel_grade: Fe500D

mep:
  electrical_load_kw: null  # MEP drawings not provided
  plumbing_fixtures: null
"""
    with open(project_dir / "owner_inputs.yaml", "w") as f:
        f.write(owner_inputs_yaml)

    # Import and run bid engine
    try:
        from bid_engine import run_bid_engine

        results = run_bid_engine(
            project_id=PROJECT_ID,
            project_dir=project_dir,
            output_dir=OUTPUT_DIR,
            drawings_boq=drawings_boq,
            scope_register=scope_register,
            owner_docs_paths=None,  # No owner docs for demo
            subcontractor_quotes=None,  # No quotes for demo
            built_up_area_sqm=4500,
            duration_months=14,
        )

        print("\n" + "=" * 70)
        print("BID ENGINE RESULTS")
        print("=" * 70)

        summary = results.get("summary", {})
        print(f"Bid Status: {summary.get('bid_status', 'UNKNOWN')}")
        print(f"Gate Score: {summary.get('gate_score', 0):.1f}/100")
        print(f"Submittable: {summary.get('is_submittable', False)}")
        print(f"Reservations: {summary.get('reservations_count', 0)}")
        print(f"High Priority RFIs: {summary.get('high_priority_rfis', 0)}")
        print()
        print(f"Phases Completed: {summary.get('phases_completed', 0)}")
        print(f"Phases Skipped: {summary.get('phases_skipped', 0)}")
        print(f"Phases Errored: {summary.get('phases_errored', 0)}")
        print()
        print(f"Total Bid Value: ₹{summary.get('total_bid_value', 0):,.2f}")
        print()
        print(f"Results saved to: {OUTPUT_DIR}")

        return results

    except Exception as e:
        print(f"Error running bid engine: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_demo()
