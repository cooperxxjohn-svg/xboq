#!/usr/bin/env python3
"""
Bid Strategy Layer Demo

Demonstrates the complete bid strategy pipeline:
1. Risk Pricing Engine
2. Rate Sensitivity Engine
3. Quote Planning Engine
4. Exclusions & Assumptions Generator
5. Bid Strategy Sheet
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.pricing import run_risk_pricing
from risk.sensitivity import run_sensitivity_analysis
from risk.quote_plan import run_quote_planning
from risk.bid_strategy import run_bid_strategy
from bid_docs.exclusions import run_exclusions_assumptions

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "bid_strategy_demo"


def create_sample_boq():
    """Sample BOQ with varied confidence and risk levels."""
    return [
        # RCC - high confidence
        {"item_id": "RCC-001", "description": "RCC M25 columns 450x450", "quantity": 45.5, "unit": "cum", "rate": 8500, "amount": 386750, "package": "rcc", "confidence": 0.92, "rate_source": "CPWD-2023"},
        {"item_id": "RCC-002", "description": "RCC M25 beams 230x450", "quantity": 38.2, "unit": "cum", "rate": 8200, "amount": 313240, "package": "rcc", "confidence": 0.88, "rate_source": "CPWD-2023"},
        {"item_id": "RCC-003", "description": "RCC M25 slab 150mm", "quantity": 125.0, "unit": "cum", "rate": 7800, "amount": 975000, "package": "rcc", "confidence": 0.95, "rate_source": "CPWD-2023"},
        {"item_id": "RCC-004", "description": "Steel Fe500D", "quantity": 18500, "unit": "kg", "rate": 85, "amount": 1572500, "package": "rcc", "confidence": 0.85, "rate_source": "ISR-2024"},

        # Masonry - medium-high confidence
        {"item_id": "MAS-001", "description": "AAC Block 200mm external", "quantity": 1850, "unit": "sqm", "rate": 1450, "amount": 2682500, "package": "masonry", "confidence": 0.90, "rate_source": "DSR-KAR"},
        {"item_id": "MAS-002", "description": "AAC Block 100mm internal", "quantity": 420, "unit": "sqm", "rate": 980, "amount": 411600, "package": "masonry", "confidence": 0.85, "rate_source": "DSR-KAR"},
        {"item_id": "MAS-003", "description": "Cement plaster 12mm internal", "quantity": 4200, "unit": "sqm", "rate": 185, "amount": 777000, "package": "masonry", "confidence": 0.82, "rate_source": "DSR-KAR"},

        # Waterproofing - medium confidence (specialized)
        {"item_id": "WP-001", "description": "APP membrane terrace", "quantity": 520, "unit": "sqm", "rate": 650, "amount": 338000, "package": "waterproof", "confidence": 0.75, "rate_source": "Market"},
        {"item_id": "WP-002", "description": "Integral WP toilets", "quantity": 185, "unit": "sqm", "rate": 420, "amount": 77700, "package": "waterproof", "confidence": 0.70, "rate_source": "Market"},

        # Flooring - medium confidence
        {"item_id": "FLR-001", "description": "Vitrified tiles 600x600", "quantity": 380, "unit": "sqm", "rate": 1850, "amount": 703000, "package": "flooring", "confidence": 0.85, "rate_source": "Market"},
        {"item_id": "FLR-002", "description": "Ceramic tiles toilets", "quantity": 95, "unit": "sqm", "rate": 1250, "amount": 118750, "package": "flooring", "confidence": 0.80, "rate_source": "Market"},

        # Doors & Windows - high confidence (schedule-based)
        {"item_id": "DW-001", "description": "Flush door 900x2100", "quantity": 24, "unit": "nos", "rate": 12500, "amount": 300000, "package": "doors_windows", "confidence": 0.90, "rate_source": "Market"},
        {"item_id": "DW-002", "description": "UPVC window 1800x1500", "quantity": 12, "unit": "nos", "rate": 18500, "amount": 222000, "package": "doors_windows", "confidence": 0.88, "rate_source": "Vendor"},

        # Plumbing - LOW confidence (provisional)
        {"item_id": "PLB-001", "description": "Internal plumbing PROVISIONAL", "quantity": 1, "unit": "LS", "rate": 850000, "amount": 850000, "package": "plumbing", "confidence": 0.40, "is_provisional": True, "rate_source": "Allowance"},

        # Electrical - LOW confidence (provisional)
        {"item_id": "ELE-001", "description": "Internal electrical PROVISIONAL", "quantity": 1, "unit": "LS", "rate": 720000, "amount": 720000, "package": "electrical", "confidence": 0.35, "is_provisional": True, "rate_source": "Allowance"},

        # External - LOW confidence (no site plan)
        {"item_id": "EXT-001", "description": "Compound wall 230mm", "quantity": 85, "unit": "rmt", "rate": 4500, "amount": 382500, "package": "external", "confidence": 0.65, "rate_source": "DSR-KAR"},
        {"item_id": "EXT-002", "description": "External paving PROVISIONAL", "quantity": 200, "unit": "sqm", "rate": 1200, "amount": 240000, "package": "external", "confidence": 0.50, "is_provisional": True, "rate_source": "Allowance"},

        # Finishes
        {"item_id": "FIN-001", "description": "Interior paint 2 coats", "quantity": 4200, "unit": "sqm", "rate": 85, "amount": 357000, "package": "finishes", "confidence": 0.80, "rate_source": "Market"},
    ]


def create_sample_rfis():
    """Sample RFIs."""
    return [
        {"id": "RFI-001", "question": "Column C5 dimensions mismatch", "priority": "high", "status": "open", "package": "rcc", "category": "structural"},
        {"id": "RFI-002", "question": "MEP drawings required", "priority": "high", "status": "open", "package": "plumbing", "category": "mep"},
        {"id": "RFI-003", "question": "External scope clarity", "priority": "medium", "status": "open", "package": "external", "category": "civil"},
        {"id": "RFI-004", "question": "Tile specifications", "priority": "low", "status": "open", "package": "flooring", "category": "finishes"},
        {"id": "RFI-005", "question": "Soil report required", "priority": "high", "status": "open", "package": "rcc", "category": "structural"},
        {"id": "RFI-006", "question": "Electrical load calculation", "priority": "medium", "status": "open", "package": "electrical", "category": "mep"},
    ]


def create_sample_prelims():
    """Sample prelims items."""
    return [
        {"item": "Site Engineer", "quantity": 14, "unit": "months", "rate": 65000, "amount": 910000},
        {"item": "Supervisors", "quantity": 28, "unit": "man-months", "rate": 45000, "amount": 1260000},
        {"item": "Site Office", "quantity": 1, "unit": "LS", "rate": 180000, "amount": 180000},
        {"item": "Temp Electricity", "quantity": 14, "unit": "months", "rate": 45000, "amount": 630000},
        {"item": "Temp Water", "quantity": 14, "unit": "months", "rate": 25000, "amount": 350000},
        {"item": "Scaffolding", "quantity": 8, "unit": "months", "rate": 85000, "amount": 680000},
        {"item": "Tower Crane", "quantity": 10, "unit": "months", "rate": 185000, "amount": 1850000},
        {"item": "Insurance", "quantity": 1, "unit": "LS", "rate": 45000, "amount": 45000},
    ]


def create_sample_gate_result():
    """Sample bid gate result."""
    return {
        "status": "PASS_WITH_RESERVATIONS",
        "score": 68.5,
        "is_submittable": True,
        "reservations_count": 6,
        "reservations": [
            {"code": "RES-001", "description": "MEP on provisional", "severity": "high"},
            {"code": "RES-002", "description": "Scale confidence 78%", "severity": "medium"},
            {"code": "RES-003", "description": "Approval drawings only", "severity": "medium"},
            {"code": "RES-004", "description": "No soil report", "severity": "high"},
            {"code": "RES-005", "description": "External scope partial", "severity": "medium"},
            {"code": "RES-006", "description": "3 high-priority RFIs", "severity": "high"},
        ],
    }


def main():
    print("=" * 70)
    print("BID STRATEGY LAYER DEMO")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_dir = OUTPUT_DIR / "risk"
    risk_dir.mkdir(parents=True, exist_ok=True)
    bid_book_dir = OUTPUT_DIR / "bid_book"
    bid_book_dir.mkdir(parents=True, exist_ok=True)

    # Sample data
    boq_items = create_sample_boq()
    rfis = create_sample_rfis()
    prelims = create_sample_prelims()
    gate_result = create_sample_gate_result()

    project_params = {
        "project_name": "Residential Villa - Whitefield",
        "built_up_area_sqm": 4500,
        "duration_months": 14,
        "location": "Bangalore",
        "drawings_type": "approval",
        "mep_drawings_provided": False,
    }

    print("Input:")
    print(f"  BOQ Items: {len(boq_items)}")
    print(f"  RFIs: {len(rfis)}")
    print(f"  Prelims Items: {len(prelims)}")
    print()

    # 1. Risk Pricing Engine
    print("1. Running Risk Pricing Engine...")
    risk_results = run_risk_pricing(
        boq_items=boq_items,
        rfis=rfis,
        owner_inputs_gaps=["soil_type", "electrical_load", "plumbing_fixtures"],
        output_path=risk_dir / "risk_pricing.csv",
    )
    print(f"   Packages analyzed: {risk_results['summary']['packages_analyzed']}")
    print(f"   High/Very High risk: {risk_results['summary']['high_risk_packages'] + risk_results['summary']['very_high_risk_packages']}")
    print(f"   Weighted contingency: {risk_results['summary']['weighted_avg_contingency']:.1f}%")
    print()

    # 2. Rate Sensitivity Engine
    print("2. Running Rate Sensitivity Engine...")
    sensitivity_results = run_sensitivity_analysis(
        boq_items=boq_items,
        prelims_items=prelims,
        project_duration_months=14,
        output_path=risk_dir / "sensitivity_report.md",
    )
    summary = sensitivity_results["summary"]
    print(f"   Scenarios analyzed: {summary['scenarios_analyzed']}")
    print(f"   Max upside risk: {summary['max_upside_risk']['scenario']} ({summary['max_upside_risk']['impact_pct']:+.2f}%)")
    print(f"   Max downside opportunity: {summary['max_downside_opportunity']['scenario']} ({summary['max_downside_opportunity']['impact_pct']:.2f}%)")
    print()

    # 3. Quote Planning Engine
    print("3. Running Quote Planning Engine...")
    quote_results = run_quote_planning(
        boq_items=boq_items,
        risk_profiles=risk_results["package_risks"],
        output_dir=risk_dir,
    )
    print(f"   Total packages: {quote_results['summary']['total_packages']}")
    print(f"   Urgent quotes: {quote_results['summary']['urgent_quotes']}")
    print(f"   Quotes needed: {quote_results['summary']['quotes_needed']}")
    print()

    # 4. Exclusions & Assumptions
    print("4. Generating Exclusions & Assumptions...")
    exc_results = run_exclusions_assumptions(
        scope_gaps=[
            {"item": "Lift installation", "reason": "No drawings"},
            {"item": "Fire fighting system", "reason": "MEP missing"},
        ],
        missing_drawings=["MEP-01", "STR-03", "SITE-02"],
        missing_specs=["Tile brand/model", "Paint shade"],
        owner_input_gaps=["soil_type", "electrical_load"],
        provisional_items=[i for i in boq_items if i.get("is_provisional")],
        project_params=project_params,
        output_dir=bid_book_dir,
    )
    print(f"   Exclusions: {exc_results['summary']['total_exclusions']}")
    print(f"   Assumptions: {exc_results['summary']['total_assumptions']}")
    print()

    # 5. Bid Strategy Sheet
    print("5. Generating Bid Strategy Sheet...")
    strategy_results = run_bid_strategy(
        risk_profiles=risk_results["package_risks"],
        sensitivity_results=sensitivity_results,
        quote_plan=quote_results["quote_plan"],
        gate_result=gate_result,
        project_params=project_params,
        output_path=risk_dir / "bid_strategy.md",
    )
    print(f"   Recommendation: {strategy_results['summary']['recommendation']}")
    print(f"   Safe packages: {strategy_results['summary']['safe_packages']}")
    print(f"   Risky packages: {strategy_results['summary']['risky_packages']}")
    print(f"   Weighted contingency: {strategy_results['summary']['weighted_contingency']:.1f}%")
    print()

    # Save master results
    all_results = {
        "risk_pricing": risk_results,
        "sensitivity": sensitivity_results,
        "quote_plan": quote_results,
        "exclusions_assumptions": exc_results,
        "bid_strategy": strategy_results["summary"],
    }

    with open(OUTPUT_DIR / "bid_strategy_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"  {risk_dir / 'risk_pricing.csv'}")
    print(f"  {risk_dir / 'sensitivity_report.md'}")
    print(f"  {risk_dir / 'quote_plan.csv'}")
    print(f"  {risk_dir / 'quote_plan.md'}")
    print(f"  {risk_dir / 'bid_strategy.md'}")
    print(f"  {bid_book_dir / 'exclusions.md'}")
    print(f"  {bid_book_dir / 'assumptions.md'}")
    print(f"  {OUTPUT_DIR / 'bid_strategy_results.json'}")
    print()

    # Show bid strategy summary
    print("=" * 70)
    print("BID STRATEGY SUMMARY")
    print("=" * 70)
    print()
    print(f"  RECOMMENDATION: {strategy_results['summary']['recommendation']}")
    print()
    print("  Risk Distribution:")
    print(f"    - Low Risk: {strategy_results['summary']['safe_packages']} packages")
    print(f"    - High Risk: {strategy_results['summary']['risky_packages']} packages")
    print()
    print(f"  Contingency Recommendation: {strategy_results['summary']['weighted_contingency']:.1f}%")
    print(f"  Urgent Quotes Required: {strategy_results['summary']['urgent_quotes']}")
    print(f"  Top Risk Driver: {strategy_results['summary']['top_risk_driver']}")
    print()

    return all_results


if __name__ == "__main__":
    main()
