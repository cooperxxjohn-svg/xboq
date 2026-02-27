#!/usr/bin/env python3
"""
Standalone Bid Engine Demo
Generates complete sample output for Phases 16-25 without module dependencies.
Creates outputs that can be compiled into a PDF report.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from enum import Enum

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "bid_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ID = "DEMO-2024-001"
PROJECT_NAME = "Residential Villa - Whitefield, Bangalore"


# ============================================================================
# DATA CLASSES
# ============================================================================

class GateStatus(Enum):
    PASS = "PASS"
    PASS_WITH_RESERVATIONS = "PASS_WITH_RESERVATIONS"
    FAIL = "FAIL"


@dataclass
class BOQItem:
    item_id: str
    description: str
    quantity: float
    unit: str
    rate: float = 0.0
    amount: float = 0.0
    package: str = ""
    drawing_ref: str = ""
    confidence: float = 0.0
    is_provisional: bool = False
    rate_source: str = ""


@dataclass
class Reservation:
    code: str
    description: str
    severity: str
    recommendation: str
    evidence: str


@dataclass
class RFI:
    rfi_id: str
    question: str
    priority: str
    status: str
    category: str


# ============================================================================
# SAMPLE DATA
# ============================================================================

def create_sample_boq() -> List[Dict]:
    """Create sample BOQ with India-specific items and rates."""
    items = [
        # RCC Items (CPWD/DSR based rates)
        BOQItem("RCC-001", "RCC M25 grade for columns 450x450mm", 45.5, "cum", 8500, 0, "rcc", "STR-01", 0.92, rate_source="CPWD-2023"),
        BOQItem("RCC-002", "RCC M25 grade for beams 230x450mm", 38.2, "cum", 8200, 0, "rcc", "STR-01", 0.88, rate_source="CPWD-2023"),
        BOQItem("RCC-003", "RCC M25 grade for slab 150mm thick", 125.0, "cum", 7800, 0, "rcc", "STR-02", 0.95, rate_source="CPWD-2023"),
        BOQItem("RCC-004", "Steel reinforcement Fe500D", 18500, "kg", 85, 0, "rcc", "STR-01,STR-02", 0.85, rate_source="ISR-2024"),

        # Masonry Items
        BOQItem("MAS-001", "AAC Block masonry 200mm thick", 1850, "sqm", 1450, 0, "masonry", "ARC-01,ARC-02", 0.90, rate_source="DSR-KAR"),
        BOQItem("MAS-002", "AAC Block masonry 100mm thick", 420, "sqm", 980, 0, "masonry", "ARC-01", 0.88, rate_source="DSR-KAR"),
        BOQItem("MAS-003", "Cement plaster 12mm internal", 4200, "sqm", 185, 0, "masonry", "ARC-01,ARC-02", 0.82, rate_source="DSR-KAR"),
        BOQItem("MAS-004", "Cement plaster 20mm external", 1100, "sqm", 245, 0, "masonry", "ARC-03", 0.80, rate_source="DSR-KAR"),

        # Waterproofing
        BOQItem("WP-001", "APP membrane waterproofing for terrace", 520, "sqm", 650, 0, "waterproof", "ARC-ROOF", 0.75, rate_source="Market-2024"),
        BOQItem("WP-002", "Integral waterproofing for toilets", 185, "sqm", 420, 0, "waterproof", "ARC-01", 0.70, rate_source="Market-2024"),

        # Flooring
        BOQItem("FLR-001", "Vitrified tiles 600x600mm living/dining", 380, "sqm", 1850, 0, "flooring", "INT-01", 0.85, rate_source="Market-2024"),
        BOQItem("FLR-002", "Ceramic tiles 300x300mm anti-skid toilets", 95, "sqm", 1250, 0, "flooring", "INT-01", 0.82, rate_source="Market-2024"),
        BOQItem("FLR-003", "Granite flooring for lobby", 65, "sqm", 3200, 0, "flooring", "INT-02", 0.78, rate_source="Market-2024"),

        # Doors and Windows
        BOQItem("DW-001", "Flush door 900x2100mm with frame", 24, "nos", 12500, 0, "doors_windows", "ARC-01,D-SCH", 0.90, rate_source="Market-2024"),
        BOQItem("DW-002", "Main entrance door teak wood", 1, "nos", 85000, 0, "doors_windows", "D-SCH", 0.88, rate_source="Market-2024"),
        BOQItem("DW-003", "UPVC sliding window 1800x1500mm", 12, "nos", 18500, 0, "doors_windows", "W-SCH", 0.85, rate_source="Vendor Quote"),
        BOQItem("DW-004", "UPVC casement window 1200x1200mm", 18, "nos", 12800, 0, "doors_windows", "W-SCH", 0.82, rate_source="Vendor Quote"),

        # Plumbing (provisional)
        BOQItem("PLB-001", "Internal plumbing complete - PROVISIONAL", 1, "LS", 850000, 0, "plumbing", "N/A", 0.40, True, rate_source="Allowance @190/sqft"),

        # Electrical (provisional)
        BOQItem("ELE-001", "Internal electrical complete - PROVISIONAL", 1, "LS", 720000, 0, "electrical", "N/A", 0.35, True, rate_source="Allowance @160/sqft"),

        # External Works
        BOQItem("EXT-001", "Compound wall brick 230mm with foundation", 85, "rmt", 4500, 0, "external", "SITE-01", 0.70, rate_source="DSR-KAR"),
        BOQItem("EXT-002", "Paver blocks 80mm for driveway", 180, "sqm", 1150, 0, "external", "SITE-01", 0.65, rate_source="Market-2024"),
    ]

    # Calculate amounts
    for item in items:
        item.amount = item.quantity * item.rate

    return [asdict(item) for item in items]


def create_sample_rfis() -> List[Dict]:
    """Create sample RFIs."""
    rfis = [
        RFI("RFI-001", "Confirm column C5 dimensions - mismatch between structural (450x450) and architectural (400x400) drawings", "high", "open", "structural"),
        RFI("RFI-002", "Provide MEP drawings for plumbing and electrical layout - currently using provisional allowances", "high", "open", "mep"),
        RFI("RFI-003", "Clarify external development scope - driveway extent and levels not clear from site plan", "medium", "open", "external"),
        RFI("RFI-004", "Confirm tile specifications for living areas - brand/model/colour to be selected", "low", "open", "finishes"),
        RFI("RFI-005", "Provide soil investigation report for foundation design verification", "high", "open", "structural"),
        RFI("RFI-006", "Confirm ceiling height - drawings show 3.0m but schedule mentions 3.15m", "medium", "open", "architectural"),
    ]
    return [asdict(rfi) for rfi in rfis]


def create_reservations() -> List[Dict]:
    """Create bid reservations based on gaps."""
    reservations = [
        Reservation("RES-001", "MEP drawings not provided - plumbing and electrical on provisional allowance", "high",
                   "Obtain MEP drawings before construction; current provisional may vary +/-25%", "No MEP sheets in drawing set"),
        Reservation("RES-002", "Scale confidence 78% on floor plans - quantities may vary", "medium",
                   "Request GFC drawings with verified scale bars", "Scale detection confidence: 78%"),
        Reservation("RES-003", "Approval drawings only - not GFC (Good For Construction)", "medium",
                   "Quantities are indicative; final BOQ subject to GFC issue", "Drawing title block shows 'For Approval'"),
        Reservation("RES-004", "Soil investigation report not provided - foundation design unverified", "high",
                   "Add foundation allowance of 10% for unforeseen ground conditions", "Missing document: Soil Report"),
        Reservation("RES-005", "External works scope partially defined", "medium",
                   "Compound wall and driveway included; landscaping excluded", "Site plan shows boundary only"),
        Reservation("RES-006", "3 high-priority RFIs unresolved", "high",
                   "Resolve RFIs before bid finalization or submit with clarifications", "RFI-001, RFI-002, RFI-005"),
    ]
    return [asdict(r) for r in reservations]


# ============================================================================
# PHASE OUTPUTS
# ============================================================================

def generate_phase_outputs():
    """Generate outputs for all phases 16-25."""

    priced_boq = create_sample_boq()
    rfis = create_sample_rfis()
    reservations = create_reservations()

    # Calculate totals
    boq_total = sum(item["amount"] for item in priced_boq)
    provisional_total = sum(item["amount"] for item in priced_boq if item["is_provisional"])

    # Prelims calculation (12% of project value)
    prelims_percent = 12.0
    prelims_total = boq_total * prelims_percent / 100

    grand_total = boq_total + prelims_total

    results = {
        "project_id": PROJECT_ID,
        "project_name": PROJECT_NAME,
        "generated_at": datetime.now().isoformat(),
        "phases": {},
    }

    # =========================================================================
    # PHASE 16: Owner Docs Parser
    # =========================================================================
    phase16_dir = OUTPUT_DIR / "phase16_owner_docs"
    phase16_dir.mkdir(parents=True, exist_ok=True)

    with open(phase16_dir / "parsed_docs.json", "w") as f:
        json.dump({
            "status": "skipped",
            "reason": "No owner documents provided for demo",
            "expected_documents": ["Tender document", "Technical specifications", "Owner BOQ", "Addenda"],
        }, f, indent=2)

    results["phases"]["phase16"] = {"status": "skipped", "reason": "No owner documents"}

    # =========================================================================
    # PHASE 17: Owner Inputs Engine
    # =========================================================================
    phase17_dir = OUTPUT_DIR / "phase17_owner_inputs"
    phase17_dir.mkdir(parents=True, exist_ok=True)

    owner_inputs = {
        "project": {"name": PROJECT_NAME, "type": "residential", "location": "Bangalore", "built_up_area_sqm": 4500},
        "finishes": {"grade": "premium", "floor_tile_brand": "TBD"},
        "structural": {"concrete_grade": "M25", "steel_grade": "Fe500D"},
        "completeness_score": 62,
        "missing_mandatory": ["soil_type", "electrical_load_kw", "plumbing_fixtures"],
        "defaults_applied": ["ceiling_height: 3000mm", "slab_thickness: 150mm", "wall_plaster: 12mm internal"],
    }

    with open(phase17_dir / "owner_inputs.json", "w") as f:
        json.dump(owner_inputs, f, indent=2)

    with open(phase17_dir / "owner_input_rfis.json", "w") as f:
        json.dump([r for r in rfis if r["category"] in ["structural", "mep"]], f, indent=2)

    results["phases"]["phase17"] = {"status": "completed", "completeness_score": 62, "rfis_generated": 2}

    # =========================================================================
    # PHASE 18: BOQ Alignment
    # =========================================================================
    phase18_dir = OUTPUT_DIR / "phase18_alignment"
    phase18_dir.mkdir(parents=True, exist_ok=True)

    with open(phase18_dir / "alignment_result.json", "w") as f:
        json.dump({
            "status": "skipped",
            "reason": "No owner BOQ provided for comparison",
            "drawings_boq_items": len(priced_boq),
        }, f, indent=2)

    results["phases"]["phase18"] = {"status": "skipped", "reason": "No owner BOQ"}

    # =========================================================================
    # PHASE 19: Pricing Engine
    # =========================================================================
    phase19_dir = OUTPUT_DIR / "phase19_pricing"
    phase19_dir.mkdir(parents=True, exist_ok=True)

    with open(phase19_dir / "priced_boq.json", "w") as f:
        json.dump(priced_boq, f, indent=2)

    # Generate rate analysis
    rate_analysis = []
    for item in priced_boq:
        if item["amount"] > 100000:  # Items > 1 lakh
            rate_analysis.append({
                "item_id": item["item_id"],
                "description": item["description"],
                "rate": item["rate"],
                "amount": item["amount"],
                "rate_source": item["rate_source"],
                "breakdown": {
                    "material": item["rate"] * 0.55,
                    "labor": item["rate"] * 0.30,
                    "equipment": item["rate"] * 0.05,
                    "overhead": item["rate"] * 0.10,
                }
            })

    with open(phase19_dir / "rate_analysis.json", "w") as f:
        json.dump(rate_analysis, f, indent=2)

    results["phases"]["phase19"] = {
        "status": "completed",
        "total_items": len(priced_boq),
        "priced_items": len(priced_boq),
        "grand_total": boq_total,
        "location": "Bangalore",
        "location_factor": 1.05,
    }

    # =========================================================================
    # PHASE 20: Quote Leveling
    # =========================================================================
    phase20_dir = OUTPUT_DIR / "phase20_quotes"
    phase20_dir.mkdir(parents=True, exist_ok=True)

    with open(phase20_dir / "quote_summary.json", "w") as f:
        json.dump({
            "status": "skipped",
            "reason": "No subcontractor quotes provided",
            "packages_needing_quotes": ["flooring", "doors_windows", "plumbing", "electrical"],
        }, f, indent=2)

    results["phases"]["phase20"] = {"status": "skipped", "reason": "No quotes provided"}

    # =========================================================================
    # PHASE 21: Prelims Generator
    # =========================================================================
    phase21_dir = OUTPUT_DIR / "phase21_prelims"
    phase21_dir.mkdir(parents=True, exist_ok=True)

    prelims_items = [
        {"item": "Site Engineer (1 no.)", "months": 14, "rate": 65000, "amount": 910000},
        {"item": "Site Supervisor (2 nos.)", "months": 14, "rate": 45000, "amount": 1260000},
        {"item": "Site Office (prefab)", "quantity": 1, "unit": "LS", "amount": 180000},
        {"item": "Labour Hutments", "quantity": 1, "unit": "LS", "amount": 120000},
        {"item": "Temporary Toilet Blocks", "quantity": 2, "unit": "nos", "rate": 35000, "amount": 70000},
        {"item": "Temporary Electricity Connection", "quantity": 1, "unit": "LS", "amount": 85000},
        {"item": "Temporary Water Connection", "quantity": 1, "unit": "LS", "amount": 45000},
        {"item": "Safety Equipment & PPE", "quantity": 1, "unit": "LS", "amount": 95000},
        {"item": "Tools & Tackles", "quantity": 1, "unit": "LS", "amount": 75000},
        {"item": "Insurance (CAR Policy)", "quantity": 1, "unit": "LS", "amount": grand_total * 0.002},
        {"item": "Scaffolding & Shuttering (hire)", "months": 8, "rate": 45000, "amount": 360000},
        {"item": "Tower Crane Hire", "months": 6, "rate": 185000, "amount": 1110000},
    ]

    actual_prelims = sum(p["amount"] for p in prelims_items)

    with open(phase21_dir / "prelims_items.json", "w") as f:
        json.dump(prelims_items, f, indent=2)

    results["phases"]["phase21"] = {
        "status": "completed",
        "total_prelims": actual_prelims,
        "prelims_percent": (actual_prelims / boq_total) * 100,
        "items_count": len(prelims_items),
    }

    # =========================================================================
    # PHASE 22: Bid Book Export
    # =========================================================================
    phase22_dir = OUTPUT_DIR / "phase22_bidbook"
    phase22_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    bid_summary = {
        "project_id": PROJECT_ID,
        "project_name": PROJECT_NAME,
        "location": "Whitefield, Bangalore",
        "built_up_area_sqm": 4500,
        "duration_months": 14,
        "boq_total": boq_total,
        "prelims_total": actual_prelims,
        "grand_total": boq_total + actual_prelims,
        "rate_per_sqft": (boq_total + actual_prelims) / (4500 * 10.764),
        "items_count": len(priced_boq),
        "provisional_items": sum(1 for i in priced_boq if i["is_provisional"]),
        "provisional_value": provisional_total,
        "provisional_percent": (provisional_total / boq_total) * 100,
    }

    with open(phase22_dir / "bid_summary.json", "w") as f:
        json.dump(bid_summary, f, indent=2)

    # BOQ CSV
    with open(phase22_dir / "priced_boq.csv", "w") as f:
        f.write("SL,Item ID,Description,Quantity,Unit,Rate (INR),Amount (INR),Package,Drawing Ref,Rate Source\n")
        for i, item in enumerate(priced_boq, 1):
            f.write(f'{i},{item["item_id"]},"{item["description"]}",{item["quantity"]},{item["unit"]},{item["rate"]:.2f},{item["amount"]:.2f},{item["package"]},{item["drawing_ref"]},{item["rate_source"]}\n')

    results["phases"]["phase22"] = {
        "status": "completed",
        "total_bid_value": boq_total + actual_prelims,
        "outputs": ["bid_summary.json", "priced_boq.csv"],
    }

    # =========================================================================
    # PHASE 23: Bid Gate
    # =========================================================================
    phase23_dir = OUTPUT_DIR / "phase23_bid_gate"
    phase23_dir.mkdir(parents=True, exist_ok=True)

    # Calculate gate score
    gate_checks = {
        "scale_confidence": {"value": 0.78, "threshold": 0.85, "score": 70, "weight": 15},
        "schedule_mapping": {"value": 0.79, "threshold": 0.90, "score": 75, "weight": 20},
        "missing_sheets": {"value": 2, "threshold": 0, "score": 60, "weight": 15},
        "owner_inputs": {"value": 0.62, "threshold": 0.80, "score": 65, "weight": 20},
        "high_priority_rfis": {"value": 3, "threshold": 0, "score": 50, "weight": 15},
        "external_works": {"value": "partial", "threshold": "known", "score": 70, "weight": 8},
        "mep_coverage": {"value": "unknown", "threshold": "covered", "score": 30, "weight": 7},
    }

    weighted_score = sum(c["score"] * c["weight"] / 100 for c in gate_checks.values())
    gate_status = "PASS_WITH_RESERVATIONS" if weighted_score >= 60 else "FAIL"

    gate_result = {
        "project_id": PROJECT_ID,
        "status": gate_status,
        "score": weighted_score,
        "is_submittable": gate_status != "FAIL",
        "checks": gate_checks,
        "reservations": reservations,
        "reservations_count": len(reservations),
        "critical_failures": [],
        "recommendation": "Submit with documented reservations and clarifications letter" if gate_status == "PASS_WITH_RESERVATIONS" else "Resolve critical issues before submission",
    }

    with open(phase23_dir / "bid_gate_report.json", "w") as f:
        json.dump(gate_result, f, indent=2)

    # Generate markdown report
    with open(phase23_dir / "bid_gate_report.md", "w") as f:
        f.write(f"# Bid Gate Report: {PROJECT_ID}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%d-%b-%Y %H:%M')}\n\n")
        f.write("---\n\n")

        status_emoji = {"PASS": "✅", "PASS_WITH_RESERVATIONS": "🟡", "FAIL": "❌"}[gate_status]
        f.write(f"## {status_emoji} Gate Status: {gate_status.replace('_', ' ')}\n\n")
        f.write(f"**Score**: {weighted_score:.1f}/100\n\n")
        f.write(f"**Submittable**: {'Yes' if gate_status != 'FAIL' else 'No'}\n\n")

        f.write("## Gate Checks\n\n")
        f.write("| Check | Value | Threshold | Score | Weight |\n")
        f.write("|-------|-------|-----------|-------|--------|\n")
        for name, check in gate_checks.items():
            status = "✅" if check["score"] >= 80 else ("🟡" if check["score"] >= 60 else "❌")
            f.write(f"| {name.replace('_', ' ').title()} | {check['value']} | {check['threshold']} | {status} {check['score']} | {check['weight']}% |\n")

        f.write("\n## Reservations\n\n")
        for i, res in enumerate(reservations, 1):
            severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[res["severity"]]
            f.write(f"### {i}. {severity_icon} {res['code']}: {res['description']}\n\n")
            f.write(f"- **Severity**: {res['severity'].upper()}\n")
            f.write(f"- **Evidence**: {res['evidence']}\n")
            f.write(f"- **Recommendation**: {res['recommendation']}\n\n")

    results["phases"]["phase23"] = {
        "status": "completed",
        "gate_status": gate_status,
        "gate_score": weighted_score,
        "reservations_count": len(reservations),
        "is_submittable": gate_status != "FAIL",
    }

    # =========================================================================
    # PHASE 24: Clarifications Letter
    # =========================================================================
    phase24_dir = OUTPUT_DIR / "phase24_clarifications"
    phase24_dir.mkdir(parents=True, exist_ok=True)

    with open(phase24_dir / "clarifications_letter.md", "w") as f:
        f.write(f"# BID CLARIFICATIONS & ASSUMPTIONS\n\n")
        f.write(f"**Project**: {PROJECT_NAME}\n")
        f.write(f"**Tender Ref**: {PROJECT_ID}\n")
        f.write(f"**Date**: {datetime.now().strftime('%d-%b-%Y')}\n")
        f.write(f"**From**: [Contractor Name]\n")
        f.write(f"**To**: [Owner/Consultant Name]\n\n")
        f.write("---\n\n")

        f.write("Dear Sir/Madam,\n\n")
        f.write("We are pleased to submit our bid for the above-referenced project. ")
        f.write("The following clarifications, assumptions, and exclusions form an integral part of our offer.\n\n")

        f.write("## 1. SCOPE INCLUSIONS\n\n")
        inclusions = [
            "All RCC structural work as per approved structural drawings (STR-01, STR-02)",
            "AAC block masonry 100mm and 200mm thick as per architectural drawings",
            "Cement plaster 12mm internal and 20mm external",
            "Waterproofing to terrace (APP membrane) and toilets (integral)",
            "Vitrified tile flooring to living/dining areas (600x600mm)",
            "Ceramic tile flooring to toilets (300x300mm anti-skid)",
            "Granite flooring to lobby areas",
            "UPVC doors and windows as per door/window schedules",
            "Teak wood main entrance door",
            "Compound wall with foundation (85 rmt)",
            "Paver block driveway (180 sqm)",
            "All preliminary and general items as per prelims schedule",
        ]
        for inc in inclusions:
            f.write(f"- {inc}\n")

        f.write("\n## 2. SCOPE EXCLUSIONS\n\n")
        exclusions = [
            "Furniture, fixtures, and furnishings",
            "Modular kitchen and kitchen appliances",
            "Air conditioning / HVAC system",
            "Lift / Elevator installation",
            "Fire fighting and fire alarm systems",
            "Solar water heating / solar PV",
            "Landscaping, gardening, and soft areas",
            "Swimming pool (if any)",
            "External development beyond site boundary",
            "Government fees, liaison charges, and approvals",
            "Rock excavation (if encountered)",
            "Dewatering beyond normal conditions",
            "Any work in existing structures",
        ]
        for exc in exclusions:
            f.write(f"- {exc}\n")

        f.write("\n## 3. ASSUMPTIONS\n\n")
        assumptions = [
            "Ceiling height assumed as 3000mm (3.0m) floor to ceiling",
            "Floor to floor height assumed as 3300mm (3.3m)",
            "Slab thickness assumed as 150mm unless noted otherwise",
            "Clear working access available for material delivery and crane operation",
            "Water and electricity available at site for construction use",
            "Normal soil conditions for foundation (no rock, no high water table)",
            "All dimensions are in millimeters unless noted otherwise",
            "Tile sizes and specifications as per schedule; brand selection pending",
            "Rates are inclusive of Bangalore location factor (1.05x Delhi base)",
        ]
        for asm in assumptions:
            f.write(f"- {asm}\n")

        f.write("\n## 4. PROVISIONAL ALLOWANCES\n\n")
        f.write("The following items are included as **provisional allowances** pending receipt of detailed drawings/specifications:\n\n")
        f.write("| Item | Allowance (INR) | Basis |\n")
        f.write("|------|-----------------|-------|\n")
        f.write(f"| Internal Plumbing (complete) | ₹8,50,000 | @₹190/sqft BUA |\n")
        f.write(f"| Internal Electrical (complete) | ₹7,20,000 | @₹160/sqft BUA |\n")
        f.write(f"| **Total Provisional** | **₹15,70,000** | {(provisional_total/boq_total)*100:.1f}% of BOQ |\n\n")
        f.write("*Note: Actual cost will be reconciled upon receipt of MEP drawings.*\n")

        f.write("\n## 5. RFIs TO BE RAISED\n\n")
        f.write("We request clarification on the following items before bid finalization:\n\n")
        high_rfis = [r for r in rfis if r["priority"] == "high"]
        for rfi in high_rfis:
            f.write(f"### {rfi['rfi_id']}: {rfi['question']}\n")
            f.write(f"- **Priority**: {rfi['priority'].upper()}\n")
            f.write(f"- **Category**: {rfi['category'].title()}\n\n")

        f.write("## 6. CONFLICTS NOTED\n\n")
        conflicts = [
            {"description": "Column C5 dimensions differ between STR-01 (450x450) and ARC-01 (400x400)", "impact": "Quantity variance ~2 cum"},
            {"description": "Staircase handrail detail not provided", "impact": "Rate assumed for MS handrail with SS top rail"},
        ]
        for conf in conflicts:
            f.write(f"- **{conf['description']}**\n  - Impact: {conf['impact']}\n\n")

        f.write("## 7. RESERVATIONS\n\n")
        f.write("This bid is submitted with the following reservations:\n\n")
        for res in reservations:
            severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[res["severity"]]
            f.write(f"- {severity_icon} **{res['code']}**: {res['description']}\n")

        f.write("\n---\n\n")
        f.write("We trust the above clarifications are acceptable. We remain available for any further discussions.\n\n")
        f.write("Yours faithfully,\n\n")
        f.write("[Authorized Signatory]\n")
        f.write("[Contractor Name]\n")

    results["phases"]["phase24"] = {
        "status": "completed",
        "inclusions_count": len(inclusions),
        "exclusions_count": len(exclusions),
        "assumptions_count": len(assumptions),
        "rfis_count": len(high_rfis),
        "reservations_count": len(reservations),
    }

    # =========================================================================
    # PHASE 25: Package Outputs / RFQ Sheets
    # =========================================================================
    phase25_dir = OUTPUT_DIR / "phase25_packages"
    phase25_dir.mkdir(parents=True, exist_ok=True)

    packages = {
        "rcc": {"name": "RCC Structural Work", "items": []},
        "masonry": {"name": "Masonry & Plastering", "items": []},
        "waterproof": {"name": "Waterproofing", "items": []},
        "flooring": {"name": "Flooring & Tiling", "items": []},
        "doors_windows": {"name": "Doors & Windows", "items": []},
        "plumbing": {"name": "Plumbing Works", "items": []},
        "electrical": {"name": "Electrical Works", "items": []},
        "external": {"name": "External Development", "items": []},
    }

    for item in priced_boq:
        pkg = item.get("package", "misc")
        if pkg in packages:
            packages[pkg]["items"].append(item)

    for pkg_key, pkg_data in packages.items():
        if not pkg_data["items"]:
            continue

        pkg_dir = phase25_dir / pkg_key
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Package BOQ
        with open(pkg_dir / f"{pkg_key}_boq.csv", "w") as f:
            f.write("SL,Item ID,Description,Quantity,Unit,Rate,Amount,Drawing Ref\n")
            for i, item in enumerate(pkg_data["items"], 1):
                f.write(f'{i},{item["item_id"]},"{item["description"]}",{item["quantity"]},{item["unit"]},{item["rate"]:.2f},{item["amount"]:.2f},{item["drawing_ref"]}\n')

        # RFQ Sheet
        with open(pkg_dir / f"{pkg_key}_rfq_sheet.csv", "w") as f:
            f.write("SL,Item Description,Quantity,Unit,Specification Notes,Drawing References,Assumptions,Subcontractor Rate,Subcontractor Amount,Remarks\n")
            for i, item in enumerate(pkg_data["items"], 1):
                notes = "As per specification" if not item["is_provisional"] else "PROVISIONAL - PENDING DETAILS"
                f.write(f'{i},"{item["description"]}",{item["quantity"]},{item["unit"]},"{notes}",{item["drawing_ref"]},"Standard specification",,,"To be filled by subcontractor"\n')

        # Package scope
        pkg_total = sum(i["amount"] for i in pkg_data["items"])
        with open(pkg_dir / f"{pkg_key}_scope.md", "w") as f:
            f.write(f"# {pkg_data['name']} - Scope of Work\n\n")
            f.write(f"**Package**: {pkg_key.upper()}\n")
            f.write(f"**Items**: {len(pkg_data['items'])}\n")
            f.write(f"**Estimated Value**: ₹{pkg_total:,.2f}\n\n")
            f.write("## Items Included\n\n")
            for item in pkg_data["items"]:
                f.write(f"- {item['description']} ({item['quantity']} {item['unit']})\n")

        # Package risks
        with open(pkg_dir / f"{pkg_key}_risks.md", "w") as f:
            f.write(f"# {pkg_data['name']} - Risk Register\n\n")
            provisional_items = [i for i in pkg_data["items"] if i["is_provisional"]]
            if provisional_items:
                f.write("## ⚠️ Provisional Items\n\n")
                for item in provisional_items:
                    f.write(f"- **{item['item_id']}**: {item['description']} (₹{item['amount']:,.2f})\n")
                f.write("\n*These items are based on allowances and may vary.*\n\n")
            else:
                f.write("No provisional items in this package.\n\n")

    results["phases"]["phase25"] = {
        "status": "completed",
        "packages_created": len([p for p in packages.values() if p["items"]]),
        "total_items": len(priced_boq),
        "packages": {k: {"items": len(v["items"]), "value": sum(i["amount"] for i in v["items"])} for k, v in packages.items() if v["items"]},
    }

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    results["summary"] = {
        "phases_completed": sum(1 for p in results["phases"].values() if p.get("status") == "completed"),
        "phases_skipped": sum(1 for p in results["phases"].values() if p.get("status") == "skipped"),
        "total_bid_value": boq_total + actual_prelims,
        "bid_status": gate_status,
        "gate_score": weighted_score,
        "is_submittable": gate_status != "FAIL",
        "reservations_count": len(reservations),
        "high_priority_rfis": len([r for r in rfis if r["priority"] == "high"]),
    }

    # Save master results
    with open(OUTPUT_DIR / "bid_engine_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary.md
    generate_summary_md(results, boq_total, actual_prelims, priced_boq, prelims_items, rfis, reservations, gate_status, weighted_score)

    return results


def generate_summary_md(results, boq_total, prelims_total, priced_boq, prelims_items, rfis, reservations, gate_status, gate_score):
    """Generate the main summary markdown file."""

    grand_total = boq_total + prelims_total

    with open(OUTPUT_DIR / "summary.md", "w") as f:
        # Header
        f.write(f"# BID SUMMARY: {PROJECT_ID}\n\n")
        f.write(f"**Project**: {PROJECT_NAME}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%d-%b-%Y %H:%M')}\n\n")

        # Status banner
        status_emoji = {"PASS": "✅", "PASS_WITH_RESERVATIONS": "🟡", "FAIL": "❌"}[gate_status]
        f.write("---\n")
        if gate_status == "FAIL":
            f.write(f"## {status_emoji} BID STATUS: NOT SUBMITTABLE\n\n")
            f.write("**This bid requires clarifications before submission.**\n")
        elif gate_status == "PASS_WITH_RESERVATIONS":
            f.write(f"## {status_emoji} BID STATUS: SUBMITTABLE WITH RESERVATIONS\n\n")
            f.write("**Review reservations before submission.**\n")
        else:
            f.write(f"## {status_emoji} BID STATUS: READY FOR SUBMISSION\n")
        f.write("---\n\n")

        # Key metrics table
        f.write("## KEY METRICS\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **BOQ Total** | ₹{boq_total:,.2f} |\n")
        f.write(f"| **Prelims** | ₹{prelims_total:,.2f} ({(prelims_total/boq_total)*100:.1f}%) |\n")
        f.write(f"| **Grand Total** | ₹{grand_total:,.2f} |\n")
        f.write(f"| **Built-up Area** | 4,500 sqm (48,438 sqft) |\n")
        f.write(f"| **Rate per Sqft** | ₹{grand_total / 48438:,.2f} |\n")
        f.write(f"| **Gate Score** | {gate_score:.1f}/100 |\n")
        f.write(f"| **Reservations** | {len(reservations)} |\n")
        f.write(f"| **High Priority RFIs** | {len([r for r in rfis if r['priority'] == 'high'])} |\n")
        f.write(f"| **Provisional Value** | ₹{sum(i['amount'] for i in priced_boq if i['is_provisional']):,.2f} |\n\n")

        # BOQ Summary by Package
        f.write("## BOQ SUMMARY BY PACKAGE\n\n")
        f.write("| Package | Items | Value (INR) | % of Total |\n")
        f.write("|---------|-------|-------------|------------|\n")

        packages = {}
        for item in priced_boq:
            pkg = item.get("package", "misc")
            if pkg not in packages:
                packages[pkg] = {"items": 0, "value": 0}
            packages[pkg]["items"] += 1
            packages[pkg]["value"] += item["amount"]

        for pkg, data in sorted(packages.items(), key=lambda x: -x[1]["value"]):
            pct = (data["value"] / boq_total) * 100
            f.write(f"| {pkg.upper()} | {data['items']} | ₹{data['value']:,.2f} | {pct:.1f}% |\n")

        f.write(f"| **TOTAL** | **{len(priced_boq)}** | **₹{boq_total:,.2f}** | **100%** |\n\n")

        # Top 5 Reservations
        f.write("## TOP RESERVATIONS\n\n")
        for i, res in enumerate(reservations[:5], 1):
            severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[res["severity"]]
            f.write(f"{i}. {severity_icon} **{res['code']}**: {res['description']}\n")
        f.write("\n")

        # High Priority RFIs
        f.write("## HIGH PRIORITY RFIs\n\n")
        high_rfis = [r for r in rfis if r["priority"] == "high"]
        for rfi in high_rfis:
            f.write(f"- **{rfi['rfi_id']}**: {rfi['question']}\n")
        f.write("\n")

        # Phase Status
        f.write("## PHASE STATUS\n\n")
        f.write("| Phase | Description | Status |\n")
        f.write("|-------|-------------|--------|\n")

        phase_names = [
            ("phase16", "Owner Docs Parser"),
            ("phase17", "Owner Inputs Engine"),
            ("phase18", "BOQ Alignment"),
            ("phase19", "Pricing Engine"),
            ("phase20", "Quote Leveling"),
            ("phase21", "Prelims Generator"),
            ("phase22", "Bid Book Export"),
            ("phase23", "Bid Gate"),
            ("phase24", "Clarifications Letter"),
            ("phase25", "Package Outputs"),
        ]

        for key, name in phase_names:
            phase_data = results["phases"].get(key, {})
            status = phase_data.get("status", "not_run")
            icon = {"completed": "✅", "skipped": "⏭️", "error": "❌"}.get(status, "⬜")
            f.write(f"| {key.upper()} | {name} | {icon} {status.title()} |\n")

        f.write("\n")

        # Output Locations
        f.write("## OUTPUT LOCATIONS\n\n")
        f.write("```\n")
        f.write("output/bid_demo/\n")
        f.write("├── summary.md                    # This file\n")
        f.write("├── bid_engine_results.json       # Complete results\n")
        f.write("├── phase17_owner_inputs/\n")
        f.write("├── phase19_pricing/\n")
        f.write("│   ├── priced_boq.json\n")
        f.write("│   └── rate_analysis.json\n")
        f.write("├── phase21_prelims/\n")
        f.write("├── phase22_bidbook/\n")
        f.write("│   ├── bid_summary.json\n")
        f.write("│   └── priced_boq.csv\n")
        f.write("├── phase23_bid_gate/\n")
        f.write("│   ├── bid_gate_report.md\n")
        f.write("│   └── bid_gate_report.json\n")
        f.write("├── phase24_clarifications/\n")
        f.write("│   └── clarifications_letter.md\n")
        f.write("└── phase25_packages/\n")
        f.write("    ├── rcc/\n")
        f.write("    ├── masonry/\n")
        f.write("    ├── flooring/\n")
        f.write("    └── ...\n")
        f.write("```\n\n")

        # Next Steps
        f.write("## NEXT STEPS\n\n")
        if gate_status == "FAIL":
            f.write("1. ❌ Review `bid_gate_report.md` for critical failures\n")
            f.write("2. ❌ Resolve high-priority RFIs\n")
            f.write("3. ❌ Obtain missing information (MEP drawings, soil report)\n")
            f.write("4. ❌ Re-run bid engine after resolution\n")
        elif gate_status == "PASS_WITH_RESERVATIONS":
            f.write("1. ✅ Review reservations in `bid_gate_report.md`\n")
            f.write("2. ✅ Document all reservations in clarifications letter\n")
            f.write("3. 🔄 Obtain subcontractor quotes from `packages/` RFQ sheets\n")
            f.write("4. 🔄 Proceed with submission, noting reservations\n")
        else:
            f.write("1. ✅ Final review of bid documents\n")
            f.write("2. ✅ Obtain management approval\n")
            f.write("3. ✅ Submit bid\n")


def main():
    print("=" * 70)
    print("XBOQ BID ENGINE - STANDALONE DEMO")
    print("=" * 70)
    print(f"Project: {PROJECT_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    results = generate_phase_outputs()

    summary = results["summary"]
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Bid Status: {summary['bid_status']}")
    print(f"Gate Score: {summary['gate_score']:.1f}/100")
    print(f"Submittable: {summary['is_submittable']}")
    print(f"Reservations: {summary['reservations_count']}")
    print(f"High Priority RFIs: {summary['high_priority_rfis']}")
    print()
    print(f"Phases Completed: {summary['phases_completed']}")
    print(f"Phases Skipped: {summary['phases_skipped']}")
    print()
    print(f"Total Bid Value: ₹{summary['total_bid_value']:,.2f}")
    print()
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
