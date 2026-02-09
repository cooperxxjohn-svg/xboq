"""
Bid Gate Adapter

Maps runner's bid gate interface to real risk and scope modules.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import real modules
from src.risk.pricing import RiskPricingEngine, run_risk_pricing
from src.scope.completeness import CompletenessScorer


def run_bid_gate(
    output_dir: Path,
    project_metadata: Dict = None,
) -> Dict[str, Any]:
    """
    Run bid gate assessment for a project.

    Runner expects this function.

    The bid gate determines if we have enough information to submit a bid.

    Args:
        output_dir: Output directory with project data
        project_metadata: Project metadata including owner inputs

    Returns:
        Bid gate result with status, score, and assessment
    """
    output_dir = Path(output_dir)

    result = {
        "status": "UNKNOWN",
        "score": 0,
        "is_submittable": False,
        "checks": [],
        "blockers": [],
        "warnings": [],
        "recommendations": [],
    }

    # Load bid gate rules
    rules_path = Path(__file__).parent.parent.parent / "rules" / "bid_gate.yaml"
    if rules_path.exists():
        with open(rules_path) as f:
            bid_gate_rules = yaml.safe_load(f)
    else:
        bid_gate_rules = {"min_completeness": 0.6, "required_packages": []}

    # Check 0: CRITICAL - Is measured.json empty?
    # If measured.json is empty, pricing MUST be disabled and bid MUST be NO-GO
    measured_path = output_dir / "boq" / "measured.json"
    measured_empty = True
    measured_count = 0

    if measured_path.exists():
        try:
            with open(measured_path) as f:
                measured_items = json.load(f)
                measured_count = len(measured_items) if isinstance(measured_items, list) else 0
                measured_empty = measured_count == 0
        except (json.JSONDecodeError, IOError):
            measured_empty = True

    if measured_empty:
        result["checks"].append({
            "check": "measured_quantities",
            "passed": False,
            "score": 0,
            "detail": "NO geometry-backed measurements - all quantities are inferred"
        })
        result["blockers"].append(
            "CRITICAL: measured.json is empty - no geometry-backed quantities. "
            "Pricing DISABLED. Bid recommendation: NO-GO. "
            "All quantities would be fabricated from templates."
        )
        # This is a hard blocker - mark pricing as disabled
        result["pricing_disabled"] = True
        result["pricing_disabled_reason"] = "No measured quantities available"
    else:
        result["checks"].append({
            "check": "measured_quantities",
            "passed": True,
            "score": 25,
            "detail": f"{measured_count} geometry-backed measurements"
        })
        result["score"] += 25
        result["pricing_disabled"] = False

    # Check 1: Do we have room data?
    rooms_found = False
    rooms_paths = [
        output_dir / "boq" / "rooms.json",
        output_dir / "scope" / "scope_summary.json",
    ]
    for rp in rooms_paths:
        if rp.exists():
            rooms_found = True
            break

    if rooms_found:
        result["checks"].append({"check": "rooms_data", "passed": True, "score": 20})
        result["score"] += 20
    else:
        result["checks"].append({"check": "rooms_data", "passed": False, "score": 0})
        result["blockers"].append("No room data found - cannot generate BOQ")

    # Check 2: Do we have opening data?
    openings_found = False
    openings_paths = [
        output_dir / "boq" / "openings.json",
    ]
    for op in openings_paths:
        if op.exists():
            openings_found = True
            break

    if openings_found:
        result["checks"].append({"check": "openings_data", "passed": True, "score": 15})
        result["score"] += 15
    else:
        result["checks"].append({"check": "openings_data", "passed": False, "score": 0})
        result["warnings"].append("No opening data - door/window schedule may be incomplete")

    # Check 3: Scope completeness
    scope_analysis_path = output_dir / "scope" / "scope_analysis.json"
    if scope_analysis_path.exists():
        with open(scope_analysis_path) as f:
            scope_data = json.load(f)
            completeness = scope_data.get("completeness", 0)

            if completeness >= 0.8:
                result["checks"].append({"check": "scope_completeness", "passed": True, "score": 25})
                result["score"] += 25
            elif completeness >= 0.6:
                result["checks"].append({"check": "scope_completeness", "passed": True, "score": 15})
                result["score"] += 15
                result["warnings"].append(f"Scope completeness is {completeness:.0%} - review gaps")
            else:
                result["checks"].append({"check": "scope_completeness", "passed": False, "score": 0})
                result["blockers"].append(f"Scope completeness too low ({completeness:.0%})")
    else:
        result["checks"].append({"check": "scope_completeness", "passed": False, "score": 0})
        result["warnings"].append("Scope analysis not run")

    # Check 4: Owner specifications
    owner_inputs = project_metadata.get("owner_inputs", {}) if project_metadata else {}
    if owner_inputs and owner_inputs.get("project", {}).get("name"):
        result["checks"].append({"check": "owner_specs", "passed": True, "score": 20})
        result["score"] += 20
    else:
        result["checks"].append({"check": "owner_specs", "passed": False, "score": 0})
        result["warnings"].append("Owner specifications not provided - using defaults")

    # Check 5: RFI status
    rfi_path = output_dir / "rfi" / "rfis.json"
    if rfi_path.exists():
        with open(rfi_path) as f:
            rfi_data = json.load(f)
            rfi_count = rfi_data.get("total_count", 0)
            critical_rfis = sum(1 for r in rfi_data.get("rfis", []) if r.get("priority") == "high")

            if critical_rfis == 0:
                result["checks"].append({"check": "critical_rfis", "passed": True, "score": 20})
                result["score"] += 20
            else:
                result["checks"].append({"check": "critical_rfis", "passed": False, "score": 0})
                result["blockers"].append(f"{critical_rfis} critical RFIs need resolution before bid")
    else:
        result["checks"].append({"check": "critical_rfis", "passed": True, "score": 10})
        result["score"] += 10

    # Determine status
    has_blockers = len(result["blockers"]) > 0

    if has_blockers:
        result["status"] = "NO-GO"
        result["is_submittable"] = False
    elif result["score"] >= 80:
        result["status"] = "GO"
        result["is_submittable"] = True
    elif result["score"] >= 60:
        result["status"] = "CONDITIONAL"
        result["is_submittable"] = True
        result["recommendations"].append("Review warnings before submission")
    else:
        result["status"] = "NEEDS-WORK"
        result["is_submittable"] = False
        result["recommendations"].append("Address gaps and re-run analysis")

    # Write bid gate report
    _write_bid_gate_report(output_dir, result)

    return result


def _write_bid_gate_report(output_dir: Path, result: Dict) -> None:
    """Write bid gate report files."""
    # Write JSON
    with open(output_dir / "bid_gate_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Write markdown report
    with open(output_dir / "bid_gate_report.md", "w") as f:
        status_emoji = {
            "GO": "✅",
            "CONDITIONAL": "⚠️",
            "NO-GO": "❌",
            "NEEDS-WORK": "🔧",
            "UNKNOWN": "❓",
        }

        f.write("# Bid Gate Report\n\n")
        f.write(f"**Status:** {status_emoji.get(result['status'], '❓')} {result['status']}\n")
        f.write(f"**Score:** {result['score']}/100\n")
        f.write(f"**Submittable:** {'Yes' if result['is_submittable'] else 'No'}\n")

        # Add pricing status warning if disabled
        if result.get("pricing_disabled"):
            f.write(f"\n⛔ **PRICING DISABLED:** {result.get('pricing_disabled_reason', 'No measured quantities')}\n")
            f.write("   All quantities are inferred from templates - cannot produce reliable pricing.\n\n")
        else:
            f.write("\n")

        f.write("## Checks\n\n")
        f.write("| Check | Status | Score |\n")
        f.write("|-------|--------|-------|\n")
        for check in result["checks"]:
            icon = "✅" if check["passed"] else "❌"
            f.write(f"| {check['check']} | {icon} | {check['score']} |\n")

        if result["blockers"]:
            f.write("\n## Blockers ❌\n\n")
            for blocker in result["blockers"]:
                f.write(f"- {blocker}\n")

        if result["warnings"]:
            f.write("\n## Warnings ⚠️\n\n")
            for warning in result["warnings"]:
                f.write(f"- {warning}\n")

        if result["recommendations"]:
            f.write("\n## Recommendations\n\n")
            for rec in result["recommendations"]:
                f.write(f"- {rec}\n")

        f.write("\n---\n")
        f.write(f"\n*Generated by XBOQ Bid Gate*\n")
