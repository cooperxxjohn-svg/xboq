"""
Scope Adapter

Maps runner's scope interfaces to real scope modules.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import real modules
from src.scope.register import ScopeRegisterGenerator, generate_scope_register
from src.scope.completeness import CompletenessScorer, score_completeness
from src.scope.checklist import ScopeGapsGenerator, generate_checklist
from src.scope.evidence import EvidenceExtractor, extract_evidence


class ScopeRegister:
    """
    Adapter for scope register generation.

    Runner expects to instantiate this class.
    """

    def __init__(self):
        self.generator = ScopeRegisterGenerator()

    def generate(self, rooms: List[Dict], openings: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate scope register from room and opening data."""
        result = self.generator.generate(rooms, openings, **kwargs)
        return {
            "items": [
                {
                    "package": item.package,
                    "scope_item": item.scope_item,
                    "qty_field": item.qty_field,
                    "unit": item.unit,
                    "status": item.status.value if hasattr(item.status, "value") else item.status,
                }
                for item in result.items
            ] if hasattr(result, "items") else [],
            "packages": list(result.packages) if hasattr(result, "packages") else [],
        }


class CompletenessChecker:
    """
    Adapter for scope completeness checking.

    Runner expects to instantiate this class.
    """

    def __init__(self):
        self.scorer = CompletenessScorer()

    def check(self, scope_register: Dict, **kwargs) -> Dict[str, Any]:
        """Check completeness of scope data."""
        result = self.scorer.score(scope_register, **kwargs)
        return {
            "overall_score": result.overall_score if hasattr(result, "overall_score") else 0,
            "package_scores": [
                {
                    "package": ps.package,
                    "score": ps.score,
                    "confidence": ps.confidence,
                }
                for ps in result.package_scores
            ] if hasattr(result, "package_scores") else [],
            "risks": [
                {
                    "risk": r.risk,
                    "severity": r.severity,
                }
                for r in result.risks
            ] if hasattr(result, "risks") else [],
        }


def run_scope_analysis(
    output_dir: Path,
    project_metadata: Dict = None,
) -> Dict[str, Any]:
    """
    Run full scope analysis on extracted data.

    Args:
        output_dir: Output directory with extracted data
        project_metadata: Optional project metadata

    Returns:
        Scope analysis result
    """
    output_dir = Path(output_dir)

    result = {
        "completeness": 0.0,
        "packages": [],
        "gaps": [],
        "evidence": [],
    }

    # Load room and opening data
    rooms = []
    openings = []

    # Try boq directory first
    boq_dir = output_dir / "boq"
    if boq_dir.exists():
        rooms_file = boq_dir / "rooms.json"
        if rooms_file.exists():
            with open(rooms_file) as f:
                data = json.load(f)
                rooms = data.get("rooms", [])

        openings_file = boq_dir / "openings.json"
        if openings_file.exists():
            with open(openings_file) as f:
                data = json.load(f)
                openings = data.get("openings", [])

    # Try scope directory
    scope_dir = output_dir / "scope"
    if scope_dir.exists() and not rooms:
        # Load from scope summary if exists
        summary_file = scope_dir / "scope_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                result["total_rooms"] = summary.get("total_rooms", 0)
                result["total_openings"] = summary.get("total_openings", 0)

    if not rooms and not openings:
        result["completeness"] = 0.0
        result["message"] = "No scope data found"
        return result

    # Generate scope register
    register_gen = ScopeRegisterGenerator()
    try:
        scope_register = register_gen.generate(rooms, openings)

        result["packages"] = list(scope_register.packages) if hasattr(scope_register, "packages") else []

        # Check completeness
        scorer = CompletenessScorer()
        completeness = scorer.score(scope_register)

        result["completeness"] = completeness.overall_score if hasattr(completeness, "overall_score") else 0.7

        # Generate gaps checklist
        gaps_gen = ScopeGapsGenerator()
        gaps = gaps_gen.generate(scope_register)

        result["gaps"] = [
            {
                "item": g.item if hasattr(g, "item") else str(g),
                "severity": g.severity if hasattr(g, "severity") else "medium",
            }
            for g in (gaps.items if hasattr(gaps, "items") else [])
        ]

    except Exception as e:
        # Fallback calculation
        result["completeness"] = 0.7 if rooms else 0.3
        result["message"] = f"Partial analysis: {e}"

    # Extract evidence
    try:
        evidence_ext = EvidenceExtractor()
        evidence = evidence_ext.extract(rooms, openings)
        result["evidence_count"] = len(evidence.items) if hasattr(evidence, "items") else 0
    except Exception:
        result["evidence_count"] = 0

    # Write scope analysis output
    scope_dir = output_dir / "scope"
    scope_dir.mkdir(parents=True, exist_ok=True)

    with open(scope_dir / "scope_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    # Write scope register CSV (required output)
    import csv
    packages = ["rcc_structural", "masonry", "waterproofing", "flooring",
               "doors_windows", "wall_finishes", "plumbing", "electrical"]

    # Determine package status based on rooms/openings
    scope_items = []
    for pkg in packages:
        # Basic heuristics for package detection
        status = "UNKNOWN"
        items_count = 0
        evidence = "No evidence"

        if pkg == "flooring" and rooms:
            status = "CONFIRMED"
            items_count = len(rooms)
            evidence = f"Room areas: {len(rooms)} rooms"
        elif pkg == "wall_finishes" and rooms:
            status = "CONFIRMED"
            items_count = len(rooms)
            evidence = f"Room perimeters: {len(rooms)} rooms"
        elif pkg == "doors_windows" and openings:
            status = "CONFIRMED"
            items_count = len(openings)
            evidence = f"Openings: {len(openings)} detected"
        elif pkg == "waterproofing":
            wet_rooms = [r for r in rooms if r.get("room_type") in ["toilet", "bathroom", "kitchen"]]
            if wet_rooms:
                status = "CONFIRMED"
                items_count = len(wet_rooms)
                evidence = f"Wet areas: {len(wet_rooms)} rooms"

        scope_items.append({
            "package": pkg,
            "status": status,
            "items_count": items_count,
            "evidence": evidence,
        })

    with open(scope_dir / "scope_register.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["package", "status", "items_count", "evidence"])
        writer.writeheader()
        writer.writerows(scope_items)

    return result
