"""
Bid Submission Engine - Phases 16-25

Integrates all bid preparation modules:
- Phase 16: Owner Docs Parser
- Phase 17: Owner Inputs Engine
- Phase 18: BOQ Alignment
- Phase 19: Rate Build-up / Pricing Engine
- Phase 20: Subcontractor Quote Leveling
- Phase 21: Prelims / General Conditions Generator
- Phase 22: Bid Book Export
- Phase 23: Bid Gate (Safety Checks)
- Phase 24: Clarifications Letter
- Phase 25: Package Outputs / RFQ Sheets

India-specific construction bidding workflow.
Never hides uncertainty - converts gaps to reservations/allowances.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def run_bid_engine(
    project_id: str,
    project_dir: Path,
    output_dir: Path,
    drawings_boq: List[Dict],
    scope_register: Dict,
    owner_docs_paths: List[Path] = None,
    subcontractor_quotes: Dict[str, List] = None,
    built_up_area_sqm: float = None,
    duration_months: int = None,
) -> Dict[str, Any]:
    """
    Run the complete Bid Submission Engine (Phases 16-22).

    Args:
        project_id: Project identifier
        project_dir: Project directory (contains owner_inputs.yaml)
        output_dir: Output directory
        drawings_boq: BOQ extracted from drawings (from Phase 11)
        scope_register: Scope register (from Phase 10)
        owner_docs_paths: Paths to owner documents (tender, specs, etc.)
        subcontractor_quotes: Dict of package -> list of quotes
        built_up_area_sqm: Built-up area (from drawings or owner input)
        duration_months: Project duration

    Returns:
        Complete bid engine results
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "project_id": project_id,
        "generated_at": datetime.now().isoformat(),
        "phases": {},
    }

    # =========================================================================
    # PHASE 16: Owner Docs Parser
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 16: Owner Docs Parser")
    logger.info("=" * 60)

    owner_docs_results = {}
    if owner_docs_paths:
        try:
            from .owner_docs import run_owner_docs_engine

            phase16_output = output_dir / "phase16_owner_docs"
            owner_docs_results = run_owner_docs_engine(
                project_id=project_id,
                doc_paths=owner_docs_paths,
                output_dir=phase16_output,
            )
            results["phases"]["phase16_owner_docs"] = {
                "status": "completed",
                "documents_parsed": owner_docs_results.get("documents_processed", 0),
                "tender_info_extracted": bool(owner_docs_results.get("tender_info")),
                "owner_boq_items": owner_docs_results.get("owner_boq_items", 0),
            }
            logger.info(f"Phase 16 complete: {owner_docs_results.get('documents_processed', 0)} documents parsed")
        except Exception as e:
            logger.error(f"Phase 16 error: {e}")
            results["phases"]["phase16_owner_docs"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Phase 16 skipped: No owner documents provided")
        results["phases"]["phase16_owner_docs"] = {"status": "skipped", "reason": "No owner documents"}

    # =========================================================================
    # PHASE 17: Owner Inputs Engine
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 17: Owner Inputs Engine")
    logger.info("=" * 60)

    try:
        from .owner_inputs import run_owner_inputs_engine

        phase17_output = output_dir / "phase17_owner_inputs"
        owner_inputs_results = run_owner_inputs_engine(
            project_id=project_id,
            project_dir=project_dir,
            output_dir=phase17_output,
            owner_docs_results=owner_docs_results,
        )

        # Extract key values for downstream phases
        final_inputs = owner_inputs_results.get("final_inputs", {})
        project_info = final_inputs.get("project", {})
        finish_grade = final_inputs.get("finishes", {}).get("grade", "standard")

        # Get built-up area and duration from inputs if not provided
        if not built_up_area_sqm:
            built_up_area_sqm = project_info.get("built_up_area_sqm", 5000)  # Default
        if not duration_months:
            duration_months = project_info.get("completion_months", 18)  # Default

        results["phases"]["phase17_owner_inputs"] = {
            "status": "completed",
            "completeness_score": owner_inputs_results.get("completeness_score", 0),
            "missing_mandatory": owner_inputs_results.get("missing_mandatory", 0),
            "defaults_applied": owner_inputs_results.get("defaults_applied", 0),
            "rfis_generated": owner_inputs_results.get("rfis_generated", 0),
        }
        logger.info(f"Phase 17 complete: {owner_inputs_results.get('completeness_score', 0):.0f}% complete")
    except Exception as e:
        logger.error(f"Phase 17 error: {e}")
        results["phases"]["phase17_owner_inputs"] = {"status": "error", "error": str(e)}
        final_inputs = {}
        project_info = {}
        finish_grade = "standard"
        if not built_up_area_sqm:
            built_up_area_sqm = 5000
        if not duration_months:
            duration_months = 18

    # =========================================================================
    # PHASE 18: BOQ Alignment
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 18: BOQ Alignment")
    logger.info("=" * 60)

    unified_boq = drawings_boq  # Default to drawings BOQ
    alignment_results = {}

    owner_boq = owner_docs_results.get("owner_boq", [])
    if owner_boq and drawings_boq:
        try:
            from .alignment import run_alignment_engine

            phase18_output = output_dir / "phase18_alignment"
            alignment_results = run_alignment_engine(
                project_id=project_id,
                drawings_boq=drawings_boq,
                owner_boq=owner_boq,
                output_dir=phase18_output,
            )
            unified_boq = alignment_results.get("unified_boq", drawings_boq)

            results["phases"]["phase18_alignment"] = {
                "status": "completed",
                "alignment_score": alignment_results.get("alignment_score", 0),
                "matched_items": alignment_results.get("matched_items", 0),
                "discrepancies": alignment_results.get("discrepancies", 0),
            }
            logger.info(f"Phase 18 complete: {alignment_results.get('alignment_score', 0):.1f}% alignment")
        except Exception as e:
            logger.error(f"Phase 18 error: {e}")
            results["phases"]["phase18_alignment"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Phase 18 skipped: No owner BOQ for comparison")
        results["phases"]["phase18_alignment"] = {"status": "skipped", "reason": "No owner BOQ"}

    # =========================================================================
    # PHASE 19: Rate Build-up / Pricing Engine
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 19: Rate Build-up / Pricing Engine")
    logger.info("=" * 60)

    try:
        from .pricing import run_pricing_engine

        phase19_output = output_dir / "phase19_pricing"
        pricing_results = run_pricing_engine(
            project_id=project_id,
            unified_boq=unified_boq if isinstance(unified_boq, list) else drawings_boq,
            owner_inputs=final_inputs,
            output_dir=phase19_output,
        )

        priced_boq = pricing_results.get("priced_boq", [])
        project_value = pricing_results.get("grand_total", 0)

        results["phases"]["phase19_pricing"] = {
            "status": "completed",
            "total_items": pricing_results.get("total_items", 0),
            "priced_items": pricing_results.get("priced_items", 0),
            "missing_rates": pricing_results.get("missing_rates", 0),
            "grand_total": pricing_results.get("grand_total", 0),
            "location_factor": pricing_results.get("location_factor", 1.0),
        }
        logger.info(f"Phase 19 complete: ₹{project_value:,.2f} total value")
    except Exception as e:
        logger.error(f"Phase 19 error: {e}")
        results["phases"]["phase19_pricing"] = {"status": "error", "error": str(e)}
        priced_boq = []
        project_value = built_up_area_sqm * 25000  # Rough estimate

    # =========================================================================
    # PHASE 20: Subcontractor Quote Leveling
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 20: Subcontractor Quote Leveling")
    logger.info("=" * 60)

    quote_results = {}
    if subcontractor_quotes:
        try:
            from .quotes import run_quote_leveling

            phase20_output = output_dir / "phase20_quotes"
            phase20_output.mkdir(parents=True, exist_ok=True)

            for package, quotes in subcontractor_quotes.items():
                if quotes:
                    # Get BOQ items for this package
                    pkg_items = [item for item in priced_boq if item.get("package") == package]

                    pkg_result = run_quote_leveling(
                        project_id=project_id,
                        quotes=quotes,
                        package=package,
                        boq_items=pkg_items,
                        output_dir=phase20_output / package,
                    )
                    quote_results[package] = pkg_result

            results["phases"]["phase20_quotes"] = {
                "status": "completed",
                "packages_leveled": len(quote_results),
                "results_by_package": {
                    pkg: {
                        "quotes_compared": r.get("quotes_leveled", 0),
                        "recommended": r.get("recommended_bidder", "N/A"),
                    }
                    for pkg, r in quote_results.items()
                },
            }
            logger.info(f"Phase 20 complete: {len(quote_results)} packages leveled")
        except Exception as e:
            logger.error(f"Phase 20 error: {e}")
            results["phases"]["phase20_quotes"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Phase 20 skipped: No subcontractor quotes provided")
        results["phases"]["phase20_quotes"] = {"status": "skipped", "reason": "No quotes provided"}

    # =========================================================================
    # PHASE 21: Prelims / General Conditions Generator
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 21: Prelims Generator")
    logger.info("=" * 60)

    try:
        from .prelims import run_prelims_engine

        phase21_output = output_dir / "phase21_prelims"
        prelims_results = run_prelims_engine(
            project_id=project_id,
            project_value=project_value,
            duration_months=duration_months,
            built_up_area_sqm=built_up_area_sqm,
            project_type=project_info.get("type", "residential"),
            output_dir=phase21_output,
            owner_inputs=final_inputs,
        )

        prelims_items = prelims_results.get("prelims_items", [])

        results["phases"]["phase21_prelims"] = {
            "status": "completed",
            "total_prelims": prelims_results.get("total_prelims", 0),
            "prelims_percent": prelims_results.get("prelims_percent", 0),
            "items_count": prelims_results.get("items_count", 0),
        }
        logger.info(f"Phase 21 complete: ₹{prelims_results.get('total_prelims', 0):,.2f} prelims ({prelims_results.get('prelims_percent', 0):.1f}%)")
    except Exception as e:
        logger.error(f"Phase 21 error: {e}")
        results["phases"]["phase21_prelims"] = {"status": "error", "error": str(e)}
        prelims_items = []

    # =========================================================================
    # PHASE 22: Bid Book Export
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 22: Bid Book Export")
    logger.info("=" * 60)

    try:
        from .bidbook import run_bidbook_export

        # Compile all bid data
        bid_data = {
            "project_id": project_id,
            "project_info": project_info,
            "priced_boq": priced_boq,
            "prelims_items": prelims_items,
            "built_up_area_sqm": built_up_area_sqm,
            "finish_grade": finish_grade,
            "location_factor": pricing_results.get("location_factor", 1.0) if "pricing_results" in dir() else 1.0,
            "applied_defaults": owner_inputs_results.get("applied_defaults", []) if "owner_inputs_results" in dir() else [],
            "missing_mandatory": owner_inputs_results.get("missing_mandatory_fields", []) if "owner_inputs_results" in dir() else [],
            "discrepancies": alignment_results.get("discrepancies", []) if alignment_results else [],
            "doubt_rfis": scope_register.get("rfis", []) if scope_register else [],
            "owner_input_rfis": [],  # Would come from Phase 17
            "scope_rfis": [],
            "allowances": owner_inputs_results.get("allowances", []) if "owner_inputs_results" in dir() else [],
            "total_rfis": sum([
                len(scope_register.get("rfis", [])) if scope_register else 0,
                owner_inputs_results.get("rfis_generated", 0) if "owner_inputs_results" in dir() else 0,
            ]),
            "assumptions": [],
            "grand_total": project_value + sum(getattr(p, 'amount', 0) if hasattr(p, 'amount') else p.get('amount', 0) for p in prelims_items),
        }

        phase22_output = output_dir / "phase22_bidbook"
        bidbook_results = run_bidbook_export(
            project_id=project_id,
            bid_data=bid_data,
            output_dir=phase22_output,
        )

        results["phases"]["phase22_bidbook"] = {
            "status": "completed",
            "outputs": bidbook_results.get("outputs", []),
            "total_bid_value": bidbook_results.get("total_bid_value", 0),
        }
        logger.info(f"Phase 22 complete: Bid book exported")
    except Exception as e:
        logger.error(f"Phase 22 error: {e}")
        results["phases"]["phase22_bidbook"] = {"status": "error", "error": str(e)}
        bid_data = {}

    # =========================================================================
    # PHASE 23: Bid Gate (Safety Checks)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 23: Bid Gate (Safety Checks)")
    logger.info("=" * 60)

    gate_result = {}
    try:
        from .bid_gate import run_bid_gate

        # Prepare gate input data
        gate_input = {
            "project_id": project_id,
            "drawings_processed": len(drawings_boq) if drawings_boq else 0,
            "priced_boq": priced_boq,
            "grand_total": bid_data.get("grand_total", 0) if bid_data else 0,
            "scale_confidence": scope_register.get("scale_confidence", 0.7) if scope_register else 0.7,
            "total_schedule_items": scope_register.get("total_schedule_items", 100) if scope_register else 100,
            "mapped_schedule_items": scope_register.get("mapped_schedule_items", 80) if scope_register else 80,
            "missing_sheets_count": scope_register.get("missing_sheets_count", 0) if scope_register else 0,
            "owner_inputs_completeness": owner_inputs_results.get("completeness_score", 50) if "owner_inputs_results" in dir() else 50,
            "high_priority_rfis_count": sum([
                len([r for r in scope_register.get("rfis", []) if r.get("priority") == "high"]) if scope_register else 0,
                len([r for r in owner_inputs_results.get("rfis", []) if r.get("priority") == "high"]) if "owner_inputs_results" in dir() else 0,
            ]),
            "external_works_status": scope_register.get("external_works_status", "unknown") if scope_register else "unknown",
            "mep_coverage_status": scope_register.get("mep_coverage_status", "unknown") if scope_register else "unknown",
            "structural_drawings_provided": scope_register.get("structural_drawings_provided", False) if scope_register else False,
            "mep_drawings_provided": scope_register.get("mep_drawings_provided", False) if scope_register else False,
            "drawings_type": scope_register.get("drawings_type", "approval") if scope_register else "approval",
            "provisional_value": sum(item.get("amount", 0) for item in priced_boq if item.get("is_provisional", False)),
            "packages_without_quotes": [pkg for pkg in ["plumbing", "electrical", "flooring", "doors_windows"] if pkg not in (subcontractor_quotes or {})],
            "items_without_rate_buildup": pricing_results.get("missing_rates", 0) if "pricing_results" in dir() else 0,
        }

        phase23_output = output_dir / "phase23_bid_gate"
        gate_result = run_bid_gate(
            project_id=project_id,
            bid_data=gate_input,
            output_dir=phase23_output,
        )

        results["phases"]["phase23_bid_gate"] = {
            "status": "completed",
            "gate_status": gate_result.get("status", "UNKNOWN"),
            "gate_score": gate_result.get("score", 0),
            "reservations_count": gate_result.get("reservations_count", 0),
            "is_submittable": gate_result.get("is_submittable", False),
        }
        logger.info(f"Phase 23 complete: Gate status = {gate_result.get('status', 'UNKNOWN')}, Score = {gate_result.get('score', 0):.1f}")
    except Exception as e:
        logger.error(f"Phase 23 error: {e}")
        results["phases"]["phase23_bid_gate"] = {"status": "error", "error": str(e)}
        gate_result = {"status": "FAIL", "is_submittable": False}

    # =========================================================================
    # PHASE 24: Clarifications Letter
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 24: Clarifications Letter")
    logger.info("=" * 60)

    try:
        from .bid_docs import run_clarifications_generator

        # Enrich bid_data for clarifications
        clarifications_data = bid_data.copy() if bid_data else {}
        clarifications_data.update({
            "inclusions": scope_register.get("inclusions", []) if scope_register else [],
            "exclusions": scope_register.get("exclusions", []) if scope_register else [],
            "missing_scope": scope_register.get("missing_scope", []) if scope_register else [],
            "conflicts": scope_register.get("conflicts", []) if scope_register else [],
            "doubt_rfis": scope_register.get("rfis", []) if scope_register else [],
            "owner_input_rfis": owner_inputs_results.get("rfis", []) if "owner_inputs_results" in dir() else [],
            "high_priority_rfis_count": gate_input.get("high_priority_rfis_count", 0) if "gate_input" in dir() else 0,
            "ceiling_height_mm": 3000,
            "floor_to_floor_mm": 3300,
            "slab_thickness_mm": 150,
        })

        phase24_output = output_dir / "phase24_clarifications"
        clarifications_result = run_clarifications_generator(
            project_id=project_id,
            bid_data=clarifications_data,
            gate_result=gate_result,
            output_dir=phase24_output,
        )

        results["phases"]["phase24_clarifications"] = {
            "status": "completed",
            "letter_generated": True,
            "inclusions_count": clarifications_result.get("inclusions_count", 0),
            "exclusions_count": clarifications_result.get("exclusions_count", 0),
        }
        logger.info(f"Phase 24 complete: Clarifications letter generated")
    except Exception as e:
        logger.error(f"Phase 24 error: {e}")
        results["phases"]["phase24_clarifications"] = {"status": "error", "error": str(e)}

    # =========================================================================
    # PHASE 25: Package Outputs / RFQ Sheets
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 25: Package Outputs / RFQ Sheets")
    logger.info("=" * 60)

    try:
        from .packages import run_package_splitter

        phase25_output = output_dir / "phase25_packages"
        packages_result = run_package_splitter(
            project_id=project_id,
            priced_boq=priced_boq,
            prelims_items=prelims_items,
            bid_data=clarifications_data if "clarifications_data" in dir() else bid_data,
            output_dir=phase25_output,
        )

        results["phases"]["phase25_packages"] = {
            "status": "completed",
            "packages_created": packages_result.get("packages_created", 0),
            "total_items": packages_result.get("total_items", 0),
            "packages": packages_result.get("packages", {}),
        }
        logger.info(f"Phase 25 complete: {packages_result.get('packages_created', 0)} packages created")
    except Exception as e:
        logger.error(f"Phase 25 error: {e}")
        results["phases"]["phase25_packages"] = {"status": "error", "error": str(e)}

    # =========================================================================
    # Final Summary with Safety Status
    # =========================================================================
    logger.info("=" * 60)
    logger.info("BID ENGINE COMPLETE")
    logger.info("=" * 60)

    # Calculate overall stats
    phases_completed = sum(1 for p in results["phases"].values() if p.get("status") == "completed")
    phases_skipped = sum(1 for p in results["phases"].values() if p.get("status") == "skipped")
    phases_errored = sum(1 for p in results["phases"].values() if p.get("status") == "error")

    # Extract safety status
    bid_status = gate_result.get("status", "UNKNOWN")
    is_submittable = gate_result.get("is_submittable", False)
    reservations_count = gate_result.get("reservations_count", 0)
    high_rfis = gate_input.get("high_priority_rfis_count", 0) if "gate_input" in dir() else 0

    results["summary"] = {
        "phases_completed": phases_completed,
        "phases_skipped": phases_skipped,
        "phases_errored": phases_errored,
        "total_bid_value": bid_data.get("grand_total", 0) if bid_data else 0,
        "output_directory": str(output_dir),
        # Safety status
        "bid_status": bid_status,
        "is_submittable": is_submittable,
        "reservations_count": reservations_count,
        "high_priority_rfis": high_rfis,
        "gate_score": gate_result.get("score", 0),
    }

    # Save master results
    with open(output_dir / "bid_engine_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate summary markdown
    _generate_summary_md(output_dir, project_id, results, gate_result)

    logger.info(f"Phases: {phases_completed} completed, {phases_skipped} skipped, {phases_errored} errored")
    logger.info(f"Bid Status: {bid_status} (Score: {gate_result.get('score', 0):.1f})")
    logger.info(f"Submittable: {is_submittable}")
    logger.info(f"Results saved to: {output_dir / 'bid_engine_results.json'}")

    return results


def _generate_summary_md(output_dir: Path, project_id: str, results: dict, gate_result: dict) -> None:
    """Generate summary.md with bid status and key metrics."""
    summary = results.get("summary", {})

    with open(output_dir / "summary.md", "w") as f:
        # Header with status
        bid_status = summary.get("bid_status", "UNKNOWN")
        status_emoji = {"PASS": "✅", "PASS_WITH_RESERVATIONS": "🟡", "FAIL": "❌"}.get(bid_status, "⚪")

        f.write(f"# Bid Summary: {project_id}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%d-%b-%Y %H:%M')}\n\n")

        # Status banner
        if bid_status == "FAIL":
            f.write("---\n")
            f.write(f"## {status_emoji} BID STATUS: NOT SUBMITTABLE\n\n")
            f.write("**This bid requires clarifications before submission.**\n\n")
            f.write("---\n\n")
        elif bid_status == "PASS_WITH_RESERVATIONS":
            f.write("---\n")
            f.write(f"## {status_emoji} BID STATUS: SUBMITTABLE WITH RESERVATIONS\n\n")
            f.write("**Review reservations before submission.**\n\n")
            f.write("---\n\n")
        else:
            f.write("---\n")
            f.write(f"## {status_emoji} BID STATUS: READY FOR SUBMISSION\n\n")
            f.write("---\n\n")

        # Key metrics
        f.write("## Key Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **Total Bid Value** | ₹{summary.get('total_bid_value', 0):,.2f} |\n")
        f.write(f"| **Gate Score** | {summary.get('gate_score', 0):.1f}/100 |\n")
        f.write(f"| **Reservations** | {summary.get('reservations_count', 0)} |\n")
        f.write(f"| **High Priority RFIs** | {summary.get('high_priority_rfis', 0)} |\n")
        f.write(f"| **Phases Completed** | {summary.get('phases_completed', 0)}/10 |\n\n")

        # Top reservations
        reservations = gate_result.get("reservations", [])
        if reservations:
            f.write("## Top Reservations\n\n")
            for i, res in enumerate(reservations[:5], 1):
                if isinstance(res, dict):
                    severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(res.get("severity", ""), "⚪")
                    f.write(f"{i}. {severity_icon} **{res.get('code', '')}**: {res.get('description', '')}\n")
            if len(reservations) > 5:
                f.write(f"\n*...and {len(reservations) - 5} more reservations*\n")
            f.write("\n")

        # Phase status
        f.write("## Phase Status\n\n")
        f.write("| Phase | Status |\n")
        f.write("|-------|--------|\n")

        phase_names = {
            "phase16_owner_docs": "Owner Docs Parser",
            "phase17_owner_inputs": "Owner Inputs Engine",
            "phase18_alignment": "BOQ Alignment",
            "phase19_pricing": "Pricing Engine",
            "phase20_quotes": "Quote Leveling",
            "phase21_prelims": "Prelims Generator",
            "phase22_bidbook": "Bid Book Export",
            "phase23_bid_gate": "Bid Gate",
            "phase24_clarifications": "Clarifications Letter",
            "phase25_packages": "Package Outputs",
        }

        for phase_key, phase_name in phase_names.items():
            phase_data = results.get("phases", {}).get(phase_key, {})
            status = phase_data.get("status", "not_run")
            status_icon = {"completed": "✅", "skipped": "⏭️", "error": "❌", "not_run": "⬜"}.get(status, "⬜")
            f.write(f"| {phase_name} | {status_icon} {status.title()} |\n")

        f.write("\n")

        # Output locations
        f.write("## Output Locations\n\n")
        f.write(f"- **Bid Book**: `phase22_bidbook/`\n")
        f.write(f"- **Gate Report**: `phase23_bid_gate/bid_gate_report.md`\n")
        f.write(f"- **Clarifications**: `phase24_clarifications/clarifications_letter.md`\n")
        f.write(f"- **Packages**: `phase25_packages/`\n\n")

        # Next steps
        f.write("## Next Steps\n\n")
        if bid_status == "FAIL":
            f.write("1. Review `bid_gate_report.md` for critical failures\n")
            f.write("2. Resolve high-priority RFIs\n")
            f.write("3. Obtain missing information\n")
            f.write("4. Re-run bid engine after resolution\n")
        elif bid_status == "PASS_WITH_RESERVATIONS":
            f.write("1. Review reservations in `bid_gate_report.md`\n")
            f.write("2. Document all reservations in clarifications letter\n")
            f.write("3. Obtain subcontractor quotes from `packages/` RFQ sheets\n")
            f.write("4. Proceed with submission, noting reservations\n")
        else:
            f.write("1. Final review of bid documents\n")
            f.write("2. Obtain management approval\n")
            f.write("3. Submit bid\n")
