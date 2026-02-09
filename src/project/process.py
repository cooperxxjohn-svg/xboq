"""
Unified Project Processing Pipeline.

Phases:
1. Index all pages (fast, low-DPI)
2. Route pages to appropriate extractors
3. Run targeted extraction (parallel, high-DPI only where needed)
4. Join results into project graph
5. Export summary and BOQ
6. Analyze coverage and generate missing inputs checklist
7. Run sanity checks
8. Export provenance/audit trail
9. Run scope completeness engine
10. Run accuracy enhancement engine (triangulation, overrides, paranoia)
11. Run RFI engine for bid-ready clarifications
12. Run revision intelligence (detect revisions, track changes)
13. Run detail page understanding (implicit scope from details)
14. Run BOM derivation (material quantities from BOQ)
15. Run estimator doubt engine (missing sheets detection)

Usage:
    python -m src.project.process --input ./data/project_set --output ./out --project_id MyProject
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Set, List, Dict, Any

from .indexer import ProjectIndexer, ProjectIndex
from .router import PageRouter, RoutingResult
from .runner import ProjectRunner, RunnerConfig, RunnerResult
from .joiner import ProjectJoiner, ProjectGraph
from .exporter import ProjectExporter
from .coverage import CoverageAnalyzer, analyze_coverage
from .sanity import SanityChecker, run_sanity_checks
from .provenance import ProvenanceTracker, create_tracker

# Import scope engine
try:
    from ..scope import run_scope_engine
    SCOPE_ENGINE_AVAILABLE = True
except ImportError:
    SCOPE_ENGINE_AVAILABLE = False

# Import accuracy engine
try:
    from ..accuracy import run_accuracy_engine, AccuracyReport
    ACCURACY_ENGINE_AVAILABLE = True
except ImportError:
    ACCURACY_ENGINE_AVAILABLE = False

# Import RFI engine
try:
    from ..rfi import run_rfi_engine
    RFI_ENGINE_AVAILABLE = True
except ImportError:
    RFI_ENGINE_AVAILABLE = False

# Import revision intelligence
try:
    from ..revision import run_revision_engine
    REVISION_ENGINE_AVAILABLE = True
except ImportError:
    REVISION_ENGINE_AVAILABLE = False

# Import detail understanding
try:
    from ..details import run_detail_engine
    DETAIL_ENGINE_AVAILABLE = True
except ImportError:
    DETAIL_ENGINE_AVAILABLE = False

# Import BOM derivation
try:
    from ..materials import run_bom_engine
    BOM_ENGINE_AVAILABLE = True
except ImportError:
    BOM_ENGINE_AVAILABLE = False

# Import doubt engine
try:
    from ..doubt import run_doubt_engine
    DOUBT_ENGINE_AVAILABLE = True
except ImportError:
    DOUBT_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


def process_project_set(
    input_path: Path,
    output_dir: Path,
    project_id: Optional[str] = None,
    enable_structural: bool = False,
    only_types: Optional[Set[str]] = None,
    resume: bool = True,
    max_workers: int = 0,
    dpi: int = 300,
    thumb_dpi: int = 100,
    default_scale: int = 100,
    save_debug_images: bool = False,
) -> dict:
    """
    Process a complete project drawing set.

    Args:
        input_path: Path to folder or PDF
        output_dir: Output directory
        project_id: Project identifier (default: folder/file name)
        enable_structural: Enable structural extraction
        only_types: Process only specific types (e.g., {"floor_plan", "schedule_table"})
        resume: Resume from cached results
        max_workers: Max parallel workers (0 = auto)
        dpi: DPI for full extraction
        thumb_dpi: DPI for indexing thumbnails
        default_scale: Default drawing scale
        save_debug_images: Save debug visualizations

    Returns:
        Dict with result summary and output paths
    """
    total_start = time.time()

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if project_id is None:
        project_id = input_path.stem

    project_dir = output_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"PROCESSING PROJECT: {project_id}")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {project_dir}")

    # =========================================================================
    # PHASE 0: INDEXING
    # =========================================================================
    logger.info("\n[PHASE 0] INDEXING")
    logger.info("-" * 40)

    indexer = ProjectIndexer(thumb_dpi=thumb_dpi)
    project_index = indexer.index_project(input_path, output_dir, project_id)

    logger.info(f"Indexed {project_index.total_pages} pages from {project_index.total_files} files")
    logger.info(f"Cache: {project_index.cache_hits} hits, {project_index.cache_misses} misses")

    # =========================================================================
    # PHASE 1: ROUTING
    # =========================================================================
    logger.info("\n[PHASE 1] ROUTING")
    logger.info("-" * 40)

    router = PageRouter()
    routing_result = router.route_project(project_index, output_dir)

    logger.info(f"Routing summary: {routing_result.routing_summary}")

    # =========================================================================
    # PHASE 2: EXTRACTION
    # =========================================================================
    logger.info("\n[PHASE 2] EXTRACTION")
    logger.info("-" * 40)

    runner_config = RunnerConfig(
        dpi=dpi,
        default_scale=default_scale,
        enable_dimension_scale=True,
        enable_structural=enable_structural,
        only_types=only_types,
        max_workers=max_workers,
        resume=resume,
        save_debug_images=save_debug_images,
    )

    runner = ProjectRunner(runner_config)
    runner_result = runner.run(project_index, routing_result, output_dir)

    logger.info(f"Processed: {runner_result.processed_pages}")
    logger.info(f"Skipped: {runner_result.skipped_pages}")
    logger.info(f"Failed: {runner_result.failed_pages}")
    logger.info(f"Extraction time: {runner_result.total_time_sec:.1f}s")

    # =========================================================================
    # PHASE 3: JOINING
    # =========================================================================
    logger.info("\n[PHASE 3] JOINING")
    logger.info("-" * 40)

    joiner = ProjectJoiner()
    project_graph = joiner.join(runner_result, output_dir)

    logger.info(f"Tag matches: {len(project_graph.tag_matches)}")
    logger.info(f"Unresolved drawing tags: {len(project_graph.unresolved_drawing_tags)}")
    logger.info(f"Unresolved schedule tags: {len(project_graph.unresolved_schedule_tags)}")

    # =========================================================================
    # PHASE 4: EXPORT (Basic)
    # =========================================================================
    logger.info("\n[PHASE 4] EXPORT")
    logger.info("-" * 40)

    exporter = ProjectExporter()
    output_paths = exporter.export_all(
        project_index, routing_result, runner_result, project_graph, output_dir
    )

    logger.info(f"Basic exports: {list(output_paths.keys())}")

    # =========================================================================
    # PHASE 5: COVERAGE ANALYSIS
    # =========================================================================
    logger.info("\n[PHASE 5] COVERAGE ANALYSIS")
    logger.info("-" * 40)

    # Prepare data for coverage analysis
    extraction_results = [r.to_dict() for r in runner_result.extraction_results]
    project_graph_dict = project_graph.to_dict()

    # Build BOQ entries list from exporter data
    boq_entries = _build_boq_entries(runner_result, project_graph)

    # Run coverage analysis
    coverage_report, missing_inputs = analyze_coverage(
        project_id=project_id,
        extraction_results=extraction_results,
        project_graph=project_graph_dict,
        boq_entries=boq_entries,
        output_dir=project_dir,
    )

    logger.info(f"Coverage score: {coverage_report.overall_score:.0f}/100 (Grade: {coverage_report.grade})")
    logger.info(f"Missing inputs: {len(missing_inputs)} items")
    output_paths["coverage_report"] = project_dir / "coverage_report.json"
    output_paths["missing_inputs"] = project_dir / "missing_inputs.md"

    # =========================================================================
    # PHASE 6: SANITY CHECKS
    # =========================================================================
    logger.info("\n[PHASE 6] SANITY CHECKS")
    logger.info("-" * 40)

    sanity_report = run_sanity_checks(
        project_id=project_id,
        extraction_results=extraction_results,
        project_graph=project_graph_dict,
        output_dir=project_dir,
        coverage_report=coverage_report.to_dict(),
    )

    logger.info(f"Sanity status: {sanity_report.overall_status.value}")
    logger.info(f"Checks: {sanity_report.pass_count} PASS, {sanity_report.warn_count} WARN, {sanity_report.fail_count} FAIL")
    output_paths["sanity_checks"] = project_dir / "sanity_checks.md"

    # =========================================================================
    # PHASE 7: PROVENANCE EXPORT
    # =========================================================================
    logger.info("\n[PHASE 7] PROVENANCE EXPORT")
    logger.info("-" * 40)

    provenance_tracker = create_tracker(project_id)
    _populate_provenance(provenance_tracker, runner_result)
    provenance_path = provenance_tracker.export(project_dir)
    provenance_summary = provenance_tracker.get_summary()

    logger.info(f"Provenance records: {provenance_summary['total_records']}")
    logger.info(f"Measured: {provenance_summary['measured_pct']:.0f}%")
    output_paths["provenance"] = provenance_path

    # =========================================================================
    # PHASE 8: ENHANCED SUMMARY
    # =========================================================================
    logger.info("\n[PHASE 8] ENHANCED SUMMARY")
    logger.info("-" * 40)

    _update_summary_with_risks(
        project_dir / "summary.md",
        coverage_report,
        sanity_report,
        missing_inputs,
    )
    logger.info("Updated summary with risks and recommendations")

    # =========================================================================
    # PHASE 9: SCOPE COMPLETENESS ENGINE
    # =========================================================================
    scope_result = None
    if SCOPE_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 9] SCOPE COMPLETENESS ENGINE")
        logger.info("-" * 40)

        try:
            # Prepare page index for scope engine
            page_index_list = [p.to_dict() for p in project_index.pages]
            routing_manifest_dict = routing_result.to_dict()

            scope_result = run_scope_engine(
                project_id=project_id,
                page_index=page_index_list,
                routing_manifest=routing_manifest_dict,
                extraction_results=extraction_results,
                project_graph=project_graph_dict,
                boq_entries=boq_entries,
                output_dir=project_dir,
            )

            logger.info(f"Evidence extracted: {scope_result['evidence_count']}")
            logger.info(f"Scope items: {scope_result['scope_items']}")
            logger.info(f"Gaps identified: {scope_result['gaps_count']}")
            logger.info(f"Checklist items: {scope_result['checklist_count']}")
            logger.info(f"Provisional items: {scope_result['provisionals_count']}")
            logger.info(f"Completeness: {scope_result['completeness_score']:.0f}/100 (Grade: {scope_result['completeness_grade']})")

            output_paths["scope_evidence"] = project_dir / "scope" / "evidence.json"
            output_paths["scope_register"] = project_dir / "scope" / "scope_register.csv"
            output_paths["scope_gaps"] = project_dir / "scope" / "scope_gaps.md"
            output_paths["estimator_checklist"] = project_dir / "scope" / "estimator_checklist.md"
            output_paths["provisional_items"] = project_dir / "boq" / "provisional_items.csv"
            output_paths["completeness_report"] = project_dir / "scope" / "completeness_report.json"

            # Update summary with scope completeness
            _update_summary_with_scope(
                project_dir / "summary.md",
                scope_result,
            )

        except Exception as e:
            logger.warning(f"Scope engine failed: {e}")
            scope_result = None
    else:
        logger.info("\n[PHASE 9] SCOPE ENGINE (skipped - not available)")

    # =========================================================================
    # PHASE 10: ACCURACY ENHANCEMENT ENGINE
    # =========================================================================
    accuracy_result = None
    if ACCURACY_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 10] ACCURACY ENHANCEMENT ENGINE")
        logger.info("-" * 40)

        try:
            # Extract schedules, notes, and legends from extraction results
            schedules = _extract_schedules(extraction_results)
            notes = _extract_notes(extraction_results)
            legends = _extract_legends(extraction_results)

            # Get scope register if available
            scope_register_dict = {}
            if scope_result:
                scope_register_path = project_dir / "scope" / "scope_register.json"
                if scope_register_path.exists():
                    import json
                    with open(scope_register_path) as f:
                        scope_register_dict = json.load(f)

            accuracy_result = run_accuracy_engine(
                project_id=project_id,
                extraction_results=extraction_results,
                project_graph=project_graph_dict,
                boq_entries=boq_entries,
                schedules=schedules,
                notes=notes,
                legends=legends,
                scope_register=scope_register_dict,
                output_dir=project_dir,
            )

            logger.info(f"Triangulation: {accuracy_result.combined_summary.get('triangulation', {}).get('quantities_verified', 0)} quantities verified")
            logger.info(f"Overrides: {accuracy_result.combined_summary.get('overrides', {}).get('total_overrides', 0)} applied")
            logger.info(f"Paranoia: {accuracy_result.combined_summary.get('paranoia', {}).get('items_inferred', 0)} items inferred")
            logger.info(f"Accuracy confidence: {accuracy_result.confidence_score:.0f}/100 (Grade: {accuracy_result.confidence_grade})")

            output_paths["triangulation"] = project_dir / "accuracy" / "triangulation.json"
            output_paths["overrides"] = project_dir / "accuracy" / "overrides.json"
            output_paths["paranoia"] = project_dir / "accuracy" / "paranoia.json"
            output_paths["accuracy_summary"] = project_dir / "accuracy" / "accuracy_summary.json"
            output_paths["discrepancies"] = project_dir / "accuracy" / "discrepancies.md"

            # Update summary with accuracy info
            _update_summary_with_accuracy(
                project_dir / "summary.md",
                accuracy_result,
            )

        except Exception as e:
            logger.warning(f"Accuracy engine failed: {e}")
            import traceback
            traceback.print_exc()
            accuracy_result = None
    else:
        logger.info("\n[PHASE 10] ACCURACY ENGINE (skipped - not available)")

    # =========================================================================
    # PHASE 11: RFI ENGINE
    # =========================================================================
    rfi_result = None
    if RFI_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 11] RFI ENGINE")
        logger.info("-" * 40)

        try:
            # Get scope register and completeness report
            scope_register_dict = {}
            completeness_report_dict = {}
            if scope_result:
                scope_register_path = project_dir / "scope" / "scope_register.json"
                completeness_path = project_dir / "scope" / "completeness_report.json"
                if scope_register_path.exists():
                    import json
                    with open(scope_register_path) as f:
                        scope_register_dict = json.load(f)
                if completeness_path.exists():
                    import json
                    with open(completeness_path) as f:
                        completeness_report_dict = json.load(f)

            # Get triangulation and override reports
            triangulation_dict = {}
            override_dict = {}
            if accuracy_result:
                tri_path = project_dir / "accuracy" / "triangulation.json"
                ovr_path = project_dir / "accuracy" / "overrides.json"
                if tri_path.exists():
                    import json
                    with open(tri_path) as f:
                        triangulation_dict = json.load(f)
                if ovr_path.exists():
                    import json
                    with open(ovr_path) as f:
                        override_dict = json.load(f)

            # Get evidence data
            evidence_data = []
            evidence_path = project_dir / "scope" / "evidence.json"
            if evidence_path.exists():
                import json
                with open(evidence_path) as f:
                    evidence_data = json.load(f)

            # Prepare page index
            page_index_list = [p.to_dict() for p in project_index.pages]

            # Prepare missing inputs as list of dicts
            missing_inputs_list = []
            for item in missing_inputs:
                missing_inputs_list.append({
                    "type": item.category if hasattr(item, 'category') else "missing_input",
                    "item": item.item if hasattr(item, 'item') else str(item),
                    "severity": item.severity if hasattr(item, 'severity') else "medium",
                    "suggestion": item.suggestion if hasattr(item, 'suggestion') else "",
                })

            rfi_result = run_rfi_engine(
                project_id=project_id,
                extraction_results=extraction_results,
                project_graph=project_graph_dict,
                page_index=page_index_list,
                scope_register=scope_register_dict,
                completeness_report=completeness_report_dict,
                coverage_report=coverage_report.to_dict(),
                missing_inputs=missing_inputs_list,
                triangulation_report=triangulation_dict,
                override_report=override_dict,
                evidence_data=evidence_data,
                schedules=schedules,
                notes=notes,
                legends=legends,
                output_dir=project_dir,
            )

            logger.info(f"Total RFIs: {rfi_result['total_rfis']}")
            logger.info(f"High priority: {rfi_result['high_priority']}")
            logger.info(f"Conflicts detected: {rfi_result['conflicts']}")
            logger.info(f"Missing references: {rfi_result['missing_references']}")

            output_paths["rfi_log_csv"] = project_dir / "rfi" / "rfi_log.csv"
            output_paths["rfi_log_md"] = project_dir / "rfi" / "rfi_log.md"
            output_paths["rfi_summary"] = project_dir / "rfi" / "rfi_summary.json"

            # Update summary with RFI info
            _update_summary_with_rfis(
                project_dir / "summary.md",
                rfi_result,
            )

        except Exception as e:
            logger.warning(f"RFI engine failed: {e}")
            import traceback
            traceback.print_exc()
            rfi_result = None
    else:
        logger.info("\n[PHASE 11] RFI ENGINE (skipped - not available)")

    # =========================================================================
    # PHASE 12: REVISION INTELLIGENCE
    # =========================================================================
    revision_result = None
    if REVISION_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 12] REVISION INTELLIGENCE")
        logger.info("-" * 40)

        try:
            revision_result = run_revision_engine(
                project_id=project_id,
                current_pages=page_index_list,
                previous_pages=[],  # No previous version in single run
                extraction_results=extraction_results,
                previous_extraction_results=[],
                scope_register=scope_register_dict,
                boq_entries=boq_entries,
                output_dir=project_dir,
            )

            logger.info(f"Sheets with revisions: {revision_result['sheets_with_revisions']}")
            logger.info(f"Total revisions: {revision_result['total_revisions']}")
            logger.info(f"Latest revision: {revision_result['latest_revision']}")

            output_paths["revision_history"] = project_dir / "revision_history.json"
            output_paths["revision_report"] = project_dir / "revision_report.md"

        except Exception as e:
            logger.warning(f"Revision engine failed: {e}")
            import traceback
            traceback.print_exc()
            revision_result = None
    else:
        logger.info("\n[PHASE 12] REVISION INTELLIGENCE (skipped - not available)")

    # =========================================================================
    # PHASE 13: DETAIL PAGE UNDERSTANDING
    # =========================================================================
    detail_result = None
    if DETAIL_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 13] DETAIL PAGE UNDERSTANDING")
        logger.info("-" * 40)

        try:
            # Get current RFI list
            current_rfis = []
            if rfi_result:
                rfi_summary_path = project_dir / "rfi" / "rfi_summary.json"
                if rfi_summary_path.exists():
                    import json
                    with open(rfi_summary_path) as f:
                        rfi_data = json.load(f)
                        current_rfis = rfi_data.get("rfis", [])

            detail_result = run_detail_engine(
                project_id=project_id,
                extraction_results=extraction_results,
                scope_register=scope_register_dict,
                rfi_list=current_rfis,
                output_dir=project_dir / "details",
            )

            logger.info(f"Detail pages found: {detail_result['detail_pages']}")
            logger.info(f"Scope items promoted: {detail_result['scope_items_promoted']}")
            logger.info(f"RFIs reduced: {detail_result['rfis_reduced']}")

            output_paths["detail_classifications"] = project_dir / "details" / "detail_classifications.json"
            output_paths["detail_report"] = project_dir / "details" / "detail_report.md"

        except Exception as e:
            logger.warning(f"Detail engine failed: {e}")
            import traceback
            traceback.print_exc()
            detail_result = None
    else:
        logger.info("\n[PHASE 13] DETAIL ENGINE (skipped - not available)")

    # =========================================================================
    # PHASE 14: BOM DERIVATION
    # =========================================================================
    bom_result = None
    if BOM_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 14] BOM DERIVATION")
        logger.info("-" * 40)

        try:
            bom_result = run_bom_engine(
                project_id=project_id,
                boq_entries=boq_entries,
                scope_register=scope_register_dict,
                output_dir=project_dir,
            )

            logger.info(f"BOQ items processed: {bom_result['boq_items_processed']}")
            logger.info(f"Material categories: {bom_result['material_categories']}")
            logger.info(f"Total cement: {bom_result['total_cement_bags']:.0f} bags")
            logger.info(f"Total steel: {bom_result['total_steel_kg']:.0f} kg")

            output_paths["material_estimate"] = project_dir / "boq" / "material_estimate.csv"
            output_paths["material_summary"] = project_dir / "boq" / "material_summary.md"

        except Exception as e:
            logger.warning(f"BOM engine failed: {e}")
            import traceback
            traceback.print_exc()
            bom_result = None
    else:
        logger.info("\n[PHASE 14] BOM ENGINE (skipped - not available)")

    # =========================================================================
    # PHASE 15: ESTIMATOR DOUBT ENGINE
    # =========================================================================
    doubt_result = None
    if DOUBT_ENGINE_AVAILABLE:
        logger.info("\n[PHASE 15] ESTIMATOR DOUBT ENGINE")
        logger.info("-" * 40)

        try:
            routing_manifest_dict = routing_result.to_dict()

            doubt_result = run_doubt_engine(
                project_id=project_id,
                page_index=page_index_list,
                routing_manifest=routing_manifest_dict,
                extraction_results=extraction_results,
                scope_register=scope_register_dict,
                output_dir=project_dir,
            )

            logger.info(f"Missing sheets: {doubt_result['missing_sheets']}")
            logger.info(f"Critical missing: {doubt_result['critical_missing']}")
            logger.info(f"Completeness score: {doubt_result['completeness_score']:.0f}/100 (Grade: {doubt_result['completeness_grade']})")
            logger.info(f"Doubt RFIs: {doubt_result['doubt_rfis']}")

            output_paths["doubt_report"] = project_dir / "doubt_report.md"
            output_paths["doubt_rfis"] = project_dir / "doubt_rfis.json"

            # Update summary with doubt info
            _update_summary_with_doubt(
                project_dir / "summary.md",
                doubt_result,
            )

        except Exception as e:
            logger.warning(f"Doubt engine failed: {e}")
            import traceback
            traceback.print_exc()
            doubt_result = None
    else:
        logger.info("\n[PHASE 15] DOUBT ENGINE (skipped - not available)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - total_start

    logger.info("\n" + "=" * 60)
    logger.info("PROJECT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Project: {project_id}")
    logger.info(f"Total pages: {project_index.total_pages}")
    logger.info(f"Floor plans processed: {runner_result.summary.get('floor_plans_processed', 0)}")
    logger.info(f"Rooms detected: {runner_result.summary.get('total_rooms_detected', 0)}")
    logger.info(f"Total area: {runner_result.summary.get('total_area_sqm', 0):.1f} sqm")
    logger.info(f"Coverage grade: {coverage_report.grade}")
    logger.info(f"Sanity status: {sanity_report.overall_status.value}")
    if scope_result:
        logger.info(f"Scope completeness: {scope_result['completeness_grade']}")
    if accuracy_result:
        logger.info(f"Accuracy confidence: {accuracy_result.confidence_grade}")
    if rfi_result:
        logger.info(f"RFIs generated: {rfi_result['total_rfis']} ({rfi_result['high_priority']} high priority)")
    if revision_result:
        logger.info(f"Revisions: {revision_result['total_revisions']} (latest: {revision_result['latest_revision']})")
    if detail_result:
        logger.info(f"Details: {detail_result['detail_pages']} pages, {detail_result['scope_items_promoted']} items promoted")
    if bom_result:
        logger.info(f"BOM: {bom_result['material_categories']} material categories")
    if doubt_result:
        logger.info(f"Doubt: {doubt_result['completeness_grade']} completeness, {doubt_result['critical_missing']} critical missing")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Output: {project_dir}")

    result = {
        "project_id": project_id,
        "total_pages": project_index.total_pages,
        "processed_pages": runner_result.processed_pages,
        "total_rooms": runner_result.summary.get("total_rooms_detected", 0),
        "total_area_sqm": runner_result.summary.get("total_area_sqm", 0),
        "coverage_score": coverage_report.overall_score,
        "coverage_grade": coverage_report.grade,
        "sanity_status": sanity_report.overall_status.value,
        "total_time_sec": total_time,
        "output_dir": str(project_dir),
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }

    if scope_result:
        result["scope_completeness_score"] = scope_result["completeness_score"]
        result["scope_completeness_grade"] = scope_result["completeness_grade"]
        result["scope_gaps"] = scope_result["gaps_count"]
        result["scope_provisionals"] = scope_result["provisionals_count"]

    if accuracy_result:
        result["accuracy_confidence_score"] = accuracy_result.confidence_score
        result["accuracy_confidence_grade"] = accuracy_result.confidence_grade
        result["triangulation_agreement"] = accuracy_result.combined_summary.get("triangulation", {}).get("agreement_score", 0)
        result["overrides_applied"] = accuracy_result.combined_summary.get("overrides", {}).get("total_overrides", 0)
        result["items_inferred"] = accuracy_result.combined_summary.get("paranoia", {}).get("items_inferred", 0)
        result["discrepancies_for_review"] = len(accuracy_result.discrepancies_for_review)

    if rfi_result:
        result["total_rfis"] = rfi_result["total_rfis"]
        result["high_priority_rfis"] = rfi_result["high_priority"]
        result["conflicts_detected"] = rfi_result["conflicts"]
        result["missing_references"] = rfi_result["missing_references"]

    if revision_result:
        result["sheets_with_revisions"] = revision_result["sheets_with_revisions"]
        result["total_revisions"] = revision_result["total_revisions"]
        result["latest_revision"] = revision_result["latest_revision"]

    if detail_result:
        result["detail_pages"] = detail_result["detail_pages"]
        result["scope_items_promoted"] = detail_result["scope_items_promoted"]
        result["rfis_reduced"] = detail_result["rfis_reduced"]

    if bom_result:
        result["material_categories"] = bom_result["material_categories"]
        result["total_cement_bags"] = bom_result["total_cement_bags"]
        result["total_steel_kg"] = bom_result["total_steel_kg"]

    if doubt_result:
        result["drawing_completeness_score"] = doubt_result["completeness_score"]
        result["drawing_completeness_grade"] = doubt_result["completeness_grade"]
        result["critical_sheets_missing"] = doubt_result["critical_missing"]
        result["doubt_rfis"] = doubt_result["doubt_rfis"]

    return result


def _extract_schedules(extraction_results: List[Dict]) -> List[Dict]:
    """Extract schedule tables from extraction results."""
    schedules = []
    for result in extraction_results:
        page_id = f"{Path(result.get('file_path', '')).stem}_p{result.get('page_number', 0) + 1}"

        # Check for schedule data
        if result.get("page_type") == "schedule_table":
            schedule_data = result.get("schedule_data", {})
            if schedule_data:
                schedules.append({
                    "type": schedule_data.get("type", "unknown"),
                    "page_id": page_id,
                    "entries": schedule_data.get("entries", []),
                })

        # Also check for embedded schedules in floor plans
        for schedule in result.get("schedules", []):
            schedules.append({
                "type": schedule.get("type", "unknown"),
                "page_id": page_id,
                "entries": schedule.get("entries", []),
            })

    return schedules


def _extract_notes(extraction_results: List[Dict]) -> List[Dict]:
    """Extract notes from extraction results."""
    notes = []
    for result in extraction_results:
        page_id = f"{Path(result.get('file_path', '')).stem}_p{result.get('page_number', 0) + 1}"

        # Check for notes
        for note in result.get("notes", []):
            notes.append({
                "text": note.get("text", ""),
                "page_id": page_id,
                "type": note.get("type", "general"),
            })

        # Also check title_block for notes
        title_block = result.get("title_block", {})
        if title_block.get("notes"):
            notes.append({
                "text": title_block["notes"],
                "page_id": page_id,
                "type": "title_block",
            })

        # Check for text annotations
        for text_item in result.get("text_items", []):
            text = text_item.get("text", "").strip()
            # Filter for note-like content
            if any(kw in text.lower() for kw in ["note", "spec", "waterproof", "thickness", "height", "finish"]):
                notes.append({
                    "text": text,
                    "page_id": page_id,
                    "type": "annotation",
                })

    return notes


def _extract_legends(extraction_results: List[Dict]) -> List[Dict]:
    """Extract legends from extraction results."""
    legends = []
    for result in extraction_results:
        page_id = f"{Path(result.get('file_path', '')).stem}_p{result.get('page_number', 0) + 1}"

        # Check for legend data
        for legend in result.get("legends", []):
            legends.append({
                "type": legend.get("type", "unknown"),
                "page_id": page_id,
                "items": legend.get("items", []),
            })

        # Check for finish legends
        if result.get("finish_legend"):
            legends.append({
                "type": "finish",
                "page_id": page_id,
                "items": result["finish_legend"],
            })

    return legends


def _update_summary_with_rfis(
    summary_path: Path,
    rfi_result: Dict[str, Any],
) -> None:
    """Update summary.md with RFI info."""
    with open(summary_path, "a") as f:
        f.write("\n## RFI Summary\n\n")
        f.write(f"- **Total RFIs**: {rfi_result['total_rfis']}\n")
        f.write(f"- **High Priority**: {rfi_result['high_priority']}\n")
        f.write(f"- **Conflicts Detected**: {rfi_result['conflicts']}\n")
        f.write(f"- **Missing References**: {rfi_result['missing_references']}\n\n")

        # Priority breakdown
        by_priority = rfi_result.get("by_priority", {})
        if by_priority:
            f.write("### RFIs by Priority\n\n")
            f.write("| Priority | Count |\n")
            f.write("|----------|-------|\n")
            for priority in ["high", "medium", "low"]:
                count = by_priority.get(priority, 0)
                emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(priority, "⚪")
                f.write(f"| {emoji} {priority.title()} | {count} |\n")
            f.write("\n")

        # Top 10 RFIs
        top_rfis = rfi_result.get("top_rfis", [])
        if top_rfis:
            f.write("### Top 10 RFIs\n\n")
            for i, rfi in enumerate(top_rfis[:10], 1):
                priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.get("priority", ""), "⚪")
                f.write(f"{i}. {priority_emoji} **[{rfi.get('id', '')}]** ({rfi.get('package', '')}): {rfi.get('question', '')}\n")
            f.write("\n")

        f.write("### RFI Outputs\n\n")
        f.write("- `rfi/rfi_log.csv` - Complete RFI log (CSV)\n")
        f.write("- `rfi/rfi_log.md` - RFI log (Markdown)\n")
        f.write("- `rfi/rfi_summary.json` - RFI summary (JSON)\n")


def _update_summary_with_accuracy(
    summary_path: Path,
    accuracy_result: "AccuracyReport",
) -> None:
    """Update summary.md with accuracy enhancement info."""
    with open(summary_path, "a") as f:
        f.write("\n## Accuracy Enhancement\n\n")
        f.write(f"- **Confidence Score**: {accuracy_result.confidence_score:.0f}/100 (Grade: {accuracy_result.confidence_grade})\n")

        tri_summary = accuracy_result.combined_summary.get("triangulation", {})
        ovr_summary = accuracy_result.combined_summary.get("overrides", {})
        par_summary = accuracy_result.combined_summary.get("paranoia", {})

        f.write(f"- **Quantities Triangulated**: {tri_summary.get('quantities_verified', 0)} items\n")
        f.write(f"- **Overrides Applied**: {ovr_summary.get('total_overrides', 0)} corrections\n")
        f.write(f"- **Items Inferred**: {par_summary.get('items_inferred', 0)} from paranoia rules\n")
        f.write(f"- **Discrepancies for Review**: {len(accuracy_result.discrepancies_for_review)} items\n\n")

        # Triangulation agreement table
        if accuracy_result.triangulation and accuracy_result.triangulation.results:
            f.write("### Triangulation Agreement\n\n")
            f.write("| Quantity | Final Value | Agreement | Variance |\n")
            f.write("|----------|-------------|-----------|----------|\n")

            for result in accuracy_result.triangulation.results:
                agreement_emoji = {
                    "EXCELLENT": "✅",
                    "GOOD": "🟢",
                    "FAIR": "🟡",
                    "POOR": "🟠",
                    "DISCREPANCY": "🔴",
                }.get(result.agreement_level.value, "⚪")

                f.write(f"| {result.quantity_name} | {result.final_value} {result.unit} | "
                        f"{agreement_emoji} {result.agreement_level.value} | {result.variance_pct}% |\n")
            f.write("\n")

        # Discrepancies requiring review
        if accuracy_result.discrepancies_for_review:
            f.write("### Items Requiring Review\n\n")
            high_priority = [d for d in accuracy_result.discrepancies_for_review if d.get("severity") == "HIGH"]
            for item in high_priority[:5]:
                f.write(f"- 🔴 **{item.get('item', 'Unknown')}**: {item.get('action', '')}\n")
            if len(high_priority) > 5:
                f.write(f"- ... and {len(high_priority) - 5} more high priority items\n")
            f.write("\n")

        # Overrides applied
        if accuracy_result.overrides and accuracy_result.overrides.overrides:
            f.write("### Overrides Applied\n\n")
            for override in accuracy_result.overrides.overrides[:5]:
                f.write(f"- **{override.target}**: {override.original_value} → {override.override_value}\n")
            if len(accuracy_result.overrides.overrides) > 5:
                f.write(f"- ... and {len(accuracy_result.overrides.overrides) - 5} more overrides\n")
            f.write("\n")

        # Inferred items summary
        if accuracy_result.paranoia and accuracy_result.paranoia.inferred_items:
            f.write("### Inferred Scope Items (Paranoia Rules)\n\n")
            f.write(f"Total items inferred: {len(accuracy_result.paranoia.inferred_items)}\n\n")

            by_trigger = accuracy_result.paranoia.summary.get("by_trigger", {})
            for trigger, count in sorted(by_trigger.items(), key=lambda x: -x[1])[:5]:
                f.write(f"- **{trigger.replace('_', ' ').title()}**: {count} items\n")

            if accuracy_result.paranoia.summary.get("estimated_value_range"):
                value_range = accuracy_result.paranoia.summary["estimated_value_range"]
                f.write(f"\nEstimated value: ₹{value_range.get('min_inr', 0):,.0f} - ₹{value_range.get('max_inr', 0):,.0f}\n")
            f.write("\n")

        f.write("### Accuracy Outputs\n\n")
        f.write("- `accuracy/triangulation.json` - Multi-method quantity verification\n")
        f.write("- `accuracy/overrides.json` - Cross-sheet corrections applied\n")
        f.write("- `accuracy/paranoia.json` - Estimator paranoia inferences\n")
        f.write("- `accuracy/discrepancies.md` - Items requiring review\n")


def _update_summary_with_scope(
    summary_path: Path,
    scope_result: Dict[str, Any],
) -> None:
    """Update summary.md with scope completeness info."""
    with open(summary_path, "a") as f:
        f.write("\n## Scope Completeness\n\n")
        f.write(f"- **Completeness Score**: {scope_result['completeness_score']:.0f}/100 (Grade: {scope_result['completeness_grade']})\n")
        f.write(f"- **Scope Gaps**: {scope_result['gaps_count']} items require clarification\n")
        f.write(f"- **Checklist Items**: {scope_result['checklist_count']} items for estimator confirmation\n")
        f.write(f"- **Provisional Items**: {scope_result['provisionals_count']} provisional BOQ lines added\n\n")

        if scope_result.get("top_risks"):
            f.write("### Top Scope Risks\n\n")
            for risk in scope_result["top_risks"]:
                f.write(f"1. **{risk['package']} > {risk['subpackage']}**: {risk['status']} - {risk['action']}\n")
            f.write("\n")

        f.write("### Scope Outputs\n\n")
        f.write("- `scope/scope_register.csv` - Complete scope register\n")
        f.write("- `scope/scope_gaps.md` - Detailed gaps analysis\n")
        f.write("- `scope/estimator_checklist.md` - Bid-ready confirmation checklist\n")
        f.write("- `boq/provisional_items.csv` - Provisional BOQ items\n")
        f.write("- `scope/completeness_report.json` - Full completeness report\n")


def _build_boq_entries(
    runner_result: RunnerResult,
    project_graph: ProjectGraph
) -> List[Dict[str, Any]]:
    """Build BOQ entries list for coverage analysis."""
    entries = []

    # Add room areas
    for result in runner_result.extraction_results:
        if result.page_type != "floor_plan":
            continue

        page_label = f"{Path(result.file_path).stem}_p{result.page_number + 1}"

        for room in result.rooms:
            if room.get("area_sqm", 0) > 0:
                entries.append({
                    "category": "Floor Area",
                    "item_code": room.get("room_id", ""),
                    "description": room.get("label", "Room"),
                    "unit": "sqm",
                    "quantity": room.get("area_sqm", 0),
                    "source_page": page_label,
                })

    # Add schedule entries
    for stype, schedule_entries in project_graph.schedules.items():
        for entry in schedule_entries:
            page_label = f"{Path(entry.source_file).stem}_p{entry.source_page + 1}"

            if stype == "door":
                entries.append({
                    "category": "Doors",
                    "item_code": entry.tag,
                    "description": f"Door",
                    "source_page": page_label,
                })
            elif stype == "window":
                entries.append({
                    "category": "Windows",
                    "item_code": entry.tag,
                    "description": f"Window",
                    "source_page": page_label,
                })

    return entries


def _populate_provenance(
    tracker: ProvenanceTracker,
    runner_result: RunnerResult
) -> None:
    """Populate provenance tracker from extraction results."""
    from .provenance import DetectionMethod

    for result in runner_result.extraction_results:
        # Add records from provenance list
        for prov in result.provenance:
            # Map detection method string to enum
            method_str = prov.get("detection_method", "raster")
            try:
                detection_method = DetectionMethod(method_str)
            except ValueError:
                detection_method = DetectionMethod.RASTER

            if prov.get("object_type") == "room":
                tracker.track_room(
                    room_id=prov.get("object_id", ""),
                    file_path=prov.get("source_file", ""),
                    page_number=prov.get("source_page", 0),
                    detection_method=detection_method,
                    confidence=prov.get("confidence", 0.8),
                    label=prov.get("method_details", {}).get("label", ""),
                    area_sqm=prov.get("method_details", {}).get("area_sqm", 0),
                    scale_method=prov.get("method_details", {}).get("scale_method", ""),
                )
            elif prov.get("object_type") == "schedule_entry":
                tracker.track_schedule_entry(
                    entry_type="schedule",
                    tag=prov.get("method_details", {}).get("tag", ""),
                    file_path=prov.get("source_file", ""),
                    page_number=prov.get("source_page", 0),
                    detection_method=DetectionMethod.OCR,
                    confidence=prov.get("confidence", 0.85),
                    properties=prov.get("method_details", {}),
                )


def _update_summary_with_risks(
    summary_path: Path,
    coverage_report,
    sanity_report,
    missing_inputs: List,
) -> None:
    """Update summary.md with risks and recommended actions."""
    # Read existing summary
    with open(summary_path, "r") as f:
        content = f.read()

    # Build risks section
    risks = []

    # From coverage
    if coverage_report.scale.high_conf_pct < 50:
        risks.append(f"**Scale uncertainty**: Only {coverage_report.scale.high_conf_pct:.0f}% of floor plans have high-confidence scale")
    if coverage_report.rooms.label_coverage_pct < 70:
        risks.append(f"**Unlabeled rooms**: {100 - coverage_report.rooms.label_coverage_pct:.0f}% of rooms lack proper labels")
    if not coverage_report.schedules.door_schedule:
        risks.append("**Missing door schedule**: Door counts cannot be verified")
    if not coverage_report.schedules.window_schedule:
        risks.append("**Missing window schedule**: Window counts cannot be verified")

    # From sanity checks
    for check in sanity_report.checks:
        if check.status.value == "FAIL":
            risks.append(f"**{check.name}**: {check.message}")
        elif check.status.value == "WARN" and len(risks) < 5:
            risks.append(f"**{check.name}**: {check.message}")

    # Limit to top 5
    risks = risks[:5]

    # Build recommendations section
    recommendations = []
    for item in missing_inputs[:5]:
        recommendations.append(f"**{item.item}**: {item.suggestion}")

    # Insert after Output Files section
    insert_text = "\n## Top Risks\n\n"
    if risks:
        for risk in risks:
            insert_text += f"- {risk}\n"
    else:
        insert_text += "No significant risks identified.\n"

    insert_text += "\n## Recommended Next Actions\n\n"
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            insert_text += f"{i}. {rec}\n"
    else:
        insert_text += "No immediate actions required.\n"

    insert_text += f"\n## Quality Metrics\n\n"
    insert_text += f"- **Coverage Score**: {coverage_report.overall_score:.0f}/100 (Grade: {coverage_report.grade})\n"
    insert_text += f"- **Sanity Status**: {sanity_report.overall_status.value}\n"
    insert_text += f"- **Checks Passed**: {sanity_report.pass_count}/{sanity_report.pass_count + sanity_report.warn_count + sanity_report.fail_count}\n"

    # Append to end
    content += insert_text

    # Write updated summary
    with open(summary_path, "w") as f:
        f.write(content)


def _update_summary_with_doubt(
    summary_path: Path,
    doubt_result: Dict[str, Any],
) -> None:
    """Update summary.md with doubt engine info."""
    with open(summary_path, "a") as f:
        f.write("\n## Drawing Set Completeness\n\n")
        f.write(f"- **Completeness Score**: {doubt_result['completeness_score']:.0f}/100 ")
        f.write(f"(Grade: {doubt_result['completeness_grade']})\n")
        f.write(f"- **Critical Sheets Missing**: {doubt_result['critical_missing']}\n")
        f.write(f"- **Important Sheets Missing**: {doubt_result['important_missing']}\n")
        f.write(f"- **Doubt RFIs**: {doubt_result['doubt_rfis']}\n\n")

        if doubt_result['critical_missing'] > 0:
            f.write("### ⚠️ Critical Missing Sheets\n\n")
            f.write("The following sheet types are essential but missing:\n\n")
            # Would need to pass more data to show specific sheets
            f.write("See `doubt_report.md` for full details.\n\n")

        f.write("### Doubt Outputs\n\n")
        f.write("- `doubt_report.md` - Missing sheet analysis\n")
        f.write("- `doubt_report.json` - Doubt report data\n")
        f.write("- `doubt_rfis.json` - RFIs from missing sheets\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process multi-page architect drawing sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder of PDFs
  python -m src.project.process --input ./data/project --output ./out --project_id MyProject

  # Process only floor plans
  python -m src.project.process --input project.pdf --output ./out --only floor_plan

  # Enable structural extraction
  python -m src.project.process --input ./data --output ./out --enable_structural

  # Resume interrupted processing
  python -m src.project.process --input ./data --output ./out --resume
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input path (folder or PDF)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./out",
        help="Output directory"
    )
    parser.add_argument(
        "--project_id", "-p",
        help="Project identifier (default: input name)"
    )
    parser.add_argument(
        "--enable_structural",
        action="store_true",
        help="Enable structural plan extraction"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["floor_plan", "schedule_table", "structural_plan"],
        help="Process only specific page types"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from cached results (default: true)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume, reprocess everything"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=0,
        help="Max parallel workers (0 = auto, default: cpu/2)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for extraction (default: 300)"
    )
    parser.add_argument(
        "--thumb-dpi",
        type=int,
        default=100,
        help="DPI for thumbnails (default: 100)"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Default drawing scale (default: 100)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Handle resume flag
    resume = args.resume and not args.no_resume

    # Handle only types
    only_types = set(args.only) if args.only else None

    try:
        result = process_project_set(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            project_id=args.project_id,
            enable_structural=args.enable_structural,
            only_types=only_types,
            resume=resume,
            max_workers=args.workers,
            dpi=args.dpi,
            thumb_dpi=args.thumb_dpi,
            default_scale=args.scale,
            save_debug_images=args.debug,
        )

        print(f"\n{'='*60}")
        print(f"RESULT: {result['project_id']}")
        print(f"{'='*60}")
        print(f"Total pages: {result['total_pages']}")
        print(f"Processed: {result['processed_pages']}")
        print(f"Rooms: {result['total_rooms']}")
        print(f"Area: {result['total_area_sqm']:.1f} sqm")
        print(f"Time: {result['total_time_sec']:.1f}s")
        print(f"\nOutputs in: {result['output_dir']}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
