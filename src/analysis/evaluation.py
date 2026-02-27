"""
Evaluation and Debug Logging for Deep Analysis

Provides:
1. EvaluationLog generation from analysis results
2. Debug output for development
3. Metrics collection for continuous improvement
4. JSON export for analysis tracking
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.analysis_models import (
    PlanSetGraph, Blocker, RFIItem, TradeCoverage, BOQSkeletonItem,
    DeepAnalysisResult, EvaluationLog, ReadinessScore,
    Severity, Trade, SheetType, Discipline,
)


# =============================================================================
# EVALUATION LOG BUILDER
# =============================================================================

class EvaluationLogBuilder:
    """
    Builds EvaluationLog from analysis results.

    Captures metrics for tracking analysis quality over time.
    """

    def build(self, result: DeepAnalysisResult) -> EvaluationLog:
        """
        Build evaluation log from analysis result.

        Args:
            result: DeepAnalysisResult to evaluate

        Returns:
            EvaluationLog with metrics
        """
        log = EvaluationLog(
            project_id=result.project_id,
            created_at=datetime.now(),
        )

        # Detection metrics from plan graph
        if result.plan_graph:
            log.detected_sheets_count = result.plan_graph.total_pages
            log.schedules_detected_count = sum([
                1 if result.plan_graph.has_door_schedule else 0,
                1 if result.plan_graph.has_window_schedule else 0,
                1 if result.plan_graph.has_finish_schedule else 0,
            ])
            log.scale_missing_pages_count = result.plan_graph.pages_without_scale

        # Blocker counts by type
        blockers_by_type: Dict[str, int] = defaultdict(int)
        for blocker in result.blockers:
            issue_type = blocker.issue_type or "unknown"
            blockers_by_type[issue_type] += 1
        log.blockers_by_type = dict(blockers_by_type)

        # RFIs by trade
        rfis_by_trade: Dict[str, int] = defaultdict(int)
        for rfi in result.rfis:
            rfis_by_trade[rfi.trade.value] += 1
        log.rfis_by_trade = dict(rfis_by_trade)

        # Coverage by trade
        coverage_by_trade: Dict[str, float] = {}
        for tc in result.trade_coverage:
            coverage_by_trade[tc.trade.value] = tc.coverage_pct
        log.coverage_by_trade = coverage_by_trade

        # Scores
        if result.readiness_score:
            log.final_score = result.readiness_score.total_score
            log.component_scores = {
                "completeness": result.readiness_score.completeness_score,
                "measurement": result.readiness_score.measurement_score,
                "coverage": result.readiness_score.coverage_score,
                "blocker": result.readiness_score.blocker_score,
            }

        return log


# =============================================================================
# DEBUG OUTPUT
# =============================================================================

class DebugLogger:
    """
    Generates debug output for development and troubleshooting.
    """

    def __init__(self, output_dir: Optional[Path] = None, verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        self.logs: List[str] = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        self.logs.append(formatted)
        if self.verbose:
            print(formatted)

    def log_plan_graph(self, graph: PlanSetGraph):
        """Log plan graph summary."""
        self.log("=== Plan Set Graph ===")
        self.log(f"Project: {graph.project_id}")
        self.log(f"Total Pages: {graph.total_pages}")
        self.log(f"Disciplines: {', '.join(graph.disciplines_found) or 'None detected'}")
        self.log(f"Sheet Types: {json.dumps(graph.sheet_types_found)}")
        self.log(f"Door Tags: {len(graph.all_door_tags)} unique")
        self.log(f"Window Tags: {len(graph.all_window_tags)} unique")
        self.log(f"Room Names: {len(graph.all_room_names)} types")
        self.log(f"Scale: {graph.pages_with_scale} with / {graph.pages_without_scale} without")
        self.log(f"Schedules: door={graph.has_door_schedule}, window={graph.has_window_schedule}, finish={graph.has_finish_schedule}")

    def log_blockers(self, blockers: List[Blocker]):
        """Log blockers summary."""
        self.log(f"=== Blockers ({len(blockers)}) ===")

        # Group by severity
        by_severity: Dict[str, List[Blocker]] = defaultdict(list)
        for b in blockers:
            by_severity[b.severity.value].append(b)

        for severity in ["critical", "high", "medium", "low"]:
            items = by_severity.get(severity, [])
            if items:
                self.log(f"{severity.upper()}: {len(items)}")
                for b in items[:3]:  # Show first 3
                    self.log(f"  - {b.id}: {b.title}")
                    self.log(f"    Evidence: {b.evidence.summary()}")

    def log_trade_coverage(self, coverage: List[TradeCoverage]):
        """Log trade coverage."""
        self.log(f"=== Trade Coverage ({len(coverage)}) ===")
        for tc in coverage:
            status = "OK" if tc.coverage_pct >= 70 else "NEEDS ATTENTION" if tc.coverage_pct >= 40 else "BLOCKED"
            self.log(f"{tc.trade.value}: {tc.coverage_pct:.0f}% [{status}]")
            if tc.missing_dependencies:
                self.log(f"  Missing: {', '.join(tc.missing_dependencies[:3])}")

    def log_score(self, score: ReadinessScore):
        """Log readiness score breakdown."""
        self.log("=== Readiness Score ===")
        self.log(f"Total: {score.total_score}/100 [{score.status}]")
        self.log(f"  Completeness: {score.completeness_score} ({score.score_breakdown.get('completeness', '')})")
        self.log(f"  Measurement:  {score.measurement_score} ({score.score_breakdown.get('measurement', '')})")
        self.log(f"  Coverage:     {score.coverage_score} ({score.score_breakdown.get('coverage', '')})")
        self.log(f"  Blocker:      {score.blocker_score} ({score.score_breakdown.get('blocker', '')})")

    def log_result(self, result: DeepAnalysisResult):
        """Log complete analysis result."""
        self.log("=" * 50)
        self.log(f"DEEP ANALYSIS RESULT: {result.project_id}")
        self.log("=" * 50)

        if result.plan_graph:
            self.log_plan_graph(result.plan_graph)

        self.log("")
        self.log_blockers(result.blockers)

        self.log("")
        self.log_trade_coverage(result.trade_coverage)

        if result.readiness_score:
            self.log("")
            self.log_score(result.readiness_score)

        self.log("=" * 50)

    def save(self, filename: str = "debug_log.txt"):
        """Save logs to file."""
        if self.output_dir:
            path = self.output_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write("\n".join(self.logs))
            return path
        return None


# =============================================================================
# JSON EXPORT
# =============================================================================

def export_analysis_to_json(
    result: DeepAnalysisResult,
    output_path: Path
) -> Path:
    """
    Export complete analysis to JSON.

    Args:
        result: DeepAnalysisResult to export
        output_path: Path to output file

    Returns:
        Path to written file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    return output_path


def export_evaluation_log(
    log: EvaluationLog,
    output_path: Path
) -> Path:
    """
    Export evaluation log to JSON.

    Args:
        log: EvaluationLog to export
        output_path: Path to output file

    Returns:
        Path to written file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(log.to_dict(), f, indent=2, default=str)

    return output_path


# =============================================================================
# METRICS COLLECTION
# =============================================================================

class MetricsCollector:
    """
    Collects metrics across multiple analyses for quality tracking.
    """

    def __init__(self, metrics_dir: Optional[Path] = None):
        self.metrics_dir = metrics_dir
        self.metrics: List[Dict[str, Any]] = []

    def add(self, log: EvaluationLog):
        """Add an evaluation log to metrics."""
        self.metrics.append({
            "project_id": log.project_id,
            "timestamp": log.created_at.isoformat(),
            "sheets_count": log.detected_sheets_count,
            "schedules_count": log.schedules_detected_count,
            "scale_missing": log.scale_missing_pages_count,
            "blocker_count": sum(log.blockers_by_type.values()),
            "rfi_count": sum(log.rfis_by_trade.values()),
            "final_score": log.final_score,
        })

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics:
            return {}

        scores = [m["final_score"] for m in self.metrics if m["final_score"]]
        blockers = [m["blocker_count"] for m in self.metrics]

        return {
            "total_analyses": len(self.metrics),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_blockers": sum(blockers) / len(blockers) if blockers else 0,
        }

    def save(self, filename: str = "metrics_history.json"):
        """Save metrics to file."""
        if self.metrics_dir:
            path = self.metrics_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump({
                    "metrics": self.metrics,
                    "summary": self.summary(),
                }, f, indent=2, default=str)
            return path
        return None


# =============================================================================
# ENTRY POINTS
# =============================================================================

def create_evaluation_log(result: DeepAnalysisResult) -> EvaluationLog:
    """
    Create evaluation log from analysis result.

    Args:
        result: DeepAnalysisResult to evaluate

    Returns:
        EvaluationLog with metrics
    """
    builder = EvaluationLogBuilder()
    return builder.build(result)


def debug_analysis(
    result: DeepAnalysisResult,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Optional[Path]:
    """
    Generate debug output for analysis result.

    Args:
        result: DeepAnalysisResult to debug
        output_dir: Optional output directory
        verbose: Print to console

    Returns:
        Path to debug log file if output_dir provided
    """
    logger = DebugLogger(output_dir, verbose)
    logger.log_result(result)
    return logger.save() if output_dir else None


def save_all_outputs(
    result: DeepAnalysisResult,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Save all analysis outputs to directory.

    Creates:
    - deep_analysis.json: Complete analysis result
    - evaluation_log.json: Metrics for tracking
    - debug_log.txt: Human-readable debug output
    - plan_graph.json: Plan set structure (if available)

    Args:
        result: DeepAnalysisResult to save
        output_dir: Output directory

    Returns:
        Dict of output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Deep analysis
    outputs["deep_analysis"] = export_analysis_to_json(
        result,
        output_dir / "deep_analysis.json"
    )

    # Evaluation log
    log = create_evaluation_log(result)
    outputs["evaluation_log"] = export_evaluation_log(
        log,
        output_dir / "evaluation_log.json"
    )

    # Debug log
    debug_path = debug_analysis(result, output_dir, verbose=False)
    if debug_path:
        outputs["debug_log"] = debug_path

    # Plan graph (separate file for easy access)
    if result.plan_graph:
        from src.analysis.plan_graph import save_plan_graph
        outputs["plan_graph"] = save_plan_graph(result.plan_graph, output_dir)

    return outputs


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with a mock result
    from src.analysis.plan_graph import build_plan_graph
    from src.analysis.dependency_reasoner import reason_dependencies
    from src.analysis.llm_enrichment import calculate_readiness_score

    # Build sample data
    sample_texts = [
        "SHEET A-101\nGROUND FLOOR PLAN\nSCALE 1:100\nD1 D2 D3\nW1 W2\nLIVING ROOM\nKITCHEN",
        "SHEET A-102\nFIRST FLOOR PLAN\nD4 D5\nW3 W4\nBEDROOM 1\nBATHROOM",
    ]

    graph = build_plan_graph("test_project", sample_texts)
    blockers, rfis, coverage, skeleton = reason_dependencies(graph)
    score = calculate_readiness_score(graph, blockers, trade_coverage=coverage)

    result = DeepAnalysisResult(
        project_id="test_project",
        plan_graph=graph,
        blockers=blockers,
        rfis=rfis,
        trade_coverage=coverage,
        boq_skeleton=skeleton,
        readiness_score=score,
    )

    # Debug output
    print("\n" + "=" * 60)
    debug_analysis(result, verbose=True)

    # Evaluation log
    log = create_evaluation_log(result)
    print("\n=== Evaluation Log ===")
    print(json.dumps(log.to_dict(), indent=2, default=str))
