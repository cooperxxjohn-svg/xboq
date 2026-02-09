"""
Project-Level Batch Processing Pipeline.

Scalable pipeline for processing large multi-page PDF drawing sets:
- Indexing with caching (fast, low-DPI)
- Page classification and routing
- Targeted extraction (high-DPI only where needed)
- Project-level joining (schedules + drawings)
- Unified BOQ export
- Coverage analysis and missing inputs
- Sanity checks and validation
- Provenance/audit trail tracking
"""

from .indexer import ProjectIndexer, PageIndex, ProjectIndex
from .router import PageRouter, PageType, RoutingResult
from .runner import ProjectRunner, RunnerConfig
from .joiner import ProjectJoiner, ProjectGraph
from .exporter import ProjectExporter
from .coverage import CoverageAnalyzer, CoverageReport, analyze_coverage
from .sanity import SanityChecker, SanityReport, run_sanity_checks
from .provenance import ProvenanceTracker, ProvenanceStore, create_tracker
from .process import process_project_set

__all__ = [
    "ProjectIndexer",
    "PageIndex",
    "ProjectIndex",
    "PageRouter",
    "PageType",
    "RoutingResult",
    "ProjectRunner",
    "RunnerConfig",
    "ProjectJoiner",
    "ProjectGraph",
    "ProjectExporter",
    "CoverageAnalyzer",
    "CoverageReport",
    "analyze_coverage",
    "SanityChecker",
    "SanityReport",
    "run_sanity_checks",
    "ProvenanceTracker",
    "ProvenanceStore",
    "create_tracker",
    "process_project_set",
]
