"""
Adapter Layer for XBOQ Pipeline

Maps the run_full_project.py interfaces to the actual module implementations.
This layer exists to decouple the runner from internal module structures.
"""

from .index_adapter import ProjectIndexer
from .route_adapter import PageRouter
from .extract_adapter import process_project_drawings
from .join_adapter import ProjectJoiner
from .scope_adapter import ScopeRegister, CompletenessChecker, run_scope_analysis
from .estimator_math_adapter import run_estimator_math
from .rfi_adapter import RFIGenerator, run_rfi_generation
from .bid_gate_adapter import run_bid_gate
from .bid_book_adapter import run_bidbook_export
from .packages_adapter import run_package_splitter

__all__ = [
    "ProjectIndexer",
    "PageRouter",
    "process_project_drawings",
    "ProjectJoiner",
    "ScopeRegister",
    "CompletenessChecker",
    "run_scope_analysis",
    "run_estimator_math",
    "RFIGenerator",
    "run_rfi_generation",
    "run_bid_gate",
    "run_bidbook_export",
    "run_package_splitter",
]
