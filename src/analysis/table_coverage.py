"""
src/analysis/table_coverage.py

Tracks and reports table extraction coverage during a pipeline run.

Usage:
    from src.analysis.table_coverage import TableCoverageTracker
    tracker = TableCoverageTracker()
    tracker.record_attempt(page=5, method="pdfplumber", success=True, n_rows=45)
    stats = tracker.summary()
    payload["table_extraction_coverage"] = stats
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TableAttempt:
    page: int
    method: str          # "pdfplumber" | "camelot_lattice" | "camelot_stream" | "ocr_rows"
    success: bool
    n_rows: int = 0
    n_cols: int = 0
    confidence: float = 1.0
    doc_type: str = ""
    error_msg: str = ""


@dataclass
class TableCoverageSummary:
    total_attempts: int
    successful: int
    failed: int
    total_rows_extracted: int
    coverage_pct: float          # successful / total_attempts * 100
    method_breakdown: Dict[str, int] = field(default_factory=dict)  # method → success count
    low_confidence_pages: List[int] = field(default_factory=list)   # pages with conf < 0.5
    boq_pages_attempted: int = 0
    boq_pages_parsed: int = 0
    boq_coverage_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "failed": self.failed,
            "total_rows_extracted": self.total_rows_extracted,
            "coverage_pct": round(self.coverage_pct, 1),
            "method_breakdown": self.method_breakdown,
            "low_confidence_pages": self.low_confidence_pages,
            "boq_pages_attempted": self.boq_pages_attempted,
            "boq_pages_parsed": self.boq_pages_parsed,
            "boq_coverage_pct": round(self.boq_coverage_pct, 1),
        }


class TableCoverageTracker:
    """
    Accumulates table extraction attempts and results during a pipeline run.
    Thread-safe for use in background threads.
    """

    def __init__(self):
        self._attempts: List[TableAttempt] = []

    def record_attempt(
        self,
        page: int,
        method: str,
        success: bool,
        n_rows: int = 0,
        n_cols: int = 0,
        confidence: float = 1.0,
        doc_type: str = "",
        error_msg: str = "",
    ) -> None:
        """Record a single table extraction attempt."""
        self._attempts.append(TableAttempt(
            page=page,
            method=method,
            success=success,
            n_rows=n_rows,
            n_cols=n_cols,
            confidence=confidence,
            doc_type=doc_type,
            error_msg=error_msg,
        ))

    def summary(self) -> TableCoverageSummary:
        """Compute coverage summary from all recorded attempts."""
        if not self._attempts:
            return TableCoverageSummary(
                total_attempts=0, successful=0, failed=0,
                total_rows_extracted=0, coverage_pct=0.0,
            )

        successful = [a for a in self._attempts if a.success]
        failed = [a for a in self._attempts if not a.success]
        boq_attempts = [a for a in self._attempts if a.doc_type == "boq"]
        boq_parsed = [a for a in boq_attempts if a.success]

        method_breakdown: Dict[str, int] = {}
        for a in successful:
            method_breakdown[a.method] = method_breakdown.get(a.method, 0) + 1

        low_conf = [a.page for a in successful if a.confidence < 0.5]

        coverage = (len(successful) / max(len(self._attempts), 1)) * 100
        boq_cov = (len(boq_parsed) / max(len(boq_attempts), 1)) * 100 if boq_attempts else 0.0

        return TableCoverageSummary(
            total_attempts=len(self._attempts),
            successful=len(successful),
            failed=len(failed),
            total_rows_extracted=sum(a.n_rows for a in successful),
            coverage_pct=coverage,
            method_breakdown=method_breakdown,
            low_confidence_pages=low_conf,
            boq_pages_attempted=len(boq_attempts),
            boq_pages_parsed=len(boq_parsed),
            boq_coverage_pct=boq_cov,
        )

    def reset(self) -> None:
        self._attempts.clear()

    @property
    def attempt_count(self) -> int:
        return len(self._attempts)
