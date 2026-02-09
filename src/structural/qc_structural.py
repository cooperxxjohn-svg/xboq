"""
Structural QC and Explainability Module
Provides quality checks, confidence scores, and full traceability
for structural takeoff calculations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from datetime import datetime

from .detect_columns import DetectedColumn, ColumnDetectionResult
from .detect_beams import DetectedBeam, BeamDetectionResult
from .detect_footings import DetectedFooting, FootingDetectionResult
from .quantity_engine import QuantityResult, ElementQuantity

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Warning/error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QCCode(Enum):
    """Standardized QC codes for issues."""
    # Detection issues
    LOW_CONFIDENCE = "D001"
    NO_LABEL = "D002"
    UNUSUAL_SIZE = "D003"
    DUPLICATE_LABEL = "D004"
    ORPHAN_BEAM = "D005"  # Beam not connected to columns
    MISSING_FOOTING = "D006"  # Column without footing

    # Schedule issues
    SCHEDULE_MISMATCH = "S001"
    MISSING_SCHEDULE = "S002"
    INCOMPLETE_SCHEDULE = "S003"

    # Quantity issues
    ASSUMPTION_USED = "Q001"
    HIGH_STEEL_RATIO = "Q002"
    LOW_STEEL_RATIO = "Q003"
    UNUSUAL_VOLUME = "Q004"

    # Structural issues
    UNBALANCED_COLUMNS = "R001"
    MISSING_BEAM = "R002"
    ISOLATED_COLUMN = "R003"

    # Input issues
    SCALE_UNCERTAIN = "I001"
    LOW_RESOLUTION = "I002"
    MISSING_PAGE = "I003"


@dataclass
class QCIssue:
    """A single QC issue."""
    code: QCCode
    severity: Severity
    message: str
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code.value,
            'severity': self.severity.value,
            'message': self.message,
            'element_id': self.element_id,
            'element_type': self.element_type,
            'details': self.details,
            'suggestion': self.suggestion
        }


@dataclass
class AssumptionLog:
    """Log of an assumption used in calculations."""
    category: str  # "size", "count", "height", "steel", etc.
    element_type: str
    element_id: Optional[str]
    assumed_value: Any
    source: str  # Where the default came from
    confidence_impact: float  # How much this affects confidence
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category,
            'element_type': self.element_type,
            'element_id': self.element_id,
            'assumed_value': str(self.assumed_value),
            'source': self.source,
            'confidence_impact': self.confidence_impact,
            'alternatives': self.alternatives
        }


@dataclass
class TraceEntry:
    """Traceability entry for a calculation."""
    element_id: str
    calculation: str
    inputs: Dict[str, Any]
    formula: str
    result: Any
    unit: str
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element_id': self.element_id,
            'calculation': self.calculation,
            'inputs': self.inputs,
            'formula': self.formula,
            'result': self.result,
            'unit': self.unit,
            'assumptions': self.assumptions
        }


@dataclass
class StructuralQCReport:
    """Complete QC report for structural takeoff."""
    # Overall scores
    overall_confidence: float = 0.0
    detection_confidence: float = 0.0
    quantity_confidence: float = 0.0

    # Issues
    issues: List[QCIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Assumptions
    assumptions: List[AssumptionLog] = field(default_factory=list)
    assumption_count: int = 0

    # Traceability
    trace_log: List[TraceEntry] = field(default_factory=list)

    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    mode: str = "assumption"  # "structural" or "assumption"
    timestamp: str = ""
    input_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'confidence': {
                'overall': round(self.overall_confidence, 2),
                'detection': round(self.detection_confidence, 2),
                'quantity': round(self.quantity_confidence, 2)
            },
            'issues': {
                'total': len(self.issues),
                'errors': self.error_count,
                'warnings': self.warning_count,
                'info': self.info_count,
                'details': [i.to_dict() for i in self.issues]
            },
            'assumptions': {
                'count': self.assumption_count,
                'details': [a.to_dict() for a in self.assumptions]
            },
            'traceability': [t.to_dict() for t in self.trace_log],
            'statistics': self.statistics,
            'metadata': {
                'mode': self.mode,
                'timestamp': self.timestamp,
                'input_files': self.input_files
            }
        }


class StructuralQC:
    """
    Quality Control and Explainability engine for structural takeoff.
    """

    def __init__(self):
        # Thresholds
        self.low_confidence_threshold = 0.5
        self.min_steel_ratio = {  # kg/m³
            'column': 100,
            'beam': 80,
            'footing': 50,
            'slab': 60
        }
        self.max_steel_ratio = {
            'column': 350,
            'beam': 250,
            'footing': 180,
            'slab': 150
        }

    def generate_report(
        self,
        column_result: ColumnDetectionResult = None,
        beam_result: BeamDetectionResult = None,
        footing_result: FootingDetectionResult = None,
        quantity_result: QuantityResult = None,
        mode: str = "assumption",
        input_files: List[str] = None
    ) -> StructuralQCReport:
        """
        Generate comprehensive QC report.

        Args:
            column_result: Column detection results
            beam_result: Beam detection results
            footing_result: Footing detection results
            quantity_result: Quantity computation results
            mode: "structural" or "assumption"
            input_files: List of input file paths

        Returns:
            StructuralQCReport
        """
        report = StructuralQCReport(
            mode=mode,
            timestamp=datetime.now().isoformat(),
            input_files=input_files or []
        )

        # Check detections
        detection_scores = []

        if column_result:
            col_issues, col_score = self._check_columns(column_result)
            report.issues.extend(col_issues)
            detection_scores.append(col_score)

        if beam_result:
            beam_issues, beam_score = self._check_beams(beam_result, column_result)
            report.issues.extend(beam_issues)
            detection_scores.append(beam_score)

        if footing_result:
            ftg_issues, ftg_score = self._check_footings(footing_result, column_result)
            report.issues.extend(ftg_issues)
            detection_scores.append(ftg_score)

        # Check quantities
        quantity_score = 1.0
        if quantity_result:
            qty_issues, qty_score, assumptions, traces = self._check_quantities(quantity_result)
            report.issues.extend(qty_issues)
            report.assumptions.extend(assumptions)
            report.trace_log.extend(traces)
            quantity_score = qty_score

        # Add mode-specific warning
        if mode == "assumption":
            report.issues.append(QCIssue(
                code=QCCode.ASSUMPTION_USED,
                severity=Severity.WARNING,
                message="Quantities computed using ASSUMPTION MODE with default values",
                suggestion="Use structural drawings for accurate takeoff"
            ))

        # Count issues by severity
        for issue in report.issues:
            if issue.severity == Severity.ERROR:
                report.error_count += 1
            elif issue.severity == Severity.WARNING:
                report.warning_count += 1
            else:
                report.info_count += 1

        # Calculate confidence scores
        if detection_scores:
            report.detection_confidence = sum(detection_scores) / len(detection_scores)
        else:
            report.detection_confidence = 0.5 if mode == "assumption" else 0.0

        report.quantity_confidence = quantity_score
        report.assumption_count = len(report.assumptions)

        # Penalize for assumptions and errors
        base_confidence = (report.detection_confidence + report.quantity_confidence) / 2
        assumption_penalty = min(0.3, report.assumption_count * 0.02)
        error_penalty = report.error_count * 0.1

        report.overall_confidence = max(0.1, base_confidence - assumption_penalty - error_penalty)

        # Build statistics
        report.statistics = self._build_statistics(
            column_result, beam_result, footing_result, quantity_result
        )

        return report

    def _check_columns(
        self,
        result: ColumnDetectionResult
    ) -> Tuple[List[QCIssue], float]:
        """Check column detection quality."""
        issues = []
        confidences = []

        # Check for duplicates
        label_counts = {}
        for col in result.columns:
            label = col.label
            label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            if count > 1 and not label.startswith('C'):
                issues.append(QCIssue(
                    code=QCCode.DUPLICATE_LABEL,
                    severity=Severity.WARNING,
                    message=f"Duplicate column label '{label}' found {count} times",
                    element_type="column",
                    suggestion="Verify column labels in drawing"
                ))

        for col in result.columns:
            confidences.append(col.confidence)

            # Low confidence
            if col.confidence < self.low_confidence_threshold:
                issues.append(QCIssue(
                    code=QCCode.LOW_CONFIDENCE,
                    severity=Severity.WARNING,
                    message=f"Low detection confidence for column {col.label}",
                    element_id=col.column_id,
                    element_type="column",
                    details={'confidence': col.confidence}
                ))

            # Missing label
            if not col.label or col.label.startswith('C') and col.source == "detection":
                issues.append(QCIssue(
                    code=QCCode.NO_LABEL,
                    severity=Severity.INFO,
                    message=f"Column {col.column_id} has auto-generated label",
                    element_id=col.column_id,
                    element_type="column"
                ))

            # Unusual size
            if col.size_mm:
                w, d = col.size_mm
                if w < 150 or d < 150:
                    issues.append(QCIssue(
                        code=QCCode.UNUSUAL_SIZE,
                        severity=Severity.WARNING,
                        message=f"Column {col.label} has very small size ({w}x{d}mm)",
                        element_id=col.column_id,
                        element_type="column"
                    ))
                elif w > 800 or d > 800:
                    issues.append(QCIssue(
                        code=QCCode.UNUSUAL_SIZE,
                        severity=Severity.INFO,
                        message=f"Column {col.label} has very large size ({w}x{d}mm)",
                        element_id=col.column_id,
                        element_type="column"
                    ))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        return issues, avg_confidence

    def _check_beams(
        self,
        result: BeamDetectionResult,
        column_result: ColumnDetectionResult = None
    ) -> Tuple[List[QCIssue], float]:
        """Check beam detection quality."""
        issues = []
        confidences = []

        for beam in result.beams:
            confidences.append(beam.confidence)

            # Orphan beam (not connected to columns)
            if not beam.from_column and not beam.to_column:
                issues.append(QCIssue(
                    code=QCCode.ORPHAN_BEAM,
                    severity=Severity.WARNING,
                    message=f"Beam {beam.label} not connected to any columns",
                    element_id=beam.beam_id,
                    element_type="beam"
                ))

            # Partially connected
            elif not beam.from_column or not beam.to_column:
                issues.append(QCIssue(
                    code=QCCode.ORPHAN_BEAM,
                    severity=Severity.INFO,
                    message=f"Beam {beam.label} connected to only one column",
                    element_id=beam.beam_id,
                    element_type="beam"
                ))

            # Low confidence
            if beam.confidence < self.low_confidence_threshold:
                issues.append(QCIssue(
                    code=QCCode.LOW_CONFIDENCE,
                    severity=Severity.WARNING,
                    message=f"Low detection confidence for beam {beam.label}",
                    element_id=beam.beam_id,
                    element_type="beam"
                ))

        # Check if all columns have beams
        if column_result and result.graph:
            for col in column_result.columns:
                connected = result.graph.get_connected_columns(col.column_id)
                if not connected:
                    issues.append(QCIssue(
                        code=QCCode.ISOLATED_COLUMN,
                        severity=Severity.WARNING,
                        message=f"Column {col.label} has no connected beams",
                        element_id=col.column_id,
                        element_type="column",
                        suggestion="Verify beam layout in drawing"
                    ))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        return issues, avg_confidence

    def _check_footings(
        self,
        result: FootingDetectionResult,
        column_result: ColumnDetectionResult = None
    ) -> Tuple[List[QCIssue], float]:
        """Check footing detection quality."""
        issues = []
        confidences = []

        for ftg in result.footings:
            confidences.append(ftg.confidence)

            # Low confidence
            if ftg.confidence < self.low_confidence_threshold:
                issues.append(QCIssue(
                    code=QCCode.LOW_CONFIDENCE,
                    severity=Severity.WARNING,
                    message=f"Low detection confidence for footing {ftg.label}",
                    element_id=ftg.footing_id,
                    element_type="footing"
                ))

        # Check columns without footings
        if column_result:
            columns_with_footings = set()
            for ftg in result.footings:
                columns_with_footings.update(ftg.associated_columns)

            for col in column_result.columns:
                if col.column_id not in columns_with_footings:
                    issues.append(QCIssue(
                        code=QCCode.MISSING_FOOTING,
                        severity=Severity.WARNING,
                        message=f"No footing detected for column {col.label}",
                        element_id=col.column_id,
                        element_type="column",
                        suggestion="Check foundation plan or verify column is on existing footing"
                    ))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        return issues, avg_confidence

    def _check_quantities(
        self,
        result: QuantityResult
    ) -> Tuple[List[QCIssue], float, List[AssumptionLog], List[TraceEntry]]:
        """Check quantity calculations."""
        issues = []
        assumptions = []
        traces = []
        confidence = 1.0

        for elem in result.elements:
            # Log assumptions
            for assumption in elem.assumptions:
                assumptions.append(AssumptionLog(
                    category="calculation",
                    element_type=elem.element_type,
                    element_id=elem.element_id,
                    assumed_value=assumption,
                    source="assumptions.yaml",
                    confidence_impact=0.05
                ))

            # Check steel ratios
            if elem.concrete_volume_m3 > 0 and elem.steel_total_kg > 0:
                steel_ratio = elem.steel_total_kg / elem.concrete_volume_m3

                min_ratio = self.min_steel_ratio.get(elem.element_type, 50)
                max_ratio = self.max_steel_ratio.get(elem.element_type, 300)

                if steel_ratio < min_ratio:
                    issues.append(QCIssue(
                        code=QCCode.LOW_STEEL_RATIO,
                        severity=Severity.WARNING,
                        message=f"Low steel ratio for {elem.label}: {steel_ratio:.0f} kg/m³",
                        element_id=elem.element_id,
                        element_type=elem.element_type,
                        details={'ratio': steel_ratio, 'min': min_ratio}
                    ))
                    confidence -= 0.05

                elif steel_ratio > max_ratio:
                    issues.append(QCIssue(
                        code=QCCode.HIGH_STEEL_RATIO,
                        severity=Severity.WARNING,
                        message=f"High steel ratio for {elem.label}: {steel_ratio:.0f} kg/m³",
                        element_id=elem.element_id,
                        element_type=elem.element_type,
                        details={'ratio': steel_ratio, 'max': max_ratio}
                    ))
                    confidence -= 0.05

            # Create trace entry
            traces.append(TraceEntry(
                element_id=elem.element_id,
                calculation="concrete_volume",
                inputs={
                    'width_mm': elem.width,
                    'depth_mm': elem.depth,
                    'length_mm': elem.length
                },
                formula="(width/1000) × (depth/1000) × (length/1000)",
                result=elem.concrete_volume_m3,
                unit="m³",
                assumptions=elem.assumptions
            ))

            if elem.size_source == "assumption":
                confidence -= 0.02

        # Mode penalty
        if result.mode == "assumption":
            confidence -= 0.2

        return issues, max(0.1, confidence), assumptions, traces

    def _build_statistics(
        self,
        column_result: ColumnDetectionResult,
        beam_result: BeamDetectionResult,
        footing_result: FootingDetectionResult,
        quantity_result: QuantityResult
    ) -> Dict[str, Any]:
        """Build summary statistics."""
        stats = {}

        if column_result:
            stats['columns'] = {
                'count': len(column_result.columns),
                'unique_labels': len(column_result.unique_labels),
                'avg_confidence': sum(c.confidence for c in column_result.columns) /
                                 len(column_result.columns) if column_result.columns else 0
            }

        if beam_result:
            connected = len([b for b in beam_result.beams if b.from_column and b.to_column])
            stats['beams'] = {
                'count': len(beam_result.beams),
                'fully_connected': connected,
                'unique_labels': len(beam_result.unique_labels)
            }

        if footing_result:
            stats['footings'] = {
                'count': len(footing_result.footings),
                'unique_labels': len(footing_result.unique_labels),
                'by_type': {}
            }
            for ftg in footing_result.footings:
                ftype = ftg.footing_type
                stats['footings']['by_type'][ftype] = \
                    stats['footings']['by_type'].get(ftype, 0) + 1

        if quantity_result:
            stats['quantities'] = quantity_result.summary.to_dict()
            stats['mode'] = quantity_result.mode

        return stats


def generate_qc_report(
    quantity_result: QuantityResult,
    mode: str = "assumption"
) -> StructuralQCReport:
    """Convenience function to generate QC report."""
    qc = StructuralQC()
    return qc.generate_report(quantity_result=quantity_result, mode=mode)


if __name__ == "__main__":
    from .quantity_engine import QuantityEngine

    logging.basicConfig(level=logging.INFO)

    # Generate test quantities
    engine = QuantityEngine()
    qty_result = engine.compute_assumption(
        total_area_sqm=100,
        floors=4
    )

    # Generate QC report
    qc = StructuralQC()
    report = qc.generate_report(
        quantity_result=qty_result,
        mode="assumption"
    )

    print("\n=== QC REPORT ===\n")
    print(f"Overall Confidence: {report.overall_confidence:.0%}")
    print(f"Detection Confidence: {report.detection_confidence:.0%}")
    print(f"Quantity Confidence: {report.quantity_confidence:.0%}")

    print(f"\nIssues: {len(report.issues)}")
    print(f"  Errors: {report.error_count}")
    print(f"  Warnings: {report.warning_count}")
    print(f"  Info: {report.info_count}")

    print(f"\nAssumptions used: {report.assumption_count}")

    print("\nTop Issues:")
    for issue in report.issues[:5]:
        symbol = "❌" if issue.severity == Severity.ERROR else "⚠️" if issue.severity == Severity.WARNING else "ℹ️"
        print(f"  {symbol} [{issue.code.value}] {issue.message}")
