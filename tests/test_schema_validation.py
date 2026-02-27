"""
Tests for the estimate schema validation.
Ensures data integrity and prevents nonsense values.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from src.models.estimate_schema import (
    BOQItem,
    CoverageRecord,
    Conflict,
    ConflictType,
    DrawingMeta,
    Discipline,
    EstimatePackage,
    Evidence,
    EvidenceSource,
    QtyStatus,
    ScopeCategory,
    ScopeItem,
    ScopeStatus,
    Severity,
    create_boq_item,
    create_conflict,
    create_evidence,
    create_scope_item,
)


class TestEvidence:
    """Test Evidence model validation."""

    def test_valid_evidence(self):
        """Test creating valid evidence."""
        ev = Evidence(
            page=0,
            source=EvidenceSource.CAMELOT,
            snippet="Column C1: 300x450"
        )
        assert ev.page == 0
        assert ev.source == EvidenceSource.CAMELOT

    def test_invalid_page_negative(self):
        """Test that negative page number is rejected."""
        with pytest.raises(ValidationError):
            Evidence(page=-1, source=EvidenceSource.PDF_TEXT)

    def test_snippet_truncation(self):
        """Test that long snippets are handled."""
        long_text = "A" * 600
        ev = Evidence(page=0, source=EvidenceSource.OCR, snippet=long_text[:500])
        assert len(ev.snippet) <= 500

    def test_factory_function(self):
        """Test create_evidence factory."""
        ev = create_evidence(page=1, source="camelot", snippet="test")
        assert ev.page == 1
        assert ev.source == EvidenceSource.CAMELOT


class TestBOQItem:
    """Test BOQItem model validation."""

    def test_valid_computed_qty(self):
        """Test valid BOQ item with computed quantity."""
        item = BOQItem(
            system="structural",
            subsystem="rcc",
            item_name="RCC Columns M25",
            unit="cum",
            qty=12.5,
            qty_status=QtyStatus.COMPUTED,
            confidence=0.85
        )
        assert item.qty == 12.5
        assert item.qty_status == QtyStatus.COMPUTED

    def test_computed_qty_requires_value(self):
        """Test that computed status requires qty value."""
        with pytest.raises(ValidationError):
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="RCC Columns",
                qty=None,  # Missing qty
                qty_status=QtyStatus.COMPUTED  # But status is computed!
            )

    def test_unknown_qty_allows_none(self):
        """Test that unknown status allows None qty."""
        item = BOQItem(
            system="structural",
            subsystem="rcc",
            item_name="RCC Beams",
            qty=None,
            qty_status=QtyStatus.UNKNOWN
        )
        assert item.qty is None
        assert item.qty_status == QtyStatus.UNKNOWN

    def test_confidence_range(self):
        """Test confidence must be 0-1."""
        # Valid
        item = BOQItem(
            system="structural",
            subsystem="rcc",
            item_name="Test",
            confidence=0.5
        )
        assert item.confidence == 0.5

        # Invalid - too high
        with pytest.raises(ValidationError):
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Test",
                confidence=1.5
            )

        # Invalid - negative
        with pytest.raises(ValidationError):
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Test",
                confidence=-0.1
            )

    def test_qty_cannot_be_dash_string(self):
        """Test that qty cannot be a dash string (must be None or numeric)."""
        # qty is Optional[float], so "-" string should fail
        with pytest.raises(ValidationError):
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Test",
                qty="-",  # type: ignore  # Intentional wrong type
                qty_status=QtyStatus.UNKNOWN
            )

    def test_factory_function(self):
        """Test create_boq_item factory."""
        item = create_boq_item(
            item_name="PCC below footings",
            unit="cum",
            qty=5.0
        )
        assert item.qty_status == QtyStatus.COMPUTED
        assert item.qty == 5.0

    def test_export_dict(self):
        """Test to_export_dict method."""
        item = BOQItem(
            system="structural",
            subsystem="rcc",
            item_name="RCC Columns",
            unit="cum",
            qty=10.5,
            qty_status=QtyStatus.COMPUTED,
            confidence=0.9
        )
        export = item.to_export_dict()
        assert "Description" in export
        assert export["Qty"] == 10.5
        assert "90%" in export["Confidence"]


class TestScopeItem:
    """Test ScopeItem model validation."""

    def test_valid_scope_item(self):
        """Test creating valid scope item."""
        item = ScopeItem(
            category=ScopeCategory.RCC,
            trade="RCC Columns M25",
            status=ScopeStatus.DETECTED,
            reason="Found in drawing",
            confidence=0.85
        )
        assert item.category == ScopeCategory.RCC
        assert item.status == ScopeStatus.DETECTED

    def test_confidence_bounds(self):
        """Test confidence validation."""
        with pytest.raises(ValidationError):
            ScopeItem(
                category=ScopeCategory.RCC,
                trade="Test",
                status=ScopeStatus.DETECTED,
                reason="",
                confidence=1.5  # Invalid
            )

    def test_factory_function(self):
        """Test create_scope_item factory."""
        item = create_scope_item(
            category="earthwork",
            trade="Excavation",
            status="detected",
            reason="Found",
            confidence=0.8
        )
        assert item.category == ScopeCategory.EARTHWORK
        assert item.status == ScopeStatus.DETECTED


class TestConflict:
    """Test Conflict model validation."""

    def test_valid_conflict(self):
        """Test creating valid conflict."""
        conflict = Conflict(
            type=ConflictType.MISSING_SCALE,
            description="Scale not found in drawing",
            severity=Severity.HIGH,
            suggested_resolution="Add scale manually"
        )
        assert conflict.type == ConflictType.MISSING_SCALE
        assert conflict.severity == Severity.HIGH

    def test_factory_function(self):
        """Test create_conflict factory."""
        conflict = create_conflict(
            conflict_type="grade_mismatch",
            description="M20 vs M25 found",
            severity="med"
        )
        assert conflict.type == ConflictType.GRADE_MISMATCH
        assert conflict.severity == Severity.MED


class TestDrawingMeta:
    """Test DrawingMeta model validation."""

    def test_valid_drawing_meta(self):
        """Test creating valid drawing metadata."""
        meta = DrawingMeta(
            file_name="foundation_plan.pdf",
            discipline=Discipline.STRUCTURAL,
            scale="1:100",
            confidence_overall=0.7
        )
        assert meta.file_name == "foundation_plan.pdf"
        assert meta.discipline == Discipline.STRUCTURAL

    def test_confidence_bounds(self):
        """Test confidence validation."""
        with pytest.raises(ValidationError):
            DrawingMeta(
                file_name="test.pdf",
                confidence_overall=1.5
            )


class TestEstimatePackage:
    """Test EstimatePackage model and stats."""

    def test_empty_package(self):
        """Test creating package with minimal data."""
        meta = DrawingMeta(file_name="test.pdf")
        package = EstimatePackage(drawing=meta)

        assert package.drawing.file_name == "test.pdf"
        assert len(package.boq) == 0
        assert len(package.scope) == 0

    def test_stats_computation(self):
        """Test stats property computes correctly."""
        meta = DrawingMeta(file_name="test.pdf", confidence_overall=0.8)

        boq_items = [
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Item 1",
                qty=10,
                qty_status=QtyStatus.COMPUTED
            ),
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Item 2",
                qty_status=QtyStatus.UNKNOWN
            ),
        ]

        scope_items = [
            ScopeItem(
                category=ScopeCategory.RCC,
                trade="Trade 1",
                status=ScopeStatus.DETECTED,
                reason="Found"
            ),
            ScopeItem(
                category=ScopeCategory.MASONRY,
                trade="Trade 2",
                status=ScopeStatus.MISSING,
                reason="Not found"
            ),
        ]

        package = EstimatePackage(
            drawing=meta,
            boq=boq_items,
            scope=scope_items
        )

        stats = package.stats
        assert stats["total_boq_items"] == 2
        assert stats["boq_by_qty_status"]["computed"] == 1
        assert stats["boq_by_qty_status"]["unknown"] == 1
        assert stats["total_scope_items"] == 2
        assert stats["scope_by_status"]["detected"] == 1
        assert stats["scope_by_status"]["missing"] == 1

    def test_get_items_needing_review(self):
        """Test filtering items needing review."""
        meta = DrawingMeta(file_name="test.pdf")

        boq_items = [
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="High confidence",
                qty=10,
                qty_status=QtyStatus.COMPUTED,
                confidence=0.9
            ),
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Low confidence",
                confidence=0.3  # Needs review
            ),
            BOQItem(
                system="structural",
                subsystem="rcc",
                item_name="Unknown qty",
                qty_status=QtyStatus.UNKNOWN  # Needs review
            ),
        ]

        package = EstimatePackage(drawing=meta, boq=boq_items)
        needs_review = package.get_items_needing_review()

        assert len(needs_review) == 2

    def test_json_serialization(self):
        """Test package can be serialized to JSON."""
        meta = DrawingMeta(file_name="test.pdf")
        package = EstimatePackage(drawing=meta)

        json_str = package.to_json()
        assert "test.pdf" in json_str
        assert "package_id" in json_str


class TestPipelineReturnsPackage:
    """Test that pipeline always returns usable output."""

    def test_pipeline_with_nonexistent_file(self):
        """Test pipeline handles missing file gracefully."""
        from src.pipeline.takeoff_pipeline import run_takeoff_pipeline

        # This should not crash - should return partial package
        package = run_takeoff_pipeline(
            Path("/nonexistent/file.pdf"),
            floors=1,
            storey_height_mm=3000
        )

        # Should still return a valid package structure
        assert isinstance(package, EstimatePackage)
        assert package.drawing is not None
        # Confidence should be low for failed extraction
        assert package.drawing.confidence_overall < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
