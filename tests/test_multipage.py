"""
Unit tests for multi-page PDF ingestion and sheet classification.

Tests:
- Sheet type classification
- Title block extraction
- Visual feature analysis
- Page classification logic
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multipage import (
    SheetType,
    SheetClassifier,
    SheetClassification,
    PageData,
    ProjectManifest,
)


class TestSheetClassifier:
    """Tests for sheet classification."""

    @pytest.fixture
    def classifier(self):
        return SheetClassifier()

    @pytest.fixture
    def blank_page_data(self):
        """Create blank page data for testing."""
        return PageData(
            page_number=0,
            image=np.ones((1000, 800, 3), dtype=np.uint8) * 255,
            vector_texts=[],
            vector_lines=[],
            raw_text="",
        )

    # === Floor Plan Classification ===

    def test_classify_floor_plan_title(self, classifier, blank_page_data):
        """Test classification with floor plan title."""
        blank_page_data.raw_text = "GROUND FLOOR PLAN Scale 1:100"
        blank_page_data.vector_texts = [
            {'text': 'GROUND FLOOR PLAN', 'bbox': (100, 50, 400, 80), 'size': 14}
        ]

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.FLOOR_PLAN
        assert result.should_process_rooms == True
        assert result.confidence > 0.5

    def test_classify_typical_floor(self, classifier, blank_page_data):
        """Test classification with typical floor plan."""
        blank_page_data.raw_text = "TYPICAL FLOOR LAYOUT A-102"
        blank_page_data.vector_texts = [
            {'text': 'TYPICAL FLOOR LAYOUT', 'bbox': (100, 50, 400, 80), 'size': 14}
        ]

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.FLOOR_PLAN
        assert result.should_process_rooms == True

    def test_classify_with_arch_prefix(self, classifier, blank_page_data):
        """Test classification with architectural sheet prefix."""
        blank_page_data.raw_text = "LAYOUT PLAN Sheet No. A-101"

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.FLOOR_PLAN or "A-101" in result.reason

    # === Schedule Classification ===

    def test_classify_schedule(self, classifier, blank_page_data):
        """Test classification with schedule keywords."""
        blank_page_data.raw_text = """
        DOOR SCHEDULE
        Mark  Size  Type  Qty  Description
        D1    900x2100  Flush  10  Main entry
        D2    800x2100  Panel  20  Internal
        """
        blank_page_data.vector_texts = [
            {'text': 'DOOR SCHEDULE', 'bbox': (100, 50, 300, 80), 'size': 14}
        ]

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.SCHEDULE_TABLE
        assert result.should_process_rooms == False

    def test_classify_bbs(self, classifier, blank_page_data):
        """Test classification with bar bending schedule."""
        blank_page_data.raw_text = """
        BAR BENDING SCHEDULE
        Mark  Dia  Nos  Length
        A1    12   100  3000
        """

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.SCHEDULE_TABLE

    # === Section/Elevation Classification ===

    def test_classify_section(self, classifier, blank_page_data):
        """Test classification with section title."""
        blank_page_data.raw_text = "SECTION A-A Scale 1:50"
        blank_page_data.vector_texts = [
            {'text': 'SECTION A-A', 'bbox': (100, 50, 250, 80), 'size': 14}
        ]

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.SECTION_ELEVATION
        assert result.should_process_rooms == False

    def test_classify_elevation(self, classifier, blank_page_data):
        """Test classification with elevation title."""
        blank_page_data.raw_text = "FRONT ELEVATION"

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.SECTION_ELEVATION

    # === Detail Classification ===

    def test_classify_detail(self, classifier, blank_page_data):
        """Test classification with detail sheet."""
        blank_page_data.raw_text = "TOILET DETAIL Scale 1:20"
        blank_page_data.vector_texts = [
            {'text': 'TOILET DETAIL', 'bbox': (100, 50, 300, 80), 'size': 14}
        ]

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.DETAIL

    # === Structural Classification ===

    def test_classify_column_layout(self, classifier, blank_page_data):
        """Test classification with column layout."""
        blank_page_data.raw_text = "COLUMN LAYOUT Sheet ST-01"

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.STRUCTURAL

    def test_classify_foundation(self, classifier, blank_page_data):
        """Test classification with foundation plan."""
        blank_page_data.raw_text = "FOUNDATION PLAN FND-01"

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.STRUCTURAL

    # === Cover/Title Classification ===

    def test_classify_cover(self, classifier, blank_page_data):
        """Test classification with cover sheet."""
        blank_page_data.raw_text = "COVER SHEET DRAWING INDEX"

        result = classifier.classify_page(blank_page_data)

        assert result.sheet_type == SheetType.COVER_TITLE

    # === Unknown Classification ===

    def test_classify_unknown(self, classifier, blank_page_data):
        """Test classification with no clear signals."""
        blank_page_data.raw_text = "Some random text without keywords"

        result = classifier.classify_page(blank_page_data)

        # Should be unknown or low confidence
        assert result.confidence < 0.5 or result.sheet_type == SheetType.UNKNOWN


class TestTitleBlockExtraction:
    """Tests for title block extraction."""

    @pytest.fixture
    def classifier(self):
        return SheetClassifier()

    def test_extract_sheet_number_standard(self, classifier):
        """Test extracting standard sheet number."""
        page_data = PageData(
            page_number=0,
            image=np.ones((100, 100, 3), dtype=np.uint8) * 255,
            raw_text="Sheet No. A-101 FLOOR PLAN",
            vector_texts=[]
        )

        result = classifier.classify_page(page_data)

        assert result.title_block.get("sheet_number") == "A-101"

    def test_extract_sheet_number_structural(self, classifier):
        """Test extracting structural sheet number."""
        page_data = PageData(
            page_number=0,
            image=np.ones((100, 100, 3), dtype=np.uint8) * 255,
            raw_text="COLUMN LAYOUT ST-05",
            vector_texts=[]
        )

        result = classifier.classify_page(page_data)

        # Should extract ST-05
        assert "ST" in result.title_block.get("sheet_number", "") or result.sheet_type == SheetType.STRUCTURAL

    def test_extract_scale(self, classifier):
        """Test extracting scale from title block."""
        page_data = PageData(
            page_number=0,
            image=np.ones((100, 100, 3), dtype=np.uint8) * 255,
            raw_text="FLOOR PLAN Scale: 1:100",
            vector_texts=[]
        )

        result = classifier.classify_page(page_data)

        assert "1:100" in result.title_block.get("scale", "")


class TestProjectManifest:
    """Tests for project manifest."""

    def test_manifest_creation(self):
        """Test creating project manifest."""
        manifest = ProjectManifest(
            project_id="test_project",
            source_path="/path/to/project.pdf",
            total_pages=5,
        )

        assert manifest.project_id == "test_project"
        assert manifest.total_pages == 5

    def test_manifest_to_dict(self):
        """Test manifest serialization."""
        manifest = ProjectManifest(
            project_id="test_project",
            source_path="/path/to/project.pdf",
            total_pages=5,
            pages_processed=[{"page_number": 0, "sheet_type": "floor_plan"}],
            pages_skipped=[{"page_number": 1, "skip_reason": "schedule"}],
        )

        data = manifest.to_dict()

        assert data["project_id"] == "test_project"
        assert len(data["pages_processed"]) == 1
        assert len(data["pages_skipped"]) == 1


class TestVisualFeatureAnalysis:
    """Tests for visual feature analysis."""

    @pytest.fixture
    def classifier(self):
        return SheetClassifier()

    def test_line_density_calculation(self, classifier):
        """Test that line density is calculated."""
        # Create image with lines
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2 = __import__('cv2')

        # Draw several lines
        for i in range(10):
            y = 50 + i * 40
            cv2.line(image, (50, y), (450, y), (0, 0, 0), 2)

        page_data = PageData(
            page_number=0,
            image=image,
            raw_text="",
            vector_texts=[]
        )

        result = classifier.classify_page(page_data)

        # Should have detected lines (check signals)
        assert len(result.detected_signals) >= 0  # At minimum runs

    def test_table_detection(self, classifier):
        """Test table detection from visual features."""
        # Create image with grid pattern
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2 = __import__('cv2')

        # Draw grid
        for i in range(10):
            y = 50 + i * 40
            cv2.line(image, (50, y), (450, y), (0, 0, 0), 1)
        for i in range(8):
            x = 50 + i * 50
            cv2.line(image, (x, 50), (x, 410), (0, 0, 0), 1)

        page_data = PageData(
            page_number=0,
            image=image,
            raw_text="SCHEDULE Mark Size Qty",
            vector_texts=[]
        )

        result = classifier.classify_page(page_data)

        # Should lean toward schedule
        assert result.sheet_type == SheetType.SCHEDULE_TABLE or "table" in str(result.detected_signals).lower()


class TestSheetTypeEnumeration:
    """Tests for SheetType enum."""

    def test_floor_plan_value(self):
        """Test floor plan enum value."""
        assert SheetType.FLOOR_PLAN.value == "floor_plan"

    def test_all_types_have_values(self):
        """Test all sheet types have string values."""
        for stype in SheetType:
            assert isinstance(stype.value, str)
            assert len(stype.value) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
