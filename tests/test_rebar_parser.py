"""
Tests for the rebar parser.
Ensures quantity vs diameter are correctly distinguished.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.column_schedule_extractor import (
    parse_rebar_spec,
    parse_tie_spec,
    parse_column_marks,
    parse_section_size,
)


class TestRebarParser:
    """Test cases for rebar specification parsing."""

    def test_pattern_18_25(self):
        """Test '18-25' pattern -> 18 bars of 25mm diameter."""
        result = parse_rebar_spec("18-25")
        assert len(result) >= 1
        assert result[0].quantity == 18
        assert result[0].diameter_mm == 25

    def test_pattern_4Y25(self):
        """Test '4Y25' pattern -> 4 bars of 25mm diameter."""
        result = parse_rebar_spec("4Y25")
        assert len(result) >= 1
        assert result[0].quantity == 4
        assert result[0].diameter_mm == 25

    def test_pattern_16_Y20(self):
        """Test '16-Y20' pattern -> 16 bars of 20mm diameter."""
        result = parse_rebar_spec("16-Y20")
        assert len(result) >= 1
        assert result[0].quantity == 16
        assert result[0].diameter_mm == 20

    def test_pattern_with_spaces(self):
        """Test patterns with spaces."""
        result = parse_rebar_spec("18 - 25")
        assert len(result) >= 1
        assert result[0].quantity == 18
        assert result[0].diameter_mm == 25

    def test_pattern_8_nos_25(self):
        """Test '8 nos 25' pattern."""
        result = parse_rebar_spec("8 nos 25")
        assert len(result) >= 1
        assert result[0].quantity == 8
        assert result[0].diameter_mm == 25

    def test_multiple_patterns(self):
        """Test multiple rebar specs in one string."""
        # This tests if the parser can handle combined specs
        result = parse_rebar_spec("4Y25 + 4Y20")
        # Should find at least one pattern
        assert len(result) >= 1

    def test_invalid_input_empty(self):
        """Test empty input returns empty list."""
        result = parse_rebar_spec("")
        assert result == []

    def test_invalid_input_none(self):
        """Test None input returns empty list."""
        result = parse_rebar_spec(None)
        assert result == []

    def test_common_diameters(self):
        """Test that common Indian rebar diameters are recognized."""
        common_dias = [8, 10, 12, 16, 20, 25, 28, 32]
        for dia in common_dias:
            result = parse_rebar_spec(f"4-{dia}")
            assert len(result) >= 1
            assert result[0].diameter_mm == dia

    def test_quantity_vs_diameter_distinction(self):
        """Critical test: ensure quantity is not confused with diameter."""
        # In "18-25", 18 is quantity, 25 is diameter
        # NOT 25 bars of 18mm!
        result = parse_rebar_spec("18-25")
        assert len(result) >= 1

        # The parser should use heuristics:
        # - Common diameters: 8,10,12,16,20,25,28,32,36,40
        # - Quantity is usually smaller (1-50)
        # - In "X-Y" format, typically quantity-diameter

        # 25mm is a common diameter, 18 is a reasonable quantity
        assert result[0].quantity == 18
        assert result[0].diameter_mm == 25


class TestTieParser:
    """Test cases for tie/stirrup specification parsing."""

    def test_basic_tie_8_at_150(self):
        """Test '8@150' pattern."""
        result = parse_tie_spec("8@150")
        assert result is not None
        assert result.diameter_mm == 8
        assert result.spacing_mm == 150

    def test_tie_with_Y_prefix(self):
        """Test 'Y8@150' pattern."""
        result = parse_tie_spec("Y8@150")
        assert result is not None
        assert result.diameter_mm == 8
        assert result.spacing_mm == 150

    def test_tie_with_legs(self):
        """Test '2L 8@150' pattern."""
        result = parse_tie_spec("2L 8@150")
        assert result is not None
        assert result.legs == 2
        assert result.diameter_mm == 8
        assert result.spacing_mm == 150

    def test_tie_4_legs(self):
        """Test '4L Y10@200' pattern."""
        result = parse_tie_spec("4L Y10@200")
        assert result is not None
        assert result.legs == 4
        assert result.diameter_mm == 10
        assert result.spacing_mm == 200

    def test_invalid_tie_returns_none(self):
        """Test invalid input returns None."""
        assert parse_tie_spec("") is None
        assert parse_tie_spec(None) is None
        assert parse_tie_spec("invalid") is None


class TestColumnMarks:
    """Test cases for column mark parsing."""

    def test_comma_separated(self):
        """Test comma-separated marks."""
        result = parse_column_marks("C1, C2, C3")
        assert "C1" in result
        assert "C2" in result
        assert "C3" in result

    def test_space_separated(self):
        """Test space-separated marks."""
        result = parse_column_marks("C1 C2 C3")
        assert len(result) >= 3

    def test_range_pattern(self):
        """Test range pattern 'C1-C5'."""
        result = parse_column_marks("C1-C5")
        assert "C1" in result
        assert "C5" in result
        # Should expand the range
        assert len(result) >= 5

    def test_mixed_format(self):
        """Test mixed format."""
        result = parse_column_marks("C36, C42, C44, C45")
        assert "C36" in result
        assert "C42" in result
        assert "C44" in result
        assert "C45" in result

    def test_empty_input(self):
        """Test empty input."""
        assert parse_column_marks("") == []
        assert parse_column_marks(None) == []


class TestSectionSize:
    """Test cases for section size parsing."""

    def test_300x600(self):
        """Test '300x600' pattern."""
        result = parse_section_size("300x600")
        assert result == (300, 600)

    def test_with_spaces(self):
        """Test '300 x 600' pattern."""
        result = parse_section_size("300 x 600")
        assert result == (300, 600)

    def test_uppercase_X(self):
        """Test '300X600' pattern."""
        result = parse_section_size("300X600")
        assert result == (300, 600)

    def test_square_section(self):
        """Test '450x450' pattern."""
        result = parse_section_size("450x450")
        assert result == (450, 450)

    def test_invalid_returns_none(self):
        """Test invalid input returns None."""
        assert parse_section_size("") is None
        assert parse_section_size(None) is None
        assert parse_section_size("invalid") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
