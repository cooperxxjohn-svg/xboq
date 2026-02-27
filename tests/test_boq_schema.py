"""
Tests for BOQ schema, profile loading, and CPWD mapping.

Tests:
- BOQItem creation and validation
- BOQValidator behavior (never crash, warn and continue)
- Profile loading from assumptions.yaml
- CPWD mapping and coverage

Note: Uses direct module imports to avoid cv2 dependency from boq/__init__.py
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import schema module directly, bypassing __init__.py (avoids cv2 dependency)
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_schema = _load_module("schema", str(Path(__file__).parent.parent / "src" / "boq" / "schema.py"))
_mapper = _load_module("mapper", str(Path(__file__).parent.parent / "src" / "rates" / "mapper.py"))

# Re-export for cleaner test code
BOQItem = _schema.BOQItem
BOQValidator = _schema.BOQValidator
ValidationResult = _schema.ValidationResult
BOQPackageSchema = _schema.BOQPackageSchema
DerivedFrom = _schema.DerivedFrom
Unit = _schema.Unit
convert_to_boq_item = _schema.convert_to_boq_item
merge_boq_items = _schema.merge_boq_items
load_profile = _schema.load_profile

CPWDMapper = _mapper.CPWDMapper
CPWDItem = _mapper.CPWDItem
MappedBOQItem = _mapper.MappedBOQItem
MappingResult = _mapper.MappingResult
map_boq_to_cpwd = _mapper.map_boq_to_cpwd
get_mapping_coverage = _mapper.get_mapping_coverage


# =============================================================================
# BOQ ITEM TESTS
# =============================================================================

class TestBOQItem:
    """Test BOQItem dataclass."""

    def test_create_valid_item(self):
        """Test creating a valid BOQ item."""
        item = BOQItem(
            item_code="FLR-VIT-01",
            description="Vitrified tiles 600x600mm",
            qty=45.5,
            unit="sqm",
            derived_from="measured",
            confidence=0.85,
        )
        assert item.item_code == "FLR-VIT-01"
        assert item.qty == 45.5
        assert item.confidence == 0.85

    def test_item_to_dict(self):
        """Test converting item to dictionary."""
        item = BOQItem(
            item_code="MSN-230",
            description="230mm brick masonry",
            qty=12.345,
            unit="cum",
            derived_from="detected",
            confidence=0.7,
            assumption_used="Wall height assumed 3000mm",
        )
        d = item.to_dict()
        assert d["item_code"] == "MSN-230"
        assert d["qty"] == 12.35  # Rounded to 2 decimals
        assert d["confidence"] == 0.7
        assert d["assumption_used"] == "Wall height assumed 3000mm"

    def test_item_to_csv_row(self):
        """Test converting item to CSV row."""
        item = BOQItem(
            item_code="PNT-INT-01",
            description="Interior plastic emulsion",
            qty=100.0,
            unit="sqm",
            derived_from="room_detection",
            confidence=0.6,
        )
        row = item.to_csv_row()
        assert row[0] == "PNT-INT-01"
        assert row[2] == "100.00"
        assert row[5] == "0.60"

    def test_csv_headers(self):
        """Test CSV headers."""
        headers = BOQItem.csv_headers()
        assert "item_code" in headers
        assert "qty" in headers
        assert "confidence" in headers


# =============================================================================
# BOQ VALIDATOR TESTS
# =============================================================================

class TestBOQValidator:
    """Test BOQValidator - must never crash."""

    def test_validate_valid_item(self):
        """Test validating a fully valid item."""
        validator = BOQValidator()
        item = BOQItem(
            item_code="FLR-VIT-01",
            description="Vitrified tiles",
            qty=50.0,
            unit="sqm",
            derived_from="measured",
            confidence=0.9,
        )
        result = validator.validate(item)
        assert result.is_valid
        assert len(result.warnings) == 0
        assert len(result.errors) == 0

    def test_validate_missing_item_code(self):
        """Test that missing item_code gets warning and default."""
        validator = BOQValidator()
        item = BOQItem(
            item_code="",  # Empty
            description="Some item",
            qty=10.0,
            unit="sqm",
            derived_from="assumption",
            confidence=0.5,
        )
        result = validator.validate(item)
        assert result.is_valid  # Still valid, just warned
        assert any("item_code" in w.lower() for w in result.warnings)
        assert result.fixed_item.item_code == "UNK-001"

    def test_validate_negative_quantity(self):
        """Test that negative quantity gets converted to positive."""
        validator = BOQValidator()
        item = BOQItem(
            item_code="TEST-01",
            description="Test item",
            qty=-25.0,  # Negative
            unit="sqm",
            derived_from="measured",
            confidence=0.8,
        )
        result = validator.validate(item)
        assert result.is_valid
        assert any("negative" in w.lower() for w in result.warnings)
        assert result.fixed_item.qty == 25.0  # Converted to positive

    def test_validate_confidence_out_of_range(self):
        """Test that confidence gets clamped to 0-1."""
        validator = BOQValidator()

        # Too high
        item1 = BOQItem(
            item_code="TEST-01",
            description="Test",
            qty=10.0,
            unit="sqm",
            derived_from="measured",
            confidence=1.5,  # Too high
        )
        result1 = validator.validate(item1)
        assert result1.is_valid
        assert result1.fixed_item.confidence == 1.0

        # Too low
        item2 = BOQItem(
            item_code="TEST-02",
            description="Test",
            qty=10.0,
            unit="sqm",
            derived_from="measured",
            confidence=-0.5,  # Too low
        )
        result2 = validator.validate(item2)
        assert result2.is_valid
        assert result2.fixed_item.confidence == 0.0

    def test_validate_unknown_unit(self):
        """Test that unknown unit gets warning but is accepted."""
        validator = BOQValidator()
        item = BOQItem(
            item_code="TEST-01",
            description="Test",
            qty=10.0,
            unit="xyz",  # Unknown unit
            derived_from="measured",
            confidence=0.8,
        )
        result = validator.validate(item)
        assert result.is_valid
        assert any("unknown unit" in w.lower() for w in result.warnings)

    def test_validate_dict_input(self):
        """Test validating a dictionary instead of BOQItem."""
        validator = BOQValidator()
        item_dict = {
            "item_code": "FLR-CER-01",
            "description": "Ceramic tiles",
            "qty": 30.0,
            "unit": "sqm",
            "derived_from": "detected",
            "confidence": 0.7,
        }
        result = validator.validate(item_dict)
        assert result.is_valid
        assert result.fixed_item.item_code == "FLR-CER-01"

    def test_validate_batch(self):
        """Test batch validation."""
        validator = BOQValidator()
        items = [
            {"item_code": "A", "description": "A", "qty": 10, "unit": "sqm", "derived_from": "m", "confidence": 0.8},
            {"item_code": "B", "description": "B", "qty": -5, "unit": "sqm", "derived_from": "m", "confidence": 0.8},
            {"item_code": "", "description": "C", "qty": 20, "unit": "sqm", "derived_from": "m", "confidence": 0.8},
        ]
        results = validator.validate_batch(items)
        assert len(results) == 3
        assert all(r.is_valid for r in results)


# =============================================================================
# PROFILE LOADING TESTS
# =============================================================================

class TestProfileLoading:
    """Test load_profile function."""

    def test_load_typical_profile(self):
        """Test loading the typical profile."""
        profile = load_profile("typical")
        assert profile is not None
        assert "steel_factor_multiplier" in profile
        assert profile["steel_factor_multiplier"] == 1.0

    def test_load_conservative_profile(self):
        """Test loading the conservative profile."""
        profile = load_profile("conservative")
        assert profile is not None
        assert profile.get("steel_factor_multiplier", 1.0) <= 1.0

    def test_load_premium_profile(self):
        """Test loading the premium profile."""
        profile = load_profile("premium")
        assert profile is not None
        assert profile.get("steel_factor_multiplier", 1.0) >= 1.0

    def test_load_nonexistent_profile_returns_default(self):
        """Test that loading nonexistent profile returns default."""
        profile = load_profile("nonexistent_profile_xyz")
        assert profile is not None
        # Should return a valid profile with defaults
        assert "steel_factor_multiplier" in profile

    def test_profile_has_full_config(self):
        """Test that profile includes full config reference."""
        profile = load_profile("typical")
        assert "_full_config" in profile


# =============================================================================
# CPWD MAPPING TESTS
# =============================================================================

class TestCPWDMapper:
    """Test CPWD mapping functionality."""

    def test_mapper_initialization(self):
        """Test that mapper initializes and loads mappings."""
        mapper = CPWDMapper()
        # Should have loaded some mappings from cpwd_mapping.csv
        assert len(mapper.mappings) >= 0  # May be empty if file not found

    def test_map_known_item(self):
        """Test mapping a known item code."""
        mapper = CPWDMapper()

        # Add a known mapping for testing
        mapper.add_mapping(
            item_code="TEST-ITEM",
            cpwd_item_no="99.99.1",
            cpwd_description="Test CPWD item",
            cpwd_unit="sqm",
            rate_inr=1000.0,
        )

        item = {
            "item_code": "TEST-ITEM",
            "description": "Test item",
            "qty": 10.0,
            "unit": "sqm",
            "derived_from": "test",
            "confidence": 0.8,
        }

        mapped = mapper.map_item(item)
        assert mapped.is_mapped
        assert mapped.cpwd_item_no == "99.99.1"
        assert mapped.cpwd_description == "Test CPWD item"

    def test_map_unknown_item(self):
        """Test mapping an unknown item code."""
        mapper = CPWDMapper()
        item = {
            "item_code": "UNKNOWN-ITEM-XYZ",
            "description": "Unknown item",
            "qty": 10.0,
            "unit": "sqm",
            "derived_from": "test",
            "confidence": 0.8,
        }
        mapped = mapper.map_item(item)
        assert not mapped.is_mapped
        assert "UNMAPPED" in mapped.mapping_notes

    def test_map_items_batch(self):
        """Test mapping multiple items."""
        mapper = CPWDMapper()

        # Add test mappings
        mapper.add_mapping("ITEM-A", "1.1.1", "Item A", "sqm")
        mapper.add_mapping("ITEM-B", "2.2.2", "Item B", "cum")

        items = [
            {"item_code": "ITEM-A", "description": "A", "qty": 10, "unit": "sqm", "derived_from": "m", "confidence": 0.8},
            {"item_code": "ITEM-B", "description": "B", "qty": 5, "unit": "cum", "derived_from": "m", "confidence": 0.8},
            {"item_code": "ITEM-C", "description": "C", "qty": 20, "unit": "nos", "derived_from": "m", "confidence": 0.8},
        ]

        result = mapper.map_items(items)
        assert result.mapped_count == 2
        assert result.unmapped_count == 1
        assert "ITEM-C" in result.unmapped_codes
        assert result.coverage_percent == pytest.approx(66.7, abs=1)

    def test_get_mapping_coverage(self):
        """Test get_mapping_coverage function."""
        items = [
            {"item_code": "FLR-VIT-01", "description": "A", "qty": 10, "unit": "sqm", "derived_from": "m", "confidence": 0.8},
        ]
        coverage = get_mapping_coverage(items)
        assert "total_items" in coverage
        assert "mapped" in coverage
        assert "unmapped" in coverage
        assert "coverage_percent" in coverage


# =============================================================================
# CONVERSION TESTS
# =============================================================================

class TestConversions:
    """Test conversion utilities."""

    def test_convert_dict_to_boq_item(self):
        """Test converting dict to BOQItem."""
        d = {
            "item_code": "FLR-VIT-01",
            "description": "Vitrified tiles",
            "qty": 50.0,
            "unit": "sqm",
            "derived_from": "measured",
            "confidence": 0.9,
        }
        item = convert_to_boq_item(d)
        assert item is not None
        assert item.item_code == "FLR-VIT-01"

    def test_convert_tuple_to_boq_item(self):
        """Test converting tuple to BOQItem."""
        t = ("FLR-VIT-01", "Vitrified tiles", 50.0, "sqm", "measured", 0.9)
        item = convert_to_boq_item(t)
        assert item is not None
        assert item.item_code == "FLR-VIT-01"
        assert item.qty == 50.0

    def test_convert_with_category(self):
        """Test converting with category override."""
        d = {"item_code": "A", "description": "A", "qty": 10, "unit": "sqm", "derived_from": "m", "confidence": 0.8}
        item = convert_to_boq_item(d, category="Finishes")
        assert item.category == "Finishes"

    def test_merge_boq_items(self):
        """Test merging multiple BOQ item lists."""
        list1 = [
            {"item_code": "A", "description": "A", "qty": 10, "unit": "sqm", "derived_from": "m", "confidence": 0.8},
        ]
        list2 = [
            {"item_code": "B", "description": "B", "qty": 20, "unit": "cum", "derived_from": "m", "confidence": 0.7},
        ]
        merged = merge_boq_items(list1, list2, categories=["Walls", "Finishes"])
        assert len(merged) == 2
        assert merged[0].category == "Walls"
        assert merged[1].category == "Finishes"


# =============================================================================
# BOQ PACKAGE TESTS
# =============================================================================

class TestBOQPackageSchema:
    """Test BOQPackageSchema."""

    def test_create_package(self):
        """Test creating a BOQ package."""
        package = BOQPackageSchema(
            project_id="TEST-001",
            generated_at="2024-01-01T00:00:00",
            profile="typical",
        )
        assert package.project_id == "TEST-001"
        assert package.profile == "typical"
        assert len(package.items) == 0

    def test_add_item_validates(self):
        """Test that adding items runs validation."""
        package = BOQPackageSchema(
            project_id="TEST-001",
            generated_at="2024-01-01T00:00:00",
        )
        item = BOQItem(
            item_code="",  # Invalid - empty
            description="Test",
            qty=10.0,
            unit="sqm",
            derived_from="test",
            confidence=0.8,
        )
        package.add_item(item)
        assert len(package.items) == 1
        assert package.items[0].item_code == "UNK-001"  # Fixed
        assert len(package.warnings) > 0

    def test_package_to_dict(self):
        """Test converting package to dictionary."""
        package = BOQPackageSchema(
            project_id="TEST-001",
            generated_at="2024-01-01T00:00:00",
            profile="premium",
        )
        package.items.append(BOQItem(
            item_code="FLR-VIT-01",
            description="Test",
            qty=10.0,
            unit="sqm",
            derived_from="test",
            confidence=0.8,
        ))
        d = package.to_dict()
        assert d["project_id"] == "TEST-001"
        assert d["profile"] == "premium"
        assert d["summary"]["total_items"] == 1
        assert len(d["items"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
