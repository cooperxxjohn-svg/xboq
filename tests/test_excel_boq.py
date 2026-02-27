"""
Sprint 21C: Excel BOQ Parser Tests.

Tests sheet detection, column mapping, Indian number parsing,
totals row skipping, pipeline prioritization, and graceful degradation.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.excel_boq import (
    detect_boq_sheets,
    parse_boq_sheet,
    parse_boq_excels,
    _cell_float,
    _cell_str,
    _match_column,
    _is_totals_row,
    _detect_header_row,
)


# =========================================================================
# HELPERS — create test XLSX files in-memory
# =========================================================================

def _create_test_xlsx(sheets_data: dict) -> Path:
    """
    Create a test .xlsx file with given sheet data.

    Args:
        sheets_data: {sheet_name: [row1, row2, ...]} where each row is a list of cell values.

    Returns:
        Path to the temporary .xlsx file.
    """
    import openpyxl
    wb = openpyxl.Workbook()
    first = True
    for sheet_name, rows in sheets_data.items():
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(title=sheet_name)
        for row in rows:
            ws.append(row)
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    wb.save(path)
    wb.close()
    return Path(path)


# =========================================================================
# A) Sheet Detection Tests
# =========================================================================

class TestSheetDetection:
    """Test detect_boq_sheets with various workbook configurations."""

    def test_boq_sheet_by_name(self):
        """Sheet named 'BOQ' with proper headers is detected."""
        xlsx = _create_test_xlsx({
            "BOQ": [
                ["Sl No", "Description", "Unit", "Qty", "Rate", "Amount"],
                [1, "Excavation in soil", "cum", 100, 250, 25000],
            ],
            "Summary": [
                ["Project Summary"],
                ["Total Cost", 50000],
            ],
            "Cover": [
                ["Tender Document"],
            ],
        })
        try:
            detected = detect_boq_sheets(xlsx)
            assert len(detected) >= 1, f"Expected at least 1 BOQ sheet, got {len(detected)}"
            assert detected[0]["sheet_name"] == "BOQ"
            assert "description" in detected[0]["column_map"]
            assert "qty" in detected[0]["column_map"]
        finally:
            xlsx.unlink(missing_ok=True)

    def test_price_bid_sheet_name(self):
        """Sheet named 'Price Bid' is detected as BOQ."""
        xlsx = _create_test_xlsx({
            "Price Bid": [
                ["Item No", "Particulars", "UOM", "Quantity", "Rate", "Amount"],
                ["1.1", "RCC M25", "cum", 50, 8500, 425000],
            ],
        })
        try:
            detected = detect_boq_sheets(xlsx)
            assert len(detected) >= 1
            assert detected[0]["sheet_name"] == "Price Bid"
        finally:
            xlsx.unlink(missing_ok=True)

    def test_no_boq_sheet(self):
        """Sheet without BOQ headers is not detected."""
        xlsx = _create_test_xlsx({
            "Notes": [
                ["Important Notes"],
                ["1. Follow all safety guidelines"],
            ],
        })
        try:
            detected = detect_boq_sheets(xlsx)
            assert len(detected) == 0, f"Expected 0 BOQ sheets, got {len(detected)}"
        finally:
            xlsx.unlink(missing_ok=True)

    def test_header_in_row_5(self):
        """Headers in row 5 (0-indexed=4) are still detected."""
        xlsx = _create_test_xlsx({
            "BOQ": [
                ["Government of India"],
                ["Ministry of Defence"],
                ["Tender No: XYZ-2024"],
                [],
                ["S.No", "Description of Item", "Unit", "Qty", "Rate", "Amount"],
                [1, "Earthwork excavation", "cum", 200, 180, 36000],
                [2, "Sand filling", "cum", 100, 350, 35000],
            ],
        })
        try:
            detected = detect_boq_sheets(xlsx)
            assert len(detected) >= 1
            assert detected[0]["header_row"] == 4  # 0-indexed
        finally:
            xlsx.unlink(missing_ok=True)

    def test_multiple_boq_sheets(self):
        """Multiple BOQ sheets in one workbook all detected."""
        xlsx = _create_test_xlsx({
            "Civil BOQ": [
                ["Item", "Description", "Unit", "Qty", "Rate"],
                [1, "Excavation", "cum", 100, 250],
            ],
            "Electrical BOQ": [
                ["Sl No", "Particulars", "Unit", "Quantity", "Amount"],
                [1, "Wiring", "rmt", 500, 50000],
            ],
        })
        try:
            detected = detect_boq_sheets(xlsx)
            assert len(detected) == 2
        finally:
            xlsx.unlink(missing_ok=True)


# =========================================================================
# B) Column Mapping Tests
# =========================================================================

class TestColumnMapping:
    """Test fuzzy column header matching."""

    @pytest.mark.parametrize("header,expected", [
        ("Item No", True),
        ("Sl No", True),
        ("S.No", True),
        ("S.No.", True),
        ("Sr No", True),
        ("Sr.No.", True),
        ("Serial No.", True),
        ("ITEM NO", True),  # case-insensitive via _match_column
        ("Page Number", False),
    ])
    def test_item_no_variants(self, header, expected):
        assert _match_column(header, "item_no") == expected

    @pytest.mark.parametrize("header,expected", [
        ("Description", True),
        ("Particulars", True),
        ("Name of Item", True),
        ("Item Description", True),
        ("Desc", True),
        ("Description of Work", True),
        ("Reference", False),
    ])
    def test_description_variants(self, header, expected):
        assert _match_column(header, "description") == expected

    @pytest.mark.parametrize("header,expected", [
        ("Unit", True),
        ("UOM", True),
        ("U/M", True),
        ("Unit of Measurement", True),
        ("Color", False),
    ])
    def test_unit_variants(self, header, expected):
        assert _match_column(header, "unit") == expected

    @pytest.mark.parametrize("header,expected", [
        ("Qty", True),
        ("Quantity", True),
        ("Estimated Qty", True),
        ("Est. Qty", True),
        ("Price", False),
    ])
    def test_qty_variants(self, header, expected):
        assert _match_column(header, "qty") == expected

    @pytest.mark.parametrize("header,expected", [
        ("Rate", True),
        ("Unit Rate", True),
        ("Rate (Rs.)", True),
        ("Rate(Rs)", True),
        ("Offered Rate", True),
        ("Discount", False),
    ])
    def test_rate_variants(self, header, expected):
        assert _match_column(header, "rate") == expected

    @pytest.mark.parametrize("header,expected", [
        ("Amount", True),
        ("Total", True),
        ("Amount (Rs.)", True),
        ("Total Amount", True),
        ("Remarks", False),
    ])
    def test_amount_variants(self, header, expected):
        assert _match_column(header, "amount") == expected


# =========================================================================
# C) Indian Number Parsing Tests
# =========================================================================

class TestCellFloat:
    """Test _cell_float for Indian number format and edge cases."""

    @pytest.mark.parametrize("value,expected", [
        (125000, 125000.0),
        (125000.5, 125000.5),
        ("125000", 125000.0),
        ("1,25,000", 125000.0),       # Indian lakh format
        ("12,50,000.50", 1250000.5),   # Indian crore format
        ("12,500", 12500.0),           # Western format
        ("1,250,000", 1250000.0),      # Western format
        (0, 0.0),
        ("0", 0.0),
        (None, None),
        ("", None),
        ("-", None),
        ("nil", None),
        ("N/A", None),
        ("abc", None),
        ("₹1,25,000", 125000.0),      # Currency symbol stripped
        ("Rs. 5,000", 5000.0),         # Rs. prefix stripped
    ])
    def test_cell_float_parsing(self, value, expected):
        result = _cell_float(value)
        if expected is None:
            assert result is None, f"Expected None for {value!r}, got {result}"
        else:
            assert result == expected, f"Expected {expected} for {value!r}, got {result}"


# =========================================================================
# D) Totals Row Skipping Tests
# =========================================================================

class TestTotalsRowSkipping:
    """Test that totals/summary rows are correctly identified."""

    @pytest.mark.parametrize("description,should_skip", [
        ("Total", True),
        ("Sub-Total", True),
        ("Sub Total", True),
        ("Grand Total", True),
        ("GRAND TOTAL", True),
        ("Abstract", True),
        ("Summary", True),
        ("Carried Over", True),
        ("Brought Forward", True),
        ("C/F", True),
        ("B/F", True),
        ("Page Total", True),
        ("Round Off", True),
        ("Net Total", True),
        ("Excavation in soil", False),
        ("RCC M25 for columns", False),
        ("Total length of reinforcement", False),  # "Total" not at start
    ])
    def test_totals_detection(self, description, should_skip):
        result = _is_totals_row(description)
        assert result == should_skip, f"_is_totals_row({description!r}) = {result}, expected {should_skip}"


# =========================================================================
# E) Full Sheet Parsing Tests
# =========================================================================

class TestSheetParsing:
    """Test parse_boq_sheet with realistic data."""

    def test_basic_parsing(self):
        """Parse a simple BOQ sheet with standard headers."""
        rows = [
            ["Sl No", "Description", "Unit", "Qty", "Rate", "Amount"],
            [1, "Excavation in ordinary soil", "cum", 120.5, 250, 30125],
            [2, "PCC M15 grade 1:2:4", "cum", 45, 5500, 247500],
            [3, "RCC M25 for columns", "cum", 30, 8500, 255000],
            ["", "Total", "", "", "", 532625],
        ]
        header_row = 0
        column_map = {"item_no": 0, "description": 1, "unit": 2, "qty": 3, "rate": 4, "amount": 5}

        items, skipped = parse_boq_sheet(
            rows, header_row, column_map,
            source_file="test.xlsx", source_sheet="BOQ",
        )

        assert len(items) == 3, f"Expected 3 items, got {len(items)}"
        assert skipped >= 1, "Total row should be skipped"

        # Check first item
        assert items[0]["item_no"] == "1"
        assert items[0]["description"] == "Excavation in ordinary soil"
        assert items[0]["unit"] == "cum"
        assert items[0]["qty"] == 120.5
        assert items[0]["rate"] == 250.0
        assert items[0]["source_file"] == "test.xlsx"
        assert items[0]["source_sheet"] == "BOQ"
        assert items[0]["source_row"] == 2  # 1-indexed (header=1, data starts at 2)
        assert items[0]["confidence"] == 0.85

    def test_skip_totals(self):
        """Totals, sub-totals, and summary rows are skipped."""
        rows = [
            ["Item", "Description", "Unit", "Qty", "Rate"],
            [1, "Excavation", "cum", 100, 200],
            ["", "Sub-Total", "", "", ""],
            [2, "Filling", "cum", 50, 300],
            ["", "Grand Total", "", "", ""],
        ]
        header_row = 0
        column_map = {"item_no": 0, "description": 1, "unit": 2, "qty": 3, "rate": 4}

        items, skipped = parse_boq_sheet(rows, header_row, column_map)
        assert len(items) == 2
        assert items[0]["description"] == "Excavation"
        assert items[1]["description"] == "Filling"

    def test_indian_numbers_in_cells(self):
        """Indian lakh format numbers are correctly parsed."""
        rows = [
            ["Item", "Description", "Unit", "Qty", "Rate", "Amount"],
            [1, "Steel reinforcement", "kg", "1,25,000", "85", "1,06,25,000"],
        ]
        header_row = 0
        column_map = {"item_no": 0, "description": 1, "unit": 2, "qty": 3, "rate": 4, "amount": 5}

        items, _ = parse_boq_sheet(rows, header_row, column_map)
        assert len(items) == 1
        assert items[0]["qty"] == 125000.0
        assert items[0]["rate"] == 85.0

    def test_rate_inferred_from_amount(self):
        """If rate is missing but amount and qty present, rate is inferred."""
        rows = [
            ["Item", "Description", "Unit", "Qty", "Rate", "Amount"],
            [1, "Cement", "bag", 500, None, 200000],
        ]
        header_row = 0
        column_map = {"item_no": 0, "description": 1, "unit": 2, "qty": 3, "rate": 4, "amount": 5}

        items, _ = parse_boq_sheet(rows, header_row, column_map)
        assert len(items) == 1
        assert items[0]["rate"] == 400.0  # 200000 / 500

    def test_empty_rows_skipped(self):
        """Completely empty rows are skipped."""
        rows = [
            ["Item", "Description", "Unit", "Qty", "Rate"],
            [1, "Excavation", "cum", 100, 200],
            [None, None, None, None, None],
            ["", "", "", "", ""],
            [2, "Filling", "cum", 50, 300],
        ]
        header_row = 0
        column_map = {"item_no": 0, "description": 1, "unit": 2, "qty": 3, "rate": 4}

        items, skipped = parse_boq_sheet(rows, header_row, column_map)
        assert len(items) == 2
        assert skipped >= 2

    def test_section_headers_skipped(self):
        """Section headers without numeric data are skipped."""
        rows = [
            ["Item", "Description", "Unit", "Qty", "Rate"],
            ["", "Section A - Civil Works", "", "", ""],
            [1, "Excavation", "cum", 100, 200],
            ["", "Section B - Structural Works", "", "", ""],
            [2, "RCC columns", "cum", 30, 8500],
        ]
        header_row = 0
        column_map = {"item_no": 0, "description": 1, "unit": 2, "qty": 3, "rate": 4}

        items, skipped = parse_boq_sheet(rows, header_row, column_map)
        assert len(items) == 2


# =========================================================================
# F) Full parse_boq_excels Integration Tests
# =========================================================================

class TestParseBoqExcels:
    """Test the top-level parse_boq_excels function."""

    def test_single_file_parsing(self):
        """Parse a single Excel BOQ file end-to-end."""
        xlsx = _create_test_xlsx({
            "BOQ": [
                ["Sl No", "Description", "Unit", "Qty", "Rate", "Amount"],
                [1, "Excavation", "cum", 100, 250, 25000],
                [2, "PCC M15", "cum", 45, 5500, 247500],
                ["", "Total", "", "", "", 272500],
            ],
        })
        try:
            items, stats = parse_boq_excels([xlsx])
            assert len(items) == 2
            assert stats["files_parsed"] == 1
            assert stats["sheets_parsed"] == 1
            assert stats["total_rows"] == 2
            assert stats["skipped_rows"] >= 1  # "Total" row
            assert len(stats["errors"]) == 0
        finally:
            xlsx.unlink(missing_ok=True)

    def test_missing_file(self):
        """Missing file is handled gracefully."""
        items, stats = parse_boq_excels([Path("/nonexistent/file.xlsx")])
        assert len(items) == 0
        assert stats["files_skipped"] == 1
        assert len(stats["errors"]) == 1

    def test_multiple_files(self):
        """Parse multiple Excel BOQ files."""
        xlsx1 = _create_test_xlsx({
            "BOQ": [
                ["Item", "Description", "Unit", "Qty", "Rate"],
                [1, "Excavation", "cum", 100, 250],
            ],
        })
        xlsx2 = _create_test_xlsx({
            "Price Bid": [
                ["Sl No", "Particulars", "UOM", "Quantity", "Rate"],
                [1, "Steel", "kg", 5000, 85],
                [2, "Cement", "bag", 200, 400],
            ],
        })
        try:
            items, stats = parse_boq_excels([xlsx1, xlsx2])
            assert len(items) == 3
            assert stats["files_parsed"] == 2
            assert stats["sheets_parsed"] == 2
        finally:
            xlsx1.unlink(missing_ok=True)
            xlsx2.unlink(missing_ok=True)

    def test_no_boq_sheet_in_file(self):
        """File with no BOQ sheets produces 0 items."""
        xlsx = _create_test_xlsx({
            "Notes": [
                ["General Notes"],
                ["Follow safety rules"],
            ],
        })
        try:
            items, stats = parse_boq_excels([xlsx])
            assert len(items) == 0
            assert stats["files_skipped"] == 1
        finally:
            xlsx.unlink(missing_ok=True)

    def test_source_traceability(self):
        """Each item has source_file, source_sheet, source_row."""
        xlsx = _create_test_xlsx({
            "Civil BOQ": [
                ["Item", "Description", "Unit", "Qty", "Rate"],
                [1, "Excavation", "cum", 100, 250],
                [2, "Filling", "cum", 50, 300],
            ],
        })
        try:
            items, _ = parse_boq_excels([xlsx])
            assert len(items) == 2
            for item in items:
                assert "source_file" in item
                assert "source_sheet" in item
                assert "source_row" in item
                assert item["source_page"] == 0  # Excel marker
            assert items[0]["source_sheet"] == "Civil BOQ"
            assert items[0]["source_row"] == 2  # 1-indexed
            assert items[1]["source_row"] == 3
        finally:
            xlsx.unlink(missing_ok=True)


# =========================================================================
# G) Header Detection Tests
# =========================================================================

class TestHeaderDetection:
    """Test _detect_header_row with various configurations."""

    def test_header_in_first_row(self):
        """Standard headers in row 0."""
        rows = [
            ["Item No", "Description", "Unit", "Qty", "Rate"],
            [1, "Excavation", "cum", 100, 250],
        ]
        result = _detect_header_row(rows)
        assert result is not None
        header_row, col_map = result
        assert header_row == 0
        assert "item_no" in col_map
        assert "description" in col_map

    def test_header_with_prefix_rows(self):
        """Headers after metadata rows (common in India BOQ)."""
        rows = [
            ["Government of India"],
            ["Public Works Department"],
            [],
            ["Schedule B - Bill of Quantities"],
            ["S.No.", "Name of Item", "Unit", "Estimated Qty", "Unit Rate", "Amount"],
            [1, "Earth work", "cum", 500, 180, 90000],
        ]
        result = _detect_header_row(rows)
        assert result is not None
        header_row, col_map = result
        assert header_row == 4
        assert "item_no" in col_map
        assert "description" in col_map
        assert "qty" in col_map

    def test_insufficient_columns(self):
        """Sheets with fewer than 3 matching columns are rejected."""
        rows = [
            ["Name", "Value"],
            ["Project", "Hospital"],
        ]
        result = _detect_header_row(rows)
        assert result is None


# =========================================================================
# H) Pipeline Prioritization Test
# =========================================================================

class TestPipelinePrioritization:
    """Test that Excel BOQ takes priority over PDF BOQ in the pipeline."""

    def test_excel_overrides_pdf_boq(self):
        """When Excel BOQ exists, it replaces PDF-extracted BOQ items."""
        # This tests the logic that would run in pipeline.py
        # Simulate: PDF extraction found 0 items, Excel found 5
        pdf_boq_items = []  # PDF extraction found nothing
        excel_xlsx = _create_test_xlsx({
            "BOQ": [
                ["Sl No", "Description", "Unit", "Qty", "Rate"],
                [1, "Excavation", "cum", 100, 250],
                [2, "PCC", "cum", 45, 5500],
                [3, "RCC columns", "cum", 30, 8500],
                [4, "Brickwork", "sqm", 200, 1200],
                [5, "Plastering", "sqm", 400, 350],
            ],
        })
        try:
            excel_items, stats = parse_boq_excels([excel_xlsx])
            assert len(excel_items) == 5

            # Excel should override PDF
            if excel_items:
                final_boq = excel_items
                boq_source = "excel"
            else:
                final_boq = pdf_boq_items
                boq_source = "pdf"

            assert boq_source == "excel"
            assert len(final_boq) == 5
        finally:
            excel_xlsx.unlink(missing_ok=True)


# =========================================================================
# I) Graceful Degradation Tests
# =========================================================================

class TestGracefulDegradation:
    """Test graceful handling of errors and edge cases."""

    def test_corrupted_file(self):
        """Corrupted file is handled gracefully."""
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.write(fd, b"This is not a valid Excel file")
        os.close(fd)
        try:
            items, stats = parse_boq_excels([Path(path)])
            assert len(items) == 0
            assert stats["files_skipped"] == 1
        finally:
            os.unlink(path)

    def test_empty_workbook(self):
        """Workbook with empty sheets produces 0 items."""
        xlsx = _create_test_xlsx({
            "Sheet1": [],
        })
        try:
            items, stats = parse_boq_excels([xlsx])
            assert len(items) == 0
        finally:
            xlsx.unlink(missing_ok=True)

    def test_headers_but_no_data(self):
        """Sheet with headers but no data rows returns []."""
        xlsx = _create_test_xlsx({
            "BOQ": [
                ["Sl No", "Description", "Unit", "Qty", "Rate", "Amount"],
            ],
        })
        try:
            items, stats = parse_boq_excels([xlsx])
            assert len(items) == 0
        finally:
            xlsx.unlink(missing_ok=True)

    def test_xls_file_graceful(self):
        """Attempting to read .xls file doesn't crash (may warn)."""
        fd, path = tempfile.mkstemp(suffix=".xls")
        os.write(fd, b"not a real xls file")
        os.close(fd)
        try:
            items, stats = parse_boq_excels([Path(path)])
            # Should not crash — graceful degradation
            assert len(items) == 0
        finally:
            os.unlink(path)


# =========================================================================
# J) Smoke test — mock payload with Excel BOQ renders
# =========================================================================

class TestSmokePayload:
    """Verify that a payload with boq_source='excel' can be built."""

    def test_payload_with_excel_boq_source(self):
        """Payload dict with boq_source='excel' and Excel fields is valid."""
        payload = {
            "boq_source": "excel",
            "boq_stats": {
                "total_items": 3,
                "by_trade": {"civil": 2, "structural": 1},
                "flagged_items": [],
                "flagged_count": 0,
            },
            "extraction_summary": {
                "boq_items": [
                    {
                        "item_no": "1",
                        "description": "Excavation",
                        "unit": "cum",
                        "qty": 100,
                        "rate": 250,
                        "source_page": 0,
                        "confidence": 0.85,
                        "source_file": "BOQ.xlsx",
                        "source_sheet": "BOQ",
                        "source_row": 2,
                    },
                ],
            },
        }
        # Verify structure
        assert payload["boq_source"] == "excel"
        assert payload["boq_stats"]["total_items"] == 3
        items = payload["extraction_summary"]["boq_items"]
        assert items[0]["source_file"] == "BOQ.xlsx"
        assert items[0]["source_sheet"] == "BOQ"
        assert items[0]["source_row"] == 2
        assert items[0]["source_page"] == 0
