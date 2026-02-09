"""
CPWD/DSR Rate Mapping Module

Maps XBOQ item codes to CPWD (Central Public Works Department) schedule items.
This enables integration with government rate contracts and cost estimation.

Features:
- Load mapping from CSV
- Map item_code to CPWD item description + unit
- Flag unmapped items
- Generate BOQ with CPWD mapping
"""

import csv
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CPWDItem:
    """CPWD schedule item."""
    item_code: str  # Our internal code
    cpwd_item_no: str
    cpwd_description: str
    cpwd_unit: str
    rate_inr: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class MappedBOQItem:
    """BOQ item with CPWD mapping."""
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    assumption_used: Optional[str] = None
    notes: Optional[str] = None
    category: Optional[str] = None
    # CPWD mapping fields
    cpwd_item_no: Optional[str] = None
    cpwd_description: Optional[str] = None
    cpwd_unit: Optional[str] = None
    cpwd_rate_inr: Optional[float] = None
    is_mapped: bool = False
    mapping_notes: Optional[str] = None


@dataclass
class MappingResult:
    """Result of BOQ to CPWD mapping."""
    mapped_items: List[MappedBOQItem]
    unmapped_count: int
    mapped_count: int
    coverage_percent: float
    unmapped_codes: List[str]


class CPWDMapper:
    """
    Map BOQ items to CPWD schedule.

    Usage:
        mapper = CPWDMapper()
        result = mapper.map_items(boq_items)
        mapper.export_csv(result.mapped_items, output_path)
    """

    def __init__(
        self,
        mapping_path: Optional[Path] = None,
    ):
        self.mappings: Dict[str, CPWDItem] = {}
        self.load_mappings(mapping_path)

    def load_mappings(self, path: Optional[Path] = None) -> None:
        """Load CPWD mappings from CSV."""
        if path is None:
            path = Path(__file__).parent.parent.parent / "rates" / "cpwd_mapping.csv"

        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item = CPWDItem(
                        item_code=row.get("item_code", ""),
                        cpwd_item_no=row.get("cpwd_item_no", ""),
                        cpwd_description=row.get("cpwd_description", ""),
                        cpwd_unit=row.get("cpwd_unit", ""),
                        rate_inr=float(row["rate_inr"]) if row.get("rate_inr") else None,
                        notes=row.get("notes"),
                    )
                    if item.item_code:
                        self.mappings[item.item_code] = item

            logger.info(f"Loaded {len(self.mappings)} CPWD mappings")

        except FileNotFoundError:
            logger.warning(f"CPWD mapping file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading CPWD mappings: {e}")

    def get_mapping(self, item_code: str) -> Optional[CPWDItem]:
        """Get CPWD mapping for item code."""
        return self.mappings.get(item_code)

    def map_item(self, item: Any) -> MappedBOQItem:
        """
        Map a single BOQ item to CPWD.

        Args:
            item: BOQ item (dict or object)

        Returns:
            MappedBOQItem with CPWD fields populated
        """
        # Extract item fields
        if isinstance(item, dict):
            item_code = item.get("item_code", "")
            description = item.get("description", "")
            qty = item.get("qty", 0)
            unit = item.get("unit", "")
            derived_from = item.get("derived_from", "")
            confidence = item.get("confidence", 0.5)
            assumption_used = item.get("assumption_used")
            notes = item.get("notes")
            category = item.get("category")
        else:
            item_code = getattr(item, "item_code", "")
            description = getattr(item, "description", "")
            qty = getattr(item, "qty", 0)
            unit = getattr(item, "unit", "")
            derived_from = getattr(item, "derived_from", "")
            confidence = getattr(item, "confidence", 0.5)
            assumption_used = getattr(item, "assumption_used", None)
            notes = getattr(item, "notes", None)
            category = getattr(item, "category", None)

        # Get CPWD mapping
        cpwd = self.get_mapping(item_code)

        if cpwd:
            return MappedBOQItem(
                item_code=item_code,
                description=description,
                qty=qty,
                unit=unit,
                derived_from=derived_from,
                confidence=confidence,
                assumption_used=assumption_used,
                notes=notes,
                category=category,
                cpwd_item_no=cpwd.cpwd_item_no,
                cpwd_description=cpwd.cpwd_description,
                cpwd_unit=cpwd.cpwd_unit,
                cpwd_rate_inr=cpwd.rate_inr,
                is_mapped=True,
                mapping_notes=cpwd.notes,
            )
        else:
            return MappedBOQItem(
                item_code=item_code,
                description=description,
                qty=qty,
                unit=unit,
                derived_from=derived_from,
                confidence=confidence,
                assumption_used=assumption_used,
                notes=notes,
                category=category,
                is_mapped=False,
                mapping_notes="UNMAPPED - No CPWD item found",
            )

    def map_items(self, items: List[Any]) -> MappingResult:
        """
        Map all BOQ items to CPWD.

        Args:
            items: List of BOQ items

        Returns:
            MappingResult with mapped items and statistics
        """
        mapped_items = []
        unmapped_codes = []

        for item in items:
            mapped = self.map_item(item)
            mapped_items.append(mapped)

            if not mapped.is_mapped:
                code = mapped.item_code
                if code and code not in unmapped_codes:
                    unmapped_codes.append(code)

        mapped_count = sum(1 for m in mapped_items if m.is_mapped)
        unmapped_count = len(mapped_items) - mapped_count
        coverage = (mapped_count / len(mapped_items) * 100) if mapped_items else 0

        return MappingResult(
            mapped_items=mapped_items,
            mapped_count=mapped_count,
            unmapped_count=unmapped_count,
            coverage_percent=round(coverage, 1),
            unmapped_codes=unmapped_codes,
        )

    def export_csv(
        self,
        items: List[MappedBOQItem],
        output_path: Path,
    ) -> Path:
        """
        Export mapped BOQ to CSV with CPWD columns.

        Args:
            items: List of MappedBOQItem
            output_path: Output CSV path

        Returns:
            Path to created file
        """
        headers = [
            "item_code",
            "description",
            "qty",
            "unit",
            "derived_from",
            "confidence",
            "category",
            "cpwd_item_no",
            "cpwd_description",
            "cpwd_unit",
            "is_mapped",
            "mapping_notes",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for item in items:
                writer.writerow([
                    item.item_code,
                    item.description,
                    f"{item.qty:.2f}" if isinstance(item.qty, float) else item.qty,
                    item.unit,
                    item.derived_from,
                    f"{item.confidence:.2f}",
                    item.category or "",
                    item.cpwd_item_no or "",
                    item.cpwd_description or "",
                    item.cpwd_unit or "",
                    "Yes" if item.is_mapped else "No",
                    item.mapping_notes or "",
                ])

        logger.info(f"Exported mapped BOQ to {output_path}")
        return output_path

    def add_mapping(
        self,
        item_code: str,
        cpwd_item_no: str,
        cpwd_description: str,
        cpwd_unit: str,
        rate_inr: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Add a new mapping programmatically."""
        self.mappings[item_code] = CPWDItem(
            item_code=item_code,
            cpwd_item_no=cpwd_item_no,
            cpwd_description=cpwd_description,
            cpwd_unit=cpwd_unit,
            rate_inr=rate_inr,
            notes=notes,
        )


def map_boq_to_cpwd(
    items: List[Any],
    output_path: Optional[Path] = None,
) -> MappingResult:
    """
    Convenience function to map BOQ items to CPWD.

    Args:
        items: List of BOQ items
        output_path: Optional path to export CSV

    Returns:
        MappingResult
    """
    mapper = CPWDMapper()
    result = mapper.map_items(items)

    if output_path:
        mapper.export_csv(result.mapped_items, output_path)

    return result


def get_mapping_coverage(items: List[Any]) -> Dict[str, Any]:
    """
    Get mapping coverage statistics.

    Args:
        items: List of BOQ items

    Returns:
        Dict with coverage stats
    """
    mapper = CPWDMapper()
    result = mapper.map_items(items)

    return {
        "total_items": len(items),
        "mapped": result.mapped_count,
        "unmapped": result.unmapped_count,
        "coverage_percent": result.coverage_percent,
        "unmapped_codes": result.unmapped_codes,
    }
