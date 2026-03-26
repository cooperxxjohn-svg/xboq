"""
Taxonomy data models.

Each TaxonomyItem represents a single construction item in the master database.
Items are organized: Discipline → Trade → Sub-trade → Category → Item
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaxonomyItem:
    """A single construction item in the master taxonomy."""

    id: str                                     # "STR.RCC.COL.001"
    standard_name: str                          # "RCC Column M25 Grade"
    discipline: str                             # "structural"
    trade: str                                  # "rcc_concrete"
    sub_trade: str = ""                         # "superstructure"
    category: str = ""                          # "columns"

    # Measurement
    unit: str = ""                              # "cum" (IS 1200 unit)
    measurement_basis: str = ""                 # "IS 1200 Part 7"

    # Applicability
    construction_types: List[str] = field(default_factory=lambda: ["all"])
    building_types: List[str] = field(default_factory=lambda: ["all"])

    # Dependencies (forward references to other taxonomy IDs)
    requires: List[str] = field(default_factory=list)

    # Code references
    is_code_ref: str = ""
    dsr_item_ref: str = ""

    # Aliases (short list for quick matching; full aliases in synonym engine)
    aliases: List[str] = field(default_factory=list)

    # Rates
    typical_rate_range: Optional[Dict[str, float]] = None  # {"min": 5000, "max": 12000}

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "standard_name": self.standard_name,
            "discipline": self.discipline,
            "trade": self.trade,
            "sub_trade": self.sub_trade,
            "category": self.category,
            "unit": self.unit,
            "measurement_basis": self.measurement_basis,
            "construction_types": self.construction_types,
            "building_types": self.building_types,
            "requires": self.requires,
            "is_code_ref": self.is_code_ref,
            "dsr_item_ref": self.dsr_item_ref,
            "aliases": self.aliases,
            "typical_rate_range": self.typical_rate_range,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], discipline: str = "", trade: str = "",
                  sub_trade: str = "", category: str = "") -> "TaxonomyItem":
        """Create TaxonomyItem from a YAML dict, inheriting parent context."""
        return TaxonomyItem(
            id=data.get("id", ""),
            standard_name=data.get("standard_name", ""),
            discipline=data.get("discipline", discipline),
            trade=data.get("trade", trade),
            sub_trade=data.get("sub_trade", sub_trade),
            category=data.get("category", category),
            unit=data.get("unit", ""),
            measurement_basis=data.get("measurement_basis", ""),
            construction_types=data.get("construction_types", ["all"]),
            building_types=data.get("building_types", ["all"]),
            requires=data.get("requires", []),
            is_code_ref=data.get("is_code_ref", ""),
            dsr_item_ref=data.get("dsr_item_ref", ""),
            aliases=data.get("aliases", []),
            typical_rate_range=data.get("typical_rate_range"),
            notes=data.get("notes", ""),
        )
