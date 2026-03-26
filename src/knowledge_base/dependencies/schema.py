"""Enhanced dependency rule data models."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnhancedDependencyRule:
    """A dependency rule with building-type and room-type awareness."""

    id: str                                     # "DEP-STR-FTG-001"
    trigger: str                                # "footing"
    required_items: List[str] = field(default_factory=list)
    trade: str = ""                             # "structural"
    priority: int = 2                           # 1=critical, 2=important, 3=good-practice

    # Conditions
    condition: str = ""                         # "wet_area", "multi_storey", etc.
    building_types: List[str] = field(default_factory=lambda: ["all"])
    room_types: List[str] = field(default_factory=list)

    # Quantitative conditions
    min_floors: int = 0
    min_area_sqm: float = 0.0
    min_height_m: float = 0.0
    min_occupancy: int = 0

    # References
    is_code_ref: str = ""
    nbc_ref: str = ""
    note: str = ""

    @staticmethod
    def from_dict(data: dict) -> "EnhancedDependencyRule":
        return EnhancedDependencyRule(
            id=data.get("id", ""),
            trigger=data.get("trigger", ""),
            required_items=data.get("required_items", []),
            trade=data.get("trade", ""),
            priority=data.get("priority", 2),
            condition=data.get("condition", ""),
            building_types=data.get("building_types", ["all"]),
            room_types=data.get("room_types", []),
            min_floors=data.get("min_floors", 0),
            min_area_sqm=data.get("min_area_sqm", 0.0),
            min_height_m=data.get("min_height_m", 0.0),
            min_occupancy=data.get("min_occupancy", 0),
            is_code_ref=data.get("is_code_ref", ""),
            nbc_ref=data.get("nbc_ref", ""),
            note=data.get("note", ""),
        )
