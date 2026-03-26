"""Room-type completeness checklist schema."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ConditionalItem:
    item: str
    condition: str  # e.g. "wet_room", "accessible", "hospital_grade"


@dataclass
class RoomChecklist:
    room_type: str          # "toilet", "kitchen", "ot", "server_room"
    display_name: str       # "Toilet / Bathroom"
    keywords: List[str]     # BOQ keywords that indicate this room is present
    required_items: List[str]    # Items that MUST appear if this room type is present
    conditional_items: List[ConditionalItem]  # Items required only under conditions
    common_omissions: List[str]  # Items most commonly forgotten
    rfi_template: str       # Template for the RFI question
    priority: str           # "critical", "high", "medium"
    building_types: List[str]  # ["all"] or specific list

    @classmethod
    def from_dict(cls, data: dict) -> "RoomChecklist":
        conditional = [
            ConditionalItem(item=c.get("item", ""), condition=c.get("if", ""))
            for c in data.get("conditional_items", [])
        ]
        return cls(
            room_type=data.get("room_type", ""),
            display_name=data.get("display_name", ""),
            keywords=data.get("keywords", []),
            required_items=data.get("required_items", []),
            conditional_items=conditional,
            common_omissions=data.get("common_omissions", []),
            rfi_template=data.get("rfi_template", ""),
            priority=data.get("priority", "medium"),
            building_types=data.get("building_types", ["all"]),
        )
