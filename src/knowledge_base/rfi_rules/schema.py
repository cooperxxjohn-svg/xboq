"""RFI rule data models."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RFIRule:
    """A rule that generates an RFI when a scope condition is met."""

    id: str                                     # "RFI-SG-WP-001"
    trigger_type: str = ""                      # "scope_gap" | "missing_spec" | "missing_doc" | "ambiguous"
    trigger_keywords: List[str] = field(default_factory=list)
    trigger_trade: str = ""

    question_template: str = ""
    why_it_matters: str = ""
    suggested_resolution: str = ""

    priority: str = "medium"                    # "critical", "high", "medium", "low"
    building_types: List[str] = field(default_factory=lambda: ["all"])
    package: str = ""

    @staticmethod
    def from_dict(data: dict) -> "RFIRule":
        return RFIRule(
            id=data.get("id", ""),
            trigger_type=data.get("trigger_type", "scope_gap"),
            trigger_keywords=data.get("trigger_keywords", []),
            trigger_trade=data.get("trigger_trade", ""),
            question_template=data.get("question_template", ""),
            why_it_matters=data.get("why_it_matters", ""),
            suggested_resolution=data.get("suggested_resolution", ""),
            priority=data.get("priority", "medium"),
            building_types=data.get("building_types", ["all"]),
            package=data.get("package", ""),
        )
