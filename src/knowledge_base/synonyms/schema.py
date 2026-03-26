"""Synonym data models."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SynonymEntry:
    """Maps a canonical construction term to all its aliases across languages."""

    canonical: str                              # "excavation"
    taxonomy_ids: List[str] = field(default_factory=list)  # ["CIV.EW.EXC.001"]

    formal_english: List[str] = field(default_factory=list)
    informal_english: List[str] = field(default_factory=list)
    hindi: List[str] = field(default_factory=list)
    abbreviations: List[str] = field(default_factory=list)
    brand_names: List[str] = field(default_factory=list)

    @property
    def all_aliases(self) -> List[str]:
        """All aliases combined."""
        return (
            self.formal_english
            + self.informal_english
            + self.hindi
            + self.abbreviations
            + self.brand_names
        )

    @staticmethod
    def from_dict(data: dict) -> "SynonymEntry":
        return SynonymEntry(
            canonical=data.get("canonical", ""),
            taxonomy_ids=data.get("taxonomy_ids", []),
            formal_english=data.get("formal_english", []),
            informal_english=data.get("informal_english", []),
            hindi=data.get("hindi", []),
            abbreviations=data.get("abbreviations", []),
            brand_names=data.get("brand_names", []),
        )
