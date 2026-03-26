"""Room checklist loader."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.knowledge_base.room_checklists.schema import RoomChecklist

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data"


class RoomChecklistLoader:
    def __init__(self):
        self._checklists: List[RoomChecklist] = []
        self._by_room_type: Dict[str, RoomChecklist] = {}
        self._loaded = False

    @property
    def checklist_count(self) -> int:
        return len(self._checklists)

    def load_all(self) -> None:
        if self._loaded:
            return
        if not DATA_DIR.exists():
            logger.warning("Room checklist data directory not found: %s", DATA_DIR)
            self._loaded = True
            return
        for yaml_file in sorted(DATA_DIR.glob("*.yaml")):
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error("Error loading room checklist file %s: %s", yaml_file.name, e)
        self._loaded = True
        logger.info("Room checklists loaded: %d room types", len(self._checklists))

    def _load_file(self, filepath: Path) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return
        for room_data in data.get("rooms", []):
            if isinstance(room_data, dict):
                checklist = RoomChecklist.from_dict(room_data)
                if checklist.room_type:
                    self._checklists.append(checklist)
                    self._by_room_type[checklist.room_type] = checklist

    def all_checklists(self) -> List[RoomChecklist]:
        self.load_all()
        return list(self._checklists)

    def get_by_room_type(self, room_type: str) -> Optional[RoomChecklist]:
        self.load_all()
        return self._by_room_type.get(room_type)

    def match_room_from_keywords(self, text: str) -> List[RoomChecklist]:
        """Find all room checklists whose keywords appear in the given text."""
        self.load_all()
        text_lower = text.lower()
        matches = []
        for checklist in self._checklists:
            if any(kw.lower() in text_lower for kw in checklist.keywords):
                matches.append(checklist)
        return matches

    def check_boq_for_room(
        self, room_type: str, boq_descriptions: List[str]
    ) -> Dict[str, List[str]]:
        """
        Check if required items for a room type are present in BOQ.
        Returns: {"missing": [...], "present": [...]}
        """
        self.load_all()
        checklist = self._by_room_type.get(room_type)
        if not checklist:
            return {"missing": [], "present": []}

        boq_text = " ".join(boq_descriptions).lower()
        missing = []
        present = []
        for item in checklist.required_items:
            item_words = item.lower().replace("_", " ").split()
            if any(word in boq_text for word in item_words):
                present.append(item)
            else:
                missing.append(item)
        return {"missing": missing, "present": present}
