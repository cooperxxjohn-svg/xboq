"""Room-type completeness checklist knowledge base module.

Provides structured checklists for each room type so the RFI engine
can detect BOQ scope gaps at the room level.
"""

from src.knowledge_base.room_checklists.schema import RoomChecklist, ConditionalItem
from src.knowledge_base.room_checklists.loader import RoomChecklistLoader

__all__ = ["RoomChecklist", "ConditionalItem", "RoomChecklistLoader"]
