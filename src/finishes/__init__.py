"""
XBOQ Finishes Takeoff Module
Calculates finish quantities for Indian residential projects.

Outputs:
- Floor finish area by room
- Skirting length by room
- Wall paint area by room
- Ceiling paint area by room
- BOQ-ready CSV exports
"""

from .calculator import FinishCalculator, RoomFinishes
from .export import FinishExporter

__all__ = [
    "FinishCalculator",
    "RoomFinishes",
    "FinishExporter",
]
