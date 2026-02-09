"""
BOQ Schema - Standardized BOQ data structures and validators

Canonical BOQ columns:
- item_code: Unique identifier (e.g., FLR-VIT-01)
- description: Full item description
- qty: Quantity (numeric)
- unit: Unit of measurement (sqm, cum, kg, rm, nos)
- derived_from: Source of data (measured, detected, assumption, rule_of_thumb)
- confidence: Confidence score (0.0 to 1.0)
- assumption_used: What assumption was made (if any)
- notes: Additional notes

This module provides:
- BOQItem dataclass
- Validators (never crash, warn and continue)
- Converters for various input formats
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class DerivedFrom(Enum):
    """Source of quantity data."""
    MEASURED = "measured"
    DETECTED = "detected"
    OCR_EXTRACTED = "ocr_extracted"
    SCHEDULE_LOOKUP = "schedule_lookup"
    INFERRED = "inferred"
    RULE_OF_THUMB = "rule_of_thumb"
    ASSUMPTION = "assumption"
    DEFAULT = "default"
    CALCULATION = "calculation"
    USER_INPUT = "user_input"


class BOQCategory(Enum):
    """BOQ item categories."""
    MASONRY = "Masonry"
    FINISHES = "Finishes"
    STRUCTURAL = "Structural"
    STEEL = "Steel"
    OPENINGS = "Openings"
    PLUMBING = "Plumbing"
    ELECTRICAL = "Electrical"
    EXTERNAL = "External"
    MISC = "Miscellaneous"


class Unit(Enum):
    """Standard units."""
    SQM = "sqm"          # Square meters
    CUM = "cum"          # Cubic meters
    RM = "rm"            # Running meters
    KG = "kg"            # Kilograms
    MT = "MT"            # Metric tonnes
    NOS = "nos"          # Numbers/pieces
    SQFT = "sqft"        # Square feet (for display)
    CFT = "cft"          # Cubic feet (for display)
    LITER = "ltr"        # Liters
    LS = "LS"            # Lump sum
    SET = "set"          # Set


@dataclass
class BOQItem:
    """
    Standard BOQ line item.

    This is the canonical format for all BOQ items across modules.
    """
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    assumption_used: Optional[str] = None
    notes: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    room_id: Optional[str] = None
    element_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_code": self.item_code,
            "description": self.description,
            "qty": round(self.qty, 2) if isinstance(self.qty, float) else self.qty,
            "unit": self.unit,
            "derived_from": self.derived_from,
            "confidence": round(self.confidence, 2),
            "assumption_used": self.assumption_used,
            "notes": self.notes,
            "category": self.category,
            "subcategory": self.subcategory,
            "room_id": self.room_id,
            "element_id": self.element_id,
        }

    def to_csv_row(self) -> List[str]:
        """Convert to CSV row."""
        return [
            self.item_code,
            self.description,
            f"{self.qty:.2f}" if isinstance(self.qty, float) else str(self.qty),
            self.unit,
            self.derived_from,
            f"{self.confidence:.2f}",
            self.assumption_used or "",
            self.notes or "",
        ]

    @classmethod
    def csv_headers(cls) -> List[str]:
        """Get CSV headers."""
        return [
            "item_code",
            "description",
            "qty",
            "unit",
            "derived_from",
            "confidence",
            "assumption_used",
            "notes",
        ]


@dataclass
class ValidationResult:
    """Result of BOQ item validation."""
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fixed_item: Optional[BOQItem] = None


class BOQValidator:
    """
    Validate BOQ items - never crash, warn and continue.

    Validates:
    - Required fields present
    - Quantity is positive
    - Unit is valid
    - Confidence is 0-1
    - Item code format
    """

    VALID_UNITS = ["sqm", "cum", "rm", "kg", "MT", "nos", "sqft", "cft", "ltr", "LS", "set", "m", "mm"]

    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate(self, item: Union[BOQItem, Dict]) -> ValidationResult:
        """
        Validate a BOQ item.

        Returns ValidationResult with is_valid, warnings, and optionally fixed item.
        """
        warnings = []
        errors = []

        # Convert dict to BOQItem if needed
        if isinstance(item, dict):
            try:
                item = self._dict_to_item(item)
            except Exception as e:
                errors.append(f"Could not convert dict to BOQItem: {e}")
                return ValidationResult(is_valid=False, errors=errors)

        # Validate item_code
        if not item.item_code:
            warnings.append("Missing item_code - assigned 'UNK-001'")
            item.item_code = "UNK-001"
        elif not isinstance(item.item_code, str):
            item.item_code = str(item.item_code)
            warnings.append(f"Item code converted to string: {item.item_code}")

        # Validate description
        if not item.description:
            warnings.append("Missing description")
            item.description = f"Item {item.item_code}"

        # Validate quantity
        if item.qty is None:
            warnings.append("Missing quantity - set to 0")
            item.qty = 0.0
        elif not isinstance(item.qty, (int, float)):
            try:
                item.qty = float(item.qty)
            except (ValueError, TypeError):
                errors.append(f"Invalid quantity: {item.qty}")
                item.qty = 0.0
        elif item.qty < 0:
            warnings.append(f"Negative quantity: {item.qty} - converted to positive")
            item.qty = abs(item.qty)

        # Validate unit
        if not item.unit:
            warnings.append("Missing unit - default 'nos'")
            item.unit = "nos"
        elif item.unit.lower() not in [u.lower() for u in self.VALID_UNITS]:
            warnings.append(f"Unknown unit: {item.unit}")

        # Validate confidence
        if item.confidence is None:
            warnings.append("Missing confidence - set to 0.5")
            item.confidence = 0.5
        elif not isinstance(item.confidence, (int, float)):
            try:
                item.confidence = float(item.confidence)
            except (ValueError, TypeError):
                warnings.append(f"Invalid confidence: {item.confidence} - set to 0.5")
                item.confidence = 0.5

        if item.confidence < 0 or item.confidence > 1:
            warnings.append(f"Confidence {item.confidence} out of range - clamped to 0-1")
            item.confidence = max(0, min(1, item.confidence))

        # Validate derived_from
        if not item.derived_from:
            warnings.append("Missing derived_from - set to 'unknown'")
            item.derived_from = "unknown"

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            fixed_item=item if is_valid else None,
        )

    def _dict_to_item(self, d: Dict) -> BOQItem:
        """Convert dictionary to BOQItem."""
        return BOQItem(
            item_code=d.get("item_code", ""),
            description=d.get("description", ""),
            qty=d.get("qty", 0),
            unit=d.get("unit", "nos"),
            derived_from=d.get("derived_from", ""),
            confidence=d.get("confidence", 0.5),
            assumption_used=d.get("assumption_used"),
            notes=d.get("notes"),
            category=d.get("category"),
            subcategory=d.get("subcategory"),
            room_id=d.get("room_id"),
            element_id=d.get("element_id"),
        )

    def validate_batch(self, items: List[Union[BOQItem, Dict]]) -> List[ValidationResult]:
        """Validate a batch of items."""
        return [self.validate(item) for item in items]


@dataclass
class BOQPackageSchema:
    """
    Complete BOQ package schema.
    """
    project_id: str
    generated_at: str
    profile: str = "typical"
    items: List[BOQItem] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_item(self, item: BOQItem) -> None:
        """Add validated item."""
        validator = BOQValidator()
        result = validator.validate(item)

        if result.warnings:
            self.warnings.extend(result.warnings)

        if result.fixed_item:
            self.items.append(result.fixed_item)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "generated_at": self.generated_at,
            "profile": self.profile,
            "summary": {
                "total_items": len(self.items),
                "total_assumptions": len(self.assumptions),
                "total_warnings": len(self.warnings),
            },
            "items": [item.to_dict() for item in self.items],
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "totals": self.totals,
            "metadata": self.metadata,
        }


def convert_to_boq_item(
    obj: Any,
    category: Optional[str] = None,
) -> Optional[BOQItem]:
    """
    Convert various item types to BOQItem.

    Handles:
    - Dict
    - Dataclass objects with compatible fields
    - Tuple (item_code, description, qty, unit)
    """
    try:
        if isinstance(obj, BOQItem):
            if category:
                obj.category = category
            return obj

        if isinstance(obj, dict):
            return BOQItem(
                item_code=obj.get("item_code", "UNK-001"),
                description=obj.get("description", ""),
                qty=float(obj.get("qty", 0)),
                unit=obj.get("unit", "nos"),
                derived_from=obj.get("derived_from", "unknown"),
                confidence=float(obj.get("confidence", 0.5)),
                assumption_used=obj.get("assumption_used"),
                notes=obj.get("notes"),
                category=category or obj.get("category"),
                subcategory=obj.get("subcategory"),
                room_id=obj.get("room_id"),
                element_id=obj.get("element_id"),
            )

        # Dataclass or object with attributes
        if hasattr(obj, "__dataclass_fields__") or hasattr(obj, "item_code"):
            return BOQItem(
                item_code=getattr(obj, "item_code", "UNK-001"),
                description=getattr(obj, "description", ""),
                qty=float(getattr(obj, "qty", 0)),
                unit=getattr(obj, "unit", "nos"),
                derived_from=getattr(obj, "derived_from", "unknown"),
                confidence=float(getattr(obj, "confidence", 0.5)),
                assumption_used=getattr(obj, "assumption_used", None),
                notes=getattr(obj, "notes", None),
                category=category or getattr(obj, "category", None),
                subcategory=getattr(obj, "subcategory", None),
                room_id=getattr(obj, "room_id", None),
                element_id=getattr(obj, "element_id", None),
            )

        # Tuple format
        if isinstance(obj, (list, tuple)) and len(obj) >= 4:
            return BOQItem(
                item_code=str(obj[0]),
                description=str(obj[1]),
                qty=float(obj[2]),
                unit=str(obj[3]),
                derived_from=obj[4] if len(obj) > 4 else "unknown",
                confidence=float(obj[5]) if len(obj) > 5 else 0.5,
                category=category,
            )

        logger.warning(f"Could not convert to BOQItem: {type(obj)}")
        return None

    except Exception as e:
        logger.warning(f"Error converting to BOQItem: {e}")
        return None


def merge_boq_items(
    *item_lists: List[Any],
    categories: Optional[List[str]] = None,
) -> List[BOQItem]:
    """
    Merge multiple BOQ item lists into one.

    Args:
        item_lists: Variable number of item lists
        categories: Optional category names for each list

    Returns:
        Merged list of BOQItem objects
    """
    merged = []
    validator = BOQValidator()

    for i, items in enumerate(item_lists):
        category = categories[i] if categories and i < len(categories) else None

        for item in items:
            boq_item = convert_to_boq_item(item, category)
            if boq_item:
                result = validator.validate(boq_item)
                if result.fixed_item:
                    merged.append(result.fixed_item)

    return merged


def load_profile(profile_name: str = "typical") -> Dict[str, Any]:
    """
    Load profile settings from assumptions.yaml.

    Args:
        profile_name: Profile name (conservative, typical, premium)

    Returns:
        Profile configuration dict
    """
    try:
        assumptions_path = Path(__file__).parent.parent.parent / "rules" / "assumptions.yaml"
        with open(assumptions_path, "r") as f:
            config = yaml.safe_load(f)

        profiles = config.get("profiles", {})
        profile = profiles.get(profile_name, profiles.get("typical", {}))

        # Add full config for reference
        profile["_full_config"] = config

        return profile

    except Exception as e:
        logger.warning(f"Could not load profile {profile_name}: {e}")
        return {
            "description": "Default profile",
            "steel_factor_multiplier": 1.0,
            "wastage_factor_multiplier": 1.0,
            "size_defaults": "standard",
            "confidence_threshold": 0.5,
        }
