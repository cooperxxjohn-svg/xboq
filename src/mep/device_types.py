"""
Device Types Registry

Loads and manages device type definitions from rules/device_types.yaml.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceType:
    """Definition of a device type."""
    id: str  # e.g., "ceiling_light"
    category: str  # e.g., "electrical"
    subcategory: str  # e.g., "lights"
    name: str  # Human-readable name

    # Detection patterns
    symbol_patterns: List[str] = field(default_factory=list)
    text_patterns: List[str] = field(default_factory=list)
    ocr_patterns: List[str] = field(default_factory=list)

    # Compiled regex patterns
    _compiled_ocr: List[re.Pattern] = field(default_factory=list, repr=False)

    # Default spec
    default_spec: Dict[str, Any] = field(default_factory=dict)
    unit: str = "NO"

    # System classification
    system: str = ""
    subsystem: str = ""

    # RFI triggers
    rfi_if_missing: List[str] = field(default_factory=list)

    # Special flags
    is_panel: bool = False
    is_equipment: bool = False

    def __post_init__(self):
        """Compile OCR patterns."""
        self._compiled_ocr = []
        for pattern in self.ocr_patterns:
            try:
                self._compiled_ocr.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid OCR pattern '{pattern}': {e}")

    def matches_text(self, text: str) -> bool:
        """Check if text matches any pattern."""
        text_upper = text.upper().strip()

        # Check text patterns
        for pattern in self.text_patterns:
            if pattern.upper() in text_upper or text_upper == pattern.upper():
                return True

        # Check OCR regex patterns
        for compiled in self._compiled_ocr:
            if compiled.search(text):
                return True

        return False

    def matches_symbol(self, symbol: str) -> bool:
        """Check if symbol matches any pattern."""
        return symbol in self.symbol_patterns

    def get_confidence(self, match_type: str) -> float:
        """Get confidence score based on match type."""
        if match_type == "exact_text":
            return 0.9
        elif match_type == "ocr_pattern":
            return 0.8
        elif match_type == "symbol":
            return 0.7
        elif match_type == "partial_text":
            return 0.5
        return 0.3


class DeviceTypeRegistry:
    """
    Registry of all device types loaded from configuration.
    """

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "rules" / "device_types.yaml"
        self.device_types: Dict[str, DeviceType] = {}
        self.by_category: Dict[str, Dict[str, DeviceType]] = {}
        self.by_system: Dict[str, List[DeviceType]] = {}
        self.connectivity_rules: Dict[str, Any] = {}
        self.systems_hierarchy: Dict[str, Any] = {}

        self._load_config()

    def _load_config(self):
        """Load device types from YAML config."""
        if not self.config_path.exists():
            logger.warning(f"Device types config not found: {self.config_path}")
            return

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Parse each category
        for category in ["electrical", "plumbing", "hvac", "fire_safety"]:
            if category not in config:
                continue

            self.by_category[category] = {}

            for subcategory, devices in config[category].items():
                if not isinstance(devices, dict):
                    continue

                for device_id, device_config in devices.items():
                    if not isinstance(device_config, dict):
                        continue

                    device_type = self._parse_device_type(
                        device_id, category, subcategory, device_config
                    )
                    self.device_types[device_id] = device_type
                    self.by_category[category][device_id] = device_type

                    # Index by system
                    system = device_type.system
                    if system:
                        if system not in self.by_system:
                            self.by_system[system] = []
                        self.by_system[system].append(device_type)

        # Load connectivity rules
        self.connectivity_rules = config.get("connectivity", {})

        # Load systems hierarchy
        self.systems_hierarchy = config.get("systems", {})

        logger.info(f"Loaded {len(self.device_types)} device types from {self.config_path}")

    def _parse_device_type(
        self,
        device_id: str,
        category: str,
        subcategory: str,
        config: Dict[str, Any]
    ) -> DeviceType:
        """Parse a single device type from config."""
        patterns = config.get("patterns", {})

        return DeviceType(
            id=device_id,
            category=category,
            subcategory=subcategory,
            name=device_id.replace("_", " ").title(),
            symbol_patterns=patterns.get("symbols", []),
            text_patterns=patterns.get("text", []),
            ocr_patterns=patterns.get("ocr_patterns", []),
            default_spec=config.get("default_spec", {}),
            unit=config.get("unit", "NO"),
            system=config.get("system", ""),
            subsystem=config.get("subsystem", ""),
            rfi_if_missing=config.get("rfi_if_missing", []),
            is_panel=config.get("is_panel", False),
            is_equipment=config.get("is_equipment", False),
        )

    def get(self, device_id: str) -> Optional[DeviceType]:
        """Get device type by ID."""
        return self.device_types.get(device_id)

    def get_all(self) -> List[DeviceType]:
        """Get all device types."""
        return list(self.device_types.values())

    def get_by_category(self, category: str) -> Dict[str, DeviceType]:
        """Get all device types in a category."""
        return self.by_category.get(category, {})

    def get_by_system(self, system: str) -> List[DeviceType]:
        """Get all device types in a system."""
        return self.by_system.get(system, [])

    def get_panels(self) -> List[DeviceType]:
        """Get all panel/distribution device types."""
        return [dt for dt in self.device_types.values() if dt.is_panel]

    def get_equipment(self) -> List[DeviceType]:
        """Get all equipment device types."""
        return [dt for dt in self.device_types.values() if dt.is_equipment]

    def match_text(self, text: str) -> List[tuple]:
        """
        Find all device types matching given text.

        Returns list of (DeviceType, confidence) tuples.
        """
        matches = []
        text = text.strip()

        if not text:
            return matches

        for dt in self.device_types.values():
            if dt.matches_text(text):
                # Determine match type for confidence
                text_upper = text.upper()
                if text_upper in [p.upper() for p in dt.text_patterns]:
                    confidence = dt.get_confidence("exact_text")
                elif any(c.search(text) for c in dt._compiled_ocr):
                    confidence = dt.get_confidence("ocr_pattern")
                else:
                    confidence = dt.get_confidence("partial_text")

                matches.append((dt, confidence))

        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def match_symbol(self, symbol: str) -> List[tuple]:
        """
        Find all device types matching given symbol.

        Returns list of (DeviceType, confidence) tuples.
        """
        matches = []

        for dt in self.device_types.values():
            if dt.matches_symbol(symbol):
                confidence = dt.get_confidence("symbol")
                matches.append((dt, confidence))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_connectivity_rules(self, category: str) -> Dict[str, Any]:
        """Get connectivity rules for a category."""
        return self.connectivity_rules.get(category, {})

    def get_system_info(self, system_id: str) -> Dict[str, Any]:
        """Get system hierarchy info."""
        return self.systems_hierarchy.get(system_id, {})


# Global registry instance
_registry: Optional[DeviceTypeRegistry] = None


def load_device_types(config_path: Path = None) -> DeviceTypeRegistry:
    """Load or get the global device types registry."""
    global _registry

    if _registry is None or config_path is not None:
        _registry = DeviceTypeRegistry(config_path)

    return _registry
