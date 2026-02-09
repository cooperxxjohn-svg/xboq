"""
Systems Grouping

Groups MEP devices into logical systems and subsystems.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging

from .device_detector import DetectedDevice
from .connectivity import Connection
from .device_types import DeviceTypeRegistry, load_device_types

logger = logging.getLogger(__name__)


@dataclass
class MEPSystem:
    """A logical MEP system."""
    id: str
    name: str
    category: str  # electrical, plumbing, hvac, fire_safety
    parent_system: Optional[str] = None

    # Devices in this system
    devices: List[DetectedDevice] = field(default_factory=list)
    device_count: int = 0

    # Subsystems
    subsystems: Dict[str, "MEPSystem"] = field(default_factory=dict)

    # Summary stats
    total_devices: int = 0
    devices_with_spec: int = 0
    devices_needing_rfi: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "parent_system": self.parent_system,
            "device_count": self.device_count,
            "total_devices": self.total_devices,
            "devices_with_spec": self.devices_with_spec,
            "devices_needing_rfi": self.devices_needing_rfi,
            "subsystems": {k: v.to_dict() for k, v in self.subsystems.items()},
            "device_ids": [d.id for d in self.devices],
        }

    def add_device(self, device: DetectedDevice):
        """Add a device to this system."""
        self.devices.append(device)
        self.device_count = len(self.devices)
        self._update_stats()

    def _update_stats(self):
        """Update summary statistics."""
        self.total_devices = len(self.devices)
        self.devices_with_spec = sum(
            1 for d in self.devices
            if d.spec_source != "default" or not d.rfi_needed
        )
        self.devices_needing_rfi = sum(1 for d in self.devices if d.rfi_needed)


class SystemGrouper:
    """
    Groups devices into logical MEP systems.
    """

    def __init__(self, registry: DeviceTypeRegistry = None):
        self.registry = registry or load_device_types()
        self.systems: Dict[str, MEPSystem] = {}

    def group_devices(self, devices: List[DetectedDevice]) -> Dict[str, MEPSystem]:
        """
        Group devices into systems.

        Args:
            devices: List of detected devices

        Returns:
            Dictionary of system_id -> MEPSystem
        """
        # Initialize systems from registry hierarchy
        self._initialize_systems()

        # Assign devices to systems
        for device in devices:
            system_id = device.system
            subsystem_id = device.subsystem

            if not system_id:
                system_id = "unclassified"

            # Create system if needed
            if system_id not in self.systems:
                self.systems[system_id] = MEPSystem(
                    id=system_id,
                    name=system_id.replace("_", " ").title(),
                    category=device.category,
                )

            system = self.systems[system_id]

            # Add to subsystem if specified
            if subsystem_id:
                if subsystem_id not in system.subsystems:
                    system.subsystems[subsystem_id] = MEPSystem(
                        id=subsystem_id,
                        name=subsystem_id.replace("_", " ").title(),
                        category=device.category,
                        parent_system=system_id,
                    )

                system.subsystems[subsystem_id].add_device(device)
            else:
                system.add_device(device)

        # Update system totals to include subsystems
        self._update_system_totals()

        return self.systems

    def _initialize_systems(self):
        """Initialize systems from registry hierarchy."""
        systems_config = self.registry.systems_hierarchy

        for system_id, config in systems_config.items():
            if not isinstance(config, dict):
                continue

            self.systems[system_id] = MEPSystem(
                id=system_id,
                name=config.get("name", system_id.replace("_", " ").title()),
                category=self._infer_category(system_id),
                parent_system=config.get("parent"),
            )

            # Initialize subsystems
            for subsystem_id in config.get("subsystems", []):
                self.systems[system_id].subsystems[subsystem_id] = MEPSystem(
                    id=subsystem_id,
                    name=subsystem_id.replace("_", " ").title(),
                    category=self._infer_category(system_id),
                    parent_system=system_id,
                )

    def _infer_category(self, system_id: str) -> str:
        """Infer category from system ID."""
        if system_id in ["lighting", "power", "electrical", "distribution"]:
            return "electrical"
        elif system_id in ["sanitary", "water_supply", "drainage"]:
            return "plumbing"
        elif system_id in ["hvac", "ventilation"]:
            return "hvac"
        elif system_id in ["fire_alarm", "fire_suppression"]:
            return "fire_safety"
        return "general"

    def _update_system_totals(self):
        """Update system totals to include subsystem counts."""
        for system in self.systems.values():
            subsystem_total = sum(
                sub.total_devices for sub in system.subsystems.values()
            )
            system.total_devices = system.device_count + subsystem_total
            system.devices_with_spec = system.devices_with_spec + sum(
                sub.devices_with_spec for sub in system.subsystems.values()
            )
            system.devices_needing_rfi = system.devices_needing_rfi + sum(
                sub.devices_needing_rfi for sub in system.subsystems.values()
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get systems summary."""
        summary = {
            "total_systems": len(self.systems),
            "systems": {},
        }

        for system_id, system in self.systems.items():
            if system.total_devices > 0:  # Only include systems with devices
                summary["systems"][system_id] = {
                    "name": system.name,
                    "category": system.category,
                    "total_devices": system.total_devices,
                    "devices_with_spec": system.devices_with_spec,
                    "devices_needing_rfi": system.devices_needing_rfi,
                    "subsystems": {
                        sub_id: {
                            "name": sub.name,
                            "device_count": sub.device_count,
                        }
                        for sub_id, sub in system.subsystems.items()
                        if sub.device_count > 0
                    },
                }

        return summary

    def get_system_hierarchy(self) -> Dict[str, Any]:
        """Get hierarchical view of systems."""
        hierarchy = {}

        # Group by category
        for system_id, system in self.systems.items():
            if system.total_devices == 0:
                continue

            category = system.category
            if category not in hierarchy:
                hierarchy[category] = {}

            hierarchy[category][system_id] = {
                "name": system.name,
                "devices": system.device_count,
                "subsystems": {
                    sub_id: sub.device_count
                    for sub_id, sub in system.subsystems.items()
                    if sub.device_count > 0
                },
            }

        return hierarchy


def group_by_systems(
    devices: List[DetectedDevice],
    registry: DeviceTypeRegistry = None,
) -> Tuple[Dict[str, MEPSystem], Dict[str, Any]]:
    """
    Convenience function to group devices by systems.

    Returns:
        (systems dict, summary dict)
    """
    grouper = SystemGrouper(registry)
    systems = grouper.group_devices(devices)
    summary = grouper.get_summary()

    return systems, summary
