"""
MEP (Mechanical, Electrical, Plumbing) Module

Device detection, connectivity inference, and MEP takeoff for construction drawings.
"""

from .device_detector import (
    DeviceDetector,
    DetectedDevice,
    detect_devices_in_plan,
)

from .device_types import (
    DeviceTypeRegistry,
    DeviceType,
    load_device_types,
)

from .connectivity import (
    ConnectivityInference,
    Connection,
    infer_connections,
)

from .systems import (
    SystemGrouper,
    MEPSystem,
    group_by_systems,
)

from .mep_takeoff import (
    MEPTakeoff,
    generate_mep_takeoff,
    export_mep_csv,
    export_mep_excel,
)

__all__ = [
    # Device Detection
    "DeviceDetector",
    "DetectedDevice",
    "detect_devices_in_plan",
    # Device Types
    "DeviceTypeRegistry",
    "DeviceType",
    "load_device_types",
    # Connectivity
    "ConnectivityInference",
    "Connection",
    "infer_connections",
    # Systems
    "SystemGrouper",
    "MEPSystem",
    "group_by_systems",
    # Takeoff
    "MEPTakeoff",
    "generate_mep_takeoff",
    "export_mep_csv",
    "export_mep_excel",
]
