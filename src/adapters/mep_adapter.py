"""
MEP Adapter

Integrates MEP detection and takeoff into the main pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def run_mep_device_detection(
    drawings_dir: Path,
    output_dir: Path,
    rooms: List[Dict[str, Any]] = None,
    scale: float = 100,
) -> Dict[str, Any]:
    """
    Run MEP device detection on drawings.

    Args:
        drawings_dir: Directory with drawing files
        output_dir: Output directory
        rooms: Extracted room data
        scale: Drawing scale

    Returns:
        Detection result with devices and summary
    """
    from src.mep import DeviceDetector, load_device_types

    registry = load_device_types()
    detector = DeviceDetector(registry)

    all_devices = []

    # Load vector text data from extraction
    boq_dir = output_dir / "boq"

    # Try to load extracted text data
    rooms_file = boq_dir / "rooms.json"
    if rooms_file.exists():
        with open(rooms_file) as f:
            rooms_data = json.load(f)
            rooms = rooms_data.get("rooms", [])

    # Process each page's extracted data
    # Look for any page-level extraction data
    for page_file in output_dir.glob("**/page_*.json"):
        try:
            with open(page_file) as f:
                page_data = json.load(f)

            devices = detector.detect_from_plan(
                vector_texts=page_data.get("vector_texts", []),
                vector_paths=page_data.get("vector_paths", []),
                page=page_data.get("page", 0),
                rooms=rooms,
            )
            all_devices.extend(devices)
        except Exception as e:
            logger.warning(f"Error processing {page_file}: {e}")

    # If no page files, try to detect from room/opening text
    if not all_devices and rooms:
        # Create synthetic text items from room labels
        synthetic_texts = []
        for room in rooms:
            label = room.get("label", "")
            bbox = room.get("bbox", [])
            if label and bbox:
                synthetic_texts.append({
                    "text": label,
                    "bbox": bbox,
                })

        devices = detector.detect_from_plan(
            vector_texts=synthetic_texts,
            page=0,
            rooms=rooms,
        )
        all_devices.extend(devices)

    # Save devices
    mep_dir = output_dir / "mep"
    mep_dir.mkdir(parents=True, exist_ok=True)

    devices_file = mep_dir / "devices.json"
    with open(devices_file, "w") as f:
        json.dump({
            "devices": [d.to_dict() for d in all_devices],
            "summary": detector.get_summary(),
        }, f, indent=2)

    return {
        "devices": all_devices,
        "devices_count": len(all_devices),
        "summary": detector.get_summary(),
    }


def run_mep_connectivity(
    output_dir: Path,
    devices: List = None,
    rooms: List[Dict[str, Any]] = None,
    scale: float = 100,
) -> Dict[str, Any]:
    """
    Run connectivity inference for MEP devices.

    Args:
        output_dir: Output directory
        devices: Detected devices (or load from file)
        rooms: Room data
        scale: Drawing scale

    Returns:
        Connectivity result
    """
    from src.mep import ConnectivityInference, DetectedDevice

    mep_dir = output_dir / "mep"

    # Load devices if not provided
    if devices is None:
        devices_file = mep_dir / "devices.json"
        if devices_file.exists():
            with open(devices_file) as f:
                data = json.load(f)
                devices = [DetectedDevice.from_dict(d) for d in data.get("devices", [])]
        else:
            devices = []

    # Load rooms if not provided
    if rooms is None:
        rooms_file = output_dir / "boq" / "rooms.json"
        if rooms_file.exists():
            with open(rooms_file) as f:
                rooms = json.load(f).get("rooms", [])
        else:
            rooms = []

    # Run connectivity inference
    inference = ConnectivityInference(devices, rooms, scale)
    connections = inference.infer_all_connections()

    # Save connections
    conn_file = mep_dir / "connections.json"
    with open(conn_file, "w") as f:
        json.dump({
            "connections": [c.to_dict() for c in connections],
            "cable_summary": inference.get_cable_summary(),
            "pipe_summary": inference.get_pipe_summary(),
        }, f, indent=2)

    return {
        "connections": connections,
        "connections_count": len(connections),
        "cable_summary": inference.get_cable_summary(),
        "pipe_summary": inference.get_pipe_summary(),
    }


def run_mep_systems_grouping(
    output_dir: Path,
    devices: List = None,
) -> Dict[str, Any]:
    """
    Group MEP devices into systems.

    Args:
        output_dir: Output directory
        devices: Detected devices (or load from file)

    Returns:
        Systems grouping result
    """
    from src.mep import SystemGrouper, DetectedDevice

    mep_dir = output_dir / "mep"

    # Load devices if not provided
    if devices is None:
        devices_file = mep_dir / "devices.json"
        if devices_file.exists():
            with open(devices_file) as f:
                data = json.load(f)
                devices = [DetectedDevice.from_dict(d) for d in data.get("devices", [])]
        else:
            devices = []

    # Group by systems
    grouper = SystemGrouper()
    systems = grouper.group_devices(devices)

    # Save systems
    systems_file = mep_dir / "systems.json"
    with open(systems_file, "w") as f:
        json.dump({
            "systems": {k: v.to_dict() for k, v in systems.items()},
            "summary": grouper.get_summary(),
            "hierarchy": grouper.get_system_hierarchy(),
        }, f, indent=2)

    return {
        "systems": systems,
        "summary": grouper.get_summary(),
        "hierarchy": grouper.get_system_hierarchy(),
    }


def run_mep_takeoff(
    output_dir: Path,
    project_id: str = "",
    devices: List = None,
    connections: List = None,
    systems: Dict = None,
) -> Dict[str, Any]:
    """
    Generate MEP takeoff.

    Args:
        output_dir: Output directory
        project_id: Project identifier
        devices: Detected devices
        connections: Inferred connections
        systems: Grouped systems

    Returns:
        Takeoff result
    """
    from src.mep import (
        MEPTakeoff,
        DetectedDevice,
        Connection,
        export_mep_csv,
        export_mep_excel,
    )

    mep_dir = output_dir / "mep"

    # Load data if not provided
    if devices is None:
        devices_file = mep_dir / "devices.json"
        if devices_file.exists():
            with open(devices_file) as f:
                data = json.load(f)
                devices = [DetectedDevice.from_dict(d) for d in data.get("devices", [])]
        else:
            devices = []

    if connections is None:
        conn_file = mep_dir / "connections.json"
        if conn_file.exists():
            with open(conn_file) as f:
                data = json.load(f)
                connections = [Connection(**c) for c in data.get("connections", [])]
        else:
            connections = []

    # Generate takeoff
    takeoff = MEPTakeoff(devices, connections, systems, project_id)
    result = takeoff.generate()

    # Save JSON
    takeoff_file = mep_dir / "mep_takeoff.json"
    with open(takeoff_file, "w") as f:
        json.dump(result, f, indent=2)

    # Export CSV
    csv_path = mep_dir / "mep_takeoff.csv"
    export_mep_csv(result, csv_path)

    # Try Excel export
    try:
        excel_path = mep_dir / "mep_takeoff.xlsx"
        export_mep_excel(result, excel_path)
        result["excel_path"] = str(excel_path)
    except Exception as e:
        logger.warning(f"Excel export failed: {e}")

    result["csv_path"] = str(csv_path)
    result["json_path"] = str(takeoff_file)

    return result


def run_full_mep_pipeline(
    drawings_dir: Path,
    output_dir: Path,
    project_id: str = "",
    rooms: List[Dict[str, Any]] = None,
    scale: float = 100,
) -> Dict[str, Any]:
    """
    Run complete MEP pipeline: detection -> connectivity -> systems -> takeoff.

    Args:
        drawings_dir: Directory with drawings
        output_dir: Output directory
        project_id: Project identifier
        rooms: Room data
        scale: Drawing scale

    Returns:
        Complete MEP result
    """
    result = {
        "success": True,
        "detection": None,
        "connectivity": None,
        "systems": None,
        "takeoff": None,
    }

    try:
        # Step 1: Device detection
        detection = run_mep_device_detection(drawings_dir, output_dir, rooms, scale)
        result["detection"] = {
            "devices_count": detection["devices_count"],
            "summary": detection["summary"],
        }

        devices = detection["devices"]

        if not devices:
            result["message"] = "No MEP devices detected"
            return result

        # Step 2: Connectivity inference
        connectivity = run_mep_connectivity(output_dir, devices, rooms, scale)
        result["connectivity"] = {
            "connections_count": connectivity["connections_count"],
            "cable_summary": connectivity["cable_summary"],
            "pipe_summary": connectivity["pipe_summary"],
        }

        connections = connectivity["connections"]

        # Step 3: Systems grouping
        systems_result = run_mep_systems_grouping(output_dir, devices)
        result["systems"] = systems_result["summary"]

        systems = systems_result["systems"]

        # Step 4: Generate takeoff
        takeoff = run_mep_takeoff(output_dir, project_id, devices, connections, systems)
        result["takeoff"] = {
            "total_lines": takeoff["total_lines"],
            "total_devices": takeoff["total_devices"],
            "measured_count": takeoff["measured_count"],
            "inferred_count": takeoff["inferred_count"],
            "rfi_count": takeoff["rfi_count"],
            "csv_path": takeoff.get("csv_path"),
            "excel_path": takeoff.get("excel_path"),
        }

        result["message"] = f"MEP pipeline complete: {takeoff['total_devices']} devices, {connectivity['connections_count']} connections, {takeoff['rfi_count']} RFIs"

    except Exception as e:
        logger.error(f"MEP pipeline error: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result
