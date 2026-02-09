"""
BOQ Splitter

Splits BOQ items into measured vs inferred based on provenance.

Rules:
- measured.csv: only lines with method != text_only and confidence >= threshold
- inferred.csv: anything template/implied/paranoia/prelims/provisionals or text_only
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .model import (
    QuantityProvenance,
    ProvenanceMethod,
    ScaleBasis,
    ProvenanceTracker,
    MEASURED_METHODS,
    MEASURED_CONFIDENCE_THRESHOLD,
    create_text_only_provenance,
    create_polygon_provenance,
    create_allowance_provenance,
)


def attach_provenance_to_boq(
    boq_items: List[Dict[str, Any]],
    rooms: List[Dict[str, Any]],
    openings: List[Dict[str, Any]],
    scale_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Attach provenance to each BOQ item based on available evidence.

    Args:
        boq_items: List of BOQ items
        rooms: Extracted room data
        openings: Extracted opening data
        scale_info: Scale determination info

    Returns:
        BOQ items with provenance attached
    """
    # Build room lookup by label
    room_lookup = {}
    for room in rooms:
        label = room.get("label", "").lower().strip()
        if label:
            room_lookup[label] = room

    # Determine scale basis
    scale_basis = ScaleBasis.UNKNOWN
    scale_value = 100  # Default
    scale_confidence = 0.0

    if scale_info.get("scale"):
        scale_value = scale_info.get("scale", 100)
        scale_confidence = scale_info.get("confidence", 0.5)
        if scale_info.get("basis") == "manual":
            scale_basis = ScaleBasis.MANUAL
        elif scale_info.get("basis") == "dimension_inferred":
            scale_basis = ScaleBasis.DIMENSION_INFERRED
        elif scale_info.get("basis") == "scale_note":
            scale_basis = ScaleBasis.SCALE_NOTE

    for item in boq_items:
        prov = _determine_item_provenance(
            item, room_lookup, openings, scale_basis, scale_value, scale_confidence
        )
        item["provenance"] = prov.to_dict()
        item["_provenance_obj"] = prov  # Keep object reference for processing

    return boq_items


def _determine_item_provenance(
    item: Dict[str, Any],
    room_lookup: Dict[str, Dict],
    openings: List[Dict],
    scale_basis: ScaleBasis,
    scale_value: float,
    scale_confidence: float,
) -> QuantityProvenance:
    """Determine provenance for a single BOQ item."""

    package = item.get("package", "").lower()
    description = item.get("description", "").lower()
    room_ref = item.get("room", "").lower().strip()
    qty = item.get("qty", 0)

    # Check if this is an allowance/provisional
    if "allowance" in description or "provisional" in description or "assumed" in description:
        return create_allowance_provenance(
            description=item.get("description", ""),
            allowance_basis="standard",
        )

    # Check if room has geometry
    room_data = room_lookup.get(room_ref, {})
    has_bbox = bool(room_data.get("bbox"))
    has_area = room_data.get("area_sqm", 0) > 0 or room_data.get("area", 0) > 0

    # Determine method based on item type and available evidence
    if package in ["flooring", "wall_finishes", "waterproofing"] and has_area:
        # Area-based items with geometry evidence
        return QuantityProvenance(
            source_pages=[room_data.get("page", 0)],
            source_files=[room_data.get("source_file", "")],
            method=ProvenanceMethod.POLYGON,
            confidence=0.7 if has_bbox else 0.5,
            scale_basis=scale_basis,
            scale_value=scale_value,
            raw_value=qty,
            adjusted_value=qty,
            calculation_notes=f"Area from room '{room_ref}' polygon",
            needs_verification=scale_basis == ScaleBasis.UNKNOWN,
        )

    if package in ["doors_windows", "doors", "windows"]:
        # Check if opening was detected
        matching_openings = [o for o in openings if o.get("room", "").lower() == room_ref]
        if matching_openings:
            return QuantityProvenance(
                source_pages=[o.get("page", 0) for o in matching_openings],
                method=ProvenanceMethod.SYMBOL_COUNT,
                confidence=0.8,
                scale_basis=ScaleBasis.MANUAL,  # Count doesn't need scale
                raw_value=len(matching_openings),
                adjusted_value=qty,
                calculation_notes=f"Counted {len(matching_openings)} openings in '{room_ref}'",
                needs_verification=False,
            )

    # Check if we at least have a room label match
    if room_data:
        return QuantityProvenance(
            source_pages=[room_data.get("page", 0)],
            method=ProvenanceMethod.TEXT_ONLY,
            confidence=0.3,
            scale_basis=ScaleBasis.UNKNOWN,
            calculation_notes=f"Room label '{room_ref}' found, but no geometry measured",
            needs_verification=True,
        )

    # Completely ungrounded item
    return QuantityProvenance(
        method=ProvenanceMethod.INFERRED,
        confidence=0.0,
        scale_basis=ScaleBasis.UNKNOWN,
        calculation_notes="No drawing evidence. Inferred from template.",
        needs_verification=True,
    )


def split_boq_by_provenance(
    boq_items: List[Dict[str, Any]],
    confidence_threshold: float = MEASURED_CONFIDENCE_THRESHOLD,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split BOQ items into measured and inferred lists (legacy 2-bucket).

    Args:
        boq_items: BOQ items with provenance attached
        confidence_threshold: Minimum confidence for measured

    Returns:
        (measured_items, inferred_items)
    """
    measured = []
    inferred = []

    for item in boq_items:
        prov = item.get("_provenance_obj")
        if not prov:
            # Try to load from dict
            prov_dict = item.get("provenance", {})
            if prov_dict:
                prov = QuantityProvenance.from_dict(prov_dict)
            else:
                prov = QuantityProvenance()  # Default to inferred

        # Determine if measured
        is_measured = (
            prov.method in MEASURED_METHODS and
            prov.confidence >= confidence_threshold and
            prov.scale_basis != ScaleBasis.UNKNOWN
        )

        # Clean item for output (remove internal objects)
        clean_item = {k: v for k, v in item.items() if not k.startswith("_")}

        if is_measured:
            measured.append(clean_item)
        else:
            inferred.append(clean_item)

    return measured, inferred


def split_boq_three_buckets(
    boq_items: List[Dict[str, Any]],
    confidence_threshold: float = MEASURED_CONFIDENCE_THRESHOLD,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split BOQ items into three strict buckets:

    1. measured: Geometry-backed items with scale confidence
       - method: POLYGON, CENTERLINE, DIMENSION
       - scale_basis: not UNKNOWN
       - confidence >= threshold

    2. counted: Symbol/text-based counts (no scale needed)
       - method: SYMBOL_COUNT, TEXT_ONLY with countable items
       - Scale not required for counting

    3. inferred: Rule-based, explicit assumptions
       - method: INFERRED, ALLOWANCE, TEMPLATE
       - All provisional/assumed items

    Args:
        boq_items: BOQ items with provenance attached
        confidence_threshold: Minimum confidence for measured

    Returns:
        (measured_items, counted_items, inferred_items)
    """
    measured = []
    counted = []
    inferred = []

    GEOMETRY_METHODS = {ProvenanceMethod.POLYGON, ProvenanceMethod.CENTERLINE, ProvenanceMethod.DIMENSION_TEXT}
    COUNT_METHODS = {ProvenanceMethod.SYMBOL_COUNT}

    for item in boq_items:
        prov = item.get("_provenance_obj")
        if not prov:
            prov_dict = item.get("provenance", {})
            if prov_dict:
                prov = QuantityProvenance.from_dict(prov_dict)
            else:
                prov = QuantityProvenance()

        # Clean item for output
        clean_item = {k: v for k, v in item.items() if not k.startswith("_")}

        # Bucket 1: MEASURED (geometry-backed)
        if (prov.method in GEOMETRY_METHODS and
            prov.confidence >= confidence_threshold and
            prov.scale_basis != ScaleBasis.UNKNOWN):
            clean_item["bucket"] = "measured"
            measured.append(clean_item)

        # Bucket 2: COUNTED (symbol/count-based, no scale needed)
        elif prov.method in COUNT_METHODS:
            clean_item["bucket"] = "counted"
            counted.append(clean_item)

        # Bucket 3: INFERRED (everything else)
        else:
            clean_item["bucket"] = "inferred"
            clean_item["infer_reason"] = prov.calculation_notes or "Rule-based inference"
            inferred.append(clean_item)

    return measured, counted, inferred


def calculate_strict_coverage(
    measured_count: int,
    counted_count: int,
    inferred_count: int,
) -> float:
    """
    Calculate coverage using strict definition:
    coverage = measured_items / measurable_items

    Measurable items = measured + inferred that SHOULD have geometry
    (excludes allowances, provisionals, template items)

    For now, simplified as:
    coverage = measured / (measured + inferred)
    Counted items are excluded (they don't need geometry).

    Args:
        measured_count: Items with geometry backing
        counted_count: Items with symbol counts (excluded from coverage)
        inferred_count: Items needing geometry but missing it

    Returns:
        Coverage percentage (0.0 to 1.0)
    """
    measurable_total = measured_count + inferred_count
    if measurable_total == 0:
        return 0.0
    return measured_count / measurable_total


def write_split_boq_files(
    output_dir: Path,
    boq_items: List[Dict[str, Any]],
    confidence_threshold: float = MEASURED_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Write split BOQ files in 3 strict buckets:
    - measured.json: Geometry-backed items
    - counted.json: Symbol/text-based counts (no scale needed)
    - inferred.json: Rule-based explicit assumptions

    Also writes CSV versions for compatibility.

    Args:
        output_dir: Output directory
        boq_items: BOQ items with provenance
        confidence_threshold: Threshold for measured classification

    Returns:
        Summary stats including strict coverage
    """
    output_dir = Path(output_dir)
    boq_dir = output_dir / "boq"
    boq_dir.mkdir(parents=True, exist_ok=True)

    # Split into 3 buckets
    measured, counted, inferred = split_boq_three_buckets(boq_items, confidence_threshold)

    # Define columns for CSV with provenance
    base_columns = ["item_id", "package", "description", "room", "qty", "unit", "rate", "amount"]
    prov_columns = ["source_pages", "method", "confidence", "scale_basis", "bucket", "geometry_refs", "infer_reason"]
    all_columns = base_columns + prov_columns

    def _item_to_row(item: Dict, bucket: str) -> Dict:
        """Convert item to CSV row."""
        row = {**item}
        prov = item.get("provenance", {})
        row.update({
            "source_pages": ";".join(str(p) for p in prov.get("source_pages", [])),
            "method": prov.get("method", "unknown"),
            "confidence": f"{prov.get('confidence', 0):.2f}",
            "scale_basis": prov.get("scale_basis", "unknown"),
            "bucket": bucket,
            "geometry_refs": ";".join(g.get("ref_id", "") for g in prov.get("geometry_refs", [])),
            "infer_reason": item.get("infer_reason", ""),
        })
        return row

    # Write measured.json and boq_measured.csv
    measured_json_path = boq_dir / "measured.json"
    measured_csv_path = boq_dir / "boq_measured.csv"

    with open(measured_json_path, "w") as f:
        json.dump(measured, f, indent=2, default=str)

    with open(measured_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for item in measured:
            writer.writerow(_item_to_row(item, "measured"))

    # Write counted.json and boq_counted.csv
    counted_json_path = boq_dir / "counted.json"
    counted_csv_path = boq_dir / "boq_counted.csv"

    with open(counted_json_path, "w") as f:
        json.dump(counted, f, indent=2, default=str)

    with open(counted_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for item in counted:
            writer.writerow(_item_to_row(item, "counted"))

    # Write inferred.json and boq_inferred.csv
    inferred_json_path = boq_dir / "inferred.json"
    inferred_csv_path = boq_dir / "boq_inferred.csv"

    with open(inferred_json_path, "w") as f:
        json.dump(inferred, f, indent=2, default=str)

    with open(inferred_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for item in inferred:
            writer.writerow(_item_to_row(item, "inferred"))

    # Update main boq_quantities.csv with all items
    quantities_path = boq_dir / "boq_quantities.csv"
    all_items = measured + counted + inferred
    with open(quantities_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for item in measured:
            writer.writerow(_item_to_row(item, "measured"))
        for item in counted:
            writer.writerow(_item_to_row(item, "counted"))
        for item in inferred:
            writer.writerow(_item_to_row(item, "inferred"))

    # Calculate strict coverage
    strict_coverage = calculate_strict_coverage(len(measured), len(counted), len(inferred))

    return {
        "total_items": len(boq_items),
        "measured_count": len(measured),
        "counted_count": len(counted),
        "inferred_count": len(inferred),
        "measured_file": str(measured_json_path),
        "counted_file": str(counted_json_path),
        "inferred_file": str(inferred_json_path),
        "measurement_coverage": strict_coverage,
        # Flag if measured is empty
        "measured_empty": len(measured) == 0,
        # Coverage = measured / measurable (excludes counted)
        "measurable_items": len(measured) + len(inferred),
    }


def generate_tbd_items(
    inferred_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert inferred items to TBD/ALLOWANCE format.

    Args:
        inferred_items: Items without geometry backing

    Returns:
        Items marked as TBD with explicit notes
    """
    tbd_items = []

    for item in inferred_items:
        tbd_item = {**item}
        tbd_item["qty_status"] = "TBD"
        tbd_item["qty_note"] = "Quantity not measured - awaiting drawings/RFI response"

        prov = item.get("provenance", {})
        if prov.get("method") == "allowance":
            tbd_item["qty_status"] = "ALLOWANCE"
            tbd_item["qty_note"] = "Provisional allowance - verify with actual scope"

        tbd_items.append(tbd_item)

    return tbd_items
