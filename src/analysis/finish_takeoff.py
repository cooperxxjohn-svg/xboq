"""
Finish Takeoff v1 — aggregate finish totals by finish type from finish schedules.

If room areas are present in schedule fields, compute area totals.
If not, output "areas missing" and list rooms lacking area data.

Pure module, no Streamlit dependency. Can be tested independently.
"""

import csv
import io
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

_FINISH_SCHEDULE_TYPES = {"finish", "finish_schedule", "finishes", "room_finish"}

# Finish type keys to look for in schedule fields
_FINISH_TYPE_KEYS = ["floor", "wall", "ceiling", "dado", "skirting"]

# Patterns to detect area values
_AREA_RE = re.compile(r'(\d+\.?\d*)\s*(sqm|sq\.?\s*m|m2|sq\.?\s*ft|sqft)?', re.IGNORECASE)


def _try_parse_area(val: Any) -> Optional[float]:
    """Try to parse an area value from a field value."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = _AREA_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            return None
    # Try direct float conversion
    try:
        f = float(s)
        if f > 0:
            return f
    except (ValueError, TypeError):
        pass
    return None


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_finish_takeoff(
    schedules: List[dict],
) -> dict:
    """
    Build finish takeoff from schedule rows where schedule_type matches finish types.

    Args:
        schedules: List of schedule row dicts.

    Returns:
        Dict with:
            has_areas: bool — whether any room has area data
            finish_rows: list of finish aggregation rows
            rooms_missing_area: list of room names/marks lacking area
            summary: human-readable 1-liner
    """
    if not schedules:
        return {
            "has_areas": False,
            "finish_rows": [],
            "rooms_missing_area": [],
            "summary": "No schedules provided.",
        }

    # Filter for finish schedules
    finish_rows_raw = []
    for s in schedules:
        if not isinstance(s, dict):
            continue
        stype = (s.get("schedule_type", "") or "").lower().strip()
        if stype in _FINISH_SCHEDULE_TYPES or "finish" in stype:
            finish_rows_raw.append(s)

    if not finish_rows_raw:
        return {
            "has_areas": False,
            "finish_rows": [],
            "rooms_missing_area": [],
            "summary": "No finish schedules detected.",
        }

    # Check for area data + aggregate
    has_areas = False
    rooms_missing_area = []

    # Group by finish type
    # Each finish schedule row may have multiple finish types (floor, wall, ceiling, etc.)
    # We aggregate by finish_type → material
    by_finish: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "total_area": 0.0,
        "rooms": [],
        "pages": set(),
    }))

    for row in finish_rows_raw:
        fields = row.get("fields", {}) or {}
        room = fields.get("room") or row.get("mark") or "Unknown"
        source_page = row.get("source_page")

        # Check for area
        area_val = None
        for area_key in ("area", "Area", "AREA", "room_area", "sqm"):
            if area_key in fields:
                area_val = _try_parse_area(fields[area_key])
                if area_val is not None:
                    break

        if area_val is not None:
            has_areas = True
        else:
            rooms_missing_area.append(str(room))

        # Process each finish type
        for ftype in _FINISH_TYPE_KEYS:
            material = fields.get(ftype)
            if material and str(material).strip() and str(material).strip() != "-":
                material_str = str(material).strip()[:60]
                agg = by_finish[ftype][material_str]
                if area_val is not None:
                    agg["total_area"] += area_val
                agg["rooms"].append(str(room))
                if source_page is not None:
                    agg["pages"].add(source_page)

    # Build finish_rows output
    finish_rows = []
    for ftype in _FINISH_TYPE_KEYS:
        for material, agg in by_finish[ftype].items():
            finish_rows.append({
                "finish_type": ftype,
                "material": material,
                "total_area_sqm": round(agg["total_area"], 2) if has_areas else None,
                "room_count": len(agg["rooms"]),
                "rooms": sorted(set(agg["rooms"]))[:20],
                "evidence_refs": [{"doc_id": None, "page": p} for p in sorted(agg["pages"])[:10]],
            })

    # Deduplicate rooms_missing_area
    rooms_missing_area = sorted(set(rooms_missing_area))

    # Summary
    if has_areas:
        total_area = sum(r["total_area_sqm"] for r in finish_rows if r["total_area_sqm"])
        summary = (
            f"{len(finish_rows)} finish entries across {len(finish_rows_raw)} rooms, "
            f"total area: {total_area:.1f} sqm"
        )
    elif finish_rows:
        summary = (
            f"{len(finish_rows)} finish entries across {len(finish_rows_raw)} rooms — "
            f"areas missing for {len(rooms_missing_area)} room(s)"
        )
    else:
        summary = f"{len(finish_rows_raw)} finish schedule rows but no typed finishes extracted."

    return {
        "has_areas": has_areas,
        "finish_rows": finish_rows,
        "rooms_missing_area": rooms_missing_area,
        "summary": summary,
    }


# =============================================================================
# EXPORT
# =============================================================================

def export_finishes_csv(takeoff: dict) -> str:
    """Export finishes_takeoff.csv from takeoff result."""
    buf = io.StringIO()
    fieldnames = ["finish_type", "material", "total_area_sqm", "room_count", "rooms"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in takeoff.get("finish_rows", []):
        writer.writerow({
            "finish_type": row.get("finish_type", ""),
            "material": row.get("material", ""),
            "total_area_sqm": row.get("total_area_sqm") if row.get("total_area_sqm") is not None else "",
            "room_count": row.get("room_count", 0),
            "rooms": "; ".join(row.get("rooms", [])),
        })
    return buf.getvalue()
