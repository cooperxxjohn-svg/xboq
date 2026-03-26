"""
Visual Measurement — AI reads dimension annotations and room labels from
floor plan images to produce a measured room schedule and QTO.

Unlike visual_element_detector.py which *counts* elements, this module:
  1. Reads the drawing scale (scale bar or "Scale: 1:100" notation)
  2. Reads dimension annotations ("3000", "4500" on dimension lines in mm)
  3. Reads each room's label ("MASTER BEDROOM", "KITCHEN")
  4. Computes room area from the annotated dimensions
  5. Produces a RoomData-compatible room schedule (input to finish_takeoff.py)
  6. Generates measured BOQ line items for flooring / wall finishes

This replaces manual estimator room measurement from drawings.

Output compatible with finish_takeoff.RoomData so measured rooms slot directly
into the existing QTO pipeline without any schema changes.

Dependencies:
  - fitz (PyMuPDF)  — PDF → image
  - Pillow          — image resize + JPEG encode
  - llm_client      — OpenAI (gpt-4o) or Anthropic (claude-opus-4-5) with vision

Cost control:
  - Only pages classified as floor_plan / drawing / plan are scanned
  - Max 6 pages by default (configurable via XBOQ_VMEAS_MAX_PAGES)
  - Images resized to ≤ 1800px longest side before sending
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Reuse the shared PDF renderer and LLM caller from visual_element_detector
try:
    from .visual_element_detector import _render_page_to_bytes, _call_vision_llm
    _HAS_RENDERER = True
except ImportError:
    _HAS_RENDERER = False

# RoomData from finish_takeoff — we produce compatible objects
try:
    from .finish_takeoff import RoomData
    _HAS_ROOM_DATA = True
except ImportError:
    # Fallback stub so module loads even without finish_takeoff
    from dataclasses import dataclass as _dc

    @_dc
    class RoomData:  # type: ignore[no-redef]
        name: str
        raw_name: str
        area_sqm: Optional[float]
        dim_l: Optional[float]
        dim_w: Optional[float]
        source_page: int
        confidence: float

    _HAS_ROOM_DATA = False


# =============================================================================
# CONSTANTS
# =============================================================================

_MAX_PAGES_TO_SCAN = int(os.environ.get("XBOQ_VMEAS_MAX_PAGES", "6"))
_IMAGE_MAX_PX      = int(os.environ.get("XBOQ_VMEAS_MAX_PX", "1800"))
_JPEG_QUALITY      = int(os.environ.get("XBOQ_VMEAS_JPEG_QUALITY", "85"))

_OPENAI_MODEL    = os.environ.get("XBOQ_OPENAI_MODEL",    "gpt-4o")
_ANTHROPIC_MODEL = os.environ.get("XBOQ_ANTHROPIC_MODEL", "claude-opus-4-5")

_DRAWING_TYPES = frozenset((
    "floor_plan", "plan", "drawing", "architectural", "site_plan",
    "structural", "layout", "unknown",
))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MeasuredRoom:
    """One room measured from a drawing page."""
    name: str               # normalised: "MASTER BEDROOM"
    raw_name: str           # as the model read it from the drawing
    length_m: float         # longer dimension, in metres
    width_m: float          # shorter dimension, in metres
    area_sqm: float         # length × width (or from annotation if irregular)
    shape: str              # "rectangular" | "L-shaped" | "irregular"
    dimensions_raw: str     # "4500 × 3000" as annotated on drawing
    confidence: float       # 0.0–1.0
    source_page: int


@dataclass
class MeasuredDrawing:
    """Measurements extracted from one drawing page."""
    source_page: int
    scale: str              # "1:100"
    scale_ratio: int        # 100
    rooms: List[MeasuredRoom] = field(default_factory=list)
    total_area_sqm: float = 0.0
    walls_detected: bool = False
    dimensions_detected: bool = False


@dataclass
class VisualMeasurementResult:
    """Aggregated result from all pages."""
    measured_drawings: List[MeasuredDrawing] = field(default_factory=list)
    all_rooms: List[MeasuredRoom] = field(default_factory=list)
    room_schedule: List[RoomData] = field(default_factory=list)   # pipeline-compatible
    line_items: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mode: str = "none"           # "vision_measurement" | "none"
    pages_scanned: int = 0
    total_area_sqm: float = 0.0
    detected_scale: str = ""
    scale_ratio: int = 0         # authoritative scale ratio (from scale_detector or LLM)
    px_per_mm: float = 0.0       # from scale_detector; 0 if unknown
    scale_source: str = ""       # "scale_detector" | "llm" | "none"


# =============================================================================
# VISION PROMPT — measurement-specific
# =============================================================================

_MEASUREMENT_SYSTEM_PROMPT = """You are an expert Indian construction quantity surveyor
reading architectural floor plan drawings. Your ONLY task is to extract precise measurements.

RETURN ONLY valid JSON using this exact schema — no markdown, no text outside the JSON:

{
  "scale": "1:100",
  "scale_ratio": 100,
  "walls_detected": true,
  "dimensions_detected": true,
  "total_area_sqm": 185.5,
  "rooms": [
    {
      "name": "MASTER BEDROOM",
      "shape": "rectangular",
      "length_m": 4.50,
      "width_m": 3.60,
      "area_sqm": 16.20,
      "dimensions_raw": "4500 × 3600",
      "confidence": 0.92
    }
  ]
}

EXTRACTION RULES:
1. SCALE — read "Scale 1:100", "Scale 1:50" in title block or drawing.
   If a scale bar is visible, note it in "scale". If not visible, use "".
2. DIMENSIONS — Indian drawings annotate in MILLIMETRES (3000 = 3.0 m, 4500 = 4.5 m).
   Divide by 1000 to convert to metres. Western drawings use metres directly.
   Look for dimension lines (thin lines with arrows/ticks and numbers).
3. ROOM NAMES — read labels inside each enclosed space exactly.
   Common Indian labels: BEDROOM, LIVING ROOM, DINING, KITCHEN, TOILET, BATHROOM,
   POOJA, LOBBY, PASSAGE, VERANDAH, BALCONY, STORE, SERVANT, STUDY, PRAYER ROOM.
4. AREA — compute length_m × width_m for rectangular rooms.
   For L-shaped or irregular rooms, estimate from visible partial dimensions.
5. total_area_sqm — sum of all room areas, OR read from title block if stated.
6. CONFIDENCE — 0.95 if dimensions clearly visible, 0.75 if inferred, 0.50 if estimated.
7. WALLS — set walls_detected:true if room outlines are visible.
8. Include ALL labelled rooms. Do NOT invent rooms not shown.
9. If no rooms visible, return empty rooms array.
10. shape: "rectangular" for 4 sides, "L-shaped" for L-shape, "irregular" for others.

CRITICAL: Return ONLY the JSON object. No extra text."""

_MEASUREMENT_USER_PROMPT = """Read this floor plan drawing carefully.
Extract every visible room name, its annotated dimensions, and compute the area.
If dimensions are in millimetres (numbers > 100), divide by 1000 for metres.
Return only the JSON object with scale, rooms, and total area."""


# =============================================================================
# JSON PARSER
# =============================================================================

def _mm_or_m(val: float) -> float:
    """If val > 100, assume mm and convert to m; else assume m already."""
    if val > 100:
        return round(val / 1000.0, 3)
    return round(val, 3)


def _parse_measurement_response(raw: str, source_page: int) -> MeasuredDrawing:
    """
    Parse the LLM JSON response into a MeasuredDrawing.
    Robust to markdown wrapping and partial responses.
    """
    # Strip markdown code fences if present
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip()

    # Extract JSON block
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            logger.debug("VM: no JSON block in response")
            return MeasuredDrawing(source_page=source_page, scale="", scale_ratio=0)
        data = json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        logger.debug("VM: JSON parse error: %s — raw: %s", exc, raw[:300])
        return MeasuredDrawing(source_page=source_page, scale="", scale_ratio=0)

    scale_str = str(data.get("scale", "") or "")
    scale_ratio = 0
    m = re.search(r'1\s*:\s*(\d+)', scale_str)
    if m:
        scale_ratio = int(m.group(1))

    walls_detected = bool(data.get("walls_detected", False))
    dims_detected  = bool(data.get("dimensions_detected", False))
    total_area     = float(data.get("total_area_sqm", 0) or 0)

    rooms: List[MeasuredRoom] = []
    for item in data.get("rooms", []):
        try:
            raw_name = str(item.get("name", "")).strip()
            if not raw_name:
                continue

            name = raw_name.upper()
            shape = str(item.get("shape", "rectangular")).lower()

            l_raw = float(item.get("length_m", 0) or 0)
            w_raw = float(item.get("width_m", 0) or 0)

            # Guard against model returning mm instead of m
            length_m = _mm_or_m(l_raw)
            width_m  = _mm_or_m(w_raw)

            area_sqm_raw = float(item.get("area_sqm", 0) or 0)
            if area_sqm_raw > 10000:
                area_sqm_raw /= 1_000_000.0  # mm² → m²

            # Prefer provided area if plausible, else compute
            if area_sqm_raw > 0.5 and area_sqm_raw < 5000:
                area_sqm = round(area_sqm_raw, 2)
            elif length_m > 0 and width_m > 0:
                area_sqm = round(length_m * width_m, 2)
            else:
                continue  # not enough data

            # Sanity: rooms between 1 and 500 sqm are realistic
            if area_sqm < 0.5 or area_sqm > 500:
                continue

            conf = min(1.0, max(0.0, float(item.get("confidence", 0.75))))
            dims_raw = str(item.get("dimensions_raw", "") or "")

            rooms.append(MeasuredRoom(
                name=name,
                raw_name=raw_name,
                length_m=length_m,
                width_m=width_m,
                area_sqm=area_sqm,
                shape=shape,
                dimensions_raw=dims_raw,
                confidence=conf,
                source_page=source_page,
            ))
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("VM: room parse error: %s — item: %s", exc, item)

    # Recompute total_area if not given or implausible
    if (total_area < 1 or total_area > 50_000) and rooms:
        total_area = round(sum(r.area_sqm for r in rooms), 2)

    return MeasuredDrawing(
        source_page=source_page,
        scale=scale_str,
        scale_ratio=scale_ratio,
        rooms=rooms,
        total_area_sqm=total_area,
        walls_detected=walls_detected,
        dimensions_detected=dims_detected,
    )


# =============================================================================
# ROOM DEDUPLICATION
# =============================================================================

def _dedup_rooms(rooms: List[MeasuredRoom]) -> List[MeasuredRoom]:
    """
    Remove duplicate rooms across pages.
    Two rooms are considered duplicates if their names match and areas are within 15%.
    Keep the one with higher confidence.
    """
    seen: Dict[str, MeasuredRoom] = {}
    for r in rooms:
        key = r.name.upper().strip()
        if key in seen:
            existing = seen[key]
            # Check area similarity
            if existing.area_sqm > 0 and r.area_sqm > 0:
                ratio = min(existing.area_sqm, r.area_sqm) / max(existing.area_sqm, r.area_sqm)
                if ratio > 0.85:  # within 15% — likely the same room
                    if r.confidence > existing.confidence:
                        seen[key] = r
                    continue
            # Different areas: keep both by adding a suffix
            key = f"{key}_p{r.source_page}"
        seen[key] = r
    return list(seen.values())


# =============================================================================
# ROOM → RoomData CONVERSION (pipeline compatibility)
# =============================================================================

def _to_room_data(room: MeasuredRoom) -> RoomData:
    """Convert MeasuredRoom → finish_takeoff.RoomData for pipeline compatibility."""
    return RoomData(
        name=room.name,
        raw_name=room.raw_name,
        area_sqm=room.area_sqm,
        dim_l=room.length_m,   # already in metres
        dim_w=room.width_m,
        source_page=room.source_page,
        confidence=room.confidence,
    )


# =============================================================================
# BOQ GENERATOR
# =============================================================================

# Room height assumptions (same as finish_takeoff._get_room_height)
_ROOM_HEIGHTS: Dict[str, float] = {
    "LIVING": 2.85, "DRAWING": 2.85, "HALL": 2.85, "LOBBY": 2.85,
    "DINING": 2.75, "BEDROOM": 2.75, "MASTER": 2.75, "STUDY": 2.75,
    "KITCHEN": 2.60, "TOILET": 2.50, "BATHROOM": 2.50,
    "BALCONY": 2.50, "VERANDAH": 2.50, "TERRACE": 2.50,
    "STORE": 2.40, "SERVANT": 2.40, "PASSAGE": 2.50,
}


def _room_height(name: str) -> float:
    name_up = name.upper()
    for key, h in _ROOM_HEIGHTS.items():
        if key in name_up:
            return h
    return 2.70  # default


def _item(description: str, qty: float, unit: str, trade: str,
          spec: str = "", room: str = "") -> dict:
    return {
        "description": description,
        "qty": round(qty, 2),
        "unit": unit,
        "trade": trade,
        "spec": spec,
        "room": room,
        "source": "visual_measurement",
    }


def generate_measurement_items(rooms: List[MeasuredRoom]) -> List[dict]:
    """
    Generate finish QTO items from visually measured rooms.
    Produces floor, wall, and ceiling items for each room.
    """
    items: List[dict] = []

    for room in rooms:
        if room.area_sqm <= 0:
            continue

        floor_area = room.area_sqm
        height = _room_height(room.name)
        perim   = 2 * (room.length_m + room.width_m) if room.length_m and room.width_m else 0
        wall_area = perim * height if perim else floor_area * 3.5  # approx if no dims

        name_str = room.raw_name or room.name

        # ── Floor ─────────────────────────────────────────────────────────
        if any(kw in room.name for kw in ("TOILET", "BATHROOM", "WET")):
            items.append(_item(
                f"Non-slip ceramic floor tile (wet area) — {name_str}",
                floor_area, "sqm", "Finishes",
                spec="IS 15622 / IS 13630", room=name_str,
            ))
        elif any(kw in room.name for kw in ("BALCONY", "VERANDAH", "TERRACE")):
            items.append(_item(
                f"Anti-skid tiles / Kota stone — {name_str}",
                floor_area, "sqm", "Finishes",
                spec="IS 1130", room=name_str,
            ))
        elif any(kw in room.name for kw in ("KITCHEN",)):
            items.append(_item(
                f"Vitrified tile floor — {name_str}",
                floor_area, "sqm", "Finishes",
                spec="IS 15622", room=name_str,
            ))
        else:
            items.append(_item(
                f"Vitrified tile / marble floor — {name_str}",
                floor_area, "sqm", "Finishes",
                spec="IS 15622", room=name_str,
            ))

        # ── Walls ─────────────────────────────────────────────────────────
        if any(kw in room.name for kw in ("TOILET", "BATHROOM", "KITCHEN", "WET")):
            items.append(_item(
                f"Ceramic wall tile (2100 mm ht) — {name_str}",
                round(perim * 2.1, 2) if perim else round(wall_area * 0.75, 2),
                "sqm", "Finishes",
                spec="IS 15622", room=name_str,
            ))
            items.append(_item(
                f"Cement plaster to walls above tile — {name_str}",
                round(wall_area - (perim * 2.1 if perim else wall_area * 0.75), 2) or round(wall_area * 0.25, 2),
                "sqm", "Finishes",
                spec="IS 1661", room=name_str,
            ))
        else:
            items.append(_item(
                f"Cement plaster + putty + paint to walls — {name_str}",
                round(wall_area, 2), "sqm", "Finishes",
                spec="IS 1661", room=name_str,
            ))

        # ── Ceiling ───────────────────────────────────────────────────────
        items.append(_item(
            f"Ceiling plaster + putty + OBD — {name_str}",
            floor_area, "sqm", "Finishes",
            spec="IS 1661", room=name_str,
        ))

    # Deduplicate via combined description (shouldn't have dupes, but safety)
    seen_descs = set()
    unique_items = []
    for it in items:
        key = (it["description"], it["qty"])
        if key not in seen_descs:
            seen_descs.add(key)
            unique_items.append(it)

    return unique_items


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def _validate_scale_consistency(
    llm_scale_ratio: int,
    known_scale_ratio: int,
    source_page: int,
) -> Optional[str]:
    """
    Compare LLM-detected scale against scale_detector result.
    Returns a warning string if they differ by more than a factor of 1.5, else None.
    """
    if llm_scale_ratio <= 0 or known_scale_ratio <= 0:
        return None
    ratio = max(llm_scale_ratio, known_scale_ratio) / min(llm_scale_ratio, known_scale_ratio)
    if ratio > 1.5:
        return (
            f"Page {source_page + 1}: scale mismatch — LLM read 1:{llm_scale_ratio} "
            f"but scale_detector found 1:{known_scale_ratio}. "
            f"Using scale_detector value for dimension conversions."
        )
    return None


def run_visual_measurement(
    pdf_path: str,
    page_texts: List[Tuple[int, str, str]],
    llm_client: Any,
    max_pages: int = _MAX_PAGES_TO_SCAN,
    known_scale_ratio: int = 0,
    known_px_per_mm: float = 0.0,
) -> VisualMeasurementResult:
    """
    Main visual measurement runner.

    Args:
        pdf_path:          path to PDF file
        page_texts:        [(page_idx, ocr_text, doc_type), ...]
        llm_client:        OpenAI or Anthropic client with vision support
        max_pages:         maximum pages to process (cost control)
        known_scale_ratio: scale ratio from scale_detector.py (e.g. 100 for 1:100).
                           Used as fallback when LLM cannot read scale from drawing.
        known_px_per_mm:   pixels per real mm from scale_detector.py.
                           Stored in result for downstream validation use.

    Returns:
        VisualMeasurementResult with room_schedule, line_items, total_area_sqm
    """
    result = VisualMeasurementResult()

    # Store scale info immediately — available even on early returns
    result.px_per_mm = known_px_per_mm
    result.scale_source = "none"
    if known_scale_ratio > 0:
        result.scale_ratio = known_scale_ratio
        result.scale_source = "scale_detector"

    if not pdf_path:
        result.warnings.append("Visual measurement skipped: no PDF path")
        return result

    if llm_client is None:
        result.warnings.append(
            "Visual measurement skipped: no LLM client — "
            "set OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )
        return result

    if not _HAS_RENDERER:
        result.warnings.append("Visual measurement skipped: PyMuPDF (fitz) not installed")
        return result

    # Prioritise floor plan pages; cap at max_pages
    def _priority(tup: Tuple[int, str, str]) -> int:
        _, _, dt = tup
        d = dt.lower()
        if "floor" in d or "plan" in d:
            return 0
        if "drawing" in d or "architectural" in d or "layout" in d:
            return 1
        if "structural" in d:
            return 2
        if "spec" in d or "boq" in d or "schedule" in d:
            return 9   # deprioritise text-heavy pages
        return 5

    pages_to_scan = sorted(page_texts, key=_priority)[:max_pages]

    all_measured: List[MeasuredDrawing] = []
    all_rooms_flat: List[MeasuredRoom] = []

    for page_idx, _text, _doc_type in pages_to_scan:
        logger.info("Visual measurement: scanning page %d (%s)", page_idx, _doc_type)

        img_bytes = _render_page_to_bytes(
            pdf_path, page_idx,
            zoom=2.0,
            max_px=_IMAGE_MAX_PX,
            quality=_JPEG_QUALITY,
        )
        if img_bytes is None:
            result.warnings.append(f"Page {page_idx + 1}: could not render to image")
            continue

        try:
            raw_response = _call_vision_llm(
                llm_client,
                img_bytes,
                system_prompt=_MEASUREMENT_SYSTEM_PROMPT,
                user_prompt=_MEASUREMENT_USER_PROMPT,
            )
        except Exception as exc:
            logger.warning("VM: LLM call failed page %d: %s", page_idx, exc)
            result.warnings.append(
                f"Page {page_idx + 1}: vision LLM error — {type(exc).__name__}: {exc}"
            )
            continue

        drawing = _parse_measurement_response(raw_response, source_page=page_idx)

        # ── Scale cross-validation ────────────────────────────────────────
        if known_scale_ratio > 0:
            if drawing.scale_ratio <= 0:
                # LLM couldn't read scale — use the authoritative scale_detector value
                drawing.scale_ratio = known_scale_ratio
                drawing.scale = drawing.scale or f"1:{known_scale_ratio}"
                result.warnings.append(
                    f"Page {page_idx + 1}: LLM did not detect scale; "
                    f"using scale_detector value 1:{known_scale_ratio}"
                )
            else:
                # Both detected — check consistency
                mismatch_warn = _validate_scale_consistency(
                    drawing.scale_ratio, known_scale_ratio, page_idx
                )
                if mismatch_warn:
                    result.warnings.append(mismatch_warn)
                    # Trust scale_detector (text-based, deterministic) over LLM read
                    drawing.scale_ratio = known_scale_ratio
                    drawing.scale = f"1:{known_scale_ratio}"

        all_measured.append(drawing)
        all_rooms_flat.extend(drawing.rooms)
        result.pages_scanned += 1

        logger.info(
            "VM: page %d → %d rooms, total %.0f sqm (scale=%s, known_ratio=%d)",
            page_idx, len(drawing.rooms), drawing.total_area_sqm, drawing.scale,
            known_scale_ratio,
        )

    if not all_rooms_flat:
        result.warnings.append(
            f"Visual measurement: no rooms measured — {result.pages_scanned} page(s) scanned. "
            "Ensure the PDF contains floor plan drawings with visible dimensions."
        )
        return result

    # Deduplicate rooms across pages
    unique_rooms = _dedup_rooms(all_rooms_flat)

    # Build aggregates
    result.mode = "vision_measurement"
    result.measured_drawings = all_measured
    result.all_rooms = unique_rooms
    result.room_schedule = [_to_room_data(r) for r in unique_rooms]
    result.line_items = generate_measurement_items(unique_rooms)
    result.total_area_sqm = round(sum(r.area_sqm for r in unique_rooms), 2)

    # Best scale from the most rooms detected
    best = max(all_measured, key=lambda d: len(d.rooms), default=None)
    result.detected_scale = best.scale if best else ""

    # Store authoritative scale info
    result.px_per_mm = known_px_per_mm
    if known_scale_ratio > 0:
        result.scale_ratio = known_scale_ratio
        result.scale_source = "scale_detector"
    elif best and best.scale_ratio > 0:
        result.scale_ratio = best.scale_ratio
        result.scale_source = "llm"
    else:
        result.scale_source = "none"

    logger.info(
        "Visual measurement complete: %d unique rooms, %.0f sqm total, "
        "%d BOQ items",
        len(unique_rooms), result.total_area_sqm, len(result.line_items),
    )

    return result


# =============================================================================
# HELPERS
# =============================================================================

def room_schedule_to_dataframe_rows(rooms: List[MeasuredRoom]) -> List[dict]:
    """Convert rooms to list of dicts suitable for st.dataframe()."""
    rows = []
    for r in rooms:
        dims = r.dimensions_raw or (
            f"{r.length_m:.2f} × {r.width_m:.2f} m"
            if r.length_m and r.width_m else "—"
        )
        rows.append({
            "Room": r.raw_name or r.name,
            "Area (sqm)": r.area_sqm,
            "Dimensions": dims,
            "Shape": r.shape.title(),
            "Confidence": f"{r.confidence:.0%}",
            "Page": r.source_page + 1,
        })
    # Sort by area descending
    return sorted(rows, key=lambda x: x["Area (sqm)"], reverse=True)
