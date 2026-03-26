"""
Visual Element Detector — AI Vision-Based Drawing Takeoff.

Renders each PDF drawing page to an image and sends it to the LLM (Claude or GPT-4o)
with a structured prompt.  The model returns a JSON payload listing detected:
  - Rooms (name, approximate area, dimensions if visible)
  - Doors (count, types: swing/sliding/double)
  - Windows (count, types)
  - Stairs / lifts
  - Columns visible as structural grid
  - Plumbing fixtures (WC, washbasin, urinal symbols)
  - Electrical symbols (lighting point symbols, socket symbols)
  - Dimensions visible on the drawing (key ones)
  - Drawing scale (if legible as a note on the drawing)
  - Building footprint or total area (if noted)

Each element is returned with a confidence score.  The BOQ items generated from
visual detection are labelled `source: "visual_detect"` so they can be blended
with or replace text-based schedule items downstream.

Dependencies:
  - fitz (PyMuPDF)  — PDF page rendering
  - Pillow (PIL)    — image encode to JPEG/PNG for base64 transmission
  - llm_client      — OpenAI or Anthropic client (same pattern as llm_enrichment.py)

Design constraints:
  - Graceful degradation: if fitz / llm_client are not available, returns empty result.
  - Caches rendered images in memory within a single run to avoid re-rendering.
  - Only runs on pages classified as 'plan', 'drawing', 'floor_plan', 'structural', etc.
  - Maximum 8 pages scanned per PDF to control API cost.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_MAX_PAGES_TO_SCAN = int(os.environ.get("XBOQ_VIS_MAX_PAGES", "8"))
_IMAGE_MAX_PX = int(os.environ.get("XBOQ_VIS_MAX_PX", "1600"))   # longest side in pixels
_JPEG_QUALITY = int(os.environ.get("XBOQ_VIS_JPEG_QUALITY", "80"))

_OPENAI_VISION_MODEL    = os.environ.get("XBOQ_OPENAI_MODEL",    "gpt-4o")
_ANTHROPIC_VISION_MODEL = os.environ.get("XBOQ_ANTHROPIC_MODEL", "claude-opus-4-5")

# Page types that are worth sending to vision
_DRAWING_DOC_TYPES = frozenset((
    "drawing", "plan", "floor_plan", "structural", "elevation",
    "section", "detail", "site_plan", "site", "architectural",
))

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VisualElement:
    """A single element detected visually on a drawing page."""
    element_type: str       # "room" | "door" | "window" | "stair" | "column" |
                            #  "wc" | "washbasin" | "light_point" | "socket" |
                            #  "lift" | "parking" | "ramp" | "balcony" | "toilet_block"
    count: int              # detected quantity
    description: str        # e.g. "Bedroom", "WC symbol in south-east corner"
    area_sqm: float         # 0.0 if not measurable
    dimensions_m: str       # "4.5 × 3.2" or "" if not detected
    source_page: int
    confidence: float       # 0.0–1.0


@dataclass
class VisualQTO:
    """Aggregated result from the visual element detection pass."""
    elements: List[VisualElement] = field(default_factory=list)
    line_items: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mode: str = "none"          # "vision_ai" | "none"
    pages_scanned: int = 0
    detected_scale: str = ""    # e.g. "1:100" if legible in drawing
    detected_area_sqm: float = 0.0
    low_confidence_elements: List[VisualElement] = field(default_factory=list)
    schedule_reconciliation: List[dict] = field(default_factory=list)  # [{type, vision, schedule, delta}]


# =============================================================================
# VISION PROMPT
# =============================================================================

_SYSTEM_PROMPT = """You are an expert construction estimator and quantity surveyor analysing
architectural and structural drawing images. Extract building elements precisely.

Return ONLY valid JSON. No markdown, no explanation. Use this exact schema:

{
  "scale": "1:100",          // drawing scale if visible, else ""
  "total_area_sqm": 0,       // floor area if annotated, else 0
  "elements": [
    {
      "type": "room",        // see type list below
      "count": 1,
      "description": "Master Bedroom",
      "area_sqm": 18.5,      // 0 if not determinable
      "dimensions_m": "4.5 x 3.0",  // "" if not visible
      "confidence": 0.90
    }
  ]
}

Element types (use exactly these strings):
  room, door, window, stair, lift, column, wc, washbasin, urinal, kitchen_sink,
  bathtub, shower, light_point, socket, exhaust_fan, parking_bay, ramp, balcony,
  terrace, toilet_block, utility_shaft, staircase_block

Count rules:
- Count individual symbols on the drawing.
- For rooms: each labelled enclosed space = 1 element with its name.
- For doors: count door-swing arcs.
- For windows: count window symbols on walls.
- For columns: count filled rectangles/circles on structural grid.
- Confidence 0.9+ = clearly visible. 0.7-0.9 = inferred. Below 0.7 = guessed.

Indian drawing conventions:
- WC symbol: small rectangle with circle inside.
- Washbasin: D-shaped or oval symbol.
- Lighting point: cross or star symbol in ceiling plan.
- Dimensions are in millimetres (3000 = 3 m).
- Scale bar may be at bottom or title block.
"""

_USER_PROMPT = """Analyse this architectural drawing page.
Detect and count all visible building elements following the schema exactly.
If this is a structural plan, focus on columns, beams, footings.
If this is a floor plan, focus on rooms, doors, windows, sanitary fixtures.
If this is a site plan, focus on parking, roads, green areas.
Return only valid JSON."""


# =============================================================================
# PDF → IMAGE
# =============================================================================

def _render_page_to_bytes(
    pdf_path: str,
    page_idx: int,
    zoom: float = 2.0,
    max_px: int = _IMAGE_MAX_PX,
    quality: int = _JPEG_QUALITY,
) -> Optional[bytes]:
    """
    Render a PDF page to JPEG bytes, resizing so the longest side ≤ max_px.
    Returns None if fitz not available or page is blank.
    """
    try:
        import fitz as _fitz
        from PIL import Image
    except ImportError:
        return None

    try:
        doc = _fitz.open(pdf_path)
        if page_idx >= len(doc):
            doc.close()
            return None

        page = doc[page_idx]
        mat = _fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Resize to max_px on the longest side
        w, h = img.size
        scale = min(1.0, max_px / max(w, h))
        if scale < 1.0:
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    except Exception as exc:
        logger.debug("Visual: render failed for page %d: %s", page_idx, exc)
        return None


# =============================================================================
# LLM VISION CALL
# =============================================================================

def _call_vision_llm(
    llm_client: Any,
    image_bytes: bytes,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Send image + prompt to OpenAI or Anthropic vision endpoint.
    Returns the model's raw text response.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    if hasattr(llm_client, "chat"):
        # OpenAI-style (GPT-4o)
        response = llm_client.chat.completions.create(
            model=_OPENAI_VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            max_tokens=1500,
            temperature=0.1,
        )
        return response.choices[0].message.content

    elif hasattr(llm_client, "messages"):
        # Anthropic-style (Claude claude-opus-4-5)
        response = llm_client.messages.create(
            model=_ANTHROPIC_VISION_MODEL,
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        return response.content[0].text

    else:
        raise ValueError("Unknown LLM client type — cannot call vision endpoint")


# =============================================================================
# JSON PARSER
# =============================================================================

def _parse_vision_response(raw: str, source_page: int) -> Tuple[List[VisualElement], str, float]:
    """
    Parse the LLM JSON response into VisualElement objects.
    Returns (elements, scale_str, total_area_sqm).
    """
    # Extract JSON block from response
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            logger.debug("Vision: no JSON block in response: %s", raw[:200])
            return [], "", 0.0
        data = json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        logger.debug("Vision: JSON parse error: %s — raw: %s", exc, raw[:200])
        return [], "", 0.0

    scale_str = data.get("scale", "")
    total_area = float(data.get("total_area_sqm", 0) or 0)

    elements: List[VisualElement] = []
    for item in data.get("elements", []):
        try:
            el = VisualElement(
                element_type=str(item.get("type", "unknown")).lower().strip(),
                count=max(0, int(item.get("count", 1))),
                description=str(item.get("description", ""))[:120],
                area_sqm=float(item.get("area_sqm", 0) or 0),
                dimensions_m=str(item.get("dimensions_m", "") or ""),
                source_page=source_page,
                confidence=min(1.0, max(0.0, float(item.get("confidence", 0.7)))),
            )
            if el.count > 0:
                elements.append(el)
        except (TypeError, ValueError) as exc:
            logger.debug("Vision: element parse error: %s — item: %s", exc, item)

    return elements, scale_str, total_area


# =============================================================================
# BOQ GENERATOR FROM VISUAL ELEMENTS
# =============================================================================

def _item(description: str, qty: float, unit: str, trade: str, spec: str = "") -> dict:
    return {
        "description": description,
        "qty": round(qty, 2),
        "unit": unit,
        "trade": trade,
        "spec": spec,
        "source": "visual_detect",
    }


def generate_visual_items(elements: List[VisualElement]) -> List[dict]:
    """
    Convert visually detected elements into BOQ line items.
    """
    items: List[dict] = []

    # Aggregate by type
    by_type: Dict[str, List[VisualElement]] = {}
    for el in elements:
        by_type.setdefault(el.element_type, []).append(el)

    # ── Structural ────────────────────────────────────────────────────────────
    if "column" in by_type:
        col_count = sum(e.count for e in by_type["column"])
        items.append(_item(
            f"RCC columns — {col_count} nos detected",
            col_count, "Nos", "Civil/Structural",
            spec="IS 456",
        ))

    # ── Rooms ─────────────────────────────────────────────────────────────────
    total_room_area = 0.0
    room_count = 0
    for el in by_type.get("room", []):
        total_room_area += el.area_sqm * el.count
        room_count += el.count
    if room_count > 0:
        items.append(_item(
            f"Total habitable area — {room_count} rooms detected",
            total_room_area or room_count,
            "sqm" if total_room_area > 0 else "Nos",
            "Architecture",
        ))

    # ── Doors ─────────────────────────────────────────────────────────────────
    if "door" in by_type:
        door_count = sum(e.count for e in by_type["door"])
        items.append(_item(
            f"Doors (timber/aluminium) — {door_count} nos detected",
            door_count, "Nos", "Joinery",
            spec="IS 4021",
        ))

    # ── Windows ───────────────────────────────────────────────────────────────
    if "window" in by_type:
        win_count = sum(e.count for e in by_type["window"])
        items.append(_item(
            f"Windows (aluminium/UPVC) — {win_count} nos detected",
            win_count, "Nos", "Joinery",
            spec="IS 1038",
        ))

    # ── Stairs ────────────────────────────────────────────────────────────────
    for stype in ("stair", "staircase_block"):
        if stype in by_type:
            stair_count = sum(e.count for e in by_type[stype])
            items.append(_item(
                f"Staircase — {stair_count} flight(s) detected",
                stair_count, "Flight", "Civil/Structural",
            ))
            break

    # ── Lifts ─────────────────────────────────────────────────────────────────
    if "lift" in by_type:
        lift_count = sum(e.count for e in by_type["lift"])
        items.append(_item(
            f"Passenger lift — {lift_count} nos",
            lift_count, "Nos", "Lift/Escalator",
            spec="IS 14665",
        ))

    # ── Sanitary fixtures ─────────────────────────────────────────────────────
    sanitary_map = {
        "wc":            ("Water closet (EWC)", "Plumbing", "IS 2556"),
        "washbasin":     ("Wash basin with CP fittings", "Plumbing", "IS 2556"),
        "urinal":        ("Urinal bowl", "Plumbing", "IS 2556"),
        "kitchen_sink":  ("Kitchen sink SS", "Plumbing", "IS 783"),
        "bathtub":       ("Bath tub with fittings", "Plumbing", "IS 2556"),
        "shower":        ("Shower unit complete", "Plumbing", "IS 2556"),
    }
    for etype, (desc, trade, spec) in sanitary_map.items():
        if etype in by_type:
            count = sum(e.count for e in by_type[etype])
            items.append(_item(
                f"{desc} — {count} nos detected from drawing",
                count, "Nos", trade, spec=spec,
            ))

    # ── Electrical symbols ────────────────────────────────────────────────────
    if "light_point" in by_type:
        lp_count = sum(e.count for e in by_type["light_point"])
        items.append(_item(
            f"Light fitting (type TBD) — {lp_count} nos detected from drawing",
            lp_count, "Nos", "Electrical",
            spec="IS 3646",
        ))
    if "socket" in by_type:
        sk_count = sum(e.count for e in by_type["socket"])
        items.append(_item(
            f"Power socket/outlet — {sk_count} nos detected from drawing",
            sk_count, "Nos", "Electrical",
            spec="IS 1293",
        ))
    if "exhaust_fan" in by_type:
        ef_count = sum(e.count for e in by_type["exhaust_fan"])
        items.append(_item(
            f"Exhaust fan — {ef_count} nos detected from drawing",
            ef_count, "Nos", "HVAC",
        ))

    # ── Parking ───────────────────────────────────────────────────────────────
    if "parking_bay" in by_type:
        pk_count = sum(e.count for e in by_type["parking_bay"])
        items.append(_item(
            f"Parking bay marking & kerb — {pk_count} bays",
            pk_count, "Nos", "External Works",
        ))

    return items


# =============================================================================
# CONFIDENCE GATING
# =============================================================================

_CONFIDENCE_THRESHOLD = float(os.environ.get("XBOQ_VIS_CONFIDENCE_MIN", "0.7"))


def _partition_by_confidence(
    elements: List[VisualElement],
    threshold: float = _CONFIDENCE_THRESHOLD,
) -> Tuple[List[VisualElement], List[VisualElement]]:
    """
    Split elements into (high_confidence, low_confidence) lists.
    High-confidence items go into the result; low-confidence are flagged.
    """
    high, low = [], []
    for el in elements:
        (high if el.confidence >= threshold else low).append(el)
    return high, low


# =============================================================================
# QUADRANT SPLITTING (for dense floor plans)
# =============================================================================

def _split_image_into_quadrants(image_bytes: bytes) -> List[bytes]:
    """
    Split a JPEG image into 4 quadrants (NW, NE, SW, SE).
    Returns list of JPEG byte strings, one per quadrant.
    Returns empty list if PIL not available.
    """
    try:
        from PIL import Image
    except ImportError:
        return []

    try:
        import io as _io
        img = Image.open(_io.BytesIO(image_bytes))
        w, h = img.size
        hw, hh = w // 2, h // 2
        quadrants = [
            img.crop((0,   0,   hw, hh)),   # NW
            img.crop((hw,  0,   w,  hh)),   # NE
            img.crop((0,   hh,  hw, h)),    # SW
            img.crop((hw,  hh,  w,  h)),    # SE
        ]
        result = []
        for q in quadrants:
            buf = _io.BytesIO()
            q.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            result.append(buf.getvalue())
        return result
    except Exception as exc:
        logger.debug("Quadrant split failed: %s", exc)
        return []


def _is_dense_drawing(elements: List[VisualElement]) -> bool:
    """
    Heuristic: if first-pass detects > 30 total elements, the drawing is
    considered dense and worth a quadrant re-pass for accuracy.
    """
    return sum(e.count for e in elements) > 30


# =============================================================================
# CROSS-PAGE ELEMENT DEDUPLICATION
# =============================================================================

# Page types that likely show the SAME floor from a different angle/purpose.
# If two pages of the same floor-type are scanned, their elements may be duplicates.
_SAME_FLOOR_TYPES = frozenset(("floor_plan", "plan", "drawing", "architectural", "layout"))
_NON_DUPLICATING_TYPES = frozenset(("structural", "elevation", "section", "site_plan", "site"))


def _dedup_elements_across_pages(
    elements: List[VisualElement],
    page_doc_types: Dict[int, str],
) -> Tuple[List[VisualElement], List[str]]:
    """
    Merge elements of the same type across pages that likely show the same floor.

    Strategy:
    - Group elements by type.
    - For pages classified as floor_plan/plan/drawing (likely same floor content):
      keep the count from the page with the highest total confidence-weighted sum.
    - For structural/elevation/section pages: never merge — different views.

    Returns (deduplicated_elements, warnings).
    """
    warnings_out: List[str] = []

    # Separate structural/elevation elements from architectural ones
    arch_elements: List[VisualElement] = []
    non_arch_elements: List[VisualElement] = []
    for el in elements:
        dt = page_doc_types.get(el.source_page, "unknown").lower()
        if any(k in dt for k in _NON_DUPLICATING_TYPES):
            non_arch_elements.append(el)
        else:
            arch_elements.append(el)

    # For architectural elements: find same-type elements across floor-plan pages
    # Group by element_type
    by_type: Dict[str, List[VisualElement]] = {}
    for el in arch_elements:
        by_type.setdefault(el.element_type, []).append(el)

    merged: List[VisualElement] = []
    for etype, group in by_type.items():
        # For each page, sum the count × confidence as a quality score
        page_scores: Dict[int, float] = {}
        page_elements: Dict[int, List[VisualElement]] = {}
        for el in group:
            page_scores[el.source_page] = page_scores.get(el.source_page, 0.0) + (
                el.count * el.confidence
            )
            page_elements.setdefault(el.source_page, []).append(el)

        if len(page_scores) <= 1:
            # Only one page → no dedup needed
            merged.extend(group)
            continue

        # Multiple pages have this element type — check if they're all floor-plan types
        pages_are_floor = [
            any(k in page_doc_types.get(p, "").lower() for k in _SAME_FLOOR_TYPES)
            for p in page_scores
        ]
        if all(pages_are_floor):
            # All floor-plan pages → likely showing same floor, deduplicate
            best_page = max(page_scores, key=lambda p: page_scores[p])
            best_els = page_elements[best_page]
            merged.extend(best_els)
            total_before = sum(el.count for el in group)
            total_after  = sum(el.count for el in best_els)
            if total_before != total_after:
                warnings_out.append(
                    f"Cross-page dedup ({etype}): {total_before} elements across "
                    f"{len(page_scores)} floor-plan pages → kept {total_after} "
                    f"from page {best_page + 1} (highest confidence)"
                )
        else:
            # Mixed page types → keep all (different views, not duplicates)
            merged.extend(group)

    merged.extend(non_arch_elements)
    return merged, warnings_out


# =============================================================================
# SCHEDULE RECONCILIATION
# =============================================================================

def _reconcile_with_schedules(
    elements: List[VisualElement],
    text_schedule_counts: Dict[str, int],
) -> Tuple[List[VisualElement], List[dict]]:
    """
    Compare vision-detected element counts against text-derived schedule counts.

    For each element type in text_schedule_counts:
    - If vision count matches within 20%: mark as validated
    - If vision count is significantly lower: flag as possible under-count
    - If vision count is significantly higher: flag as possible over-count

    Returns (elements_unchanged, reconciliation_report).
    Elements are not modified — reconciliation is advisory only.
    """
    report: List[dict] = []
    if not text_schedule_counts:
        return elements, report

    # Sum vision counts by element type
    vision_by_type: Dict[str, int] = {}
    for el in elements:
        vision_by_type[el.element_type] = vision_by_type.get(el.element_type, 0) + el.count

    for etype, sched_count in text_schedule_counts.items():
        vis_count = vision_by_type.get(etype, 0)
        if sched_count <= 0:
            continue

        delta = vis_count - sched_count
        pct = abs(delta) / sched_count

        if pct <= 0.20:
            status = "validated"
        elif delta < 0:
            status = "vision_under_count"
        else:
            status = "vision_over_count"

        report.append({
            "type": etype,
            "vision_count": vis_count,
            "schedule_count": sched_count,
            "delta": delta,
            "pct_diff": round(pct * 100, 1),
            "status": status,
        })

    return elements, report


# =============================================================================
# QUADRANT RE-PASS (for dense drawings)
# =============================================================================

def _quadrant_repass(
    llm_client: Any,
    image_bytes: bytes,
    page_idx: int,
) -> Tuple[List[VisualElement], List[str]]:
    """
    For dense drawings: split into 4 quadrants, re-detect, sum counts.
    Returns aggregated elements (not deduplicated yet) and any warnings.
    """
    warnings_out: List[str] = []
    quadrants = _split_image_into_quadrants(image_bytes)
    if not quadrants:
        return [], ["Quadrant split unavailable (PIL missing)"]

    quad_elements: List[VisualElement] = []
    for q_idx, q_bytes in enumerate(quadrants):
        try:
            raw = _call_vision_llm(
                llm_client, q_bytes,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=(
                    f"Analyse QUADRANT {q_idx + 1}/4 of this floor plan. "
                    "Count only elements visible in this quadrant section."
                ),
            )
            els, _, _ = _parse_vision_response(raw, source_page=page_idx)
            quad_elements.extend(els)
        except Exception as exc:
            warnings_out.append(f"Quadrant {q_idx + 1} re-pass failed: {exc}")

    return quad_elements, warnings_out


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_visual_detection(
    pdf_path: str,
    page_texts: List[Tuple[int, str, str]],
    llm_client: Any,
    max_pages: int = _MAX_PAGES_TO_SCAN,
    text_schedule_counts: Optional[Dict[str, int]] = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    enable_quadrant_repass: bool = True,
) -> VisualQTO:
    """
    Main visual element detection runner.

    Args:
        pdf_path:              path to the uploaded PDF
        page_texts:            [(page_idx, ocr_text, doc_type), ...]
        llm_client:            an OpenAI or Anthropic client instance (must support vision)
        max_pages:             maximum pages to send to vision LLM (cost control)
        text_schedule_counts:  dict mapping element_type → count from parsed text schedules
                               (e.g. {"door": 24, "window": 18}). Used for reconciliation.
        confidence_threshold:  elements below this confidence are segregated to
                               low_confidence_elements (default: 0.7)
        enable_quadrant_repass: if True, re-detect in quadrants for dense drawings (> 30 els)

    Returns:
        VisualQTO with detected elements, line_items, warnings, low_confidence_elements,
        and schedule_reconciliation report
    """
    result = VisualQTO()

    if not pdf_path:
        result.warnings.append("Visual detection skipped: no PDF path provided")
        return result

    if llm_client is None:
        result.warnings.append(
            "Visual detection skipped: no LLM client — set OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )
        return result

    # Select pages to scan: prefer drawing/plan types; sort by drawing-likeness
    def _page_priority(tup: Tuple[int, str, str]) -> int:
        _, _, dt = tup
        dt_low = dt.lower()
        if any(k in dt_low for k in ("floor_plan", "plan", "drawing")):
            return 0
        if any(k in dt_low for k in ("structural", "architectural")):
            return 1
        if any(k in dt_low for k in ("elevation", "section", "site")):
            return 2
        return 10  # spec/boq pages — scan last

    drawing_pages = sorted(page_texts, key=_page_priority)[:max_pages]

    # Build page_idx → doc_type map for cross-page dedup
    page_doc_types: Dict[int, str] = {
        page_idx: doc_type for page_idx, _, doc_type in drawing_pages
    }

    all_elements: List[VisualElement] = []
    scales_found: List[str] = []
    areas_found: List[float] = []

    for page_idx, _text, _doc_type in drawing_pages:
        logger.info("Visual: scanning page %d (%s)", page_idx, _doc_type)
        image_bytes = _render_page_to_bytes(pdf_path, page_idx)
        if image_bytes is None:
            result.warnings.append(f"Page {page_idx + 1}: could not render to image")
            continue

        try:
            raw = _call_vision_llm(
                llm_client,
                image_bytes,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_USER_PROMPT,
            )
        except Exception as exc:
            logger.warning("Visual: LLM call failed for page %d: %s", page_idx, exc)
            result.warnings.append(
                f"Page {page_idx + 1}: vision LLM error — {type(exc).__name__}: {exc}"
            )
            continue

        page_elements, scale_str, area_sqm = _parse_vision_response(raw, source_page=page_idx)

        # ── Quadrant re-pass for dense drawings ──────────────────────────
        if (
            enable_quadrant_repass
            and _is_dense_drawing(page_elements)
            and llm_client is not None
        ):
            logger.info(
                "Visual: page %d is dense (%d elements), running quadrant re-pass",
                page_idx, sum(e.count for e in page_elements),
            )
            quad_els, quad_warns = _quadrant_repass(llm_client, image_bytes, page_idx)
            if quad_els:
                # Replace first-pass elements with quadrant-aggregated ones if more found
                if sum(e.count for e in quad_els) > sum(e.count for e in page_elements):
                    page_elements = quad_els
                    result.warnings.append(
                        f"Page {page_idx + 1}: quadrant re-pass found more elements "
                        f"({sum(e.count for e in quad_els)} vs "
                        f"{sum(e.count for e in page_elements)}); using quadrant result"
                    )
            result.warnings.extend(quad_warns)

        all_elements.extend(page_elements)
        if scale_str:
            scales_found.append(scale_str)
        if area_sqm > 0:
            areas_found.append(area_sqm)

        result.pages_scanned += 1
        logger.info(
            "Visual: page %d → %d elements detected (scale=%s, area=%.0f sqm)",
            page_idx, len(page_elements), scale_str, area_sqm,
        )

    if all_elements:
        # ── Cross-page deduplication ──────────────────────────────────────
        deduped_elements, dedup_warnings = _dedup_elements_across_pages(
            all_elements, page_doc_types
        )
        result.warnings.extend(dedup_warnings)

        # ── Confidence gating ─────────────────────────────────────────────
        high_conf, low_conf = _partition_by_confidence(deduped_elements, confidence_threshold)
        result.low_confidence_elements = low_conf
        if low_conf:
            low_types = ", ".join(
                f"{e.element_type}(p{e.source_page+1}, conf={e.confidence:.0%})"
                for e in low_conf[:5]
            )
            result.warnings.append(
                f"{len(low_conf)} element(s) below confidence threshold "
                f"{confidence_threshold:.0%} — moved to low_confidence_elements: {low_types}"
            )

        result.mode = "vision_ai"
        result.elements = high_conf  # only high-confidence items in main list
        result.line_items = generate_visual_items(high_conf)
        if scales_found:
            result.detected_scale = scales_found[0]
        if areas_found:
            result.detected_area_sqm = max(areas_found)

        # ── Schedule reconciliation ───────────────────────────────────────
        if text_schedule_counts:
            _, result.schedule_reconciliation = _reconcile_with_schedules(
                high_conf, text_schedule_counts
            )

        logger.info(
            "Visual detection complete: %d high-conf + %d low-conf elements "
            "across %d pages → %d BOQ items",
            len(high_conf), len(low_conf), result.pages_scanned, len(result.line_items),
        )
    else:
        result.warnings.append(
            "Visual detection: no elements detected — "
            f"{result.pages_scanned} page(s) scanned"
        )

    return result


# =============================================================================
# ELEMENT SUMMARY HELPER
# =============================================================================

def summarise_visual_elements(elements: List[VisualElement]) -> Dict[str, Any]:
    """
    Return a summary dict suitable for display in the UI.
    """
    by_type: Dict[str, int] = {}
    for el in elements:
        by_type[el.element_type] = by_type.get(el.element_type, 0) + el.count

    total_room_area = sum(
        el.area_sqm * el.count
        for el in elements
        if el.element_type == "room"
    )

    return {
        "element_counts": by_type,
        "total_elements": sum(by_type.values()),
        "room_count": by_type.get("room", 0),
        "door_count": by_type.get("door", 0),
        "window_count": by_type.get("window", 0),
        "total_room_area_sqm": round(total_room_area, 1),
        "sanitary_fixture_count": sum(
            by_type.get(t, 0)
            for t in ("wc", "washbasin", "urinal", "kitchen_sink", "bathtub", "shower")
        ),
    }
