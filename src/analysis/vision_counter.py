"""
src/analysis/vision_counter.py

Thin adapter layer between visual_element_detector.VisualElement objects
and the item_normalizer.build_line_items() spec_item format.

The visual_element_detector already handles rendering, LLM calling, and
element detection.  This module:
1. Defines the lightweight VisionCount dataclass for testing & typing.
2. Converts VisualElement lists → VisionCount lists.
3. Converts VisionCount lists → spec_item dicts (for item_normalizer).

Trade mapping for Indian construction context:
  doors, windows, ventilators, stairs → architectural
  columns, beams, staircases → structural
  wc, washbasin, urinal, kitchen_sink, bathtub, shower → mep
  light_point, socket, exhaust_fan → mep (electrical)
  parking_bay, ramp → civil
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trade mapping
# ---------------------------------------------------------------------------

_TYPE_TO_TRADE: dict[str, str] = {
    "room":           "architectural",
    "door":           "architectural",
    "window":         "architectural",
    "ventilator":     "architectural",
    "stair":          "architectural",
    "staircase_block":"architectural",
    "balcony":        "architectural",
    "terrace":        "architectural",
    "column":         "structural",
    "beam":           "structural",
    "lift":           "architectural",
    "wc":             "mep",
    "washbasin":      "mep",
    "urinal":         "mep",
    "kitchen_sink":   "mep",
    "bathtub":        "mep",
    "shower":         "mep",
    "light_point":    "mep",
    "socket":         "mep",
    "exhaust_fan":    "mep",
    "utility_shaft":  "mep",
    "toilet_block":   "mep",
    "parking_bay":    "civil",
    "ramp":           "civil",
}

_DEFAULT_TRADE = "general"


def _element_type_to_trade(element_type: str) -> str:
    return _TYPE_TO_TRADE.get(element_type.lower().strip(), _DEFAULT_TRADE)


# ---------------------------------------------------------------------------
# VisionCount — lightweight dataclass for testing and inter-module typing
# ---------------------------------------------------------------------------

@dataclass
class VisionCount:
    """
    Simplified element count extracted from a drawing page.

    Source: visual_element_detector.VisualElement → VisionCount via
    visual_elements_to_vision_counts().
    """
    element_type: str     # "door", "window", "wc", "column", etc.
    count: int
    confidence: float     # 0.0–1.0 (clamped on construction)
    page: int             # 0-indexed source page
    trade: str            # "architectural" | "structural" | "mep" | "civil" | "general"
    description: str = "" # optional human-readable label from detector

    def __post_init__(self) -> None:
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
        self.count = max(0, int(self.count))


# ---------------------------------------------------------------------------
# Conversion: VisualElement → VisionCount
# ---------------------------------------------------------------------------

def visual_elements_to_vision_counts(elements: list) -> List[VisionCount]:
    """
    Convert a list of VisualElement objects (from visual_element_detector)
    into VisionCount objects.

    Aggregates counts by (element_type, page) pair so each page contributes
    one VisionCount per element type.

    Args:
        elements: List of VisualElement dataclass instances.

    Returns:
        List of VisionCount objects, sorted by (page, element_type).
    """
    # Aggregate by (element_type, page)
    aggregated: dict[tuple[str, int], VisionCount] = {}

    for el in elements:
        et = str(getattr(el, "element_type", "unknown")).lower().strip()
        page = int(getattr(el, "source_page", 0))
        count = max(0, int(getattr(el, "count", 1)))
        conf = min(1.0, max(0.0, float(getattr(el, "confidence", 0.5))))
        desc = str(getattr(el, "description", ""))[:120]
        trade = _element_type_to_trade(et)

        key = (et, page)
        if key in aggregated:
            existing = aggregated[key]
            aggregated[key] = VisionCount(
                element_type=et,
                count=existing.count + count,
                confidence=max(existing.confidence, conf),
                page=page,
                trade=trade,
                description=desc if not existing.description else existing.description,
            )
        else:
            aggregated[key] = VisionCount(
                element_type=et,
                count=count,
                confidence=conf,
                page=page,
                trade=trade,
                description=desc,
            )

    return sorted(aggregated.values(), key=lambda c: (c.page, c.element_type))


# ---------------------------------------------------------------------------
# Conversion: VisionCount → spec_item dict (for item_normalizer)
# ---------------------------------------------------------------------------

_TYPE_TO_DESCRIPTION: dict[str, str] = {
    "door":           "Door (timber/aluminium frame) — vision detected",
    "window":         "Window (aluminium/UPVC frame) — vision detected",
    "ventilator":     "Ventilator / louvre — vision detected",
    "stair":          "Staircase flight — vision detected",
    "staircase_block":"Staircase — vision detected",
    "column":         "RCC column — vision detected",
    "beam":           "RCC beam — vision detected",
    "lift":           "Passenger lift shaft — vision detected",
    "wc":             "Water closet (EWC) — vision detected",
    "washbasin":      "Wash basin with CP fittings — vision detected",
    "urinal":         "Urinal bowl — vision detected",
    "kitchen_sink":   "Kitchen sink SS — vision detected",
    "bathtub":        "Bath tub with fittings — vision detected",
    "shower":         "Shower unit complete — vision detected",
    "light_point":    "Light fitting point — vision detected",
    "socket":         "Power socket/outlet — vision detected",
    "exhaust_fan":    "Exhaust fan — vision detected",
    "parking_bay":    "Parking bay — vision detected",
    "ramp":           "Vehicular/pedestrian ramp — vision detected",
    "balcony":        "Balcony — vision detected",
    "terrace":        "Terrace/roof area — vision detected",
    "toilet_block":   "Toilet block — vision detected",
    "utility_shaft":  "Utility shaft — vision detected",
}

_COUNT_UNITS = {"nos", "each", "no"}  # units for COUNT-type items


def vision_counts_to_spec_items(counts: List[VisionCount]) -> List[dict]:
    """
    Convert VisionCount objects into spec_item dicts suitable for passing
    to item_normalizer.build_line_items() as the spec_items parameter.

    Items with count=0 or confidence<0.4 are excluded.
    Room-type counts are excluded (rooms contribute area, not discrete qty).

    Each output dict has:
      description, qty, unit, trade, source, source_page, confidence,
      is_priceable=True, unit_inferred=False
    """
    items: List[dict] = []
    for vc in counts:
        # Exclude low-confidence and zero-count
        if vc.count <= 0 or vc.confidence < 0.4:
            continue
        # Rooms are measured by area, not counted as line items here
        if vc.element_type == "room":
            continue

        desc = _TYPE_TO_DESCRIPTION.get(
            vc.element_type,
            f"{vc.element_type.replace('_', ' ').title()} — vision detected",
        )
        if vc.description:
            desc = f"{desc} ({vc.description[:60]})"

        items.append({
            "description": desc,
            "qty": float(vc.count),
            "unit": "nos",
            "trade": vc.trade,
            "source": "vision_count",
            "source_page": vc.page,
            "confidence": round(vc.confidence, 3),
            "is_priceable": True,
            "unit_inferred": False,
        })

    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def count_elements_from_visual_result(visual_result: Any) -> List[VisionCount]:
    """
    Extract VisionCount objects from a VisualQTO result object
    (returned by visual_element_detector.run_visual_detection).

    Returns empty list if visual_result is None or has no elements.
    """
    if visual_result is None:
        return []
    elements = getattr(visual_result, "elements", [])
    if not elements:
        return []
    return visual_elements_to_vision_counts(elements)
