"""
Spec-Driven BOQ Generator.

For EPC-mode tenders that contain no BOQ tables and no drawings,
this module parses the scope-of-work text to extract key build parameters
(occupancy, floors, building type), derives a plinth area from published
norms, and returns those params so the downstream QTO modules can generate
a parametric BOQ.

Usage::

    from src.analysis.spec_boq_generator import extract_spec_params, SpecParams

    params = extract_spec_params(page_texts)
    if params.total_area_sqm > 0:
        # override _st_area_sqm / _st_floors before running QTO modules
        _st_area_sqm = params.total_area_sqm
        _st_floors   = params.floor_count

Area norms (sqm per unit) are based on CPWD Schedule of Rates / DSR 2023
and common Indian institutional construction benchmarks.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Area norms: sqm per occupant / unit  (CPWD DSR 2023 / standard references)
# ---------------------------------------------------------------------------

# (occupancy_norm_sqm, floors_default)
_BUILDING_NORMS: Dict[str, Tuple[float, int]] = {
    "hostel":           (15.0,  4),   # residential hostel @ 15 sqm/student
    "residential":      (60.0,  3),   # residential quarters @ 60 sqm/unit
    "staff_quarters":   (65.0,  4),   # Type-III/IV Govt quarters
    "school":           (8.0,   2),   # classroom block @ 8 sqm/student
    "academic":         (10.0,  4),   # college/university academic block
    "office":           (12.0,  5),   # office building @ 12 sqm/person
    "hospital":         (50.0,  3),   # hospital @ 50 sqm/bed
    "laboratory":       (20.0,  3),   # lab block
    "dining":           (2.5,   1),   # dining/kitchen ancillary
    "auditorium":       (1.5,   2),   # auditorium @ 1.5 sqm/seat
    "library":          (3.0,   3),   # library block
}

# Minimum credible area (sqm) — below this we treat estimate as unreliable
_MIN_AREA_SQM = 200.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SpecParams:
    """Extracted or derived build parameters from spec text."""
    floor_count: int = 1
    total_area_sqm: float = 0.0
    occupancy: int = 0                      # students / beds / seats
    building_types: List[str] = field(default_factory=list)
    source_text: str = ""                   # snippet that drove extraction
    confidence: float = 0.0                 # 0–1
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _search_all(pattern: str, text: str, flags: int = re.IGNORECASE) -> List[str]:
    return re.findall(pattern, text, flags)


def _extract_occupancy(text: str) -> Tuple[int, str]:
    """Return (count, matched_text) for the largest occupancy figure found."""
    patterns = [
        r"(\d{2,4})\s*(?:capacity|seated|seater|students?|beds?|seats?)",
        r"(?:capacity|seating)\s+of\s+(\d{2,4})",
        r"(\d{2,4})\s*nos?\b.*?(?:room|unit|quarter|flat)",
    ]
    best = (0, "")
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = int(m.group(1))
            if val > best[0]:
                best = (val, m.group(0))
    return best


def _extract_floors(text: str) -> Tuple[int, str]:
    """Return (floor_count_above_ground, matched_text)."""
    patterns = [
        r"[gG]\s*\+\s*(\d+)",                 # G+4, G +4
        r"[bB]\+[gG]\s*\+\s*(\d+)",           # B+G+10
        r"(\d+)\s*(?:storeyed?|storey(?:ed)?|floors?)\s+(?:building|block|hostel)",
        r"(?:stilt|basement)\s*\+\s*[gG]\s*\+\s*(\d+)",
        r"ground\s*\+\s*(\d+)",
    ]
    best = (0, "")
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = int(m.group(1)) + 1  # G+4 → 5 floors total
            if val > best[0]:
                best = (val, m.group(0))
    return best


def _extract_area_direct(text: str) -> Tuple[float, str]:
    """Return (sqm, matched_text) for explicitly stated plinth/built-up area."""
    patterns = [
        r"(?:plinth|built.?up|total|floor)\s*area[^0-9]{0,20}([\d,]+(?:\.\d+)?)\s*sq\.?\s*m",
        r"([\d,]+(?:\.\d+)?)\s*sq\.?\s*m\b",
        r"([\d,]+(?:\.\d+)?)\s*sqm\b",
        r"([\d,]+(?:\.\d+)?)\s*m2\b",
    ]
    best = (0.0, "")
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            # Sanity: ignore tiny values (<50) and huge values (>500,000)
            if 50 < val < 500_000 and val > best[0]:
                best = (val, m.group(0))
    return best


def _detect_building_types(text: str) -> List[str]:
    """Return list of detected building categories from scope text."""
    kw_map = [
        (r"\bhostel\b",                     "hostel"),
        (r"\bstaff\s+quarters?\b",           "staff_quarters"),
        (r"\bresidential\s+quarters?\b",     "residential"),
        (r"\btype.?[iIvV]+\s+quarters?\b",   "staff_quarters"),
        (r"\bdining\b|\bkitchen\b",          "dining"),
        (r"\bschool\b|\bemrs\b|\bnests\b",   "school"),
        (r"\bacademic\s+block\b",            "academic"),
        (r"\boffice\s+building\b|\boffice\s+complex\b", "office"),
        (r"\bhospital\b",                    "hospital"),
        (r"\blaborator\b",                   "laboratory"),
        (r"\bauditorium\b",                  "auditorium"),
        (r"\blibrary\b",                     "library"),
    ]
    found = []
    for pat, label in kw_map:
        if re.search(pat, text, re.IGNORECASE) and label not in found:
            found.append(label)
    return found or ["academic"]  # default to academic if nothing detected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_spec_params(
    page_texts: List[Tuple[int, str, str]],
    max_pages: int = 10,
) -> SpecParams:
    """
    Parse scope-of-work text from the first ``max_pages`` pages and return
    estimated build parameters.

    Parameters
    ----------
    page_texts : list of (page_idx, text, doc_type)
    max_pages  : how many pages to scan (front matter holds scope text)

    Returns
    -------
    SpecParams
    """
    params = SpecParams()

    # Concatenate first N pages
    sample_pages = [t for _, t, _ in page_texts[:max_pages]]
    combined = "\n".join(sample_pages)

    if not combined.strip():
        params.warnings.append("No text in first pages — PDF may be scanned.")
        return params

    # 1. Direct area statement
    area_direct, area_snippet = _extract_area_direct(combined)

    # 2. Floors
    floors, floor_snippet = _extract_floors(combined)

    # 3. Occupancy
    occupancy, occ_snippet = _extract_occupancy(combined)

    # 4. Building type(s)
    btypes = _detect_building_types(combined)
    params.building_types = btypes

    # 5. Derive area if not directly stated
    if area_direct >= _MIN_AREA_SQM:
        params.total_area_sqm = area_direct
        params.source_text = area_snippet
        params.confidence = 0.85
    elif occupancy > 0:
        # Use primary building type norm
        primary = btypes[0] if btypes else "academic"
        norm_sqm, norm_floors = _BUILDING_NORMS.get(primary, (12.0, 3))
        # Multi-building scopes: hostel + school + dining — sum norms
        derived = 0.0
        for bt in btypes:
            n, _ = _BUILDING_NORMS.get(bt, (norm_sqm, 1))
            if bt == "dining":
                # dining capacity ≈ 0.3× hostel capacity
                derived += int(occupancy * 0.3) * n
            else:
                derived += occupancy * n
        params.total_area_sqm = max(derived, _MIN_AREA_SQM)
        params.source_text = occ_snippet
        params.confidence = 0.55
        params.warnings.append(
            f"Area estimated from occupancy ({occupancy}) × norm ({norm_sqm} sqm); "
            "verify against actual drawings."
        )
    else:
        params.warnings.append(
            "Could not extract occupancy or area from scope text; "
            "BOQ generation will use placeholder area."
        )

    # 6. Floor count
    if floors >= 2:
        params.floor_count = floors
        if not params.source_text:
            params.source_text = floor_snippet
    elif params.total_area_sqm > 0:
        primary = btypes[0] if btypes else "academic"
        _, default_floors = _BUILDING_NORMS.get(primary, (12.0, 3))
        params.floor_count = default_floors
        params.warnings.append(
            f"Floor count not found in text; using default {default_floors} for '{primary}'."
        )
    else:
        params.floor_count = 1

    # 7. Occupancy
    params.occupancy = occupancy

    logger.info(
        "spec_boq_generator: types=%s floors=%d area=%.0f sqm occ=%d conf=%.0f%%",
        params.building_types, params.floor_count,
        params.total_area_sqm, params.occupancy,
        params.confidence * 100,
    )
    return params


def can_spec_generate(payload: dict) -> bool:
    """
    Return True when the payload has no BOQ items but scope text is present
    and no QTO modules already produced items (i.e., can_autofill would be False).
    """
    # Already has a BOQ — no need
    if payload.get("boq_items"):
        return False
    # Already has QTO items — autofill handles it
    qto_keys = ["structural_takeoff", "qto_data", "dw_takeoff",
                 "painting_result", "waterproofing_result", "mep_qto"]
    if any(payload.get(k) for k in qto_keys):
        return False
    # Needs scope text
    page_texts = payload.get("_page_texts_for_spec") or []
    if not page_texts:
        return False
    return True
