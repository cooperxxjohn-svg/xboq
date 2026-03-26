"""
Package Classifier — Stage 0 of the analysis pipeline.

Classifies the uploaded document package as:
  DRAWING_SET  — primarily construction drawings with real discipline sheets
  TENDER       — BOQ, conditions, specs, addenda; no real drawings
  MIXED        — both drawings AND tender documents in one upload
  INCOMPLETE   — too few pages to classify reliably

Key insight: Scanned pages with no text fall back to doc_type="plan" with
discipline="other" (the _drawing_fallback_sparse_text path in page_index.py).
These are NOT real drawings.  Only pages where:
  doc_type in DRAWING_TYPES  AND  discipline not in ("other", "unknown")
count as "real" drawing pages.

Usage:
    from .package_classifier import classify_package, PackageType
    classification = classify_package(page_index_result)
    if classification.package_type == PackageType.TENDER:
        ...
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Import PageIndex from page_index module
# ---------------------------------------------------------------------------
# Avoid circular imports by using TYPE_CHECKING guard for type hints only.
if TYPE_CHECKING:
    from .page_index import PageIndex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRAWING_TYPES: frozenset = frozenset({
    "plan", "section", "elevation", "detail", "notes", "legend", "schedule",
})

TENDER_TYPES: frozenset = frozenset({
    "boq", "conditions", "spec", "addendum",
})

# Disciplines that indicate a real (engineer-authored) drawing page.
# "other" and "unknown" are used for:
#   - scanned pages with no text (fallback from page_index._drawing_fallback_sparse_text)
#   - non-drawing pages forced to discipline="other" (boq, conditions, spec, addendum)
_NON_REAL_DISCIPLINES: frozenset = frozenset({"other", "unknown"})

_MIN_PAGES_FOR_CLASSIFICATION = 5


# ---------------------------------------------------------------------------
# Enums + Dataclasses
# ---------------------------------------------------------------------------

class PackageType(str, Enum):
    """Classification of the uploaded document package."""
    DRAWING_SET = "drawing_set"
    TENDER      = "tender"
    MIXED       = "mixed"
    INCOMPLETE  = "incomplete"


@dataclass
class PackageClassification:
    """Result of classifying an uploaded document package."""
    package_type: PackageType
    confidence: float
    drawing_page_ratio: float       # fraction of pages with doc_type in DRAWING_TYPES
    tender_page_ratio: float        # fraction of pages with doc_type in TENDER_TYPES
    real_drawing_ratio: float       # drawing pages where discipline is a real discipline
    has_boq: bool
    has_conditions: bool
    has_real_drawings: bool
    reasoning: str

    def is_tender(self) -> bool:
        return self.package_type == PackageType.TENDER

    def is_drawing_set(self) -> bool:
        return self.package_type == PackageType.DRAWING_SET

    def is_mixed(self) -> bool:
        return self.package_type == PackageType.MIXED

    def is_incomplete(self) -> bool:
        return self.package_type == PackageType.INCOMPLETE


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_package(page_index: "PageIndex") -> PackageClassification:
    """
    Classify the document package based on page index metadata.

    Args:
        page_index: A PageIndex (from build_page_index) with per-page
                    doc_type and discipline already determined.

    Returns:
        PackageClassification describing the package type, ratios, and reasoning.

    Threshold matrix
    ----------------
    total < 5                                            → INCOMPLETE (0.50)
    real_drawing_ratio < 0.05 AND tender_ratio > 0.35  → TENDER     (0.85)
    real_drawing_ratio > 0.50 AND tender_ratio < 0.10  → DRAWING_SET (0.90)
    real_drawing_ratio >= 0.15 AND tender_ratio >= 0.15 → MIXED      (0.75)
    real_drawing_ratio >= 0.05                          → DRAWING_SET (0.70)
    else                                                 → MIXED      (0.60)
    """
    pages = page_index.pages
    total = len(pages)

    # --- Counters ---
    drawing_pages = 0      # pages with doc_type in DRAWING_TYPES
    tender_pages  = 0      # pages with doc_type in TENDER_TYPES
    real_drawing_pages = 0 # drawing pages with a real (non-other/unknown) discipline
    has_boq        = False
    has_conditions = False

    for p in pages:
        dt   = (p.doc_type   or "unknown").lower()
        disc = (p.discipline or "unknown").lower()

        if dt in DRAWING_TYPES:
            drawing_pages += 1
            if disc not in _NON_REAL_DISCIPLINES:
                real_drawing_pages += 1
        elif dt in TENDER_TYPES:
            tender_pages += 1
            if dt == "boq":
                has_boq = True
            if dt == "conditions":
                has_conditions = True

    # --- Ratios (guard against zero-page edge case) ---
    denom             = max(total, 1)
    drawing_ratio     = drawing_pages / denom
    tender_ratio      = tender_pages  / denom
    real_drawing_ratio = real_drawing_pages / denom
    has_real_drawings  = real_drawing_pages > 0

    # --- Classification ---
    if total < _MIN_PAGES_FOR_CLASSIFICATION:
        return PackageClassification(
            package_type=PackageType.INCOMPLETE,
            confidence=0.50,
            drawing_page_ratio=drawing_ratio,
            tender_page_ratio=tender_ratio,
            real_drawing_ratio=real_drawing_ratio,
            has_boq=has_boq,
            has_conditions=has_conditions,
            has_real_drawings=has_real_drawings,
            reasoning=(
                f"Only {total} pages — too few to classify reliably "
                f"(threshold: {_MIN_PAGES_FOR_CLASSIFICATION})."
            ),
        )

    if real_drawing_ratio < 0.05 and tender_ratio > 0.35:
        pkg_type   = PackageType.TENDER
        confidence = 0.85
        reasoning  = (
            f"real_drawing_ratio={real_drawing_ratio:.2f} (<0.05) and "
            f"tender_ratio={tender_ratio:.2f} (>0.35). "
            f"Package is a tender/conditions/BOQ bundle without drawing sheets."
        )

    elif real_drawing_ratio > 0.50 and tender_ratio < 0.10:
        pkg_type   = PackageType.DRAWING_SET
        confidence = 0.90
        reasoning  = (
            f"real_drawing_ratio={real_drawing_ratio:.2f} (>0.50) and "
            f"tender_ratio={tender_ratio:.2f} (<0.10). "
            f"Package is predominantly construction drawings."
        )

    elif real_drawing_ratio >= 0.15 and tender_ratio >= 0.15:
        pkg_type   = PackageType.MIXED
        confidence = 0.75
        reasoning  = (
            f"real_drawing_ratio={real_drawing_ratio:.2f} (>=0.15) and "
            f"tender_ratio={tender_ratio:.2f} (>=0.15). "
            f"Package contains both drawings and tender documents."
        )

    elif real_drawing_ratio >= 0.05:
        pkg_type   = PackageType.DRAWING_SET
        confidence = 0.70
        reasoning  = (
            f"real_drawing_ratio={real_drawing_ratio:.2f} (>=0.05) — "
            f"some real drawing sheets found; classified as drawing set. "
            f"tender_ratio={tender_ratio:.2f}."
        )

    else:
        # Low real drawings AND low tender content — ambiguous, default to MIXED
        pkg_type   = PackageType.MIXED
        confidence = 0.60
        reasoning  = (
            f"real_drawing_ratio={real_drawing_ratio:.2f} (<0.05) and "
            f"tender_ratio={tender_ratio:.2f} (<= 0.35). "
            f"Cannot confidently classify; defaulting to MIXED."
        )

    # Annotate with document presence
    docs_found = []
    if has_boq:        docs_found.append("BOQ")
    if has_conditions: docs_found.append("Conditions")
    if has_real_drawings: docs_found.append(f"{real_drawing_pages} real drawing pages")

    if docs_found:
        reasoning += f"  Documents detected: {', '.join(docs_found)}."

    return PackageClassification(
        package_type=pkg_type,
        confidence=confidence,
        drawing_page_ratio=drawing_ratio,
        tender_page_ratio=tender_ratio,
        real_drawing_ratio=real_drawing_ratio,
        has_boq=has_boq,
        has_conditions=has_conditions,
        has_real_drawings=has_real_drawings,
        reasoning=reasoning,
    )
