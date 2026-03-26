"""
Page Selection — Intelligent page prioritization within OCR budget.

Uses the PageIndex to select pages for full OCR:
- Tier 1 (always): cover, index, legend, notes, schedule, boq, conditions, addendum
- Tier 2 (fill budget): plan, detail, section, elevation — round-robin across disciplines
- Tier 3 (remainder): unknown, spec — sample

Ensures the most valuable pages are OCR'd first within a fixed budget.

Sprint 20G: Accepts deep_cap parameter (None = unlimited) to support
DEMO_FAST (80), STANDARD_REVIEW (220), and FULL_AUDIT (all pages) modes.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict

from .page_index import PageIndex, IndexedPage


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class SelectedPages:
    """Result of page selection with audit trail."""
    selected: List[int] = field(default_factory=list)     # page indices
    reasons: Dict[int, str] = field(default_factory=dict)  # page_idx -> reason
    budget_total: int = 80
    budget_used: int = 0
    always_include_count: int = 0
    sample_count: int = 0
    skipped_types: Dict[str, int] = field(default_factory=dict)
    coverage_summary: Dict[str, int] = field(default_factory=dict)
    selection_mode: str = "fast_budget"  # "full_read" or "fast_budget"

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "reasons": {str(k): v for k, v in self.reasons.items()},
            "budget_total": self.budget_total,
            "budget_used": self.budget_used,
            "always_include_count": self.always_include_count,
            "sample_count": self.sample_count,
            "skipped_types": self.skipped_types,
            "coverage_summary": self.coverage_summary,
            "selection_mode": self.selection_mode,
        }

    def summary_line(self) -> str:
        """Human-readable one-liner for progress messages."""
        if self.selection_mode == "full_read":
            mode = "FULL_READ"
        elif self.selection_mode == "force_text":
            mode = "FORCE_TEXT"
        else:
            mode = "FAST_BUDGET"
        return (
            f"[{mode}] Selected {self.budget_used}/{self.budget_total} pages "
            f"({self.always_include_count} must-read + "
            f"{self.sample_count} sampled)"
        )


# =============================================================================
# TIER DEFINITIONS
# =============================================================================

# Tier 1: Always include (high-value non-drawing pages)
TIER_1_ALWAYS = {"legend", "notes", "schedule", "boq", "conditions", "addendum"}

# Tier 1 priority order — highest-value types processed first so they are
# never displaced by lower-value types that happen to appear earlier in page
# order (e.g. 103 conditions pages filling the budget before 2 BOQ pages).
TIER_1_PRIORITY = ["boq", "addendum", "schedule", "conditions", "notes", "legend"]

# Types that are NEVER sub-capped — always fully included regardless of count.
TIER_1_NO_CAP = {"boq", "addendum", "schedule"}

# Sub-caps for massive Tier 1 types — evenly sampled when total exceeds cap.
TIER_1_SUB_CAPS = {
    "conditions": 20,
    "notes": 15,
    "legend": 10,
}

# Tier 1 with cap: cover/index (max 5 total)
TIER_1_CAPPED = {"cover", "index"}
TIER_1_CAP = 5

# Tier 2: Drawing pages — round-robin across disciplines
TIER_2_DRAWINGS = {"plan", "detail", "section", "elevation"}

# Tier 2 bonus: spec pages (first N after drawings)
TIER_2_SPEC = {"spec"}

# Tier 3: Unknown pages — sample from remainder
TIER_3_SAMPLE = {"unknown"}


# =============================================================================
# SELECTION LOGIC
# =============================================================================

# Auto-FULL_READ for tenders up to this many pages
SAFE_CAP_FULL_READ = 120


def select_pages(
    page_index: PageIndex,
    budget_pages: int = 80,
    force_full_read: bool = False,
    deep_cap: Optional[int] = None,
    force_text_pages: bool = False,
) -> SelectedPages:
    """
    Select pages for full OCR within a budget.

    Modes:
        FULL_READ  — Every page is deep-processed. Auto-selected when
                     total_pages <= SAFE_CAP_FULL_READ, or when
                     force_full_read=True.
        FAST_BUDGET — Tiered selection within budget_pages cap.

    Args:
        page_index: The PageIndex from build_page_index().
        budget_pages: Maximum pages to select (FAST_BUDGET only).
                      Ignored when deep_cap is provided.
        force_full_read: If True, select ALL pages regardless of count.
        deep_cap: Sprint 20G — explicit page budget from RunMode.
                  Overrides budget_pages when provided.
                  Use force_full_read=True for unlimited (FULL_AUDIT).
        force_text_pages: If True (for TENDER/MIXED packages), promotes ALL
                          text-bearing page types (spec, conditions, notes,
                          boq, addendum, schedule, legend) to Tier 1 with no
                          sub-cap, so every text page is processed regardless
                          of the drawing budget. Drawing pages remain capped.

    Returns:
        SelectedPages with selected indices and reasons.
    """
    # Sprint 20G: deep_cap overrides budget_pages
    if deep_cap is not None:
        budget_pages = deep_cap

    # --- FULL_READ auto-detection ---
    if force_full_read or page_index.total_pages <= SAFE_CAP_FULL_READ:
        result = SelectedPages(
            budget_total=page_index.total_pages,
            selection_mode="full_read",
        )
        for p in page_index.pages:
            result.selected.append(p.page_idx)
            result.reasons[p.page_idx] = "full_read"
        result.selected = sorted(result.selected)
        result.budget_used = len(result.selected)
        result.always_include_count = result.budget_used
        result.coverage_summary = _build_coverage(result.selected, page_index)
        return result

    # --- FAST_BUDGET mode ---
    _mode = "force_text" if force_text_pages else "fast_budget"
    result = SelectedPages(budget_total=budget_pages, selection_mode=_mode)
    selected_set: set = set()

    # ── force_text_pages: promote all text types to uncapped Tier 1 ──────
    # Used for TENDER / MIXED packages where every spec/conditions/notes page
    # contains valuable priceable content and must not be budget-capped.
    _TEXT_TYPES = {"spec", "conditions", "notes", "boq", "addendum", "schedule", "legend"}
    if force_text_pages:
        _effective_no_cap       = TIER_1_NO_CAP | _TEXT_TYPES
        _effective_sub_caps: dict = {}
        # Include "spec" in the priority pass so it gets Tier 1 treatment
        _effective_tier1_priority = TIER_1_PRIORITY + ["spec"]
    else:
        _effective_no_cap         = TIER_1_NO_CAP
        _effective_sub_caps       = TIER_1_SUB_CAPS
        _effective_tier1_priority = TIER_1_PRIORITY

    # ── Phase 1: Tier 1 — collect ALL must-read pages WITHOUT budget cap ──
    # Process in priority order (boq first) so high-value types are never
    # displaced by massive lower-value types (e.g. 103 conditions pages).
    for doc_type in _effective_tier1_priority:
        pages_of_type = [p for p in page_index.pages if p.doc_type == doc_type]
        if not pages_of_type:
            continue

        sub_cap = _effective_sub_caps.get(doc_type)
        if doc_type in _effective_no_cap or sub_cap is None or len(pages_of_type) <= sub_cap:
            # Include ALL pages of this type (no cap)
            for p in pages_of_type:
                if p.page_idx not in selected_set:
                    selected_set.add(p.page_idx)
                    result.reasons[p.page_idx] = f"tier1-always ({doc_type})"
        else:
            # Sub-cap: evenly sample within the cap
            step = max(1, len(pages_of_type) // sub_cap)
            sampled = pages_of_type[::step][:sub_cap]
            for p in sampled:
                if p.page_idx not in selected_set:
                    selected_set.add(p.page_idx)
                    result.reasons[p.page_idx] = (
                        f"tier1-sampled ({doc_type}, {len(sampled)}/{len(pages_of_type)})"
                    )

    # Also collect any Tier 1 types not in the priority list (defensive).
    # Skip types already processed by the priority loop above.
    _processed_types = set(_effective_tier1_priority)
    for p in page_index.pages:
        if (p.doc_type in TIER_1_ALWAYS
                and p.doc_type not in _processed_types
                and p.page_idx not in selected_set):
            selected_set.add(p.page_idx)
            result.reasons[p.page_idx] = f"tier1-always ({p.doc_type})"

    # Tier 1 with cap: cover/index (max 5)
    cover_index_count = 0
    for p in page_index.pages:
        if p.doc_type in TIER_1_CAPPED and cover_index_count < TIER_1_CAP:
            if p.page_idx not in selected_set:
                selected_set.add(p.page_idx)
                result.reasons[p.page_idx] = f"tier1-capped ({p.doc_type})"
            cover_index_count += 1

    result.always_include_count = len(selected_set)

    # If Tier 1 alone exceeds the original budget, expand budget to fit.
    effective_budget = max(budget_pages, len(selected_set))

    # ── Phase 2: Tier 2 + 3 — fill remaining budget ──

    def _add_budgeted(page_idx: int, reason: str):
        """Add a page to selection if within effective budget."""
        if page_idx in selected_set:
            return
        if len(selected_set) >= effective_budget:
            return
        selected_set.add(page_idx)
        result.reasons[page_idx] = reason

    # --- Tier 2: Drawing pages — round-robin across disciplines ---
    remaining_budget = effective_budget - len(selected_set)
    if remaining_budget > 0:
        drawing_pages = [
            p for p in page_index.pages
            if p.doc_type in TIER_2_DRAWINGS and p.page_idx not in selected_set
        ]

        # Group by discipline
        by_discipline: Dict[str, List[IndexedPage]] = defaultdict(list)
        for p in drawing_pages:
            by_discipline[p.discipline].append(p)

        # Round-robin across disciplines
        disciplines = sorted(by_discipline.keys())
        if disciplines:
            pointers = {d: 0 for d in disciplines}
            drawings_added = 0

            while drawings_added < remaining_budget:
                progress_made = False
                for disc in disciplines:
                    pages_list = by_discipline[disc]
                    ptr = pointers[disc]
                    if ptr < len(pages_list):
                        p = pages_list[ptr]
                        _add_budgeted(p.page_idx, f"tier2-drawing ({p.doc_type}, {disc})")
                        pointers[disc] = ptr + 1
                        drawings_added += 1
                        progress_made = True
                        if len(selected_set) >= effective_budget:
                            break
                if not progress_made or len(selected_set) >= effective_budget:
                    break

    # --- Tier 2 bonus: spec pages (fill remaining) ---
    remaining_budget = effective_budget - len(selected_set)
    if remaining_budget > 0:
        spec_pages = [
            p for p in page_index.pages
            if p.doc_type in TIER_2_SPEC and p.page_idx not in selected_set
        ]
        for p in spec_pages[:remaining_budget]:
            _add_budgeted(p.page_idx, f"tier2-spec ({p.doc_type})")

    # --- Tier 3: Unknown pages (sample from remainder) ---
    remaining_budget = effective_budget - len(selected_set)
    if remaining_budget > 0:
        unknown_pages = [
            p for p in page_index.pages
            if p.doc_type in TIER_3_SAMPLE and p.page_idx not in selected_set
        ]
        if unknown_pages:
            step = max(1, len(unknown_pages) // max(remaining_budget, 1))
            sampled = unknown_pages[::step][:remaining_budget]
            for p in sampled:
                _add_budgeted(p.page_idx, f"tier3-sample ({p.doc_type})")
                result.sample_count += 1

    # Build result
    result.selected = sorted(selected_set)
    result.budget_used = len(result.selected)
    result.budget_total = effective_budget

    # Track skipped types
    all_types = defaultdict(int)
    selected_types = defaultdict(int)
    for p in page_index.pages:
        all_types[p.doc_type] += 1
    for idx in result.selected:
        if idx < len(page_index.pages):
            selected_types[page_index.pages[idx].doc_type] += 1
    for dtype, total in all_types.items():
        skipped = total - selected_types.get(dtype, 0)
        if skipped > 0:
            result.skipped_types[dtype] = skipped

    result.coverage_summary = _build_coverage(result.selected, page_index)
    return result


def _build_coverage(selected: List[int], page_index: PageIndex) -> Dict[str, int]:
    """Build coverage summary: how many of each type were selected."""
    coverage: Dict[str, int] = defaultdict(int)
    for idx in selected:
        if idx < len(page_index.pages):
            coverage[page_index.pages[idx].doc_type] += 1
    return dict(coverage)
