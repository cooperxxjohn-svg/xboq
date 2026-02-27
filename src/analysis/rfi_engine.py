"""
RFI Engine — Checklist-driven RFI generation with evidence.

Evaluates a discipline checklist (~30-50 items) against:
- ExtractionResult (requirements, schedules, BOQ items, callouts)
- PageIndex (page classification counts)
- SelectedPages (what was analyzed)
- PlanSetGraph (entities, aggregates)

Two check types:
1. Missing-field checks: required info not found
2. Conflict checks: contradictions across pages

Every RFI uses the existing RFIItem model with full EvidenceRef.
Target: >15 RFIs on a 300+ page tender.
"""

import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from src.models.analysis_models import (
    RFIItem, EvidenceRef, Trade, Severity,
    PlanSetGraph, create_rfi_id,
    RunCoverage, CoverageStatus, SelectionMode,
)

from .page_index import PageIndex
from .page_selection import SelectedPages
from .extractors import ExtractionResult

DEBUG = False


# =============================================================================
# CHECKLIST DEFINITION
# =============================================================================

# Each check: (check_id, check_fn_name, trade, default_priority, question_template, why_template)
# check_fn returns: (should_fire: bool, evidence: EvidenceRef)
CHECKLIST = [
    # ---- Architectural ----
    ("CHK-A-001", "chk_door_schedule_missing", Trade.ARCHITECTURAL, Severity.HIGH,
     "Door schedule is missing — {n_tags} door tags found on plan sheets but no door schedule detected.",
     "Cannot price doors accurately without schedule specifying sizes, types, hardware, and materials."),

    ("CHK-A-002", "chk_window_schedule_missing", Trade.ARCHITECTURAL, Severity.HIGH,
     "Window schedule is missing — {n_tags} window tags found but no window schedule detected.",
     "Cannot price windows without schedule specifying sizes, glazing type, frame material, and hardware."),

    ("CHK-A-003", "chk_finish_schedule_missing", Trade.ARCHITECTURAL, Severity.HIGH,
     "Finish schedule is missing — {n_rooms} rooms found but no finish schedule detected.",
     "Cannot price finishes (flooring, wall, ceiling) without knowing materials per room."),

    ("CHK-A-004", "chk_no_general_notes", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "No general notes page found in the drawing set.",
     "General notes specify materials, workmanship, and code compliance. Without them, assumptions are needed."),

    ("CHK-A-005", "chk_no_legend", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "No legend/symbols page found in the drawing set.",
     "Legend defines drawing conventions. Missing legend may cause misinterpretation of symbols."),

    ("CHK-A-006", "chk_room_dimensions_missing", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "Room dimensions not found — {n_rooms} rooms detected but {n_dims} dimension callouts across plan pages.",
     "Area calculations need room dimensions. Missing dimensions force assumptions on carpet area."),

    # ---- Structural ----
    ("CHK-S-001", "chk_no_structural", Trade.STRUCTURAL, Severity.HIGH,
     "No structural drawings found in the set.",
     "Cannot price RCC, steel, or foundation work without structural drawings."),

    ("CHK-S-002", "chk_no_foundation", Trade.STRUCTURAL, Severity.MEDIUM,
     "No foundation details found.",
     "Foundation type, depth, and size directly impact earthwork and RCC quantities."),

    ("CHK-S-003", "chk_bbs_missing", Trade.STRUCTURAL, Severity.MEDIUM,
     "Bar bending schedule (BBS) not found.",
     "Steel reinforcement is a major cost item. BBS needed for accurate rebar quantities."),

    # ---- MEP ----
    ("CHK-M-001", "chk_no_mep", Trade.MEP, Severity.MEDIUM,
     "No MEP drawings found (mechanical, electrical, or plumbing).",
     "MEP is typically 25-40% of project cost. Missing MEP drawings means entire trade must use allowances."),

    ("CHK-M-002", "chk_no_electrical", Trade.ELECTRICAL, Severity.MEDIUM,
     "No electrical layout found.",
     "Cannot price wiring, fixtures, panels, or DBs without electrical drawings."),

    ("CHK-M-003", "chk_no_plumbing", Trade.PLUMBING, Severity.MEDIUM,
     "No plumbing layout found.",
     "Cannot price water supply, drainage, or sanitary fixtures without plumbing drawings."),

    ("CHK-M-004", "chk_no_hvac", Trade.MEP, Severity.LOW,
     "No HVAC layout found.",
     "HVAC provision may need to be carried as allowance if drawings not provided."),

    ("CHK-M-005", "chk_no_fire", Trade.MEP, Severity.MEDIUM,
     "No fire fighting/protection layout found.",
     "Fire safety system cost can be significant. Missing layout requires provisional allowance."),

    # ---- Cross-discipline ----
    ("CHK-X-001", "chk_scale_missing", Trade.GENERAL, Severity.MEDIUM,
     "Scale notation missing on {n_missing}/{n_total} drawing pages.",
     "Without scale, dimensions cannot be verified and area takeoffs become unreliable."),

    ("CHK-X-002", "chk_no_site_plan", Trade.CIVIL, Severity.MEDIUM,
     "No site plan found.",
     "Site plan needed for external works, approach roads, drainage, and compound wall quantities."),

    ("CHK-X-003", "chk_no_sections", Trade.GENERAL, Severity.MEDIUM,
     "No section drawings found.",
     "Sections provide floor-to-floor heights, slab thickness, beam depths — critical for structural quantities."),

    ("CHK-X-004", "chk_no_elevations", Trade.ARCHITECTURAL, Severity.LOW,
     "No elevation drawings found.",
     "Elevations define external finishes, window positions, and facade materials."),

    ("CHK-X-005", "chk_no_boq", Trade.GENERAL, Severity.HIGH,
     "No Bill of Quantities (BOQ) found in the tender documents.",
     "BOQ is the pricing basis. Without it, the bid format is unclear."),

    ("CHK-X-006", "chk_boq_missing_quantities", Trade.GENERAL, Severity.HIGH,
     "BOQ items found but {n_missing}/{n_total} items have missing quantities.",
     "Items without quantities cannot be priced. Need clarification on quantities."),

    ("CHK-X-007", "chk_no_conditions", Trade.GENERAL, Severity.MEDIUM,
     "No tender conditions/contract conditions found.",
     "Contract conditions define payment terms, retention, defect liability, and escalation clauses."),

    ("CHK-X-008", "chk_no_addendum", Trade.GENERAL, Severity.LOW,
     "No addendum/corrigendum found. This may or may not be an issue.",
     "Check if any addenda were issued. Missing addendum could mean bidding on outdated information."),

    # ---- Schedule field checks ----
    ("CHK-SCH-001", "chk_schedule_missing_sizes", Trade.ARCHITECTURAL, Severity.HIGH,
     "{n_missing} schedule items have marks but no sizes specified.",
     "Doors/windows cannot be priced without knowing their sizes."),

    ("CHK-SCH-002", "chk_schedule_missing_qty", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "{n_missing} schedule items have marks but no quantities specified.",
     "Need quantities to calculate total cost per door/window type."),

    # ---- Conflict checks ----
    ("CHK-C-001", "chk_conflicting_material_specs", Trade.GENERAL, Severity.HIGH,
     "Conflicting material specifications found: {conflicts}.",
     "Contradictory specs create ambiguity. Need clarification on which spec to follow."),

    ("CHK-C-002", "chk_duplicate_tags_different_sizes", Trade.ARCHITECTURAL, Severity.HIGH,
     "Same tag appears with different sizes: {conflicts}.",
     "Contradictory sizes for same door/window tag create pricing ambiguity."),

    # ---- Commercial / Contract ----
    ("CHK-COM-001", "chk_no_ld_clause", Trade.COMMERCIAL, Severity.HIGH,
     "No liquidated damages (LD) clause found in the tender conditions.",
     "LD rate directly affects bid risk pricing. Without it, penalty exposure is unknown."),

    ("CHK-COM-002", "chk_no_retention", Trade.COMMERCIAL, Severity.HIGH,
     "No retention clause found in the tender conditions.",
     "Retention percentage impacts cash flow projections. Typically 5-10% of each RA bill."),

    ("CHK-COM-003", "chk_no_warranty_dlp", Trade.COMMERCIAL, Severity.MEDIUM,
     "No warranty / defect liability period (DLP) found in the tender conditions.",
     "DLP duration affects mobilization of maintenance teams and bank guarantee costs."),

    ("CHK-COM-004", "chk_no_bid_validity", Trade.COMMERCIAL, Severity.MEDIUM,
     "No bid validity period found in the tender conditions.",
     "Bid validity affects pricing — longer validity increases material price escalation risk."),

    ("CHK-COM-005", "chk_no_emd", Trade.COMMERCIAL, Severity.HIGH,
     "No EMD / earnest money deposit / bid security amount found.",
     "EMD amount must be arranged before submission. Missing amount blocks bid preparation."),

    ("CHK-COM-006", "chk_no_performance_bond", Trade.COMMERCIAL, Severity.MEDIUM,
     "No performance bank guarantee (PBG) percentage found.",
     "PBG percentage affects bank charges and working capital planning."),

    ("CHK-COM-007", "chk_no_mobilization_advance", Trade.COMMERCIAL, Severity.LOW,
     "No mobilization advance clause found.",
     "Mobilization advance availability affects initial cash flow and site setup planning."),

    ("CHK-COM-008", "chk_no_insurance", Trade.COMMERCIAL, Severity.MEDIUM,
     "No insurance / CAR policy requirement found in the tender conditions.",
     "Insurance cost is a direct bid line item. Unknown requirements force assumptions."),

    ("CHK-COM-009", "chk_no_escalation", Trade.COMMERCIAL, Severity.MEDIUM,
     "No price escalation / variation clause found in the tender conditions.",
     "Without escalation clause, all material price risk falls on the contractor."),

    ("CHK-COM-010", "chk_ld_rate_high", Trade.COMMERCIAL, Severity.HIGH,
     "LD rate of {ld_rate}% {cadence_str}appears high — exceeds 1% threshold.",
     "High LD rate significantly increases project risk. Verify if rate is correct."),
]


# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def _make_evidence(
    pages: List[int] = None,
    snippets: List[str] = None,
    detected_entities: Dict[str, Any] = None,
    search_attempts: Dict[str, Any] = None,
    confidence: float = 0.6,
    confidence_reason: str = "",
    budget: int = 80,
) -> EvidenceRef:
    """Helper to construct EvidenceRef."""
    if not confidence_reason and confidence < 0.7:
        confidence_reason = (
            f"Not found within analyzed pages (OCR cap {budget}); "
            "may exist elsewhere in the set."
        )
    return EvidenceRef(
        pages=pages or [],
        snippets=snippets or [],
        detected_entities=detected_entities or {},
        search_attempts=search_attempts or {},
        confidence=confidence,
        confidence_reason=confidence_reason,
    )


class CheckContext:
    """All data needed by check functions."""
    def __init__(
        self,
        extracted: ExtractionResult,
        page_index: PageIndex,
        selected: SelectedPages,
        plan_graph: Optional[PlanSetGraph],
        run_coverage: Optional[RunCoverage] = None,
    ):
        self.extracted = extracted
        self.page_index = page_index
        self.selected = selected
        self.plan_graph = plan_graph
        self.run_coverage = run_coverage
        self.budget = selected.budget_total if selected else 80

        # Pre-compute useful aggregates
        self.type_counts = page_index.counts_by_type if page_index else {}
        self.disc_counts = page_index.counts_by_discipline if page_index else {}

        # Tags from plan_graph
        self.door_tags = set(plan_graph.all_door_tags) if plan_graph else set()
        self.window_tags = set(plan_graph.all_window_tags) if plan_graph else set()
        self.room_names = set(plan_graph.all_room_names) if plan_graph else set()

        # Tags from callouts
        for c in extracted.callouts:
            if c.get("callout_type") == "tag":
                tag = c["text"]
                if tag.startswith(("D", "DR")):
                    self.door_tags.add(tag)
                elif tag.startswith(("W", "WN")):
                    self.window_tags.add(tag)

        # Schedule info
        self.schedule_types_found: Set[str] = set()
        for s in extracted.schedules:
            self.schedule_types_found.add(s.get("schedule_type", "unknown"))

        # Drawing pages with dimensions
        self.dimension_callouts = [
            c for c in extracted.callouts if c.get("callout_type") == "dimension"
        ]

        # Material callouts with grades
        self.material_callouts = [
            c for c in extracted.callouts if c.get("callout_type") == "material"
        ]

        # Commercial terms (Sprint 19)
        self.commercial_terms = getattr(extracted, 'commercial_terms', [])
        self.commercial_term_types = {t.get("term_type") for t in self.commercial_terms}

    def is_covered(self, *doc_types: str) -> CoverageStatus:
        """Check if the given doc_types were fully covered during extraction."""
        if not self.run_coverage or self.run_coverage.selection_mode == SelectionMode.FULL_READ:
            return CoverageStatus.NOT_FOUND_AFTER_SEARCH
        for dt in doc_types:
            if self.run_coverage.is_doc_type_covered(dt) == CoverageStatus.UNKNOWN_NOT_PROCESSED:
                return CoverageStatus.UNKNOWN_NOT_PROCESSED
        return CoverageStatus.NOT_FOUND_AFTER_SEARCH


# =============================================================================
# CHECK-TO-DOC_TYPE MAPPING (for coverage gating)
# =============================================================================

_CHECK_DOC_TYPES: Dict[str, List[str]] = {
    "CHK-A-001": ["schedule"],      # door schedule
    "CHK-A-002": ["schedule"],      # window schedule
    "CHK-A-003": ["schedule"],      # finish schedule
    "CHK-A-004": ["notes"],         # general notes
    "CHK-A-005": ["legend"],        # legend
    "CHK-S-003": ["notes", "schedule"],  # BBS
    "CHK-X-005": ["boq"],           # BOQ
    "CHK-X-007": ["conditions"],    # tender conditions
    "CHK-X-008": ["addendum"],      # addendum
    "CHK-SCH-001": ["schedule"],    # schedule sizes
    "CHK-SCH-002": ["schedule"],    # schedule quantities
    # Commercial checks — need conditions/spec pages
    "CHK-COM-001": ["conditions", "spec"],
    "CHK-COM-002": ["conditions", "spec"],
    "CHK-COM-003": ["conditions", "spec"],
    "CHK-COM-004": ["conditions", "spec"],
    "CHK-COM-005": ["conditions", "spec"],
    "CHK-COM-006": ["conditions", "spec"],
    "CHK-COM-007": ["conditions", "spec"],
    "CHK-COM-008": ["conditions", "spec"],
    "CHK-COM-009": ["conditions", "spec"],
    "CHK-COM-010": ["conditions", "spec"],
}


# --- Individual check functions ---
# Each returns (should_fire: bool, evidence: EvidenceRef, format_kwargs: dict)

def chk_door_schedule_missing(ctx: CheckContext):
    n_tags = len(ctx.door_tags)
    has_schedule = "door" in ctx.schedule_types_found
    has_graph_schedule = ctx.plan_graph.has_door_schedule if ctx.plan_graph else False

    if n_tags > 0 and not has_schedule and not has_graph_schedule:
        pages_with_tags = list({c["source_page"] for c in ctx.extracted.callouts
                                if c.get("callout_type") == "tag" and c["text"].startswith(("D", "DR"))})
        ev = _make_evidence(
            pages=pages_with_tags[:10],
            detected_entities={"door_tags": sorted(ctx.door_tags)[:20]},
            search_attempts={"searched_for": "door schedule page", "schedule_types_found": list(ctx.schedule_types_found)},
            confidence=0.85,
            confidence_reason=f"Found {n_tags} door tags but no door schedule page.",
            budget=ctx.budget,
        )
        return True, ev, {"n_tags": n_tags}
    return False, None, {}


def chk_window_schedule_missing(ctx: CheckContext):
    n_tags = len(ctx.window_tags)
    has_schedule = "window" in ctx.schedule_types_found
    has_graph_schedule = ctx.plan_graph.has_window_schedule if ctx.plan_graph else False

    if n_tags > 0 and not has_schedule and not has_graph_schedule:
        pages_with_tags = list({c["source_page"] for c in ctx.extracted.callouts
                                if c.get("callout_type") == "tag" and c["text"].startswith(("W", "WN"))})
        ev = _make_evidence(
            pages=pages_with_tags[:10],
            detected_entities={"window_tags": sorted(ctx.window_tags)[:20]},
            search_attempts={"searched_for": "window schedule page"},
            confidence=0.85,
            confidence_reason=f"Found {n_tags} window tags but no window schedule page.",
            budget=ctx.budget,
        )
        return True, ev, {"n_tags": n_tags}
    return False, None, {}


def chk_finish_schedule_missing(ctx: CheckContext):
    n_rooms = len(ctx.room_names)
    has_schedule = "finish" in ctx.schedule_types_found
    has_graph_schedule = ctx.plan_graph.has_finish_schedule if ctx.plan_graph else False

    if n_rooms > 0 and not has_schedule and not has_graph_schedule:
        ev = _make_evidence(
            detected_entities={"room_names": sorted(ctx.room_names)[:20]},
            search_attempts={"searched_for": "finish schedule page"},
            confidence=0.75,
            confidence_reason=f"Found {n_rooms} rooms but no finish schedule.",
            budget=ctx.budget,
        )
        return True, ev, {"n_rooms": n_rooms}
    return False, None, {}


def chk_no_general_notes(ctx: CheckContext):
    if ctx.type_counts.get("notes", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "general notes page", "pages_indexed": ctx.page_index.total_pages},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_legend(ctx: CheckContext):
    if ctx.type_counts.get("legend", 0) == 0:
        has_graph_legend = ctx.plan_graph.has_legend if ctx.plan_graph else False
        if not has_graph_legend:
            ev = _make_evidence(
                search_attempts={"searched_for": "legend/symbols page"},
                confidence=0.65,
                budget=ctx.budget,
            )
            return True, ev, {}
    return False, None, {}


def chk_room_dimensions_missing(ctx: CheckContext):
    n_rooms = len(ctx.room_names)
    n_dims = len(ctx.dimension_callouts)
    # Rough heuristic: if we have rooms but very few dimensions
    if n_rooms > 2 and n_dims < n_rooms:
        ev = _make_evidence(
            detected_entities={"rooms": n_rooms, "dimension_callouts": n_dims},
            search_attempts={"searched_for": "dimension callouts on plan pages"},
            confidence=0.5,
            budget=ctx.budget,
        )
        return True, ev, {"n_rooms": n_rooms, "n_dims": n_dims}
    return False, None, {}


def chk_no_structural(ctx: CheckContext):
    if ctx.disc_counts.get("structural", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "structural drawings", "disciplines_found": list(ctx.disc_counts.keys())},
            confidence=0.8,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_foundation(ctx: CheckContext):
    # Check if any page has foundation keywords
    has_foundation = any(
        p.doc_type == "plan" and "foundation" in (p.title or "").lower()
        for p in ctx.page_index.pages
    )
    if not has_foundation and ctx.disc_counts.get("structural", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "foundation details/layout"},
            confidence=0.6,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_bbs_missing(ctx: CheckContext):
    has_structural = ctx.disc_counts.get("structural", 0) > 0
    # Search for BBS in requirements
    has_bbs = any(
        "bar bending" in r.get("text", "").lower() or "bbs" in r.get("text", "").lower()
        for r in ctx.extracted.requirements
    )
    if has_structural and not has_bbs:
        ev = _make_evidence(
            search_attempts={"searched_for": "bar bending schedule (BBS)", "structural_pages": ctx.disc_counts.get("structural", 0)},
            confidence=0.55,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_mep(ctx: CheckContext):
    mep_count = (
        ctx.disc_counts.get("mechanical", 0) +
        ctx.disc_counts.get("electrical", 0) +
        ctx.disc_counts.get("plumbing", 0)
    )
    if mep_count == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "MEP drawings (M/E/P prefixes)", "disciplines_found": list(ctx.disc_counts.keys())},
            confidence=0.75,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_electrical(ctx: CheckContext):
    if ctx.disc_counts.get("electrical", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "electrical layout drawings"},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_plumbing(ctx: CheckContext):
    if ctx.disc_counts.get("plumbing", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "plumbing/drainage layout drawings"},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_hvac(ctx: CheckContext):
    if ctx.disc_counts.get("mechanical", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "HVAC/mechanical layout drawings"},
            confidence=0.5,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_fire(ctx: CheckContext):
    has_fire = ctx.disc_counts.get("fire", 0) > 0
    # Also check in requirements/callouts
    fire_in_req = any("fire" in r.get("text", "").lower() for r in ctx.extracted.requirements)
    if not has_fire and not fire_in_req:
        ev = _make_evidence(
            search_attempts={"searched_for": "fire fighting/protection layout"},
            confidence=0.55,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_scale_missing(ctx: CheckContext):
    if ctx.plan_graph:
        n_total = ctx.plan_graph.pages_with_scale + ctx.plan_graph.pages_without_scale
        n_missing = ctx.plan_graph.pages_without_scale
    else:
        n_total = ctx.page_index.total_pages
        scale_pages = sum(1 for c in ctx.extracted.callouts if c.get("callout_type") == "scale")
        drawing_pages = sum(ctx.type_counts.get(t, 0) for t in ["plan", "detail", "section", "elevation"])
        n_missing = max(drawing_pages - scale_pages, 0)
        n_total = drawing_pages

    if n_total > 0 and n_missing > n_total * 0.3:
        ev = _make_evidence(
            detected_entities={"pages_with_scale": n_total - n_missing, "pages_without_scale": n_missing},
            search_attempts={"searched_for": "scale notation (1:100, NTS, etc.)"},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {"n_missing": n_missing, "n_total": n_total}
    return False, None, {}


def chk_no_site_plan(ctx: CheckContext):
    has_site = any(
        "site" in (p.title or "").lower() or p.doc_type == "plan" and "site" in " ".join(p.keywords_hit).lower()
        for p in ctx.page_index.pages
    )
    if not has_site:
        ev = _make_evidence(
            search_attempts={"searched_for": "site plan / site layout"},
            confidence=0.65,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_sections(ctx: CheckContext):
    if ctx.type_counts.get("section", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "section drawings", "types_found": dict(ctx.type_counts)},
            confidence=0.75,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_elevations(ctx: CheckContext):
    if ctx.type_counts.get("elevation", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "elevation drawings"},
            confidence=0.6,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_boq(ctx: CheckContext):
    if ctx.type_counts.get("boq", 0) == 0 and len(ctx.extracted.boq_items) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "Bill of Quantities (BOQ) pages"},
            confidence=0.8,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_boq_missing_quantities(ctx: CheckContext):
    items = ctx.extracted.boq_items
    if len(items) >= 2:
        missing_qty = [item for item in items if item.get("qty") is None]
        n_missing = len(missing_qty)
        n_total = len(items)
        if n_missing > 0 and n_missing / n_total > 0.2:
            pages = list({item["source_page"] for item in missing_qty})
            snippets = [f"{item['item_no']}: {item['description'][:60]}" for item in missing_qty[:5]]
            ev = _make_evidence(
                pages=pages[:10],
                snippets=snippets,
                detected_entities={"items_missing_qty": n_missing, "total_items": n_total},
                confidence=0.8,
                confidence_reason=f"{n_missing}/{n_total} BOQ items have no quantity.",
                budget=ctx.budget,
            )
            return True, ev, {"n_missing": n_missing, "n_total": n_total}
    return False, None, {}


def chk_no_conditions(ctx: CheckContext):
    if ctx.type_counts.get("conditions", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "tender/contract conditions pages"},
            confidence=0.65,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_addendum(ctx: CheckContext):
    if ctx.type_counts.get("addendum", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "addendum/corrigendum pages"},
            confidence=0.4,
            confidence_reason="No addendum found. This is informational — there may not be one.",
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_schedule_missing_sizes(ctx: CheckContext):
    missing = [s for s in ctx.extracted.schedules if not s.get("has_size")]
    if len(missing) > 0:
        pages = list({s["source_page"] for s in missing})
        snippets = [f"{s['mark']}: no size" for s in missing[:5]]
        ev = _make_evidence(
            pages=pages[:10],
            snippets=snippets,
            detected_entities={"items_missing_size": len(missing), "total_schedule_rows": len(ctx.extracted.schedules)},
            confidence=0.8,
            confidence_reason=f"{len(missing)} schedule items have marks but no sizes.",
            budget=ctx.budget,
        )
        return True, ev, {"n_missing": len(missing)}
    return False, None, {}


def chk_schedule_missing_qty(ctx: CheckContext):
    missing = [s for s in ctx.extracted.schedules if not s.get("has_qty")]
    if len(missing) > 0:
        pages = list({s["source_page"] for s in missing})
        snippets = [f"{s['mark']}: no qty" for s in missing[:5]]
        ev = _make_evidence(
            pages=pages[:10],
            snippets=snippets,
            detected_entities={"items_missing_qty": len(missing)},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {"n_missing": len(missing)}
    return False, None, {}


def chk_conflicting_material_specs(ctx: CheckContext):
    """Check for conflicting concrete grades across pages."""
    grade_pattern = re.compile(r'\bM-?(\d{2,3})\b', re.IGNORECASE)
    grade_locations: Dict[str, List[int]] = defaultdict(list)

    for c in ctx.material_callouts:
        match = grade_pattern.search(c["text"])
        if match:
            grade = f"M{match.group(1)}"
            grade_locations[grade].append(c["source_page"])

    # Conflict if same structural element has different grades
    if len(grade_locations) >= 2:
        conflicts_str = ", ".join(
            f"{grade} (p.{','.join(str(p+1) for p in pages[:3])})"
            for grade, pages in sorted(grade_locations.items())
        )
        all_pages = []
        for pages in grade_locations.values():
            all_pages.extend(pages)
        ev = _make_evidence(
            pages=sorted(set(all_pages))[:10],
            snippets=[f"{g}: pages {','.join(str(p+1) for p in ps[:3])}" for g, ps in grade_locations.items()],
            detected_entities={"grades_found": dict(grade_locations)},
            confidence=0.6,
            confidence_reason="Multiple concrete grades found — may be intentional for different elements.",
            budget=ctx.budget,
        )
        return True, ev, {"conflicts": conflicts_str}
    return False, None, {}


def chk_duplicate_tags_different_sizes(ctx: CheckContext):
    """Check if same tag has different sizes on different pages."""
    tag_sizes: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for s in ctx.extracted.schedules:
        mark = s.get("mark")
        size = s.get("size")
        page = s.get("source_page", 0)
        if mark and size:
            tag_sizes[mark][size].append(page)

    conflicts = []
    for mark, sizes in tag_sizes.items():
        if len(sizes) > 1:
            conflict_str = f"{mark}: " + " vs ".join(
                f"{sz} (p.{','.join(str(p+1) for p in pgs[:2])})"
                for sz, pgs in sizes.items()
            )
            conflicts.append(conflict_str)

    if conflicts:
        all_pages = []
        for sizes in tag_sizes.values():
            for pgs in sizes.values():
                all_pages.extend(pgs)
        ev = _make_evidence(
            pages=sorted(set(all_pages))[:10],
            snippets=conflicts[:5],
            confidence=0.85,
            confidence_reason="Same tag has different sizes on different pages.",
            budget=ctx.budget,
        )
        return True, ev, {"conflicts": "; ".join(conflicts[:3])}
    return False, None, {}


# --- Commercial check functions (Sprint 19) ---

def _com_missing_term(ctx: CheckContext, term_type: str, term_label: str):
    """Helper: fire if a commercial term_type is not found."""
    if term_type not in ctx.commercial_term_types:
        conditions_pages = ctx.type_counts.get("conditions", 0) + ctx.type_counts.get("spec", 0)
        ev = _make_evidence(
            search_attempts={
                "searched_for": f"{term_label} in conditions/spec pages",
                "conditions_pages_indexed": conditions_pages,
                "commercial_terms_found": sorted(ctx.commercial_term_types),
            },
            confidence=0.7 if conditions_pages > 0 else 0.5,
            confidence_reason=(
                f"Searched {conditions_pages} conditions/spec pages; "
                f"'{term_label}' not found."
            ) if conditions_pages > 0 else f"No conditions/spec pages found to search for '{term_label}'.",
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_ld_clause(ctx: CheckContext):
    return _com_missing_term(ctx, "ld_clause", "liquidated damages / LD clause")


def chk_no_retention(ctx: CheckContext):
    return _com_missing_term(ctx, "retention", "retention clause")


def chk_no_warranty_dlp(ctx: CheckContext):
    return _com_missing_term(ctx, "warranty_dlp", "warranty / defect liability period")


def chk_no_bid_validity(ctx: CheckContext):
    return _com_missing_term(ctx, "bid_validity", "bid validity period")


def chk_no_emd(ctx: CheckContext):
    return _com_missing_term(ctx, "emd_bid_security", "EMD / bid security amount")


def chk_no_performance_bond(ctx: CheckContext):
    return _com_missing_term(ctx, "performance_bond", "performance bank guarantee")


def chk_no_mobilization_advance(ctx: CheckContext):
    return _com_missing_term(ctx, "mobilization_advance", "mobilization advance")


def chk_no_insurance(ctx: CheckContext):
    return _com_missing_term(ctx, "insurance", "insurance / CAR policy")


def chk_no_escalation(ctx: CheckContext):
    return _com_missing_term(ctx, "escalation", "price escalation clause")


def chk_ld_rate_high(ctx: CheckContext):
    """Fire if LD rate exceeds 1% — unusually high."""
    if "ld_clause" not in ctx.commercial_term_types:
        return False, None, {}
    for term in ctx.commercial_terms:
        if term.get("term_type") == "ld_clause":
            value = term.get("value")
            if isinstance(value, (int, float)) and value > 1.0:
                cadence = term.get("cadence") or ""
                cadence_str = f"per {cadence} " if cadence else ""
                snippet = term.get("snippet", "")
                ev = _make_evidence(
                    pages=[term.get("source_page", 0)],
                    snippets=[snippet] if snippet else [],
                    detected_entities={"ld_rate": value, "cadence": cadence},
                    confidence=0.85,
                    confidence_reason=f"LD rate of {value}% {cadence_str}found in conditions.",
                    budget=ctx.budget,
                )
                return True, ev, {"ld_rate": value, "cadence_str": cadence_str}
    return False, None, {}


# Map check IDs to functions
CHECK_FN_MAP = {
    "chk_door_schedule_missing": chk_door_schedule_missing,
    "chk_window_schedule_missing": chk_window_schedule_missing,
    "chk_finish_schedule_missing": chk_finish_schedule_missing,
    "chk_no_general_notes": chk_no_general_notes,
    "chk_no_legend": chk_no_legend,
    "chk_room_dimensions_missing": chk_room_dimensions_missing,
    "chk_no_structural": chk_no_structural,
    "chk_no_foundation": chk_no_foundation,
    "chk_bbs_missing": chk_bbs_missing,
    "chk_no_mep": chk_no_mep,
    "chk_no_electrical": chk_no_electrical,
    "chk_no_plumbing": chk_no_plumbing,
    "chk_no_hvac": chk_no_hvac,
    "chk_no_fire": chk_no_fire,
    "chk_scale_missing": chk_scale_missing,
    "chk_no_site_plan": chk_no_site_plan,
    "chk_no_sections": chk_no_sections,
    "chk_no_elevations": chk_no_elevations,
    "chk_no_boq": chk_no_boq,
    "chk_boq_missing_quantities": chk_boq_missing_quantities,
    "chk_no_conditions": chk_no_conditions,
    "chk_no_addendum": chk_no_addendum,
    "chk_schedule_missing_sizes": chk_schedule_missing_sizes,
    "chk_schedule_missing_qty": chk_schedule_missing_qty,
    "chk_conflicting_material_specs": chk_conflicting_material_specs,
    "chk_duplicate_tags_different_sizes": chk_duplicate_tags_different_sizes,
    # Commercial (Sprint 19)
    "chk_no_ld_clause": chk_no_ld_clause,
    "chk_no_retention": chk_no_retention,
    "chk_no_warranty_dlp": chk_no_warranty_dlp,
    "chk_no_bid_validity": chk_no_bid_validity,
    "chk_no_emd": chk_no_emd,
    "chk_no_performance_bond": chk_no_performance_bond,
    "chk_no_mobilization_advance": chk_no_mobilization_advance,
    "chk_no_insurance": chk_no_insurance,
    "chk_no_escalation": chk_no_escalation,
    "chk_ld_rate_high": chk_ld_rate_high,
}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_rfis(
    extracted: ExtractionResult,
    page_index: PageIndex,
    selected: SelectedPages,
    plan_graph: Optional[PlanSetGraph] = None,
    run_coverage: Optional[RunCoverage] = None,
) -> List[RFIItem]:
    """
    Run the discipline checklist and generate evidence-backed RFIs.

    Args:
        extracted: ExtractionResult from run_extractors.
        page_index: PageIndex from build_page_index.
        selected: SelectedPages from select_pages.
        plan_graph: Optional PlanSetGraph (may be None for small sets).
        run_coverage: Optional RunCoverage for coverage-gated assertions.

    Returns:
        List of RFIItem with evidence.
    """
    ctx = CheckContext(extracted, page_index, selected, plan_graph, run_coverage=run_coverage)
    rfis: List[RFIItem] = []
    rfi_counter = 1

    for check_id, fn_name, trade, priority, question_tpl, why_tpl in CHECKLIST:
        fn = CHECK_FN_MAP.get(fn_name)
        if not fn:
            continue

        try:
            should_fire, evidence, fmt_kwargs = fn(ctx)
        except Exception as e:
            if DEBUG:
                import logging
                logging.getLogger(__name__).warning(f"Check {check_id} failed: {e}")
            continue

        if should_fire and evidence:
            question = question_tpl.format(**fmt_kwargs) if fmt_kwargs else question_tpl
            why = why_tpl.format(**fmt_kwargs) if fmt_kwargs else why_tpl

            # --- Coverage gating ---
            relevant_types = _CHECK_DOC_TYPES.get(check_id, [])
            cov_status = ctx.is_covered(*relevant_types) if relevant_types else CoverageStatus.NOT_FOUND_AFTER_SEARCH

            actual_priority = priority
            if cov_status == CoverageStatus.UNKNOWN_NOT_PROCESSED:
                # Downgrade priority — can't confirm absence
                actual_priority = Severity.LOW
                question = question + " [Coverage gap \u2014 may exist on unprocessed pages]"

            rfi = RFIItem(
                id=create_rfi_id(rfi_counter),
                trade=trade,
                priority=actual_priority,
                question=question,
                why_it_matters=why,
                evidence=evidence,
                suggested_resolution=_suggest_resolution(check_id),
                issue_type=check_id,
                package=_package_for_trade(trade),
                coverage_status=cov_status.value,
            )
            rfis.append(rfi)
            rfi_counter += 1

    return rfis


def _suggest_resolution(check_id: str) -> str:
    """Provide a default suggested resolution based on check ID."""
    resolutions = {
        "CHK-A-001": "Request door schedule from architect/consultant.",
        "CHK-A-002": "Request window schedule from architect/consultant.",
        "CHK-A-003": "Request finish schedule from architect/consultant.",
        "CHK-A-004": "Request general notes sheet.",
        "CHK-A-005": "Request legend/symbols sheet.",
        "CHK-A-006": "Request dimensioned floor plans or area statement.",
        "CHK-S-001": "Request structural drawings from structural engineer.",
        "CHK-S-002": "Request foundation details including type, depth, and sizes.",
        "CHK-S-003": "Request bar bending schedule from structural engineer.",
        "CHK-M-001": "Request MEP drawings from MEP consultant.",
        "CHK-M-002": "Request electrical layout drawings.",
        "CHK-M-003": "Request plumbing layout drawings.",
        "CHK-M-004": "Confirm if HVAC is in scope; request drawings if applicable.",
        "CHK-M-005": "Request fire fighting/protection layout from fire consultant.",
        "CHK-X-001": "Request drawings with scale notation or confirm NTS policy.",
        "CHK-X-002": "Request site plan with boundaries, setbacks, and external works.",
        "CHK-X-003": "Request section drawings for floor-to-floor heights and beam depths.",
        "CHK-X-004": "Request elevation drawings for facade details.",
        "CHK-X-005": "Request BOQ in the tender format for pricing.",
        "CHK-X-006": "Clarify missing quantities in BOQ items.",
        "CHK-X-007": "Request tender conditions / general conditions of contract.",
        "CHK-X-008": "Confirm if any addenda/corrigenda were issued.",
        "CHK-SCH-001": "Request complete schedule with sizes for all items.",
        "CHK-SCH-002": "Request quantities in the schedule.",
        "CHK-C-001": "Clarify which material specification governs for each element.",
        "CHK-C-002": "Clarify correct size for each conflicting door/window tag.",
        # Commercial (Sprint 19)
        "CHK-COM-001": "Request liquidated damages clause / rate from the client.",
        "CHK-COM-002": "Request retention percentage and release terms.",
        "CHK-COM-003": "Request warranty / defect liability period details.",
        "CHK-COM-004": "Confirm bid validity period before submission.",
        "CHK-COM-005": "Request EMD / bid security amount and instrument type.",
        "CHK-COM-006": "Request performance bank guarantee percentage and validity.",
        "CHK-COM-007": "Confirm if mobilization advance is available and terms.",
        "CHK-COM-008": "Request insurance requirements (CAR policy, third-party, workmen).",
        "CHK-COM-009": "Confirm if price escalation clause applies to this contract.",
        "CHK-COM-010": "Verify the LD rate — it appears unusually high.",
    }
    return resolutions.get(check_id, "Raise RFI with the architect/consultant for clarification.")


def _package_for_trade(trade: Trade) -> str:
    """Map trade to a bid package name."""
    mapping = {
        Trade.ARCHITECTURAL: "Architectural",
        Trade.STRUCTURAL: "Structural",
        Trade.ELECTRICAL: "Electrical",
        Trade.PLUMBING: "Plumbing",
        Trade.MEP: "MEP",
        Trade.CIVIL: "Civil",
        Trade.FINISHES: "Finishes",
        Trade.GENERAL: "General",
        Trade.COMMERCIAL: "Commercial",
    }
    return mapping.get(trade, "General")
