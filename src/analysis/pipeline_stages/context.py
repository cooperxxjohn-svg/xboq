"""
Dataclasses that carry shared state between pipeline stage functions.

QTOInputs  — everything run_qto_modules() needs from the upstream stages.
QTOOutputs — everything run_qto_modules() produces that downstream code needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class QTOInputs:
    """Inputs to the QTO module cascade stage."""

    # From page-index / extract stages
    page_index_result: Any          # PageIndex | None
    all_page_texts: List[str]       # flat list indexed by page_idx
    extraction_result: Any          # TableExtractionResult | None

    # Initial spec items accumulated before QTO (Sprint 21D)
    spec_items_list: list           # mutated internally; updated copy returned
    stub_items_list: list           # schedule stub items from reconciler
    recon: Any                      # ReconciliationResult | None

    # Callbacks
    fire_sub: Callable              # fire_sub(agent_id, status, msg, items=0)

    # Mutable shared state (appended to in-place)
    low_confidence_flags: list      # list of {type, message, page} dicts

    # Tenant / project identifiers (for rate overrides)
    tenant_id: Optional[str]
    project_id: Optional[str]

    # For visual detection and structural takeoff
    input_files: list               # List[Path] — original upload files
    llm_client: Any                 # anthropic.Anthropic | openai.OpenAI | None
    primary_pdf_path: Any           # Path | None — first PDF path


@dataclass
class QTOOutputs:
    """Outputs produced by the QTO module cascade stage."""

    # -----------------------------------------------------------------
    # Rate-engine outputs  (needed by bid_margin, immediately after QTO)
    # -----------------------------------------------------------------
    qto_rated_items: list = field(default_factory=list)
    qto_grand_total_inr: float = 0.0

    # -----------------------------------------------------------------
    # Rebuilt unified line-items (Sprint 41 rebuild)
    # -----------------------------------------------------------------
    line_items_payload: list = field(default_factory=list)
    dedup_stats: dict = field(default_factory=dict)

    # Spec items that have no quantity (for UI "needs qty" list)
    spec_needs_qty: list = field(default_factory=list)

    # Spec-driven params extracted from scope text (building_type, area, etc.)
    spec_params_payload: dict = field(default_factory=dict)

    # Detected floor area and count (used by quantity reconciliation)
    st_area_sqm: float = 0.0
    st_floors: int = 1

    # -----------------------------------------------------------------
    # Individual QTO module line-item outputs
    # (used in final payload assembly — keys mirror payload dict keys)
    # -----------------------------------------------------------------
    qto_paint_items: list = field(default_factory=list)
    qto_wp_items: list = field(default_factory=list)
    qto_dw_items: list = field(default_factory=list)
    qto_mep_items: list = field(default_factory=list)
    qto_sw_items: list = field(default_factory=list)
    qto_brickwork_items: list = field(default_factory=list)
    qto_plaster_items: list = field(default_factory=list)
    qto_earthwork_items: list = field(default_factory=list)
    qto_flooring_items: list = field(default_factory=list)
    qto_foundation_items: list = field(default_factory=list)
    qto_extdev_items: list = field(default_factory=list)
    qto_prelims_items: list = field(default_factory=list)
    qto_elv_items: list = field(default_factory=list)

    # Additional waterproofing / paint area subtotals (qty reconciliation)
    qto_wp_wet_area: float = 0.0
    qto_wp_roof_area: float = 0.0
    qto_paint_int_wall: float = 0.0

    # -----------------------------------------------------------------
    # qto_summary dict — assigned to payload["qto_summary"]
    # -----------------------------------------------------------------
    qto_summary_dict: dict = field(default_factory=dict)

    # Excel bytes for download (stored as payload["_excel_bytes"])
    qto_excel_bytes: bytes = b""
