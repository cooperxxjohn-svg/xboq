"""
QTO module cascade extracted from run_analysis_pipeline().

run_qto_modules(inp) runs all 21 QTO modules (finish → scale → structural →
implied → MEP → visual → vmeas → D&W → painting → waterproofing → sitework →
brickwork → plaster → earthwork → flooring → scope_disagg → LLM spec →
foundation → extdev → prelims → ELV), then applies the rate engine and
rebuilds the unified line-items list.

Returns QTOOutputs — a typed dataclass with every variable that downstream
pipeline stages need.
"""

from __future__ import annotations

import logging
import os

from .context import QTOInputs, QTOOutputs
from ..qto.registry import is_enabled as _qto_enabled

logger = logging.getLogger(__name__)


def run_qto_modules(inp: QTOInputs) -> QTOOutputs:  # noqa: C901 — intentionally long
    """Execute the full QTO module cascade and return all outputs."""

    # Unpack inputs for readability (mirrors the original pipeline locals)
    page_index_result = inp.page_index_result
    all_page_texts    = inp.all_page_texts
    extraction_result = inp.extraction_result
    _spec_items_list  = list(inp.spec_items_list)   # copy — we mutate this
    _stub_items_list  = inp.stub_items_list
    _recon            = inp.recon
    fire_sub          = inp.fire_sub
    _low_confidence_flags = inp.low_confidence_flags  # mutated in-place (list ref)
    tenant_id         = inp.tenant_id
    project_id        = inp.project_id
    input_files       = inp.input_files
    llm_client        = inp.llm_client
    primary_pdf_path  = inp.primary_pdf_path

    # ── QTO: Room-finish takeoff from plan drawings ──────────────────
    _qto_finish_items: list = []
    _qto_rooms: list = []
    fire_sub("finish_agent", "working", "finish takeoff")
    try:
        from ..qto.finish_takeoff import run_finish_takeoff
        _plan_page_texts = []
        if page_index_result and all_page_texts:
            for pg in page_index_result.pages:
                pidx = pg.page_idx
                if pidx < len(all_page_texts):
                    _plan_page_texts.append((
                        pidx,
                        all_page_texts[pidx],
                        pg.doc_type,
                    ))
        _finish_scheds = [
            s for s in getattr(extraction_result, 'schedules', [])
            if str(s.get('schedule_type', '')).lower() in ('finish', 'finishes', 'room_finish')
        ]
        if _plan_page_texts:
            _qto_rooms, _qto_finish_items = run_finish_takeoff(
                _plan_page_texts, _finish_scheds
            )
        if _qto_finish_items:
            _spec_items_list = list(_spec_items_list) + _qto_finish_items
        fire_sub("finish_agent", "done", f"{len(_qto_finish_items)} items", len(_qto_finish_items))
    except Exception as _qto_err:
        logger.warning("QTO finish takeoff failed: %s", _qto_err)
        fire_sub("finish_agent", "error", str(_qto_err)[:80])

    # ── Scale Detection (Sprint 43 — early, feeds structural + visual) ──
    _detected_scale = None
    _all_page_texts_for_scale: list = []
    fire_sub("scale_agent", "working", "detecting scale")
    try:
        from ..qto.scale_detector import detect_scale as _detect_scale_fn
        if page_index_result and all_page_texts:
            for pg in page_index_result.pages:
                pidx = pg.page_idx
                if pidx < len(all_page_texts):
                    _all_page_texts_for_scale.append((
                        pidx,
                        all_page_texts[pidx] or "",
                        getattr(pg, "doc_type", "unknown"),
                    ))
        elif all_page_texts:
            _all_page_texts_for_scale = [
                (i, t or "", "unknown") for i, t in enumerate(all_page_texts)
            ]
        _detected_scale = _detect_scale_fn(_all_page_texts_for_scale)
        _scale_msg = f"1:{_detected_scale.ratio}" if _detected_scale else "not found"
        fire_sub("scale_agent", "done", _scale_msg)
        if _detected_scale is None:
            _scale_page_num = (
                _all_page_texts_for_scale[0][0] if _all_page_texts_for_scale else 0
            )
            logger.warning(
                "Scale detector: no scale label found on page %s — assuming 1:100 (low confidence)",
                _scale_page_num
            )
            _low_confidence_flags.append({
                "type": "assumed_scale",
                "message": "No scale label found — assuming 1:100",
                "page": _scale_page_num,
            })
    except Exception as _scale_early_err:
        logger.debug("Early scale detection failed (non-critical): %s", _scale_early_err)
        fire_sub("scale_agent", "skipped", "non-critical")

    # ── QTO: Structural takeoff from schedule text ──────────────────
    _qto_structural_items: list = []
    _qto_structural_elements: list = []
    _qto_structural_mode = "none"
    _qto_structural_warnings: list = []
    _structural_page_texts: list = []
    _st_floors = 1
    _st_area_sqm = 0.0
    _spec_params_payload: dict = {}
    fire_sub("structural_agent", "working", "structural QTO")
    try:
        from ..qto.structural_takeoff import run_structural_takeoff

        # Collect (page_idx, text, doc_type) for structural/plan pages
        if page_index_result and all_page_texts:
            for pg in page_index_result.pages:
                pidx = pg.page_idx
                if pidx < len(all_page_texts):
                    _structural_page_texts.append((
                        pidx,
                        all_page_texts[pidx],
                        pg.doc_type,
                    ))

        # ── Floor count: text extractor first, metadata fallback ────────
        _st_floor_source = "default"
        try:
            from ..qto.floor_count_extractor import extract_floor_count
            _fc = extract_floor_count(_structural_page_texts)
            if _fc and _fc.count >= 1:
                _st_floors = _fc.count
                _st_floor_source = (
                    f"text:{_fc.source_text!r} (p{_fc.source_page+1}, "
                    f"conf={_fc.confidence:.0%})"
                )
        except Exception as _fc_err:
            logger.debug("Floor count extractor: %s", _fc_err)

        if hasattr(extraction_result, 'metadata') and extraction_result.metadata:
            _meta = extraction_result.metadata or {}
            if _st_floors == 1 and _st_floor_source == "default":
                _meta_floors = int(_meta.get('floors', _meta.get('storeys', 1)) or 1)
                if _meta_floors > 1:
                    _st_floors = _meta_floors
                    _st_floor_source = "metadata"

        # Built-up area from rooms or metadata
        if _qto_rooms:
            _st_area_sqm = sum(
                r.area_sqm for r in _qto_rooms if r.area_sqm and r.area_sqm > 0
            )
        if not _st_area_sqm:
            _meta2 = getattr(extraction_result, 'metadata', None) or {}
            _st_area_sqm = float(_meta2.get('total_area_sqm', 0) or 0)

        # Spec-driven fallback: parse scope text when area still unknown
        if not _st_area_sqm and _structural_page_texts:
            try:
                from ..spec_boq_generator import extract_spec_params as _esp
                _spec_p = _esp(_structural_page_texts)
                if _spec_p.total_area_sqm >= 200:
                    _st_area_sqm = _spec_p.total_area_sqm
                    if _spec_p.floor_count > _st_floors:
                        _st_floors = _spec_p.floor_count
                        _st_floor_source = "spec_text"
                    logger.info(
                        "spec_boq_generator: area=%.0f sqm floors=%d conf=%.0f%% types=%s",
                        _st_area_sqm, _st_floors,
                        _spec_p.confidence * 100, _spec_p.building_types,
                    )
                    _spec_params_payload = {
                        "total_area_sqm": _st_area_sqm,
                        "floor_count": _st_floors,
                        "building_types": _spec_p.building_types,
                        "occupancy": _spec_p.occupancy,
                        "confidence": _spec_p.confidence,
                        "warnings": _spec_p.warnings,
                    }
            except Exception as _sp_err:
                logger.warning("spec_boq_generator skipped: %s", _sp_err)

        if _structural_page_texts:
            _st_result = run_structural_takeoff(
                page_texts=_structural_page_texts,
                floors=max(1, _st_floors),
                total_area_sqm=_st_area_sqm,
                px_per_mm=_detected_scale.px_per_mm if _detected_scale else 0.0,
                known_scale_ratio=_detected_scale.ratio if _detected_scale else 0,
                pdf_path=str(primary_pdf_path or ""),
                llm_client=llm_client,
            )
            _qto_structural_items = _st_result.line_items
            _qto_structural_elements = _st_result.elements
            _qto_structural_mode = _st_result.mode
            _qto_structural_warnings = _st_result.warnings

            if _qto_structural_items:
                _spec_items_list = list(_spec_items_list) + _qto_structural_items
        fire_sub("structural_agent", "done",
                 f"{len(_qto_structural_elements)} elements, {len(_qto_structural_items)} items",
                 len(_qto_structural_items))
    except Exception as _st_err:
        logger.warning("QTO structural takeoff failed: %s", _st_err)
        fire_sub("structural_agent", "error", str(_st_err)[:80])

    # ── QTO: Implied items rule engine ──────────────────────────────
    _qto_implied_items: list = []
    _qto_implied_rules_triggered: list = []
    if _qto_enabled("implied"):
        try:
            from ..qto.implied_items import run_implied_rules, build_rule_context

            _drawing_callouts: list = []
            if extraction_result and hasattr(extraction_result, 'drawing_callouts'):
                _drawing_callouts = list(extraction_result.drawing_callouts or [])

            _implied_ctx = build_rule_context(
                structural_elements=_qto_structural_elements,
                structural_items=_qto_structural_items,
                rooms=_qto_rooms,
                total_area_sqm=_st_area_sqm,
                floors=max(1, _st_floors),
                building_type="residential",
                storey_height_mm=3000,
                drawing_callouts=_drawing_callouts,
            )
            _qto_implied_items, _qto_implied_rules_triggered = run_implied_rules(_implied_ctx)
            if _qto_implied_items:
                _spec_items_list = list(_spec_items_list) + _qto_implied_items
        except Exception as _imp_err:
            logger.warning("QTO implied items failed: %s", _imp_err)

    # ── QTO: MEP Takeoff (Sprint 37) ────────────────────────────
    _qto_mep_elements: list = []
    _qto_mep_items: list = []
    _qto_mep_mode = "none"
    _qto_mep_warnings: list = []
    _mep_page_texts: list = []
    fire_sub("mep_agent", "working", "MEP takeoff")
    try:
        from ..qto.mep_takeoff import run_mep_takeoff

        _mep_page_texts = _all_page_texts_for_scale if _all_page_texts_for_scale else []
        if not _mep_page_texts:
            if page_index_result and all_page_texts:
                for pg in page_index_result.pages:
                    pidx = pg.page_idx
                    if pidx < len(all_page_texts):
                        _mep_page_texts.append((
                            pidx,
                            all_page_texts[pidx] or "",
                            getattr(pg, "doc_type", "unknown"),
                        ))
            elif all_page_texts:
                _mep_page_texts = [
                    (i, t or "", "unknown") for i, t in enumerate(all_page_texts)
                ]

        _mep_result = run_mep_takeoff(
            page_texts=_mep_page_texts,
            floors=max(1, _st_floors),
            total_area_sqm=_st_area_sqm,
            building_type="residential",
        )
        _qto_mep_elements = _mep_result.elements
        _qto_mep_items    = _mep_result.line_items
        _qto_mep_mode     = _mep_result.mode
        _qto_mep_warnings = _mep_result.warnings

        if _qto_mep_items:
            _spec_items_list = list(_spec_items_list) + _qto_mep_items
        fire_sub("mep_agent", "done",
                 f"{len(_qto_mep_elements)} elements, {len(_qto_mep_items)} items",
                 len(_qto_mep_items))

    except Exception as _mep_err:
        logger.warning("MEP takeoff failed: %s", _mep_err)
        fire_sub("mep_agent", "error", str(_mep_err)[:80])

    # ── QTO: Visual Element Detection (Sprint 37) ────────────────
    _qto_visual_elements: list = []
    _qto_visual_items: list = []
    _qto_visual_mode = "none"
    _qto_visual_warnings: list = []
    _qto_visual_scale = ""
    _qto_visual_area = 0.0
    _primary_pdf = None
    if input_files:
        _primary_pdf = str(input_files[0])
    if not _primary_pdf:
        _primary_pdf = str(primary_pdf_path) if primary_pdf_path else None
    if llm_client is not None and _primary_pdf and _mep_page_texts:
        fire_sub("visual_detector", "working", "visual detection")
    else:
        fire_sub("visual_detector", "skipped", "no LLM / no PDF")
    try:
        if llm_client is not None and _primary_pdf and _mep_page_texts:
            from ..qto.visual_element_detector import run_visual_detection
            _vis_result = run_visual_detection(
                pdf_path=_primary_pdf,
                page_texts=_mep_page_texts,
                llm_client=llm_client,
            )
            _qto_visual_elements = _vis_result.elements
            _qto_visual_items    = _vis_result.line_items
            _qto_visual_mode     = _vis_result.mode
            _qto_visual_warnings = _vis_result.warnings
            _qto_visual_scale    = _vis_result.detected_scale
            _qto_visual_area     = _vis_result.detected_area_sqm

            if _qto_visual_items:
                _spec_items_list = list(_spec_items_list) + _qto_visual_items
            fire_sub("visual_detector", "done",
                     f"{len(_qto_visual_elements)} elements",
                     len(_qto_visual_items))
    except Exception as _vis_err:
        logger.warning("Visual element detection failed: %s", _vis_err)
        fire_sub("visual_detector", "error", str(_vis_err)[:80])

    # ── QTO: Visual Measurement (Sprint 37) ──────────────────────
    _qto_vmeas_rooms: list = []
    _qto_vmeas_items: list = []
    _qto_vmeas_mode = "none"
    _qto_vmeas_warnings: list = []
    _qto_vmeas_scale = ""
    _qto_vmeas_area = 0.0
    _qto_vmeas_room_schedule: list = []
    if llm_client is not None and _primary_pdf and _mep_page_texts:
        fire_sub("visual_measure", "working", "visual measurement")
    else:
        fire_sub("visual_measure", "skipped", "no LLM / no PDF")
    try:
        if llm_client is not None and _primary_pdf and _mep_page_texts:
            from ..qto.visual_measurement import run_visual_measurement
            _vmeas_result = run_visual_measurement(
                pdf_path=_primary_pdf,
                page_texts=_mep_page_texts,
                llm_client=llm_client,
                known_scale_ratio=_detected_scale.ratio if _detected_scale else 0,
                known_px_per_mm=_detected_scale.px_per_mm if _detected_scale else 0.0,
            )
            _qto_vmeas_rooms         = _vmeas_result.all_rooms
            _qto_vmeas_items         = _vmeas_result.line_items
            _qto_vmeas_mode          = _vmeas_result.mode
            _qto_vmeas_warnings      = _vmeas_result.warnings
            _qto_vmeas_scale         = _vmeas_result.detected_scale
            _qto_vmeas_area          = _vmeas_result.total_area_sqm
            _qto_vmeas_room_schedule = _vmeas_result.room_schedule

            if _vmeas_result.room_schedule and not _qto_rooms:
                _qto_rooms = list(_vmeas_result.room_schedule)

            if _qto_vmeas_items:
                _spec_items_list = list(_spec_items_list) + _qto_vmeas_items
            fire_sub("visual_measure", "done",
                     f"{len(_qto_vmeas_rooms)} rooms, {len(_qto_vmeas_items)} items",
                     len(_qto_vmeas_items))
    except Exception as _vmeas_err:
        logger.warning("Visual measurement failed: %s", _vmeas_err)
        fire_sub("visual_measure", "error", str(_vmeas_err)[:80])

    # ── QTO: Door & Window Takeoff (Sprint 38) ───────────────────
    _qto_dw_elements: list = []
    _qto_dw_items: list = []
    _qto_dw_mode = "none"
    _qto_dw_warnings: list = []
    _qto_dw_door_count = 0
    _qto_dw_window_count = 0
    fire_sub("dw_agent", "working", "door/window takeoff")
    try:
        from ..qto.door_window_takeoff import run_dw_takeoff
        _dw_result = run_dw_takeoff(
            page_texts=_mep_page_texts,
            floors=max(1, _st_floors),
            total_area_sqm=_st_area_sqm,
        )
        _qto_dw_elements    = _dw_result.elements
        _qto_dw_items       = _dw_result.line_items
        _qto_dw_mode        = _dw_result.mode
        _qto_dw_warnings    = _dw_result.warnings
        _qto_dw_door_count  = _dw_result.door_count
        _qto_dw_window_count= _dw_result.window_count
        if _qto_dw_items:
            _spec_items_list = list(_spec_items_list) + _qto_dw_items
        fire_sub("dw_agent", "done",
                 f"D:{_qto_dw_door_count} W:{_qto_dw_window_count}",
                 len(_qto_dw_items))
    except Exception as _dw_err:
        logger.warning("Door/window takeoff failed: %s", _dw_err)
        fire_sub("dw_agent", "error", str(_dw_err)[:80])

    # ── QTO: Painting Takeoff (Sprint 38) ────────────────────────
    _qto_paint_items: list = []
    _qto_paint_mode = "none"
    _qto_paint_warnings: list = []
    _qto_paint_int_wall = 0.0
    _qto_paint_ceiling = 0.0
    _qto_paint_ext_wall = 0.0
    fire_sub("painting_agent", "working", "painting takeoff")
    try:
        from ..qto.painting_takeoff import run_painting_takeoff
        _paint_result = run_painting_takeoff(
            rooms=_qto_rooms,
            floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
            door_count=_qto_dw_door_count,
            window_count=_qto_dw_window_count,
        )
        _qto_paint_items    = _paint_result.line_items
        _qto_paint_mode     = _paint_result.mode
        _qto_paint_warnings = _paint_result.warnings
        _qto_paint_int_wall = _paint_result.total_interior_wall_sqm
        _qto_paint_ceiling  = _paint_result.total_ceiling_sqm
        _qto_paint_ext_wall = _paint_result.total_exterior_wall_sqm
        if _qto_paint_items:
            _spec_items_list = list(_spec_items_list) + _qto_paint_items
        fire_sub("painting_agent", "done",
                 f"{len(_qto_paint_items)} items", len(_qto_paint_items))
    except Exception as _paint_err:
        logger.warning("Painting takeoff failed: %s", _paint_err)
        fire_sub("painting_agent", "error", str(_paint_err)[:80])

    # ── QTO: Waterproofing Takeoff (Sprint 38) ───────────────────
    _qto_wp_items: list = []
    _qto_wp_mode = "none"
    _qto_wp_warnings: list = []
    _qto_wp_wet_area = 0.0
    _qto_wp_roof_area = 0.0
    fire_sub("waterproof_agent", "working", "waterproofing takeoff")
    try:
        from ..qto.waterproofing_takeoff import run_waterproofing_takeoff
        _wp_result = run_waterproofing_takeoff(
            rooms=_qto_rooms,
            floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
        )
        _qto_wp_items    = _wp_result.line_items
        _qto_wp_mode     = _wp_result.mode
        _qto_wp_warnings = _wp_result.warnings
        _qto_wp_wet_area = _wp_result.wet_area_sqm
        _qto_wp_roof_area= _wp_result.roof_area_sqm
        if _qto_wp_items:
            _spec_items_list = list(_spec_items_list) + _qto_wp_items
        fire_sub("waterproof_agent", "done",
                 f"{len(_qto_wp_items)} items", len(_qto_wp_items))
    except Exception as _wp_err:
        logger.warning("Waterproofing takeoff failed: %s", _wp_err)
        fire_sub("waterproof_agent", "error", str(_wp_err)[:80])

    # ── QTO: Site Work Takeoff (Sprint 38) ───────────────────────
    _qto_sw_items: list = []
    _qto_sw_mode = "none"
    _qto_sw_warnings: list = []
    fire_sub("sitework_agent", "working", "sitework takeoff")
    try:
        from ..qto.sitework_takeoff import run_sitework_takeoff
        _sw_result = run_sitework_takeoff(
            plot_area_sqm=0.0,
            built_area_sqm=_st_area_sqm / max(1, _st_floors),
            total_floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
        )
        _qto_sw_items    = _sw_result.line_items
        _qto_sw_mode     = _sw_result.mode
        _qto_sw_warnings = _sw_result.warnings
        if _qto_sw_items:
            _spec_items_list = list(_spec_items_list) + _qto_sw_items
        fire_sub("sitework_agent", "done",
                 f"{len(_qto_sw_items)} items", len(_qto_sw_items))
    except Exception as _sw_err:
        logger.warning("Sitework takeoff failed: %s", _sw_err)
        fire_sub("sitework_agent", "error", str(_sw_err)[:80])

    # ── QTO: Brickwork Takeoff (Sprint Q1) ──────────────────────
    _qto_brickwork_items: list = []
    fire_sub("brickwork_agent", "working", "brickwork takeoff")
    try:
        from ..qto.brickwork_takeoff import run_brickwork_takeoff
        _bw_result = run_brickwork_takeoff(
            floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
            building_type=(_spec_params_payload.get("building_types") or ["academic"])[0]
                          if _spec_params_payload else "academic",
        )
        _qto_brickwork_items = _bw_result.line_items
        if _qto_brickwork_items:
            _spec_items_list = list(_spec_items_list) + _qto_brickwork_items
        fire_sub("brickwork_agent", "done", f"{len(_qto_brickwork_items)} items", len(_qto_brickwork_items))
    except Exception as _bw_err:
        logger.warning("Brickwork takeoff failed: %s", _bw_err)
        fire_sub("brickwork_agent", "error", str(_bw_err)[:80])

    # ── QTO: Plaster Takeoff (Sprint Q1) ─────────────────────────
    _qto_plaster_items: list = []
    fire_sub("plaster_agent", "working", "plaster takeoff")
    try:
        from ..qto.plaster_takeoff import run_plaster_takeoff
        _pl_result = run_plaster_takeoff(
            floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
            building_type=(_spec_params_payload.get("building_types") or ["academic"])[0]
                          if _spec_params_payload else "academic",
        )
        _qto_plaster_items = _pl_result.line_items
        if _qto_plaster_items:
            _spec_items_list = list(_spec_items_list) + _qto_plaster_items
        fire_sub("plaster_agent", "done", f"{len(_qto_plaster_items)} items", len(_qto_plaster_items))
    except Exception as _pl_err:
        logger.warning("Plaster takeoff failed: %s", _pl_err)
        fire_sub("plaster_agent", "error", str(_pl_err)[:80])

    # ── QTO: Earthwork Takeoff (Sprint Q2) ───────────────────────
    _qto_earthwork_items: list = []
    fire_sub("earthwork_agent", "working", "earthwork takeoff")
    try:
        from ..qto.earthwork_takeoff import run_earthwork_takeoff
        _ew_result = run_earthwork_takeoff(
            floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
            _footprint_is_total_bua=True,
        )
        _qto_earthwork_items = _ew_result.line_items
        if _qto_earthwork_items:
            _spec_items_list = list(_spec_items_list) + _qto_earthwork_items
        fire_sub("earthwork_agent", "done", f"{len(_qto_earthwork_items)} items", len(_qto_earthwork_items))
    except Exception as _ew_err:
        logger.warning("Earthwork takeoff failed: %s", _ew_err)
        fire_sub("earthwork_agent", "error", str(_ew_err)[:80])

    # ── QTO: Flooring Takeoff (Sprint Q2) ────────────────────────
    _qto_flooring_items: list = []
    fire_sub("flooring_agent", "working", "flooring takeoff")
    try:
        from ..qto.finish_takeoff import run_flooring_takeoff
        _fl_result = run_flooring_takeoff(
            floor_area_sqm=_st_area_sqm,
            floors=max(1, _st_floors),
            building_type=(_spec_params_payload.get("building_types") or ["academic"])[0]
                          if _spec_params_payload else "academic",
            rooms=_qto_rooms,
        )
        _qto_flooring_items = _fl_result.line_items
        if _qto_flooring_items:
            _spec_items_list = list(_spec_items_list) + _qto_flooring_items
        fire_sub("flooring_agent", "done", f"{len(_qto_flooring_items)} items", len(_qto_flooring_items))
    except Exception as _fl_err:
        logger.warning("Flooring takeoff failed: %s", _fl_err)
        fire_sub("flooring_agent", "error", str(_fl_err)[:80])

    # ── QTO: Scope Disaggregation (Sprint Q3) ────────────────────
    _qto_disagg_items: list = []
    fire_sub("disagg_agent", "working", "scope disaggregation")
    try:
        from ..scope_disaggregator import disaggregate_scope, run_qto_for_scope
        _disagg = disaggregate_scope(
            _structural_page_texts,
            fallback_area_sqm=_st_area_sqm,
            fallback_floors=max(1, _st_floors),
        )
        if not _disagg.single_building and len(_disagg.buildings) > 1:
            _qto_disagg_items = run_qto_for_scope(_disagg)
            if _qto_disagg_items:
                _spec_items_list = list(_spec_items_list) + _qto_disagg_items
                logger.info("scope_disaggregator: %d items from %d buildings",
                            len(_qto_disagg_items), len(_disagg.buildings))
        fire_sub("disagg_agent", "done",
                 f"{len(_disagg.buildings)} buildings, {len(_qto_disagg_items)} items",
                 len(_qto_disagg_items))
    except Exception as _disagg_err:
        logger.warning("Scope disaggregation failed: %s", _disagg_err)
        fire_sub("disagg_agent", "error", str(_disagg_err)[:80])

    # ── QTO: LLM Spec Extraction (Sprint Q4) ─────────────────────
    _qto_spec_llm_items: list = []
    fire_sub("spec_llm_agent", "working", "LLM spec extraction")
    try:
        from ..spec_extractor_llm import extract_spec_quantities
        _spec_extracted = extract_spec_quantities(_structural_page_texts)
        if _spec_extracted.llm_used and _spec_extracted.items:
            _qto_spec_llm_items = [
                {"description": i.description, "trade": i.trade, "unit": i.unit,
                 "qty": i.qty, "source": "spec_text", "confidence": i.confidence}
                for i in _spec_extracted.items if i.qty > 0
            ]
            _spec_items_list = list(_spec_items_list) + _qto_spec_llm_items
            logger.info("spec_extractor_llm: added %d items", len(_qto_spec_llm_items))
        fire_sub("spec_llm_agent", "done",
                 f"{len(_qto_spec_llm_items)} items (llm={'yes' if _spec_extracted.llm_used else 'no'})",
                 len(_qto_spec_llm_items))
    except Exception as _llm_err:
        logger.warning("LLM spec extraction skipped: %s", _llm_err)
        fire_sub("spec_llm_agent", "skipped", str(_llm_err)[:80])

    # ── QTO: Foundation Takeoff ───────────────────────────────────
    _qto_foundation_items: list = []
    fire_sub("foundation_agent", "working", "foundation takeoff")
    try:
        from ..qto.foundation_takeoff import run_foundation_takeoff
        _ft_btype = (_spec_params_payload.get("building_types") or ["hostel"])[0] \
                    if _spec_params_payload else "hostel"
        _ft_soil_type = "normal"
        if _spec_params_payload:
            _ft_soil_type = _spec_params_payload.get("soil_type") or "normal"
        _ft_result = run_foundation_takeoff(
            floor_area_sqm=max(_st_area_sqm, 100.0),
            floors=max(1, _st_floors),
            building_type=_ft_btype,
            has_basement=bool(_spec_params_payload.get("has_basement")) if _spec_params_payload else False,
            soil_type=_ft_soil_type,
            pile_depth_m=0,
        )
        _qto_foundation_items = _ft_result.line_items
        if _qto_foundation_items:
            _spec_items_list = list(_spec_items_list) + _qto_foundation_items
        fire_sub("foundation_agent", "done", f"{len(_qto_foundation_items)} items", len(_qto_foundation_items))
    except Exception as _ft_err:
        logger.warning("Foundation takeoff failed: %s", _ft_err)
        fire_sub("foundation_agent", "error", str(_ft_err)[:80])

    # ── QTO: External Development Takeoff ────────────────────────
    _qto_extdev_items: list = []
    fire_sub("extdev_agent", "working", "external development")
    try:
        from ..qto.external_development_takeoff import run_external_dev_takeoff
        _ed_btype = (_spec_params_payload.get("building_types") or ["hostel"])[0] \
                    if _spec_params_payload else "hostel"
        _ed_result = run_external_dev_takeoff(
            total_area_sqm=max(_st_area_sqm, 100.0),
            floors=max(1, _st_floors),
            occupancy=int(_spec_params_payload.get("occupancy") or 0) if _spec_params_payload else 0,
            building_type=_ed_btype,
        )
        _qto_extdev_items = _ed_result.line_items
        if _qto_extdev_items:
            _spec_items_list = list(_spec_items_list) + _qto_extdev_items
        fire_sub("extdev_agent", "done", f"{len(_qto_extdev_items)} items", len(_qto_extdev_items))
    except Exception as _ed_err:
        logger.warning("External dev takeoff failed: %s", _ed_err)
        fire_sub("extdev_agent", "error", str(_ed_err)[:80])

    # ── QTO: Prelims Takeoff ──────────────────────────────────────
    _qto_prelims_items: list = []
    fire_sub("prelims_agent", "working", "prelims takeoff")
    try:
        from ..qto.prelims_takeoff import run_prelims_takeoff
        _pr_btype = (_spec_params_payload.get("building_types") or ["hostel"])[0] \
                    if _spec_params_payload else "hostel"
        _pr_result = run_prelims_takeoff(
            total_area_sqm=max(_st_area_sqm, 100.0),
            floors=max(1, _st_floors),
            building_type=_pr_btype,
            occupancy=int(_spec_params_payload.get("occupancy") or 0) if _spec_params_payload else 0,
        )
        _qto_prelims_items = _pr_result.line_items
        if _qto_prelims_items:
            _spec_items_list = list(_spec_items_list) + _qto_prelims_items
        fire_sub("prelims_agent", "done", f"{len(_qto_prelims_items)} items", len(_qto_prelims_items))
    except Exception as _pr_err:
        logger.warning("Prelims takeoff failed: %s", _pr_err)
        fire_sub("prelims_agent", "error", str(_pr_err)[:80])

    # ── QTO: ELV Takeoff ─────────────────────────────────────────
    _qto_elv_items: list = []
    fire_sub("elv_agent", "working", "ELV takeoff")
    try:
        from ..qto.elv_takeoff import run_elv_takeoff
        _elv_btype = (_spec_params_payload.get("building_types") or ["hostel"])[0] \
                     if _spec_params_payload else "hostel"
        _elv_result = run_elv_takeoff(
            floor_area_sqm=max(_st_area_sqm, 100.0),
            floors=max(1, _st_floors),
            building_type=_elv_btype,
        )
        _qto_elv_items = _elv_result.line_items
        if _qto_elv_items:
            _spec_items_list = list(_spec_items_list) + _qto_elv_items
        fire_sub("elv_agent", "done", f"{len(_qto_elv_items)} items", len(_qto_elv_items))
    except Exception as _elv_err:
        logger.warning("ELV takeoff failed: %s", _elv_err)
        fire_sub("elv_agent", "error", str(_elv_err)[:80])

    # ── Filter: drop spec-text items with no quantity ─────────────
    _spec_needs_qty: list = []
    _spec_items_filtered: list = []
    for _si in _spec_items_list:
        _si_qty = _si.get("qty") or _si.get("quantity") or 0
        try:
            _si_qty = float(_si_qty)
        except (TypeError, ValueError):
            _si_qty = 0.0
        _si_source = _si.get("source", "")
        if _si_qty > 0 or _si_source not in ("spec_item", "spec_text", "implied", ""):
            _spec_items_filtered.append(_si)
        else:
            _spec_needs_qty.append(dict(_si, qty_status="needs_qty"))
    logger.info(
        "Qty filter: %d items kept, %d moved to needs_qty",
        len(_spec_items_filtered), len(_spec_needs_qty)
    )
    _spec_items_list = _spec_items_filtered

    # ── Rate Engine: Apply rates to ALL spec items (Sprint 38) ───
    _qto_rated_items: list = []
    _qto_trade_summary: dict = {}
    _qto_grand_total_inr: float = 0.0
    fire_sub("rate_engine", "working", "applying rates")
    try:
        from ..qto.rate_engine import apply_rates, compute_trade_summary
        _project_rate_overrides: dict = {}
        try:
            from src.analysis.project_rates import load_rates as _load_proj_rates
            _project_rate_overrides = _load_proj_rates(
                org_id=tenant_id or "local",
                project_id=project_id or "",
            )
        except Exception as _pre:
            logger.debug("project_rates load skipped: %s", _pre)
        _qto_rated_items = apply_rates(
            list(_spec_items_list),
            region="tier1",
            project_rates=_project_rate_overrides or None,
        )
        _qto_trade_summary = compute_trade_summary(_qto_rated_items)
        _qto_grand_total_inr = sum(
            t.get("total_amount", 0) for t in _qto_trade_summary.values()
        )
        _spec_items_list = _qto_rated_items
        fire_sub("rate_engine", "done",
                 f"₹{_qto_grand_total_inr/1e7:.1f}Cr total", len(_qto_rated_items))
    except Exception as _rate_err:
        logger.warning("Rate engine failed: %s", _rate_err)
        fire_sub("rate_engine", "error", str(_rate_err)[:80])

    # Wire rate history comparison
    try:
        from src.analysis.rate_history import compare_to_history as _compare_to_history
        _rh_items_by_trade: dict = {}
        for _rh_item in _qto_rated_items:
            _rh_trade = _rh_item.get("trade", "general")
            _rh_items_by_trade.setdefault(_rh_trade, []).append(_rh_item)
        _rate_comparison_all: list = []
        for _rh_trade, _rh_trade_items in _rh_items_by_trade.items():
            _compare_to_history(_rh_trade_items, trade=_rh_trade)
            _rate_comparison_all.extend(_rh_trade_items)
        if _rate_comparison_all:
            _rh_flagged = sum(
                1 for r in _rate_comparison_all
                if r.get("hist_flag") in ("above", "below")
            )
            logger.info(
                "Rate history: %d items compared, %d flagged",
                len(_rate_comparison_all),
                _rh_flagged,
            )
    except Exception as _rh_err:
        logger.debug("Rate history comparison skipped: %s", _rh_err)

    # ── Sprint 41: Rebuild unified line_items after all QTO modules ──
    _dedup_stats: dict = {}
    _line_items_payload: list = []
    try:
        from ..item_normalizer import build_line_items as _build_line_items_full
        _full_unified = _build_line_items_full(
            boq_items      = extraction_result.boq_items if extraction_result else [],
            spec_items     = list(_spec_items_list),
            schedule_stubs = _stub_items_list if _recon else [],
            dedup          = True,
        )
        from ..item_normalizer import get_last_dedup_stats as _get_dedup_stats
        _dedup_stats = _get_dedup_stats()
        _line_items_payload = [li.to_dict() for li in _full_unified]
    except Exception as _rebuild_err:
        logger.warning("Sprint 41: line_items rebuild failed (non-critical): %s", _rebuild_err)

    # ── Excel Export (Sprint 38) ─────────────────────────────────
    _qto_excel_bytes: bytes = b""
    fire_sub("excel_exporter", "working", "exporting Excel BOQ")
    try:
        from ..export.excel_exporter import export_to_excel
        _excel_meta = {
            "total_area_sqm": _st_area_sqm,
            "floors": _st_floors,
        }
        _excel_result = export_to_excel(
            list(_spec_items_list),
            project_name="Project",
            project_meta=_excel_meta,
        )
        _qto_excel_bytes = _excel_result or b""
        fire_sub("excel_exporter", "done", f"{len(_spec_items_list)} items")
    except Exception as _excel_err:
        logger.warning("Excel export failed: %s", _excel_err)
        fire_sub("excel_exporter", "error", str(_excel_err)[:80])

    # ── Build qto_summary dict ────────────────────────────────────
    _qto_summary_dict: dict = {
        "rooms_detected": len(_qto_rooms),
        "finish_items_generated": len(_qto_finish_items),
        "structural_elements_detected": len(_qto_structural_elements),
        "structural_items_generated": len(_qto_structural_items),
        "structural_mode": _qto_structural_mode,
        "structural_warnings": _qto_structural_warnings,
        "implied_items_generated": len(_qto_implied_items),
        "implied_rules_triggered": _qto_implied_rules_triggered,
        "mep_elements_detected": len(_qto_mep_elements),
        "mep_items_generated": len(_qto_mep_items),
        "mep_mode": _qto_mep_mode,
        "mep_warnings": _qto_mep_warnings,
        "visual_elements_detected": len(_qto_visual_elements),
        "visual_items_generated": len(_qto_visual_items),
        "visual_mode": _qto_visual_mode,
        "visual_warnings": _qto_visual_warnings,
        "visual_scale": _qto_visual_scale,
        "visual_area_sqm": _qto_visual_area,
        "vmeas_rooms_measured": len(_qto_vmeas_rooms),
        "vmeas_items_generated": len(_qto_vmeas_items),
        "vmeas_mode": _qto_vmeas_mode,
        "vmeas_warnings": _qto_vmeas_warnings,
        "vmeas_scale": _qto_vmeas_scale,
        "vmeas_area_sqm": _qto_vmeas_area,
        "vmeas_room_schedule": [
            {
                "name": getattr(r, "name", ""),
                "raw_name": getattr(r, "raw_name", ""),
                "area_sqm": getattr(r, "area_sqm", 0),
                "dim_l": getattr(r, "dim_l", None),
                "dim_w": getattr(r, "dim_w", None),
                "source_page": getattr(r, "source_page", 0),
                "confidence": getattr(r, "confidence", 0),
            }
            for r in _qto_vmeas_room_schedule
        ],
        "dw_elements_detected": len(_qto_dw_elements),
        "dw_items_generated": len(_qto_dw_items),
        "dw_mode": _qto_dw_mode,
        "dw_warnings": _qto_dw_warnings,
        "dw_door_count": _qto_dw_door_count,
        "dw_window_count": _qto_dw_window_count,
        "paint_items_generated": len(_qto_paint_items),
        "paint_mode": _qto_paint_mode,
        "paint_warnings": _qto_paint_warnings,
        "paint_int_wall_sqm": _qto_paint_int_wall,
        "paint_ceiling_sqm": _qto_paint_ceiling,
        "paint_ext_wall_sqm": _qto_paint_ext_wall,
        "wp_items_generated": len(_qto_wp_items),
        "wp_mode": _qto_wp_mode,
        "wp_warnings": _qto_wp_warnings,
        "wp_wet_area_sqm": _qto_wp_wet_area,
        "wp_roof_area_sqm": _qto_wp_roof_area,
        "sw_items_generated": len(_qto_sw_items),
        "sw_mode": _qto_sw_mode,
        "sw_warnings": _qto_sw_warnings,
        "detected_scale": (
            {
                "ratio": _detected_scale.ratio,
                "is_nts": _detected_scale.is_nts,
                "px_per_mm": round(_detected_scale.px_per_mm, 6),
                "confidence": round(_detected_scale.confidence, 3),
                "source_page": _detected_scale.source_page,
                "source_text": _detected_scale.source_text,
            }
            if _detected_scale is not None else None
        ),
        "grand_total_inr": _qto_grand_total_inr,
        "area_assumed_default": (_st_area_sqm <= 0 and not _qto_visual_area and not _qto_vmeas_area),
        "trade_summary": {
            trade: {
                "item_count": info.get("item_count", 0),
                "total_amount": info.get("total_amount", 0),
            }
            for trade, info in _qto_trade_summary.items()
        },
        "total_spec_items": len(_spec_items_list),
        "_excel_available": len(_qto_excel_bytes) > 0,
    }

    # ── Return all outputs ────────────────────────────────────────
    return QTOOutputs(
        qto_rated_items      = _qto_rated_items,
        qto_grand_total_inr  = _qto_grand_total_inr,
        line_items_payload   = _line_items_payload,
        dedup_stats          = _dedup_stats,
        spec_needs_qty       = _spec_needs_qty,
        spec_params_payload  = _spec_params_payload,
        st_area_sqm          = _st_area_sqm,
        st_floors            = _st_floors,
        qto_paint_items      = _qto_paint_items,
        qto_wp_items         = _qto_wp_items,
        qto_dw_items         = _qto_dw_items,
        qto_mep_items        = _qto_mep_items,
        qto_sw_items         = _qto_sw_items,
        qto_brickwork_items  = _qto_brickwork_items,
        qto_plaster_items    = _qto_plaster_items,
        qto_earthwork_items  = _qto_earthwork_items,
        qto_flooring_items   = _qto_flooring_items,
        qto_foundation_items = _qto_foundation_items,
        qto_extdev_items     = _qto_extdev_items,
        qto_prelims_items    = _qto_prelims_items,
        qto_elv_items        = _qto_elv_items,
        qto_wp_wet_area      = _qto_wp_wet_area,
        qto_wp_roof_area     = _qto_wp_roof_area,
        qto_paint_int_wall   = _qto_paint_int_wall,
        qto_summary_dict     = _qto_summary_dict,
        qto_excel_bytes      = _qto_excel_bytes,
    )
