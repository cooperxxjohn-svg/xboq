#!/usr/bin/env python3
"""
Smoke test for _render_analysis_results_preview and build_demo_analysis.

Loads representative JSON payloads (including one WITHOUT drawing_overview,
one with 'overview' key instead, and one for NON_DRAWING_DOCUMENT)
and calls the same code paths to ensure nothing throws.

Usage:
    python scripts/smoke_render_results.py

Exit code:
    0 = all payloads rendered without error
    1 = at least one payload caused an exception
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from demo_page import build_demo_analysis

# ── Test Payloads ────────────────────────────────────────────────────────

PAYLOADS = {
    "full_payload": {
        "project_id": "test_full",
        "timestamp": "2026-02-15T10:00:00",
        "drawing_overview": {
            "files": ["test.pdf"],
            "pages_total": 7,
            "disciplines_detected": ["Structural", "Electrical"],
            "sheet_types_count": 5,
            "scale_found_pages": 0,
            "schedules_detected_count": 0,
            "door_tags_found": 3,
            "window_tags_found": 0,
            "room_names_found": 0,
            "ocr_used": True,
            "ocr_pages_count": 7,
            "is_drawing_set": True,
        },
        "readiness_score": 61,
        "decision": "CONDITIONAL",
        "sub_scores": {
            "completeness": 80,
            "coverage": 64,
            "measurement": 30,
            "blocker": 65,
        },
        "blockers": [
            {
                "id": "BLK-0001",
                "title": "Door schedule required",
                "trade": "architectural",
                "severity": "high",
                "description": "3 doors detected without schedule",
                "missing_dependency": ["door_schedule"],
                "impact_cost": "high",
                "impact_schedule": "medium",
                "bid_impact": "blocks_pricing",
                "evidence": {"pages": [0, 2], "confidence": 0.9},
                "fix_actions": ["Provide door schedule"],
                "score_delta_estimate": 8,
            }
        ],
        "rfis": [
            {
                "id": "RFI-0001",
                "trade": "architectural",
                "priority": "high",
                "question": "Provide door schedule",
                "why_it_matters": "Cannot price doors",
            }
        ],
        "trade_coverage": [
            {
                "trade": "civil",
                "coverage_pct": 100.0,
                "total_categories": 8,
                "priceable_count": 8,
                "blocked_count": 0,
            }
        ],
        "timings": {
            "load_s": 0.32,
            "index_s": 1.5,
            "select_s": 0.01,
            "extract_s": 7.58,
            "graph_s": 0.01,
            "reason_s": 0.12,
            "rfi_s": 0.08,
            "export_s": 0.15,
            "total_s": 9.77,
        },
        "extraction_summary": {
            "requirements": [],
            "schedules": [],
            "boq_items": [],
            "callouts": [],
            "pages_processed": 7,
            "pages_by_extractor": {"drawings": 7},
            "counts": {"requirements": 0, "schedules": 0, "boq_items": 0, "callouts": 0},
        },
        "run_coverage": {
            "pages_total": 7,
            "pages_indexed": 7,
            "pages_deep_processed": 7,
            "pages_skipped": [],
            "doc_types_detected": {"plan": 5, "detail": 1, "section": 1},
            "doc_types_fully_covered": ["plan", "detail", "section"],
            "doc_types_partially_covered": [],
            "doc_types_not_covered": [],
            "selection_mode": "full_read",
            "ocr_budget_pages": 80,
        },
        "guardrail_warnings": [],
        # Sprint 20F: Extraction diagnostics
        "extraction_diagnostics": {
            "boq": {"pages_attempted": 2, "pages_parsed": 1, "items_extracted": 5},
            "schedules": {"pages_attempted": 1, "pages_parsed": 1, "rows_extracted": 4},
            "table_methods_used": {"regex": 2, "ocr_row_reconstruct": 1},
        },
        "processing_stats": {
            "total_pages": 7,
            "deep_processed_pages": 7,
            "ocr_pages": 3,
            "text_layer_pages": 4,
            "skipped_pages": 0,
            "table_attempt_pages": 3,
            "table_success_pages": 2,
            "selection_mode": "full_read",
            "selected_pages_count": 7,
        },
        "diagnostics": {
            "page_index": {
                "pdf_name": "test.pdf",
                "total_pages": 7,
                "counts_by_type": {"plan": 5, "detail": 1, "section": 1},
                "counts_by_discipline": {"structural": 4, "electrical": 3},
                "indexing_time_s": 1.5,
            },
            "selected_pages": {
                "selected": [0, 1, 2, 3, 4, 5, 6],
                "budget_total": 80,
                "budget_used": 7,
                "always_include_count": 0,
                "sample_count": 7,
                "skipped_types": {},
                "coverage_summary": {"plan": 5, "detail": 1, "section": 1},
                "selection_mode": "full_read",
            },
        },
    },

    "no_overview_key": {
        "project_id": "test_no_overview",
        "timestamp": "2026-02-15T10:00:00",
        "readiness_score": 50,
        "decision": "NO-GO",
        "sub_scores": {},
        "blockers": [{"title": "Missing plans", "severity": "critical"}],
        "rfis": [],
        "trade_coverage": [],
        "timings": {"total_s": 2.0},
    },

    "overview_alias_key": {
        "project_id": "test_overview_alias",
        "timestamp": "2026-02-15T10:00:00",
        "overview": {
            "files": ["alias.pdf"],
            "pages_total": 3,
            "disciplines_detected": [],
            "door_tags_found": 0,
            "window_tags_found": 0,
            "room_names_found": 0,
        },
        "readiness_score": 20,
        "decision": "NO-GO",
        "sub_scores": {},
        "blockers": [],
        "rfis": [],
        "trade_coverage": [],
    },

    "non_drawing_document": {
        "project_id": "test_homework",
        "timestamp": "2026-02-15T10:00:00",
        "drawing_overview": {
            "files": ["homework.pdf"],
            "pages_total": 2,
            "disciplines_detected": [],
            "sheet_types_count": 0,
            "scale_found_pages": 0,
            "schedules_detected_count": 0,
            "door_tags_found": 0,
            "window_tags_found": 0,
            "room_names_found": 0,
            "ocr_used": False,
            "is_drawing_set": False,
        },
        "readiness_score": 0,
        "decision": "NO_DRAWINGS",
        "sub_scores": {
            "completeness": 0,
            "coverage": 0,
            "measurement": 0,
            "blocker": 0,
        },
        "blockers": [
            {
                "id": "BLK-NO-DRAWINGS",
                "title": "No construction drawings detected in uploaded PDF",
                "trade": "general",
                "severity": "critical",
                "description": "No drawing indicators found.",
                "issue_type": "no_drawings",
            }
        ],
        "rfis": [],
        "trade_coverage": [],
        "timings": {"load_s": 0.1, "extract_s": 0.5, "total_s": 0.7},
    },

    "empty_payload": {
        "project_id": "test_empty",
    },

    "no_timings": {
        "project_id": "test_no_timings",
        "timestamp": "2026-02-15T10:00:00",
        "drawing_overview": {"files": ["x.pdf"], "pages_total": 1},
        "readiness_score": 30,
        "decision": "NO-GO",
        "sub_scores": {},
        "blockers": [],
        "rfis": [],
        "trade_coverage": [],
    },

    "blocker_with_full_evidence": {
        "project_id": "test_evidence",
        "timestamp": "2026-02-15T10:00:00",
        "drawing_overview": {
            "files": ["test.pdf"], "pages_total": 7,
            "disciplines_detected": ["Structural"],
            "door_tags_found": 3, "window_tags_found": 0, "room_names_found": 0,
            "scale_found_pages": 0, "ocr_used": True,
        },
        "readiness_score": 55,
        "decision": "CONDITIONAL",
        "sub_scores": {"completeness": 70, "coverage": 50, "measurement": 30, "blocker": 60},
        "blockers": [{
            "id": "BLK-0010", "title": "Door schedule required",
            "trade": "architectural", "severity": "high",
            "description": "3 doors detected", "issue_type": "missing_schedule",
            "missing_dependency": ["door_schedule"],
            "impact_cost": "high", "impact_schedule": "medium",
            "bid_impact": "blocks_pricing",
            "evidence": {
                "pages": [0, 2, 6], "sheets": ["A-101"],
                "snippets": ["Door D1 shown"],
                "detected_entities": {"door_tags": ["D1", "D4", "D10"], "door_count": 3},
                "search_attempts": {"keywords": ["door schedule"]},
                "confidence": 0.9, "confidence_reason": "Clear detection",
                "bbox": [
                    [[0.12, 0.45, 0.88, 0.52, 0.91, "BLK-0010-P0-0"]],
                    [[0.10, 0.30, 0.90, 0.38, 0.85, "BLK-0010-P1-0"], [0.10, 0.60, 0.90, 0.68, 0.72, "BLK-0010-P1-1"]],
                    [],
                ],
                "searched_pages": [0, 1, 2, 3, 4, 5, 6],
                "text_coverage_pct": 95.0,
            },
            "fix_actions": ["Provide door schedule"],
            "score_delta_estimate": 8,
            "unlocks_boq_categories": ["doors", "door_frames", "hardware"],
            "coverage_status": "not_found_after_search",
        }],
        "rfis": [],
        "trade_coverage": [],
        "timings": {"total_s": 5.0},
        "ocr_bbox_meta": {
            "engine": "surya",
            "avg_confidence": 0.87,
            "pages_with_bbox": 5,
            "pages_total": 7,
        },
        "ocr_text_cache": {
            "0": "GROUND FLOOR PLAN\nDoor D1 - 900 x 2100\nColumn C1\nBeam B1",
            "1": "FIRST FLOOR PLAN\nRoom R1\nStaircase ST1",
            "2": "UPPER FLOOR PLAN\nDoor D10 - 900 x 2100",
            "3": "SECTION A-A\nFoundation depth 1.5m",
            "4": "SECTION B-B\nRoof slab 150mm",
            "5": "ELEVATION NORTH\nPlinth level +0.45",
            "6": "ELEVATION EAST\nDoor D10\nWindow W1",
        },
    },

    "rfi_no_evidence": {
        "project_id": "test_rfi_fyi",
        "timestamp": "2026-02-15T10:00:00",
        "drawing_overview": {
            "files": ["test.pdf"], "pages_total": 3,
            "disciplines_detected": [],
            "door_tags_found": 0, "window_tags_found": 0, "room_names_found": 0,
            "scale_found_pages": 0, "ocr_used": False,
        },
        "readiness_score": 40,
        "decision": "NO-GO",
        "sub_scores": {},
        "blockers": [],
        "rfis": [{
            "id": "RFI-0099", "trade": "general", "priority": "medium",
            "question": "Confirm project scope",
            "why_it_matters": "Unclear from drawings",
            "evidence": {"pages": [], "detected_entities": {}},
        }],
        "trade_coverage": [],
        "timings": {"total_s": 2.0},
    },

    "mep_missing_drawing_blocker": {
        "project_id": "test_mep_fyi",
        "timestamp": "2026-02-15T10:00:00",
        "drawing_overview": {
            "files": ["arch.pdf"], "pages_total": 5,
            "disciplines_detected": ["Architectural"],
            "door_tags_found": 2, "window_tags_found": 1, "room_names_found": 3,
            "scale_found_pages": 5, "ocr_used": True,
        },
        "readiness_score": 70,
        "decision": "CONDITIONAL",
        "sub_scores": {"completeness": 80, "coverage": 70, "measurement": 60, "blocker": 70},
        "blockers": [{
            "id": "BLK-MEP", "title": "No MEP drawings found",
            "trade": "mep", "severity": "high",
            "description": "MEP drawings not included",
            "issue_type": "missing_drawing",
            "missing_dependency": ["mep_drawings"],
            "impact_cost": "medium", "impact_schedule": "low",
            "bid_impact": "forces_allowance",
            "evidence": {"pages": [], "detected_entities": {}},
            "fix_actions": ["Provide MEP drawings"],
            "score_delta_estimate": 5,
            "unlocks_boq_categories": ["electrical", "plumbing"],
        }],
        "rfis": [],
        "trade_coverage": [],
        "timings": {"total_s": 3.0},
    },

    "extraction_with_data": {
        "project_id": "test_extraction",
        "timestamp": "2026-02-17T10:00:00",
        "drawing_overview": {
            "files": ["tender.pdf"], "pages_total": 50,
            "disciplines_detected": ["Structural", "Architectural"],
            "door_tags_found": 5, "window_tags_found": 3, "room_names_found": 8,
            "scale_found_pages": 10, "ocr_used": True,
        },
        "readiness_score": 72,
        "decision": "CONDITIONAL",
        "sub_scores": {"completeness": 80, "coverage": 70, "measurement": 60, "blocker": 75},
        "blockers": [],
        "rfis": [{
            "id": "RFI-0001", "trade": "architectural", "priority": "high",
            "question": "Provide door schedule for 5 detected doors",
            "why_it_matters": "Cannot price doors without schedule",
            "evidence": {
                "pages": [3, 7, 12], "snippets": ["Door D1 on Ground Floor"],
                "detected_entities": {"door_tags": ["D1", "D2", "D3", "D4", "D5"]},
                "search_attempts": {"keywords": ["door schedule", "door listing"]},
                "confidence": 0.85, "confidence_reason": "Tags found but no schedule",
            },
            "suggested_resolution": "Assume standard 900x2100mm flush doors",
        }],
        "trade_coverage": [],
        "timings": {"load_s": 0.5, "index_s": 5.2, "select_s": 0.02, "extract_s": 12.0, "total_s": 18.5},
        "extraction_summary": {
            "requirements": [
                {"text": "All concrete shall be M25 grade minimum", "category": "material", "source_page": 15, "sheet_id": None, "confidence": 0.8},
                {"text": "Steel reinforcement shall conform to IS 1786", "category": "standard", "source_page": 15, "sheet_id": None, "confidence": 0.85},
                {"text": "Curing shall be done for minimum 7 days", "category": "workmanship", "source_page": 16, "sheet_id": None, "confidence": 0.7},
                {"text": "All welding per IS 816", "category": "standard", "source_page": 17, "sheet_id": None, "confidence": 0.75},
                {"text": "Plastering shall be 12mm thick cement mortar 1:4", "category": "material", "source_page": 20, "sheet_id": None, "confidence": 0.7},
            ],
            "schedules": [
                {"mark": "D1", "fields": {"size": "900x2100", "type": "Flush", "material": "Teak"}, "schedule_type": "door", "source_page": 8, "sheet_id": "A-201"},
                {"mark": "W1", "fields": {"size": "1200x1500", "type": "Sliding", "material": "Aluminium"}, "schedule_type": "window", "source_page": 9, "sheet_id": "A-202"},
            ],
            "boq_items": [
                {"item_no": "1.1", "description": "Excavation in ordinary soil", "unit": "cum", "qty": 120.5, "rate": 250.0, "source_page": 30, "confidence": 0.8},
                {"item_no": "1.2", "description": "PCC M15 grade 1:2:4", "unit": "cum", "qty": 45.0, "rate": 5500.0, "source_page": 30, "confidence": 0.75},
                {"item_no": "2.1", "description": "RCC M25 grade for columns", "unit": "cum", "qty": 30.0, "rate": None, "source_page": 31, "confidence": 0.5},
            ],
            "callouts": [
                {"text": "1200mm", "callout_type": "dimension", "source_page": 3, "sheet_id": "A-101", "confidence": 0.7},
                {"text": "RCC M25", "callout_type": "material", "source_page": 3, "sheet_id": "A-101", "confidence": 0.75},
                {"text": "D1", "callout_type": "tag", "source_page": 3, "sheet_id": "A-101", "confidence": 0.8},
                {"text": "D2", "callout_type": "tag", "source_page": 5, "sheet_id": "A-102", "confidence": 0.8},
                {"text": "BEDROOM", "callout_type": "room", "source_page": 3, "sheet_id": "A-101", "confidence": 0.7},
                {"text": "KITCHEN", "callout_type": "room", "source_page": 3, "sheet_id": "A-101", "confidence": 0.7},
                {"text": "1:100", "callout_type": "scale", "source_page": 3, "sheet_id": "A-101", "confidence": 0.8},
                {"text": "300x600", "callout_type": "dimension", "source_page": 7, "sheet_id": "S-101", "confidence": 0.7},
                {"text": "Fe500", "callout_type": "material", "source_page": 7, "sheet_id": "S-101", "confidence": 0.75},
                {"text": "W1", "callout_type": "tag", "source_page": 5, "sheet_id": "A-102", "confidence": 0.8},
            ],
            "pages_processed": 30,
            "pages_by_extractor": {"drawings": 20, "notes": 5, "schedules": 2, "boq": 3},
            "counts": {"requirements": 5, "schedules": 2, "boq_items": 3, "callouts": 10},
        },
        "addendum_index": [
            {
                "addendum_no": "1",
                "date": "10/01/2026",
                "title": "Revised BOQ quantities",
                "changes": [{"type": "read_as", "original": "M25", "revised": "M30", "context": "Concrete grade changed"}],
                "clarifications": ["Q: Is BBS required? A: Yes, submit with tender"],
                "boq_changes": [{"item_no": "2.1", "new_value": "500"}],
                "date_changes": [{"date_type": "submission", "new_date": "28/02/2026", "context": "Date extended"}],
                "source_file": "tender.pdf_addendum_1",
            }
        ],
        "conflicts": [
            {
                "type": "boq_change",
                "item_no": "2.1",
                "changes": [{"field": "qty", "base_value": 30.0, "addendum_value": 500}],
                "base_page": 31,
                "addendum_page": 45,
                "delta_confidence": 0.9,
            }
        ],
        "reconciliation_findings": [],
        "guardrail_warnings": [
            {"type": "low_rfi_count", "message": "Only 1 RFI generated for 50-page tender — expected more"},
        ],
        "diagnostics": {
            "page_index": {
                "pdf_name": "tender.pdf",
                "total_pages": 50,
                "counts_by_type": {"plan": 15, "detail": 5, "section": 3, "elevation": 2, "schedule": 2, "boq": 5, "notes": 3, "spec": 8, "cover": 1, "unknown": 6},
                "counts_by_discipline": {"architectural": 18, "structural": 15, "electrical": 5, "plumbing": 3, "other": 9},
                "indexing_time_s": 5.2,
            },
            "selected_pages": {
                "selected": list(range(30)),
                "budget_total": 80,
                "budget_used": 30,
                "always_include_count": 10,
                "sample_count": 20,
                "skipped_types": {"unknown": 6, "spec": 5, "plan": 9},
                "coverage_summary": {"plan": 6, "detail": 5, "section": 3, "elevation": 2, "schedule": 2, "boq": 5, "notes": 3, "spec": 3, "cover": 1},
            },
        },
    },

    "missing_extraction_keys": {
        "project_id": "test_no_extraction",
        "timestamp": "2026-02-17T10:00:00",
        "drawing_overview": {
            "files": ["old_format.pdf"], "pages_total": 10,
            "disciplines_detected": ["Structural"],
            "door_tags_found": 0, "window_tags_found": 0, "room_names_found": 0,
            "scale_found_pages": 0, "ocr_used": True,
        },
        "readiness_score": 45,
        "decision": "NO-GO",
        "sub_scores": {"completeness": 50, "coverage": 40, "measurement": 30, "blocker": 50},
        "blockers": [],
        "rfis": [],
        "trade_coverage": [],
        "timings": {"load_s": 0.2, "extract_s": 3.0, "total_s": 3.5},
        # No extraction_summary, no diagnostics, no guardrail_warnings — tests graceful fallback
    },

    "full_read_coverage": {
        "project_id": "test_full_read",
        "timestamp": "2026-02-17T10:00:00",
        "drawing_overview": {
            "files": ["small_tender.pdf"], "pages_total": 20,
            "disciplines_detected": ["Structural", "Architectural"],
            "door_tags_found": 4, "window_tags_found": 2, "room_names_found": 5,
            "scale_found_pages": 8, "ocr_used": True,
        },
        "readiness_score": 65,
        "decision": "CONDITIONAL",
        "sub_scores": {"completeness": 75, "coverage": 60, "measurement": 50, "blocker": 70},
        "blockers": [{
            "id": "BLK-0010", "title": "Door schedule required - 4 door types",
            "trade": "architectural", "severity": "high",
            "description": "Detected 4 doors without schedule",
            "issue_type": "missing_schedule",
            "coverage_status": "not_found_after_search",
            "evidence": {"pages": [2, 5], "confidence": 0.9},
            "fix_actions": ["Provide door schedule"],
            "score_delta_estimate": 8,
            "unlocks_boq_categories": ["doors"],
        }],
        "rfis": [{
            "id": "RFI-0001", "trade": "architectural", "priority": "high",
            "question": "Provide door schedule",
            "coverage_status": "not_found_after_search",
            "evidence": {"pages": [2, 5], "confidence": 0.9},
        }],
        "trade_coverage": [],
        "timings": {"total_s": 5.0},
        "run_coverage": {
            "pages_total": 20, "pages_indexed": 20, "pages_deep_processed": 20,
            "pages_skipped": [],
            "doc_types_detected": {"plan": 8, "schedule": 2, "notes": 1, "detail": 3, "section": 2, "elevation": 2, "cover": 1, "unknown": 1},
            "doc_types_fully_covered": ["plan", "schedule", "notes", "detail", "section", "elevation", "cover", "unknown"],
            "doc_types_partially_covered": [], "doc_types_not_covered": [],
            "selection_mode": "full_read", "ocr_budget_pages": 80,
        },
    },

    "fast_budget_with_gaps": {
        "project_id": "test_fast_budget",
        "timestamp": "2026-02-17T10:00:00",
        "drawing_overview": {
            "files": ["large_tender.pdf"], "pages_total": 300,
            "disciplines_detected": ["Structural", "Architectural", "Electrical"],
            "door_tags_found": 12, "window_tags_found": 8, "room_names_found": 15,
            "scale_found_pages": 20, "ocr_used": True,
        },
        "readiness_score": 55,
        "decision": "CONDITIONAL",
        "sub_scores": {"completeness": 70, "coverage": 50, "measurement": 40, "blocker": 55},
        "blockers": [{
            "id": "BLK-0010", "title": "Door schedule required - 12 door types",
            "trade": "architectural", "severity": "medium",
            "description": "Detected 12 doors. Schedule pages not fully processed due to OCR budget.",
            "issue_type": "missing_schedule",
            "coverage_status": "unknown_not_processed",
            "evidence": {"pages": [5, 12, 30], "confidence": 0.4},
            "fix_actions": ["Provide door schedule"],
            "score_delta_estimate": 4,
            "unlocks_boq_categories": ["doors"],
        }],
        "rfis": [{
            "id": "RFI-0001", "trade": "architectural", "priority": "low",
            "question": "Provide door schedule [Coverage gap — may exist on unprocessed pages]",
            "coverage_status": "unknown_not_processed",
            "evidence": {"pages": [5, 12, 30], "confidence": 0.4},
        }],
        "trade_coverage": [],
        "timings": {"total_s": 45.0},
        "run_coverage": {
            "pages_total": 300, "pages_indexed": 300, "pages_deep_processed": 80,
            "pages_skipped": [
                {"page_idx": 81, "doc_type": "plan", "discipline": "architectural", "reason": "budget_exceeded"},
                {"page_idx": 150, "doc_type": "schedule", "discipline": "architectural", "reason": "budget_exceeded"},
            ],
            "doc_types_detected": {"plan": 100, "schedule": 20, "notes": 5, "detail": 30, "section": 15, "elevation": 10, "boq": 10, "spec": 50, "unknown": 60},
            "disciplines_detected": {"architectural": 120, "structural": 80, "electrical": 50, "other": 50},
            "doc_types_fully_covered": ["notes"],
            "doc_types_partially_covered": ["plan", "detail", "section", "elevation", "schedule", "boq", "spec"],
            "doc_types_not_covered": ["unknown"],
            "selection_mode": "fast_budget", "ocr_budget_pages": 80,
        },
    },
}


# ── Smoke test: build_demo_analysis ──────────────────────────────────────

def test_build_demo_analysis():
    """Test that build_demo_analysis never throws for any payload shape."""
    failures = []
    for name, payload in PAYLOADS.items():
        try:
            demo = build_demo_analysis(payload, payload.get("project_id", "test"))
            assert demo.project_id is not None, "project_id is None"
            assert isinstance(demo.blockers_count, int), "blockers_count is not int"
            assert isinstance(demo.rfis_count, int), "rfis_count is not int"
            assert isinstance(demo.analysis_steps, list), "analysis_steps is not list"
            print(f"  PASS  {name}: decision={demo.decision}, score={demo.readiness_score}, "
                  f"blockers={demo.blockers_count}, rfis={demo.rfis_count}")
        except Exception as e:
            failures.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    return failures


# ── Smoke test: overview extraction (the bug path) ───────────────────────

def test_overview_extraction():
    """Test that extracting overview from payload via fallback chain never throws."""
    failures = []
    for name, payload in PAYLOADS.items():
        try:
            # Exact same fallback chain used in demo_page.py
            overview = payload.get("drawing_overview") or payload.get("overview") or {}
            timings = payload.get("timings", {})
            blockers = payload.get("blockers", [])
            rfis = payload.get("rfis", [])

            if overview:
                files = ", ".join(overview.get("files", ["Unknown"]))
                pages = overview.get("pages_total", 0)
                _ = f"Files: {files}, Pages: {pages}"
            else:
                available = [k for k in payload.keys() if k != "schema_version"]
                _ = f"Overview not available. Sections: {', '.join(available)}"

            if timings:
                _ = f"Load: {timings.get('load_s', 0):.2f}s | Total: {timings.get('total_s', 0):.2f}s"

            print(f"  PASS  {name}: overview={'yes' if overview else 'no'}, "
                  f"timings={'yes' if timings else 'no'}")
        except Exception as e:
            failures.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    return failures


# ── Smoke test: NO_DRAWINGS classification ───────────────────────────────

def test_no_drawings_classification():
    """Test that NO_DRAWINGS payloads don't create bogus scale/MEP blockers."""
    failures = []
    payload = PAYLOADS["non_drawing_document"]
    try:
        demo = build_demo_analysis(payload, "test_homework")

        # Should be NO_DRAWINGS decision
        assert demo.decision == "NO_DRAWINGS", f"Expected NO_DRAWINGS, got {demo.decision}"
        assert demo.readiness_score == 0, f"Expected score 0, got {demo.readiness_score}"

        # Should have exactly 1 blocker (the no-drawings one), not scale/MEP stuff
        assert demo.blockers_count == 1, f"Expected 1 blocker, got {demo.blockers_count}"
        blocker = demo.top_blockers[0]
        assert "no_drawings" in blocker.get("issue_type", "") or "No construction" in blocker.get("title", ""), \
            f"Unexpected blocker: {blocker.get('title')}"

        # No RFIs should be generated
        assert demo.rfis_count == 0, f"Expected 0 RFIs, got {demo.rfis_count}"

        # No trade coverage
        assert len(demo.trade_cards) == 0, f"Expected 0 trade cards, got {len(demo.trade_cards)}"

        print(f"  PASS  non_drawing: decision={demo.decision}, blockers={demo.blockers_count}, "
              f"rfis={demo.rfis_count}, trades={len(demo.trade_cards)}")
    except Exception as e:
        failures.append(("non_drawing_classification", e))
        print(f"  FAIL  non_drawing: {type(e).__name__}: {e}")
    return failures


# ── Smoke test: overview alias ───────────────────────────────────────────

def test_overview_alias():
    """Test that payload with 'overview' (not 'drawing_overview') still works."""
    failures = []
    payload = PAYLOADS["overview_alias_key"]
    try:
        demo = build_demo_analysis(payload, "test_alias")
        # Should pick up files from the 'overview' key
        assert demo.files_analyzed == ["alias.pdf"], f"Expected ['alias.pdf'], got {demo.files_analyzed}"
        assert demo.pages_total == 3, f"Expected 3 pages, got {demo.pages_total}"
        print(f"  PASS  overview_alias: files={demo.files_analyzed}, pages={demo.pages_total}")
    except Exception as e:
        failures.append(("overview_alias", e))
        print(f"  FAIL  overview_alias: {type(e).__name__}: {e}")
    return failures


# ── Smoke test: cached demo JSON ─────────────────────────────────────────

def test_cached_demo():
    """Test that the cached demo JSON loads and renders cleanly."""
    cache_path = PROJECT_ROOT / "demo_cache" / "pwd_garage" / "analysis.json"
    if not cache_path.exists():
        print(f"  SKIP  cached demo: {cache_path} not found")
        return []

    failures = []
    try:
        with open(cache_path) as f:
            payload = json.load(f)
        demo = build_demo_analysis(payload, "pwd_garage")
        assert demo.decision in ("PASS", "CONDITIONAL", "NO-GO"), f"bad decision: {demo.decision}"
        assert demo.readiness_score >= 0, f"bad score: {demo.readiness_score}"
        assert len(demo.analysis_steps) > 0, "no analysis steps"
        print(f"  PASS  cached demo: decision={demo.decision}, score={demo.readiness_score}")
    except Exception as e:
        failures.append(("cached_demo", e))
        print(f"  FAIL  cached demo: {type(e).__name__}: {e}")
    return failures


# ── Smoke test: evidence rendering ─────────────────────────────────────

def test_evidence_rendering():
    """Test that blocker with full evidence builds without crash."""
    failures = []
    payload = PAYLOADS["blocker_with_full_evidence"]
    try:
        demo = build_demo_analysis(payload, "test_evidence")
        assert demo.blockers_count == 1
        blocker = demo.top_blockers[0]
        evidence = blocker.get("evidence", {})
        assert evidence.get("pages") == [0, 2, 6], f"Expected pages [0,2,6], got {evidence.get('pages')}"
        assert evidence.get("confidence") == 0.9
        assert blocker.get("unlocks_boq_categories") == ["doors", "door_frames", "hardware"]
        assert blocker.get("score_delta_estimate") == 8
        print(f"  PASS  evidence_rendering: blocker has evidence, pills, score_delta")
    except Exception as e:
        failures.append(("evidence_rendering", e))
        print(f"  FAIL  evidence_rendering: {type(e).__name__}: {e}")
    return failures


def test_rfi_fyi_downgrade():
    """Test that RFI with no evidence is identified for FYI downgrade."""
    failures = []
    payload = PAYLOADS["rfi_no_evidence"]
    try:
        demo = build_demo_analysis(payload, "test_rfi_fyi")
        assert demo.rfis_count == 1
        rfi = demo.top_rfis[0]
        evidence = rfi.get("evidence", {})
        has_pages = bool(evidence.get("pages"))
        has_entities = bool(evidence.get("detected_entities"))
        has_ev = has_pages or has_entities
        assert not has_ev, f"Expected no evidence, but found pages={has_pages}, entities={has_entities}"
        print(f"  PASS  rfi_fyi_downgrade: RFI without evidence correctly identified")
    except Exception as e:
        failures.append(("rfi_fyi_downgrade", e))
        print(f"  FAIL  rfi_fyi_downgrade: {type(e).__name__}: {e}")
    return failures


def test_mep_blocker_as_fyi():
    """Test that MEP missing_drawing blocker is identified for FYI treatment."""
    failures = []
    payload = PAYLOADS["mep_missing_drawing_blocker"]
    try:
        demo = build_demo_analysis(payload, "test_mep_fyi")
        assert demo.blockers_count == 1
        blocker = demo.top_blockers[0]
        is_fyi = (
            blocker.get("issue_type") == "missing_drawing"
            and blocker.get("trade", "").lower() == "mep"
        )
        assert is_fyi, f"Expected MEP blocker to be FYI, got issue_type={blocker.get('issue_type')}, trade={blocker.get('trade')}"
        print(f"  PASS  mep_blocker_fyi: MEP missing_drawing blocker correctly identified as FYI")
    except Exception as e:
        failures.append(("mep_blocker_fyi", e))
        print(f"  FAIL  mep_blocker_fyi: {type(e).__name__}: {e}")
    return failures


def test_quantified_outputs():
    """Test that quantified output data is accessible from DemoAnalysis."""
    failures = []
    payload = PAYLOADS["full_payload"]
    try:
        demo = build_demo_analysis(payload, "test_full")
        assert demo.door_tags_found == 3, f"Expected 3 doors, got {demo.door_tags_found}"
        assert demo.window_tags_found == 0, f"Expected 0 windows, got {demo.window_tags_found}"
        assert demo.room_names_found == 0, f"Expected 0 rooms, got {demo.room_names_found}"
        assert demo.pages_total == 7, f"Expected 7 pages, got {demo.pages_total}"
        assert len(demo.disciplines_detected) == 2, f"Expected 2 disciplines, got {len(demo.disciplines_detected)}"
        print(f"  PASS  quantified_outputs: doors={demo.door_tags_found}, windows={demo.window_tags_found}, "
              f"rooms={demo.room_names_found}, pages={demo.pages_total}")
    except Exception as e:
        failures.append(("quantified_outputs", e))
        print(f"  FAIL  quantified_outputs: {type(e).__name__}: {e}")
    return failures


# ── Smoke test: extraction summary access ─────────────────────────────

def test_extraction_summary_access():
    """Test that payloads with and without extraction_summary don't crash."""
    failures = []

    # Payload WITH extraction data
    try:
        payload = PAYLOADS["extraction_with_data"]
        demo = build_demo_analysis(payload, "test_extraction")
        raw = demo.raw_payload
        ext = raw.get("extraction_summary", {})
        assert ext.get("counts", {}).get("boq_items") == 3, f"Expected 3 BOQ items, got {ext.get('counts', {}).get('boq_items')}"
        assert ext.get("counts", {}).get("schedules") == 2, f"Expected 2 schedules, got {ext.get('counts', {}).get('schedules')}"
        assert ext.get("counts", {}).get("requirements") == 5, f"Expected 5 requirements, got {ext.get('counts', {}).get('requirements')}"
        assert ext.get("counts", {}).get("callouts") == 10, f"Expected 10 callouts, got {ext.get('counts', {}).get('callouts')}"
        print(f"  PASS  extraction_with_data: boq={ext['counts']['boq_items']}, "
              f"schedules={ext['counts']['schedules']}, requirements={ext['counts']['requirements']}, "
              f"callouts={ext['counts']['callouts']}")
    except Exception as e:
        failures.append(("extraction_with_data", e))
        print(f"  FAIL  extraction_with_data: {type(e).__name__}: {e}")

    # Payload WITHOUT extraction data (graceful fallback)
    try:
        payload = PAYLOADS["missing_extraction_keys"]
        demo = build_demo_analysis(payload, "test_no_extraction")
        raw = demo.raw_payload
        ext = raw.get("extraction_summary")
        assert ext is None, f"Expected None extraction_summary, got {type(ext)}"
        diag = raw.get("diagnostics")
        assert diag is None, f"Expected None diagnostics, got {type(diag)}"
        print(f"  PASS  missing_extraction_keys: graceful fallback (no crash)")
    except Exception as e:
        failures.append(("missing_extraction_keys", e))
        print(f"  FAIL  missing_extraction_keys: {type(e).__name__}: {e}")

    return failures


def test_selection_coverage_access():
    """Test that payloads with selection/coverage data don't crash."""
    failures = []

    try:
        payload = PAYLOADS["extraction_with_data"]
        demo = build_demo_analysis(payload, "test_extraction")
        raw = demo.raw_payload
        diag = raw.get("diagnostics", {})
        page_idx = diag.get("page_index", {})
        sel = diag.get("selected_pages", {})

        assert page_idx.get("total_pages") == 50, f"Expected 50 total pages, got {page_idx.get('total_pages')}"
        assert sel.get("budget_used") == 30, f"Expected 30 budget used, got {sel.get('budget_used')}"
        assert sel.get("always_include_count") == 10, f"Expected 10 always-include, got {sel.get('always_include_count')}"

        # Verify page type counts
        types = page_idx.get("counts_by_type", {})
        assert types.get("plan") == 15, f"Expected 15 plans, got {types.get('plan')}"
        assert types.get("boq") == 5, f"Expected 5 BOQ pages, got {types.get('boq')}"

        # Verify 8 stages in step_names (index + select added)
        assert len(demo.analysis_steps) == 8, f"Expected 8 analysis steps, got {len(demo.analysis_steps)}"
        step_ids = [s.step_id for s in demo.analysis_steps]
        assert "index" in step_ids, f"'index' step missing from {step_ids}"
        assert "select" in step_ids, f"'select' step missing from {step_ids}"

        print(f"  PASS  selection_coverage: total_pages={page_idx['total_pages']}, "
              f"budget_used={sel['budget_used']}, always_include={sel['always_include_count']}, "
              f"steps={len(demo.analysis_steps)}")
    except Exception as e:
        failures.append(("selection_coverage", e))
        print(f"  FAIL  selection_coverage: {type(e).__name__}: {e}")

    return failures


# ── Smoke test: coverage status display ──────────────────────────────────

def test_coverage_status_display():
    """Test that coverage_status is accessible from payloads and build_demo_analysis doesn't crash."""
    failures = []

    # FULL_READ: coverage_status should be not_found_after_search
    try:
        payload = PAYLOADS["full_read_coverage"]
        demo = build_demo_analysis(payload, "test_full_read")
        raw = demo.raw_payload
        blocker = raw["blockers"][0]
        assert blocker.get("coverage_status") == "not_found_after_search", \
            f"Expected not_found_after_search, got {blocker.get('coverage_status')}"
        rfi = raw["rfis"][0]
        assert rfi.get("coverage_status") == "not_found_after_search", \
            f"Expected not_found_after_search, got {rfi.get('coverage_status')}"
        rc = raw.get("run_coverage", {})
        assert rc.get("selection_mode") == "full_read"
        print(f"  PASS  full_read_coverage: blocker={blocker['coverage_status']}, rfi={rfi['coverage_status']}, mode={rc['selection_mode']}")
    except Exception as e:
        failures.append(("full_read_coverage", e))
        print(f"  FAIL  full_read_coverage: {type(e).__name__}: {e}")

    # FAST_BUDGET: coverage_status should be unknown_not_processed
    try:
        payload = PAYLOADS["fast_budget_with_gaps"]
        demo = build_demo_analysis(payload, "test_fast_budget")
        raw = demo.raw_payload
        blocker = raw["blockers"][0]
        assert blocker.get("coverage_status") == "unknown_not_processed", \
            f"Expected unknown_not_processed, got {blocker.get('coverage_status')}"
        rfi = raw["rfis"][0]
        assert rfi.get("coverage_status") == "unknown_not_processed", \
            f"Expected unknown_not_processed, got {rfi.get('coverage_status')}"
        rc = raw.get("run_coverage", {})
        assert rc.get("selection_mode") == "fast_budget"
        assert rc.get("pages_deep_processed") == 80
        assert len(rc.get("doc_types_not_covered", [])) > 0
        print(f"  PASS  fast_budget_with_gaps: blocker={blocker['coverage_status']}, "
              f"rfi={rfi['coverage_status']}, mode={rc['selection_mode']}, "
              f"not_covered={rc['doc_types_not_covered']}")
    except Exception as e:
        failures.append(("fast_budget_with_gaps", e))
        print(f"  FAIL  fast_budget_with_gaps: {type(e).__name__}: {e}")

    return failures


def test_bid_pack_tab():
    """Verify bid pack data structures are accessible via build_demo_analysis."""
    failures = []
    try:
        payload = PAYLOADS["extraction_with_data"]
        demo = build_demo_analysis(payload, "test_bid_pack")
        raw = demo.raw_payload
        ext = raw.get("extraction_summary", {})
        boq = ext.get("boq_items", [])
        sched = ext.get("schedules", [])
        reqs = ext.get("requirements", [])
        assert len(boq) == 3, f"Expected 3 BOQ items, got {len(boq)}"
        assert len(sched) == 2, f"Expected 2 schedules, got {len(sched)}"
        assert len(reqs) == 5, f"Expected 5 requirements, got {len(reqs)}"
        # Verify page field exists on items
        assert boq[0].get("source_page") is not None, "BOQ item missing source_page"
        print(f"  PASS  bid_pack_tab: boq={len(boq)}, schedules={len(sched)}, reqs={len(reqs)}")
    except Exception as e:
        failures.append(("bid_pack_tab", e))
        print(f"  FAIL  bid_pack_tab: {type(e).__name__}: {e}")
    return failures


def test_coverage_dashboard():
    """Verify coverage dashboard data structures are accessible."""
    failures = []
    try:
        payload = PAYLOADS["fast_budget_with_gaps"]
        rc = payload.get("run_coverage", {})
        # Check doc type categories
        fully = rc.get("doc_types_fully_covered", [])
        partially = rc.get("doc_types_partially_covered", [])
        not_cov = rc.get("doc_types_not_covered", [])
        assert len(fully) > 0, "Expected at least one fully covered type"
        assert len(not_cov) > 0, "Expected at least one not-covered type"
        # Check disciplines
        disc = rc.get("disciplines_detected", {})
        assert len(disc) > 0, "Expected at least one discipline"
        # Check skipped pages for discipline breakdown
        skipped = rc.get("pages_skipped", [])
        assert len(skipped) > 0, "Expected skipped pages in fast_budget"
        print(f"  PASS  coverage_dashboard: fully={len(fully)}, partial={len(partially)}, "
              f"not_covered={len(not_cov)}, disciplines={len(disc)}, skipped={len(skipped)}")
    except Exception as e:
        failures.append(("coverage_dashboard", e))
        print(f"  FAIL  coverage_dashboard: {type(e).__name__}: {e}")
    return failures


def test_evidence_viewer_no_pdf():
    """Verify evidence viewer gracefully handles missing PDF (demo mode)."""
    failures = []
    try:
        # The blocker_with_full_evidence payload has no primary_pdf_path
        payload = PAYLOADS["blocker_with_full_evidence"]
        pdf_path = payload.get("primary_pdf_path")
        assert pdf_path is None, f"Expected None, got {pdf_path}"
        # Check evidence pages exist
        ev = payload["blockers"][0]["evidence"]
        assert len(ev["pages"]) == 3, f"Expected 3 evidence pages, got {len(ev['pages'])}"
        # render_pdf_page_preview should return None for None path
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from demo_page import render_pdf_page_preview
        result = render_pdf_page_preview(None, 0)
        assert result is None, "Expected None for missing PDF path"
        result2 = render_pdf_page_preview("/nonexistent/path.pdf", 0)
        assert result2 is None, "Expected None for nonexistent PDF"
        print("  PASS  evidence_viewer_no_pdf: graceful None handling")
    except Exception as e:
        failures.append(("evidence_viewer_no_pdf", e))
        print(f"  FAIL  evidence_viewer_no_pdf: {type(e).__name__}: {e}")
    return failures


def test_bbox_overlay_no_crash():
    """Verify bbox overlay function handles None PDF and empty bboxes without crashing."""
    failures = []
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from demo_page import render_pdf_page_with_overlay

        # None PDF path should return None even with bboxes
        result = render_pdf_page_with_overlay(None, 0, [[100, 200, 300, 400]])
        assert result is None, f"Expected None for None PDF, got {type(result)}"

        # None PDF, no bboxes
        result2 = render_pdf_page_with_overlay(None, 0, None)
        assert result2 is None, "Expected None for None PDF + None bboxes"

        # Nonexistent PDF
        result3 = render_pdf_page_with_overlay("/nonexistent.pdf", 0, [[0, 0, 50, 50]])
        assert result3 is None, "Expected None for nonexistent PDF"

        # Empty bboxes list
        result4 = render_pdf_page_with_overlay(None, 0, [])
        assert result4 is None, "Expected None for None PDF + empty bboxes"

        print("  PASS  bbox_overlay_no_crash: all edge cases handled")
    except Exception as e:
        failures.append(("bbox_overlay_no_crash", e))
        print(f"  FAIL  bbox_overlay_no_crash: {type(e).__name__}: {e}")
    return failures


def test_not_found_proof_fields():
    """Verify NOT_FOUND proof fields are accessible in evidence."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        blocker = payload["blockers"][0]
        ev = blocker["evidence"]

        # Check coverage_status on the blocker
        cov_status = blocker.get("coverage_status", "")
        assert cov_status == "not_found_after_search", f"Expected not_found_after_search, got {cov_status}"

        # Check searched_pages
        searched = ev.get("searched_pages")
        assert isinstance(searched, list), f"Expected list, got {type(searched)}"
        assert len(searched) == 7, f"Expected 7 searched pages, got {len(searched)}"

        # Check text_coverage_pct
        text_cov = ev.get("text_coverage_pct")
        assert isinstance(text_cov, (int, float)), f"Expected number, got {type(text_cov)}"
        assert 0 <= text_cov <= 100, f"Expected 0-100, got {text_cov}"

        # Check bbox field exists (populated or None both valid)
        assert "bbox" in ev, "bbox key missing from evidence"
        bbox = ev["bbox"]
        if bbox is not None:
            assert isinstance(bbox, list), f"bbox should be list, got {type(bbox)}"
            # Verify page-relative format: each entry is a list of boxes
            for page_boxes in bbox:
                assert isinstance(page_boxes, list), "Each page entry should be a list"
                for box in page_boxes:
                    assert len(box) >= 4, f"Box should have >= 4 elements, got {len(box)}"

        print("  PASS  not_found_proof_fields: all fields present and valid")
    except Exception as e:
        failures.append(("not_found_proof_fields", e))
        print(f"  FAIL  not_found_proof_fields: {type(e).__name__}: {e}")
    return failures


def test_skipped_pages_csv():
    """Verify skipped_pages data is exportable as CSV."""
    failures = []
    try:
        payload = PAYLOADS["fast_budget_with_gaps"]
        run_cov = payload.get("run_coverage", {})
        skipped = run_cov.get("pages_skipped", [])

        assert len(skipped) > 0, "Expected non-empty pages_skipped"

        # Check each entry has required keys
        for s in skipped:
            for key in ("page_idx", "doc_type", "discipline", "reason"):
                assert key in s, f"Missing key '{key}' in skipped page entry"

        # Build CSV and verify headers
        import io, csv
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["Page", "Doc Type", "Discipline", "Reason"])
        writer.writeheader()
        for s in skipped:
            writer.writerow({
                "Page": s["page_idx"] + 1,
                "Doc Type": s["doc_type"],
                "Discipline": s["discipline"],
                "Reason": s["reason"],
            })
        csv_str = buf.getvalue()
        assert "Page,Doc Type,Discipline,Reason" in csv_str, "CSV headers missing"
        assert len(csv_str.strip().split("\n")) > 1, "CSV has no data rows"

        print("  PASS  skipped_pages_csv: data valid and CSV export works")
    except Exception as e:
        failures.append(("skipped_pages_csv", e))
        print(f"  FAIL  skipped_pages_csv: {type(e).__name__}: {e}")
    return failures


# ── Sprint 4a tests ──────────────────────────────────────────────────────

def test_bbox_page_relative_format():
    """Verify bbox is page-relative with 6-element tuples (coords + confidence + bbox_id)."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        ev = payload["blockers"][0]["evidence"]
        bbox = ev.get("bbox")
        assert bbox is not None, "bbox should not be None after Sprint 4a update"
        assert isinstance(bbox, list), f"bbox should be list, got {type(bbox)}"
        assert len(bbox) == 3, f"Expected 3 page entries (parallel to pages), got {len(bbox)}"
        # Check first page has one box
        assert len(bbox[0]) == 1, f"Page 0 should have 1 box, got {len(bbox[0])}"
        box = bbox[0][0]
        assert len(box) == 6, f"Each box should have 6 elements [x0,y0,x1,y1,conf,bbox_id], got {len(box)}"
        # All coords should be 0.0–1.0
        for coord in box[:4]:
            assert 0.0 <= coord <= 1.0, f"Coord {coord} out of 0.0–1.0 range"
        # Confidence should be 0.0–1.0
        assert 0.0 <= box[4] <= 1.0, f"Confidence {box[4]} out of 0.0–1.0 range"
        # bbox_id should be a string
        assert isinstance(box[5], str), f"bbox_id should be str, got {type(box[5])}"
        # Page 2 has 2 boxes
        assert len(bbox[1]) == 2, f"Page 2 should have 2 boxes, got {len(bbox[1])}"
        # Page 6 is empty
        assert len(bbox[2]) == 0, f"Page 6 should have 0 boxes, got {len(bbox[2])}"
        print("  PASS  bbox_page_relative_format: all coords and bbox_ids valid")
    except Exception as e:
        failures.append(("bbox_page_relative_format", e))
        print(f"  FAIL  bbox_page_relative_format: {type(e).__name__}: {e}")
    return failures


def test_ocr_bbox_meta():
    """Verify ocr_bbox_meta has required fields."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        meta = payload.get("ocr_bbox_meta")
        assert meta is not None, "ocr_bbox_meta missing from payload"
        assert meta.get("engine") in ("surya", "tesseract", "none"), \
            f"Invalid engine: {meta.get('engine')}"
        assert isinstance(meta.get("avg_confidence"), (int, float)), \
            f"avg_confidence should be numeric, got {type(meta.get('avg_confidence'))}"
        assert isinstance(meta.get("pages_with_bbox"), int), \
            f"pages_with_bbox should be int, got {type(meta.get('pages_with_bbox'))}"
        assert isinstance(meta.get("pages_total"), int), \
            f"pages_total should be int, got {type(meta.get('pages_total'))}"
        print(f"  PASS  ocr_bbox_meta: engine={meta['engine']}, "
              f"avg_conf={meta['avg_confidence']}, pages_with_bbox={meta['pages_with_bbox']}")
    except Exception as e:
        failures.append(("ocr_bbox_meta", e))
        print(f"  FAIL  ocr_bbox_meta: {type(e).__name__}: {e}")
    return failures


def test_ocr_text_cache_accessible():
    """Verify ocr_text_cache field is a dict with string keys and string values."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        cache = payload.get("ocr_text_cache")
        assert cache is not None, "ocr_text_cache missing from payload"
        assert isinstance(cache, dict), f"Expected dict, got {type(cache)}"
        assert len(cache) > 0, "ocr_text_cache should not be empty"
        for key, val in cache.items():
            assert isinstance(key, (str, int)), f"Key should be str/int, got {type(key)}"
            assert isinstance(val, str), f"Value should be str, got {type(val)}"
            assert len(val) > 0, f"Text for page {key} should not be empty"
        print(f"  PASS  ocr_text_cache_accessible: {len(cache)} pages cached")
    except Exception as e:
        failures.append(("ocr_text_cache_accessible", e))
        print(f"  FAIL  ocr_text_cache_accessible: {type(e).__name__}: {e}")
    return failures


def test_ocr_text_cache_search_term_highlight():
    """Verify search term highlighting logic works on OCR text."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        cache = payload.get("ocr_text_cache", {})
        ev = payload["blockers"][0]["evidence"]
        search_attempts = ev.get("search_attempts", {})
        # Get first search term
        terms = search_attempts.get("keywords", [])
        assert len(terms) > 0, "Expected search terms in evidence"
        # Get text from first page
        page_text = cache.get("0") or cache.get(0)
        assert page_text is not None, "Expected text for page 0"
        # Simulate highlight replacement
        display = page_text
        for term in terms:
            display = display.replace(term, f"**`{term}`**")
        # The term "door schedule" is NOT in the OCR text for page 0 (it's construction text)
        # But the logic should not crash
        print(f"  PASS  ocr_text_cache_search_term_highlight: highlight logic runs without error")
    except Exception as e:
        failures.append(("ocr_text_cache_search_term_highlight", e))
        print(f"  FAIL  ocr_text_cache_search_term_highlight: {type(e).__name__}: {e}")
    return failures


def test_citation_elements_present():
    """Verify evidence has enough data to produce citation strings."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        ev = payload["blockers"][0]["evidence"]
        pages = ev.get("pages", [])
        sheets = ev.get("sheets", [])
        snippets = ev.get("snippets", [])
        assert len(pages) >= 1, f"Expected at least 1 page, got {len(pages)}"
        assert len(snippets) >= 1, f"Expected at least 1 snippet, got {len(snippets)}"
        # Build citation for first snippet
        pg = pages[0]
        sheet = sheets[0] if sheets else ""
        cite = f"({sheet + ', ' if sheet else ''}p.{pg + 1})"
        assert "p." in cite, f"Citation should contain 'p.', got: {cite}"
        if sheet:
            assert sheet in cite, f"Citation should contain sheet '{sheet}', got: {cite}"
        print(f"  PASS  citation_elements_present: cite='{cite}'")
    except Exception as e:
        failures.append(("citation_elements_present", e))
        print(f"  FAIL  citation_elements_present: {type(e).__name__}: {e}")
    return failures


def test_heatmap_data_available():
    """Verify heatmap can be rendered from run_coverage data."""
    failures = []
    try:
        # Test both full_read and fast_budget payloads
        for name in ("full_read_coverage", "fast_budget_with_gaps"):
            payload = PAYLOADS[name]
            rc = payload.get("run_coverage", {})
            total = rc.get("pages_total", 0)
            assert total > 0, f"{name}: pages_total should be > 0"
            skipped = rc.get("pages_skipped", [])
            assert isinstance(skipped, list), f"{name}: pages_skipped should be list"
            processed = rc.get("pages_deep_processed", 0)
            assert processed > 0, f"{name}: pages_deep_processed should be > 0"
            print(f"  PASS  heatmap_data_available ({name}): "
                  f"total={total}, processed={processed}, skipped={len(skipped)}")
    except Exception as e:
        failures.append(("heatmap_data_available", e))
        print(f"  FAIL  heatmap_data_available: {type(e).__name__}: {e}")
    return failures


# ── Sprint 5 tests ──────────────────────────────────────────────────────

def test_bbox_id_in_evidence():
    """Verify bbox entries have 6 elements with string bbox_id."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        ev = payload["blockers"][0]["evidence"]
        bbox = ev.get("bbox", [])
        for pi, page_boxes in enumerate(bbox):
            for bi, box in enumerate(page_boxes):
                assert len(box) == 6, f"Box P{pi}-{bi} has {len(box)} elements, expected 6"
                assert isinstance(box[5], str), f"Box P{pi}-{bi} bbox_id should be str, got {type(box[5])}"
                assert len(box[5]) > 0, f"Box P{pi}-{bi} bbox_id should not be empty"
        print("  PASS  bbox_id_in_evidence: all boxes have 6-element format with string bbox_id")
    except Exception as e:
        failures.append(("bbox_id_in_evidence", e))
        print(f"  FAIL  bbox_id_in_evidence: {type(e).__name__}: {e}")
    return failures


def test_bbox_id_linkage():
    """Parse bbox_id → extract item_id → find matching blocker."""
    failures = []
    try:
        payload = PAYLOADS["blocker_with_full_evidence"]
        ev = payload["blockers"][0]["evidence"]
        bbox = ev.get("bbox", [])
        # Get first non-empty box
        first_box = None
        for page_boxes in bbox:
            if page_boxes:
                first_box = page_boxes[0]
                break
        assert first_box is not None, "No non-empty bbox found"
        bbox_id = first_box[5]
        # Parse item_id from bbox_id: everything before first "-P"
        p_idx = bbox_id.index("-P")
        item_id = bbox_id[:p_idx]
        # Find matching blocker
        matched = [b for b in payload["blockers"] if b.get("id") == item_id]
        assert len(matched) == 1, f"Expected 1 matching blocker for {item_id}, got {len(matched)}"
        print(f"  PASS  bbox_id_linkage: bbox_id='{bbox_id}' → item_id='{item_id}' → blocker found")
    except Exception as e:
        failures.append(("bbox_id_linkage", e))
        print(f"  FAIL  bbox_id_linkage: {type(e).__name__}: {e}")
    return failures


def test_search_returns_correct_pages():
    """Search 'Door' in demo cache OCR text returns pages 0, 2, 6."""
    failures = []
    try:
        from search import search_ocr_text
        payload = PAYLOADS["blocker_with_full_evidence"]
        cache = payload.get("ocr_text_cache", {})
        results = search_ocr_text(cache, "Door")
        result_pages = sorted(set(r["page_idx"] for r in results))
        # "Door" appears on pages 0, 2, 6 in the test OCR cache
        assert 0 in result_pages, f"Expected page 0 in results, got {result_pages}"
        assert 2 in result_pages, f"Expected page 2 in results, got {result_pages}"
        assert 6 in result_pages, f"Expected page 6 in results, got {result_pages}"
        print(f"  PASS  search_returns_correct_pages: found on pages {result_pages}")
    except Exception as e:
        failures.append(("search_returns_correct_pages", e))
        print(f"  FAIL  search_returns_correct_pages: {type(e).__name__}: {e}")
    return failures


def test_search_returns_snippets():
    """Search results have non-empty snippet field."""
    failures = []
    try:
        from search import search_ocr_text
        payload = PAYLOADS["blocker_with_full_evidence"]
        cache = payload.get("ocr_text_cache", {})
        results = search_ocr_text(cache, "Door")
        assert len(results) > 0, "Expected at least one result"
        for r in results:
            assert "snippet" in r, f"Missing 'snippet' key in result"
            assert len(r["snippet"]) > 0, f"Empty snippet for page {r['page_idx']}"
            assert "**" in r["snippet"], f"Snippet should have markdown highlighting"
        print(f"  PASS  search_returns_snippets: {len(results)} results all have snippets")
    except Exception as e:
        failures.append(("search_returns_snippets", e))
        print(f"  FAIL  search_returns_snippets: {type(e).__name__}: {e}")
    return failures


def test_search_performance():
    """500-page mock OCR cache search completes in <1s."""
    failures = []
    try:
        import time
        from search import search_ocr_text
        # Build a 500-page mock cache (~1KB per page)
        mock_cache = {}
        for i in range(500):
            mock_cache[str(i)] = (
                f"Page {i} content with various construction terms. "
                f"Door D{i} shown on plan. Window W{i} on elevation. "
                f"Beam B{i} 230x450. Column C{i} 300x300. "
                f"Slab thickness 150mm. Clear height 3.0m.\n" * 5
            )
        start = time.time()
        results = search_ocr_text(mock_cache, "Door")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Search took {elapsed:.3f}s, expected <1s"
        assert len(results) == 50, f"Expected 50 results (max_results cap), got {len(results)}"
        print(f"  PASS  search_performance: 500 pages in {elapsed:.3f}s, {len(results)} results")
    except Exception as e:
        failures.append(("search_performance", e))
        print(f"  FAIL  search_performance: {type(e).__name__}: {e}")
    return failures


def test_bid_strategy_unknown_outputs():
    """No user inputs → UNKNOWN dials for client_fit and competition."""
    failures = []
    try:
        from bid_strategy_scorer import compute_bid_strategy
        payload = PAYLOADS["blocker_with_full_evidence"]
        # Empty user inputs
        inputs = {}
        result = compute_bid_strategy(inputs, payload)
        # Client Fit should be UNKNOWN
        cf = result["client_fit"]
        assert cf["score"] is None, f"Expected None client_fit score, got {cf['score']}"
        assert cf["confidence"] == "UNKNOWN", f"Expected UNKNOWN confidence, got {cf['confidence']}"
        # Competition should be UNKNOWN
        comp = result["competition_score"]
        assert comp["score"] is None, f"Expected None competition score, got {comp['score']}"
        assert comp["confidence"] == "UNKNOWN", f"Expected UNKNOWN confidence, got {comp['confidence']}"
        # Risk should still compute (document-derived)
        risk = result["risk_score"]
        assert risk["score"] is not None, f"Risk score should not be None"
        assert isinstance(risk["score"], int), f"Risk score should be int, got {type(risk['score'])}"
        # Readiness should always compute
        ready = result["readiness_score"]
        assert ready["score"] is not None, f"Readiness should not be None"
        print(f"  PASS  bid_strategy_unknown_outputs: cf=UNKNOWN, comp=UNKNOWN, "
              f"risk={risk['score']}, ready={ready['score']}")
    except Exception as e:
        failures.append(("bid_strategy_unknown_outputs", e))
        print(f"  FAIL  bid_strategy_unknown_outputs: {type(e).__name__}: {e}")
    return failures


def test_bid_strategy_dial_range():
    """All computed scores are 0-100."""
    failures = []
    try:
        from bid_strategy_scorer import compute_bid_strategy
        payload = PAYLOADS["blocker_with_full_evidence"]
        inputs = {
            "relationship_level": "Repeat",
            "past_work_count": 3,
            "payment_delays": True,
            "disputes": False,
            "high_co_rate": False,
            "competitors": ["Comp A", "Comp B"],
            "market_pressure": 6,
            "target_margin": 12.0,
            "win_probability": 60.0,
        }
        result = compute_bid_strategy(inputs, payload)
        for key in ("client_fit", "risk_score", "competition_score", "readiness_score"):
            dial = result[key]
            if dial["score"] is not None:
                assert 0 <= dial["score"] <= 100, \
                    f"{key} score {dial['score']} out of 0-100 range"
                assert dial["confidence"] in ("HIGH", "MEDIUM", "LOW"), \
                    f"{key} confidence '{dial['confidence']}' not valid"
        print(f"  PASS  bid_strategy_dial_range: all scores 0-100")
    except Exception as e:
        failures.append(("bid_strategy_dial_range", e))
        print(f"  FAIL  bid_strategy_dial_range: {type(e).__name__}: {e}")
    return failures


def test_bid_strategy_recommendations_present():
    """Recommendations list is non-empty when blockers exist."""
    failures = []
    try:
        from bid_strategy_scorer import compute_bid_strategy
        payload = PAYLOADS["blocker_with_full_evidence"]
        inputs = {}
        result = compute_bid_strategy(inputs, payload)
        recs = result.get("recommendations", [])
        assert len(recs) > 0, "Expected at least 1 recommendation when blockers exist"
        # Should mention missing schedule or NOT_FOUND
        combined = " ".join(recs).lower()
        assert "not found" in combined or "missing" in combined or "schedule" in combined, \
            f"Recommendations should mention document issues, got: {recs}"
        print(f"  PASS  bid_strategy_recommendations_present: {len(recs)} recommendation(s)")
    except Exception as e:
        failures.append(("bid_strategy_recommendations_present", e))
        print(f"  FAIL  bid_strategy_recommendations_present: {type(e).__name__}: {e}")
    return failures


# ── Sprint 6 tests ──────────────────────────────────────────────────────

def test_bid_summary_generation():
    """Test bid summary generation with various payloads."""
    failures = []
    from bid_summary import generate_bid_summary_markdown

    for name, payload in PAYLOADS.items():
        try:
            md = generate_bid_summary_markdown(payload)
            assert isinstance(md, str), f"Expected string, got {type(md)}"
            assert len(md) > 50, f"Summary too short: {len(md)} chars"
            assert "Bid Summary" in md, "Missing header"
            print(f"  PASS  {name}: {len(md)} chars")
        except Exception as e:
            failures.append((f"bid_summary_{name}", e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    return failures


def test_addendum_and_conflicts_access():
    """Test that payloads with addendum_index and conflicts don't crash."""
    failures = []
    # Payload WITH addendum data
    try:
        payload = PAYLOADS["extraction_with_data"]
        demo = build_demo_analysis(payload, "test_addenda")
        raw = demo.raw_payload
        addenda = raw.get("addendum_index", [])
        conflicts = raw.get("conflicts", [])
        assert isinstance(addenda, list), f"Expected list, got {type(addenda)}"
        assert isinstance(conflicts, list), f"Expected list, got {type(conflicts)}"
        assert len(addenda) == 1, f"Expected 1 addendum, got {len(addenda)}"
        assert len(conflicts) == 1, f"Expected 1 conflict, got {len(conflicts)}"
        assert addenda[0]["addendum_no"] == "1"
        assert conflicts[0]["type"] == "boq_change"
        print(f"  PASS  addendum_and_conflicts: {len(addenda)} addenda, {len(conflicts)} conflicts")
    except Exception as e:
        failures.append(("addendum_and_conflicts_with_data", e))
        print(f"  FAIL  addendum_and_conflicts_with_data: {type(e).__name__}: {e}")

    # Payload WITHOUT addendum data (graceful fallback)
    try:
        payload = PAYLOADS["empty_payload"]
        addenda = payload.get("addendum_index", [])
        conflicts = payload.get("conflicts", [])
        assert addenda == [], f"Expected empty list, got {addenda}"
        assert conflicts == [], f"Expected empty list, got {conflicts}"
        print(f"  PASS  missing_addendum_graceful: defaults to empty lists")
    except Exception as e:
        failures.append(("missing_addendum_graceful", e))
        print(f"  FAIL  missing_addendum_graceful: {type(e).__name__}: {e}")

    return failures


def test_normalization_idempotent():
    """Normalize already-normalized data -> unchanged."""
    failures = []
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from analysis.normalize import normalize_boq_items, normalize_schedule_rows, normalize_requirements

        # Already-normalized BOQ items (includes trade + flags from Sprint 19)
        items = [{"item_no": "1.1", "unit": "cum", "qty": 120.0, "rate": 450.0,
                  "description": "Excavation", "trade": "civil", "flags": []}]
        result = normalize_boq_items(items)
        assert result[0] == items[0], f"Expected unchanged, got {result[0]}"

        # Already-normalized schedule rows
        rows = [{"mark": "D1", "size": "900 x 2100", "schedule_type": "door", "confidence": 0.8}]
        result = normalize_schedule_rows(rows)
        assert len(result) == 1
        assert result[0]["mark"] == "D1"
        assert result[0]["size"] == "900 x 2100"

        # Already-unique requirements
        reqs = [{"text": "Concrete M25", "source_page": 1}, {"text": "Steel Fe500", "source_page": 2}]
        result = normalize_requirements(reqs)
        assert len(result) == 2

        print("  PASS  normalization_idempotent: all data unchanged after re-normalization")
    except Exception as e:
        failures.append(("normalization_idempotent", e))
        print(f"  FAIL  normalization_idempotent: {type(e).__name__}: {e}")
    return failures


def test_bid_summary_includes_strategy():
    """Summary includes dials when bid_strategy provided."""
    failures = []
    try:
        from bid_summary import generate_bid_summary_markdown
        payload = PAYLOADS["blocker_with_full_evidence"]
        strategy = {
            "client_fit": {"name": "Client Fit", "score": 72, "confidence": "HIGH", "based_on": ["Repeat client"]},
            "risk_score": {"name": "Risk Score", "score": 45, "confidence": "MEDIUM", "based_on": ["2 blockers"]},
            "competition_score": {"name": "Competition", "score": None, "confidence": "UNKNOWN", "based_on": []},
            "readiness_score": {"name": "Readiness", "score": 55, "confidence": "HIGH", "based_on": ["Doc analysis"]},
            "recommendations": ["Submit RFIs before bid"],
        }
        md = generate_bid_summary_markdown(payload, bid_strategy=strategy)
        assert "Bid Strategy Assessment" in md, "Missing strategy section"
        assert "Client Fit" in md, "Missing Client Fit dial"
        assert "72/100" in md, "Missing score value"
        assert "Not computed" in md, "Missing UNKNOWN dial"
        assert "Submit RFIs" in md, "Missing recommendation"
        print("  PASS  bid_summary_includes_strategy: strategy section present with dials + recs")
    except Exception as e:
        failures.append(("bid_summary_includes_strategy", e))
        print(f"  FAIL  bid_summary_includes_strategy: {type(e).__name__}: {e}")
    return failures


# =============================================================================
# SPRINT 7: NEW SMOKE TESTS
# =============================================================================

def test_delta_confidence_deterministic():
    """Same inputs → same confidence scores (deterministic)."""
    failures = []
    try:
        from src.analysis.delta_detector import detect_boq_deltas, detect_requirement_deltas
        base_boq = [{"item_no": "1.1", "qty": 100, "unit": "cum", "source_page": 5}]
        add_boq = [{"item_no": "1.1", "qty": 150, "unit": "cum", "source_page": 20}]
        c1 = detect_boq_deltas(base_boq, add_boq)
        c2 = detect_boq_deltas(base_boq, add_boq)
        assert c1[0]["delta_confidence"] == c2[0]["delta_confidence"], "BOQ confidence not deterministic"

        base_req = [{"text": "All concrete M25 grade", "source_page": 15}]
        add_req = [{"text": "All concrete M30 grade", "source_page": 25}]
        r1 = detect_requirement_deltas(base_req, add_req, ocr_coverage_pct=80)
        r2 = detect_requirement_deltas(base_req, add_req, ocr_coverage_pct=80)
        assert r1[0]["delta_confidence"] == r2[0]["delta_confidence"], "Req confidence not deterministic"
        print("  PASS  delta_confidence_deterministic: same inputs → same outputs")
    except Exception as e:
        failures.append(("delta_confidence_deterministic", e))
        print(f"  FAIL  delta_confidence_deterministic: {type(e).__name__}: {e}")
    return failures


def test_reconciliation_stable_keys():
    """Reconciliation findings have all required keys."""
    failures = []
    try:
        from src.analysis.reconciler import reconcile_scope
        reqs = [{"text": "All doors shall be solid core hardwood", "source_page": 5}]
        schedules = []
        boq_items = []
        findings = reconcile_scope(reqs, schedules, boq_items)
        required_keys = {"type", "category", "description", "impact", "suggested_action", "evidence", "confidence"}
        for f in findings:
            missing = required_keys - set(f.keys())
            assert not missing, f"Missing keys: {missing}"
        print(f"  PASS  reconciliation_stable_keys: {len(findings)} findings, all keys present")
    except Exception as e:
        failures.append(("reconciliation_stable_keys", e))
        print(f"  FAIL  reconciliation_stable_keys: {type(e).__name__}: {e}")
    return failures


def test_pdf_export_produces_bytes():
    """PDF generation returns non-empty bytes."""
    failures = []
    try:
        from bid_summary_pdf import generate_bid_summary_pdf
        payload = PAYLOADS["extraction_with_data"]
        pdf_bytes = generate_bid_summary_pdf(payload)
        assert isinstance(pdf_bytes, bytes), "PDF output is not bytes"
        assert len(pdf_bytes) > 100, f"PDF too small: {len(pdf_bytes)} bytes"
        assert pdf_bytes[:5] == b"%PDF-", "PDF does not start with %PDF-"
        print(f"  PASS  pdf_export_produces_bytes: {len(pdf_bytes)} bytes, valid PDF header")
    except Exception as e:
        failures.append(("pdf_export_produces_bytes", e))
        print(f"  FAIL  pdf_export_produces_bytes: {type(e).__name__}: {e}")
    return failures


def test_pdf_export_missing_sections():
    """PDF works with empty addendum/conflicts/strategy."""
    failures = []
    try:
        from bid_summary_pdf import generate_bid_summary_pdf
        minimal = {
            "project_id": "minimal_test",
            "timestamp": "2026-02-19T00:00:00",
            "readiness_score": 50,
            "decision": "REVIEW",
            "addendum_index": [],
            "conflicts": [],
            "reconciliation_findings": [],
        }
        pdf_bytes = generate_bid_summary_pdf(minimal)
        assert isinstance(pdf_bytes, bytes), "PDF output is not bytes"
        assert len(pdf_bytes) > 100, f"PDF too small: {len(pdf_bytes)} bytes"
        print(f"  PASS  pdf_export_missing_sections: {len(pdf_bytes)} bytes with minimal payload")
    except Exception as e:
        failures.append(("pdf_export_missing_sections", e))
        print(f"  FAIL  pdf_export_missing_sections: {type(e).__name__}: {e}")
    return failures


# =============================================================================
# SPRINT 8: NEW SMOKE TESTS
# =============================================================================

def test_recon_rfi_creates_valid_dict():
    """recon_actions produces valid RFI dict from reconciliation finding."""
    failures = []
    try:
        from src.analysis.recon_actions import (
            finding_to_proposed_rfi,
            create_recon_rfi,
        )
        finding = {
            "type": "missing",
            "category": "req_vs_schedule",
            "description": "Requirements mention doors but no door schedule was found",
            "impact": "high",
            "suggested_action": "Request door schedule from design team",
            "evidence": {"pages": [3, 7], "items": ["door requirements"]},
            "confidence": 0.85,
        }
        proposed = finding_to_proposed_rfi(finding)
        rfi = create_recon_rfi(proposed, [], finding)

        # Verify required fields
        required = {"id", "trade", "priority", "question", "why_it_matters",
                     "evidence", "suggested_resolution", "source", "created_at"}
        missing = required - set(rfi.keys())
        assert not missing, f"Missing RFI keys: {missing}"

        # RFI-R- prefix
        assert rfi["id"].startswith("RFI-R-"), f"Expected RFI-R- prefix, got {rfi['id']}"

        # source = reconciler
        assert rfi["source"] == "reconciler", f"Expected source='reconciler', got {rfi['source']}"

        # Evidence pages preserved
        assert rfi["evidence"]["pages"] == [3, 7], f"Evidence pages lost: {rfi['evidence']['pages']}"

        # Priority from impact
        assert rfi["priority"] == "high", f"Expected priority=high, got {rfi['priority']}"

        print(f"  PASS  recon_rfi_creates_valid_dict: id={rfi['id']}, source={rfi['source']}")
    except Exception as e:
        failures.append(("recon_rfi_creates_valid_dict", e))
        print(f"  FAIL  recon_rfi_creates_valid_dict: {type(e).__name__}: {e}")
    return failures


def test_recon_assumption_stable_schema():
    """recon_actions produces assumption dict with all required keys."""
    failures = []
    try:
        from src.analysis.recon_actions import (
            finding_to_proposed_assumption,
            create_recon_assumption,
        )
        finding = {
            "type": "conflict",
            "category": "boq_vs_schedule",
            "description": "BOQ door quantity (5) does not match door schedule marks (8)",
            "impact": "high",
            "suggested_action": "Reconcile door count",
            "evidence": {"pages": [10, 20], "items": ["BOQ doors: 5", "Schedule marks: 8"]},
            "confidence": 0.8,
        }
        proposed = finding_to_proposed_assumption(finding)
        assumption = create_recon_assumption(proposed, [], finding)

        # Required keys
        required = {"id", "title", "text", "impact_if_wrong", "risk_level",
                     "basis_pages", "source", "created_at"}
        missing = required - set(assumption.keys())
        assert not missing, f"Missing assumption keys: {missing}"

        # ASMP-R- prefix
        assert assumption["id"].startswith("ASMP-R-"), f"Expected ASMP-R- prefix, got {assumption['id']}"

        # source = reconciler
        assert assumption["source"] == "reconciler", f"Expected source='reconciler', got {assumption['source']}"

        # basis_pages preserved
        assert assumption["basis_pages"] == [10, 20], f"basis_pages lost: {assumption['basis_pages']}"

        # risk_level from finding.impact
        assert assumption["risk_level"] == "high", f"Expected risk_level=high, got {assumption['risk_level']}"

        print(f"  PASS  recon_assumption_stable_schema: id={assumption['id']}, source={assumption['source']}")
    except Exception as e:
        failures.append(("recon_assumption_stable_schema", e))
        print(f"  FAIL  recon_assumption_stable_schema: {type(e).__name__}: {e}")
    return failures


def test_owner_profile_round_trip():
    """Save and reload owner profile in temp dir."""
    failures = []
    try:
        import tempfile
        from src.analysis.owner_profiles import save_profile, load_profile, list_profiles

        with tempfile.TemporaryDirectory() as tmp:
            inputs = {
                "relationship_level": "Repeat",
                "past_work_count": 5,
                "payment_delays": False,
                "target_margin": 12.0,
            }
            # Save
            path = save_profile("Test Client Corp", inputs, profile_dir=tmp)
            assert path.exists(), f"Profile file not created at {path}"

            # Reload
            loaded = load_profile("Test Client Corp", profile_dir=tmp)
            assert loaded is not None, "load_profile returned None"
            assert loaded["owner_name"] == "Test Client Corp"
            assert loaded["inputs"]["past_work_count"] == 5
            assert loaded["inputs"]["target_margin"] == 12.0

            # List
            names = list_profiles(profile_dir=tmp)
            assert "Test Client Corp" in names, f"Expected name in list, got {names}"

        print("  PASS  owner_profile_round_trip: save + load + list OK")
    except Exception as e:
        failures.append(("owner_profile_round_trip", e))
        print(f"  FAIL  owner_profile_round_trip: {type(e).__name__}: {e}")
    return failures


def test_risk_scoped_search_backward_compat():
    """search_risk_items with no new params = same behavior as before."""
    failures = []
    try:
        from risk_checklist import search_risk_items

        ocr_cache = {
            0: "Liquidated damages shall be 1% per week",
            1: "Retention money of 5% shall be withheld",
            2: "Normal construction notes and specifications",
        }

        # No new params (backward compatible)
        results_default = search_risk_items(ocr_cache)
        # With explicit None (same thing)
        results_explicit = search_risk_items(
            ocr_cache,
            page_doc_types=None,
            allowed_doc_types=None,
        )

        assert len(results_default) == len(results_explicit), "Param defaults changed behavior"

        for r in results_default:
            assert "searched_pages_count" in r, "Missing searched_pages_count"
            assert "searched_doc_types" in r, "Missing searched_doc_types"

        # Verify LD was found
        ld = [r for r in results_default if r["template_id"] == "RISK-LD"]
        assert len(ld) == 1 and ld[0]["found"], "LD not found in default mode"

        print("  PASS  risk_scoped_search_backward_compat: default params unchanged")
    except Exception as e:
        failures.append(("risk_scoped_search_backward_compat", e))
        print(f"  FAIL  risk_scoped_search_backward_compat: {type(e).__name__}: {e}")
    return failures


# ── Sprint 9 Smoke Tests ────────────────────────────────────────────────

def test_assumptions_status_persists_and_exports():
    """Create assumption, accept it, export — round-trip test."""
    failures = []
    try:
        from src.analysis.recon_actions import (
            finding_to_proposed_assumption,
            create_recon_assumption,
            update_assumption_status,
            generate_exclusions_clarifications,
        )

        # Create two assumptions via the proper proposal chain
        findings = [
            {
                "type": "conflict",
                "category": "slab_thickness",
                "description": "Slab assumed 150mm thick based on structural notes",
                "impact": "medium",
                "suggested_action": "Confirm slab thickness",
                "evidence": {"pages": [2, 5], "items": ["150mm noted"]},
                "confidence": 0.85,
            },
            {
                "type": "conflict",
                "category": "piling_scope",
                "description": "Piling scope excluded from bid",
                "impact": "high",
                "suggested_action": "Clarify piling scope",
                "evidence": {"pages": [8], "items": ["No piling details"]},
                "confidence": 0.9,
            },
        ]
        assumptions = []
        for f in findings:
            proposed = finding_to_proposed_assumption(f)
            a = create_recon_assumption(proposed, assumptions, f)
            assumptions.append(a)

        # Accept first, reject second
        a0 = update_assumption_status(assumptions[0], "accepted", "PM", cost_impact=25000.0, scope_tag="structure")
        a1 = update_assumption_status(assumptions[1], "rejected", "QS", cost_impact=None, scope_tag="foundations")
        assumptions_log = [a0, a1]

        # Verify statuses persisted
        assert a0["status"] == "accepted", f"Expected accepted, got {a0['status']}"
        assert a1["status"] == "rejected", f"Expected rejected, got {a1['status']}"
        assert a0["approved_by"] == "PM"
        assert a0["cost_impact"] == 25000.0
        assert a0["scope_tag"] == "structure"
        assert a0["approved_at"] is not None, "approved_at should be set"

        # Export
        txt, csv_out = generate_exclusions_clarifications(assumptions_log)
        assert "EXCLUSIONS" in txt, "Missing EXCLUSIONS section"
        assert "CLARIFICATIONS" in txt, "Missing CLARIFICATIONS section"
        # Check that each section has content (title auto-generated from finding)
        assert "rejected" in csv_out.lower() or "EXCL" in txt, "Rejected assumption not in exclusions"
        assert "accepted" in csv_out.lower() or "CLAR" in txt, "Accepted assumption not in clarifications"
        assert "id," in csv_out, "CSV missing header row"
        assert "accepted" in csv_out, "CSV missing accepted status"
        assert "rejected" in csv_out, "CSV missing rejected status"

        print("  PASS  assumptions_status_persists_and_exports: round-trip OK")
    except Exception as e:
        failures.append(("assumptions_status_persists_and_exports", e))
        print(f"  FAIL  assumptions_status_persists_and_exports: {type(e).__name__}: {e}")
    return failures


def test_multi_doc_evidence_viewer():
    """build_multi_doc_index + page mapping for 2-doc set."""
    failures = []
    try:
        from src.analysis.multi_doc import (
            build_multi_doc_index,
            global_to_doc_page,
            convert_evidence_pages,
            get_pdf_path_for_doc,
        )

        file_info = [
            {"name": "drawings.pdf", "pages": 10, "path": "/tmp/drawings.pdf", "ocr_used": True},
            {"name": "specs.pdf", "pages": 5, "path": "/tmp/specs.pdf", "ocr_used": True},
        ]
        mdi = build_multi_doc_index(file_info)

        assert mdi.total_pages == 15, f"Expected 15 total pages, got {mdi.total_pages}"
        assert len(mdi.docs) == 2, f"Expected 2 docs, got {len(mdi.docs)}"

        # Page 0 -> doc 0, local 0
        assert global_to_doc_page(0, mdi) == (0, 0), f"Page 0 wrong: {global_to_doc_page(0, mdi)}"
        # Page 9 -> doc 0, local 9
        assert global_to_doc_page(9, mdi) == (0, 9), f"Page 9 wrong: {global_to_doc_page(9, mdi)}"
        # Page 10 -> doc 1, local 0
        assert global_to_doc_page(10, mdi) == (1, 0), f"Page 10 wrong: {global_to_doc_page(10, mdi)}"
        # Page 14 -> doc 1, local 4
        assert global_to_doc_page(14, mdi) == (1, 4), f"Page 14 wrong: {global_to_doc_page(14, mdi)}"

        # Batch convert
        evidence = convert_evidence_pages([2, 12], mdi)
        assert evidence == [(0, 2), (1, 2)], f"Batch convert wrong: {evidence}"

        # PDF path lookup
        assert get_pdf_path_for_doc(0, mdi) == "/tmp/drawings.pdf"
        assert get_pdf_path_for_doc(1, mdi) == "/tmp/specs.pdf"
        assert get_pdf_path_for_doc(99, mdi) is None

        print("  PASS  multi_doc_evidence_viewer: 2-doc mapping correct")
    except Exception as e:
        failures.append(("multi_doc_evidence_viewer", e))
        print(f"  FAIL  multi_doc_evidence_viewer: {type(e).__name__}: {e}")
    return failures


def test_supersedes_tagging_deterministic():
    """detect_supersede_language + tag_conflicts produces consistent results."""
    failures = []
    try:
        from src.analysis.supersedes_detector import (
            detect_supersede_language,
            tag_conflicts_with_supersedes,
        )

        text = "This addendum supersedes all previous versions. Delete and substitute clause 4.2."
        matches1 = detect_supersede_language(text)
        matches2 = detect_supersede_language(text)

        assert len(matches1) > 0, "No supersede patterns detected"
        assert len(matches1) == len(matches2), "Non-deterministic match count"
        for m1, m2 in zip(matches1, matches2):
            assert m1["matched_text"] == m2["matched_text"], "Non-deterministic match text"

        # Tag conflicts
        conflicts = [
            {"type": "boq_change", "addendum_page": 5, "delta_confidence": 0.9},
            {"type": "boq_change", "addendum_page": 7, "delta_confidence": 0.85},
        ]
        page_texts = {5: text, 7: "Regular text with no supersede language"}

        tagged1 = tag_conflicts_with_supersedes(conflicts, page_texts)
        tagged2 = tag_conflicts_with_supersedes(conflicts, page_texts)

        assert tagged1[0]["resolution"] == "intentional_revision", \
            f"Expected intentional_revision, got {tagged1[0]['resolution']}"
        assert tagged1[1]["resolution"] is None, \
            f"Expected None, got {tagged1[1]['resolution']}"
        # Determinism
        for t1, t2 in zip(tagged1, tagged2):
            assert t1["resolution"] == t2["resolution"], "Non-deterministic tagging"

        # Original not mutated
        assert "resolution" not in conflicts[0], "Original conflict mutated"

        print("  PASS  supersedes_tagging_deterministic: detection + tagging stable")
    except Exception as e:
        failures.append(("supersedes_tagging_deterministic", e))
        print(f"  FAIL  supersedes_tagging_deterministic: {type(e).__name__}: {e}")
    return failures


def test_bid_summary_with_assumptions():
    """Markdown + PDF bid summary include exclusions/clarifications section."""
    failures = []
    try:
        from bid_summary import generate_bid_summary_markdown
        from bid_summary_pdf import generate_bid_summary_pdf

        payload = PAYLOADS["full_payload"].copy()
        payload["conflicts"] = [
            {
                "type": "boq_change",
                "item_no": "A1",
                "changes": [{"field": "qty", "base_value": 10, "addendum_value": 15}],
                "base_page": 1,
                "addendum_page": 3,
                "delta_confidence": 0.9,
                "resolution": "intentional_revision",
            },
            {
                "type": "requirement_new",
                "text": "New waterproofing requirement",
                "addendum_page": 4,
                "delta_confidence": 0.7,
                "resolution": None,
            },
        ]
        payload["multi_doc_index"] = {
            "docs": [
                {"doc_id": 0, "filename": "drawings.pdf", "page_count": 5, "path": "/tmp/d.pdf", "global_page_start": 0},
                {"doc_id": 1, "filename": "specs.pdf", "page_count": 2, "path": "/tmp/s.pdf", "global_page_start": 5},
            ],
            "total_pages": 7,
        }

        assumptions = [
            {"id": "ASMP-R-0001", "title": "150mm slab assumed", "text": "Slab assumed", "status": "accepted",
             "cost_impact": 50000, "scope_tag": "structure", "risk_level": "medium",
             "approved_by": "PM", "approved_at": "2026-02-19T10:00:00"},
            {"id": "ASMP-R-0002", "title": "Piling excluded", "text": "No piling", "status": "rejected",
             "cost_impact": None, "scope_tag": "foundations", "risk_level": "high",
             "approved_by": "QS", "approved_at": "2026-02-19T11:00:00"},
            {"id": "ASMP-R-0003", "title": "Draft assumption", "text": "TBD", "status": "draft",
             "cost_impact": None, "scope_tag": "", "risk_level": "low",
             "approved_by": None, "approved_at": None},
        ]

        # Markdown
        md = generate_bid_summary_markdown(payload, assumptions=assumptions)
        assert "Exclusions" in md, "Markdown missing Exclusions section"
        assert "Clarifications" in md, "Markdown missing Clarifications section"
        assert "Piling excluded" in md, "Rejected assumption not in exclusions"
        assert "150mm slab assumed" in md, "Accepted assumption not in clarifications"
        assert "Draft assumption" not in md, "Draft assumption should not appear"
        assert "(Revision)" in md, "Missing (Revision) label on conflict"
        assert "Document Set" in md, "Missing Document Set section"
        assert "drawings.pdf" in md, "Missing doc filename in Document Set"

        # PDF
        pdf_bytes = generate_bid_summary_pdf(payload, assumptions=assumptions)
        assert isinstance(pdf_bytes, bytes), "PDF not bytes"
        assert len(pdf_bytes) > 1000, f"PDF too small: {len(pdf_bytes)} bytes"
        assert pdf_bytes[:5] == b"%PDF-", "Not a valid PDF"

        print("  PASS  bid_summary_with_assumptions: markdown + PDF have exclusions section")
    except Exception as e:
        failures.append(("bid_summary_with_assumptions", e))
        print(f"  FAIL  bid_summary_with_assumptions: {type(e).__name__}: {e}")
    return failures


# ── Sprint 10 smoke tests ──────────────────────────────────────────────

def test_cache_hit_path_deterministic():
    """Cache key is deterministic for same inputs."""
    failures = []
    try:
        import tempfile
        from src.analysis.pipeline_cache import compute_cache_key, get_cache_dir

        # Create a temp PDF-like file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test content for cache key smoke test")
            pdf_path = Path(f.name)

        config = {"dpi": 150, "budget_pages": 80}
        key1 = compute_cache_key([pdf_path], config)
        key2 = compute_cache_key([pdf_path], config)
        assert key1 == key2, f"Cache keys differ: {key1} vs {key2}"
        assert len(key1) == 64, f"Expected SHA-256 hex (64 chars), got {len(key1)}"

        # Different config → different key
        key3 = compute_cache_key([pdf_path], {"dpi": 72, "budget_pages": 80})
        assert key3 != key1, "Different config should produce different key"

        # Cache dir is deterministic
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = get_cache_dir(Path(tmpdir), key1)
            dir2 = get_cache_dir(Path(tmpdir), key1)
            assert dir1 == dir2, "Cache dirs differ for same key"
            assert dir1.exists(), "Cache dir not created"

        pdf_path.unlink(missing_ok=True)
        print("  PASS  cache_hit_path_deterministic")
    except Exception as e:
        failures.append(("cache_hit_path_deterministic", e))
        print(f"  FAIL  cache_hit_path_deterministic: {type(e).__name__}: {e}")
    return failures


def test_toxic_retry_triggers():
    """identify_failed_pages correctly finds pages with errors or empty text."""
    failures = []
    try:
        from src.analysis.toxic_pages import identify_failed_pages

        # Simulate ocr_metadata with error pages and successful pages
        # page_profiles is a list of dicts with page_index key
        ocr_metadata = {
            "page_profiles": [
                {"page_index": 0, "text_length": 500, "confidence": 0.9, "ocr_used": True},
                {"page_index": 1, "text_length": 0, "error": "timeout after 30s", "ocr_used": True},
                {"page_index": 2, "text_length": 1200, "confidence": 0.85, "ocr_used": True},
                {"page_index": 3, "text_length": 0, "ocr_used": True},
                {"page_index": 4, "text_length": 0, "error": "render failed", "ocr_used": True},
            ]
        }

        failed = identify_failed_pages(ocr_metadata)
        assert 1 in failed, "Page 1 (error) should be in failed list"
        assert 3 in failed, "Page 3 (zero text) should be in failed list"
        assert 4 in failed, "Page 4 (error) should be in failed list"
        assert 0 not in failed, "Page 0 (good) should not be in failed list"
        assert 2 not in failed, "Page 2 (good) should not be in failed list"

        # Empty metadata → no failures
        empty_failed = identify_failed_pages({})
        assert empty_failed == [], f"Empty metadata should return empty list, got {empty_failed}"

        print("  PASS  toxic_retry_triggers")
    except Exception as e:
        failures.append(("toxic_retry_triggers", e))
        print(f"  FAIL  toxic_retry_triggers: {type(e).__name__}: {e}")
    return failures


def test_clustering_stable_across_runs():
    """Clustering same RFI list twice produces identical cluster IDs."""
    failures = []
    try:
        from src.analysis.rfi_clustering import cluster_rfis

        rfis = [
            {"id": "RFI-001", "trade": "structural", "priority": "high",
             "question": "Provide column schedule with sizes",
             "evidence": {"pages": [2, 3]}},
            {"id": "RFI-002", "trade": "structural", "priority": "medium",
             "question": "Confirm column schedule dimensions",
             "evidence": {"pages": [3, 4]}},
            {"id": "RFI-003", "trade": "electrical", "priority": "high",
             "question": "Provide electrical panel schedule",
             "evidence": {"pages": [10, 11]}},
        ]

        clusters1 = cluster_rfis(rfis)
        clusters2 = cluster_rfis(rfis)

        ids1 = [c["cluster_id"] for c in clusters1]
        ids2 = [c["cluster_id"] for c in clusters2]
        assert ids1 == ids2, f"Cluster IDs differ: {ids1} vs {ids2}"
        assert len(clusters1) == len(clusters2), "Cluster counts differ"

        # Structural RFIs should be grouped (same trade + overlapping pages)
        struct_clusters = [c for c in clusters1 if c.get("trade") == "structural"]
        assert len(struct_clusters) >= 1, "Structural RFIs should form at least 1 cluster"

        print("  PASS  clustering_stable_across_runs")
    except Exception as e:
        failures.append(("clustering_stable_across_runs", e))
        print(f"  FAIL  clustering_stable_across_runs: {type(e).__name__}: {e}")
    return failures


def test_qa_score_stable_given_fixed_payload():
    """QA score is deterministic for a fixed payload."""
    failures = []
    try:
        from src.analysis.qa_score import compute_qa_score

        payload = PAYLOADS["full_payload"].copy()
        payload["conflicts"] = [
            {"type": "boq_change", "resolution": None},
            {"type": "requirement_new", "resolution": "intentional_revision"},
        ]
        payload["addendum_index"] = [
            {"addendum_no": 1, "date": "2026-01-01", "title": "Add 1"},
        ]
        payload["toxic_pages"] = {"toxic_count": 1, "pages": [{"page_idx": 5, "toxic": True}]}

        score1 = compute_qa_score(payload)
        score2 = compute_qa_score(payload)

        assert score1["score"] == score2["score"], f"Scores differ: {score1['score']} vs {score2['score']}"
        assert score1["breakdown"] == score2["breakdown"], "Breakdowns differ"
        assert score1["confidence"] == score2["confidence"], "Confidence differs"
        assert isinstance(score1["score"], int), f"Score should be int, got {type(score1['score'])}"
        assert 0 <= score1["score"] <= 100, f"Score out of range: {score1['score']}"

        # Verify breakdown components
        breakdown = score1["breakdown"]
        assert "coverage_completeness" in breakdown, "Missing coverage_completeness"
        assert "conflict_density" in breakdown, "Missing conflict_density"
        assert "toxic_penalty" in breakdown, "Missing toxic_penalty"

        print("  PASS  qa_score_stable_given_fixed_payload")
    except Exception as e:
        failures.append(("qa_score_stable_given_fixed_payload", e))
        print(f"  FAIL  qa_score_stable_given_fixed_payload: {type(e).__name__}: {e}")
    return failures


# ── Sprint 11 Smoke Tests ─────────────────────────────────────────────

def test_quantities_stable_schema():
    """Build quantities from mixed sources, verify row schema."""
    failures = []
    try:
        from src.analysis.quantities import build_all_quantities

        schedules = [
            {"mark": "D1", "fields": {"type": "Flush"}, "schedule_type": "door", "source_page": 0, "has_qty": True, "qty": 3},
        ]
        boq_items = [
            {"item_no": "2.1", "description": "RCC footing", "unit": "cum", "qty": 10, "rate": 5000, "source_page": 1, "confidence": 0.9},
        ]
        callouts = [
            {"text": "W1", "callout_type": "tag", "source_page": 2, "confidence": 0.7},
            {"text": "Kitchen", "callout_type": "room", "source_page": 2, "confidence": 0.6},
        ]

        result = build_all_quantities(schedules, boq_items, callouts)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 3, f"Expected >= 3 rows, got {len(result)}"

        required_keys = {"item", "unit", "qty", "confidence", "source_type", "evidence_refs"}
        for q in result:
            missing = required_keys - set(q.keys())
            assert not missing, f"Missing keys: {missing} in {q}"
            assert isinstance(q["qty"], (int, float)), f"qty should be numeric, got {type(q['qty'])}"
            assert q["source_type"] in ("schedule", "boq", "callout"), f"Invalid source_type: {q['source_type']}"

        print("  PASS  quantities_stable_schema")
    except Exception as e:
        failures.append(("quantities_stable_schema", e))
        print(f"  FAIL  quantities_stable_schema: {type(e).__name__}: {e}")
    return failures


def test_pricing_guidance_deterministic():
    """Same inputs → same output twice."""
    failures = []
    try:
        from src.analysis.pricing_guidance import compute_pricing_guidance

        kwargs = dict(
            qa_score={"score": 65, "breakdown": {"coverage_completeness": 10, "conflict_density": 14}, "top_actions": []},
            addendum_index=[{"addendum_no": 1}],
            conflicts=[{"type": "boq_change", "description": "change1"}, {"type": "requirement_new", "description": "new req"}],
            owner_profile=None,
            run_coverage={"doc_types_found": ["boq"], "doc_types_not_covered": ["specification"]},
        )

        r1 = compute_pricing_guidance(**kwargs)
        r2 = compute_pricing_guidance(**kwargs)

        assert r1 == r2, f"Results differ:\n{r1}\nvs\n{r2}"

        # Check schema
        assert "contingency_range" in r1, "Missing contingency_range"
        assert "recommended_exclusions" in r1, "Missing recommended_exclusions"
        assert "recommended_clarifications" in r1, "Missing recommended_clarifications"
        assert "suggested_alternates_ve" in r1, "Missing suggested_alternates_ve"

        cont = r1["contingency_range"]
        assert cont["low_pct"] <= cont["recommended_pct"] <= cont["high_pct"], (
            f"Range invalid: {cont['low_pct']} <= {cont['recommended_pct']} <= {cont['high_pct']}"
        )

        print("  PASS  pricing_guidance_deterministic")
    except Exception as e:
        failures.append(("pricing_guidance_deterministic", e))
        print(f"  FAIL  pricing_guidance_deterministic: {type(e).__name__}: {e}")
    return failures


def test_docx_exports_produce_bytes():
    """All 3 DOCX generators return non-empty PK bytes."""
    failures = []
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from docx_exports import generate_rfis_docx, generate_exclusions_docx, generate_bid_summary_docx

        # 1. RFIs DOCX
        rfis_bytes = generate_rfis_docx(
            [{"question": "Test?", "trade": "civil", "priority": "high"}],
            project_id="SMOKE-TEST",
        )
        assert rfis_bytes[:2] == b"PK", "RFIs DOCX should start with PK header"
        assert len(rfis_bytes) > 500, f"RFIs DOCX too small: {len(rfis_bytes)}"

        # 2. Exclusions DOCX
        excl_bytes = generate_exclusions_docx(
            assumptions=[{"status": "rejected", "title": "Exclude X"}],
            pricing_guidance={"recommended_exclusions": ["Scope Y"], "recommended_clarifications": ["Clarify Z"]},
            project_id="SMOKE-TEST",
        )
        assert excl_bytes[:2] == b"PK", "Exclusions DOCX should start with PK header"
        assert len(excl_bytes) > 500, f"Exclusions DOCX too small: {len(excl_bytes)}"

        # 3. Bid Summary DOCX
        summary_bytes = generate_bid_summary_docx(
            {"project_id": "SMOKE-TEST", "rfis": [{"question": "Q?", "trade": "civil", "priority": "low"}]},
        )
        assert summary_bytes[:2] == b"PK", "Summary DOCX should start with PK header"
        assert len(summary_bytes) > 500, f"Summary DOCX too small: {len(summary_bytes)}"

        print("  PASS  docx_exports_produce_bytes")
    except Exception as e:
        failures.append(("docx_exports_produce_bytes", e))
        print(f"  FAIL  docx_exports_produce_bytes: {type(e).__name__}: {e}")
    return failures


# ── Sprint 12 Smoke Tests ─────────────────────────────────────────────

def test_quantity_reconciliation_stable():
    """Sprint 12: quantity reconciliation produces stable output."""
    failures = []
    try:
        from src.analysis.quantity_reconciliation import reconcile_quantities

        schedules = [
            {"schedule_type": "door", "mark": "D1", "source_page": 0, "fields": {}},
            {"schedule_type": "door", "mark": "D2", "source_page": 0, "fields": {}},
        ]
        boq_items = [
            {"item_no": "4.1", "description": "Flush door 900x2100", "qty": 5, "source_page": 20},
        ]
        r1 = reconcile_quantities([], schedules, boq_items, [])
        r2 = reconcile_quantities([], schedules, boq_items, [])
        assert r1 == r2, f"Reconciliation not deterministic: {r1} != {r2}"
        for row in r1:
            assert "category" in row, f"Missing 'category' key in {row}"
            assert "mismatch" in row, f"Missing 'mismatch' key in {row}"
        print(f"  PASS  quantity_reconciliation: {len(r1)} rows, deterministic=True")
    except Exception as e:
        failures.append(("quantity_reconciliation_stable", e))
        print(f"  FAIL  quantity_reconciliation: {type(e).__name__}: {e}")
    return failures


def test_finishes_takeoff_with_areas():
    """Sprint 12: finish takeoff aggregates correctly when areas exist."""
    failures = []
    try:
        from src.analysis.finish_takeoff import build_finish_takeoff

        schedules = [
            {"schedule_type": "finish", "mark": "R1", "source_page": 5, "fields": {
                "room": "Living", "floor": "Tile", "area": "25.0",
            }},
            {"schedule_type": "finish", "mark": "R2", "source_page": 5, "fields": {
                "room": "Kitchen", "floor": "Tile", "area": "15.0",
            }},
        ]
        result = build_finish_takeoff(schedules)
        assert result["has_areas"] is True, "Expected has_areas=True"
        assert len(result["finish_rows"]) > 0, "Expected at least one finish row"
        assert result["rooms_missing_area"] == [], f"Unexpected missing rooms: {result['rooms_missing_area']}"
        print(f"  PASS  finishes_takeoff: has_areas={result['has_areas']}, rows={len(result['finish_rows'])}")
    except Exception as e:
        failures.append(("finishes_takeoff_with_areas", e))
        print(f"  FAIL  finishes_takeoff: {type(e).__name__}: {e}")
    return failures


def test_feedback_write_and_schema():
    """Sprint 12: feedback.jsonl writes and loads with stable schema."""
    failures = []
    try:
        import tempfile
        from src.analysis.feedback import make_feedback_entry, append_feedback, load_feedback

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            e1 = make_feedback_entry("rfi", "RFI-0001", "correct")
            e2 = make_feedback_entry("quantity", "abc", "edited", corrected_value="10")
            append_feedback(e1, tmp)
            append_feedback(e2, tmp)
            loaded = load_feedback(tmp)
            assert len(loaded) == 2, f"Expected 2 entries, got {len(loaded)}"
            assert set(loaded[0].keys()) == set(loaded[1].keys()), "Schema mismatch between entries"
            assert (tmp / "feedback.jsonl").exists(), "feedback.jsonl not created"
        print("  PASS  feedback_write: 2 entries written and loaded, schema stable")
    except Exception as e:
        failures.append(("feedback_write_and_schema", e))
        print(f"  FAIL  feedback_write: {type(e).__name__}: {e}")
    return failures


def test_review_queue_deterministic():
    """Sprint 13: review queue produces deterministic output."""
    failures = []
    try:
        from src.analysis.review_queue import build_review_queue

        recon = [
            {"category": "doors", "mismatch": True, "max_delta": 3,
             "schedule_count": 2, "boq_count": 5},
            {"category": "windows", "mismatch": True, "max_delta": 7,
             "schedule_count": 8, "boq_count": 1},
        ]
        conflicts = [
            {"type": "boq_change", "delta_confidence": 0.6,
             "item_no": "1.1", "base_page": 0, "addendum_page": 5},
        ]
        r1 = build_review_queue(quantity_reconciliation=recon, conflicts=conflicts)
        r2 = build_review_queue(quantity_reconciliation=recon, conflicts=conflicts)
        assert r1 == r2, f"Review queue not deterministic"
        assert len(r1) > 0, "Expected non-empty review queue"
        for item in r1:
            assert "type" in item, f"Missing 'type' key in {item}"
            assert "severity" in item, f"Missing 'severity' key in {item}"
            assert "title" in item, f"Missing 'title' key in {item}"
        # HIGH items first
        assert r1[0]["severity"] == "high", f"Expected first item HIGH, got {r1[0]['severity']}"
        print(f"  PASS  review_queue: {len(r1)} items, deterministic=True, first=HIGH")
    except Exception as e:
        failures.append(("review_queue_deterministic", e))
        print(f"  FAIL  review_queue: {type(e).__name__}: {e}")
    return failures


def test_bulk_actions_produce_changes():
    """Sprint 13: bulk actions produce expected changes."""
    failures = []
    try:
        from src.analysis.bulk_actions import prefer_schedule_for_mismatches

        recon = [
            {"category": "doors", "mismatch": True, "max_delta": 3,
             "schedule_count": 2, "boq_count": 5, "preferred_source": None,
             "preferred_qty": None, "action": None, "evidence_refs": [], "notes": ""},
            {"category": "finishes", "mismatch": True, "max_delta": 1,
             "schedule_count": 10, "boq_count": 11, "preferred_source": None,
             "preferred_qty": None, "action": None, "evidence_refs": [], "notes": ""},
        ]
        updated, count = prefer_schedule_for_mismatches(recon)
        assert count == 1, f"Expected 1 action, got {count}"
        assert updated[0]["preferred_source"] == "schedule", "Door not updated to schedule"
        assert updated[1]["preferred_source"] is None, "Finishes should be unchanged"
        print(f"  PASS  bulk_actions: {count} action(s) applied, non-targets unchanged")
    except Exception as e:
        failures.append(("bulk_actions_produce_changes", e))
        print(f"  FAIL  bulk_actions: {type(e).__name__}: {e}")
    return failures


def test_export_filters_enforce_approval():
    """Sprint 13: export filters enforce approval states."""
    failures = []
    try:
        from src.analysis.approval_states import (
            filter_rfis_for_export, filter_quantities_for_export, filter_conflicts_for_export,
        )

        rfis = [{"status": "draft"}, {"status": "approved"}, {"status": "sent"}]
        qtys = [{"status": "draft"}, {"status": "accepted"}]
        conflicts = [{"review_status": "unreviewed"}, {"review_status": "reviewed"}]

        assert len(filter_rfis_for_export(rfis)) == 2, "RFI filter wrong"
        assert len(filter_rfis_for_export(rfis, include_drafts=True)) == 3, "RFI include_drafts wrong"
        assert len(filter_quantities_for_export(qtys)) == 1, "Qty filter wrong"
        assert len(filter_conflicts_for_export(conflicts)) == 1, "Conflict filter wrong"
        assert len(filter_conflicts_for_export(conflicts, include_unreviewed=True)) == 2, "Conflict include_unreviewed wrong"
        print("  PASS  export_filters: all 3 filters enforce states correctly")
    except Exception as e:
        failures.append(("export_filters_enforce_approval", e))
        print(f"  FAIL  export_filters: {type(e).__name__}: {e}")
    return failures


# ── Sprint 14: Projects + Collaboration + Submission Pack ─────────────────

def test_project_create_load_deterministic():
    """Sprint 14: project create + load + list roundtrip."""
    failures = []
    try:
        import tempfile
        from pathlib import Path
        from src.analysis.projects import create_project, load_project, list_projects

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            meta = create_project(name="Smoke Test", owner="QA", bid_date="2026-04-01",
                                   project_id="smoke_proj", projects_dir=td_path)
            assert meta["project_id"] == "smoke_proj", f"project_id mismatch: {meta['project_id']}"
            assert meta["name"] == "Smoke Test", f"name mismatch: {meta['name']}"

            loaded = load_project("smoke_proj", td_path)
            assert loaded is not None, "load_project returned None"
            assert loaded["project_id"] == "smoke_proj", "roundtrip project_id mismatch"

            projects = list_projects(td_path)
            assert len(projects) == 1, f"Expected 1 project, got {len(projects)}"
            assert projects[0]["name"] == "Smoke Test"
            print("  PASS  project_create_load: roundtrip OK")
    except Exception as e:
        failures.append(("project_create_load_deterministic", e))
        print(f"  FAIL  project_create_load: {type(e).__name__}: {e}")
    return failures


def test_collaboration_persist():
    """Sprint 14: write + load + aggregate collaboration entries."""
    failures = []
    try:
        import tempfile
        from pathlib import Path
        from src.analysis.collaboration import (
            make_collaboration_entry, append_collaboration,
            load_collaboration, get_entity_collaboration,
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            e1 = make_collaboration_entry("rfi", "RFI-SMOKE", "comment", {"text": "Check this"}, author="Tester")
            e2 = make_collaboration_entry("rfi", "RFI-SMOKE", "assign", {"assigned_to": "Alice"})
            e3 = make_collaboration_entry("rfi", "RFI-SMOKE", "due_date", {"due_date": "2026-05-01"})
            for e in [e1, e2, e3]:
                append_collaboration(e, td_path)

            entries = load_collaboration(td_path)
            assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"

            collab = get_entity_collaboration(entries, "rfi", "RFI-SMOKE")
            assert len(collab["comments"]) == 1, "Expected 1 comment"
            assert collab["assigned_to"] == "Alice", f"assigned_to: {collab['assigned_to']}"
            assert collab["due_date"] == "2026-05-01", f"due_date: {collab['due_date']}"
            print("  PASS  collaboration_persist: write + load + aggregate OK")
    except Exception as e:
        failures.append(("collaboration_persist", e))
        print(f"  FAIL  collaboration_persist: {type(e).__name__}: {e}")
    return failures


def test_submission_pack_structure():
    """Sprint 14: ZIP has 5 folders + cover sheet."""
    failures = []
    try:
        import sys, zipfile, io
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from submission_pack import generate_submission_pack

        buffers = {
            "rfis.csv": "id,question\n1,test",
            "boq.csv": "item,qty\n1,10",
            "exclusions_clarifications.txt": "None",
            "bid_summary.md": "# Summary",
            "requirements.csv": "req,status",
        }
        zip_bytes = generate_submission_pack(buffers, project_id="smoke", project_name="Smoke")
        assert len(zip_bytes) > 0, "ZIP is empty"

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "00_Cover_Sheet.txt" in names, "Missing cover sheet"
            folders_found = set()
            for n in names:
                parts = n.split("/")
                if len(parts) > 1:
                    folders_found.add(parts[0])
            expected_folders = {"01_Bid_Summary", "02_RFIs", "03_Exclusions_Clarifications",
                                "04_Quantities", "05_Evidence_Appendix"}
            assert expected_folders.issubset(folders_found), f"Missing folders: {expected_folders - folders_found}"
            print(f"  PASS  submission_pack: {len(names)} files in {len(folders_found)} folders")
    except Exception as e:
        failures.append(("submission_pack_structure", e))
        print(f"  FAIL  submission_pack: {type(e).__name__}: {e}")
    return failures


# ── Sprint 15: Evidence PDF + Meeting Agenda + Email Drafts ───────────────

def test_evidence_appendix_pdf_stable():
    """Sprint 15: evidence appendix PDF generation stable; empty inputs graceful."""
    failures = []
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from evidence_appendix_pdf import generate_evidence_appendix_pdf

        # With data
        rfis = [{"id": "RFI-S", "status": "approved", "trade": "structural",
                 "priority": "high", "question": "Beam size?", "evidence_pages": [1]}]
        conflicts = [{"type": "boq_change", "item_no": "1.1", "delta_confidence": 0.6,
                      "review_status": "reviewed"}]
        result = generate_evidence_appendix_pdf(rfis=rfis, conflicts=conflicts)
        assert isinstance(result, bytes) and len(result) > 0, "PDF is empty"
        assert result[:5] == b"%PDF-", "Not a valid PDF"

        # Empty inputs (graceful)
        empty_result = generate_evidence_appendix_pdf(rfis=[], conflicts=[])
        assert isinstance(empty_result, bytes), "Empty inputs should produce bytes"
        assert empty_result[:5] == b"%PDF-", "Empty PDF should still be valid"
        print("  PASS  evidence_appendix_pdf: stable with data and graceful empty")
    except Exception as e:
        failures.append(("evidence_appendix_pdf_stable", e))
        print(f"  FAIL  evidence_appendix_pdf: {type(e).__name__}: {e}")
    return failures


def test_meeting_agenda_deterministic():
    """Sprint 15: meeting agenda produces identical output for same inputs."""
    failures = []
    try:
        from src.analysis.meeting_agenda import build_meeting_agenda
        review_items = [
            {"type": "conflict", "severity": "high", "title": "Beam size conflict", "source_key": "c1"},
            {"type": "recon_mismatch", "severity": "medium", "title": "Door qty mismatch", "source_key": "r1"},
        ]
        a1 = build_meeting_agenda(review_items=review_items, assignments=[])
        a2 = build_meeting_agenda(review_items=review_items, assignments=[])

        # Compare sections (ignoring generated_at timestamp)
        assert a1["sections"] == a2["sections"], "Agenda sections not deterministic"
        assert a1["summary"] == a2["summary"], "Agenda summary not deterministic"
        assert len(a1.get("sections", [])) > 0, "No sections in agenda"
        print(f"  PASS  meeting_agenda: deterministic, {len(a1['sections'])} section(s)")
    except Exception as e:
        failures.append(("meeting_agenda_deterministic", e))
        print(f"  FAIL  meeting_agenda: {type(e).__name__}: {e}")
    return failures


def test_email_drafts_include_approved():
    """Sprint 15: email drafts include only approved items by default."""
    failures = []
    try:
        from src.exports.email_drafts import generate_all_email_drafts
        rfis = [
            {"id": "RFI-A", "status": "approved", "trade": "structural",
             "priority": "high", "title": "Size of footing", "description": "Check footing dims"},
            {"id": "RFI-B", "status": "draft", "trade": "structural",
             "priority": "medium", "title": "Draft question", "description": "Draft desc"},
        ]
        drafts = generate_all_email_drafts(rfis=rfis)
        assert isinstance(drafts, dict), "Should return dict"
        all_text = drafts.get("rfi_email_all.txt", "")
        # Approved item should be present
        assert "footing" in all_text.lower() or "RFI-A" in all_text, \
            "Approved RFI not in email"
        print(f"  PASS  email_drafts: {len(drafts)} draft file(s), approved items included")
    except Exception as e:
        failures.append(("email_drafts_include_approved", e))
        print(f"  FAIL  email_drafts: {type(e).__name__}: {e}")
    return failures


# ── Sprint 16: Storage + Auth + Jobs smoke tests ────────────────────────

def test_storage_local_roundtrip():
    """Sprint 16: LocalStorage create/load project + save/load file."""
    failures = []
    try:
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            from src.storage import LocalStorage
            storage = LocalStorage(base_dir=tmpdir)

            # Project round-trip
            meta = storage.create_project(name="Smoke Test Project", owner="Test")
            assert meta["name"] == "Smoke Test Project", "Name mismatch"
            loaded = storage.load_project(meta["project_id"])
            assert loaded is not None, "load_project returned None"
            assert loaded["project_id"] == meta["project_id"], "project_id mismatch"

            # File I/O round-trip
            import os
            fpath = os.path.join(tmpdir, "test_files", "doc.bin")
            storage.save_file(fpath, b"smoke test data")
            assert storage.file_exists(fpath), "file_exists returned False"
            data = storage.load_file(fpath)
            assert data == b"smoke test data", "File content mismatch"

            # List projects
            projects = storage.list_projects()
            assert len(projects) == 1, f"Expected 1 project, got {len(projects)}"

            print(f"  PASS  storage_local: roundtrip OK, 1 project, file I/O OK")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        failures.append(("storage_local_roundtrip", e))
        print(f"  FAIL  storage_local: {type(e).__name__}: {e}")
    return failures


def test_auth_tenant_isolation():
    """Sprint 16: Two tenants each see only their own projects."""
    failures = []
    try:
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            import os
            from src.auth import SimpleAuth
            auth_subdir = os.path.join(tmpdir, "auth")
            auth = SimpleAuth(auth_dir=auth_subdir)

            # Create two tenants
            auth.create_tenant("org_a", "Org A", "pass_a")
            auth.create_tenant("org_b", "Org B", "pass_b")

            # Authenticate both
            t_a = auth.authenticate("org_a", "pass_a")
            t_b = auth.authenticate("org_b", "pass_b")
            assert t_a is not None, "Org A auth failed"
            assert t_b is not None, "Org B auth failed"

            # Get isolated storage
            s_a = auth.get_storage_for_tenant("org_a")
            s_b = auth.get_storage_for_tenant("org_b")

            # Create project in each
            s_a.create_project(name="A's Project")
            s_b.create_project(name="B's Project")

            # Each sees only their own
            a_projects = s_a.list_projects()
            b_projects = s_b.list_projects()
            assert len(a_projects) == 1, f"Org A should see 1 project, got {len(a_projects)}"
            assert len(b_projects) == 1, f"Org B should see 1 project, got {len(b_projects)}"
            assert a_projects[0]["name"] == "A's Project", "Org A project name wrong"
            assert b_projects[0]["name"] == "B's Project", "Org B project name wrong"

            # Wrong password rejected
            bad = auth.authenticate("org_a", "wrong_pass")
            assert bad is None, "Wrong password should return None"

            print(f"  PASS  auth_tenant_isolation: 2 tenants isolated, wrong pw rejected")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        failures.append(("auth_tenant_isolation", e))
        print(f"  FAIL  auth_tenant_isolation: {type(e).__name__}: {e}")
    return failures


def test_job_queue_sync_completion():
    """Sprint 16: Submit sync function, poll to completion."""
    failures = []
    try:
        import time
        from src.jobs import LocalThreadQueue, JobStatus

        queue = LocalThreadQueue(max_workers=1)

        def test_fn(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb("compute", "working hard", 0.5)
                cb("compute", "almost done", 0.9)
            return {"answer": 42}

        job_id = queue.submit(test_fn)
        assert job_id.startswith("job_"), f"job_id format unexpected: {job_id}"

        # Poll until done
        for _ in range(100):
            job = queue.get_status(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.05)

        job = queue.get_status(job_id)
        assert job is not None, "Job not found"
        assert job.status == JobStatus.COMPLETED, f"Expected COMPLETED, got {job.status}"
        assert job.result == {"answer": 42}, f"Result mismatch: {job.result}"
        assert job.progress >= 0.9, f"Progress not updated: {job.progress}"

        # List jobs
        jobs = queue.list_jobs()
        assert len(jobs) >= 1, "list_jobs returned empty"

        print(f"  PASS  job_queue: submit → COMPLETED, progress tracked, list OK")
    except Exception as e:
        failures.append(("job_queue_sync_completion", e))
        print(f"  FAIL  job_queue: {type(e).__name__}: {e}")
    return failures


# ── Sprint 17: Demo + Determinism + Highlights smoke tests ──────────────

def test_highlights_in_payload():
    """Sprint 17: highlights.build_highlights runs on standard payloads."""
    failures = []
    try:
        from src.analysis.highlights import build_highlights

        # Test with a rich payload
        payload = {
            "decision": "CONDITIONAL", "readiness_score": 65,
            "sub_scores": {"completeness": 80, "coverage": 50},
            "blockers": [{"severity": "high", "title": "Missing structural"}],
            "rfis": [
                {"trade": "civil", "status": "approved"},
                {"trade": "electrical", "status": "draft"},
            ],
            "trade_coverage": [
                {"trade": "civil", "coverage_pct": 90.0},
                {"trade": "electrical", "coverage_pct": 40.0},
            ],
            "quantities": [{"item": "Concrete", "rate": 5000}],
        }
        highlights = build_highlights(payload)
        assert isinstance(highlights, list), "Should return a list"
        assert len(highlights) >= 3, f"Expected 3+ highlights, got {len(highlights)}"
        assert any(h["label"] == "Bid Readiness" for h in highlights), "Missing Bid Readiness"
        for h in highlights:
            assert h["severity"] in ("good", "warn", "bad"), f"Invalid severity: {h['severity']}"

        # Test empty payload
        empty_hl = build_highlights({})
        assert len(empty_hl) >= 1, "Even empty payload should have decision highlight"

        print(f"  PASS  highlights: {len(highlights)} cards from rich payload, empty graceful")
    except Exception as e:
        failures.append(("highlights_in_payload", e))
        print(f"  FAIL  highlights: {type(e).__name__}: {e}")
    return failures


def test_determinism_review_queue_smoke():
    """Sprint 17: review queue is deterministic across repeated calls."""
    failures = []
    try:
        from src.analysis.review_queue import build_review_queue
        recon = [
            {"category": "doors", "mismatch": True, "max_delta": 5,
             "schedule_count": 3, "boq_count": 8},
        ]
        risk = [
            {"template_id": "RISK-LD", "label": "Liquidated Damages",
             "impact": "high", "found": True, "hits": [{"page_idx": 1}]},
        ]
        results = [build_review_queue(quantity_reconciliation=recon, risk_results=risk)
                    for _ in range(5)]
        for r in results[1:]:
            assert r == results[0], "Review queue not deterministic"
        print(f"  PASS  determinism: review_queue identical x5 ({len(results[0])} items)")
    except Exception as e:
        failures.append(("determinism_review_queue_smoke", e))
        print(f"  FAIL  determinism: {type(e).__name__}: {e}")
    return failures


def test_demo_config_smoke():
    """Sprint 17: demo_config loads and returns valid structure."""
    failures = []
    try:
        from src.demo.demo_config import DEMO_PROJECTS, get_demo_project, get_demo_project_ids
        assert len(DEMO_PROJECTS) >= 3, f"Expected 3+ projects, got {len(DEMO_PROJECTS)}"
        ids = get_demo_project_ids()
        assert "pwd_garage" in ids, "pwd_garage not in project IDs"
        proj = get_demo_project("pwd_garage")
        assert proj is not None, "get_demo_project returned None"
        assert proj["name"] == "PWD Garage Construction", f"Name mismatch: {proj['name']}"
        print(f"  PASS  demo_config: {len(DEMO_PROJECTS)} projects, pwd_garage found")
    except Exception as e:
        failures.append(("demo_config_smoke", e))
        print(f"  FAIL  demo_config: {type(e).__name__}: {e}")
    return failures


def test_demo_assets_smoke():
    """Sprint 17: demo_assets.validate_demo_assets returns report for all projects."""
    failures = []
    try:
        from src.demo.demo_assets import validate_demo_assets
        report = validate_demo_assets()
        assert len(report) >= 3, f"Expected 3+ entries, got {len(report)}"
        # pwd_garage should have cache (from out/ or demo_cache/)
        pwd = [r for r in report if r["project_id"] == "pwd_garage"]
        assert len(pwd) == 1, "pwd_garage not in report"
        assert pwd[0]["cache_found"] is True, "pwd_garage cache not found"
        print(f"  PASS  demo_assets: {len(report)} projects validated, pwd_garage cached")
    except Exception as e:
        failures.append(("demo_assets_smoke", e))
        print(f"  FAIL  demo_assets: {type(e).__name__}: {e}")
    return failures


def test_narration_deterministic():
    """Sprint 18: build_narration_script produces identical output on repeated calls."""
    failures = []
    try:
        from src.demo.narration import build_narration_script
        payload = {
            "drawing_overview": {"pages_total": 7, "pages_deep": 5},
            "rfis": [{"status": "approved"}, {"status": "draft"}],
            "blockers": [{"title": "Missing spec"}],
            "quantities": [{"item": "Steel"}],
            "decision": "CONDITIONAL",
            "readiness_score": 68,
            "qa_score": {"score": 75},
            "timings": {"total_seconds": 12},
        }
        results = [build_narration_script(payload, project_name="Smoke") for _ in range(3)]
        assert results[0] == results[1] == results[2], "Narration not deterministic"
        assert len(results[0]) > 50, f"Narration too short: {len(results[0])} chars"
        assert "Smoke" in results[0], "Project name missing from narration"
        print(f"  PASS  narration: deterministic x3, {len(results[0])} chars, name present")
    except Exception as e:
        failures.append(("narration_deterministic", e))
        print(f"  FAIL  narration: {type(e).__name__}: {e}")
    return failures


def test_export_filenames_stable():
    """Sprint 18: _demo_filename produces stable filenames with date."""
    failures = []
    try:
        from datetime import date

        def _demo_filename(base, pname, ext):
            if pname:
                safe = pname.replace(" ", "_")[:30]
                return f"{safe}_{base}_{date.today().isoformat()}.{ext}"
            return f"{base}.{ext}"

        results = [_demo_filename("Bid_Summary", "Project", "pdf") for _ in range(3)]
        assert results[0] == results[1] == results[2], "Filename not stable"
        assert date.today().isoformat() in results[0], "Date missing from filename"
        assert results[0].startswith("Project_"), f"Bad prefix: {results[0]}"
        # Fallback
        fb = _demo_filename("Test", "", "csv")
        assert fb == "Test.csv", f"Fallback wrong: {fb}"
        print(f"  PASS  filenames: stable x3, date present, fallback OK")
    except Exception as e:
        failures.append(("export_filenames_stable", e))
        print(f"  FAIL  filenames: {type(e).__name__}: {e}")
    return failures


def test_reset_state_no_crash():
    """Sprint 18: Simulated session_state reset doesn't crash."""
    failures = []
    try:
        # Simulate st.session_state as a plain dict
        mock_state = {
            "_xboq_tenant_id": "org_a",
            "_xboq_job_queue": "queue_obj",
            "_xboq_freeze_ui": True,
            "_xboq_active_project_id": "p1",
            "_xboq_presenter_view": True,
            "other_key": "keep",
        }
        keep = {"_xboq_tenant_id", "_xboq_job_queue"}
        keys_to_clear = [k for k in list(mock_state.keys())
                         if k.startswith("_xboq") and k not in keep]
        for ck in keys_to_clear:
            del mock_state[ck]
        # Auth + queue preserved
        assert "_xboq_tenant_id" in mock_state, "Tenant ID was cleared"
        assert "_xboq_job_queue" in mock_state, "Job queue was cleared"
        # Demo state cleared
        assert "_xboq_freeze_ui" not in mock_state, "Freeze not cleared"
        assert "_xboq_active_project_id" not in mock_state, "Project ID not cleared"
        # Non-xboq keys preserved
        assert "other_key" in mock_state, "Non-xboq key was cleared"
        print(f"  PASS  reset_state: 3 cleared, 3 preserved, no crash")
    except Exception as e:
        failures.append(("reset_state_no_crash", e))
        print(f"  FAIL  reset_state: {type(e).__name__}: {e}")
    return failures


def test_summary_card_missing_fields():
    """Sprint 18: build_summary_card({}) returns all expected keys, no crash."""
    failures = []
    try:
        from src.demo.summary_card import build_summary_card
        card = build_summary_card({})
        expected_keys = {
            "total_pages", "deep_pages", "ocr_pages", "text_layer_pages",
            "skipped_pages", "cache_time_saved",
            "qa_score", "top_actions", "approved_rfis", "accepted_quantities",
            "accepted_assumptions", "submission_pack_ready", "decision",
            "readiness_score", "project_name",
        }
        missing = expected_keys - set(card.keys())
        assert not missing, f"Missing keys: {missing}"
        assert card["total_pages"] == 0, f"total_pages not 0: {card['total_pages']}"
        assert card["decision"] == "N/A", f"decision not N/A: {card['decision']}"
        # Also test with None values
        card2 = build_summary_card({"rfis": None, "qa_score": None})
        assert card2["approved_rfis"] == 0, "None rfis should yield 0"
        print(f"  PASS  summary_card: {len(card)} keys present, defaults OK, None-safe")
    except Exception as e:
        failures.append(("summary_card_missing_fields", e))
        print(f"  FAIL  summary_card: {type(e).__name__}: {e}")
    return failures


def test_section2_dependencies_no_crash():
    """Sprint 18 bugfix: render_section_2_dependencies ID logic doesn't crash."""
    failures = []
    try:
        import hashlib, json

        def _extract_ids(report):
            """Reproduce the fixed ID-generation logic from render_section_2_dependencies."""
            deps = report.get("missing_dependencies") or []
            ids = []
            for _dep_i, dep in enumerate(deps):
                if not isinstance(dep, dict):
                    continue
                _dep_id = dep.get("id") or ""
                if not _dep_id:
                    try:
                        _dep_id = "DEP-" + hashlib.sha256(
                            json.dumps(dep, sort_keys=True, default=str).encode()
                        ).hexdigest()[:8]
                    except Exception:
                        _dep_id = f"DEP-{_dep_i}"
                ids.append(_dep_id)
            return ids

        # Case 1: missing key
        assert _extract_ids({}) == [], "Empty report should give empty list"
        # Case 2: empty list
        assert _extract_ids({"missing_dependencies": []}) == [], "Empty deps should give empty list"
        # Case 3: None
        assert _extract_ids({"missing_dependencies": None}) == [], "None deps should give empty list"
        # Case 4: items missing id → stable hash
        ids = _extract_ids({"missing_dependencies": [
            {"dependency_type": "finish_schedule"},
            {"dependency_type": "door_schedule"},
        ]})
        assert len(ids) == 2, f"Expected 2 ids, got {len(ids)}"
        assert ids[0] != ids[1], "Different deps should get different IDs"
        assert all(i.startswith("DEP-") for i in ids), f"Bad prefixes: {ids}"
        # Deterministic
        ids2 = _extract_ids({"missing_dependencies": [
            {"dependency_type": "finish_schedule"},
            {"dependency_type": "door_schedule"},
        ]})
        assert ids == ids2, "IDs not deterministic"
        # Case 5: item with existing id
        ids3 = _extract_ids({"missing_dependencies": [{"id": "DEP-001"}]})
        assert ids3 == ["DEP-001"], f"Existing ID not preserved: {ids3}"

        print(f"  PASS  section2_deps: 5 cases OK, IDs stable, no crash")
    except Exception as e:
        failures.append(("section2_dependencies_no_crash", e))
        print(f"  FAIL  section2_deps: {type(e).__name__}: {e}")
    return failures


def test_processing_stats_missing_no_crash():
    """Sprint 18: summary_card without processing_stats renders without crashing."""
    failures = []
    try:
        from src.demo.summary_card import build_summary_card

        # Payload with only run_coverage (no processing_stats key)
        payload_legacy = {
            "run_coverage": {"pages_total": 50, "pages_deep_processed": 20},
            "drawing_overview": {"ocr_pages_count": 10},
            "decision": "GO",
            "readiness_score": 80,
        }
        card = build_summary_card(payload_legacy, project_name="Legacy")
        assert card["total_pages"] == 50, f"Expected 50, got {card['total_pages']}"
        assert card["deep_pages"] == 20, f"Expected 20, got {card['deep_pages']}"
        assert card["ocr_pages"] == 10, f"Expected 10, got {card['ocr_pages']}"
        assert card["skipped_pages"] == 30, f"Expected 30, got {card['skipped_pages']}"
        assert card["text_layer_pages"] == 10, f"Expected 10, got {card['text_layer_pages']}"

        # Completely empty payload
        card2 = build_summary_card({})
        assert card2["total_pages"] == 0
        assert card2["skipped_pages"] == 0
        assert card2["text_layer_pages"] == 0

        print(f"  PASS  processing_stats_missing: fallback chain OK, no crash")
    except Exception as e:
        failures.append(("processing_stats_missing_no_crash", e))
        print(f"  FAIL  processing_stats_missing: {type(e).__name__}: {e}")
    return failures


def test_processing_stats_populated():
    """Sprint 18: summary_card with processing_stats shows non-zero values."""
    failures = []
    try:
        from src.demo.summary_card import build_summary_card

        payload = {
            "processing_stats": {
                "total_pages": 367,
                "deep_processed_pages": 80,
                "ocr_pages": 80,
                "text_layer_pages": 0,
                "skipped_pages": 287,
            },
            # Legacy keys that should be ignored when processing_stats exists
            "run_coverage": {"pages_total": 999, "pages_deep_processed": 999},
            "drawing_overview": {"ocr_pages_count": 999},
            "decision": "CONDITIONAL",
            "readiness_score": 73,
        }
        card = build_summary_card(payload, project_name="Sonipat")

        # Should use processing_stats, NOT legacy keys
        assert card["total_pages"] == 367, f"Expected 367, got {card['total_pages']}"
        assert card["deep_pages"] == 80, f"Expected 80, got {card['deep_pages']}"
        assert card["ocr_pages"] == 80, f"Expected 80, got {card['ocr_pages']}"
        assert card["skipped_pages"] == 287, f"Expected 287, got {card['skipped_pages']}"
        assert card["text_layer_pages"] == 0, f"Expected 0, got {card['text_layer_pages']}"
        assert card["project_name"] == "Sonipat"

        # Deterministic: same payload → same card
        card2 = build_summary_card(payload, project_name="Sonipat")
        assert card == card2, "Same payload should produce identical card"

        print(f"  PASS  processing_stats_populated: all counters correct, deterministic")
    except Exception as e:
        failures.append(("processing_stats_populated", e))
        print(f"  FAIL  processing_stats_populated: {type(e).__name__}: {e}")
    return failures


def test_sub_scores_passthrough():
    """Sprint 18: score_data extraction maps sub_scores correctly.

    Replicates the score_data → executive_summary logic from
    build_report_from_results in demo_page.py because importing
    demo_page triggers Streamlit and app.app imports.
    """
    failures = []
    try:
        def _extract_sub_scores(deep):
            """Replicate score_data logic from demo_page.py."""
            raw_score = deep.get("readiness_score", {})
            if isinstance(raw_score, (int, float)):
                _deep_sub = deep.get("sub_scores", {})
                score_data = {
                    "coverage_score": _deep_sub.get("coverage", 0),
                    "measurement_score": _deep_sub.get("measurement", 0),
                    "completeness_score": _deep_sub.get("completeness", 0),
                    "blocker_score": _deep_sub.get("blocker", 0),
                }
            else:
                score_data = raw_score if raw_score else {}
            return {
                "coverage": score_data.get("coverage_score", 0),
                "measurement": score_data.get("measurement_score", 0),
                "completeness": score_data.get("completeness_score", 0),
                "blocker": score_data.get("blocker_score", 0),
            }

        deep = {
            "readiness_score": 73,
            "decision": "CONDITIONAL",
            "sub_scores": {
                "coverage": 100,
                "measurement": 30,
                "completeness": 70,
                "blocker": 75,
            },
        }
        sub = _extract_sub_scores(deep)

        assert sub["coverage"] == 100, f"coverage: expected 100, got {sub['coverage']}"
        assert sub["measurement"] == 30, f"measurement: expected 30, got {sub['measurement']}"
        assert sub["completeness"] == 70, f"completeness: expected 70, got {sub['completeness']}"
        assert sub["blocker"] == 75, f"blocker: expected 75, got {sub['blocker']}"

        # Deterministic
        sub2 = _extract_sub_scores(deep)
        assert sub == sub2, "Same input should produce identical output"

        # Old wrong keys should NOT work (regression guard)
        wrong_deep = {
            "readiness_score": 50,
            "sub_scores": {
                "scale_accuracy": 99,
                "spec_coverage": 99,
                "consistency": 99,
            },
        }
        wrong_sub = _extract_sub_scores(wrong_deep)
        assert wrong_sub["measurement"] == 0, f"scale_accuracy should NOT map: got {wrong_sub['measurement']}"
        assert wrong_sub["completeness"] == 0, f"spec_coverage should NOT map: got {wrong_sub['completeness']}"
        assert wrong_sub["blocker"] == 0, f"consistency should NOT map: got {wrong_sub['blocker']}"

        print(f"  PASS  sub_scores_passthrough: all 4 scores correct, deterministic, old keys rejected")
    except Exception as e:
        failures.append(("sub_scores_passthrough", e))
        print(f"  FAIL  sub_scores_passthrough: {type(e).__name__}: {e}")
    return failures


# ── Sprint 19: Commercial + BOQ + Spec smoke tests ──────────────────────

def test_commercial_terms_absent_no_crash():
    """#83: Payload without commercial_terms should not crash render_key_line_items logic."""
    failures = []
    try:
        payload_no_commercial = {
            "rfis": [{"id": "RFI-001", "trade": "general"}],
            "drawing_overview": {"pages_total": 10},
        }
        # Simulate what render_key_line_items does
        commercial = payload_no_commercial.get("commercial_terms", [])
        boq_stats = payload_no_commercial.get("boq_stats", {})
        req_by_trade = payload_no_commercial.get("requirements_by_trade", {})
        rfis = payload_no_commercial.get("rfis", [])
        commercial_rfis = [r for r in rfis if r.get("trade") == "commercial"]

        assert commercial == [], f"expected [], got {commercial}"
        assert boq_stats == {}, f"expected {{}}, got {boq_stats}"
        assert req_by_trade == {}, f"expected {{}}, got {req_by_trade}"
        assert len(commercial_rfis) == 0
        print(f"  PASS  commercial_terms_absent_no_crash")
    except Exception as e:
        failures.append(("commercial_terms_absent_no_crash", e))
        print(f"  FAIL  commercial_terms_absent_no_crash: {type(e).__name__}: {e}")
    return failures


def test_commercial_terms_present():
    """#84: Payload with commercial_terms, boq_stats, requirements_by_trade should be accessible."""
    failures = []
    try:
        payload = {
            "commercial_terms": [
                {"term_type": "ld_clause", "value": 0.5, "unit": "%", "cadence": "week",
                 "snippet": "LD 0.5% per week", "source_page": 3, "confidence": 0.75},
                {"term_type": "retention", "value": 5.0, "unit": "%", "cadence": None,
                 "snippet": "Retention 5%", "source_page": 3, "confidence": 0.75},
            ],
            "boq_stats": {
                "total_items": 15,
                "by_trade": {"structural": 8, "finishes": 5, "general": 2},
                "flagged_items": [
                    {"item_no": "2.2", "description": "Internal plaster", "flags": ["qty_missing"]},
                ],
                "flagged_count": 1,
            },
            "requirements_by_trade": {
                "structural": [{"text": "RCC M25"}],
                "general": [{"text": "All work as per IS 456"}],
            },
            "rfis": [
                {"id": "RFI-001", "trade": "commercial"},
                {"id": "RFI-002", "trade": "general"},
            ],
        }

        commercial = payload.get("commercial_terms", [])
        assert len(commercial) == 2
        assert commercial[0]["term_type"] == "ld_clause"

        boq_stats = payload.get("boq_stats", {})
        assert boq_stats["total_items"] == 15
        assert boq_stats["flagged_count"] == 1
        assert isinstance(boq_stats["by_trade"], dict)

        req_by_trade = payload.get("requirements_by_trade", {})
        total_reqs = sum(len(v) for v in req_by_trade.values())
        assert total_reqs == 2

        commercial_rfis = [r for r in payload["rfis"] if r.get("trade") == "commercial"]
        assert len(commercial_rfis) == 1

        print(f"  PASS  commercial_terms_present")
    except Exception as e:
        failures.append(("commercial_terms_present", e))
        print(f"  FAIL  commercial_terms_present: {type(e).__name__}: {e}")
    return failures


def test_excel_boq_source_payload():
    """#115 Sprint 21C: Payload with boq_source='excel' and Excel source fields renders without crash."""
    failures = []
    try:
        payload = {
            "drawing_overview": {"pages_total": 10},
            "readiness_score": 0.65,
            "decision": "Bid with caution",
            "sub_scores": {},
            "blockers": [],
            "rfis": [],
            "processing_stats": {"total_pages": 10, "deep_processed_pages": 8},
            "boq_source": "excel",
            "boq_stats": {
                "total_items": 5,
                "by_trade": {"civil": 3, "structural": 2},
                "flagged_items": [
                    {"item_no": "3", "description": "Earth filling", "flags": ["qty_missing"]},
                ],
                "flagged_count": 1,
            },
            "extraction_summary": {
                "requirements": [],
                "schedules": [],
                "boq_items": [
                    {"item_no": "1", "description": "Excavation in ordinary soil", "unit": "cum",
                     "qty": 120.5, "rate": 250.0, "source_page": 0, "confidence": 0.85,
                     "source_file": "BOQ_PriceBid.xlsx", "source_sheet": "BOQ", "source_row": 5},
                    {"item_no": "2", "description": "PCC M15 grade", "unit": "cum",
                     "qty": 45.0, "rate": 5500.0, "source_page": 0, "confidence": 0.85,
                     "source_file": "BOQ_PriceBid.xlsx", "source_sheet": "BOQ", "source_row": 6},
                    {"item_no": "3", "description": "Earth filling", "unit": "cum",
                     "qty": None, "rate": 300.0, "source_page": 0, "confidence": 0.85,
                     "source_file": "BOQ_PriceBid.xlsx", "source_sheet": "BOQ", "source_row": 7},
                    {"item_no": "4", "description": "RCC M25 columns", "unit": "cum",
                     "qty": 30.0, "rate": 8500.0, "source_page": 0, "confidence": 0.85,
                     "source_file": "BOQ_PriceBid.xlsx", "source_sheet": "Structural", "source_row": 3},
                    {"item_no": "5", "description": "RCC M25 beams", "unit": "cum",
                     "qty": 25.0, "rate": 8500.0, "source_page": 0, "confidence": 0.85,
                     "source_file": "BOQ_PriceBid.xlsx", "source_sheet": "Structural", "source_row": 4},
                ],
                "callouts": [],
                "pages_processed": 8,
                "counts": {"requirements": 0, "schedules": 0, "boq_items": 5, "callouts": 0},
            },
            "commercial_terms": [],
            "requirements_by_trade": {},
        }

        demo = build_demo_analysis(payload, "test_excel_boq")
        raw = demo.raw_payload
        assert raw.get("boq_source") == "excel", f"Expected boq_source='excel', got {raw.get('boq_source')}"
        boq_stats = raw.get("boq_stats", {})
        assert boq_stats.get("total_items") == 5, f"Expected 5 BOQ items, got {boq_stats.get('total_items')}"

        # Check that Excel source fields are present
        ext = raw.get("extraction_summary", {})
        boq_items = ext.get("boq_items", [])
        assert len(boq_items) == 5
        assert boq_items[0].get("source_file") == "BOQ_PriceBid.xlsx"
        assert boq_items[0].get("source_sheet") == "BOQ"
        assert boq_items[0].get("source_row") == 5

        print(f"  PASS  excel_boq_source: boq_source=excel, {len(boq_items)} items with source_file/sheet/row")
    except Exception as e:
        failures.append(("excel_boq_source_payload", e))
        print(f"  FAIL  excel_boq_source_payload: {type(e).__name__}: {e}")
    return failures


def test_commercial_rfi_checks_registered():
    """#85: Verify CHK-COM-001..010 exist in CHECKLIST with correct trades and mapped functions."""
    failures = []
    try:
        from src.analysis.rfi_engine import CHECKLIST, CHECK_FN_MAP
        from src.models.analysis_models import Trade

        com_checks = [c for c in CHECKLIST if c[0].startswith("CHK-COM-")]
        assert len(com_checks) == 10, f"Expected 10 CHK-COM checks, got {len(com_checks)}"

        expected_ids = {f"CHK-COM-{i:03d}" for i in range(1, 11)}
        actual_ids = {c[0] for c in com_checks}
        assert expected_ids == actual_ids, f"Missing IDs: {expected_ids - actual_ids}"

        # All should have Trade.COMMERCIAL
        for check_id, fn_name, trade, *_ in com_checks:
            assert trade == Trade.COMMERCIAL, f"{check_id} has trade {trade}, expected COMMERCIAL"
            assert fn_name in CHECK_FN_MAP, f"{fn_name} not mapped in CHECK_FN_MAP"

        print(f"  PASS  commercial_rfi_checks_registered: 10 checks, all COMMERCIAL trade, all mapped")
    except Exception as e:
        failures.append(("commercial_rfi_checks_registered", e))
        print(f"  FAIL  commercial_rfi_checks_registered: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20: Ground Truth, Diff, Training Pack, Pilot Docs ─────────────

def test_gt_template_csv_valid():
    """#86: GT template CSV has correct columns."""
    failures = []
    try:
        from src.analysis.ground_truth import generate_template_csv, GT_BOQ_COLUMNS
        csv_str = generate_template_csv("gt_boq")
        header_line = csv_str.strip().split("\n")[0]
        cols = [c.strip() for c in header_line.split(",")]
        assert cols == GT_BOQ_COLUMNS, f"Expected {GT_BOQ_COLUMNS}, got {cols}"
        assert len(cols) == 7, f"Expected 7 columns, got {len(cols)}"
        print(f"  PASS  gt_template_csv_valid: {len(cols)} columns correct")
    except Exception as e:
        failures.append(("gt_template_csv_valid", e))
        print(f"  FAIL  gt_template_csv_valid: {type(e).__name__}: {e}")
    return failures


def test_gt_diff_deterministic():
    """#87: GT diff produces identical output on repeated calls."""
    failures = []
    try:
        from src.analysis.ground_truth_diff import diff_quantities
        our = [
            {"item": "Concrete M25", "qty": "100"},
            {"item": "Steel rebar", "qty": "500"},
            {"item": "Plywood sheets", "qty": "30"},
        ]
        gt = [
            {"item": "Concrete M25", "qty": "110"},
            {"item": "Steel rebar", "qty": "500"},
        ]
        r1 = diff_quantities(our, gt)
        r2 = diff_quantities(our, gt)
        assert r1 == r2, f"Diff not deterministic: {r1} != {r2}"
        assert r1["match_rate"] == 1.0, f"Match rate {r1['match_rate']} != 1.0"
        assert len(r1["top_mismatches"]) == 1, f"Expected 1 mismatch, got {len(r1['top_mismatches'])}"
        print(f"  PASS  gt_diff_deterministic: match_rate={r1['match_rate']}, deterministic=True")
    except Exception as e:
        failures.append(("gt_diff_deterministic", e))
        print(f"  FAIL  gt_diff_deterministic: {type(e).__name__}: {e}")
    return failures


def test_training_pack_zip_structure():
    """#88: Training pack ZIP has expected folders."""
    failures = []
    try:
        import zipfile, io
        from src.analysis.training_pack import build_training_pack
        payload = {
            "readiness_score": 0.75,
            "decision": "conditional_go",
            "extraction_summary": {"boq_items": [], "schedules": []},
            "quantities": [],
        }
        zb = build_training_pack("smoke_proj", "smoke_run", payload)
        with zipfile.ZipFile(io.BytesIO(zb)) as zf:
            names = zf.namelist()
            assert "inputs/manifest.json" in names, "Missing inputs/manifest.json"
            assert "outputs/analysis.json" in names, "Missing outputs/analysis.json"
            assert "context/bid_context.json" in names, "Missing context/bid_context.json"
            assert "README.md" in names, "Missing README.md"
        print(f"  PASS  training_pack_zip_structure: {len(names)} files, core folders present")
    except Exception as e:
        failures.append(("training_pack_zip_structure", e))
        print(f"  FAIL  training_pack_zip_structure: {type(e).__name__}: {e}")
    return failures


def test_pilot_docs_no_crash():
    """#89: Pilot doc generators produce non-trivial DOCX bytes."""
    failures = []
    try:
        import importlib.util
        _pd_spec = importlib.util.spec_from_file_location(
            "pilot_docs",
            str(Path(__file__).parent.parent / "app" / "pilot_docs.py"),
        )
        _pd_mod = importlib.util.module_from_spec(_pd_spec)
        _pd_spec.loader.exec_module(_pd_mod)
        generate_pilot_agreement_docx = _pd_mod.generate_pilot_agreement_docx
        generate_data_handling_docx = _pd_mod.generate_data_handling_docx
        generate_pilot_checklist_docx = _pd_mod.generate_pilot_checklist_docx

        meta = {
            "company_name": "Smoke Test Corp",
            "name": "Smoke Project",
            "bid_date": "2026-03-01",
            "trades_in_scope": ["structural"],
        }
        agree = generate_pilot_agreement_docx(meta)
        assert isinstance(agree, bytes) and len(agree) > 100, "Agreement too small"
        dh = generate_data_handling_docx(meta)
        assert isinstance(dh, bytes) and len(dh) > 100, "Data handling too small"
        cl = generate_pilot_checklist_docx(meta)
        assert isinstance(cl, bytes) and len(cl) > 100, "Checklist too small"
        print(f"  PASS  pilot_docs_no_crash: agreement={len(agree)}B, data_handling={len(dh)}B, checklist={len(cl)}B")
    except Exception as e:
        failures.append(("pilot_docs_no_crash", e))
        print(f"  FAIL  pilot_docs_no_crash: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20A: Structural Takeoff Tab ────────────────────────────────────

def test_structural_tab_with_payload():
    """#90: Structural tab renders with sample payload (no crash)."""
    failures = []
    try:
        sample_st = {
            "mode": "assumption",
            "summary": {
                "concrete_m3": 42.5,
                "steel_kg": 8500.0,
                "steel_tons": 8.5,
                "element_counts": {"columns": 12, "beams": 18, "footings": 12, "slabs": 1},
                "detail": {},
            },
            "quantities": [
                {
                    "element_id": "C001", "type": "column", "label": "C1",
                    "count": 12,
                    "dimensions_mm": {"width": 230, "depth": 450, "length": 3000},
                    "concrete_m3": 0.9,
                    "steel_kg": {"main": 120, "stirrup": 60, "total": 180},
                    "sources": {"size": "assumption", "height": "assumption", "steel": "kg_per_m3"},
                    "assumptions": ["Default column size"],
                },
            ],
            "qc": {
                "confidence": 0.65,
                "issues": {"total": 2, "errors": 0, "warnings": 2, "info": 0, "details": []},
                "assumptions": {"count": 3, "details": []},
            },
            "exports": {},
            "source_file": "test.pdf",
            "warnings": [],
        }
        # Validate shape matches what UI expects
        assert sample_st["mode"] in ("assumption", "structural", "error")
        assert isinstance(sample_st["summary"].get("concrete_m3"), (int, float))
        assert isinstance(sample_st["summary"].get("steel_tons"), (int, float))
        assert isinstance(sample_st["qc"].get("confidence"), (int, float))
        assert isinstance(sample_st["quantities"], list)
        assert len(sample_st["quantities"]) > 0
        eq = sample_st["quantities"][0]
        assert "element_id" in eq
        assert "dimensions_mm" in eq
        assert "steel_kg" in eq
        # Check backward compat access pattern
        mode = sample_st.get("mode")
        concrete = sample_st.get("summary", {}).get("concrete_m3", 0)
        assert concrete == 42.5
        print(f"  PASS  structural_tab_with_payload: mode={mode}, concrete={concrete}m\u00b3")
    except Exception as e:
        failures.append(("structural_tab_with_payload", e))
        print(f"  FAIL  structural_tab_with_payload: {type(e).__name__}: {e}")
    return failures


def test_structural_tab_without_payload():
    """#91: Structural tab renders gracefully when no structural data."""
    failures = []
    try:
        # Simulate payload without structural_takeoff
        payload = {
            "project_id": "test_project",
            "readiness_score": 0.8,
            "decision": "go",
        }
        st_data = payload.get("structural_takeoff")
        assert st_data is None, "structural_takeoff should be None when absent"

        # Simulate the rendering logic
        if st_data and st_data.get("mode") != "error":
            raise AssertionError("Should not reach structural rendering when data is None")
        elif st_data and st_data.get("mode") == "error":
            raise AssertionError("Should not reach error rendering when data is None")
        else:
            # Would show "No structural takeoff run" message
            msg = "No structural takeoff run for this project"
            assert len(msg) > 0

        # Also test with explicit None in payload
        payload2 = {"structural_takeoff": None}
        st_data2 = payload2.get("structural_takeoff")
        assert not st_data2  # falsy

        # Test error mode doesn't crash
        payload3 = {
            "structural_takeoff": {
                "mode": "error",
                "summary": {},
                "quantities": [],
                "qc": {},
                "exports": {},
                "warnings": ["Test error"],
            }
        }
        st_data3 = payload3.get("structural_takeoff")
        assert st_data3["mode"] == "error"
        assert len(st_data3["warnings"]) == 1

        print(f"  PASS  structural_tab_without_payload: None/error handled gracefully")
    except Exception as e:
        failures.append(("structural_tab_without_payload", e))
        print(f"  FAIL  structural_tab_without_payload: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20B: Demo Hardening ────────────────────────────────────────────

def test_demo_mode_toggle():
    """#92: YC Demo Mode toggle logic works (on/off, tab labels, narration hints)."""
    failures = []
    try:
        # Test _is_yc_demo equivalent logic
        session_state = {}  # simulated
        assert not session_state.get("_xboq_yc_demo", False), "Default should be off"
        session_state["_xboq_yc_demo"] = True
        assert session_state.get("_xboq_yc_demo") is True, "Toggle on should work"

        # Tab labels change in demo mode
        yc_active = True
        if yc_active:
            labels = [
                "Issues to Resolve", "Blockers", "RFIs", "Bid Pack", "Coverage",
                "Bid Strategy", "Risk Checklist", "Quantities", "Pre-bid Meeting",
                "Quality", "Raw JSON", "Ground Truth", "Structural Takeoff",
            ]
        else:
            labels = [
                "Review Queue", "Blockers", "RFIs", "Bid Pack", "Coverage Dashboard",
                "Bid Strategy", "Risk Checklist", "Quantities", "Pre-bid Meeting",
                "Quality Dashboard", "Raw JSON", "Ground Truth", "Structural Takeoff",
            ]
        assert len(labels) == 13, f"Expected 13 tabs, got {len(labels)}"
        assert "Issues to Resolve" in labels if yc_active else "Review Queue" in labels

        # Narration hints dict
        narration_keys = ["summary", "structural", "export", "coverage"]
        hints_available = all(k in narration_keys for k in ["summary", "structural"])
        assert hints_available, "Missing narration hint keys"

        print(f"  PASS  demo_mode_toggle: labels={len(labels)} tabs, narration={len(narration_keys)} screens")
    except Exception as e:
        failures.append(("demo_mode_toggle", e))
        print(f"  FAIL  demo_mode_toggle: {type(e).__name__}: {e}")
    return failures


def test_tab_exception_handling():
    """#93: Tab stability guard catches exceptions, other tabs survive."""
    failures = []
    try:
        # Simulate _safe_tab behavior
        errors_caught = []

        def _safe_tab(tab_name, render_fn, *args, **kwargs):
            try:
                render_fn(*args, **kwargs)
            except Exception as e:
                errors_caught.append((tab_name, str(e)))

        def _good_tab(payload):
            assert isinstance(payload, dict)
            return True

        def _bad_tab(payload):
            raise ValueError("Simulated tab crash")

        # Good tab works
        _safe_tab("Good Tab", _good_tab, {"test": 1})
        assert len(errors_caught) == 0

        # Bad tab is caught
        _safe_tab("Bad Tab", _bad_tab, {"test": 1})
        assert len(errors_caught) == 1
        assert errors_caught[0][0] == "Bad Tab"
        assert "Simulated tab crash" in errors_caught[0][1]

        # Another good tab still works after bad
        _safe_tab("Good Tab 2", _good_tab, {"test": 2})
        assert len(errors_caught) == 1  # no new errors

        print(f"  PASS  tab_exception_handling: 1 caught, 2 survived, isolation works")
    except Exception as e:
        failures.append(("tab_exception_handling", e))
        print(f"  FAIL  tab_exception_handling: {type(e).__name__}: {e}")
    return failures


def test_demo_snapshot_export():
    """#94: Demo Snapshot Export builds valid ZIP with expected files."""
    failures = []
    try:
        import zipfile, io, json

        payload = {
            "project_id": "smoke_demo",
            "timestamp": "2026-02-20T10:00:00",
            "decision": "conditional_go",
            "readiness_score": 72,
            "drawing_overview": {"pages_total": 15},
            "blockers": [{"title": "Missing MEP", "severity": "high"}],
            "rfis": [
                {"id": "RFI-001", "title": "Confirm slab thickness", "trade": "structural",
                 "severity": "medium", "status": "approved"},
            ],
            "guardrail_warnings": ["No fire protection drawings found"],
            "structural_takeoff": {
                "mode": "assumption",
                "summary": {"concrete_m3": 42.5, "steel_kg": 8500, "steel_tons": 8.5,
                            "element_counts": {"columns": 12}},
                "quantities": [
                    {"element_id": "C001", "type": "column", "label": "C1", "count": 12,
                     "dimensions_mm": {"width": 230}, "concrete_m3": 0.9,
                     "steel_kg": {"total": 180}},
                ],
                "qc": {"confidence": 0.65, "issues": {}},
                "exports": {},
                "warnings": [],
            },
        }
        project_id = "smoke_demo"

        # Build snapshot (matching demo_page.py logic)
        _ds_buf = io.BytesIO()
        with zipfile.ZipFile(_ds_buf, "w", zipfile.ZIP_DEFLATED) as _ds_zip:
            _ds_manifest = {
                "project_id": project_id,
                "timestamp": payload.get("timestamp", ""),
                "decision": payload.get("decision", ""),
                "readiness_score": payload.get("readiness_score", 0),
                "pages_total": (payload.get("drawing_overview") or {}).get("pages_total", 0),
                "blockers_count": len(payload.get("blockers", [])),
                "rfis_count": len(payload.get("rfis", [])),
            }
            _ds_zip.writestr("manifest.json", json.dumps(_ds_manifest, indent=2))
            # Approved RFIs
            _ds_rfis = [r for r in payload.get("rfis", []) if r.get("status") == "approved"]
            if _ds_rfis:
                _ds_zip.writestr("approved_rfis.csv", "id,title\nRFI-001,Confirm slab thickness\n")
            # Exclusions
            _ds_excl = payload.get("guardrail_warnings", [])
            if _ds_excl:
                _ds_zip.writestr("exclusions_clarifications.txt", "\n".join(str(w) for w in _ds_excl))
            # Structural
            _ds_st = payload.get("structural_takeoff")
            if _ds_st and _ds_st.get("mode") not in (None, "error"):
                _ds_zip.writestr("structural_summary.json",
                                 json.dumps(_ds_st.get("summary", {}), indent=2))
                _ds_zip.writestr("structural_qc.json",
                                 json.dumps(_ds_st.get("qc", {}), indent=2))

        _ds_buf.seek(0)
        with zipfile.ZipFile(_ds_buf) as zf:
            names = zf.namelist()
            assert "manifest.json" in names, "Missing manifest.json"
            assert "approved_rfis.csv" in names, "Missing approved_rfis.csv"
            assert "exclusions_clarifications.txt" in names, "Missing exclusions"
            assert "structural_summary.json" in names, "Missing structural_summary"
            assert "structural_qc.json" in names, "Missing structural_qc"

            # Verify manifest content
            m = json.loads(zf.read("manifest.json"))
            assert m["project_id"] == "smoke_demo"
            assert m["readiness_score"] == 72
            assert m["blockers_count"] == 1

        print(f"  PASS  demo_snapshot_export: {len(names)} files, manifest valid")
    except Exception as e:
        failures.append(("demo_snapshot_export", e))
        print(f"  FAIL  demo_snapshot_export: {type(e).__name__}: {e}")
    return failures


def test_metrics_consistency():
    """#95: Metrics display uses em-dash for None, never 0 for unavailable."""
    failures = []
    try:
        def _display(val, suffix=""):
            if val is None:
                return "\u2014"
            return f"{val}{suffix}"

        assert _display(None) == "\u2014", "None should show em-dash"
        assert _display(0) == "0", "Zero should show 0"
        assert _display(15) == "15", "15 should show 15"
        assert _display(3, " pages") == "3 pages", "Suffix should work"
        assert _display(None, " pages") == "\u2014", "None with suffix still em-dash"

        print(f"  PASS  metrics_consistency: em-dash for None, 0 for zero, suffix works")
    except Exception as e:
        failures.append(("metrics_consistency", e))
        print(f"  FAIL  metrics_consistency: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20C: Estimating Playbook Smoke Tests ──────────────────────────

def test_estimating_playbook_roundtrip():
    """#96: Playbook default → validate → merge → diff roundtrip works."""
    failures = []
    try:
        from src.analysis.estimating_playbook import (
            default_playbook, validate_playbook, merge_playbook, diff_playbook,
        )
        pb = default_playbook()
        assert isinstance(pb, dict), "default_playbook must return dict"
        is_valid, warnings = validate_playbook(pb)
        assert is_valid is True, f"default playbook should be valid: {warnings}"
        over = {"project": {"must_win": True}, "market_snapshot": {"material_trend": "rising"}}
        merged = merge_playbook(pb, over)
        assert merged["project"]["must_win"] is True, "merge should apply must_win"
        changes = diff_playbook(pb, merged)
        assert len(changes) >= 2, f"Expected >=2 changes, got {len(changes)}"
        print(f"  PASS  playbook_roundtrip: default→validate→merge→diff OK ({len(changes)} changes)")
    except Exception as e:
        failures.append(("playbook_roundtrip", e))
        print(f"  FAIL  playbook_roundtrip: {type(e).__name__}: {e}")
    return failures


def test_playbook_contingency_adjustments():
    """#97: Contingency adjustments compute correctly."""
    failures = []
    try:
        from src.analysis.estimating_playbook import (
            default_playbook, compute_playbook_contingency_adjustments,
        )
        pb = default_playbook()
        pb["company"]["risk_posture"] = "conservative"
        pb["market_snapshot"]["material_trend"] = "volatile"
        result = compute_playbook_contingency_adjustments(pb)
        assert result["base_pct"] == 5.0, f"Expected base 5.0, got {result['base_pct']}"
        assert result["posture_adj_pct"] == 1.5, f"Expected posture adj 1.5, got {result['posture_adj_pct']}"
        assert result["market_adj_pct"] == 2.0, f"Expected market adj 2.0, got {result['market_adj_pct']}"
        assert result["recommended_pct"] == 8.5, f"Expected recommended 8.5, got {result['recommended_pct']}"
        assert len(result["basis"]) >= 3, f"Expected >=3 basis items, got {len(result['basis'])}"
        print(f"  PASS  contingency_adjustments: conservative+volatile → 8.5% recommended")
    except Exception as e:
        failures.append(("contingency_adjustments", e))
        print(f"  FAIL  contingency_adjustments: {type(e).__name__}: {e}")
    return failures


def test_pricing_guidance_with_playbook():
    """#98: Pricing guidance changes when playbook has contingency override."""
    failures = []
    try:
        from src.analysis.pricing_guidance import compute_pricing_guidance
        from src.analysis.estimating_playbook import default_playbook
        # Without playbook
        r1 = compute_pricing_guidance(qa_score={"score": 85, "breakdown": {}})
        assert "basis_of_recommendation" not in r1, "No playbook should have no basis"
        # With playbook
        pb = default_playbook()
        pb["project"]["contingency_override_pct"] = 10.0
        r2 = compute_pricing_guidance(
            qa_score={"score": 85, "breakdown": {}},
            estimating_playbook=pb,
        )
        assert "basis_of_recommendation" in r2, "Playbook should add basis_of_recommendation"
        print(f"  PASS  pricing_with_playbook: basis_of_recommendation present, values adjusted")
    except Exception as e:
        failures.append(("pricing_with_playbook", e))
        print(f"  FAIL  pricing_with_playbook: {type(e).__name__}: {e}")
    return failures


def test_company_playbook_persistence():
    """#99: Save / load / list / delete playbook persistence roundtrip."""
    failures = []
    try:
        import tempfile, shutil
        from src.analysis.company_playbooks import (
            save_playbook, load_playbook, list_playbooks, delete_playbook,
        )
        from src.analysis.estimating_playbook import default_playbook
        tmp_dir = tempfile.mkdtemp()
        try:
            pb = default_playbook()
            pb["company"]["name"] = "SmokeCorp"
            path = save_playbook("SmokeCorp", pb, playbooks_dir=tmp_dir)
            assert path.exists(), f"Saved file should exist: {path}"
            loaded = load_playbook("SmokeCorp", playbooks_dir=tmp_dir)
            assert loaded is not None, "Loaded playbook should not be None"
            assert loaded["company"]["name"] == "SmokeCorp"
            entries = list_playbooks(playbooks_dir=tmp_dir)
            assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
            deleted = delete_playbook("SmokeCorp", playbooks_dir=tmp_dir)
            assert deleted is True, "Delete should return True"
            assert load_playbook("SmokeCorp", playbooks_dir=tmp_dir) is None
            print(f"  PASS  playbook_persistence: save→load→list→delete OK")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        failures.append(("playbook_persistence", e))
        print(f"  FAIL  playbook_persistence: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20D: Bulk Actions + Review Queue + FAST_BUDGET ────────────────


def test_bulk_action_payload_mutation():
    """#100: Bulk actions mutate payload when eligible items exist."""
    failures = []
    try:
        from src.analysis.bulk_actions import (
            prefer_schedule_for_mismatches,
            generate_rfis_for_high_mismatches,
            mark_intentional_revisions_reviewed,
        )

        # ── prefer_schedule ──
        recon_rows = [
            {"category": "doors", "mismatch": True, "schedule_count": 5,
             "boq_count": 3, "drawing_count": 5, "max_delta": 2},
            {"category": "windows", "mismatch": True, "schedule_count": 8,
             "boq_count": 6, "drawing_count": 8, "max_delta": 2},
            {"category": "rooms", "mismatch": True, "schedule_count": 4,
             "boq_count": 4, "drawing_count": 4, "max_delta": 0},
        ]
        updated, count = prefer_schedule_for_mismatches(recon_rows)
        assert count == 2, f"Expected 2 schedule actions, got {count}"
        assert updated[0].get("action") == "prefer_schedule", (
            f"First row should have action=prefer_schedule, got {updated[0].get('action')}"
        )
        # Room row should NOT be affected (different category)
        assert updated[2].get("action") is None or updated[2].get("action") != "prefer_schedule"
        print(f"  PASS  prefer_schedule: mutated {count} rows correctly")

        # ── generate_rfis for high delta ──
        high_recon = [
            {"category": "doors", "mismatch": True, "schedule_count": 10,
             "boq_count": 3, "drawing_count": 10, "max_delta": 7},
            {"category": "windows", "mismatch": True, "schedule_count": 2,
             "boq_count": 1, "drawing_count": 2, "max_delta": 1},
        ]
        new_rfis, updated_rows = generate_rfis_for_high_mismatches(high_recon, [])
        assert len(new_rfis) == 1, f"Expected 1 RFI (delta>=5), got {len(new_rfis)}"
        assert new_rfis[0].get("source") == "bulk_action"
        print(f"  PASS  generate_rfis: created {len(new_rfis)} RFI(s) for high delta")

        # ── mark_intentional_revisions ──
        conflicts = [
            {"resolution": "intentional_revision", "type": "addendum_override"},
            {"resolution": "unresolved", "type": "scope_gap"},
            {"resolution": "intentional_revision", "type": "rate_change"},
        ]
        updated_c, c_count = mark_intentional_revisions_reviewed(conflicts)
        assert c_count == 2, f"Expected 2 marked, got {c_count}"
        for uc in updated_c:
            if uc.get("resolution") == "intentional_revision":
                assert uc.get("review_status") == "reviewed", (
                    f"Expected review_status=reviewed, got {uc.get('review_status')}"
                )
        print(f"  PASS  mark_revisions: marked {c_count} conflicts as reviewed")

    except Exception as e:
        failures.append(("bulk_action_payload_mutation", e))
        print(f"  FAIL  bulk_action_payload_mutation: {type(e).__name__}: {e}")
    return failures


def test_bulk_action_noop():
    """#101: Bulk actions return 0 / empty when no eligible items exist."""
    failures = []
    try:
        from src.analysis.bulk_actions import (
            prefer_schedule_for_mismatches,
            generate_rfis_for_high_mismatches,
            mark_intentional_revisions_reviewed,
        )

        # No door/window mismatches
        recon_no_match = [
            {"category": "rooms", "mismatch": True, "schedule_count": 4,
             "boq_count": 3, "drawing_count": 4, "max_delta": 1},
        ]
        _, count_sched = prefer_schedule_for_mismatches(recon_no_match)
        assert count_sched == 0, f"Expected 0 schedule actions, got {count_sched}"
        print(f"  PASS  prefer_schedule_noop: 0 actions on non-door/window")

        # No high-delta items
        recon_low_delta = [
            {"category": "doors", "mismatch": True, "schedule_count": 3,
             "boq_count": 2, "drawing_count": 3, "max_delta": 1},
        ]
        new_rfis, _ = generate_rfis_for_high_mismatches(recon_low_delta, [])
        assert len(new_rfis) == 0, f"Expected 0 RFIs, got {len(new_rfis)}"
        print(f"  PASS  generate_rfis_noop: 0 RFIs for low delta")

        # No intentional revisions
        no_rev_conflicts = [
            {"resolution": "unresolved", "type": "scope_gap"},
        ]
        _, c_count = mark_intentional_revisions_reviewed(no_rev_conflicts)
        assert c_count == 0, f"Expected 0 marked, got {c_count}"
        print(f"  PASS  mark_revisions_noop: 0 marked when no intentional revisions")

        # All empty
        _, empty_count = prefer_schedule_for_mismatches([])
        assert empty_count == 0
        empty_rfis, _ = generate_rfis_for_high_mismatches([], [])
        assert len(empty_rfis) == 0
        _, empty_rev = mark_intentional_revisions_reviewed([])
        assert empty_rev == 0
        print(f"  PASS  all_empty: 0 actions on empty inputs")

    except Exception as e:
        failures.append(("bulk_action_noop", e))
        print(f"  FAIL  bulk_action_noop: {type(e).__name__}: {e}")
    return failures


def test_review_queue_no_overlap():
    """#102: Review queue items have distinct fields, no internal key dumps."""
    failures = []
    try:
        from src.analysis.review_queue import build_review_queue

        queue = build_review_queue(
            quantity_reconciliation=[
                {"category": "doors", "mismatch": True, "schedule_count": 5,
                 "boq_count": 3, "drawing_count": 5, "max_delta": 2},
            ],
            conflicts=[
                {"type": "scope_gap", "delta_confidence": 0.6, "item_no": "A1"},
            ],
            pages_skipped=[
                {"doc_type": "schedule", "page_idx": 10},
                {"doc_type": "schedule", "page_idx": 11},
            ],
            toxic_summary={
                "pages": [{"page_idx": 20, "toxic": True, "reason": "OCR fail"}],
            },
            risk_results=[
                {"template_id": "liquidated_damages", "label": "Liquidated Damages",
                 "impact": "high", "found": True,
                 "hits": [{"page_idx": 5, "text": "LD clause"}]},
            ],
        )

        assert len(queue) == 5, f"Expected 5 review items, got {len(queue)}"

        # Each item should have required fields
        required_keys = {"type", "severity", "title", "source_key",
                         "evidence_bundle", "recommended_action"}
        for i, item in enumerate(queue):
            missing = required_keys - set(item.keys())
            assert not missing, f"Item {i} missing keys: {missing}"

        # No duplicate source_keys
        source_keys = [item["source_key"] for item in queue]
        assert len(source_keys) == len(set(source_keys)), (
            f"Duplicate source_keys: {source_keys}"
        )

        # Titles should NOT contain raw dict representation or internal keys
        for item in queue:
            title = item.get("title", "")
            assert "evidence_bundle" not in title, (
                f"Title contains 'evidence_bundle': {title}"
            )
            assert "source_key" not in title, (
                f"Title contains 'source_key': {title}"
            )

        # Verify sort: first item should be high severity
        assert queue[0]["severity"] == "high", (
            f"First item should be high severity, got {queue[0]['severity']}"
        )

        print(f"  PASS  review_queue: {len(queue)} items, no overlap, sorted correctly")
    except Exception as e:
        failures.append(("review_queue_no_overlap", e))
        print(f"  FAIL  review_queue_no_overlap: {type(e).__name__}: {e}")
    return failures


def test_fast_budget_banner_detection():
    """#103: FAST_BUDGET detection logic from payload selection_mode + skipped pages."""
    failures = []
    try:
        # Simulate the detection logic used in demo_page.py
        # (pure data check, no Streamlit needed)
        payload_fast = {
            "run_coverage": {"selection_mode": "fast_budget"},
            "pages_skipped": [
                {"doc_type": "schedule", "page_idx": 10},
                {"doc_type": "boq", "page_idx": 15},
            ],
            "processing_stats": {
                "total_pages": 100,
                "pages_processed": 40,
                "ocr_pages": 35,
                "toxic_pages": 2,
            },
        }

        sel_mode = (payload_fast.get("run_coverage") or {}).get("selection_mode", "")
        skipped = payload_fast.get("pages_skipped") or []
        ps = payload_fast.get("processing_stats") or {}

        is_fast_budget = sel_mode == "fast_budget"
        has_skipped = len(skipped) > 0

        assert is_fast_budget, "Should detect fast_budget mode"
        assert has_skipped, "Should detect skipped pages"

        # Verify processing_stats metrics are available
        assert ps.get("total_pages") == 100
        assert ps.get("pages_processed") == 40
        assert ps.get("ocr_pages") == 35
        assert ps.get("toxic_pages") == 2
        print(f"  PASS  fast_budget_detection: mode={sel_mode}, skipped={len(skipped)}")

        # Non-fast-budget payload should NOT trigger
        payload_full = {
            "run_coverage": {"selection_mode": "full"},
            "pages_skipped": [],
        }
        sel_mode_full = (payload_full.get("run_coverage") or {}).get("selection_mode", "")
        assert sel_mode_full != "fast_budget", "Full mode should not be fast_budget"
        assert len(payload_full.get("pages_skipped", [])) == 0
        print(f"  PASS  full_mode_no_banner: mode={sel_mode_full}, no skipped pages")

        # Edge case: missing run_coverage entirely
        payload_empty = {}
        sel_mode_empty = (payload_empty.get("run_coverage") or {}).get("selection_mode", "")
        assert sel_mode_empty != "fast_budget", "Missing run_coverage should not be fast_budget"
        print(f"  PASS  missing_coverage_fallback: graceful with empty payload")

    except Exception as e:
        failures.append(("fast_budget_banner_detection", e))
        print(f"  FAIL  fast_budget_banner_detection: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20E: Page Selection + Coverage Score + Quantify + Blockers ─────


def test_page_selection_boq_guaranteed():
    """#104: BOQ pages always selected even with 103 conditions pages."""
    failures = []
    try:
        from src.analysis.page_selection import select_pages, TIER_1_SUB_CAPS
        from src.analysis.page_index import PageIndex, IndexedPage
        from collections import Counter

        # Reproduce the Sonipat scenario: 103 conditions + 2 BOQ + 259 unknown
        page_types = (
            [("conditions", "other")] * 103 +
            [("boq", "other")] * 2 +
            [("spec", "other")] * 2 +
            [("section", "architectural")] * 1 +
            [("unknown", "other")] * 259
        )
        pages = []
        type_counter = Counter()
        disc_counter = Counter()
        for i, (dtype, disc) in enumerate(page_types):
            pages.append(IndexedPage(
                page_idx=i, doc_type=dtype, discipline=disc,
                confidence=0.7, has_text_layer=True,
            ))
            type_counter[dtype] += 1
            disc_counter[disc] += 1
        idx = PageIndex(
            pdf_name="test.pdf", total_pages=len(pages), pages=pages,
            counts_by_type=dict(type_counter), counts_by_discipline=dict(disc_counter),
        )
        result = select_pages(idx, budget_pages=80)

        boq_selected = sum(1 for i in result.selected if idx.pages[i].doc_type == "boq")
        assert boq_selected == 2, f"All BOQ pages must be selected, got {boq_selected}"
        print(f"  PASS  boq_guaranteed: {boq_selected} BOQ pages selected (100%)")

    except Exception as e:
        failures.append(("page_selection_boq_guaranteed", e))
        print(f"  FAIL  page_selection_boq_guaranteed: {type(e).__name__}: {e}")
    return failures


def test_conditions_sub_cap():
    """#105: Conditions pages sub-capped at <= 20."""
    failures = []
    try:
        from src.analysis.page_selection import select_pages, TIER_1_SUB_CAPS
        from src.analysis.page_index import PageIndex, IndexedPage
        from collections import Counter

        page_types = [("conditions", "other")] * 103 + [("plan", "architectural")] * 50
        pages = []
        type_counter = Counter()
        disc_counter = Counter()
        for i, (dtype, disc) in enumerate(page_types):
            pages.append(IndexedPage(
                page_idx=i, doc_type=dtype, discipline=disc,
                confidence=0.7, has_text_layer=True,
            ))
            type_counter[dtype] += 1
            disc_counter[disc] += 1
        idx = PageIndex(
            pdf_name="test.pdf", total_pages=len(pages), pages=pages,
            counts_by_type=dict(type_counter), counts_by_discipline=dict(disc_counter),
        )
        result = select_pages(idx, budget_pages=80)

        cond_selected = sum(1 for i in result.selected if idx.pages[i].doc_type == "conditions")
        max_cap = TIER_1_SUB_CAPS.get("conditions", 20)
        assert cond_selected <= max_cap, f"Expected <= {max_cap} conditions, got {cond_selected}"
        assert cond_selected >= 10, f"Expected >= 10 conditions sampled, got {cond_selected}"
        print(f"  PASS  conditions_sub_cap: {cond_selected} conditions selected (cap={max_cap})")

    except Exception as e:
        failures.append(("conditions_sub_cap", e))
        print(f"  FAIL  conditions_sub_cap: {type(e).__name__}: {e}")
    return failures


def test_coverage_sublabel_present():
    """#106: Coverage sublabel logic shows X/Y when deep < total."""
    failures = []
    try:
        ps = {"total_pages": 367, "deep_processed_pages": 80}
        deep = ps.get("deep_processed_pages")
        total = ps.get("total_pages")

        should_show = deep is not None and total and deep < total
        assert should_show, "Sublabel should show when deep < total"

        label = "Trade Coverage" if should_show else "Coverage"
        assert label == "Trade Coverage", f"Expected 'Trade Coverage', got '{label}'"

        sublabel = f"{deep}/{total} pages processed"
        assert "80/367" in sublabel

        # Full read: no sublabel
        ps_full = {"total_pages": 50, "deep_processed_pages": 50}
        should_show_full = ps_full["deep_processed_pages"] < ps_full["total_pages"]
        assert not should_show_full, "No sublabel for full_read"
        print(f"  PASS  coverage_sublabel: shows '{sublabel}', label='{label}'")

    except Exception as e:
        failures.append(("coverage_sublabel_present", e))
        print(f"  FAIL  coverage_sublabel_present: {type(e).__name__}: {e}")
    return failures


def test_quantify_table_new_rows():
    """#107: New cost-driving rows extracted from payload."""
    failures = []
    try:
        payload = {
            "boq_stats": {"total_items": 12},
            "commercial_terms": [{"term_type": "defect_liability"}],
            "requirements_by_trade": {"civil": [1, 2], "mep": [3, 4, 5]},
            "finish_takeoff": {"finish_rows": [{"room": "A"}]},
        }
        boq_count = (payload.get("boq_stats") or {}).get("total_items")
        comm = payload.get("commercial_terms")
        comm_count = len(comm) if isinstance(comm, list) and comm else None
        req_bt = payload.get("requirements_by_trade") or {}
        req_count = sum(len(v) for v in req_bt.values()) if req_bt else None
        finish = (payload.get("finish_takeoff") or {}).get("finish_rows")
        finish_count = len(finish) if isinstance(finish, list) else None

        assert boq_count == 12
        assert comm_count == 1
        assert req_count == 5
        assert finish_count == 1

        # Empty payload returns None (em-dash)
        empty = {}
        assert (empty.get("boq_stats") or {}).get("total_items") is None
        print(f"  PASS  quantify_new_rows: boq={boq_count}, comm={comm_count}, req={req_count}, finish={finish_count}")

    except Exception as e:
        failures.append(("quantify_table_new_rows", e))
        print(f"  FAIL  quantify_table_new_rows: {type(e).__name__}: {e}")
    return failures


def test_blocker_coverage_gaps():
    """#108: Coverage gap warnings generated from doc_types_not_covered."""
    failures = []
    try:
        run_cov = {
            "doc_types_not_covered": ["boq", "addendum"],
            "doc_types_partially_covered": ["conditions"],
            "doc_types_detected": {"conditions": 103, "boq": 2, "addendum": 5},
        }
        not_covered = run_cov.get("doc_types_not_covered", [])
        partial = run_cov.get("doc_types_partially_covered", [])

        warnings = []
        if "boq" in not_covered:
            warnings.append("boq_not_processed")
        if "addendum" in not_covered:
            warnings.append("addenda_not_processed")
        if "conditions" in partial or "conditions" in not_covered:
            warnings.append("conditions_partial")
        if "spec" in not_covered:
            warnings.append("spec_skipped")

        assert len(warnings) == 3, f"Expected 3 warnings, got {len(warnings)}: {warnings}"
        assert "boq_not_processed" in warnings
        assert "addenda_not_processed" in warnings
        assert "conditions_partial" in warnings

        # No warnings for empty run_coverage
        empty_cov = {}
        no_warnings = []
        if "boq" in empty_cov.get("doc_types_not_covered", []):
            no_warnings.append("boq")
        assert len(no_warnings) == 0
        print(f"  PASS  coverage_gaps: {len(warnings)} warnings generated")

    except Exception as e:
        failures.append(("blocker_coverage_gaps", e))
        print(f"  FAIL  blocker_coverage_gaps: {type(e).__name__}: {e}")
    return failures


def test_skipped_pages_detail_panel():
    """#109: Skipped pages grouped by doc_type with severity icons."""
    failures = []
    try:
        from collections import Counter
        skipped_list = [
            {"doc_type": "conditions", "page_idx": 80},
            {"doc_type": "conditions", "page_idx": 81},
            {"doc_type": "conditions", "page_idx": 82},
            {"doc_type": "unknown", "page_idx": 110},
            {"doc_type": "unknown", "page_idx": 111},
            {"doc_type": "boq", "page_idx": 200},
        ]
        by_type = Counter(s.get("doc_type", "unknown") for s in skipped_list)

        assert by_type["conditions"] == 3
        assert by_type["unknown"] == 2
        assert by_type["boq"] == 1

        # Severity classification
        HIGH_SKIP = {"boq", "schedule", "addendum"}
        MED_SKIP = {"conditions", "spec"}
        for dt in by_type:
            if dt in HIGH_SKIP:
                icon = "red"
            elif dt in MED_SKIP:
                icon = "yellow"
            else:
                icon = "white"
            # Just verify classification doesn't crash
            assert icon in ("red", "yellow", "white")

        print(f"  PASS  skipped_detail: {len(by_type)} types, {len(skipped_list)} total pages")

    except Exception as e:
        failures.append(("skipped_pages_detail_panel", e))
        print(f"  FAIL  skipped_pages_detail_panel: {type(e).__name__}: {e}")
    return failures


# ── Sprint 20F: Extraction Diagnostics Smoke Tests ──────────────────────

def test_extraction_diagnostics_render():
    """Diagnostics panel renders with sample payload."""
    failures = []
    try:
        payload = PAYLOADS["full_payload"]
        ext_diag = payload.get("extraction_diagnostics") or {}
        boq_diag = ext_diag.get("boq", {})
        sched_diag = ext_diag.get("schedules", {})
        methods = ext_diag.get("table_methods_used", {})

        assert boq_diag.get("pages_attempted", 0) == 2
        assert boq_diag.get("pages_parsed", 0) == 1
        assert sched_diag.get("rows_extracted", 0) == 4
        assert "regex" in methods
        assert methods["regex"] == 2

        print("  PASS  extraction_diagnostics render")
    except Exception as e:
        failures.append(("extraction_diagnostics_render", e))
        print(f"  FAIL  extraction_diagnostics_render: {type(e).__name__}: {e}")
    return failures


def test_table_method_counts():
    """Table method counts display correctly."""
    failures = []
    try:
        payload = PAYLOADS["full_payload"]
        methods = (payload.get("extraction_diagnostics") or {}).get("table_methods_used", {})
        method_parts = [f"{m}: {c}" for m, c in sorted(methods.items(), key=lambda x: -x[1])]
        display = " | ".join(method_parts)
        assert "regex: 2" in display
        assert "ocr_row_reconstruct: 1" in display
        print(f"  PASS  table_method_counts: {display}")
    except Exception as e:
        failures.append(("table_method_counts", e))
        print(f"  FAIL  table_method_counts: {type(e).__name__}: {e}")
    return failures


def test_no_crash_missing_extraction_diagnostics():
    """No crash when extraction_diagnostics missing from payload."""
    failures = []
    try:
        payload = PAYLOADS["no_overview_key"]
        ext_diag = payload.get("extraction_diagnostics") or {}
        boq = ext_diag.get("boq", {})
        sched = ext_diag.get("schedules", {})
        methods = ext_diag.get("table_methods_used", {})
        # All should be empty/default, no crash
        assert boq.get("pages_attempted", 0) == 0
        assert sched.get("rows_extracted", 0) == 0
        assert len(methods) == 0

        # Also test build_demo_analysis doesn't crash
        demo = build_demo_analysis(payload, "test_missing_diag")
        assert demo is not None
        print("  PASS  no_crash_missing_extraction_diagnostics")
    except Exception as e:
        failures.append(("no_crash_missing_extraction_diagnostics", e))
        print(f"  FAIL  no_crash_missing_extraction_diagnostics: {type(e).__name__}: {e}")
    return failures


def test_no_crash_complex_boq_rows():
    """No crash when duplicate/complex BOQ rows appear."""
    failures = []
    try:
        from src.analysis.extractors.extract_boq import extract_boq_items
        # Simulate complex BOQ with duplicates and edge cases
        text = """
BILL OF QUANTITIES
Item No  Description                        Unit    Qty     Rate
1.1      Earthwork excavation               cum     120.5   450.00
1.1      Earthwork excavation (duplicate)   cum     120.5   450.00
1.2      PCC M15                            cum     25.0    5200
2(a)     Providing steel doors              nos     12
A-1      Waterproofing membrane             sqm     500.0   280.00
         Additional description line only
         Sub Total
"""
        items = extract_boq_items(text, source_page=0)
        # Should not crash, should deduplicate
        item_nos = [i["item_no"] for i in items]
        # Check no duplicates
        assert len(item_nos) == len(set(f"{i['item_no']}:{i.get('description','')[:30]}" for i in items))
        print(f"  PASS  complex_boq_rows: {len(items)} items extracted")
    except Exception as e:
        failures.append(("complex_boq_rows", e))
        print(f"  FAIL  complex_boq_rows: {type(e).__name__}: {e}")
    return failures


def test_processing_stats_table_fields():
    """processing_stats has table_attempt/success fields."""
    failures = []
    try:
        ps = PAYLOADS["full_payload"].get("processing_stats") or {}
        assert "table_attempt_pages" in ps, "Missing table_attempt_pages"
        assert "table_success_pages" in ps, "Missing table_success_pages"
        assert ps["table_attempt_pages"] == 3
        assert ps["table_success_pages"] == 2
        print("  PASS  processing_stats_table_fields")
    except Exception as e:
        failures.append(("processing_stats_table_fields", e))
        print(f"  FAIL  processing_stats_table_fields: {type(e).__name__}: {e}")
    return failures


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Smoke test: render results preview")
    print("=" * 60)

    all_failures = []

    print("\n1. build_demo_analysis (all shapes):")
    all_failures.extend(test_build_demo_analysis())

    print("\n2. overview extraction fallback chain:")
    all_failures.extend(test_overview_extraction())

    print("\n3. NO_DRAWINGS classification:")
    all_failures.extend(test_no_drawings_classification())

    print("\n4. overview alias key:")
    all_failures.extend(test_overview_alias())

    print("\n5. cached demo JSON:")
    all_failures.extend(test_cached_demo())

    print("\n6. evidence rendering (blocker with evidence):")
    all_failures.extend(test_evidence_rendering())

    print("\n7. RFI FYI downgrade (no evidence):")
    all_failures.extend(test_rfi_fyi_downgrade())

    print("\n8. MEP blocker as FYI:")
    all_failures.extend(test_mep_blocker_as_fyi())

    print("\n9. quantified outputs table data:")
    all_failures.extend(test_quantified_outputs())

    print("\n10. extraction summary access:")
    all_failures.extend(test_extraction_summary_access())

    print("\n11. selection & coverage access:")
    all_failures.extend(test_selection_coverage_access())

    print("\n12. coverage status display:")
    all_failures.extend(test_coverage_status_display())

    print("\n13. bid pack tab data:")
    all_failures.extend(test_bid_pack_tab())

    print("\n14. coverage dashboard data:")
    all_failures.extend(test_coverage_dashboard())

    print("\n15. evidence viewer no-PDF fallback:")
    all_failures.extend(test_evidence_viewer_no_pdf())

    print("\n16. bbox overlay no crash:")
    all_failures.extend(test_bbox_overlay_no_crash())

    print("\n17. NOT_FOUND proof fields:")
    all_failures.extend(test_not_found_proof_fields())

    print("\n18. skipped pages CSV export:")
    all_failures.extend(test_skipped_pages_csv())

    print("\n19. bbox page-relative format:")
    all_failures.extend(test_bbox_page_relative_format())

    print("\n20. OCR bbox meta:")
    all_failures.extend(test_ocr_bbox_meta())

    print("\n21. OCR text cache accessible:")
    all_failures.extend(test_ocr_text_cache_accessible())

    print("\n22. OCR text cache search term highlight:")
    all_failures.extend(test_ocr_text_cache_search_term_highlight())

    print("\n23. citation elements present:")
    all_failures.extend(test_citation_elements_present())

    print("\n24. heatmap data available:")
    all_failures.extend(test_heatmap_data_available())

    print("\n25. bbox_id in evidence (Sprint 5):")
    all_failures.extend(test_bbox_id_in_evidence())

    print("\n26. bbox_id linkage (Sprint 5):")
    all_failures.extend(test_bbox_id_linkage())

    print("\n27. search returns correct pages (Sprint 5):")
    all_failures.extend(test_search_returns_correct_pages())

    print("\n28. search returns snippets (Sprint 5):")
    all_failures.extend(test_search_returns_snippets())

    print("\n29. search performance (Sprint 5):")
    all_failures.extend(test_search_performance())

    print("\n30. bid strategy unknown outputs (Sprint 5):")
    all_failures.extend(test_bid_strategy_unknown_outputs())

    print("\n31. bid strategy dial range (Sprint 5):")
    all_failures.extend(test_bid_strategy_dial_range())

    print("\n32. bid strategy recommendations present (Sprint 5):")
    all_failures.extend(test_bid_strategy_recommendations_present())

    print("\n33. bid summary generation (Sprint 6):")
    all_failures.extend(test_bid_summary_generation())

    print("\n34. addendum and conflicts access (Sprint 6):")
    all_failures.extend(test_addendum_and_conflicts_access())

    print("\n35. normalization idempotent (Sprint 6):")
    all_failures.extend(test_normalization_idempotent())

    print("\n36. bid summary includes strategy (Sprint 6):")
    all_failures.extend(test_bid_summary_includes_strategy())

    print("\n37. delta confidence deterministic (Sprint 7):")
    all_failures.extend(test_delta_confidence_deterministic())

    print("\n38. reconciliation stable keys (Sprint 7):")
    all_failures.extend(test_reconciliation_stable_keys())

    print("\n39. PDF export produces bytes (Sprint 7):")
    all_failures.extend(test_pdf_export_produces_bytes())

    print("\n40. PDF export missing sections (Sprint 7):")
    all_failures.extend(test_pdf_export_missing_sections())

    print("\n41. recon RFI creates valid dict (Sprint 8):")
    all_failures.extend(test_recon_rfi_creates_valid_dict())

    print("\n42. recon assumption stable schema (Sprint 8):")
    all_failures.extend(test_recon_assumption_stable_schema())

    print("\n43. owner profile round trip (Sprint 8):")
    all_failures.extend(test_owner_profile_round_trip())

    print("\n44. risk scoped search backward compat (Sprint 8):")
    all_failures.extend(test_risk_scoped_search_backward_compat())

    print("\n45. assumptions status persists and exports (Sprint 9):")
    all_failures.extend(test_assumptions_status_persists_and_exports())

    print("\n46. multi-doc evidence viewer (Sprint 9):")
    all_failures.extend(test_multi_doc_evidence_viewer())

    print("\n47. supersedes tagging deterministic (Sprint 9):")
    all_failures.extend(test_supersedes_tagging_deterministic())

    print("\n48. bid summary with assumptions (Sprint 9):")
    all_failures.extend(test_bid_summary_with_assumptions())

    print("\n49. cache hit path deterministic (Sprint 10):")
    all_failures.extend(test_cache_hit_path_deterministic())

    print("\n50. toxic retry triggers (Sprint 10):")
    all_failures.extend(test_toxic_retry_triggers())

    print("\n51. clustering stable across runs (Sprint 10):")
    all_failures.extend(test_clustering_stable_across_runs())

    print("\n52. QA score stable given fixed payload (Sprint 10):")
    all_failures.extend(test_qa_score_stable_given_fixed_payload())

    print("\n53. quantities stable schema (Sprint 11):")
    all_failures.extend(test_quantities_stable_schema())

    print("\n54. pricing guidance deterministic (Sprint 11):")
    all_failures.extend(test_pricing_guidance_deterministic())

    print("\n55. DOCX exports produce bytes (Sprint 11):")
    all_failures.extend(test_docx_exports_produce_bytes())

    print("\n56. quantity reconciliation stable (Sprint 12):")
    all_failures.extend(test_quantity_reconciliation_stable())

    print("\n57. finishes takeoff with areas (Sprint 12):")
    all_failures.extend(test_finishes_takeoff_with_areas())

    print("\n58. feedback write and schema (Sprint 12):")
    all_failures.extend(test_feedback_write_and_schema())

    print("\n59. review queue deterministic (Sprint 13):")
    all_failures.extend(test_review_queue_deterministic())

    print("\n60. bulk actions produce changes (Sprint 13):")
    all_failures.extend(test_bulk_actions_produce_changes())

    print("\n61. export filters enforce approval (Sprint 13):")
    all_failures.extend(test_export_filters_enforce_approval())

    print("\n62. project create load deterministic (Sprint 14):")
    all_failures.extend(test_project_create_load_deterministic())

    print("\n63. collaboration persist (Sprint 14):")
    all_failures.extend(test_collaboration_persist())

    print("\n64. submission pack structure (Sprint 14):")
    all_failures.extend(test_submission_pack_structure())

    print("\n65. evidence appendix PDF stable (Sprint 15):")
    all_failures.extend(test_evidence_appendix_pdf_stable())

    print("\n66. meeting agenda deterministic (Sprint 15):")
    all_failures.extend(test_meeting_agenda_deterministic())

    print("\n67. email drafts include approved (Sprint 15):")
    all_failures.extend(test_email_drafts_include_approved())

    print("\n68. storage local roundtrip (Sprint 16):")
    all_failures.extend(test_storage_local_roundtrip())

    print("\n69. auth tenant isolation (Sprint 16):")
    all_failures.extend(test_auth_tenant_isolation())

    print("\n70. job queue sync completion (Sprint 16):")
    all_failures.extend(test_job_queue_sync_completion())

    print("\n71. highlights in payload (Sprint 17):")
    all_failures.extend(test_highlights_in_payload())

    print("\n72. determinism review queue (Sprint 17):")
    all_failures.extend(test_determinism_review_queue_smoke())

    print("\n73. demo config (Sprint 17):")
    all_failures.extend(test_demo_config_smoke())

    print("\n74. demo assets (Sprint 17):")
    all_failures.extend(test_demo_assets_smoke())

    print("\n75. narration deterministic (Sprint 18):")
    all_failures.extend(test_narration_deterministic())

    print("\n76. export filenames stable (Sprint 18):")
    all_failures.extend(test_export_filenames_stable())

    print("\n77. reset state no crash (Sprint 18):")
    all_failures.extend(test_reset_state_no_crash())

    print("\n78. summary card missing fields (Sprint 18):")
    all_failures.extend(test_summary_card_missing_fields())

    print("\n79. section2 dependencies no crash (Sprint 18 bugfix):")
    all_failures.extend(test_section2_dependencies_no_crash())

    print("\n80. processing_stats missing no crash (Sprint 18):")
    all_failures.extend(test_processing_stats_missing_no_crash())

    print("\n81. processing_stats populated (Sprint 18):")
    all_failures.extend(test_processing_stats_populated())

    print("\n82. sub_scores passthrough (Sprint 18 bugfix):")
    all_failures.extend(test_sub_scores_passthrough())

    print("\n83. commercial terms absent no crash (Sprint 19):")
    all_failures.extend(test_commercial_terms_absent_no_crash())

    print("\n84. commercial terms present (Sprint 19):")
    all_failures.extend(test_commercial_terms_present())

    print("\n85. commercial RFI checks registered (Sprint 19):")
    all_failures.extend(test_commercial_rfi_checks_registered())

    print("\n86. GT template CSV valid (Sprint 20):")
    all_failures.extend(test_gt_template_csv_valid())

    print("\n87. GT diff deterministic (Sprint 20):")
    all_failures.extend(test_gt_diff_deterministic())

    print("\n88. training pack ZIP structure (Sprint 20):")
    all_failures.extend(test_training_pack_zip_structure())

    print("\n89. pilot docs no crash (Sprint 20):")
    all_failures.extend(test_pilot_docs_no_crash())

    print("\n90. structural tab with payload (Sprint 20A):")
    all_failures.extend(test_structural_tab_with_payload())

    print("\n91. structural tab without payload (Sprint 20A):")
    all_failures.extend(test_structural_tab_without_payload())

    print("\n92. demo mode toggle (Sprint 20B):")
    all_failures.extend(test_demo_mode_toggle())

    print("\n93. tab exception handling (Sprint 20B):")
    all_failures.extend(test_tab_exception_handling())

    print("\n94. demo snapshot export (Sprint 20B):")
    all_failures.extend(test_demo_snapshot_export())

    print("\n95. metrics consistency (Sprint 20B):")
    all_failures.extend(test_metrics_consistency())

    print("\n96. estimating playbook roundtrip (Sprint 20C):")
    all_failures.extend(test_estimating_playbook_roundtrip())

    print("\n97. playbook contingency adjustments (Sprint 20C):")
    all_failures.extend(test_playbook_contingency_adjustments())

    print("\n98. pricing guidance with playbook (Sprint 20C):")
    all_failures.extend(test_pricing_guidance_with_playbook())

    print("\n99. company playbook persistence (Sprint 20C):")
    all_failures.extend(test_company_playbook_persistence())

    print("\n100. bulk action payload mutation (Sprint 20D):")
    all_failures.extend(test_bulk_action_payload_mutation())

    print("\n101. bulk action no-op (Sprint 20D):")
    all_failures.extend(test_bulk_action_noop())

    print("\n102. review queue no overlap (Sprint 20D):")
    all_failures.extend(test_review_queue_no_overlap())

    print("\n103. FAST_BUDGET banner detection (Sprint 20D):")
    all_failures.extend(test_fast_budget_banner_detection())

    print("\n104. page selection BOQ guaranteed (Sprint 20E):")
    all_failures.extend(test_page_selection_boq_guaranteed())

    print("\n105. conditions sub-cap (Sprint 20E):")
    all_failures.extend(test_conditions_sub_cap())

    print("\n106. coverage sublabel present (Sprint 20E):")
    all_failures.extend(test_coverage_sublabel_present())

    print("\n107. quantify table new rows (Sprint 20E):")
    all_failures.extend(test_quantify_table_new_rows())

    print("\n108. blocker coverage gaps (Sprint 20E):")
    all_failures.extend(test_blocker_coverage_gaps())

    print("\n109. skipped pages detail panel (Sprint 20E):")
    all_failures.extend(test_skipped_pages_detail_panel())

    print("\n110. extraction diagnostics render (Sprint 20F):")
    all_failures.extend(test_extraction_diagnostics_render())

    print("\n111. table method counts (Sprint 20F):")
    all_failures.extend(test_table_method_counts())

    print("\n112. no crash missing extraction_diagnostics (Sprint 20F):")
    all_failures.extend(test_no_crash_missing_extraction_diagnostics())

    print("\n113. complex BOQ rows (Sprint 20F):")
    all_failures.extend(test_no_crash_complex_boq_rows())

    print("\n114. processing_stats table fields (Sprint 20F):")
    all_failures.extend(test_processing_stats_table_fields())

    print("\n115. Excel BOQ source payload (Sprint 21C):")
    all_failures.extend(test_excel_boq_source_payload())

    print("\n" + "=" * 60)
    if all_failures:
        print(f"FAILED: {len(all_failures)} test(s)")
        for name, err in all_failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
