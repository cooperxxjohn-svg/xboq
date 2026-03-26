"""
QTO — Quantity Takeoff sub-package.

Standard module interface
--------------------------
Every takeoff module exposes ONE primary entry-point function named
``run_<trade>_takeoff()`` that returns a typed Result dataclass.

The Result dataclass always has at minimum:
    line_items:  List[dict]   — BOQ line items with trade/description/qty/unit
    mode:        str          — how quantities were derived ("schedule"|"computed"|etc.)
    warnings:    List[str]    — non-fatal issues the estimator should review

Modules
-------
| Module                      | Entry point                   | Result type            |
|-----------------------------|-------------------------------|------------------------|
| brickwork_takeoff           | run_brickwork_takeoff()       | BrickworkResult        |
| door_window_takeoff         | run_dw_takeoff()              | DWResult               |
| earthwork_takeoff           | run_earthwork_takeoff()       | EarthworkResult        |
| elv_takeoff                 | run_elv_takeoff()             | ELVResult              |
| external_development_takeoff| run_external_dev_takeoff()    | ExternalDevResult      |
| finish_takeoff              | run_finish_takeoff_result()   | FinishResult           |
| foundation_takeoff          | run_foundation_takeoff()      | FoundationResult       |
| implied_items               | run_implied_rules_result()    | ImpliedItemsResult     |
| mep_takeoff                 | run_mep_takeoff()             | MEPQTO                 |
| painting_takeoff            | run_painting_takeoff()        | PaintingResult         |
| plaster_takeoff             | run_plaster_takeoff()         | PlasterResult          |
| prelims_takeoff             | run_prelims_takeoff()         | PrelimsResult          |
| sitework_takeoff            | run_sitework_takeoff()        | SiteworkResult         |
| structural_takeoff          | run_structural_takeoff()      | StructuralQTO          |
| visual_element_detector     | run_visual_element_detection()| VisualQTO              |
| visual_measurement          | run_visual_measurement()      | VisualMeasurementResult|
| wall_area_calculator        | compute_wall_areas()          | WallAreaResult (helper)|
| waterproofing_takeoff       | run_waterproofing_takeoff()   | WaterproofingResult    |

Helper-only modules (not full takeoff modules):
    floor_count_extractor  — extract_floor_count() → FloorCount
    scale_detector         — detect_scale() → ScaleInfo | None
    rate_engine            — apply_rates() / compute_trade_summary()
    rate_engine_interface  — get_rate_engine() singleton, RateEngineProtocol, reset_rate_engine()
"""

# Entry points and result types — importable as `from src.analysis.qto import X`
from .brickwork_takeoff import run_brickwork_takeoff, BrickworkResult
from .door_window_takeoff import run_dw_takeoff, DWResult
from .earthwork_takeoff import run_earthwork_takeoff, EarthworkResult
from .elv_takeoff import run_elv_takeoff, ELVResult
from .external_development_takeoff import run_external_dev_takeoff, ExternalDevResult
from .finish_takeoff import run_finish_takeoff_result, FinishResult
from .foundation_takeoff import run_foundation_takeoff, FoundationResult
from .implied_items import run_implied_rules_result, ImpliedItemsResult
from .mep_takeoff import run_mep_takeoff, MEPQTO
from .painting_takeoff import run_painting_takeoff, PaintingResult
from .plaster_takeoff import run_plaster_takeoff, PlasterResult
from .prelims_takeoff import run_prelims_takeoff, PrelimsResult
from .sitework_takeoff import run_sitework_takeoff, SiteworkResult
from .structural_takeoff import run_structural_takeoff, StructuralQTO
from .waterproofing_takeoff import run_waterproofing_takeoff, WaterproofingResult
from .rate_engine_interface import RateEngineProtocol, get_rate_engine, reset_rate_engine

__all__ = [
    # Entry points
    "run_brickwork_takeoff", "run_dw_takeoff", "run_earthwork_takeoff",
    "run_elv_takeoff", "run_external_dev_takeoff", "run_finish_takeoff_result",
    "run_foundation_takeoff", "run_implied_rules_result", "run_mep_takeoff",
    "run_painting_takeoff", "run_plaster_takeoff", "run_prelims_takeoff",
    "run_sitework_takeoff", "run_structural_takeoff", "run_waterproofing_takeoff",
    # Result types
    "BrickworkResult", "DWResult", "EarthworkResult", "ELVResult",
    "ExternalDevResult", "FinishResult", "FoundationResult", "ImpliedItemsResult",
    "MEPQTO", "PaintingResult", "PlasterResult", "PrelimsResult",
    "SiteworkResult", "StructuralQTO", "WaterproofingResult",
    # Rate engine interface
    "RateEngineProtocol", "get_rate_engine", "reset_rate_engine",
]
