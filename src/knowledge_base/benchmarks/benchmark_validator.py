"""
Benchmark validator — compares computed QTO quantities against
per-sqm-BUA reference ranges from quantity_benchmarks.yaml.

Usage:
    from src.knowledge_base.benchmarks.benchmark_validator import validate_qto_quantities
    flags = validate_qto_quantities("hostel_institutional", 5000.0, qto_items)
    # flags is a list of dicts with status "OK"|"LOW"|"HIGH"|"MISSING"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Item keyword → benchmark item name mapping ────────────────────────────────
# Maps keywords found in QTO item descriptions to benchmark item names
_DESC_TO_BENCHMARK: Dict[str, str] = {
    "rcc":              "concrete_rcc_total",
    "reinforced cement concrete": "concrete_rcc_total",
    "m-25":             "concrete_rcc_total",
    "m-20":             "concrete_rcc_total",
    "tmt":              "steel_tmt_total",
    "reinforcement":    "steel_tmt_total",
    "steel bars":       "steel_tmt_total",
    "brickwork":        "brickwork_total",
    "brick masonry":    "brickwork_total",
    "aac block":        "brickwork_total",
    "plaster":          "plastering_total",
    "rendering":        "plastering_total",
    "vitrified tile":   "flooring_tile",
    "ceramic tile":     "flooring_tile",
    "flooring":         "flooring_tile",
    "kota stone":       "flooring_tile",
    "emulsion":         "painting_interior",
    "distemper":        "painting_interior",
    "obd":              "painting_interior",
    "paint":            "painting_interior",
    "excavation":       "earthwork_excavation",
    "earthwork":        "earthwork_excavation",
}

_BENCHMARKS_CACHE: Optional[dict] = None


def load_benchmarks(yaml_path: Optional[Path] = None) -> dict:
    """Load quantity_benchmarks.yaml. Returns dict keyed by project_type."""
    global _BENCHMARKS_CACHE
    if _BENCHMARKS_CACHE is not None:
        return _BENCHMARKS_CACHE

    if yaml_path is None:
        # Search standard locations
        candidates = [
            Path(__file__).parent / "quantity_benchmarks.yaml",
            Path(__file__).parent.parent.parent.parent / "data" / "benchmarks" / "quantity_benchmarks.yaml",
            Path(__file__).parent.parent / "quantity_benchmarks.yaml",
        ]
        for c in candidates:
            if c.exists():
                yaml_path = c
                break

    if yaml_path is None or not yaml_path.exists():
        logger.warning("quantity_benchmarks.yaml not found; benchmark validation disabled")
        _BENCHMARKS_CACHE = {}
        return _BENCHMARKS_CACHE

    try:
        import yaml  # type: ignore
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
    except ImportError:
        # Fallback to a minimal YAML parser for lists of dicts
        logger.warning("pyyaml not installed; benchmark validation disabled")
        _BENCHMARKS_CACHE = {}
        return _BENCHMARKS_CACHE

    result = {}
    project_types = raw if isinstance(raw, list) else raw.get("project_benchmarks", raw.get("project_types", []))
    for pt in project_types:
        ptype = pt.get("project_type", "")
        if ptype:
            result[ptype] = {
                item["item"]: item
                for item in pt.get("per_sqm_bua", [])
            }
    _BENCHMARKS_CACHE = result
    return result


def _classify_item(description: str) -> Optional[str]:
    """Map a QTO item description to a benchmark item name."""
    lower = description.lower()
    for keyword, bench_name in _DESC_TO_BENCHMARK.items():
        if keyword in lower:
            return bench_name
    return None


def validate_qto_quantities(
    building_type: str,
    floor_area_sqm: float,
    qto_items: List[dict],
) -> List[dict]:
    """
    Compare computed QTO quantities against benchmark per-sqm-BUA ranges.

    Returns list of dicts:
      {item, computed_per_sqm, benchmark_min, benchmark_max,
       benchmark_typical, status, deviation_pct}
    """
    if floor_area_sqm <= 0:
        return []

    benchmarks = load_benchmarks()
    bench_for_type = benchmarks.get(building_type, {})
    if not bench_for_type:
        logger.debug("No benchmark data for building_type=%s", building_type)
        return []

    # Aggregate QTO quantities by benchmark item name
    aggregated: Dict[str, float] = {}
    for it in qto_items:
        desc = it.get("description", "")
        qty = float(it.get("qty") or it.get("quantity") or 0)
        bench_name = _classify_item(desc)
        if bench_name:
            aggregated[bench_name] = aggregated.get(bench_name, 0.0) + qty

    flags = []
    for bench_name, ref in bench_for_type.items():
        computed_total = aggregated.get(bench_name, 0.0)
        computed_per_sqm = computed_total / floor_area_sqm
        b_min = float(ref.get("min", 0))
        b_max = float(ref.get("max", float("inf")))
        b_typ = float(ref.get("typical", (b_min + b_max) / 2))

        if computed_total == 0.0:
            status = "MISSING"
            deviation_pct = None
        elif computed_per_sqm < b_min:
            status = "LOW"
            deviation_pct = round((computed_per_sqm - b_typ) / b_typ * 100, 1)
        elif computed_per_sqm > b_max:
            status = "HIGH"
            deviation_pct = round((computed_per_sqm - b_typ) / b_typ * 100, 1)
        else:
            status = "OK"
            deviation_pct = round((computed_per_sqm - b_typ) / b_typ * 100, 1)

        flags.append({
            "item": bench_name,
            "computed_total": round(computed_total, 2),
            "computed_per_sqm": round(computed_per_sqm, 4),
            "benchmark_min": b_min,
            "benchmark_max": b_max,
            "benchmark_typical": b_typ,
            "unit": ref.get("unit", ""),
            "status": status,
            "deviation_pct": deviation_pct,
        })

    return flags
