"""
QTO Module Registry — single source of truth for all QTO module metadata.

Usage:
    from src.analysis.qto.registry import QTO_REGISTRY, is_enabled, get_agent_ids

Each entry describes one QTO module: its agent_id (used in fire_sub / SSE events),
whether it requires an LLM client, and whether it can be disabled via an env var.

To disable a module at runtime:
    XBOQ_DISABLE_QTO=implied,flooring   (comma-separated module names)
    XBOQ_DISABLE_IMPLIED_ITEMS=1        (legacy flag, still honoured)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class QTOModuleInfo:
    """Metadata for one QTO module."""

    name: str                       # internal name (matches XBOQ_DISABLE_QTO value)
    agent_id: str                   # used in fire_sub / SSE events
    description: str                # human-readable description
    requires_llm: bool = False      # True if module needs an LLM client
    requires_pdf: bool = False      # True if module needs the raw PDF bytes
    output_key: str = ""            # key in payload dict (empty = no direct payload key)
    disable_env: str = ""           # env var that disables this module (legacy)
    trades: List[str] = field(default_factory=list)  # trades this module covers


# Ordered list — matches the execution order in qto_runner.py
QTO_REGISTRY: List[QTOModuleInfo] = [
    QTOModuleInfo(
        name="finish",
        agent_id="finish_agent",
        description="Room-finish takeoff from plan drawings",
        output_key="",
        trades=["finishing", "flooring", "painting"],
    ),
    QTOModuleInfo(
        name="scale",
        agent_id="scale_agent",
        description="Drawing scale detection (feeds structural + visual modules)",
        output_key="",
        trades=[],
    ),
    QTOModuleInfo(
        name="structural",
        agent_id="structural_agent",
        description="Structural element takeoff from schedule text",
        output_key="",
        trades=["structural", "rcc", "steel"],
    ),
    QTOModuleInfo(
        name="implied",
        agent_id="implied_agent",
        description="Implied items rule engine (derived from structural elements)",
        disable_env="XBOQ_DISABLE_IMPLIED_ITEMS",
        output_key="",
        trades=["structural", "masonry", "finishing"],
    ),
    QTOModuleInfo(
        name="mep",
        agent_id="mep_agent",
        description="MEP (mechanical, electrical, plumbing) takeoff",
        output_key="mep_qto",
        trades=["mep", "electrical", "plumbing", "hvac"],
    ),
    QTOModuleInfo(
        name="visual",
        agent_id="visual_detector",
        description="Visual element detection from PDF drawings (needs LLM)",
        requires_llm=True,
        requires_pdf=True,
        output_key="",
        trades=["structural", "finishing"],
    ),
    QTOModuleInfo(
        name="vmeas",
        agent_id="visual_measure",
        description="Visual room measurement from PDF drawings (needs LLM)",
        requires_llm=True,
        requires_pdf=True,
        output_key="",
        trades=["finishing", "civil"],
    ),
    QTOModuleInfo(
        name="dw",
        agent_id="dw_agent",
        description="Door & window takeoff",
        output_key="dw_takeoff",
        trades=["doors_windows", "finishing"],
    ),
    QTOModuleInfo(
        name="painting",
        agent_id="painting_agent",
        description="Painting & decoration takeoff",
        output_key="painting_result",
        trades=["painting", "finishing"],
    ),
    QTOModuleInfo(
        name="waterproofing",
        agent_id="waterproof_agent",
        description="Waterproofing takeoff",
        output_key="waterproofing_result",
        trades=["waterproofing", "civil"],
    ),
    QTOModuleInfo(
        name="sitework",
        agent_id="sitework_agent",
        description="Site work takeoff (earthwork, roads, drainage)",
        output_key="sitework_result",
        trades=["civil", "sitework"],
    ),
    QTOModuleInfo(
        name="brickwork",
        agent_id="brickwork_agent",
        description="Brickwork / block-masonry takeoff",
        output_key="brickwork_result",
        trades=["masonry", "civil"],
    ),
    QTOModuleInfo(
        name="plaster",
        agent_id="plaster_agent",
        description="Plaster & render takeoff",
        output_key="plaster_result",
        trades=["finishing", "civil"],
    ),
    QTOModuleInfo(
        name="earthwork",
        agent_id="earthwork_agent",
        description="Earthwork (excavation, filling, compaction) takeoff",
        output_key="earthwork_result",
        trades=["civil", "sitework"],
    ),
    QTOModuleInfo(
        name="flooring",
        agent_id="flooring_agent",
        description="Flooring takeoff",
        output_key="flooring_result",
        trades=["finishing", "flooring"],
    ),
    QTOModuleInfo(
        name="scope_disagg",
        agent_id="disagg_agent",
        description="Scope disaggregation for multi-building tenders",
        output_key="",
        trades=[],
    ),
    QTOModuleInfo(
        name="spec_llm",
        agent_id="spec_llm_agent",
        description="LLM-driven spec quantity extraction",
        output_key="",
        trades=["general"],
    ),
    QTOModuleInfo(
        name="foundation",
        agent_id="foundation_agent",
        description="Foundation & substructure takeoff",
        output_key="foundation_result",
        trades=["civil", "structural"],
    ),
    QTOModuleInfo(
        name="extdev",
        agent_id="extdev_agent",
        description="External development takeoff (landscaping, parking, utilities)",
        output_key="extdev_result",
        trades=["civil", "sitework"],
    ),
    QTOModuleInfo(
        name="prelims",
        agent_id="prelims_agent",
        description="Preliminaries takeoff (mobilisation, temp facilities, overheads)",
        output_key="prelims_result",
        trades=["prelims", "general"],
    ),
    QTOModuleInfo(
        name="elv",
        agent_id="elv_agent",
        description="ELV (extra-low voltage) systems takeoff",
        output_key="elv_result",
        trades=["elv", "electrical"],
    ),
]

# Fast lookup by name
_BY_NAME: dict[str, QTOModuleInfo] = {m.name: m for m in QTO_REGISTRY}


def get_module(name: str) -> Optional[QTOModuleInfo]:
    """Return module metadata by name, or None if not found."""
    return _BY_NAME.get(name)


def is_enabled(name: str) -> bool:
    """
    Return True if the named QTO module is enabled (not disabled by env var).

    Checks two env vars:
      1. Legacy per-module var stored in QTOModuleInfo.disable_env
      2. Global XBOQ_DISABLE_QTO=name1,name2,... list
    """
    info = _BY_NAME.get(name)
    if info is None:
        return True  # unknown modules are not gated

    # Legacy per-module env var
    if info.disable_env:
        if os.environ.get(info.disable_env, "").strip() in ("1", "true", "yes"):
            return False

    # Global disable list
    disabled = {
        s.strip().lower()
        for s in os.environ.get("XBOQ_DISABLE_QTO", "").split(",")
        if s.strip()
    }
    return name not in disabled


def get_agent_ids() -> List[str]:
    """Return all agent_id strings in execution order."""
    return [m.agent_id for m in QTO_REGISTRY]


def list_modules(enabled_only: bool = False) -> List[dict]:
    """Return a serialisable list of module metadata (used by /api/qto-modules)."""
    result = []
    for m in QTO_REGISTRY:
        enabled = is_enabled(m.name)
        if enabled_only and not enabled:
            continue
        result.append({
            "name":         m.name,
            "agent_id":     m.agent_id,
            "description":  m.description,
            "requires_llm": m.requires_llm,
            "requires_pdf": m.requires_pdf,
            "output_key":   m.output_key,
            "trades":       m.trades,
            "enabled":      enabled,
        })
    return result
