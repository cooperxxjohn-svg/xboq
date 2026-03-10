"""
Agent Office — Live agent status panel for xBOQ pipeline.

Renders an always-visible sidebar panel showing all pipeline agents
as live status cards that update in real-time while the pipeline runs.

Usage (in demo_page.py):
    from agent_office import (
        build_initial_states, make_sub_callback, render_office_sidebar
    )

    # Once, before pipeline run:
    states = build_initial_states()
    placeholder = st.sidebar.empty()
    render_office_sidebar(states, placeholder)

    # Pass to pipeline:
    sub_cb = make_sub_callback(states, placeholder)
    run_analysis_pipeline(..., sub_callback=sub_cb)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class AgentState:
    """Live state for one pipeline agent."""
    agent_id: str
    label: str
    floor: str
    status: str = "idle"    # "idle" | "working" | "done" | "error" | "skipped"
    icon: str = "🤖"
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    message: str = ""
    items_found: int = 0
    pages_processed: int = 0
    error_msg: str = ""

    def elapsed_s(self) -> Optional[float]:
        """Seconds since agent started (still-running uses now as end)."""
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at else time.time()
        return end - self.started_at


# =============================================================================
# AGENT REGISTRY — 26 agents, 5 floors
# =============================================================================

# Each tuple: (agent_id, label, icon, floor)
_AGENT_REGISTRY: List[tuple] = [
    # ── Floor 1: Ingestion ────────────────────────────────────────────
    ("pdf_loader",        "PDF Loader",         "📄", "Ingestion"),
    ("ocr_scanner",       "OCR Scanner",        "🔍", "Ingestion"),
    ("page_indexer",      "Page Indexer",       "🗂️",  "Ingestion"),
    ("page_selector",     "Page Selector",      "🎯", "Ingestion"),

    # ── Floor 2: Extraction ───────────────────────────────────────────
    ("drawing_extractor", "Drawing Extractor",  "📐", "Extraction"),
    ("boq_extractor",     "BOQ Extractor",      "📊", "Extraction"),
    ("spec_extractor",    "Spec Extractor",     "📋", "Extraction"),
    ("schedule_extractor","Schedule Extractor", "📅", "Extraction"),
    ("reconciler",        "Reconciler",         "🔗", "Extraction"),

    # ── Floor 3: QTO ──────────────────────────────────────────────────
    ("finish_agent",      "Finish Takeoff",     "🏠", "QTO"),
    ("scale_agent",       "Scale Detector",     "📏", "QTO"),
    ("structural_agent",  "Structural QTO",     "🏗️",  "QTO"),
    ("mep_agent",         "MEP Takeoff",        "⚡", "QTO"),
    ("visual_detector",   "Visual Detector",    "👁️",  "QTO"),
    ("visual_measure",    "Visual Measurement", "📐", "QTO"),
    ("dw_agent",          "Door/Window QTO",    "🚪", "QTO"),
    ("painting_agent",    "Painting QTO",       "🎨", "QTO"),
    ("waterproof_agent",  "Waterproofing QTO",  "💧", "QTO"),
    ("sitework_agent",    "Sitework QTO",       "🏗",  "QTO"),
    ("rate_engine",       "Rate Engine",        "💰", "QTO"),

    # ── Floor 4: Reasoning ────────────────────────────────────────────
    ("gap_analyzer",      "Gap Analyzer",       "🔎", "Reasoning"),
    ("cost_impact",       "Cost Impact",        "💹", "Reasoning"),
    ("rfi_generator",     "RFI Generator",      "❓", "Reasoning"),
    ("bid_synthesizer",   "Bid Synthesizer",    "🧠", "Reasoning"),

    # ── Floor 5: Front Desk ───────────────────────────────────────────
    ("excel_exporter",    "Excel Exporter",     "📊", "Front Desk"),
    ("vector_indexer",    "Vector Indexer",     "🗄️",  "Front Desk"),
]

FLOOR_ORDER: List[str] = ["Ingestion", "Extraction", "QTO", "Reasoning", "Front Desk"]

FLOOR_ICONS: Dict[str, str] = {
    "Ingestion":  "📥",
    "Extraction": "📑",
    "QTO":        "🔢",
    "Reasoning":  "🧠",
    "Front Desk": "🖥️",
}

# Map high-level pipeline stage IDs → agents to mark working/done
_STAGE_START_AGENTS: Dict[str, List[str]] = {
    "load":    ["pdf_loader"],
    "index":   ["page_indexer"],
    "select":  ["page_selector"],
    "extract": ["drawing_extractor", "boq_extractor", "spec_extractor",
                "schedule_extractor", "ocr_scanner"],
    "graph":   ["reconciler"],
    "reason":  ["gap_analyzer", "cost_impact", "rfi_generator", "bid_synthesizer"],
    "rfi":     ["rfi_generator"],
    "export":  ["excel_exporter", "vector_indexer"],
}


# =============================================================================
# STATE FACTORY
# =============================================================================

def build_initial_states() -> Dict[str, AgentState]:
    """Return a fresh dict of agent_id → AgentState (all idle)."""
    return {
        agent_id: AgentState(
            agent_id=agent_id,
            label=label,
            floor=floor,
            status="idle",
            icon=icon,
        )
        for agent_id, label, icon, floor in _AGENT_REGISTRY
    }


# =============================================================================
# HTML RENDERING
# =============================================================================

def _status_color(status: str) -> str:
    return {
        "idle":    "#4b5563",
        "working": "#f59e0b",
        "done":    "#22c55e",
        "error":   "#ef4444",
        "skipped": "#6b7280",
    }.get(status, "#4b5563")


def _status_bg(status: str) -> str:
    return {
        "idle":    "#111827",
        "working": "#292524",
        "done":    "#052e16",
        "error":   "#2d0707",
        "skipped": "#111827",
    }.get(status, "#111827")


def _agent_card_html(agent: AgentState) -> str:
    """Return one agent card as an HTML string."""
    color = _status_color(agent.status)
    bg    = _status_bg(agent.status)

    # Pulse dot for working, solid dot for others
    if agent.status == "working":
        dot = (
            f'<span style="display:inline-block;width:7px;height:7px;'
            f'border-radius:50%;background:{color};'
            f'animation:xboq-pulse 1.2s ease-in-out infinite;'
            f'flex-shrink:0;"></span>'
        )
    else:
        dot = (
            f'<span style="display:inline-block;width:7px;height:7px;'
            f'border-radius:50%;background:{color};flex-shrink:0;"></span>'
        )

    elapsed = agent.elapsed_s()
    timing_html = ""
    if elapsed is not None and agent.status in ("working", "done", "error"):
        timing_html = (
            f'<span style="color:#6b7280;font-size:9px;margin-left:3px;">'
            f'{elapsed:.1f}s</span>'
        )

    items_html = ""
    if agent.items_found > 0:
        items_html = (
            f'<span style="color:#86efac;font-size:9px;margin-left:3px;">'
            f'+{agent.items_found}</span>'
        )

    msg_html = ""
    if agent.message and agent.status != "idle":
        short_msg = (agent.message[:35] + "…") if len(agent.message) > 35 else agent.message
        msg_html = (
            f'<div style="color:#6b7280;font-size:9px;margin-top:1px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
            f'{short_msg}</div>'
        )

    return (
        f'<div style="background:{bg};border-left:3px solid {color};'
        f'border-radius:3px;padding:3px 6px;margin-bottom:2px;">'
        f'  <div style="display:flex;align-items:center;gap:4px;'
        f'  flex-wrap:nowrap;overflow:hidden;">'
        f'    {dot}'
        f'    <span style="font-size:11px;line-height:1;">{agent.icon}</span>'
        f'    <span style="color:#e5e7eb;font-size:10px;font-weight:500;'
        f'    white-space:nowrap;">{agent.label}</span>'
        f'    {timing_html}{items_html}'
        f'  </div>'
        f'  {msg_html}'
        f'</div>'
    )


_CSS_BLOCK = """<style>
@keyframes xboq-pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.35; transform:scale(0.85); }
}
</style>"""


def _build_office_html(
    states: Dict[str, AgentState],
    pipeline_active: bool = False,
) -> str:
    """Build the complete Agent Office HTML (all floors + all agents)."""
    # Group agents by floor, preserving registry order
    by_floor: Dict[str, List[AgentState]] = {f: [] for f in FLOOR_ORDER}
    # Use registry order so agents appear top-to-bottom as defined
    for agent_id, _label, _icon, floor in _AGENT_REGISTRY:
        if agent_id in states and floor in by_floor:
            by_floor[floor].append(states[agent_id])

    parts: List[str] = [_CSS_BLOCK]

    # Header
    if pipeline_active:
        hdr_color = "#f59e0b"
        hdr_label = "⚙️ Agent Office · running"
    else:
        # Compute summary
        total  = len(states)
        done   = sum(1 for a in states.values() if a.status == "done")
        errors = sum(1 for a in states.values() if a.status == "error")
        hdr_color = "#6b7280"
        if done > 0 and errors == 0:
            hdr_label = f"⚙️ Agent Office · {done}/{total} done"
        elif errors > 0:
            hdr_label = f"⚙️ Agent Office · {errors} errors"
        else:
            hdr_label = "⚙️ Agent Office · idle"

    parts.append(
        f'<div style="color:{hdr_color};font-weight:700;font-size:12px;'
        f'padding:4px 0 6px 0;">{hdr_label}</div>'
    )

    for floor_name in FLOOR_ORDER:
        agents = by_floor.get(floor_name, [])
        if not agents:
            continue

        working = sum(1 for a in agents if a.status == "working")
        done_ct = sum(1 for a in agents if a.status == "done")
        err_ct  = sum(1 for a in agents if a.status == "error")

        if working > 0:
            floor_color = "#f59e0b"
        elif err_ct > 0:
            floor_color = "#ef4444"
        elif done_ct == len(agents):
            floor_color = "#22c55e"
        else:
            floor_color = "#4b5563"

        floor_icon = FLOOR_ICONS.get(floor_name, "🏢")
        parts.append(
            f'<div style="color:{floor_color};font-size:10px;font-weight:600;'
            f'margin-top:5px;margin-bottom:2px;padding-left:2px;">'
            f'{floor_icon} {floor_name}</div>'
        )
        for agent in agents:
            parts.append(_agent_card_html(agent))

    return "\n".join(parts)


def render_office_sidebar(
    states: Dict[str, AgentState],
    placeholder,
    pipeline_active: bool = False,
) -> None:
    """
    Write Agent Office HTML into a Streamlit sidebar placeholder.

    Args:
        states: agent_id → AgentState mapping (mutated live during pipeline)
        placeholder: st.sidebar.empty() object to write into
        pipeline_active: True while pipeline is running
    """
    html = _build_office_html(states, pipeline_active=pipeline_active)
    placeholder.markdown(html, unsafe_allow_html=True)


# =============================================================================
# CALLBACK FACTORIES
# =============================================================================

def make_sub_callback(
    states: Dict[str, AgentState],
    placeholder,
    pipeline_active_flag: Optional[List[bool]] = None,
) -> Callable[[str, str, str, int], None]:
    """
    Return a sub_callback(agent_id, status, message, items) suitable for
    passing as `sub_callback` to run_analysis_pipeline().

    When called, it:
      1. Mutates the given agent's state in `states`
      2. Re-renders the entire office panel into `placeholder`

    Args:
        states: agent state dict (from build_initial_states())
        placeholder: st.sidebar.empty() object
        pipeline_active_flag: single-element list used as a mutable bool;
            [True] while pipeline is running so the header shows "running".
    """
    _active = pipeline_active_flag if pipeline_active_flag is not None else [True]

    def _cb(agent_id: str, status: str, message: str = "", items: int = 0) -> None:
        if agent_id not in states:
            return
        agent = states[agent_id]
        now = time.time()

        if status == "working":
            agent.status = "working"
            agent.started_at = agent.started_at or now
            if message:
                agent.message = message
        elif status == "done":
            agent.status = "done"
            agent.ended_at = now
            if message:
                agent.message = message
            if items:
                agent.items_found = items
        elif status == "error":
            agent.status = "error"
            agent.ended_at = now
            agent.error_msg = message
            agent.message = (message[:40] + "…") if len(message) > 40 else message
        elif status == "skipped":
            agent.status = "skipped"
            agent.ended_at = now
            if message:
                agent.message = message
        elif status == "idle":
            agent.status = "idle"
            agent.started_at = None
            agent.ended_at = None
            agent.message = ""
            agent.items_found = 0

        # Re-render office panel (never crash pipeline on UI error)
        try:
            render_office_sidebar(states, placeholder, pipeline_active=bool(_active[0]))
        except Exception:
            pass

    return _cb


def make_stage_sub_callback(
    states: Dict[str, AgentState],
    placeholder,
    pipeline_active_flag: Optional[List[bool]] = None,
) -> Callable[[str, str, float], None]:
    """
    Return a progress_callback(stage_id, message, progress) that also fires
    sub-agent status updates for the high-level stage map.

    This is used in addition to make_sub_callback — the sub_callback fires
    granular QTO-level updates; this fires coarser Ingestion/Extraction-level.

    Args:
        states, placeholder, pipeline_active_flag: same as make_sub_callback
    """
    sub_cb = make_sub_callback(states, placeholder, pipeline_active_flag)

    _stage_started: set = set()
    _stage_done: set = set()

    def _stage_cb(stage_id: str, message: str, progress: float) -> None:
        if stage_id not in _stage_started and progress > 0:
            _stage_started.add(stage_id)
            for agent_id in _STAGE_START_AGENTS.get(stage_id, []):
                sub_cb(agent_id, "working", message)

        if stage_id not in _stage_done and progress >= 1.0:
            _stage_done.add(stage_id)
            for agent_id in _STAGE_START_AGENTS.get(stage_id, []):
                # Only mark done if not already set by a more specific sub_callback
                if states.get(agent_id) and states[agent_id].status == "working":
                    sub_cb(agent_id, "done", message)

    return _stage_cb
