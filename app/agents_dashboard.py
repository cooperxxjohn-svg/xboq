"""
Agent Dashboard — Visual control panel for xBOQ agent registry.

Sprint 23: Agent Registry + CLI Runner + Dashboard.

Launch:
    streamlit run app/agents_dashboard.py

Three tabs:
    1. Catalog   — Browse all agents grouped by category
    2. Run Agent — Execute any standalone agent with file upload
    3. History   — View past run results (session-scoped)
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Path setup (follows demo_page.py convention) ──
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent_registry import (
    AgentCategory,
    AgentSpec,
    AgentParam,
    get_agent,
    list_agents,
    list_categories,
    resolve_fn,
    standalone_agents,
    agent_count,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="xBOQ Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# HELPERS (follow demo_page.py patterns)
# =============================================================================

def _make_widget_key(*parts) -> str:
    """Build a unique Streamlit widget key from variable parts."""
    clean = [str(p).replace(" ", "_")[:40] for p in parts if p is not None and str(p)]
    return ":".join(clean) if clean else f"wk_{id(parts)}"


def _safe_str(val) -> str:
    """Convert any value to a clean display string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val[:10])
    if isinstance(val, dict):
        return ", ".join(f"{k}: {v}" for k, v in list(val.items())[:5])
    return str(val)


CATEGORY_ICONS = {
    "pipeline": "🔗",
    "extractor": "🔍",
    "analysis": "📊",
    "structural": "🏗️",
    "output": "📄",
    "project": "📁",
}

CATEGORY_COLORS = {
    "pipeline": "#4A90D9",
    "extractor": "#7B68EE",
    "analysis": "#27AE60",
    "structural": "#E67E22",
    "output": "#3498DB",
    "project": "#9B59B6",
}


# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .agent-card:hover {
        border-color: #4A90D9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .agent-name { font-family: monospace; color: #666; font-size: 0.85rem; }
    .standalone-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .standalone-yes { background: #E8F5E9; color: #2E7D32; }
    .standalone-no { background: #FFF3E0; color: #E65100; }
    .category-header {
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .run-status-success { color: #2E7D32; }
    .run-status-error { color: #C62828; }
    .run-status-running { color: #1565C0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with stats and filters."""
    with st.sidebar:
        st.markdown("## 🤖 Agent Registry")
        st.caption(f"{agent_count()} agents | {len(list_categories())} categories")

        st.markdown("---")

        # Category filter
        st.markdown("### Filters")
        selected_cat = st.selectbox(
            "Category",
            ["All"] + [c for c in list_categories()],
            key=_make_widget_key("sidebar", "category"),
        )

        standalone_only = st.checkbox(
            "Standalone only",
            key=_make_widget_key("sidebar", "standalone"),
        )

        search_text = st.text_input(
            "Search agents",
            placeholder="Type to filter...",
            key=_make_widget_key("sidebar", "search"),
        )

        st.markdown("---")

        # Quick stats
        st.markdown("### Quick Stats")
        all_agents = list_agents()
        standalone_count = sum(1 for a in all_agents if a.can_run_standalone)

        col1, col2 = st.columns(2)
        col1.metric("Total", agent_count())
        col2.metric("Standalone", standalone_count)

        for cat in list_categories():
            cat_agents = list_agents(category=cat)
            icon = CATEGORY_ICONS.get(cat, "")
            st.caption(f"{icon} {cat.title()}: {len(cat_agents)}")

        st.markdown("---")
        st.markdown("### CLI Usage")
        st.code("python -m src agents list", language="bash")
        st.code("python -m src agents info <name>", language="bash")
        st.code("python -m src agents run <name> -i <file>", language="bash")

    return selected_cat, standalone_only, search_text


# =============================================================================
# TAB 1: CATALOG
# =============================================================================

def render_catalog_tab(selected_cat: str, standalone_only: bool, search_text: str):
    """Agent catalog grouped by category."""

    # Filter agents
    cat_filter = selected_cat if selected_cat != "All" else None
    agents = list_agents(category=cat_filter)

    if standalone_only:
        agents = [a for a in agents if a.can_run_standalone]

    if search_text:
        q = search_text.lower()
        agents = [
            a for a in agents
            if q in a.name.lower()
            or q in a.label.lower()
            or q in a.description.lower()
            or any(q in t for t in a.tags)
        ]

    if not agents:
        st.info("No agents match the current filters.")
        return

    # Group by category
    by_category: Dict[str, List[AgentSpec]] = {}
    for a in agents:
        by_category.setdefault(a.category.value, []).append(a)

    for cat_name in list_categories():
        cat_agents = by_category.get(cat_name, [])
        if not cat_agents:
            continue

        icon = CATEGORY_ICONS.get(cat_name, "")
        st.markdown(f"### {icon} {cat_name.title()} ({len(cat_agents)})")

        # 3-column grid
        cols = st.columns(3)
        for i, agent in enumerate(cat_agents):
            with cols[i % 3]:
                with st.container(border=True):
                    # Header
                    standalone_label = "standalone" if agent.can_run_standalone else "pipeline-only"
                    standalone_class = "standalone-yes" if agent.can_run_standalone else "standalone-no"

                    st.markdown(f"**{agent.label}**")
                    st.markdown(
                        f'<span class="standalone-badge {standalone_class}">{standalone_label}</span>',
                        unsafe_allow_html=True,
                    )

                    # Description
                    st.caption(agent.description[:120])

                    # Module path
                    st.markdown(f'<span class="agent-name">{agent.name}</span>', unsafe_allow_html=True)

                    # Tags
                    if agent.tags:
                        st.caption(" ".join(f"`{t}`" for t in agent.tags[:4]))

                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(
                            "Details",
                            key=_make_widget_key("cat", "detail", agent.name),
                            use_container_width=True,
                        ):
                            st.session_state["detail_agent"] = agent.name
                    with btn_col2:
                        if agent.can_run_standalone:
                            if st.button(
                                "Run →",
                                key=_make_widget_key("cat", "run", agent.name),
                                use_container_width=True,
                                type="primary",
                            ):
                                st.session_state["run_agent"] = agent.name

    # Detail panel (shown at bottom when agent selected)
    detail_name = st.session_state.get("detail_agent")
    if detail_name:
        agent = get_agent(detail_name)
        if agent:
            st.markdown("---")
            _render_agent_detail(agent)


def _render_agent_detail(agent: AgentSpec):
    """Render detailed view for a single agent."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"## {agent.label}")
        st.markdown(agent.description)

        st.markdown("#### Module")
        st.code(f"{agent.module_path}.{agent.entry_fn}", language="python")

    with col2:
        st.markdown("#### Properties")
        st.markdown(f"**Category:** {agent.category.value}")
        st.markdown(f"**Standalone:** {'Yes' if agent.can_run_standalone else 'No'}")
        if agent.tags:
            st.markdown(f"**Tags:** {', '.join(agent.tags)}")

    # Inputs/Outputs
    inp_col, out_col = st.columns(2)

    with inp_col:
        st.markdown("#### Inputs")
        if agent.inputs:
            for inp in agent.inputs:
                req = "required" if inp.required else "optional"
                st.markdown(f"- **{inp.name}** `{inp.type}` ({req})")
                if inp.description:
                    st.caption(f"  {inp.description}")
        else:
            st.caption("No inputs")

    with out_col:
        st.markdown("#### Outputs")
        if agent.outputs:
            for out in agent.outputs:
                st.markdown(f"- **{out.name}** `{out.type}`")
                if out.description:
                    st.caption(f"  {out.description}")
        else:
            st.caption("No outputs")

    # CLI usage
    if agent.can_run_standalone:
        st.markdown("#### CLI Usage")
        st.code(f"python -m src agents run {agent.name} --input <file>", language="bash")

    if st.button("Close", key=_make_widget_key("detail", "close", agent.name)):
        del st.session_state["detail_agent"]
        st.rerun()


# =============================================================================
# TAB 2: RUN AGENT
# =============================================================================

def render_run_tab():
    """Run a single agent with file upload and parameter controls."""

    sa = standalone_agents()
    if not sa:
        st.warning("No standalone agents available.")
        return

    # Agent selector
    default_agent = st.session_state.get("run_agent", sa[0].name)
    agent_names = [a.name for a in sa]
    default_idx = agent_names.index(default_agent) if default_agent in agent_names else 0

    selected_name = st.selectbox(
        "Select Agent",
        agent_names,
        index=default_idx,
        format_func=lambda n: f"{get_agent(n).label}  ({n})",
        key=_make_widget_key("run", "agent_select"),
    )

    agent = get_agent(selected_name)
    if not agent:
        return

    # Agent info bar
    icon = CATEGORY_ICONS.get(agent.category.value, "")
    st.markdown(f"{icon} **{agent.label}** — {agent.description}")

    st.markdown("---")

    # ── Input section ──
    st.markdown("### Inputs")
    input_values: Dict[str, Any] = {}
    uploaded_file = None

    for inp in agent.inputs:
        if inp.type == "Path":
            uploaded = st.file_uploader(
                f"{inp.name} — {inp.description}",
                type=["pdf", "json", "xlsx", "xls", "txt"],
                key=_make_widget_key("run", "upload", agent.name, inp.name),
            )
            if uploaded:
                uploaded_file = uploaded
                input_values[inp.name] = uploaded
        elif inp.type == "dict":
            json_text = st.text_area(
                f"{inp.name} (JSON) — {inp.description}",
                height=200,
                placeholder='{"key": "value"}',
                key=_make_widget_key("run", "json", agent.name, inp.name),
            )
            if json_text.strip():
                try:
                    input_values[inp.name] = json.loads(json_text)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON for {inp.name}: {e}")
        elif inp.type.startswith("List"):
            json_text = st.text_area(
                f"{inp.name} (JSON array) — {inp.description}",
                height=150,
                placeholder='[{"item": 1}]',
                key=_make_widget_key("run", "list", agent.name, inp.name),
            )
            if json_text.strip():
                try:
                    input_values[inp.name] = json.loads(json_text)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON for {inp.name}: {e}")
        elif inp.type == "str":
            val = st.text_input(
                f"{inp.name} — {inp.description}",
                key=_make_widget_key("run", "str", agent.name, inp.name),
            )
            if val:
                input_values[inp.name] = val
        elif inp.type == "int":
            val = st.number_input(
                f"{inp.name} — {inp.description}",
                step=1,
                key=_make_widget_key("run", "int", agent.name, inp.name),
            )
            input_values[inp.name] = int(val)

    # ── Run button ──
    st.markdown("---")
    run_col, clear_col = st.columns([3, 1])

    with run_col:
        run_clicked = st.button(
            f"Run {agent.label}",
            type="primary",
            use_container_width=True,
            key=_make_widget_key("run", "execute", agent.name),
        )

    with clear_col:
        if st.button(
            "Clear Results",
            use_container_width=True,
            key=_make_widget_key("run", "clear", agent.name),
        ):
            run_key = f"agent_run_{agent.name}"
            if run_key in st.session_state:
                del st.session_state[run_key]
            st.rerun()

    if run_clicked:
        _execute_agent_ui(agent, input_values, uploaded_file)

    # ── Show results ──
    run_key = f"agent_run_{agent.name}"
    if run_key in st.session_state:
        run_data = st.session_state[run_key]
        _render_run_result(run_data, agent)


def _execute_agent_ui(
    agent: AgentSpec,
    input_values: Dict[str, Any],
    uploaded_file: Any,
):
    """Execute an agent and store result in session state."""
    run_key = f"agent_run_{agent.name}"

    with st.spinner(f"Running {agent.label}..."):
        try:
            fn = resolve_fn(agent)

            # Build kwargs
            kwargs: Dict[str, Any] = {}
            for inp in agent.inputs:
                if inp.name in input_values:
                    val = input_values[inp.name]
                    # Handle uploaded files
                    if hasattr(val, "read"):
                        if inp.type == "Path":
                            # Save to temp file
                            import tempfile
                            suffix = "." + val.name.split(".")[-1] if "." in val.name else ""
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                            tmp.write(val.read())
                            tmp.flush()
                            kwargs[inp.name] = Path(tmp.name)
                        else:
                            content = val.read()
                            try:
                                kwargs[inp.name] = json.loads(content)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                kwargs[inp.name] = content.decode("utf-8", errors="replace")
                    else:
                        kwargs[inp.name] = val
                elif not inp.required:
                    continue

            t0 = time.perf_counter()
            result = fn(**kwargs)
            elapsed = time.perf_counter() - t0

            # Serialize
            if hasattr(result, "to_dict"):
                result_data = result.to_dict()
            elif isinstance(result, tuple):
                result_data = {}
                for i, out in enumerate(agent.outputs):
                    if i < len(result):
                        v = result[i]
                        result_data[out.name] = v.to_dict() if hasattr(v, "to_dict") else v
            elif isinstance(result, (dict, list)):
                result_data = result
            else:
                result_data = {"result": str(result)}

            st.session_state[run_key] = {
                "status": "complete",
                "duration": elapsed,
                "result": result_data,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent.name,
                "agent_label": agent.label,
            }

            # Add to history
            _add_to_history(agent, elapsed, "complete")

        except Exception as e:
            elapsed = time.perf_counter() - t0 if "t0" in dir() else 0
            st.session_state[run_key] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent.name,
                "agent_label": agent.label,
            }
            _add_to_history(agent, elapsed, "error", str(e))


def _render_run_result(run_data: dict, agent: AgentSpec):
    """Display run results."""
    status = run_data.get("status")

    if status == "complete":
        st.success(f"Completed in {run_data['duration']:.2f}s")

        result = run_data.get("result", {})

        # Summary metrics
        if isinstance(result, dict):
            metric_cols = st.columns(min(len(result), 4))
            for i, (k, v) in enumerate(list(result.items())[:4]):
                with metric_cols[i]:
                    if isinstance(v, list):
                        st.metric(k, f"{len(v)} items")
                    elif isinstance(v, (int, float)):
                        st.metric(k, v)
                    elif isinstance(v, str) and len(v) < 30:
                        st.metric(k, v)

        # Full output
        with st.expander("Full Output (JSON)", expanded=False):
            st.json(result)

        # Download button
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{agent.name}_output.json",
            mime="application/json",
            key=_make_widget_key("run", "download", agent.name),
        )

    elif status == "error":
        st.error(f"Failed: {run_data.get('error', 'Unknown error')}")
        with st.expander("Traceback"):
            st.code(run_data.get("traceback", ""), language="python")


# =============================================================================
# TAB 3: HISTORY
# =============================================================================

def _add_to_history(agent: AgentSpec, elapsed: float, status: str, error: str = ""):
    """Add a run to session history."""
    if "agent_run_history" not in st.session_state:
        st.session_state["agent_run_history"] = []

    st.session_state["agent_run_history"].insert(0, {
        "agent_name": agent.name,
        "agent_label": agent.label,
        "category": agent.category.value,
        "status": status,
        "duration": elapsed,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    })

    # Keep last 50
    st.session_state["agent_run_history"] = st.session_state["agent_run_history"][:50]


def render_history_tab():
    """Show past run history."""
    history = st.session_state.get("agent_run_history", [])

    if not history:
        st.info("No runs yet. Go to the **Run Agent** tab to execute an agent.")
        return

    st.markdown(f"### Run History ({len(history)} runs)")

    for i, run in enumerate(history):
        status_icon = "✓" if run["status"] == "complete" else "✗"
        status_color = "run-status-success" if run["status"] == "complete" else "run-status-error"
        duration = f"{run['duration']:.2f}s" if run.get("duration") else "—"

        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
            with col1:
                st.markdown(f"**{run['agent_label']}** ({run['agent_name']})")
            with col2:
                icon = CATEGORY_ICONS.get(run["category"], "")
                st.caption(f"{icon} {run['category']}")
            with col3:
                st.markdown(
                    f'<span class="{status_color}">{status_icon} {run["status"]}</span>',
                    unsafe_allow_html=True,
                )
            with col4:
                ts = run.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        st.caption(f"{dt.strftime('%H:%M:%S')} | {duration}")
                    except ValueError:
                        st.caption(duration)

            if run.get("error"):
                st.caption(f"Error: {run['error'][:100]}")

    if st.button("Clear History", key=_make_widget_key("history", "clear")):
        st.session_state["agent_run_history"] = []
        st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for the agents dashboard."""

    # Sidebar
    selected_cat, standalone_only, search_text = render_sidebar()

    # Header
    st.markdown("# 🤖 xBOQ Agent Dashboard")
    st.caption("Browse, inspect, and run any agent independently")

    # Tabs
    tab_catalog, tab_run, tab_history = st.tabs([
        f"📋 Catalog ({agent_count()})",
        "▶️ Run Agent",
        "📜 History",
    ])

    with tab_catalog:
        render_catalog_tab(selected_cat, standalone_only, search_text)

    with tab_run:
        render_run_tab()

    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()
