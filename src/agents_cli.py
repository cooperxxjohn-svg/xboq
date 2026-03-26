"""
Agent CLI — Command-line interface for the agent registry.

Sprint 23: Agent Registry + CLI Runner + Dashboard.

Subcommands:
    agents list [--category <cat>] [--format table|json]
    agents info <name> [--format table|json]
    agents run <name> --input <file> [--output <dir>] [--params key=value ...]
"""

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_registry import (
    AgentCategory,
    AgentSpec,
    get_agent,
    list_agents,
    list_categories,
    resolve_fn,
    standalone_agents,
    agent_count,
)


# =============================================================================
# LIST COMMAND
# =============================================================================

def cmd_agents_list(args) -> int:
    """List all registered agents."""
    agents = list_agents(category=getattr(args, "category", None))

    if getattr(args, "standalone", False):
        agents = [a for a in agents if a.can_run_standalone]

    fmt = getattr(args, "format", "table")
    if fmt == "json":
        print(json.dumps([a.to_dict() for a in agents], indent=2))
        return 0

    # ── Table format ──
    print(f"\n  xBOQ Agent Registry — {len(agents)} agents\n")

    current_cat = None
    for agent in agents:
        cat_val = agent.category.value
        if cat_val != current_cat:
            current_cat = cat_val
            cat_count = sum(1 for a in agents if a.category.value == cat_val)
            print(f"  ┌─ {cat_val.upper()} ({cat_count}) ─────────────────────────────")

        standalone = "✓" if agent.can_run_standalone else "·"
        print(f"  │  {standalone}  {agent.name:<25} {agent.label}")

    print()
    standalone_count = sum(1 for a in agents if a.can_run_standalone)
    print(f"  Total: {len(agents)} agents ({standalone_count} standalone)")
    print(f"  Run 'python -m src agents info <name>' for details\n")
    return 0


# =============================================================================
# INFO COMMAND
# =============================================================================

def cmd_agents_info(args) -> int:
    """Show details for a specific agent."""
    name = args.name
    agent = get_agent(name)

    if not agent:
        print(f"\n  Agent not found: '{name}'")
        print(f"  Available agents:")
        for a in list_agents():
            print(f"    {a.name}")
        return 1

    fmt = getattr(args, "format", "table")
    if fmt == "json":
        print(json.dumps(agent.to_dict(), indent=2))
        return 0

    # ── Detail view ──
    print()
    print(f"  ╔═══════════════════════════════════════════════════")
    print(f"  ║  {agent.label}")
    print(f"  ╚═══════════════════════════════════════════════════")
    print(f"  Name:        {agent.name}")
    print(f"  Category:    {agent.category.value}")
    print(f"  Standalone:  {'Yes' if agent.can_run_standalone else 'No'}")
    print(f"  Module:      {agent.module_path}")
    print(f"  Function:    {agent.entry_fn}")
    if agent.tags:
        print(f"  Tags:        {', '.join(agent.tags)}")
    print()
    print(f"  {agent.description}")

    if agent.inputs:
        print(f"\n  Inputs:")
        for inp in agent.inputs:
            req = "required" if inp.required else "optional"
            print(f"    {'→'} {inp.name} ({inp.type}, {req})")
            if inp.description:
                print(f"      {inp.description}")

    if agent.outputs:
        print(f"\n  Outputs:")
        for out in agent.outputs:
            print(f"    {'←'} {out.name} ({out.type})")
            if out.description:
                print(f"      {out.description}")

    if agent.can_run_standalone:
        print(f"\n  Usage:")
        input_hint = "<input_file>"
        for inp in agent.inputs:
            if inp.type == "Path" and inp.required:
                input_hint = f"<{inp.name}>"
                break
            elif inp.type == "dict" and inp.required:
                input_hint = "<payload.json>"
                break
        print(f"    python -m src agents run {agent.name} --input {input_hint}")

    print()
    return 0


# =============================================================================
# RUN COMMAND
# =============================================================================

def cmd_agents_run(args) -> int:
    """Run a single agent with provided inputs."""
    name = args.name
    agent = get_agent(name)

    if not agent:
        print(f"\n  Agent not found: '{name}'")
        return 1

    if not agent.can_run_standalone:
        print(f"\n  Agent '{agent.label}' cannot run standalone.")
        print(f"  Reason: requires pipeline context or intermediate results.")
        print(f"  Use 'python -m src agents info {name}' for details.")
        return 1

    # ── Parse inputs ──
    input_path = Path(args.input) if args.input else None
    output_dir = Path(args.output) if args.output else Path("./out")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse --params key=value pairs
    params: Dict[str, str] = {}
    if args.params:
        for kv in args.params:
            if "=" not in kv:
                print(f"  Invalid param format: '{kv}' (expected key=value)")
                return 1
            k, v = kv.split("=", 1)
            params[k] = v

    # ── Validate input ──
    if input_path and not input_path.exists():
        print(f"  Input file not found: {input_path}")
        return 1

    # ── Load input data ──
    input_data = _load_input(input_path, agent)

    # ── Resolve function ──
    try:
        fn = resolve_fn(agent)
    except (ImportError, AttributeError) as e:
        print(f"\n  Failed to import agent '{agent.name}':")
        print(f"    {type(e).__name__}: {e}")
        print(f"  Module: {agent.module_path}.{agent.entry_fn}")
        return 1

    # ── Execute ──
    print(f"\n  Running: {agent.label} ({agent.name})")
    print(f"  Module:  {agent.module_path}.{agent.entry_fn}")
    if input_path:
        print(f"  Input:   {input_path}")
    print()

    t0 = time.perf_counter()
    try:
        kwargs = _build_kwargs(agent, input_data, input_path, params)
        result = fn(**kwargs)
        elapsed = time.perf_counter() - t0

        # ── Serialize output ──
        output_path = output_dir / f"{agent.name}_output.json"
        result_data = _serialize_result(result, agent)
        _save_output(result_data, output_path)

        print(f"  ✓ Completed in {elapsed:.2f}s")
        print(f"  Output: {output_path}")

        # Show summary
        _print_summary(result_data)
        return 0

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ Failed after {elapsed:.2f}s")
        print(f"    {type(e).__name__}: {e}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


# =============================================================================
# HELPERS
# =============================================================================

def _load_input(input_path: Optional[Path], agent: AgentSpec) -> Optional[Any]:
    """Load input file based on extension."""
    if not input_path or not input_path.exists():
        return None

    suffix = input_path.suffix.lower()

    if suffix == ".json":
        with open(input_path) as f:
            return json.load(f)
    elif suffix == ".pdf":
        return {"pdf_path": input_path}
    elif suffix in (".xlsx", ".xls"):
        return {"excel_paths": [input_path]}
    else:
        # Try JSON
        try:
            with open(input_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Treat as plain text
            with open(input_path) as f:
                return {"text": f.read()}


def _build_kwargs(
    agent: AgentSpec,
    input_data: Optional[Any],
    input_path: Optional[Path],
    params: Dict[str, str],
) -> Dict[str, Any]:
    """Build function kwargs from agent spec, input data, and CLI params."""
    kwargs: Dict[str, Any] = {}

    for inp in agent.inputs:
        # CLI --params override everything
        if inp.name in params:
            val = params[inp.name]
            # Type coercion
            if inp.type == "int":
                val = int(val)
            elif inp.type == "float":
                val = float(val)
            elif inp.type == "bool":
                val = val.lower() in ("true", "1", "yes")
            kwargs[inp.name] = val

        elif isinstance(input_data, dict) and inp.name in input_data:
            kwargs[inp.name] = input_data[inp.name]

        elif inp.type == "Path" and input_path:
            kwargs[inp.name] = input_path

        elif inp.type.startswith("List[Path]") and isinstance(input_data, dict):
            if "excel_paths" in input_data:
                kwargs[inp.name] = input_data["excel_paths"]
            elif input_path:
                kwargs[inp.name] = [input_path]

        elif inp.type == "dict" and isinstance(input_data, dict):
            kwargs[inp.name] = input_data

        elif inp.type.startswith("List") and isinstance(input_data, list):
            kwargs[inp.name] = input_data

        elif inp.type == "str" and isinstance(input_data, dict) and "text" in input_data:
            kwargs[inp.name] = input_data["text"]

        # Don't fail on missing optional params
        elif not inp.required:
            continue

    return kwargs


def _serialize_result(result: Any, agent: AgentSpec) -> Any:
    """Serialize agent output to JSON-compatible data."""
    if result is None:
        return None

    if hasattr(result, "to_dict"):
        return result.to_dict()

    if isinstance(result, (dict, list, str, int, float, bool)):
        return result

    if isinstance(result, tuple):
        data = {}
        for i, out in enumerate(agent.outputs):
            if i < len(result):
                val = result[i]
                if hasattr(val, "to_dict"):
                    data[out.name] = val.to_dict()
                elif isinstance(val, list):
                    data[out.name] = [
                        v.to_dict() if hasattr(v, "to_dict") else v
                        for v in val
                    ]
                else:
                    data[out.name] = val
        return data

    if hasattr(result, "__dict__"):
        return {
            k: v for k, v in result.__dict__.items()
            if not k.startswith("_")
        }

    return {"result": str(result)}


def _save_output(data: Any, output_path: Path) -> None:
    """Write JSON output to file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _print_summary(data: Any) -> None:
    """Print a brief summary of the output."""
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"\n  Output keys: {', '.join(keys[:10])}")
        for k in keys[:5]:
            v = data[k]
            if isinstance(v, list):
                print(f"    {k}: {len(v)} items")
            elif isinstance(v, dict):
                print(f"    {k}: {len(v)} keys")
            elif isinstance(v, str) and len(v) > 80:
                print(f"    {k}: {v[:80]}...")
            else:
                print(f"    {k}: {v}")
    elif isinstance(data, list):
        print(f"\n  Output: {len(data)} items")
    print()
