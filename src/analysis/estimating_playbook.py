"""
Estimating Playbook — company defaults, project overrides, market snapshot.

Pure functions, no Streamlit dependency.  Testable independently.

Sprint 20C: Controls bid posture, contingency, pricing guidance, and export
behaviour at company + project level.
"""

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================
# Schema Constants
# =========================================================================

RISK_POSTURES = ("aggressive", "balanced", "conservative")
ASSUMPTION_POLICIES = ("rfi_first", "assumption_first", "mixed")
COMPETITION_LEVELS = ("low", "med", "high")
MATERIAL_TRENDS = ("stable", "rising", "volatile")
LABOR_AVAILABILITY = ("normal", "tight")
LOGISTICS_DIFFICULTY = ("easy", "moderate", "hard")
WEATHER_FACTORS = ("normal", "seasonal_risk", "high_risk")


def default_playbook() -> Dict[str, Any]:
    """Return a blank/default estimating playbook."""
    return {
        "company": {
            "name": "",
            "risk_posture": "balanced",
            "default_contingency_pct": 5.0,
            "default_oh_pct": 10.0,
            "default_profit_pct": 8.0,
            "assumption_policy": "rfi_first",
            "trade_scope_defaults": {},
            "measurement_prefs": {},
            "output_prefs": {},
        },
        "project": {
            "project_type": "",
            "location": {"country": "", "state": "", "city": ""},
            "client_name": "",
            "contract_type": "",
            "bid_due_date": None,
            "must_win": False,
            "relationship_bid": False,
            "competition_intensity": "med",
            "contingency_override_pct": None,
        },
        "market_snapshot": {
            "material_trend": "stable",
            "labor_availability": "normal",
            "logistics_difficulty": "easy",
            "weather_factor": "normal",
            "steel_cost_index": None,
            "cement_cost_index": None,
            "labor_rate_factor": None,
            "freight_factor": None,
            "notes": "",
        },
        "updated_at": datetime.now().isoformat(),
    }


# =========================================================================
# Validation
# =========================================================================

def validate_playbook(playbook: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a playbook dict.  Returns (is_valid, list_of_warnings).

    Warnings are non-fatal (the playbook can still be used); errors would
    make *is_valid* = False.
    """
    warnings: List[str] = []
    if not isinstance(playbook, dict):
        return False, ["Playbook is not a dict"]

    company = playbook.get("company")
    if not isinstance(company, dict):
        return False, ["Missing or invalid 'company' section"]

    project = playbook.get("project")
    if not isinstance(project, dict):
        return False, ["Missing or invalid 'project' section"]

    market = playbook.get("market_snapshot")
    if not isinstance(market, dict):
        return False, ["Missing or invalid 'market_snapshot' section"]

    # Company checks
    rp = company.get("risk_posture", "")
    if rp and rp not in RISK_POSTURES:
        warnings.append(f"Unknown risk_posture '{rp}'; expected one of {RISK_POSTURES}")

    ap = company.get("assumption_policy", "")
    if ap and ap not in ASSUMPTION_POLICIES:
        warnings.append(f"Unknown assumption_policy '{ap}'")

    for pct_key in ("default_contingency_pct", "default_oh_pct", "default_profit_pct"):
        val = company.get(pct_key)
        if val is not None:
            try:
                float(val)
            except (TypeError, ValueError):
                warnings.append(f"company.{pct_key} is not numeric: {val}")

    # Project checks
    ci = project.get("competition_intensity", "")
    if ci and ci not in COMPETITION_LEVELS:
        warnings.append(f"Unknown competition_intensity '{ci}'")

    cov = project.get("contingency_override_pct")
    if cov is not None:
        try:
            float(cov)
        except (TypeError, ValueError):
            warnings.append(f"project.contingency_override_pct not numeric: {cov}")

    # Market checks
    mt = market.get("material_trend", "")
    if mt and mt not in MATERIAL_TRENDS:
        warnings.append(f"Unknown material_trend '{mt}'")

    la = market.get("labor_availability", "")
    if la and la not in LABOR_AVAILABILITY:
        warnings.append(f"Unknown labor_availability '{la}'")

    ld = market.get("logistics_difficulty", "")
    if ld and ld not in LOGISTICS_DIFFICULTY:
        warnings.append(f"Unknown logistics_difficulty '{ld}'")

    wf = market.get("weather_factor", "")
    if wf and wf not in WEATHER_FACTORS:
        warnings.append(f"Unknown weather_factor '{wf}'")

    return True, warnings


# =========================================================================
# Merge (company defaults + project overrides)
# =========================================================================

def merge_playbook(
    company_defaults: Dict[str, Any],
    project_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge company-level defaults with per-project overrides.

    Strategy:
        - Company section: company_defaults takes base, overrides win for
          non-None / non-empty values.
        - Project section: project_overrides takes priority.
        - Market snapshot: project_overrides wins if present.
        - updated_at: set to now.

    Both inputs should follow the playbook schema.
    """
    base = deepcopy(company_defaults) if company_defaults else default_playbook()
    over = project_overrides or {}

    # Merge company (overrides win for populated keys)
    over_company = over.get("company", {})
    if isinstance(over_company, dict):
        for k, v in over_company.items():
            if v is not None and v != "" and v != {} and v != []:
                base.setdefault("company", {})[k] = v

    # Merge project (full override)
    over_project = over.get("project", {})
    if isinstance(over_project, dict):
        for k, v in over_project.items():
            base.setdefault("project", {})[k] = v

    # Merge market (override populated keys)
    over_market = over.get("market_snapshot", {})
    if isinstance(over_market, dict):
        for k, v in over_market.items():
            if v is not None and v != "":
                base.setdefault("market_snapshot", {})[k] = v

    base["updated_at"] = datetime.now().isoformat()
    return base


# =========================================================================
# Diff (base vs current)
# =========================================================================

def diff_playbook(
    base: Dict[str, Any],
    current: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compute a list of changes between *base* (company defaults) and
    *current* (merged playbook).

    Returns a list of dicts:
        [{"section": str, "field": str, "from": Any, "to": Any}, ...]

    Only reports leaf-level value changes (ignores nested dicts/lists that
    are structurally identical).
    """
    changes: List[Dict[str, Any]] = []
    base = base or default_playbook()
    current = current or default_playbook()

    for section in ("company", "project", "market_snapshot"):
        base_sec = base.get(section, {})
        curr_sec = current.get(section, {})
        if not isinstance(base_sec, dict) or not isinstance(curr_sec, dict):
            continue

        all_keys = set(list(base_sec.keys()) + list(curr_sec.keys()))
        for key in sorted(all_keys):
            bval = base_sec.get(key)
            cval = curr_sec.get(key)
            if isinstance(bval, dict) or isinstance(cval, dict):
                # One-level deeper comparison for nested dicts (e.g. location)
                bd = bval if isinstance(bval, dict) else {}
                cd = cval if isinstance(cval, dict) else {}
                for sub_key in sorted(set(list(bd.keys()) + list(cd.keys()))):
                    if bd.get(sub_key) != cd.get(sub_key):
                        changes.append({
                            "section": section,
                            "field": f"{key}.{sub_key}",
                            "from": bd.get(sub_key),
                            "to": cd.get(sub_key),
                        })
            elif bval != cval:
                changes.append({
                    "section": section,
                    "field": key,
                    "from": bval,
                    "to": cval,
                })

    return changes


# =========================================================================
# Export summary
# =========================================================================

def summarize_playbook_for_exports(playbook: Dict[str, Any]) -> str:
    """
    Return a short markdown-friendly summary of the playbook suitable for
    inclusion in bid summary documents.
    """
    if not playbook or not isinstance(playbook, dict):
        return ""

    parts: List[str] = []
    company = playbook.get("company", {})
    project = playbook.get("project", {})
    market = playbook.get("market_snapshot", {})

    # Company posture
    name = company.get("name", "")
    posture = company.get("risk_posture", "balanced")
    parts.append(f"**Company:** {name or '(unnamed)'} — risk posture: {posture}")

    # Baselines
    cont = company.get("default_contingency_pct", "—")
    oh = company.get("default_oh_pct", "—")
    profit = company.get("default_profit_pct", "—")
    parts.append(f"**Baseline:** contingency {cont}% / OH {oh}% / profit {profit}%")

    # Project overrides
    cov = project.get("contingency_override_pct")
    if cov is not None:
        parts.append(f"**Project contingency override:** {cov}%")

    must_win = project.get("must_win")
    if must_win:
        parts.append("**Must-win bid**")

    comp = project.get("competition_intensity", "")
    if comp:
        parts.append(f"**Competition:** {comp}")

    # Market factors
    factors = []
    mt = market.get("material_trend", "stable")
    if mt != "stable":
        factors.append(f"materials {mt}")
    la = market.get("labor_availability", "normal")
    if la != "normal":
        factors.append(f"labor {la}")
    ld = market.get("logistics_difficulty", "easy")
    if ld != "easy":
        factors.append(f"logistics {ld}")
    wf = market.get("weather_factor", "normal")
    if wf != "normal":
        factors.append(f"weather {wf}")
    if factors:
        parts.append(f"**Market:** {', '.join(factors)}")

    return "\n\n".join(parts)


# =========================================================================
# Pricing adjustments based on playbook
# =========================================================================

def compute_playbook_contingency_adjustments(playbook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute contingency adjustments based on playbook factors.

    Returns a dict with:
        base_pct: float (company default contingency)
        override_pct: float | None (project override)
        market_adj_pct: float (adjustment from market factors)
        posture_adj_pct: float (adjustment from risk posture)
        recommended_pct: float (final recommendation)
        basis: list[str] (reasoning)
    """
    if not playbook or not isinstance(playbook, dict):
        return {
            "base_pct": 5.0,
            "override_pct": None,
            "market_adj_pct": 0.0,
            "posture_adj_pct": 0.0,
            "recommended_pct": 5.0,
            "basis": ["No playbook configured — using default 5%"],
        }

    company = playbook.get("company", {})
    project = playbook.get("project", {})
    market = playbook.get("market_snapshot", {})

    base = float(company.get("default_contingency_pct", 5.0) or 5.0)
    override = project.get("contingency_override_pct")
    basis: List[str] = [f"Company default contingency: {base}%"]

    # Risk posture adjustment
    posture = company.get("risk_posture", "balanced")
    posture_adj = 0.0
    if posture == "conservative":
        posture_adj = 1.5
        basis.append(f"+{posture_adj}% for conservative posture")
    elif posture == "aggressive":
        posture_adj = -1.0
        basis.append(f"{posture_adj}% for aggressive posture")

    # Market adjustments
    market_adj = 0.0
    mt = market.get("material_trend", "stable")
    if mt == "rising":
        market_adj += 1.0
        basis.append("+1.0% for rising material costs")
    elif mt == "volatile":
        market_adj += 2.0
        basis.append("+2.0% for volatile material costs")

    la = market.get("labor_availability", "normal")
    if la == "tight":
        market_adj += 1.0
        basis.append("+1.0% for tight labor market")

    ld = market.get("logistics_difficulty", "easy")
    if ld == "moderate":
        market_adj += 0.5
        basis.append("+0.5% for moderate logistics difficulty")
    elif ld == "hard":
        market_adj += 1.5
        basis.append("+1.5% for hard logistics")

    wf = market.get("weather_factor", "normal")
    if wf == "seasonal_risk":
        market_adj += 0.5
        basis.append("+0.5% for seasonal weather risk")
    elif wf == "high_risk":
        market_adj += 1.0
        basis.append("+1.0% for high weather risk")

    # Must-win / relationship adjustments
    if project.get("must_win"):
        posture_adj -= 0.5
        basis.append("-0.5% for must-win bid")

    # Override takes priority
    if override is not None:
        try:
            override = float(override)
            basis.append(f"Project override: {override}% (replaces computed)")
            recommended = override
        except (TypeError, ValueError):
            override = None
            recommended = round(base + posture_adj + market_adj, 1)
    else:
        recommended = round(base + posture_adj + market_adj, 1)

    # Floor at 1%
    recommended = max(1.0, recommended)

    return {
        "base_pct": base,
        "override_pct": override,
        "market_adj_pct": round(market_adj, 1),
        "posture_adj_pct": round(posture_adj, 1),
        "recommended_pct": recommended,
        "basis": basis,
    }
