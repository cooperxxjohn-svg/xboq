"""
Estimator Inputs - YAML Configuration Layer

Allows estimators to provide inputs before/after extraction:
1. Project assumptions (wall heights, door sizes, etc.)
2. Manual overrides for specific items
3. Scope inclusions/exclusions
4. Rate overrides
5. Re-run triggers

This makes XBOQ an interactive estimator workflow tool.
"""

import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT ESTIMATOR INPUTS TEMPLATE
# =============================================================================

DEFAULT_ESTIMATOR_INPUTS = """
# XBOQ Estimator Inputs
# Edit this file to provide assumptions, overrides, and scope decisions.
# Re-run the pipeline to apply changes.

# Project Information
project:
  name: "{project_id}"
  client: ""
  location: ""
  type: "residential"  # residential, commercial, institutional, industrial

# Estimator Assumptions
# These override auto-detection
assumptions:
  # Dimensional assumptions
  wall_height_m: 3.0
  door_height_m: 2.1
  window_height_m: 1.2
  skirting_height_mm: 100
  dado_height_m: 1.2
  parapet_height_m: 1.0

  # Coverage assumptions
  plaster_both_sides: true
  floor_finish_all_rooms: true
  waterproof_wet_areas: true
  frame_all_openings: true

  # Wastage factors (%)
  wastage:
    flooring: 5
    tiles: 10
    paint: 8
    steel: 3
    concrete: 2

# Scope Decisions
# Mark items as included/excluded from estimate
scope:
  # Include these items even if not detected
  force_include:
    # - "Waterproofing to terrace"
    # - "Anti-termite treatment"

  # Exclude these items from estimate
  force_exclude:
    # - "External painting"
    # - "Landscaping"

  # Mark these as provisional/PC sums
  provisional:
    # - "Electrical works": 500000
    # - "Plumbing works": 300000

# Quantity Overrides
# Override specific extracted quantities
overrides:
  # Format: item_id or description pattern -> override value
  # Example:
  # "FLR-001": 125.5
  # "Flooring in Master Bedroom": 45.0
  # "all:flooring": 1.05  # multiply all flooring by 1.05

# Rate Overrides
# Override default rates (₹ per unit)
rates:
  # Format: item_type -> rate
  # flooring_vitrified: 85
  # wall_plaster_12mm: 35
  # cement_m25: 5500

# Bid Strategy
bid:
  # Target margin (%)
  margin: 15

  # Risk contingency (%)
  contingency: 5

  # Items to quote as lump sum
  lump_sum:
    # - "Site mobilization"
    # - "Scaffolding"

  # Items requiring re-measurement
  remeasure:
    # - "Excavation"
    # - "Painting"

# Re-run Triggers
# Set to true to trigger specific re-processing
rerun:
  force_extract: false  # Re-run extraction from drawings
  force_reconcile: true  # Re-run reconciliation with these inputs
  force_pricing: false   # Re-run pricing with rate overrides

# Notes
notes: |
  Add any notes for the estimate here.
  These will appear in the bid book.
"""


@dataclass
class WastageFactors:
    """Wastage factors for different materials."""
    flooring: float = 5.0
    tiles: float = 10.0
    paint: float = 8.0
    steel: float = 3.0
    concrete: float = 2.0

    @classmethod
    def from_dict(cls, data: Dict) -> "WastageFactors":
        return cls(
            flooring=data.get("flooring", 5.0),
            tiles=data.get("tiles", 10.0),
            paint=data.get("paint", 8.0),
            steel=data.get("steel", 3.0),
            concrete=data.get("concrete", 2.0),
        )


@dataclass
class EstimatorAssumptions:
    """Estimator assumptions from YAML."""
    wall_height_m: float = 3.0
    door_height_m: float = 2.1
    window_height_m: float = 1.2
    skirting_height_mm: int = 100
    dado_height_m: float = 1.2
    parapet_height_m: float = 1.0
    plaster_both_sides: bool = True
    floor_finish_all_rooms: bool = True
    waterproof_wet_areas: bool = True
    frame_all_openings: bool = True
    wastage: WastageFactors = field(default_factory=WastageFactors)

    @classmethod
    def from_dict(cls, data: Dict) -> "EstimatorAssumptions":
        wastage_data = data.get("wastage", {})
        return cls(
            wall_height_m=data.get("wall_height_m", 3.0),
            door_height_m=data.get("door_height_m", 2.1),
            window_height_m=data.get("window_height_m", 1.2),
            skirting_height_mm=data.get("skirting_height_mm", 100),
            dado_height_m=data.get("dado_height_m", 1.2),
            parapet_height_m=data.get("parapet_height_m", 1.0),
            plaster_both_sides=data.get("plaster_both_sides", True),
            floor_finish_all_rooms=data.get("floor_finish_all_rooms", True),
            waterproof_wet_areas=data.get("waterproof_wet_areas", True),
            frame_all_openings=data.get("frame_all_openings", True),
            wastage=WastageFactors.from_dict(wastage_data),
        )


@dataclass
class ScopeDecisions:
    """Scope inclusion/exclusion decisions."""
    force_include: List[str] = field(default_factory=list)
    force_exclude: List[str] = field(default_factory=list)
    provisional: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "ScopeDecisions":
        return cls(
            force_include=data.get("force_include") or [],
            force_exclude=data.get("force_exclude") or [],
            provisional=data.get("provisional") or {},
        )


@dataclass
class BidStrategy:
    """Bid strategy settings."""
    margin: float = 15.0
    contingency: float = 5.0
    lump_sum: List[str] = field(default_factory=list)
    remeasure: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "BidStrategy":
        return cls(
            margin=data.get("margin", 15.0),
            contingency=data.get("contingency", 5.0),
            lump_sum=data.get("lump_sum") or [],
            remeasure=data.get("remeasure") or [],
        )


@dataclass
class RerunTriggers:
    """Flags to trigger re-processing."""
    force_extract: bool = False
    force_reconcile: bool = True
    force_pricing: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "RerunTriggers":
        return cls(
            force_extract=data.get("force_extract", False),
            force_reconcile=data.get("force_reconcile", True),
            force_pricing=data.get("force_pricing", False),
        )


@dataclass
class EstimatorInputs:
    """Complete estimator inputs from YAML."""
    project_name: str = ""
    project_client: str = ""
    project_location: str = ""
    project_type: str = "residential"

    assumptions: EstimatorAssumptions = field(default_factory=EstimatorAssumptions)
    scope: ScopeDecisions = field(default_factory=ScopeDecisions)
    overrides: Dict[str, float] = field(default_factory=dict)
    rates: Dict[str, float] = field(default_factory=dict)
    bid: BidStrategy = field(default_factory=BidStrategy)
    rerun: RerunTriggers = field(default_factory=RerunTriggers)
    notes: str = ""

    # Source tracking
    yaml_path: Optional[Path] = None
    loaded_from_file: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "EstimatorInputs":
        """Load estimator inputs from YAML file."""
        if not yaml_path.exists():
            logger.info(f"No estimator inputs file at {yaml_path}")
            return cls()

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}

            project = data.get("project", {})
            assumptions_data = data.get("assumptions", {})
            scope_data = data.get("scope", {})
            bid_data = data.get("bid", {})
            rerun_data = data.get("rerun", {})

            inputs = cls(
                project_name=project.get("name", ""),
                project_client=project.get("client", ""),
                project_location=project.get("location", ""),
                project_type=project.get("type", "residential"),
                assumptions=EstimatorAssumptions.from_dict(assumptions_data),
                scope=ScopeDecisions.from_dict(scope_data),
                overrides=data.get("overrides") or {},
                rates=data.get("rates") or {},
                bid=BidStrategy.from_dict(bid_data),
                rerun=RerunTriggers.from_dict(rerun_data),
                notes=data.get("notes", ""),
                yaml_path=yaml_path,
                loaded_from_file=True,
            )

            logger.info(f"Loaded estimator inputs from {yaml_path}")
            logger.info(f"  - Overrides: {len(inputs.overrides)}")
            logger.info(f"  - Rate overrides: {len(inputs.rates)}")
            logger.info(f"  - Force include: {len(inputs.scope.force_include)}")
            logger.info(f"  - Force exclude: {len(inputs.scope.force_exclude)}")

            return inputs

        except Exception as e:
            logger.error(f"Error loading estimator inputs: {e}")
            return cls()

    def apply_override(self, item_id: str, description: str, qty: float) -> float:
        """Apply override to a quantity if one exists."""
        # Check exact item_id match
        if item_id in self.overrides:
            return self.overrides[item_id]

        # Check description pattern match
        desc_lower = description.lower()
        for pattern, override_val in self.overrides.items():
            pattern_lower = pattern.lower()

            # Check for "all:" prefix (multiplier)
            if pattern_lower.startswith("all:"):
                item_type = pattern_lower[4:]
                if item_type in desc_lower:
                    return qty * override_val

            # Check for partial match
            if pattern_lower in desc_lower:
                return override_val

        return qty

    def get_rate(self, item_type: str, default_rate: float = 0) -> float:
        """Get rate for item type, with override if exists."""
        item_lower = item_type.lower().replace(" ", "_")

        for rate_key, rate_val in self.rates.items():
            if rate_key.lower().replace(" ", "_") == item_lower:
                return rate_val

        return default_rate

    def should_include(self, description: str) -> bool:
        """Check if item should be included (not excluded)."""
        desc_lower = description.lower()

        for exclude in self.scope.force_exclude:
            if exclude.lower() in desc_lower:
                return False

        return True

    def is_forced_include(self, description: str) -> bool:
        """Check if item is force-included."""
        desc_lower = description.lower()

        for include in self.scope.force_include:
            if include.lower() in desc_lower:
                return True

        return False

    def get_provisional_amount(self, description: str) -> Optional[float]:
        """Get provisional amount if item is marked as provisional."""
        desc_lower = description.lower()

        for prov_item, amount in self.scope.provisional.items():
            if prov_item.lower() in desc_lower:
                return amount

        return None

    def apply_wastage(self, item_type: str, qty: float) -> float:
        """Apply wastage factor to quantity."""
        item_lower = item_type.lower()

        if "floor" in item_lower or "tile" in item_lower:
            if "tile" in item_lower:
                return qty * (1 + self.assumptions.wastage.tiles / 100)
            return qty * (1 + self.assumptions.wastage.flooring / 100)

        if "paint" in item_lower:
            return qty * (1 + self.assumptions.wastage.paint / 100)

        if "steel" in item_lower or "rebar" in item_lower:
            return qty * (1 + self.assumptions.wastage.steel / 100)

        if "concrete" in item_lower or "rcc" in item_lower:
            return qty * (1 + self.assumptions.wastage.concrete / 100)

        return qty


def create_default_inputs_yaml(output_dir: Path, project_id: str) -> Path:
    """Create default estimator_inputs.yaml in project directory."""
    yaml_path = output_dir / "estimator_inputs.yaml"

    if yaml_path.exists():
        logger.info(f"estimator_inputs.yaml already exists at {yaml_path}")
        return yaml_path

    content = DEFAULT_ESTIMATOR_INPUTS.format(project_id=project_id)

    with open(yaml_path, "w") as f:
        f.write(content)

    logger.info(f"Created default estimator_inputs.yaml at {yaml_path}")
    return yaml_path


def load_estimator_inputs(output_dir: Path, project_id: str = "") -> EstimatorInputs:
    """Load estimator inputs, creating default if needed."""
    yaml_path = output_dir / "estimator_inputs.yaml"

    # Create default if doesn't exist
    if not yaml_path.exists():
        create_default_inputs_yaml(output_dir, project_id)

    return EstimatorInputs.from_yaml(yaml_path)
