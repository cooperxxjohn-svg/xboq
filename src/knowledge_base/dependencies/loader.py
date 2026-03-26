"""
Dependency loader — reads YAML rules and converts to existing DependencyRule format.
"""

import logging
from pathlib import Path
from typing import Dict, List

import yaml

from src.knowledge_base.dependencies.schema import EnhancedDependencyRule

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Trade name mapping: knowledge base trade names → scope_dependencies.Trade enum values
_TRADE_MAP = {
    "structural": "structural",
    "masonry": "masonry",
    "finishing": "finishing",
    "plumbing": "plumbing",
    "electrical": "electrical",
    "waterproofing": "waterproofing",
    "carpentry": "carpentry",
    "external": "external",
    "fire_safety": "fire_safety",
    "hvac": "hvac",
    "misc": "misc",
    # Extended trades map to closest existing
    "facade": "external",
    "elevator": "misc",
    "interior": "finishing",
    "prelims": "misc",
    "infrastructure": "structural",
    "mep": "electrical",
}


class DependencyLoader:
    """Loads and indexes all dependency YAML files."""

    def __init__(self):
        self._rules: List[EnhancedDependencyRule] = []
        self._loaded = False

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def load_all(self) -> None:
        if self._loaded:
            return

        if not DATA_DIR.exists():
            logger.warning("Dependency data directory not found: %s", DATA_DIR)
            self._loaded = True
            return

        for yaml_file in sorted(DATA_DIR.glob("*.yaml")):
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error("Error loading dependency file %s: %s", yaml_file.name, e)

        self._loaded = True
        logger.info("Dependencies loaded: %d rules", len(self._rules))

    def _load_file(self, filepath: Path) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        rules = data.get("rules", [])
        for rule_data in rules:
            if not isinstance(rule_data, dict):
                continue
            rule = EnhancedDependencyRule.from_dict(rule_data)
            if rule.trigger and rule.required_items:
                self._rules.append(rule)

    def all_rules(self) -> List[EnhancedDependencyRule]:
        self.load_all()
        return list(self._rules)

    def as_dependency_rules(self):
        """
        Convert to existing DependencyRule format for backward compatibility.
        Returns List[DependencyRule] from src.boq.scope_dependencies.
        """
        self.load_all()

        # Import the existing DependencyRule and Trade classes
        try:
            from src.boq.scope_dependencies import DependencyRule, Trade
        except ImportError:
            logger.error("Cannot import DependencyRule from scope_dependencies")
            return []

        # Build Trade enum lookup
        trade_enum = {t.value: t for t in Trade}

        result = []
        for rule in self._rules:
            # Map trade name to Trade enum
            mapped_trade = _TRADE_MAP.get(rule.trade, rule.trade)
            trade = trade_enum.get(mapped_trade)
            if trade is None:
                trade = trade_enum.get("misc", Trade.MISC)

            # Build condition string from enhanced fields
            condition = rule.condition
            if not condition and rule.min_floors > 0:
                condition = "multi_storey"

            dep_rule = DependencyRule(
                trigger=rule.trigger,
                required_items=rule.required_items,
                trade=trade,
                priority=rule.priority,
                condition=condition,
                note=rule.note or rule.is_code_ref,
            )
            result.append(dep_rule)

        return result
