"""
RFI rule loader — reads YAML rules and matches scope gaps to RFI questions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.knowledge_base.rfi_rules.schema import RFIRule

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

_PRIORITY_MAP = {
    "critical": "CRITICAL",
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


class RFIRuleLoader:
    """Loads and indexes all RFI rule YAML files."""

    def __init__(self):
        self._rules: List[RFIRule] = []
        self._by_trade: Dict[str, List[RFIRule]] = {}
        self._by_type: Dict[str, List[RFIRule]] = {}
        self._loaded = False

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def load_all(self) -> None:
        if self._loaded:
            return

        if not DATA_DIR.exists():
            logger.warning("RFI rules data directory not found: %s", DATA_DIR)
            self._loaded = True
            return

        for yaml_file in sorted(DATA_DIR.glob("*.yaml")):
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error("Error loading RFI rule file %s: %s", yaml_file.name, e)

        self._loaded = True
        logger.info("RFI rules loaded: %d rules", len(self._rules))

    def _load_file(self, filepath: Path) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        rules = data.get("rules", [])
        for rule_data in rules:
            if not isinstance(rule_data, dict):
                continue
            rule = RFIRule.from_dict(rule_data)
            if rule.question_template:
                self._rules.append(rule)
                self._by_trade.setdefault(rule.trigger_trade, []).append(rule)
                self._by_type.setdefault(rule.trigger_type, []).append(rule)

    def all_rules(self) -> List[RFIRule]:
        self.load_all()
        return list(self._rules)

    def rules_by_trade(self, trade: str) -> List[RFIRule]:
        self.load_all()
        return self._by_trade.get(trade, [])

    def rules_by_type(self, trigger_type: str) -> List[RFIRule]:
        self.load_all()
        return self._by_type.get(trigger_type, [])

    def match_scope_gap(
        self,
        missing_item: str,
        trade: str = "",
        building_type: str = "all",
    ) -> List[RFIRule]:
        """Find RFI rules that match a specific scope gap."""
        self.load_all()
        missing_lower = missing_item.lower()
        matches = []

        for rule in self._rules:
            # Check trigger keywords
            if not rule.trigger_keywords:
                continue

            keyword_match = any(
                kw.lower() in missing_lower or missing_lower in kw.lower()
                for kw in rule.trigger_keywords
            )
            if not keyword_match:
                continue

            # Check building type
            if "all" not in rule.building_types and building_type not in rule.building_types:
                continue

            # Check trade
            if rule.trigger_trade and trade and rule.trigger_trade != trade:
                continue

            matches.append(rule)

        return matches


def match_gaps_to_rfis(
    scope_gaps: List[Dict[str, Any]],
    boq_items: List[Dict[str, Any]],
    project_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Match scope gaps to RFI rules and generate RFI items.

    Returns List[dict] in RFIItem-compatible format for pipeline integration.
    """
    loader = RFIRuleLoader()
    loader.load_all()

    if loader.rule_count == 0:
        return []

    building_type = project_params.get("building_type", "all")
    rfis = []
    rfi_counter = 900  # Start at 900 to avoid collision with pipeline RFIs

    for gap in scope_gaps:
        missing = gap.get("missing_item", "")
        trade = gap.get("trade", "")

        matched_rules = loader.match_scope_gap(missing, trade, building_type)
        for rule in matched_rules:
            rfi_counter += 1
            question = rule.question_template
            try:
                question = question.format(
                    area=missing,
                    item=missing,
                    rooms=missing,
                    count=1,
                    room_list=missing,
                    trade=trade,
                )
            except (KeyError, IndexError):
                pass  # Template has placeholders we can't fill — use as-is

            rfis.append({
                "id": f"RFI-KB-{rfi_counter:04d}",
                "trade": trade or rule.trigger_trade or "general",
                "priority": _PRIORITY_MAP.get(rule.priority, "MEDIUM"),
                "question": question.strip(),
                "why_it_matters": rule.why_it_matters.strip(),
                "suggested_resolution": rule.suggested_resolution.strip(),
                "issue_type": rule.id,
                "package": rule.package or trade,
                "coverage_status": "not_found_after_search",
                "evidence": {
                    "pages": [],
                    "sheets": [],
                    "snippets": [f"Scope gap: {missing} not found in BOQ"],
                    "detected_entities": {},
                    "search_attempts": {},
                    "confidence": 0.7,
                    "confidence_reason": "Knowledge base rule match",
                },
            })

    # Deduplicate by question similarity (avoid 3 waterproofing RFIs)
    seen_rules = set()
    deduped = []
    for rfi in rfis:
        rule_key = rfi["issue_type"]
        if rule_key not in seen_rules:
            seen_rules.add(rule_key)
            deduped.append(rfi)

    return deduped
