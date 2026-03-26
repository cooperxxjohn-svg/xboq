"""
Taxonomy loader — reads all YAML data files and builds indexed lookups.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from src.knowledge_base.taxonomy.schema import TaxonomyItem

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class TaxonomyLoader:
    """Loads and indexes all taxonomy YAML files."""

    def __init__(self):
        self._items: List[TaxonomyItem] = []
        self._by_id: Dict[str, TaxonomyItem] = {}
        self._by_discipline: Dict[str, List[TaxonomyItem]] = {}
        self._by_trade: Dict[str, List[TaxonomyItem]] = {}
        self._by_category: Dict[str, List[TaxonomyItem]] = {}
        self._loaded = False

    @property
    def item_count(self) -> int:
        return len(self._items)

    def load_all(self) -> None:
        """Load all YAML files in the data directory."""
        if self._loaded:
            return

        if not DATA_DIR.exists():
            logger.warning("Taxonomy data directory not found: %s", DATA_DIR)
            self._loaded = True
            return

        yaml_files = sorted(DATA_DIR.glob("*.yaml"))
        if not yaml_files:
            logger.warning("No taxonomy YAML files found in %s", DATA_DIR)
            self._loaded = True
            return

        for yaml_file in yaml_files:
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error("Error loading taxonomy file %s: %s", yaml_file.name, e)

        self._loaded = True
        logger.info(
            "Taxonomy loaded: %d items from %d files, %d disciplines, %d trades",
            len(self._items), len(yaml_files),
            len(self._by_discipline), len(self._by_trade),
        )

    def _load_file(self, filepath: Path) -> None:
        """Load a single YAML file and parse items."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        discipline = data.get("discipline", "")
        trade = data.get("trade", "")

        # Walk the sub_trades → categories → items hierarchy
        sub_trades = data.get("sub_trades", {})
        for st_key, st_data in sub_trades.items():
            if not isinstance(st_data, dict):
                continue
            categories = st_data.get("categories", {})
            for cat_key, cat_data in categories.items():
                if not isinstance(cat_data, dict):
                    continue
                items = cat_data.get("items", [])
                for item_data in items:
                    if not isinstance(item_data, dict):
                        continue
                    item = TaxonomyItem.from_dict(
                        item_data,
                        discipline=discipline,
                        trade=trade,
                        sub_trade=st_key,
                        category=cat_key,
                    )
                    self._add_item(item)

    def _add_item(self, item: TaxonomyItem) -> None:
        """Add item to all indexes."""
        if not item.id:
            logger.warning("Taxonomy item without ID: %s", item.standard_name)
            return

        if item.id in self._by_id:
            logger.warning("Duplicate taxonomy ID: %s", item.id)
            return

        self._items.append(item)
        self._by_id[item.id] = item

        self._by_discipline.setdefault(item.discipline, []).append(item)
        self._by_trade.setdefault(item.trade, []).append(item)
        if item.category:
            self._by_category.setdefault(item.category, []).append(item)

    # ── Query API ──

    def all_items(self) -> List[TaxonomyItem]:
        self.load_all()
        return list(self._items)

    def get_by_id(self, item_id: str) -> Optional[TaxonomyItem]:
        self.load_all()
        return self._by_id.get(item_id)

    def items_by_discipline(self, discipline: str) -> List[TaxonomyItem]:
        self.load_all()
        return self._by_discipline.get(discipline, [])

    def items_by_trade(self, trade: str) -> List[TaxonomyItem]:
        self.load_all()
        return self._by_trade.get(trade, [])

    def items_by_category(self, category: str) -> List[TaxonomyItem]:
        self.load_all()
        return self._by_category.get(category, [])

    def search(self, query: str) -> List[TaxonomyItem]:
        """Search items by name or alias (case-insensitive substring match)."""
        self.load_all()
        query_lower = query.lower()
        results = []
        for item in self._items:
            if query_lower in item.standard_name.lower():
                results.append(item)
                continue
            for alias in item.aliases:
                if query_lower in alias.lower():
                    results.append(item)
                    break
        return results

    def disciplines(self) -> List[str]:
        self.load_all()
        return sorted(self._by_discipline.keys())

    def trades(self) -> List[str]:
        self.load_all()
        return sorted(self._by_trade.keys())

    def categories(self) -> List[str]:
        self.load_all()
        return sorted(self._by_category.keys())

    def validate(self) -> List[str]:
        """Validate all items: check for broken cross-references, etc."""
        self.load_all()
        errors = []
        all_ids = set(self._by_id.keys())
        for item in self._items:
            for req_id in item.requires:
                if req_id not in all_ids:
                    errors.append(
                        f"{item.id}: requires unknown ID '{req_id}'"
                    )
        return errors
