"""
Synonym loader — reads YAML data and builds reverse index for matching.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from src.knowledge_base.synonyms.schema import SynonymEntry

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class SynonymLoader:
    """Loads and indexes all synonym YAML files."""

    def __init__(self):
        self._entries: List[SynonymEntry] = []
        self._by_canonical: Dict[str, SynonymEntry] = {}
        self._reverse_index: Dict[str, str] = {}  # alias_lower → canonical
        self._loaded = False

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def load_all(self) -> None:
        if self._loaded:
            return

        if not DATA_DIR.exists():
            logger.warning("Synonym data directory not found: %s", DATA_DIR)
            self._loaded = True
            return

        for yaml_file in sorted(DATA_DIR.glob("*.yaml")):
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error("Error loading synonym file %s: %s", yaml_file.name, e)

        self._loaded = True
        logger.info(
            "Synonyms loaded: %d entries, %d reverse index entries",
            len(self._entries), len(self._reverse_index),
        )

    def _load_file(self, filepath: Path) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        filename = filepath.stem  # e.g. "hindi_terms", "abbreviations"

        # ── Standard format: top key "terms" with SynonymEntry fields ──
        terms = data.get("terms", [])
        for term_data in terms:
            if not isinstance(term_data, dict):
                continue
            # Normalize non-standard field names to SynonymEntry fields
            normalized = self._normalize_term(term_data, filename)
            entry = SynonymEntry.from_dict(normalized)
            self._add_entry(entry)

        # ── Abbreviations format: top key "abbreviations" ──
        for abbr_data in data.get("abbreviations", []):
            if not isinstance(abbr_data, dict):
                continue
            abbr_list = [abbr_data.get("abbr", "")]
            # Include variations (e.g. "S.S.R.", "s.s.r", "S S R")
            for v in abbr_data.get("variations", []):
                if v and v not in abbr_list:
                    abbr_list.append(v)
            entry = SynonymEntry(
                canonical=abbr_data.get("full_form", "").lower().replace(" ", "_"),
                taxonomy_ids=abbr_data.get("taxonomy_ids", []),
                abbreviations=abbr_list,
                formal_english=[abbr_data.get("full_form", "")],
            )
            if abbr_data.get("commonly_confused_with"):
                entry.informal_english.extend(abbr_data["commonly_confused_with"])
            self._add_entry(entry)

        # ── Brand mappings format: top key "brand_mappings" ──
        for brand_data in data.get("brand_mappings", []):
            if not isinstance(brand_data, dict):
                continue
            generic_terms = brand_data.get("generic_terms", [])
            brand_name = brand_data.get("brand", "")
            product_lines = brand_data.get("product_lines", [])
            # Create entry for the brand → generic mapping
            all_brand_names = [brand_name] if brand_name else []
            for pl in product_lines:
                if isinstance(pl, dict) and pl.get("brand_product"):
                    all_brand_names.append(pl["brand_product"])
            canonical = generic_terms[0].lower().replace(" ", "_") if generic_terms else brand_name.lower()
            entry = SynonymEntry(
                canonical=canonical,
                taxonomy_ids=brand_data.get("taxonomy_ids", []),
                brand_names=all_brand_names,
                formal_english=generic_terms,
            )
            self._add_entry(entry)

        # ── Unit aliases format: top key "units" ──
        for unit_data in data.get("units", []):
            if not isinstance(unit_data, dict):
                continue
            canonical = unit_data.get("canonical", "")
            entry = SynonymEntry(
                canonical=canonical,
                formal_english=[unit_data.get("canonical_full", "")] if unit_data.get("canonical_full") else [],
                informal_english=unit_data.get("aliases", []) + unit_data.get("unicode_aliases", []),
                hindi=unit_data.get("hindi_aliases", []),
            )
            self._add_entry(entry)

    @staticmethod
    def _normalize_term(term_data: dict, filename: str) -> dict:
        """Normalize non-standard field names to SynonymEntry fields."""
        result = dict(term_data)

        # hindi_terms.yaml: hindi_terms → hindi, phonetic_variants → informal_english
        if "hindi_terms" in term_data:
            result.setdefault("hindi", [])
            result["hindi"] = term_data.get("hindi_terms", []) + result.get("hindi", [])
            # Add phonetic variants as informal aliases
            result.setdefault("informal_english", [])
            result["informal_english"] = (
                term_data.get("phonetic_variants", [])
                + result.get("informal_english", [])
            )
            # Add regional variants
            regional = term_data.get("regional_variants", {})
            if isinstance(regional, dict):
                for region, variants in regional.items():
                    if isinstance(variants, list):
                        result["hindi"].extend(variants)
                    elif isinstance(variants, str):
                        result["hindi"].append(variants)

        # english_informal.yaml: site_terms → informal_english, contractor_shorthand → abbreviations
        if "site_terms" in term_data:
            result.setdefault("informal_english", [])
            result["informal_english"] = (
                term_data.get("site_terms", [])
                + term_data.get("common_misspellings", [])
                + result.get("informal_english", [])
            )
            result.setdefault("abbreviations", [])
            result["abbreviations"] = (
                term_data.get("contractor_shorthand", [])
                + result.get("abbreviations", [])
            )

        return result

    def _add_entry(self, entry: SynonymEntry) -> None:
        if not entry.canonical:
            return

        canonical_lower = entry.canonical.lower()

        if canonical_lower in self._by_canonical:
            # Merge into existing
            existing = self._by_canonical[canonical_lower]
            existing.formal_english.extend(entry.formal_english)
            existing.informal_english.extend(entry.informal_english)
            existing.hindi.extend(entry.hindi)
            existing.abbreviations.extend(entry.abbreviations)
            existing.brand_names.extend(entry.brand_names)
            existing.taxonomy_ids.extend(entry.taxonomy_ids)
        else:
            self._entries.append(entry)
            self._by_canonical[canonical_lower] = entry

        # Build reverse index
        for alias in entry.all_aliases:
            alias_lower = alias.lower().strip()
            if alias_lower and alias_lower not in self._reverse_index:
                self._reverse_index[alias_lower] = entry.canonical

    # ── Query API ──

    def lookup(self, term: str) -> Optional[str]:
        """Find the canonical term for any alias. Returns None if not found."""
        self.load_all()
        term_lower = term.lower().strip()
        # Direct canonical match
        if term_lower in self._by_canonical:
            return self._by_canonical[term_lower].canonical
        # Reverse index match
        return self._reverse_index.get(term_lower)

    def get_entry(self, canonical: str) -> Optional[SynonymEntry]:
        self.load_all()
        return self._by_canonical.get(canonical.lower())

    def as_flat_dict(self) -> Dict[str, List[str]]:
        """
        Return synonyms in same format as synonyms_india.INDIAN_SYNONYMS.
        Dict[canonical, List[all_aliases]]
        """
        self.load_all()
        result = {}
        for entry in self._entries:
            aliases = entry.all_aliases
            if aliases:
                result[entry.canonical] = aliases
        return result

    def trade_keywords(self) -> Dict[str, List[str]]:
        """
        Return keywords grouped by trade (for completeness_scorer.EXPECTED_TRADES).
        Uses taxonomy_ids to infer trade from item ID prefix.
        """
        self.load_all()
        trade_kw: Dict[str, List[str]] = {}

        # Map discipline prefixes to trade names
        _PREFIX_TO_TRADE = {
            "CIV": "earthwork",
            "STR": "structural",
            "MAS": "masonry",
            "WP": "waterproofing",
            "FIN.FL": "flooring",
            "FIN.WL": "plaster",
            "FIN.CL": "false_ceiling",
            "DW": "doors_windows",
            "PLB": "plumbing",
            "ELC": "electrical",
            "FIR": "fire_safety",
            "HVC": "hvac",
            "ELV": "lift",
            "EXT": "external",
            "PRL": "prelims",
            "FAC": "facade",
            "INT": "interior",
        }

        for entry in self._entries:
            trade = None
            for tid in entry.taxonomy_ids:
                for prefix, t in _PREFIX_TO_TRADE.items():
                    if tid.startswith(prefix):
                        trade = t
                        break
                if trade:
                    break

            if not trade:
                continue

            if trade not in trade_kw:
                trade_kw[trade] = []

            # Add canonical + all aliases as keywords
            trade_kw[trade].append(entry.canonical)
            trade_kw[trade].extend(entry.all_aliases)

        # Deduplicate
        for trade in trade_kw:
            trade_kw[trade] = list(dict.fromkeys(trade_kw[trade]))

        return trade_kw
