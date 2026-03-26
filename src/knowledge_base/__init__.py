"""
Master Construction Knowledge Base for xBOQ.ai

4-layer architecture:
  Layer 1: Taxonomy   — 6,000+ construction items (21 disciplines)
  Layer 2: Synonyms   — 3,000+ aliases (English, Hindi, abbreviations, slang)
  Layer 3: Dependencies — 500+ rules (building-type, room-type, code-aware)
  Layer 4: RFI Rules  — 200+ rules generating specific RFI questions

Public API — all functions return data compatible with existing consumers.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_taxonomy_cache = None
_synonym_cache = None
_dependency_cache = None
_rfi_cache = None


# ── Layer 1: Taxonomy ──

def get_taxonomy():
    """Get the full taxonomy (lazy-loaded singleton)."""
    global _taxonomy_cache
    if _taxonomy_cache is None:
        from src.knowledge_base.taxonomy.loader import TaxonomyLoader
        _taxonomy_cache = TaxonomyLoader()
        _taxonomy_cache.load_all()
        logger.info("Knowledge base taxonomy loaded: %d items", _taxonomy_cache.item_count)
    return _taxonomy_cache


def get_taxonomy_items():
    """Get all taxonomy items as a flat list."""
    return get_taxonomy().all_items()


def get_taxonomy_items_by_discipline(discipline: str):
    """Get taxonomy items filtered by discipline."""
    return get_taxonomy().items_by_discipline(discipline)


def get_taxonomy_items_by_trade(trade: str):
    """Get taxonomy items filtered by trade."""
    return get_taxonomy().items_by_trade(trade)


# ── Layer 2: Synonyms ──

def get_synonym_engine():
    """Get the synonym engine (lazy-loaded singleton)."""
    global _synonym_cache
    if _synonym_cache is None:
        from src.knowledge_base.synonyms.loader import SynonymLoader
        _synonym_cache = SynonymLoader()
        _synonym_cache.load_all()
        logger.info("Knowledge base synonyms loaded: %d entries", _synonym_cache.entry_count)
    return _synonym_cache


def get_all_synonyms() -> Dict[str, List[str]]:
    """
    Get all synonyms in the SAME format as synonyms_india.INDIAN_SYNONYMS.
    Returns: Dict[canonical_term, List[aliases]]
    """
    return get_synonym_engine().as_flat_dict()


def get_trade_keywords() -> Dict[str, List[str]]:
    """
    Get expanded trade keywords for completeness_scorer.EXPECTED_TRADES.
    Returns: Dict[trade_name, List[keywords]]
    """
    return get_synonym_engine().trade_keywords()


# ── Layer 3: Dependencies ──

def get_dependency_engine():
    """Get the dependency engine (lazy-loaded singleton)."""
    global _dependency_cache
    if _dependency_cache is None:
        from src.knowledge_base.dependencies.loader import DependencyLoader
        _dependency_cache = DependencyLoader()
        _dependency_cache.load_all()
        logger.info("Knowledge base dependencies loaded: %d rules", _dependency_cache.rule_count)
    return _dependency_cache


def get_all_dependency_rules():
    """
    Get all dependency rules in the SAME format as scope_dependencies.DependencyRule.
    Returns: List[DependencyRule] — compatible with existing analyze_scope_gaps().
    """
    return get_dependency_engine().as_dependency_rules()


# ── Layer 4: RFI Rules ──

def get_rfi_engine():
    """Get the RFI rule engine (lazy-loaded singleton)."""
    global _rfi_cache
    if _rfi_cache is None:
        from src.knowledge_base.rfi_rules.loader import RFIRuleLoader
        _rfi_cache = RFIRuleLoader()
        _rfi_cache.load_all()
        logger.info("Knowledge base RFI rules loaded: %d rules", _rfi_cache.rule_count)
    return _rfi_cache


def get_rfi_rules():
    """Get all RFI rules."""
    return get_rfi_engine().all_rules()


# ── Utilities ──

def get_stats() -> Dict[str, int]:
    """Get knowledge base statistics."""
    stats = {}
    try:
        stats["taxonomy_items"] = get_taxonomy().item_count
    except Exception:
        stats["taxonomy_items"] = 0
    try:
        stats["synonym_entries"] = get_synonym_engine().entry_count
    except Exception:
        stats["synonym_entries"] = 0
    try:
        stats["dependency_rules"] = get_dependency_engine().rule_count
    except Exception:
        stats["dependency_rules"] = 0
    try:
        stats["rfi_rules"] = get_rfi_engine().rule_count
    except Exception:
        stats["rfi_rules"] = 0
    return stats


def clear_cache():
    """Clear all cached data (useful for testing/reloading)."""
    global _taxonomy_cache, _synonym_cache, _dependency_cache, _rfi_cache
    _taxonomy_cache = None
    _synonym_cache = None
    _dependency_cache = None
    _rfi_cache = None
