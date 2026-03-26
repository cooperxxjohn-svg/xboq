"""
BOQ Item Deduplicator

Handles deduplication of BOQ (Bill of Quantities) line items extracted from
multi-page Indian construction tender documents.

Problem context:
    PDF extraction often yields the same BOQ item multiple times because:
    - A single item description spans two pages (continuation rows)
    - Carried-forward rows repeat totals from the previous page
    - OCR or table extraction captures the same row twice with slight
      wording differences

This module provides stdlib-only fuzzy matching and merge logic that
respects Indian BOQ conventions (common prefixes like "Providing and laying",
unit abbreviation variants, concrete grade notation, room-scoped items).

Key safety rules:
    - Items with **different concrete grades** are never merged (M20 != M25).
    - Items in **different rooms / locations** are never merged (they
      represent separate scope even if the description matches).
    - Merging always preserves the record with the most complete data.

Usage::

    from src.boq.deduplicator import deduplicate_boq, DeduplicationResult

    result: DeduplicationResult = deduplicate_boq(raw_items)
    clean_items = result.items
    print(result.original_count, "->", result.deduplicated_count)
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Common Indian BOQ description prefixes (order matters: longer first so
# "providing and" is tried before "providing").
_BOQ_PREFIXES: List[str] = [
    "providing and",
    "providing",
    "supplying and",
    "supplying",
    "fixing and",
    "fixing",
    "laying and",
    "laying",
    "constructing",
    "applying",
]

# Pre-compiled regex that matches any prefix at the start of a string.
# The pattern is built with longest-first ordering and word boundary at end.
_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in _BOQ_PREFIXES) + r")\b\s*",
    re.IGNORECASE,
)

# --- Unit normalization map --------------------------------------------------
# Maps variant spellings (lowercase) to a canonical form.
_UNIT_NORMALIZE: Dict[str, str] = {
    # Square metres
    "sqm": "sqm",
    "sq.m": "sqm",
    "sq. m": "sqm",
    "sq m": "sqm",
    "sq.m.": "sqm",
    "sq.mt": "sqm",
    "sq mt": "sqm",
    "sqmt": "sqm",
    "m2": "sqm",
    "m²": "sqm",
    # Cubic metres
    "cum": "cum",
    "cu.m": "cum",
    "cu. m": "cum",
    "cu m": "cum",
    "cu.m.": "cum",
    "cu.mt": "cum",
    "cu mt": "cum",
    "cumt": "cum",
    "m3": "cum",
    "m³": "cum",
    # Running metres
    "rmt": "rmt",
    "r.m.": "rmt",
    "r.m": "rmt",
    "rm": "rmt",
    "r m": "rmt",
    "r. m.": "rmt",
    # Numbers / pieces
    "nos": "no",
    "nos.": "no",
    "no": "no",
    "no.": "no",
    "nos ": "no",
    "each": "no",
    # Kilograms
    "kg": "kg",
    "kgs": "kg",
    "kgs.": "kg",
    "kg.": "kg",
    # Metric tonnes
    "mt": "MT",
    "MT": "MT",
    "tonne": "MT",
    "tonnes": "MT",
    "t": "MT",
    # Lump sum
    "ls": "LS",
    "LS": "LS",
    "lumpsum": "LS",
    "lump sum": "LS",
    # Litres
    "ltr": "ltr",
    "ltrs": "ltr",
    "litre": "ltr",
    "litres": "ltr",
    "liter": "ltr",
    "liters": "ltr",
}

# Regex for concrete grade variants: M-20, m20, M 20, m-20 etc.
_CONCRETE_GRADE_RE = re.compile(
    r"\b[Mm]\s*[-]?\s*(\d{2,3})\b"
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_unit(raw_unit: str) -> str:
    """Normalize a unit string to its canonical form.

    Args:
        raw_unit: Raw unit text from extraction (e.g. "sq.m", "Nos.", "cum").

    Returns:
        Canonical unit string.  Returns the stripped lowercase input if no
        mapping is found so that unknown units still compare correctly.
    """
    if not raw_unit:
        return ""
    cleaned = raw_unit.strip().lower().rstrip(".")
    # Try exact match first, then with trailing dot stripped
    canonical = _UNIT_NORMALIZE.get(cleaned)
    if canonical:
        return canonical
    # Try original (with dot) too
    canonical = _UNIT_NORMALIZE.get(raw_unit.strip().lower())
    if canonical:
        return canonical
    return cleaned


def _normalize_concrete_grade(text: str) -> str:
    """Normalize concrete grade mentions inside a description.

    Converts all variants (m-20, M 20, m20, M-20) to the canonical
    uppercase-no-dash form (M20).

    Args:
        text: Description text (expected to be already lowercased).

    Returns:
        Text with normalized concrete grades.
    """
    def _replace(m: re.Match) -> str:
        return f"M{m.group(1)}"
    return _CONCRETE_GRADE_RE.sub(_replace, text)


def _extract_concrete_grade(text: str) -> Optional[str]:
    """Extract the first concrete grade mentioned in *text*.

    Args:
        text: Raw or normalized description.

    Returns:
        Canonical grade string like ``"M20"`` or ``None`` if none found.
    """
    m = _CONCRETE_GRADE_RE.search(text)
    if m:
        return f"M{m.group(1)}"
    return None


def normalize_description(desc: str) -> str:
    """Normalize a BOQ description for comparison purposes.

    Steps:
        1. Lowercase.
        2. Collapse all whitespace to single spaces and strip.
        3. Remove common Indian BOQ prefixes ("providing and", "supplying",
           "laying and", etc.).
        4. Normalize unit abbreviations embedded in the text.
        5. Normalize concrete grade notation (m-20 -> M20).
        6. Strip trailing punctuation.

    Args:
        desc: Raw description string.

    Returns:
        Cleaned, normalized description suitable for similarity comparison.
    """
    if not desc:
        return ""

    text = desc.lower().strip()

    # Collapse whitespace (tabs, newlines, multiple spaces)
    text = re.sub(r"\s+", " ", text)

    # Remove common BOQ prefixes (apply iteratively to handle chains
    # like "providing and laying ..." -> strip "providing and" then "laying")
    prev = None
    while prev != text:
        prev = text
        text = _PREFIX_RE.sub("", text).strip()

    # Normalize unit abbreviations inside the description
    for variant, canonical in _UNIT_NORMALIZE.items():
        # Only replace whole-word occurrences (avoid mangling substrings)
        pattern = re.compile(r"\b" + re.escape(variant) + r"\b", re.IGNORECASE)
        text = pattern.sub(canonical, text)

    # Normalize concrete grades
    text = _normalize_concrete_grade(text)

    # Strip trailing punctuation
    text = text.rstrip(string.punctuation + " ")

    return text


def normalize_item_no(raw: str) -> str:
    """Normalize an item number for grouping.

    Strips whitespace, leading zeros, dots, dashes, and lowercases so that
    ``"01.01"``, ``"1.1"``, ``"1.01"`` and ``" 1.1 "`` all map to the same
    key.

    Args:
        raw: Raw item number string.

    Returns:
        Canonical item number string.
    """
    if not raw:
        return ""
    cleaned = raw.strip().lower()
    # Remove leading/trailing dots and dashes
    cleaned = cleaned.strip(".-")
    # Normalize numeric segments: strip leading zeros in each segment
    # e.g. "01.02.03" -> "1.2.3"
    parts = re.split(r"[.\-/]", cleaned)
    normalized_parts = []
    for part in parts:
        part = part.strip()
        if part.isdigit():
            part = str(int(part))  # strips leading zeros
        normalized_parts.append(part)
    return ".".join(p for p in normalized_parts if p)


# ---------------------------------------------------------------------------
# Token-based similarity (stdlib only, no fuzzywuzzy)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Set[str]:
    """Split text into a set of lowercase word tokens.

    Non-alphanumeric characters are treated as separators.  Very short
    tokens (single characters that are not digits) are discarded to reduce
    noise from punctuation remnants.

    Args:
        text: Input text (ideally already normalized).

    Returns:
        Set of word tokens.
    """
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    # Discard single-character non-digit tokens (e.g. leftover "x", "a")
    return {t for t in tokens if len(t) > 1 or t.isdigit()}


def token_similarity(desc1: str, desc2: str) -> float:
    """Compute Jaccard token similarity between two descriptions.

    .. math::

        J(A, B) = |A \\cap B| / |A \\cup B|

    Both inputs are tokenized after normalization.  Returns 1.0 for
    identical token sets and 0.0 when no tokens overlap.

    Args:
        desc1: First description (raw or normalized).
        desc2: Second description (raw or normalized).

    Returns:
        Float in [0.0, 1.0].
    """
    tokens_a = _tokenize(normalize_description(desc1))
    tokens_b = _tokenize(normalize_description(desc2))

    if not tokens_a and not tokens_b:
        return 1.0  # Both empty -> identical
    if not tokens_a or not tokens_b:
        return 0.0  # One empty -> no similarity

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _extract_location(item: Dict) -> Optional[str]:
    """Extract a location / room identifier from an item dict.

    Checks ``room_id``, ``room_name``, ``location``, and ``element_id``
    fields (case-insensitive keys).

    Args:
        item: BOQ item dictionary.

    Returns:
        Lowercase location string, or ``None`` if no location data.
    """
    for key in ("room_id", "room_name", "location"):
        val = item.get(key)
        if val:
            return str(val).strip().lower()
    return None


def is_likely_duplicate(
    item1: Dict,
    item2: Dict,
    threshold: float = 0.75,
) -> bool:
    """Determine whether two BOQ items are likely duplicates.

    Decision rules (evaluated in order):

    1. **Different concrete grades** -> never a duplicate.
    2. **Different rooms / locations** -> never a duplicate (separate scope).
    3. **Same normalized item_no + description similarity >= *threshold***
       -> definite duplicate.
    4. **Different item_no but similarity > 0.85 and same unit**
       -> probable duplicate.

    Args:
        item1: First BOQ item dict.
        item2: Second BOQ item dict.
        threshold: Minimum similarity for same-item-no matches (default 0.75).

    Returns:
        ``True`` if the items should be considered duplicates.
    """
    desc1 = item1.get("description", "")
    desc2 = item2.get("description", "")

    # --- Safety: different concrete grades are NEVER duplicates ---
    grade1 = _extract_concrete_grade(desc1)
    grade2 = _extract_concrete_grade(desc2)
    if grade1 and grade2 and grade1 != grade2:
        return False

    # --- Safety: different locations / rooms are NEVER duplicates ---
    loc1 = _extract_location(item1)
    loc2 = _extract_location(item2)
    if loc1 and loc2 and loc1 != loc2:
        return False

    sim = token_similarity(desc1, desc2)

    # --- Same item_no path ---
    item_no1 = normalize_item_no(item1.get("item_no", "") or item1.get("item_code", ""))
    item_no2 = normalize_item_no(item2.get("item_no", "") or item2.get("item_code", ""))

    if item_no1 and item_no2 and item_no1 == item_no2:
        return sim >= threshold

    # --- Different item_no but very similar description + same unit ---
    unit1 = normalize_unit(item1.get("unit", ""))
    unit2 = normalize_unit(item2.get("unit", ""))
    if unit1 and unit2 and unit1 == unit2 and sim > 0.85:
        return True

    return False


# ---------------------------------------------------------------------------
# Merging strategy
# ---------------------------------------------------------------------------

def _completeness_score(item: Dict) -> int:
    """Score how "complete" an item dict is.

    A higher score means more fields are populated, so this item is
    preferred as the merge target.

    Counts: qty, unit, rate, amount, description length.

    Args:
        item: BOQ item dict.

    Returns:
        Integer completeness score (higher = more complete).
    """
    score = 0
    if item.get("qty") is not None and item.get("qty") != 0:
        score += 3
    if item.get("unit"):
        score += 2
    if item.get("rate") is not None and item.get("rate") != 0:
        score += 3
    if item.get("amount") is not None and item.get("amount") != 0:
        score += 2
    if item.get("description"):
        score += min(len(item["description"]) // 20, 5)  # up to 5 for long desc
    # Bonus for having a page reference
    if item.get("page") is not None or item.get("source_page") is not None:
        score += 1
    return score


def _source_pages(item: Dict) -> List[int]:
    """Extract all page references from an item dict.

    Looks at ``page``, ``source_page``, ``source_pages``, and
    ``merged_from`` (if the item was previously merged).

    Args:
        item: BOQ item dict.

    Returns:
        Sorted list of unique page numbers.
    """
    pages: Set[int] = set()
    for key in ("page", "source_page"):
        val = item.get(key)
        if val is not None:
            try:
                pages.add(int(val))
            except (ValueError, TypeError):
                pass
    sp = item.get("source_pages")
    if isinstance(sp, (list, tuple)):
        for p in sp:
            try:
                pages.add(int(p))
            except (ValueError, TypeError):
                pass
    # Include pages from previous merges
    mf = item.get("merged_from")
    if isinstance(mf, (list, tuple)):
        for p in mf:
            try:
                pages.add(int(p))
            except (ValueError, TypeError):
                pass
    return sorted(pages)


def _merge_two(primary: Dict, secondary: Dict, similarity: float) -> Dict:
    """Merge two duplicate items into one.

    The *primary* item is the one with the higher completeness score.  Fields
    from *secondary* fill in any blanks on *primary*.

    Special handling:
        - ``merged_from``: union of source page numbers from both items.
        - ``dedup_confidence``: the token similarity score.
        - ``qty``: kept from the item that has rate populated (more likely to
          be the authoritative row).  If both have rate, keep primary's qty.

    Args:
        primary: More-complete item (kept as base).
        secondary: Less-complete item (fills gaps).
        similarity: Token similarity between the two descriptions.

    Returns:
        New merged dict.
    """
    merged = dict(primary)

    # Fill in missing fields from secondary
    for key, value in secondary.items():
        if key in ("merged_from", "dedup_confidence"):
            continue  # handled below
        if merged.get(key) in (None, "", 0, 0.0) and value not in (None, "", 0, 0.0):
            merged[key] = value

    # Provenance: track source pages
    all_pages = set(_source_pages(primary)) | set(_source_pages(secondary))
    merged["merged_from"] = sorted(all_pages) if all_pages else []

    # Confidence of the merge itself
    merged["dedup_confidence"] = round(similarity, 4)

    return merged


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def find_duplicate_clusters(
    items: List[Dict],
    threshold: float = 0.75,
) -> List[List[Dict]]:
    """Find groups of items that are likely duplicates of each other.

    Uses a simple greedy clustering approach:
        1. Group items by normalized item_no.
        2. Within each group, build clusters where every pair exceeds the
           similarity threshold (single-linkage).
        3. Also scan for cross-group near-duplicates (different item_no but
           >85% similar with same unit).

    Items that have no duplicates are **not** included in the output.

    Args:
        items: List of BOQ item dicts.
        threshold: Similarity threshold for same-item-no matches.

    Returns:
        List of clusters, where each cluster is a list of 2+ item dicts
        that should be reviewed for merger.
    """
    if not items:
        return []

    # Track which items have been assigned to a cluster
    assigned: Set[int] = set()
    clusters: List[List[int]] = []  # lists of indices

    # --- Phase 1: group by normalized item_no ---
    item_no_groups: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        raw_no = item.get("item_no", "") or item.get("item_code", "")
        norm_no = normalize_item_no(raw_no)
        if norm_no:
            item_no_groups.setdefault(norm_no, []).append(idx)

    for _no, indices in item_no_groups.items():
        if len(indices) < 2:
            continue
        # Build sub-clusters within this item_no group
        sub_clusters = _cluster_indices(items, indices, threshold)
        for sc in sub_clusters:
            if len(sc) >= 2:
                clusters.append(sc)
                assigned.update(sc)

    # --- Phase 2: cross-group near-duplicates ---
    unassigned = [i for i in range(len(items)) if i not in assigned]
    if len(unassigned) > 1:
        cross_clusters = _cluster_indices(items, unassigned, 0.85, require_same_unit=True)
        for cc in cross_clusters:
            if len(cc) >= 2:
                clusters.append(cc)

    # Convert index clusters to item clusters
    return [[items[i] for i in cluster] for cluster in clusters]


def _cluster_indices(
    items: List[Dict],
    indices: List[int],
    threshold: float,
    require_same_unit: bool = False,
) -> List[List[int]]:
    """Single-linkage cluster a subset of items by similarity.

    Args:
        items: Full items list (indexed by *indices*).
        indices: Subset of indices to cluster.
        threshold: Minimum similarity to link two items.
        require_same_unit: If True, only link items with the same unit.

    Returns:
        List of clusters (each a list of indices from *indices*).
    """
    # Parent array for union-find
    parent: Dict[int, int] = {i: i for i in indices}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i, idx_a in enumerate(indices):
        for idx_b in indices[i + 1:]:
            if is_likely_duplicate(items[idx_a], items[idx_b], threshold=threshold):
                if require_same_unit:
                    u1 = normalize_unit(items[idx_a].get("unit", ""))
                    u2 = normalize_unit(items[idx_b].get("unit", ""))
                    if u1 != u2:
                        continue
                _union(idx_a, idx_b)

    # Collect clusters
    groups: Dict[int, List[int]] = {}
    for idx in indices:
        root = _find(idx)
        groups.setdefault(root, []).append(idx)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class MergeLogEntry:
    """Record of a single merge operation."""

    kept_description: str
    merged_descriptions: List[str]
    source_pages: List[int]
    similarity: float
    item_no: str


@dataclass
class DeduplicationResult:
    """Result of the deduplication process.

    Attributes:
        items: The deduplicated list of BOQ item dicts.
        original_count: Number of items before deduplication.
        deduplicated_count: Number of items after deduplication.
        duplicates_found: Number of duplicate items that were merged away.
        merge_log: Detailed log of each merge operation.
    """

    items: List[Dict] = field(default_factory=list)
    original_count: int = 0
    deduplicated_count: int = 0
    duplicates_found: int = 0
    merge_log: List[MergeLogEntry] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        """Percentage of items removed by deduplication."""
        if self.original_count == 0:
            return 0.0
        return round(
            100.0 * self.duplicates_found / self.original_count, 1
        )

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Deduplication: {self.original_count} -> {self.deduplicated_count} items "
            f"({self.duplicates_found} duplicates removed, {self.reduction_pct}% reduction)"
        )


def deduplicate_boq(
    items: List[Dict],
    threshold: float = 0.75,
) -> DeduplicationResult:
    """Deduplicate a list of BOQ item dicts.

    Algorithm:
        1. Group items by normalized ``item_no`` (or ``item_code``).
        2. Within each group, iteratively merge items that pass
           ``is_likely_duplicate``.
        3. Scan remaining ungrouped items for cross-group near-duplicates
           (>85% similarity + same unit).
        4. Collect singletons and merged results.

    When merging, the record with the highest ``_completeness_score`` is kept
    as the primary and missing fields are filled from the secondary.  The
    ``merged_from`` field tracks source page numbers and ``dedup_confidence``
    records the similarity score.

    Args:
        items: List of BOQ item dicts as produced by extraction.
        threshold: Similarity threshold for same-item-no matches (default 0.75).

    Returns:
        ``DeduplicationResult`` with deduplicated items and statistics.
    """
    if not items:
        return DeduplicationResult(
            items=[],
            original_count=0,
            deduplicated_count=0,
            duplicates_found=0,
        )

    original_count = len(items)
    merge_log: List[MergeLogEntry] = []

    # --- Phase 1: group by normalized item_no ---
    item_no_groups: Dict[str, List[int]] = {}
    no_item_no: List[int] = []

    for idx, item in enumerate(items):
        raw_no = item.get("item_no", "") or item.get("item_code", "")
        norm_no = normalize_item_no(raw_no)
        if norm_no:
            item_no_groups.setdefault(norm_no, []).append(idx)
        else:
            no_item_no.append(idx)

    # Working copy so we can mark consumed items
    consumed: Set[int] = set()
    result_items: List[Dict] = []

    # --- Phase 2: merge within each item_no group ---
    for norm_no, indices in item_no_groups.items():
        group_items = [(idx, items[idx]) for idx in indices]

        merged_in_group = _merge_group(group_items, threshold)

        for merged_item, merged_indices, sim, merged_descs in merged_in_group:
            result_items.append(merged_item)
            consumed.update(merged_indices)
            if len(merged_indices) > 1:
                merge_log.append(MergeLogEntry(
                    kept_description=merged_item.get("description", ""),
                    merged_descriptions=merged_descs,
                    source_pages=merged_item.get("merged_from", []),
                    similarity=sim,
                    item_no=norm_no,
                ))

    # --- Phase 3: cross-group dedup for items not yet consumed ---
    remaining_indices = [
        i for i in range(len(items))
        if i not in consumed
    ]

    if remaining_indices:
        remaining_items = [(idx, items[idx]) for idx in remaining_indices]
        cross_merged = _merge_group(remaining_items, 0.85, require_same_unit=True)

        for merged_item, merged_indices, sim, merged_descs in cross_merged:
            result_items.append(merged_item)
            consumed.update(merged_indices)
            if len(merged_indices) > 1:
                raw_no = merged_item.get("item_no", "") or merged_item.get("item_code", "")
                merge_log.append(MergeLogEntry(
                    kept_description=merged_item.get("description", ""),
                    merged_descriptions=merged_descs,
                    source_pages=merged_item.get("merged_from", []),
                    similarity=sim,
                    item_no=normalize_item_no(raw_no),
                ))

    # --- Phase 4: add any truly unconsumed items ---
    for idx in range(len(items)):
        if idx not in consumed:
            result_items.append(dict(items[idx]))

    deduplicated_count = len(result_items)
    duplicates_found = original_count - deduplicated_count

    logger.info(
        "BOQ deduplication: %d -> %d items (%d duplicates removed)",
        original_count,
        deduplicated_count,
        duplicates_found,
    )

    return DeduplicationResult(
        items=result_items,
        original_count=original_count,
        deduplicated_count=deduplicated_count,
        duplicates_found=duplicates_found,
        merge_log=merge_log,
    )


def _merge_group(
    indexed_items: List[Tuple[int, Dict]],
    threshold: float,
    require_same_unit: bool = False,
) -> List[Tuple[Dict, List[int], float, List[str]]]:
    """Merge duplicates within a group of items.

    Uses greedy merging: the most complete item absorbs its duplicates.

    Args:
        indexed_items: List of (original_index, item_dict) tuples.
        threshold: Similarity threshold.
        require_same_unit: If True, only merge items with same unit.

    Returns:
        List of tuples:
            (merged_item, list_of_original_indices, best_similarity, merged_descriptions)
    """
    if not indexed_items:
        return []

    # Sort by completeness (most complete first) so the best record is primary
    sorted_items = sorted(
        indexed_items,
        key=lambda pair: _completeness_score(pair[1]),
        reverse=True,
    )

    consumed: Set[int] = set()
    results: List[Tuple[Dict, List[int], float, List[str]]] = []

    for i, (idx_a, item_a) in enumerate(sorted_items):
        if idx_a in consumed:
            continue

        current = dict(item_a)
        group_indices = [idx_a]
        group_descs = [item_a.get("description", "")]
        best_sim = 1.0
        consumed.add(idx_a)

        for j in range(i + 1, len(sorted_items)):
            idx_b, item_b = sorted_items[j]
            if idx_b in consumed:
                continue

            if require_same_unit:
                u1 = normalize_unit(current.get("unit", ""))
                u2 = normalize_unit(item_b.get("unit", ""))
                if u1 and u2 and u1 != u2:
                    continue

            if is_likely_duplicate(current, item_b, threshold=threshold):
                sim = token_similarity(
                    current.get("description", ""),
                    item_b.get("description", ""),
                )
                current = _merge_two(current, item_b, sim)
                group_indices.append(idx_b)
                group_descs.append(item_b.get("description", ""))
                best_sim = min(best_sim, sim)
                consumed.add(idx_b)

        results.append((current, group_indices, best_sim, group_descs))

    return results
