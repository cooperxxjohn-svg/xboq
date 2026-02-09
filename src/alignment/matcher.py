"""
BOQ Matcher - Match items between drawings BOQ and owner BOQ.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re


@dataclass
class MatchResult:
    """Result of matching a BOQ item."""
    match_type: str  # "matched", "drawings_only", "owner_only"
    drawings_item: Optional[Dict] = None
    owner_item: Optional[Dict] = None
    match_confidence: float = 0.0
    match_method: str = ""  # How the match was found

    def to_dict(self) -> dict:
        return {
            "match_type": self.match_type,
            "drawings_item": self.drawings_item,
            "owner_item": self.owner_item,
            "match_confidence": self.match_confidence,
            "match_method": self.match_method,
        }


class BOQMatcher:
    """Match BOQ items between drawings and owner BOQ."""

    # Synonyms for common construction terms (India-specific)
    SYNONYMS = {
        # Concrete terms
        "rcc": ["reinforced cement concrete", "reinforced concrete", "r.c.c.", "r.c.c"],
        "pcc": ["plain cement concrete", "plain concrete", "p.c.c.", "p.c.c", "lean concrete"],
        "m20": ["m-20", "m 20", "1:1.5:3"],
        "m25": ["m-25", "m 25", "1:1:2"],
        "m30": ["m-30", "m 30"],

        # Brickwork terms
        "brick": ["bricks", "brickwork", "brick masonry"],
        "aac": ["aac blocks", "aac block", "autoclaved aerated concrete", "siporex"],
        "fly ash": ["fly ash bricks", "fab", "flyash"],

        # Plastering terms
        "plaster": ["plastering", "cement plaster", "cement mortar plaster"],
        "cm": ["cement mortar", "c.m."],
        "internal": ["inside", "inner"],
        "external": ["outside", "outer", "exterior"],

        # Waterproofing terms
        "wp": ["waterproofing", "water proofing", "w/p", "w.p."],
        "app": ["app membrane", "modified bitumen", "torch applied"],
        "bbc": ["brick bat coba", "brick bat", "brickbat coba"],

        # Flooring terms
        "vitrified": ["vitrified tiles", "vitrified tile", "vt"],
        "ceramic": ["ceramic tiles", "ceramic tile", "ct"],
        "granite": ["granite flooring", "granite stone"],
        "marble": ["marble flooring", "marble stone"],
        "kota": ["kota stone", "kota flooring"],

        # Doors/windows
        "door": ["doors", "shutter", "shutters"],
        "window": ["windows", "ventilator", "ventilators"],
        "frame": ["frames", "chowkhat", "chowkhats"],
        "flush": ["flush door", "flush shutter"],
        "ms": ["mild steel", "m.s.", "m.s"],
        "ss": ["stainless steel", "s.s.", "s.s"],

        # Painting
        "paint": ["painting", "paints"],
        "emulsion": ["acrylic emulsion", "plastic emulsion", "plastic paint"],
        "distemper": ["oil bound distemper", "obd", "o.b.d."],
        "primer": ["priming", "primer coat"],
        "putty": ["wall putty", "puttying"],

        # Plumbing
        "cpvc": ["c.p.v.c.", "cpvc pipe", "cpvc pipes"],
        "upvc": ["u.p.v.c.", "upvc pipe", "upvc pipes", "pvc"],
        "swr": ["soil waste rain", "s.w.r.", "swp"],
        "gi": ["galvanized iron", "g.i.", "g.i"],

        # Electrical
        "point": ["points", "outlet", "outlets"],
        "conduit": ["conduits", "conduit pipe"],
        "mcb": ["m.c.b.", "miniature circuit breaker"],
        "db": ["distribution board", "d.b.", "distribution box"],

        # Units
        "sqm": ["sq.m", "sq.m.", "sq m", "m2", "m²", "square meter", "square metre"],
        "cum": ["cu.m", "cu.m.", "cu m", "m3", "m³", "cubic meter", "cubic metre"],
        "rmt": ["r.m.", "rm", "running meter", "running metre", "linear meter"],
        "nos": ["no.", "no", "nos.", "numbers", "each"],
        "kg": ["kgs", "kilogram", "kilograms"],
        "mt": ["metric ton", "metric tonne", "ton", "tonne"],
        "ls": ["lump sum", "l.s.", "lumpsum"],
    }

    # Package mapping for item classification
    PACKAGE_KEYWORDS = {
        "civil_structural": ["rcc", "pcc", "concrete", "footing", "column", "beam", "slab", "foundation", "staircase", "lintel", "chajja"],
        "masonry": ["brick", "block", "aac", "masonry", "wall"],
        "plaster_finishes": ["plaster", "putty", "paint", "emulsion", "distemper", "texture"],
        "flooring": ["flooring", "tile", "vitrified", "ceramic", "marble", "granite", "kota", "wooden"],
        "waterproofing": ["waterproof", "wp", "app", "membrane", "coba", "terrace treatment"],
        "doors_windows": ["door", "window", "shutter", "frame", "grill", "hardware"],
        "plumbing": ["pipe", "cpvc", "upvc", "swr", "drainage", "water supply", "sanitary", "cp fitting"],
        "electrical": ["wiring", "point", "conduit", "switch", "mcb", "db", "earthing", "cable"],
        "external_works": ["compound", "gate", "paving", "landscaping", "drain"],
    }

    def __init__(self):
        # Build reverse synonym lookup
        self.synonym_lookup = {}
        for canonical, synonyms in self.SYNONYMS.items():
            for syn in synonyms:
                self.synonym_lookup[syn.lower()] = canonical
            self.synonym_lookup[canonical.lower()] = canonical

    def match(
        self,
        drawings_boq: List[Dict],
        owner_boq: List[Dict],
    ) -> List[MatchResult]:
        """Match items between drawings BOQ and owner BOQ."""
        results = []
        owner_matched = set()

        # First pass: exact item_id/item_no matching
        for d_item in drawings_boq:
            d_id = str(d_item.get("item_id", "")).strip()
            matched = False

            for idx, o_item in enumerate(owner_boq):
                if idx in owner_matched:
                    continue

                o_id = str(o_item.get("item_no", "")).strip()
                if d_id and o_id and self._normalize_item_id(d_id) == self._normalize_item_id(o_id):
                    results.append(MatchResult(
                        match_type="matched",
                        drawings_item=d_item,
                        owner_item=o_item,
                        match_confidence=1.0,
                        match_method="item_id",
                    ))
                    owner_matched.add(idx)
                    matched = True
                    break

            if not matched:
                # Try description matching
                d_desc = d_item.get("description", "")
                best_match = None
                best_score = 0.0

                for idx, o_item in enumerate(owner_boq):
                    if idx in owner_matched:
                        continue

                    o_desc = o_item.get("description", "")
                    score = self._description_similarity(d_desc, o_desc)

                    # Also check unit compatibility
                    d_unit = self._normalize_unit(d_item.get("unit", ""))
                    o_unit = self._normalize_unit(o_item.get("unit", ""))
                    if d_unit == o_unit:
                        score += 0.1  # Bonus for matching units

                    if score > best_score and score >= 0.6:  # Minimum threshold
                        best_score = score
                        best_match = idx

                if best_match is not None:
                    results.append(MatchResult(
                        match_type="matched",
                        drawings_item=d_item,
                        owner_item=owner_boq[best_match],
                        match_confidence=best_score,
                        match_method="description_similarity",
                    ))
                    owner_matched.add(best_match)
                else:
                    # No match found - item only in drawings
                    results.append(MatchResult(
                        match_type="drawings_only",
                        drawings_item=d_item,
                        owner_item=None,
                        match_confidence=0.0,
                        match_method="no_match",
                    ))

        # Add unmatched owner items
        for idx, o_item in enumerate(owner_boq):
            if idx not in owner_matched:
                results.append(MatchResult(
                    match_type="owner_only",
                    drawings_item=None,
                    owner_item=o_item,
                    match_confidence=0.0,
                    match_method="no_match",
                ))

        return results

    def _normalize_item_id(self, item_id: str) -> str:
        """Normalize item ID for comparison."""
        # Remove common prefixes/suffixes
        normalized = item_id.lower().strip()
        normalized = re.sub(r"^(item\s*|sl\.?\s*no\.?\s*|sr\.?\s*no\.?\s*)", "", normalized)
        normalized = re.sub(r"[\s\.\-_]+", "", normalized)
        return normalized

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit for comparison."""
        unit_lower = unit.lower().strip()

        # Check synonyms
        if unit_lower in self.synonym_lookup:
            return self.synonym_lookup[unit_lower]

        # Direct normalization
        unit_map = {
            "sq.m": "sqm", "sq m": "sqm", "m2": "sqm", "m²": "sqm",
            "cu.m": "cum", "cu m": "cum", "m3": "cum", "m³": "cum",
            "r.m.": "rmt", "rm": "rmt",
            "no.": "nos", "no": "nos",
            "kgs": "kg",
            "l.s.": "ls",
        }

        return unit_map.get(unit_lower, unit_lower)

    def _description_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between two descriptions."""
        # Tokenize and normalize
        tokens1 = self._tokenize_description(desc1)
        tokens2 = self._tokenize_description(desc2)

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity with synonym expansion
        expanded1 = self._expand_synonyms(tokens1)
        expanded2 = self._expand_synonyms(tokens2)

        intersection = expanded1 & expanded2
        union = expanded1 | expanded2

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Boost for matching key terms
        key_terms = {"rcc", "pcc", "brick", "plaster", "waterproof", "flooring", "door", "window", "pipe", "wire"}
        key_match = len((expanded1 & key_terms) & (expanded2 & key_terms))
        if key_match > 0:
            jaccard = min(1.0, jaccard + 0.15 * key_match)

        return jaccard

    def _tokenize_description(self, desc: str) -> set:
        """Tokenize description into normalized terms."""
        # Clean and lowercase
        desc_clean = desc.lower()
        desc_clean = re.sub(r"[^\w\s]", " ", desc_clean)

        # Split into tokens
        tokens = set(desc_clean.split())

        # Remove common stopwords
        stopwords = {"the", "a", "an", "of", "for", "in", "to", "with", "and", "or", "as", "per", "etc", "including", "including"}
        tokens = tokens - stopwords

        return tokens

    def _expand_synonyms(self, tokens: set) -> set:
        """Expand tokens with synonyms."""
        expanded = set()

        for token in tokens:
            expanded.add(token)
            if token in self.synonym_lookup:
                expanded.add(self.synonym_lookup[token])

        return expanded

    def classify_package(self, item: Dict) -> str:
        """Classify item into work package."""
        desc = item.get("description", "").lower()

        for package, keywords in self.PACKAGE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc:
                    return package

        return "miscellaneous"
