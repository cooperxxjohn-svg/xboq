"""
Cross-Document Scope Matrix — Track items across all project sources.

Maps every BOQ item against every document source to detect:
1. Items in BOQ but not in any drawing → unverified quantities
2. Items in drawing but not in BOQ → missing scope ($$$ risk)
3. Items in notes/specs but not priced → unfunded requirements
4. Quantity mismatches between sources → RFI trigger

Sources tracked:
- BOQ / Bill of Quantities (priced items)
- Architectural drawings (rooms, walls, openings)
- Structural drawings (RCC elements, foundations)
- Schedules (door, window, finish, steel, MEP)
- Specifications / Notes (material grades, standards)
- Addenda / Corrigenda (changes, additions)

India-specific: IS 1200 measurement basis, CPWD BOQ format.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SOURCE TYPES
# =============================================================================

class SourceType(Enum):
    BOQ = "boq"
    ARCHITECTURAL = "architectural"
    STRUCTURAL = "structural"
    SCHEDULE = "schedule"
    SPECIFICATION = "specification"
    ADDENDUM = "addendum"
    SITE_PLAN = "site_plan"
    MEP_DRAWING = "mep_drawing"


# =============================================================================
# SCOPE MATRIX ENTRY
# =============================================================================

@dataclass
class ScopeMatrixEntry:
    """
    One row in the scope matrix = one construction item tracked across sources.
    """
    item_key: str               # Normalized item identifier
    description: str            # Human-readable description
    trade: str                  # Construction trade (structural, finishing, etc.)
    unit: str = ""

    # Source presence (True/False/None for each source type)
    in_boq: bool = False
    in_architectural: bool = False
    in_structural: bool = False
    in_schedule: bool = False
    in_specification: bool = False
    in_addendum: bool = False
    in_site_plan: bool = False
    in_mep: bool = False

    # Quantities from each source (for mismatch detection)
    boq_qty: float = 0.0
    drawing_qty: float = 0.0
    schedule_qty: float = 0.0

    # Metadata
    boq_rate: float = 0.0
    page_refs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        """How many sources confirm this item."""
        return sum([
            self.in_boq, self.in_architectural, self.in_structural,
            self.in_schedule, self.in_specification, self.in_addendum,
            self.in_site_plan, self.in_mep,
        ])

    @property
    def status(self) -> str:
        """
        Status classification:
        - 'confirmed': In BOQ + at least one other source
        - 'boq_only': In BOQ but no other source (unverified)
        - 'missing_from_boq': In drawing/schedule but NOT in BOQ (risk!)
        - 'spec_only': Only in specification text (may need pricing)
        - 'orphan': Only in one non-BOQ source
        """
        if self.in_boq and self.source_count >= 2:
            return "confirmed"
        elif self.in_boq and self.source_count == 1:
            return "boq_only"
        elif not self.in_boq and self.source_count >= 1:
            return "missing_from_boq"
        elif self.in_specification and self.source_count == 1:
            return "spec_only"
        else:
            return "orphan"

    @property
    def has_qty_mismatch(self) -> bool:
        """Check if quantities differ significantly between sources."""
        qtys = [q for q in [self.boq_qty, self.drawing_qty, self.schedule_qty] if q > 0]
        if len(qtys) < 2:
            return False
        max_q = max(qtys)
        min_q = min(qtys)
        if min_q == 0:
            return True
        return (max_q / min_q) > 1.15  # >15% difference

    @property
    def risk_level(self) -> str:
        """
        Risk level:
        - HIGH: Missing from BOQ with significant cost impact
        - MEDIUM: Quantity mismatch or unverified
        - LOW: Confirmed across sources
        """
        if self.status == "missing_from_boq":
            return "HIGH"
        elif self.has_qty_mismatch or self.status == "boq_only":
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_key": self.item_key,
            "description": self.description,
            "trade": self.trade,
            "unit": self.unit,
            "status": self.status,
            "risk_level": self.risk_level,
            "source_count": self.source_count,
            "in_boq": self.in_boq,
            "in_architectural": self.in_architectural,
            "in_structural": self.in_structural,
            "in_schedule": self.in_schedule,
            "in_specification": self.in_specification,
            "boq_qty": self.boq_qty,
            "drawing_qty": self.drawing_qty,
            "schedule_qty": self.schedule_qty,
            "has_qty_mismatch": self.has_qty_mismatch,
            "page_refs": self.page_refs,
            "notes": self.notes,
        }


# =============================================================================
# SCOPE MATRIX BUILDER
# =============================================================================

@dataclass
class ScopeMatrix:
    """
    Complete cross-document scope matrix.
    """
    entries: Dict[str, ScopeMatrixEntry] = field(default_factory=dict)
    sources_loaded: Set[str] = field(default_factory=set)

    def add_from_boq(
        self,
        boq_items: List[Dict[str, Any]],
    ) -> int:
        """
        Populate matrix from BOQ items.

        Args:
            boq_items: List of BOQ items with description, quantity, unit, rate

        Returns:
            Number of items added/updated
        """
        count = 0
        for item in boq_items:
            key = self._make_key(item.get("description", ""), item.get("trade", ""))
            desc = item.get("description", item.get("item_name", ""))
            if not desc:
                continue

            entry = self.entries.get(key, ScopeMatrixEntry(
                item_key=key, description=desc,
                trade=item.get("trade", item.get("package", "")),
                unit=item.get("unit", ""),
            ))
            entry.in_boq = True
            entry.boq_qty = float(item.get("quantity", item.get("qty", 0)))
            entry.boq_rate = float(item.get("rate", 0))
            self.entries[key] = entry
            count += 1

        self.sources_loaded.add("boq")
        return count

    def add_from_drawings(
        self,
        drawing_items: List[Dict[str, Any]],
        source_type: SourceType = SourceType.ARCHITECTURAL,
    ) -> int:
        """
        Populate matrix from drawing extraction.

        Args:
            drawing_items: Extracted components with description, quantity
            source_type: Which drawing type

        Returns:
            Number of items added/updated
        """
        count = 0
        for item in drawing_items:
            key = self._make_key(item.get("description", ""), item.get("trade", ""))
            desc = item.get("description", item.get("component_type", ""))
            if not desc:
                continue

            entry = self.entries.get(key, ScopeMatrixEntry(
                item_key=key, description=desc,
                trade=item.get("trade", ""),
                unit=item.get("unit", ""),
            ))

            if source_type == SourceType.ARCHITECTURAL:
                entry.in_architectural = True
            elif source_type == SourceType.STRUCTURAL:
                entry.in_structural = True
            elif source_type == SourceType.SITE_PLAN:
                entry.in_site_plan = True
            elif source_type == SourceType.MEP_DRAWING:
                entry.in_mep = True

            entry.drawing_qty = float(item.get("quantity", item.get("qty", 0)))
            if item.get("page_ref"):
                entry.page_refs.append(str(item["page_ref"]))

            self.entries[key] = entry
            count += 1

        self.sources_loaded.add(source_type.value)
        return count

    def add_from_schedules(
        self,
        schedule_items: List[Dict[str, Any]],
    ) -> int:
        """Populate from door/window/finish/MEP schedules."""
        count = 0
        for item in schedule_items:
            key = self._make_key(item.get("description", ""), item.get("trade", ""))
            desc = item.get("description", item.get("mark", ""))
            if not desc:
                continue

            entry = self.entries.get(key, ScopeMatrixEntry(
                item_key=key, description=desc,
                trade=item.get("trade", ""),
                unit=item.get("unit", ""),
            ))
            entry.in_schedule = True
            entry.schedule_qty = float(item.get("quantity", item.get("count", 0)))
            self.entries[key] = entry
            count += 1

        self.sources_loaded.add("schedule")
        return count

    def add_from_specifications(
        self,
        spec_items: List[Dict[str, Any]],
    ) -> int:
        """Populate from specification/notes extraction."""
        count = 0
        for item in spec_items:
            key = self._make_key(item.get("description", ""), item.get("trade", ""))
            desc = item.get("description", "")
            if not desc:
                continue

            entry = self.entries.get(key, ScopeMatrixEntry(
                item_key=key, description=desc,
                trade=item.get("trade", ""),
            ))
            entry.in_specification = True
            if item.get("note"):
                entry.notes.append(item["note"])
            self.entries[key] = entry
            count += 1

        self.sources_loaded.add("specification")
        return count

    # ─── Analysis ───

    def get_missing_from_boq(self) -> List[ScopeMatrixEntry]:
        """Items detected in drawings/schedules but NOT in BOQ."""
        return [e for e in self.entries.values() if e.status == "missing_from_boq"]

    def get_unverified_boq_items(self) -> List[ScopeMatrixEntry]:
        """BOQ items not confirmed by any other source."""
        return [e for e in self.entries.values() if e.status == "boq_only"]

    def get_quantity_mismatches(self) -> List[ScopeMatrixEntry]:
        """Items with significant quantity differences between sources."""
        return [e for e in self.entries.values() if e.has_qty_mismatch]

    def get_high_risk_items(self) -> List[ScopeMatrixEntry]:
        """All HIGH risk items."""
        return [e for e in self.entries.values() if e.risk_level == "HIGH"]

    def get_confirmed_items(self) -> List[ScopeMatrixEntry]:
        """Items confirmed across 2+ sources."""
        return [e for e in self.entries.values() if e.status == "confirmed"]

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Summary of scope coverage across sources."""
        total = len(self.entries)
        if total == 0:
            return {"total_items": 0, "coverage_score": 100.0}

        confirmed = len(self.get_confirmed_items())
        missing = len(self.get_missing_from_boq())
        unverified = len(self.get_unverified_boq_items())
        mismatches = len(self.get_quantity_mismatches())
        high_risk = len(self.get_high_risk_items())

        # Coverage score: penalize missing and mismatched items
        penalty = (missing * 3 + mismatches * 2 + unverified * 1)
        max_penalty = total * 3
        coverage_score = max(0.0, round((1 - penalty / max(max_penalty, 1)) * 100, 1))

        # Trade-level breakdown
        trade_summary = {}
        for entry in self.entries.values():
            trade = entry.trade or "unknown"
            if trade not in trade_summary:
                trade_summary[trade] = {
                    "total": 0, "confirmed": 0, "missing": 0, "unverified": 0,
                }
            trade_summary[trade]["total"] += 1
            if entry.status == "confirmed":
                trade_summary[trade]["confirmed"] += 1
            elif entry.status == "missing_from_boq":
                trade_summary[trade]["missing"] += 1
            elif entry.status == "boq_only":
                trade_summary[trade]["unverified"] += 1

        return {
            "total_items": total,
            "confirmed": confirmed,
            "missing_from_boq": missing,
            "unverified": unverified,
            "quantity_mismatches": mismatches,
            "high_risk": high_risk,
            "coverage_score": coverage_score,
            "sources_loaded": sorted(self.sources_loaded),
            "trade_summary": trade_summary,
        }

    def generate_rfis(self) -> List[Dict[str, Any]]:
        """
        Generate RFI items for all scope gaps.

        Returns:
            List of RFI dicts with description, reason, priority, trade
        """
        rfis = []

        # Missing from BOQ
        for entry in self.get_missing_from_boq():
            sources = []
            if entry.in_architectural:
                sources.append("architectural drawing")
            if entry.in_structural:
                sources.append("structural drawing")
            if entry.in_schedule:
                sources.append("schedule")
            if entry.in_specification:
                sources.append("specification")

            rfis.append({
                "type": "missing_scope",
                "description": f"'{entry.description}' found in {', '.join(sources)} but missing from BOQ",
                "item_key": entry.item_key,
                "trade": entry.trade,
                "priority": "HIGH",
                "suggested_action": "Add to BOQ or confirm exclusion",
                "page_refs": entry.page_refs,
            })

        # Quantity mismatches
        for entry in self.get_quantity_mismatches():
            rfis.append({
                "type": "quantity_mismatch",
                "description": (
                    f"'{entry.description}': BOQ qty={entry.boq_qty}, "
                    f"Drawing qty={entry.drawing_qty}, "
                    f"Schedule qty={entry.schedule_qty}"
                ),
                "item_key": entry.item_key,
                "trade": entry.trade,
                "priority": "MEDIUM",
                "suggested_action": "Verify correct quantity from authoritative source",
            })

        # Unverified BOQ items (lower priority)
        for entry in self.get_unverified_boq_items():
            if entry.boq_rate > 0 and entry.boq_qty > 0:
                amount = entry.boq_qty * entry.boq_rate
                if amount > 50000:  # Only flag significant items
                    rfis.append({
                        "type": "unverified_quantity",
                        "description": (
                            f"'{entry.description}' in BOQ (₹{amount:,.0f}) "
                            f"but not found in any drawing or schedule"
                        ),
                        "item_key": entry.item_key,
                        "trade": entry.trade,
                        "priority": "LOW",
                        "suggested_action": "Verify quantity basis and source",
                    })

        return rfis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.get_coverage_summary(),
            "entries": [e.to_dict() for e in self.entries.values()],
            "rfis": self.generate_rfis(),
        }

    # ─── Internal ───

    @staticmethod
    def _make_key(description: str, trade: str = "") -> str:
        """Create a normalized key for deduplication."""
        desc = description.lower().strip()
        # Remove common prefixes
        for prefix in ["providing and", "providing", "supplying and", "supplying",
                       "fixing and", "fixing", "laying and", "laying",
                       "constructing", "applying"]:
            if desc.startswith(prefix):
                desc = desc[len(prefix):].strip()
        # Remove extra whitespace
        desc = " ".join(desc.split())
        # Combine with trade for uniqueness
        if trade:
            return f"{trade.lower()}:{desc[:80]}"
        return desc[:80]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_scope_matrix(
    boq_items: List[Dict[str, Any]] = None,
    drawing_items: List[Dict[str, Any]] = None,
    schedule_items: List[Dict[str, Any]] = None,
    spec_items: List[Dict[str, Any]] = None,
) -> ScopeMatrix:
    """
    Build a complete scope matrix from available sources.

    Args:
        boq_items: BOQ items
        drawing_items: Drawing extraction items
        schedule_items: Schedule items
        spec_items: Specification items

    Returns:
        Populated ScopeMatrix
    """
    matrix = ScopeMatrix()

    if boq_items:
        matrix.add_from_boq(boq_items)
    if drawing_items:
        matrix.add_from_drawings(drawing_items)
    if schedule_items:
        matrix.add_from_schedules(schedule_items)
    if spec_items:
        matrix.add_from_specifications(spec_items)

    logger.info(
        f"Scope matrix: {len(matrix.entries)} items, "
        f"{len(matrix.get_missing_from_boq())} missing from BOQ, "
        f"{len(matrix.get_quantity_mismatches())} qty mismatches"
    )

    return matrix
