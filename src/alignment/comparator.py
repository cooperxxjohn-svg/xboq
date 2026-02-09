"""
BOQ Comparator - Compare quantities between matched BOQ items.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Discrepancy:
    """Represents a quantity discrepancy between drawings and owner BOQ."""
    item_id: str
    description: str
    unit: str
    drawings_qty: float
    owner_qty: float
    difference: float
    difference_percent: float
    severity: str  # low / medium / high / critical
    possible_cause: str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "description": self.description,
            "unit": self.unit,
            "drawings_qty": self.drawings_qty,
            "owner_qty": self.owner_qty,
            "difference": self.difference,
            "difference_percent": self.difference_percent,
            "severity": self.severity,
            "possible_cause": self.possible_cause,
            "recommendation": self.recommendation,
        }


class BOQComparator:
    """Compare quantities between matched BOQ items."""

    # Possible causes of discrepancy by item type
    DISCREPANCY_CAUSES = {
        "rcc": [
            "Different section/beam sizes assumed",
            "Slab thickness variation",
            "Column sizes different from assumed",
            "Additional structural members not visible in plan",
        ],
        "brick": [
            "Wall thickness difference (115mm vs 230mm)",
            "Opening deductions calculated differently",
            "Parapet/compound wall included/excluded",
        ],
        "plaster": [
            "Internal vs external areas mixed",
            "Ceiling plaster included/excluded",
            "Opening deductions differ",
        ],
        "flooring": [
            "Skirting included/excluded",
            "Wastage factor different",
            "Balcony/terrace area calculation differs",
        ],
        "waterproof": [
            "Turnup height different",
            "Treatment layers counted differently",
            "Area measurement methodology differs",
        ],
        "door": [
            "Frame included with shutter or separate",
            "Hardware counted separately",
            "Window grills counted with windows or separate",
        ],
        "pipe": [
            "Fittings percentage different",
            "Vertical risers measured differently",
            "Connection pieces included/excluded",
        ],
        "electrical": [
            "Point definition differs (socket vs outlet)",
            "Wiring lengths measured differently",
            "DB/MCB counted with main or separate",
        ],
    }

    def __init__(self, tolerance_percent: float = 10.0):
        self.tolerance_percent = tolerance_percent

    def compare(self, matches: List) -> List[Discrepancy]:
        """Compare quantities for matched items."""
        discrepancies = []

        for match in matches:
            if match.match_type != "matched":
                continue

            d_item = match.drawings_item
            o_item = match.owner_item

            if not d_item or not o_item:
                continue

            # Get quantities
            d_qty = float(d_item.get("quantity", 0) or 0)
            o_qty = float(o_item.get("quantity", 0) or 0)

            # Handle unit conversion if needed
            d_unit = self._normalize_unit(d_item.get("unit", ""))
            o_unit = self._normalize_unit(o_item.get("unit", ""))

            if d_unit != o_unit:
                # Try to convert
                converted_qty = self._convert_unit(d_qty, d_unit, o_unit)
                if converted_qty is not None:
                    d_qty = converted_qty
                else:
                    # Cannot compare different units
                    discrepancies.append(Discrepancy(
                        item_id=d_item.get("item_id", ""),
                        description=d_item.get("description", "")[:60],
                        unit=f"{d_unit} vs {o_unit}",
                        drawings_qty=d_qty,
                        owner_qty=o_qty,
                        difference=0,
                        difference_percent=0,
                        severity="high",
                        possible_cause="Unit mismatch - cannot compare",
                        recommendation=f"Verify units: drawings has {d_unit}, owner BOQ has {o_unit}",
                    ))
                    continue

            # Calculate difference
            difference = d_qty - o_qty
            if o_qty != 0:
                diff_percent = (difference / o_qty) * 100
            else:
                diff_percent = 100 if d_qty > 0 else 0

            # Check if within tolerance
            if abs(diff_percent) <= self.tolerance_percent:
                continue  # Within tolerance, no discrepancy

            # Determine severity
            severity = self._determine_severity(diff_percent)

            # Identify possible cause
            item_type = self._identify_item_type(d_item.get("description", ""))
            possible_cause = self._get_possible_cause(item_type, difference)

            # Generate recommendation
            recommendation = self._get_recommendation(diff_percent, item_type)

            discrepancies.append(Discrepancy(
                item_id=d_item.get("item_id", ""),
                description=d_item.get("description", "")[:60],
                unit=d_unit,
                drawings_qty=d_qty,
                owner_qty=o_qty,
                difference=difference,
                difference_percent=diff_percent,
                severity=severity,
                possible_cause=possible_cause,
                recommendation=recommendation,
            ))

        # Sort by severity and magnitude
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        discrepancies.sort(key=lambda d: (severity_order.get(d.severity, 4), -abs(d.difference_percent)))

        return discrepancies

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit string."""
        unit_lower = unit.lower().strip()
        unit_map = {
            "sq.m": "sqm", "sq m": "sqm", "m2": "sqm", "m²": "sqm", "square meter": "sqm",
            "cu.m": "cum", "cu m": "cum", "m3": "cum", "m³": "cum", "cubic meter": "cum",
            "r.m.": "rmt", "rm": "rmt", "running meter": "rmt",
            "no.": "nos", "no": "nos", "each": "nos",
            "kgs": "kg", "kilogram": "kg",
            "l.s.": "ls", "lump sum": "ls",
        }
        return unit_map.get(unit_lower, unit_lower)

    def _convert_unit(self, qty: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert quantity between units if possible."""
        # Common conversions
        conversions = {
            ("sqm", "sqft"): lambda x: x * 10.764,
            ("sqft", "sqm"): lambda x: x / 10.764,
            ("cum", "cuft"): lambda x: x * 35.315,
            ("cuft", "cum"): lambda x: x / 35.315,
            ("rmt", "ft"): lambda x: x * 3.281,
            ("ft", "rmt"): lambda x: x / 3.281,
            ("kg", "quintal"): lambda x: x / 100,
            ("quintal", "kg"): lambda x: x * 100,
            ("kg", "mt"): lambda x: x / 1000,
            ("mt", "kg"): lambda x: x * 1000,
        }

        key = (from_unit, to_unit)
        if key in conversions:
            return conversions[key](qty)

        return None

    def _determine_severity(self, diff_percent: float) -> str:
        """Determine discrepancy severity based on percentage difference."""
        abs_diff = abs(diff_percent)

        if abs_diff > 50:
            return "critical"
        elif abs_diff > 30:
            return "high"
        elif abs_diff > 20:
            return "medium"
        else:
            return "low"

    def _identify_item_type(self, description: str) -> str:
        """Identify item type from description."""
        desc_lower = description.lower()

        type_keywords = {
            "rcc": ["rcc", "reinforced", "concrete", "beam", "column", "slab", "footing"],
            "brick": ["brick", "masonry", "block", "aac"],
            "plaster": ["plaster", "rendering", "plastering"],
            "flooring": ["flooring", "tile", "marble", "granite", "vitrified", "ceramic"],
            "waterproof": ["waterproof", "wp", "membrane", "coba"],
            "door": ["door", "shutter", "frame", "window"],
            "pipe": ["pipe", "plumbing", "cpvc", "upvc", "drainage"],
            "electrical": ["wiring", "point", "switch", "conduit", "electrical"],
        }

        for item_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return item_type

        return "general"

    def _get_possible_cause(self, item_type: str, difference: float) -> str:
        """Get possible cause of discrepancy."""
        causes = self.DISCREPANCY_CAUSES.get(item_type, [
            "Measurement methodology differs",
            "Specification interpretation varies",
            "Drawing revision not reflected",
        ])

        if difference > 0:
            return f"Drawings higher: {causes[0] if causes else 'Check measurement basis'}"
        else:
            return f"Owner BOQ higher: {causes[1] if len(causes) > 1 else 'Check scope inclusions'}"

    def _get_recommendation(self, diff_percent: float, item_type: str) -> str:
        """Generate recommendation based on discrepancy."""
        abs_diff = abs(diff_percent)

        if abs_diff > 50:
            return "CRITICAL: Re-verify measurements from drawings. Check for scope differences."
        elif abs_diff > 30:
            return "HIGH: Review measurement methodology. Raise RFI for clarification."
        elif abs_diff > 20:
            return "MEDIUM: Compare measurement breakdowns. Document assumptions."
        else:
            return "LOW: Acceptable variance. Use owner BOQ quantity or average."
