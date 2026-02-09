"""
Quote Leveler - Level quotes to common basis for comparison.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from .parser import SubcontractorQuote


@dataclass
class LeveledComparison:
    """Result of quote leveling."""
    quotes: List[SubcontractorQuote]
    comparison_matrix: List[Dict] = field(default_factory=list)
    scope_differences: List[Dict] = field(default_factory=list)
    terms_comparison: Dict = field(default_factory=dict)
    adjustments_applied: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "quotes": [q.to_dict() for q in self.quotes],
            "comparison_matrix": self.comparison_matrix,
            "scope_differences": self.scope_differences,
            "terms_comparison": self.terms_comparison,
            "adjustments_applied": self.adjustments_applied,
        }


class QuoteLeveler:
    """Level quotes for apple-to-apple comparison."""

    # Standard exclusion costs (to add back when excluded)
    EXCLUSION_COSTS = {
        "scaffolding": {"basis": "sqm", "rate": 80},
        "water_supply": {"basis": "sqm", "rate": 5},
        "power_supply": {"basis": "sqm", "rate": 8},
        "transport": {"basis": "pct", "rate": 3},
        "unloading": {"basis": "pct", "rate": 1.5},
        "debris_removal": {"basis": "pct", "rate": 2},
        "safety": {"basis": "pct", "rate": 1},
        "testing": {"basis": "pct", "rate": 1.5},
        "supervision": {"basis": "pct", "rate": 3},
        "cleaning": {"basis": "pct", "rate": 1},
    }

    # Extra inclusions that deserve credit
    INCLUSION_CREDITS = {
        "warranty_extended": {"basis": "pct", "rate": -2},  # Negative = credit
        "fast_track": {"basis": "pct", "rate": -3},
        "guaranteed_completion": {"basis": "pct", "rate": -2},
        "quality_certification": {"basis": "pct", "rate": -1},
    }

    def __init__(self):
        pass

    def level(
        self,
        quotes: List[SubcontractorQuote],
        boq_items: List[Dict],
    ) -> LeveledComparison:
        """Level quotes for comparison."""
        if not quotes:
            return LeveledComparison(quotes=[])

        # Calculate total area for adjustments
        total_area = sum(
            float(item.get("quantity", 0))
            for item in boq_items
            if item.get("unit", "").lower() in ["sqm", "sq.m", "m2"]
        )
        if total_area == 0:
            total_area = 100  # Default assumption

        # 1. Apply exclusion adjustments
        adjustments_applied = []
        for quote in quotes:
            quote_adjustments = self._calculate_adjustments(quote, total_area)
            adjustments_applied.extend(quote_adjustments)

            # Calculate leveled total
            total_adjustment = sum(adj["amount"] for adj in quote_adjustments)
            quote.leveled_total = quote.total_amount + total_adjustment

        # 2. Build comparison matrix
        comparison_matrix = self._build_comparison_matrix(quotes, boq_items)

        # 3. Identify scope differences
        scope_differences = self._identify_scope_differences(quotes)

        # 4. Compare commercial terms
        terms_comparison = self._compare_terms(quotes)

        return LeveledComparison(
            quotes=quotes,
            comparison_matrix=comparison_matrix,
            scope_differences=scope_differences,
            terms_comparison=terms_comparison,
            adjustments_applied=adjustments_applied,
        )

    def _calculate_adjustments(
        self,
        quote: SubcontractorQuote,
        total_area: float,
    ) -> List[Dict]:
        """Calculate adjustments for a quote."""
        adjustments = []

        # Add back excluded items
        for exclusion in quote.exclusions:
            exc_lower = exclusion.lower()

            for exc_key, exc_info in self.EXCLUSION_COSTS.items():
                if exc_key in exc_lower or exc_key.replace("_", " ") in exc_lower:
                    if exc_info["basis"] == "sqm":
                        amount = exc_info["rate"] * total_area
                    else:  # pct
                        amount = quote.total_amount * (exc_info["rate"] / 100)

                    adjustments.append({
                        "subcontractor": quote.subcontractor_name,
                        "type": "exclusion_add",
                        "item": exc_key,
                        "description": f"Add back: {exclusion}",
                        "amount": round(amount, 2),
                    })
                    break

        # Give credit for extra inclusions
        for inclusion in quote.inclusions:
            inc_lower = inclusion.lower()

            for inc_key, inc_info in self.INCLUSION_CREDITS.items():
                if inc_key.replace("_", " ") in inc_lower:
                    if inc_info["basis"] == "sqm":
                        amount = inc_info["rate"] * total_area
                    else:  # pct
                        amount = quote.total_amount * (inc_info["rate"] / 100)

                    adjustments.append({
                        "subcontractor": quote.subcontractor_name,
                        "type": "inclusion_credit",
                        "item": inc_key,
                        "description": f"Credit for: {inclusion}",
                        "amount": round(amount, 2),  # Already negative for credit
                    })
                    break

        # Adjust for warranty differences (standard is 12 months)
        if quote.warranty_months > 12:
            extra_months = quote.warranty_months - 12
            credit = quote.total_amount * (extra_months * 0.5 / 100)  # 0.5% per extra month
            adjustments.append({
                "subcontractor": quote.subcontractor_name,
                "type": "warranty_credit",
                "item": "extended_warranty",
                "description": f"Credit for {extra_months} extra months warranty",
                "amount": round(-credit, 2),
            })

        return adjustments

    def _build_comparison_matrix(
        self,
        quotes: List[SubcontractorQuote],
        boq_items: List[Dict],
    ) -> List[Dict]:
        """Build item-wise comparison matrix."""
        matrix = []

        # Get all unique items
        all_items = {}
        for item in boq_items:
            item_id = item.get("unified_item_no", item.get("item_no", ""))
            all_items[item_id] = {
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
                "quantity": float(item.get("quantity", 0)),
            }

        # Build matrix rows
        for item_id, item_info in all_items.items():
            row = {
                "item": item_id,
                "description": item_info["description"][:40],
                "unit": item_info["unit"],
                "quantity": item_info["quantity"],
            }

            for quote in quotes:
                # Find this item in quote
                quote_item = None
                for qi in quote.line_items:
                    if qi.item_no == item_id:
                        quote_item = qi
                        break

                if quote_item:
                    row[quote.subcontractor_name] = quote_item.amount
                    row[f"{quote.subcontractor_name}_rate"] = quote_item.rate
                else:
                    row[quote.subcontractor_name] = 0
                    row[f"{quote.subcontractor_name}_rate"] = 0

            matrix.append(row)

        # Add total row
        total_row = {
            "item": "TOTAL",
            "description": "Grand Total (Leveled)",
            "unit": "",
            "quantity": 0,
        }
        for quote in quotes:
            total_row[quote.subcontractor_name] = quote.leveled_total
            total_row[f"{quote.subcontractor_name}_rate"] = ""

        matrix.append(total_row)

        return matrix

    def _identify_scope_differences(
        self,
        quotes: List[SubcontractorQuote],
    ) -> List[Dict]:
        """Identify scope differences between quotes."""
        differences = []

        for quote in quotes:
            diff = {
                "subcontractor": quote.subcontractor_name,
                "exclusions": quote.exclusions,
                "inclusions": quote.inclusions,
                "notes": quote.notes,
            }

            if quote.exclusions or quote.inclusions:
                differences.append(diff)

        return differences

    def _compare_terms(
        self,
        quotes: List[SubcontractorQuote],
    ) -> Dict:
        """Compare commercial terms across quotes."""
        terms = {
            "Validity (days)": {},
            "Completion (days)": {},
            "Warranty (months)": {},
            "Mobilization Advance (%)": {},
            "Retention (%)": {},
            "Payment Terms": {},
            "GST Included": {},
        }

        for quote in quotes:
            name = quote.subcontractor_name
            terms["Validity (days)"][name] = quote.validity_days
            terms["Completion (days)"][name] = quote.completion_days or "Not specified"
            terms["Warranty (months)"][name] = quote.warranty_months
            terms["Mobilization Advance (%)"][name] = quote.mobilization_advance
            terms["Retention (%)"][name] = quote.retention_percent
            terms["Payment Terms"][name] = quote.payment_terms or "Not specified"
            terms["GST Included"][name] = "Yes" if quote.gst_included else "No"

        return terms
