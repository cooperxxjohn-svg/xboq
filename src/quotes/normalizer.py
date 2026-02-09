"""
Quote Normalizer - Normalize quotes to common scope basis.
"""

from typing import List, Dict
from .parser import SubcontractorQuote, QuoteLineItem


class QuoteNormalizer:
    """Normalize subcontractor quotes to common scope."""

    # Standard scope items that should be included
    STANDARD_INCLUSIONS = {
        "flooring": [
            "material_supply",
            "labor_installation",
            "adhesive_grout",
            "skirting",
            "wastage_5pct",
        ],
        "plumbing": [
            "material_supply",
            "labor_installation",
            "fittings",
            "testing",
            "concealed_work",
        ],
        "electrical": [
            "material_supply",
            "labor_installation",
            "wiring_conduit",
            "switches_sockets",
            "testing",
        ],
        "waterproofing": [
            "material_supply",
            "labor_application",
            "surface_prep",
            "testing_flooding",
            "warranty_10yr",
        ],
        "painting": [
            "material_supply",
            "labor_application",
            "putty_2coats",
            "primer_1coat",
            "paint_2coats",
        ],
        "doors_windows": [
            "material_supply",
            "labor_fixing",
            "hardware",
            "sealant",
            "touch_up_painting",
        ],
    }

    # Items commonly excluded (need to add back for leveling)
    COMMON_EXCLUSIONS = {
        "scaffolding": {"rate_per_sqm": 80, "description": "Scaffolding for above 3m height"},
        "water_supply": {"rate_per_sqm": 5, "description": "Water for construction"},
        "power_supply": {"rate_per_sqm": 8, "description": "Power for tools and equipment"},
        "transport": {"rate_pct": 3, "description": "Material transport to site"},
        "unloading": {"rate_pct": 1.5, "description": "Unloading and stacking at site"},
        "debris_removal": {"rate_pct": 2, "description": "Debris removal and cleaning"},
        "safety_equipment": {"rate_pct": 1, "description": "PPE and safety equipment"},
        "insurance": {"rate_pct": 0.5, "description": "Contractor insurance"},
        "supervision": {"rate_pct": 3, "description": "Site supervision"},
        "testing": {"rate_pct": 1.5, "description": "Testing and quality checks"},
    }

    def __init__(self):
        pass

    def normalize(
        self,
        quotes: List[SubcontractorQuote],
        boq_items: List[Dict],
    ) -> List[SubcontractorQuote]:
        """Normalize quotes to common basis."""
        normalized = []

        for quote in quotes:
            norm_quote = self._normalize_quote(quote, boq_items)
            normalized.append(norm_quote)

        return normalized

    def _normalize_quote(
        self,
        quote: SubcontractorQuote,
        boq_items: List[Dict],
    ) -> SubcontractorQuote:
        """Normalize a single quote."""
        # Create a copy
        norm_quote = SubcontractorQuote(
            subcontractor_name=quote.subcontractor_name,
            package=quote.package,
            quote_date=quote.quote_date,
            validity_days=quote.validity_days,
            line_items=quote.line_items.copy(),
            inclusions=quote.inclusions.copy(),
            exclusions=quote.exclusions.copy(),
            total_amount=quote.total_amount,
            gst_percent=quote.gst_percent,
            gst_included=quote.gst_included,
            payment_terms=quote.payment_terms,
            completion_days=quote.completion_days,
            warranty_months=quote.warranty_months,
            mobilization_advance=quote.mobilization_advance,
            retention_percent=quote.retention_percent,
            notes=quote.notes.copy(),
        )

        # 1. Normalize GST
        if norm_quote.gst_included:
            # Remove GST to compare on same basis
            gst_factor = 1 + (norm_quote.gst_percent / 100)
            norm_quote.total_amount = norm_quote.total_amount / gst_factor
            for item in norm_quote.line_items:
                item.rate = item.rate / gst_factor
                item.amount = item.amount / gst_factor
            norm_quote.gst_included = False

        # 2. Match items to BOQ
        norm_quote.line_items = self._match_to_boq(norm_quote.line_items, boq_items)

        # 3. Identify missing items
        covered_items = set()
        for item in norm_quote.line_items:
            covered_items.add(item.item_no)

        # Add missing items with zero rate (needs pricing)
        for boq_item in boq_items:
            boq_id = boq_item.get("unified_item_no", boq_item.get("item_no", ""))
            if boq_id not in covered_items:
                norm_quote.line_items.append(QuoteLineItem(
                    item_no=boq_id,
                    description=boq_item.get("description", ""),
                    unit=boq_item.get("unit", ""),
                    quantity=float(boq_item.get("quantity", 0)),
                    rate=0,
                    amount=0,
                ))
                norm_quote.notes.append(f"Item {boq_id} not quoted - added with zero rate")

        # 4. Recalculate total
        norm_quote.total_amount = sum(item.amount for item in norm_quote.line_items)

        return norm_quote

    def _match_to_boq(
        self,
        quote_items: List[QuoteLineItem],
        boq_items: List[Dict],
    ) -> List[QuoteLineItem]:
        """Match quote items to BOQ items."""
        matched = []
        boq_lookup = self._build_boq_lookup(boq_items)

        for item in quote_items:
            # Try to match by item number
            item_no = item.item_no.strip()
            if item_no in boq_lookup:
                boq = boq_lookup[item_no]
                matched.append(QuoteLineItem(
                    item_no=item_no,
                    description=item.description,
                    unit=item.unit or boq.get("unit", ""),
                    quantity=item.quantity if item.quantity > 0 else float(boq.get("quantity", 0)),
                    rate=item.rate,
                    amount=item.amount,
                ))
            else:
                # Try description matching
                best_match = self._find_best_match(item.description, boq_items)
                if best_match:
                    matched.append(QuoteLineItem(
                        item_no=best_match.get("unified_item_no", item.item_no),
                        description=item.description,
                        unit=item.unit or best_match.get("unit", ""),
                        quantity=item.quantity if item.quantity > 0 else float(best_match.get("quantity", 0)),
                        rate=item.rate,
                        amount=item.amount,
                    ))
                else:
                    matched.append(item)

        return matched

    def _build_boq_lookup(self, boq_items: List[Dict]) -> Dict:
        """Build lookup dictionary from BOQ items."""
        lookup = {}
        for item in boq_items:
            item_no = item.get("unified_item_no", item.get("item_no", ""))
            if item_no:
                lookup[item_no.strip()] = item
        return lookup

    def _find_best_match(self, description: str, boq_items: List[Dict]) -> Dict:
        """Find best matching BOQ item by description."""
        desc_lower = description.lower()
        best_score = 0
        best_match = None

        for boq in boq_items:
            boq_desc = boq.get("description", "").lower()

            # Calculate word overlap
            desc_words = set(desc_lower.split())
            boq_words = set(boq_desc.split())
            overlap = len(desc_words & boq_words)
            score = overlap / max(len(desc_words), len(boq_words), 1)

            if score > best_score and score >= 0.5:
                best_score = score
                best_match = boq

        return best_match

    def get_exclusion_adjustments(
        self,
        quote: SubcontractorQuote,
        total_area_sqm: float,
    ) -> List[Dict]:
        """Calculate adjustments for excluded items."""
        adjustments = []

        for exclusion in quote.exclusions:
            exc_lower = exclusion.lower()

            for exc_key, exc_info in self.COMMON_EXCLUSIONS.items():
                if exc_key.replace("_", " ") in exc_lower or exc_key in exc_lower:
                    if "rate_per_sqm" in exc_info:
                        amount = exc_info["rate_per_sqm"] * total_area_sqm
                    elif "rate_pct" in exc_info:
                        amount = quote.total_amount * (exc_info["rate_pct"] / 100)
                    else:
                        amount = 0

                    adjustments.append({
                        "item": exc_key,
                        "description": exc_info["description"],
                        "amount": round(amount, 2),
                        "type": "add",  # Add back to quote
                    })
                    break

        return adjustments
