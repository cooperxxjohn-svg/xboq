"""
BOQ Reconciler - Create unified BOQ from matched items.
"""

from typing import List, Dict, Tuple


class BOQReconciler:
    """Reconcile drawings BOQ and owner BOQ into unified BOQ."""

    def reconcile(
        self,
        matches: List,
        discrepancies: List,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Reconcile matches into unified BOQ.

        Returns:
            Tuple of (unified_boq, reconciliation_notes)
        """
        unified_boq = []
        reconciliation_notes = []
        item_counter = 1

        # Build discrepancy lookup
        disc_lookup = {d.item_id: d for d in discrepancies}

        for match in matches:
            if match.match_type == "matched":
                item = self._reconcile_matched(
                    match.drawings_item,
                    match.owner_item,
                    disc_lookup.get(match.drawings_item.get("item_id", "")),
                    item_counter,
                )
                unified_boq.append(item)

                # Add note if significant discrepancy
                disc = disc_lookup.get(match.drawings_item.get("item_id", ""))
                if disc and abs(disc.difference_percent) > 20:
                    reconciliation_notes.append(
                        f"Item {item['unified_item_no']}: Qty difference {disc.difference_percent:.1f}% - "
                        f"used {item['quantity_source']} quantity"
                    )

            elif match.match_type == "drawings_only":
                item = self._create_from_drawings(match.drawings_item, item_counter)
                unified_boq.append(item)
                reconciliation_notes.append(
                    f"Item {item['unified_item_no']}: From drawings only - not in owner BOQ"
                )

            elif match.match_type == "owner_only":
                item = self._create_from_owner(match.owner_item, item_counter)
                unified_boq.append(item)
                reconciliation_notes.append(
                    f"Item {item['unified_item_no']}: From owner BOQ only - not found in drawings"
                )

            item_counter += 1

        return unified_boq, reconciliation_notes

    def _reconcile_matched(
        self,
        drawings_item: Dict,
        owner_item: Dict,
        discrepancy,
        item_no: int,
    ) -> Dict:
        """Reconcile a matched item pair."""
        # Get quantities
        d_qty = float(drawings_item.get("quantity", 0) or 0)
        o_qty = float(owner_item.get("quantity", 0) or 0)

        # Determine which quantity to use
        if discrepancy:
            abs_diff = abs(discrepancy.difference_percent)
            if abs_diff <= 10:
                # Within tolerance - use owner BOQ (contractual)
                use_qty = o_qty
                qty_source = "owner_boq"
            elif abs_diff <= 30:
                # Moderate difference - use average
                use_qty = (d_qty + o_qty) / 2
                qty_source = "average"
            else:
                # Large difference - use drawings (more accurate measurement)
                use_qty = d_qty
                qty_source = "drawings"
        else:
            # No discrepancy - use owner BOQ
            use_qty = o_qty
            qty_source = "owner_boq"

        # Prefer owner description (contractual) but note differences
        description = owner_item.get("description", drawings_item.get("description", ""))

        return {
            "unified_item_no": f"U-{item_no:04d}",
            "drawings_item_id": drawings_item.get("item_id", ""),
            "owner_item_no": owner_item.get("item_no", ""),
            "description": description,
            "unit": owner_item.get("unit", drawings_item.get("unit", "")),
            "quantity": round(use_qty, 2),
            "drawings_qty": d_qty,
            "owner_qty": o_qty,
            "quantity_source": qty_source,
            "package": drawings_item.get("package", self._classify_package(description)),
            "rate": owner_item.get("rate", 0),
            "amount": round(use_qty * float(owner_item.get("rate", 0) or 0), 2),
            "match_confidence": 1.0,
            "status": "matched",
            "remarks": "",
        }

    def _create_from_drawings(self, drawings_item: Dict, item_no: int) -> Dict:
        """Create unified item from drawings-only item."""
        qty = float(drawings_item.get("quantity", 0) or 0)
        description = drawings_item.get("description", "")

        return {
            "unified_item_no": f"U-{item_no:04d}",
            "drawings_item_id": drawings_item.get("item_id", ""),
            "owner_item_no": "",
            "description": description,
            "unit": drawings_item.get("unit", ""),
            "quantity": round(qty, 2),
            "drawings_qty": qty,
            "owner_qty": 0,
            "quantity_source": "drawings",
            "package": drawings_item.get("package", self._classify_package(description)),
            "rate": 0,  # Rate to be added
            "amount": 0,
            "match_confidence": 0,
            "status": "drawings_only",
            "remarks": "NOT IN OWNER BOQ - Confirm scope inclusion",
        }

    def _create_from_owner(self, owner_item: Dict, item_no: int) -> Dict:
        """Create unified item from owner-only item."""
        qty = float(owner_item.get("quantity", 0) or 0)
        rate = float(owner_item.get("rate", 0) or 0)
        description = owner_item.get("description", "")

        return {
            "unified_item_no": f"U-{item_no:04d}",
            "drawings_item_id": "",
            "owner_item_no": owner_item.get("item_no", ""),
            "description": description,
            "unit": owner_item.get("unit", ""),
            "quantity": round(qty, 2),
            "drawings_qty": 0,
            "owner_qty": qty,
            "quantity_source": "owner_boq",
            "package": self._classify_package(description),
            "rate": rate,
            "amount": round(qty * rate, 2),
            "match_confidence": 0,
            "status": "owner_only",
            "remarks": "NOT IN DRAWINGS - Verify scope or treat as provisional",
        }

    def _classify_package(self, description: str) -> str:
        """Classify item into work package."""
        desc_lower = description.lower()

        package_keywords = {
            "civil_structural": ["rcc", "pcc", "concrete", "footing", "column", "beam", "slab", "foundation", "staircase"],
            "masonry": ["brick", "block", "aac", "masonry", "wall"],
            "plaster_finishes": ["plaster", "putty", "paint", "emulsion", "distemper", "texture"],
            "flooring": ["flooring", "tile", "vitrified", "ceramic", "marble", "granite", "kota"],
            "waterproofing": ["waterproof", "wp", "app", "membrane", "coba"],
            "doors_windows": ["door", "window", "shutter", "frame", "grill"],
            "plumbing": ["pipe", "cpvc", "upvc", "swr", "drainage", "water supply", "sanitary"],
            "electrical": ["wiring", "point", "conduit", "switch", "mcb", "db", "earthing"],
            "external_works": ["compound", "gate", "paving", "landscaping"],
        }

        for package, keywords in package_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return package

        return "miscellaneous"

    def generate_reconciliation_summary(
        self,
        unified_boq: List[Dict],
    ) -> Dict:
        """Generate summary statistics for reconciliation."""
        total_items = len(unified_boq)
        matched = len([i for i in unified_boq if i["status"] == "matched"])
        drawings_only = len([i for i in unified_boq if i["status"] == "drawings_only"])
        owner_only = len([i for i in unified_boq if i["status"] == "owner_only"])

        # Package-wise breakdown
        by_package = {}
        for item in unified_boq:
            pkg = item["package"]
            if pkg not in by_package:
                by_package[pkg] = {"count": 0, "amount": 0}
            by_package[pkg]["count"] += 1
            by_package[pkg]["amount"] += item["amount"]

        # Quantity source breakdown
        by_source = {}
        for item in unified_boq:
            src = item["quantity_source"]
            if src not in by_source:
                by_source[src] = 0
            by_source[src] += 1

        return {
            "total_items": total_items,
            "matched": matched,
            "drawings_only": drawings_only,
            "owner_only": owner_only,
            "by_package": by_package,
            "by_quantity_source": by_source,
            "total_amount": sum(i["amount"] for i in unified_boq),
        }
