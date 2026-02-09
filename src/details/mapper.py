"""
Scope Mapper - Map extracted details to scope register.

Provides:
- Detail to scope item mapping
- IMPLIED to DETECTED status promotion
- RFI reduction based on detail evidence
- New evidence addition to scope register
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from .classifier import DetailType
from .extractor import ExtractedDetail


@dataclass
class ScopeMappingResult:
    """Result of mapping details to scope."""
    promoted_items: List[Dict] = field(default_factory=list)
    reduced_rfis: List[str] = field(default_factory=list)
    new_evidence: List[Dict] = field(default_factory=list)
    details_by_type: Dict[str, int] = field(default_factory=dict)
    packages_by_type: Dict[str, List[str]] = field(default_factory=dict)
    unmapped_details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "promoted_items": self.promoted_items,
            "reduced_rfis": self.reduced_rfis,
            "new_evidence": self.new_evidence,
            "details_by_type": self.details_by_type,
            "packages_by_type": self.packages_by_type,
            "unmapped_details": self.unmapped_details,
        }


class ScopeMapper:
    """Map details to scope register and RFIs."""

    # Mapping from detail types to scope packages
    DETAIL_TO_PACKAGE = {
        # Waterproofing
        DetailType.TOILET_WATERPROOFING.value: [
            "waterproofing",
            "plumbing_sanitary",
            "finishes_floor",
        ],
        DetailType.TERRACE_WATERPROOFING.value: [
            "waterproofing",
            "external_works",
            "finishes_floor",
        ],
        DetailType.BALCONY_WATERPROOFING.value: [
            "waterproofing",
            "external_works",
        ],
        DetailType.WATER_TANK_WATERPROOFING.value: [
            "waterproofing",
            "plumbing_water_supply",
        ],
        DetailType.BASEMENT_WATERPROOFING.value: [
            "waterproofing",
            "earthwork",
        ],
        DetailType.PLINTH_PROTECTION.value: [
            "waterproofing",
            "external_works",
            "masonry_partitions",
        ],

        # Architectural
        DetailType.PARAPET.value: [
            "masonry_partitions",
            "external_works",
            "waterproofing",
        ],
        DetailType.WINDOW_SILL.value: [
            "doors_windows",
            "masonry_partitions",
            "finishes_wall",
        ],
        DetailType.DOOR_FRAME.value: [
            "doors_windows",
            "masonry_partitions",
        ],
        DetailType.STAIR.value: [
            "rcc_concrete",
            "finishes_floor",
            "miscellaneous",
        ],
        DetailType.RAILING.value: [
            "miscellaneous",
            "external_works",
        ],
        DetailType.EXPANSION_JOINT.value: [
            "rcc_concrete",
            "miscellaneous",
        ],
        DetailType.KITCHEN_PLATFORM.value: [
            "finishes_wall",
            "plumbing_sanitary",
        ],
        DetailType.DADO.value: [
            "finishes_wall",
            "tiles_finishes",
        ],

        # Structural
        DetailType.BEAM_COLUMN_JUNCTION.value: [
            "rcc_concrete",
            "reinforcement_steel",
        ],
        DetailType.SLAB_EDGE.value: [
            "rcc_concrete",
            "formwork_shuttering",
        ],
        DetailType.FOUNDATION.value: [
            "rcc_concrete",
            "earthwork",
        ],
        DetailType.STAIRCASE_STRUCTURAL.value: [
            "rcc_concrete",
            "reinforcement_steel",
        ],

        # MEP
        DetailType.PLUMBING_RISER.value: [
            "plumbing_water_supply",
            "plumbing_sanitary_swr",
        ],
        DetailType.SANITARY_CONNECTION.value: [
            "plumbing_sanitary_swr",
        ],
        DetailType.ELECTRICAL_PANEL.value: [
            "electrical_power_lighting",
        ],
        DetailType.CONDUIT_LAYOUT.value: [
            "electrical_power_lighting",
        ],
        DetailType.AC_DRAIN.value: [
            "fire_hvac",
            "plumbing_sanitary_swr",
        ],
        DetailType.RAINWATER_PIPE.value: [
            "plumbing_storm_rainwater",
            "external_works_drainage",
        ],
        DetailType.FIRE_HYDRANT.value: [
            "fire_hvac",
        ],

        # Finishes
        DetailType.FLOOR_PATTERN.value: [
            "finishes_floor",
            "tiles_finishes",
        ],
        DetailType.CEILING_DETAIL.value: [
            "finishes_ceiling",
        ],
        DetailType.WALL_CLADDING.value: [
            "finishes_wall",
            "external_works",
        ],
    }

    # Keywords for matching RFIs
    RFI_KEYWORDS = {
        "toilet_waterproofing": [
            "toilet waterproof",
            "bathroom waterproof",
            "wet area",
            "wc waterproof",
        ],
        "terrace_waterproofing": [
            "terrace waterproof",
            "roof waterproof",
            "terrace treatment",
            "flat roof",
        ],
        "parapet": [
            "parapet detail",
            "parapet height",
            "coping",
        ],
        "window_sill": [
            "window sill",
            "sill detail",
            "drip",
        ],
        "stair": [
            "stair detail",
            "tread riser",
            "nosing",
        ],
        "plumbing_riser": [
            "shaft size",
            "riser detail",
            "pipe shaft",
        ],
    }

    def __init__(self):
        pass

    def map_to_scope(
        self,
        details: List[ExtractedDetail],
        scope_register: Dict,
        rfi_list: List,
    ) -> ScopeMappingResult:
        """
        Map extracted details to scope register and RFIs.

        Args:
            details: Extracted details from detail pages
            scope_register: Current scope register
            rfi_list: Current RFI list

        Returns:
            ScopeMappingResult with mappings
        """
        result = ScopeMappingResult()

        # Count details by type
        for detail in details:
            dtype = detail.detail_type
            result.details_by_type[dtype] = result.details_by_type.get(dtype, 0) + 1

            # Track packages
            packages = self.DETAIL_TO_PACKAGE.get(dtype, [])
            if dtype not in result.packages_by_type:
                result.packages_by_type[dtype] = []
            for pkg in packages:
                if pkg not in result.packages_by_type[dtype]:
                    result.packages_by_type[dtype].append(pkg)

        # Map to scope items
        scope_items = scope_register.get("items", [])
        if isinstance(scope_items, dict):
            scope_items = list(scope_items.values())

        for detail in details:
            mapped = self._map_detail_to_scope(detail, scope_items, result)
            if not mapped:
                result.unmapped_details.append(detail.detail_id)

        # Reduce RFIs
        self._reduce_rfis(details, rfi_list, result)

        return result

    def _map_detail_to_scope(
        self,
        detail: ExtractedDetail,
        scope_items: List[Dict],
        result: ScopeMappingResult,
    ) -> bool:
        """Map a single detail to scope items."""
        detail_type = detail.detail_type
        target_packages = self.DETAIL_TO_PACKAGE.get(detail_type, [])

        if not target_packages:
            return False

        mapped = False

        for item in scope_items:
            item_package = item.get("package", "").lower()
            item_subpackage = item.get("subpackage", "").lower()
            item_status = item.get("status", "").lower()

            # Check if package matches
            package_matches = False
            for target in target_packages:
                if target in item_package or target in item_subpackage:
                    package_matches = True
                    break

            if not package_matches:
                continue

            # Add evidence
            evidence_text = self._build_evidence_text(detail)
            result.new_evidence.append({
                "item_id": item.get("item_id", ""),
                "package": item.get("package", ""),
                "text": evidence_text,
                "source": detail.source_page,
                "detail_type": detail_type,
            })

            # Promote IMPLIED to DETECTED
            if item_status == "implied" or item_status == "missing":
                result.promoted_items.append({
                    "item_id": item.get("item_id", ""),
                    "package": item.get("package", ""),
                    "subpackage": item.get("subpackage", ""),
                    "old_status": item_status,
                    "new_status": "detected",
                    "evidence": evidence_text,
                    "detail_source": detail.source_page,
                })
                mapped = True

        return mapped

    def _build_evidence_text(self, detail: ExtractedDetail) -> str:
        """Build evidence text from detail."""
        parts = [f"Detail found: {detail.detail_type}"]

        if detail.title:
            parts.append(f"Title: {detail.title}")

        if detail.materials:
            parts.append(f"Materials: {', '.join(detail.materials[:3])}")

        if detail.dimensions:
            dim_str = ", ".join(f"{k}={v}" for k, v in list(detail.dimensions.items())[:3])
            parts.append(f"Dimensions: {dim_str}")

        if detail.layers:
            parts.append(f"Layers: {len(detail.layers)} specified")

        return "; ".join(parts)

    def _reduce_rfis(
        self,
        details: List[ExtractedDetail],
        rfi_list: List,
        result: ScopeMappingResult,
    ) -> None:
        """Reduce RFIs that are answered by details."""
        # Build set of detail types found
        found_types = set(d.detail_type for d in details)

        # Check each RFI
        for rfi in rfi_list:
            if isinstance(rfi, dict):
                rfi_id = rfi.get("rfi_id", "")
                question = rfi.get("question", "").lower()
                package = rfi.get("package", "").lower()
            else:
                continue

            # Check if RFI is about a detail type we have
            for detail_type, keywords in self.RFI_KEYWORDS.items():
                if detail_type in found_types or detail_type.replace("_", " ") in found_types:
                    # Check if RFI matches
                    for keyword in keywords:
                        if keyword in question or keyword in package:
                            if rfi_id not in result.reduced_rfis:
                                result.reduced_rfis.append(rfi_id)
                            break

    def update_scope_register(
        self,
        scope_register: Dict,
        mapping_result: ScopeMappingResult,
    ) -> Dict:
        """
        Update scope register with detail mappings.

        Args:
            scope_register: Original scope register
            mapping_result: Mapping result from map_to_scope

        Returns:
            Updated scope register
        """
        import copy
        updated = copy.deepcopy(scope_register)

        scope_items = updated.get("items", [])
        if isinstance(scope_items, dict):
            scope_items = list(scope_items.values())

        # Build lookup by item_id
        item_lookup = {
            item.get("item_id"): item for item in scope_items
        }

        # Apply promotions
        for promotion in mapping_result.promoted_items:
            item_id = promotion["item_id"]
            if item_id in item_lookup:
                item_lookup[item_id]["status"] = "detected"
                item_lookup[item_id]["detail_evidence"] = promotion["evidence"]
                item_lookup[item_id]["detail_source"] = promotion["detail_source"]

        # Add new evidence
        for evidence in mapping_result.new_evidence:
            item_id = evidence["item_id"]
            if item_id in item_lookup:
                if "evidence_list" not in item_lookup[item_id]:
                    item_lookup[item_id]["evidence_list"] = []
                item_lookup[item_id]["evidence_list"].append({
                    "text": evidence["text"],
                    "source": evidence["source"],
                    "type": "detail",
                })

        updated["items"] = list(item_lookup.values())

        return updated

    def get_coverage_by_package(
        self,
        mapping_result: ScopeMappingResult,
    ) -> Dict[str, Dict]:
        """Get detail coverage summary by package."""
        coverage = {}

        for detail_type, packages in mapping_result.packages_by_type.items():
            count = mapping_result.details_by_type.get(detail_type, 0)

            for package in packages:
                if package not in coverage:
                    coverage[package] = {
                        "detail_types": [],
                        "total_details": 0,
                        "promoted_items": 0,
                    }

                coverage[package]["detail_types"].append(detail_type)
                coverage[package]["total_details"] += count

        # Count promoted items by package
        for promotion in mapping_result.promoted_items:
            package = promotion.get("package", "").lower()
            for pkg_key in coverage:
                if pkg_key in package:
                    coverage[pkg_key]["promoted_items"] += 1

        return coverage
