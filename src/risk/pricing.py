"""
Risk Pricing Engine

Computes package-wise risk levels and contingency recommendations based on:
- Quantity certainty (from drawings confidence)
- Scope certainty (from scope completeness)
- Spec certainty (from owner inputs / specs)
- RFI count per package
- Missing input count

Output: risk_pricing.csv with risk level and suggested contingency %

India-specific construction risk assessment.
"""

import csv
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PackageRiskProfile:
    """Risk profile for a single package."""
    package: str
    package_name: str

    # Certainty scores (0-100)
    quantity_certainty: float = 0.0
    scope_certainty: float = 0.0
    spec_certainty: float = 0.0

    # Gap counts
    rfi_count: int = 0
    missing_inputs: int = 0
    provisional_items: int = 0

    # Value metrics
    package_value: float = 0.0
    provisional_value: float = 0.0

    # Derived
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.0
    suggested_contingency_pct: float = 0.0
    risk_drivers: List[str] = field(default_factory=list)


@dataclass
class RiskThresholds:
    """Thresholds for risk classification."""
    # Quantity certainty thresholds
    qty_high_confidence: float = 85.0
    qty_medium_confidence: float = 70.0
    qty_low_confidence: float = 50.0

    # Scope certainty thresholds
    scope_high_confidence: float = 90.0
    scope_medium_confidence: float = 75.0
    scope_low_confidence: float = 50.0

    # Spec certainty thresholds
    spec_high_confidence: float = 80.0
    spec_medium_confidence: float = 60.0
    spec_low_confidence: float = 40.0

    # RFI thresholds per package
    rfi_low: int = 0
    rfi_medium: int = 2
    rfi_high: int = 5

    # Contingency recommendations (%)
    contingency_low: float = 2.0
    contingency_medium: float = 5.0
    contingency_high: float = 10.0
    contingency_very_high: float = 15.0

    # Provisional threshold (% of package value)
    provisional_warning_pct: float = 20.0


class RiskPricingEngine:
    """
    Computes package-wise risk levels and contingency recommendations.

    Factors considered:
    1. Quantity certainty - How confident are we in quantities?
    2. Scope certainty - Is the scope well-defined?
    3. Spec certainty - Are specifications clear?
    4. RFI count - Outstanding clarifications
    5. Missing inputs - Gaps in owner requirements
    6. Provisional items - Items without detailed pricing
    """

    # Package display names
    PACKAGE_NAMES = {
        "rcc": "RCC Structural",
        "masonry": "Masonry & Plastering",
        "waterproof": "Waterproofing",
        "flooring": "Flooring & Tiling",
        "doors_windows": "Doors & Windows",
        "plumbing": "Plumbing Works",
        "electrical": "Electrical Works",
        "external": "External Development",
        "finishes": "Finishes & Painting",
        "prelims": "Preliminaries",
        "hvac": "HVAC",
        "firefighting": "Fire Fighting",
        "lift": "Lifts & Elevators",
    }

    def __init__(self, thresholds: RiskThresholds = None):
        self.thresholds = thresholds or RiskThresholds()
        self.package_risks: List[PackageRiskProfile] = []

    def analyze_risks(
        self,
        boq_items: List[Dict[str, Any]],
        rfis: List[Dict[str, Any]] = None,
        scope_data: Dict[str, Any] = None,
        owner_inputs_gaps: List[str] = None,
    ) -> List[PackageRiskProfile]:
        """
        Analyze risks across all packages.

        Args:
            boq_items: BOQ items with quantities and confidence
            rfis: List of RFIs with package tags
            scope_data: Scope completeness data
            owner_inputs_gaps: List of missing owner inputs

        Returns:
            List of PackageRiskProfile
        """
        self.package_risks = []
        rfis = rfis or []
        scope_data = scope_data or {}
        owner_inputs_gaps = owner_inputs_gaps or []

        # Group items by package
        packages = self._group_by_package(boq_items)

        # Index RFIs by package
        rfis_by_package = self._index_rfis_by_package(rfis)

        # Index missing inputs by package
        inputs_by_package = self._index_inputs_by_package(owner_inputs_gaps)

        for pkg_key, items in packages.items():
            profile = self._analyze_package(
                pkg_key,
                items,
                rfis_by_package.get(pkg_key, []),
                scope_data.get(pkg_key, {}),
                inputs_by_package.get(pkg_key, []),
            )
            self.package_risks.append(profile)

        # Sort by risk score (highest first)
        self.package_risks.sort(key=lambda x: -x.risk_score)

        return self.package_risks

    def _group_by_package(self, boq_items: List[Dict]) -> Dict[str, List[Dict]]:
        """Group BOQ items by package."""
        packages = {}
        for item in boq_items:
            pkg = item.get("package", "other")
            if pkg not in packages:
                packages[pkg] = []
            packages[pkg].append(item)
        return packages

    def _index_rfis_by_package(self, rfis: List[Dict]) -> Dict[str, List[Dict]]:
        """Index RFIs by package."""
        index = {}
        for rfi in rfis:
            pkg = rfi.get("package", rfi.get("category", "general"))
            # Map categories to packages
            pkg = self._map_category_to_package(pkg)
            if pkg not in index:
                index[pkg] = []
            index[pkg].append(rfi)
        return index

    def _map_category_to_package(self, category: str) -> str:
        """Map RFI category to package."""
        mappings = {
            "structural": "rcc",
            "architecture": "masonry",
            "architectural": "masonry",
            "mep": "plumbing",
            "mechanical": "hvac",
            "civil": "external",
            "interior": "finishes",
        }
        return mappings.get(category.lower(), category.lower())

    def _index_inputs_by_package(self, gaps: List[str]) -> Dict[str, List[str]]:
        """Index missing inputs by package."""
        index = {}

        # Keywords to package mapping
        keywords = {
            "rcc": ["concrete", "steel", "rcc", "structural", "foundation"],
            "masonry": ["masonry", "brick", "block", "plaster", "wall"],
            "waterproof": ["waterproof", "membrane", "terrace"],
            "flooring": ["floor", "tile", "granite", "marble"],
            "doors_windows": ["door", "window", "frame", "shutter"],
            "plumbing": ["plumb", "water", "sanitary", "pipe", "fitting"],
            "electrical": ["electric", "wiring", "switch", "light", "cable"],
            "external": ["external", "compound", "paving", "drain"],
            "finishes": ["paint", "polish", "finish", "ceiling"],
        }

        for gap in gaps:
            gap_lower = gap.lower()
            assigned = False
            for pkg, kws in keywords.items():
                if any(kw in gap_lower for kw in kws):
                    if pkg not in index:
                        index[pkg] = []
                    index[pkg].append(gap)
                    assigned = True
                    break
            if not assigned:
                if "general" not in index:
                    index["general"] = []
                index["general"].append(gap)

        return index

    def _analyze_package(
        self,
        pkg_key: str,
        items: List[Dict],
        rfis: List[Dict],
        scope: Dict,
        missing_inputs: List[str],
    ) -> PackageRiskProfile:
        """Analyze risk for a single package."""
        t = self.thresholds

        # Calculate certainty scores
        qty_certainty = self._calc_quantity_certainty(items)
        scope_certainty = self._calc_scope_certainty(scope, items)
        spec_certainty = self._calc_spec_certainty(items, missing_inputs)

        # Count metrics
        rfi_count = len(rfis)
        provisional_items = sum(1 for i in items if i.get("is_provisional", False))
        package_value = sum(i.get("amount", 0) for i in items)
        provisional_value = sum(i.get("amount", 0) for i in items if i.get("is_provisional", False))

        # Calculate composite risk score (0-100, higher = more risk)
        risk_score = self._calc_risk_score(
            qty_certainty, scope_certainty, spec_certainty,
            rfi_count, provisional_items, package_value, provisional_value
        )

        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)

        # Determine contingency
        contingency = self._determine_contingency(risk_level, provisional_value, package_value)

        # Identify risk drivers
        risk_drivers = self._identify_risk_drivers(
            qty_certainty, scope_certainty, spec_certainty,
            rfi_count, provisional_items, provisional_value, package_value
        )

        return PackageRiskProfile(
            package=pkg_key,
            package_name=self.PACKAGE_NAMES.get(pkg_key, pkg_key.replace("_", " ").title()),
            quantity_certainty=round(qty_certainty, 1),
            scope_certainty=round(scope_certainty, 1),
            spec_certainty=round(spec_certainty, 1),
            rfi_count=rfi_count,
            missing_inputs=len(missing_inputs),
            provisional_items=provisional_items,
            package_value=round(package_value, 2),
            provisional_value=round(provisional_value, 2),
            risk_level=risk_level,
            risk_score=round(risk_score, 1),
            suggested_contingency_pct=contingency,
            risk_drivers=risk_drivers,
        )

    def _calc_quantity_certainty(self, items: List[Dict]) -> float:
        """Calculate weighted average quantity certainty."""
        if not items:
            return 0.0

        total_value = sum(i.get("amount", 0) for i in items)
        if total_value == 0:
            return sum(i.get("confidence", 0.7) * 100 for i in items) / len(items)

        weighted_sum = sum(
            i.get("confidence", 0.7) * 100 * i.get("amount", 0)
            for i in items
        )
        return weighted_sum / total_value

    def _calc_scope_certainty(self, scope: Dict, items: List[Dict]) -> float:
        """Calculate scope certainty."""
        if not items:
            return 0.0

        # Use scope data if available
        if scope:
            return scope.get("completeness", 70.0)

        # Otherwise estimate from items
        provisional_count = sum(1 for i in items if i.get("is_provisional", False))
        total_count = len(items)

        if total_count == 0:
            return 0.0

        return (1 - provisional_count / total_count) * 100

    def _calc_spec_certainty(self, items: List[Dict], missing_inputs: List[str]) -> float:
        """Calculate specification certainty."""
        base_certainty = 80.0

        # Reduce for missing inputs
        for _ in missing_inputs:
            base_certainty -= 10

        # Reduce for items without rate source
        items_without_source = sum(
            1 for i in items
            if not i.get("rate_source") or i.get("rate_source") == "Allowance"
        )

        if items:
            base_certainty -= (items_without_source / len(items)) * 20

        return max(0, min(100, base_certainty))

    def _calc_risk_score(
        self,
        qty_cert: float,
        scope_cert: float,
        spec_cert: float,
        rfi_count: int,
        provisional_items: int,
        pkg_value: float,
        prov_value: float,
    ) -> float:
        """Calculate composite risk score (0-100)."""
        # Invert certainties to get risk contribution
        qty_risk = (100 - qty_cert) * 0.25
        scope_risk = (100 - scope_cert) * 0.25
        spec_risk = (100 - spec_cert) * 0.20

        # RFI contribution
        rfi_risk = min(30, rfi_count * 5)

        # Provisional contribution
        if pkg_value > 0:
            prov_pct = (prov_value / pkg_value) * 100
            prov_risk = min(20, prov_pct * 0.5)
        else:
            prov_risk = 10 if provisional_items > 0 else 0

        return qty_risk + scope_risk + spec_risk + rfi_risk + prov_risk

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score."""
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 45:
            return RiskLevel.MEDIUM
        elif risk_score < 65:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _determine_contingency(
        self,
        risk_level: RiskLevel,
        prov_value: float,
        pkg_value: float,
    ) -> float:
        """Determine suggested contingency percentage."""
        t = self.thresholds

        base = {
            RiskLevel.LOW: t.contingency_low,
            RiskLevel.MEDIUM: t.contingency_medium,
            RiskLevel.HIGH: t.contingency_high,
            RiskLevel.VERY_HIGH: t.contingency_very_high,
        }[risk_level]

        # Add extra for high provisional %
        if pkg_value > 0:
            prov_pct = (prov_value / pkg_value) * 100
            if prov_pct > 30:
                base += 5
            elif prov_pct > 15:
                base += 2.5

        return round(base, 1)

    def _identify_risk_drivers(
        self,
        qty_cert: float,
        scope_cert: float,
        spec_cert: float,
        rfi_count: int,
        provisional_items: int,
        prov_value: float,
        pkg_value: float,
    ) -> List[str]:
        """Identify top risk drivers for package."""
        drivers = []

        if qty_cert < 70:
            drivers.append(f"Low quantity confidence ({qty_cert:.0f}%)")
        if scope_cert < 70:
            drivers.append(f"Incomplete scope definition ({scope_cert:.0f}%)")
        if spec_cert < 60:
            drivers.append(f"Unclear specifications ({spec_cert:.0f}%)")
        if rfi_count > 2:
            drivers.append(f"{rfi_count} unresolved RFIs")
        if provisional_items > 0:
            drivers.append(f"{provisional_items} provisional items")
        if pkg_value > 0 and prov_value / pkg_value > 0.2:
            prov_pct = prov_value / pkg_value * 100
            drivers.append(f"High provisional value ({prov_pct:.0f}%)")

        return drivers

    def export_csv(self, output_path: Path) -> None:
        """Export risk pricing to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "package", "package_name",
                "quantity_certainty", "scope_certainty", "spec_certainty",
                "rfi_count", "missing_inputs", "provisional_items",
                "package_value", "provisional_value",
                "risk_level", "risk_score", "suggested_contingency_pct",
                "risk_drivers"
            ])

            for p in self.package_risks:
                writer.writerow([
                    p.package, p.package_name,
                    p.quantity_certainty, p.scope_certainty, p.spec_certainty,
                    p.rfi_count, p.missing_inputs, p.provisional_items,
                    f"{p.package_value:.2f}", f"{p.provisional_value:.2f}",
                    p.risk_level.value, p.risk_score, p.suggested_contingency_pct,
                    "; ".join(p.risk_drivers)
                ])

        logger.info(f"Risk pricing exported: {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get risk analysis summary."""
        if not self.package_risks:
            return {"error": "No analysis performed"}

        low_risk = [p for p in self.package_risks if p.risk_level == RiskLevel.LOW]
        medium_risk = [p for p in self.package_risks if p.risk_level == RiskLevel.MEDIUM]
        high_risk = [p for p in self.package_risks if p.risk_level == RiskLevel.HIGH]
        very_high_risk = [p for p in self.package_risks if p.risk_level == RiskLevel.VERY_HIGH]

        total_value = sum(p.package_value for p in self.package_risks)
        weighted_contingency = sum(
            p.suggested_contingency_pct * p.package_value / total_value
            for p in self.package_risks
        ) if total_value > 0 else 0

        return {
            "packages_analyzed": len(self.package_risks),
            "low_risk_packages": len(low_risk),
            "medium_risk_packages": len(medium_risk),
            "high_risk_packages": len(high_risk),
            "very_high_risk_packages": len(very_high_risk),
            "total_value": round(total_value, 2),
            "weighted_avg_contingency": round(weighted_contingency, 2),
            "top_risk_packages": [
                {"package": p.package_name, "risk_score": p.risk_score, "contingency": p.suggested_contingency_pct}
                for p in self.package_risks[:3]
            ],
        }


def run_risk_pricing(
    boq_items: List[Dict[str, Any]],
    rfis: List[Dict[str, Any]] = None,
    scope_data: Dict[str, Any] = None,
    owner_inputs_gaps: List[str] = None,
    output_path: Path = None,
) -> Dict[str, Any]:
    """
    Run risk pricing analysis.

    Args:
        boq_items: BOQ items with quantities and confidence
        rfis: List of RFIs
        scope_data: Scope completeness by package
        owner_inputs_gaps: Missing owner inputs
        output_path: Path to export CSV

    Returns:
        Analysis results
    """
    engine = RiskPricingEngine()
    risks = engine.analyze_risks(boq_items, rfis, scope_data, owner_inputs_gaps)

    if output_path:
        engine.export_csv(output_path)

    return {
        "package_risks": [
            {
                "package": p.package,
                "package_name": p.package_name,
                "risk_level": p.risk_level.value,
                "risk_score": p.risk_score,
                "suggested_contingency_pct": p.suggested_contingency_pct,
                "quantity_certainty": p.quantity_certainty,
                "scope_certainty": p.scope_certainty,
                "spec_certainty": p.spec_certainty,
                "package_value": p.package_value,
                "risk_drivers": p.risk_drivers,
            }
            for p in risks
        ],
        "summary": engine.get_summary(),
    }
