"""
Scope Register Generation - Track status of all scope packages.

For each scope package/subpackage:
- DETECTED: Hard evidence found + takeoff exists
- IMPLIED: Some evidence, but incomplete details
- UNKNOWN: No evidence, must confirm
- MISSING_INPUT: References sheets/details not present

Outputs: out/<project_id>/scope/scope_register.csv
"""

import csv
import json
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ScopeStatus(Enum):
    """Status of a scope item."""
    DETECTED = "DETECTED"      # Hard evidence + takeoff
    IMPLIED = "IMPLIED"        # Some evidence, incomplete
    UNKNOWN = "UNKNOWN"        # No evidence, must confirm
    MISSING_INPUT = "MISSING_INPUT"  # References missing sheets
    NOT_APPLICABLE = "N/A"     # Not applicable to project


@dataclass
class ScopeItem:
    """A single scope item in the register."""
    package: str
    subpackage: str
    package_name: str
    subpackage_name: str
    status: ScopeStatus
    confidence: float
    evidence_pages: List[str] = field(default_factory=list)
    evidence_snippets: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    recommended_action: str = ""
    has_takeoff: bool = False
    has_schedule: bool = False
    keywords_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package": self.package,
            "subpackage": self.subpackage,
            "package_name": self.package_name,
            "subpackage_name": self.subpackage_name,
            "status": self.status.value,
            "confidence": round(self.confidence, 2),
            "evidence_pages": self.evidence_pages,
            "evidence_snippets": self.evidence_snippets[:3],  # Limit
            "missing_info": self.missing_info,
            "recommended_action": self.recommended_action,
            "has_takeoff": self.has_takeoff,
            "has_schedule": self.has_schedule,
            "keywords_found": self.keywords_found[:10],  # Limit
        }

    def to_csv_row(self) -> List[str]:
        return [
            self.package,
            self.subpackage,
            self.status.value,
            f"{self.confidence:.0%}",
            ", ".join(self.evidence_pages[:5]),
            "; ".join(self.evidence_snippets[:2]),
            "; ".join(self.missing_info),
            self.recommended_action,
        ]


@dataclass
class ScopeRegister:
    """Complete scope register for a project."""
    project_id: str
    items: List[ScopeItem] = field(default_factory=list)
    by_status: Dict[str, List[ScopeItem]] = field(default_factory=dict)

    def add(self, item: ScopeItem) -> None:
        """Add item and update index."""
        self.items.append(item)
        status_key = item.status.value
        if status_key not in self.by_status:
            self.by_status[status_key] = []
        self.by_status[status_key].append(item)

    def get_summary(self) -> Dict[str, int]:
        return {status: len(items) for status, items in self.by_status.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "summary": self.get_summary(),
            "items": [item.to_dict() for item in self.items],
        }

    def save_json(self, path: Path) -> None:
        """Save as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved scope register JSON to: {path}")

    def save_csv(self, path: Path) -> None:
        """Save as CSV."""
        headers = [
            "Package", "Subpackage", "Status", "Confidence",
            "Evidence Pages", "Evidence Snippets", "Missing Info",
            "Recommended Action"
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in self.items:
                writer.writerow(item.to_csv_row())
        logger.info(f"Saved scope register CSV to: {path}")


class ScopeRegisterGenerator:
    """
    Generates scope register from evidence and extraction results.
    """

    TAXONOMY_PATH = Path("rules/scope_taxonomy.yaml")

    def __init__(self):
        self.taxonomy = self._load_taxonomy()

    def _load_taxonomy(self) -> Dict[str, Any]:
        """Load scope taxonomy from YAML."""
        if self.TAXONOMY_PATH.exists():
            with open(self.TAXONOMY_PATH) as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Taxonomy not found at {self.TAXONOMY_PATH}")
            return {"packages": {}}

    def generate(
        self,
        project_id: str,
        evidence_store: Dict[str, Any],
        extraction_results: List[Dict[str, Any]],
        project_graph: Dict[str, Any],
        boq_entries: List[Dict[str, Any]],
    ) -> ScopeRegister:
        """
        Generate scope register.

        Args:
            project_id: Project identifier
            evidence_store: Evidence store dict
            extraction_results: Extraction results list
            project_graph: Project graph dict
            boq_entries: BOQ entries list

        Returns:
            ScopeRegister
        """
        register = ScopeRegister(project_id=project_id)

        # Build lookups for faster checking
        evidence_items = evidence_store.get("items", [])
        evidence_by_keyword = self._build_evidence_keyword_index(evidence_items)
        schedules_found = self._get_schedules_found(project_graph)
        boq_categories = self._get_boq_categories(boq_entries)
        detected_elements = self._get_detected_elements(extraction_results)

        # Process each package in taxonomy
        packages = self.taxonomy.get("packages", {})

        for pkg_key, pkg_data in packages.items():
            pkg_name = pkg_data.get("name", pkg_key)
            subpackages = pkg_data.get("subpackages", {})

            for subpkg_key, subpkg_data in subpackages.items():
                item = self._evaluate_subpackage(
                    pkg_key, pkg_name,
                    subpkg_key, subpkg_data,
                    evidence_by_keyword,
                    schedules_found,
                    boq_categories,
                    detected_elements,
                    evidence_items,
                )
                register.add(item)

        return register

    def _build_evidence_keyword_index(
        self,
        evidence_items: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build index of evidence by matched keywords."""
        index: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence_items:
            for keyword in item.get("keywords_matched", []):
                keyword_lower = keyword.lower()
                if keyword_lower not in index:
                    index[keyword_lower] = []
                index[keyword_lower].append(item)
        return index

    def _get_schedules_found(self, project_graph: Dict[str, Any]) -> Set[str]:
        """Get set of schedule types found."""
        schedules = project_graph.get("schedules", {})
        found = set()
        for stype, entries in schedules.items():
            if entries:
                found.add(stype)
        return found

    def _get_boq_categories(self, boq_entries: List[Dict[str, Any]]) -> Set[str]:
        """Get set of BOQ categories present."""
        return {entry.get("category", "").lower() for entry in boq_entries}

    def _get_detected_elements(
        self,
        extraction_results: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Get detected element types."""
        detected = {
            "rooms": False,
            "doors": False,
            "windows": False,
            "walls": False,
            "structural": False,
        }

        for result in extraction_results:
            if result.get("rooms"):
                detected["rooms"] = True
            if result.get("openings"):
                for op in result.get("openings", []):
                    if op.get("type") == "door":
                        detected["doors"] = True
                    if op.get("type") == "window":
                        detected["windows"] = True
            if result.get("walls"):
                detected["walls"] = True
            if result.get("page_type") == "structural_plan":
                detected["structural"] = True

        return detected

    def _evaluate_subpackage(
        self,
        pkg_key: str,
        pkg_name: str,
        subpkg_key: str,
        subpkg_data: Dict[str, Any],
        evidence_by_keyword: Dict[str, List[Dict[str, Any]]],
        schedules_found: Set[str],
        boq_categories: Set[str],
        detected_elements: Dict[str, bool],
        all_evidence: List[Dict[str, Any]],
    ) -> ScopeItem:
        """Evaluate a single subpackage for status."""
        subpkg_name = subpkg_data.get("name", subpkg_key)
        keywords = subpkg_data.get("keywords", [])
        required_evidence = subpkg_data.get("required_evidence", [])
        minimum_artifacts = subpkg_data.get("minimum_artifacts", [])

        # Find matching evidence
        matched_evidence = []
        matched_keywords = []
        evidence_pages = set()
        evidence_snippets = []

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in evidence_by_keyword:
                for ev in evidence_by_keyword[keyword_lower]:
                    matched_evidence.append(ev)
                    matched_keywords.append(keyword)
                    page_ref = f"{Path(ev.get('source_file', '')).stem}_p{ev.get('source_page', 0) + 1}"
                    evidence_pages.add(page_ref)
                    if ev.get("snippet"):
                        evidence_snippets.append(ev["snippet"][:100])

        # Check for required artifacts
        missing_artifacts = []
        for artifact in minimum_artifacts:
            if artifact == "door_schedule" and "door" not in schedules_found:
                missing_artifacts.append("Door schedule not found")
            elif artifact == "window_schedule" and "window" not in schedules_found:
                missing_artifacts.append("Window schedule not found")
            elif artifact == "finish_schedule" and "finish" not in schedules_found:
                missing_artifacts.append("Finish schedule not found")
            elif artifact == "bbs_schedule" and "bbs" not in schedules_found:
                missing_artifacts.append("BBS schedule not found")
            elif artifact == "floor_plan" and not detected_elements.get("rooms"):
                missing_artifacts.append("Floor plan with rooms not detected")
            elif artifact == "structural_drawing" and not detected_elements.get("structural"):
                missing_artifacts.append("Structural drawing not detected")

        # Check for takeoff presence
        has_takeoff = self._check_has_takeoff(
            pkg_key, subpkg_key, boq_categories, detected_elements
        )

        # Check for schedule
        has_schedule = self._check_has_schedule(pkg_key, subpkg_key, schedules_found)

        # Determine status and confidence
        status, confidence, missing_info, action = self._determine_status(
            matched_evidence,
            matched_keywords,
            missing_artifacts,
            required_evidence,
            has_takeoff,
            has_schedule,
            subpkg_name,
        )

        return ScopeItem(
            package=pkg_key,
            subpackage=subpkg_key,
            package_name=pkg_name,
            subpackage_name=subpkg_name,
            status=status,
            confidence=confidence,
            evidence_pages=list(evidence_pages),
            evidence_snippets=evidence_snippets[:5],
            missing_info=missing_info,
            recommended_action=action,
            has_takeoff=has_takeoff,
            has_schedule=has_schedule,
            keywords_found=list(set(matched_keywords)),
        )

    def _check_has_takeoff(
        self,
        pkg_key: str,
        subpkg_key: str,
        boq_categories: Set[str],
        detected_elements: Dict[str, bool],
    ) -> bool:
        """Check if takeoff exists for this scope item."""
        # Map packages to BOQ categories and detected elements
        takeoff_map = {
            ("rcc_concrete", "rcc_substructure"): "floor area" in boq_categories,
            ("rcc_concrete", "rcc_superstructure"): "floor area" in boq_categories,
            ("doors_windows", "wood_doors"): "doors" in boq_categories,
            ("doors_windows", "metal_doors"): "doors" in boq_categories,
            ("doors_windows", "windows"): "windows" in boq_categories,
            ("reinforcement_steel", "tmt_bars"): "reinforcement" in boq_categories,
            ("finishes_floor", "vitrified_tiles"): detected_elements.get("rooms", False),
            ("finishes_wall", "painting"): detected_elements.get("rooms", False),
        }

        key = (pkg_key, subpkg_key)
        return takeoff_map.get(key, False)

    def _check_has_schedule(
        self,
        pkg_key: str,
        subpkg_key: str,
        schedules_found: Set[str],
    ) -> bool:
        """Check if relevant schedule exists."""
        schedule_map = {
            ("doors_windows", "wood_doors"): "door" in schedules_found,
            ("doors_windows", "metal_doors"): "door" in schedules_found,
            ("doors_windows", "windows"): "window" in schedules_found,
            ("reinforcement_steel", "tmt_bars"): "bbs" in schedules_found,
            ("finishes_floor", "vitrified_tiles"): "finish" in schedules_found,
            ("finishes_wall", "wall_tiles"): "finish" in schedules_found,
        }

        key = (pkg_key, subpkg_key)
        return schedule_map.get(key, False)

    def _determine_status(
        self,
        matched_evidence: List[Dict[str, Any]],
        matched_keywords: List[str],
        missing_artifacts: List[str],
        required_evidence: List[str],
        has_takeoff: bool,
        has_schedule: bool,
        subpkg_name: str,
    ) -> tuple:
        """
        Determine scope status based on evidence.

        Returns:
            Tuple of (status, confidence, missing_info, recommended_action)
        """
        missing_info = list(missing_artifacts)
        action = ""

        # Calculate evidence score
        evidence_score = len(matched_evidence) * 0.2 + len(set(matched_keywords)) * 0.1

        if has_takeoff:
            evidence_score += 0.3
        if has_schedule:
            evidence_score += 0.2

        # Cap at 1.0
        confidence = min(evidence_score, 1.0)

        # Determine status
        if missing_artifacts:
            status = ScopeStatus.MISSING_INPUT
            action = f"Provide: {'; '.join(missing_artifacts)}"

        elif has_takeoff and has_schedule and len(matched_evidence) >= 2:
            status = ScopeStatus.DETECTED
            confidence = max(confidence, 0.8)

        elif has_takeoff or has_schedule or len(matched_evidence) >= 1:
            status = ScopeStatus.IMPLIED
            if not has_schedule:
                missing_info.append(f"No dedicated schedule for {subpkg_name}")
            if not has_takeoff:
                missing_info.append(f"No takeoff quantities for {subpkg_name}")
            action = f"Confirm scope and quantities for {subpkg_name}"

        else:
            status = ScopeStatus.UNKNOWN
            missing_info.append(f"No evidence found for {subpkg_name}")
            action = f"Confirm if {subpkg_name} is included in scope"
            confidence = 0.0

        return status, confidence, missing_info, action


def generate_scope_register(
    project_id: str,
    evidence_store: Dict[str, Any],
    extraction_results: List[Dict[str, Any]],
    project_graph: Dict[str, Any],
    boq_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> ScopeRegister:
    """
    Convenience function to generate and save scope register.

    Args:
        project_id: Project identifier
        evidence_store: Evidence store dict
        extraction_results: Extraction results
        project_graph: Project graph dict
        boq_entries: BOQ entries
        output_dir: Output directory

    Returns:
        ScopeRegister
    """
    generator = ScopeRegisterGenerator()
    register = generator.generate(
        project_id, evidence_store, extraction_results, project_graph, boq_entries
    )

    # Save files
    scope_dir = output_dir / "scope"
    scope_dir.mkdir(parents=True, exist_ok=True)
    register.save_csv(scope_dir / "scope_register.csv")
    register.save_json(scope_dir / "scope_register.json")

    return register
