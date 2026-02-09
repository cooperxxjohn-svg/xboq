"""
Project Joiner - Joins schedules and drawings into unified project graph.

Phase 3: Build project-level connections:
- Match detected openings to door/window schedule entries
- Match structural elements to schedules
- Compute coverage metrics
- Generate unresolved tags report
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from .runner import RunnerResult, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class ScheduleEntry:
    """A single schedule entry (door, window, structural element)."""
    tag: str
    schedule_type: str  # door, window, bbs, column, etc.
    properties: Dict[str, Any]
    source_page: int
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectedElement:
    """An element detected in drawings."""
    element_id: str
    element_type: str  # opening, room, structural
    label: Optional[str]
    properties: Dict[str, Any]
    source_page: int
    source_file: str
    matched_tag: Optional[str] = None
    match_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TagMatch:
    """A match between schedule entry and detected element."""
    tag: str
    schedule_entry: ScheduleEntry
    detected_elements: List[DetectedElement]
    match_confidence: float
    match_method: str  # "exact", "ocr", "proximity"


@dataclass
class ProjectGraph:
    """Unified project graph connecting all elements."""
    project_id: str
    schedules: Dict[str, List[ScheduleEntry]]  # by schedule type
    detected_elements: Dict[str, List[DetectedElement]]  # by page
    tag_matches: List[TagMatch]
    unresolved_drawing_tags: List[str]  # Tags in drawings not in schedules
    unresolved_schedule_tags: List[str]  # Tags in schedules not in drawings
    coverage: Dict[str, float]  # Coverage metrics by type
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "schedules": {
                k: [e.to_dict() for e in v]
                for k, v in self.schedules.items()
            },
            "detected_elements": {
                k: [e.to_dict() for e in v]
                for k, v in self.detected_elements.items()
            },
            "tag_matches": [
                {
                    "tag": m.tag,
                    "schedule_entry": m.schedule_entry.to_dict(),
                    "detected_count": len(m.detected_elements),
                    "match_confidence": m.match_confidence,
                    "match_method": m.match_method,
                }
                for m in self.tag_matches
            ],
            "unresolved_drawing_tags": self.unresolved_drawing_tags,
            "unresolved_schedule_tags": self.unresolved_schedule_tags,
            "coverage": self.coverage,
            "summary": self.summary,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_unresolved_csv(self, path: Path) -> None:
        """Save unresolved tags to CSV."""
        with open(path, "w") as f:
            f.write("tag,source,type,notes\n")
            for tag in self.unresolved_drawing_tags:
                f.write(f"{tag},drawing,unmatched,Found in drawing but not in schedule\n")
            for tag in self.unresolved_schedule_tags:
                f.write(f"{tag},schedule,unused,In schedule but not found in drawings\n")


class ProjectJoiner:
    """
    Joins extraction results into a unified project graph.

    Matches:
    - Door/window tags (D1, W2, etc.) to schedule entries
    - Structural tags (C1, B1, F1) to structural schedules
    - Room labels to room schedules
    """

    # Tag patterns by type
    TAG_PATTERNS = {
        "door": [r"\bD\d+[A-Z]?\b", r"\bDR\d+\b"],
        "window": [r"\bW\d+[A-Z]?\b", r"\bWN\d+\b"],
        "column": [r"\bC\d+[A-Z]?\b", r"\bCOL\d+\b"],
        "beam": [r"\bB\d+[A-Z]?\b", r"\bBM\d+\b"],
        "footing": [r"\bF\d+[A-Z]?\b", r"\bFT\d+\b"],
    }

    def __init__(self):
        pass

    def join(
        self,
        runner_result: RunnerResult,
        output_dir: Path,
    ) -> ProjectGraph:
        """
        Join extraction results into project graph.

        Args:
            runner_result: Extraction results
            output_dir: Output directory

        Returns:
            ProjectGraph
        """
        output_dir = Path(output_dir) / runner_result.project_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Joining {len(runner_result.extraction_results)} extraction results")

        # Extract schedules
        schedules = self._extract_schedules(runner_result)
        logger.info(f"Extracted schedules: {list(schedules.keys())}")

        # Extract detected elements
        detected = self._extract_detected_elements(runner_result)
        logger.info(f"Extracted elements from {len(detected)} pages")

        # Build tag dictionary from schedules
        tag_dict = self._build_tag_dictionary(schedules)
        logger.info(f"Built tag dictionary with {len(tag_dict)} entries")

        # Match tags
        matches, drawing_tags_found = self._match_tags(tag_dict, detected)
        logger.info(f"Matched {len(matches)} tags")

        # Find unresolved
        schedule_tags = set(tag_dict.keys())
        unresolved_drawing = list(drawing_tags_found - schedule_tags)
        unresolved_schedule = list(schedule_tags - drawing_tags_found)

        # Compute coverage
        coverage = self._compute_coverage(
            schedules, detected, matches, tag_dict
        )

        # Build summary
        summary = {
            "total_schedule_entries": sum(len(v) for v in schedules.values()),
            "total_detected_elements": sum(len(v) for v in detected.values()),
            "matched_tags": len(matches),
            "unresolved_drawing_tags": len(unresolved_drawing),
            "unresolved_schedule_tags": len(unresolved_schedule),
            "coverage": coverage,
        }

        graph = ProjectGraph(
            project_id=runner_result.project_id,
            schedules=schedules,
            detected_elements=detected,
            tag_matches=matches,
            unresolved_drawing_tags=sorted(unresolved_drawing),
            unresolved_schedule_tags=sorted(unresolved_schedule),
            coverage=coverage,
            summary=summary,
        )

        # Save outputs
        graph.save(output_dir / "project_graph.json")
        graph.save_unresolved_csv(output_dir / "unresolved_tags.csv")

        logger.info(f"Saved project graph to: {output_dir}")

        return graph

    def _extract_schedules(
        self,
        runner_result: RunnerResult
    ) -> Dict[str, List[ScheduleEntry]]:
        """Extract all schedule entries from results."""
        schedules: Dict[str, List[ScheduleEntry]] = defaultdict(list)

        for result in runner_result.extraction_results:
            if result.page_type != "schedule_table":
                continue
            if not result.schedule_data:
                continue

            entries = result.schedule_data.get("entries", [])

            for entry in entries:
                tag = entry.get("tag", "")
                if not tag:
                    continue

                # Determine schedule type from tag
                schedule_type = self._classify_tag(tag)

                schedules[schedule_type].append(ScheduleEntry(
                    tag=tag.upper(),
                    schedule_type=schedule_type,
                    properties=entry,
                    source_page=result.page_number,
                    source_file=result.file_path,
                ))

        return dict(schedules)

    def _extract_detected_elements(
        self,
        runner_result: RunnerResult
    ) -> Dict[str, List[DetectedElement]]:
        """Extract detected elements from floor plans."""
        detected: Dict[str, List[DetectedElement]] = {}

        for result in runner_result.extraction_results:
            if result.page_type != "floor_plan":
                continue
            if not result.rooms:
                continue

            page_key = f"{Path(result.file_path).stem}_p{result.page_number}"
            elements = []

            for room in result.rooms:
                elements.append(DetectedElement(
                    element_id=room.get("room_id", ""),
                    element_type="room",
                    label=room.get("label"),
                    properties={
                        "area_sqm": room.get("area_sqm", 0),
                        "area_sqft": room.get("area_sqft", 0),
                    },
                    source_page=result.page_number,
                    source_file=result.file_path,
                ))

            if elements:
                detected[page_key] = elements

        return detected

    def _build_tag_dictionary(
        self,
        schedules: Dict[str, List[ScheduleEntry]]
    ) -> Dict[str, ScheduleEntry]:
        """Build tag -> entry dictionary."""
        tag_dict = {}

        for entries in schedules.values():
            for entry in entries:
                tag = entry.tag.upper()
                if tag not in tag_dict:
                    tag_dict[tag] = entry
                # If duplicate, keep first occurrence

        return tag_dict

    def _match_tags(
        self,
        tag_dict: Dict[str, ScheduleEntry],
        detected: Dict[str, List[DetectedElement]]
    ) -> Tuple[List[TagMatch], Set[str]]:
        """Match detected elements to schedule entries."""
        matches = []
        drawing_tags_found: Set[str] = set()

        # For now, simple approach: look for tags in element labels
        tag_to_elements: Dict[str, List[DetectedElement]] = defaultdict(list)

        for page_key, elements in detected.items():
            for element in elements:
                label = (element.label or "").upper()

                # Try to find tag in label
                for tag_type, patterns in self.TAG_PATTERNS.items():
                    for pattern in patterns:
                        found_tags = re.findall(pattern, label)
                        for tag in found_tags:
                            tag = tag.upper()
                            drawing_tags_found.add(tag)
                            element.matched_tag = tag
                            tag_to_elements[tag].append(element)

        # Create matches for tags that exist in both
        for tag, elements in tag_to_elements.items():
            if tag in tag_dict:
                matches.append(TagMatch(
                    tag=tag,
                    schedule_entry=tag_dict[tag],
                    detected_elements=elements,
                    match_confidence=0.9,
                    match_method="exact",
                ))

        return matches, drawing_tags_found

    def _classify_tag(self, tag: str) -> str:
        """Classify a tag by its type."""
        tag = tag.upper()

        for tag_type, patterns in self.TAG_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, tag):
                    return tag_type

        return "other"

    def _compute_coverage(
        self,
        schedules: Dict[str, List[ScheduleEntry]],
        detected: Dict[str, List[DetectedElement]],
        matches: List[TagMatch],
        tag_dict: Dict[str, ScheduleEntry],
    ) -> Dict[str, float]:
        """Compute coverage metrics."""
        coverage = {}

        # Schedule coverage: % of schedule entries that were matched
        total_schedule = len(tag_dict)
        matched_schedule = len(matches)
        coverage["schedule_matched"] = (
            matched_schedule / total_schedule if total_schedule > 0 else 1.0
        )

        # By type
        for stype, entries in schedules.items():
            matched = len([m for m in matches if m.schedule_entry.schedule_type == stype])
            total = len(entries)
            coverage[f"{stype}_matched"] = matched / total if total > 0 else 1.0

        # Element coverage: % of labeled elements that were matched
        total_labeled = sum(
            1 for elements in detected.values()
            for e in elements if e.label
        )
        matched_elements = sum(
            1 for elements in detected.values()
            for e in elements if e.matched_tag
        )
        coverage["elements_matched"] = (
            matched_elements / total_labeled if total_labeled > 0 else 1.0
        )

        return {k: round(v, 3) for k, v in coverage.items()}


def join_project(
    runner_result: RunnerResult,
    output_dir: Path,
) -> ProjectGraph:
    """
    Convenience function to join project.

    Args:
        runner_result: Extraction results
        output_dir: Output directory

    Returns:
        ProjectGraph
    """
    joiner = ProjectJoiner()
    return joiner.join(runner_result, output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")

        # Load results
        with open(results_path) as f:
            data = json.load(f)

        runner_result = RunnerResult(
            project_id=data["project_id"],
            total_pages=data["total_pages"],
            processed_pages=data["processed_pages"],
            skipped_pages=data["skipped_pages"],
            failed_pages=data["failed_pages"],
            extraction_results=[
                ExtractionResult(**r) for r in data["extraction_results"]
            ],
            total_time_sec=data["total_time_sec"],
            summary=data["summary"],
        )

        graph = join_project(runner_result, output_dir)

        print(f"\nProject Graph: {graph.project_id}")
        print(f"Summary: {graph.summary}")
    else:
        print("Usage: python joiner.py <extraction_results.json> [output_dir]")
