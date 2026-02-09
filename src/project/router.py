"""
Page Router - Classifies pages and routes to appropriate extractors.

Phase 1: Classify indexed pages into:
- floor_plan, schedule_table, structural_plan
- section_elevation, detail, notes_specs
- cover, unknown
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

from .indexer import ProjectIndex, PageIndex

logger = logging.getLogger(__name__)


class PageType(Enum):
    """Classification of page types."""
    FLOOR_PLAN = "floor_plan"
    SCHEDULE_TABLE = "schedule_table"
    STRUCTURAL_PLAN = "structural_plan"
    SECTION_ELEVATION = "section_elevation"
    DETAIL = "detail"
    NOTES_SPECS = "notes_specs"
    COVER = "cover"
    UNKNOWN = "unknown"


@dataclass
class PageRouting:
    """Routing decision for a single page."""
    file_path: str
    page_number: int
    page_type: PageType
    confidence: float
    schedule_subtype: Optional[str] = None  # door, window, finish, bbs, structural
    matched_keywords: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "page_number": self.page_number,
            "page_type": self.page_type.value,
            "confidence": round(self.confidence, 2),
            "schedule_subtype": self.schedule_subtype,
            "matched_keywords": self.matched_keywords,
            "warnings": self.warnings,
        }


@dataclass
class RoutingResult:
    """Routing result for entire project."""
    project_id: str
    total_pages: int
    pages_by_type: Dict[str, List[PageRouting]]
    type_counts: Dict[str, int]
    ambiguous_pages: List[PageRouting]
    routing_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "total_pages": self.total_pages,
            "type_counts": self.type_counts,
            "pages_by_type": {
                ptype: [p.to_dict() for p in pages]
                for ptype, pages in self.pages_by_type.items()
            },
            "ambiguous_pages": [p.to_dict() for p in self.ambiguous_pages],
            "routing_summary": self.routing_summary,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_pages(self, page_type: PageType) -> List[PageRouting]:
        """Get pages of a specific type."""
        return self.pages_by_type.get(page_type.value, [])


class PageRouter:
    """
    Routes pages to appropriate extractors based on classification.

    Uses title keywords, sheet numbers, and visual features.
    India terminology prioritized.
    """

    # Keyword patterns by page type (India-specific included)
    KEYWORDS = {
        PageType.FLOOR_PLAN: {
            "title_keywords": [
                r"(?:GROUND|FIRST|SECOND|THIRD|FOURTH|TYPICAL|TERRACE|BASEMENT|STILT|PODIUM|REFUGE)\s*(?:FLOOR)?\s*(?:PLAN|LAYOUT)",
                r"(?:GF|FF|SF|TF|UGF)\s*(?:PLAN|LAYOUT)",
                r"FLOOR\s*PLAN",
                r"(?:UNIT|FLAT|APARTMENT)\s*PLAN",
                r"LAYOUT\s*PLAN",
            ],
            "sheet_prefixes": ["A-", "AR-", "AP-", "ARCH-"],
            "weight": 1.0,
        },
        PageType.SCHEDULE_TABLE: {
            "title_keywords": [
                r"(?:DOOR|WINDOW|FINISH|ROOM|HARDWARE|FIXTURE)\s*SCHEDULE",
                r"(?:BAR\s*BENDING|BBS)\s*SCHEDULE",
                r"AREA\s*STATEMENT",
                r"SPECIFICATIONS?\b",
                r"(?:STEEL|REINFORCEMENT)\s*SCHEDULE",
            ],
            "sheet_prefixes": ["SCH-", "SC-"],
            "weight": 1.0,
        },
        PageType.STRUCTURAL_PLAN: {
            "title_keywords": [
                r"(?:COLUMN|BEAM|FOOTING|FOUNDATION|PLINTH|SLAB)\s*(?:LAYOUT|PLAN)",
                r"(?:STRUCTURAL|RCC|FRAMING)\s*(?:PLAN|LAYOUT)",
                r"(?:LINTEL|TIE\s*BEAM)\s*(?:LAYOUT|PLAN)",
                r"REINFORCEMENT\s*(?:LAYOUT|PLAN)",
            ],
            "sheet_prefixes": ["S-", "ST-", "STR-", "COL-", "FND-"],
            "weight": 1.0,
        },
        PageType.SECTION_ELEVATION: {
            "title_keywords": [
                r"SECTION\s*[A-Z]{1,2}(?:\s*-\s*[A-Z])?",
                r"(?:CROSS|LONGITUDINAL)\s*SECTION",
                r"(?:FRONT|SIDE|REAR|NORTH|SOUTH|EAST|WEST)\s*ELEVATION",
                r"SECTIONAL\s*ELEVATION",
            ],
            "sheet_prefixes": ["SEC-", "E-", "EL-"],
            "weight": 0.9,
        },
        PageType.DETAIL: {
            "title_keywords": [
                r"(?:TOILET|KITCHEN|STAIRCASE|RAILING|PARAPET|BALCONY)\s*DETAIL",
                r"(?:CONSTRUCTION|TYPICAL|JOINERY)\s*DETAIL",
                r"(?:DOOR|WINDOW)\s*DETAIL",
                r"DETAIL(?:S)?\s*(?:SHEET|PAGE)?",
            ],
            "sheet_prefixes": ["D-", "DT-", "AD-"],
            "weight": 0.8,
        },
        PageType.NOTES_SPECS: {
            "title_keywords": [
                r"(?:GENERAL|CONSTRUCTION|STRUCTURAL)\s*NOTES",
                r"SPECIFICATIONS?\b",
                r"(?:STANDARD|TYPICAL)\s*(?:NOTES|DETAILS)",
                r"LEGEND\b",
                r"ABBREVIATIONS?\b",
            ],
            "sheet_prefixes": ["N-", "GN-"],
            "weight": 0.7,
        },
        PageType.COVER: {
            "title_keywords": [
                r"(?:COVER|TITLE)\s*(?:SHEET|PAGE)",
                r"DRAWING\s*(?:LIST|INDEX)",
                r"(?:PROJECT|SITE)\s*(?:BRIEF|INFO)",
            ],
            "sheet_prefixes": ["C-", "00-", "T-"],
            "weight": 0.6,
        },
    }

    # Schedule subtypes
    SCHEDULE_SUBTYPES = {
        "door": [r"DOOR\s*SCHEDULE", r"DOOR\s*HARDWARE"],
        "window": [r"WINDOW\s*SCHEDULE", r"GLAZING\s*SCHEDULE"],
        "finish": [r"FINISH\s*SCHEDULE", r"ROOM\s*FINISH", r"FLOOR\s*FINISH"],
        "bbs": [r"(?:BAR\s*BENDING|BBS)\s*SCHEDULE", r"REINFORCEMENT\s*SCHEDULE"],
        "structural": [r"(?:BEAM|COLUMN|FOOTING)\s*SCHEDULE"],
        "area": [r"AREA\s*STATEMENT", r"CARPET\s*AREA"],
    }

    def __init__(self, ambiguity_threshold: float = 0.3):
        """
        Initialize router.

        Args:
            ambiguity_threshold: If top two scores differ by less than this, flag as ambiguous
        """
        self.ambiguity_threshold = ambiguity_threshold

    def route_project(
        self,
        project_index: ProjectIndex,
        output_dir: Optional[Path] = None,
    ) -> RoutingResult:
        """
        Route all pages in a project.

        Args:
            project_index: Indexed project
            output_dir: Optional output directory for manifest

        Returns:
            RoutingResult
        """
        logger.info(f"Routing {len(project_index.pages)} pages")

        pages_by_type: Dict[str, List[PageRouting]] = {pt.value: [] for pt in PageType}
        ambiguous_pages = []

        for page in project_index.pages:
            routing = self._route_page(page)
            pages_by_type[routing.page_type.value].append(routing)

            if routing.warnings:
                ambiguous_pages.append(routing)

        # Count types
        type_counts = {ptype: len(pages) for ptype, pages in pages_by_type.items()}

        # Build summary
        summary = {
            "floor_plans": type_counts.get("floor_plan", 0),
            "schedules": type_counts.get("schedule_table", 0),
            "structural": type_counts.get("structural_plan", 0),
            "sections_elevations": type_counts.get("section_elevation", 0),
            "details": type_counts.get("detail", 0),
            "other": (
                type_counts.get("notes_specs", 0) +
                type_counts.get("cover", 0) +
                type_counts.get("unknown", 0)
            ),
            "ambiguous": len(ambiguous_pages),
        }

        result = RoutingResult(
            project_id=project_index.project_id,
            total_pages=project_index.total_pages,
            pages_by_type=pages_by_type,
            type_counts=type_counts,
            ambiguous_pages=ambiguous_pages,
            routing_summary=summary,
        )

        # Save manifest
        if output_dir:
            manifest_path = output_dir / project_index.project_id / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(manifest_path)
            logger.info(f"Saved routing manifest to: {manifest_path}")

        logger.info(f"Routing complete: {summary}")

        return result

    def _route_page(self, page: PageIndex) -> PageRouting:
        """Route a single page."""
        text_upper = page.extracted_text.upper() if page.extracted_text else ""
        titles = " ".join(page.candidate_titles).upper()
        combined_text = f"{titles} {text_upper}"

        # Score each page type
        scores: Dict[PageType, Tuple[float, List[str]]] = {}

        for page_type, config in self.KEYWORDS.items():
            score, matched = self._score_page_type(
                combined_text, page, config
            )
            scores[page_type] = (score * config["weight"], matched)

        # Visual feature adjustments
        features = page.features

        # Schedule boost for table-like content
        if features.table_likeness > 0.5:
            current = scores[PageType.SCHEDULE_TABLE]
            scores[PageType.SCHEDULE_TABLE] = (current[0] + 0.3, current[1])

        # Floor plan boost for drawing-like content
        if features.estimated_content_type == "drawing" and features.line_density > 20:
            current = scores[PageType.FLOOR_PLAN]
            scores[PageType.FLOOR_PLAN] = (current[0] + 0.1, current[1])

        # Find best match
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        best_type, (best_score, best_matched) = sorted_scores[0]
        second_score = sorted_scores[1][1][0] if len(sorted_scores) > 1 else 0

        # Check for ambiguity
        warnings = []
        if best_score < 0.2:
            best_type = PageType.UNKNOWN
            warnings.append("Low confidence - no clear type match")
        elif abs(best_score - second_score) < self.ambiguity_threshold and second_score > 0.2:
            warnings.append(f"Ambiguous: could be {sorted_scores[1][0].value} ({second_score:.2f})")

        # Determine schedule subtype
        schedule_subtype = None
        if best_type == PageType.SCHEDULE_TABLE:
            schedule_subtype = self._classify_schedule_subtype(combined_text)

        # Calculate confidence
        confidence = min(0.95, best_score)
        if warnings:
            confidence *= 0.8

        return PageRouting(
            file_path=page.file_path,
            page_number=page.page_number,
            page_type=best_type,
            confidence=confidence,
            schedule_subtype=schedule_subtype,
            matched_keywords=best_matched,
            warnings=warnings,
        )

    def _score_page_type(
        self,
        text: str,
        page: PageIndex,
        config: Dict
    ) -> Tuple[float, List[str]]:
        """Score how well a page matches a type."""
        score = 0.0
        matched = []

        # Check title keywords
        for pattern in config.get("title_keywords", []):
            if re.search(pattern, text):
                score += 0.4
                match = re.search(pattern, text)
                if match:
                    matched.append(match.group(0))

        # Check sheet prefixes
        for prefix in config.get("sheet_prefixes", []):
            prefix_pattern = rf"\b{re.escape(prefix.rstrip('-'))}\s*-?\s*\d"
            if re.search(prefix_pattern, text, re.IGNORECASE):
                score += 0.3
                matched.append(f"sheet:{prefix}")

        # Check candidate titles
        for title in page.candidate_titles:
            for pattern in config.get("title_keywords", []):
                if re.search(pattern, title):
                    score += 0.2
                    if title not in matched:
                        matched.append(title)

        return min(1.0, score), matched

    def _classify_schedule_subtype(self, text: str) -> Optional[str]:
        """Classify schedule subtype."""
        for subtype, patterns in self.SCHEDULE_SUBTYPES.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return subtype
        return "other"


def route_project(
    project_index: ProjectIndex,
    output_dir: Optional[Path] = None,
    ambiguity_threshold: float = 0.3,
) -> RoutingResult:
    """
    Convenience function to route a project.

    Args:
        project_index: Indexed project
        output_dir: Output directory
        ambiguity_threshold: Threshold for ambiguity warnings

    Returns:
        RoutingResult
    """
    router = PageRouter(ambiguity_threshold=ambiguity_threshold)
    return router.route_project(project_index, output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 1:
        index_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")

        # Load index
        project_index = ProjectIndex.load(index_path)

        # Route
        result = route_project(project_index, output_dir)

        print(f"\nRouting Result: {result.project_id}")
        print(f"Total pages: {result.total_pages}")
        print("\nBy type:")
        for ptype, count in result.type_counts.items():
            if count > 0:
                print(f"  {ptype}: {count}")
        print(f"\nAmbiguous: {len(result.ambiguous_pages)}")
    else:
        print("Usage: python router.py <index.json> [output_dir]")
