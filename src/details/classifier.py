"""
Detail Page Classifier - Identify and classify detail drawings.

Classifies detail pages into categories:
- Waterproofing details (toilet, terrace, tank, etc.)
- Architectural details (parapet, sill, door frame, etc.)
- Structural details (beam-column, slab edge, etc.)
- MEP details (plumbing riser, electrical panel, etc.)

India-specific patterns for detail recognition.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class DetailType(Enum):
    """Type of detail drawing."""
    # Waterproofing
    TOILET_WATERPROOFING = "toilet_waterproofing"
    TERRACE_WATERPROOFING = "terrace_waterproofing"
    BALCONY_WATERPROOFING = "balcony_waterproofing"
    WATER_TANK_WATERPROOFING = "water_tank_waterproofing"
    BASEMENT_WATERPROOFING = "basement_waterproofing"
    PLINTH_PROTECTION = "plinth_protection"

    # Architectural
    PARAPET = "parapet"
    WINDOW_SILL = "window_sill"
    DOOR_FRAME = "door_frame"
    STAIR = "stair"
    RAILING = "railing"
    EXPANSION_JOINT = "expansion_joint"
    KITCHEN_PLATFORM = "kitchen_platform"
    DADO = "dado"

    # Structural
    BEAM_COLUMN_JUNCTION = "beam_column_junction"
    SLAB_EDGE = "slab_edge"
    FOUNDATION = "foundation"
    SHEAR_WALL = "shear_wall"
    STAIRCASE_STRUCTURAL = "staircase_structural"

    # MEP
    PLUMBING_RISER = "plumbing_riser"
    SANITARY_CONNECTION = "sanitary_connection"
    ELECTRICAL_PANEL = "electrical_panel"
    CONDUIT_LAYOUT = "conduit_layout"
    AC_DRAIN = "ac_drain"
    RAINWATER_PIPE = "rainwater_pipe"
    FIRE_HYDRANT = "fire_hydrant"

    # Finishes
    FLOOR_PATTERN = "floor_pattern"
    CEILING_DETAIL = "ceiling_detail"
    WALL_CLADDING = "wall_cladding"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class DetailClassification:
    """Classification of a detail page."""
    sheet_id: str
    detail_type: DetailType
    title: str = ""
    sub_details: List[str] = field(default_factory=list)
    confidence: float = 0.8
    keywords_matched: List[str] = field(default_factory=list)
    page_type: str = ""
    source_page: int = 0

    def to_dict(self) -> dict:
        return {
            "sheet_id": self.sheet_id,
            "detail_type": self.detail_type.value,
            "title": self.title,
            "sub_details": self.sub_details,
            "confidence": self.confidence,
            "keywords_matched": self.keywords_matched,
            "page_type": self.page_type,
            "source_page": self.source_page,
        }


class DetailClassifier:
    """Classify detail pages."""

    # Detail type patterns (India-specific)
    DETAIL_PATTERNS = {
        # Waterproofing details
        DetailType.TOILET_WATERPROOFING: [
            r'toilet\s+waterproof',
            r'wc\s+waterproof',
            r'bathroom\s+waterproof',
            r'wet\s+area\s+waterproof',
            r'toilet\s+detail',
            r'bathroom\s+detail',
            r'wc\s+floor\s+detail',
            r'waterproof.*toilet',
            r'toilet.*treatment',
        ],
        DetailType.TERRACE_WATERPROOFING: [
            r'terrace\s+waterproof',
            r'roof\s+waterproof',
            r'terrace\s+treatment',
            r'flat\s+roof\s+detail',
            r'terrace\s+detail',
            r'roof\s+detail',
            r'weathering\s+course',
            r'heat\s+insulation',
            r'terrace\s+slope',
        ],
        DetailType.BALCONY_WATERPROOFING: [
            r'balcony\s+waterproof',
            r'balcony\s+detail',
            r'balcony\s+floor',
            r'balcony\s+treatment',
            r'cantilever\s+slab\s+detail',
        ],
        DetailType.WATER_TANK_WATERPROOFING: [
            r'water\s+tank\s+waterproof',
            r'overhead\s+tank\s+detail',
            r'oht\s+detail',
            r'sump\s+waterproof',
            r'underground\s+tank',
            r'ugt\s+detail',
            r'water\s+tank\s+detail',
        ],
        DetailType.BASEMENT_WATERPROOFING: [
            r'basement\s+waterproof',
            r'retaining\s+wall\s+waterproof',
            r'basement\s+detail',
            r'foundation\s+waterproof',
        ],
        DetailType.PLINTH_PROTECTION: [
            r'plinth\s+protection',
            r'plinth\s+detail',
            r'apron\s+detail',
            r'dpc\s+detail',
            r'damp\s+proof\s+course',
            r'plinth\s+beam',
        ],

        # Architectural details
        DetailType.PARAPET: [
            r'parapet\s+detail',
            r'parapet\s+wall',
            r'coping\s+detail',
            r'parapet\s+coping',
            r'boundary\s+wall.*detail',
        ],
        DetailType.WINDOW_SILL: [
            r'window\s+sill',
            r'sill\s+detail',
            r'window\s+frame\s+detail',
            r'window\s+section',
            r'lintel\s+sill',
        ],
        DetailType.DOOR_FRAME: [
            r'door\s+frame\s+detail',
            r'door\s+section',
            r'chaukhat\s+detail',
            r'door\s+jamb',
            r'threshold\s+detail',
        ],
        DetailType.STAIR: [
            r'stair\s+detail',
            r'staircase\s+detail',
            r'step\s+detail',
            r'tread\s+riser',
            r'stair\s+section',
            r'riser\s+detail',
            r'nosing\s+detail',
        ],
        DetailType.RAILING: [
            r'railing\s+detail',
            r'handrail\s+detail',
            r'balustrade',
            r'ms\s+railing',
            r'ss\s+railing',
            r'glass\s+railing',
        ],
        DetailType.EXPANSION_JOINT: [
            r'expansion\s+joint',
            r'movement\s+joint',
            r'control\s+joint',
        ],
        DetailType.KITCHEN_PLATFORM: [
            r'kitchen\s+platform',
            r'kitchen\s+counter',
            r'kitchen\s+detail',
            r'granite\s+platform',
            r'sink\s+detail',
        ],
        DetailType.DADO: [
            r'dado\s+detail',
            r'wall\s+dado',
            r'tile\s+dado',
        ],

        # Structural details
        DetailType.BEAM_COLUMN_JUNCTION: [
            r'beam\s+column\s+junction',
            r'beam\s*-\s*column',
            r'junction\s+detail',
            r'reinforcement\s+detail',
        ],
        DetailType.SLAB_EDGE: [
            r'slab\s+edge',
            r'chajja\s+detail',
            r'sunshade\s+detail',
            r'canopy\s+detail',
            r'projection\s+detail',
        ],
        DetailType.FOUNDATION: [
            r'foundation\s+detail',
            r'footing\s+detail',
            r'pile\s+cap',
            r'raft\s+detail',
            r'plinth\s+beam\s+detail',
        ],
        DetailType.STAIRCASE_STRUCTURAL: [
            r'stair.*structural',
            r'waist\s+slab',
            r'flight\s+detail',
            r'landing\s+detail',
        ],

        # MEP details
        DetailType.PLUMBING_RISER: [
            r'plumbing\s+riser',
            r'shaft\s+detail',
            r'pipe\s+shaft',
            r'duct\s+detail',
            r'cpvc\s+layout',
        ],
        DetailType.SANITARY_CONNECTION: [
            r'sanitary\s+connection',
            r'wc\s+connection',
            r'floor\s+trap',
            r'nahani\s+trap',
            r'gully\s+trap',
            r'p-trap',
            r's-trap',
            r'swr\s+detail',
        ],
        DetailType.ELECTRICAL_PANEL: [
            r'db\s+detail',
            r'panel\s+detail',
            r'mcb\s+layout',
            r'electrical\s+panel',
            r'distribution\s+board',
        ],
        DetailType.CONDUIT_LAYOUT: [
            r'conduit\s+layout',
            r'conduit\s+detail',
            r'wiring\s+layout',
            r'cable\s+tray',
        ],
        DetailType.AC_DRAIN: [
            r'ac\s+drain',
            r'condensate\s+drain',
            r'split\s+ac\s+detail',
            r'ac\s+outdoor\s+unit',
        ],
        DetailType.RAINWATER_PIPE: [
            r'rainwater\s+pipe',
            r'rwp\s+detail',
            r'rain\s+water\s+down',
            r'downspout',
            r'gutter\s+detail',
        ],
        DetailType.FIRE_HYDRANT: [
            r'fire\s+hydrant',
            r'hose\s+reel',
            r'fire\s+fighting',
            r'sprinkler\s+detail',
        ],

        # Finishes
        DetailType.FLOOR_PATTERN: [
            r'floor\s+pattern',
            r'flooring\s+layout',
            r'tile\s+layout',
            r'floor\s+finish\s+detail',
        ],
        DetailType.CEILING_DETAIL: [
            r'ceiling\s+detail',
            r'false\s+ceiling',
            r'gypsum\s+ceiling',
            r'ceiling\s+section',
        ],
        DetailType.WALL_CLADDING: [
            r'wall\s+cladding',
            r'stone\s+cladding',
            r'facade\s+detail',
            r'elevation\s+detail',
        ],
    }

    # Page type patterns that indicate detail sheets
    DETAIL_PAGE_PATTERNS = [
        r'detail',
        r'section',
        r'typical',
        r'standard',
        r'enlarged',
        r'blow.*up',
    ]

    def __init__(self):
        # Compile all patterns
        self.compiled_patterns = {}
        for detail_type, patterns in self.DETAIL_PATTERNS.items():
            self.compiled_patterns[detail_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self.page_pattern = re.compile(
            '|'.join(self.DETAIL_PAGE_PATTERNS),
            re.IGNORECASE
        )

    def classify_all(self, extraction_results: List[Dict]) -> List[DetailClassification]:
        """Classify all pages and return detail pages."""
        classifications = []

        for result in extraction_results:
            classification = self.classify_page(result)
            if classification and classification.detail_type != DetailType.UNKNOWN:
                classifications.append(classification)

        return classifications

    def classify_page(self, result: Dict) -> Optional[DetailClassification]:
        """Classify a single page."""
        from pathlib import Path

        file_name = Path(result.get("file_path", "")).stem
        page_num = result.get("page_number", 0) + 1
        sheet_id = f"{file_name}_p{page_num}"
        page_type = result.get("page_type", "")

        # Collect all text from the page
        all_text = self._collect_page_text(result)

        # Check if this is a detail page
        is_detail_page = self._is_detail_page(result, all_text)

        if not is_detail_page:
            return None

        # Classify the detail type
        detail_type, keywords, confidence = self._classify_detail_type(all_text)

        # Extract title
        title = self._extract_title(result, all_text)

        # Extract sub-details
        sub_details = self._extract_sub_details(all_text)

        return DetailClassification(
            sheet_id=sheet_id,
            detail_type=detail_type,
            title=title,
            sub_details=sub_details,
            confidence=confidence,
            keywords_matched=keywords,
            page_type=page_type,
            source_page=page_num,
        )

    def _collect_page_text(self, result: Dict) -> str:
        """Collect all text from a page."""
        texts = []

        # Title block
        title_block = result.get("title_block", {})
        if title_block:
            texts.append(title_block.get("drawing_title", ""))
            texts.append(title_block.get("sheet_name", ""))
            texts.append(title_block.get("notes", ""))

        # Text items
        for item in result.get("text_items", []):
            texts.append(item.get("text", ""))

        # Notes
        for note in result.get("notes", []):
            if isinstance(note, dict):
                texts.append(note.get("text", ""))
            else:
                texts.append(str(note))

        # Annotations
        for ann in result.get("annotations", []):
            if isinstance(ann, dict):
                texts.append(ann.get("text", ""))
            else:
                texts.append(str(ann))

        return " ".join(filter(None, texts))

    def _is_detail_page(self, result: Dict, all_text: str) -> bool:
        """Check if page is a detail page."""
        # Check page type
        page_type = result.get("page_type", "").lower()
        if "detail" in page_type:
            return True

        # Check title block
        title_block = result.get("title_block", {})
        drawing_title = title_block.get("drawing_title", "").lower()
        if self.page_pattern.search(drawing_title):
            return True

        sheet_name = title_block.get("sheet_name", "").lower()
        if self.page_pattern.search(sheet_name):
            return True

        # Check text for detail indicators
        if self.page_pattern.search(all_text):
            # Also check for specific detail type patterns
            for detail_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(all_text):
                        return True

        return False

    def _classify_detail_type(self, text: str) -> tuple:
        """Classify the type of detail."""
        best_type = DetailType.UNKNOWN
        best_score = 0
        best_keywords = []

        for detail_type, patterns in self.compiled_patterns.items():
            keywords = []
            score = 0

            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    keywords.extend(matches)
                    score += len(matches)

            if score > best_score:
                best_score = score
                best_type = detail_type
                best_keywords = keywords

        # Calculate confidence based on match strength
        if best_score == 0:
            confidence = 0.0
        elif best_score == 1:
            confidence = 0.6
        elif best_score == 2:
            confidence = 0.75
        else:
            confidence = min(0.95, 0.75 + 0.05 * best_score)

        return best_type, best_keywords[:5], confidence

    def _extract_title(self, result: Dict, all_text: str) -> str:
        """Extract detail title."""
        # Try title block first
        title_block = result.get("title_block", {})
        title = title_block.get("drawing_title", "")

        if title:
            return title

        # Try first significant text item
        for item in result.get("text_items", [])[:10]:
            text = item.get("text", "").strip()
            if len(text) > 10 and len(text) < 100:
                if "detail" in text.lower() or "section" in text.lower():
                    return text

        return ""

    def _extract_sub_details(self, text: str) -> List[str]:
        """Extract list of sub-details mentioned."""
        sub_details = []

        # Pattern for numbered/lettered details
        patterns = [
            r'(?:detail|section)\s*[-:]?\s*(\d+)',
            r'(?:detail|section)\s*[-:]?\s*([a-z])',
            r'(\d+)\s*[-–]\s*([^,\n]{5,40})',
            r'([a-z])\s*[-–]\s*([^,\n]{5,40})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    sub_details.append(" - ".join(match))
                else:
                    sub_details.append(match)

        return list(set(sub_details))[:10]  # Limit to 10

    def get_detail_types_for_package(self, package: str) -> List[DetailType]:
        """Get relevant detail types for a package."""
        package_lower = package.lower()

        mapping = {
            "waterproofing": [
                DetailType.TOILET_WATERPROOFING,
                DetailType.TERRACE_WATERPROOFING,
                DetailType.BALCONY_WATERPROOFING,
                DetailType.WATER_TANK_WATERPROOFING,
                DetailType.BASEMENT_WATERPROOFING,
                DetailType.PLINTH_PROTECTION,
            ],
            "finishes": [
                DetailType.FLOOR_PATTERN,
                DetailType.CEILING_DETAIL,
                DetailType.WALL_CLADDING,
                DetailType.DADO,
                DetailType.KITCHEN_PLATFORM,
            ],
            "doors_windows": [
                DetailType.DOOR_FRAME,
                DetailType.WINDOW_SILL,
            ],
            "plumbing": [
                DetailType.PLUMBING_RISER,
                DetailType.SANITARY_CONNECTION,
                DetailType.RAINWATER_PIPE,
            ],
            "electrical": [
                DetailType.ELECTRICAL_PANEL,
                DetailType.CONDUIT_LAYOUT,
            ],
            "rcc": [
                DetailType.BEAM_COLUMN_JUNCTION,
                DetailType.SLAB_EDGE,
                DetailType.FOUNDATION,
                DetailType.STAIRCASE_STRUCTURAL,
            ],
            "external": [
                DetailType.PARAPET,
                DetailType.PLINTH_PROTECTION,
                DetailType.EXPANSION_JOINT,
            ],
        }

        for key, types in mapping.items():
            if key in package_lower:
                return types

        return []
