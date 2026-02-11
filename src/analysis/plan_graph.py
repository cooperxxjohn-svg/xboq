"""
Plan Set Graph Builder

Extracts structure from plan set to build a graph of sheets with:
- Sheet classification (discipline, type)
- Detected entities (doors, windows, rooms)
- References between sheets (sections, details, elevations)
- Schedule detection

Uses regex patterns - no OCR required. Works on extracted PDF text.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.analysis_models import (
    PlanSheet, PlanSetGraph,
    Discipline, SheetType,
    EvidenceRef
)


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Sheet number patterns (common formats)
SHEET_NO_PATTERNS = [
    r'\b([A-Z]{1,2})-?(\d{1,3}(?:\.\d{1,2})?)\b',  # A-101, S-2.01, A101
    r'\bSHEET\s*(?:NO\.?|#)?\s*([A-Z]?\d+)\b',      # SHEET NO. 5, SHEET A1
    r'\bDWG\.?\s*(?:NO\.?|#)?\s*([A-Z]?\d+)\b',     # DWG NO. 5
]

# Discipline detection from sheet prefix
DISCIPLINE_PREFIXES = {
    'A': Discipline.ARCHITECTURAL,
    'AR': Discipline.ARCHITECTURAL,
    'ARCH': Discipline.ARCHITECTURAL,
    'S': Discipline.STRUCTURAL,
    'ST': Discipline.STRUCTURAL,
    'STR': Discipline.STRUCTURAL,
    'M': Discipline.MECHANICAL,
    'ME': Discipline.MECHANICAL,
    'MECH': Discipline.MECHANICAL,
    'E': Discipline.ELECTRICAL,
    'EL': Discipline.ELECTRICAL,
    'ELEC': Discipline.ELECTRICAL,
    'P': Discipline.PLUMBING,
    'PL': Discipline.PLUMBING,
    'PLMB': Discipline.PLUMBING,
    'C': Discipline.CIVIL,
    'CV': Discipline.CIVIL,
    'L': Discipline.LANDSCAPE,
    'LS': Discipline.LANDSCAPE,
    'FP': Discipline.FIRE,
    'FA': Discipline.FIRE,
    'ID': Discipline.INTERIOR,
    'INT': Discipline.INTERIOR,
    'G': Discipline.GENERAL,
    'GEN': Discipline.GENERAL,
}

# Sheet type detection keywords
SHEET_TYPE_KEYWORDS = {
    SheetType.FLOOR_PLAN: ['floor plan', 'ground floor', 'first floor', 'second floor',
                            'typical floor', 'basement', 'terrace', 'roof plan',
                            'level', 'storey', 'story'],
    SheetType.SITE_PLAN: ['site plan', 'site layout', 'location plan', 'key plan',
                          'block plan', 'layout plan'],
    SheetType.STRUCTURAL: ['structural', 'framing', 'beam layout', 'column layout',
                           'slab layout', 'foundation layout'],
    SheetType.FOUNDATION: ['foundation', 'footing', 'pile', 'raft', 'plinth'],
    SheetType.SECTION: ['section', 'sectional', 'cross section', 'longitudinal'],
    SheetType.ELEVATION: ['elevation', 'front elevation', 'rear elevation',
                          'side elevation', 'facade'],
    SheetType.DETAIL: ['detail', 'enlarged', 'typical detail', 'construction detail'],
    SheetType.SCHEDULE: ['schedule', 'door schedule', 'window schedule',
                         'finish schedule', 'fixture schedule'],
    SheetType.LEGEND: ['legend', 'symbols', 'abbreviations', 'notes'],
    SheetType.COVER: ['cover', 'title sheet', 'index'],
    SheetType.INDEX: ['index', 'drawing list', 'sheet index'],
    SheetType.NOTES: ['notes', 'general notes', 'specifications'],
    SheetType.MEP: ['mechanical', 'hvac', 'air conditioning'],
    SheetType.ELECTRICAL: ['electrical', 'lighting', 'power', 'panel'],
    SheetType.PLUMBING: ['plumbing', 'drainage', 'water supply', 'sanitary'],
}

# Entity detection patterns
DOOR_TAG_PATTERNS = [
    r'\b(D-?\d{1,3}[A-Z]?)\b',           # D1, D-01, D1A
    r'\b(DR-?\d{1,3})\b',                 # DR1, DR-01
    r'\b(DOOR\s*\d{1,3})\b',              # DOOR 1
]

WINDOW_TAG_PATTERNS = [
    r'\b(W-?\d{1,3}[A-Z]?)\b',            # W1, W-01, W1A
    r'\b(WN-?\d{1,3})\b',                  # WN1, WN-01
    r'\b(WINDOW\s*\d{1,3})\b',             # WINDOW 1
]

ROOM_NAME_PATTERNS = [
    r'\b(BEDROOM|BED\s*ROOM|BR)\s*[-#]?\s*\d*\b',
    r'\b(LIVING\s*ROOM?|LR|LIVING)\b',
    r'\b(KITCHEN|KIT)\b',
    r'\b(BATHROOM|BATH\s*ROOM|TOILET|WC|W\.C\.)\b',
    r'\b(DINING|DINING\s*ROOM)\b',
    r'\b(BALCONY|BLCNY)\b',
    r'\b(CORRIDOR|CORR)\b',
    r'\b(LOBBY|FOYER|ENTRANCE)\b',
    r'\b(STAIR|STAIRCASE|STAIRWELL)\b',
    r'\b(UTILITY|UTILITY\s*ROOM)\b',
    r'\b(STORE|STORAGE)\b',
    r'\b(OFFICE|OFF)\b',
    r'\b(CONFERENCE|MEETING\s*ROOM)\b',
    r'\b(RECEPTION)\b',
]

# Scale detection patterns
SCALE_PATTERNS = [
    r'\b1\s*:\s*(\d+)\b',                  # 1:100
    r'\bSCALE\s*[=:]?\s*1\s*:\s*(\d+)\b',  # SCALE = 1:100
    r'\bSCALE\s*[=:]?\s*1/(\d+)\b',        # SCALE = 1/100
    r"\b(\d+)'\s*=\s*1[\"']",              # 1/4" = 1'-0"
]

# Reference patterns (callouts to other sheets)
SECTION_REF_PATTERNS = [
    r'\bSECTION\s+([A-Z](?:-[A-Z])?)\b',      # SECTION A-A
    r'\bSEC\.\s*([A-Z](?:-[A-Z])?)\b',        # SEC. A-A
    r'\b([A-Z])-([A-Z])\b',                    # A-A (in context)
]

DETAIL_REF_PATTERNS = [
    r'\bDETAIL\s+(\d+)\b',                     # DETAIL 1
    r'\b(\d+)/([A-Z]-?\d+)\b',                 # 1/A-5.01 (detail number/sheet)
]


# =============================================================================
# PLAN GRAPH BUILDER
# =============================================================================

class PlanGraphBuilder:
    """
    Builds a plan set graph from extracted PDF text.

    Takes per-page text extracts and produces structured PlanSetGraph.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.sheets: List[PlanSheet] = []

    def build(self, page_texts: List[str]) -> PlanSetGraph:
        """
        Build plan graph from page texts.

        Args:
            page_texts: List of extracted text per page (0-indexed)

        Returns:
            PlanSetGraph with all sheets and aggregates
        """
        self.sheets = []

        for page_idx, text in enumerate(page_texts):
            sheet = self._process_page(page_idx, text)
            self.sheets.append(sheet)

        return self._build_graph()

    def _process_page(self, page_idx: int, text: str) -> PlanSheet:
        """Process a single page into a PlanSheet."""
        text_upper = text.upper()
        text_lower = text.lower()

        # Extract sheet number
        sheet_no = self._extract_sheet_no(text_upper)

        # Classify discipline
        discipline = self._classify_discipline(sheet_no, text_upper)

        # Classify sheet type
        sheet_type = self._classify_sheet_type(text_lower)

        # Extract title (first non-empty line that looks like a title)
        title = self._extract_title(text)

        # Detect entities
        detected = {}
        detected['door_tags'] = self._extract_door_tags(text_upper)
        detected['window_tags'] = self._extract_window_tags(text_upper)
        detected['room_names'] = self._extract_room_names(text_upper)
        detected['has_scale'] = self._detect_scale(text)
        detected['has_legend'] = self._detect_legend(text_lower)

        # Extract references
        references = {}
        references['sections'] = self._extract_section_refs(text_upper)
        references['details'] = self._extract_detail_refs(text)

        # Calculate classification confidence
        confidence = self._calculate_confidence(sheet_no, sheet_type, detected)

        return PlanSheet(
            page_index=page_idx,
            sheet_no=sheet_no,
            title=title,
            discipline=discipline,
            sheet_type=sheet_type,
            references=references,
            detected=detected,
            text_preview=text[:500],
            classification_confidence=confidence,
        )

    def _extract_sheet_no(self, text: str) -> Optional[str]:
        """Extract sheet number from text."""
        for pattern in SHEET_NO_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return f"{groups[0]}-{groups[1]}"
                return groups[0]
        return None

    def _classify_discipline(self, sheet_no: Optional[str], text: str) -> Discipline:
        """Classify discipline from sheet number prefix."""
        if sheet_no:
            # Extract prefix (letters before number)
            prefix_match = re.match(r'^([A-Z]+)', sheet_no)
            if prefix_match:
                prefix = prefix_match.group(1)
                if prefix in DISCIPLINE_PREFIXES:
                    return DISCIPLINE_PREFIXES[prefix]

        # Fallback: check text for discipline keywords
        if any(kw in text for kw in ['STRUCTURAL', 'BEAM', 'COLUMN', 'SLAB']):
            return Discipline.STRUCTURAL
        if any(kw in text for kw in ['ELECTRICAL', 'LIGHTING', 'POWER']):
            return Discipline.ELECTRICAL
        if any(kw in text for kw in ['PLUMBING', 'DRAINAGE', 'SANITARY']):
            return Discipline.PLUMBING
        if any(kw in text for kw in ['MECHANICAL', 'HVAC', 'AIR CONDITIONING']):
            return Discipline.MECHANICAL

        return Discipline.UNKNOWN

    def _classify_sheet_type(self, text: str) -> SheetType:
        """Classify sheet type from keywords."""
        best_match = (SheetType.UNKNOWN, 0)

        for sheet_type, keywords in SHEET_TYPE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > best_match[1]:
                best_match = (sheet_type, matches)

        return best_match[0]

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract sheet title (first meaningful line)."""
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            # Skip short lines or lines that are just numbers
            if len(line) > 5 and not line.isdigit():
                # Skip lines that look like coordinates or dimensions
                if not re.match(r'^[\d\.\,\-\s]+$', line):
                    return line[:100]  # Truncate long titles
        return None

    def _extract_door_tags(self, text: str) -> List[str]:
        """Extract door tags from text."""
        tags = set()
        for pattern in DOOR_TAG_PATTERNS:
            matches = re.findall(pattern, text)
            tags.update(matches)
        return sorted(list(tags))

    def _extract_window_tags(self, text: str) -> List[str]:
        """Extract window tags from text."""
        tags = set()
        for pattern in WINDOW_TAG_PATTERNS:
            matches = re.findall(pattern, text)
            tags.update(matches)
        return sorted(list(tags))

    def _extract_room_names(self, text: str) -> List[str]:
        """Extract room names from text."""
        names = set()
        for pattern in ROOM_NAME_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    names.add(match[0])
                else:
                    names.add(match)
        return sorted(list(names))

    def _detect_scale(self, text: str) -> bool:
        """Check if scale is present."""
        for pattern in SCALE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_legend(self, text: str) -> bool:
        """Check if legend/symbols section present."""
        return 'legend' in text or 'symbols' in text or 'abbreviation' in text

    def _extract_section_refs(self, text: str) -> List[str]:
        """Extract section references."""
        refs = set()
        for pattern in SECTION_REF_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    refs.add('-'.join(match))
                else:
                    refs.add(match)
        return sorted(list(refs))

    def _extract_detail_refs(self, text: str) -> List[str]:
        """Extract detail references."""
        refs = set()
        for pattern in DETAIL_REF_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    refs.add('/'.join(match))
                else:
                    refs.add(match)
        return sorted(list(refs))

    def _calculate_confidence(
        self,
        sheet_no: Optional[str],
        sheet_type: SheetType,
        detected: Dict[str, Any]
    ) -> float:
        """Calculate confidence in classification."""
        confidence = 0.3  # Base

        if sheet_no:
            confidence += 0.2
        if sheet_type != SheetType.UNKNOWN:
            confidence += 0.2
        if detected.get('has_scale'):
            confidence += 0.1
        if detected.get('door_tags') or detected.get('window_tags'):
            confidence += 0.1
        if detected.get('room_names'):
            confidence += 0.1

        return min(confidence, 1.0)

    def _build_graph(self) -> PlanSetGraph:
        """Build the complete graph with aggregates."""
        graph = PlanSetGraph(
            project_id=self.project_id,
            sheets=self.sheets,
            total_pages=len(self.sheets),
        )

        # Aggregate disciplines
        disciplines = set()
        for sheet in self.sheets:
            if sheet.discipline != Discipline.UNKNOWN:
                disciplines.add(sheet.discipline.value)
        graph.disciplines_found = sorted(list(disciplines))

        # Aggregate sheet types
        type_counts: Dict[str, int] = defaultdict(int)
        for sheet in self.sheets:
            type_counts[sheet.sheet_type.value] += 1
        graph.sheet_types_found = dict(type_counts)

        # Aggregate entities
        all_doors: Set[str] = set()
        all_windows: Set[str] = set()
        all_rooms: Set[str] = set()
        pages_with_scale = 0
        pages_without_scale = 0

        for sheet in self.sheets:
            all_doors.update(sheet.detected.get('door_tags', []))
            all_windows.update(sheet.detected.get('window_tags', []))
            all_rooms.update(sheet.detected.get('room_names', []))

            if sheet.detected.get('has_scale'):
                pages_with_scale += 1
            else:
                pages_without_scale += 1

        graph.all_door_tags = sorted(list(all_doors))
        graph.all_window_tags = sorted(list(all_windows))
        graph.all_room_names = sorted(list(all_rooms))
        graph.pages_with_scale = pages_with_scale
        graph.pages_without_scale = pages_without_scale

        # Detect schedules
        graph.has_door_schedule = self._has_schedule_type('door')
        graph.has_window_schedule = self._has_schedule_type('window')
        graph.has_finish_schedule = self._has_schedule_type('finish')
        graph.has_legend = any(s.detected.get('has_legend') for s in self.sheets)

        return graph

    def _has_schedule_type(self, schedule_type: str) -> bool:
        """Check if a specific schedule type exists."""
        for sheet in self.sheets:
            if sheet.sheet_type == SheetType.SCHEDULE:
                if schedule_type in sheet.text_preview.lower():
                    return True
            # Also check keywords in any sheet
            if f'{schedule_type} schedule' in sheet.text_preview.lower():
                return True
        return False


# =============================================================================
# ENTRY POINT
# =============================================================================

def build_plan_graph(project_id: str, page_texts: List[str]) -> PlanSetGraph:
    """
    Build plan graph from extracted page texts.

    Args:
        project_id: Project identifier
        page_texts: List of text per page (0-indexed)

    Returns:
        PlanSetGraph with all sheets and aggregates
    """
    builder = PlanGraphBuilder(project_id)
    return builder.build(page_texts)


def load_plan_graph_from_dir(project_dir: Path) -> Optional[PlanSetGraph]:
    """
    Load plan graph from saved JSON if exists.

    Args:
        project_dir: Path to project output directory

    Returns:
        PlanSetGraph or None
    """
    graph_path = project_dir / "plan_graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            data = json.load(f)
        # Reconstruct graph
        sheets = [PlanSheet(**s) for s in data.get('sheets', [])]
        graph = PlanSetGraph(
            project_id=data.get('project_id', ''),
            sheets=sheets,
            total_pages=data.get('total_pages', 0),
            disciplines_found=data.get('disciplines_found', []),
            sheet_types_found=data.get('sheet_types_found', {}),
            all_door_tags=data.get('all_door_tags', []),
            all_window_tags=data.get('all_window_tags', []),
            all_room_names=data.get('all_room_names', []),
            has_door_schedule=data.get('has_door_schedule', False),
            has_window_schedule=data.get('has_window_schedule', False),
            has_finish_schedule=data.get('has_finish_schedule', False),
            has_legend=data.get('has_legend', False),
            pages_with_scale=data.get('pages_with_scale', 0),
            pages_without_scale=data.get('pages_without_scale', 0),
        )
        return graph
    return None


def save_plan_graph(graph: PlanSetGraph, project_dir: Path) -> Path:
    """
    Save plan graph to JSON.

    Args:
        graph: PlanSetGraph to save
        project_dir: Path to project output directory

    Returns:
        Path to saved file
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    graph_path = project_dir / "plan_graph.json"
    with open(graph_path, 'w') as f:
        json.dump(graph.to_dict(), f, indent=2)
    return graph_path


# =============================================================================
# TEST / DEBUG
# =============================================================================

if __name__ == "__main__":
    # Quick test with sample text
    sample_texts = [
        """
        SHEET A-101
        GROUND FLOOR PLAN
        SCALE 1:100

        LIVING ROOM
        KITCHEN
        BEDROOM 1
        BEDROOM 2
        BATHROOM

        D1 D2 D3 D4
        W1 W2 W3

        SEE SECTION A-A
        """,
        """
        SHEET S-01
        STRUCTURAL LAYOUT - GROUND FLOOR

        COLUMN C1, C2, C3
        BEAM B1, B2

        FOOTING DETAILS
        """,
        """
        DOOR SCHEDULE

        DOOR NO. | SIZE | TYPE | REMARKS
        D1 | 900x2100 | FLUSH | MAIN ENTRY
        D2 | 800x2100 | FLUSH | INTERNAL
        """,
    ]

    graph = build_plan_graph("test_project", sample_texts)
    print(json.dumps(graph.to_dict(), indent=2))
