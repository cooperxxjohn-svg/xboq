"""
Schedule/Table Extractor Module
Extracts structural schedules (column, beam, footing) from drawings.
Parses tabular data for element sizes, grades, and rebar details.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScheduleCell:
    """A cell in a schedule table."""
    row: int
    col: int
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 0.5


@dataclass
class ScheduleRow:
    """A row in a schedule table."""
    row_num: int
    cells: List[ScheduleCell]
    element_label: Optional[str] = None  # e.g., "C1", "B1"
    parsed_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedSchedule:
    """Extracted schedule data."""
    schedule_type: str  # "column", "beam", "footing", "slab", "unknown"
    title: str
    headers: List[str]
    rows: List[ScheduleRow]
    bbox: Tuple[int, int, int, int]  # Location in drawing
    confidence: float = 0.5
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'type': self.schedule_type,
            'title': self.title,
            'headers': self.headers,
            'data': [
                {
                    'label': row.element_label,
                    **row.parsed_data
                }
                for row in self.rows
            ],
            'confidence': self.confidence
        }


@dataclass
class ScheduleExtractionResult:
    """Result of schedule extraction."""
    schedules: List[ExtractedSchedule]
    column_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # label -> (w, d)
    beam_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # label -> (w, d)
    footing_sizes: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)  # label -> (l, w, d)
    slab_thicknesses: Dict[str, int] = field(default_factory=dict)  # label -> thickness
    concrete_grades: Dict[str, str] = field(default_factory=dict)  # element -> grade
    steel_grades: Dict[str, str] = field(default_factory=dict)  # element -> grade
    warnings: List[str] = field(default_factory=list)


class ScheduleExtractor:
    """
    Extracts and parses structural schedules from drawings.
    Supports column schedules, beam schedules, footing schedules.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize extractor with configuration."""
        self.config = self._load_config(config_path)
        self.keywords = self._load_keywords()

    def _load_config(self, path: Optional[Path]) -> Dict:
        """Load configuration."""
        default_path = Path(__file__).parent.parent.parent / "rules" / "assumptions.yaml"
        try:
            with open(path or default_path) as f:
                data = yaml.safe_load(f)
            return data.get('schedule_extraction', {})
        except Exception:
            return {}

    def _load_keywords(self) -> Dict:
        """Load keywords from structural_keywords.yaml."""
        keywords_path = Path(__file__).parent.parent.parent / "rules" / "structural_keywords.yaml"
        try:
            with open(keywords_path) as f:
                data = yaml.safe_load(f)
            return data.get('schedule_keywords', {})
        except Exception:
            return {
                'column_schedule': ['column schedule', 'column sizing', 'col. schedule'],
                'beam_schedule': ['beam schedule', 'beam sizing', 'beam details'],
                'footing_schedule': ['footing schedule', 'foundation schedule']
            }

    def extract(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None,
        page_type: str = "unknown"
    ) -> ScheduleExtractionResult:
        """
        Extract schedules from drawing.

        Args:
            image: Drawing image
            vector_texts: Text blocks from PDF
            page_type: Type of page for context

        Returns:
            ScheduleExtractionResult
        """
        logger.info("Extracting schedules...")

        schedules = []

        # First, try to find schedule regions using vector text
        if vector_texts:
            schedule_regions = self._find_schedule_regions(vector_texts, image.shape)
            for region in schedule_regions:
                schedule = self._extract_schedule_from_region(
                    image, region, vector_texts
                )
                if schedule:
                    schedules.append(schedule)

        # If no schedules found with vector, try OCR-based detection
        if not schedules:
            schedules = self._detect_schedules_ocr(image)

        # Parse all schedules and extract sizes
        result = self._build_result(schedules)

        logger.info(f"Extracted {len(schedules)} schedules, "
                   f"{len(result.column_sizes)} column sizes, "
                   f"{len(result.beam_sizes)} beam sizes")

        return result

    def _find_schedule_regions(
        self,
        vector_texts: List[Any],
        image_shape: Tuple[int, ...]
    ) -> List[Dict]:
        """Find regions containing schedule tables using text hints."""
        regions = []

        # Look for schedule title keywords
        schedule_titles = (
            self.keywords.get('column_schedule', []) +
            self.keywords.get('beam_schedule', []) +
            self.keywords.get('footing_schedule', [])
        )

        for vt in vector_texts:
            text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
            bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))
            text_lower = text.lower()

            for title in schedule_titles:
                if title in text_lower:
                    # Found schedule title - expand region below it
                    x, y = bbox[0], bbox[1]
                    h, w = image_shape[:2]

                    # Estimate schedule extends below title
                    region = {
                        'title': text,
                        'bbox': (
                            max(0, int(x - 50)),
                            int(y),
                            min(w, int(x + 800)),  # Schedules are typically wide
                            min(h, int(y + 600))   # Extend down
                        ),
                        'type': self._classify_schedule_type(text_lower)
                    }
                    regions.append(region)
                    break

        return regions

    def _classify_schedule_type(self, text: str) -> str:
        """Classify schedule type from title text."""
        text_lower = text.lower()

        if any(k in text_lower for k in ['column', 'col.']):
            return 'column'
        elif any(k in text_lower for k in ['beam', 'bm']):
            return 'beam'
        elif any(k in text_lower for k in ['footing', 'foundation', 'ftg']):
            return 'footing'
        elif any(k in text_lower for k in ['slab']):
            return 'slab'
        else:
            return 'unknown'

    def _extract_schedule_from_region(
        self,
        image: np.ndarray,
        region: Dict,
        vector_texts: List[Any]
    ) -> Optional[ExtractedSchedule]:
        """Extract schedule data from a specific region."""

        x1, y1, x2, y2 = region['bbox']

        # Filter texts within region
        region_texts = []
        for vt in vector_texts:
            bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))
            text = vt.text if hasattr(vt, 'text') else vt.get('text', '')

            # Check if text center is in region
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                region_texts.append({
                    'text': text,
                    'bbox': bbox,
                    'cy': cy,
                    'cx': cx
                })

        if not region_texts:
            return None

        # Sort by y position (rows) then x position (columns)
        region_texts.sort(key=lambda t: (t['cy'], t['cx']))

        # Group into rows (texts with similar y)
        rows = self._group_into_rows(region_texts)

        if len(rows) < 2:  # Need at least header + 1 data row
            return None

        # First row is likely headers
        headers = [t['text'] for t in rows[0]]

        # Parse remaining rows
        parsed_rows = []
        for i, row_texts in enumerate(rows[1:]):
            cells = [
                ScheduleCell(
                    row=i,
                    col=j,
                    text=t['text'],
                    bbox=t['bbox']
                )
                for j, t in enumerate(row_texts)
            ]

            sched_row = ScheduleRow(
                row_num=i,
                cells=cells
            )

            # Try to parse the row
            sched_row = self._parse_schedule_row(sched_row, headers, region['type'])
            parsed_rows.append(sched_row)

        return ExtractedSchedule(
            schedule_type=region['type'],
            title=region['title'],
            headers=headers,
            rows=parsed_rows,
            bbox=region['bbox'],
            confidence=0.7
        )

    def _group_into_rows(self, texts: List[Dict], tolerance: float = 15) -> List[List[Dict]]:
        """Group texts into rows based on y-position."""
        if not texts:
            return []

        rows = []
        current_row = [texts[0]]
        current_y = texts[0]['cy']

        for t in texts[1:]:
            if abs(t['cy'] - current_y) <= tolerance:
                current_row.append(t)
            else:
                # Sort current row by x
                current_row.sort(key=lambda x: x['cx'])
                rows.append(current_row)
                current_row = [t]
                current_y = t['cy']

        if current_row:
            current_row.sort(key=lambda x: x['cx'])
            rows.append(current_row)

        return rows

    def _parse_schedule_row(
        self,
        row: ScheduleRow,
        headers: List[str],
        schedule_type: str
    ) -> ScheduleRow:
        """Parse a schedule row to extract structured data."""

        # Find element label (first cell often)
        for cell in row.cells:
            text = cell.text.strip().upper()
            if schedule_type == 'column' and re.match(r'^C\d+$', text):
                row.element_label = text
            elif schedule_type == 'beam' and re.match(r'^B\d+$|^BM\d+$', text):
                row.element_label = text
            elif schedule_type == 'footing' and re.match(r'^[ICF]?F\d+$', text):
                row.element_label = text

        # Parse sizes from cells
        for i, cell in enumerate(row.cells):
            text = cell.text.strip()

            # Size pattern: 230x450, 230 x 450, etc.
            size_match = re.search(r'(\d{3,4})\s*[xX×]\s*(\d{3,4})', text)
            if size_match:
                w, d = int(size_match.group(1)), int(size_match.group(2))
                row.parsed_data['size_mm'] = (w, d)

            # 3D size for footings: 1500x1500x450
            size_3d_match = re.search(
                r'(\d{3,4})\s*[xX×]\s*(\d{3,4})\s*[xX×]\s*(\d{3,4})',
                text
            )
            if size_3d_match:
                l, w, d = (
                    int(size_3d_match.group(1)),
                    int(size_3d_match.group(2)),
                    int(size_3d_match.group(3))
                )
                row.parsed_data['size_mm'] = (l, w, d)

            # Concrete grade: M20, M25, M30
            grade_match = re.search(r'M\s*(\d{2})', text)
            if grade_match:
                row.parsed_data['concrete_grade'] = f"M{grade_match.group(1)}"

            # Rebar: Y12@150, 4Y16, 8-Y12, etc.
            rebar_match = re.search(
                r'(\d+)\s*[-]?\s*[YT]\s*(\d{2})\s*@?\s*(\d{3})?',
                text
            )
            if rebar_match:
                row.parsed_data['rebar'] = text

            # Number/count
            if i < len(headers):
                header_lower = headers[i].lower()
                if 'no' in header_lower or 'qty' in header_lower or 'count' in header_lower:
                    try:
                        row.parsed_data['count'] = int(text)
                    except ValueError:
                        pass

        return row

    def _detect_schedules_ocr(self, image: np.ndarray) -> List[ExtractedSchedule]:
        """Detect and extract schedules using OCR when vector text not available."""
        schedules = []

        try:
            import pytesseract

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # OCR the entire image
            data = pytesseract.image_to_data(
                gray, config='--psm 6',
                output_type=pytesseract.Output.DICT
            )

            # Build text blocks
            texts = []
            for i, text in enumerate(data['text']):
                text = text.strip()
                if not text:
                    continue

                x, y, w, h = (
                    data['left'][i], data['top'][i],
                    data['width'][i], data['height'][i]
                )
                texts.append({
                    'text': text,
                    'bbox': (x, y, x + w, y + h),
                    'cx': x + w/2,
                    'cy': y + h/2
                })

            # Look for schedule titles
            for t in texts:
                text_lower = t['text'].lower()
                schedule_type = None

                if 'column' in text_lower and 'schedule' in text_lower:
                    schedule_type = 'column'
                elif 'beam' in text_lower and 'schedule' in text_lower:
                    schedule_type = 'beam'
                elif 'footing' in text_lower and 'schedule' in text_lower:
                    schedule_type = 'footing'

                if schedule_type:
                    # Extract schedule region below title
                    region_texts = [
                        tt for tt in texts
                        if tt['cy'] > t['cy'] and
                        abs(tt['cx'] - t['cx']) < 400 and
                        tt['cy'] < t['cy'] + 500
                    ]

                    if region_texts:
                        rows = self._group_into_rows(region_texts)
                        if len(rows) >= 2:
                            headers = [tt['text'] for tt in rows[0]]
                            parsed_rows = []

                            for i, row_texts in enumerate(rows[1:]):
                                cells = [
                                    ScheduleCell(
                                        row=i, col=j,
                                        text=tt['text'],
                                        bbox=tt['bbox']
                                    )
                                    for j, tt in enumerate(row_texts)
                                ]
                                sched_row = ScheduleRow(row_num=i, cells=cells)
                                sched_row = self._parse_schedule_row(
                                    sched_row, headers, schedule_type
                                )
                                parsed_rows.append(sched_row)

                            schedules.append(ExtractedSchedule(
                                schedule_type=schedule_type,
                                title=t['text'],
                                headers=headers,
                                rows=parsed_rows,
                                bbox=t['bbox'],
                                confidence=0.5
                            ))

        except ImportError:
            logger.debug("pytesseract not available for OCR schedule detection")
        except Exception as e:
            logger.debug(f"OCR schedule detection failed: {e}")

        return schedules

    def _build_result(self, schedules: List[ExtractedSchedule]) -> ScheduleExtractionResult:
        """Build final result with all size mappings."""

        column_sizes = {}
        beam_sizes = {}
        footing_sizes = {}
        slab_thicknesses = {}
        concrete_grades = {}

        for sched in schedules:
            for row in sched.rows:
                if not row.element_label:
                    continue

                label = row.element_label

                if sched.schedule_type == 'column':
                    if 'size_mm' in row.parsed_data:
                        size = row.parsed_data['size_mm']
                        if len(size) >= 2:
                            column_sizes[label] = (size[0], size[1])

                elif sched.schedule_type == 'beam':
                    if 'size_mm' in row.parsed_data:
                        size = row.parsed_data['size_mm']
                        if len(size) >= 2:
                            beam_sizes[label] = (size[0], size[1])

                elif sched.schedule_type == 'footing':
                    if 'size_mm' in row.parsed_data:
                        size = row.parsed_data['size_mm']
                        if len(size) >= 3:
                            footing_sizes[label] = (size[0], size[1], size[2])
                        elif len(size) == 2:
                            # Assume square footing with default depth
                            footing_sizes[label] = (size[0], size[1], 450)

                elif sched.schedule_type == 'slab':
                    if 'size_mm' in row.parsed_data:
                        size = row.parsed_data['size_mm']
                        if isinstance(size, int):
                            slab_thicknesses[label] = size
                        elif len(size) == 1:
                            slab_thicknesses[label] = size[0]

                # Concrete grade
                if 'concrete_grade' in row.parsed_data:
                    concrete_grades[label] = row.parsed_data['concrete_grade']

        return ScheduleExtractionResult(
            schedules=schedules,
            column_sizes=column_sizes,
            beam_sizes=beam_sizes,
            footing_sizes=footing_sizes,
            slab_thicknesses=slab_thicknesses,
            concrete_grades=concrete_grades
        )

    def extract_from_text(self, text: str) -> ScheduleExtractionResult:
        """
        Extract schedule data from raw text (e.g., from PDF text extraction).

        Args:
            text: Raw text containing schedule data

        Returns:
            ScheduleExtractionResult
        """
        column_sizes = {}
        beam_sizes = {}
        footing_sizes = {}
        concrete_grades = {}

        # Column patterns: C1 - 230x450, C1: 230 x 450
        col_pattern = r'(C\d+)\s*[-:]\s*(\d{3})\s*[xX×]\s*(\d{3})'
        for match in re.finditer(col_pattern, text, re.IGNORECASE):
            label = match.group(1).upper()
            w, d = int(match.group(2)), int(match.group(3))
            column_sizes[label] = (w, d)

        # Beam patterns
        beam_pattern = r'(B\d+|BM\d+)\s*[-:]\s*(\d{3})\s*[xX×]\s*(\d{3})'
        for match in re.finditer(beam_pattern, text, re.IGNORECASE):
            label = match.group(1).upper()
            w, d = int(match.group(2)), int(match.group(3))
            beam_sizes[label] = (w, d)

        # Footing patterns
        ftg_pattern = r'([ICF]?F\d+)\s*[-:]\s*(\d{3,4})\s*[xX×]\s*(\d{3,4})\s*[xX×]\s*(\d{3})'
        for match in re.finditer(ftg_pattern, text, re.IGNORECASE):
            label = match.group(1).upper()
            l, w, d = int(match.group(2)), int(match.group(3)), int(match.group(4))
            footing_sizes[label] = (l, w, d)

        # Concrete grades
        grade_pattern = r'(C\d+|B\d+|F\d+)\s*[-:]\s*M\s*(\d{2})'
        for match in re.finditer(grade_pattern, text, re.IGNORECASE):
            label = match.group(1).upper()
            grade = f"M{match.group(2)}"
            concrete_grades[label] = grade

        return ScheduleExtractionResult(
            schedules=[],
            column_sizes=column_sizes,
            beam_sizes=beam_sizes,
            footing_sizes=footing_sizes,
            concrete_grades=concrete_grades
        )


def extract_schedules(
    image: np.ndarray,
    vector_texts: List[Any] = None
) -> ScheduleExtractionResult:
    """
    Convenience function to extract schedules.
    """
    extractor = ScheduleExtractor()
    return extractor.extract(image, vector_texts)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = extract_schedules(img)
            print(f"Column sizes: {result.column_sizes}")
            print(f"Beam sizes: {result.beam_sizes}")
            print(f"Footing sizes: {result.footing_sizes}")
    else:
        # Test with sample text
        sample = """
        COLUMN SCHEDULE
        C1 - 230x450 M25
        C2 - 300x600 M25
        C3 - 230x230 M20

        BEAM SCHEDULE
        B1 - 230x450 M25
        B2 - 230x600 M25

        FOOTING SCHEDULE
        F1 - 1500x1500x450 M20
        F2 - 1800x1800x500 M20
        """

        extractor = ScheduleExtractor()
        result = extractor.extract_from_text(sample)
        print(f"Column sizes: {result.column_sizes}")
        print(f"Beam sizes: {result.beam_sizes}")
        print(f"Footing sizes: {result.footing_sizes}")
