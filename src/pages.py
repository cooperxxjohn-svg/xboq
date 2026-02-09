"""
Page Classification Module
Classifies drawing pages into: architectural, structural, schedule, etc.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from enum import Enum
import numpy as np
import cv2
import yaml

logger = logging.getLogger(__name__)


class DrawingType(Enum):
    """Types of construction drawings."""
    ARCHITECTURAL_PLAN = "architectural_plan"
    STRUCTURAL_FRAMING = "structural_framing"
    COLUMN_LAYOUT = "column_layout"
    FOUNDATION_PLAN = "foundation_plan"
    SCHEDULE = "schedule"
    SECTION_DETAIL = "section_detail"
    ELEVATION = "elevation"
    SITE_PLAN = "site_plan"
    UNKNOWN = "unknown"


@dataclass
class PageInfo:
    """Information about a single drawing page."""
    page_number: int
    drawing_type: DrawingType
    confidence: float
    title: str = ""
    sheet_number: str = ""
    detected_keywords: List[str] = field(default_factory=list)
    has_table: bool = False
    line_density: float = 0.0
    text_density: float = 0.0
    scale: Optional[str] = None
    image: Optional[np.ndarray] = None
    vector_texts: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProjectPages:
    """All pages in a project."""
    project_id: str
    source_path: Path
    total_pages: int
    pages: List[PageInfo]

    def get_by_type(self, dtype: DrawingType) -> List[PageInfo]:
        """Get pages of a specific type."""
        return [p for p in self.pages if p.drawing_type == dtype]

    @property
    def architectural_pages(self) -> List[PageInfo]:
        return self.get_by_type(DrawingType.ARCHITECTURAL_PLAN)

    @property
    def structural_pages(self) -> List[PageInfo]:
        return (self.get_by_type(DrawingType.STRUCTURAL_FRAMING) +
                self.get_by_type(DrawingType.COLUMN_LAYOUT) +
                self.get_by_type(DrawingType.FOUNDATION_PLAN))

    @property
    def schedule_pages(self) -> List[PageInfo]:
        return self.get_by_type(DrawingType.SCHEDULE)


class PageClassifier:
    """
    Classifies drawing pages using OCR, keywords, and visual features.
    """

    def __init__(self, keywords_path: Optional[Path] = None):
        """
        Initialize classifier with keywords.

        Args:
            keywords_path: Path to structural_keywords.yaml
        """
        self.keywords = {}
        self.drawing_types = {}

        # Load keywords
        if keywords_path and keywords_path.exists():
            self._load_keywords(keywords_path)
        else:
            default_path = Path(__file__).parent.parent / "rules" / "structural_keywords.yaml"
            if default_path.exists():
                self._load_keywords(default_path)
            else:
                self._init_default_keywords()

    def _load_keywords(self, path: Path):
        """Load keywords from YAML."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            self.drawing_types = data.get('drawing_types', {})
            self.keywords = data
            logger.info(f"Loaded keywords from {path}")
        except Exception as e:
            logger.error(f"Failed to load keywords: {e}")
            self._init_default_keywords()

    def _init_default_keywords(self):
        """Initialize with minimal defaults."""
        self.drawing_types = {
            'architectural_plan': {
                'title_keywords': ['floor plan', 'ground floor', 'typical floor', 'layout'],
                'sheet_prefixes': ['A-', 'AR-']
            },
            'structural_framing': {
                'title_keywords': ['framing plan', 'beam layout', 'structural'],
                'sheet_prefixes': ['S-', 'ST-']
            },
            'column_layout': {
                'title_keywords': ['column layout', 'column plan', 'column grid'],
                'sheet_prefixes': ['COL-']
            },
            'foundation_plan': {
                'title_keywords': ['foundation', 'footing', 'plinth'],
                'sheet_prefixes': ['F-', 'FND-']
            },
            'schedule': {
                'title_keywords': ['schedule', 'bbs', 'bar bending'],
                'table_indicators': ['mark', 'size', 'nos', 'qty']
            },
            'section_detail': {
                'title_keywords': ['section', 'detail'],
                'sheet_prefixes': ['D-', 'SEC-']
            }
        }

    def classify_pages(self, file_path: Path, dpi: int = 200) -> ProjectPages:
        """
        Classify all pages in a PDF/image file.

        Args:
            file_path: Path to PDF or image
            dpi: DPI for rendering

        Returns:
            ProjectPages with classified pages
        """
        file_path = Path(file_path)
        project_id = file_path.stem

        logger.info(f"Classifying pages in: {file_path}")

        pages = []
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            pages = self._process_pdf(file_path, dpi)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            pages = [self._process_image(file_path, 0)]
        else:
            logger.warning(f"Unsupported file type: {suffix}")

        return ProjectPages(
            project_id=project_id,
            source_path=file_path,
            total_pages=len(pages),
            pages=pages
        )

    def _process_pdf(self, file_path: Path, dpi: int) -> List[PageInfo]:
        """Process all pages in a PDF."""
        pages = []

        try:
            import fitz
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                text = page.get_text()
                texts = self._extract_text_blocks(page)

                # Render to image
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)

                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Classify page
                page_info = self._classify_page(page_num, text, texts, img)
                pages.append(page_info)

            doc.close()

        except ImportError:
            logger.error("PyMuPDF required for PDF processing")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")

        return pages

    def _extract_text_blocks(self, page) -> List[Dict]:
        """Extract text blocks with positions from PDF page."""
        blocks = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                bbox = span.get("bbox", (0, 0, 0, 0))
                                blocks.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'size': span.get("size", 10)
                                })
        except Exception as e:
            logger.debug(f"Text extraction error: {e}")
        return blocks

    def _process_image(self, file_path: Path, page_num: int) -> PageInfo:
        """Process a single image file."""
        img = cv2.imread(str(file_path))

        if img is None:
            return PageInfo(
                page_number=page_num,
                drawing_type=DrawingType.UNKNOWN,
                confidence=0.0,
                warnings=["Failed to load image"]
            )

        # Try OCR
        text = self._ocr_image(img)
        texts = []

        return self._classify_page(page_num, text, texts, img)

    def _ocr_image(self, image: np.ndarray) -> str:
        """Perform OCR on image."""
        try:
            import pytesseract
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            text = pytesseract.image_to_string(gray)
            return text
        except ImportError:
            return ""
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return ""

    def _classify_page(
        self,
        page_num: int,
        text: str,
        text_blocks: List[Dict],
        image: np.ndarray
    ) -> PageInfo:
        """
        Classify a single page.

        Args:
            page_num: Page number (0-indexed)
            text: Full text from page
            text_blocks: Text blocks with positions
            image: Page image

        Returns:
            PageInfo
        """
        text_lower = text.lower()

        # Extract title (usually in first few text blocks or at top)
        title = self._extract_title(text_blocks, text_lower)

        # Extract sheet number
        sheet_number = self._extract_sheet_number(text_lower)

        # Calculate visual features
        line_density = self._calculate_line_density(image)
        text_density = len(text) / max(1, image.shape[0] * image.shape[1]) * 1000000

        # Check for tables
        has_table = self._detect_table(image, text_lower)

        # Score each drawing type
        scores = {}
        matched_keywords = {}

        for dtype_name, dtype_config in self.drawing_types.items():
            score, keywords = self._score_drawing_type(
                dtype_name, dtype_config, text_lower, title,
                sheet_number, has_table, line_density
            )
            scores[dtype_name] = score
            matched_keywords[dtype_name] = keywords

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Convert to DrawingType enum
        try:
            drawing_type = DrawingType(best_type)
        except ValueError:
            drawing_type = DrawingType.UNKNOWN

        # If score too low, mark as unknown
        if best_score < 0.3:
            drawing_type = DrawingType.UNKNOWN

        # Confidence based on score differential
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            confidence = min(0.95, best_score * (1 - sorted_scores[1] / sorted_scores[0] * 0.3))
        else:
            confidence = best_score

        return PageInfo(
            page_number=page_num,
            drawing_type=drawing_type,
            confidence=round(confidence, 2),
            title=title,
            sheet_number=sheet_number,
            detected_keywords=matched_keywords.get(best_type, []),
            has_table=has_table,
            line_density=line_density,
            text_density=text_density,
            image=image,
            vector_texts=text_blocks
        )

    def _extract_title(self, text_blocks: List[Dict], text_lower: str) -> str:
        """Extract drawing title from text blocks."""
        # Look for large text at top or bottom
        if text_blocks:
            # Sort by font size (descending)
            sorted_blocks = sorted(text_blocks, key=lambda x: x.get('size', 0), reverse=True)
            for block in sorted_blocks[:5]:
                text = block['text']
                if len(text) > 5 and len(text) < 100:
                    # Check if it looks like a title
                    if any(kw in text.lower() for kw in
                           ['plan', 'layout', 'schedule', 'section', 'detail', 'elevation']):
                        return text

        # Fallback: look for title patterns in full text
        patterns = [
            r'(?:drawing|dwg|sheet)?\s*(?:title|name)?\s*[:\-]?\s*([A-Z][A-Za-z\s]+(?:PLAN|LAYOUT|SCHEDULE|DETAIL|SECTION))',
            r'((?:GROUND|FIRST|SECOND|TYPICAL|TERRACE)\s+(?:FLOOR\s+)?PLAN)',
            r'((?:COLUMN|BEAM|FOOTING|FOUNDATION)\s+(?:LAYOUT|PLAN|SCHEDULE))',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()

        return ""

    def _extract_sheet_number(self, text_lower: str) -> str:
        """Extract sheet number."""
        patterns = [
            r'(?:sheet|dwg|drawing)\s*(?:no\.?|#)?\s*[:\-]?\s*([A-Z]?\-?\d+)',
            r'\b([ASFD]\-\d+)\b',
            r'(?:^|\s)([A-Z]{1,3}\-\d{1,3})(?:\s|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return ""

    def _calculate_line_density(self, image: np.ndarray) -> float:
        """Calculate density of lines in image."""
        if image.size == 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

        if lines is None:
            return 0.0

        # Density = total line length / image area
        total_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            total_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)

        area = image.shape[0] * image.shape[1]
        return total_length / area * 1000  # Normalize

    def _detect_table(self, image: np.ndarray, text_lower: str) -> bool:
        """Detect if page contains a table/schedule."""
        # Check for table keywords
        table_keywords = ['schedule', 'bbs', 'mark', 'size', 'nos', 'qty', 'quantity']
        keyword_count = sum(1 for kw in table_keywords if kw in text_lower)

        if keyword_count >= 3:
            return True

        # Visual detection: look for grid pattern
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)

        # Detect horizontal and vertical lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

        h_count = np.sum(h_lines > 0)
        v_count = np.sum(v_lines > 0)

        # If both horizontal and vertical lines are significant, likely a table
        threshold = image.shape[0] * image.shape[1] * 0.001
        return h_count > threshold and v_count > threshold

    def _score_drawing_type(
        self,
        dtype_name: str,
        dtype_config: Dict,
        text_lower: str,
        title: str,
        sheet_number: str,
        has_table: bool,
        line_density: float
    ) -> Tuple[float, List[str]]:
        """
        Score how well text matches a drawing type.

        Returns:
            Tuple of (score, matched_keywords)
        """
        score = 0.0
        matched = []

        title_lower = title.lower()

        # Check title keywords
        title_keywords = dtype_config.get('title_keywords', [])
        for kw in title_keywords:
            if kw in title_lower:
                score += 0.4
                matched.append(kw)
            elif kw in text_lower:
                score += 0.2
                matched.append(kw)

        # Check sheet prefixes
        sheet_prefixes = dtype_config.get('sheet_prefixes', [])
        for prefix in sheet_prefixes:
            if sheet_number.upper().startswith(prefix.upper()):
                score += 0.3
                matched.append(f"sheet:{prefix}")

        # Check negative keywords
        negative_keywords = dtype_config.get('negative_keywords', [])
        for neg_kw in negative_keywords:
            if neg_kw in text_lower:
                score -= 0.3

        # Special handling for schedules
        if dtype_name == 'schedule':
            if has_table:
                score += 0.4
                matched.append("has_table")

            table_indicators = dtype_config.get('table_indicators', [])
            indicator_count = sum(1 for ind in table_indicators if ind in text_lower)
            if indicator_count >= 2:
                score += 0.2
                matched.append(f"table_indicators:{indicator_count}")

        # Structural drawings typically have medium line density
        if dtype_name in ['structural_framing', 'column_layout', 'foundation_plan']:
            if 5 < line_density < 50:
                score += 0.1

        # Architectural plans have more detail
        if dtype_name == 'architectural_plan':
            if line_density > 30:
                score += 0.1

        return max(0, min(1, score)), matched


def classify_project_pages(file_path: Path, dpi: int = 200) -> ProjectPages:
    """
    Classify pages in a project file.

    Args:
        file_path: Path to PDF or image
        dpi: Rendering DPI

    Returns:
        ProjectPages
    """
    classifier = PageClassifier()
    return classifier.classify_pages(file_path, dpi)


@dataclass
class PageClassificationResult:
    """Result of single page classification."""
    drawing_type: DrawingType
    confidence: float
    detected_keywords: List[str] = field(default_factory=list)
    title: str = ""


def classify_drawing(
    image: np.ndarray,
    vector_texts: List[Any] = None
) -> PageClassificationResult:
    """
    Classify a single drawing image.

    Args:
        image: Drawing image
        vector_texts: Optional text blocks from PDF

    Returns:
        PageClassificationResult
    """
    classifier = PageClassifier()

    # Combine all text
    all_text = ""
    text_blocks = []
    if vector_texts:
        for vt in vector_texts:
            text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
            all_text += text + " "
            bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))
            text_blocks.append({'text': text, 'bbox': bbox, 'size': 10})

    # Use the existing _classify_page method
    page_info = classifier._classify_page(0, all_text, text_blocks, image)

    drawing_type = page_info.drawing_type
    confidence = page_info.confidence
    matched = page_info.detected_keywords

    return PageClassificationResult(
        drawing_type=drawing_type,
        confidence=confidence,
        detected_keywords=matched,
        title=all_text[:50] if all_text else ""
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        result = classify_project_pages(Path(sys.argv[1]))
        print(f"\nProject: {result.project_id}")
        print(f"Total pages: {result.total_pages}")
        print("\nPage Classification:")
        for page in result.pages:
            print(f"  Page {page.page_number + 1}: {page.drawing_type.value} "
                  f"({page.confidence:.0%}) - {page.title or 'No title'}")
    else:
        print("Usage: python pages.py <file.pdf>")
