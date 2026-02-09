"""
Floor Plan Ingestion Module
Vector-first extraction with raster fallback.
Handles: vector PDFs, raster PDFs, PNG/JPG images
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PlanType(Enum):
    """Type of floor plan source."""
    VECTOR_PDF = "vector_pdf"
    RASTER_PDF = "raster_pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class VectorLine:
    """Represents a vector line extracted from PDF."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float = 1.0
    color: Tuple[int, int, int] = (0, 0, 0)
    layer: str = ""

    @property
    def length(self) -> float:
        """Calculate line length."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx*dx + dy*dy)

    @property
    def is_horizontal(self) -> bool:
        """Check if line is approximately horizontal."""
        return abs(self.end[1] - self.start[1]) < 2.0

    @property
    def is_vertical(self) -> bool:
        """Check if line is approximately vertical."""
        return abs(self.end[0] - self.start[0]) < 2.0


@dataclass
class VectorText:
    """Represents text extracted from PDF."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float = 10.0
    font_name: str = ""

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of text bbox."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


@dataclass
class VectorPath:
    """Represents a complete path (potentially closed) from PDF."""
    points: List[Tuple[float, float]]
    is_closed: bool = False
    fill_color: Optional[Tuple[int, int, int]] = None
    stroke_color: Optional[Tuple[int, int, int]] = None
    stroke_width: float = 1.0


@dataclass
class IngestedPlan:
    """Result of floor plan ingestion."""
    plan_id: str
    source_path: Path
    plan_type: PlanType

    # Raster image (always available)
    image: np.ndarray = field(default_factory=lambda: np.array([]))
    image_dpi: int = 300

    # Vector data (only for vector PDFs)
    vector_lines: List[VectorLine] = field(default_factory=list)
    vector_texts: List[VectorText] = field(default_factory=list)
    vector_paths: List[VectorPath] = field(default_factory=list)

    # Metadata
    page_width_pts: float = 0.0
    page_height_pts: float = 0.0
    page_number: int = 0
    total_pages: int = 1

    # Coordinate transform (PDF points to image pixels)
    pts_to_px_scale: float = 1.0

    @property
    def has_vector_content(self) -> bool:
        """Check if plan has usable vector content."""
        return len(self.vector_lines) > 50 or len(self.vector_paths) > 10

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Get image dimensions (height, width)."""
        if self.image.size > 0:
            return self.image.shape[:2]
        return (0, 0)


class PlanIngester:
    """
    Main class for ingesting floor plans.
    Supports vector PDFs (preferred), raster PDFs, and images.
    """

    def __init__(self, dpi: int = 300):
        """
        Initialize ingester.

        Args:
            dpi: DPI for rasterizing PDFs (300-600 recommended)
        """
        self.dpi = dpi
        self._fitz = None
        self._pdfplumber = None

    def _load_fitz(self):
        """Lazy load PyMuPDF."""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                logger.warning("PyMuPDF not installed. Vector extraction disabled.")
        return self._fitz

    def _load_pdfplumber(self):
        """Lazy load pdfplumber."""
        if self._pdfplumber is None:
            try:
                import pdfplumber
                self._pdfplumber = pdfplumber
            except ImportError:
                logger.warning("pdfplumber not installed.")
        return self._pdfplumber

    def ingest(self, file_path: Path, page_num: int = 0) -> IngestedPlan:
        """
        Ingest a floor plan file.

        Args:
            file_path: Path to PDF or image file
            page_num: Page number for multi-page PDFs (0-indexed)

        Returns:
            IngestedPlan object with extracted data
        """
        file_path = Path(file_path)
        plan_id = file_path.stem

        logger.info(f"Ingesting plan: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            return self._ingest_pdf(file_path, plan_id, page_num)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            return self._ingest_image(file_path, plan_id)
        else:
            logger.warning(f"Unknown file type: {suffix}")
            return IngestedPlan(
                plan_id=plan_id,
                source_path=file_path,
                plan_type=PlanType.UNKNOWN
            )

    def _ingest_pdf(self, file_path: Path, plan_id: str, page_num: int) -> IngestedPlan:
        """Ingest a PDF file with vector-first approach."""
        fitz = self._load_fitz()

        if fitz is None:
            logger.error("PyMuPDF required for PDF processing")
            return IngestedPlan(
                plan_id=plan_id,
                source_path=file_path,
                plan_type=PlanType.UNKNOWN
            )

        doc = fitz.open(file_path)
        total_pages = len(doc)

        if page_num >= total_pages:
            logger.warning(f"Page {page_num} doesn't exist, using page 0")
            page_num = 0

        page = doc[page_num]
        page_rect = page.rect

        # Calculate scale for coordinate transform
        pts_to_px_scale = self.dpi / 72.0

        # Extract vector content
        vector_lines, vector_paths = self._extract_vector_content(page, pts_to_px_scale)
        vector_texts = self._extract_text_content(page, pts_to_px_scale)

        # Determine if this is truly vector or scanned
        has_significant_vectors = len(vector_lines) > 50 or len(vector_paths) > 10
        plan_type = PlanType.VECTOR_PDF if has_significant_vectors else PlanType.RASTER_PDF

        logger.info(f"Plan type: {plan_type.value}, Lines: {len(vector_lines)}, "
                   f"Paths: {len(vector_paths)}, Texts: {len(vector_texts)}")

        # Render to image (always needed for visualization and fallback)
        image = self._render_page_to_image(page)

        doc.close()

        return IngestedPlan(
            plan_id=plan_id,
            source_path=file_path,
            plan_type=plan_type,
            image=image,
            image_dpi=self.dpi,
            vector_lines=vector_lines,
            vector_texts=vector_texts,
            vector_paths=vector_paths,
            page_width_pts=page_rect.width,
            page_height_pts=page_rect.height,
            page_number=page_num,
            total_pages=total_pages,
            pts_to_px_scale=pts_to_px_scale
        )

    def _extract_vector_content(self, page, scale: float) -> Tuple[List[VectorLine], List[VectorPath]]:
        """Extract vector lines and paths from PDF page."""
        lines = []
        paths = []

        try:
            # Get all drawings on the page
            drawings = page.get_drawings()

            for drawing in drawings:
                items = drawing.get("items", [])
                color = drawing.get("color", (0, 0, 0))
                width = drawing.get("width", 1.0)
                if width is None:
                    width = 1.0
                fill = drawing.get("fill")

                # Convert color to RGB tuple
                if color is None:
                    color = (0, 0, 0)
                elif isinstance(color, (int, float)):
                    c = int(color * 255)
                    color = (c, c, c)
                elif len(color) == 3:
                    color = tuple(int(c * 255) for c in color)
                else:
                    color = (0, 0, 0)

                path_points = []

                for item in items:
                    item_type = item[0]

                    if item_type == "l":  # Line
                        if item[1] is None or item[2] is None:
                            continue
                        p1 = (item[1].x * scale, item[1].y * scale)
                        p2 = (item[2].x * scale, item[2].y * scale)
                        lines.append(VectorLine(
                            start=p1,
                            end=p2,
                            width=width * scale,
                            color=color
                        ))
                        path_points.extend([p1, p2])

                    elif item_type == "re":  # Rectangle
                        rect = item[1]
                        if rect is None:
                            continue
                        x0, y0 = rect.x0 * scale, rect.y0 * scale
                        x1, y1 = rect.x1 * scale, rect.y1 * scale
                        # Add rectangle as 4 lines
                        rect_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                        for i in range(4):
                            p1 = rect_points[i]
                            p2 = rect_points[(i + 1) % 4]
                            lines.append(VectorLine(
                                start=p1, end=p2,
                                width=width * scale,
                                color=color
                            ))
                        path_points.extend(rect_points)

                    elif item_type == "qu":  # Quad
                        quad = item[1]
                        if quad is None:
                            continue
                        quad_points = [
                            (quad.ul.x * scale, quad.ul.y * scale),
                            (quad.ur.x * scale, quad.ur.y * scale),
                            (quad.lr.x * scale, quad.lr.y * scale),
                            (quad.ll.x * scale, quad.ll.y * scale),
                        ]
                        for i in range(4):
                            p1 = quad_points[i]
                            p2 = quad_points[(i + 1) % 4]
                            lines.append(VectorLine(
                                start=p1, end=p2,
                                width=width * scale,
                                color=color
                            ))
                        path_points.extend(quad_points)

                    elif item_type == "c":  # Curve (bezier)
                        # Approximate curve with line segments
                        if item[1] is None or len(item) < 5 or item[4] is None:
                            continue
                        p1 = (item[1].x * scale, item[1].y * scale)
                        p4 = (item[4].x * scale, item[4].y * scale)
                        lines.append(VectorLine(
                            start=p1, end=p4,
                            width=width * scale,
                            color=color
                        ))
                        path_points.extend([p1, p4])

                # Create path if we have points
                if len(path_points) >= 2:
                    fill_color = None
                    if fill is not None:
                        if isinstance(fill, (int, float)):
                            c = int(fill * 255)
                            fill_color = (c, c, c)
                        elif len(fill) == 3:
                            fill_color = tuple(int(c * 255) for c in fill)

                    paths.append(VectorPath(
                        points=path_points,
                        is_closed=(len(path_points) >= 3 and
                                  np.allclose(path_points[0], path_points[-1], atol=2)),
                        fill_color=fill_color,
                        stroke_color=color,
                        stroke_width=width * scale
                    ))

        except Exception as e:
            logger.warning(f"Error extracting vectors: {e}")

        return lines, paths

    def _extract_text_content(self, page, scale: float) -> List[VectorText]:
        """Extract text from PDF page."""
        texts = []

        try:
            # Get text blocks with position info
            text_dict = page.get_text("dict", flags=11)

            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                bbox = span.get("bbox", (0, 0, 0, 0))
                                texts.append(VectorText(
                                    text=text,
                                    bbox=(
                                        bbox[0] * scale,
                                        bbox[1] * scale,
                                        bbox[2] * scale,
                                        bbox[3] * scale
                                    ),
                                    font_size=span.get("size", 10) * scale,
                                    font_name=span.get("font", "")
                                ))
        except Exception as e:
            logger.warning(f"Error extracting text: {e}")

        return texts

    def _render_page_to_image(self, page) -> np.ndarray:
        """Render PDF page to numpy array."""
        import cv2

        # Calculate zoom factor for desired DPI
        zoom = self.dpi / 72.0
        mat = page.get_pixmap(matrix=page.transformation_matrix *
                              self._fitz.Matrix(zoom, zoom))

        # Convert to numpy array
        img_data = np.frombuffer(mat.samples, dtype=np.uint8)
        img = img_data.reshape(mat.height, mat.width, mat.n)

        # Convert to BGR for OpenCV compatibility
        if mat.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif mat.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif mat.n == 1:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def _ingest_image(self, file_path: Path, plan_id: str) -> IngestedPlan:
        """Ingest an image file."""
        import cv2

        img = cv2.imread(str(file_path))

        if img is None:
            logger.error(f"Could not read image: {file_path}")
            return IngestedPlan(
                plan_id=plan_id,
                source_path=file_path,
                plan_type=PlanType.UNKNOWN
            )

        return IngestedPlan(
            plan_id=plan_id,
            source_path=file_path,
            plan_type=PlanType.IMAGE,
            image=img,
            image_dpi=self.dpi,  # Assume standard DPI for images
            page_width_pts=img.shape[1] * 72 / self.dpi,
            page_height_pts=img.shape[0] * 72 / self.dpi,
            pts_to_px_scale=self.dpi / 72.0
        )


def ingest_plan(file_path: Path, dpi: int = 300, page_num: int = 0) -> IngestedPlan:
    """
    Convenience function to ingest a floor plan.

    Args:
        file_path: Path to floor plan file
        dpi: DPI for rasterization
        page_num: Page number for PDFs

    Returns:
        IngestedPlan object
    """
    ingester = PlanIngester(dpi=dpi)
    return ingester.ingest(file_path, page_num)


if __name__ == "__main__":
    # Test ingestion
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        plan = ingest_plan(Path(sys.argv[1]))
        print(f"Plan ID: {plan.plan_id}")
        print(f"Type: {plan.plan_type.value}")
        print(f"Image shape: {plan.image_shape}")
        print(f"Vector lines: {len(plan.vector_lines)}")
        print(f"Vector texts: {len(plan.vector_texts)}")
    else:
        print("Usage: python ingest.py <plan_file>")
