"""
Project Indexer - Fast indexing with caching.

Phase 0: Index all pages at low DPI with:
- Thumbnail generation
- Text extraction (vector-first, OCR fallback)
- Feature computation (line density, table-likeness)
- Page hashing for cache invalidation
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PageFeatures:
    """Computed features for a page."""
    line_density: float = 0.0
    table_likeness: float = 0.0
    text_density: float = 0.0
    has_title_block: bool = False
    dominant_orientation: str = "mixed"  # horizontal, vertical, mixed
    estimated_content_type: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with proper Python types (not numpy)."""
        return {
            "line_density": float(self.line_density),
            "table_likeness": float(self.table_likeness),
            "text_density": float(self.text_density),
            "has_title_block": bool(self.has_title_block),
            "dominant_orientation": str(self.dominant_orientation),
            "estimated_content_type": str(self.estimated_content_type),
        }


@dataclass
class PageIndex:
    """Index entry for a single page."""
    file_path: str
    page_number: int
    page_hash: str
    thumb_path: str
    width: int
    height: int
    extracted_text: str
    text_snippet: str  # First 500 chars
    candidate_titles: List[str]
    features: PageFeatures
    vector_text_available: bool = False
    ocr_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "page_number": int(self.page_number),
            "page_hash": str(self.page_hash),
            "thumb_path": str(self.thumb_path),
            "width": int(self.width),
            "height": int(self.height),
            "text_snippet": str(self.text_snippet),
            "candidate_titles": list(self.candidate_titles),
            "features": self.features.to_dict(),
            "vector_text_available": bool(self.vector_text_available),
            "ocr_used": bool(self.ocr_used),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageIndex":
        features = PageFeatures(**data.get("features", {}))
        return cls(
            file_path=data["file_path"],
            page_number=data["page_number"],
            page_hash=data["page_hash"],
            thumb_path=data["thumb_path"],
            width=data.get("width", 0),
            height=data.get("height", 0),
            extracted_text=data.get("extracted_text", ""),
            text_snippet=data["text_snippet"],
            candidate_titles=data.get("candidate_titles", []),
            features=features,
            vector_text_available=data.get("vector_text_available", False),
            ocr_used=data.get("ocr_used", False),
        )


@dataclass
class ProjectIndex:
    """Index for entire project."""
    project_id: str
    source_path: str
    total_files: int
    total_pages: int
    pages: List[PageIndex]
    indexed_at: str = ""
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source_path": self.source_path,
            "total_files": self.total_files,
            "total_pages": self.total_pages,
            "indexed_at": self.indexed_at,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "pages": [p.to_dict() for p in self.pages],
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ProjectIndex":
        with open(path) as f:
            data = json.load(f)
        return cls(
            project_id=data["project_id"],
            source_path=data["source_path"],
            total_files=data["total_files"],
            total_pages=data["total_pages"],
            indexed_at=data.get("indexed_at", ""),
            cache_hits=data.get("cache_hits", 0),
            cache_misses=data.get("cache_misses", 0),
            pages=[PageIndex.from_dict(p) for p in data["pages"]],
        )


class ProjectIndexer:
    """
    Fast project indexer with caching.

    Renders thumbnails at low DPI and extracts text efficiently.
    Caches results by page hash to support resume.
    """

    # Title patterns for candidate extraction
    TITLE_PATTERNS = [
        r"(?:GROUND|FIRST|SECOND|THIRD|FOURTH|TYPICAL|TERRACE|BASEMENT|STILT|PODIUM)\s+(?:FLOOR\s+)?PLAN",
        r"(?:FLOOR|LAYOUT|UNIT|FLAT)\s+PLAN",
        r"(?:DOOR|WINDOW|FINISH|ROOM)\s+SCHEDULE",
        r"(?:BAR\s+BENDING|BBS)\s+SCHEDULE",
        r"(?:COLUMN|BEAM|FOOTING|FOUNDATION|PLINTH)\s+(?:LAYOUT|PLAN|SCHEDULE)",
        r"SECTION\s+[A-Z]{1,2}(?:\s*-\s*[A-Z])?",
        r"(?:FRONT|SIDE|REAR)\s+ELEVATION",
        r"(?:TOILET|KITCHEN|STAIRCASE|RAILING)\s+DETAIL",
        r"(?:ELECTRICAL|PLUMBING|HVAC|DRAINAGE)\s+(?:LAYOUT|PLAN)",
        r"(?:SITE|KEY|LOCATION|MASTER)\s+PLAN",
        r"(?:COVER|TITLE)\s+(?:SHEET|PAGE)",
        r"DRAWING\s+(?:LIST|INDEX)",
        r"[ASDEFMP]-\d{2,4}",  # Sheet numbers
    ]

    def __init__(
        self,
        thumb_dpi: int = 100,
        ocr_dpi: int = 150,
        enable_ocr_fallback: bool = True,
    ):
        self.thumb_dpi = thumb_dpi
        self.ocr_dpi = ocr_dpi
        self.enable_ocr_fallback = enable_ocr_fallback
        self._fitz = None

    def _load_fitz(self):
        if self._fitz is None:
            import fitz
            self._fitz = fitz
        return self._fitz

    def index_project(
        self,
        input_path: Path,
        output_dir: Path,
        project_id: str,
    ) -> ProjectIndex:
        """
        Index all pages in a project.

        Args:
            input_path: Path to folder or single PDF
            output_dir: Output directory
            project_id: Project identifier

        Returns:
            ProjectIndex
        """
        from datetime import datetime

        input_path = Path(input_path)
        output_dir = Path(output_dir) / project_id
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all files
        if input_path.is_file():
            files = [input_path]
        else:
            files = sorted(
                list(input_path.glob("*.pdf")) +
                list(input_path.glob("*.PDF")) +
                list(input_path.glob("*.png")) +
                list(input_path.glob("*.jpg")) +
                list(input_path.glob("*.jpeg"))
            )

        logger.info(f"Indexing {len(files)} files in project '{project_id}'")

        # Load existing index for cache checking
        index_path = output_dir / "index.json"
        existing_index = {}
        if index_path.exists():
            try:
                old_index = ProjectIndex.load(index_path)
                existing_index = {
                    (p.file_path, p.page_number): p
                    for p in old_index.pages
                }
                logger.info(f"Loaded existing index with {len(existing_index)} pages")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

        # Index all pages
        all_pages = []
        cache_hits = 0
        cache_misses = 0

        for file_path in files:
            logger.info(f"Indexing: {file_path.name}")

            if file_path.suffix.lower() == ".pdf":
                pages = self._index_pdf(file_path, cache_dir, existing_index)
            else:
                pages = self._index_image(file_path, cache_dir, existing_index)

            for page in pages:
                key = (page.file_path, page.page_number)
                if key in existing_index and existing_index[key].page_hash == page.page_hash:
                    cache_hits += 1
                else:
                    cache_misses += 1

            all_pages.extend(pages)

        logger.info(f"Indexed {len(all_pages)} pages (cache hits: {cache_hits}, misses: {cache_misses})")

        # Build project index
        project_index = ProjectIndex(
            project_id=project_id,
            source_path=str(input_path),
            total_files=len(files),
            total_pages=len(all_pages),
            pages=all_pages,
            indexed_at=datetime.now().isoformat(),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

        # Save index
        project_index.save(index_path)
        logger.info(f"Saved index to: {index_path}")

        return project_index

    def _index_pdf(
        self,
        file_path: Path,
        cache_dir: Path,
        existing_index: Dict
    ) -> List[PageIndex]:
        """Index all pages in a PDF."""
        fitz = self._load_fitz()
        pages = []

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]

                # Compute page hash
                page_hash = self._compute_page_hash(page)

                # Check cache
                key = (str(file_path), page_num)
                if key in existing_index and existing_index[key].page_hash == page_hash:
                    # Use cached entry
                    pages.append(existing_index[key])
                    continue

                # Process page
                page_index = self._process_pdf_page(
                    page, page_num, file_path, page_hash, cache_dir
                )
                pages.append(page_index)

                if (page_num + 1) % 10 == 0:
                    logger.info(f"  Indexed {page_num + 1}/{total_pages} pages")

            doc.close()

        except Exception as e:
            logger.error(f"Error indexing PDF {file_path}: {e}")

        return pages

    def _process_pdf_page(
        self,
        page,
        page_num: int,
        file_path: Path,
        page_hash: str,
        cache_dir: Path,
    ) -> PageIndex:
        """Process a single PDF page."""
        fitz = self._fitz
        rect = page.rect

        # Render thumbnail
        zoom = self.thumb_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        thumb_data = np.frombuffer(pix.samples, dtype=np.uint8)
        thumb = thumb_data.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

        # Save thumbnail
        thumb_filename = f"{file_path.stem}_p{page_num:04d}.jpg"
        thumb_path = cache_dir / thumb_filename
        cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # Extract text (vector first)
        vector_text = page.get_text()
        vector_text_available = len(vector_text.strip()) > 20

        extracted_text = vector_text
        ocr_used = False

        # OCR fallback if needed
        if not vector_text_available and self.enable_ocr_fallback:
            extracted_text = self._run_ocr(thumb)
            ocr_used = bool(extracted_text)

        # Compute features
        features = self._compute_features(thumb, extracted_text)

        # Extract candidate titles
        titles = self._extract_candidate_titles(extracted_text)

        return PageIndex(
            file_path=str(file_path),
            page_number=page_num,
            page_hash=page_hash,
            thumb_path=str(thumb_path),
            width=int(rect.width),
            height=int(rect.height),
            extracted_text=extracted_text,
            text_snippet=extracted_text[:500] if extracted_text else "",
            candidate_titles=titles,
            features=features,
            vector_text_available=vector_text_available,
            ocr_used=ocr_used,
        )

    def _index_image(
        self,
        file_path: Path,
        cache_dir: Path,
        existing_index: Dict
    ) -> List[PageIndex]:
        """Index a single image file."""
        img = cv2.imread(str(file_path))
        if img is None:
            logger.warning(f"Could not read image: {file_path}")
            return []

        # Compute hash
        page_hash = hashlib.md5(img.tobytes()[:10000]).hexdigest()

        # Check cache
        key = (str(file_path), 0)
        if key in existing_index and existing_index[key].page_hash == page_hash:
            return [existing_index[key]]

        # Resize for thumbnail
        h, w = img.shape[:2]
        scale = min(800 / w, 800 / h, 1.0)
        thumb = cv2.resize(img, None, fx=scale, fy=scale)

        # Save thumbnail
        thumb_filename = f"{file_path.stem}_p0000.jpg"
        thumb_path = cache_dir / thumb_filename
        cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # OCR for text
        extracted_text = ""
        if self.enable_ocr_fallback:
            extracted_text = self._run_ocr(thumb)

        # Compute features
        features = self._compute_features(thumb, extracted_text)

        # Extract titles
        titles = self._extract_candidate_titles(extracted_text)

        return [PageIndex(
            file_path=str(file_path),
            page_number=0,
            page_hash=page_hash,
            thumb_path=str(thumb_path),
            width=w,
            height=h,
            extracted_text=extracted_text,
            text_snippet=extracted_text[:500] if extracted_text else "",
            candidate_titles=titles,
            features=features,
            vector_text_available=False,
            ocr_used=bool(extracted_text),
        )]

    def _compute_page_hash(self, page) -> str:
        """Compute hash of PDF page content."""
        try:
            # Use page text + drawing count as hash source
            text = page.get_text()[:1000]
            drawings = page.get_drawings()
            hash_src = f"{text}:{len(drawings)}".encode()
            return hashlib.md5(hash_src).hexdigest()
        except Exception:
            return hashlib.md5(str(page.number).encode()).hexdigest()

    def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on image (low resolution)."""
        try:
            import pytesseract
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text
        except ImportError:
            return ""
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return ""

    def _compute_features(self, image: np.ndarray, text: str) -> PageFeatures:
        """Compute visual and text features."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Line detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)

        line_density = 0.0
        h_lines = 0
        v_lines = 0

        if lines is not None:
            total_length = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_length += length
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 15 or angle > 165:
                    h_lines += 1
                elif 75 < angle < 105:
                    v_lines += 1

            line_density = total_length / (h * w) * 1000

        # Dominant orientation
        if h_lines > v_lines * 2:
            orientation = "horizontal"
        elif v_lines > h_lines * 2:
            orientation = "vertical"
        else:
            orientation = "mixed"

        # Table-likeness (many parallel H and V lines)
        table_likeness = 0.0
        if h_lines > 10 and v_lines > 5:
            table_likeness = min(1.0, min(h_lines, v_lines) / max(h_lines, v_lines))

        # Text density
        text_density = len(text) / (h * w) * 1000000 if text else 0

        # Title block detection (usually bottom-right corner)
        title_region = gray[int(h*0.8):, int(w*0.6):]
        has_title_block = np.mean(title_region) < 240  # Has some content

        # Estimate content type
        content_type = "unknown"
        if table_likeness > 0.5 and text_density > 100:
            content_type = "schedule"
        elif line_density > 30 and orientation == "mixed":
            content_type = "drawing"
        elif text_density > 500:
            content_type = "text_heavy"

        return PageFeatures(
            line_density=round(line_density, 2),
            table_likeness=round(table_likeness, 2),
            text_density=round(text_density, 2),
            has_title_block=has_title_block,
            dominant_orientation=orientation,
            estimated_content_type=content_type,
        )

    def _extract_candidate_titles(self, text: str) -> List[str]:
        """Extract candidate titles from text."""
        titles = []
        text_upper = text.upper()

        for pattern in self.TITLE_PATTERNS:
            matches = re.findall(pattern, text_upper)
            titles.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique_titles = []
        for t in titles:
            t_clean = t.strip()
            if t_clean and t_clean not in seen:
                seen.add(t_clean)
                unique_titles.append(t_clean)

        return unique_titles[:10]  # Limit to top 10


def index_project(
    input_path: Path,
    output_dir: Path,
    project_id: str,
    thumb_dpi: int = 100,
) -> ProjectIndex:
    """
    Convenience function to index a project.

    Args:
        input_path: Path to folder or PDF
        output_dir: Output directory
        project_id: Project identifier
        thumb_dpi: Thumbnail DPI

    Returns:
        ProjectIndex
    """
    indexer = ProjectIndexer(thumb_dpi=thumb_dpi)
    return indexer.index_project(input_path, output_dir, project_id)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")
        project_id = sys.argv[3] if len(sys.argv) > 3 else input_path.stem

        index = index_project(input_path, output_dir, project_id)

        print(f"\nProject Index: {index.project_id}")
        print(f"Total files: {index.total_files}")
        print(f"Total pages: {index.total_pages}")
        print(f"Cache hits: {index.cache_hits}")
        print(f"Cache misses: {index.cache_misses}")
    else:
        print("Usage: python indexer.py <input_path> [output_dir] [project_id]")
