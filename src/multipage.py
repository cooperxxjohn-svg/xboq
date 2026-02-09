"""
Multi-Page PDF Ingestion with Sheet Classification.

Processes multi-page PDFs with India architect drawing sets:
- Classifies each page (floor_plan, schedule_table, section, etc.)
- Only runs room detection on floor_plan pages
- Generates per-project manifest with processing decisions
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np
import cv2
import re

logger = logging.getLogger(__name__)


class SheetType(Enum):
    """Classification of drawing sheet types."""
    FLOOR_PLAN = "floor_plan"
    SCHEDULE_TABLE = "schedule_table"
    SECTION_ELEVATION = "section_elevation"
    DETAIL = "detail"
    COVER_TITLE = "cover_title"
    SITE_PLAN = "site_plan"
    STRUCTURAL = "structural"
    MEP = "mep"  # Mechanical/Electrical/Plumbing
    UNKNOWN = "unknown"


@dataclass
class SheetClassification:
    """Result of classifying a single sheet."""
    page_number: int
    sheet_type: SheetType
    confidence: float
    should_process_rooms: bool  # Whether to run room detection
    reason: str  # Why this classification
    title_block: Dict[str, str] = field(default_factory=dict)
    detected_signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PageData:
    """Data for a single page."""
    page_number: int
    image: np.ndarray
    vector_texts: List[Any] = field(default_factory=list)
    vector_lines: List[Any] = field(default_factory=list)
    width_pts: float = 0.0
    height_pts: float = 0.0
    raw_text: str = ""


@dataclass
class ProjectManifest:
    """Manifest for a multi-page project."""
    project_id: str
    source_path: str
    total_pages: int
    pages_processed: List[Dict[str, Any]] = field(default_factory=list)
    pages_skipped: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source_path": self.source_path,
            "total_pages": self.total_pages,
            "pages_processed": self.pages_processed,
            "pages_skipped": self.pages_skipped,
            "summary": self.summary,
        }

    def save(self, output_path: Path) -> None:
        """Save manifest to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SheetClassifier:
    """
    Classifies drawing sheets using text signals and visual heuristics.
    Optimized for Indian architect drawing conventions.
    """

    # Title block keywords by sheet type
    TITLE_KEYWORDS = {
        SheetType.FLOOR_PLAN: [
            "floor plan", "ground floor", "first floor", "second floor",
            "typical floor", "terrace plan", "basement plan",
            "layout plan", "unit plan", "flat plan",
            "stilt plan", "podium plan", "refuge area",
            # India-specific
            "gf plan", "ff plan", "sf plan", "tf plan",
        ],
        SheetType.SCHEDULE_TABLE: [
            "schedule", "bbs", "bar bending", "door schedule",
            "window schedule", "finish schedule", "room schedule",
            "area statement", "specifications",
        ],
        SheetType.SECTION_ELEVATION: [
            "section", "elevation", "cross section", "longitudinal section",
            "front elevation", "side elevation", "rear elevation",
            "sectional elevation",
        ],
        SheetType.DETAIL: [
            "detail", "details", "typical detail", "construction detail",
            "joinery detail", "staircase detail", "toilet detail",
            "kitchen detail", "railing detail",
        ],
        SheetType.COVER_TITLE: [
            "cover", "title sheet", "index", "drawing list",
            "project brief", "location plan",
        ],
        SheetType.SITE_PLAN: [
            "site plan", "key plan", "location plan", "plot plan",
            "master plan", "landscape plan",
        ],
        SheetType.STRUCTURAL: [
            "column layout", "beam layout", "foundation", "footing",
            "plinth beam", "lintel", "slab reinforcement",
            "structural plan", "rcc layout", "framing plan",
        ],
        SheetType.MEP: [
            "electrical", "plumbing", "hvac", "drainage",
            "water supply", "sanitary", "fire fighting",
            "electrical layout", "switch layout",
        ],
    }

    # Sheet number prefixes
    SHEET_PREFIXES = {
        SheetType.FLOOR_PLAN: ["A-", "AR-", "AP-", "ARCH-"],
        SheetType.SCHEDULE_TABLE: ["SCH-", "SC-", "AS-"],
        SheetType.SECTION_ELEVATION: ["S-", "SEC-", "E-", "EL-"],
        SheetType.DETAIL: ["D-", "DT-", "AD-"],
        SheetType.COVER_TITLE: ["C-", "00-", "T-"],
        SheetType.SITE_PLAN: ["SP-", "SITE-", "L-"],
        SheetType.STRUCTURAL: ["ST-", "STR-", "COL-", "FND-"],
        SheetType.MEP: ["EL-", "PL-", "ME-", "FP-"],
    }

    def __init__(self):
        self.debug_info = []

    def classify_page(self, page_data: PageData) -> SheetClassification:
        """
        Classify a single page.

        Args:
            page_data: Page data with image and text

        Returns:
            SheetClassification
        """
        signals = []
        scores = {st: 0.0 for st in SheetType}

        # 1. Extract and analyze text
        all_text = self._extract_all_text(page_data)
        text_lower = all_text.lower()

        # 2. Detect title block info
        title_block = self._extract_title_block(page_data.vector_texts, text_lower)

        # 3. Check sheet number prefix
        sheet_num = title_block.get("sheet_number", "")
        for stype, prefixes in self.SHEET_PREFIXES.items():
            for prefix in prefixes:
                if sheet_num.upper().startswith(prefix.upper().rstrip("-")):
                    scores[stype] += 0.4
                    signals.append(f"sheet_prefix:{prefix}")

        # 4. Check title keywords
        title = title_block.get("title", "").lower()
        for stype, keywords in self.TITLE_KEYWORDS.items():
            for kw in keywords:
                if kw in title:
                    scores[stype] += 0.5
                    signals.append(f"title:{kw}")
                elif kw in text_lower:
                    scores[stype] += 0.2
                    signals.append(f"text:{kw}")

        # 5. Visual heuristics
        visual_scores, visual_signals = self._analyze_visual_features(page_data.image)
        for stype, vscore in visual_scores.items():
            scores[stype] += vscore
        signals.extend(visual_signals)

        # 6. Table detection (for schedules)
        has_table, table_score = self._detect_table(page_data.image, text_lower)
        if has_table:
            scores[SheetType.SCHEDULE_TABLE] += table_score
            signals.append(f"has_table:{table_score:.2f}")

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Default to unknown if score too low
        if best_score < 0.25:
            best_type = SheetType.UNKNOWN

        # Calculate confidence
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.95, 0.5 + margin)
        else:
            confidence = max(0.3, min(0.95, best_score))

        # Determine if we should process rooms
        should_process = best_type == SheetType.FLOOR_PLAN

        reason = self._build_reason(best_type, signals, best_score)

        return SheetClassification(
            page_number=page_data.page_number,
            sheet_type=best_type,
            confidence=round(confidence, 2),
            should_process_rooms=should_process,
            reason=reason,
            title_block=title_block,
            detected_signals=signals,
        )

    def _extract_all_text(self, page_data: PageData) -> str:
        """Extract all text from page."""
        texts = [page_data.raw_text]

        for vt in page_data.vector_texts:
            if hasattr(vt, 'text'):
                texts.append(vt.text)
            elif isinstance(vt, dict):
                texts.append(vt.get('text', ''))

        return " ".join(texts)

    def _extract_title_block(
        self,
        vector_texts: List[Any],
        text_lower: str
    ) -> Dict[str, str]:
        """Extract title block information."""
        title_block = {
            "title": "",
            "sheet_number": "",
            "scale": "",
            "project_name": "",
        }

        # Sheet number patterns
        sheet_patterns = [
            r'(?:sheet|dwg|drawing)\s*(?:no\.?|#)?\s*[:\-]?\s*([A-Z]{1,3}[\-/]?\d{1,4})',
            r'\b([ASDEF][\-/]\d{2,4})\b',
            r'(?:^|\s)([A-Z]{1,3}[\-/]\d{1,4})(?:\s|$)',
        ]

        for pattern in sheet_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                title_block["sheet_number"] = match.group(1).upper()
                break

        # Title extraction - look for large text or common patterns
        title_patterns = [
            r'(?:title|drawing)\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'((?:ground|first|second|typical|terrace)\s+floor\s+plan)',
            r'((?:floor|site|layout|key)\s+plan)',
            r'((?:column|beam|foundation|footing)\s+(?:layout|plan))',
            r'(section\s+[A-Z]{1,2}[\-\s]?[A-Z]?)',
            r'((?:front|side|rear)\s+elevation)',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                title_block["title"] = match.group(1).strip().title()
                break

        # Scale extraction
        scale_patterns = [
            r'scale\s*[:\-]?\s*(1\s*[:/]\s*\d+)',
            r'(1\s*[:/]\s*\d+)\s*(?:scale)?',
        ]

        for pattern in scale_patterns:
            match = re.search(pattern, text_lower)
            if match:
                title_block["scale"] = match.group(1).replace(" ", "")
                break

        return title_block

    def _analyze_visual_features(
        self,
        image: np.ndarray
    ) -> Tuple[Dict[SheetType, float], List[str]]:
        """Analyze visual features for classification."""
        scores = {st: 0.0 for st in SheetType}
        signals = []

        if image.size == 0:
            return scores, signals

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Line detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            # Count horizontal vs vertical lines
            h_lines = 0
            v_lines = 0
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

            # High line density with good H/V balance = floor plan
            if line_density > 30 and 0.3 < h_lines / max(1, v_lines) < 3.0:
                scores[SheetType.FLOOR_PLAN] += 0.2
                signals.append(f"line_density:{line_density:.1f}")

            # Grid-like pattern (many H and V lines) might be schedule
            if h_lines > 20 and v_lines > 20:
                hv_ratio = min(h_lines, v_lines) / max(h_lines, v_lines)
                if hv_ratio > 0.5:
                    scores[SheetType.SCHEDULE_TABLE] += 0.15
                    signals.append(f"grid_pattern:{hv_ratio:.2f}")

        # Contour analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Many rectangular regions might indicate floor plan rooms
            rect_count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    _, _, cnt_w, cnt_h = cv2.boundingRect(cnt)
                    aspect = max(cnt_w, cnt_h) / max(1, min(cnt_w, cnt_h))
                    if aspect < 5:  # Reasonably rectangular
                        rect_count += 1

            if 5 < rect_count < 100:
                scores[SheetType.FLOOR_PLAN] += 0.1
                signals.append(f"rect_regions:{rect_count}")

        return scores, signals

    def _detect_table(
        self,
        image: np.ndarray,
        text_lower: str
    ) -> Tuple[bool, float]:
        """Detect if page contains a table/schedule."""
        score = 0.0

        # Text-based detection
        table_keywords = ['schedule', 'mark', 'size', 'nos', 'qty', 'description', 'remark']
        kw_count = sum(1 for kw in table_keywords if kw in text_lower)
        if kw_count >= 3:
            score += 0.3

        # Visual detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)

        # Long horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        h_sum = np.sum(h_lines > 0)

        # Long vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        v_sum = np.sum(v_lines > 0)

        # Both H and V indicates table structure
        area = image.shape[0] * image.shape[1]
        if h_sum > area * 0.001 and v_sum > area * 0.001:
            score += 0.4

        return score > 0.3, score

    def _build_reason(
        self,
        sheet_type: SheetType,
        signals: List[str],
        score: float
    ) -> str:
        """Build human-readable reason for classification."""
        if not signals:
            return f"Default classification (score: {score:.2f})"

        signal_summary = ", ".join(signals[:3])
        if len(signals) > 3:
            signal_summary += f" (+{len(signals)-3} more)"

        return f"{sheet_type.value} based on: {signal_summary} (score: {score:.2f})"


class MultiPageIngester:
    """
    Ingests multi-page PDFs and classifies each page.
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.classifier = SheetClassifier()
        self._fitz = None

    def _load_fitz(self):
        """Lazy load PyMuPDF."""
        if self._fitz is None:
            import fitz
            self._fitz = fitz
        return self._fitz

    def ingest_project(
        self,
        file_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[Tuple[PageData, SheetClassification]], ProjectManifest]:
        """
        Ingest a multi-page PDF project.

        Args:
            file_path: Path to PDF
            output_dir: Optional output directory for debug images

        Returns:
            Tuple of (list of (page_data, classification), manifest)
        """
        file_path = Path(file_path)
        project_id = file_path.stem

        logger.info(f"Ingesting multi-page project: {file_path}")

        fitz = self._load_fitz()
        doc = fitz.open(file_path)
        total_pages = len(doc)

        pages_with_class = []
        manifest = ProjectManifest(
            project_id=project_id,
            source_path=str(file_path),
            total_pages=total_pages,
        )

        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}")

            page = doc[page_num]
            page_data = self._extract_page_data(page, page_num)
            classification = self.classifier.classify_page(page_data)

            pages_with_class.append((page_data, classification))

            # Update manifest
            page_entry = {
                "page_number": page_num,
                "sheet_type": classification.sheet_type.value,
                "confidence": classification.confidence,
                "title": classification.title_block.get("title", ""),
                "sheet_number": classification.title_block.get("sheet_number", ""),
                "reason": classification.reason,
            }

            if classification.should_process_rooms:
                manifest.pages_processed.append(page_entry)
            else:
                page_entry["skip_reason"] = f"Sheet type: {classification.sheet_type.value}"
                manifest.pages_skipped.append(page_entry)

            logger.info(
                f"  Page {page_num + 1}: {classification.sheet_type.value} "
                f"(conf: {classification.confidence:.0%}) "
                f"{'-> PROCESS' if classification.should_process_rooms else '-> SKIP'}"
            )

        doc.close()

        # Build summary
        type_counts = {}
        for _, cls in pages_with_class:
            type_counts[cls.sheet_type.value] = type_counts.get(cls.sheet_type.value, 0) + 1

        manifest.summary = {
            "total_pages": total_pages,
            "pages_to_process": len(manifest.pages_processed),
            "pages_skipped": len(manifest.pages_skipped),
            "sheet_types": type_counts,
        }

        # Save debug images if output_dir provided
        if output_dir:
            self._save_debug_images(pages_with_class, output_dir)

        return pages_with_class, manifest

    def _extract_page_data(self, page, page_num: int) -> PageData:
        """Extract data from a single PDF page."""
        fitz = self._fitz

        # Get page dimensions
        rect = page.rect

        # Scale for coordinate transform
        scale = self.dpi / 72.0

        # Extract text
        raw_text = page.get_text()

        # Extract text with positions
        vector_texts = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                bbox = span.get("bbox", (0, 0, 0, 0))
                                vector_texts.append({
                                    'text': text,
                                    'bbox': tuple(b * scale for b in bbox),
                                    'size': span.get("size", 10) * scale,
                                })
        except Exception as e:
            logger.debug(f"Text extraction error: {e}")

        # Extract vector lines
        vector_lines = []
        try:
            drawings = page.get_drawings()
            for drawing in drawings:
                for item in drawing.get("items", []):
                    if item[0] == "l":  # Line
                        vector_lines.append({
                            'start': (item[1].x * scale, item[1].y * scale),
                            'end': (item[2].x * scale, item[2].y * scale),
                        })
        except Exception as e:
            logger.debug(f"Vector extraction error: {e}")

        # Render to image
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return PageData(
            page_number=page_num,
            image=img,
            vector_texts=vector_texts,
            vector_lines=vector_lines,
            width_pts=rect.width,
            height_pts=rect.height,
            raw_text=raw_text,
        )

    def _save_debug_images(
        self,
        pages_with_class: List[Tuple[PageData, SheetClassification]],
        output_dir: Path
    ) -> None:
        """Save debug images showing classification."""
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for page_data, classification in pages_with_class:
            # Create annotated thumbnail
            img = page_data.image.copy()

            # Resize for thumbnail
            max_dim = 800
            h, w = img.shape[:2]
            scale = min(max_dim / w, max_dim / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumb = cv2.resize(img, (new_w, new_h))

            # Add classification label
            label = f"P{page_data.page_number + 1}: {classification.sheet_type.value}"
            label2 = f"Conf: {classification.confidence:.0%}"

            # Background rectangle for text
            cv2.rectangle(thumb, (0, 0), (new_w, 60), (255, 255, 255), -1)

            # Draw text
            cv2.putText(thumb, label, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(thumb, label2, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            # Color code by type
            color = (0, 255, 0) if classification.should_process_rooms else (0, 0, 255)
            cv2.rectangle(thumb, (0, 0), (new_w - 1, new_h - 1), color, 3)

            # Save
            out_path = debug_dir / f"page_{page_data.page_number + 1:03d}_classified.jpg"
            cv2.imwrite(str(out_path), thumb)

        logger.info(f"Saved classification debug images to {debug_dir}")


def ingest_multipage_project(
    file_path: Path,
    output_dir: Optional[Path] = None,
    dpi: int = 300
) -> Tuple[List[Tuple[PageData, SheetClassification]], ProjectManifest]:
    """
    Convenience function to ingest a multi-page project.

    Args:
        file_path: Path to PDF
        output_dir: Optional output directory
        dpi: Rendering DPI

    Returns:
        Tuple of (pages with classifications, manifest)
    """
    ingester = MultiPageIngester(dpi=dpi)
    return ingester.ingest_project(file_path, output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")

        pages, manifest = ingest_multipage_project(file_path, output_dir)

        print(f"\nProject: {manifest.project_id}")
        print(f"Total pages: {manifest.total_pages}")
        print(f"Pages to process: {len(manifest.pages_processed)}")
        print(f"Pages skipped: {len(manifest.pages_skipped)}")

        print("\nSheet type breakdown:")
        for stype, count in manifest.summary.get("sheet_types", {}).items():
            print(f"  {stype}: {count}")

        # Save manifest
        manifest_path = output_dir / manifest.project_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(manifest_path)
        print(f"\nManifest saved to: {manifest_path}")
    else:
        print("Usage: python multipage.py <file.pdf> [output_dir]")
