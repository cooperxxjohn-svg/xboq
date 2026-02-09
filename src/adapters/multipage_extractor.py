"""
Multipage Extractor - Extracts features from ALL candidate pages.

This extractor:
1. Processes all pages marked as candidates by the router
2. Extracts rooms, openings, and dimensions from each page
3. Detects and infers scale from dimension text
4. Saves thumbnails for proof
5. Generates overlays showing what was detected
"""

import re
import json
import csv
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRoom:
    """A room extracted from a drawing page."""
    id: str
    label: str
    room_type: str
    area_sqm: float
    perimeter_m: float
    confidence: float
    source_file: str
    source_page: int
    bbox: List[float]
    method: str = "text_label"


@dataclass
class ExtractedOpening:
    """An opening (door/window) extracted from a drawing page."""
    id: str
    tag: str
    opening_type: str
    description: str
    width_m: float
    height_m: float
    confidence: float
    source_file: str
    source_page: int
    method: str = "text_label"


@dataclass
class ScaleInfo:
    """Scale information detected or inferred."""
    scale_ratio: float  # e.g., 100 for 1:100
    mm_per_px: float
    method: str  # "text_note", "dimension_inference", "manual_override"
    confidence: float
    source_page: int
    evidence: List[str] = field(default_factory=list)


# Room type keywords
ROOM_KEYWORDS = {
    "bedroom": "bedroom", "bed room": "bedroom", "master": "bedroom",
    "bath": "toilet", "bathroom": "toilet", "toilet": "toilet", "wc": "toilet",
    "kitchen": "kitchen", "pantry": "kitchen",
    "living": "living", "lounge": "living", "hall": "living", "drawing room": "living",
    "dining": "dining",
    "balcony": "balcony", "terrace": "balcony",
    "foyer": "foyer", "entry": "foyer", "lobby": "foyer",
    "laundry": "utility", "utility": "utility", "service": "utility",
    "store": "store", "storage": "store",
    "porch": "porch", "verandah": "porch",
    "garage": "garage", "parking": "garage",
    "office": "office", "study": "office", "work": "office",
    "pooja": "pooja", "prayer": "pooja",
    "stair": "circulation", "lift": "circulation", "corridor": "circulation",
}

# Opening patterns
OPENING_PATTERNS = {
    "door": [r"D[-\s]?\d+", r"DR[-\s]?\d+", r"DOOR\s*\d*", r"MAIN\s*DOOR", r"ENTRY"],
    "window": [r"W[-\s]?\d+", r"WIN[-\s]?\d+", r"WINDOW\s*\d*", r"VENTILATOR"],
}

# Dimension patterns (in mm)
DIMENSION_TEXT_PATTERNS = [
    r"(\d{3,5})\s*(?:mm)?",  # 3-5 digit number, optionally with mm
    r"(\d+)\s*[x×]\s*(\d+)",  # WxH format
]


class MultipageExtractor:
    """
    Extracts features from all candidate pages in a drawing set.
    """

    def __init__(self, scale_override: Optional[float] = None, mm_per_px: Optional[float] = None):
        """
        Args:
            scale_override: Manual scale ratio (e.g., 100 for 1:100)
            mm_per_px: Manual mm per pixel conversion
        """
        self.scale_override = scale_override
        self.mm_per_px_override = mm_per_px
        self.room_counter = 0
        self.opening_counter = 0

    def extract_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        source_file: str,
        scale_info: Optional[ScaleInfo] = None
    ) -> Tuple[List[ExtractedRoom], List[ExtractedOpening], List[str]]:
        """
        Extract rooms, openings, and dimensions from a single page.

        Args:
            page: PyMuPDF page object
            page_num: 0-indexed page number
            source_file: Source filename
            scale_info: Optional scale info for dimension conversion

        Returns:
            Tuple of (rooms, openings, dimensions_found)
        """
        text = page.get_text("text")
        text_lower = text.lower()

        rooms = []
        openings = []
        dimensions = []

        # Get text blocks with positions
        text_dict = page.get_text("dict", flags=11)
        blocks = text_dict.get("blocks", [])

        seen_rooms = set()
        seen_openings = set()

        for block in blocks:
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    span_lower = span_text.lower()
                    bbox = span.get("bbox", [0, 0, 0, 0])

                    # Check for room labels
                    for keyword, room_type in ROOM_KEYWORDS.items():
                        if keyword in span_lower and span_lower not in seen_rooms:
                            seen_rooms.add(span_lower)
                            self.room_counter += 1

                            # Estimate area from nearby dimension text (if we have scale)
                            area_sqm = 15.0  # Default
                            perimeter_m = 16.0

                            rooms.append(ExtractedRoom(
                                id=f"R{self.room_counter:03d}",
                                label=span_text.title(),
                                room_type=room_type,
                                area_sqm=area_sqm,
                                perimeter_m=perimeter_m,
                                confidence=0.7,
                                source_file=source_file,
                                source_page=page_num,
                                bbox=list(bbox),
                            ))
                            break

                    # Check for opening labels
                    for opening_type, patterns in OPENING_PATTERNS.items():
                        for pattern in patterns:
                            if re.match(pattern, span_text.upper()) and span_text.upper() not in seen_openings:
                                seen_openings.add(span_text.upper())
                                self.opening_counter += 1

                                # Default dimensions
                                width_m = 0.9 if opening_type == "door" else 1.2
                                height_m = 2.1 if opening_type == "door" else 1.5

                                openings.append(ExtractedOpening(
                                    id=f"O{self.opening_counter:03d}",
                                    tag=span_text.upper().replace(" ", ""),
                                    opening_type=opening_type,
                                    description=f"{opening_type.title()} {span_text.upper()}",
                                    width_m=width_m,
                                    height_m=height_m,
                                    confidence=0.65,
                                    source_file=source_file,
                                    source_page=page_num,
                                ))
                                break

                    # Check for dimension text
                    for pattern in DIMENSION_TEXT_PATTERNS:
                        matches = re.findall(pattern, span_text)
                        for m in matches:
                            if isinstance(m, tuple):
                                dims = [int(d) for d in m if d.isdigit()]
                            else:
                                dims = [int(m)] if m.isdigit() else []
                            for d in dims:
                                if 100 <= d <= 50000:  # Reasonable mm range
                                    dimensions.append(str(d))

        return rooms, openings, dimensions

    def detect_scale_from_page(self, page: fitz.Page, page_num: int) -> Optional[ScaleInfo]:
        """
        Try to detect scale from page text.

        Args:
            page: PyMuPDF page object
            page_num: 0-indexed page number

        Returns:
            ScaleInfo if scale detected, None otherwise
        """
        text = page.get_text("text")

        # Scale notation patterns
        scale_patterns = [
            r"scale\s*[:\-]?\s*1\s*[:\-]\s*(\d+)",
            r"1\s*[:\-]\s*(\d+)\s*scale",
            r"@\s*1\s*[:\-]\s*(\d+)",
            r"sc\.?\s*1\s*[:\-]\s*(\d+)",
        ]

        for pattern in scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    scale_ratio = int(match.group(1))
                    if 10 <= scale_ratio <= 500:  # Reasonable scale range
                        return ScaleInfo(
                            scale_ratio=scale_ratio,
                            mm_per_px=scale_ratio / 72 * 25.4,  # Approximate
                            method="text_note",
                            confidence=0.9,
                            source_page=page_num,
                            evidence=[match.group(0)],
                        )
                except ValueError:
                    continue

        return None

    def infer_scale_from_dimensions(
        self,
        page: fitz.Page,
        page_num: int,
        dimensions: List[str]
    ) -> Optional[ScaleInfo]:
        """
        Try to infer scale by matching dimension text to line lengths.

        This looks for:
        1. Dimension lines (horizontal/vertical lines with text nearby)
        2. Compares annotated dimension (e.g., "3000") to pixel length
        3. Infers mm_per_px from multiple samples

        Args:
            page: PyMuPDF page object
            page_num: 0-indexed page number
            dimensions: List of dimension strings found on page

        Returns:
            ScaleInfo if scale inferred, None otherwise
        """
        if len(dimensions) < 3:
            return None  # Need at least 3 samples

        # Get vector drawings
        try:
            drawings = page.get_drawings()
        except Exception:
            return None

        # Find horizontal/vertical lines
        h_lines = []
        v_lines = []

        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":  # Line
                    p1, p2 = item[1], item[2]
                    dx = abs(p2[0] - p1[0])
                    dy = abs(p2[1] - p1[1])
                    length = (dx**2 + dy**2) ** 0.5

                    if length > 20:  # Minimum length
                        if dy < 5:  # Horizontal
                            h_lines.append((p1, p2, length))
                        elif dx < 5:  # Vertical
                            v_lines.append((p1, p2, length))

        if len(h_lines) + len(v_lines) < 3:
            return None

        # Try to match dimensions to lines
        # This is heuristic: assume common dimensions map to common line lengths
        dim_values = sorted(set(int(d) for d in dimensions if d.isdigit()), reverse=True)
        line_lengths = sorted(set(l[2] for l in h_lines + v_lines), reverse=True)

        if len(dim_values) < 2 or len(line_lengths) < 2:
            return None

        # Compute potential scale ratios
        ratios = []
        for dim in dim_values[:5]:  # Top 5 dimensions
            for length in line_lengths[:10]:  # Top 10 line lengths
                if length > 10:
                    ratio = dim / length  # mm per px
                    ratios.append(ratio)

        if not ratios:
            return None

        # Find most common ratio (cluster)
        from collections import Counter
        rounded_ratios = [round(r, 1) for r in ratios]
        most_common = Counter(rounded_ratios).most_common(1)

        if most_common:
            mm_per_px = most_common[0][0]
            # Convert to scale ratio (assuming 72 DPI rendering)
            scale_ratio = mm_per_px * 72 / 25.4

            if 10 <= scale_ratio <= 500:
                return ScaleInfo(
                    scale_ratio=round(scale_ratio),
                    mm_per_px=mm_per_px,
                    method="dimension_inference",
                    confidence=0.6,
                    source_page=page_num,
                    evidence=dim_values[:3],
                )

        return None

    def process_pdf(
        self,
        pdf_path: Path,
        candidate_pages: List[int],
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Process a PDF, extracting from all candidate pages.

        Args:
            pdf_path: Path to PDF file
            candidate_pages: List of page numbers to process (0-indexed)
            output_dir: Optional output directory for thumbnails/overlays

        Returns:
            Extraction result
        """
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        all_rooms = []
        all_openings = []
        all_dimensions = []
        scale_info = None
        page_results = []

        # First pass: detect scale
        if self.scale_override:
            scale_info = ScaleInfo(
                scale_ratio=self.scale_override,
                mm_per_px=self.scale_override / 72 * 25.4,
                method="manual_override",
                confidence=1.0,
                source_page=-1,
                evidence=["--scale_override"],
            )
        else:
            # Try to detect scale from pages
            for page_num in candidate_pages[:10]:  # Check first 10 candidates
                if page_num < total_pages:
                    page = doc[page_num]
                    detected = self.detect_scale_from_page(page, page_num)
                    if detected:
                        scale_info = detected
                        logger.info(f"Scale detected from page {page_num + 1}: 1:{detected.scale_ratio}")
                        break

        # Second pass: extract features
        for page_num in candidate_pages:
            if page_num >= total_pages:
                continue

            page = doc[page_num]
            rooms, openings, dims = self.extract_from_page(
                page, page_num, pdf_path.name, scale_info
            )

            all_rooms.extend(rooms)
            all_openings.extend(openings)
            all_dimensions.extend(dims)

            page_results.append({
                "page": page_num + 1,
                "rooms_found": len(rooms),
                "openings_found": len(openings),
                "dimensions_found": len(dims),
            })

            logger.info(
                f"Page {page_num + 1}: {len(rooms)} rooms, "
                f"{len(openings)} openings, {len(dims)} dimensions"
            )

        # If no scale yet, try inference
        if not scale_info and all_dimensions:
            for page_num in candidate_pages[:5]:
                if page_num < total_pages:
                    page = doc[page_num]
                    inferred = self.infer_scale_from_dimensions(page, page_num, all_dimensions)
                    if inferred:
                        scale_info = inferred
                        logger.info(f"Scale inferred from page {page_num + 1}: 1:{inferred.scale_ratio}")
                        break

        # Save thumbnails if output_dir provided
        if output_dir:
            self._save_thumbnails(doc, candidate_pages[:10], output_dir)
            self._save_overlays(doc, candidate_pages[:10], all_rooms, all_openings, scale_info, output_dir)

        doc.close()

        return {
            "pages_processed": len(candidate_pages),
            "total_rooms": len(all_rooms),
            "total_openings": len(all_openings),
            "total_dimensions": len(all_dimensions),
            "scale": {
                "detected": scale_info is not None,
                "ratio": scale_info.scale_ratio if scale_info else None,
                "method": scale_info.method if scale_info else None,
                "confidence": scale_info.confidence if scale_info else 0,
            } if scale_info else None,
            "rooms": [vars(r) for r in all_rooms],
            "openings": [vars(o) for o in all_openings],
            "page_results": page_results,
        }

    def _save_thumbnails(self, doc: fitz.Document, pages: List[int], output_dir: Path):
        """Save page thumbnails for proof."""
        thumbs_dir = output_dir / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        for page_num in pages:
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            # Render at 150 DPI
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            thumb_path = thumbs_dir / f"page_{page_num + 1:02d}.png"
            pix.save(str(thumb_path))

            logger.info(f"Saved thumbnail: {thumb_path}")

    def _save_overlays(
        self,
        doc: fitz.Document,
        pages: List[int],
        rooms: List[ExtractedRoom],
        openings: List[ExtractedOpening],
        scale_info: Optional[ScaleInfo],
        output_dir: Path
    ):
        """Save overlay images showing detected features."""
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("PIL not available, skipping overlays")
            return

        for page_num in pages:
            if page_num >= len(doc):
                continue

            page = doc[page_num]

            # Render page
            zoom = 100 / 72  # Lower res for overlay
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except Exception:
                font = ImageFont.load_default()

            # Draw rooms on this page
            page_rooms = [r for r in rooms if r.source_page == page_num]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

            for i, room in enumerate(page_rooms):
                if room.bbox:
                    color = colors[i % len(colors)]
                    # Scale bbox to overlay size
                    bbox = [b * zoom for b in room.bbox]
                    draw.rectangle(bbox, outline=color, width=2)
                    draw.text((bbox[0] + 2, bbox[1] + 2), room.label, fill=color, font=font)

            # Draw openings
            page_openings = [o for o in openings if o.source_page == page_num]
            for opening in page_openings:
                # Draw a small marker (we don't have bbox for openings)
                pass

            # Add scale info
            scale_text = "Scale: "
            if scale_info:
                scale_text += f"1:{scale_info.scale_ratio} ({scale_info.method})"
            else:
                scale_text += "UNKNOWN"

            draw.text((10, 10), f"Page {page_num + 1} | {scale_text}", fill='#333333', font=font)
            draw.text((10, 25), f"Rooms: {len(page_rooms)} | Openings: {len(page_openings)}", fill='#333333', font=font)

            # Save
            overlay_path = overlays_dir / f"overlay_page_{page_num + 1:02d}.png"
            img.save(str(overlay_path))


def extract_multipage_project(
    drawings_dir: Path,
    output_dir: Path,
    routing_result: Dict[str, Any],
    scale_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract features from all candidate pages in a project.

    Args:
        drawings_dir: Drawings directory
        output_dir: Output directory
        routing_result: Result from MultipageRouter
        scale_override: Manual scale ratio

    Returns:
        Extraction result
    """
    extractor = MultipageExtractor(scale_override=scale_override)

    # Get candidate pages from routing result
    scores = routing_result.get("scores", [])
    candidate_pages = [s.page_num for s in scores if s.is_candidate]

    if not candidate_pages:
        logger.warning("No candidate pages found - extracting from all pages")
        candidate_pages = list(range(routing_result.get("total_pages", 0)))

    # Find PDFs
    patterns = ["*.pdf", "*.PDF"]
    pdf_files = []
    for pattern in patterns:
        pdf_files.extend(Path(drawings_dir).glob(pattern))

    all_results = []
    combined_rooms = []
    combined_openings = []

    for pdf_path in sorted(pdf_files):
        result = extractor.process_pdf(pdf_path, candidate_pages, output_dir)
        all_results.append({
            "file": pdf_path.name,
            **result
        })
        combined_rooms.extend(result.get("rooms", []))
        combined_openings.extend(result.get("openings", []))

    # Write combined output
    _write_combined_output(output_dir, combined_rooms, combined_openings)

    return {
        "pages_processed": sum(r.get("pages_processed", 0) for r in all_results),
        "total_rooms": len(combined_rooms),
        "total_openings": len(combined_openings),
        "file_results": all_results,
    }


def _write_combined_output(output_dir: Path, rooms: List[Dict], openings: List[Dict]):
    """Write combined extraction output."""
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Write rooms JSON
    with open(combined_dir / "all_rooms.json", "w") as f:
        json.dump({"rooms": rooms}, f, indent=2)

    # Write openings JSON
    with open(combined_dir / "all_openings.json", "w") as f:
        json.dump({"openings": openings}, f, indent=2)

    # Write rooms CSV
    with open(combined_dir / "rooms.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "type", "area_sqm", "perimeter_m", "confidence", "source_file", "source_page"])
        for r in rooms:
            writer.writerow([
                r.get("id"), r.get("label"), r.get("room_type"),
                r.get("area_sqm"), r.get("perimeter_m"), r.get("confidence"),
                r.get("source_file"), r.get("source_page", 0) + 1
            ])

    # Write openings CSV
    with open(combined_dir / "openings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "tag", "type", "width_m", "height_m", "confidence", "source_file", "source_page"])
        for o in openings:
            writer.writerow([
                o.get("id"), o.get("tag"), o.get("opening_type"),
                o.get("width_m"), o.get("height_m"), o.get("confidence"),
                o.get("source_file"), o.get("source_page", 0) + 1
            ])
