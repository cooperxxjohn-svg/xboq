"""
Multi-Region Text Extractor
Extracts text from all regions of a drawing, including:
- General Notes
- Legend
- Schedules
- Typical Details
- Specifications
- Title Block
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TextRegion:
    """A detected text region with metadata."""
    region_type: str  # notes, legend, schedule, detail, spec, title_block, other
    header: str
    content: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float = 0.5
    page_location: str = ""  # top, bottom, left, right, center


@dataclass
class ExtractedText:
    """Complete extracted text from a drawing."""
    notes_text: str = ""
    schedule_text: str = ""
    legend_text: str = ""
    title_block_text: str = ""
    specification_text: str = ""
    all_text: str = ""
    regions: List[TextRegion] = field(default_factory=list)
    detected_headers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'notes_text': self.notes_text,
            'schedule_text': self.schedule_text,
            'legend_text': self.legend_text,
            'title_block_text': self.title_block_text,
            'specification_text': self.specification_text,
            'all_text': self.all_text,
            'regions': [
                {
                    'type': r.region_type,
                    'header': r.header,
                    'content': r.content[:200] + '...' if len(r.content) > 200 else r.content,
                    'bbox': r.bbox,
                    'location': r.page_location
                }
                for r in self.regions
            ],
            'detected_headers': self.detected_headers
        }


# =============================================================================
# HEADER PATTERNS
# =============================================================================

HEADER_PATTERNS = {
    "notes": [
        r'\b(GENERAL\s+NOTES?)\b',
        r'\b(NOTES?)\s*:?\s*$',
        r'\b(STRUCTURAL\s+NOTES?)\b',
        r'\b(DESIGN\s+NOTES?)\b',
        r'\b(IMPORTANT\s+NOTES?)\b',
        r'\b(SPECIFICATIONS?)\s*$',
        r'\b(N\.B\.)\b',
    ],
    "legend": [
        r'\b(LEGEND)\b',
        r'\b(KEY\s+PLAN)\b',
        r'\b(SYMBOL\s+LEGEND)\b',
        r'\b(ABBREVIATIONS?)\b',
        r'\b(CONVENTIONS?)\b',
    ],
    "schedule": [
        r'\b(SCHEDULE)\b',
        r'\b(COLUMN\s+SCHEDULE)\b',
        r'\b(FOOTING\s+SCHEDULE)\b',
        r'\b(BEAM\s+SCHEDULE)\b',
        r'\b(REINFORCEMENT\s+SCHEDULE)\b',
        r'\b(BAR\s+BENDING\s+SCHEDULE)\b',
        r'\b(BBS)\b',
        r'\b(TABLE)\b',
    ],
    "detail": [
        r'\b(TYPICAL\s+DETAIL)\b',
        r'\b(DETAIL\s+(?:AT|OF|FOR))\b',
        r'\b(SECTION\s+[A-Z]-[A-Z])\b',
        r'\b(ENLARGED\s+DETAIL)\b',
        r'\b(STANDARD\s+DETAIL)\b',
    ],
    "title_block": [
        r'\b(PROJECT)\s*:',
        r'\b(DRAWING\s+TITLE)\b',
        r'\b(SHEET\s+NO)\b',
        r'\b(REVISION)\b',
        r'\b(DATE)\s*:',
        r'\b(SCALE)\s*:',
        r'\b(DRAWN\s+BY)\b',
        r'\b(CHECKED\s+BY)\b',
    ]
}


# =============================================================================
# TEXT EXTRACTION FUNCTIONS
# =============================================================================

def classify_text_block(
    text: str,
    bbox: Tuple[float, float, float, float],
    page_width: float,
    page_height: float
) -> Tuple[str, str, float]:
    """
    Classify a text block into a region type.

    Args:
        text: Text content
        bbox: Bounding box (x1, y1, x2, y2)
        page_width: Page width
        page_height: Page height

    Returns:
        (region_type, page_location, confidence)
    """
    text_upper = text.upper().strip()
    x1, y1, x2, y2 = bbox

    # Determine page location
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < page_width * 0.2:
        location = "left"
    elif cx > page_width * 0.8:
        location = "right"
    elif cy < page_height * 0.15:
        location = "top"
    elif cy > page_height * 0.85:
        location = "bottom"
    else:
        location = "center"

    # Check header patterns
    for region_type, patterns in HEADER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_upper):
                confidence = 0.90 if text_upper.startswith(re.sub(r'\\b|\^|\$', '', pattern.split('(')[1].split(')')[0])) else 0.75
                return region_type, location, confidence

    # Heuristics based on location
    if location == "bottom" and len(text) > 20:
        # Likely title block or notes
        if any(kw in text_upper for kw in ['PROJECT', 'SHEET', 'DATE', 'SCALE', 'DRAWN']):
            return "title_block", location, 0.70
        elif any(kw in text_upper for kw in ['NOTE', 'N.B.', 'ALL']):
            return "notes", location, 0.65

    if location == "right" and len(text) > 50:
        # Often notes or schedules on right side
        if any(kw in text_upper for kw in ['SCHEDULE', 'TABLE', 'TYPE', 'SIZE']):
            return "schedule", location, 0.65
        return "notes", location, 0.50

    return "other", location, 0.30


def extract_notes_and_schedules(
    text_blocks: List[Dict[str, Any]],
    page_width: float = 1000,
    page_height: float = 700,
    full_text: str = ""
) -> ExtractedText:
    """
    Extract notes and schedules from text blocks.

    Args:
        text_blocks: List of text blocks with 'text' and 'bbox'
        page_width: Page width in pixels
        page_height: Page height in pixels
        full_text: Full page text (for fallback)

    Returns:
        ExtractedText with categorized content
    """
    result = ExtractedText()
    result.all_text = full_text

    notes_parts = []
    schedule_parts = []
    legend_parts = []
    title_parts = []
    spec_parts = []

    # Process each text block
    for block in text_blocks:
        text = block.get('text', '')
        if not text or len(text.strip()) < 3:
            continue

        bbox = block.get('bbox', (0, 0, 0, 0))

        # Classify the block
        region_type, location, confidence = classify_text_block(
            text, bbox, page_width, page_height
        )

        region = TextRegion(
            region_type=region_type,
            header=text[:50] if region_type != "other" else "",
            content=text,
            bbox=bbox,
            confidence=confidence,
            page_location=location
        )
        result.regions.append(region)

        # Accumulate by type
        if region_type == "notes":
            notes_parts.append(text)
        elif region_type == "schedule":
            schedule_parts.append(text)
        elif region_type == "legend":
            legend_parts.append(text)
        elif region_type == "title_block":
            title_parts.append(text)
        elif region_type in ("detail", "spec"):
            spec_parts.append(text)

        # Track headers
        if region_type != "other" and confidence >= 0.65:
            result.detected_headers.append({
                'header': text[:100],
                'type': region_type,
                'bbox': bbox,
                'confidence': confidence
            })

    # Combine parts
    result.notes_text = '\n'.join(notes_parts)
    result.schedule_text = '\n'.join(schedule_parts)
    result.legend_text = '\n'.join(legend_parts)
    result.title_block_text = '\n'.join(title_parts)
    result.specification_text = '\n'.join(spec_parts)

    # Fallback: extract from full text using patterns
    if not result.notes_text and full_text:
        result.notes_text = extract_section_from_text(full_text, "notes")

    if not result.schedule_text and full_text:
        result.schedule_text = extract_section_from_text(full_text, "schedule")

    logger.info(f"Extracted: {len(notes_parts)} notes, {len(schedule_parts)} schedules, "
               f"{len(result.detected_headers)} headers")

    return result


def extract_section_from_text(full_text: str, section_type: str) -> str:
    """
    Extract a section from full text using regex.

    Args:
        full_text: Complete text
        section_type: Type of section to extract

    Returns:
        Extracted section text
    """
    patterns = HEADER_PATTERNS.get(section_type, [])
    text_upper = full_text.upper()

    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            # Get text after the header until next section or end
            start = match.end()
            # Look for next header
            next_match = None
            for p in sum(HEADER_PATTERNS.values(), []):
                m = re.search(p, text_upper[start:])
                if m:
                    if next_match is None or m.start() < next_match.start():
                        next_match = m

            if next_match:
                end = start + next_match.start()
            else:
                end = min(start + 2000, len(full_text))  # Limit length

            return full_text[start:end].strip()

    return ""


def extract_from_pdf_page(page) -> ExtractedText:
    """
    Extract text from a PyMuPDF page object.

    Args:
        page: fitz.Page object

    Returns:
        ExtractedText
    """
    # Get page dimensions
    rect = page.rect
    page_width = rect.width
    page_height = rect.height

    # Get full text
    full_text = page.get_text()

    # Get text blocks with positions
    text_blocks = []
    blocks = page.get_text('dict')

    for block in blocks.get('blocks', []):
        if block.get('type') == 0:  # Text block
            for line in block.get('lines', []):
                line_text = ''
                line_bbox = None

                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if text:
                        line_text += text + ' '
                        span_bbox = span.get('bbox', (0, 0, 0, 0))
                        if line_bbox is None:
                            line_bbox = list(span_bbox)
                        else:
                            line_bbox[0] = min(line_bbox[0], span_bbox[0])
                            line_bbox[1] = min(line_bbox[1], span_bbox[1])
                            line_bbox[2] = max(line_bbox[2], span_bbox[2])
                            line_bbox[3] = max(line_bbox[3], span_bbox[3])

                if line_text.strip():
                    text_blocks.append({
                        'text': line_text.strip(),
                        'bbox': tuple(line_bbox) if line_bbox else (0, 0, 0, 0),
                        'size': line.get('spans', [{}])[0].get('size', 10)
                    })

    return extract_notes_and_schedules(
        text_blocks,
        page_width=page_width,
        page_height=page_height,
        full_text=full_text
    )


def find_region_by_header(
    text_blocks: List[Dict[str, Any]],
    header_keywords: List[str],
    max_lines: int = 50
) -> str:
    """
    Find text region by header keywords and return content below.

    Args:
        text_blocks: List of text blocks sorted by position
        header_keywords: Keywords to match header
        max_lines: Maximum lines to capture after header

    Returns:
        Content text
    """
    content_lines = []
    capturing = False
    lines_captured = 0

    for block in text_blocks:
        text = block.get('text', '')
        text_upper = text.upper()

        # Check for header
        if any(kw.upper() in text_upper for kw in header_keywords):
            capturing = True
            continue

        # Check for end of section (next header)
        if capturing:
            is_next_header = any(
                re.search(pattern, text_upper)
                for patterns in HEADER_PATTERNS.values()
                for pattern in patterns
            )

            if is_next_header:
                break

            content_lines.append(text)
            lines_captured += 1

            if lines_captured >= max_lines:
                break

    return '\n'.join(content_lines)


def merge_extracted_text(*extracts: ExtractedText) -> ExtractedText:
    """
    Merge multiple ExtractedText objects.

    Args:
        extracts: ExtractedText objects to merge

    Returns:
        Merged ExtractedText
    """
    result = ExtractedText()

    notes_parts = []
    schedule_parts = []
    legend_parts = []
    title_parts = []
    spec_parts = []
    all_parts = []

    for ext in extracts:
        if ext.notes_text:
            notes_parts.append(ext.notes_text)
        if ext.schedule_text:
            schedule_parts.append(ext.schedule_text)
        if ext.legend_text:
            legend_parts.append(ext.legend_text)
        if ext.title_block_text:
            title_parts.append(ext.title_block_text)
        if ext.specification_text:
            spec_parts.append(ext.specification_text)
        if ext.all_text:
            all_parts.append(ext.all_text)

        result.regions.extend(ext.regions)
        result.detected_headers.extend(ext.detected_headers)

    result.notes_text = '\n\n'.join(notes_parts)
    result.schedule_text = '\n\n'.join(schedule_parts)
    result.legend_text = '\n\n'.join(legend_parts)
    result.title_block_text = '\n\n'.join(title_parts)
    result.specification_text = '\n\n'.join(spec_parts)
    result.all_text = '\n\n'.join(all_parts)

    return result
