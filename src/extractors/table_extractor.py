"""
Table Extractor Module
Extracts tables from PDF structural drawings using multiple methods.

Methods:
1. pdfplumber - Line/grid detection with bounding boxes
2. camelot - Lattice mode for well-defined tables
3. tabula - Fallback for stream-based tables

Specifically looks for:
- Column schedules (marks, sizes, reinforcement)
- Footing schedules (marks, dimensions)
- Beam schedules (marks, sizes, spans)
- Bar schedules (BBS - bar bending schedules)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


class TableType(Enum):
    """Classification of detected table types."""
    COLUMN_SCHEDULE = "column_schedule"
    FOOTING_SCHEDULE = "footing_schedule"
    BEAM_SCHEDULE = "beam_schedule"
    BAR_SCHEDULE = "bar_schedule"
    GENERAL_NOTES = "general_notes"
    MATERIAL_LIST = "material_list"
    UNKNOWN = "unknown"


@dataclass
class ExtractedTable:
    """Represents an extracted table with metadata."""
    table_id: str
    dataframe: pd.DataFrame
    table_type: TableType = TableType.UNKNOWN
    page_number: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    confidence: float = 0.5
    extraction_method: str = "unknown"  # pdfplumber, camelot, tabula
    header_row: Optional[List[str]] = None
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'table_id': self.table_id,
            'table_type': self.table_type.value,
            'page_number': self.page_number,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'extraction_method': self.extraction_method,
            'header_row': self.header_row,
            'row_count': len(self.dataframe),
            'col_count': len(self.dataframe.columns),
            'columns': list(self.dataframe.columns),
        }


# =============================================================================
# TABLE DETECTION USING PDFPLUMBER
# =============================================================================

def detect_tables_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Detect table regions in PDF using pdfplumber line detection.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of table region dicts with bboxes and metadata
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed")
        return []

    detected_regions = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Get page dimensions
                width = page.width
                height = page.height

                # Find tables using pdfplumber's table finder
                tables = page.find_tables()

                for idx, table in enumerate(tables):
                    bbox = table.bbox  # (x0, y0, x1, y1)

                    # Calculate relative position
                    rel_x = bbox[0] / width
                    rel_y = bbox[1] / height

                    detected_regions.append({
                        'page': page_num,
                        'bbox': bbox,
                        'rel_position': (rel_x, rel_y),
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1],
                        'cell_count': len(table.cells) if hasattr(table, 'cells') else 0,
                        'method': 'pdfplumber'
                    })

                # Also detect based on line intersections
                lines = page.lines or []
                if len(lines) > 10:
                    # Group lines into potential table regions
                    h_lines = [l for l in lines if abs(l.get('top', 0) - l.get('bottom', 0)) < 2]
                    v_lines = [l for l in lines if abs(l.get('x0', 0) - l.get('x1', 0)) < 2]

                    if h_lines and v_lines:
                        # Find bounding box of line clusters
                        x_coords = [l.get('x0', 0) for l in v_lines] + [l.get('x1', 0) for l in v_lines]
                        y_coords = [l.get('top', 0) for l in h_lines] + [l.get('bottom', 0) for l in h_lines]

                        if x_coords and y_coords:
                            line_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                            # Check if this region is already detected
                            is_new = True
                            for existing in detected_regions:
                                if existing['page'] == page_num:
                                    # Check overlap
                                    ex_bbox = existing['bbox']
                                    overlap = (
                                        max(0, min(line_bbox[2], ex_bbox[2]) - max(line_bbox[0], ex_bbox[0])) *
                                        max(0, min(line_bbox[3], ex_bbox[3]) - max(line_bbox[1], ex_bbox[1]))
                                    )
                                    area1 = (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1])
                                    if area1 > 0 and overlap / area1 > 0.5:
                                        is_new = False
                                        break

                            if is_new and (line_bbox[2] - line_bbox[0]) > 50 and (line_bbox[3] - line_bbox[1]) > 30:
                                detected_regions.append({
                                    'page': page_num,
                                    'bbox': line_bbox,
                                    'rel_position': (line_bbox[0] / width, line_bbox[1] / height),
                                    'width': line_bbox[2] - line_bbox[0],
                                    'height': line_bbox[3] - line_bbox[1],
                                    'cell_count': 0,
                                    'method': 'line_detection'
                                })

        logger.info(f"Detected {len(detected_regions)} table regions using pdfplumber")

    except Exception as e:
        logger.error(f"Error detecting tables with pdfplumber: {e}")

    return detected_regions


# =============================================================================
# TABLE EXTRACTION
# =============================================================================

def extract_tables(pdf_path: Path, page_numbers: Optional[List[int]] = None) -> List[ExtractedTable]:
    """
    Extract tables from PDF using multiple methods.

    Priority:
    1. camelot (lattice mode) - best for grid-based tables
    2. pdfplumber - good general purpose
    3. tabula - fallback for stream tables

    Args:
        pdf_path: Path to PDF file
        page_numbers: Optional list of page numbers (0-indexed)

    Returns:
        List of ExtractedTable objects
    """
    all_tables = []
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    # Try camelot first (best for structural drawing tables)
    camelot_tables = _extract_with_camelot(pdf_path, page_numbers)
    all_tables.extend(camelot_tables)

    # Try pdfplumber for tables camelot might miss
    pdfplumber_tables = _extract_with_pdfplumber(pdf_path, page_numbers)

    # Add pdfplumber tables that don't overlap with camelot results
    for pt in pdfplumber_tables:
        is_duplicate = False
        for ct in camelot_tables:
            if ct.page_number == pt.page_number and _tables_overlap(ct.bbox, pt.bbox):
                is_duplicate = True
                break
        if not is_duplicate:
            all_tables.append(pt)

    # Try tabula as fallback if we found few tables
    if len(all_tables) < 2:
        tabula_tables = _extract_with_tabula(pdf_path, page_numbers)
        for tt in tabula_tables:
            is_duplicate = False
            for existing in all_tables:
                if existing.page_number == tt.page_number and _tables_overlap(existing.bbox, tt.bbox):
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_tables.append(tt)

    # Classify all extracted tables
    for table in all_tables:
        table.table_type = classify_table(table.dataframe, table.header_row)

    logger.info(f"Extracted {len(all_tables)} tables total")
    return all_tables


def _extract_with_camelot(pdf_path: Path, page_numbers: Optional[List[int]] = None) -> List[ExtractedTable]:
    """Extract tables using camelot (lattice mode)."""
    tables = []

    try:
        import camelot
    except ImportError:
        logger.warning("camelot-py not installed, skipping camelot extraction")
        return []

    try:
        # Convert to 1-indexed page string for camelot
        if page_numbers:
            pages = ','.join(str(p + 1) for p in page_numbers)
        else:
            pages = 'all'

        # Try lattice mode first (for tables with borders)
        camelot_tables = camelot.read_pdf(
            str(pdf_path),
            pages=pages,
            flavor='lattice',
            line_scale=40,
            copy_text=['v'],  # Copy text from vertical cells
        )

        for idx, ct in enumerate(camelot_tables):
            df = ct.df

            # Skip empty tables
            if df.empty or (df.shape[0] < 2 and df.shape[1] < 2):
                continue

            # Clean up the dataframe
            df = _clean_table_df(df)

            if df.empty:
                continue

            # Get header from first row
            header = list(df.iloc[0]) if len(df) > 0 else None

            # Get bounding box (camelot uses different coordinate system)
            bbox = None
            if hasattr(ct, '_bbox'):
                bbox = ct._bbox

            tables.append(ExtractedTable(
                table_id=f"camelot_{idx}",
                dataframe=df,
                page_number=ct.page - 1,  # Convert to 0-indexed
                bbox=bbox,
                confidence=ct.accuracy / 100 if hasattr(ct, 'accuracy') else 0.7,
                extraction_method='camelot_lattice',
                header_row=header
            ))

        logger.info(f"Camelot extracted {len(tables)} tables")

    except Exception as e:
        logger.warning(f"Camelot extraction failed: {e}")

    return tables


def _extract_with_pdfplumber(pdf_path: Path, page_numbers: Optional[List[int]] = None) -> List[ExtractedTable]:
    """Extract tables using pdfplumber."""
    tables = []

    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed")
        return []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages_to_process = page_numbers if page_numbers else range(len(pdf.pages))

            for page_num in pages_to_process:
                if page_num >= len(pdf.pages):
                    continue

                page = pdf.pages[page_num]

                # Extract tables
                page_tables = page.extract_tables()

                for idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(table_data)

                    # Clean up
                    df = _clean_table_df(df)

                    if df.empty:
                        continue

                    # Get header
                    header = list(df.iloc[0]) if len(df) > 0 else None

                    # Try to get bbox from table finder
                    bbox = None
                    found_tables = page.find_tables()
                    if idx < len(found_tables):
                        bbox = found_tables[idx].bbox

                    tables.append(ExtractedTable(
                        table_id=f"pdfplumber_{page_num}_{idx}",
                        dataframe=df,
                        page_number=page_num,
                        bbox=bbox,
                        confidence=0.6,
                        extraction_method='pdfplumber',
                        header_row=header
                    ))

        logger.info(f"pdfplumber extracted {len(tables)} tables")

    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")

    return tables


def _extract_with_tabula(pdf_path: Path, page_numbers: Optional[List[int]] = None) -> List[ExtractedTable]:
    """Extract tables using tabula (fallback)."""
    tables = []

    try:
        import tabula
    except ImportError:
        logger.warning("tabula-py not installed")
        return []

    try:
        # Convert to 1-indexed for tabula
        if page_numbers:
            pages = [p + 1 for p in page_numbers]
        else:
            pages = 'all'

        # Try stream mode (for tables without borders)
        tabula_tables = tabula.read_pdf(
            str(pdf_path),
            pages=pages,
            multiple_tables=True,
            guess=True,
            stream=True
        )

        for idx, df in enumerate(tabula_tables):
            if df.empty or (df.shape[0] < 2 and df.shape[1] < 2):
                continue

            df = _clean_table_df(df)

            if df.empty:
                continue

            header = list(df.columns) if not df.empty else None

            tables.append(ExtractedTable(
                table_id=f"tabula_{idx}",
                dataframe=df,
                page_number=0,  # tabula doesn't always return page info
                bbox=None,
                confidence=0.5,
                extraction_method='tabula_stream',
                header_row=header
            ))

        logger.info(f"tabula extracted {len(tables)} tables")

    except Exception as e:
        logger.warning(f"tabula extraction failed: {e}")

    return tables


def _clean_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean extracted table DataFrame."""
    if df.empty:
        return df

    # Remove completely empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Replace None/NaN with empty string
    df = df.fillna('')

    # Convert all cells to string
    df = df.astype(str)

    # Strip whitespace
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

    # Remove rows that are all empty strings
    df = df[~(df == '').all(axis=1)]

    # Reset index
    df = df.reset_index(drop=True)

    return df


def _tables_overlap(bbox1: Optional[Tuple], bbox2: Optional[Tuple], threshold: float = 0.5) -> bool:
    """Check if two bounding boxes overlap significantly."""
    if bbox1 is None or bbox2 is None:
        return False

    # Calculate intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    min_area = min(area1, area2)
    if min_area <= 0:
        return False

    return intersection / min_area > threshold


# =============================================================================
# TABLE CLASSIFICATION
# =============================================================================

# Keywords for table classification
COLUMN_SCHEDULE_KEYWORDS = [
    'column', 'col', 'mark', 'size', 'reinforcement', 'rebar', 'main bar',
    'tie', 'stirrup', 'c1', 'c2', 'c3', 'nos', 'dia', 'spacing'
]

FOOTING_SCHEDULE_KEYWORDS = [
    'footing', 'ftg', 'foundation', 'f1', 'f2', 'f3', 'length', 'breadth',
    'depth', 'l x b', 'lxb', 'size', 'pedestal', 'isolated', 'combined'
]

BEAM_SCHEDULE_KEYWORDS = [
    'beam', 'plinth', 'tie beam', 'lintel', 'b1', 'b2', 'pb', 'tb',
    'span', 'width', 'depth', 'top', 'bottom', 'stirrup', 'extra'
]

BAR_SCHEDULE_KEYWORDS = [
    'bar', 'bbs', 'bending', 'schedule', 'shape', 'dia', 'diameter',
    'length', 'nos', 'total', 'weight', 'cutting', 'member', 'bar mark'
]


def classify_table(df: pd.DataFrame, header_row: Optional[List[str]] = None) -> TableType:
    """
    Classify table type based on content and headers.

    Args:
        df: Table DataFrame
        header_row: Optional header row

    Returns:
        TableType enum
    """
    if df.empty:
        return TableType.UNKNOWN

    # Combine all text for keyword matching
    all_text = ' '.join(df.astype(str).values.flatten()).lower()
    if header_row:
        all_text += ' ' + ' '.join(str(h).lower() for h in header_row)

    # Count keyword matches
    scores = {
        TableType.COLUMN_SCHEDULE: sum(1 for kw in COLUMN_SCHEDULE_KEYWORDS if kw in all_text),
        TableType.FOOTING_SCHEDULE: sum(1 for kw in FOOTING_SCHEDULE_KEYWORDS if kw in all_text),
        TableType.BEAM_SCHEDULE: sum(1 for kw in BEAM_SCHEDULE_KEYWORDS if kw in all_text),
        TableType.BAR_SCHEDULE: sum(1 for kw in BAR_SCHEDULE_KEYWORDS if kw in all_text),
    }

    # Check for specific patterns
    if re.search(r'\bc\d+\b', all_text) and ('size' in all_text or 'mm' in all_text):
        scores[TableType.COLUMN_SCHEDULE] += 3

    if re.search(r'\bf\d+\b', all_text) and ('footing' in all_text or 'foundation' in all_text):
        scores[TableType.FOOTING_SCHEDULE] += 3

    if re.search(r'\b(pb|tb|b)\d+\b', all_text) and ('beam' in all_text or 'span' in all_text):
        scores[TableType.BEAM_SCHEDULE] += 3

    if 'bbs' in all_text or ('bar' in all_text and 'schedule' in all_text):
        scores[TableType.BAR_SCHEDULE] += 5

    # Get highest scoring type
    max_score = max(scores.values())
    if max_score < 2:
        return TableType.UNKNOWN

    for table_type, score in scores.items():
        if score == max_score:
            return table_type

    return TableType.UNKNOWN


# =============================================================================
# SCHEDULE PARSERS
# =============================================================================

def parse_column_schedule(table: ExtractedTable) -> List[Dict[str, Any]]:
    """
    Parse column schedule table into structured data.

    Expected columns: Mark, Size, Reinforcement, etc.

    Returns:
        List of column dicts with mark, size_mm, bars, etc.
    """
    columns = []
    df = table.dataframe

    if df.empty:
        return columns

    # Try to identify header row
    header_idx = _find_header_row(df, ['mark', 'column', 'size', 'reinforcement'])

    if header_idx is None:
        # Use first row as header
        header_idx = 0

    # Get headers
    headers = [str(h).lower().strip() for h in df.iloc[header_idx]]

    # Find relevant columns
    mark_col = _find_column(headers, ['mark', 'column', 'col', 'type'])
    size_col = _find_column(headers, ['size', 'dimension', 'mm', 'section'])
    rebar_col = _find_column(headers, ['reinforcement', 'rebar', 'main', 'bar', 'steel'])
    tie_col = _find_column(headers, ['tie', 'stirrup', 'link', 'ring'])

    # Parse data rows
    for idx in range(header_idx + 1, len(df)):
        row = df.iloc[idx]

        try:
            col_data = {
                'source': 'table',
                'table_id': table.table_id,
                'confidence': table.confidence * 0.9
            }

            # Extract mark
            if mark_col is not None:
                col_data['mark'] = str(row.iloc[mark_col]).strip()

            # Extract size
            if size_col is not None:
                size_str = str(row.iloc[size_col]).strip()
                size_mm = _parse_size(size_str)
                if size_mm:
                    col_data['size_mm'] = size_mm

            # Extract reinforcement
            if rebar_col is not None:
                col_data['reinforcement'] = str(row.iloc[rebar_col]).strip()

            # Extract ties
            if tie_col is not None:
                col_data['ties'] = str(row.iloc[tie_col]).strip()

            # Only add if we have at least mark and size
            if col_data.get('mark') and col_data.get('size_mm'):
                columns.append(col_data)

        except Exception as e:
            logger.debug(f"Error parsing column row {idx}: {e}")
            continue

    logger.info(f"Parsed {len(columns)} columns from schedule")
    return columns


def parse_footing_schedule(table: ExtractedTable) -> List[Dict[str, Any]]:
    """
    Parse footing schedule table into structured data.

    Expected columns: Mark, Size (L x B x D), Column, etc.

    Returns:
        List of footing dicts with mark, L_mm, B_mm, D_mm, etc.
    """
    footings = []
    df = table.dataframe

    if df.empty:
        return footings

    # Try to identify header row
    header_idx = _find_header_row(df, ['mark', 'footing', 'size', 'length', 'breadth'])

    if header_idx is None:
        header_idx = 0

    headers = [str(h).lower().strip() for h in df.iloc[header_idx]]

    # Find relevant columns
    mark_col = _find_column(headers, ['mark', 'footing', 'ftg', 'type'])
    size_col = _find_column(headers, ['size', 'dimension', 'l x b', 'lxb'])
    length_col = _find_column(headers, ['length', 'l', 'long'])
    breadth_col = _find_column(headers, ['breadth', 'b', 'width', 'short'])
    depth_col = _find_column(headers, ['depth', 'd', 'thickness', 'thk'])
    column_col = _find_column(headers, ['column', 'col', 'for', 'support'])

    # Parse data rows
    for idx in range(header_idx + 1, len(df)):
        row = df.iloc[idx]

        try:
            ftg_data = {
                'source': 'table',
                'table_id': table.table_id,
                'confidence': table.confidence * 0.9
            }

            # Extract mark
            if mark_col is not None:
                ftg_data['mark'] = str(row.iloc[mark_col]).strip()

            # Extract dimensions - try size column first, then individual L/B/D
            if size_col is not None:
                size_str = str(row.iloc[size_col]).strip()
                dims = _parse_footing_size(size_str)
                if dims:
                    ftg_data['L_mm'] = dims[0]
                    ftg_data['B_mm'] = dims[1]
                    if len(dims) > 2:
                        ftg_data['D_mm'] = dims[2]

            # Override with individual columns if available
            if length_col is not None:
                l_val = _parse_dimension(str(row.iloc[length_col]))
                if l_val:
                    ftg_data['L_mm'] = l_val

            if breadth_col is not None:
                b_val = _parse_dimension(str(row.iloc[breadth_col]))
                if b_val:
                    ftg_data['B_mm'] = b_val

            if depth_col is not None:
                d_val = _parse_dimension(str(row.iloc[depth_col]))
                if d_val:
                    ftg_data['D_mm'] = d_val

            # Extract associated column
            if column_col is not None:
                ftg_data['column_mark'] = str(row.iloc[column_col]).strip()

            # Only add if we have mark and at least one dimension
            if ftg_data.get('mark') and (ftg_data.get('L_mm') or ftg_data.get('B_mm')):
                footings.append(ftg_data)

        except Exception as e:
            logger.debug(f"Error parsing footing row {idx}: {e}")
            continue

    logger.info(f"Parsed {len(footings)} footings from schedule")
    return footings


def parse_beam_schedule(table: ExtractedTable) -> List[Dict[str, Any]]:
    """
    Parse beam schedule table into structured data.

    Returns:
        List of beam dicts with mark, width_mm, depth_mm, span_mm, etc.
    """
    beams = []
    df = table.dataframe

    if df.empty:
        return beams

    header_idx = _find_header_row(df, ['mark', 'beam', 'size', 'span', 'width'])

    if header_idx is None:
        header_idx = 0

    headers = [str(h).lower().strip() for h in df.iloc[header_idx]]

    mark_col = _find_column(headers, ['mark', 'beam', 'member', 'type'])
    size_col = _find_column(headers, ['size', 'section', 'dimension'])
    width_col = _find_column(headers, ['width', 'b', 'breadth'])
    depth_col = _find_column(headers, ['depth', 'd', 'overall'])
    span_col = _find_column(headers, ['span', 'length', 'clear'])

    for idx in range(header_idx + 1, len(df)):
        row = df.iloc[idx]

        try:
            beam_data = {
                'source': 'table',
                'table_id': table.table_id,
                'confidence': table.confidence * 0.9
            }

            if mark_col is not None:
                beam_data['mark'] = str(row.iloc[mark_col]).strip()

            if size_col is not None:
                size_str = str(row.iloc[size_col]).strip()
                size = _parse_size(size_str)
                if size:
                    beam_data['width_mm'] = size[0]
                    beam_data['depth_mm'] = size[1]

            if width_col is not None:
                w_val = _parse_dimension(str(row.iloc[width_col]))
                if w_val:
                    beam_data['width_mm'] = w_val

            if depth_col is not None:
                d_val = _parse_dimension(str(row.iloc[depth_col]))
                if d_val:
                    beam_data['depth_mm'] = d_val

            if span_col is not None:
                span_val = _parse_dimension(str(row.iloc[span_col]))
                if span_val:
                    beam_data['span_mm'] = span_val

            if beam_data.get('mark') and (beam_data.get('width_mm') or beam_data.get('depth_mm')):
                beams.append(beam_data)

        except Exception as e:
            logger.debug(f"Error parsing beam row {idx}: {e}")
            continue

    logger.info(f"Parsed {len(beams)} beams from schedule")
    return beams


def parse_bar_schedule(table: ExtractedTable) -> List[Dict[str, Any]]:
    """
    Parse bar bending schedule (BBS) into structured data.

    Returns:
        List of bar dicts with member, bar_mark, dia, nos, length, shape, etc.
    """
    bars = []
    df = table.dataframe

    if df.empty:
        return bars

    header_idx = _find_header_row(df, ['bar', 'dia', 'diameter', 'nos', 'length'])

    if header_idx is None:
        header_idx = 0

    headers = [str(h).lower().strip() for h in df.iloc[header_idx]]

    member_col = _find_column(headers, ['member', 'element', 'location'])
    bar_mark_col = _find_column(headers, ['bar mark', 'mark', 'bar no', 'ref'])
    dia_col = _find_column(headers, ['dia', 'diameter', 'size', 'mm'])
    nos_col = _find_column(headers, ['nos', 'no', 'qty', 'quantity', 'number'])
    length_col = _find_column(headers, ['length', 'cutting', 'cut length'])
    shape_col = _find_column(headers, ['shape', 'type', 'code'])

    for idx in range(header_idx + 1, len(df)):
        row = df.iloc[idx]

        try:
            bar_data = {
                'source': 'table',
                'table_id': table.table_id,
                'confidence': table.confidence * 0.85
            }

            if member_col is not None:
                bar_data['member'] = str(row.iloc[member_col]).strip()

            if bar_mark_col is not None:
                bar_data['bar_mark'] = str(row.iloc[bar_mark_col]).strip()

            if dia_col is not None:
                dia_str = str(row.iloc[dia_col]).strip()
                dia = _parse_dimension(dia_str)
                if dia:
                    bar_data['dia_mm'] = dia

            if nos_col is not None:
                nos_str = str(row.iloc[nos_col]).strip()
                match = re.search(r'(\d+)', nos_str)
                if match:
                    bar_data['nos'] = int(match.group(1))

            if length_col is not None:
                length = _parse_dimension(str(row.iloc[length_col]))
                if length:
                    bar_data['length_mm'] = length

            if shape_col is not None:
                bar_data['shape'] = str(row.iloc[shape_col]).strip()

            if bar_data.get('dia_mm') or bar_data.get('nos'):
                bars.append(bar_data)

        except Exception as e:
            logger.debug(f"Error parsing bar row {idx}: {e}")
            continue

    logger.info(f"Parsed {len(bars)} bars from BBS")
    return bars


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _find_header_row(df: pd.DataFrame, keywords: List[str]) -> Optional[int]:
    """Find the row that looks like a header."""
    for idx in range(min(5, len(df))):
        row_text = ' '.join(str(v).lower() for v in df.iloc[idx])
        matches = sum(1 for kw in keywords if kw in row_text)
        if matches >= 2:
            return idx
    return None


def _find_column(headers: List[str], keywords: List[str]) -> Optional[int]:
    """Find column index matching any keyword."""
    for idx, header in enumerate(headers):
        for kw in keywords:
            if kw in header:
                return idx
    return None


def _parse_size(size_str: str) -> Optional[Tuple[int, int]]:
    """Parse size string like '300x300' or '300 x 450' into (width, depth)."""
    if not size_str:
        return None

    # Try patterns: 300x300, 300 x 300, 300X300, 300mm x 300mm
    patterns = [
        r'(\d+)\s*[xX×]\s*(\d+)',
        r'(\d+)\s*mm\s*[xX×]\s*(\d+)\s*mm',
    ]

    for pattern in patterns:
        match = re.search(pattern, size_str)
        if match:
            return (int(match.group(1)), int(match.group(2)))

    return None


def _parse_footing_size(size_str: str) -> Optional[Tuple[int, ...]]:
    """Parse footing size like '1500x1500x450' into (L, B, D)."""
    if not size_str:
        return None

    # Try patterns for L x B x D
    patterns = [
        r'(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)',  # 1500x1500x450
        r'(\d+)\s*[xX×]\s*(\d+)',  # 1500x1500 (no depth)
    ]

    for pattern in patterns:
        match = re.search(pattern, size_str)
        if match:
            groups = match.groups()
            return tuple(int(g) for g in groups)

    return None


def _parse_dimension(dim_str: str) -> Optional[int]:
    """Parse dimension string to integer mm."""
    if not dim_str:
        return None

    dim_str = str(dim_str).strip().lower()

    # Remove units
    dim_str = re.sub(r'\s*(mm|m|cm)\s*', '', dim_str)

    # Extract number
    match = re.search(r'(\d+(?:\.\d+)?)', dim_str)
    if match:
        value = float(match.group(1))

        # Convert to mm if looks like meters
        if value < 10 and '.' in match.group(1):
            value = value * 1000

        return int(value)

    return None
