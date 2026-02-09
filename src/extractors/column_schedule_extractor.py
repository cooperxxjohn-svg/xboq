"""
Column Schedule Extractor
Extracts and parses column reinforcement schedules from PDF drawings.

Handles Indian RCC drawing conventions:
- Rebar patterns: "18-25", "4Y25", "4-Y25", "16-Y20"
- Column marks: C1, C36, C42, etc.
- Storeywise reinforcement variations
- Tie specifications

IMPORTANT: Uses safe rebar parsing to avoid confusion between:
- Quantity (number of bars)
- Diameter (mm)

Examples:
- "18-25" -> quantity=18, dia=25mm
- "4Y25" -> quantity=4, dia=25mm
- "16-Y20" -> quantity=16, dia=20mm
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RebarSpec:
    """Parsed rebar specification."""
    quantity: int = 0
    diameter_mm: int = 0
    raw_text: str = ""
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            'quantity': self.quantity,
            'diameter_mm': self.diameter_mm,
            'raw_text': self.raw_text,
            'confidence': self.confidence
        }

    def __str__(self) -> str:
        if self.quantity and self.diameter_mm:
            return f"{self.quantity}Y{self.diameter_mm}"
        return self.raw_text


@dataclass
class TieSpec:
    """Parsed tie/stirrup specification."""
    diameter_mm: int = 0
    spacing_mm: int = 0
    legs: int = 2
    raw_text: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            'diameter_mm': self.diameter_mm,
            'spacing_mm': self.spacing_mm,
            'legs': self.legs,
            'raw_text': self.raw_text,
            'confidence': self.confidence
        }


@dataclass
class ColumnScheduleEntry:
    """
    Single row from column reinforcement schedule.

    Represents one column type with its reinforcement details.
    """
    # Column identification
    column_marks: List[str] = field(default_factory=list)  # ["C36", "C42", "C44"]
    section_size: str = ""  # "300x600" or "600x600"
    section_mm: Optional[Tuple[int, int]] = None  # (300, 600)

    # Longitudinal reinforcement (may vary by storey)
    longitudinal_raw: str = ""  # Raw text from schedule
    longitudinal_parsed: List[RebarSpec] = field(default_factory=list)
    storeywise_variation: Dict[str, str] = field(default_factory=dict)  # {"G-2": "18-25", "3-5": "16-25"}

    # Ties/stirrups
    ties_raw: str = ""
    ties_parsed: Optional[TieSpec] = None

    # Metadata
    row_number: int = 0
    confidence: float = 0.9
    source_table_id: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'column_marks': self.column_marks,
            'section_size': self.section_size,
            'section_mm': self.section_mm,
            'longitudinal_raw': self.longitudinal_raw,
            'longitudinal_parsed': [r.to_dict() for r in self.longitudinal_parsed],
            'storeywise_variation': self.storeywise_variation,
            'ties_raw': self.ties_raw,
            'ties_parsed': self.ties_parsed.to_dict() if self.ties_parsed else None,
            'row_number': self.row_number,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


@dataclass
class ColumnScheduleResult:
    """Complete extracted column schedule."""
    entries: List[ColumnScheduleEntry] = field(default_factory=list)
    raw_dataframe: Optional[pd.DataFrame] = None
    headers_detected: List[str] = field(default_factory=list)
    page_number: int = 0
    extraction_method: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entries': [e.to_dict() for e in self.entries],
            'headers_detected': self.headers_detected,
            'page_number': self.page_number,
            'extraction_method': self.extraction_method,
            'entry_count': len(self.entries),
            'confidence': self.confidence
        }


# =============================================================================
# SAFE REBAR PARSER
# =============================================================================

def parse_rebar_spec(text: str) -> List[RebarSpec]:
    """
    Safely parse rebar specification text.

    CRITICAL: Never treat diameter as quantity!

    Patterns supported:
    - "18-25" -> quantity=18, dia=25
    - "4Y25" -> quantity=4, dia=25
    - "4-Y25" -> quantity=4, dia=25
    - "16-Y20" -> quantity=16, dia=20
    - "8 nos 25 dia" -> quantity=8, dia=25
    - "8-25+4-20" -> two specs: (8, 25) and (4, 20)

    Args:
        text: Raw rebar text from schedule

    Returns:
        List of parsed RebarSpec objects
    """
    if not text:
        return []

    text = str(text).strip().upper()
    results = []

    # Remove common prefixes/suffixes
    text = re.sub(r'\b(NOS|MM|DIA|BARS?|RODS?)\b', '', text, flags=re.IGNORECASE)

    # Pattern 1: "18-25" or "18 - 25" (quantity-diameter)
    pattern1 = r'(\d+)\s*[-–—]\s*(\d+)'

    # Pattern 2: "4Y25" or "4-Y25" or "4Y-25" (quantity + Y + diameter)
    pattern2 = r'(\d+)\s*[-–—]?\s*Y\s*[-–—]?\s*(\d+)'

    # Pattern 3: "8 nos 25 dia" or "8nos 25dia"
    pattern3 = r'(\d+)\s*(?:nos?|no\.?)\s*(\d+)'

    # Try pattern 2 first (more specific)
    for match in re.finditer(pattern2, text):
        qty = int(match.group(1))
        dia = int(match.group(2))

        # Validate: quantity should be small (1-50), diameter should be 8-40mm
        if 1 <= qty <= 50 and 6 <= dia <= 50:
            results.append(RebarSpec(
                quantity=qty,
                diameter_mm=dia,
                raw_text=match.group(0),
                confidence=0.95
            ))

    # If pattern 2 found matches, return them
    if results:
        return results

    # Try pattern 1 (generic quantity-diameter)
    for match in re.finditer(pattern1, text):
        val1 = int(match.group(1))
        val2 = int(match.group(2))

        # Heuristic: larger value is usually diameter (8-40mm common)
        # Quantity is usually smaller (1-30 bars typical)
        if val1 > val2:
            # val1 might be quantity if it's reasonable (e.g., 18-25)
            # But 25 could also be diameter... need context

            # Rule: If first value > 10 and second is common diameter (12,16,20,25,32)
            common_diameters = {8, 10, 12, 16, 20, 25, 28, 32, 36, 40}

            if val2 in common_diameters and val1 <= 50:
                qty, dia = val1, val2
            elif val1 in common_diameters and val2 <= 50:
                qty, dia = val2, val1
            else:
                # Default: assume format is "quantity-diameter"
                qty, dia = val1, val2
        else:
            # val2 > val1: val2 is likely diameter
            qty, dia = val1, val2

        # Final validation
        if 1 <= qty <= 50 and 6 <= dia <= 50:
            results.append(RebarSpec(
                quantity=qty,
                diameter_mm=dia,
                raw_text=match.group(0),
                confidence=0.85
            ))

    # Try pattern 3 if still no results
    if not results:
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            qty = int(match.group(1))
            dia = int(match.group(2))

            if 1 <= qty <= 50 and 6 <= dia <= 50:
                results.append(RebarSpec(
                    quantity=qty,
                    diameter_mm=dia,
                    raw_text=match.group(0),
                    confidence=0.80
                ))

    # If still no results, try to extract any reasonable numbers
    if not results:
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            # Try to identify which is quantity vs diameter
            nums = [int(n) for n in numbers[:2]]

            # Common diameters in Indian practice
            common_diameters = {8, 10, 12, 16, 20, 25, 28, 32, 36, 40}

            if nums[1] in common_diameters and nums[0] <= 50:
                results.append(RebarSpec(
                    quantity=nums[0],
                    diameter_mm=nums[1],
                    raw_text=text,
                    confidence=0.60
                ))
            elif nums[0] in common_diameters and nums[1] <= 50:
                results.append(RebarSpec(
                    quantity=nums[1],
                    diameter_mm=nums[0],
                    raw_text=text,
                    confidence=0.60
                ))

    return results


def parse_tie_spec(text: str) -> Optional[TieSpec]:
    """
    Parse tie/stirrup specification.

    Patterns:
    - "8@150" -> dia=8, spacing=150
    - "Y8 @ 150 c/c" -> dia=8, spacing=150
    - "2L 8@150" -> 2 legs, dia=8, spacing=150
    - "4L Y10@200" -> 4 legs, dia=10, spacing=200

    Args:
        text: Raw tie specification text

    Returns:
        TieSpec if parsed successfully, None otherwise
    """
    if not text:
        return None

    text = str(text).strip().upper()

    # Pattern: "2L 8@150" or "4L Y10@200"
    legs_pattern = r'(\d+)\s*L[EGS]*\s*'
    legs_match = re.search(legs_pattern, text)
    legs = int(legs_match.group(1)) if legs_match else 2

    # Pattern: "Y8@150" or "8@150" or "8 @ 150"
    tie_pattern = r'Y?\s*(\d+)\s*[@]\s*(\d+)'
    match = re.search(tie_pattern, text)

    if match:
        dia = int(match.group(1))
        spacing = int(match.group(2))

        # Validate
        if 6 <= dia <= 16 and 50 <= spacing <= 400:
            return TieSpec(
                diameter_mm=dia,
                spacing_mm=spacing,
                legs=legs,
                raw_text=text,
                confidence=0.85
            )

    # Try simpler pattern: just "8@150"
    simple_pattern = r'(\d+)\s*[@]\s*(\d+)'
    match = re.search(simple_pattern, text)

    if match:
        val1 = int(match.group(1))
        val2 = int(match.group(2))

        # val1 should be diameter (6-16mm), val2 should be spacing (50-400mm)
        if 6 <= val1 <= 16 and 50 <= val2 <= 400:
            return TieSpec(
                diameter_mm=val1,
                spacing_mm=val2,
                legs=legs,
                raw_text=text,
                confidence=0.75
            )

    return None


def parse_column_marks(text: str) -> List[str]:
    """
    Parse column mark text into list of marks.

    Examples:
    - "C36, C42, C44, C45" -> ["C36", "C42", "C44", "C45"]
    - "C1-C5" -> ["C1", "C2", "C3", "C4", "C5"]
    - "C36,C42,C44" -> ["C36", "C42", "C44"]

    Args:
        text: Raw column mark text

    Returns:
        List of column mark strings
    """
    if not text:
        return []

    text = str(text).strip().upper()
    marks = []

    # First try comma/space separated
    if ',' in text or ' ' in text:
        # Split by comma, space, or newline
        parts = re.split(r'[,\s\n]+', text)
        for part in parts:
            part = part.strip()
            # Check if it looks like a column mark
            if re.match(r'^C\d+$', part):
                marks.append(part)
            elif re.match(r'^\d+$', part):
                marks.append(f"C{part}")

    # Try range pattern: C1-C5
    range_pattern = r'C(\d+)\s*[-–—to]+\s*C?(\d+)'
    range_match = re.search(range_pattern, text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start < end and end - start < 50:  # Reasonable range
            marks.extend([f"C{i}" for i in range(start, end + 1)])

    # If no marks found, try to extract any "C" followed by digits
    if not marks:
        marks = re.findall(r'C\d+', text)

    return list(set(marks))  # Remove duplicates


def parse_section_size(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse column section size.

    Examples:
    - "300x600" -> (300, 600)
    - "600 x 600" -> (600, 600)
    - "300X450" -> (300, 450)

    Args:
        text: Raw section size text

    Returns:
        Tuple (width, depth) in mm, or None
    """
    if not text:
        return None

    text = str(text).strip()

    pattern = r'(\d+)\s*[xX×]\s*(\d+)'
    match = re.search(pattern, text)

    if match:
        w = int(match.group(1))
        d = int(match.group(2))
        # Validate reasonable column sizes (150mm to 1500mm)
        if 150 <= w <= 1500 and 150 <= d <= 1500:
            return (w, d)

    return None


# =============================================================================
# TABLE EXTRACTION
# =============================================================================

def extract_column_schedule(pdf_path: str, page_number: int = 0) -> ColumnScheduleResult:
    """
    Extract column reinforcement schedule from PDF using Camelot.

    First checks if page contains "COLUMN REINFORCEMENT SCHEDULE" header,
    then tries lattice mode (for bordered tables),
    then falls back to stream mode.

    Args:
        pdf_path: Path to PDF file
        page_number: 0-indexed page number

    Returns:
        ColumnScheduleResult with extracted data
    """
    print(f"[COLUMN_SCHEDULE_EXTRACTOR] Starting extraction from {pdf_path}, page {page_number}")

    result = ColumnScheduleResult(page_number=page_number)
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"[COLUMN_SCHEDULE_EXTRACTOR] ERROR: PDF file not found: {pdf_path}")
        return result

    # =========================================================================
    # STEP 0: Check if page contains column schedule header
    # =========================================================================
    page_contains_schedule = False
    schedule_keywords = [
        'column reinforcement schedule',
        'column schedule',
        'reinforcement schedule',
        'column details',
        'column marked',
    ]

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        if page_number < len(doc):
            page_text = doc[page_number].get_text().lower()
            for keyword in schedule_keywords:
                if keyword in page_text:
                    page_contains_schedule = True
                    print(f"[COLUMN_SCHEDULE_EXTRACTOR] Found schedule keyword: '{keyword}'")
                    break
        doc.close()
    except Exception as e:
        print(f"[COLUMN_SCHEDULE_EXTRACTOR] Warning: Could not check page text: {e}")
        # Continue anyway - we'll still try to extract tables

    if not page_contains_schedule:
        print(f"[COLUMN_SCHEDULE_EXTRACTOR] No column schedule keywords found on page {page_number}")
        # Still proceed - the table might be present without explicit header

    # Convert to 1-indexed for camelot
    camelot_page = str(page_number + 1)

    tables = []
    method = ""

    # Try Camelot lattice mode first
    try:
        import camelot

        print(f"[COLUMN_SCHEDULE_EXTRACTOR] Trying Camelot lattice mode on page {camelot_page}")
        logger.info(f"Trying Camelot lattice mode on page {camelot_page}")
        lattice_tables = camelot.read_pdf(
            str(pdf_path),
            pages=camelot_page,
            flavor='lattice',
            line_scale=40,
            copy_text=['v', 'h'],
        )

        print(f"[COLUMN_SCHEDULE_EXTRACTOR] COLUMN SCHEDULE TABLES FOUND: {len(lattice_tables)}")

        if len(lattice_tables) > 0:
            tables = lattice_tables
            method = "camelot_lattice"
            logger.info(f"Camelot lattice found {len(tables)} tables")
            print(f"[COLUMN_SCHEDULE_EXTRACTOR] Camelot lattice found {len(tables)} tables")
        else:
            # Fallback to stream mode
            print(f"[COLUMN_SCHEDULE_EXTRACTOR] No lattice tables found, trying stream mode")
            logger.info("No lattice tables found, trying stream mode")
            stream_tables = camelot.read_pdf(
                str(pdf_path),
                pages=camelot_page,
                flavor='stream',
                edge_tol=50,
            )
            print(f"[COLUMN_SCHEDULE_EXTRACTOR] COLUMN SCHEDULE TABLES FOUND (stream): {len(stream_tables)}")
            if len(stream_tables) > 0:
                tables = stream_tables
                method = "camelot_stream"
                logger.info(f"Camelot stream found {len(tables)} tables")
                print(f"[COLUMN_SCHEDULE_EXTRACTOR] Camelot stream found {len(tables)} tables")

    except ImportError:
        logger.warning("Camelot not installed")
        print("[COLUMN_SCHEDULE_EXTRACTOR] WARNING: Camelot not installed")
    except Exception as e:
        logger.error(f"Camelot extraction failed: {e}")
        print(f"[COLUMN_SCHEDULE_EXTRACTOR] ERROR: Camelot extraction failed: {e}")

    # Fallback to pdfplumber if camelot failed
    if not tables:
        try:
            import pdfplumber

            print("[COLUMN_SCHEDULE_EXTRACTOR] Falling back to pdfplumber")
            logger.info("Falling back to pdfplumber")
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_number < len(pdf.pages):
                    page = pdf.pages[page_number]
                    page_tables = page.extract_tables()

                    print(f"[COLUMN_SCHEDULE_EXTRACTOR] pdfplumber found {len(page_tables)} raw tables")

                    for table_data in page_tables:
                        if table_data and len(table_data) > 1:
                            df = pd.DataFrame(table_data)
                            tables.append(df)

                    if tables:
                        method = "pdfplumber"
                        logger.info(f"pdfplumber found {len(tables)} tables")
                        print(f"[COLUMN_SCHEDULE_EXTRACTOR] pdfplumber extracted {len(tables)} valid tables")

        except ImportError:
            logger.warning("pdfplumber not installed")
            print("[COLUMN_SCHEDULE_EXTRACTOR] WARNING: pdfplumber not installed")
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            print(f"[COLUMN_SCHEDULE_EXTRACTOR] ERROR: pdfplumber extraction failed: {e}")

    result.extraction_method = method

    if not tables:
        logger.warning("No tables extracted from page")
        print("[COLUMN_SCHEDULE_EXTRACTOR] WARNING: No tables extracted from page")
        return result

    print(f"[COLUMN_SCHEDULE_EXTRACTOR] Processing {len(tables)} tables with method: {method}")

    # Find the column schedule table
    schedule_table = None
    schedule_df = None

    # Keywords to identify column reinforcement schedule
    schedule_keywords = [
        'column marked', 'column mark',
        'longitudinal', 'reinforcement',
        'column ties', 'ties', 'stirrup',
        'section', 'size'
    ]

    for table in tables:
        # Get DataFrame
        if hasattr(table, 'df'):
            df = table.df
        elif isinstance(table, pd.DataFrame):
            df = table
        else:
            continue

        # Check if this looks like a column schedule
        all_text = ' '.join(df.astype(str).values.flatten()).lower()

        keyword_matches = sum(1 for kw in schedule_keywords if kw in all_text)

        if keyword_matches >= 2:
            schedule_table = table
            schedule_df = df
            logger.info(f"Found column schedule table with {keyword_matches} keyword matches")
            break

    if schedule_df is None and tables:
        # Use first table as fallback
        if hasattr(tables[0], 'df'):
            schedule_df = tables[0].df
        elif isinstance(tables[0], pd.DataFrame):
            schedule_df = tables[0]

    if schedule_df is None:
        print("[COLUMN_SCHEDULE_EXTRACTOR] No schedule DataFrame found")
        return result

    # Clean the DataFrame
    schedule_df = _clean_dataframe(schedule_df)
    result.raw_dataframe = schedule_df

    print(f"[COLUMN_SCHEDULE_EXTRACTOR] Raw DataFrame shape: {schedule_df.shape}")
    print(f"[COLUMN_SCHEDULE_EXTRACTOR] Raw DataFrame preview:\n{schedule_df.head(5).to_string()}")

    # Parse the schedule
    result.entries = _parse_schedule_dataframe(schedule_df)
    result.headers_detected = list(schedule_df.iloc[0]) if len(schedule_df) > 0 else []

    # Calculate confidence
    if result.entries:
        avg_conf = sum(e.confidence for e in result.entries) / len(result.entries)
        result.confidence = avg_conf
    else:
        result.confidence = 0.3

    logger.info(f"Extracted {len(result.entries)} column schedule entries")
    print(f"[COLUMN_SCHEDULE_EXTRACTOR] FINAL RESULT: {len(result.entries)} entries extracted, confidence: {result.confidence:.2f}")
    return result


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean extracted table DataFrame."""
    if df.empty:
        return df

    # Remove completely empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Replace None/NaN with empty string
    df = df.fillna('')

    # Convert all cells to string and strip whitespace
    df = df.astype(str)
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

    # Remove rows that are all empty strings
    df = df[~(df == '').all(axis=1)]

    # Reset index
    df = df.reset_index(drop=True)

    return df


def _parse_schedule_dataframe(df: pd.DataFrame) -> List[ColumnScheduleEntry]:
    """
    Parse column schedule DataFrame into structured entries.

    Args:
        df: Cleaned DataFrame from table extraction

    Returns:
        List of ColumnScheduleEntry objects
    """
    entries = []

    if df.empty or len(df) < 2:
        return entries

    # Identify header row
    header_idx = _find_header_row(df)
    headers = [str(h).lower().strip() for h in df.iloc[header_idx]]

    logger.info(f"Detected headers: {headers}")

    # Find column indices
    mark_col = _find_column_index(headers, ['column mark', 'col. mark', 'column', 'mark'])
    section_col = _find_column_index(headers, ['section', 'size', 'cross section'])
    long_col = _find_column_index(headers, ['longitudinal', 'main', 'vertical', 'reinf'])
    tie_col = _find_column_index(headers, ['tie', 'stirrup', 'link', 'lateral'])

    logger.info(f"Column indices - mark: {mark_col}, section: {section_col}, long: {long_col}, tie: {tie_col}")

    # Parse data rows
    for row_idx in range(header_idx + 1, len(df)):
        row = df.iloc[row_idx]

        # Skip rows that look like sub-headers or empty
        row_text = ' '.join(str(v) for v in row).lower()
        if not row_text.strip() or 'storey' in row_text and 'reinforcement' in row_text:
            continue

        entry = ColumnScheduleEntry(row_number=row_idx)

        # Parse column marks
        if mark_col is not None:
            marks_text = str(row.iloc[mark_col])
            entry.column_marks = parse_column_marks(marks_text)
            entry.evidence['marks_raw'] = marks_text

        # Parse section size
        if section_col is not None:
            section_text = str(row.iloc[section_col])
            entry.section_size = section_text
            entry.section_mm = parse_section_size(section_text)
            entry.evidence['section_raw'] = section_text

        # Parse longitudinal reinforcement
        if long_col is not None:
            long_text = str(row.iloc[long_col])
            entry.longitudinal_raw = long_text
            entry.longitudinal_parsed = parse_rebar_spec(long_text)
            entry.evidence['longitudinal_raw'] = long_text

        # Parse ties
        if tie_col is not None:
            tie_text = str(row.iloc[tie_col])
            entry.ties_raw = tie_text
            entry.ties_parsed = parse_tie_spec(tie_text)
            entry.evidence['ties_raw'] = tie_text

        # Calculate entry confidence
        conf_factors = []
        if entry.column_marks:
            conf_factors.append(0.9)
        if entry.section_mm:
            conf_factors.append(0.9)
        if entry.longitudinal_parsed:
            conf_factors.append(min(r.confidence for r in entry.longitudinal_parsed))
        if entry.ties_parsed:
            conf_factors.append(entry.ties_parsed.confidence)

        entry.confidence = sum(conf_factors) / len(conf_factors) if conf_factors else 0.5

        # Only add entry if it has meaningful data
        if entry.column_marks or entry.longitudinal_parsed or entry.section_mm:
            entries.append(entry)

    return entries


def _find_header_row(df: pd.DataFrame) -> int:
    """Find the row that contains column headers."""
    header_keywords = ['column', 'mark', 'section', 'longitudinal', 'tie', 'reinforcement']

    for idx in range(min(5, len(df))):
        row_text = ' '.join(str(v).lower() for v in df.iloc[idx])
        matches = sum(1 for kw in header_keywords if kw in row_text)
        if matches >= 2:
            return idx

    return 0  # Default to first row


def _find_column_index(headers: List[str], keywords: List[str]) -> Optional[int]:
    """Find column index matching any keyword."""
    for idx, header in enumerate(headers):
        for kw in keywords:
            if kw in header:
                return idx
    return None
