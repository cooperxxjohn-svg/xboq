"""
Page Index — Fast header-strip OCR classification for ALL pages.

Two-pass strategy (Pass 1):
- Text-layer pages: classify from existing text (zero rendering cost)
- Scanned pages: render top 18% header strip at low DPI, OCR, classify

Produces a PageIndex with doc_type + discipline for every page,
enabling intelligent page selection in Pass 2.
"""

import re
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)
DEBUG = os.environ.get("DEBUG_PIPELINE", "0") == "1"

# Optional imports
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    fitz = None

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    pytesseract = None
    Image = None


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class IndexedPage:
    """Classification result for a single page."""
    page_idx: int
    doc_type: str           # cover|index|notes|legend|plan|detail|section|elevation|schedule|spec|boq|conditions|addendum|unknown
    discipline: str         # architectural|structural|electrical|mechanical|plumbing|civil|fire|other|unknown
    sheet_id: Optional[str] = None
    title: Optional[str] = None
    confidence: float = 0.0
    keywords_hit: List[str] = field(default_factory=list)
    has_text_layer: bool = False
    strip_ocr_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageIndex:
    """Index of all pages in a PDF with classification metadata."""
    pdf_name: str
    total_pages: int
    pages: List[IndexedPage] = field(default_factory=list)
    counts_by_type: Dict[str, int] = field(default_factory=dict)
    counts_by_discipline: Dict[str, int] = field(default_factory=dict)
    indexing_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
            "counts_by_type": self.counts_by_type,
            "counts_by_discipline": self.counts_by_discipline,
            "indexing_time_s": round(self.indexing_time_s, 2),
        }

    def pages_of_type(self, *doc_types: str) -> List[IndexedPage]:
        """Return pages matching any of the given doc_types."""
        return [p for p in self.pages if p.doc_type in doc_types]

    def summary_line(self) -> str:
        """Human-readable one-liner for progress messages."""
        parts = []
        for dtype in ["schedule", "plan", "boq", "spec", "section", "elevation",
                       "detail", "notes", "legend", "conditions", "addendum",
                       "cover", "index", "unknown"]:
            n = self.counts_by_type.get(dtype, 0)
            if n > 0:
                parts.append(f"{n} {dtype}")
        return f"Indexed {self.total_pages} pages: {', '.join(parts)}"


# =============================================================================
# CLASSIFICATION PATTERNS (reused from owner_docs/parser.py + plan_graph.py)
# =============================================================================

# Priority order matters — first match wins.
# Each entry: (doc_type, list of regex patterns)
DOC_TYPE_RULES: List[Tuple[str, List[str]]] = [
    # 1. BOQ / Bill of Quantities
    ("boq", [
        r'bill\s+of\s+quantit',
        r'\bboq\b',
        r'schedule\s+of\s+quantit',
        r'item\s+no.*description.*unit.*qty',
        r'sr\.?\s*no.*description.*unit.*quantity',
        r'rate\s+analysis',
        # BUG-10 FIX: Match BOQ continuation pages that only have the column header row.
        # Indian CPWD/PWD BOQs use "SLNo / Sl. No." or "S.No." as the first column.
        # These pages don't repeat "Schedule of Quantities" heading so were missed.
        # NOTE: column headings are newline-separated in PDF text extraction, so we
        # must use [\s\S]{0,N} instead of .* to cross line boundaries.
        r'(?:slno|sl\.?\s*no\.?)[\s\S]{0,80}description[\s\S]{0,80}(?:qty|quantity)',
        r'(?:s\.?\s*no\.?|sr\.?\s*no\.?)[\s\S]{0,80}description[\s\S]{0,80}(?:unit|uom)[\s\S]{0,80}rate',
    ]),
    # 2. Contract conditions
    ("conditions", [
        r'general\s+conditions\s+of\s+contract',
        r'\bgcc\b',
        r'particular\s+conditions',
        r'special\s+conditions',
        r'conditions\s+of\s+contract',
        r'terms\s+and\s+conditions',
        # Sprint 22: India GCC/SCC patterns
        r'\bscc\b',
        r'special\s+conditions\s+of\s+contract',
        r'additional\s+conditions',
        r'supplementary\s+conditions',
        r'instructions?\s+to\s+(?:bidders?|tenderers?)',
        r'invitation\s+(?:to|for)\s+(?:bid|tender)',
        # BUG-9 FIX: Added \b word boundary around NIT so it doesn't match
        # "uNIT", "graNITe", "beNITonite" etc. that appear in BOQ descriptions.
        r'(?:notice\s+inviting\s+tender|\bNIT\b)',
        r'(?:form\s+of\s+tender|form\s+of\s+bid)',
        r'(?:eligibility\s+criteria|qualification\s+criteria)',
        r'(?:liquidated\s+damages|penalty\s+clause)',
        r'(?:arbitration\s+clause|dispute\s+resolution)',
        # Sprint 24: OCR-tolerant patterns for scanned India tenders
        # OCR garbles "Conditions" → "Coneltions", "Coneitions", "Consltions"
        # and may insert punctuation: "Conditions of,Contract"
        r'con\w{1,5}tions?\s+of\s*[,;.]?\s*con\w{1,5}ct',  # fuzzy "Conditions of Contract"
        r'terms?\s+(?:and|&)\s+con\w{1,5}tions?',  # fuzzy "Terms and Conditions"
    ]),
    # 3. Addendum / Corrigendum
    ("addendum", [
        r'addendum',
        r'corrigendum',
        r'amendment\s+no',
        r'pre-?bid\s+meeting',
    ]),
    # 4. Specifications (only match if "specification" is a heading/title, not inline reference)
    ("spec", [
        r'^\s*technical\s+specifications?\s*$',
        r'^\s*general\s+specifications?\s*$',
        r'^\s*particular\s+specifications?\s*$',
        r'^\s*specifications?\s*$',
        r'specification\s+clause\s+\d',
        # Sprint 24: relaxed patterns for scanned India tenders
        # Header strip OCR often gives "CLIENT: ... TECHNICAL SPECIFICATIONS" inline
        r'technical\s+spec',                    # inline match (no anchors)
        r'technica\w?\s+spec',                  # OCR error tolerance
    ]),
    # 5. Schedules (door, window, finish, fixture)
    ("schedule", [
        r'door\s+schedule',
        r'window\s+schedule',
        r'finish\s+schedule',
        r'fixture\s+schedule',
        r'hardware\s+schedule',
        r'schedule\s+of\s+doors',
        r'schedule\s+of\s+windows',
        r'schedule\s+of\s+finishes',
        # Sprint 22: India tender schedule patterns
        r'schedule\s+of\s+doors?\s*(?:&|and)\s*windows?',
        r'schedule\s+of\s+materials?',
        r'room\s+(?:data|finish)\s+sheet',
        r'room\s+finish\s+schedule',
        r'internal\s+finish\s+schedule',
        r'external\s+finish\s+schedule',
        r'plumbing\s+fixture\s+schedule',
        r'sanitary\s+fixture\s+schedule',
        r'door.*window.*schedule',
        # Content-based: detect mark patterns typical of schedules
        r'(?:D-?\d{1,3}|W-?\d{1,3})\s+.{5,80}\s+\d{3,4}\s*[xX×]\s*\d{3,4}',
    ]),
    # 6. Cover / Index
    ("cover", [
        r'\bcover\s+sheet\b',
        r'\btitle\s+sheet\b',
        r'\btitle\s+page\b',
    ]),
    ("index", [
        r'\bdrawing\s+list\b',
        r'\bsheet\s+index\b',
        r'\bdrawing\s+index\b',
        r'\bindex\s+of\s+drawings\b',
        r'\blist\s+of\s+drawings\b',
    ]),
    # 7. Notes / Legend
    ("notes", [
        r'\bgeneral\s+notes\b',
        r'\bnotes\s*:\s*$',
        r'\bnotes\s+and\s+specification',
        # Sprint 22: India tender notes patterns
        r'\bspecification\s+notes\b',
        r'\bstructural\s+notes\b',
        r'\barchitectural\s+notes\b',
        r'\bplumbing\s+notes\b',
        r'\belectrical\s+notes\b',
        r'\bmaterial\s+specifications?\b',
    ]),
    ("legend", [
        r'\blegend\b',
        r'\bsymbols\b',
        r'\babbreviations\b',
    ]),
    # 8. Drawing types (from plan_graph SHEET_TYPE_KEYWORDS)
    ("section", [
        r'^\s*section\s+[A-Z](?:\s*-\s*[A-Z])?',  # "SECTION A-A" at start of line (title, not reference)
        r'\bsectional\s+(?:drawing|elevation|plan)',
        r'\bcross\s+section\b',
        r'\blongitudinal\s+section\b',
    ]),
    ("elevation", [
        r'\belevation\b',
        r'\bfront\s+elevation\b',
        r'\brear\s+elevation\b',
        r'\bside\s+elevation\b',
        r'\bfacade\b',
    ]),
    ("detail", [
        r'^\s*detail\s+drawing',                   # "DETAIL DRAWING" as title
        r'^\s*enlarged\s+detail\b',
        r'^\s*typical\s+detail\b',
        r'^\s*construction\s+detail\b',
        r'^\s*detail\s+of\b',                      # "DETAIL OF STAIRCASE"
    ]),
    ("plan", [
        r'\bfloor\s+plan\b',
        r'\bground\s+floor\b',
        r'\bfirst\s+floor\b',
        r'\bsecond\s+floor\b',
        r'\btypical\s+floor\b',
        r'\bbasement\b',
        r'\bterrace\b',
        r'\broof\s+plan\b',
        r'\bsite\s+plan\b',
        r'\bsite\s+layout\b',
        r'\blocation\s+plan\b',
        r'\bkey\s+plan\b',
        r'\blayout\s+plan\b',
        r'\bbeam\s+layout\b',
        r'\bcolumn\s+layout\b',
        r'\bslab\s+layout\b',
        r'\bfoundation\s+layout\b',
        r'\bframing\s+plan\b',
    ]),
]

# Pre-compile all patterns (MULTILINE so ^ matches line starts)
_COMPILED_RULES: List[Tuple[str, List[re.Pattern]]] = [
    (dtype, [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns])
    for dtype, patterns in DOC_TYPE_RULES
]

# Sheet number patterns (from plan_graph.py)
SHEET_NO_PATTERNS = [
    re.compile(r'\b([A-Z]{1,2})-?(\d{1,3}(?:\.\d{1,2})?)\b'),
    re.compile(r'\bSHEET\s*(?:NO\.?|#)?\s*([A-Z]?\d+)\b', re.IGNORECASE),
    re.compile(r'\bDWG\.?\s*(?:NO\.?|#)?\s*([A-Z]?\d+)\b', re.IGNORECASE),
]

# Discipline prefixes (from plan_graph.py)
DISCIPLINE_PREFIX_MAP = {
    'A': 'architectural', 'AR': 'architectural', 'ARCH': 'architectural',
    'S': 'structural', 'ST': 'structural', 'STR': 'structural',
    'M': 'mechanical', 'ME': 'mechanical', 'MECH': 'mechanical',
    'E': 'electrical', 'EL': 'electrical', 'ELEC': 'electrical',
    'P': 'plumbing', 'PL': 'plumbing', 'PLMB': 'plumbing',
    'C': 'civil', 'CV': 'civil',
    'L': 'landscape', 'LS': 'landscape',
    'FP': 'fire', 'FA': 'fire',
    'ID': 'interior', 'INT': 'interior',
    'G': 'general', 'GEN': 'general',
    'HV': 'hvac', 'HVAC': 'hvac',
    'ELV': 'elv', 'ICT': 'elv', 'IT': 'elv',
    'FIN': 'finishes',
}

# Discipline keywords (from ocr_fallback.py)
DISCIPLINE_KEYWORDS = {
    'architectural': ['ARCHITECTURAL', 'FLOOR PLAN', 'ELEVATION', 'SECTION',
                       'DOOR SCHEDULE', 'WINDOW SCHEDULE', 'FINISH SCHEDULE'],
    'structural':    ['STRUCTURAL', 'FOUNDATION', 'BEAM', 'COLUMN', 'FOOTING',
                       'REINFORCEMENT', 'BAR BENDING'],
    'mechanical':    ['MECHANICAL'],
    'electrical':    ['ELECTRICAL', 'LIGHTING', 'POWER', 'PANEL', 'CABLE'],
    'plumbing':      ['PLUMBING', 'DRAINAGE', 'SANITARY', 'WATER SUPPLY'],
    'civil':         ['SITE PLAN', 'SITE LAYOUT', 'EARTHWORK', 'GRADING',
                       'RETAINING WALL', 'COMPOUND WALL', 'ROAD', 'PAVEMENT'],
    'fire':          ['FIRE FIGHTING', 'FIRE PROTECTION', 'FIRE ALARM',
                       'SPRINKLER'],
    'hvac':          ['HVAC', 'AIR CONDITIONING', 'DUCT', 'AHU', 'CHILLER',
                       'VRF', 'SPLIT AC'],
    'elv':           ['CCTV', 'ACCESS CONTROL', 'INTERCOM', 'BMS',
                       'NETWORKING', 'ICT', 'PA SYSTEM'],
    'finishes':      ['FINISH SCHEDULE', 'PAINTING', 'TILING', 'FLOORING',
                       'WALL FINISH'],
}


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def _extract_sheet_id(text_upper: str) -> Optional[str]:
    """Extract sheet ID from text (e.g. 'A-101', 'S-2.01')."""
    for pattern in SHEET_NO_PATTERNS:
        match = pattern.search(text_upper)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"{groups[0]}-{groups[1]}"
            return groups[0]
    return None


def _discipline_from_sheet_id(sheet_id: Optional[str]) -> Optional[str]:
    """Derive discipline from sheet ID prefix (e.g. 'A-101' -> 'architectural')."""
    if not sheet_id:
        return None
    prefix_match = re.match(r'^([A-Z]+)', sheet_id)
    if prefix_match:
        prefix = prefix_match.group(1)
        return DISCIPLINE_PREFIX_MAP.get(prefix)
    return None


def _discipline_from_keywords(text_upper: str) -> str:
    """Detect discipline from keyword scan."""
    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        if any(kw in text_upper for kw in keywords):
            return disc
    return "unknown"


def _extract_title(text: str) -> Optional[str]:
    """Extract first meaningful line as title."""
    for line in text.split('\n')[:10]:
        line = line.strip()
        if len(line) > 5 and not line.isdigit() and not re.match(r'^[\d\.\,\-\s]+$', line):
            return line[:120]
    return None


def _classify_page(text: str) -> Tuple[str, str, Optional[str], Optional[str], float, List[str]]:
    """
    Classify a page from its text content.

    Returns:
        (doc_type, discipline, sheet_id, title, confidence, keywords_hit)
    """
    text_lower = text.lower()
    text_upper = text.upper()
    keywords_hit = []

    # --- doc_type: first matching rule wins ---
    doc_type = "unknown"
    for dtype, patterns in _COMPILED_RULES:
        for pat in patterns:
            if pat.search(text_lower):
                doc_type = dtype
                keywords_hit.append(pat.pattern)
                break
        if doc_type != "unknown":
            break

    # --- sheet_id ---
    # BUG-11 FIX: Only extract sheet IDs for drawing page types.
    # Text document types (BOQ, conditions, spec, addendum) should never have a
    # sheet ID — material specification codes like M-25 (concrete grade), E-350,
    # T-5, RS-485 etc. that appear in their body text look like sheet IDs but are
    # not.  Extracting them causes false-positive discipline detection and inflates
    # pages_with_sheet_no, which in turn incorrectly flags NIT documents as drawing
    # sets and triggers PASS readiness status for spec-only packs.
    _TEXT_DOC_TYPES = {"boq", "conditions", "addendum", "spec", "notes", "legend"}
    if doc_type in _TEXT_DOC_TYPES:
        sheet_id = None
    else:
        sheet_id = _extract_sheet_id(text_upper)

    # --- discipline ---
    discipline = _discipline_from_sheet_id(sheet_id)
    if not discipline:
        discipline = _discipline_from_keywords(text_upper)

    # Non-drawing doc types get "other" discipline
    if doc_type in ("boq", "conditions", "addendum", "spec"):
        discipline = "other"

    # --- title ---
    title = _extract_title(text)

    # --- Sprint 24: Drawing fallback for blank scanned pages ---
    # Pages with very sparse OCR text (< 15 chars of recognizable content) are
    # likely scanned drawings where the header strip produced garbled nonsense.
    # Classify as "plan" (drawing) with low confidence rather than "unknown".
    if doc_type == "unknown":
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', text)
        if len(clean_text) < 15:
            doc_type = "plan"
            keywords_hit.append("_drawing_fallback_sparse_text")

    # --- confidence ---
    confidence_val = 0.2  # base
    if doc_type != "unknown":
        confidence_val += 0.3
    if sheet_id:
        confidence_val += 0.2
    if discipline != "unknown":
        confidence_val += 0.15
    if len(keywords_hit) > 1:
        confidence_val += 0.1
    if title:
        confidence_val += 0.05
    confidence_val = min(confidence_val, 1.0)
    # Preserve low confidence for drawing fallback
    if "_drawing_fallback_sparse_text" in keywords_hit:
        confidence_val = 0.15

    return doc_type, discipline, sheet_id, title, confidence_val, keywords_hit


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_page_index(
    pdf_path: Path,
    existing_texts: List[str],
    strip_fraction: float = 0.18,
    strip_dpi: int = 150,  # Sprint 24: increased from 90 for better scanned header OCR
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> PageIndex:
    """
    Build a page index by classifying every page in the PDF.

    - Text-layer pages: classify from existing text (fast, no rendering).
    - Scanned pages: render top header strip, OCR, classify.

    Args:
        pdf_path: Path to the PDF file.
        existing_texts: Pre-extracted text per page (from load_pdf_pages).
        strip_fraction: Fraction of page height to render for header strip.
        strip_dpi: DPI for header strip rendering.
        progress_cb: Optional callback(page_idx, total, message).

    Returns:
        PageIndex with classification for every page.
    """
    from .ocr_fallback import page_needs_ocr

    t0 = time.perf_counter()
    total = len(existing_texts)
    indexed_pages: List[IndexedPage] = []

    # Open PDF once for all header-strip renders
    doc = None
    if HAS_FITZ:
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.warning(f"Could not open PDF for indexing: {e}")

    for i in range(total):
        t_page = time.perf_counter()
        text = existing_texts[i] if i < len(existing_texts) else ""
        has_text = not page_needs_ocr(text)

        if has_text:
            # Text-layer page — classify from existing text
            classify_text = text
        else:
            # Scanned page — render header strip + OCR
            classify_text = ""
            if doc and HAS_OCR and i < len(doc):
                try:
                    page = doc[i]
                    rect = page.rect
                    header_rect = fitz.Rect(
                        rect.x0, rect.y0,
                        rect.x1, rect.y0 + rect.height * strip_fraction,
                    )
                    zoom = strip_dpi / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(
                        matrix=mat,
                        clip=header_rect,
                        colorspace=fitz.csGRAY,
                    )
                    import io
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    classify_text = pytesseract.image_to_string(
                        img, config='--psm 6 --oem 3'
                    )
                    del pix, img
                except Exception as e:
                    if DEBUG:
                        logger.warning(f"Header-strip OCR failed p{i}: {e}")

        # Classify
        doc_type, discipline, sheet_id, title, confidence, keywords_hit = \
            _classify_page(classify_text)

        strip_time = time.perf_counter() - t_page

        indexed_pages.append(IndexedPage(
            page_idx=i,
            doc_type=doc_type,
            discipline=discipline,
            sheet_id=sheet_id,
            title=title,
            confidence=confidence,
            keywords_hit=keywords_hit,
            has_text_layer=has_text,
            strip_ocr_time_s=round(strip_time, 4),
        ))

        if progress_cb and (i % 10 == 0 or i == total - 1):
            progress_cb(i, total, f"Indexing page {i + 1}/{total}...")

        if DEBUG and (i % 25 == 0 or i == total - 1):
            logger.info(
                f"[Index] p{i + 1}/{total} "
                f"type={doc_type} disc={discipline} "
                f"sheet={sheet_id or '-'} conf={confidence:.2f} "
                f"text_layer={'yes' if has_text else 'no'} "
                f"time={strip_time:.3f}s"
            )

    if doc:
        doc.close()

    # Compute aggregates
    type_counter: Counter = Counter()
    disc_counter: Counter = Counter()
    for p in indexed_pages:
        type_counter[p.doc_type] += 1
        disc_counter[p.discipline] += 1

    indexing_time = time.perf_counter() - t0

    page_index = PageIndex(
        pdf_name=pdf_path.name,
        total_pages=total,
        pages=indexed_pages,
        counts_by_type=dict(type_counter),
        counts_by_discipline=dict(disc_counter),
        indexing_time_s=indexing_time,
    )

    if DEBUG:
        logger.info(f"[Index] {page_index.summary_line()} in {indexing_time:.1f}s")

    return page_index
