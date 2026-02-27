"""
Addendum Adapter — converts pipeline OCR text + PageIndex
into ParsedDocument objects for the existing AddendaParser.

This is a thin adapter layer; all parsing logic lives in
src/owner_docs/addenda.py (AddendaParser).
"""

import re
from typing import Dict, List, Optional

from src.owner_docs.parser import ParsedDocument, DocType
from src.owner_docs.addenda import AddendaParser, Addendum
from src.analysis.page_index import PageIndex

# Maximum gap (non-addendum pages) allowed between addendum pages
# before splitting into separate groups.
MAX_GAP = 2


def _extract_addendum_id(text: str) -> Optional[str]:
    """
    Extract addendum number from page text header.

    Reuses AddendaParser.ADDENDUM_NO_PATTERNS for consistency.
    """
    for pattern in AddendaParser.ADDENDUM_NO_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def build_addendum_documents(
    ocr_text_by_page: Dict[int, str],
    page_index: PageIndex,
) -> List[ParsedDocument]:
    """
    Convert pipeline OCR text for addendum-classified pages into
    ParsedDocument objects consumable by AddendaParser.

    Groups addendum pages with gap tolerance (≤2 non-addendum pages
    between groups), then merges groups that share the same addendum
    number based on header parsing.

    Args:
        ocr_text_by_page: Dict of page_idx -> OCR text (only for selected pages).
        page_index: PageIndex with classification for every page.

    Returns:
        List of ParsedDocument objects with doc_type=DocType.ADDENDUM.
    """
    # Filter for addendum pages, sorted by index
    addendum_pages = sorted(
        [p for p in page_index.pages if p.doc_type == "addendum"],
        key=lambda p: p.page_idx,
    )
    if not addendum_pages:
        return []

    # ── Phase 1: Gap-tolerant grouping ────────────────────────────────
    groups: List[List] = []
    current_group = [addendum_pages[0]]
    for p in addendum_pages[1:]:
        gap = p.page_idx - current_group[-1].page_idx - 1
        if gap <= MAX_GAP:
            current_group.append(p)
        else:
            groups.append(current_group)
            current_group = [p]
    groups.append(current_group)

    # ── Phase 2: Header-based merging ─────────────────────────────────
    # Parse addendum ID from first page of each group, merge groups
    # with matching IDs even if they were separated by > MAX_GAP.
    group_ids: List[Optional[str]] = []
    for group in groups:
        first_page_idx = group[0].page_idx
        text = ocr_text_by_page.get(first_page_idx, "")
        group_ids.append(_extract_addendum_id(text))

    merged_groups: List[List] = []
    id_to_merged_idx: Dict[str, int] = {}

    for i, (group, gid) in enumerate(zip(groups, group_ids)):
        if gid and gid in id_to_merged_idx:
            # Merge into existing group with same addendum ID
            merged_groups[id_to_merged_idx[gid]].extend(group)
        else:
            if gid:
                id_to_merged_idx[gid] = len(merged_groups)
            merged_groups.append(list(group))

    # ── Phase 3: Build ParsedDocument per group ───────────────────────
    documents: List[ParsedDocument] = []
    pdf_name = page_index.pdf_name or "unknown.pdf"
    for i, group in enumerate(merged_groups):
        text_parts = []
        for p in group:
            text = ocr_text_by_page.get(p.page_idx, "")
            if text and text.strip():
                text_parts.append(text)
        if not text_parts:
            continue
        combined_text = "\n\n".join(text_parts)
        doc = ParsedDocument(
            file_path=pdf_name,
            file_name=f"{pdf_name}_addendum_{i + 1}",
            doc_type=DocType.ADDENDUM,
            text_content=combined_text,
            pages=text_parts,
            page_count=len(text_parts),
        )
        documents.append(doc)
    return documents


def extract_addenda(
    ocr_text_by_page: Dict[int, str],
    page_index: PageIndex,
) -> List[Addendum]:
    """
    Full pipeline: OCR text + PageIndex -> List[Addendum].

    Converts pipeline data to ParsedDocument objects, then delegates
    to AddendaParser for actual parsing of addendum content.

    Returns:
        Sorted list of Addendum objects (by addendum_no).
    """
    docs = build_addendum_documents(ocr_text_by_page, page_index)
    if not docs:
        return []
    parser = AddendaParser()
    return parser.parse(docs)
