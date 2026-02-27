"""
Multi-Document Index — assign doc_ids, map global pages to (doc_id, local_page).

Provides pure functions for multi-document page management without
any Streamlit dependency.  When only one document is present, all
mappings are identity transforms.

All functions are pure (no Streamlit, no I/O).
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict


@dataclass
class DocInfo:
    """Metadata for a single document in a multi-doc set."""
    doc_id: int              # 0-based index
    filename: str
    page_count: int
    path: str
    global_page_start: int   # first global page index for this doc

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiDocIndex:
    """Index mapping global page indices to (doc_id, local_page) tuples."""
    docs: List[DocInfo]
    total_pages: int

    def to_dict(self) -> dict:
        return {
            "docs": [d.to_dict() for d in self.docs],
            "total_pages": self.total_pages,
        }


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_multi_doc_index(file_info: List[dict]) -> MultiDocIndex:
    """
    Build a MultiDocIndex from the pipeline's file_info list.

    Args:
        file_info: List of dicts with keys: name, pages, path, ocr_used.
                   Items with 'error' key or pages=0 are skipped.

    Returns:
        MultiDocIndex with doc_id assignments and cumulative page ranges.
    """
    docs: List[DocInfo] = []
    offset = 0
    for i, fi in enumerate(file_info):
        if fi.get("error") or fi.get("pages", 0) == 0:
            continue
        docs.append(DocInfo(
            doc_id=len(docs),
            filename=fi.get("name", f"doc_{i}.pdf"),
            page_count=fi.get("pages", 0),
            path=fi.get("path", ""),
            global_page_start=offset,
        ))
        offset += fi.get("pages", 0)

    return MultiDocIndex(docs=docs, total_pages=offset)


# =============================================================================
# PAGE MAPPING
# =============================================================================

def global_to_doc_page(
    global_idx: int,
    mdi: MultiDocIndex,
) -> Tuple[int, int]:
    """
    Convert a global 0-indexed page to (doc_id, local_page_idx).

    Returns:
        (doc_id, local_page) or (-1, global_idx) if out of range.
    """
    for doc in reversed(mdi.docs):
        if global_idx >= doc.global_page_start:
            local = global_idx - doc.global_page_start
            if local < doc.page_count:
                return (doc.doc_id, local)
    return (-1, global_idx)


def doc_page_to_global(
    doc_id: int,
    local_page: int,
    mdi: MultiDocIndex,
) -> int:
    """
    Convert (doc_id, local_page) to global page index.

    Returns:
        Global page index, or -1 if doc_id is invalid.
    """
    for doc in mdi.docs:
        if doc.doc_id == doc_id:
            if 0 <= local_page < doc.page_count:
                return doc.global_page_start + local_page
            return -1
    return -1


def convert_evidence_pages(
    pages: List[int],
    mdi: MultiDocIndex,
) -> List[Tuple[int, int]]:
    """
    Convert a list of global page indices to (doc_id, local_page) tuples.
    """
    return [global_to_doc_page(p, mdi) for p in pages]


def get_pdf_path_for_doc(doc_id: int, mdi: MultiDocIndex) -> Optional[str]:
    """Return the file path for a given doc_id, or None if not found."""
    for doc in mdi.docs:
        if doc.doc_id == doc_id:
            return doc.path
    return None


def get_doc_id_for_global_page(global_idx: int, mdi: MultiDocIndex) -> int:
    """Return doc_id for a global page index, or -1 if out of range."""
    doc_id, _ = global_to_doc_page(global_idx, mdi)
    return doc_id
