"""
Global Search Across Analyzed Text

In-memory text search across OCR text cache with context snippets
and markdown highlighting. Used by the global search UI component.
"""

import re
from typing import Dict, List, Optional, Union


def search_ocr_text(
    ocr_cache: Dict[Union[int, str], str],
    query: str,
    context_chars: int = 100,
    max_results: int = 50,
    multi_doc_index: Optional[Dict] = None,
) -> List[Dict]:
    """
    Case-insensitive substring search across all pages in the OCR text cache.

    Args:
        ocr_cache: Dict mapping page index (int or str) to OCR text.
        query: Search query string.
        context_chars: Number of context characters before/after match.
        max_results: Maximum number of results to return.
        multi_doc_index: Optional multi-doc index dict (from payload).
            When present and has >1 doc, results include doc_id + doc_filename.

    Returns:
        List of dicts: {page_idx, page_display, snippet, match_start, match_text,
                        doc_id (optional), doc_filename (optional)}
        Snippet has **match** markdown highlighting around the matched text.
        Empty list for empty query, empty cache, or query < 2 chars.
    """
    if not query or len(query) < 2:
        return []

    if not ocr_cache:
        return []

    results = []
    escaped = re.escape(query)

    # Normalize keys: iterate sorted by page index
    items = []
    for key, text in ocr_cache.items():
        try:
            page_idx = int(key)
        except (ValueError, TypeError):
            continue
        items.append((page_idx, text))
    items.sort(key=lambda x: x[0])

    for page_idx, text in items:
        if not text:
            continue

        for match in re.finditer(escaped, text, re.IGNORECASE):
            if len(results) >= max_results:
                break

            start = match.start()
            end = match.end()
            match_text = match.group()

            # Build context snippet
            ctx_start = max(0, start - context_chars)
            ctx_end = min(len(text), end + context_chars)
            raw_snippet = text[ctx_start:ctx_end]

            # Add ellipsis for truncated context
            prefix = "..." if ctx_start > 0 else ""
            suffix = "..." if ctx_end < len(text) else ""

            # Highlight match in snippet
            # Find match position within the snippet
            match_in_snippet_start = start - ctx_start
            match_in_snippet_end = end - ctx_start
            highlighted = (
                raw_snippet[:match_in_snippet_start]
                + "**" + raw_snippet[match_in_snippet_start:match_in_snippet_end] + "**"
                + raw_snippet[match_in_snippet_end:]
            )

            # Clean up newlines for display
            highlighted = highlighted.replace("\n", " ")

            result_entry = {
                "page_idx": page_idx,
                "page_display": f"Page {page_idx + 1}",
                "snippet": prefix + highlighted + suffix,
                "match_start": start,
                "match_text": match_text,
            }

            # Sprint 9: enrich with doc_id / doc_filename when multi-doc
            if multi_doc_index:
                mdi_docs = multi_doc_index.get("docs", [])
                if len(mdi_docs) > 1:
                    for _d in reversed(mdi_docs):
                        if page_idx >= _d.get("global_page_start", 0):
                            local_p = page_idx - _d.get("global_page_start", 0)
                            if local_p < _d.get("page_count", 0):
                                result_entry["doc_id"] = _d.get("doc_id", -1)
                                result_entry["doc_filename"] = _d.get("filename", "")
                                result_entry["page_display"] = (
                                    f"{_d.get('filename', '?')} p.{local_p + 1}"
                                )
                            break

            results.append(result_entry)

        if len(results) >= max_results:
            break

    return results
