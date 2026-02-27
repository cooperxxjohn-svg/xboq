"""
Risk Checklist — templated keyword searches across OCR text cache.

Searches for common contractual risk items (LDs, warranty, retention, etc.)
in the tender document text.

Pure function, no Streamlit dependency. Can be tested independently.
"""

import re
from typing import Dict, List, Optional, Set, Union


# =============================================================================
# RISK TEMPLATES
# =============================================================================

RISK_TEMPLATES: List[dict] = [
    {
        "id": "RISK-LD",
        "label": "Liquidated Damages",
        "keywords": ["liquidated damages", "penalty", "ld clause", "penalty for delay"],
        "impact": "high",
    },
    {
        "id": "RISK-WTY",
        "label": "Warranty / DLP",
        "keywords": ["warranty", "defect liability", "dlp", "guarantee period", "maintenance period"],
        "impact": "high",
    },
    {
        "id": "RISK-RET",
        "label": "Retention Money",
        "keywords": ["retention", "retention money", "security deposit", "performance guarantee"],
        "impact": "medium",
    },
    {
        "id": "RISK-ESC",
        "label": "Escalation Clause",
        "keywords": ["escalation", "price variation", "price adjustment", "cost escalation"],
        "impact": "medium",
    },
    {
        "id": "RISK-ALT",
        "label": "Alternates / Substitutions",
        "keywords": ["alternate", "equivalent", "or equal", "approved equal", "substitution"],
        "impact": "low",
    },
    {
        "id": "RISK-SUB",
        "label": "Submittals",
        "keywords": ["submittal", "shop drawing", "material approval", "sample approval"],
        "impact": "medium",
    },
    {
        "id": "RISK-MAK",
        "label": "Approved Makes",
        "keywords": ["approved make", "approved manufacturer", "make list", "brand", "proprietary"],
        "impact": "medium",
    },
]


# =============================================================================
# SEARCH FUNCTION
# =============================================================================

def _extract_snippet(text: str, keyword: str, context_chars: int = 80) -> str:
    """Extract a short snippet around the first occurrence of keyword."""
    lower_text = text.lower()
    idx = lower_text.find(keyword.lower())
    if idx < 0:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(keyword) + context_chars)
    snippet = text[start:end].strip()
    # Add ellipsis if truncated
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def search_risk_items(
    ocr_cache: Dict[Union[int, str], str],
    templates: List[dict] = None,
    page_doc_types: Optional[Dict[int, str]] = None,
    allowed_doc_types: Optional[Set[str]] = None,
) -> List[dict]:
    """
    Search OCR text cache for risk-related keywords.

    Args:
        ocr_cache: Dict of page_idx -> OCR text (from payload["ocr_text_cache"]).
        templates: Optional list of risk templates (defaults to RISK_TEMPLATES).
        page_doc_types: Optional mapping of page_idx -> doc_type string.
            Used together with allowed_doc_types for scoped search.
        allowed_doc_types: Optional set of doc_type strings to restrict search to.
            e.g. {"spec", "conditions", "addendum"}. When None, all pages searched.

    Returns:
        List of results, one per template:
        {
            "template_id": str,
            "label": str,
            "impact": str,
            "found": bool,
            "hits": [{
                "page_idx": int,
                "page_display": str,
                "snippet": str,
                "keyword": str,
            }],
            "searched_pages_count": int,     # pages actually searched
            "searched_doc_types": [str],      # doc_types that were searched
        }
    """
    if templates is None:
        templates = RISK_TEMPLATES

    results: List[dict] = []

    for tmpl in templates:
        hits: List[dict] = []
        seen_pages: set = set()  # avoid duplicate hits on same page
        searched_pages: set = set()
        searched_doc_types_set: set = set()

        for page_key, text in ocr_cache.items():
            if not text:
                continue

            page_idx = int(page_key) if isinstance(page_key, (int, str)) else 0

            # Doc-type scoping (Sprint 8)
            if allowed_doc_types and page_doc_types:
                page_dt = page_doc_types.get(page_idx, "unknown")
                if page_dt not in allowed_doc_types:
                    continue
                searched_doc_types_set.add(page_dt)
            elif page_doc_types:
                # No filter but we have doc_types — track what we searched
                page_dt = page_doc_types.get(page_idx, "unknown")
                searched_doc_types_set.add(page_dt)

            searched_pages.add(page_idx)
            text_lower = text.lower()

            for keyword in tmpl["keywords"]:
                if keyword.lower() in text_lower:
                    if page_idx in seen_pages:
                        continue
                    seen_pages.add(page_idx)
                    snippet = _extract_snippet(text, keyword)
                    hits.append({
                        "page_idx": page_idx,
                        "page_display": f"Page {page_idx + 1}",
                        "snippet": snippet,
                        "keyword": keyword,
                    })

        results.append({
            "template_id": tmpl["id"],
            "label": tmpl["label"],
            "impact": tmpl["impact"],
            "found": len(hits) > 0,
            "hits": hits,
            "searched_pages_count": len(searched_pages),
            "searched_doc_types": sorted(searched_doc_types_set),
        })

    return results
