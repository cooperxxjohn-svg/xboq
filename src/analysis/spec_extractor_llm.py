"""
LLM-based specification quantity extractor.

Scans technical specification / scope-of-work text from tender PDFs and
extracts explicitly stated construction quantities that rule-based QTO
modules cannot capture (special finishes, ironmongery, drainage sizes,
specific structural grades, etc.).

Gracefully degrades: returns empty ExtractedSpecItems if no LLM key is
configured or if the call fails.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPEC_KEYWORDS = {
    "specification",
    "technical spec",
    "scope of work",
    "schedule of finishes",
    "special conditions",
    "materials and workmanship",
}

_LEGAL_KEYWORDS = {"clause", "penalty", "liquidated damages"}

_LEGAL_THRESHOLD = 3  # skip page if >3 legal keywords AND no spec keywords

_MAX_SPEC_CHARS = 4000
_LLM_MODEL = "claude-haiku-4-5-20251001"
_LLM_MAX_TOKENS = 1000

_SYSTEM_PROMPT = (
    "You are a quantity surveyor extracting construction items from Indian tender "
    "specifications.\n"
    "Extract ONLY items with explicitly stated specifications. Do not infer or add "
    "items not mentioned."
)

_USER_PROMPT_TEMPLATE = (
    "Extract construction BOQ items from this specification text. Return a JSON array.\n"
    'Each item: {{"description": str, "trade": str, "unit": str, "qty": number or 0, '
    '"qty_per_sqm": number or 0, "confidence": "high"|"medium"|"low", '
    '"source_text": str (max 100 chars), "spec_ref": str}}\n\n'
    "Trades: structural, masonry, finishing, waterproofing, electrical, plumbing, "
    "external, mep\n\n"
    "Text:\n{spec_text}\n\n"
    "Return ONLY valid JSON array, no markdown, no explanation."
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpecItem:
    description: str
    trade: str
    unit: str
    qty: float = 0.0
    qty_per_sqm: float = 0.0
    confidence: str = "low"  # "high" | "medium" | "low"
    source_text: str = ""    # sentence that triggered extraction, max 200 chars
    spec_ref: str = ""        # IS code or spec reference, e.g. "IS 2386"


@dataclass
class ExtractedSpecItems:
    items: List[SpecItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_used: bool = False
    token_count: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_spec_page(text: str) -> bool:
    """Return True if the page contains specification-relevant content."""
    lower = text.lower()

    has_spec = any(kw in lower for kw in _SPEC_KEYWORDS)

    legal_count = sum(1 for kw in _LEGAL_KEYWORDS if kw in lower)
    is_purely_legal = legal_count > _LEGAL_THRESHOLD and not has_spec

    return has_spec and not is_purely_legal


def _collect_spec_text(
    page_texts: List[Tuple[int, str, str]],
    max_pages: int,
) -> str:
    """Filter to spec pages, concatenate, and truncate for cost control."""
    collected: List[str] = []
    pages_used = 0

    for _page_num, _page_type, text in page_texts:
        if pages_used >= max_pages:
            break
        if not text or not text.strip():
            continue
        if _is_spec_page(text):
            collected.append(text.strip())
            pages_used += 1

    combined = "\n\n".join(collected)
    if len(combined) > _MAX_SPEC_CHARS:
        combined = combined[:_MAX_SPEC_CHARS]
    return combined


def _parse_llm_response(raw: str) -> Tuple[List[SpecItem], List[str]]:
    """Parse the JSON array returned by the LLM into SpecItem objects."""
    warnings: List[str] = []
    items: List[SpecItem] = []

    # Strip accidental markdown code fences if any
    cleaned = re.sub(r"```[a-z]*", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        warnings.append(f"LLM response JSON parse error: {exc}")
        logger.warning("spec_extractor_llm: JSON parse failed — %s", exc)
        return items, warnings

    if not isinstance(data, list):
        warnings.append("LLM response was not a JSON array — skipped")
        return items, warnings

    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            source_text = str(entry.get("source_text", ""))[:200]
            item = SpecItem(
                description=str(entry.get("description", "")).strip(),
                trade=str(entry.get("trade", "")).strip().lower(),
                unit=str(entry.get("unit", "")).strip(),
                qty=float(entry.get("qty") or 0.0),
                qty_per_sqm=float(entry.get("qty_per_sqm") or 0.0),
                confidence=str(entry.get("confidence", "low")).lower(),
                source_text=source_text,
                spec_ref=str(entry.get("spec_ref", "")).strip(),
            )
            if item.description:
                items.append(item)
        except (TypeError, ValueError) as exc:
            warnings.append(f"Skipped malformed item from LLM response: {exc}")

    return items, warnings


def _deduplicate(items: List[SpecItem]) -> List[SpecItem]:
    """Remove duplicate items by case-insensitive description."""
    seen: set[str] = set()
    unique: List[SpecItem] = []
    for item in items:
        key = item.description.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_spec_quantities(
    page_texts: List[Tuple[int, str, str]],
    max_pages: int = 20,
    model: str = None,
) -> ExtractedSpecItems:
    """
    Extract construction quantities from specification/scope-of-work pages.

    Parameters
    ----------
    page_texts:
        List of (page_number, page_type, text) tuples — same format used
        throughout the xBOQ pipeline.
    max_pages:
        Maximum number of specification pages to process (cost control).
    model:
        Override the default LLM model. Defaults to claude-haiku-4-5-20251001.

    Returns
    -------
    ExtractedSpecItems
        Always returns a valid object; never raises.
    """
    result = ExtractedSpecItems()

    # Step 1–2: filter and collect spec text
    spec_text = _collect_spec_text(page_texts, max_pages)

    # Step 3: nothing to process
    if not spec_text.strip():
        logger.debug("spec_extractor_llm: no specification pages found")
        return result

    # Step 4: check offline mode — no external calls allowed
    if os.environ.get("XBOQ_OFFLINE_MODE", "").lower() in ("1", "true", "yes"):
        result.warnings.append("XBOQ_OFFLINE_MODE=true — spec LLM extraction skipped")
        logger.info("spec_extractor_llm: offline mode — skipping LLM call")
        return result

    # Step 5: import anthropic
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        result.warnings.append("anthropic package not installed")
        logger.warning("spec_extractor_llm: anthropic not installed — skipping LLM extraction")
        return result

    # Step 6: check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        result.warnings.append("ANTHROPIC_API_KEY not set — spec LLM extraction skipped")
        logger.warning("spec_extractor_llm: ANTHROPIC_API_KEY not set")
        return result

    # Step 6–7: build prompt and call LLM
    chosen_model = model or _LLM_MODEL
    user_prompt = _USER_PROMPT_TEMPLATE.format(spec_text=spec_text)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=chosen_model,
            max_tokens=_LLM_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text: str = ""
        if response.content:
            raw_text = response.content[0].text if hasattr(response.content[0], "text") else ""

        token_count = 0
        if hasattr(response, "usage") and response.usage:
            token_count = (
                getattr(response.usage, "input_tokens", 0)
                + getattr(response.usage, "output_tokens", 0)
            )

    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"LLM call failed: {exc}")
        logger.warning("spec_extractor_llm: LLM call failed — %s", exc)
        return result

    # Step 8: parse response
    items, parse_warnings = _parse_llm_response(raw_text)
    result.warnings.extend(parse_warnings)

    # Step 9: deduplicate
    items = _deduplicate(items)

    # Step 10: populate result
    result.items = items
    result.llm_used = True
    result.token_count = token_count

    logger.info(
        "spec_extractor_llm: extracted %d spec items (tokens used: %d)",
        len(items),
        token_count,
    )
    return result


def merge_spec_items_with_qto(
    spec_items: ExtractedSpecItems,
    qto_items: List[dict],
) -> List[dict]:
    """
    Merge LLM-extracted spec items with existing QTO line items.

    For each spec item:
    - If a similar item already exists in qto_items (substring match on
      description, case-insensitive) AND the spec item has qty > 0:
      update that item's qty, set source="spec_text", confidence="high".
    - If no match: append as a new item with source="spec_text".

    Parameters
    ----------
    spec_items:
        Result from extract_spec_quantities().
    qto_items:
        Existing list of QTO line item dicts (mutated in-place for matches).

    Returns
    -------
    Merged list of dicts.
    """
    merged = list(qto_items)  # shallow copy of the list (dicts not copied)

    def _word_prefix(desc: str, trade: str = "", n: int = 2) -> str:
        """First N meaningful words (≥3 chars) + trade as a similarity key."""
        words = [w.strip("(),.:") for w in desc.lower().split() if len(w.strip("(),.:")) >= 3]
        key = " ".join(words[:n])
        return f"{trade}::{key}" if trade else key

    for spec in spec_items.items:
        spec_desc_lower = spec.description.lower()
        spec_prefix = _word_prefix(spec.description, spec.trade)

        # Look for an existing item whose description contains or is contained
        # by the spec description (substring match), or shares the same
        # first-2-word prefix (catches "Brick masonry 230mm" vs "Brick masonry in...").
        match_index: int = -1
        for idx, existing in enumerate(merged):
            existing_desc = str(existing.get("description", "")).lower()
            if spec_desc_lower in existing_desc or existing_desc in spec_desc_lower:
                match_index = idx
                break
            existing_trade = str(existing.get("trade", "")).lower()
            if spec_prefix and _word_prefix(existing_desc, existing_trade) == spec_prefix:
                match_index = idx
                break

        if match_index >= 0 and spec.qty > 0:
            # Update existing item
            merged[match_index] = dict(merged[match_index])  # avoid mutating caller's dicts
            merged[match_index]["qty"] = spec.qty
            merged[match_index]["source"] = "spec_text"
            merged[match_index]["confidence"] = "high"
        elif match_index < 0:
            # Add as new item
            new_item: dict = {
                "description": spec.description,
                "trade": spec.trade,
                "unit": spec.unit,
                "qty": spec.qty,
                "qty_per_sqm": spec.qty_per_sqm,
                "confidence": spec.confidence,
                "source": "spec_text",
                "spec_ref": spec.spec_ref,
                "source_text": spec.source_text,
            }
            merged.append(new_item)

    return merged
