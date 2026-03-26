"""
CAD/DXF Adapter — T4-2.

Parses .dxf files and converts layer/entity data into IndexedPage-compatible
dicts that can be merged into the main PageIndex.

Requires: ezdxf>=1.0.0 (optional — graceful fallback when not installed)

Usage:
    from src.adapters.cad_adapter import parse_dxf, HAS_EZDXF
    pages = parse_dxf("structural_plans.dxf")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# Optional ezdxf
HAS_EZDXF = False
try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Layer → discipline / doc_type mapping
# ---------------------------------------------------------------------------

# Keys are UPPERCASE layer name substrings; first match wins.
_LAYER_DISCIPLINE: Dict[str, str] = {
    "STRUCTURAL": "structural",
    "STRUCT":     "structural",
    "COLUMN":     "structural",
    "BEAM":       "structural",
    "SLAB":       "structural",
    "FOUND":      "structural",
    "FOOTING":    "structural",
    "ARCH":       "architectural",
    "ROOM":       "architectural",
    "WALL":       "architectural",
    "DOOR":       "architectural",
    "WINDOW":     "architectural",
    "PLUMBING":   "mep",
    "PLUMB":      "mep",
    "HVAC":       "mep",
    "DUCT":       "mep",
    "PIPE":       "mep",
    "ELEC":       "mep",
    "ELECTRICAL": "mep",
    "CABLE":      "mep",
    "CIVIL":      "civil",
    "ROAD":       "civil",
    "DRAIN":      "civil",
    "SITE":       "civil",
    "FIRE":       "fire",
    "SPRINK":     "fire",
}

_LAYER_DOC_TYPE: Dict[str, str] = {
    "PLAN":    "plan",
    "SECTION": "section",
    "ELEVAT":  "elevation",
    "DETAIL":  "detail",
    "SCHED":   "schedule",
    "LEGEND":  "legend",
}


def _classify_layer(layer_name: str) -> tuple[str, str]:
    """Return (discipline, doc_type) for a DXF layer name."""
    upper = layer_name.upper()
    discipline = "other"
    for key, val in _LAYER_DISCIPLINE.items():
        if key in upper:
            discipline = val
            break
    doc_type = "plan"  # default
    for key, val in _LAYER_DOC_TYPE.items():
        if key in upper:
            doc_type = val
            break
    return discipline, doc_type


def _majority(values: list, default: str) -> str:
    """Return the most common value in a list, or default if empty."""
    if not values:
        return default
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_dxf(dxf_path: str) -> List[dict]:
    """
    Parse a DXF file and return a list of IndexedPage-compatible dicts.

    Each dict corresponds to one modelspace layout and contains:
        page_idx, doc_type, discipline, sheet_id, title, confidence,
        keywords_hit, has_text_layer, text_content, layer_names, entities

    ALWAYS returns >= 1 page — even if ezdxf is unavailable (synthetic fallback).
    """
    path = Path(dxf_path)
    base_name = path.stem

    if not HAS_EZDXF:
        logger.debug("ezdxf not installed — returning synthetic page for %s", path.name)
        return [_synthetic_page(0, base_name, "unknown", "other", path.name)]

    try:
        doc = ezdxf.readfile(str(path))
    except Exception as exc:
        logger.warning("Could not read DXF %s: %s", path.name, exc)
        return [_synthetic_page(0, base_name, "unknown", "other", path.name)]

    pages: List[dict] = []

    # Process modelspace (index 0)
    try:
        msp = doc.modelspace()
        pages.append(_extract_layout(msp, 0, base_name))
    except Exception as exc:
        logger.debug("Error reading modelspace: %s", exc)
        pages.append(_synthetic_page(0, base_name, "unknown", "other", path.name))

    # Process paper space layouts (index 1+)
    try:
        for idx, layout in enumerate(doc.layouts, start=1):
            if layout.name.upper() == "MODEL":
                continue
            try:
                pages.append(_extract_layout(layout, len(pages), base_name))
            except Exception as exc:
                logger.debug("Error reading layout %s: %s", layout.name, exc)
    except Exception:
        pass

    return pages if pages else [_synthetic_page(0, base_name, "unknown", "other", path.name)]


def _extract_layout(layout, page_idx: int, base_name: str) -> dict:
    """Extract page dict from a single DXF layout."""
    layer_names: List[str] = []
    text_parts: List[str] = []
    entity_count = 0

    for entity in layout:
        entity_count += 1
        layer = getattr(entity, "dxf", None)
        if layer is not None:
            layer_name = getattr(layer, "layer", "") or ""
            if layer_name and layer_name not in layer_names:
                layer_names.append(layer_name)
        # Extract text from TEXT / MTEXT entities
        dxf_type = entity.dxftype()
        if dxf_type in ("TEXT", "MTEXT"):
            try:
                text = entity.plain_mtext() if dxf_type == "MTEXT" else entity.dxf.text
                if text and text.strip():
                    text_parts.append(text.strip())
            except Exception:
                pass

    disciplines = [_classify_layer(ln)[0] for ln in layer_names]
    doc_types   = [_classify_layer(ln)[1] for ln in layer_names]

    discipline = _majority(disciplines, "other")
    doc_type   = _majority(doc_types, "plan")
    text_content = " ".join(text_parts[:100])  # cap

    # Try to infer title from text or layout name
    title = None
    if hasattr(layout, "name"):
        title = layout.name
    if text_parts:
        title = title or text_parts[0][:80]

    confidence = 0.7 if layer_names else 0.3

    return {
        "page_idx": page_idx,
        "doc_type": doc_type,
        "discipline": discipline,
        "sheet_id": base_name,
        "title": title,
        "confidence": confidence,
        "keywords_hit": layer_names[:20],
        "has_text_layer": bool(text_parts),
        "text_content": text_content,
        "layer_names": layer_names,
        "entities": entity_count,
        "source": "dxf",
    }


def _synthetic_page(page_idx: int, sheet_id: str, doc_type: str, discipline: str, filename: str) -> dict:
    return {
        "page_idx": page_idx,
        "doc_type": doc_type,
        "discipline": discipline,
        "sheet_id": sheet_id,
        "title": filename,
        "confidence": 0.0,
        "keywords_hit": [],
        "has_text_layer": False,
        "text_content": "",
        "layer_names": [],
        "entities": 0,
        "source": "dxf_fallback",
    }
