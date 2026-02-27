"""
Surya OCR Integration for xBOQ

Provides line-level bounding boxes with confidence scores from Surya OCR.
Gracefully falls back when Surya is not installed.

Coordinates are returned as page-relative (0.0–1.0) for zoom-invariant rendering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Install gate ──────────────────────────────────────────────────────────
HAS_SURYA = False
try:
    from surya.ocr import run_ocr  # noqa: F401
    from surya.model.detection.model import load_model as load_det_model  # noqa: F401
    from surya.model.recognition.model import load_model as load_rec_model  # noqa: F401
    HAS_SURYA = True
except ImportError:
    pass


@dataclass
class SuryaPageResult:
    """OCR result for a single PDF page."""
    text: str = ""
    bboxes: List[List[float]] = field(default_factory=list)
    # Each bbox: [x0_rel, y0_rel, x1_rel, y1_rel, confidence]


def extract_with_surya(
    pdf_path: Path,
    page_indices: List[int],
    dpi: int = 150,
) -> Dict[int, SuryaPageResult]:
    """
    Run Surya OCR on selected pages and return text + bounding boxes.

    Args:
        pdf_path: Path to the PDF file
        page_indices: 0-indexed page numbers to process
        dpi: Resolution for PDF→image rendering

    Returns:
        Dict[int, SuryaPageResult] — page_idx → result with text and bboxes.
        Bboxes are in page-relative coords [x0_rel, y0_rel, x1_rel, y1_rel, confidence].
    """
    if not HAS_SURYA:
        logger.info("Surya not installed — returning empty results")
        return {}

    results: Dict[int, SuryaPageResult] = {}

    try:
        import fitz  # PyMuPDF for page rendering
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning(f"Failed to open PDF for Surya OCR: {e}")
        return {}

    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.recognition.model import load_model as load_rec_model

        det_model = load_det_model()
        rec_model = load_rec_model()
    except Exception as e:
        logger.warning(f"Failed to load Surya models: {e}")
        doc.close()
        return {}

    from PIL import Image
    import io

    for page_idx in page_indices:
        try:
            if page_idx >= len(doc):
                logger.warning(f"Page {page_idx} out of range (doc has {len(doc)} pages)")
                results[page_idx] = SuryaPageResult()
                continue

            # Render page to PIL Image
            page = doc[page_idx]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_w, img_h = img.size

            # Run Surya OCR
            ocr_results = run_ocr([img], [det_model, rec_model])

            if not ocr_results or len(ocr_results) == 0:
                results[page_idx] = SuryaPageResult()
                continue

            page_result = ocr_results[0]  # First (only) image result

            text_lines = []
            bboxes = []

            for line in getattr(page_result, 'text_lines', []):
                text_lines.append(getattr(line, 'text', ''))
                bbox = getattr(line, 'bbox', None)
                conf = getattr(line, 'confidence', 0.5)

                if bbox and len(bbox) >= 4:
                    # Convert pixel coords to page-relative (0.0–1.0)
                    x0_rel = max(0.0, min(1.0, bbox[0] / img_w))
                    y0_rel = max(0.0, min(1.0, bbox[1] / img_h))
                    x1_rel = max(0.0, min(1.0, bbox[2] / img_w))
                    y1_rel = max(0.0, min(1.0, bbox[3] / img_h))
                    bboxes.append([x0_rel, y0_rel, x1_rel, y1_rel, float(conf)])

            results[page_idx] = SuryaPageResult(
                text="\n".join(text_lines),
                bboxes=bboxes,
            )

        except Exception as e:
            logger.warning(f"Surya OCR failed on page {page_idx}: {e}")
            results[page_idx] = SuryaPageResult()

    doc.close()
    return results


def build_ocr_bbox_meta(
    surya_results: Dict[int, SuryaPageResult],
    total_pages: int,
    engine: str = "surya",
) -> Dict:
    """
    Build the ocr_bbox_meta summary dict for the analysis payload.

    Args:
        surya_results: Results from extract_with_surya()
        total_pages: Total pages in the PDF
        engine: OCR engine name

    Returns:
        Dict with keys: engine, avg_confidence, pages_with_bbox, pages_total
    """
    all_confs = []
    pages_with_bbox = 0

    for page_idx, result in surya_results.items():
        if result.bboxes:
            pages_with_bbox += 1
            for box in result.bboxes:
                if len(box) >= 5:
                    all_confs.append(box[4])

    avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return {
        "engine": engine,
        "avg_confidence": round(avg_confidence, 2),
        "pages_with_bbox": pages_with_bbox,
        "pages_total": total_pages,
    }
