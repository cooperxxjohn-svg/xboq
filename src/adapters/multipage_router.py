"""
Multipage Router - Enhanced page classification with drawing-likeness scoring.

This router:
1. Iterates ALL pages in multi-page PDFs
2. Computes a "drawing-likeness score" for each page
3. Classifies pages as candidate_plan if they have drawing characteristics
4. Outputs routing_debug.csv for full transparency
"""

import re
import csv
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PageType(Enum):
    """Classification of drawing page types."""
    FLOOR_PLAN = "floor_plan"
    STRUCTURAL = "structural"
    SECTION = "section"
    ELEVATION = "elevation"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    SITE_PLAN = "site_plan"
    SCHEDULE = "schedule"
    TITLE_SHEET = "title_sheet"
    COVER = "cover"
    CANDIDATE_PLAN = "candidate_plan"  # Has drawing characteristics but unknown type
    UNKNOWN = "unknown"


@dataclass
class PageScore:
    """Drawing-likeness score for a page."""
    page_num: int
    total_score: float
    vector_line_score: float  # Score from vector line density
    edge_density_score: float  # Score from raster edge density
    title_block_score: float  # Score from title block keywords
    dimension_score: float  # Score from dimension-like text
    scale_note_score: float  # Score from scale notation
    is_candidate: bool
    page_type: PageType
    reason: str
    text_sample: str = ""
    scale_detected: Optional[str] = None
    dimensions_found: List[str] = field(default_factory=list)


# Title block keywords that indicate a real drawing
TITLE_BLOCK_KEYWORDS = [
    r"drawing", r"scale", r"date", r"sheet", r"rev\.?", r"revision",
    r"drg\.?\s*no", r"dwg\.?\s*no", r"checked", r"approved",
    r"project", r"client", r"architect", r"engineer", r"contractor",
    r"north", r"grid", r"level", r"floor", r"section", r"elevation",
    r"plan", r"layout", r"detail", r"schedule"
]

# Dimension-like patterns (numbers that look like dimensions in mm)
DIMENSION_PATTERNS = [
    r"\b\d{3,5}\b",  # 3-5 digit numbers (100-99999 mm range)
    r"\b\d+\s*mm\b",  # Explicit mm
    r"\b\d+\s*[mx]\s*\d+\b",  # WxH format
    r"\b\d+'\s*-?\s*\d+\"",  # Feet-inches
]

# Scale notation patterns
SCALE_PATTERNS = [
    r"scale\s*[:\-]?\s*1\s*[:\-]\s*(\d+)",  # Scale: 1:100
    r"1\s*[:\-]\s*(\d+)\s*scale",  # 1:100 scale
    r"@\s*1\s*[:\-]\s*(\d+)",  # @1:100
    r"sc\.\s*1\s*[:\-]\s*(\d+)",  # SC. 1:100
]


class MultipageRouter:
    """
    Routes all pages in a drawing set, computing drawing-likeness scores.
    """

    def __init__(self, candidate_threshold: float = 0.3):
        """
        Args:
            candidate_threshold: Minimum score to be considered a candidate_plan
        """
        self.candidate_threshold = candidate_threshold
        self.compiled_title_keywords = [
            re.compile(kw, re.IGNORECASE) for kw in TITLE_BLOCK_KEYWORDS
        ]
        self.compiled_dimension_patterns = [
            re.compile(p, re.IGNORECASE) for p in DIMENSION_PATTERNS
        ]
        self.compiled_scale_patterns = [
            re.compile(p, re.IGNORECASE) for p in SCALE_PATTERNS
        ]

    def compute_page_score(self, page: fitz.Page, page_num: int) -> PageScore:
        """
        Compute drawing-likeness score for a single page.

        Args:
            page: PyMuPDF page object
            page_num: 0-indexed page number

        Returns:
            PageScore with detailed scoring breakdown
        """
        # Extract text
        text = page.get_text("text")
        text_lower = text.lower()

        # 1. Vector line score (from drawings/paths)
        try:
            drawings = page.get_drawings()
            line_count = sum(len(d.get("items", [])) for d in drawings)
            path_count = len(drawings)
            # Normalize: 100+ lines = 1.0, 0 = 0.0
            vector_line_score = min(1.0, line_count / 100) * 0.5 + min(1.0, path_count / 50) * 0.5
        except Exception:
            vector_line_score = 0.0
            line_count = 0

        # 2. Edge density score (for raster/scanned pages)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))  # Low res for speed
            samples = pix.samples
            # Simple edge detection: count pixels that differ from neighbors
            # This is a rough approximation
            edge_count = 0
            if len(samples) > 100:
                for i in range(1, len(samples) - 1, 3):
                    diff = abs(samples[i] - samples[i-3]) + abs(samples[i] - samples[i+3])
                    if diff > 50:
                        edge_count += 1
            edge_density_score = min(1.0, edge_count / 10000)
        except Exception:
            edge_density_score = 0.0

        # 3. Title block keyword score
        title_matches = 0
        for pattern in self.compiled_title_keywords:
            if pattern.search(text_lower):
                title_matches += 1
        title_block_score = min(1.0, title_matches / 5)  # 5+ keywords = 1.0

        # 4. Dimension-like text score
        dimensions_found = []
        for pattern in self.compiled_dimension_patterns:
            matches = pattern.findall(text)
            dimensions_found.extend(matches[:10])  # Limit to first 10
        dimension_score = min(1.0, len(dimensions_found) / 10)  # 10+ dimensions = 1.0

        # 5. Scale note detection
        scale_detected = None
        scale_note_score = 0.0
        for pattern in self.compiled_scale_patterns:
            match = pattern.search(text)
            if match:
                scale_detected = f"1:{match.group(1)}"
                scale_note_score = 1.0
                break

        # Calculate total score (weighted)
        total_score = (
            vector_line_score * 0.25 +
            edge_density_score * 0.15 +
            title_block_score * 0.20 +
            dimension_score * 0.25 +
            scale_note_score * 0.15
        )

        # Determine if candidate
        is_candidate = total_score >= self.candidate_threshold

        # Determine page type
        page_type = self._classify_page_type(text_lower, is_candidate)

        # Build reason string
        reasons = []
        if vector_line_score > 0.3:
            reasons.append(f"vectors({line_count})")
        if title_block_score > 0.3:
            reasons.append(f"title_kw({title_matches})")
        if dimension_score > 0.3:
            reasons.append(f"dims({len(dimensions_found)})")
        if scale_detected:
            reasons.append(f"scale({scale_detected})")

        reason = ", ".join(reasons) if reasons else "low_score"

        # Text sample for debugging
        text_sample = text[:200].replace("\n", " ").strip()

        return PageScore(
            page_num=page_num,
            total_score=total_score,
            vector_line_score=vector_line_score,
            edge_density_score=edge_density_score,
            title_block_score=title_block_score,
            dimension_score=dimension_score,
            scale_note_score=scale_note_score,
            is_candidate=is_candidate,
            page_type=page_type,
            reason=reason,
            text_sample=text_sample,
            scale_detected=scale_detected,
            dimensions_found=dimensions_found[:5],
        )

    def _classify_page_type(self, text_lower: str, is_candidate: bool) -> PageType:
        """Classify page type based on text content."""
        # Specific type patterns
        patterns = {
            PageType.FLOOR_PLAN: [r"floor\s*plan", r"ground\s*floor", r"first\s*floor", r"layout\s*plan"],
            PageType.STRUCTURAL: [r"structural", r"rcc", r"foundation", r"beam\s*layout", r"column"],
            PageType.SECTION: [r"section\s*[a-z]", r"cross\s*section", r"longitudinal"],
            PageType.ELEVATION: [r"elevation", r"front\s*view", r"rear\s*view", r"side\s*view"],
            PageType.ELECTRICAL: [r"electrical", r"wiring", r"lighting\s*layout", r"power\s*layout"],
            PageType.PLUMBING: [r"plumbing", r"water\s*supply", r"drainage", r"sanitary"],
            PageType.SITE_PLAN: [r"site\s*plan", r"plot\s*plan", r"location\s*plan"],
            PageType.SCHEDULE: [r"schedule", r"door.*window", r"finish\s*schedule"],
            PageType.TITLE_SHEET: [r"title\s*sheet", r"index", r"list\s*of\s*drawings"],
            PageType.COVER: [r"tender", r"contract", r"bid\s*document"],
        }

        for page_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                if re.search(pattern, text_lower):
                    return page_type

        if is_candidate:
            return PageType.CANDIDATE_PLAN

        return PageType.UNKNOWN

    def route_pdf(self, pdf_path: Path) -> Tuple[List[PageScore], Dict[str, Any]]:
        """
        Route all pages in a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (list of PageScores, summary dict)
        """
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        scores = []
        for page_num in range(total_pages):
            page = doc[page_num]
            score = self.compute_page_score(page, page_num)
            scores.append(score)

            logger.info(
                f"Page {page_num + 1}/{total_pages}: "
                f"score={score.total_score:.2f}, type={score.page_type.value}, "
                f"candidate={score.is_candidate}, reason={score.reason}"
            )

        doc.close()

        # Summary
        candidates = [s for s in scores if s.is_candidate]
        by_type = {}
        for s in scores:
            t = s.page_type.value
            by_type[t] = by_type.get(t, 0) + 1

        summary = {
            "total_pages": total_pages,
            "candidates": len(candidates),
            "by_type": by_type,
            "scales_detected": [s.scale_detected for s in scores if s.scale_detected],
        }

        return scores, summary

    def route_directory(self, drawings_dir: Path, output_dir: Path = None) -> Dict[str, Any]:
        """
        Route all drawings in a directory.

        Args:
            drawings_dir: Path to drawings folder
            output_dir: Optional output dir for routing_debug.csv

        Returns:
            Routing result with all page scores
        """
        drawings_dir = Path(drawings_dir)

        # Find all files
        patterns = ["*.pdf", "*.PDF"]
        pdf_files = []
        for pattern in patterns:
            pdf_files.extend(drawings_dir.glob(pattern))

        all_scores = []
        all_summaries = []

        for pdf_path in sorted(pdf_files):
            logger.info(f"Routing {pdf_path.name}...")
            scores, summary = self.route_pdf(pdf_path)

            for score in scores:
                score.source_file = pdf_path.name

            all_scores.extend(scores)
            all_summaries.append({
                "file": pdf_path.name,
                **summary
            })

        # Write routing_debug.csv if output_dir provided
        if output_dir:
            self._write_routing_debug(all_scores, Path(output_dir))

        # Build result
        candidates = [s for s in all_scores if s.is_candidate]
        by_type = {}
        for s in all_scores:
            t = s.page_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "routed": True,
            "total_pages": len(all_scores),
            "candidates": len(candidates),
            "types": by_type,
            "scores": all_scores,
            "file_summaries": all_summaries,
            "scales_detected": list(set(s.scale_detected for s in all_scores if s.scale_detected)),
        }

    def _write_routing_debug(self, scores: List[PageScore], output_dir: Path):
        """Write routing_debug.csv with full page analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "routing_debug.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file", "page", "total_score", "vector_score", "edge_score",
                "title_score", "dim_score", "scale_score", "is_candidate",
                "page_type", "scale_detected", "reason", "text_sample"
            ])

            for s in scores:
                writer.writerow([
                    getattr(s, 'source_file', ''),
                    s.page_num + 1,  # 1-indexed for human readability
                    f"{s.total_score:.3f}",
                    f"{s.vector_line_score:.3f}",
                    f"{s.edge_density_score:.3f}",
                    f"{s.title_block_score:.3f}",
                    f"{s.dimension_score:.3f}",
                    f"{s.scale_note_score:.3f}",
                    "YES" if s.is_candidate else "NO",
                    s.page_type.value,
                    s.scale_detected or "",
                    s.reason,
                    s.text_sample[:100],
                ])

        logger.info(f"Wrote routing debug to {csv_path}")


def route_multipage_project(drawings_dir: Path, output_dir: Path = None) -> Dict[str, Any]:
    """
    Convenience function to route all pages in a project.

    Args:
        drawings_dir: Drawings directory
        output_dir: Output directory for debug files

    Returns:
        Routing result
    """
    router = MultipageRouter()
    return router.route_directory(drawings_dir, output_dir)
