"""
Foundation Plan Extractor
Specialized extraction for Indian foundation plans.
Handles feet-inch dimensions and standard footing schedules.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from pathlib import Path

from .units import (
    feet_inch_to_mm, parse_footing_size, extract_footing_label,
    extract_column_label, FOOTING_TYPES
)
from .detect_columns import DetectedColumn
from .detect_footings import DetectedFooting, FootingDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class FoundationPlanData:
    """Extracted data from foundation plan."""
    # Columns
    columns: List[DetectedColumn] = field(default_factory=list)
    column_count: int = 0

    # Footings
    footings: List[DetectedFooting] = field(default_factory=list)
    footing_types: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # label -> (L, W) mm

    # Column-footing mapping
    column_footing_map: Dict[str, str] = field(default_factory=dict)  # C1 -> F1

    # Tie beams
    tie_beams: List[Dict] = field(default_factory=list)

    # Drawing info
    scale: str = "1:100"
    concrete_grade: str = "M25"
    steel_grade: str = "Fe500"
    soil_bearing: Optional[float] = None  # t/sqm
    foundation_depth: Optional[float] = None  # mm below FFL

    # Notes
    notes: List[str] = field(default_factory=list)

    # Confidence
    confidence: float = 0.5


class FoundationPlanExtractor:
    """
    Extracts structural data from foundation plans.
    Optimized for Indian construction drawing conventions.
    """

    def __init__(self):
        """Initialize extractor."""
        # Regex patterns for common elements
        self.column_pattern = re.compile(r'\bC(\d+)\b', re.IGNORECASE)
        self.footing_pattern = re.compile(r'\b(F\d+[A-Z]?)\b', re.IGNORECASE)
        self.size_pattern = re.compile(
            r"(\d+)[\''][\-\s]?(\d+)[\"\"']?\s*[x×]?\s*(\d+)?[\'']?[\-\s]?(\d+)?[\"\"']?",
            re.IGNORECASE
        )

    def extract(
        self,
        image: np.ndarray,
        vector_texts: List[Any] = None,
        pdf_text: str = ""
    ) -> FoundationPlanData:
        """
        Extract foundation plan data.

        Args:
            image: Drawing image
            vector_texts: Text blocks from PDF
            pdf_text: Full text content

        Returns:
            FoundationPlanData
        """
        logger.info("Extracting foundation plan data...")

        result = FoundationPlanData()

        # Combine text sources
        all_text = pdf_text
        if vector_texts:
            for vt in vector_texts:
                text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
                all_text += " " + text

        # Extract drawing metadata
        result.scale = self._extract_scale(all_text)
        result.concrete_grade = self._extract_concrete_grade(all_text)
        result.steel_grade = self._extract_steel_grade(all_text)
        result.soil_bearing = self._extract_soil_bearing(all_text)
        result.foundation_depth = self._extract_foundation_depth(all_text)
        result.notes = self._extract_notes(all_text)

        # Extract columns
        result.columns, result.column_count = self._extract_columns(all_text, vector_texts, image)

        # Extract footings with sizes
        result.footings, result.footing_types = self._extract_footings(all_text, vector_texts, image)

        # Build column-footing mapping
        result.column_footing_map = self._build_column_footing_map(
            result.columns, result.footings, vector_texts
        )

        # Extract tie beams
        result.tie_beams = self._extract_tie_beams(all_text, vector_texts)

        # Calculate confidence
        result.confidence = self._calculate_confidence(result)

        logger.info(f"Extracted {result.column_count} columns, "
                   f"{len(result.footings)} footings, "
                   f"confidence: {result.confidence:.0%}")

        return result

    def _extract_scale(self, text: str) -> str:
        """Extract drawing scale."""
        patterns = [
            r'SCALE[:\s]+1[\s]*:[\s]*(\d+)',
            r'1[\s]*:[\s]*(\d+)\s*(?:SCALE)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"1:{match.group(1)}"
        return "1:100"  # Default for Indian structural drawings

    def _extract_concrete_grade(self, text: str) -> str:
        """Extract concrete grade."""
        match = re.search(r'M[\s]?(\d{2})', text, re.IGNORECASE)
        if match:
            return f"M{match.group(1)}"
        return "M25"

    def _extract_steel_grade(self, text: str) -> str:
        """Extract steel grade."""
        text_upper = text.upper()
        if 'FE 500' in text_upper or 'FE500' in text_upper:
            return "Fe500"
        elif 'FE 415' in text_upper or 'FE415' in text_upper:
            return "Fe415"
        return "Fe500"

    def _extract_soil_bearing(self, text: str) -> Optional[float]:
        """Extract safe bearing capacity."""
        patterns = [
            r'BEARING\s+CAPACITY[^0-9]*(\d+\.?\d*)\s*T/SQ\.?M',
            r'SBC[^0-9]*(\d+\.?\d*)\s*T/SQ\.?M',
            r'(\d+\.?\d*)\s*T/SQ\.?M',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_foundation_depth(self, text: str) -> Optional[float]:
        """Extract foundation depth below FFL."""
        # Look for patterns like "4'-0" BELOW BASEMENT FFL"
        pattern = r"(\d+)[\''][\-]?(\d+)[\"\"']?\s*(?:BELOW|B/W|BGL)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            feet, inches = int(match.group(1)), int(match.group(2))
            return (feet * 12 + inches) * 25.4
        return None

    def _extract_notes(self, text: str) -> List[str]:
        """Extract important notes."""
        notes = []

        # Look for numbered notes
        note_matches = re.findall(r'\d+\.\s+([A-Z][^.]+\.)', text)
        for note in note_matches:
            if len(note) > 20 and len(note) < 500:
                notes.append(note.strip())

        return notes[:10]  # Limit to 10 notes

    def _extract_columns(
        self,
        text: str,
        vector_texts: List[Any],
        image: np.ndarray
    ) -> Tuple[List[DetectedColumn], int]:
        """Extract column information."""
        columns = []
        column_labels = set()

        # Find all column labels in text
        matches = self.column_pattern.findall(text)
        for m in matches:
            label = f"C{m}"
            column_labels.add(label)

        # Get positions from vector texts if available
        if vector_texts:
            for vt in vector_texts:
                vt_text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
                bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))

                match = self.column_pattern.search(vt_text)
                if match:
                    label = f"C{match.group(1)}"
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                    columns.append(DetectedColumn(
                        column_id=label,
                        label=label,
                        bbox=(int(bbox[0]), int(bbox[1]),
                              int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                        center=center,
                        polygon=[],
                        confidence=0.8,
                        source="text_extraction"
                    ))

        # If no positioned columns found, create from labels
        if not columns:
            for i, label in enumerate(sorted(column_labels)):
                columns.append(DetectedColumn(
                    column_id=label,
                    label=label,
                    bbox=(0, 0, 0, 0),
                    center=(0, 0),
                    polygon=[],
                    confidence=0.5,
                    source="text_extraction"
                ))

        return columns, len(column_labels)

    def _extract_footings(
        self,
        text: str,
        vector_texts: List[Any],
        image: np.ndarray
    ) -> Tuple[List[DetectedFooting], Dict[str, Tuple[float, float]]]:
        """Extract footing information with sizes."""
        footings = []
        footing_types = {}

        # Find all footing labels
        footing_labels = set()
        matches = self.footing_pattern.findall(text)
        for m in matches:
            footing_labels.add(m.upper())

        # Extract footing sizes from annotations
        # Pattern: "F1 6'-6" 6'-6"" or "6'-6" 6'-6" F1"
        for label in footing_labels:
            # Check if it's a standard type
            if label in FOOTING_TYPES:
                footing_types[label] = FOOTING_TYPES[label]['mm']
            else:
                # Try to find size near the label in text
                size = self._find_footing_size(label, text, vector_texts)
                if size:
                    footing_types[label] = size

        # Get positions from vector texts
        positioned_footings = {}
        if vector_texts:
            for vt in vector_texts:
                vt_text = vt.text if hasattr(vt, 'text') else vt.get('text', '')
                bbox = vt.bbox if hasattr(vt, 'bbox') else vt.get('bbox', (0, 0, 0, 0))

                match = self.footing_pattern.search(vt_text)
                if match:
                    label = match.group(1).upper()
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                    if label not in positioned_footings:
                        positioned_footings[label] = {
                            'label': label,
                            'center': center,
                            'bbox': bbox
                        }

        # Create footing objects
        for label in sorted(footing_labels):
            size_mm = footing_types.get(label)

            if label in positioned_footings:
                pf = positioned_footings[label]
                bbox = pf['bbox']
                ftg = DetectedFooting(
                    footing_id=label,
                    label=label,
                    footing_type="isolated",
                    bbox=(int(bbox[0]), int(bbox[1]),
                          int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                    center=pf['center'],
                    polygon=[],
                    size_mm=(int(size_mm[0]), int(size_mm[1]), 450) if size_mm else None,
                    confidence=0.8 if size_mm else 0.5,
                    source="text_extraction"
                )
            else:
                ftg = DetectedFooting(
                    footing_id=label,
                    label=label,
                    footing_type="isolated",
                    bbox=(0, 0, 0, 0),
                    center=(0, 0),
                    polygon=[],
                    size_mm=(int(size_mm[0]), int(size_mm[1]), 450) if size_mm else None,
                    confidence=0.6 if size_mm else 0.4,
                    source="text_extraction"
                )

            footings.append(ftg)

        return footings, footing_types

    def _find_footing_size(
        self,
        label: str,
        text: str,
        vector_texts: List[Any]
    ) -> Optional[Tuple[float, float]]:
        """Find footing size near label in text."""
        # Pattern for size annotation: "6'-6" 6'-6"" or "6'-6" x 6'-6""
        size_pattern = r"(\d+)[\''][\-\s]?(\d+)[\"\"']?\s*[x×]?\s*(\d+)[\''][\-\s]?(\d+)[\"\"']?"

        # Look for label followed by size
        pattern1 = rf'{label}\s+{size_pattern}'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            f1, i1, f2, i2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return ((f1 * 12 + i1) * 25.4, (f2 * 12 + i2) * 25.4)

        # Look for size followed by label
        pattern2 = rf'{size_pattern}\s*{label}'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            f1, i1, f2, i2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return ((f1 * 12 + i1) * 25.4, (f2 * 12 + i2) * 25.4)

        return None

    def _build_column_footing_map(
        self,
        columns: List[DetectedColumn],
        footings: List[DetectedFooting],
        vector_texts: List[Any]
    ) -> Dict[str, str]:
        """Build mapping of columns to footings."""
        mapping = {}

        # If we have positions, use proximity
        if any(c.center != (0, 0) for c in columns) and any(f.center != (0, 0) for f in footings):
            for col in columns:
                if col.center == (0, 0):
                    continue

                nearest_footing = None
                min_dist = float('inf')

                for ftg in footings:
                    if ftg.center == (0, 0):
                        continue

                    dist = np.sqrt(
                        (col.center[0] - ftg.center[0])**2 +
                        (col.center[1] - ftg.center[1])**2
                    )

                    if dist < min_dist:
                        min_dist = dist
                        nearest_footing = ftg

                if nearest_footing:
                    mapping[col.label] = nearest_footing.label

        return mapping

    def _extract_tie_beams(
        self,
        text: str,
        vector_texts: List[Any]
    ) -> List[Dict]:
        """Extract tie beam information."""
        tie_beams = []

        # Pattern: TB1 1'-6" x 1'-0" or TIE BEAM TB1
        tb_pattern = r"(?:TIE\s*BEAM\s*)?(TB\d+)\s*(\d+)['\"][\-\s]?(\d+)['\"]?\s*[x×]\s*(\d+)['\"][\-\s]?(\d+)['\"]?"
        matches = re.findall(tb_pattern, text, re.IGNORECASE)

        for match in matches:
            label = match[0].upper()
            width_mm = (int(match[1]) * 12 + int(match[2])) * 25.4
            depth_mm = (int(match[3]) * 12 + int(match[4])) * 25.4

            tie_beams.append({
                'label': label,
                'width_mm': width_mm,
                'depth_mm': depth_mm
            })

        # Also look for reinforcement details
        rebar_pattern = r'(TB\d+)[^Y]*(\d+)\-Y(\d+)'
        rebar_matches = re.findall(rebar_pattern, text, re.IGNORECASE)
        for rm in rebar_matches:
            label = rm[0].upper()
            # Find existing beam and add rebar info
            for tb in tie_beams:
                if tb['label'] == label:
                    tb['main_bars'] = int(rm[1])
                    tb['bar_dia'] = int(rm[2])

        return tie_beams

    def _calculate_confidence(self, result: FoundationPlanData) -> float:
        """Calculate overall extraction confidence."""
        conf = 0.0

        # Columns found
        if result.column_count > 0:
            conf += 0.2
            if result.column_count > 10:
                conf += 0.1

        # Footings with sizes
        sized_footings = len([f for f in result.footings if f.size_mm])
        if sized_footings > 0:
            conf += 0.2
            if sized_footings == len(result.footings):
                conf += 0.1

        # Column-footing mapping
        if result.column_footing_map:
            conf += 0.15

        # Metadata extracted
        if result.concrete_grade != "M25":  # Non-default
            conf += 0.05
        if result.soil_bearing:
            conf += 0.1
        if result.foundation_depth:
            conf += 0.1

        return min(0.95, conf)


def extract_foundation_plan(
    image: np.ndarray,
    vector_texts: List[Any] = None,
    pdf_text: str = ""
) -> FoundationPlanData:
    """Convenience function to extract foundation plan data."""
    extractor = FoundationPlanExtractor()
    return extractor.extract(image, vector_texts, pdf_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample text
    sample_text = """
    FOUNDATION PLAN
    SCALE: 1:100

    CONCRETE GRADE - M 25 (1 : 1 : 2)
    REINFORCEMENT - TMT FE 500

    NET SAFE BEARING CAPACITY OF SOIL IS TAKEN 17.0 T/SQ.M.

    ALL FOUNDATIONS SHALL BE RESTING 4'-0" BELOW BASEMENT FFL

    C1 C2 C3 C4 C5 C6 C7 C8 C9 C10
    C11 C12 C13 C14 C15 C16 C17 C18 C19 C20
    C21 C22 C23 C24 C25 C26 C27 C28 C29 C30
    C31 C32 C33 C34 C35 C36 C37 C38 C39 C40
    C41 C42 C43 C44 C45

    F1 7'-0" 7'-0"
    F2 6'-6" 6'-6"
    F3 7'-0" 5'-9"
    F4 6'-0" 6'-0"
    F5 5'-7" 5'-7"
    F6 5'-3" 5'-3"
    F8 5'-0" 5'-0"
    F9 4'-6" 4'-6"
    F10 4'-0" 4'-0"

    TIE BEAM TB1 1'-6" x 1'-0"
    3-Y16
    Y8@200C/C
    """

    result = extract_foundation_plan(np.zeros((100, 100, 3), dtype=np.uint8), pdf_text=sample_text)

    print(f"\nFoundation Plan Extraction:")
    print(f"  Scale: {result.scale}")
    print(f"  Concrete: {result.concrete_grade}")
    print(f"  Steel: {result.steel_grade}")
    print(f"  SBC: {result.soil_bearing} t/sqm")
    print(f"  Foundation depth: {result.foundation_depth} mm")
    print(f"  Columns: {result.column_count}")
    print(f"  Footings: {len(result.footings)}")
    print(f"  Footing types: {len(result.footing_types)}")
    print(f"  Tie beams: {result.tie_beams}")
    print(f"  Confidence: {result.confidence:.0%}")

    print(f"\nFooting sizes:")
    for label, size in sorted(result.footing_types.items()):
        print(f"  {label}: {size[0]:.0f} x {size[1]:.0f} mm")
