"""
Detail Specification Extractor - Extract specifications from detail drawings.

Extracts:
- Material specifications
- Dimensions and thicknesses
- Layer sequences (for waterproofing, etc.)
- Brand/product references
- IS code references
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from .classifier import DetailClassification, DetailType


class SpecType(Enum):
    """Type of specification."""
    MATERIAL = "material"
    DIMENSION = "dimension"
    LAYER = "layer"
    BRAND = "brand"
    CODE = "code"
    TREATMENT = "treatment"
    FINISH = "finish"


@dataclass
class DetailSpec:
    """Single specification from a detail."""
    name: str
    value: str
    spec_type: SpecType = SpecType.MATERIAL
    unit: Optional[str] = None
    sequence: Optional[int] = None  # For layered specs
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "spec_type": self.spec_type.value,
            "unit": self.unit,
            "sequence": self.sequence,
            "confidence": self.confidence,
        }


@dataclass
class ExtractedDetail:
    """Extracted detail with specifications."""
    detail_id: str
    detail_type: str
    title: str = ""
    source_page: str = ""
    specs: List[DetailSpec] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    dimensions: Dict[str, str] = field(default_factory=dict)
    is_codes: List[str] = field(default_factory=list)
    brands: List[str] = field(default_factory=list)
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "detail_id": self.detail_id,
            "detail_type": self.detail_type,
            "title": self.title,
            "source_page": self.source_page,
            "specs": [s.to_dict() for s in self.specs],
            "layers": self.layers,
            "materials": self.materials,
            "dimensions": self.dimensions,
            "is_codes": self.is_codes,
            "brands": self.brands,
            "confidence": self.confidence,
        }


class DetailExtractor:
    """Extract specifications from detail pages."""

    # Common India-specific materials
    MATERIALS = [
        # Waterproofing
        r'app\s+membrane',
        r'sbs\s+membrane',
        r'bitumen\s+membrane',
        r'crystalline\s+waterproof',
        r'cementitious\s+coating',
        r'acrylic\s+coating',
        r'polymer\s+modified',
        r'integral\s+waterproof',
        r'elastomeric\s+coating',
        r'polyurethane\s+coating',
        r'epoxy\s+coating',
        r'liquid\s+membrane',

        # Concrete/Masonry
        r'pcc\s+\d+:\d+:\d+',
        r'm\s*\d+\s+grade',
        r'm\d+',
        r'rcc\s+\d+',
        r'aac\s+block',
        r'fly\s*ash\s+brick',
        r'cement\s+block',
        r'solid\s+block',
        r'hollow\s+block',

        # Plaster/Finish
        r'cement\s+plaster',
        r'sand\s+cement\s+plaster',
        r'gypsum\s+plaster',
        r'wall\s+putty',
        r'primer',
        r'acrylic\s+emulsion',
        r'enamel\s+paint',
        r'texture\s+paint',

        # Tiles/Stone
        r'vitrified\s+tile',
        r'ceramic\s+tile',
        r'anti-?skid\s+tile',
        r'acid\s+resistant\s+tile',
        r'kota\s+stone',
        r'granite',
        r'marble',
        r'tandur\s+stone',
        r'shahabad\s+stone',

        # Metal
        r'gi\s+sheet',
        r'ms\s+flat',
        r'ss\s+\d+',
        r'aluminum\s+section',
        r'tmt\s+bar',
        r'fe\s*\d+',

        # Insulation
        r'xps\s+board',
        r'eps\s+board',
        r'polyethylene\s+sheet',
        r'china\s+mosaic',
        r'brick\s+bat\s+coba',
    ]

    # Dimension patterns
    DIMENSION_PATTERNS = [
        (r'(\d+)\s*mm\s+th(?:ick)?', 'thickness'),
        (r'th(?:ickness)?\s*[:=]?\s*(\d+)\s*mm', 'thickness'),
        (r'(\d+)\s*mm\s+thick', 'thickness'),
        (r'height\s*[:=]?\s*(\d+)\s*(?:mm|m)', 'height'),
        (r'width\s*[:=]?\s*(\d+)\s*(?:mm|m)', 'width'),
        (r'depth\s*[:=]?\s*(\d+)\s*(?:mm|m)', 'depth'),
        (r'slope\s*[:=]?\s*1\s*:\s*(\d+)', 'slope'),
        (r'(\d+)\s*mm\s+(?:coving|cove)', 'coving'),
        (r'lap\s*[:=]?\s*(\d+)\s*mm', 'lap'),
        (r'(\d+)\s*mm\s+turnup', 'turnup'),
    ]

    # IS Code patterns
    IS_CODE_PATTERN = r'is\s*[-:]?\s*(\d{3,5})'

    # Brand patterns (India-specific)
    BRAND_PATTERNS = [
        r'dr\.?\s*fixit',
        r'fosroc',
        r'sika',
        r'pidilite',
        r'myk\s+laticrete',
        r'basf',
        r'mapei',
        r'asian\s+paints',
        r'berger',
        r'nerolac',
        r'jk\s+lakshmi',
        r'ultratech',
        r'ambuja',
        r'acc\s+cement',
        r'kajaria',
        r'somany',
        r'johnson',
        r'hindware',
        r'parryware',
        r'jaquar',
    ]

    # Layer sequence indicators
    LAYER_INDICATORS = [
        r'^\s*(\d+)\s*[-.)]\s*(.+)',
        r'^\s*([a-z])\s*[-.)]\s*(.+)',
        r'^\s*[-•]\s*(.+)',
        r'first\s+layer',
        r'second\s+layer',
        r'top\s+layer',
        r'base\s+layer',
        r'primer\s+coat',
        r'finish\s+coat',
    ]

    def __init__(self):
        self.material_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.MATERIALS
        ]
        self.dimension_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.DIMENSION_PATTERNS
        ]
        self.is_code_pattern = re.compile(self.IS_CODE_PATTERN, re.IGNORECASE)
        self.brand_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BRAND_PATTERNS
        ]

    def extract_all(
        self,
        extraction_results: List[Dict],
        classifications: List[DetailClassification],
    ) -> List[ExtractedDetail]:
        """Extract specifications from all classified detail pages."""
        extracted = []

        # Index classifications by sheet
        class_by_sheet = {c.sheet_id: c for c in classifications}

        for result in extraction_results:
            from pathlib import Path
            file_name = Path(result.get("file_path", "")).stem
            page_num = result.get("page_number", 0) + 1
            sheet_id = f"{file_name}_p{page_num}"

            if sheet_id in class_by_sheet:
                classification = class_by_sheet[sheet_id]
                detail = self.extract_from_result(result, classification)
                if detail:
                    extracted.append(detail)

        return extracted

    def extract_from_result(
        self,
        result: Dict,
        classification: DetailClassification,
    ) -> Optional[ExtractedDetail]:
        """Extract specifications from a single result."""
        all_text = self._collect_text(result)

        detail = ExtractedDetail(
            detail_id=f"{classification.sheet_id}_{classification.detail_type.value}",
            detail_type=classification.detail_type.value,
            title=classification.title,
            source_page=classification.sheet_id,
        )

        # Extract materials
        detail.materials = self._extract_materials(all_text)

        # Extract dimensions
        detail.dimensions = self._extract_dimensions(all_text)

        # Extract layers
        detail.layers = self._extract_layers(all_text)

        # Extract IS codes
        detail.is_codes = self._extract_is_codes(all_text)

        # Extract brands
        detail.brands = self._extract_brands(all_text)

        # Build specs list
        detail.specs = self._build_specs(detail)

        # Calculate confidence
        evidence_count = (
            len(detail.materials) +
            len(detail.dimensions) +
            len(detail.layers)
        )
        detail.confidence = min(0.95, 0.5 + 0.1 * evidence_count)

        return detail

    def _collect_text(self, result: Dict) -> str:
        """Collect all text from result."""
        texts = []

        # Text items
        for item in result.get("text_items", []):
            texts.append(item.get("text", ""))

        # Notes
        for note in result.get("notes", []):
            if isinstance(note, dict):
                texts.append(note.get("text", ""))
            else:
                texts.append(str(note))

        # Annotations
        for ann in result.get("annotations", []):
            if isinstance(ann, dict):
                texts.append(ann.get("text", ""))
            else:
                texts.append(str(ann))

        return "\n".join(filter(None, texts))

    def _extract_materials(self, text: str) -> List[str]:
        """Extract material specifications."""
        materials = []

        for pattern in self.material_patterns:
            matches = pattern.findall(text)
            materials.extend(matches)

        # Also look for "grade" specifications
        grade_matches = re.findall(r'(m\s*\d+|fe\s*\d+)\s+grade', text, re.IGNORECASE)
        materials.extend(grade_matches)

        return list(set(materials))[:15]

    def _extract_dimensions(self, text: str) -> Dict[str, str]:
        """Extract dimension specifications."""
        dimensions = {}

        for pattern, dim_type in self.dimension_patterns:
            matches = pattern.findall(text)
            if matches:
                # Take first match or combine if multiple
                if isinstance(matches[0], tuple):
                    dimensions[dim_type] = matches[0][0]
                else:
                    dimensions[dim_type] = matches[0]

        return dimensions

    def _extract_layers(self, text: str) -> List[str]:
        """Extract layer sequence."""
        layers = []

        # Split into lines and look for numbered/bulleted items
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for numbered/lettered items
            match = re.match(r'^\s*(\d+|[a-z])\s*[-.)]\s*(.+)', line, re.IGNORECASE)
            if match:
                layer_text = match.group(2).strip()
                if len(layer_text) > 5 and len(layer_text) < 200:
                    layers.append(layer_text)
            # Check for bullet points
            elif line.startswith(('-', '•', '*')):
                layer_text = line.lstrip('-•* ').strip()
                if len(layer_text) > 5 and len(layer_text) < 200:
                    layers.append(layer_text)

        return layers[:20]  # Limit

    def _extract_is_codes(self, text: str) -> List[str]:
        """Extract IS code references."""
        matches = self.is_code_pattern.findall(text)
        return [f"IS {m}" for m in set(matches)]

    def _extract_brands(self, text: str) -> List[str]:
        """Extract brand references."""
        brands = []

        for pattern in self.brand_patterns:
            matches = pattern.findall(text)
            brands.extend(matches)

        return list(set(brands))[:10]

    def _build_specs(self, detail: ExtractedDetail) -> List[DetailSpec]:
        """Build specifications list from extracted data."""
        specs = []

        # Add materials
        for i, material in enumerate(detail.materials):
            specs.append(DetailSpec(
                name=f"material_{i+1}",
                value=material,
                spec_type=SpecType.MATERIAL,
                confidence=0.85,
            ))

        # Add dimensions
        for dim_type, value in detail.dimensions.items():
            unit = "mm"
            if "slope" in dim_type:
                unit = "ratio"
            specs.append(DetailSpec(
                name=dim_type,
                value=value,
                spec_type=SpecType.DIMENSION,
                unit=unit,
                confidence=0.9,
            ))

        # Add layers
        for i, layer in enumerate(detail.layers):
            specs.append(DetailSpec(
                name=f"layer_{i+1}",
                value=layer,
                spec_type=SpecType.LAYER,
                sequence=i + 1,
                confidence=0.75,
            ))

        # Add IS codes
        for code in detail.is_codes:
            specs.append(DetailSpec(
                name="is_code",
                value=code,
                spec_type=SpecType.CODE,
                confidence=0.95,
            ))

        # Add brands
        for brand in detail.brands:
            specs.append(DetailSpec(
                name="brand",
                value=brand,
                spec_type=SpecType.BRAND,
                confidence=0.8,
            ))

        return specs

    def get_waterproofing_spec_summary(self, detail: ExtractedDetail) -> Dict:
        """Get summary for waterproofing detail."""
        summary = {
            "type": detail.detail_type,
            "treatment_layers": len(detail.layers),
            "membrane_type": None,
            "thickness": detail.dimensions.get("thickness"),
            "slope": detail.dimensions.get("slope"),
            "turnup": detail.dimensions.get("turnup"),
            "brand": detail.brands[0] if detail.brands else None,
        }

        # Detect membrane type
        for material in detail.materials:
            material_lower = material.lower()
            if "app" in material_lower:
                summary["membrane_type"] = "APP membrane"
            elif "sbs" in material_lower:
                summary["membrane_type"] = "SBS membrane"
            elif "liquid" in material_lower or "cementitious" in material_lower:
                summary["membrane_type"] = "Cementitious coating"
            elif "crystalline" in material_lower:
                summary["membrane_type"] = "Crystalline waterproofing"

        return summary
