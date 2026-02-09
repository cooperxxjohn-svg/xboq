"""
Requirements Extractor Module
Extracts construction requirements from OCR notes and legend blocks.

Detects and classifies:
- Concrete grade (M20/M25/M30)
- Rebar grade (Fe500/Fe550D)
- Cover (25/40/50mm)
- Exposure class, waterproofing, anti-termite, DPC
- Codes mentioned (IS456, IS1786, CPWD, NBC)
- QA/QC tests (slump test, cube test, etc.)
- Execution requirements (curing, joints, compaction)
"""

import re
from typing import List, Optional, Dict, Any
from .estimator_output import Requirement
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Concrete grades
CONCRETE_PATTERNS = [
    (r'\bM[\s]?(\d{2})\b', 'Material', 'Concrete grade'),
    (r'\b(M20|M25|M30|M35|M40|M45|M50)\b', 'Material', 'Concrete grade'),
    (r'CONCRETE\s+GRADE[:\s]+M?(\d{2})', 'Material', 'Concrete grade'),
    (r'RCC\s+M(\d{2})', 'Material', 'Concrete grade'),
]

# Steel grades
STEEL_PATTERNS = [
    (r'\b(Fe[\s]?500|Fe[\s]?415|Fe[\s]?550D?)\b', 'Material', 'Reinforcement grade'),
    (r'\b(TMT[\s]?Fe[\s]?500|TMT[\s]?Fe[\s]?415)\b', 'Material', 'Reinforcement grade'),
    (r'STEEL\s+GRADE[:\s]+(Fe[\s]?\d+D?)', 'Material', 'Reinforcement grade'),
    (r'\b(HSD|CRS|TMT)\s+(Fe[\s]?\d+)', 'Material', 'Reinforcement grade'),
]

# Cover requirements
COVER_PATTERNS = [
    (r'COVER[:\s]+(\d+)\s*(?:MM|mm)', 'Material', 'Cover'),
    (r'CLEAR\s+COVER[:\s]+(\d+)', 'Material', 'Clear cover'),
    (r'(\d+)\s*(?:MM|mm)\s+COVER', 'Material', 'Cover'),
    (r'NOMINAL\s+COVER[:\s]+(\d+)', 'Material', 'Nominal cover'),
]

# Exposure class
EXPOSURE_PATTERNS = [
    (r'\b(MILD|MODERATE|SEVERE|VERY\s+SEVERE|EXTREME)\s+EXPOSURE', 'Material', 'Exposure condition'),
    (r'EXPOSURE\s+(?:CLASS|CONDITION)[:\s]+(MILD|MODERATE|SEVERE)', 'Material', 'Exposure class'),
]

# Codes and standards
CODE_PATTERNS = [
    (r'\b(IS[\s:]?456(?:[\s:-]?\d{4})?)\b', 'Code', 'Design code'),
    (r'\b(IS[\s:]?1786(?:[\s:-]?\d{4})?)\b', 'Code', 'Steel standard'),
    (r'\b(IS[\s:]?13920(?:[\s:-]?\d{4})?)\b', 'Code', 'Seismic code'),
    (r'\b(IS[\s:]?875(?:[\s:-]?\d{4})?)\b', 'Code', 'Loading code'),
    (r'\b(IS[\s:]?2502(?:[\s:-]?\d{4})?)\b', 'Code', 'Bending code'),
    (r'\b(CPWD)\b', 'Code', 'CPWD specifications'),
    (r'\b(NBC[\s:]?\d{4})\b', 'Code', 'National building code'),
    (r'\b(ACI[\s:]?\d+)\b', 'Code', 'ACI standard'),
]

# QA/QC tests
QC_PATTERNS = [
    (r'SLUMP\s+(?:TEST)?[:\s]*(\d+[-–]\d+|\d+)\s*(?:MM|mm)?', 'QA/QC', 'Slump test'),
    (r'CUBE\s+(?:TEST|STRENGTH)', 'QA/QC', 'Cube test required'),
    (r'(\d+)\s+CUBES?\s+(?:PER|FOR|EACH)', 'QA/QC', 'Cube sampling'),
    (r'7\s*(?:DAY|DAYS?)\s+(?:STRENGTH|TEST)', 'QA/QC', '7-day strength test'),
    (r'28\s*(?:DAY|DAYS?)\s+(?:STRENGTH|TEST)', 'QA/QC', '28-day strength test'),
    (r'REBOUND\s+HAMMER', 'QA/QC', 'Rebound hammer test'),
    (r'ULTRASONIC\s+(?:TEST|PULSE)', 'QA/QC', 'Ultrasonic test'),
]

# Execution requirements
EXECUTION_PATTERNS = [
    (r'CURING[:\s]+(\d+)\s*(?:DAYS?|D)', 'Execution', 'Curing period'),
    (r'(\d+)\s*(?:DAYS?)\s+(?:WET\s+)?CURING', 'Execution', 'Curing period'),
    (r'WATER\s+CURING', 'Execution', 'Water curing required'),
    (r'COMPACTION[:\s]+(VIBRATOR|NEEDLE|SURFACE)', 'Execution', 'Compaction method'),
    (r'VIBRAT(?:OR|ION)', 'Execution', 'Vibration compaction required'),
    (r'CONSTRUCTION\s+JOINT', 'Execution', 'Construction joints specified'),
    (r'EXPANSION\s+JOINT', 'Execution', 'Expansion joints required'),
    (r'COLD\s+JOINT', 'Execution', 'Cold joint treatment'),
    (r'LAP\s+(?:LENGTH|SPLICE)[:\s]*(\d+)\s*(?:MM|D|DIA)', 'Execution', 'Lap length'),
    (r'ANCHORAGE\s+LENGTH', 'Execution', 'Anchorage length specified'),
]

# Waterproofing and special treatments
SPECIAL_PATTERNS = [
    (r'WATERPROOF(?:ING)?', 'Special', 'Waterproofing required'),
    (r'ANTI[\s-]?TERMITE', 'Special', 'Anti-termite treatment'),
    (r'DPC|DAMP\s+PROOF', 'Special', 'DPC required'),
    (r'EPOXY\s+(?:COAT|PAINT)', 'Special', 'Epoxy coating'),
    (r'CORROSION\s+(?:INHIBITOR|PROTECTION)', 'Special', 'Corrosion protection'),
    (r'ADMIXTURE', 'Special', 'Admixture specified'),
    (r'PLASTICIZER|SUPERPLASTICIZER', 'Special', 'Plasticizer required'),
    (r'RETARDER', 'Special', 'Retarder specified'),
    (r'ACCELERATOR', 'Special', 'Accelerator specified'),
    (r'BONDING\s+AGENT', 'Special', 'Bonding agent required'),
]

# Mix design
MIX_PATTERNS = [
    (r'MIX\s+DESIGN', 'Material', 'Mix design required'),
    (r'DESIGN\s+MIX', 'Material', 'Design mix concrete'),
    (r'NOMINAL\s+MIX', 'Material', 'Nominal mix concrete'),
    (r'W/C\s*(?:RATIO)?[:\s]*([0-9.]+)', 'Material', 'Water-cement ratio'),
    (r'WATER[\s-]CEMENT\s+RATIO[:\s]*([0-9.]+)', 'Material', 'Water-cement ratio'),
    (r'CEMENT\s+CONTENT[:\s]*(\d+)\s*KG', 'Material', 'Cement content'),
]

# Soil and foundation
SOIL_PATTERNS = [
    (r'SBC[:\s]*(\d+\.?\d*)\s*(?:T/SQ\.?M|KN/M2)', 'Geotechnical', 'Safe bearing capacity'),
    (r'BEARING\s+CAPACITY[:\s]*(\d+\.?\d*)', 'Geotechnical', 'Bearing capacity'),
    (r'WATER\s+TABLE[:\s]*(\d+\.?\d*)\s*M', 'Geotechnical', 'Water table depth'),
    (r'FOUNDATION\s+DEPTH[:\s]*(\d+\.?\d*)\s*(?:M|MM)', 'Geotechnical', 'Foundation depth'),
    (r'SUBGRADE\s+(?:MODULUS|REACTION)', 'Geotechnical', 'Subgrade modulus'),
]


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_requirements(
    ocr_notes_text: str,
    legend_blocks: List[Dict[str, Any]] = None
) -> List[Requirement]:
    """
    Extract requirements from OCR text and legend blocks.

    Args:
        ocr_notes_text: Raw OCR text from notes/legends
        legend_blocks: Optional list of legend text blocks

    Returns:
        List of Requirement objects
    """
    requirements = []
    seen = set()  # Avoid duplicates

    # Combine all text sources
    all_text = ocr_notes_text.upper()
    if legend_blocks:
        for block in legend_blocks:
            text = block.get('text', '') if isinstance(block, dict) else str(block)
            all_text += ' ' + text.upper()

    # Process each pattern category
    all_patterns = [
        CONCRETE_PATTERNS,
        STEEL_PATTERNS,
        COVER_PATTERNS,
        EXPOSURE_PATTERNS,
        CODE_PATTERNS,
        QC_PATTERNS,
        EXECUTION_PATTERNS,
        SPECIAL_PATTERNS,
        MIX_PATTERNS,
        SOIL_PATTERNS,
    ]

    for pattern_group in all_patterns:
        for pattern, category, req_type in pattern_group:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)

            for match in matches:
                # Get the matched value
                if match.groups():
                    value = match.group(1).strip()
                else:
                    value = match.group(0).strip()

                # Normalize value
                value = _normalize_value(value, category, req_type)

                # Create unique key
                key = f"{category}:{req_type}:{value}"
                if key in seen:
                    continue
                seen.add(key)

                # Get source text context
                start = max(0, match.start() - 20)
                end = min(len(all_text), match.end() + 20)
                source_text = all_text[start:end].strip()

                # Determine confidence
                confidence = _calculate_confidence(match, all_text, category)

                requirements.append(Requirement(
                    category=category,
                    requirement=req_type,
                    value=value,
                    source_text=source_text,
                    confidence=confidence
                ))

    # Sort by category and confidence
    requirements.sort(key=lambda r: (r.category, -r.confidence))

    logger.info(f"Extracted {len(requirements)} requirements from notes")
    return requirements


def _normalize_value(value: str, category: str, req_type: str) -> str:
    """Normalize extracted value."""
    value = value.strip()

    # Normalize concrete grades
    if 'Concrete' in req_type:
        if value.isdigit():
            return f"M{value}"
        return value.replace(' ', '')

    # Normalize steel grades
    if 'Reinforcement' in req_type or 'Steel' in req_type:
        value = value.replace(' ', '')
        if not value.startswith('Fe') and not value.startswith('TMT'):
            return f"Fe{value}"
        return value

    # Normalize cover
    if 'Cover' in req_type:
        digits = re.search(r'\d+', value)
        if digits:
            return f"{digits.group()}mm"
        return value

    # Normalize codes
    if category == 'Code':
        value = value.replace(' ', '')
        value = re.sub(r'[:\-]', ' ', value)
        return value.strip()

    return value


def _calculate_confidence(match: re.Match, text: str, category: str) -> float:
    """Calculate confidence score for a requirement match."""
    confidence = 0.7  # Base confidence

    # Higher confidence for codes (exact patterns)
    if category == 'Code':
        confidence = 0.9

    # Higher confidence for explicit labels
    context_start = max(0, match.start() - 50)
    context = text[context_start:match.start()].upper()

    explicit_labels = ['NOTE', 'SPEC', 'REQUIREMENT', 'SHALL BE', 'MUST BE', 'USE']
    if any(label in context for label in explicit_labels):
        confidence += 0.1

    # Lower confidence for values that might be dimensions
    if category == 'Material' and match.group(0).isdigit():
        confidence -= 0.1

    return min(0.95, max(0.3, confidence))


def extract_requirements_from_foundation_data(foundation_data: Any) -> List[Requirement]:
    """
    Extract requirements from foundation data notes.

    Args:
        foundation_data: FoundationPlanData object

    Returns:
        List of Requirement objects
    """
    requirements = []

    # Get notes from foundation data
    notes = getattr(foundation_data, 'notes', [])
    notes_text = ' '.join(notes) if notes else ''

    # Extract from notes
    if notes_text:
        requirements = extract_requirements(notes_text)

    # Add known materials as requirements
    concrete_grade = getattr(foundation_data, 'concrete_grade', None)
    if concrete_grade:
        requirements.append(Requirement(
            category='Material',
            requirement='Concrete grade',
            value=concrete_grade,
            source_text='From drawing title block',
            confidence=0.9
        ))

    steel_grade = getattr(foundation_data, 'steel_grade', None)
    if steel_grade:
        requirements.append(Requirement(
            category='Material',
            requirement='Reinforcement grade',
            value=steel_grade,
            source_text='From drawing title block',
            confidence=0.9
        ))

    sbc = getattr(foundation_data, 'soil_bearing', None)
    if sbc:
        requirements.append(Requirement(
            category='Geotechnical',
            requirement='Safe bearing capacity',
            value=f"{sbc} t/sqm",
            source_text='From drawing notes',
            confidence=0.85
        ))

    return requirements


def attach_requirements(
    output: 'EstimatorOutput',
    ocr_notes_text: str = "",
    legend_blocks: List[Dict[str, Any]] = None
) -> 'EstimatorOutput':
    """
    Extract and attach requirements to estimator output.

    Args:
        output: EstimatorOutput to modify
        ocr_notes_text: OCR text from notes
        legend_blocks: Legend text blocks

    Returns:
        Modified EstimatorOutput with requirements populated
    """
    # Extract from OCR text
    requirements = extract_requirements(ocr_notes_text, legend_blocks)

    # Add material requirements from estimator output
    if output.materials.concrete_grade:
        # Check if already extracted
        has_concrete = any(
            r.requirement == 'Concrete grade'
            for r in requirements
        )
        if not has_concrete:
            requirements.append(Requirement(
                category='Material',
                requirement='Concrete grade',
                value=output.materials.concrete_grade,
                source_text='From drawing',
                confidence=0.9
            ))

    if output.materials.steel_grade:
        has_steel = any(
            r.requirement == 'Reinforcement grade'
            for r in requirements
        )
        if not has_steel:
            requirements.append(Requirement(
                category='Material',
                requirement='Reinforcement grade',
                value=output.materials.steel_grade,
                source_text='From drawing',
                confidence=0.9
            ))

    if output.materials.soil_bearing_capacity:
        has_sbc = any(
            'bearing' in r.requirement.lower()
            for r in requirements
        )
        if not has_sbc:
            requirements.append(Requirement(
                category='Geotechnical',
                requirement='Safe bearing capacity',
                value=f"{output.materials.soil_bearing_capacity} t/sqm",
                source_text='From drawing',
                confidence=0.85
            ))

    output.requirements = requirements
    return output
