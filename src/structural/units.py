"""
Unit Conversion Utilities
Handles Indian construction drawing conventions:
- Feet-inches (e.g., 6'-6", 4'-0")
- Meters
- Millimeters
"""

import re
from typing import Tuple, Optional, Union


def feet_inch_to_mm(feet_inch_str: str) -> Optional[float]:
    """
    Convert feet-inches string to millimeters.

    Formats supported:
    - "6'-6"" -> 1981.2 mm
    - "6'-6" -> 1981.2 mm
    - "4'-0"" -> 1219.2 mm
    - "6'-6\"" -> 1981.2 mm
    - "6-6" -> 1981.2 mm
    - "6'6\"" -> 1981.2 mm (no dash)

    Args:
        feet_inch_str: String in feet-inches format

    Returns:
        Value in millimeters, or None if parsing fails
    """
    if not feet_inch_str:
        return None

    # Clean up string
    s = feet_inch_str.strip()

    # Pattern 1: 6'-6" or 6'-6 or 6'-6"
    pattern1 = r"(\d+)[\'\'][\-\s]?(\d+)[\"\"]?"
    match = re.match(pattern1, s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return (feet * 12 + inches) * 25.4

    # Pattern 2: 6-6 (feet-inches without symbols)
    pattern2 = r"^(\d+)\-(\d+)$"
    match = re.match(pattern2, s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return (feet * 12 + inches) * 25.4

    # Pattern 3: Just feet (e.g., "6'" or "6'")
    pattern3 = r"^(\d+)[\'\']$"
    match = re.match(pattern3, s)
    if match:
        feet = int(match.group(1))
        return feet * 12 * 25.4

    # Pattern 4: Just inches (e.g., "9"")
    pattern4 = r"^(\d+)[\"\"]$"
    match = re.match(pattern4, s)
    if match:
        inches = int(match.group(1))
        return inches * 25.4

    return None


def mm_to_feet_inch(mm: float) -> str:
    """
    Convert millimeters to feet-inches string.

    Args:
        mm: Value in millimeters

    Returns:
        String in format "X'-Y""
    """
    total_inches = mm / 25.4
    feet = int(total_inches // 12)
    inches = int(round(total_inches % 12))

    # Handle rounding to 12 inches
    if inches == 12:
        feet += 1
        inches = 0

    return f"{feet}'-{inches}\""


def parse_footing_size(size_str: str) -> Optional[Tuple[float, float]]:
    """
    Parse footing size from drawing annotation.

    Formats:
    - "6'-6" x 6'-6"" -> (1981.2, 1981.2) mm
    - "6'-6" 6'-6"" -> (1981.2, 1981.2) mm
    - "5'-0" x 5'-0"" -> (1524.0, 1524.0) mm
    - "F1 6'-6" 6'-6"" -> (1981.2, 1981.2) mm

    Returns:
        (length_mm, width_mm) or None
    """
    if not size_str:
        return None

    # Find all feet-inch patterns in the string
    pattern = r"(\d+)[\'\'][\-\s]?(\d+)[\"\"]?"
    matches = list(re.finditer(pattern, size_str))

    if len(matches) >= 2:
        # First two matches are length and width
        feet1, inch1 = int(matches[0].group(1)), int(matches[0].group(2))
        feet2, inch2 = int(matches[1].group(1)), int(matches[1].group(2))

        length_mm = (feet1 * 12 + inch1) * 25.4
        width_mm = (feet2 * 12 + inch2) * 25.4

        return (length_mm, width_mm)
    elif len(matches) == 1:
        # Square footing
        feet, inch = int(matches[0].group(1)), int(matches[0].group(2))
        size_mm = (feet * 12 + inch) * 25.4
        return (size_mm, size_mm)

    return None


def parse_dimension(dim_str: str) -> Optional[float]:
    """
    Parse a dimension string that could be in various formats.

    Supports:
    - Feet-inches: "6'-6"", "4'-0""
    - Millimeters: "1500mm", "1500 mm"
    - Meters: "1.5m", "1.5 m"
    - Plain numbers (assumed mm): "1500"

    Returns:
        Value in millimeters
    """
    if not dim_str:
        return None

    s = dim_str.strip().lower()

    # Try feet-inches first
    ft_in = feet_inch_to_mm(dim_str)
    if ft_in is not None:
        return ft_in

    # Millimeters
    mm_match = re.match(r"(\d+\.?\d*)\s*mm", s)
    if mm_match:
        return float(mm_match.group(1))

    # Meters
    m_match = re.match(r"(\d+\.?\d*)\s*m(?!m)", s)
    if m_match:
        return float(m_match.group(1)) * 1000

    # Plain number (assume mm)
    num_match = re.match(r"^(\d+\.?\d*)$", s)
    if num_match:
        return float(num_match.group(1))

    return None


def extract_column_label(text: str) -> Optional[str]:
    """
    Extract column label from text.

    Patterns:
    - "C1", "C12", "C45"
    - "COL-1", "COL.1"

    Returns:
        Normalized column label (e.g., "C1")
    """
    patterns = [
        r'\b(C\d+)\b',
        r'\bCOL[\.\-]?(\d+)\b',
        r'\bCOLUMN[\.\-\s]?(\d+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'COL' in pattern.upper():
                return f"C{match.group(1)}"
            return match.group(1).upper()

    return None


def extract_footing_label(text: str) -> Optional[str]:
    """
    Extract footing label from text.

    Patterns:
    - "F1", "F12", "F4A", "F8A"
    - "IF1" (isolated), "CF1" (combined)

    Returns:
        Footing label
    """
    patterns = [
        r'\b(F\d+[A-Z]?)\b',
        r'\b([IC]F\d+)\b',
        r'\bFOOTING[\.\-\s]?(\d+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            if 'FOOTING' in pattern.upper():
                return f"F{label}"
            return label

    return None


# Footing type patterns from the drawing
FOOTING_TYPES = {
    'F1': {'size': "7'-0\" x 7'-0\"", 'mm': (2134, 2134)},
    'F2': {'size': "6'-6\" x 6'-6\"", 'mm': (1981, 1981)},
    'F3': {'size': "7'-0\" x 5'-9\"", 'mm': (2134, 1753)},
    'F4': {'size': "6'-0\" x 6'-0\"", 'mm': (1829, 1829)},
    'F4A': {'size': "6'-0\" x 6'-0\"", 'mm': (1829, 1829)},
    'F5': {'size': "5'-7\" x 5'-7\"", 'mm': (1702, 1702)},
    'F5A': {'size': "5'-7\" x 5'-7\"", 'mm': (1702, 1702)},
    'F6': {'size': "5'-3\" x 5'-3\"", 'mm': (1600, 1600)},
    'F7': {'size': "7'-0\" x 7'-0\"", 'mm': (2134, 2134)},
    'F8': {'size': "5'-0\" x 5'-0\"", 'mm': (1524, 1524)},
    'F8A': {'size': "5'-0\" x 5'-0\"", 'mm': (1524, 1524)},
    'F9': {'size': "4'-6\" x 4'-6\"", 'mm': (1372, 1372)},
    'F10': {'size': "4'-0\" x 4'-0\"", 'mm': (1219, 1219)},
    'F11': {'size': "4'-9\" x 4'-0\"", 'mm': (1448, 1219)},
    'F12': {'size': "3'-6\" x 3'-6\"", 'mm': (1067, 1067)},
    'F13': {'size': "3'-4\" x 3'-4\"", 'mm': (1016, 1016)},
    'F14': {'size': "3'-4\" x 3'-4\"", 'mm': (1016, 1016)},
    'F15': {'size': "4'-10\" x 4'-6\"", 'mm': (1473, 1372)},
    'F16': {'size': "4'-6\" x 4'-6\"", 'mm': (1372, 1372)},
    'F17': {'size': "4'-6\" x 5'-9\"", 'mm': (1372, 1753)},
    'F18': {'size': "4'-0\" x 4'-9\"", 'mm': (1219, 1448)},
}


def get_footing_size(label: str) -> Optional[Tuple[float, float]]:
    """
    Get footing size from standard types.

    Args:
        label: Footing label (e.g., "F1", "F8A")

    Returns:
        (length_mm, width_mm) or None
    """
    label = label.upper()
    if label in FOOTING_TYPES:
        return FOOTING_TYPES[label]['mm']
    return None


if __name__ == "__main__":
    # Test conversions
    test_cases = [
        "6'-6\"",
        "6'-6",
        "4'-0\"",
        "5'-7\"",
        "7'-0\"",
        "3'-6\"",
    ]

    print("Feet-Inch to MM conversions:")
    for tc in test_cases:
        mm = feet_inch_to_mm(tc)
        back = mm_to_feet_inch(mm) if mm else "N/A"
        print(f"  {tc} -> {mm:.1f} mm -> {back}")

    print("\nFooting size parsing:")
    footing_tests = [
        "F1 6'-6\" 6'-6\"",
        "5'-0\" x 5'-0\"",
        "F8A 5'-0\" 5'-0\"",
    ]
    for ft in footing_tests:
        size = parse_footing_size(ft)
        print(f"  {ft} -> {size}")

    print("\nStandard footing sizes:")
    for label, info in list(FOOTING_TYPES.items())[:5]:
        print(f"  {label}: {info['size']} = {info['mm']} mm")
