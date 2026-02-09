"""
Size Inference Module
Infers opening sizes from multiple sources:
1. Schedule tables (DOOR/WINDOW SCHEDULE)
2. Size callouts near openings (e.g., "900x2100")
3. Pixel-based inference using scale
4. Default values based on type (Indian standards)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpeningSize:
    """Inferred size for an opening."""
    width_mm: int
    height_mm: int
    source: str  # "schedule", "callout", "measured", "default"
    confidence: float
    notes: Optional[str] = None


@dataclass
class ScheduleEntry:
    """Parsed entry from a door/window schedule table."""
    tag: str
    width_mm: Optional[int] = None
    height_mm: Optional[int] = None
    quantity: int = 1
    material: Optional[str] = None
    remarks: Optional[str] = None


class SizeInferrer:
    """
    Infer opening sizes from multiple sources.

    Priority order:
    1. Schedule tables (if present and parsed)
    2. Size callouts near openings
    3. Measured from pixels using scale
    4. Defaults based on opening type (Indian standards)
    """

    # Indian standard defaults (mm)
    DOOR_DEFAULTS = {
        "main_door": {"width": 1000, "height": 2100},
        "internal_door": {"width": 900, "height": 2100},
        "toilet_door": {"width": 750, "height": 2100},
        "sliding_door": {"width": 1800, "height": 2100},
        "french_door": {"width": 1500, "height": 2100},
        "balcony_door": {"width": 900, "height": 2100},
        "wooden_door": {"width": 900, "height": 2100},
        "default": {"width": 900, "height": 2100},
    }

    WINDOW_DEFAULTS = {
        "standard_window": {"width": 1200, "height": 1200, "sill": 900},
        "large_window": {"width": 1500, "height": 1500, "sill": 600},
        "toilet_window": {"width": 600, "height": 450, "sill": 1800},
        "kitchen_window": {"width": 1200, "height": 1000, "sill": 900},
        "ventilator": {"width": 600, "height": 450, "sill": 2100},
        "default": {"width": 1200, "height": 1200, "sill": 900},
    }

    # Size pattern: 900x2100, 1200 X 1500
    SIZE_PATTERN = re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})")

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize size inferrer.

        Args:
            config_path: Path to assumptions.yaml for defaults
        """
        self.schedule_data: Dict[str, ScheduleEntry] = {}
        self.custom_defaults: Dict = {}

        if config_path and config_path.exists():
            self._load_custom_defaults(config_path)

    def _load_custom_defaults(self, config_path: Path) -> None:
        """Load custom defaults from config."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                if "doors" in data:
                    for key, val in data["doors"].items():
                        if isinstance(val, dict) and "width_mm" in val:
                            self.DOOR_DEFAULTS[key] = {
                                "width": val["width_mm"],
                                "height": val.get("height_mm", 2100),
                            }
                if "windows" in data:
                    for key, val in data["windows"].items():
                        if isinstance(val, dict) and "width_mm" in val:
                            self.WINDOW_DEFAULTS[key] = {
                                "width": val["width_mm"],
                                "height": val.get("height_mm", 1200),
                                "sill": val.get("sill_height_mm", 900),
                            }
        except Exception as e:
            logger.warning(f"Could not load custom defaults: {e}")

    def parse_schedule_table(
        self,
        table_data: List[List[str]],
        opening_type: str = "door",
    ) -> List[ScheduleEntry]:
        """
        Parse a schedule table extracted from the drawing.

        Args:
            table_data: 2D list of cell values
            opening_type: "door" or "window"

        Returns:
            List of ScheduleEntry objects
        """
        entries = []

        if not table_data or len(table_data) < 2:
            return entries

        # Find header row and column indices
        header_row = table_data[0]
        col_map = self._identify_columns(header_row, opening_type)

        if not col_map.get("tag"):
            logger.warning("Could not identify tag column in schedule")
            return entries

        # Parse data rows
        for row in table_data[1:]:
            if len(row) <= col_map["tag"]:
                continue

            tag = row[col_map["tag"]].strip().upper()
            if not tag:
                continue

            entry = ScheduleEntry(tag=tag)

            # Extract size
            if "size" in col_map and col_map["size"] < len(row):
                size_text = row[col_map["size"]]
                size = self._parse_size_text(size_text)
                if size:
                    entry.width_mm, entry.height_mm = size

            # Extract quantity
            if "qty" in col_map and col_map["qty"] < len(row):
                try:
                    entry.quantity = int(row[col_map["qty"]])
                except (ValueError, TypeError):
                    pass

            # Extract material
            if "material" in col_map and col_map["material"] < len(row):
                entry.material = row[col_map["material"]].strip()

            # Extract remarks
            if "remarks" in col_map and col_map["remarks"] < len(row):
                entry.remarks = row[col_map["remarks"]].strip()

            entries.append(entry)

        # Store for later lookup
        for entry in entries:
            self.schedule_data[entry.tag] = entry

        return entries

    def _identify_columns(
        self,
        header_row: List[str],
        opening_type: str,
    ) -> Dict[str, int]:
        """Identify column indices from header row."""
        col_map = {}

        tag_patterns = [
            r"MARK", r"NO\.?", r"TAG", r"REF",
            r"DOOR\s*NO", r"WINDOW\s*NO",
            r"D\.?\s*NO", r"W\.?\s*NO",
        ]
        size_patterns = [r"SIZE", r"DIMENSION", r"DIM"]
        qty_patterns = [r"QTY", r"QUANTITY", r"NOS?\.?", r"COUNT"]
        material_patterns = [r"MATERIAL", r"TYPE", r"SPEC"]
        remarks_patterns = [r"REMARK", r"NOTE", r"COMMENT"]

        for i, cell in enumerate(header_row):
            cell_upper = cell.upper().strip()

            if not col_map.get("tag"):
                for pattern in tag_patterns:
                    if re.search(pattern, cell_upper):
                        col_map["tag"] = i
                        break

            if not col_map.get("size"):
                for pattern in size_patterns:
                    if re.search(pattern, cell_upper):
                        col_map["size"] = i
                        break

            if not col_map.get("qty"):
                for pattern in qty_patterns:
                    if re.search(pattern, cell_upper):
                        col_map["qty"] = i
                        break

            if not col_map.get("material"):
                for pattern in material_patterns:
                    if re.search(pattern, cell_upper):
                        col_map["material"] = i
                        break

            if not col_map.get("remarks"):
                for pattern in remarks_patterns:
                    if re.search(pattern, cell_upper):
                        col_map["remarks"] = i
                        break

        return col_map

    def _parse_size_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse size from text like '900x2100' or '900 X 2100'."""
        match = self.SIZE_PATTERN.search(text)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    def infer_size(
        self,
        tag: Optional[str],
        opening_type: str,
        category: Optional[str] = None,
        width_px: Optional[float] = None,
        scale_px_per_mm: Optional[float] = None,
        nearby_text: Optional[str] = None,
    ) -> OpeningSize:
        """
        Infer size for an opening using priority sources.

        Args:
            tag: Opening tag (D1, W2, etc.)
            opening_type: "door", "window", or "ventilator"
            category: Specific category (main_door, toilet_window, etc.)
            width_px: Measured width in pixels
            scale_px_per_mm: Scale factor
            nearby_text: Text found near the opening

        Returns:
            OpeningSize with inferred dimensions
        """
        # Priority 1: Schedule data
        if tag and tag in self.schedule_data:
            entry = self.schedule_data[tag]
            if entry.width_mm and entry.height_mm:
                return OpeningSize(
                    width_mm=entry.width_mm,
                    height_mm=entry.height_mm,
                    source="schedule",
                    confidence=0.95,
                    notes=f"From schedule: {tag}",
                )

        # Priority 2: Size callout in nearby text
        if nearby_text:
            size = self._parse_size_text(nearby_text)
            if size:
                return OpeningSize(
                    width_mm=size[0],
                    height_mm=size[1],
                    source="callout",
                    confidence=0.9,
                    notes=f"From callout: {nearby_text}",
                )

        # Priority 3: Measured from pixels
        if width_px and scale_px_per_mm and scale_px_per_mm > 0:
            width_mm = int(width_px / scale_px_per_mm)

            # Round to nearest 50mm (typical increments)
            width_mm = round(width_mm / 50) * 50

            # Get default height based on type
            if opening_type == "door":
                height_mm = 2100
            elif opening_type == "ventilator":
                height_mm = 450
            else:
                # Estimate window height based on width ratio
                height_mm = width_mm  # Typically square or similar

            return OpeningSize(
                width_mm=width_mm,
                height_mm=height_mm,
                source="measured",
                confidence=0.7,
                notes=f"Measured: {width_px:.0f}px at scale {scale_px_per_mm:.3f}px/mm",
            )

        # Priority 4: Defaults based on type
        return self._get_default_size(opening_type, category, tag)

    def _get_default_size(
        self,
        opening_type: str,
        category: Optional[str],
        tag: Optional[str],
    ) -> OpeningSize:
        """Get default size based on opening type and category."""
        if opening_type == "door":
            defaults = self.DOOR_DEFAULTS

            # Try category first
            if category and category in defaults:
                d = defaults[category]
            # Try to infer from tag
            elif tag:
                tag_upper = tag.upper()
                if "MD" in tag_upper or "MAIN" in tag_upper:
                    d = defaults.get("main_door", defaults["default"])
                elif "SD" in tag_upper:
                    d = defaults.get("sliding_door", defaults["default"])
                elif "TD" in tag_upper:
                    d = defaults.get("toilet_door", defaults["default"])
                elif "FD" in tag_upper:
                    d = defaults.get("french_door", defaults["default"])
                else:
                    d = defaults["default"]
            else:
                d = defaults["default"]

            return OpeningSize(
                width_mm=d["width"],
                height_mm=d["height"],
                source="default",
                confidence=0.5,
                notes=f"Default for {category or 'door'}",
            )

        elif opening_type == "ventilator":
            d = self.WINDOW_DEFAULTS.get("ventilator", self.WINDOW_DEFAULTS["default"])
            return OpeningSize(
                width_mm=d["width"],
                height_mm=d["height"],
                source="default",
                confidence=0.5,
                notes="Default for ventilator",
            )

        else:  # window
            defaults = self.WINDOW_DEFAULTS

            if category and category in defaults:
                d = defaults[category]
            elif tag:
                tag_upper = tag.upper()
                if "LW" in tag_upper:
                    d = defaults.get("large_window", defaults["default"])
                elif "TW" in tag_upper:
                    d = defaults.get("toilet_window", defaults["default"])
                elif "KW" in tag_upper:
                    d = defaults.get("kitchen_window", defaults["default"])
                else:
                    d = defaults["default"]
            else:
                d = defaults["default"]

            return OpeningSize(
                width_mm=d["width"],
                height_mm=d["height"],
                source="default",
                confidence=0.5,
                notes=f"Default for {category or 'window'}",
            )

    def get_sill_height(
        self,
        opening_type: str,
        category: Optional[str] = None,
    ) -> int:
        """Get default sill height for windows."""
        if opening_type != "window" and opening_type != "ventilator":
            return 0

        defaults = self.WINDOW_DEFAULTS

        if category and category in defaults:
            return defaults[category].get("sill", 900)

        if opening_type == "ventilator":
            return defaults.get("ventilator", {}).get("sill", 2100)

        return 900  # Default sill height


def create_size_summary(
    openings: List[Any],
    size_inferrer: SizeInferrer,
) -> Dict[str, Dict]:
    """
    Create summary of sizes by opening type.

    Args:
        openings: List of detected openings
        size_inferrer: SizeInferrer instance

    Returns:
        Summary dict with counts and average sizes
    """
    summary = {
        "doors": {},
        "windows": {},
        "ventilators": {},
    }

    for opening in openings:
        tag = getattr(opening, "tag", None) or opening.id
        opening_type = getattr(opening, "type", "unknown")

        # Determine bucket
        if "door" in opening_type.lower():
            bucket = summary["doors"]
        elif "vent" in opening_type.lower():
            bucket = summary["ventilators"]
        else:
            bucket = summary["windows"]

        # Normalize tag
        base_tag = tag.split("-")[0] if "-" in tag else tag

        if base_tag not in bucket:
            bucket[base_tag] = {
                "count": 0,
                "widths": [],
                "heights": [],
            }

        bucket[base_tag]["count"] += 1

        width = getattr(opening, "width_m", None)
        height = getattr(opening, "height_m", None)

        if width:
            bucket[base_tag]["widths"].append(width * 1000)  # Convert to mm
        if height:
            bucket[base_tag]["heights"].append(height * 1000)

    # Calculate averages
    for category in summary.values():
        for tag_data in category.values():
            if tag_data["widths"]:
                tag_data["avg_width_mm"] = sum(tag_data["widths"]) / len(tag_data["widths"])
            if tag_data["heights"]:
                tag_data["avg_height_mm"] = sum(tag_data["heights"]) / len(tag_data["heights"])

    return summary
