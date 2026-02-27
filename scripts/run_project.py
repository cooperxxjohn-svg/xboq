#!/usr/bin/env python3
"""
XBOQ V1 Project Runner - Architect-first Finish BOQ Pipeline (India)

Complete BOQ generation pipeline for Indian residential construction.
Focuses on finishes, openings, and room-based quantities.

Usage:
    python scripts/run_project.py --project_dir data/projects/my_project
    python scripts/run_project.py --project_dir data/projects/my_project --profile premium
    python scripts/run_project.py --project_dir data/projects/my_project --output out/custom

V1 Scope:
    - Rooms + areas (from rooms.json or detection)
    - Openings schedule (doors/windows/ventilators)
    - Finishes takeoff using rules/finish_templates.yaml
    - BOQ outputs in canonical schema
    - CPWD mapping coverage report
    - Confidence warnings

Expected project structure:
    data/projects/<project_id>/
    ├── plans/              # Floor plan images (optional)
    │   ├── floor_1.png
    │   └── floor_2.png
    ├── rooms.json          # Room data (required for V1)
    ├── openings.json       # Openings data (required for V1)
    └── config.json         # Project overrides (optional)
"""

import argparse
import csv
import json
import logging
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import YAML
try:
    import yaml
except ImportError:
    yaml = None
    print("⚠️  PyYAML not available - using defaults")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

SQFT_PER_SQM = 10.764

# Valid opening types
OPENING_TYPES = {
    "door": ["door", "main door", "main_door", "flush door", "panel door"],
    "window": ["window", "sliding window", "casement window"],
    "ventilator": ["ventilator", "vent", "exhaust", "louver"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProjectConfig:
    """Project configuration with India-specific defaults."""
    project_id: str
    profile: str = "typical"
    ceiling_height_mm: int = 3000
    slab_thickness_mm: int = 125
    skirting_height_mm: int = 100
    toilet_tile_height_mm: int = 2100
    kitchen_dado_height_mm: int = 600
    # Profile multipliers
    wastage_multiplier: float = 1.0


@dataclass
class FinishTotals:
    """Finish quantities by type."""
    # Floor finishes by type (sqm)
    floor_by_type: Dict[str, float] = field(default_factory=dict)
    # Wall finishes
    wall_tile_area_sqm: float = 0.0  # Toilet tiles
    wall_dado_area_sqm: float = 0.0  # Kitchen dado
    wall_paint_area_sqm: float = 0.0
    # Ceiling
    ceiling_paint_area_sqm: float = 0.0
    # Skirting
    skirting_length_m: float = 0.0
    # Waterproofing
    waterproofing_area_sqm: float = 0.0
    # By room breakdown
    room_finishes: List[Dict] = field(default_factory=list)


@dataclass
class OpeningsSummary:
    """Openings schedule summary."""
    total_doors: int = 0
    total_windows: int = 0
    total_ventilators: int = 0
    # By tag
    doors_by_tag: Dict[str, int] = field(default_factory=dict)
    windows_by_tag: Dict[str, int] = field(default_factory=dict)
    ventilators_by_tag: Dict[str, int] = field(default_factory=dict)
    # Schedule items
    door_schedule: List[Dict] = field(default_factory=list)
    window_schedule: List[Dict] = field(default_factory=list)


@dataclass
class BOQItem:
    """Standard BOQ item."""
    item_code: str
    description: str
    qty: float
    unit: str
    derived_from: str
    confidence: float
    room_id: Optional[str] = None
    assumption_used: Optional[str] = None
    notes: Optional[str] = None
    category: str = "general"

    def to_dict(self) -> Dict:
        return {
            "item_code": self.item_code,
            "description": self.description,
            "qty": round(self.qty, 2),
            "unit": self.unit,
            "derived_from": self.derived_from,
            "confidence": round(self.confidence, 2),
            "room_id": self.room_id or "",
            "assumption_used": self.assumption_used or "",
            "notes": self.notes or "",
            "category": self.category,
        }

    def to_csv_row(self) -> List[str]:
        return [
            self.item_code,
            self.description,
            f"{self.qty:.2f}",
            self.unit,
            self.derived_from,
            f"{self.confidence:.2f}",
            self.room_id or "",
            self.assumption_used or "",
            self.notes or "",
            self.category,
        ]

    @staticmethod
    def csv_headers() -> List[str]:
        return [
            "item_code", "description", "qty", "unit", "derived_from",
            "confidence", "room_id", "assumption_used", "notes", "category"
        ]


@dataclass
class QASummary:
    """Comprehensive QA summary for V1."""
    project_id: str
    run_timestamp: str
    profile: str
    status: str = "success"

    # Plan info
    plans_processed: int = 0
    pages_processed: int = 0

    # Room counts
    rooms_count: int = 0
    rooms_by_type: Dict[str, int] = field(default_factory=dict)
    total_area_sqm: float = 0.0

    # Openings
    openings: OpeningsSummary = field(default_factory=OpeningsSummary)

    # Finishes
    finishes: FinishTotals = field(default_factory=FinishTotals)

    # BOQ
    boq_lines_count: int = 0
    boq_by_category: Dict[str, int] = field(default_factory=dict)

    # CPWD coverage
    cpwd_mapped: int = 0
    cpwd_unmapped: int = 0
    cpwd_coverage_percent: float = 0.0
    unmapped_codes: List[str] = field(default_factory=list)

    # Confidence
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0

    # Warnings and actions
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)

    # Files
    output_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested dataclasses
        d["openings"] = asdict(self.openings)
        d["finishes"] = asdict(self.finishes)
        return d

    def sqft(self, sqm: float) -> float:
        return sqm * SQFT_PER_SQM


# =============================================================================
# FINISH CALCULATOR
# =============================================================================

class FinishCalculator:
    """Calculate finish quantities from rooms using templates."""

    def __init__(self, templates_path: Path, config: ProjectConfig):
        self.config = config
        self.templates = {}
        self.defaults = {}
        self.default_template = {}
        self._load_templates(templates_path)

    def _load_templates(self, path: Path):
        """Load finish templates from YAML."""
        if not path.exists() or yaml is None:
            logger.warning(f"Finish templates not found: {path}")
            return

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            self.defaults = data.get("defaults", {})
            self.templates = data.get("templates", {})
            self.default_template = data.get("default", {})

            # Build alias lookup
            self.alias_map = {}
            for template_name, template in self.templates.items():
                aliases = template.get("aliases", [template_name])
                for alias in aliases:
                    self.alias_map[alias.lower()] = template_name

            logger.info(f"  Loaded {len(self.templates)} finish templates")
        except Exception as e:
            logger.warning(f"Could not load finish templates: {e}")

    def get_template(self, room_label: str, room_type: str = None) -> Tuple[str, Dict]:
        """Get finish template for a room."""
        # Try room_type first, then label
        for key in [room_type, room_label]:
            if key:
                key_lower = key.lower().strip()
                if key_lower in self.alias_map:
                    template_name = self.alias_map[key_lower]
                    return template_name, self.templates[template_name]

        # Return default
        return "default", self.default_template

    def calculate_room_finishes(
        self,
        room: Dict,
        openings: List[Dict] = None
    ) -> Tuple[List[BOQItem], Dict]:
        """Calculate finish BOQ for a single room."""
        items = []
        summary = {
            "room_id": room.get("id", ""),
            "room_label": room.get("label", "Unknown"),
            "room_type": room.get("room_type", ""),
            "area_sqm": room.get("area_sqm", 0),
            "perimeter_m": room.get("perimeter_m", 0),
            "template_used": "",
            "floor_type": "",
            "floor_area_sqm": 0,
            "wall_tile_sqm": 0,
            "wall_dado_sqm": 0,
            "wall_paint_sqm": 0,
            "ceiling_sqm": 0,
            "skirting_m": 0,
            "waterproofing_sqm": 0,
        }

        area_sqm = room.get("area_sqm", 0)
        perimeter_m = room.get("perimeter_m", 0)
        room_id = room.get("id", "")
        room_label = room.get("label", "Unknown")
        room_type = room.get("room_type", "")

        # Get template
        template_name, template = self.get_template(room_label, room_type)
        summary["template_used"] = template_name

        # Check if excluded
        if template.get("exclude_from_finishes"):
            return items, summary

        ceiling_height = self.config.ceiling_height_mm / 1000  # to meters
        wastage = self.config.wastage_multiplier

        # Calculate opening deductions
        door_area = 0
        window_area = 0
        door_width_total = 0
        if openings:
            for op in openings:
                if op.get("room_left_id") == room_id or op.get("room_right_id") == room_id:
                    w = op.get("width_m", 0)
                    h = op.get("height_m", 2.1)
                    if op.get("type") == "door":
                        door_area += w * h
                        door_width_total += w
                    elif op.get("type") == "window":
                        window_area += w * h

        # ----- FLOOR -----
        floor_spec = template.get("floor", {})
        if floor_spec and floor_spec.get("type") != "none":
            floor_wastage = floor_spec.get("wastage", 0.05)
            floor_area = area_sqm * (1 + floor_wastage) * wastage
            floor_type = floor_spec.get("type", "cement_flooring")

            summary["floor_type"] = floor_type
            summary["floor_area_sqm"] = floor_area

            items.append(BOQItem(
                item_code=floor_spec.get("item_code", "FLR-GEN-01"),
                description=floor_spec.get("description", f"{floor_type} flooring"),
                qty=floor_area,
                unit="sqm",
                derived_from=f"room_{room_id}",
                confidence=0.85 if area_sqm > 0 else 0.5,
                room_id=room_id,
                assumption_used=f"wastage {floor_wastage*100:.0f}%",
                category="finishes_floor",
            ))

        # ----- SKIRTING -----
        skirting_spec = template.get("skirting", {})
        if skirting_spec and skirting_spec.get("type") != "none":
            skirting_length = perimeter_m - door_width_total
            if skirting_length > 0:
                summary["skirting_m"] = skirting_length

                items.append(BOQItem(
                    item_code=skirting_spec.get("item_code", "SKT-GEN-01"),
                    description=skirting_spec.get("description", "Tile skirting"),
                    qty=skirting_length,
                    unit="m",
                    derived_from=f"room_{room_id}",
                    confidence=0.8 if perimeter_m > 0 else 0.5,
                    room_id=room_id,
                    assumption_used=f"deducted {door_width_total:.1f}m for doors",
                    category="finishes_skirting",
                ))

        # ----- WALLS -----
        wall_spec = template.get("walls", {})
        gross_wall_area = perimeter_m * ceiling_height

        if wall_spec:
            wall_type = wall_spec.get("type", "plastic_emulsion")

            # Handle toilet wall tiles (full height tiles)
            if wall_type == "ceramic_tiles":
                tile_height = wall_spec.get("height_mm", 2100) / 1000
                tile_area = perimeter_m * tile_height
                tile_wastage = wall_spec.get("wastage", 0.07)
                tile_area_with_wastage = tile_area * (1 + tile_wastage) * wastage

                summary["wall_tile_sqm"] = tile_area_with_wastage

                items.append(BOQItem(
                    item_code=wall_spec.get("item_code", "WTL-CER-01"),
                    description=wall_spec.get("description", f"Wall tiles up to {tile_height*1000:.0f}mm"),
                    qty=tile_area_with_wastage,
                    unit="sqm",
                    derived_from=f"room_{room_id}",
                    confidence=0.85,
                    room_id=room_id,
                    assumption_used=f"tile height {tile_height*1000:.0f}mm",
                    category="finishes_wall_tile",
                ))

                # Paint above tiles
                above_tiles = wall_spec.get("above_tiles", {})
                if above_tiles:
                    paint_height = ceiling_height - tile_height
                    if paint_height > 0:
                        paint_area = perimeter_m * paint_height - window_area
                        paint_area = max(0, paint_area)
                        summary["wall_paint_sqm"] = paint_area

                        items.append(BOQItem(
                            item_code=above_tiles.get("item_code", "PNT-INT-01"),
                            description=above_tiles.get("description", "Paint above tiles"),
                            qty=paint_area,
                            unit="sqm",
                            derived_from=f"room_{room_id}",
                            confidence=0.8,
                            room_id=room_id,
                            category="finishes_wall_paint",
                        ))

            # Handle kitchen (dado + paint above)
            elif wall_type == "mixed":
                dado_spec = wall_spec.get("dado", {})
                above_dado = wall_spec.get("above_dado", {})

                if dado_spec:
                    dado_height = dado_spec.get("height_mm", 600) / 1000
                    dado_area = perimeter_m * dado_height
                    dado_wastage = dado_spec.get("wastage", 0.07)
                    dado_area_with_wastage = dado_area * (1 + dado_wastage) * wastage

                    summary["wall_dado_sqm"] = dado_area_with_wastage

                    items.append(BOQItem(
                        item_code=dado_spec.get("item_code", "DAD-CER-01"),
                        description=dado_spec.get("description", f"Ceramic dado {dado_height*1000:.0f}mm"),
                        qty=dado_area_with_wastage,
                        unit="sqm",
                        derived_from=f"room_{room_id}",
                        confidence=0.85,
                        room_id=room_id,
                        assumption_used=f"dado height {dado_height*1000:.0f}mm",
                        category="finishes_wall_dado",
                    ))

                if above_dado:
                    paint_height = ceiling_height - dado_height
                    paint_area = perimeter_m * paint_height - door_area - window_area
                    paint_area = max(0, paint_area)
                    summary["wall_paint_sqm"] = paint_area

                    items.append(BOQItem(
                        item_code=above_dado.get("item_code", "PNT-INT-01"),
                        description=above_dado.get("description", "Paint above dado"),
                        qty=paint_area,
                        unit="sqm",
                        derived_from=f"room_{room_id}",
                        confidence=0.8,
                        room_id=room_id,
                        category="finishes_wall_paint",
                    ))

            # Standard paint
            else:
                paint_area = gross_wall_area - door_area - window_area
                paint_area = max(0, paint_area)
                summary["wall_paint_sqm"] = paint_area

                items.append(BOQItem(
                    item_code=wall_spec.get("item_code", "PNT-INT-01"),
                    description=wall_spec.get("description", "2 coats plastic emulsion"),
                    qty=paint_area,
                    unit="sqm",
                    derived_from=f"room_{room_id}",
                    confidence=0.85,
                    room_id=room_id,
                    assumption_used="deducted openings",
                    category="finishes_wall_paint",
                ))

        # ----- CEILING -----
        ceiling_spec = template.get("ceiling", {})
        if ceiling_spec:
            ceiling_area = area_sqm
            summary["ceiling_sqm"] = ceiling_area

            items.append(BOQItem(
                item_code=ceiling_spec.get("item_code", "PNT-CLG-01"),
                description=ceiling_spec.get("description", "Ceiling paint"),
                qty=ceiling_area,
                unit="sqm",
                derived_from=f"room_{room_id}",
                confidence=0.85,
                room_id=room_id,
                category="finishes_ceiling",
            ))

        # ----- WATERPROOFING -----
        waterproofing_spec = template.get("waterproofing", {})
        if waterproofing_spec:
            wp_area = area_sqm
            summary["waterproofing_sqm"] = wp_area

            items.append(BOQItem(
                item_code=waterproofing_spec.get("item_code", "WPR-GEN-01"),
                description=waterproofing_spec.get("description", "Waterproofing treatment"),
                qty=wp_area,
                unit="sqm",
                derived_from=f"room_{room_id}",
                confidence=0.9,
                room_id=room_id,
                category="finishes_waterproofing",
            ))

        return items, summary


# =============================================================================
# OPENINGS PROCESSOR
# =============================================================================

class OpeningsProcessor:
    """Process openings data into schedule and BOQ."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def process(self, openings: List[Dict]) -> Tuple[List[BOQItem], OpeningsSummary]:
        """Process openings into BOQ items and summary."""
        items = []
        summary = OpeningsSummary()

        # Group by type and tag
        doors = []
        windows = []
        ventilators = []

        for op in openings:
            op_type = op.get("type", "").lower()
            if op_type == "door":
                doors.append(op)
            elif op_type == "window":
                windows.append(op)
            elif op_type in ["ventilator", "vent"]:
                ventilators.append(op)

        # Process doors
        summary.total_doors = len(doors)
        door_groups = self._group_by_tag(doors)
        for tag, group in door_groups.items():
            summary.doors_by_tag[tag] = len(group)
            rep = group[0]
            summary.door_schedule.append({
                "tag": tag,
                "width_mm": int(rep.get("width_m", 0.9) * 1000),
                "height_mm": int(rep.get("height_m", 2.1) * 1000),
                "quantity": len(group),
                "material": rep.get("material", "flush_door"),
            })

            # BOQ items for doors
            items.append(BOQItem(
                item_code=f"DOR-{tag.upper().replace(' ', '-')}",
                description=f"Door {tag} ({rep.get('width_m', 0.9)*1000:.0f}x{rep.get('height_m', 2.1)*1000:.0f}mm) - {rep.get('material', 'flush door')}",
                qty=len(group),
                unit="nos",
                derived_from="openings_schedule",
                confidence=self._avg_confidence(group),
                category="openings_door",
            ))

        # Process windows
        summary.total_windows = len(windows)
        window_groups = self._group_by_tag(windows)
        for tag, group in window_groups.items():
            summary.windows_by_tag[tag] = len(group)
            rep = group[0]
            summary.window_schedule.append({
                "tag": tag,
                "width_mm": int(rep.get("width_m", 1.2) * 1000),
                "height_mm": int(rep.get("height_m", 1.2) * 1000),
                "quantity": len(group),
                "material": rep.get("material", "aluminium_sliding"),
            })

            items.append(BOQItem(
                item_code=f"WIN-{tag.upper().replace(' ', '-')}",
                description=f"Window {tag} ({rep.get('width_m', 1.2)*1000:.0f}x{rep.get('height_m', 1.2)*1000:.0f}mm) - {rep.get('material', 'aluminium')}",
                qty=len(group),
                unit="nos",
                derived_from="openings_schedule",
                confidence=self._avg_confidence(group),
                category="openings_window",
            ))

        # Process ventilators
        summary.total_ventilators = len(ventilators)
        vent_groups = self._group_by_tag(ventilators)
        for tag, group in vent_groups.items():
            summary.ventilators_by_tag[tag] = len(group)
            rep = group[0]

            items.append(BOQItem(
                item_code=f"VNT-{tag.upper().replace(' ', '-')}",
                description=f"Ventilator {tag} ({rep.get('width_m', 0.45)*1000:.0f}x{rep.get('height_m', 0.6)*1000:.0f}mm)",
                qty=len(group),
                unit="nos",
                derived_from="openings_schedule",
                confidence=self._avg_confidence(group),
                category="openings_ventilator",
            ))

        return items, summary

    def _group_by_tag(self, openings: List[Dict]) -> Dict[str, List[Dict]]:
        """Group openings by tag."""
        groups = defaultdict(list)
        for op in openings:
            tag = op.get("tag", "untagged")
            groups[tag].append(op)
        return dict(groups)

    def _avg_confidence(self, openings: List[Dict]) -> float:
        """Calculate average confidence."""
        if not openings:
            return 0.5
        return sum(op.get("confidence", 0.5) for op in openings) / len(openings)


# =============================================================================
# CPWD MAPPER
# =============================================================================

class CPWDMapper:
    """Map BOQ items to CPWD schedule."""

    def __init__(self, mapping_path: Path):
        self.mappings = {}
        self._load_mappings(mapping_path)

    def _load_mappings(self, path: Path):
        """Load CPWD mappings from CSV."""
        if not path.exists():
            logger.warning(f"CPWD mapping file not found: {path}")
            return

        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item_code = row.get("item_code", "").strip()
                    if item_code:
                        self.mappings[item_code] = {
                            "cpwd_item_no": row.get("cpwd_item_no", ""),
                            "cpwd_description": row.get("cpwd_description", ""),
                            "cpwd_unit": row.get("cpwd_unit", ""),
                            "rate_inr": float(row.get("rate_inr", 0) or 0),
                        }
            logger.info(f"  Loaded {len(self.mappings)} CPWD mappings")
        except Exception as e:
            logger.warning(f"Could not load CPWD mappings: {e}")

    def map_items(self, items: List[BOQItem]) -> Tuple[List[Dict], int, int, List[str]]:
        """Map BOQ items to CPWD. Returns (mapped_items, mapped_count, unmapped_count, unmapped_codes)."""
        mapped = []
        mapped_count = 0
        unmapped_count = 0
        unmapped_codes = []

        for item in items:
            item_dict = item.to_dict()

            if item.item_code in self.mappings:
                m = self.mappings[item.item_code]
                item_dict["cpwd_item_no"] = m["cpwd_item_no"]
                item_dict["cpwd_description"] = m["cpwd_description"]
                item_dict["cpwd_unit"] = m["cpwd_unit"]
                item_dict["is_mapped"] = True
                mapped_count += 1
            else:
                item_dict["cpwd_item_no"] = ""
                item_dict["cpwd_description"] = ""
                item_dict["cpwd_unit"] = ""
                item_dict["is_mapped"] = False
                unmapped_count += 1
                if item.item_code not in unmapped_codes:
                    unmapped_codes.append(item.item_code)

            mapped.append(item_dict)

        return mapped, mapped_count, unmapped_count, unmapped_codes


# =============================================================================
# PROJECT RUNNER
# =============================================================================

class ProjectRunner:
    """V1 Finish BOQ Pipeline Runner."""

    def __init__(
        self,
        project_dir: Path,
        output_dir: Optional[Path] = None,
        profile: str = "typical",
    ):
        self.project_dir = Path(project_dir)
        self.project_id = self.project_dir.name
        self.output_dir = output_dir or Path("out") / self.project_id
        self.profile = profile
        self.rules_dir = Path(__file__).parent.parent / "rules"
        self.rates_dir = Path(__file__).parent.parent / "rates"

        # Will be populated during run
        self.config: Optional[ProjectConfig] = None
        self.summary: Optional[QASummary] = None
        self.rooms_data: List[Dict] = []
        self.openings_data: List[Dict] = []
        self.all_boq_items: List[BOQItem] = []

    def run(self) -> QASummary:
        """Run the complete V1 pipeline."""
        start_time = datetime.now()

        # Initialize summary
        self.summary = QASummary(
            project_id=self.project_id,
            run_timestamp=start_time.isoformat(),
            profile=self.profile,
        )

        try:
            # Step 1: Load config and profile
            logger.info(f"📁 Loading project: {self.project_dir}")
            self._load_config()

            # Step 2: Load rooms
            logger.info("🏠 Loading rooms...")
            self._load_rooms()

            # Step 3: Load openings
            logger.info("🚪 Loading openings...")
            self._load_openings()

            # Step 4: Calculate finish quantities
            logger.info("🎨 Calculating finishes...")
            self._calculate_finishes()

            # Step 5: Process openings schedule
            logger.info("📋 Processing openings schedule...")
            self._process_openings()

            # Step 6: Merge BOQ
            logger.info("📊 Merging BOQ...")
            self._merge_boq()

            # Step 7: CPWD mapping
            logger.info("🏛️  Mapping to CPWD...")
            self._map_cpwd()

            # Step 8: Generate outputs
            logger.info("💾 Saving outputs...")
            self._save_outputs()

            # Step 9: Generate warnings and actions
            self._generate_warnings_and_actions()

            self.summary.status = "success" if not self.summary.errors else "partial"

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.summary.status = "failed"
            self.summary.errors.append(str(e))
            traceback.print_exc()

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️  Completed in {duration:.1f}s")

        # Generate and print summary
        self._generate_summary_md()
        self._print_summary()

        return self.summary

    def _load_config(self):
        """Load project configuration."""
        self.config = ProjectConfig(
            project_id=self.project_id,
            profile=self.profile,
        )

        # Load profile from assumptions.yaml
        if yaml:
            assumptions_file = self.rules_dir / "assumptions.yaml"
            if assumptions_file.exists():
                try:
                    with open(assumptions_file) as f:
                        data = yaml.safe_load(f)
                    profiles = data.get("profiles", {})
                    profile_config = profiles.get(self.profile, {})
                    self.config.wastage_multiplier = profile_config.get("wastage_factor_multiplier", 1.0)
                    logger.info(f"  Profile: {self.profile} (wastage multiplier: {self.config.wastage_multiplier})")
                except Exception as e:
                    self.summary.warnings.append(f"Could not load profile: {e}")

        # Load project-specific config
        config_file = self.project_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    overrides = json.load(f)
                for key, value in overrides.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                logger.info(f"  Loaded project config overrides")
            except Exception as e:
                self.summary.warnings.append(f"Could not load config.json: {e}")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "boq").mkdir(exist_ok=True)

    def _load_rooms(self):
        """Load room data."""
        rooms_file = self.project_dir / "rooms.json"

        if rooms_file.exists():
            try:
                with open(rooms_file) as f:
                    data = json.load(f)
                self.rooms_data = data.get("rooms", data) if isinstance(data, dict) else data
                logger.info(f"  Loaded {len(self.rooms_data)} rooms")
            except Exception as e:
                self.summary.errors.append(f"Could not load rooms.json: {e}")
                return
        else:
            self.summary.warnings.append("rooms.json not found - cannot calculate finishes")
            return

        # Populate summary
        self.summary.rooms_count = len(self.rooms_data)
        for room in self.rooms_data:
            area = room.get("area_sqm", 0)
            self.summary.total_area_sqm += area

            room_type = room.get("room_type", room.get("label", "unknown")).lower()
            self.summary.rooms_by_type[room_type] = self.summary.rooms_by_type.get(room_type, 0) + 1

    def _load_openings(self):
        """Load openings data."""
        openings_file = self.project_dir / "openings.json"

        if openings_file.exists():
            try:
                with open(openings_file) as f:
                    data = json.load(f)
                self.openings_data = data.get("openings", []) if isinstance(data, dict) else data
                logger.info(f"  Loaded {len(self.openings_data)} openings")
            except Exception as e:
                self.summary.warnings.append(f"Could not load openings.json: {e}")
        else:
            self.summary.warnings.append("openings.json not found - openings schedule will be empty")

    def _calculate_finishes(self):
        """Calculate finish quantities using templates."""
        templates_path = self.rules_dir / "finish_templates.yaml"
        calculator = FinishCalculator(templates_path, self.config)

        finish_totals = FinishTotals()
        floor_by_type = defaultdict(float)

        for room in self.rooms_data:
            room_openings = [
                op for op in self.openings_data
                if op.get("room_left_id") == room.get("id") or op.get("room_right_id") == room.get("id")
            ]

            items, room_summary = calculator.calculate_room_finishes(room, room_openings)
            self.all_boq_items.extend(items)

            # Aggregate
            floor_type = room_summary.get("floor_type", "unknown")
            if floor_type:
                floor_by_type[floor_type] += room_summary.get("floor_area_sqm", 0)

            finish_totals.wall_tile_area_sqm += room_summary.get("wall_tile_sqm", 0)
            finish_totals.wall_dado_area_sqm += room_summary.get("wall_dado_sqm", 0)
            finish_totals.wall_paint_area_sqm += room_summary.get("wall_paint_sqm", 0)
            finish_totals.ceiling_paint_area_sqm += room_summary.get("ceiling_sqm", 0)
            finish_totals.skirting_length_m += room_summary.get("skirting_m", 0)
            finish_totals.waterproofing_area_sqm += room_summary.get("waterproofing_sqm", 0)

            finish_totals.room_finishes.append(room_summary)

        finish_totals.floor_by_type = dict(floor_by_type)
        self.summary.finishes = finish_totals

        logger.info(f"  Generated {len(self.all_boq_items)} finish items")

    def _process_openings(self):
        """Process openings into schedule."""
        processor = OpeningsProcessor(self.config)
        items, openings_summary = processor.process(self.openings_data)
        self.all_boq_items.extend(items)
        self.summary.openings = openings_summary

        logger.info(f"  Doors: {openings_summary.total_doors}, Windows: {openings_summary.total_windows}")

    def _merge_boq(self):
        """Consolidate BOQ items."""
        # Count by category
        category_counts = defaultdict(int)
        for item in self.all_boq_items:
            category_counts[item.category] += 1

            # Confidence breakdown
            if item.confidence >= 0.75:
                self.summary.high_confidence_count += 1
            elif item.confidence >= 0.50:
                self.summary.medium_confidence_count += 1
            else:
                self.summary.low_confidence_count += 1

        self.summary.boq_lines_count = len(self.all_boq_items)
        self.summary.boq_by_category = dict(category_counts)

    def _map_cpwd(self):
        """Map BOQ items to CPWD."""
        mapping_path = self.rates_dir / "cpwd_mapping.csv"
        mapper = CPWDMapper(mapping_path)

        mapped_items, mapped, unmapped, unmapped_codes = mapper.map_items(self.all_boq_items)

        self.summary.cpwd_mapped = mapped
        self.summary.cpwd_unmapped = unmapped
        self.summary.cpwd_coverage_percent = (mapped / len(self.all_boq_items) * 100) if self.all_boq_items else 0
        self.summary.unmapped_codes = unmapped_codes[:15]  # Top 15

        # Save CPWD mapped BOQ
        cpwd_path = self.output_dir / "boq" / "boq_with_cpwd_map.csv"
        self._save_mapped_csv(mapped_items, cpwd_path)

        logger.info(f"  CPWD coverage: {self.summary.cpwd_coverage_percent:.0f}%")

    def _save_outputs(self):
        """Save all output files."""
        boq_dir = self.output_dir / "boq"

        # 1. boq_quantities.csv (main BOQ)
        boq_path = boq_dir / "boq_quantities.csv"
        self._save_boq_csv(self.all_boq_items, boq_path)
        self.summary.output_files.append(str(boq_path))

        # 2. finishes_boq.csv
        finish_items = [i for i in self.all_boq_items if i.category.startswith("finishes_")]
        finishes_path = boq_dir / "finishes_boq.csv"
        self._save_boq_csv(finish_items, finishes_path)
        self.summary.output_files.append(str(finishes_path))

        # 3. openings_schedule.csv
        openings_path = boq_dir / "openings_schedule.csv"
        self._save_openings_schedule(openings_path)
        self.summary.output_files.append(str(openings_path))

        # 4. summary.json
        summary_json = self.output_dir / "summary.json"
        with open(summary_json, "w") as f:
            json.dump(self.summary.to_dict(), f, indent=2, default=str)
        self.summary.output_files.append(str(summary_json))

        # 5. config_used.json
        config_json = self.output_dir / "config_used.json"
        with open(config_json, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        self.summary.output_files.append(str(config_json))

    def _save_boq_csv(self, items: List[BOQItem], path: Path):
        """Save BOQ items to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(BOQItem.csv_headers())
            for item in items:
                writer.writerow(item.to_csv_row())

    def _save_mapped_csv(self, items: List[Dict], path: Path):
        """Save CPWD-mapped items to CSV."""
        headers = list(items[0].keys()) if items else []
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(items)
        self.summary.output_files.append(str(path))

    def _save_openings_schedule(self, path: Path):
        """Save openings schedule to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "Tag", "Width (mm)", "Height (mm)", "Quantity", "Material"])

            for door in self.summary.openings.door_schedule:
                writer.writerow([
                    "Door", door["tag"], door["width_mm"], door["height_mm"],
                    door["quantity"], door["material"]
                ])
            for win in self.summary.openings.window_schedule:
                writer.writerow([
                    "Window", win["tag"], win["width_mm"], win["height_mm"],
                    win["quantity"], win["material"]
                ])

    def _generate_warnings_and_actions(self):
        """Generate warnings and next actions based on results."""
        s = self.summary

        # Low confidence items
        if s.low_confidence_count > 0:
            s.warnings.append(f"scale_uncertain: {s.low_confidence_count} items have low confidence (<50%)")
            s.next_actions.append("Review low-confidence items and verify quantities manually")

        # Unassigned openings
        unassigned = sum(1 for op in self.openings_data if not op.get("room_left_id"))
        if unassigned > 0:
            s.warnings.append(f"openings_unassigned: {unassigned} openings not assigned to rooms")
            s.next_actions.append("Assign unassigned openings to rooms for accurate finish deductions")

        # Low CPWD coverage
        if s.cpwd_coverage_percent < 50:
            s.warnings.append(f"cpwd_coverage_low: Only {s.cpwd_coverage_percent:.0f}% of items mapped to CPWD")
            s.next_actions.append("Add missing CPWD mappings to rates/cpwd_mapping.csv")

        # Missing waterproofing
        wet_areas = ["toilet", "bathroom", "wc", "utility", "balcony"]
        wet_room_count = sum(1 for r in self.rooms_data if any(w in r.get("label", "").lower() or w in r.get("room_type", "").lower() for w in wet_areas))
        if wet_room_count > 0 and s.finishes.waterproofing_area_sqm == 0:
            s.warnings.append("waterproofing_missing: Wet areas detected but no waterproofing calculated")
            s.next_actions.append("Verify waterproofing requirements for wet areas")

        # Template mismatches
        unmapped_rooms = [r for r in s.finishes.room_finishes if r.get("template_used") == "default"]
        if unmapped_rooms:
            labels = [r.get("room_label") for r in unmapped_rooms[:5]]
            s.warnings.append(f"template_fallback: {len(unmapped_rooms)} rooms using default template: {labels}")
            s.next_actions.append("Add room type aliases to rules/finish_templates.yaml")

        # Default actions
        if not s.next_actions:
            s.next_actions.append("All checks passed - ready for QS review")

    def _generate_summary_md(self):
        """Generate comprehensive summary.md."""
        s = self.summary
        f = s.finishes
        o = s.openings

        md = f"""# XBOQ V1 Project Summary: {s.project_id}

**Generated:** {s.run_timestamp}
**Profile:** {s.profile.title()}
**Status:** {"✅ Success" if s.status == "success" else "⚠️ " + s.status.title()}

---

## 📁 Plans Processed

| Metric | Value |
|--------|-------|
| Plans | {s.plans_processed} |
| Pages | {s.pages_processed} |

---

## 🏠 Rooms Detected

| Metric | Value |
|--------|-------|
| Total Rooms | {s.rooms_count} |
| **Total Area** | **{s.total_area_sqm:.1f} sqm** ({s.sqft(s.total_area_sqm):.0f} sqft) |

**Rooms by Type:**
"""
        for rtype, count in sorted(s.rooms_by_type.items()):
            md += f"- {rtype.title()}: {count}\n"

        md += f"""
---

## 🚪 Openings Schedule

| Type | Count |
|------|-------|
| Doors | {o.total_doors} |
| Windows | {o.total_windows} |
| Ventilators | {o.total_ventilators} |

### Doors by Tag
"""
        if o.doors_by_tag:
            md += "| Tag | Qty | Size (mm) | Material |\n|-----|-----|-----------|----------|\n"
            for sched in o.door_schedule:
                md += f"| {sched['tag']} | {sched['quantity']} | {sched['width_mm']}x{sched['height_mm']} | {sched['material']} |\n"
        else:
            md += "*No doors detected*\n"

        md += "\n### Windows by Tag\n"
        if o.windows_by_tag:
            md += "| Tag | Qty | Size (mm) | Material |\n|-----|-----|-----------|----------|\n"
            for sched in o.window_schedule:
                md += f"| {sched['tag']} | {sched['quantity']} | {sched['width_mm']}x{sched['height_mm']} | {sched['material']} |\n"
        else:
            md += "*No windows detected*\n"

        md += f"""
---

## 🎨 Finishes Summary

### Floor Finish by Type

| Type | Area (sqm) | Area (sqft) |
|------|------------|-------------|
"""
        total_floor = 0
        for ftype, area in f.floor_by_type.items():
            md += f"| {ftype.replace('_', ' ').title()} | {area:.1f} | {s.sqft(area):.0f} |\n"
            total_floor += area
        md += f"| **Total** | **{total_floor:.1f}** | **{s.sqft(total_floor):.0f}** |\n"

        md += f"""
### Wall Finishes

| Type | Area (sqm) | Area (sqft) | Notes |
|------|------------|-------------|-------|
| Wall Tiles (Toilet) | {f.wall_tile_area_sqm:.1f} | {s.sqft(f.wall_tile_area_sqm):.0f} | Up to 2100mm height |
| Wall Dado (Kitchen) | {f.wall_dado_area_sqm:.1f} | {s.sqft(f.wall_dado_area_sqm):.0f} | 600mm above counter |
| Wall Paint | {f.wall_paint_area_sqm:.1f} | {s.sqft(f.wall_paint_area_sqm):.0f} | Interior emulsion |

### Other Finishes

| Type | Qty | Unit |
|------|-----|------|
| Ceiling Paint | {f.ceiling_paint_area_sqm:.1f} | sqm |
| Skirting | {f.skirting_length_m:.1f} | m |
| Waterproofing | {f.waterproofing_area_sqm:.1f} | sqm |

---

## 📋 BOQ Summary

| Metric | Value |
|--------|-------|
| **Total BOQ Lines** | **{s.boq_lines_count}** |

### BOQ by Category
"""
        for cat, count in sorted(s.boq_by_category.items()):
            md += f"| {cat.replace('_', ' ').title()} | {count} |\n"

        md += f"""
---

## 🏛️ CPWD Mapping

| Metric | Value |
|--------|-------|
| Mapped Items | {s.cpwd_mapped} |
| Unmapped Items | {s.cpwd_unmapped} |
| **Coverage** | **{s.cpwd_coverage_percent:.0f}%** |

"""
        if s.unmapped_codes:
            md += "**Unmapped Item Codes:**\n"
            for code in s.unmapped_codes:
                md += f"- `{code}`\n"
            md += "\n*Add mappings to `rates/cpwd_mapping.csv`*\n"

        md += f"""
---

## 📊 Confidence Breakdown

| Level | Count | % |
|-------|-------|---|
| 🟢 High (≥75%) | {s.high_confidence_count} | {s.high_confidence_count/max(s.boq_lines_count,1)*100:.0f}% |
| 🟡 Medium (50-75%) | {s.medium_confidence_count} | {s.medium_confidence_count/max(s.boq_lines_count,1)*100:.0f}% |
| 🔴 Low (<50%) | {s.low_confidence_count} | {s.low_confidence_count/max(s.boq_lines_count,1)*100:.0f}% |

---

## ⚠️ Warnings

"""
        if s.warnings:
            for w in s.warnings:
                md += f"- {w}\n"
        else:
            md += "*No warnings*\n"

        md += f"""
---

## ✅ Next Best Actions

"""
        for i, action in enumerate(s.next_actions, 1):
            md += f"{i}. {action}\n"

        md += f"""
---

## 📁 Output Files

| File | Description |
|------|-------------|
"""
        for f_path in s.output_files:
            fname = Path(f_path).name
            md += f"| `{fname}` | |\n"

        md += f"""
---

*Generated by XBOQ Pre-Construction Engine V1*
*Architect-first Finish BOQ | India | IS Standards | CPWD Rates*
"""

        # Write summary.md
        summary_md = self.output_dir / "summary.md"
        with open(summary_md, "w") as f:
            f.write(md)
        self.summary.output_files.append(str(summary_md))

    def _print_summary(self):
        """Print summary to terminal."""
        s = self.summary
        o = s.openings
        f = s.finishes

        status_icon = "✅" if s.status == "success" else "⚠️"

        print("\n" + "="*70)
        print(f"  {status_icon} XBOQ V1 PROJECT SUMMARY: {s.project_id}")
        print("="*70)
        print(f"  Profile: {s.profile.title()}")
        print()
        print("  🏠 ROOMS")
        print(f"     Total: {s.rooms_count} rooms | {s.total_area_sqm:.1f} sqm ({s.sqft(s.total_area_sqm):.0f} sqft)")
        print()
        print("  🚪 OPENINGS")
        print(f"     Doors: {o.total_doors} | Windows: {o.total_windows} | Ventilators: {o.total_ventilators}")
        print()
        print("  🎨 FINISHES")
        total_floor = sum(f.floor_by_type.values())
        print(f"     Floor: {total_floor:.1f} sqm | Wall Tiles: {f.wall_tile_area_sqm:.1f} sqm | Dado: {f.wall_dado_area_sqm:.1f} sqm")
        print(f"     Wall Paint: {f.wall_paint_area_sqm:.1f} sqm | Ceiling: {f.ceiling_paint_area_sqm:.1f} sqm")
        print()
        print("  📋 BOQ")
        print(f"     Total Lines: {s.boq_lines_count}")
        print(f"     Confidence: 🟢 {s.high_confidence_count} | 🟡 {s.medium_confidence_count} | 🔴 {s.low_confidence_count}")
        print()
        print("  🏛️ CPWD MAPPING")
        print(f"     Coverage: {s.cpwd_coverage_percent:.0f}% ({s.cpwd_mapped} mapped, {s.cpwd_unmapped} unmapped)")

        if s.warnings:
            print()
            print("  ⚠️ WARNINGS")
            for w in s.warnings[:5]:
                print(f"     - {w}")

        print()
        print("  ✅ NEXT ACTIONS")
        for i, action in enumerate(s.next_actions[:3], 1):
            print(f"     {i}. {action}")

        print()
        print(f"  📁 Output: {self.output_dir}")
        print("="*70 + "\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XBOQ V1 - Architect-first Finish BOQ Pipeline (India)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V1 Scope:
  - Rooms + areas
  - Openings schedule (doors/windows/ventilators)
  - Finishes takeoff using rules/finish_templates.yaml
  - BOQ outputs in canonical schema
  - CPWD mapping coverage report

Examples:
    python scripts/run_project.py --project_dir data/projects/sample_project
    python scripts/run_project.py --project_dir data/projects/my_project --profile premium
    python scripts/run_project.py --project_dir data/projects/my_project --output out/custom
        """
    )

    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to project directory containing rooms.json and openings.json",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: out/<project_id>)",
    )
    parser.add_argument(
        "--profile",
        choices=["conservative", "typical", "premium"],
        default="typical",
        help="Estimation profile (default: typical)",
    )

    args = parser.parse_args()

    # Validate project directory
    if not args.project_dir.exists():
        print(f"❌ Project directory not found: {args.project_dir}")
        sys.exit(1)

    # Run pipeline
    runner = ProjectRunner(
        project_dir=args.project_dir,
        output_dir=args.output,
        profile=args.profile,
    )

    summary = runner.run()

    # Exit code based on status
    if summary.status == "failed":
        sys.exit(1)
    elif summary.status == "partial":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
