"""
Estimator Paranoia Engine for XBOQ.

Applies experienced estimator knowledge to infer implied scope items
that MUST be included when certain room types or elements are present.

India-specific construction estimation accuracy layer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from pathlib import Path
import logging
import yaml
import re
import math

logger = logging.getLogger(__name__)


class InferencePriority(Enum):
    """Priority level for inferred items."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class InferredItem:
    """Single item inferred by paranoia rules."""
    inference_id: str
    trigger_type: str  # "room", "element", "building"
    trigger_name: str  # e.g., "toilet", "window"
    trigger_source: str  # Room ID, element ID, or "building"
    item_code: str
    description: str
    boq_code: str
    unit: str
    quantity: float
    quantity_formula: str
    rate_range: tuple  # (min, max) INR
    priority: InferencePriority
    confidence: float  # 0-100
    notes: str = ""
    source_rule: str = ""


@dataclass
class ParanoiaReport:
    """Complete paranoia inference report."""
    project_id: str
    inferred_items: List[InferredItem] = field(default_factory=list)
    scope_additions: List[str] = field(default_factory=list)  # Scope register additions
    provisional_boq: List[Dict] = field(default_factory=list)  # BOQ additions
    summary: Dict[str, Any] = field(default_factory=dict)


class EstimatorParanoiaEngine:
    """
    Applies estimator paranoia rules to infer missing scope items.

    Encodes experienced estimator knowledge about implied items that
    are ALWAYS required when certain room types or elements exist.
    """

    def __init__(self, rules_path: Optional[Path] = None):
        """Initialize with rules file."""
        if rules_path is None:
            rules_path = Path(__file__).parent.parent.parent / "rules" / "estimator_paranoia.yaml"

        self.rules = self._load_rules(rules_path)
        self.inferred_items: List[InferredItem] = []
        self._inference_counter = 0

    def _load_rules(self, rules_path: Path) -> Dict:
        """Load paranoia rules from YAML."""
        try:
            if rules_path.exists():
                with open(rules_path) as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load paranoia rules: {e}")

        # Return minimal default rules
        return {
            "room_rules": {},
            "element_rules": {},
            "building_rules": {},
            "confidence_levels": {
                "critical": {"min_confidence": 95},
                "high": {"min_confidence": 85},
                "medium": {"min_confidence": 70},
                "low": {"min_confidence": 50},
            },
            "defaults": {
                "ceiling_height_m": 3.0,
                "wall_thickness_mm": 230,
            }
        }

    def apply_paranoia_rules(
        self,
        project_id: str,
        extraction_results: List[Dict],
        project_graph: Dict,
        scope_register: Dict,
        existing_boq: List[Dict],
    ) -> ParanoiaReport:
        """
        Apply all paranoia rules and generate inferred items.

        Args:
            project_id: Project identifier
            extraction_results: Page extraction results
            project_graph: Joined project graph
            scope_register: Current scope register
            existing_boq: Existing BOQ entries

        Returns:
            Complete paranoia report
        """
        self.inferred_items = []
        self._inference_counter = 0

        # Track what's already in BOQ to avoid duplicates
        existing_items = self._index_existing_boq(existing_boq)

        # Collect all rooms and elements
        rooms = self._collect_rooms(extraction_results, project_graph)
        elements = self._collect_elements(extraction_results, project_graph)

        # Apply room-based rules
        for room in rooms:
            self._apply_room_rules(room, existing_items)

        # Apply element-based rules
        for element in elements:
            self._apply_element_rules(element, existing_items)

        # Apply building-level rules
        self._apply_building_rules(rooms, elements, existing_items)

        # Build report
        report = self._build_report(project_id, scope_register)

        return report

    def _generate_inference_id(self) -> str:
        """Generate unique inference ID."""
        self._inference_counter += 1
        return f"INF-{self._inference_counter:04d}"

    def _index_existing_boq(self, boq_entries: List[Dict]) -> Set[str]:
        """Create index of existing BOQ items to avoid duplicates."""
        existing = set()
        for entry in boq_entries:
            # Index by item type keywords
            item_type = entry.get("item_type", "").lower()
            existing.add(item_type)

            # Also index by boq_code if present
            boq_code = entry.get("boq_code", "").upper()
            if boq_code:
                existing.add(boq_code)

        return existing

    def _collect_rooms(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
    ) -> List[Dict]:
        """Collect all rooms from extraction results and project graph."""
        rooms = []

        # From extraction results
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                page_id = page.get("page_id", "unknown")
                for room in page.get("rooms", []):
                    room_data = {
                        "id": room.get("id", f"R{len(rooms)+1:03d}"),
                        "label": room.get("label", "unlabeled"),
                        "area_sqm": room.get("area_sqm", 0),
                        "perimeter_m": room.get("perimeter_m", 0),
                        "source": f"floor_plan_{page_id}",
                    }
                    # Estimate perimeter if not provided
                    if room_data["perimeter_m"] == 0 and room_data["area_sqm"] > 0:
                        # Assume 1.5:1 aspect ratio
                        area = room_data["area_sqm"]
                        room_data["perimeter_m"] = 2 * (math.sqrt(area * 1.5) + math.sqrt(area / 1.5))
                    rooms.append(room_data)

        # From project graph (may have additional data)
        for room in project_graph.get("rooms", []):
            room_id = room.get("id", "")
            # Check if already collected
            if not any(r["id"] == room_id for r in rooms):
                rooms.append({
                    "id": room_id,
                    "label": room.get("label", "unlabeled"),
                    "area_sqm": room.get("area_sqm", 0),
                    "perimeter_m": room.get("perimeter_m", 0),
                    "source": "project_graph",
                })

        return rooms

    def _collect_elements(
        self,
        extraction_results: List[Dict],
        project_graph: Dict,
    ) -> List[Dict]:
        """Collect all elements (doors, windows, columns, etc.)."""
        elements = []

        # From extraction results
        for page in extraction_results:
            if page.get("page_type") == "floor_plan":
                page_id = page.get("page_id", "unknown")

                # Doors
                for door in page.get("doors", []):
                    elements.append({
                        "type": "door",
                        "id": door.get("tag", door.get("id", f"D{len(elements)+1:03d}")),
                        "width_mm": door.get("width_mm", 900),
                        "height_mm": door.get("height_mm", 2100),
                        "source": f"floor_plan_{page_id}",
                    })

                # Windows
                for window in page.get("windows", []):
                    elements.append({
                        "type": "window",
                        "id": window.get("tag", window.get("id", f"W{len(elements)+1:03d}")),
                        "width_mm": window.get("width_mm", 1200),
                        "height_mm": window.get("height_mm", 1200),
                        "source": f"floor_plan_{page_id}",
                    })

                # Columns
                for col in page.get("columns", []):
                    elements.append({
                        "type": "column",
                        "id": col.get("tag", col.get("id", f"C{len(elements)+1:03d}")),
                        "size": col.get("size", "300x300"),
                        "source": f"floor_plan_{page_id}",
                    })

        # From project graph elements
        for elem_type in ["doors", "windows", "columns", "beams"]:
            for elem in project_graph.get("elements", {}).get(elem_type, []):
                elem_id = elem.get("id", elem.get("tag", ""))
                if not any(e["id"] == elem_id for e in elements):
                    elements.append({
                        "type": elem_type.rstrip("s"),  # doors -> door
                        "id": elem_id,
                        "source": "project_graph",
                        **elem,
                    })

        return elements

    def _apply_room_rules(self, room: Dict, existing_items: Set[str]) -> None:
        """Apply room-based paranoia rules."""
        room_rules = self.rules.get("room_rules", {})
        label = room.get("label", "").lower()
        room_id = room.get("id", "unknown")
        area = room.get("area_sqm", 0)
        perimeter = room.get("perimeter_m", 0)

        defaults = self.rules.get("defaults", {})
        ceiling_height = defaults.get("ceiling_height_m", 3.0)

        for rule_name, rule_data in room_rules.items():
            triggers = rule_data.get("triggers", [])

            # Check if room matches any trigger
            if not any(t.lower() in label for t in triggers):
                continue

            # Room matches - apply implied items
            for item in rule_data.get("implied_items", []):
                boq_code = item.get("boq_code", "")

                # Skip if already exists
                if boq_code in existing_items:
                    continue
                if item.get("item", "").lower() in existing_items:
                    continue

                # Calculate quantity
                quantity = self._calculate_quantity(
                    item.get("quantity_formula", "1"),
                    area=area,
                    perimeter=perimeter,
                    height=ceiling_height,
                )

                # Get confidence from priority
                priority_str = item.get("priority", "medium")
                priority = InferencePriority(priority_str)
                conf_levels = self.rules.get("confidence_levels", {})
                confidence = conf_levels.get(priority_str, {}).get("min_confidence", 70)

                rate_range = item.get("typical_rate_range", [0, 0])

                self.inferred_items.append(InferredItem(
                    inference_id=self._generate_inference_id(),
                    trigger_type="room",
                    trigger_name=rule_name,
                    trigger_source=room_id,
                    item_code=item.get("item", ""),
                    description=item.get("description", ""),
                    boq_code=boq_code,
                    unit=item.get("unit", "nos"),
                    quantity=quantity,
                    quantity_formula=item.get("quantity_formula", "1"),
                    rate_range=tuple(rate_range) if rate_range else (0, 0),
                    priority=priority,
                    confidence=confidence,
                    notes=item.get("notes", ""),
                    source_rule=f"room_rules.{rule_name}",
                ))

    def _apply_element_rules(self, element: Dict, existing_items: Set[str]) -> None:
        """Apply element-based paranoia rules."""
        element_rules = self.rules.get("element_rules", {})
        elem_type = element.get("type", "").lower()
        elem_id = element.get("id", "unknown")

        if elem_type not in element_rules:
            return

        rule_data = element_rules[elem_type]
        defaults = self.rules.get("defaults", {})

        # Get element dimensions
        width_mm = element.get("width_mm", defaults.get(f"{elem_type}_width_mm", 1000))
        height_mm = element.get("height_mm", defaults.get(f"{elem_type}_height_mm", 2100))

        for item in rule_data.get("implied_items", []):
            boq_code = item.get("boq_code", "")

            if boq_code in existing_items:
                continue
            if item.get("item", "").lower() in existing_items:
                continue

            # Calculate quantity based on element
            quantity = self._calculate_element_quantity(
                item.get("quantity_formula", "1"),
                elem_type=elem_type,
                width_mm=width_mm,
                height_mm=height_mm,
            )

            priority_str = item.get("priority", "medium")
            priority = InferencePriority(priority_str)
            conf_levels = self.rules.get("confidence_levels", {})
            confidence = conf_levels.get(priority_str, {}).get("min_confidence", 70)

            rate_range = item.get("typical_rate_range", [0, 0])

            self.inferred_items.append(InferredItem(
                inference_id=self._generate_inference_id(),
                trigger_type="element",
                trigger_name=elem_type,
                trigger_source=elem_id,
                item_code=item.get("item", ""),
                description=item.get("description", ""),
                boq_code=boq_code,
                unit=item.get("unit", "nos"),
                quantity=quantity,
                quantity_formula=item.get("quantity_formula", "1"),
                rate_range=tuple(rate_range) if rate_range else (0, 0),
                priority=priority,
                confidence=confidence,
                notes=item.get("notes", ""),
                source_rule=f"element_rules.{elem_type}",
            ))

    def _apply_building_rules(
        self,
        rooms: List[Dict],
        elements: List[Dict],
        existing_items: Set[str],
    ) -> None:
        """Apply building-level paranoia rules."""
        building_rules = self.rules.get("building_rules", {})
        standard_provisions = building_rules.get("standard_provisions", [])

        # Calculate building-level metrics
        total_floor_area = sum(r.get("area_sqm", 0) for r in rooms)
        total_perimeter = sum(r.get("perimeter_m", 0) for r in rooms)

        # Estimate external wall area (rough)
        ceiling_height = self.rules.get("defaults", {}).get("ceiling_height_m", 3.0)
        # Assume external perimeter is ~40% of total room perimeters
        external_perimeter = total_perimeter * 0.4
        external_wall_area = external_perimeter * ceiling_height

        for item in standard_provisions:
            boq_code = item.get("boq_code", "")

            if boq_code in existing_items:
                continue
            if item.get("item", "").lower() in existing_items:
                continue

            quantity = self._calculate_building_quantity(
                item.get("quantity_formula", "1"),
                floor_area=total_floor_area,
                external_wall_area=external_wall_area,
                perimeter=external_perimeter,
            )

            priority_str = item.get("priority", "medium")
            priority = InferencePriority(priority_str)
            conf_levels = self.rules.get("confidence_levels", {})
            confidence = conf_levels.get(priority_str, {}).get("min_confidence", 70)

            rate_range = item.get("typical_rate_range", [0, 0])

            self.inferred_items.append(InferredItem(
                inference_id=self._generate_inference_id(),
                trigger_type="building",
                trigger_name="standard_provision",
                trigger_source="building",
                item_code=item.get("item", ""),
                description=item.get("description", ""),
                boq_code=boq_code,
                unit=item.get("unit", "ls"),
                quantity=quantity,
                quantity_formula=item.get("quantity_formula", "1"),
                rate_range=tuple(rate_range) if rate_range else (0, 0),
                priority=priority,
                confidence=confidence,
                notes=item.get("notes", ""),
                source_rule="building_rules.standard_provisions",
            ))

    def _calculate_quantity(
        self,
        formula: str,
        area: float = 0,
        perimeter: float = 0,
        height: float = 3.0,
    ) -> float:
        """Calculate quantity from formula using room metrics."""
        try:
            # Simple formula evaluation
            formula = formula.lower()

            # Replace variables
            formula = formula.replace("floor_area", str(area))
            formula = formula.replace("perimeter", str(perimeter))
            formula = formula.replace("ceiling_height", str(height))
            formula = formula.replace("height", str(height))

            # Handle max() function
            max_match = re.search(r"max\(([^,]+),([^)]+)\)", formula)
            if max_match:
                a = float(eval(max_match.group(1).strip()))
                b = float(eval(max_match.group(2).strip()))
                formula = formula.replace(max_match.group(0), str(max(a, b)))

            # Evaluate
            result = eval(formula)
            return round(float(result), 2)
        except Exception as e:
            logger.debug(f"Formula evaluation failed: {formula} - {e}")
            return 1.0

    def _calculate_element_quantity(
        self,
        formula: str,
        elem_type: str,
        width_mm: float,
        height_mm: float,
    ) -> float:
        """Calculate quantity from formula for element."""
        try:
            formula = formula.lower()

            # Convert to meters
            width_m = width_mm / 1000
            height_m = height_mm / 1000
            area_sqm = width_m * height_m

            formula = formula.replace("window_width", str(width_m))
            formula = formula.replace("door_width", str(width_m))
            formula = formula.replace("frame_area", str(area_sqm))
            formula = formula.replace("door_area", str(area_sqm))
            formula = formula.replace("element_area", str(area_sqm))

            result = eval(formula)
            return round(float(result), 2)
        except Exception:
            return 1.0

    def _calculate_building_quantity(
        self,
        formula: str,
        floor_area: float,
        external_wall_area: float,
        perimeter: float,
    ) -> float:
        """Calculate quantity for building-level items."""
        try:
            formula = formula.lower()

            formula = formula.replace("external_wall_area", str(external_wall_area))
            formula = formula.replace("plot_area", str(floor_area))
            formula = formula.replace("plot_perimeter", str(perimeter))

            result = eval(formula)
            return round(float(result), 2)
        except Exception:
            return 1.0

    def _build_report(self, project_id: str, scope_register: Dict) -> ParanoiaReport:
        """Build paranoia report with scope additions and BOQ items."""
        report = ParanoiaReport(project_id=project_id)
        report.inferred_items = self.inferred_items

        # Group by trigger type for summary
        by_trigger = {}
        by_priority = {}

        for item in self.inferred_items:
            # By trigger
            trigger = item.trigger_name
            if trigger not in by_trigger:
                by_trigger[trigger] = []
            by_trigger[trigger].append(item)

            # By priority
            priority = item.priority.value
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(item)

        # Build scope additions
        scope_items = scope_register.get("items", [])
        existing_scope = {s.get("subpackage", "").lower() for s in scope_items}

        for item in self.inferred_items:
            # Map to scope packages
            scope_key = self._map_to_scope_package(item.item_code)
            if scope_key and scope_key.lower() not in existing_scope:
                report.scope_additions.append(scope_key)

        # Build provisional BOQ entries
        for item in self.inferred_items:
            if item.priority in [InferencePriority.CRITICAL, InferencePriority.HIGH]:
                boq_entry = {
                    "item_code": item.boq_code,
                    "description": item.description,
                    "unit": item.unit,
                    "quantity": item.quantity,
                    "rate_min": item.rate_range[0] if item.rate_range else 0,
                    "rate_max": item.rate_range[1] if item.rate_range else 0,
                    "source": "PARANOIA_RULE",
                    "confidence": item.confidence,
                    "trigger": f"{item.trigger_type}:{item.trigger_name}",
                    "status": "INFERRED",
                    "notes": item.notes,
                }
                report.provisional_boq.append(boq_entry)

        # Summary
        report.summary = {
            "total_inferred": len(self.inferred_items),
            "by_trigger": {k: len(v) for k, v in by_trigger.items()},
            "by_priority": {k: len(v) for k, v in by_priority.items()},
            "scope_additions": len(report.scope_additions),
            "provisional_boq_items": len(report.provisional_boq),
            "estimated_value_range": self._calculate_value_range(report.provisional_boq),
        }

        return report

    def _map_to_scope_package(self, item_code: str) -> Optional[str]:
        """Map inferred item to scope package."""
        mappings = {
            "waterproofing": ["wp_toilet", "wp_terrace", "wp_sunken", "wp_kitchen", "wp_parapet"],
            "floor_trap": ["ft_toilet", "drain_terrace"],
            "railing": ["rail_balcony", "rail_stair"],
            "plaster": ["plaster_shaft", "plaster_parapet", "plaster_stair", "plaster_column", "plaster_beam"],
            "paint": ["paint_shaft", "paint_parapet", "paint_stair", "paint_column", "paint_beam"],
            "tiles": ["tile_antiskid", "tile_dado", "tile_outdoor", "tile_kitchen"],
            "granite": ["granite_kitchen", "sill_window"],
            "hardware": ["hw_door", "hw_window"],
            "scaffolding": ["scaffold"],
        }

        item_lower = item_code.lower()
        for scope_pkg, keywords in mappings.items():
            if any(kw in item_lower for kw in keywords):
                return scope_pkg

        return None

    def _calculate_value_range(self, boq_items: List[Dict]) -> Dict[str, float]:
        """Calculate estimated value range for provisional items."""
        min_value = 0
        max_value = 0

        for item in boq_items:
            qty = item.get("quantity", 0)
            rate_min = item.get("rate_min", 0)
            rate_max = item.get("rate_max", 0)
            min_value += qty * rate_min
            max_value += qty * rate_max

        return {
            "min_inr": round(min_value, 0),
            "max_inr": round(max_value, 0),
        }


def run_paranoia_engine(
    project_id: str,
    extraction_results: List[Dict],
    project_graph: Dict,
    scope_register: Dict,
    existing_boq: List[Dict],
    rules_path: Optional[Path] = None,
) -> ParanoiaReport:
    """
    Run estimator paranoia engine.

    Args:
        project_id: Project identifier
        extraction_results: Page extraction results
        project_graph: Joined project graph
        scope_register: Current scope register
        existing_boq: Existing BOQ entries
        rules_path: Optional path to rules file

    Returns:
        Complete paranoia report
    """
    engine = EstimatorParanoiaEngine(rules_path)
    return engine.apply_paranoia_rules(
        project_id=project_id,
        extraction_results=extraction_results,
        project_graph=project_graph,
        scope_register=scope_register,
        existing_boq=existing_boq,
    )
