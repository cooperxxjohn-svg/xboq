"""
Quantity Provenance Model

Every BOQ line must have explicit provenance tracking:
- source_pages: which drawing pages the quantity came from
- method: how the quantity was derived
- confidence: 0-1 score
- scale_basis: how scale was determined
- geometry_refs: IDs to actual detected geometry

This enforces evidence-first estimation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class ProvenanceMethod(Enum):
    """How quantity was derived."""
    POLYGON = "polygon"              # Area from closed polygon detection
    CENTERLINE = "centerline"        # Length from centerline extraction
    SYMBOL_COUNT = "symbol_count"    # Count from symbol/object detection
    SCHEDULE_TABLE = "schedule_table"  # Parsed from schedule/table in drawing
    DIMENSION_TEXT = "dimension_text"  # Extracted from dimension annotations
    TEXT_ONLY = "text_only"          # Room label only, no geometry
    TEMPLATE = "template"            # From template/typical assumptions
    ALLOWANCE = "allowance"          # Provisional/allowance item
    MANUAL = "manual"                # Manually entered
    INFERRED = "inferred"            # Inferred from other data


class ScaleBasis(Enum):
    """How scale was determined."""
    DIMENSION_INFERRED = "dimension_inferred"  # Calculated from dimension text vs geometry
    SCALE_NOTE = "scale_note"                  # From scale notation on drawing
    MANUAL = "manual"                          # Manually specified
    UNKNOWN = "unknown"                        # Could not determine scale


# Methods that count as "measured" (geometry-backed)
MEASURED_METHODS = {
    ProvenanceMethod.POLYGON,
    ProvenanceMethod.CENTERLINE,
    ProvenanceMethod.SYMBOL_COUNT,
    ProvenanceMethod.SCHEDULE_TABLE,
    ProvenanceMethod.DIMENSION_TEXT,
}

# Methods that are "inferred" (not geometry-backed)
INFERRED_METHODS = {
    ProvenanceMethod.TEXT_ONLY,
    ProvenanceMethod.TEMPLATE,
    ProvenanceMethod.ALLOWANCE,
    ProvenanceMethod.MANUAL,
    ProvenanceMethod.INFERRED,
}

# Minimum confidence threshold for measured items
MEASURED_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class GeometryRef:
    """Reference to detected geometry."""
    ref_type: str  # polygon, line, symbol, table_cell
    ref_id: str    # Unique ID
    page: int      # Page number (0-indexed)
    bbox: List[float] = field(default_factory=list)  # [x1, y1, x2, y2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_type": self.ref_type,
            "ref_id": self.ref_id,
            "page": self.page,
            "bbox": self.bbox,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometryRef":
        return cls(
            ref_type=data.get("ref_type", "unknown"),
            ref_id=data.get("ref_id", ""),
            page=data.get("page", 0),
            bbox=data.get("bbox", []),
        )


@dataclass
class QuantityProvenance:
    """
    Complete provenance record for a BOQ quantity.

    Every BOQ line MUST have this attached.
    """
    # Source tracking
    source_pages: List[int] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)

    # Method and confidence
    method: ProvenanceMethod = ProvenanceMethod.TEXT_ONLY
    confidence: float = 0.0

    # Scale basis
    scale_basis: ScaleBasis = ScaleBasis.UNKNOWN
    scale_value: Optional[float] = None  # e.g., 100 for 1:100

    # Geometry references
    geometry_refs: List[GeometryRef] = field(default_factory=list)

    # Calculation details
    calculation_notes: str = ""
    raw_value: Optional[float] = None  # Before any adjustments
    adjusted_value: Optional[float] = None  # After deductions/additions

    # Flags
    is_measured: bool = False  # Computed property
    needs_verification: bool = True
    rfi_generated: bool = False

    def __post_init__(self):
        """Compute derived properties."""
        self._update_is_measured()

    def _update_is_measured(self):
        """Update is_measured based on method and confidence."""
        self.is_measured = (
            self.method in MEASURED_METHODS and
            self.confidence >= MEASURED_CONFIDENCE_THRESHOLD and
            self.scale_basis != ScaleBasis.UNKNOWN
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pages": self.source_pages,
            "source_files": self.source_files,
            "method": self.method.value,
            "confidence": self.confidence,
            "scale_basis": self.scale_basis.value,
            "scale_value": self.scale_value,
            "geometry_refs": [g.to_dict() for g in self.geometry_refs],
            "calculation_notes": self.calculation_notes,
            "raw_value": self.raw_value,
            "adjusted_value": self.adjusted_value,
            "is_measured": self.is_measured,
            "needs_verification": self.needs_verification,
            "rfi_generated": self.rfi_generated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantityProvenance":
        method_str = data.get("method", "text_only")
        scale_str = data.get("scale_basis", "unknown")

        prov = cls(
            source_pages=data.get("source_pages", []),
            source_files=data.get("source_files", []),
            method=ProvenanceMethod(method_str) if method_str in [m.value for m in ProvenanceMethod] else ProvenanceMethod.TEXT_ONLY,
            confidence=data.get("confidence", 0.0),
            scale_basis=ScaleBasis(scale_str) if scale_str in [s.value for s in ScaleBasis] else ScaleBasis.UNKNOWN,
            scale_value=data.get("scale_value"),
            geometry_refs=[GeometryRef.from_dict(g) for g in data.get("geometry_refs", [])],
            calculation_notes=data.get("calculation_notes", ""),
            raw_value=data.get("raw_value"),
            adjusted_value=data.get("adjusted_value"),
            needs_verification=data.get("needs_verification", True),
            rfi_generated=data.get("rfi_generated", False),
        )
        prov._update_is_measured()
        return prov

    def to_csv_columns(self) -> Dict[str, str]:
        """Return columns for CSV export."""
        return {
            "source_pages": ";".join(str(p) for p in self.source_pages),
            "method": self.method.value,
            "confidence": f"{self.confidence:.2f}",
            "scale_basis": self.scale_basis.value,
            "is_measured": "YES" if self.is_measured else "NO",
            "geometry_refs": ";".join(g.ref_id for g in self.geometry_refs),
        }


class ProvenanceTracker:
    """
    Tracks provenance for all BOQ items in a project.
    """

    def __init__(self, project_id: str, output_dir: Path):
        self.project_id = project_id
        self.output_dir = Path(output_dir)
        self.items: Dict[str, QuantityProvenance] = {}  # item_id -> provenance

        # Page-level tracking
        self.page_scale_basis: Dict[int, ScaleBasis] = {}
        self.page_geometry_quality: Dict[int, float] = {}  # 0-1 score

        # Global stats
        self.total_items = 0
        self.measured_items = 0
        self.inferred_items = 0

    def add_item(self, item_id: str, provenance: QuantityProvenance):
        """Add or update provenance for an item."""
        self.items[item_id] = provenance
        self._update_stats()

    def get_item(self, item_id: str) -> Optional[QuantityProvenance]:
        """Get provenance for an item."""
        return self.items.get(item_id)

    def _update_stats(self):
        """Update summary statistics."""
        self.total_items = len(self.items)
        self.measured_items = sum(1 for p in self.items.values() if p.is_measured)
        self.inferred_items = self.total_items - self.measured_items

    def get_measured_items(self) -> Dict[str, QuantityProvenance]:
        """Get only measured items."""
        return {k: v for k, v in self.items.items() if v.is_measured}

    def get_inferred_items(self) -> Dict[str, QuantityProvenance]:
        """Get only inferred items."""
        return {k: v for k, v in self.items.items() if not v.is_measured}

    def get_measurement_coverage(self) -> float:
        """Get percentage of items that are measured."""
        if self.total_items == 0:
            return 0.0
        return self.measured_items / self.total_items

    def save(self, filename: str = "provenance.json"):
        """Save provenance data to file."""
        data = {
            "project_id": self.project_id,
            "total_items": self.total_items,
            "measured_items": self.measured_items,
            "inferred_items": self.inferred_items,
            "measurement_coverage": self.get_measurement_coverage(),
            "page_scale_basis": {str(k): v.value for k, v in self.page_scale_basis.items()},
            "page_geometry_quality": self.page_geometry_quality,
            "items": {k: v.to_dict() for k, v in self.items.items()},
        }

        prov_dir = self.output_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)

        with open(prov_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, output_dir: Path, filename: str = "provenance.json") -> "ProvenanceTracker":
        """Load provenance data from file."""
        prov_file = Path(output_dir) / "provenance" / filename

        if not prov_file.exists():
            return cls("unknown", output_dir)

        with open(prov_file) as f:
            data = json.load(f)

        tracker = cls(data.get("project_id", "unknown"), output_dir)

        for k, v in data.get("page_scale_basis", {}).items():
            tracker.page_scale_basis[int(k)] = ScaleBasis(v)

        tracker.page_geometry_quality = {
            int(k): v for k, v in data.get("page_geometry_quality", {}).items()
        }

        for item_id, prov_data in data.get("items", {}).items():
            tracker.items[item_id] = QuantityProvenance.from_dict(prov_data)

        tracker._update_stats()
        return tracker

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary report."""
        by_method = {}
        for prov in self.items.values():
            method = prov.method.value
            by_method[method] = by_method.get(method, 0) + 1

        by_scale = {}
        for prov in self.items.values():
            scale = prov.scale_basis.value
            by_scale[scale] = by_scale.get(scale, 0) + 1

        return {
            "total_items": self.total_items,
            "measured_items": self.measured_items,
            "inferred_items": self.inferred_items,
            "measurement_coverage": f"{self.get_measurement_coverage():.1%}",
            "by_method": by_method,
            "by_scale_basis": by_scale,
            "needs_verification_count": sum(1 for p in self.items.values() if p.needs_verification),
            "rfi_generated_count": sum(1 for p in self.items.values() if p.rfi_generated),
        }


def create_text_only_provenance(
    source_page: int = 0,
    source_file: str = "",
    label: str = "",
) -> QuantityProvenance:
    """Create provenance for text-only detection (room label, no geometry)."""
    return QuantityProvenance(
        source_pages=[source_page] if source_page else [],
        source_files=[source_file] if source_file else [],
        method=ProvenanceMethod.TEXT_ONLY,
        confidence=0.3,  # Low confidence
        scale_basis=ScaleBasis.UNKNOWN,
        calculation_notes=f"Label detected: {label}. No geometry measured.",
        needs_verification=True,
    )


def create_polygon_provenance(
    source_page: int,
    source_file: str,
    polygon_id: str,
    bbox: List[float],
    area_sqm: float,
    scale_basis: ScaleBasis,
    scale_value: float,
    confidence: float = 0.8,
) -> QuantityProvenance:
    """Create provenance for polygon-based area measurement."""
    return QuantityProvenance(
        source_pages=[source_page],
        source_files=[source_file],
        method=ProvenanceMethod.POLYGON,
        confidence=confidence,
        scale_basis=scale_basis,
        scale_value=scale_value,
        geometry_refs=[
            GeometryRef(
                ref_type="polygon",
                ref_id=polygon_id,
                page=source_page,
                bbox=bbox,
            )
        ],
        raw_value=area_sqm,
        adjusted_value=area_sqm,
        calculation_notes=f"Polygon area: {area_sqm:.2f} sqm at scale 1:{scale_value}",
        needs_verification=scale_basis == ScaleBasis.UNKNOWN,
    )


def create_allowance_provenance(
    description: str,
    allowance_basis: str = "standard",
) -> QuantityProvenance:
    """Create provenance for allowance/provisional items."""
    return QuantityProvenance(
        source_pages=[],
        source_files=[],
        method=ProvenanceMethod.ALLOWANCE,
        confidence=0.0,  # Zero confidence for allowances
        scale_basis=ScaleBasis.UNKNOWN,
        calculation_notes=f"ALLOWANCE: {description}. Basis: {allowance_basis}",
        needs_verification=True,
    )


def create_schedule_provenance(
    source_page: int,
    source_file: str,
    table_id: str,
    row_col: str,
    extracted_value: float,
    confidence: float = 0.9,
) -> QuantityProvenance:
    """Create provenance for schedule/table extracted data."""
    return QuantityProvenance(
        source_pages=[source_page],
        source_files=[source_file],
        method=ProvenanceMethod.SCHEDULE_TABLE,
        confidence=confidence,
        scale_basis=ScaleBasis.MANUAL,  # Schedules don't need scale
        geometry_refs=[
            GeometryRef(
                ref_type="table_cell",
                ref_id=table_id,
                page=source_page,
                bbox=[],
            )
        ],
        raw_value=extracted_value,
        adjusted_value=extracted_value,
        calculation_notes=f"From schedule table at {row_col}",
        needs_verification=False,  # Schedule data is explicit
    )
