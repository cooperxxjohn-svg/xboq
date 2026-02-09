"""
XBOQ Takeoff Pipeline
India-first preconstruction BOQ & scope extraction.

This pipeline:
1. Ingests PDF and detects sheet type (schedule vs plan)
2. Extracts structured facts from multiple sources
3. Generates BOQ items with Indian terminology
4. Produces scope checklist with detected/inferred/missing items
5. Identifies conflicts and coverage gaps
6. Returns EstimatePackage (no pricing)

ALWAYS returns usable output even if partial.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from src.models.estimate_schema import (
    BOQItem,
    Conflict,
    ConflictType,
    CoverageRecord,
    Discipline,
    DrawingMeta,
    EstimatePackage,
    Evidence,
    EvidenceSource,
    QtyStatus,
    ScopeCategory,
    ScopeItem,
    ScopeStatus,
    Severity,
    create_boq_item,
    create_conflict,
    create_evidence,
    create_scope_item,
)

# Import new scoring and analysis modules
from src.scoring.confidence import compute_confidence
from src.scoring.coverage import compute_coverage_score, compute_boq_coverage, build_coverage_records
from src.analysis.conflicts import detect_conflicts, ConflictDetector, ConflictContext

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Indian Construction Terminology
# =============================================================================

# Keywords for detecting sheet types
SCHEDULE_KEYWORDS = [
    "reinforcement schedule", "column reinforcement", "beam schedule",
    "footing schedule", "slab schedule", "bar bending schedule",
    "column schedule", "bbs", "steel schedule"
]

PLAN_KEYWORDS = [
    "foundation plan", "footing layout", "column layout", "floor plan",
    "ground floor", "first floor", "roof plan", "grid", "plinth"
]

NOTES_KEYWORDS = [
    "general notes", "notes:", "note:", "specification", "spec:",
    "special requirements", "important"
]

# Standard scope items for RCC buildings (India)
STANDARD_RCC_SCOPE = [
    # Earthwork
    ("earthwork", "Excavation in all types of soil (Khudai)"),
    ("earthwork", "Earth backfilling with excavated soil"),
    ("earthwork", "Sand filling under floors"),
    ("earthwork", "Anti-termite treatment"),

    # RCC Work
    ("rcc", "PCC (1:4:8) below footings - 75mm thick"),
    ("rcc", "RCC Footings (M25)"),
    ("rcc", "RCC Pedestals/Columns up to plinth"),
    ("rcc", "RCC Plinth Beam"),
    ("rcc", "RCC Columns above plinth"),
    ("rcc", "RCC Beams"),
    ("rcc", "RCC Slabs"),
    ("rcc", "RCC Staircase"),
    ("rcc", "RCC Lintels"),
    ("rcc", "RCC Sunshade/Chajja"),
    ("rcc", "RCC Parapet"),
    ("rcc", "Reinforcement Steel (Saria) - Fe500/Fe500D"),
    ("rcc", "Shuttering/Centering for RCC work"),

    # Masonry
    ("masonry", "Brickwork 230mm thick (9\") in CM 1:6"),
    ("masonry", "Brickwork 115mm thick (4.5\") in CM 1:4"),
    ("masonry", "AAC Block masonry 200mm"),
    ("masonry", "AAC Block masonry 100mm"),

    # Finishes
    ("finishes", "Cement plaster 12mm (internal)"),
    ("finishes", "Cement plaster 20mm (external)"),
    ("finishes", "Ceiling plaster 6mm"),
    ("finishes", "IPS flooring (Industrial floor)"),
    ("finishes", "Vitrified tile flooring"),
    ("finishes", "Ceramic tile dado"),
    ("finishes", "Putty and painting (internal)"),
    ("finishes", "External texture/paint"),

    # Waterproofing
    ("waterproofing", "Waterproofing to terrace"),
    ("waterproofing", "Waterproofing to bathrooms"),
    ("waterproofing", "DPC (Damp Proof Course)"),

    # Doors & Windows
    ("doors_windows", "MS Door frames"),
    ("doors_windows", "Wooden door shutters"),
    ("doors_windows", "UPVC/Aluminium windows"),
    ("doors_windows", "MS Grills"),

    # MEP provisions
    ("plumbing", "Plumbing provisions/sleeves"),
    ("electrical", "Electrical conduit provisions"),

    # Site works
    ("siteworks", "Compound wall"),
    ("siteworks", "Gate"),
    ("siteworks", "Paving/hardscape"),
]


# =============================================================================
# PIPELINE CLASSES
# =============================================================================

@dataclass
class ExtractedFacts:
    """Container for all facts extracted from drawing."""
    # Text sources
    pdf_text: str = ""
    ocr_text: str = ""
    notes_text: str = ""

    # Tables (from Camelot/pdfplumber)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    column_schedules: List[Dict[str, Any]] = field(default_factory=list)
    footing_schedules: List[Dict[str, Any]] = field(default_factory=list)
    beam_schedules: List[Dict[str, Any]] = field(default_factory=list)
    bar_schedules: List[Dict[str, Any]] = field(default_factory=list)

    # Detected elements
    columns: List[Dict[str, Any]] = field(default_factory=list)
    footings: List[Dict[str, Any]] = field(default_factory=list)
    beams: List[Dict[str, Any]] = field(default_factory=list)
    slabs: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata extracted
    concrete_grade: Optional[str] = None
    steel_grade: Optional[str] = None
    scale: Optional[str] = None
    storey_count: Optional[int] = None
    storey_height_mm: Optional[int] = None

    # Evidence tracking
    evidence_by_source: Dict[str, List[Evidence]] = field(default_factory=dict)


class TakeoffPipeline:
    """
    Main pipeline for extracting BOQ and scope from drawings.

    Usage:
        pipeline = TakeoffPipeline()
        package = pipeline.run(pdf_path, options)
    """

    def __init__(self):
        self.facts = ExtractedFacts()
        self.drawing_meta: Optional[DrawingMeta] = None
        self.scope_items: List[ScopeItem] = []
        self.boq_items: List[BOQItem] = []
        self.conflicts: List[Conflict] = []
        self.coverage: List[CoverageRecord] = []

    def run(
        self,
        pdf_path: Path,
        floors: int = 1,
        storey_height_mm: int = 3000,
        high_recall: bool = True
    ) -> EstimatePackage:
        """
        Run the full takeoff pipeline.

        Args:
            pdf_path: Path to PDF file
            floors: Number of floors (user input)
            storey_height_mm: Storey height (user input)
            high_recall: Enable aggressive scope detection

        Returns:
            EstimatePackage with all extracted data
        """
        print(f"\n{'='*60}")
        print(f"[PIPELINE] Starting takeoff for: {pdf_path.name}")
        print(f"[PIPELINE] Floors: {floors}, Storey Height: {storey_height_mm}mm")
        print(f"{'='*60}\n")

        try:
            # Step 1: Ingest and classify PDF
            self._ingest_pdf(pdf_path)

            # Step 2: Extract facts from all sources
            self._extract_facts(pdf_path, floors, storey_height_mm)

            # Step 3: Generate BOQ items
            self._generate_boq_items(floors, storey_height_mm, high_recall)

            # Step 4: Generate scope checklist
            self._generate_scope_checklist(high_recall)

            # Step 5: Detect conflicts
            self._detect_conflicts()

            # Step 6: Compute coverage
            self._compute_coverage()

            # Build final package
            package = EstimatePackage(
                drawing=self.drawing_meta,
                scope=self.scope_items,
                boq=self.boq_items,
                coverage=self.coverage,
                conflicts=self.conflicts,
            )

            self._print_summary(package)
            return package

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            print(f"[PIPELINE] ERROR: {e}")

            # Return partial package on error
            if self.drawing_meta is None:
                self.drawing_meta = DrawingMeta(
                    file_name=pdf_path.name,
                    confidence_overall=0.1
                )

            return EstimatePackage(
                drawing=self.drawing_meta,
                scope=self.scope_items,
                boq=self.boq_items,
                coverage=self.coverage,
                conflicts=self.conflicts,
            )

    def _ingest_pdf(self, pdf_path: Path):
        """Step 1: Ingest PDF and detect sheet type."""
        print("[PIPELINE] Step 1: Ingesting PDF...")

        try:
            import fitz
            doc = fitz.open(str(pdf_path))

            # Get first page text
            page_text = ""
            for page_num in range(min(3, len(doc))):  # Check first 3 pages
                page_text += doc[page_num].get_text().lower()

            doc.close()

            # Detect keywords
            detected_keywords = []
            has_schedule = False
            has_plan = False
            has_notes = False

            for kw in SCHEDULE_KEYWORDS:
                if kw in page_text:
                    detected_keywords.append(kw)
                    has_schedule = True

            for kw in PLAN_KEYWORDS:
                if kw in page_text:
                    detected_keywords.append(kw)
                    has_plan = True

            for kw in NOTES_KEYWORDS:
                if kw in page_text:
                    detected_keywords.append(kw)
                    has_notes = True

            # Detect scale
            scale_match = re.search(r'scale[:\s]*1\s*[:]\s*(\d+)', page_text)
            scale = f"1:{scale_match.group(1)}" if scale_match else None

            # Detect concrete grade
            grade_match = re.search(r'm\s*(\d{2,3})', page_text)
            self.facts.concrete_grade = f"M{grade_match.group(1)}" if grade_match else "M25"

            # Detect steel grade
            if "fe500d" in page_text or "fe 500d" in page_text:
                self.facts.steel_grade = "Fe500D"
            elif "fe500" in page_text or "fe 500" in page_text:
                self.facts.steel_grade = "Fe500"
            else:
                self.facts.steel_grade = "Fe500"

            # Determine discipline
            if has_schedule or "reinforcement" in page_text or "rcc" in page_text:
                discipline = Discipline.STRUCTURAL
            elif "architectural" in page_text or "elevation" in page_text:
                discipline = Discipline.ARCHITECTURAL
            else:
                discipline = Discipline.STRUCTURAL  # Default for foundation plans

            # Create drawing meta
            self.drawing_meta = DrawingMeta(
                file_name=pdf_path.name,
                discipline=discipline,
                scale=scale,
                detected_keywords=detected_keywords[:20],
                has_schedule_tables=has_schedule,
                has_plan_view=has_plan,
                has_notes_section=has_notes,
                confidence_overall=0.7 if (has_schedule or has_plan) else 0.4,
            )

            print(f"[PIPELINE]   - Keywords found: {len(detected_keywords)}")
            print(f"[PIPELINE]   - Has schedule tables: {has_schedule}")
            print(f"[PIPELINE]   - Has plan view: {has_plan}")
            print(f"[PIPELINE]   - Concrete grade: {self.facts.concrete_grade}")
            print(f"[PIPELINE]   - Steel grade: {self.facts.steel_grade}")
            print(f"[PIPELINE]   - Scale: {scale or 'Not detected'}")

        except Exception as e:
            logger.error(f"PDF ingestion failed: {e}")
            print(f"[PIPELINE] WARNING: PDF ingestion failed: {e}")

            self.drawing_meta = DrawingMeta(
                file_name=pdf_path.name,
                confidence_overall=0.2
            )

    def _extract_facts(self, pdf_path: Path, floors: int, storey_height_mm: int):
        """Step 2: Extract facts from all sources."""
        print("[PIPELINE] Step 2: Extracting facts...")

        self.facts.storey_count = floors
        self.facts.storey_height_mm = storey_height_mm

        # Extract PDF text
        self._extract_pdf_text(pdf_path)

        # Extract tables
        self._extract_tables(pdf_path)

        # Extract from existing extractors
        self._extract_elements(pdf_path)

        print(f"[PIPELINE]   - Tables found: {len(self.facts.tables)}")
        print(f"[PIPELINE]   - Column schedules: {len(self.facts.column_schedules)}")
        print(f"[PIPELINE]   - Columns detected: {len(self.facts.columns)}")
        print(f"[PIPELINE]   - Footings detected: {len(self.facts.footings)}")

    def _extract_pdf_text(self, pdf_path: Path):
        """Extract raw text from PDF."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))

            all_text = []
            for page_num in range(len(doc)):
                page_text = doc[page_num].get_text()
                all_text.append(page_text)

                # Look for notes sections
                if any(kw in page_text.lower() for kw in NOTES_KEYWORDS):
                    self.facts.notes_text += page_text + "\n"

                    # Create evidence
                    ev = create_evidence(
                        page=page_num,
                        source="pdf_text",
                        snippet=page_text[:200]
                    )
                    self.facts.evidence_by_source.setdefault("pdf_text", []).append(ev)

            self.facts.pdf_text = "\n".join(all_text)
            doc.close()

        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")

    def _extract_tables(self, pdf_path: Path):
        """Extract tables using Camelot/pdfplumber."""
        try:
            # Try existing extractor
            from src.extractors import extract_column_schedule

            result = extract_column_schedule(str(pdf_path), page_number=0)

            if result and result.entries:
                for entry in result.entries:
                    self.facts.column_schedules.append({
                        "column_marks": entry.column_marks,
                        "section_size": entry.section_size,
                        "longitudinal": entry.longitudinal_raw,
                        "ties": entry.ties_raw,
                        "confidence": entry.confidence,
                    })

                    # Create evidence
                    ev = create_evidence(
                        page=0,
                        source="camelot",
                        snippet=f"Column schedule: {', '.join(entry.column_marks[:3])}"
                    )
                    self.facts.evidence_by_source.setdefault("camelot", []).append(ev)

            if result and result.raw_dataframe is not None:
                self.facts.tables.append({
                    "type": "column_schedule",
                    "dataframe": result.raw_dataframe,
                    "method": result.extraction_method,
                })

        except ImportError:
            logger.warning("Column schedule extractor not available")
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")

        # Try generic table extraction
        try:
            from src.extractors import extract_tables

            tables = extract_tables(pdf_path, page_numbers=[0])
            for table in tables:
                self.facts.tables.append({
                    "type": table.table_type.value,
                    "dataframe": table.dataframe,
                    "method": table.extraction_method,
                    "confidence": table.confidence,
                })

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Generic table extraction failed: {e}")

    def _extract_elements(self, pdf_path: Path):
        """Extract structural elements using existing extractors."""
        try:
            import fitz
            import numpy as np
            import cv2

            # Open PDF and render
            doc = fitz.open(str(pdf_path))
            page = doc[0]

            # Render to image
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Get text blocks
            text_blocks = []
            blocks = page.get_text('dict')
            for block in blocks.get('blocks', []):
                if block.get('type') == 0:
                    for line in block.get('lines', []):
                        for span in line.get('spans', []):
                            t = span.get('text', '').strip()
                            if t:
                                text_blocks.append({
                                    'text': t,
                                    'bbox': span.get('bbox', (0, 0, 0, 0)),
                                    'size': span.get('size', 10)
                                })

            pdf_text = page.get_text()
            doc.close()

            # Use foundation extractor
            from src.structural.foundation_extractor import extract_foundation_plan
            fp_data = extract_foundation_plan(img, text_blocks, pdf_text)

            # Extract columns
            if hasattr(fp_data, 'columns') and fp_data.columns:
                for col in fp_data.columns:
                    self.facts.columns.append({
                        "mark": col.mark,
                        "size": col.size,
                        "count": col.count,
                        "concrete_volume": getattr(col, 'concrete_volume_m3', None),
                    })

                    ev = create_evidence(
                        page=0,
                        source="heuristic",
                        snippet=f"Column {col.mark}: {col.size}"
                    )
                    self.facts.evidence_by_source.setdefault("heuristic", []).append(ev)

            # Extract footings
            if hasattr(fp_data, 'footings') and fp_data.footings:
                for ftg in fp_data.footings:
                    self.facts.footings.append({
                        "mark": ftg.mark,
                        "size": ftg.size,
                        "count": ftg.count,
                        "concrete_volume": getattr(ftg, 'concrete_volume_m3', None),
                    })

        except Exception as e:
            logger.error(f"Element extraction failed: {e}")
            print(f"[PIPELINE] WARNING: Element extraction failed: {e}")

    def _generate_boq_items(self, floors: int, storey_height_mm: int, high_recall: bool):
        """Step 3: Generate BOQ items with rule-based confidence scoring."""
        print("[PIPELINE] Step 3: Generating BOQ items...")

        # RCC Columns
        if self.facts.columns:
            total_col_count = sum(c.get("count", 0) for c in self.facts.columns)
            total_col_volume = sum(c.get("concrete_volume", 0) or 0 for c in self.facts.columns)

            # Scale by floors
            total_col_volume *= floors

            evidence = self.facts.evidence_by_source.get("heuristic", [])
            evidence_sources = {e.source.value if hasattr(e.source, 'value') else str(e.source) for e in evidence}

            # Compute confidence with rule-based model
            required_fields = {
                "qty": total_col_volume > 0,
                "dimensions": len(self.facts.columns) > 0,
                "count": total_col_count > 0,
            }

            conf_result = compute_confidence(
                item_type="column",
                status="detected",
                evidence_sources=evidence_sources,
                evidence_count=len(evidence),
                required_fields_present=required_fields,
                qty_status="computed" if total_col_volume > 0 else "partial",
            )

            self.boq_items.append(BOQItem(
                system="structural",
                subsystem="rcc",
                item_name=f"RCC Columns ({self.facts.concrete_grade}) - all floors",
                unit="cum",
                qty=round(total_col_volume, 2) if total_col_volume > 0 else None,
                qty_status=QtyStatus.COMPUTED if total_col_volume > 0 else QtyStatus.PARTIAL,
                measurement_rule=f"Volume from {total_col_count} columns × {floors} floors",
                dependencies=[] if total_col_volume > 0 else ["need column dimensions"],
                evidence=evidence[:3],
                confidence=conf_result.score,
                confidence_reason=conf_result.reason,
                source="explicit",
            ))

            # Shuttering for columns
            # Rough estimate: perimeter × height
            if total_col_volume > 0:
                # Assume average column 300x450
                avg_perimeter = 1.5  # meters
                shuttering_area = avg_perimeter * (storey_height_mm / 1000) * total_col_count * floors

                # Compute confidence for inferred item
                shuttering_conf = compute_confidence(
                    item_type="column",
                    status="inferred",
                    evidence_sources=evidence_sources,
                    evidence_count=len(evidence),
                    required_fields_present={"qty": True, "dimensions": False},
                    qty_status="partial",
                )

                self.boq_items.append(BOQItem(
                    system="structural",
                    subsystem="rcc",
                    item_name="Shuttering/Centering for RCC Columns",
                    unit="sqm",
                    qty=round(shuttering_area, 2),
                    qty_status=QtyStatus.PARTIAL,
                    measurement_rule="Estimated from column perimeter × height",
                    dependencies=["verify column sizes"],
                    evidence=evidence[:2],
                    confidence=shuttering_conf.score,
                    confidence_reason=shuttering_conf.reason,
                    source="inferred",
                ))

        # RCC Footings
        if self.facts.footings:
            total_ftg_count = sum(f.get("count", 0) for f in self.facts.footings)
            total_ftg_volume = sum(f.get("concrete_volume", 0) or 0 for f in self.facts.footings)

            evidence = self.facts.evidence_by_source.get("heuristic", [])
            evidence_sources = {e.source.value if hasattr(e.source, 'value') else str(e.source) for e in evidence}

            required_fields = {
                "qty": total_ftg_volume > 0,
                "dimensions": len(self.facts.footings) > 0,
                "count": total_ftg_count > 0,
            }

            conf_result = compute_confidence(
                item_type="footing",
                status="detected",
                evidence_sources=evidence_sources,
                evidence_count=len(evidence),
                required_fields_present=required_fields,
                qty_status="computed" if total_ftg_volume > 0 else "partial",
            )

            self.boq_items.append(BOQItem(
                system="structural",
                subsystem="rcc",
                item_name=f"RCC Isolated Footings ({self.facts.concrete_grade})",
                unit="cum",
                qty=round(total_ftg_volume, 2) if total_ftg_volume > 0 else None,
                qty_status=QtyStatus.COMPUTED if total_ftg_volume > 0 else QtyStatus.PARTIAL,
                measurement_rule=f"Volume from {total_ftg_count} footings",
                dependencies=[] if total_ftg_volume > 0 else ["need footing dimensions"],
                evidence=evidence[:3],
                confidence=conf_result.score,
                confidence_reason=conf_result.reason,
                source="explicit",
            ))

            # PCC below footings
            if total_ftg_volume > 0:
                # Estimate PCC as 10% of footing volume (75mm thick layer)
                pcc_volume = total_ftg_volume * 0.15

                pcc_conf = compute_confidence(
                    item_type="footing",
                    status="inferred",
                    evidence_sources=evidence_sources,
                    evidence_count=len(evidence),
                    required_fields_present={"qty": True, "derived_from_parent": True},
                    qty_status="partial",
                )

                # Create evidence for why this was inferred
                pcc_evidence = [create_evidence(
                    page=0,
                    source="inferred",
                    snippet=f"PCC inferred from {total_ftg_count} footings (standard 75mm layer)"
                )]

                self.boq_items.append(BOQItem(
                    system="structural",
                    subsystem="rcc",
                    item_name="PCC (1:4:8) 75mm thick below footings",
                    unit="cum",
                    qty=round(pcc_volume, 2),
                    qty_status=QtyStatus.PARTIAL,
                    measurement_rule="Estimated as 15% of footing volume",
                    dependencies=["verify footing plan dimensions"],
                    evidence=pcc_evidence,
                    confidence=pcc_conf.score,
                    confidence_reason=pcc_conf.reason,
                    source="inferred",
                ))

        # Reinforcement Steel
        if self.facts.column_schedules or self.facts.columns:
            # If we have schedule data, we can be more precise
            if self.facts.column_schedules:
                evidence = self.facts.evidence_by_source.get("camelot", [])
                evidence_sources = {e.source.value if hasattr(e.source, 'value') else str(e.source) for e in evidence}

                steel_conf = compute_confidence(
                    item_type="reinforcement",
                    status="detected",
                    evidence_sources=evidence_sources,
                    evidence_count=len(evidence),
                    required_fields_present={"bar_schedule": True, "bar_lengths": False},
                    qty_status="partial",
                    has_schedule_data=True,
                )

                self.boq_items.append(BOQItem(
                    system="structural",
                    subsystem="rcc",
                    item_name=f"Reinforcement Steel ({self.facts.steel_grade}) for Columns",
                    unit="kg",
                    qty=None,  # Can't compute without bar lengths
                    qty_status=QtyStatus.PARTIAL,
                    measurement_rule="Schedule detected but bar lengths needed for weight",
                    dependencies=["need bar cutting lengths", "need lap lengths"],
                    evidence=evidence[:3],
                    confidence=steel_conf.score,
                    confidence_reason=steel_conf.reason,
                    source="explicit",
                ))
            else:
                # Rough estimate based on column volume
                total_vol = sum(c.get("concrete_volume", 0) or 0 for c in self.facts.columns)
                if total_vol > 0:
                    # Typical steel: 100-150 kg/cum for columns
                    steel_kg = total_vol * 120 * floors

                    # Create evidence explaining the inference
                    steel_evidence = [create_evidence(
                        page=0,
                        source="inferred",
                        snippet=f"Steel estimated @ 120 kg/cum from {total_vol:.2f} cum column volume"
                    )]

                    steel_conf = compute_confidence(
                        item_type="reinforcement",
                        status="inferred",
                        evidence_sources={"inferred"},
                        evidence_count=1,
                        required_fields_present={"qty": True, "bar_schedule": False},
                        qty_status="partial",
                    )

                    self.boq_items.append(BOQItem(
                        system="structural",
                        subsystem="rcc",
                        item_name=f"Reinforcement Steel ({self.facts.steel_grade}) for Columns",
                        unit="kg",
                        qty=round(steel_kg, 0),
                        qty_status=QtyStatus.PARTIAL,
                        measurement_rule="Estimated @ 120 kg/cum (verify with BBS)",
                        dependencies=["need bar bending schedule"],
                        evidence=steel_evidence,
                        confidence=steel_conf.score,
                        confidence_reason=steel_conf.reason,
                        source="inferred",
                    ))

        # High recall mode: add inferred items
        if high_recall:
            self._add_inferred_boq_items(floors, storey_height_mm)

        print(f"[PIPELINE]   - BOQ items generated: {len(self.boq_items)}")

    def _add_inferred_boq_items(self, floors: int, storey_height_mm: int):
        """Add inferred BOQ items in high recall mode with proper evidence and confidence."""

        # Always expected for RCC buildings - (system, subsystem, name, unit, deps, reason)
        inferred_items = [
            ("structural", "rcc", "RCC Plinth Beam", "rmt", ["need plinth beam layout"], "Standard for RCC framed structures"),
            ("structural", "rcc", "RCC Beams (floor level)", "cum", ["need beam schedule"], "Required for floor support in RCC buildings"),
            ("structural", "rcc", "RCC Slabs", "cum", ["need slab layout", "need slab thickness"], "Standard floor construction in RCC buildings"),
            ("structural", "rcc", "Shuttering for Beams", "sqm", ["need beam dimensions"], "Required for RCC beam casting"),
            ("structural", "rcc", "Shuttering for Slabs", "sqm", ["need slab area"], "Required for RCC slab casting"),
            ("structural", "rcc", "Reinforcement Steel for Beams", "kg", ["need beam BBS"], "Steel reinforcement for beam strength"),
            ("structural", "rcc", "Reinforcement Steel for Slabs", "kg", ["need slab BBS"], "Steel reinforcement for slab strength"),
            ("structural", "rcc", "Reinforcement Steel for Footings", "kg", ["need footing BBS"], "Steel reinforcement for footing strength"),

            # Earthwork
            ("structural", "earthwork", "Excavation for footings (all types of soil)", "cum", ["need footing depths"], "Excavation required for foundation"),
            ("structural", "earthwork", "Backfilling with excavated soil", "cum", ["computed from excavation"], "Site restoration after foundation work"),
            ("structural", "earthwork", "Sand filling 150mm under floors", "cum", ["need floor area"], "Standard bed for floor construction"),

            # Masonry (if architectural scope)
            ("architectural", "masonry", "Brickwork 230mm in CM 1:6 (external walls)", "cum", ["need wall layout"], "External walls for building enclosure"),
            ("architectural", "masonry", "Brickwork 115mm in CM 1:4 (internal walls)", "cum", ["need wall layout"], "Internal partition walls"),

            # Finishes
            ("architectural", "finishes", "Cement plaster 12mm internal", "sqm", ["need wall areas"], "Standard internal wall finish"),
            ("architectural", "finishes", "Cement plaster 20mm external", "sqm", ["need external wall areas"], "Standard external wall finish"),

            # Waterproofing
            ("architectural", "waterproofing", "DPC at plinth level", "sqm", ["need plinth perimeter"], "Damp proof course - standard requirement"),
        ]

        for system, subsystem, name, unit, deps, reason in inferred_items:
            # Check if already added
            if any(b.item_name == name for b in self.boq_items):
                continue

            # Compute confidence with rule-based model for inferred items
            conf_result = compute_confidence(
                item_type=subsystem,
                status="inferred",
                evidence_sources=set(),  # No direct evidence
                evidence_count=0,
                required_fields_present={},
                qty_status="unknown",
            )

            # Create evidence explaining why this was inferred
            inferred_evidence = [create_evidence(
                page=0,
                source="inferred",
                snippet=f"Inferred: {reason}"
            )]

            self.boq_items.append(BOQItem(
                system=system,
                subsystem=subsystem,
                item_name=name,
                unit=unit,
                qty=None,
                qty_status=QtyStatus.UNKNOWN,
                measurement_rule="Inferred - typical for RCC buildings",
                dependencies=deps,
                evidence=inferred_evidence,
                confidence=conf_result.score,
                confidence_reason=conf_result.reason,
                source="inferred",
                rule_fired="high_recall_rcc_building",
            ))

    def _generate_scope_checklist(self, high_recall: bool):
        """Step 4: Generate scope checklist with rule-based confidence."""
        print("[PIPELINE] Step 4: Generating scope checklist...")

        # Track what we've detected
        detected_trades = set()
        for boq in self.boq_items:
            if boq.qty_status in [QtyStatus.COMPUTED, QtyStatus.PARTIAL]:
                detected_trades.add(boq.item_name.lower())

        # Go through standard scope and mark status
        for category, trade in STANDARD_RCC_SCOPE:
            trade_lower = trade.lower()

            # Check if detected
            is_detected = any(
                trade_lower in dt or dt in trade_lower
                for dt in detected_trades
            )

            # Check in text
            in_text = (
                trade_lower in self.facts.pdf_text.lower() or
                trade_lower in self.facts.notes_text.lower()
            )

            if is_detected:
                status = ScopeStatus.DETECTED
                reason = "Found in drawing/schedule"
                evidence = self.facts.evidence_by_source.get("heuristic", [])[:2]
                evidence_sources = {e.source.value if hasattr(e.source, 'value') else str(e.source) for e in evidence}

                conf_result = compute_confidence(
                    item_type=category,
                    status="detected",
                    evidence_sources=evidence_sources,
                    evidence_count=len(evidence),
                )
                confidence = conf_result.score
                confidence_reason = conf_result.reason

            elif in_text:
                status = ScopeStatus.DETECTED
                reason = "Mentioned in notes/text"
                evidence = self.facts.evidence_by_source.get("pdf_text", [])[:2]
                evidence_sources = {"pdf_text"}

                conf_result = compute_confidence(
                    item_type=category,
                    status="detected",
                    evidence_sources=evidence_sources,
                    evidence_count=len(evidence),
                )
                confidence = conf_result.score
                confidence_reason = conf_result.reason

            elif high_recall and category in ["rcc", "earthwork"]:
                status = ScopeStatus.INFERRED
                reason = "Standard for RCC buildings"
                evidence = [create_evidence(
                    page=0,
                    source="inferred",
                    snippet=f"Inferred: {trade} is standard for RCC construction"
                )]

                conf_result = compute_confidence(
                    item_type=category,
                    status="inferred",
                    evidence_sources=set(),
                    evidence_count=0,
                )
                confidence = conf_result.score
                confidence_reason = conf_result.reason

            else:
                status = ScopeStatus.MISSING
                reason = "Not found in drawing"
                evidence = []

                conf_result = compute_confidence(
                    item_type=category,
                    status="missing",
                    evidence_sources=set(),
                    evidence_count=0,
                )
                confidence = conf_result.score
                confidence_reason = conf_result.reason

            self.scope_items.append(ScopeItem(
                category=ScopeCategory(category),
                trade=trade,
                status=status,
                reason=reason,
                confidence=confidence,
                confidence_reason=confidence_reason,
                evidence=evidence,
            ))

        # Add conflicts for missing critical items
        missing_critical = [
            s for s in self.scope_items
            if s.status == ScopeStatus.MISSING and s.category in [ScopeCategory.RCC, ScopeCategory.EARTHWORK]
        ]

        if missing_critical and not self.drawing_meta.has_schedule_tables:
            self.conflicts.append(create_conflict(
                conflict_type="missing_labels",
                description=f"{len(missing_critical)} critical RCC items not found - may need schedule sheet",
                severity="med",
                suggested_resolution="Upload reinforcement schedule sheet for complete extraction",
            ))

        print(f"[PIPELINE]   - Scope items: {len(self.scope_items)}")
        print(f"[PIPELINE]   - Detected: {sum(1 for s in self.scope_items if s.status == ScopeStatus.DETECTED)}")
        print(f"[PIPELINE]   - Inferred: {sum(1 for s in self.scope_items if s.status == ScopeStatus.INFERRED)}")
        print(f"[PIPELINE]   - Missing: {sum(1 for s in self.scope_items if s.status == ScopeStatus.MISSING)}")

    def _detect_conflicts(self):
        """Step 5: Detect conflicts using the analysis module."""
        print("[PIPELINE] Step 5: Detecting conflicts...")

        # Prepare data for conflict detection
        schedule_column_marks = set()
        for sched in self.facts.column_schedules:
            schedule_column_marks.update(sched.get("column_marks", []))

        plan_column_marks = {c.get("mark") for c in self.facts.columns if c.get("mark")}

        schedule_footing_marks = set()
        for sched in self.facts.footing_schedules:
            schedule_footing_marks.update(sched.get("footing_marks", []))

        plan_footing_marks = {f.get("mark") for f in self.facts.footings if f.get("mark")}

        # Use the new detect_conflicts function from analysis module
        detected_conflicts = detect_conflicts(
            pdf_text=self.facts.pdf_text,
            notes_text=self.facts.notes_text,
            scale=self.drawing_meta.scale,
            column_schedules=self.facts.column_schedules,
            columns_detected=self.facts.columns,
            footing_schedules=self.facts.footing_schedules,
            footings_detected=self.facts.footings,
            has_schedule_tables=self.drawing_meta.has_schedule_tables,
            has_plan_view=self.drawing_meta.has_plan_view,
        )

        # Add detected conflicts
        self.conflicts.extend(detected_conflicts)

        print(f"[PIPELINE]   - Conflicts detected: {len(self.conflicts)}")

    def _compute_coverage(self):
        """Step 6: Compute coverage using the coverage module."""
        print("[PIPELINE] Step 6: Computing coverage...")

        # Use the new build_coverage_records function from coverage module
        self.coverage = build_coverage_records(self.boq_items)

        avg_coverage = sum(c.coverage_score for c in self.coverage) / len(self.coverage) if self.coverage else 0
        print(f"[PIPELINE]   - Average coverage score: {avg_coverage:.2f}")

    def _print_summary(self, package: EstimatePackage):
        """Print pipeline summary."""
        stats = package.stats

        print(f"\n{'='*60}")
        print(f"[PIPELINE] TAKEOFF COMPLETE")
        print(f"{'='*60}")
        print(f"  Drawing: {package.drawing.file_name}")
        print(f"  Confidence: {package.drawing.confidence_overall:.0%}")
        print(f"")
        print(f"  Scope Items: {stats['total_scope_items']}")
        for status, count in stats['scope_by_status'].items():
            print(f"    - {status}: {count}")
        print(f"")
        print(f"  BOQ Items: {stats['total_boq_items']}")
        for status, count in stats['boq_by_qty_status'].items():
            print(f"    - {status}: {count}")
        print(f"")
        print(f"  Conflicts: {stats['total_conflicts']}")
        if stats['high_severity_conflicts'] > 0:
            print(f"    - HIGH SEVERITY: {stats['high_severity_conflicts']}")
        print(f"{'='*60}\n")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_takeoff_pipeline(
    pdf_path: Path,
    floors: int = 1,
    storey_height_mm: int = 3000,
    high_recall: bool = True
) -> EstimatePackage:
    """
    Run the takeoff pipeline on a PDF.

    Args:
        pdf_path: Path to PDF file
        floors: Number of floors
        storey_height_mm: Storey height in mm
        high_recall: Enable aggressive scope detection

    Returns:
        EstimatePackage with all extracted data
    """
    pipeline = TakeoffPipeline()
    return pipeline.run(pdf_path, floors, storey_height_mm, high_recall)
