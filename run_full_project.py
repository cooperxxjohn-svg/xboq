#!/usr/bin/env python3
"""
XBOQ Full Project Runner - Canonical Single Command

Complete India-first preconstruction BOQ & scope pipeline.

Usage:
    python run_full_project.py --project_id <id> --input_dir <path> --mode full
    python run_full_project.py --project_id villa --input_dir ~/Documents/villa_drawings/
    python run_full_project.py --project_id villa --profile typical --resume

Pipeline Phases (22+):
    EXTRACTION:
        1. Index drawings
        2. Route pages (floor plans, structural, MEP, etc.)
        3. Extract rooms, walls, openings
        4. Join multi-page project

    TAKEOFF:
        5. Takeoff generation (rooms/walls/openings/finishes/masonry)
        6. Measurement rules + deductions + formwork derivation

    ANALYSIS:
        7. Scope completeness
        8. Triangulation + overrides + paranoia
        9. BOM + procurement summary
        10. Revision intelligence + detail understanding
        11. Doubt engine
        12. RFI engine

    OWNER ALIGNMENT:
        13. Owner docs parsing
        14. Owner BOQ alignment

    PRICING:
        15. Pricing + scenarios
        16. Quote leveling
        17. Prelims generator

    BID SUBMISSION:
        18. Bid gate
        19. Bid docs (clarifications, exclusions, assumptions)
        20. Package exports + RFQ sheets
        21. Bid book export

    VERIFICATION:
        22. Output verification

Output: out/<project_id>/
"""

import argparse
import json
import logging
import os
import random
import shutil
import string
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runner")


# =============================================================================
# OUTPUT CONTRACT - Required files for mode=full
# =============================================================================

REQUIRED_OUTPUTS = {
    "summary.md": "Run summary",
    "bid_gate_report.md": "Bid gate assessment",
    "overlays/quicklook.png": "Visual overlays",
    "rfi/rfi_log.md": "RFI list",
    "scope/scope_register.csv": "Scope register",
    "boq/boq_quantities.csv": "BOQ quantities",
    "boq/provisional_items.csv": "Provisional items",
    "boq/material_estimate.csv": "Material estimate",
    "pricing/estimate_priced.csv": "Priced estimate",
    "prelims/prelims_boq.csv": "Prelims BOQ",
    "bid_book/clarifications_letter.md": "Clarifications letter",
    "bid_book/exclusions.md": "Exclusions list",
    "bid_book/assumptions.md": "Assumptions list",
    "bid_report.pdf": "Professional PDF bid report",
}

REQUIRED_PACKAGES = [
    "rcc_structural",
    "masonry",
    "waterproofing",
    "flooring",
    "doors_windows",
    "wall_finishes",
]

# Evidence-first mode: additional required outputs
EVIDENCE_REQUIRED_OUTPUTS = {
    "boq/boq_measured.csv": "Measured BOQ (geometry-backed)",
    "boq/boq_inferred.csv": "Inferred BOQ (TBD/allowances)",
    "measurement_gate_report.md": "Measurement gate assessment",
    "proof/proof_pack.md": "Measurement proof documentation",
}


@dataclass
class PhaseResult:
    """Result of a pipeline phase."""
    phase: str
    phase_name: str
    success: bool
    message: str = ""
    data: Dict = field(default_factory=dict)
    error: str = None
    stack_trace: str = None
    duration_sec: float = 0.0
    skipped: bool = False
    skip_reason: str = None


@dataclass
class OutputVerification:
    """Output verification result."""
    required_file: str
    exists: bool
    description: str
    size_bytes: int = 0
    reason: str = None


class FullProjectRunner:
    """
    Full 22+ phase project processing pipeline.

    Mode:
        - full (default): Run all phases, enforce output contract
        - quick: Extract rooms + RFIs only (development/testing)
    """

    def __init__(
        self,
        project_id: str,
        input_dir: Path = None,
        profile: str = "typical",
        mode: str = "full",
        resume: bool = False,
        rules_version: str = None,
        # Evidence-first mode options
        allow_inferred_pricing: bool = False,
        manual_scale: float = None,
        # MEP options
        enable_mep: bool = False,
        # Estimator assumption overrides
        assume_wall_height: float = 3.0,
        assume_door_height: float = 2.1,
        assume_plaster_both_sides: bool = True,
        assume_floor_finish_all_rooms: bool = True,
        # Estimator workflow options
        apply_overrides: bool = False,
        # Demo mode
        demo_mode: bool = False,
        fail_fast: bool = False,
    ):
        self.project_id = project_id
        self.input_dir = Path(input_dir) if input_dir else None
        self.profile = profile
        self.mode = mode
        self.resume = resume
        self.rules_version = rules_version

        # Evidence-first mode settings
        self.allow_inferred_pricing = allow_inferred_pricing
        self.manual_scale = manual_scale

        # MEP mode
        self.enable_mep = enable_mep

        # Paths
        self.project_dir = PROJECT_ROOT / "data" / "projects" / project_id
        self.output_dir = PROJECT_ROOT / "out" / project_id
        self.drawings_dir = self.project_dir / "drawings"
        self.owner_docs_dir = self.project_dir / "owner_docs"
        self.quotes_dir = self.project_dir / "quotes"

        # Results tracking
        self.results: List[PhaseResult] = []
        self.summary: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        # Accumulated data across phases
        self.rooms: List[Dict] = []
        self.openings: List[Dict] = []
        self.boq_items: List[Dict] = []
        self.scope_register: Dict = {}
        self.rfis: List[Dict] = []

        # Evidence-first state
        self.measurement_gate_result = None
        self.can_produce_pricing = False
        self.provenance_tracker = None
        self.source_files: List[Path] = []

        # MEP data
        self.mep_devices: List[Dict] = []
        self.mep_connections: List[Dict] = []
        self.mep_takeoff: Dict[str, Any] = {}

        # Multipage routing/extraction state
        self.routing_result: Optional[Dict] = None
        self.detected_scale: Optional[float] = None
        self.scale_method: Optional[str] = None
        self.scale_confidence: float = 0.0
        self.total_pages_indexed: int = 0

        # Estimator assumptions
        self.assume_wall_height = assume_wall_height
        self.assume_door_height = assume_door_height
        self.assume_plaster_both_sides = assume_plaster_both_sides
        self.assume_floor_finish_all_rooms = assume_floor_finish_all_rooms

        # Estimator workflow
        self.apply_overrides = apply_overrides

        # Demo mode (for YC video - runs only stable phases)
        self.demo_mode = demo_mode
        self.fail_fast = fail_fast

    def run(self) -> int:
        """
        Run the full pipeline.

        Returns:
            Exit code: 0 if success, 1 if required outputs missing
        """
        start_time = datetime.now()

        self._print_header(start_time)

        # Setup project structure
        if not self._setup_project():
            return 1

        # Create output directories
        self._setup_output_dirs()

        # Load project metadata
        self._load_project_metadata()

        # ===== INPUT GATE: Verify we have readable pages before proceeding =====
        input_validation = self._validate_input_drawings()
        if not input_validation["valid"]:
            print("\n" + "=" * 70)
            print("❌ INPUT GATE FAILED - ABORTING PIPELINE")
            print("=" * 70)
            print(f"Reason: {input_validation['reason']}")
            print(f"Files checked: {input_validation.get('files_checked', 0)}")
            print(f"Pages found: {input_validation.get('pages_found', 0)}")
            print("\nNo BOQ or report will be generated from invalid input.")
            print("=" * 70)
            return 1

        print(f"✅ Input Gate: {input_validation['pages_found']} readable pages from {input_validation['files_checked']} files")

        # Define phases based on mode
        if self.demo_mode:
            phases = self._get_demo_phases()
            print("\n" + "=" * 60)
            print("🎬 DEMO MODE: Running stable phases only for YC video")
            print("=" * 60)
        elif self.mode == "quick":
            phases = self._get_quick_phases()
        else:
            phases = self._get_full_phases()

        # Run all phases with fail-fast checks
        for phase_id, phase_name, phase_func in phases:
            self._run_phase(phase_id, phase_name, phase_func)

            # Check for phase failure in fail_fast mode
            last_result = self.results[-1] if self.results else None
            if self.fail_fast and last_result and not last_result.success and not last_result.skipped:
                print(f"\n❌ FAIL-FAST: Stopping due to phase failure: {phase_name}")
                self._generate_summary((datetime.now() - start_time).total_seconds())
                self._save_run_metadata(start_time, (datetime.now() - start_time).total_seconds(), [])
                return 1

            # ===== FAIL-FAST GATE: After extract phase =====
            if phase_id == "03_extract":
                fail_fast_result = self._check_extraction_gate(start_time)
                if fail_fast_result is not None:
                    return fail_fast_result

        # Calculate duration
        total_duration = (datetime.now() - start_time).total_seconds()

        # Generate summary
        self._generate_summary(total_duration)

        # Save metadata and summary.md BEFORE verification
        # (so verification can check summary.md exists)
        output_verification = []
        self._save_run_metadata(start_time, total_duration, output_verification)

        # Verify outputs (mode=full only) - AFTER summary.md written
        if self.mode == "full":
            output_verification = self._verify_outputs()
            # Update metadata with verification results
            self._update_run_metadata_verification(output_verification)

            # Add verification result to results list for display
            missing = [v for v in output_verification if not v.exists]
            present = [v for v in output_verification if v.exists]

            gate_status = "N/A"
            if self.measurement_gate_result:
                gate_status = self.measurement_gate_result.status.value

            verify_result = PhaseResult(
                phase="24_verify",
                phase_name="Output Verification",
                success=len(missing) == 0,
                message=f"{len(present)}/{len(output_verification)} required outputs present | Measurement gate: {gate_status}",
                data={
                    "present": len(present),
                    "missing": len(missing),
                    "missing_files": [v.required_file for v in missing],
                },
                error=f"Missing: {', '.join(v.required_file for v in missing)}" if missing else None,
            )
            self.results.append(verify_result)

            # Re-generate summary with verify result included
            self._generate_summary(total_duration)
            self._save_run_metadata(start_time, total_duration, output_verification)

        # Print summary
        self._print_summary(output_verification)

        # Determine exit code
        if self.mode == "full":
            missing_outputs = [v for v in output_verification if not v.exists]
            if missing_outputs:
                print(f"\n❌ FAILED: {len(missing_outputs)} required outputs missing")
                return 1

        return 0

    def _print_header(self, start_time: datetime) -> None:
        """Print run header."""
        print("=" * 70)
        print("XBOQ FULL PROJECT RUNNER - EVIDENCE-FIRST MODE")
        print("=" * 70)
        print(f"Project ID: {self.project_id}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Profile: {self.profile}")
        print(f"Allow Inferred Pricing: {self.allow_inferred_pricing}")
        if self.manual_scale:
            print(f"Manual Scale: 1:{self.manual_scale}")
        print(f"Input Dir: {self.input_dir or 'N/A'}")
        print(f"Resume: {self.resume}")
        print(f"Rules Version: {self.rules_version or 'current'}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()

    def _setup_project(self) -> bool:
        """Setup project directory structure."""
        # Create project directory
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.drawings_dir.mkdir(parents=True, exist_ok=True)
        self.owner_docs_dir.mkdir(parents=True, exist_ok=True)
        self.quotes_dir.mkdir(parents=True, exist_ok=True)

        # Copy/symlink input drawings if provided
        if self.input_dir:
            if not self.input_dir.exists():
                print(f"❌ Input directory not found: {self.input_dir}")
                return False

            # Find all drawing files (recursive)
            patterns = ["**/*.pdf", "**/*.PDF", "**/*.png", "**/*.PNG",
                       "**/*.jpg", "**/*.JPG", "**/*.jpeg", "**/*.JPEG",
                       "**/*.tif", "**/*.tiff", "**/*.TIF", "**/*.TIFF"]

            files_copied = 0
            for pattern in patterns:
                for src_file in self.input_dir.glob(pattern):
                    if src_file.is_file():
                        dest_file = self.drawings_dir / src_file.name
                        if not dest_file.exists():
                            shutil.copy2(src_file, dest_file)
                            files_copied += 1

            if files_copied > 0:
                print(f"✓ Copied {files_copied} drawing files to {self.drawings_dir}")
            else:
                print(f"⚠️ No new drawing files to copy from {self.input_dir}")

        # Validate drawings exist
        drawings = list(self.drawings_dir.glob("*.[pP][dD][fF]")) + \
                   list(self.drawings_dir.glob("*.[pP][nN][gG]")) + \
                   list(self.drawings_dir.glob("*.[jJ][pP][gG]"))

        if not drawings:
            print(f"⚠️ No drawings found in {self.drawings_dir}")
            print("   Place PDF or image files in the drawings folder")
            # Don't fail - might be resuming with cached data

        return True

    def _validate_input_drawings(self) -> dict:
        """
        INPUT GATE: Validate that drawings exist and are readable.
        Returns dict with: valid (bool), reason (str), files_checked (int), pages_found (int)
        """
        result = {
            "valid": False,
            "reason": "",
            "files_checked": 0,
            "pages_found": 0,
            "file_details": [],
        }

        # Find all drawing files
        patterns = ["*.[pP][dD][fF]", "*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]"]
        files = []
        for pattern in patterns:
            files.extend(self.drawings_dir.glob(pattern))

        if not files:
            result["reason"] = f"No drawing files found in {self.drawings_dir}"
            return result

        result["files_checked"] = len(files)

        # Validate each file
        try:
            import fitz
        except ImportError:
            result["reason"] = "PyMuPDF (fitz) not installed - cannot validate PDFs"
            return result

        for file_path in files:
            file_info = {"path": str(file_path), "name": file_path.name, "pages": 0, "readable": False, "error": None}

            try:
                if file_path.suffix.lower() == ".pdf":
                    doc = fitz.open(str(file_path))
                    page_count = len(doc)
                    is_encrypted = doc.is_encrypted
                    doc.close()

                    if is_encrypted:
                        file_info["error"] = "PDF is encrypted"
                    elif page_count == 0:
                        file_info["error"] = "PDF has 0 pages"
                    else:
                        file_info["pages"] = page_count
                        file_info["readable"] = True
                        result["pages_found"] += page_count
                else:
                    # Image file - count as 1 page
                    file_info["pages"] = 1
                    file_info["readable"] = True
                    result["pages_found"] += 1

            except Exception as e:
                file_info["error"] = str(e)

            result["file_details"].append(file_info)

        if result["pages_found"] == 0:
            result["reason"] = "All files failed validation - 0 readable pages"
            # List errors
            errors = [f"{f['name']}: {f['error']}" for f in result["file_details"] if f.get("error")]
            if errors:
                result["reason"] += " | Errors: " + "; ".join(errors[:3])
            return result

        result["valid"] = True
        result["reason"] = "Input validation passed"
        return result

    def _check_extraction_gate(self, start_time: datetime) -> Optional[int]:
        """
        FAIL-FAST GATE: Check if extraction produced measurable results.

        If no measurable pages found:
        - Generate minimal diagnostic outputs only
        - Return exit code 1

        Returns:
            None if extraction passed (continue pipeline)
            1 if extraction failed (stop pipeline)
        """
        # Count measurable pages from routing result
        routing_result = getattr(self, 'routing_result', {})
        page_results = routing_result.get("page_results", [])

        # Define measurable page types (exclude marketing, legend, perspective)
        MEASURABLE_TYPES = {"floor_plan", "structural", "section", "elevation", "detail", "candidate_plan"}
        NON_MEASURABLE_TYPES = {"schedule", "legend", "cover", "index", "perspective", "marketing", "unknown"}

        measurable_pages = 0
        non_measurable_pages = 0
        page_type_counts = {}

        for page in page_results:
            page_type = page.get("type", "unknown")
            page_type_counts[page_type] = page_type_counts.get(page_type, 0) + 1

            if page_type in MEASURABLE_TYPES:
                measurable_pages += 1
            else:
                non_measurable_pages += 1

        # Store for later use
        self.measurable_pages = measurable_pages
        self.non_measurable_pages = non_measurable_pages
        self.page_type_counts = page_type_counts

        # Check rooms/openings extracted
        rooms_count = len(self.rooms) if hasattr(self, 'rooms') else 0
        openings_count = len(self.openings) if hasattr(self, 'openings') else 0

        # FAIL-FAST: If no measurable content
        if measurable_pages == 0 and rooms_count == 0:
            print("\n" + "=" * 70)
            print("❌ EXTRACTION GATE FAILED - NO MEASURABLE CONTENT")
            print("=" * 70)
            print(f"Measurable pages: {measurable_pages}")
            print(f"Non-measurable pages: {non_measurable_pages}")
            print(f"Page types: {page_type_counts}")
            print(f"Rooms found: {rooms_count}")
            print(f"Openings found: {openings_count}")
            print("\nGenerating diagnostic outputs only...")
            print("=" * 70)

            # Generate minimal diagnostic outputs
            self._generate_extraction_gate_report(start_time, page_type_counts, measurable_pages)

            return 1  # Exit with error

        # Log warning if low measurable content
        if measurable_pages < 5 or rooms_count < 3:
            print(f"\n⚠️  LOW MEASURABLE CONTENT: {measurable_pages} measurable pages, {rooms_count} rooms")

        return None  # Continue pipeline

    def _generate_extraction_gate_report(
        self,
        start_time: datetime,
        page_type_counts: Dict[str, int],
        measurable_pages: int
    ) -> None:
        """Generate minimal diagnostic outputs when extraction fails."""
        duration = (datetime.now() - start_time).total_seconds()

        # 1. Generate input_gate_report.md
        input_gate_path = self.output_dir / "input_gate_report.md"
        with open(input_gate_path, "w") as f:
            f.write("# Input Gate Report - FAILED\n\n")
            f.write(f"**Project:** {self.project_id}\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration:** {duration:.1f}s\n\n")
            f.write("## Result: ❌ NO MEASURABLE CONTENT\n\n")
            f.write("The pipeline stopped because no measurable drawings were found.\n\n")
            f.write("### Page Type Analysis\n\n")
            f.write("| Type | Count | Measurable |\n")
            f.write("|------|-------|------------|\n")
            MEASURABLE_TYPES = {"floor_plan", "structural", "section", "elevation", "detail", "candidate_plan"}
            for ptype, count in sorted(page_type_counts.items(), key=lambda x: -x[1]):
                is_measurable = "✓" if ptype in MEASURABLE_TYPES else "✗"
                f.write(f"| {ptype} | {count} | {is_measurable} |\n")
            f.write(f"\n**Total Measurable Pages:** {measurable_pages}\n")
            f.write("\n### Next Steps\n\n")
            f.write("1. Verify the input files are architectural/engineering drawings\n")
            f.write("2. Ensure drawings have clear room labels and dimensions\n")
            f.write("3. Check that PDFs are not encrypted or corrupted\n")

        # 2. Generate page_type_report.md
        page_type_path = self.output_dir / "page_type_report.md"
        with open(page_type_path, "w") as f:
            f.write("# Page Type Classification Report\n\n")
            f.write(f"**Project:** {self.project_id}\n")
            f.write(f"**Total Pages:** {sum(page_type_counts.values())}\n")
            f.write(f"**Measurable Pages:** {measurable_pages}\n\n")
            f.write("## Classification Results\n\n")
            for ptype, count in sorted(page_type_counts.items(), key=lambda x: -x[1]):
                f.write(f"- **{ptype}**: {count} pages\n")

        # 3. Generate minimal run_metadata.json
        run_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{run_suffix}"

        metadata = {
            "run_id": run_id,
            "project_id": self.project_id,
            "mode": self.mode,
            "start_time": start_time.isoformat(),
            "duration_sec": round(duration, 1),
            "status": "FAILED",
            "failure_reason": "NO_MEASURABLE_CONTENT",
            "aggregates": {
                "pages_processed": sum(page_type_counts.values()),
                "measurable_pages": measurable_pages,
                "page_types": page_type_counts,
                "rooms_found": 0,
                "openings_found": 0,
                "boq_items_total": 0,
                "boq_measured": 0,
                "boq_inferred": 0,
                "coverage_percent": 0.0,
                "bid_recommendation": "NO-GO",
                "bid_score": 0,
            },
            "phases_run": ["01_index", "02_route", "03_extract"],
            "phases_failed": ["03_extract"],
            "output_files": [
                "input_gate_report.md",
                "page_type_report.md",
                "run_metadata.json",
            ],
        }

        meta_path = self.output_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✓ Generated: {input_gate_path.name}")
        print(f"✓ Generated: {page_type_path.name}")
        print(f"✓ Generated: {meta_path.name}")

    def _setup_output_dirs(self) -> None:
        """Create output directory structure."""
        dirs = [
            self.output_dir,
            self.output_dir / "overlays",
            self.output_dir / "boq",
            self.output_dir / "scope",
            self.output_dir / "rfi",
            self.output_dir / "packages",
            self.output_dir / "pricing",
            self.output_dir / "prelims",
            self.output_dir / "bid_book",
            self.output_dir / "validation",
            self.output_dir / "risk",
            self.output_dir / "measurement",
            self.output_dir / "materials",
            self.output_dir / "quotes",
            self.output_dir / "alignment",
            self.output_dir / "owner_docs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _load_project_metadata(self) -> None:
        """Load project metadata from intake files."""
        self.metadata = {
            "project_id": self.project_id,
            "profile": self.profile,
            "mode": self.mode,
        }

        # Load owner_inputs.yaml if exists
        owner_inputs_path = self.project_dir / "owner_inputs.yaml"
        if owner_inputs_path.exists():
            try:
                import yaml
                with open(owner_inputs_path) as f:
                    self.metadata["owner_inputs"] = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load owner_inputs.yaml: {e}")
                self.metadata["owner_inputs"] = {}
        else:
            self.metadata["owner_inputs"] = {}

        # Load project_intake.md if exists
        intake_path = self.project_dir / "project_intake.md"
        self.metadata["has_intake"] = intake_path.exists()

    def _get_quick_phases(self) -> List[Tuple[str, str, callable]]:
        """Get phases for quick mode (extract + RFIs only)."""
        return [
            ("01_index", "Index Drawings", self._phase_index),
            ("02_route", "Route Pages", self._phase_route),
            ("03_extract", "Extract Features", self._phase_extract),
            ("04_join", "Join Multi-Page", self._phase_join),
            ("12_rfi", "RFI Generation", self._phase_rfi),
        ]

    def _get_demo_phases(self) -> List[Tuple[str, str, callable]]:
        """Get phases for DEMO MODE (YC video).

        Only runs stable, verified phases needed for the demo:
        1. Input gate (PDF discovery + thumbnails)
        2. Page type classification (measurable vs non-measurable)
        3. Room detection (polygon)
        4. Opening count (basic)
        5. Provenance (3-bucket split: measured/counted/inferred)
        6. RFI generation (from missing schedules/scale/details)
        7. Bid gate decision

        SKIPS: Pricing, quote leveling, advanced BOQ, owner alignment, etc.
        """
        return [
            # EXTRACTION (1-4) - Core feature extraction
            ("01_index", "Index Drawings", self._phase_index),
            ("02_route", "Route Pages", self._phase_route),
            ("03_extract", "Extract Features", self._phase_extract),
            ("04_join", "Join Multi-Page", self._phase_join),

            # TAKEOFF + PROVENANCE (5-6a) - Generate BOQ with provenance
            ("05_takeoff", "Takeoff Generation", self._phase_takeoff),
            ("06_measurement", "Measurement Rules + Deductions", self._phase_measurement_rules),
            ("06a_provenance", "Attach Provenance", self._phase_attach_provenance),

            # EVIDENCE GATE (7) - For demo, this determines measured coverage
            ("07_measurement_gate", "Measurement Gate (HARD)", self._phase_measurement_gate),

            # SCOPE (8) - Show scope completeness
            ("08_scope", "Scope Completeness", self._phase_scope),

            # RFI (13) - Critical for demo: shows what's missing
            ("13_rfi", "RFI Engine", self._phase_rfi),

            # BID GATE (19) - Critical for demo: GO/NO-GO decision
            ("19_bid_gate", "Bid Gate", self._phase_bid_gate),

            # PROOF PACK (23) - Generate overlays for visual proof
            ("23_proof_pack", "Proof Pack Generation", self._phase_proof_pack),

            # ESTIMATOR RECONCILIATION (E1) - For measured/inferred CSVs
            ("E1_estimator", "Estimator Reconciliation", self._phase_estimator_reconciliation),
        ]

    def _get_full_phases(self) -> List[Tuple[str, str, callable]]:
        """Get all phases for full mode (with evidence-first gates and optional MEP)."""
        phases = [
            # EXTRACTION (1-4)
            ("01_index", "Index Drawings", self._phase_index),
            ("02_route", "Route Pages", self._phase_route),
            ("03_extract", "Extract Features", self._phase_extract),
            ("04_join", "Join Multi-Page", self._phase_join),

            # TAKEOFF + PROVENANCE (5-7)
            ("05_takeoff", "Takeoff Generation", self._phase_takeoff),
            ("06_measurement", "Measurement Rules + Deductions", self._phase_measurement_rules),
            ("06a_provenance", "Attach Provenance", self._phase_attach_provenance),

            # EVIDENCE GATE (7) - HARD STOP if geometry not reliable
            ("07_measurement_gate", "Measurement Gate (HARD)", self._phase_measurement_gate),

            # ANALYSIS (8-13) - continues regardless of gate
            ("08_scope", "Scope Completeness", self._phase_scope),
            ("09_triangulation", "Triangulation + Paranoia", self._phase_triangulation),
            ("10_bom", "BOM + Procurement", self._phase_bom),
            ("11_revision", "Revision Intelligence", self._phase_revision),
            ("12_doubt", "Doubt Engine", self._phase_doubt),
            ("13_rfi", "RFI Engine", self._phase_rfi),

            # OWNER ALIGNMENT (14-15)
            ("14_owner_docs", "Owner Docs Parsing", self._phase_owner_docs),
            ("15_alignment", "Owner BOQ Alignment", self._phase_alignment),

            # PRICING (16-18) - ONLY if measurement gate passed or allow_inferred_pricing
            ("16_pricing", "Pricing + Scenarios", self._phase_pricing),
            ("17_quotes", "Quote Leveling", self._phase_quote_leveling),
            ("18_prelims", "Prelims Generator", self._phase_prelims),

            # BID SUBMISSION (19-22)
            ("19_bid_gate", "Bid Gate", self._phase_bid_gate),
            ("20_bid_docs", "Bid Docs Generation", self._phase_bid_docs),
            ("21_packages", "Package Exports", self._phase_packages),
            ("22_bid_book", "Bid Book Export", self._phase_bid_book),

            # PROOF PACK (23) - generate overlays and proof documentation
            ("23_proof_pack", "Proof Pack Generation", self._phase_proof_pack),

            # ESTIMATOR RECONCILIATION (E1) - transforms parser output to estimator-ready
            ("E1_estimator", "Estimator Reconciliation", self._phase_estimator_reconciliation),

            # PDF REPORT (E2) - generate professional bid report PDF
            ("E2_pdf_report", "PDF Report Generation", self._phase_pdf_report),
        ]

        # Add MEP phases if enabled
        if self.enable_mep:
            phases.extend([
                # MEP DETECTION & TAKEOFF (M1-M4)
                ("M1_mep_detect", "MEP Device Detection", self._phase_mep_detect),
                ("M2_mep_connect", "MEP Connectivity", self._phase_mep_connectivity),
                ("M3_mep_systems", "MEP Systems Grouping", self._phase_mep_systems),
                ("M4_mep_takeoff", "MEP Takeoff Export", self._phase_mep_takeoff),
            ])

        # NOTE: Output verification is done AFTER summary.md is written
        # See run() method - it's not a phase anymore

        return phases

    def _run_phase(self, phase_id: str, phase_name: str, phase_func: callable) -> None:
        """Run a single phase with error handling."""
        print(f"\n{'='*60}")
        print(f"PHASE {phase_id}: {phase_name}")
        print(f"{'='*60}")

        # Before PDF report phase, save intermediate metadata so PDF can read it
        if phase_id == "E2_pdf_report":
            self._save_intermediate_metadata()

        phase_start = datetime.now()

        try:
            result = phase_func()
            result.duration_sec = (datetime.now() - phase_start).total_seconds()

            if result.skipped:
                print(f"⏭️  {phase_name} skipped: {result.skip_reason}")
            elif result.success:
                print(f"✅ {phase_name} completed ({result.duration_sec:.1f}s)")
                if result.message:
                    print(f"   {result.message}")
            else:
                print(f"⚠️  {phase_name} failed: {result.error}")

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Phase {phase_name} crashed: {e}\n{tb}")
            result = PhaseResult(
                phase=phase_id,
                phase_name=phase_name,
                success=False,
                error=str(e),
                stack_trace=tb,
            )
            result.duration_sec = (datetime.now() - phase_start).total_seconds()
            print(f"❌ {phase_name} crashed: {e}")

        self.results.append(result)

    def _save_intermediate_metadata(self) -> None:
        """Save intermediate metadata for PDF report generation."""
        # Build temporary summary from current results
        temp_summary = {
            "project_id": self.project_id,
            "mode": self.mode,
            "total_phases": len(self.results),
            "successful": sum(1 for r in self.results if r.success and not r.skipped),
            "failed": sum(1 for r in self.results if not r.success and not r.skipped),
            "skipped": sum(1 for r in self.results if r.skipped),
            "phases": [
                {
                    "phase": r.phase,
                    "phase_name": r.phase_name,
                    "success": r.success,
                    "skipped": r.skipped,
                    "message": r.message,
                    "error": r.error,
                    "skip_reason": r.skip_reason,
                    "stack_trace": r.stack_trace,
                    "duration_sec": r.duration_sec,
                    "data": r.data,
                }
                for r in self.results
            ],
        }
        self.summary = temp_summary

        # Generate run_id
        run_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_suffix}"

        # Extract aggregates
        aggregates = self._extract_run_aggregates()

        metadata = {
            "run_id": run_id,
            "project_id": self.project_id,
            "mode": self.mode,
            "profile": self.profile,
            "rules_version": self.rules_version,
            "start_time": datetime.now().isoformat(),
            "duration_sec": 0,  # Will be updated in final save
            "aggregates": aggregates,
            "phases_run": [p["phase"] for p in temp_summary.get("phases", []) if not p.get("skipped")],
            "phases_skipped": [p["phase"] for p in temp_summary.get("phases", []) if p.get("skipped")],
            "phases_failed": [p["phase"] for p in temp_summary.get("phases", []) if not p.get("success") and not p.get("skipped")],
            "output_completeness_score": 0,
            "output_verification": [],
            "summary": temp_summary,
        }

        # Save intermediate metadata
        meta_path = self.output_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved intermediate metadata with run_id={run_id} for PDF generation")

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    def _phase_index(self) -> PhaseResult:
        """Phase 1: Index drawings."""
        try:
            from src.adapters.index_adapter import ProjectIndexer

            indexer = ProjectIndexer(self.project_dir)
            index_result = indexer.index_drawings(self.drawings_dir)

            pages_found = index_result.get("total_pages", 0)
            files_found = len(index_result.get("files", []))

            # ===== HARD GUARD: Validate PDF page counts =====
            # If we have PDFs but reported only 1 page per PDF, something is wrong
            for f in index_result.get("files", []):
                if isinstance(f, dict) and f.get("suffix", "").lower() == ".pdf":
                    file_pages = f.get("page_count", 1)
                    if file_pages == 1:
                        # Double-check with fitz
                        try:
                            import fitz
                            doc = fitz.open(f.get("path", ""))
                            actual = len(doc)
                            doc.close()
                            if actual > 1:
                                print(f"⚠️  WARNING: {f.get('name')} has {actual} pages but indexer reported {file_pages}")
                                # Correct the count
                                pages_found += (actual - file_pages)
                                f["page_count"] = actual
                                index_result["total_pages"] = pages_found
                        except Exception:
                            pass

            # Track source files for proof pack
            files = index_result.get("files", [])
            self.source_files = []
            for f in files:
                if isinstance(f, dict):
                    # File is a dict with path info
                    self.source_files.append(Path(f.get("path", f.get("file", ""))))
                elif isinstance(f, (str, Path)):
                    self.source_files.append(Path(f))

            # Store actual page count for later phases
            self.total_pages_indexed = pages_found

            return PhaseResult(
                phase="01_index",
                phase_name="Index Drawings",
                success=pages_found > 0 or files_found > 0,
                message=f"Found {pages_found} pages from {files_found} files",
                data=index_result,
                error=None if pages_found > 0 or files_found > 0 else "No pages found"
            )
        except ImportError as e:
            # Fallback: manual indexing
            drawings = list(self.drawings_dir.glob("*.[pP][dD][fF]")) + \
                       list(self.drawings_dir.glob("*.[pP][nN][gG]")) + \
                       list(self.drawings_dir.glob("*.[jJ][pP][gG]"))

            # Track source files for proof pack
            self.source_files = drawings

            return PhaseResult(
                phase="01_index",
                phase_name="Index Drawings",
                success=len(drawings) > 0,
                message=f"Found {len(drawings)} drawing files (basic index)",
                data={"files": [str(d) for d in drawings]},
                error=None if drawings else "No drawings found"
            )

    def _phase_route(self) -> PhaseResult:
        """Phase 2: Route pages by type using multipage router with drawing-likeness scoring."""
        try:
            from src.adapters.multipage_router import MultipageRouter

            router = MultipageRouter(candidate_threshold=0.2)  # Lower threshold to catch more pages
            route_result = router.route_directory(self.drawings_dir, self.output_dir)

            total_pages = route_result.get("total_pages", 0)
            candidates = route_result.get("candidates", 0)
            types_found = route_result.get("types", {})
            scales_detected = route_result.get("scales_detected", [])

            # Store routing result for extract phase
            self.routing_result = route_result

            # Store detected scales
            if scales_detected and not self.manual_scale:
                # Use first detected scale
                scale_str = scales_detected[0]  # e.g., "1:100"
                try:
                    self.detected_scale = int(scale_str.split(":")[1])
                except (ValueError, IndexError):
                    self.detected_scale = None

            message = f"Routed {total_pages} pages: {candidates} candidates, types={types_found}"
            if scales_detected:
                message += f", scales={scales_detected}"

            return PhaseResult(
                phase="02_route",
                phase_name="Route Pages",
                success=True,
                message=message,
                data=route_result,
            )
        except ImportError as e:
            # Fallback to old router
            try:
                from src.adapters.route_adapter import PageRouter
                router = PageRouter()
                route_result = router.route_directory(self.drawings_dir)
                types_found = route_result.get("types", {})
                self.routing_result = route_result
                return PhaseResult(
                    phase="02_route",
                    phase_name="Route Pages",
                    success=True,
                    message=f"Routed pages: {types_found} (fallback router)",
                    data=route_result,
                )
            except Exception:
                return PhaseResult(
                    phase="02_route",
                    phase_name="Route Pages",
                    success=True,
                    skipped=True,
                    skip_reason=f"Module not available: {e}",
                )

    def _phase_extract(self) -> PhaseResult:
        """Phase 3: Extract rooms, walls, openings from ALL candidate pages."""
        try:
            from src.adapters.multipage_extractor import MultipageExtractor, extract_multipage_project

            # Get scale (manual override > detected > None)
            scale_to_use = self.manual_scale
            if not scale_to_use and hasattr(self, 'detected_scale') and self.detected_scale:
                scale_to_use = self.detected_scale

            # Use routing result if available
            routing_result = getattr(self, 'routing_result', None)

            if routing_result:
                results = extract_multipage_project(
                    self.drawings_dir,
                    self.output_dir,
                    routing_result,
                    scale_override=scale_to_use,
                )
            else:
                # Fallback to old extractor
                from src.adapters.extract_adapter import process_project_drawings
                results = process_project_drawings(
                    self.drawings_dir,
                    self.output_dir,
                    profile=self.profile,
                )

            pages_processed = results.get("pages_processed", 0)
            rooms_found = results.get("total_rooms", 0)
            openings_found = results.get("total_openings", 0)

            # Store scale info if detected
            scale_info = results.get("scale") or (results.get("file_results", [{}])[0].get("scale") if results.get("file_results") else None)
            if scale_info and scale_info.get("detected"):
                self.detected_scale = scale_info.get("ratio")
                self.scale_method = scale_info.get("method")
                self.scale_confidence = scale_info.get("confidence", 0)

            # Store for later phases
            self._load_extracted_data()

            message = f"Processed {pages_processed} pages, {rooms_found} rooms, {openings_found} openings"
            if hasattr(self, 'detected_scale') and self.detected_scale:
                message += f" (scale 1:{self.detected_scale})"

            return PhaseResult(
                phase="03_extract",
                phase_name="Extract Features",
                success=pages_processed > 0 or rooms_found > 0,
                message=message,
                data=results,
                error=None if pages_processed > 0 or rooms_found > 0 else "No data extracted"
            )
        except Exception as e:
            logger.exception("Extract phase failed")
            return PhaseResult(
                phase="03_extract",
                phase_name="Extract Features",
                success=False,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )

    def _phase_join(self) -> PhaseResult:
        """Phase 4: Join multi-page project."""
        try:
            from src.adapters.join_adapter import ProjectJoiner

            joiner = ProjectJoiner()
            join_result = joiner.join_project(self.output_dir)

            # Reload data after join
            self._load_extracted_data()

            pages_joined = join_result.get("pages_joined", 0)
            total_rooms = join_result.get("total_rooms", 0)

            return PhaseResult(
                phase="04_join",
                phase_name="Join Multi-Page",
                success=True,
                message=f"Joined {pages_joined} pages, {total_rooms} rooms",
                data=join_result,
            )
        except ImportError:
            return PhaseResult(
                phase="04_join",
                phase_name="Join Multi-Page",
                success=True,
                skipped=True,
                skip_reason="Module not available",
            )

    def _phase_takeoff(self) -> PhaseResult:
        """Phase 5: Takeoff generation (BOQ items)."""
        try:
            from src.adapters.takeoff_adapter import run_takeoff_generation

            takeoff_result = run_takeoff_generation(
                rooms=self.rooms,
                openings=self.openings,
                output_dir=self.output_dir,
                profile=self.profile,
            )

            self.boq_items = takeoff_result.get("boq_items", [])
            items_generated = len(self.boq_items)

            return PhaseResult(
                phase="05_takeoff",
                phase_name="Takeoff Generation",
                success=True,
                message=f"Generated {items_generated} BOQ items",
                data={"items_count": items_generated},
            )
        except ImportError as e:
            # Fallback: generate basic BOQ from rooms
            return self._fallback_takeoff()

    def _fallback_takeoff(self) -> PhaseResult:
        """Fallback takeoff generation."""
        import csv

        boq_items = []
        item_id = 0

        for room in self.rooms:
            area = room.get("area_sqm", 15.0)
            perimeter = room.get("perimeter_m", 16.0)
            room_type = room.get("room_type", "unknown")
            label = room.get("label", "Room")

            # Flooring
            item_id += 1
            boq_items.append({
                "item_id": f"FLR-{item_id:03d}",
                "description": f"Flooring in {label}",
                "quantity": area * 1.05,  # 5% wastage
                "unit": "sqm",
                "room_id": room.get("id"),
                "package": "flooring",
            })

            # Wall finish
            wall_area = perimeter * 3.0  # 3m height
            item_id += 1
            boq_items.append({
                "item_id": f"WF-{item_id:03d}",
                "description": f"Wall finish in {label}",
                "quantity": wall_area,
                "unit": "sqm",
                "room_id": room.get("id"),
                "package": "wall_finishes",
            })

            # Ceiling
            item_id += 1
            boq_items.append({
                "item_id": f"CLG-{item_id:03d}",
                "description": f"Ceiling in {label}",
                "quantity": area,
                "unit": "sqm",
                "room_id": room.get("id"),
                "package": "wall_finishes",
            })

            # Waterproofing for wet areas
            if room_type in ["toilet", "bathroom", "kitchen", "utility"]:
                item_id += 1
                boq_items.append({
                    "item_id": f"WP-{item_id:03d}",
                    "description": f"Waterproofing in {label}",
                    "quantity": area * 1.1,
                    "unit": "sqm",
                    "room_id": room.get("id"),
                    "package": "waterproofing",
                })

        # Door/window items
        for opening in self.openings:
            item_id += 1
            op_type = opening.get("type", "door")
            tag = opening.get("tag", f"O{item_id}")
            width = opening.get("width_m", 0.9)
            height = opening.get("height_m", 2.1)

            boq_items.append({
                "item_id": f"DW-{item_id:03d}",
                "description": f"{op_type.title()} {tag} ({width}m x {height}m)",
                "quantity": 1,
                "unit": "no",
                "package": "doors_windows",
            })

        self.boq_items = boq_items

        # Write BOQ CSV
        boq_path = self.output_dir / "boq" / "boq_quantities.csv"
        with open(boq_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["item_id", "description", "quantity", "unit", "room_id", "package"])
            writer.writeheader()
            writer.writerows(boq_items)

        # Write provisional items (empty for now)
        prov_path = self.output_dir / "boq" / "provisional_items.csv"
        with open(prov_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "description", "quantity", "unit", "reason"])

        return PhaseResult(
            phase="05_takeoff",
            phase_name="Takeoff Generation",
            success=True,
            message=f"Generated {len(boq_items)} BOQ items (fallback)",
            data={"items_count": len(boq_items)},
        )

    def _phase_measurement_rules(self) -> PhaseResult:
        """Phase 6: Measurement rules + deductions."""
        try:
            from src.adapters.estimator_math_adapter import run_estimator_math

            result = run_estimator_math(
                boq_items=self.boq_items,
                rooms=self.rooms,
                openings=self.openings,
                project_params=self.metadata.get("owner_inputs", {}).get("project", {}),
                output_dir=self.output_dir / "measurement",
            )

            deductions = result.get("summary", {}).get("deductions", {}).get("entries", 0)

            return PhaseResult(
                phase="06_measurement",
                phase_name="Measurement Rules + Deductions",
                success=True,
                message=f"Applied {deductions} deductions (IS 1200/CPWD rules)",
                data=result.get("summary", {}),
            )
        except ImportError as e:
            return PhaseResult(
                phase="06_measurement",
                phase_name="Measurement Rules + Deductions",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_attach_provenance(self) -> PhaseResult:
        """Phase 6a: Attach provenance to BOQ items."""
        try:
            from src.provenance import (
                attach_provenance_to_boq,
                write_split_boq_files,
                ProvenanceTracker,
            )

            # Load scale info from measurement output
            scale_info = {}
            meas_file = self.output_dir / "measurement" / "estimator_math_summary.json"
            if meas_file.exists():
                with open(meas_file) as f:
                    meas_data = json.load(f)
                    scale_info = {
                        "scale": meas_data.get("scale", 100),
                        "basis": meas_data.get("scale_basis", "unknown"),
                        "confidence": meas_data.get("scale_confidence", 0.0),
                    }

            # If manual scale provided, use it
            if self.manual_scale:
                scale_info = {
                    "scale": self.manual_scale,
                    "basis": "manual",
                    "confidence": 1.0,
                }

            # Attach provenance to each BOQ item
            self.boq_items = attach_provenance_to_boq(
                boq_items=self.boq_items,
                rooms=self.rooms,
                openings=self.openings,
                scale_info=scale_info,
            )

            # Create provenance tracker
            self.provenance_tracker = ProvenanceTracker(self.project_id, self.output_dir)
            for i, item in enumerate(self.boq_items):
                item_id = item.get("item_id", f"item_{i}")
                if "_provenance_obj" in item:
                    self.provenance_tracker.add_item(item_id, item["_provenance_obj"])

            # Save provenance data
            self.provenance_tracker.save()

            # Write split BOQ files (measured vs inferred vs counted)
            split_result = write_split_boq_files(
                output_dir=self.output_dir,
                boq_items=self.boq_items,
            )

            # CRITICAL GUARD: If measured.json is empty, DISABLE pricing
            # This prevents fabricated pricing based on template quantities
            if split_result.get("measured_empty", True):
                self.can_produce_pricing = False
                logger.warning("⚠️  MEASURED.JSON EMPTY - Pricing DISABLED")
                logger.warning("   All quantities are inferred/counted - no geometry-backed measurements")

            # Store split result for later phases
            self.provenance_split_result = split_result

            return PhaseResult(
                phase="06a_provenance",
                phase_name="Attach Provenance",
                success=True,
                message=f"Measured: {split_result['measured_count']}, Counted: {split_result['counted_count']}, Inferred: {split_result['inferred_count']} ({split_result['measurement_coverage']:.0%} coverage)",
                data=split_result,
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(f"Provenance attachment failed: {e}")
            return PhaseResult(
                phase="06a_provenance",
                phase_name="Attach Provenance",
                success=True,
                skipped=True,
                skip_reason=f"Provenance module error: {e}",
            )

    def _phase_measurement_gate(self) -> PhaseResult:
        """Phase 7: Measurement Gate (HARD STOP).

        This is a critical gate that determines whether we can produce
        reliable measured quantities.

        If FAIL:
        - Still produce scope register, RFIs, missing inputs checklist
        - DO NOT produce priced estimate
        - DO NOT produce "measured BOQ" quantities
        """
        try:
            from src.gates import run_measurement_gate, GateStatus

            # Build metadata with manual scale if provided
            gate_metadata = {**self.metadata}
            if self.manual_scale:
                gate_metadata["manual_scale"] = self.manual_scale
            # Also pass detected scale from routing/extraction
            if hasattr(self, 'detected_scale') and self.detected_scale:
                gate_metadata["detected_scale"] = self.detected_scale
                gate_metadata["scale_method"] = getattr(self, 'scale_method', 'detected')
                gate_metadata["scale_confidence"] = getattr(self, 'scale_confidence', 0.5)

            # Run the gate
            self.measurement_gate_result = run_measurement_gate(
                output_dir=self.output_dir,
                project_metadata=gate_metadata,
            )

            status = self.measurement_gate_result.status
            can_measure = self.measurement_gate_result.can_produce_measured_boq
            can_price = self.measurement_gate_result.can_produce_pricing

            # Update pricing permission
            self.can_produce_pricing = can_price or self.allow_inferred_pricing

            # Determine message based on status
            if status == GateStatus.PASS:
                message = f"PASS - Geometry verified (scale: {self.measurement_gate_result.scale_confidence:.0%}, coverage: {self.measurement_gate_result.geometry_coverage:.0%})"
                success = True
            elif status == GateStatus.WARN:
                message = f"WARN - Proceed with caution ({len(self.measurement_gate_result.warnings)} warnings)"
                success = True
            else:
                blockers = ", ".join(self.measurement_gate_result.blockers[:2])
                message = f"FAIL - {blockers}"
                success = False

                # Provide actionable feedback
                print("\n" + "=" * 60)
                print("⚠️  MEASUREMENT GATE FAILED")
                print("=" * 60)
                print("To unblock measurement, you can:")
                print("  1. Provide --scale <ratio>  (e.g., --scale 100 for 1:100)")
                print("  2. Add a page with explicit 'Scale 1:xx' text note")
                print("  3. Ensure dimension text is present near dimension lines")
                if hasattr(self, 'routing_result') and self.routing_result:
                    scales = self.routing_result.get("scales_detected", [])
                    if scales:
                        print(f"  → Scales detected in drawings: {scales}")
                print("=" * 60 + "\n")

            # Add note about pricing
            if not self.can_produce_pricing:
                message += " | Pricing SKIPPED (no reliable rates/measurements)"
            elif self.allow_inferred_pricing and not can_price:
                message += " | Pricing ALLOWED via --allow_inferred_pricing (CAUTION)"

            return PhaseResult(
                phase="07_measurement_gate",
                phase_name="Measurement Gate (HARD)",
                success=success,
                message=message,
                data=self.measurement_gate_result.to_dict(),
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Measurement gate error: {e}\n{tb}")

            # On error, fail safe - don't allow pricing
            self.can_produce_pricing = self.allow_inferred_pricing

            return PhaseResult(
                phase="07_measurement_gate",
                phase_name="Measurement Gate (HARD)",
                success=False,
                error=str(e),
                stack_trace=tb,
            )

    def _phase_proof_pack(self) -> PhaseResult:
        """Phase 23: Generate proof pack with overlays and documentation."""
        try:
            from src.provenance import generate_proof_pack

            result = generate_proof_pack(
                output_dir=self.output_dir,
                boq_items=self.boq_items,
                rooms=self.rooms,
                openings=self.openings,
                source_files=self.source_files,
            )

            overlays_count = len(result.get("overlays_generated", []))

            return PhaseResult(
                phase="23_proof_pack",
                phase_name="Proof Pack Generation",
                success=True,
                message=f"Generated {overlays_count} overlays, proof_pack.md created",
                data=result,
            )

        except Exception as e:
            logger.warning(f"Proof pack generation failed: {e}")
            return PhaseResult(
                phase="23_proof_pack",
                phase_name="Proof Pack Generation",
                success=True,
                skipped=True,
                skip_reason=f"Proof pack error: {e}",
            )

    # =========================================================================
    # ESTIMATOR RECONCILIATION PHASE
    # =========================================================================

    def _phase_estimator_reconciliation(self) -> PhaseResult:
        """Phase E1: Estimator Reconciliation - transforms parser output to estimator-ready.

        This is the key phase that makes XBOQ an estimator assistant, not just a parser:
        1. Loads estimator_inputs.yaml for assumptions and overrides
        2. Merges measured and inferred BOQ into estimator view
        3. Detects missing scope items using rules
        4. Generates interactive Excel workbook with editable columns
        5. Produces bid_gate.md with GO/REVIEW/NO-GO recommendation
        """
        try:
            from src.estimator.reconciler import (
                EstimatorAssumptions,
                run_estimator_reconciliation,
            )

            # Create assumptions from CLI args
            assumptions = EstimatorAssumptions(
                wall_height_m=self.assume_wall_height,
                door_height_m=self.assume_door_height,
                plaster_both_sides=self.assume_plaster_both_sides,
                floor_finish_all_rooms=self.assume_floor_finish_all_rooms,
            )

            # Run full reconciliation with inputs, Excel, and bid gate
            result = run_estimator_reconciliation(
                output_dir=self.output_dir,
                assumptions=assumptions,
                export_excel=True,
                project_id=self.project_id,
                apply_overrides=getattr(self, 'apply_overrides', False),
            )

            # Get bid gate recommendation
            bid_recommendation = "N/A"
            bid_score = 0
            try:
                from src.estimator.bid_gate import quick_bid_gate
                bid_result = quick_bid_gate(self.output_dir, self.project_id)
                bid_recommendation = bid_result.recommendation.value
                bid_score = bid_result.score
            except Exception:
                pass

            # Build summary message
            measured_pct = result.coverage_percent
            missing_count = result.total_missing
            review_count = result.needs_review_count

            message_parts = [
                f"Measured: {result.total_measured}",
                f"Inferred: {result.total_inferred}",
                f"({measured_pct:.0f}% coverage)",
                f"Missing: {missing_count}",
                f"Bid: {bid_recommendation}",
            ]

            return PhaseResult(
                phase="E1_estimator",
                phase_name="Estimator Reconciliation",
                success=True,
                message=" | ".join(message_parts),
                data={
                    "total_measured": result.total_measured,
                    "total_inferred": result.total_inferred,
                    "coverage_percent": measured_pct,
                    "missing_scope_count": missing_count,
                    "needs_review_count": review_count,
                    "rfis_generated": len(result.missing_scope_rfis),
                    "bid_recommendation": bid_recommendation,
                    "bid_score": bid_score,
                },
            )

        except ImportError as e:
            logger.warning(f"Estimator reconciliation module not available: {e}")
            return PhaseResult(
                phase="E1_estimator",
                phase_name="Estimator Reconciliation",
                success=True,
                skipped=True,
                skip_reason="Estimator module not available",
            )
        except Exception as e:
            logger.exception(f"Estimator reconciliation failed: {e}")
            return PhaseResult(
                phase="E1_estimator",
                phase_name="Estimator Reconciliation",
                success=False,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )

    def _phase_pdf_report(self) -> PhaseResult:
        """
        PDF Report Phase E2: Generate professional bid report PDF.

        Creates a comprehensive PDF report including:
        1. Title page with project info and bid recommendation
        2. Drawing analysis summary
        3. Measurement confidence by page
        4. Missing scope items (high priority)
        5. Assumptions used
        6. Final BOQ table

        Output: out/<project_id>/bid_report.pdf
        """
        try:
            from src.report.pdf_report import generate_bid_report

            pdf_path = generate_bid_report(
                output_dir=self.output_dir,
                project_id=self.project_id,
            )

            if pdf_path and pdf_path.exists():
                file_size_kb = pdf_path.stat().st_size / 1024
                return PhaseResult(
                    phase="E2_pdf_report",
                    phase_name="PDF Report Generation",
                    success=True,
                    message=f"Generated {pdf_path.name} ({file_size_kb:.1f} KB)",
                    data={
                        "pdf_path": str(pdf_path),
                        "file_size_kb": file_size_kb,
                    },
                )
            else:
                return PhaseResult(
                    phase="E2_pdf_report",
                    phase_name="PDF Report Generation",
                    success=False,
                    error="PDF generation returned None or file not created",
                )

        except ImportError as e:
            logger.warning(f"PDF report module not available: {e}")
            return PhaseResult(
                phase="E2_pdf_report",
                phase_name="PDF Report Generation",
                success=True,
                skipped=True,
                skip_reason=f"PDF module not available: {e}",
            )
        except Exception as e:
            logger.exception(f"PDF report generation failed: {e}")
            return PhaseResult(
                phase="E2_pdf_report",
                phase_name="PDF Report Generation",
                success=False,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )

    # =========================================================================
    # MEP PHASES (Optional, enabled with --enable-mep)
    # =========================================================================

    def _phase_mep_detect(self) -> PhaseResult:
        """MEP Phase M1: Device detection."""
        try:
            from src.adapters.mep_adapter import run_mep_device_detection

            scale = self.manual_scale or 100

            result = run_mep_device_detection(
                drawings_dir=self.drawings_dir,
                output_dir=self.output_dir,
                rooms=self.rooms,
                scale=scale,
            )

            devices_count = result.get("devices_count", 0)
            summary = result.get("summary", {})

            # Store for later phases
            self.mep_devices = result.get("devices", [])

            return PhaseResult(
                phase="M1_mep_detect",
                phase_name="MEP Device Detection",
                success=devices_count > 0,
                message=f"Detected {devices_count} devices ({summary.get('devices_needing_rfi', 0)} need RFIs)",
                data=summary,
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"MEP detection failed: {e}\n{tb}")
            return PhaseResult(
                phase="M1_mep_detect",
                phase_name="MEP Device Detection",
                success=False,
                error=str(e),
                stack_trace=tb,
            )

    def _phase_mep_connectivity(self) -> PhaseResult:
        """MEP Phase M2: Connectivity inference."""
        try:
            from src.adapters.mep_adapter import run_mep_connectivity

            scale = self.manual_scale or 100

            result = run_mep_connectivity(
                output_dir=self.output_dir,
                devices=self.mep_devices,
                rooms=self.rooms,
                scale=scale,
            )

            connections_count = result.get("connections_count", 0)
            cable_summary = result.get("cable_summary", {})
            total_cable = cable_summary.get("cables", {})

            # Store for later phases
            self.mep_connections = result.get("connections", [])

            total_cable_m = sum(c.get("total_length_m", 0) for c in total_cable.values()) if isinstance(total_cable, dict) else 0

            return PhaseResult(
                phase="M2_mep_connect",
                phase_name="MEP Connectivity",
                success=True,
                message=f"Inferred {connections_count} connections (~{total_cable_m:.0f}m cable)",
                data=result,
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(f"MEP connectivity failed: {e}")
            return PhaseResult(
                phase="M2_mep_connect",
                phase_name="MEP Connectivity",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_mep_systems(self) -> PhaseResult:
        """MEP Phase M3: Systems grouping."""
        try:
            from src.adapters.mep_adapter import run_mep_systems_grouping

            result = run_mep_systems_grouping(
                output_dir=self.output_dir,
                devices=self.mep_devices,
            )

            summary = result.get("summary", {})
            total_systems = summary.get("total_systems", 0)
            systems_with_devices = len([s for s in summary.get("systems", {}).values() if s.get("total_devices", 0) > 0])

            return PhaseResult(
                phase="M3_mep_systems",
                phase_name="MEP Systems Grouping",
                success=True,
                message=f"Grouped into {systems_with_devices} active systems",
                data=summary,
            )

        except Exception as e:
            logger.warning(f"MEP systems grouping failed: {e}")
            return PhaseResult(
                phase="M3_mep_systems",
                phase_name="MEP Systems Grouping",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_mep_takeoff(self) -> PhaseResult:
        """MEP Phase M4: Takeoff generation and export."""
        try:
            from src.adapters.mep_adapter import run_mep_takeoff

            result = run_mep_takeoff(
                output_dir=self.output_dir,
                project_id=self.project_id,
                devices=self.mep_devices,
                connections=self.mep_connections,
            )

            total_devices = result.get("total_devices", 0)
            measured = result.get("measured_count", 0)
            rfi_count = result.get("rfi_count", 0)

            # Store for summary
            self.mep_takeoff = result

            return PhaseResult(
                phase="M4_mep_takeoff",
                phase_name="MEP Takeoff Export",
                success=True,
                message=f"Takeoff: {total_devices} devices ({measured} measured), {rfi_count} RFIs → CSV/Excel",
                data={
                    "total_devices": total_devices,
                    "measured_count": measured,
                    "rfi_count": rfi_count,
                },
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"MEP takeoff failed: {e}")
            return PhaseResult(
                phase="M4_mep_takeoff",
                phase_name="MEP Takeoff Export",
                success=False,
                error=str(e),
                stack_trace=tb,
            )

    def _phase_scope(self) -> PhaseResult:
        """Phase 7: Scope completeness analysis."""
        try:
            from src.adapters.scope_adapter import run_scope_analysis

            scope_result = run_scope_analysis(
                output_dir=self.output_dir,
                project_metadata=self.metadata,
            )

            completeness = scope_result.get("completeness", 0)
            packages = len(scope_result.get("packages", []))

            self.scope_register = scope_result

            # Write scope register CSV
            self._write_scope_register_csv(scope_result)

            return PhaseResult(
                phase="07_scope",
                phase_name="Scope Completeness",
                success=True,
                message=f"Scope completeness: {completeness:.0%}, {packages} packages analyzed",
                data=scope_result,
            )
        except Exception as e:
            # Fallback scope analysis
            return self._fallback_scope()

    def _fallback_scope(self) -> PhaseResult:
        """Fallback scope completeness."""
        import csv

        packages = ["rcc_structural", "masonry", "waterproofing", "flooring",
                   "doors_windows", "wall_finishes", "plumbing", "electrical"]

        scope_items = []
        for pkg in packages:
            # Check if we have BOQ items for this package
            pkg_items = [b for b in self.boq_items if b.get("package") == pkg]
            status = "CONFIRMED" if pkg_items else "UNKNOWN"

            scope_items.append({
                "package": pkg,
                "status": status,
                "items_count": len(pkg_items),
                "evidence": "BOQ items" if pkg_items else "No evidence",
            })

        # Calculate completeness
        confirmed = sum(1 for s in scope_items if s["status"] == "CONFIRMED")
        completeness = confirmed / len(packages) if packages else 0

        # Write scope register
        scope_dir = self.output_dir / "scope"
        scope_dir.mkdir(parents=True, exist_ok=True)
        scope_path = scope_dir / "scope_register.csv"
        with open(scope_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["package", "status", "items_count", "evidence"])
            writer.writeheader()
            writer.writerows(scope_items)

        self.scope_register = {"completeness": completeness, "packages": scope_items}

        return PhaseResult(
            phase="07_scope",
            phase_name="Scope Completeness",
            success=True,
            message=f"Scope completeness: {completeness:.0%} (fallback)",
            data=self.scope_register,
        )

    def _phase_triangulation(self) -> PhaseResult:
        """Phase 8: Triangulation + overrides + paranoia."""
        try:
            from src.adapters.triangulation_adapter import run_triangulation

            tri_result = run_triangulation(
                boq_items=self.boq_items,
                rooms=self.rooms,
                output_dir=self.output_dir,
            )

            discrepancies = tri_result.get("discrepancies", 0)
            return PhaseResult(
                phase="08_triangulation",
                phase_name="Triangulation + Paranoia",
                success=True,
                message=f"Found {discrepancies} quantity discrepancies",
                data=tri_result,
            )
        except ImportError:
            return PhaseResult(
                phase="08_triangulation",
                phase_name="Triangulation + Paranoia",
                success=True,
                skipped=True,
                skip_reason="Module not available",
            )

    def _phase_bom(self) -> PhaseResult:
        """Phase 9: BOM + procurement summary."""
        try:
            from src.adapters.bom_adapter import run_bom_generation

            bom_result = run_bom_generation(
                boq_items=self.boq_items,
                output_dir=self.output_dir,
            )

            materials = bom_result.get("materials_count", 0)
            return PhaseResult(
                phase="09_bom",
                phase_name="BOM + Procurement",
                success=True,
                message=f"Generated BOM with {materials} materials",
                data=bom_result,
            )
        except ImportError:
            # Fallback: generate basic material estimate
            return self._fallback_bom()

    def _fallback_bom(self) -> PhaseResult:
        """Fallback BOM generation."""
        import csv

        materials = []

        # Aggregate by material type
        material_totals = {}
        for item in self.boq_items:
            pkg = item.get("package", "general")
            qty = item.get("quantity", 0)
            unit = item.get("unit", "sqm")

            key = (pkg, unit)
            if key not in material_totals:
                material_totals[key] = 0
            material_totals[key] += qty

        for (pkg, unit), total in material_totals.items():
            materials.append({
                "material": pkg.replace("_", " ").title(),
                "quantity": round(total, 2),
                "unit": unit,
                "package": pkg,
            })

        # Write material estimate
        mat_path = self.output_dir / "boq" / "material_estimate.csv"
        with open(mat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["material", "quantity", "unit", "package"])
            writer.writeheader()
            writer.writerows(materials)

        return PhaseResult(
            phase="09_bom",
            phase_name="BOM + Procurement",
            success=True,
            message=f"Generated {len(materials)} material categories (fallback)",
            data={"materials_count": len(materials)},
        )

    def _phase_revision(self) -> PhaseResult:
        """Phase 10: Revision intelligence."""
        try:
            from src.adapters.revision_adapter import run_revision_engine

            rev_result = run_revision_engine(
                drawings_dir=self.drawings_dir,
                output_dir=self.output_dir,
            )

            revisions = rev_result.get("revisions_found", 0)
            return PhaseResult(
                phase="10_revision",
                phase_name="Revision Intelligence",
                success=True,
                message=f"Detected {revisions} revision changes",
                data=rev_result,
            )
        except ImportError:
            return PhaseResult(
                phase="10_revision",
                phase_name="Revision Intelligence",
                success=True,
                skipped=True,
                skip_reason="Module not available",
            )

    def _phase_doubt(self) -> PhaseResult:
        """Phase 11: Doubt engine."""
        try:
            from src.adapters.doubt_adapter import run_doubt_engine

            doubt_result = run_doubt_engine(
                drawings_dir=self.drawings_dir,
                scope_register=self.scope_register,
                output_dir=self.output_dir,
            )

            doubts = doubt_result.get("doubts_count", 0)
            return PhaseResult(
                phase="11_doubt",
                phase_name="Doubt Engine",
                success=True,
                message=f"Flagged {doubts} estimator doubts",
                data=doubt_result,
            )
        except ImportError:
            return PhaseResult(
                phase="11_doubt",
                phase_name="Doubt Engine",
                success=True,
                skipped=True,
                skip_reason="Module not available",
            )

    def _phase_rfi(self) -> PhaseResult:
        """Phase 12: RFI generation."""
        try:
            from src.adapters.rfi_adapter import run_rfi_generation

            rfi_result = run_rfi_generation(
                output_dir=self.output_dir,
                owner_inputs=self.metadata.get("owner_inputs", {}),
            )

            rfi_count = rfi_result.get("rfi_count", 0)
            self.rfis = rfi_result.get("rfis", [])

            return PhaseResult(
                phase="12_rfi",
                phase_name="RFI Engine",
                success=True,
                message=f"Generated {rfi_count} RFIs",
                data=rfi_result,
            )
        except ImportError as e:
            return PhaseResult(
                phase="12_rfi",
                phase_name="RFI Engine",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_owner_docs(self) -> PhaseResult:
        """Phase 13: Owner docs parsing."""
        try:
            from src.adapters.owner_docs_adapter import run_owner_docs_engine

            docs_result = run_owner_docs_engine(
                owner_docs_dir=self.owner_docs_dir,
                output_dir=self.output_dir,
            )

            docs_parsed = docs_result.get("documents_parsed", 0)
            return PhaseResult(
                phase="13_owner_docs",
                phase_name="Owner Docs Parsing",
                success=True,
                message=f"Parsed {docs_parsed} owner documents",
                data=docs_result,
            )
        except ImportError:
            return PhaseResult(
                phase="13_owner_docs",
                phase_name="Owner Docs Parsing",
                success=True,
                skipped=True,
                skip_reason="No owner docs or module not available",
            )

    def _phase_alignment(self) -> PhaseResult:
        """Phase 14: Owner BOQ alignment."""
        try:
            from src.adapters.alignment_adapter import run_alignment_engine

            align_result = run_alignment_engine(
                boq_items=self.boq_items,
                owner_docs_dir=self.owner_docs_dir,
                output_dir=self.output_dir,
            )

            alignment_score = align_result.get("alignment_score", 0)
            return PhaseResult(
                phase="14_alignment",
                phase_name="Owner BOQ Alignment",
                success=True,
                message=f"Alignment score: {alignment_score:.0%}",
                data=align_result,
            )
        except ImportError:
            return PhaseResult(
                phase="14_alignment",
                phase_name="Owner BOQ Alignment",
                success=True,
                skipped=True,
                skip_reason="Module not available",
            )

    def _phase_pricing(self) -> PhaseResult:
        """Phase 16: Pricing + scenarios.

        EVIDENCE-FIRST: Only runs if measurement gate passed OR allow_inferred_pricing=True.
        """
        # Check if we're allowed to produce pricing
        if not self.can_produce_pricing:
            # Write a skip note to pricing file
            pricing_dir = self.output_dir / "pricing"
            pricing_dir.mkdir(parents=True, exist_ok=True)

            skip_note = (
                "Pricing skipped: no reliable rates / measurement not verified.\n\n"
                "Reasons:\n"
            )
            if self.measurement_gate_result:
                for blocker in self.measurement_gate_result.blockers:
                    skip_note += f"  - {blocker}\n"
            skip_note += "\nTo enable pricing with inferred quantities, run with --allow_inferred_pricing\n"

            with open(pricing_dir / "pricing_skipped.txt", "w") as f:
                f.write(skip_note)

            # Also create empty estimate_priced.csv with header only
            import csv
            with open(pricing_dir / "estimate_priced.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "description", "quantity", "unit", "package", "rate", "amount", "status"])
                writer.writerow(["PRICING_SKIPPED", "See pricing_skipped.txt", 0, "", "", 0, 0, "SKIPPED"])

            return PhaseResult(
                phase="16_pricing",
                phase_name="Pricing + Scenarios",
                success=True,
                skipped=True,
                skip_reason="Measurement gate failed - no reliable measurements for pricing",
            )

        # Pricing is allowed - check if we should use only measured items
        items_to_price = self.boq_items
        pricing_note = ""

        if not self.allow_inferred_pricing and self.provenance_tracker:
            # Only price measured items
            measured_ids = set(self.provenance_tracker.get_measured_items().keys())
            items_to_price = [
                item for item in self.boq_items
                if item.get("item_id") in measured_ids or item.get("_provenance_obj", {})
            ]
            # Filter to only measured
            items_to_price = [
                item for item in self.boq_items
                if item.get("provenance", {}).get("is_measured", False)
            ]
            pricing_note = f" (measured only: {len(items_to_price)}/{len(self.boq_items)})"

        try:
            from src.adapters.pricing_adapter import run_pricing_engine

            pricing_result = run_pricing_engine(
                boq_items=items_to_price,
                output_dir=self.output_dir,
            )

            total_value = pricing_result.get("total_value", 0)
            return PhaseResult(
                phase="16_pricing",
                phase_name="Pricing + Scenarios",
                success=True,
                message=f"Total estimate: ₹{total_value/100000:.1f}L{pricing_note}",
                data=pricing_result,
            )
        except ImportError:
            return self._fallback_pricing(items_to_price, pricing_note)

    def _fallback_pricing(self, items_to_price: List[Dict] = None, pricing_note: str = "") -> PhaseResult:
        """Fallback pricing with evidence-first constraints."""
        import csv

        if items_to_price is None:
            items_to_price = self.boq_items

        # Simple pricing with assumed rates
        rates = {
            "flooring": 800,      # per sqm
            "wall_finishes": 200, # per sqm
            "waterproofing": 400, # per sqm
            "doors_windows": 15000, # per no
            "rcc_structural": 500,
            "masonry": 300,
        }

        priced_items = []
        total = 0
        measured_total = 0
        inferred_total = 0

        for item in items_to_price:
            pkg = item.get("package", "general")
            qty = item.get("quantity", 0)
            rate = rates.get(pkg, 500)
            amount = qty * rate
            total += amount

            # Track measured vs inferred
            is_measured = item.get("provenance", {}).get("is_measured", False)
            if is_measured:
                measured_total += amount
            else:
                inferred_total += amount

            priced_items.append({
                **item,
                "rate": rate,
                "amount": amount,
                "qty_status": "MEASURED" if is_measured else "INFERRED/TBD",
            })

        # Write priced estimate with provenance columns
        pricing_dir = self.output_dir / "pricing"
        pricing_dir.mkdir(parents=True, exist_ok=True)
        pricing_path = pricing_dir / "estimate_priced.csv"

        fieldnames = ["item_id", "description", "quantity", "unit", "package", "rate", "amount", "qty_status", "method", "confidence"]
        with open(pricing_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in priced_items:
                row = {k: item.get(k, "") for k in fieldnames}
                prov = item.get("provenance", {})
                row["method"] = prov.get("method", "unknown")
                row["confidence"] = f"{prov.get('confidence', 0):.2f}"
                writer.writerow(row)

        # Write pricing summary with measured/inferred breakdown
        summary_note = f"Measured: ₹{measured_total/100000:.1f}L, Inferred/TBD: ₹{inferred_total/100000:.1f}L"

        return PhaseResult(
            phase="16_pricing",
            phase_name="Pricing + Scenarios",
            success=True,
            message=f"Total estimate: ₹{total/100000:.1f}L (fallback rates){pricing_note} [{summary_note}]",
            data={"total_value": total, "measured_value": measured_total, "inferred_value": inferred_total},
        )

    def _phase_quote_leveling(self) -> PhaseResult:
        """Phase 16: Quote leveling."""
        try:
            from src.adapters.quotes_adapter import run_quote_leveling

            quotes_result = run_quote_leveling(
                quotes_dir=self.quotes_dir,
                boq_items=self.boq_items,
                output_dir=self.output_dir,
            )

            quotes_leveled = quotes_result.get("quotes_leveled", 0)
            return PhaseResult(
                phase="16_quotes",
                phase_name="Quote Leveling",
                success=True,
                message=f"Leveled {quotes_leveled} subcontractor quotes",
                data=quotes_result,
            )
        except ImportError:
            return PhaseResult(
                phase="16_quotes",
                phase_name="Quote Leveling",
                success=True,
                skipped=True,
                skip_reason="No quotes or module not available",
            )

    def _phase_prelims(self) -> PhaseResult:
        """Phase 17: Prelims generator."""
        try:
            from src.adapters.prelims_adapter import run_prelims_engine

            prelims_result = run_prelims_engine(
                boq_items=self.boq_items,
                project_metadata=self.metadata,
                output_dir=self.output_dir,
            )

            prelims_value = prelims_result.get("total_prelims", 0)
            return PhaseResult(
                phase="17_prelims",
                phase_name="Prelims Generator",
                success=True,
                message=f"Prelims: ₹{prelims_value/100000:.1f}L",
                data=prelims_result,
            )
        except ImportError:
            return self._fallback_prelims()

    def _fallback_prelims(self) -> PhaseResult:
        """Fallback prelims generation."""
        import csv

        # Standard prelims as % of construction value
        total_boq = sum(b.get("quantity", 0) * 500 for b in self.boq_items)  # Rough estimate

        prelims_items = [
            {"item": "Site establishment", "percentage": 2.0, "amount": total_boq * 0.02},
            {"item": "Temporary facilities", "percentage": 1.5, "amount": total_boq * 0.015},
            {"item": "Site security", "percentage": 0.5, "amount": total_boq * 0.005},
            {"item": "Site supervision", "percentage": 3.0, "amount": total_boq * 0.03},
            {"item": "Insurance", "percentage": 0.5, "amount": total_boq * 0.005},
            {"item": "Testing & QA", "percentage": 1.0, "amount": total_boq * 0.01},
        ]

        total_prelims = sum(p["amount"] for p in prelims_items)

        # Write prelims BOQ
        prelims_path = self.output_dir / "prelims" / "prelims_boq.csv"
        with open(prelims_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["item", "percentage", "amount"])
            writer.writeheader()
            writer.writerows(prelims_items)

        return PhaseResult(
            phase="17_prelims",
            phase_name="Prelims Generator",
            success=True,
            message=f"Prelims: ₹{total_prelims/100000:.1f}L (fallback)",
            data={"total_prelims": total_prelims},
        )

    def _phase_bid_gate(self) -> PhaseResult:
        """Phase 18: Bid gate assessment."""
        try:
            from src.adapters.bid_gate_adapter import run_bid_gate

            gate_result = run_bid_gate(
                output_dir=self.output_dir,
                project_metadata=self.metadata,
            )

            status = gate_result.get("status", "UNKNOWN")
            score = gate_result.get("score", 0)

            return PhaseResult(
                phase="18_bid_gate",
                phase_name="Bid Gate",
                success=True,
                message=f"Gate: {status} (score: {score}/100)",
                data=gate_result,
            )
        except ImportError as e:
            return PhaseResult(
                phase="18_bid_gate",
                phase_name="Bid Gate",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_bid_docs(self) -> PhaseResult:
        """Phase 19: Bid docs generation (clarifications, exclusions, assumptions)."""
        try:
            from src.adapters.bid_docs_adapter import run_bid_docs_generation

            docs_result = run_bid_docs_generation(
                boq_items=self.boq_items,
                scope_register=self.scope_register,
                rfis=self.rfis,
                output_dir=self.output_dir,
            )

            docs_generated = docs_result.get("documents_generated", 0)
            return PhaseResult(
                phase="19_bid_docs",
                phase_name="Bid Docs Generation",
                success=True,
                message=f"Generated {docs_generated} bid documents",
                data=docs_result,
            )
        except ImportError:
            return self._fallback_bid_docs()

    def _fallback_bid_docs(self) -> PhaseResult:
        """Fallback bid docs generation."""
        # Clarifications letter
        clarif_path = self.output_dir / "bid_book" / "clarifications_letter.md"
        with open(clarif_path, "w") as f:
            f.write("# Clarifications Letter\n\n")
            f.write("## Request for Information\n\n")
            if self.rfis:
                for rfi in self.rfis[:10]:
                    f.write(f"- {rfi.get('title', 'RFI')}\n")
            else:
                f.write("No RFIs raised.\n")

        # Exclusions
        excl_path = self.output_dir / "bid_book" / "exclusions.md"
        with open(excl_path, "w") as f:
            f.write("# Exclusions\n\n")
            f.write("The following items are EXCLUDED from this bid:\n\n")
            exclusions = [
                "Architect/consultant fees",
                "Statutory approvals and permits",
                "Soil testing and investigation",
                "Furniture and furnishings",
                "Landscaping beyond basic",
                "HVAC system (provision only)",
                "Generator set",
                "Solar system",
                "Water treatment plant",
            ]
            for excl in exclusions:
                f.write(f"- {excl}\n")

        # Assumptions
        assum_path = self.output_dir / "bid_book" / "assumptions.md"
        with open(assum_path, "w") as f:
            f.write("# Assumptions\n\n")
            f.write("This bid is based on the following assumptions:\n\n")
            assumptions = [
                "8-hour working day, 26 days/month",
                "Water and electricity available at site",
                "Clear site access for material delivery",
                "No rock excavation required",
                "Standard floor height 3.0m",
                "All dimensions as per architectural drawings",
                "Rates valid for 90 days from bid date",
            ]
            for assum in assumptions:
                f.write(f"- {assum}\n")

        return PhaseResult(
            phase="19_bid_docs",
            phase_name="Bid Docs Generation",
            success=True,
            message="Generated 3 bid documents (fallback)",
            data={"documents_generated": 3},
        )

    def _phase_packages(self) -> PhaseResult:
        """Phase 20: Package exports + RFQ sheets."""
        try:
            from src.adapters.packages_adapter import run_package_splitter

            packages_result = run_package_splitter(
                output_dir=self.output_dir,
            )

            packages_count = len(packages_result.get("packages", {}))
            files_generated = len(packages_result.get("files_generated", []))

            return PhaseResult(
                phase="20_packages",
                phase_name="Package Exports",
                success=True,
                message=f"Split into {packages_count} packages ({files_generated} files)",
                data=packages_result,
            )
        except ImportError as e:
            return PhaseResult(
                phase="20_packages",
                phase_name="Package Exports",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_bid_book(self) -> PhaseResult:
        """Phase 21: Bid book export."""
        try:
            from src.adapters.bid_book_adapter import run_bidbook_export

            bid_book_result = run_bidbook_export(
                output_dir=self.output_dir,
                project_metadata=self.metadata,
            )

            files_generated = len(bid_book_result.get("files_generated", []))

            return PhaseResult(
                phase="21_bid_book",
                phase_name="Bid Book Export",
                success=True,
                message=f"Generated {files_generated} bid book files",
                data=bid_book_result,
            )
        except ImportError as e:
            return PhaseResult(
                phase="21_bid_book",
                phase_name="Bid Book Export",
                success=True,
                skipped=True,
                skip_reason=str(e),
            )

    def _phase_verify(self) -> PhaseResult:
        """Phase 24: Output verification (evidence-first mode)."""
        verifications = self._verify_outputs()

        missing = [v for v in verifications if not v.exists]
        present = [v for v in verifications if v.exists]

        # Add evidence-first status to message
        gate_status = "N/A"
        if self.measurement_gate_result:
            gate_status = self.measurement_gate_result.status.value

        return PhaseResult(
            phase="24_verify",
            phase_name="Output Verification",
            success=len(missing) == 0,
            message=f"{len(present)}/{len(verifications)} required outputs present | Measurement gate: {gate_status}",
            data={
                "present": len(present),
                "missing": len(missing),
                "missing_files": [v.required_file for v in missing],
                "measurement_gate_status": gate_status,
                "can_produce_pricing": self.can_produce_pricing,
            },
            error=f"Missing: {', '.join(v.required_file for v in missing)}" if missing else None,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _load_extracted_data(self) -> None:
        """Load rooms and openings from output directory.

        Checks multiple locations in priority order:
        1. combined/all_rooms.json (from multipage_extractor)
        2. boq/rooms.json (legacy location)
        """
        # Priority 1: combined/ directory (multipage_extractor output)
        combined_rooms_path = self.output_dir / "combined" / "all_rooms.json"
        combined_openings_path = self.output_dir / "combined" / "all_openings.json"

        # Priority 2: boq/ directory (legacy)
        boq_rooms_path = self.output_dir / "boq" / "rooms.json"
        boq_openings_path = self.output_dir / "boq" / "openings.json"

        # Load rooms
        if combined_rooms_path.exists():
            with open(combined_rooms_path) as f:
                data = json.load(f)
                self.rooms = data.get("rooms", [])
                logger.info(f"Loaded {len(self.rooms)} rooms from combined/all_rooms.json")
        elif boq_rooms_path.exists():
            with open(boq_rooms_path) as f:
                data = json.load(f)
                self.rooms = data.get("rooms", [])
                logger.info(f"Loaded {len(self.rooms)} rooms from boq/rooms.json")

        # Load openings
        if combined_openings_path.exists():
            with open(combined_openings_path) as f:
                data = json.load(f)
                self.openings = data.get("openings", [])
                logger.info(f"Loaded {len(self.openings)} openings from combined/all_openings.json")
        elif boq_openings_path.exists():
            with open(boq_openings_path) as f:
                data = json.load(f)
                self.openings = data.get("openings", [])

    def _write_scope_register_csv(self, scope_result: Dict) -> None:
        """Write scope register CSV."""
        import csv

        scope_path = self.output_dir / "scope" / "scope_register.csv"
        packages = scope_result.get("packages", [])

        if packages:
            with open(scope_path, "w", newline="") as f:
                fieldnames = list(packages[0].keys()) if packages else ["package", "status"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(packages)

    def _verify_outputs(self) -> List[OutputVerification]:
        """Verify required outputs exist (including evidence-first outputs)."""
        verifications = []

        # Check required files
        for rel_path, description in REQUIRED_OUTPUTS.items():
            full_path = self.output_dir / rel_path
            exists = full_path.exists()
            size = full_path.stat().st_size if exists else 0

            verifications.append(OutputVerification(
                required_file=rel_path,
                exists=exists,
                description=description,
                size_bytes=size,
                reason=None if exists else "File not generated",
            ))

        # Check evidence-first required outputs
        for rel_path, description in EVIDENCE_REQUIRED_OUTPUTS.items():
            full_path = self.output_dir / rel_path
            exists = full_path.exists()
            size = full_path.stat().st_size if exists else 0

            verifications.append(OutputVerification(
                required_file=rel_path,
                exists=exists,
                description=description,
                size_bytes=size,
                reason=None if exists else "File not generated (evidence-first)",
            ))

        # Check packages (at least 6)
        packages_dir = self.output_dir / "packages"
        for pkg_name in REQUIRED_PACKAGES:
            pkg_file = packages_dir / f"pkg_{pkg_name}.csv"
            exists = pkg_file.exists()
            size = pkg_file.stat().st_size if exists else 0

            verifications.append(OutputVerification(
                required_file=f"packages/pkg_{pkg_name}.csv",
                exists=exists,
                description=f"Package: {pkg_name}",
                size_bytes=size,
                reason=None if exists else "Package not generated",
            ))

        return verifications

    def _generate_summary(self, total_duration: float) -> None:
        """Generate run summary."""
        successful = sum(1 for r in self.results if r.success and not r.skipped)
        failed = sum(1 for r in self.results if not r.success and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)

        self.summary = {
            "project_id": self.project_id,
            "mode": self.mode,
            "total_phases": len(self.results),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "duration_sec": round(total_duration, 1),
            "output_dir": str(self.output_dir),
            "phases": [
                {
                    "phase": r.phase,
                    "phase_name": r.phase_name,
                    "success": r.success,
                    "skipped": r.skipped,
                    "message": r.message,
                    "error": r.error,
                    "skip_reason": r.skip_reason,
                    "stack_trace": r.stack_trace,
                    "duration_sec": r.duration_sec,
                }
                for r in self.results
            ],
        }

    def _print_summary(self, output_verification: List[OutputVerification]) -> None:
        """Print terminal summary."""
        print()
        print("=" * 70)
        print("RUN SUMMARY")
        print("=" * 70)

        s = self.summary

        # Phase status
        for phase in s["phases"]:
            if phase["skipped"]:
                icon = "⏭️"
                msg = phase.get("skip_reason", "skipped")
            elif phase["success"]:
                icon = "✅"
                msg = phase.get("message") or "completed"
            else:
                icon = "❌"
                msg = phase.get("error") or "failed"
            print(f"  {icon} {phase['phase']}: {msg[:50]}")

        print()
        print("-" * 70)

        # Key metrics
        extract_result = next((r for r in self.results if r.phase == "03_extract"), None)
        if extract_result and extract_result.data:
            print(f"  📄 Pages processed: {extract_result.data.get('pages_processed', 'N/A')}")
            print(f"  🏠 Rooms found: {extract_result.data.get('total_rooms', 'N/A')}")

        takeoff_result = next((r for r in self.results if r.phase == "05_takeoff"), None)
        if takeoff_result and takeoff_result.data:
            print(f"  📦 BOQ items: {takeoff_result.data.get('items_count', 'N/A')}")

        pricing_result = next((r for r in self.results if r.phase == "15_pricing"), None)
        if pricing_result and pricing_result.data:
            val = pricing_result.data.get('total_value', 0)
            print(f"  💰 Estimate: ₹{val/100000:.1f}L")

        gate_result = next((r for r in self.results if r.phase == "18_bid_gate"), None)
        if gate_result and gate_result.data:
            print(f"  🚦 Bid gate: {gate_result.data.get('status', 'N/A')}")

        rfi_result = next((r for r in self.results if r.phase == "12_rfi"), None)
        if rfi_result and rfi_result.data:
            print(f"  ❓ RFIs: {rfi_result.data.get('rfi_count', 'N/A')}")

        print()
        print(f"  ⏱️  Total duration: {s['duration_sec']:.1f}s")
        print(f"  📁 Output: {s['output_dir']}")
        print()

        # Output verification (mode=full only)
        if output_verification:
            print("=" * 70)
            print("OUTPUT VERIFICATION")
            print("=" * 70)

            present = [v for v in output_verification if v.exists]
            missing = [v for v in output_verification if not v.exists]

            print(f"  ✅ Present: {len(present)}/{len(output_verification)}")

            if missing:
                print(f"  ❌ Missing: {len(missing)}")
                print()
                print("  FAILED OUTPUTS:")
                for v in missing:
                    print(f"    ✗ {v.required_file} - {v.reason}")
            else:
                print("  All required outputs generated!")

        print("=" * 70)

    def _save_run_metadata(self, start_time: datetime, duration: float,
                          output_verification: List[OutputVerification]) -> None:
        """Save run metadata to output directory."""

        # Generate unique run_id (timestamp + random suffix)
        run_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{run_suffix}"

        # Calculate completeness score
        if output_verification:
            present = sum(1 for v in output_verification if v.exists)
            completeness = present / len(output_verification)
        else:
            completeness = 0

        # Extract aggregate counts from phase data
        aggregates = self._extract_run_aggregates()

        metadata = {
            "run_id": run_id,
            "project_id": self.project_id,
            "mode": self.mode,
            "profile": self.profile,
            "rules_version": self.rules_version,
            "start_time": start_time.isoformat(),
            "duration_sec": round(duration, 1),
            # Aggregate counts (single source of truth for PDF)
            "aggregates": aggregates,
            "phases_run": [p["phase"] for p in self.summary.get("phases", []) if not p.get("skipped")],
            "phases_skipped": [p["phase"] for p in self.summary.get("phases", []) if p.get("skipped")],
            "phases_failed": [p["phase"] for p in self.summary.get("phases", []) if not p.get("success") and not p.get("skipped")],
            "output_completeness_score": round(completeness, 2),
            "output_verification": [
                {
                    "file": v.required_file,
                    "exists": v.exists,
                    "size_bytes": v.size_bytes,
                    "reason": v.reason,
                }
                for v in output_verification
            ] if output_verification else [],
            "summary": self.summary,
        }

        # Save JSON
        meta_path = self.output_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save summary markdown
        summary_path = self.output_dir / "summary.md"
        self._write_summary_md(summary_path)

    def _extract_run_aggregates(self) -> Dict[str, Any]:
        """Extract aggregate counts from phase results for PDF report."""
        aggregates = {
            "pages_processed": 0,
            "pages_routed": 0,
            "candidate_pages": 0,
            "rooms_found": 0,
            "openings_found": 0,
            "dimensions_found": 0,
            "boq_items_total": 0,
            "boq_measured": 0,
            "boq_counted": 0,
            "boq_inferred": 0,
            "coverage_percent": 0.0,
            "missing_scope_count": 0,
            "rfis_generated": 0,
            "scales_detected": [],
            "drawing_types": {},
            "bid_recommendation": "N/A",
            "bid_score": 0,
            "input_files": [],
        }

        # Parse from phase results
        for phase in self.summary.get("phases", []):
            phase_id = phase.get("phase", "")
            message = phase.get("message", "")
            data = phase.get("data", {}) or {}

            if phase_id == "01_index":
                # Extract from "Found 78 pages from 1 files"
                import re
                match = re.search(r'Found (\d+) pages from (\d+) files', message)
                if match:
                    aggregates["pages_processed"] = int(match.group(1))
                if data.get("input_files"):
                    aggregates["input_files"] = data.get("input_files", [])

            elif phase_id == "02_route":
                # Extract scales and types from message
                import re
                match = re.search(r'Routed (\d+) pages: (\d+) candidates', message)
                if match:
                    aggregates["pages_routed"] = int(match.group(1))
                    aggregates["candidate_pages"] = int(match.group(2))
                # Extract scales
                scales_match = re.search(r"scales=\[(.*?)\]", message)
                if scales_match:
                    scales_str = scales_match.group(1)
                    aggregates["scales_detected"] = [s.strip().strip("'\"") for s in scales_str.split(",")]
                # Extract drawing types
                types_match = re.search(r"types=\{(.*?)\}", message)
                if types_match:
                    types_str = types_match.group(1)
                    for pair in types_str.split(","):
                        if ":" in pair:
                            k, v = pair.split(":")
                            aggregates["drawing_types"][k.strip().strip("'\"").strip()] = int(v.strip())

            elif phase_id == "03_extract":
                # Extract from "Processed 71 pages, 159 rooms, 358 openings"
                import re
                match = re.search(r'(\d+) pages.*?(\d+) rooms.*?(\d+) openings', message)
                if match:
                    aggregates["candidate_pages"] = int(match.group(1))
                    aggregates["rooms_found"] = int(match.group(2))
                    aggregates["openings_found"] = int(match.group(3))

            elif phase_id == "05_takeoff":
                # Extract from "Generated 856 BOQ items"
                import re
                match = re.search(r'Generated (\d+) BOQ items', message)
                if match:
                    aggregates["boq_items_total"] = int(match.group(1))

            elif phase_id == "06a_provenance":
                # Extract from "Measured: 358, Counted: 42, Inferred: 498 (42% coverage)"
                import re
                match = re.search(r'Measured: (\d+), Counted: (\d+), Inferred: (\d+) \((\d+)% coverage\)', message)
                if match:
                    aggregates["boq_measured"] = int(match.group(1))
                    aggregates["boq_counted"] = int(match.group(2))
                    aggregates["boq_inferred"] = int(match.group(3))
                    aggregates["coverage_percent"] = float(match.group(4))
                else:
                    # Fallback to old format without counted
                    match = re.search(r'Measured: (\d+), Inferred: (\d+) \((\d+)% coverage\)', message)
                    if match:
                        aggregates["boq_measured"] = int(match.group(1))
                        aggregates["boq_inferred"] = int(match.group(2))
                        aggregates["coverage_percent"] = float(match.group(3))

            elif phase_id == "E1_estimator":
                # Extract from "Measured: 33 | Inferred: 185 | (15% coverage) | Missing: 1476 | Bid: NO-GO"
                if data.get("total_measured") is not None:
                    aggregates["boq_measured"] = data.get("total_measured", 0)
                if data.get("total_inferred") is not None:
                    aggregates["boq_inferred"] = data.get("total_inferred", 0)
                if data.get("coverage_percent") is not None:
                    aggregates["coverage_percent"] = data.get("coverage_percent", 0)
                if data.get("missing_scope_count") is not None:
                    aggregates["missing_scope_count"] = data.get("missing_scope_count", 0)
                if data.get("rfis_generated") is not None:
                    aggregates["rfis_generated"] = data.get("rfis_generated", 0)
                if data.get("bid_recommendation"):
                    aggregates["bid_recommendation"] = data.get("bid_recommendation", "N/A")
                if data.get("bid_score") is not None:
                    aggregates["bid_score"] = data.get("bid_score", 0)

            elif phase_id == "12_rfi":
                # Extract from "Generated 1 RFIs"
                import re
                match = re.search(r'Generated (\d+) RFIs', message)
                if match:
                    aggregates["rfis_generated"] = max(aggregates["rfis_generated"], int(match.group(1)))

            elif phase_id == "18_bid_gate":
                # Extract from "Gate: NO-GO (score: 20/100)"
                import re
                match = re.search(r'Gate: (\S+) \(score: (\d+)/100\)', message)
                if match:
                    if aggregates["bid_recommendation"] == "N/A":
                        aggregates["bid_recommendation"] = match.group(1)
                    aggregates["bid_score"] = int(match.group(2))

        return aggregates

    def _update_run_metadata_verification(self, output_verification: List[OutputVerification]) -> None:
        """Update run_metadata.json with output verification results."""
        meta_path = self.output_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

            # Calculate completeness
            present = sum(1 for v in output_verification if v.exists)
            completeness = present / len(output_verification) if output_verification else 0

            metadata["output_completeness_score"] = round(completeness, 2)
            metadata["output_verification"] = [
                {
                    "file": v.required_file,
                    "exists": v.exists,
                    "size_bytes": v.size_bytes,
                    "reason": v.reason,
                }
                for v in output_verification
            ]

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    def _write_summary_md(self, path: Path) -> None:
        """Write summary.md file."""
        s = self.summary

        with open(path, "w") as f:
            f.write(f"# Run Summary: {self.project_id}\n\n")
            f.write(f"**Mode**: {self.mode.upper()}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Duration**: {s['duration_sec']:.1f}s\n\n")

            f.write("## Phase Results\n\n")
            f.write("| Phase | Status | Message | Duration |\n")
            f.write("|-------|--------|---------|----------|\n")

            for phase in s["phases"]:
                if phase["skipped"]:
                    icon = "⏭️"
                elif phase["success"]:
                    icon = "✅"
                else:
                    icon = "❌"
                msg = (phase.get("message") or phase.get("error") or phase.get("skip_reason") or "")[:40]
                f.write(f"| {phase['phase']} | {icon} | {msg} | {phase['duration_sec']:.1f}s |\n")

            f.write("\n## Statistics\n\n")
            f.write(f"- Total phases: {s['total_phases']}\n")
            f.write(f"- Successful: {s['successful']}\n")
            f.write(f"- Failed: {s['failed']}\n")
            f.write(f"- Skipped: {s['skipped']}\n")

            f.write("\n## Output Files\n\n")
            f.write(f"- Output directory: `{s['output_dir']}`\n")
            f.write("- Key files:\n")
            f.write("  - `summary.md` - This file\n")
            f.write("  - `run_metadata.json` - Complete run metadata\n")
            f.write("  - `bid_gate_report.md` - Gate assessment\n")
            f.write("  - `rfi/rfi_log.md` - RFI list\n")
            f.write("  - `boq/boq_quantities.csv` - BOQ quantities\n")
            f.write("  - `pricing/estimate_priced.csv` - Priced estimate\n")
            f.write("  - `packages/` - Package exports\n")

            # Add MEP section if enabled
            if self.enable_mep and self.mep_takeoff:
                f.write("\n## MEP Takeoff\n\n")
                mep = self.mep_takeoff
                f.write(f"- **Total devices:** {mep.get('total_devices', 0)}\n")
                f.write(f"- **Measured (from drawing):** {mep.get('measured_count', 0)}\n")
                f.write(f"- **Inferred:** {mep.get('inferred_count', 0)}\n")
                f.write(f"- **RFIs required:** {mep.get('rfi_count', 0)}\n\n")

                # Device counts by category
                if "takeoff_lines" in mep:
                    by_system = {}
                    for line in mep["takeoff_lines"]:
                        sys = line.get("system", "Other")
                        qty = line.get("qty", 0)
                        by_system[sys] = by_system.get(sys, 0) + qty

                    if by_system:
                        f.write("### Devices by System\n\n")
                        f.write("| System | Count |\n")
                        f.write("|--------|-------|\n")
                        for sys, count in sorted(by_system.items()):
                            f.write(f"| {sys.replace('_', ' ').title()} | {count} |\n")

                # Connectivity summary
                conn = mep.get("connectivity", {})
                elec = conn.get("electrical", {})
                if elec.get("total_cable_m", 0) > 0:
                    f.write(f"\n### Connectivity\n\n")
                    f.write(f"- Total cable: ~{elec.get('total_cable_m', 0):.0f}m\n")
                    f.write(f"- Total runs: {elec.get('total_runs', 0)}\n")

                # RFIs needing response
                rfis = mep.get("rfis", [])
                if rfis:
                    f.write(f"\n### MEP RFIs ({len(rfis)} items)\n\n")
                    f.write("| ID | Device | Location | Missing |\n")
                    f.write("|----|--------|----------|--------|\n")
                    for rfi in rfis[:10]:  # Show first 10
                        f.write(f"| {rfi.get('rfi_id', '')} | {rfi.get('device_type', '')} | {rfi.get('location', '')} | {rfi.get('missing_field', '')} |\n")
                    if len(rfis) > 10:
                        f.write(f"\n*...and {len(rfis) - 10} more RFIs*\n")

                f.write("\n- MEP files:\n")
                f.write("  - `mep/mep_takeoff.csv` - Device takeoff\n")
                f.write("  - `mep/mep_takeoff.xlsx` - Excel export\n")
                f.write("  - `mep/devices.json` - Detected devices\n")
                f.write("  - `mep/connections.json` - Connectivity data\n")


def main():
    parser = argparse.ArgumentParser(
        description="XBOQ Full Project Runner - Evidence-First Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline on existing project
    python run_full_project.py --project_id villa_whitefield

    # Run with input directory (auto-copy drawings)
    python run_full_project.py --project_id villa --input_dir ~/Documents/villa_drawings/

    # Quick mode (extract + RFIs only)
    python run_full_project.py --project_id test --mode quick

    # Provide manual scale for measurement verification
    python run_full_project.py --project_id villa --scale 100

    # Allow pricing even when measurement gate fails (NOT RECOMMENDED)
    python run_full_project.py --project_id villa --allow_inferred_pricing

    # Resume previous run
    python run_full_project.py --project_id villa --resume

EVIDENCE-FIRST MODE:
    By default, pricing is ONLY produced when:
    1. Scale can be reliably determined (from dimension text or manual input)
    2. Sufficient geometry is detected (walls, room polygons)
    3. Measurement gate passes

    If measurement fails, you get:
    - scope_register.csv (what we know)
    - rfi_log.md (what we need to ask)
    - boq_inferred.csv (marked as TBD/ALLOWANCE)
    - NO priced estimate (unless --allow_inferred_pricing)
        """
    )

    parser.add_argument(
        "--project_id", "-p",
        required=True,
        help="Project ID (creates data/projects/<id>/ if needed)"
    )

    parser.add_argument(
        "--input_dir", "-i",
        help="Input directory containing drawings (PDFs/images). Files will be copied to project."
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["full", "quick"],
        default="full",
        help="Pipeline mode: full (all 24+ phases) or quick (extract + RFIs only). Default: full"
    )

    parser.add_argument(
        "--profile",
        choices=["quick", "typical", "detailed"],
        default="typical",
        help="Processing profile (default: typical)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (use cached results)"
    )

    parser.add_argument(
        "--rules",
        help="Rules version to use (e.g., v1)"
    )

    # Evidence-first mode options
    parser.add_argument(
        "--allow_inferred_pricing",
        action="store_true",
        help="Allow pricing even when measurement gate fails (NOT RECOMMENDED). "
             "Produces estimates using TBD/allowance quantities."
    )

    parser.add_argument(
        "--scale",
        type=float,
        help="Manual scale value (e.g., 100 for 1:100). "
             "Provides reliable scale for measurement verification."
    )

    # MEP options
    parser.add_argument(
        "--enable-mep",
        action="store_true",
        help="Enable MEP (Mechanical, Electrical, Plumbing) device detection and takeoff. "
             "Adds phases: device detection, connectivity, systems grouping, MEP takeoff export."
    )

    # Estimator assumption overrides
    parser.add_argument(
        "--assume_wall_height",
        type=float,
        default=3.0,
        help="Assumed wall height in meters (default: 3.0m)"
    )

    parser.add_argument(
        "--assume_door_height",
        type=float,
        default=2.1,
        help="Assumed door height in meters (default: 2.1m)"
    )

    parser.add_argument(
        "--assume_plaster_both_sides",
        action="store_true",
        default=True,
        help="Assume plaster on both sides of walls (default: True)"
    )

    parser.add_argument(
        "--assume_floor_finish_all_rooms",
        action="store_true",
        default=True,
        help="Assume floor finish in all rooms (default: True)"
    )

    # Estimator workflow options
    parser.add_argument(
        "--apply_overrides",
        action="store_true",
        help="Apply overrides from estimator_inputs.yaml and bid_ready_boq.xlsx on re-run"
    )

    # Demo mode for YC video
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Run in DEMO MODE: Only execute stable phases needed for demo "
             "(input gate, routing, room detection, opening count, RFIs, bid gate). "
             "Skips pricing, quote leveling, advanced BOQ unless verified outputs exist."
    )

    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Exit immediately on first phase failure (default: continue with warnings)"
    )

    # Legacy argument support
    parser.add_argument("--project", dest="project_id_legacy", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle legacy --project argument
    project_id = args.project_id or args.project_id_legacy
    if not project_id:
        parser.error("--project_id is required")

    runner = FullProjectRunner(
        project_id=project_id,
        input_dir=args.input_dir,
        mode=args.mode,
        profile=args.profile,
        resume=args.resume,
        rules_version=args.rules,
        allow_inferred_pricing=args.allow_inferred_pricing,
        manual_scale=args.scale,
        enable_mep=args.enable_mep,
        # Estimator assumptions
        assume_wall_height=args.assume_wall_height,
        assume_door_height=args.assume_door_height,
        assume_plaster_both_sides=args.assume_plaster_both_sides,
        assume_floor_finish_all_rooms=args.assume_floor_finish_all_rooms,
        # Estimator workflow
        apply_overrides=args.apply_overrides,
        # Demo mode
        demo_mode=args.demo_mode,
        fail_fast=args.fail_fast,
    )

    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
