"""
Evidence Extraction - Extract notes, specs, legends from drawings.

Collects and structures evidence from:
- General notes pages
- Legends
- Specification text
- Schedule tables
- Cross-references (Detail 3/S-402, Ref. A-105, IS codes)

Outputs: out/<project_id>/scope/evidence.json
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """A single evidence item extracted from drawings."""
    evidence_id: str
    evidence_type: str  # notes, legend, spec, schedule, detail_ref, code_ref
    source_file: str
    source_page: int
    sheet_id: Optional[str] = None
    snippet: str = ""
    full_text: str = ""
    confidence: float = 1.0
    keywords_matched: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "sheet_id": self.sheet_id,
            "snippet": self.snippet[:500],  # Truncate for storage
            "confidence": round(self.confidence, 3),
            "keywords_matched": self.keywords_matched,
            "cross_references": self.cross_references,
        }


@dataclass
class EvidenceStore:
    """Container for all extracted evidence."""
    project_id: str
    items: List[EvidenceItem] = field(default_factory=list)
    by_type: Dict[str, List[EvidenceItem]] = field(default_factory=dict)
    by_page: Dict[str, List[EvidenceItem]] = field(default_factory=dict)
    code_references: List[Dict[str, str]] = field(default_factory=list)
    detail_references: List[Dict[str, str]] = field(default_factory=list)

    def add(self, item: EvidenceItem) -> None:
        """Add evidence item and update indexes."""
        self.items.append(item)

        if item.evidence_type not in self.by_type:
            self.by_type[item.evidence_type] = []
        self.by_type[item.evidence_type].append(item)

        page_key = f"{Path(item.source_file).stem}_p{item.source_page + 1}"
        if page_key not in self.by_page:
            self.by_page[page_key] = []
        self.by_page[page_key].append(item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "total_items": len(self.items),
            "by_type_count": {t: len(items) for t, items in self.by_type.items()},
            "code_references": self.code_references,
            "detail_references": self.detail_references,
            "items": [item.to_dict() for item in self.items],
        }

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved {len(self.items)} evidence items to: {path}")


class EvidenceExtractor:
    """
    Extracts evidence from indexed pages.

    Evidence types:
    - notes: General notes, specifications text
    - legend: Finish legends, symbol legends
    - spec: Specification references (IS codes, CPWD specs)
    - schedule: Schedule data (door/window/finish schedules)
    - detail_ref: Cross-references to details
    - code_ref: Indian Standard/code references
    """

    # Patterns for cross-reference detection
    DETAIL_REF_PATTERNS = [
        r"(?:Detail|DET|DTL)\s*[-:]?\s*(\d+)\s*/\s*([A-Z]+-?\d+)",  # Detail 3/S-402
        r"(?:Ref|REF|See)\s*[.:]?\s*([A-Z]+-?\d+)",  # Ref. A-105
        r"(?:As per|Refer)\s+(?:sheet|dwg|drawing)\s*[.:]?\s*([A-Z]+-?\d+)",
        r"@\s*([A-Z]+-?\d+)",  # @A-105
    ]

    CODE_REF_PATTERNS = [
        r"(?:IS|I\.S\.)\s*[-:]?\s*(\d+(?:[-:]\d+)?)",  # IS 456, IS 456:2000
        r"(?:CPWD|C\.P\.W\.D\.)\s*(?:spec|specification)?",
        r"(?:NBC|N\.B\.C\.)\s*\d*",  # National Building Code
        r"(?:IRC|I\.R\.C\.)\s*[-:]?\s*(\d+)",  # IRC codes
        r"(?:BIS|B\.I\.S\.)\s*\d*",
        r"(?:SP|S\.P\.)\s*[-:]?\s*(\d+)",  # SP codes
    ]

    # Keywords indicating notes/specs content
    NOTES_KEYWORDS = [
        "note", "general note", "specification", "spec",
        "material", "workmanship", "standard", "quality",
        "requirement", "shall be", "should be", "must be",
        "as per", "in accordance", "complying with",
    ]

    LEGEND_KEYWORDS = [
        "legend", "finish legend", "symbol", "abbreviation",
        "floor finish", "wall finish", "ceiling finish",
        "material code", "hatch", "pattern",
    ]

    # India-specific keywords for scope detection
    SCOPE_KEYWORDS = {
        "rcc": ["RCC", "reinforced concrete", "M20", "M25", "M30", "M35"],
        "steel": ["TMT", "Fe500", "Fe500D", "rebar", "reinforcement", "BBS"],
        "masonry": ["brick", "brickwork", "AAC", "block", "230mm", "115mm"],
        "waterproofing": ["waterproofing", "APP", "membrane", "coba", "sunken"],
        "plaster": ["plaster", "putty", "neeru", "punning", "gypsum", "POP"],
        "paint": ["paint", "primer", "emulsion", "distemper", "enamel", "apex"],
        "flooring": ["vitrified", "granite", "marble", "kota", "IPS", "tile"],
        "plumbing": ["CPVC", "UPVC", "SWR", "plumbing", "sanitary", "GI pipe"],
        "electrical": ["wiring", "conduit", "MCB", "DB", "earthing", "FRLS"],
        "doors": ["door", "flush", "frame", "chaukhat", "hardware"],
        "windows": ["window", "UPVC", "aluminum", "ventilator"],
        "fire": ["fire", "sprinkler", "hydrant", "extinguisher", "alarm"],
        "hvac": ["AC", "HVAC", "ventilation", "exhaust", "VRV"],
    }

    def __init__(self):
        self._evidence_counter = 0

    def _generate_id(self, prefix: str = "EV") -> str:
        """Generate unique evidence ID."""
        self._evidence_counter += 1
        return f"{prefix}_{self._evidence_counter:05d}"

    def extract_from_project(
        self,
        project_id: str,
        page_index: List[Dict[str, Any]],
        routing_manifest: Dict[str, Any],
        extraction_results: List[Dict[str, Any]],
    ) -> EvidenceStore:
        """
        Extract all evidence from project.

        Args:
            project_id: Project identifier
            page_index: List of indexed pages
            routing_manifest: Page routing manifest
            extraction_results: Extraction results

        Returns:
            EvidenceStore with all evidence
        """
        store = EvidenceStore(project_id=project_id)

        # Process each indexed page
        for page in page_index:
            page_type = self._get_page_type(page, routing_manifest)
            text = page.get("extracted_text", "") or page.get("text_snippet", "")

            if not text:
                continue

            # Extract based on page type
            if page_type in ["notes_specs", "unknown"]:
                self._extract_notes(store, page, text)

            if page_type in ["schedule_table"]:
                self._extract_schedule_evidence(store, page, text)

            # Always look for cross-references and code references
            self._extract_cross_references(store, page, text)
            self._extract_code_references(store, page, text)

            # Extract scope-related keywords
            self._extract_scope_keywords(store, page, text)

        # Process extraction results for schedule data
        for result in extraction_results:
            if result.get("schedule_data"):
                self._extract_from_schedule_data(store, result)

        return store

    def _get_page_type(
        self,
        page: Dict[str, Any],
        routing_manifest: Dict[str, Any]
    ) -> str:
        """Get page type from routing manifest."""
        file_path = page.get("file_path", "")
        page_num = page.get("page_number", 0)

        routings = routing_manifest.get("routings", [])
        for routing in routings:
            if routing.get("file_path") == file_path and routing.get("page_number") == page_num:
                return routing.get("page_type", "unknown")

        return "unknown"

    def _extract_notes(
        self,
        store: EvidenceStore,
        page: Dict[str, Any],
        text: str
    ) -> None:
        """Extract general notes evidence."""
        text_lower = text.lower()

        # Check for notes keywords
        notes_found = []
        for keyword in self.NOTES_KEYWORDS:
            if keyword in text_lower:
                notes_found.append(keyword)

        if notes_found:
            # Split into paragraphs/sentences
            paragraphs = re.split(r'\n\s*\n|\.\s+', text)

            for para in paragraphs:
                para = para.strip()
                if len(para) < 20:
                    continue

                # Check if paragraph contains relevant keywords
                para_lower = para.lower()
                matched_keywords = [k for k in notes_found if k in para_lower]

                if matched_keywords:
                    item = EvidenceItem(
                        evidence_id=self._generate_id("NOTE"),
                        evidence_type="notes",
                        source_file=page.get("file_path", ""),
                        source_page=page.get("page_number", 0),
                        snippet=para[:500],
                        full_text=para,
                        confidence=min(0.5 + len(matched_keywords) * 0.1, 1.0),
                        keywords_matched=matched_keywords,
                    )
                    store.add(item)

        # Check for legend content
        legend_found = [k for k in self.LEGEND_KEYWORDS if k in text_lower]
        if legend_found:
            item = EvidenceItem(
                evidence_id=self._generate_id("LEG"),
                evidence_type="legend",
                source_file=page.get("file_path", ""),
                source_page=page.get("page_number", 0),
                snippet=text[:500],
                full_text=text,
                confidence=0.7,
                keywords_matched=legend_found,
            )
            store.add(item)

    def _extract_schedule_evidence(
        self,
        store: EvidenceStore,
        page: Dict[str, Any],
        text: str
    ) -> None:
        """Extract evidence from schedule pages."""
        text_lower = text.lower()

        # Detect schedule type
        schedule_types = []
        if any(k in text_lower for k in ["door schedule", "door", "d1", "d2"]):
            schedule_types.append("door_schedule")
        if any(k in text_lower for k in ["window schedule", "window", "w1", "w2"]):
            schedule_types.append("window_schedule")
        if any(k in text_lower for k in ["finish", "floor finish", "wall finish"]):
            schedule_types.append("finish_schedule")
        if any(k in text_lower for k in ["bbs", "bar bending", "reinforcement"]):
            schedule_types.append("bbs_schedule")
        if any(k in text_lower for k in ["column schedule", "beam schedule"]):
            schedule_types.append("structural_schedule")

        for stype in schedule_types:
            item = EvidenceItem(
                evidence_id=self._generate_id("SCHED"),
                evidence_type="schedule",
                source_file=page.get("file_path", ""),
                source_page=page.get("page_number", 0),
                snippet=f"Schedule detected: {stype}",
                confidence=0.8,
                keywords_matched=[stype],
            )
            store.add(item)

    def _extract_cross_references(
        self,
        store: EvidenceStore,
        page: Dict[str, Any],
        text: str
    ) -> None:
        """Extract detail and drawing cross-references."""
        for pattern in self.DETAIL_REF_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(0)
                item = EvidenceItem(
                    evidence_id=self._generate_id("DREF"),
                    evidence_type="detail_ref",
                    source_file=page.get("file_path", ""),
                    source_page=page.get("page_number", 0),
                    snippet=ref_text,
                    confidence=0.9,
                    cross_references=[ref_text],
                )
                store.add(item)

                # Track detail references
                store.detail_references.append({
                    "reference": ref_text,
                    "source_file": page.get("file_path", ""),
                    "source_page": page.get("page_number", 0),
                })

    def _extract_code_references(
        self,
        store: EvidenceStore,
        page: Dict[str, Any],
        text: str
    ) -> None:
        """Extract Indian Standard and code references."""
        for pattern in self.CODE_REF_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(0)
                item = EvidenceItem(
                    evidence_id=self._generate_id("CODE"),
                    evidence_type="code_ref",
                    source_file=page.get("file_path", ""),
                    source_page=page.get("page_number", 0),
                    snippet=ref_text,
                    confidence=0.95,
                    cross_references=[ref_text],
                )
                store.add(item)

                # Track code references
                store.code_references.append({
                    "code": ref_text,
                    "source_file": page.get("file_path", ""),
                    "source_page": page.get("page_number", 0),
                })

    def _extract_scope_keywords(
        self,
        store: EvidenceStore,
        page: Dict[str, Any],
        text: str
    ) -> None:
        """Extract scope-related keywords for each package."""
        text_upper = text.upper()

        for scope_area, keywords in self.SCOPE_KEYWORDS.items():
            matched = []
            for keyword in keywords:
                if keyword.upper() in text_upper:
                    matched.append(keyword)

            if matched:
                item = EvidenceItem(
                    evidence_id=self._generate_id("SCOPE"),
                    evidence_type="scope_keyword",
                    source_file=page.get("file_path", ""),
                    source_page=page.get("page_number", 0),
                    snippet=f"Scope area: {scope_area}",
                    confidence=min(0.4 + len(matched) * 0.15, 0.95),
                    keywords_matched=matched,
                )
                store.add(item)

    def _extract_from_schedule_data(
        self,
        store: EvidenceStore,
        result: Dict[str, Any]
    ) -> None:
        """Extract evidence from parsed schedule data."""
        schedule_data = result.get("schedule_data", {})
        entries = schedule_data.get("entries", [])

        if entries:
            # Determine schedule type from entries
            has_doors = any("D" in str(e.get("tag", "")).upper() for e in entries)
            has_windows = any("W" in str(e.get("tag", "")).upper() for e in entries)
            has_rebar = any(e.get("diameter") for e in entries)

            schedule_type = "unknown_schedule"
            if has_doors:
                schedule_type = "door_schedule"
            elif has_windows:
                schedule_type = "window_schedule"
            elif has_rebar:
                schedule_type = "bbs_schedule"

            item = EvidenceItem(
                evidence_id=self._generate_id("SDATA"),
                evidence_type="schedule_data",
                source_file=result.get("file_path", ""),
                source_page=result.get("page_number", 0),
                snippet=f"Parsed {len(entries)} entries from {schedule_type}",
                confidence=0.85,
                keywords_matched=[schedule_type, f"{len(entries)}_entries"],
            )
            store.add(item)


def extract_evidence(
    project_id: str,
    page_index: List[Dict[str, Any]],
    routing_manifest: Dict[str, Any],
    extraction_results: List[Dict[str, Any]],
    output_dir: Path,
) -> EvidenceStore:
    """
    Convenience function to extract and save evidence.

    Args:
        project_id: Project identifier
        page_index: List of indexed pages
        routing_manifest: Page routing manifest
        extraction_results: Extraction results
        output_dir: Output directory

    Returns:
        EvidenceStore
    """
    extractor = EvidenceExtractor()
    store = extractor.extract_from_project(
        project_id, page_index, routing_manifest, extraction_results
    )

    # Save to file
    scope_dir = output_dir / "scope"
    scope_dir.mkdir(parents=True, exist_ok=True)
    store.save(scope_dir / "evidence.json")

    return store
