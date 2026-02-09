"""
Detail Page Understanding Engine - Extract implicit scope from detail drawings.

This module provides:
- Detail page classification
- Common detail type detection
- Implicit scope extraction
- Scope register updates
- RFI reduction through detail evidence

India-specific detail types:
- Toilet waterproofing details
- Terrace waterproofing details
- Parapet details
- Stair details
- Window sill details
- Door frame details
- Plinth protection details
- Plumbing riser details
- Electrical conduit details
"""

from .classifier import DetailClassifier, DetailType, DetailClassification
from .extractor import DetailExtractor, DetailSpec, ExtractedDetail
from .mapper import ScopeMapper, ScopeMappingResult

__all__ = [
    # Classification
    "DetailClassifier",
    "DetailType",
    "DetailClassification",
    # Extraction
    "DetailExtractor",
    "DetailSpec",
    "ExtractedDetail",
    # Mapping
    "ScopeMapper",
    "ScopeMappingResult",
]


def run_detail_engine(
    project_id: str,
    extraction_results: list,
    scope_register: dict,
    rfi_list: list,
    output_dir,
) -> dict:
    """
    Run the complete detail understanding pipeline.

    Args:
        project_id: Project identifier
        extraction_results: Extraction results from pages
        scope_register: Current scope register
        rfi_list: Current RFI list
        output_dir: Output directory

    Returns:
        Dict with detail analysis results
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)

    # 1. Classify all detail pages
    logger.info("Classifying detail pages...")
    classifier = DetailClassifier()
    classifications = classifier.classify_all(extraction_results)
    logger.info(f"Found {len(classifications)} detail pages")

    # 2. Extract specifications from details
    logger.info("Extracting detail specifications...")
    extractor = DetailExtractor()
    extracted_details = extractor.extract_all(extraction_results, classifications)
    logger.info(f"Extracted {len(extracted_details)} detail specifications")

    # 3. Map to scope register
    logger.info("Mapping details to scope...")
    mapper = ScopeMapper()
    mapping_result = mapper.map_to_scope(
        extracted_details,
        scope_register,
        rfi_list,
    )

    # 4. Export results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detail classifications
    classifications_data = [c.to_dict() for c in classifications]
    with open(output_dir / "detail_classifications.json", "w") as f:
        json.dump(classifications_data, f, indent=2)

    # Extracted details
    details_data = [d.to_dict() for d in extracted_details]
    with open(output_dir / "extracted_details.json", "w") as f:
        json.dump(details_data, f, indent=2)

    # Mapping result
    mapping_data = mapping_result.to_dict()
    with open(output_dir / "detail_scope_mapping.json", "w") as f:
        json.dump(mapping_data, f, indent=2)

    # Detail report MD
    _export_detail_report(
        output_dir / "detail_report.md",
        project_id,
        classifications,
        extracted_details,
        mapping_result,
    )

    return {
        "detail_pages": len(classifications),
        "details_by_type": mapping_result.details_by_type,
        "scope_items_promoted": len(mapping_result.promoted_items),
        "rfis_reduced": len(mapping_result.reduced_rfis),
        "new_evidence": len(mapping_result.new_evidence),
        "implied_to_detected": len(mapping_result.promoted_items),
    }


def _export_detail_report(
    output_path,
    project_id: str,
    classifications,
    extracted_details,
    mapping_result,
) -> None:
    """Export detail report as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Detail Page Analysis: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Detail Pages Found**: {len(classifications)}\n")
        f.write(f"- **Specifications Extracted**: {len(extracted_details)}\n")
        f.write(f"- **Scope Items Promoted**: {len(mapping_result.promoted_items)}\n")
        f.write(f"- **RFIs Reduced**: {len(mapping_result.reduced_rfis)}\n")
        f.write(f"- **New Evidence Added**: {len(mapping_result.new_evidence)}\n\n")

        # Details by Type
        f.write("## Details by Type\n\n")
        f.write("| Detail Type | Count | Packages Affected |\n")
        f.write("|-------------|-------|-------------------|\n")
        for detail_type, count in sorted(
            mapping_result.details_by_type.items(),
            key=lambda x: -x[1]
        ):
            packages = mapping_result.packages_by_type.get(detail_type, [])
            f.write(f"| {detail_type} | {count} | {', '.join(packages[:3])} |\n")
        f.write("\n")

        # Detail Pages
        f.write("## Detail Pages\n\n")
        for cls in sorted(classifications, key=lambda c: c.detail_type.value):
            f.write(f"### {cls.sheet_id}: {cls.title or cls.detail_type.value}\n\n")
            f.write(f"- **Type**: {cls.detail_type.value}\n")
            f.write(f"- **Confidence**: {cls.confidence:.0%}\n")
            f.write(f"- **Sub-details**: {len(cls.sub_details)}\n")

            if cls.sub_details:
                f.write("- **Contents**:\n")
                for sub in cls.sub_details[:5]:
                    f.write(f"  - {sub}\n")
            f.write("\n")

        # Extracted Specifications
        f.write("## Extracted Specifications\n\n")
        for detail in extracted_details[:20]:
            f.write(f"### {detail.detail_id}\n\n")
            f.write(f"- **Type**: {detail.detail_type}\n")
            f.write(f"- **Source**: {detail.source_page}\n")

            if detail.specs:
                f.write("- **Specifications**:\n")
                for spec in detail.specs[:5]:
                    f.write(f"  - {spec.name}: {spec.value} {spec.unit or ''}\n")
            f.write("\n")

        # Scope Promotions
        if mapping_result.promoted_items:
            f.write("## Scope Status Changes\n\n")
            f.write("The following scope items were promoted from IMPLIED to DETECTED:\n\n")
            f.write("| Scope Item | Package | Evidence |\n")
            f.write("|------------|---------|----------|\n")
            for item in mapping_result.promoted_items:
                f.write(f"| {item['item_id']} | {item['package']} | {item['evidence'][:40]}... |\n")
            f.write("\n")

        # Reduced RFIs
        if mapping_result.reduced_rfis:
            f.write("## RFIs Resolved by Detail Evidence\n\n")
            f.write("The following RFIs are no longer needed due to detail evidence:\n\n")
            for rfi_id in mapping_result.reduced_rfis[:10]:
                f.write(f"- ✅ {rfi_id}\n")
            if len(mapping_result.reduced_rfis) > 10:
                f.write(f"- ... and {len(mapping_result.reduced_rfis) - 10} more\n")
            f.write("\n")

        # New Evidence
        if mapping_result.new_evidence:
            f.write("## New Evidence Added\n\n")
            for evidence in mapping_result.new_evidence[:15]:
                f.write(f"- **{evidence['package']}**: {evidence['text'][:60]}...\n")
