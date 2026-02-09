"""
Owner Docs Parser - Extract bid requirements from tender documents.

Parses:
- Tender notice / NIT
- Bid forms
- Owner BOQ
- Specifications
- Addenda
- Contract conditions

Extracts:
- Inclusions/Exclusions
- Required makes/brands
- Testing requirements
- LD clauses
- Milestones
- GST terms
- Method of measurement
- Alternates and allowances
"""

from .parser import OwnerDocsParser, ParsedDocument
from .tender import TenderParser, TenderInfo
from .boq_parser import OwnerBOQParser, OwnerBOQItem
from .specs import SpecsParser, SpecRequirement
from .addenda import AddendaParser, Addendum
from .extractor import ContractExtractor, ContractTerms

__all__ = [
    "OwnerDocsParser",
    "ParsedDocument",
    "TenderParser",
    "TenderInfo",
    "OwnerBOQParser",
    "OwnerBOQItem",
    "SpecsParser",
    "SpecRequirement",
    "AddendaParser",
    "Addendum",
    "ContractExtractor",
    "ContractTerms",
]


def run_owner_docs_engine(
    project_id: str,
    owner_docs_path,
    output_dir,
) -> dict:
    """
    Run the owner docs parsing pipeline.

    Args:
        project_id: Project identifier
        owner_docs_path: Path to owner_docs/ folder
        output_dir: Output directory

    Returns:
        Dict with parsed owner docs results
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    owner_docs_path = Path(owner_docs_path)

    output_subdir = output_dir / "owner_docs"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 1. Parse all documents
    logger.info("Parsing owner documents...")
    parser = OwnerDocsParser()
    parsed_docs = parser.parse_folder(owner_docs_path)
    logger.info(f"Parsed {len(parsed_docs)} documents")

    # 2. Extract tender info
    logger.info("Extracting tender information...")
    tender_parser = TenderParser()
    tender_info = tender_parser.extract(parsed_docs)

    # 3. Parse owner BOQ
    logger.info("Parsing owner BOQ...")
    boq_parser = OwnerBOQParser()
    owner_boq = boq_parser.parse(parsed_docs)
    logger.info(f"Extracted {len(owner_boq)} BOQ items")

    # 4. Parse specifications
    logger.info("Parsing specifications...")
    specs_parser = SpecsParser()
    specs = specs_parser.parse(parsed_docs)
    logger.info(f"Extracted {len(specs)} specification requirements")

    # 5. Parse addenda
    logger.info("Parsing addenda...")
    addenda_parser = AddendaParser()
    addenda = addenda_parser.parse(parsed_docs)
    logger.info(f"Found {len(addenda)} addenda")

    # 6. Extract contract terms
    logger.info("Extracting contract terms...")
    contract_extractor = ContractExtractor()
    contract_terms = contract_extractor.extract(parsed_docs, tender_info)

    # 7. Export results
    logger.info("Exporting results...")

    # Contract summary MD
    _export_contract_summary(
        output_subdir / "contract_summary.md",
        project_id,
        tender_info,
        contract_terms,
        specs,
    )

    # Owner BOQ CSV
    _export_owner_boq_csv(output_subdir / "owner_boq.csv", owner_boq)

    # Owner BOQ JSON
    with open(output_subdir / "owner_boq.json", "w") as f:
        json.dump([item.to_dict() for item in owner_boq], f, indent=2)

    # Inclusions CSV
    _export_inclusions_csv(output_subdir / "inclusions.csv", contract_terms.inclusions)

    # Exclusions CSV
    _export_exclusions_csv(output_subdir / "exclusions.csv", contract_terms.exclusions)

    # Addenda index JSON
    with open(output_subdir / "addenda_index.json", "w") as f:
        json.dump({
            "addenda": [a.to_dict() for a in addenda],
            "total_addenda": len(addenda),
        }, f, indent=2)

    # Specifications JSON
    with open(output_subdir / "specifications.json", "w") as f:
        json.dump([s.to_dict() for s in specs], f, indent=2)

    # Full parsed data JSON
    with open(output_subdir / "owner_docs_parsed.json", "w") as f:
        json.dump({
            "tender_info": tender_info.to_dict() if tender_info else {},
            "contract_terms": contract_terms.to_dict(),
            "owner_boq_count": len(owner_boq),
            "specs_count": len(specs),
            "addenda_count": len(addenda),
        }, f, indent=2)

    return {
        "documents_parsed": len(parsed_docs),
        "owner_boq_items": len(owner_boq),
        "specifications": len(specs),
        "addenda": len(addenda),
        "inclusions": len(contract_terms.inclusions),
        "exclusions": len(contract_terms.exclusions),
        "has_tender_info": tender_info is not None,
        "has_ld_clause": contract_terms.ld_clause is not None,
    }


def _export_contract_summary(path, project_id, tender_info, contract_terms, specs):
    """Export contract summary markdown."""
    from datetime import datetime

    with open(path, "w") as f:
        f.write(f"# Contract Summary: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Tender Info
        if tender_info:
            f.write("## Tender Information\n\n")
            f.write(f"- **Tender Reference**: {tender_info.reference or 'N/A'}\n")
            f.write(f"- **Project Name**: {tender_info.project_name or 'N/A'}\n")
            f.write(f"- **Owner**: {tender_info.owner_name or 'N/A'}\n")
            f.write(f"- **Location**: {tender_info.location or 'N/A'}\n")
            f.write(f"- **Submission Date**: {tender_info.submission_date or 'N/A'}\n")
            f.write(f"- **Completion Period**: {tender_info.completion_months or 'N/A'} months\n")
            f.write(f"- **EMD Amount**: ₹{tender_info.emd_amount or 'N/A'}\n")
            f.write("\n")

        # Commercial Terms
        f.write("## Commercial Terms\n\n")
        f.write(f"- **Contract Type**: {contract_terms.contract_type or 'N/A'}\n")
        f.write(f"- **GST Terms**: {contract_terms.gst_terms or 'Inclusive as per applicable rates'}\n")
        f.write(f"- **Payment Terms**: {contract_terms.payment_terms or 'As per contract'}\n")
        f.write(f"- **Retention**: {contract_terms.retention_percent or 5}%\n")
        f.write(f"- **DLP**: {contract_terms.dlp_months or 12} months\n")
        f.write("\n")

        # LD Clause
        if contract_terms.ld_clause:
            f.write("## Liquidated Damages (LD)\n\n")
            f.write(f"- **Rate**: {contract_terms.ld_clause.rate_percent}% per {contract_terms.ld_clause.period}\n")
            f.write(f"- **Maximum**: {contract_terms.ld_clause.max_percent}%\n")
            f.write(f"- **Notes**: {contract_terms.ld_clause.notes}\n")
            f.write("\n")

        # Milestones
        if contract_terms.milestones:
            f.write("## Milestones\n\n")
            f.write("| # | Description | Timeline | Payment % |\n")
            f.write("|---|-------------|----------|----------|\n")
            for i, ms in enumerate(contract_terms.milestones, 1):
                f.write(f"| {i} | {ms.description} | {ms.timeline} | {ms.payment_percent}% |\n")
            f.write("\n")

        # Inclusions
        if contract_terms.inclusions:
            f.write("## Inclusions\n\n")
            for inc in contract_terms.inclusions[:20]:
                f.write(f"- {inc.description}\n")
            if len(contract_terms.inclusions) > 20:
                f.write(f"- ... and {len(contract_terms.inclusions) - 20} more\n")
            f.write("\n")

        # Exclusions
        if contract_terms.exclusions:
            f.write("## Exclusions\n\n")
            for exc in contract_terms.exclusions:
                f.write(f"- ⚠️ {exc.description}\n")
            f.write("\n")

        # Required Makes/Brands
        if contract_terms.required_makes:
            f.write("## Required Makes/Brands\n\n")
            f.write("| Item | Required Makes | Approved Equivalent |\n")
            f.write("|------|---------------|--------------------|\n")
            for item, makes in contract_terms.required_makes.items():
                f.write(f"| {item} | {', '.join(makes[:3])} | {makes[-1] if 'equivalent' in makes[-1].lower() else 'N/A'} |\n")
            f.write("\n")

        # Testing Requirements
        if contract_terms.testing_requirements:
            f.write("## Testing Requirements\n\n")
            for test in contract_terms.testing_requirements:
                f.write(f"- **{test.item}**: {test.test_type} ({test.frequency})\n")
            f.write("\n")

        # Method of Measurement
        if contract_terms.mom_clauses:
            f.write("## Method of Measurement Notes\n\n")
            for clause in contract_terms.mom_clauses[:10]:
                f.write(f"- {clause}\n")
            f.write("\n")

        # Key Specifications
        if specs:
            f.write("## Key Specifications\n\n")
            for spec in specs[:15]:
                f.write(f"- **{spec.item}**: {spec.requirement}\n")


def _export_owner_boq_csv(path, owner_boq):
    """Export owner BOQ as CSV."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Item No", "Description", "Unit", "Quantity",
            "Rate", "Amount", "Package", "Notes"
        ])
        for item in owner_boq:
            writer.writerow([
                item.item_no,
                item.description,
                item.unit,
                item.quantity,
                item.rate or "",
                item.amount or "",
                item.package,
                item.notes,
            ])


def _export_inclusions_csv(path, inclusions):
    """Export inclusions as CSV."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Description", "Source", "Page"])
        for inc in inclusions:
            writer.writerow([
                inc.category,
                inc.description,
                inc.source_file,
                inc.page_number,
            ])


def _export_exclusions_csv(path, exclusions):
    """Export exclusions as CSV."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Description", "Impact", "Source", "Page"])
        for exc in exclusions:
            writer.writerow([
                exc.category,
                exc.description,
                exc.impact,
                exc.source_file,
                exc.page_number,
            ])
