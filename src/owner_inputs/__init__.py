"""
Owner Inputs Engine - Validate owner inputs and generate RFIs for missing mandatory fields.

This module:
- Reads owner_inputs.yaml from project folder
- Validates against template for mandatory fields
- Generates RFIs for missing mandatory fields
- Adds allowances automatically for missing specifications
- Outputs: owner_inputs_required.md, owner_inputs_used.json

India-specific construction specifications and defaults.
"""

from .validator import OwnerInputsValidator, ValidationResult, MissingField
from .defaults import DefaultsEngine, AppliedDefault
from .rfi_generator import OwnerInputsRFIGenerator

__all__ = [
    "OwnerInputsValidator",
    "ValidationResult",
    "MissingField",
    "DefaultsEngine",
    "AppliedDefault",
    "OwnerInputsRFIGenerator",
    "run_owner_inputs_engine",
]


def run_owner_inputs_engine(
    project_id: str,
    project_dir,
    output_dir,
    owner_docs_results: dict = None,
) -> dict:
    """
    Run the owner inputs engine.

    Args:
        project_id: Project identifier
        project_dir: Project directory (contains owner_inputs.yaml)
        output_dir: Output directory
        owner_docs_results: Results from owner docs parsing (optional)

    Returns:
        Dict with owner inputs results
    """
    from pathlib import Path
    import json
    import logging
    import yaml

    logger = logging.getLogger(__name__)
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load owner inputs YAML if it exists
    owner_inputs_path = project_dir / "owner_inputs.yaml"
    owner_inputs = {}

    if owner_inputs_path.exists():
        logger.info(f"Loading owner inputs from {owner_inputs_path}")
        with open(owner_inputs_path, "r") as f:
            owner_inputs = yaml.safe_load(f) or {}
    else:
        logger.warning(f"No owner_inputs.yaml found at {owner_inputs_path}")
        logger.info("Will generate RFIs for all mandatory fields")

    # 2. Validate against template
    logger.info("Validating owner inputs against template...")
    validator = OwnerInputsValidator()
    validation_result = validator.validate(owner_inputs)

    logger.info(f"Found {len(validation_result.missing_mandatory)} missing mandatory fields")
    logger.info(f"Found {len(validation_result.missing_optional)} missing optional fields")
    logger.info(f"Completeness score: {validation_result.completeness_score:.0f}%")

    # 3. Apply defaults for missing optional fields
    logger.info("Applying defaults for missing optional fields...")
    defaults_engine = DefaultsEngine()
    applied_defaults, final_inputs = defaults_engine.apply_defaults(
        owner_inputs, validation_result.missing_optional
    )
    logger.info(f"Applied {len(applied_defaults)} defaults")

    # 4. Merge with owner docs extracted data if available
    if owner_docs_results:
        logger.info("Merging with owner docs extracted data...")
        final_inputs = _merge_owner_docs_data(final_inputs, owner_docs_results)

    # 5. Generate RFIs for missing mandatory fields
    logger.info("Generating RFIs for missing mandatory fields...")
    rfi_generator = OwnerInputsRFIGenerator()
    rfis = rfi_generator.generate(validation_result.missing_mandatory)
    logger.info(f"Generated {len(rfis)} owner input RFIs")

    # 6. Generate allowances for estimation uncertainty
    allowances = _generate_allowances(validation_result.missing_mandatory)
    logger.info(f"Generated {len(allowances)} allowances")

    # 7. Export results
    # Owner inputs used (final merged values)
    with open(output_dir / "owner_inputs_used.json", "w") as f:
        json.dump(final_inputs, f, indent=2)

    # Applied defaults
    with open(output_dir / "applied_defaults.json", "w") as f:
        json.dump([d.to_dict() for d in applied_defaults], f, indent=2)

    # Owner inputs RFIs
    with open(output_dir / "owner_inputs_rfis.json", "w") as f:
        json.dump([r.to_dict() for r in rfis], f, indent=2)

    # Allowances
    with open(output_dir / "allowances.json", "w") as f:
        json.dump(allowances, f, indent=2)

    # Export markdown report
    _export_owner_inputs_report(
        output_dir / "owner_inputs_required.md",
        project_id,
        validation_result,
        applied_defaults,
        rfis,
        allowances,
    )

    return {
        "owner_inputs_provided": len(_count_provided_fields(owner_inputs)),
        "missing_mandatory": len(validation_result.missing_mandatory),
        "missing_optional": len(validation_result.missing_optional),
        "defaults_applied": len(applied_defaults),
        "completeness_score": validation_result.completeness_score,
        "rfis_generated": len(rfis),
        "allowances_generated": len(allowances),
        "final_inputs": final_inputs,
    }


def _count_provided_fields(obj, prefix=""):
    """Count non-empty fields in nested dict."""
    count = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                count.extend(_count_provided_fields(value, path))
            elif value is not None and value != "" and value != []:
                count.append(path)
    return count


def _merge_owner_docs_data(inputs: dict, owner_docs: dict) -> dict:
    """Merge extracted owner docs data into inputs."""
    merged = inputs.copy()

    # Merge contract terms
    if "contract_terms" in owner_docs:
        terms = owner_docs["contract_terms"]

        # GST terms
        if terms.get("gst_terms") and "commercial" not in merged:
            merged["commercial"] = {}
        if terms.get("gst_terms"):
            gst = terms["gst_terms"]
            if "gst_included" in gst:
                merged.setdefault("commercial", {})["gst_included"] = gst["gst_included"]

        # DLP
        if terms.get("dlp_months"):
            merged.setdefault("commercial", {})["defect_liability_months"] = terms["dlp_months"]

        # Retention
        if terms.get("retention_percent"):
            merged.setdefault("commercial", {}).setdefault("payment_terms", {})["retention_percent"] = terms["retention_percent"]

    # Merge tender info
    if "tender_info" in owner_docs:
        tender = owner_docs["tender_info"]

        if tender.get("project_name"):
            merged.setdefault("project", {})["name"] = tender["project_name"]
        if tender.get("location"):
            merged.setdefault("project", {}).setdefault("location", {})["city"] = tender["location"]
        if tender.get("completion_months"):
            merged.setdefault("project", {})["completion_months"] = tender["completion_months"]

    # Merge required makes from specs
    if "specifications" in owner_docs:
        specs = owner_docs["specifications"]
        required_makes = {}
        for spec in specs:
            if spec.get("brands"):
                required_makes[spec["category"]] = spec["brands"]

        if required_makes:
            merged.setdefault("preferences", {})["preferred_makes"] = required_makes

    return merged


def _generate_allowances(missing_mandatory: list) -> list:
    """Generate allowances for missing mandatory fields."""
    allowances = []

    # Allowance percentages by field category
    allowance_rates = {
        "finishes": 10.0,
        "waterproofing": 8.0,
        "doors": 12.0,
        "windows": 12.0,
        "sanitary": 15.0,
        "plumbing": 8.0,
        "electrical": 10.0,
        "hvac": 20.0,
        "fire": 15.0,
        "lift": 20.0,
        "external_works": 10.0,
        "project": 5.0,
    }

    # Group missing fields by category
    by_category = {}
    for field in missing_mandatory:
        category = field.path.split(".")[0]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(field)

    # Generate allowance for each category with missing fields
    for category, fields in by_category.items():
        rate = allowance_rates.get(category, 10.0)

        # Increase rate based on number of missing critical fields
        critical_count = sum(1 for f in fields if f.priority == "critical")
        if critical_count > 3:
            rate *= 1.5  # 50% increase for many missing critical fields

        allowances.append({
            "category": category,
            "allowance_percent": round(rate, 1),
            "reason": f"Missing {len(fields)} mandatory field(s): {', '.join(f.field_name for f in fields[:3])}{'...' if len(fields) > 3 else ''}",
            "missing_fields": [f.path for f in fields],
            "recommendation": f"Obtain {category} specifications to reduce allowance",
        })

    return allowances


def _export_owner_inputs_report(
    output_path,
    project_id: str,
    validation_result,
    applied_defaults: list,
    rfis: list,
    allowances: list,
) -> None:
    """Export owner inputs report as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Owner Inputs Report: {project_id}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Inputs Completeness**: {validation_result.completeness_score:.0f}%\n")
        f.write(f"- **Missing Mandatory**: {len(validation_result.missing_mandatory)}\n")
        f.write(f"- **Missing Optional**: {len(validation_result.missing_optional)}\n")
        f.write(f"- **Defaults Applied**: {len(applied_defaults)}\n")
        f.write(f"- **RFIs Generated**: {len(rfis)}\n")
        f.write(f"- **Allowances Added**: {len(allowances)}\n\n")

        # Missing Mandatory Fields
        if validation_result.missing_mandatory:
            f.write("## Missing Mandatory Fields (RFIs Required)\n\n")

            # Group by category
            by_category = {}
            for field in validation_result.missing_mandatory:
                cat = field.path.split(".")[0]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(field)

            for category, fields in sorted(by_category.items()):
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write("| Field | Why Needed | Impact |\n")
                f.write("|-------|------------|--------|\n")
                for field in fields:
                    f.write(f"| {field.field_name} | {field.why_needed} | {field.impact} |\n")
                f.write("\n")

        # Applied Defaults
        if applied_defaults:
            f.write("## Applied Defaults\n\n")
            f.write("The following defaults were applied for missing optional fields:\n\n")
            f.write("| Field | Default Value | Basis |\n")
            f.write("|-------|---------------|-------|\n")
            for default in applied_defaults:
                f.write(f"| {default.field_path} | {default.value} | {default.basis} |\n")
            f.write("\n")

        # Allowances
        if allowances:
            f.write("## Estimation Allowances\n\n")
            f.write("Due to missing specifications, the following allowances should be added:\n\n")
            f.write("| Category | Allowance % | Reason |\n")
            f.write("|----------|-------------|--------|\n")
            for allow in allowances:
                f.write(f"| {allow['category'].title()} | {allow['allowance_percent']}% | {allow['reason']} |\n")
            f.write("\n")

            total_risk = sum(a["allowance_percent"] for a in allowances) / len(allowances) if allowances else 0
            f.write(f"**Average Risk Allowance**: {total_risk:.1f}%\n\n")

        # RFIs
        if rfis:
            f.write("## Owner Input RFIs\n\n")
            f.write("The following RFIs must be raised to obtain missing information:\n\n")

            for rfi in rfis:
                priority_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(rfi.priority, "⚪")
                f.write(f"### {priority_emoji} {rfi.rfi_id}: {rfi.field_name}\n\n")
                f.write(f"**Question**: {rfi.question}\n\n")
                f.write(f"**Why Needed**: {rfi.why_needed}\n\n")
                f.write(f"**Options**: {', '.join(rfi.options)}\n\n")
                if rfi.default_assumption:
                    f.write(f"**Default Assumption**: {rfi.default_assumption}\n\n")
                f.write("---\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        if validation_result.completeness_score < 50:
            f.write("⚠️ **Critical**: Owner inputs are significantly incomplete. ")
            f.write("Estimation will have high uncertainty. Recommend obtaining mandatory inputs before proceeding.\n\n")
        elif validation_result.completeness_score < 80:
            f.write("🟠 **Important**: Several mandatory fields are missing. ")
            f.write("Raise RFIs and add appropriate allowances to the estimate.\n\n")
        else:
            f.write("✅ **Good**: Most mandatory fields are provided. ")
            f.write("Minor gaps can be addressed with standard assumptions.\n\n")
