"""
Subcontractor Quote Leveling - Normalize and compare subcontractor quotes.

This module:
- Parses subcontractor quotes (Excel, PDF, manual entry)
- Normalizes scope across quotes
- Levels quotes to common basis (inclusions, exclusions, terms)
- Identifies best value (not just lowest price)
- Outputs: leveled_quotes.csv, quote_comparison.md, recommendation.json

India-specific: Handles common scope variations, GST treatment, transport terms.
"""

from .parser import QuoteParser, SubcontractorQuote
from .normalizer import QuoteNormalizer
from .leveler import QuoteLeveler, LeveledComparison
from .recommender import QuoteRecommender

__all__ = [
    "QuoteParser",
    "SubcontractorQuote",
    "QuoteNormalizer",
    "QuoteLeveler",
    "LeveledComparison",
    "QuoteRecommender",
    "run_quote_leveling",
]


def run_quote_leveling(
    project_id: str,
    quotes: list,
    package: str,
    boq_items: list,
    output_dir,
) -> dict:
    """
    Run the quote leveling engine.

    Args:
        project_id: Project identifier
        quotes: List of subcontractor quotes (dicts or file paths)
        package: Work package being quoted (e.g., "flooring", "plumbing")
        boq_items: BOQ items for this package
        output_dir: Output directory

    Returns:
        Dict with leveling results
    """
    from pathlib import Path
    import json
    import csv
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse quotes
    logger.info(f"Parsing {len(quotes)} quotes for {package}...")
    parser = QuoteParser()
    parsed_quotes = []

    for quote_input in quotes:
        if isinstance(quote_input, dict):
            parsed = parser.parse_dict(quote_input)
        elif isinstance(quote_input, str) and Path(quote_input).exists():
            parsed = parser.parse_file(Path(quote_input))
        else:
            logger.warning(f"Could not parse quote input: {quote_input}")
            continue

        if parsed:
            parsed_quotes.append(parsed)

    logger.info(f"Successfully parsed {len(parsed_quotes)} quotes")

    if len(parsed_quotes) < 2:
        logger.warning("Less than 2 quotes - cannot perform meaningful comparison")

    # 2. Normalize quotes
    logger.info("Normalizing quotes to common basis...")
    normalizer = QuoteNormalizer()
    normalized_quotes = normalizer.normalize(parsed_quotes, boq_items)

    # 3. Level quotes
    logger.info("Leveling quotes for comparison...")
    leveler = QuoteLeveler()
    leveled = leveler.level(normalized_quotes, boq_items)

    # 4. Generate recommendation
    logger.info("Generating recommendation...")
    recommender = QuoteRecommender()
    recommendation = recommender.recommend(leveled)

    # 5. Export results

    # Leveled quotes JSON
    with open(output_dir / "leveled_quotes.json", "w") as f:
        json.dump(leveled.to_dict(), f, indent=2)

    # Leveled quotes CSV
    if leveled.comparison_matrix:
        with open(output_dir / "leveled_quotes.csv", "w", newline="") as f:
            if leveled.comparison_matrix:
                fieldnames = leveled.comparison_matrix[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(leveled.comparison_matrix)

    # Recommendation JSON
    with open(output_dir / "recommendation.json", "w") as f:
        json.dump(recommendation, f, indent=2)

    # Quote comparison markdown
    _export_quote_comparison(
        output_dir / "quote_comparison.md",
        project_id,
        package,
        leveled,
        recommendation,
    )

    return {
        "package": package,
        "quotes_received": len(quotes),
        "quotes_parsed": len(parsed_quotes),
        "quotes_leveled": len(leveled.quotes),
        "lowest_bidder": recommendation.get("lowest_bidder", "N/A"),
        "recommended_bidder": recommendation.get("recommended_bidder", "N/A"),
        "lowest_amount": recommendation.get("lowest_amount", 0),
        "recommended_amount": recommendation.get("recommended_amount", 0),
        "savings_potential": recommendation.get("savings_potential", 0),
    }


def _export_quote_comparison(
    output_path,
    project_id: str,
    package: str,
    leveled,
    recommendation: dict,
) -> None:
    """Export quote comparison as markdown."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Quote Comparison: {project_id} - {package.replace('_', ' ').title()}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Package**: {package.replace('_', ' ').title()}\n")
        f.write(f"- **Quotes Received**: {len(leveled.quotes)}\n")
        f.write(f"- **Lowest Bid**: {recommendation.get('lowest_bidder', 'N/A')} - ₹{recommendation.get('lowest_amount', 0):,.2f}\n")
        f.write(f"- **Recommended**: {recommendation.get('recommended_bidder', 'N/A')} - ₹{recommendation.get('recommended_amount', 0):,.2f}\n\n")

        # Quote comparison table
        f.write("## Quote Comparison (Leveled)\n\n")

        if leveled.quotes:
            headers = ["Item"] + [q.subcontractor_name for q in leveled.quotes]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---" for _ in headers]) + "|\n")

            # Item-wise comparison
            for row in leveled.comparison_matrix:
                values = [row.get("item", "")]
                for q in leveled.quotes:
                    val = row.get(q.subcontractor_name, 0)
                    if isinstance(val, (int, float)):
                        values.append(f"₹{val:,.2f}")
                    else:
                        values.append(str(val))
                f.write("| " + " | ".join(values) + " |\n")

            # Total row
            f.write("| **TOTAL (Leveled)** |")
            for q in leveled.quotes:
                f.write(f" **₹{q.leveled_total:,.2f}** |")
            f.write("\n\n")

        # Scope differences
        if leveled.scope_differences:
            f.write("## Scope Differences\n\n")
            f.write("Items where subcontractors differ from base scope:\n\n")

            for diff in leveled.scope_differences:
                f.write(f"### {diff['subcontractor']}\n\n")
                if diff.get("exclusions"):
                    f.write("**Exclusions** (added back for leveling):\n")
                    for exc in diff["exclusions"]:
                        f.write(f"- {exc}\n")
                if diff.get("inclusions"):
                    f.write("\n**Extra Inclusions** (credit given):\n")
                    for inc in diff["inclusions"]:
                        f.write(f"- {inc}\n")
                f.write("\n")

        # Terms comparison
        if leveled.terms_comparison:
            f.write("## Commercial Terms Comparison\n\n")
            f.write("| Term | " + " | ".join([q.subcontractor_name for q in leveled.quotes]) + " |\n")
            f.write("|------|" + "|".join(["---" for _ in leveled.quotes]) + "|\n")

            for term, values in leveled.terms_comparison.items():
                f.write(f"| {term} |")
                for q in leveled.quotes:
                    f.write(f" {values.get(q.subcontractor_name, 'N/A')} |")
                f.write("\n")
            f.write("\n")

        # Recommendation
        f.write("## Recommendation\n\n")
        f.write(f"**Recommended Subcontractor**: {recommendation.get('recommended_bidder', 'N/A')}\n\n")

        if recommendation.get("reasons"):
            f.write("**Reasons**:\n")
            for reason in recommendation["reasons"]:
                f.write(f"- {reason}\n")
            f.write("\n")

        if recommendation.get("risks"):
            f.write("**Risks to Consider**:\n")
            for risk in recommendation["risks"]:
                f.write(f"- ⚠️ {risk}\n")
            f.write("\n")

        # Notes
        f.write("## Notes\n\n")
        f.write("- All amounts are leveled to common scope basis\n")
        f.write("- GST is excluded from comparison (to be added separately)\n")
        f.write("- Verify subcontractor credentials before award\n")
        f.write("- Get reference checks for high-value packages\n")
