"""
Bid Documents Generator - Phase 24

Generates formal bid submission documents:
- Clarifications / Exceptions Letter
- Inclusions and Exclusions
- Assumptions and Allowances

Written in standard Indian contractor bid submission style.
"""

from .clarifications import ClarificationsGenerator

__all__ = [
    "ClarificationsGenerator",
    "run_clarifications_generator",
]


def run_clarifications_generator(
    project_id: str,
    bid_data: dict,
    gate_result: dict,
    output_dir,
) -> dict:
    """
    Generate clarifications letter.

    Args:
        project_id: Project identifier
        bid_data: Complete bid data
        gate_result: Result from bid gate
        output_dir: Output directory

    Returns:
        Generator results
    """
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ClarificationsGenerator()

    # Generate clarifications letter
    letter = generator.generate_letter(project_id, bid_data, gate_result)

    # Save letter
    letter_path = output_dir / "clarifications_letter.md"
    with open(letter_path, "w") as f:
        f.write(letter)

    logger.info(f"Clarifications letter generated: {letter_path}")

    return {
        "letter_path": str(letter_path),
        "inclusions_count": len(bid_data.get("inclusions", [])),
        "exclusions_count": len(bid_data.get("exclusions", [])),
        "assumptions_count": len(bid_data.get("assumptions", [])),
        "rfis_count": bid_data.get("high_priority_rfis_count", 0),
    }
