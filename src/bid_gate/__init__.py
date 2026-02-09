"""
Bid Gate Engine - Phase 23

Hard stop safety checks before bid submission.
Computes PASS / PASS_WITH_RESERVATIONS / FAIL based on configurable thresholds.

Never hides uncertainty - converts gaps to reservations/allowances.
"""

from .gate import BidGate, GateResult, GateStatus, Reservation

__all__ = [
    "BidGate",
    "GateResult",
    "GateStatus",
    "Reservation",
    "run_bid_gate",
]


def run_bid_gate(
    project_id: str,
    bid_data: dict,
    output_dir,
) -> dict:
    """
    Run bid gate safety checks.

    Args:
        project_id: Project identifier
        bid_data: Complete bid data from all phases
        output_dir: Output directory

    Returns:
        Gate result dict
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run gate checks
    gate = BidGate()
    result = gate.evaluate(project_id, bid_data)

    # Export JSON report
    with open(output_dir / "bid_gate_report.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Export markdown report
    report_md = gate.generate_report(result)
    with open(output_dir / "bid_gate_report.md", "w") as f:
        f.write(report_md)

    logger.info(f"Bid Gate: {result.status.value}")
    logger.info(f"Score: {result.score:.1f}/100")
    logger.info(f"Reservations: {len(result.reservations)}")

    return {
        "status": result.status.value,
        "score": result.score,
        "reservations_count": len(result.reservations),
        "critical_failures": result.critical_failures,
        "is_submittable": result.status != GateStatus.FAIL,
        "stamp": result.stamp if result.status == GateStatus.FAIL else None,
    }
