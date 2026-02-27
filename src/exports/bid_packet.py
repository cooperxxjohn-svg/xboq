"""
Bid Readiness Packet Export

Generates comprehensive bid readiness report:
- Executive Summary (GO/NO-GO, scores, blockers)
- Critical Blockers with evidence
- Trade-wise Gap Summary
- Assumptions Log
- Drawing Set Metadata
- ZIP bundle with all artifacts
"""

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from .models import Blocker, Evidence, Assumption, DrawingSetMeta, TradeGap
from .pricing_readiness import compute_pricing_readiness
from .rfi_pack import parse_rfis_to_items, group_by_trade


def build_executive_summary(
    project_id: str,
    bid_gate: Dict,
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build executive summary data."""
    status = bid_gate.get("status", "NO-GO")
    score = bid_gate.get("score", 0)
    blockers = bid_gate.get("blockers", [])

    # Count RFIs by priority
    priority_counts = defaultdict(int)
    for rfi in rfis:
        priority_counts[rfi.get("priority", "medium")] += 1

    # Determine priceability
    high_critical = priority_counts.get("critical", 0) + priority_counts.get("high", 0)
    if high_critical == 0:
        priceability = "Ready for pricing"
        price_score = 85
    elif high_critical <= 2:
        priceability = "Partial pricing possible"
        price_score = 60
    else:
        priceability = "Pricing blocked"
        price_score = 30

    # What can be priced now
    priceable_trades = []
    blocked_trades = []
    for trade, data in trade_summary.items():
        if data.get("high_priority", 0) == 0:
            priceable_trades.append(trade.title())
        elif data.get("rfi_count", 0) > 2:
            blocked_trades.append(trade.title())

    return {
        "project_id": project_id,
        "status": status,
        "bid_readiness_score": score,
        "priceability_score": price_score,
        "priceability_status": priceability,
        "total_rfis": len(rfis),
        "critical_rfis": high_critical,
        "medium_rfis": priority_counts.get("medium", 0),
        "blockers": blockers[:5],
        "priceable_trades": priceable_trades,
        "blocked_trades": blocked_trades,
        "pages_analyzed": metrics.get("pages", 0),
        "rooms_detected": metrics.get("rooms", 0),
        "openings_detected": metrics.get("openings", 0),
    }


def build_blockers_with_evidence(
    bid_gate: Dict,
    rfis: List[Dict],
) -> List[Blocker]:
    """Build detailed blockers from bid gate and high-priority RFIs."""
    blockers = []

    # From bid gate blockers
    for idx, blocker_text in enumerate(bid_gate.get("blockers", [])[:5]):
        blockers.append(Blocker(
            id=f"BLK-{idx+1:03d}",
            description=blocker_text,
            severity="critical",
            impact="Blocks bid submission",
            evidence=Evidence(not_found="See RFIs for details"),
            recommended_fix="Resolve related RFIs",
            fix_score_delta=10,
        ))

    # From high-priority RFIs
    for rfi in rfis:
        if rfi.get("priority") in ["critical", "high"]:
            evidence = Evidence(
                page_numbers=rfi.get("evidence_pages", []),
                detected_entities=rfi.get("detected_tags", []),
                not_found=rfi.get("issue_type", ""),
            )
            blockers.append(Blocker(
                id=rfi.get("id", f"RFI-{len(blockers)+1:03d}"),
                description=rfi.get("title", ""),
                severity=rfi.get("priority", "high"),
                impact=rfi.get("impact", "Cost/schedule risk"),
                evidence=evidence,
                recommended_fix=rfi.get("suggested_response", "Obtain clarification"),
                fix_score_delta=5 if rfi.get("priority") == "high" else 8,
            ))

    return blockers[:10]  # Top 10 blockers


def build_assumptions_log(
    bid_gate: Dict,
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
) -> List[Assumption]:
    """Auto-generate assumptions based on gaps."""
    assumptions = []
    assumption_id = 1

    # From blockers
    for blocker in bid_gate.get("blockers", []):
        blocker_lower = blocker.lower()
        if "scale" in blocker_lower:
            assumptions.append(Assumption(
                id=f"ASM-{assumption_id:03d}",
                category="scope",
                description="Scale assumed as 1:100 for unscaled drawings",
                impact_if_wrong="Quantities may be off by 20-50%",
            ))
            assumption_id += 1
        elif "schedule" in blocker_lower:
            assumptions.append(Assumption(
                id=f"ASM-{assumption_id:03d}",
                category="specification",
                description="Standard specifications assumed for items without schedules",
                impact_if_wrong="Material costs may vary significantly",
            ))
            assumption_id += 1

    # From trade gaps
    for trade, data in trade_summary.items():
        gaps = data.get("gaps", [])
        for gap in gaps[:2]:
            gap_lower = gap.lower()
            if "mep" in gap_lower or "electrical" in gap_lower:
                assumptions.append(Assumption(
                    id=f"ASM-{assumption_id:03d}",
                    category="scope",
                    description=f"MEP scope excluded from pricing - not in provided drawings",
                    impact_if_wrong="MEP costs not included in estimate",
                ))
                assumption_id += 1
                break

    # Standard assumptions
    standard_assumptions = [
        ("pricing", "All quantities subject to site verification", "5-10% variation expected"),
        ("timeline", "Standard lead times assumed for materials", "May vary based on market"),
        ("scope", "Underground/concealed work priced on visible scope only", "Additional work may be required"),
    ]

    for cat, desc, impact in standard_assumptions:
        if assumption_id <= 10:
            assumptions.append(Assumption(
                id=f"ASM-{assumption_id:03d}",
                category=cat,
                description=desc,
                impact_if_wrong=impact,
            ))
            assumption_id += 1

    return assumptions


def build_drawing_metadata(
    metrics: Dict[str, Any],
    trade_summary: Dict[str, Any],
    bid_gate: Dict,
) -> DrawingSetMeta:
    """Build drawing set metadata."""
    # Detect disciplines from trade summary
    disciplines = []
    for trade in trade_summary.keys():
        if trade == "structural":
            disciplines.append("Structural")
        elif trade == "architectural":
            disciplines.append("Architectural")
        elif trade == "mep":
            disciplines.append("MEP")
        elif trade == "civil":
            disciplines.append("Civil")

    # Scale status from blockers
    scale_status = "partial"
    pages_no_scale = 0
    for blocker in bid_gate.get("blockers", []):
        if "scale" in blocker.lower():
            scale_status = "partial"
            # Try to extract count
            import re
            match = re.search(r"(\d+)\s*page", blocker.lower())
            if match:
                pages_no_scale = int(match.group(1))

    return DrawingSetMeta(
        total_pages=metrics.get("pages", 0),
        total_sheets=metrics.get("pages", 0),  # Assume 1:1 for now
        disciplines_detected=disciplines or ["Architectural"],
        scale_status=scale_status,
        pages_without_scale=pages_no_scale,
        revision_dates=[],  # Would need OCR to extract
        file_names=[],  # Would need to track from input
    )


def generate_bid_packet_html(
    project_id: str,
    bid_gate: Dict,
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    client_facing: bool = False,
) -> str:
    """Generate comprehensive HTML bid readiness report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build all sections
    exec_summary = build_executive_summary(project_id, bid_gate, rfis, trade_summary, metrics)
    blockers = build_blockers_with_evidence(bid_gate, rfis)
    assumptions = build_assumptions_log(bid_gate, rfis, trade_summary)
    drawing_meta = build_drawing_metadata(metrics, trade_summary, bid_gate)
    pricing_rows = compute_pricing_readiness(rfis, trade_summary, bid_gate)

    status = exec_summary["status"]
    status_color = "#10b981" if status == "GO" else "#f59e0b" if status in ["REVIEW", "CONDITIONAL"] else "#ef4444"
    status_bg = "#d1fae5" if status == "GO" else "#fef3c7" if status in ["REVIEW", "CONDITIONAL"] else "#fee2e2"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bid Readiness Packet - {project_id}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            background: #f8fafc;
            color: #1e293b;
        }}
        .header {{
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
        }}
        .header .subtitle {{
            opacity: 0.9;
        }}
        .section {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            margin: 0 0 1rem 0;
            font-size: 1.3rem;
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }}
        .status-banner {{
            background: {status_bg};
            border: 2px solid {status_color};
            border-radius: 12px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .status-text {{
            font-size: 2.5rem;
            font-weight: 800;
            color: {status_color};
            margin-bottom: 0.5rem;
        }}
        .status-score {{
            color: #64748b;
            font-size: 1rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .metric-card {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #7c3aed;
        }}
        .metric-value.red {{ color: #ef4444; }}
        .metric-label {{
            color: #64748b;
            font-size: 0.85rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
            color: #475569;
            font-size: 0.85rem;
        }}
        .blocker-item {{
            border: 1px solid #fecaca;
            background: #fef2f2;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        .blocker-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}
        .blocker-title {{
            font-weight: 600;
            color: #b91c1c;
        }}
        .blocker-severity {{
            background: #ef4444;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            text-transform: uppercase;
        }}
        .blocker-evidence {{
            background: white;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 0.5rem;
        }}
        .assumption-item {{
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .assumption-checkbox {{
            width: 18px;
            height: 18px;
            border: 2px solid #cbd5e1;
            border-radius: 4px;
            flex-shrink: 0;
            margin-top: 2px;
        }}
        .risk-high {{ background: #fef2f2; color: #b91c1c; }}
        .risk-medium {{ background: #fffbeb; color: #b45309; }}
        .risk-low {{ background: #f0fdf4; color: #15803d; }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
        @media print {{
            body {{ background: white; padding: 1rem; }}
            .section {{ box-shadow: none; border: 1px solid #e2e8f0; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Bid Readiness Packet</h1>
        <div class="subtitle">Project: {project_id} | Generated: {timestamp}</div>
    </div>

    <!-- Status Banner -->
    <div class="status-banner">
        <div class="status-text">{"✓ " if status == "GO" else "✗ " if status == "NO-GO" else "⚠ "}{status}</div>
        <div class="status-score">
            Bid Readiness: {exec_summary['bid_readiness_score']}/100 |
            Priceability: {exec_summary['priceability_score']}/100
        </div>
    </div>

    <!-- Executive Summary -->
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{exec_summary['pages_analyzed']}</div>
                <div class="metric-label">Pages Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exec_summary['total_rfis']}</div>
                <div class="metric-label">RFIs Generated</div>
            </div>
            <div class="metric-card">
                <div class="metric-value red">{exec_summary['critical_rfis']}</div>
                <div class="metric-label">Critical/High RFIs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exec_summary['rooms_detected']}</div>
                <div class="metric-label">Rooms Detected</div>
            </div>
        </div>
        <p><strong>Priceability Status:</strong> {exec_summary['priceability_status']}</p>
        {"<p><strong>Trades ready for pricing:</strong> " + ", ".join(exec_summary['priceable_trades']) + "</p>" if exec_summary['priceable_trades'] else ""}
        {"<p><strong>Trades blocked:</strong> " + ", ".join(exec_summary['blocked_trades']) + "</p>" if exec_summary['blocked_trades'] else ""}
    </div>

    <!-- Critical Blockers -->
    <div class="section">
        <h2>Critical Blockers ({len(blockers)})</h2>
        <p style="color: #64748b; margin-bottom: 1rem;">These must be resolved before pricing:</p>
"""

    for blocker in blockers[:6]:
        html += f"""
        <div class="blocker-item">
            <div class="blocker-header">
                <span class="blocker-title">{blocker.id}: {blocker.description}</span>
                <span class="blocker-severity">{blocker.severity}</span>
            </div>
            <div style="color: #64748b; font-size: 0.9rem;">Impact: {blocker.impact}</div>
            <div class="blocker-evidence">
                Evidence: {blocker.evidence.summary()}<br>
                Fix: {blocker.recommended_fix} (Score +{blocker.fix_score_delta})
            </div>
        </div>
"""

    html += """
    </div>

    <!-- Trade-wise Gap Summary -->
    <div class="section">
        <h2>Trade-Wise Gap Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Trade</th>
                    <th>Coverage</th>
                    <th>Priceable</th>
                    <th>Blocked</th>
                    <th>Cost Risk</th>
                    <th>Schedule Risk</th>
                    <th>Next Action</th>
                </tr>
            </thead>
            <tbody>
"""

    for row in pricing_rows:
        cost_class = f"risk-{row.cost_risk}"
        schedule_class = f"risk-{row.schedule_risk}"
        html += f"""
                <tr>
                    <td><strong>{row.trade.title()}</strong></td>
                    <td>{row.scope_coverage_pct:.0f}%</td>
                    <td>{row.priceable_items}</td>
                    <td>{row.blocked_items}</td>
                    <td><span class="badge {cost_class}">{row.cost_risk.upper()}</span></td>
                    <td><span class="badge {schedule_class}">{row.schedule_risk.upper()}</span></td>
                    <td>{row.next_action}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>

    <!-- Assumptions Log -->
    <div class="section">
        <h2>Assumptions & Exclusions</h2>
        <p style="color: #64748b; margin-bottom: 1rem;">Review and accept before submitting bid:</p>
"""

    for assumption in assumptions:
        html += f"""
        <div class="assumption-item">
            <div class="assumption-checkbox"></div>
            <div>
                <div><strong>{assumption.id}</strong> [{assumption.category.title()}]: {assumption.description}</div>
                <div style="color: #64748b; font-size: 0.85rem;">If wrong: {assumption.impact_if_wrong}</div>
            </div>
        </div>
"""

    html += f"""
    </div>

    <!-- Drawing Set Metadata -->
    <div class="section">
        <h2>Drawing Set Metadata</h2>
        <table>
            <tr><td><strong>Total Pages</strong></td><td>{drawing_meta.total_pages}</td></tr>
            <tr><td><strong>Disciplines Detected</strong></td><td>{", ".join(drawing_meta.disciplines_detected)}</td></tr>
            <tr><td><strong>Scale Status</strong></td><td>{drawing_meta.scale_status.title()} ({drawing_meta.pages_without_scale} pages without scale)</td></tr>
        </table>
    </div>

    <div class="footer">
        Generated by XBOQ • Pre-Bid Scope & Risk Check<br>
        This report is auto-generated. Verify all assumptions before bid submission.
    </div>
</body>
</html>
"""
    return html


def build_bid_readiness_packet(
    project_id: str,
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
    bid_gate: Dict,
    metrics: Dict[str, Any],
    output_dir: Optional[Path] = None,
    include_zip: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Build complete Bid Readiness Packet.

    Args:
        project_id: Project identifier
        rfis: List of RFI dicts
        trade_summary: Trade-wise summary
        bid_gate: Bid gate result
        metrics: Run metrics
        output_dir: Where to save files
        include_zip: Whether to create ZIP bundle

    Returns:
        Tuple of (html_path, zip_path)
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "out" / project_id / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate HTML report
    html_content = generate_bid_packet_html(
        project_id, bid_gate, rfis, trade_summary, metrics
    )
    html_path = output_dir / f"bid_readiness_packet_{timestamp}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Create ZIP bundle if requested
    zip_path = None
    if include_zip:
        zip_path = output_dir / f"bid_readiness_packet_{timestamp}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add HTML report
            zf.writestr("bid_readiness_report.html", html_content)

            # Add summary JSON
            summary = {
                "project_id": project_id,
                "generated_at": timestamp,
                "bid_gate": bid_gate,
                "trade_summary": trade_summary,
                "metrics": metrics,
                "rfi_count": len(rfis),
            }
            zf.writestr("summary.json", json.dumps(summary, indent=2))

            # Add RFIs as JSON
            zf.writestr("rfis.json", json.dumps(rfis, indent=2))

            # Add assumptions
            assumptions = build_assumptions_log(bid_gate, rfis, trade_summary)
            zf.writestr("assumptions.json", json.dumps(
                [a.to_dict() for a in assumptions], indent=2
            ))

            # Add README
            readme = f"""XBOQ Bid Readiness Packet
========================

Project: {project_id}
Generated: {timestamp}

Contents:
- bid_readiness_report.html - Full report (open in browser)
- summary.json - Machine-readable summary
- rfis.json - All generated RFIs
- assumptions.json - Assumptions log

Status: {bid_gate.get('status', 'NO-GO')}
Score: {bid_gate.get('score', 0)}/100
"""
            zf.writestr("README.txt", readme)

    return str(html_path), str(zip_path) if zip_path else None


def get_bid_packet_buffer(
    project_id: str,
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
    bid_gate: Dict,
    metrics: Dict[str, Any],
    format: str = "html",
) -> BytesIO:
    """
    Get bid packet as in-memory buffer for Streamlit download.

    Args:
        project_id: Project identifier
        rfis: RFI list
        trade_summary: Trade summary
        bid_gate: Bid gate result
        metrics: Run metrics
        format: "html" or "zip"

    Returns:
        BytesIO buffer
    """
    if format == "html":
        content = generate_bid_packet_html(
            project_id, bid_gate, rfis, trade_summary, metrics
        )
        return BytesIO(content.encode('utf-8'))

    elif format == "zip":
        buffer = BytesIO()
        html_content = generate_bid_packet_html(
            project_id, bid_gate, rfis, trade_summary, metrics
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("bid_readiness_report.html", html_content)
            zf.writestr("summary.json", json.dumps({
                "project_id": project_id,
                "bid_gate": bid_gate,
                "trade_summary": trade_summary,
            }, indent=2))
            zf.writestr("rfis.json", json.dumps(rfis, indent=2))

        buffer.seek(0)
        return buffer

    return BytesIO()
