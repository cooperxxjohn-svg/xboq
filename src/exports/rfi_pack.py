"""
RFI Pack Export

Generates operational RFI deliverables:
- CSV tracker for estimators
- Email-ready drafts (per trade + combined)
- HTML print view
"""

import csv
import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from .models import RFIItem, Evidence


def parse_rfis_to_items(rfis: List[Dict]) -> List[RFIItem]:
    """
    Convert raw RFI dicts to typed RFIItem objects with evidence.

    Args:
        rfis: List of RFI dicts from analysis

    Returns:
        List of RFIItem objects
    """
    items = []
    for rfi in rfis:
        evidence = Evidence(
            page_numbers=rfi.get("evidence_pages", []),
            detected_entities=rfi.get("detected_tags", []) + rfi.get("detected_marks", []),
            not_found=rfi.get("issue_type", ""),
        )

        items.append(RFIItem(
            id=rfi.get("id", f"RFI-{len(items)+1:04d}"),
            title=rfi.get("title", "Untitled"),
            trade=rfi.get("trade", "general"),
            priority=rfi.get("priority", "medium"),
            missing_info=rfi.get("description", ""),
            why_it_matters=rfi.get("impact", "Cost and schedule risk"),
            evidence=evidence,
            suggested_resolution=rfi.get("suggested_response", "Please provide clarification"),
            issue_type=rfi.get("issue_type", ""),
            package=rfi.get("package", ""),
        ))

    return items


def dedupe_rfis(items: List[RFIItem]) -> List[RFIItem]:
    """
    Deduplicate and merge similar RFIs.

    Merges RFIs with same issue_type + trade, combining evidence.
    """
    merged = {}

    for item in items:
        key = (item.issue_type, item.trade)

        if key in merged:
            # Merge evidence
            existing = merged[key]
            existing.evidence.page_numbers = list(set(
                existing.evidence.page_numbers + item.evidence.page_numbers
            ))
            existing.evidence.detected_entities = list(set(
                existing.evidence.detected_entities + item.evidence.detected_entities
            ))
            # Keep higher priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            if priority_order.get(item.priority, 3) < priority_order.get(existing.priority, 3):
                existing.priority = item.priority
        else:
            merged[key] = item

    return list(merged.values())


def group_by_trade(items: List[RFIItem]) -> Dict[str, List[RFIItem]]:
    """Group RFIs by trade."""
    grouped = defaultdict(list)
    for item in items:
        grouped[item.trade].append(item)

    # Sort each group by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for trade in grouped:
        grouped[trade].sort(key=lambda x: priority_order.get(x.priority, 3))

    return dict(grouped)


def sort_by_priority(items: List[RFIItem]) -> List[RFIItem]:
    """Sort RFIs by priority (critical first)."""
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(items, key=lambda x: priority_order.get(x.priority, 3))


def generate_csv_tracker(items: List[RFIItem]) -> str:
    """Generate CSV content for RFI tracker."""
    output = StringIO()
    fieldnames = [
        "RFI ID", "Title", "Trade", "Priority", "Missing Info",
        "Why It Matters", "Evidence Pages", "Detected Items", "Suggested Resolution"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for item in sort_by_priority(items):
        writer.writerow(item.to_csv_row())

    return output.getvalue()


def generate_email_draft(items: List[RFIItem], trade: Optional[str] = None) -> str:
    """
    Generate email-ready RFI text.

    Args:
        items: List of RFI items
        trade: If specified, filter to this trade only

    Returns:
        Formatted email text
    """
    if trade:
        items = [i for i in items if i.trade == trade]
        subject_trade = trade.title()
    else:
        subject_trade = "All Trades"

    if not items:
        return ""

    items = sort_by_priority(items)

    lines = [
        f"Subject: RFIs for {subject_trade} - {len(items)} Items Requiring Clarification",
        "",
        "Dear Consultant,",
        "",
        f"Please find below {len(items)} Request(s) for Information (RFIs) that require clarification before we can finalize our pricing.",
        "",
        "=" * 60,
        "",
    ]

    for idx, item in enumerate(items, 1):
        lines.append(f"RFI #{idx}: {item.title}")
        lines.append("-" * 40)
        lines.append(item.to_email_text())
        lines.append("")

    lines.extend([
        "=" * 60,
        "",
        "Please respond to these queries at your earliest convenience.",
        "",
        "Best regards,",
        "[Your Name]",
        "[Company]",
    ])

    return "\n".join(lines)


def generate_html_view(
    items: List[RFIItem],
    project_id: str,
    trade_summary: Dict[str, Any],
) -> str:
    """Generate HTML print view for RFI pack."""
    grouped = group_by_trade(items)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Count by priority
    priority_counts = defaultdict(int)
    for item in items:
        priority_counts[item.priority] += 1

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RFI Pack - {project_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 1.8rem;
        }}
        .header .meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-box {{
            background: white;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #7c3aed;
        }}
        .stat-value.critical {{ color: #dc2626; }}
        .stat-value.high {{ color: #ea580c; }}
        .stat-label {{
            color: #6b7280;
            font-size: 0.85rem;
        }}
        .trade-section {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .trade-header {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }}
        .rfi-item {{
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        .rfi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}
        .rfi-title {{
            font-weight: 600;
            color: #1f2937;
        }}
        .priority-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .priority-critical {{ background: #fef2f2; color: #dc2626; }}
        .priority-high {{ background: #fff7ed; color: #ea580c; }}
        .priority-medium {{ background: #fffbeb; color: #d97706; }}
        .priority-low {{ background: #f0fdf4; color: #16a34a; }}
        .rfi-description {{
            color: #4b5563;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
        }}
        .rfi-impact {{
            color: #dc2626;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        .rfi-evidence {{
            background: #f3f4f6;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #6b7280;
        }}
        .footer {{
            text-align: center;
            color: #9ca3af;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }}
        @media print {{
            body {{ background: white; }}
            .trade-section {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RFI Pack</h1>
        <div class="meta">Project: {project_id} | Generated: {timestamp}</div>
    </div>

    <div class="summary">
        <div class="stat-box">
            <div class="stat-value">{len(items)}</div>
            <div class="stat-label">Total RFIs</div>
        </div>
        <div class="stat-box">
            <div class="stat-value critical">{priority_counts.get('critical', 0) + priority_counts.get('high', 0)}</div>
            <div class="stat-label">Critical/High</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(grouped)}</div>
            <div class="stat-label">Trades Affected</div>
        </div>
        <div class="stat-box">
            <div class="stat-value high">{priority_counts.get('medium', 0)}</div>
            <div class="stat-label">Medium Priority</div>
        </div>
    </div>
"""

    for trade, trade_items in sorted(grouped.items()):
        html += f"""
    <div class="trade-section">
        <div class="trade-header">{trade.title()} ({len(trade_items)} RFIs)</div>
"""
        for item in trade_items:
            html += f"""
        <div class="rfi-item">
            <div class="rfi-header">
                <span class="rfi-title">{item.id}: {item.title}</span>
                <span class="priority-badge priority-{item.priority}">{item.priority}</span>
            </div>
            <div class="rfi-description">{item.missing_info}</div>
            <div class="rfi-impact">→ {item.why_it_matters}</div>
            <div class="rfi-evidence">
                <strong>Evidence:</strong> {item.evidence.summary()}<br>
                <strong>Resolution:</strong> {item.suggested_resolution}
            </div>
        </div>
"""
        html += "    </div>\n"

    html += """
    <div class="footer">
        Generated by XBOQ • Pre-Bid Scope & Risk Check
    </div>
</body>
</html>
"""
    return html


def build_rfi_pack(
    project_id: str,
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
    output_dir: Optional[Path] = None,
    dedupe: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Build complete RFI Pack with all outputs.

    Args:
        project_id: Project identifier
        rfis: List of RFI dicts from analysis
        trade_summary: Trade-wise summary
        output_dir: Where to save files
        dedupe: Whether to merge similar RFIs

    Returns:
        Tuple of (csv_path, email_txt_path, html_path)
    """
    # Parse and optionally dedupe
    items = parse_rfis_to_items(rfis)
    if dedupe:
        items = dedupe_rfis(items)

    if not items:
        return None, None, None

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "out" / project_id / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV Tracker
    csv_path = output_dir / f"rfi_tracker_{timestamp}.csv"
    csv_content = generate_csv_tracker(items)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # Email Draft (combined)
    email_path = output_dir / f"rfi_email_draft_{timestamp}.txt"
    email_content = generate_email_draft(items)
    with open(email_path, "w", encoding="utf-8") as f:
        f.write(email_content)

    # Also write per-trade emails
    grouped = group_by_trade(items)
    for trade in grouped:
        trade_email = output_dir / f"rfi_email_{trade}_{timestamp}.txt"
        trade_content = generate_email_draft(items, trade=trade)
        with open(trade_email, "w", encoding="utf-8") as f:
            f.write(trade_content)

    # HTML View
    html_path = output_dir / f"rfi_pack_{timestamp}.html"
    html_content = generate_html_view(items, project_id, trade_summary)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(csv_path), str(email_path), str(html_path)


def get_rfi_pack_buffer(
    rfis: List[Dict],
    trade_summary: Dict[str, Any],
    project_id: str,
    format: str = "csv",
) -> BytesIO:
    """
    Get RFI pack as in-memory buffer for Streamlit download.

    Args:
        rfis: List of RFI dicts
        trade_summary: Trade-wise summary
        project_id: Project identifier
        format: "csv", "txt" (email), or "html"

    Returns:
        BytesIO buffer with file contents
    """
    items = parse_rfis_to_items(rfis)
    items = dedupe_rfis(items)

    if format == "csv":
        content = generate_csv_tracker(items)
        return BytesIO(content.encode('utf-8'))

    elif format == "txt":
        content = generate_email_draft(items)
        return BytesIO(content.encode('utf-8'))

    elif format == "html":
        content = generate_html_view(items, project_id, trade_summary)
        return BytesIO(content.encode('utf-8'))

    return BytesIO()
