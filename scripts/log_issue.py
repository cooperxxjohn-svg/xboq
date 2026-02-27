#!/usr/bin/env python3
"""
Log Issue to Improvement Log

Quick way to add issues found during validation.

Usage:
    python scripts/log_issue.py --project villa_001 --type SCALE --desc "Scale detected as 1:100 but actual is 1:50"
    python scripts/log_issue.py --project villa_001 --type AREA --desc "Living room area 15% off" --cause "Deduction not applied"
"""

import argparse
import re
from datetime import datetime
from pathlib import Path


LOG_PATH = Path(__file__).parent.parent / "improvement_log.md"

VALID_TYPES = ["SCALE", "ROOM", "AREA", "OPENING", "QTY", "RULE", "CRASH"]


def add_log_entry(
    project: str,
    issue_type: str,
    description: str,
    root_cause: str = "",
    resolution: str = "",
    status: str = "open",
) -> None:
    """
    Add entry to improvement log.

    Args:
        project: Project ID
        issue_type: Issue category
        description: What happened
        root_cause: Why (optional)
        resolution: How fixed (optional)
        status: open or resolved
    """
    if issue_type not in VALID_TYPES:
        print(f"ERROR: Invalid type '{issue_type}'. Valid types: {VALID_TYPES}")
        return

    date = datetime.now().strftime("%Y-%m-%d")

    # Read existing log
    with open(LOG_PATH, "r") as f:
        content = f.read()

    # Find the log entries section
    marker = "<!-- Add new entries below this line -->"
    if marker not in content:
        print("ERROR: Could not find marker in improvement_log.md")
        return

    # Build new entry
    entry = f"| {date} | {project} | {issue_type} | {description} | {root_cause or '-'} | {resolution or '-'} | {status} |"

    # Insert entry after marker
    new_content = content.replace(
        marker,
        f"{marker}\n{entry}"
    )

    # Update statistics
    new_content = _update_statistics(new_content)

    # Write back
    with open(LOG_PATH, "w") as f:
        f.write(new_content)

    print(f"Added issue to {LOG_PATH}")
    print(f"  Project: {project}")
    print(f"  Type: {issue_type}")
    print(f"  Description: {description}")


def _update_statistics(content: str) -> str:
    """Update the statistics section."""
    # Count entries by type
    counts = {t: 0 for t in VALID_TYPES}
    open_count = 0
    resolved_count = 0

    # Find all log entries (simple regex)
    entry_pattern = r"\| \d{4}-\d{2}-\d{2} \| [\w\-_]+ \| (\w+) \| .* \| .* \| .* \| (\w+) \|"
    for match in re.finditer(entry_pattern, content):
        issue_type = match.group(1)
        status = match.group(2)

        if issue_type in counts:
            counts[issue_type] += 1

        if status == "open":
            open_count += 1
        elif status == "resolved":
            resolved_count += 1

    total = sum(counts.values())

    # Update totals
    content = re.sub(
        r"- Total Issues: \d+",
        f"- Total Issues: {total}",
        content
    )
    content = re.sub(
        r"- Open Issues: \d+",
        f"- Open Issues: {open_count}",
        content
    )
    content = re.sub(
        r"- Resolved: \d+",
        f"- Resolved: {resolved_count}",
        content
    )

    # Update by category
    for issue_type, count in counts.items():
        content = re.sub(
            f"- {issue_type}: \\d+",
            f"- {issue_type}: {count}",
            content
        )

    # Update last updated
    today = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(
        r"\*Last Updated: \d{4}-\d{2}-\d{2}\*",
        f"*Last Updated: {today}*",
        content
    )

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Add issue to improvement log"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project ID"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=VALID_TYPES,
        help="Issue type"
    )
    parser.add_argument(
        "--desc",
        required=True,
        help="Issue description"
    )
    parser.add_argument(
        "--cause",
        default="",
        help="Root cause (optional)"
    )
    parser.add_argument(
        "--fix",
        default="",
        help="Resolution (optional)"
    )
    parser.add_argument(
        "--status",
        default="open",
        choices=["open", "resolved"],
        help="Status"
    )

    args = parser.parse_args()

    add_log_entry(
        project=args.project,
        issue_type=args.type,
        description=args.desc,
        root_cause=args.cause,
        resolution=args.fix,
        status=args.status,
    )


if __name__ == "__main__":
    main()
