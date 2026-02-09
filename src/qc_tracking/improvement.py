"""
Improvement Logger - Track issues and improvements.

Provides CLI and programmatic interface to log issues found during processing.

Usage (CLI):
    python -m src.qc.log_issue --plan_id X --tag wall_thickness_wrong --note "..."

Usage (programmatic):
    from src.qc import log_issue
    log_issue(plan_id="X", tag="wall_thickness_wrong", note="...")
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import re
import logging

logger = logging.getLogger(__name__)

# Valid issue tags
VALID_TAGS = [
    "scale_wrong",
    "wall_thickness_wrong",
    "door_missing",
    "window_missing",
    "room_label_wrong",
    "room_boundary_wrong",
    "finish_mapping_wrong",
    "opening_size_wrong",
    "steel_estimate_wrong",
    "area_calculation_wrong",
    "unmapped_cpwd",
    "config_issue",
    "cv_detection_issue",
    "other",
]

# Valid modules
VALID_MODULES = [
    "rooms",
    "walls",
    "openings",
    "finishes",
    "rates",
    "scale",
    "steel",
    "slab",
    "structural",
    "other",
]


@dataclass
class IssueEntry:
    """Single issue log entry."""
    plan_id: str
    date: str
    module: str
    tag: str
    what_wrong: str
    root_cause: Optional[str] = None
    fix_applied: Optional[str] = None
    before_metric: Optional[str] = None
    after_metric: Optional[str] = None
    status: str = "open"  # open, fixed, wontfix

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"### Issue: {self.plan_id}",
            f"",
            f"**Plan ID:** {self.plan_id}",
            f"**Date:** {self.date}",
            f"**Module:** {self.module}",
            f"**Tag:** {self.tag}",
            f"**Status:** {self.status}",
            f"",
            f"**What was wrong:**",
            f"{self.what_wrong}",
            f"",
        ]

        if self.root_cause:
            lines.extend([
                f"**Root cause hypothesis:**",
                f"{self.root_cause}",
                f"",
            ])

        if self.fix_applied:
            lines.extend([
                f"**Fix:**",
                f"{self.fix_applied}",
                f"",
            ])

        if self.before_metric or self.after_metric:
            lines.extend([
                f"**Before/After:**",
                f"- Before: {self.before_metric or 'N/A'}",
                f"- After: {self.after_metric or 'N/A'}",
                f"",
            ])

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "date": self.date,
            "module": self.module,
            "tag": self.tag,
            "what_wrong": self.what_wrong,
            "root_cause": self.root_cause,
            "fix_applied": self.fix_applied,
            "before_metric": self.before_metric,
            "after_metric": self.after_metric,
            "status": self.status,
        }


class IssueLogger:
    """
    Log and manage improvement issues.

    Appends entries to improvement_log.md and maintains a JSON index.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
    ):
        if log_path is None:
            log_path = Path(__file__).parent.parent.parent / "improvement_log.md"
        if index_path is None:
            index_path = Path(__file__).parent.parent.parent / ".improvement_index.json"

        self.log_path = log_path
        self.index_path = index_path
        self.entries: List[IssueEntry] = []

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load existing index from JSON."""
        try:
            if self.index_path.exists():
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                    self.entries = [
                        IssueEntry(**entry) for entry in data.get("entries", [])
                    ]
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            self.entries = []

    def _save_index(self) -> None:
        """Save index to JSON."""
        try:
            data = {
                "updated_at": datetime.now().isoformat(),
                "count": len(self.entries),
                "entries": [e.to_dict() for e in self.entries],
            }
            with open(self.index_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save index: {e}")

    def log(
        self,
        plan_id: str,
        tag: str,
        what_wrong: str,
        module: Optional[str] = None,
        root_cause: Optional[str] = None,
        fix_applied: Optional[str] = None,
        before_metric: Optional[str] = None,
        after_metric: Optional[str] = None,
    ) -> IssueEntry:
        """
        Log a new issue.

        Args:
            plan_id: Plan identifier
            tag: Issue tag (from VALID_TAGS)
            what_wrong: Description of the problem
            module: Module name (optional, inferred from tag)
            root_cause: Root cause hypothesis
            fix_applied: Fix description
            before_metric: Before metric
            after_metric: After metric

        Returns:
            Created IssueEntry
        """
        # Validate tag
        if tag not in VALID_TAGS:
            logger.warning(f"Unknown tag: {tag}")

        # Infer module from tag if not provided
        if module is None:
            module = self._infer_module(tag)

        # Create entry
        entry = IssueEntry(
            plan_id=plan_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            module=module,
            tag=tag,
            what_wrong=what_wrong,
            root_cause=root_cause,
            fix_applied=fix_applied,
            before_metric=before_metric,
            after_metric=after_metric,
            status="open" if not fix_applied else "fixed",
        )

        # Append to log file
        self._append_to_log(entry)

        # Add to index
        self.entries.append(entry)
        self._save_index()

        logger.info(f"Logged issue: {plan_id} - {tag}")

        return entry

    def _infer_module(self, tag: str) -> str:
        """Infer module from tag."""
        tag_module_map = {
            "scale_wrong": "scale",
            "wall_thickness_wrong": "walls",
            "door_missing": "openings",
            "window_missing": "openings",
            "room_label_wrong": "rooms",
            "room_boundary_wrong": "rooms",
            "finish_mapping_wrong": "finishes",
            "opening_size_wrong": "openings",
            "steel_estimate_wrong": "steel",
            "area_calculation_wrong": "rooms",
            "unmapped_cpwd": "rates",
            "config_issue": "other",
            "cv_detection_issue": "other",
        }
        return tag_module_map.get(tag, "other")

    def _append_to_log(self, entry: IssueEntry) -> None:
        """Append entry to markdown log."""
        try:
            with open(self.log_path, "a") as f:
                f.write("\n")
                f.write(entry.to_markdown())
        except Exception as e:
            logger.error(f"Could not append to log: {e}")

    def get_by_tag(self, tag: str) -> List[IssueEntry]:
        """Get issues by tag."""
        return [e for e in self.entries if e.tag == tag]

    def get_by_module(self, module: str) -> List[IssueEntry]:
        """Get issues by module."""
        return [e for e in self.entries if e.module == module]

    def get_by_plan(self, plan_id: str) -> List[IssueEntry]:
        """Get issues by plan ID."""
        return [e for e in self.entries if e.plan_id == plan_id]

    def get_open_issues(self) -> List[IssueEntry]:
        """Get all open issues."""
        return [e for e in self.entries if e.status == "open"]

    def mark_fixed(
        self,
        plan_id: str,
        tag: str,
        fix_applied: str,
        after_metric: Optional[str] = None,
    ) -> None:
        """Mark an issue as fixed."""
        for entry in self.entries:
            if entry.plan_id == plan_id and entry.tag == tag and entry.status == "open":
                entry.status = "fixed"
                entry.fix_applied = fix_applied
                if after_metric:
                    entry.after_metric = after_metric
                break

        self._save_index()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        by_tag: Dict[str, int] = {}
        by_module: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        for entry in self.entries:
            by_tag[entry.tag] = by_tag.get(entry.tag, 0) + 1
            by_module[entry.module] = by_module.get(entry.module, 0) + 1
            by_status[entry.status] = by_status.get(entry.status, 0) + 1

        return {
            "total": len(self.entries),
            "by_tag": by_tag,
            "by_module": by_module,
            "by_status": by_status,
            "open_count": by_status.get("open", 0),
            "fixed_count": by_status.get("fixed", 0),
        }


# Convenience functions
def log_issue(
    plan_id: str,
    tag: str,
    note: str,
    module: Optional[str] = None,
    root_cause: Optional[str] = None,
) -> IssueEntry:
    """
    Log an issue (convenience function).

    Args:
        plan_id: Plan identifier
        tag: Issue tag
        note: Description of the problem
        module: Module name (optional)
        root_cause: Root cause hypothesis

    Returns:
        Created IssueEntry
    """
    logger_instance = IssueLogger()
    return logger_instance.log(
        plan_id=plan_id,
        tag=tag,
        what_wrong=note,
        module=module,
        root_cause=root_cause,
    )


def get_issues_by_tag(tag: str) -> List[IssueEntry]:
    """Get issues by tag."""
    logger_instance = IssueLogger()
    return logger_instance.get_by_tag(tag)


def get_issues_by_module(module: str) -> List[IssueEntry]:
    """Get issues by module."""
    logger_instance = IssueLogger()
    return logger_instance.get_by_module(module)


# CLI entry point
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Log issues found during XBOQ processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Valid tags: {', '.join(VALID_TAGS)}
Valid modules: {', '.join(VALID_MODULES)}

Examples:
  python -m src.qc.log_issue --plan_id test_2bhk --tag room_label_wrong --note "Living labeled as Hall"
  python -m src.qc.log_issue --plan_id plan_001 --tag wall_thickness_wrong --note "230mm detected as 115mm" --module walls
  python -m src.qc.log_issue --summary
""",
    )

    parser.add_argument("--plan_id", help="Plan identifier")
    parser.add_argument("--tag", choices=VALID_TAGS, help="Issue tag")
    parser.add_argument("--note", help="Description of the problem")
    parser.add_argument("--module", choices=VALID_MODULES, help="Module name")
    parser.add_argument("--root_cause", help="Root cause hypothesis")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--list", action="store_true", help="List all issues")
    parser.add_argument("--list_tag", help="List issues by tag")
    parser.add_argument("--list_module", help="List issues by module")

    args = parser.parse_args()

    issue_logger = IssueLogger()

    if args.summary:
        summary = issue_logger.get_summary()
        print("\n=== Issue Summary ===")
        print(f"Total issues: {summary['total']}")
        print(f"Open: {summary['open_count']}")
        print(f"Fixed: {summary['fixed_count']}")
        print("\nBy Tag:")
        for tag, count in sorted(summary["by_tag"].items()):
            print(f"  {tag}: {count}")
        print("\nBy Module:")
        for module, count in sorted(summary["by_module"].items()):
            print(f"  {module}: {count}")

    elif args.list:
        print("\n=== All Issues ===")
        for entry in issue_logger.entries:
            status_icon = "✓" if entry.status == "fixed" else "○"
            print(f"{status_icon} [{entry.date}] {entry.plan_id} | {entry.tag} | {entry.what_wrong[:50]}...")

    elif args.list_tag:
        entries = issue_logger.get_by_tag(args.list_tag)
        print(f"\n=== Issues with tag '{args.list_tag}' ===")
        for entry in entries:
            print(f"  [{entry.date}] {entry.plan_id}: {entry.what_wrong[:50]}...")

    elif args.list_module:
        entries = issue_logger.get_by_module(args.list_module)
        print(f"\n=== Issues in module '{args.list_module}' ===")
        for entry in entries:
            print(f"  [{entry.date}] {entry.plan_id}: {entry.what_wrong[:50]}...")

    elif args.plan_id and args.tag and args.note:
        entry = issue_logger.log(
            plan_id=args.plan_id,
            tag=args.tag,
            what_wrong=args.note,
            module=args.module,
            root_cause=args.root_cause,
        )
        print(f"Logged issue: {entry.plan_id} - {entry.tag}")
        print(f"Entry added to {issue_logger.log_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
