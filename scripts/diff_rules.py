#!/usr/bin/env python3
"""
Rules Diff Tool

Compare rule files between two snapshots or between a snapshot and current rules.

Usage:
    python scripts/diff_rules.py --from v1 --to v2
    python scripts/diff_rules.py --from v1 --to current
    python scripts/diff_rules.py --from v1  # compares to current

Shows:
    - New files
    - Deleted files
    - Modified files (with line-by-line diff)
"""

import argparse
import difflib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
RULES_DIR = PROJECT_ROOT / "rules"
SNAPSHOTS_DIR = RULES_DIR / "snapshots"


def get_rules_dir(tag: str) -> Path:
    """Get the rules directory for a tag (or 'current')."""
    if tag.lower() == "current":
        return RULES_DIR
    else:
        snap_dir = SNAPSHOTS_DIR / tag
        if not snap_dir.exists():
            print(f"ERROR: Snapshot '{tag}' not found at {snap_dir}")
            sys.exit(1)
        return snap_dir


def get_rule_files(rules_dir: Path) -> dict:
    """Get all rule files in a directory as {name: content}."""
    files = {}
    for f in rules_dir.glob("*.yaml"):
        if "snapshots" not in str(f):
            files[f.name] = f.read_text()
    for f in rules_dir.glob("*.yml"):
        if "snapshots" not in str(f):
            files[f.name] = f.read_text()
    return files


def diff_rules(from_tag: str, to_tag: str, context_lines: int = 3) -> None:
    """
    Compare rules between two tags.

    Args:
        from_tag: Source tag (e.g., v1)
        to_tag: Target tag (e.g., v2, current)
        context_lines: Number of context lines in diff
    """
    from_dir = get_rules_dir(from_tag)
    to_dir = get_rules_dir(to_tag)

    from_files = get_rule_files(from_dir)
    to_files = get_rule_files(to_dir)

    from_names = set(from_files.keys())
    to_names = set(to_files.keys())

    # New files
    new_files = to_names - from_names
    # Deleted files
    deleted_files = from_names - to_names
    # Common files (check for changes)
    common_files = from_names & to_names

    has_changes = False

    # Report new files
    if new_files:
        has_changes = True
        print("=" * 60)
        print(f"NEW FILES (in {to_tag}):")
        print("=" * 60)
        for f in sorted(new_files):
            print(f"  + {f}")
        print()

    # Report deleted files
    if deleted_files:
        has_changes = True
        print("=" * 60)
        print(f"DELETED FILES (removed from {to_tag}):")
        print("=" * 60)
        for f in sorted(deleted_files):
            print(f"  - {f}")
        print()

    # Report modified files
    modified_files = []
    for name in sorted(common_files):
        from_content = from_files[name]
        to_content = to_files[name]

        if from_content != to_content:
            modified_files.append(name)

    if modified_files:
        has_changes = True
        print("=" * 60)
        print(f"MODIFIED FILES ({from_tag} -> {to_tag}):")
        print("=" * 60)

        for name in modified_files:
            print(f"\n--- {name} ({from_tag})")
            print(f"+++ {name} ({to_tag})")
            print()

            from_lines = from_files[name].splitlines(keepends=True)
            to_lines = to_files[name].splitlines(keepends=True)

            diff = difflib.unified_diff(
                from_lines, to_lines,
                fromfile=f"{from_tag}/{name}",
                tofile=f"{to_tag}/{name}",
                n=context_lines
            )

            diff_lines = list(diff)
            if diff_lines:
                # Count changes
                additions = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
                deletions = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))

                print(f"  Changes: +{additions} -{deletions} lines")
                print()

                for line in diff_lines[2:]:  # Skip header lines
                    if line.startswith('+'):
                        print(f"\033[92m{line}\033[0m", end='')
                    elif line.startswith('-'):
                        print(f"\033[91m{line}\033[0m", end='')
                    elif line.startswith('@@'):
                        print(f"\033[96m{line}\033[0m", end='')
                    else:
                        print(line, end='')

            print()

    if not has_changes:
        print(f"No differences found between {from_tag} and {to_tag}")
    else:
        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  New files: {len(new_files)}")
        print(f"  Deleted files: {len(deleted_files)}")
        print(f"  Modified files: {len(modified_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare rule files between snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/diff_rules.py --from v1 --to v2
    python scripts/diff_rules.py --from v1 --to current
    python scripts/diff_rules.py --from v1
        """
    )

    parser.add_argument(
        "--from", "-f",
        dest="from_tag",
        required=True,
        help="Source snapshot tag"
    )

    parser.add_argument(
        "--to", "-t",
        dest="to_tag",
        default="current",
        help="Target snapshot tag (default: current)"
    )

    parser.add_argument(
        "--context", "-c",
        type=int,
        default=3,
        help="Number of context lines in diff (default: 3)"
    )

    args = parser.parse_args()

    print(f"Comparing rules: {args.from_tag} -> {args.to_tag}")
    print()
    diff_rules(args.from_tag, args.to_tag, args.context)


if __name__ == "__main__":
    main()
