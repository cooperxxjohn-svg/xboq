#!/usr/bin/env python3
"""
XBOQ Smoke Test

Quick validation that all modules can be imported and basic operations work.
Run before processing real drawings to catch config/setup issues.

Usage:
    python -m src.smoke_test
    python -m src.smoke_test --verbose
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_import(module_path: str) -> Tuple[bool, str]:
    """Try to import a module."""
    try:
        __import__(module_path)
        return True, "OK"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if a file exists."""
    if file_path.exists():
        return True, "OK"
    return False, f"Missing: {file_path}"


def check_rules_valid() -> Tuple[bool, str]:
    """Check if rules can be loaded."""
    import yaml
    rules_dir = Path(__file__).parent.parent / "rules"

    try:
        for yaml_file in rules_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                yaml.safe_load(f)
        return True, f"OK - {len(list(rules_dir.glob('*.yaml')))} rules loaded"
    except Exception as e:
        return False, f"YAML Error: {e}"


def check_opencv() -> Tuple[bool, str]:
    """Check OpenCV availability."""
    try:
        import cv2
        import numpy as np

        # Basic operation
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return True, f"OK - OpenCV {cv2.__version__}"
    except Exception as e:
        return False, f"Error: {e}"


def check_pdf_support() -> Tuple[bool, str]:
    """Check PDF processing support."""
    try:
        import fitz  # PyMuPDF
        return True, f"OK - PyMuPDF {fitz.version[0]}"
    except ImportError:
        try:
            import pdf2image
            return True, "OK - pdf2image available"
        except ImportError:
            return False, "No PDF library (need pymupdf or pdf2image)"


def check_output_dirs() -> Tuple[bool, str]:
    """Check output directories can be created."""
    try:
        base = Path(__file__).parent.parent / "output" / "_smoke_test"
        base.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file = base / "test.txt"
        test_file.write_text("smoke test")
        test_file.unlink()
        base.rmdir()

        return True, "OK - write access confirmed"
    except Exception as e:
        return False, f"Error: {e}"


def run_smoke_test(verbose: bool = False) -> bool:
    """
    Run all smoke tests.

    Args:
        verbose: Print detailed output

    Returns:
        True if all tests pass
    """
    print("=" * 60)
    print("XBOQ SMOKE TEST")
    print("=" * 60)
    print()

    tests = [
        ("Core imports", [
            ("src.ingest", "Ingest module"),
            ("src.scale", "Scale detection"),
            ("src.export", "Export module"),
        ]),
        ("Scope modules", [
            ("src.scope.register", "Scope register"),
            ("src.scope.completeness", "Completeness checker"),
            ("src.scope.evidence", "Evidence tracker"),
        ]),
        ("Risk modules", [
            ("src.risk.pricing", "Risk pricing"),
            ("src.risk.sensitivity", "Sensitivity analysis"),
            ("src.risk.quote_plan", "Quote planning"),
            ("src.risk.bid_strategy", "Bid strategy"),
        ]),
        ("Measurement modules", [
            ("src.measurement_rules.deductions", "Deductions engine"),
            ("src.measurement_rules.formwork", "Formwork deriver"),
            ("src.measurement_rules.prelims_from_qty", "Prelims calculator"),
        ]),
        ("BOQ modules", [
            ("src.boq.engine", "BOQ engine"),
            ("src.boq.export", "BOQ export"),
            ("src.boq.schema", "BOQ schema"),
        ]),
        ("Bid modules", [
            ("src.bid_docs.exclusions", "Exclusions generator"),
            ("src.bid_docs.clarifications", "Clarifications letter"),
        ]),
    ]

    all_passed = True
    failed_tests = []

    # Import tests
    for category, modules in tests:
        print(f"\n{category}:")
        for module_path, name in modules:
            passed, msg = check_import(module_path)
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {msg if verbose or not passed else 'OK'}")
            if not passed:
                all_passed = False
                failed_tests.append((name, msg))

    # File checks
    print(f"\nConfiguration files:")
    required_files = [
        Path(__file__).parent.parent / "rules" / "room_aliases.yaml",
        Path(__file__).parent.parent / "rules" / "scale_assumptions.yaml",
        Path(__file__).parent.parent / "rules" / "finish_templates.yaml",
        Path(__file__).parent.parent / "rules" / "measurement_rules.yaml",
        Path(__file__).parent.parent / "rules" / "rate_library.yaml",
        Path(__file__).parent.parent / "validation_template.csv",
    ]

    for fp in required_files:
        passed, msg = check_file_exists(fp)
        status = "✓" if passed else "✗"
        print(f"  {status} {fp.name}: {msg if verbose or not passed else 'OK'}")
        if not passed:
            all_passed = False
            failed_tests.append((fp.name, msg))

    # Rules validation
    print(f"\nRules validation:")
    passed, msg = check_rules_valid()
    status = "✓" if passed else "✗"
    print(f"  {status} YAML parsing: {msg}")
    if not passed:
        all_passed = False
        failed_tests.append(("Rules YAML", msg))

    # Dependencies
    print(f"\nDependencies:")

    passed, msg = check_opencv()
    status = "✓" if passed else "✗"
    print(f"  {status} OpenCV: {msg}")
    if not passed:
        all_passed = False
        failed_tests.append(("OpenCV", msg))

    passed, msg = check_pdf_support()
    status = "✓" if passed else "✗"
    print(f"  {status} PDF support: {msg}")
    if not passed:
        all_passed = False
        failed_tests.append(("PDF support", msg))

    # Output directories
    print(f"\nSystem:")
    passed, msg = check_output_dirs()
    status = "✓" if passed else "✗"
    print(f"  {status} Output directories: {msg}")
    if not passed:
        all_passed = False
        failed_tests.append(("Output dirs", msg))

    # Summary
    print()
    print("=" * 60)
    if all_passed:
        print("SMOKE TEST: PASSED ✓")
        print("All systems ready for processing.")
    else:
        print("SMOKE TEST: FAILED ✗")
        print(f"\nFailed tests ({len(failed_tests)}):")
        for name, msg in failed_tests:
            print(f"  - {name}: {msg}")
        print("\nFix these issues before processing drawings.")
    print("=" * 60)

    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XBOQ Smoke Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    success = run_smoke_test(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
