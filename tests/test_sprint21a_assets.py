"""
Sprint 21A Asset Existence Tests — verify docs, skills, and scripts exist.

These are lightweight smoke checks. They do NOT require benchmark tender data.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# A) Core doc assets
# =========================================================================

class TestDocAssets:
    """Verify documentation files exist and are non-empty."""

    def test_claude_md_exists(self):
        p = REPO_ROOT / "CLAUDE.md"
        assert p.exists(), "CLAUDE.md missing at repo root"
        assert p.stat().st_size > 100, "CLAUDE.md is suspiciously small"

    def test_architecture_md_exists(self):
        p = REPO_ROOT / "docs" / "ARCHITECTURE.md"
        assert p.exists(), "docs/ARCHITECTURE.md missing"
        assert p.stat().st_size > 100, "docs/ARCHITECTURE.md is suspiciously small"

    def test_debugging_md_exists(self):
        p = REPO_ROOT / "docs" / "DEBUGGING.md"
        assert p.exists(), "docs/DEBUGGING.md missing"
        assert p.stat().st_size > 100, "docs/DEBUGGING.md is suspiciously small"


# =========================================================================
# B) Skill files
# =========================================================================

EXPECTED_SKILLS = [
    "boq_schedule_debug.skill.md",
    "pilot_zip_ingest.skill.md",
    "regression_suite.skill.md",
    "ui_stability.skill.md",
    "performance_optim.skill.md",
    "demo_recording_mode.skill.md",
]


class TestSkillAssets:
    """Verify skill playbooks exist."""

    @pytest.mark.parametrize("skill_file", EXPECTED_SKILLS)
    def test_skill_exists(self, skill_file):
        p = REPO_ROOT / "prompts" / "skills" / skill_file
        assert p.exists(), f"prompts/skills/{skill_file} missing"
        assert p.stat().st_size > 50, f"{skill_file} is suspiciously small"


# =========================================================================
# C) Benchmark scaffolding
# =========================================================================

class TestBenchmarkScaffolding:
    """Verify benchmark template files exist."""

    def test_benchmarks_readme(self):
        p = REPO_ROOT / "benchmarks" / "README.md"
        assert p.exists(), "benchmarks/README.md missing"

    def test_manifest_example(self):
        p = REPO_ROOT / "benchmarks" / "manifest.example.json"
        assert p.exists(), "benchmarks/manifest.example.json missing"

    def test_expected_metrics_template(self):
        p = REPO_ROOT / "benchmarks" / "expected_metrics.yaml"
        assert p.exists(), "benchmarks/expected_metrics.yaml missing"

    def test_runs_dir(self):
        p = REPO_ROOT / "benchmarks" / "_runs"
        assert p.exists(), "benchmarks/_runs/ directory missing"
        assert p.is_dir(), "benchmarks/_runs should be a directory"


# =========================================================================
# D) Regression runner script
# =========================================================================

class TestRegressionRunner:
    """Verify regression runner works without manifest."""

    def test_script_exists(self):
        p = REPO_ROOT / "scripts" / "run_regression.py"
        assert p.exists(), "scripts/run_regression.py missing"

    def test_exits_zero_without_manifest(self):
        """When no manifest.json exists, script should print help and exit 0."""
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_regression.py"),
             "--manifest", "/nonexistent/manifest.json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"run_regression.py should exit 0 when manifest missing.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "manifest" in result.stdout.lower() or "setup" in result.stdout.lower(), (
            "run_regression.py should print setup instructions when manifest missing"
        )

    def test_help_flag(self):
        """--help should work."""
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_regression.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, "run_regression.py --help should exit 0"
