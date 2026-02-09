"""
Floor Plan Room & Area Engine
Production-grade pipeline for Indian residential floor plans.
"""

__version__ = "1.0.0"
__author__ = "Floor Plan Engine"

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "plans"
OUTPUT_DIR = PROJECT_ROOT / "out"
RULES_DIR = PROJECT_ROOT / "rules"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
