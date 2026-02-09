"""
Synthetic Floor Plan Generator Module.

Generates synthetic test floor plans with known ground truth for:
- Unit testing
- Benchmark dataset augmentation
- Algorithm validation
"""

from .generate_plans import (
    SyntheticRoom,
    SyntheticPlan,
    SyntheticPlanGenerator,
    generate_simple_rectangle,
    generate_1bhk,
    generate_2bhk,
    generate_3bhk,
    generate_indian_apartment,
    generate_all_plans,
)

__all__ = [
    "SyntheticRoom",
    "SyntheticPlan",
    "SyntheticPlanGenerator",
    "generate_simple_rectangle",
    "generate_1bhk",
    "generate_2bhk",
    "generate_3bhk",
    "generate_indian_apartment",
    "generate_all_plans",
]
