"""Unified bounds computation module for SSBC.

This module consolidates all statistical bounds computation functions
to reduce code duplication and provide a consistent API.
"""

from .statistical import (
    clopper_pearson_intervals,
    clopper_pearson_lower,
    clopper_pearson_upper,
    compute_all_bounds,
    cp_interval,
    prediction_bounds,
)

__all__ = [
    "clopper_pearson_intervals",
    "clopper_pearson_lower", 
    "clopper_pearson_upper",
    "compute_all_bounds",
    "cp_interval",
    "prediction_bounds",
]
