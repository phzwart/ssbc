"""Unified bounds computation module for SSBC.

This module consolidates all statistical bounds computation functions
to reduce code duplication and provide a consistent API.
"""

from .statistical import (
    clopper_pearson_intervals,
    clopper_pearson_lower,
    clopper_pearson_upper,
    cp_interval,
    ensure_ci,
    prediction_bounds,
    prediction_bounds_beta_binomial,
)

__all__ = [
    "clopper_pearson_intervals",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "cp_interval",
    "ensure_ci",
    "prediction_bounds",
    "prediction_bounds_beta_binomial",
]
