"""Operational metrics and uncertainty APIs.

This package re-exports metrics-related functions to provide a stable
namespace. The underlying modules remain in their current locations.
"""

from ssbc.loo_uncertainty import (
    compute_loo_corrected_prediction_bounds,
    compute_robust_prediction_bounds,
    estimate_loo_inflation_factor,
)
from ssbc.operational_bounds_simple import (
    compute_pac_operational_bounds_marginal,
    compute_pac_operational_bounds_perclass,
)

__all__ = [
    "compute_pac_operational_bounds_marginal",
    "compute_pac_operational_bounds_perclass",
    "estimate_loo_inflation_factor",
    "compute_loo_corrected_prediction_bounds",
    "compute_robust_prediction_bounds",
]
