"""Operational metrics and uncertainty APIs.

This package provides functions for computing operational bounds,
LOO-CV uncertainty quantification, and prediction bounds.
"""

from .loo_uncertainty import (
    compute_loo_corrected_prediction_bounds,
    compute_robust_prediction_bounds,
    estimate_loo_inflation_factor,
    format_prediction_bounds_report,
)
from .operational_bounds_simple import (
    compute_pac_operational_bounds_marginal,
    compute_pac_operational_bounds_marginal_loo_corrected,
    compute_pac_operational_bounds_perclass,
    compute_pac_operational_bounds_perclass_loo_corrected,
)

__all__ = [
    "compute_pac_operational_bounds_marginal",
    "compute_pac_operational_bounds_marginal_loo_corrected",
    "compute_pac_operational_bounds_perclass",
    "compute_pac_operational_bounds_perclass_loo_corrected",
    "estimate_loo_inflation_factor",
    "compute_loo_corrected_prediction_bounds",
    "compute_robust_prediction_bounds",
    "format_prediction_bounds_report",
]
