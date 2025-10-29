"""Calibration-related APIs.

This package provides conformal prediction, bootstrap uncertainty analysis,
and cross-conformal validation utilities.
"""

from .bootstrap import (
    bootstrap_calibration_uncertainty,
    plot_bootstrap_distributions,
)
from .conformal import (
    alpha_scan,
    compute_pac_operational_metrics,
    mondrian_conformal_calibrate,
    split_by_class,
)
from .cross_conformal import (
    cross_conformal_validation,
    print_cross_conformal_results,
)

__all__ = [
    "alpha_scan",
    "compute_pac_operational_metrics",
    "mondrian_conformal_calibrate",
    "split_by_class",
    "cross_conformal_validation",
    "print_cross_conformal_results",
    "bootstrap_calibration_uncertainty",
    "plot_bootstrap_distributions",
]
