"""Calibration-related APIs.

This package provides a stable namespace for calibration utilities without
moving existing modules yet. It re-exports functions from the current
module locations to avoid breaking imports while we reorganize.
"""

from ssbc.bootstrap import (
    bootstrap_calibration_uncertainty,
    plot_bootstrap_distributions,
)
from ssbc.conformal import (
    alpha_scan,
    compute_pac_operational_metrics,
    mondrian_conformal_calibrate,
    split_by_class,
)
from ssbc.cross_conformal import (
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
