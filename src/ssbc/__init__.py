"""Top-level package for SSBC (Small-Sample Beta Correction)."""

from importlib.metadata import version

__author__ = """Petrus H Zwart"""
__email__ = "phzwart@lbl.gov"
__version__ = version("ssbc")  # Read from package metadata (pyproject.toml)

# Core SSBC algorithm
# Conformal prediction
# Bootstrap uncertainty analysis
# Statistics utilities (now in bounds module)
from ssbc.bounds import (
    clopper_pearson_intervals,
    clopper_pearson_lower,
    clopper_pearson_upper,
    cp_interval,
    prediction_bounds,
)

# Cross-conformal validation
from ssbc.calibration import (
    alpha_scan,
    bootstrap_calibration_uncertainty,
    compute_pac_operational_metrics,
    cross_conformal_validation,
    mondrian_conformal_calibrate,
    plot_bootstrap_distributions,
    print_cross_conformal_results,
    split_by_class,
)
from ssbc.core_pkg import (
    SSBCResult,
    ssbc_correct,
)

# Hyperparameter tuning
from ssbc.hyperparameter import (
    sweep_and_plot_parallel_plotly,
    sweep_hyperparams_and_collect,
)

# LOO uncertainty quantification
from ssbc.metrics import (
    compute_robust_prediction_bounds,
    format_prediction_bounds_report,
)

# Visualization and reporting
from ssbc.reporting import (
    generate_rigorous_pac_report,
    plot_parallel_coordinates_plotly,
    report_prediction_stats,
)

# Simulation (for testing and examples)
from ssbc.simulation import (
    BinaryClassifierSimulator,
)

# Utility functions
from ssbc.utils import (
    compute_operational_rate,
    evaluate_test_dataset,
)

# Validation utilities
from ssbc.validation_pkg import (
    get_calibration_bounds_dataframe,
    plot_calibration_excess,
    plot_validation_bounds,
    print_calibration_validation_results,
    print_validation_results,
    validate_pac_bounds,
    validate_prediction_interval_calibration,
)

__all__ = [
    # Core
    "SSBCResult",
    "ssbc_correct",
    # Conformal
    "alpha_scan",
    "compute_pac_operational_metrics",
    "mondrian_conformal_calibrate",
    "split_by_class",
    # Statistics
    "clopper_pearson_intervals",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "prediction_bounds",
    "compute_robust_prediction_bounds",
    "format_prediction_bounds_report",
    "cp_interval",
    # Utilities
    "compute_operational_rate",
    "evaluate_test_dataset",
    # Simulation
    "BinaryClassifierSimulator",
    # Visualization
    "report_prediction_stats",
    "plot_parallel_coordinates_plotly",
    # Bootstrap uncertainty
    "bootstrap_calibration_uncertainty",
    "plot_bootstrap_distributions",
    # Cross-conformal validation
    "cross_conformal_validation",
    "print_cross_conformal_results",
    # Validation utilities
    "validate_pac_bounds",
    "print_validation_results",
    "plot_validation_bounds",
    "validate_prediction_interval_calibration",
    "print_calibration_validation_results",
    "get_calibration_bounds_dataframe",
    "plot_calibration_excess",
    # Rigorous reporting
    "generate_rigorous_pac_report",
    # Hyperparameter
    "sweep_hyperparams_and_collect",
    "sweep_and_plot_parallel_plotly",
]
