"""Top-level package for SSBC (Small-Sample Beta Correction)."""

from importlib.metadata import version

__author__ = """Petrus H Zwart"""
__email__ = "phzwart@lbl.gov"
__version__ = version("ssbc")  # Read from package metadata (pyproject.toml)

# Core SSBC algorithm
# Conformal prediction
from .conformal import (
    mondrian_conformal_calibrate,
    split_by_class,
)
from .core import (
    SSBCResult,
    ssbc_correct,
)

# Hyperparameter tuning
from .hyperparameter import (
    sweep_and_plot_parallel_plotly,
    sweep_hyperparams_and_collect,
)

# Simulation (for testing and examples)
from .simulation import (
    BinaryClassifierSimulator,
)

# SLA (Service Level Agreement) - operational rate bounds
from .sla import (
    OperationalRateBounds,
    OperationalRateBoundsResult,
    compute_marginal_operational_bounds,
    compute_mondrian_operational_bounds,
    cross_fit_cp_bounds,
)

# Statistics utilities
from .statistics import (
    clopper_pearson_intervals,
    clopper_pearson_lower,
    clopper_pearson_upper,
    cp_interval,
)

# Utility functions
from .utils import (
    compute_operational_rate,
)

# Visualization and reporting
from .visualization import (
    plot_parallel_coordinates_plotly,
    report_prediction_stats,
)

__all__ = [
    # Core
    "SSBCResult",
    "ssbc_correct",
    # Conformal
    "mondrian_conformal_calibrate",
    "split_by_class",
    # Statistics
    "clopper_pearson_intervals",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "cp_interval",
    # Utilities
    "compute_operational_rate",
    # SLA - Operational rate bounds
    "OperationalRateBounds",
    "OperationalRateBoundsResult",
    "compute_marginal_operational_bounds",
    "compute_mondrian_operational_bounds",
    "cross_fit_cp_bounds",
    # Simulation
    "BinaryClassifierSimulator",
    # Visualization
    "report_prediction_stats",
    "plot_parallel_coordinates_plotly",
    # Hyperparameter
    "sweep_hyperparams_and_collect",
    "sweep_and_plot_parallel_plotly",
]
