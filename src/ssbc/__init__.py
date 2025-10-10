"""Top-level package for SSBC (Small-Sample Beta Correction)."""

__author__ = """Petrus H Zwart"""
__email__ = "phzwart@lbl.gov"
__version__ = "0.1.0"

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

# Statistics utilities
from .statistics import (
    clopper_pearson_intervals,
    cp_interval,
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
    "cp_interval",
    # Simulation
    "BinaryClassifierSimulator",
    # Visualization
    "report_prediction_stats",
    "plot_parallel_coordinates_plotly",
    # Hyperparameter
    "sweep_hyperparams_and_collect",
    "sweep_and_plot_parallel_plotly",
]
