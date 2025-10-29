"""Reporting and visualization APIs.

Re-export reporting and visualization helpers under a dedicated namespace.
"""

from ssbc.rigorous_report import generate_rigorous_pac_report
from ssbc.visualization import (
    plot_parallel_coordinates_plotly,
    report_prediction_stats,
)

__all__ = [
    "generate_rigorous_pac_report",
    "plot_parallel_coordinates_plotly",
    "report_prediction_stats",
]
