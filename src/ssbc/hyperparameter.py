"""Hyperparameter sweep and optimization for Mondrian conformal prediction."""

import itertools
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from ssbc.calibration import mondrian_conformal_calibrate
from ssbc.reporting import plot_parallel_coordinates_plotly, report_prediction_stats


def sweep_hyperparams_and_collect(
    class_data: dict[int, dict[str, Any]],
    alpha_0: np.ndarray,
    delta_0: np.ndarray,
    alpha_1: np.ndarray,
    delta_1: np.ndarray,
    mode: Literal["beta", "beta-binomial"] = "beta",
    extra_metrics: dict[str, Callable] | None = None,
    quiet: bool = True,
) -> pd.DataFrame:
    """Sweep (a0,d0,a1,d1), run mondrian_conformal_calibrate + report_prediction_stats,
    and return a tidy DataFrame with hyperparams + selected metrics.

    This function performs a grid search over hyperparameter combinations and
    evaluates the resulting conformal prediction performance.

    Parameters
    ----------
    class_data : dict
        Output from split_by_class()
    alpha_0 : array-like
        Grid of alpha values for class 0
    delta_0 : array-like
        Grid of delta values for class 0
    alpha_1 : array-like
        Grid of alpha values for class 1
    delta_1 : array-like
        Grid of delta values for class 1
    mode : str, default="beta"
        "beta" or "beta-binomial" mode for SSBC
    extra_metrics : dict of {name: function}, optional
        Additional metrics to compute. Each function takes the summary dict
        and returns a scalar value.
    quiet : bool, default=True
        If True, suppress progress output

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with one row per hyperparameter combination.
        Columns include:
        - a0, d0, a1, d1: hyperparameters
        - cov: overall coverage rate
        - sing_rate: singleton prediction rate
        - err_all: overall singleton error rate
        - err_pred0, err_pred1: errors by predicted class
        - err_y0, err_y1: errors by true class
        - esc_rate: escalation rate (doublets + abstentions)
        - n_total, sing_count, m_abst, m_doublets: counts
        - Any additional metrics from extra_metrics

    Examples
    --------
    >>> import numpy as np
    >>> from ssbc import BinaryClassifierSimulator, split_by_class
    >>>
    >>> # Generate data
    >>> sim = BinaryClassifierSimulator(0.1, (2, 8), (8, 2), seed=42)
    >>> labels, probs = sim.generate(1000)
    >>> class_data = split_by_class(labels, probs)
    >>>
    >>> # Define grid
    >>> alpha_grid = np.arange(0.05, 0.20, 0.05)
    >>> delta_grid = np.arange(0.05, 0.20, 0.05)
    >>>
    >>> # Run sweep
    >>> df = sweep_hyperparams_and_collect(
    ...     class_data,
    ...     alpha_0=alpha_grid, delta_0=delta_grid,
    ...     alpha_1=alpha_grid, delta_1=delta_grid,
    ... )
    >>>
    >>> # Analyze results
    >>> print(df[['a0', 'a1', 'cov', 'sing_rate', 'err_all']].head())

    Notes
    -----
    The function performs a complete grid search, so the total number of
    evaluations is len(alpha_0) × len(delta_0) × len(alpha_1) × len(delta_1).
    For large grids, this can be computationally expensive.
    """
    rows = []
    combos = list(itertools.product(alpha_0, delta_0, alpha_1, delta_1))

    for a0, d0, a1, d1 in combos:
        if not quiet:
            print(f"a0={a0:.3f}, d0={d0:.3f}, a1={a1:.3f}, d1={d1:.3f}")

        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=class_data,
            alpha_target={0: float(a0), 1: float(a1)},
            delta={0: float(d0), 1: float(d1)},
            mode=mode,
        )
        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        # Robust getter
        def g(d, *keys, default=None):
            """Navigate nested dict safely."""
            cur = d
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        n_total = int(g(summary, "marginal", "n_total", default=0) or 0)
        cov = float(g(summary, "marginal", "coverage", "rate", default=0.0) or 0.0)
        sing_rate = float(g(summary, "marginal", "singletons", "rate", default=0.0) or 0.0)
        sing_cnt = int(g(summary, "marginal", "singletons", "count", default=0) or 0)
        abst_cnt = int(g(summary, "marginal", "abstentions", "count", default=0) or 0)
        doub_cnt = int(g(summary, "marginal", "doublets", "count", default=0) or 0)
        esc_rate = (abst_cnt + doub_cnt) / float(n_total if n_total else 1)

        err_all = float(g(summary, "marginal", "singletons", "errors", "rate", default=0.0) or 0.0)
        err_p0 = float(g(summary, "marginal", "singletons", "errors_by_pred", "pred_0", "rate", default=0.0) or 0.0)
        err_p1 = float(g(summary, "marginal", "singletons", "errors_by_pred", "pred_1", "rate", default=0.0) or 0.0)

        err_y0 = float(g(summary, 0, "singletons", "error_given_singleton", "rate", default=0.0) or 0.0)
        err_y1 = float(g(summary, 1, "singletons", "error_given_singleton", "rate", default=0.0) or 0.0)

        row = {
            "a0": float(a0),
            "d0": float(d0),
            "a1": float(a1),
            "d1": float(d1),
            "cov": cov,
            "sing_rate": sing_rate,
            "err_all": err_all,
            "err_pred0": err_p0,
            "err_pred1": err_p1,
            "err_y0": err_y0,
            "err_y1": err_y1,
            "esc_rate": esc_rate,
            "n_total": int(n_total),
            "sing_count": int(sing_cnt),
            "m_abst": abst_cnt,
            "m_doublets": doub_cnt,
        }

        if extra_metrics:
            for name, fn in extra_metrics.items():
                try:
                    row[name] = fn(summary)
                except Exception:
                    row[name] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["a0", "d0", "a1", "d1"], kind="mergesort").reset_index(drop=True)


def sweep_and_plot_parallel_plotly(
    class_data: dict[int, dict[str, Any]],
    delta_0: np.ndarray,
    delta_1: np.ndarray,
    alpha_0: np.ndarray,
    alpha_1: np.ndarray,
    mode: Literal["beta", "beta-binomial"] = "beta",
    extra_metrics: dict[str, Callable] | None = None,
    color: str = "err_all",
    color_continuous_scale=None,
    title: str | None = None,
    height: int = 600,
):
    """Convenience wrapper: run sweep + show plotly parallel coordinates figure.

    This function combines hyperparameter sweep and visualization in one call.

    Parameters
    ----------
    class_data : dict
        Output from split_by_class()
    delta_0, delta_1 : array-like
        Grid of delta values for classes 0 and 1
    alpha_0, alpha_1 : array-like
        Grid of alpha values for classes 0 and 1
    mode : str, default="beta"
        "beta" or "beta-binomial" mode for SSBC
    extra_metrics : dict of {name: function}, optional
        Additional metrics to compute
    color : str, default='err_all'
        Column to use for coloring the parallel coordinates
    color_continuous_scale : plotly colorscale, optional
        Color scale for the plot
    title : str, optional
        Plot title (defaults to auto-generated title)
    height : int, default=600
        Plot height in pixels

    Returns
    -------
    df : pd.DataFrame
        Results dataframe
    fig : plotly.graph_objects.Figure
        Interactive parallel coordinates plot

    Examples
    --------
    >>> import numpy as np
    >>> from ssbc import BinaryClassifierSimulator, split_by_class
    >>>
    >>> # Generate data
    >>> sim = BinaryClassifierSimulator(0.1, (2, 8), (8, 2), seed=42)
    >>> labels, probs = sim.generate(1000)
    >>> class_data = split_by_class(labels, probs)
    >>>
    >>> # Run sweep and plot
    >>> df, fig = sweep_and_plot_parallel_plotly(
    ...     class_data,
    ...     delta_0=np.arange(0.05, 0.20, 0.05),
    ...     delta_1=np.arange(0.05, 0.20, 0.05),
    ...     alpha_0=np.arange(0.05, 0.20, 0.05),
    ...     alpha_1=np.arange(0.05, 0.20, 0.05),
    ...     color='err_all'
    ... )
    >>> fig.show()  # Display in notebook
    >>> # Or save: fig.write_html("sweep_results.html")

    Notes
    -----
    The parallel coordinates plot allows interactive exploration of the
    hyperparameter space. You can brush (select) ranges on any axis to
    filter configurations and see their impact on other metrics.
    """
    df = sweep_hyperparams_and_collect(
        class_data=class_data,
        alpha_0=alpha_0,
        delta_0=delta_0,
        alpha_1=alpha_1,
        delta_1=delta_1,
        mode=mode,
        extra_metrics=extra_metrics,
        quiet=True,
    )

    if title is None:
        title = f"Mondrian Hyperparameter Sweep (n={len(df)} configs)"

    fig = plot_parallel_coordinates_plotly(
        df, color=color, color_continuous_scale=color_continuous_scale, title=title, height=height
    )

    return df, fig
