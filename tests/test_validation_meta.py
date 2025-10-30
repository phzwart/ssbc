import io
import sys

import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    get_calibration_bounds_dataframe,
    plot_calibration_excess,
    plot_validation_bounds,
    print_validation_results,
    validate_pac_bounds,
    validate_prediction_interval_calibration,
)


def make_small_sim(seed: int = 123) -> BinaryClassifierSimulator:
    return BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=seed)


def test_meta_validation_end_to_end() -> None:
    sim = make_small_sim()
    # Tiny configuration for speed
    results = validate_prediction_interval_calibration(
        simulator=sim,
        n_calibration=30,
        BigN=3,
        test_size=40,
        n_trials=5,
        ci_level=0.95,
        prediction_method="analytical",
        use_loo_correction=True,
        verbose=False,
    )

    # Results contain expected top-level keys
    assert results["n_calibrations"] == 3
    assert results["n_calibration"] == 30
    assert results["ci_level"] == 0.95

    # Extract DataFrame and ensure expected columns exist
    df = get_calibration_bounds_dataframe(results)
    assert {"calibration_idx", "scope", "metric", "observed_q05", "observed_q95"}.issubset(df.columns)

    # Plot excess (returns fig when requested)
    fig = plot_calibration_excess(df, scope="marginal", metric="singleton", return_fig=True)
    assert fig is not None


def test_print_validation_results_contains_legacy_markers() -> None:
    sim = make_small_sim(seed=321)
    labels, probs = sim.generate(40)

    report = {
        "calibration_result": {
            0: {"threshold": 0.2, "n": int(np.sum(labels == 0))},
            1: {"threshold": 0.8, "n": int(np.sum(labels == 1))},
        },
        "parameters": {"ci_level": 0.95, "pac_level_marginal": 0.9, "pac_level_0": 0.9, "pac_level_1": 0.9},
        "pac_bounds_marginal": {
            "singleton_rate_bounds": (0.0, 1.0),
            "doublet_rate_bounds": (0.0, 1.0),
            "abstention_rate_bounds": (0.0, 1.0),
            "singleton_error_rate_bounds": (0.0, 1.0),
        },
        "pac_bounds_class_0": {
            "singleton_rate_bounds": (0.0, 1.0),
            "doublet_rate_bounds": (0.0, 1.0),
            "abstention_rate_bounds": (0.0, 1.0),
            "singleton_error_rate_bounds": (0.0, 1.0),
        },
        "pac_bounds_class_1": {
            "singleton_rate_bounds": (0.0, 1.0),
            "doublet_rate_bounds": (0.0, 1.0),
            "abstention_rate_bounds": (0.0, 1.0),
            "singleton_error_rate_bounds": (0.0, 1.0),
        },
    }

    validation = validate_pac_bounds(report=report, simulator=sim, test_size=20, n_trials=5, verbose=False)

    # Capture output
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        print_validation_results(validation)
    finally:
        sys.stdout = old
    out = buf.getvalue()

    # Legacy and new markers both present
    assert "PREDICTION INTERVAL VALIDATION RESULTS" in out
    assert "PAC BOUNDS VALIDATION RESULTS" in out
    assert "Coverage:" in out


def test_plot_validation_bounds_return_fig() -> None:
    sim = make_small_sim(seed=99)
    labels, probs = sim.generate(50)

    from ssbc.reporting import generate_rigorous_pac_report

    report = generate_rigorous_pac_report(
        labels=labels,
        probs=probs,
        alpha_target=0.1,
        delta=0.1,
        test_size=25,
        ci_level=0.95,
        prediction_method="analytical",
        use_loo_correction=True,
        verbose=False,
    )

    validation = validate_pac_bounds(report=report, simulator=sim, test_size=25, n_trials=5, verbose=False)
    figs = plot_validation_bounds(validation, metric="singleton", return_figs=True)
    assert figs is not None
