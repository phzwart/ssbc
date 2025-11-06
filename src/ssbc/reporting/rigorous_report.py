"""Unified rigorous reporting with full PAC guarantees.

This module provides a single comprehensive report that properly accounts for
coverage volatility across all operational metrics.
"""

from datetime import datetime
from typing import Any, cast

import numpy as np
import scipy

from ssbc import __version__
from ssbc._logging import get_logger
from ssbc.calibration import mondrian_conformal_calibrate, split_by_class
from ssbc.core_pkg import ssbc_correct
from ssbc.metrics import (
    compute_pac_operational_bounds_marginal_loo_corrected,
    compute_pac_operational_bounds_perclass_loo_corrected,
)

logger = get_logger(__name__)


def generate_rigorous_pac_report(
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: float | dict[int, float] = 0.10,
    delta: float | dict[int, float] = 0.10,
    test_size: int | None = None,
    ci_level: float = 0.95,
    use_union_bound: bool = False,
    n_jobs: int = -1,
    verbose: bool = True,
    prediction_method: str = "exact",
    use_loo_correction: bool = True,
    loo_inflation_factor: float | None = None,
) -> dict[str, Any]:
    """Generate complete rigorous PAC report with coverage volatility.

    This is the UNIFIED function that gives you everything properly:
    - SSBC-corrected thresholds
    - Coverage guarantees
    - PAC-controlled operational bounds (marginal + per-class)
    - Singleton error rates with PAC guarantees
    - All bounds account for coverage volatility via BetaBinomial

    Parameters
    ----------
    labels : np.ndarray, shape (n,)
        True labels (0 or 1)
    probs : np.ndarray, shape (n, 2)
        Predicted probabilities [P(class=0), P(class=1)]
    alpha_target : float or dict[int, float], default=0.10
        Target miscoverage per class
    delta : float or dict[int, float], default=0.10
        PAC risk tolerance. Used for both:
        - Coverage guarantee (via SSBC)
        - Operational bounds (pac_level = 1 - delta)
    test_size : int, optional
        Expected test set size. If None, uses calibration size
    ci_level : float, default=0.95
        Confidence level for prediction bounds
    prediction_method : str, default="hoeffding"
        Method for LOO uncertainty quantification (when use_loo_correction=True):
        - "auto": Automatically select best method
        - "analytical": Method 1 (recommended for n>=40)
        - "exact": Method 2 (recommended for n=20-40)
        - "hoeffding": Method 3 (ultra-conservative, default)
        - "all": Compare all methods
        When use_loo_correction=False, this parameter is ignored.
    use_loo_correction : bool, default=False
        If True, uses LOO-CV uncertainty correction for small samples (n=20-40).
        This accounts for all four sources of uncertainty:
        1. LOO-CV correlation structure (variance inflation ≈2×)
        2. Threshold calibration uncertainty
        3. Parameter estimation uncertainty
        4. Test sampling uncertainty
        Recommended for small calibration sets where standard bounds may be too narrow.

        **LOO-CV Correlation Issue**: The critical challenge with LOO-CV is that the N
        LOO predictions are not independent. The training sets for different folds overlap
        substantially—folds i and j using training sets D_{-i} and D_{-j} differ by only
        two examples out of N−1. Because each fold's threshold is computed from nearly
        identical data, the resulting predictions exhibit strong positive correlation.
        This correlation structure is handled through specialized LOO-corrected methods
        that account for the dependency between folds when computing diagnostic bounds.
    loo_inflation_factor : float, optional
        Manual override for LOO variance inflation factor. If None (default),
        automatically estimated from the data using empirical variance.

        **Empirical Correction Factor Estimation**: The inflation factor is estimated by
        comparing the empirical variance of LOO predictions to the theoretical IID variance.
        Specifically, inflation = (Var_empirical / Var_IID) × (n / (n-1)), where
        Var_empirical is the sample variance of the binary LOO predictions (with Bessel's
        correction), Var_IID = p̂(1-p̂) is the expected variance under independence, and
        the n/(n-1) factor accounts for the finite-sample bias correction. For large n,
        this approaches the theoretical value of 2.0, but for small samples (n=20-40),
        the actual inflation can vary. The estimated factor is clipped to [1.0, 6.0] to
        prevent extreme values from outliers or numerical instability.

        Typical values:
        - 1.0: No inflation (assumes independent samples - usually wrong for LOO)
        - 2.0: Standard LOO inflation (theoretical value for n→∞)
        - 1.5-2.5: Empirical range for small samples
        - >2.5: High correlation scenarios
        - Up to 6.0: Extended range for very high correlation scenarios

        Note: This parameter can be used as a phenomenological control knob to
        correct for issues not modeled properly in the statistical framework.
        For example, if validation suggests the default estimation is too optimistic
        or too conservative, manually adjusting this factor can help achieve desired
        coverage behavior. Use with caution and validate empirically.
    use_union_bound : bool, default=False
        Apply Bonferroni for simultaneous guarantees
    n_jobs : int, default=-1
        Number of parallel jobs for LOO-CV computation.
        -1 = use all cores (default), 1 = single-threaded, N = use N cores.
    verbose : bool, default=True
        Print comprehensive report

    Returns
    -------
    dict
        Complete report with keys:
        - 'ssbc_class_0': SSBCResult for class 0
        - 'ssbc_class_1': SSBCResult for class 1
        - 'pac_bounds_marginal': PAC operational bounds (marginal)
        - 'pac_bounds_class_0': PAC operational bounds (class 0)
        - 'pac_bounds_class_1': PAC operational bounds (class 1)
        - 'calibration_result': From mondrian_conformal_calibrate
        - 'prediction_stats': From mondrian_conformal_calibrate

    Examples
    --------
    >>> from ssbc import BinaryClassifierSimulator
    >>> from ssbc.rigorous_report import generate_rigorous_pac_report
    >>>
    >>> sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
    >>> labels, probs = sim.generate(n_samples=1000)
    >>>
    >>> report = generate_rigorous_pac_report(
    ...     labels, probs,
    ...     alpha_target=0.10,
    ...     delta=0.10,
    ...     verbose=True
    ... )

    Notes
    -----
    **This replaces the old workflow (removed in v1.1.0):**

    OLD (removed - these functions no longer exist):
    ```python
    # These functions were removed in v1.1.0:
    # op_bounds = compute_mondrian_operational_bounds(...)  # Removed
    # marginal_bounds = compute_marginal_operational_bounds(...)  # Removed
    # report_prediction_stats(...)  # Removed
    ```

    NEW (rigorous):
    ```python
    report = generate_rigorous_pac_report(labels, probs, alpha_target, delta)
    # Done! All bounds account for coverage volatility.
    ```
    """
    # Comprehensive input validation
    logger.info("Starting rigorous PAC report generation")

    # Validate labels
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"labels must be a numpy array, got {type(labels).__name__}")
    if len(labels) < 2:
        raise ValueError(f"Need at least 2 calibration samples, got {len(labels)}")
    if labels.dtype.kind not in ("i", "u"):
        raise ValueError(f"labels must be integer array, got dtype {labels.dtype}")
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"labels must contain only 0 and 1, found {unique_labels.tolist()}. "
            "This function is for binary classification only."
        )

    # Validate probs
    if not isinstance(probs, np.ndarray):
        raise TypeError(f"probs must be a numpy array, got {type(probs).__name__}")
    if probs.shape != (len(labels), 2):
        raise ValueError(
            f"probs must have shape ({len(labels)}, 2), got {probs.shape}. "
            "Each row should contain [P(class=0), P(class=1)]."
        )
    if np.any((probs < 0) | (probs > 1)):
        invalid_mask = (probs < 0) | (probs > 1)
        invalid_count = np.sum(invalid_mask)
        raise ValueError(
            f"All probabilities must be in [0,1], found {invalid_count} invalid values. "
            f"Invalid range: [{np.min(probs[invalid_mask]):.4f}, {np.max(probs[invalid_mask]):.4f}]"
        )
    if np.any(np.isnan(probs)):
        nan_count = np.sum(np.isnan(probs))
        raise ValueError(f"probs must not contain NaN values, found {nan_count} NaNs")
    if np.any(np.isinf(probs)):
        inf_count = np.sum(np.isinf(probs))
        raise ValueError(f"probs must not contain Inf values, found {inf_count} Infs")

    # Validate probability sum (should be approximately 1, allow small numerical errors)
    prob_sums = np.sum(probs, axis=1)
    if np.any(np.abs(prob_sums - 1.0) > 0.01):
        bad_indices = np.where(np.abs(prob_sums - 1.0) > 0.01)[0]
        raise ValueError(
            f"Probabilities must sum to 1.0 for each sample, "
            f"found {len(bad_indices)} samples with sums outside [0.99, 1.01]. "
            f"Example sums: {prob_sums[bad_indices[:5]].tolist()}"
        )

    # Handle scalar inputs - convert to dict format
    if isinstance(alpha_target, int | float):
        if not (0.0 < float(alpha_target) < 1.0):
            raise ValueError(f"alpha_target must be in (0,1), got {alpha_target}")
        alpha_dict: dict[int, float] = {0: float(alpha_target), 1: float(alpha_target)}
    else:
        alpha_dict = cast(dict[int, float], alpha_target)
        if not all(0.0 < v < 1.0 for v in alpha_dict.values()):
            raise ValueError(f"All alpha_target values must be in (0,1), got {alpha_dict}")

    if isinstance(delta, int | float):
        if not (0.0 < float(delta) < 1.0):
            raise ValueError(f"delta must be in (0,1), got {delta}")
        delta_dict: dict[int, float] = {0: float(delta), 1: float(delta)}
    else:
        delta_dict = cast(dict[int, float], delta)
        if not all(0.0 < v < 1.0 for v in delta_dict.values()):
            raise ValueError(f"All delta values must be in (0,1), got {delta_dict}")

    # Validate other parameters
    if test_size is not None and test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0,1), got {ci_level}")
    if not isinstance(use_union_bound, bool):
        raise TypeError(f"use_union_bound must be bool, got {type(use_union_bound).__name__}")
    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be bool, got {type(verbose).__name__}")

    logger.debug(f"Input validation passed: n={len(labels)}, alpha_target={alpha_dict}, delta={delta_dict}")

    # Split by class
    class_data = split_by_class(labels, probs)
    n_0 = class_data[0]["n"]
    n_1 = class_data[1]["n"]
    n_total = len(labels)

    # Set test_size if not provided
    if test_size is None:
        test_size = n_total

    # Derive PAC levels from delta values
    # For marginal: use independence since split (n₀, n₁) is observed
    # Pr(both coverage guarantees hold) = (1-δ₀)(1-δ₁)
    pac_level_marginal = (1 - delta_dict[0]) * (1 - delta_dict[1])
    pac_level_0 = 1 - delta_dict[0]
    pac_level_1 = 1 - delta_dict[1]

    # Step 1: Run SSBC for each class
    ssbc_result_0 = ssbc_correct(alpha_target=alpha_dict[0], n=n_0, delta=delta_dict[0], mode="beta")
    ssbc_result_1 = ssbc_correct(alpha_target=alpha_dict[1], n=n_1, delta=delta_dict[1], mode="beta")

    # Step 2: Get calibration results (for thresholds and basic stats)
    cal_result, pred_stats = mondrian_conformal_calibrate(
        class_data=class_data, alpha_target=alpha_dict, delta=delta_dict, mode="beta"
    )

    # Step 3: Compute PAC operational bounds - MARGINAL (always LOO-corrected)
    pac_bounds_marginal = compute_pac_operational_bounds_marginal_loo_corrected(
        ssbc_result_0=ssbc_result_0,
        ssbc_result_1=ssbc_result_1,
        labels=labels,
        probs=probs,
        test_size=test_size,
        ci_level=ci_level,
        pac_level=pac_level_marginal,
        use_union_bound=use_union_bound,
        n_jobs=n_jobs,
        prediction_method=prediction_method,
        loo_inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    # Step 4: Compute PAC operational bounds - PER-CLASS (always LOO-corrected)
    pac_bounds_class_0 = compute_pac_operational_bounds_perclass_loo_corrected(
        ssbc_result_0=ssbc_result_0,
        ssbc_result_1=ssbc_result_1,
        labels=labels,
        probs=probs,
        class_label=0,
        test_size=test_size,
        ci_level=ci_level,
        pac_level=pac_level_0,
        use_union_bound=use_union_bound,
        n_jobs=n_jobs,
        prediction_method=prediction_method,
        loo_inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    pac_bounds_class_1 = compute_pac_operational_bounds_perclass_loo_corrected(
        ssbc_result_0=ssbc_result_0,
        ssbc_result_1=ssbc_result_1,
        labels=labels,
        probs=probs,
        class_label=1,
        test_size=test_size,
        ci_level=ci_level,
        pac_level=pac_level_1,
        use_union_bound=use_union_bound,
        n_jobs=n_jobs,
        prediction_method=prediction_method,
        loo_inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    # Build comprehensive report dict (common to all paths)
    # Build cleaned report with only essential information
    report = {
        # Essential SSBC results (return dataclasses as-is for tests)
        "ssbc_class_0": ssbc_result_0,
        "ssbc_class_1": ssbc_result_1,
        "pac_bounds_marginal": pac_bounds_marginal,
        "pac_bounds_class_0": pac_bounds_class_0,
        "pac_bounds_class_1": pac_bounds_class_1,
        # Calibration result as returned by mondrian_conformal_calibrate (keys 0 and 1)
        "calibration_result": cal_result,
        "prediction_stats": pred_stats,
        "parameters": {
            "alpha_target": alpha_dict,
            "delta": delta_dict,
            "test_size": test_size,
            "ci_level": ci_level,
            "pac_level_marginal": pac_level_marginal,
            "pac_level_0": pac_level_0,
            "pac_level_1": pac_level_1,
            "use_union_bound": use_union_bound,
        },
        # Metadata for reproducibility
        "metadata": {
            "ssbc_version": __version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "timestamp": datetime.now().isoformat(),
            "n_calibration": n_total,
            "n_class_0": n_0,
            "n_class_1": n_1,
            "prediction_method": prediction_method,
            "use_loo_correction": use_loo_correction,
            "loo_inflation_factor": loo_inflation_factor,
        },
    }
    logger.info(f"Report generated successfully: n={n_total}, n_0={n_0}, n_1={n_1}")

    # Print comprehensive report if verbose
    if verbose:
        _print_rigorous_report(report)

    return report


def _print_rigorous_report(report: dict) -> None:
    """Print comprehensive rigorous PAC report."""
    cal_result = report["calibration_result"]
    pred_stats = report["prediction_stats"]
    params = report["parameters"]

    print("=" * 80)
    print("OPERATIONAL PAC-CONTROLLED CONFORMAL PREDICTION REPORT")
    print("=" * 80)
    print("\nParameters:")
    print(f"  Test size: {params['test_size']}")
    print(f"  CI level: {params['ci_level']:.0%} (Clopper-Pearson)")
    pac_0 = params["pac_level_0"]
    pac_1 = params["pac_level_1"]
    delta_0 = 1.0 - pac_0
    delta_1 = 1.0 - pac_1
    print("  PAC guarantee levels:")
    print(f"    Class 0: δ = {delta_0:.2f} ({pac_0:.0%} confidence)")
    print(f"    Class 1: δ = {delta_1:.2f} ({pac_1:.0%} confidence)")
    union_bound = params["use_union_bound"]
    if union_bound:
        print("    Union bound: applied across metrics (all metrics hold simultaneously)")
        print("    Class guarantees: validated separately (no union bound across classes)")
    else:
        print("    Union bound: not applied (metrics validated independently)")
        print("    Class guarantees: validated separately")

    # Per-class reports
    for class_label in [0, 1]:
        ssbc = report[f"ssbc_class_{class_label}"]
        pac = report[f"pac_bounds_class_{class_label}"]
        cal = cal_result[class_label]

        print("\n" + "=" * 80)
        print(f"CLASS {class_label} (Conditioned on True Label = {class_label})")
        print("=" * 80)

        print(f"  Calibration size: n = {ssbc.n}")
        print(f"  Target miscoverage: α = {params['alpha_target'][class_label]:.3f}")
        print(f"  SSBC-corrected α:   α' = {ssbc.alpha_corrected:.4f}")
        print(f"  PAC risk:           δ = {params['delta'][class_label]:.3f}")
        print(f"  Conformal threshold: {cal['threshold']:.4f}")

        # Calibration data statistics
        stats = pred_stats[class_label]
        if "error" not in stats:
            print(f"\n  Calibration summary (n = {ssbc.n})")
            print("     Empirical rates on calibration data. Intervals are 95% Clopper-Pearson.")
            print("     These do not include PAC guarantees.")

            # Abstentions
            abst = stats["abstentions"]
            print(
                f"     Abstentions:            {abst['count']:4d} / {ssbc.n:4d}  = "
                f"{abst['proportion']:6.2%}   95% CI: [{abst['lower']:.3f}, {abst['upper']:.3f}]"
            )

            # Singletons
            sing = stats["singletons"]
            print(
                f"     Singletons:           {sing['count']:4d} / {ssbc.n:4d}  = "
                f"{sing['proportion']:6.2%}   95% CI: [{sing['lower']:.3f}, {sing['upper']:.3f}]"
            )

            # Correct/incorrect singletons
            sing_corr = stats["singletons_correct"]
            print(
                f"       Correct:            {sing_corr['count']:4d} / {ssbc.n:4d}  = "
                f"{sing_corr['proportion']:6.2%}   95% CI: [{sing_corr['lower']:.3f}, {sing_corr['upper']:.3f}]"
            )

            sing_incorr = stats["singletons_incorrect"]
            print(
                f"       Incorrect:           {sing_incorr['count']:4d} / {ssbc.n:4d}  = "
                f"{sing_incorr['proportion']:6.2%}   95% CI: [{sing_incorr['lower']:.3f}, {sing_incorr['upper']:.3f}]"
            )

            # Error | singleton
            if sing["count"] > 0:
                from ssbc.bounds import cp_interval

                error_cond = cp_interval(sing_incorr["count"], sing["count"])
                print(
                    f"     Error | singleton:     {sing_incorr['count']:4d} / {sing['count']:4d}  = "
                    f"{error_cond['proportion']:6.2%}   95% CI: [{error_cond['lower']:.3f}, {error_cond['upper']:.3f}]"
                )

            # Doublets
            doub = stats["doublets"]
            print(
                f"     Doublets:              {doub['count']:4d} / {ssbc.n:4d}  = "
                f"{doub['proportion']:6.2%}   95% CI: [{doub['lower']:.3f}, {doub['upper']:.3f}]"
            )

        print("\n  Operational bounds for deployment")
        pac_level_class = params[f"pac_level_{class_label}"]
        if "loo_diagnostics" in pac:
            print(
                "     Method: leave-one-out calibration at confidence 1-δ, plus binomial "
                "predictive bounds for sampling variability."
            )
        else:
            print(
                "     Method: leave-one-out calibration at confidence 1-δ, plus prediction "
                "bounds for sampling uncertainty."
            )
        print(f"     Threshold calibration level: {pac_level_class:.0%} (1-δ)")
        print(f"     Reported confidence level for bounds: {params['ci_level']:.0%}")
        print(f"     Grid points evaluated: {pac['n_grid_points']}")

        # Helper to print bounds with method comparison
        # Capture test_size in closure-safe way
        test_size_for_methods = pac.get("test_size", params["test_size"])

        def _print_rate_with_methods(rate_name: str, bounds: tuple, expected: float, diagnostics: dict | None = None):
            """Print rate bounds, showing method comparison if available."""
            lower, upper = bounds
            test_size = test_size_for_methods  # noqa: B023 (captured in closure)
            print(f"\n     {rate_name}")
            print(f"       Point estimate: {expected:.3f}")

            if diagnostics and "comparison" in diagnostics:
                # Method comparison available
                comp = diagnostics["comparison"]
                selected = diagnostics.get("selected_method", "unknown")
                print(f"       Candidate bounds (95% predictive, n_test = {test_size}):")
                for method_name, method_lower, method_upper, method_width in zip(
                    comp["method"], comp["lower"], comp["upper"], comp["width"], strict=False
                ):
                    # Replace method names for display
                    display_name = method_name.replace("Analytical", "Normal approximation")
                    display_name = display_name.replace("Exact Binomial", "Exact binomial predictive")
                    # Match selected method - handle both "exact" and "exact (auto-corrected)" cases
                    method_lower_name = method_name.lower().replace(" ", "_")
                    if "analytical" in method_lower_name and (
                        "analytical" in selected.lower() or selected.lower() == "analytical"
                    ):
                        marker = "(retained)"
                    elif "exact" in method_lower_name and "exact" in selected.lower():
                        marker = "(retained)"
                    elif "hoeffding" in method_lower_name and "hoeffding" in selected.lower():
                        marker = "(retained)"
                    else:
                        marker = ""
                    print(
                        f"         {display_name:25s} [{method_lower:.3f}, {method_upper:.3f}]   "
                        f"width {method_width:.3f}  {marker}"
                    )
                print(f"       Operational bounds: [{lower:.3f}, {upper:.3f}]")
            else:
                # Single method - show which method if available
                method_info = diagnostics.get("selected_method", "") if diagnostics else ""
                # Fallback to "method" key if "selected_method" not available
                if not method_info and diagnostics and "method" in diagnostics:
                    method_name = diagnostics["method"]
                    # Convert internal method names to user-friendly names
                    method_map = {
                        "clopper_pearson_plus_sampling": "simple",
                        "beta_binomial_loo_corrected": "beta_binomial",
                        "hoeffding_distribution_free": "hoeffding",
                    }
                    method_info = method_map.get(method_name, method_name)
                if method_info:
                    print(f"       Method: {method_info}")
                print(f"       Operational bounds: [{lower:.3f}, {upper:.3f}]")

        # Get diagnostics if available
        loo_diag = pac.get("loo_diagnostics", {})
        singleton_diag = loo_diag.get("singleton") if loo_diag else None
        doublet_diag = loo_diag.get("doublet") if loo_diag else None
        abstention_diag = loo_diag.get("abstention") if loo_diag else None
        error_diag = loo_diag.get("singleton_error") if loo_diag else None

        s_lower, s_upper = pac["singleton_rate_bounds"]
        _print_rate_with_methods("Singleton rate", (s_lower, s_upper), pac["expected_singleton_rate"], singleton_diag)

        d_lower, d_upper = pac["doublet_rate_bounds"]
        _print_rate_with_methods("Doublet rate", (d_lower, d_upper), pac["expected_doublet_rate"], doublet_diag)

        a_lower, a_upper = pac["abstention_rate_bounds"]
        _print_rate_with_methods(
            "Abstention rate", (a_lower, a_upper), pac["expected_abstention_rate"], abstention_diag
        )

        se_lower, se_upper = pac["singleton_error_rate_bounds"]
        _print_rate_with_methods(
            f"Conditional error rate given singleton (P(error | singleton, class = {class_label}))",
            (se_lower, se_upper),
            pac["expected_singleton_error_rate"],
            error_diag,
        )

        # Singleton correct rate: P(correct | singleton, class) = 1 - P(error | singleton, class)
        if "singleton_correct_rate_bounds" in pac:
            sc_lower, sc_upper = pac["singleton_correct_rate_bounds"]
            sc_expected = pac.get("expected_singleton_correct_rate", 1.0 - pac["expected_singleton_error_rate"])
            # Use same diagnostics as error rate (they're complementary)
            _print_rate_with_methods(
                f"Conditional correct rate given singleton (P(correct | singleton, class = {class_label}))",
                (sc_lower, sc_upper),
                sc_expected,
                error_diag,
            )

        # Note about per-class rates (all have random denominators)
        print("\n     Stability note:")
        print(
            f"        All rates above (singleton, doublet, abstention, conditional error) "
            f"are conditional on class {class_label}."
        )
        print(
            f"        Their denominators (number of class {class_label} samples in the test set) "
            f"are random at deployment time."
        )
        print("        This induces extra variance and can bias the reported intervals.")
        print("        For audit and Service Level Objective reporting, use the marginal rates")
        print("        in the next section (normalized by total volume), which have a fixed denominator.")

    # Marginal report
    pac_marg = report["pac_bounds_marginal"]
    marginal_stats = pred_stats["marginal"]

    print("\n" + "=" * 80)
    print("MARGINAL STATISTICS (deployment view; class labels not assumed known)")
    print("=" * 80)
    n_total = marginal_stats["n_total"]
    print(f"  Total samples: n = {n_total}")

    # Calibration data statistics (marginal)
    print(f"\n  Calibration summary (n = {n_total})")
    print("     Empirical rates on calibration data. Intervals are 95% Clopper-Pearson.")
    print("     No PAC guarantees.")

    # Coverage
    cov = marginal_stats["coverage"]
    print(
        f"     Coverage (prediction set contains true label): {cov['count']:4d} / {n_total:4d}  = "
        f"{cov['rate']:6.2%}   95% CI: [{cov['ci_95']['lower']:.3f}, {cov['ci_95']['upper']:.3f}]"
    )

    # Abstentions
    abst = marginal_stats["abstentions"]
    print(
        f"     Abstentions:            {abst['count']:4d} / {n_total:4d}  = "
        f"{abst['proportion']:6.2%}   95% CI: [{abst['lower']:.3f}, {abst['upper']:.3f}]"
    )

    # Singletons
    sing = marginal_stats["singletons"]
    print(
        f"     Singletons:           {sing['count']:4d} / {n_total:4d}  = "
        f"{sing['proportion']:6.2%}   95% CI: [{sing['lower']:.3f}, {sing['upper']:.3f}]"
    )

    # Singleton errors
    if sing["count"] > 0:
        from ssbc.bounds import cp_interval

        error_cond_marg = cp_interval(sing["errors"], sing["count"])
        err_prop = error_cond_marg["proportion"]
        err_lower = error_cond_marg["lower"]
        err_upper = error_cond_marg["upper"]
        print(
            f"       Errors:             {sing['errors']:4d} / {sing['count']:4d}  = "
            f"{err_prop:6.2%}   95% CI: [{err_lower:.3f}, {err_upper:.3f}]"
        )

    # Doublets
    doub = marginal_stats["doublets"]
    print(
        f"     Doublets:              {doub['count']:4d} / {n_total:4d}  = "
        f"{doub['proportion']:6.2%}   95% CI: [{doub['lower']:.3f}, {doub['upper']:.3f}]"
    )

    print("\n  Operational bounds for deployment")
    print("     Class-specific rates (normalized by total test set size):")
    print("     These are JOINT probabilities measuring operational events across the full test set.")
    print("     ")
    print("     Singleton rates:")
    print("     - Definition: P(true_label=class, prediction_set=singleton)")
    print("     - Count samples where TRUE label = class AND prediction set = singleton")
    print("     - Example: 'Singleton rate (Class 0)' = P(Y=0, S=singleton)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • True label is class 0")
    print("         • Prediction set is singleton (can be {{0}} or {{1}})")
    print("       This includes BOTH correct singletons (predicted {{0}}) and")
    print("       incorrect singletons (predicted {{1}} when true label is 0).")
    print("     ")
    print("     Doublet rates:")
    print("     - Definition: P(true_label=class, prediction_set=doublet)")
    print("     - Count samples where TRUE label = class AND prediction set = {{0,1}}")
    print("     - Example: 'Doublet rate (Class 0)' = P(Y=0, S=doublet)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • True label is class 0")
    print("         • Prediction set is doublet (contains both {{0, 1}})")
    print("       Doublets always contain the true label (by coverage guarantee).")
    print("     ")
    print("     Abstention rates:")
    print("     - Definition: P(true_label=class, prediction_set=empty)")
    print("     - Count samples where TRUE label = class AND prediction set = {{}}")
    print("     - Example: 'Abstention rate (Class 0)' = P(Y=0, S=abstention)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • True label is class 0")
    print("         • Prediction set is empty (abstention/rejection)")
    print("       Abstentions indicate the model is uncertain and rejects the sample.")
    print("     ")
    print("     Error rates (normalized by total, conditioned on true label):")
    print("     - Definition: P(true_label=class, prediction_set=singleton, error=1)")
    print("     - Count samples where TRUE label = class AND singleton AND prediction is wrong")
    print("     - Example: 'Error rate (Class 0 singletons)' = P(Y=0, S=singleton, E=1)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • True label is class 0")
    print("         • Prediction set is singleton (single class predicted)")
    print("         • Prediction is INCORRECT (predicted class ≠ true label)")
    print("     ")
    print("     Correct rates (normalized by total, conditioned on true label):")
    print("     - Definition: P(true_label=class, prediction_set=singleton, error=0)")
    print("     - Count samples where TRUE label = class AND singleton AND prediction is correct")
    print("     - Example: 'Correct rate (Class 0 singletons)' = P(Y=0, S=singleton, E=0)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • True label is class 0")
    print("         • Prediction set is singleton (single class predicted)")
    print("         • Prediction is CORRECT (predicted class = true label)")
    print("     ")
    print("     Error rates (normalized by total, conditioned on predicted class):")
    print("     - Definition: P(predicted_class=X, prediction_set=singleton, error=1)")
    print("     - Count samples where PREDICTED class = X AND singleton AND prediction is wrong")
    print("     - Example: 'Error rate (when singleton predicted as Class 0)' = P(predicted=0, S=singleton, E=1)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • Prediction set is singleton with predicted class = 0")
    print("         • Prediction is INCORRECT (predicted class ≠ true label)")
    print("       This answers: 'If I predict class 0, how often am I wrong?'")
    print("     ")
    print("     Correct rates (normalized by total, conditioned on predicted class):")
    print("     - Definition: P(predicted_class=X, prediction_set=singleton, error=0)")
    print("     - Count samples where PREDICTED class = X AND singleton AND prediction is correct")
    print("     - Example: 'Correct rate (when singleton predicted as Class 0)' = P(predicted=0, S=singleton, E=0)")
    print("       Meaning: Among all test samples, what fraction have:")
    print("         • Prediction set is singleton with predicted class = 0")
    print("         • Prediction is CORRECT (predicted class = true label)")
    print("       This answers: 'If I predict class 0, how often am I correct?'")
    print("     ")
    print("     Relationship:")
    print("     - singleton_rate = error_rate + correct_rate (for same class)")
    print("     - All rates normalized by total test set size (fixed denominator)")
    print("     ")
    print("     All rates use fixed denominator (total test set size) for deployment planning.")
    ci_lvl = params["ci_level"]
    print(f"     Reported confidence level for bounds: {ci_lvl:.0%}")

    # Helper to print bounds with method comparison (reused for marginal)
    def _print_rate_with_methods_marginal(
        rate_name: str, bounds: tuple, expected: float, diagnostics: dict | None = None
    ):
        """Print rate bounds, showing method comparison if available."""
        lower, upper = bounds
        test_size = pac_marg.get("test_size", params["test_size"])
        print(f"\n     {rate_name}")
        print(f"       Point estimate: {expected:.3f}")

        if diagnostics and "comparison" in diagnostics:
            # Method comparison available
            comp = diagnostics["comparison"]
            selected = diagnostics.get("selected_method", "unknown")
            print(f"       Candidate bounds (95% predictive, n_test = {test_size}):")
            for method_name, method_lower, method_upper, method_width in zip(
                comp["method"], comp["lower"], comp["upper"], comp["width"], strict=False
            ):
                # Replace method names for display
                display_name = method_name.replace("Analytical", "Normal approximation")
                display_name = display_name.replace("Exact Binomial", "Exact binomial predictive")
                # Match selected method - handle both "exact" and "exact (auto-corrected)" cases
                method_lower_name = method_name.lower().replace(" ", "_")
                if "analytical" in method_lower_name and (
                    "analytical" in selected.lower() or selected.lower() == "analytical"
                ):
                    marker = "(retained)"
                elif "exact" in method_lower_name and "exact" in selected.lower():
                    marker = "(retained)"
                elif "hoeffding" in method_lower_name and "hoeffding" in selected.lower():
                    marker = "(retained)"
                else:
                    marker = ""
                print(
                    f"         {display_name:25s} [{method_lower:.3f}, {method_upper:.3f}]   "
                    f"width {method_width:.3f}  {marker}"
                )
            print(f"       Operational bounds: [{lower:.3f}, {upper:.3f}]")
        else:
            # Single method - show which method if available
            method_info = diagnostics.get("selected_method", "") if diagnostics else ""
            # Fallback to "method" key if "selected_method" not available
            if not method_info and diagnostics and "method" in diagnostics:
                method_name = diagnostics["method"]
                # Convert internal method names to user-friendly names
                method_map = {
                    "clopper_pearson_plus_sampling": "simple",
                    "beta_binomial_loo_corrected": "beta_binomial",
                    "hoeffding_distribution_free": "hoeffding",
                }
                method_info = method_map.get(method_name, method_name)
            if method_info:
                print(f"       Method: {method_info}")
            print(f"       Operational bounds: [{lower:.3f}, {upper:.3f}]")

    # Get diagnostics if available (for class-specific rates and error rates)
    loo_diag_marg = pac_marg.get("loo_diagnostics", {})
    singleton_class0_diag_marg = loo_diag_marg.get("singleton_class0") if loo_diag_marg else None
    singleton_class1_diag_marg = loo_diag_marg.get("singleton_class1") if loo_diag_marg else None
    doublet_class0_diag_marg = loo_diag_marg.get("doublet_class0") if loo_diag_marg else None
    doublet_class1_diag_marg = loo_diag_marg.get("doublet_class1") if loo_diag_marg else None
    abstention_class0_diag_marg = loo_diag_marg.get("abstention_class0") if loo_diag_marg else None
    abstention_class1_diag_marg = loo_diag_marg.get("abstention_class1") if loo_diag_marg else None
    error_class0_diag_marg = loo_diag_marg.get("singleton_error_class0") if loo_diag_marg else None
    error_class1_diag_marg = loo_diag_marg.get("singleton_error_class1") if loo_diag_marg else None
    correct_class0_diag_marg = loo_diag_marg.get("singleton_correct_class0") if loo_diag_marg else None
    correct_class1_diag_marg = loo_diag_marg.get("singleton_correct_class1") if loo_diag_marg else None
    error_pred_class0_diag_marg = loo_diag_marg.get("singleton_error_pred_class0") if loo_diag_marg else None
    error_pred_class1_diag_marg = loo_diag_marg.get("singleton_error_pred_class1") if loo_diag_marg else None
    correct_pred_class0_diag_marg = loo_diag_marg.get("singleton_correct_pred_class0") if loo_diag_marg else None
    correct_pred_class1_diag_marg = loo_diag_marg.get("singleton_correct_pred_class1") if loo_diag_marg else None

    # Class-specific singleton rates (normalized against full dataset)
    # These are operationally meaningful for deployment planning
    if "singleton_rate_class0_bounds" in pac_marg:
        s_class0_lower, s_class0_upper = pac_marg["singleton_rate_class0_bounds"]
        s_class0_expected = pac_marg.get("expected_singleton_rate_class0", 0.0)
        _print_rate_with_methods_marginal(
            "Singleton rate (Class 0, normalized by total)",
            (s_class0_lower, s_class0_upper),
            s_class0_expected,
            singleton_class0_diag_marg,
        )

    if "singleton_rate_class1_bounds" in pac_marg:
        s_class1_lower, s_class1_upper = pac_marg["singleton_rate_class1_bounds"]
        s_class1_expected = pac_marg.get("expected_singleton_rate_class1", 0.0)
        _print_rate_with_methods_marginal(
            "Singleton rate (Class 1, normalized by total)",
            (s_class1_lower, s_class1_upper),
            s_class1_expected,
            singleton_class1_diag_marg,
        )

    # Class-specific doublet rates (normalized against full dataset)
    if "doublet_rate_class0_bounds" in pac_marg:
        d_class0_lower, d_class0_upper = pac_marg["doublet_rate_class0_bounds"]
        d_class0_expected = pac_marg.get("expected_doublet_rate_class0", 0.0)
    _print_rate_with_methods_marginal(
        "Doublet rate (Class 0, normalized by total)",
        (d_class0_lower, d_class0_upper),
        d_class0_expected,
        doublet_class0_diag_marg,
    )

    if "doublet_rate_class1_bounds" in pac_marg:
        d_class1_lower, d_class1_upper = pac_marg["doublet_rate_class1_bounds"]
        d_class1_expected = pac_marg.get("expected_doublet_rate_class1", 0.0)
    _print_rate_with_methods_marginal(
        "Doublet rate (Class 1, normalized by total)",
        (d_class1_lower, d_class1_upper),
        d_class1_expected,
        doublet_class1_diag_marg,
    )

    # Class-specific abstention rates (normalized against full dataset)
    if "abstention_rate_class0_bounds" in pac_marg:
        a_class0_lower, a_class0_upper = pac_marg["abstention_rate_class0_bounds"]
        a_class0_expected = pac_marg.get("expected_abstention_rate_class0", 0.0)
    _print_rate_with_methods_marginal(
        "Abstention rate (Class 0, normalized by total)",
        (a_class0_lower, a_class0_upper),
        a_class0_expected,
        abstention_class0_diag_marg,
    )

    if "abstention_rate_class1_bounds" in pac_marg:
        a_class1_lower, a_class1_upper = pac_marg["abstention_rate_class1_bounds"]
        a_class1_expected = pac_marg.get("expected_abstention_rate_class1", 0.0)
    _print_rate_with_methods_marginal(
        "Abstention rate (Class 1, normalized by total)",
        (a_class1_lower, a_class1_upper),
        a_class1_expected,
        abstention_class1_diag_marg,
    )

    # Class-specific error rates (normalized against full dataset)
    # Note: We do NOT report marginal singleton_error because it mixes two different
    # distributions (class 0 and class 1) which cannot be justified statistically.
    if "singleton_error_rate_class0_bounds" in pac_marg:
        se_class0_lower, se_class0_upper = pac_marg["singleton_error_rate_class0_bounds"]
        se_class0_expected = pac_marg.get("expected_singleton_error_rate_class0", 0.0)
        _print_rate_with_methods_marginal(
            "Error rate (Class 0 singletons, normalized by total)",
            (se_class0_lower, se_class0_upper),
            se_class0_expected,
            error_class0_diag_marg,
        )

    if "singleton_error_rate_class1_bounds" in pac_marg:
        se_class1_lower, se_class1_upper = pac_marg["singleton_error_rate_class1_bounds"]
        se_class1_expected = pac_marg.get("expected_singleton_error_rate_class1", 0.0)
        _print_rate_with_methods_marginal(
            "Error rate (Class 1 singletons, normalized by total)",
            (se_class1_lower, se_class1_upper),
            se_class1_expected,
            error_class1_diag_marg,
        )

    # Class-specific singleton correct rates (normalized against full dataset)
    if "singleton_correct_rate_class0_bounds" in pac_marg:
        sc_class0_lower, sc_class0_upper = pac_marg["singleton_correct_rate_class0_bounds"]
        sc_class0_expected = pac_marg.get("expected_singleton_correct_rate_class0", 0.0)
        _print_rate_with_methods_marginal(
            "Correct rate (Class 0 singletons, normalized by total)",
            (sc_class0_lower, sc_class0_upper),
            sc_class0_expected,
            correct_class0_diag_marg,
        )

    if "singleton_correct_rate_class1_bounds" in pac_marg:
        sc_class1_lower, sc_class1_upper = pac_marg["singleton_correct_rate_class1_bounds"]
        sc_class1_expected = pac_marg.get("expected_singleton_correct_rate_class1", 0.0)
        _print_rate_with_methods_marginal(
            "Correct rate (Class 1 singletons, normalized by total)",
            (sc_class1_lower, sc_class1_upper),
            sc_class1_expected,
            correct_class1_diag_marg,
        )

    # Error rates when singleton is assigned to a specific class (normalized against full dataset)
    if "singleton_error_rate_pred_class0_bounds" in pac_marg:
        se_pred_class0_lower, se_pred_class0_upper = pac_marg["singleton_error_rate_pred_class0_bounds"]
        se_pred_class0_expected = pac_marg.get("expected_singleton_error_rate_pred_class0", 0.0)
        _print_rate_with_methods_marginal(
            "Error rate (when singleton predicted as Class 0, normalized by total)",
            (se_pred_class0_lower, se_pred_class0_upper),
            se_pred_class0_expected,
            error_pred_class0_diag_marg,
        )

    if "singleton_error_rate_pred_class1_bounds" in pac_marg:
        se_pred_class1_lower, se_pred_class1_upper = pac_marg["singleton_error_rate_pred_class1_bounds"]
        se_pred_class1_expected = pac_marg.get("expected_singleton_error_rate_pred_class1", 0.0)
        _print_rate_with_methods_marginal(
            "Error rate (when singleton predicted as Class 1, normalized by total)",
            (se_pred_class1_lower, se_pred_class1_upper),
            se_pred_class1_expected,
            error_pred_class1_diag_marg,
        )

    # Correct rates when singleton is assigned to a specific class (normalized against full dataset)
    if "singleton_correct_rate_pred_class0_bounds" in pac_marg:
        sc_pred_class0_lower, sc_pred_class0_upper = pac_marg["singleton_correct_rate_pred_class0_bounds"]
        sc_pred_class0_expected = pac_marg.get("expected_singleton_correct_rate_pred_class0", 0.0)
        _print_rate_with_methods_marginal(
            "Correct rate (when singleton predicted as Class 0, normalized by total)",
            (sc_pred_class0_lower, sc_pred_class0_upper),
            sc_pred_class0_expected,
            correct_pred_class0_diag_marg,
        )

    if "singleton_correct_rate_pred_class1_bounds" in pac_marg:
        sc_pred_class1_lower, sc_pred_class1_upper = pac_marg["singleton_correct_rate_pred_class1_bounds"]
        sc_pred_class1_expected = pac_marg.get("expected_singleton_correct_rate_pred_class1", 0.0)
        _print_rate_with_methods_marginal(
            "Correct rate (when singleton predicted as Class 1, normalized by total)",
            (sc_pred_class1_lower, sc_pred_class1_upper),
            sc_pred_class1_expected,
            correct_pred_class1_diag_marg,
        )

    # Deployment expectations are not reported at marginal level since singleton/doublet
    # rates are derived from class-specific rates (already reported in CLASS 0/1 sections).
    print("\n" + "=" * 80)
