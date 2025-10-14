"""Unified rigorous reporting with full PAC guarantees.

This module provides a single comprehensive report that properly accounts for
coverage volatility across all operational metrics.
"""

from typing import Any

import numpy as np

from .conformal import mondrian_conformal_calibrate, split_by_class
from .core import ssbc_correct
from .operational_bounds_simple import (
    compute_pac_operational_bounds_marginal,
    compute_pac_operational_bounds_perclass,
)


def generate_rigorous_pac_report(
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: float | dict[int, float] = 0.10,
    delta: float | dict[int, float] = 0.10,
    test_size: int | None = None,
    ci_level: float = 0.95,
    use_union_bound: bool = True,
    n_jobs: int = -1,
    verbose: bool = True,
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
        Confidence level for Clopper-Pearson intervals
    use_union_bound : bool, default=True
        Apply Bonferroni for simultaneous guarantees (recommended)
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
    **This replaces the old workflow:**

    OLD (incomplete):
    ```python
    cal_result, pred_stats = mondrian_conformal_calibrate(...)
    op_bounds = compute_mondrian_operational_bounds(...)  # No coverage volatility!
    marginal_bounds = compute_marginal_operational_bounds(...)  # No coverage volatility!
    report_prediction_stats(...)  # Uses incomplete bounds
    ```

    NEW (rigorous):
    ```python
    report = generate_rigorous_pac_report(labels, probs, alpha_target, delta)
    # Done! All bounds account for coverage volatility.
    ```
    """
    # Handle scalar inputs
    if isinstance(alpha_target, (int, float)):
        alpha_dict = {0: float(alpha_target), 1: float(alpha_target)}
    else:
        alpha_dict = alpha_target

    if isinstance(delta, (int, float)):
        delta_dict = {0: float(delta), 1: float(delta)}
    else:
        delta_dict = delta

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

    # Step 3: Compute PAC operational bounds - MARGINAL
    # Uses minimum confidence (max delta) for conservativeness
    pac_bounds_marginal = compute_pac_operational_bounds_marginal(
        ssbc_result_0=ssbc_result_0,
        ssbc_result_1=ssbc_result_1,
        labels=labels,
        probs=probs,
        test_size=test_size,
        ci_level=ci_level,
        pac_level=pac_level_marginal,
        use_union_bound=use_union_bound,
        n_jobs=n_jobs,
    )

    # Step 4: Compute PAC operational bounds - PER-CLASS
    # Each class uses its own delta
    pac_bounds_class_0 = compute_pac_operational_bounds_perclass(
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
    )

    pac_bounds_class_1 = compute_pac_operational_bounds_perclass(
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
    )

    # Build comprehensive report dict
    report = {
        "ssbc_class_0": ssbc_result_0,
        "ssbc_class_1": ssbc_result_1,
        "pac_bounds_marginal": pac_bounds_marginal,
        "pac_bounds_class_0": pac_bounds_class_0,
        "pac_bounds_class_1": pac_bounds_class_1,
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
    }

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
    print("RIGOROUS PAC-CONTROLLED CONFORMAL PREDICTION REPORT")
    print("=" * 80)
    print("\nParameters:")
    print(f"  Test size: {params['test_size']}")
    print(f"  CI level: {params['ci_level']:.0%} (Clopper-Pearson)")
    print(
        f"  PAC confidence: Class 0: {params['pac_level_0']:.0%}, Class 1: {params['pac_level_1']:.0%}, Marginal: {params['pac_level_marginal']:.0%}"
    )
    print(f"  Union bound: {'YES (all metrics hold simultaneously)' if params['use_union_bound'] else 'NO'}")

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
            print(f"\n  📊 Statistics from Calibration Data (n={ssbc.n}):")
            print("     [Basic CP CIs without PAC guarantee - evaluated on calibration data]")

            # Abstentions
            abst = stats["abstentions"]
            print(
                f"    Abstentions:      {abst['count']:4d} / {ssbc.n:4d} = {abst['proportion']:6.2%}  "
                f"95% CI: [{abst['lower']:.3f}, {abst['upper']:.3f}]"
            )

            # Singletons
            sing = stats["singletons"]
            print(
                f"    Singletons:       {sing['count']:4d} / {ssbc.n:4d} = {sing['proportion']:6.2%}  "
                f"95% CI: [{sing['lower']:.3f}, {sing['upper']:.3f}]"
            )

            # Correct/incorrect singletons
            sing_corr = stats["singletons_correct"]
            print(
                f"      Correct:        {sing_corr['count']:4d} / {ssbc.n:4d} = {sing_corr['proportion']:6.2%}  "
                f"95% CI: [{sing_corr['lower']:.3f}, {sing_corr['upper']:.3f}]"
            )

            sing_incorr = stats["singletons_incorrect"]
            print(
                f"      Incorrect:      {sing_incorr['count']:4d} / {ssbc.n:4d} = {sing_incorr['proportion']:6.2%}  "
                f"95% CI: [{sing_incorr['lower']:.3f}, {sing_incorr['upper']:.3f}]"
            )

            # Error | singleton
            if sing["count"] > 0:
                from .statistics import cp_interval

                error_cond = cp_interval(sing_incorr["count"], sing["count"])
                print(
                    f"    Error | singleton:  {sing_incorr['count']:4d} / {sing['count']:4d} = "
                    f"{error_cond['proportion']:6.2%}  95% CI: [{error_cond['lower']:.3f}, {error_cond['upper']:.3f}]"
                )

            # Doublets
            doub = stats["doublets"]
            print(
                f"    Doublets:         {doub['count']:4d} / {ssbc.n:4d} = {doub['proportion']:6.2%}  "
                f"95% CI: [{doub['lower']:.3f}, {doub['upper']:.3f}]"
            )

        print("\n  ✅ RIGOROUS PAC-Controlled Operational Bounds")
        print("     (LOO-CV for unbiased estimates, CP for binomial sampling)")
        pac_level_class = params[f"pac_level_{class_label}"]
        print(f"     PAC level: {pac_level_class:.0%} (= 1 - δ), CP level: {params['ci_level']:.0%}")
        print(f"     Grid points evaluated: {pac['n_grid_points']}")

        s_lower, s_upper = pac["singleton_rate_bounds"]
        print("\n     SINGLETON:")
        print(f"       Bounds: [{s_lower:.3f}, {s_upper:.3f}]")
        print(f"       Expected: {pac['expected_singleton_rate']:.3f}")

        d_lower, d_upper = pac["doublet_rate_bounds"]
        print("\n     DOUBLET:")
        print(f"       Bounds: [{d_lower:.3f}, {d_upper:.3f}]")
        print(f"       Expected: {pac['expected_doublet_rate']:.3f}")

        a_lower, a_upper = pac["abstention_rate_bounds"]
        print("\n     ABSTENTION:")
        print(f"       Bounds: [{a_lower:.3f}, {a_upper:.3f}]")
        print(f"       Expected: {pac['expected_abstention_rate']:.3f}")

        se_lower, se_upper = pac["singleton_error_rate_bounds"]
        print("\n     CONDITIONAL ERROR (P(error | singleton)):")
        print(f"       Bounds: [{se_lower:.3f}, {se_upper:.3f}]")
        print(f"       Expected: {pac['expected_singleton_error_rate']:.3f}")

    # Marginal report
    pac_marg = report["pac_bounds_marginal"]
    marginal_stats = pred_stats["marginal"]

    print("\n" + "=" * 80)
    print("MARGINAL STATISTICS (Deployment View - Ignores True Labels)")
    print("=" * 80)
    n_total = marginal_stats["n_total"]
    print(f"  Total samples: n = {n_total}")

    # Calibration data statistics (marginal)
    print(f"\n  📊 Statistics from Calibration Data (n={n_total}):")
    print("     [Basic CP CIs - evaluated on calibration data]")

    # Coverage
    cov = marginal_stats["coverage"]
    print(
        f"    Coverage:          {cov['count']:4d} / {n_total:4d} = {cov['rate']:6.2%}  "
        f"95% CI: [{cov['ci_95']['lower']:.3f}, {cov['ci_95']['upper']:.3f}]"
    )

    # Abstentions
    abst = marginal_stats["abstentions"]
    print(
        f"    Abstentions:       {abst['count']:4d} / {n_total:4d} = {abst['proportion']:6.2%}  "
        f"95% CI: [{abst['lower']:.3f}, {abst['upper']:.3f}]"
    )

    # Singletons
    sing = marginal_stats["singletons"]
    print(
        f"    Singletons:        {sing['count']:4d} / {n_total:4d} = {sing['proportion']:6.2%}  "
        f"95% CI: [{sing['lower']:.3f}, {sing['upper']:.3f}]"
    )

    # Singleton errors
    if sing["count"] > 0:
        from .statistics import cp_interval

        error_cond_marg = cp_interval(sing["errors"], sing["count"])
        print(
            f"      Errors:          {sing['errors']:4d} / {sing['count']:4d} = "
            f"{error_cond_marg['proportion']:6.2%}  95% CI: [{error_cond_marg['lower']:.3f}, {error_cond_marg['upper']:.3f}]"
        )

    # Doublets
    doub = marginal_stats["doublets"]
    print(
        f"    Doublets:          {doub['count']:4d} / {n_total:4d} = {doub['proportion']:6.2%}  "
        f"95% CI: [{doub['lower']:.3f}, {doub['upper']:.3f}]"
    )

    print("\n  ✅ RIGOROUS PAC-Controlled Marginal Bounds")
    print("     (LOO-CV for unbiased estimates, CP for binomial sampling)")
    print(
        f"     PAC level: {params['pac_level_marginal']:.0%} (= (1-δ₀)×(1-δ₁), independence), CP level: {params['ci_level']:.0%}"
    )
    print(f"     Grid points evaluated: {pac_marg['n_grid_points']}")

    s_lower, s_upper = pac_marg["singleton_rate_bounds"]
    print("\n     SINGLETON:")
    print(f"       Bounds: [{s_lower:.3f}, {s_upper:.3f}]")
    print(f"       Expected: {pac_marg['expected_singleton_rate']:.3f}")

    d_lower, d_upper = pac_marg["doublet_rate_bounds"]
    print("\n     DOUBLET:")
    print(f"       Bounds: [{d_lower:.3f}, {d_upper:.3f}]")
    print(f"       Expected: {pac_marg['expected_doublet_rate']:.3f}")

    a_lower, a_upper = pac_marg["abstention_rate_bounds"]
    print("\n     ABSTENTION:")
    print(f"       Bounds: [{a_lower:.3f}, {a_upper:.3f}]")
    print(f"       Expected: {pac_marg['expected_abstention_rate']:.3f}")

    se_lower, se_upper = pac_marg["singleton_error_rate_bounds"]
    print("\n     CONDITIONAL ERROR (P(error | singleton)):")
    print(f"       Bounds: [{se_lower:.3f}, {se_upper:.3f}]")
    print(f"       Expected: {pac_marg['expected_singleton_error_rate']:.3f}")

    print("\n  📈 Deployment Expectations:")
    print(f"     Automation (singletons): {s_lower:.1%} - {s_upper:.1%}")
    print(f"     Escalation (doublets+abstentions): {a_lower+d_lower:.1%} - {a_upper+d_upper:.1%}")

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("\n✓ Bounds computed via LOO-CV for unbiased rate estimates")
    print("✓ Models test set sampling volatility for FIXED calibration")
    if params["use_union_bound"]:
        print("✓ Union bound ensures ALL metrics hold simultaneously")
    print("✓ Clopper-Pearson exact confidence intervals")
    print("✓ No data leakage - each sample evaluated on threshold from other samples")
    print("\n" + "=" * 80)
