"""Validation utilities for PAC-controlled operational bounds.

This module provides tools to empirically validate the theoretical PAC guarantees
by running simulations with fixed calibration thresholds on independent test sets.
"""

from typing import Any

import numpy as np


def validate_pac_bounds(
    report: dict[str, Any],
    simulator: Any,
    test_size: int,
    n_trials: int = 1000,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Empirically validate PAC operational bounds.

    Takes a PAC report from generate_rigorous_pac_report() and validates that
    the theoretical bounds actually hold in practice by:
    1. Extracting the FIXED thresholds from calibration
    2. Running n_trials simulations with fresh test sets
    3. Measuring empirical coverage of the PAC bounds

    Parameters
    ----------
    report : dict
        Output from generate_rigorous_pac_report()
    simulator : DataGenerator
        Simulator to generate independent test data (e.g., BinaryClassifierSimulator)
    test_size : int
        Size of each test set
    n_trials : int, default=1000
        Number of independent trials
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print validation progress

    Returns
    -------
    dict
        Validation results with:
        - 'marginal': Marginal operational rates and coverage
        - 'class_0': Class 0 operational rates and coverage
        - 'class_1': Class 1 operational rates and coverage
        Each containing:
        - 'singleton', 'doublet', 'abstention', 'singleton_error' dicts with:
          - 'rates': Array of rates across trials
          - 'mean': Mean rate
          - 'quantiles': Quantiles (5%, 25%, 50%, 75%, 95%)
          - 'bounds': PAC bounds from report
          - 'expected': Expected rate from report
          - 'empirical_coverage': Fraction of trials within bounds

    Examples
    --------
    >>> from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report, validate_pac_bounds
    >>> sim = BinaryClassifierSimulator(p_class1=0.2, seed=42)
    >>> labels, probs = sim.generate(100)
    >>> report = generate_rigorous_pac_report(labels, probs, delta=0.10)
    >>> validation = validate_pac_bounds(report, sim, test_size=1000, n_trials=1000)
    >>> print(f"Singleton coverage: {validation['marginal']['singleton']['empirical_coverage']:.1%}")

    Notes
    -----
    This function is useful for:
    - Verifying theoretical PAC guarantees empirically
    - Understanding the tightness of bounds
    - Debugging issues with bounds calculation
    - Generating validation plots for papers/reports

    The empirical coverage should be ≥ PAC level (1 - δ) for rigorous bounds.
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract FIXED thresholds from calibration
    threshold_0 = report["calibration_result"][0]["threshold"]
    threshold_1 = report["calibration_result"][1]["threshold"]

    if verbose:
        print(f"Using fixed thresholds: q̂₀={threshold_0:.4f}, q̂₁={threshold_1:.4f}")
        print(f"Running {n_trials} trials with test_size={test_size}...")

    # Storage for realized rates
    marginal_singleton_rates = []
    marginal_doublet_rates = []
    marginal_abstention_rates = []
    marginal_singleton_error_rates = []

    class_0_singleton_rates = []
    class_0_doublet_rates = []
    class_0_abstention_rates = []
    class_0_singleton_error_rates = []

    class_1_singleton_rates = []
    class_1_doublet_rates = []
    class_1_abstention_rates = []
    class_1_singleton_error_rates = []

    # Run trials
    for _ in range(n_trials):
        # Generate independent test set
        labels_test, probs_test = simulator.generate(test_size)

        # Apply FIXED Mondrian thresholds and evaluate
        n_total = len(labels_test)
        n_singletons = 0
        n_doublets = 0
        n_abstentions = 0
        n_singletons_correct = 0

        # Per-class counters
        n_0 = np.sum(labels_test == 0)
        n_1 = np.sum(labels_test == 1)

        n_singletons_0 = 0
        n_doublets_0 = 0
        n_abstentions_0 = 0
        n_singletons_correct_0 = 0

        n_singletons_1 = 0
        n_doublets_1 = 0
        n_abstentions_1 = 0
        n_singletons_correct_1 = 0

        for i in range(n_total):
            true_label = labels_test[i]
            score_0 = 1.0 - probs_test[i, 0]
            score_1 = 1.0 - probs_test[i, 1]

            # Build prediction set using FIXED thresholds
            in_0 = score_0 <= threshold_0
            in_1 = score_1 <= threshold_1

            # Marginal counts
            if in_0 and in_1:
                n_doublets += 1
            elif in_0 or in_1:
                n_singletons += 1
                if (in_0 and true_label == 0) or (in_1 and true_label == 1):
                    n_singletons_correct += 1
            else:
                n_abstentions += 1

            # Per-class counts
            if true_label == 0:
                if in_0 and in_1:
                    n_doublets_0 += 1
                elif in_0 or in_1:
                    n_singletons_0 += 1
                    if in_0:
                        n_singletons_correct_0 += 1
                else:
                    n_abstentions_0 += 1
            else:  # true_label == 1
                if in_0 and in_1:
                    n_doublets_1 += 1
                elif in_0 or in_1:
                    n_singletons_1 += 1
                    if in_1:
                        n_singletons_correct_1 += 1
                else:
                    n_abstentions_1 += 1

        # Compute marginal rates
        marginal_singleton_rates.append(n_singletons / n_total)
        marginal_doublet_rates.append(n_doublets / n_total)
        marginal_abstention_rates.append(n_abstentions / n_total)

        singleton_error_rate = (n_singletons - n_singletons_correct) / n_singletons if n_singletons > 0 else np.nan
        marginal_singleton_error_rates.append(singleton_error_rate)

        # Compute per-class rates
        if n_0 > 0:
            class_0_singleton_rates.append(n_singletons_0 / n_0)
            class_0_doublet_rates.append(n_doublets_0 / n_0)
            class_0_abstention_rates.append(n_abstentions_0 / n_0)
            singleton_error_0 = (
                (n_singletons_0 - n_singletons_correct_0) / n_singletons_0 if n_singletons_0 > 0 else np.nan
            )
            class_0_singleton_error_rates.append(singleton_error_0)

        if n_1 > 0:
            class_1_singleton_rates.append(n_singletons_1 / n_1)
            class_1_doublet_rates.append(n_doublets_1 / n_1)
            class_1_abstention_rates.append(n_abstentions_1 / n_1)
            singleton_error_1 = (
                (n_singletons_1 - n_singletons_correct_1) / n_singletons_1 if n_singletons_1 > 0 else np.nan
            )
            class_1_singleton_error_rates.append(singleton_error_1)

    # Convert to arrays
    marginal_singleton_rates = np.array(marginal_singleton_rates)
    marginal_doublet_rates = np.array(marginal_doublet_rates)
    marginal_abstention_rates = np.array(marginal_abstention_rates)
    marginal_singleton_error_rates = np.array(marginal_singleton_error_rates)

    class_0_singleton_rates = np.array(class_0_singleton_rates)
    class_0_doublet_rates = np.array(class_0_doublet_rates)
    class_0_abstention_rates = np.array(class_0_abstention_rates)
    class_0_singleton_error_rates = np.array(class_0_singleton_error_rates)

    class_1_singleton_rates = np.array(class_1_singleton_rates)
    class_1_doublet_rates = np.array(class_1_doublet_rates)
    class_1_abstention_rates = np.array(class_1_abstention_rates)
    class_1_singleton_error_rates = np.array(class_1_singleton_error_rates)

    # Helper functions
    def check_coverage(rates: np.ndarray, bounds: tuple[float, float]) -> float:
        """Check what fraction of rates fall within bounds."""
        lower, upper = bounds
        within = np.sum((rates >= lower) & (rates <= upper))
        return within / len(rates)

    def check_coverage_with_nan(rates: np.ndarray, bounds: tuple[float, float]) -> float:
        """Check coverage, ignoring NaN values."""
        lower, upper = bounds
        valid = ~np.isnan(rates)
        if np.sum(valid) == 0:
            return np.nan
        rates_valid = rates[valid]
        within = np.sum((rates_valid >= lower) & (rates_valid <= upper))
        return within / len(rates_valid)

    def compute_quantiles(rates: np.ndarray) -> dict[str, float]:
        """Compute quantiles, handling NaN."""
        valid = rates[~np.isnan(rates)] if np.any(np.isnan(rates)) else rates
        if len(valid) == 0:
            return {
                "q025": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q50": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "q975": np.nan,
            }
        return {
            "q025": float(np.percentile(valid, 2.5)),
            "q05": float(np.percentile(valid, 5)),
            "q25": float(np.percentile(valid, 25)),
            "q50": float(np.percentile(valid, 50)),
            "q75": float(np.percentile(valid, 75)),
            "q95": float(np.percentile(valid, 95)),
            "q975": float(np.percentile(valid, 97.5)),
        }

    # Get bounds from report
    pac_marg = report["pac_bounds_marginal"]
    pac_0 = report["pac_bounds_class_0"]
    pac_1 = report["pac_bounds_class_1"]

    return {
        "n_trials": n_trials,
        "test_size": test_size,
        "threshold_0": threshold_0,
        "threshold_1": threshold_1,
        "marginal": {
            "singleton": {
                "rates": marginal_singleton_rates,
                "mean": np.mean(marginal_singleton_rates),
                "quantiles": compute_quantiles(marginal_singleton_rates),
                "bounds": pac_marg["singleton_rate_bounds"],
                "expected": pac_marg["expected_singleton_rate"],
                "empirical_coverage": check_coverage(marginal_singleton_rates, pac_marg["singleton_rate_bounds"]),
            },
            "doublet": {
                "rates": marginal_doublet_rates,
                "mean": np.mean(marginal_doublet_rates),
                "quantiles": compute_quantiles(marginal_doublet_rates),
                "bounds": pac_marg["doublet_rate_bounds"],
                "expected": pac_marg["expected_doublet_rate"],
                "empirical_coverage": check_coverage(marginal_doublet_rates, pac_marg["doublet_rate_bounds"]),
            },
            "abstention": {
                "rates": marginal_abstention_rates,
                "mean": np.mean(marginal_abstention_rates),
                "quantiles": compute_quantiles(marginal_abstention_rates),
                "bounds": pac_marg["abstention_rate_bounds"],
                "expected": pac_marg["expected_abstention_rate"],
                "empirical_coverage": check_coverage(marginal_abstention_rates, pac_marg["abstention_rate_bounds"]),
            },
            "singleton_error": {
                "rates": marginal_singleton_error_rates,
                "mean": np.nanmean(marginal_singleton_error_rates),
                "quantiles": compute_quantiles(marginal_singleton_error_rates),
                "bounds": pac_marg["singleton_error_rate_bounds"],
                "expected": pac_marg["expected_singleton_error_rate"],
                "empirical_coverage": check_coverage_with_nan(
                    marginal_singleton_error_rates, pac_marg["singleton_error_rate_bounds"]
                ),
            },
        },
        "class_0": {
            "singleton": {
                "rates": class_0_singleton_rates,
                "mean": np.mean(class_0_singleton_rates),
                "quantiles": compute_quantiles(class_0_singleton_rates),
                "bounds": pac_0["singleton_rate_bounds"],
                "expected": pac_0["expected_singleton_rate"],
                "empirical_coverage": check_coverage(class_0_singleton_rates, pac_0["singleton_rate_bounds"]),
            },
            "doublet": {
                "rates": class_0_doublet_rates,
                "mean": np.mean(class_0_doublet_rates),
                "quantiles": compute_quantiles(class_0_doublet_rates),
                "bounds": pac_0["doublet_rate_bounds"],
                "expected": pac_0["expected_doublet_rate"],
                "empirical_coverage": check_coverage(class_0_doublet_rates, pac_0["doublet_rate_bounds"]),
            },
            "abstention": {
                "rates": class_0_abstention_rates,
                "mean": np.mean(class_0_abstention_rates),
                "quantiles": compute_quantiles(class_0_abstention_rates),
                "bounds": pac_0["abstention_rate_bounds"],
                "expected": pac_0["expected_abstention_rate"],
                "empirical_coverage": check_coverage(class_0_abstention_rates, pac_0["abstention_rate_bounds"]),
            },
            "singleton_error": {
                "rates": class_0_singleton_error_rates,
                "mean": np.nanmean(class_0_singleton_error_rates),
                "quantiles": compute_quantiles(class_0_singleton_error_rates),
                "bounds": pac_0["singleton_error_rate_bounds"],
                "expected": pac_0["expected_singleton_error_rate"],
                "empirical_coverage": check_coverage_with_nan(
                    class_0_singleton_error_rates, pac_0["singleton_error_rate_bounds"]
                ),
            },
        },
        "class_1": {
            "singleton": {
                "rates": class_1_singleton_rates,
                "mean": np.mean(class_1_singleton_rates),
                "quantiles": compute_quantiles(class_1_singleton_rates),
                "bounds": pac_1["singleton_rate_bounds"],
                "expected": pac_1["expected_singleton_rate"],
                "empirical_coverage": check_coverage(class_1_singleton_rates, pac_1["singleton_rate_bounds"]),
            },
            "doublet": {
                "rates": class_1_doublet_rates,
                "mean": np.mean(class_1_doublet_rates),
                "quantiles": compute_quantiles(class_1_doublet_rates),
                "bounds": pac_1["doublet_rate_bounds"],
                "expected": pac_1["expected_doublet_rate"],
                "empirical_coverage": check_coverage(class_1_doublet_rates, pac_1["doublet_rate_bounds"]),
            },
            "abstention": {
                "rates": class_1_abstention_rates,
                "mean": np.mean(class_1_abstention_rates),
                "quantiles": compute_quantiles(class_1_abstention_rates),
                "bounds": pac_1["abstention_rate_bounds"],
                "expected": pac_1["expected_abstention_rate"],
                "empirical_coverage": check_coverage(class_1_abstention_rates, pac_1["abstention_rate_bounds"]),
            },
            "singleton_error": {
                "rates": class_1_singleton_error_rates,
                "mean": np.nanmean(class_1_singleton_error_rates),
                "quantiles": compute_quantiles(class_1_singleton_error_rates),
                "bounds": pac_1["singleton_error_rate_bounds"],
                "expected": pac_1["expected_singleton_error_rate"],
                "empirical_coverage": check_coverage_with_nan(
                    class_1_singleton_error_rates, pac_1["singleton_error_rate_bounds"]
                ),
            },
        },
    }


def print_validation_results(validation: dict[str, Any]) -> None:
    """Pretty print validation results.

    Parameters
    ----------
    validation : dict
        Output from validate_pac_bounds()

    Examples
    --------
    >>> validation = validate_pac_bounds(report, sim, test_size=1000, n_trials=1000)
    >>> print_validation_results(validation)
    """
    print("=" * 80)
    print("PAC BOUNDS VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nTrials: {validation['n_trials']}")
    print(f"Test size: {validation['test_size']}")
    print(f"Thresholds: q̂₀={validation['threshold_0']:.4f}, q̂₁={validation['threshold_1']:.4f}")

    for scope in ["marginal", "class_0", "class_1"]:
        scope_name = scope.upper() if scope == "marginal" else f"CLASS {scope[-1]}"
        print(f"\n{'=' * 80}")
        print(f"{scope_name}")
        print("=" * 80)

        for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
            m = validation[scope][metric]
            q = m["quantiles"]
            coverage = m["empirical_coverage"]

            coverage_check = "✅" if coverage >= 0.90 else "❌"  # Assuming 90% PAC level

            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Empirical mean: {m['mean']:.4f}")
            print(f"  Expected (LOO): {m['expected']:.4f}")
            q_str = f"[5%: {q['q05']:.3f}, 25%: {q['q25']:.3f}, 50%: {q['q50']:.3f}, "
            q_str += f"75%: {q['q75']:.3f}, 95%: {q['q95']:.3f}]"
            print(f"  Quantiles:      {q_str}")
            print(f"  PAC bounds:     [{m['bounds'][0]:.4f}, {m['bounds'][1]:.4f}]")
            if not np.isnan(coverage):
                print(f"  Coverage:       {coverage:.1%} {coverage_check}")
            else:
                print("  Coverage:       N/A (no valid samples)")

    print("\n" + "=" * 80)
