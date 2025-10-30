"""Validation utilities for prediction interval operational bounds.

This module provides tools to empirically validate the theoretical bounds
by running simulations with fixed calibration thresholds on independent test sets.
Validates all reported methods (analytical, exact, hoeffding) when available.
"""

from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ssbc.utils import evaluate_test_dataset


def _validate_single_trial(
    trial_idx: int,
    simulator: Any,
    test_size: int,
    threshold_0: float,
    threshold_1: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a single validation trial.

    Parameters
    ----------
    trial_idx : int
        Trial index (for unique random seeds)
    simulator : Any
        Data generator
    test_size : int
        Size of test set
    threshold_0 : float
        Fixed threshold for class 0
    threshold_1 : float
        Fixed threshold for class 1
    seed : int, optional
        Base random seed

    Returns
    -------
    dict
        Trial results with marginal and per-class rates
    """
    # Create a new simulator instance with unique seed for this trial
    if seed is not None:
        # Set global random state first, then create simulator
        np.random.seed(seed + trial_idx)
        # Create a new simulator with the unique seed for this trial
        trial_simulator = type(simulator)(
            p_class1=simulator.p_class1,
            beta_params_class0=(simulator.a0, simulator.b0),
            beta_params_class1=(simulator.a1, simulator.b1),
            seed=seed + trial_idx,
        )
    else:
        trial_simulator = simulator

    # Generate independent test set
    labels_test, probs_test = trial_simulator.generate(test_size)

    # Use evaluate_test_dataset to compute rates (eliminates ~100 lines of duplicate code)
    eval_results = evaluate_test_dataset(
        test_labels=labels_test,
        test_probs=probs_test,
        threshold_0=threshold_0,
        threshold_1=threshold_1,
    )

    # Convert results to expected format
    marginal = eval_results["marginal"]
    class_0 = eval_results["class_0"]
    class_1 = eval_results["class_1"]

    return {
        "marginal": {
            "singleton": marginal["singleton_rate"],
            "doublet": marginal["doublet_rate"],
            "abstention": marginal["abstention_rate"],
            "singleton_error": marginal["singleton_error_rate"],
        },
        "class_0": {
            "singleton": class_0["singleton_rate"],
            "doublet": class_0["doublet_rate"],
            "abstention": class_0["abstention_rate"],
            "singleton_error": class_0["singleton_error_rate"],
        },
        "class_1": {
            "singleton": class_1["singleton_rate"],
            "doublet": class_1["doublet_rate"],
            "abstention": class_1["abstention_rate"],
            "singleton_error": class_1["singleton_error_rate"],
        },
    }


def validate_pac_bounds(
    report: dict[str, Any],
    simulator: Any,
    test_size: int,
    n_trials: int = 1000,
    seed: int | None = None,
    verbose: bool = True,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Empirically validate prediction interval operational bounds.

    Takes a PAC report from generate_rigorous_pac_report() and validates that
    the theoretical bounds actually hold in practice by:
    1. Extracting the FIXED thresholds from calibration
    2. Running n_trials simulations with fresh test sets
    3. Measuring empirical coverage of all reported bounds (analytical, exact, hoeffding)
    
    When the report includes method comparison (prediction_method="all"), validates
    all three methods separately. Otherwise, validates only the selected method.

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
    n_jobs : int, default=-1
        Number of parallel jobs for trial execution.
        -1 = use all cores (default), 1 = single-threaded, N = use N cores.

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
          - 'bounds': Selected/default bounds from report
          - 'expected': Expected rate from report
          - 'empirical_coverage': Fraction of trials within selected bounds
          - 'method_validations': Dict of method-specific validations (when available):
            - 'analytical': {bounds, empirical_coverage}
            - 'exact': {bounds, empirical_coverage}
            - 'hoeffding': {bounds, empirical_coverage}

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
    # Extract FIXED thresholds from calibration
    threshold_0 = report["calibration_result"][0]["threshold"]
    threshold_1 = report["calibration_result"][1]["threshold"]

    if verbose:
        print(f"Using fixed thresholds: q̂₀={threshold_0:.4f}, q̂₁={threshold_1:.4f}")
        print(f"Running {n_trials} trials with test_size={test_size}...")
        if n_jobs == -1:
            print("Using all available CPU cores for parallel execution")
        elif n_jobs == 1:
            print("Using single-threaded execution")
        else:
            print(f"Using {n_jobs} CPU cores for parallel execution")

    # Run trials in parallel, with safe fallback to serial if system forbids multiprocessing
    def _safe_parallel_map(n_jobs_local: int):
        try:
            return Parallel(n_jobs=n_jobs_local)(
                delayed(_validate_single_trial)(trial_idx, simulator, test_size, threshold_0, threshold_1, seed)
                for trial_idx in range(n_trials)
            )
        except Exception:
            return [
                _validate_single_trial(trial_idx, simulator, test_size, threshold_0, threshold_1, seed)
                for trial_idx in range(n_trials)
            ]

    trial_results = _safe_parallel_map(n_jobs)

    # Extract results from parallel execution
    marginal_singleton_rates = [result["marginal"]["singleton"] for result in trial_results]
    marginal_doublet_rates = [result["marginal"]["doublet"] for result in trial_results]
    marginal_abstention_rates = [result["marginal"]["abstention"] for result in trial_results]
    marginal_singleton_error_rates = [result["marginal"]["singleton_error"] for result in trial_results]

    class_0_singleton_rates = [result["class_0"]["singleton"] for result in trial_results]
    class_0_doublet_rates = [result["class_0"]["doublet"] for result in trial_results]
    class_0_abstention_rates = [result["class_0"]["abstention"] for result in trial_results]
    class_0_singleton_error_rates = [result["class_0"]["singleton_error"] for result in trial_results]

    class_1_singleton_rates = [result["class_1"]["singleton"] for result in trial_results]
    class_1_doublet_rates = [result["class_1"]["doublet"] for result in trial_results]
    class_1_abstention_rates = [result["class_1"]["abstention"] for result in trial_results]
    class_1_singleton_error_rates = [result["class_1"]["singleton_error"] for result in trial_results]

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

    # Extract levels from report
    params = report.get("parameters", {})
    pac_level_marginal = params.get("pac_level_marginal", 0.90)  # Default if missing
    pac_level_0 = params.get("pac_level_0", 0.90)
    pac_level_1 = params.get("pac_level_1", 0.90)
    ci_level = params.get("ci_level", 0.95)

    # Helper function to extract method comparison bounds
    def extract_method_bounds(pac_bounds: dict, metric_key: str) -> dict[str, tuple[float, float]]:
        """Extract bounds for all methods (analytical, exact, hoeffding) if available.
        
        Returns dict mapping method names to (lower, upper) bounds.
        If comparison not available, returns only the selected method bounds.
        """
        method_bounds = {}
        
        # Get the default selected bounds
        rate_key = f"{metric_key}_rate_bounds"
        default_bounds = pac_bounds.get(rate_key, (np.nan, np.nan))
        
        # Check if method comparison is available
        loo_diag = pac_bounds.get("loo_diagnostics", {})
        metric_diag = loo_diag.get(metric_key, {}) if loo_diag else {}
        
        if metric_diag and "comparison" in metric_diag:
            # Extract bounds for all methods from comparison table
            comp = metric_diag["comparison"]
            for method_name, method_lower, method_upper in zip(
                comp["method"], comp["lower"], comp["upper"]
            ):
                # Normalize method names
                method_key = method_name.lower().replace(" ", "_")
                if "analytical" in method_key:
                    method_bounds["analytical"] = (method_lower, method_upper)
                elif "exact" in method_key:
                    method_bounds["exact"] = (method_lower, method_upper)
                elif "hoeffding" in method_key:
                    method_bounds["hoeffding"] = (method_lower, method_upper)
        else:
            # No comparison available - use selected bounds with default method name
            selected_method = metric_diag.get("selected_method", "selected") if metric_diag else "selected"
            method_bounds[selected_method] = default_bounds
        
        return method_bounds

    # Helper function to validate a metric across all methods
    def validate_metric_all_methods(
        rates: np.ndarray,
        pac_bounds: dict,
        metric_key: str,
        use_nan_check: bool = False,
    ) -> dict[str, Any]:
        """Validate a metric across all reported methods."""
        # Extract bounds for all methods
        method_bounds_map = extract_method_bounds(pac_bounds, metric_key)
        
        # Get default bounds (selected method)
        rate_key = f"{metric_key}_rate_bounds"
        default_bounds = pac_bounds.get(rate_key, (np.nan, np.nan))
        
        # Get expected rate
        expected_key = f"expected_{metric_key}_rate"
        expected = pac_bounds.get(expected_key, np.nan)
        
        # Validate each method
        method_validations = {}
        for method_name, bounds in method_bounds_map.items():
            if use_nan_check:
                coverage = check_coverage_with_nan(rates, bounds)
            else:
                coverage = check_coverage(rates, bounds)
            method_validations[method_name] = {
                "bounds": bounds,
                "empirical_coverage": coverage,
            }
        
        return {
            "rates": rates,
            "mean": np.nanmean(rates) if use_nan_check else np.mean(rates),
            "quantiles": compute_quantiles(rates),
            "bounds": default_bounds,  # Selected/default bounds
            "expected": expected,
            "empirical_coverage": check_coverage_with_nan(rates, default_bounds) if use_nan_check else check_coverage(rates, default_bounds),
            "method_validations": method_validations,  # All method-specific validations
        }

    return {
        "n_trials": n_trials,
        "test_size": test_size,
        "threshold_0": threshold_0,
        "threshold_1": threshold_1,
        "pac_level_marginal": pac_level_marginal,
        "pac_level_0": pac_level_0,
        "pac_level_1": pac_level_1,
        "ci_level": ci_level,
        "marginal": {
            "singleton": validate_metric_all_methods(
                marginal_singleton_rates, pac_marg, "singleton", use_nan_check=False
            ),
            "doublet": validate_metric_all_methods(
                marginal_doublet_rates, pac_marg, "doublet", use_nan_check=False
            ),
            "abstention": validate_metric_all_methods(
                marginal_abstention_rates, pac_marg, "abstention", use_nan_check=False
            ),
            "singleton_error": validate_metric_all_methods(
                marginal_singleton_error_rates, pac_marg, "singleton_error", use_nan_check=True
            ),
        },
        "class_0": {
            "singleton": validate_metric_all_methods(
                class_0_singleton_rates, pac_0, "singleton", use_nan_check=False
            ),
            "doublet": validate_metric_all_methods(
                class_0_doublet_rates, pac_0, "doublet", use_nan_check=False
            ),
            "abstention": validate_metric_all_methods(
                class_0_abstention_rates, pac_0, "abstention", use_nan_check=False
            ),
            "singleton_error": validate_metric_all_methods(
                class_0_singleton_error_rates, pac_0, "singleton_error", use_nan_check=True
            ),
        },
        "class_1": {
            "singleton": validate_metric_all_methods(
                class_1_singleton_rates, pac_1, "singleton", use_nan_check=False
            ),
            "doublet": validate_metric_all_methods(
                class_1_doublet_rates, pac_1, "doublet", use_nan_check=False
            ),
            "abstention": validate_metric_all_methods(
                class_1_abstention_rates, pac_1, "abstention", use_nan_check=False
            ),
            "singleton_error": validate_metric_all_methods(
                class_1_singleton_error_rates, pac_1, "singleton_error", use_nan_check=True
            ),
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
    print("PREDICTION INTERVAL VALIDATION RESULTS")
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

            # Compare coverage against the prediction interval confidence level
            ci_level = validation.get("ci_level", 0.95)
            coverage_check = "✅" if coverage >= ci_level else "❌"

            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Empirical mean: {m['mean']:.4f}")
            print(f"  Expected (LOO): {m['expected']:.4f}")
            q_str = f"[5%: {q['q05']:.3f}, 25%: {q['q25']:.3f}, 50%: {q['q50']:.3f}, "
            q_str += f"75%: {q['q75']:.3f}, 95%: {q['q95']:.3f}]"
            print(f"  Quantiles:      {q_str}")
            print(f"  Selected bounds: [{m['bounds'][0]:.4f}, {m['bounds'][1]:.4f}]")
            if not np.isnan(coverage):
                print(f"  Selected coverage: {coverage:.1%} {coverage_check}")
            else:
                print("  Selected coverage: N/A (no valid samples)")
            
            # Show method-specific validations if available
            if "method_validations" in m and m["method_validations"]:
                print(f"  Method-specific validation:")
                # Show methods in order: analytical, exact, hoeffding
                method_order = ["analytical", "exact", "hoeffding"]
                for method_name in method_order:
                    if method_name in m["method_validations"]:
                        method_val = m["method_validations"][method_name]
                        method_bounds = method_val["bounds"]
                        method_coverage = method_val["empirical_coverage"]
                        method_check = "✅" if not np.isnan(method_coverage) and method_coverage >= ci_level else "❌"
                        method_width = method_bounds[1] - method_bounds[0]
                        if not np.isnan(method_coverage):
                            print(f"    {method_name.capitalize():12s}: [{method_bounds[0]:.4f}, {method_bounds[1]:.4f}] "
                                  f"(width: {method_width:.4f}, coverage: {method_coverage:.1%}) {method_check}")
                        else:
                            print(f"    {method_name.capitalize():12s}: [{method_bounds[0]:.4f}, {method_bounds[1]:.4f}] "
                                  f"(width: {method_width:.4f}, coverage: N/A)")
                # Also show any other methods (for backward compatibility)
                for method_name in m["method_validations"]:
                    if method_name not in method_order:
                        method_val = m["method_validations"][method_name]
                        method_bounds = method_val["bounds"]
                        method_coverage = method_val["empirical_coverage"]
                        method_check = "✅" if not np.isnan(method_coverage) and method_coverage >= ci_level else "❌"
                        method_width = method_bounds[1] - method_bounds[0]
                        if not np.isnan(method_coverage):
                            print(f"    {method_name.capitalize():12s}: [{method_bounds[0]:.4f}, {method_bounds[1]:.4f}] "
                                  f"(width: {method_width:.4f}, coverage: {method_coverage:.1%}) {method_check}")
                        else:
                            print(f"    {method_name.capitalize():12s}: [{method_bounds[0]:.4f}, {method_bounds[1]:.4f}] "
                                  f"(width: {method_width:.4f}, coverage: N/A)")

    print("\n" + "=" * 80)


def plot_validation_bounds(
    validation: dict[str, Any],
    metric: str = "singleton",
    show_detail: bool = True,
    main_figsize: tuple[int, int] = (18, 5),
    detail_figsize: tuple[int, int] = (18, 12),
    bins: int = 50,
    method_colors: dict[str, tuple[str, str]] | None = None,
    return_figs: bool = False,
) -> tuple | None:
    """Plot empirical distributions with prediction interval bounds for all methods.
    
    Creates visualization comparing empirical rates against bounds from analytical,
    exact, and hoeffding methods when available.
    
    Parameters
    ----------
    validation : dict
        Output from validate_pac_bounds() containing validation results
    metric : str, default="singleton"
        Which metric to plot. Options: "singleton", "doublet", "abstention", "singleton_error"
    show_detail : bool, default=True
        If True, also create detailed 3x3 grid showing each method separately
    main_figsize : tuple[int, int], default=(18, 5)
        Figure size for main comparison plot (width, height in inches)
    detail_figsize : tuple[int, int], default=(18, 12)
        Figure size for detailed method comparison grid (width, height in inches)
    bins : int, default=50
        Number of bins for histograms
    method_colors : dict or None, default=None
        Custom colors and linestyles for methods. Dict mapping method names to
        (color, linestyle) tuples. If None, uses default colors:
        - "analytical": ("#2E86AB", "solid")  # Blue
        - "exact": ("#A23B72", "dashed")       # Purple
        - "hoeffding": ("#F18F01", "dashdot")  # Orange
    return_figs : bool, default=False
        If True, returns matplotlib Figure objects for further customization.
        Returns (fig_main, fig_detail) or (fig_main, None) if show_detail=False.
        If False, calls plt.show() and returns None.
    
    Returns
    -------
    tuple or None
        If return_figs=True:
            - (fig_main, fig_detail) if show_detail=True
            - (fig_main, None) if show_detail=False
        If return_figs=False: None (displays plots directly)
    
    Examples
    --------
    >>> from ssbc import validate_pac_bounds, plot_validation_bounds
    >>> validation = validate_pac_bounds(report, sim, test_size=1000, n_trials=1000)
    >>> plot_validation_bounds(validation, metric="singleton")
    >>> # Or get figure objects for customization
    >>> fig_main, fig_detail = plot_validation_bounds(
    ...     validation, metric="singleton", return_figs=True
    ... )
    >>> fig_main.savefig("validation_main.png")
    
    Notes
    -----
    The main plot shows all three methods overlaid on the same histogram for easy
    comparison. The detailed plot shows each method separately in a 3x3 grid.
    Both plots include:
    - Empirical distribution histogram
    - Method-specific bounds (when method comparison available)
    - Expected value from LOO-CV
    - Empirical mean from validation trials
    - Coverage percentages for each method
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Validate metric exists
    valid_metrics = ["singleton", "doublet", "abstention", "singleton_error"]
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")
    
    # Check that metric exists in validation
    if metric not in validation["marginal"]:
        raise ValueError(
            f"Metric '{metric}' not found in validation results. "
            f"Available metrics: {list(validation['marginal'].keys())}"
        )
    
    # Default method colors if not provided
    if method_colors is None:
        method_colors = {
            "analytical": ("#2E86AB", "solid"),      # Blue
            "exact": ("#A23B72", "dashed"),          # Purple
            "hoeffding": ("#F18F01", "dashdot"),     # Orange
        }
    
    scopes = [("marginal", "Marginal"), ("class_0", "Class 0"), ("class_1", "Class 1")]
    method_order = ["analytical", "exact", "hoeffding"]
    
    # ===== Main comparison plot =====
    fig_main, axes = plt.subplots(1, 3, figsize=main_figsize)
    
    for idx, (scope, title) in enumerate(scopes):
        ax = axes[idx]
        rates = validation[scope][metric]["rates"]
        expected = validation[scope][metric]["expected"]
        
        # Plot histogram
        ax.hist(rates, bins=bins, alpha=0.6, edgecolor='black', color='gray', label='Empirical distribution')
        
        # Plot bounds for each method if available
        if "method_validations" in validation[scope][metric] and validation[scope][metric]["method_validations"]:
            for method_name in method_order:
                if method_name in validation[scope][metric]["method_validations"]:
                    method_val = validation[scope][metric]["method_validations"][method_name]
                    method_bounds = method_val["bounds"]
                    method_coverage = method_val["empirical_coverage"]
                    color, linestyle = method_colors.get(method_name, ("black", "solid"))
                    
                    # Plot method bounds
                    ax.axvline(
                        method_bounds[0], 
                        color=color, 
                        linestyle=linestyle, 
                        linewidth=2, 
                        alpha=0.8,
                        label=f'{method_name.capitalize()} lower'
                    )
                    ax.axvline(
                        method_bounds[1], 
                        color=color, 
                        linestyle=linestyle, 
                        linewidth=2, 
                        alpha=0.8,
                        label=f'{method_name.capitalize()} upper (cov: {method_coverage:.1%})'
                    )
        else:
            # Fallback to selected bounds if method comparison not available
            bounds = validation[scope][metric]["bounds"]
            coverage = validation[scope][metric]["empirical_coverage"]
            ax.axvline(bounds[0], color='r', linestyle='--', linewidth=2, label='Lower bound')
            ax.axvline(bounds[1], color='r', linestyle='--', linewidth=2, label=f'Upper bound (cov: {coverage:.1%})')
        
        # Plot expected and empirical mean
        ax.axvline(expected, color='g', linestyle='-', linewidth=2.5, label='Expected (LOO)', zorder=5)
        ax.axvline(np.mean(rates), color='darkorange', linestyle=':', linewidth=2, label='Empirical mean', zorder=5)
        
        ax.set_xlabel(f'{metric.replace("_", " ").title()} Rate', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        
        # Build title with coverage info
        if "method_validations" in validation[scope][metric] and validation[scope][metric]["method_validations"]:
            coverages = []
            for method_name in method_order:
                if method_name in validation[scope][metric]["method_validations"]:
                    cov = validation[scope][metric]["method_validations"][method_name]["empirical_coverage"]
                    if not np.isnan(cov):
                        coverages.append(f"{method_name[0].upper()}:{cov:.1%}")
            coverage_str = ", ".join(coverages) if coverages else ""
            ax.set_title(f'{title} {metric.replace("_", " ").title()} Rates\nCoverages: {coverage_str}', fontsize=11)
        else:
            cov = validation[scope][metric]["empirical_coverage"]
            ax.set_title(f'{title} {metric.replace("_", " ").title()} Rates\nCoverage: {cov:.1%}', fontsize=11)
        
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # ===== Detailed method comparison plot =====
    fig_detail = None
    if show_detail:
        if "method_validations" in validation["marginal"][metric] and validation["marginal"][metric]["method_validations"]:
            fig_detail, axes2 = plt.subplots(3, 3, figsize=detail_figsize)
            
            for method_idx, method_name in enumerate(method_order):
                if method_name in validation["marginal"][metric]["method_validations"]:
                    color, linestyle = method_colors.get(method_name, ("black", "solid"))
                    
                    for scope_idx, (scope, title) in enumerate(scopes):
                        ax = axes2[scope_idx, method_idx]
                        rates = validation[scope][metric]["rates"]
                        expected = validation[scope][metric]["expected"]
                        
                        if method_name in validation[scope][metric]["method_validations"]:
                            method_val = validation[scope][metric]["method_validations"][method_name]
                            method_bounds = method_val["bounds"]
                            method_coverage = method_val["empirical_coverage"]
                            
                            # Plot histogram
                            ax.hist(rates, bins=bins, alpha=0.6, edgecolor='black', color='lightgray')
                            
                            # Plot method bounds
                            ax.axvline(method_bounds[0], color=color, linestyle=linestyle, linewidth=2.5, 
                                     label=f'Lower [{method_bounds[0]:.3f}]', alpha=0.9)
                            ax.axvline(method_bounds[1], color=color, linestyle=linestyle, linewidth=2.5, 
                                     label=f'Upper [{method_bounds[1]:.3f}]', alpha=0.9)
                            
                            # Plot expected and mean
                            ax.axvline(expected, color='g', linestyle='-', linewidth=2, label='Expected', zorder=5)
                            ax.axvline(np.mean(rates), color='darkorange', linestyle=':', linewidth=2, 
                                     label=f'Mean [{np.mean(rates):.3f}]', zorder=5)
                            
                            ax.set_xlabel(f'{metric.replace("_", " ").title()} Rate', fontsize=10)
                            ax.set_ylabel('Frequency', fontsize=10)
                            cov_str = f'{method_coverage:.1%}' if not np.isnan(method_coverage) else 'N/A'
                            ax.set_title(f'{title} - {method_name.capitalize()}\nCoverage: {cov_str}', fontsize=10)
                            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
                            ax.grid(alpha=0.3)
                        else:
                            ax.text(0.5, 0.5, f'{method_name.capitalize()}\nnot available', 
                                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
                            ax.axis('off')
            
            plt.tight_layout()
    
    # Display or return figures
    if return_figs:
        return (fig_main, fig_detail) if show_detail else (fig_main, None)
    else:
        plt.show()
        return None


def validate_prediction_interval_calibration(
    simulator: Any,
    n_calibration: int,
    BigN: int,
    alpha_target: float | dict[int, float] = 0.10,
    delta: float | dict[int, float] = 0.10,
    test_size: int = 1000,
    n_trials: int = 1000,
    ci_level: float = 0.95,
    use_loo_correction: bool = True,
    prediction_method: str = "all",
    loo_inflation_factor: float | None = None,
    seed: int | None = None,
    n_jobs: int = -1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Validate that prediction interval confidence level holds across calibration datasets.
    
    This meta-validation checks if the nominal confidence level (e.g., 95%) actually
    holds when repeating the entire calibration+validation process many times with
    different calibration datasets.
    
    For each of BigN calibration datasets:
    1. Generate random calibration data
    2. Compute prediction interval bounds
    3. Validate bounds with many test sets
    4. Record empirical coverage
    
    Then aggregates results to see if ~95% of calibrations achieve ≥95% coverage.
    
    Parameters
    ----------
    simulator : DataGenerator
        Simulator for generating calibration and test data (e.g., BinaryClassifierSimulator)
    n_calibration : int
        Size of each calibration dataset
    BigN : int
        Number of different calibration datasets to test
    alpha_target : float or dict[int, float], default=0.10
        Target miscoverage rate per class
    delta : float or dict[int, float], default=0.10
        PAC risk tolerance for threshold calibration
    test_size : int, default=1000
        Size of each test set in validation
    n_trials : int, default=1000
        Number of test sets per calibration dataset (for validation)
    ci_level : float, default=0.95
        Nominal confidence level for prediction intervals (target to validate)
    use_loo_correction : bool, default=True
        Use LOO-corrected bounds
    prediction_method : str, default="all"
        Method for bounds computation ("all" to compare all methods)
    loo_inflation_factor : float, optional
        Manual override for LOO inflation factor
    seed : int, optional
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    verbose : bool, default=False
        If True, print progress for each calibration dataset
    
    Returns
    -------
    dict
        Meta-validation results with keys:
        - 'n_calibrations': BigN
        - 'n_calibration': Calibration dataset size
        - 'n_trials_per_calibration': n_trials
        - 'ci_level': Target confidence level
        - 'marginal': Dict with coverage statistics per method
        - 'class_0': Dict with coverage statistics per method
        - 'class_1': Dict with coverage statistics per method
        Each scope contains:
        - 'singleton', 'doublet', 'abstention', 'singleton_error': Dicts with:
          - 'selected': Coverage stats for selected bounds
          - 'analytical': Coverage stats if available
          - 'exact': Coverage stats if available
          - 'hoeffding': Coverage stats if available
          Each method has:
          - 'coverages': Array of empirical coverages across BigN calibrations
          - 'mean': Mean coverage
          - 'median': Median coverage
          - 'quantiles': {q05, q25, q50, q75, q95}
          - 'fraction_above_target': Fraction achieving ≥ci_level
          - 'fraction_above_95pct': Fraction achieving ≥95% (for comparison)
    
    Examples
    --------
    >>> from ssbc import BinaryClassifierSimulator, validate_prediction_interval_calibration
    >>> sim = BinaryClassifierSimulator(p_class1=0.2, seed=42)
    >>> results = validate_prediction_interval_calibration(
    ...     simulator=sim,
    ...     n_calibration=100,
    ...     BigN=50,
    ...     n_trials=500,
    ...     verbose=False
    ... )
    >>> print(f"Fraction achieving ≥95%: {results['marginal']['singleton']['selected']['fraction_above_target']:.1%}")
    """
    from ssbc import generate_rigorous_pac_report
    
    if seed is not None:
        np.random.seed(seed)
    
    # Helper to run validation for one calibration dataset
    def _single_calibration_validation(cal_idx: int) -> dict[str, Any]:
        """Run full validation for one calibration dataset."""
        # Generate unique seed for this calibration
        cal_seed = (seed + cal_idx * 10000) if seed is not None else None
        
        # Generate calibration dataset
        if cal_seed is not None:
            np.random.seed(cal_seed)
            cal_simulator = type(simulator)(
                p_class1=simulator.p_class1,
                beta_params_class0=(simulator.a0, simulator.b0),
                beta_params_class1=(simulator.a1, simulator.b1),
                seed=cal_seed,
            )
        else:
            cal_simulator = simulator
        
        cal_labels, cal_probs = cal_simulator.generate(n_calibration)
        
        # Generate PAC report (suppress verbose output)
        report = generate_rigorous_pac_report(
            labels=cal_labels,
            probs=cal_probs,
            alpha_target=alpha_target,
            delta=delta,
            test_size=test_size,
            ci_level=ci_level,
            use_union_bound=False,
            n_jobs=n_jobs,
            verbose=False,  # Always suppress report printing
            prediction_method=prediction_method,
            use_loo_correction=use_loo_correction,
            loo_inflation_factor=loo_inflation_factor,
        )
        
        # Validate bounds
        validation = validate_pac_bounds(
            report=report,
            simulator=cal_simulator,
            test_size=test_size,
            n_trials=n_trials,
            seed=cal_seed + 1 if cal_seed is not None else None,
            verbose=False,  # Suppress validation progress
            n_jobs=n_jobs,
        )
        
        # Extract coverages and quantiles for all methods
        result = {}
        for scope in ["marginal", "class_0", "class_1"]:
            result[scope] = {}
            for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
                m = validation[scope][metric]
                
                # Get observed quantiles from test set rates
                rates = m["rates"]
                # Filter NaN values for quantile calculation (especially for singleton_error)
                valid_rates = rates[~np.isnan(rates)] if np.any(np.isnan(rates)) else rates
                if len(valid_rates) > 0:
                    observed_q05 = float(np.percentile(valid_rates, 5))
                    observed_q95 = float(np.percentile(valid_rates, 95))
                else:
                    observed_q05 = np.nan
                    observed_q95 = np.nan
                
                # Get selected bounds
                selected_bounds = m["bounds"]
                
                result[scope][metric] = {
                    "selected": {
                        "coverage": m["empirical_coverage"],
                        "lower": float(selected_bounds[0]),
                        "upper": float(selected_bounds[1]),
                    },
                    "observed_q05": observed_q05,
                    "observed_q95": observed_q95,
                }
                
                # Extract method-specific coverages and bounds if available
                if "method_validations" in m and m["method_validations"]:
                    for method_name in ["analytical", "exact", "hoeffding"]:
                        if method_name in m["method_validations"]:
                            method_val = m["method_validations"][method_name]
                            method_bounds = method_val["bounds"]
                            result[scope][metric][method_name] = {
                                "coverage": method_val["empirical_coverage"],
                                "lower": float(method_bounds[0]),
                                "upper": float(method_bounds[1]),
                            }
        
        return result
    
    # Run BigN calibrations
    if verbose:
        print(f"Running meta-validation: {BigN} calibration datasets, {n_trials} trials each...")
        print(f"Progress: ", end="", flush=True)
    
    # Parallelize calibration validations
    def _safe_parallel_map_cal(n_jobs_local: int):
        try:
            results = Parallel(n_jobs=n_jobs_local, backend="threading")(
                delayed(_single_calibration_validation)(cal_idx) for cal_idx in range(BigN)
            )
            return results
        except Exception:
            # Fallback to serial
            return [_single_calibration_validation(cal_idx) for cal_idx in range(BigN)]
    
    all_results = _safe_parallel_map_cal(n_jobs)
    
    if verbose:
        print("Done!")
    
    # Aggregate coverages across all calibrations
    def compute_coverage_stats(coverages: np.ndarray, target_level: float) -> dict[str, Any]:
        """Compute statistics for coverage array."""
        valid = coverages[~np.isnan(coverages)]
        if len(valid) == 0:
            return {
                "coverages": coverages,
                "mean": np.nan,
                "median": np.nan,
                "quantiles": {
                    "q05": np.nan,
                    "q25": np.nan,
                    "q50": np.nan,
                    "q75": np.nan,
                    "q95": np.nan,
                },
                "fraction_above_target": np.nan,
                "fraction_above_95pct": np.nan,
            }
        
        return {
            "coverages": coverages,
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "quantiles": {
                "q05": float(np.percentile(valid, 5)),
                "q25": float(np.percentile(valid, 25)),
                "q50": float(np.median(valid)),
                "q75": float(np.percentile(valid, 75)),
                "q95": float(np.percentile(valid, 95)),
            },
            "fraction_above_target": float(np.mean(valid >= target_level)),
            "fraction_above_95pct": float(np.mean(valid >= 0.95)),
        }
    
    # Aggregate results
    aggregated = {
        "n_calibrations": BigN,
        "n_calibration": n_calibration,
        "n_trials_per_calibration": n_trials,
        "ci_level": ci_level,
    }
    
    for scope in ["marginal", "class_0", "class_1"]:
        aggregated[scope] = {}
        for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
            aggregated[scope][metric] = {}
            
            # Collect selected method coverages
            selected_coverages = np.array([
                all_results[i][scope][metric]["selected"]["coverage"] 
                for i in range(BigN)
            ])
            aggregated[scope][metric]["selected"] = compute_coverage_stats(selected_coverages, ci_level)
            
            # Collect method-specific coverages
            method_names = ["analytical", "exact", "hoeffding"]
            for method_name in method_names:
                method_coverages = []
                for i in range(BigN):
                    if method_name in all_results[i][scope][metric]:
                        method_coverages.append(all_results[i][scope][metric][method_name]["coverage"])
                    else:
                        method_coverages.append(np.nan)
                method_coverages = np.array(method_coverages)
                
                if not np.all(np.isnan(method_coverages)):
                    aggregated[scope][metric][method_name] = compute_coverage_stats(method_coverages, ci_level)
    
    # Store raw calibration data for quantile analysis
    aggregated["_raw_calibration_data"] = all_results
    
    return aggregated


def print_calibration_validation_results(results: dict[str, Any]) -> None:
    """Pretty print meta-validation results.
    
    Parameters
    ----------
    results : dict
        Output from validate_prediction_interval_calibration()
    
    Examples
    --------
    >>> results = validate_prediction_interval_calibration(...)
    >>> print_calibration_validation_results(results)
    """
    print("=" * 80)
    print("PREDICTION INTERVAL CALIBRATION VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Calibrations tested: {results['n_calibrations']}")
    print(f"  Calibration size: {results['n_calibration']}")
    print(f"  Trials per calibration: {results['n_trials_per_calibration']}")
    print(f"  Target confidence level: {results['ci_level']:.0%}")
    
    for scope in ["marginal", "class_0", "class_1"]:
        scope_name = scope.upper() if scope == "marginal" else f"CLASS {scope[-1]}"
        print(f"\n{'=' * 80}")
        print(f"{scope_name}")
        print("=" * 80)
        
        for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            
            # Check which methods are available
            available_methods = []
            if "selected" in results[scope][metric]:
                available_methods.append("selected")
            for method_name in ["analytical", "exact", "hoeffding"]:
                if method_name in results[scope][metric]:
                    available_methods.append(method_name)
            
            # Print stats for each method
            for method_name in available_methods:
                stats = results[scope][metric][method_name]
                q = stats["quantiles"]
                target = results["ci_level"]
                frac_target = stats["fraction_above_target"]
                frac_95 = stats["fraction_above_95pct"]
                
                method_display = "Selected" if method_name == "selected" else method_name.capitalize()
                
                print(f"  {method_display}:")
                print(f"    Mean coverage: {stats['mean']:.2%}")
                print(f"    Median coverage: {stats['median']:.2%}")
                print(f"    Quantiles: [5%: {q['q05']:.2%}, 50%: {q['q50']:.2%}, 95%: {q['q95']:.2%}]")
                print(f"    Fraction ≥ {target:.0%}: {frac_target:.1%} {'✅' if frac_target >= 0.95 else '❌'}")
                print(f"    Fraction ≥ 95%: {frac_95:.1%}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"\nIf well-calibrated, ~{results['ci_level']:.0%} of calibrations should achieve")
    print(f"≥{results['ci_level']:.0%} empirical coverage.")
    print("\n✅ = Calibration property holds (≥95% of calibrations meet target)")
    print("❌ = Calibration property may be violated")
    print("=" * 80)


def get_calibration_bounds_dataframe(
    results: dict[str, Any],
    scope: str | None = None,
    metric: str | None = None,
) -> Any:
    """Extract calibration bounds and observed quantiles as DataFrame.
    
    Converts the raw calibration data from validate_prediction_interval_calibration()
    into a pandas DataFrame format for easy plotting and analysis.
    
    Parameters
    ----------
    results : dict
        Output from validate_prediction_interval_calibration()
    scope : str, optional
        Filter to specific scope: "marginal", "class_0", or "class_1".
        If None, includes all scopes.
    metric : str, optional
        Filter to specific metric: "singleton", "doublet", "abstention", "singleton_error".
        If None, includes all metrics.
    
    Returns
    -------
    DataFrame
        Pandas DataFrame with columns:
        - calibration_idx: Index of calibration dataset (0 to BigN-1)
        - scope: marginal, class_0, or class_1
        - metric: singleton, doublet, abstention, singleton_error
        - observed_q05: 5th percentile of test set rates
        - observed_q95: 95th percentile of test set rates
        - selected_lower: Lower bound from selected method
        - selected_upper: Upper bound from selected method
        - analytical_lower: Lower bound from analytical method (NaN if not available)
        - analytical_upper: Upper bound from analytical method (NaN if not available)
        - exact_lower: Lower bound from exact method (NaN if not available)
        - exact_upper: Upper bound from exact method (NaN if not available)
        - hoeffding_lower: Lower bound from hoeffding method (NaN if not available)
        - hoeffding_upper: Upper bound from hoeffding method (NaN if not available)
    
    Examples
    --------
    >>> import pandas as pd
    >>> from ssbc import get_calibration_bounds_dataframe
    >>> results = validate_prediction_interval_calibration(...)
    >>> df = get_calibration_bounds_dataframe(results)
    >>> # Filter to singleton marginal
    >>> df_single = df[(df['scope'] == 'marginal') & (df['metric'] == 'singleton')]
    >>> # Plot lower bounds
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(df_single['analytical_lower'], df_single['observed_q05'])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")
    
    if "_raw_calibration_data" not in results:
        raise ValueError(
            "Results dict does not contain raw calibration data. "
            "Make sure to use validate_prediction_interval_calibration() output."
        )
    
    raw_data = results["_raw_calibration_data"]
    BigN = len(raw_data)
    
    # Build DataFrame rows
    rows = []
    scopes = ["marginal", "class_0", "class_1"] if scope is None else [scope]
    metrics = ["singleton", "doublet", "abstention", "singleton_error"] if metric is None else [metric]
    
    for cal_idx in range(BigN):
        for scope_name in scopes:
            for metric_name in metrics:
                if scope_name not in raw_data[cal_idx] or metric_name not in raw_data[cal_idx][scope_name]:
                    continue
                
                m = raw_data[cal_idx][scope_name][metric_name]
                
                row = {
                    "calibration_idx": cal_idx,
                    "scope": scope_name,
                    "metric": metric_name,
                    "observed_q05": m.get("observed_q05", np.nan),
                    "observed_q95": m.get("observed_q95", np.nan),
                }
                
                # Add selected bounds
                if "selected" in m:
                    row["selected_lower"] = m["selected"].get("lower", np.nan)
                    row["selected_upper"] = m["selected"].get("upper", np.nan)
                else:
                    row["selected_lower"] = np.nan
                    row["selected_upper"] = np.nan
                
                # Add method-specific bounds
                for method_name in ["analytical", "exact", "hoeffding"]:
                    if method_name in m:
                        row[f"{method_name}_lower"] = m[method_name].get("lower", np.nan)
                        row[f"{method_name}_upper"] = m[method_name].get("upper", np.nan)
                    else:
                        row[f"{method_name}_lower"] = np.nan
                        row[f"{method_name}_upper"] = np.nan
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_calibration_excess(
    df: Any,
    scope: str | None = None,
    metric: str | None = None,
    methods: list[str] | None = None,
    figsize: tuple[int, int] = (14, 6),
    bins: int = 30,
    return_fig: bool = False,
) -> Any:
    """Plot excess (difference between observed and predicted quantiles).
    
    Creates histograms showing:
    - Lower excess: observed_q05 - predicted_lower (positive = predicted too high)
    - Upper excess: predicted_upper - observed_q95 (positive = predicted too high)
    
    Parameters
    ----------
    df : DataFrame
        Output from get_calibration_bounds_dataframe()
    scope : str, optional
        Filter to specific scope: "marginal", "class_0", or "class_1".
        If None, uses all scopes (creates separate subplots).
    metric : str, optional
        Filter to specific metric: "singleton", "doublet", "abstention", "singleton_error".
        If None, uses all metrics (creates separate subplots).
    methods : list[str], optional
        Methods to plot: ["analytical", "exact", "hoeffding"].
        If None, plots all available methods.
    figsize : tuple[int, int], default=(14, 6)
        Figure size (width, height in inches)
    bins : int, default=30
        Number of histogram bins
    return_fig : bool, default=False
        If True, returns matplotlib Figure object. If False, calls plt.show()
    
    Returns
    -------
    Figure or None
        If return_fig=True, returns Figure object. Otherwise None.
    
    Examples
    --------
    >>> from ssbc import get_calibration_bounds_dataframe, plot_calibration_excess
    >>> results = validate_prediction_interval_calibration(...)
    >>> df = get_calibration_bounds_dataframe(results)
    >>> # Plot for singleton marginal
    >>> df_single = df[(df['scope'] == 'marginal') & (df['metric'] == 'singleton')]
    >>> plot_calibration_excess(df_single, scope='marginal', metric='singleton')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    # Filter DataFrame
    df_filtered = df.copy()
    if scope is not None:
        df_filtered = df_filtered[df_filtered['scope'] == scope]
    if metric is not None:
        df_filtered = df_filtered[df_filtered['metric'] == metric]
    
    if len(df_filtered) == 0:
        raise ValueError("No data matching the specified scope/metric filters")
    
    # Determine which methods are available
    available_methods = []
    method_colors = {
        "analytical": "#2E86AB",  # Blue
        "exact": "#A23B72",        # Purple
        "hoeffding": "#F18F01",    # Orange
    }
    
    if methods is None:
        # Check which methods have non-NaN data
        for method_name in ["analytical", "exact", "hoeffding"]:
            if f"{method_name}_lower" in df_filtered.columns:
                if df_filtered[f"{method_name}_lower"].notna().any():
                    available_methods.append(method_name)
    else:
        available_methods = [m for m in methods if f"{m}_lower" in df_filtered.columns]
    
    if len(available_methods) == 0:
        raise ValueError("No method data available in DataFrame")
    
    # Create figure with two subplots (lower and upper excess)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot lower excess: observed_q05 - predicted_lower
    # Positive excess = predicted lower bound is too high (conservative)
    # Negative excess = predicted lower bound is too low (underestimates)
    for method_name in available_methods:
        lower_pred = df_filtered[f"{method_name}_lower"]
        lower_obs = df_filtered['observed_q05']
        
        # Compute excess only for non-NaN pairs
        valid_mask = lower_pred.notna() & lower_obs.notna()
        if valid_mask.sum() > 0:
            excess_lower = lower_obs[valid_mask] - lower_pred[valid_mask]
            color = method_colors.get(method_name, "gray")
            
            ax1.hist(
                excess_lower,
                bins=bins,
                alpha=0.6,
                label=f'{method_name.capitalize()} (n={len(excess_lower)})',
                color=color,
                edgecolor='black',
            )
    
    ax1.axvline(0, color='k', linestyle='--', linewidth=2, label='Perfect (0 excess)', zorder=10)
    ax1.set_xlabel('Lower Excess: Observed Q05 - Predicted Lower', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Lower Bound Calibration', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Add interpretation text
    ax1.text(0.02, 0.98, 'Positive = Predicted too high (conservative)\nNegative = Predicted too low (risky)',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot upper excess: predicted_upper - observed_q95
    # Positive excess = predicted upper bound is higher than needed (conservative)
    # Negative excess = predicted upper bound is too low (underestimates)
    for method_name in available_methods:
        upper_pred = df_filtered[f"{method_name}_upper"]
        upper_obs = df_filtered['observed_q95']
        
        # Compute excess only for non-NaN pairs
        valid_mask = upper_pred.notna() & upper_obs.notna()
        if valid_mask.sum() > 0:
            excess_upper = upper_pred[valid_mask] - upper_obs[valid_mask]
            color = method_colors.get(method_name, "gray")
            
            ax2.hist(
                excess_upper,
                bins=bins,
                alpha=0.6,
                label=f'{method_name.capitalize()} (n={len(excess_upper)})',
                color=color,
                edgecolor='black',
            )
    
    ax2.axvline(0, color='k', linestyle='--', linewidth=2, label='Perfect (0 excess)', zorder=10)
    ax2.set_xlabel('Upper Excess: Predicted Upper - Observed Q95', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Upper Bound Calibration', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Add interpretation text
    ax2.text(0.02, 0.98, 'Positive = Predicted too high (conservative)\nNegative = Predicted too low (risky)',
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add overall title with scope/metric info
    scope_str = scope if scope else "All Scopes"
    metric_str = metric if metric else "All Metrics"
    fig.suptitle(f'Calibration Excess Analysis: {scope_str} - {metric_str}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None
