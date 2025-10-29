"""Simplified operational bounds for fixed calibration (LOO-CV + CP)."""

import numpy as np
from joblib import Parallel, delayed

from ssbc.core import SSBCResult
from ssbc.loo_uncertainty import compute_robust_prediction_bounds
from ssbc.statistics import prediction_bounds


def _evaluate_loo_single_sample_marginal(
    idx: int,
    labels: np.ndarray,
    probs: np.ndarray,
    k_0: int,
    k_1: int,
) -> tuple[int, int, int, int]:
    """Evaluate single LOO fold for marginal operational rates.

    Parameters
    ----------
    k_0, k_1 : int
        Quantile positions (1-indexed) from SSBC calibration

    Returns
    -------
    tuple[int, int, int, int]
        (is_singleton, is_doublet, is_abstention, is_singleton_correct)
    """
    mask_0 = labels == 0
    mask_1 = labels == 1

    # Compute LOO thresholds (using FIXED k positions)
    # Class 0
    if mask_0[idx]:
        scores_0_loo = 1.0 - probs[mask_0, 0]
        mask_0_idx = np.where(mask_0)[0]
        loo_position = np.where(mask_0_idx == idx)[0][0]
        scores_0_loo = np.delete(scores_0_loo, loo_position)
    else:
        scores_0_loo = 1.0 - probs[mask_0, 0]

    sorted_0_loo = np.sort(scores_0_loo)
    threshold_0_loo = sorted_0_loo[min(k_0 - 1, len(sorted_0_loo) - 1)]

    # Class 1
    if mask_1[idx]:
        scores_1_loo = 1.0 - probs[mask_1, 1]
        mask_1_idx = np.where(mask_1)[0]
        loo_position = np.where(mask_1_idx == idx)[0][0]
        scores_1_loo = np.delete(scores_1_loo, loo_position)
    else:
        scores_1_loo = 1.0 - probs[mask_1, 1]

    sorted_1_loo = np.sort(scores_1_loo)
    threshold_1_loo = sorted_1_loo[min(k_1 - 1, len(sorted_1_loo) - 1)]

    # Evaluate on held-out sample
    score_0 = 1.0 - probs[idx, 0]
    score_1 = 1.0 - probs[idx, 1]
    true_label = labels[idx]

    in_0 = score_0 <= threshold_0_loo
    in_1 = score_1 <= threshold_1_loo

    # Determine prediction set type
    if in_0 and in_1:
        is_singleton, is_doublet, is_abstention = 0, 1, 0
        is_singleton_correct = 0
    elif in_0 or in_1:
        is_singleton, is_doublet, is_abstention = 1, 0, 0
        is_singleton_correct = 1 if (in_0 and true_label == 0) or (in_1 and true_label == 1) else 0
    else:
        is_singleton, is_doublet, is_abstention = 0, 0, 1
        is_singleton_correct = 0

    return is_singleton, is_doublet, is_abstention, is_singleton_correct


def compute_pac_operational_bounds_marginal(
    ssbc_result_0: SSBCResult,
    ssbc_result_1: SSBCResult,
    labels: np.ndarray,
    probs: np.ndarray,
    test_size: int,  # Now used for prediction bounds
    ci_level: float = 0.95,
    pac_level: float = 0.95,  # Kept for API compatibility (not used)
    use_union_bound: bool = True,
    n_jobs: int = -1,
    prediction_method: str = "simple",
) -> dict:
    """Compute marginal operational bounds for FIXED calibration via LOO-CV.

    Enhanced approach:
    1. Use FIXED u_star positions from SSBC calibration
    2. Run LOO-CV to get unbiased rate estimates
    3. Apply prediction bounds accounting for both calibration and test set sampling uncertainty
    4. Optional union bound for simultaneous guarantees

    This models: "Given fixed calibration, what are rate distributions on future test sets?"
    The prediction bounds account for both calibration uncertainty and test set sampling variability.

    Parameters
    ----------
    ssbc_result_0 : SSBCResult
        SSBC result for class 0
    ssbc_result_1 : SSBCResult
        SSBC result for class 1
    labels : np.ndarray
        True labels
    probs : np.ndarray
        Predicted probabilities
    test_size : int
        Expected test set size for prediction bounds. Used to account for test set sampling uncertainty.
    ci_level : float, default=0.95
        Confidence level for prediction bounds
    use_union_bound : bool, default=True
        Apply Bonferroni for simultaneous guarantees
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)

    Returns
    -------
    dict
        Operational bounds with keys:
        - 'singleton_rate_bounds': [L, U]
        - 'doublet_rate_bounds': [L, U]
        - 'abstention_rate_bounds': [L, U]
        - 'singleton_error_rate_bounds': [L, U]
        - 'expected_*_rate': point estimates
    """
    n = len(labels)

    # Compute k (quantile position) from SSBC-corrected alpha
    # k = ceil((n_class + 1) * (1 - alpha_corrected))
    n_0 = ssbc_result_0.n
    n_1 = ssbc_result_1.n
    k_0 = int(np.ceil((n_0 + 1) * (1 - ssbc_result_0.alpha_corrected)))
    k_1 = int(np.ceil((n_1 + 1) * (1 - ssbc_result_1.alpha_corrected)))

    # Parallel LOO-CV: evaluate each sample
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_single_sample_marginal)(idx, labels, probs, k_0, k_1) for idx in range(n)
    )

    # Aggregate results
    results_array = np.array(results)
    n_singletons = int(np.sum(results_array[:, 0]))
    n_doublets = int(np.sum(results_array[:, 1]))
    n_abstentions = int(np.sum(results_array[:, 2]))
    n_singletons_correct = int(np.sum(results_array[:, 3]))

    # Point estimates
    singleton_rate = n_singletons / n
    doublet_rate = n_doublets / n
    abstention_rate = n_abstentions / n
    n_errors = n_singletons - n_singletons_correct
    singleton_error_rate = n_errors / n_singletons if n_singletons > 0 else 0.0

    # Apply prediction bounds accounting for both calibration and test set sampling uncertainty
    # These bound operational rates on future test sets of size test_size
    # SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test)) accounts for both sources of uncertainty

    n_metrics = 4
    if use_union_bound:
        adjusted_ci_level = 1 - (1 - ci_level) / n_metrics
    else:
        adjusted_ci_level = ci_level

    # Use prediction bounds instead of Clopper-Pearson for operational rates
    singleton_lower, singleton_upper = prediction_bounds(
        n_singletons, n, test_size, adjusted_ci_level, prediction_method
    )
    doublet_lower, doublet_upper = prediction_bounds(n_doublets, n, test_size, adjusted_ci_level, prediction_method)
    abstention_lower, abstention_upper = prediction_bounds(
        n_abstentions, n, test_size, adjusted_ci_level, prediction_method
    )

    # Singleton error (conditioned on singletons) - use prediction bounds on error rate
    if n_singletons > 0:
        error_lower, error_upper = prediction_bounds(
            n_errors, n_singletons, test_size, adjusted_ci_level, prediction_method
        )
    else:
        error_lower = 0.0
        error_upper = 1.0

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        "expected_singleton_rate": singleton_rate,
        "expected_doublet_rate": doublet_rate,
        "expected_abstention_rate": abstention_rate,
        "expected_singleton_error_rate": singleton_error_rate,
        "n_grid_points": 1,  # Single scenario (fixed thresholds)
        "pac_level": adjusted_ci_level,
        "ci_level": ci_level,
        "test_size": n,
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
    }


def compute_pac_operational_bounds_marginal_loo_corrected(
    ssbc_result_0: SSBCResult,
    ssbc_result_1: SSBCResult,
    labels: np.ndarray,
    probs: np.ndarray,
    test_size: int,
    ci_level: float = 0.95,
    pac_level: float = 0.95,  # Kept for API compatibility (not used)
    use_union_bound: bool = True,
    n_jobs: int = -1,
    prediction_method: str = "auto",
    loo_inflation_factor: float | None = None,
    verbose: bool = True,
) -> dict:
    """Compute marginal operational bounds with LOO-CV uncertainty correction.

    This function uses the new LOO uncertainty quantification that properly
    accounts for all four sources of uncertainty:
    1. LOO-CV correlation structure
    2. Threshold calibration uncertainty
    3. Parameter estimation uncertainty
    4. Test sampling uncertainty

    This is the RECOMMENDED function for small calibration sets (n=20-40).

    Parameters
    ----------
    ssbc_result_0 : SSBCResult
        SSBC result for class 0
    ssbc_result_1 : SSBCResult
        SSBC result for class 1
    labels : np.ndarray
        True labels
    probs : np.ndarray
        Predicted probabilities
    test_size : int
        Expected test set size for prediction bounds
    ci_level : float, default=0.95
        Confidence level for prediction bounds
    use_union_bound : bool, default=True
        Apply Bonferroni for simultaneous guarantees
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    prediction_method : str, default="auto"
        Method for LOO uncertainty quantification:
        - "auto": Automatically select best method
        - "analytical": Method 1 (recommended for n>=40)
        - "exact": Method 2 (recommended for n=20-40)
        - "hoeffding": Method 3 (ultra-conservative)
        - "all": Compare all methods
    loo_inflation_factor : float, optional
        Manual override for LOO variance inflation factor. If None, automatically estimated.
        Typical values: 1.0 (no inflation), 2.0 (standard LOO), 1.5-2.5 (empirical range)

    Returns
    -------
    dict
        Operational bounds with keys:
        - 'singleton_rate_bounds': [L, U]
        - 'doublet_rate_bounds': [L, U]
        - 'abstention_rate_bounds': [L, U]
        - 'singleton_error_rate_bounds': [L, U]
        - 'expected_*_rate': point estimates
        - 'loo_diagnostics': Detailed LOO uncertainty analysis
    """
    n = len(labels)

    # Compute k (quantile position) from SSBC-corrected alpha
    n_0 = ssbc_result_0.n
    n_1 = ssbc_result_1.n
    k_0 = int(np.ceil((n_0 + 1) * (1 - ssbc_result_0.alpha_corrected)))
    k_1 = int(np.ceil((n_1 + 1) * (1 - ssbc_result_1.alpha_corrected)))

    # Parallel LOO-CV: evaluate each sample
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_single_sample_marginal)(idx, labels, probs, k_0, k_1) for idx in range(n)
    )

    # Aggregate results
    results_array = np.array(results)
    n_singletons = int(np.sum(results_array[:, 0]))
    n_doublets = int(np.sum(results_array[:, 1]))
    n_abstentions = int(np.sum(results_array[:, 2]))
    n_singletons_correct = int(np.sum(results_array[:, 3]))

    # Point estimates
    singleton_rate = n_singletons / n
    doublet_rate = n_doublets / n
    abstention_rate = n_abstentions / n
    n_errors = n_singletons - n_singletons_correct
    singleton_error_rate = n_errors / n_singletons if n_singletons > 0 else 0.0

    # Convert to binary LOO predictions for each rate type
    singleton_loo_preds = results_array[:, 0].astype(int)
    doublet_loo_preds = results_array[:, 1].astype(int)
    abstention_loo_preds = results_array[:, 2].astype(int)
    error_loo_preds = np.zeros(n, dtype=int)
    if n_singletons > 0:
        # Error rate: 1 if singleton and incorrect, 0 otherwise
        error_loo_preds = (results_array[:, 0] == 1) & (results_array[:, 3] == 0)

    # Apply union bound adjustment
    n_metrics = 4
    if use_union_bound:
        adjusted_ci_level = 1 - (1 - ci_level) / n_metrics
    else:
        adjusted_ci_level = ci_level

    # Compute LOO-corrected bounds for each rate type
    singleton_lower, singleton_upper, singleton_report = compute_robust_prediction_bounds(
        singleton_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    doublet_lower, doublet_upper, doublet_report = compute_robust_prediction_bounds(
        doublet_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    abstention_lower, abstention_upper, abstention_report = compute_robust_prediction_bounds(
        abstention_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    # Singleton error (conditioned on singletons)
    if n_singletons > 0:
        error_lower, error_upper, error_report = compute_robust_prediction_bounds(
            error_loo_preds,
            test_size,
            1 - adjusted_ci_level,
            method=prediction_method,
            inflation_factor=loo_inflation_factor,
            verbose=verbose,
        )
    else:
        error_lower = 0.0
        error_upper = 1.0
        error_report = {"selected_method": "no_singletons", "diagnostics": {}}

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        "expected_singleton_rate": singleton_rate,
        "expected_doublet_rate": doublet_rate,
        "expected_abstention_rate": abstention_rate,
        "expected_singleton_error_rate": singleton_error_rate,
        "n_grid_points": 1,  # Single scenario (fixed thresholds)
        "pac_level": adjusted_ci_level,
        "ci_level": ci_level,
        "test_size": test_size,
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
        "loo_diagnostics": {
            "singleton": singleton_report,
            "doublet": doublet_report,
            "abstention": abstention_report,
            "singleton_error": error_report,
        },
    }


def _evaluate_loo_single_sample_perclass(
    idx: int,
    labels: np.ndarray,
    probs: np.ndarray,
    k_0: int,
    k_1: int,
    class_label: int,
) -> tuple[int, int, int, int]:
    """Evaluate single LOO fold for per-class operational rates.

    Returns
    -------
    tuple[int, int, int, int]
        (is_singleton, is_doublet, is_abstention, is_singleton_correct)
    """
    # Only evaluate if sample is from class_label
    if labels[idx] != class_label:
        return 0, 0, 0, 0

    mask_0 = labels == 0
    mask_1 = labels == 1

    # Compute LOO thresholds
    # Class 0
    if mask_0[idx]:
        scores_0_loo = 1.0 - probs[mask_0, 0]
        mask_0_idx = np.where(mask_0)[0]
        loo_position = np.where(mask_0_idx == idx)[0][0]
        scores_0_loo = np.delete(scores_0_loo, loo_position)
    else:
        scores_0_loo = 1.0 - probs[mask_0, 0]

    sorted_0_loo = np.sort(scores_0_loo)
    threshold_0_loo = sorted_0_loo[min(k_0 - 1, len(sorted_0_loo) - 1)]

    # Class 1
    if mask_1[idx]:
        scores_1_loo = 1.0 - probs[mask_1, 1]
        mask_1_idx = np.where(mask_1)[0]
        loo_position = np.where(mask_1_idx == idx)[0][0]
        scores_1_loo = np.delete(scores_1_loo, loo_position)
    else:
        scores_1_loo = 1.0 - probs[mask_1, 1]

    sorted_1_loo = np.sort(scores_1_loo)
    threshold_1_loo = sorted_1_loo[min(k_1 - 1, len(sorted_1_loo) - 1)]

    # Evaluate on held-out sample
    score_0 = 1.0 - probs[idx, 0]
    score_1 = 1.0 - probs[idx, 1]
    true_label = labels[idx]

    in_0 = score_0 <= threshold_0_loo
    in_1 = score_1 <= threshold_1_loo

    # Determine prediction set type
    if in_0 and in_1:
        is_singleton, is_doublet, is_abstention = 0, 1, 0
        is_singleton_correct = 0
    elif in_0 or in_1:
        is_singleton, is_doublet, is_abstention = 1, 0, 0
        is_singleton_correct = 1 if (in_0 and true_label == 0) or (in_1 and true_label == 1) else 0
    else:
        is_singleton, is_doublet, is_abstention = 0, 0, 1
        is_singleton_correct = 0

    return is_singleton, is_doublet, is_abstention, is_singleton_correct


def compute_pac_operational_bounds_perclass(
    ssbc_result_0: SSBCResult,
    ssbc_result_1: SSBCResult,
    labels: np.ndarray,
    probs: np.ndarray,
    class_label: int,
    test_size: int,  # Now used for prediction bounds
    ci_level: float = 0.95,
    pac_level: float = 0.95,  # Kept for API compatibility (not used)
    use_union_bound: bool = True,
    n_jobs: int = -1,
    prediction_method: str = "simple",
    loo_inflation_factor: float | None = None,
) -> dict:
    """Compute per-class operational bounds for FIXED calibration via LOO-CV.

    Parameters
    ----------
    class_label : int
        Which class to analyze (0 or 1)

    loo_inflation_factor : float, optional
        Manual override for LOO variance inflation factor. If None, not used.
        Note: Per-class bounds currently use standard prediction bounds, not LOO-corrected bounds.
        This parameter is included for API compatibility and future use.

    Notes
    -----
    The test_size is automatically adjusted based on the expected class distribution:
    expected_n_class_test = test_size * (n_class_cal / n_total)

    This ensures proper uncertainty quantification for class-specific rates.

    Other parameters same as marginal version.

    Returns
    -------
    dict
        Per-class operational bounds
    """
    # Compute k from alpha_corrected
    n_0 = ssbc_result_0.n
    n_1 = ssbc_result_1.n
    k_0 = int(np.ceil((n_0 + 1) * (1 - ssbc_result_0.alpha_corrected)))
    k_1 = int(np.ceil((n_1 + 1) * (1 - ssbc_result_1.alpha_corrected)))

    # Parallel LOO-CV: evaluate each sample
    n = len(labels)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_single_sample_perclass)(idx, labels, probs, k_0, k_1, class_label) for idx in range(n)
    )

    # Aggregate results (only from class_label samples)
    results_array = np.array(results)
    n_singletons = int(np.sum(results_array[:, 0]))
    n_doublets = int(np.sum(results_array[:, 1]))
    n_abstentions = int(np.sum(results_array[:, 2]))
    n_singletons_correct = int(np.sum(results_array[:, 3]))

    # Number of class_label samples in calibration
    n_class_cal = np.sum(labels == class_label)

    # Estimate expected class distribution in test set
    # Use calibration class distribution as estimate for test set
    n_total = len(labels)
    class_rate_cal = n_class_cal / n_total
    expected_n_class_test = int(test_size * class_rate_cal)

    # Ensure minimum test size for numerical stability
    expected_n_class_test = max(expected_n_class_test, 1)

    # Point estimates
    singleton_rate = n_singletons / n_class_cal
    doublet_rate = n_doublets / n_class_cal
    abstention_rate = n_abstentions / n_class_cal
    n_errors = n_singletons - n_singletons_correct
    singleton_error_rate = n_errors / n_singletons if n_singletons > 0 else 0.0

    # Apply prediction bounds accounting for both calibration and test set sampling uncertainty
    # These bound operational rates on future test sets of size expected_n_class_test
    # SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test)) accounts for both sources of uncertainty

    n_metrics = 4
    if use_union_bound:
        adjusted_ci_level = 1 - (1 - ci_level) / n_metrics
    else:
        adjusted_ci_level = ci_level

    # Use prediction bounds instead of Clopper-Pearson for operational rates
    # Use expected class-specific test size for proper uncertainty quantification
    singleton_lower, singleton_upper = prediction_bounds(
        n_singletons, n_class_cal, expected_n_class_test, adjusted_ci_level, prediction_method
    )
    doublet_lower, doublet_upper = prediction_bounds(
        n_doublets, n_class_cal, expected_n_class_test, adjusted_ci_level, prediction_method
    )
    abstention_lower, abstention_upper = prediction_bounds(
        n_abstentions, n_class_cal, expected_n_class_test, adjusted_ci_level, prediction_method
    )

    # Singleton error (conditioned on singletons) - use prediction bounds on error rate
    if n_singletons > 0:
        error_lower, error_upper = prediction_bounds(
            n_errors, n_singletons, expected_n_class_test, adjusted_ci_level, prediction_method
        )
    else:
        error_lower = 0.0
        error_upper = 1.0

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        "expected_singleton_rate": singleton_rate,
        "expected_doublet_rate": doublet_rate,
        "expected_abstention_rate": abstention_rate,
        "expected_singleton_error_rate": singleton_error_rate,
        "n_grid_points": 1,
        "pac_level": adjusted_ci_level,
        "ci_level": ci_level,
        "test_size": n_class_cal,  # Use calibration class size
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
    }


def compute_pac_operational_bounds_perclass_loo_corrected(
    ssbc_result_0: SSBCResult,
    ssbc_result_1: SSBCResult,
    labels: np.ndarray,
    probs: np.ndarray,
    class_label: int,
    test_size: int,
    ci_level: float = 0.95,
    pac_level: float = 0.95,  # Kept for API compatibility (not used)
    use_union_bound: bool = True,
    n_jobs: int = -1,
    prediction_method: str = "auto",
    loo_inflation_factor: float | None = None,
    verbose: bool = True,
) -> dict:
    """Compute per-class operational bounds with LOO-CV uncertainty correction.

    This function uses LOO uncertainty quantification for per-class bounds,
    enabling method comparison ("all") for individual classes.

    Parameters
    ----------
    ssbc_result_0 : SSBCResult
        SSBC result for class 0
    ssbc_result_1 : SSBCResult
        SSBC result for class 1
    labels : np.ndarray
        True labels
    probs : np.ndarray
        Predicted probabilities
    class_label : int
        Which class to analyze (0 or 1)
    test_size : int
        Expected test set size for prediction bounds
    ci_level : float, default=0.95
        Confidence level for prediction bounds
    use_union_bound : bool, default=True
        Apply Bonferroni for simultaneous guarantees
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    prediction_method : str, default="auto"
        Method for LOO uncertainty quantification:
        - "auto": Automatically select best method
        - "analytical": Method 1 (recommended for n>=40)
        - "exact": Method 2 (recommended for n=20-40)
        - "hoeffding": Method 3 (ultra-conservative)
        - "all": Compare all methods
    loo_inflation_factor : float, optional
        Manual override for LOO variance inflation factor. If None, automatically estimated.
    verbose : bool, default=True
        Print diagnostic information

    Returns
    -------
    dict
        Per-class operational bounds with LOO diagnostics (when method="all")
    """
    # Compute k from alpha_corrected
    n_0 = ssbc_result_0.n
    n_1 = ssbc_result_1.n
    k_0 = int(np.ceil((n_0 + 1) * (1 - ssbc_result_0.alpha_corrected)))
    k_1 = int(np.ceil((n_1 + 1) * (1 - ssbc_result_1.alpha_corrected)))

    # Parallel LOO-CV: evaluate each sample
    n = len(labels)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_single_sample_perclass)(idx, labels, probs, k_0, k_1, class_label) for idx in range(n)
    )

    # Aggregate results (only from class_label samples)
    results_array = np.array(results)
    n_singletons = int(np.sum(results_array[:, 0]))
    n_doublets = int(np.sum(results_array[:, 1]))
    n_abstentions = int(np.sum(results_array[:, 2]))
    n_singletons_correct = int(np.sum(results_array[:, 3]))

    # Number of class_label samples in calibration
    n_class_cal = np.sum(labels == class_label)

    # Estimate expected class distribution in test set
    n_total = len(labels)
    class_rate_cal = n_class_cal / n_total
    expected_n_class_test = int(test_size * class_rate_cal)
    expected_n_class_test = max(expected_n_class_test, 1)

    # Point estimates
    singleton_rate = n_singletons / n_class_cal
    doublet_rate = n_doublets / n_class_cal
    abstention_rate = n_abstentions / n_class_cal
    n_errors = n_singletons - n_singletons_correct
    singleton_error_rate = n_errors / n_singletons if n_singletons > 0 else 0.0

    # Convert to binary LOO predictions for each rate type
    singleton_loo_preds = results_array[:, 0].astype(int)
    doublet_loo_preds = results_array[:, 1].astype(int)
    abstention_loo_preds = results_array[:, 2].astype(int)
    error_loo_preds = np.zeros(n, dtype=int)
    if n_singletons > 0:
        # Error rate: 1 if singleton and incorrect, 0 otherwise
        error_loo_preds = (results_array[:, 0] == 1) & (results_array[:, 3] == 0)

    # Apply union bound adjustment
    n_metrics = 4
    if use_union_bound:
        adjusted_ci_level = 1 - (1 - ci_level) / n_metrics
    else:
        adjusted_ci_level = ci_level

    # Compute LOO-corrected bounds for each rate type
    singleton_lower, singleton_upper, singleton_report = compute_robust_prediction_bounds(
        singleton_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    doublet_lower, doublet_upper, doublet_report = compute_robust_prediction_bounds(
        doublet_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    abstention_lower, abstention_upper, abstention_report = compute_robust_prediction_bounds(
        abstention_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    error_lower, error_upper, error_report = compute_robust_prediction_bounds(
        error_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    # Build result dict
    result = {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        "expected_singleton_rate": singleton_rate,
        "expected_doublet_rate": doublet_rate,
        "expected_abstention_rate": abstention_rate,
        "expected_singleton_error_rate": singleton_error_rate,
        "n_grid_points": 1,
        "pac_level": adjusted_ci_level,
        "ci_level": ci_level,
        "test_size": n_class_cal,
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
    }

    # Add LOO diagnostics if method="all"
    if prediction_method == "all":
        result["loo_diagnostics"] = {
            "singleton": singleton_report,
            "doublet": doublet_report,
            "abstention": abstention_report,
            "singleton_error": error_report,
        }

    return result
