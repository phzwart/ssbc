"""Simplified operational bounds for fixed calibration (LOO-CV + CP)."""

import numpy as np
from joblib import Parallel, delayed

from ssbc.bounds import prediction_bounds
from ssbc.core_pkg import SSBCResult

from .loo_uncertainty import compute_loo_corrected_prediction_bounds, compute_robust_prediction_bounds


def _safe_parallel_map(n_jobs: int, func, iterable):
    """Execute jobs in parallel if possible, otherwise fall back to serial.

    This avoids sandbox/system-limit failures (e.g., PermissionError from loky)
    by retrying in-process serial execution when multiprocessing is unavailable.
    """
    try:
        return Parallel(n_jobs=n_jobs)(delayed(func)(*args) for args in iterable)
    except Exception:
        # Fallback to serial execution
        return [func(*args) for args in iterable]


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
    results = _safe_parallel_map(
        n_jobs,
        _evaluate_loo_single_sample_marginal,
        ((idx, labels, probs, k_0, k_1) for idx in range(n)),
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

    # Class-specific singleton error rates (normalized against full dataset)
    n_errors_class0 = int(np.sum((results_array[:, 0] == 1) & (labels == 0) & (results_array[:, 3] == 0)))
    n_errors_class1 = int(np.sum((results_array[:, 0] == 1) & (labels == 1) & (results_array[:, 3] == 0)))
    singleton_error_rate_class0 = n_errors_class0 / n if n > 0 else 0.0
    singleton_error_rate_class1 = n_errors_class1 / n if n > 0 else 0.0

    # Conditional error rates: P(error | singleton & class)
    singleton_class0_mask = (results_array[:, 0] == 1) & (labels == 0)
    singleton_class1_mask = (results_array[:, 0] == 1) & (labels == 1)
    n_singletons_class0 = int(np.sum(singleton_class0_mask))
    n_singletons_class1 = int(np.sum(singleton_class1_mask))
    singleton_error_rate_cond_class0 = n_errors_class0 / n_singletons_class0 if n_singletons_class0 > 0 else 0.0
    singleton_error_rate_cond_class1 = n_errors_class1 / n_singletons_class1 if n_singletons_class1 > 0 else 0.0

    # Apply prediction bounds accounting for both calibration and test set sampling uncertainty
    # These bound operational rates on future test sets of size test_size
    # SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test)) accounts for both sources of uncertainty

    # Now we have 8 metrics: singleton, doublet, abstention, error (conditional),
    # error_class0 (normalized), error_class1 (normalized), error_cond_class0, error_cond_class1
    n_metrics = 8
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

    # Class-specific singleton error rates (normalized against full dataset)
    error_class0_lower, error_class0_upper = prediction_bounds(
        n_errors_class0, n, test_size, adjusted_ci_level, prediction_method
    )
    error_class1_lower, error_class1_upper = prediction_bounds(
        n_errors_class1, n, test_size, adjusted_ci_level, prediction_method
    )

    # Conditional error rates: P(error | singleton & class)
    expected_n_singletons_class0_test = int(test_size * (n_singletons_class0 / n)) if n > 0 else 1
    expected_n_singletons_class0_test = max(expected_n_singletons_class0_test, 1) if n_singletons_class0 > 0 else 1
    expected_n_singletons_class1_test = int(test_size * (n_singletons_class1 / n)) if n > 0 else 1
    expected_n_singletons_class1_test = max(expected_n_singletons_class1_test, 1) if n_singletons_class1 > 0 else 1

    if n_singletons_class0 > 0:
        error_cond_class0_lower, error_cond_class0_upper = prediction_bounds(
            n_errors_class0,
            n_singletons_class0,
            expected_n_singletons_class0_test,
            adjusted_ci_level,
            prediction_method,
        )
    else:
        error_cond_class0_lower = 0.0
        error_cond_class0_upper = 1.0

    if n_singletons_class1 > 0:
        error_cond_class1_lower, error_cond_class1_upper = prediction_bounds(
            n_errors_class1,
            n_singletons_class1,
            expected_n_singletons_class1_test,
            adjusted_ci_level,
            prediction_method,
        )
    else:
        error_cond_class1_lower = 0.0
        error_cond_class1_upper = 1.0

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        "singleton_error_rate_class0_bounds": [error_class0_lower, error_class0_upper],
        "singleton_error_rate_class1_bounds": [error_class1_lower, error_class1_upper],
        "singleton_error_rate_cond_class0_bounds": [error_cond_class0_lower, error_cond_class0_upper],
        "singleton_error_rate_cond_class1_bounds": [error_cond_class1_lower, error_cond_class1_upper],
        "expected_singleton_rate": singleton_rate,
        "expected_doublet_rate": doublet_rate,
        "expected_abstention_rate": abstention_rate,
        "expected_singleton_error_rate": singleton_error_rate,
        "expected_singleton_error_rate_class0": singleton_error_rate_class0,
        "expected_singleton_error_rate_class1": singleton_error_rate_class1,
        "expected_singleton_error_rate_cond_class0": singleton_error_rate_cond_class0,
        "expected_singleton_error_rate_cond_class1": singleton_error_rate_cond_class1,
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
    results = _safe_parallel_map(
        n_jobs,
        _evaluate_loo_single_sample_marginal,
        ((idx, labels, probs, k_0, k_1) for idx in range(n)),
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

    # Class-specific singleton error rates (normalized against full dataset)
    # Error rate for singletons with true_label=0, normalized by total samples
    error_class0_loo_preds = ((results_array[:, 0] == 1) & (labels == 0) & (results_array[:, 3] == 0)).astype(int)
    # Error rate for singletons with true_label=1, normalized by total samples
    error_class1_loo_preds = ((results_array[:, 0] == 1) & (labels == 1) & (results_array[:, 3] == 0)).astype(int)

    # Point estimates for class-specific error rates (normalized against full dataset)
    singleton_error_rate_class0 = float(np.mean(error_class0_loo_preds)) if n > 0 else 0.0
    singleton_error_rate_class1 = float(np.mean(error_class1_loo_preds)) if n > 0 else 0.0

    # Conditional error rates: P(error | singleton & class)
    # Binary predictions: 1 if singleton with class_label AND error, 0 otherwise
    singleton_class0_mask = (results_array[:, 0] == 1) & (labels == 0)
    singleton_class1_mask = (results_array[:, 0] == 1) & (labels == 1)
    n_singletons_class0 = int(np.sum(singleton_class0_mask))
    n_singletons_class1 = int(np.sum(singleton_class1_mask))
    n_errors_class0_singleton = int(np.sum(error_class0_loo_preds))
    n_errors_class1_singleton = int(np.sum(error_class1_loo_preds))

    # Conditional rates
    singleton_error_rate_cond_class0 = (
        n_errors_class0_singleton / n_singletons_class0 if n_singletons_class0 > 0 else 0.0
    )
    singleton_error_rate_cond_class1 = (
        n_errors_class1_singleton / n_singletons_class1 if n_singletons_class1 > 0 else 0.0
    )

    # LOO predictions for conditional error rates (need to filter by singleton & class)
    error_cond_class0_loo_preds = np.zeros(n, dtype=int)
    if n_singletons_class0 > 0:
        error_cond_class0_loo_preds[singleton_class0_mask] = (results_array[singleton_class0_mask, 3] == 0).astype(int)

    error_cond_class1_loo_preds = np.zeros(n, dtype=int)
    if n_singletons_class1 > 0:
        error_cond_class1_loo_preds[singleton_class1_mask] = (results_array[singleton_class1_mask, 3] == 0).astype(int)

    # Apply union bound adjustment
    # Now we have 8 metrics: singleton, doublet, abstention, error (conditional),
    # error_class0 (normalized), error_class1 (normalized), error_cond_class0, error_cond_class1
    n_metrics = 8
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

    # Class-specific singleton error rates (normalized against full dataset)
    error_class0_lower, error_class0_upper, error_class0_report = compute_robust_prediction_bounds(
        error_class0_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    error_class1_lower, error_class1_upper, error_class1_report = compute_robust_prediction_bounds(
        error_class1_loo_preds,
        test_size,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    # Conditional error rates: P(error | singleton & class)
    # Need to compute bounds using only singleton samples from each class
    # For prediction bounds, estimate test set size for class-X singletons
    expected_n_singletons_class0_test = int(test_size * (n_singletons_class0 / n)) if n > 0 else 1
    expected_n_singletons_class0_test = max(expected_n_singletons_class0_test, 1) if n_singletons_class0 > 0 else 1
    expected_n_singletons_class1_test = int(test_size * (n_singletons_class1 / n)) if n > 0 else 1
    expected_n_singletons_class1_test = max(expected_n_singletons_class1_test, 1) if n_singletons_class1 > 0 else 1

    if n_singletons_class0 > 0:
        error_cond_class0_lower, error_cond_class0_upper, error_cond_class0_report = compute_robust_prediction_bounds(
            error_cond_class0_loo_preds[singleton_class0_mask],
            expected_n_singletons_class0_test,
            1 - adjusted_ci_level,
            method=prediction_method,
            inflation_factor=loo_inflation_factor,
            verbose=verbose,
        )
    else:
        error_cond_class0_lower = 0.0
        error_cond_class0_upper = 1.0
        error_cond_class0_report = {"selected_method": "no_singletons_class0", "diagnostics": {}}

    if n_singletons_class1 > 0:
        error_cond_class1_lower, error_cond_class1_upper, error_cond_class1_report = compute_robust_prediction_bounds(
            error_cond_class1_loo_preds[singleton_class1_mask],
            expected_n_singletons_class1_test,
            1 - adjusted_ci_level,
            method=prediction_method,
            inflation_factor=loo_inflation_factor,
            verbose=verbose,
        )
    else:
        error_cond_class1_lower = 0.0
        error_cond_class1_upper = 1.0
        error_cond_class1_report = {"selected_method": "no_singletons_class1", "diagnostics": {}}

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        "singleton_error_rate_class0_bounds": [error_class0_lower, error_class0_upper],
        "singleton_error_rate_class1_bounds": [error_class1_lower, error_class1_upper],
        "singleton_error_rate_cond_class0_bounds": [error_cond_class0_lower, error_cond_class0_upper],
        "singleton_error_rate_cond_class1_bounds": [error_cond_class1_lower, error_cond_class1_upper],
        "expected_singleton_rate": singleton_rate,
        "expected_doublet_rate": doublet_rate,
        "expected_abstention_rate": abstention_rate,
        "expected_singleton_error_rate": singleton_error_rate,
        "expected_singleton_error_rate_class0": singleton_error_rate_class0,
        "expected_singleton_error_rate_class1": singleton_error_rate_class1,
        "expected_singleton_error_rate_cond_class0": singleton_error_rate_cond_class0,
        "expected_singleton_error_rate_cond_class1": singleton_error_rate_cond_class1,
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
            "singleton_error_class0": error_class0_report,
            "singleton_error_class1": error_class1_report,
            "singleton_error_cond_class0": error_cond_class0_report,
            "singleton_error_cond_class1": error_cond_class1_report,
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
    results = _safe_parallel_map(
        n_jobs,
        _evaluate_loo_single_sample_perclass,
        ((idx, labels, probs, k_0, k_1, class_label) for idx in range(n)),
    )

    # Aggregate results (only from class_label samples)
    results_array = np.array(results)
    n_singletons = int(np.sum(results_array[:, 0] * (labels == class_label)[:, None]))
    n_doublets = int(np.sum(results_array[:, 1] * (labels == class_label)[:, None]))
    n_abstentions = int(np.sum(results_array[:, 2] * (labels == class_label)[:, None]))
    n_singletons_correct = int(np.sum(results_array[:, 3] * (labels == class_label)[:, None]))

    # Number of class_label samples in calibration
    n_class_cal = np.sum(labels == class_label)

    # Estimate expected class distribution in test set
    # Use calibration class distribution as estimate for test set
    n_total = len(labels)
    class_rate_cal = n_class_cal / n_total
    expected_n_class_test = int(test_size * class_rate_cal)

    # Ensure minimum test size for numerical stability
    expected_n_class_test = max(expected_n_class_test, 1)

    # Point estimates (calibration proportions)
    n_errors = n_singletons - n_singletons_correct

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

    # Build LOO prediction arrays for unbiased point estimates
    singleton_loo_preds = results_array[:, 0].astype(int)
    doublet_loo_preds = results_array[:, 1].astype(int)
    abstention_loo_preds = results_array[:, 2].astype(int)
    error_loo_preds = np.zeros(n, dtype=int)
    if n_singletons > 0:
        error_loo_preds = (results_array[:, 0] == 1) & (results_array[:, 3] == 0)

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [error_lower, error_upper],
        # Unbiased LOO estimates (means of LOO predictions)
        "expected_singleton_rate": float(np.mean(singleton_loo_preds)) if n_class_cal > 0 else 0.0,
        "expected_doublet_rate": float(np.mean(doublet_loo_preds)) if n_class_cal > 0 else 0.0,
        "expected_abstention_rate": float(np.mean(abstention_loo_preds)) if n_class_cal > 0 else 0.0,
        "expected_singleton_error_rate": float(np.mean(error_loo_preds)) if n_singletons > 0 else 0.0,
        "n_grid_points": 1,
        "pac_level": adjusted_ci_level,
        "ci_level": ci_level,
        # Report the intended future test size parameter
        "test_size": expected_n_class_test,
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
    results = _safe_parallel_map(
        n_jobs,
        _evaluate_loo_single_sample_perclass,
        ((idx, labels, probs, k_0, k_1, class_label) for idx in range(n)),
    )

    # Aggregate results (only from class_label samples)
    results_array = np.array(results)
    n_singletons = int(np.sum(results_array[:, 0]))

    # Number of class_label samples in calibration
    n_class_cal = np.sum(labels == class_label)

    # Estimate expected class distribution in test set
    n_total = len(labels)
    class_rate_cal = n_class_cal / n_total
    expected_n_class_test = int(test_size * class_rate_cal)
    expected_n_class_test = max(expected_n_class_test, 1)

    # Point estimates (calibration proportions)

    # Convert to binary LOO predictions for each rate type
    # Restrict LOO binary arrays to class_label rows only (for unbiased per-class means)
    class_mask = labels == class_label
    singleton_loo_preds = results_array[class_mask, 0].astype(int)
    doublet_loo_preds = results_array[class_mask, 1].astype(int)
    abstention_loo_preds = results_array[class_mask, 2].astype(int)
    error_loo_preds = np.zeros(np.sum(class_mask), dtype=int)
    if n_singletons > 0:
        error_loo_preds = (results_array[class_mask, 0] == 1) & (results_array[class_mask, 3] == 0)

    # Apply union bound adjustment
    n_metrics = 4
    if use_union_bound:
        adjusted_ci_level = 1 - (1 - ci_level) / n_metrics
    else:
        adjusted_ci_level = ci_level

    # Compute LOO-corrected bounds for each rate type
    # Use compute_robust_prediction_bounds for consistency with marginal bounds
    # This supports all methods including "all", "auto", "analytical", "exact", "hoeffding", etc.
    singleton_lower, singleton_upper, singleton_report = compute_robust_prediction_bounds(
        singleton_loo_preds,
        expected_n_class_test,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    doublet_lower, doublet_upper, doublet_report = compute_robust_prediction_bounds(
        doublet_loo_preds,
        expected_n_class_test,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    abstention_lower, abstention_upper, abstention_report = compute_robust_prediction_bounds(
        abstention_loo_preds,
        expected_n_class_test,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    error_lower, error_upper, error_report = compute_robust_prediction_bounds(
        error_loo_preds,
        expected_n_class_test,
        1 - adjusted_ci_level,
        method=prediction_method,
        inflation_factor=loo_inflation_factor,
        verbose=verbose,
    )

    # Compute both approaches with proper uncertainty quantification

    # Approach A: Fraction of whole dataset (denominator is fixed)
    # These bounds are already computed above (using expected_n_class_test)
    approach_a_singleton_bounds = [singleton_lower, singleton_upper]
    approach_a_doublet_bounds = [doublet_lower, doublet_upper]
    approach_a_abstention_bounds = [abstention_lower, abstention_upper]
    approach_a_error_bounds = [error_lower, error_upper]

    # Approach B: Fraction of class samples (denominator is uncertain)
    # Need to account for class rate uncertainty in denominator
    from ssbc.bounds import prediction_bounds

    # Class rate bounds (uncertainty in denominator)
    n_total_cal = len(labels)
    class_rate_lower, class_rate_upper = prediction_bounds(
        n_class_cal, n_total_cal, test_size, adjusted_ci_level, "simple"
    )

    # For Approach B, we need to account for both numerator and denominator uncertainty
    # This is a complex ratio estimation problem - we use conservative bounds
    # based on worst-case class rate bounds

    # Conservative approach: use worst-case class rate bounds
    min_class_rate = class_rate_lower
    max_class_rate = class_rate_upper

    # Approach B bounds (fraction of class samples)
    # Use the class-specific bounds but adjust for class rate uncertainty
    # This provides conservative bounds for the ratio of operational rates
    approach_b_singleton_bounds = [
        singleton_lower * min_class_rate / class_rate_cal,
        singleton_upper * max_class_rate / class_rate_cal,
    ]
    approach_b_doublet_bounds = [
        doublet_lower * min_class_rate / class_rate_cal,
        doublet_upper * max_class_rate / class_rate_cal,
    ]
    approach_b_abstention_bounds = [
        abstention_lower * min_class_rate / class_rate_cal,
        abstention_upper * max_class_rate / class_rate_cal,
    ]
    approach_b_error_bounds = [
        error_lower * min_class_rate / class_rate_cal,
        error_upper * max_class_rate / class_rate_cal,
    ]

    # Build result dict with both approaches
    result = {
        # Approach A: Fraction of whole dataset
        "singleton_rate_bounds_whole_dataset": approach_a_singleton_bounds,
        "doublet_rate_bounds_whole_dataset": approach_a_doublet_bounds,
        "abstention_rate_bounds_whole_dataset": approach_a_abstention_bounds,
        "singleton_error_rate_bounds_whole_dataset": approach_a_error_bounds,
        # Approach B: Fraction of class samples
        "singleton_rate_bounds_class_samples": approach_b_singleton_bounds,
        "doublet_rate_bounds_class_samples": approach_b_doublet_bounds,
        "abstention_rate_bounds_class_samples": approach_b_abstention_bounds,
        "singleton_error_rate_bounds_class_samples": approach_b_error_bounds,
        # Backward compatibility (default to Approach A)
        "singleton_rate_bounds": approach_a_singleton_bounds,
        "doublet_rate_bounds": approach_a_doublet_bounds,
        "abstention_rate_bounds": approach_a_abstention_bounds,
        "singleton_error_rate_bounds": approach_a_error_bounds,
        # Unbiased LOO estimates (means of LOO predictions)
        "expected_singleton_rate": float(np.mean(singleton_loo_preds)) if n_class_cal > 0 else 0.0,
        "expected_doublet_rate": float(np.mean(doublet_loo_preds)) if n_class_cal > 0 else 0.0,
        "expected_abstention_rate": float(np.mean(abstention_loo_preds)) if n_class_cal > 0 else 0.0,
        "expected_singleton_error_rate": float(np.mean(error_loo_preds)) if n_singletons > 0 else 0.0,
        # Class rate uncertainty
        "class_rate_bounds": [class_rate_lower, class_rate_upper],
        "class_rate_calibration": class_rate_cal,
        "n_grid_points": 1,
        "pac_level": adjusted_ci_level,
        "ci_level": ci_level,
        "test_size": expected_n_class_test,
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
        # Always include LOO diagnostics (for method reporting)
        "loo_diagnostics": {
            "singleton": singleton_report,
            "doublet": doublet_report,
            "abstention": abstention_report,
            "singleton_error": error_report,
        },
    }

    return result
