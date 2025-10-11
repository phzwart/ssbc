"""Service Level Agreement (SLA) operational rate bounds for conformal prediction.

Implements rigorous operational rate bounds using:
- Cross-fit Clopper-Pearson bounds for operational rates
- Transfer bounds from cross-fit to single refit-on-all rule

This provides contract-ready guarantees on operational rates:
- Singleton, doublet, abstention, error rates with probability ≥ 1 - δ

Works in tandem with mondrian_conformal_calibrate() which handles PAC coverage guarantees.
The workflow is:
1. Use mondrian_conformal_calibrate() to get SSBC-corrected thresholds with coverage guarantees
2. Use this module to add rigorous CP bounds on operational rates via cross-validation

Reference: Based on Appendix B of the SSBC theoretical framework.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from .statistics import clopper_pearson_lower, clopper_pearson_upper
from .utils import compute_operational_rate


@dataclass
class OperationalRateBounds:
    """Bounds for a single operational rate with PAC guarantees.

    Attributes
    ----------
    rate_name : str
        Name of the operational rate (e.g., "singleton", "doublet")
    lower_bound : float
        Lower bound for deployment on single refit-on-all rule
    upper_bound : float
        Upper bound for deployment on single refit-on-all rule
    cross_fit_lower : float
        Lower bound from cross-fit analysis (before transfer)
    cross_fit_upper : float
        Upper bound from cross-fit analysis (before transfer)
    cushion : float
        Transfer cushion ε_n for single rule deployment
    ci_width : float
        Width of Clopper-Pearson confidence intervals used (e.g., 0.95 for 95% CIs)
    fold_results : list[dict]
        Detailed results from each cross-validation fold
    """

    rate_name: str
    lower_bound: float
    upper_bound: float
    cross_fit_lower: float
    cross_fit_upper: float
    cushion: float
    ci_width: float
    fold_results: list[dict]


@dataclass
class OperationalRateBoundsResult:
    """Operational rate bounds for conformal prediction deployment.

    Provides rigorous bounds on operational rates (singleton, doublet, abstention, error)
    computed via cross-validation and transferred to single refit-on-all rule.

    Use in combination with mondrian_conformal_calibrate() output for complete SLA:
    - Mondrian provides: PAC coverage guarantees (via SSBC)
    - This provides: Operational rate bounds (via cross-fit Clopper-Pearson)

    Attributes
    ----------
    rate_bounds : dict[str, OperationalRateBounds]
        Bounds for each operational rate type
    rate_confidence : float
        Confidence level for rate bounds (1 - δ)
    thresholds : dict[int, float] | float
        Conformal prediction thresholds used (per-class for Mondrian, scalar otherwise)
    n_calibration : int
        Total calibration set size
    n_folds : int
        Number of cross-validation folds used
    """

    # Operational rate bounds
    rate_bounds: dict[str, OperationalRateBounds]
    rate_confidence: float  # 1 - delta

    # Deployment info
    thresholds: dict[int, float] | float
    n_calibration: int
    n_folds: int


# ============================================================================
# Cross-Fit Clopper-Pearson Bounds for Operational Rates
# ============================================================================


def cross_fit_cp_bounds(
    class_data: dict[str, Any],
    true_class: int,
    alpha_target: float,
    delta_class: float,
    rate_types: list[str],
    n_folds: int,
    delta: float,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> dict[str, dict]:
    """Compute cross-fit Clopper-Pearson bounds for operational rates (per-class).

    Uses K-fold cross-validation on the same calibration data used for Mondrian
    conformal prediction. For each fold:
    1. Compute SSBC-corrected threshold on training folds
    2. Apply to test fold to get operational rates
    3. Compute Clopper-Pearson bounds with PAC margin

    Parameters
    ----------
    class_data : dict
        Per-class data from split_by_class() containing:
        - 'probs': probability matrix (n, 2)
        - 'labels': true labels (all same class)
        - 'n': number of samples
    true_class : int
        The class label (0 or 1) for this data
    alpha_target : float
        Target miscoverage rate for this class
    delta_class : float
        PAC risk for coverage (used for SSBC in each fold)
    rate_types : list[str]
        List of operational rates to compute. Options:
        "singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"
    n_folds : int
        Number of cross-validation folds (typically 5-10)
    delta : float
        PAC risk budget for operational rate bounds. Each rate independently gets
        confidence 1-δ via union bound across folds only (NOT split across rates).
    confidence_level : float, default=0.95
        Confidence level for Clopper-Pearson intervals (e.g., 0.95 for 95% CIs)
    random_seed : int, optional
        Random seed for reproducible fold splits

    Returns
    -------
    results : dict[str, dict]
        Dictionary mapping rate_type to:
        - "fold_results": list of per-fold statistics
        - "weights": fold weights (proportional to fold size)
        - "cf_lower": cross-fit lower bound
        - "cf_upper": cross-fit upper bound

    Notes
    -----
    Union bound applies ONLY across folds (δ split K ways), NOT across rate types.
    Each rate type independently achieves confidence 1-δ. This means if you request
    3 rate types with δ=0.05, each rate has 95% confidence (not 95%/3).
    The CI width is controlled separately by confidence_level parameter.
    """
    from .core import ssbc_correct

    probs = class_data["probs"]
    n = class_data["n"]
    fold_size = n // n_folds

    # Note: Delta is NOT split across rate types - each rate independently
    # gets confidence 1-δ via union bound across folds only

    results = {rt: {"fold_results": [], "weights": []} for rt in rate_types}

    # Create fold indices with optional random seed
    indices = np.arange(n)
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    rng.shuffle(indices)

    for f in range(n_folds):
        # Define fold
        start_idx = f * fold_size
        end_idx = (f + 1) * fold_size if f < n_folds - 1 else n
        test_indices = indices[start_idx:end_idx]
        train_indices = np.setdiff1d(indices, test_indices)

        m_f = len(test_indices)
        n_train = len(train_indices)

        # Compute nonconformity scores on training fold: s = 1 - P(true_class|x)
        train_probs = probs[train_indices]
        train_scores = 1.0 - train_probs[:, true_class]

        # Apply SSBC to get corrected alpha for this fold
        ssbc_result = ssbc_correct(alpha_target=alpha_target, n=n_train, delta=delta_class, mode="beta")
        alpha_corrected = ssbc_result.alpha_corrected

        # Compute threshold: k-th order statistic where k = ceil((n+1)*(1-alpha_corrected))
        k = int(np.ceil((n_train + 1) * (1 - alpha_corrected)))
        k = min(k, n_train)
        sorted_train_scores = np.sort(train_scores)
        q_minus_f = sorted_train_scores[k - 1] if k > 0 else sorted_train_scores[0]

        # Band for cushion: [S_(k), S_(k+1)] per paper Eq. B.4
        # This is [threshold, next_order_statistic]
        # If k >= n_train, there's no next order statistic - use a tight band around threshold
        q_band_lower = sorted_train_scores[k - 1] if k >= 1 else -np.inf  # S_(k) = threshold
        if k < n_train:
            q_band_upper = sorted_train_scores[k]  # S_(k+1)
        else:
            # Edge case: threshold is at/beyond last order statistic
            # Use threshold itself as both bounds (no flip risk)
            q_band_upper = sorted_train_scores[k - 1] if k >= 1 else np.inf

        # Generate prediction sets on test fold using this threshold
        test_probs = probs[test_indices]
        test_prediction_sets = []

        for i in range(m_f):
            # For each test example, check which labels pass the threshold
            score_0 = 1.0 - test_probs[i, 0]
            score_1 = 1.0 - test_probs[i, 1]

            pred_set = []
            if score_0 <= q_minus_f:
                pred_set.append(0)
            if score_1 <= q_minus_f:
                pred_set.append(1)

            test_prediction_sets.append(pred_set)

        # All test labels are the same (true_class) since this is per-class data
        test_labels = np.full(m_f, true_class)

        # Compute operational rates for this fold
        for rate_type in rate_types:
            Z = compute_operational_rate(
                test_prediction_sets,
                test_labels,
                cast(
                    Literal["singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"],
                    rate_type,
                ),
            )
            K_fr = int(np.sum(Z))

            # Compute Clopper-Pearson bounds at specified confidence level
            L_fr = clopper_pearson_lower(K_fr, m_f, confidence_level)
            U_fr = clopper_pearson_upper(K_fr, m_f, confidence_level)

            results[rate_type]["fold_results"].append(
                {
                    "fold": f,
                    "m_f": m_f,
                    "K_fr": K_fr,
                    "L_fr": L_fr,
                    "U_fr": U_fr,
                    "q_minus_f": q_minus_f,
                    "q_band": (q_band_lower, q_band_upper),
                    "test_indices": test_indices,
                }
            )
            results[rate_type]["weights"].append(m_f / n)

    # Aggregate cross-fit bounds (weighted average)
    for rate_type in rate_types:
        fold_res = results[rate_type]["fold_results"]
        weights = results[rate_type]["weights"]

        cf_lower = sum(w * fr["L_fr"] for w, fr in zip(weights, fold_res, strict=False))
        cf_upper = sum(w * fr["U_fr"] for w, fr in zip(weights, fold_res, strict=False))

        results[rate_type]["cf_lower"] = cf_lower
        results[rate_type]["cf_upper"] = cf_upper

    return results


# ============================================================================
# Transfer to Single Refit-On-All Rule
# ============================================================================


def compute_transfer_cushion(
    class_data: dict[str, Any],
    true_class: int,
    cross_fit_results: dict,
    rate_type: str,
) -> float:
    """Compute empirical cushion ε_n for transferring cross-fit bounds to single rule.

    The cushion accounts for the difference between cross-fit thresholds
    (computed on K-1 folds) and the final threshold (computed on all n samples).
    Points near the threshold boundary may flip between different prediction
    set sizes.

    Parameters
    ----------
    class_data : dict
        Per-class data from split_by_class() containing probs and labels
    true_class : int
        The class label (0 or 1) for this data
    cross_fit_results : dict
        Results from cross_fit_cp_bounds for this rate_type
    rate_type : str
        Operational rate type being analyzed

    Returns
    -------
    epsilon_n : float
        Aggregate cushion for rate transfer (≥ 0)

    Notes
    -----
    The cushion is computed as the weighted average fraction of test points
    in each fold that have at least one label with score in the critical band
    [q^{(-f)}_{k-1}, q^{(-f)}_{k+1}] around the threshold.
    """
    probs = class_data["probs"]
    fold_results = cross_fit_results["fold_results"]
    weights = cross_fit_results["weights"]

    epsilon_total = 0.0

    for fold_data, w_f in zip(fold_results, weights, strict=False):
        q_band = fold_data["q_band"]
        q_lower, q_upper = q_band
        test_indices = fold_data["test_indices"]
        m_f = len(test_indices)

        # Count points in test fold where any label has score in the critical band
        flip_count = 0
        for i in test_indices:
            test_probs = probs[i]
            # Check if any label has score in the critical band
            # Score for each label: 1 - P(y|x)
            for y in [0, 1]:
                score = 1.0 - test_probs[y]
                if q_lower <= score <= q_upper:
                    flip_count += 1
                    break  # Count this point once

        epsilon_f = flip_count / m_f if m_f > 0 else 0.0
        epsilon_total += w_f * epsilon_f

    return epsilon_total


def transfer_bounds_to_single_rule(cross_fit_results: dict, cushions: dict[str, float]) -> dict[str, dict]:
    """Transfer cross-fit CP bounds to single refit-on-all rule.

    Applies the empirical cushion to widen the cross-fit bounds, accounting
    for the difference between K-fold thresholds and the final single threshold.

    Parameters
    ----------
    cross_fit_results : dict
        Results from cross_fit_cp_bounds
    cushions : dict[str, float]
        Cushions for each rate type from compute_transfer_cushion

    Returns
    -------
    transferred_bounds : dict[str, dict]
        For each rate type, contains:
        - "single_lower": lower bound for single rule
        - "single_upper": upper bound for single rule
        - "cf_lower": original cross-fit lower bound
        - "cf_upper": original cross-fit upper bound
        - "cushion": cushion applied

    Notes
    -----
    The transferred bounds are:
    - L_single = L_cf - ε_n
    - U_single = U_cf + ε_n

    These bounds are valid for the single refit-on-all conformal predictor
    deployed in production.
    """
    transferred = {}

    for rate_type in cross_fit_results.keys():
        cf_lower = cross_fit_results[rate_type]["cf_lower"]
        cf_upper = cross_fit_results[rate_type]["cf_upper"]
        epsilon = cushions[rate_type]

        # Apply cushion (widen bounds for safety)
        single_lower = max(0.0, cf_lower - epsilon)  # Clamp to [0, 1]
        single_upper = min(1.0, cf_upper + epsilon)

        transferred[rate_type] = {
            "single_lower": single_lower,
            "single_upper": single_upper,
            "cf_lower": cf_lower,
            "cf_upper": cf_upper,
            "cushion": epsilon,
        }

    return transferred


# ============================================================================
# Mondrian Operational Bounds
# ============================================================================


def compute_marginal_operational_bounds(
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: float | dict[int, float],
    delta_coverage: float | dict[int, float],
    delta: float,
    rate_types: list[str] | None = None,
    confidence_level: float = 0.95,
    n_folds: int = 5,
    random_seed: int | None = None,
) -> OperationalRateBoundsResult:
    """Compute marginal operational rate bounds via cross-validated Mondrian.

    Uses K-fold cross-validation on the MARGINAL (mixed-class) data:
    1. Split marginal data into K folds
    2. For each fold:
       - Train Mondrian on other folds (split by class, compute thresholds)
       - Test on this fold (marginal) → count operational rates
       - Compute CP bounds at specified confidence level
    3. Average across folds and apply transfer cushion

    This gives unbiased marginal rate estimates because test folds never saw training data.

    Parameters
    ----------
    labels : np.ndarray
        True labels (shape: n,)
    probs : np.ndarray
        Probability matrix (shape: n, 2)
    alpha_target : float or dict[int, float]
        Target miscoverage rate per class
    delta_coverage : float or dict[int, float]
        PAC risk for coverage (used for SSBC in each fold)
    delta : float
        PAC risk for operational rate bounds. Each rate independently gets
        confidence 1-δ via union bound across folds only (NOT split across rates).
    rate_types : list[str], optional
        Operational rates to compute. Each gets confidence 1-δ independently.
    confidence_level : float, default=0.95
        Confidence level for Clopper-Pearson intervals (e.g., 0.95 for 95% CIs)
    n_folds : int, default=5
        Number of cross-validation folds
    random_seed : int, optional
        Random seed for reproducible results

    Returns
    -------
    result : OperationalRateBoundsResult
        Marginal operational rate bounds with CP guarantees

    Examples
    --------
    >>> # Get rigorous marginal rate bounds
    >>> marginal_bounds = compute_marginal_operational_bounds(
    ...     labels, probs, alpha_target=0.1, delta_coverage=0.05, delta=0.05,
    ...     confidence_level=0.95
    ... )
    >>> print(f"Marginal singleton rate: "
    ...       f"[{marginal_bounds.rate_bounds['singleton'].lower_bound:.2f}, "
    ...       f"{marginal_bounds.rate_bounds['singleton'].upper_bound:.2f}]")

    Notes
    -----
    Union bound applies ONLY across folds (δ split K ways), NOT across rate types.
    Each rate type independently achieves confidence 1-δ with specified CI width.
    """
    from .conformal import split_by_class
    from .core import ssbc_correct

    n = len(labels)
    fold_size = n // n_folds

    # Handle scalar or dict inputs for alpha and delta
    if isinstance(alpha_target, int | float):
        alpha_dict = {0: float(alpha_target), 1: float(alpha_target)}
    else:
        alpha_dict = {k: float(v) for k, v in alpha_target.items()}

    if isinstance(delta_coverage, int | float):
        delta_dict = {0: float(delta_coverage), 1: float(delta_coverage)}
    else:
        delta_dict = {k: float(v) for k, v in delta_coverage.items()}

    # Default rate types (include conditional singleton rates)
    if rate_types is None:
        rate_types = ["singleton", "doublet", "abstention", "correct_in_singleton", "error_in_singleton"]

    # Note: Delta is NOT split across rate types - each rate independently
    # gets confidence 1-δ via union bound across folds only

    results = {rt: {"fold_results": [], "weights": []} for rt in rate_types}

    # Create fold indices with optional random seed
    indices = np.arange(n)
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    rng.shuffle(indices)

    # Store Mondrian thresholds for each fold (for cushion computation)
    fold_thresholds = []

    for f in range(n_folds):
        # Define fold
        start_idx = f * fold_size
        end_idx = (f + 1) * fold_size if f < n_folds - 1 else n
        test_indices = indices[start_idx:end_idx]
        train_indices = np.setdiff1d(indices, test_indices)

        m_f = len(test_indices)

        # Train: split training data by class and compute Mondrian thresholds
        train_labels = labels[train_indices]
        train_probs = probs[train_indices]

        # Split training data by class
        train_class_data = split_by_class(train_labels, train_probs)

        # Compute Mondrian thresholds and bands for each class
        thresholds = {}
        bands = {}
        for class_label in [0, 1]:
            class_data = train_class_data[class_label]
            n_class = class_data["n"]

            if n_class == 0:
                thresholds[class_label] = np.inf
                bands[class_label] = (-np.inf, np.inf)
                continue

            # Compute scores for this class
            class_probs = class_data["probs"]
            class_scores = 1.0 - class_probs[:, class_label]

            # Apply SSBC
            ssbc_result = ssbc_correct(
                alpha_target=alpha_dict[class_label], n=n_class, delta=delta_dict[class_label], mode="beta"
            )
            alpha_corrected = ssbc_result.alpha_corrected

            # Compute threshold index
            k = int(np.ceil((n_class + 1) * (1 - alpha_corrected)))
            k = min(k, n_class)
            sorted_scores = np.sort(class_scores)
            thresholds[class_label] = sorted_scores[k - 1] if k > 0 else sorted_scores[0]

            # Band for cushion: [S_(k), S_(k+1)] per paper Eq. B.4
            band_lower = sorted_scores[k - 1] if k >= 1 else -np.inf  # S_(k) = threshold
            if k < n_class:
                band_upper = sorted_scores[k]  # S_(k+1)
            else:
                # Edge case: no next order statistic, use threshold (no flip risk)
                band_upper = sorted_scores[k - 1] if k >= 1 else np.inf
            bands[class_label] = (band_lower, band_upper)

        fold_thresholds.append(thresholds)

        # Test: apply Mondrian thresholds to test fold (marginal data)
        test_probs = probs[test_indices]
        test_labels = labels[test_indices]

        test_prediction_sets = []
        for i in range(m_f):
            # Compute scores for each class
            score_0 = 1.0 - test_probs[i, 0]
            score_1 = 1.0 - test_probs[i, 1]

            # Apply Mondrian thresholds
            pred_set = []
            if score_0 <= thresholds[0]:
                pred_set.append(0)
            if score_1 <= thresholds[1]:
                pred_set.append(1)

            test_prediction_sets.append(pred_set)

        # Compute marginal operational rates for this fold
        for rate_type in rate_types:
            Z = compute_operational_rate(
                test_prediction_sets,
                test_labels,
                cast(
                    Literal["singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"],
                    rate_type,
                ),
            )
            K_fr = int(np.sum(Z))

            # Compute Clopper-Pearson bounds at specified confidence level
            L_fr = clopper_pearson_lower(K_fr, m_f, confidence_level)
            U_fr = clopper_pearson_upper(K_fr, m_f, confidence_level)

            results[rate_type]["fold_results"].append(
                {
                    "fold": f,
                    "m_f": m_f,
                    "K_fr": K_fr,
                    "L_fr": L_fr,
                    "U_fr": U_fr,
                    "thresholds": thresholds.copy(),
                    "bands": bands.copy(),
                    "test_indices": test_indices,
                }
            )
            results[rate_type]["weights"].append(m_f / n)

    # Aggregate cross-fit bounds (weighted average)
    for rate_type in rate_types:
        fold_res = results[rate_type]["fold_results"]
        weights = results[rate_type]["weights"]

        cf_lower = sum(w * fr["L_fr"] for w, fr in zip(weights, fold_res, strict=False))
        cf_upper = sum(w * fr["U_fr"] for w, fr in zip(weights, fold_res, strict=False))

        results[rate_type]["cf_lower"] = cf_lower
        results[rate_type]["cf_upper"] = cf_upper

    # Compute transfer cushions using bands per paper Eq. B.4
    # Cushion = fraction of test points where ANY label has score in the band
    cushions = {}
    for rate_type in rate_types:
        epsilon_total = 0.0
        fold_res = results[rate_type]["fold_results"]
        weights = results[rate_type]["weights"]

        for fold_data, w_f in zip(fold_res, weights, strict=False):
            test_indices = fold_data["test_indices"]
            bands_f = fold_data["bands"]
            m_f = len(test_indices)

            # Count test points where ANY label has score in ANY class band
            flip_count = 0
            for i in test_indices:
                test_probs_i = probs[i]

                # Check if any label has score in any class band
                in_band = False
                for class_label in [0, 1]:
                    score = 1.0 - test_probs_i[class_label]
                    band_lower, band_upper = bands_f[class_label]
                    if band_lower <= score <= band_upper:
                        in_band = True
                        break

                if in_band:
                    flip_count += 1

            epsilon_f = flip_count / m_f if m_f > 0 else 0.0
            epsilon_total += w_f * epsilon_f

        cushions[rate_type] = epsilon_total

    # Transfer bounds to single rule
    transferred = transfer_bounds_to_single_rule(results, cushions)

    # Package results
    rate_bounds = {}
    for rate_type in rate_types:
        rate_bounds[rate_type] = OperationalRateBounds(
            rate_name=rate_type,
            lower_bound=transferred[rate_type]["single_lower"],
            upper_bound=transferred[rate_type]["single_upper"],
            cross_fit_lower=transferred[rate_type]["cf_lower"],
            cross_fit_upper=transferred[rate_type]["cf_upper"],
            cushion=transferred[rate_type]["cushion"],
            ci_width=confidence_level,
            fold_results=results[rate_type]["fold_results"],
        )

    # Thresholds dict from last fold (for reference)
    final_thresholds = fold_thresholds[-1] if fold_thresholds else {0: 0.0, 1: 0.0}

    return OperationalRateBoundsResult(
        rate_bounds=rate_bounds,
        rate_confidence=1 - delta,
        thresholds=final_thresholds,
        n_calibration=n,
        n_folds=n_folds,
    )


def compute_mondrian_operational_bounds(
    calibration_result: dict[int, dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    delta: float,
    rate_types: list[str] | None = None,
    confidence_level: float = 0.95,
    n_folds: int = 5,
    random_seed: int | None = None,
) -> dict[int, OperationalRateBoundsResult]:
    """Compute PER-CLASS operational rate bounds via cross-validated Mondrian.

    Uses cross-validation on MARGINAL data, then separates results by true class.
    This is the ONLY valid way to get per-class operational bounds for Mondrian,
    because Mondrian prediction sets require BOTH class thresholds simultaneously.

    Process:
    1. K-fold split MARGINAL data
    2. For each fold: train Mondrian (both thresholds), test on marginal fold
    3. Separate test results BY TRUE CLASS to get per-class rates
    4. Compute CP bounds per class at specified confidence level
    5. Average and transfer

    Parameters
    ----------
    calibration_result : dict[int, dict]
        Output from mondrian_conformal_calibrate() (provides alpha_target, delta)
    labels : np.ndarray
        True labels (shape: n,)
    probs : np.ndarray
        Probability matrix (shape: n, 2)
    delta : float
        PAC risk for operational rate bounds. Split across classes only (union bound).
        Each class independently gets confidence 1-δ for all its rates.
    rate_types : list[str], optional
        Operational rates to compute. Each gets the same confidence independently.
    confidence_level : float, default=0.95
        Confidence level for Clopper-Pearson intervals (e.g., 0.95 for 95% CIs)
    n_folds : int, default=5
        Number of cross-validation folds
    random_seed : int, optional
        Random seed for reproducible results

    Returns
    -------
    results : dict[int, OperationalRateBoundsResult]
        Dictionary mapping class_label -> OperationalRateBoundsResult

    Examples
    --------
    >>> # Step 1: Mondrian calibration
    >>> class_data = split_by_class(labels, probs)
    >>> cal_result, pred_stats = mondrian_conformal_calibrate(class_data, 0.1, 0.05)
    >>>
    >>> # Step 2: Per-class operational bounds
    >>> bounds = compute_mondrian_operational_bounds(
    ...     cal_result, labels, probs, delta=0.05, confidence_level=0.95
    ... )

    Notes
    -----
    Union bound applies across classes and folds, but NOT across rate types.
    Each rate type independently achieves confidence 1-δ_class where δ_class = δ/2.
    The CI width is controlled separately by confidence_level parameter.
    """
    from .conformal import split_by_class
    from .core import ssbc_correct

    n = len(labels)
    fold_size = n // n_folds
    n_classes = 2

    # Get alpha and delta from calibration result
    alpha_target = {k: calibration_result[k]["alpha_target"] for k in [0, 1]}
    delta_coverage = {k: calibration_result[k]["delta"] for k in [0, 1]}

    # Default rate types (include conditional singleton rates)
    if rate_types is None:
        rate_types = ["singleton", "doublet", "abstention", "correct_in_singleton", "error_in_singleton"]

    # Allocate delta: split across classes only (NOT rates or folds)
    # Each rate within a class gets confidence 1-delta_per_class independently
    delta_per_class = delta / n_classes

    # Initialize results for each class
    per_class_results = {
        class_label: {rt: {"fold_results": [], "weights": []} for rt in rate_types} for class_label in [0, 1]
    }

    # Create fold indices
    indices = np.arange(n)
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    rng.shuffle(indices)

    fold_thresholds_all = []

    for f in range(n_folds):
        # Define fold
        start_idx = f * fold_size
        end_idx = (f + 1) * fold_size if f < n_folds - 1 else n
        test_indices = indices[start_idx:end_idx]
        train_indices = np.setdiff1d(indices, test_indices)

        m_f = len(test_indices)

        # Train: compute BOTH Mondrian thresholds on training data
        train_labels = labels[train_indices]
        train_probs = probs[train_indices]
        train_class_data = split_by_class(train_labels, train_probs)

        thresholds = {}
        bands = {}
        for class_label in [0, 1]:
            class_data = train_class_data[class_label]
            n_class = class_data["n"]

            if n_class == 0:
                thresholds[class_label] = np.inf
                bands[class_label] = (-np.inf, np.inf)
                continue

            # Compute scores
            class_probs = class_data["probs"]
            class_scores = 1.0 - class_probs[:, class_label]

            # SSBC correction
            ssbc_result = ssbc_correct(
                alpha_target=alpha_target[class_label], n=n_class, delta=delta_coverage[class_label], mode="beta"
            )
            alpha_corrected = ssbc_result.alpha_corrected

            # Threshold index
            k = int(np.ceil((n_class + 1) * (1 - alpha_corrected)))
            k = min(k, n_class)
            sorted_scores = np.sort(class_scores)
            thresholds[class_label] = sorted_scores[k - 1] if k > 0 else sorted_scores[0]

            # Band for cushion: [S_(k), S_(k+1)] per paper Eq. B.4
            band_lower = sorted_scores[k - 1] if k >= 1 else -np.inf  # S_(k) = threshold
            if k < n_class:
                band_upper = sorted_scores[k]  # S_(k+1)
            else:
                # Edge case: no next order statistic, use threshold (no flip risk)
                band_upper = sorted_scores[k - 1] if k >= 1 else np.inf
            bands[class_label] = (band_lower, band_upper)

        fold_thresholds_all.append(thresholds)

        # Test: apply Mondrian to test fold, THEN separate by true class
        test_probs = probs[test_indices]
        test_labels = labels[test_indices]

        # Generate prediction sets using BOTH Mondrian thresholds
        test_prediction_sets = []
        for i in range(m_f):
            score_0 = 1.0 - test_probs[i, 0]
            score_1 = 1.0 - test_probs[i, 1]

            pred_set = []
            if score_0 <= thresholds[0]:
                pred_set.append(0)
            if score_1 <= thresholds[1]:
                pred_set.append(1)

            test_prediction_sets.append(pred_set)

        # NOW separate by true class and compute rates PER CLASS
        for true_class in [0, 1]:
            # Filter test data to this true class
            class_mask = test_labels == true_class
            class_pred_sets = [test_prediction_sets[i] for i in range(m_f) if class_mask[i]]
            class_test_labels = test_labels[class_mask]
            m_class = len(class_test_labels)

            if m_class == 0:
                continue

            # Compute operational rates for this class
            for rate_type in rate_types:
                Z = compute_operational_rate(
                    class_pred_sets,
                    class_test_labels,
                    cast(
                        Literal["singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"],
                        rate_type,
                    ),
                )
                K_fr = int(np.sum(Z))

                # Compute Clopper-Pearson bounds at specified confidence level
                L_fr = clopper_pearson_lower(K_fr, m_class, confidence_level)
                U_fr = clopper_pearson_upper(K_fr, m_class, confidence_level)

                per_class_results[true_class][rate_type]["fold_results"].append(
                    {
                        "fold": f,
                        "m_f": m_class,
                        "K_fr": K_fr,
                        "L_fr": L_fr,
                        "U_fr": U_fr,
                        "thresholds": thresholds.copy(),
                        "bands": bands.copy(),
                        "test_indices": test_indices[class_mask],
                    }
                )
                # Weight by class size in this fold
                n_class_total = np.sum(labels == true_class)
                per_class_results[true_class][rate_type]["weights"].append(m_class / n_class_total)

    # Aggregate for each class
    results = {}
    for class_label in [0, 1]:
        # Aggregate cross-fit bounds
        for rate_type in rate_types:
            fold_res = per_class_results[class_label][rate_type]["fold_results"]
            weights = per_class_results[class_label][rate_type]["weights"]

            if len(fold_res) == 0:
                continue

            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            cf_lower = sum(w * fr["L_fr"] for w, fr in zip(weights, fold_res, strict=False))
            cf_upper = sum(w * fr["U_fr"] for w, fr in zip(weights, fold_res, strict=False))

            per_class_results[class_label][rate_type]["cf_lower"] = cf_lower
            per_class_results[class_label][rate_type]["cf_upper"] = cf_upper

        # Compute transfer cushions using bands per paper Eq. B.4
        cushions = {}
        for rate_type in rate_types:
            epsilon_total = 0.0
            fold_res = per_class_results[class_label][rate_type]["fold_results"]
            weights = per_class_results[class_label][rate_type]["weights"]

            if len(fold_res) == 0:
                cushions[rate_type] = 0.0
                continue

            for fold_data, w_f in zip(fold_res, weights, strict=False):
                test_indices = fold_data["test_indices"]
                bands_f = fold_data["bands"]
                m_f = len(test_indices)

                # Count test points where ANY label has score in ANY class band
                flip_count = 0
                for idx in test_indices:
                    test_probs_i = probs[idx]

                    # Check if any label has score in any class band
                    in_band = False
                    for c_label in [0, 1]:
                        score = 1.0 - test_probs_i[c_label]
                        band_lower, band_upper = bands_f[c_label]
                        if band_lower <= score <= band_upper:
                            in_band = True
                            break

                    if in_band:
                        flip_count += 1

                epsilon_f = flip_count / m_f if m_f > 0 else 0.0
                epsilon_total += w_f * epsilon_f

            cushions[rate_type] = epsilon_total

        # Transfer bounds
        transferred = transfer_bounds_to_single_rule(per_class_results[class_label], cushions)

        # Package results
        rate_bounds = {}
        for rate_type in rate_types:
            if rate_type not in transferred:
                continue

            rate_bounds[rate_type] = OperationalRateBounds(
                rate_name=rate_type,
                lower_bound=transferred[rate_type]["single_lower"],
                upper_bound=transferred[rate_type]["single_upper"],
                cross_fit_lower=transferred[rate_type]["cf_lower"],
                cross_fit_upper=transferred[rate_type]["cf_upper"],
                cushion=transferred[rate_type]["cushion"],
                ci_width=confidence_level,
                fold_results=per_class_results[class_label][rate_type]["fold_results"],
            )

        threshold = calibration_result[class_label]["threshold"]
        n_class_total = np.sum(labels == class_label)

        results[class_label] = OperationalRateBoundsResult(
            rate_bounds=rate_bounds,
            rate_confidence=1 - delta_per_class,
            thresholds=threshold,
            n_calibration=n_class_total,
            n_folds=n_folds,
        )

    return results
