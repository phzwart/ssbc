"""Service Level Agreement (SLA) bounds for conformal prediction.

Implements PAC coverage guarantees and operational rate bounds using:
- SSBC correction for finite-sample PAC coverage
- Cross-fit Clopper-Pearson bounds for operational rates
- Transfer bounds from cross-fit to single refit-on-all rule

This enables contract-ready guarantees on:
- Coverage: P(Y ∈ C(X)) ≥ 1 - α with probability ≥ 1 - δ₁
- Operational rates: singleton, doublet, abstention, error rates with probability ≥ 1 - δ₂

Reference: Based on Appendix B of the SSBC theoretical framework.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from .core import ssbc_correct
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
    confidence_level : float
        Confidence level for the bounds (1 - δ₂)
    fold_results : list[dict]
        Detailed results from each cross-validation fold
    """

    rate_name: str
    lower_bound: float
    upper_bound: float
    cross_fit_lower: float
    cross_fit_upper: float
    cushion: float
    confidence_level: float
    fold_results: list[dict]


@dataclass
class ConformalSLAResult:
    """Complete SLA result with coverage and operational rate bounds.

    Provides joint guarantees on coverage and operational rates that hold
    simultaneously with probability ≥ 1 - (δ₁ + δ₂).

    Attributes
    ----------
    alpha_target : float
        Target miscoverage rate
    alpha_adjusted : float
        SSBC-adjusted miscoverage rate for PAC guarantee
    coverage_guarantee : float
        Guaranteed coverage (1 - α_target)
    coverage_confidence : float
        Confidence level for coverage (1 - δ₁)
    rate_bounds : dict[str, OperationalRateBounds]
        Bounds for each operational rate
    rate_confidence : float
        Confidence level for rate bounds (1 - δ₂)
    joint_confidence : float
        Joint confidence for all guarantees (1 - (δ₁ + δ₂))
    threshold : float
        Conformal prediction threshold
    n_calibration : int
        Calibration set size
    n_folds : int
        Number of cross-validation folds used
    """

    # Coverage guarantees
    alpha_target: float
    alpha_adjusted: float
    coverage_guarantee: float  # 1 - alpha_target
    coverage_confidence: float  # 1 - delta_1

    # Operational rate bounds
    rate_bounds: dict[str, OperationalRateBounds]
    rate_confidence: float  # 1 - delta_2

    # Joint guarantee
    joint_confidence: float  # 1 - (delta_1 + delta_2)

    # Deployment info
    threshold: float
    n_calibration: int
    n_folds: int


# ============================================================================
# PAC Coverage via SSBC
# ============================================================================


def compute_pac_coverage(scores: np.ndarray, alpha_target: float, delta_1: float) -> tuple[float, float, int]:
    """Compute SSBC-adjusted miscoverage for PAC coverage guarantee.

    Uses SSBC correction to ensure P(Coverage ≥ 1 - α_target) ≥ 1 - δ₁
    for the conformal prediction rule.

    Parameters
    ----------
    scores : np.ndarray
        Nonconformity scores A(X_i, Y_i) for calibration set
    alpha_target : float
        Target miscoverage rate
    delta_1 : float
        Risk tolerance for coverage guarantee

    Returns
    -------
    alpha_adj : float
        Adjusted miscoverage rate (≤ α_target)
    threshold : float
        Score threshold q for conformal prediction
    u_star : int
        Threshold index (u_star-th order statistic)

    Examples
    --------
    >>> scores = np.random.rand(100)
    >>> alpha_adj, threshold, u_star = compute_pac_coverage(scores, 0.1, 0.05)
    >>> print(f"Adjusted alpha: {alpha_adj:.4f}")

    Notes
    -----
    The threshold is the u_star-th order statistic of the calibration scores.
    The prediction rule is: predict y if A(x, y) ≤ threshold.
    """
    n = len(scores)

    # Run SSBC to get adjusted alpha
    ssbc_result = ssbc_correct(alpha_target=alpha_target, n=n, delta=delta_1, mode="beta")

    alpha_adj = ssbc_result.alpha_corrected
    u_star = ssbc_result.u_star

    # Compute threshold from adjusted alpha
    # u = ceil((n+1)*alpha), q is the u-th order statistic
    sorted_scores = np.sort(scores)
    threshold = sorted_scores[u_star - 1] if u_star <= n else np.inf

    return alpha_adj, threshold, u_star


# ============================================================================
# Cross-Fit Clopper-Pearson Bounds for Operational Rates
# ============================================================================


def cross_fit_cp_bounds(
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    score_function: Callable,
    alpha_adj: float,
    rate_types: list[str],
    n_folds: int,
    delta_2: float,
    random_seed: int | None = None,
) -> dict[str, dict]:
    """Compute cross-fit Clopper-Pearson bounds for operational rates.

    Uses K-fold cross-validation to compute PAC bounds on operational rates
    (singleton, doublet, abstention, error rates) without overfitting to
    the calibration set.

    Parameters
    ----------
    cal_features : np.ndarray
        Calibration features X_i (shape: n × d)
    cal_labels : np.ndarray
        Calibration labels Y_i (shape: n,)
    score_function : Callable
        Nonconformity score function A(x, y) that takes (features, label)
        and returns a score (higher = less conforming)
    alpha_adj : float
        SSBC-adjusted miscoverage rate from PAC coverage step
    rate_types : list[str]
        List of operational rates to compute. Options:
        "singleton", "doublet", "abstention", "error_in_singleton"
    n_folds : int
        Number of cross-validation folds (typically 5-10)
    delta_2 : float
        Total risk budget for all rate bounds
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
    The risk budget δ₂ is split evenly across folds and rate types.
    Each fold uses Clopper-Pearson exact binomial confidence intervals.
    """
    n = len(cal_labels)
    fold_size = n // n_folds

    # Allocate delta evenly across folds and rates
    n_rates = len(rate_types)
    delta_per_bound = delta_2 / (n_folds * n_rates)

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

        # Compute threshold q^{(-f)} on training fold
        train_scores = np.array([score_function(cal_features[i], cal_labels[i]) for i in train_indices])

        u_f = math.ceil((n_train + 1) * alpha_adj)
        u_f = min(u_f, n_train)
        sorted_train_scores = np.sort(train_scores)
        q_minus_f = sorted_train_scores[u_f - 1] if u_f > 0 else -np.inf

        # Also get the adjacent order statistics for cushion computation
        q_minus_f_lower = sorted_train_scores[u_f - 2] if u_f >= 2 else -np.inf
        q_minus_f_upper = sorted_train_scores[u_f] if u_f < n_train else np.inf

        # Generate prediction sets on test fold using frozen threshold
        # Infer label space from calibration set
        possible_labels = np.unique(cal_labels)

        test_prediction_sets = []
        for i in test_indices:
            x_i = cal_features[i]
            pred_set = {y for y in possible_labels if score_function(x_i, y) <= q_minus_f}
            test_prediction_sets.append(pred_set)

        test_labels = cal_labels[test_indices]

        # Compute operational rates for this fold
        for rate_type in rate_types:
            Z = compute_operational_rate(
                test_prediction_sets,
                test_labels,
                cast(Literal["singleton", "doublet", "abstention", "error_in_singleton"], rate_type),
            )
            K_fr = int(np.sum(Z))

            # Compute Clopper-Pearson bounds (one-sided confidence = 1 - delta_per_bound)
            confidence = 1 - delta_per_bound
            L_fr = clopper_pearson_lower(K_fr, m_f, confidence)
            U_fr = clopper_pearson_upper(K_fr, m_f, confidence)

            results[rate_type]["fold_results"].append(
                {
                    "fold": f,
                    "m_f": m_f,
                    "K_fr": K_fr,
                    "L_fr": L_fr,
                    "U_fr": U_fr,
                    "q_minus_f": q_minus_f,
                    "q_band": (q_minus_f_lower, q_minus_f_upper),
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
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    score_function: Callable,
    cross_fit_results: dict,
    rate_type: str,
) -> float:
    """Compute empirical cushion ε_n for transferring cross-fit bounds to single rule.

    The cushion accounts for the difference between cross-fit thresholds
    (computed on n-1 folds) and the final threshold (computed on all n samples).
    Points near the threshold boundary may flip between different prediction
    set sizes.

    Parameters
    ----------
    cal_features : np.ndarray
        Full calibration features
    cal_labels : np.ndarray
        Full calibration labels
    score_function : Callable
        Nonconformity score function A(x, y)
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
    [q^{(-f)}_{u-1}, q^{(-f)}_{u+1}] around the threshold.
    """
    fold_results = cross_fit_results["fold_results"]
    weights = cross_fit_results["weights"]

    # Get all possible labels
    possible_labels = np.unique(cal_labels)

    epsilon_total = 0.0

    for fold_data, w_f in zip(fold_results, weights, strict=False):
        q_band = fold_data["q_band"]
        q_lower, q_upper = q_band
        test_indices = fold_data["test_indices"]
        m_f = len(test_indices)

        # Count points in test fold where any label has score in the band
        flip_count = 0
        for i in test_indices:
            x_i = cal_features[i]
            # Check if any label has score in the critical band
            for y in possible_labels:
                score = score_function(x_i, y)
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
# Complete SLA Pipeline
# ============================================================================


def compute_conformal_sla(
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    score_function: Callable,
    alpha_target: float,
    delta_1: float,
    delta_2: float,
    rate_types: list[str] | None = None,
    n_folds: int = 5,
    random_seed: int | None = None,
) -> ConformalSLAResult:
    """Complete SLA pipeline: PAC coverage + operational rate bounds.

    Computes comprehensive service-level guarantees for conformal prediction:
    1. PAC coverage guarantee using SSBC
    2. Cross-fit operational rate bounds
    3. Transfer to single refit-on-all rule for deployment

    Parameters
    ----------
    cal_features : np.ndarray
        Calibration features (shape: n × d)
    cal_labels : np.ndarray
        Calibration labels (shape: n,)
    score_function : Callable
        Nonconformity score function A(x, y)
    alpha_target : float
        Target miscoverage rate (e.g., 0.1 for 90% coverage)
    delta_1 : float
        Risk for coverage guarantee (e.g., 0.05 for 95% confidence)
    delta_2 : float
        Risk for operational rate bounds (e.g., 0.05 for 95% confidence)
    rate_types : list[str], optional
        Operational rates to compute. Defaults to ["singleton", "doublet", "abstention"]
    n_folds : int, default=5
        Number of cross-validation folds
    random_seed : int, optional
        Random seed for reproducible results

    Returns
    -------
    result : ConformalSLAResult
        Complete SLA with coverage and rate guarantees

    Examples
    --------
    >>> n = 200
    >>> cal_features = np.random.randn(n, 10)
    >>> cal_labels = np.random.randint(0, 5, n)
    >>> def score_fn(x, y): return np.linalg.norm(x - y)
    >>> sla = compute_conformal_sla(cal_features, cal_labels, score_fn, 0.1, 0.05, 0.05)
    >>> print(f"Coverage: {sla.coverage_guarantee:.1%} @ {sla.coverage_confidence:.1%}")

    Notes
    -----
    The joint confidence is 1 - (δ₁ + δ₂) by union bound. All guarantees
    (coverage and rate bounds) hold simultaneously with at least this probability.
    """
    n = len(cal_labels)

    # Default rate types
    if rate_types is None:
        rate_types = ["singleton", "doublet", "abstention"]

    # Step 1: Compute PAC coverage
    all_scores = np.array([score_function(cal_features[i], cal_labels[i]) for i in range(n)])
    alpha_adj, threshold, u_star = compute_pac_coverage(all_scores, alpha_target, delta_1)

    # Step 2: Cross-fit CP bounds
    cf_results = cross_fit_cp_bounds(
        cal_features, cal_labels, score_function, alpha_adj, rate_types, n_folds, delta_2, random_seed
    )

    # Step 3: Compute cushions and transfer bounds
    cushions = {}
    for rate_type in rate_types:
        cushions[rate_type] = compute_transfer_cushion(
            cal_features, cal_labels, score_function, cf_results[rate_type], rate_type
        )

    transferred = transfer_bounds_to_single_rule(cf_results, cushions)

    # Step 4: Package results
    rate_bounds = {}
    for rate_type in rate_types:
        rate_bounds[rate_type] = OperationalRateBounds(
            rate_name=rate_type,
            lower_bound=transferred[rate_type]["single_lower"],
            upper_bound=transferred[rate_type]["single_upper"],
            cross_fit_lower=transferred[rate_type]["cf_lower"],
            cross_fit_upper=transferred[rate_type]["cf_upper"],
            cushion=transferred[rate_type]["cushion"],
            confidence_level=1 - delta_2,
            fold_results=cf_results[rate_type]["fold_results"],
        )

    return ConformalSLAResult(
        alpha_target=alpha_target,
        alpha_adjusted=alpha_adj,
        coverage_guarantee=1 - alpha_target,
        coverage_confidence=1 - delta_1,
        rate_bounds=rate_bounds,
        rate_confidence=1 - delta_2,
        joint_confidence=1 - (delta_1 + delta_2),
        threshold=threshold,
        n_calibration=n,
        n_folds=n_folds,
    )


# ============================================================================
# Mondrian (Class-Conditional) Extension
# ============================================================================


def compute_conformal_sla_mondrian(
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    score_function: Callable,
    alpha_target: float,
    delta_1: float,
    delta_2: float,
    rate_types: list[str] | None = None,
    n_folds: int = 5,
    random_seed: int | None = None,
) -> dict[int, ConformalSLAResult]:
    """Mondrian (class-conditional) conformal prediction with SLA.

    Runs the full SLA pipeline separately for each class, providing
    class-conditional guarantees. Uses union bound to split δ₂ across classes.

    Parameters
    ----------
    cal_features : np.ndarray
        Calibration features (shape: n × d)
    cal_labels : np.ndarray
        Calibration labels (shape: n,)
    score_function : Callable
        Nonconformity score function A(x, y)
    alpha_target : float
        Target miscoverage rate per class
    delta_1 : float
        Risk for coverage guarantee per class
    delta_2 : float
        Total risk for operational rate bounds (split across classes)
    rate_types : list[str], optional
        Operational rates to compute
    n_folds : int, default=5
        Number of cross-validation folds
    random_seed : int, optional
        Random seed for reproducible results

    Returns
    -------
    results : dict[int, ConformalSLAResult]
        Dictionary mapping class_label -> ConformalSLAResult

    Notes
    -----
    The risk δ₂ is split evenly across classes using union bound.
    Each class gets its own threshold and rate bounds.
    This is useful when different classes have different prevalences
    or difficulty levels.
    """
    unique_classes = np.unique(cal_labels)
    n_classes = len(unique_classes)

    # Split delta_2 across classes (union bound)
    delta_2_per_class = delta_2 / n_classes

    results = {}
    for class_label in unique_classes:
        # Filter to this class
        class_mask = cal_labels == class_label
        class_features = cal_features[class_mask]
        class_labels = cal_labels[class_mask]

        # Run standard pipeline on this class
        results[int(class_label)] = compute_conformal_sla(
            class_features,
            class_labels,
            score_function,
            alpha_target,
            delta_1,
            delta_2_per_class,
            rate_types,
            n_folds,
            random_seed,
        )

    return results
