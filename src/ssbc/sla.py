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
    """Bounds for a single operational rate with PAC guarantees from LOO-CV.

    Attributes
    ----------
    rate_name : str
        Name of the operational rate (e.g., "singleton", "doublet")
    lower_bound : float
        Lower bound from Clopper-Pearson on LOO evaluations
    upper_bound : float
        Upper bound from Clopper-Pearson on LOO evaluations
    confidence_level : float
        Confidence level for the bounds (1 - δ_rate)
    n_evaluations : int
        Number of LOO evaluations (n for marginal, n_class for per-class)
    n_successes : int
        Number of times the event occurred (K)
    """

    rate_name: str
    lower_bound: float
    upper_bound: float
    confidence_level: float
    n_evaluations: int
    n_successes: int


@dataclass
class OperationalRateBoundsResult:
    """Operational rate bounds for conformal prediction from LOO-CV.

    Provides rigorous bounds on operational rates (singleton, doublet, abstention, error)
    computed via leave-one-out cross-validation (LOO-CV) with Clopper-Pearson CIs.

    Use in combination with mondrian_conformal_calibrate() output for complete SLA:
    - Mondrian provides: PAC coverage guarantees (via SSBC)
    - This provides: Operational rate bounds (via LOO + Clopper-Pearson)

    Attributes
    ----------
    rate_bounds : dict[str, OperationalRateBounds]
        Bounds for each operational rate type
    rate_confidence : float
        Confidence level for rate bounds (1 - δ)
    thresholds : dict[int, float] | float
        Reference thresholds from full calibration (for display only)
    n_calibration : int
        Total calibration set size (equals n_evaluations for marginal)
    """

    rate_bounds: dict[str, OperationalRateBounds]
    rate_confidence: float
    thresholds: dict[int, float] | float
    n_calibration: int


# ============================================================================
# LOO-CV (Leave-One-Out Cross-Validation) Operational Bounds
# ============================================================================


def _evaluate_loo_marginal(
    i: int,
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_dict: dict[int, float],
    delta_dict: dict[int, float],
    rate_types: list[str],
) -> dict[str, bool]:
    """Evaluate single LOO iteration for marginal operational bounds.

    Parameters
    ----------
    i : int
        Index of held-out point
    labels : np.ndarray
        All labels
    probs : np.ndarray
        All probabilities
    alpha_dict : dict[int, float]
        Alpha values per class
    delta_dict : dict[int, float]
        Delta values per class
    rate_types : list[str]
        Rate types to evaluate

    Returns
    -------
    dict[str, bool]
        Dictionary mapping rate_type -> whether event occurred
    """
    from .conformal import split_by_class
    from .core import ssbc_correct

    n = len(labels)

    # Create training set = all points except i
    train_indices = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
    train_labels = labels[train_indices]
    train_probs = probs[train_indices]

    # Split training data by class
    train_class_data = split_by_class(train_labels, train_probs)

    # Compute Mondrian thresholds for each class
    thresholds = {}
    for class_label in [0, 1]:
        class_data = train_class_data[class_label]
        n_class = class_data["n"]

        if n_class == 0:
            thresholds[class_label] = np.inf
            continue

        # Compute nonconformity scores for this class
        class_probs = class_data["probs"]
        class_scores = 1.0 - class_probs[:, class_label]

        # Apply SSBC
        ssbc_result = ssbc_correct(
            alpha_target=alpha_dict[class_label], n=n_class, delta=delta_dict[class_label], mode="beta"
        )
        alpha_corrected = ssbc_result.alpha_corrected

        # Conformal threshold
        k = int(np.ceil((n_class + 1) * (1 - alpha_corrected)))
        k = min(k, n_class)
        sorted_scores = np.sort(class_scores)
        thresholds[class_label] = sorted_scores[k - 1] if k > 0 else sorted_scores[0]

    # Apply Mondrian predictor to held-out point i
    test_prob = probs[i]
    test_label = labels[i]

    # Generate prediction set
    score_0 = 1.0 - test_prob[0]
    score_1 = 1.0 - test_prob[1]

    pred_set = []
    if score_0 <= thresholds[0]:
        pred_set.append(0)
    if score_1 <= thresholds[1]:
        pred_set.append(1)

    # Evaluate each rate type
    results = {}
    for rate_type in rate_types:
        event_occurred = compute_operational_rate(
            [pred_set],
            np.array([test_label]),
            cast(
                Literal["singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"],
                rate_type,
            ),
        )[0]
        results[rate_type] = bool(event_occurred)

    return results


def compute_marginal_operational_bounds(
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: float | dict[int, float],
    delta_coverage: float | dict[int, float],
    delta: float,
    rate_types: list[str] | None = None,
    ci_width: float = 0.95,
    n_jobs: int = 1,
) -> OperationalRateBoundsResult:
    """Compute marginal operational rate bounds via LOO-CV.

    Uses leave-one-out cross-validation (LOO-CV) on MARGINAL (mixed-class) data:
    1. For each point i (i=0 to n-1):
       - Train Mondrian on all other points (split by class, compute thresholds)
       - Apply Mondrian predictor to held-out point i
       - Evaluate operational rates for point i
    2. For each rate type:
       - Count K = number of times event occurred across all n evaluations
       - Apply Clopper-Pearson to (K, n) for rigorous bounds

    Parameters
    ----------
    labels : np.ndarray
        True labels (shape: n,)
    probs : np.ndarray
        Probability matrix (shape: n, 2)
    alpha_target : float or dict[int, float]
        Target miscoverage rate per class for Mondrian thresholds
    delta_coverage : float or dict[int, float]
        PAC risk for coverage (used for SSBC)
    delta : float
        PAC risk for operational rate bounds. Each rate independently gets
        confidence 1-δ (NOT split across rates).
    rate_types : list[str], optional
        Operational rates to compute. Defaults to singleton, doublet, abstention, and conditional rates.
    ci_width : float, default=0.95
        Width of Clopper-Pearson confidence intervals (e.g., 0.95 for 95% CIs)
    n_jobs : int, default=1
        Number of parallel jobs. 1 = single-threaded, -1 = use all cores, N = use N cores

    Returns
    -------
    result : OperationalRateBoundsResult
        Marginal operational rate bounds with PAC guarantees

    Examples
    --------
    >>> marginal_bounds = compute_marginal_operational_bounds(
    ...     labels, probs, alpha_target=0.1, delta_coverage=0.05, delta=0.05
    ... )
    >>> print(f"Marginal singleton rate: "
    ...       f"[{marginal_bounds.rate_bounds['singleton'].lower_bound:.2f}, "
    ...       f"{marginal_bounds.rate_bounds['singleton'].upper_bound:.2f}]")

    Notes
    -----
    Each rate type independently achieves confidence 1-δ. Union bound does NOT
    split δ across rates - if you request 5 rates with δ=0.05, each gets 95% confidence.
    """
    from .conformal import split_by_class
    from .core import ssbc_correct

    n = len(labels)

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

    # Parallel LOO-CV: evaluate each point independently
    from joblib import Parallel, delayed

    loo_results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_marginal)(i, labels, probs, alpha_dict, delta_dict, rate_types) for i in range(n)
    )

    # Aggregate indicators from parallel results
    indicators = {rt: np.zeros(n, dtype=bool) for rt in rate_types}
    for i, result in enumerate(loo_results):
        for rate_type in rate_types:
            indicators[rate_type][i] = result[rate_type]

    # Compute Clopper-Pearson bounds for each rate type
    rate_bounds = {}
    for rate_type in rate_types:
        K = int(np.sum(indicators[rate_type]))  # Number of successes

        # Apply Clopper-Pearson at specified CI width
        lower_bound = clopper_pearson_lower(K, n, ci_width)
        upper_bound = clopper_pearson_upper(K, n, ci_width)

        # PAC confidence (probability bounds hold) is separate from CI width
        pac_confidence = 1 - delta

        rate_bounds[rate_type] = OperationalRateBounds(
            rate_name=rate_type,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=pac_confidence,
            n_evaluations=n,
            n_successes=K,
        )

    # Get reference thresholds from full calibration for display
    full_class_data = split_by_class(labels, probs)
    ref_thresholds = {}
    for class_label in [0, 1]:
        n_class = full_class_data[class_label]["n"]
        if n_class > 0:
            class_probs = full_class_data[class_label]["probs"]
            class_scores = 1.0 - class_probs[:, class_label]
            ssbc_result = ssbc_correct(
                alpha_target=alpha_dict[class_label], n=n_class, delta=delta_dict[class_label], mode="beta"
            )
            alpha_corrected = ssbc_result.alpha_corrected
            k = int(np.ceil((n_class + 1) * (1 - alpha_corrected)))
            k = min(k, n_class)
            sorted_scores = np.sort(class_scores)
            ref_thresholds[class_label] = sorted_scores[k - 1] if k > 0 else sorted_scores[0]
        else:
            ref_thresholds[class_label] = np.inf

    return OperationalRateBoundsResult(
        rate_bounds=rate_bounds,
        rate_confidence=1 - delta,
        thresholds=ref_thresholds,
        n_calibration=n,
    )


def _evaluate_loo_mondrian(
    i: int,
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: dict[int, float],
    delta_coverage: dict[int, float],
    rate_types: list[str],
) -> tuple[int, dict[str, bool]]:
    """Evaluate single LOO iteration for Mondrian per-class operational bounds.

    Parameters
    ----------
    i : int
        Index of held-out point
    labels : np.ndarray
        All labels
    probs : np.ndarray
        All probabilities
    alpha_target : dict[int, float]
        Alpha values per class
    delta_coverage : dict[int, float]
        Delta values per class
    rate_types : list[str]
        Rate types to evaluate

    Returns
    -------
    true_class : int
        True class of held-out point
    results : dict[str, bool]
        Dictionary mapping rate_type -> whether event occurred
    """
    from .conformal import split_by_class
    from .core import ssbc_correct

    n = len(labels)

    # Create training set = all points except i
    train_indices = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
    train_labels = labels[train_indices]
    train_probs = probs[train_indices]

    # Split training data by class
    train_class_data = split_by_class(train_labels, train_probs)

    # Compute BOTH Mondrian thresholds
    thresholds = {}
    for class_label in [0, 1]:
        class_data = train_class_data[class_label]
        n_class = class_data["n"]

        if n_class == 0:
            thresholds[class_label] = np.inf
            continue

        # Compute nonconformity scores for this class
        class_probs = class_data["probs"]
        class_scores = 1.0 - class_probs[:, class_label]

        # Apply SSBC
        ssbc_result = ssbc_correct(
            alpha_target=alpha_target[class_label], n=n_class, delta=delta_coverage[class_label], mode="beta"
        )
        alpha_corrected = ssbc_result.alpha_corrected

        # Conformal threshold
        k = int(np.ceil((n_class + 1) * (1 - alpha_corrected)))
        k = min(k, n_class)
        sorted_scores = np.sort(class_scores)
        thresholds[class_label] = sorted_scores[k - 1] if k > 0 else sorted_scores[0]

    # Apply Mondrian predictor to held-out point i
    test_prob = probs[i]
    test_label = labels[i]

    # Generate prediction set using BOTH thresholds
    score_0 = 1.0 - test_prob[0]
    score_1 = 1.0 - test_prob[1]

    pred_set = []
    if score_0 <= thresholds[0]:
        pred_set.append(0)
    if score_1 <= thresholds[1]:
        pred_set.append(1)

    # Evaluate each rate type
    results = {}
    for rate_type in rate_types:
        event_occurred = compute_operational_rate(
            [pred_set],
            np.array([test_label]),
            cast(
                Literal["singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"],
                rate_type,
            ),
        )[0]
        results[rate_type] = bool(event_occurred)

    return int(test_label), results


def compute_mondrian_operational_bounds(
    calibration_result: dict[int, dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    delta: float,
    rate_types: list[str] | None = None,
    ci_width: float = 0.95,
    n_jobs: int = 1,
) -> dict[int, OperationalRateBoundsResult]:
    """Compute PER-CLASS operational rate bounds via LOO-CV.

    Uses leave-one-out on MARGINAL data, then separates results by true class.
    This is the ONLY valid way to get per-class operational bounds for Mondrian,
    because Mondrian prediction sets require BOTH class thresholds simultaneously.

    Process:
    1. For each point i (i=0 to n-1):
       - Train Mondrian on all other points (both thresholds)
       - Apply to held-out point i
       - Record (event_indicator, true_class_i) for each rate type
    2. Separate by true class:
       - Class 0: n_class_0 evaluations
       - Class 1: n_class_1 evaluations
    3. For each class, for each rate type:
       - Count K_class = successes for that class
       - Apply Clopper-Pearson to (K_class, n_class)

    Parameters
    ----------
    calibration_result : dict[int, dict]
        Output from mondrian_conformal_calibrate() (provides alpha_target, delta)
    labels : np.ndarray
        True labels (shape: n,)
    probs : np.ndarray
        Probability matrix (shape: n, 2)
    delta : float
        PAC risk for operational rate bounds. Split across classes only.
        Each class independently gets confidence 1-δ_class for all its rates.
    rate_types : list[str], optional
        Operational rates to compute. Each gets the same confidence independently.
    ci_width : float, default=0.95
        Width of Clopper-Pearson confidence intervals (e.g., 0.95 for 95% CIs)
    n_jobs : int, default=1
        Number of parallel jobs. 1 = single-threaded, -1 = use all cores, N = use N cores

    Returns
    -------
    results : dict[int, OperationalRateBoundsResult]
        Dictionary mapping class_label -> OperationalRateBoundsResult

    Examples
    --------
    >>> class_data = split_by_class(labels, probs)
    >>> cal_result, _ = mondrian_conformal_calibrate(class_data, 0.1, 0.05)
    >>> bounds = compute_mondrian_operational_bounds(
    ...     cal_result, labels, probs, delta=0.05
    ... )

    Notes
    -----
    Union bound applies across classes only, NOT across rate types.
    Each rate within a class independently achieves confidence 1-δ_class where δ_class = δ/2.
    """

    n = len(labels)
    n_classes = 2

    # Get alpha and delta from calibration result
    alpha_target = {k: calibration_result[k]["alpha_target"] for k in [0, 1]}
    delta_coverage = {k: calibration_result[k]["delta"] for k in [0, 1]}

    # Default rate types (include conditional singleton rates)
    if rate_types is None:
        rate_types = ["singleton", "doublet", "abstention", "correct_in_singleton", "error_in_singleton"]

    # Allocate delta: split across classes only (NOT rates)
    delta_per_class = delta / n_classes

    # Parallel LOO-CV: evaluate each point independently
    from joblib import Parallel, delayed

    loo_results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_mondrian)(i, labels, probs, alpha_target, delta_coverage, rate_types) for i in range(n)
    )

    # Aggregate indicators by true class from parallel results
    indicators_per_class = {class_label: {rt: [] for rt in rate_types} for class_label in [0, 1]}

    for true_class, result in loo_results:
        for rate_type in rate_types:
            indicators_per_class[true_class][rate_type].append(result[rate_type])

    # Compute Clopper-Pearson bounds for each class
    results = {}
    for class_label in [0, 1]:
        rate_bounds = {}
        n_class = np.sum(labels == class_label)

        for rate_type in rate_types:
            class_indicators = indicators_per_class[class_label][rate_type]
            K_class = int(np.sum(class_indicators))  # Number of successes for this class

            # Apply Clopper-Pearson at specified CI width
            lower_bound = clopper_pearson_lower(K_class, n_class, ci_width)
            upper_bound = clopper_pearson_upper(K_class, n_class, ci_width)

            # PAC confidence (probability bounds hold) is separate from CI width
            pac_confidence = 1 - delta_per_class

            rate_bounds[rate_type] = OperationalRateBounds(
                rate_name=rate_type,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=pac_confidence,
                n_evaluations=n_class,
                n_successes=K_class,
            )

        # Get reference threshold from calibration result
        threshold = calibration_result[class_label]["threshold"]

        results[class_label] = OperationalRateBoundsResult(
            rate_bounds=rate_bounds,
            rate_confidence=1 - delta_per_class,
            thresholds=threshold,
            n_calibration=n_class,
        )

    return results
