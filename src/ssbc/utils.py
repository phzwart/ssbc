"""Utility functions for conformal prediction."""

from typing import Any, Literal

import numpy as np


def build_mondrian_prediction_sets(
    probs: np.ndarray,
    threshold_0: float,
    threshold_1: float,
    return_lists: bool = False,
) -> list[set[int] | list[int]]:
    """Build prediction sets using Mondrian conformal prediction thresholds.

    This function implements the standard Mondrian conformal prediction approach:
    - For each sample, include class 0 if score_0 <= threshold_0
    - For each sample, include class 1 if score_1 <= threshold_1
    - Where score_k = 1 - P(class=k)

    Parameters
    ----------
    probs : np.ndarray, shape (n, 2)
        Probability predictions for each sample.
        probs[i, 0] = P(class=0), probs[i, 1] = P(class=1)
    threshold_0 : float
        Conformal prediction threshold for class 0
    threshold_1 : float
        Conformal prediction threshold for class 1

    return_lists : bool, default=False
        If True, returns lists instead of sets

    Returns
    -------
    list[set[int]] or list[list[int]]
        List of prediction sets, where each set/list contains the classes included
        in the prediction set for that sample.

    Examples
    --------
    >>> import numpy as np
    >>> from ssbc.utils import build_mondrian_prediction_sets
    >>>
    >>> probs = np.array([
    ...     [0.8, 0.2],  # High confidence class 0
    ...     [0.5, 0.5],  # Uncertain
    ...     [0.2, 0.8],  # High confidence class 1
    ... ])
    >>> threshold_0, threshold_1 = 0.3, 0.3
    >>> pred_sets = build_mondrian_prediction_sets(probs, threshold_0, threshold_1)
    >>> print(pred_sets)  # [{0}, {0, 1}, {1}]

    Notes
    -----
    This function is used throughout the codebase for building Mondrian conformal
    prediction sets. It centralizes the logic to ensure consistency across all
    modules that perform conformal prediction evaluation.
    """
    n = len(probs)
    if probs.shape != (n, 2):
        raise ValueError(f"probs must have shape (n, 2), got {probs.shape}")

    # Vectorize score computation
    scores_0 = 1.0 - probs[:, 0]
    scores_1 = 1.0 - probs[:, 1]

    prediction_sets = []
    for score_0, score_1 in zip(scores_0, scores_1, strict=False):
        if return_lists:
            pred_set = []
            if score_0 <= threshold_0:
                pred_set.append(0)
            if score_1 <= threshold_1:
                pred_set.append(1)
        else:
            pred_set = set()
            if score_0 <= threshold_0:
                pred_set.add(0)
            if score_1 <= threshold_1:
                pred_set.add(1)
        prediction_sets.append(pred_set)

    return prediction_sets


def compute_operational_rate(
    prediction_sets: list[set[int] | list[int]],
    true_labels: np.ndarray,
    rate_type: Literal["singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"],
) -> np.ndarray:
    """Compute operational rate indicators for prediction sets.

    For each prediction set, compute a binary indicator showing whether
    a specific operational event occurred (singleton, doublet, abstention,
    error in singleton, or correct in singleton).

    Parameters
    ----------
    prediction_sets : list[set | list]
        Prediction sets for each sample. Each set contains predicted labels.
    true_labels : np.ndarray
        True labels for each sample
    rate_type : {"singleton", "doublet", "abstention", "error_in_singleton", "correct_in_singleton"}
        Type of operational rate to compute:
        - "singleton": prediction set contains exactly one label
        - "doublet": prediction set contains exactly two labels
        - "abstention": prediction set is empty
        - "error_in_singleton": singleton prediction that doesn't contain true label
        - "correct_in_singleton": singleton prediction that contains true label

    Returns
    -------
    np.ndarray
        Binary indicators (0 or 1) for whether the event holds for each sample

    Examples
    --------
    >>> pred_sets = [{0}, {0, 1}, set(), {1}]
    >>> true_labels = np.array([0, 0, 1, 0])
    >>> indicators = compute_operational_rate(pred_sets, true_labels, "singleton")
    >>> print(indicators)  # [1, 0, 0, 1]
    >>> indicators = compute_operational_rate(pred_sets, true_labels, "correct_in_singleton")
    >>> print(indicators)  # [1, 0, 0, 0] - first and last are singletons, first is correct

    Notes
    -----
    This function is useful for computing operational statistics on conformal
    prediction sets, such as singleton rates, escalation rates, and error rates.
    """
    n = len(prediction_sets)
    indicators = np.zeros(n, dtype=int)

    for i in range(n):
        pred_set = prediction_sets[i]
        y_true = true_labels[i]

        if rate_type == "singleton":
            indicators[i] = int(len(pred_set) == 1)
        elif rate_type == "doublet":
            indicators[i] = int(len(pred_set) == 2)
        elif rate_type == "abstention":
            indicators[i] = int(len(pred_set) == 0)
        elif rate_type == "error_in_singleton":
            indicators[i] = int(len(pred_set) == 1 and y_true not in pred_set)
        elif rate_type == "correct_in_singleton":
            indicators[i] = int(len(pred_set) == 1 and y_true in pred_set)
        else:
            raise ValueError(f"Unknown rate_type: {rate_type}")

    return indicators


def evaluate_test_dataset(
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    threshold_0: float,
    threshold_1: float,
) -> dict[str, Any]:
    """Evaluate a test dataset and compute empirical operational rates.

    This function takes a test dataset with true labels and probability predictions,
    applies Mondrian conformal prediction thresholds, and returns comprehensive
    empirical rates for both marginal and per-class statistics.

    Parameters
    ----------
    test_labels : np.ndarray
        True labels for test samples (0 or 1)
    test_probs : np.ndarray
        Probability predictions for test samples, shape (n_samples, 2)
        test_probs[i, 0] = P(class=0), test_probs[i, 1] = P(class=1)
    threshold_0 : float
        Conformal prediction threshold for class 0
    threshold_1 : float
        Conformal prediction threshold for class 1

    Returns
    -------
    dict
        Dictionary containing empirical rates with structure:
        - 'marginal': Marginal rates across all samples
        - 'class_0': Rates for class 0 samples only
        - 'class_1': Rates for class 1 samples only
        Each containing:
        - 'singleton_rate': Fraction of samples with singleton predictions
        - 'doublet_rate': Fraction of samples with doublet predictions
        - 'abstention_rate': Fraction of samples with abstention (empty set)
        - 'singleton_error_rate': Fraction of singleton predictions that are incorrect
        - 'n_samples': Number of samples in this group
        - 'n_singletons': Number of singleton predictions
        - 'n_doublets': Number of doublet predictions
        - 'n_abstentions': Number of abstentions

    Examples
    --------
    >>> import numpy as np
    >>> from ssbc import evaluate_test_dataset
    >>>
    >>> # Generate test data
    >>> test_labels = np.array([0, 0, 1, 1, 0])
    >>> test_probs = np.array([
    ...     [0.8, 0.2],  # High confidence class 0
    ...     [0.6, 0.4],  # Medium confidence class 0
    ...     [0.3, 0.7],  # High confidence class 1
    ...     [0.4, 0.6],  # Medium confidence class 1
    ...     [0.5, 0.5],  # Uncertain
    ... ])
    >>>
    >>> # Evaluate with thresholds
    >>> results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)
    >>> print(f"Marginal singleton rate: {results['marginal']['singleton_rate']:.3f}")
    >>> print(f"Class 0 singleton rate: {results['class_0']['singleton_rate']:.3f}")

    Notes
    -----
    This function is useful for:
    - Evaluating conformal prediction performance on test data
    - Comparing empirical rates to theoretical bounds
    - Computing operational statistics for reporting
    - Validating that thresholds work as expected

    The function builds prediction sets using the Mondrian approach:
    - For each sample, include class 0 if score_0 <= threshold_0
    - For each sample, include class 1 if score_1 <= threshold_1
    - Where score_k = 1 - P(class=k)
    """
    n_test = len(test_labels)
    if n_test == 0:
        raise ValueError("Test dataset cannot be empty")

    if test_probs.shape != (n_test, 2):
        raise ValueError(f"test_probs must have shape ({n_test}, 2), got {test_probs.shape}")

    # Build prediction sets using Mondrian thresholds
    prediction_sets = build_mondrian_prediction_sets(test_probs, threshold_0, threshold_1)

    # Compute indicators for all rate types
    singleton_indicators = compute_operational_rate(prediction_sets, test_labels, "singleton")
    doublet_indicators = compute_operational_rate(prediction_sets, test_labels, "doublet")
    abstention_indicators = compute_operational_rate(prediction_sets, test_labels, "abstention")
    error_in_singleton_indicators = compute_operational_rate(prediction_sets, test_labels, "error_in_singleton")

    # Split by class
    class_0_mask = test_labels == 0
    class_1_mask = test_labels == 1

    def compute_rates(indicators: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
        """Compute rates for a subset of samples."""
        if mask is not None:
            subset_indicators = indicators[mask]
            n_subset = np.sum(mask)
        else:
            subset_indicators = indicators
            n_subset = len(indicators)

        if n_subset == 0:
            return {
                "singleton_rate": np.nan,
                "doublet_rate": np.nan,
                "abstention_rate": np.nan,
                "singleton_error_rate": np.nan,
                "n_samples": 0,
                "n_singletons": 0,
                "n_doublets": 0,
                "n_abstentions": 0,
            }

        # Compute rates
        singleton_rate = np.mean(subset_indicators)
        n_singletons = int(np.sum(subset_indicators))

        # For other rates, use the appropriate indicators
        if mask is not None:
            doublet_indicators_subset = doublet_indicators[mask]
            abstention_indicators_subset = abstention_indicators[mask]
            error_indicators_subset = error_in_singleton_indicators[mask]
        else:
            doublet_indicators_subset = doublet_indicators
            abstention_indicators_subset = abstention_indicators
            error_indicators_subset = error_in_singleton_indicators

        doublet_rate = np.mean(doublet_indicators_subset)
        abstention_rate = np.mean(abstention_indicators_subset)

        n_doublets = int(np.sum(doublet_indicators_subset))
        n_abstentions = int(np.sum(abstention_indicators_subset))

        # Singleton error rate: errors among singletons
        if n_singletons > 0:
            singleton_error_rate = np.mean(error_indicators_subset[subset_indicators == 1])
        else:
            singleton_error_rate = np.nan

        return {
            "singleton_rate": float(singleton_rate),
            "doublet_rate": float(doublet_rate),
            "abstention_rate": float(abstention_rate),
            "singleton_error_rate": float(singleton_error_rate) if not np.isnan(singleton_error_rate) else np.nan,
            "n_samples": int(n_subset),
            "n_singletons": n_singletons,
            "n_doublets": n_doublets,
            "n_abstentions": n_abstentions,
        }

    # Compute rates for all groups
    marginal_rates = compute_rates(singleton_indicators)
    class_0_rates = compute_rates(singleton_indicators, class_0_mask)
    class_1_rates = compute_rates(singleton_indicators, class_1_mask)

    return {
        "marginal": marginal_rates,
        "class_0": class_0_rates,
        "class_1": class_1_rates,
        "thresholds": {"threshold_0": threshold_0, "threshold_1": threshold_1},
        "n_test": n_test,
    }
