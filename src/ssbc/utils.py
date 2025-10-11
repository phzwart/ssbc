"""Utility functions for conformal prediction."""

from typing import Literal

import numpy as np


def compute_operational_rate(
    prediction_sets: list[set | list],
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
