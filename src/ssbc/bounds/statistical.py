"""Statistical bounds computation functions.

This module provides unified statistical bounds computation functions
to reduce code duplication across the SSBC codebase.
"""

from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist

# Exported public API
__all__ = [
    "clopper_pearson_lower",
    "clopper_pearson_upper", 
    "clopper_pearson_intervals",
    "cp_interval",
    "prediction_bounds",
    "compute_all_bounds",
]


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """Compute lower Clopper-Pearson (one-sided) confidence bound.

    Parameters
    ----------
    k : int
        Number of successes
    n : int
        Total number of trials
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% confidence)

    Returns
    -------
    float
        Lower confidence bound for the true proportion

    Examples
    --------
    >>> lower = clopper_pearson_lower(k=5, n=10, confidence=0.95)
    >>> print(f"Lower bound: {lower:.3f}")

    Notes
    -----
    Uses Beta distribution quantiles for exact binomial confidence bounds.
    For PAC-style guarantees, you may want to use delta = 1 - confidence.
    """
    if k == 0:
        return 0.0
    # L = Beta^{-1}(1-confidence; k, n-k+1)
    # Note: Using (1-confidence) as the lower tail probability
    alpha = 1 - confidence
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """Compute upper Clopper-Pearson (one-sided) confidence bound.

    Parameters
    ----------
    k : int
        Number of successes
    n : int
        Total number of trials
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% confidence)

    Returns
    -------
    float
        Upper confidence bound for the true proportion

    Examples
    --------
    >>> upper = clopper_pearson_upper(k=5, n=10, confidence=0.95)
    >>> print(f"Upper bound: {upper:.3f}")

    Notes
    -----
    Uses Beta distribution quantiles for exact binomial confidence bounds.
    For PAC-style guarantees, you may want to use delta = 1 - confidence.
    """
    if k == n:
        return 1.0
    # U = Beta^{-1}(confidence; k+1, n-k)
    # Note: Using confidence directly for upper tail
    return float(beta_dist.ppf(confidence, k + 1, n - k))


def clopper_pearson_intervals(labels: np.ndarray, confidence: float = 0.95) -> dict[int, dict[str, Any]]:
    """Compute Clopper-Pearson (exact binomial) confidence intervals for class prevalences.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels (0 or 1)
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)

    Returns
    -------
    dict
        Dictionary with keys 0 and 1, each containing:
        - 'count': number of samples in this class
        - 'proportion': observed proportion
        - 'lower': lower CI bound
        - 'upper': upper CI bound

    Examples
    --------
    >>> labels = np.array([0, 0, 1, 1, 0, 1])
    >>> intervals = clopper_pearson_intervals(labels, confidence=0.95)
    >>> print(f"Class 0: {intervals[0]['proportion']:.3f}")
    """
    intervals = {}
    for class_label in [0, 1]:
        mask = labels == class_label
        count = int(np.sum(mask))
        n_total = len(labels)
        
        if count == 0:
            intervals[class_label] = {
                "count": 0,
                "proportion": 0.0,
                "lower": 0.0,
                "upper": clopper_pearson_upper(0, n_total, confidence),
            }
        elif count == n_total:
            intervals[class_label] = {
                "count": count,
                "proportion": 1.0,
                "lower": clopper_pearson_lower(count, n_total, confidence),
                "upper": 1.0,
            }
        else:
            intervals[class_label] = {
                "count": count,
                "proportion": count / n_total,
                "lower": clopper_pearson_lower(count, n_total, confidence),
                "upper": clopper_pearson_upper(count, n_total, confidence),
            }
    
    return intervals


def cp_interval(count: int, total: int, confidence: float = 0.95) -> dict[str, float]:
    """Compute Clopper-Pearson confidence interval for a proportion.

    Parameters
    ----------
    count : int
        Number of successes
    total : int
        Total number of trials
    confidence : float, default=0.95
        Confidence level

    Returns
    -------
    dict
        Dictionary with keys:
        - 'proportion': observed proportion
        - 'lower': lower CI bound
        - 'upper': upper CI bound

    Examples
    --------
    >>> ci = cp_interval(count=5, total=10, confidence=0.95)
    >>> print(f"Proportion: {ci['proportion']:.3f}")
    >>> print(f"95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    """
    if total == 0:
        return {"proportion": 0.0, "lower": 0.0, "upper": 1.0}
    
    proportion = count / total
    lower = clopper_pearson_lower(count, total, confidence)
    upper = clopper_pearson_upper(count, total, confidence)
    
    return {
        "proportion": proportion,
        "lower": lower,
        "upper": upper,
    }


def prediction_bounds_lower(k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95) -> float:
    """Compute lower prediction bound for future test set.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration set
    n_cal : int
        Size of calibration set
    n_test : int
        Expected size of test set
    confidence : float, default=0.95
        Confidence level

    Returns
    -------
    float
        Lower prediction bound
    """
    if n_cal == 0:
        return 0.0
    
    # Use Clopper-Pearson for calibration uncertainty
    p_lower = clopper_pearson_lower(k_cal, n_cal, confidence)
    
    # Add sampling uncertainty for test set
    # Conservative approach: assume worst case for test set
    se = np.sqrt(p_lower * (1 - p_lower) * (1 / n_cal + 1 / n_test))
    z_alpha = stats.norm.ppf(1 - confidence)
    
    return max(0.0, p_lower - z_alpha * se)


def prediction_bounds_upper(k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95) -> float:
    """Compute upper prediction bound for future test set.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration set
    n_cal : int
        Size of calibration set
    n_test : int
        Expected size of test set
    confidence : float, default=0.95
        Confidence level

    Returns
    -------
    float
        Upper prediction bound
    """
    if n_cal == 0:
        return 1.0
    
    # Use Clopper-Pearson for calibration uncertainty
    p_upper = clopper_pearson_upper(k_cal, n_cal, confidence)
    
    # Add sampling uncertainty for test set
    # Conservative approach: assume worst case for test set
    se = np.sqrt(p_upper * (1 - p_upper) * (1 / n_cal + 1 / n_test))
    z_alpha = stats.norm.ppf(1 - confidence)
    
    return min(1.0, p_upper + z_alpha * se)


def prediction_bounds(k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95, method: str = "simple") -> tuple[float, float]:
    """Compute prediction bounds for future test set.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration set
    n_cal : int
        Size of calibration set
    n_test : int
        Expected size of test set
    confidence : float, default=0.95
        Confidence level
    method : str, default="simple"
        Method for computing bounds

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound)
    """
    if method == "simple":
        lower = prediction_bounds_lower(k_cal, n_cal, n_test, confidence)
        upper = prediction_bounds_upper(k_cal, n_cal, n_test, confidence)
        return lower, upper
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_all_bounds(
    counts: np.ndarray,
    totals: np.ndarray,
    confidence_levels: float | np.ndarray = 0.95,
) -> dict[str, Any]:
    """Compute all types of bounds for multiple proportions.

    Parameters
    ----------
    counts : np.ndarray
        Array of success counts
    totals : np.ndarray
        Array of total counts
    confidence_levels : float or np.ndarray, default=0.95
        Confidence level(s) for bounds

    Returns
    -------
    dict
        Dictionary containing all computed bounds
    """
    if isinstance(confidence_levels, (int, float)):
        confidence_levels = np.full_like(counts, confidence_levels, dtype=float)
    
    results = {
        "proportions": counts / totals,
        "clopper_pearson_lower": np.array([clopper_pearson_lower(k, n, conf) for k, n, conf in zip(counts, totals, confidence_levels)]),
        "clopper_pearson_upper": np.array([clopper_pearson_upper(k, n, conf) for k, n, conf in zip(counts, totals, confidence_levels)]),
        "cp_intervals": [cp_interval(k, n, conf) for k, n, conf in zip(counts, totals, confidence_levels)],
    }
    
    return results


def ensure_ci(d: dict[str, Any] | Any, count: int, total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Ensure a dictionary has confidence interval information.

    Parameters
    ----------
    d : dict or Any
        Dictionary that may or may not have CI information
    count : int
        Number of successes
    total : int
        Total number of trials
    confidence : float, default=0.95
        Confidence level

    Returns
    -------
    tuple[float, float, float]
        (proportion, lower_bound, upper_bound)
    """
    if isinstance(d, dict) and "proportion" in d and "lower" in d and "upper" in d:
        return d["proportion"], d["lower"], d["upper"]
    else:
        ci = cp_interval(count, total, confidence)
        return ci["proportion"], ci["lower"], ci["upper"]
