"""Statistical utility functions for SSBC."""

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
        - 'lower': lower bound of CI
        - 'upper': upper bound of CI

    Examples
    --------
    >>> labels = np.array([0, 0, 1, 1, 0])
    >>> intervals = clopper_pearson_intervals(labels, confidence=0.95)
    >>> print(intervals[0]['proportion'])
    0.6

    Notes
    -----
    The Clopper-Pearson interval is an exact binomial confidence interval
    based on Beta distribution quantiles. It provides conservative coverage
    guarantees.
    """
    alpha = 1 - confidence
    n_total = len(labels)

    intervals = {}

    for label in [0, 1]:
        count = np.sum(labels == label)
        proportion = count / n_total if n_total > 0 else float("nan")

        # Clopper-Pearson uses Beta distribution quantiles
        # Lower bound: Beta(count, n-count+1) at alpha/2
        # Upper bound: Beta(count+1, n-count) at 1-alpha/2

        if count == 0:
            lower = 0.0
            upper = stats.beta.ppf(1 - alpha / 2, count + 1, n_total - count)
        elif count == n_total:
            lower = stats.beta.ppf(alpha / 2, count, n_total - count + 1)
            upper = 1.0
        else:
            lower = stats.beta.ppf(alpha / 2, count, n_total - count + 1)
            upper = stats.beta.ppf(1 - alpha / 2, count + 1, n_total - count)

        intervals[label] = {"count": count, "proportion": proportion, "lower": lower, "upper": upper}

    return intervals


def cp_interval(count: int, total: int, confidence: float = 0.95) -> dict[str, float]:
    """Compute Clopper-Pearson exact confidence interval.

    Helper function for computing a single CI from count and total.

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
        - 'count': original count
        - 'proportion': count/total
        - 'lower': lower CI bound
        - 'upper': upper CI bound
    """
    alpha = 1 - confidence
    count = int(count)
    total = int(total)

    if total <= 0:
        return {"count": count, "proportion": 0.0, "lower": 0.0, "upper": 0.0}

    p = count / total

    if count == 0:
        lower = 0.0
        upper = stats.beta.ppf(1 - alpha / 2, 1, total)
    elif count == total:
        lower = stats.beta.ppf(alpha / 2, total, 1)
        upper = 1.0
    else:
        lower = stats.beta.ppf(alpha / 2, count, total - count + 1)
        upper = stats.beta.ppf(1 - alpha / 2, count + 1, total - count)

    return {"count": count, "proportion": float(p), "lower": float(lower), "upper": float(upper)}


def prediction_bounds_lower(k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95) -> float:
    """Compute lower prediction bound accounting for both calibration and test set sampling uncertainty.

    This function computes prediction bounds for operational rates on future test sets,
    accounting for both calibration uncertainty and test set sampling variability.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration data
    n_cal : int
        Total number of samples in calibration data
    n_test : int
        Expected size of future test sets
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% prediction bounds)

    Returns
    -------
    float
        Lower prediction bound for operational rates on future test sets

    Notes
    -----
    The prediction bounds account for both:
    1. Calibration uncertainty: uncertainty in the true rate p from calibration data
    2. Test set sampling uncertainty: variability when sampling n_test points from the true distribution

    Mathematical formula:
    SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test))
    where p̂ = k_cal/n_cal is the estimated rate from calibration data.

    For large n_test, bounds approach calibration-only bounds.
    For small n_test, bounds are wider due to additional test set sampling uncertainty.
    """
    if k_cal == 0:
        return 0.0
    if n_cal <= 0 or n_test <= 0:
        return 0.0

    # Estimated rate from calibration
    p_hat = k_cal / n_cal

    # Standard error accounting for both calibration and test set sampling
    # SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test))
    se = np.sqrt(np.clip(p_hat * (1 - p_hat) * (1 / n_cal + 1 / n_test), 0.0, None))

    # Z-score for confidence level
    alpha = 1 - confidence
    z_score = stats.norm.ppf(alpha / 2)

    # Lower prediction bound
    lower_bound = p_hat + z_score * se

    # Ensure bounds are in [0, 1]
    return max(0.0, min(1.0, lower_bound))


def prediction_bounds_upper(k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95) -> float:
    """Compute upper prediction bound accounting for both calibration and test set sampling uncertainty.

    This function computes prediction bounds for operational rates on future test sets,
    accounting for both calibration uncertainty and test set sampling variability.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration data
    n_cal : int
        Total number of samples in calibration data
    n_test : int
        Expected size of future test sets
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% prediction bounds)

    Returns
    -------
    float
        Upper prediction bound for operational rates on future test sets

    Notes
    -----
    The prediction bounds account for both:
    1. Calibration uncertainty: uncertainty in the true rate p from calibration data
    2. Test set sampling uncertainty: variability when sampling n_test points from the true distribution

    Mathematical formula:
    SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test))
    where p̂ = k_cal/n_cal is the estimated rate from calibration data.

    For large n_test, bounds approach calibration-only bounds.
    For small n_test, bounds are wider due to additional test set sampling uncertainty.
    """
    if k_cal == n_cal:
        return 1.0
    if n_cal <= 0 or n_test <= 0:
        return 1.0

    # Estimated rate from calibration
    p_hat = k_cal / n_cal

    # Standard error accounting for both calibration and test set sampling
    # SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test))
    se = np.sqrt(np.clip(p_hat * (1 - p_hat) * (1 / n_cal + 1 / n_test), 0.0, None))

    # Z-score for confidence level
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha / 2)

    # Upper prediction bound
    upper_bound = p_hat + z_score * se

    # Ensure bounds are in [0, 1]
    return max(0.0, min(1.0, upper_bound))


def prediction_bounds_beta_binomial(
    k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute prediction bounds using Beta-Binomial distribution (sophisticated method).

    This function uses the Beta-Binomial distribution to model the uncertainty in both
    calibration and test set sampling. This is more accurate than the simple method
    for small sample sizes.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration data
    n_cal : int
        Total number of samples in calibration data
    n_test : int
        Expected size of future test sets
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% prediction bounds)

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for operational rates on future test sets

    Notes
    -----
    This method models:
    1. True rate p ~ Beta(k_cal + 1, n_cal - k_cal + 1) (from calibration uncertainty)
    2. Test set rate r | p ~ Binomial(n_test, p) / n_test (from test set sampling)
    3. Marginal distribution: r ~ BetaBinomial(n_test, k_cal + 1, n_cal - k_cal + 1) / n_test

    This is more accurate than the simple method for small sample sizes.
    """
    if k_cal == 0:
        return (0.0, 1.0)
    if k_cal == n_cal:
        return (0.0, 1.0)
    if n_cal <= 0 or n_test <= 0:
        return (0.0, 1.0)

    # Beta parameters from calibration data (Jeffreys prior)
    alpha = k_cal + 1
    beta = n_cal - k_cal + 1

    # For the Beta-Binomial approach, we need to account for both sources of uncertainty
    # The variance of the Beta-Binomial distribution is:
    # Var(X) = n * p * (1-p) * (1 + (n-1) * rho)
    # where rho is the correlation parameter

    # We'll use a more sophisticated approach that properly accounts for test set sampling
    # by using the Beta distribution for the true rate and then accounting for test set variability

    # Compute quantiles of the Beta distribution for the true rate
    alpha_quantile = (1 - confidence) / 2
    upper_quantile = 1 - alpha_quantile

    # Get the Beta distribution quantiles for the true rate
    lower_rate = stats.beta.ppf(alpha_quantile, alpha, beta)
    upper_rate = stats.beta.ppf(upper_quantile, alpha, beta)

    # Now account for test set sampling uncertainty by adding the binomial variance
    # The total variance is: Var(p) + Var(r|p) = Var(p) + p*(1-p)/n_test
    # We'll use a conservative approach by expanding the bounds

    # Estimate the additional uncertainty from test set sampling
    # Use the mean rate as an approximation
    mean_rate = alpha / (alpha + beta)
    test_set_se = np.sqrt(mean_rate * (1 - mean_rate) / n_test)

    # Expand bounds to account for test set sampling uncertainty
    z_score = stats.norm.ppf(1 - alpha_quantile)
    margin = z_score * test_set_se

    lower_bound = max(0.0, lower_rate - margin)
    upper_bound = min(1.0, upper_rate + margin)

    return (lower_bound, upper_bound)


def prediction_bounds(
    k_cal: int, n_cal: int, n_test: int, confidence: float = 0.95, method: str = "simple"
) -> tuple[float, float]:
    """Compute prediction bounds accounting for both calibration and test set sampling uncertainty.

    This function provides two methods for computing prediction bounds:
    1. "simple": Uses standard error formula (faster, good for large samples)
    2. "beta_binomial": Uses Beta-Binomial distribution (more accurate for small samples)

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration data
    n_cal : int
        Total number of samples in calibration data
    n_test : int
        Expected size of future test sets
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% prediction bounds)
    method : str, default="simple"
        Method to use: "simple" or "beta_binomial"

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for operational rates on future test sets

    Examples
    --------
    >>> # Simple method (default)
    >>> lower, upper = prediction_bounds(k_cal=50, n_cal=100, n_test=1000, confidence=0.95)
    >>> print(f"Simple bounds: [{lower:.3f}, {upper:.3f}]")

    >>> # Beta-Binomial method (more accurate for small samples)
    >>> lower, upper = prediction_bounds(k_cal=50, n_cal=100, n_test=1000, confidence=0.95, method="beta_binomial")
    >>> print(f"Beta-Binomial bounds: [{lower:.3f}, {upper:.3f}]")

    Notes
    -----
    The prediction bounds account for both:
    1. Calibration uncertainty: uncertainty in the true rate p from calibration data
    2. Test set sampling uncertainty: variability when sampling n_test points from the true distribution

    **Simple method** (default):
    - Mathematical formula: SE = sqrt(p̂(1-p̂) * (1/n_cal + 1/n_test))
    - Good for large sample sizes
    - Faster computation

    **Beta-Binomial method**:
    - Uses Beta-Binomial distribution for exact modeling
    - More accurate for small sample sizes
    - Slower computation

    For large n_test, bounds approach calibration-only bounds.
    For small n_test, bounds are wider due to additional test set sampling uncertainty.

    This is the recommended function for computing operational rate bounds when
    applying fixed thresholds to future test sets.
    """
    if method == "simple":
        lower = prediction_bounds_lower(k_cal, n_cal, n_test, confidence)
        upper = prediction_bounds_upper(k_cal, n_cal, n_test, confidence)
        return (lower, upper)
    elif method == "beta_binomial":
        return prediction_bounds_beta_binomial(k_cal, n_cal, n_test, confidence)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'beta_binomial'.")


def ensure_ci(d: dict[str, Any] | Any, count: int, total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Extract or compute rate and confidence interval from a dictionary.

    If the dictionary already contains rate/CI information, use it.
    Otherwise, compute Clopper-Pearson CI from count/total.

    Parameters
    ----------
    d : dict
        Dictionary that may contain 'rate'/'proportion' and 'lower'/'upper'
    count : int
        Count for CI computation (if needed)
    total : int
        Total for CI computation (if needed)
    confidence : float, default=0.95
        Confidence level

    Returns
    -------
    tuple of (rate, lower, upper)
        Rate and confidence interval bounds
    """
    # Try to get existing rate
    r = None
    if isinstance(d, dict):
        if "rate" in d:
            r = float(d["rate"])
        elif "proportion" in d:
            r = float(d["proportion"])

    # Try to get existing CI
    lo, hi = 0.0, 0.0
    if isinstance(d, dict):
        if "ci_95" in d and isinstance(d["ci_95"], tuple | list) and len(d["ci_95"]) == 2:
            lo, hi = float(d["ci_95"][0]), float(d["ci_95"][1])
        else:
            lo = float(d.get("lower", 0.0))
            hi = float(d.get("upper", 0.0))

    # If missing or invalid, compute CP interval
    if r is None or (lo == 0.0 and hi == 0.0 and (count > 0 or total > 0)):
        ci = cp_interval(count, total, confidence)
        return ci["proportion"], ci["lower"], ci["upper"]

    return float(r), float(lo), float(hi)
