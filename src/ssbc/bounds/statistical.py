"""Statistical utility functions for SSBC."""

from typing import Any

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln

# Exported public API
__all__ = [
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "clopper_pearson_intervals",
    "cp_interval",
    "prediction_bounds",
    "prediction_bounds_beta_binomial",
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
    return float(stats.beta.ppf(alpha, k, n - k + 1))


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
    return float(stats.beta.ppf(confidence, k + 1, n - k))


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
        return {
            "count": count,
            "proportion": float("nan"),
            "lower": 0.0,
            "upper": 1.0,
        }

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


def _log_binomial_coef(n: int, k: int) -> float:
    """Compute log(n choose k) = log(n!) - log(k!) - log((n-k)!)."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _beta_binomial_pmf(k: int, n: int, alpha: float, beta: float) -> float:
    """Beta-binomial PMF: pmf(k; n, alpha, beta) = C(n,k) * Beta(k+alpha, n-k+beta) / Beta(alpha, beta).

    Parameters
    ----------
    k : int
        Number of successes
    n : int
        Total number of trials
    alpha : float
        First Beta parameter
    beta : float
        Second Beta parameter

    Returns
    -------
    float
        Probability mass at k
    """
    return np.exp(_log_binomial_coef(n, k) + betaln(k + alpha, n - k + beta) - betaln(alpha, beta))


def prediction_bounds_beta_binomial(
    k_cal: int,
    n_cal: int,
    n_test: int,
    confidence: float = 0.95,
    use_jeffreys: bool = False,
    tail: str = "equal_tailed",
) -> tuple[float, float]:
    """Beta-Binomial predictive interval for future empirical rate.

    This function computes exact prediction bounds using the Beta-Binomial distribution,
    which properly accounts for both calibration uncertainty and test set sampling variability.

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration data
    n_cal : int
        Total number of samples in calibration data
    n_test : int
        Expected size of future test sets
    confidence : float, default=0.95
        Desired predictive mass (confidence level)
    use_jeffreys : bool, default=False
        If False, use uniform prior Beta(1,1), giving posterior Beta(k_cal+1, n_cal-k_cal+1).
        If True, use Jeffreys prior Beta(1/2,1/2), giving posterior Beta(k_cal+0.5, n_cal-k_cal+0.5).
    tail : str, default="equal_tailed"
        Interval type:
        - "equal_tailed": Invert predictive CDF (α/2 each side)
        - "hpd": Shortest high posterior density predictive interval

    Returns
    -------
    tuple[float, float]
        (lower_rate, upper_rate) for operational rates on future test sets

    Notes
    -----
    This method models:
    1. True rate p ~ Beta(alpha, beta) (posterior from calibration data)
    2. Future successes k_test | p ~ Binomial(n_test, p)
    3. Marginal predictive distribution: k_test ~ BetaBinomial(n_test, alpha, beta)
    4. Return bounds on the rate k_test / n_test

    This provides exact finite-sample prediction bounds that account for both sources
    of uncertainty without normal approximations.
    """
    if n_cal <= 0 or n_test <= 0:
        return (0.0, 1.0)

    offset = 0.5 if use_jeffreys else 1.0
    alpha = k_cal + offset
    beta = (n_cal - k_cal) + offset

    # Predictive PMF over k_test = 0..n_test
    ks = np.arange(n_test + 1)
    pmf = np.array([_beta_binomial_pmf(k, n_test, alpha, beta) for k in ks])
    pmf = pmf / pmf.sum()  # Numerical safety (normalize to ensure sum = 1)

    if tail == "equal_tailed":
        cdf = np.cumsum(pmf)
        alpha_tail = (1.0 - confidence) / 2.0

        # Lower index = smallest k with CDF >= alpha_tail
        k_lo = int(np.searchsorted(cdf, alpha_tail))
        # Upper index = largest k with CDF <= 1 - alpha_tail
        k_hi = int(np.searchsorted(cdf, 1.0 - alpha_tail, side="right") - 1)

    elif tail == "hpd":
        # Sort ks by PMF descending, then add mass until >= confidence,
        # then take min/max k in that included set. This gives a highest-density
        # predictive set, then we report its hull.
        order = np.argsort(-pmf)
        keep = []
        mass = 0.0
        for idx in order:
            keep.append(idx)
            mass += pmf[idx]
            if mass >= confidence:
                break
        k_lo = ks[min(keep)]
        k_hi = ks[max(keep)]

    else:
        raise ValueError("tail must be 'equal_tailed' or 'hpd'")

    return (k_lo / n_test, k_hi / n_test)


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
    - Uses Beta-Binomial distribution for exact finite-sample modeling
    - More accurate for small sample sizes
    - Slower computation
    - Uses uniform prior Beta(1,1) and equal-tailed intervals by default
    - For advanced use (Jeffreys prior or HPD intervals), call
      prediction_bounds_beta_binomial() directly

    For large n_test, bounds approach calibration-only bounds.
    For small n_test, bounds are wider due to additional test set sampling uncertainty.

    This is the recommended function for computing operational rate bounds when
    applying fixed thresholds to future test sets.
    """
    if method == "simple":
        raw_lower = prediction_bounds_lower(k_cal, n_cal, n_test, confidence)
        raw_upper = prediction_bounds_upper(k_cal, n_cal, n_test, confidence)
        # Ensure monotonicity (clip in case of numerical noise)
        lower = min(raw_lower, raw_upper)
        upper = max(raw_lower, raw_upper)
        return (lower, upper)
    elif method == "beta_binomial":
        return prediction_bounds_beta_binomial(k_cal, n_cal, n_test, confidence)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'beta_binomial'.")


def ensure_ci(d: dict[str, Any] | Any, count: int, total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Extract or compute rate and confidence interval from a dictionary.

    If the dictionary already contains rate/CI information, use it.
    Otherwise, compute Clopper-Pearson CI from count/total.

    This function re-normalizes to the requested confidence level if the
    provided dictionary is missing bounds or if the provided bounds look
    degenerate (NaN values).

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
    # Initialize with NaN to detect missing values
    r = np.nan
    lo = np.nan
    hi = np.nan

    if isinstance(d, dict):
        if "rate" in d:
            r = float(d["rate"])
        elif "proportion" in d:
            r = float(d["proportion"])

        if "ci_95" in d and isinstance(d["ci_95"], tuple | list) and len(d["ci_95"]) == 2:
            lo, hi = map(float, d["ci_95"])

        else:
            lo = float(d.get("lower", np.nan))
            hi = float(d.get("upper", np.nan))

    # If missing or invalid (NaN), compute CP interval
    need_ci = np.isnan(r) or np.isnan(lo) or np.isnan(hi)
    if need_ci:
        ci = cp_interval(count, total, confidence)
        return ci["proportion"], ci["lower"], ci["upper"]

    return r, lo, hi
