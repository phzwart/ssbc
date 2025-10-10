"""Core SSBC (Small-Sample Beta Correction) algorithm."""

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import beta as beta_dist
from scipy.stats import betabinom, norm


@dataclass
class SSBCResult:
    """Result of SSBC correction.

    Attributes:
        alpha_target: Target miscoverage rate
        alpha_corrected: Corrected miscoverage rate (u_star / (n+1))
        u_star: Optimal u value found by the algorithm
        n: Calibration set size
        satisfied_mass: Probability that coverage >= target
        mode: "beta" for infinite test window, "beta-binomial" for finite
        details: Additional diagnostic information
    """

    alpha_target: float
    alpha_corrected: float
    u_star: int
    n: int
    satisfied_mass: float
    mode: Literal["beta", "beta-binomial"]
    details: dict


def ssbc_correct(
    alpha_target: float,
    n: int,
    delta: float,
    *,
    mode: Literal["beta", "beta-binomial"] = "beta",
    m: int | None = None,
    bracket_width: int | None = None,
) -> SSBCResult:
    """Small-Sample Beta Correction (SSBC), corrected acceptance rule.

    Find the largest α' = u/(n+1) ≤ α_target such that:
    P(Coverage(α') ≥ 1 - α_target) ≥ 1 - δ

    where Coverage(α') ~ Beta(n+1-u, u) for infinite test window.

    Parameters
    ----------
    alpha_target : float
        Target miscoverage rate (must be in (0,1))
    n : int
        Calibration set size (must be >= 1)
    delta : float
        Risk tolerance / PAC parameter (must be in (0,1))
    mode : {"beta", "beta-binomial"}, default="beta"
        "beta" for infinite test window
        "beta-binomial" for finite test window
    m : int, optional
        Test window size (required for beta-binomial mode)
    bracket_width : int, optional
        Search radius around initial guess (default: adaptive based on n)

    Returns
    -------
    SSBCResult
        Dataclass containing correction results and diagnostic details

    Raises
    ------
    ValueError
        If parameters are out of valid ranges

    Examples
    --------
    >>> result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10)
    >>> print(f"Corrected alpha: {result.alpha_corrected:.4f}")

    Notes
    -----
    The algorithm uses a bracketed search with an initial guess based on
    normal approximation to the Beta distribution. For large n, the search
    is adaptive to maintain efficiency.
    """
    # Input validation
    if not (0.0 < alpha_target < 1.0):
        raise ValueError("alpha_target must be in (0,1).")
    if n < 1:
        raise ValueError("n must be >= 1.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    if mode not in ("beta", "beta-binomial"):
        raise ValueError("mode must be 'beta' or 'beta-binomial'.")

    # Maximum u to search (α' must be ≤ α_target)
    u_max = min(n, math.floor(alpha_target * (n + 1)))
    target_coverage = 1 - alpha_target

    # Initial guess for u using normal approximation to Beta distribution
    # We want P(Beta(n+1-u, u) >= target_coverage) ≈ 1-δ
    # Using normal approximation: u ≈ u_target - z_δ * sqrt(u_target)
    # where u_target = (n+1)*α_target and z_δ = Φ^(-1)(1-δ)
    u_target = (n + 1) * alpha_target
    z_delta = norm.ppf(1 - delta)  # quantile function (inverse CDF)
    u_star_guess = max(1, math.floor(u_target - z_delta * math.sqrt(u_target)))

    # Clamp to valid range
    u_star_guess = min(u_max, u_star_guess)

    # Bracket width (Δ in Algorithm 1)
    if bracket_width is None:
        # Adaptive bracket: wider for small n, scales with √n for large n
        # For large n, the uncertainty scales as √u_target ~ (n*α)^(1/2)
        bracket_width = max(5, min(int(2 * z_delta * math.sqrt(u_target)), n // 10))
        bracket_width = min(bracket_width, 100)  # cap at 100 for efficiency

    # Search bounds - ensure we don't go outside [1, u_max]
    u_min = max(1, u_star_guess - bracket_width)
    u_search_max = min(u_max, u_star_guess + bracket_width)

    # If the guess is way off (e.g., guess > u_max), fall back to full search
    if u_min > u_search_max:
        u_min = 1
        u_search_max = u_max

    if mode == "beta-binomial":
        m_eval = m if m is not None else n
        if m_eval < 1:
            raise ValueError("m must be >= 1 for beta-binomial mode.")
        k_thresh = math.ceil(target_coverage * m_eval)

    u_star = None
    mass_star = None

    # Search from u_min up to u_search_max to find the largest u that satisfies the condition
    # Keep updating u_star as we find larger values that work
    search_log = []
    for u in range(u_min, u_search_max + 1):
        # When we calibrate at α' = u/(n+1), coverage follows:
        a = n + 1 - u  # first parameter
        b = u  # second parameter
        alpha_prime = u / (n + 1)

        if mode == "beta":
            # P(Coverage ≥ target_coverage) where Coverage ~ Beta(a, b)
            # Using: P(X >= x) = 1 - CDF(x) for continuous distributions
            ptail = 1 - beta_dist.cdf(target_coverage, a, b)
        else:
            # P(X ≥ k_thresh) where X ~ BetaBinomial(m, a, b)
            ptail = betabinom.sf(k_thresh - 1, m_eval, a, b)

        passes = ptail >= 1 - delta
        search_log.append(
            {
                "u": u,
                "alpha_prime": alpha_prime,
                "a": a,
                "b": b,
                "ptail": ptail,
                "threshold": 1 - delta,
                "passes": passes,
            }
        )

        # Accept if probability is high enough - keep updating to find the largest
        if passes:
            u_star = u
            mass_star = ptail

    # If nothing passes, fall back to u=1 (most conservative)
    if u_star is None:
        u_star = 1
        a = n + 1 - u_star
        b = u_star
        mass_star = (
            1 - beta_dist.cdf(target_coverage, a, b)
            if mode == "beta"
            else betabinom.sf(k_thresh - 1, (m if m else n), a, b)
        )

    alpha_corrected = u_star / (n + 1)

    return SSBCResult(
        alpha_target=alpha_target,
        alpha_corrected=alpha_corrected,
        u_star=u_star,
        n=n,
        satisfied_mass=mass_star,
        mode=mode,
        details=dict(
            u_max=u_max,
            u_star_guess=u_star_guess,
            search_range=(u_min, u_search_max),
            bracket_width=bracket_width,
            delta=delta,
            m=(m if (mode == "beta-binomial") else None),
            acceptance_rule="P(Coverage >= target) >= 1-delta",
            search_log=search_log,
        ),
    )
