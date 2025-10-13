"""Distribution of realized coverage for conformal prediction.

When we calibrate at a threshold determined by SSBC, the coverage on future
test sets is a random variable. This module computes that distribution.
"""

import numpy as np
from scipy.stats import betabinom

from .core import SSBCResult
from .statistics import clopper_pearson_lower, clopper_pearson_upper


def compute_coverage_distribution(
    ssbc_result: SSBCResult,
    test_size: int,
    coverage_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the distribution of realized coverage on a future test set.

    When we calibrate conformal prediction using SSBC, we choose a quantile
    position k based on n calibration points. For a future test set of size m,
    the number of covered points follows a Beta-Binomial distribution:

        X ~ BetaBinomial(m, n+1-k, k)

    where coverage = X/m.

    Parameters
    ----------
    ssbc_result : SSBCResult
        Result from ssbc_correct() containing n, u_star (=k), alpha_corrected
    test_size : int
        Size of the future test set (m)
    coverage_grid : np.ndarray, optional
        Grid of coverage values to evaluate PMF at. If None, uses
        [0/m, 1/m, 2/m, ..., m/m]

    Returns
    -------
    dict
        Dictionary with keys:
        - 'coverage_values': array of coverage rates (x/m)
        - 'pmf': probability mass at each coverage value
        - 'cdf': cumulative distribution function
        - 'mean_coverage': expected coverage E[X/m]
        - 'std_coverage': standard deviation of coverage
        - 'quantiles': dict with keys '0.05', '0.50', '0.95' etc.

    Examples
    --------
    >>> from ssbc import ssbc_correct
    >>> ssbc_result = ssbc_correct(alpha_target=0.10, n=100, delta=0.10)
    >>> dist = compute_coverage_distribution(ssbc_result, test_size=100)
    >>> print(f"Expected coverage: {dist['mean_coverage']:.3f}")
    >>> print(f"95% quantile: {dist['quantiles']['0.95']:.3f}")

    Notes
    -----
    **Mathematical Details:**

    When we use quantile position k on n calibration points:
    - Coverage on future data ~ BetaBinomial(m, a=n+1-k, b=k)
    - This comes from the exchangeability assumption
    - The Beta parameters (a, b) encode our uncertainty about true coverage

    **Relationship to SSBC:**

    SSBC chooses k such that Pr(coverage >= 1-alpha_target) >= 1-delta.
    This function shows you the FULL distribution, not just that guarantee.

    **Interpretation:**

    - If test_size = calibration size (m = n): "typical" scenario
    - If test_size >> n: coverage concentrates around its mean
    - If test_size << n: coverage has more variability
    """
    n = ssbc_result.n
    k = ssbc_result.u_star  # This is the quantile position
    m = test_size

    # Beta-Binomial parameters
    a = n + 1 - k  # First parameter
    b = k  # Second parameter

    # Create coverage grid if not provided
    if coverage_grid is None:
        # Use all possible values: 0/m, 1/m, ..., m/m
        num_covered = np.arange(0, m + 1)
        coverage_values = num_covered / m
    else:
        coverage_values = coverage_grid
        num_covered = (coverage_values * m).astype(int)

    # Compute PMF: Pr(X = x) where X ~ BetaBinomial(m, a, b)
    pmf = betabinom.pmf(num_covered, n=m, a=a, b=b)

    # Compute CDF
    cdf = betabinom.cdf(num_covered, n=m, a=a, b=b)

    # Compute moments
    # E[X] for BetaBinomial(m, a, b) = m * a / (a + b)
    mean_num_covered = m * a / (a + b)
    mean_coverage = mean_num_covered / m

    # Var[X] for BetaBinomial
    # Var[X] = m * a * b * (a + b + m) / [(a+b)^2 * (a+b+1)]
    var_num_covered = m * a * b * (a + b + m) / ((a + b) ** 2 * (a + b + 1))
    std_coverage = np.sqrt(var_num_covered) / m

    # Compute quantiles
    quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    quantiles = {}

    for q in quantile_levels:
        # Find the smallest x such that CDF(x) >= q
        # BetaBinomial.ppf returns number of successes
        x_quantile = betabinom.ppf(q, n=m, a=a, b=b)
        quantiles[f"{q:.2f}"] = x_quantile / m

    return {
        "coverage_values": coverage_values,
        "pmf": pmf,
        "cdf": cdf,
        "mean_coverage": mean_coverage,
        "std_coverage": std_coverage,
        "quantiles": quantiles,
        "n_calibration": n,
        "test_size": m,
        "k": k,
        "alpha_adj": ssbc_result.alpha_corrected,
        "beta_params": (a, b),
    }


def plot_coverage_distribution(
    ssbc_result: SSBCResult,
    test_size: int,
    ax=None,
):
    """Plot the distribution of realized coverage.

    Parameters
    ----------
    ssbc_result : SSBCResult
        Result from ssbc_correct()
    test_size : int
        Size of future test set
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : matplotlib axes
        The axes object with the plot
    """
    import matplotlib.pyplot as plt

    dist = compute_coverage_distribution(ssbc_result, test_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PMF as bars
    ax.bar(
        dist["coverage_values"],
        dist["pmf"],
        width=1 / test_size * 0.8,
        alpha=0.6,
        label="PMF",
    )

    # Mark mean
    ax.axvline(
        dist["mean_coverage"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {dist['mean_coverage']:.3f}",
    )

    # Mark target coverage
    target_coverage = 1 - ssbc_result.alpha_target
    ax.axvline(
        target_coverage,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Target: {target_coverage:.3f}",
    )

    # Mark quantiles
    ax.axvline(
        dist["quantiles"]["0.05"],
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"5% quantile: {dist['quantiles']['0.05']:.3f}",
    )

    ax.set_xlabel("Coverage Rate")
    ax.set_ylabel("Probability Mass")
    ax.set_title(
        f"Distribution of Coverage on Test Set (m={test_size})\n"
        f"Calibrated with n={ssbc_result.n}, α_adj={ssbc_result.alpha_corrected:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def compute_operational_response_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    class_label: int = 1,
    ci_level: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute operational metrics across all possible alpha values with CP bounds.

    For each possible alpha (corresponding to each quantile position k),
    compute the operational metrics (abstentions, singletons, doublets) and
    their Clopper-Pearson confidence intervals.

    This creates a "response curve" showing how operational metrics vary with alpha.

    Parameters
    ----------
    labels : np.ndarray, shape (n,)
        True labels (0 or 1)
    probs : np.ndarray, shape (n,) or (n, 2)
        Predicted probabilities. If 1D, interpreted as P(class=1).
        If 2D, uses column for class_label.
    class_label : int, default=1
        Which class to analyze (0 or 1)
    ci_level : float, default=0.95
        Confidence level for Clopper-Pearson intervals

    Returns
    -------
    dict
        Dictionary with keys:
        - 'alpha_values': array of alpha values (one per k)
        - 'k_values': array of k values (quantile positions)
        - 'thresholds': array of conformal thresholds
        - 'n_abstentions': count of abstentions at each alpha
        - 'n_singletons': count of singletons at each alpha
        - 'n_doublets': count of doublets at each alpha
        - 'abstention_rate': abstention rate at each alpha
        - 'singleton_rate': singleton rate at each alpha
        - 'doublet_rate': doublet rate at each alpha
        - 'abstention_ci_lower': CP lower bound for abstentions
        - 'abstention_ci_upper': CP upper bound for abstentions
        - 'singleton_ci_lower': CP lower bound for singletons
        - 'singleton_ci_upper': CP upper bound for singletons
        - 'doublet_ci_lower': CP lower bound for doublets
        - 'doublet_ci_upper': CP upper bound for doublets
        - 'n_calibration': total number of calibration points

    Examples
    --------
    >>> from ssbc import BinaryClassifierSimulator
    >>> sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
    >>> labels, probs = sim.generate(n_samples=100)
    >>> curve = compute_operational_response_curve(labels, probs, class_label=1)
    >>> # Plot singleton rate with bounds
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(curve['alpha_values'], curve['singleton_rate'])
    >>> plt.fill_between(curve['alpha_values'],
    ...                   curve['singleton_ci_lower'],
    ...                   curve['singleton_ci_upper'], alpha=0.3)

    Notes
    -----
    **Interpretation:**

    This function evaluates operational metrics ON THE CALIBRATION DATA.
    - These are in-sample estimates
    - Clopper-Pearson CIs account for binomial uncertainty
    - For out-of-sample guarantees, combine with coverage_distribution

    **Per-Class Analysis:**

    For Mondrian conformal prediction, we analyze each class separately.
    - "Singleton" = true class is covered (included in prediction set)
    - "Abstention" = true class is not covered (excluded)
    - "Doublet" = both classes covered (only in full binary analysis)

    In single-class analysis (conditioned on true label = class_label):
    - Doublets are always 0
    - Singleton + Abstention = 100%
    """
    # Handle probability array format
    if probs.ndim == 1:
        if class_label == 1:
            p_class = probs
        else:
            p_class = 1 - probs
    elif probs.ndim == 2:
        p_class = probs[:, class_label]
    else:
        raise ValueError("probs must be 1D or 2D array")

    # Filter to class_label only (Mondrian per-class)
    mask = labels == class_label
    y_class = labels[mask]
    p_class = p_class[mask]
    n = len(y_class)

    if n == 0:
        raise ValueError(f"No calibration samples for class {class_label}")

    # Compute nonconformity scores
    scores = 1.0 - p_class
    sorted_scores = np.sort(scores)

    # Initialize storage
    alpha_values = []
    k_values = []
    thresholds = []
    n_abstentions_list = []
    n_singletons_list = []
    n_doublets_list = []

    # For each possible k (from 1 to n)
    for k in range(1, n + 1):
        # Alpha corresponding to this k
        alpha = k / (n + 1)
        alpha_values.append(alpha)
        k_values.append(k)

        # Threshold is the k-th smallest score (0-indexed: k-1)
        threshold = sorted_scores[k - 1]
        thresholds.append(threshold)

        # Evaluate operational metrics on calibration data
        # For each sample, check if its score <= threshold
        covered = scores <= threshold

        # In per-class analysis:
        # - Covered = singleton (true class in prediction set)
        # - Not covered = abstention (true class not in prediction set)
        n_singletons = np.sum(covered)
        n_abstentions = n - n_singletons
        n_doublets = 0  # No doublets in single-class analysis

        n_abstentions_list.append(n_abstentions)
        n_singletons_list.append(n_singletons)
        n_doublets_list.append(n_doublets)

    # Convert to arrays
    alpha_values = np.array(alpha_values)
    k_values = np.array(k_values)
    thresholds = np.array(thresholds)
    n_abstentions = np.array(n_abstentions_list)
    n_singletons = np.array(n_singletons_list)
    n_doublets = np.array(n_doublets_list)

    # Compute rates
    abstention_rate = n_abstentions / n
    singleton_rate = n_singletons / n
    doublet_rate = n_doublets / n

    # Compute Clopper-Pearson confidence intervals
    abstention_ci_lower = np.array([clopper_pearson_lower(count, n, ci_level) for count in n_abstentions])
    abstention_ci_upper = np.array([clopper_pearson_upper(count, n, ci_level) for count in n_abstentions])

    singleton_ci_lower = np.array([clopper_pearson_lower(count, n, ci_level) for count in n_singletons])
    singleton_ci_upper = np.array([clopper_pearson_upper(count, n, ci_level) for count in n_singletons])

    doublet_ci_lower = np.array([clopper_pearson_lower(count, n, ci_level) for count in n_doublets])
    doublet_ci_upper = np.array([clopper_pearson_upper(count, n, ci_level) for count in n_doublets])

    return {
        "alpha_values": alpha_values,
        "k_values": k_values,
        "thresholds": thresholds,
        "n_abstentions": n_abstentions,
        "n_singletons": n_singletons,
        "n_doublets": n_doublets,
        "abstention_rate": abstention_rate,
        "singleton_rate": singleton_rate,
        "doublet_rate": doublet_rate,
        "abstention_ci_lower": abstention_ci_lower,
        "abstention_ci_upper": abstention_ci_upper,
        "singleton_ci_lower": singleton_ci_lower,
        "singleton_ci_upper": singleton_ci_upper,
        "doublet_ci_lower": doublet_ci_lower,
        "doublet_ci_upper": doublet_ci_upper,
        "n_calibration": n,
        "class_label": class_label,
        "ci_level": ci_level,
    }


def compute_mondrian_response_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    ci_level: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute full Mondrian operational metrics with doublets across all alphas.

    Evaluates FULL binary prediction sets using thresholds for both classes.
    Unlike compute_operational_response_curve, this gives you doublets.

    Parameters
    ----------
    labels : np.ndarray, shape (n,)
        True labels (0 or 1)
    probs : np.ndarray, shape (n, 2)
        Predicted probabilities [P(class=0), P(class=1)]
    ci_level : float, default=0.95
        Confidence level for Clopper-Pearson intervals

    Returns
    -------
    dict
        Dictionary with keys:
        - 'alpha_values': array of alpha values
        - 'threshold_0': threshold for class 0 at each alpha
        - 'threshold_1': threshold for class 1 at each alpha
        - 'n_abstentions': count of empty prediction sets
        - 'n_singletons': count of singleton sets (|pred_set| = 1)
        - 'n_doublets': count of doublet sets (|pred_set| = 2)
        - 'abstention_rate', 'singleton_rate', 'doublet_rate': rates
        - Clopper-Pearson CI bounds for each rate
        - 'n_total': total samples

    Notes
    -----
    **Full Binary Mondrian:**

    For each sample, prediction set = {c : score_c <= threshold_c}
    - Abstention: pred_set = {} (both scores > thresholds)
    - Singleton: pred_set = {0} or {1} (one score <= threshold)
    - Doublet: pred_set = {0, 1} (both scores <= thresholds)

    **Marginal Analysis:**

    This evaluates prediction sets marginally (ignoring true labels).
    Shows what a user/deployment system would see.
    """
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("probs must be (n, 2) array for full Mondrian")

    n = len(labels)

    # Split by class to compute per-class thresholds
    mask_0 = labels == 0
    mask_1 = labels == 1

    scores_0_cal = 1.0 - probs[mask_0, 0]
    scores_1_cal = 1.0 - probs[mask_1, 1]

    n_0 = len(scores_0_cal)
    n_1 = len(scores_1_cal)

    sorted_scores_0 = np.sort(scores_0_cal)
    sorted_scores_1 = np.sort(scores_1_cal)

    # Use the MINIMUM n to ensure we have thresholds for both classes
    # For each alpha, we'll use the same quantile level for both classes
    n_min = min(n_0, n_1)

    alpha_values = []
    threshold_0_list = []
    threshold_1_list = []
    n_abstentions_list = []
    n_singletons_list = []
    n_doublets_list = []

    # For each k from 1 to n_min
    for k in range(1, n_min + 1):
        # Alpha for this k (using minimum n for consistency)
        alpha = k / (n_min + 1)
        alpha_values.append(alpha)

        # Compute thresholds for each class using their own calibration sets
        # Use the same quantile level (1-alpha) for both
        k_0 = int(np.ceil((n_0 + 1) * (1 - alpha)))
        k_0 = min(k_0, n_0)
        k_0 = max(k_0, 1)
        threshold_0 = sorted_scores_0[k_0 - 1]

        k_1 = int(np.ceil((n_1 + 1) * (1 - alpha)))
        k_1 = min(k_1, n_1)
        k_1 = max(k_1, 1)
        threshold_1 = sorted_scores_1[k_1 - 1]

        threshold_0_list.append(threshold_0)
        threshold_1_list.append(threshold_1)

        # Evaluate prediction sets on ALL samples
        n_abstentions = 0
        n_singletons = 0
        n_doublets = 0

        for i in range(n):
            score_0 = 1.0 - probs[i, 0]
            score_1 = 1.0 - probs[i, 1]

            # Build prediction set
            in_0 = score_0 <= threshold_0
            in_1 = score_1 <= threshold_1

            if in_0 and in_1:
                n_doublets += 1
            elif in_0 or in_1:
                n_singletons += 1
            else:
                n_abstentions += 1

        n_abstentions_list.append(n_abstentions)
        n_singletons_list.append(n_singletons)
        n_doublets_list.append(n_doublets)

    # Convert to arrays
    alpha_values = np.array(alpha_values)
    threshold_0 = np.array(threshold_0_list)
    threshold_1 = np.array(threshold_1_list)
    n_abstentions = np.array(n_abstentions_list)
    n_singletons = np.array(n_singletons_list)
    n_doublets = np.array(n_doublets_list)

    # Compute rates
    abstention_rate = n_abstentions / n
    singleton_rate = n_singletons / n
    doublet_rate = n_doublets / n

    # Clopper-Pearson CIs
    abstention_ci_lower = np.array([clopper_pearson_lower(count, n, ci_level) for count in n_abstentions])
    abstention_ci_upper = np.array([clopper_pearson_upper(count, n, ci_level) for count in n_abstentions])

    singleton_ci_lower = np.array([clopper_pearson_lower(count, n, ci_level) for count in n_singletons])
    singleton_ci_upper = np.array([clopper_pearson_upper(count, n, ci_level) for count in n_singletons])

    doublet_ci_lower = np.array([clopper_pearson_lower(count, n, ci_level) for count in n_doublets])
    doublet_ci_upper = np.array([clopper_pearson_upper(count, n, ci_level) for count in n_doublets])

    return {
        "alpha_values": alpha_values,
        "threshold_0": threshold_0,
        "threshold_1": threshold_1,
        "n_abstentions": n_abstentions,
        "n_singletons": n_singletons,
        "n_doublets": n_doublets,
        "abstention_rate": abstention_rate,
        "singleton_rate": singleton_rate,
        "doublet_rate": doublet_rate,
        "abstention_ci_lower": abstention_ci_lower,
        "abstention_ci_upper": abstention_ci_upper,
        "singleton_ci_lower": singleton_ci_lower,
        "singleton_ci_upper": singleton_ci_upper,
        "doublet_ci_lower": doublet_ci_lower,
        "doublet_ci_upper": doublet_ci_upper,
        "n_total": n,
        "n_class_0": n_0,
        "n_class_1": n_1,
        "ci_level": ci_level,
    }


def _evaluate_loo_scenario_marginal(
    alpha_0: float,
    alpha_1: float,
    labels: np.ndarray,
    probs: np.ndarray,
) -> tuple[int, int, int, int]:
    """Evaluate LOO-CV for a single (alpha_0, alpha_1) scenario (helper for parallelization).
    
    Returns
    -------
    tuple[int, int, int, int]
        (n_singletons, n_doublets, n_abstentions, n_singletons_correct)
    """
    n = len(labels)
    n_abstentions = 0
    n_singletons = 0
    n_doublets = 0
    n_singletons_correct = 0
    
    # Get per-class masks
    mask_0 = labels == 0
    mask_1 = labels == 1
    
    for idx in range(n):
        # Compute thresholds leaving out sample idx
        # Class 0 threshold (leave-one-out)
        if mask_0[idx]:
            scores_0_loo = 1.0 - probs[mask_0, 0]
            mask_0_indices = np.where(mask_0)[0]
            loo_idx_in_class0 = np.where(mask_0_indices == idx)[0][0]
            scores_0_loo = np.delete(scores_0_loo, loo_idx_in_class0)
            n_0_loo = len(scores_0_loo)
        else:
            scores_0_loo = 1.0 - probs[mask_0, 0]
            n_0_loo = len(scores_0_loo)
        
        # Class 1 threshold (leave-one-out)
        if mask_1[idx]:
            scores_1_loo = 1.0 - probs[mask_1, 1]
            mask_1_indices = np.where(mask_1)[0]
            loo_idx_in_class1 = np.where(mask_1_indices == idx)[0][0]
            scores_1_loo = np.delete(scores_1_loo, loo_idx_in_class1)
            n_1_loo = len(scores_1_loo)
        else:
            scores_1_loo = 1.0 - probs[mask_1, 1]
            n_1_loo = len(scores_1_loo)
        
        # Compute thresholds at (alpha_0, alpha_1) for this LOO fold
        k_0_loo = int(np.ceil((n_0_loo + 1) * (1 - alpha_0)))
        k_0_loo = min(k_0_loo, n_0_loo)
        k_0_loo = max(k_0_loo, 1)
        
        k_1_loo = int(np.ceil((n_1_loo + 1) * (1 - alpha_1)))
        k_1_loo = min(k_1_loo, n_1_loo)
        k_1_loo = max(k_1_loo, 1)
        
        sorted_scores_0_loo = np.sort(scores_0_loo)
        sorted_scores_1_loo = np.sort(scores_1_loo)
        
        threshold_0_loo = sorted_scores_0_loo[k_0_loo - 1]
        threshold_1_loo = sorted_scores_1_loo[k_1_loo - 1]
        
        # Evaluate on held-out sample idx
        score_0 = 1.0 - probs[idx, 0]
        score_1 = 1.0 - probs[idx, 1]
        true_label = labels[idx]
        
        in_0 = score_0 <= threshold_0_loo
        in_1 = score_1 <= threshold_1_loo
        
        # Build prediction set
        pred_set = []
        if in_0:
            pred_set.append(0)
        if in_1:
            pred_set.append(1)
        
        # Count by set size
        if len(pred_set) == 0:
            n_abstentions += 1
        elif len(pred_set) == 1:
            n_singletons += 1
            if true_label in pred_set:
                n_singletons_correct += 1
        else:  # len == 2
            n_doublets += 1
    
    return n_singletons, n_doublets, n_abstentions, n_singletons_correct


def compute_pac_operational_bounds_marginal(
    ssbc_result_0: SSBCResult,
    ssbc_result_1: SSBCResult,
    labels: np.ndarray,
    probs: np.ndarray,
    test_size: int,
    ci_level: float = 0.95,
    pac_level: float = 0.95,
    use_union_bound: bool = True,
    n_jobs: int = 1,
) -> dict[str, float | list]:
    """Compute marginal PAC-controlled operational bounds with coverage volatility.

    Framework:
    1. User chooses α₀, δ₀, α₁, δ₁
    2. SSBC finds α_adj_0, α_adj_1
    3. BetaBinomial induces P(α_realized_0), P(α_realized_1)
    4. For each (α_realized_0, α_realized_1):
       - Compute operational metrics on calibration data
       - Get CP bounds for abstentions, singletons, doublets
       - Compute singleton error rate with CP bounds
    5. Marginalize over joint coverage distribution
    6. Use union bound for simultaneous PAC guarantees

    Parameters
    ----------
    ssbc_result_0 : SSBCResult
        SSBC result for class 0
    ssbc_result_1 : SSBCResult
        SSBC result for class 1
    labels : np.ndarray, shape (n,)
        True labels (0 or 1)
    probs : np.ndarray, shape (n, 2)
        Predicted probabilities
    test_size : int
        Expected test set size (for coverage distribution)
    ci_level : float, default=0.95
        Confidence level for Clopper-Pearson intervals
    pac_level : float, default=0.95
        PAC confidence level for final bounds
    use_union_bound : bool, default=True
        If True, applies Bonferroni correction for simultaneous guarantees
        on all 4 metrics (singleton, doublet, abstention, singleton_error).
        Recommended for operational deployment.
    n_jobs : int, default=1
        Number of parallel jobs for LOO-CV computation.
        1 = single-threaded, -1 = use all cores, N = use N cores.
        Parallelization provides significant speedup for large datasets.

    Returns
    -------
    dict
        PAC-controlled marginal bounds with keys:
        - 'singleton_rate_bounds': [L, U] where Pr(rate ∈ [L,U]) ≥ pac_level
        - 'doublet_rate_bounds': [L, U] (same guarantee)
        - 'abstention_rate_bounds': [L, U] (same guarantee)
        - 'singleton_error_rate_bounds': [L, U] for P(error | singleton)
        - 'expected_singleton_rate': probability-weighted mean
        - 'expected_doublet_rate': probability-weighted mean
        - 'expected_abstention_rate': probability-weighted mean
        - 'expected_singleton_error_rate': probability-weighted mean
        - 'n_grid_points': number of (α₀, α₁) points evaluated

    Notes
    -----
    **SSBC-Style One-Sided Guarantees:**

    Lower bound L: Pr(true_rate ≥ L) ≥ pac_level (or adjusted_pac_level with union bound)
    Upper bound U: Pr(true_rate ≤ U) ≥ pac_level (or adjusted_pac_level with union bound)

    If use_union_bound=True (default):
    - All 4 metrics' bounds hold SIMULTANEOUSLY with probability ≥ pac_level
    - Uses Bonferroni: adjusted_pac_level = 1 - (1-pac_level)/4 per metric
    """
    # Get coverage distributions for both classes
    cov_dist_0 = compute_coverage_distribution(ssbc_result_0, test_size)
    cov_dist_1 = compute_coverage_distribution(ssbc_result_1, test_size)

    # Convert to alpha values
    alpha_from_cov_0 = 1.0 - cov_dist_0["coverage_values"]
    alpha_from_cov_1 = 1.0 - cov_dist_1["coverage_values"]
    pmf_0 = cov_dist_0["pmf"]
    pmf_1 = cov_dist_1["pmf"]

    # Build joint grid (alpha_0, alpha_1) with joint probabilities
    # Assuming independence: P(alpha_0, alpha_1) = P(alpha_0) * P(alpha_1)

    # Storage for weighted results
    singleton_rates_weighted = []
    doublet_rates_weighted = []
    abstention_rates_weighted = []
    singleton_ci_lower_weighted = []
    singleton_ci_upper_weighted = []
    doublet_ci_lower_weighted = []
    doublet_ci_upper_weighted = []
    abstention_ci_lower_weighted = []
    abstention_ci_upper_weighted = []
    singleton_error_rates = []
    singleton_error_ci_lower = []
    singleton_error_ci_upper = []
    joint_weights = []

    # For computational efficiency, only iterate over high-probability region
    # Use alphas with non-negligible probability mass
    threshold = 1e-6
    idx_0_active = pmf_0 > threshold
    idx_1_active = pmf_1 > threshold

    alpha_0_active = alpha_from_cov_0[idx_0_active]
    alpha_1_active = alpha_from_cov_1[idx_1_active]
    pmf_0_active = pmf_0[idx_0_active]
    pmf_1_active = pmf_1[idx_1_active]

    # Build list of scenarios to evaluate  
    scenarios = []
    joint_probs = []
    
    for alpha_0, p_0 in zip(alpha_0_active, pmf_0_active, strict=False):
        for alpha_1, p_1 in zip(alpha_1_active, pmf_1_active, strict=False):
            joint_prob = p_0 * p_1
            
            if joint_prob < threshold:
                continue
            
            scenarios.append((alpha_0, alpha_1))
            joint_probs.append(joint_prob)
    
    # Parallel LOO-CV evaluation for each scenario
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_scenario_marginal)(alpha_0, alpha_1, labels, probs)
        for alpha_0, alpha_1 in scenarios
    )
    
    # Process results
    n = len(labels)
    for (alpha_0, alpha_1), joint_prob, (n_singletons, n_doublets, n_abstentions, n_singletons_correct) in zip(
        scenarios, joint_probs, results, strict=False
    ):
        # Compute rates
        singleton_rate = n_singletons / n
        doublet_rate = n_doublets / n
        abstention_rate = n_abstentions / n
        
        # Singleton error rate (conditioned on singleton)
        singleton_error_rate = (n_singletons - n_singletons_correct) / n_singletons if n_singletons > 0 else 0.0
        
        # Clopper-Pearson bounds on operational rates
        s_lower = clopper_pearson_lower(n_singletons, n, ci_level)
        s_upper = clopper_pearson_upper(n_singletons, n, ci_level)
        d_lower = clopper_pearson_lower(n_doublets, n, ci_level)
        d_upper = clopper_pearson_upper(n_doublets, n, ci_level)
        a_lower = clopper_pearson_lower(n_abstentions, n, ci_level)
        a_upper = clopper_pearson_upper(n_abstentions, n, ci_level)
        
        # CP bounds on singleton error rate (conditioned on singletons)
        n_singletons_incorrect = n_singletons - n_singletons_correct
        if n_singletons > 0:
            se_lower = clopper_pearson_lower(n_singletons_incorrect, n_singletons, ci_level)
            se_upper = clopper_pearson_upper(n_singletons_incorrect, n_singletons, ci_level)
        else:
            se_lower = 0.0
            se_upper = 1.0
        
        # Store with weight
        singleton_rates_weighted.append(singleton_rate)
        doublet_rates_weighted.append(doublet_rate)
        abstention_rates_weighted.append(abstention_rate)
        singleton_ci_lower_weighted.append(s_lower)
        singleton_ci_upper_weighted.append(s_upper)
        doublet_ci_lower_weighted.append(d_lower)
        doublet_ci_upper_weighted.append(d_upper)
        abstention_ci_lower_weighted.append(a_lower)
        abstention_ci_upper_weighted.append(a_upper)
        singleton_error_rates.append(singleton_error_rate)
        singleton_error_ci_lower.append(se_lower)
        singleton_error_ci_upper.append(se_upper)
        joint_weights.append(joint_prob)

    # Convert to arrays and normalize weights
    joint_weights = np.array(joint_weights)
    joint_weights = joint_weights / joint_weights.sum()

    singleton_rates_weighted = np.array(singleton_rates_weighted)
    doublet_rates_weighted = np.array(doublet_rates_weighted)
    abstention_rates_weighted = np.array(abstention_rates_weighted)
    singleton_ci_lower_weighted = np.array(singleton_ci_lower_weighted)
    singleton_ci_upper_weighted = np.array(singleton_ci_upper_weighted)
    doublet_ci_lower_weighted = np.array(doublet_ci_lower_weighted)
    doublet_ci_upper_weighted = np.array(doublet_ci_upper_weighted)
    abstention_ci_lower_weighted = np.array(abstention_ci_lower_weighted)
    abstention_ci_upper_weighted = np.array(abstention_ci_upper_weighted)
    singleton_error_rates = np.array(singleton_error_rates)
    singleton_error_ci_lower = np.array(singleton_error_ci_lower)
    singleton_error_ci_upper = np.array(singleton_error_ci_upper)

    # Weighted quantile function
    def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        """Compute weighted quantile."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum, q)
        idx = min(idx, len(sorted_values) - 1)
        return float(sorted_values[idx])

    # Compute PAC bounds using weighted quantiles
    # SSBC-style one-sided guarantees
    n_metrics = 4  # singleton, doublet, abstention, singleton_error

    if use_union_bound:
        # Bonferroni correction: each metric gets adjusted confidence level
        # Pr(all k hold) ≥ 1 - k*(1-pac_level_individual)
        # Set: 1 - k*(1-pac_level_individual) = pac_level
        # => pac_level_individual = 1 - (1-pac_level)/k
        adjusted_pac_level = 1 - (1 - pac_level) / n_metrics
    else:
        # No union bound: each metric guaranteed separately
        adjusted_pac_level = pac_level

    # Quantiles of rate distributions (weighted by scenario probability)
    # This uses the RATES themselves, not CI bounds - less conservative
    # Lower bound: Pr(rate >= L) >= pac_level  =>  L at (1-adjusted_pac_level) quantile
    # Upper bound: Pr(rate <= U) >= pac_level  =>  U at adjusted_pac_level quantile
    lower_quantile = 1 - adjusted_pac_level
    upper_quantile = adjusted_pac_level

    singleton_lower = weighted_quantile(singleton_rates_weighted, joint_weights, lower_quantile)
    singleton_upper = weighted_quantile(singleton_rates_weighted, joint_weights, upper_quantile)

    doublet_lower = weighted_quantile(doublet_rates_weighted, joint_weights, lower_quantile)
    doublet_upper = weighted_quantile(doublet_rates_weighted, joint_weights, upper_quantile)

    abstention_lower = weighted_quantile(abstention_rates_weighted, joint_weights, lower_quantile)
    abstention_upper = weighted_quantile(abstention_rates_weighted, joint_weights, upper_quantile)

    singleton_error_lower = weighted_quantile(singleton_error_rates, joint_weights, lower_quantile)
    singleton_error_upper = weighted_quantile(singleton_error_rates, joint_weights, upper_quantile)

    # Expected values (probability-weighted means)
    singleton_expected = np.sum(joint_weights * singleton_rates_weighted)
    doublet_expected = np.sum(joint_weights * doublet_rates_weighted)
    abstention_expected = np.sum(joint_weights * abstention_rates_weighted)
    singleton_error_expected = np.sum(joint_weights * singleton_error_rates)

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [singleton_error_lower, singleton_error_upper],
        "expected_singleton_rate": singleton_expected,
        "expected_doublet_rate": doublet_expected,
        "expected_abstention_rate": abstention_expected,
        "expected_singleton_error_rate": singleton_error_expected,
        "n_grid_points": len(joint_weights),
        "pac_level": pac_level,
        "ci_level": ci_level,
        "test_size": test_size,
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
    }


def _evaluate_loo_scenario_perclass(
    alpha_0: float,
    alpha_1: float,
    labels: np.ndarray,
    probs: np.ndarray,
    class_label: int,
) -> tuple[int, int, int, int]:
    """Evaluate LOO-CV for a single (alpha_0, alpha_1) scenario per-class (helper for parallelization).
    
    Returns
    -------
    tuple[int, int, int, int]
        (n_singletons, n_doublets, n_abstentions, n_singletons_correct)
    """
    # Filter to class_label samples
    mask = labels == class_label
    labels_class = labels[mask]
    probs_class = probs[mask]
    n_class = len(labels_class)
    
    n_abstentions = 0
    n_singletons = 0
    n_doublets = 0
    n_singletons_correct = 0
    
    # Get full masks (needed for LOO)
    mask_0_full = labels == 0
    mask_1_full = labels == 1
    
    # For each class_label sample, compute LOO thresholds and evaluate
    for local_idx in range(n_class):
        # Map to global index
        global_idx = np.where(mask)[0][local_idx]
        
        # Compute thresholds with LOO
        # Class 0 threshold
        if mask_0_full[global_idx]:
            scores_0_loo = 1.0 - probs[mask_0_full, 0]
            mask_0_indices = np.where(mask_0_full)[0]
            loo_idx_in_class0 = np.where(mask_0_indices == global_idx)[0][0]
            scores_0_loo = np.delete(scores_0_loo, loo_idx_in_class0)
            n_0_loo = len(scores_0_loo)
        else:
            scores_0_loo = 1.0 - probs[mask_0_full, 0]
            n_0_loo = len(scores_0_loo)
        
        # Class 1 threshold
        if mask_1_full[global_idx]:
            scores_1_loo = 1.0 - probs[mask_1_full, 1]
            mask_1_indices = np.where(mask_1_full)[0]
            loo_idx_in_class1 = np.where(mask_1_indices == global_idx)[0][0]
            scores_1_loo = np.delete(scores_1_loo, loo_idx_in_class1)
            n_1_loo = len(scores_1_loo)
        else:
            scores_1_loo = 1.0 - probs[mask_1_full, 1]
            n_1_loo = len(scores_1_loo)
        
        # Compute thresholds at (alpha_0, alpha_1)
        k_0_loo = int(np.ceil((n_0_loo + 1) * (1 - alpha_0)))
        k_0_loo = min(k_0_loo, n_0_loo)
        k_0_loo = max(k_0_loo, 1)
        
        k_1_loo = int(np.ceil((n_1_loo + 1) * (1 - alpha_1)))
        k_1_loo = min(k_1_loo, n_1_loo)
        k_1_loo = max(k_1_loo, 1)
        
        sorted_scores_0_loo = np.sort(scores_0_loo)
        sorted_scores_1_loo = np.sort(scores_1_loo)
        
        threshold_0_loo = sorted_scores_0_loo[k_0_loo - 1]
        threshold_1_loo = sorted_scores_1_loo[k_1_loo - 1]
        
        # Evaluate on held-out sample
        score_0 = 1.0 - probs_class[local_idx, 0]
        score_1 = 1.0 - probs_class[local_idx, 1]
        true_label = labels_class[local_idx]
        
        in_0 = score_0 <= threshold_0_loo
        in_1 = score_1 <= threshold_1_loo
        
        pred_set = []
        if in_0:
            pred_set.append(0)
        if in_1:
            pred_set.append(1)
        
        if len(pred_set) == 0:
            n_abstentions += 1
        elif len(pred_set) == 1:
            n_singletons += 1
            if true_label in pred_set:
                n_singletons_correct += 1
        else:
            n_doublets += 1
    
    return n_singletons, n_doublets, n_abstentions, n_singletons_correct


def compute_pac_operational_bounds_perclass(
    ssbc_result_0: SSBCResult,
    ssbc_result_1: SSBCResult,
    labels: np.ndarray,
    probs: np.ndarray,
    class_label: int,
    test_size: int,
    ci_level: float = 0.95,
    pac_level: float = 0.95,
    use_union_bound: bool = True,
    n_jobs: int = 1,
) -> dict[str, float | list]:
    """Compute per-class PAC-controlled operational bounds.

    Computes operational bounds conditioned on true_label = class_label,
    accounting for coverage volatility from both classes' thresholds.

    Parameters
    ----------
    ssbc_result_0 : SSBCResult
        SSBC result for class 0
    ssbc_result_1 : SSBCResult
        SSBC result for class 1
    labels : np.ndarray
        True labels
    probs : np.ndarray, shape (n, 2)
        Predicted probabilities
    class_label : int
        Which class to analyze (0 or 1)
    test_size : int
        Expected test set size
    ci_level : float, default=0.95
        Confidence level for CP intervals
    pac_level : float, default=0.95
        PAC confidence level
    use_union_bound : bool, default=True
        If True, applies Bonferroni correction for simultaneous guarantees.
        Recommended for operational deployment.
    n_jobs : int, default=1
        Number of parallel jobs for LOO-CV computation.
        1 = single-threaded, -1 = use all cores, N = use N cores.

    Returns
    -------
    dict
        Per-class PAC bounds with SSBC-style one-sided guarantees:
        - 'singleton_rate_bounds': [L, U] where Pr(rate ∈ [L,U]) ≥ pac_level
        - Similar for doublet, abstention, and singleton_error rates
        - Expected values (probability-weighted means)

    Notes
    -----
    **SSBC-Style One-Sided Guarantees:**

    Bounds hold with probability ≥ pac_level (or adjusted with union bound).
    Uses one-sided quantiles like SSBC for maximum rigor.
    """
    # Get coverage distributions
    cov_dist_0 = compute_coverage_distribution(ssbc_result_0, test_size)
    cov_dist_1 = compute_coverage_distribution(ssbc_result_1, test_size)

    alpha_from_cov_0 = 1.0 - cov_dist_0["coverage_values"]
    alpha_from_cov_1 = 1.0 - cov_dist_1["coverage_values"]
    pmf_0 = cov_dist_0["pmf"]
    pmf_1 = cov_dist_1["pmf"]

    # Storage
    singleton_rates = []
    doublet_rates = []
    abstention_rates = []
    singleton_ci_lower_list = []
    singleton_ci_upper_list = []
    doublet_ci_lower_list = []
    doublet_ci_upper_list = []
    abstention_ci_lower_list = []
    abstention_ci_upper_list = []
    singleton_error_rates = []
    singleton_error_ci_lower = []
    singleton_error_ci_upper = []
    joint_weights = []

    # Filter to class_label samples
    mask = labels == class_label
    labels_class = labels[mask]
    probs_class = probs[mask]
    n_class = len(labels_class)

    # Only iterate over high-probability region
    threshold = 1e-6
    idx_0_active = pmf_0 > threshold
    idx_1_active = pmf_1 > threshold

    alpha_0_active = alpha_from_cov_0[idx_0_active]
    alpha_1_active = alpha_from_cov_1[idx_1_active]
    pmf_0_active = pmf_0[idx_0_active]
    pmf_1_active = pmf_1[idx_1_active]
    
    # Build list of scenarios to evaluate
    scenarios = []
    joint_probs = []
    
    for alpha_0, p_0 in zip(alpha_0_active, pmf_0_active, strict=False):
        for alpha_1, p_1 in zip(alpha_1_active, pmf_1_active, strict=False):
            joint_prob = p_0 * p_1
            
            if joint_prob < threshold:
                continue
            
            scenarios.append((alpha_0, alpha_1))
            joint_probs.append(joint_prob)
    
    # Parallel LOO-CV evaluation for each scenario
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_loo_scenario_perclass)(alpha_0, alpha_1, labels, probs, class_label)
        for alpha_0, alpha_1 in scenarios
    )
    
    # Process results
    for (alpha_0, alpha_1), joint_prob, (n_singletons, n_doublets, n_abstentions, n_singletons_correct) in zip(
        scenarios, joint_probs, results, strict=False
    ):
        # Rates
        singleton_rate = n_singletons / n_class
        doublet_rate = n_doublets / n_class
        abstention_rate = n_abstentions / n_class
        singleton_error_rate = (n_singletons - n_singletons_correct) / n_singletons if n_singletons > 0 else 0.0
        
        # Clopper-Pearson bounds
        s_lower = clopper_pearson_lower(n_singletons, n_class, ci_level)
        s_upper = clopper_pearson_upper(n_singletons, n_class, ci_level)
        d_lower = clopper_pearson_lower(n_doublets, n_class, ci_level)
        d_upper = clopper_pearson_upper(n_doublets, n_class, ci_level)
        a_lower = clopper_pearson_lower(n_abstentions, n_class, ci_level)
        a_upper = clopper_pearson_upper(n_abstentions, n_class, ci_level)
        
        n_singletons_incorrect = n_singletons - n_singletons_correct
        if n_singletons > 0:
            se_lower = clopper_pearson_lower(n_singletons_incorrect, n_singletons, ci_level)
            se_upper = clopper_pearson_upper(n_singletons_incorrect, n_singletons, ci_level)
        else:
            se_lower = 0.0
            se_upper = 1.0
        
        # Store
        singleton_rates.append(singleton_rate)
        doublet_rates.append(doublet_rate)
        abstention_rates.append(abstention_rate)
        singleton_ci_lower_list.append(s_lower)
        singleton_ci_upper_list.append(s_upper)
        doublet_ci_lower_list.append(d_lower)
        doublet_ci_upper_list.append(d_upper)
        abstention_ci_lower_list.append(a_lower)
        abstention_ci_upper_list.append(a_upper)
        singleton_error_rates.append(singleton_error_rate)
        singleton_error_ci_lower.append(se_lower)
        singleton_error_ci_upper.append(se_upper)
        joint_weights.append(joint_prob)

    # Convert and normalize
    joint_weights = np.array(joint_weights)
    joint_weights = joint_weights / joint_weights.sum()

    singleton_rates = np.array(singleton_rates)
    doublet_rates = np.array(doublet_rates)
    abstention_rates = np.array(abstention_rates)
    singleton_ci_lower_arr = np.array(singleton_ci_lower_list)
    singleton_ci_upper_arr = np.array(singleton_ci_upper_list)
    doublet_ci_lower_arr = np.array(doublet_ci_lower_list)
    doublet_ci_upper_arr = np.array(doublet_ci_upper_list)
    abstention_ci_lower_arr = np.array(abstention_ci_lower_list)
    abstention_ci_upper_arr = np.array(abstention_ci_upper_list)
    singleton_error_rates = np.array(singleton_error_rates)
    singleton_error_ci_lower = np.array(singleton_error_ci_lower)
    singleton_error_ci_upper = np.array(singleton_error_ci_upper)

    # Weighted quantile
    def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum, q)
        idx = min(idx, len(sorted_values) - 1)
        return float(sorted_values[idx])

    # PAC bounds with union bound correction
    # SSBC-style one-sided guarantees
    n_metrics = 4  # singleton, doublet, abstention, singleton_error

    if use_union_bound:
        # Bonferroni correction for simultaneous guarantees
        adjusted_pac_level = 1 - (1 - pac_level) / n_metrics
    else:
        adjusted_pac_level = pac_level

    # Quantiles of rate distributions (weighted by scenario probability)
    # This uses the RATES themselves, not CI bounds - less conservative
    # Lower bound: Pr(rate >= L) >= adjusted_pac_level
    # Upper bound: Pr(rate <= U) >= adjusted_pac_level
    lower_quantile = 1 - adjusted_pac_level
    upper_quantile = adjusted_pac_level

    singleton_lower = weighted_quantile(singleton_rates, joint_weights, lower_quantile)
    singleton_upper = weighted_quantile(singleton_rates, joint_weights, upper_quantile)
    doublet_lower = weighted_quantile(doublet_rates, joint_weights, lower_quantile)
    doublet_upper = weighted_quantile(doublet_rates, joint_weights, upper_quantile)
    abstention_lower = weighted_quantile(abstention_rates, joint_weights, lower_quantile)
    abstention_upper = weighted_quantile(abstention_rates, joint_weights, upper_quantile)
    singleton_error_lower = weighted_quantile(singleton_error_rates, joint_weights, lower_quantile)
    singleton_error_upper = weighted_quantile(singleton_error_rates, joint_weights, upper_quantile)

    # Expected values
    singleton_expected = np.sum(joint_weights * singleton_rates)
    doublet_expected = np.sum(joint_weights * doublet_rates)
    abstention_expected = np.sum(joint_weights * abstention_rates)
    singleton_error_expected = np.sum(joint_weights * singleton_error_rates)

    return {
        "singleton_rate_bounds": [singleton_lower, singleton_upper],
        "doublet_rate_bounds": [doublet_lower, doublet_upper],
        "abstention_rate_bounds": [abstention_lower, abstention_upper],
        "singleton_error_rate_bounds": [singleton_error_lower, singleton_error_upper],
        "expected_singleton_rate": singleton_expected,
        "expected_doublet_rate": doublet_expected,
        "expected_abstention_rate": abstention_expected,
        "expected_singleton_error_rate": singleton_error_expected,
        "n_grid_points": len(joint_weights),
        "n_class": n_class,
        "class_label": class_label,
        "pac_level": pac_level,
        "ci_level": ci_level,
        "test_size": test_size,
        "use_union_bound": use_union_bound,
        "n_metrics": n_metrics if use_union_bound else None,
    }


def compute_pac_operational_bounds(
    ssbc_result: SSBCResult,
    response_curve: dict,
    test_size: int,
    pac_level: float = 0.95,
) -> dict[str, float]:
    """Compute PAC-controlled operational bounds by weighting response curve.

    Combines:
    1. Coverage distribution from SSBC (uncertainty in which alpha is realized)
    2. Response curve with CP bounds (uncertainty in operational rates at each alpha)

    Weights the CP bounds by probability of achieving each alpha level.

    Parameters
    ----------
    ssbc_result : SSBCResult
        Result from ssbc_correct() - determines coverage distribution
    response_curve : dict
        Output from compute_mondrian_response_curve() or compute_operational_response_curve()
    test_size : int
        Expected test set size (for coverage distribution)
    pac_level : float, default=0.95
        PAC confidence level (e.g., 0.95 means bounds hold with 95% probability)

    Returns
    -------
    dict
        Dictionary with PAC-controlled bounds:
        - 'singleton_lower': Lower bound on singleton rate
        - 'singleton_upper': Upper bound on singleton rate
        - 'doublet_lower': Lower bound on doublet rate
        - 'doublet_upper': Upper bound on doublet rate
        - 'abstention_lower': Lower bound on abstention rate
        - 'abstention_upper': Upper bound on abstention rate
        - 'singleton_expected': Expected singleton rate (weighted mean)
        - 'doublet_expected': Expected doublet rate
        - 'abstention_expected': Expected abstention rate
        - 'weights': Probability weights used
        - 'pac_level': Confidence level used

    Examples
    --------
    >>> from ssbc import ssbc_correct, BinaryClassifierSimulator
    >>> from ssbc.coverage_distribution import (
    ...     compute_coverage_distribution,
    ...     compute_mondrian_response_curve,
    ...     compute_pac_operational_bounds
    ... )
    >>> sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
    >>> labels, probs = sim.generate(n_samples=200)
    >>>
    >>> # Step 1: SSBC
    >>> ssbc_result = ssbc_correct(alpha_target=0.10, n=100, delta=0.10)
    >>>
    >>> # Step 2: Response curve
    >>> curve = compute_mondrian_response_curve(labels, probs)
    >>>
    >>> # Step 3: PAC bounds
    >>> bounds = compute_pac_operational_bounds(ssbc_result, curve, test_size=100)
    >>> print(f"Singleton rate: [{bounds['singleton_lower']:.3f}, {bounds['singleton_upper']:.3f}]")

    Notes
    -----
    **Two-Layer Uncertainty:**

    1. Coverage uncertainty: Which alpha will be realized on test set?
       → BetaBinomial distribution based on SSBC calibration

    2. Operational uncertainty: What are operational rates at that alpha?
       → Clopper-Pearson bounds from calibration data

    **PAC Guarantee:**

    The bounds hold with probability ≥ pac_level over:
    - Random test sets (coverage distribution)
    - Binomial sampling (CP intervals)

    This avoids "accepting risk by fiat" by properly accounting for nested uncertainty.
    """
    # Get coverage distribution
    cov_dist = compute_coverage_distribution(ssbc_result, test_size)

    # The coverage distribution gives us probabilities over coverage values
    # We need to map these to alpha values
    # Coverage = 1 - alpha, so alpha = 1 - coverage
    alpha_from_coverage = 1.0 - cov_dist["coverage_values"]
    weights_from_coverage = cov_dist["pmf"]

    # Match response curve alphas to coverage distribution alphas
    # For each alpha in response curve, find corresponding weight
    response_alphas = response_curve["alpha_values"]
    weights = np.zeros(len(response_alphas))

    for i, alpha_resp in enumerate(response_alphas):
        # Find closest alpha in coverage distribution
        idx = np.argmin(np.abs(alpha_from_coverage - alpha_resp))
        weights[i] = weights_from_coverage[idx]

    # Normalize weights (in case of rounding/matching issues)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        # Fallback: uniform weights
        weights = np.ones(len(response_alphas)) / len(response_alphas)

    # Helper function for weighted quantile
    def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        """Compute weighted quantile."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum, q)
        idx = min(idx, len(sorted_values) - 1)
        return float(sorted_values[idx])

    # Compute PAC bounds using weighted quantiles
    # Lower bound: (1-pac_level)/2 quantile of lower CP bounds
    # Upper bound: 1-(1-pac_level)/2 quantile of upper CP bounds
    tail_prob = (1 - pac_level) / 2

    singleton_lower = weighted_quantile(response_curve["singleton_ci_lower"], weights, tail_prob)
    singleton_upper = weighted_quantile(response_curve["singleton_ci_upper"], weights, 1 - tail_prob)

    doublet_lower = weighted_quantile(response_curve["doublet_ci_lower"], weights, tail_prob)
    doublet_upper = weighted_quantile(response_curve["doublet_ci_upper"], weights, 1 - tail_prob)

    abstention_lower = weighted_quantile(response_curve["abstention_ci_lower"], weights, tail_prob)
    abstention_upper = weighted_quantile(response_curve["abstention_ci_upper"], weights, 1 - tail_prob)

    # Compute expected values (probability-weighted means)
    singleton_expected = np.sum(weights * response_curve["singleton_rate"])
    doublet_expected = np.sum(weights * response_curve["doublet_rate"])
    abstention_expected = np.sum(weights * response_curve["abstention_rate"])

    return {
        "singleton_lower": singleton_lower,
        "singleton_upper": singleton_upper,
        "doublet_lower": doublet_lower,
        "doublet_upper": doublet_upper,
        "abstention_lower": abstention_lower,
        "abstention_upper": abstention_upper,
        "singleton_expected": singleton_expected,
        "doublet_expected": doublet_expected,
        "abstention_expected": abstention_expected,
        "weights": weights.tolist(),
        "pac_level": pac_level,
        "test_size": test_size,
        "n_calibration": ssbc_result.n,
        "alpha_adj": ssbc_result.alpha_corrected,
    }
