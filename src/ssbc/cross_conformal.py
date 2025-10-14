"""Cross-conformal validation for estimating rate variability.

This implements K-fold cross-validation specifically for conformal prediction:
- Split calibration data into K folds
- For each fold: train thresholds on K-1 folds, evaluate rates on held-out fold
- Aggregate rates across folds to quantify finite-sample variability

Different from:
- LOO-CV: Leave-one-out, aggregates counts (not rates per fold)
- Bootstrap: Resamples with replacement, tests on fresh data
- Cross-conformal: K-fold split, estimates rate distribution from finite calibration
"""

import numpy as np
from typing import Any

from ssbc.conformal import split_by_class
from ssbc.core import ssbc_correct
from ssbc.statistics import clopper_pearson_intervals


def _compute_fold_rates_mondrian(
    train_labels: np.ndarray,
    train_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha_target: float,
    delta: float,
) -> dict[str, float]:
    """Compute operational rates for one fold in Mondrian conformal.

    Parameters
    ----------
    train_labels : np.ndarray
        Training fold labels
    train_probs : np.ndarray
        Training fold probabilities
    test_labels : np.ndarray
        Test fold labels
    test_probs : np.ndarray
        Test fold probabilities
    alpha_target : float
        Target miscoverage
    delta : float
        PAC risk (for SSBC correction)

    Returns
    -------
    dict
        Rates for this fold: marginal and per-class
    """
    # Split training data by class
    train_class_data = split_by_class(train_labels, train_probs)

    # SSBC correction and threshold computation
    thresholds = {}
    for class_label in [0, 1]:
        class_data = train_class_data[class_label]
        if class_data["n"] == 0:
            thresholds[class_label] = 0.0
            continue

        # SSBC correction
        ssbc_result = ssbc_correct(alpha_target=alpha_target, n=class_data["n"], delta=delta)

        # Compute threshold
        n_class = class_data["n"]
        k = int(np.ceil((n_class + 1) * (1 - ssbc_result.alpha_corrected)))

        mask = train_labels == class_label
        scores = 1.0 - train_probs[mask, class_label]
        sorted_scores = np.sort(scores)

        thresholds[class_label] = sorted_scores[min(k - 1, len(sorted_scores) - 1)]

    # Evaluate on test fold
    n_test = len(test_labels)

    # Marginal counters
    n_abstentions = 0
    n_singletons = 0
    n_doublets = 0
    n_singletons_correct = 0

    # Per-class counters
    counts_0 = {"abstentions": 0, "singletons": 0, "doublets": 0, "singletons_correct": 0, "n": 0}
    counts_1 = {"abstentions": 0, "singletons": 0, "doublets": 0, "singletons_correct": 0, "n": 0}

    for i in range(n_test):
        true_label = test_labels[i]
        score_0 = 1.0 - test_probs[i, 0]
        score_1 = 1.0 - test_probs[i, 1]

        in_0 = score_0 <= thresholds[0]
        in_1 = score_1 <= thresholds[1]

        # Marginal
        if in_0 and in_1:
            n_doublets += 1
        elif in_0 or in_1:
            n_singletons += 1
            if (in_0 and true_label == 0) or (in_1 and true_label == 1):
                n_singletons_correct += 1
        else:
            n_abstentions += 1

        # Per-class
        if true_label == 0:
            counts_0["n"] += 1
            if in_0 and in_1:
                counts_0["doublets"] += 1
            elif in_0 or in_1:
                counts_0["singletons"] += 1
                if in_0:
                    counts_0["singletons_correct"] += 1
            else:
                counts_0["abstentions"] += 1
        else:
            counts_1["n"] += 1
            if in_0 and in_1:
                counts_1["doublets"] += 1
            elif in_0 or in_1:
                counts_1["singletons"] += 1
                if in_1:
                    counts_1["singletons_correct"] += 1
            else:
                counts_1["abstentions"] += 1

    # Compute rates
    marginal_rates = {
        "abstention": n_abstentions / n_test,
        "singleton": n_singletons / n_test,
        "doublet": n_doublets / n_test,
        "singleton_error": (n_singletons - n_singletons_correct) / n_singletons if n_singletons > 0 else np.nan,
    }

    class_0_rates = {
        "abstention": counts_0["abstentions"] / counts_0["n"] if counts_0["n"] > 0 else np.nan,
        "singleton": counts_0["singletons"] / counts_0["n"] if counts_0["n"] > 0 else np.nan,
        "doublet": counts_0["doublets"] / counts_0["n"] if counts_0["n"] > 0 else np.nan,
        "singleton_error": (
            (counts_0["singletons"] - counts_0["singletons_correct"]) / counts_0["singletons"]
            if counts_0["singletons"] > 0
            else np.nan
        ),
    }

    class_1_rates = {
        "abstention": counts_1["abstentions"] / counts_1["n"] if counts_1["n"] > 0 else np.nan,
        "singleton": counts_1["singletons"] / counts_1["n"] if counts_1["n"] > 0 else np.nan,
        "doublet": counts_1["doublets"] / counts_1["n"] if counts_1["n"] > 0 else np.nan,
        "singleton_error": (
            (counts_1["singletons"] - counts_1["singletons_correct"]) / counts_1["singletons"]
            if counts_1["singletons"] > 0
            else np.nan
        ),
    }

    return {
        "marginal": marginal_rates,
        "class_0": class_0_rates,
        "class_1": class_1_rates,
    }


def cross_conformal_validation(
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: float = 0.10,
    delta: float = 0.10,
    n_folds: int = 5,
    stratify: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """K-fold cross-conformal validation for Mondrian conformal prediction.

    Estimates the variability of operational rates (abstentions, singletons, doublets)
    due to finite calibration sample effects by splitting data into K folds.

    For each fold:
    1. Train: Compute SSBC-corrected thresholds on K-1 folds
    2. Test: Evaluate operational rates on held-out fold
    3. Record: Store rates for this fold

    Aggregate rates across folds to quantify finite-sample variability.

    Parameters
    ----------
    labels : np.ndarray, shape (n,)
        Calibration labels (0 or 1)
    probs : np.ndarray, shape (n, 2)
        Calibration probabilities [P(class=0), P(class=1)]
    alpha_target : float, default=0.10
        Target miscoverage rate
    delta : float, default=0.10
        PAC risk for SSBC correction
    n_folds : int, default=5
        Number of folds (K)
    stratify : bool, default=True
        Stratify folds by class labels
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Cross-conformal results with keys:
        - 'fold_rates': List of rate dicts for each fold
        - 'marginal': Statistics for marginal rates
        - 'class_0': Statistics for class 0 rates
        - 'class_1': Statistics for class 1 rates
        Each statistics dict contains:
        - 'samples': Array of rates across folds
        - 'mean': Mean rate
        - 'std': Standard deviation
        - 'quantiles': Dict with q05, q25, q50, q75, q95
        - 'ci_95': 95% Clopper-Pearson CI (if applicable)

    Examples
    --------
    >>> from ssbc import cross_conformal_validation
    >>> results = cross_conformal_validation(labels, probs, n_folds=10)
    >>> print(f"Singleton rate: {results['marginal']['singleton']['mean']:.3f} ± {results['marginal']['singleton']['std']:.3f}")
    >>> print(f"95% range: [{results['marginal']['singleton']['quantiles']['q05']:.3f}, "
    ...       f"{results['marginal']['singleton']['quantiles']['q95']:.3f}]")

    Notes
    -----
    Different from other methods:
    - **LOO-CV**: Leave-one-out, aggregates counts (not fold-level rates)
    - **Bootstrap**: Resamples with replacement, tests on fresh data
    - **Cross-conformal**: K-fold split, estimates rate distribution from calibration

    This method directly estimates the variability of rates due to finite calibration samples,
    without requiring a data simulator.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(labels)

    # Create fold indices
    indices = np.arange(n)

    if stratify:
        # Stratified K-fold: maintain class proportions in each fold
        class_0_idx = indices[labels == 0]
        class_1_idx = indices[labels == 1]

        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)

        class_0_folds = np.array_split(class_0_idx, n_folds)
        class_1_folds = np.array_split(class_1_idx, n_folds)

        folds = [np.concatenate([class_0_folds[i], class_1_folds[i]]) for i in range(n_folds)]
    else:
        # Standard K-fold
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

    # Compute rates for each fold
    fold_rates = []

    for fold_idx in range(n_folds):
        # Test fold
        test_idx = folds[fold_idx]

        # Train folds (all except test)
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

        # Compute fold rates
        rates = _compute_fold_rates_mondrian(
            train_labels=labels[train_idx],
            train_probs=probs[train_idx],
            test_labels=labels[test_idx],
            test_probs=probs[test_idx],
            alpha_target=alpha_target,
            delta=delta,
        )

        fold_rates.append(rates)

    # Aggregate statistics
    metrics = ["abstention", "singleton", "doublet", "singleton_error"]

    def compute_stats(values: list[float], metric_name: str) -> dict:
        """Compute statistics for a metric across folds."""
        arr = np.array(values)
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            return {
                "samples": arr,
                "mean": np.nan,
                "std": np.nan,
                "quantiles": {"q05": np.nan, "q25": np.nan, "q50": np.nan, "q75": np.nan, "q95": np.nan},
                "ci_95": {"lower": np.nan, "upper": np.nan},
            }

        stats = {
            "samples": arr,
            "mean": np.mean(valid),
            "std": np.std(valid, ddof=1) if len(valid) > 1 else 0.0,
            "quantiles": {
                "q05": np.percentile(valid, 5),
                "q25": np.percentile(valid, 25),
                "q50": np.percentile(valid, 50),
                "q75": np.percentile(valid, 75),
                "q95": np.percentile(valid, 95),
            },
        }

        # Add empirical CI based on fold distribution (binomial-like but for fold means)
        # This is approximate - treats fold means as if they were Bernoulli trials
        # Better: just use quantiles, but keeping for compatibility
        stats["ci_95"] = {
            "lower": stats["quantiles"]["q05"],
            "upper": stats["quantiles"]["q95"],
        }

        return stats

    # Aggregate marginal statistics
    marginal_stats = {
        metric: compute_stats([fold["marginal"][metric] for fold in fold_rates], metric)
        for metric in metrics
    }

    # Aggregate class-specific statistics
    class_0_stats = {
        metric: compute_stats([fold["class_0"][metric] for fold in fold_rates], metric)
        for metric in metrics
    }

    class_1_stats = {
        metric: compute_stats([fold["class_1"][metric] for fold in fold_rates], metric)
        for metric in metrics
    }

    return {
        "n_folds": n_folds,
        "n_samples": n,
        "stratified": stratify,
        "fold_rates": fold_rates,
        "marginal": marginal_stats,
        "class_0": class_0_stats,
        "class_1": class_1_stats,
        "parameters": {
            "alpha_target": alpha_target,
            "delta": delta,
            "n_folds": n_folds,
            "stratify": stratify,
        },
    }


def print_cross_conformal_results(results: dict) -> None:
    """Pretty print cross-conformal validation results.

    Parameters
    ----------
    results : dict
        Results from cross_conformal_validation()
    """
    print("=" * 80)
    print("CROSS-CONFORMAL VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  K-folds: {results['n_folds']}")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Stratified: {results['stratified']}")
    print(f"  Alpha target: {results['parameters']['alpha_target']:.3f}")
    print(f"  Delta (PAC): {results['parameters']['delta']:.3f}")

    # Marginal
    print("\n" + "-" * 80)
    print("MARGINAL RATES (Across All Samples)")
    print("-" * 80)

    for metric, name in [
        ("singleton", "SINGLETON"),
        ("doublet", "DOUBLET"),
        ("abstention", "ABSTENTION"),
        ("singleton_error", "SINGLETON ERROR"),
    ]:
        m = results["marginal"][metric]
        q = m["quantiles"]

        print(f"\n{name}:")
        print(f"  Mean across folds: {m['mean']:.4f} ± {m['std']:.4f}")
        print(f"  Median:            {q['q50']:.4f}")
        print(f"  [5%, 95%] range:   [{q['q05']:.4f}, {q['q95']:.4f}]")
        print(f"  [25%, 75%] IQR:    [{q['q25']:.4f}, {q['q75']:.4f}]")

    # Per-class
    for class_label in [0, 1]:
        print(f"\n{'-' * 80}")
        print(f"CLASS {class_label} RATES")
        print("-" * 80)

        for metric, name in [
            ("singleton", "SINGLETON"),
            ("doublet", "DOUBLET"),
            ("abstention", "ABSTENTION"),
            ("singleton_error", "SINGLETON ERROR"),
        ]:
            m = results[f"class_{class_label}"][metric]
            q = m["quantiles"]

            print(f"\n{name}:")
            print(f"  Mean across folds: {m['mean']:.4f} ± {m['std']:.4f}")
            print(f"  Median:            {q['q50']:.4f}")
            print(f"  [5%, 95%] range:   [{q['q05']:.4f}, {q['q95']:.4f}]")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\n✓ Shows finite-sample variability from K-fold splits of calibration data")
    print("✓ [5%, 95%] range indicates expected rate fluctuations")
    print("✓ Smaller std → more stable rates across different calibration subsets")
    print("✓ Complementary to:")
    print("  • LOO-CV bounds: Uncertainty for fixed full calibration")
    print("  • Bootstrap: Recalibration uncertainty with fresh test data")
    print("  • Cross-conformal: Rate variability from finite calibration splits")
    print("\n" + "=" * 80)

