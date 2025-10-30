"""STANDALONE Bootstrap analysis demo with plotting and detailed analysis.

This is a comprehensive standalone example showing bootstrap calibration uncertainty
with visualizations. Use bootstrap_calibration_uncertainty() from ssbc.calibration
for standalone bootstrap analysis.

This models: "If I recalibrate many times on similar datasets, how do rates vary?"
Different from LOO-CV which models: "Given ONE fixed calibration, how do test sets vary?"

Note: Bootstrap is available as a standalone function, not integrated into
generate_rigorous_pac_report(). The rigorous report focuses on LOO-CV based
PAC bounds with prediction interval uncertainty quantification.
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from ssbc import BinaryClassifierSimulator, split_by_class, ssbc_correct


def _bootstrap_single_trial(
    labels: np.ndarray,
    probs: np.ndarray,
    alpha_target: float,
    delta: float,
    test_size: int,
    bootstrap_seed: int,
    simulator: BinaryClassifierSimulator,
) -> dict[str, float]:
    """Single bootstrap trial: resample calibration → calibrate → evaluate on fresh test set.

    Returns
    -------
    dict
        Operational rates for this bootstrap sample
    """
    np.random.seed(bootstrap_seed)

    n = len(labels)

    # Bootstrap resample calibration data (with replacement)
    bootstrap_idx = np.random.choice(n, size=n, replace=True)
    labels_boot = labels[bootstrap_idx]
    probs_boot = probs[bootstrap_idx]

    # Split by class
    class_data_boot = split_by_class(labels_boot, probs_boot)

    # Calibrate on bootstrap sample
    try:
        ssbc_0 = ssbc_correct(alpha_target=alpha_target, n=class_data_boot[0]["n"], delta=delta)
        ssbc_1 = ssbc_correct(alpha_target=alpha_target, n=class_data_boot[1]["n"], delta=delta)
    except Exception:
        # Handle edge cases (e.g., all samples from one class)
        return {
            "singleton": np.nan,
            "doublet": np.nan,
            "abstention": np.nan,
            "singleton_error": np.nan,
            "singleton_0": np.nan,
            "doublet_0": np.nan,
            "abstention_0": np.nan,
            "singleton_error_0": np.nan,
            "singleton_1": np.nan,
            "doublet_1": np.nan,
            "abstention_1": np.nan,
            "singleton_error_1": np.nan,
        }

    # Compute thresholds
    n_0 = class_data_boot[0]["n"]
    n_1 = class_data_boot[1]["n"]

    k_0 = int(np.ceil((n_0 + 1) * (1 - ssbc_0.alpha_corrected)))
    k_1 = int(np.ceil((n_1 + 1) * (1 - ssbc_1.alpha_corrected)))

    mask_0 = labels_boot == 0
    mask_1 = labels_boot == 1

    scores_0 = 1.0 - probs_boot[mask_0, 0]
    scores_1 = 1.0 - probs_boot[mask_1, 1]

    sorted_0 = np.sort(scores_0)
    sorted_1 = np.sort(scores_1)

    threshold_0 = sorted_0[min(k_0 - 1, len(sorted_0) - 1)]
    threshold_1 = sorted_1[min(k_1 - 1, len(sorted_1) - 1)]

    # Generate FRESH test set
    labels_test, probs_test = simulator.generate(test_size)

    # Evaluate on test set
    n_test = len(labels_test)
    n_singletons = 0
    n_doublets = 0
    n_abstentions = 0
    n_singletons_correct = 0

    # Per-class counters
    n_singletons_0 = 0
    n_doublets_0 = 0
    n_abstentions_0 = 0
    n_singletons_correct_0 = 0
    n_class_0 = 0

    n_singletons_1 = 0
    n_doublets_1 = 0
    n_abstentions_1 = 0
    n_singletons_correct_1 = 0
    n_class_1 = 0

    for i in range(n_test):
        true_label = labels_test[i]
        score_0 = 1.0 - probs_test[i, 0]
        score_1 = 1.0 - probs_test[i, 1]

        in_0 = score_0 <= threshold_0
        in_1 = score_1 <= threshold_1

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
            n_class_0 += 1
            if in_0 and in_1:
                n_doublets_0 += 1
            elif in_0 or in_1:
                n_singletons_0 += 1
                if in_0:
                    n_singletons_correct_0 += 1
            else:
                n_abstentions_0 += 1
        else:
            n_class_1 += 1
            if in_0 and in_1:
                n_doublets_1 += 1
            elif in_0 or in_1:
                n_singletons_1 += 1
                if in_1:
                    n_singletons_correct_1 += 1
            else:
                n_abstentions_1 += 1

    # Compute rates
    singleton_rate = n_singletons / n_test
    doublet_rate = n_doublets / n_test
    abstention_rate = n_abstentions / n_test
    singleton_error_rate = (n_singletons - n_singletons_correct) / n_singletons if n_singletons > 0 else np.nan

    # Per-class rates
    singleton_0 = n_singletons_0 / n_class_0 if n_class_0 > 0 else np.nan
    doublet_0 = n_doublets_0 / n_class_0 if n_class_0 > 0 else np.nan
    abstention_0 = n_abstentions_0 / n_class_0 if n_class_0 > 0 else np.nan
    singleton_error_0 = (n_singletons_0 - n_singletons_correct_0) / n_singletons_0 if n_singletons_0 > 0 else np.nan

    singleton_1 = n_singletons_1 / n_class_1 if n_class_1 > 0 else np.nan
    doublet_1 = n_doublets_1 / n_class_1 if n_class_1 > 0 else np.nan
    abstention_1 = n_abstentions_1 / n_class_1 if n_class_1 > 0 else np.nan
    singleton_error_1 = (n_singletons_1 - n_singletons_correct_1) / n_singletons_1 if n_singletons_1 > 0 else np.nan

    return {
        "singleton": singleton_rate,
        "doublet": doublet_rate,
        "abstention": abstention_rate,
        "singleton_error": singleton_error_rate,
        "singleton_0": singleton_0,
        "doublet_0": doublet_0,
        "abstention_0": abstention_0,
        "singleton_error_0": singleton_error_0,
        "singleton_1": singleton_1,
        "doublet_1": doublet_1,
        "abstention_1": abstention_1,
        "singleton_error_1": singleton_error_1,
    }


def bootstrap_calibration_uncertainty(
    labels: np.ndarray,
    probs: np.ndarray,
    simulator: BinaryClassifierSimulator,
    alpha_target: float = 0.10,
    delta: float = 0.10,
    test_size: int = 1000,
    n_bootstrap: int = 1000,
    n_jobs: int = -1,
    seed: int | None = None,
) -> dict:
    """Bootstrap analysis of calibration uncertainty.

    For each bootstrap iteration:
    1. Resample calibration data with replacement
    2. Calibrate (compute SSBC thresholds)
    3. Evaluate on fresh independent test set
    4. Record operational rates

    This models: "If I recalibrate on similar datasets, how do rates vary?"

    Parameters
    ----------
    labels : np.ndarray
        Calibration labels
    probs : np.ndarray
        Calibration probabilities
    simulator : BinaryClassifierSimulator
        Simulator to generate independent test sets
    alpha_target : float, default=0.10
        Target miscoverage
    delta : float, default=0.10
        PAC risk
    test_size : int, default=1000
        Size of test sets for evaluation
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    n_jobs : int, default=-1
        Parallel jobs
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Bootstrap distributions with keys:
        - 'marginal': dict with 'singleton', 'doublet', 'abstention', 'singleton_error'
        - 'class_0': dict with same metrics
        - 'class_1': dict with same metrics
        Each metric contains:
        - 'samples': array of rates across bootstrap trials
        - 'mean': mean rate
        - 'quantiles': dict with q05, q25, q50, q75, q95
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate bootstrap seeds
    bootstrap_seeds = np.random.randint(0, 2**31, size=n_bootstrap)

    print(f"Running {n_bootstrap} bootstrap trials...")
    print(f"  Calibration size: n={len(labels)}")
    print(f"  Test size per trial: {test_size}")

    # Parallel bootstrap
    results = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_single_trial)(labels, probs, alpha_target, delta, test_size, bs_seed, simulator)
        for bs_seed in bootstrap_seeds
    )

    # Extract metrics
    metrics = ["singleton", "doublet", "abstention", "singleton_error"]

    def compute_stats(values):
        """Compute statistics for a metric."""
        arr = np.array(values)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return {
                "samples": arr,
                "mean": np.nan,
                "std": np.nan,
                "quantiles": {
                    "q05": np.nan,
                    "q25": np.nan,
                    "q50": np.nan,
                    "q75": np.nan,
                    "q95": np.nan,
                },
            }
        return {
            "samples": arr,
            "mean": np.mean(valid),
            "std": np.std(valid),
            "quantiles": {
                "q05": np.percentile(valid, 5),
                "q25": np.percentile(valid, 25),
                "q50": np.percentile(valid, 50),
                "q75": np.percentile(valid, 75),
                "q95": np.percentile(valid, 95),
            },
        }

    # Organize results
    return {
        "n_bootstrap": n_bootstrap,
        "n_calibration": len(labels),
        "test_size": test_size,
        "marginal": {metric: compute_stats([r[metric] for r in results]) for metric in metrics},
        "class_0": {metric: compute_stats([r[f"{metric}_0"] for r in results]) for metric in metrics},
        "class_1": {metric: compute_stats([r[f"{metric}_1"] for r in results]) for metric in metrics},
    }


def print_bootstrap_results(bootstrap_results):
    """Pretty print bootstrap results."""
    print("\n" + "=" * 80)
    print("BOOTSTRAP CALIBRATION UNCERTAINTY")
    print(f"({bootstrap_results['n_bootstrap']} bootstrap samples)")
    print("=" * 80)

    print(f"\nCalibration size: n={bootstrap_results['n_calibration']}")
    print(f"Test size: {bootstrap_results['test_size']}")

    # Marginal
    print("\n" + "-" * 80)
    print("MARGINAL (All Samples)")
    print("-" * 80)
    for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
        m = bootstrap_results["marginal"][metric]
        q = m["quantiles"]

        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Mean:      {m['mean']:.4f} ± {m['std']:.4f}")
        print(f"  Median:    {q['q50']:.4f}")
        print(f"  [5%, 95%]: [{q['q05']:.4f}, {q['q95']:.4f}]")
        print(f"  [25%, 75%]: [{q['q25']:.4f}, {q['q75']:.4f}]")

    # Per-class
    for class_label in [0, 1]:
        print("\n" + "-" * 80)
        print(f"CLASS {class_label}")
        print("-" * 80)
        for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
            m = bootstrap_results[f"class_{class_label}"][metric]
            q = m["quantiles"]

            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Mean:      {m['mean']:.4f} ± {m['std']:.4f}")
            print(f"  Median:    {q['q50']:.4f}")
            print(f"  [5%, 95%]: [{q['q05']:.4f}, {q['q95']:.4f}]")

    print("\n" + "=" * 80)


def plot_bootstrap_distributions(bootstrap_results, figsize=(16, 12)):
    """Plot bootstrap distributions."""
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle(
        f"Bootstrap Calibration Uncertainty ({bootstrap_results['n_bootstrap']} trials)\n"
        f"Calibration n={bootstrap_results['n_calibration']}, Test size={bootstrap_results['test_size']}",
        fontsize=14,
        fontweight="bold",
    )

    metrics = ["singleton", "doublet", "abstention", "singleton_error"]
    metric_names = ["Singleton Rate", "Doublet Rate", "Abstention Rate", "Singleton Error Rate"]
    colors = ["steelblue", "coral", "mediumpurple"]
    row_names = ["MARGINAL", "CLASS 0", "CLASS 1"]
    data_keys = ["marginal", "class_0", "class_1"]

    for row, (row_name, data_key, color) in enumerate(zip(row_names, data_keys, colors, strict=False)):
        for col, (metric, name) in enumerate(zip(metrics, metric_names, strict=False)):
            ax = axes[row, col]
            m = bootstrap_results[data_key][metric]

            # Filter NaNs
            samples = m["samples"]
            samples = samples[~np.isnan(samples)]

            if len(samples) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            # Histogram
            ax.hist(samples, bins=50, alpha=0.7, color=color, edgecolor="black")

            # Quantiles
            q = m["quantiles"]
            ax.axvline(q["q50"], color="green", linestyle="-", linewidth=2, label=f"Median: {q['q50']:.3f}")
            ax.axvline(q["q05"], color="red", linestyle="--", linewidth=2, label=f"5%: {q['q05']:.3f}")
            ax.axvline(q["q95"], color="red", linestyle="--", linewidth=2, label=f"95%: {q['q95']:.3f}")
            ax.axvline(m["mean"], color="orange", linestyle=":", linewidth=2, label=f"Mean: {m['mean']:.3f}")

            ax.set_title(f"{row_name}: {name}", fontweight="bold")
            ax.set_xlabel("Rate")
            ax.set_ylabel("Count")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create simulator
    sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

    # Generate ONE calibration dataset
    labels, probs = sim.generate(n_samples=100)

    print("=" * 80)
    print("BOOTSTRAP CALIBRATION UNCERTAINTY ANALYSIS")
    print("=" * 80)
    print("\nQuestion: If I recalibrate on similar datasets, how do rates vary?")
    print("Method: Bootstrap resample calibration → calibrate → test on fresh data")

    # Run bootstrap analysis
    bootstrap_results = bootstrap_calibration_uncertainty(
        labels=labels,
        probs=probs,
        simulator=sim,
        alpha_target=0.10,
        delta=0.10,
        test_size=1000,
        n_bootstrap=1000,
        n_jobs=-1,
        seed=123,
    )

    # Print results
    print_bootstrap_results(bootstrap_results)

    # Plot
    print("\nCreating visualizations...")
    fig = plot_bootstrap_distributions(bootstrap_results)
    plt.savefig("bootstrap_calibration_uncertainty.png", dpi=300, bbox_inches="tight")
    print("✅ Saved to: bootstrap_calibration_uncertainty.png")
    plt.show()

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\n✓ Shows variability from RECALIBRATION (not just test set sampling)")
    print("✓ [5%, 95%] quantiles show range of rates across calibrations")
    print("✓ Compare to LOO-CV bounds to see calibration vs sampling uncertainty")
    print("=" * 80)
