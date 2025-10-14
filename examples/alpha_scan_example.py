"""Example: Alpha Scan Analysis.

This example demonstrates how to use the alpha_scan function to analyze
how prediction set statistics (abstentions, singletons, doublets) vary
across different alpha thresholds.
"""

import matplotlib.pyplot as plt
import numpy as np

from ssbc import BinaryClassifierSimulator, alpha_scan


def main():
    """Run alpha scan example."""

    print("=" * 80)
    print("ALPHA SCAN ANALYSIS")
    print("=" * 80)

    # ========== Step 1: Generate Simulated Data ==========
    print("\n1. Generating simulated binary classification data...")

    # Create simulator with balanced classes
    sim = BinaryClassifierSimulator(
        p_class1=0.50,
        beta_params_class0=(2, 7),
        beta_params_class1=(7, 2),
        seed=42,
    )

    labels, probs = sim.generate(n_samples=200)

    print(f"   Generated {len(labels)} samples")
    print(f"   Class balance: Class 0: {np.sum(labels == 0)}, Class 1: {np.sum(labels == 1)}")

    # ========== Step 2: Run Alpha Scan ==========
    print("\n2. Running alpha scan across all possible thresholds...")

    # Scan with fixed threshold of 0.5
    df, fixed_result = alpha_scan(labels, probs, fixed_threshold=0.5)

    print(f"   Total scan points: {len(df)}")
    print(f"   Alpha range: [{df['alpha'].min():.4f}, {df['alpha'].max():.4f}]")
    print("   Fixed threshold evaluated: qhat=0.5")

    # ========== Step 3: Display Sample Results ==========
    print("\n3. Sample scan results:")
    print("\n   First 5 rows:")
    print(df.head().to_string(index=False))

    print("\n   Last 5 rows:")
    print(df.tail().to_string(index=False))

    print("\n   Fixed threshold result:")
    print(f"   Alpha: {fixed_result['alpha']:.6f}")
    print(f"   Singletons: {fixed_result['n_singletons']}, Coverage: {fixed_result['singleton_coverage']:.4f}")

    # ========== Step 4: Visualize Results ==========
    print("\n4. Creating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Prediction set counts vs alpha
    ax = axes[0, 0]
    ax.plot(df["alpha"], df["n_abstentions"], label="Abstentions", marker="o", markersize=2)
    ax.plot(df["alpha"], df["n_singletons"], label="Singletons", marker="s", markersize=2)
    ax.plot(df["alpha"], df["n_doublets"], label="Doublets", marker="^", markersize=2)

    # Highlight fixed threshold
    ax.axvline(fixed_result["alpha"], color="red", linestyle="--", alpha=0.5, label="Fixed threshold")

    ax.set_xlabel("Alpha (miscoverage rate)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Set Counts vs Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Thresholds vs alpha
    ax = axes[0, 1]
    ax.plot(df["alpha"], df["qhat_0"], label="qhat_0 (Class 0 threshold)", marker="o", markersize=2)
    ax.plot(df["alpha"], df["qhat_1"], label="qhat_1 (Class 1 threshold)", marker="s", markersize=2)
    ax.axvline(fixed_result["alpha"], color="red", linestyle="--", alpha=0.5, label="Fixed threshold")
    ax.set_xlabel("Alpha (miscoverage rate)")
    ax.set_ylabel("Threshold (qhat)")
    ax.set_title("Conformal Thresholds vs Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Proportions
    ax = axes[1, 0]
    n_total = len(labels)
    ax.plot(df["alpha"], df["n_abstentions"] / n_total, label="Abstention rate", marker="o", markersize=2)
    ax.plot(df["alpha"], df["n_singletons"] / n_total, label="Singleton rate", marker="s", markersize=2)
    ax.plot(df["alpha"], df["n_doublets"] / n_total, label="Doublet rate", marker="^", markersize=2)
    ax.axvline(fixed_result["alpha"], color="red", linestyle="--", alpha=0.5, label="Fixed threshold")
    ax.set_xlabel("Alpha (miscoverage rate)")
    ax.set_ylabel("Proportion")
    ax.set_title("Prediction Set Proportions vs Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Singleton rate focus with common alpha values marked
    ax = axes[1, 1]
    ax.plot(df["alpha"], df["n_singletons"] / n_total, label="Singleton rate", marker="o", markersize=3, color="green")
    ax.axvline(
        fixed_result["alpha"], color="red", linestyle="--", alpha=0.5, label=f"Fixed (α≈{fixed_result['alpha']:.3f})"
    )

    # Mark common alpha values
    common_alphas = [0.05, 0.10, 0.20]
    for alpha_val in common_alphas:
        if (df["alpha"] - alpha_val).abs().min() < 0.01:
            closest_idx = (df["alpha"] - alpha_val).abs().idxmin()
            ax.axvline(df.loc[closest_idx, "alpha"], color="blue", linestyle=":", alpha=0.4)
            ax.text(
                df.loc[closest_idx, "alpha"],
                0.02,
                f"α={alpha_val}",
                rotation=90,
                verticalalignment="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Alpha (miscoverage rate)")
    ax.set_ylabel("Singleton Rate")
    ax.set_title("Singleton Rate vs Alpha (with common values)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Singleton coverage (marginal)
    ax = axes[0, 2]
    ax.plot(df["alpha"], df["singleton_coverage"], label="Marginal coverage", marker="o", markersize=2, color="purple")
    ax.axvline(fixed_result["alpha"], color="red", linestyle="--", alpha=0.5, label="Fixed threshold")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect coverage")
    ax.set_xlabel("Alpha (miscoverage rate)")
    ax.set_ylabel("Singleton Coverage")
    ax.set_title("Singleton Coverage (Marginal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 6: Per-class singleton coverage (Mondrian)
    ax = axes[1, 2]
    ax.plot(df["alpha"], df["singleton_coverage_0"], label="Class 0 coverage", marker="o", markersize=2, color="blue")
    ax.plot(df["alpha"], df["singleton_coverage_1"], label="Class 1 coverage", marker="s", markersize=2, color="orange")
    ax.axvline(fixed_result["alpha"], color="red", linestyle="--", alpha=0.5, label="Fixed threshold")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect coverage")
    ax.set_xlabel("Alpha (miscoverage rate)")
    ax.set_ylabel("Singleton Coverage (Per-Class)")
    ax.set_title("Mondrian Singleton Coverage (Per-Class)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    output_path = "examples/alpha_scan_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"   Saved plot to: {output_path}")

    # ========== Step 5: Key Statistics ==========
    print("\n5. Key Statistics:")

    # Find alpha with maximum singletons
    max_singleton_idx = df["n_singletons"].idxmax()
    max_singleton_row = df.loc[max_singleton_idx]

    max_sing_count = max_singleton_row["n_singletons"]
    print(f"\n   Maximum singleton count: {max_sing_count} ({max_sing_count/n_total:.1%})")
    print(f"   At alpha: {max_singleton_row['alpha']:.4f}")
    print(f"   Thresholds: qhat_0={max_singleton_row['qhat_0']:.4f}, qhat_1={max_singleton_row['qhat_1']:.4f}")
    print(f"   Coverage: {max_singleton_row['singleton_coverage']:.4f}")

    # Find alpha with best singleton coverage
    max_coverage_idx = df["singleton_coverage"].idxmax()
    max_coverage_row = df.loc[max_coverage_idx]

    print(f"\n   Best singleton coverage: {max_coverage_row['singleton_coverage']:.4f}")
    print(f"   At alpha: {max_coverage_row['alpha']:.4f}")
    n_sing = max_coverage_row["n_singletons"]
    print(f"   Singletons: {n_sing} ({n_sing/n_total:.1%})")
    cov_0 = max_coverage_row["singleton_coverage_0"]
    cov_1 = max_coverage_row["singleton_coverage_1"]
    print(f"   Coverage by class: Class 0={cov_0:.4f}, Class 1={cov_1:.4f}")

    # Find alpha with minimum abstentions
    min_abstention_idx = df["n_abstentions"].idxmin()
    min_abstention_row = df.loc[min_abstention_idx]

    min_abst_count = min_abstention_row["n_abstentions"]
    print(f"\n   Minimum abstention count: {min_abst_count} ({min_abst_count/n_total:.1%})")
    print(f"   At alpha: {min_abstention_row['alpha']:.4f}")

    # Statistics for fixed threshold
    print("\n   Fixed threshold (qhat=0.5) statistics:")
    print(f"   - Effective alpha: {fixed_result['alpha']:.4f}")
    print(f"   - Abstentions: {fixed_result['n_abstentions']} ({fixed_result['n_abstentions']/n_total:.1%})")
    print(f"   - Singletons: {fixed_result['n_singletons']} ({fixed_result['n_singletons']/n_total:.1%})")
    print(f"   - Doublets: {fixed_result['n_doublets']} ({fixed_result['n_doublets']/n_total:.1%})")
    print(f"   - Singleton coverage: {fixed_result['singleton_coverage']:.4f}")
    fix_cov_0 = fixed_result["singleton_coverage_0"]
    fix_cov_1 = fixed_result["singleton_coverage_1"]
    print(f"   - Coverage by class: Class 0={fix_cov_0:.4f}, Class 1={fix_cov_1:.4f}")

    # ========== Step 6: Save Results ==========
    print("\n6. Saving results to CSV...")
    output_csv = "examples/alpha_scan_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"   Saved to: {output_csv}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
